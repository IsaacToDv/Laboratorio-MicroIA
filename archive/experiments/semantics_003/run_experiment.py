"""
Experimento SEMANTICS-003: Named Entity Recognition (NER)
==========================================================

Objetivo: Reconocer entidades nombradas en texto.

Enfoque:
- Clasificar cada token como NE (Named Entity) o no-NE
- Para tokens NE, clasificar el tipo (PER, ORG, LOC, OTHER)
- Usar PROPN del corpus como proxy de NE

Features:
- Capitalización
- Sufijos
- POS del token y vecinos
- Contexto (palabras vecinas)

Métrica: F1 score (precision + recall) para detección de NE.
"""

import sys
import os
import time
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from conllu import parse_incr


def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'form': token['form'],
                    'lemma': token['lemma'],
                    'upos': token['upos'],
                    'feats': token['feats'] or {},
                    'head': token['head'],
                    'deprel': token['deprel'],
                    'misc': token.get('misc', {}) or {},
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


def is_named_entity(token: dict) -> bool:
    """Determina si un token es una entidad nombrada."""
    return token['upos'] == 'PROPN'


def classify_ne_type(token: dict, sentence: List[dict], idx: int) -> str:
    """
    Clasifica el tipo de entidad nombrada usando heurísticas.
    
    Tipos:
    - PER: persona
    - ORG: organización
    - LOC: lugar
    - OTHER: otros
    """
    # Heurísticas basadas en contexto
    form = token['form']
    deprel = token['deprel']
    
    # Buscar contexto
    left_word = sentence[idx-1]['form'].lower() if idx > 0 else ''
    right_word = sentence[idx+1]['form'].lower() if idx < len(sentence)-1 else ''
    
    # Indicadores de persona
    person_indicators = ['sr', 'sra', 'srta', 'dr', 'dra', 'don', 'doña', 
                         'presidente', 'ministro', 'director']
    if any(ind in left_word for ind in person_indicators):
        return 'PER'
    
    # Indicadores de organización
    org_indicators = ['banco', 'empresa', 'compañía', 'sociedad', 'grupo',
                      's.a', 's.l', 'fundación', 'asociación']
    if any(ind in left_word or ind in right_word for ind in org_indicators):
        return 'ORG'
    
    # Indicadores de lugar
    loc_indicators = ['ciudad', 'país', 'estado', 'provincia', 'río', 'monte',
                      'calle', 'avenida', 'plaza']
    if any(ind in left_word or ind in right_word for ind in loc_indicators):
        return 'LOC'
    
    # Basado en MISC field del corpus (si tiene Entity annotation)
    misc_str = str(token.get('misc', {}))
    if 'location' in misc_str.lower() or 'place' in misc_str.lower():
        return 'LOC'
    if 'organization' in misc_str.lower():
        return 'ORG'
    if 'person' in misc_str.lower():
        return 'PER'
    
    return 'OTHER'


def extract_ner_features(sentence: List[dict], idx: int) -> dict:
    """Extrae features para NER."""
    token = sentence[idx]
    form = token['form']
    
    # Capitalización
    is_capitalized = form[0].isupper() if form else False
    is_all_caps = form.isupper() and len(form) > 1
    
    # Sufijos
    form_lower = form.lower()
    suf2 = form_lower[-2:] if len(form_lower) >= 2 else form_lower
    suf3 = form_lower[-3:] if len(form_lower) >= 3 else form_lower
    
    # POS
    pos = token['upos']
    left_pos = sentence[idx-1]['upos'] if idx > 0 else '<BOS>'
    right_pos = sentence[idx+1]['upos'] if idx < len(sentence)-1 else '<EOS>'
    
    # Forma de vecinos
    left_form = sentence[idx-1]['form'].lower() if idx > 0 else '<BOS>'
    right_form = sentence[idx+1]['form'].lower() if idx < len(sentence)-1 else '<EOS>'
    
    # Longitud
    length = len(form)
    len_bucket = 'short' if length <= 3 else 'med' if length <= 7 else 'long'
    
    # Posición en oración
    rel_pos = idx / max(1, len(sentence) - 1)
    pos_bucket = 'start' if rel_pos < 0.2 else 'end' if rel_pos > 0.8 else 'mid'
    
    # Tiene dígitos
    has_digit = any(c.isdigit() for c in form)
    
    return {
        'is_capitalized': is_capitalized,
        'is_all_caps': is_all_caps,
        'pos': pos,
        'left_pos': left_pos,
        'right_pos': right_pos,
        'left_form': left_form,
        'right_form': right_form,
        'suf2': suf2,
        'suf3': suf3,
        'len_bucket': len_bucket,
        'pos_bucket': pos_bucket,
        'has_digit': has_digit,
    }


class NERModel:
    """Modelo NER basado en item memory."""
    
    def __init__(self):
        self.ne_lookup = {}      # context_key → is_ne
        self.type_lookup = {}    # context_key → ne_type
        self.fallback_ne = False
    
    def _make_ne_key(self, features: dict) -> str:
        return (f"pos:{features['pos']}|cap:{features['is_capitalized']}|"
                f"lp:{features['left_pos']}|rp:{features['right_pos']}|"
                f"suf3:{features['suf3']}|len:{features['len_bucket']}")
    
    def _make_type_key(self, features: dict) -> str:
        return (f"lf:{features['left_form']}|rf:{features['right_form']}|"
                f"suf3:{features['suf3']}|pos:{features['pos']}")
    
    def train(self, sentences: List[List[dict]]):
        """Entrena el modelo NER."""
        ne_counters = defaultdict(Counter)
        type_counters = defaultdict(Counter)
        ne_global = Counter()
        
        for sent in sentences:
            for i, token in enumerate(sent):
                features = extract_ner_features(sent, i)
                is_ne = is_named_entity(token)
                
                ne_key = self._make_ne_key(features)
                ne_counters[ne_key][is_ne] += 1
                ne_global[is_ne] += 1
                
                if is_ne:
                    ne_type = classify_ne_type(token, sent, i)
                    type_key = self._make_type_key(features)
                    type_counters[type_key][ne_type] += 1
        
        # Crear lookups
        self.ne_lookup = {
            k: c.most_common(1)[0][0] for k, c in ne_counters.items()
        }
        self.type_lookup = {
            k: c.most_common(1)[0][0] for k, c in type_counters.items()
        }
        self.fallback_ne = ne_global.most_common(1)[0][0] if ne_global else False
        
        n_ne = sum(1 for v in ne_global if v == True) * ne_global.get(True, 0)
        print(f"  Entrenado: {len(self.ne_lookup):,} claves NE")
        print(f"  {len(self.type_lookup):,} claves de tipo")
    
    def predict_ne(self, features: dict) -> bool:
        key = self._make_ne_key(features)
        return self.ne_lookup.get(key, self.fallback_ne)
    
    def predict_type(self, features: dict) -> str:
        key = self._make_type_key(features)
        return self.type_lookup.get(key, 'OTHER')


def evaluate_ner(model: NERModel, sentences: List[List[dict]]) -> dict:
    """Evalúa NER con métricas de precision, recall, F1."""
    tp = fp = fn = tn = 0
    
    # Por tipo
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)
    
    for sent in sentences:
        for i, token in enumerate(sent):
            features = extract_ner_features(sent, i)
            true_ne = is_named_entity(token)
            pred_ne = model.predict_ne(features)
            
            if true_ne and pred_ne:
                tp += 1
                # Tipo
                true_type = classify_ne_type(token, sent, i)
                pred_type = model.predict_type(features)
                if true_type == pred_type:
                    type_tp[true_type] += 1
                else:
                    type_fn[true_type] += 1
                    type_fp[pred_type] += 1
            elif true_ne and not pred_ne:
                fn += 1
                true_type = classify_ne_type(token, sent, i)
                type_fn[true_type] += 1
            elif not true_ne and pred_ne:
                fp += 1
            else:
                tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Por tipo
    per_type = {}
    for ne_type in set(list(type_tp.keys()) + list(type_fp.keys()) + list(type_fn.keys())):
        t_tp = type_tp[ne_type]
        t_fp = type_fp[ne_type]
        t_fn = type_fn[ne_type]
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        per_type[ne_type] = {
            'precision': t_prec, 'recall': t_rec, 'f1': t_f1,
            'tp': t_tp, 'fp': t_fp, 'fn': t_fn,
        }
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'per_type': per_type,
    }


def baseline_all_propn(sentences: List[List[dict]]) -> dict:
    """Baseline: predice NE si POS=PROPN."""
    tp = fp = fn = tn = 0
    
    for sent in sentences:
        for token in sent:
            true_ne = is_named_entity(token)
            pred_ne = (token['upos'] == 'PROPN')
            
            if true_ne and pred_ne: tp += 1
            elif true_ne and not pred_ne: fn += 1
            elif not true_ne and pred_ne: fp += 1
            else: tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SEMANTICS-003: Named Entity Recognition (NER)")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Estadísticas de NE
    print(f"\n{'─'*70}")
    print("Estadísticas de entidades nombradas (train)")
    print(f"{'─'*70}")
    
    ne_count = sum(1 for sent in train_s for tok in sent if is_named_entity(tok))
    total_count = sum(len(sent) for sent in train_s)
    print(f"  Total tokens: {total_count:,}")
    print(f"  Entidades (PROPN): {ne_count:,} ({ne_count/total_count:.2%})")
    
    # Distribución de tipos
    type_counter = Counter()
    for sent in train_s:
        for i, tok in enumerate(sent):
            if is_named_entity(tok):
                type_counter[classify_ne_type(tok, sent, i)] += 1
    
    print(f"\n  Distribución de tipos:")
    for ne_type, count in type_counter.most_common():
        print(f"    {ne_type:<10} {count:>6,} ({count/ne_count:.2%})")
    
    # Entrenar modelo
    print(f"\n{'─'*70}")
    print("Entrenando modelo NER")
    print(f"{'─'*70}")
    
    model = NERModel()
    t0 = time.time()
    model.train(train_s)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Baseline
    print(f"\n{'─'*70}")
    print("Baseline: POS=PROPN")
    print(f"{'─'*70}")
    
    bl = baseline_all_propn(test_s)
    print(f"  Precision: {bl['precision']:.2%}")
    print(f"  Recall:    {bl['recall']:.2%}")
    print(f"  F1:        {bl['f1']:.2%}")
    
    # Evaluar modelo
    print(f"\n{'─'*70}")
    print("Evaluación del modelo NER")
    print(f"{'─'*70}")
    
    t0 = time.time()
    result = evaluate_ner(model, test_s)
    eval_time = time.time() - t0
    
    print(f"  Precision: {result['precision']:.2%}")
    print(f"  Recall:    {result['recall']:.2%}")
    print(f"  F1:        {result['f1']:.2%}")
    print(f"  TP={result['tp']:,} FP={result['fp']:,} FN={result['fn']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    print(f"\n  Resultados por tipo:")
    for ne_type, data in sorted(result['per_type'].items()):
        print(f"    {ne_type:<10} P={data['precision']:.2%} R={data['recall']:.2%} "
              f"F1={data['f1']:.2%}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Baseline (POS=PROPN): F1={bl['f1']:.2%}")
    print(f"  SEMANTICS-003: F1={result['f1']:.2%}")
    print(f"  Mejora: {result['f1'] - bl['f1']:+.2%}")
    
    # Guardar resultados
    data = {
        "experiment_id": "SEMANTICS-003",
        "description": "Named Entity Recognition",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "ne_stats": {"total": total_count, "ne_count": ne_count},
        "type_distribution": dict(type_counter),
        "results": {
            "baseline": bl,
            "semantics_003": result,
        },
    }
    
    output_path = os.path.join(output_dir, "semantics_003_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
