"""
Experimento SEMANTICS-001: Word Sense Disambiguation (WSD)
===========================================================

Objetivo: Desambiguar palabras polisémicas usando contexto HDC.

Enfoque:
- Identificar palabras con múltiples lemas posibles (misma forma superficial)
- Usar contexto (palabras vecinas, POS) para predecir el lemma correcto
- Evaluar accuracy de desambiguación

Ejemplos de ambigüedad en español:
- "banco" → banco (NOUN: institución) vs banco (NOUN: asiento) vs bancar (VERB)
- "amo" → amar (VERB: yo amo) vs amo (NOUN: dueño)
- "río" → río (NOUN: corriente) vs reír (VERB: yo río)

Métrica: Accuracy de desambiguación de lemma para palabras ambiguas.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'form': token['form'].lower(),
                    'lemma': token['lemma'],
                    'upos': token['upos'],
                    'feats': token['feats'] or {},
                    'head': token['head'],
                    'deprel': token['deprel'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


def find_ambiguous_words(sentences: List[List[dict]], min_freq: int = 10) -> Dict[str, List[str]]:
    """
    Encuentra palabras ambiguas: misma forma superficial, diferentes lemas.
    
    Retorna dict: form → [lista de lemas posibles]
    """
    form_to_lemmas = defaultdict(set)
    form_freq = Counter()
    
    for sent in sentences:
        for token in sent:
            form = token['form']
            lemma = token['lemma']
            form_to_lemmas[form].add(lemma)
            form_freq[form] += 1
    
    # Filtrar palabras ambiguas con frecuencia mínima
    ambiguous = {}
    for form, lemmas in form_to_lemmas.items():
        if len(lemmas) > 1 and form_freq[form] >= min_freq:
            ambiguous[form] = sorted(list(lemmas))
    
    return ambiguous


def extract_context_features(sentence: List[dict], idx: int, window: int = 2) -> dict:
    """
    Extrae features del contexto para desambiguación.
    
    Features:
    - POS de tokens en ventana izquierda/derecha
    - Forms de tokens en ventana
    - Sufijos de la palabra objetivo
    - Posición en oración
    """
    token = sentence[idx]
    form = token['form']
    
    # Sufijos
    suf2 = form[-2:] if len(form) >= 2 else form
    suf3 = form[-3:] if len(form) >= 3 else form
    
    # Contexto izquierdo
    left_pos = []
    left_forms = []
    for i in range(max(0, idx - window), idx):
        left_pos.append(sentence[i]['upos'])
        left_forms.append(sentence[i]['form'])
    
    # Contexto derecho
    right_pos = []
    right_forms = []
    for i in range(idx + 1, min(len(sentence), idx + window + 1)):
        right_pos.append(sentence[i]['upos'])
        right_forms.append(sentence[i]['form'])
    
    # POS vecinos inmediatos
    left_pos_1 = sentence[idx-1]['upos'] if idx > 0 else '<BOS>'
    right_pos_1 = sentence[idx+1]['upos'] if idx < len(sentence) - 1 else '<EOS>'
    
    # Posición relativa
    rel_pos = idx / max(1, len(sentence) - 1)
    pos_bucket = 'start' if rel_pos < 0.2 else 'end' if rel_pos > 0.8 else 'mid'
    
    return {
        'form': form,
        'suf2': suf2,
        'suf3': suf3,
        'left_pos': tuple(left_pos),
        'right_pos': tuple(right_pos),
        'left_pos_1': left_pos_1,
        'right_pos_1': right_pos_1,
        'pos_bucket': pos_bucket,
    }


class WSDModel:
    """
    Modelo de desambiguación basado en item memory.
    
    Estrategia: Para cada palabra ambigua, almacenar ejemplos de cada lemma
    con sus features contextuales. Predecir el lemma más similar.
    """
    
    def __init__(self):
        self.lemma_lookup = {}  # form → {context_key → lemma}
        self.fallback_lookup = {}  # form → lemma más frecuente
    
    def _make_context_key(self, features: dict) -> str:
        """Crea clave contextual para lookup."""
        return (f"lp1:{features['left_pos_1']}|rp1:{features['right_pos_1']}|"
                f"suf3:{features['suf3']}|pos:{features['pos_bucket']}")
    
    def _make_fallback_key(self, features: dict) -> str:
        """Clave simplificada."""
        return f"lp1:{features['left_pos_1']}|suf3:{features['suf3']}"
    
    def train(self, sentences: List[List[dict]], ambiguous_words: Dict[str, List[str]]):
        """Entrena con ejemplos de palabras ambiguas."""
        # Contadores por form
        form_lemma_contexts = defaultdict(lambda: defaultdict(Counter))
        form_lemma_freq = defaultdict(Counter)
        
        for sent in sentences:
            for i, token in enumerate(sent):
                form = token['form']
                if form not in ambiguous_words:
                    continue
                
                lemma = token['lemma']
                features = extract_context_features(sent, i)
                
                ctx_key = self._make_context_key(features)
                form_lemma_contexts[form][ctx_key][lemma] += 1
                form_lemma_freq[form][lemma] += 1
        
        # Crear lookups
        for form in ambiguous_words:
            # Lookup contextual
            self.lemma_lookup[form] = {}
            for ctx_key, lemma_counts in form_lemma_contexts[form].items():
                self.lemma_lookup[form][ctx_key] = lemma_counts.most_common(1)[0][0]
            
            # Fallback: lemma más frecuente
            if form_lemma_freq[form]:
                self.fallback_lookup[form] = form_lemma_freq[form].most_common(1)[0][0]
        
        n_forms = len(self.lemma_lookup)
        n_contexts = sum(len(v) for v in self.lemma_lookup.values())
        print(f"  Entrenado: {n_forms} formas ambiguas, {n_contexts:,} claves contextuales")
    
    def predict(self, form: str, features: dict) -> str:
        """Predice el lemma para una forma ambigua."""
        if form not in self.lemma_lookup:
            return None
        
        # Intento 1: clave contextual
        ctx_key = self._make_context_key(features)
        if ctx_key in self.lemma_lookup[form]:
            return self.lemma_lookup[form][ctx_key]
        
        # Intento 2: fallback
        if form in self.fallback_lookup:
            return self.fallback_lookup[form]
        
        return None


def evaluate_wsd(model: WSDModel, sentences: List[List[dict]], 
                 ambiguous_words: Dict[str, List[str]]) -> dict:
    """Evalúa el modelo de WSD."""
    correct = 0
    total = 0
    per_form_results = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for sent in sentences:
        for i, token in enumerate(sent):
            form = token['form']
            if form not in ambiguous_words:
                continue
            
            true_lemma = token['lemma']
            features = extract_context_features(sent, i)
            pred_lemma = model.predict(form, features)
            
            if pred_lemma is None:
                continue
            
            total += 1
            per_form_results[form]['total'] += 1
            
            if pred_lemma == true_lemma:
                correct += 1
                per_form_results[form]['correct'] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Top 10 formas con más ejemplos
    top_forms = sorted(per_form_results.items(), 
                       key=lambda x: -x[1]['total'])[:10]
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'per_form': {form: {
            'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
            'total': data['total'],
            'correct': data['correct'],
        } for form, data in top_forms},
    }


def baseline_most_frequent(sentences: List[List[dict]], 
                           ambiguous_words: Dict[str, List[str]]) -> dict:
    """Baseline: siempre predice el lemma más frecuente."""
    form_lemma_freq = defaultdict(Counter)
    
    for sent in sentences:
        for token in sent:
            if token['form'] in ambiguous_words:
                form_lemma_freq[token['form']][token['lemma']] += 1
    
    form_most_frequent = {
        form: counts.most_common(1)[0][0]
        for form, counts in form_lemma_freq.items()
    }
    
    correct = 0
    total = 0
    
    for sent in sentences:
        for token in sent:
            form = token['form']
            if form not in ambiguous_words:
                continue
            
            total += 1
            if form_most_frequent.get(form) == token['lemma']:
                correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SEMANTICS-001: Word Sense Disambiguation (WSD)")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Encontrar palabras ambiguas
    print(f"\n{'─'*70}")
    print("Identificando palabras ambiguas")
    print(f"{'─'*70}")
    
    ambiguous = find_ambiguous_words(train_s, min_freq=10)
    print(f"  Palabras ambiguas encontradas: {len(ambiguous)}")
    
    # Mostrar top 15
    print(f"\n  Top 15 palabras ambiguas:")
    for form in sorted(ambiguous.keys(), key=lambda x: -len(ambiguous[x]))[:15]:
        lemmas = ambiguous[form]
        print(f"    {form}: {lemmas}")
    
    # Entrenar modelo
    print(f"\n{'─'*70}")
    print("Entrenando modelo WSD")
    print(f"{'─'*70}")
    
    model = WSDModel()
    t0 = time.time()
    model.train(train_s, ambiguous)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Baseline
    print(f"\n{'─'*70}")
    print("Baseline: lemma más frecuente")
    print(f"{'─'*70}")
    
    baseline = baseline_most_frequent(train_s, ambiguous)
    print(f"  Accuracy: {baseline['accuracy']:.2%}")
    print(f"  Total: {baseline['total']:,}")
    
    # Evaluar modelo
    print(f"\n{'─'*70}")
    print("Evaluación del modelo WSD")
    print(f"{'─'*70}")
    
    t0 = time.time()
    result = evaluate_wsd(model, test_s, ambiguous)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  Correctos: {result['correct']:,}/{result['total']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    print(f"\n  Top 10 formas ambiguas (resultados):")
    for form, data in result['per_form'].items():
        print(f"    {form}: {data['accuracy']:.2%} ({data['correct']}/{data['total']})")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Palabras ambiguas: {len(ambiguous)}")
    print(f"  Baseline (most freq): {baseline['accuracy']:.2%}")
    print(f"  SEMANTICS-001: {result['accuracy']:.2%}")
    print(f"  Mejora: {result['accuracy'] - baseline['accuracy']:+.2%}")
    
    # Guardar resultados
    data = {
        "experiment_id": "SEMANTICS-001",
        "description": "Word Sense Disambiguation",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "n_ambiguous_words": len(ambiguous),
        "results": {
            "baseline": baseline,
            "semantics_001": result,
        },
    }
    
    output_path = os.path.join(output_dir, "semantics_001_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
