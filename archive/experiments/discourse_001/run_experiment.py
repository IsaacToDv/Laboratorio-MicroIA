"""
Experimento DISCOURSE-001: Correferencia
=========================================

Objetivo: Resolver pronombres a sus antecedentes.

Enfoque:
- Para cada pronombre (él, ella, ellos, ellas, lo, la, le, les, etc.)
- Buscar candidatos en oraciones anteriores
- Predecir el antecedente correcto usando features

Features:
- Género y número del pronombre vs candidato
- Distancia (en tokens y oraciones)
- POS del candidato (NOUN, PROPN, PRON)
- Si el candidato es sujeto
- Frecuencia del candidato en el discurso

Métrica: Accuracy de resolución de correferencia.
"""

import sys
import os
import time
import json
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from conllu import parse_incr


# Pronombres que requieren resolución
PRONOMBRES = {
    # Personales sujeto
    'él', 'ella', 'ellos', 'ellas', 'ello',
    'yo', 'tú', 'nosotros', 'nosotras', 'vosotros', 'vosotras',
    # Personales objeto
    'lo', 'la', 'los', 'las', 'le', 'les',
    'me', 'te', 'nos', 'os',
    # Posesivos
    'su', 'sus', 'suyo', 'suya', 'suyos', 'suyas',
    # Demonstrativos
    'este', 'esta', 'estos', 'estas', 'esto',
    'ese', 'esa', 'esos', 'esas', 'eso',
    'aquel', 'aquella', 'aquellos', 'aquellas', 'aquello',
}

# Pronombres de tercera persona (los que más necesitan resolución)
THIRD_PERSON_PRONOUNS = {
    'él', 'ella', 'ellos', 'ellas', 'ello',
    'lo', 'la', 'los', 'las', 'le', 'les',
    'se',
    'su', 'sus',
}


def load_conllu_documents(conllu_path: str) -> List[List[List[dict]]]:
    """
    Carga oraciones agrupadas por documento.
    
    Retorna lista de documentos, cada documento es una lista de oraciones.
    """
    documents = []
    current_doc = []
    
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
                current_doc.append(sentence_data)
                
                # Agrupar en documentos de ~5 oraciones
                if len(current_doc) >= 5:
                    documents.append(current_doc)
                    current_doc = []
    
    if current_doc:
        documents.append(current_doc)
    
    return documents


def get_gender_number(token: dict) -> Tuple[str, str]:
    """Extrae género y número de un token."""
    feats = token.get('feats', {})
    gender = feats.get('Gender', '<UNK>')
    number = feats.get('Number', '<UNK>')
    return gender, number


def is_pronoun_requiring_resolution(token: dict) -> bool:
    """Determina si un pronombre necesita resolución de correferencia."""
    if token['upos'] not in ['PRON', 'DET']:
        return False
    
    form = token['form'].lower()
    return form in THIRD_PERSON_PRONOUNS


def extract_candidates(document: List[List[dict]], sent_idx: int, 
                       token_idx: int, window: int = 3) -> List[dict]:
    """
    Extrae candidatos a antecedente en las oraciones anteriores.
    
    window: número de oraciones hacia atrás a considerar.
    """
    candidates = []
    
    start_sent = max(0, sent_idx - window)
    
    for s in range(start_sent, sent_idx + 1):
        sentence = document[s]
        
        for t, token in enumerate(sentence):
            # Saltar el pronombre mismo
            if s == sent_idx and t == token_idx:
                continue
            # Solo considerar tokens después del pronombre en la misma oración no
            if s == sent_idx and t > token_idx:
                continue
            
            # Candidatos: NOUN, PROPN, PRON (no tercera persona)
            if token['upos'] in ['NOUN', 'PROPN']:
                candidates.append({
                    'token': token,
                    'sent_idx': s,
                    'token_idx': t,
                    'sent_distance': sent_idx - s,
                    'token_distance': sum(len(document[i]) 
                                          for i in range(s, sent_idx)) + token_idx - t,
                })
            elif token['upos'] == 'PRON':
                form = token['form'].lower()
                if form not in THIRD_PERSON_PRONOUNS:
                    candidates.append({
                        'token': token,
                        'sent_idx': s,
                        'token_idx': t,
                        'sent_distance': sent_idx - s,
                        'token_distance': sum(len(document[i]) 
                                              for i in range(s, sent_idx)) + token_idx - t,
                    })
    
    return candidates


def extract_coref_features(pronoun: dict, candidate: dict, 
                            document: List[List[dict]]) -> dict:
    """Extrae features para el par (pronombre, candidato)."""
    p_gender, p_number = get_gender_number(pronoun)
    c_gender, c_number = get_gender_number(candidate['token'])
    
    # Acuerdo de género y número
    gender_match = (p_gender == c_gender) or p_gender == '<UNK>' or c_gender == '<UNK>'
    number_match = (p_number == c_number) or p_number == '<UNK>' or c_number == '<UNK>'
    
    # Distancia
    sent_dist = candidate['sent_distance']
    token_dist = candidate['token_distance']
    
    dist_bucket = 'same' if sent_dist == 0 else 'near' if sent_dist <= 1 else 'far'
    
    # POS del candidato
    c_pos = candidate['token']['upos']
    
    # Si el candidato es sujeto (deprel = nsubj)
    is_subject = candidate['token']['deprel'] in ['nsubj', 'nsubj:pass']
    
    # Si el candidato es PROPN (más probable como antecedente)
    is_proper = c_pos == 'PROPN'
    
    return {
        'gender_match': gender_match,
        'number_match': number_match,
        'full_agreement': gender_match and number_match,
        'sent_distance': sent_dist,
        'token_distance': token_dist,
        'dist_bucket': dist_bucket,
        'c_pos': c_pos,
        'is_subject': is_subject,
        'is_proper': is_proper,
        'p_form': pronoun['form'].lower(),
        'p_gender': p_gender,
        'p_number': p_number,
    }


class CorefModel:
    """Modelo de correferencia basado en scoring."""
    
    def __init__(self):
        self.feature_weights = {}
    
    def train(self, documents: List[List[List[dict]]]):
        """
        Entrena pesos de features basándose en estadísticas del corpus.
        
        Aproximación simple: contar qué features correlacionan con 
        antecedente correcto (definido por proximidad + acuerdo).
        """
        # Heurística: el antecedente más cercano con acuerdo es usualmente correcto
        # Entrenamos un scoring basado en features
        
        feature_scores = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for doc in documents:
            for s, sent in enumerate(doc):
                for t, token in enumerate(sent):
                    if not is_pronoun_requiring_resolution(token):
                        continue
                    
                    candidates = extract_candidates(doc, s, t)
                    if not candidates:
                        continue
                    
                    # "Gold": candidato más cercano con acuerdo completo
                    best_candidate = None
                    for cand in sorted(candidates, key=lambda x: x['token_distance']):
                        features = extract_coref_features(token, cand, doc)
                        if features['full_agreement']:
                            best_candidate = cand
                            break
                    
                    if best_candidate is None:
                        continue
                    
                    # Actualizar scores de features
                    for cand in candidates:
                        features = extract_coref_features(token, cand, doc)
                        is_correct = (cand['token'] == best_candidate['token'])
                        
                        for key, value in features.items():
                            if isinstance(value, bool):
                                feat_key = f"{key}={value}"
                                feature_scores[feat_key]['total'] += 1
                                if is_correct:
                                    feature_scores[feat_key]['correct'] += 1
        
        # Calcular pesos como log-odds
        for feat_key, stats in feature_scores.items():
            if stats['total'] > 10:
                p_correct = stats['correct'] / stats['total']
                # Peso: probabilidad de ser correcto
                self.feature_weights[feat_key] = p_correct
        
        print(f"  Entrenado: {len(self.feature_weights)} features con pesos")
    
    def score_candidate(self, features: dict) -> float:
        """Calcula score de un candidato."""
        score = 0.0
        
        for key, value in features.items():
            if isinstance(value, bool):
                feat_key = f"{key}={value}"
                if feat_key in self.feature_weights:
                    score += self.feature_weights[feat_key]
        
        # Bonus por cercanía
        score += max(0, 1.0 - features['sent_distance'] * 0.2)
        
        # Bonus por ser sujeto
        if features['is_subject']:
            score += 0.3
        
        # Bonus por ser nombre propio
        if features['is_proper']:
            score += 0.2
        
        # Penalización fuerte por desacuerdo
        if not features['full_agreement']:
            score -= 2.0
        
        return score
    
    def resolve(self, pronoun: dict, candidates: List[dict], 
                document: List[List[dict]]) -> Optional[dict]:
        """Resuelve un pronombre al mejor candidato."""
        if not candidates:
            return None
        
        scored = []
        for cand in candidates:
            features = extract_coref_features(pronoun, cand, document)
            score = self.score_candidate(features)
            scored.append((cand, score))
        
        scored.sort(key=lambda x: -x[1])
        return scored[0][0] if scored else None


def baseline_recent_agreement(pronoun: dict, candidates: List[dict],
                               document: List[List[dict]]) -> Optional[dict]:
    """Baseline: seleccionar el candidato más reciente con acuerdo."""
    p_gender, p_number = get_gender_number(pronoun)
    
    for cand in sorted(candidates, key=lambda x: x['token_distance']):
        c_gender, c_number = get_gender_number(cand['token'])
        
        g_match = (p_gender == c_gender) or p_gender == '<UNK>' or c_gender == '<UNK>'
        n_match = (p_number == c_number) or p_number == '<UNK>' or c_number == '<UNK>'
        
        if g_match and n_match:
            return cand
    
    return None


def evaluate_coref(documents: List[List[List[dict]]], model=None) -> dict:
    """
    Evalúa resolución de correferencia.
    
    Gold estándar heurístico: el antecedente más cercano con acuerdo
    que sea sujeto o PROPN.
    """
    total = 0
    correct_model = 0
    correct_baseline = 0
    has_candidates = 0
    
    for doc in documents:
        for s, sent in enumerate(doc):
            for t, token in enumerate(sent):
                if not is_pronoun_requiring_resolution(token):
                    continue
                
                total += 1
                candidates = extract_candidates(doc, s, t)
                
                if not candidates:
                    continue
                
                has_candidates += 1
                
                # Gold: más cercano con acuerdo Y (sujeto O PROPN)
                gold = None
                for cand in sorted(candidates, key=lambda x: x['token_distance']):
                    features = extract_coref_features(token, cand, doc)
                    if features['full_agreement'] and (features['is_subject'] or features['is_proper']):
                        gold = cand
                        break
                
                if gold is None:
                    # Fallback: más cercano con acuerdo
                    gold = baseline_recent_agreement(token, candidates, doc)
                
                if gold is None:
                    continue
                
                # Modelo
                if model:
                    pred = model.resolve(token, candidates, doc)
                else:
                    pred = baseline_recent_agreement(token, candidates, doc)
                
                if pred and pred['token']['form'] == gold['token']['form'] and \
                   pred['token_idx'] == gold['token_idx']:
                    if model:
                        correct_model += 1
                    else:
                        correct_baseline += 1
    
    return {
        'total_pronouns': total,
        'with_candidates': has_candidates,
        'correct': correct_model if model else correct_baseline,
        'accuracy': (correct_model if model else correct_baseline) / has_candidates if has_candidates > 0 else 0,
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("DISCOURSE-001: Correferencia")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando documentos...")
    train_docs = load_conllu_documents(corpus_path / "es_ancora-ud-train.conllu")
    test_docs = load_conllu_documents(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_docs):,} documentos")
    print(f"  Test:  {len(test_docs):,} documentos")
    
    # Estadísticas de pronombres
    print(f"\n{'─'*70}")
    print("Estadísticas de pronombres")
    print(f"{'─'*70}")
    
    pronoun_counter = Counter()
    for doc in train_docs:
        for sent in doc:
            for token in sent:
                if is_pronoun_requiring_resolution(token):
                    pronoun_counter[token['form'].lower()] += 1
    
    print(f"  Pronombres que requieren resolución:")
    for form, count in pronoun_counter.most_common(10):
        print(f"    {form:<10} {count:>6,}")
    
    # Entrenar modelo
    print(f"\n{'─'*70}")
    print("Entrenando modelo de correferencia")
    print(f"{'─'*70}")
    
    model = CorefModel()
    t0 = time.time()
    model.train(train_docs)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Baseline
    print(f"\n{'─'*70}")
    print("Evaluación baseline (recent + agreement)")
    print(f"{'─'*70}")
    
    bl_result = evaluate_coref(test_docs, model=None)
    print(f"  Total pronombres: {bl_result['total_pronouns']:,}")
    print(f"  Con candidatos:   {bl_result['with_candidates']:,}")
    print(f"  Accuracy:         {bl_result['accuracy']:.2%}")
    
    # Modelo
    print(f"\n{'─'*70}")
    print("Evaluación del modelo")
    print(f"{'─'*70}")
    
    t0 = time.time()
    model_result = evaluate_coref(test_docs, model=model)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {model_result['accuracy']:.2%}")
    print(f"  Correctos: {model_result['correct']:,}/{model_result['with_candidates']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Baseline: {bl_result['accuracy']:.2%}")
    print(f"  DISCOURSE-001: {model_result['accuracy']:.2%}")
    print(f"  Mejora: {model_result['accuracy'] - bl_result['accuracy']:+.2%}")
    
    data = {
        "experiment_id": "DISCOURSE-001",
        "description": "Resolución de correferencia",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "pronoun_distribution": dict(pronoun_counter.most_common(20)),
        "results": {
            "baseline": bl_result,
            "discourse_001": model_result,
        },
    }
    
    output_path = os.path.join(output_dir, "discourse_001_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
