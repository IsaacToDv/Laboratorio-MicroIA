"""
Experimento SEMANTICS-002: Semantic Role Labeling (SRL)
========================================================

Objetivo: Identificar roles semánticos de argumentos verbales.

Enfoque simplificado:
- Dado un verbo (predicado), identificar sus argumentos
- Clasificar cada argumento según su relación con el verbo:
  * A0 (Agente): nsubj
  * A1 (Paciente/Tema): obj, iobj
  * AM (Modificador): obl, advmod, nmod
  * Otros: mark, case, etc.

Métrica: Accuracy de clasificación de rol semántico.
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
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


# Mapeo de deprel a rol semántico
DEPREL_TO_ROLE = {
    'nsubj': 'A0',      # Agente/Experimentador
    'nsubj:pass': 'A1', # Paciente (sujeto pasivo)
    'obj': 'A1',        # Paciente/Tema
    'iobj': 'A2',       # Beneficiario/Receptor
    'obl': 'AM',        # Modificador circunstancial
    'advmod': 'AM',     # Modificador adverbial
    'nmod': 'AM',       # Modificador nominal
    'mark': 'AM',       # Marca (subordinante)
    'case': 'AM',       # Caso (preposición)
    'xcomp': 'A1',      # Complemento abierto
    'ccomp': 'A1',      # Complemento clausal
    'advcl': 'AM',      # Cláusula adverbial
    'expl': 'A1',       # Expletivo
    'vocative': 'AM',   # Vocativo
}

def deprel_to_role(deprel: str) -> str:
    """Convierte deprel a rol semántico."""
    # Quitar subtipos (ej: obl:arg → obl)
    base_deprel = deprel.split(':')[0]
    return DEPREL_TO_ROLE.get(base_deprel, 'OTHER')


def extract_argument_features(sentence: List[dict], verb_idx: int, arg_idx: int) -> dict:
    """
    Extrae features de un argumento respecto a su verbo.
    
    Features:
    - POS del argumento
    - POS del verbo
    - Distancia entre verbo y argumento
    - Dirección (izquierda/derecha)
    - Deprel (gold, para análisis)
    - Lemma del verbo
    - Sufijos del argumento
    """
    verb = sentence[verb_idx]
    arg = sentence[arg_idx]
    
    distance = arg_idx - verb_idx
    direction = 'left' if distance < 0 else 'right'
    abs_distance = abs(distance)
    
    # Distancia bucket
    if abs_distance <= 1:
        dist_bucket = 'adjacent'
    elif abs_distance <= 3:
        dist_bucket = 'near'
    elif abs_distance <= 6:
        dist_bucket = 'mid'
    else:
        dist_bucket = 'far'
    
    # Sufijos del argumento
    arg_form = arg['form'].lower()
    suf2 = arg_form[-2:] if len(arg_form) >= 2 else arg_form
    suf3 = arg_form[-3:] if len(arg_form) >= 3 else arg_form
    
    # POS del token entre verbo y argumento (si hay)
    intervening_pos = []
    start = min(verb_idx, arg_idx) + 1
    end = max(verb_idx, arg_idx)
    for i in range(start, end):
        intervening_pos.append(sentence[i]['upos'])
    
    return {
        'verb_lemma': verb['lemma'],
        'verb_pos': verb['upos'],
        'arg_pos': arg['upos'],
        'arg_suf3': suf3,
        'direction': direction,
        'distance': abs_distance,
        'dist_bucket': dist_bucket,
        'deprel': arg['deprel'],
        'intervening_pos': tuple(intervening_pos[:3]),  # Max 3
    }


class SRLModel:
    """
    Modelo de SRL basado en item memory.
    
    Predice el rol semántico de un argumento dado un verbo.
    """
    
    def __init__(self):
        self.context_lookup = {}  # (verb_lemma, arg_pos, dir, dist) → role
        self.fallback_lookup = {}  # (arg_pos, dir) → role
        self.global_fallback = 'AM'
    
    def _make_context_key(self, features: dict) -> str:
        return (f"vl:{features['verb_lemma']}|ap:{features['arg_pos']}|"
                f"d:{features['direction']}|db:{features['dist_bucket']}")
    
    def _make_fallback_key(self, features: dict) -> str:
        return f"ap:{features['arg_pos']}|d:{features['direction']}"
    
    def train(self, sentences: List[List[dict]]):
        """Entrena con pares (verbo, argumento) del corpus."""
        ctx_counters = defaultdict(Counter)
        fb_counters = defaultdict(Counter)
        global_counter = Counter()
        
        for sent in sentences:
            for i, token in enumerate(sent):
                # Solo considerar argumentos de verbos
                head_idx = token['head']
                if head_idx == 0:
                    continue
                
                head_0based = head_idx - 1
                if head_0based >= len(sent):
                    continue
                
                head_token = sent[head_0based]
                
                # Solo si la cabeza es un verbo
                if head_token['upos'] not in ['VERB', 'AUX']:
                    continue
                
                # Extraer features
                features = extract_argument_features(sent, head_0based, i)
                role = deprel_to_role(token['deprel'])
                
                ctx_key = self._make_context_key(features)
                ctx_counters[ctx_key][role] += 1
                
                fb_key = self._make_fallback_key(features)
                fb_counters[fb_key][role] += 1
                
                global_counter[role] += 1
        
        # Crear lookups
        self.context_lookup = {k: c.most_common(1)[0][0] for k, c in ctx_counters.items()}
        self.fallback_lookup = {k: c.most_common(1)[0][0] for k, c in fb_counters.items()}
        self.global_fallback = global_counter.most_common(1)[0][0] if global_counter else 'AM'
        
        print(f"  Entrenado: {len(self.context_lookup):,} claves contextuales")
        print(f"  {len(self.fallback_lookup):,} claves fallback")
    
    def predict(self, features: dict) -> str:
        ctx_key = self._make_context_key(features)
        if ctx_key in self.context_lookup:
            return self.context_lookup[ctx_key]
        
        fb_key = self._make_fallback_key(features)
        if fb_key in self.fallback_lookup:
            return self.fallback_lookup[fb_key]
        
        return self.global_fallback


def evaluate_srl(model: SRLModel, sentences: List[List[dict]]) -> dict:
    """Evalúa el modelo de SRL."""
    correct = 0
    total = 0
    
    per_role_correct = defaultdict(int)
    per_role_total = defaultdict(int)
    
    for sent in sentences:
        for i, token in enumerate(sent):
            head_idx = token['head']
            if head_idx == 0:
                continue
            
            head_0based = head_idx - 1
            if head_0based >= len(sent):
                continue
            
            head_token = sent[head_0based]
            if head_token['upos'] not in ['VERB', 'AUX']:
                continue
            
            features = extract_argument_features(sent, head_0based, i)
            true_role = deprel_to_role(token['deprel'])
            pred_role = model.predict(features)
            
            total += 1
            per_role_total[true_role] += 1
            
            if pred_role == true_role:
                correct += 1
                per_role_correct[true_role] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    per_role = {}
    for role in sorted(per_role_total, key=lambda x: -per_role_total[x]):
        t = per_role_total[role]
        c = per_role_correct[role]
        per_role[role] = {
            'accuracy': c / t if t > 0 else 0,
            'total': t,
            'correct': c,
        }
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'per_role': per_role,
    }


def baseline_majority(train_sentences: List[List[dict]], 
                      test_sentences: List[List[dict]]) -> dict:
    """Baseline: siempre predice el rol más frecuente."""
    counter = Counter()
    
    for sent in train_sentences:
        for i, token in enumerate(sent):
            head_idx = token['head']
            if head_idx == 0:
                continue
            head_0based = head_idx - 1
            if head_0based >= len(sent):
                continue
            if sent[head_0based]['upos'] in ['VERB', 'AUX']:
                role = deprel_to_role(token['deprel'])
                counter[role] += 1
    
    majority_role = counter.most_common(1)[0][0]
    
    correct = 0
    total = 0
    for sent in test_sentences:
        for i, token in enumerate(sent):
            head_idx = token['head']
            if head_idx == 0:
                continue
            head_0based = head_idx - 1
            if head_0based >= len(sent):
                continue
            if sent[head_0based]['upos'] in ['VERB', 'AUX']:
                total += 1
                if majority_role == deprel_to_role(token['deprel']):
                    correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'majority_role': majority_role,
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SEMANTICS-002: Semantic Role Labeling (SRL)")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Distribución de roles
    print(f"\n{'─'*70}")
    print("Distribución de roles semánticos (train)")
    print(f"{'─'*70}")
    
    role_counter = Counter()
    for sent in train_s:
        for i, token in enumerate(sent):
            head_idx = token['head']
            if head_idx == 0: continue
            head_0based = head_idx - 1
            if head_0based >= len(sent): continue
            if sent[head_0based]['upos'] in ['VERB', 'AUX']:
                role_counter[deprel_to_role(token['deprel'])] += 1
    
    for role, count in role_counter.most_common():
        pct = count / sum(role_counter.values())
        print(f"  {role:<10} {count:>8,} ({pct:.2%})")
    
    # Entrenar modelo
    print(f"\n{'─'*70}")
    print("Entrenando modelo SRL")
    print(f"{'─'*70}")
    
    model = SRLModel()
    t0 = time.time()
    model.train(train_s)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Baseline
    print(f"\n{'─'*70}")
    print("Baseline: rol mayoritario")
    print(f"{'─'*70}")
    
    bl = baseline_majority(train_s, test_s)
    print(f"  Accuracy: {bl['accuracy']:.2%} (rol: {bl['majority_role']})")
    
    # Evaluar modelo
    print(f"\n{'─'*70}")
    print("Evaluación del modelo SRL")
    print(f"{'─'*70}")
    
    t0 = time.time()
    result = evaluate_srl(model, test_s)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  Correctos: {result['correct']:,}/{result['total']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    print(f"\n  Accuracy por rol:")
    for role, data in result['per_role'].items():
        print(f"    {role:<10} {data['accuracy']:.2%} ({data['correct']:,}/{data['total']:,})")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Baseline (mayority): {bl['accuracy']:.2%}")
    print(f"  SEMANTICS-002: {result['accuracy']:.2%}")
    print(f"  Mejora: {result['accuracy'] - bl['accuracy']:+.2%}")
    
    # Guardar resultados
    data = {
        "experiment_id": "SEMANTICS-002",
        "description": "Semantic Role Labeling",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "role_distribution": {k: v for k, v in role_counter.items()},
        "results": {
            "baseline": bl,
            "semantics_002": {
                "accuracy": result['accuracy'],
                "total": result['total'],
                "correct": result['correct'],
                "per_role": result['per_role'],
            },
        },
    }
    
    output_path = os.path.join(output_dir, "semantics_002_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
