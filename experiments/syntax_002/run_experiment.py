"""
Experimento SYNTAX-002: Clasificación de Relaciones de Dependencia
===================================================================

Objetivo
--------
Predecir la etiqueta de relación de dependencia (nsubj, obj, det, etc.)
dado un par (dependiente, cabeza) con sus features.

Métrica: Accuracy de clasificación de relaciones.

Enfoque
-------
Para cada par (dependiente, cabeza), codificamos:
- POS del dependiente + POS de la cabeza
- Sufijos del dependiente
- Distancia entre tokens
- Dirección (izquierda/derecha)
- POS de tokens intermedios

Líneas base:
- Majority: siempre la relación más frecuente (~25%)
- POS-pair: relación más frecuente para cada par (POS_dep, POS_head)
- Sin features contextuales

Clasificación: E1 — Resultado reproducido (al ejecutarse).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from conllu import parse_incr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ===========================================================================
# Carga de datos
# ===========================================================================

def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    """Carga oraciones con dependencias."""
    sentences = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            
            for token in tokens:
                sentence_data.append({
                    'word': token['form'],
                    'lemma': token['lemma'],
                    'pos': token['upos'],
                    'feats': token['feats'] or {},
                    'head': token['head'],
                    'deprel': token['deprel'],
                })
            
            if sentence_data:
                sentences.append(sentence_data)
    
    return sentences


# ===========================================================================
# Extractores de pares de dependencia
# ===========================================================================

def extract_dependency_pairs(sentences: List[List[dict]]) -> List[dict]:
    """
    Extrae pares (dependiente, cabeza) con features.
    
    Retorna lista de diccionarios con:
    - dep_word, dep_pos, dep_suffix (del dependiente)
    - head_word, head_pos (de la cabeza)
    - distance: distancia entre tokens
    - direction: "left" o "right"
    - deprel: etiqueta gold
    """
    pairs = []
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            head_idx = token['head']
            
            if head_idx == 0:
                # Raíz: cabeza virtual
                head_word = "<ROOT>"
                head_pos = "<ROOT>"
                distance = i + 1
                direction = "root"
            else:
                head_0based = head_idx - 1
                head_word = sentence[head_0based]['word']
                head_pos = sentence[head_0based]['pos']
                distance = abs(head_0based - i)
                direction = "left" if head_0based < i else "right"
            
            dep_word = token['word']
            dep_pos = token['pos']
            
            # Sufijos del dependiente
            dep_lower = dep_word.lower()
            suf2 = dep_lower[-2:] if len(dep_lower) >= 2 else dep_lower
            suf3 = dep_lower[-3:] if len(dep_lower) >= 3 else dep_lower
            
            # POS del token anterior al dependiente
            left_pos = sentence[i-1]['pos'] if i > 0 else "<BOS>"
            
            # POS del token anterior a la cabeza
            if head_idx > 0 and head_0based > 0:
                head_left_pos = sentence[head_0based-1]['pos']
            else:
                head_left_pos = "<BOS>"
            
            pairs.append({
                'dep_word': dep_word,
                'dep_pos': dep_pos,
                'dep_suf2': suf2,
                'dep_suf3': suf3,
                'head_word': head_word,
                'head_pos': head_pos,
                'distance': distance,
                'direction': direction,
                'dep_left_pos': left_pos,
                'head_left_pos': head_left_pos,
                'deprel': token['deprel'],
            })
    
    return pairs


# ===========================================================================
# Clasificador de relaciones
# ===========================================================================

class DeprelClassifier:
    """Clasificador de etiquetas de dependencia."""
    
    def __init__(self):
        self.contextual_lookup: Dict[str, str] = {}
        self.fallback_lookup: Dict[str, str] = {}
        self.pos_pair_lookup: Dict[str, str] = {}
        self.global_majority: str = ""
    
    def _make_full_key(self, pair: dict) -> str:
        """Clave completa con todas las features."""
        dist_bucket = min(pair['distance'], 10)
        
        return (f"dep_pos:{pair['dep_pos']} | head_pos:{pair['head_pos']} | "
                f"suf3:{pair['dep_suf3']} | dir:{pair['direction']} | "
                f"dist:{dist_bucket} | dep_lp:{pair['dep_left_pos']}")
    
    def _make_fallback_key(self, pair: dict) -> str:
        """Clave simplificada (POS pair + dirección)."""
        return f"dep_pos:{pair['dep_pos']} | head_pos:{pair['head_pos']} | dir:{pair['direction']}"
    
    def _make_pos_pair_key(self, pair: dict) -> str:
        """Clave mínima (solo POS pair)."""
        return f"{pair['dep_pos']}_{pair['head_pos']}"
    
    def train(self, pairs: List[dict]):
        """Entrena con pares de dependencia."""
        full_counters: Dict[str, Counter] = defaultdict(Counter)
        fallback_counters: Dict[str, Counter] = defaultdict(Counter)
        pos_pair_counters: Dict[str, Counter] = defaultdict(Counter)
        global_counter = Counter()
        
        for pair in pairs:
            deprel = pair['deprel']
            
            full_key = self._make_full_key(pair)
            full_counters[full_key][deprel] += 1
            
            fb_key = self._make_fallback_key(pair)
            fallback_counters[fb_key][deprel] += 1
            
            pp_key = self._make_pos_pair_key(pair)
            pos_pair_counters[pp_key][deprel] += 1
            
            global_counter[deprel] += 1
        
        # Crear lookups
        self.contextual_lookup = {
            k: c.most_common(1)[0][0] for k, c in full_counters.items()
        }
        self.fallback_lookup = {
            k: c.most_common(1)[0][0] for k, c in fallback_counters.items()
        }
        self.pos_pair_lookup = {
            k: c.most_common(1)[0][0] for k, c in pos_pair_counters.items()
        }
        self.global_majority = global_counter.most_common(1)[0][0]
        
        print(f"  Entrenado con {len(pairs):,} pares de dependencia")
        print(f"  {len(self.contextual_lookup):,} claves completas")
        print(f"  {len(self.fallback_lookup):,} claves fallback")
        print(f"  {len(self.pos_pair_lookup):,} pares POS")
        print(f"  Relación mayoritaria: {self.global_majority}")
    
    def predict(self, pair: dict) -> str:
        """Predice la etiqueta de dependencia."""
        full_key = self._make_full_key(pair)
        if full_key in self.contextual_lookup:
            return self.contextual_lookup[full_key]
        
        fb_key = self._make_fallback_key(pair)
        if fb_key in self.fallback_lookup:
            return self.fallback_lookup[fb_key]
        
        pp_key = self._make_pos_pair_key(pair)
        if pp_key in self.pos_pair_lookup:
            return self.pos_pair_lookup[pp_key]
        
        return self.global_majority


# ===========================================================================
# Evaluación
# ===========================================================================

def evaluate_deprel(classifier: DeprelClassifier, 
                     pairs: List[dict]) -> dict:
    """Evalúa accuracy de clasificación de relaciones."""
    correct = 0
    total = len(pairs)
    
    per_deprel_correct = defaultdict(int)
    per_deprel_total = defaultdict(int)
    
    for pair in pairs:
        pred = classifier.predict(pair)
        true = pair['deprel']
        
        per_deprel_total[true] += 1
        if pred == true:
            correct += 1
            per_deprel_correct[true] += 1
    
    accuracy = correct / total if total > 0 else 0
    
    # Accuracy por relación (top 10)
    per_deprel_accuracy = {}
    for deprel in sorted(per_deprel_total, key=lambda x: -per_deprel_total[x])[:10]:
        total_dep = per_deprel_total[deprel]
        correct_dep = per_deprel_correct[deprel]
        per_deprel_accuracy[deprel] = {
            'accuracy': correct_dep / total_dep if total_dep > 0 else 0,
            'total': total_dep,
            'correct': correct_dep,
        }
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'per_deprel': per_deprel_accuracy,
    }


def baseline_majority(pairs: List[dict]) -> dict:
    """Baseline: siempre la relación más frecuente."""
    counter = Counter(p['deprel'] for p in pairs)
    majority = counter.most_common(1)[0][0]
    
    correct = sum(1 for p in pairs if p['deprel'] == majority)
    total = len(pairs)
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'majority_class': majority,
    }


def baseline_pos_pair(train_pairs: List[dict], test_pairs: List[dict]) -> dict:
    """Baseline: relación más frecuente para cada par (POS_dep, POS_head)."""
    pos_pair_counters: Dict[str, Counter] = defaultdict(Counter)
    
    for pair in train_pairs:
        key = f"{pair['dep_pos']}_{pair['head_pos']}"
        pos_pair_counters[key][pair['deprel']] += 1
    
    pos_pair_pred = {
        k: c.most_common(1)[0][0] for k, c in pos_pair_counters.items()
    }
    
    correct = 0
    total = len(test_pairs)
    
    for pair in test_pairs:
        key = f"{pair['dep_pos']}_{pair['head_pos']}"
        pred = pos_pair_pred.get(key)
        if pred == pair['deprel']:
            correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
    }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SYNTAX-002: Clasificación de Relaciones de Dependencia")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_sentences):,} oraciones")
    print(f"  Test:  {len(test_sentences):,} oraciones")
    
    # Extraer pares
    print("\nExtrayendo pares de dependencia...")
    train_pairs = extract_dependency_pairs(train_sentences)
    test_pairs = extract_dependency_pairs(test_sentences)
    
    print(f"  Train pairs: {len(train_pairs):,}")
    print(f"  Test pairs:  {len(test_pairs):,}")
    
    # Distribución de relaciones
    print(f"\n{'─'*70}")
    print("Distribución de relaciones de dependencia (train)")
    print(f"{'─'*70}")
    
    deprel_counter = Counter(p['deprel'] for p in train_pairs)
    for deprel, count in deprel_counter.most_common(10):
        pct = count / len(train_pairs)
        print(f"  {deprel:<15} {count:>8,} ({pct:.2%})")
    
    # Entrenar clasificador
    print(f"\n{'─'*70}")
    print("Entrenando clasificador de relaciones")
    print(f"{'─'*70}")
    
    classifier = DeprelClassifier()
    t0 = time.time()
    classifier.train(train_pairs)
    train_time = time.time() - t0
    print(f"  Tiempo de entrenamiento: {train_time:.1f}s")
    
    # Baselines
    print(f"\n{'─'*70}")
    print("Líneas base")
    print(f"{'─'*70}")
    
    bl_majority = baseline_majority(train_pairs)
    print(f"  Majority:    accuracy = {bl_majority['accuracy']:.2%} "
          f"(clase: {bl_majority['majority_class']})")
    
    bl_pos_pair = baseline_pos_pair(train_pairs, test_pairs)
    print(f"  POS-pair:    accuracy = {bl_pos_pair['accuracy']:.2%}")
    
    # Evaluar clasificador
    print(f"\n{'─'*70}")
    print("Evaluación del clasificador")
    print(f"{'─'*70}")
    
    t0 = time.time()
    result = evaluate_deprel(classifier, test_pairs)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {result['accuracy']:.2%}")
    print(f"  Correctos: {result['correct']:,}/{result['total']:,}")
    print(f"  Tiempo:    {eval_time:.2f}s")
    
    # Accuracy por relación
    print(f"\n  Accuracy por relación (top 10):")
    for deprel, info in result['per_deprel'].items():
        print(f"    {deprel:<15} {info['accuracy']:.2%} "
              f"({info['correct']:,}/{info['total']:,})")
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Método':<30} {'Accuracy':>10} {'Δ vs Majority':>15}")
    print("-"*60)
    
    baseline_acc = bl_majority['accuracy']
    
    methods = [
        ("Majority", bl_majority['accuracy']),
        ("POS-pair", bl_pos_pair['accuracy']),
        ("SYNTAX-002 (HDC)", result['accuracy']),
    ]
    
    for name, acc in methods:
        delta = acc - baseline_acc
        print(f"{name:<30} {acc:>10.2%} {delta:>+15.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "SYNTAX-002",
        "description": "Clasificación de relaciones de dependencia",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "train_time_s": train_time,
        "results": {
            "baseline_majority": bl_majority,
            "baseline_pos_pair": bl_pos_pair,
            "syntax_002": {
                "accuracy": result['accuracy'],
                "total": result['total'],
                "correct": result['correct'],
                "per_deprel": result['per_deprel'],
            },
        },
    }
    
    output_path = os.path.join(output_dir, "syntax_002_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
