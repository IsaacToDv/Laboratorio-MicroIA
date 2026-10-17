"""
Experimento MORPH-002: k-NN en Espacio HDC para Morfología
===========================================================

Objetivo
--------
Evaluar si k-NN en espacio HDC (sin bundles acumulativos) mejora
la clasificación morfológica respecto a MORPH-001.

Hipótesis
---------
H1: k-NN en espacio HDC supera al enfoque de bundles acumulativos
H2: k-NN en espacio HDC se acerca a baseline suffix-3

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

from hdc import HyperdimensionalVectors, ItemMemory


# ===========================================================================
# Carga de datos (reutilizada de MORPH-001)
# ===========================================================================

def load_conllu_data(conllu_path: str) -> List[dict]:
    """Carga datos de un archivo CoNLL-U."""
    examples = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            
            for i, token in enumerate(tokens):
                left_word = tokens[i-1]['form'] if i > 0 else "<BOS>"
                right_word = tokens[i+1]['form'] if i < len(tokens) - 1 else "<EOS>"
                
                example = {
                    'word': token['form'],
                    'lemma': token['lemma'],
                    'pos': token['upos'],
                    'feats': token['feats'] or {},
                    'context_left': left_word,
                    'context_right': right_word,
                }
                examples.append(example)
    
    return examples


# ===========================================================================
# Codificador mejorado (sufijos explícitos)
# ===========================================================================

class SuffixEncoder:
    """
    Codifica palabras usando sufijos explícitos + n-gramas.
    
    Estrategia:
    - Codificar sufijos de longitud 1, 2, 3, 4
    - Cada sufijo se bindea con su posición (desde el final)
    - Bundle de todos los sufijos
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, max_suffix_len: int = 4):
        self.hdc = hdc
        self.max_suffix_len = max_suffix_len
        self.suffix_memory = ItemMemory(hdc)
        self.position_memory = ItemMemory(hdc)
        
        # Posiciones desde el final: -1, -2, -3, -4
        positions = [f"__pos_neg_{i}" for i in range(1, max_suffix_len + 1)]
        self.position_memory.store_batch(positions)
    
    def encode(self, word: str) -> np.ndarray:
        """Codifica una palabra usando sus sufijos."""
        word = word.lower()
        vectors = []
        
        for suffix_len in range(1, self.max_suffix_len + 1):
            if len(word) >= suffix_len:
                suffix = word[-suffix_len:]
                suffix_key = f"suf_{suffix_len}_{suffix}"
                
                # Obtener o crear hipervector
                suffix_hv = self.suffix_memory.get(suffix_key)
                if suffix_hv is None:
                    suffix_hv = self.suffix_memory.store(suffix_key)
                
                # Bindear con posición negativa
                pos_key = f"__pos_neg_{suffix_len}"
                pos_hv = self.position_memory.get(pos_key)
                
                if pos_hv is not None:
                    bound = self.hdc.bind(suffix_hv, pos_hv)
                    vectors.append(bound)
        
        if not vectors:
            return self.hdc.generate(1)
        
        return self.hdc.bundle(*vectors)


# ===========================================================================
# Clasificador k-NN en espacio HDC
# ===========================================================================

class HDCKNNClassifier:
    """
    Clasificador k-NN en espacio hiperdimensional.
    
    En lugar de hacer bundles acumulativos, almacena todos los ejemplos
    y predice por votación de los k vecinos más cercanos.
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, encoder, k: int = 5):
        self.hdc = hdc
        self.encoder = encoder
        self.k = k
        
        # Almacenar ejemplos de entrenamiento
        self._train_vectors: List[np.ndarray] = []
        self._train_labels: List[str] = []
        self._train_matrix: Optional[np.ndarray] = None
    
    def _get_feature_value(self, example: dict, feature_name: str) -> Optional[str]:
        """Extrae el valor de la feature del ejemplo."""
        if feature_name == 'POS':
            return example['pos']
        return example['feats'].get(feature_name)
    
    def train(self, train_data: List[dict], feature_name: str):
        """Entrena almacenando todos los ejemplos."""
        n_stored = 0
        
        for example in train_data:
            feature_value = self._get_feature_value(example, feature_name)
            if feature_value is None:
                continue
            
            # Codificar palabra
            word_hv = self.encoder.encode(example['word'])
            
            self._train_vectors.append(word_hv)
            self._train_labels.append(feature_value)
            n_stored += 1
        
        # Crear matriz para búsqueda rápida
        if self._train_vectors:
            self._train_matrix = np.stack(self._train_vectors, axis=0)
        
        print(f"  Almacenados {n_stored:,} ejemplos de entrenamiento")
    
    def predict(self, word: str) -> Tuple[str, float]:
        """Predice usando k-NN."""
        if not self._train_vectors:
            return None, 0.0
        
        # Codificar palabra
        word_hv = self.encoder.encode(word)
        
        # Calcular similitud con todos los ejemplos
        matrix = self._train_matrix.astype(np.float64)
        query = word_hv.astype(np.float64)
        
        # Similitud coseno vectorizada
        dots = matrix @ query
        norms_matrix = np.linalg.norm(matrix, axis=1)
        norm_query = np.linalg.norm(query)
        norms = norms_matrix * norm_query
        norms = np.where(norms == 0, 1.0, norms)
        similarities = dots / norms
        
        # Encontrar k vecinos más cercanos
        k_indices = np.argsort(similarities)[-self.k:][::-1]
        k_labels = [self._train_labels[i] for i in k_indices]
        k_sims = [similarities[i] for i in k_indices]
        
        # Votación ponderada por similitud
        label_weights = defaultdict(float)
        for label, sim in zip(k_labels, k_sims):
            label_weights[label] += sim
        
        # Seleccionar clase con mayor peso
        best_label = max(label_weights, key=label_weights.get)
        best_weight = label_weights[best_label]
        
        return best_label, best_weight
    
    def evaluate(self, test_data: List[dict], feature_name: str) -> dict:
        """Evalúa el clasificador."""
        correct = 0
        total = 0
        
        for example in test_data:
            true_value = self._get_feature_value(example, feature_name)
            if true_value is None:
                continue
            
            pred_value, _ = self.predict(example['word'])
            total += 1
            
            if pred_value == true_value:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'k': self.k,
        }


# ===========================================================================
# Baselines (reutilizadas de MORPH-001)
# ===========================================================================

def baseline_suffix(train_data: List[dict], test_data: List[dict],
                    feature_name: str, suffix_len: int = 3) -> dict:
    """Baseline basado en sufijos."""
    suffix_counter = defaultdict(Counter)
    
    for ex in train_data:
        val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if val is None:
            continue
        word = ex['word'].lower()
        suffix = word[-suffix_len:] if len(word) >= suffix_len else word
        suffix_counter[suffix][val] += 1
    
    suffix_pred = {}
    for suffix, counter in suffix_counter.items():
        suffix_pred[suffix] = counter.most_common(1)[0][0]
    
    correct = 0
    total = 0
    
    for ex in test_data:
        true_val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if true_val is None:
            continue
        total += 1
        
        word = ex['word'].lower()
        suffix = word[-suffix_len:] if len(word) >= suffix_len else word
        pred = suffix_pred.get(suffix)
        
        if pred == true_val:
            correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
    }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_task_knn(datasets: dict, feature_name: str, dimension: int = 10000,
                 k_values: List[int] = [1, 3, 5, 10],
                 seed: int = 42, max_train: int = None) -> dict:
    """Ejecuta una tarea con k-NN para diferentes valores de k."""
    print(f"\n{'='*70}")
    print(f"TAREA: Clasificación de {feature_name} con k-NN")
    print(f"Dimensión: D = {dimension}")
    print(f"{'='*70}")
    
    train_data = datasets['train']
    test_data = datasets['test']
    
    if max_train:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(train_data), max_train, replace=False)
        train_subset = [train_data[i] for i in sorted(indices)]
    else:
        train_subset = train_data
    
    results = {'feature': feature_name, 'dimension': dimension}
    
    # Baseline
    bl_suffix = baseline_suffix(train_data, test_data, feature_name, suffix_len=3)
    print(f"\nBaseline suffix-3: accuracy = {bl_suffix['accuracy']:.2%}")
    results['baseline_suffix_3'] = bl_suffix
    
    # k-NN para diferentes valores de k
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    encoder = SuffixEncoder(hdc, max_suffix_len=4)
    
    knn_results = {}
    
    for k in k_values:
        print(f"\n--- k-NN con k={k} ---")
        
        classifier = HDCKNNClassifier(hdc, encoder, k=k)
        
        # Entrenamiento
        print("  Entrenando (almacenando ejemplos)...")
        t0 = time.time()
        classifier.train(train_subset, feature_name)
        train_time = time.time() - t0
        print(f"  Tiempo: {train_time:.1f}s")
        
        # Evaluación
        print("  Evaluando en test...")
        t0 = time.time()
        eval_results = classifier.evaluate(test_data, feature_name)
        eval_time = time.time() - t0
        
        print(f"  k={k}: accuracy = {eval_results['accuracy']:.2%}")
        print(f"  Tiempo: {eval_time:.1f}s ({eval_time/max(1,eval_results['total'])*1000:.2f}ms/token)")
        
        knn_results[f"k_{k}"] = {
            'accuracy': eval_results['accuracy'],
            'total': eval_results['total'],
            'correct': eval_results['correct'],
            'train_time_s': train_time,
            'eval_time_s': eval_time,
        }
    
    results['knn'] = knn_results
    
    # Comparación
    print(f"\n--- Comparación ---")
    print(f"  {'Método':<20} {'Accuracy':>10} {'Δ vs suffix-3':>15}")
    print(f"  {'─'*50}")
    print(f"  {'Suffix-3':<20} {bl_suffix['accuracy']:>10.2%} {'—':>15}")
    
    for k in k_values:
        acc = knn_results[f"k_{k}"]['accuracy']
        delta = acc - bl_suffix['accuracy']
        print(f"  {'HDC k-NN (k='+str(k)+')':<20} {acc:>10.2%} {delta:>+15.2%}")
    
    return results


def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("MORPH-002: k-NN en Espacio HDC para Morfología")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos
    corpus_path = Path(corpus_dir)
    print("\nCargando datos...")
    train_data = load_conllu_data(corpus_path / "es_ancora-ud-train.conllu")
    test_data = load_conllu_data(corpus_path / "es_ancora-ud-test.conllu")
    
    datasets = {
        'train': train_data,
        'test': test_data,
    }
    
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Test:  {len(test_data):,} tokens")
    
    # Tareas (solo 2 para velocidad)
    tasks = [
        ('POS', [1, 3, 5]),
        ('Gender', [1, 3, 5, 10]),
    ]
    
    all_results = {}
    
    for feature_name, k_values in tasks:
        result = run_task_knn(datasets, feature_name, dimension=10000,
                              k_values=k_values, seed=42, max_train=20000)
        all_results[feature_name] = result
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"{'Tarea':<12} {'Suffix-3':>10} {'k-NN k=1':>10} {'k-NN k=3':>10} {'k-NN k=5':>10}")
    print("-"*58)
    
    for feature_name, result in all_results.items():
        suf3 = result['baseline_suffix_3']['accuracy']
        k1 = result['knn'].get('k_1', {}).get('accuracy', 0)
        k3 = result['knn'].get('k_3', {}).get('accuracy', 0)
        k5 = result['knn'].get('k_5', {}).get('accuracy', 0)
        
        print(f"{feature_name:<12} {suf3:>10.2%} {k1:>10.2%} {k3:>10.2%} {k5:>10.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "MORPH-002",
        "description": "k-NN en espacio HDC para morfología",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "dimension": 10000,
        "max_train_samples": 20000,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "morph_002_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
