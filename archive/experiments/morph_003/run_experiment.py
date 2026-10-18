"""
Experimento MORPH-003: Codificación de Sufijos Explícita con HDC
================================================================

Objetivo
--------
Evaluar si HDC con codificación explícita de sufijos y item memory
(sin bundles acumulativos) puede igualar o superar a la baseline suffix-3.

Hipótesis
---------
H1: HDC con item memory de sufijos puede alcanzar accuracy similar a suffix-3
H2: HDC puede generalizar mejor a palabras no vistas (OOV) que lookup simple

Enfoque
-------
1. Crear item memory de sufijos (no bundles acumulativos)
2. Para cada sufijo en entrenamiento, almacenar feature mayoritaria
3. Para predecir, extraer sufijos de la palabra y consultar item memory
4. Votación ponderada entre múltiples sufijos

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
# Carga de datos
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
# Clasificador basado en sufijos con HDC
# ===========================================================================

class SuffixBasedHDCClassifier:
    """
    Clasificador morfológico que usa HDC para representar sufijos
    y item memory para almacenar asociaciones sufijo→feature.
    
    A diferencia de MORPH-001, NO usa bundles acumulativos.
    En su lugar, almacena cada sufijo individualmente en item memory.
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, 
                 suffix_lengths: List[int] = [1, 2, 3, 4]):
        self.hdc = hdc
        self.suffix_lengths = suffix_lengths
        
        # Item memory para sufijos (uno por feature)
        self.suffix_memories: Dict[str, ItemMemory] = {}
        
        # Lookup directo sufijo → feature (para comparación)
        self.suffix_lookup: Dict[str, Dict[str, str]] = {}
    
    def _get_feature_value(self, example: dict, feature_name: str) -> Optional[str]:
        """Extrae el valor de la feature del ejemplo."""
        if feature_name == 'POS':
            return example['pos']
        return example['feats'].get(feature_name)
    
    def _extract_suffixes(self, word: str) -> List[Tuple[int, str]]:
        """Extrae todos los sufijos de una palabra."""
        word = word.lower()
        suffixes = []
        
        for length in self.suffix_lengths:
            if len(word) >= length:
                suffix = word[-length:]
                suffixes.append((length, suffix))
        
        return suffixes
    
    def train(self, train_data: List[dict], feature_name: str):
        """Entrena creando lookup de sufijos → feature mayoritaria."""
        print(f"  Entrenando clasificador para {feature_name}...")
        
        # Contar sufijos → features
        suffix_counters: Dict[str, Dict[str, Counter]] = {
            f"suf_{length}": defaultdict(Counter)
            for length in self.suffix_lengths
        }
        
        n_examples = 0
        
        for example in train_data:
            feature_value = self._get_feature_value(example, feature_name)
            if feature_value is None:
                continue
            
            word = example['word']
            suffixes = self._extract_suffixes(word)
            
            for length, suffix in suffixes:
                key = f"suf_{length}"
                suffix_counters[key][suffix][feature_value] += 1
            
            n_examples += 1
        
        # Crear item memory y lookup para cada longitud de sufijo
        self.suffix_memories[feature_name] = ItemMemory(self.hdc)
        self.suffix_lookup[feature_name] = {}
        
        total_suffixes = 0
        
        for length in self.suffix_lengths:
            key = f"suf_{length}"
            counters = suffix_counters[key]
            
            for suffix, feat_counter in counters.items():
                # Feature mayoritaria
                majority_feature = feat_counter.most_common(1)[0][0]
                
                # Almacenar en item memory con sufijo como clave
                suffix_key = f"{length}_{suffix}"
                self.suffix_memories[feature_name].store(suffix_key)
                
                # Almacenar en lookup directo
                if suffix_key not in self.suffix_lookup[feature_name]:
                    self.suffix_lookup[feature_name][suffix_key] = majority_feature
                
                total_suffixes += 1
        
        print(f"  Entrenado con {n_examples:,} ejemplos")
        print(f"  {total_suffixes:,} sufijos únicos almacenados")
    
    def predict(self, word: str, feature_name: str, 
                method: str = "weighted_vote") -> Tuple[str, float]:
        """
        Predice la feature para una palabra.
        
        Métodos:
        - longest_suffix: usa el sufijo más largo disponible
        - weighted_vote: votación ponderada por longitud
        - hdc_similarity: consulta por similitud en item memory
        """
        suffixes = self._extract_suffixes(word)
        
        if not suffixes:
            return None, 0.0
        
        if method == "longest_suffix":
            return self._predict_longest_suffix(suffixes, feature_name)
        elif method == "weighted_vote":
            return self._predict_weighted_vote(suffixes, feature_name)
        elif method == "hdc_similarity":
            return self._predict_hdc_similarity(suffixes, feature_name)
        else:
            raise ValueError(f"Método desconocido: {method}")
    
    def _predict_longest_suffix(self, suffixes: List[Tuple[int, str]], 
                                feature_name: str) -> Tuple[str, float]:
        """Usa el sufijo más largo disponible."""
        # Ordenar por longitud descendente
        suffixes_sorted = sorted(suffixes, key=lambda x: x[0], reverse=True)
        
        for length, suffix in suffixes_sorted:
            suffix_key = f"{length}_{suffix}"
            if suffix_key in self.suffix_lookup[feature_name]:
                return self.suffix_lookup[feature_name][suffix_key], 1.0
        
        return None, 0.0
    
    def _predict_weighted_vote(self, suffixes: List[Tuple[int, str]], 
                               feature_name: str) -> Tuple[str, float]:
        """Votación ponderada por longitud del sufijo."""
        feature_weights = defaultdict(float)
        
        for length, suffix in suffixes:
            suffix_key = f"{length}_{suffix}"
            if suffix_key in self.suffix_lookup[feature_name]:
                feature = self.suffix_lookup[feature_name][suffix_key]
                # Peso proporcional a la longitud
                feature_weights[feature] += length
        
        if not feature_weights:
            return None, 0.0
        
        best_feature = max(feature_weights, key=feature_weights.get)
        total_weight = sum(feature_weights.values())
        confidence = feature_weights[best_feature] / total_weight
        
        return best_feature, confidence
    
    def _predict_hdc_similarity(self, suffixes: List[Tuple[int, str]], 
                                feature_name: str) -> Tuple[str, float]:
        """Usa similitud HDC para encontrar sufijo más similar."""
        # Este método es más lento pero demuestra uso de HDC
        # Para cada sufijo, consultar item memory
        
        best_feature = None
        best_sim = -1.0
        
        memory = self.suffix_memories[feature_name]
        
        for length, suffix in suffixes:
            suffix_key = f"{length}_{suffix}"
            suffix_hv = memory.get(suffix_key)
            
            if suffix_hv is not None:
                # Buscar sufijo más similar en memoria
                retrieved_key, sim = memory.query_exact(suffix_hv)
                
                if sim > best_sim and retrieved_key in self.suffix_lookup[feature_name]:
                    best_sim = sim
                    best_feature = self.suffix_lookup[feature_name][retrieved_key]
        
        return best_feature, best_sim
    
    def evaluate(self, test_data: List[dict], feature_name: str,
                 method: str = "weighted_vote") -> dict:
        """Evalúa el clasificador."""
        correct = 0
        total = 0
        no_pred = 0
        
        for example in test_data:
            true_value = self._get_feature_value(example, feature_name)
            if true_value is None:
                continue
            
            pred_value, confidence = self.predict(example['word'], feature_name, method)
            total += 1
            
            if pred_value is None:
                no_pred += 1
                continue
            
            if pred_value == true_value:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'no_prediction': no_pred,
            'method': method,
        }


# ===========================================================================
# Baselines
# ===========================================================================

def baseline_majority(train_data: List[dict], test_data: List[dict],
                      feature_name: str) -> dict:
    """Baseline: siempre predice la clase mayoritaria."""
    counter = Counter()
    
    for ex in train_data:
        val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if val:
            counter[val] += 1
    
    majority_class = counter.most_common(1)[0][0] if counter else None
    
    correct = 0
    total = 0
    
    for ex in test_data:
        true_val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if true_val is None:
            continue
        total += 1
        if majority_class == true_val:
            correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
        'majority_class': majority_class,
    }


def baseline_suffix(train_data: List[dict], test_data: List[dict],
                    feature_name: str, suffix_len: int = 3) -> dict:
    """Baseline: predice basado en terminación (sufijo)."""
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

def run_task(datasets: dict, feature_name: str, dimension: int = 10000,
             seed: int = 42) -> dict:
    """Ejecuta una tarea de clasificación completa."""
    print(f"\n{'='*70}")
    print(f"TAREA: Clasificación de {feature_name}")
    print(f"Dimensión: D = {dimension}")
    print(f"{'='*70}")
    
    train_data = datasets['train']
    test_data = datasets['test']
    
    results = {'feature': feature_name, 'dimension': dimension}
    
    # --- Baselines ---
    print("\n--- Líneas base ---")
    
    bl_majority = baseline_majority(train_data, test_data, feature_name)
    print(f"  Majority:  accuracy = {bl_majority['accuracy']:.2%}")
    results['baseline_majority'] = bl_majority
    
    bl_suffix2 = baseline_suffix(train_data, test_data, feature_name, suffix_len=2)
    print(f"  Suffix-2:  accuracy = {bl_suffix2['accuracy']:.2%}")
    results['baseline_suffix_2'] = bl_suffix2
    
    bl_suffix3 = baseline_suffix(train_data, test_data, feature_name, suffix_len=3)
    print(f"  Suffix-3:  accuracy = {bl_suffix3['accuracy']:.2%}")
    results['baseline_suffix_3'] = bl_suffix3
    
    bl_suffix4 = baseline_suffix(train_data, test_data, feature_name, suffix_len=4)
    print(f"  Suffix-4:  accuracy = {bl_suffix4['accuracy']:.2%}")
    results['baseline_suffix_4'] = bl_suffix4
    
    # --- Clasificador HDC basado en sufijos ---
    print(f"\n--- Clasificador HDC basado en sufijos (D={dimension}) ---")
    
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    classifier = SuffixBasedHDCClassifier(hdc, suffix_lengths=[1, 2, 3, 4])
    
    # Entrenamiento
    t0 = time.time()
    classifier.train(train_data, feature_name)
    train_time = time.time() - t0
    print(f"  Tiempo de entrenamiento: {train_time:.1f}s")
    results['train_time_s'] = train_time
    
    # Evaluación con diferentes métodos
    methods = ["longest_suffix", "weighted_vote"]
    hdc_results = {}
    
    for method in methods:
        print(f"\n  Evaluando con método '{method}'...")
        t0 = time.time()
        eval_result = classifier.evaluate(test_data, feature_name, method=method)
        eval_time = time.time() - t0
        
        print(f"  {method}: accuracy = {eval_result['accuracy']:.2%}")
        print(f"  Tiempo: {eval_time:.1f}s ({eval_time/max(1,eval_result['total'])*1000:.2f}ms/token)")
        
        hdc_results[method] = {
            'accuracy': eval_result['accuracy'],
            'total': eval_result['total'],
            'correct': eval_result['correct'],
            'no_prediction': eval_result['no_prediction'],
            'eval_time_s': eval_time,
        }
    
    results['hdc_suffix'] = hdc_results
    
    # Comparación
    print(f"\n--- Comparación ---")
    print(f"  {'Método':<25} {'Accuracy':>10} {'Δ vs Majority':>15}")
    print(f"  {'─'*55}")
    print(f"  {'Majority':<25} {bl_majority['accuracy']:>10.2%} {'—':>15}")
    print(f"  {'Suffix-2 (baseline)':<25} {bl_suffix2['accuracy']:>10.2%} "
          f"{bl_suffix2['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'Suffix-3 (baseline)':<25} {bl_suffix3['accuracy']:>10.2%} "
          f"{bl_suffix3['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'Suffix-4 (baseline)':<25} {bl_suffix4['accuracy']:>10.2%} "
          f"{bl_suffix4['accuracy']-bl_majority['accuracy']:>+15.2%}")
    
    for method in methods:
        acc = hdc_results[method]['accuracy']
        print(f"  {'HDC-'+method:<25} {acc:>10.2%} "
              f"{acc-bl_majority['accuracy']:>+15.2%}")
    
    return results


def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("MORPH-003: Codificación de Sufijos Explícita con HDC")
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
    
    # Tareas
    tasks = ['POS', 'Gender', 'Number', 'Tense']
    
    all_results = {}
    
    for feature_name in tasks:
        result = run_task(datasets, feature_name, dimension=10000, seed=42)
        all_results[feature_name] = result
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"{'Tarea':<12} {'Majority':>10} {'Suffix-3':>10} "
          f"{'HDC-longest':>12} {'HDC-weighted':>13}")
    print("-"*63)
    
    for feature_name, result in all_results.items():
        maj = result['baseline_majority']['accuracy']
        suf3 = result['baseline_suffix_3']['accuracy']
        hdc_long = result['hdc_suffix']['longest_suffix']['accuracy']
        hdc_weight = result['hdc_suffix']['weighted_vote']['accuracy']
        
        print(f"{feature_name:<12} {maj:>10.2%} {suf3:>10.2%} "
              f"{hdc_long:>12.2%} {hdc_weight:>13.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "MORPH-003",
        "description": "Codificación de sufijos explícita con HDC (sin bundles acumulativos)",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "dimension": 10000,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "morph_003_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
