"""
Experimento MORPH-001: Clasificación Morfológica con HDC
=========================================================

Fase 1 del currículo experimental — Morfología

Objetivo
--------
Evaluar si HDC puede aprender a predecir features morfológicas del español
a partir de la forma de la palabra y su contexto local.

Tareas
------
1. Clasificación de POS (17 categorías)
2. Clasificación de Género (Masc/Fem) para sustantivos/adjetivos
3. Clasificación de Número (Sing/Plur)
4. Clasificación de Tiempo verbal (Pres/Past/Imp/Fut)

Enfoque HDC
-----------
- Codificar palabra como hipervector de n-gramas de caracteres
- Codificar contexto local (palabras vecinas)
- Crear memoria asociativa: palabra_hv → feature_hv (bundling)
- Recuperar feature por similitud

Líneas base
-----------
1. Mayoría (baseline trivial)
2. Terminación (sufijos de 1-4 caracteres)
3. Lookup (tabla de frecuencias)

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
    """
    Carga datos de un archivo CoNLL-U y extrae información morfológica.
    
    Retorna una lista de ejemplos con:
    - word: forma superficial
    - lemma: lema
    - pos: categoría gramatical (UPOS)
    - features: diccionario de features morfológicas
    - context_left: palabra anterior
    - context_right: palabra siguiente
    """
    examples = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            
            for i, token in enumerate(tokens):
                # Contexto
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


def prepare_datasets(corpus_dir: str) -> dict:
    """Carga y prepara train, dev y test."""
    corpus_path = Path(corpus_dir)
    
    print("Cargando datos de UD Spanish-AnCora...")
    
    train_data = load_conllu_data(corpus_path / "es_ancora-ud-train.conllu")
    dev_data = load_conllu_data(corpus_path / "es_ancora-ud-dev.conllu")
    test_data = load_conllu_data(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Dev:   {len(dev_data):,} tokens")
    print(f"  Test:  {len(test_data):,} tokens")
    
    return {
        'train': train_data,
        'dev': dev_data,
        'test': test_data,
    }


# ===========================================================================
# Codificadores HDC
# ===========================================================================

class CharNgramEncoder:
    """
    Codifica palabras como bundles de n-gramas de caracteres posicionales.
    
    Para una palabra "casa":
    - Extraer n-gramas: "c", "ca", "cas", "casa", "a", "as", "asa", "s", "sa", "a"
    - Cada n-grama se bindea con su posición
    - Se agrupan todos los n-gramas en un bundle
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, n_range: Tuple[int, int] = (1, 4)):
        self.hdc = hdc
        self.n_range = n_range
        self.ngram_memory = ItemMemory(hdc)
        self.position_memory = ItemMemory(hdc)
        
        # Pre-generar posiciones (hasta 30 caracteres)
        positions = [f"__pos_{i}" for i in range(30)]
        self.position_memory.store_batch(positions)
    
    def encode(self, word: str) -> np.ndarray:
        """Codifica una palabra como hipervector de n-gramas."""
        word = word.lower()
        vectors = []
        
        for n in range(self.n_range[0], self.n_range[1] + 1):
            for i in range(len(word) - n + 1):
                ngram = word[i:i+n]
                ngram_key = f"ng_{n}_{ngram}"
                
                # Obtener o crear hipervector para el n-grama
                ngram_hv = self.ngram_memory.get(ngram_key)
                if ngram_hv is None:
                    ngram_hv = self.ngram_memory.store(ngram_key)
                
                # Bindear con posición
                pos_key = f"__pos_{i}"
                pos_hv = self.position_memory.get(pos_key)
                
                if pos_hv is not None:
                    bound = self.hdc.bind(ngram_hv, pos_hv)
                    vectors.append(bound)
        
        if not vectors:
            return self.hdc.generate(1)
        
        return self.hdc.bundle(*vectors)


class MorphologicalClassifier:
    """
    Clasificador morfológico basado en HDC.
    
    Aprendizaje:
    - Para cada ejemplo de entrenamiento, crear asociación:
      word_hv ⊕ feature_value_hv → bundle en memoria
    
    Predicción:
    - Codificar palabra como hipervector
    - Consultar memoria asociativa por similitud
    - La feature más similar es la predicción
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, 
                 encoder: CharNgramEncoder,
                 feature_name: str):
        self.hdc = hdc
        self.encoder = encoder
        self.feature_name = feature_name
        
        # Memoria de valores de feature
        self.feature_memory = ItemMemory(hdc)
        
        # Memoria asociativa: acumula bundles por cada valor de feature
        self._class_bundles: Dict[str, np.ndarray] = {}
        self._class_counts: Dict[str, int] = defaultdict(int)
    
    def _get_feature_value(self, example: dict) -> Optional[str]:
        """Extrae el valor de la feature del ejemplo."""
        if self.feature_name == 'POS':
            return example['pos']
        return example['feats'].get(self.feature_name)
    
    def train(self, train_data: List[dict]):
        """Entrena el clasificador con datos de entrenamiento."""
        n_trained = 0
        
        for example in train_data:
            feature_value = self._get_feature_value(example)
            if feature_value is None:
                continue
            
            # Codificar palabra
            word_hv = self.encoder.encode(example['word'])
            
            # Acumular bundle para esta clase
            if feature_value not in self._class_bundles:
                self._class_bundles[feature_value] = word_hv.copy().astype(np.int32)
                self.feature_memory.store(feature_value)
            else:
                self._class_bundles[feature_value] += word_hv
            
            self._class_counts[feature_value] += 1
            n_trained += 1
        
        # Cuantizar bundles acumulados
        for key in self._class_bundles:
            bundle_sum = self._class_bundles[key]
            self._class_bundles[key] = np.where(bundle_sum >= 0, 1, -1).astype(np.int8)
        
        print(f"  Entrenado con {n_trained:,} ejemplos, "
              f"{len(self._class_bundles)} clases")
    
    def predict(self, word: str) -> Tuple[str, float]:
        """Predice la feature para una palabra."""
        word_hv = self.encoder.encode(word)
        
        best_class = None
        best_sim = -2.0
        
        for class_name, class_bundle in self._class_bundles.items():
            sim = self.hdc.similarity(word_hv, class_bundle)
            if sim > best_sim:
                best_sim = sim
                best_class = class_name
        
        return best_class, best_sim
    
    def evaluate(self, test_data: List[dict]) -> dict:
        """Evalúa el clasificador en datos de prueba."""
        correct = 0
        total = 0
        predictions = []
        
        # Por clase
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        for example in test_data:
            true_value = self._get_feature_value(example)
            if true_value is None:
                continue
            
            pred_value, confidence = self.predict(example['word'])
            total += 1
            class_total[true_value] += 1
            
            if pred_value == true_value:
                correct += 1
                class_correct[true_value] += 1
            
            predictions.append({
                'word': example['word'],
                'true': true_value,
                'pred': pred_value,
                'confidence': confidence,
                'correct': pred_value == true_value,
            })
        
        accuracy = correct / total if total > 0 else 0
        
        # Macro-F1
        f1_scores = []
        for cls in class_total:
            precision = class_correct[cls] / max(1, sum(
                1 for p in predictions if p['pred'] == cls
            ))
            recall = class_correct[cls] / class_total[cls]
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            f1_scores.append(f1)
        
        macro_f1 = np.mean(f1_scores) if f1_scores else 0
        
        return {
            'accuracy': accuracy,
            'macro_f1': float(macro_f1),
            'total': total,
            'correct': correct,
            'per_class': {
                cls: {
                    'accuracy': class_correct[cls] / class_total[cls],
                    'total': class_total[cls],
                    'correct': class_correct[cls],
                }
                for cls in class_total
            },
        }


# ===========================================================================
# Líneas base
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
        'distribution': dict(counter),
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
    
    # Crear lookup de sufijos
    suffix_pred = {}
    for suffix, counter in suffix_counter.items():
        suffix_pred[suffix] = counter.most_common(1)[0][0]
    
    # Evaluar
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
        'suffix_len': suffix_len,
        'n_suffixes': len(suffix_pred),
    }


def baseline_lookup(train_data: List[dict], test_data: List[dict],
                    feature_name: str) -> dict:
    """Baseline: lookup directo de palabra → feature."""
    word_counter = defaultdict(Counter)
    
    for ex in train_data:
        val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if val is None:
            continue
        word_counter[ex['word'].lower()][val] += 1
    
    # Crear lookup
    word_pred = {}
    for word, counter in word_counter.items():
        word_pred[word] = counter.most_common(1)[0][0]
    
    # Evaluar
    correct = 0
    total = 0
    oov = 0
    
    for ex in test_data:
        true_val = ex['feats'].get(feature_name) if feature_name != 'POS' else ex['pos']
        if true_val is None:
            continue
        total += 1
        
        pred = word_pred.get(ex['word'].lower())
        if pred is None:
            oov += 1
            continue
        
        if pred == true_val:
            correct += 1
    
    return {
        'accuracy': correct / total if total > 0 else 0,
        'total': total,
        'oov': oov,
        'oov_rate': oov / total if total > 0 else 0,
        'coverage': 1.0 - (oov / total) if total > 0 else 0,
        'n_words': len(word_pred),
    }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_task(datasets: dict, feature_name: str, dimension: int = 10000,
             seed: int = 42, max_train: int = None) -> dict:
    """
    Ejecuta una tarea de clasificación morfológica completa.
    """
    print(f"\n{'='*70}")
    print(f"TAREA: Clasificación de {feature_name}")
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
    
    # --- Baselines ---
    print("\n--- Líneas base ---")
    
    bl_majority = baseline_majority(train_data, test_data, feature_name)
    print(f"  Majority:  accuracy = {bl_majority['accuracy']:.2%} "
          f"(clase: {bl_majority['majority_class']})")
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
    
    bl_lookup = baseline_lookup(train_data, test_data, feature_name)
    print(f"  Lookup:    accuracy = {bl_lookup['accuracy']:.2%} "
          f"(cobertura: {bl_lookup['coverage']:.2%})")
    results['baseline_lookup'] = bl_lookup
    
    # --- Clasificador HDC ---
    print(f"\n--- Clasificador HDC (D={dimension}) ---")
    
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    encoder = CharNgramEncoder(hdc, n_range=(1, 4))
    classifier = MorphologicalClassifier(hdc, encoder, feature_name)
    
    # Entrenamiento
    print("  Entrenando...")
    t0 = time.time()
    classifier.train(train_subset)
    train_time = time.time() - t0
    print(f"  Tiempo de entrenamiento: {train_time:.1f}s")
    results['train_time_s'] = train_time
    
    # Evaluación
    print("  Evaluando en test...")
    t0 = time.time()
    hdc_results = classifier.evaluate(test_data)
    eval_time = time.time() - t0
    
    print(f"  HDC:       accuracy = {hdc_results['accuracy']:.2%}, "
          f"macro-F1 = {hdc_results['macro_f1']:.2%}")
    print(f"  Tiempo de evaluación: {eval_time:.1f}s "
          f"({eval_time/max(1,hdc_results['total'])*1000:.2f}ms/token)")
    
    results['hdc'] = {
        'accuracy': hdc_results['accuracy'],
        'macro_f1': hdc_results['macro_f1'],
        'total': hdc_results['total'],
        'correct': hdc_results['correct'],
        'eval_time_s': eval_time,
        'per_class': hdc_results['per_class'],
    }
    
    # Comparación
    print(f"\n--- Comparación ---")
    print(f"  {'Método':<20} {'Accuracy':>10} {'Δ vs Majority':>15}")
    print(f"  {'─'*50}")
    print(f"  {'Majority':<20} {bl_majority['accuracy']:>10.2%} {'—':>15}")
    print(f"  {'Suffix-2':<20} {bl_suffix2['accuracy']:>10.2%} "
          f"{bl_suffix2['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'Suffix-3':<20} {bl_suffix3['accuracy']:>10.2%} "
          f"{bl_suffix3['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'Suffix-4':<20} {bl_suffix4['accuracy']:>10.2%} "
          f"{bl_suffix4['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'Lookup':<20} {bl_lookup['accuracy']:>10.2%} "
          f"{bl_lookup['accuracy']-bl_majority['accuracy']:>+15.2%}")
    print(f"  {'HDC':<20} {hdc_results['accuracy']:>10.2%} "
          f"{hdc_results['accuracy']-bl_majority['accuracy']:>+15.2%}")
    
    return results


def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo de clasificación morfológica."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("MORPH-001: Clasificación Morfológica con HDC")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos
    datasets = prepare_datasets(corpus_dir)
    
    # Tareas
    tasks = [
        ('POS', 10000),
        ('Gender', 10000),
        ('Number', 10000),
        ('Tense', 10000),
    ]
    
    all_results = {}
    
    for feature_name, dimension in tasks:
        result = run_task(datasets, feature_name, dimension=dimension, 
                          seed=42, max_train=50000)  # Subconjunto para velocidad
        all_results[feature_name] = result
    
    # Resumen final
    print(f"\n{'='*70}")
    print("RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"{'Tarea':<12} {'Majority':>10} {'Suffix-3':>10} "
          f"{'Lookup':>10} {'HDC':>10} {'HDC F1':>10}")
    print("-"*68)
    
    for feature_name, result in all_results.items():
        maj = result['baseline_majority']['accuracy']
        suf3 = result['baseline_suffix_3']['accuracy']
        lkup = result['baseline_lookup']['accuracy']
        hdc_acc = result['hdc']['accuracy']
        hdc_f1 = result['hdc']['macro_f1']
        
        print(f"{feature_name:<12} {maj:>10.2%} {suf3:>10.2%} "
              f"{lkup:>10.2%} {hdc_acc:>10.2%} {hdc_f1:>10.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "MORPH-001",
        "description": "Clasificación morfológica con HDC",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "dimension": 10000,
        "max_train_samples": 50000,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "morph_001_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
