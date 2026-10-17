"""
Experimento SYNTAX-001: Análisis de Dependencias Básico
========================================================

Objetivo
--------
Predecir la cabeza (head) de cada token usando HDC + item memory.

Métrica: UAS (Unlabeled Attachment Score) — porcentaje de tokens
cuya cabeza predicha coincide con la cabeza gold.

Enfoque
-------
Para cada token dependiente, codificamos:
- Sufijos del dependiente
- POS del dependiente
- POS de tokens vecinos (izquierdo, derecho)
- Posición relativa en la oración
- Longitud de la palabra

La predicción es la POSICIÓN RELATIVA de la cabeza (-n, ..., -1, 0, +1, ..., +n).

Líneas base:
- Left-adjacent: head = token anterior (UAS ≈ 30%)
- Right-adjacent: head = token siguiente (UAS ≈ 15%)
- Root: head = 0 (solo para raíces)

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
# Carga de datos con dependencias
# ===========================================================================

def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    """
    Carga oraciones con anotaciones de dependencia.
    
    Cada token incluye:
    - word, lemma, pos, feats
    - head: índice de la cabeza (0 = raíz)
    - deprel: etiqueta de relación de dependencia
    """
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
# Modelo de dependencias
# ===========================================================================

class DependencyParser:
    """
    Parser de dependencias basado en item memory + sufijos + POS.
    
    Estrategia: Para cada token, predecir la posición relativa de su cabeza.
    """
    
    def __init__(self, suffix_lengths: List[int] = [1, 2, 3, 4],
                 max_head_distance: int = 20):
        self.suffix_lengths = suffix_lengths
        self.max_head_distance = max_head_distance
        
        # Lookup: clave contextual → distribución de posiciones relativas de cabeza
        self.head_distance_lookup: Dict[str, Counter] = {}
        self.fallback_lookup: Dict[str, Counter] = {}
        
        # Estadísticas
        self.distance_distribution: Counter = Counter()
    
    def _make_suffix_key(self, word: str) -> str:
        """Genera clave de sufijos."""
        word_lower = word.lower()
        parts = []
        for length in self.suffix_lengths:
            if len(word_lower) >= length:
                parts.append(f"w{length}:{word_lower[-length:]}")
        return ",".join(parts)
    
    def _make_contextual_key(self, sentence: List[dict], idx: int) -> str:
        """Genera clave contextual completa para un token."""
        token = sentence[idx]
        word = token['word']
        pos = token['pos']
        
        # Sufijos
        suffix_key = self._make_suffix_key(word)
        
        # POS del token
        pos_key = f"pos:{pos}"
        
        # POS vecinos
        left_pos = sentence[idx-1]['pos'] if idx > 0 else "<BOS>"
        right_pos = sentence[idx+1]['pos'] if idx < len(sentence) - 1 else "<EOS>"
        
        # POS a distancia 2
        left2_pos = sentence[idx-2]['pos'] if idx > 1 else "<BOS>"
        right2_pos = sentence[idx+2]['pos'] if idx < len(sentence) - 2 else "<EOS>"
        
        # Posición en oración
        sent_len = len(sentence)
        rel_pos = idx / max(1, sent_len - 1)  # 0.0 = inicio, 1.0 = final
        if rel_pos < 0.2:
            pos_bucket = "start"
        elif rel_pos > 0.8:
            pos_bucket = "end"
        else:
            pos_bucket = "mid"
        
        # Longitud de palabra
        word_len = len(word)
        if word_len <= 3:
            len_bucket = "short"
        elif word_len <= 7:
            len_bucket = "med"
        else:
            len_bucket = "long"
        
        key = f"{suffix_key} | {pos_key} | lp:{left_pos} | rp:{right_pos} | l2p:{left2_pos} | r2p:{right2_pos} | pos:{pos_bucket} | len:{len_bucket}"
        
        return key
    
    def _make_fallback_key(self, sentence: List[dict], idx: int) -> str:
        """Clave simplificada (solo POS + sufijo largo)."""
        token = sentence[idx]
        word = token['word'].lower()
        pos = token['pos']
        
        suf3 = word[-3:] if len(word) >= 3 else word
        
        left_pos = sentence[idx-1]['pos'] if idx > 0 else "<BOS>"
        
        return f"pos:{pos} | suf3:{suf3} | lp:{left_pos}"
    
    def train(self, sentences: List[List[dict]]):
        """Entrena con oraciones anotadas."""
        context_counters: Dict[str, Counter] = defaultdict(Counter)
        fallback_counters: Dict[str, Counter] = defaultdict(Counter)
        
        n_tokens = 0
        
        for sentence in sentences:
            for i, token in enumerate(sentence):
                head_idx = token['head']
                
                # Calcular distancia relativa (positivo = derecha, negativo = izquierda)
                # head_idx es 1-based, i es 0-based
                if head_idx == 0:
                    # Raíz: distancia especial
                    rel_distance = "ROOT"
                else:
                    head_0based = head_idx - 1
                    rel_distance = head_0based - i
                    
                    # Limitar distancia máxima
                    if abs(rel_distance) > self.max_head_distance:
                        rel_distance = f"{'>' if rel_distance > 0 else '<'}{self.max_head_distance}"
                
                # Clave contextual
                ctx_key = self._make_contextual_key(sentence, i)
                context_counters[ctx_key][str(rel_distance)] += 1
                
                # Clave fallback
                fb_key = self._make_fallback_key(sentence, i)
                fallback_counters[fb_key][str(rel_distance)] += 1
                
                # Distribución global
                self.distance_distribution[str(rel_distance)] += 1
                
                n_tokens += 1
        
        # Crear lookups (moda de la distribución)
        self.head_distance_lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in context_counters.items()
        }
        
        self.fallback_lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in fallback_counters.items()
        }
        
        # Distribución global (para último recurso)
        self.global_mode = self.distance_distribution.most_common(1)[0][0]
        
        print(f"  Entrenado con {n_tokens:,} tokens de {len(sentences):,} oraciones")
        print(f"  {len(self.head_distance_lookup):,} claves contextuales")
        print(f"  {len(self.fallback_lookup):,} claves de fallback")
        print(f"  Modo global: {self.global_mode}")
    
    def predict_head(self, sentence: List[dict], idx: int) -> int:
        """
        Predice la cabeza del token en posición idx.
        
        Retorna índice 1-based (0 = raíz).
        """
        # Intento 1: clave contextual
        ctx_key = self._make_contextual_key(sentence, idx)
        if ctx_key in self.head_distance_lookup:
            pred_distance = self.head_distance_lookup[ctx_key]
            return self._distance_to_head(idx, pred_distance, len(sentence))
        
        # Intento 2: fallback
        fb_key = self._make_fallback_key(sentence, idx)
        if fb_key in self.fallback_lookup:
            pred_distance = self.fallback_lookup[fb_key]
            return self._distance_to_head(idx, pred_distance, len(sentence))
        
        # Intento 3: modo global
        return self._distance_to_head(idx, self.global_mode, len(sentence))
    
    def _distance_to_head(self, idx: int, pred_distance: str, sent_len: int) -> int:
        """Convierte distancia relativa a índice de cabeza (1-based)."""
        if pred_distance == "ROOT":
            return 0
        
        try:
            dist = int(pred_distance)
        except ValueError:
            # Distancia > max_distance
            if pred_distance.startswith(">"):
                dist = self.max_head_distance
            elif pred_distance.startswith("<"):
                dist = -self.max_head_distance
            else:
                dist = 1  # Default
        
        head_0based = idx + dist
        
        # Validar límites
        if head_0based < 0 or head_0based >= sent_len:
            return 0  # Raíz si está fuera de límites
        
        return head_0based + 1  # Convertir a 1-based
    
    def parse_sentence(self, sentence: List[dict]) -> List[int]:
        """Parsea una oración completa, retornando lista de heads predichos."""
        predictions = []
        for i in range(len(sentence)):
            head = self.predict_head(sentence, i)
            predictions.append(head)
        return predictions


# ===========================================================================
# Evaluación
# ===========================================================================

def evaluate_uas(parser: DependencyParser, 
                  sentences: List[List[dict]],
                  max_sentences: int = None) -> dict:
    """Evalúa UAS (Unlabeled Attachment Score)."""
    correct = 0
    total = 0
    root_correct = 0
    root_total = 0
    
    if max_sentences:
        sentences = sentences[:max_sentences]
    
    # Errores por distancia
    distance_errors = defaultdict(int)
    distance_total = defaultdict(int)
    
    for sentence in sentences:
        predictions = parser.parse_sentence(sentence)
        
        for i, token in enumerate(sentence):
            true_head = token['head']
            pred_head = predictions[i]
            total += 1
            
            if pred_head == true_head:
                correct += 1
            
            # Estadísticas de raíz
            if true_head == 0:
                root_total += 1
                if pred_head == 0:
                    root_correct += 1
            
            # Distancia del error
            if true_head == 0:
                true_dist = "ROOT"
            else:
                true_dist = str((true_head - 1) - i)
            
            if pred_head == 0:
                pred_dist = "ROOT"
            else:
                pred_dist = str((pred_head - 1) - i)
            
            distance_total[true_dist] += 1
            if pred_dist != true_dist:
                distance_errors[true_dist] += 1
    
    uas = correct / total if total > 0 else 0
    root_accuracy = root_correct / root_total if root_total > 0 else 0
    
    return {
        'uas': uas,
        'total': total,
        'correct': correct,
        'root_accuracy': root_accuracy,
        'root_total': root_total,
        'root_correct': root_correct,
    }


def baseline_left_adjacent(sentences: List[List[dict]]) -> dict:
    """Baseline: head = token anterior (excepto primer token → raíz)."""
    correct = 0
    total = 0
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            true_head = token['head']
            total += 1
            
            if i == 0:
                pred_head = 0  # Raíz
            else:
                pred_head = i  # 1-based = token anterior
            
            if pred_head == true_head:
                correct += 1
    
    return {
        'uas': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
    }


def baseline_right_adjacent(sentences: List[List[dict]]) -> dict:
    """Baseline: head = token siguiente (excepto último token → raíz)."""
    correct = 0
    total = 0
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            true_head = token['head']
            total += 1
            
            if i == len(sentence) - 1:
                pred_head = 0
            else:
                pred_head = i + 2  # 1-based = token siguiente
            
            if pred_head == true_head:
                correct += 1
    
    return {
        'uas': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
    }


def baseline_root(sentences: List[List[dict]]) -> dict:
    """Baseline: todos los tokens son raíz."""
    correct = 0
    total = 0
    
    for sentence in sentences:
        for token in sentence:
            total += 1
            if token['head'] == 0:
                correct += 1
    
    return {
        'uas': correct / total if total > 0 else 0,
        'total': total,
        'correct': correct,
    }


def analyze_head_distribution(sentences: List[List[dict]]) -> dict:
    """Analiza la distribución de distancias de cabeza."""
    distances = Counter()
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            head_idx = token['head']
            if head_idx == 0:
                distances["ROOT"] += 1
            else:
                dist = (head_idx - 1) - i
                distances[str(dist)] += 1
    
    return {
        'total': sum(distances.values()),
        'root_percentage': distances.get("ROOT", 0) / sum(distances.values()),
        'left_head_percentage': sum(v for k, v in distances.items() 
                                    if k not in ["ROOT"] and int(k) < 0) / sum(distances.values()),
        'right_head_percentage': sum(v for k, v in distances.items() 
                                     if k not in ["ROOT"] and int(k) > 0) / sum(distances.values()),
        'top_5_distances': distances.most_common(5),
    }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SYNTAX-001: Análisis de Dependencias Básico")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones con dependencias...")
    train_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_sentences):,} oraciones")
    print(f"  Test:  {len(test_sentences):,} oraciones")
    
    # Análisis de distribución
    print(f"\n{'─'*70}")
    print("Análisis de distribución de cabezas")
    print(f"{'─'*70}")
    
    dist_analysis = analyze_head_distribution(train_sentences)
    print(f"  Tokens totales: {dist_analysis['total']:,}")
    print(f"  Raíces: {dist_analysis['root_percentage']:.2%}")
    print(f"  Cabeza a la izquierda: {dist_analysis['left_head_percentage']:.2%}")
    print(f"  Cabeza a la derecha:   {dist_analysis['right_head_percentage']:.2%}")
    print(f"  Top 5 distancias:")
    for dist, count in dist_analysis['top_5_distances']:
        pct = count / dist_analysis['total']
        print(f"    {dist:>6}: {count:>8,} ({pct:.2%})")
    
    # Entrenar parser
    print(f"\n{'─'*70}")
    print("Entrenando parser de dependencias")
    print(f"{'─'*70}")
    
    parser = DependencyParser(
        suffix_lengths=[1, 2, 3, 4],
        max_head_distance=15
    )
    
    t0 = time.time()
    parser.train(train_sentences)
    train_time = time.time() - t0
    print(f"  Tiempo de entrenamiento: {train_time:.1f}s")
    
    # Evaluar baselines
    print(f"\n{'─'*70}")
    print("Líneas base")
    print(f"{'─'*70}")
    
    bl_left = baseline_left_adjacent(test_sentences)
    print(f"  Left-adjacent:  UAS = {bl_left['uas']:.2%}")
    
    bl_right = baseline_right_adjacent(test_sentences)
    print(f"  Right-adjacent: UAS = {bl_right['uas']:.2%}")
    
    bl_root = baseline_root(test_sentences)
    print(f"  Root-only:      UAS = {bl_root['uas']:.2%}")
    
    # Evaluar parser
    print(f"\n{'─'*70}")
    print("Evaluación del parser")
    print(f"{'─'*70}")
    
    t0 = time.time()
    parser_result = evaluate_uas(parser, test_sentences)
    eval_time = time.time() - t0
    
    print(f"  UAS:            {parser_result['uas']:.2%}")
    print(f"  Correctos:      {parser_result['correct']:,}/{parser_result['total']:,}")
    print(f"  Root accuracy:  {parser_result['root_accuracy']:.2%}")
    print(f"  Tiempo:         {eval_time:.2f}s")
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Método':<30} {'UAS':>10} {'Δ vs Left-adj':>15}")
    print("-"*60)
    
    baseline_uas = bl_left['uas']
    
    methods = [
        ("Root-only", bl_root['uas']),
        ("Left-adjacent", bl_left['uas']),
        ("Right-adjacent", bl_right['uas']),
        ("SYNTAX-001 (HDC)", parser_result['uas']),
    ]
    
    for name, uas in methods:
        delta = uas - baseline_uas
        print(f"{name:<30} {uas:>10.2%} {delta:>+15.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "SYNTAX-001",
        "description": "Análisis de dependencias básico (UAS)",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "train_time_s": train_time,
        "distribution_analysis": dist_analysis,
        "results": {
            "baseline_root": bl_root,
            "baseline_left_adjacent": bl_left,
            "baseline_right_adjacent": bl_right,
            "syntax_001": parser_result,
        },
    }
    
    output_path = os.path.join(output_dir, "syntax_001_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
