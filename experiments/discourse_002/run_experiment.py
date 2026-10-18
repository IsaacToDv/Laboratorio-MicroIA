"""
Experimento DISCOURSE-002: Continuidad Temática
================================================

Objetivo: Detectar cambios de tema en el discurso.

Enfoque:
- Representar cada oración como un hipervector de su contenido
- Calcular similitud entre oraciones consecutivas
- Identificar puntos de quiebre temático

Métrica: Precision/Recall en detección de cambios de tema.
"""

import sys
import os
import time
import json
import numpy as np
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
                    'form': token['form'],
                    'lemma': token['lemma'],
                    'upos': token['upos'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


class TopicSegmenter:
    """
    Segmentador temático basado en similitud léxica entre oraciones.
    
    Calcula similitud entre pares de oraciones consecutivas y detecta
    puntos donde la similitud cae por debajo de un umbral.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.word_freq = Counter()
    
    def _sentence_to_bag(self, sentence: List[dict]) -> Counter:
        """Convierte oración a bag-of-words (lemmas de content words)."""
        words = Counter()
        for token in sentence:
            if token['upos'] in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                words[token['lemma'].lower()] += 1
        return words
    
    def _cosine_similarity(self, bag1: Counter, bag2: Counter) -> float:
        """Similitud coseno entre dos bags of words."""
        if not bag1 or not bag2:
            return 0.0
        
        # Vector común
        all_words = set(bag1.keys()) | set(bag2.keys())
        
        dot = sum(bag1.get(w, 0) * bag2.get(w, 0) for w in all_words)
        norm1 = np.sqrt(sum(v**2 for v in bag1.values()))
        norm2 = np.sqrt(sum(v**2 for v in bag2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def train(self, sentences: List[List[dict]]):
        """Entrena: calcula frecuencia de palabras y ajusta umbral."""
        for sent in sentences:
            for token in sent:
                if token['upos'] in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN']:
                    self.word_freq[token['lemma'].lower()] += 1
        
        # Calcular similitudes entre oraciones consecutivas
        similarities = []
        for i in range(len(sentences) - 1):
            bag1 = self._sentence_to_bag(sentences[i])
            bag2 = self._sentence_to_bag(sentences[i+1])
            sim = self._cosine_similarity(bag1, bag2)
            similarities.append(sim)
        
        if similarities:
            self.mean_similarity = float(np.mean(similarities))
            self.std_similarity = float(np.std(similarities))
            # Umbral adaptativo: similitudes por debajo de la media - 0.5*std
            # indica caída significativa en cohesión léxica
            self.threshold = max(0.01, self.mean_similarity - 0.5 * self.std_similarity)
        
        print(f"  Umbral ajustado: {self.threshold:.4f}")
        print(f"  Similitud media: {self.mean_similarity:.4f} ± {self.std_similarity:.4f}")
    
    def segment(self, sentences: List[List[dict]]) -> List[bool]:
        """
        Detecta cambios de tema.
        
        Retorna lista de booleanos: True si hay cambio de tema después de la oración i.
        """
        boundaries = []
        
        for i in range(len(sentences) - 1):
            bag1 = self._sentence_to_bag(sentences[i])
            bag2 = self._sentence_to_bag(sentences[i+1])
            sim = self._cosine_similarity(bag1, bag2)
            boundaries.append(sim < self.threshold)
        
        return boundaries


def create_gold_boundaries(sentences: List[List[dict]], 
                            window: int = 5) -> List[bool]:
    """
    Crea gold estándar heurístico de boundaries temáticos.
    
    Basado en cambios bruscos en el vocabulario entre ventanas de oraciones.
    """
    boundaries = []
    
    for i in range(len(sentences) - 1):
        # Ventana izquierda
        left_start = max(0, i - window + 1)
        left_words = set()
        for s in range(left_start, i + 1):
            for token in sentences[s]:
                if token['upos'] in ['NOUN', 'PROPN']:
                    left_words.add(token['lemma'].lower())
        
        # Ventana derecha
        right_end = min(len(sentences), i + window + 1)
        right_words = set()
        for s in range(i + 1, right_end):
            for token in sentences[s]:
                if token['upos'] in ['NOUN', 'PROPN']:
                    right_words.add(token['lemma'].lower())
        
        # Jaccard similarity
        if left_words and right_words:
            jaccard = len(left_words & right_words) / len(left_words | right_words)
        else:
            jaccard = 0.0
        
        boundaries.append(jaccard < 0.05)  # Muy baja solapamiento = cambio
    
    return boundaries


def evaluate_segmentation(predicted: List[bool], gold: List[bool]) -> dict:
    """Evalúa segmentación con precision, recall, F1."""
    tp = fp = fn = tn = 0
    
    for p, g in zip(predicted, gold):
        if p and g: tp += 1
        elif p and not g: fp += 1
        elif not p and g: fn += 1
        else: tn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn),
        'predicted_boundaries': int(sum(predicted)),
        'gold_boundaries': int(sum(gold)),
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("DISCOURSE-002: Continuidad Temática")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Entrenar segmentador
    print(f"\n{'─'*70}")
    print("Entrenando segmentador temático")
    print(f"{'─'*70}")
    
    segmenter = TopicSegmenter()
    t0 = time.time()
    segmenter.train(train_s)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar en test
    print(f"\n{'─'*70}")
    print("Evaluación de segmentación")
    print(f"{'─'*70}")
    
    # Predicciones
    t0 = time.time()
    predicted = segmenter.segment(test_s)
    eval_time = time.time() - t0
    
    # Gold
    gold = create_gold_boundaries(test_s)
    
    # Evaluar
    result = evaluate_segmentation(predicted, gold)
    
    print(f"  Boundaries predichos: {result['predicted_boundaries']}")
    print(f"  Boundaries gold:      {result['gold_boundaries']}")
    print(f"  Precision: {result['precision']:.2%}")
    print(f"  Recall:    {result['recall']:.2%}")
    print(f"  F1:        {result['f1']:.2%}")
    print(f"  TP={result['tp']} FP={result['fp']} FN={result['fn']}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Baseline: predecir boundary cada N oraciones
    print(f"\n{'─'*70}")
    print("Baselines")
    print(f"{'─'*70}")
    
    for period in [3, 5, 10]:
        baseline_pred = [i % period == period - 1 for i in range(len(test_s) - 1)]
        bl_result = evaluate_segmentation(baseline_pred, gold)
        print(f"  Cada {period} oraciones: P={bl_result['precision']:.2%} "
              f"R={bl_result['recall']:.2%} F1={bl_result['f1']:.2%}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  DISCOURSE-002: F1={result['f1']:.2%}")
    print(f"  Precision={result['precision']:.2%}, Recall={result['recall']:.2%}")
    
    data = {
        "experiment_id": "DISCOURSE-002",
        "description": "Continuidad temática / segmentación de discurso",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "threshold": segmenter.threshold,
        "mean_similarity": segmenter.mean_similarity,
        "results": result,
    }
    
    output_path = os.path.join(output_dir, "discourse_002_results.json")
    
    # Convertir tipos numpy a nativos
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    data = convert(data)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
