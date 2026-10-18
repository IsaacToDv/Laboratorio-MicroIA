"""
Experimento MORPH-005: Decodificación Secuencial para POS Tagging
==================================================================

Objetivo
--------
Implementar decodificación secuencial izquierda-derecha que use POS
predicho (no gold) del contexto, y evaluar beam search.

Hipótesis
---------
H1: Decod greedy mantiene accuracy > 90% sin usar POS gold
H2: Beam search mejora sobre greedy decoding
H3: k mayor en beam search mejora accuracy (con rendimientos decrecientes)

Configuraciones
---------------
- Baseline: MORPH-004 con POS gold (techo teórico)
- Greedy: decodificación izquierda-derecha simple
- Beam-3: beam search con k=3
- Beam-5: beam search con k=5
- Beam-10: beam search con k=10

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
from dataclasses import dataclass, field
from conllu import parse_incr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


# ===========================================================================
# Carga de datos (oraciones completas, no tokens sueltos)
# ===========================================================================

def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    """
    Carga datos como lista de oraciones (no tokens planos).
    Cada oración es una lista de tokens.
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
                })
            
            if sentence_data:
                sentences.append(sentence_data)
    
    return sentences


# ===========================================================================
# Modelo contextual (similar a MORPH-004)
# ===========================================================================

class ContextualPOSModel:
    """
    Modelo que predice POS usando sufijos de palabra + POS de vecinos.
    
    Entrena con datos gold, pero puede usarse con POS predicho en inferencia.
    """
    
    def __init__(self, suffix_lengths: List[int] = [1, 2, 3, 4],
                 context_type: str = "bilateral_pos"):
        self.suffix_lengths = suffix_lengths
        self.context_type = context_type
        
        # Lookup tables
        self.contextual_lookup: Dict[str, str] = {}
        self.fallback_lookup: Dict[str, str] = {}
        self.contextual_probs: Dict[str, Dict[str, float]] = {}  # Para beam search
        
        # Contadores para POS transitions
        self.pos_bigram_counts: Dict[str, Counter] = defaultdict(Counter)
    
    def _make_suffix_key(self, word: str) -> str:
        """Genera clave de sufijos para una palabra."""
        word_lower = word.lower()
        parts = []
        for length in self.suffix_lengths:
            if len(word_lower) >= length:
                parts.append(f"w{length}:{word_lower[-length:]}")
        return " | ".join(parts)
    
    def _make_contextual_key(self, word: str, left_pos: str, 
                              right_pos: str = "<UNK>") -> str:
        """Genera clave contextual con POS de vecinos."""
        suffix_key = self._make_suffix_key(word)
        
        if self.context_type == "left_pos":
            return f"{suffix_key} || lp:{left_pos}"
        elif self.context_type == "bilateral_pos":
            return f"{suffix_key} || lp:{left_pos} | rp:{right_pos}"
        else:
            return suffix_key
    
    def train(self, sentences: List[List[dict]]):
        """Entrena con oraciones (usando POS gold del contexto)."""
        context_counters: Dict[str, Counter] = defaultdict(Counter)
        fallback_counters: Dict[str, Counter] = defaultdict(Counter)
        
        n_tokens = 0
        
        for sentence in sentences:
            for i, token in enumerate(sentence):
                pos = token['pos']
                word = token['word']
                
                # POS del contexto (gold durante entrenamiento)
                left_pos = sentence[i-1]['pos'] if i > 0 else "<BOS>"
                right_pos = sentence[i+1]['pos'] if i < len(sentence) - 1 else "<EOS>"
                
                # Clave contextual (con POS gold)
                ctx_key = self._make_contextual_key(word, left_pos, right_pos)
                context_counters[ctx_key][pos] += 1
                
                # Clave sin contexto (fallback)
                fb_key = self._make_suffix_key(word)
                fallback_counters[fb_key][pos] += 1
                
                # Transiciones POS bigrama
                if i > 0:
                    self.pos_bigram_counts[left_pos][pos] += 1
                
                n_tokens += 1
        
        # Crear lookups (predicción mayoritaria)
        self.contextual_lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in context_counters.items()
        }
        
        self.fallback_lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in fallback_counters.items()
        }
        
        # Crear tabla de probabilidades para beam search
        for key, counter in context_counters.items():
            total = sum(counter.values())
            self.contextual_probs[key] = {
                pos: count / total for pos, count in counter.items()
            }
        
        for key, counter in fallback_counters.items():
            if key not in self.contextual_probs:
                total = sum(counter.values())
                self.contextual_probs[f"FB:{key}"] = {
                    pos: count / total for pos, count in counter.items()
                }
        
        print(f"  Entrenado con {n_tokens:,} tokens de {len(sentences):,} oraciones")
        print(f"  {len(self.contextual_lookup):,} claves contextuales")
        print(f"  {len(self.fallback_lookup):,} claves de fallback")
    
    def predict_with_context(self, word: str, left_pos: str, 
                              right_pos: str = "<UNK>") -> Tuple[str, float]:
        """Predicción puntual con contexto dado."""
        ctx_key = self._make_contextual_key(word, left_pos, right_pos)
        
        if ctx_key in self.contextual_lookup:
            # Buscar probabilidad
            prob_dict = self.contextual_probs.get(ctx_key, {})
            pred = self.contextual_lookup[ctx_key]
            prob = prob_dict.get(pred, 0.5)
            return pred, prob
        
        # Fallback a sufijos
        fb_key = self._make_suffix_key(word)
        if fb_key in self.fallback_lookup:
            prob_dict = self.contextual_probs.get(f"FB:{fb_key}", {})
            pred = self.fallback_lookup[fb_key]
            prob = prob_dict.get(pred, 0.3)
            return pred, prob
        
        # Último recurso: NOUN (clase mayoritaria)
        return "NOUN", 0.1
    
    def get_top_k_predictions(self, word: str, left_pos: str,
                               right_pos: str = "<UNK>", 
                               k: int = 5) -> List[Tuple[str, float]]:
        """Retorna top-k predicciones con sus probabilidades."""
        ctx_key = self._make_contextual_key(word, left_pos, right_pos)
        
        if ctx_key in self.contextual_probs:
            prob_dict = self.contextual_probs[ctx_key]
            sorted_preds = sorted(prob_dict.items(), key=lambda x: -x[1])
            return sorted_preds[:k]
        
        fb_key = self._make_suffix_key(word)
        fb_full_key = f"FB:{fb_key}"
        if fb_full_key in self.contextual_probs:
            prob_dict = self.contextual_probs[fb_full_key]
            sorted_preds = sorted(prob_dict.items(), key=lambda x: -x[1])
            return sorted_preds[:k]
        
        # Último recurso
        return [("NOUN", 0.3), ("VERB", 0.2), ("ADJ", 0.15), 
                ("ADV", 0.1), ("DET", 0.1)]


# ===========================================================================
# Decodificadores
# ===========================================================================

@dataclass
class Hypothesis:
    """Una hipótesis en beam search."""
    pos_sequence: List[str]
    score: float = 0.0


def decode_greedy(model: ContextualPOSModel, 
                  sentence: List[dict]) -> List[str]:
    """
    Decodificación greedy izquierda-derecha.
    
    Para cada token:
    - Usa el POS predicho del token anterior (no gold)
    - Para el contexto derecho, usa "<UNK>" (no disponible aún)
    """
    predictions = []
    
    for i, token in enumerate(sentence):
        word = token['word']
        
        # POS izquierdo (predicho, no gold)
        left_pos = predictions[i-1] if i > 0 else "<BOS>"
        
        # Contexto derecho desconocido (decodificación left-to-right)
        right_pos = "<UNK>"
        
        pred, _ = model.predict_with_context(word, left_pos, right_pos)
        predictions.append(pred)
    
    return predictions


def decode_beam_search(model: ContextualPOSModel, 
                        sentence: List[dict],
                        beam_width: int = 5) -> List[str]:
    """
    Decodificación beam search izquierda-derecha.
    
    Mantiene las k mejores hipótesis en cada paso.
    """
    if not sentence:
        return []
    
    # Inicializar beam con hipótesis vacía
    beam = [Hypothesis(pos_sequence=[], score=0.0)]
    
    for i, token in enumerate(sentence):
        word = token['word']
        new_beam = []
        
        for hyp in beam:
            # POS izquierdo de la hipótesis
            left_pos = hyp.pos_sequence[-1] if hyp.pos_sequence else "<BOS>"
            
            # Obtener top-k predicciones
            top_k = model.get_top_k_predictions(word, left_pos, "<UNK>", 
                                                 k=beam_width)
            
            for pos, prob in top_k:
                # Score logarítmico (evita underflow)
                import math
                log_prob = math.log(max(prob, 1e-10))
                new_score = hyp.score + log_prob
                
                new_hyp = Hypothesis(
                    pos_sequence=hyp.pos_sequence + [pos],
                    score=new_score
                )
                new_beam.append(new_hyp)
        
        # Seleccionar las beam_width mejores hipótesis
        new_beam.sort(key=lambda h: -h.score)
        beam = new_beam[:beam_width]
    
    # Retornar la mejor hipótesis
    best_hyp = beam[0]
    return best_hyp.pos_sequence


def decode_bidirectional(model: ContextualPOSModel,
                          sentence: List[dict],
                          n_iterations: int = 2) -> List[str]:
    """
    Decodificación iterativa bidireccional.
    
    1. Primera pasada: greedy izquierda-derecha
    2. Pasadas siguientes: refina usando predicciones anteriores
       (ahora sí tenemos contexto bilateral)
    """
    # Primera pasada: greedy
    predictions = decode_greedy(model, sentence)
    
    # Pasadas de refinamiento
    for iteration in range(n_iterations):
        new_predictions = []
        changes = 0
        
        for i, token in enumerate(sentence):
            word = token['word']
            
            left_pos = predictions[i-1] if i > 0 else "<BOS>"
            right_pos = predictions[i+1] if i < len(predictions) - 1 else "<EOS>"
            
            pred, _ = model.predict_with_context(word, left_pos, right_pos)
            new_predictions.append(pred)
            
            if pred != predictions[i]:
                changes += 1
        
        predictions = new_predictions
        
        if changes == 0:
            break  # Convergencia
    
    return predictions


# ===========================================================================
# Evaluación
# ===========================================================================

def evaluate_sentence_level(model: ContextualPOSModel,
                            sentences: List[List[dict]],
                            decoder: str = "greedy",
                            beam_width: int = 5,
                            max_sentences: int = None) -> dict:
    """Evalúa un decodificador sobre oraciones completas."""
    correct = 0
    total = 0
    sentence_correct = 0
    n_sentences = 0
    
    if max_sentences:
        sentences = sentences[:max_sentences]
    
    for sentence in sentences:
        # Decodificar
        if decoder == "greedy":
            predictions = decode_greedy(model, sentence)
        elif decoder == "beam":
            predictions = decode_beam_search(model, sentence, beam_width)
        elif decoder == "bidirectional":
            predictions = decode_bidirectional(model, sentence)
        else:
            raise ValueError(f"Decoder desconocido: {decoder}")
        
        # Evaluar
        sent_all_correct = True
        for token, pred in zip(sentence, predictions):
            total += 1
            if pred == token['pos']:
                correct += 1
            else:
                sent_all_correct = False
        
        if sent_all_correct:
            sentence_correct += 1
        n_sentences += 1
    
    token_accuracy = correct / total if total > 0 else 0
    sentence_accuracy = sentence_correct / n_sentences if n_sentences > 0 else 0
    
    return {
        'token_accuracy': token_accuracy,
        'sentence_accuracy': sentence_accuracy,
        'total_tokens': total,
        'correct_tokens': correct,
        'total_sentences': n_sentences,
        'correct_sentences': sentence_correct,
    }


def evaluate_gold_context(model: ContextualPOSModel,
                          sentences: List[List[dict]]) -> dict:
    """Evalúa usando POS gold del contexto (techo teórico, MORPH-004)."""
    correct = 0
    total = 0
    
    for sentence in sentences:
        for i, token in enumerate(sentence):
            word = token['word']
            true_pos = token['pos']
            
            left_pos = sentence[i-1]['pos'] if i > 0 else "<BOS>"
            right_pos = sentence[i+1]['pos'] if i < len(sentence) - 1 else "<EOS>"
            
            pred, _ = model.predict_with_context(word, left_pos, right_pos)
            total += 1
            if pred == true_pos:
                correct += 1
    
    return {
        'token_accuracy': correct / total if total > 0 else 0,
        'total_tokens': total,
        'correct_tokens': correct,
    }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("MORPH-005: Decodificación Secuencial para POS Tagging")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos (oraciones completas)
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_sentences = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_sentences):,} oraciones")
    print(f"  Test:  {len(test_sentences):,} oraciones")
    
    # Entrenar modelo
    print(f"\n{'='*70}")
    print("Entrenando modelo contextual (POS bilateral)")
    print(f"{'='*70}")
    
    model = ContextualPOSModel(
        suffix_lengths=[1, 2, 3, 4],
        context_type="bilateral_pos"
    )
    
    t0 = time.time()
    model.train(train_sentences)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar configuraciones
    results = {}
    
    # 1. Techo teórico (POS gold, como MORPH-004)
    print(f"\n{'─'*70}")
    print("1. Techo teórico: POS gold del contexto (MORPH-004)")
    print(f"{'─'*70}")
    t0 = time.time()
    gold_result = evaluate_gold_context(model, test_sentences)
    eval_time = time.time() - t0
    print(f"  Token accuracy: {gold_result['token_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['gold_context'] = {
        **gold_result,
        'eval_time_s': eval_time,
        'description': 'POS gold (techo teórico)'
    }
    
    # 2. Greedy decoding
    print(f"\n{'─'*70}")
    print("2. Decodificación greedy izquierda-derecha")
    print(f"{'─'*70}")
    t0 = time.time()
    greedy_result = evaluate_sentence_level(model, test_sentences, "greedy")
    eval_time = time.time() - t0
    print(f"  Token accuracy:    {greedy_result['token_accuracy']:.2%}")
    print(f"  Sentence accuracy: {greedy_result['sentence_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['greedy'] = {
        **greedy_result,
        'eval_time_s': eval_time,
        'description': 'Greedy decoding'
    }
    
    # 3. Beam search k=3
    print(f"\n{'─'*70}")
    print("3. Beam search con k=3")
    print(f"{'─'*70}")
    t0 = time.time()
    beam3_result = evaluate_sentence_level(model, test_sentences, "beam", beam_width=3)
    eval_time = time.time() - t0
    print(f"  Token accuracy:    {beam3_result['token_accuracy']:.2%}")
    print(f"  Sentence accuracy: {beam3_result['sentence_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['beam_3'] = {
        **beam3_result,
        'beam_width': 3,
        'eval_time_s': eval_time,
        'description': 'Beam search k=3'
    }
    
    # 4. Beam search k=5
    print(f"\n{'─'*70}")
    print("4. Beam search con k=5")
    print(f"{'─'*70}")
    t0 = time.time()
    beam5_result = evaluate_sentence_level(model, test_sentences, "beam", beam_width=5)
    eval_time = time.time() - t0
    print(f"  Token accuracy:    {beam5_result['token_accuracy']:.2%}")
    print(f"  Sentence accuracy: {beam5_result['sentence_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['beam_5'] = {
        **beam5_result,
        'beam_width': 5,
        'eval_time_s': eval_time,
        'description': 'Beam search k=5'
    }
    
    # 5. Beam search k=10
    print(f"\n{'─'*70}")
    print("5. Beam search con k=10")
    print(f"{'─'*70}")
    t0 = time.time()
    beam10_result = evaluate_sentence_level(model, test_sentences, "beam", beam_width=10)
    eval_time = time.time() - t0
    print(f"  Token accuracy:    {beam10_result['token_accuracy']:.2%}")
    print(f"  Sentence accuracy: {beam10_result['sentence_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['beam_10'] = {
        **beam10_result,
        'beam_width': 10,
        'eval_time_s': eval_time,
        'description': 'Beam search k=10'
    }
    
    # 6. Bidireccional iterativo
    print(f"\n{'─'*70}")
    print("6. Decodificación bidireccional iterativa")
    print(f"{'─'*70}")
    t0 = time.time()
    bidir_result = evaluate_sentence_level(model, test_sentences, "bidirectional")
    eval_time = time.time() - t0
    print(f"  Token accuracy:    {bidir_result['token_accuracy']:.2%}")
    print(f"  Sentence accuracy: {bidir_result['sentence_accuracy']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    results['bidirectional'] = {
        **bidir_result,
        'eval_time_s': eval_time,
        'description': 'Bidireccional iterativo'
    }
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Método':<30} {'Token Acc':>10} {'Sent Acc':>10} {'Δ vs Gold':>12}")
    print("-"*68)
    
    gold_acc = results['gold_context']['token_accuracy']
    
    for method in ['gold_context', 'greedy', 'beam_3', 'beam_5', 'beam_10', 'bidirectional']:
        r = results[method]
        delta = r['token_accuracy'] - gold_acc
        sent_acc = r.get('sentence_accuracy', 0)
        print(f"{r['description']:<30} {r['token_accuracy']:>10.2%} "
              f"{sent_acc:>10.2%} {delta:>+12.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "MORPH-005",
        "description": "Decodificación secuencial para POS tagging",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "train_time_s": train_time,
        "results": results,
    }
    
    output_path = os.path.join(output_dir, "morph_005_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
