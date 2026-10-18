"""
Experimento FINAL-001: SNN para POS Tagging
=============================================

Objetivo: Implementar una Spiking Neural Network (Red Neuronal de Impulsos)
para clasificación morfológica (POS tagging), evaluando si el paradigma
de cómputo basado en spikes es viable para NLP.

Enfoque biomimético:
- Neuronas LIF (Leaky Integrate-and-Fire)
- Codificación rate-based: features → frecuencias de disparo
- Competencia winner-take-all para selección de clase
- Aprendizaje STDP (Spike-Timing Dependent Plasticity) simplificado

Métrica: Accuracy POS vs baseline de sufijos.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from conllu import parse_incr


def load_conllu_sentences(conllu_path: str, max_sents: int = None) -> List:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            sentences.append(sentence)
            if max_sents and len(sentences) >= max_sents:
                break
    return sentences


class LIFNeuron:
    """
    Neurona Leaky Integrate-and-Fire.
    
    v(t+1) = v(t) * decay + input_current - threshold * spike
    """
    
    def __init__(self, threshold: float = 1.0, decay: float = 0.9,
                 refractory: int = 2):
        self.threshold = threshold
        self.decay = decay
        self.refractory = refractory
        self.voltage = 0.0
        self.spike_count = 0
        self.last_spike_time = -100
        self.spike_times = []
    
    def step(self, input_current: float, t: int) -> bool:
        """Un paso de simulación. Retorna True si dispara."""
        # Período refractario
        if t - self.last_spike_time < self.refractory:
            self.voltage *= self.decay
            return False
        
        # Integrar
        self.voltage = self.voltage * self.decay + input_current
        
        # Disparo
        if self.voltage >= self.threshold:
            self.voltage = 0.0
            self.spike_count += 1
            self.last_spike_time = t
            self.spike_times.append(t)
            return True
        
        return False
    
    def reset(self):
        self.voltage = 0.0
        self.spike_count = 0
        self.last_spike_time = -100
        self.spike_times = []


class SpikingNeuralNetwork:
    """
    SNN para clasificación POS.
    
    Arquitectura:
    - Capa de entrada: neuronas sensoriales (features del token)
    - Capa oculta: neuronas de integración
    - Capa de salida: una neurona por clase POS (WTA)
    """
    
    # Clases POS del español (UD)
    POS_CLASSES = [
        'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'DET',
        'ADP', 'CCONJ', 'SCONJ', 'NUM', 'PUNCT', 'AUX', 'SYM'
    ]
    
    def __init__(self, n_features: int = 50, n_hidden: int = 30,
                 T: int = 20, dt: float = 1.0):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = len(self.POS_CLASSES)
        self.pos_to_idx = {p: i for i, p in enumerate(self.POS_CLASSES)}
        self.T = T  # pasos de tiempo de simulación
        
        # Pesos sinápticos
        self.W_input_hidden = np.random.randn(n_features, n_hidden) * 0.1
        self.W_hidden_output = np.random.randn(n_hidden, self.n_classes) * 0.1
        
        # Bias de salida (aprendido de frecuencias)
        self.class_bias = np.zeros(self.n_classes)
        
        # Neuronas
        self.hidden_neurons = [LIFNeuron(threshold=0.8, decay=0.85) 
                               for _ in range(n_hidden)]
        self.output_neurons = [LIFNeuron(threshold=1.0, decay=0.9) 
                               for _ in range(self.n_classes)]
        
        # Feature extractors
        self.suffix_features = {}  # suffix -> feature_vector_index
        self.feature_idx = 0
        
        # Estadísticas
        self.train_stats = Counter()
    
    def _extract_features(self, form: str, pos: str = None) -> np.ndarray:
        """Extrae vector de features para un token."""
        features = np.zeros(self.n_features)
        
        # Features de sufijo (índices 0-19)
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key not in self.suffix_features:
                    if self.feature_idx < 20:
                        self.suffix_features[key] = self.feature_idx
                        self.feature_idx += 1
                if key in self.suffix_features:
                    idx = self.suffix_features[key]
                    if idx < self.n_features:
                        features[idx] = 1.0
        
        # Features de forma (índices 20-29)
        features[20] = 1.0 if form[0].isupper() else 0.0  # Capitalización
        features[21] = 1.0 if form.isupper() else 0.0  # All caps
        features[22] = 1.0 if any(c.isdigit() for c in form) else 0.0  # Has digits
        features[23] = 1.0 if form.isalpha() else 0.0  # All alpha
        features[24] = min(len(form) / 15.0, 1.0)  # Length normalized
        features[25] = 1.0 if form.endswith('s') else 0.0  # Ends with s
        features[26] = 1.0 if form.endswith('r') else 0.0  # Ends with r (infinitivo)
        features[27] = 1.0 if form.endswith('o') or form.endswith('a') else 0.0  # Gender
        features[28] = 1.0 if '-' in form else 0.0  # Compound
        features[29] = 1.0 if not form.isalnum() else 0.0  # Punctuation
        
        return features
    
    def _rate_encode(self, features: np.ndarray, scale: float = 2.0) -> np.ndarray:
        """Codifica features como tasas de disparo (rate coding)."""
        return features * scale
    
    def train_step(self, form: str, gold_pos: str, lr: float = 0.01):
        """Un paso de entrenamiento con STDP simplificado."""
        if gold_pos not in self.pos_to_idx:
            return
        
        features = self._extract_features(form, gold_pos)
        input_rates = self._rate_encode(features)
        target_idx = self.pos_to_idx[gold_pos]
        
        # Reset neuronas
        for n in self.hidden_neurons:
            n.reset()
        for n in self.output_neurons:
            n.reset()
        
        # Simulación forward
        hidden_spikes = np.zeros((self.T, self.n_hidden))
        output_spikes = np.zeros((self.T, self.n_classes))
        
        for t in range(self.T):
            # Input → Hidden
            hidden_input = input_rates @ self.W_input_hidden
            for j in range(self.n_hidden):
                if self.hidden_neurons[j].step(hidden_input[j], t):
                    hidden_spikes[t, j] = 1.0
            
            # Hidden → Output
            hidden_current = hidden_spikes[t] @ self.W_hidden_output + self.class_bias
            for k in range(self.n_classes):
                if self.output_neurons[k].step(hidden_current[k], t):
                    output_spikes[t, k] = 1.0
        
        # STDP simplificado: fortalecer conexiones al target, debilitar otros
        total_hidden = hidden_spikes.sum(axis=0)
        total_output = output_spikes.sum(axis=0)
        
        # Error signal
        if total_output[target_idx] > 0:
            # Correcto: reforzar
            for t in range(self.T):
                if output_spikes[t, target_idx] > 0:
                    for t2 in range(max(0, t-5), t):
                        if hidden_spikes[t2].sum() > 0:
                            delta = lr * hidden_spikes[t2]
                            self.W_hidden_output[:, target_idx] += delta
                    break
        else:
            # Incorrecto: reforzar target, debilitar otros
            self.W_hidden_output[:, target_idx] += lr * total_hidden * 0.5
            predicted = np.argmax(total_output + self.class_bias)
            if predicted != target_idx:
                self.W_hidden_output[:, predicted] -= lr * total_hidden * 0.3
        
        # Actualizar bias
        self.class_bias[target_idx] += lr * 0.1
    
    def train(self, sentences, epochs: int = 2, lr: float = 0.01):
        """Entrena la SNN."""
        all_tokens = []
        pos_counts = Counter()
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                all_tokens.append((token['form'], token['upos']))
                pos_counts[token['upos']] += 1
        
        # Inicializar bias con log-frecuencias
        total = sum(pos_counts.values())
        for pos, count in pos_counts.items():
            if pos in self.pos_to_idx:
                self.class_bias[self.pos_to_idx[pos]] = np.log(count / total + 1e-6) * 0.1
        
        # Registrar features de entrenamiento
        for form, pos in all_tokens:
            self._extract_features(form, pos)
        
        # Entrenamiento por épocas
        for epoch in range(epochs):
            np.random.shuffle(all_tokens)
            correct = 0
            total_eval = min(len(all_tokens), 1000)
            
            for i, (form, pos) in enumerate(all_tokens):
                self.train_step(form, pos, lr=lr * (1 - epoch * 0.2))
                
                # Evaluar cada 100 tokens
                if i % 100 == 0 and i > 0:
                    pred = self.predict(form)
                    if pred == pos:
                        correct += 1
            
            # Evaluar accuracy en subset
            eval_correct = 0
            eval_total = 0
            for form, pos in all_tokens[:total_eval]:
                pred = self.predict(form)
                if pred == pos:
                    eval_correct += 1
                eval_total += 1
            
            epoch_acc = eval_correct / eval_total if eval_total > 0 else 0
            print(f"    Epoch {epoch+1}: Acc={epoch_acc:.2%} ({eval_correct}/{eval_total})")
    
    def predict(self, form: str) -> str:
        """Predice POS tag para un token."""
        features = self._extract_features(form)
        input_rates = self._rate_encode(features)
        
        # Reset
        for n in self.hidden_neurons:
            n.reset()
        for n in self.output_neurons:
            n.reset()
        
        # Simulación
        total_output_spikes = np.zeros(self.n_classes)
        
        for t in range(self.T):
            hidden_input = input_rates @ self.W_input_hidden
            hidden_spikes = np.zeros(self.n_hidden)
            
            for j in range(self.n_hidden):
                if self.hidden_neurons[j].step(hidden_input[j], t):
                    hidden_spikes[j] = 1.0
            
            output_current = hidden_spikes @ self.W_hidden_output + self.class_bias
            for k in range(self.n_classes):
                if self.output_neurons[k].step(output_current[k], t):
                    total_output_spikes[k] += 1
        
        # Winner-take-all
        predicted_idx = np.argmax(total_output_spikes + self.class_bias)
        return self.POS_CLASSES[predicted_idx]
    
    def evaluate(self, sentences) -> dict:
        """Evalúa accuracy."""
        correct = 0
        total = 0
        per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                pred = self.predict(token['form'])
                gold = token['upos']
                total += 1
                per_class[gold]['total'] += 1
                
                if pred == gold:
                    correct += 1
                    per_class[gold]['correct'] += 1
        
        accuracy = correct / total if total > 0 else 0
        
        class_results = {}
        for pos, data in per_class.items():
            class_results[pos] = data['correct'] / data['total'] if data['total'] > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'per_class': class_results,
        }


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("FINAL-001: SNN (Spiking Neural Network) para POS Tagging")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=3000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=500)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Crear SNN
    print(f"\n{'─' * 70}")
    print("Configuración de la SNN")
    print(f"{'─' * 70}")
    
    snn = SpikingNeuralNetwork(n_features=50, n_hidden=30, T=15)
    print(f"  Neuronas input: 50 features")
    print(f"  Neuronas hidden: 30 (LIF)")
    print(f"  Neuronas output: {snn.n_classes} (WTA)")
    print(f"  Pasos de tiempo: {snn.T}")
    print(f"  Modelo neuronal: Leaky Integrate-and-Fire")
    print(f"  Aprendizaje: STDP simplificado")
    
    # Entrenar
    print(f"\n{'─' * 70}")
    print("Entrenamiento")
    print(f"{'─' * 70}")
    
    t0 = time.time()
    snn.train(train_sents, epochs=2, lr=0.02)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar
    print(f"\n{'─' * 70}")
    print("Evaluación")
    print(f"{'─' * 70}")
    
    t0 = time.time()
    results = snn.evaluate(test_sents)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {results['accuracy']:.2%}")
    print(f"  Correctos: {results['correct']:,} / {results['total']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    print(f"\n  Accuracy por clase POS:")
    for pos in snn.POS_CLASSES:
        if pos in results['per_class']:
            acc = results['per_class'][pos]
            print(f"    {pos:8s} {acc:.2%}")
    
    # Baseline: sufijo-only model
    print(f"\n{'─' * 70}")
    print("Comparación con baseline")
    print(f"{'─' * 70}")
    
    # Entrenar baseline simple
    suffix_model = defaultdict(Counter)
    for sent in train_sents:
        tokens = [t for t in sent if not isinstance(t['id'], tuple)]
        for token in tokens:
            form = token['form']
            for suf_len in [4, 3, 2]:
                if len(form) > suf_len:
                    suffix = form[-suf_len:].lower()
                    suffix_model[f"{suf_len}:{suffix}"][token['upos']] += 1
    
    bl_correct = 0
    bl_total = 0
    for sent in test_sents:
        tokens = [t for t in sent if not isinstance(t['id'], tuple)]
        for token in tokens:
            form = token['form']
            pred = 'NOUN'
            for suf_len in [4, 3, 2]:
                if len(form) > suf_len:
                    suffix = form[-suf_len:].lower()
                    key = f"{suf_len}:{suffix}"
                    if key in suffix_model:
                        pred = suffix_model[key].most_common(1)[0][0]
                        break
            if form[0].isupper() and pred == 'NOUN':
                pred = 'PROPN'
            bl_total += 1
            if pred == token['upos']:
                bl_correct += 1
    
    baseline_acc = bl_correct / bl_total if bl_total > 0 else 0
    
    print(f"  Baseline (sufijos): {baseline_acc:.2%}")
    print(f"  SNN (LIF + STDP):   {results['accuracy']:.2%}")
    print(f"  Diferencia: {(results['accuracy'] - baseline_acc):+.2%}")
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  SNN Accuracy: {results['accuracy']:.2%}")
    print(f"  Baseline: {baseline_acc:.2%}")
    print(f"  Arquitectura: 50→30→{snn.n_classes} LIF neurons")
    print(f"  Paradigma: Spiking Neural Network con STDP")
    
    data = {
        "experiment_id": "FINAL-001",
        "description": "Spiking Neural Network para POS tagging",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "snn_config": {
            "n_features": 50,
            "n_hidden": 30,
            "n_classes": snn.n_classes,
            "T": snn.T,
            "model": "Leaky Integrate-and-Fire",
            "learning": "STDP simplificado",
        },
        "results": {
            "accuracy": float(results['accuracy']),
            "correct": int(results['correct']),
            "total": int(results['total']),
        },
        "baseline_accuracy": float(baseline_acc),
        "per_class": {k: float(v) for k, v in results['per_class'].items()},
    }
    
    output_path = os.path.join(output_dir, "final_001_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
