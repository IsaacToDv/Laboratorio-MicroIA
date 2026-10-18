"""
Experimento FINAL-002: Free Energy Principle (FEP) para Desambiguación
=======================================================================

Objetivo: Implementar un modelo de Active Inference basado en el
Free Energy Principle (Friston, 2010) para desambiguación POS.

Enfoque biomimético:
- El cerebro como máquina de inferencia bayesiana
- Minimización de energía libre variacional
- Creencias internas actualizadas por evidencia sensorial
- Active inference: seleccionar acción (POS tag) que minimiza surprise

Formalismo:
  F = D_KL[Q(θ) || P(θ|x)] + log P(x)
  ≈ Surprise + Complexity
  
  Donde:
  - Q(θ) = creencias internas (posterior aproximado)
  - P(θ|x) = posterior verdadero
  - P(x) = verosimilitud de la observación

Métrica: Accuracy POS, reducción de energía libre.
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


class FreeEnergyMinimizer:
    """
    Modelo de Active Inference basado en FEP.
    
    El modelo mantiene:
    - Prior P(θ): distribución previa sobre clases POS
    - Likelihood P(x|θ): probabilidad de observar features dado POS
    - Posterior Q(θ|x): creencia actualizada (aproximación variacional)
    
    La energía libre variacional es:
    F = Σ Q(θ) log[Q(θ)/P(x,θ)]
      = D_KL[Q || P(θ|x)] - log P(x)
      ≥ -log P(x)  (cota inferior de la sorpresa)
    
    Minimizar F ≈ minimizar sorpresa (surprisal).
    """
    
    POS_CLASSES = [
        'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'PRON', 'DET',
        'ADP', 'CCONJ', 'SCONJ', 'NUM', 'PUNCT', 'AUX', 'SYM'
    ]
    
    def __init__(self):
        self.n_classes = len(self.POS_CLASSES)
        self.pos_to_idx = {p: i for i, p in enumerate(self.POS_CLASSES)}
        
        # Modelo generativo
        self.prior = np.ones(self.n_classes) / self.n_classes  # P(θ) uniforme
        self.likelihood = {}  # feature -> P(feature|θ) para cada clase
        
        # Estadísticas de entrenamiento
        self.class_counts = np.zeros(self.n_classes)
        self.feature_class_counts = defaultdict(lambda: np.zeros(self.n_classes))
        self.feature_totals = defaultdict(float)
        
        # Contexto (bigramas POS)
        self.transition_counts = np.ones((self.n_classes, self.n_classes)) * 0.1
        
        # Métricas de energía libre
        self.free_energy_history = []
    
    def _extract_features(self, form: str) -> List[str]:
        """Extrae features observables del token."""
        features = []
        
        # Sufijos
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                features.append(f"suf{suf_len}:{form[-suf_len:].lower()}")
        
        # Forma
        features.append(f"cap:{form[0].isupper()}")
        features.append(f"allcap:{form.isupper()}")
        features.append(f"digit:{any(c.isdigit() for c in form)}")
        features.append(f"alpha:{form.isalpha()}")
        features.append(f"len:{min(len(form), 15)}")
        features.append(f"ends_s:{form.endswith('s')}")
        features.append(f"ends_r:{form.endswith('r')}")
        features.append(f"ends_oa:{form.endswith('o') or form.endswith('a')}")
        
        return features
    
    def train(self, sentences):
        """Entrena el modelo generativo P(x,θ)."""
        prev_pos_idx = None
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            prev_pos_idx = None
            
            for token in tokens:
                form = token['form']
                pos = token['upos']
                
                if pos not in self.pos_to_idx:
                    continue
                
                pos_idx = self.pos_to_idx[pos]
                self.class_counts[pos_idx] += 1
                
                # Transiciones
                if prev_pos_idx is not None:
                    self.transition_counts[prev_pos_idx, pos_idx] += 1
                prev_pos_idx = pos_idx
                
                # Features
                features = self._extract_features(form)
                for feat in features:
                    self.feature_class_counts[feat][pos_idx] += 1
                    self.feature_totals[feat] += 1
        
        # Calcular prior
        total = self.class_counts.sum()
        self.prior = (self.class_counts + 1) / (total + self.n_classes)
        
        # Calcular likelihood P(feature|class) con suavizado
        for feat in self.feature_class_counts:
            counts = self.feature_class_counts[feat] + 0.01  # Laplace smoothing
            self.likelihood[feat] = counts / (self.class_counts + 0.01 * self.n_classes)
        
        print(f"  Clases: {self.n_classes}")
        print(f"  Features únicas: {len(self.likelihood):,}")
        print(f"  Tokens entrenados: {total:,.0f}")
        print(f"  Prior (top 5):")
        top5 = np.argsort(-self.prior)[:5]
        for idx in top5:
            print(f"    {self.POS_CLASSES[idx]:8s} {self.prior[idx]:.4f}")
    
    def _compute_log_likelihood(self, features: List[str]) -> np.ndarray:
        """Calcula log P(x|θ) para cada clase."""
        log_lik = np.zeros(self.n_classes)
        
        for feat in features:
            if feat in self.likelihood:
                # P(feature|class) — clip para evitar log(0)
                probs = np.clip(self.likelihood[feat], 1e-10, 1.0)
                log_lik += np.log(probs)
            else:
                # Feature no vista: probabilidad uniforme baja
                log_lik += np.log(1.0 / self.n_classes) * 0.1
        
        return log_lik
    
    def compute_free_energy(self, features: List[str], 
                            context_pos: int = None) -> Tuple[np.ndarray, float]:
        """
        Calcula la energía libre variacional.
        
        F = -Σ Q(θ) [log P(x|θ) + log P(θ)] + Σ Q(θ) log Q(θ)
          = -E_Q[log P(x,θ)] + H[Q]  (entropía de Q)
        
        Con Q = softmax(log P(x|θ) + log P(θ)) (aproximación de Laplace)
        """
        # Log-likelihood
        log_lik = self._compute_log_likelihood(features)
        
        # Log-prior + contexto
        log_prior = np.log(self.prior + 1e-10)
        
        if context_pos is not None:
            # Agregar probabilidad de transición
            trans_probs = self.transition_counts[context_pos] / self.transition_counts[context_pos].sum()
            log_prior += np.log(trans_probs + 1e-10) * 0.5
        
        # Log-evidence (unnormalized posterior)
        log_posterior = log_lik + log_prior
        
        # Softmax → Q(θ)
        log_posterior -= np.max(log_posterior)  # numerical stability
        Q = np.exp(log_posterior)
        Q = Q / Q.sum()
        
        # Energía libre variacional
        # F = -log P(x) ≈ -log Σ P(x|θ)P(θ)
        log_evidence = np.log(np.exp(log_posterior).sum() + 1e-10) + np.max(log_lik + log_prior)
        
        # Free energy = -log evidence (negativo de log-evidence)
        F = -log_evidence
        
        # Descomposición
        accuracy_term = -np.sum(Q * (log_lik + log_prior))  # -E_Q[log P(x,θ)]
        complexity_term = np.sum(Q * np.log(Q + 1e-10))  # D_KL term
        
        return Q, F
    
    def predict(self, form: str, context_pos: str = None) -> Tuple[str, float]:
        """
        Predice POS tag minimizando energía libre.
        
        Retorna (POS_tag, free_energy).
        """
        features = self._extract_features(form)
        
        context_idx = None
        if context_pos and context_pos in self.pos_to_idx:
            context_idx = self.pos_to_idx[context_pos]
        
        Q, F = self.compute_free_energy(features, context_idx)
        
        predicted_idx = np.argmax(Q)
        return self.POS_CLASSES[predicted_idx], F
    
    def evaluate(self, sentences) -> dict:
        """Evalúa accuracy y energía libre promedio."""
        correct = 0
        total = 0
        total_free_energy = 0.0
        per_class = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            prev_pos = None
            
            for token in tokens:
                pred, F = self.predict(token['form'], prev_pos)
                gold = token['upos']
                total += 1
                total_free_energy += F
                per_class[gold]['total'] += 1
                
                if pred == gold:
                    correct += 1
                    per_class[gold]['correct'] += 1
                
                prev_pos = pred
        
        accuracy = correct / total if total > 0 else 0
        avg_F = total_free_energy / total if total > 0 else 0
        
        class_results = {}
        for pos, data in per_class.items():
            class_results[pos] = {
                'accuracy': data['correct'] / data['total'] if data['total'] > 0 else 0,
                'support': data['total'],
            }
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'avg_free_energy': avg_F,
            'per_class': class_results,
        }
    
    def evaluate_without_context(self, sentences) -> dict:
        """Evalúa sin contexto (sin transiciones)."""
        correct = 0
        total = 0
        total_F = 0.0
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                pred, F = self.predict(token['form'])
                total += 1
                total_F += F
                if pred == token['upos']:
                    correct += 1
        
        return {
            'accuracy': correct / total if total > 0 else 0,
            'avg_free_energy': total_F / total if total > 0 else 0,
        }


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("FINAL-002: Free Energy Principle (FEP) — Active Inference")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=5000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=500)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Crear modelo FEP
    print(f"\n{'─' * 70}")
    print("Entrenando modelo generativo P(x,θ)")
    print(f"{'─' * 70}")
    
    fep = FreeEnergyMinimizer()
    t0 = time.time()
    fep.train(train_sents)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar con contexto
    print(f"\n{'─' * 70}")
    print("Evaluación: Active Inference con contexto")
    print(f"{'─' * 70}")
    
    t0 = time.time()
    results_ctx = fep.evaluate(test_sents)
    eval_time = time.time() - t0
    
    print(f"  Accuracy: {results_ctx['accuracy']:.2%}")
    print(f"  Energía libre promedio: {results_ctx['avg_free_energy']:.4f}")
    print(f"  Correctos: {results_ctx['correct']:,} / {results_ctx['total']:,}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Evaluar sin contexto
    print(f"\n{'─' * 70}")
    print("Evaluación: Sin contexto (solo features)")
    print(f"{'─' * 70}")
    
    results_noctx = fep.evaluate_without_context(test_sents)
    print(f"  Accuracy: {results_noctx['accuracy']:.2%}")
    print(f"  Energía libre promedio: {results_noctx['avg_free_energy']:.4f}")
    print(f"  Mejora por contexto: {(results_ctx['accuracy'] - results_noctx['accuracy']):+.2%}")
    
    # Energía libre por clase
    print(f"\n{'─' * 70}")
    print("Accuracy por clase POS")
    print(f"{'─' * 70}")
    
    for pos in fep.POS_CLASSES:
        if pos in results_ctx['per_class']:
            data = results_ctx['per_class'][pos]
            print(f"  {pos:8s} Acc={data['accuracy']:.2%} (n={data['support']:,})")
    
    # Comparación con baseline
    print(f"\n{'─' * 70}")
    print("Comparaciones")
    print(f"{'─' * 70}")
    
    # Baseline: clase mayoritaria
    majority = max(fep.prior)
    majority_pos = fep.POS_CLASSES[np.argmax(fep.prior)]
    majority_count = sum(1 for sent in test_sents 
                        for t in sent if not isinstance(t['id'], tuple) 
                        and t['upos'] == majority_pos)
    total_tokens = sum(1 for sent in test_sents 
                       for t in sent if not isinstance(t['id'], tuple))
    baseline_acc = majority_count / total_tokens
    
    print(f"  Baseline (mayoritaria={majority_pos}): {baseline_acc:.2%}")
    print(f"  FEP sin contexto: {results_noctx['accuracy']:.2%}")
    print(f"  FEP con contexto: {results_ctx['accuracy']:.2%}")
    print(f"  Mejora FEP vs baseline: +{(results_ctx['accuracy'] - baseline_acc):.2%}")
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  FEP (Active Inference): {results_ctx['accuracy']:.2%}")
    print(f"  FEP (sin contexto):     {results_noctx['accuracy']:.2%}")
    print(f"  Energía libre media:    {results_ctx['avg_free_energy']:.4f}")
    print(f"  Modelo: Free Energy Minimization + Context Transitions")
    
    data = {
        "experiment_id": "FINAL-002",
        "description": "Free Energy Principle para desambiguación POS",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "fep_config": {
            "n_classes": fep.n_classes,
            "model": "Free Energy Minimization + Active Inference",
            "features": "sufijos + forma + capitalización",
            "context": "bigramas POS (transiciones)",
        },
        "results_with_context": {
            "accuracy": float(results_ctx['accuracy']),
            "correct": int(results_ctx['correct']),
            "total": int(results_ctx['total']),
            "avg_free_energy": float(results_ctx['avg_free_energy']),
        },
        "results_without_context": {
            "accuracy": float(results_noctx['accuracy']),
            "avg_free_energy": float(results_noctx['avg_free_energy']),
        },
        "baseline_accuracy": float(baseline_acc),
        "per_class": {k: {'accuracy': float(v['accuracy']), 'support': v['support']} 
                      for k, v in results_ctx['per_class'].items()},
    }
    
    output_path = os.path.join(output_dir, "final_002_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
