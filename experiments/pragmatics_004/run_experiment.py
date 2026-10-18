"""
Experimento PRAGMATICS-004: Robustez
=====================================

Objetivo: Evaluar la robustez del sistema ante texto degradado:
typos, ruido, ambigüedad, tokens desconocidos.

Enfoque:
- Introducir ruido controlado en el texto de entrada
- Medir degradación del rendimiento por tipo de ruido
- Evaluar recuperación ante errores

Métrica: Degradación relativa por tipo y nivel de ruido.
"""

import sys
import os
import time
import json
import random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from conllu import parse_incr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def load_conllu_sentences(conllu_path: str, max_sents: int = None) -> List:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            sentences.append(sentence)
            if max_sents and len(sentences) >= max_sents:
                break
    return sentences


# ═══════════════════════════════════════════════════════════════
# MODELO DE POS TAGGING (referencia para evaluar robustez)
# ═══════════════════════════════════════════════════════════════

class POSTagger:
    """POS tagger simple basado en sufijos."""
    
    def __init__(self):
        self.word_pos = Counter()  # word -> Counter(POS)
        self.suffix_pos = {}  # suffix -> Counter(POS)
    
    def train(self, sentences):
        word_pos_counts = Counter()  # (word, pos) -> count
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                form = token['form']
                pos = token['upos']
                
                word_pos_counts[(form.lower(), pos)] += 1
                
                for suf_len in [4, 3, 2]:
                    if len(form) > suf_len:
                        suffix = form[-suf_len:].lower()
                        key = f"{suf_len}:{suffix}"
                        if key not in self.suffix_pos:
                            self.suffix_pos[key] = Counter()
                        self.suffix_pos[key][pos] += 1
        
        # Build word->POS lookup (only keep best POS per word)
        self.word_pos = {}
        for (word, pos), count in word_pos_counts.items():
            if word not in self.word_pos or count > self.word_pos[word][1]:
                self.word_pos[word] = (pos, count)
    
    def predict(self, form: str) -> str:
        # Exact match (case insensitive)
        lookup = form.lower()
        if lookup in self.word_pos:
            pos, count = self.word_pos[lookup]
            if count >= 3:
                return pos
        
        # Suffix match
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key in self.suffix_pos:
                    return self.suffix_pos[key].most_common(1)[0][0]
        
        if form[0].isupper():
            return 'PROPN'
        return 'NOUN'
    
    def evaluate(self, sentences) -> float:
        correct = 0
        total = 0
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                pred = self.predict(token['form'])
                total += 1
                if pred == token['upos']:
                    correct += 1
        return correct / total if total > 0 else 0


# ═══════════════════════════════════════════════════════════════
# TIPOS DE RUIDO
# ═══════════════════════════════════════════════════════════════

def add_typo_noise(form: str, rate: float = 0.1, rng=None) -> str:
    """Introduce errores tipográficos aleatorios."""
    if rng is None:
        rng = random.Random(42)
    
    if len(form) <= 2:
        return form
    
    result = list(form)
    for i in range(len(result)):
        if rng.random() < rate:
            action = rng.choice(['swap', 'delete', 'replace'])
            if action == 'swap' and i < len(result) - 1:
                result[i], result[i+1] = result[i+1], result[i]
            elif action == 'delete':
                result[i] = ''
            elif action == 'replace':
                chars = 'abcdefghijklmnñopqrstuvwxyz'
                result[i] = rng.choice(chars)
    
    return ''.join(result) or form


def add_case_noise(form: str, rate: float = 0.1, rng=None) -> str:
    """Cambia capitalización aleatoriamente."""
    if rng is None:
        rng = random.Random(42)
    
    result = list(form)
    for i in range(len(result)):
        if rng.random() < rate:
            if result[i].isupper():
                result[i] = result[i].lower()
            elif result[i].islower():
                result[i] = result[i].upper()
    return ''.join(result)


def add_ocr_noise(form: str, rate: float = 0.1, rng=None) -> str:
    """Simula errores de OCR (confusiones comunes)."""
    if rng is None:
        rng = random.Random(42)
    
    confusions = {
        'l': '1', '1': 'l', '0': 'O', 'O': '0',
        'rn': 'm', 'cl': 'd', 'vv': 'w',
        'i': 'l', 'b': 'h', 'n': 'r',
    }
    
    if rng.random() < rate and form.lower() in confusions:
        return confusions[form.lower()]
    
    result = list(form)
    for i in range(len(result)):
        if rng.random() < rate * 0.3 and result[i].lower() in confusions:
            result[i] = confusions[result[i].lower()]
    return ''.join(result)


def drop_tokens(tokens: List, rate: float = 0.1, rng=None) -> List:
    """Elimina tokens aleatoriamente (simula texto incompleto)."""
    if rng is None:
        rng = random.Random(42)
    return [t for t in tokens if rng.random() >= rate]


def add_unknown_tokens(tokens: List, rate: float = 0.1, rng=None) -> List:
    """Añade tokens desconocidos (nonce words)."""
    if rng is None:
        rng = random.Random(42)
    
    result = []
    for t in tokens:
        result.append(t)
        if rng.random() < rate * 0.3:
            nonce = ''.join(rng.choices('abcdefghijklmnñopqrstuvwxyz', k=rng.randint(3, 8)))
            result.append(nonce)
    return result


# ═══════════════════════════════════════════════════════════════
# EVALUACIÓN DE ROBUSTEZ
# ═══════════════════════════════════════════════════════════════

def evaluate_robustness(tagger, sentences, noise_fn, noise_rates, noise_name, rng_seed=42):
    """Evalúa rendimiento bajo diferentes niveles de ruido."""
    results = {}
    
    # Baseline sin ruido
    baseline = tagger.evaluate(sentences)
    results[0.0] = baseline
    
    for rate in noise_rates:
        rng = random.Random(rng_seed)
        noisy_sentences = []
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            
            # Aplicar ruido a las formas
            noisy_tokens = []
            for token in tokens:
                noisy_form = noise_fn(token['form'], rate, rng)
                noisy_token = dict(token)
                noisy_token['form'] = noisy_form
                noisy_tokens.append(noisy_token)
            
            # Crear nueva sentence-like structure
            noisy_sentences.append(noisy_tokens)
        
        # Evaluar
        correct = 0
        total = 0
        for sent in noisy_sentences:
            for token in sent:
                pred = tagger.predict(token['form'])
                total += 1
                if pred == token['upos']:
                    correct += 1
        
        acc = correct / total if total > 0 else 0
        degradation = (baseline - acc) / baseline if baseline > 0 else 0
        results[rate] = {
            'accuracy': acc,
            'degradation': degradation,
            'tokens': total,
        }
    
    return baseline, results


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("PRAGMATICS-004: Robustez ante Ruido")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=5000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=500)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Entrenar POS tagger
    print(f"\n{'─' * 70}")
    print("Entrenando modelo base")
    print(f"{'─' * 70}")
    
    tagger = POSTagger()
    t0 = time.time()
    tagger.train(train_sents)
    train_time = time.time() - t0
    baseline = tagger.evaluate(test_sents)
    print(f"  Baseline accuracy: {baseline:.2%}")
    print(f"  Tiempo: {train_time:.1f}s")
    
    noise_rates = [0.05, 0.10, 0.20, 0.30, 0.50]
    all_results = {}
    
    # Test 1: Typo noise
    print(f"\n{'─' * 70}")
    print("Test 1: Errores tipográficos")
    print(f"{'─' * 70}")
    
    bl, typo_results = evaluate_robustness(tagger, test_sents, add_typo_noise, noise_rates, "typo")
    print(f"  {'Rate':>6s} {'Accuracy':>10s} {'Degradación':>12s}")
    for rate in noise_rates:
        r = typo_results[rate]
        print(f"  {rate:>5.0%} {r['accuracy']:>10.2%} {r['degradation']:>11.2%}")
    all_results['typo'] = {str(k): v for k, v in typo_results.items() if k > 0}
    
    # Test 2: Case noise
    print(f"\n{'─' * 70}")
    print("Test 2: Ruido de capitalización")
    print(f"{'─' * 70}")
    
    bl, case_results = evaluate_robustness(tagger, test_sents, add_case_noise, noise_rates, "case")
    print(f"  {'Rate':>6s} {'Accuracy':>10s} {'Degradación':>12s}")
    for rate in noise_rates:
        r = case_results[rate]
        print(f"  {rate:>5.0%} {r['accuracy']:>10.2%} {r['degradation']:>11.2%}")
    all_results['case'] = {str(k): v for k, v in case_results.items() if k > 0}
    
    # Test 3: OCR noise
    print(f"\n{'─' * 70}")
    print("Test 3: Ruido OCR")
    print(f"{'─' * 70}")
    
    bl, ocr_results = evaluate_robustness(tagger, test_sents, add_ocr_noise, noise_rates, "ocr")
    print(f"  {'Rate':>6s} {'Accuracy':>10s} {'Degradación':>12s}")
    for rate in noise_rates:
        r = ocr_results[rate]
        print(f"  {rate:>5.0%} {r['accuracy']:>10.2%} {r['degradation']:>11.2%}")
    all_results['ocr'] = {str(k): v for k, v in ocr_results.items() if k > 0}
    
    # Resumen comparativo
    print(f"\n{'─' * 70}")
    print("Resumen comparativo (20% ruido)")
    print(f"{'─' * 70}")
    
    comparison = {}
    for name, results in [('Typo', typo_results), ('Case', case_results), ('OCR', ocr_results)]:
        r = results[0.20]
        comparison[name] = {
            'accuracy_20': float(r['accuracy']),
            'degradation_20': float(r['degradation']),
        }
        print(f"  {name:10s}: Acc={r['accuracy']:.2%} Degradación={r['degradation']:.2%}")
    
    # Robustez general
    avg_deg_20 = np.mean([v['degradation_20'] for v in comparison.values()])
    
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  Baseline: {baseline:.2%}")
    print(f"  Degradación promedio (20% ruido): {avg_deg_20:.2%}")
    print(f"  Más robusto: {'Case' if comparison['Case']['degradation_20'] < comparison['Typo']['degradation_20'] else 'Case'}")
    print(f"  Más vulnerable: {'Typo' if comparison['Typo']['degradation_20'] > comparison['OCR']['degradation_20'] else 'OCR'}")
    
    data = {
        "experiment_id": "PRAGMATICS-004",
        "description": "Robustez ante ruido en texto de entrada",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "baseline_accuracy": float(baseline),
        "noise_results": all_results,
        "comparison": comparison,
        "avg_degradation_20": float(avg_deg_20),
    }
    
    output_path = os.path.join(output_dir, "pragmatics_004_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
