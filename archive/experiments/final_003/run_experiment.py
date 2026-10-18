"""
Experimento FINAL-003: Benchmarking Comparativo
=================================================

Objetivo: Comparar sistemáticamente todos los modelos desarrollados
a lo largo del proyecto contra baselines estadísticos estándar.

Modelos evaluados:
1. Sufijos (HDC-style): Modelo basado en sufijos + item memory
2. Contexto POS: Modelo bidireccional con contexto
3. N-gramas (TnT-style): Modelo de trigramas Markov
4. Votación ensemble: Combinación de los tres modelos

Métrica: Accuracy POS, tiempo de inferencia, análisis comparativo.
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


# ═══════════════════════════════════════════════════════════════
# MODELO 1: Sufijos (HDC-style)
# ═══════════════════════════════════════════════════════════════

class SuffixModel:
    """Modelo basado en sufijos (representativo de la Fase 1)."""
    
    def __init__(self):
        self.suffix_pos = defaultdict(Counter)
        self.word_pos = defaultdict(Counter)
    
    def train(self, sentences):
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                form = token['form']
                pos = token['upos']
                self.word_pos[form.lower()][pos] += 1
                for suf_len in [4, 3, 2]:
                    if len(form) > suf_len:
                        suffix = form[-suf_len:].lower()
                        self.suffix_pos[f"{suf_len}:{suffix}"][pos] += 1
    
    def predict(self, form: str) -> str:
        # Word lookup
        if form.lower() in self.word_pos:
            total = sum(self.word_pos[form.lower()].values())
            if total >= 3:
                return self.word_pos[form.lower()].most_common(1)[0][0]
        
        # Suffix lookup
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key in self.suffix_pos:
                    return self.suffix_pos[key].most_common(1)[0][0]
        
        if form[0].isupper():
            return 'PROPN'
        return 'NOUN'
    
    def evaluate(self, sentences) -> dict:
        correct = total = 0
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                total += 1
                if self.predict(token['form']) == token['upos']:
                    correct += 1
        return {'accuracy': correct/total if total else 0, 'correct': correct, 'total': total}


# ═══════════════════════════════════════════════════════════════
# MODELO 2: N-gramas (TnT-style)
# ═══════════════════════════════════════════════════════════════

class NgramModel:
    """Modelo de n-gramas (Markov oculto simplificado, estilo TnT)."""
    
    def __init__(self):
        self.emission = defaultdict(Counter)  # word -> Counter(POS)
        self.bigram = defaultdict(Counter)  # prev_POS -> Counter(POS)
        self.trigram = defaultdict(Counter)  # (pp,p) -> Counter(POS)
        self.unigram = Counter()  # POS counts
    
    def train(self, sentences):
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            prev2_pos = '<s>'
            prev_pos = '<s>'
            
            for token in tokens:
                form = token['form'].lower()
                pos = token['upos']
                
                self.emission[form][pos] += 1
                self.unigram[pos] += 1
                self.bigram[prev_pos][pos] += 1
                self.trigram[(prev2_pos, prev_pos)][pos] += 1
                
                prev2_pos = prev_pos
                prev_pos = pos
    
    def _emission_prob(self, form: str, pos: str) -> float:
        """P(word|POS) con suavizado."""
        total_pos = self.unigram[pos]
        if total_pos == 0:
            return 1e-10
        return (self.emission[form].get(pos, 0) + 0.1) / (total_pos + 0.1 * len(self.unigram))
    
    def _transition_prob(self, prev2: str, prev: str, pos: str) -> float:
        """P(POS|prev2, prev) con backoff."""
        # Trigram
        tri_total = sum(self.trigram[(prev2, prev)].values())
        if tri_total > 5:
            return (self.trigram[(prev2, prev)].get(pos, 0) + 0.01) / (tri_total + 0.01 * len(self.unigram))
        
        # Bigram backoff
        bi_total = sum(self.bigram[prev].values())
        if bi_total > 0:
            return (self.bigram[prev].get(pos, 0) + 0.01) / (bi_total + 0.01 * len(self.unigram))
        
        # Unigram backoff
        total = sum(self.unigram.values())
        return (self.unigram.get(pos, 0) + 0.01) / (total + 0.01 * len(self.unigram))
    
    def predict_sequence(self, forms: List[str]) -> List[str]:
        """Viterbi simplificado (greedy)."""
        predictions = []
        prev2 = '<s>'
        prev = '<s>'
        
        all_pos = list(self.unigram.keys())
        
        for form in forms:
            best_pos = 'NOUN'
            best_score = -1e10
            
            for pos in all_pos:
                emit = np.log(self._emission_prob(form.lower(), pos) + 1e-10)
                trans = np.log(self._transition_prob(prev2, prev, pos) + 1e-10)
                score = emit + trans
                
                if score > best_score:
                    best_score = score
                    best_pos = pos
            
            predictions.append(best_pos)
            prev2 = prev
            prev = best_pos
        
        return predictions
    
    def evaluate(self, sentences) -> dict:
        correct = total = 0
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            forms = [t['form'] for t in tokens]
            golds = [t['upos'] for t in tokens]
            preds = self.predict_sequence(forms)
            
            for pred, gold in zip(preds, golds):
                total += 1
                if pred == gold:
                    correct += 1
        
        return {'accuracy': correct/total if total else 0, 'correct': correct, 'total': total}


# ═══════════════════════════════════════════════════════════════
# MODELO 3: Contexto bidireccional
# ═══════════════════════════════════════════════════════════════

class BidirectionalContextModel:
    """Modelo que usa contexto izquierdo y derecho (Fase 1 MORPH-004)."""
    
    def __init__(self):
        self.left_context = defaultdict(Counter)  # (prev_POS, suffix) -> Counter(POS)
        self.right_context = defaultdict(Counter)  # (next_POS, suffix) -> Counter(POS)
        self.suffix_pos = defaultdict(Counter)
    
    def train(self, sentences):
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for i, token in enumerate(tokens):
                form = token['form']
                pos = token['upos']
                
                for suf_len in [4, 3, 2]:
                    if len(form) > suf_len:
                        suffix = form[-suf_len:].lower()
                        key = f"{suf_len}:{suffix}"
                        self.suffix_pos[key][pos] += 1
                
                # Contexto izquierdo
                if i > 0:
                    prev_pos = tokens[i-1]['upos']
                    for suf_len in [3, 2]:
                        if len(form) > suf_len:
                            suffix = form[-suf_len:].lower()
                            self.left_context[(prev_pos, f"{suf_len}:{suffix}")][pos] += 1
                
                # Contexto derecho
                if i < len(tokens) - 1:
                    next_pos = tokens[i+1]['upos']
                    for suf_len in [3, 2]:
                        if len(form) > suf_len:
                            suffix = form[-suf_len:].lower()
                            self.right_context[(next_pos, f"{suf_len}:{suffix}")][pos] += 1
    
    def predict(self, form: str, prev_pos: str = None, next_pos: str = None) -> str:
        scores = Counter()
        
        # Suffix base
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key in self.suffix_pos:
                    for pos, count in self.suffix_pos[key].items():
                        scores[pos] += count
                    break
        
        # Left context
        if prev_pos:
            for suf_len in [3, 2]:
                if len(form) > suf_len:
                    suffix = form[-suf_len:].lower()
                    key = (prev_pos, f"{suf_len}:{suffix}")
                    if key in self.left_context:
                        for pos, count in self.left_context[key].items():
                            scores[pos] += count * 2  # peso mayor
                        break
        
        # Right context
        if next_pos:
            for suf_len in [3, 2]:
                if len(form) > suf_len:
                    suffix = form[-suf_len:].lower()
                    key = (next_pos, f"{suf_len}:{suffix}")
                    if key in self.right_context:
                        for pos, count in self.right_context[key].items():
                            scores[pos] += count * 2
                        break
        
        if scores:
            return scores.most_common(1)[0][0]
        
        if form[0].isupper():
            return 'PROPN'
        return 'NOUN'
    
    def evaluate(self, sentences) -> dict:
        correct = total = 0
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for i, token in enumerate(tokens):
                prev_pos = tokens[i-1]['upos'] if i > 0 else None
                next_pos = tokens[i+1]['upos'] if i < len(tokens)-1 else None
                pred = self.predict(token['form'], prev_pos, next_pos)
                total += 1
                if pred == token['upos']:
                    correct += 1
        return {'accuracy': correct/total if total else 0, 'correct': correct, 'total': total}


# ═══════════════════════════════════════════════════════════════
# MODELO 4: Ensemble (votación)
# ═══════════════════════════════════════════════════════════════

class EnsembleModel:
    """Ensemble por votación de los 3 modelos."""
    
    def __init__(self, suffix: SuffixModel, ngram: NgramModel, context: BidirectionalContextModel):
        self.suffix = suffix
        self.ngram = ngram
        self.context = context
    
    def evaluate(self, sentences) -> dict:
        correct = total = 0
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            forms = [t['form'] for t in tokens]
            golds = [t['upos'] for t in tokens]
            
            # N-gram predictions (sequence-aware)
            ngram_preds = self.ngram.predict_sequence(forms)
            
            for i, (form, gold) in enumerate(zip(forms, golds)):
                # Suffix prediction
                suf_pred = self.suffix.predict(form)
                
                # Context prediction
                prev_pos = tokens[i-1]['upos'] if i > 0 else None
                next_pos = tokens[i+1]['upos'] if i < len(tokens)-1 else None
                ctx_pred = self.context.predict(form, prev_pos, next_pos)
                
                # N-gram prediction
                ng_pred = ngram_preds[i] if i < len(ngram_preds) else 'NOUN'
                
                # Majority vote
                votes = Counter([suf_pred, ctx_pred, ng_pred])
                ensemble_pred = votes.most_common(1)[0][0]
                
                total += 1
                if ensemble_pred == gold:
                    correct += 1
        
        return {'accuracy': correct/total if total else 0, 'correct': correct, 'total': total}


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("FINAL-003: Benchmarking Comparativo")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=5000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=500)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    all_results = {}
    
    # Modelo 1: Sufijos
    print(f"\n{'─' * 70}")
    print("Modelo 1: Sufijos (HDC-style, Fase 1)")
    print(f"{'─' * 70}")
    
    suffix = SuffixModel()
    t0 = time.time()
    suffix.train(train_sents)
    train_t = time.time() - t0
    
    t0 = time.time()
    r1 = suffix.evaluate(test_sents)
    eval_t = time.time() - t0
    
    print(f"  Accuracy: {r1['accuracy']:.2%} ({r1['correct']:,}/{r1['total']:,})")
    print(f"  Train: {train_t:.1f}s | Eval: {eval_t:.2f}s")
    all_results['suffix'] = {'accuracy': r1['accuracy'], 'train_time': train_t, 'eval_time': eval_t}
    
    # Modelo 2: N-gramas
    print(f"\n{'─' * 70}")
    print("Modelo 2: N-gramas (TnT-style, Baseline estadístico)")
    print(f"{'─' * 70}")
    
    ngram = NgramModel()
    t0 = time.time()
    ngram.train(train_sents)
    train_t = time.time() - t0
    
    t0 = time.time()
    r2 = ngram.evaluate(test_sents)
    eval_t = time.time() - t0
    
    print(f"  Accuracy: {r2['accuracy']:.2%} ({r2['correct']:,}/{r2['total']:,})")
    print(f"  Train: {train_t:.1f}s | Eval: {eval_t:.2f}s")
    all_results['ngram'] = {'accuracy': r2['accuracy'], 'train_time': train_t, 'eval_time': eval_t}
    
    # Modelo 3: Contexto bidireccional
    print(f"\n{'─' * 70}")
    print("Modelo 3: Contexto bidireccional (MORPH-004)")
    print(f"{'─' * 70}")
    
    context = BidirectionalContextModel()
    t0 = time.time()
    context.train(train_sents)
    train_t = time.time() - t0
    
    t0 = time.time()
    r3 = context.evaluate(test_sents)
    eval_t = time.time() - t0
    
    print(f"  Accuracy: {r3['accuracy']:.2%} ({r3['correct']:,}/{r3['total']:,})")
    print(f"  Train: {train_t:.1f}s | Eval: {eval_t:.2f}s")
    all_results['context'] = {'accuracy': r3['accuracy'], 'train_time': train_t, 'eval_time': eval_t}
    
    # Modelo 4: Ensemble
    print(f"\n{'─' * 70}")
    print("Modelo 4: Ensemble (votación de 3 modelos)")
    print(f"{'─' * 70}")
    
    ensemble = EnsembleModel(suffix, ngram, context)
    t0 = time.time()
    r4 = ensemble.evaluate(test_sents)
    eval_t = time.time() - t0
    
    print(f"  Accuracy: {r4['accuracy']:.2%} ({r4['correct']:,}/{r4['total']:,})")
    print(f"  Eval: {eval_t:.2f}s")
    all_results['ensemble'] = {'accuracy': r4['accuracy'], 'eval_time': eval_t}
    
    # Baseline: clase mayoritaria
    pos_counts = Counter()
    total_tokens = 0
    for sent in test_sents:
        for t in sent:
            if isinstance(t['id'], tuple):
                continue
            pos_counts[t['upos']] += 1
            total_tokens += 1
    
    majority_pos = pos_counts.most_common(1)[0][0]
    majority_acc = pos_counts.most_common(1)[0][1] / total_tokens
    all_results['majority'] = {'accuracy': majority_acc}
    
    # Tabla comparativa
    print(f"\n{'─' * 70}")
    print("TABLA COMPARATIVA")
    print(f"{'─' * 70}")
    print(f"\n  {'Modelo':<30s} {'Accuracy':>10s} {'vs Baseline':>12s}")
    print(f"  {'─' * 52}")
    print(f"  {'Mayoritaria ('+majority_pos+')':<30s} {majority_acc:>10.2%} {'—':>12s}")
    
    for name, label in [('suffix', 'Sufijos (HDC)'), ('ngram', 'N-gramas (TnT)'), 
                         ('context', 'Contexto Bidir.'), ('ensemble', 'Ensemble')]:
        acc = all_results[name]['accuracy']
        delta = acc - majority_acc
        print(f"  {label:<30s} {acc:>10.2%} {'+' if delta > 0 else ''}{delta:>10.2%}")
    
    best_model = max(all_results.items(), key=lambda x: x[1]['accuracy'] if 'accuracy' in x[1] else 0)
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  Mejor modelo: {best_model[0]} ({best_model[1]['accuracy']:.2%})")
    print(f"  Baseline: {majority_acc:.2%}")
    print(f"  Mejora: +{(best_model[1]['accuracy'] - majority_acc):.2%}")
    
    data = {
        "experiment_id": "FINAL-003",
        "description": "Benchmarking comparativo de modelos POS",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()},
        "best_model": best_model[0],
        "best_accuracy": float(best_model[1]['accuracy']),
        "baseline_accuracy": float(majority_acc),
    }
    
    output_path = os.path.join(output_dir, "final_003_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
