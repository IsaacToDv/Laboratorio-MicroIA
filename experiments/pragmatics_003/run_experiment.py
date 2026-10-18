"""
Experimento PRAGMATICS-003: Análisis de Errores
================================================

Objetivo: Análisis sistemático de errores a través de todas las fases,
identificando patrones, cascadas y categorías de error.

Enfoque:
- Clasificar errores por tipo y frecuencia
- Identificar errores en cascada (un error morfológico → error sintáctico → error semántico)
- Analizar distribución de errores por categoría gramatical

Métrica: Distribución de errores, tasa de cascada, categorías problemáticas.
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


def load_conllu_sentences(conllu_path: str, max_sents: int = None) -> List:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            sentences.append(sentence)
            if max_sents and len(sentences) >= max_sents:
                break
    return sentences


class ErrorAnalyzer:
    """Analizador de errores a través de niveles lingüísticos."""
    
    def __init__(self):
        # Modelos simples por nivel
        self.pos_suffix_model = {}  # suffix -> Counter(POS)
        self.deprel_pos_model = defaultdict(Counter)  # POS -> Counter(deprel)
        
        # Contadores de errores
        self.morph_errors = Counter()  # tipo de error -> count
        self.syntax_errors = Counter()
        self.semantic_errors = Counter()
        self.cascade_errors = 0  # errores causados por error en nivel previo
        self.total_tokens = 0
    
    def train(self, sentences):
        """Entrena modelos de cada nivel."""
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                form = token['form']
                pos = token['upos']
                deprel = token['deprel']
                
                # Modelo POS por sufijo
                for suf_len in [4, 3, 2]:
                    if len(form) > suf_len:
                        suffix = form[-suf_len:].lower()
                        key = f"{suf_len}:{suffix}"
                        if key not in self.pos_suffix_model:
                            self.pos_suffix_model[key] = Counter()
                        self.pos_suffix_model[key][pos] += 1
                
                # Modelo deprel por POS
                self.deprel_pos_model[pos][deprel] += 1
    
    def predict_pos(self, form: str) -> str:
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key in self.pos_suffix_model:
                    return self.pos_suffix_model[key].most_common(1)[0][0]
        if form[0].isupper():
            return 'PROPN'
        return 'NOUN'
    
    def predict_deprel(self, pos: str) -> str:
        if pos in self.deprel_pos_model:
            return self.deprel_pos_model[pos].most_common(1)[0][0]
        return 'dep'
    
    def analyze_sentence(self, sentence) -> dict:
        """Analiza errores de una oración en todos los niveles."""
        tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
        errors = []
        
        for token in tokens:
            self.total_tokens += 1
            form = token['form']
            gold_pos = token['upos']
            gold_deprel = token['deprel']
            gold_head = token['head']
            
            # Nivel 1: POS
            pred_pos = self.predict_pos(form)
            pos_correct = pred_pos == gold_pos
            
            if not pos_correct:
                error_type = f"{gold_pos}→{pred_pos}"
                self.morph_errors[error_type] += 1
                errors.append({
                    'level': 'morphology',
                    'token': form,
                    'gold': gold_pos,
                    'predicted': pred_pos,
                    'type': 'pos_error',
                })
            
            # Nivel 2: Dependency (solo si POS correcto)
            pred_deprel = self.predict_deprel(pred_pos)
            deprel_correct = pred_deprel == gold_deprel
            
            if not deprel_correct:
                self.syntax_errors[f"{gold_deprel}→{pred_deprel}"] += 1
                
                is_cascade = not pos_correct  # Si POS falló, es cascada
                if is_cascade:
                    self.cascade_errors += 1
                
                errors.append({
                    'level': 'syntax',
                    'token': form,
                    'gold': gold_deprel,
                    'predicted': pred_deprel,
                    'type': 'deprel_error',
                    'cascade': is_cascade,
                })
            
            # Nivel 3: Semantic (NER proxy: PROPN detection)
            gold_is_entity = gold_pos == 'PROPN'
            pred_is_entity = pred_pos == 'PROPN'
            
            if gold_is_entity != pred_is_entity:
                self.semantic_errors['ner_miss' if gold_is_entity else 'ner_false'] += 1
                errors.append({
                    'level': 'semantics',
                    'token': form,
                    'gold': gold_is_entity,
                    'predicted': pred_is_entity,
                    'type': 'ner_error',
                    'cascade': not pos_correct,
                })
        
        return {
            'errors': errors,
            'n_tokens': len(tokens),
            'n_errors': len(errors),
        }
    
    def analyze_corpus(self, sentences) -> dict:
        """Analiza todo el corpus."""
        total_errors = []
        sent_error_rates = []
        
        for sent in sentences:
            result = self.analyze_sentence(sent)
            total_errors.extend(result['errors'])
            if result['n_tokens'] > 0:
                sent_error_rates.append(result['n_errors'] / result['n_tokens'])
        
        return {
            'total_tokens': self.total_tokens,
            'total_errors': len(total_errors),
            'error_rate': len(total_errors) / self.total_tokens if self.total_tokens > 0 else 0,
            'avg_sent_error_rate': float(np.mean(sent_error_rates)) if sent_error_rates else 0,
            'morph_errors': dict(self.morph_errors.most_common(15)),
            'syntax_errors': dict(self.syntax_errors.most_common(15)),
            'semantic_errors': dict(self.semantic_errors.most_common(10)),
            'cascade_rate': self.cascade_errors / max(len(total_errors), 1),
        }


class POSConfusionMatrix:
    """Matriz de confusión para POS tagging."""
    
    def __init__(self):
        self.matrix = defaultdict(Counter)
        self.pos_list = set()
    
    def add(self, gold: str, pred: str):
        self.matrix[gold][pred] += 1
        self.pos_list.add(gold)
        self.pos_list.add(pred)
    
    def get_confusion_pairs(self, top_n: int = 10) -> List[Tuple]:
        """Retorna los pares de confusión más frecuentes."""
        pairs = []
        for gold in self.matrix:
            for pred, count in self.matrix[gold].items():
                if gold != pred:
                    pairs.append((gold, pred, count))
        pairs.sort(key=lambda x: -x[2])
        return pairs[:top_n]
    
    def get_accuracy_by_pos(self) -> Dict:
        """Accuracy por clase POS."""
        results = {}
        for pos in sorted(self.pos_list):
            total = sum(self.matrix[pos].values())
            correct = self.matrix[pos].get(pos, 0)
            results[pos] = {
                'accuracy': correct / total if total > 0 else 0,
                'total': total,
                'correct': correct,
            }
        return results


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("PRAGMATICS-003: Análisis de Errores")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=5000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=1000)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Entrenar
    print(f"\n{'─' * 70}")
    print("Entrenando modelos para análisis")
    print(f"{'─' * 70}")
    
    analyzer = ErrorAnalyzer()
    t0 = time.time()
    analyzer.train(train_sents)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Analizar errores
    print(f"\n{'─' * 70}")
    print("Análisis de errores")
    print(f"{'─' * 70}")
    
    t0 = time.time()
    results = analyzer.analyze_corpus(test_sents)
    eval_time = time.time() - t0
    
    print(f"  Tokens analizados: {results['total_tokens']:,}")
    print(f"  Errores totales: {results['total_errors']:,}")
    print(f"  Tasa de error: {results['error_rate']:.2%}")
    print(f"  Error rate promedio por oración: {results['avg_sent_error_rate']:.2%}")
    print(f"  Tasa de cascada: {results['cascade_rate']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Errores morfológicos más frecuentes
    print(f"\n  Top 10 errores morfológicos (POS):")
    for error, count in list(results['morph_errors'].items())[:10]:
        pct = count / results['total_tokens']
        print(f"    {error:20s} {count:5,d} ({pct:.2%})")
    
    # Errores sintácticos más frecuentes
    print(f"\n  Top 10 errores sintácticos (deprel):")
    for error, count in list(results['syntax_errors'].items())[:10]:
        pct = count / results['total_tokens']
        print(f"    {error:30s} {count:5,d} ({pct:.2%})")
    
    # Matriz de confusión POS
    print(f"\n{'─' * 70}")
    print("Matriz de confusión POS")
    print(f"{'─' * 70}")
    
    confusion = POSConfusionMatrix()
    for sent in test_sents:
        tokens = [t for t in sent if not isinstance(t['id'], tuple)]
        for token in tokens:
            pred = analyzer.predict_pos(token['form'])
            confusion.add(token['upos'], pred)
    
    pos_accuracy = confusion.get_accuracy_by_pos()
    print(f"\n  Accuracy por clase POS:")
    for pos, data in sorted(pos_accuracy.items(), key=lambda x: -x[1]['total']):
        print(f"    {pos:8s} Acc={data['accuracy']:.2%} (n={data['total']:,})")
    
    print(f"\n  Top 10 pares de confusión:")
    for gold, pred, count in confusion.get_confusion_pairs(10):
        print(f"    {gold:8s} → {pred:8s} ({count:,} veces)")
    
    # Errores semánticos
    print(f"\n  Errores semánticos (NER):")
    for error, count in results['semantic_errors'].items():
        print(f"    {error:15s} {count:5,d}")
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  Error rate total: {results['error_rate']:.2%}")
    print(f"  Cascade rate: {results['cascade_rate']:.2%}")
    
    # Calcular accuracy por nivel
    morph_correct = results['total_tokens'] - sum(results['morph_errors'].values())
    morph_acc = morph_correct / results['total_tokens']
    
    print(f"  Morph accuracy: {morph_acc:.2%}")
    print(f"  Errores en cascada: {analyzer.cascade_errors:,}")
    
    data = {
        "experiment_id": "PRAGMATICS-003",
        "description": "Análisis sistemático de errores por nivel",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": {
            'total_tokens': results['total_tokens'],
            'total_errors': results['total_errors'],
            'error_rate': float(results['error_rate']),
            'avg_sent_error_rate': float(results['avg_sent_error_rate']),
            'cascade_rate': float(results['cascade_rate']),
            'morph_accuracy': float(morph_acc),
        },
        "top_morph_errors": results['morph_errors'],
        "top_syntax_errors": results['syntax_errors'],
        "semantic_errors": results['semantic_errors'],
        "pos_accuracy": {k: {'accuracy': float(v['accuracy']), 'total': v['total']} 
                         for k, v in pos_accuracy.items()},
    }
    
    output_path = os.path.join(output_dir, "pragmatics_003_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
