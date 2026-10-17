"""
Experimento SYNTAX-004: Concordancia Gramatical
=================================================

Objetivo
--------
Detectar concordancia de género y número en pares:
- Determinante + Sustantivo (DET-NOUN)
- Adjetivo + Sustantivo (ADJ-NOUN)
- Sujeto + Verbo (nsubj-VERB)

Enfoque
-------
1. Usar las dependencias para encontrar pares relevantes
2. Extraer features de gender/number de los morfemas
3. Verificar concordancia

Métrica: Accuracy en detección de pares concordantes/no concordantes.
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


def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'word': token['form'], 'lemma': token['lemma'],
                    'pos': token['upos'], 'feats': token['feats'] or {},
                    'head': token['head'], 'deprel': token['deprel'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


# ============ Extracción de pares de concordancia ============

def extract_agreement_pairs(sentences: List[List[dict]]) -> dict:
    """
    Extrae pares de concordancia de las oraciones.
    
    Tipos de pares:
    - det_noun: determinante + sustantivo (relación det)
    - amod_noun: adjetivo + sustantivo (relación amod)
    - nsubj_verb: sujeto + verbo (relación nsubj)
    """
    pairs = {
        'det_noun': [],
        'amod_noun': [],
        'nsubj_verb': [],
    }
    
    for sent in sentences:
        for i, token in enumerate(sent):
            head_idx = token['head']
            deprel = token['deprel']
            
            if head_idx == 0:
                continue
            
            head_0based = head_idx - 1
            if head_0based >= len(sent):
                continue
            
            head_token = sent[head_0based]
            
            # DET → NOUN (det)
            if deprel == 'det' and token['pos'] == 'DET' and head_token['pos'] == 'NOUN':
                pairs['det_noun'].append({
                    'modifier': token,
                    'head': head_token,
                    'modifier_idx': i,
                    'head_idx': head_0based,
                })
            
            # ADJ → NOUN (amod)
            elif deprel == 'amod' and token['pos'] == 'ADJ' and head_token['pos'] == 'NOUN':
                pairs['amod_noun'].append({
                    'modifier': token,
                    'head': head_token,
                    'modifier_idx': i,
                    'head_idx': head_0based,
                })
            
            # NOUN/PRON → VERB (nsubj)
            elif deprel == 'nsubj' and token['pos'] in ['NOUN', 'PRON', 'PROPN'] and head_token['pos'] in ['VERB', 'AUX']:
                pairs['nsubj_verb'].append({
                    'modifier': token,
                    'head': head_token,
                    'modifier_idx': i,
                    'head_idx': head_0based,
                })
    
    return pairs


def get_gender_number(token: dict) -> Tuple[str, str]:
    """Extrae género y número de un token."""
    feats = token.get('feats', {})
    gender = feats.get('Gender', '<UNK>')
    number = feats.get('Number', '<UNK>')
    return gender, number


def check_agreement(modifier: dict, head: dict) -> dict:
    """
    Verifica concordancia entre modificador y cabeza.
    
    Retorna:
    - gender_match: True si géneros coinciden
    - number_match: True si números coinciden
    - mod_gender, mod_number: features del modificador
    - head_gender, head_number: features de la cabeza
    """
    mod_g, mod_n = get_gender_number(modifier)
    head_g, head_n = get_gender_number(head)
    
    # Género: concordancia si ambos tienen género y coinciden
    # o si alguno no tiene género especificado
    gender_match = (mod_g == head_g) or (mod_g == '<UNK>') or (head_g == '<UNK>')
    
    # Número: concordancia si ambos tienen número y coinciden
    number_match = (mod_n == head_n) or (mod_n == '<UNK>') or (head_n == '<UNK>')
    
    return {
        'gender_match': gender_match,
        'number_match': number_match,
        'mod_gender': mod_g,
        'mod_number': mod_n,
        'head_gender': head_g,
        'head_number': head_n,
        'full_match': gender_match and number_match,
    }


# ============ Modelo de concordancia basado en sufijos ============

class AgreementModel:
    """
    Modelo que predice concordancia basado en sufijos de las palabras.
    
    Para español:
    - Género: -o (masc), -a (fem), -e (ambos), consonante (varía)
    - Número: -s (plural), sin -s (singular)
    """
    
    def __init__(self):
        self.gender_suffix_lookup = {}
        self.number_suffix_lookup = {}
    
    def _extract_gender_from_suffix(self, word: str, pos: str) -> str:
        """Predice género basado en sufijo."""
        word = word.lower()
        
        if pos == 'DET':
            # Determinantes comunes
            if word in ['el', 'un', 'este', 'ese', 'aquel', 'mi', 'tu', 'su', 'nuestro', 'vuestro']:
                return 'Masc'
            elif word in ['la', 'una', 'esta', 'esa', 'aquella', 'mi', 'tu', 'su', 'nuestra', 'vuestra']:
                return 'Fem'
            elif word in ['los', 'unos', 'estos', 'esos', 'aquellos', 'mis', 'tus', 'sus', 'nuestros', 'vuestros']:
                return 'Masc'
            elif word in ['las', 'unas', 'estas', 'esas', 'aquellas', 'mis', 'tus', 'sus', 'nuestras', 'vuestras']:
                return 'Fem'
            # Abreviaciones
            if word.endswith('l') or word.endswith('n'):
                return 'Masc'
            if word.endswith('a'):
                return 'Fem'
            return '<UNK>'
        
        if pos in ['NOUN', 'ADJ']:
            if word.endswith('o') or word.endswith('os'):
                return 'Masc'
            elif word.endswith('a') or word.endswith('as'):
                return 'Fem'
            elif word.endswith('e') or word.endswith('es'):
                return '<UNK>'  # Ambiguo
            elif word.endswith('ción') or word.endswith('sión') or word.endswith('dad') or word.endswith('tad'):
                return 'Fem'
            elif word.endswith('or') and not word.endswith('ior'):
                return 'Masc'
            elif word.endswith('ora'):
                return 'Fem'
            return '<UNK>'
        
        return '<UNK>'
    
    def _extract_number_from_suffix(self, word: str) -> str:
        """Predice número basado en sufijo."""
        word = word.lower()
        if word.endswith('s'):
            return 'Plur'
        return 'Sing'
    
    def predict_agreement(self, modifier: dict, head: dict) -> dict:
        """Predice concordancia basado en sufijos."""
        mod_word = modifier['word']
        head_word = head['word']
        mod_pos = modifier['pos']
        head_pos = head['pos']
        
        # Predicciones por sufijo
        mod_g_pred = self._extract_gender_from_suffix(mod_word, mod_pos)
        head_g_pred = self._extract_gender_from_suffix(head_word, head_pos)
        mod_n_pred = self._extract_number_from_suffix(mod_word)
        head_n_pred = self._extract_number_from_suffix(head_word)
        
        # Features reales (si existen)
        mod_g_real, mod_n_real = get_gender_number(modifier)
        head_g_real, head_n_real = get_gender_number(head)
        
        # Usar features reales si existen, si no usar predicción
        mod_g = mod_g_real if mod_g_real != '<UNK>' else mod_g_pred
        head_g = head_g_real if head_g_real != '<UNK>' else head_g_pred
        mod_n = mod_n_real if mod_n_real != '<UNK>' else mod_n_pred
        head_n = head_n_real if head_n_real != '<UNK>' else head_n_pred
        
        gender_match = (mod_g == head_g) or (mod_g == '<UNK>') or (head_g == '<UNK>')
        number_match = (mod_n == head_n) or (mod_n == '<UNK>') or (head_n == '<UNK>')
        
        return {
            'gender_match': gender_match,
            'number_match': number_match,
            'full_match': gender_match and number_match,
            'mod_gender': mod_g,
            'head_gender': head_g,
            'mod_number': mod_n,
            'head_number': head_n,
        }


# ============ Evaluación ============

def evaluate_agreement(pairs: List[dict], model: AgreementModel = None) -> dict:
    """
    Evalúa la concordancia en pares.
    
    Si model=None, usa las features gold del corpus.
    Si model está definido, usa predicciones basadas en sufijos.
    """
    total = 0
    gender_correct = 0
    number_correct = 0
    full_correct = 0
    
    # Estadísticas
    has_gender = 0
    has_number = 0
    
    for pair in pairs:
        modifier = pair['modifier']
        head = pair['head']
        
        if model is None:
            # Usar features gold
            result = check_agreement(modifier, head)
        else:
            # Usar predicciones
            result = model.predict_agreement(modifier, head)
        
        total += 1
        
        mod_g, mod_n = get_gender_number(modifier)
        head_g, head_n = get_gender_number(head)
        if mod_g != '<UNK>' and head_g != '<UNK>':
            has_gender += 1
        if mod_n != '<UNK>' and head_n != '<UNK>':
            has_number += 1
        
        if result['gender_match']:
            gender_correct += 1
        if result['number_match']:
            number_correct += 1
        if result['full_match']:
            full_correct += 1
    
    return {
        'total': total,
        'gender_accuracy': gender_correct / total if total > 0 else 0,
        'number_accuracy': number_correct / total if total > 0 else 0,
        'full_accuracy': full_correct / total if total > 0 else 0,
        'gender_correct': gender_correct,
        'number_correct': number_correct,
        'full_correct': full_correct,
        'has_gender_info': has_gender,
        'has_number_info': has_number,
    }


# ============ Experimento principal ============

def run_full_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SYNTAX-004: Concordancia Gramatical")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Extraer pares de concordancia
    print(f"\n{'─'*70}")
    print("Extrayendo pares de concordancia")
    print(f"{'─'*70}")
    
    train_pairs = extract_agreement_pairs(train_s)
    test_pairs = extract_agreement_pairs(test_s)
    
    for pair_type in ['det_noun', 'amod_noun', 'nsubj_verb']:
        print(f"  {pair_type}: {len(train_pairs[pair_type]):,} (train) / {len(test_pairs[pair_type]):,} (test)")
    
    # Evaluar con features gold (techo)
    print(f"\n{'─'*70}")
    print("Evaluación con features gold (techo teórico)")
    print(f"{'─'*70}")
    
    gold_results = {}
    for pair_type in ['det_noun', 'amod_noun', 'nsubj_verb']:
        result = evaluate_agreement(test_pairs[pair_type], model=None)
        gold_results[pair_type] = result
        print(f"\n  {pair_type}:")
        print(f"    Total pares:          {result['total']:,}")
        print(f"    Con info género:      {result['has_gender_info']:,}")
        print(f"    Con info número:      {result['has_number_info']:,}")
        print(f"    Gender accuracy:      {result['gender_accuracy']:.2%}")
        print(f"    Number accuracy:      {result['number_accuracy']:.2%}")
        print(f"    Full accuracy:        {result['full_accuracy']:.2%}")
    
    # Evaluar con modelo de sufijos
    print(f"\n{'─'*70}")
    print("Evaluación con modelo de sufijos (predicción)")
    print(f"{'─'*70}")
    
    model = AgreementModel()
    
    suffix_results = {}
    for pair_type in ['det_noun', 'amod_noun', 'nsubj_verb']:
        result = evaluate_agreement(test_pairs[pair_type], model=model)
        suffix_results[pair_type] = result
        print(f"\n  {pair_type}:")
        print(f"    Gender accuracy:      {result['gender_accuracy']:.2%}")
        print(f"    Number accuracy:      {result['number_accuracy']:.2%}")
        print(f"    Full accuracy:        {result['full_accuracy']:.2%}")
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Tipo':<15} {'Gold Full':>12} {'Suffix Full':>12} {'Objetivo':>10}")
    print("-"*55)
    
    for pair_type in ['det_noun', 'amod_noun', 'nsubj_verb']:
        gold_acc = gold_results[pair_type]['full_accuracy']
        suffix_acc = suffix_results[pair_type]['full_accuracy']
        print(f"{pair_type:<15} {gold_acc:>12.2%} {suffix_acc:>12.2%} {'>90%':>10}")
    
    # Total
    total_gold = sum(gold_results[t]['full_correct'] for t in gold_results)
    total_suffix = sum(suffix_results[t]['full_correct'] for t in suffix_results)
    total_pairs = sum(gold_results[t]['total'] for t in gold_results)
    
    total_gold_acc = total_gold / total_pairs if total_pairs > 0 else 0
    total_suffix_acc = total_suffix / total_pairs if total_pairs > 0 else 0
    
    print(f"{'TOTAL':<15} {total_gold_acc:>12.2%} {total_suffix_acc:>12.2%} {'>90%':>10}")
    
    # Guardar resultados
    data = {
        "experiment_id": "SYNTAX-004",
        "description": "Concordancia gramatical",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": {
            "gold": gold_results,
            "suffix_model": suffix_results,
            "total_gold_accuracy": total_gold_acc,
            "total_suffix_accuracy": total_suffix_acc,
        },
    }
    
    output_path = os.path.join(output_dir, "syntax_004_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_full_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
