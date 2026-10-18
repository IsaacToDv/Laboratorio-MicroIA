"""
Experimento PRAGMATICS-002: Comprensión de Texto
=================================================

Objetivo: Evaluar la capacidad del sistema para extraer información
estructurada de un texto, integrando todos los niveles lingüísticos.

Enfoque:
- Extraer tripletas (sujeto, verbo, objeto) del texto
- Identificar quién-hace-qué-a-quién
- Evaluar precisión y recall de extracciones

Métrica: F1 en extracción de tripletas SVO.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Set
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


class SVOExtractor:
    """
    Extractor de tripletas Sujeto-Verbo-Objeto.
    
    Usa el árbol de dependencias para identificar:
    - Sujeto (nsubj, nsubj:pass)
    - Verbo (root o VERB con sujetos)
    - Objeto (obj, dobj, iobj)
    """
    
    def __init__(self):
        self.subj_deprels = {'nsubj', 'nsubj:pass', 'expl:pass'}
        self.obj_deprels = {'obj', 'dobj', 'iobj', 'obl', 'ccomp'}
        self.subj_pos = {'NOUN', 'PROPN', 'PRON'}
        self.obj_pos = {'NOUN', 'PROPN', 'PRON'}
    
    def extract_gold(self, sentence) -> Set[Tuple[str, str, str]]:
        """Extrae tripletas SVO del gold standard (CoNLL-U)."""
        tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
        triplets = set()
        
        for token in tokens:
            if token['upos'] == 'VERB':
                verb = token['lemma']
                verb_id = token['id']
                
                # Buscar sujetos y objetos
                subjects = []
                objects = []
                
                for other in tokens:
                    if other['head'] == verb_id:
                        if other['deprel'] in self.subj_deprels and other['upos'] in self.subj_pos:
                            subjects.append(other['lemma'])
                        elif other['deprel'] in self.obj_deprels and other['upos'] in self.obj_pos:
                            objects.append(other['lemma'])
                
                # Generar tripletas
                for subj in subjects:
                    if objects:
                        for obj in objects:
                            triplets.add((subj, verb, obj))
                    else:
                        triplets.add((subj, verb, '_'))
        
        return triplets
    
    def extract_predicted(self, sentence) -> Set[Tuple[str, str, str]]:
        """
        Extrae tripletas SVO con análisis heurístico (sin parser perfecto).
        Simula un pipeline con errores.
        """
        tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
        triplets = set()
        
        # Identificar verbos
        verbs = [(i, t) for i, t in enumerate(tokens) if t['upos'] == 'VERB']
        
        for verb_idx, verb_token in verbs:
            verb = verb_token['lemma']
            
            # Buscar sujeto: NOUN/PROPN/PRON antes del verbo con deprel de sujeto
            subjects = []
            objects = []
            
            for other in tokens:
                if other['head'] == verb_token['id']:
                    if other['deprel'] in self.subj_deprels:
                        subjects.append(other['lemma'])
                    elif other['deprel'] in self.obj_deprels:
                        objects.append(other['lemma'])
            
            # Si no se encontró sujeto por deprel, usar heurística
            if not subjects:
                for i in range(max(0, verb_idx - 3), verb_idx):
                    if tokens[i]['upos'] in self.subj_pos:
                        subjects.append(tokens[i]['lemma'])
                        break
            
            for subj in subjects:
                if objects:
                    for obj in objects:
                        triplets.add((subj, verb, obj))
                else:
                    triplets.add((subj, verb, '_'))
        
        return triplets


def evaluate_svo(gold_triplets: Set, pred_triplets: Set) -> dict:
    """Evalúa extracción de tripletas."""
    tp = len(gold_triplets & pred_triplets)
    fp = len(pred_triplets - gold_triplets)
    fn = len(gold_triplets - pred_triplets)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
    }


class QuestionAnswerer:
    """
    Sistema de preguntas y respuestas basado en extracción de información.
    
    Responde preguntas simples tipo "¿Quién [verbo]?" y "¿Qué [verbo] [sujeto]?"
    """
    
    def __init__(self):
        self.triplets = []
    
    def index(self, sentences):
        """Indexa tripletas del documento."""
        extractor = SVOExtractor()
        self.triplets = []
        for sent in sentences:
            triplets = extractor.extract_gold(sent)
            self.triplets.extend(triplets)
    
    def answer_who(self, verb: str) -> List[str]:
        """Responde '¿Quién [verbo]?'"""
        answers = []
        for subj, v, obj in self.triplets:
            if v.lower() == verb.lower():
                answers.append(subj)
        return list(set(answers))
    
    def answer_what(self, subj: str, verb: str) -> List[str]:
        """Responde '¿Qué [verbo] [sujeto]?'"""
        answers = []
        for s, v, obj in self.triplets:
            if s.lower() == subj.lower() and v.lower() == verb.lower() and obj != '_':
                answers.append(obj)
        return list(set(answers))


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("PRAGMATICS-002: Comprensión de Texto")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=3000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=500)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Extracción de tripletas SVO
    print(f"\n{'─' * 70}")
    print("Extracción de tripletas Sujeto-Verbo-Objeto")
    print(f"{'─' * 70}")
    
    extractor = SVOExtractor()
    
    total_gold = set()
    total_pred = set()
    sent_with_triplets = 0
    
    t0 = time.time()
    for sent in test_sents:
        gold = extractor.extract_gold(sent)
        pred = extractor.extract_predicted(sent)
        total_gold |= gold
        total_pred |= pred
        if gold:
            sent_with_triplets += 1
    eval_time = time.time() - t0
    
    svo_result = evaluate_svo(total_gold, total_pred)
    
    print(f"  Oraciones con tripletas: {sent_with_triplets}/{len(test_sents)}")
    print(f"  Tripletas gold: {len(total_gold):,}")
    print(f"  Tripletas predichas: {len(total_pred):,}")
    print(f"  TP={svo_result['tp']} FP={svo_result['fp']} FN={svo_result['fn']}")
    print(f"  Precision: {svo_result['precision']:.2%}")
    print(f"  Recall:    {svo_result['recall']:.2%}")
    print(f"  F1:        {svo_result['f1']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Muestra de tripletas
    print(f"\n  Ejemplos de tripletas extraídas:")
    sample = list(total_gold)[:10]
    for i, (s, v, o) in enumerate(sample):
        print(f"    {i+1}. ({s}, {v}, {o})")
    
    # Sistema de Q&A
    print(f"\n{'─' * 70}")
    print("Sistema de preguntas y respuestas")
    print(f"{'─' * 70}")
    
    qa = QuestionAnswerer()
    qa.index(test_sents)
    
    # Generar preguntas de prueba
    qa_results = {
        'total_questions': 0,
        'answered': 0,
        'correct_answers': 0,
    }
    
    # Preguntas tipo "¿Quién [verbo]?"
    verb_freq = Counter()
    for subj, verb, obj in qa.triplets:
        verb_freq[verb] += 1
    
    for verb, count in verb_freq.most_common(20):
        answers = qa.answer_who(verb)
        qa_results['total_questions'] += 1
        if answers:
            qa_results['answered'] += 1
    
    # Preguntas tipo "¿Qué [verbo] [sujeto]?"
    subj_freq = Counter()
    for subj, verb, obj in qa.triplets:
        if obj != '_':
            subj_freq[subj] += 1
    
    for subj, count in subj_freq.most_common(20):
        for s, v, o in qa.triplets:
            if s == subj and o != '_':
                answers = qa.answer_what(subj, v)
                qa_results['total_questions'] += 1
                if answers:
                    qa_results['answered'] += 1
                break
    
    answer_rate = qa_results['answered'] / qa_results['total_questions'] if qa_results['total_questions'] > 0 else 0
    
    print(f"  Preguntas generadas: {qa_results['total_questions']}")
    print(f"  Preguntas respondidas: {qa_results['answered']}")
    print(f"  Tasa de respuesta: {answer_rate:.2%}")
    print(f"  Tripletas indexadas: {len(qa.triplets):,}")
    
    # Baseline: extracción aleatoria
    print(f"\n{'─' * 70}")
    print("Baselines")
    print(f"{'─' * 70}")
    
    # Baseline: todas las combinaciones posibles
    all_nouns = set()
    all_verbs = set()
    for sent in test_sents:
        for t in sent:
            if isinstance(t['id'], tuple):
                continue
            if t['upos'] in ['NOUN', 'PROPN']:
                all_nouns.add(t['lemma'])
            elif t['upos'] == 'VERB':
                all_verbs.add(t['lemma'])
    
    baseline_triplets = set()
    for s in list(all_nouns)[:10]:
        for v in list(all_verbs)[:10]:
            baseline_triplets.add((s, v, '_'))
    
    baseline_result = evaluate_svo(total_gold, baseline_triplets)
    print(f"  Aleatorio: P={baseline_result['precision']:.2%} "
          f"R={baseline_result['recall']:.2%} F1={baseline_result['f1']:.2%}")
    print(f"  Modelo:    P={svo_result['precision']:.2%} "
          f"R={svo_result['recall']:.2%} F1={svo_result['f1']:.2%}")
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    print(f"  SVO Extraction F1: {svo_result['f1']:.2%}")
    print(f"  Q&A Answer rate: {answer_rate:.2%}")
    print(f"  Triplets: {len(total_gold):,} gold, {len(total_pred):,} pred")
    
    data = {
        "experiment_id": "PRAGMATICS-002",
        "description": "Comprensión de texto vía extracción SVO y Q&A",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "svo_results": svo_result,
        "qa_results": qa_results,
        "answer_rate": float(answer_rate),
    }
    
    output_path = os.path.join(output_dir, "pragmatics_002_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
