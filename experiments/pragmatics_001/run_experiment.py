"""
Experimento PRAGMATICS-001: Pipeline Integrado
===============================================

Objetivo: Integrar los modelos de las 4 fases en un pipeline secuencial
y evaluar cómo el rendimiento de cada nivel afecta al siguiente.

Enfoque:
- Nivel 1: POS tagging + lematización (Morfología)
- Nivel 2: Parsing de dependencias (Sintaxis)
- Nivel 3: SRL + NER (Semántica)
- Nivel 4: Entity tracking + segmentación (Discurso)

Métrica: Throughput del pipeline, accuracy por nivel, error propagation.
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


# ═══════════════════════════════════════════════════════════════
# NIVEL 1: MORFOLOGÍA
# ═══════════════════════════════════════════════════════════════

class MorphologyLevel:
    """Nivel morfológico: POS tagging + lematización."""
    
    def __init__(self):
        self.pos_model = {}  # suffix -> Counter(POS)
        self.lemma_model = {}  # suffix -> Counter(lemma_pattern)
    
    def train(self, sentences):
        """Entrena modelos de POS y lematización."""
        pos_correct = 0
        pos_total = 0
        
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                form = token['form']
                pos = token['upos']
                lemma = token['lemma']
                
                # Extraer sufijos (últimos 3-4 caracteres)
                for suf_len in [4, 3, 2]:
                    if len(form) > suf_len:
                        suffix = form[-suf_len:].lower()
                        key = f"{suf_len}:{suffix}"
                        if key not in self.pos_model:
                            self.pos_model[key] = Counter()
                        self.pos_model[key][pos] += 1
        
        print(f"  POS model: {len(self.pos_model):,} suffix entries")
    
    def predict_pos(self, form: str) -> str:
        """Predice POS tag para un token."""
        for suf_len in [4, 3, 2]:
            if len(form) > suf_len:
                suffix = form[-suf_len:].lower()
                key = f"{suf_len}:{suffix}"
                if key in self.pos_model:
                    return self.pos_model[key].most_common(1)[0][0]
        
        # Heurística por capitalización
        if form[0].isupper():
            return 'PROPN'
        if form.isdigit():
            return 'NUM'
        return 'NOUN'  # default
    
    def predict_lemma(self, form: str, pos: str) -> str:
        """Lematización simplificada."""
        # Heurísticas por POS
        if pos == 'VERB':
            for ending, stem in [('ando', 'ar'), ('iendo', 'er'), 
                                  ('ado', 'ar'), ('ido', 'er'),
                                  ('aba', 'ar'), ('ía', 'er')]:
                if form.lower().endswith(ending):
                    return form[:-(len(ending))] + stem
        elif pos == 'NOUN' or pos == 'ADJ':
            for ending in ['s', 'es']:
                if form.lower().endswith(ending) and len(form) > len(ending) + 2:
                    return form[:-len(ending)]
        return form.lower()
    
    def process(self, raw_tokens: List[str]) -> List[dict]:
        """Procesa tokens crudos → tokens con POS + lemma."""
        result = []
        for i, form in enumerate(raw_tokens):
            pos = self.predict_pos(form)
            lemma = self.predict_lemma(form, pos)
            result.append({
                'id': i + 1,
                'form': form,
                'upos': pos,
                'lemma': lemma,
            })
        return result


# ═══════════════════════════════════════════════════════════════
# NIVEL 2: SINTAXIS
# ═══════════════════════════════════════════════════════════════

class SyntaxLevel:
    """Nivel sintáctico: parsing de dependencias."""
    
    def __init__(self):
        self.pos_deprel = defaultdict(Counter)  # POS -> Counter(deprel)
        self.pos_head_pos = defaultdict(Counter)  # POS -> Counter(head_POS)
    
    def train(self, sentences):
        """Entrena modelo de dependencias."""
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            for token in tokens:
                pos = token['upos']
                deprel = token['deprel']
                head_id = token['head']
                
                self.pos_deprel[pos][deprel] += 1
                
                if head_id and head_id > 0:
                    head_token = tokens[head_id - 1] if head_id <= len(tokens) else None
                    if head_token:
                        self.pos_head_pos[pos][head_token['upos']] += 1
        
        print(f"  Syntax model: {len(self.pos_deprel)} POS → deprel mappings")
    
    def parse(self, morph_tokens: List[dict]) -> List[dict]:
        """Añade dependencias a tokens morfológicos."""
        result = []
        
        # Encontrar verbo principal (root)
        root_idx = 0
        for i, token in enumerate(morph_tokens):
            if token['upos'] == 'VERB':
                root_idx = i
                break
        
        for i, token in enumerate(morph_tokens):
            pos = token['upos']
            
            if i == root_idx:
                head = 0
                deprel = 'root'
            else:
                # Heurística: asignar head basado en POS
                deprel = self.pos_deprel[pos].most_common(1)[0][0] if pos in self.pos_deprel else 'dep'
                
                # Buscar head más cercano compatible
                head = root_idx + 1  # default: root
                best_dist = abs(i - root_idx)
                
                for j, other in enumerate(morph_tokens):
                    if j == i:
                        continue
                    if other['upos'] in self.pos_head_pos.get(pos, {}):
                        dist = abs(i - j)
                        if dist < best_dist:
                            best_dist = dist
                            head = j + 1
            
            token_copy = dict(token)
            token_copy['head'] = head
            token_copy['deprel'] = deprel
            result.append(token_copy)
        
        return result


# ═══════════════════════════════════════════════════════════════
# NIVEL 3: SEMÁNTICA
# ═══════════════════════════════════════════════════════════════

class SemanticsLevel:
    """Nivel semántico: SRL + NER."""
    
    def __init__(self):
        self.verb_roles = defaultdict(Counter)  # verb_lemma -> Counter(role)
        self.entities = set()  # known entity names
    
    def train(self, sentences):
        """Entrena modelos semánticos."""
        for sent in sentences:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            
            # SRL: mapear verbos a roles
            for token in tokens:
                if token['upos'] == 'VERB':
                    self.verb_roles[token['lemma']][token['deprel']] += 1
            
            # NER: colectar PROPNs
            for token in tokens:
                if token['upos'] == 'PROPN':
                    self.entities.add(token['form'])
        
        print(f"  Semantic model: {len(self.verb_roles):,} verbs, {len(self.entities):,} entities")
    
    def annotate(self, syntax_tokens: List[dict]) -> List[dict]:
        """Añade anotación semántica."""
        result = []
        for token in syntax_tokens:
            token_copy = dict(token)
            
            # SRL: asignar rol semántico
            if token['head'] and token['head'] > 0:
                head_idx = token['head'] - 1
                if head_idx < len(syntax_tokens) and syntax_tokens[head_idx]['upos'] == 'VERB':
                    verb = syntax_tokens[head_idx]['lemma']
                    deprel = token.get('deprel', 'dep')
                    token_copy['sem_role'] = deprel  # simplified
                else:
                    token_copy['sem_role'] = 'O'
            else:
                token_copy['sem_role'] = 'O'
            
            # NER: marcar entidades
            if token['upos'] == 'PROPN' or token['form'] in self.entities:
                token_copy['ner'] = 'B-ENT'
            else:
                token_copy['ner'] = 'O'
            
            result.append(token_copy)
        
        return result


# ═══════════════════════════════════════════════════════════════
# NIVEL 4: DISCURSO
# ═══════════════════════════════════════════════════════════════

class DiscourseLevel:
    """Nivel discursivo: entity tracking + segmentación."""
    
    def __init__(self):
        self.entity_registry = {}  # name -> entity_id
        self.next_id = 0
    
    def reset(self):
        self.entity_registry = {}
        self.next_id = 0
    
    def process_document(self, annotated_sentences: List[List[dict]]) -> dict:
        """Procesa documento completo: entity tracking + segmentación."""
        self.reset()
        
        entity_mentions = []
        topic_boundaries = []
        
        prev_content_words = set()
        
        for sent_idx, sent in enumerate(annotated_sentences):
            # Entity tracking
            for token in sent:
                if token.get('ner') != 'O':
                    name = token['form']
                    if name not in self.entity_registry:
                        self.entity_registry[name] = self.next_id
                        self.next_id += 1
                    entity_mentions.append({
                        'entity_id': self.entity_registry[name],
                        'mention': name,
                        'sentence': sent_idx,
                    })
            
            # Topic segmentation
            content_words = set(t['lemma'].lower() for t in sent 
                               if t['upos'] in ['NOUN', 'VERB', 'ADJ', 'PROPN'])
            if prev_content_words and content_words:
                overlap = len(prev_content_words & content_words) / len(prev_content_words | content_words)
                topic_boundaries.append(overlap < 0.02)
            else:
                topic_boundaries.append(False)
            prev_content_words = content_words
        
        return {
            'entities': len(self.entity_registry),
            'mentions': len(entity_mentions),
            'boundaries': sum(topic_boundaries),
        }


# ═══════════════════════════════════════════════════════════════
# PIPELINE INTEGRADO
# ═══════════════════════════════════════════════════════════════

class IntegratedPipeline:
    """Pipeline completo: Morfología → Sintaxis → Semántica → Discurso."""
    
    def __init__(self):
        self.morphology = MorphologyLevel()
        self.syntax = SyntaxLevel()
        self.semantics = SemanticsLevel()
        self.discourse = DiscourseLevel()
    
    def train(self, sentences):
        """Entrena todos los niveles."""
        print("  Nivel 1: Morfología")
        self.morphology.train(sentences)
        
        print("  Nivel 2: Sintaxis")
        self.syntax.train(sentences)
        
        print("  Nivel 3: Semántica")
        self.semantics.train(sentences)
    
    def process_sentence(self, raw_tokens: List[str]) -> List[dict]:
        """Procesa una oración a través del pipeline."""
        # Nivel 1
        morph_out = self.morphology.process(raw_tokens)
        # Nivel 2
        syntax_out = self.syntax.parse(morph_out)
        # Nivel 3
        sem_out = self.semantics.annotate(syntax_out)
        return sem_out
    
    def process_document(self, sentences: List[List[str]]) -> Tuple[List[List[dict]], dict]:
        """Procesa un documento completo."""
        annotated = []
        for sent_tokens in sentences:
            annotated.append(self.process_sentence(sent_tokens))
        
        discourse_info = self.discourse.process_document(annotated)
        return annotated, discourse_info
    
    def evaluate_propagation(self, gold_sentences, max_sents=500):
        """Evalúa propagación de errores a través de niveles."""
        level_results = {
            'morph_pos': {'correct': 0, 'total': 0},
            'syntax_uas': {'correct': 0, 'total': 0},
            'syntax_las': {'correct': 0, 'total': 0},
            'sem_ner': {'correct': 0, 'total': 0},
        }
        
        for sent in gold_sentences[:max_sents]:
            tokens = [t for t in sent if not isinstance(t['id'], tuple)]
            raw = [t['form'] for t in tokens]
            
            # Pipeline prediction
            predicted = self.process_sentence(raw)
            
            for pred, gold in zip(predicted, tokens):
                # Morph: POS
                level_results['morph_pos']['total'] += 1
                if pred['upos'] == gold['upos']:
                    level_results['morph_pos']['correct'] += 1
                
                # Syntax: UAS (head correcto)
                level_results['syntax_uas']['total'] += 1
                if pred['head'] == gold['head']:
                    level_results['syntax_uas']['correct'] += 1
                
                # Syntax: LAS (head + deprel correctos)
                level_results['syntax_las']['total'] += 1
                if pred['head'] == gold['head'] and pred['deprel'] == gold['deprel']:
                    level_results['syntax_las']['correct'] += 1
                
                # Sem: NER
                level_results['sem_ner']['total'] += 1
                gold_is_ent = gold['upos'] == 'PROPN'
                pred_is_ent = pred.get('ner', 'O') != 'O'
                if gold_is_ent == pred_is_ent:
                    level_results['sem_ner']['correct'] += 1
        
        # Calcular accuracies
        for key in level_results:
            c = level_results[key]['correct']
            t = level_results[key]['total']
            level_results[key]['accuracy'] = c / t if t > 0 else 0
        
        return level_results


def run_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("=" * 70)
    print("PRAGMATICS-001: Pipeline Integrado")
    print("=" * 70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando datos...")
    train_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu", max_sents=5000)
    test_sents = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu", max_sents=1000)
    print(f"  Train: {len(train_sents):,} | Test: {len(test_sents):,}")
    
    # Crear y entrenar pipeline
    print(f"\n{'─' * 70}")
    print("Entrenando pipeline integrado")
    print(f"{'─' * 70}")
    
    pipeline = IntegratedPipeline()
    t0 = time.time()
    pipeline.train(train_sents)
    train_time = time.time() - t0
    print(f"  Tiempo de entrenamiento: {train_time:.1f}s")
    
    # Evaluar propagación de errores
    print(f"\n{'─' * 70}")
    print("Evaluación: Propagación de errores por nivel")
    print(f"{'─' * 70}")
    
    t0 = time.time()
    results = pipeline.evaluate_propagation(test_sents, max_sents=500)
    eval_time = time.time() - t0
    
    print(f"\n  {'Nivel':<20s} {'Accuracy':>10s} {'Correctos':>10s} {'Total':>10s}")
    print(f"  {'─' * 50}")
    for key in ['morph_pos', 'syntax_uas', 'syntax_las', 'sem_ner']:
        r = results[key]
        label = key.replace('_', ' ').title()
        print(f"  {label:<20s} {r['accuracy']:>10.2%} {r['correct']:>10,d} {r['total']:>10,d}")
    
    print(f"\n  Tiempo de evaluación: {eval_time:.2f}s")
    
    # Medir throughput
    print(f"\n{'─' * 70}")
    print("Throughput del pipeline")
    print(f"{'─' * 70}")
    
    test_doc = [[t['form'] for t in sent if not isinstance(t['id'], tuple)] 
                for sent in test_sents[:100]]
    
    t0 = time.time()
    annotated, disc_info = pipeline.process_document(test_doc)
    process_time = time.time() - t0
    
    total_tokens = sum(len(s) for s in test_doc)
    throughput = total_tokens / process_time if process_time > 0 else 0
    
    print(f"  Oraciones procesadas: {len(test_doc)}")
    print(f"  Tokens procesados: {total_tokens:,}")
    print(f"  Tiempo: {process_time:.3f}s")
    print(f"  Throughput: {throughput:,.0f} tokens/s")
    print(f"  Entidades: {disc_info['entities']}")
    print(f"  Menciones: {disc_info['mentions']}")
    print(f"  Boundaries: {disc_info['boundaries']}")
    
    # Cascada de errores
    print(f"\n{'─' * 70}")
    print("Análisis de cascada de errores")
    print(f"{'─' * 70}")
    
    morph_acc = results['morph_pos']['accuracy']
    uas_acc = results['syntax_uas']['accuracy']
    las_acc = results['syntax_las']['accuracy']
    ner_acc = results['sem_ner']['accuracy']
    
    # Teórico: si cada nivel es independiente, accuracy cascada = producto
    cascade_theoretical = morph_acc * uas_acc * ner_acc
    
    print(f"  Morph POS:  {morph_acc:.2%}")
    print(f"  Syntax UAS: {uas_acc:.2%}")
    print(f"  Syntax LAS: {las_acc:.2%}")
    print(f"  Sem NER:    {ner_acc:.2%}")
    print(f"  Cascade teórico (producto): {cascade_theoretical:.2%}")
    
    # Resumen
    print(f"\n{'=' * 70}")
    print("RESUMEN")
    print(f"{'=' * 70}")
    avg_acc = np.mean([morph_acc, uas_acc, las_acc, ner_acc])
    print(f"  Pipeline accuracy promedio: {avg_acc:.2%}")
    print(f"  Throughput: {throughput:,.0f} tokens/s")
    print(f"  Niveles: 4 (Morph → Syntax → Sem → Discourse)")
    
    data = {
        "experiment_id": "PRAGMATICS-001",
        "description": "Pipeline integrado de 4 niveles",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "level_results": {k: {kk: float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else vv 
                              for kk, vv in v.items()} 
                          for k, v in results.items()},
        "throughput_tokens_per_sec": float(throughput),
        "discourse_info": disc_info,
        "cascade_theoretical": float(cascade_theoretical),
        "average_accuracy": float(avg_acc),
    }
    
    output_path = os.path.join(output_dir, "pragmatics_001_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
