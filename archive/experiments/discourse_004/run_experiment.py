"""
Experimento DISCOURSE-004: Entity Tracking
==========================================

Objetivo: Rastrear entidades a lo largo del discurso, manteniendo
un modelo de estado de cada entidad mencionada.

Enfoque:
- Identificar menciones de entidades (PROPN + patrones)
- Vincular menciones correferentes
- Mantener atributos actualizados de cada entidad

Métrica: Purity de clusters de entidades, tracking accuracy.
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
from conllu import parse_incr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


def load_conllu_documents(conllu_path: str, max_docs: int = None) -> List[List[List[dict]]]:
    documents = []
    current_doc = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'form': token['form'],
                    'lemma': token['lemma'],
                    'upos': token['upos'],
                    'feats': token.get('feats', {}) or {},
                    'deprel': token['deprel'],
                    'head': token['head'],
                })
            if sentence_data:
                current_doc.append(sentence_data)
            
            # Nuevo documento cada 10 oraciones (proxy)
            if len(current_doc) >= 10:
                documents.append(current_doc)
                current_doc = []
                if max_docs and len(documents) >= max_docs:
                    break
    
    if current_doc:
        documents.append(current_doc)
    
    return documents


class EntityTracker:
    """
    Rastreador de entidades a lo largo del discurso.
    
    Mantiene un registro de entidades activas con sus atributos
    y resuelve menciones subsecuentes.
    """
    
    def __init__(self):
        self.entities = {}  # entity_id -> {attrs}
        self.mention_to_entity = {}  # mention_key -> entity_id
        self.next_id = 0
        self.entity_counts = Counter()  # estadísticas
    
    def reset(self):
        """Reinicia para nuevo documento."""
        self.entities = {}
        self.mention_to_entity = {}
        self.next_id = 0
    
    def _extract_mentions(self, sentence: List[dict]) -> List[dict]:
        """Extrae menciones de entidades de una oración."""
        mentions = []
        i = 0
        while i < len(sentence):
            token = sentence[i]
            
            # PROPN: posible entidad nombrada
            if token['upos'] == 'PROPN':
                # Agrupar PROPNs consecutivos
                span = [token]
                j = i + 1
                while j < len(sentence) and sentence[j]['upos'] == 'PROPN':
                    span.append(sentence[j])
                    j += 1
                
                mention_text = ' '.join(t['form'] for t in span)
                mentions.append({
                    'text': mention_text,
                    'lemmas': ' '.join(t['lemma'] for t in span),
                    'start': i,
                    'end': j,
                    'type': 'named',
                    'tokens': span,
                })
                i = j
                continue
            
            # Pronombres personales
            if token['upos'] == 'PRON':
                feats = token.get('feats', {}) or {}
                person = feats.get('Person', '')
                if person in ['1', '2', '3']:
                    mentions.append({
                        'text': token['form'],
                        'lemmas': token['lemma'],
                        'start': i,
                        'end': i + 1,
                        'type': 'pronoun',
                        'person': person,
                        'gender': feats.get('Gender', ''),
                        'number': feats.get('Number', ''),
                        'tokens': [token],
                    })
            
            i += 1
        
        return mentions
    
    def _find_entity(self, mention: dict, sentence_idx: int) -> int:
        """
        Busca o crea una entidad para esta mención.
        Retorna entity_id.
        """
        # Para entidades nombradas: buscar por texto exacto o similar
        if mention['type'] == 'named':
            key = mention['text'].lower()
            
            # Búsqueda exacta
            if key in self.mention_to_entity:
                return self.mention_to_entity[key]
            
            # Búsqueda parcial (substring)
            for existing_key, eid in self.mention_to_entity.items():
                if key in existing_key or existing_key in key:
                    # Vincular a entidad existente
                    self.mention_to_entity[key] = eid
                    return eid
            
            # Nueva entidad
            eid = self.next_id
            self.next_id += 1
            self.entities[eid] = {
                'first_mention': mention['text'],
                'mentions': [mention],
                'first_sentence': sentence_idx,
                'last_sentence': sentence_idx,
                'type': 'named',
            }
            self.mention_to_entity[key] = eid
            return eid
        
        # Para pronombres: buscar entidad compatible más reciente
        elif mention['type'] == 'pronoun':
            gender = mention.get('gender', '')
            number = mention.get('number', '')
            person = mention.get('person', '3')
            
            best_entity = None
            best_recency = -1
            
            for eid, entity in self.entities.items():
                # Verificar compatibilidad
                if entity['type'] == 'pronoun_only' and entity.get('person') != person:
                    continue
                
                # Recencia
                recency = sentence_idx - entity['last_sentence']
                if recency > 5:  # Demasiado lejos
                    continue
                
                if recency < best_recency or best_recency == -1:
                    best_recency = recency
                    best_entity = eid
            
            if best_entity is not None:
                self.entities[best_entity]['mentions'].append(mention)
                self.entities[best_entity]['last_sentence'] = sentence_idx
                return best_entity
            
            # Crear nueva entidad para pronombre sin antecedente
            eid = self.next_id
            self.next_id += 1
            self.entities[eid] = {
                'first_mention': mention['text'],
                'mentions': [mention],
                'first_sentence': sentence_idx,
                'last_sentence': sentence_idx,
                'type': 'pronoun_only',
                'person': person,
                'gender': gender,
                'number': number,
            }
            return eid
        
        return -1
    
    def track_document(self, document: List[List[dict]]) -> Dict:
        """
        Rastrea entidades a lo largo de un documento.
        Retorna clusters de entidades.
        """
        self.reset()
        
        for sent_idx, sentence in enumerate(document):
            mentions = self._extract_mentions(sentence)
            
            for mention in mentions:
                eid = self._find_entity(mention, sent_idx)
                if eid >= 0:
                    self.entity_counts['total_mentions'] += 1
                    if mention['type'] == 'named':
                        self.entity_counts['named_mentions'] += 1
                    else:
                        self.entity_counts['pronoun_mentions'] += 1
        
        # Retornar clusters
        clusters = {}
        for eid, entity in self.entities.items():
            clusters[eid] = {
                'mention_count': len(entity['mentions']),
                'first_mention': entity['first_mention'],
                'span_sentences': entity['last_sentence'] - entity['first_sentence'] + 1,
                'type': entity['type'],
            }
        
        return clusters


def evaluate_tracking(documents: List[List[List[dict]]], tracker: EntityTracker) -> Dict:
    """Evalúa la calidad del tracking de entidades."""
    
    all_clusters = []
    total_mentions = 0
    total_entities = 0
    singletons = 0
    multi_mention = 0
    
    mention_counts = []
    span_lengths = []
    
    for doc in documents:
        clusters = tracker.track_document(doc)
        
        for eid, cluster in clusters.items():
            total_entities += 1
            total_mentions += cluster['mention_count']
            mention_counts.append(cluster['mention_count'])
            span_lengths.append(cluster['span_sentences'])
            
            if cluster['mention_count'] == 1:
                singletons += 1
            else:
                multi_mention += 1
    
    avg_mentions = np.mean(mention_counts) if mention_counts else 0
    avg_span = np.mean(span_lengths) if span_lengths else 0
    
    # Purity: proporción de menciones en clusters multi-mención
    multi_mentions = total_mentions - singletons
    purity = multi_mentions / total_mentions if total_mentions > 0 else 0
    
    return {
        'total_entities': total_entities,
        'total_mentions': total_mentions,
        'singletons': singletons,
        'multi_mention_entities': multi_mention,
        'avg_mentions_per_entity': float(avg_mentions),
        'avg_span_sentences': float(avg_span),
        'cluster_purity': purity,
        'multi_mention_ratio': multi_mention / total_entities if total_entities > 0 else 0,
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("DISCOURSE-004: Entity Tracking")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    
    print("\nCargando documentos...")
    train_docs = load_conllu_documents(corpus_path / "es_ancora-ud-train.conllu", max_docs=500)
    test_docs = load_conllu_documents(corpus_path / "es_ancora-ud-test.conllu", max_docs=100)
    print(f"  Train: {len(train_docs)} documentos")
    print(f"  Test:  {len(test_docs)} documentos")
    
    # Crear tracker
    tracker = EntityTracker()
    
    # Estadísticas de entrenamiento
    print(f"\n{'─'*70}")
    print("Análisis de entidades en train")
    print(f"{'─'*70}")
    
    t0 = time.time()
    for doc in train_docs:
        tracker.track_document(doc)
    train_time = time.time() - t0
    
    print(f"  Entidades encontradas: {tracker.next_id:,}")
    print(f"  Menciones totales: {tracker.entity_counts.get('total_mentions', 0):,}")
    print(f"  Nombradas: {tracker.entity_counts.get('named_mentions', 0):,}")
    print(f"  Pronominales: {tracker.entity_counts.get('pronoun_mentions', 0):,}")
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar en test
    print(f"\n{'─'*70}")
    print("Evaluación en test")
    print(f"{'─'*70}")
    
    t0 = time.time()
    results = evaluate_tracking(test_docs, tracker)
    eval_time = time.time() - t0
    
    print(f"  Entidades: {results['total_entities']:,}")
    print(f"  Menciones: {results['total_mentions']:,}")
    print(f"  Singletons: {results['singletons']:,}")
    print(f"  Multi-mención: {results['multi_mention_entities']:,}")
    print(f"  Promedio menciones/entidad: {results['avg_mentions_per_entity']:.2f}")
    print(f"  Promedio span (oraciones): {results['avg_span_sentences']:.2f}")
    print(f"  Cluster purity: {results['cluster_purity']:.2%}")
    print(f"  Ratio multi-mención: {results['multi_mention_ratio']:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    # Baseline: cada mención es una entidad nueva
    baseline_purity = 0.0  # Sin clusters
    
    print(f"\n{'─'*70}")
    print("Baselines")
    print(f"{'─'*70}")
    print(f"  Sin clustering (1 mención = 1 entidad): purity = 0.00%")
    print(f"  Modelo: purity = {results['cluster_purity']:.2%}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Entity Tracking: {results['total_entities']} entidades, "
          f"{results['total_mentions']} menciones")
    print(f"  Cluster purity: {results['cluster_purity']:.2%}")
    print(f"  Multi-mención: {results['multi_mention_ratio']:.2%}")
    
    data = {
        "experiment_id": "DISCOURSE-004",
        "description": "Entity tracking a lo largo del discurso",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    
    output_path = os.path.join(output_dir, "discourse_004_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
