"""
Experimento DISCOURSE-003: Relaciones entre Oraciones
=====================================================

Objetivo: Clasificar relaciones retóricas entre pares de oraciones
(causa, consecuencia, contraste, elaboración, etc.)

Enfoque:
- Extraer marcadores discursivos (conectores)
- Features léxicas y estructurales
- Clasificar tipo de relación

Métrica: Accuracy y F1 por tipo de relación.
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


# Conectores discursivos del español organizados por tipo de relación
DISCOURSE_MARKERS = {
    'causa': [
        'porque', 'pues', 'ya que', 'puesto que', 'debido a', 'por',
        'como', 'dado que', 'puesto que', 'gracias a',
    ],
    'consecuencia': [
        'por tanto', 'por lo tanto', 'así que', 'entonces', 'en consecuencia',
        'por consiguiente', 'de modo que', 'de manera que', 'luego',
    ],
    'contraste': [
        'pero', 'sin embargo', 'no obstante', 'aunque', 'si bien',
        'mientras que', 'en cambio', 'al contrario', 'sino', 'aun así',
    ],
    'adición': [
        'además', 'también', 'asimismo', 'igualmente', 'encima',
        'por otra parte', 'por otro lado', 'aparte',
    ],
    'elaboración': [
        'es decir', 'o sea', 'en otras palabras', 'dicho de otro modo',
        'por ejemplo', 'en efecto', 'de hecho', 'concretamente',
    ],
    'temporal': [
        'cuando', 'mientras', 'después', 'antes', 'luego', 'entonces',
        'al mismo tiempo', 'simultáneamente', 'posteriormente',
    ],
    'condición': [
        'si', 'en caso de que', 'a condición de que', 'siempre que',
        'con tal de que',
    ],
}


def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'form': token['form'],
                    'lemma': token['lemma'],
                    'upos': token['upos'],
                    'deprel': token['deprel'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


class DiscourseRelationClassifier:
    """
    Clasificador de relaciones discursivas basado en marcadores y features léxicas.
    """
    
    def __init__(self):
        self.marker_weights = defaultdict(lambda: defaultdict(float))
        self.marker_positions = defaultdict(list)  # posición típica del marcador
    
    def _find_markers(self, sentence: List[dict]) -> List[Tuple[str, str, int]]:
        """Encuentra marcadores discursivos en la oración."""
        found = []
        forms = [t['form'].lower() for t in sentence]
        lemmas = [t['lemma'].lower() for t in sentence]
        
        for rel_type, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                marker_tokens = marker.split()
                # Buscar secuencia de tokens
                for i in range(len(forms) - len(marker_tokens) + 1):
                    if all(forms[i+j] == marker_tokens[j] or 
                           lemmas[i+j] == marker_tokens[j] 
                           for j in range(len(marker_tokens))):
                        found.append((rel_type, marker, i))
        
        return found
    
    def _extract_features(self, sent1: List[dict], sent2: List[dict]) -> Dict:
        """Extrae features del par de oraciones."""
        features = {}
        
        # Marcadores en sent2 (la que inicia la relación)
        markers_sent2 = self._find_markers(sent2)
        
        # Tipo de marcador dominante
        marker_types = Counter(r for r, _, _ in markers_sent2)
        features['dominant_marker_type'] = marker_types.most_common(1)[0][0] if marker_types else 'none'
        features['has_marker'] = len(markers_sent2) > 0
        
        # Posición del marcador (inicio, medio, final)
        if markers_sent2:
            pos = markers_sent2[0][2]
            n = len(sent2)
            if pos <= 2:
                features['marker_position'] = 'initial'
            elif pos >= n - 3:
                features['marker_position'] = 'final'
            else:
                features['marker_position'] = 'middle'
        else:
            features['marker_position'] = 'none'
        
        # Overlap léxico entre oraciones
        words1 = set(t['lemma'].lower() for t in sent1 if t['upos'] in ['NOUN','VERB','ADJ'])
        words2 = set(t['lemma'].lower() for t in sent2 if t['upos'] in ['NOUN','VERB','ADJ'])
        if words1 and words2:
            features['lexical_overlap'] = len(words1 & words2) / len(words1 | words2)
        else:
            features['lexical_overlap'] = 0.0
        
        # Longitud relativa
        features['length_ratio'] = len(sent2) / max(len(sent1), 1)
        
        return features
    
    def train(self, sentences: List[List[dict]]):
        """Entrena el clasificador contando marcadores y relaciones."""
        pair_count = 0
        
        for i in range(len(sentences) - 1):
            markers = self._find_markers(sentences[i+1])
            for rel_type, marker, pos in markers:
                self.marker_weights[rel_type][marker] += 1
                self.marker_positions[rel_type].append(pos / max(len(sentences[i+1]), 1))
            pair_count += 1
        
        # Normalizar pesos
        for rel_type in self.marker_weights:
            total = sum(self.marker_weights[rel_type].values())
            for marker in self.marker_weights[rel_type]:
                self.marker_weights[rel_type][marker] /= total
        
        # Estadísticas
        rel_counts = Counter()
        for sent_idx in range(1, len(sentences)):
            markers = self._find_markers(sentences[sent_idx])
            for rel_type, _, _ in markers:
                rel_counts[rel_type] += 1
        
        print(f"  Pares analizados: {pair_count:,}")
        print(f"  Relaciones detectadas:")
        for rel, count in rel_counts.most_common():
            print(f"    {rel:15s} {count:5d} ({count/pair_count:.1%})")
        
        self.relation_distribution = rel_counts
    
    def predict_pair(self, sent1: List[dict], sent2: List[dict]) -> str:
        """Predice la relación entre dos oraciones."""
        markers = self._find_markers(sent2)
        
        if not markers:
            # Sin marcador: elaboración por defecto
            return 'elaboración'
        
        # Votar por tipo de marcador más frecuente
        type_scores = Counter()
        for rel_type, marker, pos in markers:
            type_scores[rel_type] += self.marker_weights[rel_type].get(marker, 0.5)
        
        return type_scores.most_common(1)[0][0]
    
    def predict_all(self, sentences: List[List[dict]]) -> List[str]:
        """Predice relaciones para todos los pares consecutivos."""
        predictions = []
        for i in range(len(sentences) - 1):
            pred = self.predict_pair(sentences[i], sentences[i+1])
            predictions.append(pred)
        return predictions


def create_gold_labels(sentences: List[List[dict]]) -> List[str]:
    """
    Crea gold estándar basado en marcadores explícitos.
    Solo oraciones donde hay marcadores claros.
    """
    gold = []
    
    for i in range(len(sentences) - 1):
        sent2 = sentences[i + 1]
        forms = [t['form'].lower() for t in sent2]
        lemmas = [t['lemma'].lower() for t in sent2]
        
        # Solo considerar marcadores al inicio de la oración (posición 0-2)
        found_relation = None
        
        for rel_type, markers in DISCOURSE_MARKERS.items():
            for marker in markers:
                marker_tokens = marker.split()
                for start in range(min(3, len(forms) - len(marker_tokens) + 1)):
                    if all(forms[start+j] == marker_tokens[j] or 
                           lemmas[start+j] == marker_tokens[j]
                           for j in range(len(marker_tokens))):
                        found_relation = rel_type
                        break
                if found_relation:
                    break
            if found_relation:
                break
        
        if found_relation:
            gold.append(found_relation)
        else:
            gold.append('elaboración')
    
    return gold


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("DISCOURSE-003: Relaciones entre Oraciones")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Entrenar clasificador
    print(f"\n{'─'*70}")
    print("Entrenando clasificador de relaciones")
    print(f"{'─'*70}")
    
    classifier = DiscourseRelationClassifier()
    t0 = time.time()
    classifier.train(train_s)
    train_time = time.time() - t0
    print(f"  Tiempo: {train_time:.1f}s")
    
    # Evaluar
    print(f"\n{'─'*70}")
    print("Evaluación")
    print(f"{'─'*70}")
    
    t0 = time.time()
    predictions = classifier.predict_all(test_s)
    gold = create_gold_labels(test_s)
    eval_time = time.time() - t0
    
    # Calcular métricas
    from collections import Counter
    correct = sum(1 for p, g in zip(predictions, gold) if p == g)
    accuracy = correct / len(gold) if gold else 0
    
    # F1 por clase
    pred_counts = Counter(predictions)
    gold_counts = Counter(gold)
    
    f1_scores = {}
    for rel_type in set(list(pred_counts.keys()) + list(gold_counts.keys())):
        tp = sum(1 for p, g in zip(predictions, gold) if p == rel_type and g == rel_type)
        fp = sum(1 for p, g in zip(predictions, gold) if p == rel_type and g != rel_type)
        fn = sum(1 for p, g in zip(predictions, gold) if p != rel_type and g == rel_type)
        
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores[rel_type] = {'precision': p, 'recall': r, 'f1': f1, 
                                'support': gold_counts.get(rel_type, 0)}
    
    # F1 macro ponderado
    total_support = sum(v['support'] for v in f1_scores.values())
    weighted_f1 = sum(v['f1'] * v['support'] for v in f1_scores.values()) / total_support if total_support > 0 else 0
    
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  F1 ponderado: {weighted_f1:.2%}")
    print(f"  Tiempo: {eval_time:.2f}s")
    
    print(f"\n  Detalle por tipo:")
    for rel, scores in sorted(f1_scores.items(), key=lambda x: -x[1]['support']):
        print(f"    {rel:15s} P={scores['precision']:.2%} R={scores['recall']:.2%} "
              f"F1={scores['f1']:.2%} (n={scores['support']})")
    
    # Baseline: siempre predecir la clase mayoritaria
    majority = gold_counts.most_common(1)[0][0]
    baseline_acc = gold_counts[majority] / len(gold)
    
    print(f"\n{'─'*70}")
    print("Baselines")
    print(f"{'─'*70}")
    print(f"  Clase mayoritaria ({majority}): {baseline_acc:.2%}")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Accuracy: {accuracy:.2%}")
    print(f"  F1 ponderado: {weighted_f1:.2%}")
    print(f"  Baseline: {baseline_acc:.2%}")
    print(f"  Mejora: {(accuracy - baseline_acc):.2%}")
    
    data = {
        "experiment_id": "DISCOURSE-003",
        "description": "Clasificación de relaciones discursivas entre oraciones",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "baseline_accuracy": baseline_acc,
        "f1_per_class": f1_scores,
    }
    
    output_path = os.path.join(output_dir, "discourse_003_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
