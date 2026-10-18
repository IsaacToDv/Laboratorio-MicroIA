"""
Experimento SEMANTICS-004: Similitud Semántica
===============================================

Objetivo: Calcular similitud semántica entre palabras usando HDC.

Enfoque:
- Construir "representaciones semánticas" basadas en contexto (distributional semantics)
- Para cada palabra, crear un hipervector que capture su contexto de aparición
- Calcular similitud entre pares de palabras usando sus representaciones

Métrica: Correlación de Spearman con juicios humanos (simulados).

Dado que no tenemos un dataset de similitud semántica para español,
evaluamos la capacidad de agrupar palabras por categoría semántica.
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

try:
    from hdc import HyperdimensionalVectors, ItemMemory
    HDC_AVAILABLE = True
except ImportError:
    HDC_AVAILABLE = False


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
                    'head': token['head'],
                    'deprel': token['deprel'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


class SemanticSpace:
    """
    Espacio semántico basado en contextos de palabras.
    
    Construye representaciones distribucionales: cada palabra se representa
    por el conjunto de contextos en que aparece.
    """
    
    def __init__(self, dimension: int = 10000, seed: int = 42):
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Representaciones de palabras (lemma → vector de contexto)
        self.word_vectors = {}
        
        # Vocabulario de contextos (lemma + deprel → vector aleatorio)
        self.context_vectors = {}
        
        # Frecuencia de palabras
        self.word_freq = Counter()
    
    def _get_context_vector(self, context_key: str) -> np.ndarray:
        """Obtiene o crea un vector para un contexto."""
        if context_key not in self.context_vectors:
            self.context_vectors[context_key] = self.rng.choice(
                [-1, 1], size=self.dimension
            ).astype(np.int8)
        return self.context_vectors[context_key]
    
    def build(self, sentences: List[List[dict]], min_freq: int = 5):
        """
        Construye el espacio semántico.
        
        Para cada palabra, acumula los vectores de sus contextos.
        """
        # Paso 1: Contar frecuencias
        for sent in sentences:
            for token in sent:
                if token['upos'] in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    self.word_freq[token['lemma'].lower()] += 1
        
        # Paso 2: Construir representaciones
        for sent in sentences:
            for i, token in enumerate(sent):
                lemma = token['lemma'].lower()
                
                if self.word_freq[lemma] < min_freq:
                    continue
                if token['upos'] not in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    continue
                
                # Inicializar vector si no existe
                if lemma not in self.word_vectors:
                    self.word_vectors[lemma] = np.zeros(self.dimension, dtype=np.int32)
                
                # Contextos: palabras vecinas con su deprel
                if i > 0:
                    left = sent[i-1]
                    ctx_key = f"L:{left['lemma'].lower()}_{left['upos']}"
                    self.word_vectors[lemma] += self._get_context_vector(ctx_key)
                
                if i < len(sent) - 1:
                    right = sent[i+1]
                    ctx_key = f"R:{right['lemma'].lower()}_{right['upos']}"
                    self.word_vectors[lemma] += self._get_context_vector(ctx_key)
                
                # Contexto: cabeza (si existe)
                head_idx = token['head']
                if head_idx > 0 and head_idx - 1 < len(sent):
                    head = sent[head_idx - 1]
                    ctx_key = f"H:{head['lemma'].lower()}_{token['deprel']}"
                    self.word_vectors[lemma] += self._get_context_vector(ctx_key)
        
        # Normalizar a bipolar
        for lemma in self.word_vectors:
            self.word_vectors[lemma] = np.where(
                self.word_vectors[lemma] >= 0, 1, -1
            ).astype(np.int8)
        
        print(f"  Espacio semántico: {len(self.word_vectors):,} palabras")
        print(f"  {len(self.context_vectors):,} contextos únicos")
    
    def similarity(self, word1: str, word2: str) -> float:
        """Calcula similitud coseno entre dos palabras."""
        w1 = word1.lower()
        w2 = word2.lower()
        
        if w1 not in self.word_vectors or w2 not in self.word_vectors:
            return 0.0
        
        v1 = self.word_vectors[w1].astype(np.float64)
        v2 = self.word_vectors[w2].astype(np.float64)
        
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot / (norm1 * norm2))
    
    def nearest_neighbors(self, word: str, k: int = 10) -> List[Tuple[str, float]]:
        """Encuentra los k vecinos más cercanos."""
        w = word.lower()
        if w not in self.word_vectors:
            return []
        
        similarities = []
        for other_word in self.word_vectors:
            if other_word == w:
                continue
            sim = self.similarity(w, other_word)
            similarities.append((other_word, sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:k]


def create_test_categories() -> Dict[str, List[str]]:
    """
    Crea categorías semánticas de prueba para evaluar clustering.
    
    Basado en palabras comunes del español que deberían agruparse.
    """
    return {
        'animales': ['perro', 'gato', 'caballo', 'vaca', 'toro', 'cerdo',
                     'ave', 'pez', 'animal', 'lobo'],
        'profesiones': ['presidente', 'ministro', 'director', 'jugador',
                       'entrenador', 'periodista', 'doctor', 'autor'],
        'lugares': ['ciudad', 'país', 'estado', 'pueblo', 'zona', 'región',
                   'provincia', 'territorio'],
        'tiempo': ['año', 'mes', 'día', 'hora', 'semana', 'momento',
                  'siglo', 'época'],
        'economía': ['millón', 'dólar', 'euro', 'mercado', 'empresa',
                    'banco', 'dinero', 'precio'],
    }


def evaluate_clustering(space: SemanticSpace, 
                        categories: Dict[str, List[str]]) -> dict:
    """
    Evalúa la calidad del clustering semántico.
    
    Para cada palabra de una categoría, verifica si sus vecinos más cercanos
    pertenecen a la misma categoría.
    """
    results = {}
    
    total_correct = 0
    total_evaluated = 0
    
    for category, words in categories.items():
        correct = 0
        evaluated = 0
        
        for word in words:
            if word.lower() not in space.word_vectors:
                continue
            
            neighbors = space.nearest_neighbors(word, k=10)
            neighbor_words = [w for w, _ in neighbors]
            
            # ¿Cuántos vecinos pertenecen a la misma categoría?
            same_category = sum(1 for w in neighbor_words 
                              if w in [x.lower() for x in words])
            
            correct += same_category
            evaluated += len(neighbors)
            total_correct += same_category
            total_evaluated += len(neighbors)
        
        precision = correct / evaluated if evaluated > 0 else 0
        results[category] = {
            'precision': precision,
            'correct': correct,
            'evaluated': evaluated,
            'words_found': sum(1 for w in words if w.lower() in space.word_vectors),
            'total_words': len(words),
        }
    
    results['overall'] = {
        'precision': total_correct / total_evaluated if total_evaluated > 0 else 0,
        'correct': total_correct,
        'evaluated': total_evaluated,
    }
    
    return results


def evaluate_similarity_pairs(space: SemanticSpace) -> dict:
    """
    Evalúa pares de palabras que deberían ser similares o diferentes.
    """
    # Pares que deberían ser similares
    similar_pairs = [
        ('perro', 'gato'),
        ('presidente', 'ministro'),
        ('ciudad', 'país'),
        ('año', 'mes'),
        ('millón', 'dólar'),
        ('empresa', 'banco'),
        ('día', 'semana'),
    ]
    
    # Pares que deberían ser diferentes
    different_pairs = [
        ('perro', 'presidente'),
        ('ciudad', 'millón'),
        ('año', 'gato'),
        ('empresa', 'día'),
        ('país', 'dólar'),
    ]
    
    similar_scores = []
    for w1, w2 in similar_pairs:
        sim = space.similarity(w1, w2)
        similar_scores.append({'pair': f"{w1}-{w2}", 'similarity': sim})
    
    different_scores = []
    for w1, w2 in different_pairs:
        sim = space.similarity(w1, w2)
        different_scores.append({'pair': f"{w1}-{w2}", 'similarity': sim})
    
    avg_similar = np.mean([s['similarity'] for s in similar_scores]) if similar_scores else 0
    avg_different = np.mean([s['similarity'] for s in different_scores]) if different_scores else 0
    
    return {
        'similar_pairs': similar_scores,
        'different_pairs': different_scores,
        'avg_similar': float(avg_similar),
        'avg_different': float(avg_different),
        'separation': float(avg_similar - avg_different),
    }


def run_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SEMANTICS-004: Similitud Semántica")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    print(f"HDC disponible: {HDC_AVAILABLE}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    print(f"  Train: {len(train_s):,} oraciones")
    
    # Construir espacio semántico
    print(f"\n{'─'*70}")
    print("Construyendo espacio semántico")
    print(f"{'─'*70}")
    
    space = SemanticSpace(dimension=10000, seed=42)
    t0 = time.time()
    space.build(train_s, min_freq=5)
    build_time = time.time() - t0
    print(f"  Tiempo: {build_time:.1f}s")
    
    # Evaluar clustering por categorías
    print(f"\n{'─'*70}")
    print("Evaluación de clustering semántico")
    print(f"{'─'*70}")
    
    categories = create_test_categories()
    clustering = evaluate_clustering(space, categories)
    
    print(f"\n  Resultados por categoría:")
    for cat, data in clustering.items():
        if cat == 'overall':
            continue
        print(f"    {cat:<15} precision={data['precision']:.2%} "
              f"({data['words_found']}/{data['total_words']} palabras encontradas)")
    
    overall = clustering['overall']
    print(f"\n  Overall: precision={overall['precision']:.2%}")
    
    # Evaluar pares de similitud
    print(f"\n{'─'*70}")
    print("Evaluación de pares de similitud")
    print(f"{'─'*70}")
    
    pairs_result = evaluate_similarity_pairs(space)
    
    print(f"\n  Pares similares:")
    for p in pairs_result['similar_pairs']:
        print(f"    {p['pair']:<25} sim={p['similarity']:.4f}")
    
    print(f"\n  Pares diferentes:")
    for p in pairs_result['different_pairs']:
        print(f"    {p['pair']:<25} sim={p['similarity']:.4f}")
    
    print(f"\n  Separación promedio: {pairs_result['separation']:.4f}")
    print(f"  (similar={pairs_result['avg_similar']:.4f} vs "
          f"diferente={pairs_result['avg_different']:.4f})")
    
    # Mostrar vecinos más cercanos de algunas palabras
    print(f"\n{'─'*70}")
    print("Vecinos más cercanos (ejemplos)")
    print(f"{'─'*70}")
    
    for word in ['perro', 'presidente', 'ciudad', 'empresa']:
        neighbors = space.nearest_neighbors(word, k=5)
        if neighbors:
            n_str = ", ".join(f"{w}({s:.3f})" for w, s in neighbors)
            print(f"  {word}: {n_str}")
        else:
            print(f"  {word}: no encontrado")
    
    # Resumen
    print(f"\n{'='*70}")
    print("RESUMEN")
    print(f"{'='*70}")
    print(f"  Palabras en espacio: {len(space.word_vectors):,}")
    print(f"  Clustering precision: {overall['precision']:.2%}")
    print(f"  Separación similar/diferente: {pairs_result['separation']:.4f}")
    
    # Guardar resultados
    data = {
        "experiment_id": "SEMANTICS-004",
        "description": "Similitud semántica (distributional semantics con HDC)",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "n_words": len(space.word_vectors),
        "n_contexts": len(space.context_vectors),
        "build_time_s": build_time,
        "results": {
            "clustering": clustering,
            "similarity_pairs": pairs_result,
        },
    }
    
    output_path = os.path.join(output_dir, "semantics_004_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
