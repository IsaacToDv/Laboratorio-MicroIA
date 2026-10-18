"""
GENERATION-002: Recuperación por Similitud Contextual

Este experimento evalúa la capacidad del sistema para recuperar
continuaciones relevantes dado un contexto parcial.

Métricas:
- Precisión en recuperación de siguientes palabras
- Diversidad de candidatas recuperadas
- Relevancia contextual
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from generation.text_generator import HDCSequenceMemory
from conllu import parse_incr


def load_corpus_sentences(corpus_path: str, max_sentences: int = 1000):
    """Carga oraciones del corpus."""
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, sent in enumerate(parse_incr(f)):
            if i >= max_sentences:
                break
            words = [token['form'] for token in sent 
                    if not isinstance(token['id'], tuple)]
            if len(words) >= 4:  # Necesitamos al menos 4 palabras para el test
                sentences.append(words)
    return sentences


def main():
    print("=" * 70)
    print("GENERATION-002: Recuperación por Similitud Contextual")
    print("=" * 70)
    print()
    
    # Configuración
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    dim = 10000
    max_sentences = 1000
    
    print(f"Configuración:")
    print(f"  - Dimensiones HDC: {dim}")
    print(f"  - Máximo de oraciones: {max_sentences}")
    print()
    
    # Cargar corpus
    print(f"Cargando corpus...")
    sentences = load_corpus_sentences(corpus_path, max_sentences)
    print(f"✓ Cargadas {len(sentences)} oraciones")
    print()
    
    # Dividir en train/test
    np.random.seed(42)
    indices = np.random.permutation(len(sentences))
    train_idx = indices[:int(0.8 * len(sentences))]
    test_idx = indices[int(0.8 * len(sentences)):]
    
    train_sentences = [sentences[i] for i in train_idx]
    test_sentences = [sentences[i] for i in test_idx]
    
    print(f"Split: {len(train_sentences)} train, {len(test_sentences)} test")
    print()
    
    # Construir memoria con train
    print("Construyendo memoria HDC con datos de entrenamiento...")
    memory = HDCSequenceMemory(dim=dim)
    for sent in train_sentences:
        memory.store_sequence(sent)
    
    print(f"✓ Memoria construida con {len(memory.sequence_memory)} secuencias")
    print()
    
    # Evaluar recuperación
    print("=" * 70)
    print("Evaluación de Recuperación Contextual")
    print("=" * 70)
    print()
    
    top_k_values = [1, 3, 5, 10]
    results_by_k = {}
    
    for top_k in top_k_values:
        print(f"Evaluando con top_k={top_k}...")
        
        correct = 0
        total = 0
        
        for sent in test_sentences:
            # Usar primeras 3 palabras como contexto
            context = sent[:3]
            target = sent[3] if len(sent) > 3 else None
            
            if target is None:
                continue
            
            # Buscar continuaciones
            similar = memory.find_similar(context, top_k=top_k)
            
            # Extraer candidatas (palabra siguiente en cada secuencia similar)
            candidates = []
            for sim, seq in similar:
                # Buscar dónde aparece el contexto
                for i in range(len(seq) - 3):
                    if all(seq[i+j].lower() == context[j].lower() for j in range(3)):
                        if i + 3 < len(seq):
                            candidates.append(seq[i + 3])
                        break
            
            # Verificar si el target está en las candidatas
            if target.lower() in [c.lower() for c in candidates]:
                correct += 1
            
            total += 1
        
        precision = correct / total if total > 0 else 0
        results_by_k[top_k] = {
            'precision': precision,
            'correct': correct,
            'total': total
        }
        
        print(f"  Precisión: {precision:.2%} ({correct}/{total})")
    
    print()
    
    # Análisis de diversidad
    print("=" * 70)
    print("Análisis de Diversidad de Candidatas")
    print("=" * 70)
    print()
    
    all_candidates = []
    for sent in test_sentences[:50]:  # Muestra de 50
        context = sent[:3]
        similar = memory.find_similar(context, top_k=10)
        
        for sim, seq in similar:
            for i in range(len(seq) - 3):
                if all(seq[i+j].lower() == context[j].lower() for j in range(3)):
                    if i + 3 < len(seq):
                        all_candidates.append(seq[i + 3])
                    break
    
    unique_candidates = len(set(all_candidates))
    avg_candidates = len(all_candidates) / min(50, len(test_sentences))
    
    print(f"Candidatas totales recuperadas: {len(all_candidates)}")
    print(f"Candidatas únicas: {unique_candidates}")
    print(f"Promedio de candidatas por query: {avg_candidates:.1f}")
    print()
    
    # Ejemplos de recuperación
    print("=" * 70)
    print("Ejemplos de Recuperación")
    print("=" * 70)
    print()
    
    example_queries = [
        ["El", "presidente", "firmó"],
        ["La", "economía", "creció"],
        ["En", "la", "ciudad"],
    ]
    
    for query in example_queries:
        print(f"Contexto: {' '.join(query)}")
        print("-" * 70)
        
        similar = memory.find_similar(query, top_k=5)
        
        candidates = []
        for sim, seq in similar:
            for i in range(len(seq) - len(query)):
                if all(seq[i+j].lower() == query[j].lower() for j in range(len(query))):
                    if i + len(query) < len(seq):
                        candidates.append(seq[i + len(query)])
                    break
        
        candidate_counts = Counter(candidates)
        print(f"Candidatas más frecuentes:")
        for word, count in candidate_counts.most_common(5):
            print(f"  - {word}: {count} veces")
        
        print()
    
    # Guardar resultados
    results = {
        'experiment_id': 'GENERATION-002',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dimensions': dim,
            'max_sentences': max_sentences,
            'corpus': corpus_path,
            'train_size': len(train_sentences),
            'test_size': len(test_sentences)
        },
        'metrics': {
            'precision_by_k': {str(k): v for k, v in results_by_k.items()},
            'unique_candidates': unique_candidates,
            'avg_candidates_per_query': avg_candidates
        }
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("Resultados guardados en:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
