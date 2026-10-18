"""
GENERATION-001: Memoria de Secuencias con HDC

Este experimento construye una memoria asociativa hiperdimensional
que almacena secuencias del corpus como vectores individuales.

Métricas:
- Capacidad de almacenamiento
- Similitud entre secuencias relacionadas
- Recuperación de secuencias por query
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

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
            if len(words) >= 3:
                sentences.append(words)
    return sentences


def main():
    print("=" * 70)
    print("GENERATION-001: Memoria de Secuencias con HDC")
    print("=" * 70)
    print()
    
    # Configuración
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    dim = 10000
    max_sentences = 1000
    
    # Crear memoria
    print(f"Configuración:")
    print(f"  - Dimensiones HDC: {dim}")
    print(f"  - Máximo de oraciones: {max_sentences}")
    print()
    
    memory = HDCSequenceMemory(dim=dim)
    
    # Cargar corpus
    print(f"Cargando corpus desde {corpus_path}...")
    sentences = load_corpus_sentences(corpus_path, max_sentences)
    print(f"✓ Cargadas {len(sentences)} oraciones")
    print()
    
    # Almacenar secuencias
    print("Almacenando secuencias en memoria HDC...")
    for sent in sentences:
        memory.store_sequence(sent)
    
    print(f"✓ {len(memory.sequence_memory)} secuencias almacenadas")
    print(f"✓ {len(memory.item_memory)} palabras en item memory")
    print()
    
    # Estadísticas de la memoria
    print("Estadísticas de la memoria:")
    print(f"  - Tamaño de cada vector: {dim} floats")
    print(f"  - Memoria total: {len(memory.sequence_memory) * dim * 4 / 1024 / 1024:.2f} MB")
    print(f"  - Promedio de palabras por secuencia: {np.mean([len(s['words']) for s in memory.sequence_memory]):.1f}")
    print()
    
    # Test de similitud
    print("=" * 70)
    print("Test de Similitud entre Secuencias")
    print("=" * 70)
    print()
    
    test_queries = [
        ["El", "presidente"],
        ["La", "economía"],
        ["En", "la", "ciudad"],
    ]
    
    for query in test_queries:
        print(f"Query: {' '.join(query)}")
        print("-" * 70)
        
        similar = memory.find_similar(query, top_k=5)
        
        for i, (sim, seq) in enumerate(similar, 1):
            seq_str = ' '.join(seq[:10])  # Mostrar primeras 10 palabras
            if len(seq) > 10:
                seq_str += "..."
            print(f"{i}. [sim={sim:.3f}] {seq_str}")
        
        print()
    
    # Test de recuperación de secuencias completas
    print("=" * 70)
    print("Test de Recuperación de Secuencias Completas")
    print("=" * 70)
    print()
    
    # Tomar 5 secuencias aleatorias y verificar que se pueden recuperar
    test_indices = np.random.choice(len(sentences), 5, replace=False)
    
    recovery_success = 0
    for idx in test_indices:
        original = sentences[idx]
        query = original[:3]  # Usar primeras 3 palabras como query
        
        similar = memory.find_similar(query, top_k=10)
        
        # Verificar si la secuencia original está en los resultados
        found = any(
            ' '.join(seq[:3]) == ' '.join(original[:3])
            for _, seq in similar
        )
        
        if found:
            recovery_success += 1
            print(f"✓ Query: {' '.join(query)}")
            print(f"  Original: {' '.join(original[:8])}...")
        else:
            print(f"✗ Query: {' '.join(query)}")
            print(f"  No se encontró la secuencia original")
        
        print()
    
    recovery_rate = recovery_success / len(test_indices) * 100
    print(f"Tasa de recuperación: {recovery_rate:.1f}%")
    print()
    
    # Guardar resultados
    results = {
        'experiment_id': 'GENERATION-001',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dimensions': dim,
            'max_sentences': max_sentences,
            'corpus': corpus_path
        },
        'metrics': {
            'sequences_stored': len(memory.sequence_memory),
            'vocabulary_size': len(memory.item_memory),
            'memory_size_mb': len(memory.sequence_memory) * dim * 4 / 1024 / 1024,
            'avg_words_per_sequence': float(np.mean([len(s['words']) for s in memory.sequence_memory])),
            'recovery_rate': recovery_rate
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
