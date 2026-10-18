"""
GENERATION-003: Generación de Texto con FEP

Este experimento demuestra la generación completa de texto usando:
1. Memoria HDC para almacenar secuencias
2. Recuperación por similitud contextual
3. Selección de siguiente palabra con Free Energy Principle (FEP)

Métricas:
- Calidad de texto generado (evaluación manual)
- Longitud promedio de generaciones
- Diversidad léxica
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from generation.text_generator import TextGenerator


def main():
    print("=" * 70)
    print("GENERATION-003: Generación de Texto con FEP")
    print("=" * 70)
    print()
    
    # Configuración
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    dim = 10000
    max_sentences = 2000
    
    print(f"Configuración:")
    print(f"  - Dimensiones HDC: {dim}")
    print(f"  - Máximo de oraciones para entrenamiento: {max_sentences}")
    print()
    
    # Crear y entrenar generador
    print("Creando generador de texto...")
    generator = TextGenerator(dim=dim)
    
    print("Entrenando generador...")
    generator.train_from_corpus(corpus_path, max_sentences=max_sentences)
    print()
    
    # Generar ejemplos
    print("=" * 70)
    print("Generación de Texto")
    print("=" * 70)
    print()
    
    prompts = [
        ("El presidente", 15, 0.5),
        ("La economía", 15, 0.5),
        ("En la ciudad", 15, 0.5),
        ("Los científicos", 15, 0.5),
        ("El gobierno", 15, 0.7),
        ("La empresa", 15, 0.7),
    ]
    
    generated_texts = []
    vocabulary_used = set()
    
    for prompt, max_words, temp in prompts:
        print(f"Prompt: '{prompt}' (max={max_words}, temp={temp})")
        print("-" * 70)
        
        # Generar múltiples veces para ver variabilidad
        for i in range(3):
            text = generator.generate(prompt, max_words=max_words, temperature=temp)
            generated_texts.append(text)
            vocabulary_used.update(text.split())
            
            print(f"{i+1}. {text}")
        
        print()
    
    # Estadísticas de las generaciones
    print("=" * 70)
    print("Estadísticas de Generación")
    print("=" * 70)
    print()
    
    lengths = [len(text.split()) for text in generated_texts]
    avg_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    print(f"Total de textos generados: {len(generated_texts)}")
    print(f"Longitud promedio: {avg_length:.1f} ± {std_length:.1f} palabras")
    print(f"Longitud mínima: {min(lengths)} palabras")
    print(f"Longitud máxima: {max(lengths)} palabras")
    print(f"Vocabulario usado en generaciones: {len(vocabulary_used)} palabras")
    print()
    
    # Análisis de diversidad
    print("=" * 70)
    print("Análisis de Diversidad")
    print("=" * 70)
    print()
    
    # Type-Token Ratio (TTR) - medida de diversidad léxica
    all_words = []
    for text in generated_texts:
        all_words.extend(text.split())
    
    total_tokens = len(all_words)
    unique_types = len(set(all_words))
    ttr = unique_types / total_tokens if total_tokens > 0 else 0
    
    print(f"Total de tokens generados: {total_tokens}")
    print(f"Tipos únicos (vocabulario): {unique_types}")
    print(f"Type-Token Ratio (TTR): {ttr:.3f}")
    print(f"  (TTR más alto = más diversidad léxica)")
    print()
    
    # Palabras más frecuentes
    from collections import Counter
    word_freq = Counter(all_words)
    
    print("Palabras más frecuentes en generaciones:")
    for word, count in word_freq.most_common(10):
        print(f"  - {word}: {count} veces ({count/total_tokens*100:.1f}%)")
    print()
    
    # Ejemplos de completado de oraciones
    print("=" * 70)
    print("Completado de Oraciones")
    print("=" * 70)
    print()
    
    completion_prompts = [
        "El acuerdo",
        "La decisión",
        "Durante el",
        "Según el",
    ]
    
    for prompt in completion_prompts:
        completed = generator.complete_sentence(prompt)
        print(f"Prompt: {prompt}")
        print(f"Completado: {completed}")
        print()
    
    # Guardar resultados
    results = {
        'experiment_id': 'GENERATION-003',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dimensions': dim,
            'max_sentences': max_sentences,
            'corpus': corpus_path
        },
        'metrics': {
            'total_generated': len(generated_texts),
            'avg_length': float(avg_length),
            'std_length': float(std_length),
            'min_length': int(min(lengths)),
            'max_length': int(max(lengths)),
            'vocabulary_size': len(vocabulary_used),
            'type_token_ratio': float(ttr),
            'total_tokens': total_tokens,
            'unique_types': unique_types
        },
        'examples': {
            'generated_texts': generated_texts[:10],  # Primeros 10 ejemplos
            'top_words': word_freq.most_common(10)
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
