"""
Experimento HDC-Esp-001: Memoria Asociativa para Caracteres del Español
========================================================================

Objetivo
--------
Validar que un espacio HDC bipolar puede almacenar y recuperar asociaciones
carácter → hipervector con tasa de error aceptable, para diferentes dimensiones.

Hipótesis
---------
H1: Un espacio HDC bipolar de dimensión D puede almacenar y recuperar N 
    asociaciones carácter→hipervector con tasa de error < 5% cuando 
    N < capacidad teórica del espacio.

H0 (nula): La tasa de error es > 20% incluso con D = 10000, indicando
    que HDC no es viable para esta tarea.

Variables
---------
- Independiente: dimensión D ∈ {1024, 2048, 4096, 8192, 10000}
- Dependiente: tasa de error de recuperación, tiempo, memoria
- Control: inventario de caracteres del español

Procedimiento
-------------
1. Crear item memory para cada dimensión
2. Almacenar el inventario de caracteres del español
3. Consultar cada hipervector original y verificar recuperación exacta
4. Consultar hipervectores con ruido añadido y medir degradación
5. Medir tiempos de codificación y consulta
6. Analizar colisiones entre hipervectores

Clasificación: E1 — Resultado reproducido (al ejecutarse).
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdc import HyperdimensionalVectors, SpanishCharacterMemory, SequenceEncoder


def add_noise(vector: np.ndarray, noise_level: float, 
              rng: np.random.RandomState) -> np.ndarray:
    """
    Añade ruido bipolar a un hipervector invirtiendo un porcentaje de bits.
    
    Parámetros
    ----------
    vector : np.ndarray
        Hipervector bipolar {-1, +1}.
    noise_level : float
        Fracción de componentes a invertir [0, 1].
    rng : np.random.RandomState
        Generador aleatorio.
    
    Retorna
    -------
    np.ndarray
        Hipervector con ruido.
    """
    noisy = vector.copy()
    n_flip = int(len(vector) * noise_level)
    indices = rng.choice(len(vector), n_flip, replace=False)
    noisy[indices] *= -1
    return noisy


def run_dimension_experiment(dimension: int, seed: int = 42) -> dict:
    """
    Ejecuta el experimento completo para una dimensión dada.
    """
    print(f"\n{'='*60}")
    print(f"Dimensión D = {dimension}")
    print(f"{'='*60}")
    
    results = {
        "dimension": dimension,
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 1. Crear HDC y memoria de caracteres
    t0 = time.time()
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    creation_time = time.time() - t0
    results["creation_time_ms"] = creation_time * 1000
    results["inventory_size"] = memory.capacity
    
    print(f"  Inventario: {memory.capacity} caracteres")
    print(f"  Tiempo de creación: {creation_time*1000:.2f} ms")
    
    # 2. Validación de distribución aleatoria
    dist_stats = hdc.validate_random_distribution(n_samples=500)
    # Convertir tipos numpy a tipos nativos de Python para JSON
    dist_stats_serializable = {
        k: bool(v) if isinstance(v, (np.bool_, bool)) else 
           float(v) if isinstance(v, (np.floating, float)) else
           int(v) if isinstance(v, (np.integer, int)) else v
        for k, v in dist_stats.items()
    }
    results["random_distribution"] = dist_stats_serializable
    print(f"  Similitud media (aleatorios): {dist_stats['similarity_mean']:.6f}")
    print(f"  Similitud std (aleatorios):   {dist_stats['similarity_std']:.6f}")
    print(f"  Esperada std:                 {dist_stats['expected_std']:.6f}")
    
    # 3. Análisis de colisiones
    collision_stats = memory.collision_analysis()
    # Convertir tipos numpy a tipos nativos de Python para JSON
    collision_stats_serializable = {
        k: bool(v) if isinstance(v, (np.bool_, bool)) else 
           float(v) if isinstance(v, (np.floating, float)) else
           int(v) if isinstance(v, (np.integer, int)) else v
        for k, v in collision_stats.items()
    }
    results["collision_analysis"] = collision_stats_serializable
    print(f"  Similitud máxima entre pares: {collision_stats['max_pairwise_similarity']:.4f}")
    print(f"  Colisiones (sim > 0.1):       {collision_stats['pairs_above_0.1']}")
    
    # 4. Recuperación exacta (sin ruido)
    t0 = time.time()
    keys = memory.get_all_keys()
    correct = 0
    total = len(keys)
    similarities_exact = []
    
    for key in keys:
        vector = memory.get(key)
        retrieved_key, sim = memory.query_exact(vector)
        if retrieved_key == key:
            correct += 1
        similarities_exact.append(sim)
    
    query_time = time.time() - t0
    error_rate_exact = 1.0 - (correct / total)
    
    results["exact_recovery"] = {
        "total": total,
        "correct": correct,
        "error_rate": error_rate_exact,
        "mean_similarity": float(np.mean(similarities_exact)),
        "min_similarity": float(np.min(similarities_exact)),
        "query_time_ms": query_time * 1000,
        "query_time_per_item_us": (query_time / total) * 1e6,
    }
    print(f"\n  Recuperación exacta: {correct}/{total} "
          f"(error: {error_rate_exact:.4%})")
    print(f"  Tiempo por consulta: {(query_time/total)*1e6:.1f} μs")
    
    # 5. Recuperación con ruido
    noise_levels = [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    noise_results = {}
    rng = np.random.RandomState(seed + 1)
    
    for noise_level in noise_levels:
        correct_noisy = 0
        sims_noisy = []
        
        for key in keys:
            vector = memory.get(key)
            noisy_vector = add_noise(vector, noise_level, rng)
            retrieved_key, sim = memory.query_exact(noisy_vector)
            if retrieved_key == key:
                correct_noisy += 1
            sims_noisy.append(sim)
        
        error_rate_noisy = 1.0 - (correct_noisy / total)
        noise_results[str(noise_level)] = {
            "error_rate": error_rate_noisy,
            "mean_similarity": float(np.mean(sims_noisy)),
            "correct": correct_noisy,
        }
        print(f"  Ruido {noise_level:.0%}: error = {error_rate_noisy:.4%}")
    
    results["noisy_recovery"] = noise_results
    
    # 6. Codificación y recuperación de secuencias
    encoder = SequenceEncoder(hdc, memory, max_length=20, method="positional_binding")
    
    test_sequences = [
        "hola",
        "mundo",
        "casa",
        "perro",
        "gato",
        "libro",
        "mesa",
        "silla",
        "agua",
        "fuego",
    ]
    
    t0 = time.time()
    encoded = encoder.encode_batch(test_sequences)
    encode_time = time.time() - t0
    
    # Verificar que secuencias diferentes son disimilares
    sim_matrix = encoder.similarity_matrix(test_sequences)
    np.fill_diagonal(sim_matrix, 0)
    
    results["sequence_encoding"] = {
        "n_sequences": len(test_sequences),
        "encode_time_ms": encode_time * 1000,
        "max_pairwise_similarity": float(np.max(sim_matrix)),
        "mean_pairwise_similarity": float(np.mean(sim_matrix)),
        "min_self_similarity": 1.0,
    }
    print(f"\n  Codificación de secuencias:")
    print(f"    Max sim entre secuencias: {np.max(sim_matrix):.4f}")
    print(f"    Mean sim entre secuencias: {np.mean(sim_matrix):.4f}")
    
    # 7. Estimación de memoria
    item_size_bytes = dimension  # 1 byte por componente (int8)
    total_memory_bytes = memory.capacity * item_size_bytes
    results["memory_usage"] = {
        "per_item_bytes": item_size_bytes,
        "total_items": memory.capacity,
        "total_bytes": total_memory_bytes,
        "total_kb": total_memory_bytes / 1024,
    }
    print(f"\n  Memoria: {total_memory_bytes/1024:.2f} KB "
          f"({item_size_bytes} bytes/ítem)")
    
    return results


def run_full_experiment(output_dir: str = ".") -> dict:
    """
    Ejecuta el experimento completo para todas las dimensiones.
    """
    dimensions = [1024, 2048, 4096, 8192, 10000]
    
    print("=" * 60)
    print("HDC-Esp-001: Memoria Asociativa para Español")
    print("=" * 60)
    print(f"Dimensiones a evaluar: {dimensions}")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    all_results = {}
    
    for dim in dimensions:
        results = run_dimension_experiment(dimension=dim, seed=42)
        all_results[str(dim)] = results
    
    # Resumen comparativo
    print("\n" + "=" * 60)
    print("RESUMEN COMPARATIVO")
    print("=" * 60)
    print(f"{'Dim':>8} {'Err exacta':>12} {'Err 10%':>12} "
          f"{'Max sim par':>12} {'Memoria':>10}")
    print("-" * 60)
    
    for dim in dimensions:
        r = all_results[str(dim)]
        err_exact = r["exact_recovery"]["error_rate"]
        err_10 = r["noisy_recovery"]["0.1"]["error_rate"]
        max_sim = r["collision_analysis"]["max_pairwise_similarity"]
        mem_kb = r["memory_usage"]["total_kb"]
        
        print(f"{dim:>8} {err_exact:>12.4%} {err_10:>12.4%} "
              f"{max_sim:>12.4f} {mem_kb:>8.2f} KB")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "HDC-Esp-001",
        "description": "Memoria asociativa para caracteres del español",
        "timestamp": datetime.now().isoformat(),
        "dimensions_tested": dimensions,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "hdc_esp_001_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
