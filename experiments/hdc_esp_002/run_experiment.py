"""
Experimento HDC-Esp-002: Escalabilidad a Vocabularios Grandes
==============================================================

Objetivo
--------
Determinar el límite práctico de capacidad de HDC para vocabularios de tokens,
validando la fórmula teórica de capacidad O(D / log D).

Hipótesis
---------
H1: La capacidad práctica de HDC sigue la fórmula teórica C ≈ D / (2 * log D)
    con tasa de error < 5% cuando N < C.

H0 (nula): La capacidad práctica es significativamente menor que la teórica,
    o el error excede 5% antes de alcanzar el límite teórico.

Variables
---------
- Independiente: 
  * Dimensión D ∈ {1024, 2048, 4096, 8192, 10000, 20000}
  * Tamaño de vocabulario N ∈ {100, 500, 1000, 5000, 10000, 20000, 50000}
- Dependiente: tasa de error de recuperación
- Control: semilla fija, misma distribución de hipervectores

Procedimiento
-------------
1. Para cada dimensión D, generar vocabularios de tamaño N
2. Almacenar todos los tokens en item memory
3. Consultar cada token y medir tasa de error
4. Comparar con capacidad teórica
5. Identificar punto de quiebre (error > 5%)

Clasificación: E1 — Resultado reproducido (al ejecutarse).
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdc import HyperdimensionalVectors, ItemMemory


def theoretical_capacity(dimension: int) -> int:
    """
    Calcula la capacidad teórica de memoria asociativa HDC.
    
    Fórmula: C ≈ D / (2 * log2(D))
    
    Esta es una aproximación conservadora basada en:
    - Kanerva (2009): capacidad escala como O(D / log D)
    - Factor de seguridad para garantizar baja tasa de error
    
    Parámetros
    ----------
    dimension : int
        Dimensión del espacio HDC.
    
    Retorna
    -------
    int
        Capacidad teórica estimada (número de ítems).
    """
    if dimension < 2:
        return 0
    return int(dimension / (2 * np.log2(dimension)))


def run_single_experiment(dimension: int, vocab_size: int, 
                          seed: int = 42) -> dict:
    """
    Ejecuta un experimento para una combinación D, N específica.
    """
    # Generar vocabulario sintético
    rng = np.random.RandomState(seed)
    tokens = [f"token_{i:06d}" for i in range(vocab_size)]
    
    # Crear HDC y memoria
    t0 = time.time()
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    memory = ItemMemory(hdc)
    
    # Almacenar todos los tokens
    vectors = hdc.generate(vocab_size)
    memory.store_batch(tokens, vectors)
    store_time = time.time() - t0
    
    # Recuperación
    t0 = time.time()
    correct = 0
    similarities = []
    
    for i, token in enumerate(tokens):
        v = memory.get(token)
        retrieved_token, sim = memory.query_exact(v)
        if retrieved_token == token:
            correct += 1
        similarities.append(sim)
    
    query_time = time.time() - t0
    
    error_rate = 1.0 - (correct / vocab_size)
    mean_similarity = float(np.mean(similarities))
    min_similarity = float(np.min(similarities))
    
    # Análisis de colisiones (muestreo para N grande)
    max_pair_sim = 0.0
    if vocab_size <= 1000:
        collision_stats = memory.collision_analysis()
        max_pair_sim = collision_stats["max_pairwise_similarity"]
    else:
        # Muestreo: comparar 10000 pares aleatorios
        n_sample_pairs = min(10000, vocab_size * 10)
        sample_rng = np.random.RandomState(seed + 100)
        idx_a = sample_rng.randint(0, vocab_size, n_sample_pairs)
        idx_b = sample_rng.randint(0, vocab_size, n_sample_pairs)
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        
        for i, j in zip(idx_a[:5000], idx_b[:5000]):
            sim = float(hdc.similarity(vectors[i], vectors[j]))
            max_pair_sim = max(max_pair_sim, sim)
    
    return {
        "dimension": dimension,
        "vocab_size": vocab_size,
        "seed": seed,
        "error_rate": float(error_rate),
        "correct": correct,
        "total": vocab_size,
        "mean_similarity": mean_similarity,
        "min_similarity": min_similarity,
        "store_time_ms": store_time * 1000,
        "query_time_ms": query_time * 1000,
        "query_time_per_item_us": (query_time / vocab_size) * 1e6,
        "theoretical_capacity": theoretical_capacity(dimension),
        "load_factor": vocab_size / theoretical_capacity(dimension),
        "max_pairwise_similarity": max_pair_sim,
        "timestamp": datetime.now().isoformat(),
    }


def find_breaking_point(dimension: int, target_error: float = 0.05,
                        seed: int = 42) -> dict:
    """
    Encuentra el tamaño de vocabulario donde el error excede el objetivo.
    
    Usa búsqueda binaria para eficiencia.
    """
    theo_cap = theoretical_capacity(dimension)
    
    # Rango de búsqueda: 10% a 300% de capacidad teórica
    low = max(100, int(theo_cap * 0.1))
    high = int(theo_cap * 3.0)
    
    print(f"\nBuscando punto de quiebre para D={dimension}")
    print(f"  Capacidad teórica: {theo_cap}")
    print(f"  Rango de búsqueda: [{low}, {high}]")
    
    results = []
    tested_sizes = set()
    
    # Probar puntos clave primero
    key_points = [
        int(theo_cap * 0.25),
        int(theo_cap * 0.50),
        int(theo_cap * 0.75),
        int(theo_cap * 1.00),
        int(theo_cap * 1.25),
        int(theo_cap * 1.50),
        int(theo_cap * 2.00),
    ]
    
    for size in key_points:
        if size not in tested_sizes and low <= size <= high:
            result = run_single_experiment(dimension, size, seed)
            results.append(result)
            tested_sizes.add(size)
            
            status = "✗" if result["error_rate"] > target_error else "✓"
            print(f"  {status} N={size:>6}: error={result['error_rate']:.2%}, "
                  f"carga={result['load_factor']:.2f}")
    
    # Búsqueda binaria refinada
    if len(results) >= 2:
        results.sort(key=lambda x: x["vocab_size"])
        
        # Encontrar intervalo donde cruza el umbral
        for i in range(len(results) - 1):
            if (results[i]["error_rate"] <= target_error and 
                results[i+1]["error_rate"] > target_error):
                
                # Refinar con puntos intermedios
                low_size = results[i]["vocab_size"]
                high_size = results[i+1]["vocab_size"]
                
                for _ in range(3):  # 3 iteraciones de refinamiento
                    mid_size = (low_size + high_size) // 2
                    if mid_size not in tested_sizes and mid_size > low_size:
                        result = run_single_experiment(dimension, mid_size, seed)
                        results.append(result)
                        tested_sizes.add(mid_size)
                        
                        status = "✗" if result["error_rate"] > target_error else "✓"
                        print(f"  {status} N={mid_size:>6}: "
                              f"error={result['error_rate']:.2%} (refinado)")
                        
                        if result["error_rate"] <= target_error:
                            low_size = mid_size
                        else:
                            high_size = mid_size
    
    results.sort(key=lambda x: x["vocab_size"])
    
    # Encontrar punto de quiebre
    breaking_point = None
    for result in results:
        if result["error_rate"] > target_error:
            breaking_point = result["vocab_size"]
            break
    
    return {
        "dimension": dimension,
        "theoretical_capacity": theo_cap,
        "breaking_point": breaking_point,
        "target_error": target_error,
        "results": results,
    }


def run_full_experiment(output_dir: str = ".") -> dict:
    """
    Ejecuta el experimento completo de escalabilidad.
    """
    dimensions = [1024, 2048, 4096, 8192, 10000, 20000]
    
    print("=" * 70)
    print("HDC-Esp-002: Escalabilidad a Vocabularios Grandes")
    print("=" * 70)
    print(f"Dimensiones a evaluar: {dimensions}")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    all_results = {}
    
    # Fase 1: Matriz de experimentos
    print("\n" + "=" * 70)
    print("FASE 1: Matriz de experimentos (D × N)")
    print("=" * 70)
    
    vocab_sizes = [100, 500, 1000, 5000, 10000, 20000, 50000]
    
    for dim in dimensions:
        print(f"\n{'─' * 70}")
        print(f"Dimensión D = {dim}")
        print(f"{'─' * 70}")
        
        theo_cap = theoretical_capacity(dim)
        print(f"Capacidad teórica: {theo_cap} tokens")
        print()
        
        dim_results = []
        
        for size in vocab_sizes:
            # Saltar si el tamaño es mucho mayor que la capacidad teórica
            if size > theo_cap * 4:
                print(f"  N={size:>6}: omitido (excede 4× capacidad teórica)")
                continue
            
            result = run_single_experiment(dim, size, seed=42)
            dim_results.append(result)
            
            status = "✗" if result["error_rate"] > 0.05 else "✓"
            print(f"{status} N={size:>6}: error={result['error_rate']:>7.2%}, "
                  f"sim_min={result['min_similarity']:.3f}, "
                  f"carga={result['load_factor']:.2f}, "
                  f"tiempo={result['query_time_per_item_us']:.0f}μs")
        
        all_results[str(dim)] = {
            "dimension": dim,
            "theoretical_capacity": theo_cap,
            "experiments": dim_results,
        }
    
    # Fase 2: Búsqueda de puntos de quiebre
    print("\n" + "=" * 70)
    print("FASE 2: Búsqueda de puntos de quiebre (error > 5%)")
    print("=" * 70)
    
    breaking_points = {}
    
    for dim in dimensions:
        bp_result = find_breaking_point(dim, target_error=0.05, seed=42)
        breaking_points[str(dim)] = bp_result
    
    # Resumen comparativo
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Dim':>8} {'Cap teórica':>12} {'Punto quiebre':>14} "
          f"{'Ratio':>8} {'Error en quiebre':>17}")
    print("-" * 70)
    
    for dim in dimensions:
        theo = breaking_points[str(dim)]["theoretical_capacity"]
        bp = breaking_points[str(dim)]["breaking_point"]
        
        if bp is not None:
            # Encontrar error en punto de quiebre
            bp_error = None
            for r in breaking_points[str(dim)]["results"]:
                if r["vocab_size"] == bp:
                    bp_error = r["error_rate"]
                    break
            
            ratio = bp / theo if bp and theo > 0 else 0
            print(f"{dim:>8} {theo:>12} {bp:>14} {ratio:>8.2f} "
                  f"{bp_error:>17.2%}")
        else:
            print(f"{dim:>8} {theo:>12} {'N/A':>14} {'N/A':>8} {'N/A':>17}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "HDC-Esp-002",
        "description": "Escalabilidad a vocabularios grandes",
        "timestamp": datetime.now().isoformat(),
        "dimensions_tested": dimensions,
        "vocab_sizes_tested": vocab_sizes,
        "target_error_rate": 0.05,
        "matrix_results": all_results,
        "breaking_points": breaking_points,
    }
    
    output_path = os.path.join(output_dir, "hdc_esp_002_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
