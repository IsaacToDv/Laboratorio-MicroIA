"""
Experimento HDC-Esp-002: Escalabilidad a Vocabularios Grandes (Optimizado)
===========================================================================

Versión optimizada que evita análisis O(N²) para vocabularios grandes.
Usa consulta vectorizada y muestreo de colisiones.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdc import HyperdimensionalVectors, ItemMemory


def theoretical_capacity(dimension: int) -> int:
    """Capacidad teórica: C ≈ D / (2 * log2(D))"""
    if dimension < 2:
        return 0
    return int(dimension / (2 * np.log2(dimension)))


def fast_recovery_test(hdc, vectors, top_k: int = 1) -> dict:
    """
    Prueba de recuperación vectorizada para todos los vectores.
    
    En lugar de consultar uno por uno (O(N²·D)), hacemos una sola
    operación matricial (O(N²·D) pero vectorizada en NumPy).
    """
    n = vectors.shape[0]
    d = vectors.shape[1]
    
    # Convertir a float64 para precisión
    matrix = vectors.astype(np.float64)
    
    # Normalizar
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
    
    # Matriz de similitud completa: (N, N)
    # Para N=10000 y D=10000, esto requiere ~800MB de RAM (float64)
    # Para N=20000, ~3.2GB - puede ser problemático
    # Usamos procesamiento por lotes para N grande
    
    batch_size = 2000
    correct = 0
    min_sims = []
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normalized[start:end]  # (batch_size, D)
        
        # Similitud de cada vector del lote con todos
        sim_matrix = batch @ normalized.T  # (batch_size, N)
        
        # Top-1 para cada vector del lote (incluyendo auto-similitud)
        best_indices = np.argmax(sim_matrix, axis=1)
        best_sims = np.max(sim_matrix, axis=1)
        
        for i in range(end - start):
            global_idx = start + i
            # El mejor match debería ser el propio vector (auto-similitud = 1.0)
            if best_indices[i] == global_idx:
                correct += 1
            min_sims.append(float(best_sims[i]))
    
    return {
        "correct": correct,
        "total": n,
        "error_rate": 1.0 - (correct / n),
        "min_similarity": float(np.min(min_sims)) if min_sims else 0.0,
        "mean_best_similarity": float(np.mean(min_sims)) if min_sims else 0.0,
    }


def sample_max_pairwise_similarity(hdc, vectors, seed, 
                                    n_samples=5000) -> float:
    """Estima la similitud máxima entre pares por muestreo."""
    n = vectors.shape[0]
    if n < 2:
        return 0.0
    
    rng = np.random.RandomState(seed)
    idx_a = rng.randint(0, n, n_samples)
    idx_b = rng.randint(0, n, n_samples)
    mask = idx_a != idx_b
    idx_a, idx_b = idx_a[mask], idx_b[mask]
    
    sims = []
    for i, j in zip(idx_a, idx_b):
        sim = float(hdc.similarity(vectors[i], vectors[j]))
        sims.append(sim)
    
    return float(max(sims)) if sims else 0.0


def run_single_experiment(dimension: int, vocab_size: int, 
                          seed: int = 42) -> dict:
    """Ejecuta un experimento para una combinación D, N específica."""
    print(f"  D={dimension}, N={vocab_size}... ", end="", flush=True)
    
    rng = np.random.RandomState(seed)
    t0 = time.time()
    
    # Generar todos los vectores de una vez
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    vectors = hdc.generate(vocab_size)
    gen_time = time.time() - t0
    
    # Prueba de recuperación
    t0 = time.time()
    recovery = fast_recovery_test(hdc, vectors)
    recovery_time = time.time() - t0
    
    # Similitud máxima por muestreo
    max_pair_sim = sample_max_pairwise_similarity(hdc, vectors, seed)
    
    result = {
        "dimension": dimension,
        "vocab_size": vocab_size,
        "seed": seed,
        "error_rate": recovery["error_rate"],
        "correct": recovery["correct"],
        "total": recovery["total"],
        "mean_best_similarity": recovery["mean_best_similarity"],
        "min_similarity": recovery["min_similarity"],
        "generation_time_ms": gen_time * 1000,
        "recovery_time_ms": recovery_time * 1000,
        "recovery_time_per_item_us": (recovery_time / vocab_size) * 1e6,
        "theoretical_capacity": theoretical_capacity(dimension),
        "load_factor": vocab_size / theoretical_capacity(dimension),
        "max_pairwise_similarity_sampled": max_pair_sim,
        "timestamp": datetime.now().isoformat(),
    }
    
    status = "✗" if result["error_rate"] > 0.05 else "✓"
    print(f"{status} error={result['error_rate']:>7.2%}, "
          f"carga={result['load_factor']:.2f}, "
          f"tiempo={result['recovery_time_per_item_us']:.0f}μs")
    
    return result


def run_full_experiment(output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo de escalabilidad."""
    dimensions = [1024, 2048, 4096, 8192, 10000]
    vocab_sizes = [100, 500, 1000, 2000, 5000, 10000, 20000]
    
    print("=" * 70)
    print("HDC-Esp-002: Escalabilidad a Vocabularios Grandes")
    print("=" * 70)
    print(f"Dimensiones: {dimensions}")
    print(f"Tamaños de vocabulario: {vocab_sizes}")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    all_results = {}
    
    for dim in dimensions:
        theo_cap = theoretical_capacity(dim)
        print(f"\n{'─' * 70}")
        print(f"Dimensión D = {dim} (capacidad teórica: {theo_cap})")
        print(f"{'─' * 70}")
        
        dim_results = []
        
        for size in vocab_sizes:
            # Saltar si excede 4× capacidad teórica
            if size > theo_cap * 4:
                print(f"  D={dim}, N={size}: omitido "
                      f"(excede 4× capacidad teórica)")
                continue
            
            result = run_single_experiment(dim, size, seed=42)
            dim_results.append(result)
        
        all_results[str(dim)] = {
            "dimension": dim,
            "theoretical_capacity": theo_cap,
            "experiments": dim_results,
        }
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Dim':>6} {'Cap teórica':>12} {'Primer error>5%':>16} "
          f"{'Ratio/Cap':>10}")
    print("-" * 50)
    
    for dim in dimensions:
        theo = all_results[str(dim)]["theoretical_capacity"]
        experiments = all_results[str(dim)]["experiments"]
        
        breaking_point = None
        for exp in sorted(experiments, key=lambda x: x["vocab_size"]):
            if exp["error_rate"] > 0.05:
                breaking_point = exp["vocab_size"]
                break
        
        if breaking_point:
            ratio = breaking_point / theo
            print(f"{dim:>6} {theo:>12} {breaking_point:>16} {ratio:>10.2f}")
        else:
            max_tested = max(e["vocab_size"] for e in experiments)
            print(f"{dim:>6} {theo:>12} {'>' + str(max_tested):>16} {'N/A':>10}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "HDC-Esp-002",
        "description": "Escalabilidad a vocabularios grandes (optimizado)",
        "timestamp": datetime.now().isoformat(),
        "dimensions_tested": dimensions,
        "vocab_sizes_tested": vocab_sizes,
        "target_error_rate": 0.05,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "hdc_esp_002_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
