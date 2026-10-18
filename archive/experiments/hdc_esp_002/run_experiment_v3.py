"""
Experimento HDC-Esp-002: Escalabilidad a Vocabularios Grandes (v3)
===================================================================

Versión final que prueba rangos más amplios sin límites artificiales.
"""

import sys
import os
import time
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdc import HyperdimensionalVectors


def theoretical_capacity_conservative(dimension: int) -> int:
    """Capacidad conservadora: C ≈ D / (2 * log2(D))"""
    if dimension < 2:
        return 0
    return int(dimension / (2 * np.log2(dimension)))


def theoretical_capacity_optimistic(dimension: int) -> int:
    """Capacidad optimista: C ≈ D / log2(D)"""
    if dimension < 2:
        return 0
    return int(dimension / np.log2(dimension))


def fast_recovery_test(vectors: np.ndarray, batch_size: int = 2000) -> dict:
    """Prueba de recuperación vectorizada."""
    n = vectors.shape[0]
    
    matrix = vectors.astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    normalized = matrix / norms
    
    correct = 0
    min_sims = []
    
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normalized[start:end]
        
        sim_matrix = batch @ normalized.T
        
        best_indices = np.argmax(sim_matrix, axis=1)
        best_sims = np.max(sim_matrix, axis=1)
        
        for i in range(end - start):
            global_idx = start + i
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


def sample_max_pairwise_similarity(vectors: np.ndarray, seed: int,
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
    
    matrix = vectors.astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1)
    
    sims = []
    for i, j in zip(idx_a[:min(5000, len(idx_a))], 
                    idx_b[:min(5000, len(idx_b))]):
        dot = np.dot(matrix[i], matrix[j])
        sim = dot / (norms[i] * norms[j])
        sims.append(float(sim))
    
    return float(max(sims)) if sims else 0.0


def run_single_experiment(dimension: int, vocab_size: int, 
                          seed: int = 42) -> dict:
    """Ejecuta un experimento para una combinación D, N específica."""
    print(f"  D={dimension}, N={vocab_size:>6}... ", end="", flush=True)
    
    t0 = time.time()
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    vectors = hdc.generate(vocab_size)
    gen_time = time.time() - t0
    
    t0 = time.time()
    recovery = fast_recovery_test(vectors)
    recovery_time = time.time() - t0
    
    max_pair_sim = sample_max_pairwise_similarity(vectors, seed)
    
    cap_cons = theoretical_capacity_conservative(dimension)
    cap_opt = theoretical_capacity_optimistic(dimension)
    
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
        "theoretical_capacity_conservative": cap_cons,
        "theoretical_capacity_optimistic": cap_opt,
        "load_factor_conservative": vocab_size / cap_cons,
        "load_factor_optimistic": vocab_size / cap_opt,
        "max_pairwise_similarity_sampled": max_pair_sim,
        "timestamp": datetime.now().isoformat(),
    }
    
    status = "✗" if result["error_rate"] > 0.05 else "✓"
    print(f"{status} error={result['error_rate']:>7.2%}, "
          f"sim_min={result['min_similarity']:.3f}, "
          f"sim_max_par={max_pair_sim:.3f}, "
          f"carga={result['load_factor_conservative']:.1f}×")
    
    return result


def run_full_experiment(output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo de escalabilidad."""
    dimensions = [1024, 2048, 4096, 8192, 10000]
    
    # Tamaños de vocabulario adaptativos por dimensión
    vocab_sizes_by_dim = {
        1024: [100, 200, 500, 1000, 2000, 5000],
        2048: [100, 500, 1000, 2000, 5000, 10000],
        4096: [100, 500, 1000, 2000, 5000, 10000, 20000],
        8192: [100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
        10000: [100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
    }
    
    print("=" * 70)
    print("HDC-Esp-002: Escalabilidad a Vocabularios Grandes (v3)")
    print("=" * 70)
    print(f"Dimensiones: {dimensions}")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    all_results = {}
    
    for dim in dimensions:
        cap_cons = theoretical_capacity_conservative(dim)
        cap_opt = theoretical_capacity_optimistic(dim)
        vocab_sizes = vocab_sizes_by_dim[dim]
        
        print(f"\n{'─' * 70}")
        print(f"Dimensión D = {dim}")
        print(f"  Capacidad teórica conservadora: {cap_cons}")
        print(f"  Capacidad teórica optimista:    {cap_opt}")
        print(f"{'─' * 70}")
        
        dim_results = []
        
        for size in vocab_sizes:
            result = run_single_experiment(dim, size, seed=42)
            dim_results.append(result)
        
        all_results[str(dim)] = {
            "dimension": dim,
            "theoretical_capacity_conservative": cap_cons,
            "theoretical_capacity_optimistic": cap_opt,
            "experiments": dim_results,
        }
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Dim':>6} {'Cap cons':>10} {'Cap opt':>10} {'Max N probado':>14} "
          f"{'Error':>8}")
    print("-" * 55)
    
    for dim in dimensions:
        cap_cons = all_results[str(dim)]["theoretical_capacity_conservative"]
        cap_opt = all_results[str(dim)]["theoretical_capacity_optimistic"]
        experiments = all_results[str(dim)]["experiments"]
        
        max_n = max(e["vocab_size"] for e in experiments)
        max_n_error = [e["error_rate"] for e in experiments 
                       if e["vocab_size"] == max_n][0]
        
        print(f"{dim:>6} {cap_cons:>10} {cap_opt:>10} {max_n:>14} "
              f"{max_n_error:>8.2%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "HDC-Esp-002",
        "description": "Escalabilidad a vocabularios grandes (v3 - rangos amplios)",
        "timestamp": datetime.now().isoformat(),
        "dimensions_tested": dimensions,
        "vocab_sizes_by_dim": vocab_sizes_by_dim,
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
