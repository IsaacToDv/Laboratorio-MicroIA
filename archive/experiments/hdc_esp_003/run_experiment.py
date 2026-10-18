"""
Experimento HDC-Esp-003: Capacidad de Memoria Asociativa con Bundles
=====================================================================

Objetivo
--------
Determinar cuántas asociaciones pueden combinarse en un solo hipervector
mediante bundle y recuperarse con error < 5%.

Este es el verdadero test de capacidad de memoria asociativa HDC.

Hipótesis
---------
H1: Un bundle de L vectores permite recuperar cada componente con error < 5%
    cuando L < C = D / (2 * log₂(D)).

Variables
---------
- Independiente: 
  * Dimensión D ∈ {1024, 2048, 4096, 8192, 10000}
  * Longitud del bundle L ∈ {2, 5, 10, 20, 50, 100, 200}
- Dependiente: tasa de error de recuperación de componentes
- Control: semilla fija, mismos vectores base

Clasificación: E1 — Resultado reproducido (al ejecutarse).
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
    """Capacidad teórica para bundles: C ≈ D / (2 * log₂(D))"""
    if dimension < 2:
        return 0
    return int(dimension / (2 * np.log2(dimension)))


def test_bundle_recovery(hdc: HyperdimensionalVectors,
                         item_memory: ItemMemory,
                         bundle_size: int,
                         n_trials: int = 100,
                         seed: int = 42) -> dict:
    """
    Prueba la recuperación de componentes de bundles.
    
    Para cada trial:
    1. Selecciona L tokens aleatorios del item memory
    2. Crea un bundle de esos L tokens
    3. Intenta recuperar cada token del bundle
    4. Registra cuántos se recuperaron correctamente
    
    Retorna estadísticas de recuperación.
    """
    rng = np.random.RandomState(seed)
    all_keys = item_memory.get_all_keys()
    n_keys = len(all_keys)
    
    if bundle_size > n_keys:
        return {
            "bundle_size": bundle_size,
            "n_trials": 0,
            "error_rate": 1.0,
            "mean_recovered": 0,
            "message": f"Bundle size {bundle_size} > vocabulary size {n_keys}"
        }
    
    total_correct = 0
    total_attempts = 0
    trial_results = []
    
    for trial in range(n_trials):
        # Seleccionar L tokens aleatorios
        selected_keys = rng.choice(all_keys, size=bundle_size, replace=False)
        
        # Obtener vectores
        vectors = [item_memory.get(k) for k in selected_keys]
        
        # Crear bundle
        bundle = hdc.bundle(*vectors)
        
        # Intentar recuperar cada componente
        correct = 0
        for key in selected_keys:
            original_vec = item_memory.get(key)
            # El bundle debería ser similar al vector original
            # Recuperamos el más similar al bundle
            retrieved_key, sim = item_memory.query_exact(bundle)
            
            # Pero esto solo recupera UN token, no todos
            # Necesitamos verificar si el token original está entre los más similares
            results = item_memory.query(bundle, top_k=bundle_size)
            retrieved_keys = [r[0] for r in results]
            
            if key in retrieved_keys:
                correct += 1
        
        total_correct += correct
        total_attempts += bundle_size
        trial_results.append(correct / bundle_size)
    
    error_rate = 1.0 - (total_correct / total_attempts)
    
    return {
        "bundle_size": bundle_size,
        "n_trials": n_trials,
        "error_rate": error_rate,
        "mean_recovered": np.mean(trial_results),
        "std_recovered": np.std(trial_results),
        "min_recovered": np.min(trial_results),
        "max_recovered": np.max(trial_results),
        "total_correct": total_correct,
        "total_attempts": total_attempts,
    }


def run_single_experiment(dimension: int, bundle_size: int,
                          vocab_size: int = 1000,
                          n_trials: int = 50,
                          seed: int = 42) -> dict:
    """Ejecuta un experimento para una combinación D, L específica."""
    print(f"  D={dimension}, L={bundle_size:>3}... ", end="", flush=True)
    
    t0 = time.time()
    
    # Crear HDC y vocabulario
    hdc = HyperdimensionalVectors(dimension=dimension, seed=seed)
    memory = ItemMemory(hdc)
    
    # Generar vocabulario
    tokens = [f"token_{i:06d}" for i in range(vocab_size)]
    vectors = hdc.generate(vocab_size)
    memory.store_batch(tokens, vectors)
    
    setup_time = time.time() - t0
    
    # Probar recuperación de bundles
    t0 = time.time()
    recovery = test_bundle_recovery(hdc, memory, bundle_size, 
                                     n_trials=n_trials, seed=seed+1)
    test_time = time.time() - t0
    
    cap = theoretical_capacity(dimension)
    
    result = {
        "dimension": dimension,
        "bundle_size": bundle_size,
        "vocab_size": vocab_size,
        "n_trials": n_trials,
        "seed": seed,
        "error_rate": recovery["error_rate"],
        "mean_recovered": recovery.get("mean_recovered", 0),
        "std_recovered": recovery.get("std_recovered", 0),
        "min_recovered": recovery.get("min_recovered", 0),
        "max_recovered": recovery.get("max_recovered", 0),
        "setup_time_ms": setup_time * 1000,
        "test_time_ms": test_time * 1000,
        "test_time_per_trial_ms": (test_time / n_trials) * 1000 if n_trials > 0 else 0,
        "theoretical_capacity": cap,
        "load_factor": bundle_size / cap,
        "timestamp": datetime.now().isoformat(),
    }
    
    status = "✗" if result["error_rate"] > 0.05 else "✓"
    print(f"{status} error={result['error_rate']:>7.2%}, "
          f"recuperados={result['mean_recovered']:.2f}±{result['std_recovered']:.2f}, "
          f"carga={result['load_factor']:.2f}×")
    
    return result


def run_full_experiment(output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo de capacidad de bundles."""
    dimensions = [1024, 2048, 4096, 8192, 10000]
    bundle_sizes = [2, 5, 10, 20, 50, 100, 200]
    
    print("=" * 70)
    print("HDC-Esp-003: Capacidad de Memoria Asociativa con Bundles")
    print("=" * 70)
    print(f"Dimensiones: {dimensions}")
    print(f"Tamaños de bundle: {bundle_sizes}")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    all_results = {}
    
    for dim in dimensions:
        cap = theoretical_capacity(dim)
        print(f"\n{'─' * 70}")
        print(f"Dimensión D = {dim} (capacidad teórica: {cap})")
        print(f"{'─' * 70}")
        
        dim_results = []
        
        for bundle_size in bundle_sizes:
            # Usar vocabulario suficientemente grande
            vocab_size = max(1000, bundle_size * 5)
            
            result = run_single_experiment(dim, bundle_size, 
                                           vocab_size=vocab_size,
                                           n_trials=50, seed=42)
            dim_results.append(result)
        
        all_results[str(dim)] = {
            "dimension": dim,
            "theoretical_capacity": cap,
            "experiments": dim_results,
        }
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN COMPARATIVO")
    print("=" * 70)
    print(f"{'Dim':>6} {'Cap teórica':>12} {'Primer L con error>5%':>22} "
          f"{'Ratio/Cap':>10}")
    print("-" * 55)
    
    for dim in dimensions:
        cap = all_results[str(dim)]["theoretical_capacity"]
        experiments = all_results[str(dim)]["experiments"]
        
        breaking_point = None
        for exp in sorted(experiments, key=lambda x: x["bundle_size"]):
            if exp["error_rate"] > 0.05:
                breaking_point = exp["bundle_size"]
                break
        
        if breaking_point:
            ratio = breaking_point / cap
            print(f"{dim:>6} {cap:>12} {breaking_point:>22} {ratio:>10.2f}")
        else:
            max_tested = max(e["bundle_size"] for e in experiments)
            print(f"{dim:>6} {cap:>12} {'>' + str(max_tested):>22} {'N/A':>10}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "HDC-Esp-003",
        "description": "Capacidad de memoria asociativa con bundles",
        "timestamp": datetime.now().isoformat(),
        "dimensions_tested": dimensions,
        "bundle_sizes_tested": bundle_sizes,
        "target_error_rate": 0.05,
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "hdc_esp_003_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
