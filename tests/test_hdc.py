"""
Pruebas unitarias para el módulo HDC.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hdc import HyperdimensionalVectors, ItemMemory, SpanishCharacterMemory, SequenceEncoder


def test_generate_bipolar():
    """Los hipervectores generados son bipolares {-1, +1}."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    v = hdc.generate(1)
    assert v.shape == (1000,), f"Forma incorrecta: {v.shape}"
    assert set(np.unique(v)) == {-1, 1}, "Valores no bipolares"
    print("✓ test_generate_bipolar")


def test_generate_batch():
    """La generación en lote produce la forma correcta."""
    hdc = HyperdimensionalVectors(dimension=500, seed=42)
    vectors = hdc.generate(10)
    assert vectors.shape == (10, 500), f"Forma incorrecta: {vectors.shape}"
    print("✓ test_generate_batch")


def test_bind_self_inverse():
    """El binding es autoinverso: bind(bind(a, b), b) = a."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    a = hdc.generate(1)
    b = hdc.generate(1)
    
    bound = hdc.bind(a, b)
    recovered = hdc.bind(bound, b)
    
    assert np.array_equal(recovered, a), "Binding no es autoinverso"
    print("✓ test_bind_self_inverse")


def test_bind_commutative():
    """El binding es conmutativo: bind(a, b) = bind(b, a)."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    a = hdc.generate(1)
    b = hdc.generate(1)
    
    assert np.array_equal(hdc.bind(a, b), hdc.bind(b, a))
    print("✓ test_bind_commutative")


def test_bundle_commutative():
    """El bundle es conmutativo: bundle(a, b) = bundle(b, a)."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    a = hdc.generate(1)
    b = hdc.generate(1)
    
    assert np.array_equal(hdc.bundle(a, b), hdc.bundle(b, a))
    print("✓ test_bundle_commutative")


def test_similarity_self():
    """La similitud de un vector consigo mismo es 1.0."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    v = hdc.generate(1)
    sim = hdc.similarity(v, v)
    assert abs(sim - 1.0) < 1e-10, f"Similitud propia: {sim}"
    print("✓ test_similarity_self")


def test_similarity_random_orthogonal():
    """Vectores aleatorios tienen similitud cercana a 0."""
    hdc = HyperdimensionalVectors(dimension=10000, seed=42)
    a = hdc.generate(1)
    b = hdc.generate(1)
    sim = hdc.similarity(a, b)
    
    # En D=10000, la similitud debería estar dentro de ±3σ = ±0.03
    assert abs(sim) < 0.1, f"Similitud inesperada: {sim}"
    print(f"✓ test_similarity_random_orthogonal (sim={sim:.4f})")


def test_permute_inverse():
    """La permutación es invertible: permute(permute(v, k), -k) = v."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    v = hdc.generate(1)
    
    permuted = hdc.permute(v, shift=5)
    recovered = hdc.permute(permuted, shift=-5)
    
    assert np.array_equal(recovered, v), "Permutación no es invertible"
    print("✓ test_permute_inverse")


def test_permute_dissimilar():
    """Un vector permutado es disimilar al original."""
    hdc = HyperdimensionalVectors(dimension=10000, seed=42)
    v = hdc.generate(1)
    permuted = hdc.permute(v, shift=1)
    
    sim = hdc.similarity(v, permuted)
    # La permutación circular debería hacer el vector muy diferente
    assert abs(sim) < 0.1, f"Similitud inesperada tras permutar: {sim}"
    print(f"✓ test_permute_dissimilar (sim={sim:.4f})")


def test_item_memory_store_and_retrieve():
    """La memoria de ítems almacena y recupera correctamente."""
    hdc = HyperdimensionalVectors(dimension=1000, seed=42)
    memory = ItemMemory(hdc)
    
    v = memory.store("test_key")
    retrieved = memory.get("test_key")
    
    assert np.array_equal(v, retrieved), "Recuperación incorrecta"
    print("✓ test_item_memory_store_and_retrieve")


def test_item_memory_query():
    """La consulta por similitud retorna la clave correcta."""
    hdc = HyperdimensionalVectors(dimension=5000, seed=42)
    memory = ItemMemory(hdc)
    
    keys = ["a", "b", "c", "d", "e"]
    memory.store_batch(keys)
    
    # Consultar cada clave
    for key in keys:
        v = memory.get(key)
        result_key, sim = memory.query_exact(v)
        assert result_key == key, f"Esperaba {key}, obtenida {result_key}"
        assert abs(sim - 1.0) < 1e-10, f"Similitud incorrecta: {sim}"
    
    print("✓ test_item_memory_query")


def test_spanish_character_memory():
    """La memoria de caracteres del español funciona correctamente."""
    hdc = HyperdimensionalVectors(dimension=5000, seed=42)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    
    assert memory.capacity > 30, f"Capacidad insuficiente: {memory.capacity}"
    
    # Verificar que ñ está en el inventario básico
    assert "ñ" in memory, "ñ no está en el inventario"
    
    # Verificar recuperación
    v = memory.get("ñ")
    result_key, sim = memory.query_exact(v)
    assert result_key == "ñ", f"Error recuperando ñ: {result_key}"
    
    print(f"✓ test_spanish_character_memory (capacity={memory.capacity})")


def test_sequence_encoder_bag():
    """El codificador bag ignora el orden."""
    hdc = HyperdimensionalVectors(dimension=5000, seed=42)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    encoder = SequenceEncoder(hdc, memory, method="bag")
    
    v1 = encoder.encode("abc")
    v2 = encoder.encode("cba")
    
    # Deberían ser idénticos (bag ignora orden)
    sim = hdc.similarity(v1, v2)
    assert abs(sim - 1.0) < 1e-10, f"Bag no ignora orden: sim={sim}"
    print(f"✓ test_sequence_encoder_bag (sim={sim:.4f})")


def test_sequence_encoder_positional():
    """El codificador posicional distingue el orden."""
    hdc = HyperdimensionalVectors(dimension=10000, seed=42)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    encoder = SequenceEncoder(hdc, memory, method="positional_binding")
    
    v1 = encoder.encode("abc")
    v2 = encoder.encode("cba")
    
    sim = hdc.similarity(v1, v2)
    # Deberían ser diferentes
    assert sim < 0.5, f"Codificador posicional no distingue orden: sim={sim}"
    print(f"✓ test_sequence_encoder_positional (sim={sim:.4f})")


def test_sequence_encoder_different_words_dissimilar():
    """Palabras diferentes producen hipervectores disimilares."""
    hdc = HyperdimensionalVectors(dimension=10000, seed=42)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    encoder = SequenceEncoder(hdc, memory, method="positional_binding")
    
    words = ["hola", "mundo", "casa", "perro", "gato"]
    encoded = encoder.encode_batch(words)
    
    # Verificar similitudes por pares
    max_sim = -1
    for i in range(len(words)):
        for j in range(i+1, len(words)):
            sim = hdc.similarity(encoded[i], encoded[j])
            max_sim = max(max_sim, sim)
    
    assert max_sim < 0.3, f"Similitud máxima demasiado alta: {max_sim}"
    print(f"✓ test_sequence_encoder_different_words (max_sim={max_sim:.4f})")


def test_noise_robustness():
    """La recuperación es robusta a ruido moderado."""
    hdc = HyperdimensionalVectors(dimension=10000, seed=42)
    memory = SpanishCharacterMemory(hdc, inventory="basic")
    
    rng = np.random.RandomState(42)
    correct_5pct = 0
    total = memory.capacity
    
    for key in memory.get_all_keys():
        v = memory.get(key)
        # Añadir 5% de ruido
        n_flip = int(len(v) * 0.05)
        noisy = v.copy()
        indices = rng.choice(len(v), n_flip, replace=False)
        noisy[indices] *= -1
        
        result_key, _ = memory.query_exact(noisy)
        if result_key == key:
            correct_5pct += 1
    
    error_rate = 1.0 - correct_5pct / total
    assert error_rate < 0.05, f"Error con 5% ruido: {error_rate:.4%}"
    print(f"✓ test_noise_robustness (5% noise error: {error_rate:.4%})")


def test_reproducibility():
    """Los resultados son reproducibles con la misma semilla."""
    hdc1 = HyperdimensionalVectors(dimension=1000, seed=42)
    hdc2 = HyperdimensionalVectors(dimension=1000, seed=42)
    
    v1 = hdc1.generate(5)
    v2 = hdc2.generate(5)
    
    assert np.array_equal(v1, v2), "Resultados no reproducibles"
    print("✓ test_reproducibility")


if __name__ == "__main__":
    tests = [
        test_generate_bipolar,
        test_generate_batch,
        test_bind_self_inverse,
        test_bind_commutative,
        test_bundle_commutative,
        test_similarity_self,
        test_similarity_random_orthogonal,
        test_permute_inverse,
        test_permute_dissimilar,
        test_item_memory_store_and_retrieve,
        test_item_memory_query,
        test_spanish_character_memory,
        test_sequence_encoder_bag,
        test_sequence_encoder_positional,
        test_sequence_encoder_different_words_dissimilar,
        test_noise_robustness,
        test_reproducibility,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("Pruebas unitarias — Módulo HDC")
    print("=" * 60)
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Resultados: {passed} pasadas, {failed} fallidas, "
          f"{len(tests)} totales")
    print(f"{'='*60}")
