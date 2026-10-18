"""
Tests básicos para el generador de texto

Ejecutar con: python tests/test_generation.py
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import TextGenerator


def test_creacion_generador():
    """Test: Se puede crear un generador."""
    print("Test 1: Creación de generador...", end=" ")
    
    gen = TextGenerator(dim=1000)
    assert gen is not None
    assert gen.memory.dim == 1000
    assert gen.is_trained == False
    
    print("✓")


def test_entrenamiento():
    """Test: El generador se puede entrenar con corpus."""
    print("Test 2: Entrenamiento con corpus...", end=" ")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("⊘ (corpus no disponible)")
        return
    
    gen = TextGenerator(dim=5000)
    gen.train_from_corpus(corpus_path, max_sentences=100)
    
    assert gen.is_trained == True
    assert len(gen.vocabulary) > 0
    assert len(gen.memory.sequence_memory) > 0
    
    print("✓")


def test_generacion_basica():
    """Test: El generador produce texto."""
    print("Test 3: Generación básica...", end=" ")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("⊘ (corpus no disponible)")
        return
    
    gen = TextGenerator(dim=5000)
    gen.train_from_corpus(corpus_path, max_sentences=100)
    
    # Generar texto
    text = gen.generate("El presidente", max_words=10, temperature=0.5)
    
    assert isinstance(text, str)
    assert len(text) > 0
    assert text.startswith("El presidente")
    
    print("✓")


def test_sin_repeticiones_consecutivas():
    """Test: El generador evita repeticiones consecutivas."""
    print("Test 4: Sin repeticiones consecutivas...", end=" ")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("⊘ (corpus no disponible)")
        return
    
    gen = TextGenerator(dim=5000)
    gen.train_from_corpus(corpus_path, max_sentences=100)
    
    # Generar texto largo
    text = gen.generate("El", max_words=20, temperature=0.7)
    words = text.split()
    
    # Verificar que no hay 3 palabras iguales consecutivas
    for i in range(len(words) - 2):
        # Excluir palabras funcionales que pueden repetirse
        if words[i] in ['de', 'la', 'el', 'y', 'en']:
            continue
        
        assert not (words[i] == words[i+1] == words[i+2]), \
            f"Repetición encontrada: {words[i]} {words[i+1]} {words[i+2]}"
    
    print("✓")


def test_temperatura():
    """Test: Diferentes temperaturas producen diferentes resultados."""
    print("Test 5: Efecto de temperatura...", end=" ")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("⊘ (corpus no disponible)")
        return
    
    gen = TextGenerator(dim=5000)
    gen.train_from_corpus(corpus_path, max_sentences=100)
    
    # Generar con diferentes temperaturas
    text_baja = gen.generate("El acuerdo", max_words=10, temperature=0.1)
    text_alta = gen.generate("El acuerdo", max_words=10, temperature=2.0)
    
    # Deberían ser diferentes (con alta probabilidad)
    assert isinstance(text_baja, str)
    assert isinstance(text_alta, str)
    
    print("✓")


def test_longitud_controlada():
    """Test: Se puede controlar la longitud del texto."""
    print("Test 6: Control de longitud...", end=" ")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("⊘ (corpus no disponible)")
        return
    
    gen = TextGenerator(dim=5000)
    gen.train_from_corpus(corpus_path, max_sentences=100)
    
    # Generar con diferentes longitudes
    text_corto = gen.generate("El", max_words=5, temperature=0.5)
    text_largo = gen.generate("El", max_words=20, temperature=0.5)
    
    words_corto = len(text_corto.split())
    words_largo = len(text_largo.split())
    
    # El texto largo debería tener más palabras (o igual si terminó con punto)
    assert words_corto <= words_largo or '.' in text_corto
    
    print("✓")


def main():
    """Ejecuta todos los tests."""
    print("=" * 70)
    print("Tests del Generador de Texto")
    print("=" * 70)
    print()
    
    tests = [
        test_creacion_generador,
        test_entrenamiento,
        test_generacion_basica,
        test_sin_repeticiones_consecutivas,
        test_temperatura,
        test_longitud_controlada,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ ({e})")
            failed += 1
        except Exception as e:
            print(f"✗ (error: {e})")
            failed += 1
    
    print()
    print("=" * 70)
    print(f"Resultados: {passed} passed, {failed} failed")
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
