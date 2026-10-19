#!/usr/bin/env python3
"""
Demo de generación por disciplinas

Muestra cómo el generador puede producir texto en diferentes dominios:
- Política
- Economía
- Ciencia
- Tecnología
- Cultura

Uso:
    python examples/generacion_por_disciplina.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import TextGenerator


def main():
    """Genera texto en diferentes disciplinas."""
    print("=" * 70)
    print("Generación de Texto por Disciplinas")
    print("=" * 70)
    print()
    
    # Agregar scripts al path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    
    # Cargar generador desde cualquier fuente disponible
    print("Cargando generador...")
    gen = None
    
    # Intentar 1: Modelo pre-entrenado
    trained_corpus_path = Path("models/trained_corpus.json")
    if trained_corpus_path.exists():
        try:
            import json
            with open(trained_corpus_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw_sentences = data.get('sentences', [])
            sentences = []
            for text in raw_sentences:
                words = text.strip().split()
                if len(words) >= 3:
                    sentences.append(words)
            
            if sentences:
                gen = TextGenerator(dim=20000)
                for sent in sentences:
                    gen.memory.store_sequence(sent)
                gen.selector.train(sentences)
                gen.is_trained = True
                print(f"✓ Modelo pre-entrenado cargado ({len(sentences)} oraciones)\n")
        except Exception:
            pass
    
    # Intentar 2: Corpus UD AnCora
    if gen is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
        if Path(corpus_path).exists():
            gen = TextGenerator(dim=20000)
            gen.train_from_corpus(corpus_path, max_sentences=3000)
            print("✓ Generador listo (UD AnCora)\n")
    
    # Intentar 3: data_loader
    if gen is None:
        try:
            from data_loader import ensure_data_available
            all_sentences, sources = ensure_data_available(min_sentences=50)
            if all_sentences:
                gen = TextGenerator(dim=20000)
                for sent in all_sentences:
                    gen.memory.store_sequence(sent)
                gen.selector.train(all_sentences)
                gen.is_trained = True
                print(f"✓ Generador listo ({len(all_sentences)} oraciones)\n")
        except Exception:
            pass
    
    if gen is None:
        print("❌ No hay datos para el generador. Descarga al menos una fuente:")
        print("   bash scripts/setup_corpus.sh")
        print("   python scripts/download_literature.py")
        print("   Luego: python scripts/train_final.py")
        return
    
    # Definir prompts por disciplina
    disciplinas = {
        "Política": [
            "El presidente",
            "El gobierno",
            "La oposición",
            "Las elecciones"
        ],
        "Economía": [
            "La economía",
            "El mercado",
            "La inflación",
            "Las exportaciones"
        ],
        "Ciencia": [
            "Los científicos",
            "El estudio",
            "La investigación",
            "Los resultados"
        ],
        "Tecnología": [
            "La tecnología",
            "El software",
            "La inteligencia artificial",
            "Los datos"
        ],
        "Cultura": [
            "La película",
            "El libro",
            "La música",
            "El arte"
        ]
    }
    
    # Generar ejemplos
    for disciplina, prompts in disciplinas.items():
        print("=" * 70)
        print(f"📚 {disciplina}")
        print("=" * 70)
        print()
        
        for prompt in prompts:
            print(f"Prompt: '{prompt}'")
            print("-" * 70)
            
            # Generar 2 variaciones
            for i in range(2):
                text = gen.generate(prompt, max_words=15, temperature=0.7)
                print(f"  {i+1}. {text}")
            
            print()
    
    print("=" * 70)
    print("✅ Demo completado")
    print("=" * 70)


if __name__ == "__main__":
    main()
