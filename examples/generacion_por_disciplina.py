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
    
    # Cargar generador
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus no encontrado. Ejecuta primero:")
        print("   bash scripts/setup_corpus.sh")
        return
    
    print("Cargando generador...")
    gen = TextGenerator(dim=10000)
    gen.train_from_corpus(corpus_path, max_sentences=3000)
    print("✓ Generador listo\n")
    
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
