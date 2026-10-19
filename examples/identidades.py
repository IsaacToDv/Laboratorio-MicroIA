#!/usr/bin/env python3
"""
Demo de generación con diferentes identidades de IA

Muestra cómo el generador puede adoptar diferentes personalidades:
- Poeta (Neruda, Lorca, Paz)
- Filósofo (Unamuno, Ortega y Gasset, Zambrano)
- Narrador (García Márquez, Borges, Cortázar)
- Ensayista (Borges, Paz, Vargas Llosa)
- Contemporáneo (Bolaño, Marías, Grandes)

Uso:
    python examples/identidades.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import TextGenerator, load_identity, list_identities


def main():
    """Demuestra la generación con diferentes identidades."""
    print("=" * 70)
    print("Generación de Texto con Identidades de IA")
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
    
    # Listar identidades disponibles
    print("=" * 70)
    print("Identidades disponibles")
    print("=" * 70)
    print()
    
    for identity_name in list_identities():
        identity = load_identity(identity_name)
        print(f"• {identity.name}: {identity.description}")
        print(f"  Autores: {', '.join(identity.top_authors[:3])}")
        print()
    
    # Generar con diferentes identidades
    prompts = [
        "El amor",
        "El tiempo",
        "La vida",
        "La muerte",
        "La memoria"
    ]
    
    for identity_name in list_identities():
        identity = load_identity(identity_name)
        
        print("=" * 70)
        print(f"🎭 Identidad: {identity.name}")
        print("=" * 70)
        print(f"Estilo: {identity.style}")
        print(f"Tono: {identity.emotional_tone}")
        print(f"Autores de referencia: {', '.join(identity.top_authors)}")
        print()
        
        # Configurar identidad
        gen.set_identity(identity)
        
        # Generar ejemplos
        for prompt in prompts:
            print(f"Prompt: '{prompt}'")
            print("-" * 70)
            
            # Generar 2 variaciones
            for i in range(2):
                text = gen.generate(prompt, max_words=20)
                print(f"  {i+1}. {text}")
            
            print()
        
        input("Presiona Enter para continuar con la siguiente identidad...")
        print()
    
    print("=" * 70)
    print("✅ Demo completado")
    print("=" * 70)


if __name__ == "__main__":
    main()
