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
    
    # Agregar scripts al path
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    
    # Cargar generador desde cualquier fuente disponible
    print("Cargando generador...")
    
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
                gen = TextGenerator(dim=10000)
                gen.train_from_sentences(sentences)
                print(f"✓ Modelo pre-entrenado cargado ({len(sentences)} oraciones)\n")
                _run_demo(gen)
                return
        except Exception:
            pass
    
    # Intentar 2: Corpus UD AnCora
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    if Path(corpus_path).exists():
        gen = TextGenerator(dim=10000)
        gen.train_from_corpus(corpus_path, max_sentences=3000)
        print("✓ Generador listo (UD AnCora)\n")
        _run_demo(gen)
        return
    
    # Intentar 3: data_loader
    try:
        from data_loader import ensure_data_available
        all_sentences, sources = ensure_data_available(min_sentences=50)
        if all_sentences:
            gen = TextGenerator(dim=10000)
            gen.train_from_sentences(all_sentences)
            print(f"✓ Generador listo ({len(all_sentences)} oraciones)\n")
            _run_demo(gen)
            return
    except Exception:
        pass
    
    print("❌ No hay datos para el generador. Descarga al menos una fuente:")
    print("   bash scripts/setup_corpus.sh")
    print("   python scripts/download_literature.py")
    print("   Luego: python scripts/train_final.py")


def _run_demo(gen):
    """Ejecuta el demo con el generador ya cargado."""
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
