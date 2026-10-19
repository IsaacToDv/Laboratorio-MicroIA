#!/usr/bin/env python3
"""
Entrenamiento del generador con énfasis en literatura clásica

Busca automáticamente TODAS las fuentes de datos disponibles:
- UD Spanish-AnCora (corpus lingüístico)
- Datasets de Hugging Face (multi-disciplina)
- Literatura clásica (poesía, filosofía, narrativa)

Si no hay datos, da instrucciones claras de qué descargar.

Uso:
    python scripts/train_with_literature.py
"""

import json
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent))

from generation import TextGenerator, load_identity
from data_loader import ensure_data_available, Colors


def main():
    """Entrena el generador con énfasis en literatura."""
    print()
    print("=" * 70)
    print("Entrenamiento con Literatura Clásica")
    print("=" * 70)
    
    # Cargar todos los datos disponibles
    all_sentences, sources = ensure_data_available(min_sentences=50)
    
    if all_sentences is None:
        return
    
    # Verificar si hay literatura
    has_literature = any('literature' in k or 'classics' in k for k in sources)
    
    print()
    print("=" * 70)
    print(f"Total de oraciones: {len(all_sentences)}")
    print("=" * 70)
    
    if has_literature:
        print(f"{Colors.GREEN}✓ Incluye literatura clásica{Colors.END}")
    else:
        print(f"{Colors.YELLOW}⚠ Sin literatura clásica (solo datos generales){Colors.END}")
        print(f"  Para mejor calidad literaria, ejecuta: python scripts/download_literature.py")
    
    print()
    
    # Entrenar generador
    print(f"{Colors.BLUE}🧠 Entrenando generador con literatura...{Colors.END}")
    gen = TextGenerator(dim=10000)
    
    # Entrenar memoria HDC
    print("   Almacenando secuencias en memoria HDC...")
    for sent in all_sentences:
        gen.memory.store_sequence(sent)
    
    print(f"   {Colors.GREEN}✓{Colors.END} {len(gen.memory.sequence_memory)} secuencias almacenadas")
    
    # Entrenar selector FEP
    print("   Entrenando selector FEP...")
    gen.selector.train(all_sentences)
    
    print(f"   {Colors.GREEN}✓{Colors.END} Selector FEP entrenado")
    print(f"   {Colors.GREEN}✓{Colors.END} Vocabulario: {len(gen.selector.unigram_counts)} palabras")
    
    gen.is_trained = True
    
    # Guardar modelo entrenado
    print()
    print("=" * 70)
    print("Guardando modelo entrenado")
    print("=" * 70)
    print()
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    config = {
        'dim': gen.memory.dim,
        'total_sentences': len(all_sentences),
        'vocabulary_size': len(gen.selector.unigram_counts),
        'sequences_stored': len(gen.memory.sequence_memory),
        'includes_literature': has_literature,
        'sources': sources
    }
    
    config_file = model_dir / "config_literature.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"{Colors.GREEN}✓{Colors.END} Configuración guardada en {config_file}")
    
    # Guardar el corpus completo para uso posterior
    corpus_file = model_dir / "trained_corpus.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump({
            'sentences': [' '.join(s) for s in all_sentences],
            'sources': sources
        }, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}✓{Colors.END} Corpus de entrenamiento guardado en {corpus_file}")
    
    # Probar generación con identidad
    print()
    print("=" * 70)
    print("Probando generación con identidades")
    print("=" * 70)
    
    identities_to_test = ['poeta', 'filosofo', 'narrador']
    
    for identity_name in identities_to_test:
        identity = load_identity(identity_name)
        gen.set_identity(identity)
        
        print(f"\n{Colors.BLUE}🎭 {identity.name} ({identity.style}){Colors.END}")
        print("-" * 70)
        
        test_prompts = ["El amor", "El tiempo", "La vida"]
        
        for prompt in test_prompts:
            text = gen.generate(prompt, max_words=20)
            print(f"  {prompt}... → {text}")
    
    print()
    print("=" * 70)
    print(f"{Colors.GREEN}✅ Entrenamiento completado{Colors.END}")
    print("=" * 70)
    print()
    print(f"{Colors.BLUE}Próximos pasos:{Colors.END}")
    print(f"  {Colors.YELLOW}python demo.py{Colors.END}                          # Demo interactivo")
    print(f"  {Colors.YELLOW}python examples/identidades.py{Colors.END}            # Generar con identidades")
    print(f"  {Colors.YELLOW}python examples/generacion_por_disciplina.py{Colors.END} # Generar por disciplina")
    print()


if __name__ == "__main__":
    main()
