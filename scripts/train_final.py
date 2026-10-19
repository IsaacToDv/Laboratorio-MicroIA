#!/usr/bin/env python3
"""
Entrenamiento del generador de texto

Busca automáticamente TODAS las fuentes de datos disponibles:
- UD Spanish-AnCora (corpus principal)
- Datasets de Hugging Face (Wikipedia, noticias, legal, científico)
- Literatura clásica (Don Quijote, poesía, filosofía, narrativa)

Si no hay datos, da instrucciones claras de qué descargar.

Uso:
    python scripts/train_final.py
"""

import json
import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Agregar scripts al path para importar data_loader
sys.path.insert(0, str(Path(__file__).parent))

from generation import TextGenerator
from data_loader import ensure_data_available, Colors


def main():
    """Entrena el generador con todos los datos disponibles."""
    print()
    print("=" * 70)
    print("Entrenamiento del Generador de Texto")
    print("=" * 70)
    
    # Cargar todos los datos disponibles
    all_sentences, sources = ensure_data_available(min_sentences=50)
    
    if all_sentences is None:
        return
    
    print()
    print("=" * 70)
    print(f"Total de oraciones: {len(all_sentences)}")
    print("=" * 70)
    print()
    
    # Entrenar generador (usa train_from_sentences para activar todos los componentes)
    print(f"{Colors.BLUE}🧠 Entrenando generador (Deep Cognition)...{Colors.END}")
    gen = TextGenerator(dim=20000)
    gen.train_from_sentences(all_sentences)
    
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
        'sources': sources
    }
    
    config_file = model_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"{Colors.GREEN}✓{Colors.END} Configuración guardada en {config_file}")
    
    # Guardar el corpus completo para uso posterior por demo.py y ejemplos
    corpus_file = model_dir / "trained_corpus.json"
    with open(corpus_file, 'w', encoding='utf-8') as f:
        json.dump({
            'sentences': [' '.join(s) for s in all_sentences],
            'sources': sources
        }, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}✓{Colors.END} Corpus de entrenamiento guardado en {corpus_file}")
    
    # Probar generación
    print()
    print("=" * 70)
    print("Probando generación")
    print("=" * 70)
    print()
    
    test_prompts = [
        "El presidente",
        "La economía",
        "En la ciudad",
        "Los científicos",
        "El gobierno",
        "La tecnología",
    ]
    
    for prompt in test_prompts:
        text = gen.generate(prompt, max_words=15, temperature=0.6)
        print(f"  {Colors.BLUE}{prompt}{Colors.END} → {text}")
    
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
