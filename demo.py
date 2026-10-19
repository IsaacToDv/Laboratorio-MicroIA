#!/usr/bin/env python3
"""
Demo interactivo del Laboratorio de Micro-IA Biomimética

Este script muestra las capacidades principales del proyecto:
1. Generación de texto con HDC + FEP
2. Análisis morfológico (POS tagging)
3. Desambiguación semántica

Uso:
    python demo.py
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generation import TextGenerator


def print_header(text: str):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")


def demo_generacion():
    """Demuestra la generación de texto."""
    print_header("🧬 Demo: Generación de Texto con HDC + Memoria + FEP")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus no encontrado en:", corpus_path)
        print("\nDescarga el corpus con:")
        print("  mkdir -p data/corpus && cd data/corpus")
        print("  git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git")
        return
    
    print("Cargando generador...")
    gen = TextGenerator(dim=10000)
    gen.train_from_corpus(corpus_path, max_sentences=2000)
    
    print("✓ Generador entrenado\n")
    
    # Ejemplos de generación
    prompts = [
        "El presidente",
        "La economía",
        "En la ciudad",
        "Los científicos",
        "El gobierno"
    ]
    
    print("Ejemplos de generación:\n")
    for prompt in prompts:
        print(f"Prompt: '{prompt}'")
        print("-" * 70)
        
        # Generar 3 variaciones
        for i in range(3):
            text = gen.generate(prompt, max_words=15, temperature=0.6)
            print(f"  {i+1}. {text}")
        
        print()
    
    # Modo interactivo
    print("\n" + "=" * 70)
    print("🎮 Modo interactivo")
    print("=" * 70)
    print("\nEscribe un prompt y presiona Enter para generar texto.")
    print("Escribe 'salir' para terminar.\n")
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            if prompt.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not prompt:
                continue
            
            text = gen.generate(prompt, max_words=20, temperature=0.7)
            print(f"→ {text}\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")
    
    print("\n¡Gracias por probar el demo!")


def demo_comparacion_temperaturas():
    """Compara generación con diferentes temperaturas."""
    print_header("🌡️ Demo: Efecto de la Temperatura")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus no encontrado")
        return
    
    gen = TextGenerator(dim=10000)
    gen.train_from_corpus(corpus_path, max_sentences=1000)
    
    prompt = "El acuerdo"
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    print(f"Prompt: '{prompt}'\n")
    
    for temp in temperatures:
        print(f"Temperatura {temp}:")
        print("-" * 70)
        
        for i in range(2):
            text = gen.generate(prompt, max_words=12, temperature=temp)
            print(f"  {text}")
        
        print()
    
    print("💡 Temperatura baja (0.1) = más determinista y conservador")
    print("💡 Temperatura alta (2.0) = más creativo y variado")


def main():
    """Menú principal del demo."""
    print_header("🧬 Laboratorio de Micro-IA Biomimética - Demo")
    
    print("Este demo muestra las capacidades del proyecto:\n")
    print("  1. Generación de texto con HDC + Memoria + FEP")
    print("  2. Comparación de temperaturas")
    print("  3. Salir")
    
    while True:
        try:
            choice = input("\nSelecciona una opción (1-3): ").strip()
            
            if choice == '1':
                demo_generacion()
                break
            elif choice == '2':
                demo_comparacion_temperaturas()
                break
            elif choice == '3':
                print("¡Hasta luego!")
                break
            else:
                print("Opción no válida. Intenta de nuevo.")
                
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break


if __name__ == "__main__":
    main()
