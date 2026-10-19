#!/usr/bin/env python3
"""
Entrenamiento final del generador con múltiples datasets

Combina:
- UD Spanish-AnCora (corpus lingüístico)
- Wikipedia (conocimiento general)
- Noticias (actualidad)
- Textos legales
- Textos científicos
- Literatura

Uso:
    python scripts/train_final.py
"""

import json
import sys
from pathlib import Path
from conllu import parse_incr

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from generation import TextGenerator


def load_ud_ancora(corpus_path: str, max_sentences: int = 3000):
    """Carga oraciones del corpus UD Spanish-AnCora."""
    print(f"📖 Cargando UD Spanish-AnCora...")
    
    sentences = []
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, sent in enumerate(parse_incr(f)):
                if i >= max_sentences:
                    break
                
                words = [token['form'] for token in sent 
                        if not isinstance(token['id'], tuple)]
                
                if len(words) >= 3:
                    sentences.append(words)
        
        print(f"   ✓ {len(sentences)} oraciones de UD AnCora")
        
    except FileNotFoundError:
        print(f"   ⚠ Corpus no encontrado en {corpus_path}")
    
    return sentences


def load_json_dataset(json_path: str, max_sentences: int = 2000):
    """Carga oraciones de un dataset JSON."""
    print(f"📚 Cargando {Path(json_path).name}...")
    
    sentences = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_sentences = data.get('sentences', [])
        
        for text in raw_sentences[:max_sentences]:
            # Tokenizar por espacios
            words = text.strip().split()
            if len(words) >= 3 and len(words) <= 50:
                sentences.append(words)
        
        print(f"   ✓ {len(sentences)} oraciones de {Path(json_path).stem}")
        
    except FileNotFoundError:
        print(f"   ⚠ Dataset no encontrado: {json_path}")
    except Exception as e:
        print(f"   ⚠ Error cargando {json_path}: {e}")
    
    return sentences


def main():
    """Entrena el generador con todos los datasets disponibles."""
    print("=" * 70)
    print("Entrenamiento Final del Generador")
    print("=" * 70)
    print()
    
    # Recopilar todas las oraciones
    all_sentences = []
    
    # 1. UD Spanish-AnCora (corpus principal)
    ud_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    ud_sentences = load_ud_ancora(ud_path, max_sentences=3000)
    all_sentences.extend(ud_sentences)
    
    # 2. Datasets de Hugging Face
    datasets_dir = Path("data/datasets")
    if datasets_dir.exists():
        for json_file in datasets_dir.glob("*.json"):
            if json_file.name == "combined.json":
                continue  # Saltar el combinado
            
            dataset_sentences = load_json_dataset(str(json_file), max_sentences=1500)
            all_sentences.extend(dataset_sentences)
    else:
        print("\n⚠ Directorio data/datasets no encontrado")
        print("  Ejecuta primero: python scripts/download_datasets.py")
    
    print("\n" + "=" * 70)
    print(f"Total de oraciones: {len(all_sentences)}")
    print("=" * 70)
    print()
    
    if len(all_sentences) < 100:
        print("❌ No hay suficientes datos para entrenar")
        print("   Asegúrate de tener el corpus UD AnCora o ejecuta:")
        print("   python scripts/download_datasets.py")
        return
    
    # Entrenar generador
    print("🧠 Entrenando generador...")
    gen = TextGenerator(dim=10000)
    
    # Entrenar memoria HDC
    print("   Almacenando secuencias en memoria HDC...")
    for sent in all_sentences:
        gen.memory.store_sequence(sent)
    
    print(f"   ✓ {len(gen.memory.sequence_memory)} secuencias almacenadas")
    
    # Entrenar selector FEP
    print("   Entrenando selector FEP...")
    gen.selector.train(all_sentences)
    
    print(f"   ✓ Selector FEP entrenado")
    print(f"   ✓ Vocabulario: {len(gen.selector.unigram_counts)} palabras")
    
    gen.is_trained = True
    
    # Guardar modelo entrenado
    print("\n" + "=" * 70)
    print("Guardando modelo entrenado")
    print("=" * 70)
    print()
    
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Guardar configuración
    config = {
        'dim': gen.memory.dim,
        'total_sentences': len(all_sentences),
        'vocabulary_size': len(gen.selector.unigram_counts),
        'sequences_stored': len(gen.memory.sequence_memory)
    }
    
    config_file = model_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuración guardada en {config_file}")
    
    # Probar generación
    print("\n" + "=" * 70)
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
        "Según el estudio"
    ]
    
    for prompt in test_prompts:
        text = gen.generate(prompt, max_words=15, temperature=0.6)
        print(f"Prompt: {prompt}")
        print(f"  → {text}")
        print()
    
    print("=" * 70)
    print("✅ Entrenamiento completado")
    print("=" * 70)
    print()
    print("Siguiente paso:")
    print("  python demo.py")
    print()


if __name__ == "__main__":
    main()
