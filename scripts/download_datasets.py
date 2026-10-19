#!/usr/bin/env python3
"""
Descarga datasets de Hugging Face en español para entrenamiento

Datasets incluidos:
- Wikipedia en español (conocimiento general)
- News en español (actualidad)
- Legal (textos jurídicos)
- Científico (papers y artículos)

Uso:
    python scripts/download_datasets.py
"""

import json
from pathlib import Path
from datasets import load_dataset


def download_wikipedia():
    """Descarga artículos de Wikipedia en español."""
    print("📚 Descargando Wikipedia en español...")
    
    try:
        dataset = load_dataset("wikipedia", "20220301.es", split="train[:5000]")
        
        articles = []
        for item in dataset:
            text = item['text']
            # Limpiar y dividir en oraciones
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            articles.extend(sentences[:10])  # Tomar primeras 10 oraciones por artículo
        
        print(f"   ✓ {len(articles)} oraciones de Wikipedia")
        return articles
        
    except Exception as e:
        print(f"   ⚠ Error descargando Wikipedia: {e}")
        return []


def download_news():
    """Descarga noticias en español."""
    print("📰 Descargando noticias en español...")
    
    try:
        # Usar dataset de noticias
        dataset = load_dataset("mlsum", "es", split="train[:2000]")
        
        articles = []
        for item in dataset:
            text = item['text']
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            articles.extend(sentences[:5])
        
        print(f"   ✓ {len(articles)} oraciones de noticias")
        return articles
        
    except Exception as e:
        print(f"   ⚠ Error descargando noticias: {e}")
        return []


def download_legal():
    """Descarga textos legales en español."""
    print("⚖️  Descargando textos legales...")
    
    try:
        # Dataset legal español
        dataset = load_dataset("Iker/LegalES", split="train[:1000]")
        
        sentences = []
        for item in dataset:
            text = item['text']
            # Dividir en oraciones
            sents = [s.strip() for s in text.split('.') if len(s.strip()) > 30]
            sentences.extend(sents[:5])
        
        print(f"   ✓ {len(sentences)} oraciones legales")
        return sentences
        
    except Exception as e:
        print(f"   ⚠ Error descargando textos legales: {e}")
        return []


def download_scientific():
    """Descarga textos científicos en español."""
    print("🔬 Descargando textos científicos...")
    
    try:
        # Usar subset científico de MLSUM
        dataset = load_dataset("mlsum", "es", split="train[:1000]")
        
        sentences = []
        for item in dataset:
            text = item.get('text', '')
            if 'ciencia' in text.lower() or 'tecnología' in text.lower():
                sents = [s.strip() for s in text.split('.') if len(s.strip()) > 25]
                sentences.extend(sents[:5])
        
        print(f"   ✓ {len(sentences)} oraciones científicas")
        return sentences
        
    except Exception as e:
        print(f"   ⚠ Error descargando textos científicos: {e}")
        return []


def download_literature():
    """Descarga literatura en español."""
    print("📖 Descargando literatura en español...")
    
    try:
        # Usar dataset de libros
        dataset = load_dataset("BookCorpus", split="train[:1000]")
        
        sentences = []
        for item in dataset:
            text = item.get('text', '')
            sents = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
            sentences.extend(sents[:5])
        
        print(f"   ✓ {len(sentences)} oraciones literarias")
        return sentences
        
    except Exception as e:
        print(f"   ⚠ Error descargando literatura: {e}")
        return []


def main():
    """Descarga todos los datasets y los guarda."""
    print("=" * 70)
    print("Descarga de Datasets de Hugging Face")
    print("=" * 70)
    print()
    
    # Crear directorio de datos
    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar cada dataset
    datasets = {
        'wikipedia': download_wikipedia(),
        'news': download_news(),
        'legal': download_legal(),
        'scientific': download_scientific(),
        'literature': download_literature()
    }
    
    # Guardar cada dataset
    print("\n" + "=" * 70)
    print("Guardando datasets")
    print("=" * 70)
    print()
    
    total_sentences = 0
    for name, sentences in datasets.items():
        if sentences:
            output_file = data_dir / f"{name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'sentences': sentences}, f, ensure_ascii=False, indent=2)
            print(f"✓ {name}: {len(sentences)} oraciones guardadas en {output_file}")
            total_sentences += len(sentences)
    
    # Crear dataset combinado
    print("\n" + "=" * 70)
    print("Creando dataset combinado")
    print("=" * 70)
    print()
    
    all_sentences = []
    for sentences in datasets.values():
        all_sentences.extend(sentences)
    
    combined_file = data_dir / "combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'sentences': all_sentences,
            'sources': list(datasets.keys()),
            'total': len(all_sentences)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Dataset combinado: {len(all_sentences)} oraciones en {combined_file}")
    
    print("\n" + "=" * 70)
    print("Resumen")
    print("=" * 70)
    print()
    print(f"Total de oraciones descargadas: {total_sentences}")
    print(f"Datasets: {len([d for d in datasets.values() if d])}")
    print()
    print("Siguiente paso:")
    print("  python scripts/train_final.py")
    print()


if __name__ == "__main__":
    main()
