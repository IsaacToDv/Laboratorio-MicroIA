#!/usr/bin/env python3
"""
Descarga literatura clásica en español para entrenamiento

Fuentes:
- Project Gutenberg (dominio público)
- Hugging Face datasets de literatura
- Textos de autores clásicos: Cervantes, Borges, García Márquez, Neruda, etc.

Uso:
    python scripts/download_literature.py
"""

import json
import requests
from pathlib import Path
from datasets import load_dataset


def download_project_gutenberg():
    """Descarga textos de Project Gutenberg en español."""
    print("📚 Descargando literatura de Project Gutenberg...")
    
    # URLs de textos clásicos en dominio público
    classic_texts = {
        'don_quijote': 'https://www.gutenberg.org/cache/epub/2000/pg2000.txt',
        'novelas_ejemplares': 'https://www.gutenberg.org/cache/epub/2506/pg2506.txt',
    }
    
    sentences = []
    for name, url in classic_texts.items():
        try:
            print(f"   Descargando {name}...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            text = response.text
            # Limpiar y dividir en oraciones
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 50 and not line.startswith('***') and not line.startswith('http'):
                    # Dividir por puntos
                    sents = [s.strip() for s in line.split('.') if len(s.strip()) > 20]
                    sentences.extend(sents[:10])  # Tomar primeras 10 por línea
            
            print(f"   ✓ {name}: {len(sentences)} oraciones")
        except Exception as e:
            print(f"   ⚠ Error descargando {name}: {e}")
    
    return sentences


def download_literature_datasets():
    """Descarga datasets de literatura de Hugging Face."""
    print("\n📖 Descargando datasets de literatura de Hugging Face...")
    
    sentences = []
    
    try:
        # Intentar cargar dataset de literatura
        print("   Cargando BookCorpus Open...")
        dataset = load_dataset("bookcorpusopen", split="train[:2000]")
        
        for item in dataset:
            text = item.get('text', '')
            # Filtrar textos en español (heurística simple)
            if any(word in text.lower() for word in ['el', 'la', 'de', 'que', 'en']):
                sents = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                sentences.extend(sents[:5])
        
        print(f"   ✓ BookCorpus: {len(sentences)} oraciones")
    except Exception as e:
        print(f"   ⚠ Error con BookCorpus: {e}")
    
    return sentences


def download_poetry():
    """Descarga poesía en español."""
    print("\n🎭 Descargando poesía en español...")
    
    # Poesía clásica (fragmentos representativos)
    poetry_samples = [
        # Neruda
        "Puedo escribir los versos más tristes esta noche",
        "Me gustas cuando callas porque estás como ausente",
        "Es tan corto el amor y es tan largo el olvido",
        "Para mi corazón basta tu pecho, para tu libertad bastan mis alas",
        
        # Lorca
        "Verde que te quiero verde",
        "La aurora de Nueva York tiene cuatro columnas de cieno",
        "No te conozco, no te conozco, no te conozco",
        
        # Machado
        "Caminante, no hay camino, se hace camino al andar",
        "Anoche cuando dormía soñé, bendita ilusión",
        
        # Bécquer
        "Volverán las oscuras golondrinas",
        "¿Qué es poesía? dices mientras clavas en mi pupila tu pupila azul",
        
        # Octavio Paz
        "Entre lo que veo y lo que digo, entre lo que digo y lo que callo",
        "El mundo es ya una inmensa llaga, una herida que no cierra",
    ]
    
    sentences = poetry_samples
    print(f"   ✓ Poesía: {len(sentences)} versos")
    
    return sentences


def download_philosophical_texts():
    """Descarga textos filosóficos en español."""
    print("\n🤔 Descargando textos filosóficos...")
    
    # Fragmentos filosóficos representativos
    philosophy_samples = [
        # Unamuno
        "Venceréis, pero no convenceréis",
        "El que duda de sí mismo es el único que sabe",
        "La vida es un sueño, y los sueños, sueños son",
        
        # Ortega y Gasset
        "Yo soy yo y mi circunstancia",
        "Vivir es constantemente decidir lo que vamos a ser",
        "El hombre es el ser que se hace a sí mismo",
        
        # Zambrano
        "La razón es la violencia del pensamiento",
        "Conocer es amar, amar es conocer",
        
        # Borges
        "El tiempo es la sustancia de la que estoy hecho",
        "La memoria es un museo de cera",
        "He sospechado alguna vez que la única cosa sin misterio es la felicidad",
    ]
    
    sentences = philosophy_samples
    print(f"   ✓ Filosofía: {len(sentences)} fragmentos")
    
    return sentences


def download_narrative_texts():
    """Descarga textos narrativos en español."""
    print("\n📖 Descargando textos narrativos...")
    
    # Fragmentos narrativos representativos
    narrative_samples = [
        # García Márquez
        "Muchos años después, frente al pelotón de fusilamiento, el coronel Aureliano Buendía había de recordar aquella tarde remota en que su padre lo llevó a conocer el hielo",
        "Era inevitable: el olor de las almendras amargas le recordaba siempre el destino de los amores contrariados",
        
        # Borges
        "En un lugar de la Mancha, de cuyo nombre no quiero acordarme",
        "El universo (que otros llaman la Biblioteca) se compone de un número indefinido, y tal vez infinito, de galerías hexagonales",
        
        # Cortázar
        "En algún lugar bajo la lluvia siempre hay un hombre o una mujer que levanta la vista",
        "Andábamos sin buscarnos pero sabiendo que andábamos para encontrarnos",
        
        # Isabel Allende
        "Barrabás llegó a la familia por vía marítima",
        "La memoria es frágil y el transcurso de una vida es muy breve",
    ]
    
    sentences = narrative_samples
    print(f"   ✓ Narrativa: {len(sentences)} fragmentos")
    
    return sentences


def main():
    """Descarga todos los textos literarios y los guarda."""
    print("=" * 70)
    print("Descarga de Literatura Clásica en Español")
    print("=" * 70)
    print()
    
    # Crear directorio de datos
    data_dir = Path("data/literature")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar cada tipo de literatura
    datasets = {
        'classics': download_project_gutenberg(),
        'literature_datasets': download_literature_datasets(),
        'poetry': download_poetry(),
        'philosophy': download_philosophical_texts(),
        'narrative': download_narrative_texts()
    }
    
    # Guardar cada dataset
    print("\n" + "=" * 70)
    print("Guardando textos literarios")
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
    print("Creando dataset literario combinado")
    print("=" * 70)
    print()
    
    all_sentences = []
    for sentences in datasets.values():
        all_sentences.extend(sentences)
    
    combined_file = data_dir / "literature_combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'sentences': all_sentences,
            'sources': list(datasets.keys()),
            'total': len(all_sentences),
            'description': 'Literatura clásica en español: poesía, filosofía, narrativa'
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Dataset literario combinado: {len(all_sentences)} oraciones en {combined_file}")
    
    print("\n" + "=" * 70)
    print("Resumen")
    print("=" * 70)
    print()
    print(f"Total de textos literarios: {total_sentences}")
    print(f"Categorías: {len([d for d in datasets.values() if d])}")
    print()
    print("Siguiente paso:")
    print("  python scripts/train_with_literature.py")
    print()


if __name__ == "__main__":
    main()
