#!/usr/bin/env python3
"""
Descarga datasets en español para entrenamiento

Descarga datos de múltiples fuentes:
- Wikipedia en español (via Hugging Face o descarga directa)
- Noticias y textos generales
- Textos legales
- Textos científicos

Si algún dataset falla, continúa con los demás. El proyecto funciona
incluso sin estos datos (usa UD AnCora + literatura como fallback).

Uso:
    python scripts/download_datasets.py
"""

import json
import re
from pathlib import Path

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def clean_text(text):
    """Limpia y divide texto en oraciones."""
    if not text or not isinstance(text, str):
        return []
    
    # Eliminar URLs, referencias, marcadores especiales
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'==.*?==', '', text)
    
    # Dividir en oraciones
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filtrar oraciones válidas
    valid = []
    for s in sentences:
        s = s.strip()
        if len(s) >= 20 and len(s) <= 500:
            # Evitar oraciones con demasiados caracteres especiales
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in s) / max(len(s), 1)
            if alpha_ratio > 0.7:
                valid.append(s)
    
    return valid


def download_wikipedia():
    """Descarga artículos de Wikipedia en español."""
    print(f"{Colors.BLUE}📚 Descargando Wikipedia en español...{Colors.END}")
    
    try:
        from datasets import load_dataset
        
        # Intentar diferentes nombres del dataset
        dataset_names = [
            ("wikimedia/wikipedia", "20231101.es"),
            ("wikipedia", "20220301.es"),
        ]
        
        for name, config in dataset_names:
            try:
                kwargs = {"split": "train[:5000]"}
                if name == "wikimedia/wikipedia":
                    kwargs["trust_remote_code"] = True
                
                dataset = load_dataset(name, config, **kwargs)
                
                articles = []
                for item in dataset:
                    text = item.get('text', '')
                    sentences = clean_text(text)
                    articles.extend(sentences[:10])
                
                if articles:
                    print(f"   {Colors.GREEN}✓ {len(articles)} oraciones de Wikipedia{Colors.END}")
                    return articles
            except Exception:
                continue
        
        print(f"   {Colors.YELLOW}⚠ Wikipedia no disponible, usando textos de respaldo{Colors.END}")
        return get_fallback_text("wikipedia")
        
    except ImportError:
        print(f"   {Colors.YELLOW}⚠ Módulo 'datasets' no disponible{Colors.END}")
        return get_fallback_text("wikipedia")


def download_news():
    """Descarga noticias en español."""
    print(f"{Colors.BLUE}📰 Descargando noticias en español...{Colors.END}")
    
    try:
        from datasets import load_dataset
        
        dataset_names = [
            ("mlsum", "es"),
            ("PereLluis/mlsum-es-clean", None),
        ]
        
        for name, config in dataset_names:
            try:
                kwargs = {"split": "train[:2000]"}
                if config:
                    dataset = load_dataset(name, config, **kwargs)
                else:
                    dataset = load_dataset(name, **kwargs)
                
                articles = []
                for item in dataset:
                    text = item.get('text', '') or item.get('summary', '')
                    sentences = clean_text(text)
                    articles.extend(sentences[:5])
                
                if articles:
                    print(f"   {Colors.GREEN}✓ {len(articles)} oraciones de noticias{Colors.END}")
                    return articles
            except Exception:
                continue
        
        print(f"   {Colors.YELLOW}⚠ Noticias no disponibles, usando textos de respaldo{Colors.END}")
        return get_fallback_text("news")
        
    except ImportError:
        print(f"   {Colors.YELLOW}⚠ Módulo 'datasets' no disponible{Colors.END}")
        return get_fallback_text("news")


def download_legal():
    """Descarga textos legales en español."""
    print(f"{Colors.BLUE}⚖️  Descargando textos legales...{Colors.END}")
    
    try:
        from datasets import load_dataset
        
        dataset_names = [
            ("Iker/LegalES", None),
            ("PlanTL-GOB-ES/LegalES", None),
        ]
        
        for name, config in dataset_names:
            try:
                dataset = load_dataset(name, split="train[:1000]")
                
                sentences = []
                for item in dataset:
                    text = item.get('text', '') or item.get('sentence', '')
                    sents = clean_text(text)
                    sentences.extend(sents[:5])
                
                if sentences:
                    print(f"   {Colors.GREEN}✓ {len(sentences)} oraciones legales{Colors.END}")
                    return sentences
            except Exception:
                continue
        
        print(f"   {Colors.YELLOW}⚠ Textos legales no disponibles, usando textos de respaldo{Colors.END}")
        return get_fallback_text("legal")
        
    except ImportError:
        print(f"   {Colors.YELLOW}⚠ Módulo 'datasets' no disponible{Colors.END}")
        return get_fallback_text("legal")


def download_scientific():
    """Descarga textos científicos en español."""
    print(f"{Colors.BLUE}🔬 Descargando textos científicos...{Colors.END}")
    
    try:
        from datasets import load_dataset
        
        dataset_names = [
            ("scientific_papers", "arxiv"),
            ("PlanTL-GOB-ES/bsc_te3c", None),
        ]
        
        for name, config in dataset_names:
            try:
                kwargs = {"split": "train[:1000]"}
                if config:
                    dataset = load_dataset(name, config, **kwargs)
                else:
                    dataset = load_dataset(name, **kwargs)
                
                sentences = []
                for item in dataset:
                    text = item.get('text', '') or item.get('article', '')
                    sents = clean_text(text)
                    # Filtrar solo textos en español
                    spanish_sents = [s for s in sents if any(w in s.lower() for w in ['el', 'la', 'de', 'en', 'del', 'que', 'los'])]
                    sentences.extend(spanish_sents[:5])
                
                if sentences:
                    print(f"   {Colors.GREEN}✓ {len(sentences)} oraciones científicas{Colors.END}")
                    return sentences
            except Exception:
                continue
        
        print(f"   {Colors.YELLOW}⚠ Textos científicos no disponibles, usando textos de respaldo{Colors.END}")
        return get_fallback_text("scientific")
        
    except ImportError:
        print(f"   {Colors.YELLOW}⚠ Módulo 'datasets' no disponible{Colors.END}")
        return get_fallback_text("scientific")


def get_fallback_text(category):
    """Proporciona textos de respaldo cuando los datasets no están disponibles."""
    fallback_texts = {
        "wikipedia": [
            "La inteligencia artificial es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana.",
            "El aprendizaje automático es un subcampo de la inteligencia artificial que permite a las máquinas aprender de datos sin ser programadas explícitamente.",
            "Las redes neuronales artificiales están inspiradas en la estructura del cerebro humano y son fundamentales en el aprendizaje profundo.",
            "El procesamiento del lenguaje natural permite a las computadoras entender, interpretar y generar texto en lenguaje humano.",
            "La computación en la nube ofrece recursos de procesamiento y almacenamiento accesibles a través de internet.",
            "La ciencia de datos combina estadística, programación y conocimiento del dominio para extraer información de grandes conjuntos de datos.",
            "El internet de las cosas conecta dispositivos cotidianos a internet, permitiendo la recolección y el intercambio de datos.",
            "La realidad virtual crea entornos simulados que pueden ser explorados e interactuados por un usuario.",
            "El big data se refiere a conjuntos de datos demasiado grandes y complejos para ser procesados por herramientas tradicionales.",
            "La ciberseguridad protege los sistemas informáticos, las redes y los datos contra ataques y accesos no autorizados.",
            "La memoria humana funciona mediante la creación de asociaciones entre conceptos y experiencias previas.",
            "El cerebro humano contiene aproximadamente cien mil millones de neuronas interconectadas entre sí.",
            "La plasticidad neuronal permite al cerebro adaptarse y reorganizarse a lo largo de toda la vida.",
            "Los neurotransmisores son sustancias químicas que transmiten señales entre las neuronas del sistema nervioso.",
            "La conciencia es uno de los fenómenos más estudiados y menos comprendidos de la neurociencia moderna.",
        ],
        "news": [
            "El gobierno anunció nuevas medidas económicas para impulsar el crecimiento del país en los próximos meses.",
            "Los investigadores han descubierto una nueva especie de dinosaurio en la Patagonia argentina.",
            "La economía global enfrenta desafíos significativos debido a las tensiones comerciales entre las principales potencias.",
            "El cambio climático está provocando eventos meteorológicos extremos en diversas regiones del planeta.",
            "Las elecciones presidenciales se celebrarán el próximo mes con una alta participación esperada.",
            "El mercado bursátil experimentó una caída significativa debido a las preocupaciones sobre la inflación.",
            "Los científicos advierten que la biodiversidad está disminuyendo a un ritmo alarmante en todo el mundo.",
            "La tecnología blockchain está transformando la manera en que se realizan las transacciones financieras.",
            "El presidente firmó un acuerdo internacional para combatir el cambio climático y reducir las emisiones.",
            "Los expertos en salud pública recomiendan la vacunación como medida preventiva contra enfermedades infecciosas.",
            "La inflación acumulada durante el último año supera las expectativas de los analistas económicos.",
            "El congreso aprobó la nueva ley de protección de datos personales con amplio apoyo parlamentario.",
        ],
        "legal": [
            "La Constitución establece los derechos fundamentales de los ciudadanos y la organización del Estado.",
            "El artículo primero de la ley establece que todos los ciudadanos son iguales ante la ley sin distinción.",
            "El tribunal determinó que la demanda carecía de fundamento jurídico suficiente para prosperar.",
            "La legislación vigente protege los derechos de los trabajadores en materia de seguridad social.",
            "El contrato de arrendamiento establece las obligaciones del arrendador y del arrendatario.",
            "La jurisprudencia del Tribunal Supremo ha sentado doctrina sobre la interpretación de esta norma.",
            "El derecho internacional regula las relaciones entre los Estados y los organismos internacionales.",
            "La reforma del código penal introduce nuevas figuras delictivas relacionadas con la tecnología.",
            "El procedimiento administrativo garantiza el derecho de los ciudadanos a ser escuchados.",
            "La normativa ambiental establece límites estrictos para las emisiones industriales contaminantes.",
        ],
        "scientific": [
            "Los resultados del experimento confirman la hipótesis planteada sobre la estructura molecular.",
            "La investigación publicada en la revista Nature demuestra una nueva forma de generar energía limpia.",
            "Los datos obtenidos sugieren una correlación significativa entre las variables estudiadas.",
            "El método utilizado combina técnicas de microscopía electrónica y espectroscopía de masas.",
            "La teoría de la relatividad general predice la curvatura del espacio-tiempo alrededor de objetos masivos.",
            "Los genes responsables de la resistencia a antibióticos pueden transferirse entre diferentes especies bacterianas.",
            "El cambio climático está alterando los patrones migratorios de numerosas especies animales.",
            "La mecánica cuántica describe el comportamiento de las partículas subatómicas con gran precisión.",
            "Los ensayos clínicos de fase tres mostraron una eficacia superior al noventa por ciento.",
            "La fotosíntesis convierte la energía luminosa en energía química almacenada en moléculas orgánicas.",
        ]
    }
    
    texts = fallback_texts.get(category, fallback_texts["wikipedia"])
    print(f"   {Colors.GREEN}✓ {len(texts)} oraciones de respaldo para {category}{Colors.END}")
    return texts


def main():
    """Descarga todos los datasets y los guarda."""
    print()
    print("=" * 70)
    print("Descarga de Datasets en Español")
    print("=" * 70)
    print()
    
    # Crear directorio de datos
    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Descargar cada dataset
    all_datasets = {}
    
    all_datasets['wikipedia'] = download_wikipedia()
    all_datasets['news'] = download_news()
    all_datasets['legal'] = download_legal()
    all_datasets['scientific'] = download_scientific()
    
    # Guardar cada dataset
    print()
    print("=" * 70)
    print("Guardando datasets")
    print("=" * 70)
    print()
    
    total_sentences = 0
    successful_datasets = 0
    
    for name, sentences in all_datasets.items():
        if sentences:
            output_file = data_dir / f"{name}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({'sentences': sentences}, f, ensure_ascii=False, indent=2)
            print(f"{Colors.GREEN}✓{Colors.END} {name}: {len(sentences)} oraciones en {output_file}")
            total_sentences += len(sentences)
            successful_datasets += 1
    
    # Crear dataset combinado
    print()
    print("=" * 70)
    print("Creando dataset combinado")
    print("=" * 70)
    print()
    
    all_sentences = []
    for sentences in all_datasets.values():
        all_sentences.extend(sentences)
    
    combined_file = data_dir / "combined.json"
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump({
            'sentences': all_sentences,
            'sources': [k for k, v in all_datasets.items() if v],
            'total': len(all_sentences)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"{Colors.GREEN}✓{Colors.END} Dataset combinado: {len(all_sentences)} oraciones en {combined_file}")
    
    # Resumen
    print()
    print("=" * 70)
    print("Resumen")
    print("=" * 70)
    print()
    print(f"Total de oraciones descargadas: {total_sentences}")
    print(f"Datasets exitosos: {successful_datasets}/4")
    print()
    
    if total_sentences > 0:
        print(f"{Colors.BLUE}Próximos pasos:{Colors.END}")
        print(f"  {Colors.YELLOW}python scripts/train_final.py{Colors.END}              # Entrenar con datasets")
        print(f"  {Colors.YELLOW}python scripts/download_literature.py{Colors.END}     # Descargar literatura (opcional)")
        print(f"  {Colors.YELLOW}python scripts/train_with_literature.py{Colors.END}   # Entrenar con todo (opcional)")
    else:
        print(f"{Colors.YELLOW}⚠ No se pudieron descargar datasets externos{Colors.END}")
        print(f"  El proyecto funciona con el corpus UD AnCora y literatura.")
        print(f"  {Colors.YELLOW}bash scripts/setup_corpus.sh{Colors.END}")
    print()


if __name__ == "__main__":
    main()
