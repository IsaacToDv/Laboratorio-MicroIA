#!/usr/bin/env python3
"""
Utilidades para carga de datos del Laboratorio de Micro-IA

Busca automáticamente todas las fuentes de datos disponibles:
1. UD Spanish-AnCora (corpus principal)
2. Datasets de Hugging Face (data/datasets/)
3. Literatura clásica (data/literature/)

Si no encuentra una fuente, continúa con las demás.
El proyecto funciona con cualquier combinación de fuentes.
"""

import json
from pathlib import Path
from conllu import parse_incr


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def load_ud_ancora(corpus_path=None, max_sentences=3000):
    """Carga oraciones del corpus UD Spanish-AnCora."""
    if corpus_path is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    sentences = []
    
    if not Path(corpus_path).exists():
        return sentences
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, sent in enumerate(parse_incr(f)):
                if i >= max_sentences:
                    break
                words = [token['form'] for token in sent 
                        if not isinstance(token['id'], tuple)]
                if len(words) >= 3:
                    sentences.append(words)
    except Exception:
        pass
    
    return sentences


def load_json_dataset(json_path, max_sentences=2000):
    """Carga oraciones de un dataset JSON."""
    sentences = []
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        raw_sentences = data.get('sentences', [])
        
        for text in raw_sentences[:max_sentences]:
            words = text.strip().split()
            if len(words) >= 3 and len(words) <= 50:
                sentences.append(words)
    except Exception:
        pass
    
    return sentences


def load_all_available_data(max_per_source=3000):
    """
    Busca y carga todos los datos disponibles.
    
    Returns:
        tuple: (all_sentences, sources_info)
            - all_sentences: lista de listas de palabras
            - sources_info: dict con info de cada fuente
    """
    all_sentences = []
    sources = {}
    
    # 1. UD Spanish-AnCora
    ud_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    if Path(ud_path).exists():
        ud_sents = load_ud_ancora(ud_path, max_sentences=max_per_source)
        if ud_sents:
            all_sentences.extend(ud_sents)
            sources['UD AnCora'] = len(ud_sents)
            print(f"   {Colors.GREEN}✓{Colors.END} {len(ud_sents)} oraciones de UD AnCora")
    else:
        print(f"   {Colors.YELLOW}⚠{Colors.END} UD AnCora no encontrado (opcional)")
    
    # 2. Datasets de Hugging Face
    datasets_dir = Path("data/datasets")
    if datasets_dir.exists():
        for json_file in sorted(datasets_dir.glob("*.json")):
            if json_file.name == "combined.json":
                continue
            ds_sents = load_json_dataset(str(json_file), max_sentences=max_per_source)
            if ds_sents:
                all_sentences.extend(ds_sents)
                sources[json_file.stem] = len(ds_sents)
                print(f"   {Colors.GREEN}✓{Colors.END} {len(ds_sents)} oraciones de {json_file.stem}")
    else:
        print(f"   {Colors.YELLOW}⚠{Colors.END} Datasets HF no encontrados (opcional)")
    
    # 3. Literatura clásica
    literature_dir = Path("data/literature")
    if literature_dir.exists():
        for json_file in sorted(literature_dir.glob("*.json")):
            if json_file.name == "literature_combined.json":
                continue
            lit_sents = load_json_dataset(str(json_file), max_sentences=max_per_source)
            if lit_sents:
                all_sentences.extend(lit_sents)
                sources[f"literature/{json_file.stem}"] = len(lit_sents)
                print(f"   {Colors.GREEN}✓{Colors.END} {len(lit_sents)} oraciones de {json_file.stem}")
    else:
        print(f"   {Colors.YELLOW}⚠{Colors.END} Literatura no encontrada (opcional)")
    
    return all_sentences, sources


def ensure_data_available(min_sentences=50):
    """
    Verifica que haya suficientes datos para entrenar.
    Si no hay datos, da instrucciones claras.
    
    Returns:
        tuple: (all_sentences, sources) o (None, None) si no hay datos
    """
    print(f"\n{Colors.BLUE}📖 Buscando datos disponibles...{Colors.END}\n")
    
    all_sentences, sources = load_all_available_data()
    
    print(f"\n   Total: {len(all_sentences)} oraciones de {len(sources)} fuente(s)")
    
    if len(all_sentences) < min_sentences:
        print(f"\n{Colors.RED}❌ No hay suficientes datos para entrenar ({len(all_sentences)}/{min_sentences} mínimo){Colors.END}")
        print(f"\n{Colors.YELLOW}Descarga al menos una fuente de datos:{Colors.END}\n")
        print(f"  {Colors.BLUE}Opción 1 - Corpus UD AnCora (recomendado):{Colors.END}")
        print(f"    {Colors.YELLOW}bash scripts/setup_corpus.sh{Colors.END}")
        print()
        print(f"  {Colors.BLUE}Opción 2 - Literatura clásica:{Colors.END}")
        print(f"    {Colors.YELLOW}python scripts/download_literature.py{Colors.END}")
        print()
        print(f"  {Colors.BLUE}Opción 3 - Datasets multi-disciplina:{Colors.END}")
        print(f"    {Colors.YELLOW}python scripts/download_datasets.py{Colors.END}")
        print()
        return None, None
    
    return all_sentences, sources
