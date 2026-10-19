#!/usr/bin/env python3
"""
Script de verificación de instalación

Verifica que todas las dependencias y archivos necesarios estén presentes
antes de ejecutar el proyecto.

Uso:
    python scripts/check_installation.py
"""

import sys
import os
from pathlib import Path


class Colors:
    """Colores para terminal."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_success(msg):
    """Imprime mensaje de éxito."""
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")


def print_error(msg):
    """Imprime mensaje de error."""
    print(f"{Colors.RED}✗{Colors.END} {msg}")


def print_warning(msg):
    """Imprime mensaje de advertencia."""
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")


def print_info(msg):
    """Imprime mensaje informativo."""
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")


def check_python_version():
    """Verifica la versión de Python."""
    print_info("Verificando versión de Python...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} (se requiere 3.7+)")
        return False


def check_dependencies():
    """Verifica que las dependencias estén instaladas."""
    print_info("Verificando dependencias...")
    
    required = ['numpy', 'conllu']
    optional = ['datasets', 'requests']
    
    all_ok = True
    
    for package in required:
        try:
            __import__(package)
            print_success(f"{package} instalado")
        except ImportError:
            print_error(f"{package} NO instalado (requerido)")
            print_info(f"  Instala con: pip install {package}")
            all_ok = False
    
    for package in optional:
        try:
            __import__(package)
            print_success(f"{package} instalado")
        except ImportError:
            print_warning(f"{package} NO instalado (opcional)")
            print_info(f"  Instala con: pip install {package} (para datasets de Hugging Face)")
    
    return all_ok


def check_corpus():
    """Verifica que el corpus UD AnCora esté descargado."""
    print_info("Verificando corpus UD AnCora...")
    
    corpus_path = Path("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")
    
    if corpus_path.exists():
        size_mb = corpus_path.stat().st_size / (1024 * 1024)
        print_success(f"Corpus encontrado ({size_mb:.1f} MB)")
        return True
    else:
        print_warning("Corpus NO descargado")
        print_info("  Descarga con: bash scripts/setup_corpus.sh")
        return False


def check_datasets():
    """Verifica que los datasets de Hugging Face estén descargados."""
    print_info("Verificando datasets de Hugging Face...")
    
    datasets_dir = Path("data/datasets")
    
    if not datasets_dir.exists():
        print_warning("Directorio de datasets NO existe")
        print_info("  Descarga con: python scripts/download_datasets.py")
        return False
    
    json_files = list(datasets_dir.glob("*.json"))
    
    if len(json_files) > 0:
        print_success(f"{len(json_files)} datasets encontrados")
        for json_file in json_files:
            size_mb = json_file.stat().st_size / (1024 * 1024)
            print_info(f"  - {json_file.name} ({size_mb:.1f} MB)")
        return True
    else:
        print_warning("NO hay datasets descargados")
        print_info("  Descarga con: python scripts/download_datasets.py")
        return False


def check_literature():
    """Verifica que la literatura clásica esté descargada."""
    print_info("Verificando literatura clásica...")
    
    literature_dir = Path("data/literature")
    
    if not literature_dir.exists():
        print_warning("Directorio de literatura NO existe")
        print_info("  Descarga con: python scripts/download_literature.py")
        return False
    
    json_files = list(literature_dir.glob("*.json"))
    
    if len(json_files) > 0:
        print_success(f"{len(json_files)} archivos de literatura encontrados")
        return True
    else:
        print_warning("NO hay literatura descargada")
        print_info("  Descarga con: python scripts/download_literature.py")
        return False


def check_models():
    """Verifica que haya modelos entrenados."""
    print_info("Verificando modelos entrenados...")
    
    models_dir = Path("models")
    
    if not models_dir.exists():
        print_warning("Directorio de modelos NO existe")
        print_info("  Entrena con: python scripts/train_with_literature.py")
        return False
    
    config_files = list(models_dir.glob("*.json"))
    
    if len(config_files) > 0:
        print_success(f"{len(config_files)} configuración(es) de modelo encontrada(s)")
        return True
    else:
        print_warning("NO hay modelos entrenados")
        print_info("  Entrena con: python scripts/train_with_literature.py")
        return False


def print_next_steps(corpus_ok, datasets_ok, literature_ok, models_ok):
    """Imprime los siguientes pasos recomendados."""
    print(f"\n{Colors.BOLD}📋 Próximos pasos recomendados:{Colors.END}\n")
    
    step = 1
    
    if not corpus_ok:
        print(f"{step}. Descargar corpus UD AnCora:")
        print(f"   {Colors.BLUE}bash scripts/setup_corpus.sh{Colors.END}\n")
        step += 1
    
    if not datasets_ok:
        print(f"{step}. Descargar datasets de Hugging Face:")
        print(f"   {Colors.BLUE}python scripts/download_datasets.py{Colors.END}\n")
        step += 1
    
    if not literature_ok:
        print(f"{step}. Descargar literatura clásica:")
        print(f"   {Colors.BLUE}python scripts/download_literature.py{Colors.END}\n")
        step += 1
    
    if not models_ok:
        print(f"{step}. Entrenar modelo:")
        print(f"   {Colors.BLUE}python scripts/train_with_literature.py{Colors.END}\n")
        step += 1
    
    print(f"{step}. Probar el generador:")
    print(f"   {Colors.BLUE}python demo.py{Colors.END}\n")
    
    print(f"{step + 1}. Explorar ejemplos:")
    print(f"   {Colors.BLUE}python examples/identidades.py{Colors.END}\n")


def main():
    """Función principal."""
    print(f"\n{Colors.BOLD}🔍 Verificación de Instalación{Colors.END}\n")
    print("=" * 60)
    
    python_ok = check_python_version()
    print()
    
    deps_ok = check_dependencies()
    print()
    
    corpus_ok = check_corpus()
    print()
    
    datasets_ok = check_datasets()
    print()
    
    literature_ok = check_literature()
    print()
    
    models_ok = check_models()
    print()
    
    print("=" * 60)
    
    if python_ok and deps_ok:
        print_success("Instalación básica correcta")
    else:
        print_error("Problemas con la instalación básica")
        print_info("Revisa los errores arriba y corrígelos")
        return 1
    
    if corpus_ok and datasets_ok and literature_ok and models_ok:
        print_success("Todo listo para usar el proyecto")
        print_info("Ejecuta: python demo.py")
    else:
        print_warning("Faltan algunos componentes")
        print_next_steps(corpus_ok, datasets_ok, literature_ok, models_ok)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
