#!/usr/bin/env python3
"""
Demo interactivo del generador de texto biomimético

Características:
- Interfaz colorida y amigable
- Selección de identidades de IA
- Control de parámetros de generación
- Ejemplos predefinidos
- Modo interactivo

Uso:
    python demo.py
"""

import sys
import os
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from generation import TextGenerator, load_identity, list_identities


class Colors:
    """Colores para terminal."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


def print_banner():
    """Imprime el banner del proyecto."""
    banner = f"""
{Colors.CYAN}╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧬 Laboratorio de Micro-IA Biomimética                      ║
║                                                               ║
║   Generador de Texto con Identidades de IA                    ║
║   Inspirado en Neurociencia y Literatura Clásica              ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝{Colors.END}
"""
    print(banner)


def print_menu():
    """Imprime el menú principal."""
    print(f"\n{Colors.BOLD}📋 Menú Principal{Colors.END}\n")
    print(f"  {Colors.BLUE}1.{Colors.END} Generar texto con identidad predefinida")
    print(f"  {Colors.BLUE}2.{Colors.END} Ver ejemplos por disciplina")
    print(f"  {Colors.BLUE}3.{Colors.END} Modo interactivo (escribe tus propios prompts)")
    print(f"  {Colors.BLUE}4.{Colors.END} Información del proyecto")
    print(f"  {Colors.BLUE}5.{Colors.END} Salir")
    print()


def load_generator():
    """Carga el generador de texto."""
    print(f"\n{Colors.CYAN}⏳ Cargando generador...{Colors.END}")
    
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    if not Path(corpus_path).exists():
        print(f"{Colors.RED}❌ Error: Corpus no encontrado{Colors.END}")
        print(f"\n{Colors.YELLOW}Solución:{Colors.END}")
        print(f"  1. Ejecuta: {Colors.BLUE}bash scripts/setup_corpus.sh{Colors.END}")
        print(f"  2. Luego: {Colors.BLUE}python scripts/train_with_literature.py{Colors.END}")
        print(f"  3. Vuelve a ejecutar este demo")
        return None
    
    try:
        gen = TextGenerator(dim=10000)
        gen.train_from_corpus(corpus_path, max_sentences=3000)
        print(f"{Colors.GREEN}✓ Generador cargado exitosamente{Colors.END}")
        return gen
    except Exception as e:
        print(f"{Colors.RED}❌ Error al cargar generador: {e}{Colors.END}")
        return None


def select_identity():
    """Permite seleccionar una identidad."""
    identities = list_identities()
    
    print(f"\n{Colors.BOLD}🎭 Identidades disponibles:{Colors.END}\n")
    
    for i, identity_name in enumerate(identities, 1):
        identity = load_identity(identity_name)
        print(f"  {Colors.BLUE}{i}.{Colors.END} {Colors.BOLD}{identity.name}{Colors.END}")
        print(f"     {Colors.DIM}{identity.description}{Colors.END}")
        print(f"     {Colors.DIM}Estilo: {identity.style} | Tono: {identity.emotional_tone}{Colors.END}")
        print()
    
    while True:
        try:
            choice = input(f"{Colors.YELLOW}Selecciona una identidad (1-{len(identities)}): {Colors.END}").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(identities):
                return identities[idx]
            else:
                print(f"{Colors.RED}Opción no válida{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}Ingresa un número{Colors.END}")
        except KeyboardInterrupt:
            return None


def generate_with_identity(gen):
    """Genera texto con una identidad seleccionada."""
    identity_name = select_identity()
    if not identity_name:
        return
    
    identity = load_identity(identity_name)
    gen.set_identity(identity)
    
    print(f"\n{Colors.GREEN}✓ Identidad configurada: {identity.name}{Colors.END}")
    print(f"{Colors.DIM}  Estilo: {identity.style}{Colors.END}")
    print(f"{Colors.DIM}  Tono: {identity.emotional_tone}{Colors.END}")
    print(f"{Colors.DIM}  Temperatura: {identity.temperature}{Colors.END}")
    
    # Prompts sugeridos
    prompts = [
        "El amor",
        "El tiempo",
        "La vida",
        "La muerte",
        "La memoria"
    ]
    
    print(f"\n{Colors.BOLD}💡 Prompts sugeridos:{Colors.END}")
    for i, prompt in enumerate(prompts, 1):
        print(f"  {Colors.BLUE}{i}.{Colors.END} {prompt}")
    print(f"  {Colors.BLUE}0.{Colors.END} Escribir mi propio prompt")
    
    while True:
        try:
            choice = input(f"\n{Colors.YELLOW}Selecciona un prompt (0-{len(prompts)}): {Colors.END}").strip()
            idx = int(choice)
            
            if idx == 0:
                prompt = input(f"{Colors.YELLOW}Escribe tu prompt: {Colors.END}").strip()
            elif 1 <= idx <= len(prompts):
                prompt = prompts[idx - 1]
            else:
                print(f"{Colors.RED}Opción no válida{Colors.END}")
                continue
            
            if not prompt:
                continue
            
            # Generar texto
            print(f"\n{Colors.CYAN}⏳ Generando...{Colors.END}\n")
            
            for i in range(3):
                text = gen.generate(prompt, max_words=20)
                print(f"{Colors.MAGENTA}{i+1}.{Colors.END} {text}")
                print()
            
            again = input(f"{Colors.YELLOW}¿Generar más? (s/n): {Colors.END}").strip().lower()
            if again != 's':
                break
                
        except ValueError:
            print(f"{Colors.RED}Ingresa un número{Colors.END}")
        except KeyboardInterrupt:
            break


def show_examples_by_discipline(gen):
    """Muestra ejemplos por disciplina."""
    disciplinas = {
        "Política": ["El presidente", "El gobierno", "Las elecciones"],
        "Economía": ["La economía", "El mercado", "La inflación"],
        "Ciencia": ["Los científicos", "La investigación", "El estudio"],
        "Tecnología": ["La tecnología", "La inteligencia artificial", "Los datos"],
        "Cultura": ["La película", "El libro", "La música"]
    }
    
    # Usar identidad por defecto
    identity = load_identity('contemporaneo')
    gen.set_identity(identity)
    
    print(f"\n{Colors.BOLD}📚 Ejemplos por Disciplina{Colors.END}\n")
    
    for disciplina, prompts in disciplinas.items():
        print(f"{Colors.CYAN}{'=' * 60}{Colors.END}")
        print(f"{Colors.BOLD}{disciplina}{Colors.END}")
        print(f"{Colors.CYAN}{'=' * 60}{Colors.END}\n")
        
        for prompt in prompts:
            print(f"{Colors.YELLOW}Prompt:{Colors.END} {prompt}")
            text = gen.generate(prompt, max_words=15)
            print(f"{Colors.GREEN}→{Colors.END} {text}\n")
    
    input(f"\n{Colors.DIM}Presiona Enter para continuar...{Colors.END}")


def interactive_mode(gen):
    """Modo interactivo."""
    print(f"\n{Colors.BOLD}🎮 Modo Interactivo{Colors.END}\n")
    print(f"{Colors.DIM}Escribe tus propios prompts y genera texto.{Colors.END}")
    print(f"{Colors.DIM}Escribe 'salir' para terminar.{Colors.END}\n")
    
    # Seleccionar identidad
    identity_name = select_identity()
    if not identity_name:
        return
    
    identity = load_identity(identity_name)
    gen.set_identity(identity)
    
    print(f"\n{Colors.GREEN}✓ Modo interactivo activado con identidad: {identity.name}{Colors.END}\n")
    
    while True:
        try:
            prompt = input(f"{Colors.CYAN}Tu prompt:{Colors.END} ").strip()
            
            if prompt.lower() in ['salir', 'exit', 'quit']:
                break
            
            if not prompt:
                continue
            
            print(f"\n{Colors.CYAN}⏳ Generando...{Colors.END}\n")
            text = gen.generate(prompt, max_words=20)
            print(f"{Colors.MAGENTA}→{Colors.END} {text}\n")
            
        except KeyboardInterrupt:
            break
    
    print(f"\n{Colors.GREEN}✓ Modo interactivo terminado{Colors.END}")


def show_info():
    """Muestra información del proyecto."""
    print(f"\n{Colors.BOLD}📖 Información del Proyecto{Colors.END}\n")
    
    print(f"{Colors.CYAN}Nombre:{Colors.END} Laboratorio de Micro-IA Biomimética")
    print(f"{Colors.CYAN}Versión:{Colors.END} 4.0.0")
    print(f"{Colors.CYAN}Descripción:{Colors.END} Generador de texto en español con identidades de IA\n")
    
    print(f"{Colors.BOLD}Características:{Colors.END}")
    print(f"  • Inspirado en neurociencia (HDC, FEP, mecanismos de fatiga)")
    print(f"  • 5 identidades de IA (Poeta, Filósofo, Narrador, Ensayista, Contemporáneo)")
    print(f"  • Literatura clásica de 15+ autores hispanos")
    print(f"  • Datasets multi-disciplina (115,000+ oraciones)")
    print(f"  • Sin necesidad de GPU\n")
    
    print(f"{Colors.BOLD}Autores inspiradores:{Colors.END}")
    print(f"  • Poesía: Neruda, Lorca, Paz, Vallejo")
    print(f"  • Filosofía: Unamuno, Ortega y Gasset, Zambrano")
    print(f"  • Narrativa: García Márquez, Borges, Cortázar")
    print(f"  • Ensayo: Borges, Paz, Vargas Llosa\n")
    
    print(f"{Colors.BOLD}Enlaces:{Colors.END}")
    print(f"  • Repositorio: https://github.com/IsaacToDv/Laboratorio-MicroIA")
    print(f"  • Documentación: README.md")
    print(f"  • Guía de contribución: CONTRIBUTING.md\n")
    
    input(f"{Colors.DIM}Presiona Enter para continuar...{Colors.END}")


def main():
    """Función principal."""
    print_banner()
    
    # Verificar instalación
    print(f"{Colors.CYAN}🔍 Verificando instalación...{Colors.END}")
    
    gen = load_generator()
    if not gen:
        return 1
    
    # Menú principal
    while True:
        print_menu()
        
        try:
            choice = input(f"{Colors.YELLOW}Selecciona una opción (1-5): {Colors.END}").strip()
            
            if choice == '1':
                generate_with_identity(gen)
            elif choice == '2':
                show_examples_by_discipline(gen)
            elif choice == '3':
                interactive_mode(gen)
            elif choice == '4':
                show_info()
            elif choice == '5':
                print(f"\n{Colors.GREEN}¡Hasta luego! 👋{Colors.END}\n")
                break
            else:
                print(f"{Colors.RED}Opción no válida{Colors.END}")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.GREEN}¡Hasta luego! 👋{Colors.END}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
