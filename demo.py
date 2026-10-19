#!/usr/bin/env python3
"""
Demo simplificado - Prompt Libre

Escribe lo que quieras. La IA identifica automáticamente el tipo de
consulta y se adapta. Sin menús, sin opciones complicadas.

Comandos opcionales:
    /poeta       - Cambiar a voz poética
    /filosofo    - Cambiar a voz filosófica
    /narrador    - Cambiar a voz narrativa
    /ensayista   - Cambiar a voz ensayística
    /contemporaneo - Voz moderna
    /auto        - Auto-detectar identidad (default)
    /help        - Ver comandos
    /salir       - Terminar

Uso:
    python demo.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent / 'scripts'))

from generation import TextGenerator, load_identity, list_identities
from generation.text_generator import classify_intent


class C:
    """Colores para terminal."""
    G = '\033[92m'    # Verde
    R = '\033[91m'    # Rojo
    Y = '\033[93m'    # Amarillo
    B = '\033[94m'    # Azul
    M = '\033[95m'    # Magenta
    CY = '\033[96m'   # Cyan
    D = '\033[2m'     # Dim
    BOLD = '\033[1m'  # Bold
    E = '\033[0m'     # End


# Comandos para cambiar identidad
IDENTITY_COMMANDS = {
    '/poeta': 'poeta',
    '/filosofo': 'filosofo',
    '/narrador': 'narrador',
    '/ensayista': 'ensayista',
    '/contemporaneo': 'contemporaneo',
    '/contemporáneo': 'contemporaneo',
}


def load_gen():
    """Carga el generador desde cualquier fuente disponible."""
    print(f"\n{C.CY}⏳ Cargando...{C.E}", end="", flush=True)
    
    # Intentar 1: Modelo pre-entrenado
    trained = Path("models/trained_corpus.json")
    if trained.exists():
        try:
            import json
            with open(trained, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            raw = data.get('sentences', [])
            sentences = []
            for text in raw:
                words = text.strip().split()
                if len(words) >= 3:
                    sentences.append(words)
            
            if sentences:
                gen = TextGenerator(dim=10000)
                gen.train_from_sentences(sentences)
                print(f"\r{C.G}✓ Modelo cargado ({len(sentences):,} oraciones, Deep Cognition){C.E}            ")
                return gen
        except Exception:
            pass
    
    # Intentar 2: Corpus UD AnCora
    corpus = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    if Path(corpus).exists():
        gen = TextGenerator(dim=10000)
        gen.train_from_corpus(corpus, max_sentences=5000)
        print(f"\r{C.G}✓ Generador cargado (UD AnCora, dim=20,000){C.E}            ")
        return gen
    
    # Intentar 3: data_loader
    try:
        from data_loader import ensure_data_available
        all_sents, sources = ensure_data_available(min_sentences=50)
        if all_sents:
            gen = TextGenerator(dim=10000)
            gen.train_from_sentences(all_sents)
            print(f"\r{C.G}✓ Generador cargado ({len(all_sents):,} oraciones, Deep Cognition){C.E}            ")
            return gen
    except Exception:
        pass
    
    print(f"\r{C.R}✗ No hay datos. Ejecuta: bash scripts/setup_corpus.sh{C.E}            ")
    return None


def show_help():
    """Muestra la ayuda."""
    print(f"""
{C.BOLD}🧬 Laboratorio de Micro-IA Biomimética{C.E}
{C.D}Escribe lo que quieras. La IA se adapta automáticamente.{C.E}

{C.BOLD}Comandos:{C.E}
  {C.B}/poeta{C.E}        Voz poética (Neruda, Lorca, Paz)
  {C.B}/filosofo{C.E}     Voz filosófica (Unamuno, Ortega, Zambrano)
  {C.B}/narrador{C.E}     Voz narrativa (García Márquez, Borges)
  {C.B}/ensayista{C.E}    Voz ensayística (Paz, Vargas Llosa)
  {C.B}/contemporaneo{C.E} Voz moderna (Bolaño, Marías)
  {C.B}/auto{C.E}         Auto-detectar identidad {C.D}(default){C.E}
  {C.B}/help{C.E}         Esta ayuda
  {C.B}/salir{C.E}        Terminar

{C.BOLD}Ejemplos:{C.E}
  {C.D}El amor es...{C.E}              → {C.M}detecta: poeta{C.E}
  {C.D}¿Quién fue Platón?{C.E}         → {C.M}detecta: filósofo{C.E}
  {C.D}Había una vez...{C.E}           → {C.M}detecta: narrador{C.E}
  {C.D}La sociedad moderna...{C.E}      → {C.M}detecta: ensayista{C.E}
  {C.D}¿Qué es Google?{C.E}            → {C.M}detecta: contemporáneo{C.E}
""")


def main():
    """Demo con prompt libre simplificado."""
    # Banner minimalista
    print(f"\n{C.BOLD}🧬 Micro-IA Biomimética v6 — Deep Cognition{C.E}")
    print(f"{C.D}Escribe libremente. Usa /help para comandos.{C.E}")
    
    gen = load_gen()
    if not gen:
        return 1
    
    auto_mode = True
    current_identity = None
    turn_count = 0
    
    print(f"{C.D}{'─' * 60}{C.E}")
    print()
    
    while True:
        try:
            # Prompt simple
            prompt = input(f"{C.CY}❯{C.E} ").strip()
            
            if not prompt:
                continue
            
            # ═══ COMANDOS ═══
            if prompt.startswith('/'):
                cmd = prompt.lower().strip()
                
                if cmd in ('/salir', '/exit', '/quit', '/q'):
                    print(f"\n{C.G}¡Hasta luego! 👋{C.E}\n")
                    break
                
                if cmd == '/help':
                    show_help()
                    continue
                
                if cmd == '/auto':
                    auto_mode = True
                    gen.set_auto_identity(True)
                    gen.identity = None
                    current_identity = None
                    print(f"{C.G}✓{C.E} Modo automático activado {C.D}(detecta identidad por tema){C.E}")
                    continue
                
                if cmd in IDENTITY_COMMANDS:
                    identity_name = IDENTITY_COMMANDS[cmd]
                    identity = load_identity(identity_name)
                    gen.set_identity(identity)
                    gen._auto_identity = False
                    auto_mode = False
                    current_identity = identity
                    print(f"{C.G}✓{C.E} {identity.name} {C.D}({identity.emotional_tone}){C.E}")
                    continue
                
                # Comando no reconocido
                print(f"{C.Y}?{C.E} Comando no reconocido. Usa {C.B}/help{C.E}")
                continue
            
            # ═══ GENERACIÓN ═══
            turn_count += 1
            
            # Auto-detectar identidad si está en modo auto
            if auto_mode:
                intent = classify_intent(prompt)
                try:
                    new_identity = load_identity(intent['identity'])
                    if current_identity is None or current_identity.name != new_identity.name:
                        gen.identity = new_identity
                        current_identity = new_identity
                        # Mostrar identidad detectada
                        if intent['confidence'] > 0:
                            print(f"  {C.D}🎭 {new_identity.name} ({new_identity.emotional_tone}){C.E}")
                        else:
                            gen.identity = new_identity
                except Exception:
                    pass
            
            # Generar respuesta
            text = gen.generate(prompt, max_words=30)
            
            # Mostrar respuesta
            print(f"  {C.G}→{C.E} {text}")
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{C.G}¡Hasta luego! 👋{C.E}\n")
            break
        except Exception as e:
            print(f"  {C.R}✗ Error: {e}{C.E}")
            print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
