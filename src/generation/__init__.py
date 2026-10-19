"""
Módulo de generación de texto biomimética con HDC + Memoria Asociativa + FEP

Este módulo implementa un sistema de generación de texto inspirado en:
- Memoria asociativa del hipocampo (HDC)
- Selección basada en Free Energy Principle (FEP)
- Mecanismos de fatiga para diversidad
- Identidades configurables (poeta, filósofo, narrador, etc.)

Ejemplo de uso:
    >>> from src.generation import TextGenerator, load_identity
    >>> gen = TextGenerator()
    >>> gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")
    >>> 
    >>> # Configurar identidad de poeta
    >>> poeta = load_identity('poeta')
    >>> gen.set_identity(poeta)
    >>> 
    >>> # Generar texto con estilo poético
    >>> print(gen.generate("El amor", max_words=15))
"""

from .text_generator import TextGenerator, create_generator
from .identity import (
    AIIdentity, 
    load_identity, 
    list_identities, 
    IDENTITIES,
    save_identity,
    load_custom_identity
)

__all__ = [
    'TextGenerator', 
    'create_generator',
    'AIIdentity',
    'load_identity',
    'list_identities',
    'IDENTITIES',
    'save_identity',
    'load_custom_identity'
]
__version__ = '4.0.0'
