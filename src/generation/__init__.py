"""
Módulo de generación de texto biomimética con HDC + Memoria Asociativa + FEP

Este módulo implementa un sistema de generación de texto inspirado en:
- Memoria asociativa del hipocampo (HDC)
- Selección basada en Free Energy Principle (FEP)
- Mecanismos de fatiga para diversidad

Ejemplo de uso:
    >>> from src.generation import TextGenerator
    >>> gen = TextGenerator()
    >>> gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")
    >>> print(gen.generate("El presidente", max_words=15))
"""

from .text_generator import TextGenerator, create_generator

__all__ = ['TextGenerator', 'create_generator']
__version__ = '1.1.0'
