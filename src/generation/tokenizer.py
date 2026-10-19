"""
Tokenizador robusto para español

Maneja:
- Separación de puntuación
- Normalización de acentos (búsqueda)
- Contracciones (del, al)
- Preservación de mayúsculas en output
- Reconstrucción de texto original
"""

import re
from typing import List, Tuple


# Puntuación que debe separarse como tokens independientes
PUNCTUATION = set('.,;:!?¡¿()[]{}""\'—–-…/')

# Contracciones del español
CONTRACTIONS = {
    'del': ('de', 'el'),
    'al': ('a', 'el'),
}

EXPANSIONS = {
    ('de', 'el'): 'del',
    ('a', 'el'): 'al',
}


class Tokenizer:
    """Tokenizador robusto para español."""
    
    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokeniza texto separando puntuación y preservando información.
        
        "Hola, ¿cómo estás?" → ["Hola", ",", "¿", "cómo", "estás", "?"]
        """
        # Insertar espacios alrededor de puntuación
        result = []
        i = 0
        while i < len(text):
            ch = text[i]
            
            if ch in PUNCTUATION:
                # Signos de apertura (¿¡) van solos
                if ch in '¿¡':
                    result.append(ch)
                # Guión entre palabras no se separa
                elif ch == '-' and i > 0 and i < len(text) - 1:
                    if text[i-1].isalpha() and text[i+1].isalpha():
                        result[-1] = result[-1] + ch
                    else:
                        result.append(ch)
                else:
                    result.append(ch)
                i += 1
            elif ch.isspace():
                i += 1
            else:
                # Acumular caracteres hasta espacio o puntuación
                word = []
                while i < len(text) and not text[i].isspace() and text[i] not in PUNCTUATION:
                    # Guión interno: incluirlo en la palabra
                    if text[i] == '-' and i + 1 < len(text) and text[i+1].isalpha() and word:
                        word.append(text[i])
                        i += 1
                        continue
                    word.append(text[i])
                    i += 1
                if word:
                    result.append(''.join(word))
        
        return result
    
    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        """
        Reconstruye texto a partir de tokens.
        
        Maneja puntuación correctamente:
        - "Hola", "," , "mundo" → "Hola, mundo"
        - "¿", "Cómo", "estás", "?" → "¿Cómo estás?"
        """
        if not tokens:
            return ""
        
        result = []
        
        for i, token in enumerate(tokens):
            if not token:
                continue
            
            if token in '.,;:!?)]}—–…/':
                # Puntuación de cierre: pegar al token anterior
                if result:
                    result[-1] = result[-1] + token
                else:
                    result.append(token)
            elif token in '¿¡([{':
                # Puntuación de apertura: pegar al siguiente
                result.append(token)
            elif result and result[-1] in '¿¡([{':
                # Después de apertura: pegar
                result[-1] = result[-1] + token
            else:
                result.append(token)
        
        return ' '.join(result)
    
    @staticmethod
    def normalize(word: str) -> str:
        """
        Normaliza una palabra para búsqueda (lowercase, sin puntuación).
        Preserva acentos para matching en español.
        """
        return re.sub(r'[^\wáéíóúñüÁÉÍÓÚÑÜ]', '', word.lower())
    
    @staticmethod
    def extract_content_words(tokens: List[str], stop_words: set) -> List[str]:
        """Extrae palabras de contenido (no stop words, no puntuación)."""
        result = []
        for t in tokens:
            norm = Tokenizer.normalize(t)
            if len(norm) > 2 and norm not in stop_words and not all(c in PUNCTUATION for c in t):
                result.append(norm)
        return result
