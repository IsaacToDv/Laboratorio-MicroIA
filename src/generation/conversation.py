"""
Memoria conversacional

Mantiene contexto de la conversación:
- Historial de prompts y respuestas
- Extracción de temas recurrentes
- Contexto para mejorar respuestas futuras
"""

from collections import deque
from typing import List, Dict, Optional, Tuple
from .tokenizer import Tokenizer


class ConversationMemory:
    """Memoria de contexto conversacional."""
    
    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns)
        self.topics = []  # Palabras clave recurrentes
    
    def add_turn(self, prompt: str, response: str):
        """Agrega un turno a la memoria."""
        prompt_tokens = Tokenizer.tokenize(prompt)
        response_tokens = Tokenizer.tokenize(response)
        
        self.history.append({
            'prompt': prompt,
            'response': response,
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens
        })
        
        # Actualizar temas recurrentes
        self._update_topics()
    
    def _update_topics(self):
        """Extrae temas recurrentes de la conversación."""
        from collections import Counter
        
        all_content = []
        for turn in self.history:
            content = Tokenizer.extract_content_words(
                turn['prompt_tokens'] + turn['response_tokens'],
                _STOP_WORDS
            )
            all_content.extend(content)
        
        # Temas = palabras que aparecen en múltiples turnos
        word_turns = {}
        for i, turn in enumerate(self.history):
            content = set(Tokenizer.extract_content_words(
                turn['prompt_tokens'], _STOP_WORDS
            ))
            for word in content:
                if word not in word_turns:
                    word_turns[word] = set()
                word_turns[word].add(i)
        
        # Palabras en 2+ turnos son temas
        self.topics = [word for word, turns in word_turns.items() if len(turns) >= 2]
    
    def get_context(self, n_turns: int = 3) -> List[str]:
        """Obtiene palabras de contexto de los últimos N turnos."""
        context = []
        
        recent = list(self.history)[-n_turns:]
        for turn in recent:
            content = Tokenizer.extract_content_words(
                turn['prompt_tokens'] + turn['response_tokens'],
                _STOP_WORDS
            )
            context.extend(content)
        
        # Eliminar duplicados preservando orden
        seen = set()
        unique = []
        for w in context:
            if w not in seen:
                seen.add(w)
                unique.append(w)
        
        return unique
    
    def get_active_topics(self) -> List[str]:
        """Retorna los temas activos de la conversación."""
        return list(self.topics)
    
    def get_last_prompt_keywords(self) -> List[str]:
        """Retorna keywords del último prompt."""
        if not self.history:
            return []
        
        last = self.history[-1]
        return Tokenizer.extract_content_words(last['prompt_tokens'], _STOP_WORDS)
    
    def enrich_prompt(self, prompt_words: List[str]) -> List[str]:
        """
        Enriquece las palabras del prompt con contexto conversacional.
        
        Si el prompt es corto o ambiguo, agrega temas de la conversación.
        """
        content_words = [w.lower() for w in prompt_words 
                        if Tokenizer.normalize(w) not in _STOP_WORDS 
                        and len(Tokenizer.normalize(w)) > 2]
        
        # Si el prompt tiene pocas keywords, agregar contexto
        if len(content_words) <= 2 and self.topics:
            # Agregar temas relevantes que no estén ya en el prompt
            enriched = list(content_words)
            for topic in self.topics[:3]:
                if topic not in enriched:
                    enriched.append(topic)
            return enriched
        
        return content_words
    
    def clear(self):
        """Limpia la memoria conversacional."""
        self.history.clear()
        self.topics = []
    
    @property
    def turn_count(self) -> int:
        return len(self.history)


# Stop words (compartidas)
_STOP_WORDS = {
    'de', 'la', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por',
    'un', 'para', 'con', 'no', 'una', 'su', 'al', 'es', 'lo', 'como',
    'más', 'pero', 'sus', 'le', 'ya', 'o', 'este', 'sí', 'porque',
    'esta', 'son', 'entre', 'cuando', 'muy', 'sin', 'sobre', 'ser',
    'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo',
    'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros',
    'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos',
    'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa',
    'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella',
    'estar', 'estas', 'algunas', 'algo', 'nosotros', 'que', 'mi', 'tu',
    'ha', 'he', 'te', 'si', 'han', 'fue', 'era', 'tan', 'les', 'ven',
    'the', 'of', 'and', 'to', 'in', 'is', 'it', 'that', 'was', 'for',
}
