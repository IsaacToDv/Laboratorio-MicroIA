"""
Scoring de coherencia y relevancia de texto generado

Evalúa una respuesta completa (no palabra por palabra) usando:
- Similitud semántica con el prompt original
- Diversidad léxica (evitar repeticiones)
- Coherencia local (transiciones suaves entre palabras)
- Completitud estructural (empieza y termina bien)
"""

import numpy as np
from typing import List, Optional, Dict
from collections import Counter


class CoherenceScorer:
    """Evalúa la calidad de texto generado."""
    
    def __init__(self, semantic_memory=None):
        self.semantic = semantic_memory
    
    def score(self, generated_words: List[str], prompt_words: List[str],
              identity_bias: Optional[Dict] = None) -> float:
        """
        Score de calidad global (0-1, mayor es mejor).
        
        Combina:
        - Relevancia semántica con el prompt (40%)
        - Diversidad léxica (25%)
        - Coherencia local (20%)
        - Estructura (15%)
        """
        scores = []
        weights = []
        
        # 1. Relevancia semántica con el prompt
        if self.semantic and self.semantic.is_trained:
            relevance = self._semantic_relevance(generated_words, prompt_words)
            scores.append(relevance)
            weights.append(0.40)
        
        # 2. Diversidad léxica
        diversity = self._lexical_diversity(generated_words)
        scores.append(diversity)
        weights.append(0.25)
        
        # 3. Coherencia local
        local_coherence = self._local_coherence(generated_words)
        scores.append(local_coherence)
        weights.append(0.20)
        
        # 4. Estructura
        structure = self._structural_score(generated_words)
        scores.append(structure)
        weights.append(0.15)
        
        # Bonus por identidad
        if identity_bias:
            identity_bonus = self._identity_match(generated_words, identity_bias)
            scores.append(identity_bonus)
            weights.append(0.10)
        
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    def _semantic_relevance(self, generated: List[str], prompt: List[str]) -> float:
        """Similitud semántica entre el texto generado y el prompt."""
        if not self.semantic or not self.semantic.is_trained:
            return 0.5
        
        gen_emb = self.semantic.sentence_embedding(generated)
        prompt_emb = self.semantic.sentence_embedding(prompt)
        
        if gen_emb is None or prompt_emb is None:
            return 0.5
        
        sim = np.dot(gen_emb, prompt_emb) / (
            np.linalg.norm(gen_emb) * np.linalg.norm(prompt_emb) + 1e-8
        )
        return float(max(0, sim))
    
    def _lexical_diversity(self, words: List[str]) -> float:
        """Type-Token Ratio normalizado. Mayor = más diverso."""
        if len(words) <= 1:
            return 0.5
        
        # Excluir stop words y puntuación para diversidad real
        content = [w.lower() for w in words if w.isalpha() and len(w) > 2]
        if not content:
            return 0.5
        
        unique = len(set(content))
        total = len(content)
        
        # TTR normalizado (MATTR simplificado)
        ttr = unique / max(total, 1)
        
        # Penalizar palabras muy repetidas
        counts = Counter(content)
        max_repeat = max(counts.values()) if counts else 0
        repetition_penalty = min(1.0, max_repeat / max(total, 1))
        
        return ttr * (1 - 0.5 * repetition_penalty)
    
    def _local_coherence(self, words: List[str]) -> float:
        """
        Evalúa si las transiciones entre palabras consecutivas son suaves.
        Usa bigramas comunes vs bigramas raros.
        """
        if len(words) < 2:
            return 0.5
        
        # Heurística: puntuación bien colocada = buena coherencia
        score = 0.5
        
        # Puntuación no debe estar al inicio (excepto ¿¡)
        if words[0] in '.,;:!?)]':
            score -= 0.2
        
        # No debe tener 2+ signos seguidos
        for i in range(len(words) - 1):
            if words[i] in '.,;:!?¿¡' and words[i+1] in '.,;:!?¿¡':
                score -= 0.1
        
        # Conectores entre oraciones
        connectors = {'y', 'pero', 'sin', 'además', 'también', 'porque', 'aunque', 'o', 'ni'}
        lower_words = [w.lower() for w in words]
        connector_count = sum(1 for w in lower_words if w in connectors)
        
        # Tener conectores es bueno para coherencia
        if len(words) > 10:
            connector_ratio = connector_count / len(words)
            if 0.02 < connector_ratio < 0.15:
                score += 0.2
        
        return max(0, min(1, score))
    
    def _structural_score(self, words: List[str]) -> float:
        """Evalúa estructura: empieza bien, termina bien, longitud apropiada."""
        if len(words) < 3:
            return 0.3
        
        score = 0.5
        
        # Empieza con mayúscula o signo de apertura
        if words[0] and (words[0][0].isupper() or words[0] in '¿¡'):
            score += 0.1
        
        # Termina con puntuación
        if words[-1] in '.!?':
            score += 0.1
        
        # Longitud apropiada (no muy corta, no muy larga para una oración)
        if 5 < len(words) < 40:
            score += 0.1
        
        # No termina con preposición o artículo (oración incompleta)
        bad_endings = {'de', 'la', 'el', 'en', 'a', 'por', 'con', 'un', 'una', 'los', 'las'}
        if words[-1].lower() in bad_endings:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _identity_match(self, words: List[str], bias: Dict) -> float:
        """Qué tan bien el texto generado usa palabras de la identidad."""
        if not bias:
            return 0.5
        
        lower_words = set(w.lower() for w in words)
        bias_words = set(k.lower() for k in bias.keys())
        
        matches = len(lower_words & bias_words)
        ratio = matches / max(len(bias_words), 1)
        
        return min(1.0, ratio * 3)  # Escalar para que pocos matches ya den buen score
    
    def rerank(self, candidates: List[str], prompt_words: List[str],
               generated_so_far: List[str], top_k: int = 5) -> List[str]:
        """
        Re-rankeo de candidatos según relevancia con prompt y contexto.
        
        Args:
            candidates: Palabras candidatas
            prompt_words: Palabras del prompt original
            generated_so_far: Palabras ya generadas
            top_k: Cuántas retornar
        
        Returns:
            Top-k candidatas re-rankeadas
        """
        if not candidates or not self.semantic or not self.semantic.is_trained:
            return candidates[:top_k]
        
        context = generated_so_far[-6:] if len(generated_so_far) >= 6 else generated_so_far
        
        scored = []
        for word in candidates:
            score = 0.0
            
            # Similitud con el prompt
            prompt_sim = self.semantic.similarity(word, ' '.join(prompt_words[:3]))
            score += prompt_sim * 0.3
            
            # Similitud con contexto reciente
            if context:
                for ctx_word in context[-3:]:
                    ctx_sim = self.semantic.similarity(word, ctx_word)
                    score += ctx_sim * 0.1
            
            # Diversidad: bonus si no está en las últimas palabras
            if word.lower() not in [w.lower() for w in generated_so_far[-4:]]:
                score += 0.2
            
            scored.append((word, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in scored[:top_k]]
