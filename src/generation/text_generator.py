"""
Generador de Texto Biomimético con HDC + Memoria Asociativa + FEP

Este módulo implementa un sistema de generación de texto que:
1. Almacena secuencias del corpus como vectores hiperdimensionales
2. Recupera continuaciones por similitud contextual
3. Selecciona la siguiente palabra minimizando energía libre (FEP)

Autor: Laboratorio Micro-IA Biomimética
Fase: 7 - Generación de Texto
"""

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from conllu import parse_incr
from typing import List, Tuple, Optional, Dict
import json


class HDCSequenceMemory:
    """
    Memoria asociativa hiperdimensional para secuencias de texto.
    
    Usa binding (XOR) y bundling (suma) para representar secuencias
    completas como vectores individuales.
    """
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.item_memory = {}  # word -> HDC vector
        self.sequence_memory = []  # lista de (seq_vector, words, context)
        self.rng = np.random.RandomState(42)
    
    def _get_item_vector(self, word: str) -> np.ndarray:
        """Obtiene o crea vector HDC para una palabra."""
        if word not in self.item_memory:
            # Vector bipolar aleatorio {-1, +1}
            vec = self.rng.choice([-1, 1], size=self.dim).astype(np.float32)
            self.item_memory[word] = vec
        return self.item_memory[word]
    
    def _bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Binding: combina dos vectores (XOR en bipolar = multiplicación)."""
        return v1 * v2
    
    def _bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundling: combina múltiples vectores (suma + signo)."""
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        bundled = np.sum(vectors, axis=0)
        return np.sign(bundled).astype(np.float32)
    
    def encode_sequence(self, words: List[str], position_encoding: bool = True) -> np.ndarray:
        """
        Codifica una secuencia de palabras como un vector HDC.
        
        Args:
            words: Lista de palabras
            position_encoding: Si True, incluye información de posición
        
        Returns:
            Vector HDC representando la secuencia completa
        """
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        
        sequence_vectors = []
        for i, word in enumerate(words):
            word_vec = self._get_item_vector(word)
            
            if position_encoding:
                # Crear vector de posición
                pos_vec = self._get_item_vector(f"__POS_{i}__")
                # Bind palabra con posición
                bound = self._bind(word_vec, pos_vec)
            else:
                bound = word_vec
            
            sequence_vectors.append(bound)
        
        # Bundle todos los vectores de la secuencia
        return self._bundle(sequence_vectors)
    
    def store_sequence(self, words: List[str], context: Optional[str] = None):
        """
        Almacena una secuencia en memoria.
        
        Args:
            words: Secuencia de palabras
            context: Contexto adicional (ej: categoría temática)
        """
        seq_vec = self.encode_sequence(words)
        self.sequence_memory.append({
            'vector': seq_vec,
            'words': words,
            'context': context
        })
    
    def find_similar(self, query_words: List[str], top_k: int = 5) -> List[Tuple[float, List[str]]]:
        """
        Encuentra secuencias similares a las palabras dadas.
        
        Args:
            query_words: Palabras de consulta
            top_k: Número de resultados a retornar
        
        Returns:
            Lista de (similitud, secuencia)
        """
        if not self.sequence_memory:
            return []
        
        query_vec = self.encode_sequence(query_words)
        
        # Calcular similitud coseno con todas las secuencias
        similarities = []
        for stored in self.sequence_memory:
            sim = np.dot(query_vec, stored['vector']) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored['vector']) + 1e-8
            )
            similarities.append((sim, stored['words']))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]


class FreeEnergySelector:
    """
    Selector de palabras basado en el Principio de Energía Libre (FEP).
    
    Selecciona la siguiente palabra que minimiza la energía libre,
    combinando probabilidad del contexto y coherencia con el prefijo.
    """
    
    def __init__(self):
        self.bigram_counts = defaultdict(Counter)  # word -> Counter(next_word)
        self.trigram_counts = defaultdict(Counter)  # (w1, w2) -> Counter(next_word)
        self.unigram_counts = Counter()
    
    def train(self, sentences: List[List[str]]):
        """Entrena el modelo con oraciones del corpus."""
        for sent in sentences:
            words = [w.lower() for w in sent]
            
            for i, word in enumerate(words):
                self.unigram_counts[word] += 1
                
                if i > 0:
                    prev = words[i-1]
                    self.bigram_counts[prev][word] += 1
                
                if i > 1:
                    prev2 = (words[i-2], words[i-1])
                    self.trigram_counts[prev2][word] += 1
    
    def compute_free_energy(self, prefix: List[str], candidate: str) -> float:
        """
        Calcula la energía libre de un candidato dado un prefijo.
        
        F = -log P(candidate | prefix)
        
        Menor energía = mayor probabilidad = mejor candidato
        
        Args:
            prefix: Palabras anteriores
            candidate: Palabra candidata
        
        Returns:
            Energía libre (negativo = mejor)
        """
        candidate = candidate.lower()
        prefix_lower = [w.lower() for w in prefix]
        
        # Probabilidad trigram (si hay al menos 2 palabras)
        if len(prefix_lower) >= 2:
            context = (prefix_lower[-2], prefix_lower[-1])
            if context in self.trigram_counts:
                total = sum(self.trigram_counts[context].values())
                count = self.trigram_counts[context].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * len(self.unigram_counts))
                return -np.log(prob + 1e-10)
        
        # Probabilidad bigram (fallback)
        if len(prefix_lower) >= 1:
            prev = prefix_lower[-1]
            if prev in self.bigram_counts:
                total = sum(self.bigram_counts[prev].values())
                count = self.bigram_counts[prev].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * len(self.unigram_counts))
                return -np.log(prob + 1e-10)
        
        # Probabilidad unigram (fallback final)
        total = sum(self.unigram_counts.values())
        count = self.unigram_counts.get(candidate, 0)
        prob = (count + 0.1) / (total + 0.1 * len(self.unigram_counts))
        return -np.log(prob + 1e-10)
    
    def select_next(self, prefix: List[str], candidates: List[str]) -> Tuple[str, float]:
        """
        Selecciona la mejor palabra candidata minimizando energía libre.
        
        Args:
            prefix: Palabras anteriores
            candidates: Lista de palabras candidatas
        
        Returns:
            (mejor_palabra, energía_libre)
        """
        if not candidates:
            return "", float('inf')
        
        best_word = candidates[0]
        best_energy = float('inf')
        
        for candidate in candidates:
            energy = self.compute_free_energy(prefix, candidate)
            if energy < best_energy:
                best_energy = energy
                best_word = candidate
        
        return best_word, best_energy


class TextGenerator:
    """
    Generador de texto biomimético completo.
    
    Combina:
    - Memoria asociativa HDC para almacenar secuencias
    - Recuperación por similitud contextual
    - Selección con FEP para coherencia
    """
    
    def __init__(self, dim: int = 10000):
        self.memory = HDCSequenceMemory(dim=dim)
        self.selector = FreeEnergySelector()
        self.vocabulary = set()
        self.is_trained = False
    
    def train_from_corpus(self, corpus_path: str, max_sentences: int = 5000):
        """
        Entrena el generador con un corpus CoNLL-U.
        
        Args:
            corpus_path: Ruta al archivo .conllu
            max_sentences: Máximo de oraciones a usar
        """
        print(f"Cargando corpus desde {corpus_path}...")
        
        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, sent in enumerate(parse_incr(f)):
                if i >= max_sentences:
                    break
                
                words = [token['form'] for token in sent 
                        if not isinstance(token['id'], tuple)]
                
                if len(words) >= 3:  # Solo oraciones con al menos 3 palabras
                    sentences.append(words)
                    self.vocabulary.update(words)
                    
                    # Almacenar en memoria HDC
                    self.memory.store_sequence(words)
        
        print(f"Entrenando selector FEP con {len(sentences)} oraciones...")
        self.selector.train(sentences)
        
        self.is_trained = True
        print(f"✓ Entrenamiento completo")
        print(f"  - Vocabulario: {len(self.vocabulary)} palabras")
        print(f"  - Secuencias en memoria: {len(self.memory.sequence_memory)}")
    
    def generate(self, prefix: str, max_words: int = 15, 
                temperature: float = 1.0) -> str:
        """
        Genera texto a partir de un prefijo.
        
        Usa una estrategia de ventana deslizante:
        1. Busca secuencias en memoria que contengan el contexto reciente
        2. Extrae la siguiente palabra de cada match
        3. Selecciona con FEP (energía libre mínima)
        4. Aplica mecanismo de fatiga para evitar repeticiones
        
        Args:
            prefix: Texto inicial (ej: "El presidente")
            max_words: Máximo de palabras a generar
            temperature: Controla creatividad (0.1=determinista, 2.0=creativo)
        
        Returns:
            Texto generado
        """
        if not self.is_trained:
            raise RuntimeError("El generador no está entrenado. Llama a train_from_corpus() primero.")
        
        # Tokenizar prefijo
        words = prefix.strip().split()
        if not words:
            words = ["El"]
        
        generated = list(words)
        
        for step in range(max_words):
            candidates = self._find_next_candidates(generated)
            
            if not candidates:
                # Fallback: usar selector FEP con vocabulario completo
                candidates = list(self.vocabulary)[:100]
            
            # Filtrar palabras repetidas consecutivamente (mecanismo de fatiga)
            if len(generated) >= 2:
                last_two = set(generated[-2:])
                filtered_candidates = [c for c in candidates if c not in last_two]
                if filtered_candidates:
                    candidates = filtered_candidates
            
            # Evitar repetir la misma palabra más de 2 veces en ventana de 5
            if len(generated) >= 5:
                recent_window = generated[-5:]
                # Contar ocurrencias de cada candidata en ventana reciente
                diverse_candidates = []
                for c in candidates:
                    count = recent_window.count(c)
                    # Palabras funcionales pueden repetirse más
                    if c in ['de', 'la', 'el', 'y', 'en', 'a', 'que', 'por', 'con']:
                        if count < 3:
                            diverse_candidates.append(c)
                    else:
                        if count < 2:
                            diverse_candidates.append(c)
                
                if diverse_candidates:
                    candidates = diverse_candidates
            
            # Seleccionar siguiente palabra
            if temperature <= 0.1:
                # Determinista: mínima energía libre
                next_word, _ = self.selector.select_next(generated, candidates)
            else:
                # Estocástico: muestreo ponderado por energía libre
                next_word = self._sample_with_temperature(generated, candidates, temperature)
            
            # Manejo mejorado de puntuación
            if next_word in ['.', '!', '?', ',', ';']:
                if step > 3:  # Solo parar si ya generamos algo
                    generated.append(next_word)
                    break
                else:
                    # Muy corto, intentar otra candidata
                    continue
            
            # Evitar empezar con puntuación
            if step == 0 and next_word in [',', '.', '!', '?', ';', ':']:
                continue
            
            generated.append(next_word)
        
        return ' '.join(generated)
    
    def _find_next_candidates(self, generated: List[str]) -> List[str]:
        """
        Encuentra palabras candidatas para continuar la secuencia.
        
        Estrategia: buscar en memoria secuencias que contengan las últimas
        1-3 palabras generadas y extraer la palabra siguiente.
        """
        candidates = []
        
        # Intentar con contextos de diferentes longitudes (3, 2, 1 palabras)
        for context_len in [3, 2, 1]:
            if len(generated) < context_len:
                continue
            
            context = [w.lower() for w in generated[-context_len:]]
            
            # Buscar en todas las secuencias almacenadas
            for stored in self.memory.sequence_memory:
                seq_lower = [w.lower() for w in stored['words']]
                
                # Buscar dónde aparece el contexto
                for i in range(len(seq_lower) - context_len):
                    if seq_lower[i:i+context_len] == context:
                        # Encontró match - agregar la siguiente palabra
                        next_idx = i + context_len
                        if next_idx < len(stored['words']):
                            next_word = stored['words'][next_idx]
                            # Evitar palabras de posición (__POS_x__)
                            if not next_word.startswith('__'):
                                candidates.append(next_word)
            
            # Si encontramos candidatas con este contexto, usarlas
            if candidates:
                break
        
        return candidates
    
    def _sample_with_temperature(self, prefix: List[str], candidates: List[str], 
                                  temperature: float) -> str:
        """Muestreo ponderado por energía libre con temperatura."""
        if not candidates:
            return ""
        
        # Calcular energías para cada candidata
        energies = np.array([
            self.selector.compute_free_energy(prefix, c) 
            for c in candidates
        ])
        
        # Convertir a probabilidades con temperatura
        # Energía más baja = probabilidad más alta
        scaled = -energies / max(temperature, 0.01)
        scaled -= np.max(scaled)  # estabilidad numérica
        probs = np.exp(scaled)
        prob_sum = probs.sum()
        
        if prob_sum == 0:
            return candidates[0]
        
        probs = probs / prob_sum
        
        # Muestrear
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def complete_sentence(self, prefix: str) -> str:
        """
        Completa una oración de forma coherente.
        
        Args:
            prefix: Inicio de la oración
        
        Returns:
            Oración completa
        """
        generated = self.generate(prefix, max_words=20, temperature=0.5)
        
        # Asegurar que termine con puntuación
        if not generated.rstrip().endswith(('.', '!', '?')):
            generated += '.'
        
        return generated


# Función de conveniencia para uso rápido
def create_generator(corpus_path: str = None) -> TextGenerator:
    """
    Crea y entrena un generador de texto.
    
    Args:
        corpus_path: Ruta al corpus (por defecto usa UD Spanish-AnCora)
    
    Returns:
        Generador entrenado
    """
    if corpus_path is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    generator = TextGenerator(dim=10000)
    generator.train_from_corpus(corpus_path, max_sentences=5000)
    return generator


if __name__ == "__main__":
    # Ejemplo de uso
    print("=" * 70)
    print("Generador de Texto Biomimético - Demo")
    print("=" * 70)
    
    gen = create_generator()
    
    print("\nEjemplos de generación:")
    print("-" * 70)
    
    prompts = [
        "El presidente",
        "La economía",
        "En la ciudad",
        "Los científicos"
    ]
    
    for prompt in prompts:
        result = gen.complete_sentence(prompt)
        print(f"\n{prompt}...")
        print(f"→ {result}")
