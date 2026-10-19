"""
Generador de Texto Biomimético v2 - HDC + Memoria Asociativa + FEP

Mejoras sobre v1:
- Búsqueda multi-estrategia (exacta, palabras clave, similitud HDC)
- Quadgramas (4-gramas) para mejor coherencia
- Splicing: combina fragmentos de diferentes oraciones
- Identidad influye en búsqueda Y selección
- Anti-repetición mejorada (no repite el prompt como respuesta)
- Normalización de texto para mejor matching
- Mecanismo de "continuación" cuando no hay coincidencia exacta

Autor: Laboratorio Micro-IA Biomimética
Fase: 7 - Generación de Texto v2
"""

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from conllu import parse_incr
from typing import List, Tuple, Optional, Dict, Set
import json
import re


# Palabras funcionales que no aportan significado para búsqueda
STOP_WORDS = {
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
    'the', 'of', 'and', 'to', 'in', 'is', 'it', 'that', 'was', 'for',
}


def normalize_word(word: str) -> str:
    """Normaliza una palabra para matching: lowercase y sin puntuación."""
    return re.sub(r'[^\wáéíóúñü]', '', word.lower())


def is_content_word(word: str) -> bool:
    """Verifica si una palabra es de contenido (no stop word)."""
    norm = normalize_word(word)
    return len(norm) > 2 and norm not in STOP_WORDS


class HDCSequenceMemory:
    """
    Memoria asociativa hiperdimensional para secuencias de texto.
    
    Usa binding (XOR) y bundling (suma) para representar secuencias
    completas como vectores individuales.
    """
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.item_memory = {}  # word -> HDC vector
        self.sequence_memory = []  # lista de dicts con seq info
        self.word_to_sequences = defaultdict(list)  # word -> [indices] donde aparece
        self.rng = np.random.RandomState(42)
    
    def _get_item_vector(self, word: str) -> np.ndarray:
        """Obtiene o crea vector HDC para una palabra."""
        if word not in self.item_memory:
            vec = self.rng.choice([-1, 1], size=self.dim).astype(np.float32)
            self.item_memory[word] = vec
        return self.item_memory[word]
    
    def _bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Binding: combina dos vectores (multiplicación elemento a elemento)."""
        return v1 * v2
    
    def _bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundling: combina múltiples vectores (suma + signo)."""
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        bundled = np.sum(vectors, axis=0)
        return np.sign(bundled).astype(np.float32)
    
    def encode_sequence(self, words: List[str], position_encoding: bool = True) -> np.ndarray:
        """Codifica una secuencia de palabras como un vector HDC."""
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        
        sequence_vectors = []
        for i, word in enumerate(words):
            word_vec = self._get_item_vector(word.lower())
            
            if position_encoding:
                pos_vec = self._get_item_vector(f"__POS_{i}__")
                bound = self._bind(word_vec, pos_vec)
            else:
                bound = word_vec
            
            sequence_vectors.append(bound)
        
        return self._bundle(sequence_vectors)
    
    def store_sequence(self, words: List[str], context: Optional[str] = None):
        """
        Almacena una secuencia en memoria con índice invertido para búsqueda rápida.
        """
        seq_vec = self.encode_sequence(words)
        idx = len(self.sequence_memory)
        
        self.sequence_memory.append({
            'vector': seq_vec,
            'words': words,
            'words_lower': [w.lower() for w in words],
            'context': context
        })
        
        # Índice invertido: cada palabra apunta a las secuencias que la contienen
        seen_words = set()
        for word in words:
            norm = normalize_word(word)
            if norm and norm not in seen_words:
                self.word_to_sequences[norm].append(idx)
                seen_words.add(norm)
    
    def find_by_keywords(self, keywords: List[str], top_k: int = 20) -> List[Tuple[float, int]]:
        """
        Busca secuencias que contengan palabras clave.
        
        Args:
            keywords: Palabras clave a buscar
            top_k: Máximo de resultados
        
        Returns:
            Lista de (score, sequence_index) ordenada por relevancia
        """
        if not keywords:
            return []
        
        # Contar cuántas keywords aparecen en cada secuencia
        seq_scores = Counter()
        norm_keywords = [normalize_word(k) for k in keywords if normalize_word(k)]
        
        for kw in norm_keywords:
            if kw in self.word_to_sequences:
                for idx in self.word_to_sequences[kw]:
                    seq_scores[idx] += 1
        
        # Ordenar por score descendente
        ranked = seq_scores.most_common(top_k)
        return [(score, idx) for idx, score in ranked]
    
    def find_similar(self, query_words: List[str], top_k: int = 5) -> List[Tuple[float, List[str]]]:
        """Encuentra secuencias similares por similitud coseno HDC."""
        if not self.sequence_memory:
            return []
        
        query_vec = self.encode_sequence(query_words)
        
        similarities = []
        for stored in self.sequence_memory:
            sim = np.dot(query_vec, stored['vector']) / (
                np.linalg.norm(query_vec) * np.linalg.norm(stored['vector']) + 1e-8
            )
            similarities.append((sim, stored['words']))
        
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]


class FreeEnergySelector:
    """
    Selector de palabras basado en el Principio de Energía Libre (FEP).
    
    Usa n-gramas de orden 1 a 4 para mejor coherencia.
    """
    
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.quadgram_counts = defaultdict(Counter)
        self.total_words = 0
    
    def train(self, sentences: List[List[str]]):
        """Entrena el modelo con oraciones del corpus."""
        for sent in sentences:
            words = [w.lower() for w in sent]
            
            for i, word in enumerate(words):
                self.unigram_counts[word] += 1
                self.total_words += 1
                
                if i > 0:
                    self.bigram_counts[words[i-1]][word] += 1
                
                if i > 1:
                    ctx = (words[i-2], words[i-1])
                    self.trigram_counts[ctx][word] += 1
                
                if i > 2:
                    ctx = (words[i-3], words[i-2], words[i-1])
                    self.quadgram_counts[ctx][word] += 1
    
    def compute_free_energy(self, prefix: List[str], candidate: str) -> float:
        """
        Calcula la energía libre de un candidato dado un prefijo.
        Usa interpolación de n-gramas de orden 1-4.
        """
        candidate = candidate.lower()
        prefix_lower = [w.lower() for w in prefix]
        vocab_size = max(len(self.unigram_counts), 1)
        
        # Pesos para interpolación (más peso a n-gramas de orden superior)
        energies = []
        weights = []
        
        # Quadgrama (peso 8)
        if len(prefix_lower) >= 3:
            ctx = (prefix_lower[-3], prefix_lower[-2], prefix_lower[-1])
            if ctx in self.quadgram_counts:
                total = sum(self.quadgram_counts[ctx].values())
                count = self.quadgram_counts[ctx].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * vocab_size)
                energies.append(-np.log(prob + 1e-10))
                weights.append(8.0)
        
        # Trigram (peso 4)
        if len(prefix_lower) >= 2:
            ctx = (prefix_lower[-2], prefix_lower[-1])
            if ctx in self.trigram_counts:
                total = sum(self.trigram_counts[ctx].values())
                count = self.trigram_counts[ctx].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * vocab_size)
                energies.append(-np.log(prob + 1e-10))
                weights.append(4.0)
        
        # Bigram (peso 2)
        if len(prefix_lower) >= 1:
            prev = prefix_lower[-1]
            if prev in self.bigram_counts:
                total = sum(self.bigram_counts[prev].values())
                count = self.bigram_counts[prev].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * vocab_size)
                energies.append(-np.log(prob + 1e-10))
                weights.append(2.0)
        
        # Unigram (peso 1)
        total = max(self.total_words, 1)
        count = self.unigram_counts.get(candidate, 0)
        prob = (count + 0.1) / (total + 0.1 * vocab_size)
        energies.append(-np.log(prob + 1e-10))
        weights.append(1.0)
        
        # Interpolar energías
        total_weight = sum(weights)
        avg_energy = sum(e * w for e, w in zip(energies, weights)) / total_weight
        
        return avg_energy
    
    def select_next(self, prefix: List[str], candidates: List[str]) -> Tuple[str, float]:
        """Selecciona la mejor palabra minimizando energía libre."""
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
    
    def get_top_continuations(self, prefix: List[str], top_k: int = 20) -> List[str]:
        """
        Obtiene las top-k continuaciones más probables según n-gramas.
        
        Esto es clave: en vez de buscar solo en la memoria HDC,
        usa las estadísticas de n-gramas para sugerir palabras.
        """
        prefix_lower = [w.lower() for w in prefix]
        candidate_scores = Counter()
        
        # Quadgrama
        if len(prefix_lower) >= 3:
            ctx = (prefix_lower[-3], prefix_lower[-2], prefix_lower[-1])
            if ctx in self.quadgram_counts:
                for word, count in self.quadgram_counts[ctx].most_common(top_k):
                    candidate_scores[word] += count * 8
        
        # Trigram
        if len(prefix_lower) >= 2:
            ctx = (prefix_lower[-2], prefix_lower[-1])
            if ctx in self.trigram_counts:
                for word, count in self.trigram_counts[ctx].most_common(top_k):
                    candidate_scores[word] += count * 4
        
        # Bigram
        if len(prefix_lower) >= 1:
            prev = prefix_lower[-1]
            if prev in self.bigram_counts:
                for word, count in self.bigram_counts[prev].most_common(top_k):
                    candidate_scores[word] += count * 2
        
        return [word for word, _ in candidate_scores.most_common(top_k)]


class TextGenerator:
    """
    Generador de texto biomimético v2.
    
    Mejoras sobre v1:
    - Búsqueda multi-estrategia (exacta + keywords + n-gramas)
    - Quadgramas para mejor coherencia
    - Splicing: combina fragmentos de diferentes oraciones
    - Identidad influye en búsqueda Y selección
    - Anti-repetición mejorada
    """
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory = HDCSequenceMemory(dim=dim)
        self.selector = FreeEnergySelector()
        self.vocabulary = set()
        self.is_trained = False
        self.identity = None
        self._recently_used = set()  # Para fatiga global
    
    def train_from_corpus(self, corpus_path: str, max_sentences: int = 5000):
        """Entrena el generador con un corpus CoNLL-U."""
        print(f"Cargando corpus desde {corpus_path}...")
        
        sentences = []
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for i, sent in enumerate(parse_incr(f)):
                if i >= max_sentences:
                    break
                
                words = [token['form'] for token in sent 
                        if not isinstance(token['id'], tuple)]
                
                if len(words) >= 3:
                    sentences.append(words)
                    self.vocabulary.update(w.lower() for w in words)
                    self.memory.store_sequence(words)
        
        print(f"Entrenando selector FEP con {len(sentences)} oraciones...")
        self.selector.train(sentences)
        
        self.is_trained = True
        print(f"✓ Entrenamiento completo")
        print(f"  - Vocabulario: {len(self.vocabulary)} palabras")
        print(f"  - Secuencias en memoria: {len(self.memory.sequence_memory)}")
    
    def set_identity(self, identity):
        """Configura la identidad/personalidad de la IA."""
        self.identity = identity
        print(f"✓ Identidad configurada: {identity.name}")
        print(f"  - Estilo: {identity.style}")
        print(f"  - Tono emocional: {identity.emotional_tone}")
        print(f"  - Temperatura: {identity.temperature}")
    
    def generate(self, prefix: str, max_words: int = 20, 
                temperature: float = None) -> str:
        """
        Genera texto a partir de un prefijo usando búsqueda multi-estrategia.
        
        Estrategias (en orden de prioridad):
        1. Coincidencia exacta de las últimas 3/2/1 palabras en secuencias
        2. Búsqueda por palabras clave del prompt (contenido)
        3. Continuaciones estadísticas (n-gramas)
        4. Fallback con palabras frecuentes
        """
        if not self.is_trained:
            raise RuntimeError("El generador no está entrenado.")
        
        if temperature is None:
            temperature = self.identity.temperature if self.identity else 1.0
        
        # Tokenizar y limpiar prefijo
        words = prefix.strip().split()
        if not words:
            words = ["El"]
        
        # Extraer palabras de contenido del prompt para búsqueda
        prompt_keywords = [w for w in words if is_content_word(w)]
        
        generated = list(words)
        original_prefix_lower = set(w.lower() for w in words)
        
        # Flag para saber si estamos generando contenido nuevo
        found_context = False
        
        for step in range(max_words):
            candidates = []
            
            # ═══ ESTRATEGIA 1: Coincidencia exacta en memoria ═══
            exact_candidates = self._find_exact_continuations(generated)
            if exact_candidates:
                candidates.extend(exact_candidates)
                found_context = True
            
            # ═══ ESTRATEGIA 2: Búsqueda por palabras clave ═══
            if not candidates or len(candidates) < 5:
                keyword_candidates = self._find_keyword_continuations(
                    generated, prompt_keywords
                )
                if keyword_candidates:
                    candidates.extend(keyword_candidates)
                    found_context = True
            
            # ═══ ESTRATEGIA 3: Continuaciones estadísticas (n-gramas) ═══
            if not candidates or len(candidates) < 5:
                ngram_candidates = self.selector.get_top_continuations(generated, top_k=30)
                if ngram_candidates:
                    candidates.extend(ngram_candidates)
                    found_context = True
            
            # ═══ ESTRATEGIA 4: Fallback ═══
            if not candidates:
                # Usar palabras frecuentes del vocabulario
                top_words = self.selector.unigram_counts.most_common(50)
                candidates = [w for w, _ in top_words]
            
            # ═══ FILTROS ═══
            candidates = self._apply_filters(candidates, generated, original_prefix_lower)
            
            # ═══ APLICAR IDENTIDAD ═══
            if self.identity and self.identity.vocabulary_bias:
                candidates = self._apply_vocabulary_bias(candidates)
            
            if not candidates:
                break
            
            # ═══ SELECCIÓN ═══
            if temperature <= 0.1:
                next_word, _ = self.selector.select_next(generated, candidates)
            else:
                next_word = self._sample_with_temperature(generated, candidates, temperature)
            
            if not next_word:
                break
            
            # ═══ MANEJO DE PUNTUACIÓN ═══
            if next_word in ['.', '!', '?']:
                if step > 5:
                    generated.append(next_word)
                    break
                else:
                    continue
            
            if next_word in [',', ';', ':']:
                if step > 2:
                    generated.append(next_word)
                    continue
                else:
                    continue
            
            generated.append(next_word)
        
        # ═══ POST-PROCESAMIENTO ═══
        result = ' '.join(generated)
        
        # Si solo repitió el prompt, intentar generar algo más
        if result.strip().lower() == prefix.strip().lower():
            result = self._force_generation(prefix, prompt_keywords, temperature)
        
        return result
    
    def _find_exact_continuations(self, generated: List[str]) -> List[str]:
        """
        Estrategia 1: Busca coincidencias exactas de las últimas 1-4 palabras
        en las secuencias almacenadas.
        """
        candidates = []
        
        for context_len in [4, 3, 2, 1]:
            if len(generated) < context_len:
                continue
            
            context = [w.lower() for w in generated[-context_len:]]
            
            for stored in self.memory.sequence_memory:
                seq_lower = stored['words_lower']
                
                for i in range(len(seq_lower) - context_len):
                    if seq_lower[i:i+context_len] == context:
                        next_idx = i + context_len
                        if next_idx < len(stored['words']):
                            next_word = stored['words'][next_idx]
                            if not next_word.startswith('__'):
                                candidates.append(next_word)
            
            if candidates:
                break
        
        return candidates
    
    def _find_keyword_continuations(self, generated: List[str], 
                                     keywords: List[str]) -> List[str]:
        """
        Estrategia 2: Busca secuencias que contengan palabras clave del prompt
        y extrae palabras que siguen a esas keywords en las secuencias.
        """
        if not keywords:
            return []
        
        candidates = []
        
        # Buscar secuencias que contengan las keywords
        matches = self.memory.find_by_keywords(keywords, top_k=30)
        
        last_word_lower = generated[-1].lower() if generated else ""
        
        for score, idx in matches:
            stored = self.memory.sequence_memory[idx]
            seq_lower = stored['words_lower']
            seq_words = stored['words']
            
            # Buscar dónde aparece la última palabra generada en esta secuencia
            for i, word in enumerate(seq_lower):
                if word == last_word_lower and i + 1 < len(seq_words):
                    candidates.append(seq_words[i + 1])
            
            # También extraer palabras que siguen a las keywords
            for kw in keywords:
                kw_lower = kw.lower()
                for i, word in enumerate(seq_lower):
                    if word == kw_lower:
                        # Extraer la secuencia desde esta keyword
                        for j in range(i + 1, min(i + 4, len(seq_words))):
                            candidates.append(seq_words[j])
        
        return candidates
    
    def _apply_filters(self, candidates: List[str], generated: List[str],
                       original_prefix: Set[str]) -> List[str]:
        """Aplica filtros de calidad a las candidatas."""
        if not candidates:
            return candidates
        
        filtered = []
        gen_lower = [w.lower() for w in generated]
        
        for c in candidates:
            c_lower = c.lower()
            
            # No repetir exactamente el prompt
            if c_lower in original_prefix and len(generated) <= len(original_prefix):
                continue
            
            # No repetir las últimas 2 palabras
            if c_lower in gen_lower[-2:]:
                continue
            
            # Limitar repeticiones en ventana de 6 palabras
            if len(gen_lower) >= 6:
                recent = gen_lower[-6:]
                if recent.count(c_lower) >= 2:
                    continue
            
            # Filtrar tokens especiales
            if c.startswith('__') or c.startswith('http'):
                continue
            
            filtered.append(c)
        
        return filtered if filtered else candidates
    
    def _force_generation(self, prefix: str, keywords: List[str], 
                          temperature: float) -> str:
        """
        Cuando el generador solo repite el prompt, fuerza una generación
        buscando las oraciones más relevantes y extrayendo fragmentos.
        """
        words = prefix.strip().split()
        
        # Buscar oraciones relevantes por keywords
        if keywords:
            matches = self.memory.find_by_keywords(keywords, top_k=10)
            
            if matches:
                # Tomar la mejor match y extraer un fragmento relevante
                _, best_idx = matches[0]
                stored = self.memory.sequence_memory[best_idx]
                seq_words = stored['words']
                
                # Buscar dónde aparece la keyword
                for kw in keywords:
                    kw_lower = kw.lower()
                    for i, w in enumerate(stored['words_lower']):
                        if w == kw_lower:
                            # Extraer desde esta posición
                            start = max(0, i - 2)
                            end = min(len(seq_words), i + 25)
                            fragment = seq_words[start:end]
                            
                            # Combinar prefijo con fragmento
                            result_words = words + [w for w in fragment 
                                                    if w.lower() not in set(x.lower() for x in words)]
                            return ' '.join(result_words[:len(words) + 20])
        
        # Si no hay keywords o no se encontró, usar n-gramas puros
        generated = list(words)
        for _ in range(15):
            top = self.selector.get_top_continuations(generated, top_k=10)
            if not top:
                break
            
            # Filtrar
            top = [w for w in top if w.lower() not in set(x.lower() for x in generated[-3:])]
            if not top:
                break
            
            if temperature <= 0.5:
                next_word = top[0]
            else:
                next_word = np.random.choice(top[:min(5, len(top))])
            
            if next_word in ['.', '!', '?'] and len(generated) > len(words) + 5:
                generated.append(next_word)
                break
            
            generated.append(next_word)
        
        return ' '.join(generated)
    
    def _sample_with_temperature(self, prefix: List[str], candidates: List[str], 
                                  temperature: float) -> str:
        """Muestreo ponderado por energía libre con temperatura."""
        if not candidates:
            return ""
        
        energies = np.array([
            self.selector.compute_free_energy(prefix, c) 
            for c in candidates
        ])
        
        scaled = -energies / max(temperature, 0.01)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        prob_sum = probs.sum()
        
        if prob_sum == 0:
            return candidates[0]
        
        probs = probs / prob_sum
        
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def _apply_vocabulary_bias(self, candidates: List[str]) -> List[str]:
        """
        Aplica el vocabulary_bias de la identidad.
        Las palabras del estilo se priorizan con más peso.
        """
        if not self.identity or not self.identity.vocabulary_bias:
            return candidates
        
        weighted = []
        bias_lower = {k.lower(): v for k, v in self.identity.vocabulary_bias.items()}
        
        for word in candidates:
            weighted.append(word)
            w_lower = word.lower()
            if w_lower in bias_lower:
                weight = bias_lower[w_lower]
                for _ in range(max(1, int(weight))):
                    weighted.append(word)
        
        return weighted
    
    def complete_sentence(self, prefix: str) -> str:
        """Completa una oración de forma coherente."""
        generated = self.generate(prefix, max_words=25, temperature=0.6)
        
        if not generated.rstrip().endswith(('.', '!', '?')):
            generated += '.'
        
        return generated


def create_generator(corpus_path: str = None) -> TextGenerator:
    """Crea y entrena un generador de texto."""
    if corpus_path is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    
    generator = TextGenerator(dim=10000)
    generator.train_from_corpus(corpus_path, max_sentences=5000)
    return generator
