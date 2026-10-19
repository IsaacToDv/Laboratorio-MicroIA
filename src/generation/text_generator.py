"""
Generador de Texto Biomimético v3 - Arquitectura Expandida

Mejoras v3:
- Dimensión HDC expandida (20,000)
- 5-gramas para coherencia de largo alcance
- Splicing multi-oración: combina fragmentos de múltiples oraciones
- Clasificador de intención del prompt
- Selección automática de identidad según tema
- Ventana de contexto extendida (6 palabras)

Autor: Laboratorio Micro-IA Biomimética
"""

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from conllu import parse_incr
from typing import List, Tuple, Optional, Dict, Set
import json
import re


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
    'ha', 'he', 'te', 'si', 'han', 'fue', 'era', 'tan', 'les', 'ven',
}


def normalize_word(word: str) -> str:
    return re.sub(r'[^\wáéíóúñü]', '', word.lower())


def is_content_word(word: str) -> bool:
    norm = normalize_word(word)
    return len(norm) > 2 and norm not in STOP_WORDS


# ═══════════════════════════════════════════════════════════════════
# CLASIFICADOR DE INTENCIÓN (para UX libre)
# ═══════════════════════════════════════════════════════════════════

# Palabras clave por tema/identidad
TOPIC_KEYWORDS = {
    'poeta': {
        'amor', 'corazón', 'alma', 'sueño', 'luna', 'noche', 'muerte',
        'dolor', 'esperanza', 'tristeza', 'pasión', 'sentimiento', 'llanto',
        'verso', 'poema', 'poesía', 'belleza', 'flor', 'mar', 'cielo',
        'estrella', 'silencio', 'soledad', 'beso', 'suspiro', 'eterno'
    },
    'filosofo': {
        'dios', 'existencia', 'verdad', 'razón', 'libertad', 'conciencia',
        'ser', 'pensar', 'pensamiento', 'duda', 'fe', 'ética', 'moral',
        'conocimiento', 'sabiduría', 'vida', 'muerte', 'sentido', 'alma',
        'espíritu', 'realidad', 'tiempo', 'infinito', 'nada', 'absoluto',
        'filosofía', 'platón', 'aristóteles', 'kant', 'hegel', 'sartre',
        'descartes', 'sócrates', 'nietzsche', 'metafísica', 'ontología'
    },
    'narrador': {
        'había', 'cuenta', 'historia', 'personaje', 'pueblo', 'familia',
        'recuerdo', 'memoria', 'pasado', 'infancia', 'abuelo', 'abuela',
        'casa', 'calle', 'viaje', 'aventura', 'secreto', 'destino',
        'guerra', 'paz', 'tierra', 'lluvia', 'amanecer', 'atardecer',
        'garcía márquez', 'borges', 'cortázar', 'allende', 'ruiz'
    },
    'ensayista': {
        'cultura', 'sociedad', 'literatura', 'lenguaje', 'tradición',
        'modernidad', 'identidad', 'historia', 'civilización', 'política',
        'economía', 'educación', 'ciencia', 'tecnología', 'arte',
        'análisis', 'tesis', 'argumento', 'evidencia', 'conclusión',
        'perspectiva', 'paradigma', 'discurso', 'contexto', 'fenómeno'
    },
    'contemporaneo': {
        'google', 'internet', 'app', 'red', 'digital', 'tecnología',
        'datos', 'software', 'programa', 'código', 'web', 'pantalla',
        'usuario', 'plataforma', ' startup', 'innovación', 'global',
        'actual', 'hoy', 'ahora', 'presente', 'futuro', 'cambio'
    }
}

# Patrones para detectar tipo de intención
QUESTION_MARKERS = ['qué', 'quién', 'cómo', 'cuándo', 'dónde', 'por qué',
                     'cuál', 'cuánto', 'cuántos', 'cuántas']


def classify_intent(text: str) -> Dict:
    """
    Clasifica la intención y tema del prompt.
    
    Returns:
        dict con: {'identity': str, 'is_question': bool, 'keywords': list, 'confidence': float}
    """
    words = [normalize_word(w) for w in text.split()]
    content_words = [w for w in words if is_content_word(w)]
    
    # ¿Es pregunta?
    is_question = any(m in words for m in QUESTION_MARKERS) or text.strip().startswith('?') or text.strip().endswith('?')
    
    # Buscar mejor identidad por palabras clave
    scores = {}
    for identity, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for w in content_words if w in keywords)
        scores[identity] = score
    
    best_identity = max(scores, key=scores.get) if scores else 'contemporaneo'
    confidence = scores.get(best_identity, 0)
    
    # Si no hay match fuerte, usar contemporáneo como default
    if confidence == 0:
        best_identity = 'contemporaneo'
    
    return {
        'identity': best_identity,
        'is_question': is_question,
        'keywords': content_words[:5],
        'confidence': confidence
    }


# ═══════════════════════════════════════════════════════════════════
# MEMORIA HDC EXPANDIDA
# ═══════════════════════════════════════════════════════════════════

class HDCSequenceMemory:
    """
    Memoria asociativa hiperdimensional expandida.
    Dimensión por defecto: 20,000 (antes 10,000)
    """
    
    def __init__(self, dim: int = 20000):
        self.dim = dim
        self.item_memory = {}
        self.sequence_memory = []
        self.word_to_sequences = defaultdict(list)
        self.bigram_index = defaultdict(list)  # (w1,w2) -> [indices]
        self.rng = np.random.RandomState(42)
    
    def _get_item_vector(self, word: str) -> np.ndarray:
        if word not in self.item_memory:
            vec = self.rng.choice([-1, 1], size=self.dim).astype(np.float32)
            self.item_memory[word] = vec
        return self.item_memory[word]
    
    def _bind(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        return v1 * v2
    
    def _bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros(self.dim, dtype=np.float32)
        bundled = np.sum(vectors, axis=0)
        return np.sign(bundled).astype(np.float32)
    
    def encode_sequence(self, words: List[str], position_encoding: bool = True) -> np.ndarray:
        if not words:
            return np.zeros(self.dim, dtype=np.float32)
        
        sequence_vectors = []
        for i, word in enumerate(words):
            word_vec = self._get_item_vector(word.lower())
            if position_encoding:
                pos_vec = self._get_item_vector(f"__POS_{min(i, 30)}__")
                bound = self._bind(word_vec, pos_vec)
            else:
                bound = word_vec
            sequence_vectors.append(bound)
        
        return self._bundle(sequence_vectors)
    
    def store_sequence(self, words: List[str], context: Optional[str] = None):
        seq_vec = self.encode_sequence(words)
        idx = len(self.sequence_memory)
        
        self.sequence_memory.append({
            'vector': seq_vec,
            'words': words,
            'words_lower': [w.lower() for w in words],
            'context': context,
            'len': len(words)
        })
        
        # Índice invertido de palabras
        seen_words = set()
        for word in words:
            norm = normalize_word(word)
            if norm and norm not in seen_words:
                self.word_to_sequences[norm].append(idx)
                seen_words.add(norm)
        
        # Índice de bigramas
        words_lower = [w.lower() for w in words]
        for i in range(len(words_lower) - 1):
            bigram = (words_lower[i], words_lower[i+1])
            self.bigram_index[bigram].append((idx, i))
    
    def find_by_keywords(self, keywords: List[str], top_k: int = 30) -> List[Tuple[float, int]]:
        if not keywords:
            return []
        
        seq_scores = Counter()
        norm_keywords = [normalize_word(k) for k in keywords if normalize_word(k)]
        
        for kw in norm_keywords:
            if kw in self.word_to_sequences:
                for idx in self.word_to_sequences[kw]:
                    seq_scores[idx] += 1
        
        ranked = seq_scores.most_common(top_k)
        return [(score, idx) for idx, score in ranked]
    
    def find_by_bigram(self, w1: str, w2: str) -> List[Tuple[int, int]]:
        """Busca secuencias que contengan el bigrama (w1, w2)."""
        key = (w1.lower(), w2.lower())
        return self.bigram_index.get(key, [])


# ═══════════════════════════════════════════════════════════════════
# SELECTOR FEP EXPANDIDO (5-gramas)
# ═══════════════════════════════════════════════════════════════════

class FreeEnergySelector:
    """
    Selector FEP con n-gramas de orden 1 a 5.
    """
    
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.quadgram_counts = defaultdict(Counter)
        self.pentagram_counts = defaultdict(Counter)
        self.total_words = 0
    
    def train(self, sentences: List[List[str]]):
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
                
                if i > 3:
                    ctx = (words[i-4], words[i-3], words[i-2], words[i-1])
                    self.pentagram_counts[ctx][word] += 1
    
    def compute_free_energy(self, prefix: List[str], candidate: str) -> float:
        candidate = candidate.lower()
        prefix_lower = [w.lower() for w in prefix]
        vocab_size = max(len(self.unigram_counts), 1)
        
        energies = []
        weights = []
        
        # Pentagram (peso 16)
        if len(prefix_lower) >= 4:
            ctx = tuple(prefix_lower[-4:])
            if ctx in self.pentagram_counts:
                total = sum(self.pentagram_counts[ctx].values())
                count = self.pentagram_counts[ctx].get(candidate, 0)
                prob = (count + 0.1) / (total + 0.1 * vocab_size)
                energies.append(-np.log(prob + 1e-10))
                weights.append(16.0)
        
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
        
        total_weight = sum(weights)
        avg_energy = sum(e * w for e, w in zip(energies, weights)) / total_weight
        
        return avg_energy
    
    def select_next(self, prefix: List[str], candidates: List[str]) -> Tuple[str, float]:
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
    
    def get_top_continuations(self, prefix: List[str], top_k: int = 30) -> List[str]:
        prefix_lower = [w.lower() for w in prefix]
        candidate_scores = Counter()
        
        # Pentagram
        if len(prefix_lower) >= 4:
            ctx = tuple(prefix_lower[-4:])
            if ctx in self.pentagram_counts:
                for word, count in self.pentagram_counts[ctx].most_common(top_k):
                    candidate_scores[word] += count * 16
        
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


# ═══════════════════════════════════════════════════════════════════
# GENERADOR DE TEXTO v3
# ═══════════════════════════════════════════════════════════════════

class TextGenerator:
    """
    Generador de texto biomimético v3 - Arquitectura Expandida.
    
    Nuevas características:
    - HDC dim=20,000 (expandido)
    - 5-gramas para coherencia extendida
    - Splicing multi-oración
    - Clasificador de intención integrado
    - Selección automática de identidad
    """
    
    def __init__(self, dim: int = 20000):
        self.dim = dim
        self.memory = HDCSequenceMemory(dim=dim)
        self.selector = FreeEnergySelector()
        self.vocabulary = set()
        self.is_trained = False
        self.identity = None
        self._auto_identity = True  # Auto-selección de identidad
    
    def train_from_corpus(self, corpus_path: str, max_sentences: int = 5000):
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
        print(f"  - Dimensión HDC: {self.dim}")
        print(f"  - Vocabulario: {len(self.vocabulary)} palabras")
        print(f"  - Secuencias en memoria: {len(self.memory.sequence_memory)}")
    
    def set_identity(self, identity):
        self.identity = identity
        self._auto_identity = False
    
    def set_auto_identity(self, enabled: bool = True):
        """Activa/desactiva la selección automática de identidad."""
        self._auto_identity = enabled
    
    def generate(self, prefix: str, max_words: int = 25, 
                temperature: float = None) -> str:
        """
        Genera texto con búsqueda multi-estrategia y splicing multi-oración.
        """
        if not self.is_trained:
            raise RuntimeError("El generador no está entrenado.")
        
        # Auto-seleccionar identidad si está activado
        if self._auto_identity and self.identity is None:
            intent = classify_intent(prefix)
            try:
                from .identity import load_identity
                self.identity = load_identity(intent['identity'])
            except:
                pass
        
        if temperature is None:
            temperature = self.identity.temperature if self.identity else 1.0
        
        words = prefix.strip().split()
        if not words:
            words = ["El"]
        
        prompt_keywords = [w for w in words if is_content_word(w)]
        
        # ═══ SPLICE MULTI-ORACIÓN ═══
        # Buscar múltiples oraciones relevantes y combinar fragmentos
        splice_result = self._multi_sentence_splice(words, prompt_keywords, max_words)
        if splice_result and len(splice_result.split()) > len(words) + 3:
            return splice_result
        
        # ═══ GENERACIÓN SECUENCIAL (fallback) ═══
        generated = list(words)
        
        for step in range(max_words):
            candidates = []
            
            # Estrategia 1: Coincidencia exacta (5,4,3,2,1 palabras)
            for ctx_len in [5, 4, 3, 2, 1]:
                exact = self._find_exact(generated, ctx_len)
                if exact:
                    candidates.extend(exact)
                    break
            
            # Estrategia 2: Keywords
            if len(candidates) < 5:
                kw = self._find_keyword_continuations(generated, prompt_keywords)
                if kw:
                    candidates.extend(kw)
            
            # Estrategia 3: N-gramas
            if len(candidates) < 5:
                ngram = self.selector.get_top_continuations(generated, top_k=30)
                if ngram:
                    candidates.extend(ngram)
            
            # Estrategia 4: Fallback
            if not candidates:
                top_words = self.selector.unigram_counts.most_common(50)
                candidates = [w for w, _ in top_words]
            
            # Filtros
            candidates = self._apply_filters(candidates, generated)
            
            # Identidad bias
            if self.identity and self.identity.vocabulary_bias:
                candidates = self._apply_vocabulary_bias(candidates)
            
            if not candidates:
                break
            
            # Selección
            if temperature <= 0.1:
                next_word, _ = self.selector.select_next(generated, candidates)
            else:
                next_word = self._sample_with_temperature(generated, candidates, temperature)
            
            if not next_word:
                break
            
            if next_word in ['.', '!', '?']:
                if step > 8:
                    generated.append(next_word)
                    break
                continue
            
            if next_word in [',', ';', ':'] and step > 3:
                generated.append(next_word)
                continue
            
            generated.append(next_word)
        
        result = ' '.join(generated)
        
        # Si solo repitió el prompt
        if result.strip().lower() == prefix.strip().lower():
            result = self._force_generation(prefix, prompt_keywords, temperature)
        
        return result
    
    def _multi_sentence_splice(self, prompt_words: List[str], keywords: List[str], 
                                max_words: int) -> Optional[str]:
        """
        Splicing multi-oración: busca varias oraciones relevantes y combina
        fragmentos para crear una respuesta coherente más larga.
        """
        if not keywords:
            return None
        
        matches = self.memory.find_by_keywords(keywords, top_k=10)
        if not matches:
            return None
        
        # Tomar las 3 mejores oraciones
        result_parts = list(prompt_words)
        used_words_lower = set(w.lower() for w in prompt_words)
        words_added = len(prompt_words)
        
        for score, idx in matches[:3]:
            if words_added >= max_words + len(prompt_words):
                break
            
            stored = self.memory.sequence_memory[idx]
            seq_words = stored['words']
            seq_lower = stored['words_lower']
            
            # Encontrar la mejor posición de splice
            best_start = None
            best_score = 0
            
            for kw in keywords:
                kw_lower = kw.lower()
                for i, w in enumerate(seq_lower):
                    if w == kw_lower:
                        kw_score = sum(1 for k in keywords if k.lower() in seq_lower)
                        if kw_score > best_score:
                            best_score = kw_score
                            best_start = max(0, i - 1)
            
            if best_start is not None:
                # Extraer fragmento desde la keyword
                fragment = seq_words[best_start:]
                
                for word in fragment:
                    w_lower = word.lower()
                    if w_lower not in used_words_lower or not is_content_word(word):
                        result_parts.append(word)
                        used_words_lower.add(w_lower)
                        words_added += 1
                        
                        if words_added >= max_words + len(prompt_words):
                            break
                
                # Agregar conector si hay más oraciones
                if words_added < max_words + len(prompt_words) and result_parts[-1] not in ['.', '!', '?']:
                    connectors = [',', 'y', 'además']
                    # Solo agregar conector si no es el último match
                    if matches.index((score, idx)) < 2:
                        pass  # No forzar conectores
        
        if len(result_parts) > len(prompt_words) + 2:
            return ' '.join(result_parts)
        
        return None
    
    def _find_exact(self, generated: List[str], context_len: int) -> List[str]:
        if len(generated) < context_len:
            return []
        
        context = [w.lower() for w in generated[-context_len:]]
        candidates = []
        
        for stored in self.memory.sequence_memory:
            seq_lower = stored['words_lower']
            for i in range(len(seq_lower) - context_len):
                if seq_lower[i:i+context_len] == context:
                    next_idx = i + context_len
                    if next_idx < len(stored['words']):
                        word = stored['words'][next_idx]
                        if not word.startswith('__'):
                            candidates.append(word)
        
        return candidates
    
    def _find_keyword_continuations(self, generated: List[str], 
                                     keywords: List[str]) -> List[str]:
        if not keywords:
            return []
        
        candidates = []
        matches = self.memory.find_by_keywords(keywords, top_k=30)
        last_lower = generated[-1].lower() if generated else ""
        
        for score, idx in matches:
            stored = self.memory.sequence_memory[idx]
            seq_lower = stored['words_lower']
            seq_words = stored['words']
            
            for i, word in enumerate(seq_lower):
                if word == last_lower and i + 1 < len(seq_words):
                    candidates.append(seq_words[i + 1])
            
            for kw in keywords:
                kw_lower = kw.lower()
                for i, word in enumerate(seq_lower):
                    if word == kw_lower:
                        for j in range(i + 1, min(i + 5, len(seq_words))):
                            candidates.append(seq_words[j])
        
        return candidates
    
    def _apply_filters(self, candidates: List[str], generated: List[str]) -> List[str]:
        if not candidates:
            return candidates
        
        gen_lower = [w.lower() for w in generated]
        filtered = []
        
        for c in candidates:
            c_lower = c.lower()
            
            if c_lower in gen_lower[-2:]:
                continue
            
            if len(gen_lower) >= 6:
                recent = gen_lower[-6:]
                if recent.count(c_lower) >= 2:
                    continue
            
            if c.startswith('__') or c.startswith('http'):
                continue
            
            filtered.append(c)
        
        return filtered if filtered else candidates
    
    def _force_generation(self, prefix: str, keywords: List[str], 
                          temperature: float) -> str:
        words = prefix.strip().split()
        
        if keywords:
            matches = self.memory.find_by_keywords(keywords, top_k=10)
            if matches:
                _, best_idx = matches[0]
                stored = self.memory.sequence_memory[best_idx]
                seq_words = stored['words']
                seq_lower = stored['words_lower']
                
                for kw in keywords:
                    kw_lower = kw.lower()
                    for i, w in enumerate(seq_lower):
                        if w == kw_lower:
                            start = max(0, i - 2)
                            end = min(len(seq_words), i + 25)
                            fragment = seq_words[start:end]
                            result = words + [w for w in fragment 
                                            if w.lower() not in set(x.lower() for x in words)]
                            return ' '.join(result[:len(words) + 25])
        
        # Fallback n-gramas
        generated = list(words)
        for _ in range(20):
            top = self.selector.get_top_continuations(generated, top_k=10)
            if not top:
                break
            top = [w for w in top if w.lower() not in set(x.lower() for x in generated[-3:])]
            if not top:
                break
            if temperature <= 0.5:
                next_word = top[0]
            else:
                next_word = np.random.choice(top[:min(5, len(top))])
            if next_word in ['.', '!', '?'] and len(generated) > len(words) + 8:
                generated.append(next_word)
                break
            generated.append(next_word)
        
        return ' '.join(generated)
    
    def _sample_with_temperature(self, prefix, candidates, temperature):
        if not candidates:
            return ""
        
        energies = np.array([self.selector.compute_free_energy(prefix, c) for c in candidates])
        scaled = -energies / max(temperature, 0.01)
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        prob_sum = probs.sum()
        
        if prob_sum == 0:
            return candidates[0]
        
        probs = probs / prob_sum
        idx = np.random.choice(len(candidates), p=probs)
        return candidates[idx]
    
    def _apply_vocabulary_bias(self, candidates):
        if not self.identity or not self.identity.vocabulary_bias:
            return candidates
        
        weighted = []
        bias_lower = {k.lower(): v for k, v in self.identity.vocabulary_bias.items()}
        
        for word in candidates:
            weighted.append(word)
            if word.lower() in bias_lower:
                weight = bias_lower[word.lower()]
                for _ in range(max(1, int(weight))):
                    weighted.append(word)
        
        return weighted
    
    def complete_sentence(self, prefix: str) -> str:
        generated = self.generate(prefix, max_words=30, temperature=0.6)
        if not generated.rstrip().endswith(('.', '!', '?')):
            generated += '.'
        return generated


def create_generator(corpus_path: str = None) -> TextGenerator:
    if corpus_path is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    generator = TextGenerator(dim=20000)
    generator.train_from_corpus(corpus_path, max_sentences=5000)
    return generator
