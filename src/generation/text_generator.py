"""
Generador de Texto Biomimético v4 - Deep Cognition

Arquitectura completa:
- Embeddings semánticos (co-ocurrencia + SVD) para comprensión real
- Tokenizador robusto (maneja puntuación, contracciones)
- Memoria conversacional (contexto de turns anteriores)
- Modelo de temas (clustering de oraciones)
- Scoring de coherencia global (evalúa respuesta completa)
- Re-ranking semántico de candidatos
- HDC expandido (20K dim) + 5-gramas + splicing multi-oración
- Clasificador de intención mejorado
- Selección automática de identidad
"""

import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from conllu import parse_incr
from typing import List, Tuple, Optional, Dict, Set
import json
import re

from .tokenizer import Tokenizer
from .semantic_memory import SemanticMemory
from .coherence import CoherenceScorer
from .conversation import ConversationMemory
from .topics import TopicModel


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

TOPIC_KEYWORDS = {
    'poeta': {
        'amor', 'corazón', 'alma', 'sueño', 'luna', 'noche', 'muerte',
        'dolor', 'esperanza', 'tristeza', 'pasión', 'sentimiento', 'llanto',
        'verso', 'poema', 'poesía', 'belleza', 'flor', 'mar', 'cielo',
        'estrella', 'silencio', 'soledad', 'beso', 'suspiro', 'eterno',
        'agua', 'río', 'viento', 'fuego', 'tierra', 'sol', 'sombrío'
    },
    'filosofo': {
        'dios', 'existencia', 'verdad', 'razón', 'libertad', 'conciencia',
        'ser', 'pensar', 'pensamiento', 'duda', 'fe', 'ética', 'moral',
        'conocimiento', 'sabiduría', 'vida', 'muerte', 'sentido', 'alma',
        'espíritu', 'realidad', 'tiempo', 'infinito', 'nada', 'absoluto',
        'filosofía', 'platón', 'aristóteles', 'kant', 'hegel', 'sartre',
        'descartes', 'sócrates', 'nietzsche', 'metafísica', 'ontología',
        'cosmos', 'universo', 'naturaleza', 'humano', 'esencia', 'causa'
    },
    'narrador': {
        'había', 'cuenta', 'historia', 'personaje', 'pueblo', 'familia',
        'recuerdo', 'memoria', 'pasado', 'infancia', 'abuelo', 'abuela',
        'casa', 'calle', 'viaje', 'aventura', 'secreto', 'destino',
        'guerra', 'paz', 'tierra', 'lluvia', 'amanecer', 'atardecer',
        'garcía', 'borges', 'cortázar', 'allende', 'señor', 'señora',
        'don', 'doña', 'puerta', 'camino', 'tarde', 'mañana'
    },
    'ensayista': {
        'cultura', 'sociedad', 'literatura', 'lenguaje', 'tradición',
        'modernidad', 'identidad', 'historia', 'civilización', 'política',
        'economía', 'educación', 'ciencia', 'tecnología', 'arte',
        'análisis', 'tesis', 'argumento', 'evidencia', 'conclusión',
        'perspectiva', 'paradigma', 'discurso', 'contexto', 'fenómeno',
        'desarrollo', 'proceso', 'estructura', 'sistema', 'problema'
    },
    'contemporaneo': {
        'google', 'internet', 'app', 'red', 'digital', 'tecnología',
        'datos', 'software', 'programa', 'código', 'web', 'pantalla',
        'usuario', 'plataforma', 'innovación', 'global', 'startup',
        'actual', 'hoy', 'ahora', 'presente', 'futuro', 'cambio',
        'mercado', 'empresa', 'negocio', 'producto', 'servicio'
    }
}

QUESTION_MARKERS = ['qué', 'quién', 'cómo', 'cuándo', 'dónde', 'por qué',
                     'cuál', 'cuánto', 'cuántos', 'cuántas', 'acaso']


def classify_intent(text: str) -> Dict:
    """Clasifica intención y tema del prompt con contexto conversacional."""
    tokens = Tokenizer.tokenize(text)
    norm_words = [Tokenizer.normalize(t) for t in tokens]
    content_words = [w for w in norm_words if w and w not in STOP_WORDS and len(w) > 2]
    
    # ¿Es pregunta?
    is_question = (
        any(m in norm_words for m in QUESTION_MARKERS) or
        text.strip().startswith('¿') or
        text.strip().endswith('?')
    )
    
    # Score por identidad
    scores = {}
    for identity, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for w in content_words if w in keywords)
        scores[identity] = score
    
    best_identity = max(scores, key=scores.get) if scores else 'contemporaneo'
    confidence = scores.get(best_identity, 0)
    
    if confidence == 0:
        # Heurística: preguntas → filósofo, afirmaciones cortas → contemporáneo
        if is_question and len(content_words) >= 2:
            best_identity = 'filosofo'
            confidence = 1
        else:
            best_identity = 'contemporaneo'
    
    return {
        'identity': best_identity,
        'is_question': is_question,
        'keywords': content_words[:5],
        'confidence': confidence
    }


def normalize_word(word: str) -> str:
    return Tokenizer.normalize(word)


def is_content_word(word: str) -> bool:
    norm = Tokenizer.normalize(word)
    return len(norm) > 2 and norm not in STOP_WORDS


# ═══════════════════════════════════════════════════════════════════
# MEMORIA HDC EXPANDIDA
# ═══════════════════════════════════════════════════════════════════

class HDCSequenceMemory:
    """Memoria asociativa hiperdimensional optimizada en memoria."""
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.item_memory = {}
        self.sequence_memory = []
        self.word_to_sequences = defaultdict(list)
        self.bigram_index = defaultdict(list)
        self.rng = np.random.RandomState(42)
    
    def _get_item_vector(self, word: str) -> np.ndarray:
        if word not in self.item_memory:
            # Usar int8 para reducir memoria (solo {-1, +1})
            vec = self.rng.choice([-1, 1], size=self.dim).astype(np.int8)
            self.item_memory[word] = vec
        return self.item_memory[word]
    
    def _bind(self, v1, v2):
        return v1 * v2
    
    def _bundle(self, vectors):
        if not vectors:
            return np.zeros(self.dim, dtype=np.int8)
        return np.sign(np.sum(vectors, axis=0)).astype(np.int8)
    
    def encode_sequence(self, words: List[str], position_encoding=True) -> np.ndarray:
        if not words:
            return np.zeros(self.dim, dtype=np.int8)
        vecs = []
        for i, word in enumerate(words):
            wv = self._get_item_vector(word.lower())
            if position_encoding:
                pv = self._get_item_vector(f"__POS_{min(i, 30)}__")
                vecs.append(self._bind(wv, pv))
            else:
                vecs.append(wv)
        return self._bundle(vecs)
    
    def store_sequence(self, words: List[str], context: Optional[str] = None):
        # NO almacenar vector HDC (ahorra ~10KB por oración)
        # Solo se genera bajo demanda en find_similar
        idx = len(self.sequence_memory)
        self.sequence_memory.append({
            'words': words,
            'words_lower': [w.lower() for w in words],
            'context': context,
            'len': len(words)
        })
        
        seen = set()
        for word in words:
            norm = normalize_word(word)
            if norm and norm not in seen:
                self.word_to_sequences[norm].append(idx)
                seen.add(norm)
        
        wl = [w.lower() for w in words]
        for i in range(len(wl) - 1):
            self.bigram_index[(wl[i], wl[i+1])].append((idx, i))
    
    def find_by_keywords(self, keywords, top_k=30):
        if not keywords:
            return []
        seq_scores = Counter()
        for kw in keywords:
            norm = normalize_word(kw)
            if norm in self.word_to_sequences:
                for idx in self.word_to_sequences[norm]:
                    seq_scores[idx] += 1
        return [(s, i) for i, s in seq_scores.most_common(top_k)]
    
    def find_by_bigram(self, w1, w2):
        return self.bigram_index.get((w1.lower(), w2.lower()), [])


# ═══════════════════════════════════════════════════════════════════
# SELECTOR FEP (5-gramas)
# ═══════════════════════════════════════════════════════════════════

class FreeEnergySelector:
    """Selector FEP con n-gramas de orden 1-5."""
    
    def __init__(self):
        self.unigram_counts = Counter()
        self.bigram_counts = defaultdict(Counter)
        self.trigram_counts = defaultdict(Counter)
        self.quadgram_counts = defaultdict(Counter)
        self.pentagram_counts = defaultdict(Counter)
        self.total_words = 0
    
    def train(self, sentences):
        for sent in sentences:
            words = [w.lower() for w in sent]
            for i, word in enumerate(words):
                self.unigram_counts[word] += 1
                self.total_words += 1
                if i > 0:
                    self.bigram_counts[words[i-1]][word] += 1
                if i > 1:
                    self.trigram_counts[(words[i-2], words[i-1])][word] += 1
                if i > 2:
                    self.quadgram_counts[(words[i-3], words[i-2], words[i-1])][word] += 1
                if i > 3:
                    self.pentagram_counts[(words[i-4], words[i-3], words[i-2], words[i-1])][word] += 1
    
    def compute_free_energy(self, prefix, candidate):
        candidate = candidate.lower()
        pl = [w.lower() for w in prefix]
        vs = max(len(self.unigram_counts), 1)
        energies, weights = [], []
        
        if len(pl) >= 4:
            ctx = tuple(pl[-4:])
            if ctx in self.pentagram_counts:
                t = sum(self.pentagram_counts[ctx].values())
                c = self.pentagram_counts[ctx].get(candidate, 0)
                energies.append(-np.log((c + 0.1) / (t + 0.1 * vs) + 1e-10))
                weights.append(16.0)
        if len(pl) >= 3:
            ctx = (pl[-3], pl[-2], pl[-1])
            if ctx in self.quadgram_counts:
                t = sum(self.quadgram_counts[ctx].values())
                c = self.quadgram_counts[ctx].get(candidate, 0)
                energies.append(-np.log((c + 0.1) / (t + 0.1 * vs) + 1e-10))
                weights.append(8.0)
        if len(pl) >= 2:
            ctx = (pl[-2], pl[-1])
            if ctx in self.trigram_counts:
                t = sum(self.trigram_counts[ctx].values())
                c = self.trigram_counts[ctx].get(candidate, 0)
                energies.append(-np.log((c + 0.1) / (t + 0.1 * vs) + 1e-10))
                weights.append(4.0)
        if len(pl) >= 1:
            prev = pl[-1]
            if prev in self.bigram_counts:
                t = sum(self.bigram_counts[prev].values())
                c = self.bigram_counts[prev].get(candidate, 0)
                energies.append(-np.log((c + 0.1) / (t + 0.1 * vs) + 1e-10))
                weights.append(2.0)
        
        t = max(self.total_words, 1)
        c = self.unigram_counts.get(candidate, 0)
        energies.append(-np.log((c + 0.1) / (t + 0.1 * vs) + 1e-10))
        weights.append(1.0)
        
        tw = sum(weights)
        return sum(e * w for e, w in zip(energies, weights)) / tw
    
    def select_next(self, prefix, candidates):
        if not candidates:
            return "", float('inf')
        best_w, best_e = candidates[0], float('inf')
        for c in candidates:
            e = self.compute_free_energy(prefix, c)
            if e < best_e:
                best_e, best_w = e, c
        return best_w, best_e
    
    def get_top_continuations(self, prefix, top_k=30):
        pl = [w.lower() for w in prefix]
        scores = Counter()
        
        if len(pl) >= 4:
            ctx = tuple(pl[-4:])
            if ctx in self.pentagram_counts:
                for w, c in self.pentagram_counts[ctx].most_common(top_k):
                    scores[w] += c * 16
        if len(pl) >= 3:
            ctx = (pl[-3], pl[-2], pl[-1])
            if ctx in self.quadgram_counts:
                for w, c in self.quadgram_counts[ctx].most_common(top_k):
                    scores[w] += c * 8
        if len(pl) >= 2:
            ctx = (pl[-2], pl[-1])
            if ctx in self.trigram_counts:
                for w, c in self.trigram_counts[ctx].most_common(top_k):
                    scores[w] += c * 4
        if len(pl) >= 1:
            if pl[-1] in self.bigram_counts:
                for w, c in self.bigram_counts[pl[-1]].most_common(top_k):
                    scores[w] += c * 2
        
        return [w for w, _ in scores.most_common(top_k)]


# ═══════════════════════════════════════════════════════════════════
# GENERADOR v4 - DEEP COGNITION
# ═══════════════════════════════════════════════════════════════════

class TextGenerator:
    """
    Generador de texto biomimético v4 - Deep Cognition.
    
    Componentes:
    - HDCSequenceMemory: memoria asociativa hiperdimensional
    - FreeEnergySelector: selección por n-gramas 1-5
    - SemanticMemory: embeddings por co-ocurrencia (LSA)
    - TopicModel: clustering de oraciones por tema
    - CoherenceScorer: evaluación de calidad global
    - ConversationMemory: contexto conversacional
    - Tokenizer: tokenización robusta
    """
    
    def __init__(self, dim: int = 10000):
        self.dim = dim
        self.memory = HDCSequenceMemory(dim=dim)
        self.selector = FreeEnergySelector()
        self.semantic = SemanticMemory(dim=150, window=4, min_count=2)
        self.scorer = CoherenceScorer(self.semantic)
        self.conversation = ConversationMemory(max_turns=10)
        self.topics = TopicModel(n_topics=20)
        self.vocabulary = set()
        self.is_trained = False
        self.identity = None
        self._auto_identity = True
        self._all_sentences = []  # Guardar para topic model
    
    def train_from_corpus(self, corpus_path: str, max_sentences: int = 5000):
        """Entrena todos los componentes desde un corpus CoNLL-U."""
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
        
        self._train_all(sentences)
    
    def train_from_sentences(self, sentences: List[List[str]]):
        """Entrena desde una lista de oraciones (lista de listas de palabras)."""
        for sent in sentences:
            self.vocabulary.update(w.lower() for w in sent)
        self._train_all(sentences)
    
    def _train_all(self, sentences: List[List[str]]):
        """Entrena todos los componentes."""
        self._all_sentences = sentences
        n = len(sentences)
        
        # 1. Memoria HDC
        print(f"  Almacenando {n} secuencias en memoria HDC...")
        for sent in sentences:
            self.memory.store_sequence(sent)
        
        # 2. Selector FEP (n-gramas)
        print(f"  Entrenando selector FEP (n-gramas 1-5)...")
        self.selector.train(sentences)
        
        # 3. Embeddings semánticos
        print(f"  Construyendo embeddings semánticos (SVD)...")
        self.semantic.train(sentences)
        
        # 4. Modelo de temas
        if self.semantic.is_trained:
            print(f"  Construyendo modelo de temas...")
            self.topics.train(sentences, self.semantic)
        
        # Actualizar scorer con semántica
        self.scorer = CoherenceScorer(self.semantic)
        
        self.is_trained = True
        print(f"✓ Entrenamiento completo")
        print(f"  - Dimensión HDC: {self.dim}")
        print(f"  - Vocabulario: {len(self.vocabulary)} palabras")
        print(f"  - Embeddings: {self.semantic.vocab_size} palabras × {self.semantic.dim} dim")
        print(f"  - Secuencias: {len(self.memory.sequence_memory)}")
    
    def set_identity(self, identity):
        self.identity = identity
        self._auto_identity = False
    
    def set_auto_identity(self, enabled: bool = True):
        self._auto_identity = enabled
    
    def generate(self, prefix: str, max_words: int = 25,
                temperature: float = None) -> str:
        """
        Genera texto con todas las estrategias integradas.
        """
        if not self.is_trained:
            raise RuntimeError("El generador no está entrenado.")
        
        # Tokenizar prompt con tokenizador robusto
        prompt_tokens = Tokenizer.tokenize(prefix)
        prompt_words = [t for t in prompt_tokens if t.strip()]
        
        if not prompt_words:
            prompt_words = ["El"]
        
        # Auto-identidad
        if self._auto_identity and self.identity is None:
            intent = classify_intent(prefix)
            try:
                from .identity import load_identity
                self.identity = load_identity(intent['identity'])
            except:
                pass
        
        if temperature is None:
            temperature = self.identity.temperature if self.identity else 1.0
        
        # Enriquecer con contexto conversacional
        prompt_keywords = [w for w in prompt_words if is_content_word(w)]
        enriched_keywords = self.conversation.enrich_prompt(prompt_keywords)
        
        # ═══ GENERAR MÚLTIPLES CANDIDATOS ═══
        candidates_text = []
        
        # Candidato 1: Splicing multi-oración
        splice = self._multi_sentence_splice(prompt_words, enriched_keywords, max_words)
        if splice:
            candidates_text.append(splice)
        
        # Candidato 2: Generación secuencial con re-ranking semántico
        seq_text = self._sequential_generate(prompt_words, enriched_keywords,
                                              max_words, temperature)
        if seq_text and seq_text.strip().lower() != prefix.strip().lower():
            candidates_text.append(seq_text)
        
        # Candidato 3: Búsqueda temática si hay topic model
        if self.topics.is_trained:
            topic_text = self._topic_generate(prompt_words, enriched_keywords, max_words)
            if topic_text:
                candidates_text.append(topic_text)
        
        # ═══ SELECCIONAR MEJOR CANDIDATO ═══
        if not candidates_text:
            # Fallback: forzar generación
            result = self._force_generation(prefix, enriched_keywords, temperature)
        elif len(candidates_text) == 1:
            result = candidates_text[0]
        else:
            # Scoring de coherencia para elegir el mejor
            best_score = -1
            result = candidates_text[0]
            identity_bias = self.identity.vocabulary_bias if self.identity else None
            
            for text in candidates_text:
                words = text.split()
                score = self.scorer.score(words, prompt_words, identity_bias)
                if score > best_score:
                    best_score = score
                    result = text
        
        # ═══ POST-PROCESAMIENTO ═══
        result = self._post_process(result, prefix)
        
        # Guardar en memoria conversacional
        self.conversation.add_turn(prefix, result)
        
        return result
    
    def _sequential_generate(self, prompt_words, keywords, max_words, temperature):
        """Generación secuencial con re-ranking semántico."""
        generated = list(prompt_words)
        
        for step in range(max_words):
            candidates = []
            
            # Estrategia 1: Coincidencia exacta
            for ctx_len in [5, 4, 3, 2, 1]:
                exact = self._find_exact(generated, ctx_len)
                if exact:
                    candidates.extend(exact)
                    break
            
            # Estrategia 2: Keywords (incluyendo contexto conversacional)
            if len(candidates) < 5:
                kw = self._find_keyword_continuations(generated, keywords)
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
            
            # ═══ RE-RANKING SEMÁNTICO ═══
            if self.semantic.is_trained and len(candidates) > 5:
                candidates = self.scorer.rerank(
                    candidates, prompt_words, generated, top_k=15
                )
            
            # Identidad bias
            if self.identity and self.identity.vocabulary_bias:
                candidates = self._apply_vocabulary_bias(candidates)
            
            if not candidates:
                break
            
            # Selección con temperatura
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
        
        return ' '.join(generated) if len(generated) > len(prompt_words) else None
    
    def _topic_generate(self, prompt_words, keywords, max_words):
        """Genera texto buscando por tema semántico."""
        if not self.topics.is_trained or not self.semantic.is_trained:
            return None
        
        relevant = self.topics.find_relevant_sentences(
            keywords, self.semantic, self._all_sentences, top_k=5
        )
        
        if not relevant:
            return None
        
        # Combinar fragmentos de las oraciones más relevantes
        result_parts = list(prompt_words)
        used_lower = set(w.lower() for w in prompt_words)
        words_added = len(prompt_words)
        
        for sim, idx in relevant[:3]:
            if sim < 0.1:  # Muy baja similitud
                break
            if idx >= len(self._all_sentences):
                continue
            
            sent = self._all_sentences[idx]
            for word in sent:
                wl = word.lower()
                if wl not in used_lower or not is_content_word(word):
                    result_parts.append(word)
                    used_lower.add(wl)
                    words_added += 1
                    if words_added >= max_words + len(prompt_words):
                        break
            
            if words_added >= max_words + len(prompt_words):
                break
        
        if len(result_parts) > len(prompt_words) + 2:
            return ' '.join(result_parts)
        return None
    
    def _multi_sentence_splice(self, prompt_words, keywords, max_words):
        """Splicing multi-oración (legacy, mejorado)."""
        if not keywords:
            return None
        
        matches = self.memory.find_by_keywords(keywords, top_k=10)
        if not matches:
            return None
        
        result_parts = list(prompt_words)
        used_lower = set(w.lower() for w in prompt_words)
        words_added = len(prompt_words)
        
        for score, idx in matches[:3]:
            if words_added >= max_words + len(prompt_words):
                break
            
            stored = self.memory.sequence_memory[idx]
            seq_words = stored['words']
            seq_lower = stored['words_lower']
            
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
                for word in seq_words[best_start:]:
                    wl = word.lower()
                    if wl not in used_lower or not is_content_word(word):
                        result_parts.append(word)
                        used_lower.add(wl)
                        words_added += 1
                        if words_added >= max_words + len(prompt_words):
                            break
        
        if len(result_parts) > len(prompt_words) + 2:
            return ' '.join(result_parts)
        return None
    
    def _find_exact(self, generated, context_len):
        if len(generated) < context_len:
            return []
        context = [w.lower() for w in generated[-context_len:]]
        candidates = []
        for stored in self.memory.sequence_memory:
            sl = stored['words_lower']
            for i in range(len(sl) - context_len):
                if sl[i:i+context_len] == context:
                    ni = i + context_len
                    if ni < len(stored['words']):
                        w = stored['words'][ni]
                        if not w.startswith('__'):
                            candidates.append(w)
        return candidates
    
    def _find_keyword_continuations(self, generated, keywords):
        if not keywords:
            return []
        candidates = []
        matches = self.memory.find_by_keywords(keywords, top_k=30)
        last_lower = generated[-1].lower() if generated else ""
        
        for score, idx in matches:
            stored = self.memory.sequence_memory[idx]
            sl = stored['words_lower']
            sw = stored['words']
            for i, w in enumerate(sl):
                if w == last_lower and i + 1 < len(sw):
                    candidates.append(sw[i + 1])
            for kw in keywords:
                kl = kw.lower()
                for i, w in enumerate(sl):
                    if w == kl:
                        for j in range(i + 1, min(i + 5, len(sw))):
                            candidates.append(sw[j])
        return candidates
    
    def _apply_filters(self, candidates, generated):
        if not candidates:
            return candidates
        gl = [w.lower() for w in generated]
        filtered = []
        for c in candidates:
            cl = c.lower()
            if cl in gl[-2:]:
                continue
            if len(gl) >= 6 and gl[-6:].count(cl) >= 2:
                continue
            if c.startswith('__') or c.startswith('http'):
                continue
            filtered.append(c)
        return filtered if filtered else candidates
    
    def _force_generation(self, prefix, keywords, temperature):
        words = prefix.strip().split()
        if keywords:
            matches = self.memory.find_by_keywords(keywords, top_k=10)
            if matches:
                _, best_idx = matches[0]
                stored = self.memory.sequence_memory[best_idx]
                sw = stored['words']
                sl = stored['words_lower']
                for kw in keywords:
                    kl = kw.lower()
                    for i, w in enumerate(sl):
                        if w == kl:
                            start = max(0, i - 2)
                            end = min(len(sw), i + 25)
                            fragment = sw[start:end]
                            result = words + [w for w in fragment if w.lower() not in set(x.lower() for x in words)]
                            return ' '.join(result[:len(words) + 25])
        
        generated = list(words)
        for _ in range(20):
            top = self.selector.get_top_continuations(generated, top_k=10)
            if not top:
                break
            top = [w for w in top if w.lower() not in set(x.lower() for x in generated[-3:])]
            if not top:
                break
            next_word = np.random.choice(top[:min(5, len(top))]) if temperature > 0.5 else top[0]
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
        s = probs.sum()
        if s == 0:
            return candidates[0]
        probs /= s
        return candidates[np.random.choice(len(candidates), p=probs)]
    
    def _apply_vocabulary_bias(self, candidates):
        if not self.identity or not self.identity.vocabulary_bias:
            return candidates
        weighted = []
        bias_lower = {k.lower(): v for k, v in self.identity.vocabulary_bias.items()}
        for word in candidates:
            weighted.append(word)
            if word.lower() in bias_lower:
                for _ in range(max(1, int(bias_lower[word.lower()]))):
                    weighted.append(word)
        return weighted
    
    def _post_process(self, text, original_prompt):
        """Post-procesamiento: limpieza y formato."""
        if not text or text.strip() == original_prompt.strip():
            return text
        
        # Eliminar espacios dobles
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Asegurar que empieza con mayúscula
        if text and text[0].islower() and text[0] not in '¿¡':
            text = text[0].upper() + text[1:]
        
        return text
    
    def complete_sentence(self, prefix: str) -> str:
        generated = self.generate(prefix, max_words=30, temperature=0.6)
        if not generated.rstrip().endswith(('.', '!', '?')):
            generated += '.'
        return generated


def create_generator(corpus_path=None):
    if corpus_path is None:
        corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    gen = TextGenerator(dim=10000)
    gen.train_from_corpus(corpus_path, max_sentences=5000)
    return gen
