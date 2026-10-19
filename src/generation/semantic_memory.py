"""
Embeddings semánticos por co-ocurrencia (LSA simplificado)

Construye vectores de palabras basados en co-ocurrencia real en el corpus.
Palabras semánticamente similares tendrán vectores similares.

Esto permite:
- Búsqueda semántica (no solo por keyword exacta)
- Re-ranking de candidatos por similitud con el prompt
- Detección de temas por similitud entre oraciones
"""

import numpy as np
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math


class SemanticMemory:
    """
    Modelo de embeddings semánticos basado en co-ocurrencia.
    
    Construye una matriz de co-ocurrencia y la reduce con SVD truncado
    para obtener embeddings densos de palabras.
    """
    
    def __init__(self, dim: int = 150, window: int = 4, min_count: int = 2):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None  # numpy array (vocab_size, dim)
        self._cooccurrence = None
        self._trained = False
    
    def train(self, sentences: List[List[str]]):
        """Entrena embeddings desde oraciones del corpus."""
        # Paso 1: Contar frecuencias y filtrar vocabulario
        word_freq = Counter()
        for sent in sentences:
            words = [w.lower() for w in sent]
            word_freq.update(words)
        
        # Solo palabras con frecuencia mínima
        vocab = [w for w, c in word_freq.most_common() if c >= self.min_count]
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        
        vocab_size = len(vocab)
        if vocab_size < 10:
            return
        
        # Paso 2: Construir matriz de co-ocurrencia (sparse como dict)
        cooc = defaultdict(float)
        
        for sent in sentences:
            words = [w.lower() for w in sent]
            for i, word in enumerate(words):
                if word not in self.word_to_idx:
                    continue
                idx_i = self.word_to_idx[word]
                
                for j in range(max(0, i - self.window), min(len(words), i + self.window + 1)):
                    if i == j:
                        continue
                    other = words[j]
                    if other not in self.word_to_idx:
                        continue
                    idx_j = self.word_to_idx[other]
                    
                    # Peso inversamente proporcional a la distancia
                    weight = 1.0 / abs(i - j)
                    cooc[(idx_i, idx_j)] += weight
        
        # Paso 3: PPMI (Positive Pointwise Mutual Information)
        # PPMI(w,c) = max(0, log(P(w,c) / (P(w)*P(c))))
        total = sum(cooc.values())
        if total == 0:
            return
        
        # Frecuencias marginales
        row_sums = defaultdict(float)
        col_sums = defaultdict(float)
        for (i, j), v in cooc.items():
            row_sums[i] += v
            col_sums[j] += v
        
        # Construir matriz PPMI densa
        ppmi = np.zeros((vocab_size, vocab_size), dtype=np.float32)
        
        for (i, j), v in cooc.items():
            p_joint = v / total
            p_i = row_sums[i] / total
            p_j = col_sums[j] / total
            
            if p_i > 0 and p_j > 0:
                pmi = math.log(p_joint / (p_i * p_j) + 1e-10)
                ppmi[i, j] = max(0, pmi)  # PPMI: solo positivos
        
        # Paso 4: SVD truncado para reducir dimensionalidad
        try:
            U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
            k = min(self.dim, len(S))
            self.embeddings = U[:, :k] * np.sqrt(S[:k])  # Weighted
            self._trained = True
        except np.linalg.LinAlgError:
            # Fallback: usar PPMI directamente con PCA simplificado
            self._fallback_embeddings(ppmi)
    
    def _fallback_embeddings(self, matrix: np.ndarray):
        """Fallback si SVD falla: normalizar filas de PPMI."""
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        self.embeddings = matrix / norms
        if self.embeddings.shape[1] > self.dim:
            # Reducir con selección de columnas por varianza
            variances = np.var(self.embeddings, axis=0)
            top_cols = np.argsort(variances)[-self.dim:]
            self.embeddings = self.embeddings[:, sorted(top_cols)]
        self._trained = True
    
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Obtiene el embedding de una palabra."""
        if not self._trained:
            return None
        word = word.lower()
        idx = self.word_to_idx.get(word)
        if idx is not None and idx < len(self.embeddings):
            return self.embeddings[idx]
        return None
    
    def similarity(self, word1: str, word2: str) -> float:
        """Similitud coseno entre dos palabras."""
        e1 = self.get_embedding(word1)
        e2 = self.get_embedding(word2)
        if e1 is None or e2 is None:
            return 0.0
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
    
    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Encuentra las palabras más similares a una dada."""
        emb = self.get_embedding(word)
        if emb is None:
            return []
        
        # Similitud coseno con todas las palabras
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        sims = self.embeddings @ emb / (norms * np.linalg.norm(emb) + 1e-8)
        
        # Top-k (excluyendo la palabra misma)
        word_idx = self.word_to_idx.get(word.lower(), -1)
        sims[word_idx] = -1  # Excluir
        
        top_indices = np.argsort(sims)[-top_k:][::-1]
        return [(self.idx_to_word[i], float(sims[i])) for i in top_indices]
    
    def sentence_embedding(self, words: List[str]) -> Optional[np.ndarray]:
        """Embedding de oración como promedio de embeddings de palabras."""
        if not self._trained:
            return None
        
        embeddings = []
        for w in words:
            emb = self.get_embedding(w)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return None
        
        return np.mean(embeddings, axis=0)
    
    def sentence_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Similitud coseno entre dos oraciones."""
        e1 = self.sentence_embedding(words1)
        e2 = self.sentence_embedding(words2)
        if e1 is None or e2 is None:
            return 0.0
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8))
    
    @property
    def vocab_size(self) -> int:
        return len(self.word_to_idx)
    
    @property
    def is_trained(self) -> bool:
        return self._trained
