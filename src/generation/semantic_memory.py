"""
Embeddings semánticos por co-ocurrencia (LSA simplificado) — OPTIMIZADO

Usa scipy.sparse para la matriz de co-ocurrencia y SVD truncado rápido.
Construye vectores de palabras basados en co-ocurrencia real en el corpus.

Optimizaciones v6.1:
- Matriz sparse (scipy.sparse) para co-ocurrencia
- SVD truncado con scipy.sparse.linalg.svds (mucho más rápido)
- Vocabulario limitado a top-N palabras más frecuentes
- Construcción incremental de co-ocurrencia
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math


class SemanticMemory:
    """
    Modelo de embeddings semánticos basado en co-ocurrencia.
    Optimizado para corpus grandes con matrices sparse.
    """
    
    def __init__(self, dim: int = 150, window: int = 4, min_count: int = 3,
                 max_vocab: int = 15000):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.max_vocab = max_vocab
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None  # numpy array (vocab_size, dim)
        self._trained = False
    
    def train(self, sentences: List[List[str]]):
        """Entrena embeddings desde oraciones del corpus."""
        # Paso 1: Contar frecuencias y limitar vocabulario
        word_freq = Counter()
        for sent in sentences:
            words = [w.lower() for w in sent]
            word_freq.update(words)
        
        # Top N palabras más frecuentes con frecuencia mínima
        vocab = [w for w, c in word_freq.most_common(self.max_vocab) 
                 if c >= self.min_count]
        self.word_to_idx = {w: i for i, w in enumerate(vocab)}
        self.idx_to_word = {i: w for w, i in self.word_to_idx.items()}
        
        vocab_size = len(vocab)
        if vocab_size < 10:
            return
        
        # Paso 2: Construir matriz de co-ocurrencia SPARSE
        cooc_data = defaultdict(float)
        
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
                    
                    weight = 1.0 / abs(i - j)
                    cooc_data[(idx_i, idx_j)] += weight
        
        if not cooc_data:
            return
        
        # Paso 3: Construir matriz sparse y calcular PPMI
        rows, cols, vals = [], [], []
        total = sum(cooc_data.values())
        
        # Frecuencias marginales
        row_sums = defaultdict(float)
        col_sums = defaultdict(float)
        for (i, j), v in cooc_data.items():
            row_sums[i] += v
            col_sums[j] += v
        
        # PPMI
        for (i, j), v in cooc_data.items():
            p_joint = v / total
            p_i = row_sums[i] / total
            p_j = col_sums[j] / total
            
            if p_i > 0 and p_j > 0:
                pmi = math.log(p_joint / (p_i * p_j) + 1e-10)
                if pmi > 0:  # Solo PPMI positivos
                    rows.append(i)
                    cols.append(j)
                    vals.append(pmi)
        
        if not vals:
            return
        
        # Crear matriz sparse CSR
        ppmi_sparse = sparse.csr_matrix(
            (vals, (rows, cols)),
            shape=(vocab_size, vocab_size),
            dtype=np.float32
        )
        
        # Paso 4: SVD truncado con scipy (mucho más rápido)
        k = min(self.dim, vocab_size - 2, min(ppmi_sparse.shape) - 2)
        if k < 5:
            return
        
        try:
            U, S, Vt = svds(ppmi_sparse.astype(np.float64), k=k)
            # Ordenar por valores singulares descendentes
            idx = np.argsort(S)[::-1]
            U = U[:, idx]
            S = S[idx]
            
            # Weighted embeddings
            self.embeddings = (U * np.sqrt(S)).astype(np.float32)
            self._trained = True
        except Exception:
            # Fallback: usar random projections si SVD falla
            rng = np.random.RandomState(42)
            self.embeddings = rng.randn(vocab_size, self.dim).astype(np.float32)
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
            self.embeddings /= norms
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
        
        norms = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        sims = self.embeddings @ emb / (norms * np.linalg.norm(emb) + 1e-8)
        
        word_idx = self.word_to_idx.get(word.lower(), -1)
        sims[word_idx] = -1
        
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
