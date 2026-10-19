"""
Modelo de temas simplificado (LSA/clustering)

Agrupa oraciones del corpus por tema usando similitud semántica.
Permite búsqueda temática: "habla de filosofía" → encuentra oraciones filosóficas.
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class TopicModel:
    """Modelo de temas basado en clustering de embeddings."""
    
    def __init__(self, n_topics: int = 20):
        self.n_topics = n_topics
        self.topic_words = {}  # topic_id -> list of top words
        self.topic_sentences = defaultdict(list)  # topic_id -> [sentence_indices]
        self.sentence_topics = {}  # sentence_index -> topic_id
        self.topic_centroids = {}  # topic_id -> centroid vector
        self._trained = False
    
    def train(self, sentences: List[List[str]], semantic_memory):
        """
        Agrupa oraciones en temas usando embeddings.
        
        Usa k-means simplificado sobre los embeddings de oraciones.
        """
        if not semantic_memory or not semantic_memory.is_trained:
            return
        
        # Obtener embeddings de todas las oraciones
        sentence_embeddings = []
        valid_indices = []
        
        for i, sent in enumerate(sentences):
            emb = semantic_memory.sentence_embedding(sent)
            if emb is not None:
                sentence_embeddings.append(emb)
                valid_indices.append(i)
        
        if len(sentence_embeddings) < self.n_topics * 2:
            # Muy pocas oraciones para clustering, usar temas por keywords
            self._fallback_topics(sentences)
            return
        
        embeddings = np.array(sentence_embeddings)
        
        # K-means simplificado
        n = len(embeddings)
        k = min(self.n_topics, n // 3)
        
        # Inicialización: seleccionar k puntos aleatorios como centroides
        rng = np.random.RandomState(42)
        centroid_indices = rng.choice(n, k, replace=False)
        centroids = embeddings[centroid_indices].copy()
        
        # Iterar k-means (10 iteraciones)
        assignments = np.zeros(n, dtype=int)
        
        for _ in range(10):
            # Asignar cada punto al centroide más cercano
            for i in range(n):
                dists = np.linalg.norm(centroids - embeddings[i], axis=1)
                assignments[i] = np.argmin(dists)
            
            # Recalcular centroides
            for j in range(k):
                mask = assignments == j
                if mask.any():
                    centroids[j] = embeddings[mask].mean(axis=0)
        
        # Guardar resultados
        for i, sent_idx in enumerate(valid_indices):
            topic = int(assignments[i])
            self.sentence_topics[sent_idx] = topic
            self.topic_sentences[topic].append(sent_idx)
        
        # Extraer palabras representativas de cada tema
        for topic_id in range(k):
            topic_sents_indices = self.topic_sentences.get(topic_id, [])
            word_freq = Counter()
            
            for idx in topic_sents_indices:
                if idx < len(sentences):
                    for word in sentences[idx]:
                        norm = word.lower()
                        if len(norm) > 3:
                            word_freq[norm] += 1
            
            self.topic_words[topic_id] = [w for w, _ in word_freq.most_common(15)]
            self.topic_centroids[topic_id] = centroids[topic_id]
        
        self._trained = True
    
    def _fallback_topics(self, sentences: List[List[str]]):
        """Agrupación simple por longitud y primeras palabras."""
        chunk_size = max(1, len(sentences) // self.n_topics)
        
        for i, sent in enumerate(sentences):
            topic = min(i // chunk_size, self.n_topics - 1)
            self.sentence_topics[i] = topic
            self.topic_sentences[topic].append(i)
        
        for topic_id in range(self.n_topics):
            word_freq = Counter()
            for idx in self.topic_sentences.get(topic_id, []):
                if idx < len(sentences):
                    for word in sentences[idx]:
                        if len(word.lower()) > 3:
                            word_freq[word.lower()] += 1
            self.topic_words[topic_id] = [w for w, _ in word_freq.most_common(10)]
        
        self._trained = True
    
    def find_relevant_sentences(self, query_words: List[str], 
                                 semantic_memory, sentences: List[List[str]],
                                 top_k: int = 10) -> List[Tuple[float, int]]:
        """
        Encuentra oraciones relevantes por similitud temática y semántica.
        """
        if not self._trained or not semantic_memory or not semantic_memory.is_trained:
            return []
        
        query_emb = semantic_memory.sentence_embedding(query_words)
        if query_emb is None:
            return []
        
        # Buscar en los temas más relevantes primero
        topic_scores = {}
        for topic_id, centroid in self.topic_centroids.items():
            sim = np.dot(query_emb, centroid) / (
                np.linalg.norm(query_emb) * np.linalg.norm(centroid) + 1e-8
            )
            topic_scores[topic_id] = sim
        
        # Ordenar temas por relevancia
        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Buscar oraciones en los top temas
        results = []
        
        for topic_id, topic_sim in sorted_topics[:5]:
            for sent_idx in self.topic_sentences.get(topic_id, []):
                if sent_idx < len(sentences):
                    sent_emb = semantic_memory.sentence_embedding(sentences[sent_idx])
                    if sent_emb is not None:
                        sim = np.dot(query_emb, sent_emb) / (
                            np.linalg.norm(query_emb) * np.linalg.norm(sent_emb) + 1e-8
                        )
                        results.append((float(sim), sent_idx))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]
    
    def get_topic_description(self, topic_id: int) -> str:
        """Describe un tema por sus palabras más frecuentes."""
        words = self.topic_words.get(topic_id, [])
        return ', '.join(words[:5]) if words else "desconocido"
    
    @property
    def is_trained(self) -> bool:
        return self._trained
