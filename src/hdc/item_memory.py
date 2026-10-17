"""
Memoria de Ítems (Item Memory) para Computación Hiperdimensional
=================================================================

Almacena asociaciones clave→hipervector para consulta por similitud.
Fundamental para codificar símbolos discretos (caracteres, tokens, lemas)
como hipervectores estables y recuperables.

Referencias:
    - Kanerva, P. (2009). Hyperdimensional Computing.
      Cognitive Computation, 1(2), 139-159.

Clasificación: E3 — Fundamento formal.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .core import HyperdimensionalVectors


class ItemMemory:
    """
    Memoria asociativa de ítems para HDC.
    
    Almacena pares (clave, hipervector) y permite recuperación por similitud.
    
    Parámetros
    ----------
    hdc : HyperdimensionalVectors
        Instancia de HDC que define el espacio de representación.
    
    Atributos
    ---------
    capacity : int
        Número de ítems almacenados.
    collision_rate : float
        Tasa estimada de colisiones entre hipervectores almacenados.
    """
    
    def __init__(self, hdc: HyperdimensionalVectors):
        self.hdc = hdc
        self._items: Dict[Any, np.ndarray] = {}
        self._reverse_index: Optional[np.ndarray] = None
        self._keys_list: List[Any] = []
        self._dirty = True  # Flag para reconstruir índice
    
    @property
    def capacity(self) -> int:
        """Número de ítems almacenados."""
        return len(self._items)
    
    @property
    def dimension(self) -> int:
        """Dimensión del espacio HDC."""
        return self.hdc.dimension
    
    def store(self, key: Any, vector: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Almacena un ítem en la memoria.
        
        Si no se proporciona un hipervector, se genera uno aleatorio.
        
        Parámetros
        ----------
        key : Any
            Clave identificadora (hashable).
        vector : np.ndarray, optional
            Hipervector a almacenar. Si None, se genera aleatoriamente.
        
        Retorna
        -------
        np.ndarray
            El hipervector almacenado.
        
        Raises
        ------
        ValueError
            Si el vector tiene dimensión incorrecta.
        """
        if key in self._items:
            return self._items[key]
        
        if vector is None:
            vector = self.hdc.generate(1)
        else:
            vector = np.asarray(vector, dtype=np.int8)
            if vector.shape != (self.hdc.dimension,):
                raise ValueError(
                    f"Dimensión esperada: {self.hdc.dimension}, "
                    f"recibida: {vector.shape}"
                )
        
        self._items[key] = vector
        self._keys_list.append(key)
        self._dirty = True
        return vector
    
    def store_batch(self, keys: List[Any], 
                    vectors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Almacena múltiples ítems simultáneamente.
        
        Parámetros
        ----------
        keys : list
            Lista de claves.
        vectors : np.ndarray, optional
            Array de forma (len(keys), D). Si None, se generan aleatoriamente.
        
        Retorna
        -------
        np.ndarray
            Array de hipervectores almacenados.
        """
        n = len(keys)
        if vectors is None:
            vectors = self.hdc.generate(n)
        elif vectors.shape[0] != n:
            raise ValueError(
                f"Número de vectores ({vectors.shape[0]}) no coincide "
                f"con número de claves ({n})"
            )
        
        for i, key in enumerate(keys):
            if key not in self._items:
                self._items[key] = vectors[i]
                self._keys_list.append(key)
        
        self._dirty = True
        return vectors
    
    def get(self, key: Any) -> Optional[np.ndarray]:
        """
        Recupera el hipervector asociado a una clave exacta.
        
        Parámetros
        ----------
        key : Any
            Clave a buscar.
        
        Retorna
        -------
        np.ndarray o None
            Hipervector si existe, None si no.
        """
        return self._items.get(key)
    
    def _rebuild_index(self):
        """Reconstruye el índice matricial para búsqueda rápida."""
        if not self._dirty:
            return
        
        if len(self._items) == 0:
            self._reverse_index = np.empty((0, self.hdc.dimension), dtype=np.int8)
            self._keys_list = []
        else:
            self._reverse_index = np.stack(
                [self._items[k] for k in self._keys_list], axis=0
            )
        
        self._dirty = False
    
    def query(self, vector: np.ndarray, top_k: int = 1,
              threshold: float = -1.0) -> List[Tuple[Any, float]]:
        """
        Consulta por similitud: recupera los ítems más similares a un vector.
        
        Parámetros
        ----------
        vector : np.ndarray
            Hipervector de consulta, forma (D,).
        top_k : int
            Número de resultados a retornar.
        threshold : float
            Umbral mínimo de similitud. Por defecto, sin umbral.
        
        Retorna
        -------
        list de (clave, similitud)
            Los top_k ítems más similares, ordenados de mayor a menor similitud.
        """
        self._rebuild_index()
        
        if self.capacity == 0:
            return []
        
        vector = np.asarray(vector, dtype=np.float64)
        matrix = self._reverse_index.astype(np.float64)
        
        # Similitud coseno vectorizada
        dots = matrix @ vector
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
        norms = np.where(norms == 0, 1.0, norms)
        similarities = dots / norms
        
        # Aplicar umbral
        if threshold > -1.0:
            mask = similarities >= threshold
            similarities = similarities * mask + (-2.0) * (~mask)
        
        # Top-k
        k = min(top_k, self.capacity)
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        results = [
            (self._keys_list[i], float(similarities[i]))
            for i in top_indices
            if similarities[i] >= threshold
        ]
        
        return results
    
    def query_exact(self, vector: np.ndarray) -> Tuple[Any, float]:
        """
        Consulta el ítem más similar (equivalente a query con top_k=1).
        
        Retorna
        -------
        tuple (clave, similitud)
        """
        results = self.query(vector, top_k=1)
        if results:
            return results[0]
        return (None, 0.0)
    
    def collision_analysis(self) -> dict:
        """
        Analiza la distribución de similitudes entre todos los pares almacenados.
        
        Retorna estadísticas sobre:
        - Similitud máxima entre pares distintos (potencial colisión)
        - Similitud media
        - Número de pares con similitud > 0.1 (umbral de interferencia)
        
        Clasificación: E1 cuando se ejecuta con datos reales.
        """
        self._rebuild_index()
        
        if self.capacity < 2:
            return {
                "capacity": self.capacity,
                "max_pairwise_similarity": 0.0,
                "mean_pairwise_similarity": 0.0,
                "pairs_above_0.1": 0,
                "expected_max_similarity": 0.0,
            }
        
        matrix = self._reverse_index.astype(np.float64)
        n = matrix.shape[0]
        
        # Matriz de similitud completa
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = matrix / norms
        sim_matrix = normalized @ normalized.T
        
        # Excluir diagonal
        np.fill_diagonal(sim_matrix, -2.0)
        
        max_sim = float(np.max(sim_matrix))
        mean_sim = float(np.mean(sim_matrix[sim_matrix > -2.0]))
        pairs_above_threshold = int(np.sum(sim_matrix > 0.1))
        
        # Similitud máxima esperada teóricamente
        # Para n vectores aleatorios en dimensión D, la similitud máxima
        # esperada es aproximadamente sqrt(2 * ln(n) / D)
        expected_max = np.sqrt(2 * np.log(n) / self.hdc.dimension) if n > 1 else 0.0
        
        return {
            "capacity": self.capacity,
            "dimension": self.hdc.dimension,
            "max_pairwise_similarity": max_sim,
            "mean_pairwise_similarity": mean_sim,
            "pairs_above_0.1": pairs_above_threshold,
            "total_pairs": n * (n - 1) // 2,
            "expected_max_similarity": float(expected_max),
            "collision_rate": pairs_above_threshold / max(1, n * (n - 1) // 2),
        }
    
    def get_all_keys(self) -> List[Any]:
        """Retorna todas las claves almacenadas."""
        return list(self._keys_list)
    
    def __contains__(self, key: Any) -> bool:
        return key in self._items
    
    def __len__(self) -> int:
        return len(self._items)
    
    def __repr__(self) -> str:
        return (
            f"ItemMemory(capacity={self.capacity}, "
            f"dimension={self.dimension})"
        )


class SpanishCharacterMemory(ItemMemory):
    """
    Memoria de ítems especializada para caracteres del español.
    
    Incluye el inventario completo:
    - Letras minúsculas y mayúsculas (incluyendo ñ, á, é, í, ó, ú, ü)
    - Dígitos
    - Signos de puntuación
    - Espacio
    """
    
    # Inventario de caracteres del español
    LOWERCASE = "abcdefghijklmnñopqrstuvwxyz"
    UPPERCASE = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZ"
    ACCENTED_LOWER = "áéíóúü"
    ACCENTED_UPPER = "ÁÉÍÓÚÜ"
    DIGITS = "0123456789"
    PUNCTUATION = ".,;:!?¡¿\"'()-—…«»"
    WHITESPACE = " \t\n"
    
    @classmethod
    def full_inventory(cls) -> List[str]:
        """Retorna el inventario completo de caracteres del español."""
        return list(
            cls.LOWERCASE + cls.UPPERCASE +
            cls.ACCENTED_LOWER + cls.ACCENTED_UPPER +
            cls.DIGITS + cls.PUNCTUATION + cls.WHITESPACE
        )
    
    @classmethod
    def basic_inventory(cls) -> List[str]:
        """Inventario básico: minúsculas + dígitos + espacio + puntuación común."""
        return list(
            cls.LOWERCASE + cls.DIGITS + " .,;:!?"
        )
    
    def __init__(self, hdc: HyperdimensionalVectors, 
                 inventory: str = "basic"):
        """
        Inicializa la memoria con el inventario de caracteres del español.
        
        Parámetros
        ----------
        hdc : HyperdimensionalVectors
            Instancia de HDC.
        inventory : str
            "basic" o "full". Determina el conjunto de caracteres.
        """
        super().__init__(hdc)
        
        if inventory == "basic":
            chars = self.basic_inventory()
        elif inventory == "full":
            chars = self.full_inventory()
        else:
            raise ValueError(f"Inventario desconocido: {inventory}")
        
        self.store_batch(chars)
        self.inventory_type = inventory
