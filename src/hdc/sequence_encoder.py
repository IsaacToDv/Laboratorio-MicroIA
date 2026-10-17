"""
Codificador de secuencias para Computación Hiperdimensional
============================================================

Transforma secuencias de caracteres o tokens en hipervectores
mediante composición de operaciones HDC.

Estrategias de codificación:
    1. Bag-of-characters: bundle de todos los caracteres (sin orden)
    2. N-grama con permutación: permute(char, position) + bundle
    3. Binding posicional: bind(char_hv, position_hv) + bundle

Referencias:
    - Kanerva, P. (2009). Hyperdimensional Computing.
    - Schlegel, P., et al. (2022). A Comparison of Vector Symbolic Architectures.
"""

import numpy as np
from typing import List, Optional
from .core import HyperdimensionalVectors
from .item_memory import ItemMemory


class SequenceEncoder:
    """
    Codificador de secuencias mediante HDC.
    
    Parámetros
    ----------
    hdc : HyperdimensionalVectors
        Instancia de HDC.
    item_memory : ItemMemory
        Memoria de ítems con los símbolos codificables.
    max_length : int
        Longitud máxima de secuencia a codificar.
    method : str
        Método de codificación: "positional_binding", "permutation", "bag".
    """
    
    def __init__(self, hdc: HyperdimensionalVectors,
                 item_memory: ItemMemory,
                 max_length: int = 100,
                 method: str = "positional_binding"):
        self.hdc = hdc
        self.item_memory = item_memory
        self.max_length = max_length
        self.method = method
        
        # Generar hipervectores de posición
        self._position_memory = ItemMemory(hdc)
        position_keys = [f"__pos_{i}" for i in range(max_length)]
        self._position_memory.store_batch(position_keys)
    
    def encode(self, sequence: str) -> np.ndarray:
        """
        Codifica una secuencia como un hipervector.
        
        Parámetros
        ----------
        sequence : str
            Secuencia de caracteres a codificar.
        
        Retorna
        -------
        np.ndarray
            Hipervector de forma (D,).
        
        Raises
        ------
        ValueError
            Si la secuencia excede max_length.
        """
        if len(sequence) > self.max_length:
            raise ValueError(
                f"Secuencia de longitud {len(sequence)} excede "
                f"max_length={self.max_length}"
            )
        
        if self.method == "bag":
            return self._encode_bag(sequence)
        elif self.method == "positional_binding":
            return self._encode_positional_binding(sequence)
        elif self.method == "permutation":
            return self._encode_permutation(sequence)
        else:
            raise ValueError(f"Método desconocido: {self.method}")
    
    def _encode_bag(self, sequence: str) -> np.ndarray:
        """
        Codificación bag-of-characters: ignora el orden.
        
        bundle(char_1, char_2, ..., char_n)
        
        Limitación: "abc" y "cba" producen el mismo hipervector.
        """
        vectors = []
        for char in sequence:
            v = self.item_memory.get(char)
            if v is not None:
                vectors.append(v)
        
        if not vectors:
            return self.hdc.generate(1)
        
        return self.hdc.bundle(*vectors)
    
    def _encode_positional_binding(self, sequence: str) -> np.ndarray:
        """
        Codificación con binding posicional.
        
        bundle(bind(char_1, pos_1), bind(char_2, pos_2), ...)
        
        Ventaja: preserva información posicional exacta.
        Limitación: secuencias de diferente longitud son muy disimilares.
        """
        vectors = []
        for i, char in enumerate(sequence):
            char_v = self.item_memory.get(char)
            pos_v = self.item_memory.get(f"__pos_{i}")
            
            if char_v is not None and pos_v is not None:
                bound = self.hdc.bind(char_v, pos_v)
                vectors.append(bound)
        
        if not vectors:
            return self.hdc.generate(1)
        
        return self.hdc.bundle(*vectors)
    
    def _encode_permutation(self, sequence: str) -> np.ndarray:
        """
        Codificación con permutación circular.
        
        bundle(permute(char_1, 0), permute(char_2, 1), ...)
        
        Ventaja: preserva orden y es más robusta a desplazamientos.
        """
        vectors = []
        for i, char in enumerate(sequence):
            char_v = self.item_memory.get(char)
            if char_v is not None:
                permuted = self.hdc.permute(char_v, shift=i)
                vectors.append(permuted)
        
        if not vectors:
            return self.hdc.generate(1)
        
        return self.hdc.bundle(*vectors)
    
    def encode_batch(self, sequences: List[str]) -> np.ndarray:
        """
        Codifica múltiples secuencias.
        
        Retorna
        -------
        np.ndarray
            Array de forma (len(sequences), D).
        """
        return np.stack([self.encode(s) for s in sequences], axis=0)
    
    def similarity_matrix(self, sequences: List[str]) -> np.ndarray:
        """
        Calcula la matriz de similitud entre todas las secuencias.
        
        Retorna
        -------
        np.ndarray
            Matriz de forma (n, n) con similitudes coseno.
        """
        encoded = self.encode_batch(sequences)
        return self.hdc.similarity(encoded, encoded)
