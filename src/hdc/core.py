"""
Módulo A — Computación Hiperdimensional (HDC/VSA)
===================================================

Implementación de operaciones fundamentales de Hyperdimensional Computing
con Vectores Simbólicos (VSA) para representación lingüística.

Referencias:
    - Kanerva, P. (2009). Hyperdimensional Computing: An Introduction to
      Computing in Distributed Representation with High-Dimensional Random Vectors.
      Cognitive Computation, 1(2), 139-159. https://doi.org/10.1007/s12559-009-9009-8

    - Schlegel, P., Neubert, P., & Protzel, P. (2022). A Comparison of
      Vector Symbolic Architectures. arXiv:2201.10668

Clasificación de evidencia: E3 — Fundamento formal (propiedades matemáticas derivadas)
"""

import numpy as np
from typing import Optional, Union, List, Tuple


class HyperdimensionalVectors:
    """
    Operaciones fundamentales de Computación Hiperdimensional.
    
    Representación: vectores bipolares {-1, +1}^D
    
    Operaciones implementadas:
        - generate(): generación de hipervectores aleatorios
        - bind(): vinculación (binding) — producto componente a componente
        - bundle(): agrupación (bundling) — suma con cuantización
        - permute(): permutación circular
        - similarity(): similitud coseno
        - unbind(): desvinculación (inversa del binding)
    
    Parámetros
    ----------
    dimension : int
        Dimensión D del espacio hiperdimensional.
        Configuración de referencia: D = 10.000.
        Rango explorado: {1024, 2048, 4096, 8192, 10000, 20000}.
    seed : int, optional
        Semilla para reproducibilidad.
    
    Notas
    -----
    - Los vectores aleatorios bipolares en dimensión D tienen similitud
      coseno esperada E[cos] = 0 con varianza σ² = 1/D.
    - La capacidad de memoria asociativa escala como O(D / log D) para
      recuperación con tasa de error acotada.
    - El binding es autoinverso: bind(bind(a, b), b) = a.
    - El bundle es conmutativo: bundle(a, b) = bundle(b, a).
    """
    
    def __init__(self, dimension: int = 10000, seed: Optional[int] = None):
        if dimension < 1:
            raise ValueError(f"La dimensión debe ser >= 1, recibida: {dimension}")
        
        self.dimension = dimension
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
        # Estadísticas teóricas para validación
        self._expected_similarity_mean = 0.0
        self._expected_similarity_std = 1.0 / np.sqrt(dimension)
    
    def generate(self, n: int = 1) -> np.ndarray:
        """
        Genera n hipervectores bipolares aleatorios {-1, +1}^D.
        
        Parámetros
        ----------
        n : int
            Número de hipervectores a generar.
        
        Retorna
        -------
        np.ndarray
            Array de forma (n, D) si n > 1, o (D,) si n = 1.
            Valores en {-1, +1}.
        
        Ejemplo
        -------
        >>> hdc = HyperdimensionalVectors(dimension=10000, seed=42)
        >>> v = hdc.generate(1)
        >>> v.shape
        (10000,)
        >>> set(np.unique(v))
        {-1, 1}
        """
        if n < 1:
            raise ValueError(f"n debe ser >= 1, recibido: {n}")
        
        bits = self._rng.randint(0, 2, size=(n, self.dimension))
        vectors = 2 * bits - 1  # {0, 1} -> {-1, +1}
        
        if n == 1:
            return vectors[0]
        return vectors
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Vinculación (Binding): producto componente a componente.
        
        Para vectores bipolares: bind(a, b) = a ⊙ b (Hadamard product).
        
        Propiedades:
            - Autoinverso: bind(bind(a, b), b) = a
            - Conmutativo: bind(a, b) = bind(b, a)
            - Preserva similitud: bind(a, b) es similar a bind(a, c) 
              solo si b es similar a c
        
        Parámetros
        ----------
        a, b : np.ndarray
            Hipervectores de forma (..., D).
        
        Retorna
        -------
        np.ndarray
            Hipervector resultado de forma (..., D).
        
        Clasificación: E3 — Propiedad algebraica derivada.
        """
        a = np.asarray(a, dtype=np.int8)
        b = np.asarray(b, dtype=np.int8)
        return a * b
    
    def bundle(self, *vectors: np.ndarray, quantize: bool = True) -> np.ndarray:
        """
        Agrupación (Bundling): suma con cuantización a signo.
        
        bundle(v1, v2, ..., vn) = sign(sum(vi))
        
        Regla de desempate: sign(0) = +1 (convención arbitraria pero consistente).
        
        Propiedades:
            - Conmutativo: bundle(a, b) = bundle(b, a)
            - El resultado es similar a todos los operandos
            - La similitud con cada operando decrece como O(1/√n)
        
        Parámetros
        ----------
        *vectors : np.ndarray
            Hipervectores a agrupar, cada uno de forma (..., D).
        quantize : bool
            Si True, cuantiza el resultado a {-1, +1}.
            Si False, retorna la suma sin cuantizar (útil para 
            weighted bundling).
        
        Retorna
        -------
        np.ndarray
            Hipervector agrupado de forma (..., D).
        
        Clasificación: E3 — Propiedad algebraica derivada.
        """
        if len(vectors) == 0:
            raise ValueError("Se requiere al menos un vector para agrupar.")
        
        stacked = np.stack([np.asarray(v) for v in vectors], axis=0)
        result = np.sum(stacked, axis=0)
        
        if quantize:
            # sign(0) = +1 por convención
            result = np.where(result >= 0, 1, -1).astype(np.int8)
        
        return result
    
    def bundle_weighted(self, vectors: np.ndarray, 
                       weights: np.ndarray) -> np.ndarray:
        """
        Agrupación ponderada (Weighted Bundling).
        
        bundle_weighted(V, w) = sign(sum(wi * vi))
        
        Parámetros
        ----------
        vectors : np.ndarray
            Array de forma (n, D).
        weights : np.ndarray
            Array de forma (n,).
        
        Retorna
        -------
        np.ndarray
            Hipervector agrupado de forma (D,).
        """
        vectors = np.asarray(vectors)
        weights = np.asarray(weights).reshape(-1, 1)
        result = np.sum(vectors * weights, axis=0)
        return np.where(result >= 0, 1, -1).astype(np.int8)
    
    def permute(self, v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permutación circular (rotation/shift).
        
        permute(v, k) = v desplazado circularmente k posiciones.
        
        Propiedades:
            - Inyectiva: permute(v, k1) ≠ permute(v, k2) si k1 ≠ k2
            - Invertible: permute(permute(v, k), -k) = v
            - El resultado es disimilar al original para k > 0
        
        Parámetros
        ----------
        v : np.ndarray
            Hipervector de forma (D,).
        shift : int
            Número de posiciones a desplazar. Positivo = derecha,
            negativo = izquierda.
        
        Retorna
        -------
        np.ndarray
            Hipervector permutado de forma (D,).
        
        Uso lingüístico: codificar posición en secuencia.
        """
        v = np.asarray(v)
        return np.roll(v, shift)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """
        Similitud coseno entre hipervectores.
        
        Para vectores bipolares {-1, +1}^D:
            cos(a, b) = (a · b) / (||a|| · ||b||) = (a · b) / D
        
        Rango: [-1, +1]
            +1 = vectores idénticos
             0 = vectores ortogonales (esperado para aleatorios)
            -1 = vectores opuestos
        
        Parámetros
        ----------
        a, b : np.ndarray
            Hipervectores de forma (D,) o (n, D).
        
        Retorna
        -------
        float o np.ndarray
            Similitud coseno.
        """
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        
        if a.ndim == 1 and b.ndim == 1:
            dot = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(dot / (norm_a * norm_b))
        
        # Caso matricial
        if a.ndim == 1:
            a = a.reshape(1, -1)
        if b.ndim == 1:
            b = b.reshape(1, -1)
        
        # Similitud de cada fila de a con cada fila de b
        dot = np.dot(a, b.T)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True)
        
        denominador = norm_a * norm_b.T
        denominador = np.where(denominador == 0, 1.0, denominador)
        
        return dot / denominador
    
    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> Union[float, np.ndarray]:
        """
        Distancia de Hamming normalizada entre hipervectores bipolares.
        
        hamming(a, b) = fracción de componentes diferentes.
        
        Para vectores bipolares:
            hamming(a, b) = (D - a·b) / (2D) = (1 - cos(a,b)) / 2
        
        Rango: [0, 1]
            0   = vectores idénticos
            0.5 = vectores ortogonales (esperado para aleatorios)
            1   = vectores opuestos
        """
        sim = self.similarity(a, b)
        return (1.0 - sim) / 2.0
    
    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Desvinculación (Unbinding): inversa del binding.
        
        unbind(bound, key) = bind(bound, key) para bipolares.
        
        Si bound = bind(a, b), entonces:
            unbind(bound, b) = a
            unbind(bound, a) = b
        
        Parámetros
        ----------
        bound : np.ndarray
            Hipervector resultado de un binding previo.
        key : np.ndarray
            Uno de los operandos originales.
        
        Retorna
        -------
        np.ndarray
            El otro operando recuperado.
        """
        return self.bind(bound, key)
    
    def validate_random_distribution(self, n_samples: int = 1000) -> dict:
        """
        Valida que los hipervectores generados siguen la distribución esperada.
        
        Verifica:
        1. Media de similitud entre pares ≈ 0
        2. Desviación estándar ≈ 1/√D
        3. Todos los componentes en {-1, +1}
        
        Retorna un diccionario con estadísticas y verificación.
        """
        vectors = self.generate(n_samples)
        
        # Similitudes por pares (muestra)
        n_pairs = min(n_samples * (n_samples - 1) // 2, 10000)
        idx_a = self._rng.randint(0, n_samples, n_pairs)
        idx_b = self._rng.randint(0, n_samples, n_pairs)
        # Evitar pares idénticos
        mask = idx_a != idx_b
        idx_a, idx_b = idx_a[mask], idx_b[mask]
        
        sims = np.array([
            self.similarity(vectors[i], vectors[j])
            for i, j in zip(idx_a[:5000], idx_b[:5000])
        ])
        
        return {
            "dimension": self.dimension,
            "n_samples": n_samples,
            "similarity_mean": float(np.mean(sims)),
            "similarity_std": float(np.std(sims)),
            "expected_mean": self._expected_similarity_mean,
            "expected_std": self._expected_similarity_std,
            "mean_deviation": abs(float(np.mean(sims)) - self._expected_similarity_mean),
            "std_deviation": abs(float(np.std(sims)) - self._expected_similarity_std),
            "all_bipolar": bool(np.all((vectors == 1) | (vectors == -1))),
            "mean_within_3sigma": abs(float(np.mean(sims))) < 3 * self._expected_similarity_std,
            "std_within_tolerance": abs(float(np.std(sims)) - self._expected_similarity_std) < 0.2 * self._expected_similarity_std,
        }
