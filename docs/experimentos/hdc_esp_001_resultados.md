# HDC-Esp-001: Memoria Asociativa para Caracteres del Español
## Informe de Resultados

**Fecha:** 2026-10-17  
**Experimento:** HDC-Esp-001  
**Estado:** Completado  
**Clasificación:** E1 — Resultado reproducido

---

## 1. Objetivo experimental

Validar que un espacio HDC bipolar de dimensión D puede almacenar y recuperar asociaciones carácter→hipervector con tasa de error < 5%, para diferentes valores de D.

## 2. Estado de evidencia

**Hechos conocidos:**
- Las propiedades matemáticas de espacios hiperdimensionales están bien establecidas (Kanerva, 2009)
- La capacidad teórica de memoria asociativa escala como O(D / log D)
- La similitud esperada entre vectores aleatorios es 0 con varianza 1/D

**Hipótesis evaluada:**
- H1: Tasa de error < 5% para D ≥ 4096
- H0: Tasa de error > 20% incluso con D = 10000

## 3. Contribución de los agentes

- **Agente 2 (Arquitecto HDC):** Diseño de operaciones y validación algebraica
- **Agente 7 (Evaluador):** Verificación de pruebas unitarias y análisis de colisiones

## 4. Propuesta técnica

### 4.1. Representación
- Vectores bipolares {-1, +1}^D
- Binding: producto componente a componente (Hadamard)
- Bundle: suma con cuantización a signo
- Similitud: coseno

### 4.2. Inventario de caracteres
- Caracteres básicos: 44 símbolos
  - Letras minúsculas (incluyendo ñ)
  - Dígitos (0-9)
  - Espacio y puntuación común (.,;:!?)

### 4.3. Dimensiones evaluadas
D ∈ {1024, 2048, 4096, 8192, 10000}

## 5. Diseño experimental

**Controles:**
- Semilla fija: 42
- Mismo inventario para todas las dimensiones
- Mismas pruebas de ruido

**Métricas:**
- Tasa de error de recuperación exacta
- Tasa de error con ruido (1%, 5%, 10%, 15%, 20%, 30%, 40%)
- Similitud máxima entre pares almacenados
- Tiempo de consulta
- Memoria utilizada

**Criterio de refutación:**
- Error > 20% con D = 10000

## 6. Resultados

### 6.1. Recuperación exacta (sin ruido)

| Dimensión | Error | Similitud media | Similitud mínima | Tiempo/consulta |
|-----------|-------|-----------------|------------------|-----------------|
| 1024      | 0%    | 1.0000          | 1.0000           | 74.6 μs         |
| 2048      | 0%    | 1.0000          | 1.0000           | 160.2 μs        |
| 4096      | 0%    | 1.0000          | 1.0000           | 324.4 μs        |
| 8192      | 0%    | 1.0000          | 1.0000           | 716.7 μs        |
| 10000     | 0%    | 1.0000          | 1.0000           | 870.1 μs        |

**Resultado:** Recuperación perfecta en todas las dimensiones.

### 6.2. Robustez al ruido

| Dimensión | 1% ruido | 5% ruido | 10% ruido | 20% ruido | 40% ruido |
|-----------|----------|----------|-----------|-----------|-----------|
| 1024      | 0%       | 0%       | 0%        | 0%        | 0%        |
| 2048      | 0%       | 0%       | 0%        | 0%        | 0%        |
| 4096      | 0%       | 0%       | 0%        | 0%        | 0%        |
| 8192      | 0%       | 0%       | 0%        | 0%        | 0%        |
| 10000     | 0%       | 0%       | 0%        | 0%        | 0%        |

**Resultado:** Robustez excepcional incluso con 40% de ruido.

### 6.3. Análisis de colisiones

| Dimensión | Max sim entre pares | Colisiones (sim > 0.1) | Max sim esperada |
|-----------|---------------------|------------------------|------------------|
| 1024      | 0.0977              | 0                      | 0.105            |
| 2048      | 0.0752              | 0                      | 0.074            |
| 4096      | 0.0532              | 0                      | 0.052            |
| 8192      | 0.0381              | 0                      | 0.037            |
| 10000     | 0.0334              | 0                      | 0.033            |

**Resultado:** Sin colisiones observadas. Las similitudes máximas están cerca del valor teórico esperado.

### 6.4. Codificación de secuencias

| Dimensión | Max sim entre palabras | Mean sim entre palabras |
|-----------|------------------------|-------------------------|
| 1024      | 0.0645                 | 0.0050                  |
| 2048      | 0.0723                 | 0.0049                  |
| 4096      | 0.0415                 | -0.0038                 |
| 8192      | 0.0220                 | 0.0010                  |
| 10000     | 0.0208                 | -0.0010                 |

**Resultado:** Palabras diferentes producen hipervectores altamente disimilares.

### 6.5. Uso de memoria

| Dimensión | Memoria por ítem | Memoria total (44 ítems) |
|-----------|------------------|--------------------------|
| 1024      | 1 KB             | 44 KB                    |
| 2048      | 2 KB             | 88 KB                    |
| 4096      | 4 KB             | 176 KB                   |
| 8192      | 8 KB             | 352 KB                   |
| 10000     | 10 KB            | 430 KB                   |

### 6.6. Validación de distribución aleatoria

Para D = 10000:
- Similitud media observada: 0.000008 (esperada: 0.0)
- Desviación estándar observada: 0.009913 (esperada: 0.01)
- Desviación de la media: < 0.001
- Todos los vectores son bipolares: ✓

## 7. Revisión adversarial

### 7.1. Limitaciones identificadas

1. **Inventario pequeño:** Solo 44 caracteres. No se ha probado con vocabularios grandes (> 10k tokens).

2. **Recuperación trivial:** Con solo 44 ítems en D ≥ 1024, la capacidad está muy por debajo del límite teórico. El experimento no prueba los límites de la memoria.

3. **Ruido sintético:** El ruido añadido es aleatorio y uniforme. No simula errores de codificación reales ni degradación de memoria biológica.

4. **Sin aprendizaje:** Este experimento solo prueba almacenamiento y recuperación estática. No hay plasticidad ni adaptación.

5. **Tiempo de consulta lineal:** La consulta por similitud es O(N·D) donde N es el número de ítems. Para vocabularios grandes, esto puede ser prohibitivo.

### 7.2. Explicaciones alternativas

La recuperación perfecta con 44 ítems es esperada matemáticamente:
- Para N = 44 y D = 1024, la capacidad teórica es ~1024 / log(44) ≈ 270 ítems
- Estamos usando solo el 16% de la capacidad disponible
- El experimento no prueba el régimen de alta carga donde emergen errores

### 7.3. Posibles errores

- Ningún error de implementación detectado (todas las pruebas unitarias pasaron)
- La semilla fija garantiza reproducibilidad
- Los resultados son consistentes con la teoría

## 8. Decisión científica

### 8.1. Hipótesis H1: RESPALDADA

**Afirmación:** HDC puede almacenar y recuperar asociaciones carácter→hipervector con error < 5% para D ≥ 1024.

**Evidencia:** Error = 0% en todas las dimensiones probadas.

**Clasificación:** E1 — Resultado reproducido.

**Condiciones:**
- Inventario de 44 caracteres
- Vectores bipolares
- Sin ruido o con ruido ≤ 40%
- Recuperación por similitud coseno

### 8.2. Hipótesis H0: REFUTADA

**Afirmación:** La tasa de error sería > 20% incluso con D = 10000.

**Evidencia:** Error = 0% en todas las dimensiones.

### 8.3. Conclusiones provisionales

1. **HDC es viable para representar caracteres del español** con las dimensiones probadas.

2. **D = 1024 es suficiente** para inventarios pequeños (< 100 símbolos), pero se requieren dimensiones mayores para vocabularios grandes.

3. **La robustez al ruido es excepcional**, lo que sugiere que HDC puede tolerar degradación de memoria y errores de codificación.

4. **El coste de memoria es aceptable** para inventarios pequeños (430 KB para D = 10000).

5. **El tiempo de consulta es lineal** y puede ser un cuello de botella para vocabularios grandes.

### 8.4. Lo que NO se ha demostrado

- Que HDC pueda escalar a vocabularios de 10k-100k tokens
- Que la codificación de secuencias preserve información sintáctica o semántica
- Que HDC sea competitivo con embeddings densos (Word2Vec, GloVe)
- Que la recuperación sea eficiente para búsquedas en tiempo real

## 9. Registro de memoria

### CLAIM-002: HDC es viable para representar caracteres del español

```json
{
  "id": "CLAIM-002",
  "claim": "HDC bipolar puede almacenar y recuperar asociaciones carácter→hipervector con error < 5% para D ≥ 1024",
  "status": "validado",
  "evidence_type": "E1",
  "source": "Experimento HDC-Esp-001, Laboratorio-MicroIA, 2026-10-17",
  "conditions": "Inventario de 44 caracteres, vectores bipolares, recuperación por similitud coseno",
  "method": "Almacenamiento de item memory + consulta por similitud con ruido controlado",
  "metrics": {
    "error_rate_exact": "0%",
    "error_rate_40pct_noise": "0%",
    "max_pairwise_similarity": "0.0977 (D=1024)",
    "query_time": "74.6-870.1 μs"
  },
  "limitations": "Probado solo con 44 caracteres. No validado para vocabularios grandes (> 1k tokens)",
  "contradictions": "Ninguna",
  "next_test": "HDC-Esp-002: Escalabilidad a vocabularios de 1k-100k tokens"
}
```

### CLAIM-003: Robustez excepcional al ruido

```json
{
  "id": "CLAIM-003",
  "claim": "HDC bipolar mantiene recuperación perfecta con hasta 40% de ruido para inventarios pequeños",
  "status": "validado",
  "evidence_type": "E1",
  "source": "Experimento HDC-Esp-001, Laboratorio-MicroIA, 2026-10-17",
  "conditions": "44 caracteres, ruido bipolar aleatorio uniforme",
  "method": "Inversión de bits aleatorios + consulta por similitud",
  "metrics": {
    "error_rate_40pct_noise": "0% (todas las dimensiones)"
  },
  "limitations": "Ruido sintético, no errores de codificación reales",
  "contradictions": "Ninguna",
  "next_test": "Investigar límite de ruido donde emerge error"
}
```

## 10. Siguiente experimento

### HDC-Esp-002: Escalabilidad a vocabularios grandes

**Objetivo:** Determinar el límite práctico de capacidad de HDC para vocabularios de tokens.

**Procedimiento:**
1. Generar vocabularios de tamaño N ∈ {100, 1000, 10000, 50000}
2. Medir tasa de error de recuperación para cada N y D
3. Identificar el punto donde error > 5%
4. Comparar con capacidad teórica O(D / log D)

**Resultado esperado:**
- Identificar la dimensión mínima requerida para vocabularios de diferentes tamaños
- Validar o refutar la fórmula de capacidad teórica

**Criterio de decisión:**
- Si D = 10000 soporta < 10k tokens con error < 5%: proceder a Fase 1 (morfología)
- Si D = 10000 no soporta 10k tokens: evaluar D = 20000 o representación alternativa

**Coste estimado:** 1-2 horas de implementación + experimentación

---

**Firma:** Agente 1 — Director Científico y Orquestador  
**Próxima revisión:** Después de completar HDC-Esp-002
