# DISCOURSE-002: Continuidad Temática

**ID:** DISCOURSE-002  
**Fase:** 4 — Discurso  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso

---

## Objetivo

Detectar cambios de tema (topic boundaries) en el discurso, identificando
puntos donde la cohesión temática entre oraciones consecutivas se rompe.

## Enfoque

Segmentador temático basado en similitud léxica:
- Cada oración se representa como bag-of-words (lemmas de content words: NOUN, VERB, ADJ, ADV, PROPN)
- Se calcula similitud coseno entre oraciones consecutivas
- Umbral adaptativo: media - 0.5 × desviación estándar
- Puntos por debajo del umbral se marcan como cambio de tema

## Datos

| Conjunto | Oraciones |
|----------|----------|
| Train    | 14,287   |
| Test     | 1,721    |

### Estadísticas de similitud
- Similitud media entre oraciones consecutivas: 0.0675 ± 0.0957
- Umbral ajustado: 0.0196

## Resultados

| Métrica | Modelo | Baseline mejor |
|---------|--------|----------------|
| **F1** | **66.71%** | 37.24% |
| Precision | 61.14% | 43.28% |
| Recall | 73.39% | 32.67% |

### Comparación con baselines

| Estrategia | Precision | Recall | F1 |
|-----------|-----------|--------|-----|
| **Modelo (umbral adaptativo)** | **61.14%** | **73.39%** | **66.71%** |
| Boundary cada 3 oraciones | 43.28% | 32.67% | 37.24% |
| Boundary cada 5 oraciones | 44.19% | 20.03% | 27.56% |
| Boundary cada 10 oraciones | 42.44% | 9.62% | 15.68% |

### Detalle de conteos
- TP = 557 (cambios correctamente detectados)
- FP = 354 (falsos cambios)
- FN = 202 (cambios no detectados)

## Análisis

El modelo supera al mejor baseline por **+29.47 puntos F1** (+79% mejora relativa).
La clave del éxito es el umbral adaptativo basado en la distribución de
similitudes del corpus, que captura la estructura temática real del discurso.

El recall (73.39%) es mayor que la precision (61.14%), indicando que el
modelo tiende a sobre-detectar boundaries, pero captura la mayoría de los
cambios reales.

## Conclusión

La segmentación temática basada en similitud léxica con umbral adaptativo
es efectiva para detectar cambios de tema en discurso en español. El modelo
captura la estructura temática del texto sin necesidad de conocimiento
externo ni embeddings complejos.

**Claim asociado:** CLAIM-026 — El modelo detecta estructura discursiva
(segmentación temática con F1=66.71%).
