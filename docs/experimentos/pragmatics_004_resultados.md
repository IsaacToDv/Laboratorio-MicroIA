# PRAGMATICS-004: Robustez ante Ruido

**ID:** PRAGMATICS-004  
**Fase:** 5 — Pragmática / Integración  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso

---

## Objetivo

Evaluar la robustez del sistema POS tagger ante texto degradado:
errores tipográficos, ruido de capitalización y errores de OCR.

## Metodología

Se introduce ruido controlado en las formas de los tokens del test
y se mide la degradación del accuracy respecto al baseline (90.38%).

### Tipos de ruido evaluados
1. **Typo noise:** Swaps, deleciones y reemplazos de caracteres
2. **Case noise:** Cambios aleatorios de mayúsculas/minúsculas
3. **OCR noise:** Confusiones típicas de reconocimiento óptico (l↔1, O↔0, etc.)

## Resultados

### Errores Tipográficos

| Tasa ruido | Accuracy | Degradación |
|-----------|----------|-------------|
| 0% (baseline) | 90.38% | — |
| 5% | 84.51% | 6.50% |
| 10% | 80.09% | 11.39% |
| 20% | 72.85% | 19.40% |
| 30% | 66.79% | 26.10% |
| 50% | 59.72% | 33.92% |

### Ruido de Capitalización

| Tasa ruido | Accuracy | Degradación |
|-----------|----------|-------------|
| 0% (baseline) | 90.38% | — |
| 5% | 90.38% | 0.00% |
| 10% | 90.37% | 0.01% |
| 20% | 90.34% | 0.04% |
| 30% | 90.34% | 0.05% |
| 50% | 90.31% | 0.08% |

### Ruido OCR

| Tasa ruido | Accuracy | Degradación |
|-----------|----------|-------------|
| 0% (baseline) | 90.38% | — |
| 5% | 89.93% | 0.50% |
| 10% | 89.40% | 1.09% |
| 20% | 88.55% | 2.03% |
| 30% | 87.63% | 3.05% |
| 50% | 85.36% | 5.56% |

## Comparación a 20% de Ruido

| Tipo de ruido | Accuracy | Degradación |
|--------------|----------|-------------|
| **Capitalización** | **90.34%** | **0.04%** ✅ |
| OCR | 88.55% | 2.03% |
| Tipográfico | 72.85% | 19.40% ⚠️ |
| **Promedio** | — | **7.16%** |

## Análisis

### Robustez excelente ante capitalización (0.04%)
El modelo usa case-insensitive lookup y sufijos, por lo que cambios
de mayúsculas/minúsculas no afectan significativamente.

### Robustez buena ante OCR (2.03%)
Las confusiones OCR (l↔1, O↔0) afectan pocos tokens y los sufijos
siguen siendo reconocibles.

### Vulnerabilidad a errores tipográficos (19.40%)
Los typos alteran los sufijos, que son la principal fuente de
información del modelo. A 50% de ruido, la degradación es 33.92%.

## Conclusión

El sistema es robusto ante ruido de capitalización y OCR, pero
vulnerable a errores tipográficos que afectan sufijos. La degradación
promedio a 20% de ruido es 7.16%, demostrando resiliencia general.

**Claim asociado:** CLAIM-033 — Sistema robusto: degradación promedio
7.16% ante 20% ruido; vulnerable solo a typos (19.40%).
