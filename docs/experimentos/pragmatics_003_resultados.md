# PRAGMATICS-003: Análisis de Errores

**ID:** PRAGMATICS-003  
**Fase:** 5 — Pragmática / Integración  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso (diagnóstico completo)

---

## Objetivo

Análisis sistemático de errores a través de todos los niveles
lingüísticos, identificando patrones, cascadas y categorías problemáticas.

## Métricas Clave

| Métrica | Valor |
|---------|-------|
| Tokens analizados | 35,708 |
| Errores totales (multi-nivel) | 49,527 |
| **Cascade rate** | **40.10%** |
| Morph accuracy | 50.57% |
| Errores en cascada | 19,860 |

## Accuracy por Clase POS

| POS | Accuracy | Total | Observación |
|-----|----------|-------|-------------|
| SCONJ | 89.71% | 826 | Mejor |
| VERB | 86.97% | 3,017 | — |
| NOUN | 82.54% | 6,363 | — |
| ADV | 63.47% | 1,199 | — |
| ADJ | 60.34% | 2,466 | — |
| PROPN | 56.73% | 2,614 | — |
| NUM | 42.97% | 740 | — |
| AUX | 28.11% | 875 | — |
| ADP | 22.46% | 5,615 | Confusión con NOUN |
| DET | 22.39% | 5,346 | Confusión con NOUN |
| CCONJ | 11.33% | 980 | — |
| PRON | 5.46% | 1,629 | Peor |
| PUNCT | 0.15% | 4,026 | Sin sufijo característico |

## Top 10 Errores Morfológicos

| Error | Frecuencia | % del total |
|-------|-----------|-------------|
| ADP → NOUN | 4,211 | 11.79% |
| PUNCT → NOUN | 4,020 | 11.26% |
| DET → NOUN | 3,797 | 10.63% |
| CCONJ → NOUN | 857 | 2.40% |
| PRON → NOUN | 795 | 2.23% |
| PROPN → NOUN | 660 | 1.85% |
| ADJ → NOUN | 599 | 1.68% |
| PRON → SCONJ | 577 | 1.62% |
| NOUN → ADJ | 431 | 1.21% |
| NUM → NOUN | 351 | 0.98% |

## Errores Semánticos (NER)

| Tipo | Frecuencia |
|------|-----------|
| NER miss (entidad no detectada) | 1,131 |
| NER false positive | 807 |

## Análisis de Cascada

El 40.10% de los errores en niveles superiores (sintaxis, semántica)
son causados directamente por un error morfológico en el nivel previo.
Esto confirma que la morfología es el cuello de botella del sistema:
mejorar el POS tagging tendría un efecto multiplicador positivo.

### Patrón dominante
ADP, DET y PUNCT (funcionales, sin sufijos característicos) se
confunden masivamente con NOUN → genera dependencias incorrectas →
roles semánticos erróneos.

## Conclusión

El análisis revela que el 40% de los errores son en cascada, y las
clases funcionales (ADP, DET, PUNCT, PRON) son las más problemáticas
para un modelo basado en sufijos. Mejorar el tratamiento de palabras
funcionales tendría el mayor impacto en el rendimiento global.

**Claim asociado:** CLAIM-032 — 40.10% de errores son en cascada;
clases funcionales son el cuello de botella morfológico.
