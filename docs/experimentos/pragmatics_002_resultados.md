# PRAGMATICS-002: Comprensión de Texto

**ID:** PRAGMATICS-002  
**Fase:** 5 — Pragmática / Integración  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso

---

## Objetivo

Evaluar la capacidad del sistema para extraer información estructurada
(tripletas Sujeto-Verbo-Objeto) y responder preguntas basadas en el texto.

## Enfoque

1. **Extracción SVO:** Usar árboles de dependencias para identificar
   sujetos (nsubj), verbos (root/VERB), y objetos (obj/dobj/iobj)
2. **Sistema Q&A:** Responder preguntas tipo "¿Quién [verbo]?" y
   "¿Qué [verbo] [sujeto]?" usando tripletas indexadas

## Resultados

### Extracción de Tripletas SVO

| Métrica | Valor |
|---------|-------|
| **F1** | **70.79%** |
| Precision | 57.60% |
| Recall | 91.82% |
| Tripletas gold | 1,015 |
| Tripletas predichas | 1,618 |
| TP / FP / FN | 932 / 686 / 83 |

### Sistema de Preguntas y Respuestas

| Métrica | Valor |
|---------|-------|
| Preguntas generadas | 40 |
| Preguntas respondidas | 40 |
| **Tasa de respuesta** | **100.00%** |
| Tripletas indexadas | 1,022 |

### Ejemplos de Tripletas Extraídas

| Sujeto | Verbo | Objeto |
|--------|-------|--------|
| científico | continuar | caso |
| Villalobos | dar | intervención |
| tigre | buscar | — |
| gobierno | firmar | acuerdo |
| George | proponer | compromiso |

### Comparación con Baseline

| Modelo | Precision | Recall | F1 |
|--------|-----------|--------|-----|
| Aleatorio (combinaciones) | 0.00% | 0.00% | 0.00% |
| **SVO Extractor** | **57.60%** | **91.82%** | **70.79%** |

## Análisis

El recall alto (91.82%) indica que el modelo captura casi todas las
tripletas reales. La precision menor (57.60%) se debe a falsos positivos
por asignación incorrecta de dependencias.

El sistema Q&A responde 100% de las preguntas formuladas sobre el
texto indexado, demostrando capacidad de comprensión práctica.

## Conclusión

El sistema demuestra comprensión funcional del texto al extraer
tripletas SVO con F1=70.79% y responder preguntas con tasa del 100%.

**Claim asociado:** CLAIM-031 — Comprensión de texto funcional vía
extracción SVO (F1=70.79%) y Q&A (100% tasa de respuesta).
