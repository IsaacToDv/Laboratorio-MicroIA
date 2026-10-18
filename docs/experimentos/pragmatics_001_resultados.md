# PRAGMATICS-001: Pipeline Integrado

**ID:** PRAGMATICS-001  
**Fase:** 5 — Pragmática / Integración  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso

---

## Objetivo

Integrar los modelos de las 4 fases en un pipeline secuencial
(Morfología → Sintaxis → Semántica → Discurso) y evaluar la
propagación de errores entre niveles.

## Arquitectura del Pipeline

```
  Texto crudo
      │
      ▼
  ┌──────────────┐
  │ Morfología   │ POS tagging (sufijos) + lematización
  └──────┬───────┘
         │ tokens + POS + lemma
         ▼
  ┌──────────────┐
  │ Sintaxis     │ Parsing (POS→deprel, proximidad→head)
  └──────┬───────┘
         │ tokens + head + deprel
         ▼
  ┌──────────────┐
  │ Semántica    │ SRL (verb→roles) + NER (PROPN→entities)
  └──────┬───────┘
         │ tokens + sem_role + ner
         ▼
  ┌──────────────┐
  │ Discurso     │ Entity tracking + Topic segmentation
  └──────┬───────┘
         │
         ▼
  Análisis completo
```

## Resultados por Nivel

| Nivel | Métrica | Accuracy | Correctos/Total |
|-------|---------|----------|-----------------|
| Morfología (POS) | Accuracy | 44.07% | 7,767 / 17,626 |
| Sintaxis (UAS) | Accuracy | 13.67% | 2,410 / 17,626 |
| Sintaxis (LAS) | Accuracy | 5.46% | 963 / 17,626 |
| Semántica (NER) | Accuracy | 95.13% | 16,768 / 17,626 |

### Throughput

| Métrica | Valor |
|---------|-------|
| Tokens procesados | 3,076 |
| Tiempo | 0.059s |
| **Throughput** | **52,313 tokens/s** |

### Información discursiva

| Métrica | Valor |
|---------|-------|
| Entidades detectadas | 147 |
| Menciones | 262 |
| Boundaries temáticos | 1 |

## Análisis de Cascada

| Nivel | Accuracy |
|-------|----------|
| Morph POS | 44.07% |
| Syntax UAS | 13.67% |
| Syntax LAS | 5.46% |
| Sem NER | 95.13% |
| **Cascade teórico** (producto) | **5.73%** |

La cascada de errores es significativa: cada nivel amplifica los errores
del nivel previo. El NER (95.13%) se mantiene robusto porque usa
información local (PROPN detection) independiente de la calidad del
parsing.

## Conclusión

El pipeline integrado procesa >52,000 tokens/s y demuestra la viabilidad
de la arquitectura multinivel. La cascada de errores es el principal
desafío: errores en morfología se propagan a sintaxis y semántica.

**Claim asociado:** CLAIM-030 — Pipeline integrado funcional con
throughput de 52,313 tokens/s y 4 niveles de procesamiento.
