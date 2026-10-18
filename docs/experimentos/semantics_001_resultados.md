# SEMANTICS-001: Word Sense Disambiguation (WSD)
## Informe de Resultados — Fase 3: Semántica

**Fecha:** 2026-10-18  
**Estado:** ✅ Completado  
**Corpus:** UD Spanish-AnCora

---

## 1. Objetivo

Desambiguar palabras polisémicas usando contexto HDC + item memory.

## 2. Resultados

| Método | Accuracy | Total |
|--------|----------|-------|
| Baseline (most frequent) | 95.52% | 185,686 |
| **SEMANTICS-001** | **97.00%** | **21,533** |
| **Mejora** | **+1.48%** | |

- **Palabras ambiguas identificadas:** 1,322
- **Claves contextuales:** 30,242

### Top palabras ambiguas (resultados)

| Palabra | Accuracy | Ejemplos |
|---------|----------|----------|
| de | 99.84% | 3,753/3,759 |
| la | 99.53% | 1,899/1,908 |
| en | 99.93% | 1,369/1,370 |
| a | 99.92% | 1,291/1,292 |
| y | 100.00% | 1,113/1,113 |

## 3. CLAIM-022: WSD 97.00% con contexto HDC

```json
{
  "id": "CLAIM-022",
  "claim": "HDC con item memory y contexto alcanza 97.00% accuracy en desambiguación de palabras polisémicas, superando baseline most-frequent (95.52%)",
  "status": "validado",
  "evidence_type": "E1"
}
```
