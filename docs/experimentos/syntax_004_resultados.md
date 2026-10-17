# SYNTAX-004: Concordancia Gramatical
## Informe de Resultados — Fase 2: Sintaxis

**Fecha:** 2026-10-17  
**Estado:** ✅ Completado  
**Corpus:** UD Spanish-AnCora

---

## 1. Objetivo

Detectar concordancia de género y número en pares:
- Determinante + Sustantivo (det_noun)
- Adjetivo + Sustantivo (amod_noun)
- Sujeto + Verbo (nsubj_verb)

---

## 2. Pares de concordancia encontrados

| Tipo | Train | Test |
|------|-------|------|
| det_noun | 55,526 | 6,574 |
| amod_noun | 22,236 | 2,623 |
| nsubj_verb | 20,837 | 2,482 |
| **Total** | **98,599** | **11,679** |

---

## 3. Resultados

### Con features gold (techo teórico)

| Tipo | Gender | Number | Full |
|------|--------|--------|------|
| det_noun | 99.44% | 99.82% | **99.33%** |
| amod_noun | 99.35% | 99.09% | **98.78%** |
| nsubj_verb | 97.70% | 96.70% | **95.08%** |
| **TOTAL** | — | — | **98.30%** |

### Con modelo de sufijos (predicción)

| Tipo | Gender | Number | Full | Objetivo |
|------|--------|--------|------|----------|
| det_noun | 93.82% | 98.80% | **92.71%** | >90% ✅ |
| amod_noun | 96.19% | 97.71% | **94.28%** | >90% ✅ |
| nsubj_verb | 97.58% | 86.74% | **85.05%** | >90% ⚠️ |
| **TOTAL** | — | — | **91.44%** | >90% ✅ |

**Observación:** El modelo de sufijos alcanza 91.44% total, superando el objetivo de 90%.

- det_noun: 92.71% — muy predecible por terminaciones -o/-a
- amod_noun: 94.28% — adjetivos tienen patrones claros
- nsubj_verb: 85.05% — más difícil por irregularidades verbales

---

## 4. CLAIM-021: Concordancia gramatical 91.44%

```json
{
  "id": "CLAIM-021",
  "claim": "Modelo de sufijos alcanza 91.44% accuracy en detección de concordancia gramatical (género + número) en pares det-noun, adj-noun y nsubj-verb",
  "status": "validado",
  "evidence_type": "E1"
}
```
