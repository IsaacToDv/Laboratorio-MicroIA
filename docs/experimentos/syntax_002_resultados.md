# SYNTAX-002: Clasificación de Relaciones de Dependencia
## Informe de Resultados — Fase 2: Sintaxis

**Fecha:** 2026-10-17  
**Estado:** ✅ Completado  
**Corpus:** UD Spanish-AnCora (453k pares train / 54k test)

---

## 1. Objetivo

Predecir la etiqueta de relación de dependencia (nsubj, obj, det, etc.) dado un par (dependiente, cabeza).

**Métrica:** Accuracy de clasificación

---

## 2. Resultados

| Método | Accuracy | Δ vs Majority |
|--------|----------|---------------|
| Majority (det) | 15.01% | — |
| POS-pair | 80.16% | +65.14% |
| **SYNTAX-002** | **87.59%** | **+72.57%** |

**Objetivo:** 60-75% → ✅ Superado (87.59%)

### Accuracy por relación (top 10)

| Relación | Accuracy | Count |
|----------|----------|-------|
| det | 99.63% | 8,048 |
| case | 99.33% | 7,315 |
| punct | 100.00% | 6,334 |
| nmod | 91.13% | 3,686 |
| amod | 98.55% | 2,836 |
| mark | 97.17% | 1,907 |
| advmod | 93.78% | 1,802 |
| nsubj | 81.72% | 2,905 |
| obj | 66.58% | 2,277 |
| obl | 57.40% | 2,474 |

**Observación:** Relaciones funcionales (det, case, punct) son muy predecibles. Relaciones semánticas (obl, obj) son más ambiguas.

---

## 3. CLAIM-019: 87.59% en clasificación de dependencias

```json
{
  "id": "CLAIM-019",
  "claim": "HDC alcanza 87.59% accuracy en clasificación de relaciones de dependencia, superando baseline POS-pair (80.16%)",
  "status": "validado",
  "evidence_type": "E1"
}
```
