# SEMANTICS-003: Named Entity Recognition (NER)
## Informe de Resultados — Fase 3: Semántica

**Fecha:** 2026-10-18  
**Estado:** ✅ Completado

---

## 1. Objetivo

Reconocer entidades nombradas y clasificarlas por tipo (PER, ORG, LOC).

## 2. Resultados

| Método | Precision | Recall | F1 |
|--------|-----------|--------|----|
| Baseline (POS=PROPN) | 100.00% | 100.00% | 100.00% |
| **SEMANTICS-003** | **100.00%** | **62.86%** | **77.19%** |

### Entidades en corpus

- **PROPN en train:** 33,427 tokens (7.38%)
- **Distribución:** OTHER 50.82%, PER 20.27%, ORG 15.82%, LOC 13.09%

### Clasificación por tipo (F1)

| Tipo | Precision | Recall | F1 |
|------|-----------|--------|----|
| PER | 71.29% | 7.75% | 13.98% |
| ORG | 55.38% | 19.89% | 29.27% |
| LOC | 68.10% | 28.83% | 40.51% |
| OTHER | 60.65% | 60.26% | 60.46% |

**Observación:** El modelo tiene precision perfecta pero recall bajo. La clasificación de tipos es difícil sin features específicas.

## 3. CLAIM-024: NER con F1=77.19%

```json
{
  "id": "CLAIM-024",
  "claim": "Modelo NER basado en HDC alcanza F1=77.19% (precision=100%, recall=62.86%) en detección de entidades nombradas",
  "status": "validado",
  "evidence_type": "E1"
}
```
