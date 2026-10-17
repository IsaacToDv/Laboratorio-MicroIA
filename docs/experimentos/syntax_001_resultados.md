# SYNTAX-001: Análisis de Dependencias Básico (UAS)
## Informe de Resultados — Fase 2: Sintaxis

**Fecha:** 2026-10-17  
**Estado:** ✅ Completado  
**Corpus:** UD Spanish-AnCora (14,287 train / 1,721 test)

---

## 1. Objetivo

Predecir la cabeza (head) de cada token usando HDC + item memory.

**Métrica:** UAS (Unlabeled Attachment Score)

---

## 2. Resultados

| Método | UAS | Δ vs Left-adj |
|--------|-----|---------------|
| Root-only | 3.21% | -6.59% |
| Left-adjacent | 9.80% | — |
| Right-adjacent | 30.55% | +20.75% |
| **SYNTAX-001** | **62.43%** | **+52.63%** |

**Root accuracy:** 80.71%  
**Objetivo:** 60-70% → ✅ Alcanzado (62.43%)

### Distribución de distancias de cabeza (train)

| Distancia | Frecuencia | % |
|-----------|-----------|---|
| +1 (token siguiente) | 138,195 | 30.50% |
| +2 | 63,809 | 14.08% |
| -1 (token anterior) | 44,149 | 9.75% |
| -2 | 43,031 | 9.50% |
| -3 | 33,116 | 7.31% |

- Cabeza a la derecha: 54.99%
- Cabeza a la izquierda: 41.86%
- Raíces: 3.15%

---

## 3. CLAIM-018: UAS de 62.43% con HDC

```json
{
  "id": "CLAIM-018",
  "claim": "HDC con item memory y features contextuales alcanza 62.43% UAS en análisis de dependencias del español",
  "status": "validado",
  "evidence_type": "E1"
}
```
