# SEMANTICS-002: Semantic Role Labeling (SRL)
## Informe de Resultados — Fase 3: Semántica

**Fecha:** 2026-10-18  
**Estado:** ✅ Completado

---

## 1. Objetivo

Clasificar roles semánticos de argumentos verbales (A0=agente, A1=paciente, AM=modificador).

## 2. Resultados

| Método | Accuracy |
|--------|----------|
| Baseline (majority=AM) | 34.95% |
| **SEMANTICS-002** | **78.62%** |
| **Mejora** | **+43.67%** |

### Accuracy por rol

| Rol | Accuracy | Count |
|-----|----------|-------|
| AM (modificador) | 73.72% | 6,168 |
| OTHER | 88.78% | 5,020 |
| A1 (paciente) | 79.61% | 3,909 |
| A0 (agente) | 68.94% | 2,550 |

## 3. CLAIM-023: SRL 78.62% con HDC

```json
{
  "id": "CLAIM-023",
  "claim": "HDC con contexto verbal alcanza 78.62% accuracy en clasificación de roles semánticos, +43.67% sobre baseline",
  "status": "validado",
  "evidence_type": "E1"
}
```
