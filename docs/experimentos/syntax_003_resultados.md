# SYNTAX-003: Análisis de Dependencias Completo (LAS)
## Informe de Resultados — Fase 2: Sintaxis

**Fecha:** 2026-10-17  
**Estado:** ✅ Completado  
**Corpus:** UD Spanish-AnCora

---

## 1. Objetivo

Combinar SYNTAX-001 (cabeza) + SYNTAX-002 (relación) para calcular LAS (Labeled Attachment Score).

---

## 2. Resultados

| Métrica | Valor | Objetivo |
|---------|-------|----------|
| **UAS** | **62.43%** | 60-70% ✅ |
| **LAS** | **58.68%** | 55-65% ✅ |
| Exact Match | 0.23% | — |
| Brecha UAS-LAS | 3.75% | — |

**Correctos:**
- UAS: 33,474/53,622
- LAS: 31,465/53,622
- Oraciones completas: 4/1,721

**Observación:** La brecha UAS-LAS de 3.75% indica que cuando la cabeza es correcta, la relación también suele ser correcta (~94% de las veces).

---

## 3. Comparación con estado del arte

| Sistema | UAS | LAS | Método |
|---------|-----|-----|--------|
| **SYNTAX-003** | **62.43%** | **58.68%** | HDC + item memory |
| MaltParser | ~85% | ~80% | ML tradicional |
| UDPipe (stanza) | ~95% | ~93% | Deep learning |
| SpaCy es_core_news_lg | ~94% | ~92% | Deep learning |

**Brecha:** ~25-30 puntos por debajo de sistemas profesionales.

---

## 4. CLAIM-020: LAS de 58.68% con pipeline HDC

```json
{
  "id": "CLAIM-020",
  "claim": "Pipeline HDC (cabeza + relación) alcanza LAS=58.68% y UAS=62.43% en análisis de dependencias completo",
  "status": "validado",
  "evidence_type": "E1"
}
```
