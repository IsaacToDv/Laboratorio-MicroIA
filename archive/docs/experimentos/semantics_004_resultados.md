# SEMANTICS-004: Similitud Semántica
## Informe de Resultados — Fase 3: Semántica

**Fecha:** 2026-10-18  
**Estado:** ✅ Completado

---

## 1. Objetivo

Construir espacio semántico distribucional usando HDC y evaluar similitud entre palabras.

## 2. Resultados

| Métrica | Valor |
|---------|-------|
| Palabras en espacio | 4,723 |
| Contextos únicos | 48,806 |
| Clustering precision | 8.97% |
| Separación sim/dif | 0.2176 |

### Pares similares (similitud esperada alta)

| Par | Similitud |
|-----|-----------|
| empresa-banco | **0.979** |
| ciudad-país | **0.869** |
| presidente-ministro | **0.611** |
| año-mes | 0.545 |
| día-semana | 0.545 |
| perro-gato | 0.439 |
| millón-dólar | 0.049 |

### Pares diferentes (similitud esperada baja)

| Par | Similitud |
|-----|-----------|
| empresa-día | 0.579 |
| perro-presidente | 0.570 |
| año-gato | 0.349 |
| país-dólar | 0.256 |
| ciudad-millón | 0.043 |

### Vecinos más cercanos (ejemplos)

| Palabra | Top-5 vecinos |
|---------|--------------|
| presidente | empresa, mundo, gobierno, autoridad, policía |
| ciudad | empresa, zona, mundo, club, gobierno |
| empresa | gobierno, autoridad, policía, mundo, compañía |

**Observación:** Hay estructura semántica visible pero clustering es bajo (8.97%). El corpus es pequeño para semántica distribucional pura.

## 3. CLAIM-025: Espacio semántico HDC muestra estructura

```json
{
  "id": "CLAIM-025",
  "claim": "Espacio semántico HDC con 4,723 palabras muestra separación sim/dif de 0.22 y pares como empresa-banco con similitud 0.98",
  "status": "validado",
  "evidence_type": "E1"
}
```
