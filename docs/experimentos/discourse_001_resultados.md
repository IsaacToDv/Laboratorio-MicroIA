# DISCOURSE-001: Correferencia

**ID:** DISCOURSE-001  
**Fase:** 4 — Discurso  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ⚠️ Completado (por debajo del baseline)

---

## Objetivo

Evaluar la capacidad del modelo para resolver correferencia pronominal,
vinculando pronombres (lo, la, los, las, le, les, su, sus, él) con sus
antecedentes en el discurso.

## Enfoque

Modelo basado en features heurísticas:
- **Concordancia** de género y número entre pronombre y candidato
- **Distancia** (preferencia por candidatos recientes)
- **Posición sintáctica** (bonus para sujetos y nombres propios)
- **Compatibilidad léxica** (overlap de features morfológicas)

## Datos

| Conjunto | Documentos | Pronombres con candidatos |
|----------|-----------|--------------------------|
| Train    | 2,858     | ~35,000                   |
| Test     | 345       | 4,688                     |

### Pronombres más frecuentes (train)
| Pronombre | Frecuencia |
|-----------|-----------|
| la        | 16,739    |
| los       | 7,036     |
| se        | 5,145     |
| las       | 4,350     |
| su        | 2,924     |
| lo        | 1,686     |

## Resultados

| Métrica | Baseline | Modelo | Δ |
|---------|----------|--------|---|
| Accuracy | 53.28% | 31.80% | -21.48% |

### Baseline
Recent + agreement: seleccionar el candidato más reciente que concuerde
en género y número con el pronombre.

## Análisis

El modelo basado en features ponderadas no logró superar al baseline simple
de recencia + concordancia. Esto indica que:

1. **La recencia es un signal muy fuerte** en correferencia — el antecedente
   correcto suele ser el más reciente compatible.
2. **Los pesos aprendidos sobre-ajustan** a features que no generalizan bien
   al gold estándar heurístico.
3. **La correferencia es inherentemente difícil** sin información semántica
   profunda (embedding contextuales, world knowledge).

### Limitaciones
- El gold estándar es heurístico (basado en proximidad + concordancia),
  lo cual favorece al baseline que usa la misma heurística.
- No se usan embeddings contextuales ni información semántica profunda.
- La resolución de "se" (reflexivo/recíproco) requiere análisis sintáctico
  más sofisticado.

## Conclusión

La correferencia requiere mecanismos más sofisticados que features
superficiales. Un sistema biomimético necesitaría memoria de trabajo con
representaciones semánticas ricas y mecanismos de atención contextual.

**Claim asociado:** CLAIM-026 (parcial — el modelo identifica candidatos
pero no supera la heurística de recencia).
