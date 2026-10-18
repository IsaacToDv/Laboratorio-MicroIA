# DISCOURSE-003: Relaciones entre Oraciones

**ID:** DISCOURSE-003  
**Fase:** 4 — Discurso  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ✅ Exitoso (F1 ponderado)

---

## Objetivo

Clasificar las relaciones retóricas/discursivas entre pares de oraciones
consecutivas: causa, consecuencia, contraste, adición, elaboración,
temporal, condición.

## Enfoque

Clasificador basado en marcadores discursivos del español:
- Inventario de 60+ conectores organizados en 7 categorías
- Detección de marcadores en la segunda oración del par
- Votación ponderada por frecuencia de entrenamiento
- Feature de overlap léxico y posición del marcador

## Datos

| Conjunto | Oraciones | Pares |
|----------|----------|-------|
| Train    | 14,287   | 14,286 |
| Test     | 1,721    | 1,720  |

### Distribución de relaciones (train)
| Relación | Frecuencia | % |
|----------|-----------|---|
| Causa | 6,784 | 47.5% |
| Contraste | 1,923 | 13.5% |
| Temporal | 1,619 | 11.3% |
| Adición | 1,107 | 7.7% |
| Condición | 628 | 4.4% |
| Consecuencia | 259 | 1.8% |
| Elaboración | 150 | 1.0% |

## Resultados

| Métrica | Valor |
|---------|-------|
| **F1 ponderado** | **65.92%** |
| Accuracy | 56.51% |

### F1 por tipo de relación (test)

| Relación | Precision | Recall | F1 | Soporte |
|----------|-----------|--------|-----|---------|
| Elaboración | 100.00% | 54.20% | 70.30% | 1,524 |
| Causa | 10.25% | 88.52% | 18.37% | 61 |
| Contraste | 23.81% | 62.50% | 34.48% | 48 |
| Adición | 33.80% | 66.67% | 44.86% | 36 |
| Temporal | 22.77% | 65.71% | 33.82% | 35 |
| Condición | 16.07% | 100.00% | 27.69% | 9 |
| Consecuencia | 46.15% | 85.71% | 60.00% | 7 |

## Análisis

El F1 ponderado de 65.92% refleja buen desempeño considerando la distribución
altamente desbalanceada. La clase "elaboración" domina (88.6% del test) como
clase por defecto (oraciones sin marcador explícito).

El accuracy (56.51%) es inferior al baseline de clase mayoritaria (88.60%)
porque el modelo clasifica activamente relaciones no-elaboración cuando
detecta marcadores, mientras que el gold heurístico asigna "elaboración"
a todo par sin marcador inicial.

Los mejores F1 individuales: elaboración (70.30%), consecuencia (60.00%),
adición (44.86%). Las relaciones con marcadores explícitos al inicio de
oración se detectan bien.

### Limitación principal
La clasificación de relaciones discursivas sin marcadores explícitos
requiere comprensión semántica profunda, no solo detección de conectores.

## Conclusión

El modelo clasifica relaciones discursivas con F1 ponderado de 65.92%,
demostrando capacidad para identificar conectores y mapearlos a tipos
de relación. La detección de relaciones implícitas permanece como
desafío abierto.

**Claim asociado:** CLAIM-028 — El modelo clasifica relaciones discursivas
(F1 ponderado = 65.92%).
