# Fase 2 — Sintaxis: Resumen Ejecutivo
## Estado de la Investigación: Micro-IA Biomimética HDC–SNN–FEP

**Fecha:** 2026-10-17  
**Responsable:** Agente 1 — Director Científico y Orquestador  
**Estado:** ✅ Fase 2 COMPLETADA con éxito

---

## 📊 Resumen de Resultados

### Experimentos completados

| ID | Experimento | Estado | Resultado principal |
|----|-------------|--------|---------------------|
| **SYNTAX-001** | Análisis de dependencias (UAS) | ✅ Completado | **62.43% UAS** |
| **SYNTAX-002** | Clasificación de relaciones | ✅ Completado | **87.59% accuracy** |
| **SYNTAX-003** | Pipeline completo (LAS) | ✅ Completado | **58.68% LAS** |
| **SYNTAX-004** | Concordancia gramatical | ✅ Completado | **91.44% accuracy** |

---

## 🎯 Resultados Clave

### SYNTAX-001: Análisis de Dependencias (UAS)

**Resultado:** 62.43% UAS (objetivo: 60-70%) ✅

| Método | UAS |
|--------|-----|
| Left-adjacent | 9.80% |
| Right-adjacent | 30.55% |
| **SYNTAX-001** | **62.43%** |

**Hallazgo:** 54.99% de cabezas están a la derecha, 41.86% a la izquierda. Distancia +1 es la más frecuente (30.50%).

### SYNTAX-002: Clasificación de Relaciones

**Resultado:** 87.59% accuracy (objetivo: 60-75%) ✅ (superado)

| Relación | Accuracy |
|----------|----------|
| det | 99.63% |
| punct | 100.00% |
| case | 99.33% |
| amod | 98.55% |
| obl | 57.40% (más difícil) |

### SYNTAX-003: Pipeline Completo (LAS)

**Resultado:** LAS = 58.68% (objetivo: 55-65%) ✅

| Métrica | Valor |
|---------|-------|
| UAS | 62.43% |
| LAS | 58.68% |
| Brecha UAS-LAS | 3.75% |
| Exact Match | 0.23% |

### SYNTAX-004: Concordancia Gramatical

**Resultado:** 91.44% accuracy (objetivo: >90%) ✅

| Tipo | Gold (techo) | Suffix Model |
|------|-------------|--------------|
| det_noun | 99.33% | 92.71% |
| amod_noun | 98.78% | 94.28% |
| nsubj_verb | 95.08% | 85.05% |
| **TOTAL** | **98.30%** | **91.44%** |

---

## 🔬 Decisiones Científicas

### Afirmaciones validadas (CLAIM-018 a CLAIM-021)

✅ **CLAIM-018:** HDC alcanza 62.43% UAS en análisis de dependencias  
✅ **CLAIM-019:** 87.59% accuracy en clasificación de relaciones  
✅ **CLAIM-020:** Pipeline completo LAS=58.68% y UAS=62.43%  
✅ **CLAIM-021:** Concordancia gramatical 91.44% con modelo de sufijos

---

## 📈 Comparación con Estado del Arte

| Sistema | UAS | LAS | Método |
|---------|-----|-----|--------|
| **SYNTAX-003 (nuestro)** | **62.43%** | **58.68%** | HDC + item memory |
| MaltParser | ~85% | ~80% | ML tradicional |
| UDPipe (stanza) | ~95% | ~93% | Deep learning |
| SpaCy es_core | ~94% | ~92% | Deep learning |

**Brecha:** ~25-30 puntos por debajo de sistemas profesionales.

**Causas:**
- Sin embeddings densos
- Sin modelo de secuencia (RNN/Transformer)
- Sin features de grafos
- Sin entrenamiento end-to-end

---

## 💡 Conclusión General

### Estado final de la Fase 2

**Fase 2 COMPLETADA CON ÉXITO.**

Los 4 experimentos alcanzaron o superaron sus objetivos:
- UAS: 62.43% ✓ (objetivo 60-70%)
- LAS: 58.68% ✓ (objetivo 55-65%)
- Concordancia: 91.44% ✓ (objetivo >90%)
- Clasificación: 87.59% ✓ (objetivo 60-75%)

### Estado acumulado del proyecto

| Fase | Experimentos | Afirmaciones | Estado |
|------|-------------|--------------|--------|
| Fase 0: Validación HDC | 3 | CLAIM-001 a 007 | ✅ |
| Fase 1: Morfología | 5 | CLAIM-008 a 017 | ✅ |
| **Fase 2: Sintaxis** | **4** | **CLAIM-018 a 021** | **✅** |
| Fase 3: Semántica | — | — | ⏳ Pendiente |
| Fase 4: Discurso | — | — | ⏳ Pendiente |

**Total:** 12 experimentos, 21 afirmaciones científicas validadas.

---

## 📁 Entregables de la Fase 2

**Documentación:**
- ✅ `docs/fase_2_resumen_ejecutivo.md` (este documento)
- ✅ `docs/experimentos/syntax_001_resultados.md`
- ✅ `docs/experimentos/syntax_002_resultados.md`
- ✅ `docs/experimentos/syntax_003_resultados.md`
- ✅ `docs/experimentos/syntax_004_resultados.md`

**Código:**
- ✅ `experiments/syntax_001/` (análisis de dependencias)
- ✅ `experiments/syntax_002/` (clasificación de relaciones)
- ✅ `experiments/syntax_003/` (pipeline completo LAS)
- ✅ `experiments/syntax_004/` (concordancia gramatical)

**Datos:**
- ✅ `experiments/syntax_001/syntax_001_results.json`
- ✅ `experiments/syntax_002/syntax_002_results.json`
- ✅ `experiments/syntax_003/syntax_003_results.json`
- ✅ `experiments/syntax_004/syntax_004_results.json`

---

**Firma:** Agente 1 — Director Científico y Orquestador  
**Fecha:** 2026-10-17  
**Próxima fase:** Fase 3 — Semántica
