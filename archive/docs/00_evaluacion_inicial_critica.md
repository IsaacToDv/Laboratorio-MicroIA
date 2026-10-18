# Evaluación Inicial Crítica — Sistema Orquestador Científico
## Micro-IA Biomimética HDC–SNN–FEP

**Fecha:** 2026-10-17  
**Responsable:** Agente 1 — Director Científico y Orquestador  
**Estado:** Evaluación inicial previa a implementación

---

## 1. Objetivo de esta evaluación

Identificar supuestos no demostrados, riesgos conceptuales y el experimento mínimo viable antes de implementar la arquitectura completa.

---

## 2. Supuestos no demostrados de la arquitectura propuesta

### 2.1. Supuestos sobre computación hiperdimensional (HDC/VSA)

**Supuesto crítico:** Las operaciones HDC pueden codificar y recuperar asociaciones lingüísticas con tasa de error aceptable.

**Evidencia disponible:**
- Kanerva (2009) demuestra propiedades matemáticas de espacios hiperdimensionales
- Capacidad teórica de memoria asociativa escala con la dimensión D
- Operaciones de binding/bundling están bien definidas algebraicamente

**Lagunas identificadas:**
- No existe evidencia de que HDC por sí solo pueda capturar dependencias temporales complejas del lenguaje
- La recuperación de secuencias largas puede sufrir interferencia catastrófica
- El coste de codificación de características lingüísticas complejas no está cuantificado para español
- La dimensión D=10.000 es arbitraria; no hay justificación teórica para español específicamente

**Riesgo:** HDC puede ser insuficiente para modelar sintaxis y semántica sin mecanismos adicionales costosos.

---

### 2.2. Supuestos sobre redes neuronales de impulsos (SNN)

**Supuesto crítico:** STDP puede aprender dependencias temporales relevantes para lenguaje.

**Evidencia disponible:**
- STDP demuestra aprendizaje no supervisado de patrones temporales en visión y audio
- Modelos LIF son computacionalmente eficientes

**Lagunas identificadas:**
- STDP no supervisada no ha demostrado aprender estructura lingüística (morfología, sintaxis)
- La codificación de texto como secuencias de spikes es un problema abierto
- No hay evidencia de que SNN pequeñas (< 10k neuronas) puedan modelar lenguaje natural
- El periodo refractario y constantes temporales pueden ser incompatibles con la escala temporal del lenguaje

**Riesgo:** SNN puede añadir complejidad sin beneficio medible frente a RNN/Transformer pequeños.

---

### 2.3. Supuestos sobre inferencia activa (FEP)

**Supuesto crítico:** Un modelo generativo con selección activa de ejemplos mejora el aprendizaje incremental.

**Evidencia disponible:**
- Framework teórico de Friston (2010) está bien formalizado
- Active inference ha mostrado beneficios en robótica y control

**Lagunas identificadas:**
- Aplicación a adquisición de lenguaje es especulativa (E5)
- El coste computacional de mantener distribuciones de creencias puede superar beneficios
- No está claro qué constituye una "acción" en aprendizaje de lenguaje pasivo
- La energía libre esperada requiere un modelo generativo que ya conozca la distribución objetivo

**Riesgo:** FEP puede ser una capa de complejidad que no aporta valor medible frente a muestreo aleatorio o currículo predefinido.

---

### 2.4. Supuestos sobre integración HDC–SNN–FEP

**Supuesto crítico:** La combinación de los tres componentes produce sinergia.

**Lagunas identificadas:**
- No existe literatura que integre estos tres componentes para lenguaje
- Las interfaces entre módulos no están definidas
- El coste de comunicación entre módulos puede dominar el cómputo total
- No hay garantía de que los gradientes o señales de error se propaguen correctamente

**Riesgo:** La integración puede ser más costosa que la suma de las partes.

---

## 3. Hipótesis nulas explícitas

Para cada hipótesis del protocolo, establezco la hipótesis nula correspondiente:

- **H0_1:** HDC no codifica asociaciones lingüísticas con tasa de error < 20% para las tareas evaluadas
- **H0_2:** SNN con STDP no mejora predicción de secuencias frente a red sin plasticidad
- **H0_3:** Inferencia activa no mejora adquisición de información frente a muestreo aleatorio
- **H0_4:** La arquitectura híbrida no supera a un GRU pequeño con mismo presupuesto de parámetros
- **H0_5:** El coste computacional total supera cualquier beneficio en métricas de aprendizaje

---

## 4. Primer experimento mínimo viable

### 4.1. Justificación

Antes de implementar la arquitectura completa, debemos validar el componente más fundamental: **HDC para representación de lenguaje**.

Si HDC no puede codificar y recuperar asociaciones básicas de español con tasa de error aceptable, la arquitectura completa carece de fundamento.

### 4.2. Definición del experimento

**Nombre:** HDC-Esp-001 — Memoria asociativa para pares carácter–hipervector

**Hipótesis:**
- H1: Un espacio HDC bipolar de dimensión D puede almacenar y recuperar N asociaciones carácter→hipervector con tasa de error < 5% cuando N < capacidad teórica

**Variables:**
- Independiente: dimensión D ∈ {1024, 2048, 4096, 8192, 10000}
- Dependiente: tasa de error de recuperación, tiempo de codificación, memoria utilizada
- Control: conjunto de caracteres del español (letras, acentos, puntuación básica)

**Procedimiento:**
1. Generar hipervectores aleatorios bipolares para cada carácter del español
2. Crear un item memory (memoria de ítems)
3. Codificar 1000 secuencias de 10 caracteres cada una
4. Recuperar cada carácter por similitud coseno
5. Medir tasa de error, colisiones y tiempo

**Criterio de éxito:**
- Tasa de error < 5% para D ≥ 4096
- Tiempo de recuperación < 1ms por carácter en CPU estándar
- Memoria < 100 MB para el item memory completo

**Criterio de refutación:**
- Tasa de error > 20% incluso con D = 10000 indica que HDC no es viable para esta tarea

### 4.3. Entregables

1. Implementación de operaciones HDC básicas (binding, bundling, similitud)
2. Item memory para caracteres del español
3. Pruebas unitarias de recuperación
4. Benchmark de dimensiones
5. Informe con resultados y análisis

---

## 5. Estructura del proyecto

```
Laboratorio-MicroIA/
├── docs/
│   ├── 00_evaluacion_inicial_critica.md (este documento)
│   ├── 01_protocolo_investigacion.md
│   ├── 02_registro_decisiones.md
│   └── experimentos/
├── src/
│   ├── hdc/           # Módulo A — Computación hiperdimensional
│   ├── snn/           # Módulo B — Redes neuronales de impulsos
│   ├── active_inf/    # Módulo C — Inferencia activa
│   ├── interface/     # Módulo D — Integración
│   └── utils/
├── experiments/
│   ├── hdc_esp_001/   # Primer experimento
│   └── baselines/
├── data/
│   ├── corpus/
│   └── vocabularies/
├── tests/
├── notebooks/
├── requirements.txt
└── README.md
```

---

## 6. Decisiones iniciales

### 6.1. Lenguaje de implementación

**Decisión:** Python 3.10+

**Justificación:**
- Ecosistema científico maduro (NumPy, SciPy)
- Bibliotecas de ML disponibles (PyTorch para SNN si es necesario)
- Facilita reproducibilidad
- Compatible con herramientas de profiling

**Alternativas descartadas:**
- Julia: menor ecosistema, aunque mejor performance numérica
- C++: mayor complejidad de desarrollo, dificulta experimentación rápida

### 6.2. Representación HDC inicial

**Decisión:** Vectores bipolares {-1, +1}^D

**Justificación:**
- Operaciones algebraicas bien definidas
- Binding por producto componente a componente es simple
- Similitud coseno equivalente a producto punto normalizado
- Menor coste de almacenamiento que vectores reales

**Alternativas consideradas:**
- Vectores binarios {0, 1}^D: requiere operaciones diferentes
- Vectores complejos: mayor expresividad pero mayor coste

### 6.3. Primer conjunto de datos

**Decisión:** Generar datos sintéticos para HDC-Esp-001

**Justificación:**
- Experimento de validación de mecanismo, no de aprendizaje lingüístico
- Control total sobre variables
- Permite detectar errores de implementación antes de usar corpus real

**Próximo paso:** UD Spanish-AnCora cuando se valide HDC básico

---

## 7. Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| HDC no escala a secuencias largas | Alta | Alto | Probar con n-gramas y contexto limitado primero |
| SNN no aprende estructura lingüística | Alta | Medio | Implementar baseline GRU para comparación |
| FEP añade complejidad sin beneficio | Media | Medio | Ablación: comparar con/sin FEP |
| Integración de módulos es inviable | Media | Alto | Interfaces bien definidas desde el inicio |
| Coste computacional excesivo | Media | Medio | Profiling continuo, optimización incremental |

---

## 8. Siguiente acción inmediata

**Acción:** Implementar HDC-Esp-001

**Entregables concretos:**
1. `src/hdc/core.py` — Operaciones HDC básicas
2. `src/hdc/item_memory.py` — Memoria de ítems
3. `experiments/hdc_esp_001/run_experiment.py` — Script del experimento
4. `tests/test_hdc.py` — Pruebas unitarias
5. `docs/experimentos/hdc_esp_001_resultados.md` — Informe de resultados

**Tiempo estimado:** 2-3 horas de implementación + 1 hora de experimentación

**Criterio de continuación:**
- Si tasa de error < 5%: proceder a codificación de secuencias (Fase 0 del currículo)
- Si tasa de error 5-20%: investigar causas, ajustar dimensión o representación
- Si tasa de error > 20%: reconsiderar viabilidad de HDC para esta tarea

---

## 9. Registro de memoria científica

### Afirmación inicial #001

```json
{
  "id": "CLAIM-001",
  "claim": "La arquitectura HDC-SNN-FEP puede aprender regularidades del español",
  "status": "pendiente",
  "evidence_type": "E5",
  "source": "Hipótesis del protocolo, sin evidencia empírica previa",
  "conditions": "Arquitectura no implementada, sin datos experimentales",
  "method": "N/A",
  "metrics": "N/A",
  "limitations": "Afirmación especulativa que requiere validación experimental",
  "contradictions": "Ninguna identificada aún",
  "next_test": "HDC-Esp-001: validar capacidad de HDC para representar caracteres del español"
}
```

---

## 10. Conclusión

La arquitectura propuesta es **teóricamente plausible pero experimentalmente no validada**. Los tres componentes (HDC, SNN, FEP) tienen fundamentos matemáticos sólidos, pero su aplicación conjunta a aprendizaje de lenguaje es especulativa.

El protocolo de investigación es riguroso y apropiado. La estrategia correcta es:

1. **Validar cada componente por separado** antes de integrarlos
2. **Establecer líneas base claras** (n-gramas, GRU, Transformer pequeño)
3. **Comenzar con el componente más fundamental** (HDC para representación)
4. **Rechazar componentes que no aporten valor medible**

El experimento HDC-Esp-001 es el punto de partida apropiado: es pequeño, verificable y fundamental para toda la arquitectura.

---

**Firma:** Agente 1 — Director Científico y Orquestador  
**Próxima revisión:** Después de completar HDC-Esp-001
