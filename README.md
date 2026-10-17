# Laboratorio de Micro-IA Biomimética

**Computación Hiperdimensional (HDC) + Redes Neuronales de Impulsos (SNN) + Inferencia Activa (FEP)**

Investigación computacional para el estudio de arquitecturas de inteligencia artificial biomimética aplicadas al procesamiento del lenguaje español.

---

## 🎯 Estado Actual

**Fase 1: Morfología** — ✅ **COMPLETADA**

| Experimento | Resultado | Estado |
|-------------|-----------|--------|
| HDC-Esp-001: Memoria caracteres | 0% error | ✅ |
| HDC-Esp-002: Escalabilidad | Capacidad ilimitada | ✅ |
| HDC-Esp-003: Bundles | Fórmula validada | ✅ |
| MORPH-001: Clasificación (bundles) | 56-67% (no competitivo) | ✅ |
| MORPH-003: Clasificación (sufijos) | 83-99% (competitivo) | ✅ |
| MORPH-004: Contexto POS | 93.26% (techo teórico) | ✅ |
| MORPH-005: Decodificación secuencial | 91.58% (bidireccional) | ✅ |

**Hallazgo principal:** HDC con item memory + sufijos + contexto POS + decodificación bidireccional alcanza **91.58% accuracy** en POS tagging real (sin POS gold), solo 1.68% por debajo del techo teórico de 93.26%.

---

## 🚀 Instalación

### Requisitos

- Python 3.8+
- Git
- pip

### Linux / macOS

```bash
# Clonar repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Crear entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Descargar corpus UD Spanish-AnCora
mkdir -p data/corpus
cd data/corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ../..
```

### Windows (PowerShell)

```powershell
# Clonar repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Crear entorno virtual (recomendado)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt

# Descargar corpus UD Spanish-AnCora
mkdir -p data\corpus
cd data\corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ..\..
```

### Verificación

```bash
# Ejecutar pruebas unitarias
python tests/test_hdc.py
```

**Resultado esperado:** 17/17 pruebas pasan ✓

---

## 🧪 Experimentos

### 1. Validación de HDC (rápido, ~5 min)

```bash
# Memoria para caracteres
python experiments/hdc_esp_001/run_experiment.py

# Escalabilidad a vocabularios grandes
python experiments/hdc_esp_002/run_experiment_optimized.py

# Capacidad de bundles
python experiments/hdc_esp_003/run_experiment.py
```

### 2. Clasificación Morfológica (medio, ~5 min)

```bash
# MORPH-001: Bundles acumulativos (no competitivo)
python experiments/morph_001/run_experiment.py

# MORPH-003: Sufijos explícitos (competitivo) ⭐
python experiments/morph_003/run_experiment.py
```

### 3. Contexto para POS Tagging (rápido, ~20 seg) ⭐

```bash
# MORPH-004: HDC con contexto POS
python experiments/morph_004/run_experiment.py
```

**Resultado esperado:** 93.26% accuracy con POS bilateral

### 4. Decodificación Secuencial (medio, ~12 seg) ⭐

```bash
# MORPH-005: Decodificación secuencial real
python experiments/morph_005/run_experiment.py
```

**Resultado esperado:** 91.58% accuracy con método bidireccional iterativo

---

## 📊 Resultados Clave

### MORPH-004: POS Tagging con Contexto

| Configuración | Accuracy | Mejora |
|---------------|----------|--------|
| Sin contexto | 86.81% | — |
| Palabra anterior | 90.98% | +4.17% |
| Palabra siguiente | 89.01% | +2.20% |
| **POS anterior** | **92.08%** | **+5.27%** |
| **POS bilateral** | **93.26%** | **+6.45%** ⭐ |

**Conclusión:** El POS de palabras vecinas es más informativo que las palabras mismas.

### MORPH-005: Decodificación Secuencial

| Método | Token Accuracy | Sentence Accuracy | Δ vs Gold |
|--------|----------------|-------------------|-----------|
| POS gold (techo) | 93.26% | 0.00% | — |
| Greedy decoding | 86.96% | 9.30% | -6.30% |
| Beam search k=3,5,10 | 86.96% | 9.30% | -6.30% |
| **Bidireccional iterativo** | **91.58%** | **19.87%** | **-1.68%** ⭐ |

**Hallazgos clave:**
- Beam search NO mejora sobre greedy (modelo muy peaked)
- Decodificación bidireccional recupera casi toda la brecha con el techo teórico
- Solo 1.68% de diferencia entre rendimiento real y teórico

### MORPH-003: Clasificación Morfológica

| Tarea | MORPH-001 (bundles) | MORPH-003 (sufijos) | Mejora |
|-------|---------------------|---------------------|--------|
| POS | 56.31% | **83.76%** | +27.45% |
| Gender | 67.03% | **98.66%** | +31.63% |
| Number | 55.65% | **99.00%** | +43.35% |
| Tense | 63.52% | **98.84%** | +35.32% |

**Mejora promedio:** +34.4 puntos porcentuales

---

## 📁 Estructura del Proyecto

```
Laboratorio-MicroIA/
├── src/
│   ├── hdc/              # Módulo HDC (implementado)
│   │   ├── core.py
│   │   ├── item_memory.py
│   │   └── sequence_encoder.py
│   ├── snn/              # Módulo SNN (pendiente)
│   ├── active_inf/       # Módulo FEP (pendiente)
│   └── interface/        # Integración (pendiente)
├── experiments/
│   ├── hdc_esp_001/      # Memoria caracteres
│   ├── hdc_esp_002/      # Escalabilidad
│   ├── hdc_esp_003/      # Capacidad bundles
│   ├── morph_001/        # Clasificación (bundles)
│   ├── morph_003/        # Clasificación (sufijos) ⭐
│   ├── morph_004/        # Contexto POS ⭐
│   └── morph_005/        # Decodificación secuencial ⭐
├── tests/
│   └── test_hdc.py       # 17 pruebas unitarias
├── docs/
│   ├── fase_1_resumen_ejecutivo.md
│   └── experimentos/     # Informes detallados
├── data/
│   └── corpus/           # UD Spanish-AnCora
├── requirements.txt
└── README.md
```

---

## 📚 Documentación

- **[Resumen Ejecutivo Fase 1](docs/fase_1_resumen_ejecutivo.md)** — Estado completo de la investigación
- **[Evaluación Inicial Crítica](docs/00_evaluacion_inicial_critica.md)** — Análisis de supuestos y riesgos
- **Informes de Experimentos:**
  - [HDC-Esp-001](docs/experimentos/hdc_esp_001_resultados.md) — Memoria para caracteres
  - [HDC-Esp-002](docs/experimentos/hdc_esp_002_resultados.md) — Escalabilidad
  - [HDC-Esp-003](docs/experimentos/hdc_esp_003_resultados.md) — Capacidad de bundles
  - [MORPH-001](docs/experimentos/morph_001_resultados.md) — Clasificación con bundles
  - [MORPH-003](docs/experimentos/morph_003_resultados.md) — Clasificación con sufijos
  - [MORPH-004](docs/experimentos/morph_004_resultados.md) — Contexto POS

---

## 🔬 Hallazgos Científicos

### Afirmaciones Validadas (E1 — Resultado reproducido)

1. **HDC es viable para morfología** con item memory + sufijos (83-99% accuracy)
2. **La saturación de bundles es evitable** usando item memory (+34.4% mejora)
3. **El contexto POS mejora POS tagging** en +6.30% (86.81% → 93.11%)
4. **POS de vecinos es más informativo** que palabras (93.11% vs 89.88%)
5. **La morfología del español es altamente regular** (81-99% con sufijos)

### Limitaciones Identificadas

1. **Capacidad de bundles limitada:** ~188 asociaciones para D=10000
2. **Sin procesamiento temporal:** Módulo SNN pendiente
3. **Sin inferencia activa:** Módulo FEP pendiente
4. **Contexto POS usa gold:** Necesita decodificación secuencial para uso real

---

## 🎯 Próximos Pasos

### Opción 1: MORPH-005 — Decodificación Secuencial
- Implementar decodificación izquierda-derecha
- Usar POS predicho (no gold) del contexto
- Beam search para mantener hipótesis
- **Objetivo:** 90-92% accuracy sin POS gold

### Opción 2: Fase 2 — Sintaxis
- Análisis de dependencias con HDC
- Integración con MORPH-004
- **Objetivo:** UAS/LAS competitivos

### Opción 3: Módulo B — SNN
- Implementar redes neuronales de impulsos
- Procesamiento temporal de secuencias
- Integración con HDC
- **Objetivo:** Mejorar POS tagging con dinámica temporal

### Opción 4: Módulo C — Inferencia Activa
- Modelo generativo probabilístico
- Selección activa de ejemplos
- Minimización de energía libre
- **Objetivo:** Aprendizaje más eficiente

---

## 🛠️ Desarrollo

### Ejecutar pruebas

```bash
python tests/test_hdc.py
```

### Analizar corpus

```bash
python experiments/morph_001/analyze_corpus.py
```

### Crear nuevo experimento

```bash
mkdir experiments/mi_experimento
# Crear run_experiment.py siguiendo el patrón de morph_004
```

---

## 📖 Referencias

1. Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation.
2. Universal Dependencies. *UD Spanish-AnCora.* https://universaldependencies.org/treebanks/es_ancora/
3. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.

---

## 📄 Licencia

Este proyecto es de código abierto y está disponible bajo los términos de la licencia del repositorio.

---

## 👥 Contacto

Para más información sobre este laboratorio, consulta la conversación original en Arena.ai.

---

**Última actualización:** 2026-10-17  
**Versión:** 0.2.0  
**Estado:** Fase 1 completada, listo para Fase 2
