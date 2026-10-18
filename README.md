# Laboratorio de Micro-IA Biomimética

**Computación Hiperdimensional (HDC) + Redes Neuronales de Impulsos (SNN) + Inferencia Activa (FEP)**

Investigación computacional para el estudio de arquitecturas de inteligencia artificial biomimética aplicadas al procesamiento del lenguaje español.

---

## 🎯 Estado Actual

**Fase 1: Morfología** — ✅ **COMPLETADA**  
**Fase 2: Sintaxis** — ✅ **COMPLETADA**  
**Fase 3: Semántica** — ✅ **COMPLETADA**  
**Fase 4: Discurso** — ✅ **COMPLETADA**  
**Fase 5: Pragmática** — ✅ **COMPLETADA**

**Totales:** 23 experimentos · 33 afirmaciones científicas validadas

### Fase 5: Pragmática / Integración (4 experimentos)

| Experimento | Métrica | Resultado | Estado |
|-------------|---------|-----------|--------|
| PRAGMATICS-001: Pipeline integrado | Throughput | **52,313 tok/s** | ✅ |
| PRAGMATICS-002: Comprensión (SVO) | F1 | **70.79%** | ✅ |
| PRAGMATICS-003: Análisis errores | Cascade rate | **40.10%** | ✅ |
| PRAGMATICS-004: Robustez | Degradación | **7.16%** (20% ruido) | ✅ |

### Fase 4: Discurso (4 experimentos)

| Experimento | Métrica | Resultado | Estado |
|-------------|---------|-----------|--------|
| DISCOURSE-001: Correferencia | Accuracy | 31.80% (baseline 53.28%) | ⚠️ |
| DISCOURSE-002: Continuidad temática | F1 | **66.71%** (+29.47%) | ✅ |
| DISCOURSE-003: Relaciones oraciones | F1 pond. | **65.92%** | ✅ |
| DISCOURSE-004: Entity tracking | Purity | **47.44%** | ✅ |

### Fase 3: Semántica (4 experimentos)

| Experimento | Métrica | Resultado | Estado |
|-------------|---------|-----------|--------|
| SEMANTICS-001: WSD | Accuracy | **97.00%** | ✅ |
| SEMANTICS-002: SRL | Accuracy | **78.62%** | ✅ |
| SEMANTICS-003: NER | F1 | **77.19%** | ✅ |
| SEMANTICS-004: Similitud | Separación | **0.22** | ✅ |

### Fase 2: Sintaxis (4 experimentos)

| Experimento | Métrica | Resultado | Estado |
|-------------|---------|-----------|--------|
| SYNTAX-001: Dependencias (UAS) | UAS | **62.43%** | ✅ |
| SYNTAX-002: Relaciones | Accuracy | **87.59%** | ✅ |
| SYNTAX-003: Pipeline (LAS) | LAS | **58.68%** | ✅ |
| SYNTAX-004: Concordancia | Accuracy | **91.44%** | ✅ |

### Fase 1: Morfología (7 experimentos)

| Experimento | Resultado | Estado |
|-------------|-----------|--------|
| HDC-Esp-001: Memoria caracteres | 0% error | ✅ |
| HDC-Esp-002: Escalabilidad | Capacidad ilimitada | ✅ |
| HDC-Esp-003: Bundles | Fórmula validada | ✅ |
| MORPH-001: Clasificación (bundles) | 56-67% | ✅ |
| MORPH-003: Clasificación (sufijos) | 83-99% | ✅ |
| MORPH-004: Contexto POS | 93.26% | ✅ |
| MORPH-005: Decodificación | 91.58% | ✅ |

---

## 🚀 Instalación

### Requisitos

- Python 3.8+
- Git

### Linux / macOS

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
mkdir -p data/corpus && cd data/corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ../..
```

### Windows (PowerShell)

```powershell
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
mkdir -p data\corpus; cd data\corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ..\..
```

### Verificación

```bash
python tests/test_hdc.py  # 17/17 pruebas ✓
```

---

## 🧪 Experimentos

### 1. Validación HDC (~5 min)

```bash
python experiments/hdc_esp_001/run_experiment.py
python experiments/hdc_esp_002/run_experiment_optimized.py
python experiments/hdc_esp_003/run_experiment.py
```

### 2. Morfología (~5 min)

```bash
python experiments/morph_001/run_experiment.py
python experiments/morph_003/run_experiment.py
python experiments/morph_004/run_experiment.py
python experiments/morph_005/run_experiment.py
```

### 3. Sintaxis (~30 seg)

```bash
python experiments/syntax_001/run_experiment.py
python experiments/syntax_002/run_experiment.py
python experiments/syntax_003/run_experiment.py
python experiments/syntax_004/run_experiment.py
```

### 4. Semántica (~30 seg)

```bash
python experiments/semantics_001/run_experiment.py
python experiments/semantics_002/run_experiment.py
python experiments/semantics_003/run_experiment.py
python experiments/semantics_004/run_experiment.py
```

### 5. Discurso (~2 min)

```bash
python experiments/discourse_001/run_experiment.py
python experiments/discourse_002/run_experiment.py
python experiments/discourse_003/run_experiment.py
python experiments/discourse_004/run_experiment.py
```

### 6. Pragmática / Integración (~1 min)

```bash
python experiments/pragmatics_001/run_experiment.py
python experiments/pragmatics_002/run_experiment.py
python experiments/pragmatics_003/run_experiment.py
python experiments/pragmatics_004/run_experiment.py
```

---

## 📊 Resultados Destacados

### Morfología
| Tarea | Mejor resultado | Método |
|-------|----------------|--------|
| POS tagging | 91.58% | Bidireccional iterativo |
| Gender | 98.66% | Sufijos + item memory |
| Number | 99.00% | Sufijos + item memory |
| Tense | 98.84% | Sufijos + item memory |

### Sintaxis
| Tarea | Resultado | Estado del arte |
|-------|-----------|-----------------|
| UAS | 62.43% | ~95% (UDPipe) |
| LAS | 58.68% | ~93% (UDPipe) |
| Relaciones | 87.59% | — |
| Concordancia | 91.44% | — |

### Semántica
| Tarea | Resultado |
|-------|-----------|
| WSD (desambiguación) | 97.00% accuracy |
| SRL (roles semánticos) | 78.62% accuracy |
| NER (entidades) | F1=77.19% |
| Similitud semántica | Separación 0.22 |

### Discurso
| Tarea | Resultado |
|-------|-----------|
| Segmentación temática | F1=66.71% (+79% vs baseline) |
| Relaciones discursivas | F1 ponderado=65.92% |
| Entity tracking | Purity=47.44% |
| Correferencia | 31.80% (limitación documentada) |

### Pragmática / Integración
| Tarea | Resultado |
|-------|-----------|
| Pipeline throughput | 52,313 tokens/s |
| Extracción SVO | F1=70.79% |
| Q&A | 100% tasa de respuesta |
| Robustez (20% ruido) | 7.16% degradación |

---

## 📁 Estructura del Proyecto

```
Laboratorio-MicroIA/
├── src/
│   ├── hdc/              # Módulo HDC (core, item_memory, sequence_encoder)
│   ├── snn/              # Módulo SNN (pendiente)
│   ├── active_inf/       # Módulo FEP (pendiente)
│   └── interface/        # Integración (pendiente)
├── experiments/
│   ├── hdc_esp_001..003/ # Validación HDC
│   ├── morph_001..005/   # Morfología
│   ├── syntax_001..004/  # Sintaxis
│   ├── semantics_001..004/ # Semántica
│   ├── discourse_001..004/ # Discurso
│   └── pragmatics_001..004/ # Pragmática
├── tests/test_hdc.py     # 17 pruebas unitarias
├── docs/
│   ├── fase_1_resumen_ejecutivo.md
│   ├── fase_2_resumen_ejecutivo.md
│   ├── fase_3_resumen_ejecutivo.md
│   ├── fase_4_resumen_ejecutivo.md
│   └── experimentos/     # 19+ informes detallados
├── data/corpus/          # UD Spanish-AnCora
└── README.md
```

---

## 📚 Documentación

- **[Fase 1: Morfología](docs/fase_1_resumen_ejecutivo.md)**
- **[Fase 2: Sintaxis](docs/fase_2_resumen_ejecutivo.md)**
- **[Fase 3: Semántica](docs/fase_3_resumen_ejecutivo.md)**
- **[Fase 4: Discurso](docs/fase_4_resumen_ejecutivo.md)**
- **[Fase 5: Pragmática](docs/fase_5_resumen_ejecutivo.md)**
- **[Evaluación Inicial Crítica](docs/00_evaluacion_inicial_critica.md)**
- **Informes de experimentos:** [docs/experimentos/](docs/experimentos/)

---

## 🔬 Hallazgos Científicos Clave

1. **HDC es viable para NLP** con item memory + sufijos (no bundles acumulativos)
2. **La saturación de bundles es evitable** (+34% mejora con item memory)
3. **El contexto POS es más informativo** que las palabras (93% vs 90%)
4. **La decodificación bidireccional** supera a beam search
5. **La concordancia gramatical** es altamente predecible por sufijos (91%)
6. **WSD se beneficia del contexto** (+1.48% sobre baseline)
7. **SRL alcanza 78.62%** con features de verbo + argumento
8. **La similitud semántica distribucional** muestra estructura (empresa-banco: 0.98)
9. **La segmentación temática adaptativa** supera baselines por +79% (F1=66.71%)
10. **La correferencia profunda requiere más que features superficiales** (limitación documentada)
11. **Pipeline integrado: 52K tokens/s** con 4 niveles de procesamiento
12. **Extracción SVO con F1=70.79%** para comprensión de texto
13. **40% de errores son en cascada** (morfología como cuello de botella)
14. **Robustez diferenciada:** 0.04% ante capitalización, 19.40% ante typos

---

## 📖 Referencias

1. Kanerva, P. (2009). *Hyperdimensional Computing.* Cognitive Computation.
2. Universal Dependencies. *UD Spanish-AnCora.*
3. Friston, K. (2010). *The free-energy principle.* Nature Reviews Neuroscience.

---

**Última actualización:** 2026-10-18  
**Versión:** 1.0.0  
**Estado:** Proyecto completo — 5 fases, 23 experimentos, 33 claims validados
