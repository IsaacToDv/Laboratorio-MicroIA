# 🧬 Laboratorio de Micro-IA Biomimética

> Un laboratorio de investigación que explora si las arquitecturas de inteligencia artificial inspiradas en el cerebro pueden aprender las regularidades del idioma español.

**HDC** (Computación Hiperdimensional) · **SNN** (Redes Neuronales de Impulsos) · **FEP** (Principio de Energía Libre)

---

## ¿Qué es este proyecto?

Es un conjunto de **20 experimentos científicos** que implementan y evalúan modelos de IA biomimética para procesar el español, organizados en niveles lingüísticos progresivos:

```
 Texto en español
       ↓
 ┌─────────────────┐
 │   Morfología    │  ¿Qué categoría es cada palabra? (POS tagging)
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │   Semántica     │  ¿Qué significado tiene? (roles, entidades, desambiguación)
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │    Discurso     │  ¿Cómo se conectan las oraciones? (segmentación, tracking)
 └────────┬────────┘
          ↓
 ┌─────────────────┐
 │  SNN / FEP      │  ¿Puede un cerebro artificial hacer lo mismo?
 └─────────────────┘
```

A diferencia de los modelos de lenguaje modernos (GPT, BERT), este proyecto usa **arquitecturas ligeras** inspiradas en neurociencia:
- **HDC**: vectores de 10,000 dimensiones donde el significado emerge de operaciones algebraicas
- **SNN**: neuronas que se comunican con impulsos discretos (spikes), como las del cerebro
- **FEP**: el sistema elige la interpretación que minimiza su "sorpresa" interna

---

## ¿Qué puede hacer?

### 1. Etiquetar palabras por categoría gramatical

```python
# El modelo analiza "El gato duerme tranquilamente"
# y asigna: DET NOUN VERB ADV
```
**Resultado:** 93.26% de accuracy con contexto bidireccional · 92.78% con ensemble de 3 modelos

### 2. Desambiguar significados según el contexto

```python
# "banco" → ¿institución financiera o asiento?
# El modelo mira las palabras alrededor para decidir
```
**Resultado:** 97.00% de accuracy en desambiguación de sentido (WSD)

### 3. Identificar quién hace qué a quién

```python
# "María firmó el acuerdo en Caracas"
# → Sujeto: María | Verbo: firmar | Objeto: acuerdo | Lugar: Caracas
```
**Resultado:** F1 = 70.79% en extracción de tripletas Sujeto-Verbo-Objeto

### 4. Detectar cambios de tema en un texto

```python
# Párrafo 1: habla de economía → Párrafo 2: habla de deportes
# El modelo detecta el punto exacto donde cambia el tema
```
**Resultado:** F1 = 66.71% (+79% mejor que baselines de segmentación fija)

### 5. Rastrear entidades a lo largo del texto

```python
# "El presidente llegó. Fernández saludó. Él habló."
# → Las tres menciones se vinculan a la misma entidad
```
**Resultado:** 47.44% de menciones correctamente agrupadas en clusters

### 6. Reconocer entidades nombradas

```python
# "Google abrió oficinas en Madrid"
# → Google: ORG | Madrid: LOC
```
**Resultado:** F1 = 77.19% (Precision 100%, Recall 62.86%)

### 7. Minimizar energía libre para decidir

```python
# Modelo FEP: ante "la" ambigua, calcula cuál interpretación
# genera menos "sorpresa" dado el contexto
```
**Resultado:** 76.73% accuracy con Active Inference

---

## Instalación rápida

```bash
# 1. Clonar el repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# 2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# 3. Instalar dependencias
pip install numpy scipy conllu

# 4. Descargar el corpus en español
mkdir -p data/corpus && cd data/corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ../..

# 5. Ejecutar un experimento
python experiments/semantics_001/run_experiment.py
```

### Ejecutar todos los experimentos

```bash
# Validación HDC (3 experimentos)
for i in 001 002 003; do python experiments/hdc_esp_$i/run_experiment.py; done

# Morfología (4 experimentos)
for i in 001 003 004 005; do python experiments/morph_$i/run_experiment.py; done

# Semántica (4 experimentos)
for i in 001 002 003 004; do python experiments/semantics_$i/run_experiment.py; done

# Discurso (4 experimentos)
for i in 001 002 003 004; do python experiments/discourse_$i/run_experiment.py; done

# Cierre SNN-FEP (4 experimentos)
for i in 001 002 003 004; do python experiments/final_$i/run_experiment.py; done
```

> ⏱️ **Tiempo total estimado:** ~5 minutos en un laptop moderno.

---

## Resultados clave

| Experimento | Tarea | Métrica | Resultado |
|-------------|-------|---------|-----------|
| MORPH-004 | POS tagging (contexto) | Accuracy | **93.26%** |
| MORPH-005 | Decodificación morfológica | Accuracy | **91.58%** |
| MORPH-003 | POS por sufijos | Accuracy | **83–99%** |
| SEMANTICS-001 | Desambiguación (WSD) | Accuracy | **97.00%** |
| SEMANTICS-002 | Roles semánticos (SRL) | Accuracy | **78.62%** |
| SEMANTICS-003 | Entidades nombradas (NER) | F1 | **77.19%** |
| DISCOURSE-002 | Segmentación temática | F1 | **66.71%** |
| DISCOURSE-003 | Relaciones discursivas | F1 pond. | **65.92%** |
| DISCOURSE-004 | Entity tracking | Purity | **47.44%** |
| FINAL-002 | FEP (Active Inference) | Accuracy | **76.73%** |
| FINAL-003 | Ensemble (3 modelos) | Accuracy | **92.78%** |

### Comparación con estado del arte

| Tarea | Este proyecto | Estado del arte | Gap |
|-------|--------------|-----------------|-----|
| POS tagging | 93.26% | ~97% (Stanza) | ~4 pts |
| NER | F1=77.19% | ~90% (Stanza) | ~13 pts |
| WSD | 97.00% | ~95% (BERT-based) | comparable |

---

## Limitaciones honestas

Este proyecto es **investigación exploratoria**, no un producto listo para producción.

### 🔴 Lo que NO funciona bien
- **Correferencia profunda** (31.80%): resolver "él" → "Fernández" requiere comprensión semántica que el modelo no tiene
- **SNN con STDP** (17.93%): la red de impulsos no converge con entrenamiento corto; necesita mucho más tiempo o mejor implementación
- **Palabras funcionales** (ADP, DET, PUNCT): el modelo basado en sufijos las confunde con sustantivos porque no tienen sufijos característicos

### 🟡 Lo que funciona parcialmente
- **Entity tracking**: agrupa el 47% de las menciones correctamente
- **Relaciones discursivas**: detecta bien conectores explícitos (*porque*, *pero*, *además*) pero falla en relaciones implícitas
- **Robustez a typos**: degradación del 19% cuando hay errores tipográficos

### 🟢 Lo que funciona bien
- **Desambiguación léxica** (97%): el contexto local es muy informativo
- **POS tagging** (93%): sufijos + ensemble dan resultados competitivos
- **Segmentación temática** (F1=67%): umbral adaptativo supera baselines por +79%
- **Velocidad**: el pipeline procesa 52,000+ tokens/segundo en CPU

### Lo que NO incluye
- ❌ No es un modelo de lenguaje generativo (no genera texto)
- ❌ No usa GPU ni training profundo
- ❌ No tiene embeddings pre-entrenados (todo se aprende del corpus)
- ❌ No es multilingüe (solo español)

---

## Estructura del proyecto

```
Laboratorio-MicroIA/
│
├── src/                        # Código fuente de los módulos
│   └── hdc/                    #   Computación Hiperdimensional
│
├── experiments/                # Experimentos ejecutables (20)
│   ├── hdc_esp_001..003/       #   Validación de HDC
│   ├── morph_001..005/         #   Morfología
│   ├── semantics_001..004/     #   Semántica
│   ├── discourse_001..004/     #   Discurso
│   └── final_001..004/         #   Cierre SNN-FEP
│
├── docs/                       # Documentación científica
│   ├── fase_*_resumen_*.md     #   Resúmenes ejecutivos por fase
│   └── experimentos/           #   Reportes detallados por experimento
│
├── tests/                      # Pruebas unitarias
│   └── test_hdc.py             #   17 pruebas de HDC
│
└── data/corpus/                # Corpus UD Spanish-AnCora (descargable)
```

Cada experimento tiene:
- `run_experiment.py` — Script ejecutable con `python`
- `*_results.json` — Resultados cuantitativos reproducibles
- Un reporte en `docs/experimentos/` con análisis detallado

---

## Documentación

| Documento | Contenido |
|-----------|-----------|
| [Fase 1: Morfología](docs/fase_1_resumen_ejecutivo.md) | HDC + POS tagging + lematización |
| [Fase 3: Semántica](docs/fase_3_resumen_ejecutivo.md) | WSD + SRL + NER + espacio semántico |
| [Fase 4: Discurso](docs/fase_4_resumen_ejecutivo.md) | Correferencia + segmentación + tracking |
| [Fase 6: Cierre](docs/fase_6_resumen_ejecutivo.md) | SNN + FEP + benchmarking + síntesis |
| [Síntesis final](experiments/final_004/sintesis_final.md) | Documento consolidado del proyecto |
| [Reportes individuales](docs/experimentos/) | 19 informes de experimentos |

---

## Hallazgos principales

1. **HDC funciona para NLP** — pero solo con *item memory* (los bundles acumulativos se saturan)
2. **Los sufijos del español son muy informativos** — predicen POS con 83-99% de accuracy
3. **El contexto bidireccional** (palabra anterior + siguiente) mejora POS tagging en +6%
4. **Los ensembles ganan** — la votación de 3 modelos simples (92.78%) supera a cada uno individual
5. **La energía libre minimiza sorpresa** — el FEP de Friston aplicado a desambiguación logra 76.73%
6. **La segmentación temática emerge** de medir similitud léxica entre oraciones consecutivas
7. **Las SNN necesitan más desarrollo** — el paradigma es prometedor pero STDP simple no basta
8. **40% de los errores se propagan en cascada** — un error morfológico genera errores sintácticos y semánticos

---

## Corpus

Todos los experimentos usan **[UD Spanish-AnCora](https://universaldependencies.org/treebanks/es_ancora/)** de Universal Dependencies:
- ~500,000 tokens etiquetados
- 14 categorías POS
- Dependencias sintácticas
- Features morfológicas (género, número, tiempo, modo)

---

## Referencias

- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation.*
- Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience.
- Maass, W. (1997). *Networks of spiking neurons: The third generation of neural network models.*
- Universal Dependencies Consortium. *UD Spanish-AnCora.*

---

**Licencia:** Ver archivo LICENSE en el repositorio.  
**Corpus:** UD Spanish-AnCora bajo licencia CC BY-SA 4.0.  
**Versión:** 1.0.0 · Octubre 2026
