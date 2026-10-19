# 🧬 Laboratorio de Micro-IA Biomimética

**Generador de texto en español con identidades configurables, entrenado con literatura clásica y datasets multi-disciplina.**

---

## ✨ Características principales

- 🧠 **Inspirado en el cerebro**: Usa memoria asociativa (hipocampo), predicción bayesiana (Free Energy Principle) y mecanismos de fatiga neuronal
- 🎭 **Identidades configurables**: Poeta, Filósofo, Narrador, Ensayista, Contemporáneo
- 📚 **Literatura clásica**: Entrenado con textos de Cervantes, Borges, García Márquez, Neruda, Unamuno, y más
- 🤗 **Datasets de Hugging Face**: Wikipedia, noticias, textos legales, científicos y literatura
- ⚡ **Sin GPU**: Funciona en cualquier laptop moderna
- 🎯 **Fácil de usar**: Instala, descarga datos y genera texto en 5 minutos

---

## 🎭 Identidades de IA

El generador puede adoptar diferentes **personalidades** inspiradas en grandes autores:

### 🎭 Poeta
**Estilo:** Lírico y emocional  
**Inspiración:** Pablo Neruda, Federico García Lorca, Octavio Paz, César Vallejo  
**Tono:** Melancólico y apasionado

```python
from src.generation import TextGenerator, load_identity

gen = TextGenerator()
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

poeta = load_identity('poeta')
gen.set_identity(poeta)

print(gen.generate("El amor"))
# → "El amor es como el agua que corre entre los dedos del tiempo"
```

### 🤔 Filósofo
**Estilo:** Reflexivo y profundo  
**Inspiración:** Miguel de Unamuno, José Ortega y Gasset, María Zambrano  
**Tono:** Reflexivo y cuestionador

```python
filosofo = load_identity('filosofo')
gen.set_identity(filosofo)

print(gen.generate("La existencia"))
# → "La existencia no es sino la conciencia de ser en el tiempo"
```

### 📖 Narrador
**Estilo:** Rico en detalles, realismo mágico  
**Inspiración:** Gabriel García Márquez, Jorge Luis Borges, Julio Cortázar  
**Tono:** Nostálgico y vívido

```python
narrador = load_identity('narrador')
gen.set_identity(narrador)

print(gen.generate("Muchos años después"))
# → "Muchos años después, el coronel habría de recordar aquella tarde remota"
```

### 📝 Ensayista
**Estilo:** Analítico y culto  
**Inspiración:** Jorge Luis Borges, Octavio Paz, Mario Vargas Llosa  
**Tono:** Analítico y erudito

### 🏙️ Contemporáneo
**Estilo:** Moderno y accesible  
**Inspiración:** Roberto Bolaño, Javier Marías, Almudena Grandes  
**Tono:** Directo y observador

---

## 🚀 Inicio rápido

### 1. Instalación (2 minutos)

```bash
# Clonar repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Crear entorno virtual (RECOMENDADO)
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

**⚠️ ¿Error "externally-managed-environment"?**  
Esto es normal en sistemas Linux modernos (Arch, Fedora, Ubuntu 23.04+). **Usa el entorno virtual** como se muestra arriba.  
Ver [FAQ](FAQ.md#error-externally-managed-environment) para más detalles.

**Opción alternativa: Script automático**
```bash
# Instala todo automáticamente (crea venv y descarga datos)
bash install.sh
```

### 2. Descargar datasets (3 minutos)

**Opción A: Solo corpus UD AnCora (básico)**
```bash
bash scripts/setup_corpus.sh
```

**Opción B: Datasets multi-disciplina (recomendado)**
```bash
python scripts/download_datasets.py
```

**Opción C: Literatura clásica (completo)**
```bash
python scripts/download_literature.py
```

Esto descarga:
- 📖 Wikipedia en español
- 📰 Noticias en español
- ⚖️ Textos legales
- 🔬 Textos científicos
- 📚 Literatura clásica (poesía, filosofía, narrativa)

### 3. Entrenar generador (1 minuto)

```bash
# Entrenamiento básico
python scripts/train_final.py

# Entrenamiento con literatura
python scripts/train_with_literature.py
```

### 4. Generar texto

**Demo interactivo:**
```bash
python demo.py
```

**Ejemplos con identidades:**
```bash
python examples/identidades.py
```

**Ejemplos por disciplina:**
```bash
python examples/generacion_por_disciplina.py
```

**Uso en código Python:**
```python
from src.generation import TextGenerator, load_identity

# Cargar generador entrenado
gen = TextGenerator()
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

# Configurar identidad
poeta = load_identity('poeta')
gen.set_identity(poeta)

# Generar texto
texto = gen.generate("El amor", max_words=20)
print(texto)
```

---

## 📚 Literatura clásica incluida

El generador se entrena con textos de grandes autores hispanos:

### Poesía
- **Pablo Neruda** - Versos melancólicos y apasionados
- **Federico García Lorca** - Poesía surrealista y popular
- **Octavio Paz** - Poesía filosófica y reflexiva
- **César Vallejo** - Poesía social y existencial
- **Antonio Machado** - Poesía del tiempo y el camino
- **Gustavo Adolfo Bécquer** - Rimas y leyendas

### Filosofía
- **Miguel de Unamuno** - Existencialismo español
- **José Ortega y Gasset** - Racionalismo vital
- **María Zambrano** - Filosofía poética

### Narrativa
- **Gabriel García Márquez** - Realismo mágico
- **Jorge Luis Borges** - Ficciones y laberintos
- **Julio Cortázar** - Rayuela y lo fantástico
- **Isabel Allende** - Saga familiar y memoria

### Ensayo
- **Octavio Paz** - El laberinto de la soledad
- **Mario Vargas Llosa** - La verdad de las mentiras

---

## 🎛️ Parámetros de generación

```python
gen.generate(
    prompt="El amor",        # Texto inicial
    max_words=20,            # Máximo de palabras a generar
    temperature=None         # Usa la temperatura de la identidad (o especifica 0.1-2.0)
)
```

| Identidad | Temperatura | Estilo |
|-----------|-------------|--------|
| Poeta | 1.2 | Creativo y lírico |
| Filósofo | 0.9 | Reflexivo y preciso |
| Narrador | 1.1 | Descriptivo y vívido |
| Ensayista | 0.8 | Analítico y culto |
| Contemporáneo | 1.0 | Moderno y accesible |

---

## 🏗️ Estructura del proyecto

```
Laboratorio-MicroIA/
├── src/                          # Código fuente
│   └── generation/               # Generador de texto
│       ├── text_generator.py     # Generador principal
│       ├── identity.py           # Sistema de identidades
│       └── __init__.py
│
├── scripts/                      # Scripts de utilidad
│   ├── setup_corpus.sh           # Descargar UD AnCora
│   ├── download_datasets.py      # Descargar datasets Hugging Face
│   ├── download_literature.py    # Descargar literatura clásica
│   ├── train_final.py            # Entrenar modelo básico
│   └── train_with_literature.py  # Entrenar con literatura
│
├── examples/                     # Ejemplos de uso
│   ├── identidades.py            # Demo con identidades
│   └── generacion_por_disciplina.py
│
├── archive/                      # Investigación (fases 1-7)
│   ├── experiments/              # 24 experimentos científicos
│   ├── docs/                     # Documentación técnica
│   └── tests/                    # Tests unitarios
│
├── models/                       # Modelos entrenados
├── data/                         # Datasets
│   ├── corpus/                   # UD AnCora
│   ├── datasets/                 # Hugging Face datasets
│   └── literature/               # Literatura clásica
│
├── demo.py                       # Demo interactivo
├── requirements.txt              # Dependencias
├── CHANGELOG.md                  # Historial de versiones
└── README.md                     # Este archivo
```

---

## 🔬 Base científica

Este proyecto implementa **7 fases de investigación** (ver `archive/`):

1. **Morfología**: POS tagging con HDC (93.26% accuracy)
2. **Sintaxis**: Parsing de dependencias (88.33% UAS)
3. **Semántica**: WSD, SRL, NER (97% WSD accuracy)
4. **Discurso**: Correferencia y segmentación (66.71% F1)
5. **Pragmática**: Pipeline integrado y Q&A (52K tokens/s)
6. **SNN-FEP**: Spiking neural networks y Free Energy Principle
7. **Generación**: Sistema final con memoria HDC + FEP + identidades

**Resultados clave:**
- 24 experimentos científicos completados
- 41 claims validados
- 5 identidades de IA configurables
- Literatura clásica de 15+ autores hispanos

---

## 📊 Datasets disponibles

| Dataset | Dominio | Oraciones | Fuente |
|---------|---------|-----------|--------|
| UD AnCora | Lingüística | 15,000 | Universal Dependencies |
| Wikipedia ES | Conocimiento general | 50,000+ | Hugging Face |
| Noticias ES | Actualidad | 20,000+ | MLSum |
| Legal ES | Jurídico | 10,000+ | LegalES |
| Científico ES | Investigación | 10,000+ | MLSum-Science |
| Literatura ES | Narrativa | 10,000+ | BookCorpus |
| **Literatura Clásica** | **Poesía, Filosofía, Narrativa** | **500+** | **Project Gutenberg + Manuales** |

**Total: ~115,000+ oraciones de entrenamiento**

---

## 📈 Rendimiento

| Métrica | Valor |
|---------|-------|
| Velocidad de generación | ~100 palabras/segundo |
| Memoria requerida | ~500 MB |
| Tiempo de entrenamiento | ~1 minuto |
| Tamaño del modelo | ~50 MB |

---

## 🛠️ Desarrollo

### Entrenar desde cero
```bash
# 1. Descargar todos los datasets
bash scripts/setup_corpus.sh
python scripts/download_datasets.py
python scripts/download_literature.py

# 2. Entrenar modelo con literatura
python scripts/train_with_literature.py

# 3. Probar con identidades
python examples/identidades.py
```

### Crear identidad personalizada
```python
from src.generation import AIIdentity, save_identity

mi_identidad = AIIdentity(
    name="Mi Estilo",
    description="Descripción de mi estilo",
    style="personalizado",
    temperature=1.0,
    top_authors=["Autor 1", "Autor 2"],
    vocabulary_bias={
        'palabra1': 2.0,
        'palabra2': 1.5
    },
    sentence_patterns=["Patrón 1", "Patrón 2"],
    emotional_tone="alegre"
)

save_identity(mi_identidad, "mi_identidad.json")
```

---

## 🤝 Contribuir

Este proyecto está **terminado y consolidado**, pero aceptamos:
- Nuevas identidades de IA
- Más textos literarios en dominio público
- Reportes de bugs
- Mejoras de documentación
- Optimizaciones de rendimiento

Ver [CHANGELOG.md](CHANGELOG.md) para historial completo.

---

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.

Los textos literarios incluidos son de **dominio público** o están bajo licencias compatibles.

---

## 📚 Referencias

1. Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation*
2. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience
3. Maass, W. (1997). *Networks of spiking neurons: the third generation of neural network models*
4. Universal Dependencies Consortium. *UD Spanish AnCora*

---

## 🙏 Agradecimientos

- **Universal Dependencies** por el corpus UD Spanish AnCora
- **Hugging Face** por los datasets en español
- **Project Gutenberg** por la literatura clásica en dominio público
- **Autores hispanos** cuya obra inspira este proyecto:
  - Miguel de Cervantes
  - Jorge Luis Borges
  - Gabriel García Márquez
  - Pablo Neruda
  - Federico García Lorca
  - Miguel de Unamuno
  - Y muchos más...

---

**Versión:** 4.0.0 · Enero 2025 · Con identidades de IA y literatura clásica

**Estado:** ✅ Listo para producción
