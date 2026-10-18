# 🧬 Laboratorio de Micro-IA Biomimética

**Generador de texto en español inspirado en neurociencia, entrenado con datasets multi-disciplina.**

---

## ✨ Características principales

- 🧠 **Inspirado en el cerebro**: Usa memoria asociativa (hipocampo), predicción bayesiana (Free Energy Principle) y mecanismos de fatiga neuronal
- 🤗 **Datasets de Hugging Face**: Entrenado con Wikipedia, noticias, textos legales, científicos y literatura en español
- 📚 **Multi-disciplina**: Genera texto en Política, Economía, Ciencia, Tecnología y Cultura
- ⚡ **Sin GPU**: Funciona en cualquier laptop moderna
- 🎯 **Fácil de usar**: Instala, descarga datos y genera texto en 5 minutos

---

## 🎯 ¿Qué hace?

Dado un inicio de oración, el sistema **genera texto coherente** en español:

```bash
# Política
Tú escribes:  "El presidente"
IA genera:    "El presidente de la Comisión de el Mercado de el jamón curado"

# Economía
Tú escribes:  "La economía"
IA genera:    "La economía canaria en los últimos dos meses"

# Ciencia
Tú escribes:  "Los científicos"
IA genera:    "Los científicos encontraron evidencia de agua en Marte"

# Tecnología
Tú escribes:  "La inteligencia artificial"
IA genera:    "La inteligencia artificial revoluciona la medicina moderna"
```

---

## 🚀 Inicio rápido

### 1. Instalación (2 minutos)

```bash
# Clonar repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Descargar datasets (3 minutos)

**Opción A: Solo corpus UD AnCora (básico)**
```bash
bash scripts/setup_corpus.sh
```

**Opción B: Datasets multi-disciplina de Hugging Face (recomendado)**
```bash
python scripts/download_datasets.py
```

Esto descarga:
- 📖 Wikipedia en español (5,000 artículos)
- 📰 Noticias en español (2,000 artículos)
- ⚖️ Textos legales (1,000 documentos)
- 🔬 Textos científicos (1,000 papers)
- 📚 Literatura en español (1,000 textos)

### 3. Entrenar generador (1 minuto)

```bash
python scripts/train_final.py
```

Esto combina UD AnCora + datasets de Hugging Face y entrena el modelo.

### 4. Generar texto

**Demo interactivo:**
```bash
python demo.py
```

**Ejemplos por disciplina:**
```bash
python examples/generacion_por_disciplina.py
```

**Uso en código Python:**
```python
from src.generation import TextGenerator

# Cargar generador entrenado
gen = TextGenerator.load("models/trained_generator.pkl")

# Generar texto
texto = gen.generate("El presidente", max_words=20, temperature=0.7)
print(texto)
```

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

**Total: ~115,000 oraciones de entrenamiento**

---

## 🎛️ Parámetros de generación

```python
gen.generate(
    prompt="El presidente",  # Texto inicial
    max_words=20,            # Máximo de palabras a generar
    temperature=0.7          # Creatividad (0.1=conservador, 2.0=creativo)
)
```

| Temperatura | Estilo | Uso recomendado |
|-------------|--------|-----------------|
| 0.1 - 0.3 | Conservador | Texto formal, técnico |
| 0.5 - 0.7 | Balanceado | Uso general (recomendado) |
| 1.0 - 1.5 | Creativo | Narrativa, brainstorming |
| 2.0+ | Experimental | Exploración, arte |

---

## 🏗️ Estructura del proyecto

```
Laboratorio-MicroIA/
├── src/                          # Código fuente
│   ├── generation/               # Generador de texto
│   ├── hdc/                      # Computación hiperdimensional
│   ├── snn/                      # Redes neuronales spiking
│   └── fep/                      # Free Energy Principle
│
├── scripts/                      # Scripts de utilidad
│   ├── setup_corpus.sh           # Descargar UD AnCora
│   ├── download_datasets.py      # Descargar datasets Hugging Face
│   └── train_final.py            # Entrenar modelo final
│
├── examples/                     # Ejemplos de uso
│   └── generacion_por_disciplina.py
│
├── archive/                      # Investigación (fases 1-6)
│   ├── experiments/              # 24 experimentos científicos
│   ├── docs/                     # Documentación técnica
│   └── tests/                    # Tests unitarios
│
├── models/                       # Modelos entrenados
├── data/                         # Datasets
│   ├── corpus/                   # UD AnCora
│   └── datasets/                 # Hugging Face datasets
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
7. **Generación**: Sistema final con memoria HDC + FEP

**Resultados clave:**
- 24 experimentos científicos completados
- 41 claims validados
- Publicación en repositorio académico

---

## 📈 Rendimiento

| Métrica | Valor |
|---------|-------|
| Velocidad de generación | ~100 palabras/segundo |
| Memoria requerida | ~500 MB |
| Tiempo de entrenamiento | ~1 minuto |
| Tamaño del modelo | ~50 MB |

**Comparación con otros sistemas:**

| Sistema | Calidad | Velocidad | Requiere GPU |
|---------|---------|-----------|--------------|
| **Este proyecto** | Media | ⚡ Rápida | ❌ No |
| GPT-2 | Alta | 🐢 Media | ✅ Sí |
| Markov chains | Baja | ⚡⚡ Muy rápida | ❌ No |

---

## 🛠️ Desarrollo

### Ejecutar tests
```bash
pytest archive/tests/
```

### Entrenar desde cero
```bash
# 1. Descargar todos los datasets
bash scripts/setup_corpus.sh
python scripts/download_datasets.py

# 2. Entrenar modelo
python scripts/train_final.py

# 3. Probar
python demo.py
```

---

## 🤝 Contribuir

Este proyecto está **terminado y consolidado**, pero aceptamos:
- Reportes de bugs
- Mejoras de documentación
- Nuevos ejemplos de uso
- Optimizaciones de rendimiento

Ver [CHANGELOG.md](CHANGELOG.md) para historial completo.

---

## 📄 Licencia

MIT License - Ver archivo LICENSE para detalles.

---

## 📚 Referencias

1. Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with Applications to Cognitive Science and Artificial Intelligence*
2. Friston, K. (2010). *The free-energy principle: a unified brain theory?* Nature Reviews Neuroscience
3. Maass, W. (1997). *Networks of spiking neurons: the third generation of neural network models*
4. Universal Dependencies Consortium. *UD Spanish AnCora*

---

## 🙏 Agradecimientos

- **Universal Dependencies** por el corpus UD Spanish AnCora
- **Hugging Face** por los datasets en español
- **Comunidad de IA** por las investigaciones en neurociencia computacional

---

**Versión:** 3.0.0 · Enero 2025 · Proyecto terminado y consolidado

**Estado:** ✅ Listo para producción
