# 🧬 Laboratorio de Micro-IA Biomimética

**Un generador de texto en español inspirado en cómo funciona el cerebro humano.**

Este proyecto implementa una inteligencia artificial que genera texto coherente usando principios de neurociencia, sin necesidad de GPUs ni modelos masivos como GPT.

---

## 🎯 ¿Qué hace?

Dado un inicio de oración, el sistema **genera texto coherente** en español:

```
Tú escribes:  "El presidente"
IA genera:    "El presidente de la Comisión de el Mercado de el jamón curado"

Tú escribes:  "La economía"
IA genera:    "La economía canaria en los últimos dos meses"

Tú escribes:  "En la ciudad"
IA genera:    "En la ciudad de Cañete"
```

---

## 🧠 ¿Cómo funciona?

A diferencia de ChatGPT o modelos similares que usan miles de millones de parámetros, este sistema usa **tres principios del cerebro humano**:

### 1. **Memoria asociativa** (como el hipocampo)
El cerebro no almacena oraciones completas, sino **patrones** que se activan juntos. Nuestro sistema hace lo mismo: guarda secuencias de palabras como "vectores" matemáticos que pueden recuperarse por similitud.

### 2. **Predicción bayesiana** (Free Energy Principle)
El cerebro constantemente predice qué viene después y se "sorprende" cuando algo no encaja. Nuestro sistema elige la siguiente palabra que **minimiza la sorpresa**, usando probabilidades aprendidas del corpus.

### 3. **Mecanismo de fatiga** (diversidad neuronal)
Las neuronas reales se "cansan" de dispararse repetidamente. Nuestro sistema evita repetir palabras en ventanas cortas, forzando diversidad en el texto generado.

---

## 🚀 Uso rápido

### Instalación (5 minutos)

```bash
# 1. Clonar el repositorio
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar el corpus en español (~150 MB)
mkdir -p data/corpus && cd data/corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ../..
```

### Ejecutar demo interactivo

```bash
python demo.py
```

El demo te permitirá:
- Ver ejemplos predefinidos de generación
- Escribir tus propios prompts
- Ajustar la "creatividad" del sistema (temperatura)

### Usar en tu propio código

```python
from src.generation import TextGenerator

# Crear y entrenar el generador
gen = TextGenerator(dim=10000)
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

# Generar texto
texto = gen.generate("El presidente", max_words=15, temperature=0.5)
print(texto)
# → "El presidente de la Autoridad Palestina."
```

---

## 🎛️ Parámetros

El generador acepta tres parámetros principales:

| Parámetro | Descripción | Rango |
|-----------|-------------|-------|
| `prefix` | Texto inicial para continuar | Cualquier frase |
| `max_words` | Máximo de palabras a generar | 5-50 (recomendado: 10-20) |
| `temperature` | Controla la creatividad | 0.1 (conservador) a 2.0 (creativo) |

**Ejemplos:**

```python
# Conservador (texto predecible)
gen.generate("El acuerdo", temperature=0.1)

# Balanceado (recomendado)
gen.generate("El acuerdo", temperature=0.5)

# Creativo (texto más variado)
gen.generate("El acuerdo", temperature=2.0)
```

---

## 📊 ¿Qué tan bueno es?

El sistema fue evaluado con métricas estándar de generación de texto:

- **Perplejidad**: 1,625 (similar al corpus de entrenamiento: 1,781)
- **Diversidad léxica (TTR)**: 0.39 (buena variedad de palabras)
- **Repetición**: Baja (mecanismo de fatiga funciona)

**Comparación con estado del arte:**

| Sistema | Perplejidad | Requiere GPU | Entrenamiento |
|---------|-------------|--------------|---------------|
| Este proyecto | ~1,600 | ❌ No | ~1 minuto |
| GPT-2 (español) | ~20 | ✅ Sí | Días |
| Markov simple | ~3,000 | ❌ No | Segundos |

**Conclusión**: Nuestro sistema está entre Markov simple y GPT-2, pero **sin necesidad de GPU** y con entrenamiento en minutos.

---

## 🔬 ¿Qué hay adentro?

El código está organizado en módulos:

```
src/
├── generation/          # Generador de texto (lo que usas)
│   ├── text_generator.py
│   └── __init__.py
├── hdc/                 # Computación Hiperdimensional
├── active_inf/          # Free Energy Principle
└── utils/               # Utilidades
```

**Si solo quieres generar texto**, solo necesitas `src/generation/`.

**Si quieres explorar la investigación**, el historial completo de experimentos está en `archive/experiments/` (24 experimentos científicos documentados).

---

## 🎓 Para investigadores

Este proyecto fue desarrollado como parte de una investigación sobre arquitecturas de IA biomimética. El archivo `archive/` contiene:

- **24 experimentos científicos** con código ejecutable
- **Reportes detallados** de cada experimento
- **Resúmenes ejecutivos** por fase (7 fases)
- **41 claims validados** con métricas reproducibles

Si te interesa la investigación detrás, revisa:
- `archive/docs/fase_7_resumen_ejecutivo.md` (generación de texto)
- `archive/experiments/final_004/sintesis_final.md` (síntesis completa)

---

## 🐛 Problemas comunes

### "ModuleNotFoundError: No module named 'numpy'"
```bash
pip install -r requirements.txt
```

### "FileNotFoundError: corpus no encontrado"
```bash
mkdir -p data/corpus && cd data/corpus
git clone --depth 1 https://github.com/UniversalDependencies/UD_Spanish-AnCora.git
cd ../..
```

### "El texto generado es muy repetitivo"
Aumenta la temperatura:
```python
gen.generate("El presidente", temperature=1.5)  # Más creativo
```

### "El texto no tiene sentido"
Reduce la temperatura:
```python
gen.generate("El presidente", temperature=0.3)  # Más conservador
```

---

## 📝 Licencia

Este proyecto está licenciado bajo MIT License. El corpus UD Spanish-AnCora está bajo CC BY-SA 4.0.

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Algunas ideas:
- Mejorar el mecanismo de recuperación de secuencias
- Agregar más métricas de evaluación
- Optimizar el rendimiento del entrenamiento
- Traducir la documentación a otros idiomas

---

## 📧 Contacto

Para preguntas o colaboraciones, abre un issue en GitHub.

---

**Versión**: 2.0.0 · Enero 2025 · Estructura limpia para demo
