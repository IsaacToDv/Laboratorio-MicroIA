# ❓ Preguntas Frecuentes (FAQ)

Respuestas a las preguntas más comunes sobre el Laboratorio de Micro-IA Biomimética.

---

## 🚀 Instalación y Configuración

### ¿Qué versión de Python necesito?

**Python 3.7 o superior.** Recomendamos Python 3.8+ para mejor rendimiento.

Verifica tu versión:
```bash
python --version
```

### ¿Necesito una GPU?

**No.** El proyecto está diseñado para funcionar en CPU. Una laptop moderna es suficiente.

### ¿Cuánto espacio en disco necesito?

- **Instalación básica:** ~200 MB
- **Con todos los datasets:** ~500 MB
- **Modelos entrenados:** ~50 MB

### ¿Puedo usar Windows?

**Sí.** El proyecto funciona en Windows, Mac y Linux. Usa PowerShell o CMD en lugar de bash.

### ¿Cómo actualizo el proyecto?

```bash
git pull origin main
pip install -r requirements.txt
```

---

## 📦 Datasets y Entrenamiento

### ¿Qué es el corpus UD AnCora?

Es un corpus de español con anotaciones lingüísticas (POS tags, dependencias, etc.) mantenido por Universal Dependencies. Incluye ~500,000 palabras de noticias y textos literarios.

### ¿Por qué descargar datasets adicionales?

Los datasets adicionales (Wikipedia, noticias, literatura) mejoran la **diversidad** y **calidad** del texto generado. Sin ellos, el generador solo usa el corpus UD AnCora.

### ¿Cuánto tarda el entrenamiento?

- **Básico (solo UD AnCora):** 1-2 minutos
- **Completo (con datasets):** 3-5 minutos

### ¿Puedo usar mi propio corpus?

**Sí.** Prepara tu corpus en formato CoNLL-U y pásalo a `train_from_corpus()`:

```python
gen.train_from_corpus("ruta/a/mi_corpus.conllu")
```

### ¿Qué pasa si el entrenamiento falla?

Verifica:
1. Tienes suficiente RAM (mínimo 2 GB)
2. El corpus existe y es válido
3. Las dependencias están instaladas

Ejecuta el verificador:
```bash
python scripts/check_installation.py
```

---

## 🎭 Identidades de IA

### ¿Qué son las identidades de IA?

Son **personalidades** que modifican el estilo de generación. Cada identidad tiene:
- **Vocabulary bias:** Palabras priorizadas
- **Temperatura:** Nivel de creatividad
- **Patrones:** Estructuras de oraciones típicas
- **Tono emocional:** Melancólico, reflexivo, etc.

### ¿Puedo crear mi propia identidad?

**Sí.** Crea un objeto `AIIdentity`:

```python
from src.generation import AIIdentity

mi_identidad = AIIdentity(
    name="Periodista",
    description="Estilo periodístico objetivo",
    style="informativo",
    temperature=0.7,
    top_authors=["García Márquez", "Kapuściński"],
    vocabulary_bias={
        'investigación': 2.0,
        'fuente': 1.8,
        'reportaje': 1.7
    },
    sentence_patterns=[
        "Según fuentes cercanas...",
        "La investigación revela..."
    ],
    emotional_tone="objetivo"
)

gen.set_identity(mi_identidad)
```

### ¿Cuál identidad es mejor para poesía?

**Poeta** - Inspirada en Neruda, Lorca, Paz. Usa vocabulario lírico y temperatura alta (1.2).

### ¿Cuál identidad es mejor para textos técnicos?

**Ensayista** - Analítica y precisa. Temperatura baja (0.8) para mayor coherencia.

### ¿Puedo mezclar identidades?

No directamente, pero puedes crear una identidad personalizada combinando elementos de varias.

---

## 📝 Generación de Texto

### ¿Por qué el texto generado a veces no tiene sentido?

El generador usa **memoria asociativa** y **predicción estadística**, no comprensión semántica profunda. A veces produce combinaciones gramaticalmente correctas pero semánticamente extrañas.

**Soluciones:**
- Reduce la temperatura (0.3-0.5)
- Usa prompts más específicos
- Genera múltiples veces y elige el mejor resultado

### ¿Por qué el texto es muy repetitivo?

Puede pasar con temperatura muy baja o prompts muy cortos.

**Soluciones:**
- Aumenta la temperatura (1.0-1.5)
- Usa prompts más largos
- Aumenta `max_words`

### ¿Cómo genero texto más largo?

```python
texto = gen.generate("El presidente", max_words=50)
```

**Nota:** Textos muy largos (>100 palabras) pueden perder coherencia.

### ¿Puedo controlar la longitud exacta?

No exactamente. El generador puede terminar antes si encuentra un punto final. Usa `max_words` como límite superior.

### ¿Cómo genero texto en un tema específico?

Usa prompts específicos:

```python
# Política
gen.generate("El presidente de la república")

# Ciencia
gen.generate("Los investigadores del laboratorio")

# Literatura
gen.generate("En un lugar de la Mancha")
```

---

## 🔧 Problemas Técnicos

### Error: "ModuleNotFoundError"

Faltan dependencias. Instala:
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: Corpus no encontrado"

Descarga el corpus:
```bash
bash scripts/setup_corpus.sh
```

### Error: "MemoryError" durante entrenamiento

Tu sistema no tiene suficiente RAM. Reduce el tamaño del entrenamiento:
```python
gen.train_from_corpus(corpus_path, max_sentences=1000)
```

### El generador es muy lento

- Reduce `dim` (dimensiones HDC): `TextGenerator(dim=5000)`
- Reduce `max_sentences` en entrenamiento
- Cierra otras aplicaciones que consuman RAM

### Los caracteres especiales no se muestran bien

Asegúrate de que tu terminal use UTF-8:

**Linux/Mac:**
```bash
export LANG=en_US.UTF-8
```

**Windows:**
```cmd
chcp 65001
```

---

## 🎨 Personalización

### ¿Puedo cambiar el idioma?

El proyecto está diseñado para **español**. Para otros idiomas necesitarías:
1. Un corpus en ese idioma
2. Ajustar el tokenizer
3. Posiblemente modificar las identidades

### ¿Puedo usar el generador en una aplicación web?

**Sí.** El generador es una librería Python. Puedes integrarlo en Flask, Django, FastAPI, etc.

**Ejemplo con Flask:**
```python
from flask import Flask, request, jsonify
from src.generation import TextGenerator, load_identity

app = Flask(__name__)
gen = TextGenerator()
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    identity = data.get('identity', 'contemporaneo')
    
    gen.set_identity(load_identity(identity))
    text = gen.generate(prompt, max_words=20)
    
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run()
```

### ¿Puedo guardar y cargar modelos entrenados?

Actualmente no hay serialización de modelos. Debes reentrenar cada vez.

**Workaround:** Guarda el corpus y reentrena (es rápido, 1-2 minutos).

---

## 📊 Rendimiento y Calidad

### ¿Cómo se compara con GPT-3/ChatGPT?

**No es comparable.** Este proyecto es:
- ✅ Más ligero (sin GPU)
- ✅ Más rápido de entrenar (minutos vs días)
- ✅ Más transparente (código abierto)
- ❌ Menos potente (no usa transformers)
- ❌ Menos coherente en textos largos

Es una **alternativa educativa y experimental**, no un reemplazo de modelos comerciales.

### ¿Qué métricas se usan para evaluar?

- **Perplejidad:** ~1,600 (menor es mejor)
- **Diversidad léxica (TTR):** ~0.39 (mayor es mejor)
- **Repetición de bigramas:** ~0.18 (menor es mejor)

Ver `archive/experiments/generation_004/` para detalles.

### ¿Por qué la perplejidad es alta comparada con GPT?

Porque usamos un modelo más simple (HDC + FEP) sin las miles de millones de parámetros de GPT. La ventaja es que funciona sin GPU y es interpretable.

---

## 🤝 Contribución

### ¿Cómo reporto un bug?

Abre un issue en GitHub:
https://github.com/IsaacToDv/Laboratorio-MicroIA/issues

Incluye:
- Descripción del problema
- Pasos para reproducir
- Versión de Python y sistema operativo
- Mensajes de error completos

### ¿Cómo propongo una nueva característica?

Abre un issue con la etiqueta "enhancement" o "feature request".

### ¿Puedo contribuir con código?

**Sí.** Lee CONTRIBUTING.md para las guías de contribución.

---

## 📚 Aprendizaje

### ¿Dónde aprendo más sobre HDC?

- **Kanerva, P. (2009).** *Hyperdimensional Computing*
- Ver `archive/docs/` para documentación técnica

### ¿Dónde aprendo más sobre FEP (Free Energy Principle)?

- **Friston, K. (2010).** *The free-energy principle*
- Ver `archive/docs/fase_6_resumen_ejecutivo.md`

### ¿Hay papers científicos del proyecto?

El proyecto incluye 24 experimentos científicos documentados en `archive/experiments/` y `archive/docs/`.

---

## 💼 Uso Comercial

### ¿Puedo usar esto comercialmente?

**Sí.** El proyecto está bajo licencia MIT, que permite uso comercial.

### ¿Qué pasa con los textos generados?

Los textos generados son tuyos. Puedes usarlos libremente.

### ¿Y los textos de entrenamiento?

Verifica las licencias de los datasets:
- **UD AnCora:** CC BY-SA 4.0
- **Hugging Face:** Ver licencias individuales
- **Literatura clásica:** Dominio público

---

## 🎓 Para Educadores

### ¿Puedo usar esto en clase?

**Sí.** Es excelente para enseñar:
- Procesamiento de lenguaje natural
- Neurociencia computacional
- Generación de texto
- Inteligencia artificial biomimética

### ¿Hay material educativo?

- **TUTORIAL.md** - Guía paso a paso
- **README.md** - Documentación completa
- **archive/** - Investigación científica

### ¿Cuánto tiempo toma una sesión práctica?

- **Instalación:** 10 minutos
- **Tutorial básico:** 20 minutos
- **Experimentación:** 30-60 minutos

---

## 🆘 ¿No encuentras tu pregunta?

1. **Busca en issues de GitHub:** https://github.com/IsaacToDv/Laboratorio-MicroIA/issues
2. **Abre un nuevo issue:** Describe tu problema detalladamente
3. **Consulta CONTRIBUTING.md:** Para guías de contribución

---

**Última actualización:** Enero 2025  
**Versión del proyecto:** 4.0.0
