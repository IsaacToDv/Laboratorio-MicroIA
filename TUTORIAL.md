# 📚 Tutorial: Laboratorio de Micro-IA Biomimética

**Guía paso a paso para principiantes**

Este tutorial te llevará desde la instalación hasta la generación de tu primer texto con identidades de IA.

---

## 🎯 ¿Qué aprenderás?

1. ✅ Instalar el proyecto desde cero
2. ✅ Descargar datasets de entrenamiento
3. ✅ Entrenar tu propio generador de texto
4. ✅ Generar texto con diferentes identidades (poeta, filósofo, etc.)
5. ✅ Personalizar la generación

**Tiempo estimado:** 15-20 minutos

---

## 📋 Requisitos previos

- **Python 3.7+** instalado en tu sistema
- **Git** para clonar el repositorio
- **Conexión a internet** para descargar datasets
- **Espacio en disco:** ~500 MB

---

## 🚀 Paso 1: Instalación (5 minutos)

### 1.1 Clonar el repositorio

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA
```

### 1.2 Crear entorno virtual (recomendado)

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

### 1.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 1.4 Verificar instalación

```bash
python scripts/check_installation.py
```

Deberías ver:
```
✓ Python 3.9.7
✓ numpy instalado
✓ conllu instalado
```

---

## 📦 Paso 2: Descargar datos (5 minutos)

### 2.1 Descargar corpus principal

El corpus UD AnCora es el dataset base en español:

```bash
bash scripts/setup_corpus.sh
```

**¿Qué hace?**
- Descarga ~150 MB de texto en español
- Incluye noticias, artículos y textos literarios
- Se guarda en `data/corpus/UD_Spanish-AnCora/`

### 2.2 (Opcional) Descargar datasets adicionales

Para mejorar la calidad del generador:

```bash
# Datasets de Hugging Face (Wikipedia, noticias, etc.)
python scripts/download_datasets.py

# Literatura clásica (Neruda, Borges, García Márquez, etc.)
python scripts/download_literature.py
```

**Nota:** Estos pasos son opcionales pero recomendados para mejor calidad.

---

## 🧠 Paso 3: Entrenar el generador (2-3 minutos)

### 3.1 Entrenamiento básico

```bash
python scripts/train_final.py
```

**¿Qué hace?**
- Lee el corpus UD AnCora
- Construye memoria asociativa (HDC)
- Entrena modelo de predicción (FEP)
- Guarda el modelo en `models/`

Deberías ver:
```
Cargando corpus...
✓ 3000 oraciones cargadas
Entrenando memoria HDC...
✓ Memoria construida
Entrenando selector FEP...
✓ Modelo entrenado
```

### 3.2 (Opcional) Entrenamiento completo con literatura

Si descargaste los datasets adicionales:

```bash
python scripts/train_with_literature.py
```

Esto entrena con Wikipedia, noticias, literatura clásica, etc.

---

## 🎮 Paso 4: Probar el generador (5 minutos)

### 4.1 Ejecutar el demo interactivo

```bash
python demo.py
```

Verás un menú interactivo:

```
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   🧬 Laboratorio de Micro-IA Biomimética                      ║
║                                                               ║
║   Generador de Texto con Identidades de IA                    ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝

📋 Menú Principal

  1. Generar texto con identidad predefinida
  2. Ver ejemplos por disciplina
  3. Modo interactivo (escribe tus propios prompts)
  4. Información del proyecto
  5. Salir
```

### 4.2 Generar tu primer texto

1. Selecciona opción **1** (Generar texto con identidad)
2. Elige una identidad (ej: **Poeta**)
3. Selecciona un prompt sugerido (ej: **El amor**)
4. ¡Observa la magia! ✨

**Ejemplo de salida:**
```
✓ Identidad configurada: Poeta
  Estilo: poético
  Tono: melancólico y apasionado
  Temperatura: 1.2

1. El amor es como el agua que corre entre los dedos del tiempo
2. El amor, esa herida que no cicatriza en el alma
3. El amor es el silencio que habita en la memoria
```

### 4.3 Probar diferentes identidades

Repite el proceso con otras identidades:

- **Filósofo**: Texto reflexivo y profundo
- **Narrador**: Estilo narrativo con detalles
- **Ensayista**: Analítico y culto
- **Contemporáneo**: Moderno y accesible

### 4.4 Modo interactivo

Selecciona opción **3** para escribir tus propios prompts:

```
Tu prompt: La inteligencia artificial

→ La inteligencia artificial revoluciona la manera en que entendemos la conciencia
```

---

## 💻 Paso 5: Usar en código Python

### 5.1 Script básico

Crea un archivo `mi_script.py`:

```python
from src.generation import TextGenerator, load_identity

# Cargar generador
gen = TextGenerator()
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

# Configurar identidad
poeta = load_identity('poeta')
gen.set_identity(poeta)

# Generar texto
texto = gen.generate("El amor", max_words=20)
print(texto)
```

Ejecuta:
```bash
python mi_script.py
```

### 5.2 Generar múltiples textos

```python
prompts = ["El tiempo", "La memoria", "La muerte"]

for prompt in prompts:
    texto = gen.generate(prompt, max_words=15)
    print(f"{prompt}: {texto}")
```

### 5.3 Ajustar parámetros

```python
# Más creativo (temperatura alta)
texto = gen.generate("El amor", max_words=20, temperature=1.5)

# Más conservador (temperatura baja)
texto = gen.generate("El amor", max_words=20, temperature=0.5)
```

---

## 🎨 Paso 6: Explorar ejemplos

### 6.1 Ejemplos por disciplina

```bash
python examples/generacion_por_disciplina.py
```

Genera texto en diferentes dominios:
- Política
- Economía
- Ciencia
- Tecnología
- Cultura

### 6.2 Ejemplos con identidades

```bash
python examples/identidades.py
```

Compara cómo cada identidad genera texto diferente para el mismo prompt.

---

## 🔧 Solución de problemas

### Problema: "ModuleNotFoundError: No module named 'numpy'"

**Solución:**
```bash
pip install -r requirements.txt
```

### Problema: "FileNotFoundError: Corpus no encontrado"

**Solución:**
```bash
bash scripts/setup_corpus.sh
python scripts/train_final.py
```

### Problema: "El texto generado es muy repetitivo"

**Solución:** Aumenta la temperatura:
```python
texto = gen.generate("El amor", temperature=1.5)
```

### Problema: "El texto no tiene sentido"

**Solución:** Reduce la temperatura:
```python
texto = gen.generate("El amor", temperature=0.5)
```

### Problema: "El entrenamiento tarda mucho"

**Solución:** Reduce el número de oraciones:
```python
gen.train_from_corpus(corpus_path, max_sentences=1000)
```

---

## 🎓 Próximos pasos

### Aprende más

1. **Lee el README.md** - Documentación completa
2. **Explora archive/** - Investigación científica (fases 1-7)
3. **Lee CONTRIBUTING.md** - Cómo contribuir al proyecto

### Experimenta

1. **Crea tu propia identidad** - Personaliza el estilo de generación
2. **Entrena con tus propios textos** - Usa tu corpus personalizado
3. **Modifica los parámetros** - Experimenta con temperatura y max_words

### Contribuye

1. **Reporta bugs** - Abre issues en GitHub
2. **Agrega identidades** - Nuevos estilos literarios
3. **Mejora documentación** - Tutoriales, ejemplos, traducciones

---

## 📚 Recursos adicionales

- **Repositorio:** https://github.com/IsaacToDv/Laboratorio-MicroIA
- **Issues:** https://github.com/IsaacToDv/Laboratorio-MicroIA/issues
- **Discussions:** https://github.com/IsaacToDv/Laboratorio-MicroIA/discussions

---

## 🎉 ¡Felicidades!

Has completado el tutorial básico. Ahora puedes:

✅ Generar texto con diferentes identidades de IA  
✅ Personalizar parámetros de generación  
✅ Usar el generador en tus propios scripts  
✅ Explorar ejemplos avanzados  

**¡Sigue experimentando y divirtiéndote con el Laboratorio de Micro-IA Biomimética!**

---

**¿Preguntas o problemas?** Abre un issue en GitHub o consulta CONTRIBUTING.md
