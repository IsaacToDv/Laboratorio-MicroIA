# 📚 Tutorial: Laboratorio de Micro-IA Biomimética

**Guía paso a paso para principiantes**

Desde la instalación hasta tu primera generación de texto. Sin conocimientos previos requeridos.

**Tiempo estimado:** 10-15 minutos

---

## 📋 Requisitos

- **Python 3.8+** (verifica con `python3 --version`)
- **Git** (verifica con `git --version`)
- **Conexión a internet** (para descargar datos)
- **Espacio en disco:** ~500 MB

---

## 🚀 Paso 1: Instalar

### Opción rápida (recomendada)

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA
bash install.sh
```

Este script crea el entorno virtual, instala dependencias y te pregunta qué datos descargar.

### Opción manual

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Entorno virtual (obligatorio en Linux moderno)
python3 -m venv venv
source venv/bin/activate     # Linux/Mac
# venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

> **¿Error "externally-managed-environment"?** Es normal en Arch, Fedora, Ubuntu 23.04+. El entorno virtual lo soluciona.

### Verificar instalación

```bash
python scripts/check_installation.py
```

Deberías ver ✓ en Python, numpy, scipy y conllu.

---

## 📦 Paso 2: Descargar datos

Necesitas al menos una fuente de datos. Elige según tu tiempo y conexión:

### Opción A: Corpus UD AnCora (~2 min, 150 MB)

```bash
bash scripts/setup_corpus.sh
```

Corpus lingüístico con ~15,000 oraciones de noticias y textos en español.

### Opción B: Wikipedia (~5 min, ~500 MB)

```bash
python scripts/download_datasets.py
```

Descarga ~47,000 oraciones de Wikipedia en español. Mucho más vocabulario.

### Opción C: Literatura clásica (~1 min)

```bash
python scripts/download_literature.py
```

Textos de Cervantes, poesía, filosofía y narrativa (~30,000 oraciones).

> **Puedes combinar:** cuantos más datos descargues, mejor será la generación.

---

## 🧠 Paso 3: Entrenar

```bash
# Entrenamiento estándar
python scripts/train_final.py
```

Busca automáticamente todos los datos disponibles (UD AnCora, Wikipedia, literatura) y entrena el generador.

Deberías ver algo como:
```
📖 Buscando datos disponibles...
   ✓ 47288 oraciones de wikipedia
   ✓ 15000 oraciones de UD AnCora
   Total: 62288 oraciones de 2 fuente(s)

🧠 Entrenando generador...
   ✓ 62288 secuencias almacenadas
   ✓ Selector FEP entrenado
   ✓ Vocabulario: 85000 palabras

✅ Entrenamiento completado
```

Si descargaste literatura, puedes usar:
```bash
python scripts/train_with_literature.py
```

---

## 🎮 Paso 4: Generar texto

### El demo de prompt libre

```bash
python demo.py
```

Verás:
```
🧬 Micro-IA Biomimética v6 — Deep Cognition
Escribe libremente. Usa /help para comandos.
────────────────────────────────────────────

❯ _
```

**Solo escribe lo que quieras.** La IA detecta el tema y adopta la identidad más adecuada automáticamente.

### Ejemplos de uso

```
❯ El amor es como el agua
  🎭 Poeta (melancólico y apasionado)
  → El amor es como el agua que corre entre las piedras del tiempo...

❯ ¿Quién fue Platón?
  🎭 Filósofo (reflexivo y cuestionador)
  → Quién fue Platón y su influencia en el pensamiento occidental...

❯ ¿Qué es Google?
  🎭 Contemporáneo (directo y observador)
  → Qué es Google y su impacto en la tecnología moderna...
```

### Comandos útiles

```
/poeta         ← forzar voz poética
/filosofo      ← forzar voz filosófica
/narrador      ← forzar voz narrativa
/ensayista     ← forzar voz ensayística
/contemporaneo ← forzar voz moderna
/auto          ← volver a auto-detección
/help          ← ver todos los comandos
/salir         ← terminar
```

### Sin comandos: solo habla

No necesitas memorizar comandos. Escribe normalmente y la IA se adapta:

- Habla de **amor, alma, sueños** → responde como poeta
- Pregunta sobre **dios, verdad, Platón** → responde como filósofo
- Cuenta una **historia, un recuerdo** → responde como narrador
- Analiza **cultura, sociedad** → responde como ensayista
- Pregunta sobre **tecnología, internet** → responde como contemporáneo

---

## 💻 Paso 5: Usar en tu propio código

### Script básico

```python
from src.generation import TextGenerator, load_identity

gen = TextGenerator(dim=20000)
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

# Con identidad fija
poeta = load_identity('poeta')
gen.set_identity(poeta)
print(gen.generate("El amor", max_words=25))

# Con auto-detección de identidad
gen.set_auto_identity(True)
print(gen.generate("¿Quién fue Platón?"))
```

### Generar varios textos

```python
prompts = ["El tiempo", "La memoria", "La muerte"]
for prompt in prompts:
    print(f"{prompt}: {gen.generate(prompt, max_words=20)}")
```

### Ajustar creatividad

```python
# Más creativo (temperatura alta)
gen.generate("El amor", temperature=1.5)

# Más conservador (temperatura baja)
gen.generate("El amor", temperature=0.5)
```

---

## 🎨 Paso 6: Explorar ejemplos

```bash
# Comparar las 5 identidades con los mismos prompts
python examples/identidades.py

# Generar texto en diferentes disciplinas
python examples/generacion_por_disciplina.py
```

---

## 🔧 Problemas comunes

| Problema | Solución |
|----------|----------|
| `externally-managed-environment` | Usa `python3 -m venv venv && source venv/bin/activate` |
| `ModuleNotFoundError: numpy` | `pip install -r requirements.txt` |
| `Corpus no encontrado` | `bash scripts/setup_corpus.sh` |
| `No hay suficientes datos` | Descarga al menos una fuente (Paso 2) |
| Texto repetitivo | Aumenta temperatura: `temperature=1.5` |
| Texto sin sentido | Reduce temperatura: `temperature=0.5` |

Para más detalles, consulta [FAQ.md](FAQ.md).

---

## 🎓 Próximos pasos

- **[FAQ.md](FAQ.md)** — Preguntas frecuentes
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Cómo contribuir al proyecto
- **[CHANGELOG.md](CHANGELOG.md)** — Historial de versiones
- **`archive/`** — Investigación científica completa (7 fases, 24 experimentos)

---

**¿Preguntas?** Abre un [issue en GitHub](https://github.com/IsaacToDv/Laboratorio-MicroIA/issues).
