# 🧬 Laboratorio de Micro-IA Biomimética

**Generador de texto en español con identidades de IA que se adaptan automáticamente al tema.**

Inspirado en neurociencia (memoria asociativa HDC), lingüística computacional (Free Energy Principle) y literatura clásica hispana. Funciona sin GPU en cualquier laptop.

```
❯ El amor es como el agua
  🎭 Poeta (melancólico y apasionado)
  → El amor es como el agua que corre entre las piedras del tiempo y no se detiene...

❯ ¿Quién fue Platón?
  🎭 Filósofo (reflexivo y cuestionador)
  → Quién fue Platón y su influencia en el pensamiento occidental de la filosofía griega...

❯ ¿Qué es Google?
  🎭 Contemporáneo (directo y observador)
  → Qué es Google y su impacto en la tecnología moderna y el acceso a la información...
```

La IA detecta el tema de tu prompt y adopta la identidad más adecuada. Sin menús, sin configuración. Solo escribe.

---

## 🚀 Instalación

### Opción A: Script automático (recomendado)

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA
bash install.sh
```

### Opción B: Manual

```bash
git clone https://github.com/IsaacToDv/Laboratorio-MicroIA.git
cd Laboratorio-MicroIA

# Entorno virtual (obligatorio en Arch, Fedora, Ubuntu 23.04+)
python3 -m venv venv
source venv/bin/activate    # Linux/Mac
# venv\Scripts\activate     # Windows

pip install -r requirements.txt

# Descargar datos y entrenar
bash scripts/setup_corpus.sh
python scripts/train_final.py
```

> **⚠️ ¿Error "externally-managed-environment"?** Es normal en Linux moderno. El entorno virtual lo resuelve. Ver [FAQ](FAQ.md) para detalles.

---

## 🎮 Uso

### Prompt libre (demo interactivo)

```bash
python demo.py
```

Escribe lo que quieras. La IA detecta el tema y elige la identidad automáticamente.

```
🧬 Micro-IA Biomimética v6 — Deep Cognition
Escribe libremente. Usa /help para comandos.
────────────────────────────────────────────

❯ La vida es un sueño
  🎭 Filósofo
  → La vida es un sueño y los sueños son la materia de la existencia...

❯ /poeta           ← forzar identidad
✓ Poeta (melancólico y apasionado)

❯ El mar
  → El mar es ese silencio antiguo que golpea las puertas de la memoria...

❯ /auto            ← volver a auto-detección
❯ /help            ← ver todos los comandos
❯ /salir           ← terminar
```

### Comandos

| Comando | Acción |
|---------|--------|
| `/poeta` `/filosofo` `/narrador` `/ensayista` `/contemporaneo` | Forzar identidad |
| `/auto` | Auto-detectar identidad (default) |
| `/help` | Ver ayuda |
| `/salir` | Terminar |

### Uso en código Python

```python
from src.generation import TextGenerator, load_identity

# Cargar generador (usa dim=20000 para mejor calidad)
gen = TextGenerator(dim=20000)
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

# Opción 1: Identidad fija
poeta = load_identity('poeta')
gen.set_identity(poeta)
print(gen.generate("El amor", max_words=25))

# Opción 2: Auto-detección (la IA elige según el tema)
gen.set_auto_identity(True)
print(gen.generate("¿Quién fue Platón?", max_words=25))
```

### Ejemplos incluidos

```bash
python examples/identidades.py             # Comparar las 5 identidades
python examples/generacion_por_disciplina.py  # Generar por disciplina
```

---

## 🎭 Identidades de IA

La IA puede adoptar 5 personalidades inspiradas en grandes autores hispanos:

| Identidad | Estilo | Inspiración | Temp. | Se activa con... |
|-----------|--------|-------------|-------|-------------------|
| 🎭 **Poeta** | Lírico, emocional | Neruda, Lorca, Paz, Vallejo | 1.2 | amor, alma, sueño, luna |
| 🤔 **Filósofo** | Reflexivo, profundo | Unamuno, Ortega, Zambrano | 0.9 | dios, verdad, razón, Platón |
| 📖 **Narrador** | Descriptivo, vívido | García Márquez, Borges, Cortázar | 1.1 | había, historia, recuerdo |
| 📝 **Ensayista** | Analítico, erudito | Paz, Vargas Llosa, Borges | 0.8 | cultura, sociedad, análisis |
| 🏙️ **Contemporáneo** | Directo, moderno | Bolaño, Marías, Grandes | 1.0 | google, internet, tecnología |

En modo `/auto`, la IA analiza las palabras del prompt y selecciona la identidad más relevante.

---

## 📦 Datasets de entrenamiento

| Fuente | Dominio | Oraciones |
|--------|---------|-----------|
| UD Spanish AnCora | Lingüística | ~15,000 |
| Wikipedia ES | Conocimiento general | ~47,000 |
| Noticias ES | Actualidad | Respaldo |
| Textos legales | Jurídico | Respaldo |
| Textos científicos | Investigación | Respaldo |
| Literatura clásica | Poesía, filosofía, narrativa | ~30,000 |

Descarga con:
```bash
bash scripts/setup_corpus.sh              # Corpus principal
python scripts/download_datasets.py       # Datasets multi-disciplina
python scripts/download_literature.py     # Literatura clásica
```

---

## 🏗️ Estructura

```
Laboratorio-MicroIA/
├── src/generation/          # Motor de generación
│   ├── text_generator.py    # Generador v3 (HDC 20K + 5-gramas + splicing)
│   └── identity.py          # Sistema de identidades
├── scripts/                 # Descarga y entrenamiento
│   ├── setup_corpus.sh      # Descargar UD AnCora
│   ├── download_datasets.py # Datasets Hugging Face
│   ├── download_literature.py # Literatura clásica
│   ├── data_loader.py       # Carga datos de múltiples fuentes
│   ├── train_final.py       # Entrenamiento básico
│   ├── train_with_literature.py # Entrenamiento completo
│   └── check_installation.py # Verificar instalación
├── examples/                # Ejemplos de uso
├── demo.py                  # Demo de prompt libre
├── install.sh               # Instalador automático
├── TUTORIAL.md              # Guía paso a paso
├── FAQ.md                   # Preguntas frecuentes
├── CONTRIBUTING.md          # Guía de contribución
└── CHANGELOG.md             # Historial de versiones
```

---

## 🔬 Arquitectura (v6 Deep Cognition)

El generador combina **ocho componentes biomiméticos**:

1. **Memoria HDC** (hipocampo) — 20,000 dimensiones, almacena secuencias como vectores hiperdimensionales con índice invertido para búsqueda O(1)

2. **Selector FEP** (Free Energy Principle) — Interpolación de n-gramas (1 a 5) con pesos exponenciales. Minimiza energía libre para seleccionar la siguiente palabra

3. **Embeddings semánticos** (LSA/SVD) — Matriz de co-ocurrencia PPMI reducida a 150 dimensiones. La IA **entiende** que "amor" ≈ "corazón" ≈ "alma"

4. **Clasificador de intención** — Analiza palabras clave del prompt y selecciona automáticamente la identidad más apropiada

5. **Memoria conversacional** — Recuerda los últimos 10 turnos, extrae temas recurrentes, enriquece prompts cortos con contexto

6. **Modelo de temas** (k-means) — Agrupa oraciones del corpus en 20 temas semánticos. Permite búsqueda por tema, no solo por keyword

7. **Scoring de coherencia** — Evalúa la respuesta completa en 4 dimensiones: relevancia semántica (40%), diversidad léxica (25%), coherencia local (20%), estructura (15%)

8. **Tokenizador robusto** — Separa puntuación, maneja signos de apertura (¿¡), guiones internos, normalización

**Generación multi-candidato con scoring:**
```
1. Splicing multi-oración (combina fragmentos relevantes)
2. Generación secuencial con re-ranking semántico
3. Búsqueda temática por similitud de embeddings
4. CoherenceScorer evalúa los 3 candidatos
5. Selecciona el de mayor calidad global
```

---

## 📚 Base científica

Este proyecto implementa **7 fases de investigación** documentadas en `archive/`:

| Fase | Tema | Métrica clave |
|------|------|---------------|
| 1 | Morfología (POS tagging) | 93.26% accuracy |
| 2 | Sintaxis (dependencias) | 88.33% UAS |
| 3 | Semántica (WSD, SRL, NER) | 97% WSD |
| 4 | Discurso (correferencia) | 66.71% F1 |
| 5 | Pragmática (pipeline) | 52K tokens/s |
| 6 | SNN-FEP (spiking + energía libre) | — |
| 7 | Generación (HDC + FEP + identidades) | v6.0.0 |

**Total:** 24 experimentos, 41 claims validados.

---

## 🤝 Contribuir

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para guías detalladas.

**Áreas bienvenidas:** nuevas identidades, textos literarios (dominio público), mejoras de documentación, reportes de bugs, optimizaciones.

---

## 📄 Licencia

MIT License. Ver [LICENSE](LICENSE).

Los textos literarios incluidos son de dominio público o licencias compatibles. Los datasets se usan de acuerdo con sus licencias individuales (UD AnCora: CC BY-SA 4.0, Hugging Face: ver individuales).

---

**Versión:** 6.0.0 · Octubre 2025 · Deep Cognition — Semántica real + memoria conversacional + scoring de coherencia
