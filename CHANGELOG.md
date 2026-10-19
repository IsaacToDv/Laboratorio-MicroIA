# Changelog

Todas las versiones notables de este proyecto se documentan en este archivo.

## [4.0.0] - 2025-01-18

### 🎭 Identidades de IA y Literatura Clásica

#### Añadido
- **Sistema de identidades de IA** (`src/generation/identity.py`)
  - 5 identidades predefinidas: Poeta, Filósofo, Narrador, Ensayista, Contemporáneo
  - Inspiradas en autores hispanos clásicos (Neruda, Lorca, Borges, García Márquez, Unamuno, etc.)
  - Vocabulary bias para priorizar palabras del estilo
  - Temperatura automática según identidad
  - Patrones de oraciones y tono emocional configurables
  - Soporte para identidades personalizadas (JSON)

- **Literatura clásica en español** (`scripts/download_literature.py`)
  - Textos de Project Gutenberg (dominio público)
  - Poesía: Neruda, Lorca, Machado, Bécquer, Paz
  - Filosofía: Unamuno, Ortega y Gasset, Zambrano
  - Narrativa: García Márquez, Borges, Cortázar, Allende
  - Datasets de Hugging Face (BookCorpus)

- **Entrenamiento con literatura** (`scripts/train_with_literature.py`)
  - Combina UD AnCora + Hugging Face + literatura clásica
  - Hasta 115,000+ oraciones de entrenamiento
  - Preserva voces de autores clásicos

- **Ejemplos con identidades** (`examples/identidades.py`)
  - Demo interactivo con las 5 identidades
  - Generación con diferentes estilos literarios

#### Mejorado
- **Generador de texto** (`src/generation/text_generator.py`)
  - Método `set_identity()` para configurar personalidad
  - `_apply_vocabulary_bias()` para priorizar palabras del estilo
  - Temperatura automática si no se especifica
  - Integración completa con sistema de identidades

- **Módulo generation** (`src/generation/__init__.py`)
  - Exporta todas las clases de identidad
  - Documentación actualizada con ejemplos
  - Versión actualizada a 4.0.0

#### Cambiado
- **README.md**
  - Sección completa de identidades de IA
  - Lista de autores hispanos incluidos
  - Ejemplos de uso con diferentes identidades
  - Instrucciones para crear identidades personalizadas

### [3.0.0] - 2025-01-XX

### 🎉 Versión Final - Proyecto Consolidado

#### Añadido
- **Scripts de descarga de datasets de Hugging Face**
  - Wikipedia en español (conocimiento general)
  - Noticias en español (actualidad)
  - Textos legales (jurídico)
  - Textos científicos (investigación)
  - Literatura en español
  
- **Entrenamiento multi-disciplina**
  - Script `scripts/train_final.py` para entrenar con múltiples datasets
  - Soporte para combinar UD AnCora + datasets de Hugging Face
  - Hasta 10,000+ oraciones de entrenamiento
  
- **Ejemplos por disciplina**
  - `examples/generacion_por_disciplina.py`
  - Demuestra generación en: Política, Economía, Ciencia, Tecnología, Cultura
  
- **Documentación final**
  - CHANGELOG.md con historial completo
  - README actualizado como proyecto terminado
  - Guía de uso completa

#### Mejorado
- **Generador de texto**
  - Mejor manejo de contexto largo
  - Filtro de repeticiones mejorado
  - Mecanismo de fatiga optimizado
  
- **Estructura del proyecto**
  - Scripts organizados en `scripts/`
  - Ejemplos en `examples/`
  - Modelos entrenados en `models/`

#### Cambiado
- **README.md**
  - Reescrito como proyecto terminado
  - Enfoque en uso final
  - Sección de datasets de Hugging Face
  
- **demo.py**
  - Actualizado para usar modelo entrenado
  - Mejor manejo de errores

### [2.0.0] - 2025-01-XX

#### Cambiado
- **Limpieza para demo**
  - Movidos `tests/`, `experiments/`, `docs/` a `archive/`
  - README reescrito para audiencia no técnica
  - Estructura simplificada

### [1.1.1] - 2025-01-XX

#### Añadido
- **Mejoras al generador**
  - Filtro de repeticiones consecutivas
  - Mecanismo de fatiga
  - Mejor manejo de puntuación
  
- **Demo interactivo** (`demo.py`)
  - Modo interactivo
  - Comparación de temperaturas
  
- **Tests básicos** (`tests/test_generation.py`)
  - 6 tests de funcionalidad
  
- **Módulo exportable**
  - `src/generation/__init__.py`
  - `requirements.txt` limpio

### [1.1.0] - 2025-01-XX

#### Añadido
- **Fase 7: Generación de Texto**
  - Sistema funcional con HDC + Memoria + FEP
  - 4 experimentos ejecutables
  - Ejemplos: "El presidente de la Comisión", "La economía canaria"
  - TTR=0.390, Perplejidad=1,625

### [1.0.0] - 2025-01-XX

#### Añadido
- **Fases 1-6 completadas**
  - Fase 1: Morfología (HDC + POS tagging)
  - Fase 2: Sintaxis (parsing de dependencias)
  - Fase 3: Semántica (WSD, SRL, NER)
  - Fase 4: Discurso (correferencia, segmentación)
  - Fase 5: Pragmática (pipeline, comprensión)
  - Fase 6: Cierre SNN-FEP (SNN, FEP, benchmarking)
  
- **24 experimentos científicos**
  - 41 claims validados
  - Documentación completa

---

## Resumen de Versiones

| Versión | Fecha | Descripción |
|---------|-------|-------------|
| 4.0.0 | 2025-01 | **Identidades de IA** - Literatura clásica, 5 personalidades |
| 3.0.0 | 2025-01 | **Versión Final** - Datasets Hugging Face, multi-disciplina |
| 2.0.0 | 2025-01 | Limpieza para demo - Estructura simplificada |
| 1.1.1 | 2025-01 | Mejoras y consolidación - Demo interactivo |
| 1.1.0 | 2025-01 | Fase 7 - Generación de texto funcional |
| 1.0.0 | 2025-01 | Fases 1-6 - Investigación completa |

---

## Estadísticas del Proyecto

- **Fases de investigación**: 7
- **Experimentos científicos**: 24
- **Claims validados**: 41
- **Identidades de IA**: 5 (Poeta, Filósofo, Narrador, Ensayista, Contemporáneo)
- **Autores hispanos**: 15+ (Neruda, Lorca, Borges, García Márquez, Unamuno, etc.)
- **Datasets soportados**: 7+ (UD AnCora + Hugging Face + Literatura clásica)
- **Oraciones de entrenamiento**: ~115,000+
- **Líneas de código**: ~5,500+
- **Documentación**: ~60 páginas

---

**Proyecto terminado y consolidado con identidad y voz humana.**
