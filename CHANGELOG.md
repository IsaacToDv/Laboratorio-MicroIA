# Changelog

Todas las versiones notables de este proyecto se documentan en este archivo.

## [3.0.0] - 2025-01-XX

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
- **Datasets soportados**: 5+ (UD AnCora + Hugging Face)
- **Líneas de código**: ~5,000+
- **Documentación**: ~50 páginas

---

**Proyecto terminado y consolidado como demo funcional.**
