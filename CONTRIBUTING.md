# Guía de Contribución

¡Gracias por tu interés en contribuir al Laboratorio de Micro-IA Biomimética!

## 🎯 Estado del Proyecto

Este proyecto está **terminado y consolidado** (v4.0.0), pero aceptamos contribuciones en las siguientes áreas:

### Áreas de Contribución Bienvenidas

✅ **Nuevas identidades de IA**
- Estilos literarios adicionales
- Autores hispanos no incluidos
- Identidades temáticas (científico, periodista, etc.)

✅ **Más textos literarios**
- Obras en dominio público
- Textos de autores hispanos clásicos
- Poesía, filosofía, narrativa, ensayo

✅ **Mejoras de documentación**
- Traducciones a otros idiomas
- Tutoriales y ejemplos adicionales
- Correcciones y aclaraciones

✅ **Optimizaciones de rendimiento**
- Mejoras en velocidad de generación
- Reducción de uso de memoria
- Optimizaciones de entrenamiento

✅ **Reportes de bugs**
- Errores en generación de texto
- Problemas de instalación
- Inconsistencias en documentación

❌ **No aceptamos**
- Cambios mayores a la arquitectura
- Nuevas fases de investigación (el proyecto está terminado)
- Modificaciones que rompan compatibilidad

---

## 🚀 Cómo Contribuir

### 1. Reportar un Bug

Abre un issue en GitHub con:
- Descripción clara del problema
- Pasos para reproducir
- Resultado esperado vs. actual
- Versión de Python y sistema operativo
- Capturas de pantalla (si aplica)

### 2. Proponer una Nueva Identidad

```python
from src.generation import AIIdentity

mi_identidad = AIIdentity(
    name="Periodista",
    description="Voz informativa y objetiva",
    style="periodístico",
    temperature=0.7,
    top_authors=["Gabriel García Márquez", "Ryszard Kapuściński"],
    vocabulary_bias={
        'información': 2.0,
        'fuente': 1.8,
        'reportaje': 1.7,
        'investigación': 1.8
    },
    sentence_patterns=[
        "Según fuentes cercanas a {tema}",
        "La información revela que {evento}"
    ],
    emotional_tone="objetivo e informativo"
)
```

### 3. Agregar Textos Literarios

Los textos deben ser:
- **Dominio público** (autor fallecido hace más de 70 años)
- **En español**
- **Relevantes** para las identidades existentes

Formato JSON:
```json
{
  "sentences": [
    "En un lugar de la Mancha, de cuyo nombre no quiero acordarme",
    "Muchos años después, frente al pelotón de fusilamiento..."
  ]
}
```

### 4. Enviar un Pull Request

1. Fork el repositorio
2. Crea una branch para tu contribución:
   ```bash
   git checkout -b feature/nueva-identidad-poeta
   ```
3. Haz tus cambios
4. Ejecuta tests (si aplica):
   ```bash
   python -m pytest archive/tests/
   ```
5. Commit con mensaje claro:
   ```bash
   git commit -m "Add: Nueva identidad de periodista"
   ```
6. Push y abre Pull Request

---

## 📝 Estándares de Código

### Python

- **Estilo**: PEP 8
- **Docstrings**: Google style
- **Type hints**: Recomendados
- **Testing**: pytest (para cambios en `src/`)

### Ejemplo de Docstring

```python
def generate_text(prompt: str, max_words: int = 20) -> str:
    """
    Genera texto a partir de un prompt.
    
    Args:
        prompt: Texto inicial para generar
        max_words: Máximo de palabras a generar
        
    Returns:
        Texto generado
        
    Raises:
        ValueError: Si max_words < 1
    """
    pass
```

### Commits

Usa mensajes claros y descriptivos:

```
✅ Add: Nueva identidad de poeta surrealista
✅ Fix: Error en generación con temperatura alta
✅ Update: Documentación de identidades
✅ Refactor: Optimizar búsqueda en memoria HDC

❌ Cambios
❌ Fix
❌ Update
```

---

## 🎭 Crear una Nueva Identidad

### Pasos

1. **Define la identidad** en `src/generation/identity.py`:

```python
IDENTITIES['periodista'] = AIIdentity(
    name="Periodista",
    description="Voz informativa y objetiva, inspirada en el periodismo literario",
    style="periodístico",
    temperature=0.7,
    top_authors=["Gabriel García Márquez", "Ryszard Kapuściński", "Leila Guerriero"],
    vocabulary_bias={
        'información': 2.0,
        'fuente': 1.8,
        'reportaje': 1.7,
        'investigación': 1.8,
        'testimonio': 1.6,
        'documento': 1.5
    },
    sentence_patterns=[
        "Según fuentes cercanas a {tema}",
        "La información revela que {evento}",
        "En un reportaje publicado en {medio}"
    ],
    emotional_tone="objetivo e informativo"
)
```

2. **Agrega textos literarios** en `scripts/download_literature.py`:

```python
journalism_samples = [
    "La crónica de los hechos revela una verdad incómoda",
    "Las fuentes consultadas prefirieron mantener el anonimato",
    "El reportaje destapó una red de corrupción"
]
```

3. **Prueba la identidad**:

```python
from src.generation import TextGenerator, load_identity

gen = TextGenerator()
gen.train_from_corpus("data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu")

periodista = load_identity('periodista')
gen.set_identity(periodista)

print(gen.generate("La investigación"))
```

4. **Actualiza la documentación**:
   - Agrega la identidad al README.md
   - Actualiza CHANGELOG.md
   - Agrega ejemplos en `examples/`

---

## 🐛 Reportar Bugs

### Información Necesaria

```markdown
**Descripción del bug:**
[Descripción clara y concisa]

**Pasos para reproducir:**
1. Ejecutar `python demo.py`
2. Ingresar prompt "El amor"
3. Usar identidad "poeta"

**Resultado esperado:**
Texto poético coherente

**Resultado actual:**
Error: KeyError 'alma'

**Entorno:**
- Python: 3.9.7
- Sistema: Ubuntu 20.04
- Versión del proyecto: 4.0.0

**Logs:**
[Incluir logs relevantes]
```

---

## 📚 Recursos

- **Documentación completa**: [README.md](README.md)
- **Historial de versiones**: [CHANGELOG.md](CHANGELOG.md)
- **Investigación científica**: [archive/docs/](archive/docs/)
- **Ejemplos de uso**: [examples/](examples/)

---

## 💬 Contacto

- **Issues de GitHub**: Para bugs y propuestas
- **Discussions**: Para preguntas y debates
- **Pull Requests**: Para contribuciones de código

---

## 🙏 Agradecimientos

Todas las contribuciones son valoradas y reconocidas. Los contribuidores serán mencionados en:
- README.md (sección de agradecimientos)
- CHANGELOG.md (en la versión correspondiente)
- Releases de GitHub

---

**¡Gracias por ayudar a mejorar el Laboratorio de Micro-IA Biomimética!**
