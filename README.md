# Laboratorio-MicroIA
Esto es un laboratorio donde se orquestaron varios agentes en el entorno de LLM Arena, para más información, entra a el link de la coversación
https://chatgpt.com/share/6ad3d93a-55a8-83e9-aabc-c8c9a2dd4723 

comentario de: arena/28129a0f-laboratorio-microia
# Laboratorio de Micro-IA Biomimética
## Computación Hiperdimensional — Redes Neuronales de Impulsos — Inferencia Activa

Laboratorio de investigación computacional para el estudio de una arquitectura híbrida de Micro-IA biomimética que combina:

- **HDC/VSA**: Computación Hiperdimensional y Arquitecturas de Símbolos Vectoriales
- **SNN**: Redes Neuronales de Impulsos (Spiking Neural Networks)
- **FEP**: Principio de Energía Libre e Inferencia Activa

### Objetivo

Investigar si esta arquitectura híbrida puede adquirir regularidades lingüísticas del español de forma incremental, con ventajas medibles en memoria, retención o coste computacional frente a sistemas de referencia.

### Estado actual

**Fase:** Validación de componentes fundamentales  
**Experimentos completados:**
- ✓ HDC-Esp-001: Memoria asociativa para caracteres
- ✓ HDC-Esp-002: Escalabilidad a vocabularios grandes
- ✓ HDC-Esp-003: Capacidad de memoria asociativa con bundles

### Estructura del proyecto

```
Laboratorio-MicroIA/
├── src/
│   ├── hdc/              # Módulo A — Computación Hiperdimensional
│   │   ├── core.py       # Operaciones HDC fundamentales
│   │   ├── item_memory.py # Memoria asociativa de ítems
│   │   └── sequence_encoder.py # Codificador de secuencias
│   ├── snn/              # Módulo B — Redes Neuronales de Impulsos (pendiente)
│   ├── active_inf/       # Módulo C — Inferencia Activa (pendiente)
│   └── interface/        # Módulo D — Integración (pendiente)
├── experiments/
│   └── hdc_esp_001/      # Primer experimento: memoria asociativa
├── tests/                # Pruebas unitarias
├── docs/
│   ├── 00_evaluacion_inicial_critica.md
│   └── experimentos/     # Informes de resultados
└── data/                 # Corpus y vocabularios
```

### Resultados principales

#### HDC-Esp-001: Memoria Asociativa para Caracteres del Español

**Estado:** Completado ✓  
**Fecha:** 2026-10-17

**Hallazgos:**
- Recuperación perfecta (0% error) para 44 caracteres en todas las dimensiones (D ∈ {1024, 2048, 4096, 8192, 10000})
- Robustez excepcional al ruido: 0% error incluso con 40% de ruido
- Sin colisiones observadas entre hipervectores
- Tiempo de consulta: 75-870 μs dependiendo de la dimensión
- Memoria: 44-430 KB para el inventario completo

**Conclusión:** HDC es viable para representar caracteres del español. La dimensión D = 1024 es suficiente para inventarios pequeños.

**Informe completo:** [docs/experimentos/hdc_esp_001_resultados.md](docs/experimentos/hdc_esp_001_resultados.md)

---

#### HDC-Esp-002: Escalabilidad a Vocabularios Grandes

**Estado:** Completado ✓  
**Fecha:** 2026-10-17

**Hallazgos:**
- **Item memory tiene capacidad prácticamente ilimitada** para recuperación exacta
- D = 1024 soporta 5000 tokens (98× capacidad teórica conservadora)
- D = 4096 soporta 20000 tokens (117× capacidad teórica)
- 0% error en todos los casos probados
- Similitud cruzada máxima < 0.12 incluso con 20000 vectores

**Conclusión:** El límite de capacidad NO está en almacenar vocabulario, sino en representar secuencias y estructuras complejas.

**Informe completo:** [docs/experimentos/hdc_esp_002_resultados.md](docs/experimentos/hdc_esp_002_resultados.md)

---

#### HDC-Esp-003: Capacidad de Memoria Asociativa con Bundles

**Estado:** Completado (parcial) ✓  
**Fecha:** 2026-10-17

**Hallazgos:**
- La fórmula teórica C ≈ D / (2 × log₂(D)) **es válida** para bundles
- D = 1024: capacidad práctica ≈ 30-50 asociaciones (teórica: 51)
- Punto de quiebre (error > 5%) ocurre cerca de la capacidad teórica
- Degradación gradual, no abrupta

**Conclusión:** Para secuencias codificadas con bundles:
- D = 1024: hasta ~25 caracteres
- D = 4096: hasta ~85 caracteres
- D = 10000: hasta ~188 caracteres

**Informe completo:** [docs/experimentos/hdc_esp_003_resultados.md](docs/experimentos/hdc_esp_003_resultados.md)

---

#### MORPH-001: Clasificación Morfológica con HDC (Fase 1)

**Estado:** Completado ✓  
**Fecha:** 2026-10-17  
**Corpus:** UD Spanish-AnCora (453k tokens train, 54k test)

**Hallazgos:**
- ⚠️ **HDC NO es competitivo** con baselines simples para morfología
- HDC accuracy: 56-67% vs Suffix-3: 81-99%
- Causa: saturación de bundles (excediendo capacidad 17×-66×)
- La morfología del español es altamente regular (81-99% con sufijos de 3 chars)

**Conclusión:** HDC con bundles acumulativos falla para clasificación morfológica. Se requieren métodos alternativos (k-NN, contexto, integración con SNN).

**Informe completo:** [docs/experimentos/morph_001_resultados.md](docs/experimentos/morph_001_resultados.md)  
**Resumen ejecutivo:** [docs/fase_1_resumen_ejecutivo.md](docs/fase_1_resumen_ejecutivo.md)

### Instalación

```bash
pip install -r requirements.txt
```

### Ejecución de pruebas

```bash
# Pruebas unitarias del módulo HDC
python tests/test_hdc.py

# Experimento HDC-Esp-001
python experiments/hdc_esp_001/run_experiment.py
```

### Documentación

- [Evaluación inicial crítica](docs/00_evaluacion_inicial_critica.md) — Análisis de supuestos y riesgos
- [Protocolo de investigación](docs/01_protocolo_investigacion.md) — Metodología y criterios (pendiente)
- [Resultados de experimentos](docs/experimentos/) — Informes detallados

### Hipótesis en evaluación

- **H1 — Representación:** Las operaciones HDC/VSA permiten codificar y recuperar asociaciones lingüísticas con tasa de error compatible con las tareas evaluadas.
- **H2 — Temporalidad:** Una SNN puede aprovechar patrones de disparo temporales para modelar dependencias de secuencias.
- **H3 — Aprendizaje local:** Las reglas de plasticidad sináptica mejoran el aprendizaje incremental.
- **H4 — Inferencia activa:** Un modelo generativo probabilístico puede mejorar la adquisición de información.
- **H5 — Integración:** El sistema híbrido mejora al menos una métrica principal sin degradar las restantes.
- **H6 — Eficiencia:** Las ventajas de energía o latencia persisten al incluir todos los costes relevantes.

### Principios científicos

1. **Evidencia sobre consenso:** Ninguna hipótesis se convierte en resultado por acuerdo entre agentes.
2. **Reproducibilidad:** Todo experimento debe ser reproducible con configuración registrada.
3. **Separación de capacidades:** No confundir aprender asociaciones con comprender lenguaje.
4. **Líneas base claras:** Comparar con modelos de referencia (n-gramas, GRU, Transformer).
5. **Ablaciones sistemáticas:** Identificar qué componente aporta cada mejora.
6. **Clasificación de evidencia:** Distinguir entre resultados reproducidos (E1), publicados (E2), fundamentos formales (E3), hipótesis teóricas (E4) y especulaciones (E5).

### Referencias principales

1. Kanerva, P. (2009). Hyperdimensional Computing. *Cognitive Computation*, 1(2), 139-159.
2. Friston, K. (2010). The free-energy principle. *Nature Reviews Neuroscience*, 11(2), 127-138.
3. Universal Dependencies. UD Spanish-AnCora. https://universaldependencies.org/treebanks/es_ancora/
4. Intel. Loihi 2. https://www.intel.com/content/www/us/en/research/neuromorphic-computing-loihi-2-technology-brief.html
5. Gonzalez, H. A., et al. (2024). SpiNNaker 2. arXiv:2401.04491.

### Licencia

Este proyecto es de código abierto y está disponible bajo los términos de la licencia del repositorio.

### Contacto

Para más información sobre este laboratorio, consulta la conversación original en Arena.ai.

---

**Última actualización:** 2026-10-17  
**Versión:** 0.1.0  
**Estado:** Investigación activa
=======
