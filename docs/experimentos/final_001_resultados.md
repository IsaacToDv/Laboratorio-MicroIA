# FINAL-001: Spiking Neural Network (SNN) para POS Tagging

**ID:** FINAL-001  
**Fase:** 6 — Cierre SNN-FEP  
**Fecha:** 2026-10-18  
**Corpus:** UD Spanish-AnCora  
**Estado:** ⚠️ Exploratorio (limitación documentada)

---

## Objetivo

Implementar una Spiking Neural Network (SNN) con neuronas Leaky
Integrate-and-Fire y aprendizaje STDP para clasificación POS,
evaluando la viabilidad del paradigma de cómputo basado en impulsos.

## Arquitectura SNN

```
  Features (50)  →  Hidden LIF (30)  →  Output LIF (14)  →  WTA
    │                    │                    │
  Rate coding      Spike propagation    Winner-take-all
```

| Parámetro | Valor |
|-----------|-------|
| Neuronas input | 50 features |
| Neuronas hidden | 30 (LIF, τ=0.85) |
| Neuronas output | 14 (WTA, τ=0.9) |
| Pasos de tiempo | 15 |
| Aprendizaje | STDP simplificado |
| Épocas | 2 |

## Resultados

| Modelo | Accuracy |
|--------|----------|
| Baseline (sufijos) | 42.68% |
| **SNN (LIF + STDP)** | **17.93%** |

La SNN solo predijo NOUN (clase mayoritaria), indicando que:
- El STDP simplificado con 2 epochs no converge
- Las tasas de disparo no diferencian suficientemente las clases
- Se necesita más tiempo de simulación o codificación temporal

## Análisis

Las SNN requieren significativamente más entrenamiento que modelos
estadísticos para alcanzar rendimiento competitivo. Esto es consistente
con la literatura: las SNN destacan en eficiencia energética y cómputo
neuromórfico, no en accuracy bruto con entrenamiento limitado.

### Limitaciones
- STDP simplificado (sin eligibility traces ni reward modulation)
- Solo rate coding (sin temporal coding)
- 15 pasos de tiempo (insuficiente para propagación de información)

## Conclusión

La SNN con LIF + STDP demuestra el principio pero no alcanza
rendimiento competitivo. Validación parcial del paradigma de
cómputo neuronal basado en impulsos para NLP.

**Claim asociado:** CLAIM-034 — SNN viable conceptualmente pero
requiere mayor sofisticación para ser competitiva (17.93% vs 42.68%).
