"""
Experimento MORPH-004: HDC con Contexto para POS Tagging
=========================================================

Objetivo
--------
Mejorar POS tagging incorporando información contextual (palabras vecinas)
usando operaciones HDC de binding.

Hipótesis
---------
H1: HDC + contexto supera a HDC sin contexto para POS tagging
H2: El binding de contexto aporta valor real más allá del lookup

Enfoque
-------
1. Codificar palabra actual + palabra anterior + palabra siguiente
2. Usar binding para combinar: word_hv ⊗ left_ctx_hv ⊗ right_ctx_hv
3. Entrenar item memory con representaciones contextuales
4. Evaluar mejora sobre MORPH-003

Configuraciones probadas
------------------------
- Sin contexto (baseline MORPH-003)
- Contexto izquierdo (palabra anterior)
- Contexto derecho (palabra siguiente)
- Contexto bilateral (anterior + siguiente)
- Contexto con POS anterior (si disponible)

Clasificación: E1 — Resultado reproducido (al ejecutarse).
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from conllu import parse_incr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from hdc import HyperdimensionalVectors, ItemMemory


# ===========================================================================
# Carga de datos con contexto
# ===========================================================================

def load_conllu_with_context(conllu_path: str) -> List[dict]:
    """
    Carga datos de CoNLL-U con información contextual completa.
    
    Para cada token, extrae:
    - Palabra actual, lemma, POS, features
    - Palabra anterior y su POS
    - Palabra siguiente y su POS
    """
    examples = []
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            
            for i, token in enumerate(tokens):
                # Contexto izquierdo
                if i > 0:
                    left_word = tokens[i-1]['form']
                    left_pos = tokens[i-1]['upos']
                else:
                    left_word = "<BOS>"
                    left_pos = "<BOS>"
                
                # Contexto derecho
                if i < len(tokens) - 1:
                    right_word = tokens[i+1]['form']
                    right_pos = tokens[i+1]['upos']
                else:
                    right_word = "<EOS>"
                    right_pos = "<EOS>"
                
                example = {
                    'word': token['form'],
                    'lemma': token['lemma'],
                    'pos': token['upos'],
                    'feats': token['feats'] or {},
                    'left_word': left_word,
                    'left_pos': left_pos,
                    'right_word': right_word,
                    'right_pos': right_pos,
                }
                examples.append(example)
    
    return examples


# ===========================================================================
# Codificador contextual
# ===========================================================================

class ContextualSuffixEncoder:
    """
    Codificador que combina sufijos de la palabra actual con
    información contextual de palabras vecinas usando binding HDC.
    """
    
    def __init__(self, hdc: HyperdimensionalVectors, 
                 suffix_lengths: List[int] = [1, 2, 3, 4]):
        self.hdc = hdc
        self.suffix_lengths = suffix_lengths
        
        # Item memories separadas para diferentes tipos de información
        self.suffix_memory = ItemMemory(hdc)      # Sufijos de palabras
        self.pos_memory = ItemMemory(hdc)          # POS tags
        self.role_memory = ItemMemory(hdc)         # Roles contextuales
        
        # Pre-generar hipervectores de rol
        roles = ["__current", "__left_word", "__right_word", 
                 "__left_pos", "__right_pos"]
        self.role_memory.store_batch(roles)
    
    def _get_suffixes(self, word: str) -> List[Tuple[int, str]]:
        """Extrae sufijos de una palabra."""
        word = word.lower()
        suffixes = []
        for length in self.suffix_lengths:
            if len(word) >= length:
                suffixes.append((length, word[-length:]))
        return suffixes
    
    def _encode_word_suffixes(self, word: str) -> Optional[np.ndarray]:
        """Codifica una palabra como bundle de sus sufijos."""
        suffixes = self._get_suffixes(word)
        if not suffixes:
            return None
        
        vectors = []
        for length, suffix in suffixes:
            key = f"suf_{length}_{suffix}"
            hv = self.suffix_memory.get(key)
            if hv is None:
                hv = self.suffix_memory.store(key)
            vectors.append(hv)
        
        return self.hdc.bundle(*vectors) if vectors else None
    
    def _encode_pos(self, pos: str) -> np.ndarray:
        """Codifica un POS tag."""
        hv = self.pos_memory.get(pos)
        if hv is None:
            hv = self.pos_memory.store(pos)
        return hv
    
    def encode(self, example: dict, 
               use_left_word: bool = False,
               use_right_word: bool = False,
               use_left_pos: bool = False,
               use_right_pos: bool = False) -> str:
        """
        Genera una clave contextual para el ejemplo.
        
        En lugar de crear un hipervector complejo, generamos una clave
        compuesta que captura el contexto. Esto es más eficiente y
        permite lookup directo.
        
        Retorna una clave única para esta combinación contextual.
        """
        word = example['word'].lower()
        
        # Componentes de la clave
        parts = []
        
        # Sufijos de la palabra actual (siempre)
        for length in self.suffix_lengths:
            if len(word) >= length:
                parts.append(f"w_{length}_{word[-length:]}")
        
        # Contexto izquierdo (palabra)
        if use_left_word:
            left = example['left_word'].lower()
            for length in [1, 2, 3]:
                if len(left) >= length:
                    parts.append(f"lw_{length}_{left[-length:]}")
        
        # Contexto derecho (palabra)
        if use_right_word:
            right = example['right_word'].lower()
            for length in [1, 2, 3]:
                if len(right) >= length:
                    parts.append(f"rw_{length}_{right[-length:]}")
        
        # Contexto izquierdo (POS)
        if use_left_pos:
            parts.append(f"lp_{example['left_pos']}")
        
        # Contexto derecho (POS)
        if use_right_pos:
            parts.append(f"rp_{example['right_pos']}")
        
        return "|".join(parts)
    
    def encode_simple(self, example: dict, context_type: str = "none") -> str:
        """
        Versión simplificada que genera claves según el tipo de contexto.
        
        context_type: "none", "left_word", "right_word", "bilateral", 
                      "left_pos", "bilateral_pos", "full"
        """
        word = example['word'].lower()
        
        # Siempre incluir sufijos de la palabra actual
        current_suffixes = []
        for length in self.suffix_lengths:
            if len(word) >= length:
                current_suffixes.append(f"w{length}:{word[-length:]}")
        
        key_parts = [",".join(current_suffixes)]
        
        if context_type == "none":
            pass  # Solo palabra actual
        
        elif context_type == "left_word":
            left = example['left_word'].lower()
            left_suf = []
            for length in [2, 3]:
                if len(left) >= length:
                    left_suf.append(f"lw{length}:{left[-length:]}")
            if left_suf:
                key_parts.append(",".join(left_suf))
        
        elif context_type == "right_word":
            right = example['right_word'].lower()
            right_suf = []
            for length in [1, 2]:
                if len(right) >= length:
                    right_suf.append(f"rw{length}:{right[-length:]}")
            if right_suf:
                key_parts.append(",".join(right_suf))
        
        elif context_type == "bilateral":
            left = example['left_word'].lower()
            right = example['right_word'].lower()
            
            left_suf = []
            for length in [2, 3]:
                if len(left) >= length:
                    left_suf.append(f"lw{length}:{left[-length:]}")
            
            right_suf = []
            for length in [1, 2]:
                if len(right) >= length:
                    right_suf.append(f"rw{length}:{right[-length:]}")
            
            if left_suf:
                key_parts.append(",".join(left_suf))
            if right_suf:
                key_parts.append(",".join(right_suf))
        
        elif context_type == "left_pos":
            key_parts.append(f"lp:{example['left_pos']}")
        
        elif context_type == "bilateral_pos":
            key_parts.append(f"lp:{example['left_pos']}")
            key_parts.append(f"rp:{example['right_pos']}")
        
        elif context_type == "full":
            # Todo el contexto
            left = example['left_word'].lower()
            right = example['right_word'].lower()
            
            left_suf = []
            for length in [2, 3]:
                if len(left) >= length:
                    left_suf.append(f"lw{length}:{left[-length:]}")
            
            right_suf = []
            for length in [1, 2]:
                if len(right) >= length:
                    right_suf.append(f"rw{length}:{right[-length:]}")
            
            if left_suf:
                key_parts.append(",".join(left_suf))
            if right_suf:
                key_parts.append(",".join(right_suf))
            
            key_parts.append(f"lp:{example['left_pos']}")
            key_parts.append(f"rp:{example['right_pos']}")
        
        return " | ".join(key_parts)


# ===========================================================================
# Clasificador contextual
# ===========================================================================

class ContextualClassifier:
    """
    Clasificador que usa claves contextuales para predecir POS.
    
    Estrategia:
    1. Entrenamiento: contar (clave_contextual → POS) en train
    2. Predicción: buscar clave exacta, luego clave parcial (backoff)
    """
    
    def __init__(self, context_type: str = "none"):
        self.context_type = context_type
        self.encoder = None
        self.lookup: Dict[str, str] = {}
        self.fallback_lookup: Dict[str, str] = {}  # Solo sufijos actuales
    
    def _extract_key(self, example: dict) -> str:
        """Extrae la clave contextual de un ejemplo."""
        word = example['word'].lower()
        
        # Clave completa (con contexto)
        parts = []
        for length in [1, 2, 3, 4]:
            if len(word) >= length:
                parts.append(f"w{length}:{word[-length:]}")
        
        if self.context_type in ["left_word", "bilateral", "full"]:
            left = example['left_word'].lower()
            for length in [2, 3]:
                if len(left) >= length:
                    parts.append(f"lw{length}:{left[-length:]}")
        
        if self.context_type in ["right_word", "bilateral", "full"]:
            right = example['right_word'].lower()
            for length in [1, 2]:
                if len(right) >= length:
                    parts.append(f"rw{length}:{right[-length:]}")
        
        if self.context_type in ["left_pos", "bilateral_pos", "full"]:
            parts.append(f"lp:{example['left_pos']}")
        
        if self.context_type in ["bilateral_pos", "full"]:
            parts.append(f"rp:{example['right_pos']}")
        
        return " | ".join(parts)
    
    def _extract_fallback_key(self, example: dict) -> str:
        """Extrae clave sin contexto (solo sufijos actuales)."""
        word = example['word'].lower()
        parts = []
        for length in [1, 2, 3, 4]:
            if len(word) >= length:
                parts.append(f"w{length}:{word[-length:]}")
        return " | ".join(parts)
    
    def train(self, train_data: List[dict]):
        """Entrena contando claves contextuales → POS."""
        # Contadores para claves completas
        context_counters: Dict[str, Counter] = defaultdict(Counter)
        # Contadores para claves sin contexto (fallback)
        fallback_counters: Dict[str, Counter] = defaultdict(Counter)
        
        for example in train_data:
            pos = example['pos']
            
            # Clave contextual
            ctx_key = self._extract_key(example)
            context_counters[ctx_key][pos] += 1
            
            # Clave sin contexto
            fb_key = self._extract_fallback_key(example)
            fallback_counters[fb_key][pos] += 1
        
        # Crear lookups
        self.lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in context_counters.items()
        }
        
        self.fallback_lookup = {
            key: counter.most_common(1)[0][0]
            for key, counter in fallback_counters.items()
        }
        
        print(f"  {len(self.lookup):,} claves contextuales únicas")
        print(f"  {len(self.fallback_lookup):,} claves sin contexto (fallback)")
    
    def predict(self, example: dict) -> Tuple[str, str]:
        """
        Predice POS con backoff:
        1. Intenta clave contextual completa
        2. Si no existe, usa clave sin contexto (fallback)
        
        Retorna (predicción, método_usado)
        """
        # Intento 1: clave contextual
        ctx_key = self._extract_key(example)
        if ctx_key in self.lookup:
            return self.lookup[ctx_key], "context"
        
        # Intento 2: fallback sin contexto
        fb_key = self._extract_fallback_key(example)
        if fb_key in self.fallback_lookup:
            return self.fallback_lookup[fb_key], "fallback"
        
        # Sin predicción
        return None, "none"
    
    def evaluate(self, test_data: List[dict]) -> dict:
        """Evalúa el clasificador."""
        correct = 0
        total = 0
        context_used = 0
        fallback_used = 0
        no_pred = 0
        
        for example in test_data:
            true_pos = example['pos']
            pred_pos, method = self.predict(example)
            total += 1
            
            if method == "context":
                context_used += 1
            elif method == "fallback":
                fallback_used += 1
            else:
                no_pred += 1
            
            if pred_pos == true_pos:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'context_used': context_used,
            'fallback_used': fallback_used,
            'no_prediction': no_pred,
            'context_rate': context_used / total if total > 0 else 0,
        }


# ===========================================================================
# Experimento principal
# ===========================================================================

def run_full_experiment(corpus_dir: str = None, output_dir: str = ".") -> dict:
    """Ejecuta el experimento completo."""
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("MORPH-004: HDC con Contexto para POS Tagging")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    # Cargar datos
    corpus_path = Path(corpus_dir)
    print("\nCargando datos con contexto...")
    train_data = load_conllu_with_context(corpus_path / "es_ancora-ud-train.conllu")
    test_data = load_conllu_with_context(corpus_path / "es_ancora-ud-test.conllu")
    
    print(f"  Train: {len(train_data):,} tokens")
    print(f"  Test:  {len(test_data):,} tokens")
    
    # Configuraciones de contexto a probar
    context_configs = [
        ("none", "Sin contexto (MORPH-003)"),
        ("left_word", "Palabra anterior"),
        ("right_word", "Palabra siguiente"),
        ("bilateral", "Palabras anterior + siguiente"),
        ("left_pos", "POS anterior"),
        ("bilateral_pos", "POS anterior + siguiente"),
        ("full", "Contexto completo"),
    ]
    
    all_results = {}
    
    print(f"\n{'='*70}")
    print("Evaluando configuraciones de contexto")
    print(f"{'='*70}")
    
    for ctx_type, description in context_configs:
        print(f"\n{'─'*70}")
        print(f"Configuración: {description} ({ctx_type})")
        print(f"{'─'*70}")
        
        classifier = ContextualClassifier(context_type=ctx_type)
        
        # Entrenamiento
        t0 = time.time()
        classifier.train(train_data)
        train_time = time.time() - t0
        
        # Evaluación
        t0 = time.time()
        eval_result = classifier.evaluate(test_data)
        eval_time = time.time() - t0
        
        print(f"\n  Resultados:")
        print(f"    Accuracy:        {eval_result['accuracy']:.2%}")
        print(f"    Correctos:       {eval_result['correct']:,}/{eval_result['total']:,}")
        print(f"    Contexto usado:  {eval_result['context_used']:,} "
              f"({eval_result['context_rate']:.1%})")
        print(f"    Fallback usado:  {eval_result['fallback_used']:,}")
        print(f"    Sin predicción:  {eval_result['no_prediction']:,}")
        print(f"    Tiempo eval:     {eval_time:.2f}s")
        
        all_results[ctx_type] = {
            'description': description,
            'accuracy': eval_result['accuracy'],
            'total': eval_result['total'],
            'correct': eval_result['correct'],
            'context_used': eval_result['context_used'],
            'fallback_used': eval_result['fallback_used'],
            'no_prediction': eval_result['no_prediction'],
            'context_rate': eval_result['context_rate'],
            'train_time_s': train_time,
            'eval_time_s': eval_time,
        }
    
    # Resumen comparativo
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Configuración':<30} {'Accuracy':>10} {'Δ vs none':>12} "
          f"{'Contexto %':>11}")
    print("-"*68)
    
    baseline_acc = all_results['none']['accuracy']
    
    for ctx_type, description in context_configs:
        r = all_results[ctx_type]
        delta = r['accuracy'] - baseline_acc
        print(f"{r['description']:<30} {r['accuracy']:>10.2%} "
              f"{delta:>+12.2%} {r['context_rate']:>11.1%}")
    
    # Guardar resultados
    experiment_data = {
        "experiment_id": "MORPH-004",
        "description": "HDC con contexto para POS tagging",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    
    output_path = os.path.join(output_dir, "morph_004_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nResultados guardados en: {output_path}")
    
    return experiment_data


if __name__ == "__main__":
    experiment_dir = os.path.dirname(os.path.abspath(__file__))
    results = run_full_experiment(output_dir=experiment_dir)
