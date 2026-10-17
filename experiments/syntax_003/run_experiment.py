"""
Experimento SYNTAX-003: Análisis de Dependencias Completo (LAS)
================================================================

Combina SYNTAX-001 (cabeza) + SYNTAX-002 (relación) para calcular LAS.
"""

import sys
import os
import time
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple
from conllu import parse_incr


def load_conllu_sentences(conllu_path: str) -> List[List[dict]]:
    sentences = []
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            tokens = [t for t in sentence if not isinstance(t['id'], tuple)]
            sentence_data = []
            for token in tokens:
                sentence_data.append({
                    'word': token['form'], 'lemma': token['lemma'],
                    'pos': token['upos'], 'feats': token['feats'] or {},
                    'head': token['head'], 'deprel': token['deprel'],
                })
            if sentence_data:
                sentences.append(sentence_data)
    return sentences


# ============ Modelo de cabezas (de SYNTAX-001) ============

class DependencyParser:
    def __init__(self, suffix_lengths=[1,2,3,4], max_head_distance=15):
        self.suffix_lengths = suffix_lengths
        self.max_head_distance = max_head_distance
        self.head_distance_lookup = {}
        self.fallback_lookup = {}
        self.global_mode = "1"
    
    def _make_suffix_key(self, word):
        word_lower = word.lower()
        parts = []
        for length in self.suffix_lengths:
            if len(word_lower) >= length:
                parts.append(f"w{length}:{word_lower[-length:]}")
        return ",".join(parts)
    
    def _make_contextual_key(self, sentence, idx):
        token = sentence[idx]
        word = token['word']; pos = token['pos']
        suffix_key = self._make_suffix_key(word)
        left_pos = sentence[idx-1]['pos'] if idx > 0 else "<BOS>"
        right_pos = sentence[idx+1]['pos'] if idx < len(sentence)-1 else "<EOS>"
        left2_pos = sentence[idx-2]['pos'] if idx > 1 else "<BOS>"
        right2_pos = sentence[idx+2]['pos'] if idx < len(sentence)-2 else "<EOS>"
        rel_pos = idx / max(1, len(sentence)-1)
        pos_bucket = "start" if rel_pos < 0.2 else "end" if rel_pos > 0.8 else "mid"
        word_len = len(word)
        len_bucket = "short" if word_len <= 3 else "med" if word_len <= 7 else "long"
        return f"{suffix_key} | pos:{pos} | lp:{left_pos} | rp:{right_pos} | l2p:{left2_pos} | r2p:{right2_pos} | pos:{pos_bucket} | len:{len_bucket}"
    
    def _make_fallback_key(self, sentence, idx):
        token = sentence[idx]
        word = token['word'].lower(); pos = token['pos']
        suf3 = word[-3:] if len(word) >= 3 else word
        left_pos = sentence[idx-1]['pos'] if idx > 0 else "<BOS>"
        return f"pos:{pos} | suf3:{suf3} | lp:{left_pos}"
    
    def train(self, sentences):
        ctx = defaultdict(Counter); fb = defaultdict(Counter); glob = Counter()
        for sent in sentences:
            for i, token in enumerate(sent):
                head_idx = token['head']
                if head_idx == 0:
                    rel_dist = "ROOT"
                else:
                    rel_dist = (head_idx - 1) - i
                    if abs(rel_dist) > self.max_head_distance:
                        rel_dist = f"{'>' if rel_dist > 0 else '<'}{self.max_head_distance}"
                    rel_dist = str(rel_dist)
                ctx[self._make_contextual_key(sent, i)][rel_dist] += 1
                fb[self._make_fallback_key(sent, i)][rel_dist] += 1
                glob[rel_dist] += 1
        self.head_distance_lookup = {k: c.most_common(1)[0][0] for k, c in ctx.items()}
        self.fallback_lookup = {k: c.most_common(1)[0][0] for k, c in fb.items()}
        self.global_mode = glob.most_common(1)[0][0]
        print(f"  Head parser: {len(self.head_distance_lookup):,} claves")
    
    def predict_head(self, sentence, idx):
        ctx_key = self._make_contextual_key(sentence, idx)
        if ctx_key in self.head_distance_lookup:
            return self._dist_to_head(idx, self.head_distance_lookup[ctx_key], len(sentence))
        fb_key = self._make_fallback_key(sentence, idx)
        if fb_key in self.fallback_lookup:
            return self._dist_to_head(idx, self.fallback_lookup[fb_key], len(sentence))
        return self._dist_to_head(idx, self.global_mode, len(sentence))
    
    def _dist_to_head(self, idx, pred_dist, sent_len):
        if pred_dist == "ROOT": return 0
        try:
            dist = int(pred_dist)
        except ValueError:
            dist = self.max_head_distance if pred_dist.startswith(">") else -self.max_head_distance
        head_0 = idx + dist
        if head_0 < 0 or head_0 >= sent_len: return 0
        return head_0 + 1
    
    def parse_sentence(self, sentence):
        return [self.predict_head(sentence, i) for i in range(len(sentence))]


# ============ Modelo de relaciones (de SYNTAX-002) ============

def extract_dep_pairs(sentences):
    pairs = []
    for sent in sentences:
        for i, token in enumerate(sent):
            head_idx = token['head']
            if head_idx == 0:
                hw, hp = "<ROOT>", "<ROOT>"
                dist, d = i+1, "root"
            else:
                h0 = head_idx - 1
                hw = sent[h0]['word']; hp = sent[h0]['pos']
                dist = abs(h0 - i); d = "left" if h0 < i else "right"
            dw = token['word'].lower()
            pairs.append({
                'dep_word': token['word'], 'dep_pos': token['pos'],
                'dep_suf2': dw[-2:] if len(dw)>=2 else dw,
                'dep_suf3': dw[-3:] if len(dw)>=3 else dw,
                'head_word': hw, 'head_pos': hp,
                'distance': dist, 'direction': d,
                'dep_left_pos': sent[i-1]['pos'] if i>0 else "<BOS>",
                'head_left_pos': sent[head_idx-2]['pos'] if head_idx>1 else "<BOS>",
                'deprel': token['deprel'],
            })
    return pairs


class DeprelClassifier:
    def __init__(self):
        self.ctx = {}; self.fb = {}; self.pp = {}; self.gm = ""
    
    def _full_key(self, p):
        d = min(p['distance'], 10)
        return f"dp:{p['dep_pos']}|hp:{p['head_pos']}|s3:{p['dep_suf3']}|d:{p['direction']}|dist:{d}|dlp:{p['dep_left_pos']}"
    
    def _fb_key(self, p):
        return f"dp:{p['dep_pos']}|hp:{p['head_pos']}|d:{p['direction']}"
    
    def _pp_key(self, p):
        return f"{p['dep_pos']}_{p['head_pos']}"
    
    def train(self, pairs):
        fc = defaultdict(Counter); fbc = defaultdict(Counter); ppc = defaultdict(Counter); gc = Counter()
        for p in pairs:
            r = p['deprel']
            fc[self._full_key(p)][r] += 1; fbc[self._fb_key(p)][r] += 1
            ppc[self._pp_key(p)][r] += 1; gc[r] += 1
        self.ctx = {k: c.most_common(1)[0][0] for k, c in fc.items()}
        self.fb = {k: c.most_common(1)[0][0] for k, c in fbc.items()}
        self.pp = {k: c.most_common(1)[0][0] for k, c in ppc.items()}
        self.gm = gc.most_common(1)[0][0]
        print(f"  Deprel classifier: {len(self.ctx):,} claves")
    
    def predict(self, p):
        fk = self._full_key(p)
        if fk in self.ctx: return self.ctx[fk]
        fbk = self._fb_key(p)
        if fbk in self.fb: return self.fb[fbk]
        ppk = self._pp_key(p)
        if ppk in self.pp: return self.pp[ppk]
        return self.gm


# ============ Pipeline completo ============

def evaluate_las(head_parser, deprel_clf, sentences):
    uas_ok = las_ok = total = sent_em = n_sent = 0
    
    for sent in sentences:
        heads = head_parser.parse_sentence(sent)
        sent_ok = True
        
        for i, token in enumerate(sent):
            true_h = token['head']; true_r = token['deprel']
            pred_h = heads[i]
            total += 1
            
            if pred_h == 0:
                hw, hp = "<ROOT>", "<ROOT>"; dist, d = i+1, "root"
            else:
                h0 = pred_h - 1
                if h0 < len(sent):
                    hw = sent[h0]['word']; hp = sent[h0]['pos']
                else:
                    hw, hp = "<UNK>", "<UNK>"
                dist = abs(h0 - i); d = "left" if h0 < i else "right"
            
            dw = token['word'].lower()
            pair = {
                'dep_word': token['word'], 'dep_pos': token['pos'],
                'dep_suf2': dw[-2:] if len(dw)>=2 else dw,
                'dep_suf3': dw[-3:] if len(dw)>=3 else dw,
                'head_word': hw, 'head_pos': hp,
                'distance': dist, 'direction': d,
                'dep_left_pos': sent[i-1]['pos'] if i>0 else "<BOS>",
                'head_left_pos': sent[pred_h-2]['pos'] if pred_h>1 else "<BOS>",
                'deprel': '',
            }
            pred_r = deprel_clf.predict(pair)
            
            if pred_h == true_h:
                uas_ok += 1
                if pred_r == true_r:
                    las_ok += 1
                else:
                    sent_ok = False
            else:
                sent_ok = False
        
        if sent_ok: sent_em += 1
        n_sent += 1
    
    return {
        'uas': uas_ok/total, 'las': las_ok/total,
        'exact_match': sent_em/n_sent,
        'total': total, 'uas_correct': uas_ok, 'las_correct': las_ok,
        'n_sentences': n_sent, 'sentence_exact_match': sent_em,
    }


# ============ Experimento ============

def run_full_experiment(corpus_dir=None, output_dir="."):
    if corpus_dir is None:
        corpus_dir = "/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora"
    
    print("="*70)
    print("SYNTAX-003: Análisis de Dependencias Completo (UAS + LAS)")
    print("="*70)
    print(f"Corpus: UD Spanish-AnCora")
    print(f"Fecha: {datetime.now().isoformat()}")
    
    corpus_path = Path(corpus_dir)
    print("\nCargando oraciones...")
    train_s = load_conllu_sentences(corpus_path / "es_ancora-ud-train.conllu")
    test_s = load_conllu_sentences(corpus_path / "es_ancora-ud-test.conllu")
    print(f"  Train: {len(train_s):,} | Test: {len(test_s):,}")
    
    # Entrenar parser de cabezas
    print(f"\n{'─'*70}")
    print("Entrenando parser de cabezas (SYNTAX-001)")
    print(f"{'─'*70}")
    hp = DependencyParser()
    t0 = time.time(); hp.train(train_s); t_head = time.time()-t0
    print(f"  Tiempo: {t_head:.1f}s")
    
    # Entrenar clasificador de relaciones
    print(f"\n{'─'*70}")
    print("Entrenando clasificador de relaciones (SYNTAX-002)")
    print(f"{'─'*70}")
    train_pairs = extract_dep_pairs(train_s)
    dc = DeprelClassifier()
    t0 = time.time(); dc.train(train_pairs); t_deprel = time.time()-t0
    print(f"  Tiempo: {t_deprel:.1f}s")
    
    # Evaluar pipeline
    print(f"\n{'─'*70}")
    print("Evaluación del pipeline completo (LAS)")
    print(f"{'─'*70}")
    t0 = time.time()
    r = evaluate_las(hp, dc, test_s)
    t_eval = time.time()-t0
    
    print(f"  UAS:            {r['uas']:.2%}")
    print(f"  LAS:            {r['las']:.2%}")
    print(f"  Exact Match:    {r['exact_match']:.2%}")
    print(f"  UAS correctos:  {r['uas_correct']:,}/{r['total']:,}")
    print(f"  LAS correctos:  {r['las_correct']:,}/{r['total']:,}")
    print(f"  Oraciones OK:   {r['sentence_exact_match']}/{r['n_sentences']:,}")
    print(f"  Tiempo eval:    {t_eval:.2f}s")
    print(f"  Brecha UAS-LAS: {r['uas']-r['las']:.2%}")
    
    print(f"\n{'='*70}")
    print("RESUMEN COMPARATIVO")
    print(f"{'='*70}")
    print(f"{'Métrica':<25} {'Valor':>10} {'Objetivo':>12}")
    print("-"*50)
    print(f"{'UAS':<25} {r['uas']:>10.2%} {'60-70%':>12}")
    print(f"{'LAS':<25} {r['las']:>10.2%} {'55-65%':>12}")
    print(f"{'Exact Match':<25} {r['exact_match']:>10.2%} {'—':>12}")
    
    data = {
        "experiment_id": "SYNTAX-003",
        "description": "Análisis de dependencias completo (UAS + LAS)",
        "corpus": "UD Spanish-AnCora",
        "timestamp": datetime.now().isoformat(),
        "results": r,
    }
    output_path = os.path.join(output_dir, "syntax_003_results.json")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en: {output_path}")
    return data


if __name__ == "__main__":
    run_full_experiment(output_dir=os.path.dirname(os.path.abspath(__file__)))
