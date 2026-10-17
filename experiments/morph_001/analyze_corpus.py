"""
Análisis de UD Spanish-AnCora para Fase 1 — Morfología
=======================================================

Extrae estadísticas del corpus para diseñar experimentos de clasificación morfológica.
"""

import sys
from pathlib import Path
from collections import Counter
from conllu import parse_incr

def analyze_corpus(conllu_path: str):
    """Analiza un archivo CoNLL-U y extrae estadísticas."""
    print(f"\n{'='*70}")
    print(f"Analizando: {Path(conllu_path).name}")
    print(f"{'='*70}")
    
    n_sentences = 0
    n_tokens = 0
    
    pos_counter = Counter()
    feats_counter = Counter()
    lemma_counter = Counter()
    word_counter = Counter()
    
    # Features específicas
    gender_counter = Counter()
    number_counter = Counter()
    person_counter = Counter()
    tense_counter = Counter()
    mood_counter = Counter()
    
    with open(conllu_path, 'r', encoding='utf-8') as f:
        for sentence in parse_incr(f):
            n_sentences += 1
            
            for token in sentence:
                # Saltar tokens multi-palabra
                if isinstance(token['id'], tuple):
                    continue
                
                n_tokens += 1
                
                # POS
                pos = token['upos']
                pos_counter[pos] += 1
                
                # Lema
                lemma = token['lemma']
                lemma_counter[lemma] += 1
                
                # Palabra
                word = token['form']
                word_counter[word] += 1
                
                # Features morfológicas
                feats = token['feats']
                if feats:
                    for feat, value in feats.items():
                        feats_counter[f"{feat}={value}"] += 1
                        
                        if feat == 'Gender':
                            gender_counter[value] += 1
                        elif feat == 'Number':
                            number_counter[value] += 1
                        elif feat == 'Person':
                            person_counter[value] += 1
                        elif feat == 'Tense':
                            tense_counter[value] += 1
                        elif feat == 'Mood':
                            mood_counter[value] += 1
    
    print(f"\nEstadísticas generales:")
    print(f"  Oraciones: {n_sentences:,}")
    print(f"  Tokens:    {n_tokens:,}")
    print(f"  Lemas únicos:  {len(lemma_counter):,}")
    print(f"  Palabras únicas: {len(word_counter):,}")
    
    print(f"\nCategorías gramaticales (POS):")
    for pos, count in pos_counter.most_common(15):
        pct = count / n_tokens * 100
        print(f"  {pos:<12} {count:>8,} ({pct:5.2f}%)")
    
    print(f"\nFeatures morfológicas principales:")
    print(f"  Género:")
    for gender, count in gender_counter.most_common():
        print(f"    {gender:<10} {count:>8,}")
    
    print(f"  Número:")
    for number, count in number_counter.most_common():
        print(f"    {number:<10} {count:>8,}")
    
    print(f"  Persona:")
    for person, count in person_counter.most_common():
        print(f"    {person:<10} {count:>8,}")
    
    print(f"  Tiempo verbal:")
    for tense, count in tense_counter.most_common(10):
        print(f"    {tense:<10} {count:>8,}")
    
    print(f"  Modo:")
    for mood, count in mood_counter.most_common():
        print(f"    {mood:<10} {count:>8,}")
    
    print(f"\nLemas más frecuentes:")
    for lemma, count in lemma_counter.most_common(20):
        print(f"  {lemma:<20} {count:>6,}")
    
    return {
        'n_sentences': n_sentences,
        'n_tokens': n_tokens,
        'n_lemmas': len(lemma_counter),
        'n_words': len(word_counter),
        'pos_distribution': dict(pos_counter),
        'gender_distribution': dict(gender_counter),
        'number_distribution': dict(number_counter),
        'person_distribution': dict(person_counter),
        'tense_distribution': dict(tense_counter),
        'mood_distribution': dict(mood_counter),
    }


if __name__ == "__main__":
    corpus_dir = Path("/home/user/Laboratorio-MicroIA/data/corpus/UD_Spanish-AnCora")
    
    print("="*70)
    print("Análisis de UD Spanish-AnCora")
    print("="*70)
    
    train_stats = analyze_corpus(corpus_dir / "es_ancora-ud-train.conllu")
    dev_stats = analyze_corpus(corpus_dir / "es_ancora-ud-dev.conllu")
    test_stats = analyze_corpus(corpus_dir / "es_ancora-ud-test.conllu")
    
    print(f"\n{'='*70}")
    print("RESUMEN TOTAL")
    print(f"{'='*70}")
    total_tokens = train_stats['n_tokens'] + dev_stats['n_tokens'] + test_stats['n_tokens']
    total_sentences = train_stats['n_sentences'] + dev_stats['n_sentences'] + test_stats['n_sentences']
    print(f"Total oraciones: {total_sentences:,}")
    print(f"Total tokens:    {total_tokens:,}")
