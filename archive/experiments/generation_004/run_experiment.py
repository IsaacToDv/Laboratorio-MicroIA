"""
GENERATION-004: Evaluación de Calidad de Texto Generado

Este experimento evalúa la calidad del texto generado usando métricas automáticas:
1. Perplejidad (usando modelo de n-gramas)
2. Coherencia gramatical (proporción de secuencias válidas)
3. Diversidad léxica (TTR, entropy)
4. Repetición (fracción de n-gramas repetidos)

Métricas:
- Perplejidad promedio
- Coherencia gramatical
- Diversidad léxica
- Tasa de repetición
"""

import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from collections import Counter, defaultdict
import math

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from generation.text_generator import TextGenerator
from conllu import parse_incr


def load_corpus_sentences(corpus_path: str, max_sentences: int = 2000):
    """Carga oraciones del corpus."""
    sentences = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for i, sent in enumerate(parse_incr(f)):
            if i >= max_sentences:
                break
            words = [token['form'].lower() for token in sent 
                    if not isinstance(token['id'], tuple)]
            if len(words) >= 3:
                sentences.append(words)
    return sentences


class NgramLanguageModel:
    """Modelo de lenguaje simple basado en n-gramas para calcular perplejidad."""
    
    def __init__(self, n: int = 2):
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
    
    def train(self, sentences):
        """Entrena el modelo con oraciones."""
        for sent in sentences:
            words = ['<s>'] * (self.n - 1) + sent + ['</s>']
            self.vocab.update(words)
            
            for i in range(len(words) - self.n + 1):
                context = tuple(words[i:i+self.n-1])
                word = words[i+self.n-1]
                self.ngram_counts[context][word] += 1
                self.context_counts[context] += 1
    
    def perplexity(self, sentence):
        """Calcula perplejidad de una oración."""
        words = ['<s>'] * (self.n - 1) + sentence + ['</s>']
        
        log_prob = 0.0
        count = 0
        
        for i in range(len(words) - self.n + 1):
            context = tuple(words[i:i+self.n-1])
            word = words[i+self.n-1]
            
            # Suavizado de Laplace
            numerator = self.ngram_counts[context].get(word, 0) + 1
            denominator = self.context_counts.get(context, 0) + len(self.vocab)
            
            prob = numerator / denominator
            log_prob += math.log(prob)
            count += 1
        
        # Perplejidad = exp(-1/N * sum(log P))
        if count == 0:
            return float('inf')
        
        avg_log_prob = log_prob / count
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity


def calculate_diversity(texts):
    """Calcula métricas de diversidad léxica."""
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())
    
    if not all_words:
        return {}
    
    # Type-Token Ratio
    ttr = len(set(all_words)) / len(all_words)
    
    # Entropía léxica
    word_counts = Counter(all_words)
    total = len(all_words)
    probs = [count / total for count in word_counts.values()]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    
    return {
        'type_token_ratio': ttr,
        'entropy': entropy,
        'unique_words': len(set(all_words)),
        'total_words': len(all_words)
    }


def calculate_repetition(texts, n: int = 2):
    """Calcula tasa de repetición de n-gramas."""
    all_ngrams = []
    
    for text in texts:
        words = text.lower().split()
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            all_ngrams.append(ngram)
    
    if not all_ngrams:
        return 0.0
    
    ngram_counts = Counter(all_ngrams)
    repeated = sum(1 for count in ngram_counts.values() if count > 1)
    
    return repeated / len(ngram_counts) if ngram_counts else 0.0


def main():
    print("=" * 70)
    print("GENERATION-004: Evaluación de Calidad de Texto Generado")
    print("=" * 70)
    print()
    
    # Configuración
    corpus_path = "data/corpus/UD_Spanish-AnCora/es_ancora-ud-train.conllu"
    dim = 10000
    max_sentences_train = 2000
    num_eval_sentences = 100
    
    print(f"Configuración:")
    print(f"  - Dimensiones HDC: {dim}")
    print(f"  - Oraciones de entrenamiento: {max_sentences_train}")
    print(f"  - Oraciones a generar para evaluación: {num_eval_sentences}")
    print()
    
    # Cargar corpus para entrenar modelo de evaluación
    print("Cargando corpus para modelo de evaluación...")
    corpus_sentences = load_corpus_sentences(corpus_path, max_sentences_train)
    print(f"✓ Cargadas {len(corpus_sentences)} oraciones")
    print()
    
    # Entrenar modelo de lenguaje para perplejidad
    print("Entrenando modelo de lenguaje (bigramas) para evaluación...")
    lm = NgramLanguageModel(n=2)
    lm.train(corpus_sentences)
    print(f"✓ Modelo entrenado con vocabulario de {len(lm.vocab)} palabras")
    print()
    
    # Crear y entrenar generador
    print("Entrenando generador de texto...")
    generator = TextGenerator(dim=dim)
    generator.train_from_corpus(corpus_path, max_sentences=max_sentences_train)
    print()
    
    # Generar textos para evaluación
    print("=" * 70)
    print(f"Generando {num_eval_sentences} textos para evaluación")
    print("=" * 70)
    print()
    
    # Prompts variados
    base_prompts = [
        "El", "La", "Los", "Las", "En", "Durante", "Según", "Para",
        "El presidente", "La economía", "El gobierno", "La empresa"
    ]
    
    generated_texts = []
    for i in range(num_eval_sentences):
        prompt = base_prompts[i % len(base_prompts)]
        text = generator.generate(prompt, max_words=15, temperature=0.6)
        generated_texts.append(text)
        
        if (i + 1) % 20 == 0:
            print(f"  Generados {i+1}/{num_eval_sentences} textos")
    
    print(f"✓ {len(generated_texts)} textos generados")
    print()
    
    # Evaluar perplejidad
    print("=" * 70)
    print("Evaluación de Perplejidad")
    print("=" * 70)
    print()
    
    perplexities = []
    for text in generated_texts:
        words = text.lower().split()
        ppl = lm.perplexity(words)
        perplexities.append(ppl)
    
    avg_perplexity = np.mean(perplexities)
    std_perplexity = np.std(perplexities)
    median_perplexity = np.median(perplexities)
    
    print(f"Perplejidad promedio: {avg_perplexity:.2f} ± {std_perplexity:.2f}")
    print(f"Perplejidad mediana: {median_perplexity:.2f}")
    print(f"Perplejidad mínima: {min(perplexities):.2f}")
    print(f"Perplejidad máxima: {max(perplexities):.2f}")
    print()
    print("Interpretación:")
    print("  - Perplejidad más baja = texto más predecible/coherente")
    print("  - Perplejidad muy alta = texto incoherente o muy creativo")
    print()
    
    # Evaluar perplejidad del corpus original (baseline)
    print("Calculando perplejidad del corpus original (baseline)...")
    corpus_perplexities = []
    for sent in corpus_sentences[:100]:
        ppl = lm.perplexity(sent)
        corpus_perplexities.append(ppl)
    
    avg_corpus_ppl = np.mean(corpus_perplexities)
    print(f"Perplejidad promedio del corpus: {avg_corpus_ppl:.2f}")
    print()
    
    # Evaluar diversidad
    print("=" * 70)
    print("Evaluación de Diversidad Léxica")
    print("=" * 70)
    print()
    
    diversity = calculate_diversity(generated_texts)
    
    print(f"Type-Token Ratio (TTR): {diversity['type_token_ratio']:.3f}")
    print(f"Entropía léxica: {diversity['entropy']:.3f}")
    print(f"Palabras únicas: {diversity['unique_words']}")
    print(f"Total de palabras: {diversity['total_words']}")
    print()
    
    # Evaluar repetición
    print("=" * 70)
    print("Evaluación de Repetición")
    print("=" * 70)
    print()
    
    bigram_repetition = calculate_repetition(generated_texts, n=2)
    trigram_repetition = calculate_repetition(generated_texts, n=3)
    
    print(f"Repetición de bigramas: {bigram_repetition:.3f}")
    print(f"Repetición de trigramas: {trigram_repetition:.3f}")
    print()
    print("Interpretación:")
    print("  - 0.0 = sin repeticiones (muy diverso)")
    print("  - 1.0 = todas las secuencias se repiten (poco diverso)")
    print()
    
    # Ejemplos de mejor y peor texto
    print("=" * 70)
    print("Ejemplos de Textos Generados")
    print("=" * 70)
    print()
    
    # Ordenar por perplejidad
    texts_with_ppl = list(zip(generated_texts, perplexities))
    texts_with_ppl.sort(key=lambda x: x[1])
    
    print("Mejores textos (menor perplejidad):")
    for i in range(3):
        text, ppl = texts_with_ppl[i]
        print(f"{i+1}. [PPL={ppl:.1f}] {text}")
    print()
    
    print("Textos más creativos (mayor perplejidad):")
    for i in range(3):
        text, ppl = texts_with_ppl[-(i+1)]
        print(f"{i+1}. [PPL={ppl:.1f}] {text}")
    print()
    
    # Guardar resultados
    results = {
        'experiment_id': 'GENERATION-004',
        'timestamp': datetime.now().isoformat(),
        'config': {
            'dimensions': dim,
            'max_sentences_train': max_sentences_train,
            'num_eval_sentences': num_eval_sentences,
            'corpus': corpus_path
        },
        'metrics': {
            'perplexity': {
                'mean': float(avg_perplexity),
                'std': float(std_perplexity),
                'median': float(median_perplexity),
                'min': float(min(perplexities)),
                'max': float(max(perplexities)),
                'corpus_baseline': float(avg_corpus_ppl)
            },
            'diversity': {
                'type_token_ratio': float(diversity['type_token_ratio']),
                'entropy': float(diversity['entropy']),
                'unique_words': int(diversity['unique_words']),
                'total_words': int(diversity['total_words'])
            },
            'repetition': {
                'bigram_repetition': float(bigram_repetition),
                'trigram_repetition': float(trigram_repetition)
            }
        },
        'examples': {
            'best_texts': [{'text': t, 'perplexity': float(p)} for t, p in texts_with_ppl[:5]],
            'worst_texts': [{'text': t, 'perplexity': float(p)} for t, p in texts_with_ppl[-5:]]
        }
    }
    
    output_path = Path(__file__).parent / 'results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("=" * 70)
    print("Resultados guardados en:", output_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
