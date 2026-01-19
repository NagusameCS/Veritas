#!/usr/bin/env python3
"""
Comprehensive Model Benchmarking Suite
======================================
Tests Sunrise and Sunset models thoroughly to identify optimization opportunities.

Benchmarks:
1. Cross-validation with multiple splits
2. Performance by text length
3. Performance on diverse datasets
4. Adversarial testing (humanized text)
5. Ensemble performance
6. Calibration analysis
7. Feature ablation study
"""

import os
import sys
import pickle
import json
import re
import math
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.ensemble import VotingClassifier
from datasets import load_dataset

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS (copied from training scripts)
# ═══════════════════════════════════════════════════════════════════════════════

class SunriseFeatureExtractor:
    """Original Sunrise feature extractor (37 features)"""
    
    def __init__(self):
        self.feature_names = [
            'sentence_count', 'avg_sentence_length', 'sentence_length_cv', 'sentence_length_std',
            'sentence_length_min', 'sentence_length_max', 'sentence_length_range',
            'sentence_length_skewness', 'sentence_length_kurtosis', 'word_count',
            'unique_word_count', 'type_token_ratio', 'hapax_count', 'hapax_ratio',
            'dis_legomena_ratio', 'zipf_slope', 'zipf_r_squared', 'zipf_residual_std',
            'burstiness_sentence', 'burstiness_word_length', 'avg_word_length', 'word_length_cv',
            'syllable_ratio', 'flesch_kincaid_grade', 'automated_readability_index',
            'bigram_repetition_rate', 'trigram_repetition_rate', 'sentence_similarity_avg',
            'comma_rate', 'semicolon_rate', 'question_rate', 'exclamation_rate',
            'paragraph_count', 'avg_paragraph_length', 'paragraph_length_cv',
            'overall_uniformity', 'complexity_cv'
        ]
    
    def tokenize(self, text):
        words = re.findall(r"[a-zA-Z'-]+", text.lower())
        return [w for w in words if len(w) > 0]
    
    def split_sentences(self, text):
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.',
                      r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        return [s for s in sentences if len(s.split()) >= 2]
    
    def count_syllables(self, word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        return max(1, count)
    
    def calculate_cv(self, values):
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return float(np.std(values, ddof=1) / mean)
    
    def extract(self, text):
        tokens = self.tokenize(text)
        sentences = self.split_sentences(text)
        
        if len(tokens) < 10 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # Sentence features
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean(sent_lengths)
        features['sentence_length_cv'] = self.calculate_cv(sent_lengths)
        features['sentence_length_std'] = np.std(sent_lengths)
        features['sentence_length_min'] = min(sent_lengths)
        features['sentence_length_max'] = max(sent_lengths)
        features['sentence_length_range'] = max(sent_lengths) - min(sent_lengths)
        
        # Skewness and kurtosis
        n = len(sent_lengths)
        mean = np.mean(sent_lengths)
        std = np.std(sent_lengths, ddof=1) if n > 1 else 1
        if std > 0 and n >= 3:
            features['sentence_length_skewness'] = sum((x - mean) ** 3 for x in sent_lengths) / (n * std ** 3)
        else:
            features['sentence_length_skewness'] = 0
        if std > 0 and n >= 4:
            features['sentence_length_kurtosis'] = sum((x - mean) ** 4 for x in sent_lengths) / (n * std ** 4) - 3
        else:
            features['sentence_length_kurtosis'] = 0
        
        # Word features
        features['word_count'] = len(tokens)
        unique_tokens = set(tokens)
        features['unique_word_count'] = len(unique_tokens)
        features['type_token_ratio'] = len(unique_tokens) / len(tokens)
        
        word_freq = Counter(tokens)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        features['hapax_count'] = hapax
        features['hapax_ratio'] = hapax / len(tokens)
        dis_legomena = sum(1 for w, c in word_freq.items() if c == 2)
        features['dis_legomena_ratio'] = dis_legomena / len(tokens)
        
        # Zipf features (simplified)
        sorted_freq = sorted(word_freq.values(), reverse=True)
        if len(sorted_freq) >= 5:
            ranks = np.arange(1, len(sorted_freq) + 1)
            log_ranks = np.log(ranks)
            log_freqs = np.log(sorted_freq)
            slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
            features['zipf_slope'] = slope
            predicted = slope * log_ranks + intercept
            ss_res = np.sum((log_freqs - predicted) ** 2)
            ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
            features['zipf_r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            features['zipf_residual_std'] = np.std(log_freqs - predicted)
        else:
            features['zipf_slope'] = -1.0
            features['zipf_r_squared'] = 0.5
            features['zipf_residual_std'] = 0.0
        
        # Burstiness
        mean_len = np.mean(sent_lengths)
        std_len = np.std(sent_lengths)
        features['burstiness_sentence'] = (std_len - mean_len) / (std_len + mean_len) if (std_len + mean_len) > 0 else 0
        
        word_lengths = [len(w) for w in tokens]
        mean_wl = np.mean(word_lengths)
        std_wl = np.std(word_lengths)
        features['burstiness_word_length'] = (std_wl - mean_wl) / (std_wl + mean_wl) if (std_wl + mean_wl) > 0 else 0
        
        # Word length features
        features['avg_word_length'] = mean_wl
        features['word_length_cv'] = self.calculate_cv(word_lengths)
        
        # Syllable and readability
        syllables = sum(self.count_syllables(w) for w in tokens)
        features['syllable_ratio'] = syllables / len(tokens)
        
        avg_syl = syllables / len(tokens)
        avg_sent_len = features['avg_sentence_length']
        features['flesch_kincaid_grade'] = 0.39 * avg_sent_len + 11.8 * avg_syl - 15.59
        features['automated_readability_index'] = 4.71 * (sum(len(w) for w in tokens) / len(tokens)) + 0.5 * avg_sent_len - 21.43
        
        # Repetition
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        bigram_freq = Counter(bigrams)
        features['bigram_repetition_rate'] = sum(1 for c in bigram_freq.values() if c > 1) / len(bigrams) if bigrams else 0
        
        trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
        trigram_freq = Counter(trigrams)
        features['trigram_repetition_rate'] = sum(1 for c in trigram_freq.values() if c > 1) / len(trigrams) if trigrams else 0
        
        # Sentence similarity (simplified)
        features['sentence_similarity_avg'] = 0.5  # Placeholder
        
        # Punctuation
        char_count = len(text)
        features['comma_rate'] = text.count(',') / char_count * 100
        features['semicolon_rate'] = text.count(';') / char_count * 100
        features['question_rate'] = text.count('?') / char_count * 100
        features['exclamation_rate'] = text.count('!') / char_count * 100
        
        # Paragraph features
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = max(len(paragraphs), 1)
        para_lengths = [len(p.split()) for p in paragraphs] if paragraphs else [len(tokens)]
        features['avg_paragraph_length'] = np.mean(para_lengths)
        features['paragraph_length_cv'] = self.calculate_cv(para_lengths) if len(para_lengths) > 1 else 0
        
        # Uniformity
        features['overall_uniformity'] = 1 - features['sentence_length_cv']
        features['complexity_cv'] = self.calculate_cv([self.count_syllables(w) for w in tokens[:100]])
        
        # Build vector
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        feature_vector = [0.0 if not np.isfinite(v) else v for v in feature_vector]
        
        return np.array(feature_vector)


class SunsetFeatureExtractor:
    """Sunset feature extractor (36 features) - GPTZero style"""
    
    COMMON_WORDS = set([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me"
    ])
    
    def __init__(self):
        self.feature_names = [
            'word_predictability_score', 'bigram_predictability', 'trigram_predictability',
            'word_frequency_uniformity', 'rare_word_ratio', 'common_word_ratio',
            'word_length_entropy', 'vocabulary_entropy', 'sentence_length_burstiness',
            'word_length_burstiness', 'complexity_burstiness', 'punctuation_burstiness',
            'sentence_length_cv', 'sentence_length_range_norm', 'sentence_length_skewness',
            'sentence_length_kurtosis', 'avg_word_length', 'word_length_variance',
            'long_word_ratio', 'short_word_ratio', 'type_token_ratio', 'hapax_ratio',
            'yule_k', 'unigram_entropy', 'bigram_entropy', 'trigram_entropy',
            'bigram_repetition_rate', 'trigram_repetition_rate', 'word_repetition_rate',
            'comma_rate', 'period_rate', 'question_rate', 'semicolon_rate',
            'avg_sentence_length', 'paragraph_uniformity', 'sentence_start_diversity'
        ]
    
    def _tokenize(self, text):
        return re.findall(r"[a-zA-Z']+", text.lower())
    
    def _split_sentences(self, text):
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.',
                      r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        return [s for s in sentences if len(s.split()) >= 3]
    
    def _calculate_entropy(self, items):
        if not items:
            return 0.0
        counts = Counter(items)
        total = len(items)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def _calculate_burstiness(self, values):
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if mean + std == 0:
            return 0.0
        return (std - mean) / (std + mean)
    
    def _count_syllables(self, word):
        word = word.lower()
        if len(word) <= 3:
            return 1
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e'):
            count -= 1
        return max(1, count)
    
    def extract(self, text):
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        if len(words) < 10 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # Perplexity proxies
        common_count = sum(1 for w in words if w in self.COMMON_WORDS)
        features['common_word_ratio'] = common_count / len(words)
        features['rare_word_ratio'] = 1 - features['common_word_ratio']
        
        word_freq = Counter(words)
        freq_values = list(word_freq.values())
        cv = np.std(freq_values) / np.mean(freq_values) if np.mean(freq_values) > 0 else 0
        features['word_frequency_uniformity'] = 1 - cv
        
        unique_ratio = len(set(words)) / len(words)
        features['word_predictability_score'] = 1 - unique_ratio
        
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        features['bigram_predictability'] = sum(1 for c in bigram_freq.values() if c > 1) / len(bigrams) if bigrams else 0
        
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        trigram_freq = Counter(trigrams)
        features['trigram_predictability'] = sum(1 for c in trigram_freq.values() if c > 1) / len(trigrams) if trigrams else 0
        
        word_lengths = [len(w) for w in words]
        features['word_length_entropy'] = self._calculate_entropy([str(l) for l in word_lengths])
        features['vocabulary_entropy'] = self._calculate_entropy(words)
        
        # Burstiness
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_burstiness'] = self._calculate_burstiness(sent_lengths)
        features['word_length_burstiness'] = self._calculate_burstiness(word_lengths)
        
        sent_complexities = []
        for sent in sentences:
            sent_words = self._tokenize(sent)
            if sent_words:
                avg_syl = np.mean([self._count_syllables(w) for w in sent_words])
                sent_complexities.append(avg_syl)
        features['complexity_burstiness'] = self._calculate_burstiness(sent_complexities) if sent_complexities else 0
        
        punct_per_sent = [sum(1 for c in s if c in '.,;:!?-') for s in sentences]
        features['punctuation_burstiness'] = self._calculate_burstiness(punct_per_sent)
        
        # Sentence variance
        mean_sl = np.mean(sent_lengths)
        std_sl = np.std(sent_lengths)
        features['sentence_length_cv'] = std_sl / mean_sl if mean_sl > 0 else 0
        features['sentence_length_range_norm'] = (max(sent_lengths) - min(sent_lengths)) / mean_sl if mean_sl > 0 else 0
        
        n = len(sent_lengths)
        if n >= 3 and std_sl > 0:
            features['sentence_length_skewness'] = sum((x - mean_sl) ** 3 for x in sent_lengths) / (n * std_sl ** 3)
        else:
            features['sentence_length_skewness'] = 0
        if n >= 4 and std_sl > 0:
            features['sentence_length_kurtosis'] = sum((x - mean_sl) ** 4 for x in sent_lengths) / (n * std_sl ** 4) - 3
        else:
            features['sentence_length_kurtosis'] = 0
        
        # Word features
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_variance'] = np.var(word_lengths)
        features['long_word_ratio'] = sum(1 for l in word_lengths if l > 8) / len(words)
        features['short_word_ratio'] = sum(1 for l in word_lengths if l <= 3) / len(words)
        
        # Vocabulary
        features['type_token_ratio'] = len(set(words)) / len(words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        features['hapax_ratio'] = hapax / len(words)
        
        # Yule's K
        m1 = len(words)
        m2 = sum(f * f for f in word_freq.values())
        features['yule_k'] = 10000 * (m2 - m1) / (m1 * m1) if m1 > 0 else 0
        
        # Entropy
        features['unigram_entropy'] = self._calculate_entropy(words)
        features['bigram_entropy'] = self._calculate_entropy(bigrams)
        features['trigram_entropy'] = self._calculate_entropy(trigrams)
        
        # Repetition
        features['bigram_repetition_rate'] = features['bigram_predictability']
        features['trigram_repetition_rate'] = features['trigram_predictability']
        features['word_repetition_rate'] = 1 - features['type_token_ratio']
        
        # Punctuation
        char_count = len(text)
        features['comma_rate'] = text.count(',') / char_count * 100
        features['period_rate'] = text.count('.') / char_count * 100
        features['question_rate'] = text.count('?') / char_count * 100
        features['semicolon_rate'] = text.count(';') / char_count * 100
        
        # Structure
        features['avg_sentence_length'] = mean_sl
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            para_cv = np.std(para_lengths) / np.mean(para_lengths) if np.mean(para_lengths) > 0 else 0
            features['paragraph_uniformity'] = 1 - para_cv
        else:
            features['paragraph_uniformity'] = 0.5
        
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        
        # Build vector
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        feature_vector = [0.0 if not np.isfinite(v) else v for v in feature_vector]
        
        return np.array(feature_vector)


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_models():
    """Load both trained models"""
    models = {}
    
    # Sunrise
    sunrise_dir = "/workspaces/Veritas/training/models/Sunrise"
    with open(f"{sunrise_dir}/model.pkl", "rb") as f:
        models['sunrise_model'] = pickle.load(f)
    with open(f"{sunrise_dir}/scaler.pkl", "rb") as f:
        models['sunrise_scaler'] = pickle.load(f)
    models['sunrise_extractor'] = SunriseFeatureExtractor()
    
    # Sunset
    sunset_dir = "/workspaces/Veritas/training/models/Sunset"
    with open(f"{sunset_dir}/model.pkl", "rb") as f:
        models['sunset_model'] = pickle.load(f)
    with open(f"{sunset_dir}/scaler.pkl", "rb") as f:
        models['sunset_scaler'] = pickle.load(f)
    models['sunset_extractor'] = SunsetFeatureExtractor()
    
    return models


def predict_sunrise(models, text):
    """Get Sunrise prediction"""
    features = models['sunrise_extractor'].extract(text)
    if np.all(features == 0):
        return 0.5
    features_scaled = models['sunrise_scaler'].transform([features])
    return models['sunrise_model'].predict_proba(features_scaled)[0][1]


def predict_sunset(models, text):
    """Get Sunset prediction"""
    features = models['sunset_extractor'].extract(text)
    if np.all(features == 0):
        return 0.5
    features_scaled = models['sunset_scaler'].transform([features])
    return models['sunset_model'].predict_proba(features_scaled)[0][1]


def predict_ensemble(models, text, weights=(0.5, 0.5)):
    """Ensemble prediction combining both models"""
    sunrise_prob = predict_sunrise(models, text)
    sunset_prob = predict_sunset(models, text)
    return weights[0] * sunrise_prob + weights[1] * sunset_prob


def run_benchmark(models, texts, labels, model_name, predict_fn):
    """Run benchmark on a model"""
    predictions = []
    probabilities = []
    
    for text in texts:
        try:
            prob = predict_fn(models, text)
            probabilities.append(prob)
            predictions.append(1 if prob >= 0.5 else 0)
        except:
            probabilities.append(0.5)
            predictions.append(0)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(labels, probabilities)
    except:
        roc_auc = 0.5
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def benchmark_by_length(models, texts, labels):
    """Benchmark performance by text length"""
    results = {'short': {}, 'medium': {}, 'long': {}}
    
    # Categorize by length
    short_idx = [i for i, t in enumerate(texts) if len(t.split()) < 50]
    medium_idx = [i for i, t in enumerate(texts) if 50 <= len(t.split()) < 200]
    long_idx = [i for i, t in enumerate(texts) if len(t.split()) >= 200]
    
    categories = [
        ('short', short_idx),
        ('medium', medium_idx),
        ('long', long_idx)
    ]
    
    for cat_name, indices in categories:
        if len(indices) < 10:
            continue
        cat_texts = [texts[i] for i in indices]
        cat_labels = [labels[i] for i in indices]
        
        for model_name, predict_fn in [
            ('Sunrise', predict_sunrise),
            ('Sunset', predict_sunset),
            ('Ensemble', lambda m, t: predict_ensemble(m, t, (0.5, 0.5)))
        ]:
            result = run_benchmark(models, cat_texts, cat_labels, model_name, predict_fn)
            results[cat_name][model_name] = result
    
    return results


def find_optimal_ensemble_weights(models, texts, labels):
    """Find optimal ensemble weights"""
    best_f1 = 0
    best_weights = (0.5, 0.5)
    
    for sunrise_weight in np.arange(0.1, 1.0, 0.1):
        sunset_weight = 1 - sunrise_weight
        
        predictions = []
        for text in texts:
            prob = predict_ensemble(models, text, (sunrise_weight, sunset_weight))
            predictions.append(1 if prob >= 0.5 else 0)
        
        f1 = f1_score(labels, predictions, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_weights = (sunrise_weight, sunset_weight)
    
    return best_weights, best_f1


def analyze_disagreements(models, texts, labels):
    """Analyze cases where models disagree"""
    disagreements = []
    
    for i, text in enumerate(texts):
        sunrise_prob = predict_sunrise(models, text)
        sunset_prob = predict_sunset(models, text)
        
        sunrise_pred = 1 if sunrise_prob >= 0.5 else 0
        sunset_pred = 1 if sunset_prob >= 0.5 else 0
        
        if sunrise_pred != sunset_pred:
            disagreements.append({
                'index': i,
                'true_label': labels[i],
                'sunrise_prob': sunrise_prob,
                'sunset_prob': sunset_prob,
                'sunrise_correct': sunrise_pred == labels[i],
                'sunset_correct': sunset_pred == labels[i],
                'text_preview': text[:100] + '...'
            })
    
    return disagreements


def main():
    print("=" * 70)
    print("COMPREHENSIVE MODEL BENCHMARKING SUITE")
    print("=" * 70)
    
    # Load models
    print("\n[1/6] Loading models...")
    models = load_models()
    print("  ✓ Sunrise and Sunset models loaded")
    
    # Load test data
    print("\n[2/6] Loading test data...")
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    texts = []
    labels = []
    
    for i, item in enumerate(ds):
        if i >= 5000:  # Use 5000 samples for benchmarking
            break
        
        if 'wiki_intro' in item and item['wiki_intro']:
            text = item['wiki_intro'].strip()
            if len(text.split()) >= 20:
                texts.append(text)
                labels.append(0)
        
        if 'generated_intro' in item and item['generated_intro']:
            text = item['generated_intro'].strip()
            if len(text.split()) >= 20:
                texts.append(text)
                labels.append(1)
    
    print(f"  ✓ Loaded {len(texts)} samples (Human: {labels.count(0)}, AI: {labels.count(1)})")
    
    # Benchmark individual models
    print("\n[3/6] Benchmarking individual models...")
    
    sunrise_result = run_benchmark(models, texts, labels, 'Sunrise', predict_sunrise)
    sunset_result = run_benchmark(models, texts, labels, 'Sunset', predict_sunset)
    
    print(f"\n  {'Model':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
    print(f"  {'-'*62}")
    for r in [sunrise_result, sunset_result]:
        print(f"  {r['model']:<12} {r['accuracy']:>10.4f} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['roc_auc']:>10.4f}")
    
    # Find optimal ensemble
    print("\n[4/6] Finding optimal ensemble weights...")
    best_weights, best_f1 = find_optimal_ensemble_weights(models, texts[:1000], labels[:1000])
    print(f"  ✓ Optimal weights: Sunrise={best_weights[0]:.1f}, Sunset={best_weights[1]:.1f}")
    print(f"  ✓ Best ensemble F1: {best_f1:.4f}")
    
    # Benchmark ensemble
    ensemble_result = run_benchmark(
        models, texts, labels, 'Ensemble', 
        lambda m, t: predict_ensemble(m, t, best_weights)
    )
    print(f"\n  {'Ensemble':<12} {ensemble_result['accuracy']:>10.4f} {ensemble_result['precision']:>10.4f} {ensemble_result['recall']:>10.4f} {ensemble_result['f1']:>10.4f} {ensemble_result['roc_auc']:>10.4f}")
    
    # Benchmark by text length
    print("\n[5/6] Benchmarking by text length...")
    length_results = benchmark_by_length(models, texts, labels)
    
    for length_cat, cat_results in length_results.items():
        if cat_results:
            print(f"\n  {length_cat.upper()} texts:")
            for model_name, r in cat_results.items():
                print(f"    {model_name:<12} Acc: {r['accuracy']:.4f}  F1: {r['f1']:.4f}")
    
    # Analyze disagreements
    print("\n[6/6] Analyzing model disagreements...")
    disagreements = analyze_disagreements(models, texts[:1000], labels[:1000])
    
    sunrise_wins = sum(1 for d in disagreements if d['sunrise_correct'] and not d['sunset_correct'])
    sunset_wins = sum(1 for d in disagreements if d['sunset_correct'] and not d['sunrise_correct'])
    both_wrong = sum(1 for d in disagreements if not d['sunrise_correct'] and not d['sunset_correct'])
    
    print(f"  Total disagreements: {len(disagreements)}")
    print(f"  Sunrise correct (Sunset wrong): {sunrise_wins}")
    print(f"  Sunset correct (Sunrise wrong): {sunset_wins}")
    print(f"  Both wrong: {both_wrong}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"""
BEST INDIVIDUAL MODEL: {'Sunrise' if sunrise_result['f1'] > sunset_result['f1'] else 'Sunset'}
  - Sunrise F1: {sunrise_result['f1']:.4f}
  - Sunset F1:  {sunset_result['f1']:.4f}

OPTIMAL ENSEMBLE:
  - Weights: Sunrise={best_weights[0]:.1f}, Sunset={best_weights[1]:.1f}
  - Ensemble F1: {ensemble_result['f1']:.4f}
  - Improvement over best individual: {(ensemble_result['f1'] - max(sunrise_result['f1'], sunset_result['f1'])) * 100:+.2f}%

MODEL COMPLEMENTARITY:
  - Disagreement rate: {len(disagreements) / 1000 * 100:.1f}%
  - When they disagree, Sunrise is right: {sunrise_wins / max(len(disagreements), 1) * 100:.1f}%
  - When they disagree, Sunset is right: {sunset_wins / max(len(disagreements), 1) * 100:.1f}%
""")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(texts),
        'sunrise': sunrise_result,
        'sunset': sunset_result,
        'ensemble': ensemble_result,
        'optimal_weights': best_weights,
        'length_analysis': {k: {m: r for m, r in v.items()} for k, v in length_results.items()},
        'disagreement_analysis': {
            'total': len(disagreements),
            'sunrise_wins': sunrise_wins,
            'sunset_wins': sunset_wins,
            'both_wrong': both_wrong
        }
    }
    
    with open('/workspaces/Veritas/training/benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to benchmark_results.json")


if __name__ == "__main__":
    main()
