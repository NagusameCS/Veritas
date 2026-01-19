#!/usr/bin/env python3
"""
Humanized Text Detection Benchmark
===================================
Tests Sunrise and Sunset models against:
1. Pure human text
2. Pure AI text
3. Humanized AI text (simulated)

Also evaluates a dedicated humanization detector.
"""

import os
import pickle
import json
import re
import random
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Humanization simulation patterns
HUMANIZATION_TRANSFORMS = {
    'add_disfluencies': [
        ('Furthermore,', 'So, furthermore,'),
        ('Additionally,', 'Well, additionally,'),
        ('However,', 'I mean, however,'),
        ('Therefore,', 'Basically, therefore,'),
        ('In conclusion,', 'Anyway, in conclusion,'),
        ('It is important', 'Honestly, it is important'),
        ('Moreover,', 'Like, moreover,'),
    ],
    'add_contractions': [
        ('it is', "it's"),
        ('do not', "don't"),
        ('cannot', "can't"),
        ('will not', "won't"),
        ('would not', "wouldn't"),
        ('should not', "shouldn't"),
        ('could not', "couldn't"),
        ('I am', "I'm"),
        ('they are', "they're"),
        ('we are', "we're"),
        ('that is', "that's"),
        ('there is', "there's"),
    ],
    'add_typos': [
        ('the', 'teh'),
        ('and', 'adn'),
        ('that', 'taht'),
        ('with', 'wiht'),
        ('have', 'ahve'),
        ('this', 'tihs'),
        ('because', 'becuase'),
        ('different', 'diffrent'),
        ('probably', 'probaly'),
        ('definitely', 'definately'),
    ],
    'add_slang': [
        ('very', 'super'),
        ('extremely', 'totally'),
        ('certainly', 'for sure'),
        ('understand', 'get'),
        ('excellent', 'awesome'),
        ('remarkable', 'cool'),
    ],
    'add_rhetorical': [
        ('. ', ', right? '),
        ('? ', ', you know? '),
    ]
}


def humanize_text(text: str, intensity: float = 0.5) -> str:
    """
    Simulate humanization of AI text.
    intensity: 0.0 (minimal) to 1.0 (heavy)
    """
    result = text
    
    # Add sentence-start disfluencies
    sentences = re.split(r'(?<=[.!?])\s+', result)
    disfluencies = ['Well, ', 'So, ', 'I mean, ', 'Basically, ', 'Honestly, ', 'Like, ', 'Actually, ']
    
    modified_sentences = []
    for i, sent in enumerate(sentences):
        if random.random() < intensity * 0.3 and len(sent) > 20:
            sent = random.choice(disfluencies) + sent[0].lower() + sent[1:]
        modified_sentences.append(sent)
    result = ' '.join(modified_sentences)
    
    # Apply contractions
    for old, new in HUMANIZATION_TRANSFORMS['add_contractions']:
        if random.random() < intensity:
            result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    # Add occasional typos (low rate)
    if intensity > 0.3:
        words = result.split()
        for i, word in enumerate(words):
            if random.random() < intensity * 0.02:
                for old, new in HUMANIZATION_TRANSFORMS['add_typos']:
                    if word.lower() == old:
                        words[i] = new
                        break
        result = ' '.join(words)
    
    # Add slang substitutions
    if intensity > 0.4:
        for old, new in HUMANIZATION_TRANSFORMS['add_slang']:
            if random.random() < intensity * 0.3:
                result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    # Add rhetorical questions (very sparingly)
    if intensity > 0.6 and random.random() < 0.2:
        result = result.replace('. ', ', right? ', 1)
    
    return result


class SunriseExtractor:
    """Feature extractor matching Sunrise model training"""
    
    def __init__(self):
        self.feature_names = [
            'sentence_count', 'avg_sentence_length', 'sentence_length_cv',
            'sentence_length_std', 'sentence_length_min', 'sentence_length_max',
            'sentence_length_range', 'sentence_length_skewness', 'sentence_length_kurtosis',
            'word_count', 'unique_word_count', 'type_token_ratio',
            'hapax_count', 'hapax_ratio', 'dis_legomena_ratio',
            'zipf_slope', 'zipf_r_squared', 'zipf_residual_std',
            'burstiness_sentence', 'burstiness_word_length',
            'avg_word_length', 'word_length_cv', 'syllable_ratio',
            'flesch_kincaid_grade', 'automated_readability_index',
            'bigram_repetition_rate', 'trigram_repetition_rate', 'sentence_similarity_avg',
            'comma_rate', 'semicolon_rate', 'question_rate', 'exclamation_rate',
            'paragraph_count', 'avg_paragraph_length', 'paragraph_length_cv',
            'overall_uniformity', 'complexity_cv'
        ]
    
    def extract(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(words) < 10 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # Sentence features
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_count'] = len(sentences)
        features['avg_sentence_length'] = np.mean(sent_lengths)
        features['sentence_length_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['sentence_length_cv'] = features['sentence_length_std'] / features['avg_sentence_length'] if features['avg_sentence_length'] > 0 else 0
        features['sentence_length_min'] = min(sent_lengths)
        features['sentence_length_max'] = max(sent_lengths)
        features['sentence_length_range'] = features['sentence_length_max'] - features['sentence_length_min']
        features['sentence_length_skewness'] = self._skewness(sent_lengths)
        features['sentence_length_kurtosis'] = self._kurtosis(sent_lengths)
        
        # Word features
        features['word_count'] = len(words)
        features['unique_word_count'] = len(set(words))
        features['type_token_ratio'] = features['unique_word_count'] / features['word_count']
        
        word_freq = Counter(words)
        features['hapax_count'] = sum(1 for w, c in word_freq.items() if c == 1)
        features['hapax_ratio'] = features['hapax_count'] / len(words)
        features['dis_legomena_ratio'] = sum(1 for w, c in word_freq.items() if c == 2) / len(words)
        
        # Zipf features
        zipf = self._zipf_analysis(word_freq)
        features['zipf_slope'] = zipf['slope']
        features['zipf_r_squared'] = zipf['r_squared']
        features['zipf_residual_std'] = zipf['residual_std']
        
        # Burstiness
        features['burstiness_sentence'] = self._burstiness(sent_lengths)
        word_lengths = [len(w) for w in words]
        features['burstiness_word_length'] = self._burstiness(word_lengths)
        
        # Word length
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_cv'] = np.std(word_lengths) / np.mean(word_lengths) if np.mean(word_lengths) > 0 else 0
        
        # Syllables
        syllables = [self._count_syllables(w) for w in words]
        features['syllable_ratio'] = sum(syllables) / len(words)
        
        # Readability
        features['flesch_kincaid_grade'] = self._flesch_kincaid(len(words), len(sentences), sum(syllables))
        features['automated_readability_index'] = self._ari(len(text), len(words), len(sentences))
        
        # Repetition
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        features['bigram_repetition_rate'] = self._repetition_rate(bigrams)
        features['trigram_repetition_rate'] = self._repetition_rate(trigrams)
        features['sentence_similarity_avg'] = 0.5  # Simplified
        
        # Punctuation
        char_count = max(len(text), 1)
        features['comma_rate'] = text.count(',') / char_count
        features['semicolon_rate'] = text.count(';') / char_count
        features['question_rate'] = text.count('?') / char_count
        features['exclamation_rate'] = text.count('!') / char_count
        
        # Paragraph features
        features['paragraph_count'] = max(len(paragraphs), 1)
        para_lengths = [len(p.split()) for p in paragraphs] if paragraphs else [len(words)]
        features['avg_paragraph_length'] = np.mean(para_lengths)
        features['paragraph_length_cv'] = np.std(para_lengths) / np.mean(para_lengths) if np.mean(para_lengths) > 0 else 0
        
        # Uniformity
        features['overall_uniformity'] = 1 - features['sentence_length_cv']
        features['complexity_cv'] = features['word_length_cv']
        
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def _split_sentences(self, text):
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.replace('<PERIOD>', '.').strip() for s in sentences if len(s.split()) >= 2]
    
    def _skewness(self, values):
        if len(values) < 3: return 0
        mean, std = np.mean(values), np.std(values, ddof=1)
        if std == 0: return 0
        return sum((x - mean) ** 3 for x in values) / (len(values) * std ** 3)
    
    def _kurtosis(self, values):
        if len(values) < 4: return 0
        mean, std = np.mean(values), np.std(values, ddof=1)
        if std == 0: return 0
        return sum((x - mean) ** 4 for x in values) / (len(values) * std ** 4) - 3
    
    def _burstiness(self, values):
        if len(values) < 2: return 0
        mean, std = np.mean(values), np.std(values)
        if mean + std == 0: return 0
        return (std - mean) / (std + mean)
    
    def _zipf_analysis(self, word_freq):
        if len(word_freq) < 5:
            return {'slope': -1, 'r_squared': 0, 'residual_std': 0}
        sorted_freq = sorted(word_freq.values(), reverse=True)[:100]
        ranks = np.arange(1, len(sorted_freq) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(sorted_freq)
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        predicted = slope * log_ranks + intercept
        residuals = log_freqs - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return {'slope': slope, 'r_squared': r_squared, 'residual_std': np.std(residuals)}
    
    def _count_syllables(self, word):
        word = word.lower()
        if len(word) <= 3: return 1
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'): count -= 1
        return max(1, count)
    
    def _flesch_kincaid(self, words, sentences, syllables):
        if sentences == 0 or words == 0: return 0
        return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
    
    def _ari(self, chars, words, sentences):
        if words == 0 or sentences == 0: return 0
        return 4.71 * (chars / words) + 0.5 * (words / sentences) - 21.43
    
    def _repetition_rate(self, ngrams):
        if not ngrams: return 0
        counts = Counter(ngrams)
        return sum(1 for c in counts.values() if c > 1) / len(ngrams)


class SunsetExtractor:
    """Feature extractor matching Sunset model training"""
    
    def __init__(self):
        self.feature_names = [
            'word_predictability_score', 'bigram_predictability', 'trigram_predictability',
            'word_frequency_uniformity', 'rare_word_ratio', 'common_word_ratio',
            'word_length_entropy', 'vocabulary_entropy',
            'sentence_length_burstiness', 'word_length_burstiness',
            'complexity_burstiness', 'punctuation_burstiness',
            'sentence_length_cv', 'sentence_length_range_norm',
            'sentence_length_skewness', 'sentence_length_kurtosis',
            'avg_word_length', 'word_length_variance', 'long_word_ratio', 'short_word_ratio',
            'type_token_ratio', 'hapax_ratio', 'yule_k',
            'unigram_entropy', 'bigram_entropy', 'trigram_entropy',
            'bigram_repetition_rate', 'trigram_repetition_rate', 'word_repetition_rate',
            'comma_rate', 'period_rate', 'question_rate', 'semicolon_rate',
            'avg_sentence_length', 'paragraph_uniformity', 'sentence_start_diversity'
        ]
        self.common_words = set(['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at'])
    
    def extract(self, text: str) -> np.ndarray:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        sentences = self._split_sentences(text)
        
        if len(words) < 10 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        word_freq = Counter(words)
        
        # Predictability
        features['common_word_ratio'] = sum(1 for w in words if w in self.common_words) / len(words)
        features['rare_word_ratio'] = 1 - features['common_word_ratio']
        features['word_predictability_score'] = 1 - len(set(words)) / len(words)
        features['word_frequency_uniformity'] = 1 - (np.std(list(word_freq.values())) / np.mean(list(word_freq.values())) if word_freq else 0)
        
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        features['bigram_predictability'] = sum(1 for c in Counter(bigrams).values() if c > 1) / len(bigrams) if bigrams else 0
        features['trigram_predictability'] = sum(1 for c in Counter(trigrams).values() if c > 1) / len(trigrams) if trigrams else 0
        
        # Entropy
        features['word_length_entropy'] = self._entropy([str(len(w)) for w in words])
        features['vocabulary_entropy'] = self._entropy(words)
        features['unigram_entropy'] = features['vocabulary_entropy']
        features['bigram_entropy'] = self._entropy(bigrams)
        features['trigram_entropy'] = self._entropy(trigrams)
        
        # Burstiness
        sent_lengths = [len(s.split()) for s in sentences]
        word_lengths = [len(w) for w in words]
        features['sentence_length_burstiness'] = self._burstiness(sent_lengths)
        features['word_length_burstiness'] = self._burstiness(word_lengths)
        features['complexity_burstiness'] = features['word_length_burstiness']
        features['punctuation_burstiness'] = self._burstiness([sum(1 for c in s if c in '.,;:!?') for s in sentences])
        
        # Sentence stats
        features['sentence_length_cv'] = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
        features['sentence_length_range_norm'] = (max(sent_lengths) - min(sent_lengths)) / np.mean(sent_lengths) if sent_lengths else 0
        features['sentence_length_skewness'] = self._skewness(sent_lengths)
        features['sentence_length_kurtosis'] = self._kurtosis(sent_lengths)
        features['avg_sentence_length'] = np.mean(sent_lengths)
        
        # Word stats
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_variance'] = np.var(word_lengths)
        features['long_word_ratio'] = sum(1 for l in word_lengths if l > 8) / len(words)
        features['short_word_ratio'] = sum(1 for l in word_lengths if l <= 3) / len(words)
        
        # Vocabulary
        features['type_token_ratio'] = len(set(words)) / len(words)
        features['hapax_ratio'] = sum(1 for c in word_freq.values() if c == 1) / len(words)
        features['yule_k'] = self._yule_k(word_freq, len(words))
        
        # Repetition
        features['bigram_repetition_rate'] = features['bigram_predictability']
        features['trigram_repetition_rate'] = features['trigram_predictability']
        features['word_repetition_rate'] = 1 - features['type_token_ratio']
        
        # Punctuation
        char_count = max(len(text), 1)
        features['comma_rate'] = text.count(',') / char_count * 100
        features['period_rate'] = text.count('.') / char_count * 100
        features['question_rate'] = text.count('?') / char_count * 100
        features['semicolon_rate'] = text.count(';') / char_count * 100
        
        # Structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - (np.std(para_lengths) / np.mean(para_lengths) if np.mean(para_lengths) > 0 else 0)
        else:
            features['paragraph_uniformity'] = 0.5
        
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def _split_sentences(self, text):
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.replace('<PERIOD>', '.').strip() for s in sentences if len(s.split()) >= 3]
    
    def _entropy(self, items):
        if not items: return 0
        counts = Counter(items)
        total = len(items)
        return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)
    
    def _burstiness(self, values):
        if len(values) < 2: return 0
        mean, std = np.mean(values), np.std(values)
        if mean + std == 0: return 0
        return (std - mean) / (std + mean)
    
    def _skewness(self, values):
        if len(values) < 3: return 0
        mean, std = np.mean(values), np.std(values, ddof=1)
        if std == 0: return 0
        return sum((x - mean) ** 3 for x in values) / (len(values) * std ** 3)
    
    def _kurtosis(self, values):
        if len(values) < 4: return 0
        mean, std = np.mean(values), np.std(values, ddof=1)
        if std == 0: return 0
        return sum((x - mean) ** 4 for x in values) / (len(values) * std ** 4) - 3
    
    def _yule_k(self, freq, n):
        if n < 10: return 0
        m2 = sum(f * f for f in freq.values())
        return 10000 * (m2 - n) / (n * n) if n > 0 else 0


class HumanizationDetector:
    """Dedicated detector for humanized AI text"""
    
    def __init__(self):
        self.disfluencies = ['well,', 'so,', 'i mean,', 'basically,', 'honestly,', 'like,', 'actually,', 'anyway,']
        self.informal_contractions = ["gonna", "wanna", "gotta", "kinda", "sorta", "'cause", "dunno", "gimme"]
        self.ai_phrases = ["it is important to note", "furthermore", "moreover", "in conclusion", "demonstrates", "comprehensive"]
        self.common_typos = ["teh", "adn", "taht", "wiht", "ahve", "becuase", "definately", "probaly"]
    
    def detect(self, text: str) -> Dict:
        """Returns humanization score and signals"""
        lower = text.lower()
        sentences = re.split(r'[.!?]+\s+', text)
        words = re.findall(r"[a-zA-Z']+", lower)
        
        signals = {}
        
        # Disfluency at sentence starts
        disfluency_starts = sum(1 for s in sentences if any(s.lower().strip().startswith(d) for d in self.disfluencies))
        signals['disfluency_ratio'] = disfluency_starts / len(sentences) if sentences else 0
        
        # Informal contractions
        informal_count = sum(1 for w in words if w in self.informal_contractions)
        signals['informal_contraction_rate'] = informal_count / len(words) * 100 if words else 0
        
        # AI phrases remaining
        ai_phrase_count = sum(1 for phrase in self.ai_phrases if phrase in lower)
        signals['ai_phrase_count'] = ai_phrase_count
        
        # Typos
        typo_count = sum(1 for w in words if w in self.common_typos)
        signals['typo_count'] = typo_count
        
        # Contraction distribution uniformity
        contractions = re.findall(r"\b\w+'\w+\b", lower)
        if len(sentences) > 3 and len(contractions) > 2:
            contr_per_sent = [len(re.findall(r"\b\w+'\w+\b", s.lower())) for s in sentences]
            mean = np.mean(contr_per_sent)
            cv = np.std(contr_per_sent) / mean if mean > 0 else 1
            signals['contraction_uniformity'] = 1 - min(cv, 1)  # Low CV = more uniform = suspicious
        else:
            signals['contraction_uniformity'] = 0
        
        # Calculate humanization score
        score = 0
        if signals['disfluency_ratio'] > 0.2: score += 0.25
        if signals['informal_contraction_rate'] > 0.5: score += 0.15
        if signals['ai_phrase_count'] > 0 and signals['disfluency_ratio'] > 0.1: score += 0.3  # Co-occurrence
        if signals['typo_count'] > 0: score += 0.2
        if signals['contraction_uniformity'] > 0.6: score += 0.1
        
        return {
            'humanization_score': min(score, 1.0),
            'is_likely_humanized': score > 0.3,
            'signals': signals
        }


def run_benchmark():
    """Run comprehensive benchmark"""
    print("=" * 70)
    print("HUMANIZED TEXT DETECTION BENCHMARK")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    
    sunrise_path = "/workspaces/Veritas/training/models/Sunrise"
    sunset_path = "/workspaces/Veritas/training/models/Sunset"
    
    with open(f"{sunrise_path}/model.pkl", "rb") as f:
        sunrise_model = pickle.load(f)
    with open(f"{sunrise_path}/scaler.pkl", "rb") as f:
        sunrise_scaler = pickle.load(f)
    
    with open(f"{sunset_path}/model.pkl", "rb") as f:
        sunset_model = pickle.load(f)
    with open(f"{sunset_path}/scaler.pkl", "rb") as f:
        sunset_scaler = pickle.load(f)
    
    sunrise_extractor = SunriseExtractor()
    sunset_extractor = SunsetExtractor()
    humanization_detector = HumanizationDetector()
    
    print("Models loaded!")
    
    # Test samples
    human_samples = [
        "So I was thinking about this yesterday, right? My friend Dave told me about this crazy thing that happened at work. Honestly, I couldn't believe it. Like, who does that? Anyway, we ended up laughing about it for hours.",
        "I've been trying to figure out what's wrong with my car for weeks now. Took it to three different mechanics and they all said different things. Super frustrating, you know? My brother thinks it might be the transmission but I'm not so sure.",
        "The sunset last night was absolutely breathtaking. I sat on my porch for about an hour just watching the colors change. Reminded me of when I was a kid and we'd go camping every summer. Those were good times.",
    ]
    
    ai_samples = [
        "It is important to note that artificial intelligence has demonstrated significant capabilities in various domains. The implementation of these systems requires comprehensive understanding of the underlying mechanisms. Furthermore, one must consider the ethical implications that subsequently arise from such technological advancements.",
        "The phenomenon of climate change represents one of the most pressing challenges facing humanity in the contemporary era. Scientific research has conclusively demonstrated that anthropogenic factors contribute significantly to global temperature increases. Consequently, it is imperative that coordinated international efforts be undertaken to mitigate these effects.",
        "Education plays a pivotal role in shaping the future of society. It is essential to recognize that quality education provides individuals with the necessary tools to succeed in an increasingly competitive global economy. Moreover, educational institutions serve as the foundation for cultivating critical thinking skills.",
    ]
    
    # Generate humanized versions
    humanized_light = [humanize_text(text, 0.3) for text in ai_samples]
    humanized_medium = [humanize_text(text, 0.5) for text in ai_samples]
    humanized_heavy = [humanize_text(text, 0.8) for text in ai_samples]
    
    # Build test set
    test_data = []
    test_data.extend([(t, 'human', 'none') for t in human_samples])
    test_data.extend([(t, 'ai', 'none') for t in ai_samples])
    test_data.extend([(t, 'ai', 'light') for t in humanized_light])
    test_data.extend([(t, 'ai', 'medium') for t in humanized_medium])
    test_data.extend([(t, 'ai', 'heavy') for t in humanized_heavy])
    
    # Run predictions
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    
    results = []
    
    for text, true_origin, humanization in test_data:
        # Sunrise prediction
        sunrise_features = sunrise_extractor.extract(text)
        sunrise_scaled = sunrise_scaler.transform([sunrise_features])
        sunrise_prob = sunrise_model.predict_proba(sunrise_scaled)[0][1]
        
        # Sunset prediction
        sunset_features = sunset_extractor.extract(text)
        sunset_scaled = sunset_scaler.transform([sunset_features])
        sunset_prob = sunset_model.predict_proba(sunset_scaled)[0][1]
        
        # Humanization detection
        humanization_result = humanization_detector.detect(text)
        
        # Ensemble
        ensemble_prob = (sunrise_prob + sunset_prob) / 2
        
        results.append({
            'text': text[:80] + '...',
            'true_origin': true_origin,
            'humanization': humanization,
            'sunrise_prob': sunrise_prob,
            'sunset_prob': sunset_prob,
            'ensemble_prob': ensemble_prob,
            'humanization_score': humanization_result['humanization_score'],
            'is_humanized_detected': humanization_result['is_likely_humanized']
        })
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Origin: {true_origin.upper()} | Humanization: {humanization}")
        print(f"Text: {text[:100]}...")
        print(f"  Sunrise AI prob:     {sunrise_prob:.3f} {'✓' if (sunrise_prob > 0.5) == (true_origin == 'ai') else '✗'}")
        print(f"  Sunset AI prob:      {sunset_prob:.3f} {'✓' if (sunset_prob > 0.5) == (true_origin == 'ai') else '✗'}")
        print(f"  Ensemble AI prob:    {ensemble_prob:.3f}")
        print(f"  Humanization score:  {humanization_result['humanization_score']:.3f} {'✓' if humanization_result['is_likely_humanized'] == (humanization != 'none') else '✗'}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)
    
    categories = [
        ('human', 'none'),
        ('ai', 'none'),
        ('ai', 'light'),
        ('ai', 'medium'),
        ('ai', 'heavy')
    ]
    
    for origin, hum in categories:
        cat_results = [r for r in results if r['true_origin'] == origin and r['humanization'] == hum]
        if not cat_results:
            continue
        
        avg_sunrise = np.mean([r['sunrise_prob'] for r in cat_results])
        avg_sunset = np.mean([r['sunset_prob'] for r in cat_results])
        avg_ensemble = np.mean([r['ensemble_prob'] for r in cat_results])
        avg_humanization = np.mean([r['humanization_score'] for r in cat_results])
        
        label = f"{origin.upper()}" + (f" + {hum} humanization" if hum != 'none' else " (pure)")
        print(f"\n{label}:")
        print(f"  Sunrise avg:      {avg_sunrise:.3f}")
        print(f"  Sunset avg:       {avg_sunset:.3f}")
        print(f"  Ensemble avg:     {avg_ensemble:.3f}")
        print(f"  Humanization avg: {avg_humanization:.3f}")
    
    # Detection accuracy summary
    print("\n" + "=" * 70)
    print("DETECTION ACCURACY")
    print("=" * 70)
    
    # For pure texts (no humanization)
    pure_results = [r for r in results if r['humanization'] == 'none']
    
    sunrise_correct = sum(1 for r in pure_results if (r['sunrise_prob'] > 0.5) == (r['true_origin'] == 'ai'))
    sunset_correct = sum(1 for r in pure_results if (r['sunset_prob'] > 0.5) == (r['true_origin'] == 'ai'))
    ensemble_correct = sum(1 for r in pure_results if (r['ensemble_prob'] > 0.5) == (r['true_origin'] == 'ai'))
    
    print(f"\nPure text detection (no humanization):")
    print(f"  Sunrise:  {sunrise_correct}/{len(pure_results)} = {sunrise_correct/len(pure_results)*100:.1f}%")
    print(f"  Sunset:   {sunset_correct}/{len(pure_results)} = {sunset_correct/len(pure_results)*100:.1f}%")
    print(f"  Ensemble: {ensemble_correct}/{len(pure_results)} = {ensemble_correct/len(pure_results)*100:.1f}%")
    
    # For humanized texts
    humanized_results = [r for r in results if r['humanization'] != 'none']
    
    if humanized_results:
        sunrise_correct = sum(1 for r in humanized_results if r['sunrise_prob'] > 0.5)  # Should detect as AI
        sunset_correct = sum(1 for r in humanized_results if r['sunset_prob'] > 0.5)
        ensemble_correct = sum(1 for r in humanized_results if r['ensemble_prob'] > 0.5)
        humanization_correct = sum(1 for r in humanized_results if r['is_humanized_detected'])
        
        print(f"\nHumanized AI text detection:")
        print(f"  Sunrise (as AI):     {sunrise_correct}/{len(humanized_results)} = {sunrise_correct/len(humanized_results)*100:.1f}%")
        print(f"  Sunset (as AI):      {sunset_correct}/{len(humanized_results)} = {sunset_correct/len(humanized_results)*100:.1f}%")
        print(f"  Ensemble (as AI):    {ensemble_correct}/{len(humanized_results)} = {ensemble_correct/len(humanized_results)*100:.1f}%")
        print(f"  Humanization detector: {humanization_correct}/{len(humanized_results)} = {humanization_correct/len(humanized_results)*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_benchmark()
