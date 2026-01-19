#!/usr/bin/env python3
"""
Sunset Model Trainer
====================
GPTZero-style detection using perplexity proxies and burstiness metrics.

Key Metrics:
1. Perplexity Proxies (word predictability patterns)
2. Burstiness (sentence complexity variation)
3. Entropy-based features
"""

import os
import sys
import pickle
import hashlib
import json
import math
import re
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset

# Logging
LOG_FILE = "sunset_training.log"

def log(msg):
    """Log to file and stdout"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


class SunsetFeatureExtractor:
    """
    GPTZero-style feature extraction focusing on:
    - Perplexity proxies (n-gram predictability, word frequency patterns)
    - Burstiness (sentence complexity variance)
    - Entropy metrics
    """
    
    # Common English words for frequency analysis
    COMMON_WORDS = set([
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
        "when", "make", "can", "like", "time", "no", "just", "him", "know", "take",
        "people", "into", "year", "your", "good", "some", "could", "them", "see", "other",
        "than", "then", "now", "look", "only", "come", "its", "over", "think", "also"
    ])
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        return [
            # Perplexity proxies
            'word_predictability_score',
            'bigram_predictability',
            'trigram_predictability',
            'word_frequency_uniformity',
            'rare_word_ratio',
            'common_word_ratio',
            'word_length_entropy',
            'vocabulary_entropy',
            
            # Burstiness metrics
            'sentence_length_burstiness',
            'word_length_burstiness',
            'complexity_burstiness',
            'punctuation_burstiness',
            
            # Sentence-level variance
            'sentence_length_cv',
            'sentence_length_range_norm',
            'sentence_length_skewness',
            'sentence_length_kurtosis',
            
            # Word-level patterns
            'avg_word_length',
            'word_length_variance',
            'long_word_ratio',
            'short_word_ratio',
            
            # Vocabulary richness
            'type_token_ratio',
            'hapax_ratio',
            'yule_k',
            
            # N-gram entropy
            'unigram_entropy',
            'bigram_entropy',
            'trigram_entropy',
            
            # Repetition patterns
            'bigram_repetition_rate',
            'trigram_repetition_rate',
            'word_repetition_rate',
            
            # Punctuation patterns
            'comma_rate',
            'period_rate',
            'question_rate',
            'semicolon_rate',
            
            # Structural patterns
            'avg_sentence_length',
            'paragraph_uniformity',
            'sentence_start_diversity',
        ]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return re.findall(r"[a-zA-Z']+", text.lower())
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences"""
        # Handle abbreviations
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.',
                      r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        return [s for s in sentences if len(s.split()) >= 3]
    
    def _calculate_entropy(self, items: List[str]) -> float:
        """Calculate Shannon entropy"""
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
    
    def _calculate_burstiness(self, values: List[float]) -> float:
        """
        Calculate burstiness coefficient.
        B = (σ - μ) / (σ + μ)
        Human text has high burstiness (more variance)
        AI text has low burstiness (uniform)
        """
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        std = np.std(values)
        if mean + std == 0:
            return 0.0
        return (std - mean) / (std + mean)
    
    def _calculate_cv(self, values: List[float]) -> float:
        """Coefficient of variation"""
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return np.std(values) / mean
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness"""
        if len(values) < 3:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        skew = sum((x - mean) ** 3 for x in values) / n
        return skew / (std ** 3)
    
    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate excess kurtosis"""
        if len(values) < 4:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        kurt = sum((x - mean) ** 4 for x in values) / n
        return (kurt / (std ** 4)) - 3
    
    def _calculate_yule_k(self, words: List[str]) -> float:
        """Yule's K characteristic - vocabulary richness"""
        if len(words) < 10:
            return 0.0
        freq = Counter(words)
        m1 = len(words)
        m2 = sum(f * f for f in freq.values())
        if m1 == 0:
            return 0.0
        return 10000 * (m2 - m1) / (m1 * m1)
    
    def extract(self, text: str) -> np.ndarray:
        """Extract all features from text"""
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        if len(words) < 10 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # === PERPLEXITY PROXIES ===
        
        # Word predictability (common words = more predictable)
        common_count = sum(1 for w in words if w in self.COMMON_WORDS)
        features['common_word_ratio'] = common_count / len(words)
        features['rare_word_ratio'] = 1 - features['common_word_ratio']
        
        # Word frequency uniformity (AI tends to have more uniform distribution)
        word_freq = Counter(words)
        freq_values = list(word_freq.values())
        features['word_frequency_uniformity'] = 1 - self._calculate_cv(freq_values) if freq_values else 0
        
        # Word predictability score (proxy for perplexity)
        # Lower entropy in word choices = more predictable = lower perplexity
        unique_ratio = len(set(words)) / len(words)
        features['word_predictability_score'] = 1 - unique_ratio
        
        # Bigram predictability
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        bigram_freq = Counter(bigrams)
        repeated_bigrams = sum(1 for c in bigram_freq.values() if c > 1)
        features['bigram_predictability'] = repeated_bigrams / len(bigrams) if bigrams else 0
        
        # Trigram predictability
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        trigram_freq = Counter(trigrams)
        repeated_trigrams = sum(1 for c in trigram_freq.values() if c > 1)
        features['trigram_predictability'] = repeated_trigrams / len(trigrams) if trigrams else 0
        
        # Word length entropy
        word_lengths = [len(w) for w in words]
        length_str = [str(l) for l in word_lengths]
        features['word_length_entropy'] = self._calculate_entropy(length_str)
        
        # Vocabulary entropy
        features['vocabulary_entropy'] = self._calculate_entropy(words)
        
        # === BURSTINESS METRICS ===
        
        # Sentence length burstiness
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_burstiness'] = self._calculate_burstiness(sent_lengths)
        
        # Word length burstiness
        features['word_length_burstiness'] = self._calculate_burstiness(word_lengths)
        
        # Complexity burstiness (syllables per word per sentence)
        sent_complexities = []
        for sent in sentences:
            sent_words = self._tokenize(sent)
            if sent_words:
                avg_syllables = np.mean([self._count_syllables(w) for w in sent_words])
                sent_complexities.append(avg_syllables)
        features['complexity_burstiness'] = self._calculate_burstiness(sent_complexities) if sent_complexities else 0
        
        # Punctuation burstiness (punctuation per sentence)
        punct_per_sent = []
        for sent in sentences:
            punct_count = sum(1 for c in sent if c in '.,;:!?-')
            punct_per_sent.append(punct_count)
        features['punctuation_burstiness'] = self._calculate_burstiness(punct_per_sent)
        
        # === SENTENCE-LEVEL VARIANCE ===
        
        features['sentence_length_cv'] = self._calculate_cv(sent_lengths)
        features['sentence_length_range_norm'] = (max(sent_lengths) - min(sent_lengths)) / np.mean(sent_lengths) if sent_lengths else 0
        features['sentence_length_skewness'] = self._calculate_skewness(sent_lengths)
        features['sentence_length_kurtosis'] = self._calculate_kurtosis(sent_lengths)
        
        # === WORD-LEVEL PATTERNS ===
        
        features['avg_word_length'] = np.mean(word_lengths)
        features['word_length_variance'] = np.var(word_lengths)
        features['long_word_ratio'] = sum(1 for l in word_lengths if l > 8) / len(words)
        features['short_word_ratio'] = sum(1 for l in word_lengths if l <= 3) / len(words)
        
        # === VOCABULARY RICHNESS ===
        
        features['type_token_ratio'] = len(set(words)) / len(words)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        features['hapax_ratio'] = hapax / len(words)
        features['yule_k'] = self._calculate_yule_k(words)
        
        # === N-GRAM ENTROPY ===
        
        features['unigram_entropy'] = self._calculate_entropy(words)
        features['bigram_entropy'] = self._calculate_entropy(bigrams)
        features['trigram_entropy'] = self._calculate_entropy(trigrams)
        
        # === REPETITION PATTERNS ===
        
        features['bigram_repetition_rate'] = features['bigram_predictability']
        features['trigram_repetition_rate'] = features['trigram_predictability']
        features['word_repetition_rate'] = 1 - features['type_token_ratio']
        
        # === PUNCTUATION PATTERNS ===
        
        char_count = len(text)
        features['comma_rate'] = text.count(',') / char_count * 100
        features['period_rate'] = text.count('.') / char_count * 100
        features['question_rate'] = text.count('?') / char_count * 100
        features['semicolon_rate'] = text.count(';') / char_count * 100
        
        # === STRUCTURAL PATTERNS ===
        
        features['avg_sentence_length'] = np.mean(sent_lengths)
        
        # Paragraph uniformity
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - self._calculate_cv(para_lengths)
        else:
            features['paragraph_uniformity'] = 0.5
        
        # Sentence start diversity
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        
        # Build feature vector in correct order
        feature_vector = [features.get(name, 0.0) for name in self.feature_names]
        
        # Handle NaN/Inf
        feature_vector = [0.0 if not np.isfinite(v) else v for v in feature_vector]
        
        return np.array(feature_vector)
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
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


def load_training_data(max_samples: int = 15000) -> Tuple[List[str], List[int]]:
    """Load balanced human/AI training data"""
    log(f"Loading training data (max {max_samples} samples)...")
    
    texts = []
    labels = []
    
    half = max_samples // 2
    
    # Load GPT-wiki-intro (has both human and AI)
    try:
        log("Loading GPT-wiki-intro dataset...")
        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
        
        human_count = 0
        ai_count = 0
        
        for item in ds:
            if human_count >= half and ai_count >= half:
                break
            
            # Human text
            if human_count < half and 'wiki_intro' in item and item['wiki_intro']:
                text = item['wiki_intro'].strip()
                if len(text.split()) >= 20:
                    texts.append(text)
                    labels.append(0)  # Human
                    human_count += 1
            
            # AI text
            if ai_count < half and 'generated_intro' in item and item['generated_intro']:
                text = item['generated_intro'].strip()
                if len(text.split()) >= 20:
                    texts.append(text)
                    labels.append(1)  # AI
                    ai_count += 1
            
            if (human_count + ai_count) % 2000 == 0:
                log(f"  Loaded {human_count} human, {ai_count} AI samples")
        
        log(f"Final: {human_count} human, {ai_count} AI samples from GPT-wiki-intro")
        
    except Exception as e:
        log(f"Error loading data: {e}")
        raise
    
    return texts, labels


def train_sunset():
    """Train the Sunset model"""
    log("=" * 60)
    log("SUNSET MODEL TRAINING")
    log("GPTZero-style Perplexity + Burstiness Detection")
    log("=" * 60)
    
    start_time = datetime.now()
    
    # Load data
    texts, labels = load_training_data(max_samples=20000)
    
    log(f"\nTotal samples: {len(texts)}")
    log(f"Human: {labels.count(0)}, AI: {labels.count(1)}")
    
    # Extract features
    log("\nExtracting features...")
    extractor = SunsetFeatureExtractor()
    
    X = []
    valid_labels = []
    
    for i, text in enumerate(texts):
        try:
            features = extractor.extract(text)
            if np.any(features != 0):  # Skip empty feature vectors
                X.append(features)
                valid_labels.append(labels[i])
        except Exception as e:
            pass
        
        if (i + 1) % 2000 == 0:
            log(f"  Processed {i + 1}/{len(texts)} texts")
    
    X = np.array(X)
    y = np.array(valid_labels)
    
    log(f"\nFeature matrix shape: {X.shape}")
    log(f"Valid samples: {len(y)} (Human: {sum(y==0)}, AI: {sum(y==1)})")
    
    # Scale features
    log("\nScaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    log(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    log("\nTraining RandomForest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    log("\nEvaluating...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    
    log(f"\n{'=' * 40}")
    log("RESULTS:")
    log(f"{'=' * 40}")
    log(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    log(f"Precision: {precision:.4f}")
    log(f"Recall:    {recall:.4f}")
    log(f"F1 Score:  {f1:.4f}")
    log(f"ROC-AUC:   {roc_auc:.4f}")
    log(f"\nConfusion Matrix:")
    log(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    log(f"  FN={cm[1][0]}, TP={cm[1][1]}")
    
    # Feature importance
    importance = dict(zip(extractor.feature_names, model.feature_importances_))
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    log(f"\nTop 10 Features:")
    for name, imp in sorted_importance[:10]:
        log(f"  {name}: {imp:.4f}")
    
    # Save model
    model_dir = "/workspaces/Veritas/training/models/Sunset"
    os.makedirs(model_dir, exist_ok=True)
    
    log(f"\nSaving model to {model_dir}...")
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Metadata
    metadata = {
        "model_name": "Sunset",
        "version": "1.0.0",
        "methodology": "GPTZero-style perplexity + burstiness",
        "feature_names": extractor.feature_names,
        "feature_count": len(extractor.feature_names),
        "training_samples": len(y),
        "human_samples": int(sum(y == 0)),
        "ai_samples": int(sum(y == 1)),
        "results": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc)
        },
        "feature_importance": {k: float(v) for k, v in sorted_importance},
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Training receipt
    receipt = {
        "model_name": "Sunset",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "datasets_used": ["aadityaubhat/GPT-wiki-intro"],
        "total_samples": len(y),
        "human_samples": int(sum(y == 0)),
        "ai_samples": int(sum(y == 1)),
        "results": metadata["results"],
        "top_features": sorted_importance[:15],
        "training_time_seconds": (datetime.now() - start_time).total_seconds()
    }
    
    with open(f"{model_dir}/training_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
    
    # Veritas config
    veritas_config = f"""// Sunset Model Configuration
// GPTZero-style Perplexity + Burstiness Detection
// Generated: {datetime.now().isoformat()}

const SunsetConfig = {{
    name: 'Sunset',
    version: '1.0.0',
    methodology: 'GPTZero-style perplexity + burstiness',
    featureCount: {len(extractor.feature_names)},
    features: {json.dumps(extractor.feature_names, indent=8)},
    featureWeights: {json.dumps({k: round(v, 4) for k, v in sorted_importance[:15]}, indent=8)},
    metrics: {{
        accuracy: {accuracy:.4f},
        precision: {precision:.4f},
        recall: {recall:.4f},
        f1: {f1:.4f},
        rocAuc: {roc_auc:.4f}
    }}
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = SunsetConfig;
}}
"""
    
    with open(f"{model_dir}/veritas_config.js", "w") as f:
        f.write(veritas_config)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    log(f"\nTraining complete in {elapsed:.1f} seconds")
    log(f"Model saved to {model_dir}")
    
    # Check model size
    model_size = os.path.getsize(f"{model_dir}/model.pkl") / (1024 * 1024)
    log(f"Model size: {model_size:.2f} MB")
    
    return accuracy, f1


if __name__ == "__main__":
    train_sunset()
