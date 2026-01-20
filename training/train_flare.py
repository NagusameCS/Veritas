#!/usr/bin/env python3
"""
FLARE Detection System Training Script
=======================================
Specialized model for detecting humanized AI content vs genuine human writing.

Unlike other models that focus on AI vs Human, Flare assumes the content
appears human-like and specifically looks for signs of humanization:
- Artificial variance injection
- Broken natural correlations  
- Synonym substitution patterns
- Mechanical contraction insertion
- Surface-level chaos with deep structural uniformity

Training Data Sources (HuggingFace):
- Human samples: Genuine human writing from multiple sources
- Humanized samples: AI text processed through humanization tools
"""

import os
import json
import random
import math
import re
import sys
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Try to import HuggingFace datasets
try:
    from datasets import load_dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    print("Installing datasets library...")
    os.system(f"{sys.executable} -m pip install datasets")
    from datasets import load_dataset, concatenate_datasets
    HF_AVAILABLE = True


class FlareFeatureExtractor:
    """
    Specialized feature extraction for humanization detection.
    Focuses on second-order patterns that humanizers fail to replicate.
    """
    
    def __init__(self):
        # Common words that humanizers often substitute
        self.synonym_clusters = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb', 'outstanding'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor', 'disappointing'],
            'big': ['large', 'huge', 'enormous', 'massive', 'substantial', 'significant'],
            'small': ['tiny', 'little', 'minute', 'minor', 'slight', 'minimal'],
            'important': ['crucial', 'vital', 'essential', 'critical', 'significant', 'key'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'enable', 'contribute'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'display', 'exhibit'],
            'use': ['utilize', 'employ', 'leverage', 'apply', 'implement', 'harness'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build'],
            'get': ['obtain', 'acquire', 'receive', 'gain', 'secure', 'attain'],
        }
        
        # AI-typical hedging phrases
        self.hedging_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'generally speaking', 'in many cases', 'it could be argued',
            'from this perspective', 'in this context', 'as we can see',
            'this suggests that', 'this indicates that', 'this demonstrates',
        ]
        
        # Transition markers AI overuses
        self.transition_markers = [
            'furthermore', 'moreover', 'additionally', 'consequently', 'therefore',
            'however', 'nevertheless', 'nonetheless', 'subsequently', 'accordingly',
        ]
        
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all humanization-detection features."""
        features = {}
        
        # Tokenize
        sentences = self._split_sentences(text)
        words = self._tokenize(text)
        
        if len(words) < 20 or len(sentences) < 3:
            return self._empty_features()
        
        # 1. Variance-of-Variance Analysis (humanizers add first-order variance but stable second-order)
        features.update(self._variance_analysis(sentences, words))
        
        # 2. Autocorrelation Pattern Analysis
        features.update(self._autocorrelation_analysis(sentences))
        
        # 3. Feature Correlation Analysis (humanizers break natural correlations)
        features.update(self._correlation_analysis(text, sentences, words))
        
        # 4. Synonym Substitution Detection
        features.update(self._synonym_analysis(words))
        
        # 5. Contraction Pattern Analysis
        features.update(self._contraction_analysis(text, sentences))
        
        # 6. Sentence Structure Uniformity
        features.update(self._structure_analysis(sentences))
        
        # 7. Lexical Sophistication Consistency
        features.update(self._sophistication_analysis(words, sentences))
        
        # 8. N-gram Predictability
        features.update(self._ngram_analysis(words))
        
        # 9. Punctuation Pattern Analysis
        features.update(self._punctuation_analysis(text, sentences))
        
        # 10. Discourse Flow Analysis
        features.update(self._discourse_analysis(text, sentences))
        
        # 11. Word Length Distribution
        features.update(self._word_length_analysis(words))
        
        # 12. Entropy Analysis
        features.update(self._entropy_analysis(words, sentences))
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict for short texts."""
        return {f: 0.0 for f in self.get_feature_names()}
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return [
            # Variance analysis
            'variance_of_variance', 'variance_stability', 'local_variance_consistency',
            # Autocorrelation
            'autocorr_decay_rate', 'autocorr_flatness', 'autocorr_periodicity',
            # Correlation
            'length_complexity_corr', 'vocab_structure_corr', 'feature_correlation_breaks',
            # Synonym
            'synonym_cluster_usage', 'rare_synonym_ratio', 'sophistication_jumps',
            # Contraction
            'contraction_rate', 'contraction_uniformity', 'artificial_contraction_score',
            # Structure
            'sentence_start_diversity', 'structure_template_score', 'parallelism_score',
            # Sophistication
            'sophistication_variance', 'sophistication_autocorr', 'word_choice_consistency',
            # N-gram
            'bigram_predictability', 'trigram_predictability', 'ngram_surprise_variance',
            # Punctuation
            'comma_density', 'punctuation_variety', 'punctuation_position_entropy',
            # Discourse
            'transition_density', 'hedging_density', 'discourse_marker_variety',
            # Word length
            'word_length_variance', 'word_length_entropy', 'long_word_clustering',
            # Entropy
            'lexical_entropy', 'sentence_entropy', 'entropy_stability',
        ]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def _variance_analysis(self, sentences: List[str], words: List[str]) -> Dict[str, float]:
        """Analyze variance patterns - key humanization indicator."""
        features = {}
        
        # Sentence length variance
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if len(sent_lengths) < 3:
            return {'variance_of_variance': 0.5, 'variance_stability': 0.5, 'local_variance_consistency': 0.5}
        
        # First-order variance
        mean_len = np.mean(sent_lengths)
        first_order_var = np.var(sent_lengths) if mean_len > 0 else 0
        
        # Second-order variance (variance of local variances)
        window_size = max(3, len(sent_lengths) // 4)
        local_variances = []
        for i in range(0, len(sent_lengths) - window_size + 1):
            window = sent_lengths[i:i + window_size]
            local_variances.append(np.var(window))
        
        if local_variances:
            variance_of_variance = np.var(local_variances)
            # Humanized text: high first-order variance, low variance-of-variance
            # Normalize
            vov_normalized = min(1.0, variance_of_variance / (first_order_var + 1))
            features['variance_of_variance'] = vov_normalized
            features['variance_stability'] = 1.0 - min(1.0, np.std(local_variances) / (np.mean(local_variances) + 0.1))
        else:
            features['variance_of_variance'] = 0.5
            features['variance_stability'] = 0.5
        
        # Local variance consistency (humanizers produce too-consistent local variance)
        if len(local_variances) > 2:
            cv = np.std(local_variances) / (np.mean(local_variances) + 0.1)
            features['local_variance_consistency'] = 1.0 - min(1.0, cv)
        else:
            features['local_variance_consistency'] = 0.5
        
        return features
    
    def _autocorrelation_analysis(self, sentences: List[str]) -> Dict[str, float]:
        """Analyze autocorrelation patterns."""
        features = {}
        
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if len(sent_lengths) < 5:
            return {'autocorr_decay_rate': 0.5, 'autocorr_flatness': 0.5, 'autocorr_periodicity': 0.5}
        
        # Compute autocorrelation at different lags
        autocorrs = []
        mean_len = np.mean(sent_lengths)
        var_len = np.var(sent_lengths)
        
        if var_len < 0.01:
            return {'autocorr_decay_rate': 0.5, 'autocorr_flatness': 1.0, 'autocorr_periodicity': 0.0}
        
        for lag in range(1, min(6, len(sent_lengths) // 2)):
            cov = np.mean([(sent_lengths[i] - mean_len) * (sent_lengths[i + lag] - mean_len) 
                          for i in range(len(sent_lengths) - lag)])
            autocorrs.append(cov / var_len)
        
        if not autocorrs:
            return {'autocorr_decay_rate': 0.5, 'autocorr_flatness': 0.5, 'autocorr_periodicity': 0.5}
        
        # Decay rate (human: gradual decay; AI: flat or periodic; humanized: random noise)
        if len(autocorrs) > 1:
            decay_rate = (autocorrs[0] - autocorrs[-1]) / len(autocorrs) if len(autocorrs) > 0 else 0
            features['autocorr_decay_rate'] = min(1.0, max(0.0, decay_rate + 0.5))
        else:
            features['autocorr_decay_rate'] = 0.5
        
        # Flatness (humanized tends to be flat/random)
        autocorr_var = np.var(autocorrs)
        features['autocorr_flatness'] = 1.0 - min(1.0, autocorr_var * 10)
        
        # Periodicity detection
        if len(autocorrs) >= 3:
            # Check for alternating pattern
            alternating = sum(1 for i in range(len(autocorrs) - 1) 
                            if autocorrs[i] * autocorrs[i + 1] < 0)
            features['autocorr_periodicity'] = alternating / (len(autocorrs) - 1)
        else:
            features['autocorr_periodicity'] = 0.0
        
        return features
    
    def _correlation_analysis(self, text: str, sentences: List[str], words: List[str]) -> Dict[str, float]:
        """Analyze natural feature correlations that humanizers break."""
        features = {}
        
        if len(sentences) < 5:
            return {'length_complexity_corr': 0.5, 'vocab_structure_corr': 0.5, 'feature_correlation_breaks': 0.5}
        
        # Length vs complexity correlation (longer sentences should be more complex)
        sent_data = []
        for s in sentences:
            s_words = self._tokenize(s)
            if len(s_words) > 0:
                complexity = sum(1 for w in s_words if len(w) > 6) / len(s_words)
                sent_data.append((len(s_words), complexity))
        
        if len(sent_data) >= 5:
            lengths = [d[0] for d in sent_data]
            complexities = [d[1] for d in sent_data]
            corr = self._pearson_correlation(lengths, complexities)
            # Natural human text has positive correlation
            features['length_complexity_corr'] = (corr + 1) / 2
        else:
            features['length_complexity_corr'] = 0.5
        
        # Vocabulary richness vs structure correlation
        unique_ratios = []
        struct_scores = []
        window = 3
        for i in range(0, len(sentences) - window + 1):
            chunk = sentences[i:i + window]
            chunk_words = [w for s in chunk for w in self._tokenize(s)]
            if len(chunk_words) > 0:
                unique_ratios.append(len(set(chunk_words)) / len(chunk_words))
                # Structure: sentence length variance in chunk
                chunk_lens = [len(self._tokenize(s)) for s in chunk]
                struct_scores.append(np.std(chunk_lens) / (np.mean(chunk_lens) + 0.1))
        
        if len(unique_ratios) >= 3:
            corr = self._pearson_correlation(unique_ratios, struct_scores)
            features['vocab_structure_corr'] = (corr + 1) / 2
        else:
            features['vocab_structure_corr'] = 0.5
        
        # Overall correlation break score
        expected_corr = 0.3  # Natural correlation expected
        actual_corr = features.get('length_complexity_corr', 0.5)
        features['feature_correlation_breaks'] = abs(actual_corr - expected_corr - 0.5)
        
        return features
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient."""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x, mean_y = np.mean(x), np.mean(y)
        std_x, std_y = np.std(x), np.std(y)
        
        if std_x < 0.0001 or std_y < 0.0001:
            return 0.0
        
        cov = np.mean([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
        return cov / (std_x * std_y)
    
    def _synonym_analysis(self, words: List[str]) -> Dict[str, float]:
        """Detect synonym substitution patterns."""
        features = {}
        
        if len(words) < 20:
            return {'synonym_cluster_usage': 0.5, 'rare_synonym_ratio': 0.5, 'sophistication_jumps': 0.5}
        
        # Check for rare synonym usage
        word_freq = Counter(words)
        
        rare_synonyms_used = 0
        common_words_used = 0
        
        for base_word, synonyms in self.synonym_clusters.items():
            if base_word in word_freq:
                common_words_used += word_freq[base_word]
            for syn in synonyms:
                if syn in word_freq:
                    rare_synonyms_used += word_freq[syn]
        
        total_cluster_words = rare_synonyms_used + common_words_used
        if total_cluster_words > 0:
            features['rare_synonym_ratio'] = rare_synonyms_used / total_cluster_words
        else:
            features['rare_synonym_ratio'] = 0.0
        
        # Synonym cluster usage density
        features['synonym_cluster_usage'] = total_cluster_words / len(words)
        
        # Sophistication jumps (sudden changes in word complexity)
        sophistication_scores = []
        for i in range(0, len(words) - 5, 5):
            window = words[i:i + 5]
            avg_len = np.mean([len(w) for w in window])
            sophistication_scores.append(avg_len)
        
        if len(sophistication_scores) > 2:
            jumps = [abs(sophistication_scores[i + 1] - sophistication_scores[i]) 
                    for i in range(len(sophistication_scores) - 1)]
            features['sophistication_jumps'] = np.mean(jumps) / 3.0  # Normalize
        else:
            features['sophistication_jumps'] = 0.0
        
        return features
    
    def _contraction_analysis(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Analyze contraction usage patterns."""
        features = {}
        
        contractions = re.findall(r"\b\w+'\w+\b", text.lower())
        total_words = len(re.findall(r'\b\w+\b', text))
        
        if total_words < 20:
            return {'contraction_rate': 0.5, 'contraction_uniformity': 0.5, 'artificial_contraction_score': 0.5}
        
        # Contraction rate
        features['contraction_rate'] = min(1.0, len(contractions) / (total_words / 20))
        
        # Contraction uniformity (artificial insertion = too uniform distribution)
        if len(sentences) >= 3 and len(contractions) >= 2:
            contractions_per_sent = []
            for s in sentences:
                sent_contractions = len(re.findall(r"\b\w+'\w+\b", s.lower()))
                contractions_per_sent.append(sent_contractions)
            
            if np.mean(contractions_per_sent) > 0:
                cv = np.std(contractions_per_sent) / (np.mean(contractions_per_sent) + 0.1)
                features['contraction_uniformity'] = 1.0 - min(1.0, cv)
            else:
                features['contraction_uniformity'] = 0.5
        else:
            features['contraction_uniformity'] = 0.5
        
        # Artificial contraction score (combines rate + uniformity)
        if features['contraction_rate'] > 0.3 and features['contraction_uniformity'] > 0.7:
            features['artificial_contraction_score'] = (features['contraction_rate'] + features['contraction_uniformity']) / 2
        else:
            features['artificial_contraction_score'] = 0.0
        
        return features
    
    def _structure_analysis(self, sentences: List[str]) -> Dict[str, float]:
        """Analyze sentence structure patterns."""
        features = {}
        
        if len(sentences) < 3:
            return {'sentence_start_diversity': 0.5, 'structure_template_score': 0.5, 'parallelism_score': 0.5}
        
        # Sentence start diversity
        starts = [self._tokenize(s)[0] if self._tokenize(s) else '' for s in sentences]
        starts = [s for s in starts if s]
        
        if starts:
            unique_starts = len(set(starts))
            features['sentence_start_diversity'] = unique_starts / len(starts)
        else:
            features['sentence_start_diversity'] = 0.5
        
        # Structure template score (AI tends to repeat structures)
        # Use first 3 words as structure signature
        signatures = []
        for s in sentences:
            words = self._tokenize(s)[:3]
            sig = '_'.join([w[0] if w else '' for w in words])  # First letter pattern
            signatures.append(sig)
        
        sig_counts = Counter(signatures)
        repeated_sigs = sum(1 for c in sig_counts.values() if c > 1)
        features['structure_template_score'] = repeated_sigs / len(sentences) if sentences else 0.5
        
        # Parallelism score
        len_pairs = list(zip(sentences[:-1], sentences[1:]))
        similar_pairs = sum(1 for a, b in len_pairs 
                          if abs(len(self._tokenize(a)) - len(self._tokenize(b))) <= 2)
        features['parallelism_score'] = similar_pairs / len(len_pairs) if len_pairs else 0.5
        
        return features
    
    def _sophistication_analysis(self, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Analyze lexical sophistication consistency."""
        features = {}
        
        if len(words) < 20 or len(sentences) < 3:
            return {'sophistication_variance': 0.5, 'sophistication_autocorr': 0.5, 'word_choice_consistency': 0.5}
        
        # Per-sentence sophistication (avg word length as proxy)
        sent_sophistication = []
        for s in sentences:
            s_words = self._tokenize(s)
            if s_words:
                avg_len = np.mean([len(w) for w in s_words])
                sent_sophistication.append(avg_len)
        
        if len(sent_sophistication) >= 3:
            features['sophistication_variance'] = min(1.0, np.var(sent_sophistication) / 4)
            
            # Autocorrelation of sophistication
            if len(sent_sophistication) >= 4:
                lag1_pairs = list(zip(sent_sophistication[:-1], sent_sophistication[1:]))
                corr = self._pearson_correlation(
                    [p[0] for p in lag1_pairs], 
                    [p[1] for p in lag1_pairs]
                )
                features['sophistication_autocorr'] = (corr + 1) / 2
            else:
                features['sophistication_autocorr'] = 0.5
        else:
            features['sophistication_variance'] = 0.5
            features['sophistication_autocorr'] = 0.5
        
        # Word choice consistency (vocab richness over windows)
        window_richness = []
        for i in range(0, len(words) - 10, 10):
            window = words[i:i + 10]
            window_richness.append(len(set(window)) / 10)
        
        if len(window_richness) > 2:
            features['word_choice_consistency'] = 1.0 - min(1.0, np.std(window_richness) * 5)
        else:
            features['word_choice_consistency'] = 0.5
        
        return features
    
    def _ngram_analysis(self, words: List[str]) -> Dict[str, float]:
        """Analyze n-gram predictability patterns."""
        features = {}
        
        if len(words) < 30:
            return {'bigram_predictability': 0.5, 'trigram_predictability': 0.5, 'ngram_surprise_variance': 0.5}
        
        # Bigram analysis
        bigrams = list(zip(words[:-1], words[1:]))
        bigram_counts = Counter(bigrams)
        
        repeated_bigrams = sum(1 for c in bigram_counts.values() if c > 1)
        features['bigram_predictability'] = repeated_bigrams / len(bigrams) if bigrams else 0.5
        
        # Trigram analysis
        trigrams = list(zip(words[:-2], words[1:-1], words[2:]))
        trigram_counts = Counter(trigrams)
        
        repeated_trigrams = sum(1 for c in trigram_counts.values() if c > 1)
        features['trigram_predictability'] = repeated_trigrams / len(trigrams) if trigrams else 0.5
        
        # N-gram surprise variance
        # Calculate "surprise" of each word given previous context
        word_given_prev = defaultdict(Counter)
        for w1, w2 in bigrams:
            word_given_prev[w1][w2] += 1
        
        surprises = []
        for i, (w1, w2) in enumerate(bigrams[1:], 1):
            total = sum(word_given_prev[w1].values())
            if total > 0:
                prob = word_given_prev[w1][w2] / total
                surprise = -math.log(prob + 0.01)
                surprises.append(surprise)
        
        if surprises:
            features['ngram_surprise_variance'] = min(1.0, np.var(surprises) / 2)
        else:
            features['ngram_surprise_variance'] = 0.5
        
        return features
    
    def _punctuation_analysis(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Analyze punctuation patterns."""
        features = {}
        
        words = len(re.findall(r'\b\w+\b', text))
        if words < 20:
            return {'comma_density': 0.5, 'punctuation_variety': 0.5, 'punctuation_position_entropy': 0.5}
        
        # Comma density
        commas = text.count(',')
        features['comma_density'] = min(1.0, commas / (words / 10))
        
        # Punctuation variety
        punct_chars = re.findall(r'[^\w\s]', text)
        if punct_chars:
            unique_punct = len(set(punct_chars))
            features['punctuation_variety'] = min(1.0, unique_punct / 8)
        else:
            features['punctuation_variety'] = 0.0
        
        # Punctuation position entropy (where commas appear in sentences)
        comma_positions = []
        for s in sentences:
            s_words = s.split()
            for i, w in enumerate(s_words):
                if ',' in w:
                    rel_pos = i / len(s_words) if s_words else 0.5
                    comma_positions.append(rel_pos)
        
        if len(comma_positions) >= 3:
            # Calculate entropy of positions
            bins = [0, 0.25, 0.5, 0.75, 1.0]
            hist, _ = np.histogram(comma_positions, bins=bins)
            hist = hist / (sum(hist) + 0.001)
            entropy = -sum(p * math.log(p + 0.001) for p in hist if p > 0)
            features['punctuation_position_entropy'] = min(1.0, entropy / 1.4)  # Max entropy ~1.4
        else:
            features['punctuation_position_entropy'] = 0.5
        
        return features
    
    def _discourse_analysis(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Analyze discourse patterns."""
        features = {}
        
        text_lower = text.lower()
        words = len(re.findall(r'\b\w+\b', text))
        
        if words < 20:
            return {'transition_density': 0.5, 'hedging_density': 0.5, 'discourse_marker_variety': 0.5}
        
        # Transition marker density
        transition_count = sum(1 for t in self.transition_markers if t in text_lower)
        features['transition_density'] = min(1.0, transition_count / (len(sentences) / 3 + 0.1))
        
        # Hedging density
        hedging_count = sum(1 for h in self.hedging_phrases if h in text_lower)
        features['hedging_density'] = min(1.0, hedging_count / (len(sentences) / 5 + 0.1))
        
        # Discourse marker variety
        markers_found = [t for t in self.transition_markers if t in text_lower]
        if transition_count > 0:
            features['discourse_marker_variety'] = len(set(markers_found)) / transition_count
        else:
            features['discourse_marker_variety'] = 0.5
        
        return features
    
    def _word_length_analysis(self, words: List[str]) -> Dict[str, float]:
        """Analyze word length distribution."""
        features = {}
        
        if len(words) < 20:
            return {'word_length_variance': 0.5, 'word_length_entropy': 0.5, 'long_word_clustering': 0.5}
        
        lengths = [len(w) for w in words]
        
        # Word length variance
        features['word_length_variance'] = min(1.0, np.var(lengths) / 10)
        
        # Word length entropy
        length_counts = Counter(lengths)
        total = sum(length_counts.values())
        probs = [c / total for c in length_counts.values()]
        entropy = -sum(p * math.log(p + 0.001) for p in probs if p > 0)
        features['word_length_entropy'] = min(1.0, entropy / 3)
        
        # Long word clustering (humanizers may cluster complex words)
        long_word_positions = [i for i, w in enumerate(words) if len(w) >= 8]
        if len(long_word_positions) >= 2:
            gaps = [long_word_positions[i + 1] - long_word_positions[i] 
                   for i in range(len(long_word_positions) - 1)]
            # High variance = natural clustering, low variance = artificial distribution
            features['long_word_clustering'] = 1.0 - min(1.0, np.var(gaps) / 50) if gaps else 0.5
        else:
            features['long_word_clustering'] = 0.5
        
        return features
    
    def _entropy_analysis(self, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Analyze entropy patterns."""
        features = {}
        
        if len(words) < 20:
            return {'lexical_entropy': 0.5, 'sentence_entropy': 0.5, 'entropy_stability': 0.5}
        
        # Lexical entropy
        word_counts = Counter(words)
        total = sum(word_counts.values())
        probs = [c / total for c in word_counts.values()]
        lex_entropy = -sum(p * math.log(p + 0.001) for p in probs if p > 0)
        # Normalize by maximum possible entropy
        max_entropy = math.log(len(set(words)) + 1)
        features['lexical_entropy'] = lex_entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Sentence length entropy
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if sent_lengths:
            length_counts = Counter(sent_lengths)
            total = sum(length_counts.values())
            probs = [c / total for c in length_counts.values()]
            sent_entropy = -sum(p * math.log(p + 0.001) for p in probs if p > 0)
            max_entropy = math.log(len(set(sent_lengths)) + 1)
            features['sentence_entropy'] = sent_entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            features['sentence_entropy'] = 0.5
        
        # Entropy stability (window-based)
        window_entropies = []
        window_size = 20
        for i in range(0, len(words) - window_size, window_size // 2):
            window = words[i:i + window_size]
            wc = Counter(window)
            probs = [c / len(window) for c in wc.values()]
            ent = -sum(p * math.log(p + 0.001) for p in probs if p > 0)
            window_entropies.append(ent)
        
        if len(window_entropies) > 2:
            features['entropy_stability'] = 1.0 - min(1.0, np.std(window_entropies) * 2)
        else:
            features['entropy_stability'] = 0.5
        
        return features


class FlareDatasetLoader:
    """Load humanized vs human samples from HuggingFace."""
    
    def __init__(self):
        self.human_samples = []
        self.humanized_samples = []
    
    def load_datasets(self, max_samples: int = 50000) -> Tuple[List[str], List[str]]:
        """Load training data from HuggingFace."""
        print("=" * 60)
        print("FLARE Dataset Loading")
        print("=" * 60)
        
        # Human text sources - diverse, authentic human writing
        human_sources = [
            ('reddit_casual', self._load_reddit_casual),
            ('essays', self._load_essays),
            ('amazon_reviews', self._load_amazon_reviews),
            ('blog_posts', self._load_blog_posts),
            ('news_articles', self._load_news),
        ]
        
        # Humanized text sources - AI text processed through humanizers
        humanized_sources = [
            ('humanized_ai', self._load_humanized_ai),
            ('paraphrased_ai', self._load_paraphrased_ai),
            ('mixed_authorship', self._load_mixed_authorship),
        ]
        
        samples_per_source = max_samples // max(len(human_sources), len(humanized_sources))
        
        print(f"\nTarget: {max_samples} samples per class")
        print(f"Loading from {len(human_sources)} human sources, {len(humanized_sources)} humanized sources")
        
        # Load human samples
        print("\n--- Loading Human Samples ---")
        for source_name, loader_fn in human_sources:
            try:
                samples = loader_fn(samples_per_source)
                self.human_samples.extend(samples)
                print(f"  ✓ {source_name}: {len(samples)} samples")
            except Exception as e:
                print(f"  ✗ {source_name}: Failed - {str(e)[:50]}")
        
        # Load humanized samples
        print("\n--- Loading Humanized Samples ---")
        for source_name, loader_fn in humanized_sources:
            try:
                samples = loader_fn(samples_per_source)
                self.humanized_samples.extend(samples)
                print(f"  ✓ {source_name}: {len(samples)} samples")
            except Exception as e:
                print(f"  ✗ {source_name}: Failed - {str(e)[:50]}")
        
        # If we don't have enough humanized samples, generate synthetic ones
        if len(self.humanized_samples) < max_samples // 2:
            print("\n--- Generating Synthetic Humanized Samples ---")
            additional = self._generate_synthetic_humanized(
                max_samples // 2 - len(self.humanized_samples)
            )
            self.humanized_samples.extend(additional)
            print(f"  ✓ Generated {len(additional)} synthetic humanized samples")
        
        # Balance datasets
        min_samples = min(len(self.human_samples), len(self.humanized_samples))
        self.human_samples = random.sample(self.human_samples, min_samples)
        self.humanized_samples = random.sample(self.humanized_samples, min_samples)
        
        print(f"\n✓ Final dataset: {len(self.human_samples)} human, {len(self.humanized_samples)} humanized")
        
        return self.human_samples, self.humanized_samples
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        return text.strip()
    
    def _filter_text(self, text: str, min_words: int = 50, max_words: int = 1000) -> bool:
        """Filter text by length and quality."""
        if not text:
            return False
        words = len(text.split())
        return min_words <= words <= max_words
    
    def _load_reddit_casual(self, n: int) -> List[str]:
        """Load casual Reddit comments (authentic human)."""
        samples = []
        try:
            # Try to load Reddit-style data
            ds = load_dataset("reddit", split="train", streaming=True, trust_remote_code=True)
            for i, row in enumerate(ds):
                if i >= n * 3:  # Get more, filter later
                    break
                text = self._clean_text(row.get('body', row.get('text', '')))
                if self._filter_text(text, min_words=50, max_words=500):
                    samples.append(text)
                    if len(samples) >= n:
                        break
        except:
            # Fallback: use other casual text sources
            try:
                ds = load_dataset("stanfordnlp/imdb", split="train")
                for row in ds.shuffle().select(range(min(n * 2, len(ds)))):
                    text = self._clean_text(row.get('text', ''))
                    if self._filter_text(text, min_words=50, max_words=500):
                        samples.append(text)
                        if len(samples) >= n:
                            break
            except:
                pass
        return samples[:n]
    
    def _load_essays(self, n: int) -> List[str]:
        """Load essay-style human writing."""
        samples = []
        try:
            # Essays dataset
            ds = load_dataset("qwedsacf/ivypanda-essays", split="train", trust_remote_code=True)
            for row in ds.shuffle().select(range(min(n * 2, len(ds)))):
                text = self._clean_text(row.get('TEXT', row.get('text', '')))
                if self._filter_text(text, min_words=100, max_words=800):
                    samples.append(text)
                    if len(samples) >= n:
                        break
        except:
            try:
                # Fallback to another essay source
                ds = load_dataset("ChristophSchuhmann/essays-with-instructions", split="train", trust_remote_code=True)
                for row in ds.shuffle().select(range(min(n * 2, len(ds)))):
                    text = self._clean_text(row.get('essay', row.get('text', '')))
                    if self._filter_text(text):
                        samples.append(text)
                        if len(samples) >= n:
                            break
            except:
                pass
        return samples[:n]
    
    def _load_amazon_reviews(self, n: int) -> List[str]:
        """Load Amazon reviews (authentic human opinions)."""
        samples = []
        try:
            ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", 
                            split="full", streaming=True, trust_remote_code=True)
            for i, row in enumerate(ds):
                if i >= n * 5:
                    break
                text = self._clean_text(row.get('text', ''))
                if self._filter_text(text, min_words=50, max_words=400):
                    samples.append(text)
                    if len(samples) >= n:
                        break
        except:
            try:
                ds = load_dataset("amazon_reviews_multi", "en", split="train", trust_remote_code=True)
                for row in ds.shuffle().select(range(min(n * 3, len(ds)))):
                    text = self._clean_text(row.get('review_body', ''))
                    if self._filter_text(text, min_words=50, max_words=400):
                        samples.append(text)
                        if len(samples) >= n:
                            break
            except:
                pass
        return samples[:n]
    
    def _load_blog_posts(self, n: int) -> List[str]:
        """Load blog post content."""
        samples = []
        try:
            ds = load_dataset("blog_authorship_corpus", split="train", trust_remote_code=True)
            for row in ds.shuffle().select(range(min(n * 2, len(ds)))):
                text = self._clean_text(row.get('text', ''))
                if self._filter_text(text, min_words=80, max_words=600):
                    samples.append(text)
                    if len(samples) >= n:
                        break
        except:
            pass
        return samples[:n]
    
    def _load_news(self, n: int) -> List[str]:
        """Load news articles."""
        samples = []
        try:
            ds = load_dataset("cc_news", split="train", streaming=True, trust_remote_code=True)
            for i, row in enumerate(ds):
                if i >= n * 3:
                    break
                text = self._clean_text(row.get('text', ''))
                if self._filter_text(text, min_words=100, max_words=800):
                    samples.append(text)
                    if len(samples) >= n:
                        break
        except:
            pass
        return samples[:n]
    
    def _load_humanized_ai(self, n: int) -> List[str]:
        """Load known humanized AI content."""
        samples = []
        
        # Try several humanized/AI detection datasets
        datasets_to_try = [
            ("Hello-SimpleAI/HC3", "all", "chatgpt_answers"),  # ChatGPT responses
            ("aadityaubhat/GPT-wiki-intro", None, "generated_intro"),  # GPT wiki intros
            ("HuggingFaceH4/no_robots", None, "messages"),  # Instruction-following responses
        ]
        
        for ds_name, config, text_field in datasets_to_try:
            try:
                if config:
                    ds = load_dataset(ds_name, config, split="train", trust_remote_code=True)
                else:
                    ds = load_dataset(ds_name, split="train", trust_remote_code=True)
                
                for row in ds.shuffle().select(range(min(n, len(ds)))):
                    if text_field == "messages":
                        # Handle message format
                        messages = row.get('messages', [])
                        for msg in messages:
                            if msg.get('role') == 'assistant':
                                text = self._clean_text(msg.get('content', ''))
                                if self._filter_text(text):
                                    # Apply synthetic humanization
                                    humanized = self._apply_humanization(text)
                                    samples.append(humanized)
                    else:
                        content = row.get(text_field, [])
                        if isinstance(content, list):
                            for item in content:
                                text = self._clean_text(item if isinstance(item, str) else str(item))
                                if self._filter_text(text):
                                    humanized = self._apply_humanization(text)
                                    samples.append(humanized)
                        else:
                            text = self._clean_text(str(content))
                            if self._filter_text(text):
                                humanized = self._apply_humanization(text)
                                samples.append(humanized)
                    
                    if len(samples) >= n:
                        break
                        
            except Exception as e:
                continue
        
        return samples[:n]
    
    def _load_paraphrased_ai(self, n: int) -> List[str]:
        """Load or generate paraphrased AI content."""
        samples = []
        
        try:
            # Try to find paraphrased datasets
            ds = load_dataset("google/paws", "labeled_final", split="train", trust_remote_code=True)
            for row in ds.shuffle().select(range(min(n * 2, len(ds)))):
                text = self._clean_text(row.get('sentence1', '') + ' ' + row.get('sentence2', ''))
                if self._filter_text(text, min_words=20, max_words=200):
                    samples.append(text)
        except:
            pass
        
        return samples[:n]
    
    def _load_mixed_authorship(self, n: int) -> List[str]:
        """Generate mixed human/AI content (simulating partial AI assistance)."""
        samples = []
        
        # We'll generate these synthetically by mixing human samples
        # with AI-like insertions
        try:
            # Get some human samples first
            human_ds = load_dataset("stanfordnlp/imdb", split="train")
            for row in human_ds.shuffle().select(range(min(n * 2, len(human_ds)))):
                text = self._clean_text(row.get('text', ''))
                if self._filter_text(text, min_words=100, max_words=500):
                    # Inject AI-like patterns into human text
                    mixed = self._inject_ai_patterns(text)
                    samples.append(mixed)
                    if len(samples) >= n:
                        break
        except:
            pass
        
        return samples[:n]
    
    def _apply_humanization(self, text: str) -> str:
        """Apply synthetic humanization techniques to AI text."""
        # Simulate humanizer tool effects
        
        # 1. Add some contractions
        replacements = [
            ('it is', "it's"), ('do not', "don't"), ('cannot', "can't"),
            ('will not', "won't"), ('they are', "they're"), ('we are', "we're"),
            ('is not', "isn't"), ('are not', "aren't"), ('would not', "wouldn't"),
        ]
        for old, new in replacements:
            if random.random() < 0.6:
                text = text.replace(old, new, 1)
        
        # 2. Add some casual filler words occasionally
        fillers = [' actually ', ' basically ', ' honestly ', ' really ']
        sentences = text.split('. ')
        for i in range(len(sentences)):
            if random.random() < 0.15 and len(sentences[i]) > 30:
                pos = random.randint(0, len(sentences[i]) // 2)
                sentences[i] = sentences[i][:pos] + random.choice(fillers) + sentences[i][pos:]
        text = '. '.join(sentences)
        
        # 3. Vary sentence lengths artificially
        sentences = text.split('. ')
        modified = []
        for s in sentences:
            if random.random() < 0.2 and len(s) > 50:
                # Split long sentence
                mid = len(s) // 2
                # Find a good split point near middle
                for split_word in [', and ', ', but ', '; ', ', ']:
                    idx = s.find(split_word, mid - 20, mid + 20)
                    if idx > 0:
                        modified.append(s[:idx])
                        modified.append(s[idx + len(split_word):])
                        break
                else:
                    modified.append(s)
            else:
                modified.append(s)
        text = '. '.join(modified)
        
        # 4. Synonym substitutions
        substitutions = [
            ('important', 'crucial'), ('significant', 'notable'),
            ('however', 'though'), ('therefore', 'so'),
            ('utilize', 'use'), ('demonstrate', 'show'),
        ]
        for old, new in substitutions:
            if random.random() < 0.4:
                text = re.sub(rf'\b{old}\b', new, text, count=1, flags=re.IGNORECASE)
        
        return text
    
    def _inject_ai_patterns(self, text: str) -> str:
        """Inject AI-like patterns into human text (simulating AI assistance)."""
        sentences = text.split('. ')
        
        # Insert some AI-like transitions
        transitions = ['Furthermore,', 'Additionally,', 'Moreover,', 'Consequently,']
        for i in range(1, len(sentences)):
            if random.random() < 0.15:
                sentences[i] = random.choice(transitions) + ' ' + sentences[i].lower()
        
        # Add some hedging
        hedges = ['it should be noted that', 'it is important to understand that', 'one could argue that']
        if len(sentences) > 3 and random.random() < 0.3:
            idx = random.randint(1, len(sentences) - 2)
            sentences[idx] = random.choice(hedges).capitalize() + ' ' + sentences[idx].lower()
        
        return '. '.join(sentences)
    
    def _generate_synthetic_humanized(self, n: int) -> List[str]:
        """Generate synthetic humanized samples when real ones are scarce."""
        samples = []
        
        # Use AI-generated text templates and humanize them
        ai_templates = [
            "The concept of {topic} is fundamentally important in modern society. It represents a significant shift in how we understand {related_topic}. Furthermore, the implications of this are far-reaching. Additionally, we must consider the various perspectives that exist on this matter. In conclusion, it is clear that this topic deserves careful consideration.",
            "When examining {topic}, several key factors emerge. First, there is the question of {factor1}. Second, we must consider {factor2}. Moreover, the relationship between these elements is complex. Therefore, a nuanced approach is necessary. Ultimately, this analysis reveals important insights.",
            "The evolution of {topic} has been remarkable. Initially, it was seen as {initial_view}. However, perspectives have shifted significantly. Today, most experts agree that {current_view}. This transformation reflects broader changes in our understanding.",
        ]
        
        topics = ['technology', 'education', 'healthcare', 'environment', 'economics', 'culture', 'science', 'politics']
        factors = ['cost', 'efficiency', 'accessibility', 'sustainability', 'innovation', 'tradition', 'equity']
        
        while len(samples) < n:
            template = random.choice(ai_templates)
            text = template.format(
                topic=random.choice(topics),
                related_topic=random.choice(topics),
                factor1=random.choice(factors),
                factor2=random.choice(factors),
                initial_view=f"a {random.choice(['minor', 'limited', 'specialized'])} concern",
                current_view=f"a {random.choice(['major', 'critical', 'essential'])} priority"
            )
            
            # Apply humanization
            humanized = self._apply_humanization(text)
            samples.append(humanized)
        
        return samples[:n]


class FlareModel:
    """
    Flare Detection Model - Logistic Regression with feature engineering
    for humanization detection.
    """
    
    def __init__(self):
        self.weights = {}
        self.bias = 0.0
        self.feature_names = []
        self.feature_stats = {}
        self.trained = False
        
    def train(self, human_samples: List[str], humanized_samples: List[str],
              epochs: int = 100, learning_rate: float = 0.01, batch_size: int = 32):
        """Train the model."""
        print("\n" + "=" * 60)
        print("FLARE Model Training")
        print("=" * 60)
        
        extractor = FlareFeatureExtractor()
        self.feature_names = extractor.get_feature_names()
        
        # Extract features
        print("\nExtracting features...")
        X_human = []
        X_humanized = []
        
        for i, text in enumerate(human_samples):
            if i % 1000 == 0:
                print(f"  Human samples: {i}/{len(human_samples)}")
            features = extractor.extract_features(text)
            X_human.append([features.get(f, 0.5) for f in self.feature_names])
        
        for i, text in enumerate(humanized_samples):
            if i % 1000 == 0:
                print(f"  Humanized samples: {i}/{len(humanized_samples)}")
            features = extractor.extract_features(text)
            X_humanized.append([features.get(f, 0.5) for f in self.feature_names])
        
        # Combine with labels (0 = human, 1 = humanized)
        X = np.array(X_human + X_humanized)
        y = np.array([0] * len(X_human) + [1] * len(X_humanized))
        
        # Normalize features
        print("\nNormalizing features...")
        self.feature_stats = {
            'mean': np.mean(X, axis=0),
            'std': np.std(X, axis=0) + 1e-8
        }
        X = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Split train/val
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)}")
        
        # Initialize weights
        self.weights = np.zeros(len(self.feature_names))
        self.bias = 0.0
        
        # Training loop
        best_acc = 0
        best_weights = None
        best_bias = None
        patience = 10
        no_improve = 0
        
        print("\nTraining...")
        for epoch in range(epochs):
            # Mini-batch gradient descent
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                
                # Forward pass
                z = np.dot(X_batch, self.weights) + self.bias
                predictions = self._sigmoid(z)
                
                # Backward pass
                error = predictions - y_batch
                gradient_w = np.dot(X_batch.T, error) / len(X_batch)
                gradient_b = np.mean(error)
                
                # L2 regularization
                gradient_w += 0.01 * self.weights
                
                # Update
                self.weights -= learning_rate * gradient_w
                self.bias -= learning_rate * gradient_b
            
            # Validation
            val_pred = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
            val_pred_binary = (val_pred >= 0.5).astype(int)
            val_acc = np.mean(val_pred_binary == y_val)
            
            # Track best
            if val_acc > best_acc:
                best_acc = val_acc
                best_weights = self.weights.copy()
                best_bias = self.bias
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 10 == 0:
                train_pred = self._sigmoid(np.dot(X_train, self.weights) + self.bias)
                train_acc = np.mean((train_pred >= 0.5).astype(int) == y_train)
                print(f"  Epoch {epoch}: Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")
            
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break
            
            # Reduce learning rate
            if epoch > 0 and epoch % 20 == 0:
                learning_rate *= 0.9
        
        # Use best weights
        self.weights = best_weights
        self.bias = best_bias
        self.trained = True
        
        # Final evaluation
        print("\n" + "-" * 40)
        print("Final Evaluation")
        print("-" * 40)
        
        val_pred = self._sigmoid(np.dot(X_val, self.weights) + self.bias)
        val_pred_binary = (val_pred >= 0.5).astype(int)
        
        # Metrics
        tp = np.sum((val_pred_binary == 1) & (y_val == 1))
        tn = np.sum((val_pred_binary == 0) & (y_val == 0))
        fp = np.sum((val_pred_binary == 1) & (y_val == 0))
        fn = np.sum((val_pred_binary == 0) & (y_val == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        # Feature importance
        print("\nTop 10 Most Important Features:")
        importance = sorted(enumerate(np.abs(self.weights)), key=lambda x: x[1], reverse=True)
        for i, (idx, imp) in enumerate(importance[:10]):
            direction = "→ Humanized" if self.weights[idx] > 0 else "→ Human"
            print(f"  {i + 1}. {self.feature_names[idx]}: {imp:.4f} {direction}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _sigmoid(self, z):
        """Sigmoid activation with numerical stability."""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict if text is humanized."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        extractor = FlareFeatureExtractor()
        features = extractor.extract_features(text)
        
        X = np.array([[features.get(f, 0.5) for f in self.feature_names]])
        X = (X - self.feature_stats['mean']) / self.feature_stats['std']
        
        prob = self._sigmoid(np.dot(X, self.weights) + self.bias)[0]
        
        return {
            'humanized_probability': float(prob),
            'is_humanized': prob >= 0.5,
            'confidence': float(abs(prob - 0.5) * 2),
            'features': features
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export model configuration for JavaScript."""
        return {
            'modelName': 'Flare',
            'version': '1.0',
            'type': 'humanization-detection',
            'description': 'Specialized model for detecting humanized AI content vs genuine human writing',
            'targetTask': 'humanized vs human classification',
            'features': self.feature_names,
            'weights': {f: float(w) for f, w in zip(self.feature_names, self.weights)},
            'bias': float(self.bias),
            'featureStats': {
                'mean': {f: float(m) for f, m in zip(self.feature_names, self.feature_stats['mean'])},
                'std': {f: float(s) for f, s in zip(self.feature_names, self.feature_stats['std'])}
            },
            'thresholds': {
                'humanized': 0.5,
                'highConfidence': 0.75,
                'uncertain': [0.35, 0.65]
            }
        }


def main():
    """Main training pipeline."""
    print("=" * 70)
    print("  FLARE DETECTION SYSTEM - Training Pipeline")
    print("  Humanized vs Human Classification")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load datasets
    loader = FlareDatasetLoader()
    human_samples, humanized_samples = loader.load_datasets(max_samples=30000)
    
    if len(human_samples) < 1000 or len(humanized_samples) < 1000:
        print("\n⚠ Warning: Limited training data available")
        print("  Model accuracy may be lower than expected")
    
    # Train model
    model = FlareModel()
    metrics = model.train(human_samples, humanized_samples, epochs=150, learning_rate=0.05)
    
    # Check if we hit target accuracy
    if metrics['accuracy'] < 0.95:
        print("\n⚠ Accuracy below 95%, attempting additional training...")
        # Try with more epochs and adjusted learning rate
        metrics = model.train(human_samples, humanized_samples, epochs=200, learning_rate=0.02)
    
    # Save model
    output_dir = os.path.join(os.path.dirname(__file__), 'models', 'Flare')
    os.makedirs(output_dir, exist_ok=True)
    
    # Export JavaScript config
    config = model.export_config()
    config['trainingStats'] = {
        'testAccuracy': metrics['accuracy'],
        'testPrecision': metrics['precision'],
        'testRecall': metrics['recall'],
        'testF1': metrics['f1'],
        'humanSamples': len(human_samples),
        'humanizedSamples': len(humanized_samples),
        'trainedAt': datetime.now().isoformat()
    }
    
    # Save as JavaScript config
    js_config_path = os.path.join(output_dir, 'veritas_config.js')
    with open(js_config_path, 'w') as f:
        f.write("// FLARE Detection System - Model Configuration\n")
        f.write("// Specialized for humanized vs human classification\n")
        f.write(f"// Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"// Accuracy: {metrics['accuracy'] * 100:.2f}%\n\n")
        f.write("const VERITAS_FLARE_CONFIG = ")
        f.write(json.dumps(config, indent=2))
        f.write(";\n")
    
    # Save metadata
    metadata = {
        'modelName': 'Flare',
        'version': '1.0',
        'type': 'humanization-detection',
        'created': datetime.now().isoformat(),
        'accuracy': metrics['accuracy'],
        'f1Score': metrics['f1'],
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'featureCount': len(model.feature_names),
        'trainingSamples': len(human_samples) + len(humanized_samples)
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Final Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"  F1 Score: {metrics['f1'] * 100:.2f}%")
    print(f"  Model saved to: {output_dir}")
    print(f"\n  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
