"""
Veritas Feature Extractor
Extracts statistical features from text for ML training.
Mirrors the JavaScript analyzer logic for consistency.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional
import numpy as np


class FeatureExtractor:
    """Extract statistical features from text for AI detection."""
    
    # Human baseline values (from variance-utils.js)
    HUMAN_BASELINES = {
        'sentenceLengthCV': {'mean': 0.55, 'std': 0.15, 'min': 0.25, 'max': 0.90},
        'vocabularyTTR': {'mean': 0.55, 'std': 0.12, 'min': 0.35, 'max': 0.80},
        'hapaxRatio': {'mean': 0.50, 'std': 0.12, 'min': 0.30, 'max': 0.70},
        'zipfSlope': {'mean': -1.0, 'std': 0.15, 'min': -1.3, 'max': -0.7},
        'burstiness': {'mean': 0.25, 'std': 0.15, 'min': -0.1, 'max': 0.5},
        'avgSentenceLength': {'mean': 18, 'std': 6, 'min': 8, 'max': 35},
        'readabilityGrade': {'mean': 10, 'std': 3, 'min': 5, 'max': 16},
    }
    
    def __init__(self):
        self.feature_names = [
            # Sentence-level features
            'sentence_count',
            'avg_sentence_length',
            'sentence_length_cv',
            'sentence_length_std',
            'sentence_length_min',
            'sentence_length_max',
            'sentence_length_range',
            'sentence_length_skewness',
            'sentence_length_kurtosis',
            
            # Vocabulary features
            'word_count',
            'unique_word_count',
            'type_token_ratio',
            'hapax_count',
            'hapax_ratio',
            'dis_legomena_ratio',  # words appearing exactly twice
            
            # Zipf's law features
            'zipf_slope',
            'zipf_r_squared',
            'zipf_residual_std',
            
            # Burstiness features
            'burstiness_sentence',
            'burstiness_word_length',
            
            # Readability features
            'avg_word_length',
            'word_length_cv',
            'syllable_ratio',
            'flesch_kincaid_grade',
            'automated_readability_index',
            
            # Repetition features
            'bigram_repetition_rate',
            'trigram_repetition_rate',
            'sentence_similarity_avg',
            
            # Punctuation features
            'comma_rate',
            'semicolon_rate',
            'question_rate',
            'exclamation_rate',
            
            # Structure features
            'paragraph_count',
            'avg_paragraph_length',
            'paragraph_length_cv',
            
            # Uniformity scores
            'overall_uniformity',
            'complexity_cv',
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r"[a-zA-Z'-]+", text.lower())
        return [w for w in words if len(w) > 0]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.',
                      r'\1<PERIOD>', text, flags=re.IGNORECASE)
        
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        
        # Filter empty sentences
        return [s for s in sentences if len(s.split()) >= 2]
    
    def split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def count_syllables(self, word: str) -> int:
        """Estimate syllable count for a word."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        # Count vowel groups
        vowels = 'aeiouy'
        count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel
        
        # Adjust for silent e
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        
        return max(1, count)
    
    def calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return float(np.std(values, ddof=1) / mean)
    
    def calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        return float(np.sum(((values - mean) / std) ** 3) * n / ((n-1) * (n-2)))
    
    def calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate excess kurtosis."""
        if len(values) < 4:
            return 0.0
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((values - mean) ** 4)
        return float(m4 / (std ** 4) - 3)
    
    def calculate_burstiness(self, inter_event_times: List[float]) -> float:
        """Calculate burstiness coefficient B = (σ - μ) / (σ + μ)."""
        if len(inter_event_times) < 2:
            return 0.0
        mean = np.mean(inter_event_times)
        std = np.std(inter_event_times, ddof=1)
        if mean + std == 0:
            return 0.0
        return float((std - mean) / (std + mean))
    
    def calculate_zipf(self, word_counts: Counter) -> Tuple[float, float, float]:
        """
        Calculate Zipf's law parameters.
        Returns: (slope, r_squared, residual_std)
        """
        if len(word_counts) < 5:
            return -1.0, 0.0, 0.0
        
        # Get frequencies sorted by rank
        frequencies = sorted(word_counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1)
        
        # Log-log regression
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        
        # Linear regression
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        
        # R-squared
        predicted = slope * log_ranks + intercept
        ss_res = np.sum((log_freqs - predicted) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Residual standard deviation
        residual_std = np.std(log_freqs - predicted)
        
        return float(slope), float(r_squared), float(residual_std)
    
    def calculate_ngram_repetition(self, words: List[str], n: int) -> float:
        """Calculate n-gram repetition rate."""
        if len(words) < n + 1:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts) if counts else 0.0
    
    def calculate_sentence_similarity(self, sentences: List[str]) -> float:
        """Calculate average Jaccard similarity between consecutive sentences."""
        if len(sentences) < 2:
            return 0.0
        
        similarities = []
        for i in range(len(sentences) - 1):
            words1 = set(self.tokenize(sentences[i]))
            words2 = set(self.tokenize(sentences[i + 1]))
            if words1 or words2:
                jaccard = len(words1 & words2) / len(words1 | words2)
                similarities.append(jaccard)
        
        return float(np.mean(similarities)) if similarities else 0.0
    
    def flesch_kincaid_grade(self, words: List[str], sentences: List[str]) -> float:
        """Calculate Flesch-Kincaid grade level."""
        if not words or not sentences:
            return 0.0
        
        total_syllables = sum(self.count_syllables(w) for w in words)
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        return 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    
    def automated_readability_index(self, text: str, words: List[str], 
                                     sentences: List[str]) -> float:
        """Calculate Automated Readability Index."""
        if not words or not sentences:
            return 0.0
        
        char_count = len(re.sub(r'\s', '', text))
        avg_word_length = char_count / len(words)
        avg_sentence_length = len(words) / len(sentences)
        
        return 4.71 * avg_word_length + 0.5 * avg_sentence_length - 21.43
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all features from text."""
        # Basic tokenization
        words = self.tokenize(text)
        sentences = self.split_sentences(text)
        paragraphs = self.split_paragraphs(text)
        
        if len(words) < 10 or len(sentences) < 2:
            # Return zeros for too-short text
            return {name: 0.0 for name in self.feature_names}
        
        # Word counts
        word_counts = Counter(words)
        hapax = [w for w, c in word_counts.items() if c == 1]
        dis_legomena = [w for w, c in word_counts.items() if c == 2]
        
        # Sentence lengths
        sentence_lengths = [len(self.tokenize(s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0]
        
        if not sentence_lengths:
            sentence_lengths = [1]
        
        sentence_lengths_arr = np.array(sentence_lengths, dtype=float)
        
        # Word lengths
        word_lengths = [len(w) for w in words]
        word_lengths_arr = np.array(word_lengths, dtype=float)
        
        # Paragraph lengths
        para_word_counts = [len(self.tokenize(p)) for p in paragraphs]
        para_word_counts = [c for c in para_word_counts if c > 0] or [1]
        
        # Zipf analysis
        zipf_slope, zipf_r2, zipf_residual = self.calculate_zipf(word_counts)
        
        # Burstiness
        burstiness_sentence = self.calculate_burstiness(sentence_lengths)
        burstiness_word = self.calculate_burstiness(word_lengths[:100])  # Sample
        
        # Punctuation counts
        comma_count = text.count(',')
        semicolon_count = text.count(';')
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Calculate all features
        features = {
            # Sentence-level features
            'sentence_count': len(sentences),
            'avg_sentence_length': float(np.mean(sentence_lengths_arr)),
            'sentence_length_cv': self.calculate_cv(sentence_lengths),
            'sentence_length_std': float(np.std(sentence_lengths_arr, ddof=1)),
            'sentence_length_min': float(np.min(sentence_lengths_arr)),
            'sentence_length_max': float(np.max(sentence_lengths_arr)),
            'sentence_length_range': float(np.max(sentence_lengths_arr) - np.min(sentence_lengths_arr)),
            'sentence_length_skewness': self.calculate_skewness(sentence_lengths_arr),
            'sentence_length_kurtosis': self.calculate_kurtosis(sentence_lengths_arr),
            
            # Vocabulary features
            'word_count': len(words),
            'unique_word_count': len(word_counts),
            'type_token_ratio': len(word_counts) / len(words) if words else 0,
            'hapax_count': len(hapax),
            'hapax_ratio': len(hapax) / len(word_counts) if word_counts else 0,
            'dis_legomena_ratio': len(dis_legomena) / len(word_counts) if word_counts else 0,
            
            # Zipf's law features
            'zipf_slope': zipf_slope,
            'zipf_r_squared': zipf_r2,
            'zipf_residual_std': zipf_residual,
            
            # Burstiness features
            'burstiness_sentence': burstiness_sentence,
            'burstiness_word_length': burstiness_word,
            
            # Readability features
            'avg_word_length': float(np.mean(word_lengths_arr)),
            'word_length_cv': self.calculate_cv(word_lengths),
            'syllable_ratio': sum(self.count_syllables(w) for w in words) / len(words) if words else 0,
            'flesch_kincaid_grade': self.flesch_kincaid_grade(words, sentences),
            'automated_readability_index': self.automated_readability_index(text, words, sentences),
            
            # Repetition features
            'bigram_repetition_rate': self.calculate_ngram_repetition(words, 2),
            'trigram_repetition_rate': self.calculate_ngram_repetition(words, 3),
            'sentence_similarity_avg': self.calculate_sentence_similarity(sentences),
            
            # Punctuation features
            'comma_rate': comma_count / len(sentences) if sentences else 0,
            'semicolon_rate': semicolon_count / len(sentences) if sentences else 0,
            'question_rate': question_count / len(sentences) if sentences else 0,
            'exclamation_rate': exclamation_count / len(sentences) if sentences else 0,
            
            # Structure features
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': float(np.mean(para_word_counts)),
            'paragraph_length_cv': self.calculate_cv(para_word_counts),
            
            # Uniformity scores
            'overall_uniformity': 1.0 - self.calculate_cv(sentence_lengths),
            'complexity_cv': self.calculate_cv(sentence_lengths),
        }
        
        return features
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """Extract features as numpy array."""
        features = self.extract_features(text)
        return np.array([features[name] for name in self.feature_names])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


if __name__ == '__main__':
    # Test the extractor
    extractor = FeatureExtractor()
    
    test_text = """
    Artificial intelligence has revolutionized the way we interact with technology.
    Machine learning algorithms have become increasingly sophisticated.
    These advancements have led to significant improvements in various industries.
    Natural language processing has made it possible for machines to understand text.
    Computer vision has enabled machines to interpret visual information.
    """
    
    features = extractor.extract_features(test_text)
    print("Extracted features:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
