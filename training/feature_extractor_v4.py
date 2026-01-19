"""
Veritas Feature Extractor V4 - Comprehensive Edition
=====================================================
Extracts ALL possible statistical features from text for ML training.
Mirrors and extends the JavaScript analyzer logic for maximum detection accuracy.

Key Changes from V3:
- 80+ features (up from 37)
- Metadata features EXCLUDED (handled as absolute flag, not ML-weighted)
- Added: readability indices, POS patterns, AI signatures, humanizer signals
- Added: advanced statistical tests, autocorrelation, perplexity approximation
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


class FeatureExtractorV4:
    """
    Comprehensive feature extraction for AI detection.
    Excludes Metadata features (Category 11) - those are absolute flags.
    """
    
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
    
    # Common function words for analysis
    FUNCTION_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
        'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'to',
        'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
        'again', 'further', 'then', 'once', 'here', 'there', 'all', 'each',
        'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
        'only', 'own', 'same', 'so', 'than', 'too', 'very', 'can', 'will',
        'just', 'should', 'now', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
        'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'would',
        'could', 'ought', 'might', 'must', 'shall', 'may', 'need', 'dare',
        'as', 'until', 'while', 'of', 'because', 'although', 'though'
    }
    
    # AI-typical hedging phrases
    AI_HEDGING_PHRASES = [
        'it is important to note', 'it should be noted', 'it is worth mentioning',
        'it is essential to', 'it is crucial to', 'it is vital to',
        'in conclusion', 'to summarize', 'in summary', 'to conclude',
        'furthermore', 'moreover', 'additionally', 'in addition',
        'however', 'nevertheless', 'nonetheless', 'on the other hand',
        'firstly', 'secondly', 'thirdly', 'finally', 'lastly',
        'as a result', 'consequently', 'therefore', 'thus', 'hence',
        'for instance', 'for example', 'such as', 'including',
        'in order to', 'so as to', 'with the aim of',
        'it can be argued', 'one might argue', 'some would say',
        'generally speaking', 'broadly speaking', 'in general',
        'on balance', 'all things considered', 'taking everything into account'
    ]
    
    # Discourse markers
    DISCOURSE_MARKERS = [
        'well', 'so', 'now', 'then', 'anyway', 'actually', 'basically',
        'essentially', 'literally', 'obviously', 'clearly', 'certainly',
        'definitely', 'absolutely', 'exactly', 'indeed', 'of course',
        'naturally', 'apparently', 'presumably', 'supposedly', 'arguably',
        'admittedly', 'frankly', 'honestly', 'personally', 'importantly'
    ]
    
    def __init__(self):
        self.feature_names = self._build_feature_names()
    
    def _build_feature_names(self) -> List[str]:
        """Build comprehensive list of all feature names."""
        return [
            # ===== SENTENCE-LEVEL FEATURES (12) =====
            'sentence_count',
            'avg_sentence_length',
            'sentence_length_cv',
            'sentence_length_std',
            'sentence_length_min',
            'sentence_length_max',
            'sentence_length_range',
            'sentence_length_skewness',
            'sentence_length_kurtosis',
            'sentence_length_gini',
            'sentence_length_entropy',
            'sentence_length_iqr',
            
            # ===== VOCABULARY FEATURES (15) =====
            'word_count',
            'unique_word_count',
            'type_token_ratio',
            'log_ttr',
            'msttr_50',  # Moving-window TTR
            'hapax_count',
            'hapax_ratio',
            'dis_legomena_count',
            'dis_legomena_ratio',
            'yules_k',
            'simpsons_d',
            'honores_r',
            'brunets_w',
            'sichels_s',
            'vocabulary_entropy',
            
            # ===== ZIPF'S LAW FEATURES (4) =====
            'zipf_slope',
            'zipf_r_squared',
            'zipf_residual_std',
            'zipf_compliance',
            
            # ===== BURSTINESS & UNIFORMITY (6) =====
            'burstiness_sentence',
            'burstiness_word_length',
            'overall_uniformity',
            'variance_of_variance',
            'local_variance_cv',
            'autocorrelation_avg',
            
            # ===== READABILITY FEATURES (10) =====
            'avg_word_length',
            'word_length_cv',
            'syllable_ratio',
            'flesch_reading_ease',
            'flesch_kincaid_grade',
            'gunning_fog_index',
            'coleman_liau_index',
            'smog_index',
            'ari_index',
            'complex_word_percentage',
            
            # ===== N-GRAM REPETITION (8) =====
            'bigram_repetition_rate',
            'trigram_repetition_rate',
            'quadgram_repetition_rate',
            'pentagram_repetition_rate',
            'repeated_phrase_count',
            'repeated_phrase_score',
            'high_order_repetition',
            'sentence_similarity_avg',
            
            # ===== PUNCTUATION FEATURES (6) =====
            'comma_rate',
            'semicolon_rate',
            'colon_rate',
            'question_rate',
            'exclamation_rate',
            'punctuation_diversity',
            
            # ===== STRUCTURE FEATURES (4) =====
            # NOTE: paragraph_* features EXCLUDED (Metadata category)
            'avg_words_per_paragraph',
            'paragraph_count_normalized',
            'list_marker_count',
            'heading_marker_count',
            
            # ===== FUNCTION WORD FEATURES (4) =====
            'function_word_ratio',
            'content_word_ratio',
            'function_word_diversity',
            'pronoun_density',
            
            # ===== AI SIGNATURE FEATURES (8) =====
            'hedging_phrase_density',
            'discourse_marker_density',
            'contraction_rate',
            'contraction_uniformity',
            'sentence_starter_variety',
            'transition_word_density',
            'passive_voice_estimate',
            'nominalization_ratio',
            
            # ===== HUMANIZER DETECTION (6) =====
            'variance_stability',
            'autocorrelation_flatness',
            'feature_correlation_strength',
            'sophistication_variance',
            'artificial_variance_score',
            'humanizer_probability',
            
            # ===== ADVANCED STATISTICAL (8) =====
            'runs_test_score',
            'chi_squared_uniformity',
            'perplexity_estimate',
            'predictability_score',
            'periodicity_score',
            'mahalanobis_distance',
            'variance_naturalness',
            'extreme_variance_indicator',
            
            # ===== WORD PATTERN FEATURES (5) =====
            'verb_adverb_ratio',
            'adjective_density',
            'preposition_density',
            'determiner_ratio',
            'modal_verb_density',
            
            # Total: 96 features (excluding Metadata/paragraph structure from ML)
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r"[a-zA-Z'-]+", text.lower())
        return [w for w in words if len(w) > 0]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        abbrevs = r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd|i\.e|e\.g|cf|viz)'
        text = re.sub(rf'{abbrevs}\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        
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
        word = word.lower().strip()
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
        
        # Adjust for silent e
        if word.endswith('e') and count > 1:
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            count += 1
        if word.endswith('es') or word.endswith('ed'):
            if len(word) > 3 and word[-3] not in vowels:
                count -= 1
        
        return max(1, count)
    
    # ===== STATISTICAL HELPERS =====
    
    def calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if len(values) < 2:
            return 0.0
        arr = np.array(values)
        mean = np.mean(arr)
        if mean == 0:
            return 0.0
        return float(np.std(arr, ddof=1) / mean)
    
    def calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        arr = np.array(values)
        n = len(arr)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std == 0:
            return 0.0
        return float(np.sum(((arr - mean) / std) ** 3) * n / ((n-1) * (n-2)))
    
    def calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate excess kurtosis."""
        if len(values) < 4:
            return 0.0
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((arr - mean) ** 4)
        return float(m4 / (std ** 4) - 3)
    
    def calculate_gini(self, values: List[float]) -> float:
        """Calculate Gini coefficient."""
        if len(values) < 2:
            return 0.0
        arr = np.array(sorted(values))
        n = len(arr)
        cumsum = np.cumsum(arr)
        return float((n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n) if cumsum[-1] > 0 else 0.0
    
    def calculate_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy."""
        if len(values) < 2:
            return 0.0
        arr = np.array(values)
        total = np.sum(arr)
        if total == 0:
            return 0.0
        probs = arr / total
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))
    
    def calculate_iqr(self, values: List[float]) -> float:
        """Calculate interquartile range."""
        if len(values) < 4:
            return 0.0
        arr = np.array(values)
        q75, q25 = np.percentile(arr, [75, 25])
        return float(q75 - q25)
    
    def calculate_burstiness(self, inter_event_times: List[float]) -> float:
        """Calculate burstiness coefficient B = (σ - μ) / (σ + μ)."""
        if len(inter_event_times) < 2:
            return 0.0
        arr = np.array(inter_event_times)
        mean = np.mean(arr)
        std = np.std(arr, ddof=1)
        if mean + std == 0:
            return 0.0
        return float((std - mean) / (std + mean))
    
    # ===== VOCABULARY METRICS =====
    
    def calculate_yules_k(self, word_counts: Counter) -> float:
        """Calculate Yule's K (vocabulary richness)."""
        if len(word_counts) < 2:
            return 0.0
        N = sum(word_counts.values())
        M1 = len(word_counts)
        M2 = sum(f * f for f in word_counts.values())
        if M1 == 0:
            return 0.0
        K = 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0
        return float(K)
    
    def calculate_simpsons_d(self, word_counts: Counter) -> float:
        """Calculate Simpson's D (probability two random words are same)."""
        N = sum(word_counts.values())
        if N < 2:
            return 0.0
        return float(sum(f * (f - 1) for f in word_counts.values()) / (N * (N - 1)))
    
    def calculate_honores_r(self, word_counts: Counter) -> float:
        """Calculate Honoré's R statistic."""
        N = sum(word_counts.values())
        V = len(word_counts)
        V1 = sum(1 for f in word_counts.values() if f == 1)
        if V == 0 or V1 == V:
            return 0.0
        return float(100 * np.log(N) / (1 - V1/V)) if V1 < V else 0.0
    
    def calculate_brunets_w(self, word_counts: Counter) -> float:
        """Calculate Brunet's W."""
        N = sum(word_counts.values())
        V = len(word_counts)
        if N < 2 or V < 2:
            return 0.0
        return float(N ** (V ** -0.172))
    
    def calculate_sichels_s(self, word_counts: Counter) -> float:
        """Calculate Sichel's S (proportion of dis legomena)."""
        V = len(word_counts)
        if V == 0:
            return 0.0
        V2 = sum(1 for f in word_counts.values() if f == 2)
        return float(V2 / V)
    
    def calculate_log_ttr(self, tokens: List[str]) -> float:
        """Calculate log TTR."""
        if len(tokens) < 2:
            return 0.0
        return float(np.log(len(set(tokens))) / np.log(len(tokens)))
    
    def calculate_msttr(self, tokens: List[str], window: int = 50) -> float:
        """Calculate mean segmental TTR."""
        if len(tokens) < window:
            return len(set(tokens)) / len(tokens) if tokens else 0.0
        
        ttrs = []
        for i in range(0, len(tokens) - window + 1, window):
            segment = tokens[i:i+window]
            ttrs.append(len(set(segment)) / len(segment))
        
        return float(np.mean(ttrs)) if ttrs else 0.0
    
    # ===== ZIPF'S LAW =====
    
    def calculate_zipf(self, word_counts: Counter) -> Tuple[float, float, float, float]:
        """Calculate Zipf's law parameters. Returns (slope, r_squared, residual_std, compliance)."""
        if len(word_counts) < 5:
            return -1.0, 0.0, 0.0, 0.0
        
        frequencies = sorted(word_counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1)
        
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        
        # Linear regression
        slope, intercept = np.polyfit(log_ranks, log_freqs, 1)
        
        # R-squared
        predicted = slope * log_ranks + intercept
        ss_res = np.sum((log_freqs - predicted) ** 2)
        ss_tot = np.sum((log_freqs - np.mean(log_freqs)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Residual std
        residual_std = float(np.std(log_freqs - predicted))
        
        # Compliance score (how close to ideal -1.0 slope)
        ideal_slope = -1.0
        compliance = max(0, 1 - abs(slope - ideal_slope) / 0.5) * r_squared
        
        return float(slope), float(r_squared), residual_std, float(compliance)
    
    # ===== N-GRAM ANALYSIS =====
    
    def calculate_ngram_repetition(self, words: List[str], n: int) -> float:
        """Calculate n-gram repetition rate."""
        if len(words) < n + 1:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return float(repeated / len(counts)) if counts else 0.0
    
    def find_repeated_phrases(self, words: List[str]) -> Dict:
        """Find repeated phrases (4+ grams appearing more than once)."""
        if len(words) < 8:
            return {'count': 0, 'phrases': [], 'score': 0.0}
        
        repeated = []
        for n in [4, 5, 6, 7]:
            if len(words) < n + 1:
                continue
            ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
            counts = Counter(ngrams)
            for phrase, count in counts.items():
                if count > 1:
                    repeated.append({'phrase': phrase, 'count': count, 'length': n})
        
        # Score based on length and frequency
        score = sum(p['count'] * p['length'] for p in repeated) / max(1, len(words)) * 10
        
        return {
            'count': len(repeated),
            'phrases': sorted(repeated, key=lambda x: -x['count'] * x['length'])[:10],
            'score': min(1.0, score)
        }
    
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
    
    # ===== READABILITY INDICES =====
    
    def calculate_readability(self, text: str, words: List[str], sentences: List[str]) -> Dict:
        """Calculate all readability indices."""
        if not words or not sentences:
            return {k: 0.0 for k in ['flesch_re', 'flesch_kg', 'gunning_fog', 
                                      'coleman_liau', 'smog', 'ari', 'complex_pct']}
        
        total_syllables = sum(self.count_syllables(w) for w in words)
        avg_syllables = total_syllables / len(words)
        avg_sentence_len = len(words) / len(sentences)
        
        # Complex words (3+ syllables)
        complex_words = [w for w in words if self.count_syllables(w) >= 3]
        complex_pct = len(complex_words) / len(words) * 100
        
        # Flesch Reading Ease
        flesch_re = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
        
        # Flesch-Kincaid Grade
        flesch_kg = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
        
        # Gunning Fog
        gunning_fog = 0.4 * (avg_sentence_len + complex_pct)
        
        # Coleman-Liau
        char_count = sum(len(w) for w in words)
        L = (char_count / len(words)) * 100
        S = (len(sentences) / len(words)) * 100
        coleman_liau = 0.0588 * L - 0.296 * S - 15.8
        
        # SMOG
        if len(sentences) >= 3:
            polysyllables = len([w for w in words if self.count_syllables(w) >= 3])
            smog = 1.0430 * np.sqrt(polysyllables * (30 / len(sentences))) + 3.1291
        else:
            smog = 0.0
        
        # ARI
        char_count_alpha = len(re.sub(r'[^a-zA-Z0-9]', '', text))
        ari = 4.71 * (char_count_alpha / len(words)) + 0.5 * avg_sentence_len - 21.43
        
        return {
            'flesch_re': max(0, min(100, flesch_re)),
            'flesch_kg': max(0, flesch_kg),
            'gunning_fog': gunning_fog,
            'coleman_liau': max(0, coleman_liau),
            'smog': max(0, smog),
            'ari': max(0, ari),
            'complex_pct': complex_pct
        }
    
    # ===== AI SIGNATURE DETECTION =====
    
    def detect_ai_signatures(self, text: str, words: List[str], sentences: List[str]) -> Dict:
        """Detect AI-typical patterns."""
        text_lower = text.lower()
        
        # Hedging phrase density
        hedging_count = sum(1 for phrase in self.AI_HEDGING_PHRASES if phrase in text_lower)
        hedging_density = hedging_count / max(1, len(sentences))
        
        # Discourse marker density
        discourse_count = sum(1 for marker in self.DISCOURSE_MARKERS 
                             if re.search(rf'\b{marker}\b', text_lower))
        discourse_density = discourse_count / max(1, len(sentences))
        
        # Contraction analysis
        contraction_pattern = r"\b(don't|won't|can't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|I'm|you're|we're|they're|he's|she's|it's|that's|there's|here's|what's|who's|let's)\b"
        contractions = re.findall(contraction_pattern, text, re.IGNORECASE)
        contraction_rate = len(contractions) / max(1, len(sentences))
        
        # Contraction uniformity
        if len(contractions) >= 3:
            positions = [m.start() / len(text) for m in re.finditer(contraction_pattern, text, re.IGNORECASE)]
            spacing = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
            contraction_uniformity = 1 - self.calculate_cv(spacing) if len(spacing) > 1 else 0.5
        else:
            contraction_uniformity = 0.5
        
        # Sentence starter variety
        starters = [s.split()[0].lower() if s.split() else '' for s in sentences]
        starter_variety = len(set(starters)) / max(1, len(starters))
        
        # Transition words
        transitions = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                      'nevertheless', 'nonetheless', 'thus', 'hence', 'accordingly']
        transition_count = sum(1 for t in transitions if re.search(rf'\b{t}\b', text_lower))
        transition_density = transition_count / max(1, len(sentences))
        
        # Passive voice estimate (simple heuristic)
        passive_patterns = r'\b(is|are|was|were|been|being)\s+\w+ed\b'
        passive_count = len(re.findall(passive_patterns, text_lower))
        passive_estimate = passive_count / max(1, len(sentences))
        
        # Nominalization ratio (words ending in -tion, -ment, -ness, -ity)
        nominalizations = len(re.findall(r'\b\w+(tion|ment|ness|ity)\b', text_lower))
        nominalization_ratio = nominalizations / max(1, len(words))
        
        return {
            'hedging_density': hedging_density,
            'discourse_density': discourse_density,
            'contraction_rate': contraction_rate,
            'contraction_uniformity': contraction_uniformity,
            'starter_variety': starter_variety,
            'transition_density': transition_density,
            'passive_estimate': passive_estimate,
            'nominalization_ratio': nominalization_ratio
        }
    
    # ===== HUMANIZER DETECTION =====
    
    def detect_humanizer_signals(self, text: str, words: List[str], 
                                  sentence_lengths: List[int]) -> Dict:
        """Detect signals of humanized AI text."""
        if len(sentence_lengths) < 5:
            return {
                'variance_stability': 0.5,
                'autocorrelation_flatness': 0.5,
                'correlation_strength': 0.5,
                'sophistication_variance': 0.5,
                'artificial_variance': 0.0,
                'humanizer_prob': 0.0
            }
        
        # Variance of variance (second-order variance)
        window_size = max(3, len(sentence_lengths) // 5)
        local_vars = []
        for i in range(0, len(sentence_lengths) - window_size + 1, window_size):
            window = sentence_lengths[i:i+window_size]
            if len(window) > 1:
                local_vars.append(np.var(window))
        
        variance_of_variance = self.calculate_cv(local_vars) if len(local_vars) > 1 else 0.5
        cv = self.calculate_cv(sentence_lengths)
        # Stable variance with high CV = suspicious
        variance_stability = 1.0 if (cv > 0.4 and variance_of_variance < 0.3) else 0.0
        
        # Autocorrelation flatness
        ac_coeffs = self._calculate_autocorrelation(sentence_lengths, 5)
        ac_variance = np.var(ac_coeffs) if len(ac_coeffs) > 1 else 0.5
        ac_mean = np.mean(np.abs(ac_coeffs)) if ac_coeffs else 0.5
        autocorrelation_flatness = 1.0 if (ac_variance < 0.01 and 0.05 < ac_mean < 0.25) else 0.0
        
        # Word sophistication variance
        word_ranks = [self._get_word_frequency_rank(w) for w in words[:500]]
        sophistication_variance = self.calculate_cv(word_ranks) if word_ranks else 0.5
        
        # Feature correlation (simplified)
        if len(words) > 50 and len(sentence_lengths) >= 5:
            chunks = 5
            chunk_size = len(words) // chunks
            local_ttrs = [len(set(words[i*chunk_size:(i+1)*chunk_size])) / chunk_size 
                         for i in range(chunks)]
            sent_chunks = [np.mean(sentence_lengths[i*len(sentence_lengths)//chunks:
                                                    (i+1)*len(sentence_lengths)//chunks]) 
                          for i in range(chunks)]
            correlation = abs(np.corrcoef(local_ttrs, sent_chunks)[0, 1]) if len(local_ttrs) == len(sent_chunks) else 0.5
            correlation_strength = correlation if not np.isnan(correlation) else 0.5
        else:
            correlation_strength = 0.5
        
        # Artificial variance score
        flags = [
            variance_stability > 0.5,
            autocorrelation_flatness > 0.5,
            correlation_strength < 0.2 and cv > 0.4,
            sophistication_variance > 1.2
        ]
        artificial_variance = sum(flags) / len(flags)
        humanizer_prob = min(1.0, sum(flags) / 2)
        
        return {
            'variance_stability': variance_stability,
            'autocorrelation_flatness': autocorrelation_flatness,
            'correlation_strength': correlation_strength,
            'sophistication_variance': sophistication_variance,
            'artificial_variance': artificial_variance,
            'humanizer_prob': humanizer_prob
        }
    
    def _calculate_autocorrelation(self, values: List[float], max_lag: int) -> List[float]:
        """Calculate autocorrelation coefficients."""
        if len(values) < max_lag + 2:
            return []
        
        arr = np.array(values)
        mean = np.mean(arr)
        var = np.var(arr)
        
        if var == 0:
            return [0.0] * max_lag
        
        coeffs = []
        for lag in range(1, max_lag + 1):
            if lag >= len(arr):
                break
            coeff = np.mean((arr[:-lag] - mean) * (arr[lag:] - mean)) / var
            coeffs.append(float(coeff))
        
        return coeffs
    
    def _get_word_frequency_rank(self, word: str) -> float:
        """Estimate word frequency rank (higher = rarer)."""
        # Simplified frequency estimation based on word length and common patterns
        word = word.lower()
        
        # Very common words
        if word in self.FUNCTION_WORDS:
            return 50
        
        # Length-based heuristic
        base_rank = len(word) * 100
        
        # Common endings suggest more common words
        if word.endswith(('ing', 'ed', 'ly', 'tion', 'ness')):
            base_rank *= 0.7
        
        return base_rank
    
    # ===== ADVANCED STATISTICAL TESTS =====
    
    def calculate_advanced_stats(self, sentence_lengths: List[int], words: List[str]) -> Dict:
        """Calculate advanced statistical features."""
        if len(sentence_lengths) < 5:
            return {
                'runs_test': 0.5,
                'chi_squared': 0.5,
                'perplexity': 0.5,
                'predictability': 0.5,
                'periodicity': 0.0,
                'mahalanobis': 0.0,
                'variance_naturalness': 0.5,
                'extreme_variance': 0.0
            }
        
        # Runs test for randomness
        median = np.median(sentence_lengths)
        binary = [1 if x > median else 0 for x in sentence_lengths]
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        n1 = sum(binary)
        n0 = len(binary) - n1
        if n0 > 0 and n1 > 0:
            expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
            runs_test = min(1.0, max(0.0, runs / expected_runs))
        else:
            runs_test = 0.5
        
        # Chi-squared uniformity (binned)
        bins = 5
        hist, _ = np.histogram(sentence_lengths, bins=bins)
        expected = len(sentence_lengths) / bins
        chi_sq = sum((h - expected) ** 2 / expected for h in hist) if expected > 0 else 0
        chi_squared = 1 / (1 + chi_sq / 10)  # Normalize
        
        # Perplexity estimate (based on bigram predictability)
        if len(words) > 10:
            bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            unigram_counts = Counter(words)
            
            log_probs = []
            for i in range(1, min(len(words), 200)):
                prev_word = words[i-1]
                curr_word = words[i]
                bigram = (prev_word, curr_word)
                
                p_bigram = bigram_counts.get(bigram, 0) / max(1, unigram_counts.get(prev_word, 1))
                if p_bigram > 0:
                    log_probs.append(-np.log2(p_bigram))
            
            avg_log_prob = np.mean(log_probs) if log_probs else 10
            perplexity = 2 ** avg_log_prob
            predictability = 1 / (1 + perplexity / 100)
        else:
            perplexity = 50
            predictability = 0.5
        
        # Periodicity (FFT-based)
        if len(sentence_lengths) >= 8:
            fft = np.abs(np.fft.fft(sentence_lengths - np.mean(sentence_lengths)))
            fft = fft[1:len(fft)//2]  # Remove DC and mirror
            if len(fft) > 0 and np.max(fft) > 0:
                periodicity = float(np.max(fft) / np.sum(fft))
            else:
                periodicity = 0.0
        else:
            periodicity = 0.0
        
        # Mahalanobis distance from human baseline
        cv = self.calculate_cv(sentence_lengths)
        features = np.array([cv, np.mean(sentence_lengths), np.std(sentence_lengths)])
        baseline = np.array([0.55, 18, 8])
        cov_inv = np.diag([1/0.15**2, 1/6**2, 1/4**2])  # Simplified
        diff = features - baseline
        mahalanobis = float(np.sqrt(diff @ cov_inv @ diff))
        
        # Variance naturalness (bell curve fit)
        cv = self.calculate_cv(sentence_lengths)
        baseline = self.HUMAN_BASELINES['sentenceLengthCV']
        z_score = abs(cv - baseline['mean']) / baseline['std']
        variance_naturalness = float(np.exp(-z_score**2 / 2))
        
        # Extreme variance indicator
        extreme_low = cv < 0.2
        extreme_high = cv > 0.9
        extreme_variance = 1.0 if (extreme_low or extreme_high) else 0.0
        
        return {
            'runs_test': runs_test,
            'chi_squared': chi_squared,
            'perplexity': min(1.0, perplexity / 100),
            'predictability': predictability,
            'periodicity': periodicity,
            'mahalanobis': min(5.0, mahalanobis),
            'variance_naturalness': variance_naturalness,
            'extreme_variance': extreme_variance
        }
    
    # ===== WORD PATTERN ANALYSIS =====
    
    def analyze_word_patterns(self, words: List[str]) -> Dict:
        """Analyze word patterns (simplified POS-like features)."""
        if len(words) < 10:
            return {
                'verb_adverb_ratio': 0.0,
                'adjective_density': 0.0,
                'preposition_density': 0.0,
                'determiner_ratio': 0.0,
                'modal_density': 0.0
            }
        
        # Simplified pattern detection
        adverb_endings = ('ly',)
        verb_endings = ('ed', 'ing', 'es', 's')
        adj_endings = ('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ic')
        
        adverbs = [w for w in words if w.endswith(adverb_endings) and len(w) > 4]
        verbs = [w for w in words if w.endswith(verb_endings) and len(w) > 3]
        adjectives = [w for w in words if w.endswith(adj_endings) and len(w) > 4]
        
        prepositions = {'in', 'on', 'at', 'by', 'for', 'with', 'about', 'against',
                       'between', 'into', 'through', 'during', 'before', 'after',
                       'above', 'below', 'to', 'from', 'up', 'down', 'out', 'over', 'under'}
        preps = [w for w in words if w.lower() in prepositions]
        
        determiners = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your',
                      'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every'}
        dets = [w for w in words if w.lower() in determiners]
        
        modals = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}
        modal_count = sum(1 for w in words if w.lower() in modals)
        
        n = len(words)
        return {
            'verb_adverb_ratio': len(adverbs) / max(1, len(verbs)),
            'adjective_density': len(adjectives) / n,
            'preposition_density': len(preps) / n,
            'determiner_ratio': len(dets) / n,
            'modal_density': modal_count / n
        }
    
    # ===== MAIN EXTRACTION METHOD =====
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all features from text."""
        # Basic tokenization
        words = self.tokenize(text)
        sentences = self.split_sentences(text)
        paragraphs = self.split_paragraphs(text)
        
        if len(words) < 10 or len(sentences) < 2:
            return {name: 0.0 for name in self.feature_names}
        
        # Word counts
        word_counts = Counter(words)
        hapax = [w for w, c in word_counts.items() if c == 1]
        dis_legomena = [w for w, c in word_counts.items() if c == 2]
        
        # Sentence lengths
        sentence_lengths = [len(self.tokenize(s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0] or [1]
        sent_arr = np.array(sentence_lengths, dtype=float)
        
        # Word lengths
        word_lengths = [len(w) for w in words]
        
        # Zipf analysis
        zipf_slope, zipf_r2, zipf_residual, zipf_compliance = self.calculate_zipf(word_counts)
        
        # Readability
        readability = self.calculate_readability(text, words, sentences)
        
        # N-gram analysis
        repeated_phrases = self.find_repeated_phrases(words)
        
        # AI signatures
        ai_sigs = self.detect_ai_signatures(text, words, sentences)
        
        # Humanizer signals
        humanizer = self.detect_humanizer_signals(text, words, sentence_lengths)
        
        # Advanced stats
        adv_stats = self.calculate_advanced_stats(sentence_lengths, words)
        
        # Word patterns
        word_patterns = self.analyze_word_patterns(words)
        
        # Function words
        func_words = [w for w in words if w.lower() in self.FUNCTION_WORDS]
        pronouns = [w for w in words if w.lower() in {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}]
        
        # Punctuation
        punct_counts = {
            'comma': text.count(','),
            'semicolon': text.count(';'),
            'colon': text.count(':'),
            'question': text.count('?'),
            'exclamation': text.count('!')
        }
        punct_diversity = sum(1 for c in punct_counts.values() if c > 0) / 5
        
        # Autocorrelation
        ac_coeffs = self._calculate_autocorrelation(sentence_lengths, 5)
        ac_avg = float(np.mean(np.abs(ac_coeffs))) if ac_coeffs else 0.0
        
        # Local variance
        window = max(3, len(sentence_lengths) // 5)
        local_vars = []
        for i in range(0, len(sentence_lengths) - window + 1, window):
            w = sentence_lengths[i:i+window]
            if len(w) > 1:
                local_vars.append(np.var(w))
        local_var_cv = self.calculate_cv(local_vars) if len(local_vars) > 1 else 0.0
        
        # Build feature dictionary
        features = {
            # Sentence features
            'sentence_count': float(len(sentences)),
            'avg_sentence_length': float(np.mean(sent_arr)),
            'sentence_length_cv': self.calculate_cv(sentence_lengths),
            'sentence_length_std': float(np.std(sent_arr, ddof=1)),
            'sentence_length_min': float(np.min(sent_arr)),
            'sentence_length_max': float(np.max(sent_arr)),
            'sentence_length_range': float(np.max(sent_arr) - np.min(sent_arr)),
            'sentence_length_skewness': self.calculate_skewness(sent_arr),
            'sentence_length_kurtosis': self.calculate_kurtosis(sent_arr),
            'sentence_length_gini': self.calculate_gini(sentence_lengths),
            'sentence_length_entropy': self.calculate_entropy(np.histogram(sent_arr, bins=10)[0]),
            'sentence_length_iqr': self.calculate_iqr(sentence_lengths),
            
            # Vocabulary features
            'word_count': float(len(words)),
            'unique_word_count': float(len(word_counts)),
            'type_token_ratio': len(word_counts) / len(words),
            'log_ttr': self.calculate_log_ttr(words),
            'msttr_50': self.calculate_msttr(words, 50),
            'hapax_count': float(len(hapax)),
            'hapax_ratio': len(hapax) / len(word_counts) if word_counts else 0,
            'dis_legomena_count': float(len(dis_legomena)),
            'dis_legomena_ratio': len(dis_legomena) / len(word_counts) if word_counts else 0,
            'yules_k': self.calculate_yules_k(word_counts),
            'simpsons_d': self.calculate_simpsons_d(word_counts),
            'honores_r': self.calculate_honores_r(word_counts),
            'brunets_w': self.calculate_brunets_w(word_counts),
            'sichels_s': self.calculate_sichels_s(word_counts),
            'vocabulary_entropy': self.calculate_entropy(list(word_counts.values())),
            
            # Zipf
            'zipf_slope': zipf_slope,
            'zipf_r_squared': zipf_r2,
            'zipf_residual_std': zipf_residual,
            'zipf_compliance': zipf_compliance,
            
            # Burstiness
            'burstiness_sentence': self.calculate_burstiness(sentence_lengths),
            'burstiness_word_length': self.calculate_burstiness(word_lengths[:100]),
            'overall_uniformity': 1.0 - self.calculate_cv(sentence_lengths),
            'variance_of_variance': self.calculate_cv(local_vars) if len(local_vars) > 1 else 0.0,
            'local_variance_cv': local_var_cv,
            'autocorrelation_avg': ac_avg,
            
            # Readability
            'avg_word_length': float(np.mean(word_lengths)),
            'word_length_cv': self.calculate_cv(word_lengths),
            'syllable_ratio': sum(self.count_syllables(w) for w in words) / len(words),
            'flesch_reading_ease': readability['flesch_re'],
            'flesch_kincaid_grade': readability['flesch_kg'],
            'gunning_fog_index': readability['gunning_fog'],
            'coleman_liau_index': readability['coleman_liau'],
            'smog_index': readability['smog'],
            'ari_index': readability['ari'],
            'complex_word_percentage': readability['complex_pct'],
            
            # N-grams
            'bigram_repetition_rate': self.calculate_ngram_repetition(words, 2),
            'trigram_repetition_rate': self.calculate_ngram_repetition(words, 3),
            'quadgram_repetition_rate': self.calculate_ngram_repetition(words, 4),
            'pentagram_repetition_rate': self.calculate_ngram_repetition(words, 5),
            'repeated_phrase_count': float(repeated_phrases['count']),
            'repeated_phrase_score': repeated_phrases['score'],
            'high_order_repetition': (self.calculate_ngram_repetition(words, 4) + 
                                      self.calculate_ngram_repetition(words, 5)) / 2,
            'sentence_similarity_avg': self.calculate_sentence_similarity(sentences),
            
            # Punctuation
            'comma_rate': punct_counts['comma'] / len(sentences),
            'semicolon_rate': punct_counts['semicolon'] / len(sentences),
            'colon_rate': punct_counts['colon'] / len(sentences),
            'question_rate': punct_counts['question'] / len(sentences),
            'exclamation_rate': punct_counts['exclamation'] / len(sentences),
            'punctuation_diversity': punct_diversity,
            
            # Structure (normalized, not raw paragraph counts - those are metadata)
            'avg_words_per_paragraph': len(words) / max(1, len(paragraphs)),
            'paragraph_count_normalized': len(paragraphs) / max(1, len(sentences)) * 10,
            'list_marker_count': float(len(re.findall(r'^\s*[-•*]\s', text, re.MULTILINE))),
            'heading_marker_count': float(len(re.findall(r'^#+\s|^[A-Z][^.!?]*:\s*$', text, re.MULTILINE))),
            
            # Function words
            'function_word_ratio': len(func_words) / len(words),
            'content_word_ratio': (len(words) - len(func_words)) / len(words),
            'function_word_diversity': len(set(func_words)) / max(1, len(func_words)),
            'pronoun_density': len(pronouns) / len(words),
            
            # AI signatures
            'hedging_phrase_density': ai_sigs['hedging_density'],
            'discourse_marker_density': ai_sigs['discourse_density'],
            'contraction_rate': ai_sigs['contraction_rate'],
            'contraction_uniformity': ai_sigs['contraction_uniformity'],
            'sentence_starter_variety': ai_sigs['starter_variety'],
            'transition_word_density': ai_sigs['transition_density'],
            'passive_voice_estimate': ai_sigs['passive_estimate'],
            'nominalization_ratio': ai_sigs['nominalization_ratio'],
            
            # Humanizer detection
            'variance_stability': humanizer['variance_stability'],
            'autocorrelation_flatness': humanizer['autocorrelation_flatness'],
            'feature_correlation_strength': humanizer['correlation_strength'],
            'sophistication_variance': humanizer['sophistication_variance'],
            'artificial_variance_score': humanizer['artificial_variance'],
            'humanizer_probability': humanizer['humanizer_prob'],
            
            # Advanced stats
            'runs_test_score': adv_stats['runs_test'],
            'chi_squared_uniformity': adv_stats['chi_squared'],
            'perplexity_estimate': adv_stats['perplexity'],
            'predictability_score': adv_stats['predictability'],
            'periodicity_score': adv_stats['periodicity'],
            'mahalanobis_distance': adv_stats['mahalanobis'],
            'variance_naturalness': adv_stats['variance_naturalness'],
            'extreme_variance_indicator': adv_stats['extreme_variance'],
            
            # Word patterns
            'verb_adverb_ratio': word_patterns['verb_adverb_ratio'],
            'adjective_density': word_patterns['adjective_density'],
            'preposition_density': word_patterns['preposition_density'],
            'determiner_ratio': word_patterns['determiner_ratio'],
            'modal_verb_density': word_patterns['modal_density'],
        }
        
        return features
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """Extract features as numpy array."""
        features = self.extract_features(text)
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


if __name__ == '__main__':
    # Test the extractor
    extractor = FeatureExtractorV4()
    
    print(f"Feature Extractor V4 - Comprehensive Edition")
    print(f"Total features: {len(extractor.feature_names)}")
    print(f"\nFeature categories:")
    
    categories = {
        'Sentence': [f for f in extractor.feature_names if 'sentence' in f],
        'Vocabulary': [f for f in extractor.feature_names if any(x in f for x in ['word', 'hapax', 'ttr', 'yule', 'simpson', 'honore', 'brunet', 'sichel', 'vocab'])],
        'Zipf': [f for f in extractor.feature_names if 'zipf' in f],
        'Burstiness': [f for f in extractor.feature_names if any(x in f for x in ['burst', 'uniform', 'variance', 'autocor'])],
        'Readability': [f for f in extractor.feature_names if any(x in f for x in ['flesch', 'gunning', 'coleman', 'smog', 'ari', 'syllable', 'complex'])],
        'N-gram': [f for f in extractor.feature_names if any(x in f for x in ['gram', 'phrase', 'similar'])],
        'Punctuation': [f for f in extractor.feature_names if any(x in f for x in ['comma', 'semicolon', 'colon', 'question', 'exclamation', 'punct'])],
        'AI Signature': [f for f in extractor.feature_names if any(x in f for x in ['hedging', 'discourse', 'contraction', 'starter', 'transition', 'passive', 'nominal'])],
        'Humanizer': [f for f in extractor.feature_names if any(x in f for x in ['humanizer', 'artificial', 'correlation_strength'])],
        'Advanced Stats': [f for f in extractor.feature_names if any(x in f for x in ['runs', 'chi_squared', 'perplexity', 'predict', 'periodic', 'mahal', 'natural', 'extreme'])],
    }
    
    for cat, features in categories.items():
        print(f"  {cat}: {len(features)} features")
    
    test_text = """
    Artificial intelligence has revolutionized the way we interact with technology.
    Machine learning algorithms have become increasingly sophisticated over time.
    These advancements have led to significant improvements in various industries.
    Natural language processing has made it possible for machines to understand text.
    Computer vision has enabled machines to interpret visual information accurately.
    The future of AI holds tremendous promise for humanity's advancement.
    """
    
    print(f"\nTesting with sample text ({len(test_text.split())} words)...")
    features = extractor.extract_features(test_text)
    
    print(f"\nTop 20 features by value:")
    sorted_features = sorted(features.items(), key=lambda x: -abs(x[1]))[:20]
    for name, value in sorted_features:
        print(f"  {name:40} {value:.4f}")
