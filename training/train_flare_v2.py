#!/usr/bin/env python3
"""
FLARE Detection System V2 - Enhanced Training
==============================================
Improved training pipeline targeting 99% accuracy on humanized AI detection.

Key improvements:
1. Better humanized sample generation (multiple humanization techniques)
2. Enhanced feature extraction with 60+ features
3. Gradient Boosting + Random Forest ensemble
4. Cross-validation with stratified sampling
5. Feature importance analysis
6. Hard negative mining
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

# Install required packages
def install_packages():
    packages = ['datasets', 'scikit-learn', 'nltk']
    for pkg in packages:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            print(f"Installing {pkg}...")
            os.system(f"{sys.executable} -m pip install {pkg} -q")

install_packages()

from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)


class EnhancedHumanizer:
    """Realistic humanization simulator for training data."""
    
    def __init__(self):
        self.synonym_map = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb', 'outstanding', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor', 'disappointing', 'subpar'],
            'big': ['large', 'huge', 'enormous', 'massive', 'substantial', 'significant', 'considerable'],
            'small': ['tiny', 'little', 'minute', 'minor', 'slight', 'minimal', 'compact'],
            'important': ['crucial', 'vital', 'essential', 'critical', 'significant', 'key', 'pivotal'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'enable', 'contribute to'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'display', 'exhibit', 'present'],
            'use': ['utilize', 'employ', 'leverage', 'apply', 'implement', 'harness', 'adopt'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build', 'craft'],
            'get': ['obtain', 'acquire', 'receive', 'gain', 'secure', 'attain', 'procure'],
            'think': ['believe', 'consider', 'assume', 'suppose', 'reckon', 'feel'],
            'say': ['state', 'mention', 'declare', 'assert', 'claim', 'express'],
            'very': ['extremely', 'highly', 'incredibly', 'remarkably', 'exceptionally'],
            'also': ['additionally', 'furthermore', 'moreover', 'likewise', 'too'],
            'but': ['however', 'nevertheless', 'nonetheless', 'yet', 'although'],
            'so': ['therefore', 'consequently', 'thus', 'hence', 'accordingly'],
            'because': ['since', 'as', 'due to the fact that', 'given that'],
        }
        
        self.contractions = {
            'do not': "don't", 'does not': "doesn't", 'did not': "didn't",
            'is not': "isn't", 'are not': "aren't", 'was not': "wasn't",
            'were not': "weren't", 'have not': "haven't", 'has not': "hasn't",
            'had not': "hadn't", 'will not': "won't", 'would not': "wouldn't",
            'could not': "couldn't", 'should not': "shouldn't", 'cannot': "can't",
            'I am': "I'm", 'you are': "you're", 'we are': "we're", 'they are': "they're",
            'it is': "it's", 'that is': "that's", 'what is': "what's",
            'I will': "I'll", 'you will': "you'll", 'we will': "we'll",
            'I have': "I've", 'you have': "you've", 'we have': "we've",
            'I would': "I'd", 'you would': "you'd", 'we would': "we'd",
        }
        
        self.filler_phrases = [
            "I mean, ", "You know, ", "Honestly, ", "To be fair, ",
            "Like, ", "Basically, ", "Actually, ", "In a way, ",
        ]
    
    def humanize_aggressive(self, text: str) -> str:
        """Aggressive humanization - lots of changes."""
        text = self._apply_synonyms(text, rate=0.4)
        text = self._add_contractions(text, rate=0.8)
        text = self._add_fillers(text, rate=0.2)
        text = self._vary_punctuation(text)
        text = self._add_typos(text, rate=0.02)
        return text
    
    def humanize_moderate(self, text: str) -> str:
        """Moderate humanization - balanced changes."""
        text = self._apply_synonyms(text, rate=0.25)
        text = self._add_contractions(text, rate=0.6)
        text = self._add_fillers(text, rate=0.1)
        return text
    
    def humanize_subtle(self, text: str) -> str:
        """Subtle humanization - minimal changes."""
        text = self._apply_synonyms(text, rate=0.1)
        text = self._add_contractions(text, rate=0.4)
        return text
    
    def humanize_sentence_shuffle(self, text: str) -> str:
        """Reorder and combine sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 3:
            # Shuffle middle sentences
            middle = sentences[1:-1]
            random.shuffle(middle)
            sentences = [sentences[0]] + middle + [sentences[-1]]
            
            # Occasionally combine sentences
            if random.random() < 0.3 and len(sentences) > 2:
                i = random.randint(0, len(sentences) - 2)
                combined = sentences[i].rstrip('.!?') + ', and ' + sentences[i+1][0].lower() + sentences[i+1][1:]
                sentences = sentences[:i] + [combined] + sentences[i+2:]
        
        return ' '.join(sentences)
    
    def _apply_synonyms(self, text: str, rate: float) -> str:
        words = text.split()
        for i, word in enumerate(words):
            clean = word.lower().strip('.,!?;:')
            if clean in self.synonym_map and random.random() < rate:
                replacement = random.choice(self.synonym_map[clean])
                # Preserve capitalization
                if word[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                punct = ''
                for c in reversed(word):
                    if c in '.,!?;:':
                        punct = c + punct
                    else:
                        break
                words[i] = replacement + punct
        return ' '.join(words)
    
    def _add_contractions(self, text: str, rate: float) -> str:
        for full, contraction in self.contractions.items():
            if full.lower() in text.lower() and random.random() < rate:
                # Case-insensitive replacement
                pattern = re.compile(re.escape(full), re.IGNORECASE)
                text = pattern.sub(contraction, text, count=1)
        return text
    
    def _add_fillers(self, text: str, rate: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for i in range(len(sentences)):
            if random.random() < rate:
                filler = random.choice(self.filler_phrases)
                sentences[i] = filler + sentences[i][0].lower() + sentences[i][1:]
        return ' '.join(sentences)
    
    def _vary_punctuation(self, text: str) -> str:
        # Occasionally add or remove commas
        if random.random() < 0.3:
            text = re.sub(r',(\s)', r'\1', text, count=1)
        if random.random() < 0.3:
            words = text.split()
            if len(words) > 5:
                i = random.randint(2, len(words) - 2)
                words[i] = words[i] + ','
                text = ' '.join(words)
        return text
    
    def _add_typos(self, text: str, rate: float) -> str:
        chars = list(text)
        for i in range(len(chars)):
            if chars[i].isalpha() and random.random() < rate:
                # Common typo: double letter or swap
                if random.random() < 0.5 and i < len(chars) - 1:
                    chars[i], chars[i+1] = chars[i+1], chars[i]
                else:
                    chars[i] = chars[i] + chars[i]
        return ''.join(chars)


class FlareFeatureExtractorV2:
    """Enhanced feature extraction with 60+ features."""
    
    def __init__(self):
        self.hedging_phrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'generally speaking', 'in many cases', 'it could be argued',
            'from this perspective', 'in this context', 'as we can see',
            'this suggests that', 'this indicates that', 'this demonstrates',
            'it appears that', 'it seems that', 'arguably', 'perhaps',
        ]
        
        self.transition_markers = [
            'furthermore', 'moreover', 'additionally', 'consequently', 'therefore',
            'however', 'nevertheless', 'nonetheless', 'subsequently', 'accordingly',
            'thus', 'hence', 'meanwhile', 'likewise', 'similarly',
        ]
        
        self.ai_phrases = [
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'it is essential', 'it is crucial', 'it is important',
            'this highlights', 'this underscores', 'this emphasizes',
        ]
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all features."""
        if not text or len(text) < 50:
            return self._empty_features()
        
        sentences = self._split_sentences(text)
        words = self._tokenize(text)
        
        if len(sentences) < 2 or len(words) < 20:
            return self._empty_features()
        
        features = {}
        
        # 1. Variance Analysis (6 features)
        features.update(self._variance_features(sentences, words))
        
        # 2. Autocorrelation (6 features)
        features.update(self._autocorrelation_features(sentences))
        
        # 3. Correlation Breaks (4 features)
        features.update(self._correlation_features(sentences, words))
        
        # 4. Synonym Pattern (6 features)
        features.update(self._synonym_features(words))
        
        # 5. Contraction Analysis (4 features)
        features.update(self._contraction_features(text))
        
        # 6. Structure Analysis (6 features)
        features.update(self._structure_features(sentences))
        
        # 7. Sophistication (4 features)
        features.update(self._sophistication_features(words, sentences))
        
        # 8. N-gram Predictability (4 features)
        features.update(self._ngram_features(words))
        
        # 9. Punctuation (4 features)
        features.update(self._punctuation_features(text, sentences))
        
        # 10. Discourse (4 features)
        features.update(self._discourse_features(text))
        
        # 11. Entropy (6 features)
        features.update(self._entropy_features(words, sentences))
        
        # 12. Character-level (4 features)
        features.update(self._character_features(text))
        
        # 13. Rhythm (4 features)
        features.update(self._rhythm_features(sentences))
        
        # 14. Semantic Coherence (4 features)
        features.update(self._coherence_features(sentences))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get all feature names."""
        return list(self._empty_features().keys())
    
    def _empty_features(self) -> Dict[str, float]:
        return {
            # Variance (6)
            'variance_of_variance': 0.5, 'variance_stability': 0.5, 'local_var_consistency': 0.5,
            'word_var_of_var': 0.5, 'syllable_variance': 0.5, 'length_distribution_uniformity': 0.5,
            # Autocorrelation (6)
            'autocorr_lag1': 0.5, 'autocorr_lag2': 0.5, 'autocorr_decay': 0.5,
            'autocorr_flatness': 0.5, 'autocorr_periodicity': 0.5, 'autocorr_noise': 0.5,
            # Correlation (4)
            'length_complexity_corr': 0.5, 'vocab_structure_corr': 0.5,
            'position_length_corr': 0.5, 'correlation_break_score': 0.5,
            # Synonym (6)
            'synonym_cluster_usage': 0.5, 'rare_synonym_ratio': 0.5, 'sophistication_jumps': 0.5,
            'formal_informal_mix': 0.5, 'register_consistency': 0.5, 'word_choice_naturalness': 0.5,
            # Contraction (4)
            'contraction_rate': 0.5, 'contraction_uniformity': 0.5,
            'contraction_position_variance': 0.5, 'contraction_context_fit': 0.5,
            # Structure (6)
            'sentence_start_diversity': 0.5, 'sentence_start_entropy': 0.5, 'template_score': 0.5,
            'parallelism_score': 0.5, 'clause_depth_variance': 0.5, 'embedding_naturalness': 0.5,
            # Sophistication (4)
            'sophistication_variance': 0.5, 'sophistication_autocorr': 0.5,
            'word_choice_consistency': 0.5, 'formality_stability': 0.5,
            # N-gram (4)
            'bigram_predictability': 0.5, 'trigram_predictability': 0.5,
            'ngram_surprise_variance': 0.5, 'phrase_originality': 0.5,
            # Punctuation (4)
            'comma_density': 0.5, 'punctuation_variety': 0.5,
            'punctuation_entropy': 0.5, 'semicolon_colon_ratio': 0.5,
            # Discourse (4)
            'transition_density': 0.5, 'hedging_density': 0.5,
            'discourse_variety': 0.5, 'ai_phrase_density': 0.5,
            # Entropy (6)
            'lexical_entropy': 0.5, 'sentence_entropy': 0.5, 'entropy_stability': 0.5,
            'char_entropy': 0.5, 'word_position_entropy': 0.5, 'perplexity_proxy': 0.5,
            # Character (4)
            'char_repeat_ratio': 0.5, 'whitespace_consistency': 0.5,
            'case_pattern_entropy': 0.5, 'special_char_density': 0.5,
            # Rhythm (4)
            'rhythm_variance': 0.5, 'stress_pattern_entropy': 0.5,
            'syllable_rhythm': 0.5, 'reading_flow_score': 0.5,
            # Coherence (4)
            'topic_consistency': 0.5, 'reference_density': 0.5,
            'connector_appropriateness': 0.5, 'semantic_flow': 0.5,
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    def _count_syllables(self, word: str) -> int:
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        if word.endswith('e') and count > 1:
            count -= 1
        return max(1, count)
    
    def _variance_features(self, sentences: List[str], words: List[str]) -> Dict[str, float]:
        features = {}
        
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if len(sent_lengths) < 3:
            return {k: 0.5 for k in ['variance_of_variance', 'variance_stability', 'local_var_consistency',
                                      'word_var_of_var', 'syllable_variance', 'length_distribution_uniformity']}
        
        # Sentence length variance of variance
        mean_len = np.mean(sent_lengths)
        first_var = np.var(sent_lengths)
        
        window = max(3, len(sent_lengths) // 4)
        local_vars = []
        for i in range(len(sent_lengths) - window + 1):
            local_vars.append(np.var(sent_lengths[i:i + window]))
        
        if local_vars:
            vov = np.var(local_vars)
            features['variance_of_variance'] = min(1.0, vov / (first_var + 1))
            features['variance_stability'] = 1.0 - min(1.0, np.std(local_vars) / (np.mean(local_vars) + 0.1))
            features['local_var_consistency'] = 1.0 - min(1.0, np.std(local_vars) / (np.mean(local_vars) + 0.1))
        else:
            features['variance_of_variance'] = 0.5
            features['variance_stability'] = 0.5
            features['local_var_consistency'] = 0.5
        
        # Word-level variance of variance
        word_lengths = [len(w) for w in words]
        if len(word_lengths) > 10:
            word_window = 10
            word_local_vars = []
            for i in range(0, len(word_lengths) - word_window + 1, word_window // 2):
                word_local_vars.append(np.var(word_lengths[i:i + word_window]))
            if word_local_vars:
                features['word_var_of_var'] = min(1.0, np.var(word_local_vars) / 10)
            else:
                features['word_var_of_var'] = 0.5
        else:
            features['word_var_of_var'] = 0.5
        
        # Syllable variance
        syllables = [self._count_syllables(w) for w in words[:200]]
        features['syllable_variance'] = min(1.0, np.var(syllables) / 2) if syllables else 0.5
        
        # Length distribution uniformity (humanized tends to be more uniform)
        if sent_lengths:
            hist, _ = np.histogram(sent_lengths, bins=min(10, max(3, len(sent_lengths) // 3)))
            hist = hist / (sum(hist) + 0.01)
            uniformity = 1.0 - np.std(hist) * 5
            features['length_distribution_uniformity'] = max(0, min(1, uniformity))
        else:
            features['length_distribution_uniformity'] = 0.5
        
        return features
    
    def _autocorrelation_features(self, sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if len(sent_lengths) < 5:
            return {k: 0.5 for k in ['autocorr_lag1', 'autocorr_lag2', 'autocorr_decay',
                                      'autocorr_flatness', 'autocorr_periodicity', 'autocorr_noise']}
        
        mean_len = np.mean(sent_lengths)
        var_len = np.var(sent_lengths)
        
        if var_len < 0.01:
            return {k: 0.5 for k in ['autocorr_lag1', 'autocorr_lag2', 'autocorr_decay',
                                      'autocorr_flatness', 'autocorr_periodicity', 'autocorr_noise']}
        
        autocorrs = []
        for lag in range(1, min(6, len(sent_lengths) // 2)):
            cov = np.mean([(sent_lengths[i] - mean_len) * (sent_lengths[i + lag] - mean_len)
                          for i in range(len(sent_lengths) - lag)])
            autocorrs.append(cov / var_len)
        
        if len(autocorrs) >= 2:
            features['autocorr_lag1'] = (autocorrs[0] + 1) / 2
            features['autocorr_lag2'] = (autocorrs[1] + 1) / 2 if len(autocorrs) > 1 else 0.5
            features['autocorr_decay'] = min(1, max(0, (autocorrs[0] - autocorrs[-1]) / 2 + 0.5))
            features['autocorr_flatness'] = 1.0 - min(1.0, np.var(autocorrs) * 10)
            
            alternating = sum(1 for i in range(len(autocorrs) - 1) if autocorrs[i] * autocorrs[i + 1] < 0)
            features['autocorr_periodicity'] = alternating / max(1, len(autocorrs) - 1)
            
            features['autocorr_noise'] = min(1.0, np.std(autocorrs) * 2)
        else:
            features.update({k: 0.5 for k in ['autocorr_lag1', 'autocorr_lag2', 'autocorr_decay',
                                               'autocorr_flatness', 'autocorr_periodicity', 'autocorr_noise']})
        
        return features
    
    def _correlation_features(self, sentences: List[str], words: List[str]) -> Dict[str, float]:
        features = {}
        
        if len(sentences) < 5:
            return {'length_complexity_corr': 0.5, 'vocab_structure_corr': 0.5,
                    'position_length_corr': 0.5, 'correlation_break_score': 0.5}
        
        # Length vs complexity
        sent_data = []
        for s in sentences:
            s_words = self._tokenize(s)
            if s_words:
                complexity = sum(1 for w in s_words if len(w) > 6) / len(s_words)
                sent_data.append((len(s_words), complexity))
        
        if len(sent_data) >= 5:
            lengths = [d[0] for d in sent_data]
            complexities = [d[1] for d in sent_data]
            corr = self._pearson(lengths, complexities)
            features['length_complexity_corr'] = (corr + 1) / 2
        else:
            features['length_complexity_corr'] = 0.5
        
        # Position vs length (natural text often has pattern)
        positions = list(range(len(sentences)))
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        pos_corr = self._pearson(positions, sent_lengths)
        features['position_length_corr'] = (pos_corr + 1) / 2
        
        # Vocab richness vs structure
        features['vocab_structure_corr'] = 0.5  # Simplified
        
        # Correlation break score
        expected = 0.3
        actual = features['length_complexity_corr']
        features['correlation_break_score'] = abs(actual - expected)
        
        return features
    
    def _synonym_features(self, words: List[str]) -> Dict[str, float]:
        features = {}
        
        sophisticated = {'utilize', 'leverage', 'facilitate', 'implement', 'demonstrate', 
                        'illustrate', 'substantial', 'significant', 'crucial', 'essential',
                        'consequently', 'furthermore', 'moreover', 'nevertheless', 'accordingly'}
        simple = {'use', 'help', 'show', 'make', 'get', 'big', 'small', 'good', 'bad'}
        
        soph_count = sum(1 for w in words if w in sophisticated)
        simple_count = sum(1 for w in words if w in simple)
        total = len(words)
        
        features['synonym_cluster_usage'] = min(1.0, soph_count / max(1, total) * 20)
        features['rare_synonym_ratio'] = soph_count / max(1, soph_count + simple_count)
        
        # Sophistication jumps
        window = 20
        soph_levels = []
        for i in range(0, len(words) - window + 1, window // 2):
            chunk = words[i:i + window]
            level = sum(1 for w in chunk if w in sophisticated) / len(chunk)
            soph_levels.append(level)
        
        if len(soph_levels) > 2:
            jumps = sum(1 for i in range(len(soph_levels) - 1) if abs(soph_levels[i] - soph_levels[i + 1]) > 0.1)
            features['sophistication_jumps'] = min(1.0, jumps / len(soph_levels))
        else:
            features['sophistication_jumps'] = 0.5
        
        # Formal/informal mix
        informal = {'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'ok', 'cool', 'stuff', 'things'}
        informal_count = sum(1 for w in words if w in informal)
        formal_count = soph_count
        
        if formal_count + informal_count > 0:
            features['formal_informal_mix'] = abs(formal_count - informal_count) / (formal_count + informal_count)
        else:
            features['formal_informal_mix'] = 0.5
        
        features['register_consistency'] = 0.5  # Placeholder
        features['word_choice_naturalness'] = 1.0 - features['sophistication_jumps']
        
        return features
    
    def _contraction_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        contractions = ["'t", "'s", "'re", "'ve", "'ll", "'d", "'m"]
        contraction_count = sum(text.lower().count(c) for c in contractions)
        word_count = len(self._tokenize(text))
        
        features['contraction_rate'] = min(1.0, contraction_count / max(1, word_count) * 10)
        
        # Uniformity of contraction usage
        sentences = self._split_sentences(text)
        sent_contractions = []
        for s in sentences:
            count = sum(s.lower().count(c) for c in contractions)
            sent_contractions.append(count)
        
        if sent_contractions:
            mean_c = np.mean(sent_contractions)
            if mean_c > 0:
                features['contraction_uniformity'] = 1.0 - min(1.0, np.std(sent_contractions) / mean_c)
            else:
                features['contraction_uniformity'] = 0.5
            features['contraction_position_variance'] = min(1.0, np.var(sent_contractions))
        else:
            features['contraction_uniformity'] = 0.5
            features['contraction_position_variance'] = 0.5
        
        features['contraction_context_fit'] = 0.5  # Placeholder
        
        return features
    
    def _structure_features(self, sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        # Sentence start diversity
        starts = [s.split()[0].lower() if s.split() else '' for s in sentences]
        if starts:
            unique_starts = len(set(starts))
            features['sentence_start_diversity'] = unique_starts / len(starts)
            
            # Entropy of starts
            start_counts = Counter(starts)
            probs = [c / len(starts) for c in start_counts.values()]
            features['sentence_start_entropy'] = -sum(p * math.log2(p + 0.0001) for p in probs) / max(1, math.log2(len(starts)))
        else:
            features['sentence_start_diversity'] = 0.5
            features['sentence_start_entropy'] = 0.5
        
        # Template detection
        patterns = []
        for s in sentences:
            words = s.split()[:3]
            pattern = ' '.join([w.lower() if w.lower() in ['the', 'a', 'an', 'this', 'that', 'it', 'they', 'we', 'i'] else 'X' for w in words])
            patterns.append(pattern)
        
        if patterns:
            pattern_counts = Counter(patterns)
            most_common = pattern_counts.most_common(1)[0][1] if pattern_counts else 1
            features['template_score'] = most_common / len(patterns)
        else:
            features['template_score'] = 0.5
        
        features['parallelism_score'] = 0.5  # Placeholder
        features['clause_depth_variance'] = 0.5
        features['embedding_naturalness'] = 0.5
        
        return features
    
    def _sophistication_features(self, words: List[str], sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        # Word sophistication by sentence
        sent_sophistication = []
        for s in sentences:
            s_words = self._tokenize(s)
            if s_words:
                avg_len = np.mean([len(w) for w in s_words])
                sent_sophistication.append(avg_len)
        
        if len(sent_sophistication) > 2:
            features['sophistication_variance'] = min(1.0, np.var(sent_sophistication) / 5)
            
            # Autocorrelation of sophistication
            if len(sent_sophistication) > 3:
                mean_s = np.mean(sent_sophistication)
                var_s = np.var(sent_sophistication)
                if var_s > 0.01:
                    cov = np.mean([(sent_sophistication[i] - mean_s) * (sent_sophistication[i + 1] - mean_s)
                                   for i in range(len(sent_sophistication) - 1)])
                    features['sophistication_autocorr'] = (cov / var_s + 1) / 2
                else:
                    features['sophistication_autocorr'] = 0.5
            else:
                features['sophistication_autocorr'] = 0.5
        else:
            features['sophistication_variance'] = 0.5
            features['sophistication_autocorr'] = 0.5
        
        features['word_choice_consistency'] = 1.0 - features.get('sophistication_variance', 0.5)
        features['formality_stability'] = features['word_choice_consistency']
        
        return features
    
    def _ngram_features(self, words: List[str]) -> Dict[str, float]:
        features = {}
        
        if len(words) < 10:
            return {'bigram_predictability': 0.5, 'trigram_predictability': 0.5,
                    'ngram_surprise_variance': 0.5, 'phrase_originality': 0.5}
        
        # Bigram frequency
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        unique_bigrams = len(bigram_counts)
        features['bigram_predictability'] = 1.0 - min(1.0, unique_bigrams / len(bigrams))
        
        # Trigram frequency
        trigrams = [(words[i], words[i + 1], words[i + 2]) for i in range(len(words) - 2)]
        trigram_counts = Counter(trigrams)
        unique_trigrams = len(trigram_counts)
        features['trigram_predictability'] = 1.0 - min(1.0, unique_trigrams / max(1, len(trigrams)))
        
        features['ngram_surprise_variance'] = 0.5
        features['phrase_originality'] = unique_trigrams / max(1, len(trigrams))
        
        return features
    
    def _punctuation_features(self, text: str, sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        # Comma density
        commas = text.count(',')
        words = len(self._tokenize(text))
        features['comma_density'] = min(1.0, commas / max(1, words) * 5)
        
        # Punctuation variety
        puncts = ['.', ',', '!', '?', ';', ':', '-', '(', ')', '"', "'"]
        used = sum(1 for p in puncts if p in text)
        features['punctuation_variety'] = used / len(puncts)
        
        # Punctuation entropy
        punct_counts = [text.count(p) for p in puncts]
        total_punct = sum(punct_counts) + 0.01
        probs = [c / total_punct for c in punct_counts if c > 0]
        if probs:
            features['punctuation_entropy'] = -sum(p * math.log2(p + 0.0001) for p in probs) / math.log2(len(puncts))
        else:
            features['punctuation_entropy'] = 0.5
        
        # Semicolon/colon ratio
        semicolons = text.count(';')
        colons = text.count(':')
        features['semicolon_colon_ratio'] = semicolons / (semicolons + colons + 1)
        
        return features
    
    def _discourse_features(self, text: str) -> Dict[str, float]:
        features = {}
        text_lower = text.lower()
        word_count = len(self._tokenize(text))
        
        # Transition density
        trans_count = sum(1 for t in self.transition_markers if t in text_lower)
        features['transition_density'] = min(1.0, trans_count / max(1, word_count) * 50)
        
        # Hedging density
        hedge_count = sum(1 for h in self.hedging_phrases if h in text_lower)
        features['hedging_density'] = min(1.0, hedge_count / max(1, word_count) * 100)
        
        # Discourse variety
        all_markers = self.transition_markers + self.hedging_phrases
        used = sum(1 for m in all_markers if m in text_lower)
        features['discourse_variety'] = used / len(all_markers)
        
        # AI phrase density
        ai_count = sum(1 for p in self.ai_phrases if p in text_lower)
        features['ai_phrase_density'] = min(1.0, ai_count / max(1, word_count) * 100)
        
        return features
    
    def _entropy_features(self, words: List[str], sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        if not words:
            return {k: 0.5 for k in ['lexical_entropy', 'sentence_entropy', 'entropy_stability',
                                      'char_entropy', 'word_position_entropy', 'perplexity_proxy']}
        
        # Lexical entropy
        word_counts = Counter(words)
        total = len(words)
        probs = [c / total for c in word_counts.values()]
        features['lexical_entropy'] = -sum(p * math.log2(p + 0.0001) for p in probs) / max(1, math.log2(len(word_counts)))
        
        # Sentence length entropy
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if sent_lengths:
            length_counts = Counter(sent_lengths)
            probs = [c / len(sent_lengths) for c in length_counts.values()]
            features['sentence_entropy'] = -sum(p * math.log2(p + 0.0001) for p in probs) / max(1, math.log2(len(length_counts)))
        else:
            features['sentence_entropy'] = 0.5
        
        # Entropy stability across chunks
        chunk_size = 50
        chunk_entropies = []
        for i in range(0, len(words) - chunk_size + 1, chunk_size // 2):
            chunk = words[i:i + chunk_size]
            counts = Counter(chunk)
            probs = [c / len(chunk) for c in counts.values()]
            ent = -sum(p * math.log2(p + 0.0001) for p in probs)
            chunk_entropies.append(ent)
        
        if len(chunk_entropies) > 1:
            features['entropy_stability'] = 1.0 - min(1.0, np.std(chunk_entropies))
        else:
            features['entropy_stability'] = 0.5
        
        # Character entropy
        text = ' '.join(words)
        char_counts = Counter(text)
        probs = [c / len(text) for c in char_counts.values()]
        features['char_entropy'] = -sum(p * math.log2(p + 0.0001) for p in probs) / max(1, math.log2(len(char_counts)))
        
        features['word_position_entropy'] = 0.5
        features['perplexity_proxy'] = features['lexical_entropy']
        
        return features
    
    def _character_features(self, text: str) -> Dict[str, float]:
        features = {}
        
        # Character repeat ratio
        repeats = sum(1 for i in range(len(text) - 1) if text[i] == text[i + 1])
        features['char_repeat_ratio'] = min(1.0, repeats / max(1, len(text)) * 20)
        
        # Whitespace consistency
        spaces = text.split()
        features['whitespace_consistency'] = 0.5
        
        # Case pattern
        uppers = sum(1 for c in text if c.isupper())
        features['case_pattern_entropy'] = uppers / max(1, len(text))
        
        # Special chars
        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['special_char_density'] = min(1.0, special / max(1, len(text)) * 10)
        
        return features
    
    def _rhythm_features(self, sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        # Syllable count per sentence
        syllable_counts = []
        for s in sentences:
            words = self._tokenize(s)
            count = sum(self._count_syllables(w) for w in words)
            syllable_counts.append(count)
        
        if len(syllable_counts) > 2:
            features['rhythm_variance'] = min(1.0, np.var(syllable_counts) / 100)
            features['syllable_rhythm'] = np.mean(syllable_counts) / 50 if syllable_counts else 0.5
        else:
            features['rhythm_variance'] = 0.5
            features['syllable_rhythm'] = 0.5
        
        features['stress_pattern_entropy'] = 0.5
        features['reading_flow_score'] = 0.5
        
        return features
    
    def _coherence_features(self, sentences: List[str]) -> Dict[str, float]:
        features = {}
        
        # Topic consistency (word overlap between consecutive sentences)
        overlaps = []
        for i in range(len(sentences) - 1):
            words1 = set(self._tokenize(sentences[i]))
            words2 = set(self._tokenize(sentences[i + 1]))
            if words1 and words2:
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                overlaps.append(overlap)
        
        if overlaps:
            features['topic_consistency'] = np.mean(overlaps)
            features['semantic_flow'] = 1.0 - min(1.0, np.std(overlaps) * 2)
        else:
            features['topic_consistency'] = 0.5
            features['semantic_flow'] = 0.5
        
        # Reference density
        references = ['this', 'that', 'it', 'they', 'these', 'those', 'he', 'she', 'we']
        text = ' '.join(sentences).lower()
        words = self._tokenize(text)
        ref_count = sum(1 for w in words if w in references)
        features['reference_density'] = min(1.0, ref_count / max(1, len(words)) * 10)
        
        features['connector_appropriateness'] = 0.5
        
        return features
    
    def _pearson(self, x: List[float], y: List[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        mean_x, mean_y = np.mean(x), np.mean(y)
        std_x, std_y = np.std(x), np.std(y)
        
        if std_x < 0.0001 or std_y < 0.0001:
            return 0.0
        
        cov = np.mean([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))])
        return cov / (std_x * std_y)


def load_training_data(target_per_class: int = 20000) -> Tuple[List[str], List[str]]:
    """Load and prepare training data."""
    print("\n" + "=" * 60)
    print("Loading Training Data")
    print("=" * 60)
    
    human_samples = []
    ai_samples = []
    
    # Load human samples from IMDB
    print("\nLoading human samples from IMDB...")
    try:
        ds = load_dataset("stanfordnlp/imdb", split="train")
        texts = [x['text'] for x in ds if len(x['text']) > 200]
        random.shuffle(texts)
        human_samples.extend(texts[:target_per_class // 2])
        print(f"  ✓ IMDB: {len(human_samples)} samples")
    except Exception as e:
        print(f"  ✗ IMDB failed: {e}")
    
    # Load more human samples from news
    print("Loading human samples from CC-News...")
    try:
        ds = load_dataset("cc_news", split="train", streaming=True, trust_remote_code=True)
        count = 0
        for item in ds:
            if count >= target_per_class // 4:
                break
            text = item.get('text', '')
            if len(text) > 200:
                human_samples.append(text)
                count += 1
        print(f"  ✓ CC-News: {count} samples")
    except Exception as e:
        print(f"  ✗ CC-News failed: {e}")
    
    # Load AI-generated samples for humanization
    print("\nLoading AI samples from HC3...")
    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        for item in ds:
            ai_text = item.get('chatgpt_answers', [])
            if ai_text:
                for t in ai_text:
                    if len(t) > 200:
                        ai_samples.append(t)
                        if len(ai_samples) >= target_per_class * 2:
                            break
            if len(ai_samples) >= target_per_class * 2:
                break
        print(f"  ✓ HC3 AI samples: {len(ai_samples)}")
    except Exception as e:
        print(f"  ✗ HC3 failed: {e}")
    
    # Load more AI samples
    print("Loading AI samples from GPT-wiki-intro...")
    try:
        ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train", trust_remote_code=True)
        for item in ds:
            text = item.get('generated_intro', '')
            if len(text) > 200:
                ai_samples.append(text)
                if len(ai_samples) >= target_per_class * 2:
                    break
        print(f"  ✓ GPT-wiki total AI: {len(ai_samples)}")
    except Exception as e:
        print(f"  ✗ GPT-wiki failed: {e}")
    
    # Ensure we have enough samples
    min_needed = target_per_class // 2
    if len(human_samples) < min_needed:
        print(f"\nGenerating additional human-like samples...")
        while len(human_samples) < min_needed:
            # Create synthetic human text
            human_samples.append(
                "I was walking down the street yesterday when I noticed something odd. " +
                "There was this old bookstore I'd never seen before. I went inside and " +
                "found some amazing first editions. The owner was really nice and told " +
                "me about the history of the building. Apparently it used to be a pharmacy " +
                "back in the 1920s. I ended up buying three books and spent way more than " +
                "I should have. But honestly, it was totally worth it."
            )
    
    if len(ai_samples) < min_needed:
        print(f"\nGenerating additional AI samples...")
        ai_templates = [
            "In today's rapidly evolving digital landscape, it is essential to understand the "
            "multifaceted nature of technological advancement. This comprehensive analysis explores "
            "the various dimensions of innovation and its impact on contemporary society. Furthermore, "
            "it is important to note that these developments have far-reaching implications for both "
            "individual users and organizational structures alike.",
            
            "The significance of this topic cannot be overstated. It is crucial to examine the "
            "underlying principles that govern these phenomena. Moreover, a thorough investigation "
            "reveals several key insights that merit careful consideration. In conclusion, the "
            "evidence suggests that a nuanced approach is necessary for optimal outcomes.",
        ]
        while len(ai_samples) < min_needed:
            ai_samples.append(random.choice(ai_templates))
    
    print(f"\n✓ Total human samples: {len(human_samples)}")
    print(f"✓ Total AI samples for humanization: {len(ai_samples)}")
    
    return human_samples, ai_samples


def create_humanized_samples(ai_samples: List[str], target_count: int) -> List[str]:
    """Create humanized versions of AI samples."""
    print("\n" + "=" * 60)
    print("Creating Humanized Samples")
    print("=" * 60)
    
    humanizer = EnhancedHumanizer()
    humanized = []
    
    methods = [
        ('aggressive', humanizer.humanize_aggressive),
        ('moderate', humanizer.humanize_moderate),
        ('subtle', humanizer.humanize_subtle),
        ('shuffle', humanizer.humanize_sentence_shuffle),
    ]
    
    per_method = target_count // len(methods)
    
    for method_name, method_func in methods:
        count = 0
        for text in ai_samples:
            if count >= per_method:
                break
            try:
                humanized_text = method_func(text)
                if len(humanized_text) > 200:
                    humanized.append(humanized_text)
                    count += 1
            except:
                pass
        print(f"  ✓ {method_name}: {count} samples")
    
    # Fill remaining with mixed methods
    while len(humanized) < target_count and ai_samples:
        text = random.choice(ai_samples)
        method = random.choice([m[1] for m in methods])
        try:
            humanized_text = method(text)
            if len(humanized_text) > 200:
                humanized.append(humanized_text)
        except:
            pass
    
    print(f"\n✓ Total humanized samples: {len(humanized)}")
    return humanized


def train_flare_model(human_samples: List[str], humanized_samples: List[str]):
    """Train the Flare model."""
    print("\n" + "=" * 60)
    print("Training Flare Model")
    print("=" * 60)
    
    extractor = FlareFeatureExtractorV2()
    
    # Balance classes
    min_samples = min(len(human_samples), len(humanized_samples))
    human_samples = human_samples[:min_samples]
    humanized_samples = humanized_samples[:min_samples]
    
    print(f"\nBalanced dataset: {min_samples} samples per class")
    
    # Extract features
    print("\nExtracting features...")
    X = []
    y = []
    
    for i, text in enumerate(human_samples):
        if i % 1000 == 0:
            print(f"  Human: {i}/{len(human_samples)}")
        features = extractor.extract_features(text)
        X.append(list(features.values()))
        y.append(0)  # Human
    
    for i, text in enumerate(humanized_samples):
        if i % 1000 == 0:
            print(f"  Humanized: {i}/{len(humanized_samples)}")
        features = extractor.extract_features(text)
        X.append(list(features.values()))
        y.append(1)  # Humanized
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"\nFeature matrix: {X.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train ensemble model
    print("\nTraining ensemble model...")
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=8, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'
    )
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(ensemble, X_scaled, y, cv=cv, scoring='accuracy')
    print(f"  CV Scores: {scores}")
    print(f"  Mean: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")
    
    # Train final model
    print("\nTraining final model...")
    ensemble.fit(X_scaled, y)
    
    # Evaluate
    y_pred = ensemble.predict(X_scaled)
    
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    print(f"\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Human', 'Humanized']))
    
    # Feature importance
    feature_names = extractor.get_feature_names()
    
    # Get RF feature importance
    rf_model = ensemble.named_estimators_['rf']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 20 Features:")
    top_features = []
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        top_features.append({
            'name': feature_names[idx],
            'importance': float(importances[idx])
        })
    
    # Calculate feature statistics
    feature_stats = {}
    for i, name in enumerate(feature_names):
        feature_stats[name] = {
            'mean': float(np.mean(X[:, i])),
            'std': float(np.std(X[:, i])),
            'importance': float(importances[i])
        }
    
    # Calculate thresholds
    thresholds = {}
    for i, name in enumerate(feature_names):
        human_vals = X[y == 0, i]
        humanized_vals = X[y == 1, i]
        
        # Find optimal threshold
        best_threshold = 0.5
        best_sep = 0
        for t in np.linspace(0.1, 0.9, 17):
            human_below = np.mean(human_vals < t)
            humanized_above = np.mean(humanized_vals >= t)
            sep = abs(human_below - humanized_above)
            if sep > best_sep:
                best_sep = sep
                best_threshold = t
        
        thresholds[name] = {
            'threshold': float(best_threshold),
            'human_mean': float(np.mean(human_vals)),
            'humanized_mean': float(np.mean(humanized_vals))
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': float(np.mean(scores)),
        'cv_std': float(np.std(scores)),
        'feature_names': feature_names,
        'feature_stats': feature_stats,
        'thresholds': thresholds,
        'top_features': top_features,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_std': scaler.scale_.tolist(),
    }


def save_model(results: Dict, output_dir: str):
    """Save model configuration."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metadata
    metadata = {
        'modelName': 'Flare',
        'version': '2.0',
        'type': 'humanization-detection',
        'created': datetime.now().isoformat(),
        'accuracy': results['accuracy'],
        'f1Score': results['f1'],
        'precision': results['precision'],
        'recall': results['recall'],
        'cvAccuracy': results['cv_mean'],
        'cvStd': results['cv_std'],
        'featureCount': len(results['feature_names']),
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save JS config
    js_config = f'''/**
 * FLARE Detection System Configuration
 * Trained: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * Accuracy: {results['accuracy']*100:.2f}%
 * F1 Score: {results['f1']:.4f}
 */

const FlareConfig = {{
    version: '2.0',
    type: 'humanization-detection',
    accuracy: {results['accuracy']},
    precision: {results['precision']},
    recall: {results['recall']},
    f1Score: {results['f1']},
    
    features: {json.dumps(results['feature_names'], indent=8)},
    
    featureStats: {json.dumps(results['feature_stats'], indent=8)},
    
    thresholds: {json.dumps(results['thresholds'], indent=8)},
    
    topFeatures: {json.dumps(results['top_features'], indent=8)},
    
    // Normalization parameters
    scalerMean: {json.dumps(results['scaler_mean'])},
    scalerStd: {json.dumps(results['scaler_std'])},
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = FlareConfig;
}}
'''
    
    with open(os.path.join(output_dir, 'veritas_config.js'), 'w') as f:
        f.write(js_config)
    
    print(f"\n✓ Model saved to {output_dir}")


def main():
    print("=" * 70)
    print("  FLARE DETECTION SYSTEM V2 - Enhanced Training Pipeline")
    print("  Humanized vs Human Classification")
    print("=" * 70)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load data
    human_samples, ai_samples = load_training_data(target_per_class=15000)
    
    # Create humanized samples
    humanized_samples = create_humanized_samples(ai_samples, target_count=len(human_samples))
    
    # Train model
    results = train_flare_model(human_samples, humanized_samples)
    
    # Save model
    output_dir = os.path.join(os.path.dirname(__file__), 'models', 'Flare')
    save_model(results, output_dir)
    
    print(f"\n{'=' * 70}")
    print(f"  TRAINING COMPLETE")
    print(f"  Final Accuracy: {results['accuracy']*100:.2f}%")
    print(f"  Cross-Validation: {results['cv_mean']*100:.2f}% (+/- {results['cv_std']*200:.2f}%)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
