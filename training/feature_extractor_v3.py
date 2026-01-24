"""
Veritas Feature Extractor v3 - SUPERNOVA v2 Edition
===================================================

MAJOR IMPROVEMENTS over v1/v2:
1. ESL/Non-native detection features
2. Student/academic writing patterns  
3. Speech/presentation markers
4. Awkward construction detection
5. Authenticity signals that humanizers CAN'T fake
6. Domain-aware feature adjustments

Key insight: AI text is PERFECT. Human text has authentic imperfections.
We need to detect those imperfections, not just statistical patterns.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set
import numpy as np


class FeatureExtractorV3:
    """
    Enhanced feature extractor for SUPERNOVA v2.
    
    Focus areas:
    1. Authenticity signals - things humans do that AI doesn't
    2. ESL patterns - non-native but still human
    3. Student writing - formal but imperfect
    4. Domain detection - adjust expectations by context
    """
    
    def __init__(self):
        self.feature_names = self._build_feature_names()
        
        # Common ESL mistakes (authentic human signals)
        self.esl_patterns = [
            r'\bthe\s+\w+\s+the\b',  # Double articles
            r'\b(a|an)\s+[aeiou]\w+\b.*\b(a|an)\s+[^aeiou]\w+\b',  # a/an confusion
            r'\bmore\s+\w+er\b',  # more better
            r'\bvery\s+\w+ly\b',  # very quickly (often overused)
            r'\bmost\s+\w+est\b',  # most biggest
            r'\bin\s+the\s+\d{4}\b',  # "in the 2024" instead of "in 2024"
            r'\bsince\s+\d+\s+years?\b',  # "since 5 years" vs "for 5 years"
            r'\bI\s+am\s+agree\b',  # common ESL error
            r'\bthe\s+(my|your|his|her)\b',  # "the my friend"
        ]
        
        # Student/academic writing markers (Model UN, debate, presentations)
        self.student_markers = [
            # Greetings and salutations
            r'\b(dear|distinguished|honorable|hello)\s+(delegates?|chair|committee|members?)\b',
            r'\bhonor(able|ed)\s+(delegates?|chair)\b',
            # Self-introduction
            r'\bmy\s+name\s+is\b',
            r'\bI\s+am\s+representing\b',
            r'\btoday\s+I\s+will\s+(be\s+)?(discussing|presenting|addressing)\b',
            r'\bI\s+will\s+be\s+(discussing|representing|presenting|addressing)\b',
            # Delegation/country references
            r'\bmy\s+(country|delegation|position|nation)\b',
            r'\bthe\s+delegation\s+of\b',
            r'\bour\s+(country|delegation|nation|position)\s+(believes?|proposes?|supports?)\b',
            # Structural markers
            r'\bIn\s+conclusion\b',
            r'\bto\s+(summarize|conclude|sum\s+up)\b',
            r'\bthank\s+you\s+for\s+(your\s+)?(time|attention|listening)\b',
            r'\bmodel\s+un\b',
            # Proposal language
            r'\bproposes?\s+(several|the\s+following|that)\b',
            r'\bwe\s+(firmly\s+)?believe\b',
            r'\bcommittee\s+sessions?\b',
            r'\bFirst[,.].*Second[,.].*Third\b',
            # Opinion markers
            r'\bI\s+believe\s+that\b',
            r'\bbelieve\s+that\b',
            # MUN-specific
            r'\bposition\s+paper\b',
            r'\bresolution\s+draft\b',
            r'\bworking\s+paper\b',
            r'\burges?\s+(all\s+)?(member\s+)?states?\b',
        ]
        
        # Authentic human uncertainty markers
        self.uncertainty_markers = [
            r'\bI\s+(think|believe|feel)\s+that\b',
            r'\bmaybe\b',
            r'\bprobably\b',
            r'\bkind\s+of\b',
            r'\bsort\s+of\b',
            r'\blike\b(?!\s*$)',  # filler "like" (not at end)
            r'\byou\s+know\b',
            r'\bI\s+guess\b',
            r'\bI\'?m\s+not\s+sure\b',
            r'\bif\s+I\'?m\s+(being\s+)?honest\b',
        ]
        
        # Run-on sentence patterns (human error)
        self.run_on_patterns = [
            r',[^,]{50,},[^,]{50,},[^,]{50,}',  # Very long comma chains
            r'\band\s+\w+\s+and\s+\w+\s+and\b',  # Multiple ands
            r'[^.!?]{200,}[.!?]',  # Extremely long sentence
        ]
        
        # Informal/casual markers
        self.casual_markers = [
            r'\blol\b', r'\bhaha\b', r'\bbtw\b', r'\bomg\b',
            r'\bwow\b', r'\bugh\b', r'\byeah\b', r'\bnah\b',
            r'\bgonna\b', r'\bwanna\b', r'\bgotta\b', r'\bkinda\b',
            r'\bdunno\b', r'\bcuz\b', r'\bcause\b', r'\btho\b',
        ]
        
        # AI-typical patterns (things AI does that humans rarely do)
        self.ai_typical_patterns = [
            r'\bLet\s+me\s+(explain|help|provide)\b',
            r'\bHere\'?s?\s+(a|the|an|my)\b',
            r'\bI\'?d\s+be\s+happy\s+to\b',
            r'\bCertainly[!.]',
            r'\bAbsolutely[!.]',
            r'\bGreat\s+question\b',
            r'\bThat\'?s\s+a\s+(great|good|excellent)\s+question\b',
            r'\bI\s+hope\s+this\s+helps\b',
            r'\bFeel\s+free\s+to\b',
            r'\bDon\'?t\s+hesitate\s+to\b',
            r'\bPlease\s+let\s+me\s+know\b',
            r':\s*\n\s*\d+\.',  # Numbered list after colon
            r'\bFirstly\b.*\bSecondly\b',  # Formal listing
            r'\bIn\s+summary[,:]',
            r'\bTo\s+summarize[,:]',
        ]
        
        # Formality inconsistency (human trait)
        self.formal_words = {
            'utilize', 'facilitate', 'implement', 'commence', 'terminate',
            'endeavor', 'subsequently', 'aforementioned', 'henceforth',
            'notwithstanding', 'whereby', 'thereof', 'herein', 'pursuant'
        }
        self.informal_words = {
            'stuff', 'things', 'a lot', 'lots', 'really', 'pretty',
            'kind of', 'sort of', 'basically', 'actually', 'just',
            'so', 'very', 'super', 'totally', 'literally'
        }
    
    def _build_feature_names(self) -> List[str]:
        """Build comprehensive feature name list."""
        return [
            # === AUTHENTICITY SIGNALS (Human quirks) ===
            'esl_pattern_count',
            'esl_pattern_density',
            'awkward_construction_count',
            'run_on_sentence_count',
            'fragment_sentence_count',
            'comma_splice_count',
            'formality_inconsistency',
            'uncertainty_marker_count',
            'casual_marker_count',
            'student_marker_count',
            
            # === AI-TYPICAL SIGNALS ===
            'ai_phrase_count',
            'ai_phrase_density',
            'numbered_list_count',
            'bullet_structure_score',
            'perfect_parallelism_score',
            
            # === SENTENCE VARIANCE (AI = uniform) ===
            'sentence_count',
            'avg_sentence_length',
            'sentence_length_cv',
            'sentence_length_std',
            'sentence_length_min',
            'sentence_length_max',
            'sentence_length_range',
            'sentence_length_skewness',
            'sentence_length_kurtosis',
            'short_sentence_ratio',
            'long_sentence_ratio',
            'sentence_length_entropy',
            
            # === VOCABULARY DIVERSITY ===
            'word_count',
            'unique_word_count',
            'type_token_ratio',
            'hapax_ratio',
            'dis_legomena_ratio',
            'vocab_sophistication',
            'rare_word_ratio',
            
            # === STATISTICAL FEATURES ===
            'zipf_slope',
            'zipf_r_squared',
            'burstiness_sentence',
            'burstiness_word_length',
            
            # === READABILITY ===
            'avg_word_length',
            'word_length_cv',
            'syllable_ratio',
            'flesch_kincaid_grade',
            'automated_readability_index',
            
            # === REPETITION PATTERNS ===
            'bigram_repetition_rate',
            'trigram_repetition_rate',
            'sentence_similarity_avg',
            'word_repetition_local',
            'phrase_repetition_rate',
            
            # === PUNCTUATION PATTERNS ===
            'comma_rate',
            'semicolon_rate',
            'colon_rate',
            'question_rate',
            'exclamation_rate',
            'dash_rate',
            'parenthesis_rate',
            'ellipsis_count',
            
            # === STRUCTURE ===
            'paragraph_count',
            'avg_paragraph_length',
            'paragraph_length_cv',
            'has_greeting',
            'has_closing',
            'has_signature',
            
            # === PRONOUN PATTERNS ===
            'first_person_singular_rate',
            'first_person_plural_rate',
            'second_person_rate',
            'third_person_rate',
            'pronoun_variety',
            
            # === DISCOURSE MARKERS ===
            'transition_word_count',
            'transition_density',
            'causal_connector_count',
            'contrast_connector_count',
            'addition_connector_count',
            
            # === DOMAIN SIGNALS ===
            'is_academic_style',
            'is_casual_style',
            'is_speech_style',
            'is_technical_style',
            'proper_noun_ratio',
            'citation_count',
            'url_count',
            'date_mention_count',
            
            # === STRONG HUMAN INDICATORS ===
            'formal_speech_strength',  # Strong indicator for MUN/debate
            'human_authenticity_score',  # Combined authenticity signal
            
            # === EMBEDDING-INDEPENDENT SIGNALS ===
            'typo_density',
            'spelling_inconsistency',
            'contraction_rate',
            'contraction_consistency',
            'capitalization_errors',
        ]
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        words = re.findall(r"[a-zA-Z'-]+", text.lower())
        return [w for w in words if len(w) > 0]
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        abbrevs = r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd|i\.e|e\.g)'
        text = re.sub(f'{abbrevs}\\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        
        # Split on sentence-ending punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore periods
        sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences]
        
        # Filter empty and too-short sentences
        return [s for s in sentences if len(s.split()) >= 2]
    
    def count_pattern_matches(self, text: str, patterns: List[str]) -> int:
        """Count how many patterns match in the text."""
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            count += len(matches)
        return count
    
    def calculate_cv(self, values: List[float]) -> float:
        """Calculate coefficient of variation."""
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        if mean == 0:
            return 0.0
        return float(np.std(values, ddof=1) / mean)
    
    def calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a distribution."""
        if len(values) < 2:
            return 0.0
        # Normalize to probabilities
        total = sum(values)
        if total == 0:
            return 0.0
        probs = [v / total for v in values]
        # Calculate entropy
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return float(entropy)
    
    def calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of distribution."""
        if len(values) < 3:
            return 0.0
        values = np.array(values)
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
        values = np.array(values)
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if std == 0:
            return 0.0
        m4 = np.mean((values - mean) ** 4)
        return float(m4 / (std ** 4) - 3)
    
    def detect_typos(self, words: List[str]) -> float:
        """
        Detect potential typos/misspellings.
        Uses heuristics since we don't want heavy dependencies.
        """
        typo_indicators = 0
        
        for word in words:
            # Repeated characters (typo indicator)
            if re.search(r'(.)\1{2,}', word):
                typo_indicators += 1
            # Unusual character sequences
            if re.search(r'[qwx]{2}|[zxv]{2}', word):
                typo_indicators += 1
            # Missing vowels in long words
            if len(word) > 5 and not re.search(r'[aeiou]', word):
                typo_indicators += 1
        
        return typo_indicators / max(len(words), 1)
    
    def detect_comma_splices(self, text: str) -> int:
        """Detect comma splices (run-on sentences joined by comma)."""
        # Pattern: complete thought, complete thought
        # Look for: Subject Verb ... , Subject Verb
        pattern = r'[A-Z][^.!?]*\b(is|are|was|were|has|have|do|does|did|will|would|can|could)\b[^.!?]*,\s*[A-Z][^.!?]*\b(is|are|was|were|has|have|do|does|did|will|would|can|could)\b'
        return len(re.findall(pattern, text))
    
    def detect_fragment_sentences(self, sentences: List[str]) -> int:
        """Detect sentence fragments (incomplete sentences)."""
        fragments = 0
        for sent in sentences:
            words = sent.split()
            if len(words) < 4:
                # Very short - might be fragment
                if not re.search(r'\b(is|are|was|were|am|be|been|being|have|has|had|do|does|did|will|would|shall|should|can|could|may|might|must)\b', sent, re.I):
                    fragments += 1
            # Starts with subordinating conjunction but no main clause
            if re.match(r'^(Because|Although|While|When|If|Since|Unless|Until)\b', sent, re.I):
                if len(words) < 8 and sent[-1] not in '.!?':
                    fragments += 1
        return fragments
    
    def calculate_formality_inconsistency(self, text: str) -> float:
        """
        Detect mixing of formal and informal language.
        Humans often mix; AI is consistently formal.
        """
        text_lower = text.lower()
        words = set(self.tokenize(text_lower))
        
        formal_count = len(words & self.formal_words)
        informal_count = len(words & self.informal_words)
        
        # If both present, there's inconsistency (human trait)
        total = formal_count + informal_count
        if total == 0:
            return 0.0
        
        # Inconsistency = presence of both styles
        inconsistency = min(formal_count, informal_count) / max(total, 1)
        return inconsistency
    
    def detect_perfect_parallelism(self, text: str) -> float:
        """
        Detect perfect parallel structure (AI trait).
        Humans naturally vary their phrasing.
        """
        sentences = self.split_sentences(text)
        if len(sentences) < 3:
            return 0.0
        
        # Look for sentences starting the same way
        starts = [' '.join(s.split()[:3]).lower() for s in sentences if len(s.split()) >= 3]
        start_counts = Counter(starts)
        
        # High repetition of starts = AI pattern
        repeated_starts = sum(1 for c in start_counts.values() if c > 1)
        return repeated_starts / max(len(sentences), 1)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all features from text."""
        # Basic tokenization
        words = self.tokenize(text)
        sentences = self.split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        text_lower = text.lower()
        
        if len(words) < 10 or len(sentences) < 1:
            # Return zeros for too-short text
            return {name: 0.0 for name in self.feature_names}
        
        # Word counts
        word_counts = Counter(words)
        hapax = [w for w, c in word_counts.items() if c == 1]
        dis_legomena = [w for w, c in word_counts.items() if c == 2]
        
        # Sentence lengths
        sentence_lengths = [len(self.tokenize(s)) for s in sentences]
        sentence_lengths = [l for l in sentence_lengths if l > 0] or [1]
        sentence_lengths_arr = np.array(sentence_lengths, dtype=float)
        
        # Word lengths
        word_lengths = [len(w) for w in words]
        
        # Paragraph lengths
        para_word_counts = [len(self.tokenize(p)) for p in paragraphs] or [len(words)]
        
        # === AUTHENTICITY SIGNALS ===
        esl_count = self.count_pattern_matches(text, self.esl_patterns)
        student_count = self.count_pattern_matches(text, self.student_markers)
        uncertainty_count = self.count_pattern_matches(text, self.uncertainty_markers)
        casual_count = self.count_pattern_matches(text, self.casual_markers)
        run_on_count = self.count_pattern_matches(text, self.run_on_patterns)
        
        # AI patterns
        ai_phrase_count = self.count_pattern_matches(text, self.ai_typical_patterns)
        
        # Structural patterns
        numbered_lists = len(re.findall(r'^\s*\d+[.)]\s+', text, re.MULTILINE))
        bullet_items = len(re.findall(r'^\s*[-*•]\s+', text, re.MULTILINE))
        
        # === BUILD FEATURE DICT ===
        features = {
            # Authenticity signals
            'esl_pattern_count': esl_count,
            'esl_pattern_density': esl_count / max(len(sentences), 1),
            'awkward_construction_count': esl_count + run_on_count,
            'run_on_sentence_count': run_on_count,
            'fragment_sentence_count': self.detect_fragment_sentences(sentences),
            'comma_splice_count': self.detect_comma_splices(text),
            'formality_inconsistency': self.calculate_formality_inconsistency(text),
            'uncertainty_marker_count': uncertainty_count,
            'casual_marker_count': casual_count,
            'student_marker_count': student_count,
            
            # AI-typical signals
            'ai_phrase_count': ai_phrase_count,
            'ai_phrase_density': ai_phrase_count / max(len(sentences), 1),
            'numbered_list_count': numbered_lists,
            'bullet_structure_score': bullet_items / max(len(paragraphs), 1),
            'perfect_parallelism_score': self.detect_perfect_parallelism(text),
            
            # Sentence variance
            'sentence_count': len(sentences),
            'avg_sentence_length': float(np.mean(sentence_lengths_arr)),
            'sentence_length_cv': self.calculate_cv(sentence_lengths),
            'sentence_length_std': float(np.std(sentence_lengths_arr, ddof=1)) if len(sentence_lengths_arr) > 1 else 0,
            'sentence_length_min': float(np.min(sentence_lengths_arr)),
            'sentence_length_max': float(np.max(sentence_lengths_arr)),
            'sentence_length_range': float(np.max(sentence_lengths_arr) - np.min(sentence_lengths_arr)),
            'sentence_length_skewness': self.calculate_skewness(sentence_lengths),
            'sentence_length_kurtosis': self.calculate_kurtosis(sentence_lengths),
            'short_sentence_ratio': sum(1 for l in sentence_lengths if l < 8) / len(sentence_lengths),
            'long_sentence_ratio': sum(1 for l in sentence_lengths if l > 30) / len(sentence_lengths),
            'sentence_length_entropy': self.calculate_entropy(sentence_lengths),
            
            # Vocabulary
            'word_count': len(words),
            'unique_word_count': len(word_counts),
            'type_token_ratio': len(word_counts) / len(words) if words else 0,
            'hapax_ratio': len(hapax) / len(word_counts) if word_counts else 0,
            'dis_legomena_ratio': len(dis_legomena) / len(word_counts) if word_counts else 0,
            'vocab_sophistication': sum(1 for w in words if len(w) > 8) / len(words) if words else 0,
            'rare_word_ratio': len([w for w in words if word_counts[w] == 1]) / len(words) if words else 0,
            
            # Statistical
            'zipf_slope': self._calculate_zipf_slope(word_counts),
            'zipf_r_squared': 0.0,  # Placeholder
            'burstiness_sentence': self._calculate_burstiness(sentence_lengths),
            'burstiness_word_length': self._calculate_burstiness(word_lengths[:100]),
            
            # Readability
            'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0,
            'word_length_cv': self.calculate_cv(word_lengths),
            'syllable_ratio': self._syllable_ratio(words),
            'flesch_kincaid_grade': self._flesch_kincaid(words, sentences),
            'automated_readability_index': self._ari(text, words, sentences),
            
            # Repetition
            'bigram_repetition_rate': self._ngram_repetition(words, 2),
            'trigram_repetition_rate': self._ngram_repetition(words, 3),
            'sentence_similarity_avg': self._sentence_similarity(sentences),
            'word_repetition_local': self._local_word_repetition(words),
            'phrase_repetition_rate': self._phrase_repetition(text),
            
            # Punctuation
            'comma_rate': text.count(',') / max(len(sentences), 1),
            'semicolon_rate': text.count(';') / max(len(sentences), 1),
            'colon_rate': text.count(':') / max(len(sentences), 1),
            'question_rate': text.count('?') / max(len(sentences), 1),
            'exclamation_rate': text.count('!') / max(len(sentences), 1),
            'dash_rate': len(re.findall(r'[-–—]', text)) / max(len(sentences), 1),
            'parenthesis_rate': text.count('(') / max(len(sentences), 1),
            'ellipsis_count': len(re.findall(r'\.{3}|…', text)),
            
            # Structure
            'paragraph_count': len(paragraphs),
            'avg_paragraph_length': float(np.mean(para_word_counts)) if para_word_counts else 0,
            'paragraph_length_cv': self.calculate_cv(para_word_counts),
            'has_greeting': 1.0 if re.search(r'^(dear|hello|hi|hey|greetings)\b', text_lower) else 0.0,
            'has_closing': 1.0 if re.search(r'(sincerely|regards|thanks?|cheers|best)\s*[,.]?\s*$', text_lower) else 0.0,
            'has_signature': 1.0 if re.search(r'\n[-–]\s*\w+\s*$', text) else 0.0,
            
            # Pronouns
            'first_person_singular_rate': len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / len(words) if words else 0,
            'first_person_plural_rate': len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.I)) / len(words) if words else 0,
            'second_person_rate': len(re.findall(r'\b(you|your|yours|yourself)\b', text, re.I)) / len(words) if words else 0,
            'third_person_rate': len(re.findall(r'\b(he|she|it|him|her|his|hers|its|they|them|their)\b', text, re.I)) / len(words) if words else 0,
            'pronoun_variety': self._pronoun_variety(text),
            
            # Discourse markers
            'transition_word_count': len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently|nevertheless|thus|hence|meanwhile|subsequently)\b', text, re.I)),
            'transition_density': len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently)\b', text, re.I)) / max(len(sentences), 1),
            'causal_connector_count': len(re.findall(r'\b(because|since|therefore|thus|hence|so|consequently)\b', text, re.I)),
            'contrast_connector_count': len(re.findall(r'\b(but|however|although|though|yet|nevertheless|whereas)\b', text, re.I)),
            'addition_connector_count': len(re.findall(r'\b(and|also|additionally|furthermore|moreover|plus)\b', text, re.I)),
            
            # Domain signals
            'is_academic_style': self._detect_academic_style(text),
            'is_casual_style': 1.0 if casual_count > 2 else 0.0,
            'is_speech_style': 1.0 if student_count > 0 or re.search(r'(dear\s+delegates?|thank\s+you\s+for)', text_lower) else 0.0,
            'is_technical_style': self._detect_technical_style(text),
            'proper_noun_ratio': len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / len(words) if words else 0,
            'citation_count': len(re.findall(r'\([^)]*\d{4}[^)]*\)|\[\d+\]', text)),
            'url_count': len(re.findall(r'https?://\S+', text)),
            'date_mention_count': len(re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|\b\d{4}\b', text)),
            
            # Strong human indicators
            'formal_speech_strength': self._formal_speech_strength(text, student_count),
            'human_authenticity_score': self._human_authenticity_score(
                student_count, uncertainty_count, esl_count, run_on_count, casual_count
            ),
            
            # Typos and errors
            'typo_density': self.detect_typos(words),
            'spelling_inconsistency': self._spelling_inconsistency(text),
            'contraction_rate': len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)) / len(words) if words else 0,
            'contraction_consistency': self._contraction_consistency(text),
            'capitalization_errors': self._capitalization_errors(text),
        }
        
        return features
    
    def _calculate_zipf_slope(self, word_counts: Counter) -> float:
        """Calculate Zipf's law slope."""
        if len(word_counts) < 5:
            return -1.0
        frequencies = sorted(word_counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1)
        log_ranks = np.log(ranks)
        log_freqs = np.log(frequencies)
        slope, _ = np.polyfit(log_ranks, log_freqs, 1)
        return float(slope)
    
    def _calculate_burstiness(self, values: List[float]) -> float:
        """Calculate burstiness coefficient."""
        if len(values) < 2:
            return 0.0
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        if mean + std == 0:
            return 0.0
        return float((std - mean) / (std + mean))
    
    def _syllable_ratio(self, words: List[str]) -> float:
        """Calculate average syllables per word."""
        if not words:
            return 0.0
        
        def count_syllables(word):
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
        
        return sum(count_syllables(w) for w in words) / len(words)
    
    def _flesch_kincaid(self, words: List[str], sentences: List[str]) -> float:
        """Calculate Flesch-Kincaid grade level."""
        if not words or not sentences:
            return 0.0
        syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w, re.I))) for w in words)
        return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
    
    def _ari(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calculate Automated Readability Index."""
        if not words or not sentences:
            return 0.0
        chars = len(re.sub(r'\s', '', text))
        return 4.71 * (chars / len(words)) + 0.5 * (len(words) / len(sentences)) - 21.43
    
    def _ngram_repetition(self, words: List[str], n: int) -> float:
        """Calculate n-gram repetition rate."""
        if len(words) < n + 1:
            return 0.0
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        counts = Counter(ngrams)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts) if counts else 0.0
    
    def _sentence_similarity(self, sentences: List[str]) -> float:
        """Calculate average Jaccard similarity between sentences."""
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
    
    def _local_word_repetition(self, words: List[str]) -> float:
        """Detect repeated words within sliding window."""
        if len(words) < 10:
            return 0.0
        repetitions = 0
        window_size = 10
        if len(words) <= window_size:
            return 0.0
        for i in range(len(words) - window_size):
            window = words[i:i + window_size]
            if len(window) != len(set(window)):
                repetitions += 1
        return repetitions / (len(words) - window_size)
    
    def _phrase_repetition(self, text: str) -> float:
        """Detect repeated 3+ word phrases."""
        words = text.lower().split()
        if len(words) < 10:
            return 0.0
        phrases = [' '.join(words[i:i+3]) for i in range(len(words) - 2)]
        counts = Counter(phrases)
        repeated = sum(1 for c in counts.values() if c > 1)
        return repeated / len(counts) if counts else 0.0
    
    def _pronoun_variety(self, text: str) -> float:
        """Calculate pronoun variety score."""
        pronouns = ['i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 
                    'he', 'she', 'it', 'him', 'her', 'his', 'they', 'them', 'their']
        found = set()
        for pronoun in pronouns:
            if re.search(rf'\b{pronoun}\b', text, re.I):
                found.add(pronoun)
        return len(found) / len(pronouns)
    
    def _detect_academic_style(self, text: str) -> float:
        """Detect academic writing style."""
        academic_markers = [
            r'\b(furthermore|moreover|consequently|nevertheless)\b',
            r'\b(hypothesis|methodology|analysis|findings|conclusion)\b',
            r'\([^)]*\d{4}[^)]*\)',  # Citations
            r'\baccording\s+to\b',
            r'\bthe\s+study\b',
        ]
        count = sum(len(re.findall(p, text, re.I)) for p in academic_markers)
        return min(1.0, count / 5)
    
    def _detect_technical_style(self, text: str) -> float:
        """Detect technical writing style."""
        technical_markers = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\d+(\.\d+)?\s*(MB|GB|KB|ms|Hz|GHz)\b',  # Units
            r'```',  # Code blocks
            r'\bfunction\b|\bclass\b|\bdef\b',  # Code keywords
        ]
        count = sum(len(re.findall(p, text)) for p in technical_markers)
        return min(1.0, count / 5)
    
    def _spelling_inconsistency(self, text: str) -> float:
        """Detect spelling inconsistencies (color vs colour)."""
        variants = [
            (r'\bcolor\b', r'\bcolour\b'),
            (r'\borganize\b', r'\borganise\b'),
            (r'\brealize\b', r'\brealise\b'),
            (r'\bcenter\b', r'\bcentre\b'),
        ]
        inconsistencies = 0
        for us, uk in variants:
            has_us = bool(re.search(us, text, re.I))
            has_uk = bool(re.search(uk, text, re.I))
            if has_us and has_uk:
                inconsistencies += 1
        return inconsistencies
    
    def _contraction_consistency(self, text: str) -> float:
        """Check if contractions are used consistently."""
        contractions = len(re.findall(r"\b\w+'(t|re|ve|ll|d)\b", text))
        full_forms = len(re.findall(r'\b(do not|does not|will not|would not|can not|have not|has not|is not|are not|was not|were not)\b', text, re.I))
        
        if contractions + full_forms == 0:
            return 1.0  # No data
        
        # Inconsistent = both present
        if contractions > 0 and full_forms > 0:
            return 0.5
        return 1.0
    
    def _capitalization_errors(self, text: str) -> float:
        """Detect capitalization errors."""
        errors = 0
        # Sentence not starting with capital
        errors += len(re.findall(r'[.!?]\s+[a-z]', text))
        # Random capitals mid-word
        errors += len(re.findall(r'\b[a-z]+[A-Z][a-z]+\b', text))
        # "i" not capitalized
        errors += len(re.findall(r'\bi\b(?![.])', text))
        
        sentences = self.split_sentences(text)
        return errors / max(len(sentences), 1)
    
    def _formal_speech_strength(self, text: str, student_count: int) -> float:
        """
        Calculate formal speech strength indicator.
        High values strongly indicate human-written formal speeches (MUN, debate, etc.)
        """
        text_lower = text.lower()
        strength = 0.0
        
        # Direct student markers - each one adds to strength
        strength += student_count * 0.15  # 10 markers = 1.5 base
        
        # Specific MUN/formal speech patterns (weighted heavily)
        mun_patterns = [
            (r'\b(united\s+nations?|un|unhcr)\b', 0.3),
            (r'\b(delegates?|committee|council)\b', 0.2),
            (r'\b(representing|delegation\s+of)\b', 0.25),
            (r'\bhello\s+delegates\b', 0.4),
            (r'\bdistinguished\s+delegates\b', 0.4),
            (r'\bmy\s+name\s+is\b', 0.3),
            (r'\btoday\s+I\s+will\b', 0.3),
            (r'\bour\s+(country|nation)\b', 0.2),
            (r'\bfirst[,.].*second[,.].*third\b', 0.3),  # Structured proposals
            (r'\bproposes?\s+(several|the\s+following)\b', 0.25),
        ]
        
        for pattern, weight in mun_patterns:
            if re.search(pattern, text, re.I):
                strength += weight
        
        # Cap at 3.0 (very strong indicator)
        return min(strength, 3.0)
    
    def _human_authenticity_score(self, student_count: int, uncertainty_count: int, 
                                   esl_count: int, run_on_count: int, casual_count: int) -> float:
        """
        Calculate overall human authenticity score.
        Combines multiple signals that indicate human writing.
        """
        score = 0.0
        
        # Student markers are strong indicators
        if student_count >= 5:
            score += 1.0
        elif student_count >= 2:
            score += 0.5
        elif student_count >= 1:
            score += 0.25
        
        # Uncertainty markers
        score += uncertainty_count * 0.1
        
        # ESL patterns
        score += esl_count * 0.15
        
        # Run-on sentences (human error)
        score += run_on_count * 0.1
        
        # Casual markers
        score += casual_count * 0.1
        
        return min(score, 2.0)  # Cap at 2.0
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """Extract features as numpy array."""
        features = self.extract_features(text)
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names.copy()


if __name__ == '__main__':
    # Test the extractor
    extractor = FeatureExtractorV3()
    
    # Test with ESL-like text
    test_text = """
    Dear delegates and chair,
    I will be discussing the problem of statelessness. Now going more in depth, 
    stateless people they are people that do not fit in anywhere because of the lack of 
    a true identity. Because stateless people are not part of one concrete country they 
    go through very hard times and face many struggles like the lack of education.
    All these problems combined lead to a major difficulty to live day to day.
    Lebanon specifically has a very bad political situation and a lot of money problems 
    so it's a very hard problem to solve. Thank you for your time.
    """
    
    features = extractor.extract_features(test_text)
    print("Extracted features (top 20):")
    print("=" * 60)
    
    # Show authenticity signals
    print("\n=== AUTHENTICITY SIGNALS (Human indicators) ===")
    auth_features = ['esl_pattern_count', 'student_marker_count', 'uncertainty_marker_count',
                     'formality_inconsistency', 'run_on_sentence_count', 'casual_marker_count']
    for name in auth_features:
        print(f"  {name}: {features[name]:.4f}")
    
    print("\n=== AI-TYPICAL SIGNALS ===")
    ai_features = ['ai_phrase_count', 'ai_phrase_density', 'perfect_parallelism_score']
    for name in ai_features:
        print(f"  {name}: {features[name]:.4f}")
    
    print("\n=== SENTENCE VARIANCE ===")
    var_features = ['sentence_length_cv', 'sentence_length_range', 'sentence_length_entropy']
    for name in var_features:
        print(f"  {name}: {features[name]:.4f}")
    
    print(f"\nTotal features: {len(features)}")
