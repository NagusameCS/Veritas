#!/usr/bin/env python3
"""
Veritas Feature Extractor V2 - Humanization Detection Features
===============================================================
Designed specifically to detect:
1. Pure Human text
2. Raw AI text  
3. Humanized AI text (AI processed through bypass tools)

Feature Categories:
- Basic Statistical (word/sentence patterns)
- AI Signature Detection (phrases, formality, structure)
- Humanization Artifact Detection (contractions, typos, disfluencies)
- Consistency Analysis (register mixing, style shifts)
- Combination Pattern Detection (suspicious co-occurrences)
"""

import re
import math
import string
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np


class FeatureExtractorV2:
    """
    Enhanced feature extractor with 80+ features for 3-class detection.
    """
    
    def __init__(self):
        self._init_patterns()
        self._init_word_lists()
        self._feature_names = None
    
    def _init_patterns(self):
        """Initialize regex patterns for feature extraction"""
        
        # Sentence splitting
        self.sentence_pattern = re.compile(r'[.!?]+\s+|[.!?]+$')
        
        # Contractions
        self.contraction_pattern = re.compile(
            r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|"
            r"we're|we've|we'll|we'd|they're|they've|they'll|they'd|"
            r"isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|won't|wouldn't|"
            r"don't|doesn't|didn't|can't|couldn't|shouldn't|mightn't|mustn't|"
            r"let's|that's|who's|what's|where's|there's|here's|"
            r"could've|would've|should've|might've|must've|"
            r"gonna|wanna|gotta|kinda|sorta|outta|'cause|'bout)\b",
            re.IGNORECASE
        )
        
        # Informal contractions (more casual)
        self.informal_contraction_pattern = re.compile(
            r"\b(gonna|wanna|gotta|kinda|sorta|outta|'cause|'bout|dunno|"
            r"gimme|lemme|coulda|woulda|shoulda|mighta|oughta)\b",
            re.IGNORECASE
        )
        
        # Disfluencies
        self.disfluency_pattern = re.compile(
            r"\b(well|so|um|uh|hmm|like|basically|honestly|actually|"
            r"anyway|anyhow|I mean|you know|I guess|I think|I suppose|"
            r"kind of|sort of|more or less|in a way)\b",
            re.IGNORECASE
        )
        
        # Sentence-start disfluencies (humanizers often add these)
        self.sentence_start_disfluency = re.compile(
            r'^(Well,|So,|I mean,|You know,|Like,|Basically,|Honestly,|'
            r'Actually,|Look,|See,|Thing is,|Okay so,|Right,|Anyway,)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # AI typical phrases
        self.ai_phrases = [
            "it is important to note", "it's important to note",
            "it is worth noting", "it's worth noting",
            "it is essential to", "it's essential to",
            "furthermore", "moreover", "additionally",
            "in conclusion", "to summarize", "in summary",
            "this demonstrates", "this illustrates", "this shows that",
            "it can be observed", "one must consider",
            "plays a pivotal role", "plays a crucial role",
            "in today's world", "in the modern era",
            "a myriad of", "a plethora of", "a multitude of",
            "it is crucial to understand", "it is imperative",
            "the importance of", "the significance of",
            "delve into", "delve deeper", "dive into",
            "comprehensive understanding", "thorough analysis",
            "in the realm of", "in the context of",
            "subsequently", "consequently", "henceforth",
            "notwithstanding", "nevertheless", "nonetheless",
        ]
        
        # Formal/academic words AI overuses
        self.formal_words = [
            "utilize", "facilitate", "implement", "comprehensive",
            "significant", "demonstrate", "indicate", "require",
            "provide", "obtain", "sufficient", "approximately",
            "primarily", "frequently", "occasionally", "immediately",
            "subsequently", "consequently", "furthermore", "moreover",
            "therefore", "hence", "thus", "thereby", "wherein",
            "whereas", "insofar", "notwithstanding", "aforementioned",
            "henceforth", "heretofore", "therein", "thereof",
        ]
        
        # Hedging language (humans use naturally, humanizers inject)
        self.hedging_words = [
            "probably", "maybe", "perhaps", "possibly", "likely",
            "might", "could", "seems", "appears", "apparently",
            "supposedly", "arguably", "presumably", "sort of",
            "kind of", "more or less", "in a way", "to some extent",
        ]
        
        # Emphatics
        self.emphatics = [
            "seriously", "literally", "absolutely", "totally",
            "completely", "definitely", "certainly", "obviously",
            "clearly", "honestly", "frankly", "really", "truly",
            "genuinely", "actually", "for real", "no joke",
        ]
        
        # Rhetorical questions patterns
        self.rhetorical_pattern = re.compile(
            r'\b(right\?|you know\?|make sense\?|get it\?|'
            r'don\'t you think\?|wouldn\'t you agree\?|isn\'t it\?|'
            r'crazy,? right\?|wild,? huh\?|fair enough\?)\s*',
            re.IGNORECASE
        )
        
        # Common typos (humanizers inject these)
        self.common_typos = [
            "teh", "hte", "adn", "nad", "taht", "tht", "ahve", "hvae",
            "wiht", "wtih", "tihs", "thsi", "form", "fomr", "tehy", "thye",
            "bene", "thier", "ther", "woudl", "owuld", "abuot", "abotu",
            "coudl", "cuold", "whcih", "wich", "theer", "tehre", "wehre",
            "becuase", "beacuse", "somethign", "soemthing", "realy", "raelly",
            "definately", "definitly", "porbably", "probabl", "diferent",
            "diffrent", "trough", "thru", "acutally", "actualy",
            "rly", "def", "prob", "bc", "b/c", "smth", "sth", "tmrw", "tmr",
        ]
        
        # Slang/casual words
        self.slang_words = [
            "cool", "awesome", "dope", "sick", "fire", "lit", "legit",
            "vibe", "vibes", "mood", "lowkey", "highkey", "lowkey",
            "cap", "no cap", "bet", "slay", "stan", "flex", "sus",
            "bro", "bruh", "dude", "fam", "tbh", "ngl", "imo", "imho",
            "lol", "lmao", "omg", "wtf", "idk", "ikr", "smh", "fyi",
        ]
        
        # Sentence fragments humans use
        self.fragments = [
            "crazy.", "wild.", "insane.", "nuts.", "ridiculous.",
            "absolutely.", "definitely.", "obviously.", "clearly.",
            "for sure.", "no doubt.", "big time.", "huge.",
            "makes sense.", "fair point.", "good question.",
            "my bad.", "no way.", "for real.", "no joke.",
            "seriously.", "honestly.", "basically.",
        ]
        
        # Punctuation patterns
        self.ellipsis_pattern = re.compile(r'\.{2,}')
        self.em_dash_pattern = re.compile(r'—|--')
        self.double_punct_pattern = re.compile(r'[!?]{2,}')
        
    def _init_word_lists(self):
        """Initialize word frequency lists"""
        # Common English words (for vocabulary analysis)
        self.common_words = set([
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
        ])
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_words(self, text: str) -> List[str]:
        """Extract words from text"""
        return re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BASIC STATISTICAL FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_basic_stats(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Basic statistical features"""
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sentence_count'] = max(len(sentences), 1)
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = len(words) / max(len(sentences), 1)
        
        # Vocabulary richness
        unique_words = set(words)
        features['unique_word_ratio'] = len(unique_words) / max(len(words), 1)
        features['hapax_ratio'] = sum(1 for w, c in Counter(words).items() if c == 1) / max(len(words), 1)
        
        # Sentence length variance (burstiness)
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['sentence_length_range'] = max(sent_lengths) - min(sent_lengths) if sent_lengths else 0
        
        # Burstiness coefficient (human text has high variance)
        mean_len = np.mean(sent_lengths) if sent_lengths else 1
        features['burstiness_coef'] = features['sentence_length_std'] / mean_len if mean_len > 0 else 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AI SIGNATURE DETECTION FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_ai_signatures(self, text: str, words: List[str]) -> Dict[str, float]:
        """Features that detect AI-generated text patterns"""
        features = {}
        text_lower = text.lower()
        word_count = max(len(words), 1)
        
        # AI phrase detection
        ai_phrase_count = sum(1 for phrase in self.ai_phrases if phrase in text_lower)
        features['ai_phrase_count'] = ai_phrase_count
        features['ai_phrase_density'] = ai_phrase_count / word_count * 100
        
        # Formal word usage
        formal_count = sum(1 for w in words if w in self.formal_words)
        features['formal_word_count'] = formal_count
        features['formal_word_density'] = formal_count / word_count * 100
        
        # Transition word patterns (AI overuses these)
        transitions = ["furthermore", "moreover", "additionally", "however", 
                      "therefore", "consequently", "nevertheless", "nonetheless"]
        transition_count = sum(1 for w in words if w in transitions)
        features['transition_density'] = transition_count / word_count * 100
        
        # Sentence structure uniformity (AI tends to be more uniform)
        sentences = self._split_sentences(text)
        if len(sentences) > 2:
            starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
            unique_starts = len(set(starts))
            features['sentence_start_variety'] = unique_starts / len(sentences)
        else:
            features['sentence_start_variety'] = 1.0
        
        # Passive voice indicators
        passive_patterns = ["is being", "was being", "has been", "have been", 
                          "had been", "will be", "is done", "was done", "are done"]
        passive_count = sum(1 for p in passive_patterns if p in text_lower)
        features['passive_voice_density'] = passive_count / word_count * 100
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HUMANIZATION ARTIFACT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_humanization_artifacts(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Features that detect humanization tool artifacts"""
        features = {}
        word_count = max(len(words), 1)
        sent_count = max(len(sentences), 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # CONTRACTION ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        all_contractions = self.contraction_pattern.findall(text)
        informal_contractions = self.informal_contraction_pattern.findall(text)
        
        features['contraction_count'] = len(all_contractions)
        features['contraction_density'] = len(all_contractions) / word_count * 100
        features['informal_contraction_count'] = len(informal_contractions)
        features['informal_contraction_density'] = len(informal_contractions) / word_count * 100
        
        # Contraction uniformity (humanizers distribute evenly, humans cluster)
        if len(sentences) > 2:
            contractions_per_sent = []
            for sent in sentences:
                c = len(self.contraction_pattern.findall(sent))
                contractions_per_sent.append(c)
            features['contraction_distribution_std'] = np.std(contractions_per_sent)
            # Low std = humanizer (uniform), high std = human (clustered)
        else:
            features['contraction_distribution_std'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # DISFLUENCY ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        disfluencies = self.disfluency_pattern.findall(text)
        features['disfluency_count'] = len(disfluencies)
        features['disfluency_density'] = len(disfluencies) / word_count * 100
        
        # Sentence-start disfluencies (humanizers love to add these)
        start_disfluencies = self.sentence_start_disfluency.findall(text)
        features['sentence_start_disfluency_count'] = len(start_disfluencies)
        features['sentence_start_disfluency_ratio'] = len(start_disfluencies) / sent_count
        
        # Disfluency variety (humanizers use limited set repeatedly)
        unique_disfluencies = set([d.lower() for d in disfluencies])
        features['disfluency_variety'] = len(unique_disfluencies) / max(len(disfluencies), 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # TYPO DETECTION
        # ─────────────────────────────────────────────────────────────────────
        typo_count = sum(1 for w in words if w in self.common_typos)
        features['typo_count'] = typo_count
        features['typo_density'] = typo_count / word_count * 100
        
        # Typo clustering (humanizers distribute randomly, humans cluster near each other)
        if typo_count > 0:
            typo_positions = [i for i, w in enumerate(words) if w in self.common_typos]
            if len(typo_positions) > 1:
                typo_gaps = [typo_positions[i+1] - typo_positions[i] for i in range(len(typo_positions)-1)]
                features['typo_position_variance'] = np.std(typo_gaps)
            else:
                features['typo_position_variance'] = 0
        else:
            features['typo_position_variance'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # HEDGING ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        hedges = [w for w in words if w in self.hedging_words]
        features['hedging_count'] = len(hedges)
        features['hedging_density'] = len(hedges) / word_count * 100
        
        # ─────────────────────────────────────────────────────────────────────
        # EMPHATIC ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        emphatics = [w for w in words if w in self.emphatics]
        features['emphatic_count'] = len(emphatics)
        features['emphatic_density'] = len(emphatics) / word_count * 100
        
        # ─────────────────────────────────────────────────────────────────────
        # RHETORICAL QUESTIONS
        # ─────────────────────────────────────────────────────────────────────
        rhetorical = self.rhetorical_pattern.findall(text)
        features['rhetorical_question_count'] = len(rhetorical)
        features['rhetorical_question_density'] = len(rhetorical) / sent_count
        
        # ─────────────────────────────────────────────────────────────────────
        # SLANG DETECTION
        # ─────────────────────────────────────────────────────────────────────
        slang = [w for w in words if w in self.slang_words]
        features['slang_count'] = len(slang)
        features['slang_density'] = len(slang) / word_count * 100
        
        # ─────────────────────────────────────────────────────────────────────
        # FRAGMENT DETECTION
        # ─────────────────────────────────────────────────────────────────────
        text_lower = text.lower()
        fragment_count = sum(1 for f in self.fragments if f in text_lower)
        features['fragment_count'] = fragment_count
        features['fragment_density'] = fragment_count / sent_count
        
        # ─────────────────────────────────────────────────────────────────────
        # PUNCTUATION ANOMALIES
        # ─────────────────────────────────────────────────────────────────────
        features['ellipsis_count'] = len(self.ellipsis_pattern.findall(text))
        features['em_dash_count'] = len(self.em_dash_pattern.findall(text))
        features['double_punct_count'] = len(self.double_punct_pattern.findall(text))
        features['exclamation_count'] = text.count('!')
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSISTENCY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_consistency_features(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Features detecting inconsistency (key humanization tell)"""
        features = {}
        word_count = max(len(words), 1)
        
        # ─────────────────────────────────────────────────────────────────────
        # REGISTER MIXING (humanizers mix formal + casual unnaturally)
        # ─────────────────────────────────────────────────────────────────────
        formal_count = sum(1 for w in words if w in self.formal_words)
        slang_count = sum(1 for w in words if w in self.slang_words)
        
        # High values of both = likely humanized
        features['formal_slang_product'] = (formal_count * slang_count) / word_count
        
        # Register inconsistency score
        formal_ratio = formal_count / word_count
        slang_ratio = slang_count / word_count
        features['register_inconsistency'] = abs(formal_ratio - slang_ratio) if (formal_count + slang_count) > 0 else 0
        
        # ─────────────────────────────────────────────────────────────────────
        # STYLE SHIFT DETECTION
        # ─────────────────────────────────────────────────────────────────────
        if len(sentences) >= 4:
            # Compare first half vs second half
            mid = len(sentences) // 2
            first_half = " ".join(sentences[:mid])
            second_half = " ".join(sentences[mid:])
            
            first_words = self._get_words(first_half)
            second_words = self._get_words(second_half)
            
            # Formality shift
            first_formal = sum(1 for w in first_words if w in self.formal_words) / max(len(first_words), 1)
            second_formal = sum(1 for w in second_words if w in self.formal_words) / max(len(second_words), 1)
            features['formality_shift'] = abs(first_formal - second_formal)
            
            # Contraction shift
            first_contr = len(self.contraction_pattern.findall(first_half)) / max(len(first_words), 1)
            second_contr = len(self.contraction_pattern.findall(second_half)) / max(len(second_words), 1)
            features['contraction_shift'] = abs(first_contr - second_contr)
        else:
            features['formality_shift'] = 0
            features['contraction_shift'] = 0
        
        # ─────────────────────────────────────────────────────────────────────
        # AI + HUMAN MARKER CO-OCCURRENCE (suspicious combinations)
        # ─────────────────────────────────────────────────────────────────────
        ai_marker_count = sum(1 for p in self.ai_phrases if p in text.lower())
        human_marker_count = len(self.disfluency_pattern.findall(text)) + len(self.informal_contraction_pattern.findall(text))
        
        # If both AI and human markers present in high amounts = humanized
        features['ai_human_cooccurrence'] = (ai_marker_count * human_marker_count) / word_count
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINATION PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_combination_features(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Detect suspicious combinations that indicate humanization"""
        features = {}
        word_count = max(len(words), 1)
        text_lower = text.lower()
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 1: Formal structure + Casual language
        # ─────────────────────────────────────────────────────────────────────
        has_formal_structure = any(p in text_lower for p in ["furthermore", "moreover", "additionally", "in conclusion"])
        has_casual_language = len(self.informal_contraction_pattern.findall(text)) > 0 or len([w for w in words if w in self.slang_words]) > 0
        features['formal_structure_casual_language'] = 1.0 if (has_formal_structure and has_casual_language) else 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 2: Technical vocabulary + Typos
        # ─────────────────────────────────────────────────────────────────────
        technical_count = sum(1 for w in words if w in self.formal_words)
        typo_count = sum(1 for w in words if w in self.common_typos)
        features['technical_with_typos'] = 1.0 if (technical_count >= 3 and typo_count >= 1) else 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 3: Long sentences + Many disfluencies
        # ─────────────────────────────────────────────────────────────────────
        avg_sent_len = len(words) / max(len(sentences), 1)
        disfluency_count = len(self.disfluency_pattern.findall(text))
        features['long_sentences_with_disfluencies'] = 1.0 if (avg_sent_len > 20 and disfluency_count >= 3) else 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 4: Perfect grammar + Deliberate typos
        # ─────────────────────────────────────────────────────────────────────
        # (Low grammar errors but has typo list words = suspicious)
        features['deliberate_typo_score'] = typo_count / word_count * 100 if typo_count > 0 else 0
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 5: Uniform hedging distribution
        # ─────────────────────────────────────────────────────────────────────
        if len(sentences) > 3:
            hedges_per_sent = []
            for sent in sentences:
                sent_words = self._get_words(sent)
                h = sum(1 for w in sent_words if w in self.hedging_words)
                hedges_per_sent.append(h)
            hedge_uniformity = 1 - (np.std(hedges_per_sent) / (np.mean(hedges_per_sent) + 0.001))
            features['hedge_uniformity'] = max(0, min(1, hedge_uniformity))
        else:
            features['hedge_uniformity'] = 0.5
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 6: Rhetorical questions with formal text
        # ─────────────────────────────────────────────────────────────────────
        rhetorical_count = len(self.rhetorical_pattern.findall(text))
        formal_count = sum(1 for w in words if w in self.formal_words)
        features['rhetorical_formal_combo'] = 1.0 if (rhetorical_count >= 2 and formal_count >= 3) else 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # PATTERN 7: High contraction density + AI phrases remaining
        # ─────────────────────────────────────────────────────────────────────
        contraction_count = len(self.contraction_pattern.findall(text))
        ai_phrase_count = sum(1 for p in self.ai_phrases if p in text_lower)
        features['contractions_with_ai_phrases'] = 1.0 if (contraction_count >= 5 and ai_phrase_count >= 1) else 0.0
        
        # ─────────────────────────────────────────────────────────────────────
        # COMPOSITE HUMANIZATION SCORE
        # ─────────────────────────────────────────────────────────────────────
        humanization_signals = [
            features['formal_structure_casual_language'],
            features['technical_with_typos'],
            features['long_sentences_with_disfluencies'],
            min(features['deliberate_typo_score'] / 2, 1),
            features['rhetorical_formal_combo'],
            features['contractions_with_ai_phrases'],
        ]
        features['composite_humanization_score'] = sum(humanization_signals) / len(humanization_signals)
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERPLEXITY-LIKE FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_perplexity_features(self, text: str, words: List[str]) -> Dict[str, float]:
        """Features approximating perplexity patterns"""
        features = {}
        
        if len(words) < 5:
            features['word_length_variance'] = 0
            features['word_rarity_variance'] = 0
            features['bigram_repeat_ratio'] = 0
            return features
        
        # Word length variance (proxy for vocabulary diversity)
        word_lengths = [len(w) for w in words]
        features['word_length_variance'] = np.var(word_lengths)
        
        # Word commonality variance
        common_scores = [1 if w in self.common_words else 0 for w in words]
        features['word_rarity_variance'] = np.var(common_scores)
        
        # Bigram repetition (AI often repeats patterns)
        if len(words) > 2:
            bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
            bigram_counts = Counter(bigrams)
            repeated = sum(1 for c in bigram_counts.values() if c > 1)
            features['bigram_repeat_ratio'] = repeated / len(bigrams)
        else:
            features['bigram_repeat_ratio'] = 0
        
        # Trigram repetition
        if len(words) > 3:
            trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
            trigram_counts = Counter(trigrams)
            repeated = sum(1 for c in trigram_counts.values() if c > 1)
            features['trigram_repeat_ratio'] = repeated / len(trigrams)
        else:
            features['trigram_repeat_ratio'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN EXTRACTION METHOD
    # ═══════════════════════════════════════════════════════════════════════════
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all features from text"""
        if not text or len(text.strip()) < 10:
            return {name: 0.0 for name in self.get_feature_names()}
        
        # Preprocess
        words = self._get_words(text)
        sentences = self._split_sentences(text)
        
        # Extract all feature groups
        features = {}
        features.update(self._extract_basic_stats(text, words, sentences))
        features.update(self._extract_ai_signatures(text, words))
        features.update(self._extract_humanization_artifacts(text, words, sentences))
        features.update(self._extract_consistency_features(text, words, sentences))
        features.update(self._extract_combination_features(text, words, sentences))
        features.update(self._extract_perplexity_features(text, words))
        
        # Handle NaN/Inf
        for key in features:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        return features
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """Extract features as numpy array"""
        features = self.extract_features(text)
        names = self.get_feature_names()
        return np.array([features.get(name, 0.0) for name in names])
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        if self._feature_names is None:
            # Extract from dummy text to get all names
            dummy_features = self.extract_features("This is a sample text with enough words to extract all features properly.")
            self._feature_names = sorted(dummy_features.keys())
        return self._feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    extractor = FeatureExtractorV2()
    
    # Test samples
    human_text = """
    So I was thinking about this the other day, right? Like, honestly I don't 
    really get why everyone's so obsessed with AI these days. Sure, it's cool 
    and all, but come on. We've been hearing about "the future" for years now.
    
    My friend Dave - you know Dave - he was saying the exact same thing last 
    week. Wild, huh? Anyway, I guess we'll just have to wait and see what happens.
    """
    
    ai_text = """
    It is important to note that artificial intelligence has demonstrated 
    significant capabilities in various domains. The implementation of these 
    systems requires comprehensive understanding of the underlying mechanisms.
    Furthermore, one must consider the ethical implications that subsequently 
    arise from such technological advancements. In conclusion, the utilization 
    of AI will continue to facilitate numerous applications in the foreseeable 
    future.
    """
    
    humanized_text = """
    So, it's worth noting that AI has shown some pretty significant capabilities 
    in different areas. Thing is, implementing these systems kinda requires you 
    to understand the underlying mechanisms, you know? Plus, you gotta consider 
    the ethical implications that come up from tech advancements like this. 
    Basically, AI's gonna keep helping out with tons of applications going forward.
    Right? Wild stuff, honestly.
    """
    
    print("=" * 70)
    print("FEATURE EXTRACTOR V2 - HUMANIZATION DETECTION")
    print("=" * 70)
    print(f"Total features: {len(extractor.get_feature_names())}")
    
    for name, text in [("HUMAN", human_text), ("AI", ai_text), ("HUMANIZED", humanized_text)]:
        features = extractor.extract_features(text)
        print(f"\n{'=' * 70}")
        print(f"{name} TEXT ANALYSIS:")
        print(f"{'=' * 70}")
        
        # Key differentiating features
        key_features = [
            'ai_phrase_count', 'ai_phrase_density',
            'contraction_count', 'contraction_density',
            'disfluency_count', 'disfluency_density',
            'sentence_start_disfluency_ratio',
            'typo_count', 'formal_word_density',
            'slang_density', 'register_inconsistency',
            'ai_human_cooccurrence', 'composite_humanization_score',
            'formal_structure_casual_language',
            'contractions_with_ai_phrases',
        ]
        
        for feat in key_features:
            if feat in features:
                print(f"  {feat:40} {features[feat]:.4f}")
