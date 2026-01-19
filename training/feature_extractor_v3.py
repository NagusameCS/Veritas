#!/usr/bin/env python3
"""
Veritas Feature Extractor V3 - Ultimate Detection Features
============================================================
Designed for maximum discrimination between:
1. Pure Human text
2. Raw AI text  
3. Humanized AI text (AI processed through bypass tools)
4. Partially humanized text (mixed segments)

NEW in V3:
- Neural embeddings via sentence transformers
- Perplexity scoring from language models
- Watermark detection patterns
- Semantic coherence analysis
- Null combination patterns (A & !B detection)
- Partial humanization detection
- 100+ features for maximum discrimination

Feature Categories:
1. Basic Statistical (word/sentence patterns)
2. AI Signature Detection (phrases, formality, structure)
3. Humanization Artifact Detection (contractions, typos, disfluencies)
4. Consistency Analysis (register mixing, style shifts)
5. Combination Pattern Detection (suspicious co-occurrences)
6. Null Combination Patterns (A & !B logic)
7. Neural Embedding Features (transformer-based)
8. Perplexity Features (language model based)
9. Watermark Detection (AI-specific patterns)
10. Semantic Coherence Analysis (deep coherence)
11. Partial Humanization Detection (segment analysis)
"""

import re
import math
import string
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Optional imports for neural features
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import torch
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class FeatureExtractorV3:
    """
    Ultimate feature extractor with 100+ features for 3-class detection.
    Includes neural embeddings, perplexity scoring, watermark detection,
    and null combination patterns.
    """
    
    def __init__(self, use_neural: bool = True, use_perplexity: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            use_neural: Whether to use neural embedding features (requires sentence-transformers)
            use_perplexity: Whether to use perplexity features (requires transformers)
        """
        self.use_neural = use_neural and HAS_SENTENCE_TRANSFORMERS
        self.use_perplexity = use_perplexity and HAS_TRANSFORMERS
        
        self._init_patterns()
        self._init_word_lists()
        self._init_models()
        self._feature_names = None
    
    def _init_models(self):
        """Initialize neural models if available"""
        self.embedding_model = None
        self.perplexity_model = None
        self.perplexity_tokenizer = None
        
        if self.use_neural:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                self.use_neural = False
        
        if self.use_perplexity:
            try:
                self.perplexity_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
                self.perplexity_model = GPT2LMHeadModel.from_pretrained('gpt2')
                self.perplexity_model.eval()
                if torch.cuda.is_available():
                    self.perplexity_model = self.perplexity_model.cuda()
            except Exception:
                self.use_perplexity = False
    
    def _init_patterns(self):
        """Initialize regex patterns for feature extraction"""
        
        # Sentence splitting
        self.sentence_pattern = re.compile(r'[.!?]+\s+|[.!?]+$')
        
        # Contractions - standard
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
        
        # Informal contractions
        self.informal_contraction_pattern = re.compile(
            r"\b(gonna|wanna|gotta|kinda|sorta|outta|'cause|'bout|dunno|"
            r"gimme|lemme|coulda|woulda|shoulda|mighta|oughta|y'all|ain't)\b",
            re.IGNORECASE
        )
        
        # Disfluencies
        self.disfluency_pattern = re.compile(
            r"\b(well|so|um|uh|hmm|like|basically|honestly|actually|"
            r"anyway|anyhow|I mean|you know|I guess|I think|I suppose|"
            r"kind of|sort of|more or less|in a way)\b",
            re.IGNORECASE
        )
        
        # Sentence-start disfluencies
        self.sentence_start_disfluency = re.compile(
            r'^(Well,|So,|I mean,|You know,|Like,|Basically,|Honestly,|'
            r'Actually,|Look,|See,|Thing is,|Okay so,|Right,|Anyway,)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Hedging language (humanizers add this)
        self.hedging_pattern = re.compile(
            r"\b(perhaps|maybe|possibly|probably|might|could be|"
            r"I think|I believe|in my opinion|it seems|apparently|"
            r"somewhat|fairly|rather|quite|sort of|kind of)\b",
            re.IGNORECASE
        )
        
        # Rhetorical questions (human-like)
        self.rhetorical_question = re.compile(
            r'\b(right\?|you know\?|isn\'t it\?|don\'t you think\?|wouldn\'t you say\?|'
            r'know what I mean\?|see what I mean\?|make sense\?)',
            re.IGNORECASE
        )
        
        # First-person references (more human)
        self.first_person = re.compile(
            r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b',
            re.IGNORECASE
        )
        
        # Second-person direct address (human conversational)
        self.second_person = re.compile(
            r'\b(you|your|yours|yourself|yourselves)\b',
            re.IGNORECASE
        )
        
        # Typo patterns (humanization artifacts - often too regular)
        self.typo_patterns = [
            (re.compile(r'\b(\w+)teh\b'), "teh typo"),
            (re.compile(r'\b(\w+)hte\b'), "hte typo"),
            (re.compile(r'\bteh\s'), "teh standalone"),
            (re.compile(r'\bthier\b'), "thier"),
            (re.compile(r'\brecieve\b'), "recieve"),
            (re.compile(r'\boccured\b'), "occured"),
            (re.compile(r'\bseperate\b'), "seperate"),
            (re.compile(r'\bdefinately\b'), "definately"),
            (re.compile(r'\baccommodate\b'), "accommodate"),
            (re.compile(r'\bwierd\b'), "wierd"),
        ]
        
        # Watermark patterns (AI-specific hidden markers)
        self.watermark_patterns = [
            # Unicode zero-width characters often used as watermarks
            re.compile(r'[\u200b\u200c\u200d\u2060\ufeff]'),
            # Repeated specific word patterns that may be watermarks
            re.compile(r'\b(indeed|moreover|furthermore|additionally)\s+,?\s*(indeed|moreover|furthermore|additionally)\b', re.IGNORECASE),
            # Unusual punctuation spacing that could be markers
            re.compile(r'\s{2,}[.!?]'),
            # Non-standard quotation marks
            re.compile(r'[""''„‚«»‹›]'),
        ]
        
        # AI phrase patterns
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
            "it should be noted", "it must be emphasized",
            "this is particularly", "this is especially",
            "a key aspect", "a crucial element",
            "multifaceted", "paradigm", "synergy",
            "leverage", "utilize", "optimize",
        ]
        
        # Formal transition words (AI overuses)
        self.formal_transitions = [
            "furthermore", "moreover", "additionally", "consequently",
            "subsequently", "nevertheless", "nonetheless", "however",
            "therefore", "thus", "hence", "accordingly", "meanwhile",
            "conversely", "alternatively", "similarly", "likewise",
        ]
        
        # Decorative unicode (AI-specific)
        self.decorative_unicode = re.compile(
            r'[★☆●○◆◇■□▪▫►◄▲△▼▽◈◊♦♢♠♤♣♧♥♡✓✔✗✘➔→←↑↓↔↕⇒⇐⇔'
            r'✦✧❖❤❥❣⚡☀☁☂☃☄★☆✡☪☯☸✡✝✞♈♉♊♋♌♍♎♏♐♑♒♓'
            r'│┃┄┅┆┇┈┉┊┋╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿'
            r'═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬]'
        )
    
    def _init_word_lists(self):
        """Initialize word lists for detection"""
        
        # Formal/academic words AI overuses
        self.formal_words = {
            "utilize", "facilitate", "implement", "comprehensive",
            "significant", "demonstrate", "indicate", "require",
            "essential", "crucial", "fundamental", "paramount",
            "subsequently", "furthermore", "moreover", "consequently",
            "aforementioned", "methodology", "paradigm", "framework",
            "leverage", "optimize", "streamline", "synergy",
            "multifaceted", "holistic", "nuanced", "intricate",
            "elucidate", "delineate", "articulate", "substantiate",
            "juxtapose", "encompass", "constitute", "exemplify",
        }
        
        # Common words humans prefer over formal equivalents
        self.informal_equivalents = {
            "utilize": "use",
            "facilitate": "help",
            "implement": "do",
            "demonstrate": "show",
            "indicate": "show",
            "require": "need",
            "sufficient": "enough",
            "subsequently": "then",
            "furthermore": "also",
            "moreover": "also",
            "commence": "start",
            "terminate": "end",
            "endeavor": "try",
            "acquire": "get",
            "comprehend": "understand",
        }
        
        # Sophisticated words (thesaurus substitution detection)
        self.sophisticated_words = {
            "ubiquitous", "ephemeral", "quintessential", "exacerbate",
            "ameliorate", "obfuscate", "pontificate", "ruminate",
            "cogitate", "deliberate", "contemplate", "speculate",
            "extrapolate", "interpolate", "correlate", "juxtapose",
        }
        
        # Common filler words humans use
        self.filler_words = {
            "just", "really", "very", "quite", "rather", "pretty",
            "actually", "basically", "literally", "honestly",
            "obviously", "clearly", "definitely", "certainly",
            "probably", "maybe", "perhaps", "possibly",
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BASIC STATISTICAL FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_basic_stats(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Extract basic statistical features"""
        features = {}
        
        n_chars = len(text)
        n_words = len(words)
        n_sentences = len(sentences)
        
        features['char_count'] = n_chars
        features['word_count'] = n_words
        features['sentence_count'] = n_sentences
        
        # Averages
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sentence_length'] = n_words / n_sentences if n_sentences > 0 else 0
        
        # Variance metrics
        word_lengths = [len(w) for w in words]
        sentence_lengths = [len(s.split()) for s in sentences]
        
        features['word_length_std'] = np.std(word_lengths) if len(word_lengths) > 1 else 0
        features['sentence_length_std'] = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
        
        # Coefficient of variation (key for AI detection)
        mean_sent_len = np.mean(sentence_lengths) if sentence_lengths else 0
        features['sentence_length_cv'] = (features['sentence_length_std'] / mean_sent_len) if mean_sent_len > 0 else 0
        
        # Burstiness coefficient
        if mean_sent_len > 0 and features['sentence_length_std'] > 0:
            sigma = features['sentence_length_std']
            mu = mean_sent_len
            features['burstiness_coef'] = (sigma - mu) / (sigma + mu)
        else:
            features['burstiness_coef'] = 0
        
        # Paragraph analysis
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        if paragraphs:
            para_lengths = [len(p.split()) for p in paragraphs]
            features['avg_paragraph_length'] = np.mean(para_lengths)
            features['paragraph_length_std'] = np.std(para_lengths) if len(para_lengths) > 1 else 0
        else:
            features['avg_paragraph_length'] = 0
            features['paragraph_length_std'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AI SIGNATURE DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_ai_signatures(self, text: str, words: List[str]) -> Dict[str, float]:
        """Detect AI-specific language patterns"""
        features = {}
        text_lower = text.lower()
        n_words = len(words) if words else 1
        
        # Count AI phrases
        ai_phrase_count = 0
        for phrase in self.ai_phrases:
            ai_phrase_count += text_lower.count(phrase)
        features['ai_phrase_count'] = ai_phrase_count
        features['ai_phrase_density'] = (ai_phrase_count / n_words) * 100
        
        # Count formal transitions
        transition_count = 0
        for trans in self.formal_transitions:
            transition_count += len(re.findall(r'\b' + trans + r'\b', text_lower))
        features['transition_count'] = transition_count
        features['transition_density'] = (transition_count / n_words) * 100
        
        # Formal word density
        formal_count = sum(1 for w in words if w.lower() in self.formal_words)
        features['formal_word_count'] = formal_count
        features['formal_word_density'] = (formal_count / n_words) * 100
        
        # Sophisticated word detection (thesaurus substitution)
        sophisticated_count = sum(1 for w in words if w.lower() in self.sophisticated_words)
        features['sophisticated_word_count'] = sophisticated_count
        features['sophisticated_density'] = (sophisticated_count / n_words) * 100
        
        # Decorative unicode
        unicode_matches = len(self.decorative_unicode.findall(text))
        features['decorative_unicode_count'] = unicode_matches
        
        # Sentence start analysis (AI often starts with "This", "The", "It")
        sentences = self._split_sentences(text)
        ai_starts = 0
        for sent in sentences:
            words_in_sent = sent.strip().split()
            if words_in_sent:
                first_word = words_in_sent[0].lower().strip(string.punctuation)
                if first_word in {'this', 'the', 'it', 'these', 'there', 'one'}:
                    ai_starts += 1
        features['ai_sentence_start_ratio'] = ai_starts / len(sentences) if sentences else 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HUMANIZATION ARTIFACT DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_humanization_artifacts(self, text: str, words: List[str]) -> Dict[str, float]:
        """Detect humanization tool artifacts"""
        features = {}
        n_words = len(words) if words else 1
        
        # Contractions (humanizers inject these)
        contraction_matches = len(self.contraction_pattern.findall(text))
        features['contraction_count'] = contraction_matches
        features['contraction_density'] = (contraction_matches / n_words) * 100
        
        # Informal contractions
        informal_contractions = len(self.informal_contraction_pattern.findall(text))
        features['informal_contraction_count'] = informal_contractions
        
        # Disfluencies (um, uh, like, basically)
        disfluency_matches = len(self.disfluency_pattern.findall(text))
        features['disfluency_count'] = disfluency_matches
        features['disfluency_density'] = (disfluency_matches / n_words) * 100
        
        # Sentence-start disfluencies
        sent_start_disf = len(self.sentence_start_disfluency.findall(text))
        features['sentence_start_disfluency_count'] = sent_start_disf
        
        # Hedging language
        hedging_matches = len(self.hedging_pattern.findall(text))
        features['hedging_count'] = hedging_matches
        features['hedging_density'] = (hedging_matches / n_words) * 100
        
        # Rhetorical questions
        rhetorical_q = len(self.rhetorical_question.findall(text))
        features['rhetorical_question_count'] = rhetorical_q
        
        # First-person usage
        first_person_matches = len(self.first_person.findall(text))
        features['first_person_count'] = first_person_matches
        features['first_person_density'] = (first_person_matches / n_words) * 100
        
        # Second-person usage
        second_person_matches = len(self.second_person.findall(text))
        features['second_person_count'] = second_person_matches
        features['second_person_density'] = (second_person_matches / n_words) * 100
        
        # Typo count (humanizers add fake typos)
        typo_count = 0
        for pattern, _ in self.typo_patterns:
            typo_count += len(pattern.findall(text))
        features['typo_count'] = typo_count
        
        # Filler word density
        filler_count = sum(1 for w in words if w.lower() in self.filler_words)
        features['filler_word_count'] = filler_count
        features['filler_word_density'] = (filler_count / n_words) * 100
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONSISTENCY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_consistency_features(self, text: str, words: List[str], sentences: List[str]) -> Dict[str, float]:
        """Analyze consistency patterns that reveal humanization"""
        features = {}
        
        if len(sentences) < 2:
            features['register_inconsistency'] = 0
            features['formality_shift_count'] = 0
            features['style_variance'] = 0
            features['vocabulary_consistency'] = 0
            return features
        
        # Analyze formality per sentence
        formality_scores = []
        for sent in sentences:
            sent_words = sent.lower().split()
            formal_in_sent = sum(1 for w in sent_words if w in self.formal_words)
            informal_in_sent = len(self.contraction_pattern.findall(sent))
            informal_in_sent += len(self.disfluency_pattern.findall(sent))
            
            if len(sent_words) > 0:
                score = (formal_in_sent - informal_in_sent) / len(sent_words)
            else:
                score = 0
            formality_scores.append(score)
        
        # Register inconsistency (high variance = humanizer mixing styles)
        features['register_inconsistency'] = np.std(formality_scores) if formality_scores else 0
        
        # Formality shifts (count direction changes)
        shift_count = 0
        for i in range(1, len(formality_scores)):
            if formality_scores[i] * formality_scores[i-1] < 0:  # Sign change
                shift_count += 1
        features['formality_shift_count'] = shift_count
        
        # Vocabulary consistency per paragraph
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) > 1:
            vocab_sizes = []
            for para in paragraphs:
                para_words = para.lower().split()
                if para_words:
                    vocab_sizes.append(len(set(para_words)) / len(para_words))
            features['vocabulary_consistency'] = np.std(vocab_sizes) if vocab_sizes else 0
        else:
            features['vocabulary_consistency'] = 0
        
        # Style variance (sentence length patterns)
        sent_lengths = [len(s.split()) for s in sentences]
        if len(sent_lengths) > 3:
            # Moving window variance
            window_size = 3
            local_variances = []
            for i in range(len(sent_lengths) - window_size + 1):
                window = sent_lengths[i:i+window_size]
                local_variances.append(np.var(window))
            features['style_variance'] = np.std(local_variances) if local_variances else 0
        else:
            features['style_variance'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINATION PATTERN DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_combination_patterns(self, text: str, words: List[str],
                                       ai_features: Dict, humanization_features: Dict) -> Dict[str, float]:
        """Detect suspicious feature combinations"""
        features = {}
        
        # Get key indicators
        ai_phrases = ai_features.get('ai_phrase_count', 0)
        formal_density = ai_features.get('formal_word_density', 0)
        contractions = humanization_features.get('contraction_count', 0)
        disfluencies = humanization_features.get('disfluency_count', 0)
        hedging = humanization_features.get('hedging_count', 0)
        typos = humanization_features.get('typo_count', 0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # COMBINATION: AI residue + Human markers = Humanized
        # ═══════════════════════════════════════════════════════════════════════
        
        # Formal language + contractions (suspicious - humanizer artifact)
        features['formal_with_contractions'] = 1 if (formal_density > 2 and contractions > 2) else 0
        
        # AI phrases + disfluencies (suspicious - humanizer artifact)
        features['ai_phrases_with_disfluencies'] = 1 if (ai_phrases > 1 and disfluencies > 2) else 0
        
        # Formal structure + casual language (humanizer pattern)
        features['formal_structure_casual_lang'] = 1 if (formal_density > 3 and contractions > 3) else 0
        
        # Technical vocabulary + typos (thesaurus + typo injection)
        features['technical_with_typos'] = 1 if (formal_density > 2 and typos > 0) else 0
        
        # AI phrase density + hedging (AI + humanizer hedging)
        features['ai_density_with_hedging'] = ai_features.get('ai_phrase_density', 0) * (1 + hedging * 0.1)
        
        # Composite humanization score
        ai_score = ai_phrases + formal_density
        human_score = contractions + disfluencies + hedging
        if ai_score > 0 and human_score > 0:
            features['ai_human_cooccurrence'] = min(1.0, (ai_score * human_score) / 20)
        else:
            features['ai_human_cooccurrence'] = 0
        
        # Mixed register score
        if formal_density > 2 and (contractions > 1 or disfluencies > 1):
            features['mixed_register_score'] = (formal_density + contractions + disfluencies) / 10
        else:
            features['mixed_register_score'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NULL COMBINATION PATTERNS (A & !B logic)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_null_combinations(self, text: str, words: List[str],
                                    ai_features: Dict, humanization_features: Dict,
                                    basic_features: Dict) -> Dict[str, float]:
        """
        Detect null combination patterns where the ABSENCE of something is key.
        
        Patterns like:
        - High formality & NO contractions = Pure AI
        - High contractions & NO AI phrases = Pure Human
        - High formality & HAS contractions = Humanized
        """
        features = {}
        
        # Get indicators
        formal_density = ai_features.get('formal_word_density', 0)
        ai_phrases = ai_features.get('ai_phrase_count', 0)
        contractions = humanization_features.get('contraction_count', 0)
        disfluencies = humanization_features.get('disfluency_count', 0)
        first_person = humanization_features.get('first_person_count', 0)
        sentence_cv = basic_features.get('sentence_length_cv', 0)
        burstiness = basic_features.get('burstiness_coef', 0)
        
        # ═══════════════════════════════════════════════════════════════════════
        # PURE AI PATTERNS (formal & !human_markers)
        # ═══════════════════════════════════════════════════════════════════════
        
        # High formality with NO contractions = likely AI
        features['formal_no_contractions'] = 1 if (formal_density > 3 and contractions == 0) else 0
        
        # AI phrases with NO disfluencies = likely AI
        features['ai_phrases_no_disfluencies'] = 1 if (ai_phrases > 2 and disfluencies == 0) else 0
        
        # Low sentence variance with NO burstiness = likely AI
        features['uniform_no_burstiness'] = 1 if (sentence_cv < 0.3 and burstiness < 0) else 0
        
        # Formal with NO first-person = likely AI
        features['formal_no_first_person'] = 1 if (formal_density > 2 and first_person == 0) else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # PURE HUMAN PATTERNS (!formal & has human_markers)
        # ═══════════════════════════════════════════════════════════════════════
        
        # Contractions with NO AI phrases = likely human
        features['contractions_no_ai'] = 1 if (contractions > 3 and ai_phrases == 0) else 0
        
        # Disfluencies with NO formal words = likely human
        features['disfluencies_no_formal'] = 1 if (disfluencies > 2 and formal_density < 1) else 0
        
        # High burstiness with NO uniformity = likely human
        features['bursty_not_uniform'] = 1 if (burstiness > 0.2 and sentence_cv > 0.4) else 0
        
        # First-person heavy with NO AI phrases = likely human
        features['personal_no_ai'] = 1 if (first_person > 5 and ai_phrases == 0) else 0
        
        # ═══════════════════════════════════════════════════════════════════════
        # HUMANIZED PATTERNS (has AI & has human & suspicious combinations)
        # ═══════════════════════════════════════════════════════════════════════
        
        # AI phrases WITH contractions = humanized
        features['ai_with_contractions'] = 1 if (ai_phrases > 0 and contractions > 2) else 0
        
        # Formal language WITH disfluencies = humanized
        features['formal_with_disfluencies'] = 1 if (formal_density > 2 and disfluencies > 1) else 0
        
        # Low variance WITH contractions = humanized (fake variance + human markers)
        features['uniform_with_human_markers'] = 1 if (sentence_cv < 0.35 and (contractions > 2 or disfluencies > 1)) else 0
        
        # Both AI and human indicators present = humanized
        ai_indicators = (ai_phrases > 1) + (formal_density > 2) + (sentence_cv < 0.3)
        human_indicators = (contractions > 2) + (disfluencies > 1) + (first_person > 3)
        features['both_ai_and_human_present'] = 1 if (ai_indicators >= 2 and human_indicators >= 2) else 0
        
        # Aggregate null pattern scores
        features['pure_ai_null_score'] = (
            features['formal_no_contractions'] +
            features['ai_phrases_no_disfluencies'] +
            features['uniform_no_burstiness'] +
            features['formal_no_first_person']
        ) / 4
        
        features['pure_human_null_score'] = (
            features['contractions_no_ai'] +
            features['disfluencies_no_formal'] +
            features['bursty_not_uniform'] +
            features['personal_no_ai']
        ) / 4
        
        features['humanized_null_score'] = (
            features['ai_with_contractions'] +
            features['formal_with_disfluencies'] +
            features['uniform_with_human_markers'] +
            features['both_ai_and_human_present']
        ) / 4
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # NEURAL EMBEDDING FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_embedding_features(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Extract features using neural sentence embeddings"""
        features = {}
        
        if not self.use_neural or not self.embedding_model or len(sentences) < 2:
            # Return zero features if neural not available
            features['embedding_coherence'] = 0
            features['embedding_diversity'] = 0
            features['embedding_drift'] = 0
            features['embedding_clustering'] = 0
            features['embedding_uniformity'] = 0
            return features
        
        try:
            # Get embeddings for all sentences
            embeddings = self.embedding_model.encode(sentences[:50])  # Limit for speed
            
            # Coherence: average cosine similarity between adjacent sentences
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i+1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i+1]) + 1e-10
                )
                coherence_scores.append(sim)
            features['embedding_coherence'] = np.mean(coherence_scores) if coherence_scores else 0
            
            # Diversity: average distance from centroid
            centroid = np.mean(embeddings, axis=0)
            distances = [np.linalg.norm(e - centroid) for e in embeddings]
            features['embedding_diversity'] = np.mean(distances)
            
            # Drift: change in similarity over document
            if len(coherence_scores) > 2:
                first_half = np.mean(coherence_scores[:len(coherence_scores)//2])
                second_half = np.mean(coherence_scores[len(coherence_scores)//2:])
                features['embedding_drift'] = abs(first_half - second_half)
            else:
                features['embedding_drift'] = 0
            
            # Clustering: variance in pairwise similarities
            pairwise_sims = []
            for i in range(min(20, len(embeddings))):
                for j in range(i+1, min(20, len(embeddings))):
                    sim = np.dot(embeddings[i], embeddings[j]) / (
                        np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10
                    )
                    pairwise_sims.append(sim)
            features['embedding_clustering'] = np.std(pairwise_sims) if pairwise_sims else 0
            
            # Uniformity: how evenly distributed embeddings are
            features['embedding_uniformity'] = np.std(distances) if distances else 0
            
        except Exception:
            features['embedding_coherence'] = 0
            features['embedding_diversity'] = 0
            features['embedding_drift'] = 0
            features['embedding_clustering'] = 0
            features['embedding_uniformity'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PERPLEXITY FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_perplexity_features(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Extract perplexity-based features using GPT-2"""
        features = {}
        
        if not self.use_perplexity or not self.perplexity_model:
            features['perplexity_mean'] = 0
            features['perplexity_std'] = 0
            features['perplexity_min'] = 0
            features['perplexity_max'] = 0
            features['perplexity_burstiness'] = 0
            features['low_perplexity_ratio'] = 0
            return features
        
        try:
            perplexities = []
            
            for sent in sentences[:30]:  # Limit for speed
                if len(sent.strip()) < 10:
                    continue
                    
                inputs = self.perplexity_tokenizer(sent, return_tensors='pt', truncation=True, max_length=512)
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.perplexity_model(**inputs, labels=inputs['input_ids'])
                    loss = outputs.loss.item()
                    perplexity = math.exp(loss)
                    perplexities.append(min(perplexity, 1000))  # Cap extreme values
            
            if perplexities:
                features['perplexity_mean'] = np.mean(perplexities)
                features['perplexity_std'] = np.std(perplexities)
                features['perplexity_min'] = np.min(perplexities)
                features['perplexity_max'] = np.max(perplexities)
                
                # Burstiness in perplexity
                mean_ppl = features['perplexity_mean']
                std_ppl = features['perplexity_std']
                if mean_ppl + std_ppl > 0:
                    features['perplexity_burstiness'] = (std_ppl - mean_ppl) / (std_ppl + mean_ppl)
                else:
                    features['perplexity_burstiness'] = 0
                
                # Ratio of low perplexity sentences (AI indicator)
                low_ppl_count = sum(1 for p in perplexities if p < 50)
                features['low_perplexity_ratio'] = low_ppl_count / len(perplexities)
            else:
                features['perplexity_mean'] = 0
                features['perplexity_std'] = 0
                features['perplexity_min'] = 0
                features['perplexity_max'] = 0
                features['perplexity_burstiness'] = 0
                features['low_perplexity_ratio'] = 0
                
        except Exception:
            features['perplexity_mean'] = 0
            features['perplexity_std'] = 0
            features['perplexity_min'] = 0
            features['perplexity_max'] = 0
            features['perplexity_burstiness'] = 0
            features['low_perplexity_ratio'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WATERMARK DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_watermark_features(self, text: str) -> Dict[str, float]:
        """Detect potential AI watermark patterns"""
        features = {}
        
        # Zero-width character count
        zwc_count = 0
        for pattern in self.watermark_patterns[:1]:  # First pattern is zero-width
            zwc_count += len(pattern.findall(text))
        features['zero_width_chars'] = zwc_count
        
        # Repeated transition pattern
        repeat_trans = 0
        for pattern in self.watermark_patterns[1:2]:
            repeat_trans += len(pattern.findall(text))
        features['repeated_transitions'] = repeat_trans
        
        # Unusual spacing
        unusual_spacing = 0
        for pattern in self.watermark_patterns[2:3]:
            unusual_spacing += len(pattern.findall(text))
        features['unusual_spacing'] = unusual_spacing
        
        # Non-standard quotes
        nonstd_quotes = 0
        for pattern in self.watermark_patterns[3:]:
            nonstd_quotes += len(pattern.findall(text))
        features['nonstandard_quotes'] = nonstd_quotes
        
        # Token pattern analysis (some watermarks affect token distribution)
        words = text.split()
        if len(words) > 10:
            # Check for unusual word length patterns
            word_lengths = [len(w) for w in words]
            
            # Detect periodic patterns in word lengths
            autocorr = self._compute_autocorrelation(word_lengths, lag=5)
            features['word_length_autocorr'] = autocorr
            
            # Detect unusual first letter distribution (some watermarks bias this)
            first_letters = [w[0].lower() for w in words if w and w[0].isalpha()]
            letter_freq = Counter(first_letters)
            if first_letters:
                expected_freq = len(first_letters) / 26
                chi_sq = sum((letter_freq.get(chr(ord('a')+i), 0) - expected_freq)**2 / (expected_freq + 1) 
                            for i in range(26))
                features['first_letter_chi_sq'] = chi_sq
            else:
                features['first_letter_chi_sq'] = 0
        else:
            features['word_length_autocorr'] = 0
            features['first_letter_chi_sq'] = 0
        
        # Composite watermark score
        features['watermark_score'] = (
            features['zero_width_chars'] * 10 +
            features['repeated_transitions'] * 5 +
            features['unusual_spacing'] * 3 +
            min(features['word_length_autocorr'] * 2, 5)
        )
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SEMANTIC COHERENCE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_semantic_coherence(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Analyze semantic coherence at a deep level"""
        features = {}
        
        if len(sentences) < 3:
            features['topic_drift'] = 0
            features['coherence_variance'] = 0
            features['semantic_gaps'] = 0
            features['theme_consistency'] = 0
            return features
        
        # Extract key content words per sentence
        content_words_per_sent = []
        for sent in sentences:
            words = re.findall(r'\b[a-z]{4,}\b', sent.lower())
            # Filter out common words
            stopwords = {'this', 'that', 'with', 'have', 'from', 'they', 'been', 
                        'were', 'said', 'each', 'which', 'their', 'would', 'there',
                        'could', 'other', 'about', 'very', 'just', 'also', 'your'}
            content = [w for w in words if w not in stopwords]
            content_words_per_sent.append(set(content))
        
        # Topic drift: measure overlap between distant sentences
        if len(content_words_per_sent) > 4:
            early_words = set().union(*content_words_per_sent[:len(content_words_per_sent)//3])
            late_words = set().union(*content_words_per_sent[-len(content_words_per_sent)//3:])
            if early_words and late_words:
                overlap = len(early_words & late_words) / len(early_words | late_words)
                features['topic_drift'] = 1 - overlap
            else:
                features['topic_drift'] = 0
        else:
            features['topic_drift'] = 0
        
        # Coherence variance: overlap between adjacent sentences
        overlaps = []
        for i in range(len(content_words_per_sent) - 1):
            set1 = content_words_per_sent[i]
            set2 = content_words_per_sent[i + 1]
            if set1 and set2:
                overlap = len(set1 & set2) / len(set1 | set2)
            else:
                overlap = 0
            overlaps.append(overlap)
        features['coherence_variance'] = np.std(overlaps) if overlaps else 0
        
        # Semantic gaps: count sentences with no overlap
        gap_count = sum(1 for o in overlaps if o == 0)
        features['semantic_gaps'] = gap_count / len(overlaps) if overlaps else 0
        
        # Theme consistency: are the same key words repeated?
        all_content = [w for s in content_words_per_sent for w in s]
        if all_content:
            word_freq = Counter(all_content)
            repeated = sum(1 for w, c in word_freq.items() if c > 1)
            features['theme_consistency'] = repeated / len(word_freq)
        else:
            features['theme_consistency'] = 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PARTIAL HUMANIZATION DETECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_partial_humanization(self, text: str, sentences: List[str]) -> Dict[str, float]:
        """Detect partially humanized text (some segments humanized, others not)"""
        features = {}
        
        if len(sentences) < 4:
            features['humanization_variance'] = 0
            features['segment_inconsistency'] = 0
            features['humanized_segment_ratio'] = 0
            features['ai_segment_ratio'] = 0
            features['mixed_segment_score'] = 0
            return features
        
        # Score each sentence for AI-ness and human-ness
        ai_scores = []
        human_scores = []
        
        for sent in sentences:
            sent_lower = sent.lower()
            sent_words = sent.split()
            n_words = len(sent_words) if sent_words else 1
            
            # AI indicators in this sentence
            ai_score = 0
            for phrase in self.ai_phrases[:15]:  # Check top phrases
                if phrase in sent_lower:
                    ai_score += 1
            formal_in_sent = sum(1 for w in sent_words if w.lower() in self.formal_words)
            ai_score += formal_in_sent / n_words * 5
            ai_scores.append(ai_score)
            
            # Human indicators in this sentence
            human_score = 0
            human_score += len(self.contraction_pattern.findall(sent))
            human_score += len(self.disfluency_pattern.findall(sent))
            human_score += len(self.first_person.findall(sent)) * 0.5
            human_scores.append(human_score)
        
        # Variance in scores (high variance = partial humanization)
        features['humanization_variance'] = np.std(human_scores) if human_scores else 0
        
        # Segment analysis
        segment_types = []
        for ai, human in zip(ai_scores, human_scores):
            if ai > 1 and human < 0.5:
                segment_types.append('ai')
            elif human > 1 and ai < 0.5:
                segment_types.append('human')
            elif ai > 0.5 and human > 0.5:
                segment_types.append('mixed')
            else:
                segment_types.append('neutral')
        
        n_segments = len(segment_types)
        features['ai_segment_ratio'] = segment_types.count('ai') / n_segments if n_segments else 0
        features['humanized_segment_ratio'] = segment_types.count('human') / n_segments if n_segments else 0
        features['mixed_segment_score'] = segment_types.count('mixed') / n_segments if n_segments else 0
        
        # Segment inconsistency: transitions between types
        transitions = 0
        for i in range(1, len(segment_types)):
            if segment_types[i] != segment_types[i-1]:
                transitions += 1
        features['segment_inconsistency'] = transitions / (n_segments - 1) if n_segments > 1 else 0
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VOCABULARY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_vocabulary_features(self, words: List[str]) -> Dict[str, float]:
        """Analyze vocabulary patterns"""
        features = {}
        
        if not words:
            return {
                'type_token_ratio': 0,
                'hapax_ratio': 0,
                'dis_legomena_ratio': 0,
                'vocabulary_richness': 0,
                'word_length_entropy': 0,
            }
        
        n_words = len(words)
        word_lower = [w.lower() for w in words]
        word_freq = Counter(word_lower)
        
        # Type-token ratio
        n_unique = len(word_freq)
        features['type_token_ratio'] = n_unique / n_words
        
        # Hapax legomena (words appearing once)
        hapax = sum(1 for w, c in word_freq.items() if c == 1)
        features['hapax_ratio'] = hapax / n_words
        
        # Dis legomena (words appearing twice)
        dis = sum(1 for w, c in word_freq.items() if c == 2)
        features['dis_legomena_ratio'] = dis / n_words
        
        # Vocabulary richness (Yule's K approximation)
        m1 = n_words
        m2 = sum(c * c for c in word_freq.values())
        if m1 > 0 and m1 != m2:
            features['vocabulary_richness'] = 10000 * (m2 - m1) / (m1 * m1)
        else:
            features['vocabulary_richness'] = 0
        
        # Word length entropy
        word_lengths = [len(w) for w in words]
        length_freq = Counter(word_lengths)
        total = sum(length_freq.values())
        entropy = 0
        for count in length_freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        features['word_length_entropy'] = entropy
        
        return features
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        sentences = self.sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip() and len(s.split()) > 2]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text)
    
    def _compute_autocorrelation(self, values: List[float], lag: int = 1) -> float:
        """Compute autocorrelation at given lag"""
        if len(values) <= lag:
            return 0
        
        n = len(values)
        mean = np.mean(values)
        var = np.var(values)
        
        if var == 0:
            return 0
        
        autocorr = np.sum((np.array(values[:-lag]) - mean) * (np.array(values[lag:]) - mean)) / ((n - lag) * var)
        return autocorr
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN EXTRACTION METHOD
    # ═══════════════════════════════════════════════════════════════════════════
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary of feature name -> value
        """
        if not text or len(text.strip()) < 10:
            return {name: 0.0 for name in self.get_feature_names()}
        
        # Preprocess
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        # Extract all feature categories
        features = {}
        
        # 1. Basic statistical features
        basic = self._extract_basic_stats(text, words, sentences)
        features.update(basic)
        
        # 2. AI signature detection
        ai_features = self._extract_ai_signatures(text, words)
        features.update(ai_features)
        
        # 3. Humanization artifact detection
        humanization = self._extract_humanization_artifacts(text, words)
        features.update(humanization)
        
        # 4. Consistency analysis
        consistency = self._extract_consistency_features(text, words, sentences)
        features.update(consistency)
        
        # 5. Combination patterns
        combinations = self._extract_combination_patterns(text, words, ai_features, humanization)
        features.update(combinations)
        
        # 6. Null combination patterns
        null_combos = self._extract_null_combinations(text, words, ai_features, humanization, basic)
        features.update(null_combos)
        
        # 7. Neural embedding features
        embeddings = self._extract_embedding_features(text, sentences)
        features.update(embeddings)
        
        # 8. Perplexity features
        perplexity = self._extract_perplexity_features(text, sentences)
        features.update(perplexity)
        
        # 9. Watermark detection
        watermarks = self._extract_watermark_features(text)
        features.update(watermarks)
        
        # 10. Semantic coherence
        coherence = self._extract_semantic_coherence(text, sentences)
        features.update(coherence)
        
        # 11. Partial humanization detection
        partial = self._extract_partial_humanization(text, sentences)
        features.update(partial)
        
        # 12. Vocabulary features
        vocab = self._extract_vocabulary_features(words)
        features.update(vocab)
        
        return features
    
    def extract_feature_vector(self, text: str) -> np.ndarray:
        """
        Extract features as a numpy array.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Numpy array of feature values
        """
        features = self.extract_features(text)
        names = self.get_feature_names()
        return np.array([features.get(name, 0.0) for name in names])
    
    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names"""
        if self._feature_names is None:
            # Generate feature names from a sample extraction
            sample = self.extract_features("Sample text for feature name extraction.")
            self._feature_names = sorted(sample.keys())
        return self._feature_names


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Feature Extractor V3 - Testing")
    print("=" * 70)
    
    # Initialize extractor (disable neural for quick test)
    extractor = FeatureExtractorV3(use_neural=False, use_perplexity=False)
    
    # Test texts
    human_text = """
    I was thinking about this the other day, and honestly, it's kind of weird 
    how we've gotten so used to everything being digital. Like, remember when 
    we actually had to go to the library? My friend Sarah still does that, 
    and I think it's pretty cool. Anyway, I'm probably overthinking this, 
    but it just feels like something's changed, you know?
    """
    
    ai_text = """
    In the contemporary landscape of digital transformation, it is crucial to 
    understand the multifaceted implications of technological advancement. 
    The significance of this paradigm shift cannot be overstated, as it 
    fundamentally alters the way individuals interact with information. 
    Furthermore, the comprehensive integration of digital tools has 
    demonstrated remarkable efficacy in enhancing productivity and 
    facilitating seamless communication across diverse platforms.
    """
    
    humanized_text = """
    So, in today's world of digital transformation, it's really important to 
    understand all the different ways technology is changing things. I think 
    this shift is pretty significant - it basically changes how we interact 
    with information, you know? Also, using digital tools has shown it can 
    really help with productivity. Honestly, it just makes communication 
    easier across different platforms and stuff.
    """
    
    print(f"\nTotal features: {len(extractor.get_feature_names())}")
    print("\n" + "=" * 70)
    
    for name, text in [("HUMAN", human_text), ("AI", ai_text), ("HUMANIZED", humanized_text)]:
        print(f"\n{name} TEXT:")
        features = extractor.extract_features(text)
        
        # Print key features
        key_features = [
            'ai_phrase_count', 'ai_phrase_density', 'formal_word_density',
            'contraction_count', 'disfluency_count', 'first_person_count',
            'ai_human_cooccurrence', 'pure_ai_null_score', 'pure_human_null_score',
            'humanized_null_score', 'sentence_length_cv', 'burstiness_coef',
            'formal_no_contractions', 'contractions_no_ai', 'ai_with_contractions',
        ]
        
        for feat in key_features:
            if feat in features:
                print(f"  {feat}: {features[feat]:.4f}")
    
    print("\n" + "=" * 70)
    print("Feature Extractor V3 - Test Complete")
    print("=" * 70)
