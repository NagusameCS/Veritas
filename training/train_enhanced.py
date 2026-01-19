#!/usr/bin/env python3
"""
Enhanced Feature Extractor with Tone Detection
==============================================
Adds tone, emotional patterns, and additional indicators to improve accuracy.

New Feature Categories:
1. TONE DETECTION - Formal vs informal, emotional tone
2. HEDGING PATTERNS - AI uses more hedging language
3. CERTAINTY MARKERS - AI tends to be more definitive
4. PERSONAL VOICE - Humans use more personal expressions
5. RHETORICAL PATTERNS - Question usage, emphasis patterns
6. SEMANTIC COHERENCE - Topic consistency across text
7. PUNCTUATION RHYTHM - Natural vs mechanical punctuation
8. OPENING/CLOSING PATTERNS - How text starts and ends
"""

import os
import pickle
import json
import re
import random
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset


class EnhancedFeatureExtractor:
    """
    Comprehensive feature extractor with tone detection and advanced indicators.
    """
    
    def __init__(self):
        # Define all feature categories
        self.feature_names = [
            # === EXISTING CORE FEATURES ===
            'paragraph_uniformity',
            'sentence_length_cv',
            'sentence_length_range_norm',
            'sentence_start_diversity',
            'avg_sentence_length',
            'type_token_ratio',
            'hapax_ratio',
            'bigram_entropy',
            'trigram_entropy',
            
            # === TONE DETECTION ===
            'formality_score',           # 0 = very informal, 1 = very formal
            'emotional_intensity',       # Strength of emotional language
            'sentiment_variance',        # How much sentiment changes
            'tone_consistency',          # Is tone uniform throughout?
            
            # === HEDGING PATTERNS (AI signature) ===
            'hedge_word_density',        # "might", "could", "perhaps", etc.
            'qualifier_density',         # "somewhat", "relatively", etc.
            'uncertainty_marker_rate',   # Expressions of uncertainty
            
            # === CERTAINTY MARKERS ===
            'definitive_statement_rate', # "is", "will", "must" without hedging
            'absolute_word_density',     # "always", "never", "all", etc.
            'assertion_confidence',      # Ratio of confident to hedged statements
            
            # === PERSONAL VOICE ===
            'first_person_singular',     # I, me, my
            'first_person_plural',       # We, us, our
            'second_person_rate',        # You, your
            'personal_anecdote_markers', # "remember", "once", "I think"
            'opinion_marker_density',    # "I believe", "in my view"
            
            # === RHETORICAL PATTERNS ===
            'question_density',          # Questions per sentence
            'exclamation_density',       # Emphasis markers
            'rhetorical_question_rate',  # Questions that don't need answers
            'emphasis_word_density',     # "very", "really", "extremely"
            
            # === SEMANTIC COHERENCE ===
            'topic_drift_score',         # How much topic changes
            'lexical_chain_strength',    # Word repetition for coherence
            'pronoun_reference_density', # Pronouns referring back
            
            # === PUNCTUATION RHYTHM ===
            'comma_rhythm_variance',     # Regularity of comma usage
            'sentence_end_diversity',    # Variety in sentence endings
            'parenthetical_usage',       # Use of parentheses, dashes
            'ellipsis_rate',             # Use of "..."
            
            # === OPENING/CLOSING PATTERNS ===
            'opening_formula_score',     # How formulaic the opening is
            'closing_formula_score',     # How formulaic the closing is
            'hook_strength',             # Engaging opening vs dry
            
            # === ADDITIONAL INDICATORS ===
            'passive_voice_ratio',       # AI tends to use more passive
            'nominalization_rate',       # Abstract noun usage
            'concrete_noun_ratio',       # Specific vs abstract
            'action_verb_density',       # Dynamic vs static
            'transition_density',        # Explicit transitions
        ]
        
        # Word lists for detection
        self.hedge_words = [
            'might', 'could', 'may', 'perhaps', 'possibly', 'probably',
            'somewhat', 'relatively', 'fairly', 'rather', 'quite',
            'seems', 'appears', 'suggests', 'indicates', 'tends',
            'generally', 'typically', 'usually', 'often', 'sometimes'
        ]
        
        self.certainty_words = [
            'always', 'never', 'definitely', 'certainly', 'absolutely',
            'undoubtedly', 'clearly', 'obviously', 'must', 'will',
            'is', 'are', 'proves', 'demonstrates', 'shows'
        ]
        
        self.emotional_words = {
            'positive': ['love', 'great', 'amazing', 'wonderful', 'fantastic', 
                        'excellent', 'happy', 'excited', 'thrilled', 'delighted'],
            'negative': ['hate', 'terrible', 'awful', 'horrible', 'disgusting',
                        'angry', 'frustrated', 'disappointed', 'sad', 'upset'],
            'surprise': ['wow', 'whoa', 'unbelievable', 'shocking', 'stunning',
                        'incredible', 'unexpected', 'surprising', 'amazed']
        }
        
        self.opinion_markers = [
            'i think', 'i believe', 'i feel', 'in my opinion', 'in my view',
            'it seems to me', 'personally', 'from my perspective', 'i find'
        ]
        
        self.formal_indicators = [
            'furthermore', 'moreover', 'consequently', 'therefore', 'thus',
            'nevertheless', 'nonetheless', 'accordingly', 'hence', 'whereby'
        ]
        
        self.informal_indicators = [
            'gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'yeah', 'nope',
            'cool', 'awesome', 'stuff', 'things', 'basically', 'literally'
        ]
        
        self.opening_formulas = [
            'it is important', 'in today', 'in this', 'this article',
            'the purpose', 'the following', 'as we know', 'it has been'
        ]
        
        self.closing_formulas = [
            'in conclusion', 'to summarize', 'in summary', 'finally',
            'to conclude', 'in closing', 'overall', 'all in all'
        ]
        
        self.passive_markers = ['is', 'are', 'was', 'were', 'been', 'being', 'be']
        self.by_phrase = re.compile(r'\b(?:is|are|was|were|been|being)\s+\w+ed\s+by\b', re.IGNORECASE)
        
    def extract(self, text: str) -> np.ndarray:
        """Extract all features from text."""
        lower = text.lower()
        words = re.findall(r"[a-zA-Z']+", lower)
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(words) < 30 or len(sentences) < 3:
            return np.zeros(len(self.feature_names))
        
        features = {}
        word_freq = Counter(words)
        
        # === CORE FEATURES ===
        features.update(self._extract_core_features(text, words, sentences, paragraphs, word_freq))
        
        # === TONE DETECTION ===
        features.update(self._extract_tone_features(text, words, sentences))
        
        # === HEDGING PATTERNS ===
        features.update(self._extract_hedging_features(words, sentences))
        
        # === CERTAINTY MARKERS ===
        features.update(self._extract_certainty_features(words, sentences))
        
        # === PERSONAL VOICE ===
        features.update(self._extract_personal_voice_features(text, words, sentences))
        
        # === RHETORICAL PATTERNS ===
        features.update(self._extract_rhetorical_features(text, words, sentences))
        
        # === SEMANTIC COHERENCE ===
        features.update(self._extract_coherence_features(words, sentences))
        
        # === PUNCTUATION RHYTHM ===
        features.update(self._extract_punctuation_features(text, sentences))
        
        # === OPENING/CLOSING ===
        features.update(self._extract_opening_closing_features(text, sentences))
        
        # === ADDITIONAL INDICATORS ===
        features.update(self._extract_additional_features(text, words, sentences))
        
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
    
    def _extract_core_features(self, text, words, sentences, paragraphs, word_freq):
        features = {}
        
        # Paragraph uniformity
        if len(paragraphs) > 1:
            para_lens = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - min(np.std(para_lens) / (np.mean(para_lens) + 1), 1)
        else:
            features['paragraph_uniformity'] = 0.5
        
        # Sentence stats
        sent_lens = [len(s.split()) for s in sentences]
        features['sentence_length_cv'] = np.std(sent_lens) / (np.mean(sent_lens) + 0.01)
        features['sentence_length_range_norm'] = (max(sent_lens) - min(sent_lens)) / (np.mean(sent_lens) + 0.01)
        
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        features['avg_sentence_length'] = np.mean(sent_lens)
        
        # Vocabulary
        features['type_token_ratio'] = len(set(words)) / len(words)
        features['hapax_ratio'] = sum(1 for c in word_freq.values() if c == 1) / len(words)
        
        # Entropy
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        features['bigram_entropy'] = self._entropy(bigrams)
        features['trigram_entropy'] = self._entropy(trigrams)
        
        return features
    
    def _extract_tone_features(self, text, words, sentences):
        features = {}
        lower = text.lower()
        
        # Formality score
        formal_count = sum(1 for w in self.formal_indicators if w in lower)
        informal_count = sum(1 for w in self.informal_indicators if w in lower)
        total = formal_count + informal_count + 1
        features['formality_score'] = formal_count / total
        
        # Emotional intensity
        emotion_count = 0
        for category in self.emotional_words.values():
            emotion_count += sum(1 for w in words if w in category)
        features['emotional_intensity'] = emotion_count / len(words) * 100
        
        # Sentiment variance across sentences
        sent_sentiments = []
        for sent in sentences:
            sent_words = set(sent.lower().split())
            pos = sum(1 for w in self.emotional_words['positive'] if w in sent_words)
            neg = sum(1 for w in self.emotional_words['negative'] if w in sent_words)
            sent_sentiments.append(pos - neg)
        
        features['sentiment_variance'] = np.std(sent_sentiments) if len(sent_sentiments) > 1 else 0
        
        # Tone consistency
        if len(sent_sentiments) > 2:
            changes = sum(1 for i in range(1, len(sent_sentiments)) 
                         if (sent_sentiments[i] > 0) != (sent_sentiments[i-1] > 0) and 
                            (sent_sentiments[i] != 0 or sent_sentiments[i-1] != 0))
            features['tone_consistency'] = 1 - (changes / len(sent_sentiments))
        else:
            features['tone_consistency'] = 1.0
        
        return features
    
    def _extract_hedging_features(self, words, sentences):
        features = {}
        
        hedge_count = sum(1 for w in words if w in self.hedge_words)
        features['hedge_word_density'] = hedge_count / len(words) * 100
        
        qualifiers = ['somewhat', 'relatively', 'fairly', 'rather', 'quite', 'slightly', 'mostly']
        qual_count = sum(1 for w in words if w in qualifiers)
        features['qualifier_density'] = qual_count / len(words) * 100
        
        uncertainty = ['perhaps', 'maybe', 'possibly', 'might', 'could']
        unc_count = sum(1 for w in words if w in uncertainty)
        features['uncertainty_marker_rate'] = unc_count / len(sentences)
        
        return features
    
    def _extract_certainty_features(self, words, sentences):
        features = {}
        
        definitive = ['is', 'are', 'will', 'must', 'does', 'proves', 'shows']
        def_count = sum(1 for w in words if w in definitive)
        features['definitive_statement_rate'] = def_count / len(sentences)
        
        absolutes = ['always', 'never', 'all', 'none', 'every', 'completely', 'totally']
        abs_count = sum(1 for w in words if w in absolutes)
        features['absolute_word_density'] = abs_count / len(words) * 100
        
        # Assertion confidence: definitive / (definitive + hedging)
        hedge_count = sum(1 for w in words if w in self.hedge_words)
        if def_count + hedge_count > 0:
            features['assertion_confidence'] = def_count / (def_count + hedge_count)
        else:
            features['assertion_confidence'] = 0.5
        
        return features
    
    def _extract_personal_voice_features(self, text, words, sentences):
        features = {}
        lower = text.lower()
        
        first_singular = ['i', 'me', 'my', 'mine', 'myself']
        first_plural = ['we', 'us', 'our', 'ours', 'ourselves']
        second_person = ['you', 'your', 'yours', 'yourself']
        
        features['first_person_singular'] = sum(1 for w in words if w in first_singular) / len(words) * 100
        features['first_person_plural'] = sum(1 for w in words if w in first_plural) / len(words) * 100
        features['second_person_rate'] = sum(1 for w in words if w in second_person) / len(words) * 100
        
        # Personal anecdote markers
        anecdote_markers = ['remember', 'once', 'one time', 'this one', 'i recall', 'years ago']
        anecdote_count = sum(1 for m in anecdote_markers if m in lower)
        features['personal_anecdote_markers'] = anecdote_count / len(sentences)
        
        # Opinion markers
        opinion_count = sum(1 for m in self.opinion_markers if m in lower)
        features['opinion_marker_density'] = opinion_count / len(sentences)
        
        return features
    
    def _extract_rhetorical_features(self, text, words, sentences):
        features = {}
        
        questions = text.count('?')
        exclamations = text.count('!')
        
        features['question_density'] = questions / len(sentences)
        features['exclamation_density'] = exclamations / len(sentences)
        
        # Rhetorical questions (questions without "who", "what", "when", etc.)
        question_sents = [s for s in sentences if s.strip().endswith('?')]
        interrogatives = ['who', 'what', 'when', 'where', 'why', 'how', 'which']
        rhetorical = sum(1 for s in question_sents 
                        if not any(w in s.lower().split()[:3] for w in interrogatives))
        features['rhetorical_question_rate'] = rhetorical / len(sentences) if question_sents else 0
        
        # Emphasis words
        emphasis = ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely']
        emph_count = sum(1 for w in words if w in emphasis)
        features['emphasis_word_density'] = emph_count / len(words) * 100
        
        return features
    
    def _extract_coherence_features(self, words, sentences):
        features = {}
        
        # Topic drift: compare vocabulary across chunks
        chunk_size = len(sentences) // 3
        if chunk_size >= 1:
            chunks = [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]
            chunk_vocabs = [set(re.findall(r'\b[a-z]+\b', c.lower())) for c in chunks[:3]]
            
            if len(chunk_vocabs) >= 2:
                overlaps = []
                for i in range(len(chunk_vocabs) - 1):
                    intersection = len(chunk_vocabs[i] & chunk_vocabs[i+1])
                    union = len(chunk_vocabs[i] | chunk_vocabs[i+1])
                    overlaps.append(intersection / union if union > 0 else 0)
                features['topic_drift_score'] = 1 - np.mean(overlaps)
            else:
                features['topic_drift_score'] = 0
        else:
            features['topic_drift_score'] = 0
        
        # Lexical chain strength (content word repetition)
        content_words = [w for w in words if len(w) > 4]  # Assume longer words are content
        if content_words:
            word_counts = Counter(content_words)
            repeated = sum(1 for c in word_counts.values() if c > 1)
            features['lexical_chain_strength'] = repeated / len(set(content_words))
        else:
            features['lexical_chain_strength'] = 0
        
        # Pronoun reference density
        pronouns = ['he', 'she', 'it', 'they', 'this', 'that', 'these', 'those', 'which', 'who']
        pron_count = sum(1 for w in words if w in pronouns)
        features['pronoun_reference_density'] = pron_count / len(words) * 100
        
        return features
    
    def _extract_punctuation_features(self, text, sentences):
        features = {}
        
        # Comma rhythm variance
        commas_per_sent = [s.count(',') for s in sentences]
        features['comma_rhythm_variance'] = np.std(commas_per_sent) if commas_per_sent else 0
        
        # Sentence ending diversity
        endings = [s.strip()[-1] if s.strip() else '.' for s in sentences]
        features['sentence_end_diversity'] = len(set(endings)) / 3  # Normalize by max (. ! ?)
        
        # Parenthetical usage
        parens = text.count('(') + text.count(')')
        dashes = text.count(' - ') + text.count('—')
        features['parenthetical_usage'] = (parens + dashes) / len(sentences)
        
        # Ellipsis rate
        ellipsis = text.count('...') + text.count('…')
        features['ellipsis_rate'] = ellipsis / len(sentences)
        
        return features
    
    def _extract_opening_closing_features(self, text, sentences):
        features = {}
        lower = text.lower()
        
        # Opening formula score
        opening = sentences[0].lower() if sentences else ""
        formula_count = sum(1 for f in self.opening_formulas if f in opening)
        features['opening_formula_score'] = min(formula_count / 2, 1)  # Cap at 1
        
        # Closing formula score
        closing = sentences[-1].lower() if sentences else ""
        close_count = sum(1 for f in self.closing_formulas if f in closing)
        features['closing_formula_score'] = min(close_count / 2, 1)
        
        # Hook strength (first sentence engagement)
        hook_indicators = ['?', '!', 'imagine', 'picture', 'consider', 'you']
        hook_count = sum(1 for h in hook_indicators if h in opening)
        features['hook_strength'] = min(hook_count / 3, 1)
        
        return features
    
    def _extract_additional_features(self, text, words, sentences):
        features = {}
        
        # Passive voice ratio (approximate)
        passive_matches = len(self.by_phrase.findall(text))
        features['passive_voice_ratio'] = passive_matches / len(sentences)
        
        # Nominalization rate (words ending in -tion, -ment, -ness, -ity)
        nominalizations = sum(1 for w in words if re.match(r'.*(?:tion|ment|ness|ity)$', w))
        features['nominalization_rate'] = nominalizations / len(words) * 100
        
        # Concrete noun indicators (approximate - proper nouns, specific terms)
        concrete = sum(1 for w in words if len(w) > 6 and w[0].islower())
        features['concrete_noun_ratio'] = concrete / len(words)
        
        # Action verb density
        action_verbs = ['run', 'walk', 'say', 'make', 'do', 'go', 'take', 'give', 'find', 'tell']
        action_count = sum(1 for w in words if w in action_verbs)
        features['action_verb_density'] = action_count / len(words) * 100
        
        # Transition density
        transitions = ['however', 'therefore', 'furthermore', 'moreover', 'additionally',
                      'consequently', 'nevertheless', 'thus', 'hence', 'meanwhile']
        trans_count = sum(1 for t in transitions if t in text.lower())
        features['transition_density'] = trans_count / len(sentences)
        
        return features


def train_enhanced_model():
    """Train model with enhanced features including tone detection."""
    print("=" * 70)
    print("ENHANCED MODEL TRAINING WITH TONE DETECTION")
    print("=" * 70)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    human_texts = []
    ai_texts = []
    
    for item in dataset:
        wiki = item.get('wiki_intro', '')
        gen = item.get('generated_intro', '')
        
        if len(wiki.split()) >= 50:
            human_texts.append(wiki)
        if len(gen.split()) >= 50:
            ai_texts.append(gen)
        
        if len(human_texts) >= 5000 and len(ai_texts) >= 5000:
            break
    
    human_texts = human_texts[:5000]
    ai_texts = ai_texts[:5000]
    
    print(f"Samples: {len(human_texts)} human, {len(ai_texts)} AI")
    
    # Extract features
    extractor = EnhancedFeatureExtractor()
    
    print("\nExtracting features...")
    X_human = []
    X_ai = []
    
    for i, text in enumerate(human_texts):
        if i % 1000 == 0:
            print(f"  Human: {i}/{len(human_texts)}")
        feat = extractor.extract(text)
        if np.sum(np.abs(feat)) > 0:
            X_human.append(feat)
    
    for i, text in enumerate(ai_texts):
        if i % 1000 == 0:
            print(f"  AI: {i}/{len(ai_texts)}")
        feat = extractor.extract(text)
        if np.sum(np.abs(feat)) > 0:
            X_ai.append(feat)
    
    print(f"\nValid: {len(X_human)} human, {len(X_ai)} AI")
    
    # Balance
    min_samples = min(len(X_human), len(X_ai))
    X_human = X_human[:min_samples]
    X_ai = X_ai[:min_samples]
    
    X = np.vstack([X_human, X_ai])
    y = np.array([0] * len(X_human) + [1] * len(X_ai))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("\nTraining Enhanced Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    # Feature importance
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Top 20)")
    print("=" * 70)
    
    importance = list(zip(extractor.feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    
    for name, imp in importance[:20]:
        category = ""
        if name in ['formality_score', 'emotional_intensity', 'sentiment_variance', 'tone_consistency']:
            category = "[TONE]"
        elif name in ['hedge_word_density', 'qualifier_density', 'uncertainty_marker_rate']:
            category = "[HEDGE]"
        elif name in ['first_person_singular', 'first_person_plural', 'opinion_marker_density']:
            category = "[VOICE]"
        elif name in ['passive_voice_ratio', 'nominalization_rate', 'transition_density']:
            category = "[STYLE]"
        print(f"  {name}: {imp*100:.2f}% {category}")
    
    # Cross-validation
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    # Save model
    model_dir = "/workspaces/Veritas/training/models/Enhanced"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "model_name": "Enhanced",
        "version": "1.0.0",
        "description": "Enhanced detector with tone detection and advanced indicators",
        "feature_count": len(extractor.feature_names),
        "feature_names": extractor.feature_names,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "results": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
        },
        "feature_importance": {name: float(imp) for name, imp in importance},
        "new_features": {
            "tone_detection": ["formality_score", "emotional_intensity", "sentiment_variance", "tone_consistency"],
            "hedging_patterns": ["hedge_word_density", "qualifier_density", "uncertainty_marker_rate"],
            "certainty_markers": ["definitive_statement_rate", "absolute_word_density", "assertion_confidence"],
            "personal_voice": ["first_person_singular", "first_person_plural", "personal_anecdote_markers", "opinion_marker_density"],
            "rhetorical": ["question_density", "rhetorical_question_rate", "emphasis_word_density"],
            "coherence": ["topic_drift_score", "lexical_chain_strength"],
            "punctuation": ["comma_rhythm_variance", "parenthetical_usage", "ellipsis_rate"],
            "opening_closing": ["opening_formula_score", "closing_formula_score", "hook_strength"],
        },
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_dir}")
    
    # Compare old vs new features
    print("\n" + "=" * 70)
    print("NEW FEATURE CONTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Group features by category
    categories = {
        'Core (existing)': ['paragraph_uniformity', 'sentence_length_cv', 'type_token_ratio', 'hapax_ratio', 'bigram_entropy', 'trigram_entropy'],
        'Tone Detection': ['formality_score', 'emotional_intensity', 'sentiment_variance', 'tone_consistency'],
        'Hedging Patterns': ['hedge_word_density', 'qualifier_density', 'uncertainty_marker_rate'],
        'Personal Voice': ['first_person_singular', 'first_person_plural', 'opinion_marker_density', 'personal_anecdote_markers'],
        'Rhetorical': ['question_density', 'rhetorical_question_rate', 'emphasis_word_density'],
        'Style Indicators': ['passive_voice_ratio', 'nominalization_rate', 'transition_density'],
    }
    
    imp_dict = dict(importance)
    for cat_name, features in categories.items():
        cat_imp = sum(imp_dict.get(f, 0) for f in features)
        print(f"\n{cat_name}:")
        print(f"  Total importance: {cat_imp*100:.2f}%")
        for f in features:
            if f in imp_dict:
                print(f"    - {f}: {imp_dict[f]*100:.2f}%")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    return model, scaler, extractor, metadata


if __name__ == "__main__":
    train_enhanced_model()
