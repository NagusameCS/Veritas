#!/usr/bin/env python3
"""
VERITAS Ultimate AI Detector - Training Script
Target: 99% accuracy on diverse human/AI/humanized text
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS Ultimate AI Detector - Training")
print("=" * 70)

# =============================================================================
# COMPREHENSIVE FEATURE EXTRACTION (100+ features)
# =============================================================================

def extract_features(text):
    """Extract 100+ linguistic, stylistic, and tone features."""
    if not text or len(text) < 10:
        return None
    
    # Basic tokenization
    words = text.split()
    word_count = len(words) if words else 1
    chars = len(text)
    
    # Sentence detection
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    # Paragraph detection
    paragraphs = text.split('\n\n')
    para_count = len([p for p in paragraphs if p.strip()])
    
    # Word lengths
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0
    word_len_std = np.std(word_lengths) if len(word_lengths) > 1 else 0
    
    # Sentence lengths
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    sent_cv = sent_len_std / avg_sent_len if avg_sent_len > 0 else 0
    
    # Unique words (vocabulary richness)
    unique_words = set(w.lower() for w in words)
    vocab_richness = len(unique_words) / word_count if word_count > 0 else 0
    
    # Hapax legomena (words appearing only once)
    word_freq = Counter(w.lower() for w in words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / word_count if word_count > 0 else 0
    
    features = {}
    
    # =========================================================================
    # 1. PRONOUN PATTERNS (Human vs AI distinctive)
    # =========================================================================
    features['first_person_rate'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['first_person_plural'] = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.I)) / word_count
    features['second_person_rate'] = len(re.findall(r'\b(you|your|yours|yourself|yourselves)\b', text, re.I)) / word_count
    features['third_person_rate'] = len(re.findall(r'\b(he|she|they|him|her|them|his|hers|their|theirs|it|its)\b', text, re.I)) / word_count
    
    # =========================================================================
    # 2. SENTENCE STRUCTURE
    # =========================================================================
    features['avg_sent_len'] = avg_sent_len
    features['sent_len_std'] = sent_len_std
    features['sent_cv'] = sent_cv
    features['max_sent_len'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    features['sent_len_range'] = features['max_sent_len'] - features['min_sent_len']
    
    # Sentence starters
    features['sent_start_conj'] = len(re.findall(r'[.!?]\s+(But|And|So|Or|Yet|Because)\s', text, re.I))
    features['sent_start_however'] = len(re.findall(r'[.!?]\s+(However|Therefore|Moreover|Furthermore)\s', text, re.I))
    features['sent_start_i'] = len(re.findall(r'[.!?]\s+I\s', text))
    
    # =========================================================================
    # 3. PUNCTUATION PATTERNS
    # =========================================================================
    features['comma_rate'] = text.count(',') / sent_count
    features['semicolon_rate'] = text.count(';') / sent_count
    features['colon_rate'] = text.count(':') / sent_count
    features['exclaim_rate'] = text.count('!') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / sent_count
    features['paren_count'] = text.count('(') + text.count(')')
    features['quote_count'] = text.count('"') + text.count("'") + text.count('"') + text.count('"')
    
    # =========================================================================
    # 4. VOCABULARY & WORD PATTERNS
    # =========================================================================
    features['avg_word_len'] = avg_word_len
    features['word_len_std'] = word_len_std
    features['vocab_richness'] = vocab_richness
    features['hapax_ratio'] = hapax_ratio
    
    # Long words (complexity)
    long_words = sum(1 for w in words if len(w) > 10)
    features['long_word_rate'] = long_words / word_count
    
    # Short words
    short_words = sum(1 for w in words if len(w) <= 3)
    features['short_word_rate'] = short_words / word_count
    
    # =========================================================================
    # 5. CONTRACTIONS (Strong human marker)
    # =========================================================================
    contractions = re.findall(r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't|gonna|wanna|gotta|kinda|sorta)\b", text, re.I)
    features['contraction_rate'] = len(contractions) / word_count
    features['contraction_count'] = len(contractions)
    
    # =========================================================================
    # 6. DISCOURSE MARKERS (AI tendency)
    # =========================================================================
    discourse = re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly|subsequently|nonetheless|conversely|alternatively)\b', text, re.I)
    features['discourse_marker_rate'] = len(discourse) / word_count
    features['discourse_count'] = len(discourse)
    
    # =========================================================================
    # 7. AI-SPECIFIC PATTERNS
    # =========================================================================
    # Hedging language
    hedges = re.findall(r'\b(might|may|could|possibly|perhaps|potentially|likely|unlikely|generally|typically|usually|often|sometimes|occasionally|arguably|seemingly|apparently)\b', text, re.I)
    features['hedge_rate'] = len(hedges) / word_count
    
    # Modal verbs
    modals = re.findall(r'\b(can|could|will|would|shall|should|may|might|must)\b', text, re.I)
    features['modal_rate'] = len(modals) / word_count
    
    # "Can be" pattern
    features['can_be_rate'] = len(re.findall(r'\b(can|could|may|might) be\b', text, re.I)) / sent_count
    
    # "It is important/essential" pattern
    features['it_is_important'] = len(re.findall(r'\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning|noteworthy)\b', text, re.I))
    
    # AI assistant phrases
    helpful_phrases = re.findall(r'\b(here is|here are|feel free|I hope this helps|let me|I can help|I\'d be happy|happy to help|sure thing|great question|good question|certainly|absolutely|definitely)\b', text, re.I)
    features['helpful_phrase_count'] = len(helpful_phrases)
    
    # Instructional markers
    instructional = re.findall(r'\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|consider the|it\'s important to|you should|you can|you might)\b', text, re.I)
    features['instructional_count'] = len(instructional)
    
    # =========================================================================
    # 8. HUMAN-SPECIFIC PATTERNS
    # =========================================================================
    # Casual/informal markers
    casual = re.findall(r'\b(lol|lmao|haha|hehe|omg|wtf|idk|tbh|imo|imho|ngl|btw|fyi|smh|tho|tho|rn|af|nvm|yep|yeah|nah|ok|okay|yea|ya|ugh|meh|hmm|oops|wow|whoa|damn|dang|hell|crap|shit|fuck|bro|dude|guys|man|hey|hi|hello|bye|thanks|thx|pls|plz)\b', text, re.I)
    features['casual_marker_count'] = len(casual)
    features['casual_rate'] = len(casual) / word_count
    
    # Emotional expressions
    emotional = re.findall(r'\b(love|hate|amazing|awesome|terrible|horrible|wonderful|fantastic|beautiful|ugly|excited|scared|worried|happy|sad|angry|frustrated|annoyed|surprised|shocked)\b', text, re.I)
    features['emotional_word_rate'] = len(emotional) / word_count
    
    # Personal anecdotes
    features['personal_story'] = len(re.findall(r'\b(I remember|I recall|I once|when I was|my experience|personally|in my opinion|I think|I feel|I believe|I guess|I suppose)\b', text, re.I))
    
    # Filler words
    fillers = re.findall(r'\b(like|you know|I mean|basically|actually|literally|honestly|seriously|obviously|clearly)\b', text, re.I)
    features['filler_rate'] = len(fillers) / word_count
    
    # =========================================================================
    # 9. STRUCTURAL PATTERNS
    # =========================================================================
    # Lists and bullets
    features['has_bullets'] = 1 if re.search(r'^[\s]*[-•*]\s', text, re.M) else 0
    features['has_numbers'] = 1 if re.search(r'^\s*\d+[.)]\s', text, re.M) else 0
    
    # Headers
    features['has_headers'] = 1 if re.search(r'^#+\s|^[A-Z][^.!?]*:$', text, re.M) else 0
    
    # Code/technical content
    features['has_code'] = 1 if re.search(r'```|`[^`]+`|<code>|<pre>|function\s*\(|def\s+\w+|class\s+\w+', text) else 0
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    
    # URLs and references
    features['has_urls'] = 1 if re.search(r'https?://|www\.', text) else 0
    
    # =========================================================================
    # 10. TENSE PATTERNS
    # =========================================================================
    # Past tense (storytelling, human)
    past_tense = re.findall(r'\b\w+ed\b', text)
    features['past_tense_rate'] = len(past_tense) / word_count
    
    # Present tense
    present = re.findall(r'\b(is|are|am|has|have|do|does|go|goes|come|comes|get|gets|make|makes|take|takes|see|sees|know|knows|think|thinks|want|wants|need|needs|feel|feels|seem|seems|look|looks)\b', text, re.I)
    features['present_tense_rate'] = len(present) / word_count
    
    # Future tense
    future = re.findall(r'\b(will|shall|going to|gonna)\b', text, re.I)
    features['future_tense_rate'] = len(future) / word_count
    
    # =========================================================================
    # 11. PROPER NOUNS & NAMED ENTITIES
    # =========================================================================
    # Capitalized words (not at sentence start) - proper nouns
    proper_nouns = re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', text)
    features['proper_noun_rate'] = len(proper_nouns) / word_count
    
    # All-caps words
    all_caps = re.findall(r'\b[A-Z]{2,}\b', text)
    features['all_caps_rate'] = len(all_caps) / word_count
    
    # =========================================================================
    # 12. QUOTE PATTERNS
    # =========================================================================
    # Quoted speech (journalism)
    quoted_speech = re.findall(r'"[^"]{10,}"', text)
    features['quoted_speech_count'] = len(quoted_speech)
    
    # Said/says attribution
    features['attribution_count'] = len(re.findall(r'\b(said|says|told|asked|replied|responded|explained|noted|added|stated|claimed|argued|suggested|warned|announced)\b', text, re.I))
    
    # =========================================================================
    # 13. REPETITION PATTERNS
    # =========================================================================
    # Word repetition (consecutive)
    features['word_repetition'] = len(re.findall(r'\b(\w+)\s+\1\b', text, re.I))
    
    # Phrase repetition (AI tendency for consistency)
    bigrams = [' '.join(words[i:i+2]).lower() for i in range(len(words)-1)]
    bigram_freq = Counter(bigrams)
    repeated_bigrams = sum(1 for b, c in bigram_freq.items() if c > 1)
    features['repeated_bigram_rate'] = repeated_bigrams / len(bigrams) if bigrams else 0
    
    # =========================================================================
    # 14. TONE MARKERS
    # =========================================================================
    # Formal tone
    formal = re.findall(r'\b(therefore|hence|thus|consequently|furthermore|moreover|additionally|subsequently|regarding|concerning|pertaining|hereby|hereafter|wherein|whereby|therein|thereof|notwithstanding|aforementioned|heretofore)\b', text, re.I)
    features['formal_tone_rate'] = len(formal) / word_count
    
    # Informal tone
    informal = re.findall(r'\b(gonna|wanna|gotta|kinda|sorta|yeah|yep|nope|nah|okay|ok|hey|hi|bye|thanks|cool|awesome|great|nice|sweet|sick|dope|lit|fire|bruh|fam)\b', text, re.I)
    features['informal_tone_rate'] = len(informal) / word_count
    
    # Confident tone
    confident = re.findall(r'\b(certainly|definitely|absolutely|clearly|obviously|undoubtedly|unquestionably|without doubt|no doubt|surely|indeed|in fact|actually)\b', text, re.I)
    features['confident_tone_rate'] = len(confident) / word_count
    
    # Uncertain tone
    uncertain = re.findall(r'\b(maybe|perhaps|possibly|probably|might|could|seemingly|apparently|supposedly|allegedly|presumably|reportedly)\b', text, re.I)
    features['uncertain_tone_rate'] = len(uncertain) / word_count
    
    # =========================================================================
    # 15. COMPLEXITY METRICS
    # =========================================================================
    # Syllable estimation (rough)
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
    
    syllables = sum(count_syllables(w) for w in words)
    features['avg_syllables'] = syllables / word_count if word_count > 0 else 0
    
    # Flesch Reading Ease approximation
    if sent_count > 0 and word_count > 0:
        features['flesch_approx'] = 206.835 - 1.015 * (word_count / sent_count) - 84.6 * (syllables / word_count)
    else:
        features['flesch_approx'] = 0
    
    # =========================================================================
    # 16. SPECIAL PATTERNS
    # =========================================================================
    # Emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    features['emoji_count'] = len(emoji_pattern.findall(text))
    
    # Numbers
    features['number_count'] = len(re.findall(r'\b\d+\b', text))
    features['number_rate'] = features['number_count'] / word_count
    
    # Acronyms
    features['acronym_count'] = len(re.findall(r'\b[A-Z]{2,5}\b', text))
    
    # =========================================================================
    # 17. AI REFUSAL PATTERNS
    # =========================================================================
    features['refusal_pattern'] = 1 if re.search(r"I (can't|cannot|won't|don't|shouldn't) (help|assist|provide|generate|create|write)", text, re.I) else 0
    features['safety_language'] = len(re.findall(r'\b(harmful|inappropriate|offensive|unethical|illegal|dangerous|sensitive|controversial)\b', text, re.I))
    
    # =========================================================================
    # 18. HUMANIZED AI DETECTION PATTERNS
    # =========================================================================
    # Artificial informality (too casual for context)
    features['forced_casual'] = len(re.findall(r'\b(honestly|basically|literally|actually|seriously)\b', text, re.I)) / word_count
    
    # Inconsistent formality
    formal_count = len(formal)
    informal_count = len(informal)
    features['formality_inconsistency'] = abs(formal_count - informal_count) / (formal_count + informal_count + 1)
    
    # Unnatural contractions (AI trying to seem human)
    features['contraction_density'] = features['contraction_count'] / (para_count + 1)
    
    # Text length
    features['text_length'] = chars
    features['word_count'] = word_count
    features['sent_count'] = sent_count
    features['para_count'] = para_count
    
    return features


# =============================================================================
# LOAD AND PROCESS DATA
# =============================================================================

print("\nLoading dataset...")
with open('large_samples.json', 'r') as f:
    data = json.load(f)

samples = data if isinstance(data, list) else data.get('samples', [])
print(f"Total samples: {len(samples)}")

# Extract features for all samples
print("\nExtracting features from all samples...")
X_data = []
y_data = []
sources = []
failed = 0

for i, sample in enumerate(samples):
    if i % 10000 == 0:
        print(f"  Processing {i}/{len(samples)}...")
    
    text = sample.get('text', '')
    label = sample.get('label', '')
    source = sample.get('source', 'unknown')
    
    if not text or not label:
        continue
    
    features = extract_features(text)
    if features is None:
        failed += 1
        continue
    
    X_data.append(list(features.values()))
    y_data.append(1 if label == 'ai' else 0)
    sources.append(source)

feature_names = list(extract_features("Sample text for feature names.").keys())
print(f"\nExtracted {len(X_data)} samples with {len(feature_names)} features")
print(f"Failed to extract: {failed}")

X = np.array(X_data)
y = np.array(y_data)

# Handle NaN/Inf
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

print(f"\nLabel distribution:")
print(f"  Human: {sum(y == 0)}")
print(f"  AI: {sum(y == 1)}")

# =============================================================================
# TRAIN ENSEMBLE MODEL
# =============================================================================

print("\n" + "=" * 70)
print("Training Ensemble Model")
print("=" * 70)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)}")
print(f"Test set: {len(X_test)}")

# Create scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
print("\nTraining individual models...")

# 1. Gradient Boosting (our strongest)
print("  Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42,
    verbose=0
)
gb.fit(X_train_scaled, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test_scaled))
print(f"    Gradient Boosting accuracy: {gb_acc:.4f}")

# 2. Random Forest
print("  Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_scaled))
print(f"    Random Forest accuracy: {rf_acc:.4f}")

# 3. Logistic Regression
print("  Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"    Logistic Regression accuracy: {lr_acc:.4f}")

# 4. Voting Ensemble
print("  Creating Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('gb', gb),
        ('rf', rf),
        ('lr', lr)
    ],
    voting='soft',
    weights=[3, 2, 1]  # Weight GB higher
)
ensemble.fit(X_train_scaled, y_train)
ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test_scaled))
print(f"    Ensemble accuracy: {ensemble_acc:.4f}")

# =============================================================================
# DETAILED EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("DETAILED EVALUATION")
print("=" * 70)

# Use best model
best_model = ensemble if ensemble_acc >= max(gb_acc, rf_acc, lr_acc) else gb
best_name = "Ensemble" if ensemble_acc >= max(gb_acc, rf_acc, lr_acc) else "Gradient Boosting"
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

print(f"\nBest Model: {best_name}")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Human, Pred Human: {cm[0][0]}")
print(f"  True Human, Pred AI:    {cm[0][1]} (False Positive)")
print(f"  True AI, Pred Human:    {cm[1][0]} (False Negative)")
print(f"  True AI, Pred AI:       {cm[1][1]}")

# Accuracy by source
print("\nAccuracy by Source:")
source_unique = list(set(sources_test))
source_results = {}
for source in sorted(source_unique):
    mask = [s == source for s in sources_test]
    if sum(mask) > 0:
        acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = "AI" if y_test[mask].mean() > 0.5 else "Human"
        source_results[source] = {'accuracy': acc, 'count': count, 'label': label}
        print(f"  {source:20s}: {acc:.1%} ({count:5d} samples) [{label}]")

# Cross-validation for robust estimate
print("\n" + "=" * 70)
print("CROSS-VALIDATION (5-fold)")
print("=" * 70)

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print("\n" + "=" * 70)
print("TOP FEATURES")
print("=" * 70)

if hasattr(gb, 'feature_importances_'):
    importance = gb.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    print("\nTop 20 Most Important Features:")
    for i in range(min(20, len(feature_names))):
        idx = indices[i]
        print(f"  {i+1:2d}. {feature_names[idx]:30s}: {importance[idx]:.4f}")

# =============================================================================
# SAVE MODEL
# =============================================================================

print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_data = {
    'model': best_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'accuracy': accuracy_score(y_test, y_pred),
    'source_results': source_results
}

with open('veritas_detector.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to 'veritas_detector.pkl'")
print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.2%}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)
print(f"""
Model: {best_name}
Features: {len(feature_names)}
Training samples: {len(X_train)}
Test samples: {len(X_test)}
Test Accuracy: {accuracy_score(y_test, y_pred):.2%}
CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})

Human Accuracy: {cm[0][0]/(cm[0][0]+cm[0][1]):.2%}
AI Accuracy: {cm[1][1]/(cm[1][0]+cm[1][1]):.2%}
""")
