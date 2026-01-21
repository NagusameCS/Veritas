#!/usr/bin/env python3
"""
VERITAS ML Classifier - Target 99% Accuracy
Uses comprehensive feature extraction with tone analysis
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
with open('large_samples.json', 'r') as f:
    data = json.load(f)

samples = data if isinstance(data, list) else data.get('samples', [])
print(f"Loaded {len(samples)} samples")

# =============================================================================
# COMPREHENSIVE FEATURE EXTRACTION (50+ features with tone analysis)
# =============================================================================

def extract_features(text):
    """Extract 50+ features including tone, style, and linguistic patterns"""
    
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    chars = len(text)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    # Sentence length statistics
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    sent_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    sent_cv = sent_std / avg_sent_len if avg_sent_len > 0 else 0
    
    # Word length statistics
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0
    
    features = {}
    
    # ==========================================================================
    # 1. BASIC STATISTICS
    # ==========================================================================
    features['word_count'] = word_count
    features['char_count'] = chars
    features['sent_count'] = sent_count
    features['avg_sent_len'] = avg_sent_len
    features['sent_cv'] = sent_cv
    features['avg_word_len'] = avg_word_len
    features['chars_per_word'] = chars / word_count
    
    # ==========================================================================
    # 2. PUNCTUATION PATTERNS
    # ==========================================================================
    features['comma_rate'] = len(re.findall(r',', text)) / sent_count
    features['semicolon_rate'] = len(re.findall(r';', text)) / sent_count
    features['colon_rate'] = len(re.findall(r':', text)) / sent_count
    features['exclaim_rate'] = len(re.findall(r'!', text)) / sent_count
    features['question_rate'] = len(re.findall(r'\?', text)) / sent_count
    features['quote_rate'] = len(re.findall(r'"', text)) / word_count
    features['paren_rate'] = len(re.findall(r'[()]', text)) / sent_count
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / sent_count
    
    # ==========================================================================
    # 3. PRONOUN USAGE (Critical for AI detection)
    # ==========================================================================
    features['first_person_rate'] = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', text, re.I)) / word_count
    features['second_person_rate'] = len(re.findall(r'\b(you|your|yours|yourself|yourselves)\b', text, re.I)) / word_count
    features['third_person_rate'] = len(re.findall(r'\b(he|she|they|him|her|them|his|hers|their|theirs|it|its)\b', text, re.I)) / word_count
    
    # ==========================================================================
    # 4. VERB TENSE PATTERNS
    # ==========================================================================
    features['past_tense_rate'] = len(re.findall(r'\b\w+ed\b', text, re.I)) / word_count
    features['present_cont_rate'] = len(re.findall(r'\b\w+ing\b', text, re.I)) / word_count
    features['modal_rate'] = len(re.findall(r'\b(can|could|would|should|might|may|must|will|shall)\b', text, re.I)) / word_count
    features['can_be_rate'] = len(re.findall(r'\b(can|could|may|might) be\b', text, re.I)) / sent_count
    
    # ==========================================================================
    # 5. DISCOURSE MARKERS (AI tends to overuse these)
    # ==========================================================================
    discourse = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly|specifically|essentially|ultimately|primarily)\b', text, re.I))
    features['discourse_markers'] = discourse / word_count
    features['has_discourse'] = 1 if discourse > 0 else 0
    
    # Transition words
    transitions = len(re.findall(r'\b(first|second|third|finally|next|then|also|in addition|on the other hand|in contrast|similarly|likewise)\b', text, re.I))
    features['transition_rate'] = transitions / sent_count
    
    # ==========================================================================
    # 6. CONTRACTIONS (Humans use more)
    # ==========================================================================
    contractions = len(re.findall(r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b", text, re.I))
    features['contraction_rate'] = contractions / word_count
    features['has_contractions'] = 1 if contractions > 0 else 0
    
    # ==========================================================================
    # 7. SENTENCE STARTERS
    # ==========================================================================
    features['sent_start_conj'] = len(re.findall(r'[.!?]\s+(But|And|So|Or|Yet|Because)\s', text, re.I))
    features['sent_start_i'] = len(re.findall(r'[.!?]\s+I\s', text))
    
    # ==========================================================================
    # 8. AI ASSISTANT MARKERS
    # ==========================================================================
    helpful = len(re.findall(r'\b(here is|here are|feel free|I hope this helps|let me|I can help|I\'d be happy|happy to help|sure thing|great question|good question|I understand|as requested|as mentioned)\b', text, re.I))
    features['helpful_phrases'] = helpful
    features['has_helpful'] = 1 if helpful > 0 else 0
    
    instructional = len(re.findall(r'\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|it\'s important to|consider the|you can|you should|you need to|you might want to)\b', text, re.I))
    features['instructional_markers'] = instructional
    
    # "It is important/essential" pattern
    features['it_is_important'] = len(re.findall(r'\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning|recommended)\b', text, re.I))
    
    # ==========================================================================
    # 9. CASUAL/INFORMAL MARKERS (Humans)
    # ==========================================================================
    casual = len(re.findall(r'\b(lol|lmao|haha|hehe|omg|wtf|idk|tbh|imo|imho|ngl|kinda|gonna|wanna|gotta|yeah|nah|yep|nope|ok|okay|cool|awesome|wow|geez|damn|shit|fuck|crap|dude|bro|yo)\b', text, re.I))
    features['casual_markers'] = casual
    features['has_casual'] = 1 if casual > 0 else 0
    
    # Filler words
    fillers = len(re.findall(r'\b(um|uh|like|you know|I mean|basically|literally|actually|honestly|seriously)\b', text, re.I))
    features['filler_words'] = fillers / word_count
    
    # ==========================================================================
    # 10. TECHNICAL/CODE MARKERS (AI)
    # ==========================================================================
    features['has_html'] = 1 if re.search(r'<[a-z]+>', text, re.I) else 0
    features['has_code_blocks'] = 1 if re.search(r'```|<code>|<pre>', text) else 0
    features['has_urls'] = 1 if re.search(r'https?://', text) else 0
    
    # ==========================================================================
    # 11. PROPER NOUNS (Journalism/formal human)
    # ==========================================================================
    proper_nouns = len(re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', text))
    features['proper_noun_rate'] = proper_nouns / word_count
    
    # Quoted speech (journalism)
    features['quoted_speech'] = len(re.findall(r'"[^"]{10,}"', text))
    
    # ==========================================================================
    # 12. TONE ANALYSIS (Critical for accuracy)
    # ==========================================================================
    
    # Positive tone words
    positive = len(re.findall(r'\b(good|great|excellent|amazing|wonderful|fantastic|brilliant|perfect|love|loved|enjoy|enjoyed|happy|glad|pleased|delighted|beautiful|best|favorite|favourite)\b', text, re.I))
    features['positive_tone'] = positive / word_count
    
    # Negative tone words
    negative = len(re.findall(r'\b(bad|terrible|awful|horrible|worst|hate|hated|angry|sad|disappointed|frustrated|annoyed|boring|poor|wrong|fail|failed|problem|issue|difficult)\b', text, re.I))
    features['negative_tone'] = negative / word_count
    
    # Neutral/objective tone (AI tends to be more neutral)
    neutral = len(re.findall(r'\b(is|are|was|were|has|have|had|does|do|did|may|might|could|would|should|appears|seems|tends|generally|typically|usually|often|sometimes|approximately|roughly)\b', text, re.I))
    features['neutral_tone'] = neutral / word_count
    
    # Hedging language (AI hedges more)
    hedging = len(re.findall(r'\b(perhaps|maybe|possibly|probably|likely|unlikely|might|could|may|seems|appears|suggest|indicate|tend to|in general|typically|usually|often|sometimes)\b', text, re.I))
    features['hedging_rate'] = hedging / word_count
    
    # Certainty language (humans more certain/direct)
    certainty = len(re.findall(r'\b(definitely|certainly|absolutely|obviously|clearly|surely|always|never|must|undoubtedly|without doubt|no doubt)\b', text, re.I))
    features['certainty_rate'] = certainty / word_count
    
    # Emotional intensity
    emotional = len(re.findall(r'\b(love|hate|amazing|terrible|incredible|awful|wonderful|horrible|fantastic|disgusting|beautiful|ugly|brilliant|stupid|genius|idiot)\b', text, re.I))
    features['emotional_intensity'] = emotional / word_count
    
    # ==========================================================================
    # 13. FORMALITY INDICATORS
    # ==========================================================================
    
    # Formal vocabulary
    formal = len(re.findall(r'\b(therefore|consequently|furthermore|moreover|nevertheless|notwithstanding|wherein|whereby|herein|therein|aforementioned|henceforth|subsequently|accordingly|respectively)\b', text, re.I))
    features['formal_vocab'] = formal / word_count
    
    # Informal vocabulary
    informal = len(re.findall(r'\b(gonna|wanna|gotta|kinda|sorta|dunno|ain\'t|yeah|nope|yep|ok|okay|stuff|things|guy|guys|kid|kids|pretty|really|very|so|just|like)\b', text, re.I))
    features['informal_vocab'] = informal / word_count
    
    # ==========================================================================
    # 14. STRUCTURAL PATTERNS
    # ==========================================================================
    
    # List patterns (AI loves lists)
    features['bullet_points'] = len(re.findall(r'^\s*[-*•]\s', text, re.M))
    features['numbered_list'] = len(re.findall(r'^\s*\d+[.)]\s', text, re.M))
    
    # Paragraph structure
    paragraphs = text.split('\n\n')
    features['paragraph_count'] = len([p for p in paragraphs if p.strip()])
    
    # ==========================================================================
    # 15. VOCABULARY DIVERSITY
    # ==========================================================================
    unique_words = len(set(w.lower() for w in words))
    features['vocab_diversity'] = unique_words / word_count if word_count > 0 else 0
    
    # Hapax legomena (words appearing once)
    word_freq = Counter(w.lower() for w in words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    features['hapax_ratio'] = hapax / word_count if word_count > 0 else 0
    
    # ==========================================================================
    # 16. AI REFUSAL PATTERNS
    # ==========================================================================
    refusal = len(re.findall(r"I (don't|can't|won't|cannot|shouldn't|am not able to|am unable to) (want to|agree|think|help with|assist|provide|recommend|support)", text, re.I))
    features['refusal_patterns'] = refusal
    features['has_refusal'] = 1 if refusal > 0 else 0
    
    return features

# =============================================================================
# EXTRACT FEATURES FOR ALL SAMPLES
# =============================================================================

print("Extracting features...")
X_data = []
y_data = []
sources = []
valid_count = 0
skipped = 0

for sample in samples:
    text = sample.get('text', '')
    label = sample.get('label', '')
    source = sample.get('source', 'unknown')
    
    if len(text) < 50:  # Skip very short samples
        skipped += 1
        continue
    
    features = extract_features(text)
    if features is None:
        skipped += 1
        continue
    
    X_data.append(list(features.values()))
    y_data.append(1 if label == 'ai' else 0)
    sources.append(source)
    valid_count += 1

print(f"Valid samples: {valid_count}, Skipped: {skipped}")

# Convert to numpy arrays
X = np.array(X_data)
y = np.array(y_data)
feature_names = list(extract_features("Sample text for feature names.").keys())

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {len(feature_names)}")

# Handle any NaN/Inf values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# TRAIN MULTIPLE CLASSIFIERS
# =============================================================================

print("\n" + "="*60)
print("TRAINING CLASSIFIERS")
print("="*60)

classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
}

best_clf = None
best_acc = 0
best_name = ""

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    
    # Train on full training set
    clf.fit(X_train_scaled, y_train)
    
    # Test accuracy
    y_pred = clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy: {test_acc*100:.2f}%")
    
    if test_acc > best_acc:
        best_acc = test_acc
        best_clf = clf
        best_name = name

print("\n" + "="*60)
print(f"BEST MODEL: {best_name} ({best_acc*100:.2f}%)")
print("="*60)

# =============================================================================
# DETAILED ANALYSIS OF BEST MODEL
# =============================================================================

y_pred = best_clf.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Human:  {cm[0][0]:5d} correct, {cm[0][1]:5d} false positive (human→AI)")
print(f"  True AI:     {cm[1][1]:5d} correct, {cm[1][0]:5d} false negative (AI→human)")

# Calculate rates
fp_rate = cm[0][1] / (cm[0][0] + cm[0][1]) * 100
fn_rate = cm[1][0] / (cm[1][0] + cm[1][1]) * 100
print(f"\n  False Positive Rate: {fp_rate:.2f}%")
print(f"  False Negative Rate: {fn_rate:.2f}%")

# =============================================================================
# PER-SOURCE ANALYSIS
# =============================================================================

print("\n" + "="*60)
print("PER-SOURCE ACCURACY")
print("="*60)

source_stats = {}
for i, (true, pred, src) in enumerate(zip(y_test, y_pred, sources_test)):
    if src not in source_stats:
        source_stats[src] = {'correct': 0, 'total': 0, 'label': 'AI' if true == 1 else 'Human'}
    source_stats[src]['total'] += 1
    if true == pred:
        source_stats[src]['correct'] += 1

for src, stats in sorted(source_stats.items(), key=lambda x: x[1]['correct']/x[1]['total']):
    acc = stats['correct'] / stats['total'] * 100
    print(f"  {src:20s}: {acc:5.1f}% ({stats['correct']:4d}/{stats['total']:4d}) [{stats['label']}]")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print("\n" + "="*60)
print("TOP FEATURE IMPORTANCE")
print("="*60)

if hasattr(best_clf, 'feature_importances_'):
    importances = best_clf.feature_importances_
elif hasattr(best_clf, 'coef_'):
    importances = np.abs(best_clf.coef_[0])
else:
    importances = None

if importances is not None:
    feature_imp = list(zip(feature_names, importances))
    feature_imp.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Most Important Features:")
    for name, imp in feature_imp[:20]:
        print(f"  {name:25s}: {imp:.4f}")

# =============================================================================
# SAVE MODEL CONFIG FOR JAVASCRIPT
# =============================================================================

print("\n" + "="*60)
print("GENERATING JAVASCRIPT CONFIG")
print("="*60)

if hasattr(best_clf, 'coef_'):
    # For logistic regression, save coefficients
    config = {
        'model_type': 'logistic_regression',
        'accuracy': float(best_acc),
        'feature_names': feature_names,
        'coefficients': best_clf.coef_[0].tolist(),
        'intercept': float(best_clf.intercept_[0]),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    with open('ml_model_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Model config saved to ml_model_config.json")

print("\n" + "="*60)
print(f"FINAL ACCURACY: {best_acc*100:.2f}%")
print("="*60)
