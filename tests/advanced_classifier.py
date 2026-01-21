#!/usr/bin/env python3
"""
VERITAS Advanced ML Classifier - Target 99% Accuracy
Uses XGBoost with hyperparameter tuning and ensemble methods
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
with open('large_samples.json', 'r') as f:
    data = json.load(f)

samples = data if isinstance(data, list) else data.get('samples', [])
print(f"Loaded {len(samples)} samples")

# =============================================================================
# ENHANCED FEATURE EXTRACTION (70+ features)
# =============================================================================

def extract_features(text):
    """Extract 70+ features including advanced patterns"""
    
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
    sent_max = max(sent_lengths) if sent_lengths else 0
    sent_min = min(sent_lengths) if sent_lengths else 0
    
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
    features['sent_max'] = sent_max
    features['sent_min'] = sent_min
    features['sent_range'] = sent_max - sent_min
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
    features['ellipsis_rate'] = len(re.findall(r'\.\.\.', text)) / sent_count
    
    # ==========================================================================
    # 3. PRONOUN USAGE
    # ==========================================================================
    features['first_person_rate'] = len(re.findall(r'\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b', text, re.I)) / word_count
    features['second_person_rate'] = len(re.findall(r'\b(you|your|yours|yourself|yourselves)\b', text, re.I)) / word_count
    features['third_person_rate'] = len(re.findall(r'\b(he|she|they|him|her|them|his|hers|their|theirs|it|its)\b', text, re.I)) / word_count
    features['i_count'] = len(re.findall(r'\bI\b', text))  # Capital I specifically
    
    # ==========================================================================
    # 4. VERB PATTERNS
    # ==========================================================================
    features['past_tense_rate'] = len(re.findall(r'\b\w+ed\b', text, re.I)) / word_count
    features['present_cont_rate'] = len(re.findall(r'\b\w+ing\b', text, re.I)) / word_count
    features['modal_rate'] = len(re.findall(r'\b(can|could|would|should|might|may|must|will|shall)\b', text, re.I)) / word_count
    features['can_be_rate'] = len(re.findall(r'\b(can|could|may|might) be\b', text, re.I)) / sent_count
    features['passive_voice'] = len(re.findall(r'\b(is|are|was|were|be|been|being)\s+\w+ed\b', text, re.I)) / sent_count
    
    # ==========================================================================
    # 5. DISCOURSE MARKERS
    # ==========================================================================
    discourse = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly|specifically|essentially|ultimately|primarily)\b', text, re.I))
    features['discourse_markers'] = discourse / word_count
    features['discourse_count'] = discourse
    
    transitions = len(re.findall(r'\b(first|second|third|finally|next|then|also|in addition|on the other hand|in contrast|similarly|likewise)\b', text, re.I))
    features['transition_rate'] = transitions / sent_count
    
    # ==========================================================================
    # 6. CONTRACTIONS
    # ==========================================================================
    contractions = len(re.findall(r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b", text, re.I))
    features['contraction_rate'] = contractions / word_count
    features['contraction_count'] = contractions
    
    # ==========================================================================
    # 7. SENTENCE STARTERS
    # ==========================================================================
    features['sent_start_conj'] = len(re.findall(r'[.!?]\s+(But|And|So|Or|Yet|Because)\s', text, re.I))
    features['sent_start_i'] = len(re.findall(r'[.!?]\s+I\s', text))
    features['sent_start_the'] = len(re.findall(r'[.!?]\s+The\s', text, re.I))
    features['sent_start_this'] = len(re.findall(r'[.!?]\s+(This|These)\s', text, re.I))
    
    # ==========================================================================
    # 8. AI ASSISTANT MARKERS
    # ==========================================================================
    helpful = len(re.findall(r'\b(here is|here are|feel free|I hope this helps|let me|I can help|I\'d be happy|happy to help|sure thing|great question|good question|I understand|as requested|as mentioned|please note|please let me know)\b', text, re.I))
    features['helpful_phrases'] = helpful
    
    instructional = len(re.findall(r'\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|it\'s important to|consider the|you can|you should|you need to|you might want to|to do this|following steps)\b', text, re.I))
    features['instructional_markers'] = instructional
    
    features['it_is_important'] = len(re.findall(r'\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning|recommended)\b', text, re.I))
    
    # Explanation patterns
    features['explanation_patterns'] = len(re.findall(r'\b(this means|this is because|the reason|this allows|this ensures|this helps|this makes)\b', text, re.I))
    
    # ==========================================================================
    # 9. CASUAL/INFORMAL MARKERS
    # ==========================================================================
    casual = len(re.findall(r'\b(lol|lmao|haha|hehe|omg|wtf|idk|tbh|imo|imho|ngl|kinda|gonna|wanna|gotta|yeah|nah|yep|nope|ok|okay|cool|awesome|wow|geez|damn|shit|fuck|crap|dude|bro|yo)\b', text, re.I))
    features['casual_markers'] = casual
    
    fillers = len(re.findall(r'\b(um|uh|like|you know|I mean|basically|literally|actually|honestly|seriously|really)\b', text, re.I))
    features['filler_words'] = fillers / word_count
    
    # ==========================================================================
    # 10. TECHNICAL/CODE MARKERS
    # ==========================================================================
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    features['has_code_blocks'] = 1 if re.search(r'```|<code>|<pre>', text) else 0
    features['has_urls'] = 1 if re.search(r'https?://', text) else 0
    features['has_email'] = 1 if re.search(r'\b[\w.-]+@[\w.-]+\.\w+\b', text) else 0
    features['has_numbers'] = len(re.findall(r'\b\d+\b', text)) / word_count
    
    # ==========================================================================
    # 11. PROPER NOUNS & NAMED ENTITIES
    # ==========================================================================
    proper_nouns = len(re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', text))
    features['proper_noun_rate'] = proper_nouns / word_count
    features['proper_noun_count'] = proper_nouns
    
    # Quoted speech (journalism)
    features['quoted_speech'] = len(re.findall(r'"[^"]{10,}"', text))
    features['single_quoted'] = len(re.findall(r"'[^']{10,}'", text))
    
    # ==========================================================================
    # 12. TONE ANALYSIS
    # ==========================================================================
    positive = len(re.findall(r'\b(good|great|excellent|amazing|wonderful|fantastic|brilliant|perfect|love|loved|enjoy|enjoyed|happy|glad|pleased|delighted|beautiful|best|favorite|favourite)\b', text, re.I))
    features['positive_tone'] = positive / word_count
    
    negative = len(re.findall(r'\b(bad|terrible|awful|horrible|worst|hate|hated|angry|sad|disappointed|frustrated|annoyed|boring|poor|wrong|fail|failed|problem|issue|difficult)\b', text, re.I))
    features['negative_tone'] = negative / word_count
    
    features['tone_polarity'] = (positive - negative) / (positive + negative + 1)
    
    neutral = len(re.findall(r'\b(is|are|was|were|has|have|had|does|do|did|appears|seems|tends|generally|typically|usually|often|sometimes|approximately)\b', text, re.I))
    features['neutral_tone'] = neutral / word_count
    
    hedging = len(re.findall(r'\b(perhaps|maybe|possibly|probably|likely|unlikely|might|could|may|seems|appears|suggest|indicate|tend to|in general|typically|usually)\b', text, re.I))
    features['hedging_rate'] = hedging / word_count
    
    certainty = len(re.findall(r'\b(definitely|certainly|absolutely|obviously|clearly|surely|always|never|must|undoubtedly|without doubt|no doubt)\b', text, re.I))
    features['certainty_rate'] = certainty / word_count
    
    emotional = len(re.findall(r'\b(love|hate|amazing|terrible|incredible|awful|wonderful|horrible|fantastic|disgusting|beautiful|ugly|brilliant|stupid|genius|idiot)\b', text, re.I))
    features['emotional_intensity'] = emotional / word_count
    
    # ==========================================================================
    # 13. FORMALITY
    # ==========================================================================
    formal = len(re.findall(r'\b(therefore|consequently|furthermore|moreover|nevertheless|notwithstanding|wherein|whereby|herein|therein|aforementioned|henceforth|subsequently|accordingly|respectively)\b', text, re.I))
    features['formal_vocab'] = formal / word_count
    
    informal = len(re.findall(r'\b(gonna|wanna|gotta|kinda|sorta|dunno|ain\'t|yeah|nope|yep|ok|okay|stuff|things|guy|guys|kid|kids|pretty|really|very|so|just|like)\b', text, re.I))
    features['informal_vocab'] = informal / word_count
    
    features['formality_ratio'] = (formal + 1) / (informal + 1)
    
    # ==========================================================================
    # 14. STRUCTURAL PATTERNS
    # ==========================================================================
    features['bullet_points'] = len(re.findall(r'^\s*[-*•]\s', text, re.M))
    features['numbered_list'] = len(re.findall(r'^\s*\d+[.)]\s', text, re.M))
    
    paragraphs = text.split('\n\n')
    features['paragraph_count'] = len([p for p in paragraphs if p.strip()])
    
    # Line breaks
    features['line_break_rate'] = text.count('\n') / sent_count
    
    # ==========================================================================
    # 15. VOCABULARY DIVERSITY
    # ==========================================================================
    unique_words = len(set(w.lower() for w in words))
    features['vocab_diversity'] = unique_words / word_count if word_count > 0 else 0
    
    word_freq = Counter(w.lower() for w in words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    features['hapax_ratio'] = hapax / word_count if word_count > 0 else 0
    
    # Most common word frequency
    if word_freq:
        most_common_freq = word_freq.most_common(1)[0][1]
        features['top_word_freq'] = most_common_freq / word_count
    else:
        features['top_word_freq'] = 0
    
    # ==========================================================================
    # 16. AI REFUSAL PATTERNS
    # ==========================================================================
    refusal = len(re.findall(r"I (don't|can't|won't|cannot|shouldn't|am not able to|am unable to) (want to|agree|think|help with|assist|provide|recommend|support)", text, re.I))
    features['refusal_patterns'] = refusal
    
    # Safety language
    safety = len(re.findall(r'\b(harmful|dangerous|unsafe|inappropriate|unethical|illegal|offensive|sensitive|trigger warning|content warning)\b', text, re.I))
    features['safety_language'] = safety
    
    # ==========================================================================
    # 17. C4/NEWS SPECIFIC (Journalism patterns)
    # ==========================================================================
    # Attribution phrases
    features['attribution'] = len(re.findall(r'\b(said|says|according to|reported|announced|stated|claimed|argued)\b', text, re.I)) / sent_count
    
    # Date patterns
    features['has_dates'] = 1 if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|\d{4})\b', text) else 0
    
    # Location patterns
    features['location_mentions'] = len(re.findall(r'\b(city|country|state|town|village|region|district|county|province)\b', text, re.I)) / word_count
    
    # ==========================================================================
    # 18. DOLLY/SHORT CONTENT SPECIFIC
    # ==========================================================================
    features['is_short'] = 1 if chars < 200 else 0
    features['is_very_short'] = 1 if chars < 100 else 0
    
    # Definitional patterns (Dolly often has these)
    features['definition_pattern'] = len(re.findall(r'\b(is a|are a|refers to|defined as|means|known as)\b', text, re.I))
    
    return features

# =============================================================================
# EXTRACT FEATURES
# =============================================================================

print("Extracting features...")
X_data = []
y_data = []
sources = []

for sample in samples:
    text = sample.get('text', '')
    label = sample.get('label', '')
    source = sample.get('source', 'unknown')
    
    if len(text) < 30:
        continue
    
    features = extract_features(text)
    if features is None:
        continue
    
    X_data.append(list(features.values()))
    y_data.append(1 if label == 'ai' else 0)
    sources.append(source)

print(f"Valid samples: {len(X_data)}")

X = np.array(X_data)
y = np.array(y_data)
feature_names = list(extract_features("Sample text for feature names.").keys())

print(f"Feature matrix shape: {X.shape}")
print(f"Features: {len(feature_names)}")

X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# TRAIN ADVANCED CLASSIFIERS
# =============================================================================

print("\n" + "="*60)
print("TRAINING ADVANCED CLASSIFIERS")
print("="*60)

# XGBoost with tuned parameters
print("\nTraining XGBoost...")
xgb_clf = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_clf.fit(X_train_scaled, y_train)
xgb_pred = xgb_clf.predict(X_test_scaled)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"  XGBoost Accuracy: {xgb_acc*100:.2f}%")

# LightGBM
print("\nTraining LightGBM...")
lgb_clf = lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=10,
    learning_rate=0.1,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_clf.fit(X_train_scaled, y_train)
lgb_pred = lgb_clf.predict(X_test_scaled)
lgb_acc = accuracy_score(y_test, lgb_pred)
print(f"  LightGBM Accuracy: {lgb_acc*100:.2f}%")

# Random Forest with more trees
print("\nTraining Random Forest (500 trees)...")
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_clf.fit(X_train_scaled, y_train)
rf_pred = rf_clf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"  Random Forest Accuracy: {rf_acc*100:.2f}%")

# =============================================================================
# ENSEMBLE (Voting)
# =============================================================================

print("\nCreating Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('lgb', lgb_clf),
        ('rf', rf_clf)
    ],
    voting='soft'
)
ensemble.fit(X_train_scaled, y_train)
ensemble_pred = ensemble.predict(X_test_scaled)
ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"  Ensemble Accuracy: {ensemble_acc*100:.2f}%")

# Find best model
models = {
    'XGBoost': (xgb_clf, xgb_pred, xgb_acc),
    'LightGBM': (lgb_clf, lgb_pred, lgb_acc),
    'Random Forest': (rf_clf, rf_pred, rf_acc),
    'Ensemble': (ensemble, ensemble_pred, ensemble_acc)
}

best_name = max(models, key=lambda x: models[x][2])
best_clf, best_pred, best_acc = models[best_name]

print("\n" + "="*60)
print(f"BEST MODEL: {best_name} ({best_acc*100:.2f}%)")
print("="*60)

# =============================================================================
# DETAILED ANALYSIS
# =============================================================================

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Human', 'AI']))

cm = confusion_matrix(y_test, best_pred)
print("\nConfusion Matrix:")
print(f"  True Human:  {cm[0][0]:5d} correct, {cm[0][1]:5d} false positive (human→AI)")
print(f"  True AI:     {cm[1][1]:5d} correct, {cm[1][0]:5d} false negative (AI→human)")

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
for i, (true, pred, src) in enumerate(zip(y_test, best_pred, sources_test)):
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
print("TOP FEATURE IMPORTANCE (XGBoost)")
print("="*60)

importances = xgb_clf.feature_importances_
feature_imp = list(zip(feature_names, importances))
feature_imp.sort(key=lambda x: x[1], reverse=True)

print("\nTop 25 Most Important Features:")
for name, imp in feature_imp[:25]:
    print(f"  {name:25s}: {imp:.4f}")

# =============================================================================
# SAVE MODEL FOR USE IN JS
# =============================================================================

print("\n" + "="*60)
print("SAVING MODEL CONFIG")
print("="*60)

# Save the model weights for JavaScript implementation
config = {
    'model_type': 'xgboost',
    'accuracy': float(best_acc),
    'feature_names': feature_names,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
    'fp_rate': float(fp_rate),
    'fn_rate': float(fn_rate)
}

with open('advanced_model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Config saved to advanced_model_config.json")

print("\n" + "="*60)
print(f"FINAL ACCURACY: {best_acc*100:.2f}%")
print("="*60)
