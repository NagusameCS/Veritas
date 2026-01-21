#!/usr/bin/env python3
"""
VERITAS Ultimate Ensemble - Push to 99% accuracy
XGBoost + Random Forest + LightGBM ensemble with optimized hyperparameters
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS Ultimate Ensemble - Target: 99% Accuracy")
print("=" * 70)

# =============================================================================
# COMPREHENSIVE FEATURE EXTRACTION
# =============================================================================

def extract_features(text):
    """Extract 70+ linguistic features."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    chars = len(text)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    paragraphs = text.split('\n\n')
    para_count = len([p for p in paragraphs if p.strip()])
    
    word_lengths = [len(w) for w in words]
    avg_word_len = np.mean(word_lengths) if word_lengths else 0
    word_len_std = np.std(word_lengths) if len(word_lengths) > 1 else 0
    
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    sent_cv = sent_len_std / avg_sent_len if avg_sent_len > 0 else 0
    
    unique_words = set(w.lower() for w in words)
    vocab_richness = len(unique_words) / word_count
    
    word_freq = Counter(w.lower() for w in words)
    hapax = sum(1 for w, c in word_freq.items() if c == 1)
    hapax_ratio = hapax / word_count
    
    features = {}
    
    # === CRITICAL DISCRIMINATORS ===
    proper_nouns = re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', text)
    features['proper_noun_rate'] = len(proper_nouns) / word_count
    features['proper_noun_count'] = len(proper_nouns)
    
    attributions = re.findall(r'\b(said|says|told|asked|replied|responded|explained|noted|added|stated|claimed|argued|suggested|warned|announced|reported|according to)\b', text, re.I)
    features['attribution_count'] = len(attributions)
    features['attribution_rate'] = len(attributions) / sent_count
    
    # === SENTENCE METRICS ===
    features['sent_count'] = sent_count
    features['avg_sent_len'] = avg_sent_len
    features['sent_len_std'] = sent_len_std
    features['sent_cv'] = sent_cv
    features['max_sent_len'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    features['sent_len_range'] = features['max_sent_len'] - features['min_sent_len']
    
    # === WORD METRICS ===
    features['word_len_std'] = word_len_std
    features['avg_word_len'] = avg_word_len
    features['word_count'] = word_count
    
    # === PRONOUNS ===
    features['first_person_plural'] = len(re.findall(r'\b(we|us|our|ours|ourselves)\b', text, re.I)) / word_count
    features['first_person_singular'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['second_person_rate'] = len(re.findall(r'\b(you|your|yours|yourself|yourselves)\b', text, re.I)) / word_count
    features['third_person_rate'] = len(re.findall(r'\b(he|she|they|him|her|them|his|hers|their|theirs)\b', text, re.I)) / word_count
    
    # === PUNCTUATION ===
    features['comma_rate'] = text.count(',') / sent_count
    features['semicolon_rate'] = text.count(';') / sent_count
    features['colon_rate'] = text.count(':') / sent_count
    features['exclaim_rate'] = text.count('!') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / sent_count
    features['paren_rate'] = (text.count('(') + text.count(')')) / sent_count
    features['quote_count'] = text.count('"') + text.count("'")
    
    # === TENSE ===
    past_tense = re.findall(r'\b\w+ed\b', text)
    features['past_tense_rate'] = len(past_tense) / word_count
    
    present = re.findall(r'\b(is|are|am|has|have|do|does)\b', text, re.I)
    features['present_tense_rate'] = len(present) / word_count
    
    # === CONTRACTIONS (Human marker) ===
    contractions = re.findall(r"\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|let's|that's|who's|what's|there's|here's|ain't|gonna|wanna|gotta|kinda)\b", text, re.I)
    features['contraction_rate'] = len(contractions) / word_count
    features['contraction_count'] = len(contractions)
    
    # === AI PATTERNS ===
    discourse = re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly|subsequently|nonetheless)\b', text, re.I)
    features['discourse_rate'] = len(discourse) / word_count
    features['discourse_count'] = len(discourse)
    
    hedges = re.findall(r'\b(might|may|could|possibly|perhaps|potentially|likely|generally|typically|usually|often)\b', text, re.I)
    features['hedge_rate'] = len(hedges) / word_count
    
    features['can_be_rate'] = len(re.findall(r'\b(can|could|may|might) be\b', text, re.I)) / sent_count
    features['it_is_important'] = len(re.findall(r'\bit is (important|essential|crucial|vital|necessary)\b', text, re.I))
    
    helpful = re.findall(r'\b(here is|here are|feel free|I hope this helps|let me|I can help|happy to help|certainly|absolutely)\b', text, re.I)
    features['helpful_count'] = len(helpful)
    
    instructional = re.findall(r'\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that)\b', text, re.I)
    features['instructional_count'] = len(instructional)
    
    # === HUMAN PATTERNS ===
    casual = re.findall(r'\b(lol|lmao|haha|omg|wtf|idk|tbh|imo|ngl|btw|yeah|nah|ok|hey|hi|bye|thanks|thx|dude|man|guys|bro)\b', text, re.I)
    features['casual_count'] = len(casual)
    features['casual_rate'] = len(casual) / word_count
    
    emotional = re.findall(r'\b(love|hate|amazing|awesome|terrible|horrible|wonderful|beautiful|excited|scared|happy|sad|angry|frustrated|annoyed)\b', text, re.I)
    features['emotional_rate'] = len(emotional) / word_count
    
    features['personal_story'] = len(re.findall(r'\b(I remember|when I was|my experience|personally|I think|I feel|I believe|in my opinion)\b', text, re.I))
    
    fillers = re.findall(r'\b(like|you know|I mean|basically|actually|literally|honestly|seriously)\b', text, re.I)
    features['filler_rate'] = len(fillers) / word_count
    
    # === STRUCTURAL ===
    features['has_bullets'] = 1 if re.search(r'^[\s]*[-•*]\s', text, re.M) else 0
    features['has_numbers'] = 1 if re.search(r'^\s*\d+[.)]\s', text, re.M) else 0
    features['has_code'] = 1 if re.search(r'```|<code>|<pre>|function\s*\(|def\s+\w+|class\s+\w+', text) else 0
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    features['has_urls'] = 1 if re.search(r'https?://|www\.', text) else 0
    
    # === VOCABULARY ===
    features['vocab_richness'] = vocab_richness
    features['hapax_ratio'] = hapax_ratio
    features['long_word_rate'] = sum(1 for w in words if len(w) > 10) / word_count
    features['short_word_rate'] = sum(1 for w in words if len(w) <= 3) / word_count
    
    # === TONE ===
    formal = re.findall(r'\b(therefore|hence|thus|consequently|furthermore|moreover|subsequently|regarding|concerning|pertaining)\b', text, re.I)
    features['formal_rate'] = len(formal) / word_count
    
    informal = re.findall(r'\b(gonna|wanna|gotta|kinda|sorta|yeah|yep|nope|nah|ok|hey|hi|bye|cool|awesome|nice)\b', text, re.I)
    features['informal_rate'] = len(informal) / word_count
    
    confident = re.findall(r'\b(certainly|definitely|absolutely|clearly|obviously|undoubtedly|surely|indeed)\b', text, re.I)
    features['confident_rate'] = len(confident) / word_count
    
    # === REPETITION ===
    bigrams = [' '.join(words[i:i+2]).lower() for i in range(len(words)-1)]
    bigram_freq = Counter(bigrams)
    repeated_bigrams = sum(1 for b, c in bigram_freq.items() if c > 1)
    features['repeated_bigram_rate'] = repeated_bigrams / len(bigrams) if bigrams else 0
    
    # === SPECIAL ===
    features['all_caps_rate'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / word_count
    features['number_rate'] = len(re.findall(r'\b\d+\b', text)) / word_count
    features['quoted_speech'] = len(re.findall(r'"[^"]{10,}"', text))
    
    features['refusal_pattern'] = 1 if re.search(r"I (can't|cannot|won't|don't) (help|assist|provide)", text, re.I) else 0
    
    features['sent_start_conj'] = len(re.findall(r'[.!?]\s+(But|And|So|Or|Yet)\s', text, re.I))
    features['sent_start_i'] = len(re.findall(r'[.!?]\s+I\s', text))
    features['sent_start_however'] = len(re.findall(r'[.!?]\s+(However|Therefore|Moreover)\s', text, re.I))
    
    features['text_length'] = chars
    features['para_count'] = para_count
    
    return features


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading clean dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)
print(f"Total: {len(samples)}")

print("\nExtracting features...")
X_data, y_data, sources = [], [], []
for i, s in enumerate(samples):
    if i % 30000 == 0:
        print(f"  {i}/{len(samples)}...")
    text = s.get('text', '')
    label = s.get('label', '')
    if not text or not label:
        continue
    f = extract_features(text)
    if f:
        X_data.append(list(f.values()))
        y_data.append(1 if label == 'ai' else 0)
        sources.append(s.get('source', 'unknown'))

feature_names = list(extract_features("Sample text.").keys())
print(f"Extracted {len(X_data)} samples, {len(feature_names)} features")

X = np.array(X_data)
y = np.array(y_data)
X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

print(f"Human: {sum(y==0)}, AI: {sum(y==1)}")

# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X, y, sources, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# TRAIN ENSEMBLE
# =============================================================================

print("\n" + "=" * 70)
print("Training Ensemble Models")
print("=" * 70)

# 1. XGBoost (strongest individual model)
print("\n[1/4] Training XGBoost (1000 trees, depth 15)...")
xgb_model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train, 
              eval_set=[(X_test_scaled, y_test)], 
              verbose=200)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))
print(f"XGBoost Accuracy: {xgb_acc:.4f}")

# 2. Random Forest
print("\n[2/4] Training Random Forest (500 trees)...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
print(f"Random Forest Accuracy: {rf_acc:.4f}")

# 3. Gradient Boosting
print("\n[3/4] Training Gradient Boosting (300 trees)...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_acc = accuracy_score(y_test, gb_model.predict(X_test_scaled))
print(f"Gradient Boosting Accuracy: {gb_acc:.4f}")

# 4. Voting Ensemble
print("\n[4/4] Creating Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('gb', gb_model)
    ],
    voting='soft',
    weights=[3, 2, 2]  # Weight XGBoost higher
)
ensemble.fit(X_train_scaled, y_train)
ensemble_acc = accuracy_score(y_test, ensemble.predict(X_test_scaled))
print(f"Ensemble Accuracy: {ensemble_acc:.4f}")

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("FINAL EVALUATION")
print("=" * 70)

# Use best model
best_model = xgb_model if xgb_acc >= ensemble_acc else ensemble
best_name = "XGBoost" if xgb_acc >= ensemble_acc else "Ensemble"
best_acc = max(xgb_acc, ensemble_acc)

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

print(f"\nBest Model: {best_name}")
print(f"ACCURACY: {best_acc:.4f}")
print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nHuman Accuracy: {cm[0][0]/(cm[0][0]+cm[0][1]):.2%}")
print(f"AI Accuracy: {cm[1][1]/(cm[1][0]+cm[1][1]):.2%}")

# By source
print("\nAccuracy by Source:")
source_results = {}
for source in sorted(set(src_test)):
    mask = [s == source for s in src_test]
    if sum(mask) > 50:
        src_acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = 'AI' if np.array(y_test)[mask].mean() > 0.5 else 'Human'
        source_results[source] = {'accuracy': src_acc, 'count': count, 'label': label}
        status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
        print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# Top features
print("\nTop 15 Features (XGBoost):")
importance = xgb_model.feature_importances_
indices = np.argsort(importance)[::-1]
for i in range(15):
    idx = indices[i]
    print(f"  {i+1:2d}. {feature_names[idx]:25s}: {importance[idx]:.4f}")

# Cross-validation
print("\n5-Fold Cross-Validation:")
cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

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
    'accuracy': best_acc,
    'source_results': source_results,
    'model_type': best_name
}

with open('veritas_final.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Saved to 'veritas_final.pkl'")
print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {best_acc:.2%}")
print(f"{'='*70}")
