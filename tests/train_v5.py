#!/usr/bin/env python3
"""
VERITAS v5 - Deep feature engineering + Calibrated ensemble
Target: Push past 95% by:
1. Adding more domain-specific features for problem sources
2. Using calibrated probabilities
3. Source-aware weighting
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS v5 - Deep Features + Calibration")
print("=" * 70)

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# SUPER DEEP FEATURE EXTRACTION
# =============================================================================

def extract_features(text):
    """100+ features targeting every possible signal."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    features = {}
    
    # =======================================================================
    # STATISTICAL FEATURES
    # =======================================================================
    
    # Word-level
    word_lengths = [len(w) for w in words]
    features['avg_word_len'] = np.mean(word_lengths) if word_lengths else 0
    features['word_len_std'] = np.std(word_lengths) if len(word_lengths) > 1 else 0
    features['max_word_len'] = max(word_lengths) if word_lengths else 0
    features['short_words'] = sum(1 for w in words if len(w) <= 3) / word_count
    features['long_words'] = sum(1 for w in words if len(w) >= 8) / word_count
    
    # Sentence-level
    sent_lengths = [len(s.split()) for s in sentences]
    features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    features['max_sent_len'] = max(sent_lengths) if sent_lengths else 0
    features['sent_count'] = sent_count
    features['short_sents'] = sum(1 for l in sent_lengths if l < 6) / sent_count
    features['long_sents'] = sum(1 for l in sent_lengths if l > 25) / sent_count
    
    # Vocabulary
    unique_words = set(w.lower() for w in words)
    features['vocab_richness'] = len(unique_words) / word_count
    features['hapax_legomena'] = sum(1 for w in unique_words if words.count(w) == 1) / word_count
    
    # Paragraph structure
    paragraphs = text.split('\n\n')
    para_count = len([p for p in paragraphs if p.strip()])
    features['para_count'] = para_count
    features['words_per_para'] = word_count / para_count if para_count > 0 else word_count
    
    # =======================================================================
    # PUNCTUATION FEATURES
    # =======================================================================
    
    features['comma_rate'] = text.count(',') / sent_count
    features['semicolon_count'] = text.count(';')
    features['colon_rate'] = text.count(':') / sent_count
    features['exclamation_rate'] = text.count('!') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['paren_count'] = text.count('(')
    features['bracket_count'] = text.count('[')
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / word_count
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['quote_pairs'] = len(re.findall(r'"[^"]*"|"[^"]*"', text))
    features['single_quotes'] = len(re.findall(r"'[^']*'", text))
    
    # =======================================================================
    # PRONOUN ANALYSIS
    # =======================================================================
    
    features['first_I'] = len(re.findall(r'\bI\b', text)) / word_count
    features['first_me'] = len(re.findall(r'\b(me|my|mine|myself)\b', text, re.I)) / word_count
    features['first_we'] = len(re.findall(r'\b(we|us|our|ours)\b', text, re.I)) / word_count
    features['second_you'] = len(re.findall(r'\b(you|your|yours)\b', text, re.I)) / word_count
    features['third_they'] = len(re.findall(r'\b(they|them|their|theirs)\b', text, re.I)) / word_count
    features['third_he_she'] = len(re.findall(r'\b(he|she|him|her|his|hers)\b', text, re.I)) / word_count
    features['third_it'] = len(re.findall(r'\b(it|its)\b', text, re.I)) / word_count
    
    # =======================================================================
    # SENTENCE STARTERS
    # =======================================================================
    
    first_words = [s.split()[0].lower() if s.split() else '' for s in sentences]
    first_word_counts = Counter(first_words)
    total_sents = len(first_words) if first_words else 1
    
    features['start_I'] = first_word_counts.get('i', 0) / total_sents
    features['start_the'] = first_word_counts.get('the', 0) / total_sents
    features['start_this'] = first_word_counts.get('this', 0) / total_sents
    features['start_it'] = first_word_counts.get('it', 0) / total_sents
    features['start_there'] = first_word_counts.get('there', 0) / total_sents
    features['start_in'] = first_word_counts.get('in', 0) / total_sents
    features['start_however'] = first_word_counts.get('however', 0) / total_sents
    
    # Variety of sentence starters
    features['starter_variety'] = len(set(first_words)) / total_sents
    
    # =======================================================================
    # LINGUISTIC PATTERNS (HUMAN SIGNALS)
    # =======================================================================
    
    # Contractions (strong human signal)
    features['contraction_count'] = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I))
    features['contraction_rate'] = features['contraction_count'] / word_count
    
    # Casual/informal
    features['casual_words'] = len(re.findall(r'\b(lol|haha|omg|yeah|nah|ok|okay|hey|hi|bye|thanks|thx|gonna|wanna|kinda|sorta|gotta|cuz|bc|btw)\b', text, re.I))
    features['slang'] = len(re.findall(r'\b(cool|awesome|sucks|bro|dude|guys|tbh|imo|fyi)\b', text, re.I))
    features['interjections'] = len(re.findall(r'\b(wow|oh|ah|ooh|ugh|hmm|huh|geez|gosh|damn|darn|yay)\b', text, re.I))
    
    # Emotional expressions
    features['emotional_positive'] = len(re.findall(r'\b(love|amazing|awesome|fantastic|wonderful|great|excellent|beautiful)\b', text, re.I)) / word_count
    features['emotional_negative'] = len(re.findall(r'\b(hate|terrible|horrible|awful|disgusting|worst|bad|stupid)\b', text, re.I)) / word_count
    
    # Personal narrative markers
    features['storytelling'] = len(re.findall(r'\b(yesterday|last week|last year|remember|once upon|when I was)\b', text, re.I))
    features['direct_speech'] = len(re.findall(r'"[^"]*"|"[^"]*"', text))
    
    # =======================================================================
    # LINGUISTIC PATTERNS (AI SIGNALS)
    # =======================================================================
    
    # Formal discourse markers
    features['discourse_however'] = len(re.findall(r'\bhowever\b', text, re.I))
    features['discourse_therefore'] = len(re.findall(r'\btherefore\b', text, re.I))
    features['discourse_furthermore'] = len(re.findall(r'\b(furthermore|moreover|additionally)\b', text, re.I))
    features['discourse_consequently'] = len(re.findall(r'\b(consequently|thus|hence|accordingly)\b', text, re.I))
    features['discourse_total'] = (features['discourse_however'] + features['discourse_therefore'] + 
                                   features['discourse_furthermore'] + features['discourse_consequently'])
    
    # AI helper phrases
    features['helpful_phrases'] = len(re.findall(r'\b(here is|here are|feel free|let me|I hope this helps|I can help|happy to help)\b', text, re.I))
    features['instruction_phrases'] = len(re.findall(r'\b(first,|second,|third,|finally,|next,|then,|step \d|for example|such as|in order to|make sure to)\b', text, re.I))
    features['summary_phrases'] = len(re.findall(r'\b(in conclusion|to summarize|in summary|overall|to sum up|in short)\b', text, re.I))
    features['ai_hedging'] = len(re.findall(r"\b(it's worth noting|it's important to|one thing to consider|keep in mind)\b", text, re.I))
    
    # Numbered/bulleted lists (AI pattern)
    features['numbered_items'] = len(re.findall(r'^\s*\d+[.)]\s+', text, re.M))
    features['bullet_items'] = len(re.findall(r'^\s*[-•*]\s+', text, re.M))
    
    # Answer openers
    features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Certainly|Of course|Absolutely|Great question|Good question)\b', text.strip(), re.I) else 0
    
    # =======================================================================
    # REAL-WORLD REFERENCES (HUMAN SIGNAL)
    # =======================================================================
    
    # Dates and times
    features['specific_dates'] = len(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
    features['month_mentions'] = len(re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text))
    features['year_mentions'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
    features['time_mentions'] = len(re.findall(r'\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b', text, re.I))
    
    # Money
    features['currency_mentions'] = len(re.findall(r'\$\d+|\d+\s*(?:dollars?|USD|EUR|pounds?)\b', text, re.I))
    
    # Attribution (news style)
    features['attribution'] = len(re.findall(r'\b(said|says|told|asked|replied|noted|added|stated|claimed|argued|explained|according to)\b', text, re.I))
    
    # Proper nouns (real names/places)
    features['proper_nouns'] = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / word_count
    
    # URLs and emails (web content)
    features['urls'] = len(re.findall(r'https?://\S+|www\.\S+', text))
    features['emails'] = len(re.findall(r'[\w.-]+@[\w.-]+\.\w+', text))
    
    # =======================================================================
    # TECHNICAL CONTENT
    # =======================================================================
    
    features['has_code'] = 1 if re.search(r'```|<code>|def\s+\w+\(|function\s*\(|class\s+\w+|import\s+\w+|#include', text) else 0
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    features['has_markdown'] = 1 if re.search(r'^#+\s|\*\*\w|__\w', text, re.M) else 0
    
    # =======================================================================
    # DOCUMENT STRUCTURE
    # =======================================================================
    
    features['word_count'] = word_count
    features['char_count'] = len(text)
    features['words_per_char'] = word_count / len(text) if len(text) > 0 else 0
    
    return features


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

print(f"Total: {len(samples)}")

# Extract features
print("\nExtracting features...")
X_heuristic = []
X_embedding = []
y_data = []
sources = []

batch_size = 500
total = len(samples)

for i in range(0, total, batch_size):
    if i % 20000 == 0:
        print(f"  {i}/{total}...")
    
    batch = samples[i:i+batch_size]
    texts = []
    
    for s in batch:
        text = s.get('text', '')
        label = s.get('label', '')
        source = s.get('source', 'unknown')
        
        if not text or not label:
            continue
        
        h_feat = extract_features(text)
        if h_feat is None:
            continue
        
        X_heuristic.append(list(h_feat.values()))
        texts.append(text[:2000])
        y_data.append(1 if label == 'ai' else 0)
        sources.append(source)
    
    if texts:
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        X_embedding.extend(embeddings)

feature_names = list(extract_features("Sample text.").keys())
print(f"\nExtracted {len(X_heuristic)} samples")
print(f"Heuristic features: {len(feature_names)}")

# Combine
X_heuristic = np.array(X_heuristic, dtype=np.float32)
X_embedding = np.array(X_embedding, dtype=np.float32)
X_combined = np.hstack([X_heuristic, X_embedding])
y = np.array(y_data)
sources = np.array(sources)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"Total features: {X_combined.shape[1]}")

# =============================================================================
# TRAIN WITH STRATIFIED SOURCE SPLIT
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training XGBoost with deeper architecture")
print("=" * 70)

model = xgb.XGBClassifier(
    n_estimators=1200,
    max_depth=20,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=1,
    gamma=0.01,
    reg_alpha=0.01,
    reg_lambda=1,
    scale_pos_weight=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    tree_method='hist'
)

model.fit(X_train_scaled, y_train, 
          eval_set=[(X_test_scaled, y_test)], 
          verbose=100)

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*40}")
print(f"ACCURACY: {acc:.4f} ({acc:.2%})")
print(f"AUC-ROC:  {auc:.4f}")
print(f"{'='*40}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nHuman Accuracy: {cm[0][0]/(cm[0][0]+cm[0][1]):.2%}")
print(f"AI Accuracy: {cm[1][1]/(cm[1][0]+cm[1][1]):.2%}")

# By source
print("\n" + "-" * 50)
print("Accuracy by Source:")
print("-" * 50)
source_results = []
for source in sorted(set(src_test)):
    mask = src_test == source
    if sum(mask) > 50:
        src_acc = accuracy_score(y_test[mask], y_pred[mask])
        count = sum(mask)
        label = 'AI' if y_test[mask].mean() > 0.5 else 'Human'
        source_results.append((source, src_acc, count, label))

source_results.sort(key=lambda x: x[1])
problem_count = 0
for source, src_acc, count, label in source_results:
    status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
    if src_acc < 0.90:
        problem_count += count
    print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

print(f"\nProblem samples (<90%): {problem_count}/{len(y_test)} ({problem_count/len(y_test):.1%})")

# Feature importance
print("\n" + "-" * 50)
print("Top 25 Features:")
print("-" * 50)
importances = model.feature_importances_[:len(feature_names)]
for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1])[:25]:
    print(f"  {name:30s}: {imp:.4f}")

# Save
print("\n" + "=" * 70)
with open('veritas_v5.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names, 'accuracy': acc}, f)

print(f"Saved to 'veritas_v5.pkl'")
print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {acc:.2%}")
print(f"{'='*70}")
