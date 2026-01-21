#!/usr/bin/env python3
"""
VERITAS Ultimate Detector v2
Target: 99% accuracy through:
1. Enhanced heuristic features (especially for problem sources)
2. Neural embeddings 
3. Larger model with optimized hyperparameters
4. Full dataset training
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS Ultimate Detector v2")
print("=" * 70)

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded embedding model")

# =============================================================================
# ENHANCED FEATURE EXTRACTION
# =============================================================================

def extract_features(text):
    """Extract 50+ enhanced features targeting problem sources."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    char_count = len(text)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    paragraphs = text.split('\n\n')
    para_count = len([p for p in paragraphs if p.strip()])
    
    features = {}
    
    # =========================================================================
    # C4 FIXES - C4 is formal web content that looks AI-like
    # Need features that distinguish natural formal writing from AI
    # =========================================================================
    
    # Web content markers (C4 is from web)
    features['has_url'] = 1 if re.search(r'https?://|www\.', text) else 0
    features['has_copyright'] = 1 if re.search(r'©|copyright|\(c\)|all rights reserved', text, re.I) else 0
    features['has_date'] = 1 if re.search(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+', text) else 0
    
    # Human web writing patterns
    features['informal_start'] = 1 if re.match(r'^(So|Well|OK|Okay|Hey|Hi|Oh|Wow)\b', text.strip(), re.I) else 0
    features['trailing_thought'] = len(re.findall(r'[.!?]\s*-\s*\w|[.!?]\s*\.\.\.|[.!?]\s*…', text))
    
    # Specific entities (real events, people, places)
    features['proper_noun_density'] = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / word_count
    features['quoted_text'] = len(re.findall(r'"[^"]+"|"[^"]+"', text))
    
    # =========================================================================
    # DOLLY FIXES - Dolly is short factual AI that looks human
    # Need features that catch instructional/educational patterns
    # =========================================================================
    
    # Instructional patterns
    features['step_by_step'] = len(re.findall(r'\b(step \d|first,|second,|third,|finally,|next,|then,)\b', text, re.I))
    features['definition_pattern'] = len(re.findall(r'\bis\s+(a|an|the)\s+\w+|means\s+(that|to)|refers\s+to|known\s+as', text, re.I))
    features['educational_phrases'] = len(re.findall(r'\bit is important|one should|you should|this means|in other words', text, re.I))
    
    # Facts vs opinions
    features['certainty_markers'] = len(re.findall(r'\b(always|never|definitely|certainly|absolutely|must be)\b', text, re.I)) / word_count
    features['hedging_markers'] = len(re.findall(r'\b(maybe|perhaps|possibly|might|could|probably|seems|appears)\b', text, re.I)) / word_count
    
    # =========================================================================
    # GPT4ALL FIXES
    # =========================================================================
    
    # Response patterns
    features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Of course|Certainly|Absolutely)\b', text.strip()) else 0
    features['helpful_language'] = len(re.findall(r'\bhere is|here are|I hope|feel free|let me|I can|I would|I recommend\b', text, re.I))
    features['list_response'] = len(re.findall(r'^\s*[-•*]\s+\w', text, re.M))
    features['numbered_list'] = len(re.findall(r'^\s*\d+[.)]\s+', text, re.M))
    
    # =========================================================================
    # CORE DISCRIMINATORS (from previous analysis)
    # =========================================================================
    
    # Sentence structure
    sent_lengths = [len(s.split()) for s in sentences]
    features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    features['max_sent_len'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    features['sent_count'] = sent_count
    
    # Word complexity
    word_lengths = [len(w) for w in words]
    features['avg_word_len'] = np.mean(word_lengths) if word_lengths else 0
    features['long_word_rate'] = sum(1 for w in words if len(w) > 8) / word_count
    
    # Vocabulary richness
    unique_words = len(set(w.lower() for w in words))
    features['vocab_richness'] = unique_words / word_count
    
    # Pronouns - key discriminator
    features['first_person'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['second_person'] = len(re.findall(r'\b(you|your|yours|yourself)\b', text, re.I)) / word_count
    features['third_person'] = len(re.findall(r'\b(he|she|they|him|her|them|his|their)\b', text, re.I)) / word_count
    
    # Punctuation patterns
    features['comma_rate'] = text.count(',') / sent_count
    features['semicolon_rate'] = text.count(';') / sent_count
    features['colon_rate'] = text.count(':') / sent_count
    features['exclamation_rate'] = text.count('!') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / sent_count
    features['paren_rate'] = text.count('(') / sent_count
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    
    # Contractions (human signal)
    contractions = re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)
    features['contraction_rate'] = len(contractions) / word_count
    
    # Discourse markers (AI signal)
    discourse = re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence)\b', text, re.I)
    features['discourse_rate'] = len(discourse) / word_count
    
    # Attribution (news/human signal)
    attributions = re.findall(r'\b(said|says|told|asked|replied|noted|added|stated|claimed|according to)\b', text, re.I)
    features['attribution_count'] = len(attributions)
    
    # Casual language (human signal)
    casual = re.findall(r'\b(lol|haha|omg|yeah|nah|ok|okay|hey|hi|thanks|thx|gonna|wanna|kinda|sorta|gotta)\b', text, re.I)
    features['casual_count'] = len(casual)
    
    # Emotional words
    emotional = re.findall(r'\b(love|hate|amazing|awesome|terrible|horrible|wonderful|fantastic|great|bad|worst|best)\b', text, re.I)
    features['emotional_rate'] = len(emotional) / word_count
    
    # Sentence starters
    features['sent_start_I'] = len(re.findall(r'[.!?]\s+I\s', text))
    features['sent_start_The'] = len(re.findall(r'[.!?]\s+The\s', text))
    features['sent_start_This'] = len(re.findall(r'[.!?]\s+This\s', text))
    features['sent_start_It'] = len(re.findall(r'[.!?]\s+It\s', text))
    
    # Passive voice (AI tends to use more)
    passive = re.findall(r'\b(is|are|was|were|been|be|being)\s+\w+ed\b', text, re.I)
    features['passive_rate'] = len(passive) / sent_count
    
    # Structural
    features['word_count'] = word_count
    features['para_count'] = para_count
    features['avg_para_len'] = word_count / para_count if para_count > 0 else word_count
    
    # Code/technical
    features['has_code'] = 1 if re.search(r'```|def\s+\w+|function\s*\(|class\s+\w+|import\s+\w+', text) else 0
    features['has_markdown'] = 1 if re.search(r'^#+\s|\*\*\w|__\w', text, re.M) else 0
    
    return features


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

print(f"Total: {len(samples)}")

# Use all data for maximum accuracy
print("\nExtracting features and embeddings for all samples...")

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

feature_names = list(extract_features("Sample text for feature names.").keys())
print(f"\nExtracted {len(X_heuristic)} samples")
print(f"Heuristic features: {len(feature_names)}")
print(f"Embedding dimensions: {len(X_embedding[0])}")

# Combine
X_heuristic = np.array(X_heuristic, dtype=np.float32)
X_embedding = np.array(X_embedding, dtype=np.float32)
X_combined = np.hstack([X_heuristic, X_embedding])
y = np.array(y_data)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"\nTotal features: {X_combined.shape[1]}")
print(f"Human: {sum(y==0)}, AI: {sum(y==1)}")

# =============================================================================
# TRAIN MODEL
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training XGBoost with Optimized Hyperparameters")
print("=" * 70)

model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=1,
    gamma=0.05,
    reg_alpha=0.01,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    tree_method='hist'  # Faster training
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

# By source (sorted by accuracy)
print("\n" + "-" * 50)
print("Accuracy by Source (sorted):")
print("-" * 50)
source_results = []
for source in sorted(set(src_test)):
    mask = [s == source for s in src_test]
    if sum(mask) > 50:
        src_acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = 'AI' if np.array(y_test)[mask].mean() > 0.5 else 'Human'
        source_results.append((source, src_acc, count, label))

# Sort by accuracy
source_results.sort(key=lambda x: x[1])
for source, src_acc, count, label in source_results:
    status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
    print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# Feature importance (top 20)
print("\n" + "-" * 50)
print("Top 20 Features:")
print("-" * 50)
importances = model.feature_importances_
# First N are heuristic features
n_heuristic = len(feature_names)
heuristic_importance = importances[:n_heuristic]
for name, imp in sorted(zip(feature_names, heuristic_importance), key=lambda x: -x[1])[:20]:
    print(f"  {name:30s}: {imp:.4f}")

embedding_importance = importances[n_heuristic:].sum()
print(f"\n  Embedding features total: {embedding_importance:.4f}")

# Save
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_data = {
    'model': model,
    'scaler': scaler,
    'embed_model_name': 'all-MiniLM-L6-v2',
    'feature_names': feature_names,
    'accuracy': acc,
    'auc': auc
}

with open('veritas_ultimate_v2.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Saved to 'veritas_ultimate_v2.pkl'")

print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {acc:.2%}")
print(f"{'='*70}")
