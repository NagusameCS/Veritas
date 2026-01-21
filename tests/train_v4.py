#!/usr/bin/env python3
"""
VERITAS v4 - Dual Embedding Approach
Use two different embedding models to capture different semantic aspects:
1. all-MiniLM-L6-v2: General semantic similarity
2. paraphrase-multilingual-MiniLM-L12-v2: Better at style/paraphrase detection

Also try ensemble of XGBoost + LightGBM for stronger classification.
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS v4 - Dual Embedding + Ensemble")
print("=" * 70)

# =============================================================================
# LOAD EMBEDDING MODELS
# =============================================================================

from sentence_transformers import SentenceTransformer
print("\nLoading embedding models...")
embed_model1 = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dims
embed_model2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # 384 dims
print("Loaded: all-MiniLM-L6-v2 + paraphrase-MiniLM-L6-v2")

# =============================================================================
# REFINED FEATURES (keeping best from v3)
# =============================================================================

def extract_features(text):
    """Streamlined features focusing on highest-impact signals."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    features = {}
    
    # TOP DISCRIMINATORS (from feature importance)
    features['sent_start_I'] = len(re.findall(r'(?:^|[.!?]\s+)I\s', text))
    features['attribution_count'] = len(re.findall(r'\b(said|says|told|asked|noted|added|stated|claimed|according to)\b', text, re.I))
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Certainly|Of course|Absolutely)\b', text.strip()) else 0
    features['numbered_list'] = len(re.findall(r'^\s*\d+[.)]\s+', text, re.M))
    features['has_code'] = 1 if re.search(r'```|def\s+\w+|function\s*\(', text) else 0
    
    # Pronouns
    features['first_person'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['second_person'] = len(re.findall(r'\b(you|your|yours)\b', text, re.I)) / word_count
    
    # Sentence structure
    sent_lengths = [len(s.split()) for s in sentences]
    features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    features['sent_count'] = sent_count
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    
    # Punctuation
    features['comma_rate'] = text.count(',') / sent_count
    features['colon_rate'] = text.count(':') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['exclamation_rate'] = text.count('!') / sent_count
    features['paren_count'] = text.count('(')
    
    # Style signals
    features['contraction_rate'] = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)) / word_count
    features['discourse_rate'] = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally)\b', text, re.I)) / word_count
    features['casual_count'] = len(re.findall(r'\b(lol|haha|omg|yeah|nah|ok|hey|thanks|gonna|wanna)\b', text, re.I))
    features['emotional_rate'] = len(re.findall(r'\b(love|hate|amazing|awesome|terrible|wonderful|great)\b', text, re.I)) / word_count
    
    # AI patterns
    features['helpful_phrases'] = len(re.findall(r'\b(here is|feel free|let me|I hope this|I can help)\b', text, re.I))
    features['step_by_step'] = len(re.findall(r'\b(step \d|first,|second,|finally,|next,)\b', text, re.I))
    features['formal_transitions'] = len(re.findall(r'\b(In conclusion|To summarize|Overall|As mentioned)\b', text, re.I))
    
    # Human authenticity
    features['quote_pairs'] = len(re.findall(r'"[^"]*"|"[^"]*"', text))
    features['specific_years'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
    features['interjections'] = len(re.findall(r'\b(wow|oh|ah|ugh|hmm|huh|geez)\b', text, re.I))
    
    features['word_count'] = word_count
    features['vocab_richness'] = len(set(w.lower() for w in words)) / word_count
    
    return features


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

print(f"Total: {len(samples)}")

# Extract features
print("\nExtracting features and dual embeddings...")
X_heuristic = []
X_embed1 = []
X_embed2 = []
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
        emb1 = embed_model1.encode(texts, show_progress_bar=False, batch_size=64)
        emb2 = embed_model2.encode(texts, show_progress_bar=False, batch_size=64)
        X_embed1.extend(emb1)
        X_embed2.extend(emb2)

feature_names = list(extract_features("Sample text.").keys())
print(f"\nExtracted {len(X_heuristic)} samples")
print(f"Heuristic: {len(feature_names)}, Embed1: 384, Embed2: 384")

# Combine all features
X_heuristic = np.array(X_heuristic, dtype=np.float32)
X_embed1 = np.array(X_embed1, dtype=np.float32)
X_embed2 = np.array(X_embed2, dtype=np.float32)
X_combined = np.hstack([X_heuristic, X_embed1, X_embed2])
y = np.array(y_data)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"Total features: {X_combined.shape[1]}")

# =============================================================================
# TRAIN
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training XGBoost + LightGBM Ensemble")
print("=" * 70)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    verbosity=0
)

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=800,
    max_depth=15,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("\nTraining XGBoost...")
xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
print(f"XGBoost accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

print("\nTraining LightGBM...")
lgb_model.fit(X_train_scaled, y_train)
lgb_pred = lgb_model.predict(X_test_scaled)
print(f"LightGBM accuracy: {accuracy_score(y_test, lgb_pred):.4f}")

# Ensemble via soft voting
print("\nCreating ensemble...")
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
ensemble_proba = (xgb_proba + lgb_proba) / 2
y_pred = (ensemble_proba > 0.5).astype(int)

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, ensemble_proba)

print(f"\n{'='*40}")
print(f"ENSEMBLE ACCURACY: {acc:.4f} ({acc:.2%})")
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
    mask = [s == source for s in src_test]
    if sum(mask) > 50:
        src_acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = 'AI' if np.array(y_test)[mask].mean() > 0.5 else 'Human'
        source_results.append((source, src_acc, count, label))

source_results.sort(key=lambda x: x[1])
for source, src_acc, count, label in source_results:
    status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
    print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# Save
print("\n" + "=" * 70)
model_data = {
    'xgb_model': xgb_model,
    'lgb_model': lgb_model,
    'scaler': scaler,
    'feature_names': feature_names,
    'embed_models': ['all-MiniLM-L6-v2', 'paraphrase-MiniLM-L6-v2'],
    'accuracy': acc
}
with open('veritas_v4.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Saved to 'veritas_v4.pkl'")
print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {acc:.2%}")
print(f"{'='*70}")
