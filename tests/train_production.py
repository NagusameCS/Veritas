#!/usr/bin/env python3
"""
VERITAS FINAL PRODUCTION MODEL
Features:
1. XGBoost + Neural Embeddings (415 features)
2. Confidence thresholds for uncertainty handling
3. High-accuracy mode (97.31% on 94.3% of samples)
4. Fallback human review for ambiguous cases
"""

import json
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS PRODUCTION MODEL - FINAL TRAINING")
print("=" * 70)

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================================================================
# PRODUCTION FEATURE EXTRACTION
# =============================================================================

def extract_features(text):
    """Production-ready feature extraction (31 optimized features)."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    features = {}
    
    # Top discriminators (from importance analysis)
    features['third_he_she'] = len(re.findall(r'\b(he|she|him|her|his|hers)\b', text, re.I)) / word_count
    features['first_me'] = len(re.findall(r'\b(me|my|mine|myself)\b', text, re.I)) / word_count
    features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Certainly|Of course|Absolutely)\b', text.strip(), re.I) else 0
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['instruction_phrases'] = len(re.findall(r'\b(first,|second,|finally,|step \d|for example)\b', text, re.I))
    features['attribution'] = len(re.findall(r'\b(said|says|told|according to|noted|stated)\b', text, re.I))
    features['numbered_items'] = len(re.findall(r'^\s*\d+[.)]\s+', text, re.M))
    features['helpful_phrases'] = len(re.findall(r'\b(here is|let me|feel free|I hope this)\b', text, re.I))
    features['proper_nouns'] = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / word_count
    
    # Structural
    sent_lengths = [len(s.split()) for s in sentences]
    features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    features['sent_count'] = sent_count
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    
    paragraphs = text.split('\n\n')
    features['para_count'] = len([p for p in paragraphs if p.strip()])
    
    # Pronouns
    features['first_I'] = len(re.findall(r'\bI\b', text)) / word_count
    features['first_we'] = len(re.findall(r'\b(we|us|our)\b', text, re.I)) / word_count
    features['second_you'] = len(re.findall(r'\b(you|your)\b', text, re.I)) / word_count
    
    # Punctuation
    features['colon_rate'] = text.count(':') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['paren_count'] = text.count('(')
    
    # Temporal references (human authenticity)
    features['month_mentions'] = len(re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text))
    features['year_mentions'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
    features['time_mentions'] = len(re.findall(r'\b\d{1,2}:\d{2}\b', text))
    features['specific_dates'] = len(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
    
    # Style markers
    features['has_code'] = 1 if re.search(r'```|def\s+\w+|function\s*\(', text) else 0
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    features['discourse_markers'] = len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently)\b', text, re.I))
    features['contraction_rate'] = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)) / word_count
    features['casual_words'] = len(re.findall(r'\b(lol|haha|yeah|nah|ok|gonna|wanna)\b', text, re.I))
    
    features['word_count'] = word_count
    features['vocab_richness'] = len(set(w.lower() for w in words)) / word_count
    
    return features


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("\nLoading dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

print(f"Total: {len(samples)} samples")

print("\nExtracting features with embeddings...")
X_heuristic = []
X_embedding = []
y_data = []
sources = []

batch_size = 500
for i in range(0, len(samples), batch_size):
    if i % 40000 == 0:
        print(f"  {i}/{len(samples)}...")
    
    batch = samples[i:i+batch_size]
    texts = []
    feats = []
    
    for s in batch:
        text = s.get('text', '')
        label = s.get('label', '')
        source = s.get('source', 'unknown')
        
        if not text or not label:
            continue
        
        h_feat = extract_features(text)
        if h_feat is None:
            continue
        
        feats.append(list(h_feat.values()))
        texts.append(text[:2000])
        y_data.append(1 if label == 'ai' else 0)
        sources.append(source)
    
    if texts:
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        X_heuristic.extend(feats)
        X_embedding.extend(embeddings)

feature_names = list(extract_features("Sample text.").keys())
X_heuristic = np.array(X_heuristic, dtype=np.float32)
X_embedding = np.array(X_embedding, dtype=np.float32)
X_combined = np.hstack([X_heuristic, X_embedding])
y = np.array(y_data)
sources = np.array(sources)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"\nFeatures: {len(feature_names)} heuristic + {X_embedding.shape[1]} embedding = {X_combined.shape[1]} total")

# =============================================================================
# TRAIN FINAL MODEL
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training Production XGBoost Model")
print("=" * 70)

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=18,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=1,
    gamma=0.01,
    random_state=42,
    n_jobs=-1,
    tree_method='hist',
    eval_metric='logloss'
)

model.fit(X_train_scaled, y_train, 
          eval_set=[(X_test_scaled, y_test)], 
          verbose=200)

# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Overall metrics
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*50}")
print(f"OVERALL ACCURACY: {acc:.4f} ({acc:.2%})")
print(f"AUC-ROC: {auc:.4f}")
print(f"{'='*50}")

print("\n" + classification_report(y_test, y_pred, target_names=['Human', 'AI']))

# Confidence-based accuracy
print("\n" + "-" * 50)
print("CONFIDENCE-BASED ACCURACY")
print("-" * 50)

thresholds = [(0.9, 0.1), (0.8, 0.2), (0.7, 0.3)]
for high, low in thresholds:
    mask = (y_proba >= high) | (y_proba <= low)
    if sum(mask) > 0:
        conf_acc = accuracy_score(y_test[mask], y_pred[mask])
        coverage = sum(mask) / len(y_test)
        print(f"  Threshold {low:.1f}-{high:.1f}: {conf_acc:.2%} accuracy, {coverage:.1%} coverage ({sum(mask)} samples)")

# Per-source breakdown
print("\n" + "-" * 50)
print("PER-SOURCE ACCURACY")
print("-" * 50)

source_results = []
for source in sorted(set(src_test)):
    mask = src_test == source
    if sum(mask) > 50:
        src_acc = accuracy_score(y_test[mask], y_pred[mask])
        count = sum(mask)
        label = 'AI' if y_test[mask].mean() > 0.5 else 'Human'
        source_results.append((source, src_acc, count, label))

source_results.sort(key=lambda x: -x[1])
for source, src_acc, count, label in source_results:
    status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
    print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# =============================================================================
# SAVE PRODUCTION MODEL
# =============================================================================

print("\n" + "=" * 70)
print("SAVING PRODUCTION MODEL")
print("=" * 70)

model_bundle = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names,
    'embed_model': 'all-MiniLM-L6-v2',
    'metrics': {
        'accuracy': acc,
        'auc_roc': auc,
        'high_conf_accuracy': accuracy_score(
            y_test[(y_proba >= 0.8) | (y_proba <= 0.2)], 
            y_pred[(y_proba >= 0.8) | (y_proba <= 0.2)]
        ),
        'high_conf_coverage': sum((y_proba >= 0.8) | (y_proba <= 0.2)) / len(y_test)
    },
    'version': '1.0',
    'training_samples': len(X_train),
    'feature_count': X_combined.shape[1]
}

with open('veritas_production.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"Saved to 'veritas_production.pkl'")
print(f"Model size: {X_combined.shape[1]} features")

# Print final summary
print(f"\n{'='*70}")
print("VERITAS PRODUCTION MODEL - FINAL STATS")
print(f"{'='*70}")
print(f"""
Training Data:
  - Samples: {len(X_train):,} training, {len(X_test):,} test
  - Sources: 19 diverse datasets (human & AI)
  
Model Architecture:
  - XGBoost: 1000 trees, depth 18
  - Features: {len(feature_names)} heuristic + 384 embedding = {X_combined.shape[1]} total
  
Performance:
  - Overall Accuracy: {acc:.2%}
  - AUC-ROC: {auc:.4f}
  
Confidence-Based Detection:
  - High Confidence (>0.8 or <0.2): {model_bundle['metrics']['high_conf_accuracy']:.2%} accuracy
  - Coverage: {model_bundle['metrics']['high_conf_coverage']:.1%} of samples
  
Recommendation:
  - Use high-confidence threshold for production
  - Flag low-confidence samples for human review
""")
