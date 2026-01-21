#!/usr/bin/env python3
"""
VERITAS Hybrid Detector - Combine heuristic features with neural embeddings
Target: 99% accuracy using both hand-crafted features AND semantic embeddings
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
print("VERITAS Hybrid Detector - Heuristics + Neural Embeddings")
print("=" * 70)

# =============================================================================
# LOAD SENTENCE TRANSFORMER
# =============================================================================

print("\nLoading sentence transformer model...")
from sentence_transformers import SentenceTransformer
# Use a smaller, faster model for efficiency
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded: all-MiniLM-L6-v2 (384 dimensions)")

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_heuristic_features(text):
    """Extract heuristic features."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    sent_lengths = [len(s.split()) for s in sentences]
    avg_sent_len = np.mean(sent_lengths) if sent_lengths else 0
    sent_len_std = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    
    features = {}
    
    # Top discriminators
    proper_nouns = re.findall(r'(?<![.!?]\s)[A-Z][a-z]+', text)
    features['proper_noun_rate'] = len(proper_nouns) / word_count
    
    attributions = re.findall(r'\b(said|says|told|asked|replied|responded|explained|noted|added|stated|claimed|argued|according to)\b', text, re.I)
    features['attribution_count'] = len(attributions)
    
    features['sent_count'] = sent_count
    features['avg_sent_len'] = avg_sent_len
    features['sent_len_std'] = sent_len_std
    
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    
    # Pronouns
    features['first_person'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['second_person'] = len(re.findall(r'\b(you|your|yours)\b', text, re.I)) / word_count
    
    # Punctuation
    features['comma_rate'] = text.count(',') / sent_count
    features['colon_rate'] = text.count(':') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    
    # Contractions
    contractions = re.findall(r"\b(i'm|i've|you're|it's|don't|can't|won't|isn't|aren't|wasn't|weren't)\b", text, re.I)
    features['contraction_rate'] = len(contractions) / word_count
    
    # AI patterns
    features['discourse_rate'] = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally)\b', text, re.I)) / word_count
    features['helpful_count'] = len(re.findall(r'\b(here is|feel free|I hope this helps|let me)\b', text, re.I))
    features['instructional_count'] = len(re.findall(r'\b(first,|second,|for example|such as|in order to)\b', text, re.I))
    
    # Human patterns
    features['casual_count'] = len(re.findall(r'\b(lol|haha|omg|yeah|nah|ok|hey|thanks|dude)\b', text, re.I))
    features['emotional_rate'] = len(re.findall(r'\b(love|hate|amazing|awesome|terrible)\b', text, re.I)) / word_count
    
    # Structural
    features['has_code'] = 1 if re.search(r'```|<code>|function\s*\(|def\s+\w+', text) else 0
    features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
    features['has_numbers'] = 1 if re.search(r'^\s*\d+[.)]\s', text, re.M) else 0
    
    features['sent_start_i'] = len(re.findall(r'[.!?]\s+I\s', text))
    features['word_count'] = word_count
    features['text_length'] = len(text)
    
    return features


def get_embedding(text, max_length=512):
    """Get sentence embedding (truncate to max_length tokens)."""
    # Truncate text to avoid memory issues
    words = text.split()[:max_length]
    truncated = ' '.join(words)
    return embed_model.encode(truncated, show_progress_bar=False)


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading clean dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)
print(f"Total: {len(samples)}")

# Use a subset for faster training with embeddings
# We'll use 100k samples (50k each) for speed
print("\nSampling 100k samples for hybrid training...")
human_samples = [s for s in samples if s.get('label') == 'human']
ai_samples = [s for s in samples if s.get('label') == 'ai']

import random
random.seed(42)
human_subset = random.sample(human_samples, min(50000, len(human_samples)))
ai_subset = random.sample(ai_samples, min(50000, len(ai_samples)))
samples = human_subset + ai_subset
random.shuffle(samples)
print(f"Using {len(samples)} samples")

# Extract features and embeddings
print("\nExtracting features and embeddings...")
X_heuristic = []
X_embedding = []
y_data = []
sources = []

batch_size = 1000
for i in range(0, len(samples), batch_size):
    if i % 10000 == 0:
        print(f"  {i}/{len(samples)}...")
    
    batch = samples[i:i+batch_size]
    texts = []
    
    for s in batch:
        text = s.get('text', '')
        label = s.get('label', '')
        source = s.get('source', 'unknown')
        
        if not text or not label:
            continue
        
        h_feat = extract_heuristic_features(text)
        if h_feat is None:
            continue
        
        X_heuristic.append(list(h_feat.values()))
        texts.append(text[:2000])  # Truncate for embedding
        y_data.append(1 if label == 'ai' else 0)
        sources.append(source)
    
    # Batch encode embeddings
    if texts:
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=32)
        X_embedding.extend(embeddings)

feature_names = list(extract_heuristic_features("Sample text.").keys())
print(f"\nExtracted {len(X_heuristic)} samples")
print(f"Heuristic features: {len(feature_names)}")
print(f"Embedding dimensions: {len(X_embedding[0])}")

# Combine features
X_heuristic = np.array(X_heuristic)
X_embedding = np.array(X_embedding)
X_combined = np.hstack([X_heuristic, X_embedding])
y = np.array(y_data)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"\nTotal features: {X_combined.shape[1]} (heuristic: {X_heuristic.shape[1]}, embedding: {X_embedding.shape[1]})")
print(f"Human: {sum(y==0)}, AI: {sum(y==1)}")

# =============================================================================
# TRAIN MODEL
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training XGBoost on Hybrid Features")
print("=" * 70)

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=12,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
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

print(f"\nACCURACY: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nHuman Accuracy: {cm[0][0]/(cm[0][0]+cm[0][1]):.2%}")
print(f"AI Accuracy: {cm[1][1]/(cm[1][0]+cm[1][1]):.2%}")

# By source
print("\nAccuracy by Source:")
for source in sorted(set(src_test)):
    mask = [s == source for s in src_test]
    if sum(mask) > 50:
        src_acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = 'AI' if np.array(y_test)[mask].mean() > 0.5 else 'Human'
        status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
        print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# Save
print("\n" + "=" * 70)
print("SAVING MODEL")
print("=" * 70)

model_data = {
    'model': model,
    'scaler': scaler,
    'embed_model_name': 'all-MiniLM-L6-v2',
    'feature_names': feature_names,
    'accuracy': acc
}

with open('veritas_hybrid.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print(f"Saved to 'veritas_hybrid.pkl'")
print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {acc:.2%}")
print(f"{'='*70}")
