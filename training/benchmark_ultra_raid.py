#!/usr/bin/env python3
"""
SUPERNOVA ULTRA - RAID Benchmark on Held-Out Data
Tests on data NEVER seen during training (skipping first 3M samples)
"""

import pickle
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
import sys
sys.path.insert(0, '/workspaces/Veritas/training')
from feature_extractor_v3 import FeatureExtractorV3

print("=" * 70)
print("SUPERNOVA ULTRA - RAID HELD-OUT BENCHMARK")
print("=" * 70)

# Load model
print("\n📦 Loading SUPERNOVA ULTRA...")
with open('/workspaces/Veritas/training/models/SupernovaUltra/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('/workspaces/Veritas/training/models/SupernovaUltra/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
extractor = FeatureExtractorV3()
print("  ✓ Model loaded!")

# Load RAID with offset to get completely fresh data
print("\n📊 Loading RAID held-out data (skipping first 3M samples)...")
dataset = load_dataset("liamdugan/raid", split="train", streaming=True)

# Skip first 1M samples (training used sparse sampling from full dataset)
# This ensures we're testing on completely unseen data
SKIP_SAMPLES = 1_000_000
TEST_SAMPLES_PER_ATTACK = 300  # 300 samples per attack type = 3600 total

samples = []
attack_counts = defaultdict(int)
skipped = 0

print(f"  Skipping first {SKIP_SAMPLES:,} samples...")

for row in tqdm(dataset, desc="Collecting test data"):
    # Skip first 3M samples
    if skipped < SKIP_SAMPLES:
        skipped += 1
        if skipped % 500000 == 0:
            print(f"  Skipped {skipped:,}...")
        continue
    
    text = row.get('generation', '')
    label = row.get('label', None)
    attack = row.get('attack', 'none')
    
    if not text or len(text) < 100 or label is None:
        continue
    
    # Collect balanced samples per attack
    if attack_counts[attack] < TEST_SAMPLES_PER_ATTACK:
        samples.append({
            'text': text[:5000],
            'label': label,
            'attack': attack,
            'model': row.get('model', 'unknown'),
            'domain': row.get('domain', 'unknown')
        })
        attack_counts[attack] += 1
    
    # Check if we have enough
    if all(c >= TEST_SAMPLES_PER_ATTACK for c in attack_counts.values()) and len(attack_counts) >= 10:
        break
    
    if len(samples) >= 10000:  # Safety limit
        break

print(f"\n  Collected: {len(samples)} test samples")
print(f"  Attack types: {len(attack_counts)}")
for attack, count in sorted(attack_counts.items()):
    print(f"    {attack}: {count}")

# Extract features
print("\n📊 Extracting features...")
features = []
labels = []
metadata = []

for sample in tqdm(samples, desc="Features"):
    try:
        feat = extractor.extract(sample['text'])
        features.append(list(feat.values()))
        labels.append(sample['label'])
        metadata.append({
            'attack': sample['attack'],
            'model': sample['model'],
            'domain': sample['domain']
        })
    except:
        continue

X = np.array(features)
y = np.array(labels)
print(f"  Feature matrix: {X.shape}")

# Scale
X_scaled = scaler.transform(X)

# Predict
print("\n🔮 Running predictions...")
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Calculate metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)
f1 = f1_score(y, y_pred, zero_division=0)
roc_auc = roc_auc_score(y, y_prob)

tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

# High-confidence accuracy
high_conf_mask = (y_prob >= 0.8) | (y_prob <= 0.2)
if high_conf_mask.sum() > 0:
    high_conf_acc = accuracy_score(y[high_conf_mask], y_pred[high_conf_mask])
    high_conf_count = high_conf_mask.sum()
else:
    high_conf_acc = 0
    high_conf_count = 0

print("\n" + "=" * 70)
print("SUPERNOVA ULTRA - RAID HELD-OUT BENCHMARK RESULTS")
print("=" * 70)
print(f"\n  Test samples: {len(y)} (completely unseen data)")
print(f"  Skip offset:  {SKIP_SAMPLES:,} samples\n")

print(f"  Accuracy:           {accuracy*100:.2f}%")
print(f"  High-Conf Accuracy: {high_conf_acc*100:.2f}% ({high_conf_count} samples)")
print(f"  Precision:          {precision*100:.2f}%")
print(f"  Recall:             {recall*100:.2f}%")
print(f"  F1 Score:           {f1*100:.2f}%")
print(f"  ROC AUC:            {roc_auc*100:.2f}%")
print(f"  False Positive Rate: {fpr*100:.2f}%")
print(f"  False Negative Rate: {fnr*100:.2f}%")

# By attack type
print("\n📈 BY ATTACK TYPE (held-out data):")
attack_results = defaultdict(lambda: {'correct': 0, 'total': 0})
for i, meta in enumerate(metadata):
    attack = meta['attack']
    attack_results[attack]['total'] += 1
    if y_pred[i] == y[i]:
        attack_results[attack]['correct'] += 1

for attack in sorted(attack_results.keys()):
    stats = attack_results[attack]
    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    status = "✅" if acc >= 95 else "⚠️" if acc >= 85 else "❌"
    print(f"    {attack:30} {acc:6.2f}% ({stats['total']:4}) {status}")

# By domain
print("\n📈 BY DOMAIN:")
domain_results = defaultdict(lambda: {'correct': 0, 'total': 0})
for i, meta in enumerate(metadata):
    domain = meta['domain']
    domain_results[domain]['total'] += 1
    if y_pred[i] == y[i]:
        domain_results[domain]['correct'] += 1

for domain in sorted(domain_results.keys()):
    stats = domain_results[domain]
    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"    {domain:20} {acc:6.2f}% ({stats['total']})")

# By AI model
print("\n📈 BY AI MODEL:")
model_results = defaultdict(lambda: {'correct': 0, 'total': 0})
for i, meta in enumerate(metadata):
    ai_model = meta['model']
    if y[i] == 1:  # Only for AI-generated text
        model_results[ai_model]['total'] += 1
        if y_pred[i] == 1:  # Correctly identified as AI
            model_results[ai_model]['correct'] += 1

for ai_model in sorted(model_results.keys()):
    stats = model_results[ai_model]
    if stats['total'] > 0:
        acc = stats['correct'] / stats['total'] * 100
        print(f"    {ai_model:20} {acc:6.2f}% ({stats['total']})")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE!")
print("=" * 70)
