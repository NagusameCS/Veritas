#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v7 Training
===========================
Enhanced student/speech detection with 28 patterns for Model UN and formal writing.
"""

import pickle
import json
import numpy as np
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datasets import load_dataset
from tqdm import tqdm
from feature_extractor_v3 import FeatureExtractorV3
import os

print('='*60)
print('SUPERNOVA ULTRA v7 - Enhanced Student/Speech Detection')
print('='*60)

extractor = FeatureExtractorV3()

# Load balanced RAID data
print('\nLoading RAID dataset...')
texts, labels = [], []

# Targets
target_human = 35000
target_ai = 35000

raid = load_dataset('liamdugan/raid', split='train', streaming=True)
human_count, ai_count = 0, 0

for item in tqdm(raid, desc='Loading RAID', total=200000):
    if human_count >= target_human and ai_count >= target_ai:
        break
    
    text = item.get('generation', '') or item.get('text', '')
    if not text or len(text) < 100:
        continue
    
    model = item.get('model', 'unknown')
    is_human = model.lower() == 'human'
    
    if is_human and human_count < target_human:
        texts.append(text[:3000])
        labels.append(0)
        human_count += 1
    elif not is_human and ai_count < target_ai:
        texts.append(text[:3000])
        labels.append(1)
        ai_count += 1

print(f'Loaded {len(texts)} samples (Human: {human_count}, AI: {ai_count})')

# Extract features with updated extractor
print('\nExtracting features with updated extractor...')
features = []
for text in tqdm(texts, desc='Features'):
    f = extractor.extract_feature_vector(text)
    features.append(f)
features = np.array(features, dtype=np.float32)
labels = np.array(labels)
features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)

print(f'Features shape: {features.shape}')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.15, stratify=labels, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost
print('\nTraining XGBoost...')
model = xgb.XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train, verbose=True)

# Evaluate
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

# High confidence accuracy
high_conf = (y_prob >= 0.8) | (y_prob <= 0.2)
if high_conf.sum() > 0:
    hc_pred = (y_prob[high_conf] >= 0.5).astype(int)
    hc_acc = accuracy_score(y_test[high_conf], hc_pred)
else:
    hc_acc = 0

print(f'\n📊 Results:')
print(f'  Accuracy: {acc*100:.2f}%')
print(f'  Precision: {prec*100:.2f}%')
print(f'  Recall: {rec*100:.2f}%')
print(f'  F1: {f1*100:.2f}%')
print(f'  AUC: {auc*100:.2f}%')
print(f'  High-Conf Accuracy: {hc_acc*100:.2f}%')

# Save
out_dir = './models/SupernovaUltraV7'
os.makedirs(out_dir, exist_ok=True)

model.save_model(f'{out_dir}/model.json')
with open(f'{out_dir}/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

metadata = {
    'model': 'SUPERNOVA ULTRA v7',
    'version': '7.0.0',
    'accuracy': float(acc),
    'precision': float(prec),
    'recall': float(rec),
    'f1': float(f1),
    'roc_auc': float(auc),
    'high_confidence_accuracy': float(hc_acc),
    'training_samples': len(texts),
    'features': 85,
    'improvements': 'Enhanced student/speech pattern detection (28 patterns)'
}
with open(f'{out_dir}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'\n✅ Saved to {out_dir}')
