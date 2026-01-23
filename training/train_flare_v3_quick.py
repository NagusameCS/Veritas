#!/usr/bin/env python3
"""
Flare v3 - Humanization Detection Training
==========================================
Specialized for detecting AI text that's been humanized/paraphrased.
"""

import os
import sys
import json
import pickle
import time
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import FeatureExtractorV3

# Humanization attacks
HUMANIZATION_ATTACKS = ['paraphrase', 'synonym', 'perplexity_misspelling', 'alternative_spelling']

def main():
    print("="*60)
    print("FLARE v3 - Humanization Detection Training")
    print("="*60)
    
    # Load RAID - focus on humanization attacks
    print("\n📥 Loading RAID (humanization focus)...")
    texts, labels, attacks = [], [], []
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    target_human = 20000
    target_humanized = 20000
    human_count = humanized_count = 0
    
    for item in tqdm(raid, total=target_human*4, desc="Loading"):
        if human_count >= target_human and humanized_count >= target_humanized:
            break
        
        text = item.get('generation', '') or item.get('text', '')
        if not text or len(text) < 100:
            continue
        
        model = item.get('model', 'unknown')
        attack = item.get('attack', 'none')
        is_human = model.lower() == 'human'
        
        # Human samples
        if is_human and human_count < target_human:
            texts.append(text[:3000])
            labels.append(0)  # Genuine human
            attacks.append('none')
            human_count += 1
        # Humanized AI samples (paraphrase, synonym, etc.)
        elif not is_human and attack in HUMANIZATION_ATTACKS:
            if humanized_count < target_humanized:
                texts.append(text[:3000])
                labels.append(1)  # Humanized AI
                attacks.append(attack)
                humanized_count += 1
    
    print(f"  Loaded {len(texts):,} samples")
    print(f"    Human: {human_count:,}")
    print(f"    Humanized AI: {humanized_count:,}")
    
    # Feature extraction
    print("\n📊 Extracting features...")
    extractor = FeatureExtractorV3()
    
    features = []
    for text in tqdm(texts, desc="Features"):
        try:
            feat = extractor.extract_feature_vector(text)
            features.append(feat)
        except Exception:
            features.append(np.zeros(85))
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"  Feature shape: {X.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    _, attacks_test = train_test_split(attacks, test_size=0.2, stratify=y, random_state=42)
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\n🚀 Training RandomForest for humanization detection...")
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
        min_samples_split=4,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # High confidence
    high_conf_mask = (y_prob >= 0.8) | (y_prob <= 0.2)
    if high_conf_mask.sum() > 0:
        high_conf_pred = (y_prob[high_conf_mask] >= 0.5).astype(int)
        high_conf_acc = accuracy_score(y_test[high_conf_mask], high_conf_pred)
        high_conf_count = high_conf_mask.sum()
    else:
        high_conf_acc = 0
        high_conf_count = 0
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\n" + "="*60)
    print("RESULTS - HUMANIZATION DETECTION")
    print("="*60)
    print(f"  Accuracy:           {accuracy*100:.2f}%")
    print(f"  High-Conf Accuracy: {high_conf_acc*100:.2f}% ({high_conf_count:,} samples)")
    print(f"  Precision:          {precision*100:.2f}%")
    print(f"  Recall:             {recall*100:.2f}%")
    print(f"  F1 Score:           {f1*100:.2f}%")
    print(f"  ROC AUC:            {auc*100:.2f}%")
    print(f"\n  False Positive Rate: {fpr*100:.2f}%")
    print(f"  False Negative Rate: {fnr*100:.2f}%")
    
    # Per-attack analysis
    print("\n📈 BY ATTACK TYPE")
    attacks_arr = np.array(attacks_test)
    for attack in sorted(set(attacks_test)):
        mask = attacks_arr == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {attack:<25} {attack_acc*100:>6.2f}% ({mask.sum():,} samples)")
    
    # Save
    os.makedirs('./models/FlareV3', exist_ok=True)
    with open('./models/FlareV3/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('./models/FlareV3/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'name': 'Flare v3',
        'version': '3.0',
        'task': 'Humanization Detection',
        'accuracy': accuracy,
        'high_conf_accuracy': high_conf_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'fpr': fpr,
        'fnr': fnr,
        'samples': len(texts),
        'features': 85,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open('./models/FlareV3/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Model saved to ./models/FlareV3/")
    print("="*60)

if __name__ == '__main__':
    main()
