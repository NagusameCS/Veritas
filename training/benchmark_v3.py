#!/usr/bin/env python3
"""
RAID Benchmark for SUPERNOVA v3
================================
Tests on completely held-out data (skip 1.5M samples used in training).
"""

import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, roc_auc_score
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import FeatureExtractorV3

def load_model(model_dir):
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def main():
    print("="*70)
    print("SUPERNOVA v3 - RAID BENCHMARK (HELD-OUT DATA)")
    print("="*70)
    
    # Load model
    print("\n📦 Loading SUPERNOVA v3...")
    model, scaler = load_model('./models/SupernovaV3')
    extractor = FeatureExtractorV3()
    
    # Load held-out test data (skip first 2M samples)
    print("\n📥 Loading held-out RAID samples...")
    
    texts, labels = [], []
    attacks, domains, models_list = [], [], []
    
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    skip_count = 2000000  # Skip samples used in training
    target = 25000
    human_count = ai_count = 0
    skipped = 0
    
    for item in tqdm(raid, desc="Loading", total=skip_count + target*2):
        if skipped < skip_count:
            skipped += 1
            continue
        
        if human_count >= target and ai_count >= target:
            break
        
        text = item.get('generation', '') or item.get('text', '')
        if not text or len(text) < 100:
            continue
        
        model_name = item.get('model', 'unknown')
        attack = item.get('attack', 'none')
        domain = item.get('domain', 'unknown')
        is_human = model_name.lower() == 'human'
        
        if is_human and human_count < target:
            texts.append(text[:3000])
            labels.append(0)
            attacks.append('none')
            domains.append(domain)
            models_list.append('human')
            human_count += 1
        elif not is_human and ai_count < target:
            texts.append(text[:3000])
            labels.append(1)
            attacks.append(attack)
            domains.append(domain)
            models_list.append(model_name)
            ai_count += 1
    
    print(f"  Loaded {len(texts):,} samples (Human: {human_count:,}, AI: {ai_count:,})")
    
    # Extract features
    print("\n📊 Extracting features...")
    features = []
    for text in tqdm(texts, desc="Features"):
        try:
            feat = extractor.extract_feature_vector(text)
            features.append(feat)
        except:
            features.append(np.zeros(85))
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Predict
    print("\n🔮 Running predictions...")
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    # High confidence
    high_conf = (y_prob >= 0.85) | (y_prob <= 0.15)
    if high_conf.sum() > 0:
        hc_pred = (y_prob[high_conf] >= 0.5).astype(int)
        hc_acc = accuracy_score(y[high_conf], hc_pred)
    else:
        hc_acc = 0
    
    print("\n" + "="*70)
    print("RESULTS - SUPERNOVA v3 ON HELD-OUT DATA")
    print("="*70)
    print(f"\n📊 OVERALL")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc*100:.2f}% ({high_conf.sum():,} samples)")
    print(f"  ROC AUC:            {auc*100:.2f}%")
    
    # By attack type
    print(f"\n📈 BY ATTACK TYPE:")
    attacks_arr = np.array(attacks)
    for attack in sorted(set(attacks)):
        mask = attacks_arr == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {attack:<25} {attack_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By domain
    print(f"\n📈 BY DOMAIN:")
    domains_arr = np.array(domains)
    for domain in sorted(set(domains)):
        mask = domains_arr == domain
        if mask.sum() > 10:
            domain_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {domain:<25} {domain_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By AI model
    print(f"\n📈 BY AI MODEL:")
    models_arr = np.array(models_list)
    for model_name in sorted(set(models_list)):
        if model_name == 'human':
            continue
        mask = models_arr == model_name
        if mask.sum() > 10:
            model_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {model_name:<25} {model_acc*100:>6.2f}% ({mask.sum():,})")
    
    # False positives/negatives analysis
    fp_mask = (y == 0) & (y_pred == 1)
    fn_mask = (y == 1) & (y_pred == 0)
    print(f"\n⚠️ ERROR ANALYSIS:")
    print(f"  False Positives: {fp_mask.sum():,} ({fp_mask.sum()/len(y)*100:.2f}%)")
    print(f"  False Negatives: {fn_mask.sum():,} ({fn_mask.sum()/len(y)*100:.2f}%)")
    
    # Save results
    results = {
        'model': 'SUPERNOVA v3',
        'test_samples': len(texts),
        'accuracy': float(acc),
        'high_conf_accuracy': float(hc_acc),
        'high_conf_samples': int(high_conf.sum()),
        'auc': float(auc),
        'false_positives': int(fp_mask.sum()),
        'false_negatives': int(fn_mask.sum())
    }
    
    with open('./benchmark_supernova_v3.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Benchmark complete! Results saved.")

if __name__ == '__main__':
    main()
