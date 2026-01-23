#!/usr/bin/env python3
"""
Comprehensive RAID Benchmark for SUPERNOVA v2 and Flare v3
==========================================================
Tests models on held-out RAID data across all attack types and domains.
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
    """Load model and scaler."""
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def main():
    print("="*70)
    print("COMPREHENSIVE RAID BENCHMARK")
    print("="*70)
    
    # Load models
    print("\n📦 Loading models...")
    supernova_v2, supernova_v2_scaler = load_model('./models/SupernovaV2')
    supernova_v3, supernova_v3_scaler = load_model('./models/SupernovaV3')
    flare, flare_scaler = load_model('./models/FlareV3')
    extractor = FeatureExtractorV3()
    
    # Load fresh RAID test data (different samples from training)
    print("\n📥 Loading RAID test samples...")
    
    # Skip first 100k samples (used in training), take next 50k
    texts, labels = [], []
    attacks, domains, models = [], [], []
    
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    skip_count = 800000  # Skip samples likely used in training
    target = 30000
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
            models.append('human')
            human_count += 1
        elif not is_human and ai_count < target:
            texts.append(text[:3000])
            labels.append(1)
            attacks.append(attack)
            domains.append(domain)
            models.append(model_name)
            ai_count += 1
    
    print(f"  Loaded {len(texts):,} samples")
    print(f"    Human: {human_count:,}")
    print(f"    AI: {ai_count:,}")
    
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
    
    # Benchmark SUPERNOVA v2
    print("\n" + "="*70)
    print("SUPERNOVA v2 BENCHMARK")
    print("="*70)
    
    X_scaled = supernova_v2_scaler.transform(X)
    y_pred = supernova_v2.predict(X_scaled)
    y_prob = supernova_v2.predict_proba(X_scaled)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    
    # High confidence
    high_conf = (y_prob >= 0.8) | (y_prob <= 0.2)
    if high_conf.sum() > 0:
        hc_pred = (y_prob[high_conf] >= 0.5).astype(int)
        hc_acc = accuracy_score(y[high_conf], hc_pred)
    else:
        hc_acc = 0
    
    print(f"\n📊 OVERALL")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc*100:.2f}% ({high_conf.sum():,} samples)")
    print(f"  ROC AUC:            {auc*100:.2f}%")
    
    # By attack type
    print(f"\n📈 BY ATTACK TYPE")
    attacks_arr = np.array(attacks)
    for attack in sorted(set(attacks)):
        mask = attacks_arr == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {attack:<25} {attack_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By domain
    print(f"\n📈 BY DOMAIN")
    domains_arr = np.array(domains)
    for domain in sorted(set(domains)):
        mask = domains_arr == domain
        if mask.sum() > 10:
            domain_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {domain:<25} {domain_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By AI model
    print(f"\n📈 BY AI MODEL")
    models_arr = np.array(models)
    for model_name in sorted(set(models)):
        if model_name == 'human':
            continue
        mask = models_arr == model_name
        if mask.sum() > 10:
            model_acc = accuracy_score(y[mask], y_pred[mask])
            print(f"    {model_name:<25} {model_acc*100:>6.2f}% ({mask.sum():,})")
    
    v2_results = {'accuracy': acc, 'high_conf_accuracy': hc_acc, 'auc': auc}

    # Benchmark SUPERNOVA v3
    print("\n" + "="*70)
    print("SUPERNOVA v3 BENCHMARK")
    print("="*70)
    
    X_scaled_v3 = supernova_v3_scaler.transform(X)
    y_pred_v3 = supernova_v3.predict(X_scaled_v3)
    y_prob_v3 = supernova_v3.predict_proba(X_scaled_v3)[:, 1]
    
    acc_v3 = accuracy_score(y, y_pred_v3)
    auc_v3 = roc_auc_score(y, y_prob_v3)
    
    high_conf_v3 = (y_prob_v3 >= 0.8) | (y_prob_v3 <= 0.2)
    if high_conf_v3.sum() > 0:
        hc_pred_v3 = (y_prob_v3[high_conf_v3] >= 0.5).astype(int)
        hc_acc_v3 = accuracy_score(y[high_conf_v3], hc_pred_v3)
    else:
        hc_acc_v3 = 0
    
    print(f"\n📊 OVERALL")
    print(f"  Accuracy:           {acc_v3*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc_v3*100:.2f}% ({high_conf_v3.sum():,} samples)")
    print(f"  ROC AUC:            {auc_v3*100:.2f}%")
    
    # By attack type
    print(f"\n📈 BY ATTACK TYPE")
    for attack in sorted(set(attacks)):
        mask = attacks_arr == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y[mask], y_pred_v3[mask])
            print(f"    {attack:<25} {attack_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By domain
    print(f"\n📈 BY DOMAIN")
    for domain in sorted(set(domains)):
        mask = domains_arr == domain
        if mask.sum() > 10:
            domain_acc = accuracy_score(y[mask], y_pred_v3[mask])
            print(f"    {domain:<25} {domain_acc*100:>6.2f}% ({mask.sum():,})")
    
    # By AI model
    print(f"\n📈 BY AI MODEL")
    for model_name in sorted(set(models)):
        if model_name == 'human':
            continue
        mask = models_arr == model_name
        if mask.sum() > 10:
            model_acc = accuracy_score(y[mask], y_pred_v3[mask])
            print(f"    {model_name:<25} {model_acc*100:>6.2f}% ({mask.sum():,})")
    
    v3_results = {'accuracy': acc_v3, 'high_conf_accuracy': hc_acc_v3, 'auc': auc_v3}
    
    # Save results
    results = {
        'model': 'SUPERNOVA v2',
        'accuracy': acc,
        'high_conf_accuracy': hc_acc,
        'auc': auc,
        'test_samples': len(texts),
    }
    
    with open('./benchmark_results_supernova_v2.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Benchmark Flare v3
    print("\n" + "="*70)
    print("FLARE v3 BENCHMARK (on all samples)")
    print("="*70)
    
    X_scaled_flare = flare_scaler.transform(X)
    y_pred_flare = flare.predict(X_scaled_flare)
    y_prob_flare = flare.predict_proba(X_scaled_flare)[:, 1]
    
    acc_flare = accuracy_score(y, y_pred_flare)
    auc_flare = roc_auc_score(y, y_prob_flare)
    
    high_conf_flare = (y_prob_flare >= 0.8) | (y_prob_flare <= 0.2)
    if high_conf_flare.sum() > 0:
        hc_pred_flare = (y_prob_flare[high_conf_flare] >= 0.5).astype(int)
        hc_acc_flare = accuracy_score(y[high_conf_flare], hc_pred_flare)
    else:
        hc_acc_flare = 0
    
    print(f"\n📊 OVERALL")
    print(f"  Accuracy:           {acc_flare*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc_flare*100:.2f}% ({high_conf_flare.sum():,})")
    print(f"  ROC AUC:            {auc_flare*100:.2f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<20} {'Accuracy':>12} {'High-Conf':>12} {'AUC':>12}")
    print("-"*60)
    print(f"{'SUPERNOVA v2':<20} {acc*100:>11.2f}% {hc_acc*100:>11.2f}% {auc*100:>11.2f}%")
    print(f"{'SUPERNOVA v3':<20} {acc_v3*100:>11.2f}% {hc_acc_v3*100:>11.2f}% {auc_v3*100:>11.2f}%")
    print(f"{'Flare v3':<20} {acc_flare*100:>11.2f}% {hc_acc_flare*100:>11.2f}% {auc_flare*100:>11.2f}%")
    
    # Check for improvements
    print("\n📊 V2 vs V3 IMPROVEMENT")
    print("-"*60)
    delta_acc = (acc_v3 - acc) * 100
    delta_hc = (hc_acc_v3 - hc_acc) * 100
    delta_auc = (auc_v3 - auc) * 100
    print(f"  Accuracy:           {delta_acc:+.2f}%")
    print(f"  High-Conf Accuracy: {delta_hc:+.2f}%")
    print(f"  ROC AUC:            {delta_auc:+.2f}%")
    
    # Save all results
    all_results = {
        'supernova_v2': v2_results,
        'supernova_v3': v3_results,
        'flare_v3': {'accuracy': acc_flare, 'high_conf_accuracy': hc_acc_flare, 'auc': auc_flare},
        'test_samples': len(texts),
    }
    with open('./benchmark_results_all.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n✅ Benchmark complete!")

if __name__ == '__main__':
    main()
