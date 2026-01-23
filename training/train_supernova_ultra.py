#!/usr/bin/env python3
"""
SUPERNOVA ULTRA - Maximum Coverage Training
=============================================
Trains on ALL attack types and domains from RAID dataset.
Goal: True 99% generalization across all conditions.
"""

import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import xgboost as xgb
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import FeatureExtractorV3

def load_comprehensive_samples(target_total=200000):
    """Load samples covering ALL attack types and domains."""
    print("\n📥 Loading comprehensive training samples...")
    print("   Goal: Cover ALL attack types and domains")
    
    # Track counts per category
    attack_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    model_counts = defaultdict(int)
    
    texts, labels = [], []
    attacks, domains, models = [], [], []
    
    # We want samples from EVERY attack type we can find
    target_per_attack = target_total // 20  # ~10k per attack type
    target_human = target_total // 3
    
    human_count = 0
    
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    pbar = tqdm(raid, desc="Loading", total=target_total * 5)
    
    for item in pbar:
        if len(texts) >= target_total:
            break
        
        text = item.get('generation', '') or item.get('text', '')
        if not text or len(text) < 100:
            continue
        
        model_name = item.get('model', 'unknown')
        attack = item.get('attack', 'none') or 'none'
        domain = item.get('domain', 'unknown')
        is_human = model_name.lower() == 'human'
        
        # Balanced sampling logic
        if is_human:
            if human_count >= target_human:
                continue
            human_count += 1
            labels.append(0)
        else:
            # Accept all attack types, but cap each
            if attack_counts[attack] >= target_per_attack:
                continue
            labels.append(1)
        
        texts.append(text[:3000])
        attacks.append(attack)
        domains.append(domain)
        models.append(model_name)
        
        attack_counts[attack] += 1
        domain_counts[domain] += 1
        model_counts[model_name] += 1
        
        pbar.set_postfix({
            'total': len(texts),
            'human': human_count,
            'attacks': len(attack_counts)
        })
    
    print(f"\n  Total: {len(texts):,}")
    print(f"  Human: {labels.count(0):,}")
    print(f"  AI: {labels.count(1):,}")
    
    print(f"\n  ATTACK TYPES ({len(attack_counts)}):")
    for attack, count in sorted(attack_counts.items(), key=lambda x: -x[1]):
        print(f"    {attack:<30} {count:,}")
    
    print(f"\n  DOMAINS ({len(domain_counts)}):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<20} {count:,}")
    
    print(f"\n  AI MODELS ({len([m for m in model_counts if m != 'human'])}):")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        if model != 'human':
            print(f"    {model:<20} {count:,}")
    
    return texts, np.array(labels), attacks, domains, models


def main():
    print("="*70)
    print("SUPERNOVA ULTRA - MAXIMUM COVERAGE TRAINING")
    print("="*70)
    
    # Load comprehensive samples
    texts, labels, attacks, domains, models = load_comprehensive_samples(
        target_total=200000
    )
    
    # Extract features
    print("\n📊 Extracting features...")
    extractor = FeatureExtractorV3()
    
    features = []
    for text in tqdm(texts, desc="Features"):
        try:
            feat = extractor.extract_feature_vector(text)
            features.append(feat)
        except Exception as e:
            features.append(np.zeros(85))
    
    X = np.array(features, dtype=np.float32)
    y = labels
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    
    print(f"  Feature matrix: {X.shape}")
    
    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    # Keep metadata for test analysis
    attacks_arr = np.array(attacks)
    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=42, stratify=y
    )
    test_attacks = attacks_arr[test_idx]
    
    # Scale features
    print("\n⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create validation set for early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Train XGBoost with strong regularization
    print("\n🚀 Training XGBoost ULTRA...")
    
    model = xgb.XGBClassifier(
        n_estimators=1500,
        max_depth=10,          # Deeper for more complex patterns
        learning_rate=0.03,    # Lower LR for better generalization
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.5,
        gamma=0.05,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=75
    )
    
    model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    best_iteration = model.best_iteration
    print(f"\n  Best iteration: {best_iteration}")
    
    # Evaluate
    print("\n📊 Evaluating on held-out test set...")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    
    # High-confidence accuracy
    high_conf = (y_prob >= 0.85) | (y_prob <= 0.15)
    if high_conf.sum() > 0:
        hc_pred = (y_prob[high_conf] >= 0.5).astype(int)
        hc_acc = accuracy_score(y_test[high_conf], hc_pred)
    else:
        hc_acc = 0
    
    print("\n" + "="*70)
    print("RESULTS - SUPERNOVA ULTRA")
    print("="*70)
    print(f"\n  Accuracy:           {acc*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc*100:.2f}% ({high_conf.sum():,} samples)")
    print(f"  Precision:          {prec*100:.2f}%")
    print(f"  Recall:             {rec*100:.2f}%")
    print(f"  F1 Score:           {f1*100:.2f}%")
    print(f"  ROC AUC:            {auc*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print(f"  False Negative Rate: {fnr*100:.2f}%")
    
    # Performance by attack type
    print(f"\n📈 BY ATTACK TYPE:")
    for attack in sorted(set(test_attacks)):
        mask = test_attacks == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {attack:<30} {attack_acc*100:>6.2f}% ({mask.sum():,})")
    
    # Save model
    os.makedirs('./models/SupernovaUltra', exist_ok=True)
    
    with open('./models/SupernovaUltra/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('./models/SupernovaUltra/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'name': 'SUPERNOVA ULTRA',
        'version': '4.0.0',
        'algorithm': 'XGBoost',
        'n_estimators': best_iteration,
        'max_depth': 10,
        'learning_rate': 0.03,
        'accuracy': float(acc),
        'high_conf_accuracy': float(hc_acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'auc': float(auc),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'samples': len(texts),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features': 85,
        'attack_types_covered': len(set(attacks)),
        'domains_covered': len(set(domains)),
        'ai_models_covered': len(set(models)) - 1,  # -1 for human
        'improvements': [
            'Comprehensive coverage of ALL attack types',
            'All domains represented',
            'All AI models represented',
            'Deeper XGBoost with more regularization',
            'Lower learning rate for generalization'
        ]
    }
    
    with open('./models/SupernovaUltra/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Model saved to ./models/SupernovaUltra/")
    
    # Feature importance
    print("\n📊 Top 10 Features:")
    feature_names = [
        'char_count', 'word_count', 'sentence_count', 'paragraph_count',
        'avg_word_length', 'avg_sentence_length', 'avg_paragraph_length',
        'type_token_ratio', 'hapax_ratio', 'vocabulary_richness',
        'punctuation_ratio', 'comma_ratio', 'semicolon_ratio', 'question_ratio',
        'exclamation_ratio', 'contraction_count', 'contraction_ratio',
        'sentence_length_variance', 'word_length_variance', 'paragraph_length_variance',
        'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio',
        'pronoun_ratio', 'preposition_ratio', 'conjunction_ratio',
        'first_person_ratio', 'second_person_ratio', 'third_person_ratio',
        'passive_ratio', 'gerund_ratio', 'infinitive_ratio',
        'modal_count', 'modal_ratio', 'hedge_word_count', 'hedge_ratio',
        'filler_word_count', 'filler_ratio', 'discourse_marker_count', 'discourse_ratio',
        'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 'sentiment_compound',
        'reading_ease', 'grade_level', 'difficult_word_ratio',
        'transition_word_ratio', 'cohesion_score',
        'repetition_ratio', 'ngram_repetition', 'phrase_repetition',
        'perplexity_estimate', 'burstiness_score', 'entropy',
        'formality_score', 'lexical_density',
        'clause_depth_avg', 'clause_depth_max', 'embedded_clause_ratio',
        'sentence_start_variety', 'sentence_structure_variety',
        'ai_phrase_count', 'ai_phrase_ratio', 'ai_pattern_score',
        'esl_pattern_count', 'esl_pattern_ratio', 'article_error_ratio',
        'preposition_error_ratio', 'word_order_score',
        'student_marker_count', 'student_marker_ratio', 'informal_ratio',
        'uncertainty_count', 'uncertainty_ratio', 'hedging_pattern_score',
        'personal_experience_ratio', 'opinion_marker_ratio', 'concrete_detail_ratio',
        'formality_inconsistency', 'style_variation_score',
        'flow_score', 'coherence_score', 'authenticity_score'
    ]
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(indices))):
        idx = indices[i]
        name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
