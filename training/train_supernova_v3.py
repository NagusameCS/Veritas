#!/usr/bin/env python3
"""
SUPERNOVA v3 Training - Enhanced Generalization
================================================
Addresses weaknesses found in RAID benchmark:
1. Poor performance on chat models (mistral-chat, mpt-chat)
2. Lower accuracy on humanization attacks (synonym, upper_lower)
3. Domain imbalance (books dominating)

Improvements:
- More diverse training samples (100k+)
- Balanced sampling across attack types, domains, models
- Stronger regularization to prevent overfitting
- XGBoost with early stopping
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

def load_balanced_samples(target_per_category=3000, max_total=150000):
    """Load balanced samples across attack types, domains, and AI models."""
    print("\n📥 Loading balanced training samples...")
    
    # Track counts per category
    attack_counts = defaultdict(int)
    domain_counts = defaultdict(int)
    model_counts = defaultdict(int)
    
    texts, labels = [], []
    attacks, domains, models = [], [], []
    
    # Target distribution
    attack_types = ['none', 'synonym', 'paraphrase', 'upper_lower', 'whitespace', 
                    'misspelling', 'article_deletion', 'number_swap', 'homoglyph',
                    'zero_width_space', 'alternative_spelling']
    
    domains_target = ['abstracts', 'books', 'news', 'poetry', 'recipes', 
                      'reddit', 'reviews', 'wiki']
    
    ai_models = ['chatgpt', 'cohere', 'gpt2', 'gpt3', 'gpt4', 'llama-chat',
                 'mistral', 'mistral-chat', 'mpt', 'mpt-chat']
    
    human_target = max_total // 3  # ~50k human
    ai_per_model = (max_total - human_target) // len(ai_models)  # ~10k per AI model
    
    human_count = 0
    ai_counts = defaultdict(int)
    
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    
    pbar = tqdm(raid, desc="Loading", total=max_total * 3)
    
    for item in pbar:
        if len(texts) >= max_total:
            break
        
        text = item.get('generation', '') or item.get('text', '')
        if not text or len(text) < 100:
            continue
        
        model_name = item.get('model', 'unknown')
        attack = item.get('attack', 'none')
        domain = item.get('domain', 'unknown')
        is_human = model_name.lower() == 'human'
        
        # Balanced sampling logic
        if is_human:
            if human_count >= human_target:
                continue
            human_count += 1
            labels.append(0)
        else:
            if model_name not in ai_models:
                continue
            if ai_counts[model_name] >= ai_per_model:
                continue
            ai_counts[model_name] += 1
            labels.append(1)
        
        texts.append(text[:3000])
        attacks.append(attack)
        domains.append(domain)
        models.append(model_name)
        
        attack_counts[attack] += 1
        domain_counts[domain] += 1
        model_counts[model_name] += 1
        
        pbar.set_postfix({
            'samples': len(texts),
            'human': human_count,
            'ai': sum(ai_counts.values())
        })
    
    print(f"\n  Total: {len(texts):,}")
    print(f"  Human: {human_count:,}")
    print(f"  AI: {sum(ai_counts.values()):,}")
    
    print("\n  BY AI MODEL:")
    for model, count in sorted(ai_counts.items(), key=lambda x: -x[1]):
        print(f"    {model:<20} {count:,}")
    
    print("\n  BY ATTACK TYPE:")
    for attack, count in sorted(attack_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {attack:<20} {count:,}")
    
    print("\n  BY DOMAIN:")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"    {domain:<20} {count:,}")
    
    return texts, np.array(labels), attacks, domains, models


def main():
    print("="*70)
    print("SUPERNOVA v3 - ENHANCED GENERALIZATION TRAINING")
    print("="*70)
    
    # Load balanced samples
    texts, labels, attacks, domains, models = load_balanced_samples(
        target_per_category=5000,
        max_total=120000
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
    
    # Keep attack/domain/model info for test analysis
    attacks_arr = np.array(attacks)
    domains_arr = np.array(domains)
    models_arr = np.array(models)
    
    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.15, random_state=42, stratify=y
    )
    test_attacks = attacks_arr[test_idx]
    test_domains = domains_arr[test_idx]
    test_models = models_arr[test_idx]
    
    # Scale features
    print("\n⚙️ Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create validation set for early stopping
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # Train XGBoost with regularization
    print("\n🚀 Training XGBoost with regularization...")
    
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=8,           # Limit depth for generalization
        learning_rate=0.05,    # Lower LR with more trees
        min_child_weight=5,    # Higher regularization
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,         # L1 regularization
        reg_lambda=1.0,        # L2 regularization
        gamma=0.1,             # Min loss reduction
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_train_final, y_train_final,
        eval_set=[(X_val, y_val)],
        verbose=50
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
    print("RESULTS - SUPERNOVA v3")
    print("="*70)
    print(f"\n  Accuracy:           {acc*100:.2f}%")
    print(f"  High-Conf Accuracy: {hc_acc*100:.2f}% ({high_conf.sum():,} samples)")
    print(f"  Precision:          {prec*100:.2f}%")
    print(f"  Recall:             {rec*100:.2f}%")
    print(f"  F1 Score:           {f1*100:.2f}%")
    print(f"  ROC AUC:            {auc*100:.2f}%")
    print(f"  False Positive Rate: {fpr*100:.2f}%")
    print(f"  False Negative Rate: {fnr*100:.2f}%")
    
    # Performance by AI model
    print(f"\n📈 BY AI MODEL:")
    for model_name in sorted(set(test_models)):
        if model_name == 'human':
            continue
        mask = test_models == model_name
        if mask.sum() > 10:
            model_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {model_name:<20} {model_acc*100:>6.2f}% ({mask.sum():,})")
    
    # Performance by attack type
    print(f"\n📈 BY ATTACK TYPE:")
    for attack in sorted(set(test_attacks)):
        mask = test_attacks == attack
        if mask.sum() > 10:
            attack_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {attack:<20} {attack_acc*100:>6.2f}% ({mask.sum():,})")
    
    # Performance by domain
    print(f"\n📈 BY DOMAIN:")
    for domain in sorted(set(test_domains)):
        mask = test_domains == domain
        if mask.sum() > 10:
            domain_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"    {domain:<20} {domain_acc*100:>6.2f}% ({mask.sum():,})")
    
    # Save model
    os.makedirs('./models/SupernovaV3', exist_ok=True)
    
    with open('./models/SupernovaV3/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('./models/SupernovaV3/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    metadata = {
        'name': 'SUPERNOVA v3',
        'version': '3.0.0',
        'algorithm': 'XGBoost',
        'n_estimators': best_iteration,
        'max_depth': 8,
        'learning_rate': 0.05,
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
        'feature_extractor': 'FeatureExtractorV3',
        'improvements': [
            'Balanced sampling across AI models',
            'XGBoost with regularization',
            'Early stopping to prevent overfitting',
            'Lower learning rate, more trees'
        ]
    }
    
    with open('./models/SupernovaV3/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ Model saved to ./models/SupernovaV3/")
    
    # Feature importance
    print("\n📊 Top 15 Features:")
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
    
    for i in range(min(15, len(indices))):
        idx = indices[i]
        name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
