#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v8 Training Script
===================================
Builds on v7 with 2 additional authenticity features (87 total):
- formal_speech_strength: Weighted detection of MUN/debate speech patterns
- human_authenticity_score: Combined authenticity signals

Goal: Fix false positives on Model UN speeches while maintaining 99%+ accuracy
"""

import json
import pickle
import os
import numpy as np
from datetime import datetime
from datasets import load_dataset
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from feature_extractor_v3 import FeatureExtractorV3

def train_supernova_ultra_v8():
    print("=" * 70)
    print("SUPERNOVA ULTRA v8 TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize feature extractor
    extractor = FeatureExtractorV3()
    print(f"Feature Extractor: {len(extractor.feature_names)} features")
    print(f"New features: formal_speech_strength, human_authenticity_score")
    print()
    
    # Load RAID dataset
    print("Loading RAID dataset...")
    dataset = load_dataset("liamdugan/raid", "raid")
    train_data = dataset['train']
    
    # Balance dataset
    human_samples = [(x['generation'], 0) for x in train_data if x['model'] == 'human']
    ai_samples = [(x['generation'], 1) for x in train_data if x['model'] != 'human']
    
    print(f"Available: {len(human_samples)} human, {len(ai_samples)} AI")
    
    # Use 35k of each for balanced training
    np.random.seed(42)
    human_indices = np.random.choice(len(human_samples), min(35000, len(human_samples)), replace=False)
    ai_indices = np.random.choice(len(ai_samples), min(35000, len(ai_samples)), replace=False)
    
    human_selected = [human_samples[i] for i in human_indices]
    ai_selected = [ai_samples[i] for i in ai_indices]
    
    all_samples = human_selected + ai_selected
    np.random.shuffle(all_samples)
    
    print(f"Training with: {len(human_selected)} human + {len(ai_selected)} AI = {len(all_samples)} total")
    print()
    
    # Extract features
    print("Extracting 87 features...")
    X = []
    y = []
    errors = 0
    
    for i, (text, label) in enumerate(all_samples):
        if i % 5000 == 0:
            print(f"  Progress: {i}/{len(all_samples)} ({100*i/len(all_samples):.1f}%)")
        
        try:
            features = extractor.extract_features(text)
            feature_vector = [features[name] for name in extractor.feature_names]
            X.append(feature_vector)
            y.append(label)
        except Exception as e:
            errors += 1
            continue
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Extracted features: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Errors: {errors}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    print()
    print("Training XGBoost classifier...")
    print("  n_estimators: 800")
    print("  max_depth: 7")
    print("  learning_rate: 0.05")
    print()
    
    model = XGBClassifier(
        n_estimators=800,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    # Evaluate
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")
    print()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(f"  True Negatives (Human correct):  {cm[0][0]}")
    print(f"  False Positives (Human as AI):   {cm[0][1]}")
    print(f"  False Negatives (AI as Human):   {cm[1][0]}")
    print(f"  True Positives (AI correct):     {cm[1][1]}")
    print()
    
    # High confidence metrics
    high_conf_mask = (y_prob >= 0.8) | (y_prob <= 0.2)
    if np.sum(high_conf_mask) > 0:
        high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])
        high_conf_pct = 100 * np.sum(high_conf_mask) / len(y_test)
        print(f"High Confidence (>80%):")
        print(f"  Samples: {np.sum(high_conf_mask)} ({high_conf_pct:.1f}%)")
        print(f"  Accuracy: {high_conf_acc * 100:.2f}%")
    print()
    
    # Feature importance - focus on new features
    feature_importance = dict(zip(extractor.feature_names, model.feature_importances_))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 15 Most Important Features:")
    for i, (name, importance) in enumerate(sorted_features[:15], 1):
        marker = " <-- NEW" if name in ['formal_speech_strength', 'human_authenticity_score'] else ""
        print(f"  {i}. {name}: {importance:.4f}{marker}")
    
    # Check new feature rankings
    print()
    print("New Feature Rankings:")
    for name in ['formal_speech_strength', 'human_authenticity_score']:
        rank = [i for i, (n, _) in enumerate(sorted_features, 1) if n == name][0]
        importance = feature_importance[name]
        print(f"  {name}: Rank #{rank}, Importance: {importance:.4f}")
    
    # Save model
    print()
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    model_dir = "models/SupernovaUltraV8"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save XGBoost model as JSON
    model.save_model(f"{model_dir}/model.json")
    print(f"Saved: {model_dir}/model.json")
    
    # Save scaler
    with open(f"{model_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Saved: {model_dir}/scaler.pkl")
    
    # Save metadata
    metadata = {
        "name": "SUPERNOVA ULTRA v8",
        "version": "8.0",
        "created": datetime.now().isoformat(),
        "feature_extractor": "FeatureExtractorV3",
        "features": len(extractor.feature_names),
        "feature_names": extractor.feature_names,
        "new_features": ["formal_speech_strength", "human_authenticity_score"],
        "training_samples": len(all_samples),
        "dataset": "liamdugan/raid",
        "model_type": "XGBClassifier",
        "model_params": {
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "high_conf_accuracy": float(high_conf_acc) if np.sum(high_conf_mask) > 0 else None,
            "high_conf_percentage": float(high_conf_pct) if np.sum(high_conf_mask) > 0 else None
        },
        "feature_importance_top10": {name: float(imp) for name, imp in sorted_features[:10]}
    }
    
    with open(f"{model_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved: {model_dir}/metadata.json")
    
    # Create veritas_config.js
    config_js = f'''// SUPERNOVA ULTRA v8 Configuration
// Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
// Features: {len(extractor.feature_names)} (including formal_speech_strength, human_authenticity_score)
// Accuracy: {accuracy * 100:.2f}%

const SUPERNOVA_ULTRA_V8_CONFIG = {{
    name: "SUPERNOVA ULTRA v8",
    version: "8.0",
    features: {len(extractor.feature_names)},
    accuracy: {accuracy:.4f},
    highConfAccuracy: {high_conf_acc:.4f},
    newFeatures: ["formal_speech_strength", "human_authenticity_score"]
}};

export default SUPERNOVA_ULTRA_V8_CONFIG;
'''
    
    with open(f"{model_dir}/veritas_config.js", 'w') as f:
        f.write(config_js)
    print(f"Saved: {model_dir}/veritas_config.js")
    
    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model: SUPERNOVA ULTRA v8")
    print(f"Location: {model_dir}/")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"High-Conf Accuracy: {high_conf_acc * 100:.2f}%")
    print(f"Features: {len(extractor.feature_names)}")
    print()
    
    return model, scaler, extractor

if __name__ == "__main__":
    train_supernova_ultra_v8()
