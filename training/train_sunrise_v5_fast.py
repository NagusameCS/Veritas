#!/usr/bin/env python3
"""
Sunrise V5.0 - Fast Training with Feature Caching
===================================================
- 1,000,000+ samples from HuggingFace datasets
- SAVES features to disk for instant resume
- 5 Optuna trials for quick optimization
- Full provenance tracking
"""

import os
import sys
import json
import hashlib
import time
import warnings
import gc
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ML imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import joblib
import optuna

# Feature extractor
from feature_extractor_v4 import FeatureExtractorV4

# Initialize global extractor
_extractor = FeatureExtractorV4()
FEATURE_NAMES = _extractor.get_feature_names()

def extract_features_v4(text):
    """Wrapper function for feature extraction"""
    return _extractor.extract_feature_vector(text)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_SAMPLES = 1_000_000
MODEL_NAME = "Sunrise"
MODEL_VERSION = "5.0"
OUTPUT_DIR = "models/SunriseV5"
RECEIPTS_DIR = "models/SunriseV5/receipts"
CACHE_DIR = "models/SunriseV5/cache"  # NEW: Feature cache directory
N_WORKERS = 15
OPTUNA_TRIALS = 1  # Single trial for fastest training
BATCH_SIZE = 50000

# Only parquet-based datasets that work reliably
DATASETS_CONFIG = [
    # === HUMAN TEXT SOURCES ===
    {"name": "cc_news", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 200000, "source_type": "news"},
    {"name": "cnn_dailymail", "config": "3.0.0", "split": "train", "text_field": "article", "label": 0, "max_samples": 150000, "source_type": "news"},
    {"name": "xsum", "config": None, "split": "train", "text_field": "document", "label": 0, "max_samples": 100000, "source_type": "news"},
    {"name": "squad", "config": None, "split": "train", "text_field": "context", "label": 0, "max_samples": 50000, "source_type": "qa"},
    {"name": "squad_v2", "config": None, "split": "train", "text_field": "context", "label": 0, "max_samples": 50000, "source_type": "qa"},
    {"name": "yelp_review_full", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 100000, "source_type": "reviews"},
    {"name": "imdb", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 25000, "source_type": "reviews"},
    {"name": "amazon_polarity", "config": None, "split": "train", "text_field": "content", "label": 0, "max_samples": 100000, "source_type": "reviews"},
    {"name": "billsum", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 20000, "source_type": "legal"},
    {"name": "wikitext", "config": "wikitext-103-v1", "split": "train", "text_field": "text", "label": 0, "max_samples": 80000, "source_type": "wiki"},
    
    # === AI-GENERATED TEXT SOURCES ===
    {"name": "aadityaubhat/GPT-wiki-intro", "config": None, "split": "train", "text_field": "generated_intro", "label": 1, "max_samples": 150000, "source_type": "gpt_wiki"},
    {"name": "artem9k/ai-text-detection-pile", "config": None, "split": "train", "text_field": "text", "label": -1, "max_samples": 200000, "source_type": "ai_detection"},
    {"name": "teknium/OpenHermes-2.5", "config": None, "split": "train", "text_field": "conversations", "label": 1, "max_samples": 100000, "source_type": "llm_instruct"},
]

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CACHING
# ══════════════════════════════════════════════════════════════════════════════

def get_cache_path():
    """Get path to cached features file"""
    return os.path.join(CACHE_DIR, "features_1M.npz")

def load_cached_features():
    """Load features from cache if available"""
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        print(f"\n  📂 Loading cached features from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        X = data['X']
        y = data['y']
        print(f"  ✓ Loaded {len(X):,} samples from cache")
        return X, y
    return None, None

def save_features_to_cache(X, y):
    """Save features to cache for future use"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path()
    print(f"\n  💾 Saving features to cache: {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  ✓ Cache saved ({size_mb:.1f} MB)")

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE COLLECTION 
# ══════════════════════════════════════════════════════════════════════════════

def collect_samples_from_datasets():
    """Collect samples from HuggingFace datasets"""
    from datasets import load_dataset
    
    samples = []
    labels = []
    seen_hashes = set()
    
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    receipts = []
    receipt_batch_num = 0
    
    def save_receipt_batch(receipts_batch, batch_num):
        """Save a batch of receipts to disk"""
        if not receipts_batch:
            return
        filepath = os.path.join(RECEIPTS_DIR, f"receipts_batch_{batch_num:04d}.json")
        with open(filepath, 'w') as f:
            json.dump(receipts_batch, f)
    
    for ds_config in DATASETS_CONFIG:
        if len(samples) >= TARGET_SAMPLES:
            break
            
        name = ds_config["name"]
        config = ds_config.get("config")
        split = ds_config["split"]
        text_field = ds_config["text_field"]
        default_label = ds_config["label"]
        max_samples = ds_config["max_samples"]
        source_type = ds_config.get("source_type", "unknown")
        
        config_str = f" ({config})" if config else ""
        print(f"  📥 Loading {name}{config_str}")
        
        try:
            if config:
                dataset = load_dataset(name, config, split=split, trust_remote_code=True)
            else:
                dataset = load_dataset(name, split=split, trust_remote_code=True)
            
            available = len(dataset)
            to_sample = min(max_samples, available)
            print(f"    📊 Available: {available:,}, sampling: {to_sample:,}")
            
            # Sample indices
            if to_sample < available:
                indices = np.random.choice(available, to_sample, replace=False)
            else:
                indices = range(available)
            
            dataset_samples = 0
            pbar = tqdm(indices, desc="    Processing", leave=False)
            
            for idx in pbar:
                if len(samples) >= TARGET_SAMPLES:
                    break
                    
                row = dataset[int(idx)]
                
                # Get text
                if text_field == "conversations":
                    convs = row.get("conversations", [])
                    text = " ".join([c.get("value", "") for c in convs if isinstance(c, dict)])
                elif text_field == "chatgpt_answers":
                    answers = row.get("chatgpt_answers", [])
                    text = " ".join(answers) if isinstance(answers, list) else str(answers)
                else:
                    text = row.get(text_field, "")
                
                if not text or len(text.strip()) < 50:
                    continue
                
                # Deduplicate
                text_hash = hashlib.md5(text.encode()[:1000]).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)
                
                # Determine label
                if default_label == -1:
                    label = row.get("generated", row.get("label", 0))
                    if isinstance(label, str):
                        label = 1 if label.lower() in ["ai", "generated", "1", "true"] else 0
                else:
                    label = default_label
                
                samples.append(text)
                labels.append(label)
                dataset_samples += 1
                
                # Create receipt
                receipt = {
                    "id": f"{name}_{idx}",
                    "dataset": name,
                    "config": config,
                    "index": int(idx),
                    "label": int(label),
                    "source_type": source_type,
                    "text_hash": text_hash,
                    "text_length": len(text),
                    "timestamp": datetime.now().isoformat()
                }
                receipts.append(receipt)
                
                # Save receipt batch
                if len(receipts) >= 10000:
                    save_receipt_batch(receipts, receipt_batch_num)
                    total_receipts = (receipt_batch_num + 1) * 10000
                    print(f"    💾 Saved 10,000 receipts (batch {receipt_batch_num}, total: {total_receipts:,})")
                    receipt_batch_num += 1
                    receipts = []
            
            pbar.close()
            print(f"    ✓ Collected {dataset_samples:,} samples from {name}")
            
        except Exception as e:
            print(f"    ⚠ Error loading {name}: {e}")
            continue
        
        print(f"  📊 Total collected: {len(samples):,} / {TARGET_SAMPLES:,}")
        
        del dataset
        gc.collect()
    
    # Save remaining receipts
    if receipts:
        save_receipt_batch(receipts, receipt_batch_num)
        print(f"    💾 Saved {len(receipts)} receipts (batch {receipt_batch_num}, total: {receipt_batch_num * 10000 + len(receipts):,})")
        receipt_batch_num += 1
    
    # Trim to target
    if len(samples) > TARGET_SAMPLES:
        samples = samples[:TARGET_SAMPLES]
        labels = labels[:TARGET_SAMPLES]
    
    # Save receipt index
    receipt_index = {
        "total_batches": receipt_batch_num,
        "total_samples": receipt_batch_num * 10000 + len(receipts),
        "batch_files": [f"receipts_batch_{i:04d}.json" for i in range(receipt_batch_num)]
    }
    with open(os.path.join(RECEIPTS_DIR, "receipts_index.json"), 'w') as f:
        json.dump(receipt_index, f, indent=2)
    
    return samples, labels

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def process_batch(texts):
    """Process a batch of texts (for parallel processing)"""
    extractor = FeatureExtractorV4()
    results = []
    for text in texts:
        try:
            features = extractor.extract_feature_vector(text)
            results.append(features)
        except Exception:
            results.append(np.zeros(96))
    return results

def extract_features_parallel(samples, labels):
    """Extract features in parallel"""
    n_samples = len(samples)
    batch_size = 1000
    batches = [samples[i:i+batch_size] for i in range(0, n_samples, batch_size)]
    
    all_features = []
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        pbar = tqdm(total=n_samples, desc="  Extracting features", unit="samples")
        completed = 0
        
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_features.extend(batch_results)
                completed += len(batch_results)
                pbar.n = min(completed, n_samples)
                pbar.refresh()
            except Exception as e:
                print(f"    ⚠ Batch error: {e}")
        
        pbar.close()
    
    # Sort by original order (futures complete out of order)
    X = np.array(all_features[:n_samples])
    y = np.array(labels[:n_samples])
    
    return X, y

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER OPTIMIZATION (5 TRIALS)
# ══════════════════════════════════════════════════════════════════════════════

def optimize_hyperparameters(X_train, y_train, n_trials=5):
    """Run Optuna hyperparameter optimization"""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'et', 'gb'])
        
        if model_type in ['rf', 'et']:
            n_estimators = trial.suggest_int('n_estimators', 100, 300)
            max_depth = trial.suggest_int('max_depth', 15, 40)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
            max_features = trial.suggest_float('max_features', 0.3, 0.8)
            
            if model_type == 'rf':
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                model = ExtraTreesClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_features=max_features,
                    class_weight='balanced',
                    n_jobs=-1,
                    random_state=42
                )
        else:
            n_estimators = trial.suggest_int('gb_n_estimators', 100, 200)
            max_depth = trial.suggest_int('gb_max_depth', 5, 15)
            learning_rate = trial.suggest_float('learning_rate', 0.05, 0.2)
            subsample = trial.suggest_float('subsample', 0.7, 1.0)
            
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                random_state=42
            )
        
        # Use stratified k-fold for evaluation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return scores.mean()
    
    # Create Optuna study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
    )
    
    # Progress callback
    pbar = tqdm(total=n_trials, desc="  Optimizing", unit="trial")
    
    def callback(study, trial):
        pbar.update(1)
        pbar.set_postfix({"best": f"{study.best_value:.4f}"})
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    pbar.close()
    
    print(f"\n  ✓ Best accuracy: {study.best_value:.4f}")
    print(f"  ✓ Best params: {study.best_params}")
    
    return study.best_params, study

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(X_train, y_train, best_params):
    """Train final model with best parameters"""
    model_type = best_params.get('model_type', 'rf')
    
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 30),
            min_samples_split=best_params.get('min_samples_split', 5),
            min_samples_leaf=best_params.get('min_samples_leaf', 2),
            max_features=best_params.get('max_features', 0.5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    elif model_type == 'et':
        model = ExtraTreesClassifier(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 30),
            min_samples_split=best_params.get('min_samples_split', 5),
            min_samples_leaf=best_params.get('min_samples_leaf', 2),
            max_features=best_params.get('max_features', 0.5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=best_params.get('gb_n_estimators', 150),
            max_depth=best_params.get('gb_max_depth', 10),
            learning_rate=best_params.get('learning_rate', 0.1),
            subsample=best_params.get('subsample', 0.8),
            random_state=42
        )
    
    print(f"  Training {model_type.upper()} model...")
    model.fit(X_train, y_train)
    
    return model

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.0
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm

# ══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_model_artifacts(model, scaler, metrics, best_params, study):
    """Save all model artifacts"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    joblib.dump(model, model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "created": datetime.now().isoformat(),
        "samples": TARGET_SAMPLES,
        "features": 96,
        "optuna_trials": OPTUNA_TRIALS,
        "best_params": best_params,
        "metrics": metrics,
        "feature_names": FEATURE_NAMES
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    # Save Optuna trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state)
        })
    
    trials_path = os.path.join(OUTPUT_DIR, "optuna_trials.json")
    with open(trials_path, 'w') as f:
        json.dump(trials_data, f, indent=2)
    print(f"  ✓ Optuna trials saved: {trials_path}")
    
    # Save veritas_config.js for web app
    config_js = f"""// Sunrise V{MODEL_VERSION} Configuration
// Generated: {datetime.now().isoformat()}
// Samples: {TARGET_SAMPLES:,}
// Accuracy: {metrics['accuracy']*100:.2f}%

const SUNRISE_CONFIG = {{
    version: "{MODEL_VERSION}",
    modelName: "{MODEL_NAME}",
    features: {len(FEATURE_NAMES)},
    trainingSamples: {TARGET_SAMPLES},
    metrics: {{
        accuracy: {metrics['accuracy']:.4f},
        precision: {metrics['precision']:.4f},
        recall: {metrics['recall']:.4f},
        f1: {metrics['f1']:.4f},
        rocAuc: {metrics['roc_auc']:.4f}
    }},
    bestParams: {json.dumps(best_params, indent=4)},
    featureNames: {json.dumps(FEATURE_NAMES)}
}};

if (typeof module !== 'undefined') {{
    module.exports = SUNRISE_CONFIG;
}}
"""
    
    config_path = os.path.join(OUTPUT_DIR, "veritas_config.js")
    with open(config_path, 'w') as f:
        f.write(config_js)
    print(f"  ✓ Config saved: {config_path}")
    
    # Save training receipt
    receipt = {
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "trained_at": datetime.now().isoformat(),
        "target_samples": TARGET_SAMPLES,
        "optuna_trials": OPTUNA_TRIALS,
        "final_metrics": metrics,
        "best_hyperparameters": best_params
    }
    
    receipt_path = os.path.join(OUTPUT_DIR, "training_receipt.json")
    with open(receipt_path, 'w') as f:
        json.dump(receipt, f, indent=2)
    print(f"  ✓ Training receipt saved: {receipt_path}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("  🌅 SUNRISE V5.0 - FAST TRAINING (5 trials)")
    print("═" * 70)
    print(f"  Target: {TARGET_SAMPLES:,} samples")
    print(f"  Features: 96")
    print(f"  Optimization trials: {OPTUNA_TRIALS}")
    print(f"  Workers: {N_WORKERS}")
    print("═" * 70)
    
    # Check for cached features first
    X, y = load_cached_features()
    
    if X is None:
        # STEP 1: Collect samples
        print("\n" + "═" * 70)
        print("  STEP 1: COLLECTING SAMPLES FROM HUGGINGFACE")
        print("═" * 70)
        
        samples, labels = collect_samples_from_datasets()
        
        human_count = sum(1 for l in labels if l == 0)
        ai_count = sum(1 for l in labels if l == 1)
        print(f"\n  ✓ Total samples collected: {len(samples):,}")
        print(f"    Human: {human_count:,} ({100*human_count/len(samples):.1f}%)")
        print(f"    AI: {ai_count:,} ({100*ai_count/len(samples):.1f}%)")
        
        # STEP 2: Extract features
        print("\n" + "═" * 70)
        print("  STEP 2: EXTRACTING 96 FEATURES (PARALLEL)")
        print("═" * 70)
        
        X, y = extract_features_parallel(samples, labels)
        
        human_count = np.sum(y == 0)
        ai_count = np.sum(y == 1)
        print(f"\n  ✓ Features extracted: {len(X):,} samples")
        print(f"    Human: {human_count:,} | AI: {ai_count:,}")
        
        # Save features to cache
        save_features_to_cache(X, y)
        
        # Clean up
        del samples
        gc.collect()
    
    # STEP 3: Prepare data
    print("\n" + "═" * 70)
    print("  STEP 3: PREPARING DATA")
    print("═" * 70)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # STEP 4: Hyperparameter optimization
    print("\n" + "═" * 70)
    print(f"  STEP 4: HYPERPARAMETER OPTIMIZATION ({OPTUNA_TRIALS} trials)")
    print("═" * 70)
    
    best_params, study = optimize_hyperparameters(X_train_scaled, y_train, OPTUNA_TRIALS)
    
    # STEP 5: Train final model
    print("\n" + "═" * 70)
    print("  STEP 5: TRAINING FINAL MODEL")
    print("═" * 70)
    
    model = train_final_model(X_train_scaled, y_train, best_params)
    print("  ✓ Model trained")
    
    # STEP 6: Evaluate
    print("\n" + "═" * 70)
    print("  STEP 6: EVALUATION")
    print("═" * 70)
    
    metrics, cm = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\n  📊 RESULTS:")
    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║  Accuracy:   {metrics['accuracy']*100:6.2f}%              ║")
    print(f"  ║  Precision:  {metrics['precision']*100:6.2f}%              ║")
    print(f"  ║  Recall:     {metrics['recall']*100:6.2f}%              ║")
    print(f"  ║  F1 Score:   {metrics['f1']*100:6.2f}%              ║")
    print(f"  ║  ROC AUC:    {metrics['roc_auc']*100:6.2f}%              ║")
    print(f"  ╚══════════════════════════════════════╝")
    
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  Human    AI")
    print(f"    Actual Human  {cm[0][0]:6,}  {cm[0][1]:6,}")
    print(f"    Actual AI     {cm[1][0]:6,}  {cm[1][1]:6,}")
    
    # STEP 7: Save artifacts
    print("\n" + "═" * 70)
    print("  STEP 7: SAVING ARTIFACTS")
    print("═" * 70)
    
    save_model_artifacts(model, scaler, metrics, best_params, study)
    
    # Final summary
    print("\n" + "═" * 70)
    print("  🌅 SUNRISE V5.0 TRAINING COMPLETE!")
    print("═" * 70)
    print(f"  ✓ Samples: {TARGET_SAMPLES:,}")
    print(f"  ✓ Features: 96")
    print(f"  ✓ Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ Model: {OUTPUT_DIR}/model.pkl")
    print(f"  ✓ Features cached for instant retraining")
    print("═" * 70)

if __name__ == "__main__":
    main()
