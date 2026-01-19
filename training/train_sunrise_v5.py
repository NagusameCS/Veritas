#!/usr/bin/env python3
"""
Sunrise V5.0 - Million Sample Training Pipeline
================================================
- 1,000,000+ samples from HuggingFace datasets
- Full provenance tracking for every sample
- Maximum accuracy optimization
- Comprehensive receipts with sample links, IDs, parameters
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
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
N_WORKERS = 15
OPTUNA_TRIALS = 100  # More trials for better optimization
BATCH_SIZE = 50000  # Process in batches to manage memory

# HuggingFace datasets configuration - comprehensive list
DATASETS_CONFIG = [
    # === HUMAN TEXT SOURCES ===
    # Wikipedia and encyclopedic
    {"name": "wikipedia", "config": "20220301.en", "split": "train", "text_field": "text", "label": 0, "max_samples": 150000, "source_type": "encyclopedia"},
    
    # Books and literature
    {"name": "bookcorpus", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 100000, "source_type": "books"},
    {"name": "pg19", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 50000, "source_type": "books"},
    
    # News articles
    {"name": "cc_news", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 80000, "source_type": "news"},
    {"name": "cnn_dailymail", "config": "3.0.0", "split": "train", "text_field": "article", "label": 0, "max_samples": 60000, "source_type": "news"},
    {"name": "multi_news", "config": None, "split": "train", "text_field": "document", "label": 0, "max_samples": 40000, "source_type": "news"},
    {"name": "xsum", "config": None, "split": "train", "text_field": "document", "label": 0, "max_samples": 50000, "source_type": "news"},
    
    # Academic and scientific
    {"name": "scientific_papers", "config": "arxiv", "split": "train", "text_field": "article", "label": 0, "max_samples": 40000, "source_type": "academic"},
    {"name": "scientific_papers", "config": "pubmed", "split": "train", "text_field": "article", "label": 0, "max_samples": 40000, "source_type": "academic"},
    
    # Q&A and forums
    {"name": "squad", "config": None, "split": "train", "text_field": "context", "label": 0, "max_samples": 30000, "source_type": "qa"},
    {"name": "eli5", "config": "LFQA_reddit", "split": "train_eli5", "text_field": "answers.text", "label": 0, "max_samples": 30000, "source_type": "forum"},
    {"name": "reddit_tifu", "config": "long", "split": "train", "text_field": "documents", "label": 0, "max_samples": 30000, "source_type": "forum"},
    
    # Reviews and opinions
    {"name": "imdb", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 25000, "source_type": "reviews"},
    {"name": "yelp_review_full", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 40000, "source_type": "reviews"},
    {"name": "amazon_polarity", "config": None, "split": "train", "text_field": "content", "label": 0, "max_samples": 50000, "source_type": "reviews"},
    
    # Social media and casual
    {"name": "tweet_eval", "config": "sentiment", "split": "train", "text_field": "text", "label": 0, "max_samples": 20000, "source_type": "social"},
    
    # Legal and formal
    {"name": "billsum", "config": None, "split": "train", "text_field": "text", "label": 0, "max_samples": 20000, "source_type": "legal"},
    {"name": "eur_lex_sum", "config": "english", "split": "train", "text_field": "reference", "label": 0, "max_samples": 15000, "source_type": "legal"},
    
    # Dialogue and conversation
    {"name": "daily_dialog", "config": None, "split": "train", "text_field": "dialog", "label": 0, "max_samples": 20000, "source_type": "dialogue"},
    {"name": "empathetic_dialogues", "config": None, "split": "train", "text_field": "utterance", "label": 0, "max_samples": 20000, "source_type": "dialogue"},
    
    # === AI-GENERATED TEXT SOURCES ===
    # GPT-generated datasets
    {"name": "aadityaubhat/GPT-wiki-intro", "config": None, "split": "train", "text_field": "generated_intro", "label": 1, "max_samples": 100000, "source_type": "gpt_wiki"},
    {"name": "Hello-SimpleAI/HC3", "config": "all", "split": "train", "text_field": "chatgpt_answers", "label": 1, "max_samples": 80000, "source_type": "chatgpt"},
    {"name": "Hello-SimpleAI/HC3-Chinese", "config": "all", "split": "train", "text_field": "chatgpt_answers", "label": 1, "max_samples": 30000, "source_type": "chatgpt"},
    
    # AI detection datasets
    {"name": "artem9k/ai-text-detection-pile", "config": None, "split": "train", "text_field": "text", "label": -1, "max_samples": 100000, "source_type": "mixed_ai"},
    {"name": "Hello-SimpleAI/chatgpt-comparison-corpus", "config": None, "split": "train", "text_field": "generated", "label": 1, "max_samples": 50000, "source_type": "chatgpt"},
    
    # LLM outputs
    {"name": "teknium/OpenHermes-2.5", "config": None, "split": "train", "text_field": "conversations", "label": 1, "max_samples": 80000, "source_type": "llm_chat"},
    {"name": "Open-Orca/OpenOrca", "config": None, "split": "train", "text_field": "response", "label": 1, "max_samples": 80000, "source_type": "llm_response"},
    {"name": "WizardLM/WizardLM_evol_instruct_V2_196k", "config": None, "split": "train", "text_field": "output", "label": 1, "max_samples": 60000, "source_type": "llm_instruct"},
    
    # Code and technical AI
    {"name": "codeparrot/github-code", "config": "all-all", "split": "train", "text_field": "code", "label": 0, "max_samples": 30000, "source_type": "code_human"},
    {"name": "sahil2801/CodeAlpaca-20k", "config": None, "split": "train", "text_field": "output", "label": 1, "max_samples": 20000, "source_type": "code_ai"},
    
    # Synthetic and paraphrased
    {"name": "humarin/chatgpt-paraphrases", "config": None, "split": "train", "text_field": "paraphrase", "label": 1, "max_samples": 40000, "source_type": "paraphrase_ai"},
    
    # Additional diverse sources
    {"name": "wikihow", "config": "all", "split": "train", "text_field": "text", "label": 0, "max_samples": 30000, "source_type": "howto"},
    {"name": "OpenAssistant/oasst1", "config": None, "split": "train", "text_field": "text", "label": -1, "max_samples": 50000, "source_type": "assistant"},
]

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE RECEIPT TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class SampleReceipt:
    """Tracks provenance for each sample"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.receipts = []
        self.batch_count = 0
        os.makedirs(output_dir, exist_ok=True)
        
    def add_sample(self, sample_data):
        """Add a sample receipt"""
        receipt = {
            "id": sample_data.get("id"),
            "hash": sample_data.get("hash"),
            "dataset": sample_data.get("dataset"),
            "config": sample_data.get("config"),
            "split": sample_data.get("split"),
            "index": sample_data.get("index"),
            "source_type": sample_data.get("source_type"),
            "label": sample_data.get("label"),
            "label_name": "AI" if sample_data.get("label") == 1 else "Human",
            "text_length": sample_data.get("text_length"),
            "word_count": sample_data.get("word_count"),
            "huggingface_url": f"https://huggingface.co/datasets/{sample_data.get('dataset')}",
            "timestamp": datetime.now().isoformat(),
            "text_preview": sample_data.get("text_preview", "")[:200],
        }
        self.receipts.append(receipt)
        
        # Save in batches to manage memory
        if len(self.receipts) >= 10000:
            self._save_batch()
            
    def _save_batch(self):
        """Save current batch of receipts"""
        if not self.receipts:
            return
            
        batch_file = os.path.join(self.output_dir, f"receipts_batch_{self.batch_count:04d}.json")
        with open(batch_file, 'w') as f:
            json.dump(self.receipts, f, indent=2)
        
        print(f"    💾 Saved {len(self.receipts)} receipts to batch {self.batch_count}")
        self.batch_count += 1
        self.receipts = []
        
    def finalize(self):
        """Save remaining receipts and create index"""
        self._save_batch()
        
        # Create index file
        index = {
            "total_batches": self.batch_count,
            "batch_files": [f"receipts_batch_{i:04d}.json" for i in range(self.batch_count)],
            "created": datetime.now().isoformat(),
        }
        
        index_file = os.path.join(self.output_dir, "receipts_index.json")
        with open(index_file, 'w') as f:
            json.dump(index, f, indent=2)
            
        return self.batch_count

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def generate_sample_id(text, dataset_name, index):
    """Generate unique ID for a sample"""
    content = f"{dataset_name}:{index}:{text[:100]}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]

def generate_text_hash(text):
    """Generate hash of text content for deduplication"""
    return hashlib.md5(text.encode()).hexdigest()

def extract_text_from_item(item, text_field, dataset_name):
    """Extract text from various dataset formats"""
    try:
        # Handle nested fields like "answers.text"
        if '.' in text_field:
            parts = text_field.split('.')
            value = item
            for part in parts:
                if isinstance(value, dict):
                    value = value.get(part, "")
                elif isinstance(value, list) and len(value) > 0:
                    value = value[0] if isinstance(value[0], str) else value[0].get(part, "") if isinstance(value[0], dict) else ""
                else:
                    value = ""
            return value if isinstance(value, str) else str(value) if value else ""
        
        value = item.get(text_field, "")
        
        # Handle list fields
        if isinstance(value, list):
            if len(value) > 0:
                if isinstance(value[0], str):
                    return " ".join(value)
                elif isinstance(value[0], dict):
                    # For conversation formats
                    texts = []
                    for v in value:
                        if isinstance(v, dict):
                            texts.append(v.get('value', v.get('content', v.get('text', str(v)))))
                        else:
                            texts.append(str(v))
                    return " ".join(texts)
            return ""
        
        return value if isinstance(value, str) else str(value) if value else ""
        
    except Exception as e:
        return ""

def load_dataset_samples(config, receipt_tracker, collected_hashes):
    """Load samples from a single dataset configuration"""
    from datasets import load_dataset, DownloadConfig
    
    dataset_name = config["name"]
    dataset_config = config.get("config")
    split = config["split"]
    text_field = config["text_field"]
    label = config["label"]
    max_samples = config["max_samples"]
    source_type = config["source_type"]
    
    samples = []
    
    try:
        print(f"  📥 Loading {dataset_name}" + (f" ({dataset_config})" if dataset_config else ""))
        
        # Configure download
        dl_config = DownloadConfig(
            resume_download=True,
            max_retries=3,
        )
        
        # Load dataset
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        
        # Sample if needed
        total_available = len(dataset)
        n_to_load = min(max_samples, total_available)
        
        if n_to_load < total_available:
            indices = np.random.choice(total_available, n_to_load, replace=False)
            dataset = dataset.select(indices.tolist())
        
        # Process samples
        for idx, item in enumerate(tqdm(dataset, desc=f"    Processing", leave=False)):
            text = extract_text_from_item(item, text_field, dataset_name)
            
            if not text or len(text) < 50:
                continue
            
            # Truncate very long texts
            if len(text) > 10000:
                text = text[:10000]
            
            # Handle mixed label datasets
            actual_label = label
            if label == -1:
                # Try to get label from dataset
                if 'label' in item:
                    actual_label = int(item['label'])
                elif 'is_generated' in item:
                    actual_label = 1 if item['is_generated'] else 0
                elif 'source' in item:
                    actual_label = 1 if 'gpt' in str(item['source']).lower() or 'ai' in str(item['source']).lower() else 0
                else:
                    continue  # Skip if can't determine label
            
            # Deduplication check
            text_hash = generate_text_hash(text)
            if text_hash in collected_hashes:
                continue
            collected_hashes.add(text_hash)
            
            # Generate sample ID
            sample_id = generate_sample_id(text, dataset_name, idx)
            
            # Create sample
            sample = {
                "text": text,
                "label": actual_label,
                "id": sample_id,
                "hash": text_hash,
                "dataset": dataset_name,
                "config": dataset_config,
                "split": split,
                "index": idx,
                "source_type": source_type,
                "text_length": len(text),
                "word_count": len(text.split()),
                "text_preview": text[:200],
            }
            
            samples.append(sample)
            receipt_tracker.add_sample(sample)
        
        print(f"    ✓ Collected {len(samples)} samples from {dataset_name}")
        
        # Cleanup
        del dataset
        gc.collect()
        
    except Exception as e:
        print(f"    ⚠ Error loading {dataset_name}: {str(e)[:100]}")
    
    return samples

def collect_all_samples(target_count, receipt_tracker):
    """Collect samples from all configured datasets"""
    print("\n" + "═" * 70)
    print("  STEP 1: COLLECTING 1M+ SAMPLES FROM HUGGINGFACE")
    print("═" * 70 + "\n")
    
    all_samples = []
    collected_hashes = set()
    
    # Calculate scaling factor to reach target
    total_configured = sum(d["max_samples"] for d in DATASETS_CONFIG)
    scale_factor = max(1.0, target_count / total_configured * 1.2)  # 20% buffer
    
    print(f"  Target: {target_count:,} samples")
    print(f"  Configured datasets: {len(DATASETS_CONFIG)}")
    print(f"  Scale factor: {scale_factor:.2f}\n")
    
    for config in DATASETS_CONFIG:
        if len(all_samples) >= target_count * 1.1:  # Stop if we have enough
            break
            
        # Scale up max_samples
        scaled_config = config.copy()
        scaled_config["max_samples"] = int(config["max_samples"] * scale_factor)
        
        samples = load_dataset_samples(scaled_config, receipt_tracker, collected_hashes)
        all_samples.extend(samples)
        
        print(f"  📊 Total collected: {len(all_samples):,} / {target_count:,}")
        
        # Memory management
        if len(all_samples) % 100000 == 0:
            gc.collect()
    
    # Trim to exact target if needed
    if len(all_samples) > target_count:
        all_samples = all_samples[:target_count]
    
    # Statistics
    human_count = sum(1 for s in all_samples if s["label"] == 0)
    ai_count = sum(1 for s in all_samples if s["label"] == 1)
    
    print(f"\n  ✓ Total samples collected: {len(all_samples):,}")
    print(f"    Human: {human_count:,} ({human_count/len(all_samples)*100:.1f}%)")
    print(f"    AI: {ai_count:,} ({ai_count/len(all_samples)*100:.1f}%)")
    
    return all_samples

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_single_feature(args):
    """Extract features for a single sample"""
    sample, idx = args
    try:
        features = extract_features_v4(sample["text"])
        return idx, features, sample["label"], sample["id"]
    except Exception as e:
        return idx, None, None, None

def extract_all_features(samples):
    """Extract features from all samples with parallel processing"""
    print("\n" + "═" * 70)
    print("  STEP 2: EXTRACTING 96 FEATURES (PARALLEL)")
    print("═" * 70 + "\n")
    
    features_list = []
    labels_list = []
    sample_ids = []
    
    # Prepare args
    args_list = [(sample, idx) for idx, sample in enumerate(samples)]
    
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(extract_single_feature, args): args[1] for args in args_list}
        
        with tqdm(total=len(samples), desc="  Extracting features", unit="samples") as pbar:
            for future in as_completed(futures):
                idx, features, label, sample_id = future.result()
                if features is not None:
                    features_list.append(features)
                    labels_list.append(label)
                    sample_ids.append(sample_id)
                pbar.update(1)
    
    X = np.array(features_list)
    y = np.array(labels_list)
    
    # Clean invalid values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\n  ✓ Features extracted: {len(features_list):,} samples")
    print(f"    Human: {sum(y == 0):,} | AI: {sum(y == 1):,}")
    
    return X, y, sample_ids

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER OPTIMIZATION
# ══════════════════════════════════════════════════════════════════════════════

def create_optuna_objective(X_train, y_train, X_val, y_val):
    """Create Optuna objective for hyperparameter optimization"""
    
    def objective(trial):
        # Model selection
        model_type = trial.suggest_categorical("model_type", ["rf", "et", "gb"])
        
        if model_type == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_jobs": -1,
                "random_state": 42,
            }
            model = RandomForestClassifier(**params)
            
        elif model_type == "et":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 10, 50),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "n_jobs": -1,
                "random_state": 42,
            }
            model = ExtraTreesClassifier(**params)
            
        else:  # Gradient Boosting
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
                "random_state": 42,
            }
            model = GradientBoostingClassifier(**params)
        
        # Train and evaluate
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        return accuracy
    
    return objective

def optimize_hyperparameters(X_train, y_train, n_trials=100):
    """Run extensive hyperparameter optimization"""
    print("\n" + "═" * 70)
    print(f"  STEP 4: HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
    print("═" * 70 + "\n")
    
    # Create validation split
    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=42)
    
    # Create study
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    objective = create_optuna_objective(X_t, y_t, X_val, y_val)
    
    # Optimize with progress bar
    with tqdm(total=n_trials, desc="  Optimizing", unit="trial") as pbar:
        def callback(study, trial):
            pbar.update(1)
            pbar.set_postfix({"best": f"{study.best_value:.4f}"})
        
        study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    
    print(f"\n  ✓ Best accuracy: {study.best_value:.4f}")
    print(f"  ✓ Best model type: {study.best_params.get('model_type', 'rf')}")
    print(f"  ✓ Best params: {study.best_params}")
    
    return study.best_params, study

# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING AND EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def train_final_model(X_train, y_train, best_params):
    """Train final model with best parameters"""
    print("\n" + "═" * 70)
    print("  STEP 5: TRAINING FINAL MODEL")
    print("═" * 70 + "\n")
    
    model_type = best_params.pop("model_type", "rf")
    
    # Remove any extra parameters that don't belong
    valid_rf_params = ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", 
                       "max_features", "class_weight", "bootstrap", "n_jobs", "random_state"]
    
    if model_type == "rf":
        params = {k: v for k, v in best_params.items() if k in valid_rf_params}
        params["n_jobs"] = -1
        params["random_state"] = 42
        model = RandomForestClassifier(**params)
        print(f"  Training RandomForest with {params.get('n_estimators', 100)} estimators...")
        
    elif model_type == "et":
        params = {k: v for k, v in best_params.items() if k in valid_rf_params}
        params["n_jobs"] = -1
        params["random_state"] = 42
        model = ExtraTreesClassifier(**params)
        print(f"  Training ExtraTrees with {params.get('n_estimators', 100)} estimators...")
        
    else:
        valid_gb_params = ["n_estimators", "max_depth", "learning_rate", "min_samples_split", 
                          "min_samples_leaf", "subsample", "max_features", "random_state"]
        params = {k: v for k, v in best_params.items() if k in valid_gb_params}
        params["random_state"] = 42
        model = GradientBoostingClassifier(**params)
        print(f"  Training GradientBoosting with {params.get('n_estimators', 100)} estimators...")
    
    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    print(f"  ✓ Model trained in {train_time:.1f}s")
    
    # Add model_type back for saving
    best_params["model_type"] = model_type
    
    return model, model_type

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    print("\n" + "═" * 70)
    print("  STEP 6: EVALUATION")
    print("═" * 70 + "\n")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    
    cm = confusion_matrix(y_test, y_pred)
    
    print("  ┌" + "─" * 50 + "┐")
    print(f"  │  {'Metric':<40} {'Value':>6}  │")
    print("  ├" + "─" * 50 + "┤")
    print(f"  │  {'Accuracy':<40} {metrics['accuracy']*100:>5.2f}%  │")
    print(f"  │  {'Precision':<40} {metrics['precision']*100:>5.2f}%  │")
    print(f"  │  {'Recall':<40} {metrics['recall']*100:>5.2f}%  │")
    print(f"  │  {'F1 Score':<40} {metrics['f1']*100:>5.2f}%  │")
    print(f"  │  {'ROC-AUC':<40} {metrics['roc_auc']:>6.4f}  │")
    print("  └" + "─" * 50 + "┘")
    
    print(f"\n  Confusion Matrix:")
    print(f"    TN: {cm[0][0]:,}  FP: {cm[0][1]:,}")
    print(f"    FN: {cm[1][0]:,}  TP: {cm[1][1]:,}")
    
    return metrics, cm

def get_feature_importance(model, model_type):
    """Extract feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\n  Top 20 Most Important Features:")
        print("  " + "─" * 50)
        for i in range(min(20, len(indices))):
            idx = indices[i]
            name = FEATURE_NAMES[idx] if idx < len(FEATURE_NAMES) else f"feature_{idx}"
            bar = "█" * int(importances[idx] * 50)
            print(f"  {i+1:2}. {name:<30} {importances[idx]:.4f} {bar}")
        
        return {FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feature_{i}": float(importances[i]) 
                for i in range(len(importances))}
    return {}

# ══════════════════════════════════════════════════════════════════════════════
# MODEL SAVING
# ══════════════════════════════════════════════════════════════════════════════

def save_model(model, scaler, metrics, best_params, feature_importance, 
               training_stats, study, output_dir):
    """Save all model artifacts"""
    print("\n" + "═" * 70)
    print("  STEP 7: SAVING MODEL")
    print("═" * 70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, "model.pkl"))
    print("  ✓ Saved: model.pkl")
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    print("  ✓ Saved: scaler.pkl")
    
    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "created": datetime.now().isoformat(),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1"],
        "roc_auc": metrics["roc_auc"],
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "best_params": best_params,
        "training_stats": training_stats,
        "feature_importance": feature_importance,
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Saved: metadata.json")
    
    # Save training receipt
    receipt = {
        "model_name": f"{MODEL_NAME} V{MODEL_VERSION}",
        "training_completed": datetime.now().isoformat(),
        "total_samples": training_stats["total_samples"],
        "human_samples": training_stats["human_samples"],
        "ai_samples": training_stats["ai_samples"],
        "training_samples": training_stats["training_samples"],
        "testing_samples": training_stats["testing_samples"],
        "feature_count": len(FEATURE_NAMES),
        "optimization_trials": OPTUNA_TRIALS,
        "best_model_type": best_params.get("model_type", "rf"),
        "best_accuracy": study.best_value,
        "final_accuracy": metrics["accuracy"],
        "datasets_used": [d["name"] for d in DATASETS_CONFIG],
        "receipts_location": RECEIPTS_DIR,
    }
    
    with open(os.path.join(output_dir, "training_receipt.json"), 'w') as f:
        json.dump(receipt, f, indent=2)
    print("  ✓ Saved: training_receipt.json")
    
    # Save Optuna study trials
    trials_data = []
    for trial in study.trials:
        trials_data.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": str(trial.state),
        })
    
    with open(os.path.join(output_dir, "optuna_trials.json"), 'w') as f:
        json.dump(trials_data, f, indent=2)
    print("  ✓ Saved: optuna_trials.json")
    
    # Save veritas_config.js for frontend
    config_js = f"""// Sunrise V{MODEL_VERSION} - Trained {datetime.now().strftime('%Y-%m-%d %H:%M')}
// {training_stats['total_samples']:,} samples | {len(FEATURE_NAMES)} features | {metrics['accuracy']*100:.2f}% accuracy

const VERITAS_MODEL_CONFIG = {{
    name: "{MODEL_NAME}",
    version: "{MODEL_VERSION}",
    accuracy: {metrics['accuracy']:.6f},
    precision: {metrics['precision']:.6f},
    recall: {metrics['recall']:.6f},
    f1Score: {metrics['f1']:.6f},
    rocAuc: {metrics['roc_auc']:.6f},
    features: {len(FEATURE_NAMES)},
    trainingSamples: {training_stats['total_samples']},
    optimizationTrials: {OPTUNA_TRIALS},
    bestModelType: "{best_params.get('model_type', 'rf')}",
    trained: "{datetime.now().isoformat()}",
    featureImportance: {json.dumps({k: round(v, 6) for k, v in sorted(feature_importance.items(), key=lambda x: -x[1])[:20]}, indent=8)}
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_MODEL_CONFIG;
}}
"""
    
    with open(os.path.join(output_dir, "veritas_config.js"), 'w') as f:
        f.write(config_js)
    print("  ✓ Saved: veritas_config.js")
    
    print(f"\n  📁 Model saved to: {output_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    
    print("\n" + "═" * 70)
    print(f"  🌅 SUNRISE V{MODEL_VERSION} - MILLION SAMPLE TRAINING")
    print("═" * 70)
    print(f"  Target: {TARGET_SAMPLES:,} samples")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Optimization trials: {OPTUNA_TRIALS}")
    print(f"  Workers: {N_WORKERS}")
    print("═" * 70)
    
    # Initialize receipt tracker
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    receipt_tracker = SampleReceipt(RECEIPTS_DIR)
    
    # Step 1: Collect samples
    samples = collect_all_samples(TARGET_SAMPLES, receipt_tracker)
    
    # Finalize receipts
    batch_count = receipt_tracker.finalize()
    print(f"\n  📝 Saved {batch_count} receipt batches to {RECEIPTS_DIR}")
    
    # Step 2: Extract features
    X, y, sample_ids = extract_all_features(samples)
    
    # Free memory
    del samples
    gc.collect()
    
    # Step 3: Prepare data
    print("\n" + "═" * 70)
    print("  STEP 3: PREPARING DATA")
    print("═" * 70 + "\n")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    
    training_stats = {
        "total_samples": len(X),
        "human_samples": int(sum(y == 0)),
        "ai_samples": int(sum(y == 1)),
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
    }
    
    # Step 4: Hyperparameter optimization
    best_params, study = optimize_hyperparameters(X_train, y_train, OPTUNA_TRIALS)
    
    # Step 5: Train final model
    model, model_type = train_final_model(X_train, y_train, best_params.copy())
    
    # Step 6: Evaluate
    metrics, confusion = evaluate_model(model, X_test, y_test)
    feature_importance = get_feature_importance(model, model_type)
    
    # Step 7: Save
    save_model(model, scaler, metrics, best_params, feature_importance, 
               training_stats, study, OUTPUT_DIR)
    
    # Summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "═" * 70)
    print("  ✓ TRAINING COMPLETE")
    print("═" * 70)
    print(f"  Model: {MODEL_NAME} V{MODEL_VERSION}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  Features: {len(FEATURE_NAMES)}")
    print(f"  Training samples: {training_stats['total_samples']:,}")
    print(f"  Receipt batches: {batch_count}")
    print(f"  Time: {hours}h {minutes}m {seconds}s")
    print("═" * 70 + "\n")

if __name__ == "__main__":
    main()
