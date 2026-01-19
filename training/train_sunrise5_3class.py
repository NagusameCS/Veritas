#!/usr/bin/env python3
"""
Sunrise 5 - Three-Class Detection Model
=========================================
Distinguishes between:
  - Class 0: HUMAN - Pure human-written text
  - Class 1: AI - Raw AI-generated text  
  - Class 2: HUMANIZED - AI text that's been paraphrased/edited to seem human

Target: ~1,000,000 samples across all three classes
"""

import os
import sys
import json
import hashlib
import time
import warnings
import gc
import re
import random
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ML imports
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import optuna

# Feature extractor
from feature_extractor_v4 import FeatureExtractorV4

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

TARGET_SAMPLES = 1_000_000
MODEL_NAME = "Sunrise"
MODEL_VERSION = "6.0"
OUTPUT_DIR = "models/SunriseV6"
CACHE_DIR = "models/SunriseV6/cache"
RECEIPTS_DIR = "models/SunriseV6/receipts"
N_WORKERS = 15
OPTUNA_TRIALS = 1  # Single trial for fastest training

# Class labels
CLASS_HUMAN = 0
CLASS_AI = 1
CLASS_HUMANIZED = 2

CLASS_NAMES = {0: "Human", 1: "AI", 2: "Humanized"}

# Target distribution (roughly balanced with slight bias toward what we can collect)
TARGET_HUMAN = 400_000      # 40%
TARGET_AI = 350_000         # 35%  
TARGET_HUMANIZED = 250_000  # 25% (hardest to get - we'll generate some)

# ══════════════════════════════════════════════════════════════════════════════
# DATASETS CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Human text sources
HUMAN_DATASETS = [
    {"name": "cc_news", "config": None, "split": "train", "text_field": "text", "max_samples": 150000},
    {"name": "cnn_dailymail", "config": "3.0.0", "split": "train", "text_field": "article", "max_samples": 100000},
    {"name": "xsum", "config": None, "split": "train", "text_field": "document", "max_samples": 80000},
    {"name": "yelp_review_full", "config": None, "split": "train", "text_field": "text", "max_samples": 50000},
    {"name": "imdb", "config": None, "split": "train", "text_field": "text", "max_samples": 25000},
    {"name": "amazon_polarity", "config": None, "split": "train", "text_field": "content", "max_samples": 50000},
]

# AI text sources
AI_DATASETS = [
    {"name": "aadityaubhat/GPT-wiki-intro", "config": None, "split": "train", "text_field": "generated_intro", "max_samples": 150000},
    {"name": "teknium/OpenHermes-2.5", "config": None, "split": "train", "text_field": "conversations", "max_samples": 100000},
    {"name": "artem9k/ai-text-detection-pile", "config": None, "split": "train", "text_field": "text", "max_samples": 150000, "label_field": "generated"},
]

# Humanized AI datasets (datasets with paraphrased/rewritten AI content)
HUMANIZED_DATASETS = [
    # DIPPER paraphrased datasets
    {"name": "kalpeshk2011/dipper-paraphrased-text", "config": None, "split": "train", "text_field": "output", "max_samples": 100000},
    # AI text that's been post-edited
    {"name": "humarin/chatgpt-paraphrases", "config": None, "split": "train", "text_field": "paraphrase", "max_samples": 50000},
]

# ══════════════════════════════════════════════════════════════════════════════
# HUMANIZATION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

class TextHumanizer:
    """
    Simulates humanization techniques used to disguise AI text.
    These are common patterns used by people trying to bypass AI detection.
    """
    
    def __init__(self):
        # Common filler words humans use
        self.fillers = ["basically", "actually", "honestly", "literally", "you know", 
                        "I mean", "like", "kind of", "sort of", "pretty much"]
        
        # Contractions map
        self.contractions = {
            "I am": "I'm", "you are": "you're", "he is": "he's", "she is": "she's",
            "it is": "it's", "we are": "we're", "they are": "they're",
            "I will": "I'll", "you will": "you'll", "we will": "we'll",
            "I have": "I've", "you have": "you've", "we have": "we've",
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't",
            "cannot": "can't", "could not": "couldn't", "would not": "wouldn't",
            "should not": "shouldn't", "will not": "won't",
        }
        
        # Informal replacements
        self.informal = {
            "however": "but", "therefore": "so", "additionally": "also",
            "furthermore": "plus", "consequently": "so", "nevertheless": "still",
            "utilize": "use", "implement": "do", "facilitate": "help",
            "demonstrate": "show", "indicate": "show", "commence": "start",
            "terminate": "end", "approximately": "about", "sufficient": "enough",
        }
        
    def add_typos(self, text: str, rate: float = 0.02) -> str:
        """Add realistic typos"""
        words = text.split()
        result = []
        
        typo_patterns = [
            lambda w: w[:-2] + w[-1] + w[-2] if len(w) > 3 else w,  # swap last two
            lambda w: w[0] + w[2] + w[1] + w[3:] if len(w) > 3 else w,  # swap middle
            lambda w: w + w[-1] if len(w) > 2 else w,  # double last letter
            lambda w: w[:-1] if len(w) > 3 else w,  # drop last letter
        ]
        
        for word in words:
            if random.random() < rate and len(word) > 3 and word.isalpha():
                pattern = random.choice(typo_patterns)
                word = pattern(word)
            result.append(word)
        
        return " ".join(result)
    
    def add_contractions(self, text: str) -> str:
        """Convert formal phrases to contractions"""
        for formal, informal in self.contractions.items():
            text = re.sub(rf'\b{formal}\b', informal, text, flags=re.IGNORECASE)
        return text
    
    def add_fillers(self, text: str, rate: float = 0.05) -> str:
        """Insert filler words"""
        sentences = text.split('. ')
        result = []
        
        for sent in sentences:
            if random.random() < rate and len(sent) > 20:
                filler = random.choice(self.fillers)
                words = sent.split()
                if len(words) > 3:
                    pos = random.randint(1, min(3, len(words)-1))
                    words.insert(pos, filler + ",")
                    sent = " ".join(words)
            result.append(sent)
        
        return '. '.join(result)
    
    def make_informal(self, text: str) -> str:
        """Replace formal words with informal equivalents"""
        for formal, informal in self.informal.items():
            text = re.sub(rf'\b{formal}\b', informal, text, flags=re.IGNORECASE)
        return text
    
    def vary_punctuation(self, text: str) -> str:
        """Add natural punctuation variations"""
        # Sometimes people use multiple punctuation
        text = re.sub(r'!', lambda m: '!' if random.random() > 0.1 else '!!', text)
        text = re.sub(r'\?', lambda m: '?' if random.random() > 0.1 else '??', text)
        
        # Sometimes commas are forgotten
        if random.random() < 0.1:
            text = text.replace(', ', ' ', 1)
        
        return text
    
    def add_personal_phrases(self, text: str) -> str:
        """Add personal/subjective phrases"""
        starters = ["I think ", "In my opinion, ", "From what I understand, ", 
                    "As far as I know, ", "I believe ", "It seems like "]
        
        sentences = text.split('. ')
        if len(sentences) > 2 and random.random() < 0.3:
            idx = random.randint(0, min(2, len(sentences)-1))
            sentences[idx] = random.choice(starters) + sentences[idx].lower()
        
        return '. '.join(sentences)
    
    def humanize(self, text: str, intensity: str = "medium") -> str:
        """
        Apply humanization transforms to AI text.
        
        intensity: "light", "medium", "heavy"
        """
        if intensity == "light":
            # Just contractions and slight informality
            text = self.add_contractions(text)
            if random.random() < 0.3:
                text = self.make_informal(text)
            return text
        
        elif intensity == "medium":
            # Contractions + fillers + informality
            text = self.add_contractions(text)
            text = self.make_informal(text)
            text = self.add_fillers(text, rate=0.03)
            if random.random() < 0.2:
                text = self.add_personal_phrases(text)
            return text
        
        else:  # heavy
            # Everything including typos
            text = self.add_contractions(text)
            text = self.make_informal(text)
            text = self.add_fillers(text, rate=0.05)
            text = self.add_personal_phrases(text)
            text = self.add_typos(text, rate=0.01)
            text = self.vary_punctuation(text)
            return text


# ══════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════

def collect_from_dataset(ds_config: dict, target_class: int, max_samples: int, seen_hashes: set) -> Tuple[List[str], List[int], List[dict]]:
    """Collect samples from a single dataset"""
    from datasets import load_dataset
    
    samples = []
    labels = []
    receipts = []
    
    name = ds_config["name"]
    config = ds_config.get("config")
    split = ds_config["split"]
    text_field = ds_config["text_field"]
    label_field = ds_config.get("label_field")
    max_samples = min(ds_config.get("max_samples", max_samples), max_samples)
    
    config_str = f" ({config})" if config else ""
    print(f"    📥 Loading {name}{config_str}")
    
    try:
        if config:
            dataset = load_dataset(name, config, split=split, trust_remote_code=True)
        else:
            dataset = load_dataset(name, split=split, trust_remote_code=True)
        
        available = len(dataset)
        to_sample = min(max_samples, available)
        print(f"      📊 Available: {available:,}, sampling: {to_sample:,}")
        
        if to_sample < available:
            indices = np.random.choice(available, to_sample, replace=False)
        else:
            indices = range(available)
        
        for idx in tqdm(indices, desc="      Processing", leave=False):
            row = dataset[int(idx)]
            
            # Get text based on field type
            if text_field == "conversations":
                convs = row.get("conversations", [])
                text = " ".join([c.get("value", "") for c in convs if isinstance(c, dict)])
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
            if label_field and target_class == CLASS_AI:
                # Dataset has its own labels (for AI detection datasets)
                label_val = row.get(label_field, 0)
                if isinstance(label_val, str):
                    label_val = 1 if label_val.lower() in ["ai", "generated", "1", "true"] else 0
                if label_val != 1:  # We only want AI samples from this
                    continue
                label = CLASS_AI
            else:
                label = target_class
            
            samples.append(text)
            labels.append(label)
            
            receipt = {
                "id": f"{name}_{idx}",
                "dataset": name,
                "index": int(idx),
                "label": int(label),
                "class_name": CLASS_NAMES[label],
                "text_hash": text_hash,
                "text_length": len(text),
                "timestamp": datetime.now().isoformat()
            }
            receipts.append(receipt)
            
            if len(samples) >= max_samples:
                break
        
        print(f"      ✓ Collected {len(samples):,} samples")
        
    except Exception as e:
        print(f"      ⚠ Error: {e}")
    
    return samples, labels, receipts


def generate_humanized_samples(ai_texts: List[str], target_count: int) -> Tuple[List[str], List[dict]]:
    """Generate humanized versions of AI text"""
    print(f"\n    🔄 Generating {target_count:,} humanized samples from AI text...")
    
    humanizer = TextHumanizer()
    humanized = []
    receipts = []
    
    # Shuffle and select AI texts to humanize
    selected = random.sample(ai_texts, min(target_count, len(ai_texts)))
    
    for i, text in enumerate(tqdm(selected, desc="      Humanizing", leave=False)):
        # Vary intensity
        intensity = random.choice(["light", "medium", "medium", "heavy"])
        humanized_text = humanizer.humanize(text, intensity)
        
        humanized.append(humanized_text)
        
        receipt = {
            "id": f"humanized_{i}",
            "source": "generated",
            "method": f"humanization_{intensity}",
            "original_hash": hashlib.md5(text.encode()[:1000]).hexdigest(),
            "label": CLASS_HUMANIZED,
            "class_name": "Humanized",
            "text_length": len(humanized_text),
            "timestamp": datetime.now().isoformat()
        }
        receipts.append(receipt)
    
    print(f"      ✓ Generated {len(humanized):,} humanized samples")
    
    return humanized, receipts


def collect_all_samples():
    """Collect samples for all three classes"""
    all_samples = []
    all_labels = []
    all_receipts = []
    seen_hashes = set()
    ai_texts_for_humanization = []
    
    os.makedirs(RECEIPTS_DIR, exist_ok=True)
    
    # ══════════════════════════════════════════════════════════════════════
    # COLLECT HUMAN TEXT
    # ══════════════════════════════════════════════════════════════════════
    print("\n  📚 Collecting HUMAN text...")
    human_samples = []
    human_labels = []
    
    for ds_config in HUMAN_DATASETS:
        remaining = TARGET_HUMAN - len(human_samples)
        if remaining <= 0:
            break
        
        samples, labels, receipts = collect_from_dataset(
            ds_config, CLASS_HUMAN, remaining, seen_hashes
        )
        human_samples.extend(samples)
        human_labels.extend(labels)
        all_receipts.extend(receipts)
    
    print(f"  ✓ Human samples: {len(human_samples):,}")
    
    # ══════════════════════════════════════════════════════════════════════
    # COLLECT AI TEXT
    # ══════════════════════════════════════════════════════════════════════
    print("\n  🤖 Collecting AI text...")
    ai_samples = []
    ai_labels = []
    
    for ds_config in AI_DATASETS:
        remaining = TARGET_AI - len(ai_samples)
        if remaining <= 0:
            break
        
        samples, labels, receipts = collect_from_dataset(
            ds_config, CLASS_AI, remaining, seen_hashes
        )
        ai_samples.extend(samples)
        ai_labels.extend(labels)
        all_receipts.extend(receipts)
        
        # Save some AI text for humanization
        ai_texts_for_humanization.extend(samples[:50000])
    
    print(f"  ✓ AI samples: {len(ai_samples):,}")
    
    # ══════════════════════════════════════════════════════════════════════
    # COLLECT & GENERATE HUMANIZED TEXT
    # ══════════════════════════════════════════════════════════════════════
    print("\n  🎭 Collecting HUMANIZED text...")
    humanized_samples = []
    humanized_labels = []
    
    # First try to get from actual humanized/paraphrased datasets
    for ds_config in HUMANIZED_DATASETS:
        remaining = TARGET_HUMANIZED - len(humanized_samples)
        if remaining <= 0:
            break
        
        try:
            samples, labels, receipts = collect_from_dataset(
                ds_config, CLASS_HUMANIZED, remaining, seen_hashes
            )
            humanized_samples.extend(samples)
            humanized_labels.extend([CLASS_HUMANIZED] * len(samples))
            all_receipts.extend(receipts)
        except Exception as e:
            print(f"      ⚠ Dataset not available: {e}")
    
    # Generate remaining humanized samples from AI text
    remaining_humanized = TARGET_HUMANIZED - len(humanized_samples)
    if remaining_humanized > 0 and ai_texts_for_humanization:
        generated, receipts = generate_humanized_samples(
            ai_texts_for_humanization, remaining_humanized
        )
        humanized_samples.extend(generated)
        humanized_labels.extend([CLASS_HUMANIZED] * len(generated))
        all_receipts.extend(receipts)
    
    print(f"  ✓ Humanized samples: {len(humanized_samples):,}")
    
    # ══════════════════════════════════════════════════════════════════════
    # COMBINE ALL
    # ══════════════════════════════════════════════════════════════════════
    all_samples = human_samples + ai_samples + humanized_samples
    all_labels = human_labels + ai_labels + humanized_labels
    
    # Shuffle
    combined = list(zip(all_samples, all_labels))
    random.shuffle(combined)
    all_samples, all_labels = zip(*combined)
    all_samples = list(all_samples)
    all_labels = list(all_labels)
    
    # Save receipts
    receipt_batches = [all_receipts[i:i+10000] for i in range(0, len(all_receipts), 10000)]
    for i, batch in enumerate(receipt_batches):
        path = os.path.join(RECEIPTS_DIR, f"receipts_batch_{i:04d}.json")
        with open(path, 'w') as f:
            json.dump(batch, f)
    
    # Save index
    receipt_index = {
        "total_batches": len(receipt_batches),
        "total_samples": len(all_receipts),
        "class_distribution": {
            "human": len(human_samples),
            "ai": len(ai_samples),
            "humanized": len(humanized_samples)
        }
    }
    with open(os.path.join(RECEIPTS_DIR, "receipts_index.json"), 'w') as f:
        json.dump(receipt_index, f, indent=2)
    
    return all_samples, all_labels


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def process_batch(texts):
    """Process a batch of texts"""
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
    
    X = np.array(all_features[:n_samples])
    y = np.array(labels[:n_samples])
    
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# CACHING
# ══════════════════════════════════════════════════════════════════════════════

def get_cache_path():
    return os.path.join(CACHE_DIR, "features_3class.npz")

def load_cached_features():
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
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = get_cache_path()
    print(f"\n  💾 Saving features to cache: {cache_path}")
    np.savez_compressed(cache_path, X=X, y=y)
    size_mb = os.path.getsize(cache_path) / (1024 * 1024)
    print(f"  ✓ Cache saved ({size_mb:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def optimize_hyperparameters(X_train, y_train, n_trials=3):
    """Optimize for 3-class classification"""
    
    def objective(trial):
        model_type = trial.suggest_categorical('model_type', ['rf', 'et'])
        
        n_estimators = trial.suggest_int('n_estimators', 150, 300)
        max_depth = trial.suggest_int('max_depth', 20, 45)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)
        max_features = trial.suggest_float('max_features', 0.3, 0.7)
        
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
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return scores.mean()
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
    )
    
    pbar = tqdm(total=n_trials, desc="  Optimizing", unit="trial")
    
    def callback(study, trial):
        pbar.update(1)
        pbar.set_postfix({"best": f"{study.best_value:.4f}"})
    
    study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
    pbar.close()
    
    print(f"\n  ✓ Best accuracy: {study.best_value:.4f}")
    print(f"  ✓ Best params: {study.best_params}")
    
    return study.best_params, study


def train_final_model(X_train, y_train, best_params):
    """Train final 3-class model"""
    model_type = best_params.get('model_type', 'rf')
    
    if model_type == 'rf':
        model = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 35),
            min_samples_split=best_params.get('min_samples_split', 5),
            min_samples_leaf=best_params.get('min_samples_leaf', 2),
            max_features=best_params.get('max_features', 0.5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 35),
            min_samples_split=best_params.get('min_samples_split', 5),
            min_samples_leaf=best_params.get('min_samples_leaf', 2),
            max_features=best_params.get('max_features', 0.5),
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
    
    print(f"  Training {model_type.upper()} model...")
    model.fit(X_train, y_train)
    
    return model


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test):
    """Evaluate 3-class model"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    
    # Per-class metrics
    for cls_id, cls_name in CLASS_NAMES.items():
        y_binary = (y_test == cls_id).astype(int)
        y_pred_binary = (y_pred == cls_id).astype(int)
        metrics[f'precision_{cls_name.lower()}'] = precision_score(y_binary, y_pred_binary, zero_division=0)
        metrics[f'recall_{cls_name.lower()}'] = recall_score(y_binary, y_pred_binary, zero_division=0)
        metrics[f'f1_{cls_name.lower()}'] = f1_score(y_binary, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=list(CLASS_NAMES.values()))
    
    return metrics, cm, report


# ══════════════════════════════════════════════════════════════════════════════
# SAVE ARTIFACTS
# ══════════════════════════════════════════════════════════════════════════════

def save_model_artifacts(model, scaler, metrics, best_params, study, cm):
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
    
    # Get feature names
    extractor = FeatureExtractorV4()
    feature_names = extractor.get_feature_names()
    
    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "version": MODEL_VERSION,
        "model_type": "3-class",
        "classes": CLASS_NAMES,
        "created": datetime.now().isoformat(),
        "samples": TARGET_SAMPLES,
        "features": 96,
        "optuna_trials": OPTUNA_TRIALS,
        "best_params": best_params,
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "feature_names": feature_names
    }
    
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved: {metadata_path}")
    
    # Save veritas_config.js
    config_js = f"""// Sunrise5 3-Class Configuration
// Generated: {datetime.now().isoformat()}
// Classes: Human (0), AI (1), Humanized (2)
// Accuracy: {metrics['accuracy']*100:.2f}%

const SUNRISE5_CONFIG = {{
    version: "{MODEL_VERSION}",
    modelName: "{MODEL_NAME}",
    modelType: "3-class",
    classes: {json.dumps(CLASS_NAMES)},
    features: {len(feature_names)},
    trainingSamples: {TARGET_SAMPLES},
    metrics: {{
        accuracy: {metrics['accuracy']:.4f},
        precision_macro: {metrics['precision_macro']:.4f},
        recall_macro: {metrics['recall_macro']:.4f},
        f1_macro: {metrics['f1_macro']:.4f},
        // Per-class metrics
        human: {{
            precision: {metrics.get('precision_human', 0):.4f},
            recall: {metrics.get('recall_human', 0):.4f},
            f1: {metrics.get('f1_human', 0):.4f}
        }},
        ai: {{
            precision: {metrics.get('precision_ai', 0):.4f},
            recall: {metrics.get('recall_ai', 0):.4f},
            f1: {metrics.get('f1_ai', 0):.4f}
        }},
        humanized: {{
            precision: {metrics.get('precision_humanized', 0):.4f},
            recall: {metrics.get('recall_humanized', 0):.4f},
            f1: {metrics.get('f1_humanized', 0):.4f}
        }}
    }},
    confusionMatrix: {json.dumps(cm.tolist())},
    bestParams: {json.dumps(best_params, indent=4)},
    featureNames: {json.dumps(feature_names)}
}};

if (typeof module !== 'undefined') {{
    module.exports = SUNRISE5_CONFIG;
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
        "model_type": "3-class (Human/AI/Humanized)",
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
    print("  🌅 SUNRISE V6.0 - THREE-CLASS DETECTION MODEL")
    print("═" * 70)
    print(f"  Classes: Human | AI | Humanized")
    print(f"  Target: {TARGET_SAMPLES:,} samples")
    print(f"  Features: 96")
    print(f"  Optimization trials: {OPTUNA_TRIALS}")
    print("═" * 70)
    
    # Check for cached features
    X, y = load_cached_features()
    
    if X is None:
        # STEP 1: Collect samples
        print("\n" + "═" * 70)
        print("  STEP 1: COLLECTING SAMPLES (3 CLASSES)")
        print("═" * 70)
        
        samples, labels = collect_all_samples()
        
        # Count distribution
        human_count = sum(1 for l in labels if l == CLASS_HUMAN)
        ai_count = sum(1 for l in labels if l == CLASS_AI)
        humanized_count = sum(1 for l in labels if l == CLASS_HUMANIZED)
        
        print(f"\n  ✓ Total samples: {len(samples):,}")
        print(f"    Human:     {human_count:,} ({100*human_count/len(samples):.1f}%)")
        print(f"    AI:        {ai_count:,} ({100*ai_count/len(samples):.1f}%)")
        print(f"    Humanized: {humanized_count:,} ({100*humanized_count/len(samples):.1f}%)")
        
        # STEP 2: Extract features
        print("\n" + "═" * 70)
        print("  STEP 2: EXTRACTING 96 FEATURES (PARALLEL)")
        print("═" * 70)
        
        X, y = extract_features_parallel(samples, labels)
        
        print(f"\n  ✓ Features extracted: {len(X):,} samples")
        
        # Save to cache
        save_features_to_cache(X, y)
        
        del samples
        gc.collect()
    
    # STEP 3: Prepare data
    print("\n" + "═" * 70)
    print("  STEP 3: PREPARING DATA")
    print("═" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Training samples: {len(X_train):,}")
    print(f"  Testing samples: {len(X_test):,}")
    
    # Class distribution
    for cls_id, cls_name in CLASS_NAMES.items():
        train_count = np.sum(y_train == cls_id)
        test_count = np.sum(y_test == cls_id)
        print(f"    {cls_name}: {train_count:,} train / {test_count:,} test")
    
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # STEP 4: Optimize
    print("\n" + "═" * 70)
    print(f"  STEP 4: HYPERPARAMETER OPTIMIZATION ({OPTUNA_TRIALS} trials)")
    print("═" * 70)
    
    best_params, study = optimize_hyperparameters(X_train_scaled, y_train, OPTUNA_TRIALS)
    
    # STEP 5: Train
    print("\n" + "═" * 70)
    print("  STEP 5: TRAINING FINAL MODEL")
    print("═" * 70)
    
    model = train_final_model(X_train_scaled, y_train, best_params)
    print("  ✓ Model trained")
    
    # STEP 6: Evaluate
    print("\n" + "═" * 70)
    print("  STEP 6: EVALUATION")
    print("═" * 70)
    
    metrics, cm, report = evaluate_model(model, X_test_scaled, y_test)
    
    print(f"\n  📊 RESULTS:")
    print(f"  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  Overall Accuracy:     {metrics['accuracy']*100:6.2f}%                ║")
    print(f"  ║  Macro Precision:      {metrics['precision_macro']*100:6.2f}%                ║")
    print(f"  ║  Macro Recall:         {metrics['recall_macro']*100:6.2f}%                ║")
    print(f"  ║  Macro F1:             {metrics['f1_macro']*100:6.2f}%                ║")
    print(f"  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Human F1:             {metrics.get('f1_human', 0)*100:6.2f}%                ║")
    print(f"  ║  AI F1:                {metrics.get('f1_ai', 0)*100:6.2f}%                ║")
    print(f"  ║  Humanized F1:         {metrics.get('f1_humanized', 0)*100:6.2f}%                ║")
    print(f"  ╚══════════════════════════════════════════════════╝")
    
    print(f"\n  Confusion Matrix:")
    print(f"                         Predicted")
    print(f"                   Human      AI    Humanized")
    print(f"    Actual Human   {cm[0][0]:6,}  {cm[0][1]:6,}  {cm[0][2]:6,}")
    print(f"    Actual AI      {cm[1][0]:6,}  {cm[1][1]:6,}  {cm[1][2]:6,}")
    print(f"    Actual Human.  {cm[2][0]:6,}  {cm[2][1]:6,}  {cm[2][2]:6,}")
    
    print(f"\n  Classification Report:")
    print(report)
    
    # STEP 7: Save
    print("\n" + "═" * 70)
    print("  STEP 7: SAVING ARTIFACTS")
    print("═" * 70)
    
    save_model_artifacts(model, scaler, metrics, best_params, study, cm)
    
    # Final summary
    print("\n" + "═" * 70)
    print("  🌅 SUNRISE V6.0 TRAINING COMPLETE!")
    print("═" * 70)
    print(f"  ✓ 3-Class Model: Human | AI | Humanized")
    print(f"  ✓ Samples: {len(X):,}")
    print(f"  ✓ Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  ✓ Model: {OUTPUT_DIR}/model.pkl")
    print("═" * 70)


if __name__ == "__main__":
    main()
