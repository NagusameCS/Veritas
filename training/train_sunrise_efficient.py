#!/usr/bin/env python3
"""
Veritas Sunrise Training - Memory Efficient Version
Processes ONE dataset at a time to avoid memory/storage issues.
Extracts features incrementally and discards raw text immediately.
"""

import os
import sys
import gc
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Memory-efficient imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import pickle

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: datasets library required")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Import feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor

# ============================================================================
# CURATED DATASET CONFIGURATIONS
# Hand-selected based on: downloads, size, language, label quality, reputation
# Processed ONE AT A TIME to avoid memory issues
# ============================================================================

# Import from curated registry
from curated_datasets import get_all_curated_datasets, TIER_1_EXCELLENT, TIER_2_VERY_GOOD, TIER_3_GOOD, TIER_4_SPECIALIZED

# Convert curated datasets to the format needed by the trainer
def _build_datasets_list(tiers=[1, 2, 3]):
    """Build datasets list from curated registry, optionally filtering by tier."""
    datasets = []
    all_curated = get_all_curated_datasets()
    
    for ds in all_curated:
        if ds['tier'] in tiers:
            datasets.append({
                'name': ds['name'],
                'text_col': ds['text_col'],
                'label_col': ds['label_col'],
                'label_map': ds['label_map'],
                'max_samples': ds['max_samples'],
                'description': f"[TIER {ds['tier']}] {ds.get('why_excellent', ds.get('why_good', ds.get('why_useful', ds.get('why_specialized', ''))))[:50]}",
                'url': ds['url'],
                'tier': ds['tier'],
            })
    
    return datasets

# Default: Use Tier 1, 2, and 3 (skip specialized for faster training)
DATASETS = _build_datasets_list(tiers=[1, 2, 3])

BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║   ███████╗██╗   ██╗███╗   ██╗██████╗ ██╗███████╗███████╗     ║
║   ██╔════╝██║   ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝     ║
║   ███████╗██║   ██║██╔██╗ ██║██████╔╝██║███████╗█████╗       ║
║   ╚════██║██║   ██║██║╚██╗██║██╔══██╗██║╚════██║██╔══╝       ║
║   ███████║╚██████╔╝██║ ╚████║██║  ██║██║███████║███████╗     ║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝     ║
║                                                               ║
║        Memory-Efficient Sunrise Training v2.1                 ║
║        Processes ONE dataset at a time                        ║
╚═══════════════════════════════════════════════════════════════╝
"""

class MemoryEfficientTrainer:
    """Trains on datasets one at a time, immediately extracting features."""
    
    FEATURE_NAMES = [
        'sentence_count', 'avg_sentence_length', 'sentence_length_cv',
        'sentence_length_std', 'sentence_length_min', 'sentence_length_max',
        'sentence_length_range', 'sentence_length_skewness', 'sentence_length_kurtosis',
        'word_count', 'unique_word_count', 'type_token_ratio',
        'hapax_count', 'hapax_ratio', 'dis_legomena_ratio',
        'zipf_slope', 'zipf_r_squared', 'zipf_residual_std',
        'burstiness_sentence', 'burstiness_word_length',
        'avg_word_length', 'word_length_cv', 'syllable_ratio',
        'flesch_kincaid_grade', 'automated_readability_index',
        'bigram_repetition_rate', 'trigram_repetition_rate',
        'sentence_similarity_avg', 'comma_rate', 'semicolon_rate',
        'question_rate', 'exclamation_rate',
        'paragraph_count', 'avg_paragraph_length', 'paragraph_length_cv',
        'overall_uniformity', 'complexity_cv'
    ]
    
    def __init__(self, output_dir: str = './models'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_extractor = FeatureExtractor()
        self.all_features = []  # List of feature vectors
        self.all_labels = []    # List of labels
        self.datasets_used = []
        self.scaler = StandardScaler()
        self.model = None
        
    def process_single_dataset(self, config: Dict) -> Tuple[int, int]:
        """
        Load ONE dataset, extract features, then DELETE the raw data.
        Returns (human_count, ai_count) added.
        """
        name = config['name']
        print(f"\n  Loading: {name}...")
        
        human_added = 0
        ai_added = 0
        
        try:
            # Load dataset
            ds = None
            for split in ['train', 'test', 'validation']:
                try:
                    ds = load_dataset(name, split=split, streaming=True)
                    break
                except:
                    continue
            
            if ds is None:
                try:
                    ds = load_dataset(name, streaming=True)
                    if hasattr(ds, 'keys'):
                        ds = ds[list(ds.keys())[0]]
                except Exception as e:
                    print(f"    ✗ Failed to load: {e}")
                    return 0, 0
            
            # Process samples with streaming (memory efficient)
            text_col = config['text_col']
            label_col = config['label_col']
            label_map = config['label_map']
            max_samples = config['max_samples']
            
            count = 0
            for row in tqdm(ds, desc=f"    Processing", total=max_samples, leave=False):
                if count >= max_samples:
                    break
                
                try:
                    text = str(row.get(text_col, ''))
                    label_raw = row.get(label_col)
                    
                    if not text or len(text) < 100:
                        continue
                    
                    # Map label
                    label = label_map.get(label_raw, label_raw)
                    if isinstance(label, str):
                        label_lower = label.lower()
                        if label_lower in ['human', 'real', '0']:
                            label = 0
                        elif label_lower in ['ai', 'machine', 'generated', 'gpt', '1']:
                            label = 1
                        else:
                            continue
                    
                    if label not in [0, 1]:
                        continue
                    
                    # Extract features IMMEDIATELY
                    features = self.feature_extractor.extract_feature_vector(text)
                    if features is not None and len(features) == 37:
                        self.all_features.append(features)
                        self.all_labels.append(label)
                        if label == 0:
                            human_added += 1
                        else:
                            ai_added += 1
                        count += 1
                        
                except Exception:
                    continue
            
            # Force garbage collection to free memory
            del ds
            gc.collect()
            
            if human_added + ai_added > 0:
                self.datasets_used.append({
                    'name': name,
                    'description': config.get('description', ''),
                    'human_samples': human_added,
                    'ai_samples': ai_added,
                    'url': f'https://huggingface.co/datasets/{name}'
                })
                print(f"    ✓ Added {human_added} human + {ai_added} AI samples")
            else:
                print(f"    ✗ No valid samples found")
                
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:50]}")
            gc.collect()
            
        return human_added, ai_added
    
    def load_all_datasets(self):
        """Process all datasets one at a time."""
        print(f"\n{'='*60}")
        print(f"STEP 1: LOADING DATASETS (one at a time)")
        print(f"{'='*60}")
        print(f"  Datasets to process: {len(DATASETS)}")
        
        total_human = 0
        total_ai = 0
        
        for config in DATASETS:
            h, a = self.process_single_dataset(config)
            total_human += h
            total_ai += a
            
            # Clear HuggingFace cache after each dataset
            cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
            if os.path.exists(cache_dir):
                try:
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except:
                    pass
            gc.collect()
        
        print(f"\n  Total loaded: {total_human} human + {total_ai} AI = {total_human + total_ai}")
        return total_human, total_ai
    
    def balance_data(self):
        """Balance human and AI samples."""
        print(f"\n{'='*60}")
        print("STEP 2: BALANCING DATA")
        print(f"{'='*60}")
        
        X = np.array(self.all_features)
        y = np.array(self.all_labels)
        
        human_idx = np.where(y == 0)[0]
        ai_idx = np.where(y == 1)[0]
        
        min_count = min(len(human_idx), len(ai_idx))
        
        np.random.seed(42)
        selected_human = np.random.choice(human_idx, min_count, replace=False)
        selected_ai = np.random.choice(ai_idx, min_count, replace=False)
        
        selected = np.concatenate([selected_human, selected_ai])
        np.random.shuffle(selected)
        
        self.X = X[selected]
        self.y = y[selected]
        
        # Clear old data
        self.all_features = []
        self.all_labels = []
        gc.collect()
        
        print(f"  Balanced: {min_count} human + {min_count} AI = {len(self.y)} total")
    
    def train(self, n_trials: int = 50):
        """Train with hyperparameter optimization."""
        print(f"\n{'='*60}")
        print("STEP 3: TRAINING MODEL")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Optimize hyperparameters
        best_params = {}
        if OPTUNA_AVAILABLE and n_trials > 0:
            print(f"  Optimizing hyperparameters ({n_trials} trials)...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            best_params = study.best_params
            print(f"  Best F1: {study.best_value:.4f}")
        
        # Train final model
        print("  Training final model...")
        final_params = {
            'n_estimators': best_params.get('n_estimators', 200),
            'max_depth': best_params.get('max_depth', 15),
            'min_samples_split': best_params.get('min_samples_split', 5),
            'min_samples_leaf': best_params.get('min_samples_leaf', 1),
        }
        self.model = RandomForestClassifier(**final_params, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.results = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'hyperparameters': final_params,
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.FEATURE_NAMES,
            [float(x) for x in self.model.feature_importances_]
        ))
        self.sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: -x[1]
        )
        
        print(f"\n  {'='*40}")
        print(f"  RESULTS")
        print(f"  {'='*40}")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1 Score:  {self.results['f1']:.4f}")
        print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")
    
    def save(self, model_name: str = 'Sunrise'):
        """Save all outputs."""
        print(f"\n{'='*60}")
        print("STEP 4: SAVING MODEL & TRAINING RECEIPT")
        print(f"{'='*60}")
        
        model_dir = self.output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Create and save training receipt
        receipt = {
            'model_name': model_name,
            'version': '2.1.0-sunrise',
            'timestamp': datetime.now().isoformat(),
            'datasets_used': self.datasets_used,
            'total_samples': len(self.y),
            'human_samples': int(sum(self.y == 0)),
            'ai_samples': int(sum(self.y == 1)),
            'results': self.results,
            'feature_importance': self.feature_importance,
            'top_features': self.sorted_features[:20],
            'feature_weights_normalized': {
                name: round(imp / sum(self.feature_importance.values()) * 100, 2)
                for name, imp in self.sorted_features
            },
            'data_hash': hashlib.sha256(self.X.tobytes()).hexdigest()[:16],
            'model_hash': hashlib.sha256(pickle.dumps(self.model)).hexdigest()[:16],
        }
        
        with open(model_dir / 'training_receipt.json', 'w') as f:
            json.dump(receipt, f, indent=2)
        
        # Save JS config
        self._save_js_config(model_dir, model_name, receipt)
        
        # Save JSON config for web
        with open(model_dir / 'veritas_ml_config.json', 'w') as f:
            json.dump(receipt, f, indent=2)
        
        print(f"  ✓ Model saved to: {model_dir}")
        print(f"  ✓ Training receipt: {model_dir / 'training_receipt.json'}")
        print(f"  ✓ JS config: {model_dir / 'veritas_ml_config.js'}")
        
        return receipt
    
    def _save_js_config(self, model_dir: Path, model_name: str, receipt: Dict):
        """Save JavaScript config."""
        weights = receipt['feature_weights_normalized']
        
        js_content = f"""/**
 * Veritas {model_name} ML Configuration
 * Generated: {receipt['timestamp']}
 * 
 * Training Statistics:
 * - Datasets: {len(receipt['datasets_used'])}
 * - Total Samples: {receipt['total_samples']:,}
 * - Test Accuracy: {receipt['results']['accuracy']:.4f}
 * - Test F1 Score: {receipt['results']['f1']:.4f}
 * - ROC AUC: {receipt['results']['roc_auc']:.4f}
 * 
 * Dataset Sources:
{chr(10).join(f" * - {d['name']}: {d['url']}" for d in receipt['datasets_used'])}
 * 
 * Training Receipt: See training_receipt.json for full verification
 */

const VERITAS_{model_name.upper()}_CONFIG = {{
    version: '{receipt['version']}',
    modelName: '{model_name}',
    
    // ML-derived feature weights (percentages, sum to 100)
    featureWeights: {{
{chr(10).join(f"        {name.replace('-', '_')}: {weight}," for name, weight in sorted(weights.items(), key=lambda x: -x[1]))}
    }},
    
    // Primary detection features (top 10 by importance)
    primaryFeatures: {json.dumps([f[0] for f in receipt['top_features'][:10]])},
    
    // Scoring thresholds
    scoring: {{
        aiThreshold: 0.65,
        humanThreshold: 0.35,
        highConfidenceThreshold: 0.80,
    }},
    
    // Training verification
    trainingStats: {{
        datasetsUsed: {len(receipt['datasets_used'])},
        totalSamples: {receipt['total_samples']},
        testAccuracy: {receipt['results']['accuracy']:.4f},
        testF1: {receipt['results']['f1']:.4f},
        rocAuc: {receipt['results']['roc_auc']:.4f},
        dataHash: '{receipt['data_hash']}',
        modelHash: '{receipt['model_hash']}',
    }},
    
    // Dataset sources for transparency
    datasetSources: {json.dumps([d['url'] for d in receipt['datasets_used']], indent=8)},
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_{model_name.upper()}_CONFIG;
}}
"""
        
        with open(model_dir / 'veritas_ml_config.js', 'w') as f:
            f.write(js_content)


def main():
    print(BANNER)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--name', type=str, default='Sunrise', help='Model name')
    parser.add_argument('--tiers', type=str, default='1,2,3', 
                        help='Dataset tiers to use (1=excellent, 2=very good, 3=good, 4=specialized)')
    parser.add_argument('--quick', action='store_true', help='Quick mode: tier 1 only, 20 trials')
    args = parser.parse_args()
    
    # Parse tiers
    if args.quick:
        tiers = [1]
        args.trials = 20
    else:
        tiers = [int(t.strip()) for t in args.tiers.split(',')]
    
    # Rebuild datasets list with selected tiers
    global DATASETS
    DATASETS = _build_datasets_list(tiers=tiers)
    
    print(f"\nConfiguration:")
    print(f"  Model name: {args.name}")
    print(f"  Optimization trials: {args.trials}")
    print(f"  Dataset tiers: {tiers}")
    print(f"  Datasets to process: {len(DATASETS)}")
    
    # Show datasets
    print(f"\n  Selected datasets:")
    for ds in DATASETS:
        print(f"    [T{ds['tier']}] {ds['name']}")
    
    trainer = MemoryEfficientTrainer()
    
    # Process datasets one at a time
    trainer.load_all_datasets()
    
    # Balance
    trainer.balance_data()
    
    # Train
    trainer.train(n_trials=args.trials)
    
    # Save
    receipt = trainer.save(args.name)
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nTop 10 Features:")
    for i, (name, imp) in enumerate(receipt['top_features'][:10], 1):
        bar = '█' * int(imp * 50)
        print(f"  {i:2}. {name:30} {imp:.4f} {bar}")
    
    print(f"\nDatasets used ({len(receipt['datasets_used'])}):")
    for d in receipt['datasets_used']:
        print(f"  - {d['name']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
