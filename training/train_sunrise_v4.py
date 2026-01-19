#!/usr/bin/env python3
"""
Veritas Sunrise V4 Training - Comprehensive Feature Training
=============================================================

Key Changes from V3:
- Uses 96 features (up from 37)
- Metadata (Category 11) EXCLUDED from ML training - treated as absolute flag
- Includes all advanced statistical tests, AI signatures, humanizer detection
- Enhanced crash recovery and progress tracking

Usage:
    python train_sunrise_v4.py                    # Full training
    python train_sunrise_v4.py --quick            # Quick test mode
    python train_sunrise_v4.py --no-resume        # Start fresh
"""

import os
import sys
import gc
import json
import time
import pickle
import hashlib
import signal
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: datasets library required. Install with: pip install datasets")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not available, hyperparameter optimization disabled")

# Import V4 feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v4 import FeatureExtractorV4
from curated_datasets import get_all_curated_datasets


BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ███████╗██╗   ██╗███╗   ██╗██████╗ ██╗███████╗███████╗                      ║
║   ██╔════╝██║   ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝                      ║
║   ███████╗██║   ██║██╔██╗ ██║██████╔╝██║███████╗█████╗                        ║
║   ╚════██║██║   ██║██║╚██╗██║██╔══██╗██║╚════██║██╔══╝                        ║
║   ███████║╚██████╔╝██║ ╚████║██║  ██║██║███████║███████╗                      ║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝                      ║
║                                                                               ║
║    V4.0 - Comprehensive Feature Training (96 features, Metadata excluded)     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""


class ProgressTracker:
    """Tracks progress and estimates time remaining."""
    
    def __init__(self, total_items: int, description: str = "Progress"):
        self.total = total_items
        self.completed = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, n: int = 1):
        self.completed += n
            
    def get_eta(self) -> str:
        if self.completed == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        avg_time = elapsed / self.completed
        remaining = (self.total - self.completed) * avg_time
        
        if remaining < 60:
            return f"{int(remaining)}s"
        elif remaining < 3600:
            return f"{int(remaining // 60)}m {int(remaining % 60)}s"
        else:
            hours = int(remaining // 3600)
            mins = int((remaining % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def get_elapsed(self) -> str:
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}m"


class CheckpointManager:
    """Manages training checkpoints for crash recovery."""
    
    def __init__(self, checkpoint_dir: Path, model_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.checkpoint_file = checkpoint_dir / f"{model_name}_v4_checkpoint.pkl"
        self.log_file = checkpoint_dir / f"{model_name}_v4_training.log"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, state: Dict):
        state['timestamp'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        self.log(f"Checkpoint saved: {state.get('datasets_completed', 0)} datasets completed")
        
    def load(self) -> Optional[Dict]:
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def clear(self):
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"  📝 {message}")


class SunriseV4Trainer:
    """
    Sunrise V4 trainer with comprehensive feature extraction.
    
    Key differences from V3:
    - Uses FeatureExtractorV4 with 96 features
    - Metadata features excluded (handled as absolute flag, not ML)
    - Includes AI signatures, humanizer detection, advanced stats
    """
    
    def __init__(self, output_dir: str = './models', model_name: str = 'SunriseV4',
                 batch_size: int = 3, tiers: List[int] = [1, 2, 3]):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize V4 feature extractor
        self.feature_extractor = FeatureExtractorV4()
        self.feature_names = self.feature_extractor.get_feature_names()
        
        print(f"  Feature Extractor V4 loaded: {len(self.feature_names)} features")
        
        # Checkpoint manager
        self.checkpoint_mgr = CheckpointManager(self.output_dir / '.checkpoints', model_name)
        
        # Build dataset list
        self.datasets = self._build_datasets_list(tiers)
        
        # Training state
        self.all_features = []
        self.all_labels = []
        self.datasets_used = []
        self.datasets_completed = 0
        self.total_human = 0
        self.total_ai = 0
        
        # Model artifacts
        self.model = None
        self.scaler = None
        self.X = None
        self.y = None
        self.results = None
        self.feature_importance = None
        self.sorted_features = None
        
        # Timing
        self.start_time = None
        
        # Graceful shutdown
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        print("\n\n  ⚠️  Shutdown requested - saving checkpoint...")
        self.shutdown_requested = True
        self._save_checkpoint()
        print("  ✓ Checkpoint saved. Resume with: python train_sunrise_v4.py")
        sys.exit(0)
        
    def _build_datasets_list(self, tiers: List[int]) -> List[Dict]:
        """Build datasets list from curated registry."""
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
                    'tier': ds['tier'],
                    'special': ds.get('special', ''),
                    'human_text_col': ds.get('human_text_col', ''),
                    'ai_text_col': ds.get('ai_text_col', ''),
                })
        
        return datasets
    
    def _save_checkpoint(self):
        state = {
            'datasets_completed': self.datasets_completed,
            'all_features': self.all_features,
            'all_labels': self.all_labels,
            'datasets_used': self.datasets_used,
            'total_human': self.total_human,
            'total_ai': self.total_ai,
        }
        self.checkpoint_mgr.save(state)
        
    def _restore_checkpoint(self) -> bool:
        state = self.checkpoint_mgr.load()
        if state:
            self.datasets_completed = state.get('datasets_completed', 0)
            self.all_features = state.get('all_features', [])
            self.all_labels = state.get('all_labels', [])
            self.datasets_used = state.get('datasets_used', [])
            self.total_human = state.get('total_human', 0)
            self.total_ai = state.get('total_ai', 0)
            
            self.checkpoint_mgr.log(f"Resumed from checkpoint at dataset {self.datasets_completed}")
            return True
        return False
    
    def process_single_dataset(self, config: Dict) -> Tuple[int, int]:
        """Process a single dataset and extract V4 features."""
        name = config['name']
        human_added = 0
        ai_added = 0
        special = config.get('special', None)
        
        try:
            # Load dataset
            ds = None
            configs_to_try = [None, 'default', 'all']
            splits_to_try = ['train', 'test', 'validation']
            
            for cfg in configs_to_try:
                if ds is not None:
                    break
                for split in splits_to_try:
                    try:
                        if cfg:
                            ds = load_dataset(name, cfg, split=split, streaming=True)
                        else:
                            ds = load_dataset(name, split=split, streaming=True)
                        break
                    except Exception:
                        continue
            
            if ds is None:
                print(f"    ⚠️  Could not load dataset {name}")
                return 0, 0
            
            # Process samples
            max_samples = config['max_samples']
            samples_processed = 0
            
            for item in ds:
                if samples_processed >= max_samples:
                    break
                    
                if self.shutdown_requested:
                    break
                
                try:
                    # Handle special dataset types
                    if special in ('paired', 'dual_column'):
                        # Paired datasets with human and AI columns
                        human_text = item.get(config['human_text_col'], '')
                        ai_text = item.get(config['ai_text_col'], '')
                        
                        if human_text and len(human_text) >= 100:
                            features = self.feature_extractor.extract_feature_vector(human_text)
                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                self.all_features.append(features)
                                self.all_labels.append(0)  # Human
                                human_added += 1
                        
                        if ai_text and len(ai_text) >= 100:
                            features = self.feature_extractor.extract_feature_vector(ai_text)
                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                self.all_features.append(features)
                                self.all_labels.append(1)  # AI
                                ai_added += 1
                        
                        samples_processed += 1
                        
                    elif special == 'human_only':
                        # Human-only datasets (100% human written)
                        text = item.get(config['text_col'], '')
                        
                        if text and len(text) >= 100:
                            features = self.feature_extractor.extract_feature_vector(text)
                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                self.all_features.append(features)
                                self.all_labels.append(0)  # Human
                                human_added += 1
                        
                        samples_processed += 1
                        
                    elif special == 'ai_only':
                        # AI-only datasets (100% AI generated)
                        text = item.get(config['text_col'], '')
                        
                        if text and len(text) >= 100:
                            features = self.feature_extractor.extract_feature_vector(text)
                            if not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                                self.all_features.append(features)
                                self.all_labels.append(1)  # AI
                                ai_added += 1
                        
                        samples_processed += 1
                        
                    else:
                        # Standard labeled dataset
                        text = item.get(config['text_col'], '')
                        label_raw = item.get(config['label_col'], None)
                        
                        if not text or len(text) < 100:
                            continue
                        
                        # Map label
                        label_map = config['label_map']
                        if label_raw in label_map:
                            label = label_map[label_raw]
                        elif str(label_raw) in label_map:
                            label = label_map[str(label_raw)]
                        else:
                            continue
                        
                        # Extract features
                        features = self.feature_extractor.extract_feature_vector(text)
                        
                        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                            continue
                        
                        self.all_features.append(features)
                        self.all_labels.append(label)
                        
                        if label == 0:
                            human_added += 1
                        else:
                            ai_added += 1
                        
                        samples_processed += 1
                        
                except Exception as e:
                    continue
            
            return human_added, ai_added
            
        except Exception as e:
            print(f"    ⚠️  Error processing {name}: {e}")
            return 0, 0
    
    def run_training(self, n_trials: int = 50, resume: bool = True):
        """Run the complete training pipeline."""
        print(BANNER)
        self.start_time = time.time()
        
        print(f"{'═'*70}")
        print(f"  SUNRISE V4 TRAINING - COMPREHENSIVE FEATURES")
        print(f"{'═'*70}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Datasets: {len(self.datasets)}")
        print(f"  Note: Metadata (Category 11) EXCLUDED - handled as absolute flag")
        print()
        
        # ===== STEP 1: CHECK FOR RESUME =====
        if resume and self._restore_checkpoint():
            print(f"  ✓ Resumed from checkpoint ({self.datasets_completed}/{len(self.datasets)} datasets)")
        else:
            print(f"  Starting fresh training...")
        
        # ===== STEP 2: PROCESS DATASETS =====
        print(f"\n{'═'*70}")
        print(f"  STEP 1: DATA COLLECTION")
        print(f"{'═'*70}")
        
        remaining_datasets = self.datasets[self.datasets_completed:]
        
        for i, config in enumerate(remaining_datasets):
            if self.shutdown_requested:
                break
                
            ds_num = self.datasets_completed + 1
            print(f"\n  [{ds_num}/{len(self.datasets)}] {config['name']}")
            print(f"      Tier: {config['tier']} | Max samples: {config['max_samples']}")
            
            human_added, ai_added = self.process_single_dataset(config)
            
            if human_added > 0 or ai_added > 0:
                self.datasets_used.append({
                    'name': config['name'],
                    'human': human_added,
                    'ai': ai_added,
                    'tier': config['tier']
                })
                self.total_human += human_added
                self.total_ai += ai_added
                print(f"      ✓ Added: {human_added} human, {ai_added} AI")
            else:
                print(f"      ⚠️  No samples extracted")
            
            self.datasets_completed += 1
            
            # Checkpoint every batch_size datasets
            if self.datasets_completed % self.batch_size == 0:
                self._save_checkpoint()
                gc.collect()
        
        # Check minimum samples
        if len(self.all_features) < 100:
            print(f"\n  ❌ Insufficient data: {len(self.all_features)} samples")
            print(f"     Need at least 100 samples for training.")
            return
        
        print(f"\n  {'─'*50}")
        print(f"  DATA COLLECTION COMPLETE")
        print(f"  {'─'*50}")
        print(f"  Total samples: {len(self.all_features):,}")
        print(f"  Human samples: {self.total_human:,}")
        print(f"  AI samples: {self.total_ai:,}")
        print(f"  Balance ratio: {self.total_human / max(1, self.total_ai):.2f}")
        
        # ===== STEP 3: PREPARE DATA =====
        print(f"\n{'═'*70}")
        print(f"  STEP 2: DATA PREPARATION")
        print(f"{'═'*70}")
        
        self.X = np.array(self.all_features)
        self.y = np.array(self.all_labels)
        
        # Replace any remaining NaN/Inf
        self.X = np.nan_to_num(self.X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"  Feature matrix shape: {self.X.shape}")
        print(f"  Labels shape: {self.y.shape}")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"  Train set: {len(X_train):,} samples")
        print(f"  Test set: {len(X_test):,} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"  ✓ Features scaled")
        
        # ===== STEP 4: HYPERPARAMETER OPTIMIZATION =====
        print(f"\n{'═'*70}")
        print(f"  STEP 3: MODEL OPTIMIZATION")
        print(f"{'═'*70}")
        
        best_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
        }
        
        if OPTUNA_AVAILABLE and n_trials > 0:
            print(f"  Running {n_trials} optimization trials...")
            
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 400),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                }
                
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='f1')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            
            best_params = study.best_params
            print(f"\n  ✓ Best F1 score: {study.best_value:.4f}")
            print(f"  ✓ Best params: {best_params}")
        else:
            print(f"  Using default hyperparameters (optuna disabled)")
        
        # ===== STEP 5: TRAIN FINAL MODEL =====
        print(f"\n{'═'*70}")
        print(f"  STEP 4: FINAL TRAINING")
        print(f"{'═'*70}")
        
        print(f"  Training Random Forest with {best_params['n_estimators']} trees...")
        
        self.model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
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
            'hyperparameters': best_params,
        }
        
        # Feature importance
        self.feature_importance = dict(zip(
            self.feature_names,
            [float(x) for x in self.model.feature_importances_]
        ))
        self.sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: -x[1]
        )
        
        print(f"\n  {'─'*50}")
        print(f"  EVALUATION RESULTS")
        print(f"  {'─'*50}")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1 Score:  {self.results['f1']:.4f}")
        print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")
        
        # ===== STEP 6: SAVE MODEL =====
        print(f"\n{'═'*70}")
        print(f"  STEP 5: SAVING MODEL")
        print(f"{'═'*70}")
        
        self._save_model()
        
        # Clear checkpoint
        self.checkpoint_mgr.clear()
        
        # Summary
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f"\n{'═'*70}")
        print(f"  🎉 SUNRISE V4 TRAINING COMPLETE!")
        print(f"{'═'*70}")
        print(f"  Total time: {elapsed_str}")
        print(f"  Datasets processed: {len(self.datasets_used)}")
        print(f"  Total samples: {len(self.all_features):,}")
        print(f"  Model saved to: {self.output_dir / self.model_name}")
        
        print(f"\n  Top 15 Features:")
        for i, (name, imp) in enumerate(self.sorted_features[:15], 1):
            pct = imp * 100
            bar = '█' * int(pct * 2)
            print(f"    {i:2}. {name:40} {pct:5.2f}% {bar}")
        
        print(f"\n  NOTE: Metadata features are NOT included in ML training.")
        print(f"        They are handled as an absolute detection flag.")
        
        self.checkpoint_mgr.log(f"Training completed successfully. F1={self.results['f1']:.4f}")
    
    def _save_model(self):
        """Save model and all artifacts."""
        model_dir = self.output_dir / self.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  ✓ Saved model.pkl")
        
        # Save scaler
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Saved scaler.pkl")
        
        # Create training receipt
        receipt = {
            'model_name': self.model_name,
            'version': '4.0.0',
            'timestamp': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'feature_names': self.feature_names,
            'datasets_used': self.datasets_used,
            'total_samples': len(self.y),
            'human_samples': int(sum(self.y == 0)),
            'ai_samples': int(sum(self.y == 1)),
            'results': self.results,
            'feature_importance': self.feature_importance,
            'top_features': self.sorted_features[:30],
            'feature_weights_normalized': {
                name: round(imp / sum(self.feature_importance.values()) * 100, 2)
                for name, imp in self.sorted_features
            },
            'training_time_seconds': time.time() - self.start_time,
            'data_hash': hashlib.sha256(self.X.tobytes()).hexdigest()[:16],
            'model_hash': hashlib.sha256(pickle.dumps(self.model)).hexdigest()[:16],
            'metadata_note': 'Metadata (Category 11) excluded from ML - treated as absolute detection flag',
        }
        
        # Save JSON receipt
        with open(model_dir / 'training_receipt.json', 'w') as f:
            json.dump(receipt, f, indent=2)
        print(f"  ✓ Saved training_receipt.json")
        
        # Save JS config
        self._save_js_config(model_dir, receipt)
        print(f"  ✓ Saved veritas_config.js")
        
        # Save metadata
        metadata = {
            'name': self.model_name,
            'version': '4.0.0',
            'timestamp': receipt['timestamp'],
            'accuracy': self.results['accuracy'],
            'f1': self.results['f1'],
            'roc_auc': self.results['roc_auc'],
            'features': len(self.feature_names),
            'datasets': len(self.datasets_used),
            'samples': receipt['total_samples'],
            'note': 'Metadata excluded from ML training - absolute flag',
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata.json")
    
    def _save_js_config(self, model_dir: Path, receipt: Dict):
        """Generate JavaScript configuration file for Sunrise V4."""
        weights = receipt['feature_weights_normalized']
        
        # Group features by category for documentation
        feature_categories = {
            'sentence': [f for f in self.feature_names if 'sentence' in f.lower()],
            'vocabulary': [f for f in self.feature_names if any(x in f.lower() for x in ['word', 'hapax', 'ttr', 'yule', 'simpson', 'honore', 'brunet', 'sichel', 'vocab'])],
            'zipf': [f for f in self.feature_names if 'zipf' in f.lower()],
            'burstiness': [f for f in self.feature_names if any(x in f.lower() for x in ['burst', 'uniform', 'variance', 'autocor'])],
            'readability': [f for f in self.feature_names if any(x in f.lower() for x in ['flesch', 'gunning', 'coleman', 'smog', 'ari', 'syllable', 'complex'])],
            'ngram': [f for f in self.feature_names if any(x in f.lower() for x in ['gram', 'phrase', 'similar'])],
            'ai_signature': [f for f in self.feature_names if any(x in f.lower() for x in ['hedging', 'discourse', 'contraction', 'starter', 'transition', 'passive', 'nominal'])],
            'humanizer': [f for f in self.feature_names if any(x in f.lower() for x in ['humanizer', 'artificial', 'correlation_strength'])],
        }
        
        # Calculate category weights
        category_weights = {}
        for cat, features in feature_categories.items():
            cat_weight = sum(weights.get(f, 0) for f in features)
            category_weights[cat] = round(cat_weight, 2)
        
        js_content = f"""/**
 * Veritas Sunrise V4 ML Configuration
 * Generated: {receipt['timestamp']}
 * Version: 4.0.0
 * 
 * COMPREHENSIVE FEATURE TRAINING
 * - {len(self.feature_names)} features (up from 37 in V3)
 * - Metadata (Category 11) EXCLUDED - handled as absolute detection flag
 * - Includes: AI signatures, humanizer detection, advanced statistical tests
 * 
 * Training Statistics:
 * - Datasets: {len(receipt['datasets_used'])}
 * - Total Samples: {receipt['total_samples']:,}
 * - Test Accuracy: {receipt['results']['accuracy']:.4f}
 * - Test F1 Score: {receipt['results']['f1']:.4f}
 * - ROC AUC: {receipt['results']['roc_auc']:.4f}
 * - Training Time: {int(receipt['training_time_seconds'])}s
 */

const VERITAS_SUNRISE_CONFIG = {{
    version: '4.0.0',
    modelName: 'Sunrise',
    modelGeneration: 'V4',
    
    // Total features used in ML model (Metadata excluded)
    featureCount: {len(self.feature_names)},
    
    // ML-derived feature weights (percentages, sum to 100)
    // Top 50 features shown for readability
    featureWeights: {{
{chr(10).join(f"        {name.replace('-', '_')}: {weight}," for name, weight in sorted(weights.items(), key=lambda x: -x[1])[:50])}
    }},
    
    // Aggregated category weights (derived from individual features)
    categoryWeights: {{
        sentence: {category_weights.get('sentence', 0)},      // Sentence structure analysis
        vocabulary: {category_weights.get('vocabulary', 0)},    // Vocabulary richness metrics
        zipf: {category_weights.get('zipf', 0)},          // Zipf's law compliance
        burstiness: {category_weights.get('burstiness', 0)},    // Variance and uniformity
        readability: {category_weights.get('readability', 0)},   // Readability indices
        ngram: {category_weights.get('ngram', 0)},         // N-gram repetition patterns
        aiSignature: {category_weights.get('ai_signature', 0)},  // AI-specific patterns
        humanizer: {category_weights.get('humanizer', 0)},     // Humanizer detection signals
        // Note: Metadata NOT included - it's an absolute flag, not ML-weighted
    }},
    
    // Primary detection features (top 15 by importance)
    primaryFeatures: {json.dumps([f[0] for f in receipt['top_features'][:15]])},
    
    // Scoring thresholds
    scoring: {{
        aiThreshold: 0.60,
        humanThreshold: 0.40,
        highConfidenceThreshold: 0.85,
    }},
    
    // Metadata Detection (Absolute Flag - NOT in ML model)
    metadataDetection: {{
        enabled: true,
        isAbsoluteFlag: true,
        flagMessage: "Unknown metadata was detected within the text. This may likely be an artifact of LLM usage.",
        weight: 0,  // Not weighted in ML - just adds flag
    }},
    
    // Training verification
    trainingStats: {{
        datasetsUsed: {len(receipt['datasets_used'])},
        totalSamples: {receipt['total_samples']},
        testAccuracy: {receipt['results']['accuracy']:.4f},
        testF1: {receipt['results']['f1']:.4f},
        testPrecision: {receipt['results']['precision']:.4f},
        testRecall: {receipt['results']['recall']:.4f},
        rocAuc: {receipt['results']['roc_auc']:.4f},
        dataHash: '{receipt['data_hash']}',
        modelHash: '{receipt['model_hash']}',
    }},
    
    // Feature categories for reference
    featureCategories: {{
        sentence: {len(feature_categories.get('sentence', []))},
        vocabulary: {len(feature_categories.get('vocabulary', []))},
        zipf: {len(feature_categories.get('zipf', []))},
        burstiness: {len(feature_categories.get('burstiness', []))},
        readability: {len(feature_categories.get('readability', []))},
        ngram: {len(feature_categories.get('ngram', []))},
        aiSignature: {len(feature_categories.get('ai_signature', []))},
        humanizer: {len(feature_categories.get('humanizer', []))},
    }},
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_SUNRISE_CONFIG;
}}
"""
        
        with open(model_dir / 'veritas_config.js', 'w') as f:
            f.write(js_content)


def main():
    parser = argparse.ArgumentParser(
        description='Veritas Sunrise V4 Training - Comprehensive Features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_sunrise_v4.py                    # Full training
  python train_sunrise_v4.py --quick            # Quick test mode (Tier 1 only)
  python train_sunrise_v4.py --no-resume        # Start fresh, ignore checkpoints
  python train_sunrise_v4.py --trials 100       # More optimization trials
        """
    )
    parser.add_argument('--name', type=str, default='SunriseV4', help='Model name')
    parser.add_argument('--trials', type=int, default=50, help='Optuna optimization trials')
    parser.add_argument('--batch-size', type=int, default=3, help='Datasets per batch checkpoint')
    parser.add_argument('--tiers', type=str, default='1,2,3', help='Dataset tiers (1-4)')
    parser.add_argument('--quick', action='store_true', help='Quick mode: Tier 1 only, 20 trials')
    parser.add_argument('--no-resume', action='store_true', help='Start fresh, ignore checkpoints')
    parser.add_argument('--output', type=str, default='./models', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse tiers
    if args.quick:
        tiers = [1]
        args.trials = 20
        args.batch_size = 2
    else:
        tiers = [int(t.strip()) for t in args.tiers.split(',')]
    
    # Create trainer
    trainer = SunriseV4Trainer(
        output_dir=args.output,
        model_name=args.name,
        batch_size=args.batch_size,
        tiers=tiers
    )
    
    # Run training
    trainer.run_training(
        n_trials=args.trials,
        resume=not args.no_resume
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
