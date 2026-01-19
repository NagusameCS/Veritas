#!/usr/bin/env python3
"""
Veritas Sunrise Training - Robust Batch Training with Crash Recovery
=====================================================================

Features:
- Processes datasets in configurable batches
- Saves checkpoint after each batch for crash recovery
- Shows progress bar with time estimates
- Automatic resume from last checkpoint
- Detailed logging
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
from sklearn.ensemble import RandomForestClassifier
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

# Import feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor import FeatureExtractor
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
║    Robust Batch Training v3.0 - With Crash Recovery & Time Estimates          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

class ProgressTracker:
    """Tracks progress and estimates time remaining."""
    
    def __init__(self, total_items: int, description: str = "Progress"):
        self.total = total_items
        self.completed = 0
        self.description = description
        self.start_time = time.time()
        self.item_times = []
        
    def update(self, item_time: float = None):
        """Update progress after completing an item."""
        self.completed += 1
        if item_time:
            self.item_times.append(item_time)
            
    def get_eta(self) -> str:
        """Get estimated time remaining."""
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
        """Get elapsed time."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            hours = int(elapsed // 3600)
            mins = int((elapsed % 3600) // 60)
            return f"{hours}h {mins}m"
    
    def display(self):
        """Display progress bar with ETA."""
        pct = (self.completed / self.total) * 100 if self.total > 0 else 0
        bar_width = 40
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0
        bar = '█' * filled + '░' * (bar_width - filled)
        
        print(f"\r  {self.description}: [{bar}] {pct:5.1f}% ({self.completed}/{self.total}) | "
              f"Elapsed: {self.get_elapsed()} | ETA: {self.get_eta()}  ", end='', flush=True)


class CheckpointManager:
    """Manages training checkpoints for crash recovery."""
    
    def __init__(self, checkpoint_dir: Path, model_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        self.checkpoint_file = checkpoint_dir / f"{model_name}_checkpoint.pkl"
        self.log_file = checkpoint_dir / f"{model_name}_training.log"
        
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, state: Dict):
        """Save checkpoint state."""
        state['timestamp'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        self.log(f"Checkpoint saved: {state.get('datasets_completed', 0)} datasets completed")
        
    def load(self) -> Optional[Dict]:
        """Load checkpoint if exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def clear(self):
        """Clear checkpoint after successful completion."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            
    def log(self, message: str):
        """Append to training log."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"  📝 {message}")


class RobustBatchTrainer:
    """Robust trainer with batch processing and crash recovery."""
    
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
    
    def __init__(self, output_dir: str = './models', model_name: str = 'Sunrise',
                 batch_size: int = 3, tiers: List[int] = [1, 2, 3]):
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.batch_size = batch_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
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
        
        # Timing
        self.start_time = None
        self.dataset_times = []
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown on interrupt."""
        print("\n\n  ⚠️  Shutdown requested - saving checkpoint...")
        self.shutdown_requested = True
        self._save_checkpoint()
        print("  ✓ Checkpoint saved. You can resume training later.")
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
                    'url': ds.get('url', f"https://huggingface.co/datasets/{ds['name']}"),
                    # Important: copy special dataset type info
                    'special': ds.get('special', ''),
                    'human_text_col': ds.get('human_text_col', ''),
                    'ai_text_col': ds.get('ai_text_col', ''),
                })
        
        return datasets
    
    def _save_checkpoint(self):
        """Save current training state."""
        state = {
            'datasets_completed': self.datasets_completed,
            'all_features': self.all_features,
            'all_labels': self.all_labels,
            'datasets_used': self.datasets_used,
            'total_human': self.total_human,
            'total_ai': self.total_ai,
            'dataset_times': self.dataset_times,
        }
        self.checkpoint_mgr.save(state)
        
    def _restore_checkpoint(self) -> bool:
        """Restore from checkpoint if available."""
        state = self.checkpoint_mgr.load()
        if state:
            self.datasets_completed = state.get('datasets_completed', 0)
            self.all_features = state.get('all_features', [])
            self.all_labels = state.get('all_labels', [])
            self.datasets_used = state.get('datasets_used', [])
            self.total_human = state.get('total_human', 0)
            self.total_ai = state.get('total_ai', 0)
            self.dataset_times = state.get('dataset_times', [])
            
            self.checkpoint_mgr.log(f"Resumed from checkpoint at dataset {self.datasets_completed}")
            return True
        return False
    
    def process_single_dataset(self, config: Dict) -> Tuple[int, int]:
        """Process a single dataset and extract features."""
        name = config['name']
        human_added = 0
        ai_added = 0
        special = config.get('special', None)
        
        try:
            # Load dataset with streaming - try different configs
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
                try:
                    ds = load_dataset(name, streaming=True)
                    if hasattr(ds, 'keys'):
                        ds = ds[list(ds.keys())[0]]
                except Exception as e:
                    self.checkpoint_mgr.log(f"Failed to load {name}: {str(e)[:50]}")
                    return 0, 0
            
            # Process samples based on special type
            max_samples = config['max_samples']
            count = 0
            errors = 0
            
            for row in ds:
                if count >= max_samples or self.shutdown_requested:
                    break
                if errors > 200 and count == 0:
                    # Too many errors, skip this dataset
                    break
                
                try:
                    texts_to_process = []  # List of (text, label) tuples
                    
                    if special == 'dual_column':
                        # Dataset has both human and AI text columns
                        human_text = str(row.get(config.get('human_text_col', ''), ''))
                        ai_text = str(row.get(config.get('ai_text_col', ''), ''))
                        if human_text and len(human_text) >= 100:
                            texts_to_process.append((human_text, 0))
                        if ai_text and len(ai_text) >= 100:
                            texts_to_process.append((ai_text, 1))
                    elif special == 'human_only':
                        # Pure human text dataset
                        text = str(row.get(config['text_col'], ''))
                        if text and len(text) >= 100:
                            texts_to_process.append((text, 0))
                    elif special == 'ai_only':
                        # Pure AI text dataset
                        text = str(row.get(config['text_col'], ''))
                        if text and len(text) >= 100:
                            texts_to_process.append((text, 1))
                    else:
                        # Standard labeled dataset
                        text_col = config['text_col']
                        label_col = config['label_col']
                        label_map = config['label_map']
                        
                        text_raw = row.get(text_col, '')
                        if isinstance(text_raw, list):
                            text = ' '.join(str(t) for t in text_raw[:3]) if text_raw else ''
                        else:
                            text = str(text_raw)
                        
                        label_raw = row.get(label_col)
                        
                        if text and len(text) >= 100:
                            label = label_map.get(label_raw, label_raw)
                            
                            if isinstance(label, str):
                                label_lower = label.lower()
                                if label_lower in ['human', 'real', '0']:
                                    label = 0
                                elif label_lower in ['ai', 'machine', 'generated', 'gpt', '1']:
                                    label = 1
                                else:
                                    continue
                            
                            if label in [0, 1]:
                                texts_to_process.append((text, label))
                    
                    # Process all texts
                    for text, label in texts_to_process:
                        if count >= max_samples:
                            break
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
                    errors += 1
                    continue
            
            # Cleanup
            del ds
            gc.collect()
            
            if human_added + ai_added > 0:
                self.datasets_used.append({
                    'name': name,
                    'human_samples': human_added,
                    'ai_samples': ai_added,
                    'url': config.get('url', ''),
                    'tier': config.get('tier', 0),
                })
                
        except Exception as e:
            self.checkpoint_mgr.log(f"Error processing {name}: {str(e)[:100]}")
            gc.collect()
            
        return human_added, ai_added
    
    def run_training(self, n_trials: int = 50, resume: bool = True):
        """Run the full training pipeline with batch processing."""
        print(BANNER)
        self.start_time = time.time()
        
        # Check for existing checkpoint
        if resume and self._restore_checkpoint():
            print(f"\n  📂 Resuming from checkpoint: {self.datasets_completed}/{len(self.datasets)} datasets completed")
            print(f"     Samples loaded: {self.total_human} human + {self.total_ai} AI = {self.total_human + self.total_ai}")
        else:
            self.checkpoint_mgr.log(f"Starting new training run: {self.model_name}")
        
        # Print configuration
        print(f"\n{'═'*70}")
        print(f"  CONFIGURATION")
        print(f"{'═'*70}")
        print(f"  Model name:      {self.model_name}")
        print(f"  Batch size:      {self.batch_size} datasets per batch")
        print(f"  Total datasets:  {len(self.datasets)}")
        print(f"  Optuna trials:   {n_trials}")
        print(f"  Resume enabled:  {resume}")
        
        # Estimate time
        avg_time_per_dataset = 120  # 2 minutes estimate per dataset
        if self.dataset_times:
            avg_time_per_dataset = sum(self.dataset_times) / len(self.dataset_times)
        remaining_datasets = len(self.datasets) - self.datasets_completed
        est_total = remaining_datasets * avg_time_per_dataset
        print(f"  Estimated time:  ~{int(est_total // 60)} minutes")
        
        # ===== STEP 1: LOAD DATASETS IN BATCHES =====
        print(f"\n{'═'*70}")
        print(f"  STEP 1: LOADING DATASETS (batch size: {self.batch_size})")
        print(f"{'═'*70}")
        
        remaining = self.datasets[self.datasets_completed:]
        total_batches = (len(remaining) + self.batch_size - 1) // self.batch_size
        
        progress = ProgressTracker(len(remaining), "Datasets")
        
        batch_num = 0
        for i in range(0, len(remaining), self.batch_size):
            if self.shutdown_requested:
                break
                
            batch = remaining[i:i + self.batch_size]
            batch_num += 1
            
            print(f"\n  ┌─ Batch {batch_num}/{total_batches} ─────────────────────────────────────")
            
            for config in batch:
                if self.shutdown_requested:
                    break
                    
                ds_start = time.time()
                tier_badge = f"[T{config['tier']}]"
                print(f"  │  {tier_badge} {config['name'][:45]:<45}", end='', flush=True)
                
                h, a = self.process_single_dataset(config)
                
                ds_time = time.time() - ds_start
                self.dataset_times.append(ds_time)
                
                if h + a > 0:
                    self.total_human += h
                    self.total_ai += a
                    print(f" ✓ +{h:,} H +{a:,} AI ({ds_time:.1f}s)")
                else:
                    print(f" ✗ failed ({ds_time:.1f}s)")
                
                self.datasets_completed += 1
                progress.update(ds_time)
                
            # Save checkpoint after each batch
            self._save_checkpoint()
            
            # Clear HuggingFace cache
            cache_dir = os.path.expanduser('~/.cache/huggingface/datasets')
            if os.path.exists(cache_dir):
                try:
                    import shutil
                    shutil.rmtree(cache_dir, ignore_errors=True)
                except:
                    pass
            gc.collect()
            
            print(f"  └─ Batch complete. Total: {self.total_human:,} H + {self.total_ai:,} AI")
            progress.display()
            print()
        
        print(f"\n  ✓ Data loading complete!")
        print(f"    Total samples: {self.total_human:,} human + {self.total_ai:,} AI = {self.total_human + self.total_ai:,}")
        
        if self.shutdown_requested:
            return
        
        # ===== STEP 2: BALANCE DATA =====
        print(f"\n{'═'*70}")
        print(f"  STEP 2: BALANCING DATA")
        print(f"{'═'*70}")
        
        if len(self.all_features) == 0:
            print("  ✗ ERROR: No samples were loaded from any dataset!")
            print("  ✗ Please check dataset availability and column configurations.")
            self.checkpoint_mgr.log("Training failed: No samples loaded")
            return
        
        X = np.array(self.all_features)
        y = np.array(self.all_labels)
        
        human_idx = np.where(y == 0)[0]
        ai_idx = np.where(y == 1)[0]
        
        print(f"  Raw counts: {len(human_idx):,} human, {len(ai_idx):,} AI")
        
        if len(human_idx) == 0 or len(ai_idx) == 0:
            print("  ✗ ERROR: Need both human and AI samples to train!")
            print(f"    Human samples: {len(human_idx)}")
            print(f"    AI samples: {len(ai_idx)}")
            self.checkpoint_mgr.log("Training failed: Imbalanced data (missing class)")
            return
        
        min_count = min(len(human_idx), len(ai_idx))
        
        # Ensure we have enough samples
        if min_count < 100:
            print(f"  ⚠ WARNING: Very few samples ({min_count} per class). Results may be unreliable.")
        
        np.random.seed(42)
        selected_human = np.random.choice(human_idx, min_count, replace=False)
        selected_ai = np.random.choice(ai_idx, min_count, replace=False)
        
        selected = np.concatenate([selected_human, selected_ai])
        np.random.shuffle(selected)
        
        self.X = X[selected]
        self.y = y[selected]
        
        # Free memory
        self.all_features = []
        self.all_labels = []
        gc.collect()
        
        print(f"  ✓ Balanced to {min_count:,} samples per class ({min_count * 2:,} total)")
        
        # ===== STEP 3: TRAIN MODEL =====
        print(f"\n{'═'*70}")
        print(f"  STEP 3: TRAINING MODEL")
        print(f"{'═'*70}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        print(f"  Train set: {len(X_train):,} samples")
        print(f"  Test set:  {len(X_test):,} samples")
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Hyperparameter optimization
        best_params = {}
        if OPTUNA_AVAILABLE and n_trials > 0:
            print(f"\n  Optimizing hyperparameters ({n_trials} trials)...")
            
            opt_progress = tqdm(total=n_trials, desc="  Optimization", leave=True, 
                               bar_format='  {desc}: {bar:40} {percentage:3.0f}% | {n}/{total} trials | Best: {postfix}')
            
            best_score = 0
            
            def objective(trial):
                nonlocal best_score
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 25),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
                }
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1')
                score = scores.mean()
                
                if score > best_score:
                    best_score = score
                opt_progress.set_postfix_str(f"F1={best_score:.4f}")
                opt_progress.update(1)
                
                return score
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            opt_progress.close()
            
            best_params = study.best_params
            print(f"\n  ✓ Best F1 score: {study.best_value:.4f}")
            print(f"  ✓ Best params: {best_params}")
        
        # Train final model
        print(f"\n  Training final model...")
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
        
        print(f"\n  {'─'*50}")
        print(f"  EVALUATION RESULTS")
        print(f"  {'─'*50}")
        print(f"  Accuracy:  {self.results['accuracy']:.4f}")
        print(f"  Precision: {self.results['precision']:.4f}")
        print(f"  Recall:    {self.results['recall']:.4f}")
        print(f"  F1 Score:  {self.results['f1']:.4f}")
        print(f"  ROC AUC:   {self.results['roc_auc']:.4f}")
        
        # ===== STEP 4: SAVE MODEL =====
        print(f"\n{'═'*70}")
        print(f"  STEP 4: SAVING MODEL")
        print(f"{'═'*70}")
        
        self._save_model()
        
        # Clear checkpoint after successful completion
        self.checkpoint_mgr.clear()
        
        # Print summary
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        
        print(f"\n{'═'*70}")
        print(f"  🎉 TRAINING COMPLETE!")
        print(f"{'═'*70}")
        print(f"  Total time: {elapsed_str}")
        print(f"  Datasets processed: {len(self.datasets_used)}")
        print(f"  Final model: {self.output_dir / self.model_name}")
        
        print(f"\n  Top 10 Features:")
        for i, (name, imp) in enumerate(self.sorted_features[:10], 1):
            bar = '█' * int(imp * 40)
            print(f"    {i:2}. {name:30} {imp:.4f} {bar}")
        
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
            'version': '3.0.0-robust',
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
            'training_time_seconds': time.time() - self.start_time,
            'data_hash': hashlib.sha256(self.X.tobytes()).hexdigest()[:16],
            'model_hash': hashlib.sha256(pickle.dumps(self.model)).hexdigest()[:16],
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
            'version': receipt['version'],
            'timestamp': receipt['timestamp'],
            'accuracy': self.results['accuracy'],
            'f1': self.results['f1'],
            'datasets': len(self.datasets_used),
            'samples': receipt['total_samples'],
        }
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved metadata.json")
        
    def _save_js_config(self, model_dir: Path, receipt: Dict):
        """Generate JavaScript configuration file."""
        weights = receipt['feature_weights_normalized']
        
        js_content = f"""/**
 * Veritas {self.model_name} ML Configuration
 * Generated: {receipt['timestamp']}
 * Version: {receipt['version']}
 * 
 * Training Statistics:
 * - Datasets: {len(receipt['datasets_used'])}
 * - Total Samples: {receipt['total_samples']:,}
 * - Test Accuracy: {receipt['results']['accuracy']:.4f}
 * - Test F1 Score: {receipt['results']['f1']:.4f}
 * - ROC AUC: {receipt['results']['roc_auc']:.4f}
 * - Training Time: {int(receipt['training_time_seconds'])}s
 */

const VERITAS_{self.model_name.upper()}_CONFIG = {{
    version: '{receipt['version']}',
    modelName: '{self.model_name}',
    
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
}};

if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_{self.model_name.upper()}_CONFIG;
}}
"""
        
        with open(model_dir / 'veritas_config.js', 'w') as f:
            f.write(js_content)


def main():
    parser = argparse.ArgumentParser(
        description='Veritas Sunrise Robust Training with Crash Recovery',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_sunrise_robust.py                    # Default training
  python train_sunrise_robust.py --quick            # Quick mode (Tier 1 only)
  python train_sunrise_robust.py --no-resume        # Start fresh, ignore checkpoints
  python train_sunrise_robust.py --batch-size 5     # Larger batches
  python train_sunrise_robust.py --tiers 1,2        # Only Tier 1 and 2 datasets
        """
    )
    parser.add_argument('--name', type=str, default='Sunrise', help='Model name')
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
    trainer = RobustBatchTrainer(
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
