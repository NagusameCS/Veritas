#!/usr/bin/env python3
"""
Veritas Sunrise V4 Training - FAST Edition with Real-Time Feedback
===================================================================

Optimizations:
- Parallel feature extraction with multiprocessing
- Progress bars with tqdm for every stage
- Batch processing for memory efficiency
- Numpy warning suppression (handled in code)
- Optimized dataset streaming
- Real-time sample counts and ETA
"""

import os
import sys
import gc
import json
import time
import pickle
import warnings
import signal
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
from tqdm import tqdm

# Suppress numpy warnings (we handle NaN/Inf in code)
warnings.filterwarnings('ignore', category=RuntimeWarning)
np.seterr(divide='ignore', invalid='ignore')

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# Import feature extractor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v4 import FeatureExtractorV4

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║   ███████╗██╗   ██╗███╗   ██╗██████╗ ██╗███████╗███████╗                      ║
║   ██╔════╝██║   ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝                      ║
║   ███████╗██║   ██║██╔██╗ ██║██████╔╝██║███████╗█████╗                        ║
║   ╚════██║██║   ██║██║╚██╗██║██╔══██╗██║╚════██║██╔══╝                        ║
║   ███████║╚██████╔╝██║ ╚████║██║  ██║██║███████║███████╗                      ║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝                      ║
║                                                                               ║
║    V4.0 FAST - Parallel Training with Real-Time Feedback                      ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# Global feature extractor for multiprocessing
_extractor = None

def init_extractor():
    """Initialize feature extractor in worker process."""
    global _extractor
    _extractor = FeatureExtractorV4()

def extract_features_worker(args):
    """Worker function for parallel feature extraction."""
    text, label = args
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractorV4()
    
    try:
        if not text or len(text) < 100:
            return None
        
        features = _extractor.extract_feature_vector(text)
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return None
        
        return (features, label)
    except Exception:
        return None


# Optimized dataset configurations - balanced for speed + quality
FAST_DATASETS = [
    # Tier 1: Primary paired datasets (both human and AI)
    {
        'name': 'aadityaubhat/GPT-wiki-intro',
        'max_samples': 8000,  # 8k pairs = 16k samples
        'special': 'dual_column',
        'human_text_col': 'wiki_intro',
        'ai_text_col': 'generated_intro',
        'tier': 1
    },
    # Tier 1: Human-only datasets
    {
        'name': 'imdb',
        'max_samples': 8000,
        'special': 'human_only',
        'text_col': 'text',
        'tier': 1
    },
    {
        'name': 'squad',
        'max_samples': 6000,
        'special': 'human_only',
        'text_col': 'context',
        'tier': 1
    },
    # Tier 2: Additional human text
    {
        'name': 'billsum',
        'max_samples': 4000,
        'special': 'human_only',
        'text_col': 'text',
        'tier': 2
    },
    {
        'name': 'cnn_dailymail',
        'config': '3.0.0',
        'max_samples': 5000,
        'special': 'human_only',
        'text_col': 'article',
        'tier': 2
    },
    # Tier 2: AI detection specific
    {
        'name': 'Hello-SimpleAI/HC3',
        'config': 'all',
        'max_samples': 6000,
        'special': 'hc3',
        'tier': 2
    },
]


class SunriseV4FastTrainer:
    """Fast Sunrise V4 trainer with parallel processing and progress feedback."""
    
    def __init__(self, output_dir: str = './models', n_workers: int = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use CPU count - 1 workers, minimum 2
        self.n_workers = n_workers or max(2, mp.cpu_count() - 1)
        
        # Initialize feature extractor for main process
        self.feature_extractor = FeatureExtractorV4()
        self.feature_names = self.feature_extractor.get_feature_names()
        
        # Training data
        self.all_features = []
        self.all_labels = []
        self.total_human = 0
        self.total_ai = 0
        
        # Model artifacts
        self.model = None
        self.scaler = None
        self.results = None
        
        # Timing
        self.start_time = None
        
        # Shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
    def _handle_shutdown(self, signum, frame):
        print("\n\n  ⚠️  Shutdown requested - cleaning up...")
        self.shutdown_requested = True
        sys.exit(0)
        
    def collect_texts_from_dataset(self, config: Dict) -> List[Tuple[str, int]]:
        """Collect texts from a dataset without extracting features yet."""
        texts_and_labels = []
        name = config['name']
        max_samples = config['max_samples']
        special = config.get('special', '')
        
        try:
            # Load dataset
            ds = None
            dataset_config = config.get('config', None)
            splits_to_try = ['train', 'test', 'validation']
            
            for split in splits_to_try:
                try:
                    if dataset_config:
                        ds = load_dataset(name, dataset_config, split=split, streaming=True, trust_remote_code=True)
                    else:
                        ds = load_dataset(name, split=split, streaming=True, trust_remote_code=True)
                    break
                except Exception:
                    continue
            
            if ds is None:
                return []
            
            # Collect texts based on dataset type
            samples_collected = 0
            
            for item in ds:
                if samples_collected >= max_samples:
                    break
                
                if self.shutdown_requested:
                    break
                
                try:
                    if special == 'dual_column':
                        human_text = item.get(config['human_text_col'], '')
                        ai_text = item.get(config['ai_text_col'], '')
                        
                        if human_text and len(human_text) >= 100:
                            texts_and_labels.append((human_text, 0))
                        if ai_text and len(ai_text) >= 100:
                            texts_and_labels.append((ai_text, 1))
                        samples_collected += 1
                        
                    elif special == 'human_only':
                        text = item.get(config['text_col'], '')
                        if text and len(text) >= 100:
                            texts_and_labels.append((text, 0))
                            samples_collected += 1
                            
                    elif special == 'ai_only':
                        text = item.get(config['text_col'], '')
                        if text and len(text) >= 100:
                            texts_and_labels.append((text, 1))
                            samples_collected += 1
                            
                    elif special == 'hc3':
                        # HC3 has human_answers and chatgpt_answers lists
                        human_answers = item.get('human_answers', [])
                        ai_answers = item.get('chatgpt_answers', [])
                        
                        for ans in human_answers[:2]:  # Take up to 2 per sample
                            if ans and len(ans) >= 100:
                                texts_and_labels.append((ans, 0))
                        for ans in ai_answers[:2]:
                            if ans and len(ans) >= 100:
                                texts_and_labels.append((ans, 1))
                        samples_collected += 1
                        
                except Exception:
                    continue
            
            return texts_and_labels
            
        except Exception as e:
            print(f"      ⚠️  Error: {str(e)[:50]}")
            return []
    
    def run_training(self, n_trials: int = 30):
        """Run the complete training pipeline with parallel processing."""
        print(BANNER)
        self.start_time = time.time()
        
        print(f"{'═'*70}")
        print(f"  SUNRISE V4 FAST TRAINING")
        print(f"{'═'*70}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Workers: {self.n_workers}")
        print(f"  Datasets: {len(FAST_DATASETS)}")
        print(f"  Note: Metadata EXCLUDED - handled as absolute flag")
        print()
        
        # =====================================================================
        # STEP 1: COLLECT ALL TEXTS
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 1: COLLECTING TEXTS FROM DATASETS")
        print(f"{'═'*70}")
        print()
        
        all_texts_labels = []
        
        for i, config in enumerate(FAST_DATASETS, 1):
            name = config['name']
            print(f"  [{i}/{len(FAST_DATASETS)}] {name}")
            print(f"      Max samples: {config['max_samples']} | Type: {config.get('special', 'labeled')}")
            
            texts = self.collect_texts_from_dataset(config)
            
            human_count = sum(1 for _, l in texts if l == 0)
            ai_count = sum(1 for _, l in texts if l == 1)
            
            print(f"      ✓ Collected: {len(texts)} texts (Human: {human_count}, AI: {ai_count})")
            all_texts_labels.extend(texts)
            
            if self.shutdown_requested:
                break
        
        print()
        total_human = sum(1 for _, l in all_texts_labels if l == 0)
        total_ai = sum(1 for _, l in all_texts_labels if l == 1)
        print(f"  📊 Total texts collected: {len(all_texts_labels)}")
        print(f"      Human: {total_human} | AI: {total_ai}")
        print()
        
        if len(all_texts_labels) < 1000:
            print("  ❌ Not enough samples collected. Aborting.")
            return
        
        # =====================================================================
        # STEP 2: PARALLEL FEATURE EXTRACTION
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 2: PARALLEL FEATURE EXTRACTION ({self.n_workers} workers)")
        print(f"{'═'*70}")
        print()
        
        # Process in batches to show progress
        batch_size = 500
        total_batches = (len(all_texts_labels) + batch_size - 1) // batch_size
        
        with ProcessPoolExecutor(max_workers=self.n_workers, initializer=init_extractor) as executor:
            with tqdm(total=len(all_texts_labels), desc="  Extracting features", 
                      unit="samples", ncols=80, colour='green') as pbar:
                
                # Submit all tasks
                futures = []
                for text, label in all_texts_labels:
                    future = executor.submit(extract_features_worker, (text, label))
                    futures.append(future)
                
                # Collect results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        features, label = result
                        self.all_features.append(features)
                        self.all_labels.append(label)
                        if label == 0:
                            self.total_human += 1
                        else:
                            self.total_ai += 1
                    pbar.update(1)
        
        print()
        print(f"  ✓ Features extracted: {len(self.all_features)} samples")
        print(f"      Human: {self.total_human} | AI: {self.total_ai}")
        print()
        
        # Convert to numpy arrays
        X = np.array(self.all_features, dtype=np.float32)
        y = np.array(self.all_labels, dtype=np.int32)
        
        # Clean any remaining NaN/Inf
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        print(f"  ✓ Clean samples: {len(X)}")
        
        # =====================================================================
        # STEP 3: TRAIN/TEST SPLIT
        # =====================================================================
        print()
        print(f"{'═'*70}")
        print(f"  STEP 3: PREPARING DATA")
        print(f"{'═'*70}")
        print()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print()
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # =====================================================================
        # STEP 4: HYPERPARAMETER OPTIMIZATION
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 4: HYPERPARAMETER OPTIMIZATION ({n_trials} trials)")
        print(f"{'═'*70}")
        print()
        
        best_params = None
        
        if OPTUNA_AVAILABLE:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 300),
                    'max_depth': trial.suggest_int('max_depth', 10, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                }
                
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
                return scores.mean()
            
            study = optuna.create_study(direction='maximize')
            
            with tqdm(total=n_trials, desc="  Optimizing", unit="trial", ncols=80, colour='blue') as pbar:
                def callback(study, trial):
                    pbar.update(1)
                    pbar.set_postfix({'best': f"{study.best_value:.4f}"})
                
                study.optimize(objective, n_trials=n_trials, callbacks=[callback], show_progress_bar=False)
            
            best_params = study.best_params
            print()
            print(f"  ✓ Best accuracy: {study.best_value:.4f}")
            print(f"  ✓ Best params: {best_params}")
        else:
            print("  Using default parameters (optuna not available)")
            best_params = {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt'
            }
        
        print()
        
        # =====================================================================
        # STEP 5: FINAL MODEL TRAINING
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 5: TRAINING FINAL MODEL")
        print(f"{'═'*70}")
        print()
        
        print("  Training RandomForest with optimized parameters...")
        self.model = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)
        print("  ✓ Model trained")
        print()
        
        # =====================================================================
        # STEP 6: EVALUATION
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 6: EVALUATION")
        print(f"{'═'*70}")
        print()
        
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"  ┌{'─'*50}┐")
        print(f"  │  {'Metric':<20} {'Value':>25}   │")
        print(f"  ├{'─'*50}┤")
        print(f"  │  {'Accuracy':<20} {accuracy*100:>24.2f}%  │")
        print(f"  │  {'Precision':<20} {precision*100:>24.2f}%  │")
        print(f"  │  {'Recall':<20} {recall*100:>24.2f}%  │")
        print(f"  │  {'F1 Score':<20} {f1*100:>24.2f}%  │")
        print(f"  │  {'ROC-AUC':<20} {roc_auc:>25.4f}  │")
        print(f"  └{'─'*50}┘")
        print()
        
        self.results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc
        }
        
        # Feature importance
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[::-1]
        
        print("  Top 15 Most Important Features:")
        print("  " + "─" * 50)
        for i, idx in enumerate(sorted_idx[:15]):
            bar_len = int(importance[idx] * 100)
            bar = '█' * min(bar_len, 20)
            print(f"  {i+1:2}. {self.feature_names[idx]:<30} {importance[idx]:.4f} {bar}")
        print()
        
        # =====================================================================
        # STEP 7: SAVE MODEL
        # =====================================================================
        print(f"{'═'*70}")
        print(f"  STEP 7: SAVING MODEL")
        print(f"{'═'*70}")
        print()
        
        self._save_model(X, y, best_params)
        
        # =====================================================================
        # COMPLETE
        # =====================================================================
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        
        print()
        print(f"{'═'*70}")
        print(f"  ✓ TRAINING COMPLETE")
        print(f"{'═'*70}")
        print(f"  Model: Sunrise V4.0")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Training samples: {len(X)}")
        print(f"  Time: {minutes}m {seconds}s")
        print()
        
    def _save_model(self, X, y, best_params):
        """Save all model artifacts."""
        model_dir = self.output_dir / 'SunriseV4'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_dir / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print(f"  ✓ Saved: model.pkl")
        
        # Save scaler
        with open(model_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"  ✓ Saved: scaler.pkl")
        
        # Save metadata
        metadata = {
            'name': 'Sunrise',
            'version': '4.0',
            'created': datetime.now().isoformat(),
            'features': len(self.feature_names),
            'feature_names': self.feature_names,
            'training_samples': len(X),
            'human_samples': self.total_human,
            'ai_samples': self.total_ai,
            'accuracy': self.results['accuracy'],
            'precision': self.results['precision'],
            'recall': self.results['recall'],
            'f1': self.results['f1'],
            'roc_auc': self.results['roc_auc'],
            'hyperparameters': best_params,
            'metadata_handling': 'absolute_flag',
            'metadata_flag_message': 'Unknown metadata was detected within the text, this may likely be an artifact of LLM usage.'
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Saved: metadata.json")
        
        # Generate training receipt
        receipt = {
            'model': 'Sunrise V4.0',
            'trained': datetime.now().isoformat(),
            'accuracy': f"{self.results['accuracy']*100:.2f}%",
            'precision': f"{self.results['precision']*100:.2f}%",
            'recall': f"{self.results['recall']*100:.2f}%",
            'f1_score': f"{self.results['f1']*100:.2f}%",
            'roc_auc': f"{self.results['roc_auc']:.4f}",
            'training_samples': len(X),
            'human_samples': self.total_human,
            'ai_samples': self.total_ai,
            'features': len(self.feature_names),
            'datasets_used': [d['name'] for d in FAST_DATASETS],
            'metadata_excluded': True
        }
        
        with open(model_dir / 'training_receipt.json', 'w') as f:
            json.dump(receipt, f, indent=2)
        print(f"  ✓ Saved: training_receipt.json")
        
        # Generate veritas_config.js for frontend
        # Calculate category weights from feature importance
        importance = self.model.feature_importances_
        
        # Group features by category
        category_importance = {
            'sentence': 0, 'vocabulary': 0, 'zipf': 0, 'burstiness': 0,
            'readability': 0, 'ngram': 0, 'punctuation': 0, 'structure': 0,
            'function_words': 0, 'ai_signatures': 0, 'humanizer': 0,
            'advanced': 0, 'word_patterns': 0
        }
        
        for i, name in enumerate(self.feature_names):
            for cat in category_importance:
                if cat in name.lower():
                    category_importance[cat] += importance[i]
                    break
        
        # Normalize to sum to 1
        total_imp = sum(category_importance.values())
        if total_imp > 0:
            for cat in category_importance:
                category_importance[cat] /= total_imp
        
        config_js = f'''/**
 * VERITAS Sunrise V4.0 Configuration
 * Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
 * 
 * This model uses {len(self.feature_names)} features for AI detection.
 * Metadata (Category 11) is EXCLUDED from ML - operates as absolute flag.
 */

const VERITAS_SUNRISE_CONFIG = {{
    name: 'Sunrise',
    version: '4.0',
    features: {len(self.feature_names)},
    accuracy: {self.results['accuracy']:.4f},
    precision: {self.results['precision']:.4f},
    recall: {self.results['recall']:.4f},
    f1Score: {self.results['f1']:.4f},
    rocAuc: {self.results['roc_auc']:.4f},
    trainingSamples: {len(X)},
    humanSamples: {self.total_human},
    aiSamples: {self.total_ai},
    
    // Category weights derived from feature importance
    categoryWeights: {{
        syntaxVariance: {category_importance.get('sentence', 0.15):.4f},
        lexicalDiversity: {category_importance.get('vocabulary', 0.20):.4f},
        repetitionUniformity: {category_importance.get('ngram', 0.08):.4f},
        toneStability: {category_importance.get('burstiness', 0.05):.4f},
        grammarEntropy: {category_importance.get('structure', 0.05):.4f},
        perplexity: {category_importance.get('zipf', 0.07):.4f},
        authorshipDrift: {category_importance.get('advanced', 0.10):.4f},
        aiSignatures: {category_importance.get('ai_signatures', 0.15):.4f},
        humanizerDetection: {category_importance.get('humanizer', 0.10):.4f},
        metadataFormatting: 0.0  // EXCLUDED - absolute flag only
    }},
    
    // Metadata detection configuration
    metadataDetection: {{
        isAbsoluteFlag: true,
        excludedFromML: true,
        flagMessage: 'Unknown metadata was detected within the text, this may likely be an artifact of LLM usage.'
    }},
    
    // Top features by importance
    topFeatures: {json.dumps([self.feature_names[i] for i in np.argsort(importance)[::-1][:20]])},
    
    // Training metadata
    trainedAt: '{datetime.now().isoformat()}',
    datasetsUsed: {len(FAST_DATASETS)}
}};

// Export for Node.js/testing
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_SUNRISE_CONFIG;
}}
'''
        
        with open(model_dir / 'veritas_config.js', 'w') as f:
            f.write(config_js)
        print(f"  ✓ Saved: veritas_config.js")
        
        print()
        print(f"  📁 Model saved to: {model_dir}")


def main():
    trainer = SunriseV4FastTrainer(
        output_dir='./models',
        n_workers=None  # Auto-detect
    )
    trainer.run_training(n_trials=30)


if __name__ == '__main__':
    main()
