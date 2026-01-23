#!/usr/bin/env python3
"""
FLARE v3 Training Pipeline - Enhanced Humanization Detection
=============================================================

Specialized model for detecting AI text that has been processed through
humanization tools (Undetectable AI, Quillbot, StealthWriter, etc.)

KEY IMPROVEMENTS over v2:
1. More granular attack type detection
2. Cross-validation on RAID attack types
3. Authenticity signal integration
4. Hard negative mining from RAID paraphrase attacks

USAGE:
    python train_flare_v3.py --full          # Full training
    python train_flare_v3.py --quick         # Quick test
"""

import os
import sys
import json
import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("Warning: XGBoost not available")

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    sys.exit(1)

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import FeatureExtractorV3


BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ███████╗██╗      █████╗ ██████╗ ███████╗    ██╗   ██╗██████╗                ║
║   ██╔════╝██║     ██╔══██╗██╔══██╗██╔════╝    ██║   ██║╚════██╗               ║
║   █████╗  ██║     ███████║██████╔╝█████╗      ██║   ██║ █████╔╝               ║
║   ██╔══╝  ██║     ██╔══██║██╔══██╗██╔══╝      ╚██╗ ██╔╝ ╚═══██╗               ║
║   ██║     ███████╗██║  ██║██║  ██║███████╗     ╚████╔╝ ██████╔╝               ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝      ╚═══╝  ╚═════╝                ║
║                                                                               ║
║              Advanced Humanization Detection Model                            ║
║         Detects AI text modified by paraphrasing & humanization tools         ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# Attack types that represent humanization attempts
HUMANIZATION_ATTACKS = [
    'paraphrase',           # Main paraphrasing attack
    'synonym',              # Synonym substitution
    'perplexity_misspelling',  # Strategic typos to evade
    'alternative_spelling',    # Alternative spellings
]

# Attacks that are just noise/obfuscation (not semantic changes)
OBFUSCATION_ATTACKS = [
    'homoglyph',
    'whitespace', 
    'upper_lower',
    'zero_width_space',
    'insert_paragraphs',
    'article_deletion',
    'number',
]


@dataclass
class Sample:
    """Training sample."""
    text: str
    label: int  # 0 = genuine human, 1 = humanized AI
    attack: str
    source_model: str
    domain: str


@dataclass
class FlareConfig:
    """Flare v3 configuration."""
    max_samples: int = 300000
    test_size: float = 0.15
    min_text_length: int = 100
    max_text_length: int = 5000
    use_embeddings: bool = True
    embedding_model: str = 'all-MiniLM-L6-v2'
    n_optuna_trials: int = 50
    random_state: int = 42


class FlareDataLoader:
    """Load data for humanization detection."""
    
    def __init__(self, config: FlareConfig):
        self.config = config
    
    def load_raid_humanization_data(self, max_samples: int = 200000) -> List[Sample]:
        """
        Load RAID with focus on humanization attacks.
        
        Label scheme:
        - 0 = Genuine human text
        - 1 = Humanized AI text (paraphrase, synonym, etc.)
        """
        print("\n📥 Loading RAID for humanization detection...")
        samples = []
        
        try:
            raid = load_dataset('liamdugan/raid', split='train', streaming=True)
            
            human_count = 0
            humanized_count = 0
            target_per_class = max_samples // 2
            
            for item in tqdm(raid, desc="Loading RAID", total=max_samples):
                if human_count >= target_per_class and humanized_count >= target_per_class:
                    break
                
                text = item.get('generation', '') or item.get('text', '')
                if not text or len(text) < self.config.min_text_length:
                    continue
                if len(text) > self.config.max_text_length:
                    text = text[:self.config.max_text_length]
                
                model = item.get('model', 'unknown')
                attack = item.get('attack', 'none')
                domain = item.get('domain', 'unknown')
                
                # Determine label
                if model.lower() == 'human':
                    if human_count < target_per_class:
                        samples.append(Sample(
                            text=text,
                            label=0,  # Genuine human
                            attack='none',
                            source_model='human',
                            domain=domain
                        ))
                        human_count += 1
                else:
                    # AI-generated - check if humanization attack
                    if attack in HUMANIZATION_ATTACKS:
                        if humanized_count < target_per_class:
                            samples.append(Sample(
                                text=text,
                                label=1,  # Humanized AI
                                attack=attack,
                                source_model=model,
                                domain=domain
                            ))
                            humanized_count += 1
            
            print(f"  ✓ Loaded {len(samples):,} samples")
            print(f"    Human: {human_count:,}")
            print(f"    Humanized AI: {humanized_count:,}")
            
        except Exception as e:
            print(f"  ✗ Loading failed: {e}")
        
        return samples
    
    def load_additional_human_samples(self, max_samples: int = 50000) -> List[Sample]:
        """Load additional genuine human samples for better balance."""
        print("\n📥 Loading additional human samples...")
        samples = []
        
        # WritingPrompts - creative human writing
        try:
            ds = load_dataset('euclaise/writingprompts', split='train')
            for item in tqdm(ds, desc="WritingPrompts"):
                if len(samples) >= max_samples // 2:
                    break
                text = item.get('story', '')
                if text and len(text) >= self.config.min_text_length:
                    samples.append(Sample(
                        text=text[:self.config.max_text_length],
                        label=0,
                        attack='none',
                        source_model='human',
                        domain='creative'
                    ))
        except Exception as e:
            print(f"  ⚠ WritingPrompts failed: {e}")
        
        # OpenWebText - web content
        try:
            ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
            count = 0
            for item in ds:
                if count >= max_samples // 2:
                    break
                text = item.get('text', '')
                if text and len(text) >= self.config.min_text_length:
                    samples.append(Sample(
                        text=text[:self.config.max_text_length],
                        label=0,
                        attack='none',
                        source_model='human',
                        domain='web'
                    ))
                    count += 1
        except Exception as e:
            print(f"  ⚠ OpenWebText failed: {e}")
        
        print(f"  ✓ Loaded {len(samples):,} additional human samples")
        return samples
    
    def balance_dataset(self, samples: List[Sample]) -> List[Sample]:
        """Balance the dataset."""
        human = [s for s in samples if s.label == 0]
        humanized = [s for s in samples if s.label == 1]
        
        min_count = min(len(human), len(humanized))
        np.random.shuffle(human)
        np.random.shuffle(humanized)
        
        balanced = human[:min_count] + humanized[:min_count]
        np.random.shuffle(balanced)
        
        print(f"\n📊 Balanced to {len(balanced):,} samples ({min_count:,} per class)")
        return balanced


class FlareFeatureEngine:
    """Feature extraction for Flare v3."""
    
    def __init__(self, config: FlareConfig):
        self.config = config
        self.extractor = FeatureExtractorV3()
        self.scaler = StandardScaler()
        self.embedder = None
        
        if config.use_embeddings and ST_AVAILABLE:
            print(f"🔧 Loading embedding model: {config.embedding_model}")
            self.embedder = SentenceTransformer(config.embedding_model)
    
    def extract_humanization_features(self, text: str) -> np.ndarray:
        """
        Extract features specifically tuned for humanization detection.
        
        Key features for detecting humanization:
        1. Inconsistent formality (human vs polished)
        2. Awkward phrasings from paraphrasing
        3. Unnatural synonym substitutions
        4. Sentence rhythm disruption
        """
        base_features = self.extractor.extract_feature_vector(text)
        
        # Additional humanization-specific features
        extra_features = self._extract_paraphrase_signals(text)
        
        return np.concatenate([base_features, extra_features])
    
    def _extract_paraphrase_signals(self, text: str) -> np.ndarray:
        """Extract features that indicate paraphrasing."""
        import re
        
        features = []
        text_lower = text.lower()
        words = text.split()
        
        # 1. Unusual word combinations (from synonym substitution)
        # Look for formal word + casual word patterns
        formal_casual_mismatch = 0
        formal_words = {'utilize', 'implement', 'facilitate', 'endeavor', 'commence'}
        casual_words = {'kinda', 'gonna', 'wanna', 'stuff', 'things'}
        has_formal = any(w in text_lower for w in formal_words)
        has_casual = any(w in text_lower for w in casual_words)
        if has_formal and has_casual:
            formal_casual_mismatch = 1
        features.append(formal_casual_mismatch)
        
        # 2. Sentence-initial variety (paraphrasing often creates uniform starts)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) >= 3:
            starts = [s.split()[0].lower() if s.split() else '' for s in sentences[:10]]
            unique_starts = len(set(starts)) / len(starts) if starts else 1
            features.append(1 - unique_starts)  # Low variety = potential paraphrase
        else:
            features.append(0)
        
        # 3. Awkward transitions (paraphrasing breaks natural flow)
        awkward_transitions = len(re.findall(
            r'\b(but however|and also|so therefore|yet still)\b', 
            text_lower
        ))
        features.append(awkward_transitions)
        
        # 4. Redundant phrases (common in paraphrased text)
        redundant = len(re.findall(
            r'\b(in order to|at this point in time|due to the fact that|for the purpose of)\b',
            text_lower
        ))
        features.append(redundant)
        
        # 5. Unnatural emphasis patterns
        emphasis = len(re.findall(r'\b(very|really|extremely|incredibly|absolutely)\b', text_lower))
        features.append(emphasis / max(len(words), 1))
        
        # 6. Passive voice overuse (common in AI + paraphrasing)
        passive = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', text_lower))
        features.append(passive / max(len(sentences), 1))
        
        # 7. Sentence length inconsistency after paraphrasing
        if len(sentences) >= 3:
            sent_lens = [len(s.split()) for s in sentences]
            sent_lens = [l for l in sent_lens if l > 0]
            if sent_lens:
                # High variance in consecutive sentences suggests paraphrasing
                diffs = [abs(sent_lens[i] - sent_lens[i-1]) for i in range(1, len(sent_lens))]
                avg_diff = np.mean(diffs) if diffs else 0
                features.append(avg_diff)
            else:
                features.append(0)
        else:
            features.append(0)
        
        # 8. Conjunction density (paraphrasing often adds connectors)
        conjunctions = len(re.findall(r'\b(and|but|or|so|yet|however|therefore|thus)\b', text_lower))
        features.append(conjunctions / max(len(words), 1))
        
        return np.array(features)
    
    def extract_all_features(self, samples: List[Sample]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract all features."""
        print("\n📊 Extracting features...")
        
        texts = [s.text for s in samples]
        labels = np.array([s.label for s in samples])
        
        # Heuristic features
        heuristic = []
        for text in tqdm(texts, desc="Heuristic features"):
            heuristic.append(self.extract_humanization_features(text))
        heuristic = np.array(heuristic)
        print(f"  Heuristic features: {heuristic.shape}")
        
        # Embeddings
        if self.config.use_embeddings and self.embedder:
            print("  Extracting embeddings...")
            embeddings = self.embedder.encode(texts, batch_size=64, show_progress_bar=True)
            features = np.hstack([heuristic, embeddings])
        else:
            features = heuristic
        
        print(f"  Total features: {features.shape[1]}")
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return features, labels
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.scaler.transform(features)


class FlareTrainer:
    """Train Flare v3 model."""
    
    def __init__(self, config: FlareConfig, output_dir: str = './models/FlareV3'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.feature_engine = None
    
    def train(self, samples: List[Sample], optimize: bool = True) -> Dict:
        """Train the model."""
        print("\n" + "="*70)
        print("TRAINING FLARE v3")
        print("="*70)
        
        start_time = time.time()
        
        # Feature extraction
        self.feature_engine = FlareFeatureEngine(self.config)
        features, labels = self.feature_engine.extract_all_features(samples)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=self.config.test_size,
            stratify=labels,
            random_state=self.config.random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=0.1,
            stratify=y_train,
            random_state=self.config.random_state
        )
        
        print(f"\nData splits: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Scale
        X_train_scaled = self.feature_engine.fit_transform(X_train)
        X_val_scaled = self.feature_engine.transform(X_val)
        X_test_scaled = self.feature_engine.transform(X_test)
        
        # Train
        if optimize and OPTUNA_AVAILABLE:
            best_params = self._optimize(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            best_params = {
                'n_estimators': 1000,
                'max_depth': 7,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
            }
        
        print("\n🚀 Training final model...")
        self.model = xgb.XGBClassifier(
            **best_params,
            use_label_encoder=False,
            eval_metric='logloss',
            early_stopping_rounds=50,
            random_state=self.config.random_state,
            n_jobs=-1
        )
        
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=True
        )
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'training_time': f"{(time.time() - start_time)/60:.1f} minutes",
            'best_params': best_params,
        }
        
        print(f"\n📊 Results:")
        print(f"  Accuracy:  {results['accuracy']*100:.2f}%")
        print(f"  Precision: {results['precision']*100:.2f}%")
        print(f"  Recall:    {results['recall']*100:.2f}%")
        print(f"  F1:        {results['f1']*100:.2f}%")
        print(f"  ROC AUC:   {results['roc_auc']*100:.2f}%")
        
        cm = results['confusion_matrix']
        print(f"\n  Confusion Matrix:")
        print(f"    True Human:     {cm[0][0]:5d} correct, {cm[0][1]:5d} false positive")
        print(f"    True Humanized: {cm[1][0]:5d} false negative, {cm[1][1]:5d} correct")
        
        # Save
        self._save(results)
        
        return results
    
    def _optimize(self, X_train, y_train, X_val, y_val) -> Dict:
        """Hyperparameter optimization."""
        print("\n🔍 Optimizing hyperparameters...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
            
            model = xgb.XGBClassifier(
                **params,
                use_label_encoder=False,
                eval_metric='logloss',
                early_stopping_rounds=20,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            return accuracy_score(y_val, model.predict(X_val))
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.config.n_optuna_trials, show_progress_bar=True)
        
        print(f"  Best accuracy: {study.best_value:.4f}")
        return study.best_params
    
    def _save(self, results: Dict):
        """Save model."""
        print("\n💾 Saving model...")
        
        self.model.save_model(str(self.output_dir / 'flare_v3.json'))
        
        with open(self.output_dir / 'scaler_v3.pkl', 'wb') as f:
            pickle.dump(self.feature_engine.scaler, f)
        
        metadata = {
            'version': '3.0',
            'name': 'Flare v3',
            'task': 'Humanization Detection',
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'created': datetime.now().isoformat(),
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        with open(self.output_dir / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"  Saved to: {self.output_dir}")


def main():
    print(BANNER)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--output-dir', default='./models/FlareV3')
    args = parser.parse_args()
    
    config = FlareConfig()
    
    if args.quick:
        config.max_samples = 50000
        config.n_optuna_trials = 10
        print("⚡ Quick mode")
    
    # Load data
    loader = FlareDataLoader(config)
    samples = loader.load_raid_humanization_data()
    samples.extend(loader.load_additional_human_samples(30000))
    samples = loader.balance_dataset(samples)
    
    # Train
    trainer = FlareTrainer(config, args.output_dir)
    results = trainer.train(samples)
    
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
