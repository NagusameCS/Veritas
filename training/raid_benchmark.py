#!/usr/bin/env python3
"""
RAID Benchmark Suite for VERITAS Models
========================================

Comprehensive evaluation on the RAID benchmark dataset.
Tests models across all attack types, domains, and AI generators.

This script evaluates:
1. Overall accuracy
2. Per-attack-type performance
3. Per-domain performance
4. Per-AI-model performance
5. High-confidence accuracy
6. False positive / false negative analysis

USAGE:
    python raid_benchmark.py --model supernova     # Benchmark SUPERNOVA
    python raid_benchmark.py --model flare         # Benchmark Flare
    python raid_benchmark.py --model all           # Benchmark all models
"""

import os
import sys
import json
import argparse
import pickle
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

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
    print("❌ datasets library required")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from feature_extractor_v3 import FeatureExtractorV3


BANNER = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║   ██████╗  █████╗ ██╗██████╗     ██████╗ ███████╗███╗   ██╗ ██████╗██╗  ██╗  ║
║   ██╔══██╗██╔══██╗██║██╔══██╗    ██╔══██╗██╔════╝████╗  ██║██╔════╝██║  ██║  ║
║   ██████╔╝███████║██║██║  ██║    ██████╔╝█████╗  ██╔██╗ ██║██║     ███████║  ║
║   ██╔══██╗██╔══██║██║██║  ██║    ██╔══██╗██╔══╝  ██║╚██╗██║██║     ██╔══██║  ║
║   ██║  ██║██║  ██║██║██████╔╝    ██████╔╝███████╗██║ ╚████║╚██████╗██║  ██║  ║
║   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═════╝     ╚═════╝ ╚══════╝╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝  ║
║                                                                               ║
║              Comprehensive AI Detection Benchmark Suite                       ║
║                  Testing VERITAS Models on RAID Dataset                       ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

# Attack types in RAID
ATTACK_TYPES = [
    'none',                     # No attack - raw AI output
    'paraphrase',               # Paraphrasing attack
    'synonym',                  # Synonym substitution
    'homoglyph',                # Character substitution
    'zero_width_space',         # Zero-width characters
    'whitespace',               # Whitespace manipulation
    'upper_lower',              # Case manipulation
    'insert_paragraphs',        # Paragraph insertion
    'article_deletion',         # Article word deletion
    'perplexity_misspelling',   # Strategic misspellings
    'alternative_spelling',     # Alternative spellings
    'number',                   # Number substitution
]

# Domains in RAID
DOMAINS = [
    'abstracts',    # Scientific abstracts
    'books',        # Book excerpts
    'news',         # News articles
    'poetry',       # Poetry
    'recipes',      # Cooking recipes
    'reddit',       # Reddit posts
    'reviews',      # Reviews
    'wiki',         # Wikipedia-style
]

# AI models in RAID
AI_MODELS = [
    'chatgpt',
    'gpt4',
    'llama2_70b',
    'llama2_13b',
    'llama2_7b',
    'mistral',
    'mixtral',
    'mpt',
    'cohere',
    'davinci',
    'llama_65b',
    'opt_175b',
]


@dataclass
class BenchmarkResult:
    """Results for a specific model."""
    model_name: str
    overall_accuracy: float = 0.0
    overall_precision: float = 0.0
    overall_recall: float = 0.0
    overall_f1: float = 0.0
    overall_auc: float = 0.0
    high_conf_accuracy: float = 0.0  # Only predictions >= 80%
    by_attack: Dict[str, Dict] = field(default_factory=dict)
    by_domain: Dict[str, Dict] = field(default_factory=dict)
    by_ai_model: Dict[str, Dict] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0
    total_samples: int = 0
    eval_time_seconds: float = 0.0


class ModelLoader:
    """Load VERITAS models for benchmarking."""
    
    def __init__(self, models_dir: str = './models'):
        self.models_dir = Path(models_dir)
        self.embedder = None
        self.extractor = FeatureExtractorV3()
    
    def load_supernova(self) -> Tuple[xgb.XGBClassifier, object]:
        """Load SUPERNOVA model."""
        # Try v2 (Zenith) first
        zenith_dir = self.models_dir / 'Zenith'
        supernova_dir = self.models_dir / 'Supernova'
        
        model_path = None
        scaler_path = None
        
        if (zenith_dir / 'supernova_v2.json').exists():
            model_path = zenith_dir / 'supernova_v2.json'
            scaler_path = zenith_dir / 'scaler_v2.pkl'
            print(f"  Loading SUPERNOVA v2 from {zenith_dir}")
        elif (supernova_dir / 'supernova_xgb.onnx').exists():
            # Load original ONNX model
            print(f"  Loading SUPERNOVA v1 from {supernova_dir}")
            # Note: Would need ONNX runtime for v1
            raise ValueError("ONNX loading not implemented - run train_supernova_v2.py first")
        else:
            raise FileNotFoundError("No SUPERNOVA model found")
        
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    
    def load_flare(self) -> Tuple[xgb.XGBClassifier, object]:
        """Load Flare model."""
        flare_v3_dir = self.models_dir / 'FlareV3'
        flare_v2_dir = self.models_dir / 'FlareV2_best'
        
        if (flare_v3_dir / 'flare_v3.json').exists():
            print(f"  Loading Flare v3 from {flare_v3_dir}")
            model = xgb.XGBClassifier()
            model.load_model(str(flare_v3_dir / 'flare_v3.json'))
            with open(flare_v3_dir / 'scaler_v3.pkl', 'rb') as f:
                scaler = pickle.load(f)
            return model, scaler
        else:
            raise FileNotFoundError("No Flare v3 model found - run train_flare_v3.py first")
    
    def get_embedder(self):
        """Get sentence embedder."""
        if self.embedder is None and ST_AVAILABLE:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedder
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """Extract features from texts."""
        # Heuristic features
        heuristics = []
        for text in texts:
            heuristics.append(self.extractor.extract_feature_vector(text))
        heuristics = np.array(heuristics)
        
        # Embeddings
        embedder = self.get_embedder()
        if embedder:
            embeddings = embedder.encode(texts, batch_size=64, show_progress_bar=False)
            features = np.hstack([heuristics, embeddings])
        else:
            features = heuristics
        
        return np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)


class RAIDBenchmark:
    """Run RAID benchmark on models."""
    
    def __init__(self, max_samples: int = 100000):
        self.max_samples = max_samples
        self.loader = ModelLoader()
    
    def load_raid_data(self) -> Tuple[List[str], List[int], List[str], List[str], List[str]]:
        """Load RAID benchmark data."""
        print("\n📥 Loading RAID benchmark data...")
        
        texts = []
        labels = []
        attacks = []
        domains = []
        ai_models = []
        
        raid = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        for item in tqdm(raid, desc="Loading RAID", total=self.max_samples):
            if len(texts) >= self.max_samples:
                break
            
            text = item.get('generation', '') or item.get('text', '')
            if not text or len(text) < 50:
                continue
            
            model = item.get('model', 'unknown')
            attack = item.get('attack', 'none')
            domain = item.get('domain', 'unknown')
            
            # Label: 0 = human, 1 = AI
            label = 0 if model.lower() == 'human' else 1
            
            texts.append(text[:5000])
            labels.append(label)
            attacks.append(attack)
            domains.append(domain)
            ai_models.append(model)
        
        print(f"  ✓ Loaded {len(texts):,} samples")
        print(f"    Human: {sum(1 for l in labels if l == 0):,}")
        print(f"    AI: {sum(1 for l in labels if l == 1):,}")
        
        return texts, labels, attacks, domains, ai_models
    
    def benchmark_model(
        self, 
        model: xgb.XGBClassifier, 
        scaler, 
        model_name: str,
        texts: List[str],
        labels: List[int],
        attacks: List[str],
        domains: List[str],
        ai_models: List[str]
    ) -> BenchmarkResult:
        """Run benchmark on a single model."""
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {model_name}")
        print('='*70)
        
        start_time = time.time()
        result = BenchmarkResult(model_name=model_name)
        
        # Extract features
        print("Extracting features...")
        features = self.loader.extract_features(texts)
        features_scaled = scaler.transform(features)
        
        # Predictions
        print("Running predictions...")
        y_pred = model.predict(features_scaled)
        y_prob = model.predict_proba(features_scaled)[:, 1]
        
        y_true = np.array(labels)
        
        # Overall metrics
        result.overall_accuracy = accuracy_score(y_true, y_pred)
        result.overall_precision = precision_score(y_true, y_pred)
        result.overall_recall = recall_score(y_true, y_pred)
        result.overall_f1 = f1_score(y_true, y_pred)
        result.overall_auc = roc_auc_score(y_true, y_prob)
        result.total_samples = len(texts)
        
        # High confidence accuracy
        high_conf_mask = (y_prob >= 0.8) | (y_prob <= 0.2)
        if high_conf_mask.sum() > 0:
            high_conf_pred = (y_prob[high_conf_mask] >= 0.5).astype(int)
            result.high_conf_accuracy = accuracy_score(y_true[high_conf_mask], high_conf_pred)
        
        # Confusion matrix analysis
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        result.false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        result.false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # By attack type
        print("Analyzing by attack type...")
        for attack in set(attacks):
            mask = np.array([a == attack for a in attacks])
            if mask.sum() > 10:
                result.by_attack[attack] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'count': int(mask.sum()),
                }
        
        # By domain
        print("Analyzing by domain...")
        for domain in set(domains):
            mask = np.array([d == domain for d in domains])
            if mask.sum() > 10:
                result.by_domain[domain] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'count': int(mask.sum()),
                }
        
        # By AI model
        print("Analyzing by AI model...")
        for ai_model in set(ai_models):
            if ai_model.lower() == 'human':
                continue
            mask = np.array([m == ai_model for m in ai_models])
            if mask.sum() > 10:
                result.by_ai_model[ai_model] = {
                    'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                    'count': int(mask.sum()),
                }
        
        result.eval_time_seconds = time.time() - start_time
        
        return result
    
    def print_results(self, result: BenchmarkResult):
        """Print benchmark results."""
        print(f"\n{'='*70}")
        print(f"RESULTS: {result.model_name}")
        print('='*70)
        
        print(f"\n📊 OVERALL METRICS")
        print(f"  {'Accuracy:':<20} {result.overall_accuracy*100:>6.2f}%")
        print(f"  {'High-Conf Accuracy:':<20} {result.high_conf_accuracy*100:>6.2f}%")
        print(f"  {'Precision:':<20} {result.overall_precision*100:>6.2f}%")
        print(f"  {'Recall:':<20} {result.overall_recall*100:>6.2f}%")
        print(f"  {'F1 Score:':<20} {result.overall_f1*100:>6.2f}%")
        print(f"  {'ROC AUC:':<20} {result.overall_auc*100:>6.2f}%")
        print(f"  {'False Positive Rate:':<20} {result.false_positive_rate*100:>6.2f}%")
        print(f"  {'False Negative Rate:':<20} {result.false_negative_rate*100:>6.2f}%")
        
        print(f"\n📈 BY ATTACK TYPE")
        print(f"  {'Attack':<25} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'-'*45}")
        for attack, metrics in sorted(result.by_attack.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {attack:<25} {metrics['accuracy']*100:>9.2f}% {metrics['count']:>10,}")
        
        print(f"\n📈 BY DOMAIN")
        print(f"  {'Domain':<25} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'-'*45}")
        for domain, metrics in sorted(result.by_domain.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {domain:<25} {metrics['accuracy']*100:>9.2f}% {metrics['count']:>10,}")
        
        print(f"\n📈 BY AI MODEL")
        print(f"  {'AI Model':<25} {'Accuracy':>10} {'Samples':>10}")
        print(f"  {'-'*45}")
        for ai_model, metrics in sorted(result.by_ai_model.items(), key=lambda x: -x[1]['accuracy']):
            print(f"  {ai_model:<25} {metrics['accuracy']*100:>9.2f}% {metrics['count']:>10,}")
        
        print(f"\n⏱️  Evaluation Time: {result.eval_time_seconds:.1f} seconds")
        print(f"📊 Total Samples: {result.total_samples:,}")
    
    def save_results(self, result: BenchmarkResult, output_dir: str = './benchmark_results'):
        """Save benchmark results."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"raid_benchmark_{result.model_name}_{timestamp}.json"
        
        results_dict = {
            'model_name': result.model_name,
            'timestamp': timestamp,
            'overall': {
                'accuracy': result.overall_accuracy,
                'high_conf_accuracy': result.high_conf_accuracy,
                'precision': result.overall_precision,
                'recall': result.overall_recall,
                'f1': result.overall_f1,
                'auc': result.overall_auc,
                'false_positive_rate': result.false_positive_rate,
                'false_negative_rate': result.false_negative_rate,
            },
            'by_attack': result.by_attack,
            'by_domain': result.by_domain,
            'by_ai_model': result.by_ai_model,
            'total_samples': result.total_samples,
            'eval_time_seconds': result.eval_time_seconds,
        }
        
        with open(output_path / filename, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path / filename}")
    
    def run(self, model_name: str = 'all'):
        """Run the benchmark."""
        print(BANNER)
        
        # Load data
        texts, labels, attacks, domains, ai_models = self.load_raid_data()
        
        results = []
        
        if model_name in ['supernova', 'all']:
            try:
                model, scaler = self.loader.load_supernova()
                result = self.benchmark_model(
                    model, scaler, 'SUPERNOVA',
                    texts, labels, attacks, domains, ai_models
                )
                results.append(result)
                self.print_results(result)
                self.save_results(result)
            except Exception as e:
                print(f"⚠️ SUPERNOVA benchmark failed: {e}")
        
        if model_name in ['flare', 'all']:
            try:
                model, scaler = self.loader.load_flare()
                result = self.benchmark_model(
                    model, scaler, 'Flare',
                    texts, labels, attacks, domains, ai_models
                )
                results.append(result)
                self.print_results(result)
                self.save_results(result)
            except Exception as e:
                print(f"⚠️ Flare benchmark failed: {e}")
        
        # Summary comparison
        if len(results) > 1:
            print(f"\n{'='*70}")
            print("MODEL COMPARISON SUMMARY")
            print('='*70)
            print(f"\n  {'Model':<20} {'Accuracy':>12} {'High-Conf':>12} {'F1':>12}")
            print(f"  {'-'*56}")
            for r in sorted(results, key=lambda x: -x.overall_accuracy):
                print(f"  {r.model_name:<20} {r.overall_accuracy*100:>11.2f}% {r.high_conf_accuracy*100:>11.2f}% {r.overall_f1*100:>11.2f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['supernova', 'flare', 'all'], default='all')
    parser.add_argument('--max-samples', type=int, default=100000)
    args = parser.parse_args()
    
    benchmark = RAIDBenchmark(max_samples=args.max_samples)
    benchmark.run(model_name=args.model)


if __name__ == '__main__':
    main()
