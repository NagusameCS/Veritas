#!/usr/bin/env python3
"""
Veritas Sunrise Training - One Command Training Script
Run with: python train_sunrise.py

This script performs massive-scale ML training using ALL available
HuggingFace AI detection datasets to derive optimal parameters.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from massive_dataset_loader import MassiveDatasetLoader, MASSIVE_DATASET_REGISTRY
from sunrise_trainer import SunriseTrainer

BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║   ███████╗██╗   ██╗███╗   ██╗██████╗ ██╗███████╗███████╗             ║
║   ██╔════╝██║   ██║████╗  ██║██╔══██╗██║██╔════╝██╔════╝             ║
║   ███████╗██║   ██║██╔██╗ ██║██████╔╝██║███████╗█████╗               ║
║   ╚════██║██║   ██║██║╚██╗██║██╔══██╗██║╚════██║██╔══╝               ║
║   ███████║╚██████╔╝██║ ╚████║██║  ██║██║███████║███████╗             ║
║   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝             ║
║                                                                       ║
║              Veritas ML Training Pipeline v2.0                        ║
║              Massive-Scale AI Detection Training                      ║
║                                                                       ║
║   Training on ALL available datasets for optimal parameters           ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

def parse_args():
    parser = argparse.ArgumentParser(
        description='Veritas Sunrise Training - Massive ML Training Pipeline'
    )
    parser.add_argument(
        '--max-samples-per-dataset', type=int, default=50000,
        help='Maximum samples to load per dataset (default: 50000)'
    )
    parser.add_argument(
        '--min-priority', type=int, default=2,
        help='Minimum dataset priority (1=highest, 3=lowest, default: 2)'
    )
    parser.add_argument(
        '--trials', type=int, default=100,
        help='Number of Optuna optimization trials (default: 100)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick training mode (fewer samples and trials)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='./models',
        help='Output directory for models (default: ./models)'
    )
    parser.add_argument(
        '--name', type=str, default='Sunrise',
        help='Model name (default: Sunrise)'
    )
    return parser.parse_args()


def main():
    print(BANNER)
    
    args = parse_args()
    
    # Adjust for quick mode
    if args.quick:
        args.max_samples_per_dataset = 5000
        args.trials = 20
        args.min_priority = 1
    
    print(f"Configuration:")
    print(f"  - Max samples per dataset: {args.max_samples_per_dataset:,}")
    print(f"  - Min priority level: {args.min_priority}")
    print(f"  - Optimization trials: {args.trials}")
    print(f"  - Model name: {args.name}")
    print(f"  - Output directory: {args.output_dir}")
    print(f"\n  Total datasets in registry: {len(MASSIVE_DATASET_REGISTRY)}")
    
    # =========================================================================
    # STEP 1: LOAD ALL DATASETS
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 1: LOADING ALL DATASETS")
    print(f"{'='*70}")
    
    loader = MassiveDatasetLoader()
    samples = loader.load_all_datasets(
        max_samples_per_dataset=args.max_samples_per_dataset,
        min_priority=args.min_priority
    )
    
    # Get loading report
    loading_report = loader.get_loading_report()
    
    print(f"\n{'='*70}")
    print("DATASET LOADING SUMMARY")
    print(f"{'='*70}")
    print(f"  Datasets attempted: {loading_report['total_datasets_attempted']}")
    print(f"  Datasets successful: {loading_report['successful_datasets']}")
    print(f"  Datasets failed: {loading_report['failed_datasets']}")
    print(f"  Total samples: {loading_report['total_samples_loaded']:,}")
    print(f"  Human samples: {loading_report['human_samples']:,}")
    print(f"  AI samples: {loading_report['ai_samples']:,}")
    
    if not samples:
        print("\nERROR: No samples loaded. Check dataset availability.")
        sys.exit(1)
    
    # Balance dataset
    print(f"\n{'='*70}")
    print("STEP 2: BALANCING DATASET")
    print(f"{'='*70}")
    
    balanced_samples = loader.balance_dataset(samples)
    
    # =========================================================================
    # STEP 3: TRAIN MODEL
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 3: TRAINING MODEL")
    print(f"{'='*70}")
    
    trainer = SunriseTrainer(output_dir=args.output_dir)
    
    dataset_info = {
        'datasets_used': loading_report['datasets_used'],
        'loading_report': loading_report
    }
    
    receipt = trainer.train(
        samples=balanced_samples,
        dataset_info=dataset_info,
        model_name=args.name,
        optimize_hyperparams=True,
        n_trials=args.trials
    )
    
    # =========================================================================
    # STEP 4: SAVE MODEL AND RECEIPTS
    # =========================================================================
    print(f"\n{'='*70}")
    print("STEP 4: SAVING MODEL AND TRAINING RECEIPTS")
    print(f"{'='*70}")
    
    model_dir = trainer.save_model(args.name)
    
    # Save comprehensive loading report
    with open(model_dir / 'dataset_loading_report.json', 'w') as f:
        json.dump(loading_report, f, indent=2, default=str)
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"""
Model: {args.name}
Location: {model_dir}

TRAINING RESULTS:
  - Test Accuracy:  {receipt.test_accuracy:.4f}
  - Test Precision: {receipt.test_precision:.4f}
  - Test Recall:    {receipt.test_recall:.4f}
  - Test F1 Score:  {receipt.test_f1:.4f}
  - ROC AUC:        {receipt.test_roc_auc:.4f}

DATASET COVERAGE:
  - Datasets used: {len(receipt.datasets_used)}
  - Total samples: {receipt.total_samples:,}
  - Human samples: {receipt.human_samples:,}
  - AI samples:    {receipt.ai_samples:,}

SAVED FILES:
  - model.pkl              (trained ML model)
  - scaler.pkl             (feature scaler)
  - training_receipt.json  (complete training proof)
  - metadata.json          (model metadata)
  - veritas_ml_config.js   (JavaScript config)
  - veritas_ml_config.json (JSON config for web)
  - dataset_loading_report.json (dataset details)

VERIFICATION HASHES:
  - Data Hash:  {receipt.data_hash}
  - Model Hash: {receipt.model_hash}
""")
    
    print(f"\n{'='*70}")
    print("TOP 10 FEATURES BY IMPORTANCE:")
    print(f"{'='*70}")
    for i, (name, importance) in enumerate(receipt.top_features[:10], 1):
        bar = '█' * int(importance * 50)
        print(f"  {i:2}. {name:30} {importance:.4f} {bar}")
    
    print(f"\n✓ All training receipts saved to: {model_dir / 'training_receipt.json'}")
    print(f"✓ JavaScript config saved to: {model_dir / 'veritas_ml_config.js'}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
