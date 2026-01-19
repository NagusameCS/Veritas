#!/usr/bin/env python3
"""
Veritas Training Bot
Main entry point for training the AI detection model.

Usage:
    python train.py                    # Full training with defaults
    python train.py --quick            # Quick training (fewer samples)
    python train.py --samples 10000    # Specify sample count
    python train.py --model gb         # Use gradient boosting
    python train.py --evaluate-only    # Just evaluate existing model
"""

import os
import sys
import argparse
from pathlib import Path

# Add training directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def install_dependencies():
    """Install required dependencies."""
    import subprocess
    
    requirements_path = Path(__file__).parent / 'requirements.txt'
    
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '-q', '-r', str(requirements_path)
    ])
    print("Dependencies installed!\n")


def check_dependencies():
    """Check if dependencies are installed."""
    required = ['numpy', 'sklearn', 'datasets', 'tqdm']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    return missing


def main():
    parser = argparse.ArgumentParser(
        description='Veritas AI Detection Training Bot',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                     # Full training
  python train.py --quick             # Quick test run (500 samples)
  python train.py --samples 20000     # Train with 20k samples
  python train.py --model gb          # Use Gradient Boosting
  python train.py --no-optimize       # Skip hyperparameter tuning
  python train.py --install           # Install dependencies only
        """
    )
    
    parser.add_argument('--install', action='store_true',
                        help='Install dependencies and exit')
    parser.add_argument('--quick', action='store_true',
                        help='Quick training with minimal samples (500)')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Maximum samples per dataset (default: 5000)')
    parser.add_argument('--model', type=str, default='rf',
                        choices=['rf', 'gb', 'lr', 'random_forest', 
                                'gradient_boosting', 'logistic_regression'],
                        help='Model type (rf=Random Forest, gb=Gradient Boosting, lr=Logistic Regression)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization')
    parser.add_argument('--evaluate-only', type=str, metavar='MODEL_PATH',
                        help='Only evaluate an existing model')
    parser.add_argument('--output-dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--test-text', type=str,
                        help='Test the model with a specific text')
    
    args = parser.parse_args()
    
    # Handle install
    if args.install:
        install_dependencies()
        return 0
    
    # Check dependencies
    missing = check_dependencies()
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Run: python train.py --install")
        return 1
    
    # Now we can import our modules
    from trainer import VeritasTrainer, run_full_training
    from dataset_loader import load_default_datasets
    
    # Map model shortcuts
    model_map = {
        'rf': 'random_forest',
        'gb': 'gradient_boosting', 
        'lr': 'logistic_regression',
        'random_forest': 'random_forest',
        'gradient_boosting': 'gradient_boosting',
        'logistic_regression': 'logistic_regression'
    }
    model_type = model_map[args.model]
    
    # Quick mode
    if args.quick:
        args.samples = 500
        args.no_optimize = True
    
    print("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ██╗   ██╗███████╗██████╗ ██╗████████╗ █████╗ ███████╗      ║
║   ██║   ██║██╔════╝██╔══██╗██║╚══██╔══╝██╔══██╗██╔════╝      ║
║   ██║   ██║█████╗  ██████╔╝██║   ██║   ███████║███████╗      ║
║   ╚██╗ ██╔╝██╔══╝  ██╔══██╗██║   ██║   ██╔══██║╚════██║      ║
║    ╚████╔╝ ███████╗██║  ██║██║   ██║   ██║  ██║███████║      ║
║     ╚═══╝  ╚══════╝╚═╝  ╚═╝╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝      ║
║                                                               ║
║              ML Training Bot v1.0.0                           ║
║              Fine-tuning AI Detection                         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Handle evaluate-only mode
    if args.evaluate_only:
        print(f"Loading model from: {args.evaluate_only}")
        trainer = VeritasTrainer(output_dir=args.output_dir)
        trainer.load_model(args.evaluate_only)
        
        if args.test_text:
            result = trainer.predict(args.test_text)
            print(f"\nPrediction for text:")
            print(f"  AI Probability: {result['ai_probability']*100:.1f}%")
            print(f"  Prediction: {result['prediction']}")
            print(f"  Confidence: {result['confidence']*100:.1f}%")
        else:
            # Load test data and evaluate
            print("Loading test data...")
            loader = load_default_datasets(max_per_dataset=1000)
            _, test_samples = loader.get_train_test_split(test_ratio=0.3)
            trainer.evaluate(test_samples)
        
        return 0
    
    # Run full training
    print(f"Configuration:")
    print(f"  - Samples per dataset: {args.samples}")
    print(f"  - Model type: {model_type}")
    print(f"  - Hyperparameter optimization: {not args.no_optimize}")
    print(f"  - Output directory: {args.output_dir}")
    print()
    
    trainer = run_full_training(
        max_samples=args.samples * 2,  # Total samples = 2x per-dataset
        model_type=model_type,
        optimize=not args.no_optimize
    )
    
    if trainer and args.test_text:
        result = trainer.predict(args.test_text)
        print(f"\nTest prediction:")
        print(f"  Text: {args.test_text[:100]}...")
        print(f"  AI Probability: {result['ai_probability']*100:.1f}%")
        print(f"  Prediction: {result['prediction']}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
