"""
Veritas ML Trainer
Fine-tunes detection thresholds and weights using machine learning.
"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, GridSearchCV
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Run: pip install scikit-learn")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: optuna not installed for hyperparameter tuning")

from feature_extractor import FeatureExtractor
from dataset_loader import DatasetLoader, TextSample


class VeritasTrainer:
    """Train and optimize Veritas detection model."""
    
    def __init__(self, output_dir: str = './models'):
        """Initialize the trainer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_importance = {}
        self.training_stats = {}
        
    def extract_features_batch(self, samples: List[TextSample], 
                                show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from a batch of samples."""
        features_list = []
        labels = []
        
        iterator = tqdm(samples, desc="Extracting features") if show_progress else samples
        
        for sample in iterator:
            try:
                features = self.extractor.extract_feature_vector(sample.text)
                features_list.append(features)
                labels.append(sample.label)
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
        
        X = np.array(features_list)
        y = np.array(labels)
        
        # Handle NaN/Inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X, y
    
    def train(self, train_samples: List[TextSample],
              model_type: str = 'random_forest',
              optimize_hyperparams: bool = True) -> Dict:
        """Train the detection model."""
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required for training")
        
        print(f"\n{'='*60}")
        print(f"Training Veritas Model")
        print(f"{'='*60}")
        print(f"Samples: {len(train_samples)}")
        print(f"Model type: {model_type}")
        print(f"{'='*60}\n")
        
        # Extract features
        X_train, y_train = self.extract_features_batch(train_samples)
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Class distribution: {np.bincount(y_train)}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Select and train model
        if optimize_hyperparams and OPTUNA_AVAILABLE:
            self.model = self._optimize_hyperparameters(
                X_train_scaled, y_train, model_type
            )
        else:
            self.model = self._get_default_model(model_type)
            self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        # Feature importance
        self._calculate_feature_importance()
        
        # Training stats
        self.training_stats = {
            'model_type': model_type,
            'n_samples': len(train_samples),
            'n_features': X_train.shape[1],
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std': float(cv_scores.std()),
            'timestamp': datetime.now().isoformat(),
        }
        
        print(f"\nCross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        return self.training_stats
    
    def _get_default_model(self, model_type: str):
        """Get a model with default parameters."""
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                probability=True,
                random_state=42
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return models[model_type]
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                                   model_type: str, n_trials: int = 50) -> Any:
        """Optimize hyperparameters using Optuna."""
        print(f"\nOptimizing hyperparameters with {n_trials} trials...")
        
        def objective(trial):
            if model_type == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'random_state': 42,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
                
            elif model_type == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'max_iter': 1000,
                    'random_state': 42
                }
                model = LogisticRegression(**params)
            else:
                raise ValueError(f"Optimization not supported for {model_type}")
            
            scores = cross_val_score(model, X, y, cv=5, scoring='f1')
            return scores.mean()
        
        sampler = TPESampler(seed=42)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        print(f"Best F1 score: {study.best_value:.4f}")
        print(f"Best parameters: {study.best_params}")
        
        # Train final model with best params
        best_params = study.best_params
        best_params['random_state'] = 42
        
        if model_type == 'random_forest':
            best_params['n_jobs'] = -1
            model = RandomForestClassifier(**best_params)
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(**best_params)
        elif model_type == 'logistic_regression':
            best_params['max_iter'] = 1000
            model = LogisticRegression(**best_params)
        
        model.fit(X, y)
        return model
    
    def _calculate_feature_importance(self):
        """Calculate and store feature importance."""
        feature_names = self.extractor.get_feature_names()
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            importances = np.zeros(len(feature_names))
        
        self.feature_importance = {
            name: float(imp)
            for name, imp in zip(feature_names, importances)
        }
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), 
                   key=lambda x: x[1], reverse=True)
        )
    
    def evaluate(self, test_samples: List[TextSample]) -> Dict:
        """Evaluate the model on test data."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        print("\nEvaluating model...")
        
        X_test, y_test = self.extract_features_batch(test_samples)
        X_test_scaled = self.scaler.transform(X_test)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred)),
            'recall': float(recall_score(y_test, y_pred)),
            'f1': float(f1_score(y_test, y_pred)),
            'roc_auc': float(roc_auc_score(y_test, y_prob)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        }
        
        print(f"\n{'='*40}")
        print("EVALUATION RESULTS")
        print(f"{'='*40}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['confusion_matrix'][0][0]}  FP: {metrics['confusion_matrix'][0][1]}")
        print(f"  FN: {metrics['confusion_matrix'][1][0]}  TP: {metrics['confusion_matrix'][1][1]}")
        print(f"{'='*40}\n")
        
        return metrics
    
    def predict(self, text: str) -> Dict:
        """Predict AI probability for a single text."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        features = self.extractor.extract_feature_vector(text)
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        prob = self.model.predict_proba(features_scaled)[0]
        
        return {
            'ai_probability': float(prob[1]),
            'human_probability': float(prob[0]),
            'prediction': 'AI' if prob[1] > 0.5 else 'Human',
            'confidence': float(max(prob))
        }
    
    def save_model(self, name: str = 'veritas_model'):
        """Save the trained model and configuration."""
        if self.model is None:
            raise RuntimeError("No model to save. Train first.")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = self.output_dir / f"{name}_{timestamp}"
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(model_path / 'model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(model_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save metadata
        metadata = {
            'feature_names': self.extractor.get_feature_names(),
            'feature_importance': self.feature_importance,
            'training_stats': self.training_stats,
            'timestamp': timestamp
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Export optimized thresholds for JavaScript
        self._export_js_config(model_path / 'veritas_config.js')
        
        print(f"Model saved to: {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str):
        """Load a trained model."""
        model_path = Path(model_path)
        
        with open(model_path / 'model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(model_path / 'scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(model_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.feature_importance = metadata.get('feature_importance', {})
        self.training_stats = metadata.get('training_stats', {})
        
        print(f"Model loaded from: {model_path}")
    
    def _export_js_config(self, filepath: Path):
        """Export optimized configuration for JavaScript analyzer."""
        # Get top features and their thresholds
        top_features = list(self.feature_importance.items())[:15]
        
        # Calculate optimal thresholds based on feature distributions
        config = {
            'version': '2.1.0',
            'generated': datetime.now().isoformat(),
            'feature_weights': {
                name: round(weight, 4) 
                for name, weight in top_features
            },
            'training_accuracy': self.training_stats.get('cv_accuracy_mean', 0),
        }
        
        js_content = f"""/**
 * Veritas ML-Optimized Configuration
 * Generated: {config['generated']}
 * Training Accuracy: {config['training_accuracy']:.4f}
 */

const VERITAS_ML_CONFIG = {{
    version: '{config['version']}',
    
    // ML-derived feature weights (higher = more important for AI detection)
    featureWeights: {{
{self._format_js_object(config['feature_weights'], indent=8)}
    }},
    
    // Use these weights to adjust the scoring in analyzer-engine.js
    // Higher weighted features should have more influence on final score
}};

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {{
    module.exports = VERITAS_ML_CONFIG;
}}
"""
        
        with open(filepath, 'w') as f:
            f.write(js_content)
        
        print(f"JavaScript config exported to: {filepath}")
    
    def _format_js_object(self, obj: Dict, indent: int = 0) -> str:
        """Format a dictionary as JavaScript object properties."""
        lines = []
        prefix = ' ' * indent
        for key, value in obj.items():
            # Convert Python key to valid JS identifier
            js_key = key.replace('-', '_')
            lines.append(f"{prefix}{js_key}: {value},")
        return '\n'.join(lines)
    
    def print_feature_importance(self, top_n: int = 20):
        """Print feature importance rankings."""
        print(f"\n{'='*50}")
        print("FEATURE IMPORTANCE (Top {})".format(top_n))
        print(f"{'='*50}")
        
        for i, (name, importance) in enumerate(list(self.feature_importance.items())[:top_n]):
            bar = '█' * int(importance * 50)
            print(f"{i+1:2}. {name:30} {importance:.4f} {bar}")
        
        print(f"{'='*50}\n")


def run_full_training(max_samples: int = 10000, 
                      model_type: str = 'random_forest',
                      optimize: bool = True):
    """Run the full training pipeline."""
    from dataset_loader import load_default_datasets
    
    print("\n" + "="*60)
    print("VERITAS ML TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # Load datasets
    print("Step 1: Loading datasets...")
    loader = load_default_datasets(max_per_dataset=max_samples // 2)
    
    if not loader.samples:
        print("ERROR: No samples loaded. Check dataset availability.")
        return None
    
    stats = loader.get_statistics()
    print(f"Loaded {stats['total_samples']} samples "
          f"({stats['human_samples']} human, {stats['ai_samples']} AI)")
    
    # Split data
    print("\nStep 2: Splitting data...")
    train_samples, test_samples = loader.get_train_test_split(test_ratio=0.2)
    print(f"Train: {len(train_samples)}, Test: {len(test_samples)}")
    
    # Train model
    print("\nStep 3: Training model...")
    trainer = VeritasTrainer()
    trainer.train(train_samples, model_type=model_type, optimize_hyperparams=optimize)
    
    # Evaluate
    print("\nStep 4: Evaluating model...")
    metrics = trainer.evaluate(test_samples)
    
    # Print feature importance
    trainer.print_feature_importance()
    
    # Save model
    print("\nStep 5: Saving model...")
    model_path = trainer.save_model()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved to: {model_path}")
    print(f"Final F1 Score: {metrics['f1']:.4f}")
    print("="*60 + "\n")
    
    return trainer


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Veritas AI detection model')
    parser.add_argument('--samples', type=int, default=5000,
                        help='Maximum samples per dataset')
    parser.add_argument('--model', type=str, default='random_forest',
                        choices=['random_forest', 'gradient_boosting', 'logistic_regression'],
                        help='Model type to train')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization')
    
    args = parser.parse_args()
    
    run_full_training(
        max_samples=args.samples,
        model_type=args.model,
        optimize=not args.no_optimize
    )
