#!/usr/bin/env python3
"""
Flare v2 Production Trainer
===========================

Trains an ML model specifically for HUMAN vs HUMANIZED classification.
This model does NOT detect raw AI - it only determines if human-looking text
is genuinely human or AI that's been paraphrased/obfuscated.

Goal: 99%+ accuracy
Use case: Secondary layer after SUPERNOVA when text is classified as "human"

Datasets:
- RAID paraphrase attacks (~470k humanized samples)
- RAID human baseline (~48k human samples)
- GPT-wiki-intro human texts (~150k human samples)
- Additional human sources for balance

Architecture: XGBoost + Sentence Embeddings (same as SUPERNOVA)
"""

import os
import sys
import json
import random
import logging
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flare_v2_production.log')
    ]
)
logger = logging.getLogger(__name__)

# Check dependencies
try:
    from datasets import load_dataset
    from tqdm import tqdm
    import xgboost as xgb
    from sklearn.model_selection import train_test_split, StratifiedKFold
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sentence_transformers import SentenceTransformer
    import joblib
except ImportError as e:
    logger.error(f"Missing dependency: {e}")
    logger.error("Install: pip install datasets tqdm xgboost scikit-learn sentence-transformers joblib")
    sys.exit(1)


# ============================================================================
# Feature Extraction (Humanization-specific)
# ============================================================================

class FlareV2FeatureExtractor:
    """
    Extract features specifically designed to detect humanized AI text.
    
    Key signals:
    - Paraphrase artifacts (synonyms, restructuring)
    - Inconsistent style within text
    - Unnatural word choices
    - Modified sentence patterns
    """
    
    # Common AI phrases that might survive humanization
    AI_RESIDUE_PHRASES = [
        'it is important to note', 'it should be noted', 'in conclusion',
        'furthermore', 'moreover', 'additionally', 'consequently',
        'in summary', 'to summarize', 'in essence', 'fundamentally',
        'it is worth mentioning', 'it is essential', 'it is crucial',
        'comprehensive', 'robust', 'leverage', 'utilize', 'facilitate'
    ]
    
    # Human informal markers
    HUMAN_MARKERS = [
        "i'm", "i've", "i'll", "i'd", "can't", "won't", "don't", "didn't",
        "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't", "gonna", "wanna", "gotta",
        "kinda", "sorta", "y'all", "ain't", "lemme", "gimme", "dunno",
        "tbh", "imo", "imho", "lol", "lmao", "omg", "wtf", "btw", "fyi"
    ]
    
    # Paraphrasing tool artifacts
    PARAPHRASE_ARTIFACTS = [
        'in other words', 'put differently', 'to put it another way',
        'that is to say', 'namely', 'specifically', 'particularly',
        'in particular', 'especially', 'notably'
    ]
    
    # Thesaurus-swapped words (uncommon synonyms)
    THESAURUS_WORDS = [
        'utilize', 'commence', 'terminate', 'endeavor', 'ascertain',
        'facilitate', 'implement', 'subsequent', 'prior', 'regarding',
        'pertaining', 'aforementioned', 'henceforth', 'thereby', 'whereby',
        'heretofore', 'notwithstanding', 'inasmuch', 'insofar', 'whilst'
    ]
    
    def __init__(self, embedding_model: Optional[SentenceTransformer] = None):
        self.embedding_model = embedding_model
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """Extract all humanization detection features."""
        if not text or len(text.strip()) < 10:
            return self._empty_features()
        
        text_lower = text.lower()
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        features = {}
        
        # === Basic Statistics ===
        features['char_count'] = len(text)
        features['word_count'] = len(words)
        features['sent_count'] = len(sentences)
        features['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
        features['avg_sent_len'] = np.mean([len(self._tokenize(s)) for s in sentences]) if sentences else 0
        
        # === Sentence Variation (key humanization signal) ===
        sent_lengths = [len(self._tokenize(s)) for s in sentences]
        if len(sent_lengths) > 1:
            features['sent_len_std'] = np.std(sent_lengths)
            features['sent_len_cv'] = np.std(sent_lengths) / np.mean(sent_lengths) if np.mean(sent_lengths) > 0 else 0
            features['sent_len_range'] = max(sent_lengths) - min(sent_lengths)
        else:
            features['sent_len_std'] = 0
            features['sent_len_cv'] = 0
            features['sent_len_range'] = 0
        
        # === AI Residue Detection ===
        features['ai_residue_count'] = sum(1 for phrase in self.AI_RESIDUE_PHRASES if phrase in text_lower)
        features['ai_residue_density'] = features['ai_residue_count'] / max(len(sentences), 1)
        
        # === Humanization Artifact Detection ===
        features['paraphrase_artifacts'] = sum(1 for phrase in self.PARAPHRASE_ARTIFACTS if phrase in text_lower)
        features['thesaurus_words'] = sum(1 for word in self.THESAURUS_WORDS if word in text_lower)
        features['thesaurus_density'] = features['thesaurus_words'] / max(len(words), 1) * 100
        
        # === Genuine Human Markers ===
        features['human_markers'] = sum(1 for marker in self.HUMAN_MARKERS if marker in text_lower)
        features['human_marker_density'] = features['human_markers'] / max(len(words), 1) * 100
        
        # === Contraction Analysis ===
        contraction_pattern = r"\b\w+[''](?:t|s|re|ve|ll|d|m)\b"
        contractions = re.findall(contraction_pattern, text_lower)
        features['contraction_count'] = len(contractions)
        features['contraction_rate'] = len(contractions) / max(len(words), 1)
        
        # === Punctuation Patterns ===
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['ellipsis_count'] = text.count('...')
        features['dash_count'] = text.count('—') + text.count('–') + text.count(' - ')
        features['parenthetical_count'] = text.count('(')
        
        # === First Person Usage ===
        first_person = re.findall(r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', text_lower)
        features['first_person_count'] = len(first_person)
        features['first_person_rate'] = len(first_person) / max(len(words), 1)
        
        # === Style Consistency (humanized text often has inconsistent style) ===
        features['style_inconsistency'] = self._compute_style_inconsistency(sentences)
        
        # === Vocabulary Analysis ===
        unique_words = set(words)
        features['vocab_richness'] = len(unique_words) / max(len(words), 1)
        features['hapax_ratio'] = sum(1 for w in unique_words if words.count(w) == 1) / max(len(unique_words), 1)
        
        # === N-gram Repetition ===
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        features['bigram_repetition'] = 1 - (len(set(bigrams)) / max(len(bigrams), 1))
        features['trigram_repetition'] = 1 - (len(set(trigrams)) / max(len(trigrams), 1))
        
        # === Readability Proxy ===
        features['complex_word_ratio'] = sum(1 for w in words if len(w) > 10) / max(len(words), 1)
        
        # === Transition Words (often added by humanizers) ===
        transitions = ['however', 'therefore', 'meanwhile', 'nevertheless', 'furthermore',
                       'consequently', 'accordingly', 'hence', 'thus', 'besides']
        features['transition_count'] = sum(1 for t in transitions if t in text_lower)
        features['transition_density'] = features['transition_count'] / max(len(sentences), 1)
        
        # === Sentence Starters Diversity ===
        if sentences:
            starters = [self._tokenize(s)[0].lower() if self._tokenize(s) else '' for s in sentences]
            features['starter_diversity'] = len(set(starters)) / max(len(starters), 1)
        else:
            features['starter_diversity'] = 0
        
        # === Paragraph Structure ===
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        features['paragraph_count'] = len(paragraphs)
        if len(paragraphs) > 1:
            para_lengths = [len(self._tokenize(p)) for p in paragraphs]
            features['para_len_std'] = np.std(para_lengths)
        else:
            features['para_len_std'] = 0
        
        return features
    
    def _compute_style_inconsistency(self, sentences: List[str]) -> float:
        """Detect style inconsistency (common in humanized text)."""
        if len(sentences) < 3:
            return 0
        
        # Measure variation in formality across sentences
        formality_scores = []
        for sent in sentences:
            words = self._tokenize(sent.lower())
            if not words:
                continue
            
            # Simple formality heuristics
            formal_markers = sum(1 for w in words if w in ['therefore', 'consequently', 'furthermore', 'moreover', 'thus'])
            informal_markers = sum(1 for w in words if w in ["i'm", "don't", "can't", "won't", "it's", "that's"])
            contractions = len(re.findall(r"\w+[''](?:t|s|re|ve|ll|d|m)\b", sent.lower()))
            
            score = formal_markers - informal_markers - contractions * 0.5
            formality_scores.append(score)
        
        if len(formality_scores) > 1:
            return np.std(formality_scores)
        return 0
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization."""
        return re.findall(r'\b\w+\b', text.lower())
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty feature dict."""
        return {name: 0.0 for name in self.get_feature_names()}
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return [
            'char_count', 'word_count', 'sent_count', 'avg_word_len', 'avg_sent_len',
            'sent_len_std', 'sent_len_cv', 'sent_len_range',
            'ai_residue_count', 'ai_residue_density',
            'paraphrase_artifacts', 'thesaurus_words', 'thesaurus_density',
            'human_markers', 'human_marker_density',
            'contraction_count', 'contraction_rate',
            'exclamation_count', 'question_count', 'ellipsis_count', 'dash_count', 'parenthetical_count',
            'first_person_count', 'first_person_rate',
            'style_inconsistency', 'vocab_richness', 'hapax_ratio',
            'bigram_repetition', 'trigram_repetition',
            'complex_word_ratio', 'transition_count', 'transition_density',
            'starter_diversity', 'paragraph_count', 'para_len_std'
        ]


# ============================================================================
# Dataset Loading
# ============================================================================

def load_humanized_dataset(max_samples: int = 150000) -> Tuple[List[str], List[int]]:
    """
    Load balanced dataset for human vs humanized classification.
    
    Label mapping:
    - 0: Human (genuine human text)
    - 1: Humanized (AI text that's been paraphrased/obfuscated)
    
    Returns:
        texts: List of text samples
        labels: List of binary labels (0=human, 1=humanized)
    """
    logger.info("Loading humanized text datasets...")
    
    texts = []
    labels = []
    
    samples_per_class = max_samples // 2
    
    # === Load Humanized Text (RAID paraphrase attacks) ===
    logger.info(f"Loading RAID paraphrase attacks (target: {samples_per_class})...")
    try:
        raid = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        humanized_count = 0
        for item in tqdm(raid, desc="RAID paraphrased", total=samples_per_class):
            if humanized_count >= samples_per_class:
                break
            
            # Only get paraphrase attacks (humanized AI)
            if item.get('attack') == 'paraphrase' and item.get('model') != 'human':
                text = item.get('generation', '').strip()
                if text and 50 < len(text) < 10000:
                    texts.append(text)
                    labels.append(1)  # Humanized
                    humanized_count += 1
        
        logger.info(f"Loaded {humanized_count} humanized samples from RAID")
        
    except Exception as e:
        logger.error(f"Error loading RAID paraphrase: {e}")
        raise
    
    # === Load Human Text ===
    logger.info(f"Loading human text samples (target: {samples_per_class})...")
    
    human_count = 0
    
    # 1. RAID human text
    try:
        raid = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        for item in tqdm(raid, desc="RAID human", total=50000):
            if human_count >= samples_per_class:
                break
            
            if item.get('model') == 'human':
                text = item.get('generation', '').strip()
                if text and 50 < len(text) < 10000:
                    texts.append(text)
                    labels.append(0)  # Human
                    human_count += 1
        
        logger.info(f"Loaded {human_count} human samples from RAID")
        
    except Exception as e:
        logger.warning(f"Error loading RAID human: {e}")
    
    # 2. GPT-wiki-intro human texts
    if human_count < samples_per_class:
        remaining = samples_per_class - human_count
        logger.info(f"Loading GPT-wiki-intro human texts (need {remaining} more)...")
        
        try:
            wiki = load_dataset('aadityaubhat/GPT-wiki-intro', split='train', streaming=True)
            
            wiki_count = 0
            for item in tqdm(wiki, desc="Wiki human", total=remaining):
                if human_count >= samples_per_class:
                    break
                
                text = item.get('wiki_intro', '').strip()
                if text and 50 < len(text) < 10000:
                    texts.append(text)
                    labels.append(0)  # Human
                    human_count += 1
                    wiki_count += 1
            
            logger.info(f"Loaded {wiki_count} human samples from GPT-wiki-intro")
            
        except Exception as e:
            logger.warning(f"Error loading GPT-wiki-intro: {e}")
    
    # 3. Writing prompts for more diverse human text
    if human_count < samples_per_class:
        remaining = samples_per_class - human_count
        logger.info(f"Loading writing prompts (need {remaining} more)...")
        
        try:
            prompts = load_dataset('euclaise/writingprompts', split='train', streaming=True)
            
            wp_count = 0
            for item in tqdm(prompts, desc="Writing prompts", total=remaining):
                if human_count >= samples_per_class:
                    break
                
                text = item.get('story', item.get('text', '')).strip()
                if text and 50 < len(text) < 10000:
                    texts.append(text)
                    labels.append(0)  # Human
                    human_count += 1
                    wp_count += 1
            
            logger.info(f"Loaded {wp_count} human samples from writing prompts")
            
        except Exception as e:
            logger.warning(f"Error loading writing prompts: {e}")
    
    logger.info(f"Total dataset: {len(texts)} samples")
    logger.info(f"  Human: {labels.count(0)}")
    logger.info(f"  Humanized: {labels.count(1)}")
    
    return texts, labels


# ============================================================================
# Training
# ============================================================================

def train_flare_v2(
    max_samples: int = 200000,
    use_embeddings: bool = True,
    xgb_params: Optional[Dict] = None
) -> Tuple[xgb.XGBClassifier, Dict]:
    """
    Train Flare v2 model for human vs humanized detection.
    
    Args:
        max_samples: Maximum training samples
        use_embeddings: Whether to include sentence embeddings
        xgb_params: XGBoost hyperparameters
    
    Returns:
        model: Trained XGBoost model
        metrics: Evaluation metrics
    """
    
    # Load data
    texts, labels = load_humanized_dataset(max_samples)
    
    # Initialize feature extractor
    embedding_model = None
    if use_embeddings:
        logger.info("Loading sentence embedding model...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    extractor = FlareV2FeatureExtractor(embedding_model)
    
    # Extract features
    logger.info("Extracting features...")
    features_list = []
    embeddings_list = []
    
    for text in tqdm(texts, desc="Extracting features"):
        # Heuristic features
        feat = extractor.extract_features(text)
        features_list.append([feat[name] for name in extractor.get_feature_names()])
        
        # Embeddings
        if use_embeddings and embedding_model:
            emb = embedding_model.encode(text[:1000], show_progress_bar=False)
            embeddings_list.append(emb)
    
    X_heuristic = np.array(features_list, dtype=np.float32)
    
    if use_embeddings and embeddings_list:
        X_embeddings = np.array(embeddings_list, dtype=np.float32)
        X = np.concatenate([X_heuristic, X_embeddings], axis=1)
        logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} features ({X_heuristic.shape[1]} heuristic + {X_embeddings.shape[1]} embedding)")
    else:
        X = X_heuristic
        logger.info(f"Feature matrix: {X.shape[0]} samples × {X.shape[1]} heuristic features")
    
    y = np.array(labels)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # XGBoost parameters
    if xgb_params is None:
        xgb_params = {
            'n_estimators': 500,
            'max_depth': 12,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'early_stopping_rounds': 30
        }
    
    # Train
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(**xgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50
    )
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': X.shape[1],
        'heuristic_features': len(extractor.get_feature_names()),
        'embedding_dim': 384 if use_embeddings else 0
    }
    
    logger.info("\n" + "="*60)
    logger.info("FLARE V2 EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info(f"ROC AUC:   {roc_auc:.4f}")
    logger.info("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Human', 'Humanized']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"                 Predicted")
    print(f"              Human  Humanized")
    print(f"Actual Human    {cm[0,0]:5d}     {cm[0,1]:5d}")
    print(f"     Humanized  {cm[1,0]:5d}     {cm[1,1]:5d}")
    
    return model, metrics, extractor


def convert_to_onnx(model: xgb.XGBClassifier, output_path: str, feature_count: int):
    """Convert XGBoost model to ONNX format for browser deployment."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        
        logger.info("Converting to ONNX...")
        
        initial_type = [('float_input', FloatTensorType([None, feature_count]))]
        
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=12
        )
        
        onnx.save_model(onnx_model, output_path)
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"ONNX model saved: {output_path} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"ONNX conversion failed: {e}")
        return False


def main():
    """Main training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Flare v2 model')
    parser.add_argument('--samples', type=int, default=200000, help='Max training samples')
    parser.add_argument('--no-embeddings', action='store_true', help='Disable embeddings')
    parser.add_argument('--output-dir', type=str, default='models/FlareV2', help='Output directory')
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("FLARE V2 PRODUCTION TRAINING")
    logger.info("Binary classification: Human vs Humanized")
    logger.info("="*60)
    
    start_time = datetime.now()
    
    # Train
    model, metrics, extractor = train_flare_v2(
        max_samples=args.samples,
        use_embeddings=not args.no_embeddings
    )
    
    # Check if we meet the 99% threshold
    if metrics['accuracy'] >= 0.99:
        logger.info("\n✅ TARGET ACCURACY REACHED (99%+)")
        
        # Save model
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        model_path = output_dir / 'flare_v2.json'
        model.save_model(str(model_path))
        logger.info(f"Saved XGBoost model: {model_path}")
        
        # Convert to ONNX
        onnx_path = output_dir / 'flare_v2.onnx'
        convert_to_onnx(model, str(onnx_path), metrics['total_features'])
        
        # Save metadata
        metadata = {
            'model_name': 'Flare V2',
            'version': '2.0.0',
            'type': 'human_vs_humanized_detector',
            'description': 'Detects if text is genuinely human or AI-generated that has been humanized/paraphrased',
            'created': datetime.now().isoformat(),
            'training_time': str(datetime.now() - start_time),
            'metrics': metrics,
            'feature_names': extractor.get_feature_names(),
            'labels': {
                '0': 'human',
                '1': 'humanized'
            },
            'threshold': 0.5,
            'accuracy_target_met': True
        }
        
        metadata_path = output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_path}")
        
        # Generate JS config for Veritas
        js_config = f"""// Flare V2 Configuration
// Auto-generated by train_flare_v2_production.py

const FLARE_V2_CONFIG = {{
    modelName: 'Flare V2',
    version: '2.0.0',
    type: 'human_vs_humanized',
    accuracy: {metrics['accuracy']:.4f},
    threshold: 0.5,
    featureCount: {metrics['total_features']},
    heuristicFeatures: {metrics['heuristic_features']},
    embeddingDim: {metrics['embedding_dim']},
    labels: {{
        0: 'human',
        1: 'humanized'
    }},
    featureNames: {json.dumps(extractor.get_feature_names())}
}};

if (typeof module !== 'undefined') {{
    module.exports = FLARE_V2_CONFIG;
}}
"""
        
        js_path = output_dir / 'flare_v2_config.js'
        with open(js_path, 'w') as f:
            f.write(js_config)
        logger.info(f"Saved JS config: {js_path}")
        
        logger.info("\n" + "="*60)
        logger.info("FLARE V2 TRAINING COMPLETE")
        logger.info(f"Accuracy: {metrics['accuracy']*100:.2f}% ✅")
        logger.info(f"Models saved to: {output_dir}")
        logger.info("="*60)
        
    else:
        logger.warning(f"\n❌ TARGET NOT MET: {metrics['accuracy']*100:.2f}% < 99%")
        logger.warning("Model not saved. Consider:")
        logger.warning("  - Increasing training samples")
        logger.warning("  - Tuning hyperparameters")
        logger.warning("  - Adding more features")
        
        # Save debug info anyway
        output_dir = Path(args.output_dir + '_debug')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        debug_info = {
            'metrics': metrics,
            'training_time': str(datetime.now() - start_time),
            'reason': f"Accuracy {metrics['accuracy']*100:.2f}% below 99% threshold"
        }
        
        with open(output_dir / 'debug_info.json', 'w') as f:
            json.dump(debug_info, f, indent=2)
    
    return metrics


if __name__ == '__main__':
    main()
