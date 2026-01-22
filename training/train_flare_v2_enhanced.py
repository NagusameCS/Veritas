#!/usr/bin/env python3
"""
Flare v2 Enhanced Training
==========================

Enhanced version targeting 99%+ accuracy with:
- More training samples (400k+)
- Deeper XGBoost trees
- Additional humanization-specific features
- Better hyperparameter tuning
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
from collections import Counter
import re

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('flare_v2_enhanced.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from datasets import load_dataset
    from tqdm import tqdm
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score
    )
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.error(f"Missing: {e}")
    sys.exit(1)


class EnhancedFlareFeatureExtractor:
    """
    Enhanced feature extraction for humanized text detection.
    
    Additional features targeting paraphrase detection:
    - Semantic density patterns
    - Word frequency anomalies
    - Syntactic structure shifts
    """
    
    AI_RESIDUE = [
        'it is important to note', 'it should be noted', 'in conclusion',
        'furthermore', 'moreover', 'additionally', 'consequently',
        'in summary', 'to summarize', 'in essence', 'fundamentally',
        'it is worth mentioning', 'it is essential', 'it is crucial',
        'comprehensive', 'robust', 'leverage', 'utilize', 'facilitate',
        'delve', 'crucial', 'tapestry', 'multifaceted', 'intricacies',
        'landscape', 'navigate', 'nuanced', 'holistic', 'overarching'
    ]
    
    HUMAN_INFORMAL = [
        "i'm", "i've", "i'll", "i'd", "can't", "won't", "don't", "didn't",
        "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
        "weren't", "hasn't", "haven't", "hadn't", "gonna", "wanna", "gotta",
        "kinda", "sorta", "y'all", "ain't", "lemme", "gimme", "dunno",
        "tbh", "imo", "imho", "lol", "lmao", "omg", "wtf", "btw", "fyi",
        "haha", "hehe", "lmfao", "bruh", "bro", "dude", "yeah", "yep", "nope",
        "ugh", "meh", "idk", "rn", "ngl", "smh", "ikr", "ofc"
    ]
    
    PARAPHRASE_MARKERS = [
        'in other words', 'put differently', 'to put it another way',
        'that is to say', 'namely', 'specifically', 'particularly',
        'in particular', 'especially', 'notably', 'as mentioned',
        'as stated', 'as noted', 'put simply', 'simply put'
    ]
    
    THESAURUS_SWAPS = [
        'utilize', 'commence', 'terminate', 'endeavor', 'ascertain',
        'facilitate', 'implement', 'subsequent', 'prior', 'regarding',
        'pertaining', 'aforementioned', 'henceforth', 'thereby', 'whereby',
        'heretofore', 'notwithstanding', 'inasmuch', 'insofar', 'whilst',
        'amongst', 'towards', 'upon', 'thus', 'hence', 'furthermore',
        'moreover', 'nevertheless', 'nonetheless', 'whereas', 'whereby'
    ]
    
    # Words commonly changed by paraphrasers
    PARAPHRASE_SYNONYMS = {
        'big': ['large', 'substantial', 'significant', 'considerable'],
        'small': ['tiny', 'minor', 'diminutive', 'modest'],
        'good': ['excellent', 'beneficial', 'advantageous', 'favorable'],
        'bad': ['negative', 'detrimental', 'adverse', 'unfavorable'],
        'important': ['crucial', 'vital', 'essential', 'significant'],
        'show': ['demonstrate', 'illustrate', 'exhibit', 'display'],
        'use': ['utilize', 'employ', 'leverage', 'implement'],
        'make': ['create', 'produce', 'generate', 'construct'],
        'help': ['assist', 'aid', 'facilitate', 'support'],
        'get': ['obtain', 'acquire', 'receive', 'attain']
    }
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model
    
    def extract(self, text: str) -> Dict[str, float]:
        if not text or len(text.strip()) < 10:
            return self._empty()
        
        text_lower = text.lower()
        words = self._tokenize(text)
        sentences = self._split_sentences(text)
        
        f = {}
        
        # === Basic Stats ===
        f['chars'] = len(text)
        f['words'] = len(words)
        f['sents'] = len(sentences)
        f['avg_word_len'] = np.mean([len(w) for w in words]) if words else 0
        
        # === Sentence Analysis ===
        sent_lens = [len(self._tokenize(s)) for s in sentences] if sentences else [0]
        f['sent_mean'] = np.mean(sent_lens)
        f['sent_std'] = np.std(sent_lens) if len(sent_lens) > 1 else 0
        f['sent_cv'] = f['sent_std'] / f['sent_mean'] if f['sent_mean'] > 0 else 0
        f['sent_range'] = max(sent_lens) - min(sent_lens) if sent_lens else 0
        f['sent_skew'] = self._skewness(sent_lens)
        f['sent_kurt'] = self._kurtosis(sent_lens)
        
        # === AI Residue ===
        f['ai_residue'] = sum(1 for p in self.AI_RESIDUE if p in text_lower)
        f['ai_residue_pct'] = f['ai_residue'] / max(len(sentences), 1) * 100
        
        # === Paraphrase Markers ===
        f['paraphrase_markers'] = sum(1 for p in self.PARAPHRASE_MARKERS if p in text_lower)
        f['thesaurus_words'] = sum(1 for w in self.THESAURUS_SWAPS if w in text_lower)
        f['formal_synonyms'] = self._count_formal_synonyms(words)
        
        # === Human Markers ===
        f['human_informal'] = sum(1 for m in self.HUMAN_INFORMAL if m in text_lower)
        f['human_pct'] = f['human_informal'] / max(len(words), 1) * 100
        
        # === Contractions ===
        contractions = re.findall(r"\b\w+[''](?:t|s|re|ve|ll|d|m)\b", text_lower)
        f['contractions'] = len(contractions)
        f['contraction_rate'] = len(contractions) / max(len(words), 1)
        
        # === Punctuation ===
        f['exclamations'] = text.count('!')
        f['questions'] = text.count('?')
        f['ellipses'] = text.count('...')
        f['dashes'] = text.count('—') + text.count('–') + text.count(' - ')
        f['parens'] = text.count('(')
        f['commas'] = text.count(',')
        f['semicolons'] = text.count(';')
        f['colons'] = text.count(':')
        f['quotes'] = text.count('"') + text.count("'")
        
        # === First Person ===
        first_person = re.findall(r'\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b', text_lower)
        f['first_person'] = len(first_person)
        f['first_person_rate'] = len(first_person) / max(len(words), 1)
        
        # === Vocabulary Richness ===
        unique = set(words)
        f['vocab_richness'] = len(unique) / max(len(words), 1)
        f['hapax_ratio'] = sum(1 for w in unique if words.count(w) == 1) / max(len(unique), 1)
        word_freq = Counter(words)
        f['top_word_freq'] = word_freq.most_common(1)[0][1] / len(words) if words else 0
        
        # === N-gram Analysis ===
        bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        quadgrams = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
        
        f['bigram_rep'] = 1 - (len(set(bigrams)) / max(len(bigrams), 1))
        f['trigram_rep'] = 1 - (len(set(trigrams)) / max(len(trigrams), 1))
        f['quadgram_rep'] = 1 - (len(set(quadgrams)) / max(len(quadgrams), 1))
        
        # === Sentence Starters ===
        if sentences:
            starters = [self._tokenize(s)[0] if self._tokenize(s) else '' for s in sentences]
            f['starter_diversity'] = len(set(starters)) / max(len(starters), 1)
            # Check for repeated sentence patterns
            f['starter_rep'] = 1 - f['starter_diversity']
        else:
            f['starter_diversity'] = 0
            f['starter_rep'] = 0
        
        # === Paragraph Analysis ===
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        f['paragraphs'] = len(paras)
        if len(paras) > 1:
            para_lens = [len(self._tokenize(p)) for p in paras]
            f['para_std'] = np.std(para_lens)
            f['para_cv'] = np.std(para_lens) / np.mean(para_lens) if np.mean(para_lens) > 0 else 0
        else:
            f['para_std'] = 0
            f['para_cv'] = 0
        
        # === Style Consistency ===
        f['style_shift'] = self._style_consistency(sentences)
        
        # === Word Length Distribution ===
        word_lens = [len(w) for w in words]
        f['short_words'] = sum(1 for l in word_lens if l <= 3) / max(len(words), 1)
        f['medium_words'] = sum(1 for l in word_lens if 4 <= l <= 7) / max(len(words), 1)
        f['long_words'] = sum(1 for l in word_lens if l > 7) / max(len(words), 1)
        f['very_long_words'] = sum(1 for l in word_lens if l > 10) / max(len(words), 1)
        
        # === Transition Words ===
        transitions = ['however', 'therefore', 'meanwhile', 'nevertheless', 'furthermore',
                      'consequently', 'accordingly', 'hence', 'thus', 'besides', 'although',
                      'whereas', 'while', 'despite', 'since', 'because', 'unless', 'until']
        f['transitions'] = sum(1 for t in transitions if t in text_lower)
        f['transition_rate'] = f['transitions'] / max(len(sentences), 1)
        
        # === Passive Voice Proxy ===
        passive_patterns = re.findall(r'\b(?:is|are|was|were|been|being)\s+\w+ed\b', text_lower)
        f['passive_voice'] = len(passive_patterns)
        f['passive_rate'] = len(passive_patterns) / max(len(sentences), 1)
        
        # === Hedging Language ===
        hedges = ['perhaps', 'maybe', 'possibly', 'probably', 'might', 'could', 'may',
                  'somewhat', 'relatively', 'fairly', 'rather', 'quite', 'seem', 'appear',
                  'tend', 'suggest', 'indicate', 'likely', 'unlikely']
        f['hedges'] = sum(1 for h in hedges if h in text_lower)
        f['hedge_rate'] = f['hedges'] / max(len(sentences), 1)
        
        # === Emphatic Language ===
        emphatics = ['very', 'really', 'extremely', 'absolutely', 'definitely', 'certainly',
                     'clearly', 'obviously', 'undoubtedly', 'surely', 'totally', 'completely']
        f['emphatics'] = sum(1 for e in emphatics if e in text_lower)
        
        # === Number Usage ===
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        f['numbers'] = len(numbers)
        f['number_rate'] = len(numbers) / max(len(words), 1)
        
        # === Capitalization ===
        caps = sum(1 for c in text if c.isupper())
        f['caps_ratio'] = caps / max(len(text), 1)
        
        # === Average Information Content ===
        f['entropy'] = self._entropy(words)
        
        return f
    
    def _count_formal_synonyms(self, words: List[str]) -> int:
        """Count formal synonyms that might indicate thesaurus usage."""
        count = 0
        for word in words:
            for simple, formal_list in self.PARAPHRASE_SYNONYMS.items():
                if word in formal_list:
                    count += 1
        return count
    
    def _style_consistency(self, sentences: List[str]) -> float:
        """Measure style shifts between sentences."""
        if len(sentences) < 3:
            return 0
        
        scores = []
        for sent in sentences:
            words = self._tokenize(sent.lower())
            if not words:
                continue
            
            formal = sum(1 for w in words if w in self.THESAURUS_SWAPS)
            informal = sum(1 for w in words if w in self.HUMAN_INFORMAL)
            contractions = len(re.findall(r"\w+[''](?:t|s|re|ve|ll|d|m)\b", sent.lower()))
            
            score = formal - informal - contractions * 0.5
            scores.append(score)
        
        return np.std(scores) if len(scores) > 1 else 0
    
    def _skewness(self, data: List[float]) -> float:
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 3 for x in data]) / (std ** 3)
    
    def _kurtosis(self, data: List[float]) -> float:
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean([(x - mean) ** 4 for x in data]) / (std ** 4) - 3
    
    def _entropy(self, words: List[str]) -> float:
        if not words:
            return 0
        freq = Counter(words)
        probs = [c / len(words) for c in freq.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())
    
    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _empty(self) -> Dict[str, float]:
        return {n: 0.0 for n in self.feature_names()}
    
    def feature_names(self) -> List[str]:
        return [
            'chars', 'words', 'sents', 'avg_word_len',
            'sent_mean', 'sent_std', 'sent_cv', 'sent_range', 'sent_skew', 'sent_kurt',
            'ai_residue', 'ai_residue_pct',
            'paraphrase_markers', 'thesaurus_words', 'formal_synonyms',
            'human_informal', 'human_pct',
            'contractions', 'contraction_rate',
            'exclamations', 'questions', 'ellipses', 'dashes', 'parens', 'commas', 'semicolons', 'colons', 'quotes',
            'first_person', 'first_person_rate',
            'vocab_richness', 'hapax_ratio', 'top_word_freq',
            'bigram_rep', 'trigram_rep', 'quadgram_rep',
            'starter_diversity', 'starter_rep',
            'paragraphs', 'para_std', 'para_cv',
            'style_shift',
            'short_words', 'medium_words', 'long_words', 'very_long_words',
            'transitions', 'transition_rate',
            'passive_voice', 'passive_rate',
            'hedges', 'hedge_rate',
            'emphatics',
            'numbers', 'number_rate',
            'caps_ratio',
            'entropy'
        ]


def load_dataset_large(max_samples: int = 400000) -> Tuple[List[str], List[int]]:
    """Load larger balanced dataset."""
    logger.info("Loading large dataset...")
    
    texts, labels = [], []
    per_class = max_samples // 2
    
    # === Humanized from RAID (paraphrase attacks) ===
    logger.info(f"Loading {per_class} humanized samples from RAID...")
    
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    humanized_count = 0
    
    for item in tqdm(raid, desc="RAID humanized", total=per_class * 2):
        if humanized_count >= per_class:
            break
        if item.get('attack') == 'paraphrase' and item.get('model') != 'human':
            text = item.get('generation', '').strip()
            if text and 50 < len(text) < 10000:
                texts.append(text)
                labels.append(1)
                humanized_count += 1
    
    logger.info(f"Loaded {humanized_count} humanized samples")
    
    # === Human from multiple sources ===
    logger.info(f"Loading {per_class} human samples...")
    human_count = 0
    
    # RAID human
    raid = load_dataset('liamdugan/raid', split='train', streaming=True)
    for item in tqdm(raid, desc="RAID human", total=per_class):
        if human_count >= per_class:
            break
        if item.get('model') == 'human':
            text = item.get('generation', '').strip()
            if text and 50 < len(text) < 10000:
                texts.append(text)
                labels.append(0)
                human_count += 1
    
    # GPT-wiki human
    if human_count < per_class:
        remaining = per_class - human_count
        try:
            wiki = load_dataset('aadityaubhat/GPT-wiki-intro', split='train', streaming=True)
            for item in tqdm(wiki, desc="Wiki human", total=remaining):
                if human_count >= per_class:
                    break
                text = item.get('wiki_intro', '').strip()
                if text and 50 < len(text) < 10000:
                    texts.append(text)
                    labels.append(0)
                    human_count += 1
        except:
            pass
    
    logger.info(f"Loaded {human_count} human samples")
    logger.info(f"Total: {len(texts)} ({labels.count(0)} human, {labels.count(1)} humanized)")
    
    return texts, labels


def train_enhanced(max_samples: int = 400000):
    """Train enhanced Flare v2."""
    
    texts, labels = load_dataset_large(max_samples)
    
    logger.info("Loading embedding model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    extractor = EnhancedFlareFeatureExtractor(embedder)
    
    logger.info("Extracting features...")
    features_list = []
    embeddings_list = []
    
    for text in tqdm(texts, desc="Features"):
        feat = extractor.extract(text)
        features_list.append([feat[n] for n in extractor.feature_names()])
        emb = embedder.encode(text[:1000], show_progress_bar=False)
        embeddings_list.append(emb)
    
    X_feat = np.array(features_list, dtype=np.float32)
    X_emb = np.array(embeddings_list, dtype=np.float32)
    X = np.concatenate([X_feat, X_emb], axis=1)
    y = np.array(labels)
    
    logger.info(f"Features: {X.shape[1]} ({X_feat.shape[1]} heuristic + {X_emb.shape[1]} embedding)")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Enhanced XGBoost params
    params = {
        'n_estimators': 1000,
        'max_depth': 16,
        'learning_rate': 0.03,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'min_child_weight': 2,
        'reg_alpha': 0.05,
        'reg_lambda': 1.5,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
        'early_stopping_rounds': 50
    }
    
    logger.info("Training XGBoost...")
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    logger.info("\n" + "="*60)
    logger.info("FLARE V2 ENHANCED RESULTS")
    logger.info("="*60)
    logger.info(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1:        {f1:.4f}")
    logger.info(f"ROC AUC:   {auc:.4f}")
    logger.info("="*60)
    
    print(classification_report(y_test, y_pred, target_names=['Human', 'Humanized']))
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Human  Humanized")
    print(f"Human        {cm[0,0]:6d}     {cm[0,1]:6d}")
    print(f"Humanized    {cm[1,0]:6d}     {cm[1,1]:6d}")
    
    metrics = {
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'roc_auc': auc,
        'train_samples': len(X_train), 'test_samples': len(X_test),
        'features': X.shape[1], 'heuristic_features': len(extractor.feature_names())
    }
    
    if acc >= 0.99:
        logger.info("\n✅ 99% TARGET REACHED!")
        save_model(model, metrics, extractor)
    else:
        logger.info(f"\n⚠️ {acc*100:.2f}% - Close but below 99%")
        logger.info("Saving anyway for evaluation...")
        save_model(model, metrics, extractor, suffix='_best')
    
    return model, metrics


def save_model(model, metrics, extractor, suffix=''):
    """Save model and configs."""
    out_dir = Path('models/FlareV2' + suffix)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model.save_model(str(out_dir / 'flare_v2.json'))
    
    # ONNX conversion
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        
        initial_type = [('float_input', FloatTensorType([None, metrics['features']]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
        onnx.save_model(onnx_model, str(out_dir / 'flare_v2.onnx'))
        logger.info(f"Saved ONNX: {out_dir / 'flare_v2.onnx'}")
    except Exception as e:
        logger.warning(f"ONNX conversion failed: {e}")
    
    # Metadata
    meta = {
        'name': 'Flare V2',
        'version': '2.0.0',
        'type': 'human_vs_humanized',
        'created': datetime.now().isoformat(),
        'metrics': metrics,
        'features': extractor.feature_names(),
        'labels': {'0': 'human', '1': 'humanized'}
    }
    with open(out_dir / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)
    
    # JS config
    js = f"""// Flare V2 Config
const FLARE_V2_CONFIG = {{
    name: 'Flare V2',
    version: '2.0.0',
    type: 'human_vs_humanized',
    accuracy: {metrics['accuracy']:.4f},
    features: {metrics['features']},
    heuristicFeatures: {metrics['heuristic_features']},
    embeddingDim: 384,
    labels: {{ 0: 'human', 1: 'humanized' }},
    featureNames: {json.dumps(extractor.feature_names())}
}};

if (typeof module !== 'undefined') module.exports = FLARE_V2_CONFIG;
"""
    with open(out_dir / 'flare_v2_config.js', 'w') as f:
        f.write(js)
    
    logger.info(f"Saved to {out_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=400000)
    args = parser.parse_args()
    
    train_enhanced(args.samples)
