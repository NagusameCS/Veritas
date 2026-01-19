#!/usr/bin/env python3
"""
Humanization Detector Training
==============================
Dedicated model to detect humanized AI text vs pure human text.

The key insight: We're not trying to detect AI vs Human.
We're trying to detect HUMANIZED AI vs PURE HUMAN.

Humanized AI text has:
1. AI structural patterns (paragraph uniformity, predictable structure)
2. Human surface features (contractions, disfluencies, typos)
3. Inconsistencies between structure and style

This creates detectable artifacts that pure human text doesn't have.
"""

import os
import pickle
import json
import re
import random
import numpy as np
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset

# Humanization simulation
HUMANIZATION_TRANSFORMS = {
    'disfluencies': ['Well, ', 'So, ', 'I mean, ', 'Basically, ', 'Honestly, ', 'Like, ', 'Actually, ', 'Anyway, '],
    'contractions': [
        ('it is', "it's"), ('do not', "don't"), ('cannot', "can't"), ('will not', "won't"),
        ('would not', "wouldn't"), ('should not', "shouldn't"), ('I am', "I'm"),
        ('they are', "they're"), ('we are', "we're"), ('that is', "that's"),
        ('there is', "there's"), ('have not', "haven't"), ('has not', "hasn't"),
    ],
    'typos': [
        ('the', 'teh'), ('and', 'adn'), ('that', 'taht'), ('with', 'wiht'),
        ('have', 'ahve'), ('this', 'tihs'), ('because', 'becuase'),
    ],
    'slang': [
        ('very', 'super'), ('extremely', 'totally'), ('certainly', 'for sure'),
        ('excellent', 'awesome'), ('remarkable', 'cool'),
    ],
}


def humanize_text(text: str, intensity: float = 0.5) -> str:
    """Simulate humanization of AI text"""
    result = text
    
    # Add disfluencies at sentence starts
    sentences = re.split(r'(?<=[.!?])\s+', result)
    modified = []
    for i, sent in enumerate(sentences):
        if random.random() < intensity * 0.4 and len(sent) > 20:
            sent = random.choice(HUMANIZATION_TRANSFORMS['disfluencies']) + sent[0].lower() + sent[1:]
        modified.append(sent)
    result = ' '.join(modified)
    
    # Apply contractions
    for old, new in HUMANIZATION_TRANSFORMS['contractions']:
        if random.random() < intensity * 0.8:
            result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    # Add typos
    if intensity > 0.3:
        words = result.split()
        for i, word in enumerate(words):
            if random.random() < intensity * 0.015:
                for old, new in HUMANIZATION_TRANSFORMS['typos']:
                    if word.lower() == old:
                        words[i] = new
                        break
        result = ' '.join(words)
    
    # Add slang
    if intensity > 0.4:
        for old, new in HUMANIZATION_TRANSFORMS['slang']:
            if random.random() < intensity * 0.4:
                result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    return result


class HumanizationFeatureExtractor:
    """
    Extract features that distinguish humanized AI from pure human text.
    
    Key insight: Humanized text has AI STRUCTURE with human SURFACE.
    We detect this mismatch.
    """
    
    def __init__(self):
        self.feature_names = [
            # === STRUCTURE-SURFACE MISMATCH ===
            'structure_surface_mismatch',      # High uniformity BUT high contraction rate
            'formal_informal_cooccurrence',    # Formal phrases with informal elements
            'disfluency_with_structure',       # Disfluencies in well-structured text
            
            # === CONTRACTION PATTERNS ===
            'contraction_rate',                # Overall contraction frequency
            'contraction_distribution_cv',     # How evenly distributed contractions are
            'contraction_sentence_correlation', # Contractions appearing in streaks
            
            # === DISFLUENCY PATTERNS ===
            'sentence_start_disfluency_rate',  # "Well," "So," at sentence starts
            'mid_sentence_disfluency_rate',    # "like," "you know" in middle
            'disfluency_uniformity',           # How evenly spread disfluencies are
            
            # === AI STRUCTURE MARKERS ===
            'paragraph_uniformity',            # AI paragraphs are very uniform
            'sentence_length_cv',              # AI sentence lengths are uniform
            'sentence_start_diversity',        # AI uses similar sentence starters
            'transition_density',              # AI uses more transition words
            
            # === HUMAN SURFACE MARKERS ===
            'typo_density',                    # Humanizers add typos
            'slang_density',                   # Humanizers add slang
            'informal_marker_density',         # Overall informal markers
            
            # === ENTROPY ANOMALIES ===
            'local_entropy_variance',          # Entropy varies unnaturally
            'bigram_entropy',                  # N-gram predictability
            'vocabulary_sophistication_cv',    # Word complexity varies unnaturally
            
            # === CO-OCCURRENCE SIGNALS ===
            'formal_phrase_count',             # AI formal phrases present
            'informal_marker_count',           # Informal markers present
            'formal_informal_ratio',           # Ratio of formal to informal
            
            # === STRUCTURAL INCONSISTENCY ===
            'complexity_uniformity_mismatch',  # Complex words in simple structure
            'register_consistency',            # Consistency of formality level
            'punctuation_anomaly_score',       # Unusual punctuation patterns
        ]
        
        self.formal_phrases = [
            'it is important', 'furthermore', 'moreover', 'consequently',
            'in conclusion', 'demonstrates', 'comprehensive', 'significant',
            'subsequently', 'nevertheless', 'therefore', 'additionally',
            'it should be noted', 'one must consider', 'it is essential',
        ]
        
        self.informal_markers = [
            'gonna', 'wanna', 'kinda', 'sorta', 'gotta', 'dunno', 'yeah',
            'nope', 'awesome', 'cool', 'super', 'totally', 'basically',
        ]
        
        self.disfluencies = [
            'well,', 'so,', 'i mean,', 'basically,', 'honestly,', 'like,',
            'actually,', 'anyway,', 'you know,', 'right?',
        ]
        
        self.transitions = [
            'however', 'therefore', 'furthermore', 'moreover', 'consequently',
            'additionally', 'nevertheless', 'subsequently', 'thus', 'hence',
        ]
    
    def extract(self, text: str) -> np.ndarray:
        """Extract all features for humanization detection"""
        lower = text.lower()
        words = re.findall(r"[a-zA-Z']+", lower)
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(words) < 20 or len(sentences) < 3:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # === CONTRACTION ANALYSIS ===
        contractions = re.findall(r"\b\w+'\w+\b", lower)
        features['contraction_rate'] = len(contractions) / len(words) * 100
        
        # Contraction distribution across sentences
        contr_per_sent = [len(re.findall(r"\b\w+'\w+\b", s.lower())) for s in sentences]
        if sum(contr_per_sent) > 0:
            mean_c = np.mean(contr_per_sent)
            features['contraction_distribution_cv'] = np.std(contr_per_sent) / mean_c if mean_c > 0 else 0
            # Correlation: contractions appearing in streaks
            if len(contr_per_sent) > 2:
                features['contraction_sentence_correlation'] = self._autocorrelation(contr_per_sent)
            else:
                features['contraction_sentence_correlation'] = 0
        else:
            features['contraction_distribution_cv'] = 0
            features['contraction_sentence_correlation'] = 0
        
        # === DISFLUENCY ANALYSIS ===
        sent_start_disfluencies = sum(1 for s in sentences 
            if any(s.lower().strip().startswith(d) for d in self.disfluencies))
        features['sentence_start_disfluency_rate'] = sent_start_disfluencies / len(sentences)
        
        mid_disfluencies = sum(1 for d in ['like,', 'you know,', 'i mean,'] 
            if f' {d} ' in lower or f' {d}' in lower)
        features['mid_sentence_disfluency_rate'] = mid_disfluencies / len(sentences)
        
        # Disfluency uniformity (are they evenly spread?)
        disf_per_sent = [sum(1 for d in self.disfluencies if d in s.lower()) for s in sentences]
        if sum(disf_per_sent) > 0:
            features['disfluency_uniformity'] = 1 - (np.std(disf_per_sent) / (np.mean(disf_per_sent) + 0.01))
        else:
            features['disfluency_uniformity'] = 0
        
        # === AI STRUCTURE MARKERS ===
        if len(paragraphs) > 1:
            para_lengths = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - (np.std(para_lengths) / (np.mean(para_lengths) + 0.01))
        else:
            features['paragraph_uniformity'] = 0.5
        
        sent_lengths = [len(s.split()) for s in sentences]
        features['sentence_length_cv'] = np.std(sent_lengths) / (np.mean(sent_lengths) + 0.01)
        
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        
        transition_count = sum(1 for t in self.transitions if t in lower)
        features['transition_density'] = transition_count / len(sentences)
        
        # === HUMAN SURFACE MARKERS ===
        typos = ['teh', 'adn', 'taht', 'wiht', 'ahve', 'becuase', 'definately', 'probaly']
        features['typo_density'] = sum(1 for w in words if w in typos) / len(words) * 100
        
        slang_words = ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'awesome', 'cool', 'super', 'totally']
        features['slang_density'] = sum(1 for w in words if w in slang_words) / len(words) * 100
        
        features['informal_marker_density'] = (features['typo_density'] + features['slang_density'] + 
            features['sentence_start_disfluency_rate'] * 10)
        
        # === CO-OCCURRENCE ===
        features['formal_phrase_count'] = sum(1 for p in self.formal_phrases if p in lower)
        features['informal_marker_count'] = sum(1 for m in self.informal_markers if m in lower)
        
        total_markers = features['formal_phrase_count'] + features['informal_marker_count']
        if total_markers > 0:
            features['formal_informal_ratio'] = features['formal_phrase_count'] / total_markers
        else:
            features['formal_informal_ratio'] = 0.5
        
        # === STRUCTURE-SURFACE MISMATCH ===
        # High paragraph uniformity (AI) with high contraction rate (humanized)
        features['structure_surface_mismatch'] = (
            features['paragraph_uniformity'] * features['contraction_rate'] / 10
        )
        
        # Formal phrases co-occurring with informal markers
        features['formal_informal_cooccurrence'] = (
            features['formal_phrase_count'] * features['informal_marker_count']
        )
        
        # Disfluencies in structurally uniform text
        features['disfluency_with_structure'] = (
            features['sentence_start_disfluency_rate'] * (1 - features['sentence_length_cv'])
        )
        
        # === ENTROPY ANOMALIES ===
        # Local entropy variance
        chunk_size = max(len(words) // 5, 10)
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size) if len(words[i:i+chunk_size]) >= 5]
        if len(chunks) >= 2:
            chunk_entropies = [self._entropy(c) for c in chunks]
            features['local_entropy_variance'] = np.std(chunk_entropies)
        else:
            features['local_entropy_variance'] = 0
        
        # Bigram entropy
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        features['bigram_entropy'] = self._entropy(bigrams)
        
        # Word sophistication variance
        word_lengths = [len(w) for w in words]
        features['vocabulary_sophistication_cv'] = np.std(word_lengths) / (np.mean(word_lengths) + 0.01)
        
        # === STRUCTURAL INCONSISTENCY ===
        # Complex words in simple structure
        complex_words = sum(1 for w in words if len(w) > 8)
        simple_structure = 1 - features['sentence_length_cv']
        features['complexity_uniformity_mismatch'] = (complex_words / len(words)) * simple_structure
        
        # Register consistency
        formal_count = features['formal_phrase_count']
        informal_count = features['sentence_start_disfluency_rate'] * len(sentences) + features['informal_marker_count']
        total = formal_count + informal_count + 0.01
        features['register_consistency'] = 1 - abs(formal_count - informal_count) / total
        
        # Punctuation anomaly
        comma_rate = text.count(',') / len(text) * 100
        period_rate = text.count('.') / len(text) * 100
        expected_comma = 1.5
        expected_period = 1.0
        features['punctuation_anomaly_score'] = (
            abs(comma_rate - expected_comma) + abs(period_rate - expected_period)
        )
        
        return np.array([features.get(name, 0.0) for name in self.feature_names])
    
    def _split_sentences(self, text):
        text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|inc|ltd)\.', r'\1<PERIOD>', text, flags=re.IGNORECASE)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.replace('<PERIOD>', '.').strip() for s in sentences if len(s.split()) >= 3]
    
    def _entropy(self, items):
        if not items: return 0
        counts = Counter(items)
        total = len(items)
        return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)
    
    def _autocorrelation(self, values, lag=1):
        if len(values) <= lag: return 0
        n = len(values)
        mean = np.mean(values)
        var = np.var(values)
        if var == 0: return 0
        cov = np.sum([(values[i] - mean) * (values[i+lag] - mean) for i in range(n-lag)]) / (n - lag)
        return cov / var


def create_training_data(n_samples=5000):
    """Create training data for humanization detection"""
    print("Loading dataset...")
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    human_texts = []
    ai_texts = []
    
    for item in dataset:
        if len(item.get('wiki_intro', '').split()) >= 50:
            human_texts.append(item['wiki_intro'])
        if len(item.get('generated_intro', '').split()) >= 50:
            ai_texts.append(item['generated_intro'])
        
        if len(human_texts) >= n_samples and len(ai_texts) >= n_samples:
            break
    
    human_texts = human_texts[:n_samples]
    ai_texts = ai_texts[:n_samples]
    
    print(f"Collected {len(human_texts)} human, {len(ai_texts)} AI samples")
    
    # Create humanized versions with varying intensity
    humanized_texts = []
    for text in ai_texts:
        intensity = random.uniform(0.3, 0.9)  # Random intensity
        humanized_texts.append(humanize_text(text, intensity))
    
    print(f"Created {len(humanized_texts)} humanized samples")
    
    return human_texts, humanized_texts


def train_humanization_detector():
    """Train the humanization detector"""
    print("=" * 70)
    print("HUMANIZATION DETECTOR TRAINING")
    print("=" * 70)
    
    # Create training data
    human_texts, humanized_texts = create_training_data(n_samples=3000)
    
    # Extract features
    extractor = HumanizationFeatureExtractor()
    
    print("\nExtracting features...")
    X_human = []
    X_humanized = []
    
    for i, text in enumerate(human_texts):
        if i % 500 == 0:
            print(f"  Human samples: {i}/{len(human_texts)}")
        features = extractor.extract(text)
        if np.sum(np.abs(features)) > 0:  # Valid features
            X_human.append(features)
    
    for i, text in enumerate(humanized_texts):
        if i % 500 == 0:
            print(f"  Humanized samples: {i}/{len(humanized_texts)}")
        features = extractor.extract(text)
        if np.sum(np.abs(features)) > 0:
            X_humanized.append(features)
    
    print(f"\nValid samples: {len(X_human)} human, {len(X_humanized)} humanized")
    
    # Create balanced dataset
    min_samples = min(len(X_human), len(X_humanized))
    X_human = X_human[:min_samples]
    X_humanized = X_humanized[:min_samples]
    
    X = np.vstack([X_human, X_humanized])
    y = np.array([0] * len(X_human) + [1] * len(X_humanized))  # 0 = pure human, 1 = humanized AI
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Gradient Boosting (better for imbalanced patterns)
    print("\nTraining Gradient Boosting classifier...")
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Human, Pred Human:     {cm[0][0]}")
    print(f"  True Human, Pred Humanized: {cm[0][1]} (False Positives)")
    print(f"  True Humanized, Pred Human: {cm[1][0]} (False Negatives)")
    print(f"  True Humanized, Pred Humanized: {cm[1][1]}")
    
    # Feature importance
    print("\nTop 10 Most Important Features:")
    importance = list(zip(extractor.feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importance[:10]:
        print(f"  {name}: {imp*100:.2f}%")
    
    # Save model
    model_dir = "/workspaces/Veritas/training/models/HumanizationDetector"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "model_name": "HumanizationDetector",
        "version": "1.0.0",
        "description": "Detects humanized AI text vs pure human text",
        "task": "Binary classification: Pure Human (0) vs Humanized AI (1)",
        "feature_names": extractor.feature_names,
        "feature_count": len(extractor.feature_names),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "results": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
        },
        "confusion_matrix": {
            "true_human_pred_human": int(cm[0][0]),
            "true_human_pred_humanized": int(cm[0][1]),
            "true_humanized_pred_human": int(cm[1][0]),
            "true_humanized_pred_humanized": int(cm[1][1]),
        },
        "feature_importance": {name: float(imp) for name, imp in importance},
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_dir}")
    print("=" * 70)
    
    return model, scaler, extractor, metadata


def test_on_examples():
    """Test the trained model on example texts"""
    model_dir = "/workspaces/Veritas/training/models/HumanizationDetector"
    
    with open(f"{model_dir}/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{model_dir}/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    extractor = HumanizationFeatureExtractor()
    
    # Test samples
    pure_human = """
    So I was thinking about this yesterday, right? My friend Dave told me about this crazy thing 
    that happened at work. Honestly, I couldn't believe it. Like, who does that? Anyway, we ended 
    up laughing about it for hours. The whole situation was just absurd.
    """
    
    pure_ai = """
    It is important to note that artificial intelligence has demonstrated significant capabilities 
    in various domains. The implementation of these systems requires comprehensive understanding 
    of the underlying mechanisms. Furthermore, one must consider the ethical implications that 
    subsequently arise from such technological advancements.
    """
    
    humanized_ai = """
    So, it's important to note that artificial intelligence has demonstrated significant capabilities 
    in various domains. Honestly, the implementation of these systems requires comprehensive understanding 
    of the underlying mechanisms. Like, furthermore, one must consider the ethical implications that 
    subsequently arise from such technological advancements, you know?
    """
    
    print("\n" + "=" * 70)
    print("TESTING ON EXAMPLES")
    print("=" * 70)
    
    for name, text in [("Pure Human", pure_human), ("Pure AI", pure_ai), ("Humanized AI", humanized_ai)]:
        features = extractor.extract(text)
        scaled = scaler.transform([features])
        prob = model.predict_proba(scaled)[0][1]
        pred = "Humanized AI" if prob > 0.5 else "Pure Human"
        
        print(f"\n{name}:")
        print(f"  Humanization Probability: {prob:.3f}")
        print(f"  Prediction: {pred}")


if __name__ == "__main__":
    model, scaler, extractor, metadata = train_humanization_detector()
    test_on_examples()
