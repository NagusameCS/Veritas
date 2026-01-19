#!/usr/bin/env python3
"""
Humanization Detector v2.0
==========================
Better feature engineering focused on MISMATCH between AI structure and human surface.

Key insight: Humanized AI has:
1. AI structural regularity (paragraph uniformity, sentence uniformity)
2. Human surface additions (contractions, disfluencies)
3. This COMBINATION is unnatural

Pure human text has natural structure-surface consistency.
Pure AI text has neither human markers nor mismatch.
Humanized AI text has the telltale MISMATCH.
"""

import os
import pickle
import json
import re
import random
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from datasets import load_dataset


# Humanization simulation
def humanize_text(text: str, intensity: float = 0.5) -> str:
    """Simulate humanization of AI text"""
    result = text
    
    disfluencies = ['Well, ', 'So, ', 'I mean, ', 'Basically, ', 'Honestly, ', 'Like, ', 'Actually, ']
    contractions = [
        ('it is', "it's"), ('do not', "don't"), ('cannot', "can't"), ('will not', "won't"),
        ('would not', "wouldn't"), ('should not', "shouldn't"), ('I am', "I'm"),
        ('they are', "they're"), ('we are', "we're"), ('that is', "that's"),
        ('there is', "there's"), ('have not', "haven't"), ('has not', "hasn't"),
    ]
    
    # Add disfluencies at sentence starts
    sentences = re.split(r'(?<=[.!?])\s+', result)
    modified = []
    for sent in sentences:
        if random.random() < intensity * 0.35 and len(sent) > 20:
            sent = random.choice(disfluencies) + sent[0].lower() + sent[1:]
        modified.append(sent)
    result = ' '.join(modified)
    
    # Apply contractions
    for old, new in contractions:
        if random.random() < intensity * 0.7:
            result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    return result


class HumanizationDetectorV2:
    """
    Detects humanized AI by looking for structure-surface mismatch.
    """
    
    def __init__(self):
        self.feature_names = [
            # === CORE MISMATCH FEATURES ===
            'structural_regularity',           # How uniform the structure is (AI signature)
            'surface_informality',             # How informal the surface is
            'mismatch_score',                  # Structure × Surface = mismatch
            
            # === STRUCTURE METRICS ===
            'paragraph_uniformity',
            'sentence_length_cv',
            'sentence_start_repetition',
            'transition_word_density',
            
            # === SURFACE METRICS ===
            'contraction_rate',
            'disfluency_rate',
            'first_person_rate',
            
            # === ENTROPY FEATURES ===
            'bigram_entropy',
            'local_entropy_variance',
            
            # === CONSISTENCY FEATURES ===
            'formality_consistency',           # How consistent is formality level
            'vocabulary_consistency',          # Word complexity consistency
        ]
        
        self.formal_words = ['furthermore', 'moreover', 'consequently', 'therefore', 'subsequently',
                            'comprehensive', 'significant', 'demonstrates', 'implementation']
        self.transitions = ['however', 'therefore', 'furthermore', 'moreover', 'consequently',
                           'additionally', 'nevertheless', 'thus', 'hence']
        self.disfluencies = ['well,', 'so,', 'i mean,', 'basically,', 'honestly,', 'like,', 'actually,']
    
    def extract(self, text: str) -> np.ndarray:
        lower = text.lower()
        words = re.findall(r"[a-zA-Z']+", lower)
        sentences = self._split_sentences(text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(words) < 20 or len(sentences) < 2:
            return np.zeros(len(self.feature_names))
        
        features = {}
        
        # === STRUCTURE METRICS ===
        # Paragraph uniformity
        if len(paragraphs) > 1:
            para_lens = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - min(np.std(para_lens) / (np.mean(para_lens) + 1), 1)
        else:
            features['paragraph_uniformity'] = 0.5
        
        # Sentence length CV
        sent_lens = [len(s.split()) for s in sentences]
        features['sentence_length_cv'] = min(np.std(sent_lens) / (np.mean(sent_lens) + 1), 1)
        
        # Sentence start repetition
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        start_counts = Counter(starts)
        repeated = sum(1 for c in start_counts.values() if c > 1)
        features['sentence_start_repetition'] = repeated / len(sentences)
        
        # Transition density
        trans_count = sum(1 for t in self.transitions if t in lower)
        features['transition_word_density'] = trans_count / len(sentences)
        
        # Overall structural regularity (high = AI-like)
        features['structural_regularity'] = (
            features['paragraph_uniformity'] * 0.4 +
            (1 - features['sentence_length_cv']) * 0.3 +
            features['sentence_start_repetition'] * 0.15 +
            min(features['transition_word_density'], 1) * 0.15
        )
        
        # === SURFACE METRICS ===
        # Contraction rate
        contractions = re.findall(r"\b\w+'\w+\b", lower)
        features['contraction_rate'] = len(contractions) / len(words) * 10
        
        # Disfluency rate
        disf_count = sum(1 for s in sentences if any(s.lower().strip().startswith(d) for d in self.disfluencies))
        features['disfluency_rate'] = disf_count / len(sentences)
        
        # First person rate
        first_person = sum(1 for w in words if w in ['i', 'me', 'my', 'we', 'us', 'our'])
        features['first_person_rate'] = first_person / len(words) * 10
        
        # Overall surface informality
        features['surface_informality'] = (
            features['contraction_rate'] * 0.4 +
            features['disfluency_rate'] * 0.4 +
            features['first_person_rate'] * 0.2
        )
        
        # === CORE MISMATCH ===
        # Mismatch = high structure regularity × high surface informality
        # Pure human: low structure, high surface → low mismatch
        # Pure AI: high structure, low surface → low mismatch
        # Humanized: high structure, high surface → HIGH mismatch
        features['mismatch_score'] = features['structural_regularity'] * features['surface_informality']
        
        # === ENTROPY FEATURES ===
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        features['bigram_entropy'] = self._entropy(bigrams)
        
        # Local entropy variance
        chunk_size = max(len(words) // 4, 10)
        chunks = [words[i:i+chunk_size] for i in range(0, len(words), chunk_size) if len(words[i:i+chunk_size]) >= 5]
        if len(chunks) >= 2:
            chunk_ents = [self._entropy(c) for c in chunks]
            features['local_entropy_variance'] = np.std(chunk_ents)
        else:
            features['local_entropy_variance'] = 0
        
        # === CONSISTENCY FEATURES ===
        # Formality consistency: check if formal and informal markers coexist
        formal_count = sum(1 for w in self.formal_words if w in lower)
        informal_count = features['disfluency_rate'] * len(sentences) + len(contractions)
        if formal_count + informal_count > 0:
            features['formality_consistency'] = abs(formal_count - informal_count) / (formal_count + informal_count + 1)
        else:
            features['formality_consistency'] = 1.0  # No markers = consistent
        
        # Vocabulary consistency
        word_lengths = [len(w) for w in words]
        features['vocabulary_consistency'] = 1 - min(np.std(word_lengths) / (np.mean(word_lengths) + 1), 1)
        
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


def create_training_data(n_samples=4000):
    """Create training data: pure human vs humanized AI"""
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
        intensity = random.uniform(0.4, 0.9)
        humanized_texts.append(humanize_text(text, intensity))
    
    print(f"Created {len(humanized_texts)} humanized samples")
    
    return human_texts, humanized_texts


def train_model():
    """Train the humanization detector"""
    print("=" * 70)
    print("HUMANIZATION DETECTOR V2 TRAINING")
    print("=" * 70)
    
    human_texts, humanized_texts = create_training_data(n_samples=4000)
    
    extractor = HumanizationDetectorV2()
    
    print("\nExtracting features...")
    X_human = []
    X_humanized = []
    
    for i, text in enumerate(human_texts):
        if i % 1000 == 0:
            print(f"  Human: {i}/{len(human_texts)}")
        features = extractor.extract(text)
        if np.sum(np.abs(features)) > 0:
            X_human.append(features)
    
    for i, text in enumerate(humanized_texts):
        if i % 1000 == 0:
            print(f"  Humanized: {i}/{len(humanized_texts)}")
        features = extractor.extract(text)
        if np.sum(np.abs(features)) > 0:
            X_humanized.append(features)
    
    print(f"\nValid: {len(X_human)} human, {len(X_humanized)} humanized")
    
    # Balance
    min_samples = min(len(X_human), len(X_humanized))
    X_human = X_human[:min_samples]
    X_humanized = X_humanized[:min_samples]
    
    X = np.vstack([X_human, X_humanized])
    y = np.array([0] * len(X_human) + [1] * len(X_humanized))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train ensemble
    print("\nTraining ensemble classifier...")
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    rf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    
    model = VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft')
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
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  True Human, Pred Human:        {cm[0][0]} ({cm[0][0]/(cm[0][0]+cm[0][1])*100:.1f}%)")
    print(f"  True Human, Pred Humanized:    {cm[0][1]} (FP: {cm[0][1]/(cm[0][0]+cm[0][1])*100:.1f}%)")
    print(f"  True Humanized, Pred Human:    {cm[1][0]} (FN: {cm[1][0]/(cm[1][0]+cm[1][1])*100:.1f}%)")
    print(f"  True Humanized, Pred Humanized:{cm[1][1]} ({cm[1][1]/(cm[1][0]+cm[1][1])*100:.1f}%)")
    
    # Feature importance (from GB component)
    print("\nFeature Importance (from Gradient Boosting):")
    importance = list(zip(extractor.feature_names, gb.fit(X_train_scaled, y_train).feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importance:
        print(f"  {name}: {imp*100:.2f}%")
    
    # Save
    model_dir = "/workspaces/Veritas/training/models/HumanizationDetector"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "model_name": "HumanizationDetector",
        "version": "2.0.0",
        "description": "Detects humanized AI text via structure-surface mismatch",
        "feature_names": extractor.feature_names,
        "feature_count": len(extractor.feature_names),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "results": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
        },
        "confusion_matrix": {
            "tn": int(cm[0][0]), "fp": int(cm[0][1]),
            "fn": int(cm[1][0]), "tp": int(cm[1][1]),
        },
        "feature_importance": {name: float(imp) for name, imp in importance},
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_dir}")
    
    # Test on examples
    print("\n" + "=" * 70)
    print("TESTING ON EXAMPLES")
    print("=" * 70)
    
    examples = {
        "Pure Human": """
            So I was thinking about this yesterday, right? My friend Dave told me about this crazy thing 
            that happened at work. Honestly, I couldn't believe it. Like, who does that? We ended up 
            laughing about it for hours. The whole situation was just absurd and kind of hilarious.
        """,
        "Pure AI": """
            It is important to note that artificial intelligence has demonstrated significant capabilities 
            in various domains. The implementation of these systems requires comprehensive understanding 
            of the underlying mechanisms. Furthermore, one must consider the ethical implications that 
            subsequently arise from such technological advancements.
        """,
        "Humanized AI": """
            So, it's important to note that artificial intelligence has demonstrated significant capabilities 
            in various domains. Honestly, the implementation of these systems requires comprehensive understanding 
            of the underlying mechanisms. Like, furthermore, one must consider the ethical implications that 
            subsequently arise from such technological advancements.
        """,
    }
    
    for name, text in examples.items():
        features = extractor.extract(text)
        scaled = scaler.transform([features])
        prob = model.predict_proba(scaled)[0][1]
        
        print(f"\n{name}:")
        print(f"  Humanization Probability: {prob:.3f}")
        print(f"  Prediction: {'Humanized AI' if prob > 0.5 else 'Pure Human/AI'}")
        print(f"  Key features:")
        print(f"    - Structure regularity: {features[0]:.3f}")
        print(f"    - Surface informality:  {features[1]:.3f}")
        print(f"    - Mismatch score:       {features[2]:.3f}")
    
    return model, scaler, extractor, metadata


if __name__ == "__main__":
    train_model()
