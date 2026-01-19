#!/usr/bin/env python3
"""
VERITAS 3-Way Detection System
==============================
Instead of training separate models, we train ONE model that classifies into:
1. Pure Human (organic human writing)
2. Pure AI (unmodified AI output)
3. Humanized AI (AI text that's been modified to seem human)

The key is using ALL three classes together so the model learns the differences.
"""

import os
import pickle
import json
import re
import random
import numpy as np
from datetime import datetime
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import load_dataset


def humanize_text(text: str, intensity: float = 0.6) -> str:
    """Simulate humanization of AI text"""
    result = text
    
    disfluencies = ['Well, ', 'So, ', 'I mean, ', 'Basically, ', 'Honestly, ', 'Like, ', 'Actually, ']
    contractions = [
        ('it is', "it's"), ('do not', "don't"), ('cannot', "can't"), ('will not', "won't"),
        ('would not', "wouldn't"), ('should not', "shouldn't"), ('I am', "I'm"),
        ('they are', "they're"), ('we are', "we're"), ('that is', "that's"),
    ]
    
    sentences = re.split(r'(?<=[.!?])\s+', result)
    modified = []
    for sent in sentences:
        if random.random() < intensity * 0.35 and len(sent) > 20:
            sent = random.choice(disfluencies) + sent[0].lower() + sent[1:]
        modified.append(sent)
    result = ' '.join(modified)
    
    for old, new in contractions:
        if random.random() < intensity * 0.7:
            result = re.sub(rf'\b{old}\b', new, result, flags=re.IGNORECASE)
    
    return result


class TriClassExtractor:
    """
    Extract features for 3-way classification.
    Focus on features that differentiate ALL THREE classes.
    """
    
    def __init__(self):
        self.feature_names = [
            # === STRUCTURAL FEATURES (AI signature) ===
            'paragraph_uniformity',
            'sentence_length_cv',
            'sentence_length_range_norm',
            'sentence_start_diversity',
            'avg_sentence_length',
            
            # === VOCABULARY FEATURES ===
            'type_token_ratio',
            'hapax_ratio',
            'avg_word_length',
            'long_word_ratio',
            
            # === ENTROPY FEATURES ===
            'unigram_entropy',
            'bigram_entropy',
            'trigram_entropy',
            
            # === FORMALITY MARKERS ===
            'formal_phrase_density',
            'transition_density',
            'passive_voice_markers',
            
            # === INFORMALITY MARKERS ===
            'contraction_rate',
            'disfluency_rate',
            'first_person_rate',
            'question_rate',
            
            # === MISMATCH INDICATORS ===
            'formality_score',          # High = formal
            'informality_score',        # High = informal
            'structure_uniformity',     # High = AI-like structure
            'surface_naturalness',      # High = human-like surface
        ]
        
        self.formal_phrases = ['it is important', 'furthermore', 'moreover', 'consequently',
                               'demonstrates', 'significant', 'comprehensive', 'subsequently']
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
        word_freq = Counter(words)
        
        # === STRUCTURAL FEATURES ===
        if len(paragraphs) > 1:
            para_lens = [len(p.split()) for p in paragraphs]
            features['paragraph_uniformity'] = 1 - min(np.std(para_lens) / (np.mean(para_lens) + 1), 1)
        else:
            features['paragraph_uniformity'] = 0.5
        
        sent_lens = [len(s.split()) for s in sentences]
        features['sentence_length_cv'] = np.std(sent_lens) / (np.mean(sent_lens) + 0.01)
        features['sentence_length_range_norm'] = (max(sent_lens) - min(sent_lens)) / (np.mean(sent_lens) + 0.01)
        
        starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
        features['sentence_start_diversity'] = len(set(starts)) / len(sentences)
        features['avg_sentence_length'] = np.mean(sent_lens)
        
        # === VOCABULARY FEATURES ===
        features['type_token_ratio'] = len(set(words)) / len(words)
        features['hapax_ratio'] = sum(1 for c in word_freq.values() if c == 1) / len(words)
        
        word_lens = [len(w) for w in words]
        features['avg_word_length'] = np.mean(word_lens)
        features['long_word_ratio'] = sum(1 for l in word_lens if l > 8) / len(words)
        
        # === ENTROPY FEATURES ===
        features['unigram_entropy'] = self._entropy(words)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        features['bigram_entropy'] = self._entropy(bigrams)
        trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)]
        features['trigram_entropy'] = self._entropy(trigrams)
        
        # === FORMALITY MARKERS ===
        formal_count = sum(1 for p in self.formal_phrases if p in lower)
        features['formal_phrase_density'] = formal_count / len(sentences)
        
        trans_count = sum(1 for t in self.transitions if t in lower)
        features['transition_density'] = trans_count / len(sentences)
        
        passive_markers = ['is', 'are', 'was', 'were', 'been', 'being']
        passive_count = sum(1 for w in words if w in passive_markers)
        features['passive_voice_markers'] = passive_count / len(words)
        
        # === INFORMALITY MARKERS ===
        contractions = re.findall(r"\b\w+'\w+\b", lower)
        features['contraction_rate'] = len(contractions) / len(words)
        
        disf_count = sum(1 for s in sentences if any(s.lower().strip().startswith(d) for d in self.disfluencies))
        features['disfluency_rate'] = disf_count / len(sentences)
        
        first_person = sum(1 for w in words if w in ['i', 'me', 'my', 'we', 'us', 'our'])
        features['first_person_rate'] = first_person / len(words)
        
        features['question_rate'] = text.count('?') / len(sentences)
        
        # === COMPOSITE SCORES ===
        features['formality_score'] = (
            features['formal_phrase_density'] * 0.3 +
            features['transition_density'] * 0.3 +
            features['passive_voice_markers'] * 0.2 +
            (1 - features['contraction_rate']) * 0.2
        )
        
        features['informality_score'] = (
            features['contraction_rate'] * 0.3 +
            features['disfluency_rate'] * 0.3 +
            features['first_person_rate'] * 0.2 +
            features['question_rate'] * 0.2
        )
        
        features['structure_uniformity'] = (
            features['paragraph_uniformity'] * 0.4 +
            (1 - features['sentence_length_cv']) * 0.3 +
            (1 - features['sentence_start_diversity']) * 0.3
        )
        
        features['surface_naturalness'] = (
            features['sentence_length_cv'] * 0.3 +
            features['sentence_start_diversity'] * 0.3 +
            features['type_token_ratio'] * 0.2 +
            features['hapax_ratio'] * 0.2
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


def train_three_way():
    """Train 3-way classifier: Human vs AI vs Humanized"""
    print("=" * 70)
    print("3-WAY CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Load data
    print("\nLoading dataset...")
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    human_texts = []
    ai_texts = []
    
    for item in dataset:
        if len(item.get('wiki_intro', '').split()) >= 50:
            human_texts.append(item['wiki_intro'])
        if len(item.get('generated_intro', '').split()) >= 50:
            ai_texts.append(item['generated_intro'])
        
        if len(human_texts) >= 4000 and len(ai_texts) >= 4000:
            break
    
    human_texts = human_texts[:4000]
    ai_texts = ai_texts[:4000]
    
    # Create humanized versions (from AI texts)
    humanized_texts = [humanize_text(t, random.uniform(0.4, 0.9)) for t in ai_texts[:4000]]
    
    print(f"Samples: {len(human_texts)} human, {len(ai_texts)} AI, {len(humanized_texts)} humanized")
    
    # Extract features
    extractor = TriClassExtractor()
    
    print("\nExtracting features...")
    X_human = [extractor.extract(t) for t in human_texts]
    X_ai = [extractor.extract(t) for t in ai_texts]
    X_humanized = [extractor.extract(t) for t in humanized_texts]
    
    # Filter valid
    X_human = [x for x in X_human if np.sum(np.abs(x)) > 0]
    X_ai = [x for x in X_ai if np.sum(np.abs(x)) > 0]
    X_humanized = [x for x in X_humanized if np.sum(np.abs(x)) > 0]
    
    print(f"Valid: {len(X_human)} human, {len(X_ai)} AI, {len(X_humanized)} humanized")
    
    # Balance classes
    min_samples = min(len(X_human), len(X_ai), len(X_humanized))
    X_human = X_human[:min_samples]
    X_ai = X_ai[:min_samples]
    X_humanized = X_humanized[:min_samples]
    
    # Create dataset
    X = np.vstack([X_human, X_ai, X_humanized])
    y = np.array([0] * len(X_human) + [1] * len(X_ai) + [2] * len(X_humanized))
    # 0 = Human, 1 = AI, 2 = Humanized AI
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest (good for multiclass)
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    class_names = ['Human', 'AI', 'Humanized']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"             Pred Human  Pred AI  Pred Humanized")
    for i, name in enumerate(class_names):
        print(f"True {name:10s} {cm[i][0]:8d} {cm[i][1]:8d} {cm[i][2]:14d}")
    
    # Per-class accuracy
    print("\nPer-Class Accuracy:")
    for i, name in enumerate(class_names):
        class_acc = cm[i][i] / sum(cm[i]) * 100
        print(f"  {name}: {class_acc:.1f}%")
    
    # Feature importance
    print("\nTop 10 Features:")
    importance = list(zip(extractor.feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, imp in importance[:10]:
        print(f"  {name}: {imp*100:.2f}%")
    
    # Save model
    model_dir = "/workspaces/Veritas/training/models/TriClass"
    os.makedirs(model_dir, exist_ok=True)
    
    with open(f"{model_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    metadata = {
        "model_name": "TriClass",
        "version": "1.0.0",
        "description": "3-way classifier: Human vs AI vs Humanized AI",
        "classes": ["Human", "AI", "Humanized"],
        "feature_names": extractor.feature_names,
        "training_samples": len(X_train),
        "results": {
            "accuracy": float(accuracy),
            "per_class": {
                "Human": float(cm[0][0] / sum(cm[0])),
                "AI": float(cm[1][1] / sum(cm[1])),
                "Humanized": float(cm[2][2] / sum(cm[2])),
            }
        },
        "feature_importance": {name: float(imp) for name, imp in importance},
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(f"{model_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_dir}")
    
    # Test examples
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)
    
    examples = {
        "Pure Human": """
            So I was thinking about this yesterday, right? My friend Dave told me about this crazy thing 
            that happened at work. Honestly, I couldn't believe it. Like, who does that? We ended up 
            laughing about it for hours. The whole situation was just absurd and hilarious.
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
        probs = model.predict_proba(scaled)[0]
        pred_class = class_names[np.argmax(probs)]
        
        print(f"\n{name}:")
        print(f"  Prediction: {pred_class}")
        print(f"  Human: {probs[0]:.1%}  |  AI: {probs[1]:.1%}  |  Humanized: {probs[2]:.1%}")
    
    return model, scaler, extractor, metadata


if __name__ == "__main__":
    train_three_way()
