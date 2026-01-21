#!/usr/bin/env python3
"""
VERITAS Ultimate Detector v3 - C4 Focus
Target: 99%+ accuracy by solving C4 (web content) classification
"""

import json
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("VERITAS Ultimate v3 - Solving C4")
print("=" * 70)

from sentence_transformers import SentenceTransformer
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Loaded embedding model")

# =============================================================================
# C4 ANALYSIS: Why does it fail?
# C4 = Common Crawl curated. Formal, polished web text.
# Problem: formal/polished like AI but written by humans
# Solution: Find signals unique to human web writing
# =============================================================================

def extract_features(text):
    """60+ features with C4-specific additions."""
    if not text or len(text) < 10:
        return None
    
    words = text.split()
    word_count = len(words) if words else 1
    char_count = len(text)
    
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_count = len(sentences) if sentences else 1
    
    paragraphs = text.split('\n\n')
    para_count = len([p for p in paragraphs if p.strip()])
    
    features = {}
    
    # =========================================================================
    # C4 SPECIFIC - Web content patterns unique to HUMAN web writing
    # =========================================================================
    
    # Real-world references (dates, times, locations)
    features['specific_dates'] = len(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}', text, re.I))
    features['specific_times'] = len(re.findall(r'\b\d{1,2}:\d{2}(?:\s*[ap]m)?\b', text, re.I))
    features['specific_years'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
    features['currency'] = len(re.findall(r'\$\d+|\d+\s*(?:dollars?|USD|EUR|GBP)', text, re.I))
    features['percentages'] = len(re.findall(r'\d+\.?\d*\s*%', text))
    
    # Contact/location info
    features['has_email'] = 1 if re.search(r'[\w.-]+@[\w.-]+\.\w+', text) else 0
    features['has_phone'] = 1 if re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text) else 0
    features['has_address'] = 1 if re.search(r'\b\d+\s+\w+\s+(St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Dr|Drive)\b', text, re.I) else 0
    
    # URLs and links (web content)
    features['url_count'] = len(re.findall(r'https?://\S+|www\.\S+', text))
    
    # Specific nouns/names (more in human content)
    features['all_caps_words'] = len(re.findall(r'\b[A-Z]{2,}\b', text)) / word_count
    features['title_case_sequences'] = len(re.findall(r'\b(?:[A-Z][a-z]+\s+){2,}[A-Z][a-z]+\b', text))
    
    # Incomplete sentences/fragments (human web style)
    features['sentence_fragments'] = len([s for s in sentences if len(s.split()) < 4])
    features['very_short_paragraphs'] = len([p for p in paragraphs if p.strip() and len(p.split()) < 10])
    
    # =========================================================================
    # HUMAN AUTHENTICITY SIGNALS
    # =========================================================================
    
    # Typos and informal spelling (very human)
    features['double_letters'] = len(re.findall(r'(.)\1{2,}', text))  # sooo, reallyyy
    features['informal_spelling'] = len(re.findall(r'\b(ur|u|r|cuz|bcuz|tho|thru|pls|plz)\b', text, re.I))
    
    # Human speech patterns
    features['filler_words'] = len(re.findall(r'\b(um|uh|er|well|like|you know|I mean|kind of|sort of)\b', text, re.I)) / word_count
    features['self_corrections'] = len(re.findall(r'\bI mean\b|--\s*I|wait,?\s+\w', text, re.I))
    
    # Emotional/personal interjections
    features['interjections'] = len(re.findall(r'\b(wow|oh|ah|oops|yay|ugh|hmm|huh|geez|gosh|dang|damn)\b', text, re.I))
    
    # =========================================================================
    # AI INSTRUCTION PATTERNS
    # =========================================================================
    
    # Explicit instruction format
    features['instruction_format'] = len(re.findall(r'^(?:Step\s+\d|First,|Next,|Finally,|To\s+\w+,|Here\'s how)', text, re.M|re.I))
    features['bullet_points'] = len(re.findall(r'^[\s]*[-•*]\s+', text, re.M))
    
    # Formal transitions (AI loves these)
    features['formal_transitions'] = len(re.findall(r'\b(In conclusion|To summarize|In summary|Overall|As mentioned|As discussed|With that said)\b', text, re.I))
    
    # AI hedging and certainty
    features['ai_hedging'] = len(re.findall(r'\bIt\'s worth noting|It\'s important to|One thing to consider|Keep in mind\b', text, re.I))
    
    # =========================================================================
    # CORE STATISTICAL FEATURES
    # =========================================================================
    
    # Sentence structure
    sent_lengths = [len(s.split()) for s in sentences]
    features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    features['max_sent_len'] = max(sent_lengths) if sent_lengths else 0
    features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
    features['sent_count'] = sent_count
    
    # Word stats
    word_lengths = [len(w) for w in words]
    features['avg_word_len'] = np.mean(word_lengths) if word_lengths else 0
    features['long_word_rate'] = sum(1 for w in words if len(w) > 8) / word_count
    
    # Vocabulary
    unique_words = len(set(w.lower() for w in words))
    features['vocab_richness'] = unique_words / word_count
    
    # Pronouns
    features['first_person'] = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.I)) / word_count
    features['second_person'] = len(re.findall(r'\b(you|your|yours)\b', text, re.I)) / word_count
    features['third_person'] = len(re.findall(r'\b(he|she|they|him|her|them|his|their)\b', text, re.I)) / word_count
    features['we_us'] = len(re.findall(r'\b(we|us|our|ours)\b', text, re.I)) / word_count
    
    # Punctuation
    features['comma_rate'] = text.count(',') / sent_count
    features['semicolon_count'] = text.count(';')
    features['colon_rate'] = text.count(':') / sent_count
    features['exclamation_rate'] = text.count('!') / sent_count
    features['question_rate'] = text.count('?') / sent_count
    features['dash_rate'] = len(re.findall(r'[-–—]', text)) / sent_count
    features['paren_count'] = text.count('(')
    features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
    features['quote_pairs'] = len(re.findall(r'"[^"]*"|"[^"]*"', text))
    
    # Contractions
    contractions = re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)
    features['contraction_rate'] = len(contractions) / word_count
    
    # Discourse markers
    features['discourse_rate'] = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless)\b', text, re.I)) / word_count
    
    # Attribution
    features['attribution_count'] = len(re.findall(r'\b(said|says|told|asked|noted|added|stated|claimed|according to)\b', text, re.I))
    
    # Casual language
    features['casual_count'] = len(re.findall(r'\b(lol|haha|omg|yeah|nah|ok|okay|hey|hi|thanks|gonna|wanna)\b', text, re.I))
    
    # Emotional words
    features['emotional_rate'] = len(re.findall(r'\b(love|hate|amazing|awesome|terrible|horrible|wonderful|great|bad|worst|best)\b', text, re.I)) / word_count
    
    # Sentence starters
    features['sent_start_I'] = len(re.findall(r'(?:^|[.!?]\s+)I\s', text))
    features['sent_start_The'] = len(re.findall(r'[.!?]\s+The\s', text))
    
    # Response patterns
    features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Certainly|Of course|Absolutely)\b', text.strip()) else 0
    features['helpful_phrases'] = len(re.findall(r'\b(here is|here are|feel free|let me|I hope this|I can help)\b', text, re.I))
    
    # Structural
    features['word_count'] = word_count
    features['para_count'] = para_count
    
    # Technical
    features['has_code'] = 1 if re.search(r'```|def\s+\w+|function\s*\(|class\s+\w+', text) else 0
    features['has_markdown'] = 1 if re.search(r'^#+\s|\*\*\w', text, re.M) else 0
    
    return features


# =============================================================================
# LOAD DATA
# =============================================================================

print("\nLoading dataset...")
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

print(f"Total: {len(samples)}")

# Extract features
print("\nExtracting features...")
X_heuristic = []
X_embedding = []
y_data = []
sources = []

batch_size = 500
total = len(samples)

for i in range(0, total, batch_size):
    if i % 20000 == 0:
        print(f"  {i}/{total}...")
    
    batch = samples[i:i+batch_size]
    texts = []
    
    for s in batch:
        text = s.get('text', '')
        label = s.get('label', '')
        source = s.get('source', 'unknown')
        
        if not text or not label:
            continue
        
        h_feat = extract_features(text)
        if h_feat is None:
            continue
        
        X_heuristic.append(list(h_feat.values()))
        texts.append(text[:2000])
        y_data.append(1 if label == 'ai' else 0)
        sources.append(source)
    
    if texts:
        embeddings = embed_model.encode(texts, show_progress_bar=False, batch_size=64)
        X_embedding.extend(embeddings)

feature_names = list(extract_features("Sample text for feature names.").keys())
print(f"\nExtracted {len(X_heuristic)} samples")
print(f"Heuristic features: {len(feature_names)}")

# Combine
X_heuristic = np.array(X_heuristic, dtype=np.float32)
X_embedding = np.array(X_embedding, dtype=np.float32)
X_combined = np.hstack([X_heuristic, X_embedding])
y = np.array(y_data)

X_combined = np.nan_to_num(X_combined, nan=0, posinf=0, neginf=0)

print(f"Total features: {X_combined.shape[1]}")

# =============================================================================
# TRAIN
# =============================================================================

X_train, X_test, y_train, y_test, src_train, src_test = train_test_split(
    X_combined, y, sources, test_size=0.15, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

print("\n" + "=" * 70)
print("Training XGBoost (more trees, deeper)")
print("=" * 70)

model = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=18,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0.02,
    reg_alpha=0.005,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss',
    tree_method='hist'
)

model.fit(X_train_scaled, y_train, 
          eval_set=[(X_test_scaled, y_test)], 
          verbose=100)

# =============================================================================
# EVALUATION
# =============================================================================

print("\n" + "=" * 70)
print("EVALUATION")
print("=" * 70)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*40}")
print(f"ACCURACY: {acc:.4f} ({acc:.2%})")
print(f"AUC-ROC:  {auc:.4f}")
print(f"{'='*40}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))

cm = confusion_matrix(y_test, y_pred)
print(f"\nHuman Accuracy: {cm[0][0]/(cm[0][0]+cm[0][1]):.2%}")
print(f"AI Accuracy: {cm[1][1]/(cm[1][0]+cm[1][1]):.2%}")

# By source
print("\n" + "-" * 50)
print("Accuracy by Source:")
print("-" * 50)
source_results = []
for source in sorted(set(src_test)):
    mask = [s == source for s in src_test]
    if sum(mask) > 50:
        src_acc = accuracy_score(np.array(y_test)[mask], np.array(y_pred)[mask])
        count = sum(mask)
        label = 'AI' if np.array(y_test)[mask].mean() > 0.5 else 'Human'
        source_results.append((source, src_acc, count, label))

source_results.sort(key=lambda x: x[1])
for source, src_acc, count, label in source_results:
    status = '✓' if src_acc >= 0.95 else '○' if src_acc >= 0.90 else '✗'
    print(f"  {status} {source:25s}: {src_acc:.1%} ({count:5d}) [{label}]")

# Save
print("\n" + "=" * 70)
with open('veritas_ultimate_v3.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'feature_names': feature_names, 'accuracy': acc}, f)

print(f"Saved to 'veritas_ultimate_v3.pkl'")
print(f"\n{'='*70}")
print(f"FINAL ACCURACY: {acc:.2%}")
print(f"{'='*70}")
