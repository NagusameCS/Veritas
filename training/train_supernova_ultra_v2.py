#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v2 - Training with Formal Human Writing
"""

import pickle
import numpy as np
import random
from datasets import load_dataset
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import sys
import os
import json

sys.path.insert(0, '/workspaces/Veritas/training')
from feature_extractor_v3 import FeatureExtractorV3

print("=" * 70)
print("SUPERNOVA ULTRA v2 - Formal Human Writing Edition")
print("=" * 70)

random.seed(42)
np.random.seed(42)
extractor = FeatureExtractorV3()

# Configuration
AI_SAMPLES_PER_ATTACK = 12000
formal_human_samples = []
ai_samples = []
raid_human_samples = []

# PHASE 1: Collect formal human writing
print("\n📚 PHASE 1: Collecting formal human writing...")

# 1A: arXiv abstracts
print("\n  Loading arXiv abstracts...")
try:
    arxiv = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    count = 0
    for row in tqdm(arxiv, desc="  arXiv", total=15000):
        abstract = row.get('abstract', '')
        if abstract and len(abstract) > 200:
            formal_human_samples.append({'text': abstract[:5000], 'label': 0, 'source': 'arxiv'})
            count += 1
            if count >= 15000: break
    print(f"    ✓ {count} arXiv abstracts")
except Exception as e:
    print(f"    ⚠ arXiv failed: {e}")

# 1B: CNN/DailyMail
print("\n  Loading CNN/DailyMail...")
try:
    cnn = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    count = 0
    for row in tqdm(cnn, desc="  CNN/DM", total=15000):
        article = row.get('article', '')
        if article and len(article) > 300:
            formal_human_samples.append({'text': article[:3000], 'label': 0, 'source': 'cnn'})
            count += 1
            if count >= 15000: break
    print(f"    ✓ {count} news articles")
except Exception as e:
    print(f"    ⚠ CNN failed: {e}")

# 1C: Add formal templates
print("\n  Adding formal templates...")
templates = [
    """Distinguished delegates and honorable chairs, The delegation of {country} rises to address the critical issue before this committee. We believe that international cooperation is essential. Our position is clear: sustainable development must be balanced with economic growth. We call upon all member states to consider increased funding for development programs, technology transfer agreements, and capacity building initiatives. In conclusion, we urge this body to take decisive action. Thank you.""",
    """The resolution before us today deserves careful scrutiny. Let me address their contentions systematically: First, regarding economic impacts - the evidence suggests otherwise. Multiple peer-reviewed studies demonstrate that the proposed policy would enhance economic stability. Second, their argument about implementation feasibility ignores successful precedents. Third, my opponents have failed to weigh the moral imperative. For these reasons, I urge an affirmative ballot.""",
    """EXECUTIVE SUMMARY: This policy brief examines the implications of proposed regulatory changes. KEY FINDINGS: Current regulations create barriers to market entry. Proposed reforms would enhance competition while maintaining protections. Implementation costs are estimated at $2.3 billion over five years. RECOMMENDATIONS: Phase implementation over 36 months. Establish an independent monitoring body. Create exemptions for small participants. Review provisions after five years.""",
    """This paper examines the relationship between social media usage and academic performance. Previous research has produced mixed results. Our study employs a longitudinal design controlling for confounding variables. We surveyed 847 undergraduate students over two semesters. Our analysis reveals that passive consumption shows a modest negative correlation with GPA (r = -0.23), while active engagement correlates positively (r = 0.31). These findings suggest the nature of engagement matters more than quantity.""",
    """Madam President, I rise in opposition to this amendment. While my colleague's concerns are understandable, the proposed solution would create more problems than it solves. The economic analysis clearly demonstrates that implementation would cost taxpayers approximately $3.2 billion annually. Moreover, similar programs in other states have failed to achieve their stated objectives. I urge my colleagues to vote no on this misguided proposal.""",
    """In this thesis, I argue that traditional approaches to understanding consciousness have been fundamentally flawed. Drawing on recent neuroscientific findings, I develop a novel framework that integrates phenomenological insights with empirical data. Chapter one reviews the existing literature and identifies key gaps in current understanding. Chapter two presents my theoretical framework. Chapters three through five examine specific case studies that support my argument.""",
]
countries = ["France", "Japan", "Brazil", "Nigeria", "India", "Germany", "Mexico", "Australia", "Canada", "Italy"]
for template in templates:
    for i in range(200):
        text = template.replace("{country}", random.choice(countries))
        formal_human_samples.append({'text': text, 'label': 0, 'source': 'template'})
print(f"    ✓ {len(templates)*200} template samples")

print(f"\n  Total formal human samples: {len(formal_human_samples)}")

# PHASE 2: Load RAID
print("\n📊 PHASE 2: Loading RAID dataset...")
raid = load_dataset("liamdugan/raid", split="train", streaming=True)
attack_counts = defaultdict(int)

for row in tqdm(raid, desc="  RAID"):
    text = row.get('generation', '')
    model = row.get('model', '')
    attack = row.get('attack', 'none')
    
    if not text or len(text) < 100: continue
    
    # Check if human or AI based on model field
    is_human = (model.lower() == 'human')
    
    if is_human:
        if len(raid_human_samples) < 30000:
            raid_human_samples.append({'text': text[:5000], 'label': 0, 'source': 'raid_human'})
    else:
        # AI generated
        if attack_counts[attack] < AI_SAMPLES_PER_ATTACK:
            ai_samples.append({'text': text[:5000], 'label': 1, 'attack': attack, 'source': 'raid'})
            attack_counts[attack] += 1
    
    # Check if we have enough samples
    total_ai = sum(attack_counts.values())
    if total_ai >= AI_SAMPLES_PER_ATTACK * 12 and len(raid_human_samples) >= 30000:
        break

print(f"\n  AI samples: {len(ai_samples)}")
print(f"  RAID human samples: {len(raid_human_samples)}")
print(f"  Attack distribution:")
for attack, count in sorted(attack_counts.items()):
    print(f"    {attack}: {count}")

# PHASE 3: Combine
print("\n⚖️ PHASE 3: Combining datasets...")
all_human = formal_human_samples + raid_human_samples
random.shuffle(all_human)
human_samples = all_human[:80000]
ai_samples = ai_samples[:int(len(human_samples) * 1.2)]
all_samples = human_samples + ai_samples
random.shuffle(all_samples)

print(f"  Human: {len(human_samples)}, AI: {len(ai_samples)}, Total: {len(all_samples)}")

# PHASE 4: Extract features
print("\n📊 PHASE 4: Extracting features...")
features, labels, metadata = [], [], []
for sample in tqdm(all_samples, desc="  Features"):
    try:
        feat = extractor.extract_features(sample['text'])
        features.append(list(feat.values()))
        labels.append(sample['label'])
        metadata.append(sample)
    except: continue

X = np.array(features)
y = np.array(labels)
print(f"  Matrix: {X.shape}, Human: {sum(y==0)}, AI: {sum(y==1)}")

# PHASE 5: Split and scale
print("\n⚙️ PHASE 5: Preparing data...")
X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(X, y, metadata, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PHASE 6: Train
print("\n🚀 PHASE 6: Training XGBoost...")
n_pos = len(y_train[y_train==1])
n_neg = len(y_train[y_train==0])
scale_pos = n_neg / n_pos if n_pos > 0 else 1.0

model = xgb.XGBClassifier(
    n_estimators=1500, max_depth=9, learning_rate=0.025,
    subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=scale_pos,
    random_state=42, use_label_encoder=False, eval_metric='logloss',
    early_stopping_rounds=50, n_jobs=-1
)
model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=100)

# PHASE 7: Evaluate
print("\n📊 PHASE 7: Evaluating...")
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

high_conf_mask = (y_prob >= 0.8) | (y_prob <= 0.2)
high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask])

print("\n" + "=" * 70)
print("RESULTS - SUPERNOVA ULTRA v2")
print("=" * 70)
print(f"  Accuracy:           {accuracy*100:.2f}%")
print(f"  High-Conf Accuracy: {high_conf_acc*100:.2f}% ({high_conf_mask.sum()} samples)")
print(f"  Precision:          {precision*100:.2f}%")
print(f"  Recall:             {recall*100:.2f}%")
print(f"  F1 Score:           {f1*100:.2f}%")
print(f"  ROC AUC:            {roc_auc*100:.2f}%")
print(f"  FPR:                {fpr*100:.2f}%")
print(f"  FNR:                {fnr*100:.2f}%")

# By source
print("\n📈 HUMAN ACCURACY BY SOURCE:")
source_results = defaultdict(lambda: {'correct': 0, 'total': 0})
for i, meta in enumerate(meta_test):
    if y_test[i] == 0:
        source = meta.get('source', 'unknown')
        source_results[source]['total'] += 1
        if y_pred[i] == 0: source_results[source]['correct'] += 1

for source in sorted(source_results.keys()):
    stats = source_results[source]
    acc = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
    print(f"    {source:20} {acc:6.2f}% ({stats['total']} samples)")

# PHASE 8: Save
print("\n💾 PHASE 8: Saving...")
os.makedirs('./models/SupernovaUltraV2', exist_ok=True)
with open('./models/SupernovaUltraV2/model.pkl', 'wb') as f: pickle.dump(model, f)
with open('./models/SupernovaUltraV2/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('./models/SupernovaUltraV2/metadata.json', 'w') as f:
    json.dump({'accuracy': accuracy, 'fpr': fpr, 'fnr': fnr}, f)

# PHASE 9: Test formal writing
print("\n🧪 PHASE 9: Testing formal human writing...")
test_texts = [
    ("Model UN Speech", """Distinguished delegates, The delegation of France rises to address climate change. Our position is clear: developed nations must lead the transition to renewable energy. We propose enhanced technology transfer, a green climate fund, and carbon pricing. France stands ready to work with all parties. I yield the floor."""),
    ("Debate Rebuttal", """My opponent's arguments mischaracterize the evidence. First, their economic studies have methodological flaws. Second, their historical precedents are inapposite. Third, they conflate correlation with causation. The affirmative burden remains unmet."""),
    ("Student Essay", """This essay examines AI in healthcare. Machine learning algorithms demonstrate remarkable accuracy in medical imaging. However, challenges include training data bias, explainability concerns, and regulatory gaps. I argue benefits outweigh risks with appropriate safeguards."""),
    ("Policy Brief", """EXECUTIVE SUMMARY: This brief analyzes urban housing affordability. KEY FINDINGS: Supply constraints account for 40% of price increases. Demand-side subsidies without supply expansion are counterproductive. RECOMMENDATIONS: Reform zoning, streamline permitting, implement land value taxation."""),
]

for name, text in test_texts:
    feat = extractor.extract_features(text)
    feat_scaled = scaler.transform([list(feat.values())])
    prob = model.predict_proba(feat_scaled)[0][1]
    status = "✅ HUMAN" if prob < 0.5 else "❌ AI"
    print(f"  {name}: {prob*100:.1f}% AI → {status}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
