#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v6 - Balanced Precision
========================================
Fix v5's over-lenience while maintaining good calibration.

Strategy:
1. Use the proper 85-feature extractor (FeatureExtractorV3)
2. Train on balanced RAID data WITHOUT fake templates
3. Use moderate regularization (not extreme like v5)
4. Standard 50% threshold with calibrated probabilities
5. Focus on accurate probability estimation, not threshold manipulation

This should give us:
- High accuracy on clear AI content
- Low false positives on casual human writing  
- Reasonable detection on formal AI content
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_extractor_v3 import FeatureExtractorV3

print("=" * 70)
print("SUPERNOVA ULTRA v6 - Balanced Precision")
print("=" * 70)

def load_arxiv(n=15000):
    """Load arXiv abstracts - academic human writing"""
    print("  Loading arXiv abstracts...")
    try:
        from datasets import load_dataset
        from tqdm import tqdm
        
        samples = []
        ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
        
        for item in tqdm(ds, desc="  arXiv", total=n+100):
            if 'abstract' in item and len(item['abstract']) > 100:
                samples.append(("arxiv", item['abstract'][:2000], 0))
            if len(samples) >= n:
                break
        
        print(f"    ✓ {len(samples)} arXiv abstracts")
        return samples
    except Exception as e:
        print(f"    ✗ arXiv failed: {e}")
        return []

def load_cnn(n=15000):
    """Load CNN/DailyMail - journalistic human writing"""
    print("  Loading CNN/DailyMail...")
    try:
        from datasets import load_dataset
        from tqdm import tqdm
        
        samples = []
        ds = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
        
        for item in tqdm(ds, desc="  CNN/DM", total=n+100):
            if 'article' in item and len(item['article']) > 200:
                samples.append(("cnn", item['article'][:2000], 0))
            if len(samples) >= n:
                break
        
        print(f"    ✓ {len(samples)} news articles")
        return samples
    except Exception as e:
        print(f"    ✗ CNN failed: {e}")
        return []

def load_raid():
    """Load RAID dataset - balanced human and AI"""
    print("  Loading RAID from HuggingFace...")
    try:
        from datasets import load_dataset
        from tqdm import tqdm
        
        human_samples = []
        ai_samples = []
        
        ds = load_dataset("liamdugan/raid", split="train", streaming=True)
        
        for item in tqdm(ds, desc="  RAID"):
            if 'generation' not in item or len(item['generation']) < 100:
                continue
            
            text = item['generation'][:2000]
            model = item.get('model', '')
            
            is_human = model.lower() == "human"
            
            if is_human:
                if len(human_samples) < 40000:
                    human_samples.append(("raid_human", text, 0))
            else:
                if len(ai_samples) < 100000:
                    ai_samples.append(("raid_ai", text, 1))
            
            if len(human_samples) >= 40000 and len(ai_samples) >= 100000:
                break
        
        print(f"    AI samples: {len(ai_samples)}")
        print(f"    RAID human: {len(human_samples)}")
        
        return human_samples, ai_samples
        
    except Exception as e:
        print(f"  RAID failed: {e}")
        return [], []

def main():
    # Phase 1: Load human data
    print("\n📚 PHASE 1: Loading human writing sources...")
    arxiv = load_arxiv(15000)
    cnn = load_cnn(15000)
    
    # Phase 2: Load RAID
    print("\n📊 PHASE 2: Loading RAID dataset...")
    raid_human, raid_ai = load_raid()
    
    # Phase 3: Combine with 1:1 balance
    print("\n⚖️ PHASE 3: Balancing datasets...")
    all_human = arxiv + cnn + raid_human
    all_ai = raid_ai
    
    n_human = len(all_human)
    n_ai = min(len(all_ai), n_human)
    
    random.shuffle(all_ai)
    all_ai = all_ai[:n_ai]
    
    all_data = all_human + all_ai
    random.shuffle(all_data)
    
    print(f"  Human: {n_human}, AI: {n_ai}, Total: {len(all_data)}")
    
    # Phase 4: Extract 85 features
    print("\n📊 PHASE 4: Extracting 85 features...")
    from tqdm import tqdm
    extractor = FeatureExtractorV3()
    
    features = []
    labels = []
    sources = []
    
    for source, text, label in tqdm(all_data, desc="  Features"):
        try:
            feat = extractor.extract_features(text)
            if feat and len(feat) == 85:
                features.append(list(feat.values()))
                labels.append(label)
                sources.append(source)
        except:
            continue
    
    X = np.array(features)
    y = np.array(labels)
    
    print(f"  Matrix: {X.shape}, Human: {sum(1 for l in y if l==0)}, AI: {sum(1 for l in y if l==1)}")
    
    # Phase 5: Prepare data
    print("\n⚙️ PHASE 5: Preparing data...")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X_train, X_test, y_train, y_test, sources_train, sources_test = train_test_split(
        X, y, sources, test_size=0.15, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Phase 6: Train XGBoost with MODERATE regularization
    print("\n🚀 PHASE 6: Training XGBoost with moderate regularization...")
    import xgboost as xgb
    
    # Moderate regularization - not too aggressive
    model = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=8,  # Moderate depth
        learning_rate=0.025,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.2,  # Moderate
        reg_alpha=0.3,  # Moderate L1
        reg_lambda=1.5,  # Moderate L2
        min_child_weight=5,  # Moderate
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        early_stopping_rounds=100
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=100
    )
    
    # Phase 7: Evaluate with 50% threshold
    print("\n📊 PHASE 7: Evaluating...")
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    
    THRESHOLD = 0.50  # Standard threshold
    
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_proba >= THRESHOLD).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # High confidence
    high_conf_mask = (y_proba <= 0.20) | (y_proba >= 0.80)
    high_conf_acc = accuracy_score(y_test[high_conf_mask], y_pred[high_conf_mask]) if sum(high_conf_mask) > 0 else 0
    
    print("\n" + "=" * 70)
    print(f"RESULTS - SUPERNOVA ULTRA v6 (threshold={int(THRESHOLD*100)}%)")
    print("=" * 70)
    print(f"  Accuracy:           {accuracy*100:.2f}%")
    print(f"  High-Conf Accuracy: {high_conf_acc*100:.2f}% ({sum(high_conf_mask)} samples)")
    print(f"  Precision:          {precision*100:.2f}%")
    print(f"  Recall:             {recall*100:.2f}%")
    print(f"  F1 Score:           {f1*100:.2f}%")
    print(f"  ROC AUC:            {roc_auc*100:.2f}%")
    print(f"  FPR:                {fpr*100:.2f}%")
    print(f"  FNR:                {fnr*100:.2f}%")
    
    # Per-source accuracy
    print("\n📈 ACCURACY BY SOURCE:")
    source_test = np.array(sources_test)
    for src in sorted(set(sources_test)):
        mask = source_test == src
        if sum(mask) > 0:
            src_preds = y_pred[mask]
            src_labels = y_test[mask]
            src_acc = accuracy_score(src_labels, src_preds)
            print(f"    {src:15s} {src_acc*100:.2f}% ({sum(mask)} samples)")
    
    # Phase 8: Save
    print("\n💾 PHASE 8: Saving...")
    save_dir = "./models/SupernovaUltraV6"
    os.makedirs(save_dir, exist_ok=True)
    
    model.save_model(f"{save_dir}/model.json")
    with open(f"{save_dir}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{save_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    feature_names = list(extractor.extract_features("test").keys())
    
    metadata = {
        "model": "SUPERNOVA ULTRA v6",
        "version": "6.0.0",
        "threshold": THRESHOLD,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "fpr": fpr,
        "fnr": fnr,
        "high_confidence_accuracy": high_conf_acc,
        "training_samples": len(all_data),
        "human_samples": n_human,
        "ai_samples": n_ai,
        "features": len(feature_names),
        "feature_extractor": "FeatureExtractorV3 (85 features)"
    }
    
    with open(f"{save_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Phase 9: Calibration test
    print(f"\n🧪 PHASE 9: Calibration tests...")
    
    test_samples = [
        # Clear AI
        ("ChatGPT-style", """Certainly! Here's a comprehensive overview of the topic you've requested. First, it's important to understand that there are several key factors to consider. Let me break this down into manageable sections for you. The first aspect involves understanding the fundamental principles. Additionally, it's worth noting that many experts in the field have highlighted the importance of this approach. In conclusion, I hope this explanation has been helpful in clarifying the matter for you."""),
        ("AI Listicle", """Here are 5 key benefits of regular exercise:
1. Improved cardiovascular health
2. Enhanced mental well-being
3. Better sleep quality
4. Increased energy levels
5. Weight management
In conclusion, incorporating regular exercise into your daily routine can significantly improve your overall quality of life."""),
        
        # Casual Human
        ("Reddit comment", """lmao yeah I tried that once and it completely backfired. my boss was NOT happy about it. learned my lesson tho, never again. anyway does anyone know if the new update fixed that bug? been waiting for weeks"""),
        ("Casual email", """hey just wanted to check in about friday - are we still on for dinner? tom said he might be late bc of work stuff but should make it by 7ish. let me know if that works or if we need to push it back"""),
        
        # Formal AI
        ("Model UN (AI)", """Distinguished delegates, the committee must recognize that sustainable development cannot be achieved without addressing systemic inequalities in global trade frameworks. Our nation proposes a comprehensive resolution that balances economic growth with environmental stewardship while respecting national sovereignty."""),
        ("Debate (AI)", """The opposition's argument fundamentally mischaracterizes our position. While they claim economic growth requires environmental sacrifice, the evidence demonstrates otherwise. Consider Denmark's green transition: GDP grew 40% while emissions fell 35%. This proves sustainable development is not just possible but profitable."""),
    ]
    
    print(f"\n  Sample calibration (50% threshold):")
    for name, text in test_samples:
        features = extractor.extract_features(text)
        X_sample = scaler.transform([list(features.values())])
        prob = model.predict_proba(X_sample)[0][1]
        pred = "AI" if prob >= THRESHOLD else "HUMAN"
        print(f"    {name:20s}: {prob*100:5.1f}% AI → {pred}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
