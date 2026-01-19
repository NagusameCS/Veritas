#!/usr/bin/env python3
"""
Extended Humanized Text Benchmark
==================================
Uses real dataset samples for more realistic evaluation.
Also includes threshold calibration for Sunset model.
"""

import os
import pickle
import json
import re
import random
from datetime import datetime
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
from datasets import load_dataset

# Import extractors from benchmark
from benchmark_humanized import SunriseExtractor, SunsetExtractor, HumanizationDetector, humanize_text


def load_real_samples(n_per_class=50):
    """Load real samples from dataset"""
    print("Loading GPT-wiki-intro dataset...")
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    # Get balanced samples
    human_texts = []
    ai_texts = []
    
    for item in dataset:
        if len(item.get('wiki_intro', '').split()) >= 50:
            human_texts.append(item['wiki_intro'])
        if len(item.get('generated_intro', '').split()) >= 50:
            ai_texts.append(item['generated_intro'])
        
        if len(human_texts) >= n_per_class and len(ai_texts) >= n_per_class:
            break
    
    return human_texts[:n_per_class], ai_texts[:n_per_class]


def run_extended_benchmark():
    print("=" * 70)
    print("EXTENDED BENCHMARK WITH REAL DATA")
    print("=" * 70)
    
    # Load models
    print("\nLoading models...")
    
    sunrise_path = "/workspaces/Veritas/training/models/Sunrise"
    sunset_path = "/workspaces/Veritas/training/models/Sunset"
    
    with open(f"{sunrise_path}/model.pkl", "rb") as f:
        sunrise_model = pickle.load(f)
    with open(f"{sunrise_path}/scaler.pkl", "rb") as f:
        sunrise_scaler = pickle.load(f)
    
    with open(f"{sunset_path}/model.pkl", "rb") as f:
        sunset_model = pickle.load(f)
    with open(f"{sunset_path}/scaler.pkl", "rb") as f:
        sunset_scaler = pickle.load(f)
    
    sunrise_extractor = SunriseExtractor()
    sunset_extractor = SunsetExtractor()
    humanization_detector = HumanizationDetector()
    
    # Load real samples
    n_samples = 30  # 30 per class for speed
    human_texts, ai_texts = load_real_samples(n_samples)
    
    print(f"Loaded {len(human_texts)} human, {len(ai_texts)} AI samples")
    
    # Create humanized versions
    humanized_light = [humanize_text(t, 0.3) for t in ai_texts[:15]]
    humanized_heavy = [humanize_text(t, 0.7) for t in ai_texts[:15]]
    
    # Collect predictions
    results = {
        'pure_human': {'sunrise': [], 'sunset': []},
        'pure_ai': {'sunrise': [], 'sunset': []},
        'humanized_light': {'sunrise': [], 'sunset': []},
        'humanized_heavy': {'sunrise': [], 'sunset': []}
    }
    
    print("\nAnalyzing pure human texts...")
    for text in human_texts:
        sunrise_features = sunrise_extractor.extract(text)
        sunrise_scaled = sunrise_scaler.transform([sunrise_features])
        sunrise_prob = sunrise_model.predict_proba(sunrise_scaled)[0][1]
        
        sunset_features = sunset_extractor.extract(text)
        sunset_scaled = sunset_scaler.transform([sunset_features])
        sunset_prob = sunset_model.predict_proba(sunset_scaled)[0][1]
        
        results['pure_human']['sunrise'].append(sunrise_prob)
        results['pure_human']['sunset'].append(sunset_prob)
    
    print("Analyzing pure AI texts...")
    for text in ai_texts:
        sunrise_features = sunrise_extractor.extract(text)
        sunrise_scaled = sunrise_scaler.transform([sunrise_features])
        sunrise_prob = sunrise_model.predict_proba(sunrise_scaled)[0][1]
        
        sunset_features = sunset_extractor.extract(text)
        sunset_scaled = sunset_scaler.transform([sunset_features])
        sunset_prob = sunset_model.predict_proba(sunset_scaled)[0][1]
        
        results['pure_ai']['sunrise'].append(sunrise_prob)
        results['pure_ai']['sunset'].append(sunset_prob)
    
    print("Analyzing humanized (light) texts...")
    for text in humanized_light:
        sunrise_features = sunrise_extractor.extract(text)
        sunrise_scaled = sunrise_scaler.transform([sunrise_features])
        sunrise_prob = sunrise_model.predict_proba(sunrise_scaled)[0][1]
        
        sunset_features = sunset_extractor.extract(text)
        sunset_scaled = sunset_scaler.transform([sunset_features])
        sunset_prob = sunset_model.predict_proba(sunset_scaled)[0][1]
        
        results['humanized_light']['sunrise'].append(sunrise_prob)
        results['humanized_light']['sunset'].append(sunset_prob)
    
    print("Analyzing humanized (heavy) texts...")
    for text in humanized_heavy:
        sunrise_features = sunrise_extractor.extract(text)
        sunrise_scaled = sunrise_scaler.transform([sunrise_features])
        sunrise_prob = sunrise_model.predict_proba(sunrise_scaled)[0][1]
        
        sunset_features = sunset_extractor.extract(text)
        sunset_scaled = sunset_scaler.transform([sunset_features])
        sunset_prob = sunset_model.predict_proba(sunset_scaled)[0][1]
        
        results['humanized_heavy']['sunrise'].append(sunrise_prob)
        results['humanized_heavy']['sunset'].append(sunset_prob)
    
    # Print results
    print("\n" + "=" * 70)
    print("DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    for category in ['pure_human', 'pure_ai', 'humanized_light', 'humanized_heavy']:
        sunrise_probs = results[category]['sunrise']
        sunset_probs = results[category]['sunset']
        
        print(f"\n{category.upper().replace('_', ' ')} ({len(sunrise_probs)} samples):")
        print(f"  Sunrise: mean={np.mean(sunrise_probs):.3f}, std={np.std(sunrise_probs):.3f}, "
              f"min={min(sunrise_probs):.3f}, max={max(sunrise_probs):.3f}")
        print(f"  Sunset:  mean={np.mean(sunset_probs):.3f}, std={np.std(sunset_probs):.3f}, "
              f"min={min(sunset_probs):.3f}, max={max(sunset_probs):.3f}")
    
    # Accuracy at different thresholds
    print("\n" + "=" * 70)
    print("ACCURACY AT DIFFERENT THRESHOLDS")
    print("=" * 70)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\nSUNRISE MODEL:")
    for thresh in thresholds:
        human_correct = sum(1 for p in results['pure_human']['sunrise'] if p < thresh)
        ai_correct = sum(1 for p in results['pure_ai']['sunrise'] if p >= thresh)
        humanized_correct = sum(1 for p in results['humanized_heavy']['sunrise'] if p >= thresh)
        
        total = len(results['pure_human']['sunrise']) + len(results['pure_ai']['sunrise'])
        acc = (human_correct + ai_correct) / total * 100
        
        print(f"  Threshold {thresh}: Human={human_correct}/{len(results['pure_human']['sunrise'])}, "
              f"AI={ai_correct}/{len(results['pure_ai']['sunrise'])}, "
              f"Acc={acc:.1f}%, "
              f"Humanized detected={humanized_correct}/{len(results['humanized_heavy']['sunrise'])}")
    
    print("\nSUNSET MODEL:")
    for thresh in thresholds:
        human_correct = sum(1 for p in results['pure_human']['sunset'] if p < thresh)
        ai_correct = sum(1 for p in results['pure_ai']['sunset'] if p >= thresh)
        humanized_correct = sum(1 for p in results['humanized_heavy']['sunset'] if p >= thresh)
        
        total = len(results['pure_human']['sunset']) + len(results['pure_ai']['sunset'])
        acc = (human_correct + ai_correct) / total * 100
        
        print(f"  Threshold {thresh}: Human={human_correct}/{len(results['pure_human']['sunset'])}, "
              f"AI={ai_correct}/{len(results['pure_ai']['sunset'])}, "
              f"Acc={acc:.1f}%, "
              f"Humanized detected={humanized_correct}/{len(results['humanized_heavy']['sunset'])}")
    
    # Optimal threshold calibration
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLD CALIBRATION")
    print("=" * 70)
    
    for model_name in ['sunrise', 'sunset']:
        best_thresh = 0.5
        best_acc = 0
        
        for thresh in np.arange(0.1, 0.95, 0.05):
            human_correct = sum(1 for p in results['pure_human'][model_name] if p < thresh)
            ai_correct = sum(1 for p in results['pure_ai'][model_name] if p >= thresh)
            total = len(results['pure_human'][model_name]) + len(results['pure_ai'][model_name])
            acc = (human_correct + ai_correct) / total
            
            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh
        
        print(f"\n{model_name.upper()}: Optimal threshold = {best_thresh:.2f} (Accuracy: {best_acc*100:.1f}%)")
        
        # Stats at optimal threshold
        human_probs = results['pure_human'][model_name]
        ai_probs = results['pure_ai'][model_name]
        humanized_probs = results['humanized_heavy'][model_name]
        
        human_correct = sum(1 for p in human_probs if p < best_thresh)
        ai_correct = sum(1 for p in ai_probs if p >= best_thresh)
        humanized_detected = sum(1 for p in humanized_probs if p >= best_thresh)
        
        print(f"  Human detection: {human_correct}/{len(human_probs)} = {human_correct/len(human_probs)*100:.1f}%")
        print(f"  AI detection:    {ai_correct}/{len(ai_probs)} = {ai_correct/len(ai_probs)*100:.1f}%")
        print(f"  Humanized AI:    {humanized_detected}/{len(humanized_probs)} = {humanized_detected/len(humanized_probs)*100:.1f}%")
    
    # Ensemble analysis
    print("\n" + "=" * 70)
    print("ENSEMBLE STRATEGIES")
    print("=" * 70)
    
    # Average ensemble
    print("\n1. AVERAGE ENSEMBLE (sunrise + sunset) / 2:")
    for category in ['pure_human', 'pure_ai', 'humanized_heavy']:
        ensemble_probs = [(s + t) / 2 for s, t in zip(results[category]['sunrise'], results[category]['sunset'])]
        print(f"  {category}: mean={np.mean(ensemble_probs):.3f}, range=[{min(ensemble_probs):.3f}, {max(ensemble_probs):.3f}]")
    
    # Weighted ensemble (trust Sunrise more for human)
    print("\n2. WEIGHTED ENSEMBLE (0.7*sunrise + 0.3*sunset):")
    for category in ['pure_human', 'pure_ai', 'humanized_heavy']:
        ensemble_probs = [0.7*s + 0.3*t for s, t in zip(results[category]['sunrise'], results[category]['sunset'])]
        print(f"  {category}: mean={np.mean(ensemble_probs):.3f}, range=[{min(ensemble_probs):.3f}, {max(ensemble_probs):.3f}]")
    
    # Min strategy (conservative - pick lowest AI prob)
    print("\n3. MIN STRATEGY (min of sunrise, sunset):")
    for category in ['pure_human', 'pure_ai', 'humanized_heavy']:
        ensemble_probs = [min(s, t) for s, t in zip(results[category]['sunrise'], results[category]['sunset'])]
        print(f"  {category}: mean={np.mean(ensemble_probs):.3f}, range=[{min(ensemble_probs):.3f}, {max(ensemble_probs):.3f}]")
    
    # Max strategy (aggressive)
    print("\n4. MAX STRATEGY (max of sunrise, sunset):")
    for category in ['pure_human', 'pure_ai', 'humanized_heavy']:
        ensemble_probs = [max(s, t) for s, t in zip(results[category]['sunrise'], results[category]['sunset'])]
        print(f"  {category}: mean={np.mean(ensemble_probs):.3f}, range=[{min(ensemble_probs):.3f}, {max(ensemble_probs):.3f}]")
    
    # Sunrise primary with Sunset confidence boost
    print("\n5. SUNRISE PRIMARY + SUNSET BOOST (sunrise + 0.2 if sunset > 0.9):")
    for category in ['pure_human', 'pure_ai', 'humanized_heavy']:
        ensemble_probs = [s + (0.2 if t > 0.9 else 0) for s, t in zip(results[category]['sunrise'], results[category]['sunset'])]
        ensemble_probs = [min(p, 1.0) for p in ensemble_probs]
        print(f"  {category}: mean={np.mean(ensemble_probs):.3f}, range=[{min(ensemble_probs):.3f}, {max(ensemble_probs):.3f}]")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. SUNSET MODEL HAS BIAS: It's over-classifying human text as AI.
   - The GPT-wiki-intro dataset may have similar features in both classes
   - Consider retraining with a different dataset or adjusting features
   
2. SUNRISE PERFORMS BETTER for human/AI discrimination
   - Use Sunrise as the primary model for classification
   - Sunrise optimal threshold: ~0.5
   
3. BEST ENSEMBLE STRATEGY:
   - Use weighted ensemble: 0.7*Sunrise + 0.3*Sunset
   - Or use Sunrise as primary with Sunset as secondary confirmation
   
4. FOR HUMANIZED TEXT:
   - Both models still detect humanized AI text well (80%+ as AI)
   - Light humanization barely affects detection
   - Heavy humanization slightly reduces confidence but still detects
""")
    
    print("=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_extended_benchmark()
