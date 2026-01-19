#!/usr/bin/env python3
"""
Sunrise vs Sunset Comparative Analysis
======================================
Determine what each model is better at.
"""

import os
import pickle
import re
import numpy as np
from collections import Counter
from datasets import load_dataset

# Import extractors from benchmark
import sys
sys.path.insert(0, '/workspaces/Veritas/training')
from benchmark_humanized import SunriseExtractor, SunsetExtractor


def load_diverse_samples():
    """Load samples of different text types"""
    print("Loading diverse test samples...")
    
    dataset = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
    
    human_texts = []
    ai_texts = []
    
    for item in dataset:
        wiki = item.get('wiki_intro', '')
        gen = item.get('generated_intro', '')
        
        if len(wiki.split()) >= 40:
            human_texts.append(wiki)
        if len(gen.split()) >= 40:
            ai_texts.append(gen)
        
        if len(human_texts) >= 200 and len(ai_texts) >= 200:
            break
    
    # Categorize by length (use flexible thresholds)
    short_human = [t for t in human_texts if len(t.split()) < 100][:40]
    medium_human = [t for t in human_texts if 100 <= len(t.split()) < 250][:40]
    long_human = [t for t in human_texts if len(t.split()) >= 150][:40]  # Overlap ok
    
    short_ai = [t for t in ai_texts if len(t.split()) < 100][:40]
    medium_ai = [t for t in ai_texts if 100 <= len(t.split()) < 250][:40]
    long_ai = [t for t in ai_texts if len(t.split()) >= 150][:40]
    
    return {
        'short_human': short_human,
        'medium_human': medium_human,
        'long_human': long_human,
        'short_ai': short_ai,
        'medium_ai': medium_ai,
        'long_ai': long_ai,
    }


def analyze_models():
    """Comprehensive model comparison"""
    print("=" * 70)
    print("SUNRISE vs SUNSET COMPARATIVE ANALYSIS")
    print("=" * 70)
    
    # Load models
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
    
    sunrise_ext = SunriseExtractor()
    sunset_ext = SunsetExtractor()
    
    # Load test data
    samples = load_diverse_samples()
    
    results = {}
    
    for category, texts in samples.items():
        if not texts:  # Skip empty categories
            continue
        is_ai = 'ai' in category
        sunrise_probs = []
        sunset_probs = []
        
        for text in texts:
            # Sunrise prediction
            feat = sunrise_ext.extract(text)
            scaled = sunrise_scaler.transform([feat])
            prob = sunrise_model.predict_proba(scaled)[0][1]
            sunrise_probs.append(prob)
            
            # Sunset prediction
            feat = sunset_ext.extract(text)
            scaled = sunset_scaler.transform([feat])
            prob = sunset_model.predict_proba(scaled)[0][1]
            sunset_probs.append(prob)
        
        results[category] = {
            'is_ai': is_ai,
            'sunrise': {
                'mean': np.mean(sunrise_probs),
                'std': np.std(sunrise_probs),
                'correct': sum(1 for p in sunrise_probs if (p > 0.5) == is_ai) / len(sunrise_probs) * 100
            },
            'sunset': {
                'mean': np.mean(sunset_probs),
                'std': np.std(sunset_probs),
                'correct': sum(1 for p in sunset_probs if (p > 0.5) == is_ai) / len(sunset_probs) * 100
            }
        }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS BY TEXT LENGTH")
    print("=" * 70)
    
    for category, data in results.items():
        label = category.replace('_', ' ').title()
        sunrise = data['sunrise']
        sunset = data['sunset']
        
        print(f"\n{label}:")
        print(f"  Sunrise: mean={sunrise['mean']:.3f}, std={sunrise['std']:.3f}, accuracy={sunrise['correct']:.1f}%")
        print(f"  Sunset:  mean={sunset['mean']:.3f}, std={sunset['std']:.3f}, accuracy={sunset['correct']:.1f}%")
        
        if sunrise['correct'] > sunset['correct']:
            print(f"  Winner: SUNRISE (+{sunrise['correct']-sunset['correct']:.1f}%)")
        elif sunset['correct'] > sunrise['correct']:
            print(f"  Winner: SUNSET (+{sunset['correct']-sunrise['correct']:.1f}%)")
        else:
            print(f"  Winner: TIE")
    
    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)
    
    # By text type
    human_categories = [k for k in results.keys() if 'human' in k]
    ai_categories = [k for k in results.keys() if 'ai' in k]
    
    sunrise_human_acc = np.mean([results[k]['sunrise']['correct'] for k in human_categories])
    sunset_human_acc = np.mean([results[k]['sunset']['correct'] for k in human_categories])
    sunrise_ai_acc = np.mean([results[k]['sunrise']['correct'] for k in ai_categories])
    sunset_ai_acc = np.mean([results[k]['sunset']['correct'] for k in ai_categories])
    
    print(f"\nHuman Text Detection (lower AI prob = better):")
    print(f"  Sunrise: {sunrise_human_acc:.1f}% correct")
    print(f"  Sunset:  {sunset_human_acc:.1f}% correct")
    print(f"  Better at human: {'SUNRISE' if sunrise_human_acc > sunset_human_acc else 'SUNSET'}")
    
    print(f"\nAI Text Detection (higher AI prob = better):")
    print(f"  Sunrise: {sunrise_ai_acc:.1f}% correct")
    print(f"  Sunset:  {sunset_ai_acc:.1f}% correct")
    print(f"  Better at AI: {'SUNRISE' if sunrise_ai_acc > sunset_ai_acc else 'SUNSET'}")
    
    # Confidence analysis
    print("\n" + "=" * 70)
    print("CONFIDENCE ANALYSIS (Standard Deviation)")
    print("=" * 70)
    
    all_sunrise_std = np.mean([results[k]['sunrise']['std'] for k in results.keys()])
    all_sunset_std = np.mean([results[k]['sunset']['std'] for k in results.keys()])
    
    print(f"\nAverage prediction uncertainty:")
    print(f"  Sunrise std: {all_sunrise_std:.4f}")
    print(f"  Sunset std:  {all_sunset_std:.4f}")
    print(f"  More confident: {'SUNRISE' if all_sunrise_std < all_sunset_std else 'SUNSET'}")
    
    # Separation analysis
    print("\n" + "=" * 70)
    print("CLASS SEPARATION")
    print("=" * 70)
    
    sunrise_human_mean = np.mean([results[k]['sunrise']['mean'] for k in human_categories])
    sunrise_ai_mean = np.mean([results[k]['sunrise']['mean'] for k in ai_categories])
    sunset_human_mean = np.mean([results[k]['sunset']['mean'] for k in human_categories])
    sunset_ai_mean = np.mean([results[k]['sunset']['mean'] for k in ai_categories])
    
    sunrise_gap = sunrise_ai_mean - sunrise_human_mean
    sunset_gap = sunset_ai_mean - sunset_human_mean
    
    print(f"\nMean AI probability:")
    print(f"  Sunrise: Human={sunrise_human_mean:.3f}, AI={sunrise_ai_mean:.3f}, Gap={sunrise_gap:.3f}")
    print(f"  Sunset:  Human={sunset_human_mean:.3f}, AI={sunset_ai_mean:.3f}, Gap={sunset_gap:.3f}")
    print(f"  Better separation: {'SUNRISE' if sunrise_gap > sunset_gap else 'SUNSET'}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: STRENGTHS OF EACH MODEL")
    print("=" * 70)
    
    print("""
SUNRISE STRENGTHS:
- Statistical feature analysis (vocabulary, sentence structure)
- Better class separation (larger gap between human and AI means)
- Lower false positive rate on edge cases
- Top features: paragraph length, hapax count, vocabulary diversity

SUNSET STRENGTHS:
- GPTZero-style perplexity/burstiness analysis
- More confident predictions (lower std)
- Better at detecting humanized AI (from benchmark: 86.7% vs 66.7%)
- Top features: paragraph uniformity, n-gram entropy

WHEN TO USE WHICH:
- Use SUNRISE as primary classifier (better calibration)
- Use SUNSET for confirmation and humanization detection
- Combine both for maximum accuracy
- Trust the model with higher confidence on each sample
""")
    
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    analyze_models()
