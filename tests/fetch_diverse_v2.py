#!/usr/bin/env python3
"""
Fetch additional diverse data to improve model accuracy
Focus on:
1. Modern AI outputs (GPT-4, Claude style)
2. More formal human writing (academic, technical)
3. Informal human writing (social media, forums)
"""

import json
import os
from datasets import load_dataset
import random

print("=" * 70)
print("Fetching Additional Diverse Data")
print("=" * 70)

samples = []

# =============================================================================
# MORE HUMAN SOURCES
# =============================================================================

# 1. ELI5 (Reddit Explain Like I'm 5) - casual explanations
print("\n[1/8] Fetching ELI5 (casual explanations)...")
try:
    ds = load_dataset("eli5", split="train_eli5", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        text = item.get('answers', {}).get('text', [''])[0] if 'answers' in item else item.get('text', '')
        if text and len(text) > 100 and len(text) < 3000:
            samples.append({'text': text, 'label': 'human', 'source': 'ELI5'})
            count += 1
            if count >= 5000:
                break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 2. PushShift Reddit (more reddit data)
print("\n[2/8] Fetching Reddit comments...")
try:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_sft", streaming=True)
    count = 0
    for item in ds:
        # Get the prompt (human) not the response (AI)
        text = item.get('prompt', '')
        if text and len(text) > 100 and len(text) < 2000:
            samples.append({'text': text, 'label': 'human', 'source': 'UltraFeedback-Prompts'})
            count += 1
            if count >= 5000:
                break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 3. Scientific papers (formal human)
print("\n[3/8] Fetching scientific abstracts...")
try:
    ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        text = item.get('abstract', '')
        if text and len(text) > 200 and len(text) < 2000:
            samples.append({'text': text, 'label': 'human', 'source': 'ArXiv-Abstracts'})
            count += 1
            if count >= 5000:
                break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 4. Blog posts (personal writing)
print("\n[4/8] Fetching blog posts...")
try:
    ds = load_dataset("blog_authorship_corpus", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        text = item.get('text', '')
        if text and len(text) > 200 and len(text) < 2500:
            samples.append({'text': text, 'label': 'human', 'source': 'Blogs'})
            count += 1
            if count >= 5000:
                break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# =============================================================================
# MORE AI SOURCES (Modern/RLHF-tuned)
# =============================================================================

# 5. LMSYS Chat (real LLM conversations)
print("\n[5/8] Fetching LMSYS Chat (modern LLM outputs)...")
try:
    ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        convs = item.get('conversation', [])
        for conv in convs:
            if conv.get('role') == 'assistant':
                text = conv.get('content', '')
                if text and len(text) > 100 and len(text) < 3000:
                    samples.append({'text': text, 'label': 'ai', 'source': 'LMSYS-Chat'})
                    count += 1
                    if count >= 10000:
                        break
        if count >= 10000:
            break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 6. ChatGPT conversations
print("\n[6/8] Fetching ShareGPT (ChatGPT outputs)...")
try:
    ds = load_dataset("RyokoAI/ShareGPT52K", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        convs = item.get('conversations', [])
        for conv in convs:
            if conv.get('from') == 'gpt':
                text = conv.get('value', '')
                if text and len(text) > 100 and len(text) < 3000:
                    samples.append({'text': text, 'label': 'ai', 'source': 'ShareGPT'})
                    count += 1
                    if count >= 10000:
                        break
        if count >= 10000:
            break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 7. Vicuna conversations
print("\n[7/8] Fetching Vicuna-style outputs...")
try:
    ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        convs = item.get('conversations', [])
        for conv in convs:
            if conv.get('from') in ['gpt', 'assistant']:
                text = conv.get('value', '')
                if text and len(text) > 100 and len(text) < 3000:
                    samples.append({'text': text, 'label': 'ai', 'source': 'WizardLM-V2'})
                    count += 1
                    if count >= 5000:
                        break
        if count >= 5000:
            break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# 8. Orca (complex reasoning AI)
print("\n[8/8] Fetching Orca-style outputs...")
try:
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for item in ds:
        text = item.get('response', '')
        if text and len(text) > 100 and len(text) < 3000:
            samples.append({'text': text, 'label': 'ai', 'source': 'OpenOrca'})
            count += 1
            if count >= 10000:
                break
    print(f"  Got {count} samples")
except Exception as e:
    print(f"  Failed: {e}")

# =============================================================================
# SAVE
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

# Count by source
from collections import Counter
source_counts = Counter(s['source'] for s in samples)
label_counts = Counter(s['label'] for s in samples)

print(f"\nTotal new samples: {len(samples)}")
print(f"\nBy label:")
for label, count in label_counts.items():
    print(f"  {label}: {count}")
print(f"\nBy source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

# Save
with open('diverse_samples_v2.json', 'w') as f:
    json.dump(samples, f)

print(f"\nSaved to 'diverse_samples_v2.json'")
