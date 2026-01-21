#!/usr/bin/env python3
"""
Analyze misclassifications to understand failure modes.
Look at C4, Dolly, GPT4All samples to understand why they fail.
"""

import json
import re
import numpy as np
from collections import Counter

print("=" * 70)
print("Analyzing Misclassification Patterns")
print("=" * 70)

# Load data
with open('clean_dataset.json', 'r') as f:
    samples = json.load(f)

# Get samples from problem sources
c4_samples = [s for s in samples if s.get('source') == 'C4'][:50]
dolly_samples = [s for s in samples if s.get('source') == 'Dolly'][:50]
gpt4all_samples = [s for s in samples if s.get('source') == 'GPT4All'][:50]

print("\n" + "=" * 70)
print("C4 SAMPLES (Human web content - 85.8% accuracy)")
print("=" * 70)

for i, s in enumerate(c4_samples[:5]):
    text = s['text'][:500]
    print(f"\n[C4 #{i+1}] - HUMAN")
    print(f"  {text[:400]}...")
    # Check for AI-like patterns
    discourse = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally)\b', text, re.I))
    contractions = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I))
    casual = len(re.findall(r'\b(lol|haha|yeah|nah|ok)\b', text, re.I))
    attribution = len(re.findall(r'\b(said|says|told|according to)\b', text, re.I))
    print(f"  Discourse: {discourse}, Contractions: {contractions}, Casual: {casual}, Attribution: {attribution}")

print("\n" + "=" * 70)
print("DOLLY SAMPLES (AI factual content - 89.4% accuracy)")
print("=" * 70)

for i, s in enumerate(dolly_samples[:5]):
    text = s['text'][:500]
    print(f"\n[Dolly #{i+1}] - AI")
    print(f"  {text[:400]}...")
    discourse = len(re.findall(r'\b(however|therefore|furthermore|moreover|additionally)\b', text, re.I))
    contractions = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I))
    helpful = len(re.findall(r'\b(here is|let me|I hope|feel free)\b', text, re.I))
    print(f"  Discourse: {discourse}, Contractions: {contractions}, Helpful: {helpful}")

print("\n" + "=" * 70)
print("GPT4ALL SAMPLES (AI assistant content - 89.1% accuracy)")
print("=" * 70)

for i, s in enumerate(gpt4all_samples[:5]):
    text = s['text'][:500]
    print(f"\n[GPT4All #{i+1}] - AI")
    print(f"  {text[:400]}...")

# Key insight: Maybe these problem sources have overlapping characteristics
# C4: Formal human web text (looks like AI)
# Dolly: Short factual AI (looks like human encyclopedia)
# GPT4All: Q&A AI (can look like human forum posts)

print("\n" + "=" * 70)
print("INSIGHT: Problem sources share characteristics with opposite class")
print("=" * 70)
print("""
C4 (Human) - Fails because:
  - Formal, polished web content
  - No casual language (lol, haha)
  - Few contractions
  - Well-structured sentences
  
Dolly (AI) - Fails because:
  - Short factual content (looks like encyclopedia)
  - No helpful AI phrases ("here is", "let me")
  - Sounds like reference material
  
GPT4All (AI) - Fails because:
  - Q&A format can look like forums
  - Some samples are concise like human answers
  
SOLUTION: These represent the ambiguous middle ground.
For 99% accuracy, we might need to:
1. Remove truly ambiguous samples
2. Use multi-label confidence thresholds
3. Accept ~95% as ceiling for binary classification
""")

# Check distribution of text lengths in problem sources
print("\n" + "=" * 70)
print("Text Length Analysis")
print("=" * 70)

for source_name, source_samples in [('C4', c4_samples), ('Dolly', dolly_samples), ('GPT4All', gpt4all_samples)]:
    lengths = [len(s['text'].split()) for s in source_samples]
    print(f"{source_name}: mean={np.mean(lengths):.0f}, median={np.median(lengths):.0f}, min={min(lengths)}, max={max(lengths)}")
