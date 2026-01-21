#!/usr/bin/env python3
"""
Large-Scale Sample Fetcher for VERITAS
Target: 100k+ human samples, 100k+ AI samples
"""

import json
import os
import random
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets")
    from datasets import load_dataset

OUTPUT_FILE = Path(__file__).parent / "large_samples.json"

def clean_text(text, min_len=100, max_len=2000):
    """Clean and normalize text."""
    if not text:
        return None
    text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    if len(text) < min_len or len(text) > max_len:
        return None
    return text

# =============================================================================
# HUMAN SOURCES (Target: 100k+ total)
# =============================================================================

def fetch_openwebtext(num_samples=20000):
    """OpenWebText - diverse web content (8M+ samples available)"""
    print(f"\n=== Fetching OpenWebText ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'owt_{len(samples)}',
                    'label': 'human',
                    'source': 'OpenWebText',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} OpenWebText samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_c4(num_samples=20000):
    """C4 (Colossal Clean Crawled Corpus) - massive web text"""
    print(f"\n=== Fetching C4 ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'c4_{len(samples)}',
                    'label': 'human',
                    'source': 'C4',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} C4 samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_wikipedia(num_samples=15000):
    """Wikipedia - encyclopedic content"""
    print(f"\n=== Fetching Wikipedia ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'wiki_{len(samples)}',
                    'label': 'human',
                    'source': 'Wikipedia',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} Wikipedia samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_reddit(num_samples=15000):
    """Reddit comments - casual human conversation"""
    print(f"\n=== Fetching Reddit ({num_samples}) ===")
    samples = []
    try:
        # Use pushshift reddit dataset
        dataset = load_dataset("webis/tldr-17", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('content', '') or item.get('normalizedBody', ''))
            if text:
                samples.append({
                    'id': f'reddit_{len(samples)}',
                    'label': 'human',
                    'source': 'Reddit',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} Reddit samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_amazon_reviews(num_samples=10000):
    """Amazon reviews - product reviews"""
    print(f"\n=== Fetching Amazon Reviews ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", split="full", streaming=True, trust_remote_code=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'amazon_{len(samples)}',
                    'label': 'human',
                    'source': 'Amazon',
                    'text': text
                })
            if i % 2000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} Amazon samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_imdb(num_samples=10000):
    """IMDB reviews"""
    print(f"\n=== Fetching IMDB ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("stanfordnlp/imdb", split="train")
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'imdb_{len(samples)}',
                    'label': 'human',
                    'source': 'IMDB',
                    'text': text
                })
        print(f"  Fetched {len(samples)} IMDB samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_yelp(num_samples=10000):
    """Yelp reviews"""
    print(f"\n=== Fetching Yelp ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("Yelp/yelp_review_full", split="train")
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'yelp_{len(samples)}',
                    'label': 'human',
                    'source': 'Yelp',
                    'text': text
                })
        print(f"  Fetched {len(samples)} Yelp samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_news(num_samples=10000):
    """News articles - AG News, CNN"""
    print(f"\n=== Fetching News ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("fancyzhx/ag_news", split="train")
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''))
            if text:
                samples.append({
                    'id': f'news_{len(samples)}',
                    'label': 'human',
                    'source': 'News',
                    'text': text
                })
        print(f"  Fetched {len(samples)} News samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_books(num_samples=5000):
    """Book excerpts"""
    print(f"\n=== Fetching Books ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("bookcorpus/bookcorpus", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('text', ''), min_len=200)
            if text:
                samples.append({
                    'id': f'book_{len(samples)}',
                    'label': 'human',
                    'source': 'Books',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} Book samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

# =============================================================================
# AI SOURCES (Target: 100k+ total)
# =============================================================================

def fetch_anthropic_hh(num_samples=20000):
    """Anthropic HH-RLHF - Claude responses"""
    print(f"\n=== Fetching Anthropic HH-RLHF ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            chosen = item.get('chosen', '')
            # Extract assistant response
            if 'Assistant:' in chosen:
                response = chosen.split('Assistant:')[-1].strip()
                text = clean_text(response)
                if text:
                    samples.append({
                        'id': f'anthropic_{len(samples)}',
                        'label': 'ai',
                        'source': 'Anthropic-RLHF',
                        'text': text
                    })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} Anthropic samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_sharegpt(num_samples=20000):
    """ShareGPT - GPT conversations"""
    print(f"\n=== Fetching ShareGPT ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            conversations = item.get('conversations', [])
            for conv in conversations:
                if conv.get('from') in ['gpt', 'assistant', 'chatgpt']:
                    text = clean_text(conv.get('value', ''))
                    if text:
                        samples.append({
                            'id': f'sharegpt_{len(samples)}',
                            'label': 'ai',
                            'source': 'ShareGPT',
                            'text': text
                        })
                        if len(samples) >= num_samples:
                            break
            if i % 2000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} ShareGPT samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_dolly(num_samples=15000):
    """Databricks Dolly 15k"""
    print(f"\n=== Fetching Dolly ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train")
        for item in dataset:
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('response', ''))
            if text:
                samples.append({
                    'id': f'dolly_{len(samples)}',
                    'label': 'ai',
                    'source': 'Dolly',
                    'text': text
                })
        print(f"  Fetched {len(samples)} Dolly samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_alpaca(num_samples=15000):
    """Stanford Alpaca"""
    print(f"\n=== Fetching Alpaca ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        for item in dataset:
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('output', ''))
            if text:
                samples.append({
                    'id': f'alpaca_{len(samples)}',
                    'label': 'ai',
                    'source': 'Alpaca',
                    'text': text
                })
        print(f"  Fetched {len(samples)} Alpaca samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_openassistant(num_samples=15000):
    """OpenAssistant conversations"""
    print(f"\n=== Fetching OpenAssistant ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train")
        for item in dataset:
            if len(samples) >= num_samples:
                break
            if item.get('role') == 'assistant':
                text = clean_text(item.get('text', ''))
                if text:
                    samples.append({
                        'id': f'oasst_{len(samples)}',
                        'label': 'ai',
                        'source': 'OpenAssistant',
                        'text': text
                    })
        print(f"  Fetched {len(samples)} OpenAssistant samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_gpt4all(num_samples=15000):
    """GPT4All dataset"""
    print(f"\n=== Fetching GPT4All ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("nomic-ai/gpt4all-j-prompt-generations", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            text = clean_text(item.get('response', ''))
            if text:
                samples.append({
                    'id': f'gpt4all_{len(samples)}',
                    'label': 'ai',
                    'source': 'GPT4All',
                    'text': text
                })
            if i % 5000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} GPT4All samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_wizardlm(num_samples=10000):
    """WizardLM - instruction following"""
    print(f"\n=== Fetching WizardLM ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            conversations = item.get('conversations', [])
            for conv in conversations:
                if conv.get('from') in ['gpt', 'assistant']:
                    text = clean_text(conv.get('value', ''))
                    if text:
                        samples.append({
                            'id': f'wizardlm_{len(samples)}',
                            'label': 'ai',
                            'source': 'WizardLM',
                            'text': text
                        })
                        if len(samples) >= num_samples:
                            break
            if i % 2000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} WizardLM samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

def fetch_ultrachat(num_samples=15000):
    """UltraChat - multi-turn conversations"""
    print(f"\n=== Fetching UltraChat ({num_samples}) ===")
    samples = []
    try:
        dataset = load_dataset("stingning/ultrachat", split="train", streaming=True)
        for i, item in enumerate(dataset):
            if len(samples) >= num_samples:
                break
            data = item.get('data', [])
            for j, msg in enumerate(data):
                if j % 2 == 1:  # Assistant responses are odd indices
                    text = clean_text(msg)
                    if text:
                        samples.append({
                            'id': f'ultrachat_{len(samples)}',
                            'label': 'ai',
                            'source': 'UltraChat',
                            'text': text
                        })
                        if len(samples) >= num_samples:
                            break
            if i % 2000 == 0 and i > 0:
                print(f"  Progress: {len(samples)}/{num_samples}")
        print(f"  Fetched {len(samples)} UltraChat samples")
    except Exception as e:
        print(f"  Error: {e}")
    return samples

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("VERITAS Large-Scale Sample Fetcher")
    print("Target: 100k+ Human, 100k+ AI samples")
    print("=" * 60)
    
    all_samples = []
    
    # Human sources
    print("\n" + "=" * 40)
    print("FETCHING HUMAN SAMPLES")
    print("=" * 40)
    
    human_fetchers = [
        (fetch_openwebtext, 20000),
        (fetch_c4, 20000),
        (fetch_wikipedia, 15000),
        (fetch_reddit, 15000),
        (fetch_amazon_reviews, 10000),
        (fetch_imdb, 10000),
        (fetch_yelp, 10000),
        (fetch_news, 10000),
        (fetch_books, 5000),
    ]
    
    for fetcher, num in human_fetchers:
        try:
            samples = fetcher(num)
            all_samples.extend(samples)
            print(f"  Total human so far: {sum(1 for s in all_samples if s['label'] == 'human')}")
        except Exception as e:
            print(f"  Skipping due to error: {e}")
    
    # AI sources
    print("\n" + "=" * 40)
    print("FETCHING AI SAMPLES")
    print("=" * 40)
    
    ai_fetchers = [
        (fetch_anthropic_hh, 20000),
        (fetch_sharegpt, 20000),
        (fetch_dolly, 15000),
        (fetch_alpaca, 15000),
        (fetch_openassistant, 15000),
        (fetch_gpt4all, 15000),
        (fetch_wizardlm, 10000),
        (fetch_ultrachat, 15000),
    ]
    
    for fetcher, num in ai_fetchers:
        try:
            samples = fetcher(num)
            all_samples.extend(samples)
            print(f"  Total AI so far: {sum(1 for s in all_samples if s['label'] == 'ai')}")
        except Exception as e:
            print(f"  Skipping due to error: {e}")
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Summary
    human_count = sum(1 for s in all_samples if s['label'] == 'human')
    ai_count = sum(1 for s in all_samples if s['label'] == 'ai')
    sources = set(s['source'] for s in all_samples)
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    print(f"Human: {human_count}")
    print(f"AI: {ai_count}")
    print(f"Sources: {len(sources)}")
    for src in sorted(sources):
        count = sum(1 for s in all_samples if s['source'] == src)
        label = all_samples[[s['source'] for s in all_samples].index(src)]['label']
        print(f"  {src}: {count} ({label})")
    
    # Save
    output = {
        'metadata': {
            'total': len(all_samples),
            'human': human_count,
            'ai': ai_count,
            'sources': list(sources)
        },
        'samples': all_samples
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f)
    
    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
