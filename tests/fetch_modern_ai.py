#!/usr/bin/env python3
"""
Fetch modern AI-generated text samples from diverse sources.
Targets: GPT-4 level outputs, Claude, newer instruction-tuned models
Also fetches "humanized" AI text patterns for detection.
"""

import json
import os
from datasets import load_dataset
import random

def fetch_lmsys_chat():
    """LMSYS Chat Arena - real conversations with multiple LLMs including GPT-4, Claude"""
    print("Fetching LMSYS Chat Arena (GPT-4, Claude conversations)...")
    samples = []
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= 15000:
                break
            # Get assistant responses
            for conv in item.get('conversation', []):
                if conv.get('role') == 'assistant' and len(conv.get('content', '')) > 100:
                    model = item.get('model', 'unknown')
                    samples.append({
                        'text': conv['content'][:2000],
                        'label': 'ai',
                        'source': f'LMSYS-{model[:20]}',
                        'model': model
                    })
                    count += 1
                    if count >= 15000:
                        break
            if count % 2000 == 0:
                print(f"  LMSYS: {count} samples")
    except Exception as e:
        print(f"  LMSYS error: {e}")
    return samples

def fetch_wildchat():
    """WildChat - real GPT conversations in the wild"""
    print("Fetching WildChat (GPT-4 in-the-wild)...")
    samples = []
    try:
        dataset = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= 10000:
                break
            # Get assistant turns
            for turn in item.get('conversation', []):
                if turn.get('role') == 'assistant' and len(turn.get('content', '')) > 100:
                    samples.append({
                        'text': turn['content'][:2000],
                        'label': 'ai',
                        'source': 'WildChat-GPT',
                        'model': item.get('model', 'gpt-4')
                    })
                    count += 1
                    if count >= 10000:
                        break
            if count % 2000 == 0:
                print(f"  WildChat: {count} samples")
    except Exception as e:
        print(f"  WildChat error: {e}")
    return samples

def fetch_chatgpt_prompts():
    """ChatGPT prompts dataset with responses"""
    print("Fetching ChatGPT prompts/responses...")
    samples = []
    try:
        dataset = load_dataset("MohamedRashad/ChatGPT-prompts", split="train")
        for item in list(dataset)[:5000]:
            if len(item.get('response', '')) > 100:
                samples.append({
                    'text': item['response'][:2000],
                    'label': 'ai',
                    'source': 'ChatGPT-Prompts',
                    'model': 'chatgpt'
                })
        print(f"  ChatGPT prompts: {len(samples)} samples")
    except Exception as e:
        print(f"  ChatGPT prompts error: {e}")
    return samples

def fetch_sharegpt_vicuna():
    """ShareGPT/Vicuna format conversations"""
    print("Fetching ShareGPT/Vicuna...")
    samples = []
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= 8000:
                break
            convs = item.get('conversations', [])
            for conv in convs:
                if conv.get('from') == 'gpt' and len(conv.get('value', '')) > 100:
                    samples.append({
                        'text': conv['value'][:2000],
                        'label': 'ai',
                        'source': 'ShareGPT-Vicuna',
                        'model': 'gpt-3.5/4'
                    })
                    count += 1
                    if count >= 8000:
                        break
        print(f"  ShareGPT/Vicuna: {len(samples)} samples")
    except Exception as e:
        print(f"  ShareGPT/Vicuna error: {e}")
    return samples

def fetch_open_assistant():
    """Open Assistant v2 - high quality assistant responses"""
    print("Fetching Open Assistant v2...")
    samples = []
    try:
        dataset = load_dataset("OpenAssistant/oasst2", split="train")
        for item in list(dataset):
            if item.get('role') == 'assistant' and len(item.get('text', '')) > 100:
                samples.append({
                    'text': item['text'][:2000],
                    'label': 'ai',
                    'source': 'OpenAssistant-v2',
                    'model': 'oasst'
                })
                if len(samples) >= 10000:
                    break
        print(f"  Open Assistant v2: {len(samples)} samples")
    except Exception as e:
        print(f"  Open Assistant v2 error: {e}")
    return samples

def fetch_human_essays():
    """Human essays for comparison"""
    print("Fetching human essays...")
    samples = []
    try:
        dataset = load_dataset("qwedsacf/ivypanda-essays", split="train")
        for item in list(dataset)[:8000]:
            text = item.get('TEXT', '')
            if len(text) > 200:
                samples.append({
                    'text': text[:2000],
                    'label': 'human',
                    'source': 'IvyPanda-Essays'
                })
        print(f"  IvyPanda essays: {len(samples)} samples")
    except Exception as e:
        print(f"  IvyPanda essays error: {e}")
    return samples

def fetch_human_blogs():
    """Human blog posts"""
    print("Fetching human blogs...")
    samples = []
    try:
        dataset = load_dataset("blog_authorship_corpus", split="train", trust_remote_code=True)
        for item in list(dataset)[:10000]:
            text = item.get('text', '')
            if len(text) > 200:
                samples.append({
                    'text': text[:2000],
                    'label': 'human',
                    'source': 'Blog-Corpus'
                })
        print(f"  Blog corpus: {len(samples)} samples")
    except Exception as e:
        print(f"  Blog corpus error: {e}")
    return samples

def fetch_human_reddit():
    """Human Reddit posts"""
    print("Fetching human Reddit posts...")
    samples = []
    try:
        dataset = load_dataset("webis/tldr-17", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= 10000:
                break
            text = item.get('content', '')
            if len(text) > 200:
                samples.append({
                    'text': text[:2000],
                    'label': 'human',
                    'source': 'Reddit-TLDR'
                })
                count += 1
        print(f"  Reddit TLDR: {len(samples)} samples")
    except Exception as e:
        print(f"  Reddit TLDR error: {e}")
    return samples

def fetch_human_stories():
    """Human creative writing"""
    print("Fetching human stories...")
    samples = []
    try:
        dataset = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= 10000:
                break
            text = item.get('story', '')
            if len(text) > 200:
                samples.append({
                    'text': text[:2000],
                    'label': 'human',
                    'source': 'WritingPrompts'
                })
                count += 1
        print(f"  WritingPrompts: {len(samples)} samples")
    except Exception as e:
        print(f"  WritingPrompts error: {e}")
    return samples

def fetch_ai_detector_datasets():
    """Existing AI detection datasets with labeled samples"""
    print("Fetching AI detection benchmark datasets...")
    samples = []
    
    # Try HC3 dataset (human vs ChatGPT)
    try:
        dataset = load_dataset("Hello-SimpleAI/HC3", "all", split="train")
        for item in list(dataset)[:5000]:
            # Human answers
            for ans in item.get('human_answers', [])[:1]:
                if len(ans) > 100:
                    samples.append({
                        'text': ans[:2000],
                        'label': 'human',
                        'source': 'HC3-Human'
                    })
            # ChatGPT answers
            for ans in item.get('chatgpt_answers', [])[:1]:
                if len(ans) > 100:
                    samples.append({
                        'text': ans[:2000],
                        'label': 'ai',
                        'source': 'HC3-ChatGPT'
                    })
        print(f"  HC3 dataset: {len(samples)} samples")
    except Exception as e:
        print(f"  HC3 error: {e}")
    
    return samples

def main():
    print("="*60)
    print("Fetching Modern AI Data for 99% Detection Accuracy")
    print("="*60)
    
    all_samples = []
    
    # Modern AI sources
    all_samples.extend(fetch_lmsys_chat())
    all_samples.extend(fetch_wildchat())
    all_samples.extend(fetch_chatgpt_prompts())
    all_samples.extend(fetch_sharegpt_vicuna())
    all_samples.extend(fetch_open_assistant())
    
    # Human sources
    all_samples.extend(fetch_human_essays())
    all_samples.extend(fetch_human_blogs())
    all_samples.extend(fetch_human_reddit())
    all_samples.extend(fetch_human_stories())
    
    # AI detection benchmarks
    all_samples.extend(fetch_ai_detector_datasets())
    
    # Summary
    human_count = sum(1 for s in all_samples if s['label'] == 'human')
    ai_count = sum(1 for s in all_samples if s['label'] == 'ai')
    
    print("\n" + "="*60)
    print(f"Total samples: {len(all_samples)}")
    print(f"Human: {human_count}")
    print(f"AI: {ai_count}")
    print("="*60)
    
    # Save
    with open('modern_ai_samples.json', 'w') as f:
        json.dump(all_samples, f)
    
    print(f"Saved to modern_ai_samples.json")
    
    # Source breakdown
    sources = {}
    for s in all_samples:
        src = s['source']
        if src not in sources:
            sources[src] = {'human': 0, 'ai': 0}
        sources[src][s['label']] += 1
    
    print("\nBy Source:")
    for src, counts in sorted(sources.items()):
        print(f"  {src}: {counts}")

if __name__ == "__main__":
    main()
