#!/usr/bin/env python3
"""
Fetch Authentic Samples for VERITAS Benchmark
Using modern Hugging Face datasets that support standard formats.
"""

import json
import os
import random
import re
import time
from pathlib import Path

try:
    import requests
except ImportError:
    os.system("pip install requests")
    import requests

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets")
    from datasets import load_dataset

OUTPUT_FILE = Path(__file__).parent / "authentic_samples.json"

def clean_text(text):
    """Clean and normalize text."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    if len(text) > 2000:
        text = text[:2000]
    return text

def fetch_eli5_dataset(num_samples=30):
    """Fetch ELI5 (Explain Like I'm 5) - human Q&A."""
    print("\n=== Fetching ELI5 Dataset ===")
    samples = []
    
    try:
        dataset = load_dataset("eli5_category", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
            
            answers = item.get('answers', {})
            answer_texts = answers.get('text', [])
            if answer_texts:
                cleaned = clean_text(answer_texts[0])
                if 150 < len(cleaned) < 1200:
                    samples.append({
                        'id': f'eli5_{count}',
                        'label': 'human',
                        'source': 'ELI5 (Reddit)',
                        'category': 'qa_explanation',
                        'text': cleaned
                    })
                    count += 1
        
        print(f"  Fetched {count} ELI5 samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_openwebtext(num_samples=30):
    """Fetch OpenWebText - human web content."""
    print("\n=== Fetching OpenWebText ===")
    samples = []
    
    try:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
            
            text = item.get('text', '')
            if len(text) > 400:
                excerpt = text[:900]
                last_period = excerpt.rfind('. ')
                if last_period > 250:
                    excerpt = excerpt[:last_period + 1]
                
                cleaned = clean_text(excerpt)
                if 250 < len(cleaned) < 1000:
                    samples.append({
                        'id': f'openwebtext_{count}',
                        'label': 'human',
                        'source': 'OpenWebText',
                        'category': 'web_article',
                        'text': cleaned
                    })
                    count += 1
        
        print(f"  Fetched {count} OpenWebText samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_writing_prompts(num_samples=25):
    """Fetch WritingPrompts - human creative writing."""
    print("\n=== Fetching WritingPrompts ===")
    samples = []
    
    try:
        dataset = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
            
            story = item.get('story', '') or item.get('text', '')
            cleaned = clean_text(story)
            
            if 200 < len(cleaned) < 1200:
                samples.append({
                    'id': f'writingprompts_{count}',
                    'label': 'human',
                    'source': 'WritingPrompts',
                    'category': 'creative_writing',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} WritingPrompts samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_imdb_reviews(num_samples=25):
    """Fetch IMDB movie reviews - human opinions."""
    print("\n=== Fetching IMDB Reviews ===")
    samples = []
    
    try:
        dataset = load_dataset("stanfordnlp/imdb", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
            
            review = item.get('text', '')
            cleaned = clean_text(review)
            
            if 150 < len(cleaned) < 1000:
                samples.append({
                    'id': f'imdb_{count}',
                    'label': 'human',
                    'source': 'IMDB Reviews',
                    'category': 'review',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} IMDB samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_yelp_reviews(num_samples=25):
    """Fetch Yelp reviews - human opinions."""
    print("\n=== Fetching Yelp Reviews ===")
    samples = []
    
    try:
        dataset = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
        count = 0
        for item in dataset:
            if count >= num_samples:
                break
            
            review = item.get('text', '')
            cleaned = clean_text(review)
            
            if 100 < len(cleaned) < 800:
                samples.append({
                    'id': f'yelp_{count}',
                    'label': 'human',
                    'source': 'Yelp Reviews',
                    'category': 'review',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} Yelp samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_squad_context(num_samples=20):
    """Fetch SQuAD context passages - curated Wikipedia (human)."""
    print("\n=== Fetching SQuAD Contexts ===")
    samples = []
    
    try:
        dataset = load_dataset("rajpurkar/squad", split="train", streaming=True)
        seen_contexts = set()
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            context = item.get('context', '')
            context_hash = hash(context[:100])
            
            if context_hash not in seen_contexts:
                seen_contexts.add(context_hash)
                cleaned = clean_text(context)
                
                if 200 < len(cleaned) < 1000:
                    samples.append({
                        'id': f'squad_{count}',
                        'label': 'human',
                        'source': 'SQuAD (Wikipedia)',
                        'category': 'encyclopedia',
                        'text': cleaned
                    })
                    count += 1
        
        print(f"  Fetched {count} SQuAD samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_cnn_dailymail(num_samples=20):
    """Fetch CNN/DailyMail - news articles (human journalism)."""
    print("\n=== Fetching CNN/DailyMail ===")
    samples = []
    
    try:
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            article = item.get('article', '')
            if len(article) > 400:
                excerpt = article[:1000]
                last_period = excerpt.rfind('. ')
                if last_period > 300:
                    excerpt = excerpt[:last_period + 1]
                
                cleaned = clean_text(excerpt)
                if 300 < len(cleaned) < 1000:
                    samples.append({
                        'id': f'cnn_{count}',
                        'label': 'human',
                        'source': 'CNN/DailyMail',
                        'category': 'news',
                        'text': cleaned
                    })
                    count += 1
        
        print(f"  Fetched {count} CNN/DailyMail samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_anthropic_hh(num_samples=40):
    """Fetch Anthropic HH-RLHF - contains AI assistant responses."""
    print("\n=== Fetching Anthropic HH-RLHF (AI Responses) ===")
    samples = []
    
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            chosen = item.get('chosen', '')
            if 'Assistant:' in chosen:
                parts = chosen.split('Assistant:')
                if len(parts) > 1:
                    response = parts[-1].strip()
                    if 'Human:' in response:
                        response = response.split('Human:')[0].strip()
                    
                    cleaned = clean_text(response)
                    if 100 < len(cleaned) < 1000:
                        samples.append({
                            'id': f'hh_rlhf_{count}',
                            'label': 'ai',
                            'source': 'Anthropic HH-RLHF',
                            'category': 'assistant_response',
                            'model': 'Claude (early)',
                            'text': cleaned
                        })
                        count += 1
        
        print(f"  Fetched {count} HH-RLHF AI samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_dolly(num_samples=30):
    """Fetch Dolly dataset - contains AI instruction responses."""
    print("\n=== Fetching Dolly (AI Responses) ===")
    samples = []
    
    try:
        dataset = load_dataset("databricks/databricks-dolly-15k", split="train", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            response = item.get('response', '')
            cleaned = clean_text(response)
            
            if 100 < len(cleaned) < 1000:
                samples.append({
                    'id': f'dolly_{count}',
                    'label': 'ai',
                    'source': 'Dolly-15k',
                    'category': 'instruction_response',
                    'model': 'Dolly',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} Dolly AI samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_alpaca(num_samples=30):
    """Fetch Alpaca dataset - AI instruction-tuned responses."""
    print("\n=== Fetching Alpaca (AI Responses) ===")
    samples = []
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            output = item.get('output', '')
            cleaned = clean_text(output)
            
            if 100 < len(cleaned) < 1000:
                samples.append({
                    'id': f'alpaca_{count}',
                    'label': 'ai',
                    'source': 'Alpaca',
                    'category': 'instruction_response',
                    'model': 'GPT-3.5 (text-davinci-003)',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} Alpaca AI samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_sharegpt(num_samples=30):
    """Fetch ShareGPT - real ChatGPT conversations."""
    print("\n=== Fetching ShareGPT (ChatGPT Responses) ===")
    samples = []
    
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            conversations = item.get('conversations', [])
            for conv in conversations:
                if count >= num_samples:
                    break
                if conv.get('from') == 'gpt':
                    response = conv.get('value', '')
                    cleaned = clean_text(response)
                    
                    if 150 < len(cleaned) < 1000:
                        samples.append({
                            'id': f'sharegpt_{count}',
                            'label': 'ai',
                            'source': 'ShareGPT',
                            'category': 'chatgpt_response',
                            'model': 'ChatGPT',
                            'text': cleaned
                        })
                        count += 1
        
        print(f"  Fetched {count} ShareGPT samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_oasst(num_samples=25):
    """Fetch OpenAssistant - mix of human and AI."""
    print("\n=== Fetching OpenAssistant ===")
    samples = []
    
    try:
        dataset = load_dataset("OpenAssistant/oasst1", split="train", streaming=True)
        human_count = 0
        ai_count = 0
        target = num_samples // 2
        
        for item in dataset:
            if human_count >= target and ai_count >= target:
                break
            
            text = item.get('text', '')
            role = item.get('role', '')
            cleaned = clean_text(text)
            
            if 100 < len(cleaned) < 800:
                if role == 'prompter' and human_count < target:
                    samples.append({
                        'id': f'oasst_human_{human_count}',
                        'label': 'human',
                        'source': 'OpenAssistant',
                        'category': 'user_prompt',
                        'text': cleaned
                    })
                    human_count += 1
                elif role == 'assistant' and ai_count < target:
                    samples.append({
                        'id': f'oasst_ai_{ai_count}',
                        'label': 'ai',
                        'source': 'OpenAssistant',
                        'category': 'assistant_response',
                        'model': 'OpenAssistant',
                        'text': cleaned
                    })
                    ai_count += 1
        
        print(f"  Fetched {human_count} human + {ai_count} AI OpenAssistant samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def fetch_truthful_qa(num_samples=20):
    """Fetch TruthfulQA - human-written correct answers."""
    print("\n=== Fetching TruthfulQA ===")
    samples = []
    
    try:
        dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", streaming=True)
        count = 0
        
        for item in dataset:
            if count >= num_samples:
                break
            
            best_answer = item.get('best_answer', '')
            cleaned = clean_text(best_answer)
            
            if 50 < len(cleaned) < 500:
                samples.append({
                    'id': f'truthfulqa_{count}',
                    'label': 'human',
                    'source': 'TruthfulQA',
                    'category': 'factual_answer',
                    'text': cleaned
                })
                count += 1
        
        print(f"  Fetched {count} TruthfulQA samples")
    except Exception as e:
        print(f"  Error: {e}")
    
    return samples

def main():
    print("=" * 60)
    print("VERITAS Authentic Sample Fetcher v2")
    print("=" * 60)
    
    all_samples = []
    
    # Human sources
    all_samples.extend(fetch_openwebtext(30))
    all_samples.extend(fetch_writing_prompts(25))
    all_samples.extend(fetch_imdb_reviews(25))
    all_samples.extend(fetch_yelp_reviews(25))
    all_samples.extend(fetch_eli5_dataset(30))
    all_samples.extend(fetch_squad_context(20))
    all_samples.extend(fetch_cnn_dailymail(20))
    all_samples.extend(fetch_truthful_qa(20))
    
    # AI sources
    all_samples.extend(fetch_anthropic_hh(40))
    all_samples.extend(fetch_dolly(30))
    all_samples.extend(fetch_alpaca(30))
    all_samples.extend(fetch_sharegpt(30))
    all_samples.extend(fetch_oasst(25))
    
    # Summary
    human_samples = [s for s in all_samples if s['label'] == 'human']
    ai_samples = [s for s in all_samples if s['label'] == 'ai']
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total samples: {len(all_samples)}")
    print(f"  Human: {len(human_samples)}")
    print(f"  AI: {len(ai_samples)}")
    
    sources = {}
    for s in all_samples:
        src = s['source']
        if src not in sources:
            sources[src] = {'human': 0, 'ai': 0}
        sources[src][s['label']] += 1
    
    print("\nBy source:")
    for src, counts in sorted(sources.items()):
        print(f"  {src}: {counts['human']} human, {counts['ai']} AI")
    
    output = {
        'metadata': {
            'total_samples': len(all_samples),
            'human_count': len(human_samples),
            'ai_count': len(ai_samples),
            'sources': list(sources.keys()),
            'fetched_at': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'samples': all_samples
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("=" * 60)

if __name__ == "__main__":
    main()
