#!/usr/bin/env python3
"""Discover all AI text detection datasets on HuggingFace."""

from huggingface_hub import list_datasets
import json

search_terms = [
    'ai text detection', 'ai generated text', 'chatgpt detection', 'gpt detection',
    'llm detection', 'machine generated text', 'human vs ai', 'human vs machine',
    'synthetic text', 'generated text detection', 'ai written', 'deepfake text',
    'fake text detection', 'chatgpt generated', 'llm generated', 'ai content', 
    'gpt4 generated', 'claude generated', 'llama generated', 'human written',
    'authentic text', 'real fake text', 'ai detector', 'gpt2 output', 'gpt3 output',
    'human ai', 'machine human', 'generated content', 'llm output', 'ai essay',
    'chatgpt essay', 'ai news', 'fake news ai', 'ai story', 'ai article'
]

all_datasets = {}
for term in search_terms:
    try:
        datasets = list(list_datasets(search=term, limit=100))
        for ds in datasets:
            if ds.id not in all_datasets:
                all_datasets[ds.id] = {
                    'downloads': ds.downloads or 0, 
                    'tags': list(ds.tags) if ds.tags else []
                }
        print(f"Searched '{term}': total {len(all_datasets)} unique datasets")
    except Exception as e:
        print(f"Error searching '{term}': {e}")

with open('discovered_datasets.json', 'w') as f:
    json.dump(all_datasets, f, indent=2)

print(f"\n=== FOUND {len(all_datasets)} UNIQUE DATASETS ===")
for ds_id in sorted(all_datasets.keys()):
    print(ds_id)
