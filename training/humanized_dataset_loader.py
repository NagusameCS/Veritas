#!/usr/bin/env python3
"""
Humanized AI Text Dataset Loader

Loads datasets specifically designed for training humanized AI detection:
- RAID dataset with paraphrase attacks
- GPT-wiki-intro (human vs AI paired)
- Various human text sources for baseline

Total available: >15M samples
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

try:
    from datasets import load_dataset
    from tqdm import tqdm
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Install required packages: pip install datasets tqdm")


@dataclass
class TextSample:
    """A single text sample with metadata."""
    text: str
    label: str  # 'human', 'ai', 'humanized'
    source: str
    attack_type: Optional[str] = None
    model: Optional[str] = None
    domain: Optional[str] = None


class HumanizedDatasetLoader:
    """
    Load humanized AI text datasets from HuggingFace.
    
    Key datasets:
    - liamdugan/raid: 5.6M samples with 12 attack types including paraphrase
    - dmitva/human_ai_generated_text: 1.5M paired samples
    - aadityaubhat/GPT-wiki-intro: 150K paired Wikipedia intros
    """
    
    # Dataset configurations
    HUMANIZED_DATASETS = {
        'raid_paraphrase': {
            'hf_id': 'liamdugan/raid',
            'description': 'RAID dataset - paraphrase attacks only',
            'filter': lambda x: x['attack'] == 'paraphrase' and x['model'] != 'human',
            'text_col': 'generation',
            'label': 'humanized',
            'estimated_size': 470000
        },
        'raid_all_attacks': {
            'hf_id': 'liamdugan/raid', 
            'description': 'RAID dataset - all adversarial attacks',
            'filter': lambda x: x['attack'] != 'none' and x['model'] != 'human',
            'text_col': 'generation',
            'label': 'humanized',
            'estimated_size': 5150000
        },
        'raid_raw_ai': {
            'hf_id': 'liamdugan/raid',
            'description': 'RAID dataset - raw AI (no attacks)',
            'filter': lambda x: x['attack'] == 'none' and x['model'] != 'human',
            'text_col': 'generation',
            'label': 'ai',
            'estimated_size': 420000
        },
        'raid_human': {
            'hf_id': 'liamdugan/raid',
            'description': 'RAID dataset - human text',
            'filter': lambda x: x['model'] == 'human',
            'text_col': 'generation',
            'label': 'human',
            'estimated_size': 48000
        }
    }
    
    AI_DETECTION_DATASETS = {
        'gpt_wiki_human': {
            'hf_id': 'aadityaubhat/GPT-wiki-intro',
            'description': 'Wikipedia intros - human written',
            'text_col': 'wiki_intro',
            'label': 'human',
            'estimated_size': 150000
        },
        'gpt_wiki_ai': {
            'hf_id': 'aadityaubhat/GPT-wiki-intro',
            'description': 'Wikipedia intros - GPT generated',
            'text_col': 'generated_intro',
            'label': 'ai',
            'estimated_size': 150000
        },
        'detection_pile': {
            'hf_id': 'artem9k/ai-text-detection-pile',
            'description': 'AI text detection pile (mixed)',
            'text_col': 'text',
            'label_col': 'source',
            'label_map': {'human': 'human', 'ai': 'ai', 'Human': 'human', 'AI': 'ai'},
            'estimated_size': 1400000
        },
        'human_ai_paired': {
            'hf_id': 'dmitva/human_ai_generated_text',
            'description': 'Human vs AI paired essays',
            'paired': True,
            'human_col': 'human_text',
            'ai_col': 'ai_text',
            'estimated_size': 1500000
        }
    }
    
    HUMAN_TEXT_DATASETS = {
        'openwebtext': {
            'hf_id': 'Skylion007/openwebtext',
            'description': 'OpenWebText - web content',
            'text_col': 'text',
            'label': 'human',
            'estimated_size': 8000000
        },
        'writingprompts': {
            'hf_id': 'euclaise/writingprompts',
            'description': 'Reddit WritingPrompts stories',
            'text_col': 'story',
            'label': 'human',
            'estimated_size': 300000
        },
        'imdb': {
            'hf_id': 'stanfordnlp/imdb',
            'description': 'IMDB movie reviews',
            'text_col': 'text',
            'label': 'human',
            'estimated_size': 50000
        },
        'ivypanda_essays': {
            'hf_id': 'qwedsacf/ivypanda-essays',
            'description': 'Academic student essays',
            'text_col': 'TEXT',
            'label': 'human',
            'estimated_size': 100000
        },
        'reddit_confessions': {
            'hf_id': 'SocialGrep/one-million-reddit-confessions',
            'description': 'Reddit confessions',
            'text_col': 'body',
            'label': 'human',
            'estimated_size': 1000000
        },
        'reddit_tifu': {
            'hf_id': 'ctr4si/reddit_tifu',
            'description': 'Reddit TIFU stories',
            'text_col': 'selftext',
            'label': 'human', 
            'estimated_size': 100000
        },
        'eli5': {
            'hf_id': 'defunct-datasets/eli5',
            'description': 'ELI5 explanations',
            'text_col': 'answers',  # This is a list
            'label': 'human',
            'estimated_size': 300000,
            'extract_list': True
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None, min_words: int = 50):
        """Initialize loader."""
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '.cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.min_words = min_words
        
    def _is_valid_text(self, text: str) -> bool:
        """Check if text meets minimum requirements."""
        if not text or not isinstance(text, str):
            return False
        return len(text.split()) >= self.min_words
    
    def load_raid_humanized(self, max_samples: int = 50000, 
                           attack_types: Optional[List[str]] = None) -> List[TextSample]:
        """
        Load humanized samples from RAID dataset.
        
        Args:
            max_samples: Maximum samples to load
            attack_types: List of attack types to include. 
                         Default: ['paraphrase', 'synonym', 'perplexity_misspelling']
                         All types: ['paraphrase', 'synonym', 'perplexity_misspelling', 
                                    'homoglyph', 'whitespace', 'upper_lower', 
                                    'insert_paragraphs', 'article_deletion',
                                    'alternative_spelling', 'number', 'zero_width_space']
        """
        if attack_types is None:
            # Focus on semantic modifications (most relevant for humanization)
            attack_types = ['paraphrase', 'synonym', 'perplexity_misspelling']
        
        print(f"Loading RAID humanized samples (attacks: {attack_types})...")
        
        ds = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        samples = []
        attack_counts = defaultdict(int)
        
        for item in tqdm(ds, desc="Scanning RAID", total=max_samples * 2):
            if len(samples) >= max_samples:
                break
                
            attack = item.get('attack', 'none')
            model = item.get('model', 'unknown')
            
            # Skip human text and non-targeted attacks
            if model == 'human' or attack not in attack_types:
                continue
            
            text = item.get('generation', '')
            if not self._is_valid_text(text):
                continue
            
            samples.append(TextSample(
                text=text,
                label='humanized',
                source='raid',
                attack_type=attack,
                model=model,
                domain=item.get('domain', 'unknown')
            ))
            attack_counts[attack] += 1
        
        print(f"Loaded {len(samples)} humanized samples")
        print(f"Attack distribution: {dict(attack_counts)}")
        
        return samples
    
    def load_raid_raw_ai(self, max_samples: int = 30000) -> List[TextSample]:
        """Load raw (unmodified) AI text from RAID for comparison."""
        print("Loading RAID raw AI samples...")
        
        ds = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        samples = []
        model_counts = defaultdict(int)
        
        for item in tqdm(ds, desc="Scanning RAID raw AI"):
            if len(samples) >= max_samples:
                break
            
            if item.get('attack') != 'none' or item.get('model') == 'human':
                continue
            
            text = item.get('generation', '')
            if not self._is_valid_text(text):
                continue
            
            model = item.get('model', 'unknown')
            samples.append(TextSample(
                text=text,
                label='ai',
                source='raid',
                model=model,
                domain=item.get('domain', 'unknown')
            ))
            model_counts[model] += 1
        
        print(f"Loaded {len(samples)} raw AI samples")
        print(f"Model distribution: {dict(model_counts)}")
        
        return samples
    
    def load_raid_human(self, max_samples: int = 30000) -> List[TextSample]:
        """Load human text from RAID."""
        print("Loading RAID human samples...")
        
        ds = load_dataset('liamdugan/raid', split='train', streaming=True)
        
        samples = []
        domain_counts = defaultdict(int)
        
        for item in tqdm(ds, desc="Scanning RAID human"):
            if len(samples) >= max_samples:
                break
            
            if item.get('model') != 'human':
                continue
            
            text = item.get('generation', '')
            if not self._is_valid_text(text):
                continue
            
            domain = item.get('domain', 'unknown')
            samples.append(TextSample(
                text=text,
                label='human',
                source='raid',
                domain=domain
            ))
            domain_counts[domain] += 1
        
        print(f"Loaded {len(samples)} human samples from RAID")
        print(f"Domain distribution: {dict(domain_counts)}")
        
        return samples
    
    def load_gpt_wiki(self, max_samples: int = 30000) -> Tuple[List[TextSample], List[TextSample]]:
        """Load paired human/AI Wikipedia intros."""
        print("Loading GPT-wiki-intro dataset...")
        
        ds = load_dataset('aadityaubhat/GPT-wiki-intro', split='train', streaming=True)
        
        human_samples = []
        ai_samples = []
        
        for item in tqdm(ds, desc="Loading GPT-wiki"):
            if len(human_samples) >= max_samples:
                break
            
            human_text = item.get('wiki_intro', '')
            ai_text = item.get('generated_intro', '')
            
            if self._is_valid_text(human_text):
                human_samples.append(TextSample(
                    text=human_text,
                    label='human',
                    source='gpt-wiki-intro',
                    domain='wikipedia'
                ))
            
            if self._is_valid_text(ai_text):
                ai_samples.append(TextSample(
                    text=ai_text,
                    label='ai',
                    source='gpt-wiki-intro',
                    model='gpt-3',
                    domain='wikipedia'
                ))
        
        print(f"Loaded {len(human_samples)} human and {len(ai_samples)} AI samples")
        
        return human_samples, ai_samples
    
    def load_detection_pile(self, max_samples: int = 50000) -> List[TextSample]:
        """Load from artem9k/ai-text-detection-pile."""
        print("Loading AI text detection pile...")
        
        ds = load_dataset('artem9k/ai-text-detection-pile', split='train', streaming=True)
        
        samples = []
        label_counts = defaultdict(int)
        
        for item in tqdm(ds, desc="Loading detection pile"):
            if len(samples) >= max_samples:
                break
            
            text = item.get('text', '')
            source_label = item.get('source', '').lower()
            
            if not self._is_valid_text(text):
                continue
            
            label = 'human' if source_label in ['human'] else 'ai'
            
            samples.append(TextSample(
                text=text,
                label=label,
                source='detection-pile'
            ))
            label_counts[label] += 1
        
        print(f"Loaded {len(samples)} samples: {dict(label_counts)}")
        
        return samples
    
    def load_human_essays(self, max_samples: int = 20000) -> List[TextSample]:
        """Load human essays from ivypanda dataset."""
        print("Loading human essays (ivypanda)...")
        
        ds = load_dataset('qwedsacf/ivypanda-essays', split='train', streaming=True)
        
        samples = []
        
        for item in tqdm(ds, desc="Loading essays"):
            if len(samples) >= max_samples:
                break
            
            text = item.get('TEXT', '')
            if not self._is_valid_text(text):
                continue
            
            samples.append(TextSample(
                text=text,
                label='human',
                source='ivypanda-essays',
                domain='academic'
            ))
        
        print(f"Loaded {len(samples)} essay samples")
        return samples
    
    def load_human_creative(self, max_samples: int = 20000) -> List[TextSample]:
        """Load human creative writing from WritingPrompts."""
        print("Loading creative writing (WritingPrompts)...")
        
        ds = load_dataset('euclaise/writingprompts', split='train', streaming=True)
        
        samples = []
        
        for item in tqdm(ds, desc="Loading stories"):
            if len(samples) >= max_samples:
                break
            
            text = item.get('story', '')
            if not self._is_valid_text(text):
                continue
            
            samples.append(TextSample(
                text=text,
                label='human',
                source='writingprompts',
                domain='creative'
            ))
        
        print(f"Loaded {len(samples)} creative writing samples")
        return samples
    
    def load_openwebtext(self, max_samples: int = 30000) -> List[TextSample]:
        """Load human web text from OpenWebText."""
        print("Loading OpenWebText...")
        
        ds = load_dataset('Skylion007/openwebtext', split='train', streaming=True)
        
        samples = []
        
        for item in tqdm(ds, desc="Loading webtext"):
            if len(samples) >= max_samples:
                break
            
            text = item.get('text', '')
            if not self._is_valid_text(text):
                continue
            
            samples.append(TextSample(
                text=text,
                label='human',
                source='openwebtext',
                domain='web'
            ))
        
        print(f"Loaded {len(samples)} web text samples")
        return samples
    
    def load_balanced_dataset(self, 
                             humanized_count: int = 30000,
                             ai_count: int = 30000,
                             human_count: int = 30000) -> Dict[str, List[TextSample]]:
        """
        Load a balanced dataset with humanized, raw AI, and human samples.
        
        Returns:
            Dictionary with 'humanized', 'ai', 'human' keys
        """
        print("=" * 60)
        print("Loading balanced dataset for tri-class training")
        print("=" * 60)
        
        # Load humanized samples (primary focus)
        humanized = self.load_raid_humanized(max_samples=humanized_count)
        
        # Load raw AI samples
        raw_ai = self.load_raid_raw_ai(max_samples=ai_count // 2)
        wiki_human, wiki_ai = self.load_gpt_wiki(max_samples=ai_count // 4)
        
        # Combine AI samples
        all_ai = raw_ai + wiki_ai
        if len(all_ai) > ai_count:
            all_ai = random.sample(all_ai, ai_count)
        
        # Load human samples from multiple sources
        human_samples = []
        human_samples.extend(self.load_raid_human(max_samples=human_count // 4))
        human_samples.extend(wiki_human[:human_count // 4])
        human_samples.extend(self.load_human_essays(max_samples=human_count // 4))
        human_samples.extend(self.load_human_creative(max_samples=human_count // 4))
        
        if len(human_samples) > human_count:
            human_samples = random.sample(human_samples, human_count)
        
        result = {
            'humanized': humanized,
            'ai': all_ai,
            'human': human_samples
        }
        
        print("\n" + "=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        for label, samples in result.items():
            print(f"{label.upper()}: {len(samples)} samples")
        print(f"TOTAL: {sum(len(s) for s in result.values())} samples")
        
        return result
    
    def save_dataset(self, samples: List[TextSample], filepath: str):
        """Save samples to JSON file."""
        data = [asdict(s) for s in samples]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(samples)} samples to {filepath}")
    
    def load_cached_dataset(self, filepath: str) -> List[TextSample]:
        """Load samples from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [TextSample(**d) for d in data]


def main():
    """Example usage."""
    loader = HumanizedDatasetLoader(min_words=50)
    
    # Load balanced dataset
    dataset = loader.load_balanced_dataset(
        humanized_count=30000,
        ai_count=30000, 
        human_count=30000
    )
    
    # Save to files
    for label, samples in dataset.items():
        filepath = f"training_data_{label}.json"
        loader.save_dataset(samples, filepath)
    
    print("\nDataset loading complete!")


if __name__ == "__main__":
    main()
