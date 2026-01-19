"""
Veritas Dataset Loader
Loads AI detection datasets from HuggingFace for training.
"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset, concatenate_datasets
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' library not installed. Run: pip install datasets")


@dataclass
class TextSample:
    """A single text sample with label."""
    text: str
    label: int  # 0 = human, 1 = AI
    source: str  # dataset name
    metadata: Optional[Dict] = None


class DatasetLoader:
    """Load and manage AI detection training datasets."""
    
    # HuggingFace datasets for AI text detection
    DATASETS = {
        'ai-text-detection-pile': {
            'name': 'artem9k/ai-text-detection-pile',
            'text_column': 'text',
            'label_column': 'source',
            'label_map': {'human': 0, 'ai': 1, 'Human': 0, 'AI': 1},
            'description': 'Large pile of AI vs human text samples (1.4M samples)'
        },
        'ai-text-detection-pile-cleaned': {
            'name': 'srikanthgali/ai-text-detection-pile-cleaned',
            'text_column': 'text', 
            'label_column': 'source',
            'label_map': {'human': 0, 'ai': 1, 'Human': 0, 'AI': 1},
            'description': 'Cleaned version of AI detection pile'
        },
        'deepfake-text': {
            'name': 'yaful/DeepfakeTextDetect',
            'text_column': 'text',
            'label_column': 'label',
            'label_map': {0: 0, 1: 1, 'human': 0, 'machine': 1},
            'description': 'Deepfake text detection dataset'
        },
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the dataset loader."""
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), '.cache'
        )
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.samples: List[TextSample] = []
        
    def load_dataset(self, dataset_key: str, max_samples: Optional[int] = None,
                     split: str = 'train') -> List[TextSample]:
        """Load a specific dataset by key."""
        if not HF_AVAILABLE:
            raise RuntimeError("datasets library not installed")
        
        if dataset_key not in self.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}. "
                           f"Available: {list(self.DATASETS.keys())}")
        
        config = self.DATASETS[dataset_key]
        samples = []
        
        print(f"Loading dataset: {config['name']}...")
        
        try:
            # Load from HuggingFace
            if 'config' in config:
                ds = load_dataset(
                    config['name'],
                    config['config'],
                    split=split,
                    cache_dir=self.cache_dir
                )
            else:
                ds = load_dataset(
                    config['name'],
                    split=split,
                    cache_dir=self.cache_dir
                )
            
            # Handle paired datasets (human + AI in same row)
            if config.get('is_paired'):
                samples = self._process_paired_dataset(ds, config, max_samples)
            else:
                samples = self._process_labeled_dataset(ds, config, max_samples)
            
            print(f"Loaded {len(samples)} samples from {dataset_key}")
            
        except Exception as e:
            print(f"Error loading {dataset_key}: {e}")
            return []
        
        return samples
    
    def _process_labeled_dataset(self, ds, config: Dict, 
                                  max_samples: Optional[int]) -> List[TextSample]:
        """Process a dataset with explicit labels."""
        samples = []
        text_col = config['text_column']
        label_col = config['label_column']
        label_map = config.get('label_map', {0: 0, 1: 1})
        
        indices = list(range(len(ds)))
        if max_samples and len(indices) > max_samples:
            indices = random.sample(indices, max_samples)
        
        for idx in tqdm(indices, desc="Processing samples"):
            row = ds[idx]
            text = row.get(text_col, '')
            label_raw = row.get(label_col)
            
            if not text or label_raw is None:
                continue
            
            # Map label
            label = label_map.get(label_raw, label_raw)
            if label not in [0, 1]:
                continue
            
            # Filter short texts
            if len(text.split()) < 20:
                continue
            
            samples.append(TextSample(
                text=text,
                label=int(label),
                source=config['name'],
                metadata={'original_label': label_raw}
            ))
        
        return samples
    
    def _process_paired_dataset(self, ds, config: Dict,
                                 max_samples: Optional[int]) -> List[TextSample]:
        """Process a dataset with human/AI text pairs."""
        samples = []
        human_col = config['text_column']
        ai_col = config['ai_column']
        
        indices = list(range(len(ds)))
        if max_samples and len(indices) > max_samples // 2:
            indices = random.sample(indices, max_samples // 2)
        
        for idx in tqdm(indices, desc="Processing pairs"):
            row = ds[idx]
            
            # Human texts
            human_texts = row.get(human_col, [])
            if isinstance(human_texts, str):
                human_texts = [human_texts]
            
            for text in human_texts:
                if text and len(text.split()) >= 20:
                    samples.append(TextSample(
                        text=text,
                        label=0,
                        source=config['name'],
                        metadata={'type': 'human'}
                    ))
            
            # AI texts
            ai_texts = row.get(ai_col, [])
            if isinstance(ai_texts, str):
                ai_texts = [ai_texts]
            
            for text in ai_texts:
                if text and len(text.split()) >= 20:
                    samples.append(TextSample(
                        text=text,
                        label=1,
                        source=config['name'],
                        metadata={'type': 'ai'}
                    ))
        
        return samples
    
    def load_multiple_datasets(self, dataset_keys: List[str],
                               max_per_dataset: Optional[int] = None,
                               balance: bool = True) -> List[TextSample]:
        """Load and combine multiple datasets."""
        all_samples = []
        
        for key in dataset_keys:
            samples = self.load_dataset(key, max_samples=max_per_dataset)
            all_samples.extend(samples)
        
        if balance:
            all_samples = self._balance_classes(all_samples)
        
        self.samples = all_samples
        return all_samples
    
    def _balance_classes(self, samples: List[TextSample]) -> List[TextSample]:
        """Balance the dataset to have equal human/AI samples."""
        human_samples = [s for s in samples if s.label == 0]
        ai_samples = [s for s in samples if s.label == 1]
        
        min_count = min(len(human_samples), len(ai_samples))
        
        if len(human_samples) > min_count:
            human_samples = random.sample(human_samples, min_count)
        if len(ai_samples) > min_count:
            ai_samples = random.sample(ai_samples, min_count)
        
        balanced = human_samples + ai_samples
        random.shuffle(balanced)
        
        print(f"Balanced dataset: {len(human_samples)} human, {len(ai_samples)} AI")
        return balanced
    
    def get_train_test_split(self, test_ratio: float = 0.2,
                             seed: int = 42) -> Tuple[List[TextSample], List[TextSample]]:
        """Split samples into train/test sets."""
        random.seed(seed)
        samples = self.samples.copy()
        random.shuffle(samples)
        
        split_idx = int(len(samples) * (1 - test_ratio))
        return samples[:split_idx], samples[split_idx:]
    
    def save_samples(self, filepath: str):
        """Save samples to JSON file."""
        data = [
            {
                'text': s.text,
                'label': s.label,
                'source': s.source,
                'metadata': s.metadata
            }
            for s in self.samples
        ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} samples to {filepath}")
    
    def load_samples(self, filepath: str):
        """Load samples from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = [
            TextSample(
                text=d['text'],
                label=d['label'],
                source=d.get('source', 'unknown'),
                metadata=d.get('metadata')
            )
            for d in data
        ]
        
        print(f"Loaded {len(self.samples)} samples from {filepath}")
        return self.samples
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.samples:
            return {}
        
        human_count = sum(1 for s in self.samples if s.label == 0)
        ai_count = sum(1 for s in self.samples if s.label == 1)
        
        word_counts = [len(s.text.split()) for s in self.samples]
        
        sources = {}
        for s in self.samples:
            sources[s.source] = sources.get(s.source, 0) + 1
        
        return {
            'total_samples': len(self.samples),
            'human_samples': human_count,
            'ai_samples': ai_count,
            'balance_ratio': human_count / ai_count if ai_count > 0 else float('inf'),
            'avg_word_count': np.mean(word_counts),
            'min_word_count': min(word_counts),
            'max_word_count': max(word_counts),
            'sources': sources
        }


def load_default_datasets(max_per_dataset: int = 5000) -> DatasetLoader:
    """Load default combination of datasets for training."""
    loader = DatasetLoader()
    
    # Try loading available datasets
    datasets_to_try = [
        'ai-text-detection-pile',
    ]
    
    for ds_key in datasets_to_try:
        try:
            samples = loader.load_dataset(ds_key, max_samples=max_per_dataset)
            loader.samples.extend(samples)
        except Exception as e:
            print(f"Could not load {ds_key}: {e}")
    
    if loader.samples:
        loader.samples = loader._balance_classes(loader.samples)
    
    return loader


if __name__ == '__main__':
    # Test loading
    print("Available datasets:")
    for key, config in DatasetLoader.DATASETS.items():
        print(f"  {key}: {config['description']}")
    
    print("\nAttempting to load datasets...")
    loader = load_default_datasets(max_per_dataset=100)
    
    if loader.samples:
        stats = loader.get_statistics()
        print("\nDataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
