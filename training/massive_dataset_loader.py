#!/usr/bin/env python3
"""
Veritas Sunrise Training Pipeline
Massive-scale ML training using ALL available HuggingFace AI detection datasets.
Trains on every possible dataset to achieve optimal detection parameters.
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter

import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset, Dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Error: datasets library required. Run: pip install datasets")
    sys.exit(1)


@dataclass
class TextSample:
    """A single text sample with label."""
    text: str
    label: int  # 0 = human, 1 = AI
    source: str
    metadata: Optional[Dict] = None


# ============================================================================
# MASSIVE DATASET REGISTRY - EVERY KNOWN AI DETECTION DATASET
# ============================================================================

MASSIVE_DATASET_REGISTRY = {
    # ========== PRIMARY AI DETECTION DATASETS ==========
    'artem9k/ai-text-detection-pile': {
        'text_column': 'text',
        'label_column': 'source',
        'label_map': {'human': 0, 'ai': 1, 'Human': 0, 'AI': 1},
        'priority': 1,
        'description': 'Large AI detection pile - 1.4M samples'
    },
    'srikanthgali/ai-text-detection-pile-cleaned': {
        'text_column': 'text',
        'label_column': 'source', 
        'label_map': {'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'Cleaned AI detection pile'
    },
    'coai/ai-text-detection-training': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'COAI AI text detection training set'
    },
    'coai/ai-text-detection-benchmark': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'COAI benchmark dataset'
    },
    
    # ========== HUMAN VS AI DATASETS ==========
    'NicolaiSivesind/human-vs-machine': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'machine': 1},
        'priority': 1,
        'description': 'Human vs machine text classification'
    },
    'Yashodhar29/finalized-ai-vs-human-300K': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': '300K human vs AI samples'
    },
    'Yashodhar29/finalized-ai-vs-human-100k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': '100K human vs AI samples'
    },
    'Yashodhar29/finalized-ai-vs-human-50k': {
        'text_column': 'text',
        'label_column': 'label', 
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': '50K human vs AI samples'
    },
    'Yashodhar29/finalized-ai-vs-human-40k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': '40K human vs AI samples'
    },
    'Yashodhar29/finalized-ai-vs-human-10k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': '10K human vs AI samples'
    },
    'Ransaka/ai-vs-human-generated-dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'AI vs human generated text'
    },
    'ilyasoulk/ai-vs-human': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI vs human text dataset'
    },
    'danibor/human-vs-ai-spanish-65k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 2,
        'description': 'Spanish human vs AI 65K samples'
    },
    'andythetechnerd03/AI-human-text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI and human text classification'
    },
    'ardavey/human-ai-generated-text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'Human vs AI generated text'
    },
    'Monteiroo/human_ai_pt-br': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Portuguese human vs AI'
    },
    'bulkbeings/human-ai-v3': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Human AI dataset v3'
    },
    'okemdad/human_vs_ai_master_dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'Master human vs AI dataset'
    },
    
    # ========== CHATGPT DETECTION DATASETS ==========
    'FreedomIntelligence/ChatGPT-Detection-PR-HPPT': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'chatgpt': 1},
        'priority': 1,
        'description': 'ChatGPT detection dataset'
    },
    'Ateeqq/AI-and-Human-Generated-Text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI and Human generated text'
    },
    
    # ========== GPT OUTPUT DATASETS ==========
    'anugrahap/gpt2-output': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'gpt2': 1},
        'priority': 2,
        'description': 'GPT-2 output detection'
    },
    'spacerini/gpt2-outputs': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'GPT-2 outputs'
    },
    'byunggill/gpt-2-output': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'GPT-2 output samples'
    },
    
    # ========== LLM DETECTION DATASETS ==========
    'Zarakun/llm-detection-dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'llm': 1},
        'priority': 1,
        'description': 'LLM detection dataset'
    },
    'jjz5463/llm-detection-generation-1.0': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'LLM detection generation v1.0'
    },
    
    # ========== MACHINE GENERATED TEXT ==========
    'ThanaritKanjanametawatAU/Machine-Generated-Text-Detection-Dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'machine': 1},
        'priority': 1,
        'description': 'Machine generated text detection'
    },
    'readerbench/ro-human-machine-60k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'machine': 1},
        'priority': 2,
        'description': 'Romanian human vs machine 60K'
    },
    
    # ========== AI DETECTOR SPECIFIC ==========
    'TsingyuAI-Tech/ai-detector-ref-en': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'AI detector reference (English)'
    },
    'TsingyuAI-Tech/ai-detector-ref-cn': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'AI detector reference (Chinese)'
    },
    'mhb-maaz/ai-detector-dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI detector dataset'
    },
    'optimization-hashira/ai-text-detection-dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI text detection dataset'
    },
    'Varun53/AI_text_detection': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI text detection'
    },
    'akoukas/AITextDetectionDataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI text detection dataset'
    },
    'ninaaaaddd/AI_text_detection_dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI text detection dataset'
    },
    'silentone0725/ai-human-text-detection-v1': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI human text detection v1'
    },
    
    # ========== ESSAY/ARTICLE DATASETS ==========
    'artfultom/llm-generated-essays': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'llm': 1},
        'priority': 1,
        'description': 'LLM generated essays'
    },
    'artnitolog/llm-generated-texts': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'LLM generated texts'
    },
    'thisisHJLee/ai_essay': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI essay dataset'
    },
    'dshihk/llm-generated-essay': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'LLM generated essay'
    },
    'BlueSkyXN/AI-Essay': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI Essay dataset'
    },
    
    # ========== NEWS DATASETS ==========
    'AnhNguyen2299/vietnamese_news_human_ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 2,
        'description': 'Vietnamese news human vs AI'
    },
    'ICCIES-2025-DetectAI/vietnamese_news_human_ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Vietnamese news AI detection'
    },
    'trieunh/vietnamese.human.ai.news.sets': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Vietnamese human AI news'
    },
    'lvulpecula/ChatGPT-generated_fake_news_dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'ChatGPT generated fake news'
    },
    'lvulpecula/AI_rewritten_fake_news': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI rewritten fake news'
    },
    
    # ========== MULTILINGUAL DATASETS ==========
    'aycabostancioglu/Arabic-English-Turkish-AI-Human-Text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Multilingual AI human text'
    },
    'kanwal-mehreen18/Multilingual_Machine_Generated_Text_Detection': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Multilingual machine generated text'
    },
    'ko-human-ai/ko-human-ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 2,
        'description': 'Korean human vs AI'
    },
    'Asyq/kaz-lang-human-vs-ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Kazakh language human vs AI'
    },
    'Yuvrajg2107/Marathi-Human-vs-AI-Authentic-1k': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 2,
        'description': 'Marathi human vs AI'
    },
    
    # ========== SPECIALIZED DATASETS ==========
    'AlekseyKorshuk/ai-detection-gutenberg-human-formatted-ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Gutenberg AI detection'
    },
    'AlekseyKorshuk/ai-detection-booksum-complete-cleaned-human-ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'BookSum AI detection'
    },
    'hassanpositive/human-vs-ai-stories': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Human vs AI stories'
    },
    'ronanhansel/data-ai-slop-detector': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI slop detector dataset'
    },
    'Roxanne-WANG/AI-Text_Detection': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI text detection'
    },
    'Killerwhale-Park/AI_Generated_Text_Detection': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI generated text detection'
    },
    'dmitva/human_ai_generated_text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'Human AI generated text'
    },
    'NabeelShar/ai_and_human_text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI and human text'
    },
    'omarawad11/ai_human_text': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI human text'
    },
    'likhithasapu/ai-human-gen': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI human generation'
    },
    'ernanhughes/ai-human': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI human dataset'
    },
    'shahxeebhassan/human_vs_ai_sentences': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Human vs AI sentences'
    },
    'ahmadreza13/human-vs-Ai-generated-dataset': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Human vs AI generated'
    },
    'humancert/unique_ai_human': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Unique AI human'
    },
    'hjl/tulu_human_vs_ai': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Tulu human vs AI'
    },
    'ziq/ai-generated-text-classification': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI generated text classification'
    },
    'ash12321/ai-detector-benchmark-test-data': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'AI detector benchmark'
    },
    'Mharis205/ai-text-detection-pile': {
        'text_column': 'text',
        'label_column': 'source',
        'label_map': {'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'AI text detection pile copy'
    },
    'Mharis205/ai-text-detection-pile-cleaned': {
        'text_column': 'text',
        'label_column': 'source',
        'label_map': {'human': 0, 'ai': 1},
        'priority': 1,
        'description': 'Cleaned AI detection pile'
    },
    
    # ========== SYNTHETIC TEXT DATASETS (use with AI label) ==========
    'HAERAE-HUB/KOREAN-SyntheticText-1.5B': {
        'text_column': 'text',
        'synthetic_only': True,
        'priority': 3,
        'description': 'Korean synthetic text (AI-only)'
    },
    'OLAResearch/KOREAN-SyntheticText': {
        'text_column': 'text',
        'synthetic_only': True,
        'priority': 3,
        'description': 'Korean synthetic text'
    },
    'v-urushkin/SyntheticTexts6M': {
        'text_column': 'text',
        'synthetic_only': True,
        'priority': 3,
        'description': 'Synthetic texts 6M samples'
    },
    'tay-yozhik/SyntheticTexts': {
        'text_column': 'text',
        'synthetic_only': True,
        'priority': 3,
        'description': 'Synthetic texts'
    },
    
    # ========== HUMAN-ONLY DATASETS (use with human label) ==========
    'badhanr/wikipedia_human_written_text': {
        'text_column': 'text',
        'human_only': True,
        'priority': 2,
        'description': 'Wikipedia human written'
    },
    'Just999999/human-written-text-collection': {
        'text_column': 'text',
        'human_only': True,
        'priority': 2,
        'description': 'Human written text collection'
    },
    
    # ========== zcamz AI vs Human series ==========
    'zcamz/ai-vs-human-HuggingFaceTB-SmolLM2-1.7B-Instruct': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'SmolLM2 AI vs human'
    },
    'zcamz/ai-vs-human-Qwen-Qwen2.5-1.5B-Instruct': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Qwen AI vs human'
    },
    'zcamz/ai-vs-human-google-gemma-2-2b-it': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Gemma AI vs human'
    },
    'zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Llama 3.2 AI vs human'
    },
    
    # ========== ilyasoulk series ==========
    'ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Llama 3.1 8B AI vs human'
    },
    'ilyasoulk/ai-vs-human-meta-llama-Llama-3.1-8B-Instruct-CNN': {
        'text_column': 'text',
        'label_column': 'label',
        'label_map': {0: 0, 1: 1},
        'priority': 1,
        'description': 'Llama 3.1 8B CNN AI vs human'
    },
}


class MassiveDatasetLoader:
    """Load ALL available datasets for massive training."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(__file__), '.cache')
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        self.loaded_samples = []
        self.loading_stats = {}
        self.failed_datasets = []
        self.successful_datasets = []
        
    def load_all_datasets(self, 
                          max_samples_per_dataset: int = 50000,
                          min_priority: int = 3,
                          min_text_length: int = 50) -> List[TextSample]:
        """Load all datasets from the registry."""
        all_samples = []
        
        # Sort by priority
        sorted_datasets = sorted(
            MASSIVE_DATASET_REGISTRY.items(),
            key=lambda x: x[1].get('priority', 99)
        )
        
        print(f"\n{'='*60}")
        print(f"LOADING {len(sorted_datasets)} DATASETS")
        print(f"{'='*60}\n")
        
        for dataset_name, config in tqdm(sorted_datasets, desc="Loading datasets"):
            if config.get('priority', 99) > min_priority:
                continue
                
            try:
                samples = self._load_single_dataset(
                    dataset_name, config, max_samples_per_dataset, min_text_length
                )
                
                if samples:
                    all_samples.extend(samples)
                    self.successful_datasets.append({
                        'name': dataset_name,
                        'samples': len(samples),
                        'description': config.get('description', '')
                    })
                    self.loading_stats[dataset_name] = {
                        'samples_loaded': len(samples),
                        'status': 'success'
                    }
                    print(f"  ✓ {dataset_name}: {len(samples)} samples")
                    
            except Exception as e:
                self.failed_datasets.append({
                    'name': dataset_name,
                    'error': str(e)
                })
                self.loading_stats[dataset_name] = {
                    'samples_loaded': 0,
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"  ✗ {dataset_name}: {str(e)[:50]}")
        
        self.loaded_samples = all_samples
        return all_samples
    
    def _load_single_dataset(self, name: str, config: Dict, 
                             max_samples: int, min_length: int) -> List[TextSample]:
        """Load a single dataset with error handling."""
        samples = []
        
        try:
            # Try different splits
            for split in ['train', 'test', 'validation']:
                try:
                    ds = load_dataset(name, split=split, cache_dir=self.cache_dir)
                    break
                except:
                    continue
            else:
                # Try loading without split
                ds = load_dataset(name, cache_dir=self.cache_dir)
                if hasattr(ds, 'keys'):
                    split_name = list(ds.keys())[0]
                    ds = ds[split_name]
            
            # Get column info
            text_col = config.get('text_column', 'text')
            
            # Find text column if not specified
            if text_col not in ds.column_names:
                for possible_col in ['text', 'content', 'sentence', 'document', 'article', 'body']:
                    if possible_col in ds.column_names:
                        text_col = possible_col
                        break
            
            if text_col not in ds.column_names:
                return []
            
            # Handle different dataset types
            if config.get('synthetic_only'):
                samples = self._extract_synthetic_samples(ds, text_col, name, max_samples, min_length)
            elif config.get('human_only'):
                samples = self._extract_human_samples(ds, text_col, name, max_samples, min_length)
            else:
                samples = self._extract_labeled_samples(ds, config, name, max_samples, min_length)
                
        except Exception as e:
            raise e
            
        return samples
    
    def _extract_labeled_samples(self, ds, config: Dict, source: str,
                                  max_samples: int, min_length: int) -> List[TextSample]:
        """Extract samples with labels."""
        samples = []
        text_col = config.get('text_column', 'text')
        label_col = config.get('label_column', 'label')
        label_map = config.get('label_map', {0: 0, 1: 1})
        
        # Find label column
        if label_col not in ds.column_names:
            for possible_col in ['label', 'source', 'class', 'category', 'is_ai', 'generated']:
                if possible_col in ds.column_names:
                    label_col = possible_col
                    break
        
        if label_col not in ds.column_names:
            return []
        
        indices = list(range(min(len(ds), max_samples * 2)))
        random.shuffle(indices)
        
        for idx in indices[:max_samples]:
            try:
                row = ds[idx]
                text = str(row.get(text_col, ''))
                label_raw = row.get(label_col)
                
                if not text or len(text) < min_length:
                    continue
                
                # Try to map label
                label = label_map.get(label_raw, label_raw)
                
                # Handle string labels
                if isinstance(label, str):
                    label_lower = label.lower()
                    if label_lower in ['human', 'real', 'authentic', '0']:
                        label = 0
                    elif label_lower in ['ai', 'machine', 'generated', 'chatgpt', 'gpt', 'llm', '1']:
                        label = 1
                    else:
                        continue
                
                if label not in [0, 1]:
                    continue
                
                samples.append(TextSample(
                    text=text,
                    label=int(label),
                    source=source
                ))
                
            except:
                continue
        
        return samples
    
    def _extract_synthetic_samples(self, ds, text_col: str, source: str,
                                    max_samples: int, min_length: int) -> List[TextSample]:
        """Extract synthetic-only samples (labeled as AI)."""
        samples = []
        indices = list(range(min(len(ds), max_samples)))
        random.shuffle(indices)
        
        for idx in indices[:max_samples]:
            try:
                text = str(ds[idx].get(text_col, ''))
                if text and len(text) >= min_length:
                    samples.append(TextSample(text=text, label=1, source=source))
            except:
                continue
        
        return samples
    
    def _extract_human_samples(self, ds, text_col: str, source: str,
                                max_samples: int, min_length: int) -> List[TextSample]:
        """Extract human-only samples."""
        samples = []
        indices = list(range(min(len(ds), max_samples)))
        random.shuffle(indices)
        
        for idx in indices[:max_samples]:
            try:
                text = str(ds[idx].get(text_col, ''))
                if text and len(text) >= min_length:
                    samples.append(TextSample(text=text, label=0, source=source))
            except:
                continue
        
        return samples
    
    def balance_dataset(self, samples: List[TextSample]) -> List[TextSample]:
        """Balance human and AI samples."""
        human = [s for s in samples if s.label == 0]
        ai = [s for s in samples if s.label == 1]
        
        min_count = min(len(human), len(ai))
        
        random.shuffle(human)
        random.shuffle(ai)
        
        balanced = human[:min_count] + ai[:min_count]
        random.shuffle(balanced)
        
        print(f"Balanced: {min_count} human + {min_count} AI = {len(balanced)} total")
        
        return balanced
    
    def get_loading_report(self) -> Dict:
        """Get comprehensive loading report."""
        return {
            'total_datasets_attempted': len(MASSIVE_DATASET_REGISTRY),
            'successful_datasets': len(self.successful_datasets),
            'failed_datasets': len(self.failed_datasets),
            'total_samples_loaded': len(self.loaded_samples),
            'human_samples': sum(1 for s in self.loaded_samples if s.label == 0),
            'ai_samples': sum(1 for s in self.loaded_samples if s.label == 1),
            'datasets_used': self.successful_datasets,
            'datasets_failed': self.failed_datasets,
            'per_dataset_stats': self.loading_stats
        }


if __name__ == '__main__':
    # Test loading
    loader = MassiveDatasetLoader()
    samples = loader.load_all_datasets(max_samples_per_dataset=1000, min_priority=1)
    print(f"\nTotal samples: {len(samples)}")
    report = loader.get_loading_report()
    print(f"Successful: {report['successful_datasets']}")
    print(f"Failed: {report['failed_datasets']}")
