#!/usr/bin/env python3
"""
Veritas Curated Dataset Registry
Hand-selected, quality-rated datasets for AI text detection training.

Each dataset has been evaluated based on:
- Download count (popularity/reliability indicator)
- Dataset size (more data = better generalization)
- Language (English primary, multilingual secondary)
- Label quality (clear human vs AI labels)
- Source reputation
- Specific use case for AI detection
"""

# ============================================================================
# TIER 1: EXCELLENT - Primary training datasets
# These are the gold standard - high downloads, large size, clear labels
# ============================================================================

TIER_1_EXCELLENT = [
    {
        'name': 'aadityaubhat/GPT-wiki-intro',
        'downloads': 500,
        'size': '150K+ samples',
        'language': 'English',
        'text_col': 'wiki_intro',
        'label_col': '_human',
        'label_map': {},
        'max_samples': 30000,
        'special': 'dual_column',  # Has both human (wiki_intro) and AI (generated_intro)
        'ai_text_col': 'generated_intro',
        'human_text_col': 'wiki_intro',
        'why_excellent': 'GPT vs Wikipedia intros. Parquet format (no loading scripts). '
                        'Same topics, direct comparison. Well-maintained.',
        'url': 'https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro'
    },
    {
        'name': 'imdb',
        'downloads': 50000,
        'size': '50K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': '_human_only',
        'label_map': {},
        'max_samples': 20000,
        'special': 'human_only',  # Pure human text
        'why_excellent': 'Movie reviews - 100% human written. High quality, '
                        'natural language. Great for human baseline.',
        'url': 'https://huggingface.co/datasets/imdb'
    },
    {
        'name': 'openwebtext',
        'downloads': 10000,
        'size': '8M+ samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': '_human_only',
        'label_map': {},
        'max_samples': 20000,
        'special': 'human_only',  # Pure human text
        'why_excellent': 'Web text from reddit links. 100% human written. '
                        'Diverse topics and styles.',
        'url': 'https://huggingface.co/datasets/openwebtext'
    },
]

# ============================================================================
# TIER 2: VERY GOOD - Strong secondary datasets  
# High quality, good size, reliable sources
# ============================================================================

TIER_2_VERY_GOOD = [
    {
        'name': 'billsum',
        'downloads': 5000,
        'size': '20K+ samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': '_human_only',
        'label_map': {},
        'max_samples': 10000,
        'special': 'human_only',
        'why_good': 'US Congressional bills. 100% human written. '
                   'Formal/technical language variety.',
        'url': 'https://huggingface.co/datasets/billsum'
    },
    {
        'name': 'squad',
        'downloads': 100000,
        'size': '100K+ samples',
        'language': 'English',
        'text_col': 'context',
        'label_col': '_human_only',
        'label_map': {},
        'max_samples': 15000,
        'special': 'human_only',
        'why_good': 'Stanford QA dataset. Wikipedia passages. '
                   '100% human-curated content.',
        'url': 'https://huggingface.co/datasets/squad'
    },
]

# ============================================================================
# TIER 3: GOOD - Useful supplementary datasets
# Smaller but still valuable for diversity
# ============================================================================

TIER_3_GOOD = [
    {
        'name': 'shahxeebhassan/human_vs_ai_sentences',
        'downloads': 89,
        'size': '100K-1M samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 20000,
        'why_useful': 'Sentence-level detection. Good for short text patterns. '
                     'Large dataset, reasonable download count.',
        'url': 'https://huggingface.co/datasets/shahxeebhassan/human_vs_ai_sentences'
    },
    {
        'name': 'Yashodhar29/finalized-ai-vs-human-300K',
        'downloads': 78,
        'size': '100K-1M samples',
        'language': 'Mixed',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 25000,
        'why_useful': '300K finalized samples. Good size, curated dataset. '
                     'The "finalized" suggests cleaned and verified.',
        'url': 'https://huggingface.co/datasets/Yashodhar29/finalized-ai-vs-human-300K'
    },
    {
        'name': 'ilyasoulk/ai-vs-human',
        'downloads': 70,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 10000,
        'why_useful': 'Smaller but high quality. Multiple related datasets from same author '
                     'covering different LLMs (Llama, etc).',
        'url': 'https://huggingface.co/datasets/ilyasoulk/ai-vs-human'
    },
    {
        'name': 'Ransaka/ai-vs-human-generated-dataset',
        'downloads': 59,
        'size': '10K-100K samples',
        'language': 'Mixed',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'ai': 1},
        'max_samples': 15000,
        'why_useful': 'Well-structured AI vs human dataset. '
                     'Good size for supplementary training.',
        'url': 'https://huggingface.co/datasets/Ransaka/ai-vs-human-generated-dataset'
    },
    {
        'name': 'jjz5463/llm-detection-generation-1.0',
        'downloads': 48,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 10000,
        'why_useful': 'Specifically designed for LLM detection. '
                     'Modern dataset with recent LLM outputs.',
        'url': 'https://huggingface.co/datasets/jjz5463/llm-detection-generation-1.0'
    },
    {
        'name': 'abhi099k/machine-and-human-text',
        'downloads': 47,
        'size': '10K-100K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1, 'human': 0, 'machine': 1},
        'max_samples': 15000,
        'why_useful': 'Machine vs human classification. Good for variety. '
                     'Different labeling approach may catch different patterns.',
        'url': 'https://huggingface.co/datasets/abhi099k/machine-and-human-text'
    },
]

# ============================================================================
# TIER 4: SPECIALIZED - Niche but valuable
# Model-specific or unique approaches
# ============================================================================

TIER_4_SPECIALIZED = [
    {
        'name': 'zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct',
        'downloads': 44,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 5000,
        'why_specialized': 'Specific to Llama 3.2 outputs. Helps detect modern LLM patterns.',
        'url': 'https://huggingface.co/datasets/zcamz/ai-vs-human-meta-llama-Llama-3.2-1B-Instruct'
    },
    {
        'name': 'zcamz/ai-vs-human-google-gemma-2-2b-it',
        'downloads': 35,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 5000,
        'why_specialized': 'Specific to Google Gemma outputs. Catches Google-specific patterns.',
        'url': 'https://huggingface.co/datasets/zcamz/ai-vs-human-google-gemma-2-2b-it'
    },
    {
        'name': 'artnitolog/llm-generated-texts',
        'downloads': 34,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 5000,
        'why_specialized': 'Various LLM-generated texts. Good for generalization.',
        'url': 'https://huggingface.co/datasets/artnitolog/llm-generated-texts'
    },
    {
        'name': 'Lyra-stellAI/AI_Human_generated_movie_reviews',
        'downloads': 33,
        'size': '10K-100K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 10000,
        'why_specialized': 'Domain-specific (movie reviews). Tests detection in creative writing.',
        'url': 'https://huggingface.co/datasets/Lyra-stellAI/AI_Human_generated_movie_reviews'
    },
    {
        'name': 'coai/ai-text-detection-benchmark',
        'downloads': 33,
        'size': '1K-10K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 5000,
        'why_specialized': 'Academic benchmark dataset. Well-validated ground truth.',
        'url': 'https://huggingface.co/datasets/coai/ai-text-detection-benchmark'
    },
    {
        'name': 'bulkbeings/human-ai-v3',
        'downloads': 24,
        'size': '10K-100K samples',
        'language': 'English',
        'text_col': 'text',
        'label_col': 'label',
        'label_map': {0: 0, 1: 1},
        'max_samples': 10000,
        'why_specialized': 'Version 3 indicates iterative improvement. Curated quality.',
        'url': 'https://huggingface.co/datasets/bulkbeings/human-ai-v3'
    },
]

# ============================================================================
# REJECTED DATASETS - Not suitable for training
# ============================================================================

REJECTED_DATASETS = {
    'CoIR-Retrieval/synthetic-text2sql': 'Not AI detection - SQL query generation',
    'philschmid/gretel-synthetic-text-to-sql': 'Not AI detection - SQL synthesis',
    'ritaranx/clinical-synthetic-text-*': 'Medical domain, not general AI detection',
    'pszemraj/synthetic-text-similarity': 'Text similarity, not detection',
    'hllj/synthetic-text-embedding': 'Vietnamese + embeddings, not detection',
    'argilla/synthetic-text-classification-news': 'Too small (<1K samples)',
    'Yuvrajg2107/ai_vs_human_images_*': 'Image data, not text',
    'ICCIES-2025-DetectAI/vietnamese_news_human_ai': 'Vietnamese only',
    'wanghw/human-ai-comparison': 'Chinese only',
    'open-llm-leaderboard/*': 'Leaderboard metadata, not training data',
}

# ============================================================================
# COMBINED CURATED LIST - What we'll actually use
# ============================================================================

def get_all_curated_datasets():
    """Return all curated datasets in priority order."""
    all_datasets = []
    
    for ds in TIER_1_EXCELLENT:
        ds['tier'] = 1
        ds['tier_name'] = 'EXCELLENT'
        all_datasets.append(ds)
    
    for ds in TIER_2_VERY_GOOD:
        ds['tier'] = 2
        ds['tier_name'] = 'VERY_GOOD'
        all_datasets.append(ds)
    
    for ds in TIER_3_GOOD:
        ds['tier'] = 3
        ds['tier_name'] = 'GOOD'
        all_datasets.append(ds)
    
    for ds in TIER_4_SPECIALIZED:
        ds['tier'] = 4
        ds['tier_name'] = 'SPECIALIZED'
        all_datasets.append(ds)
    
    return all_datasets


def print_dataset_summary():
    """Print summary of curated datasets."""
    datasets = get_all_curated_datasets()
    
    print("\n" + "="*70)
    print("VERITAS CURATED DATASET REGISTRY")
    print("="*70)
    
    print(f"\nTIER 1 - EXCELLENT ({len(TIER_1_EXCELLENT)} datasets):")
    print("-" * 50)
    for ds in TIER_1_EXCELLENT:
        print(f"  ★ {ds['name']}")
        print(f"    {ds['downloads']} downloads | {ds['size']} | {ds['language']}")
        print(f"    → {ds['why_excellent'][:70]}...")
    
    print(f"\nTIER 2 - VERY GOOD ({len(TIER_2_VERY_GOOD)} datasets):")
    print("-" * 50)
    for ds in TIER_2_VERY_GOOD:
        print(f"  ● {ds['name']}")
        print(f"    {ds['downloads']} downloads | {ds['size']}")
    
    print(f"\nTIER 3 - GOOD ({len(TIER_3_GOOD)} datasets):")
    print("-" * 50)
    for ds in TIER_3_GOOD:
        print(f"  ○ {ds['name']}")
    
    print(f"\nTIER 4 - SPECIALIZED ({len(TIER_4_SPECIALIZED)} datasets):")
    print("-" * 50)
    for ds in TIER_4_SPECIALIZED:
        print(f"  ◇ {ds['name']}")
    
    total = len(datasets)
    total_max_samples = sum(ds['max_samples'] for ds in datasets)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {total} curated datasets")
    print(f"MAX POTENTIAL SAMPLES: ~{total_max_samples:,}")
    print(f"{'='*70}")


if __name__ == '__main__':
    print_dataset_summary()
