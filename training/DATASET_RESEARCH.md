# AI Text Detection Dataset Research

## Summary

This document catalogs publicly available datasets for training AI text detection models, with a focus on:
1. **Humanized/Paraphrased AI Text** - AI-generated text that has been modified to evade detection
2. **Pure Human Text** - Genuine human-written text for baseline training

**Total Available Samples**: >15 million samples across all datasets

---

## 🔴 Category 1: Humanized/Paraphrased AI Text (CRITICAL FOR TRAINING)

These datasets contain AI-generated text with various obfuscation/paraphrasing attacks applied.

### 1. RAID Dataset (★★★★★ HIGHEST PRIORITY)
- **HuggingFace ID**: `liamdugan/raid`
- **URL**: https://huggingface.co/datasets/liamdugan/raid
- **Size**: **5,615,820 samples** (1M-10M category)
- **Category**: Humanized AI + Raw AI + Human
- **Description**: The most comprehensive AI detection benchmark with multiple adversarial attacks

**Attack Types Included**:
| Attack | Count | Description |
|--------|-------|-------------|
| `none` | 467,985 | Unmodified AI text |
| `paraphrase` | 467,985 | **Paraphrased to evade detection** |
| `synonym` | 467,985 | Synonym substitution |
| `homoglyph` | 467,985 | Unicode homoglyph replacement |
| `whitespace` | 467,985 | Whitespace manipulation |
| `upper_lower` | 467,985 | Case modifications |
| `perplexity_misspelling` | 467,985 | Strategic misspellings |
| `insert_paragraphs` | 467,985 | Paragraph insertion |
| `article_deletion` | 467,985 | Article word removal |
| `alternative_spelling` | 467,985 | Alternative spellings |
| `number` | 467,985 | Number format changes |
| `zero_width_space` | 467,985 | Invisible character insertion |

**Models Covered**: `human`, `gpt2`, `gpt3`, `gpt4`, `chatgpt`, `cohere`, `cohere-chat`, `llama-chat`, `mistral`, `mistral-chat`, `mpt`, `mpt-chat`

**Domains**: `abstracts`, `books`, `news`, `poetry`, `recipes`, `reddit`, `reviews`, `wiki`

**How to Load**:
```python
from datasets import load_dataset

# Load full dataset (streaming recommended due to size)
raid = load_dataset("liamdugan/raid", split="train", streaming=True)

# Filter for paraphrased samples only
paraphrased = [s for s in raid if s['attack'] == 'paraphrase' and s['model'] != 'human']
```

---

### 2. dmitva/human_ai_generated_text (★★★★☆)
- **HuggingFace ID**: `dmitva/human_ai_generated_text`
- **URL**: https://huggingface.co/datasets/dmitva/human_ai_generated_text
- **Size**: 1M-10M samples
- **Category**: Human + AI (paired)
- **Description**: Large paired dataset with human text and AI-generated counterparts

**Columns**: `id`, `human_text`, `ai_text`, `instructions`

**How to Load**:
```python
from datasets import load_dataset
ds = load_dataset("dmitva/human_ai_generated_text", split="train")
```

---

### 3. GPT-Wiki-Intro (★★★★☆)
- **HuggingFace ID**: `aadityaubhat/GPT-wiki-intro`
- **URL**: https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro
- **Size**: ~150,000 samples (100K-1M)
- **Category**: Human + AI (paired Wikipedia introductions)
- **Description**: Human-written Wikipedia intros paired with GPT-generated versions

**Columns**: `wiki_intro` (human), `generated_intro` (AI), `title`, `prompt`

**How to Load**:
```python
from datasets import load_dataset
ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train")
# Human text: ds['wiki_intro']
# AI text: ds['generated_intro']
```

---

### 4. artem9k/ai-text-detection-pile (★★★★☆)
- **HuggingFace ID**: `artem9k/ai-text-detection-pile`
- **URL**: https://huggingface.co/datasets/artem9k/ai-text-detection-pile
- **Size**: ~1.4M samples (1M-10M)
- **Category**: Human + AI (labeled)
- **Description**: Large pile of AI vs human text samples with binary labels

**Columns**: `source` (human/ai), `id`, `text`

**How to Load**:
```python
from datasets import load_dataset
ds = load_dataset("artem9k/ai-text-detection-pile", split="train")
# Filter by label: ds.filter(lambda x: x['source'] == 'ai')
```

---

### 5. Ateeqq/AI-and-Human-Generated-Text (★★★☆☆)
- **HuggingFace ID**: `Ateeqq/AI-and-Human-Generated-Text`
- **URL**: https://huggingface.co/datasets/Ateeqq/AI-and-Human-Generated-Text
- **Size**: 10K-100K samples
- **Category**: Human + AI
- **Description**: Mixed AI and human-generated text dataset

---

### 6. andythetechnerd03/AI-human-text (★★★☆☆)
- **HuggingFace ID**: `andythetechnerd03/AI-human-text`
- **URL**: https://huggingface.co/datasets/andythetechnerd03/AI-human-text
- **Size**: 100K-1M samples
- **Category**: Human + AI
- **Description**: Binary labeled AI vs human text

---

### 7. ahmadreza13/human-vs-Ai-generated-dataset (★★★☆☆)
- **HuggingFace ID**: `ahmadreza13/human-vs-Ai-generated-dataset`
- **URL**: https://huggingface.co/datasets/ahmadreza13/human-vs-Ai-generated-dataset
- **Size**: 1M-10M samples
- **Category**: Human + AI
- **Description**: Large human vs AI detection dataset

---

### 8. Hello-SimpleAI/HC3 (Human ChatGPT Comparison) (★★★☆☆)
- **HuggingFace ID**: `Hello-SimpleAI/HC3`
- **URL**: https://huggingface.co/datasets/Hello-SimpleAI/HC3
- **Size**: 10K-100K samples
- **Category**: Human + ChatGPT (paired Q&A)
- **Description**: Human answers vs ChatGPT answers to the same questions
- **Note**: Uses dataset script (may need older `datasets` version)

---

### 9. silentone0725/ai-human-text-detection-v1 (★★☆☆☆)
- **HuggingFace ID**: `silentone0725/ai-human-text-detection-v1`
- **Size**: 10K-100K samples
- **Category**: Human + AI

---

### 10. abhi099k/machine-and-human-text (★★☆☆☆)
- **HuggingFace ID**: `abhi099k/machine-and-human-text`
- **Size**: 10K-100K samples
- **Category**: Human + Machine

---

## 🔴 Additional Humanization-Related Datasets

### Academic/Research Datasets (May Require Manual Download)

| Dataset | Source | Description |
|---------|--------|-------------|
| **OUTFOX** | Academic Paper | Adversarial paraphrasing attacks on AI detectors |
| **DIPPER** | Academic Paper | Discourse paraphrase attacks |
| **SynSciPass** | Academic Paper | Scientific text with paraphrasing |
| **TuringBench** | `turingbench/TuringBench` | Multi-generator benchmark (script-based) |
| **Ghostbuster** | Various repos | Anti-AI detection evaluation set |

---

## 🟢 Category 2: Pure Human Text (Baseline)

### 1. Skylion007/openwebtext (★★★★★)
- **HuggingFace ID**: `Skylion007/openwebtext`
- **URL**: https://huggingface.co/datasets/Skylion007/openwebtext
- **Size**: ~8M samples (1M-10M)
- **Category**: Human (web content)
- **Description**: Open recreation of OpenAI's WebText dataset, all human-written

```python
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
```

---

### 2. euclaise/writingprompts (★★★★★)
- **HuggingFace ID**: `euclaise/writingprompts`
- **URL**: https://huggingface.co/datasets/euclaise/writingprompts
- **Size**: ~300K samples (100K-1M)
- **Category**: Human (creative writing)
- **Description**: Reddit r/WritingPrompts stories - authentic human creative writing

```python
from datasets import load_dataset
ds = load_dataset("euclaise/writingprompts", split="train")
```

---

### 3. stanfordnlp/imdb (★★★★★)
- **HuggingFace ID**: `stanfordnlp/imdb`
- **URL**: https://huggingface.co/datasets/stanfordnlp/imdb
- **Size**: 50K samples
- **Category**: Human (movie reviews)
- **Description**: Classic IMDB movie review dataset, all human-written

```python
from datasets import load_dataset
ds = load_dataset("stanfordnlp/imdb", split="train")
```

---

### 4. bookcorpus/bookcorpus (★★★★☆)
- **HuggingFace ID**: `bookcorpus/bookcorpus`
- **URL**: https://huggingface.co/datasets/bookcorpus/bookcorpus
- **Size**: 10M-100M samples (sentences from books)
- **Category**: Human (literature)
- **Description**: Text from thousands of unpublished books

---

### 5. SocialGrep/one-million-reddit-confessions (★★★★☆)
- **HuggingFace ID**: `SocialGrep/one-million-reddit-confessions`
- **URL**: https://huggingface.co/datasets/SocialGrep/one-million-reddit-confessions
- **Size**: 1M samples
- **Category**: Human (Reddit posts)
- **Description**: Reddit confessions - authentic human writing

---

### 6. ctr4si/reddit_tifu (★★★★☆)
- **HuggingFace ID**: `ctr4si/reddit_tifu`
- **URL**: https://huggingface.co/datasets/ctr4si/reddit_tifu
- **Size**: 100K-1M samples
- **Category**: Human (Reddit stories)
- **Description**: r/tifu stories - casual human storytelling

---

### 7. defunct-datasets/eli5 (★★★★☆)
- **HuggingFace ID**: `defunct-datasets/eli5`
- **URL**: https://huggingface.co/datasets/defunct-datasets/eli5
- **Size**: 100K-1M samples
- **Category**: Human (Reddit Q&A)
- **Description**: "Explain Like I'm 5" - human explanations

---

### 8. qwedsacf/ivypanda-essays (★★★★☆)
- **HuggingFace ID**: `qwedsacf/ivypanda-essays`
- **URL**: https://huggingface.co/datasets/qwedsacf/ivypanda-essays
- **Size**: 100K-1M samples
- **Category**: Human (academic essays)
- **Description**: Student academic essays - excellent for academic writing patterns

**Columns**: `TEXT`, `SOURCE`

---

### 9. jonathanli/human-essays-reddit (★★★☆☆)
- **HuggingFace ID**: `jonathanli/human-essays-reddit`
- **URL**: https://huggingface.co/datasets/jonathanli/human-essays-reddit
- **Size**: 10K-100K samples
- **Category**: Human (Reddit writing prompts)

---

### 10. gfissore/arxiv-abstracts-2021 (★★★☆☆)
- **HuggingFace ID**: `gfissore/arxiv-abstracts-2021`
- **URL**: https://huggingface.co/datasets/gfissore/arxiv-abstracts-2021
- **Size**: 1M-10M samples
- **Category**: Human (academic abstracts)
- **Description**: ArXiv paper abstracts (pre-LLM era, so genuinely human)

---

### 11. ChristophSchuhmann/essays-with-instructions (★★★☆☆)
- **HuggingFace ID**: `ChristophSchuhmann/essays-with-instructions`
- **Size**: 1K-10K samples
- **Category**: Human (essays)

---

### 12. community-datasets/gutenberg_time (★★★☆☆)
- **HuggingFace ID**: `community-datasets/gutenberg_time`
- **URL**: https://huggingface.co/datasets/community-datasets/gutenberg_time
- **Size**: 100K-1M samples
- **Category**: Human (classic literature)
- **Description**: Project Gutenberg books - classic literature

---

## 📊 Dataset Size Summary

| Category | Dataset | Approximate Samples |
|----------|---------|-------------------|
| **Humanized/Paraphrased** | liamdugan/raid (paraphrase attack only) | ~470,000 |
| **Humanized/Paraphrased** | liamdugan/raid (all attacks) | ~5,150,000 |
| **AI vs Human** | artem9k/ai-text-detection-pile | ~1,400,000 |
| **AI vs Human** | dmitva/human_ai_generated_text | ~1,500,000 |
| **AI vs Human** | aadityaubhat/GPT-wiki-intro | ~150,000 |
| **AI vs Human** | ahmadreza13/human-vs-Ai-generated-dataset | ~1,000,000 |
| **Pure Human** | Skylion007/openwebtext | ~8,000,000 |
| **Pure Human** | bookcorpus/bookcorpus | ~70,000,000 |
| **Pure Human** | euclaise/writingprompts | ~300,000 |
| **Pure Human** | stanfordnlp/imdb | ~50,000 |
| **Pure Human** | SocialGrep/one-million-reddit-confessions | ~1,000,000 |
| **Pure Human** | qwedsacf/ivypanda-essays | ~100,000 |
| | **TOTAL** | **>88,000,000** |

---

## 🚀 Recommended Loading Strategy

### Priority 1: Humanized AI Text (Most Important)

```python
from datasets import load_dataset

# RAID with paraphrase attacks (PRIMARY SOURCE)
raid = load_dataset("liamdugan/raid", split="train", streaming=True)

# Filter for humanized samples
humanized_samples = []
for sample in raid:
    if sample['attack'] == 'paraphrase' and sample['model'] != 'human':
        humanized_samples.append({
            'text': sample['generation'],
            'label': 'humanized',
            'model': sample['model'],
            'domain': sample['domain']
        })
    if len(humanized_samples) >= 50000:
        break

print(f"Collected {len(humanized_samples)} humanized samples")
```

### Priority 2: Pure Human Text

```python
# Combine multiple human sources
human_sources = [
    ("Skylion007/openwebtext", "text", 30000),
    ("euclaise/writingprompts", "story", 20000),
    ("stanfordnlp/imdb", "text", 10000),
    ("qwedsacf/ivypanda-essays", "TEXT", 10000),
]

human_samples = []
for ds_name, text_col, limit in human_sources:
    ds = load_dataset(ds_name, split="train", streaming=True)
    count = 0
    for sample in ds:
        text = sample.get(text_col, '')
        if text and len(text.split()) >= 50:
            human_samples.append({'text': text, 'label': 'human'})
            count += 1
            if count >= limit:
                break
```

### Priority 3: Raw AI Text (for comparison)

```python
# RAID raw AI (no attacks)
raw_ai = []
for sample in raid:
    if sample['attack'] == 'none' and sample['model'] != 'human':
        raw_ai.append({
            'text': sample['generation'],
            'label': 'ai',
            'model': sample['model']
        })
    if len(raw_ai) >= 30000:
        break
```

---

## 🔧 Complete Dataset Loader Script

See [training/humanized_dataset_loader.py](humanized_dataset_loader.py) for a complete implementation.

---

## 📚 Academic Papers to Reference

1. **RAID**: "RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors" (Dugan et al., 2024)
2. **DIPPER**: "Paraphrase Generation for Machine-Generated Text Detection" 
3. **OUTFOX**: "OUTFOX: LLM-Generated Essay Detection Through In-Context Learning with Adversarially Generated Examples"
4. **Ghostbuster**: "Ghostbuster: Detecting Text Ghostwritten by Large Language Models"
5. **DetectGPT**: "DetectGPT: Zero-Shot Machine-Generated Text Detection"

---

## ⚠️ Important Notes

1. **Pre-2022 Data**: For human text baselines, prefer datasets with content from before November 2022 (pre-ChatGPT) to ensure genuine human authorship
2. **Domain Matching**: Match domains between human and AI text (e.g., academic essays vs academic essays)
3. **Streaming**: Use streaming for large datasets to avoid memory issues
4. **Balanced Training**: Aim for roughly equal samples of humanized, raw AI, and human text
5. **Gated Datasets**: Some datasets require HuggingFace authentication (`NicolaiSivesind/human-vs-machine`, `antebe1/paraphrased_AI_text`)
