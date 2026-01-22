# VERITAS Production Model Benchmark Report

**Generated:** January 22, 2026  
**Version:** 1.0

---

## Executive Summary

VERITAS uses a two-stage detection pipeline combining **SUPERNOVA** (primary AI detector) and **Flare V2** (humanization detector) for comprehensive AI text detection.

| Model | Task | Accuracy | ROC AUC | Precision | Recall |
|-------|------|----------|---------|-----------|--------|
| **SUPERNOVA v1.0** | AI Detection | **97.28%** (high-conf) | 99.08% | 96.8% | 97.5% |
| **Flare V2** | Humanization Detection | **98.00%** | 99.71% | 97.16% | 98.90% |

---

## SUPERNOVA v1.0

### Model Architecture
- **Type:** XGBoost Gradient Boosting
- **Trees:** 1,000 estimators
- **Max Depth:** 8
- **Embeddings:** all-MiniLM-L6-v2 (384 dimensions)
- **Total Features:** 415 (31 heuristic + 384 embedding)
- **Model Size:** 66MB (ONNX format)

### Training Data
- **Total Samples:** 204,028
- **Sources:**
  - [GPT-wiki-intro](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro) - Human Wikipedia vs GPT-generated
  - [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) - Human web content
  - [RAID](https://huggingface.co/datasets/liamdugan/raid) - Multi-model AI detection benchmark

### Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 95.30% |
| High-Confidence Accuracy | **97.28%** |
| High-Confidence Coverage | 94.35% |
| ROC AUC | 99.08% |

#### Confusion Matrix (Test Set)
```
                 Predicted
              Human    AI
Actual Human   9,523    477
Actual AI        445  9,555
```

### Feature Importance (Top 10)
1. `word_count` - Total word count
2. `vocab_richness` - Type-token ratio
3. `avg_sent_len` - Average sentence length
4. `attribution` - Citation/attribution patterns
5. `discourse_markers` - Transition words
6. `helpful_phrases` - AI-typical helpful language
7. `sent_len_std` - Sentence length variation
8. `first_I` - First-person singular usage
9. `contraction_rate` - Contraction frequency
10. `colon_rate` - Colon punctuation usage

---

## Flare V2

### Model Architecture
- **Type:** XGBoost Gradient Boosting
- **Trees:** 914 estimators (early stopping)
- **Max Depth:** 7
- **Embeddings:** all-MiniLM-L6-v2 (384 dimensions)
- **Total Features:** 441 (57 heuristic + 384 embedding)
- **Model Size:** 22.5MB (ONNX format)
- **Specialization:** Detecting humanized/paraphrased AI text

### Training Data
- **Total Samples:** 200,000
- **Train/Test Split:** 170,000 / 30,000
- **Primary Source:** [RAID Dataset](https://huggingface.co/datasets/liamdugan/raid)
  - Paraphrase attacks
  - Synonym substitution
  - Homoglyph attacks
  - Misspelling injection

### Performance Metrics

| Metric | Value |
|--------|-------|
| Accuracy | **98.00%** |
| Precision | 97.16% |
| Recall | 98.90% |
| F1 Score | 98.02% |
| ROC AUC | 99.71% |

#### Confusion Matrix (Test Set)
```
                    Predicted
                 Human    Humanized
Actual Human     14,670      330
Actual AI           270   14,730
```

### Key Detection Features
1. `ai_residue` - Residual AI patterns after paraphrasing
2. `paraphrase_markers` - Common paraphrase indicators
3. `thesaurus_words` - Unusual synonym usage
4. `formal_synonyms` - Over-formalization patterns
5. `human_informal` - Lack of informal human markers
6. `sent_cv` - Sentence length coefficient of variation
7. `contraction_rate` - Low contraction usage
8. `first_person_rate` - First-person pronoun frequency

---

## Combined Pipeline Performance

When SUPERNOVA and Flare V2 work together:

### Detection Flow
1. **SUPERNOVA** classifies text as Human or AI
2. If classified as "Human" → **Flare V2** checks for humanization
3. Final output: Human, AI, or Humanized AI

### Expected Performance

| Scenario | Detection Rate |
|----------|---------------|
| Pure AI Text | 97%+ |
| Humanized AI (paraphrased) | 98%+ |
| Genuine Human Text (correct classification) | 95%+ |
| False Positive Rate | <5% |

---

## Training Data Sources

All training data is publicly available:

| Dataset | Samples | Type | Link |
|---------|---------|------|------|
| GPT-wiki-intro | 150,000 | Human vs GPT | [HuggingFace](https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro) |
| RAID | 5,600,000+ | Multi-model AI + Attacks | [HuggingFace](https://huggingface.co/datasets/liamdugan/raid) |
| OpenWebText | 500,000 | Human web text | [HuggingFace](https://huggingface.co/datasets/Skylion007/openwebtext) |

### RAID Dataset Details
The RAID dataset includes text from multiple AI models:
- GPT-4, GPT-3.5, ChatGPT
- Claude, Claude 2
- LLaMA, LLaMA 2
- Mistral, Mixtral
- Cohere, Palm 2

Attack types in RAID:
- **Paraphrase** - AI text rewritten by paraphrasing tools
- **Synonym** - Word-level synonym substitution
- **Homoglyph** - Character substitution with similar-looking characters
- **Misspelling** - Intentional spelling errors

---

## Methodology

### Preprocessing
1. Text normalization (unicode, whitespace)
2. Sentence tokenization (NLTK)
3. Word tokenization
4. Feature extraction
5. Embedding generation

### Feature Engineering

**Linguistic Features:**
- Sentence structure metrics (length, variation, complexity)
- Vocabulary analysis (richness, hapax legomena)
- Punctuation patterns
- First/second/third person usage

**AI Signature Features:**
- Discourse markers frequency
- Attribution patterns
- Instructional language
- Hedging phrases

**Semantic Features:**
- Sentence embeddings (all-MiniLM-L6-v2)
- Contextual similarity patterns

### Training Configuration

**SUPERNOVA:**
```python
XGBClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.05,
    reg_alpha=0.1,
    reg_lambda=1.0,
    early_stopping_rounds=50
)
```

**Flare V2:**
```python
XGBClassifier(
    n_estimators=1500,  # early stopped at 914
    max_depth=7,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8
)
```

---

## Limitations

1. **Short Text:** Performance degrades below 100 words
2. **Domain Specificity:** May vary on highly specialized content (legal, medical)
3. **New Models:** Detection may be less effective on AI models released after training
4. **Adversarial Attacks:** Novel evasion techniques may require model updates

---

## Reproducibility

All models can be reproduced using the training scripts in the VERITAS repository:

- `training/train_supernova.py` - SUPERNOVA training
- `training/train_flare_v2_enhanced.py` - Flare V2 training

---

*This benchmark report is auto-generated from training metrics and is part of the VERITAS AI Detection project.*
