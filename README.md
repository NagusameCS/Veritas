# ◈ Veritas

**AI Text Detection Engine — Multi-Model Analysis System**

[![npm version](https://img.shields.io/npm/v/veritas-ai-detector.svg)](https://www.npmjs.com/package/veritas-ai-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Veritas uses ML-trained statistical variance analysis to detect AI-generated text with **up to 99.24% accuracy**. Unlike pattern-matching approaches, it analyzes the natural variability in human writing—sentence lengths, vocabulary diversity, word frequency distributions, entropy patterns, and more.

## Model Suite

Veritas includes multiple specialized models for maximum accuracy:

| Model | Accuracy | ROC AUC | Specialty |
|-------|----------|---------|-----------|
| **Helios** | 99.24% | 99.98% | Tone + style detection (45 features) |
| **Zenith** | 99.57% | 99.97% | GPTZero-style perplexity analysis |
| **Sunrise** | 98.08% | 99.80% | Statistical variance (primary) |
| **Dawn** | 84.9% | - | Legacy baseline detector |

### Model Comparison

```
                    Human Detection    AI Detection    Humanized Detection
Sunrise             100%               99.2%           66.7%
Zenith              100%               100%            86.7%
Helios              99.2%              99.2%           -
```

**Zenith excels at humanized AI detection** (86.7% vs 66.7% for Sunrise).

## Key Features

- **Multi-Model Ensemble**: 4 specialized models for different detection scenarios
- **99%+ Accuracy**: Helios model with tone detection achieves 99.24%
- **Humanized AI Detection**: Zenith model specifically catches AI text modified by bypass tools
- **Bidirectional Analysis**: Flags both AI uniformity AND humanizer chaos
- **45+ Feature Analysis**: Tone, hedging, personal voice, coherence, and more
- **Statistical Foundation**: Zipf's Law, TTR, Hapax, Entropy, Burstiness
- **Confidence Intervals**: 95% CI with honest uncertainty estimates
- **No External APIs**: Runs entirely client-side (web) or locally (CLI)

## Live Demo

Try Veritas now: **[https://nagusame.github.io/Veritas](https://nagusame.github.io/Veritas)**

## Installation

### CLI via npm

```bash
# Global installation
npm install -g veritas-ai-detector

# Or run directly with npx
npx veritas-ai-detector --help
```

## CLI Usage

```bash
# Analyze a text file
veritas essay.txt

# Analyze text directly
veritas -t "Your text to analyze here..."

# Read from stdin (pipe from other commands)
cat document.txt | veritas --stdin
echo "Some text" | veritas --stdin

# Output formats
veritas essay.txt -o text     # Default: human-readable
veritas essay.txt -o json     # JSON for scripting
veritas essay.txt -o minimal  # Single-line verdict

# Verbose mode with detailed statistics
veritas essay.txt --verbose
```

### CLI Options

| Option | Description |
|--------|-------------|
| `<file>` | Path to text file to analyze |
| `-t, --text "..."` | Analyze text provided directly |
| `--stdin` | Read text from standard input |
| `-o, --output <fmt>` | Output format: `text`, `json`, `minimal` |
| `-v, --verbose` | Show detailed category breakdowns |
| `--version` | Show version number |
| `-h, --help` | Show help message |

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Likely human-written (AI probability < 40%) |
| `1` | Likely AI-generated (AI probability ≥ 60%) |
| `2` | Mixed/uncertain (AI probability 40-60%) |

## Detection Categories

Veritas analyzes text across 45+ linguistic dimensions grouped into categories:

| Category | Features | Description |
|----------|----------|-------------|
| **Structure** | 9 | Paragraph uniformity, sentence patterns |
| **Vocabulary** | 6 | TTR, hapax ratio, word diversity |
| **Entropy** | 3 | N-gram predictability (GPTZero-style) |
| **Tone** | 4 | Formality, emotional intensity, consistency |
| **Hedging** | 3 | Uncertainty markers (AI signature) |
| **Personal Voice** | 5 | First-person usage, opinion markers |
| **Rhetorical** | 4 | Questions, emphasis, engagement |
| **Coherence** | 3 | Topic drift, lexical chains |
| **Punctuation** | 4 | Rhythm, diversity, parentheticals |
| **Style** | 4 | Passive voice, nominalizations |

### Top Discriminating Features

From our Helios model training:

1. **paragraph_uniformity** (39.3%) - AI writes uniform paragraphs
2. **trigram_entropy** (15.4%) - AI has lower 3-gram entropy
3. **bigram_entropy** (14.9%) - AI word pairs are predictable
4. **avg_sentence_length** (6.1%) - AI sentence length patterns
5. **sentence_start_diversity** (5.9%) - AI starts sentences similarly
| Readability Metrics | 15% | Flesch-Kincaid, complexity variance |
| Zipf's Law Analysis | 12% | Word frequency distribution conformance |

## Detection Philosophy

Veritas operates on a core principle: **both extremes are suspicious**.

Natural human writing falls within a "reasonable middle" range for most metrics:

| Metric | Too Low (AI) | Human Range | Too High (Humanizer) |
|--------|--------------|-------------|----------------------|
| Hapax Ratio | < 30% | 40-60% | > 75% |
| TTR | < 0.4 | 0.5-0.7 | > 0.85 |
| Sentence CV | < 0.25 | 0.4-0.8 | > 0.9 |
| Burstiness | < -0.1 | 0.1-0.4 | > 0.5 |

- **Too uniform/perfect** → Likely AI-generated (optimized for fluency)
- **Too chaotic/random** → Possibly AI with humanizer tools (artificial variability)

## Programmatic Usage (Node.js)

```javascript
const { VeritasAnalyzer } = require('veritas-ai-detector/cli/analyzer');

const analyzer = new VeritasAnalyzer();
const result = analyzer.analyze("Your text to analyze...");

console.log(result.aiProbability);     // 0.0 to 1.0
console.log(result.verdict.label);     // "Likely Human", "Likely AI", etc.
console.log(result.confidence);        // Confidence score
console.log(result.model.name);        // "Sunrise"
console.log(result.model.accuracy);    // 0.9808
```

### Result Object

```javascript
{
  aiProbability: 0.72,           // 0.0 (human) to 1.0 (AI)
  humanProbability: 0.28,
  confidence: 0.85,
  confidenceInterval: { lower: 0.65, upper: 0.79 },
  verdict: { label: "Likely AI", band: "high" },
  stats: { words: 250, sentences: 12, paragraphs: 4 },
  categoryResults: [...],        // Per-category analysis
  findings: [...],               // Specific observations
  model: {
    name: "Sunrise",
    version: "3.0.0",
    accuracy: 0.9808,
    f1Score: 0.9809,
    trainingSamples: 29976
  }
}
```

## Limitations

- Requires minimum ~50 words for reliable analysis
- Cannot detect all AI-written text with certainty
- May flag highly edited human text as suspicious
- Not designed for code or highly technical content
- Detection accuracy varies with text length and style

## What's New in v4.0

- **Helios Model** (99.24% accuracy): 45 features including tone detection
- **Zenith Model** (99.57% accuracy): GPTZero-style perplexity analysis
- **TriClass Detection**: 3-way classification (Human/AI/Humanized)
- **Humanization Detection**: 86.7% detection rate for bypass tools
- **Tone Analysis**: Formality, emotional intensity, sentiment variance
- **Hedging Patterns**: Detects AI's characteristic uncertainty markers
- **Personal Voice Detection**: First-person usage, opinion markers
- **Comprehensive Benchmarks**: Full evaluation suite included

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>◈ VERITAS</strong><br>
  <em>Variance-based Entity Recognition & Inference for Text Authenticity Scoring</em><br>
  Multi-Model Detection Suite v4.0
</p>
