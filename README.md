# ◈ Veritas

**AI Text Detection Engine — Powered by Sunrise ML Model**

[![npm version](https://img.shields.io/npm/v/veritas-ai-detector.svg)](https://www.npmjs.com/package/veritas-ai-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Veritas uses ML-trained statistical variance analysis to detect AI-generated text with **98.08% accuracy**. Unlike pattern-matching approaches, it analyzes the natural variability in human writing—sentence lengths, vocabulary diversity, word frequency distributions, and more.

## 🌅 Sunrise Model v3.0

Veritas is powered by the **Sunrise ML Model**, trained on 29,976 samples from diverse datasets:

| Metric | Score |
|--------|-------|
| **Accuracy** | 98.08% |
| **F1 Score** | 98.09% |
| **ROC AUC** | 99.80% |
| **Training Samples** | 29,976 |

The model uses ML-derived feature weights to optimize detection across 14 linguistic analysis categories.

## Key Features

- **ML-Powered Detection**: Sunrise model trained on diverse human/AI text samples
- **Bidirectional Analysis**: Flags both AI uniformity (too perfect) AND humanizer chaos (too random)
- **Humanized AI Detection**: Specifically identifies "Humanized AI" text that's been modified to evade detection
- **Statistical Foundation**: Based on Zipf's Law, TTR, Hapax Ratio, Burstiness coefficients
- **Confidence Intervals**: Provides 95% CI with honest uncertainty estimates
- **No External APIs**: Runs entirely client-side (web) or locally (CLI)
- **Multiple Interfaces**: Web UI + Command Line Tool

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

Veritas analyzes text across 14 linguistic dimensions:

| Category | Weight | Description |
|----------|--------|-------------|
| Sentence Structure | 22% | Variance in sentence lengths and patterns |
| Vocabulary Diversity | 18% | TTR, hapax ratio, word choice patterns |
| Burstiness Patterns | 18% | Temporal clustering of similar structures |
| Repetition Analysis | 15% | N-gram repetition distribution |
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

## What's New in v3.0

- 🌅 **Sunrise ML Model**: Trained on 29,976 samples with 98.08% accuracy
- 🔍 **Humanized AI Detection**: Specifically identifies AI text modified by humanizer tools
- 📊 **Enhanced Trend Analysis**: Shows observed linguistic patterns instead of per-sentence highlighting
- 📄 **Improved Reports**: Open full report in new tab for easy PDF/print export
- ⚡ **Performance**: Faster analysis with optimized feature weights

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>◈ VERITAS</strong><br>
  <em>Variance-based Entity Recognition & Inference for Text Authenticity Scoring</em><br>
  Powered by Sunrise Model v3.0
</p>
