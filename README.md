# Veritas

**Variance-Based AI Text Detection Engine**

[![npm version](https://img.shields.io/npm/v/veritas-ai-detector.svg)](https://www.npmjs.com/package/veritas-ai-detector)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Veritas uses statistical variance analysis to detect AI-generated text. Unlike pattern-matching approaches, it analyzes the natural variability in human writing—sentence lengths, vocabulary diversity, word frequency distributions, and more.

## Key Features

- **Bidirectional Detection**: Flags both AI uniformity (too perfect) AND humanizer chaos (too random)
- **Statistical Foundation**: Based on Zipf's Law, TTR, Hapax Ratio, Burstiness coefficients
- **Confidence Intervals**: Provides 95% CI with honest uncertainty estimates
- **No External APIs**: Runs entirely client-side (web) or locally (CLI)
- **Multiple Interfaces**: Web UI + Command Line Tool

## Installation

### Web Interface
Simply open [the web interface](https://github.com/agarwalnitika/Veritas) in your browser—no installation required.

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

## Statistical Methods

### Sentence Length Variance
Uses Coefficient of Variation (CV) to measure variability in sentence lengths.

### Zipf's Law Analysis
Compares word frequency distribution to expected Zipfian slopes.

### Vocabulary Metrics
- **Type-Token Ratio (TTR)**: Unique words / total words
- **Hapax Legomena Ratio**: Words appearing exactly once

### Burstiness Coefficient
Measures temporal clustering patterns in word usage:
- B = -1: Perfectly periodic (AI-like)
- B = 0: Poisson random
- B = +1: Maximally bursty (human-like)

## Programmatic Usage (Node.js)

```javascript
const { VeritasAnalyzer } = require('veritas-ai-detector/cli/analyzer');

const analyzer = new VeritasAnalyzer();
const result = analyzer.analyze("Your text to analyze...");

console.log(result.aiProbability);    // 0.0 to 1.0
console.log(result.verdict.label);    // "Likely Human", "Likely AI", etc.
console.log(result.confidence);       // Confidence score
```

## Limitations

- Requires minimum ~50 words for reliable analysis
- Cannot detect all AI-written text with certainty
- May flag highly edited human text as suspicious
- Not designed for code or highly technical content

## License

MIT License - see [LICENSE](LICENSE) for details.