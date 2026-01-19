#!/usr/bin/env node

/**
 * VERITAS CLI — AI Text Detection
 * Command-line interface for the Veritas AI detection engine
 * 
 * Usage:
 *   veritas <file>           Analyze a text file
 *   veritas -t "text"        Analyze text directly
 *   veritas --stdin          Read from stdin
 *   veritas --help           Show help
 */

const fs = require('fs');
const path = require('path');

// Import the analysis engine
const { VeritasAnalyzer } = require('./analyzer');

// CLI colors (ANSI escape codes)
const colors = {
    reset: '\x1b[0m',
    bold: '\x1b[1m',
    dim: '\x1b[2m',
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    blue: '\x1b[34m',
    magenta: '\x1b[35m',
    cyan: '\x1b[36m',
    white: '\x1b[37m',
    gray: '\x1b[90m'
};

// Version from package.json
const packageJson = require('../package.json');
const VERSION = packageJson.version;

/**
 * Print styled text
 */
function print(text, color = 'reset') {
    console.log(`${colors[color]}${text}${colors.reset}`);
}

/**
 * Print the banner
 */
function printBanner() {
    console.log(`
${colors.bold}${colors.cyan}◈ VERITAS${colors.reset} ${colors.dim}v${VERSION}${colors.reset}
${colors.gray}AI Text Detection Engine${colors.reset}
`);
}

/**
 * Print help message
 */
function printHelp() {
    printBanner();
    console.log(`${colors.bold}USAGE${colors.reset}
  veritas <file>              Analyze a text file
  veritas -t, --text "..."    Analyze text directly
  veritas --stdin             Read text from stdin
  veritas -o, --output <fmt>  Output format: text, json, minimal
  veritas -v, --verbose       Show detailed analysis
  veritas --version           Show version
  veritas -h, --help          Show this help

${colors.bold}EXAMPLES${colors.reset}
  ${colors.dim}# Analyze a file${colors.reset}
  veritas essay.txt

  ${colors.dim}# Analyze inline text${colors.reset}
  veritas -t "The quick brown fox jumps over the lazy dog."

  ${colors.dim}# Pipe from another command${colors.reset}
  cat document.txt | veritas --stdin

  ${colors.dim}# Output as JSON${colors.reset}
  veritas essay.txt -o json

  ${colors.dim}# Verbose analysis with all statistics${colors.reset}
  veritas essay.txt --verbose

${colors.bold}OUTPUT${colors.reset}
  The analysis provides:
  • AI Probability (0-100%)
  • Human Likelihood score
  • Confidence level
  • Key findings and indicators
  • Detailed category breakdown (with --verbose)

${colors.bold}DETECTION PHILOSOPHY${colors.reset}
  Veritas uses variance-based detection. Both extremes are suspicious:
  • ${colors.red}Too uniform/perfect${colors.reset} → Likely AI-generated
  • ${colors.yellow}Too chaotic/extreme${colors.reset} → Possibly AI with humanizer tools

  Natural human writing falls within a "reasonable middle" range.
`);
}

/**
 * Print version
 */
function printVersion() {
    console.log(`veritas ${VERSION}`);
}

/**
 * Format a percentage with color
 */
function formatPercent(value, thresholds = { low: 30, high: 60 }) {
    const pct = Math.round(value * 100);
    let color = 'green';
    if (pct >= thresholds.high) color = 'red';
    else if (pct >= thresholds.low) color = 'yellow';
    return `${colors[color]}${pct}%${colors.reset}`;
}

/**
 * Format the verdict
 */
function formatVerdict(verdict) {
    const verdictColors = {
        'Human': 'green',
        'Probably Human': 'green',
        'Mixed': 'yellow',
        'Probably AI': 'red',
        'AI': 'red'
    };
    const color = verdictColors[verdict.label] || 'white';
    return `${colors[color]}${colors.bold}${verdict.label}${colors.reset}`;
}

/**
 * Print analysis results in text format
 */
function printTextResults(result, verbose = false) {
    printBanner();
    
    // Main score
    console.log(`${colors.bold}DETECTION RESULT${colors.reset}`);
    console.log(`${'─'.repeat(50)}`);
    
    const aiProb = result.aiProbability;
    const humanProb = 1 - aiProb;
    
    console.log(`  Verdict:         ${formatVerdict(result.verdict)}`);
    console.log(`  AI Probability:  ${formatPercent(aiProb, { low: 0.3, high: 0.6 })}`);
    console.log(`  Human Likelihood: ${formatPercent(humanProb, { low: 0.4, high: 0.7 })}`);
    console.log(`  Confidence:      ${formatPercent(result.confidence)}`);
    console.log();
    
    // Text stats
    console.log(`${colors.bold}TEXT STATISTICS${colors.reset}`);
    console.log(`${'─'.repeat(50)}`);
    console.log(`  Words:      ${result.stats.words}`);
    console.log(`  Sentences:  ${result.stats.sentences}`);
    console.log(`  Paragraphs: ${result.stats.paragraphs}`);
    console.log();
    
    // Key findings
    if (result.findings && result.findings.length > 0) {
        console.log(`${colors.bold}KEY FINDINGS${colors.reset}`);
        console.log(`${'─'.repeat(50)}`);
        
        const topFindings = result.findings.slice(0, 8);
        for (const finding of topFindings) {
            const icon = finding.indicator === 'ai' ? `${colors.red}▸${colors.reset}` :
                        finding.indicator === 'human' ? `${colors.green}▸${colors.reset}` :
                        `${colors.yellow}▸${colors.reset}`;
            console.log(`  ${icon} ${finding.text}`);
        }
        console.log();
    }
    
    // Verbose: Category breakdown
    if (verbose && result.categoryResults) {
        console.log(`${colors.bold}CATEGORY ANALYSIS${colors.reset}`);
        console.log(`${'─'.repeat(50)}`);
        
        for (const cat of result.categoryResults) {
            const prob = Math.round(cat.aiProbability * 100);
            const bar = createBar(prob, 20);
            const probColor = prob >= 60 ? 'red' : prob >= 40 ? 'yellow' : 'green';
            console.log(`  ${cat.name.padEnd(35)} ${colors[probColor]}${prob.toString().padStart(3)}%${colors.reset} ${bar}`);
        }
        console.log();
        
        // Advanced stats
        if (result.advancedStats) {
            console.log(`${colors.bold}ADVANCED STATISTICS${colors.reset}`);
            console.log(`${'─'.repeat(50)}`);
            
            const stats = result.advancedStats;
            
            if (stats.sentences) {
                console.log(`  ${colors.dim}Sentence Length:${colors.reset}`);
                console.log(`    Mean: ${stats.sentences.mean?.toFixed(1)} words`);
                console.log(`    CV:   ${stats.sentences.cv?.toFixed(3)}`);
            }
            
            if (stats.vocabulary) {
                console.log(`  ${colors.dim}Vocabulary:${colors.reset}`);
                console.log(`    TTR:         ${stats.vocabulary.ttr?.toFixed(3)}`);
                console.log(`    Hapax Ratio: ${stats.vocabulary.hapaxRatio?.toFixed(3)}`);
            }
            
            if (stats.zipf) {
                console.log(`  ${colors.dim}Zipf's Law:${colors.reset}`);
                console.log(`    Slope:      ${stats.zipf.slope?.toFixed(3)}`);
                console.log(`    Compliance: ${(stats.zipf.compliance * 100)?.toFixed(1)}%`);
            }
            
            if (stats.burstiness) {
                console.log(`  ${colors.dim}Burstiness:${colors.reset}`);
                console.log(`    Score: ${stats.burstiness.sentenceLength?.toFixed(3)}`);
            }
            
            if (stats.humanizerSignals) {
                const signals = stats.humanizerSignals;
                if (signals.isLikelyHumanized) {
                    console.log();
                    console.log(`  ${colors.yellow}⚠ Humanizer Detection Warning${colors.reset}`);
                    console.log(`    This text may have been AI-generated and then`);
                    console.log(`    processed with humanizer tools to evade detection.`);
                }
            }
            
            console.log();
        }
    }
    
    // Confidence interval
    if (result.confidenceInterval) {
        const ci = result.confidenceInterval;
        console.log(`${colors.dim}95% CI: ${Math.round(ci.lower * 100)}% - ${Math.round(ci.upper * 100)}%${colors.reset}`);
    }
    
    // Disclaimer
    console.log(`${colors.dim}Note: No AI detection is 100% accurate. Use as one factor among many.${colors.reset}`);
}

/**
 * Create a progress bar
 */
function createBar(percent, width = 20) {
    const filled = Math.round((percent / 100) * width);
    const empty = width - filled;
    const filledChar = '█';
    const emptyChar = '░';
    return `${colors.gray}[${filledChar.repeat(filled)}${emptyChar.repeat(empty)}]${colors.reset}`;
}

/**
 * Print analysis results as JSON
 */
function printJsonResults(result) {
    console.log(JSON.stringify(result, null, 2));
}

/**
 * Print minimal results (just verdict and probability)
 */
function printMinimalResults(result) {
    const aiProb = Math.round(result.aiProbability * 100);
    console.log(`${result.verdict.label}: ${aiProb}% AI probability`);
}

/**
 * Read text from file
 */
function readFile(filePath) {
    const absolutePath = path.resolve(filePath);
    
    if (!fs.existsSync(absolutePath)) {
        console.error(`${colors.red}Error: File not found: ${filePath}${colors.reset}`);
        process.exit(1);
    }
    
    try {
        return fs.readFileSync(absolutePath, 'utf-8');
    } catch (err) {
        console.error(`${colors.red}Error reading file: ${err.message}${colors.reset}`);
        process.exit(1);
    }
}

/**
 * Read text from stdin
 */
async function readStdin() {
    return new Promise((resolve, reject) => {
        let data = '';
        
        process.stdin.setEncoding('utf-8');
        
        process.stdin.on('readable', () => {
            let chunk;
            while ((chunk = process.stdin.read()) !== null) {
                data += chunk;
            }
        });
        
        process.stdin.on('end', () => {
            resolve(data);
        });
        
        process.stdin.on('error', reject);
        
        // Timeout for stdin
        setTimeout(() => {
            if (data.length === 0) {
                reject(new Error('No input received from stdin'));
            }
        }, 100);
    });
}

/**
 * Main CLI entry point
 */
async function main() {
    const args = process.argv.slice(2);
    
    // Parse arguments
    let text = null;
    let outputFormat = 'text';
    let verbose = false;
    
    for (let i = 0; i < args.length; i++) {
        const arg = args[i];
        
        if (arg === '-h' || arg === '--help') {
            printHelp();
            process.exit(0);
        }
        
        if (arg === '--version') {
            printVersion();
            process.exit(0);
        }
        
        if (arg === '-v' || arg === '--verbose') {
            verbose = true;
            continue;
        }
        
        if (arg === '-o' || arg === '--output') {
            outputFormat = args[++i] || 'text';
            continue;
        }
        
        if (arg === '-t' || arg === '--text') {
            text = args[++i];
            if (!text) {
                console.error(`${colors.red}Error: --text requires an argument${colors.reset}`);
                process.exit(1);
            }
            continue;
        }
        
        if (arg === '--stdin') {
            try {
                text = await readStdin();
            } catch (err) {
                console.error(`${colors.red}Error reading stdin: ${err.message}${colors.reset}`);
                process.exit(1);
            }
            continue;
        }
        
        // Assume it's a file path
        if (!arg.startsWith('-')) {
            text = readFile(arg);
        }
    }
    
    // Check if we have text to analyze
    if (!text || text.trim().length === 0) {
        // Check if stdin has data (non-interactive)
        if (!process.stdin.isTTY) {
            try {
                text = await readStdin();
            } catch (err) {
                printHelp();
                process.exit(0);
            }
        } else {
            printHelp();
            process.exit(0);
        }
    }
    
    // Validate text length
    const words = text.trim().split(/\s+/).length;
    if (words < 10) {
        console.error(`${colors.yellow}Warning: Text too short (${words} words). Need at least 10 words for reliable analysis.${colors.reset}`);
    }
    
    // Run analysis
    try {
        const analyzer = new VeritasAnalyzer();
        const result = analyzer.analyze(text);
        
        // Output results
        switch (outputFormat) {
            case 'json':
                printJsonResults(result);
                break;
            case 'minimal':
                printMinimalResults(result);
                break;
            default:
                printTextResults(result, verbose);
        }
        
        // Exit with code based on result
        // 0 = likely human, 1 = likely AI, 2 = mixed
        const aiProb = result.aiProbability;
        if (aiProb >= 0.6) process.exit(1);
        if (aiProb >= 0.4) process.exit(2);
        process.exit(0);
        
    } catch (err) {
        console.error(`${colors.red}Analysis error: ${err.message}${colors.reset}`);
        if (verbose) {
            console.error(err.stack);
        }
        process.exit(1);
    }
}

// Run CLI
main().catch(err => {
    console.error(`${colors.red}Fatal error: ${err.message}${colors.reset}`);
    process.exit(1);
});
