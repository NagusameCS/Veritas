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
    console.log(`${'─'.repeat(60)}`);
    
    const aiProb = result.aiProbability;
    const humanProb = 1 - aiProb;
    
    console.log(`  Verdict:          ${formatVerdict(result.verdict)}`);
    console.log(`  AI Probability:   ${formatPercent(aiProb, { low: 0.3, high: 0.6 })}`);
    console.log(`  Human Likelihood: ${formatPercent(humanProb, { low: 0.4, high: 0.7 })}`);
    console.log(`  Confidence:       ${formatPercent(result.confidence)}`);
    
    if (result.confidenceInterval) {
        const ci = result.confidenceInterval;
        console.log(`  95% CI:           ${Math.round(ci.lower * 100)}% - ${Math.round(ci.upper * 100)}%`);
    }
    console.log();
    
    // Document Overview
    console.log(`${colors.bold}DOCUMENT OVERVIEW${colors.reset}`);
    console.log(`${'─'.repeat(60)}`);
    console.log(`  Characters:       ${result.stats.characters?.toLocaleString() || 0}`);
    console.log(`  Words:            ${result.stats.words?.toLocaleString() || 0}`);
    console.log(`  Sentences:        ${result.stats.sentences?.toLocaleString() || 0}`);
    console.log(`  Paragraphs:       ${result.stats.paragraphs?.toLocaleString() || 0}`);
    console.log(`  Avg Words/Sent:   ${result.stats.avgWordsPerSentence || 'N/A'}`);
    console.log(`  Analysis Time:    ${result.analysisTime || 'N/A'}`);
    console.log();
    
    // Key findings
    if (result.findings && result.findings.length > 0) {
        console.log(`${colors.bold}KEY FINDINGS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        
        const aiFindings = result.findings.filter(f => f.indicator === 'ai');
        const humanFindings = result.findings.filter(f => f.indicator === 'human');
        
        console.log(`  ${colors.red}[AI]${colors.reset} ${aiFindings.length} indicators  |  ${colors.green}[Human]${colors.reset} ${humanFindings.length} indicators`);
        console.log();
        
        const topFindings = result.findings.slice(0, 10);
        for (const finding of topFindings) {
            const icon = finding.indicator === 'ai' ? `${colors.red}[AI]${colors.reset}` :
                        finding.indicator === 'human' ? `${colors.green}[HU]${colors.reset}` :
                        `${colors.yellow}[--]${colors.reset}`;
            const label = finding.label || '';
            const value = finding.value || finding.text || '';
            console.log(`  ${icon} ${colors.bold}${label}${colors.reset}`);
            console.log(`      ${value}`);
            if (finding.stats && finding.stats.measured) {
                console.log(`      ${colors.dim}Measured: ${finding.stats.measured}${colors.reset}`);
            }
        }
        console.log();
    }
    
    // Verbose: Full statistics
    if (verbose) {
        printVerboseStatistics(result);
    }
    
    // Disclaimer
    console.log(`${colors.dim}Note: No AI detection is 100% accurate. Use as one factor among many.${colors.reset}`);
}

/**
 * Print comprehensive verbose statistics
 */
function printVerboseStatistics(result) {
    // Category breakdown
    if (result.categoryResults) {
        console.log(`${colors.bold}CATEGORY ANALYSIS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        
        for (const cat of result.categoryResults) {
            const prob = Math.round(cat.aiProbability * 100);
            const conf = Math.round((cat.confidence || 0) * 100);
            const bar = createBar(prob, 20);
            const probColor = prob >= 60 ? 'red' : prob >= 40 ? 'yellow' : 'green';
            console.log(`  ${cat.name.padEnd(35)} ${colors[probColor]}${prob.toString().padStart(3)}%${colors.reset} ${bar} (${conf}% conf)`);
        }
        console.log();
    }
    
    // Advanced stats
    if (result.advancedStats) {
        const stats = result.advancedStats;
        
        // Vocabulary Metrics
        console.log(`${colors.bold}VOCABULARY METRICS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.vocabulary) {
            console.log(`  Unique Words:       ${stats.vocabulary.uniqueWords?.toLocaleString() || 'N/A'}`);
            console.log(`  Type-Token Ratio:   ${formatStatWithIndicator(stats.vocabulary.typeTokenRatio, 0.3, 0.5, true)}`);
            console.log(`  Root TTR:           ${stats.vocabulary.rootTTR?.toFixed(3) || 'N/A'}`);
            console.log(`  Log TTR:            ${stats.vocabulary.logTTR?.toFixed(4) || 'N/A'}`);
            console.log(`  MSTTR:              ${stats.vocabulary.msttr?.toFixed(3) || 'N/A'}`);
            console.log(`  Hapax Legomena:     ${formatStatWithIndicator(stats.vocabulary.hapaxLegomenaRatio, 0.35, 0.5, true)}`);
            console.log(`  Dis Legomena:       ${(stats.vocabulary.disLegomenaRatio * 100)?.toFixed(2) || 'N/A'}%`);
            console.log(`  Yule's K:           ${formatStatWithIndicator(stats.vocabulary.yulesK, 150, 100, false)}`);
            console.log(`  Simpson's D:        ${stats.vocabulary.simpsonsD?.toFixed(5) || 'N/A'}`);
            console.log(`  Honore's R:         ${stats.vocabulary.honoresR?.toFixed(1) || 'N/A'}`);
            console.log(`  Brunet's W:         ${stats.vocabulary.brunetsW?.toFixed(2) || 'N/A'}`);
            console.log(`  Sichel's S:         ${stats.vocabulary.sichelsS?.toFixed(4) || 'N/A'}`);
        }
        console.log();
        
        // Sentence Statistics
        console.log(`${colors.bold}SENTENCE STATISTICS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.sentences) {
            console.log(`  Mean Length:        ${stats.sentences.mean?.toFixed(2) || 'N/A'} words`);
            console.log(`  Median Length:      ${stats.sentences.median?.toFixed(2) || 'N/A'} words`);
            console.log(`  Std Deviation:      ${stats.sentences.stdDev?.toFixed(2) || 'N/A'}`);
            console.log(`  Min / Max:          ${stats.sentences.min || 'N/A'} / ${stats.sentences.max || 'N/A'} words`);
            console.log(`  Coeff. of Var:      ${formatStatWithIndicator(stats.sentences.coefficientOfVariation, 0.35, 0.5, true)}`);
            console.log(`  Skewness:           ${stats.sentences.skewness?.toFixed(3) || 'N/A'}`);
            console.log(`  Kurtosis:           ${stats.sentences.kurtosis?.toFixed(3) || 'N/A'}`);
            console.log(`  Gini Coefficient:   ${formatStatWithIndicator(stats.sentences.gini, 0.15, 0.25, true)}`);
        }
        console.log();
        
        // Zipf's Law
        console.log(`${colors.bold}ZIPF'S LAW ANALYSIS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.zipf) {
            console.log(`  Compliance:         ${formatStatWithIndicator(stats.zipf.compliance, 0.7, 0.85, true)}`);
            console.log(`  Log-Log Slope:      ${stats.zipf.slope?.toFixed(4) || 'N/A'} (ideal: -1.0)`);
            console.log(`  R-Squared:          ${stats.zipf.rSquared?.toFixed(4) || 'N/A'}`);
            console.log(`  Deviation:          ${stats.zipf.deviation?.toFixed(4) || 'N/A'}`);
        }
        console.log();
        
        // Readability
        console.log(`${colors.bold}READABILITY METRICS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.readability) {
            console.log(`  Avg Syllables/Word: ${stats.readability.avgSyllablesPerWord?.toFixed(2) || 'N/A'}`);
            console.log(`  Flesch Reading:     ${stats.readability.fleschReadingEase?.toFixed(1) || 'N/A'}`);
            console.log(`  Flesch-Kincaid:     ${stats.readability.fleschKincaidGrade?.toFixed(1) || 'N/A'} grade`);
            console.log(`  Gunning Fog:        ${stats.readability.gunningFogIndex?.toFixed(1) || 'N/A'}`);
            console.log(`  Coleman-Liau:       ${stats.readability.colemanLiauIndex?.toFixed(1) || 'N/A'}`);
            console.log(`  SMOG Index:         ${stats.readability.smogIndex?.toFixed(1) || 'N/A'}`);
            console.log(`  ARI:                ${stats.readability.ariIndex?.toFixed(1) || 'N/A'}`);
            console.log(`  Complex Words:      ${stats.readability.complexWordPercentage?.toFixed(1) || 'N/A'}%`);
            console.log(`  Polysyllables:      ${stats.readability.polysyllablePercentage?.toFixed(1) || 'N/A'}%`);
        }
        console.log();
        
        // Burstiness & Uniformity
        console.log(`${colors.bold}BURSTINESS & UNIFORMITY${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.burstiness) {
            console.log(`  Sentence Burst:     ${formatStatWithIndicator(stats.burstiness.sentenceLength, 0.1, 0.25, true)}`);
            console.log(`  Word Length Burst:  ${stats.burstiness.wordLength?.toFixed(4) || 'N/A'}`);
            console.log(`  Overall Uniformity: ${formatStatWithIndicator(stats.burstiness.overallUniformity, 0.7, 0.5, false)}`);
        }
        console.log();
        
        // N-gram Analysis
        console.log(`${colors.bold}N-GRAM ANALYSIS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.ngrams) {
            console.log(`  Unique Bigrams:     ${stats.ngrams.uniqueBigrams?.toLocaleString() || 'N/A'}`);
            console.log(`  Unique Trigrams:    ${stats.ngrams.uniqueTrigrams?.toLocaleString() || 'N/A'}`);
            console.log(`  Unique Quadgrams:   ${stats.ngrams.uniqueQuadgrams?.toLocaleString() || 'N/A'}`);
            console.log(`  Bigram Rep Rate:    ${formatStatWithIndicator(stats.ngrams.bigramRepetitionRate, 0.4, 0.25, false)}`);
            console.log(`  Trigram Rep Rate:   ${formatStatWithIndicator(stats.ngrams.trigramRepetitionRate, 0.2, 0.1, false)}`);
            console.log(`  Quadgram Rep Rate:  ${formatStatWithIndicator(stats.ngrams.quadgramRepetitionRate, 0.1, 0.05, false)}`);
            console.log(`  Repeated Phrases:   ${stats.ngrams.repeatedPhraseCount || 0}`);
            if (stats.ngrams.repeatedPhrases && stats.ngrams.repeatedPhrases.length > 0) {
                console.log(`  ${colors.dim}Top phrases:${colors.reset}`);
                for (const p of stats.ngrams.repeatedPhrases.slice(0, 3)) {
                    console.log(`    "${p.phrase}" (${p.count}x)`);
                }
            }
        }
        console.log();
        
        // Advanced Statistical Tests
        console.log(`${colors.bold}STATISTICAL TESTS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.autocorrelation) {
            console.log(`  Periodicity Score:  ${formatStatWithIndicator(stats.autocorrelation.periodicityScore, 0.6, 0.3, false)}`);
        }
        if (stats.perplexity) {
            console.log(`  Predictability:     ${formatStatWithIndicator(stats.perplexity.predictability, 0.6, 0.4, false)}`);
            console.log(`  Perplexity:         ${stats.perplexity.perplexity?.toFixed(2) || 'N/A'}`);
        }
        if (stats.runsTest) {
            console.log(`  Randomness Score:   ${formatStatWithIndicator(stats.runsTest.randomnessScore, 0.4, 0.6, true)}`);
        }
        if (stats.chiSquared) {
            console.log(`  Chi-Sq Uniformity:  ${formatStatWithIndicator(stats.chiSquared.uniformityScore, 0.7, 0.4, false)}`);
        }
        if (typeof stats.varianceStability !== 'undefined') {
            console.log(`  Variance Stability: ${formatStatWithIndicator(stats.varianceStability, 0.7, 0.5, false)}`);
        }
        if (typeof stats.mahalanobisDistance !== 'undefined') {
            console.log(`  Mahalanobis Dist:   ${stats.mahalanobisDistance?.toFixed(3) || 'N/A'}σ`);
        }
        console.log();
        
        // Human Likelihood
        console.log(`${colors.bold}HUMAN LIKELIHOOD ANALYSIS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (typeof stats.overallHumanLikelihood !== 'undefined') {
            console.log(`  Overall Likelihood: ${formatStatWithIndicator(stats.overallHumanLikelihood, 0.4, 0.6, true)}`);
        }
        if (stats.humanLikelihood) {
            console.log(`  Sentence Length CV: ${formatStatWithIndicator(stats.humanLikelihood.sentenceLengthCV, 0.4, 0.7, true)}`);
            console.log(`  Unique Word Dist:   ${formatStatWithIndicator(stats.humanLikelihood.hapaxRatio, 0.4, 0.7, true)}`);
            console.log(`  Word Burstiness:    ${formatStatWithIndicator(stats.humanLikelihood.burstiness, 0.4, 0.7, true)}`);
            console.log(`  Zipf Compliance:    ${formatStatWithIndicator(stats.humanLikelihood.zipfSlope, 0.4, 0.7, true)}`);
            console.log(`  Vocabulary Rich:    ${formatStatWithIndicator(stats.humanLikelihood.ttr, 0.4, 0.7, true)}`);
        }
        if (typeof stats.varianceNaturalness !== 'undefined') {
            console.log(`  Variance Natural:   ${formatStatWithIndicator(stats.varianceNaturalness, 0.4, 0.7, true)}`);
        }
        console.log();
        
        // AI Signature Metrics
        console.log(`${colors.bold}AI SIGNATURE METRICS${colors.reset}`);
        console.log(`${'─'.repeat(60)}`);
        if (stats.aiSignatures) {
            console.log(`  Hedging Density:    ${formatStatWithIndicator(stats.aiSignatures.hedgingDensity, 0.02, 0.01, false)}`);
            console.log(`  Discourse Markers:  ${stats.aiSignatures.discourseMarkerDensity?.toFixed(3) || 'N/A'}/sentence`);
            console.log(`  Unicode Anomalies:  ${stats.aiSignatures.unicodeAnomalyDensity?.toFixed(3) || 'N/A'}/1000 chars`);
            console.log(`  Decorative Divs:    ${stats.aiSignatures.decorativeDividerCount || 0}`);
            console.log(`  Contraction Rate:   ${formatStatWithIndicator(stats.aiSignatures.contractionRate, 0.3, 0.5, true)}`);
            console.log(`  Starter Variety:    ${formatStatWithIndicator(stats.aiSignatures.sentenceStarterVariety, 0.4, 0.6, true)}`);
            console.log(`  Passive Voice:      ${stats.aiSignatures.passiveVoiceRate?.toFixed(3) || 'N/A'}/sentence`);
        }
        if (stats.wordPatterns) {
            console.log(`  First-Person:       ${formatStatWithIndicator(stats.wordPatterns.firstPersonRatio, 0.01, 0.03, true)}`);
            console.log(`  Hedging Words:      ${(stats.wordPatterns.hedgingRatio * 100)?.toFixed(2) || 'N/A'}%`);
            console.log(`  Starter Diversity:  ${formatStatWithIndicator(stats.wordPatterns.starterDiversity, 0.4, 0.7, true)}`);
            console.log(`  AI Starter Ratio:   ${formatStatWithIndicator(stats.wordPatterns.aiStarterRatio, 0.5, 0.3, false)}`);
        }
        console.log();
        
        // Humanizer Detection
        if (result.humanizerSignals) {
            const signals = result.humanizerSignals;
            console.log(`${colors.bold}HUMANIZER DETECTION${colors.reset}`);
            console.log(`${'─'.repeat(60)}`);
            console.log(`  Flag Count:         ${signals.flagCount || 0}/5`);
            console.log(`  Likely Humanized:   ${signals.isLikelyHumanized ? `${colors.yellow}Yes${colors.reset}` : `${colors.green}No${colors.reset}`}`);
            if (signals.isLikelyHumanized) {
                console.log();
                console.log(`  ${colors.yellow}[!] Warning: This text may be AI-generated but processed${colors.reset}`);
                console.log(`  ${colors.yellow}    with humanizer tools to evade detection.${colors.reset}`);
            }
            console.log();
        }
    }
}

/**
 * Format a statistic with color indicator
 */
function formatStatWithIndicator(value, aiThreshold, humanThreshold, invertThresholds = false) {
    if (typeof value !== 'number' || isNaN(value)) return `${colors.dim}N/A${colors.reset}`;
    
    const formatted = (value * 100).toFixed(2) + '%';
    let color = 'white';
    
    if (invertThresholds) {
        if (value < aiThreshold) color = 'red';
        else if (value > humanThreshold) color = 'green';
        else color = 'yellow';
    } else {
        if (value > aiThreshold) color = 'red';
        else if (value < humanThreshold) color = 'green';
        else color = 'yellow';
    }
    
    return `${colors[color]}${formatted}${colors.reset}`;
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
