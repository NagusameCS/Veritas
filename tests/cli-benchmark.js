#!/usr/bin/env node
/**
 * VERITAS CLI Benchmark Runner
 * Run benchmarks from command line using Node.js
 */

const fs = require('fs');
const path = require('path');

// Load the benchmark test samples
const benchmarkPath = path.join(__dirname, 'benchmark.js');
const benchmarkCode = fs.readFileSync(benchmarkPath, 'utf8');

// Extract testSamples from the benchmark file
const testSamplesMatch = benchmarkCode.match(/const testSamples = \{[\s\S]*?\n\};/);
if (!testSamplesMatch) {
    console.error('Could not parse test samples from benchmark.js');
    process.exit(1);
}

// Create a sandbox to eval the test samples
const vm = require('vm');
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(testSamplesMatch[0], sandbox);
const testSamples = sandbox.testSamples;

// Simple utility functions for CLI analysis
const Utils = {
    tokenize(text) {
        return text.toLowerCase()
            .replace(/[^\w\s'-]/g, ' ')
            .split(/\s+/)
            .filter(t => t.length > 0);
    },
    
    splitSentences(text) {
        return text
            .replace(/([.!?])\s+/g, '$1|')
            .split('|')
            .filter(s => s.trim().length > 5);
    },
    
    mean(arr) {
        return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    },
    
    variance(arr) {
        if (arr.length < 2) return 0;
        const m = this.mean(arr);
        return arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / arr.length;
    },
    
    standardDeviation(arr) {
        return Math.sqrt(this.variance(arr));
    }
};

// Simplified analyzer for CLI testing
function analyzeText(text) {
    const words = Utils.tokenize(text);
    const sentences = Utils.splitSentences(text);
    const lowerText = text.toLowerCase();
    
    let aiScore = 0;
    let signals = [];
    
    // 1. Check for AI vocabulary markers
    const aiWords = [
        'delve', 'multifaceted', 'tapestry', 'landscape', 'realm', 'pivotal',
        'myriad', 'plethora', 'intricate', 'nuanced', 'comprehensive', 'crucial',
        'furthermore', 'moreover', 'consequently', 'nevertheless', 'subsequently',
        'facilitate', 'leverage', 'utilize', 'implement', 'enhance', 'optimize'
    ];
    const aiWordCount = aiWords.filter(w => lowerText.includes(w)).length;
    if (aiWordCount >= 3) {
        aiScore += 0.15;
        signals.push(`AI vocabulary: ${aiWordCount} markers`);
    }
    
    // 2. Check for AI phrase patterns
    const aiPhrases = [
        'it is important to note', 'it should be noted', 'it is worth mentioning',
        'let\'s delve', 'here\'s a breakdown', 'in conclusion', 'to summarize',
        'firstly', 'secondly', 'thirdly', 'furthermore', 'in addition',
        'play a crucial role', 'key takeaways', 'it is essential'
    ];
    const phraseCount = aiPhrases.filter(p => lowerText.includes(p)).length;
    if (phraseCount >= 2) {
        aiScore += 0.2;
        signals.push(`AI phrases: ${phraseCount} patterns`);
    }
    
    // 3. Check sentence length variance (AI tends to be uniform)
    const sentenceLengths = sentences.map(s => Utils.tokenize(s).length);
    const lengthCV = sentenceLengths.length > 3 
        ? Utils.standardDeviation(sentenceLengths) / Utils.mean(sentenceLengths)
        : 0.5;
    if (lengthCV < 0.3) {
        aiScore += 0.15;
        signals.push(`Low sentence variance: CV=${lengthCV.toFixed(2)}`);
    } else if (lengthCV > 0.5) {
        aiScore -= 0.1;
        signals.push(`High sentence variance: CV=${lengthCV.toFixed(2)} (human-like)`);
    }
    
    // 4. Check for Gemini/Claude helper tone
    const helperPhrases = [
        'i\'d be happy to', 'great question', 'let me help', 'here to help',
        'feel free to', 'hope this helps', 'absolutely!', 'certainly!'
    ];
    const helperCount = helperPhrases.filter(p => lowerText.includes(p)).length;
    if (helperCount >= 2) {
        aiScore += 0.2;
        signals.push(`AI helper tone: ${helperCount} markers`);
    }
    
    // 5. Check for personal/emotional language (human indicator)
    const personalPatterns = /\b(i felt|i thought|i realized|honestly|literally|like,|lol|omg)\b/gi;
    const personalMatches = (text.match(personalPatterns) || []).length;
    if (personalMatches >= 2) {
        aiScore -= 0.2;
        signals.push(`Personal language: ${personalMatches} markers (human-like)`);
    }
    
    // 6. Check for structured format (headers, bullets)
    const hasHeaders = /^#+\s|\*\*[A-Z]/.test(text);
    const hasBullets = /^[\s]*[-•*]\s/m.test(text);
    const hasNumberedList = /^\d+\.\s/m.test(text);
    if (hasHeaders || hasBullets || hasNumberedList) {
        aiScore += 0.1;
        signals.push('Structured formatting detected');
    }
    
    // 7. Check for hedging (AI tends to over-hedge)
    const hedgingPhrases = [
        'it could be argued', 'one might suggest', 'it is possible that',
        'in some cases', 'generally speaking', 'it seems that'
    ];
    const hedgingCount = hedgingPhrases.filter(p => lowerText.includes(p)).length;
    if (hedgingCount >= 3) {
        aiScore += 0.15;
        signals.push(`Excessive hedging: ${hedgingCount} phrases`);
    }
    
    // Normalize score
    const finalScore = Math.max(0, Math.min(1, 0.5 + aiScore));
    
    return {
        aiProbability: finalScore,
        confidence: sentences.length > 10 ? 0.8 : 0.5,
        signals,
        wordCount: words.length,
        sentenceCount: sentences.length
    };
}

// Run benchmarks
console.log('='.repeat(60));
console.log('VERITAS CLI Benchmark Runner');
console.log('='.repeat(60));
console.log('');

const results = {
    human: { correct: 0, total: 0 },
    ai: { correct: 0, total: 0 },
    edgeCases: { correct: 0, total: 0 }
};

const detailedResults = [];

// Test human samples
console.log('Testing Human Samples...');
console.log('-'.repeat(40));
for (const sample of testSamples.human) {
    const result = analyzeText(sample.text);
    const isCorrect = result.aiProbability < 0.5;
    results.human.total++;
    if (isCorrect) results.human.correct++;
    
    const status = isCorrect ? '✓' : '✗ FP';
    console.log(`  ${status} ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI`);
    if (result.signals.length > 0 && !isCorrect) {
        console.log(`     Signals: ${result.signals.join(', ')}`);
    }
    
    detailedResults.push({ ...sample, result, isCorrect });
}
console.log('');

// Test AI samples
console.log('Testing AI Samples...');
console.log('-'.repeat(40));
for (const sample of testSamples.ai) {
    const result = analyzeText(sample.text);
    const isCorrect = result.aiProbability >= 0.5;
    results.ai.total++;
    if (isCorrect) results.ai.correct++;
    
    const status = isCorrect ? '✓' : '✗ FN';
    console.log(`  ${status} ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI`);
    if (result.signals.length > 0) {
        console.log(`     Signals: ${result.signals.slice(0, 3).join(', ')}`);
    }
    
    detailedResults.push({ ...sample, result, isCorrect });
}
console.log('');

// Test edge cases
console.log('Testing Edge Cases...');
console.log('-'.repeat(40));
for (const sample of testSamples.edgeCases) {
    const result = analyzeText(sample.text);
    const isCorrect = sample.label === 'human' 
        ? result.aiProbability < 0.5 
        : result.aiProbability >= 0.5;
    results.edgeCases.total++;
    if (isCorrect) results.edgeCases.correct++;
    
    const status = isCorrect ? '✓' : (sample.label === 'human' ? '✗ FP' : '✗ FN');
    console.log(`  ${status} ${sample.id} (${sample.label}): ${(result.aiProbability * 100).toFixed(1)}% AI`);
    console.log(`     ${sample.description}`);
    
    detailedResults.push({ ...sample, result, isCorrect });
}
console.log('');

// Summary
console.log('='.repeat(60));
console.log('SUMMARY');
console.log('='.repeat(60));

const humanAcc = (results.human.correct / results.human.total * 100).toFixed(1);
const aiAcc = (results.ai.correct / results.ai.total * 100).toFixed(1);
const edgeAcc = (results.edgeCases.correct / results.edgeCases.total * 100).toFixed(1);

const totalCorrect = results.human.correct + results.ai.correct + results.edgeCases.correct;
const totalSamples = results.human.total + results.ai.total + results.edgeCases.total;
const overallAcc = (totalCorrect / totalSamples * 100).toFixed(1);

console.log(`Human Samples:    ${results.human.correct}/${results.human.total} (${humanAcc}%)`);
console.log(`AI Samples:       ${results.ai.correct}/${results.ai.total} (${aiAcc}%)`);
console.log(`Edge Cases:       ${results.edgeCases.correct}/${results.edgeCases.total} (${edgeAcc}%)`);
console.log('-'.repeat(40));
console.log(`OVERALL:          ${totalCorrect}/${totalSamples} (${overallAcc}%)`);
console.log('');

// Recommendations
console.log('='.repeat(60));
console.log('ML TRAINING ASSESSMENT');
console.log('='.repeat(60));

const falsePositiveRate = ((results.human.total - results.human.correct) / results.human.total * 100);
const falseNegativeRate = ((results.ai.total - results.ai.correct) / results.ai.total * 100);

console.log(`False Positive Rate: ${falsePositiveRate.toFixed(1)}%`);
console.log(`False Negative Rate: ${falseNegativeRate.toFixed(1)}%`);
console.log('');

if (parseFloat(overallAcc) >= 85) {
    console.log('✅ RECOMMENDATION: Current heuristic approach is performing well.');
    console.log('   ML training may provide marginal improvements but is not urgent.');
} else if (parseFloat(overallAcc) >= 70) {
    console.log('⚠️  RECOMMENDATION: Consider ML training for calibration.');
    console.log('   Focus areas:');
    if (falsePositiveRate > 20) console.log('   - Reduce false positives on formal/instructional content');
    if (falseNegativeRate > 20) console.log('   - Improve detection of newer AI models (Gemini, Claude)');
} else {
    console.log('❌ RECOMMENDATION: ML training strongly recommended.');
    console.log('   Current heuristics need significant improvement.');
    console.log('   Consider using these HuggingFace datasets:');
    console.log('   - artem9k/ai-text-detection-pile (658 downloads, large dataset)');
    console.log('   - srikanthgali/ai-text-detection-pile-cleaned (cleaned version)');
    console.log('   - DeepNLP/ChatGPT-Gemini-Claude-Perplexity-Human-Evaluation dataset');
}

console.log('');
