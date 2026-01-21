#!/usr/bin/env node
/**
 * VERITAS — Authentic Sample Benchmark
 * Tests detection accuracy on real human and AI samples from public datasets.
 */

const fs = require('fs');
const path = require('path');

// Load authentic samples
const samplesPath = path.join(__dirname, 'authentic_samples.json');
if (!fs.existsSync(samplesPath)) {
    console.error('ERROR: authentic_samples.json not found.');
    console.error('Run: python tests/fetch_authentic_samples.py');
    process.exit(1);
}

const data = JSON.parse(fs.readFileSync(samplesPath, 'utf-8'));
const samples = data.samples;

console.log('='.repeat(70));
console.log('VERITAS — Authentic Sample Benchmark');
console.log('='.repeat(70));
console.log(`Loaded: ${data.metadata.total_samples} samples`);
console.log(`  Human: ${data.metadata.human_count}`);
console.log(`  AI: ${data.metadata.ai_count}`);
console.log(`  Sources: ${data.metadata.sources.join(', ')}`);
console.log('='.repeat(70));

// =============================================================================
// DETECTION ENGINE (inline for benchmarking)
// Data-driven thresholds from analysis of 289 authentic samples
// =============================================================================

function analyze(text) {
    if (!text || text.length < 20) {
        return { prediction: 'unknown', confidence: 0, signals: {} };
    }
    
    const signals = {};
    const words = text.split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const wordCount = words.length;
    const sentCount = sentences.length;
    
    // =============================================================================
    // FEATURE EXTRACTION (based on statistical analysis)
    // =============================================================================
    
    // --- Sentence Statistics ---
    const sentenceLengths = sentences.map(s => s.trim().split(/\s+/).length);
    const avgSentLen = sentCount > 0 ? sentenceLengths.reduce((a, b) => a + b, 0) / sentCount : 0;
    const sentVariance = sentCount > 0 ? sentenceLengths.reduce((acc, len) => acc + Math.pow(len - avgSentLen, 2), 0) / sentCount : 0;
    const sentCV = avgSentLen > 0 ? Math.sqrt(sentVariance) / avgSentLen : 0;
    signals.avgSentLen = avgSentLen;
    signals.sentCV = sentCV;
    signals.sentCount = sentCount;
    signals.wordCount = wordCount;
    
    // --- Exclamation Rate (Human: 0.053, AI: 0.013 — 75% difference!) ---
    const exclamations = (text.match(/!/g) || []).length;
    signals.exclamRate = sentCount > 0 ? exclamations / sentCount : 0;
    
    // --- Contraction Rate (Human: 0.0056, AI: 0.0031 — 44% difference) ---
    const contractions = (text.match(/\b(I'm|you're|we're|they're|he's|she's|it's|that's|what's|there's|here's|who's|can't|won't|don't|didn't|couldn't|wouldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|I'll|you'll|we'll|they'll|I've|you've|we've|they've|I'd|you'd|we'd|they'd|let's|ain't)\b/gi) || []).length;
    signals.contractionRate = contractions / wordCount;
    
    // --- First Person Rate (Human: 0.023, AI: 0.014 — 41% difference) ---
    const firstPerson = (text.match(/\b(I|me|my|mine|myself)\b/gi) || []).length;
    signals.firstPersonRate = firstPerson / wordCount;
    
    // --- Comma Rate per Sentence (Human: 0.72, AI: 1.19 — AI uses 65% more!) ---
    const commas = (text.match(/,/g) || []).length;
    signals.commaRate = sentCount > 0 ? commas / sentCount : 0;
    
    // --- Question Rate (Human: 0.082, AI: 0.064 — 22% difference) ---
    const questions = (text.match(/\?/g) || []).length;
    signals.questionRate = sentCount > 0 ? questions / sentCount : 0;
    
    // --- Hedging Words ---
    const hedges = (text.match(/\b(maybe|perhaps|probably|might|could|seem|think|guess|feel|believe|suppose|apparently|basically|actually|honestly|frankly|literally)\b/gi) || []).length;
    signals.hedgeRate = hedges / wordCount;
    
    // --- Discourse Markers (formal connectors - AI uses more) ---
    const discourse = (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length;
    signals.discourseRate = discourse / wordCount;
    
    // --- Casual Markers ---
    const casualPatterns = [
        /\b(lol|lmao|haha|hehe|omg|wtf|idk|tbh|imo|ngl|smh|fml|brb|gtg|btw|afaik)\b/gi,
        /\b(kinda|sorta|gonna|wanna|gotta|dunno|lemme|gimme|ya|yea|yeah|nah|nope|yep|yup)\b/gi,
        /\.\.\./g,
        /!{2,}/g,
        /\?{2,}/g
    ];
    let casualCount = 0;
    for (const pattern of casualPatterns) {
        const matches = text.match(pattern);
        if (matches) casualCount += matches.length;
    }
    signals.casualMarkers = casualCount;
    
    // --- Typos (human indicator) ---
    const typoPatterns = [
        /\b(teh|recieve|seperate|definately|occured|accomodate)\b/gi,
        /\bi\s+[a-z]/g,
        /[a-z]\s{2,}[a-z]/g,
        /[.!?]\s*[a-z]/g
    ];
    let typoCount = 0;
    for (const pattern of typoPatterns) {
        const matches = text.match(pattern);
        if (matches) typoCount += matches.length;
    }
    signals.typos = typoCount;
    
    // --- Structured formatting (slight AI indicator) ---
    const hasNumberedList = /^\s*\d+[\.\)]/m.test(text);
    const hasBulletPoints = /^\s*[-•*]\s/m.test(text);
    signals.structuredFormatting = (hasNumberedList ? 1 : 0) + (hasBulletPoints ? 1 : 0);
    
    // --- Past Tense Rate (Formal Human: 0.038, AI: 0.020 — 48% difference) ---
    // News/Wikipedia articles use more past-tense verbs describing events
    const pastTenseVerbs = (text.match(/\b\w+ed\b/gi) || []).length;
    signals.pastRate = pastTenseVerbs / wordCount;
    
    // --- Proper Noun Rate (Formal Human: 0.157, AI: 0.083 — 47% difference) ---
    // Factual content has more names, places, organizations
    const properNouns = (text.match(/(?<=[.!?\s])[A-Z][a-z]+/g) || []).length;
    signals.properRate = properNouns / wordCount;
    
    // === NEW HIGH-SIGNAL FEATURES ===
    
    // --- Sentence Starting with Conjunction (Human: 0.23, AI: 0.04 — 81% diff!) ---
    // Humans casually start sentences with "But", "And", "So"
    const sentStartConj = (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length;
    signals.sentStartConj = sentStartConj;
    
    // --- "It is important/essential/crucial" (Human: 0, AI: 0.006 — 100% AI!) ---
    const itIsImportant = (text.match(/\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning)\b/gi) || []).length;
    signals.itIsImportant = itIsImportant;
    
    // --- Colon Rate (Human: 0.04, AI: 0.13 — 71% AI indicator) ---
    const colons = (text.match(/:/g) || []).length;
    signals.colonRate = sentCount > 0 ? colons / sentCount : 0;
    
    // --- "Can be/Could be/May be" (Human: 0.01, AI: 0.03 — 64% AI indicator) ---
    const canBe = (text.match(/\b(can|could|may|might) be\b/gi) || []).length;
    signals.canBeRate = sentCount > 0 ? canBe / sentCount : 0;
    
    // --- "This" starting sentences (Human: 0.01, AI: 0.02 — 31% AI) ---
    const thisStart = (text.match(/[.!?]\s+This\s/gi) || []).length;
    signals.thisStartRate = sentCount > 0 ? thisStart / sentCount : 0;
    
    // =============================================================================
    // SCORING (Data-driven from 289 authentic samples)
    // Human averages: exclamRate=0.053, contractionRate=0.0056, sentCV=0.46, commaRate=0.72
    // AI averages:    exclamRate=0.013, contractionRate=0.0031, sentCV=0.35, commaRate=1.19
    // =============================================================================
    
    let aiScore = 0;
    let humanScore = 0;
    
    // === EXCLAMATION RATE (75% difference - strongest signal!) ===
    // Human: 0.053, AI: 0.013
    if (signals.exclamRate > 0.08) humanScore += 25;
    else if (signals.exclamRate > 0.04) humanScore += 15;
    else if (signals.exclamRate > 0.02) humanScore += 8;
    else if (signals.exclamRate < 0.01 && sentCount >= 3) aiScore += 10;
    
    // === CONTRACTION RATE (44% difference) ===
    // Human: 0.0056, AI: 0.0031
    if (signals.contractionRate > 0.008) humanScore += 20;
    else if (signals.contractionRate > 0.005) humanScore += 12;
    else if (signals.contractionRate > 0.003) humanScore += 5;
    else if (signals.contractionRate < 0.002 && wordCount > 50) aiScore += 8;
    
    // === SENTENCE CV (23% difference) ===
    // Human: 0.46, AI: 0.35
    if (signals.sentCV > 0.55) humanScore += 15;
    else if (signals.sentCV > 0.45) humanScore += 8;
    else if (signals.sentCV < 0.30) aiScore += 15;
    else if (signals.sentCV < 0.38) aiScore += 8;
    
    // === COMMA RATE (39% difference - AI uses MORE commas) ===
    // Human: 0.72, AI: 1.19
    if (signals.commaRate > 1.5) aiScore += 15;
    else if (signals.commaRate > 1.1) aiScore += 8;
    else if (signals.commaRate < 0.5) humanScore += 8;
    
    // === FIRST PERSON RATE (41% difference) ===
    // Human: 0.023, AI: 0.014
    if (signals.firstPersonRate > 0.035) humanScore += 15;
    else if (signals.firstPersonRate > 0.02) humanScore += 8;
    else if (signals.firstPersonRate < 0.01 && wordCount > 40) aiScore += 5;
    
    // === QUESTION RATE (22% difference) ===
    // Human: 0.082, AI: 0.064
    if (signals.questionRate > 0.15) humanScore += 10;
    else if (signals.questionRate > 0.08) humanScore += 5;
    
    // === CASUAL MARKERS (strong human signal) ===
    if (signals.casualMarkers >= 3) humanScore += 25;
    else if (signals.casualMarkers >= 1) humanScore += 12;
    
    // === DISCOURSE MARKERS (AI uses formal connectors) ===
    if (signals.discourseRate > 0.015) aiScore += 12;
    else if (signals.discourseRate > 0.008) aiScore += 6;
    
    // === HEDGING (slight human indicator) ===
    if (signals.hedgeRate > 0.01) humanScore += 8;
    else if (signals.hedgeRate > 0.005) humanScore += 4;
    
    // === TYPOS (human indicator) ===
    if (signals.typos >= 2) humanScore += 15;
    else if (signals.typos >= 1) humanScore += 8;
    
    // === STRUCTURED FORMATTING (slight AI indicator) ===
    if (signals.structuredFormatting >= 2) aiScore += 8;
    else if (signals.structuredFormatting >= 1) aiScore += 4;
    
    // === AVERAGE SENTENCE LENGTH ===
    // Human: 16.9, AI: 19.0
    if (signals.avgSentLen > 22) aiScore += 10;
    else if (signals.avgSentLen > 18) aiScore += 5;
    else if (signals.avgSentLen < 12) humanScore += 5;
    
    // === PAST TENSE RATE (48% difference - key for formal human content) ===
    // Formal Human: 0.038, AI: 0.020
    // News/Wikipedia describe past events with past-tense verbs
    if (signals.pastRate > 0.05) humanScore += 15;
    else if (signals.pastRate > 0.035) humanScore += 10;
    else if (signals.pastRate > 0.025) humanScore += 5;
    else if (signals.pastRate < 0.015 && wordCount > 50) aiScore += 5;
    
    // === PROPER NOUN RATE (47% difference - key for formal human content) ===
    // Formal Human: 0.157, AI: 0.083
    // Factual articles mention names, places, organizations
    if (signals.properRate > 0.18) humanScore += 15;
    else if (signals.properRate > 0.12) humanScore += 10;
    else if (signals.properRate > 0.08) humanScore += 5;
    else if (signals.properRate < 0.05 && wordCount > 50) aiScore += 5;
    
    // === FORMAL HUMAN COMBO (past tense + proper nouns = likely news/wiki) ===
    // If text has BOTH high past-tense AND high proper-noun rates, likely formal human
    if (signals.pastRate > 0.03 && signals.properRate > 0.10) {
        humanScore += 20;  // Strong boost for formal human content
    }
    
    // =============================================================================
    // NEW HIGH-SIGNAL FEATURES (from data analysis)
    // =============================================================================
    
    // === SENTENCE STARTING WITH CONJUNCTION (81% diff - strongest new signal!) ===
    // Human: 0.23, AI: 0.04 — Humans casually start with "But", "And", "So"
    if (signals.sentStartConj >= 2) humanScore += 25;
    else if (signals.sentStartConj >= 1) humanScore += 15;
    
    // === "IT IS IMPORTANT/ESSENTIAL" (100% AI indicator!) ===
    // Human: 0, AI: 0.006 — This phrase is almost never used by humans
    if (signals.itIsImportant >= 1) aiScore += 30;  // Strong AI signal
    
    // === COLON RATE (71% AI indicator) ===
    // Human: 0.04, AI: 0.13 — AI uses more structured formatting with colons
    if (signals.colonRate > 0.15) aiScore += 15;
    else if (signals.colonRate > 0.08) aiScore += 8;
    else if (signals.colonRate < 0.03) humanScore += 5;
    
    // === "CAN BE/COULD BE/MAY BE" (64% AI indicator) ===
    // Human: 0.01, AI: 0.03 — AI hedges with passive constructions
    if (signals.canBeRate > 0.04) aiScore += 15;
    else if (signals.canBeRate > 0.02) aiScore += 8;
    
    // === "THIS" STARTING SENTENCES (31% AI indicator) ===
    // Human: 0.01, AI: 0.02 — AI uses "This" for cohesion
    if (signals.thisStartRate > 0.03) aiScore += 8;
    else if (signals.thisStartRate > 0.015) aiScore += 4;
    
    // =============================================================================
    // FINAL PREDICTION
    // =============================================================================
    
    const netScore = aiScore - humanScore;
    let prediction, confidence;
    
    if (netScore > 20) {
        prediction = 'ai';
        confidence = Math.min(95, 55 + netScore);
    } else if (netScore > 8) {
        prediction = 'ai';
        confidence = 45 + netScore;
    } else if (netScore < -20) {
        prediction = 'human';
        confidence = Math.min(95, 55 + Math.abs(netScore));
    } else if (netScore < -8) {
        prediction = 'human';
        confidence = 45 + Math.abs(netScore);
    } else {
        // Close call - use secondary signals to break tie
        if (signals.casualMarkers >= 1 || signals.typos >= 1 || signals.exclamRate > 0.03) {
            prediction = 'human';
            confidence = 40 + Math.abs(netScore);
        } else if (signals.sentCV < 0.38 || signals.commaRate > 1.0) {
            prediction = 'ai';
            confidence = 40 + Math.abs(netScore);
        } else {
            // Default based on slight lean
            prediction = netScore >= 0 ? 'ai' : 'human';
            confidence = 35 + Math.abs(netScore);
        }
    }
    
    return {
        prediction,
        confidence,
        signals,
        scores: { ai: aiScore, human: humanScore, net: netScore }
    };
}

// =============================================================================
// RUN BENCHMARK
// =============================================================================

function runBenchmark() {
    let correct = 0;
    let incorrect = 0;
    let uncertain = 0;
    
    const results = {
        human: { correct: 0, incorrect: 0, uncertain: 0, total: 0 },
        ai: { correct: 0, incorrect: 0, uncertain: 0, total: 0 }
    };
    
    const errors = [];
    const bySource = {};
    
    for (const sample of samples) {
        const result = analyze(sample.text);
        const expected = sample.label;
        const predicted = result.prediction;
        
        // Track by source
        if (!bySource[sample.source]) {
            bySource[sample.source] = { correct: 0, incorrect: 0, uncertain: 0, total: 0, label: sample.label };
        }
        bySource[sample.source].total++;
        
        results[expected].total++;
        
        if (predicted === 'uncertain') {
            uncertain++;
            results[expected].uncertain++;
            bySource[sample.source].uncertain++;
        } else if (predicted === expected) {
            correct++;
            results[expected].correct++;
            bySource[sample.source].correct++;
        } else {
            incorrect++;
            results[expected].incorrect++;
            bySource[sample.source].incorrect++;
            errors.push({
                id: sample.id,
                source: sample.source,
                category: sample.category,
                expected,
                predicted,
                confidence: result.confidence,
                text: sample.text.substring(0, 150) + '...',
                signals: result.signals,
                scores: result.scores
            });
        }
    }
    
    const total = samples.length;
    const accuracy = ((correct / total) * 100).toFixed(1);
    const errorRate = ((incorrect / total) * 100).toFixed(1);
    const uncertainRate = ((uncertain / total) * 100).toFixed(1);
    
    console.log('\n' + '='.repeat(70));
    console.log('BENCHMARK RESULTS');
    console.log('='.repeat(70));
    console.log(`Total Samples: ${total}`);
    console.log(`Correct: ${correct} (${accuracy}%)`);
    console.log(`Incorrect: ${incorrect} (${errorRate}%)`);
    console.log(`Uncertain: ${uncertain} (${uncertainRate}%)`);
    
    console.log('\n--- By Label ---');
    console.log(`Human: ${results.human.correct}/${results.human.total} correct (${((results.human.correct / results.human.total) * 100).toFixed(1)}%)`);
    console.log(`AI: ${results.ai.correct}/${results.ai.total} correct (${((results.ai.correct / results.ai.total) * 100).toFixed(1)}%)`);
    
    // False positive/negative rates
    const falsePositives = results.human.incorrect;  // Human misclassified as AI
    const falseNegatives = results.ai.incorrect;     // AI misclassified as human
    const fpRate = ((falsePositives / results.human.total) * 100).toFixed(1);
    const fnRate = ((falseNegatives / results.ai.total) * 100).toFixed(1);
    
    console.log(`\nFalse Positive Rate (human→AI): ${fpRate}%`);
    console.log(`False Negative Rate (AI→human): ${fnRate}%`);
    
    console.log('\n--- By Source ---');
    for (const [source, stats] of Object.entries(bySource).sort((a, b) => a[0].localeCompare(b[0]))) {
        const pct = ((stats.correct / stats.total) * 100).toFixed(0);
        const label = stats.label.toUpperCase();
        console.log(`  ${source}: ${stats.correct}/${stats.total} (${pct}%) [${label}]`);
    }
    
    if (errors.length > 0 && errors.length <= 30) {
        console.log('\n--- Error Details (First 30) ---');
        for (const err of errors.slice(0, 30)) {
            console.log(`\n[${err.id}] ${err.source} | ${err.category}`);
            console.log(`  Expected: ${err.expected} | Predicted: ${err.predicted} (${err.confidence}%)`);
            console.log(`  Scores: AI=${err.scores.ai}, Human=${err.scores.human}, Net=${err.scores.net}`);
            console.log(`  Key signals: vocab=${err.signals.ai_vocabulary}, phrases=${err.signals.ai_phrases}, casual=${err.signals.casual_markers}`);
            console.log(`  Text: "${err.text}"`);
        }
    } else if (errors.length > 30) {
        console.log(`\n--- Too many errors to display (${errors.length}). Showing summary ---`);
        
        // Group errors by pattern
        const fpErrors = errors.filter(e => e.expected === 'human');
        const fnErrors = errors.filter(e => e.expected === 'ai');
        
        console.log(`\nFalse Positives (human→AI): ${fpErrors.length}`);
        const fpSources = {};
        for (const e of fpErrors) {
            fpSources[e.source] = (fpSources[e.source] || 0) + 1;
        }
        for (const [src, count] of Object.entries(fpSources).sort((a, b) => b[1] - a[1])) {
            console.log(`  ${src}: ${count}`);
        }
        
        console.log(`\nFalse Negatives (AI→human): ${fnErrors.length}`);
        const fnSources = {};
        for (const e of fnErrors) {
            fnSources[e.source] = (fnSources[e.source] || 0) + 1;
        }
        for (const [src, count] of Object.entries(fnSources).sort((a, b) => b[1] - a[1])) {
            console.log(`  ${src}: ${count}`);
        }
    }
    
    console.log('\n' + '='.repeat(70));
    
    return { accuracy: parseFloat(accuracy), errors };
}

// Run it
runBenchmark();
