#!/usr/bin/env node
/**
 * VERITAS Data-Driven Benchmark
 * Using statistically validated signals from 195k samples
 * 
 * Key signals (Cohen's d > 0.3 = meaningful effect):
 * - pastTenseRate: d=0.462 (Human higher)
 * - sentStartConj: d=0.461 (Human higher) 
 * - sentCV: d=0.427 (Human higher)
 * - secondPersonRate: d=-0.391 (AI higher)
 * - firstPersonRate: d=0.341 (Human higher)
 * - discourseMarkers: d=-0.340 (AI higher)
 * - canBeRate: d=-0.324 (AI higher)
 * - commaRate: d=-0.243 (AI higher)
 */

const fs = require('fs');
const path = require('path');

// Load dataset
const dataPath = path.join(__dirname, 'large_samples.json');
const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
const samples = data.samples;

console.log('=' .repeat(70));
console.log('VERITAS — Data-Driven Benchmark (195k samples)');
console.log('=' .repeat(70));
console.log(`Total: ${samples.length} | Human: ${data.metadata.human} | AI: ${data.metadata.ai}`);
console.log('');

// =============================================================================
// FEATURE EXTRACTION (exactly matching analysis)
// =============================================================================

function extractFeatures(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    // Sentence length stats
    const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
    const avgSentLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length || 0;
    const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - avgSentLen, 2), 0) / sentLengths.length || 0;
    const sentCV = avgSentLen > 0 ? Math.sqrt(variance) / avgSentLen : 0;
    
    // Proper nouns (capitalized words not at sentence start)
    const properNounMatches = text.match(/(?<![.!?]\s)[A-Z][a-z]+/g) || [];
    const properNouns = properNounMatches.length / wordCount;
    
    // Helpful phrases (AI assistant markers)
    const helpfulPhrases = (text.match(/\b(here is|here are|feel free|I hope this helps|let me|I can help|I'd be happy|happy to help|sure thing|great question|good question)\b/gi) || []).length;
    
    // Instructional/explanation markers
    const instructionalMarkers = (text.match(/\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|it's important to|consider the)\b/gi) || []).length;
    
    // Quoted speech (journalism marker)
    const quotedSpeech = (text.match(/"[^"]{10,}"/g) || []).length;
    
    // Third person pronouns (news/formal writing)
    const thirdPerson = (text.match(/\b(he|she|they|him|her|them|his|hers|their|theirs)\b/gi) || []).length / wordCount;
    
    return {
        // Top discriminative features (sorted by effect size)
        pastTenseRate: (text.match(/\b\w+ed\b/gi) || []).length / wordCount,
        sentStartConj: (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length,
        sentCV: sentCV,
        secondPersonRate: (text.match(/\b(you|your|yours|yourself|yourselves)\b/gi) || []).length / wordCount,
        firstPersonRate: (text.match(/\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b/gi) || []).length / wordCount,
        discourseMarkers: (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length / wordCount,
        canBeRate: (text.match(/\b(can|could|may|might) be\b/gi) || []).length / sentCount,
        commaRate: (text.match(/,/g) || []).length / sentCount,
        exclamRate: (text.match(/!/g) || []).length / sentCount,
        
        // Additional useful features
        contractionRate: (text.match(/\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b/gi) || []).length / wordCount,
        casualMarkers: (text.match(/\b(lol|lmao|haha|omg|wtf|idk|tbh|imo|ngl|kinda|gonna|wanna|gotta|yeah|nah|yep)\b/gi) || []).length,
        itIsImportant: (text.match(/\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning)\b/gi) || []).length,
        avgSentLen: avgSentLen,
        
        // NEW: Features to reduce FP/FN on problematic sources
        properNouns: properNouns,           // News: 0.21, AI: 0.05-0.08
        helpfulPhrases: helpfulPhrases,     // AI: 0.12-0.33, News: 0.00
        instructionalMarkers: instructionalMarkers,  // UltraChat: 0.97, News: 0.16
        quotedSpeech: quotedSpeech,         // News has journalism quotes
        thirdPerson: thirdPerson,           // News/formal writing
        
        // NEW: GPT4All/Dolly/Anthropic specific patterns
        hasHtmlTags: /<[a-z]+>/i.test(text),            // GPT4All: 10%, Human: 0%
        hasCodeBlocks: /```|<code>|<pre>/.test(text),   // GPT4All: 24%, Human: 0%
        textLength: text.length,                        // Anthropic/Dolly: 50%/32% short
        refusalPattern: /I (don't|can't|won't|cannot|shouldn't) (want to|agree|think|help with|assist|provide)/i.test(text)
    };
}

// =============================================================================
// SCORING (Using statistically validated thresholds from 195k analysis)
// =============================================================================

function analyze(text) {
    const features = extractFeatures(text);
    
    let humanScore = 0;
    let aiScore = 0;
    
    // === PAST TENSE RATE (d=0.462, Human higher) ===
    // Thresholds: Human P50=0.031, AI P50=0.018, AI P75=0.036
    if (features.pastTenseRate > 0.0335) humanScore += 12;
    else if (features.pastTenseRate > 0.025) humanScore += 6;
    else if (features.pastTenseRate < 0.015) aiScore += 8;
    
    // === SENTENCE STARTING WITH CONJUNCTION (d=0.461, Human higher) ===
    // Any occurrence is a human signal
    if (features.sentStartConj >= 2) humanScore += 15;
    else if (features.sentStartConj >= 1) humanScore += 10;
    
    // === SENTENCE CV (d=0.427, Human higher) ===
    // Threshold: Human P50=0.527, AI P50=0.357, optimal > 0.566
    if (features.sentCV > 0.566) humanScore += 12;
    else if (features.sentCV > 0.45) humanScore += 6;
    else if (features.sentCV < 0.30) aiScore += 12;
    else if (features.sentCV < 0.38) aiScore += 6;
    
    // === SECOND PERSON RATE (d=-0.391, AI higher) ===
    // Threshold: AI P75=0.032, Human P75=0.013
    if (features.secondPersonRate > 0.025) aiScore += 10;
    else if (features.secondPersonRate > 0.015) aiScore += 5;
    
    // === FIRST PERSON RATE (d=0.341, Human higher) ===
    // Threshold: Human P50=0.012, AI P50=0.000
    if (features.firstPersonRate > 0.030) humanScore += 10;
    else if (features.firstPersonRate > 0.015) humanScore += 6;
    else if (features.firstPersonRate < 0.005 && features.secondPersonRate < 0.01) aiScore += 4;
    
    // === DISCOURSE MARKERS (d=-0.340, AI higher) ===
    // Any occurrence is an AI signal
    if (features.discourseMarkers > 0.003) aiScore += 12;
    else if (features.discourseMarkers > 0.001) aiScore += 6;
    
    // === CAN BE RATE (d=-0.324, AI higher) ===
    // Any occurrence is an AI signal
    if (features.canBeRate > 0.05) aiScore += 10;
    else if (features.canBeRate > 0.02) aiScore += 6;
    
    // === COMMA RATE (d=-0.243, AI higher) ===
    // Threshold: Human P50=0.667, AI P50=0.909
    if (features.commaRate > 1.2) aiScore += 8;
    else if (features.commaRate > 0.95) aiScore += 4;
    else if (features.commaRate < 0.5) humanScore += 6;
    
    // === EXCLAMATION RATE (d=0.234, Human higher) ===
    if (features.exclamRate > 0.08) humanScore += 10;
    else if (features.exclamRate > 0.04) humanScore += 5;
    
    // === CONTRACTIONS (d=0.194, Human higher) ===
    if (features.contractionRate > 0.012) humanScore += 8;
    else if (features.contractionRate > 0.006) humanScore += 4;
    else if (features.contractionRate < 0.002) aiScore += 4;
    
    // === CASUAL MARKERS (strong human signal when present) ===
    if (features.casualMarkers >= 2) humanScore += 20;
    else if (features.casualMarkers >= 1) humanScore += 12;
    
    // === "IT IS IMPORTANT" (d=-0.200, AI signal) ===
    if (features.itIsImportant >= 1) aiScore += 15;
    
    // === AVERAGE SENTENCE LENGTH (d=-0.166, AI longer) ===
    if (features.avgSentLen > 22) aiScore += 6;
    else if (features.avgSentLen < 14) humanScore += 4;
    
    // =========================================================================
    // NEW FEATURES: Target FP (News/C4) and FN (Anthropic/UltraChat) reduction
    // =========================================================================
    
    // === PROPER NOUNS (News: 0.21, AI: 0.05-0.08) ===
    // High proper noun density = formal human writing (journalism, reports)
    if (features.properNouns > 0.18) humanScore += 15;
    else if (features.properNouns > 0.12) humanScore += 10;
    else if (features.properNouns < 0.05) aiScore += 5;
    
    // === HELPFUL PHRASES (AI: 0.12-0.33, News: 0.00) ===
    // Strong AI assistant marker
    if (features.helpfulPhrases >= 2) aiScore += 20;
    else if (features.helpfulPhrases >= 1) aiScore += 15;
    
    // === INSTRUCTIONAL MARKERS (UltraChat: 0.97, News: 0.16) ===
    // Tutorial/explanation style is AI marker
    if (features.instructionalMarkers >= 3) aiScore += 15;
    else if (features.instructionalMarkers >= 1) aiScore += 8;
    
    // === QUOTED SPEECH (Journalism marker) ===
    // Direct quotes from sources = human journalism
    if (features.quotedSpeech >= 2) humanScore += 12;
    else if (features.quotedSpeech >= 1) humanScore += 6;
    
    // === THIRD PERSON (News/formal writing) ===
    // Combined with low first person = journalism style
    if (features.thirdPerson > 0.015 && features.firstPersonRate < 0.01) humanScore += 8;
    
    // === COMBO: Formal Human (News pattern) ===
    // High proper nouns + low second person + no helpful phrases = formal journalism
    if (features.properNouns > 0.12 && features.secondPersonRate < 0.005 && features.helpfulPhrases === 0) {
        humanScore += 12;
    }
    
    // === COMBO: Conversational AI (RLHF pattern) ===
    // High second person + helpful phrases + instructional = AI assistant
    if (features.secondPersonRate > 0.03 && (features.helpfulPhrases >= 1 || features.instructionalMarkers >= 2)) {
        aiScore += 12;
    }
    
    // =========================================================================
    // NEW: GPT4All/Dolly/Anthropic specific patterns
    // =========================================================================
    
    // === HTML TAGS (GPT4All: 10%, Human: ~0%) ===
    // Strong AI signal - technical Q&A with HTML formatting
    if (features.hasHtmlTags) aiScore += 18;
    
    // === CODE BLOCKS (GPT4All: 24%, Human: ~0%) ===
    // Very strong AI signal - programming assistance
    if (features.hasCodeBlocks) aiScore += 20;
    
    // === REFUSAL PATTERN (Anthropic RLHF safety responses) ===
    // AI safety/refusal language is a clear AI marker
    if (features.refusalPattern) aiScore += 15;
    
    // Final prediction
    const netScore = aiScore - humanScore;
    let prediction, confidence;
    
    if (netScore > 15) {
        prediction = 'ai';
        confidence = Math.min(95, 50 + netScore);
    } else if (netScore > 5) {
        prediction = 'ai';
        confidence = 40 + netScore;
    } else if (netScore < -15) {
        prediction = 'human';
        confidence = Math.min(95, 50 + Math.abs(netScore));
    } else if (netScore < -5) {
        prediction = 'human';
        confidence = 40 + Math.abs(netScore);
    } else {
        // Tie-breaker: use strongest signals
        if (features.casualMarkers >= 1 || features.sentStartConj >= 1) {
            prediction = 'human';
            confidence = 35;
        } else if (features.discourseMarkers > 0 || features.itIsImportant >= 1) {
            prediction = 'ai';
            confidence = 35;
        } else if (features.sentCV > 0.45) {
            prediction = 'human';
            confidence = 30;
        } else {
            prediction = 'ai';
            confidence = 30;
        }
    }
    
    return {
        prediction,
        confidence,
        features,
        scores: { human: humanScore, ai: aiScore, net: netScore }
    };
}

// =============================================================================
// RUN BENCHMARK
// =============================================================================

console.log('Running benchmark on all samples...');
const startTime = Date.now();

// Use stratified sample for faster testing (10% of each)
const humanSamples = samples.filter(s => s.label === 'human');
const aiSamples = samples.filter(s => s.label === 'ai');

// Shuffle and take sample
function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

// Test on 10% sample for speed (can run full later)
const testHuman = shuffle([...humanSamples]).slice(0, 10000);
const testAI = shuffle([...aiSamples]).slice(0, 10000);
const testSamples = shuffle([...testHuman, ...testAI]);

console.log(`Testing on ${testSamples.length} samples (10% stratified sample)...`);

let correct = 0;
let incorrect = 0;
const errors = { fp: [], fn: [] };
const sourceStats = {};

for (const sample of testSamples) {
    const result = analyze(sample.text);
    const isCorrect = result.prediction === sample.label;
    
    if (isCorrect) {
        correct++;
    } else {
        incorrect++;
        if (sample.label === 'human') {
            errors.fp.push({ source: sample.source, text: sample.text.substring(0, 100) });
        } else {
            errors.fn.push({ source: sample.source, text: sample.text.substring(0, 100) });
        }
    }
    
    // Track by source
    if (!sourceStats[sample.source]) {
        sourceStats[sample.source] = { correct: 0, total: 0, label: sample.label };
    }
    sourceStats[sample.source].total++;
    if (isCorrect) sourceStats[sample.source].correct++;
}

const elapsed = (Date.now() - startTime) / 1000;

// =============================================================================
// RESULTS
// =============================================================================

console.log('\n' + '=' .repeat(70));
console.log('BENCHMARK RESULTS');
console.log('=' .repeat(70));

const accuracy = (correct / testSamples.length * 100).toFixed(1);
console.log(`Total Samples: ${testSamples.length}`);
console.log(`Correct: ${correct} (${accuracy}%)`);
console.log(`Incorrect: ${incorrect}`);

const humanCorrect = testHuman.filter(s => analyze(s.text).prediction === 'human').length;
const aiCorrect = testAI.filter(s => analyze(s.text).prediction === 'ai').length;

console.log(`\nHuman: ${humanCorrect}/${testHuman.length} (${(humanCorrect/testHuman.length*100).toFixed(1)}%)`);
console.log(`AI: ${aiCorrect}/${testAI.length} (${(aiCorrect/testAI.length*100).toFixed(1)}%)`);

console.log(`\nFalse Positive Rate (human→AI): ${(errors.fp.length/testHuman.length*100).toFixed(1)}%`);
console.log(`False Negative Rate (AI→human): ${(errors.fn.length/testAI.length*100).toFixed(1)}%`);

console.log('\n--- By Source ---');
for (const [source, stats] of Object.entries(sourceStats).sort((a, b) => a[0].localeCompare(b[0]))) {
    const pct = (stats.correct / stats.total * 100).toFixed(0);
    console.log(`  ${source.padEnd(20)} ${stats.correct}/${stats.total} (${pct}%) [${stats.label.toUpperCase()}]`);
}

// Error breakdown
console.log('\n--- Top Error Sources ---');
const fpBySource = {};
const fnBySource = {};
for (const e of errors.fp) fpBySource[e.source] = (fpBySource[e.source] || 0) + 1;
for (const e of errors.fn) fnBySource[e.source] = (fnBySource[e.source] || 0) + 1;

console.log('False Positives (human→AI):');
for (const [src, count] of Object.entries(fpBySource).sort((a, b) => b[1] - a[1]).slice(0, 5)) {
    console.log(`  ${src}: ${count}`);
}

console.log('False Negatives (AI→human):');
for (const [src, count] of Object.entries(fnBySource).sort((a, b) => b[1] - a[1]).slice(0, 5)) {
    console.log(`  ${src}: ${count}`);
}

console.log(`\nCompleted in ${elapsed.toFixed(1)}s`);
console.log('=' .repeat(70));
