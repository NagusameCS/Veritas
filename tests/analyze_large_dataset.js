#!/usr/bin/env node
/**
 * Statistical Analysis of Large Dataset (195k+ samples)
 * Calculate feature distributions with statistical significance
 */

const fs = require('fs');
const path = require('path');

const dataPath = path.join(__dirname, 'large_samples.json');
console.log('Loading large dataset...');
const data = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
const samples = data.samples;

console.log(`Loaded ${samples.length} samples`);
console.log(`Human: ${data.metadata.human}, AI: ${data.metadata.ai}`);
console.log(`Sources: ${data.metadata.sources ? data.metadata.sources.length : 'unknown'}\n`);

// =============================================================================
// FEATURE EXTRACTION
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
    
    return {
        // Punctuation
        exclamRate: (text.match(/!/g) || []).length / sentCount,
        questionRate: (text.match(/\?/g) || []).length / sentCount,
        commaRate: (text.match(/,/g) || []).length / sentCount,
        colonRate: (text.match(/:/g) || []).length / sentCount,
        semicolonRate: (text.match(/;/g) || []).length / sentCount,
        
        // Contractions
        contractionRate: (text.match(/\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b/gi) || []).length / wordCount,
        
        // Pronouns
        firstPersonRate: (text.match(/\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b/gi) || []).length / wordCount,
        secondPersonRate: (text.match(/\b(you|your|yours|yourself|yourselves)\b/gi) || []).length / wordCount,
        
        // Sentence structure
        sentCV: sentCV,
        avgSentLen: avgSentLen,
        
        // AI indicators
        discourseMarkers: (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length / wordCount,
        itIsImportant: (text.match(/\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning)\b/gi) || []).length,
        canBeRate: (text.match(/\b(can|could|may|might) be\b/gi) || []).length / sentCount,
        thisStart: (text.match(/[.!?]\s+This\s/gi) || []).length / sentCount,
        inOrderTo: (text.match(/\bin order to\b/gi) || []).length,
        
        // Human indicators
        sentStartConj: (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length,
        casualMarkers: (text.match(/\b(lol|lmao|haha|omg|wtf|idk|tbh|imo|ngl|kinda|gonna|wanna|gotta|yeah|nah|yep)\b/gi) || []).length,
        hedgeWords: (text.match(/\b(maybe|perhaps|probably|might|could|seem|think|guess|feel|believe|suppose|basically|actually|honestly)\b/gi) || []).length / wordCount,
        
        // Past tense and proper nouns (formal human)
        pastTenseRate: (text.match(/\b\w+ed\b/gi) || []).length / wordCount,
        
        // Word-level
        avgWordLen: words.reduce((sum, w) => sum + w.length, 0) / wordCount,
        
        // Additional patterns
        numberedLists: (text.match(/^\s*\d+[.)]/gm) || []).length,
        bulletPoints: (text.match(/^\s*[-•*]\s/gm) || []).length,
    };
}

// =============================================================================
// STATISTICAL FUNCTIONS
// =============================================================================

function mean(arr) {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function std(arr) {
    const m = mean(arr);
    return Math.sqrt(arr.reduce((sum, x) => sum + Math.pow(x - m, 2), 0) / arr.length);
}

function percentile(arr, p) {
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.floor(sorted.length * p);
    return sorted[idx];
}

function cohensD(arr1, arr2) {
    const m1 = mean(arr1);
    const m2 = mean(arr2);
    const s1 = std(arr1);
    const s2 = std(arr2);
    const pooledStd = Math.sqrt((s1 * s1 + s2 * s2) / 2);
    return pooledStd > 0 ? (m1 - m2) / pooledStd : 0;
}

function tTest(arr1, arr2) {
    const n1 = arr1.length;
    const n2 = arr2.length;
    const m1 = mean(arr1);
    const m2 = mean(arr2);
    const s1 = std(arr1);
    const s2 = std(arr2);
    
    const se = Math.sqrt((s1 * s1 / n1) + (s2 * s2 / n2));
    const t = se > 0 ? (m1 - m2) / se : 0;
    
    // Simplified p-value approximation for large samples
    const df = Math.min(n1, n2) - 1;
    const pValue = Math.abs(t) > 10 ? 0.0001 : Math.abs(t) > 5 ? 0.001 : Math.abs(t) > 3 ? 0.01 : Math.abs(t) > 2 ? 0.05 : 0.1;
    
    return { t, pValue };
}

// =============================================================================
// MAIN ANALYSIS
// =============================================================================

console.log('Extracting features from all samples...');
const startTime = Date.now();

const humanSamples = samples.filter(s => s.label === 'human');
const aiSamples = samples.filter(s => s.label === 'ai');

console.log(`Processing ${humanSamples.length} human samples...`);
const humanFeatures = humanSamples.map((s, i) => {
    if (i % 20000 === 0) process.stdout.write(`  ${i}/${humanSamples.length}\r`);
    return extractFeatures(s.text);
});

console.log(`\nProcessing ${aiSamples.length} AI samples...`);
const aiFeatures = aiSamples.map((s, i) => {
    if (i % 20000 === 0) process.stdout.write(`  ${i}/${aiSamples.length}\r`);
    return extractFeatures(s.text);
});

console.log(`\nFeature extraction complete in ${((Date.now() - startTime) / 1000).toFixed(1)}s\n`);

// Get all feature names
const featureNames = Object.keys(humanFeatures[0]);

// Calculate statistics for each feature
console.log('=' .repeat(100));
console.log('STATISTICAL ANALYSIS (Human vs AI)');
console.log('=' .repeat(100));
console.log('Feature'.padEnd(20) + 'Human Mean'.padEnd(12) + 'AI Mean'.padEnd(12) + 'Diff%'.padEnd(10) + 'Cohen\'s d'.padEnd(12) + 'p-value'.padEnd(12) + 'Discriminative');
console.log('-'.repeat(100));

const results = [];

for (const feature of featureNames) {
    const humanVals = humanFeatures.map(f => f[feature]);
    const aiVals = aiFeatures.map(f => f[feature]);
    
    const humanMean = mean(humanVals);
    const aiMean = mean(aiVals);
    const diffPct = (humanMean > 0 || aiMean > 0) ? ((humanMean - aiMean) / Math.max(humanMean, aiMean) * 100) : 0;
    const d = cohensD(humanVals, aiVals);
    const { t, pValue } = tTest(humanVals, aiVals);
    
    // Classify discriminative power
    let power = '';
    if (Math.abs(d) > 0.8) power = '*** STRONG';
    else if (Math.abs(d) > 0.5) power = '** MEDIUM';
    else if (Math.abs(d) > 0.2) power = '* SMALL';
    else power = '  WEAK';
    
    results.push({
        feature,
        humanMean,
        aiMean,
        diffPct,
        d,
        pValue,
        power
    });
    
    console.log(
        feature.padEnd(20) +
        humanMean.toFixed(4).padEnd(12) +
        aiMean.toFixed(4).padEnd(12) +
        (diffPct >= 0 ? '+' : '') + diffPct.toFixed(1) + '%'.padEnd(6) +
        (d >= 0 ? '+' : '') + d.toFixed(3).padEnd(10) +
        (pValue < 0.001 ? '<0.001' : pValue.toFixed(3)).padEnd(12) +
        power
    );
}

// Sort by effect size
console.log('\n' + '=' .repeat(100));
console.log('TOP DISCRIMINATIVE FEATURES (sorted by |Cohen\'s d|)');
console.log('=' .repeat(100));

const sortedResults = [...results].sort((a, b) => Math.abs(b.d) - Math.abs(a.d));
for (const r of sortedResults.slice(0, 15)) {
    const dir = r.d > 0 ? 'Human>' : 'AI>';
    console.log(`${r.feature.padEnd(20)} d=${r.d.toFixed(3).padStart(7)} ${dir.padEnd(8)} ${r.power}`);
}

// =============================================================================
// OPTIMAL THRESHOLDS
// =============================================================================

console.log('\n' + '=' .repeat(100));
console.log('OPTIMAL THRESHOLDS FOR TOP FEATURES');
console.log('=' .repeat(100));

for (const r of sortedResults.slice(0, 10)) {
    const humanVals = humanFeatures.map(f => f[r.feature]);
    const aiVals = aiFeatures.map(f => f[r.feature]);
    
    // Calculate percentiles for threshold selection
    const humanP25 = percentile(humanVals, 0.25);
    const humanP50 = percentile(humanVals, 0.50);
    const humanP75 = percentile(humanVals, 0.75);
    const aiP25 = percentile(aiVals, 0.25);
    const aiP50 = percentile(aiVals, 0.50);
    const aiP75 = percentile(aiVals, 0.75);
    
    console.log(`\n${r.feature}:`);
    console.log(`  Human: P25=${humanP25.toFixed(4)}, P50=${humanP50.toFixed(4)}, P75=${humanP75.toFixed(4)}`);
    console.log(`  AI:    P25=${aiP25.toFixed(4)}, P50=${aiP50.toFixed(4)}, P75=${aiP75.toFixed(4)}`);
    
    // Suggest threshold
    if (r.d > 0) {
        // Human higher - suggest threshold above AI P75
        const threshold = (aiP75 + humanP50) / 2;
        console.log(`  → If ${r.feature} > ${threshold.toFixed(4)}, likely HUMAN`);
    } else {
        // AI higher - suggest threshold above Human P75
        const threshold = (humanP75 + aiP50) / 2;
        console.log(`  → If ${r.feature} > ${threshold.toFixed(4)}, likely AI`);
    }
}

// =============================================================================
// BY SOURCE ANALYSIS
// =============================================================================

console.log('\n' + '=' .repeat(100));
console.log('ANALYSIS BY SOURCE');
console.log('=' .repeat(100));

const sources = [...new Set(samples.map(s => s.source))];
for (const source of sources.sort()) {
    const sourceSamples = samples.filter(s => s.source === source);
    const label = sourceSamples[0].label;
    
    const sourceFeatures = sourceSamples.slice(0, 5000).map(s => extractFeatures(s.text));
    
    // Just show key stats
    const exclamRate = mean(sourceFeatures.map(f => f.exclamRate));
    const contractionRate = mean(sourceFeatures.map(f => f.contractionRate));
    const sentCV = mean(sourceFeatures.map(f => f.sentCV));
    const commaRate = mean(sourceFeatures.map(f => f.commaRate));
    
    console.log(`${source.padEnd(20)} [${label.padEnd(5)}] n=${sourceSamples.length.toString().padEnd(6)} excl=${exclamRate.toFixed(3)} contr=${contractionRate.toFixed(4)} sentCV=${sentCV.toFixed(3)} comma=${commaRate.toFixed(2)}`);
}

// Save results
const outputPath = path.join(__dirname, 'feature_statistics.json');
fs.writeFileSync(outputPath, JSON.stringify({
    metadata: {
        humanCount: humanSamples.length,
        aiCount: aiSamples.length,
        analyzedAt: new Date().toISOString()
    },
    features: results
}, null, 2));

console.log(`\nResults saved to ${outputPath}`);
