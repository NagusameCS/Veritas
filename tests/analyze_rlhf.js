const data = require('./authentic_samples.json');
const samples = data.samples;

// Compare Anthropic RLHF to casual human sources
const rlhfSamples = samples.filter(s => s.source === 'Anthropic HH-RLHF');
const casualHuman = samples.filter(s => ['WritingPrompts', 'IMDB', 'Yelp', 'Amazon'].includes(s.source));

function analyze(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    // Looking for patterns that differentiate RLHF from casual human
    return {
        // Helpful language patterns
        helpful: (text.match(/\b(help|helpful|hope this helps|let me know|feel free|happy to|glad to|I'd be|I would be|you can|you could|you might|you may)\b/gi) || []).length,
        
        // Instructional patterns
        instructional: (text.match(/\b(first|second|third|next|then|finally|step|follow|make sure|remember to|note that|keep in mind)\b/gi) || []).length,
        
        // Qualifiers/hedging
        qualifiers: (text.match(/\b(generally|typically|usually|often|sometimes|depending|varies|various|several|multiple)\b/gi) || []).length,
        
        // "Here's/Here are" patterns
        hereIs: (text.match(/\bhere('s| is| are)\b/gi) || []).length,
        
        // Question answering patterns  
        qaPatterns: (text.match(/\b(the answer|to answer|in answer|short answer|simply put|in short|basically|essentially)\b/gi) || []).length,
        
        // Assistant-like phrases
        assistantPhrases: (text.match(/\b(I hope|I can|I will|I would|I'd|let me|allow me|please|thank you)\b/gi) || []).length,
        
        // Word count (RLHF tends to be thorough)
        wordCount: wordCount,
        
        // Sentence count
        sentCount: sentCount,
        
        // Contains numbered steps
        numberedSteps: (text.match(/\b[1-9]\./g) || []).length
    };
}

const rlhfSigs = rlhfSamples.map(s => analyze(s.text));
const humanSigs = casualHuman.map(s => analyze(s.text));

const avg = (arr, key) => arr.reduce((a, b) => a + b[key], 0) / arr.length;

console.log('\n=== ANTHROPIC RLHF vs CASUAL HUMAN ===');
console.log('Feature          | RLHF     | Human    | Diff');
console.log('-'.repeat(55));

const features = Object.keys(rlhfSigs[0]);

for (const f of features) {
    const r = avg(rlhfSigs, f);
    const h = avg(humanSigs, f);
    const diffPct = h > 0 || r > 0 ? ((h - r) / Math.max(h, r) * 100) : 0;
    console.log(`${f.padEnd(16)} | ${r.toFixed(2).padEnd(8)} | ${h.toFixed(2).padEnd(8)} | ${diffPct.toFixed(1)}%`);
}

// Also show some example RLHF text
console.log('\n=== SAMPLE RLHF TEXTS ===');
for (let i = 0; i < 3; i++) {
    console.log('\n--- Sample ' + (i+1) + ' ---');
    console.log(rlhfSamples[i].text.substring(0, 300) + '...');
}
