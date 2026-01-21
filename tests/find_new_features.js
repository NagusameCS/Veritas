const data = require('./authentic_samples.json');
const samples = data.samples;

function analyze(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    // New potential features to explore
    
    // 1. Colon usage (AI often uses more structured formatting)
    const colons = (text.match(/:/g) || []).length;
    
    // 2. Parenthetical asides (human conversational style)
    const parens = (text.match(/\([^)]+\)/g) || []).length;
    
    // 3. "In order to" - formal AI phrasing
    const inOrderTo = (text.match(/\bin order to\b/gi) || []).length;
    
    // 4. "It is important" - hedging formal AI
    const itIsImportant = (text.match(/\bit is (important|essential|crucial|vital|necessary)\b/gi) || []).length;
    
    // 5. "There are/is" - expletive construction (AI uses more)
    const thereIs = (text.match(/\bthere (is|are|was|were)\b/gi) || []).length;
    
    // 6. Dash usage (em-dash, en-dash) - human uses for asides
    const dashes = (text.match(/[-–—]{1,2}/g) || []).length;
    
    // 7. Starting sentence with conjunctions (But, And, So) - casual human
    const sentenceStartConj = (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length;
    
    // 8. Adverb overuse (particularly, especially, specifically) - AI 
    const adverbs = (text.match(/\b\w+ly\b/gi) || []).length;
    
    // 9. Sentence starting with "This" (AI often uses for reference)
    const thisStart = (text.match(/[.!?]\s+This\s/gi) || []).length;
    
    // 10. Questions as rhetorical devices
    const questions = (text.match(/\?/g) || []).length;
    
    // 11. Semicolons (formal writing indicator)
    const semicolons = (text.match(/;/g) || []).length;
    
    // 12. Text in quotes
    const quotes = (text.match(/[""][^""]+[""]|'[^']+'/g) || []).length;
    
    // 13. Numbers and stats
    const numbers = (text.match(/\b\d+(?:\.\d+)?%?|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b/gi) || []).length;
    
    // 14. "Can be" construction (AI passive voice)
    const canBe = (text.match(/\bcan be\b|\bcould be\b|\bmay be\b/gi) || []).length;
    
    return {
        colonRate: colons / sentCount,
        parenRate: parens / sentCount,
        inOrderTo: inOrderTo,
        itIsImportant: itIsImportant,
        thereIsRate: thereIs / sentCount,
        dashRate: dashes / sentCount,
        sentStartConj: sentenceStartConj,
        adverbRate: adverbs / wordCount,
        thisStartRate: thisStart / sentCount,
        questionRate: questions / sentCount,
        semicolonRate: semicolons / sentCount,
        quoteRate: quotes / sentCount,
        numberRate: numbers / wordCount,
        canBeRate: canBe / sentCount
    };
}

const humanSamples = samples.filter(s => s.label === 'human');
const aiSamples = samples.filter(s => s.label === 'ai');

const humanSigs = humanSamples.map(s => analyze(s.text));
const aiSigs = aiSamples.map(s => analyze(s.text));

const avg = (arr, key) => arr.reduce((a, b) => a + b[key], 0) / arr.length;

console.log('\n=== NEW FEATURE EXPLORATION ===');
console.log('Feature          | Human    | AI       | Diff    | Potential');
console.log('-'.repeat(65));

const features = Object.keys(humanSigs[0]);

for (const f of features) {
    const h = avg(humanSigs, f);
    const a = avg(aiSigs, f);
    const diffPct = h > 0 || a > 0 ? ((h - a) / Math.max(h, a) * 100) : 0;
    const potential = Math.abs(diffPct) > 30 ? '***' : Math.abs(diffPct) > 20 ? '**' : Math.abs(diffPct) > 10 ? '*' : '';
    console.log(`${f.padEnd(16)} | ${h.toFixed(4).padEnd(8)} | ${a.toFixed(4).padEnd(8)} | ${diffPct.toFixed(1).padStart(6)}% | ${potential}`);
}
