const data = require('./authentic_samples.json');
const samples = data.samples;

function analyze(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    const contractions = (text.match(/\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b/gi) || []).length;
    const commas = (text.match(/,/g) || []).length;
    const exclamations = (text.match(/!/g) || []).length;
    const firstPerson = (text.match(/\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b/gi) || []).length;
    const pastTense = (text.match(/\b\w+ed\b/gi) || []).length;
    
    // Sentence length variance
    const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
    const avgLen = sentLengths.reduce((a,b) => a+b, 0) / sentLengths.length || 0;
    const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - avgLen, 2), 0) / sentLengths.length || 0;
    const sentCV = avgLen > 0 ? Math.sqrt(variance) / avgLen : 0;
    
    // Discourse markers
    const discourse = (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length;
    
    return {
        contractionRate: contractions / wordCount,
        commaRate: commas / sentCount,
        exclamRate: exclamations / sentCount,
        firstPersonRate: firstPerson / wordCount,
        pastRate: pastTense / wordCount,
        sentCV: sentCV,
        avgSentLen: wordCount / sentCount,
        discourseRate: discourse / wordCount
    };
}

const humanSamples = samples.filter(s => s.label === 'human');
const aiSamples = samples.filter(s => s.label === 'ai');

const humanSigs = humanSamples.map(s => analyze(s.text));
const aiSigs = aiSamples.map(s => analyze(s.text));

const avg = (arr, key) => arr.reduce((a, b) => a + b[key], 0) / arr.length;

console.log('\n=== OVERALL HUMAN vs AI ===');
console.log('Feature          | Human    | AI       | Diff');
console.log('-'.repeat(50));

const features = ['contractionRate', 'commaRate', 'exclamRate', 'firstPersonRate', 'pastRate', 'sentCV', 'avgSentLen', 'discourseRate'];

for (const f of features) {
    const h = avg(humanSigs, f);
    const a = avg(aiSigs, f);
    const diff = ((h - a) / Math.max(h, a) * 100).toFixed(1);
    console.log(`${f.padEnd(16)} | ${h.toFixed(4).padEnd(8)} | ${a.toFixed(4).padEnd(8)} | ${diff}%`);
}
