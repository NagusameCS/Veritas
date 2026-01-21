const fs = require('fs');

// Load data
const raw = fs.readFileSync('large_samples.json', 'utf8');
let data = JSON.parse(raw);
let samples = Array.isArray(data) ? data : (data.samples || []);

// Get problematic sources
const c4 = samples.filter(s => s.source === 'C4').slice(0, 500);
const news = samples.filter(s => s.source === 'News').slice(0, 500);
const openwebtext = samples.filter(s => s.source === 'OpenWebText').slice(0, 500);

// Good human sources for comparison
const writingprompts = samples.filter(s => s.source === 'WritingPrompts').slice(0, 500);
const imdb = samples.filter(s => s.source === 'IMDB').slice(0, 500);

// AI sources
const anthropic = samples.filter(s => s.source === 'Anthropic-RLHF').slice(0, 500);
const ultrachat = samples.filter(s => s.source === 'UltraChat').slice(0, 500);

function extractFeatures(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    return {
        instructionalMarkers: (text.match(/\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|it's important to|consider the)\b/gi) || []).length,
        hasHtmlTags: /<[a-z]+>/i.test(text),
        hasCodeBlocks: /```|<code>|<pre>/.test(text),
        helpfulPhrases: (text.match(/\b(here is|here are|feel free|I hope this helps|let me|I can help|I'd be happy|happy to help|sure thing|great question|good question)\b/gi) || []).length,
        secondPersonRate: (text.match(/\b(you|your|yours|yourself|yourselves)\b/gi) || []).length / wordCount,
        discourseMarkers: (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length / wordCount,
        commaRate: (text.match(/,/g) || []).length / sentCount,
        properNouns: (text.match(/(?<![.!?]\s)[A-Z][a-z]+/g) || []).length / wordCount,
        firstPersonRate: (text.match(/\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b/gi) || []).length / wordCount,
        pastTenseRate: (text.match(/\b\w+ed\b/gi) || []).length / wordCount,
        avgWordLen: words.reduce((sum, w) => sum + w.length, 0) / wordCount,
        sentStartConj: (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length,
        contractionRate: (text.match(/\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b/gi) || []).length / wordCount
    };
}

function analyzeSource(arr, name) {
    const features = arr.map(s => extractFeatures(s.text));
    const avg = (key) => features.reduce((sum, f) => sum + (typeof f[key] === 'boolean' ? (f[key] ? 1 : 0) : f[key]), 0) / features.length;
    
    return {
        name,
        count: arr.length,
        instructionalMarkers: avg('instructionalMarkers').toFixed(2),
        hasHtmlTags: (avg('hasHtmlTags') * 100).toFixed(1) + '%',
        hasCodeBlocks: (avg('hasCodeBlocks') * 100).toFixed(1) + '%',
        helpfulPhrases: avg('helpfulPhrases').toFixed(2),
        secondPersonRate: avg('secondPersonRate').toFixed(4),
        discourseMarkers: avg('discourseMarkers').toFixed(5),
        commaRate: avg('commaRate').toFixed(2),
        properNouns: avg('properNouns').toFixed(4),
        firstPersonRate: avg('firstPersonRate').toFixed(4),
        pastTenseRate: avg('pastTenseRate').toFixed(4),
        avgWordLen: avg('avgWordLen').toFixed(2),
        sentStartConj: avg('sentStartConj').toFixed(2),
        contractionRate: avg('contractionRate').toFixed(4)
    };
}

console.log('=== FEATURE ANALYSIS BY SOURCE ===\n');

const sources = [
    { arr: c4, name: 'C4 [HUMAN-FP]' },
    { arr: news, name: 'News [HUMAN-FP]' },
    { arr: openwebtext, name: 'OpenWebText [HUMAN]' },
    { arr: writingprompts, name: 'WritingPrompts [HUMAN-GOOD]' },
    { arr: imdb, name: 'IMDB [HUMAN-GOOD]' },
    { arr: anthropic, name: 'Anthropic [AI-FN]' },
    { arr: ultrachat, name: 'UltraChat [AI]' }
];

const results = sources.map(s => analyzeSource(s.arr, s.name));

// Print comparison table
console.log('Feature              | C4      | News    | OWT     | WP      | IMDB    | Anthr   | Ultra');
console.log('---------------------|---------|---------|---------|---------|---------|---------|--------');

const keys = ['instructionalMarkers', 'helpfulPhrases', 'secondPersonRate', 'discourseMarkers', 
              'commaRate', 'properNouns', 'firstPersonRate', 'pastTenseRate', 'sentStartConj', 'contractionRate'];

keys.forEach(key => {
    const row = results.map(r => r[key].toString().padStart(7));
    console.log(`${key.padEnd(20)} | ${row.join(' | ')}`);
});

console.log('\n=== KEY DISTINGUISHING FEATURES ===\n');

// Find features where C4/News differ from good human sources
console.log('C4 vs WritingPrompts (why C4 looks like AI):');
const c4Features = analyzeSource(c4, 'C4');
const wpFeatures = analyzeSource(writingprompts, 'WritingPrompts');

Object.keys(c4Features).forEach(key => {
    if (key === 'name' || key === 'count') return;
    const c4Val = parseFloat(c4Features[key]);
    const wpVal = parseFloat(wpFeatures[key]);
    if (Math.abs(c4Val - wpVal) > 0.1 * Math.max(c4Val, wpVal, 0.01)) {
        console.log(`  ${key}: C4=${c4Features[key]}, WP=${wpFeatures[key]}`);
    }
});
