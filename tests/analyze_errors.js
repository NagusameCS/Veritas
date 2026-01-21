/**
 * Deep analysis of misclassified samples to find better features
 */

const fs = require('fs');

// Load data
let data = JSON.parse(fs.readFileSync('large_samples.json', 'utf8'));
let samples = Array.isArray(data) ? data : (data.samples || []);
console.log(`Loaded ${samples.length} samples`);

// Feature extraction (same as benchmark)
function extractFeatures(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sentences.length || 1;
    
    const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
    const avgSentLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length || 0;
    const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - avgSentLen, 2), 0) / sentLengths.length || 0;
    const sentCV = avgSentLen > 0 ? Math.sqrt(variance) / avgSentLen : 0;
    
    const properNounMatches = text.match(/(?<![.!?]\s)[A-Z][a-z]+/g) || [];
    const properNouns = properNounMatches.length / wordCount;
    const helpfulPhrases = (text.match(/\b(here is|here are|feel free|I hope this helps|let me|I can help|I'd be happy|happy to help|sure thing|great question|good question)\b/gi) || []).length;
    const instructionalMarkers = (text.match(/\b(first,|second,|third,|step \d|for example|such as|in order to|make sure|keep in mind|note that|remember that|it's important to|consider the)\b/gi) || []).length;
    const quotedSpeech = (text.match(/"[^"]{10,}"/g) || []).length;
    const thirdPerson = (text.match(/\b(he|she|they|him|her|them|his|hers|their|theirs)\b/gi) || []).length / wordCount;
    
    return {
        pastTenseRate: (text.match(/\b\w+ed\b/gi) || []).length / wordCount,
        sentStartConj: (text.match(/[.!?]\s+(But|And|So|Or|Yet)\s/gi) || []).length,
        sentCV: sentCV,
        secondPersonRate: (text.match(/\b(you|your|yours|yourself|yourselves)\b/gi) || []).length / wordCount,
        firstPersonRate: (text.match(/\b(I|me|my|mine|myself|we|us|our|ours|ourselves)\b/gi) || []).length / wordCount,
        discourseMarkers: (text.match(/\b(however|therefore|furthermore|moreover|additionally|consequently|nevertheless|thus|hence|accordingly)\b/gi) || []).length / wordCount,
        canBeRate: (text.match(/\b(can|could|may|might) be\b/gi) || []).length / sentCount,
        commaRate: (text.match(/,/g) || []).length / sentCount,
        exclamRate: (text.match(/!/g) || []).length / sentCount,
        contractionRate: (text.match(/\b(i'm|i've|i'll|i'd|you're|you've|you'll|you'd|he's|she's|it's|we're|we've|we'll|we'd|they're|they've|they'll|they'd|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|doesn't|don't|didn't|won't|wouldn't|can't|couldn't|shouldn't|mustn't|let's|that's|who's|what's|where's|when's|there's|here's|ain't)\b/gi) || []).length / wordCount,
        casualMarkers: (text.match(/\b(lol|lmao|haha|omg|wtf|idk|tbh|imo|ngl|kinda|gonna|wanna|gotta|yeah|nah|yep)\b/gi) || []).length,
        itIsImportant: (text.match(/\bit is (important|essential|crucial|vital|necessary|worth noting|worth mentioning)\b/gi) || []).length,
        avgSentLen: avgSentLen,
        properNouns: properNouns,
        helpfulPhrases: helpfulPhrases,
        instructionalMarkers: instructionalMarkers,
        quotedSpeech: quotedSpeech,
        thirdPerson: thirdPerson,
        hasHtmlTags: /<[a-z]+>/i.test(text),
        hasCodeBlocks: /```|<code>|<pre>/.test(text),
        textLength: text.length,
        refusalPattern: /I (don't|can't|won't|cannot|shouldn't) (want to|agree|think|help with|assist|provide)/i.test(text),
        
        // NEW features to explore
        questionMarks: (text.match(/\?/g) || []).length / sentCount,
        colonRate: (text.match(/:/g) || []).length / sentCount,
        semicolonRate: (text.match(/;/g) || []).length / sentCount,
        numbersRate: (text.match(/\b\d+\b/g) || []).length / wordCount,
        allCaps: (text.match(/\b[A-Z]{3,}\b/g) || []).length,
        repeatWords: (() => {
            const wordFreq = {};
            words.forEach(w => { wordFreq[w.toLowerCase()] = (wordFreq[w.toLowerCase()] || 0) + 1; });
            return Object.values(wordFreq).filter(v => v > 2).length / wordCount;
        })(),
        avgWordLen: words.reduce((sum, w) => sum + w.length, 0) / wordCount,
        uniqueWords: new Set(words.map(w => w.toLowerCase())).size / wordCount,
        paragraphs: text.split(/\n\n+/).length,
        listItems: (text.match(/^\s*[\-\*\d]\.\s/gm) || []).length,
        emphasisMarkers: (text.match(/\*\*|__|\*|_/g) || []).length,
        hedging: (text.match(/\b(probably|perhaps|maybe|might|could|possibly|likely|seems|appears|tend to|in general)\b/gi) || []).length / wordCount,
        certainty: (text.match(/\b(definitely|certainly|clearly|obviously|always|never|must|absolutely|undoubtedly)\b/gi) || []).length / wordCount
    };
}

// Compare sources
function compareSourceFeatures() {
    const sources = {};
    
    samples.forEach(s => {
        if (!sources[s.source]) sources[s.source] = { features: [], label: s.label };
        if (sources[s.source].features.length < 1000) {
            sources[s.source].features.push(extractFeatures(s.text));
        }
    });
    
    // Calculate averages
    const avgFeatures = {};
    for (const [source, data] of Object.entries(sources)) {
        const avg = {};
        const keys = Object.keys(data.features[0] || {});
        for (const key of keys) {
            if (typeof data.features[0][key] === 'boolean') {
                avg[key] = data.features.filter(f => f[key]).length / data.features.length;
            } else if (typeof data.features[0][key] === 'number') {
                avg[key] = data.features.reduce((sum, f) => sum + f[key], 0) / data.features.length;
            }
        }
        avgFeatures[source] = { ...avg, label: data.label };
    }
    
    // Find features with biggest differences between problematic sources
    console.log('\n=== FEATURE COMPARISON ===\n');
    
    // C4 vs Anthropic (both often misclassified)
    const c4 = avgFeatures['C4'];
    const news = avgFeatures['News'];
    const anthropic = avgFeatures['Anthropic-RLHF'];
    const gpt4all = avgFeatures['GPT4All'];
    const dolly = avgFeatures['Dolly'];
    const writingPrompts = avgFeatures['WritingPrompts'];
    const imdb = avgFeatures['IMDB'];
    
    const featureKeys = Object.keys(c4 || {}).filter(k => k !== 'label' && typeof c4[k] === 'number');
    
    console.log('Feature                  | C4(H) | News(H) | WriPro(H) | IMDB(H) | Anthro(AI) | GPT4All(AI) | Dolly(AI)');
    console.log('-'.repeat(110));
    
    for (const key of featureKeys) {
        const c4Val = c4?.[key]?.toFixed(4) || 'N/A';
        const newsVal = news?.[key]?.toFixed(4) || 'N/A';
        const wpVal = writingPrompts?.[key]?.toFixed(4) || 'N/A';
        const imdbVal = imdb?.[key]?.toFixed(4) || 'N/A';
        const anthVal = anthropic?.[key]?.toFixed(4) || 'N/A';
        const gpt4Val = gpt4all?.[key]?.toFixed(4) || 'N/A';
        const dollyVal = dolly?.[key]?.toFixed(4) || 'N/A';
        
        console.log(`${key.padEnd(24)} | ${c4Val.padStart(5)} | ${newsVal.padStart(7)} | ${wpVal.padStart(9)} | ${imdbVal.padStart(7)} | ${anthVal.padStart(10)} | ${gpt4Val.padStart(11)} | ${dollyVal.padStart(9)}`);
    }
    
    // Find best discriminating features for problematic sources
    console.log('\n=== KEY DIFFERENCES ===\n');
    
    // C4 vs AI average
    const aiAvg = {};
    const aiSources = Object.entries(avgFeatures).filter(([k, v]) => v.label === 'ai');
    for (const key of featureKeys) {
        aiAvg[key] = aiSources.reduce((sum, [k, v]) => sum + v[key], 0) / aiSources.length;
    }
    
    console.log('Features where C4/News differ from AI:');
    const diffs = featureKeys.map(key => ({
        key,
        c4Diff: (c4?.[key] || 0) - aiAvg[key],
        newsDiff: (news?.[key] || 0) - aiAvg[key]
    })).sort((a, b) => Math.abs(b.c4Diff) - Math.abs(a.c4Diff));
    
    diffs.slice(0, 15).forEach(d => {
        const c4Dir = d.c4Diff > 0 ? 'HIGHER' : 'lower';
        const newsDir = d.newsDiff > 0 ? 'HIGHER' : 'lower';
        console.log(`  ${d.key}: C4 ${c4Dir} by ${Math.abs(d.c4Diff).toFixed(4)}, News ${newsDir} by ${Math.abs(d.newsDiff).toFixed(4)}`);
    });
}

compareSourceFeatures();
