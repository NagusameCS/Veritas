const data = require('./authentic_samples.json');
const samples = data.samples;

const aiSources = ['Anthropic HH-RLHF', 'GPT4All', 'OpenAssistant-AI', 'Dolly', 'Alpaca'];

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
    
    return {
        contractionRate: contractions / wordCount,
        commaRate: commas / sentCount,
        exclamRate: exclamations / sentCount,
        firstPersonRate: firstPerson / wordCount,
        pastRate: pastTense / wordCount,
        sentCV: sentCV,
        avgSentLen: wordCount / sentCount
    };
}

for (const source of aiSources) {
    const sourceSamples = samples.filter(s => s.source === source);
    const sigs = sourceSamples.map(s => analyze(s.text));
    
    const avg = (arr, key) => arr.reduce((a, b) => a + b[key], 0) / arr.length;
    
    console.log(source + ' (' + sourceSamples.length + '):');
    console.log('  contractionRate:', avg(sigs, 'contractionRate').toFixed(4));
    console.log('  commaRate:', avg(sigs, 'commaRate').toFixed(2));
    console.log('  exclamRate:', avg(sigs, 'exclamRate').toFixed(4));
    console.log('  firstPerson:', avg(sigs, 'firstPersonRate').toFixed(4));
    console.log('  pastRate:', avg(sigs, 'pastRate').toFixed(4));
    console.log('  sentCV:', avg(sigs, 'sentCV').toFixed(3));
    console.log('  avgSentLen:', avg(sigs, 'avgSentLen').toFixed(1));
    console.log();
}
