const fs = require('fs');
const data = JSON.parse(fs.readFileSync('large_samples.json', 'utf8'));
const samples = data.samples;

function extractFeatures(text) {
    const words = text.split(/\s+/).filter(w => w.length > 0);
    const wordCount = words.length || 1;
    const sents = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const sentCount = sents.length || 1;
    
    const sentLengths = sents.map(s => s.trim().split(/\s+/).length);
    const avgLen = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length || 0;
    const variance = sentLengths.reduce((sum, len) => sum + Math.pow(len - avgLen, 2), 0) / sentLengths.length || 0;
    
    return {
        sentCV: avgLen > 0 ? Math.sqrt(variance) / avgLen : 0,
        pastTenseRate: (text.match(/\b\w+ed\b/gi) || []).length / wordCount,
        secondPersonRate: (text.match(/\b(you|your|yours|yourself)\b/gi) || []).length / wordCount,
        commaRate: (text.match(/,/g) || []).length / sentCount,
        contractionRate: (text.match(/'(s|t|re|ve|ll|d|m)\b/gi) || []).length / wordCount,
        avgSentLen: sentCount > 0 ? wordCount / sentCount : wordCount,
        // AI assistant patterns
        helpfulPhrases: (text.match(/\b(here is|here are|here's|hope this helps|feel free|let me know|happy to help|I'd be glad|I can help|I'll|I would)\b/gi) || []).length,
        instructionalMarkers: (text.match(/\b(first|second|third|step|to do this|you can|you should|make sure|note that|keep in mind|remember|important to|be sure to)\b/gi) || []).length,
        // Formal news patterns
        quotedSpeech: (text.match(/"[^"]+"/g) || []).length,
        properNouns: (text.match(/[.!?\s][A-Z][a-z]+/g) || []).length / wordCount,
        thirdPerson: (text.match(/\b(he|she|they|his|her|their|him|them)\b/gi) || []).length / wordCount
    };
}

// Compare problematic sources
const problematic = {
    'News': samples.filter(s => s.source === 'News').slice(0, 2000),
    'C4': samples.filter(s => s.source === 'C4').slice(0, 2000),
    'Anthropic-RLHF': samples.filter(s => s.source === 'Anthropic-RLHF').slice(0, 2000),
    'UltraChat': samples.filter(s => s.source === 'UltraChat').slice(0, 2000),
};

console.log('=== COMPARING PROBLEMATIC SOURCES ===\n');

for (const [source, sourceSamples] of Object.entries(problematic)) {
    const features = sourceSamples.map(s => extractFeatures(s.text));
    const avg = (arr, key) => arr.reduce((a, b) => a + b[key], 0) / arr.length;
    
    console.log(source + ' [' + sourceSamples[0].label + ']:');
    console.log('  sentCV:', avg(features, 'sentCV').toFixed(3));
    console.log('  pastTenseRate:', avg(features, 'pastTenseRate').toFixed(4));
    console.log('  secondPersonRate:', avg(features, 'secondPersonRate').toFixed(4));
    console.log('  commaRate:', avg(features, 'commaRate').toFixed(2));
    console.log('  contractionRate:', avg(features, 'contractionRate').toFixed(4));
    console.log('  avgSentLen:', avg(features, 'avgSentLen').toFixed(1));
    console.log('  helpfulPhrases:', avg(features, 'helpfulPhrases').toFixed(2));
    console.log('  instructionalMarkers:', avg(features, 'instructionalMarkers').toFixed(2));
    console.log('  quotedSpeech:', avg(features, 'quotedSpeech').toFixed(2));
    console.log('  properNouns:', avg(features, 'properNouns').toFixed(4));
    console.log('  thirdPerson:', avg(features, 'thirdPerson').toFixed(4));
    console.log();
}
