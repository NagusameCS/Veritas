const fs = require('fs');

// Load data
const raw = fs.readFileSync('large_samples.json', 'utf8');
let data;
try {
    data = JSON.parse(raw);
} catch (e) {
    // Try line-by-line parsing
    data = raw.trim().split('\n').map(line => JSON.parse(line));
}

// Handle different structures
let samples = Array.isArray(data) ? data : (data.samples || []);
console.log('Total samples:', samples.length);

// Get problematic AI sources
const gpt4all = samples.filter(s => s.source === 'GPT4All');
const dolly = samples.filter(s => s.source === 'Dolly');
const ultrachat = samples.filter(s => s.source === 'UltraChat');
const anthropic = samples.filter(s => s.source === 'Anthropic-RLHF');

console.log('\n=== GPT4All samples (57% acc - BAD) ===');
console.log(`Count: ${gpt4all.length}`);
gpt4all.slice(0, 3).forEach((s, i) => {
    console.log(`\nSample ${i+1}:\n${s.text.substring(0, 400)}...\n`);
});

console.log('\n=== Dolly samples (63% acc - BAD) ===');
console.log(`Count: ${dolly.length}`);
dolly.slice(0, 3).forEach((s, i) => {
    console.log(`\nSample ${i+1}:\n${s.text.substring(0, 400)}...\n`);
});

console.log('\n=== Anthropic-RLHF samples (66% acc) ===');
console.log(`Count: ${anthropic.length}`);
anthropic.slice(0, 3).forEach((s, i) => {
    console.log(`\nSample ${i+1}:\n${s.text.substring(0, 400)}...\n`);
});
