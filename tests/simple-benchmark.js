#!/usr/bin/env node
const testSamples = {
    human: [
        { id: 'casual', label: 'human', text: "I honestly can't believe how much time I wasted trying to fix that stupid bug yesterday. Like, I literally spent 4 hours staring at my screen before I realized I'd misspelled a variable name. lol" },
        { id: 'narrative', label: 'human', text: "The door creaked open and I froze. My heart was pounding so loud I was sure whoever was on the other side could hear it. I shouldn't have come here. Mom always said curiosity killed the cat." },
        { id: 'academic', label: 'human', text: "The philosophical implications of Wittgenstein's later work remain hotly contested among contemporary scholars. My own view - which I acknowledge many will find controversial - is that both camps have seized upon genuine insights." },
        { id: 'email', label: 'human', text: "Hi Sarah, Thanks for the reports - honestly, they're better than I expected. A few things: that typo on page 7 (says 'pubic' instead of 'public' - oops!). Can we chat tomorrow? Best, Mike" }
    ],
    ai: [
        { id: 'gpt_essay', label: 'ai', model: 'GPT-4', text: "Climate change represents one of the most significant challenges facing global agriculture. Firstly, rising temperatures directly affect crop growth. Furthermore, heat stress during critical periods can result in substantial losses. In conclusion, addressing these impacts requires a multifaceted approach." },
        { id: 'gemini', label: 'ai', model: 'Gemini', text: "That's a great question! I'd be happy to help you understand machine learning. Here's a breakdown: Machine learning enables systems to learn from experience. There are three main types: 1. Supervised 2. Unsupervised 3. Reinforcement. Hope this helps! Feel free to ask more." },
        { id: 'claude', label: 'ai', model: 'Claude', text: "Let me analyze this carefully. The question of whether AI will replace workers is nuanced. On one hand, AI systems are becoming increasingly capable. On the other hand, there are important limitations. It's worth noting that historical transitions generally created more jobs." },
        { id: 'chatgpt', label: 'ai', model: 'ChatGPT', text: "How to Make Perfect Scrambled Eggs. Step 1: Crack eggs into a bowl and whisk thoroughly. Step 2: Heat your pan over medium-low. Step 3: Add butter and let it melt. Step 4: Pour in the eggs. By following these steps, you'll achieve perfect results every time." }
    ],
    edge: [
        { id: 'esl', label: 'human', text: "Yesterday I go to market for buy vegetables. The price is very high because of weather problem. I bought tomato and onion. My mother she tell me get rice but I forget. When I come home she was angry but then laugh." },
        { id: 'formal', label: 'human', text: "Furthermore, it is essential to note that policy implementation has resulted in significant improvements. Consequently, the committee recommends continuation. Nevertheless, careful monitoring remains necessary. Thus, a quarterly review has been established." },
        { id: 'humanized', label: 'ai', text: "Okay so like, here's the thing about AI that nobody talks about? It's not gonna take over the world lol. That's just not how it works. I mean sure, it can do wild stuff. But creativity? Real understanding? Nah. Just fancy pattern matching." }
    ]
};

function analyze(text) {
    const lower = text.toLowerCase();
    const words = lower.split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
    let score = 0, signals = [];
    
    // AI vocabulary
    const aiWords = ['delve','multifaceted','tapestry','comprehensive','crucial','furthermore','moreover','consequently','nevertheless','facilitate','leverage'];
    const aiCount = aiWords.filter(w => lower.includes(w)).length;
    if (aiCount >= 2) { score += 0.15; signals.push('AI vocab:'+aiCount); }
    
    // AI phrases
    const aiPhrases = ['in conclusion','firstly','step 1','step 2','great question','happy to help','hope this helps','feel free','here\'s a breakdown','it is essential','on one hand','on the other hand'];
    const phraseCount = aiPhrases.filter(p => lower.includes(p)).length;
    if (phraseCount >= 2) { score += 0.2; signals.push('AI phrases:'+phraseCount); }
    
    // Sentence uniformity
    const lens = sentences.map(s => s.split(/\s+/).length);
    const mean = lens.reduce((a,b)=>a+b,0)/lens.length||0;
    const variance = lens.reduce((s,x)=>s+Math.pow(x-mean,2),0)/lens.length;
    const cv = mean > 0 ? Math.sqrt(variance)/mean : 0.5;
    if (cv < 0.3 && sentences.length > 3) { score += 0.1; signals.push('uniform:'+cv.toFixed(2)); }
    
    // Human markers
    const humanWords = ['lol','honestly','literally','damn','stupid','oops','like,'];
    const humanCount = humanWords.filter(w => lower.includes(w)).length;
    if (humanCount >= 1) { score -= 0.2; signals.push('human:'+humanCount); }
    
    // Helper tone
    const helpers = ['happy to help','great question','hope this helps','feel free'];
    const helperCount = helpers.filter(p => lower.includes(p)).length;
    if (helperCount >= 2) { score += 0.15; signals.push('helper:'+helperCount); }
    
    return { prob: Math.max(0, Math.min(1, 0.45 + score)), signals };
}

console.log('=' .repeat(50));
console.log('VERITAS Benchmark');
console.log('='.repeat(50));

let results = { human: {ok:0,n:0}, ai: {ok:0,n:0}, edge: {ok:0,n:0} };

console.log('\nHuman:');
for (const s of testSamples.human) {
    const r = analyze(s.text);
    const ok = r.prob < 0.5;
    results.human.n++; if(ok) results.human.ok++;
    console.log('  ' + (ok?'✓':'✗FP') + ' ' + s.id + ': ' + (r.prob*100).toFixed(0) + '% ' + (r.signals.join(' ')||''));
}

console.log('\nAI:');
for (const s of testSamples.ai) {
    const r = analyze(s.text);
    const ok = r.prob >= 0.5;
    results.ai.n++; if(ok) results.ai.ok++;
    console.log('  ' + (ok?'✓':'✗FN') + ' ' + s.id + '(' + s.model + '): ' + (r.prob*100).toFixed(0) + '% ' + (r.signals.join(' ')||''));
}

console.log('\nEdge:');
for (const s of testSamples.edge) {
    const r = analyze(s.text);
    const ok = s.label==='human' ? r.prob < 0.5 : r.prob >= 0.5;
    results.edge.n++; if(ok) results.edge.ok++;
    console.log('  ' + (ok?'✓':'✗') + ' ' + s.id + '(' + s.label + '): ' + (r.prob*100).toFixed(0) + '% ' + (r.signals.join(' ')||''));
}

console.log('\n' + '='.repeat(50));
const total = results.human.n + results.ai.n + results.edge.n;
const correct = results.human.ok + results.ai.ok + results.edge.ok;
console.log('Human: ' + results.human.ok + '/' + results.human.n);
console.log('AI:    ' + results.ai.ok + '/' + results.ai.n);
console.log('Edge:  ' + results.edge.ok + '/' + results.edge.n);
console.log('TOTAL: ' + correct + '/' + total + ' (' + (correct/total*100).toFixed(0) + '%)');

const fpRate = (results.human.n - results.human.ok) / results.human.n * 100;
const fnRate = (results.ai.n - results.ai.ok) / results.ai.n * 100;
console.log('FP: ' + fpRate.toFixed(0) + '% | FN: ' + fnRate.toFixed(0) + '%');

if (correct/total >= 0.85) console.log('\n✅ ML training optional');
else if (correct/total >= 0.7) console.log('\n⚠️ Consider ML calibration');
else console.log('\n❌ ML training recommended');
