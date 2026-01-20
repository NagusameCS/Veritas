/**
 * VERITAS Benchmark Suite
 * Tests detection accuracy across different text types and AI models
 */

// Test samples - mix of human and AI-generated text
const testSamples = {
    // Human-written samples (should be classified as human)
    human: [
        {
            id: 'human_casual_1',
            label: 'human',
            text: `I honestly can't believe how much time I wasted trying to fix that stupid bug yesterday. Like, I literally spent 4 hours staring at my screen before I realized I'd misspelled a variable name. And of course it was something super obvious like 'recieve' instead of 'receive'. My coworkers are probably judging me so hard right now lol. Anyway, finally got it working at like 2am and now I'm running on zero sleep but at least the demo went well this morning.`,
            description: 'Casual conversational writing'
        },
        {
            id: 'human_academic_1',
            label: 'human',
            text: `The philosophical implications of Wittgenstein's later work remain hotly contested among contemporary scholars. While some, like Kripke, have argued for a particular interpretation of the private language argument, others maintain that this reading fundamentally misunderstands Wittgenstein's therapeutic aims. My own view - which I acknowledge many will find controversial - is that both camps have seized upon genuine insights while missing the broader point. The Investigations isn't primarily a work of argumentation at all, but an invitation to see our linguistic practices differently.`,
            description: 'Academic/philosophical writing with personal voice'
        },
        {
            id: 'human_narrative_1',
            label: 'human',
            text: `The door creaked open and I froze. My heart was pounding so loud I was sure whoever was on the other side could hear it. I shouldn't have come here. Mom always said curiosity killed the cat, and here I was, being the dumbest cat in the whole damn neighborhood. The flashlight in my hand was shaking - or maybe that was just my hands. Hard to tell when you're terrified. I took a breath. Then another. Okay. I could do this. Probably.`,
            description: 'Narrative/creative writing with internal monologue'
        },
        {
            id: 'human_technical_1',
            label: 'human',
            text: `So I've been debugging this race condition for the past week and I think I finally figured it out. The issue is that our message queue consumer is pulling messages faster than the database can handle the writes, but only under specific conditions when the connection pool gets exhausted. What's really annoying is that this only happens in prod - our staging environment doesn't have enough traffic to trigger it. My fix involves adding a semaphore to throttle the consumer, but I'm not 100% sure it won't cause other issues.`,
            description: 'Technical blog/discussion'
        },
        {
            id: 'human_email_1',
            label: 'human',
            text: `Hi Sarah,

Thanks for sending over those reports yesterday - I've had a chance to review them and honestly, they're better than I expected given how rushed we were. A few things I noticed:

- The Q3 projections seem a bit optimistic? Maybe worth double-checking with finance
- That typo on page 7 (it says "pubic" instead of "public" - oops!)
- Love the new charts though, way clearer than what we had before

Can we chat tomorrow afternoon? I'm swamped in the morning but should be free after 2pm.

Best,
Mike`,
            description: 'Business email with informal tone'
        }
    ],

    // AI-generated samples (should be classified as AI)
    ai: [
        {
            id: 'ai_gpt_essay_1',
            label: 'ai',
            model: 'GPT-4',
            text: `The Impact of Climate Change on Global Agriculture

Climate change represents one of the most significant challenges facing global agriculture in the twenty-first century. As temperatures continue to rise and weather patterns become increasingly unpredictable, farmers around the world are confronting unprecedented difficulties in maintaining crop yields and ensuring food security.

Firstly, rising temperatures directly affect crop growth cycles. Many staple crops, such as wheat, rice, and corn, have optimal temperature ranges for growth. When temperatures exceed these ranges, photosynthesis becomes less efficient, leading to reduced yields. Furthermore, heat stress during critical growth periods can result in substantial crop losses.

Secondly, changing precipitation patterns pose significant challenges. Some regions are experiencing increased drought conditions, while others face more frequent and intense flooding events. Both extremes can devastate agricultural production. Drought conditions lead to water stress in plants, while excessive rainfall can damage crops and promote the spread of fungal diseases.

In conclusion, addressing the impacts of climate change on agriculture requires a multifaceted approach involving adaptation strategies, technological innovation, and international cooperation. The stakes could not be higher, as the future of global food security hangs in the balance.`,
            description: 'Typical GPT essay with structured format'
        },
        {
            id: 'ai_gemini_helpful_1',
            label: 'ai',
            model: 'Gemini',
            text: `That's a great question! I'd be happy to help you understand the basics of machine learning.

Here's a quick breakdown:

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.

There are three main types of machine learning:

1. **Supervised Learning**: The algorithm learns from labeled training data and makes predictions based on that data. It's like learning with a teacher who provides the correct answers.

2. **Unsupervised Learning**: The algorithm works with unlabeled data and must find patterns and relationships on its own. It's more like self-study.

3. **Reinforcement Learning**: The algorithm learns by interacting with an environment and receiving rewards or penalties for its actions.

I hope this helps! Feel free to ask if you have any more questions about machine learning or related topics. I'm here to assist you in any way I can.`,
            description: 'Gemini-style helpful response'
        },
        {
            id: 'ai_claude_analysis_1',
            label: 'ai',
            model: 'Claude',
            text: `Let me analyze this situation carefully.

The question of whether artificial intelligence will replace human workers is nuanced and requires examining multiple factors. There are reasonable arguments on both sides, and the reality will likely fall somewhere between the extremist positions.

On one hand, AI systems are becoming increasingly capable at tasks that previously required human intelligence. This includes not just routine cognitive work, but also creative tasks like writing, coding, and even artistic creation. The pace of advancement has accelerated dramatically, suggesting continued expansion of AI capabilities.

On the other hand, there are important limitations to consider. AI systems still struggle with genuine understanding, common sense reasoning, and handling novel situations that fall outside their training data. Additionally, many jobs involve social and emotional dimensions that current AI cannot replicate.

It's worth noting that historical technological transitions have generally created more jobs than they eliminated, though with significant displacement and adjustment costs. The key question is whether AI represents a continuation of this pattern or a fundamental discontinuity.

My assessment is that AI will substantially transform the labor market over the next two decades, eliminating some roles while creating others, and fundamentally changing the nature of many jobs that persist.`,
            description: 'Claude-style balanced analysis'
        },
        {
            id: 'ai_chatgpt_howto_1',
            label: 'ai',
            model: 'ChatGPT',
            text: `How to Make Perfect Scrambled Eggs

Scrambled eggs might seem simple, but achieving that perfect, creamy texture requires attention to detail. Here's a comprehensive guide to making restaurant-quality scrambled eggs at home.

**Ingredients:**
- 3 large eggs
- 1 tablespoon butter
- Salt and pepper to taste
- Optional: 1 tablespoon milk or cream

**Instructions:**

1. **Crack the eggs** into a bowl and whisk thoroughly until the yolks and whites are completely combined. The mixture should be uniform in color.

2. **Heat your pan** over medium-low heat. This is crucial – high heat is the enemy of creamy scrambled eggs.

3. **Add the butter** and let it melt completely, swirling to coat the pan. Don't let the butter brown.

4. **Pour in the eggs** and wait about 30 seconds before stirring. Then, using a spatula, gently push the eggs from the edges toward the center.

5. **Remove from heat** when the eggs are still slightly wet – they will continue cooking from residual heat.

6. **Season with salt and pepper** and serve immediately on a warm plate.

By following these steps, you'll achieve perfectly creamy scrambled eggs every time. The key is patience and low heat.`,
            description: 'ChatGPT how-to article'
        }
    ],

    // Edge cases that might cause false positives/negatives
    edgeCases: [
        {
            id: 'edge_esl_writer',
            label: 'human',
            text: `Yesterday I go to the market for buy vegetables. The price is very high now because of weather problem. I bought tomato, potato, and onion. My mother she tell me to also get rice but I forget. When I come home she was little angry but then she laugh. Family is like this always. Tomorrow I will go again and remember the rice.`,
            description: 'ESL/non-native speaker - should not flag as AI'
        },
        {
            id: 'edge_formal_human',
            label: 'human',
            text: `Furthermore, it is essential to note that the implementation of these policies has resulted in significant improvements across all measured parameters. Consequently, the committee recommends the continuation and expansion of the current program. Nevertheless, careful monitoring remains necessary to ensure that unintended consequences do not emerge. Thus, a quarterly review process has been established.`,
            description: 'Very formal human writing - high false positive risk'
        },
        {
            id: 'edge_instructional',
            label: 'human',
            text: `Step 1: First, you need to gather all the materials. Make sure you have everything before starting.

Step 2: Second, prepare your workspace. It should be clean and well-lit.

Step 3: Third, follow the diagram carefully. If you make a mistake, don't worry - just undo and try again.

Step 4: Finally, check your work. Compare it to the finished example in the guide.

Remember: patience is key. This might take several attempts to get right.`,
            description: 'Human instructional content - often false flagged'
        },
        {
            id: 'edge_humanized_ai',
            label: 'ai',
            text: `Okay so like, here's the thing about AI that nobody really talks about? It's not gonna take over the world or whatever. That's just... not how it works lol. I mean sure, it can do some pretty wild stuff - I've seen it write code that's honestly better than what some of my coworkers produce (don't @ me). But creativity? Real understanding? Nah. It's basically just really fancy pattern matching. Anyway that's just my take, feel free to disagree or whatever.`,
            description: 'AI text that has been humanized to appear casual'
        }
    ]
};

// Benchmark runner
async function runBenchmarks() {
    console.log('='.repeat(60));
    console.log('VERITAS Benchmark Suite');
    console.log('='.repeat(60));
    console.log('');
    
    const results = {
        human: { correct: 0, total: 0, samples: [] },
        ai: { correct: 0, total: 0, samples: [] },
        edgeCases: { correct: 0, total: 0, samples: [] }
    };
    
    // Test human samples
    console.log('Testing Human Samples...');
    console.log('-'.repeat(40));
    for (const sample of testSamples.human) {
        const result = await testSample(sample);
        results.human.total++;
        results.human.samples.push(result);
        if (result.aiProbability < 0.5) {
            results.human.correct++;
            console.log(`  ✓ ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI (CORRECT)`);
        } else {
            console.log(`  ✗ ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI (FALSE POSITIVE)`);
        }
    }
    console.log('');
    
    // Test AI samples
    console.log('Testing AI Samples...');
    console.log('-'.repeat(40));
    for (const sample of testSamples.ai) {
        const result = await testSample(sample);
        results.ai.total++;
        results.ai.samples.push(result);
        if (result.aiProbability >= 0.5) {
            results.ai.correct++;
            console.log(`  ✓ ${sample.id} (${sample.model}): ${(result.aiProbability * 100).toFixed(1)}% AI (CORRECT)`);
        } else {
            console.log(`  ✗ ${sample.id} (${sample.model}): ${(result.aiProbability * 100).toFixed(1)}% AI (FALSE NEGATIVE)`);
        }
    }
    console.log('');
    
    // Test edge cases
    console.log('Testing Edge Cases...');
    console.log('-'.repeat(40));
    for (const sample of testSamples.edgeCases) {
        const result = await testSample(sample);
        results.edgeCases.total++;
        results.edgeCases.samples.push(result);
        const isCorrect = sample.label === 'human' 
            ? result.aiProbability < 0.5 
            : result.aiProbability >= 0.5;
        if (isCorrect) {
            results.edgeCases.correct++;
            console.log(`  ✓ ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI (CORRECT - ${sample.label})`);
        } else {
            const errorType = sample.label === 'human' ? 'FALSE POSITIVE' : 'FALSE NEGATIVE';
            console.log(`  ✗ ${sample.id}: ${(result.aiProbability * 100).toFixed(1)}% AI (${errorType} - expected ${sample.label})`);
        }
        if (result.falsePositiveRisk?.risks?.length > 0) {
            console.log(`    → Risks detected: ${result.falsePositiveRisk.risks.map(r => r.type).join(', ')}`);
        }
    }
    console.log('');
    
    // Summary
    console.log('='.repeat(60));
    console.log('SUMMARY');
    console.log('='.repeat(60));
    
    const humanAccuracy = (results.human.correct / results.human.total * 100).toFixed(1);
    const aiAccuracy = (results.ai.correct / results.ai.total * 100).toFixed(1);
    const edgeAccuracy = (results.edgeCases.correct / results.edgeCases.total * 100).toFixed(1);
    const overallCorrect = results.human.correct + results.ai.correct + results.edgeCases.correct;
    const overallTotal = results.human.total + results.ai.total + results.edgeCases.total;
    const overallAccuracy = (overallCorrect / overallTotal * 100).toFixed(1);
    
    console.log(`Human samples:     ${results.human.correct}/${results.human.total} (${humanAccuracy}%)`);
    console.log(`AI samples:        ${results.ai.correct}/${results.ai.total} (${aiAccuracy}%)`);
    console.log(`Edge cases:        ${results.edgeCases.correct}/${results.edgeCases.total} (${edgeAccuracy}%)`);
    console.log('-'.repeat(40));
    console.log(`OVERALL ACCURACY:  ${overallCorrect}/${overallTotal} (${overallAccuracy}%)`);
    console.log('');
    
    // Calculate specific metrics
    const totalHuman = results.human.total + results.edgeCases.samples.filter(s => s.expectedLabel === 'human').length;
    const falsePositives = results.human.samples.filter(s => s.aiProbability >= 0.5).length + 
                          results.edgeCases.samples.filter(s => s.expectedLabel === 'human' && s.aiProbability >= 0.5).length;
    const falseNegatives = results.ai.samples.filter(s => s.aiProbability < 0.5).length +
                          results.edgeCases.samples.filter(s => s.expectedLabel === 'ai' && s.aiProbability < 0.5).length;
    
    console.log(`False Positive Rate: ${((falsePositives / totalHuman) * 100).toFixed(2)}%`);
    console.log(`False Negative Rate: ${((falseNegatives / results.ai.total) * 100).toFixed(2)}%`);
    
    return results;
}

// Simulate testing a sample (this would use the actual AnalyzerEngine in browser)
async function testSample(sample) {
    // In browser, this would be: const result = await AnalyzerEngine.analyze(sample.text);
    // For now, return mock result structure
    return {
        id: sample.id,
        expectedLabel: sample.label,
        aiProbability: 0.5, // Placeholder
        humanProbability: 0.5,
        confidence: 0.8,
        falsePositiveRisk: { risks: [] }
    };
}

// Export for browser/Node usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { testSamples, runBenchmarks };
}
if (typeof window !== 'undefined') {
    window.VeritasBenchmark = { testSamples, runBenchmarks };
}
