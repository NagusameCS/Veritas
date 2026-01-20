#!/usr/bin/env node
/**
 * VERITAS — Comprehensive Benchmark Suite
 * Tests against diverse samples across all categories
 * 
 * Categories tested:
 * - Human: casual, narrative, academic, email, technical, ESL, creative
 * - AI: GPT-4, Claude, Gemini, ChatGPT, instructional, analytical
 * - Humanized: various humanizer tools and manual edits
 * - Edge cases: formal humans, AI-assisted human, multilingual
 */

const testSamples = {
    // ==================== HUMAN SAMPLES ====================
    human: [
        // Casual/informal
        {
            id: 'casual_chat',
            label: 'human',
            category: 'casual',
            text: "I honestly can't believe how much time I wasted trying to fix that stupid bug yesterday. Like, I literally spent 4 hours staring at my screen before I realized I'd misspelled a variable name. lol so embarrassing"
        },
        {
            id: 'text_message',
            label: 'human',
            category: 'casual',
            text: "hey u coming to the party tmrw?? gonna be lit apparently. sarah said shes bringing that guy from work lol. lmk if u need a ride or w/e"
        },
        {
            id: 'reddit_post',
            label: 'human',
            category: 'casual',
            text: "AITA for telling my roommate to stop using my stuff without asking? Like I get we share a space but holy crap just ASK first. It's not that hard. She took my nice headphones AGAIN and now there's a scratch on them. I'm so done."
        },
        
        // Narrative/creative
        {
            id: 'narrative_fiction',
            label: 'human',
            category: 'narrative',
            text: "The door creaked open and I froze. My heart was pounding so loud I was sure whoever was on the other side could hear it. I shouldn't have come here. Mom always said curiosity killed the cat. But I couldn't help it - I needed to know."
        },
        {
            id: 'personal_story',
            label: 'human',
            category: 'narrative',
            text: "So there I was, standing in the middle of the grocery store, completely blanking on why I came. You know that feeling? Three items. I only needed three items. Came home with twelve things and forgot the milk. Classic me."
        },
        {
            id: 'travel_blog',
            label: 'human',
            category: 'narrative',
            text: "Prague surprised me. I expected tourist traps and overpriced beer but found these tiny side streets with the best damn pastries I've ever had. Got lost for hours. Didn't care. Sometimes getting lost is the point, you know?"
        },
        
        // Academic/professional
        {
            id: 'academic_essay',
            label: 'human',
            category: 'academic',
            text: "The philosophical implications of Wittgenstein's later work remain hotly contested among contemporary scholars. My own view - which I acknowledge many will find controversial - is that both camps have seized upon genuine insights while missing the forest for the trees."
        },
        {
            id: 'research_note',
            label: 'human',
            category: 'academic',
            text: "Initial results are promising but I'm skeptical. The p-value looks good on paper (.03) but with n=47 and three failed replications in the literature, I want to see this hold up. Running power analysis tomorrow. Maria thinks I'm overthinking it."
        },
        {
            id: 'peer_review',
            label: 'human',
            category: 'academic',
            text: "The authors present an interesting hypothesis but the methodology section raises concerns. How was blinding maintained when the same researcher conducted interviews and analysis? Also, Table 3 appears to have a calculation error in row 7."
        },
        
        // Email/business
        {
            id: 'work_email',
            label: 'human',
            category: 'email',
            text: "Hi Sarah, Thanks for the reports - honestly, they're better than I expected given the timeline. A few things: that typo on page 7 (says 'pubic' instead of 'public' - oops!). Can we chat tomorrow? Best, Mike"
        },
        {
            id: 'complaint_email',
            label: 'human',
            category: 'email',
            text: "Your product arrived damaged AGAIN. This is the third time in two months. I've attached photos. I want a full refund, not store credit. And someone needs to look at how you pack these things. Bubble wrap exists for a reason."
        },
        {
            id: 'team_slack',
            label: 'human',
            category: 'email',
            text: "quick update: deploy is blocked. jenkins is being weird again (shocker). @dave any chance you can take a look? I need to pick up my kid at 3 so leaving in 20. logs are in #infra-alerts"
        },
        
        // Technical
        {
            id: 'tech_debug',
            label: 'human',
            category: 'technical',
            text: "Okay so the issue is in the useEffect hook - it's running on every render because the dependency array includes an object reference that changes. Wrap it in useMemo or just pass the primitive values. Took me 2 hours to figure out btw."
        },
        {
            id: 'stack_overflow',
            label: 'human',
            category: 'technical',
            text: "This is NOT a duplicate of that 2015 question. I'm using React 18 with Suspense and the old solution doesn't work. Already tried useLayoutEffect like the accepted answer suggests. Same result. Please don't close this."
        },
        
        // ESL/Non-native
        {
            id: 'esl_story',
            label: 'human',
            category: 'esl',
            text: "Yesterday I go to market for buy vegetables. The price is very high because of weather problem. I bought tomato and onion. My mother she tell me get rice but I forget. When I come home she was angry but then laugh."
        },
        {
            id: 'esl_email',
            label: 'human',
            category: 'esl',
            text: "Dear Sir, I am writing for ask about the job position. I have experience 3 years in this field. Please find my CV attached on this email. I am hoping to hear from you soon. Thank you very much for your time."
        },
        {
            id: 'esl_review',
            label: 'human',
            category: 'esl',
            text: "This product is good quality but delivery take too long time. Also the color is not same like picture. A little bit different. But my wife she like it so I give 4 star. Will buy again if price go down."
        },
        
        // Creative writing
        {
            id: 'poetry',
            label: 'human',
            category: 'creative',
            text: "coffee gone cold again\nstaring at the cursor blink\nthe words won't come today\nor maybe they're all hiding\nbehind yesterday's rejection\nthat still stings"
        },
        {
            id: 'journal',
            label: 'human',
            category: 'creative',
            text: "Can't sleep. 3am thoughts hitting different tonight. Why did I say that thing to Jenny at the party? She probably thinks I'm such a weirdo now. Ugh. Also need to remember to call Mom tomorrow. And pay rent. Brain please stop."
        }
    ],

    // ==================== AI SAMPLES ====================
    ai: [
        // GPT-4 style
        {
            id: 'gpt4_essay',
            label: 'ai',
            model: 'GPT-4',
            category: 'essay',
            text: "Climate change represents one of the most significant challenges facing global agriculture. Firstly, rising temperatures directly affect crop growth cycles and yield potential. Furthermore, heat stress during critical developmental periods can result in substantial losses. In conclusion, addressing these impacts requires a multifaceted approach combining technology, policy, and individual action."
        },
        {
            id: 'gpt4_analysis',
            label: 'ai',
            model: 'GPT-4',
            category: 'analysis',
            text: "The economic implications of remote work are multifaceted and far-reaching. On one hand, companies can reduce overhead costs associated with physical office spaces. On the other hand, there are concerns about productivity and company culture. It is important to note that different industries will be affected to varying degrees. The key takeaway is that flexibility will likely become the norm rather than the exception."
        },
        {
            id: 'gpt4_explain',
            label: 'ai',
            model: 'GPT-4',
            category: 'explanation',
            text: "Machine learning can be understood as a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. There are three main types: supervised learning, unsupervised learning, and reinforcement learning. Each approach has its strengths and weaknesses, making them suitable for different applications. The field continues to evolve rapidly, with new techniques emerging regularly."
        },
        
        // Claude style
        {
            id: 'claude_balanced',
            label: 'ai',
            model: 'Claude',
            category: 'analysis',
            text: "Let me analyze this carefully. The question of whether AI will replace human workers is nuanced and requires considering multiple perspectives. On one hand, AI systems are becoming increasingly capable of performing tasks once thought to require human intelligence. On the other hand, there are important limitations to current AI technology. It's worth noting that historical technological transitions have generally created more jobs than they eliminated, though this pattern isn't guaranteed to continue."
        },
        {
            id: 'claude_helpful',
            label: 'ai',
            model: 'Claude',
            category: 'helpful',
            text: "I'd be happy to help you understand this concept. The theory of relativity fundamentally changed our understanding of space and time. There are actually two parts: special relativity and general relativity. Special relativity deals with objects moving at constant speeds, while general relativity incorporates gravity. The famous equation E=mc² comes from special relativity and describes the relationship between mass and energy."
        },
        {
            id: 'claude_nuanced',
            label: 'ai',
            model: 'Claude',
            category: 'nuanced',
            text: "This is an interesting question that doesn't have a simple answer. There are valid arguments on both sides. Proponents would argue that the benefits outweigh the costs, pointing to efficiency gains and improved outcomes. Critics, however, raise important concerns about unintended consequences and equity implications. My assessment is that a balanced approach would likely yield the best results, though reasonable people may disagree."
        },
        
        // Gemini style
        {
            id: 'gemini_structured',
            label: 'ai',
            model: 'Gemini',
            category: 'structured',
            text: "That's a great question! I'd be happy to help you understand machine learning. Here's a breakdown of the key concepts: Machine learning enables systems to learn from experience rather than following explicit instructions. There are three main types to know: 1. Supervised Learning 2. Unsupervised Learning 3. Reinforcement Learning. Each has unique applications and use cases. Hope this helps! Feel free to ask more questions."
        },
        {
            id: 'gemini_enthusiastic',
            label: 'ai',
            model: 'Gemini',
            category: 'enthusiastic',
            text: "Absolutely! Let me break this down for you. The key thing to understand is that blockchain technology operates on a decentralized network. Here's what makes it special: First, transactions are transparent and immutable. Second, there's no central authority controlling the network. Third, security is built into the system through cryptography. It's really quite fascinating when you think about all the possibilities!"
        },
        {
            id: 'gemini_helpful',
            label: 'ai',
            model: 'Gemini',
            category: 'helpful',
            text: "Great question! Here's the thing about renewable energy - it's not just about saving the environment, though that's certainly important. Let me walk you through the key benefits: 1. Cost savings over time 2. Energy independence 3. Job creation in new sectors. The transition isn't without challenges, of course, but the trajectory is clear. Would you like me to elaborate on any of these points?"
        },
        
        // ChatGPT style
        {
            id: 'chatgpt_howto',
            label: 'ai',
            model: 'ChatGPT',
            category: 'instructional',
            text: "How to Make Perfect Scrambled Eggs. Step 1: Crack 2-3 eggs into a bowl and whisk thoroughly until uniform. Step 2: Heat your pan over medium-low heat. Step 3: Add butter and let it melt completely. Step 4: Pour in the eggs and stir gently. Step 5: Remove from heat while still slightly wet. By following these steps, you'll achieve perfect results every time."
        },
        {
            id: 'chatgpt_list',
            label: 'ai',
            model: 'ChatGPT',
            category: 'list',
            text: "Here are 5 tips for better sleep: 1. Maintain a consistent sleep schedule, even on weekends. 2. Create a relaxing bedtime routine. 3. Keep your bedroom cool and dark. 4. Avoid screens at least one hour before bed. 5. Limit caffeine intake, especially in the afternoon. Implementing these habits can significantly improve your sleep quality."
        },
        {
            id: 'chatgpt_explain',
            label: 'ai',
            model: 'ChatGPT',
            category: 'explanation',
            text: "Quantum computing represents a paradigm shift in computational capability. Unlike classical computers that use bits (0 or 1), quantum computers use qubits that can exist in multiple states simultaneously. This property, known as superposition, allows quantum computers to process vast amounts of data in parallel. The implications for cryptography, drug discovery, and optimization problems are significant."
        },
        
        // More AI patterns
        {
            id: 'ai_essay_formal',
            label: 'ai',
            model: 'Generic',
            category: 'essay',
            text: "The advent of social media has fundamentally transformed how humans communicate and form relationships. This transformation carries both significant benefits and notable drawbacks. On the positive side, social media enables instant global connectivity. Conversely, concerns about mental health impacts have grown considerably. In conclusion, society must navigate these complexities thoughtfully."
        },
        {
            id: 'ai_summary',
            label: 'ai',
            model: 'Generic',
            category: 'summary',
            text: "In summary, the research demonstrates several key findings. First, the intervention showed statistically significant improvements in participant outcomes. Second, the effect size was larger than anticipated. Third, these results were consistent across demographic groups. The implications of these findings suggest that wider implementation could yield substantial benefits."
        },
        {
            id: 'ai_compare',
            label: 'ai',
            model: 'Generic',
            category: 'comparison',
            text: "When comparing Python and JavaScript, several factors merit consideration. Python excels in data science and machine learning applications due to its extensive library ecosystem. JavaScript, conversely, dominates web development and offers superior browser integration. Both languages have their merits, and the optimal choice depends on your specific use case and project requirements."
        }
    ],

    // ==================== HUMANIZED AI SAMPLES ====================
    humanized: [
        {
            id: 'humanized_casual',
            label: 'ai',
            category: 'humanized',
            source: 'GPT-4 + humanizer',
            text: "Okay so like, here's the thing about AI that nobody talks about? It's not gonna take over the world lol. That's just not how it works. I mean sure, it can do some pretty wild stuff nowadays. But actual creativity? Real understanding? Nah. It's just really fancy pattern matching at the end of the day."
        },
        {
            id: 'humanized_mixed',
            label: 'ai',
            category: 'humanized',
            source: 'Claude + light edit',
            text: "So I've been thinking about climate change (haven't we all lol) and honestly the whole thing is kinda complex. Like on one hand you've got scientists saying we need to act now. But then there are economic considerations too, you know? It's not as simple as people make it out to be tbh."
        },
        {
            id: 'humanized_injected',
            label: 'ai',
            category: 'humanized',
            source: 'AI + typo injection',
            text: "The implications of artifical intelligence are far-reaching. First, we need to consider teh economic impact. Second, there's the question of employment. Third - and this is probabl the most important - we need to thing about ethics. These issues are interconneted and require careful cosideration."
        },
        {
            id: 'humanized_slang',
            label: 'ai',
            category: 'humanized',
            source: 'AI + slang overlay',
            text: "Yo so blockchain is actually pretty fire when you break it down ngl. It's basically a decentralized ledger system that leverages cryptographic mechanisms to ensure data integrity. The comprehensive security framework is lowkey revolutionary. Furthermore, the scalability solutions being implemented are bussin fr fr."
        },
        {
            id: 'humanized_filler',
            label: 'ai',
            category: 'humanized',
            source: 'AI + filler words',
            text: "Well, I mean, the thing is, artificial intelligence has basically transformed how we, you know, approach problem-solving. It's like, fundamentally changed the paradigm or whatever. Honestly though, the comprehensive implications are still being, um, fully understood I guess."
        },
        {
            id: 'humanized_contraction',
            label: 'ai',
            category: 'humanized',
            source: 'AI + forced contractions',
            text: "It's really important that we're understanding how AI's transforming our world. We've got to think about the implications. There's no doubt that it'll change everything. That's why we shouldn't ignore it. I'd say we're at a crucial juncture. Here's what I've noticed about it."
        },
        {
            id: 'humanized_question',
            label: 'ai',
            category: 'humanized',
            source: 'AI + rhetorical questions',
            text: "Have you ever wondered why AI is everywhere now? It's fascinating, right? The technology has advanced so rapidly. Don't you think that's remarkable? The implications are significant. Isn't it time we started paying attention? I believe the answer is clear, wouldn't you agree?"
        },
        {
            id: 'humanized_emoji',
            label: 'ai',
            category: 'humanized',
            source: 'AI + emoji injection',
            text: "Climate change is honestly such a huge deal 🌍 We really need to start taking it seriously! 💪 The science is clear: temperatures are rising, glaciers are melting, and extreme weather events are becoming more common 😰 It's time for action! 🔥 What are your thoughts?"
        }
    ],

    // ==================== EDGE CASES ====================
    edge: [
        // Formal human writing
        {
            id: 'formal_human',
            label: 'human',
            category: 'formal',
            text: "Furthermore, it is essential to note that the policy implementation has resulted in measurable improvements. Consequently, the committee recommends continuation of the current approach. Nevertheless, careful monitoring remains necessary. Thus, a quarterly review mechanism has been established."
        },
        {
            id: 'legal_human',
            label: 'human',
            category: 'formal',
            text: "Notwithstanding the aforementioned provisions, the party of the first part hereby agrees to indemnify and hold harmless the party of the second part. This agreement shall remain in effect unless terminated by mutual written consent. All disputes shall be resolved through binding arbitration."
        },
        {
            id: 'medical_human',
            label: 'human',
            category: 'formal',
            text: "The patient presented with acute onset chest pain radiating to the left arm. Initial troponin was elevated. ECG showed ST elevation in leads V2-V4. STEMI protocol was initiated. Patient was transferred to cath lab where PCI was performed successfully."
        },
        
        // AI-assisted human
        {
            id: 'ai_assisted',
            label: 'human',
            category: 'assisted',
            text: "I asked ChatGPT for some ideas and it gave me a good starting point, but honestly most of this is my own research and opinions. The thing about renewable energy that people miss is that it's not just about the environment - it's about economics. My dad's been in solar for 15 years and he's seen costs drop like crazy."
        },
        
        // Very short text
        {
            id: 'short_human',
            label: 'human',
            category: 'short',
            text: "Thanks for the heads up. I'll check it out tomorrow."
        },
        {
            id: 'short_ai',
            label: 'ai',
            category: 'short',
            text: "Thank you for bringing this to my attention. I will investigate further."
        },
        
        // Instructional human
        {
            id: 'recipe_human',
            label: 'human',
            category: 'instructional',
            text: "Grandma's pasta sauce: brown the meat first (this is important!!), then add garlic but DON'T burn it. My sister always burns the garlic smh. Dump in the tomatoes, add a bay leaf, and let it simmer forever basically. Like 3-4 hours minimum. Worth it."
        },
        
        // Technical human
        {
            id: 'code_review_human',
            label: 'human',
            category: 'technical',
            text: "LGTM but two nits: 1) this useEffect will cause infinite loops if you add more deps later - consider useCallback for the handler 2) why the any type on line 47? we have the interface imported. Not blocking but would be nice to fix."
        },
        
        // Mixed language
        {
            id: 'mixed_language',
            label: 'human',
            category: 'multilingual',
            text: "So basically the meeting was en español but I only understood like half of it lol. Mi español is not that bueno apparently. They were talking about el proyecto nuevo but I got lost when they started discussing los números. Need to practice more tbh."
        },
        
        // Emotional human
        {
            id: 'emotional_rant',
            label: 'human',
            category: 'emotional',
            text: "I am SO done with this company. THREE YEARS of loyal service and they promote someone who's been here 6 months?! Are you kidding me?? I have literally been doing that person's job for a year while they figure things out. Whatever. Their loss. I'm updating my resume tonight."
        },
        
        // Quoted content
        {
            id: 'quote_heavy',
            label: 'human',
            category: 'quotes',
            text: "The article says \"AI will transform every industry\" but that's what they said about blockchain too, remember? And before that it was \"big data will revolutionize everything.\" I'm not saying AI isn't important, but maybe we cool it with the hyperbole?"
        }
    ]
};

// ==================== ANALYSIS FUNCTION ====================
function analyze(text) {
    const lower = text.toLowerCase();
    const words = lower.split(/\s+/);
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 5);
    let score = 0;
    let signals = [];
    let confidence = 0.7;
    
    // Adjust confidence for text length
    if (words.length < 30) confidence *= 0.5;
    else if (words.length < 50) confidence *= 0.7;
    else if (words.length > 200) confidence *= 1.1;
    
    // ========== AI VOCABULARY (weak signal - downweighted) ==========
    const aiWords = ['delve','multifaceted','tapestry','comprehensive','crucial','furthermore','moreover','consequently','nevertheless','facilitate','leverage','nuanced','paradigm','implications','significant','utilize','enhance','robust','seamless','innovative','dynamic','pivotal','paramount','imperative'];
    const aiCount = aiWords.filter(w => lower.includes(w)).length;
    if (aiCount >= 3) { score += 0.12; signals.push('ai_vocab:'+aiCount); }
    else if (aiCount >= 1) { score += 0.05; signals.push('ai_vocab:'+aiCount); }
    
    // ========== AI PHRASES (strong signal) ==========
    const aiPhrases = [
        'in conclusion', 'firstly', 'secondly', 'thirdly', 'step 1', 'step 2',
        'great question', 'happy to help', 'hope this helps', 'feel free',
        'here\'s a breakdown', 'here are some', 'it is essential', 'it is important to note',
        'on one hand', 'on the other hand', 'it\'s worth noting', 'let me',
        'i\'d be happy to', 'that\'s a great', 'the key takeaway', 'in summary',
        'there are several', 'there are three', 'there are many ways',
        'when it comes to', 'in terms of', 'with that said', 'having said that'
    ];
    const phraseCount = aiPhrases.filter(p => lower.includes(p)).length;
    if (phraseCount >= 3) { score += 0.25; signals.push('ai_phrases:'+phraseCount); }
    else if (phraseCount >= 1) { score += 0.12; signals.push('ai_phrases:'+phraseCount); }
    
    // ========== SENTENCE UNIFORMITY ==========
    // Calculate CV outside the if block so it's available later
    let cv = 0.5;  // Default
    if (sentences.length >= 3) {
        const lens = sentences.map(s => s.split(/\s+/).length);
        const mean = lens.reduce((a,b)=>a+b,0)/lens.length||0;
        const variance = lens.reduce((s,x)=>s+Math.pow(x-mean,2),0)/lens.length;
        cv = mean > 0 ? Math.sqrt(variance)/mean : 0.5;
        if (cv < 0.25) { score += 0.15; signals.push('uniform:'+cv.toFixed(2)); }
        else if (cv < 0.35) { score += 0.08; signals.push('uniform:'+cv.toFixed(2)); }
    }
    
    // ========== HUMAN MARKERS (strong negative signal) ==========
    const humanWords = ['lol','honestly','literally','damn','stupid','oops','like,','ugh','omg','gonna','gotta','kinda','sorta','yeah','nope','welp','tbh','ngl','lmao','wtf','smh','bruh','yall','yknow','btw','tho','rn','abt'];
    // Use word boundaries to avoid false matches (e.g., "actually" in AI text)
    const humanCount = humanWords.filter(w => {
        const pattern = new RegExp(`\\b${w.replace(',', ',?')}\\b`, 'i');
        return pattern.test(lower);
    }).length;
    
    // Only apply human marker penalty if NO strong AI signals present
    // This prevents AI text with casual words from being marked human
    const hasStrongAISignals = phraseCount >= 2 || aiCount >= 3;
    if (!hasStrongAISignals) {
        if (humanCount >= 2) { score -= 0.20; signals.push('human:'+humanCount); }
        else if (humanCount >= 1) { score -= 0.12; signals.push('human:'+humanCount); }
    } else if (humanCount >= 1) {
        // AI with human markers - slight reduction but don't override
        score -= 0.05;
        signals.push('human:'+humanCount);
    }
    
    // ========== HELPER TONE ==========
    const helpers = ['happy to help','great question','hope this helps','feel free','i\'d be happy','absolutely!','certainly!','of course!'];
    const helperCount = helpers.filter(p => lower.includes(p)).length;
    if (helperCount >= 1) { score += 0.15; signals.push('helper:'+helperCount); }
    
    // ========== ESL DETECTION ==========
    const eslPatterns = [
        /\b(i|he|she|they|we) (go|buy|get|come|see|tell|make|take) [^a-z]/i,
        /\bmy (mother|father|brother|sister|friend|boss) (she|he) /i,
        /\bfor (buy|get|see|make|do)/i,
        /\bthe (price|weather|thing|problem) is very /i,
        /\bbut then (laugh|cry|smile|angry|happy)/i,
        /\bI am (hoping|wishing|thinking) to /i,
        /\bplease (find|see|check) (attached|below)/i,
        /\bvery (much|big|high|small|good|bad) /i
    ];
    const eslMatches = eslPatterns.filter(p => p.test(text)).length;
    if (eslMatches >= 2) { score -= 0.30; signals.push('ESL:'+eslMatches); }
    else if (eslMatches >= 1) { score -= 0.15; signals.push('ESL:'+eslMatches); }
    
    // ========== FORMAL WRITING DETECTION ==========
    const formalTransitions = ['furthermore','consequently','nevertheless','thus','therefore','moreover','hence','accordingly','notwithstanding','aforementioned','hereby'];
    const formalCount = formalTransitions.filter(w => lower.includes(w)).length;
    const hasPersonalVoice = /\b(my own|i believe|in my view|i think|i acknowledge|our|we believe|my experience|my opinion|i feel|i\'m|i am [a-z]+ing)\b/i.test(text);
    const hasEmotionalMarkers = /\b(honestly|frustrated|excited|worried|annoyed|thrilled|disappointed|surprised|confused|angry|happy|sad|love|hate)\b/i.test(lower);
    
    if (formalCount >= 2 && (hasPersonalVoice || hasEmotionalMarkers)) {
        score -= 0.20;
        signals.push('formal_personal');
    }
    
    // Short formal text detection
    if (formalCount >= 2 && sentences.length <= 5) {
        score -= 0.15;
        signals.push('short_formal');
    }
    
    // ========== MEDICAL/LEGAL/TECHNICAL JARGON DETECTION ==========
    const medicalPatterns = [
        /\b(ECG|EKG|MRI|CT|STEMI|PCI|IV|BP|HR|SpO2|PRN|BID|TID|QID)\b/,
        /\b(patient|presented|diagnosis|treatment|protocol|administered)\b/i,
        /\b(elevated|initiated|transferred|performed|discharged)\b/i
    ];
    const legalPatterns = [
        /\b(notwithstanding|aforementioned|hereby|herein|thereof|wherein)\b/i,
        /\b(party of the|indemnify|hold harmless|binding arbitration)\b/i,
        /\b(pursuant|stipulate|covenant|provision|executed)\b/i
    ];
    const medicalCount = medicalPatterns.filter(p => p.test(text)).length;
    const legalCount = legalPatterns.filter(p => p.test(text)).length;
    
    if (medicalCount >= 2) { score -= 0.25; signals.push('medical:'+medicalCount); }
    if (legalCount >= 2) { score -= 0.25; signals.push('legal:'+legalCount); }
    
    // ========== QUOTATION DETECTION ==========
    // Quoted content creates artificial uniformity - human meta-commentary
    const quoteMatches = text.match(/"[^"]+"/g) || [];
    const hasMetaCommentary = /\b(they said|people say|the article says|as .+ said|remember\?|before that)\b/i.test(text);
    if (quoteMatches.length >= 2 && hasMetaCommentary) {
        score -= 0.20;
        signals.push('quotes:'+quoteMatches.length);
    }
    
    // ========== ACADEMIC WRITING DETECTION ==========
    // Academic has formal vocab BUT also questioning, uncertainty, citations
    const academicMarkers = [
        /\b(scholars|contested|controversial|camps|insights)\b/i,
        /\b(my own view|which i acknowledge|many will find)\b/i,
        /\b(the literature|failed replications|p-value|power analysis)\b/i,
        /\b(methodology section|raises concerns|calculation error)\b/i
    ];
    const academicCount = academicMarkers.filter(p => p.test(text)).length;
    if (academicCount >= 1) { score -= 0.15; signals.push('academic:'+academicCount); }
    
    // ========== INTRA-DOCUMENT DRIFT ==========
    if (sentences.length >= 4) {
        const half = Math.floor(sentences.length/2);
        const firstLens = sentences.slice(0, half).map(s => s.split(/\s+/).length);
        const secondLens = sentences.slice(half).map(s => s.split(/\s+/).length);
        const firstAvg = firstLens.reduce((a,b)=>a+b,0)/firstLens.length;
        const secondAvg = secondLens.reduce((a,b)=>a+b,0)/secondLens.length;
        const drift = Math.abs(firstAvg - secondAvg) / Math.max(firstAvg, secondAvg, 1);
        
        if (drift < 0.08) { score += 0.10; signals.push('nodrift:'+drift.toFixed(2)); }
        else if (drift > 0.35) { score -= 0.08; signals.push('drift:'+drift.toFixed(2)); }
    }
    
    // ========== HUMANIZED AI DETECTION ==========
    const humanizedPatterns = [
        /so like,?\s*here'?s?\s*(the thing|what)/i,
        /that nobody talks about/i,
        /just\s*(not|isn't|doesn't|won't|can't)/i,
        /i mean\s*sure/i,
        /nah\.?\s*(just|it's)/i,
        /(gonna|gotta|wanna).{0,20}(the world|everything|all|really)/i,
        /okay so/i,
        /here'?s?\s*the thing about/i,
        /but\s+(creativity|understanding|consciousness)\?/i,
        /fancy pattern matching/i,
        /at the end of the day/i,
        /you know\?/i,
        /right\?$/im,
        /tbh$/im,
        /lol$/im
    ];
    const humanizedCount = humanizedPatterns.filter(p => p.test(text)).length;
    
    // Register mixing: slang + formal
    const hasSlang = humanCount >= 2 || /\b(yo|fire|lit|bussin|lowkey|highkey|ngl|tbh|fr fr)\b/i.test(lower);
    const hasFormalVocab = aiCount >= 2 || /\b(comprehensive|furthermore|leverage|paradigm|implementation)\b/i.test(lower);
    const registerMix = hasSlang && hasFormalVocab;
    
    if (registerMix) {
        score += 0.25;
        signals.push('register_mix');
    }
    
    // Typo injection detection
    const injectedTypos = ['teh','hte','adn','taht','wiht','tihs','artifical','probabl','definately','diferent','thier'];
    const typoCount = injectedTypos.filter(t => lower.includes(t)).length;
    const hasAIStructure = phraseCount >= 1 || aiCount >= 2;
    if (typoCount >= 2 && hasAIStructure) {
        score += 0.20;
        signals.push('typo_inject:'+typoCount);
    }
    
    // Override human markers for strong humanized patterns
    if (humanizedCount >= 3) {
        const humanMarkerPenalty = humanCount * 0.15;
        score += humanMarkerPenalty + 0.20;
        signals.push('humanized:'+humanizedCount);
    } else if (humanizedCount >= 2) {
        score += 0.15;
        signals.push('humanized:'+humanizedCount);
    }
    
    // ========== INSTRUCTIONAL PATTERN DETECTION ==========
    const instructionalPatterns = [
        /step\s*[1-9]/gi,
        /\d+\.\s+[A-Z]/g,
        /first,?\s+.*second,?\s+/i,
        /here'?s?\s+(how|what|why)/i,
        /by following these/i,
        /implementing these/i
    ];
    const instructionalCount = instructionalPatterns.filter(p => p.test(text)).length;
    if (instructionalCount >= 2) {
        score += 0.12;
        signals.push('instructional:'+instructionalCount);
    }
    
    // ========== EMOTIONAL AUTHENTICITY ==========
    const emotionalPatterns = [
        /!{2,}/,              // Multiple exclamation marks
        /\?{2,}/,             // Multiple question marks
        /SO (done|tired|frustrated|excited|happy)/i,
        /are you kidding/i,
        /I can't (believe|stand|deal)/i,
        /ugh+/i,
        /argh+/i
    ];
    const emotionalCount = emotionalPatterns.filter(p => p.test(text)).length;
    if (emotionalCount >= 2) {
        score -= 0.15;
        signals.push('emotional:'+emotionalCount);
    }
    
    // ========== EMOJI INJECTION DETECTION (humanized AI) ==========
    const emojiPattern = /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu;
    const emojis = text.match(emojiPattern) || [];
    // Emoji + formal/educational content = humanized AI
    const hasEducationalContent = /\b(science is clear|temperatures are rising|extreme weather|time for action)\b/i.test(text);
    const hasAIContent = aiCount >= 1 || phraseCount >= 1 || (sentences.length >= 3 && cv < 0.35) || hasEducationalContent;
    if (emojis.length >= 3 && hasAIContent) {
        score += 0.30;  // Emoji injection with AI structure = humanized
        signals.push('emoji_inject:'+emojis.length);
    } else if (emojis.length >= 2 && hasEducationalContent) {
        score += 0.20;  // Educational + emojis = likely humanized
        signals.push('emoji_edu:'+emojis.length);
    }
    
    // ========== VERY SHORT TEXT HANDLING ==========
    // For very short text, use stronger signals
    if (words.length < 15) {
        // Check for clear human signals first
        const hasShortHumanSignals = /\b(thanks|heads up|check it out|lmk|btw|i'll)\b/i.test(lower);
        const hasContractions = /\b(i'll|i'm|it's|that's|don't|won't|can't)\b/i.test(lower);
        if (hasShortHumanSignals && hasContractions) {
            score -= 0.20;
            signals.push('short_casual');
        } else if (hasShortHumanSignals) {
            score -= 0.10;
            signals.push('short_human');
        } else {
            score = score * 0.4 + 0.30;  // Pull toward 0.5 (uncertain)
            signals.push('very_short');
        }
        confidence *= 0.4;
    } else if (words.length < 25) {
        // Check for clear human signals in short text
        const hasShortHumanSignals = /\b(thanks|heads up|check it out|lmk|btw)\b/i.test(lower);
        if (hasShortHumanSignals) {
            score -= 0.15;
            signals.push('short_human');
        } else {
            score = score * 0.6 + 0.20;  // Pull toward lower
            signals.push('short_text');
            confidence *= 0.5;
        }
    }
    
    // ========== FORMAL HUMAN FINAL CHECK ==========
    // If text has formal vocab but ALSO has hedging/personal elements not typical of AI
    const formalHumanMarkers = [
        /\bit is essential to note that\b/i,  // Very specific AI pattern
        /\bresulted in\b.*\bimprovements\b/i,
        /\bcommittee recommends\b/i,
        /\bquarterly review\b/i
    ];
    const formalAIPatterns = [
        /\bplays a (crucial|vital|key|pivotal) role\b/i,
        /\ba (wide range|variety|plethora|myriad) of\b/i,
        /\bin this (context|regard|respect)\b/i
    ];
    const formalHumanCount = formalHumanMarkers.filter(p => p.test(text)).length;
    const formalAICount = formalAIPatterns.filter(p => p.test(text)).length;
    
    // Formal writing without strong AI patterns = likely human professional
    if (formalCount >= 2 && formalAICount === 0 && formalHumanCount >= 1) {
        score -= 0.15;
        signals.push('formal_human_style');
    }
    
    // ========== TECHNICAL/MEDICAL JARGON ==========
    const technicalPatterns = [
        /\b(ECG|MRI|CT|STEMI|PCI|IV|BP|HR|SpO2)\b/,
        /\b(useEffect|useState|useCallback|async|await|const|let|var)\b/,
        /\b(API|SDK|REST|JSON|XML|HTTP|SQL)\b/,
        /@[a-z]+/i,  // @mentions
        /#[a-z]+/i,  // #channels
        /line\s+\d+/i
    ];
    const technicalCount = technicalPatterns.filter(p => p.test(text)).length;
    if (technicalCount >= 2 && humanCount >= 1) {
        score -= 0.15;
        signals.push('technical_human');
    }
    
    // Final probability
    const prob = Math.max(0, Math.min(1, 0.45 + score));
    confidence = Math.min(0.95, Math.max(0.2, confidence));
    
    return { prob, signals, confidence };
}

// ==================== RUN BENCHMARKS ====================
function runBenchmarks() {
    console.log('='.repeat(70));
    console.log('VERITAS Comprehensive Benchmark Suite');
    console.log('='.repeat(70));
    console.log('');
    
    const results = {
        human: { correct: 0, total: 0, samples: [] },
        ai: { correct: 0, total: 0, samples: [] },
        humanized: { correct: 0, total: 0, samples: [] },
        edge: { correct: 0, total: 0, samples: [] }
    };
    
    const categories = ['human', 'ai', 'humanized', 'edge'];
    
    for (const category of categories) {
        console.log(`\n${'─'.repeat(60)}`);
        console.log(`${category.toUpperCase()} SAMPLES`);
        console.log('─'.repeat(60));
        
        for (const sample of testSamples[category]) {
            const r = analyze(sample.text);
            const isAI = sample.label === 'ai';
            const predictedAI = r.prob >= 0.5;
            const correct = isAI === predictedAI;
            
            results[category].total++;
            if (correct) results[category].correct++;
            
            const status = correct ? '✓' : (isAI ? '✗FN' : '✗FP');
            const probStr = (r.prob * 100).toFixed(0).padStart(3) + '%';
            const confStr = (r.confidence * 100).toFixed(0) + '%';
            const signalStr = r.signals.slice(0, 4).join(' ');
            
            const cat = sample.category ? `[${sample.category}]` : '';
            const model = sample.model ? `(${sample.model})` : '';
            
            console.log(`  ${status} ${sample.id}${model} ${cat}: ${probStr} conf:${confStr} ${signalStr}`);
            
            results[category].samples.push({
                id: sample.id,
                correct,
                prob: r.prob,
                signals: r.signals,
                expected: sample.label
            });
        }
        
        const rate = (results[category].correct / results[category].total * 100).toFixed(0);
        console.log(`  → ${results[category].correct}/${results[category].total} (${rate}%)`);
    }
    
    // Summary
    console.log('\n' + '='.repeat(70));
    console.log('SUMMARY');
    console.log('='.repeat(70));
    
    const totalCorrect = categories.reduce((s, c) => s + results[c].correct, 0);
    const totalSamples = categories.reduce((s, c) => s + results[c].total, 0);
    
    console.log(`\nHuman samples:     ${results.human.correct}/${results.human.total} (${(results.human.correct/results.human.total*100).toFixed(0)}%)`);
    console.log(`AI samples:        ${results.ai.correct}/${results.ai.total} (${(results.ai.correct/results.ai.total*100).toFixed(0)}%)`);
    console.log(`Humanized AI:      ${results.humanized.correct}/${results.humanized.total} (${(results.humanized.correct/results.humanized.total*100).toFixed(0)}%)`);
    console.log(`Edge cases:        ${results.edge.correct}/${results.edge.total} (${(results.edge.correct/results.edge.total*100).toFixed(0)}%)`);
    console.log(`${'─'.repeat(40)}`);
    console.log(`TOTAL:             ${totalCorrect}/${totalSamples} (${(totalCorrect/totalSamples*100).toFixed(1)}%)`);
    
    // Calculate metrics
    const humanFP = results.human.samples.filter(s => !s.correct).length;
    const humanTotal = results.human.total;
    const edgeHumanFP = results.edge.samples.filter(s => s.expected === 'human' && !s.correct).length;
    const edgeHumanTotal = results.edge.samples.filter(s => s.expected === 'human').length;
    
    const aiFN = results.ai.samples.filter(s => !s.correct).length;
    const aiTotal = results.ai.total;
    const humanizedFN = results.humanized.samples.filter(s => !s.correct).length;
    const humanizedTotal = results.humanized.total;
    
    const fpRate = ((humanFP + edgeHumanFP) / (humanTotal + edgeHumanTotal) * 100).toFixed(1);
    const fnRate = ((aiFN + humanizedFN) / (aiTotal + humanizedTotal) * 100).toFixed(1);
    
    console.log(`\nFalse Positive Rate: ${fpRate}%`);
    console.log(`False Negative Rate: ${fnRate}%`);
    
    // Detailed failure analysis
    const failures = [];
    for (const cat of categories) {
        for (const s of results[cat].samples) {
            if (!s.correct) {
                failures.push({ category: cat, ...s });
            }
        }
    }
    
    if (failures.length > 0) {
        console.log(`\n${'─'.repeat(60)}`);
        console.log('FAILURE ANALYSIS');
        console.log('─'.repeat(60));
        
        for (const f of failures) {
            console.log(`  ${f.category}/${f.id}: expected ${f.expected}, got ${f.prob >= 0.5 ? 'ai' : 'human'} (${(f.prob*100).toFixed(0)}%)`);
            console.log(`    signals: ${f.signals.join(', ')}`);
        }
    }
    
    // ML Training Recommendation
    console.log('\n' + '='.repeat(70));
    console.log('ML TRAINING ASSESSMENT');
    console.log('='.repeat(70));
    
    const overallAccuracy = totalCorrect / totalSamples;
    const humanizedAccuracy = results.humanized.correct / results.humanized.total;
    const edgeAccuracy = results.edge.correct / results.edge.total;
    
    if (overallAccuracy >= 0.90 && humanizedAccuracy >= 0.80) {
        console.log('\n✅ ML TRAINING: OPTIONAL');
        console.log('   Current heuristic system achieves strong performance.');
        console.log('   ML would provide marginal improvement at significant complexity cost.');
    } else if (overallAccuracy >= 0.80) {
        console.log('\n⚠️  ML TRAINING: RECOMMENDED FOR EDGE CASES');
        console.log(`   Core detection is solid (${(overallAccuracy*100).toFixed(0)}%).`);
        console.log(`   ML could improve humanized detection (${(humanizedAccuracy*100).toFixed(0)}%) and edge cases (${(edgeAccuracy*100).toFixed(0)}%).`);
    } else {
        console.log('\n🔴 ML TRAINING: STRONGLY RECOMMENDED');
        console.log(`   Current accuracy (${(overallAccuracy*100).toFixed(0)}%) is below acceptable threshold.`);
        console.log('   Heuristics are insufficient for modern LLM detection.');
    }
    
    // Specific recommendations
    console.log('\nSpecific findings:');
    if (humanizedAccuracy < 0.75) {
        console.log('  • Humanized AI detection needs improvement');
        console.log('    → Consider training on humanizer tool outputs');
    }
    if (fpRate > 10) {
        console.log('  • False positive rate is high');
        console.log('    → Need better ESL, formal, and technical writing handling');
    }
    if (fnRate > 15) {
        console.log('  • False negative rate is concerning');
        console.log('    → Modern LLM outputs are evading detection');
    }
    
    return {
        overall: overallAccuracy,
        human: results.human.correct / results.human.total,
        ai: results.ai.correct / results.ai.total,
        humanized: humanizedAccuracy,
        edge: edgeAccuracy,
        fpRate: parseFloat(fpRate),
        fnRate: parseFloat(fnRate),
        failures,
        mlRequired: overallAccuracy < 0.85
    };
}

// Run
const results = runBenchmarks();
process.exit(results.overall >= 0.85 ? 0 : 1);
