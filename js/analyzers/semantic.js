/**
 * VERITAS — Semantic & Pragmatic Features Analyzer
 * Category 7: Content Depth, Informational Density, Human Experience Signals
 */

const SemanticAnalyzer = {
    name: 'Semantic & Pragmatic Features',
    category: 7,
    weight: 1.0,

    // Generic/vague phrases (low informational density)
    genericPhrases: [
        'in various ways', 'for various reasons', 'in many cases',
        'a number of', 'a variety of', 'a range of', 'a wide range of',
        'plays an important role', 'plays a crucial role', 'plays a significant role',
        'has a significant impact', 'has a profound impact', 'has a major impact',
        'is of great importance', 'is of paramount importance',
        'in the context of', 'with regard to', 'in terms of',
        'it is worth noting', 'it is important to', 'it is essential to',
        'can be seen', 'can be observed', 'can be noted',
        'there are many', 'there are various', 'there are numerous',
        'this is because', 'this is due to', 'this is a result of'
    ],

    // Personal experience markers (human indicators)
    experienceMarkers: [
        'i remember', 'i recall', 'i think', 'i believe', 'i feel',
        'in my experience', 'from my perspective', 'personally',
        'when i was', 'i once', 'i used to', 'i never thought',
        'i\'m not sure', 'i don\'t know', 'i wonder', 'i guess',
        'honestly', 'frankly', 'to be honest', 'truth be told',
        'it seemed to me', 'i always', 'i sometimes', 'i often',
        'my mother', 'my father', 'my friend', 'my family',
        'growing up', 'as a child', 'back then', 'years ago'
    ],

    // Uncertainty markers (can indicate both)
    uncertaintyMarkers: [
        'perhaps', 'maybe', 'possibly', 'probably', 'likely',
        'might', 'could', 'may', 'seems', 'appears',
        'i think', 'i believe', 'in my opinion', 'as far as i know',
        'it\'s possible', 'there\'s a chance', 'it could be',
        'not entirely sure', 'hard to say', 'difficult to determine'
    ],

    // Safe/non-committal phrases (AI indicator)
    safeNoncommittal: [
        'on the one hand', 'on the other hand', 'there are pros and cons',
        'it depends on', 'it varies', 'it can be argued',
        'some people believe', 'others argue', 'while some',
        'both sides have merit', 'there are valid points',
        'it\'s a complex issue', 'it\'s nuanced', 'it\'s multifaceted',
        'there is no simple answer', 'it requires careful consideration',
        'different perspectives exist', 'opinions differ'
    ],

    // Concrete/specific detail patterns
    concretePatterns: [
        /\d+\s*(years?|months?|days?|hours?|minutes?|seconds?)/gi,
        /\$\d+[\d,.]*/g,  // Dollar amounts
        /\b\d{1,2}:\d{2}\b/g,  // Times
        /\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d+/gi,
        /"[^"]{10,}"/g,  // Quoted speech
        /\b[A-Z][a-z]+\s+[A-Z][a-z]+\b/g,  // Proper names
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text);
        
        if (tokens.length < 30) {
            return this.getEmptyResult();
        }

        // Analyze content depth and specificity
        const depthAnalysis = this.analyzeContentDepth(text, sentences);
        
        // Analyze human experience signals
        const experienceAnalysis = this.analyzeHumanExperience(text);
        
        // Analyze semantic smoothness vs grounding
        const groundingAnalysis = this.analyzeGrounding(text, sentences);
        
        // Analyze stance and commitment
        const stanceAnalysis = this.analyzeStance(text, sentences);

        // Calculate AI probability
        // Low specificity = AI-like
        // No experience signals = AI-like
        // High smoothness = AI-like
        // Non-committal stance = AI-like

        const scores = {
            lowSpecificity: 1 - depthAnalysis.specificityScore,
            noExperience: 1 - experienceAnalysis.experienceScore,
            semanticSmooth: groundingAnalysis.smoothnessScore,
            noncommittal: stanceAnalysis.noncommittalScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.lowSpecificity, scores.noExperience, scores.semanticSmooth, scores.noncommittal],
            [0.25, 0.3, 0.2, 0.25]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(tokens.length),
            details: {
                depthAnalysis,
                experienceAnalysis,
                groundingAnalysis,
                stanceAnalysis
            },
            findings: this.generateFindings(depthAnalysis, experienceAnalysis, groundingAnalysis, stanceAnalysis),
            scores
        };
    },

    /**
     * Analyze content depth and specificity
     */
    analyzeContentDepth(text, sentences) {
        const lower = text.toLowerCase();
        
        // Count generic phrases
        let genericCount = 0;
        const genericFound = [];
        for (const phrase of this.genericPhrases) {
            if (lower.includes(phrase.toLowerCase())) {
                genericCount++;
                genericFound.push(phrase);
            }
        }
        
        // Count concrete details
        let concreteCount = 0;
        const concreteExamples = [];
        for (const pattern of this.concretePatterns) {
            const matches = text.match(pattern);
            if (matches) {
                concreteCount += matches.length;
                concreteExamples.push(...matches.slice(0, 2));
            }
        }
        
        // Calculate information density (content words per sentence)
        const contentWords = Utils.tokenize(text).filter(w => 
            !Utils.functionWords.has(w) && w.length > 3
        );
        const infoPerSentence = sentences.length > 0 
            ? contentWords.length / sentences.length 
            : 0;
        
        // Specificity score: high concrete, low generic
        const genericRatio = sentences.length > 0 ? genericCount / sentences.length : 0;
        const concreteRatio = sentences.length > 0 ? concreteCount / sentences.length : 0;
        
        const specificityScore = (
            (1 - Utils.normalize(genericRatio, 0, 0.3)) * 0.5 +
            Utils.normalize(concreteRatio, 0, 0.5) * 0.5
        );

        return {
            genericPhraseCount: genericCount,
            genericFound: genericFound.slice(0, 5),
            concreteDetailCount: concreteCount,
            concreteExamples: [...new Set(concreteExamples)].slice(0, 5),
            infoPerSentence: infoPerSentence.toFixed(1),
            specificityScore
        };
    },

    /**
     * Analyze human experience signals
     */
    analyzeHumanExperience(text) {
        const lower = text.toLowerCase();
        
        let experienceCount = 0;
        const experienceFound = [];
        
        for (const marker of this.experienceMarkers) {
            const pattern = new RegExp(`\\b${marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
            if (pattern.test(lower)) {
                experienceCount++;
                experienceFound.push(marker);
            }
        }
        
        // Check for personal anecdotes (sentences with "I" + past tense)
        const sentences = Utils.splitSentences(text);
        let anecdoteCount = 0;
        for (const sentence of sentences) {
            if (/\bI\s+(was|had|did|went|saw|heard|felt|thought|knew|found|met|got|came|made|took|gave)\b/i.test(sentence)) {
                anecdoteCount++;
            }
        }
        
        // Check for emotional language
        const emotionalWords = ['love', 'hate', 'angry', 'sad', 'happy', 'scared', 'worried', 'excited', 'frustrated', 'surprised', 'confused', 'embarrassed', 'proud', 'ashamed'];
        let emotionalCount = 0;
        for (const word of emotionalWords) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                emotionalCount++;
            }
        }
        
        // Experience score
        const experienceScore = Utils.normalize(experienceCount + anecdoteCount * 2, 0, 8);

        return {
            experienceMarkerCount: experienceCount,
            experienceFound: experienceFound.slice(0, 5),
            anecdoteCount,
            emotionalWordCount: emotionalCount,
            experienceScore
        };
    },

    /**
     * Analyze semantic smoothness vs grounding
     */
    analyzeGrounding(text, sentences) {
        // Check for high coherence with low information (AI pattern)
        // AI text often reads smoothly but says little concrete
        
        // Count qualified statements
        const qualifiedPattern = /\b(generally|typically|usually|often|sometimes|can|may|might|could)\b/gi;
        const qualifiedCount = (text.match(qualifiedPattern) || []).length;
        
        // Count assertive statements
        const assertivePattern = /\b(is|are|was|were|will|must|always|never|definitely|certainly)\b/gi;
        const assertiveCount = (text.match(assertivePattern) || []).length;
        
        const qualificationRatio = (qualifiedCount + assertiveCount) > 0 
            ? qualifiedCount / (qualifiedCount + assertiveCount)
            : 0.5;
        
        // Check for claims without evidence
        const claimPatterns = [
            /studies show/gi, /research indicates/gi, /experts say/gi,
            /according to/gi, /evidence suggests/gi, /data shows/gi
        ];
        let unsubstantiatedClaims = 0;
        for (const pattern of claimPatterns) {
            if (pattern.test(text)) {
                // Check if followed by specifics
                const match = text.match(pattern);
                if (match) {
                    const index = text.indexOf(match[0]);
                    const following = text.slice(index, index + 100);
                    // If no specific citation or number follows, it's vague
                    if (!/\d{4}|[A-Z][a-z]+\s+et al|university|journal/i.test(following)) {
                        unsubstantiatedClaims++;
                    }
                }
            }
        }
        
        // Smoothness score: high qualification + vague claims = smooth but empty
        const smoothnessScore = (
            Utils.normalize(qualificationRatio, 0.3, 0.7) * 0.5 +
            Utils.normalize(unsubstantiatedClaims, 0, 3) * 0.5
        );

        return {
            qualifiedStatements: qualifiedCount,
            assertiveStatements: assertiveCount,
            qualificationRatio: qualificationRatio.toFixed(2),
            unsubstantiatedClaims,
            smoothnessScore
        };
    },

    /**
     * Analyze stance and commitment
     */
    analyzeStance(text, sentences) {
        const lower = text.toLowerCase();
        
        // Count non-committal phrases
        let noncommittalCount = 0;
        const noncommittalFound = [];
        
        for (const phrase of this.safeNoncommittal) {
            if (lower.includes(phrase.toLowerCase())) {
                noncommittalCount++;
                noncommittalFound.push(phrase);
            }
        }
        
        // Count strong stance markers
        const strongStance = [
            'i strongly believe', 'i am convinced', 'without a doubt',
            'absolutely', 'undeniably', 'clearly wrong', 'clearly right',
            'must be', 'cannot be', 'should never', 'should always',
            'i disagree', 'i agree completely', 'that\'s ridiculous',
            'that\'s wrong', 'that\'s right', 'obviously'
        ];
        
        let strongStanceCount = 0;
        for (const phrase of strongStance) {
            if (lower.includes(phrase)) {
                strongStanceCount++;
            }
        }
        
        // Check for argument balance (AI tends to be too balanced)
        const proPatterns = /\b(advantage|benefit|positive|good|helpful|useful)\b/gi;
        const conPatterns = /\b(disadvantage|drawback|negative|bad|harmful|problematic)\b/gi;
        const proCount = (text.match(proPatterns) || []).length;
        const conCount = (text.match(conPatterns) || []).length;
        
        const isOverBalanced = proCount > 0 && conCount > 0 && 
            Math.abs(proCount - conCount) <= 1;
        
        // Non-committal score
        const noncommittalScore = (
            Utils.normalize(noncommittalCount, 0, 4) * 0.4 +
            (1 - Utils.normalize(strongStanceCount, 0, 3)) * 0.3 +
            (isOverBalanced ? 0.3 : 0)
        );

        return {
            noncommittalCount,
            noncommittalFound: noncommittalFound.slice(0, 5),
            strongStanceCount,
            argumentBalance: {
                pro: proCount,
                con: conCount,
                isOverBalanced
            },
            noncommittalScore
        };
    },

    /**
     * Generate findings
     */
    generateFindings(depthAnalysis, experienceAnalysis, groundingAnalysis, stanceAnalysis) {
        const findings = [];

        // Generality
        if (depthAnalysis.genericPhraseCount > 3) {
            findings.push({
                label: 'Content Specificity',
                value: 'High use of generic phrases',
                note: `Examples: ${depthAnalysis.genericFound.slice(0, 2).join(', ')}`,
                indicator: 'ai'
            });
        }

        // Concrete details
        if (depthAnalysis.concreteDetailCount > 3) {
            findings.push({
                label: 'Concrete Details',
                value: 'Contains specific facts and details',
                note: `Examples: ${depthAnalysis.concreteExamples.slice(0, 2).join(', ')}`,
                indicator: 'human'
            });
        }

        // Personal experience
        if (experienceAnalysis.experienceScore > 0.4) {
            findings.push({
                label: 'Personal Experience',
                value: 'Contains personal/experiential language',
                note: `${experienceAnalysis.anecdoteCount} personal anecdote(s) detected`,
                indicator: 'human'
            });
        } else if (experienceAnalysis.experienceMarkerCount === 0 && experienceAnalysis.anecdoteCount === 0) {
            findings.push({
                label: 'Personal Experience',
                value: 'No personal experience markers',
                note: 'Text lacks first-person experiential content',
                indicator: 'ai'
            });
        }

        // Stance
        if (stanceAnalysis.noncommittalScore > 0.5) {
            findings.push({
                label: 'Stance/Commitment',
                value: 'Non-committal, balanced perspective',
                note: 'Avoids taking strong positions',
                indicator: 'ai'
            });
        }

        if (stanceAnalysis.strongStanceCount > 2) {
            findings.push({
                label: 'Strong Opinions',
                value: 'Contains definitive stance markers',
                note: 'Strong opinions are more common in human writing',
                indicator: 'human'
            });
        }

        // Grounding
        if (groundingAnalysis.unsubstantiatedClaims > 1) {
            findings.push({
                label: 'Unsubstantiated Claims',
                value: 'Vague appeals to authority',
                note: '"Studies show" without citation is an AI pattern',
                indicator: 'ai'
            });
        }

        return findings;
    },

    calculateConfidence(tokenCount) {
        if (tokenCount < 50) return 0.3;
        if (tokenCount < 100) return 0.5;
        if (tokenCount < 200) return 0.7;
        if (tokenCount < 500) return 0.85;
        return 0.9;
    },

    getEmptyResult() {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            details: {},
            findings: [],
            scores: {}
        };
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SemanticAnalyzer;
}
