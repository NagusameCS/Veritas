/**
 * VERITAS — Part of Speech Pattern Analyzer
 * Analyzes verb, adverb, adjective patterns for AI detection
 * 
 * AI text often shows predictable verb/adverb patterns:
 * - Overuse of certain adverbs (clearly, certainly, importantly, significantly)
 * - Verb tense consistency that's too perfect
 * - Hedging verb patterns (may, might, could, would)
 * - Adverb-verb collocations that are formulaic
 */

const PartOfSpeechAnalyzer = {
    name: 'Part of Speech Patterns',
    category: 14,
    weight: 1.0,

    // Common AI-overused adverbs
    aiAdverbs: [
        'certainly', 'clearly', 'importantly', 'significantly', 'essentially',
        'fundamentally', 'particularly', 'specifically', 'generally', 'typically',
        'ultimately', 'primarily', 'notably', 'remarkably', 'considerably',
        'substantially', 'effectively', 'consequently', 'accordingly', 'furthermore',
        'moreover', 'additionally', 'subsequently', 'previously', 'currently',
        'increasingly', 'undoubtedly', 'inevitably', 'inherently', 'seemingly'
    ],

    // Hedging verbs common in AI
    hedgingVerbs: [
        'may', 'might', 'could', 'would', 'should', 'can',
        'appears', 'seems', 'tends', 'suggests', 'indicates',
        'implies', 'represents', 'demonstrates', 'illustrates'
    ],

    // AI-typical verb phrases
    aiVerbPhrases: [
        'it is important to', 'it is essential to', 'it is worth noting',
        'it should be noted', 'it is crucial to', 'it is necessary to',
        'plays a crucial role', 'plays an important role', 'serves as',
        'contributes to', 'leads to', 'results in', 'enables us to',
        'allows us to', 'helps to', 'aims to', 'seeks to', 'strives to',
        'is designed to', 'is intended to', 'is meant to', 'is used to',
        'can be seen', 'can be found', 'can be used', 'can be applied',
        'has been shown', 'has been demonstrated', 'has been established',
        'there is a need', 'there is evidence', 'there are several',
        'it is clear that', 'it is evident that', 'it is apparent that'
    ],

    // Adverb positions (AI tends to front-load)
    adverbPositions: {
        initial: 0,    // At sentence start
        preverbal: 0,  // Before main verb
        postverbal: 0, // After main verb
        final: 0       // At sentence end
    },

    // Common human-like verb patterns
    humanVerbPatterns: [
        'i think', 'i believe', 'i feel', 'i guess', 'i suppose',
        'honestly', 'frankly', 'actually', 'basically', 'literally',
        'kinda', 'sorta', 'gonna', 'wanna', 'gotta',
        'you know', 'i mean', 'like i said'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        if (!text || text.length < 50) {
            return this.getEmptyResult();
        }

        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text.toLowerCase());
        
        const adverbAnalysis = this.analyzeAdverbs(text, sentences);
        const verbAnalysis = this.analyzeVerbs(text, sentences);
        const patternAnalysis = this.analyzeVerbPhrases(text);
        const positionAnalysis = this.analyzeAdverbPositions(sentences);
        const tenseAnalysis = this.analyzeTenseConsistency(sentences);

        // Calculate uniformity scores
        const uniformityScores = {
            adverbDistribution: adverbAnalysis.uniformityScore,
            verbPatterns: verbAnalysis.uniformityScore,
            tenseConsistency: tenseAnalysis.uniformityScore,
            adverbPosition: positionAnalysis.uniformityScore
        };

        // Calculate AI probability
        let aiProbability = 0.5;
        
        // AI adverb density
        const aiAdverbRatio = adverbAnalysis.aiAdverbCount / Math.max(1, adverbAnalysis.totalAdverbs);
        if (aiAdverbRatio > 0.5) {
            aiProbability += 0.15;
        }
        
        // AI verb phrase count
        if (patternAnalysis.aiPhraseCount > 3) {
            aiProbability += Math.min(0.2, patternAnalysis.aiPhraseCount * 0.03);
        }
        
        // Tense too consistent = AI
        if (tenseAnalysis.uniformityScore > 0.9) {
            aiProbability += 0.1;
        }
        
        // Adverb front-loading (AI pattern)
        if (positionAnalysis.frontLoadingRatio > 0.6) {
            aiProbability += 0.1;
        }
        
        // Human patterns detected
        if (verbAnalysis.humanPatternCount > 2) {
            aiProbability -= 0.15;
        }
        
        // Hedging verb overuse
        if (verbAnalysis.hedgingRatio > 0.3) {
            aiProbability += 0.1;
        }

        const confidence = this.calculateConfidence(text, sentences, adverbAnalysis);
        const findings = this.generateFindings(adverbAnalysis, verbAnalysis, patternAnalysis, positionAnalysis, tenseAnalysis);

        return {
            name: this.name,
            category: this.category,
            aiProbability: Math.max(0, Math.min(1, aiProbability)),
            confidence,
            uniformityScores,
            details: {
                adverbs: adverbAnalysis,
                verbs: verbAnalysis,
                phrases: patternAnalysis,
                positions: positionAnalysis,
                tense: tenseAnalysis
            },
            findings
        };
    },

    /**
     * Analyze adverb usage patterns
     */
    analyzeAdverbs(text, sentences) {
        const lowerText = text.toLowerCase();
        const words = Utils.tokenize(lowerText);
        
        // Find all adverbs (words ending in -ly, plus common adverbs)
        const adverbPattern = /\b\w+ly\b/gi;
        const allAdverbs = text.match(adverbPattern) || [];
        
        // Count AI-typical adverbs
        let aiAdverbCount = 0;
        const foundAiAdverbs = {};
        
        for (const adv of this.aiAdverbs) {
            const regex = new RegExp(`\\b${adv}\\b`, 'gi');
            const matches = text.match(regex);
            if (matches) {
                aiAdverbCount += matches.length;
                foundAiAdverbs[adv] = matches.length;
            }
        }

        // Calculate adverb density per sentence
        const adverbsPerSentence = sentences.map(s => {
            const matches = s.match(adverbPattern);
            return matches ? matches.length : 0;
        });

        // Uniformity of adverb distribution
        const uniformityScore = adverbsPerSentence.length > 0 
            ? 1 - (Utils.standardDeviation(adverbsPerSentence) / (Utils.mean(adverbsPerSentence) + 0.1))
            : 0.5;

        return {
            totalAdverbs: allAdverbs.length,
            aiAdverbCount,
            foundAiAdverbs,
            adverbDensity: allAdverbs.length / Math.max(1, words.length),
            adverbsPerSentence: Utils.mean(adverbsPerSentence),
            uniformityScore: Math.max(0, Math.min(1, uniformityScore))
        };
    },

    /**
     * Analyze verb patterns
     */
    analyzeVerbs(text, sentences) {
        const lowerText = text.toLowerCase();
        
        // Count hedging verbs
        let hedgingCount = 0;
        const foundHedging = {};
        
        for (const verb of this.hedgingVerbs) {
            const regex = new RegExp(`\\b${verb}\\b`, 'gi');
            const matches = text.match(regex);
            if (matches) {
                hedgingCount += matches.length;
                foundHedging[verb] = matches.length;
            }
        }

        // Count human-like patterns
        let humanPatternCount = 0;
        const foundHumanPatterns = {};
        
        for (const pattern of this.humanVerbPatterns) {
            const regex = new RegExp(pattern, 'gi');
            const matches = text.match(regex);
            if (matches) {
                humanPatternCount += matches.length;
                foundHumanPatterns[pattern] = matches.length;
            }
        }

        // Estimate total verbs (rough approximation)
        const verbPatterns = /\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|shall|should|may|might|can|could|must|\w+ed|\w+ing|\w+s)\b/gi;
        const allVerbs = text.match(verbPatterns) || [];

        const hedgingRatio = hedgingCount / Math.max(1, allVerbs.length);
        
        // Calculate verb usage uniformity
        const verbsPerSentence = sentences.map(s => {
            const matches = s.match(verbPatterns);
            return matches ? matches.length : 0;
        });
        
        const uniformityScore = verbsPerSentence.length > 0
            ? 1 - Math.min(1, Utils.standardDeviation(verbsPerSentence) / 3)
            : 0.5;

        return {
            hedgingCount,
            foundHedging,
            humanPatternCount,
            foundHumanPatterns,
            hedgingRatio,
            uniformityScore
        };
    },

    /**
     * Analyze AI-typical verb phrases
     */
    analyzeVerbPhrases(text) {
        const lowerText = text.toLowerCase();
        let aiPhraseCount = 0;
        const foundPhrases = {};

        for (const phrase of this.aiVerbPhrases) {
            const regex = new RegExp(phrase.replace(/\s+/g, '\\s+'), 'gi');
            const matches = text.match(regex);
            if (matches) {
                aiPhraseCount += matches.length;
                foundPhrases[phrase] = matches.length;
            }
        }

        // Calculate phrase density
        const wordCount = Utils.tokenize(text).length;
        const phraseDensity = aiPhraseCount / Math.max(1, wordCount / 100);

        return {
            aiPhraseCount,
            foundPhrases,
            phraseDensity
        };
    },

    /**
     * Analyze adverb positions in sentences
     */
    analyzeAdverbPositions(sentences) {
        const positions = { initial: 0, preverbal: 0, postverbal: 0, final: 0 };
        let totalPositioned = 0;

        for (const sentence of sentences) {
            const words = sentence.trim().split(/\s+/);
            if (words.length < 3) continue;

            // Check for adverbs at sentence start
            if (this.aiAdverbs.includes(words[0].toLowerCase().replace(/[^a-z]/g, ''))) {
                positions.initial++;
                totalPositioned++;
            }

            // Check for adverbs ending in -ly
            for (let i = 0; i < words.length; i++) {
                const word = words[i].toLowerCase().replace(/[^a-z]/g, '');
                if (word.endsWith('ly') && word.length > 4) {
                    if (i === 0) {
                        // Already counted
                    } else if (i < words.length / 3) {
                        positions.preverbal++;
                        totalPositioned++;
                    } else if (i >= words.length - 2) {
                        positions.final++;
                        totalPositioned++;
                    } else {
                        positions.postverbal++;
                        totalPositioned++;
                    }
                }
            }
        }

        // Front-loading ratio (AI tends to put adverbs at start)
        const frontLoadingRatio = totalPositioned > 0 
            ? (positions.initial + positions.preverbal) / totalPositioned 
            : 0.5;

        // Uniformity of positions (AI tends to be consistent)
        const positionValues = Object.values(positions);
        const uniformityScore = positionValues.some(v => v > 0)
            ? 1 - (Utils.standardDeviation(positionValues) / (Utils.mean(positionValues) + 0.1))
            : 0.5;

        return {
            positions,
            totalPositioned,
            frontLoadingRatio,
            uniformityScore: Math.max(0, Math.min(1, uniformityScore))
        };
    },

    /**
     * Analyze verb tense consistency
     */
    analyzeTenseConsistency(sentences) {
        let pastCount = 0;
        let presentCount = 0;
        let futureCount = 0;

        const pastPattern = /\b(was|were|had|did|\w+ed)\b/gi;
        const presentPattern = /\b(is|are|am|has|have|do|does|\w+s)\b/gi;
        const futurePattern = /\b(will|shall|going to)\b/gi;

        for (const sentence of sentences) {
            const past = (sentence.match(pastPattern) || []).length;
            const present = (sentence.match(presentPattern) || []).length;
            const future = (sentence.match(futurePattern) || []).length;

            if (past > present && past > future) pastCount++;
            else if (present > past && present > future) presentCount++;
            else if (future > 0) futureCount++;
        }

        const total = pastCount + presentCount + futureCount;
        const dominant = Math.max(pastCount, presentCount, futureCount);
        
        // High uniformity = mostly one tense = AI-like
        const uniformityScore = total > 0 ? dominant / total : 0.5;

        return {
            pastCount,
            presentCount,
            futureCount,
            dominantTense: pastCount > presentCount && pastCount > futureCount ? 'past' :
                          presentCount > futureCount ? 'present' : 'future',
            uniformityScore
        };
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(text, sentences, adverbAnalysis) {
        let confidence = 0.4;
        
        // More text = more confidence
        if (text.length > 500) confidence += 0.1;
        if (text.length > 1000) confidence += 0.1;
        if (text.length > 2000) confidence += 0.1;
        
        // More adverbs to analyze = more confidence
        if (adverbAnalysis.totalAdverbs > 5) confidence += 0.1;
        if (adverbAnalysis.totalAdverbs > 10) confidence += 0.1;
        
        return Math.min(1, confidence);
    },

    /**
     * Generate findings
     */
    generateFindings(adverbs, verbs, phrases, positions, tense) {
        const findings = [];

        // AI adverb overuse
        const topAiAdverbs = Object.entries(adverbs.foundAiAdverbs)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
        
        if (topAiAdverbs.length > 0 && adverbs.aiAdverbCount > 3) {
            const adverbList = topAiAdverbs.map(([word, count]) => `"${word}" (${count}×)`).join(', ');
            findings.push({
                text: `High frequency of AI-typical adverbs: ${adverbList}`,
                category: this.name,
                indicator: 'ai',
                severity: adverbs.aiAdverbCount > 6 ? 'high' : 'medium'
            });
        }

        // AI verb phrases
        if (phrases.aiPhraseCount > 2) {
            const topPhrases = Object.entries(phrases.foundPhrases)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 2)
                .map(([phrase]) => `"${phrase}"`)
                .join(', ');
            findings.push({
                text: `AI-typical verb phrases detected: ${topPhrases} (${phrases.aiPhraseCount} total instances)`,
                category: this.name,
                indicator: 'ai',
                severity: phrases.aiPhraseCount > 5 ? 'high' : 'medium'
            });
        }

        // Human patterns detected
        if (verbs.humanPatternCount > 0) {
            findings.push({
                text: `Human-like verb patterns detected (${verbs.humanPatternCount} instances) - suggests authentic human writing`,
                category: this.name,
                indicator: 'human',
                severity: 'medium'
            });
        }

        // Hedging overuse
        if (verbs.hedgingRatio > 0.15) {
            findings.push({
                text: `High hedging verb frequency (${Math.round(verbs.hedgingRatio * 100)}%) - common in AI-generated text`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Adverb front-loading
        if (positions.frontLoadingRatio > 0.6 && positions.totalPositioned > 3) {
            findings.push({
                text: `Adverbs concentrated at sentence beginnings (${Math.round(positions.frontLoadingRatio * 100)}%) - AI writing pattern`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Tense consistency
        if (tense.uniformityScore > 0.85) {
            findings.push({
                text: `Very consistent verb tense (${Math.round(tense.uniformityScore * 100)}% ${tense.dominantTense}) - unusually uniform for natural writing`,
                category: this.name,
                indicator: 'ai',
                severity: 'low'
            });
        }

        return findings;
    },

    /**
     * Empty result
     */
    getEmptyResult() {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            uniformityScores: {},
            details: {},
            findings: []
        };
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PartOfSpeechAnalyzer;
}
