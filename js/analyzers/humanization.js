/**
 * VERITAS — Humanization Artifact Detector
 * Category 14: Detects patterns left by AI humanization/bypass tools
 * 
 * KEY INSIGHT: Humanizers inject specific artifacts to bypass detection:
 * - Forced contractions distributed unnaturally evenly
 * - Sentence-start disfluencies ("Well," "So," "I mean,")
 * - Deliberate typos from a common list
 * - Slang mixed with formal vocabulary (register mixing)
 * - AI phrases remaining alongside human markers
 */

const HumanizationAnalyzer = {
    name: 'Humanization Artifacts',
    category: 14,
    weight: 1.4,

    // Sentence-start disfluencies (humanizers love to add these)
    sentenceStartDisfluencies: [
        'well,', 'so,', 'i mean,', 'you know,', 'like,', 'basically,',
        'honestly,', 'actually,', 'look,', 'see,', 'thing is,', 'okay so,',
        'right,', 'anyway,', 'anyhow,', 'now,', 'okay,', 'alright,'
    ],

    // Informal contractions humanizers inject
    informalContractions: [
        "gonna", "wanna", "gotta", "kinda", "sorta", "outta", "'cause", "'bout",
        "dunno", "gimme", "lemme", "coulda", "woulda", "shoulda", "mighta", "oughta"
    ],

    // Standard contractions
    standardContractions: [
        "i'm", "i've", "i'll", "i'd", "you're", "you've", "you'll", "you'd",
        "he's", "she's", "it's", "we're", "we've", "we'll", "we'd",
        "they're", "they've", "they'll", "they'd", "isn't", "aren't",
        "wasn't", "weren't", "hasn't", "haven't", "hadn't", "won't",
        "wouldn't", "don't", "doesn't", "didn't", "can't", "couldn't",
        "shouldn't", "mightn't", "mustn't", "let's", "that's", "who's",
        "what's", "where's", "there's", "here's", "could've", "would've",
        "should've", "might've", "must've"
    ],

    // Common deliberate typos humanizers inject
    commonTypos: [
        "teh", "hte", "adn", "nad", "taht", "tht", "ahve", "hvae",
        "wiht", "wtih", "tihs", "thsi", "form", "fomr", "tehy", "thye",
        "bene", "thier", "ther", "woudl", "owuld", "abuot", "abotu",
        "coudl", "cuold", "whcih", "wich", "theer", "tehre", "wehre",
        "becuase", "beacuse", "somethign", "soemthing", "realy", "raelly",
        "definately", "definitly", "porbably", "probabl", "diferent",
        "diffrent", "trough", "thru", "acutally", "actualy", "rly", "def",
        "prob", "bc", "b/c", "smth", "sth", "tmrw", "tmr"
    ],

    // Slang words
    slangWords: [
        "cool", "awesome", "dope", "sick", "fire", "lit", "legit",
        "vibe", "vibes", "mood", "lowkey", "highkey", "cap", "bet",
        "slay", "stan", "flex", "sus", "bro", "bruh", "dude", "fam",
        "tbh", "ngl", "imo", "imho", "lol", "lmao", "omg", "wtf", "idk", "ikr", "smh"
    ],

    // Formal words that shouldn't coexist with slang
    formalWords: [
        "utilize", "facilitate", "implement", "comprehensive",
        "significant", "demonstrate", "indicate", "furthermore",
        "moreover", "therefore", "consequently", "nevertheless",
        "notwithstanding", "aforementioned", "henceforth", "wherein"
    ],

    // AI phrases that often remain after humanization
    aiPhrases: [
        "it is important to note", "it's important to note",
        "it is worth noting", "it's worth noting",
        "furthermore", "moreover", "additionally",
        "in conclusion", "to summarize", "in summary",
        "this demonstrates", "this illustrates",
        "plays a pivotal role", "plays a crucial role",
        "a myriad of", "a plethora of", "a multitude of",
        "delve into", "delve deeper", "in the realm of"
    ],

    // Rhetorical questions humanizers add
    rhetoricalQuestions: [
        "right?", "you know?", "make sense?", "get it?",
        "don't you think?", "wouldn't you agree?", "isn't it?",
        "crazy, right?", "wild, huh?", "fair enough?"
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text);
        
        if (tokens.length < 20) {
            return this.getEmptyResult();
        }

        // Analyze contraction distribution
        const contractionAnalysis = this.analyzeContractionDistribution(text, sentences);
        
        // Analyze sentence-start disfluencies
        const disfluencyAnalysis = this.analyzeSentenceStartDisfluencies(sentences);
        
        // Detect deliberate typos
        const typoAnalysis = this.analyzeDeliberateTypos(tokens);
        
        // Detect register mixing
        const registerMixing = this.analyzeRegisterMixing(tokens);
        
        // Detect AI + human marker co-occurrence
        const cooccurrenceAnalysis = this.analyzeCooccurrence(text, tokens);
        
        // Analyze rhetorical question injection
        const rhetoricalAnalysis = this.analyzeRhetoricalQuestions(text, sentences);

        // Calculate scores
        const scores = {
            contractionArtifacts: contractionAnalysis.artifactScore,
            disfluencyInjection: disfluencyAnalysis.injectionScore,
            deliberateTypos: typoAnalysis.suspicionScore,
            registerMixing: registerMixing.mixingScore,
            aiHumanCooccurrence: cooccurrenceAnalysis.cooccurrenceScore,
            rhetoricalInjection: rhetoricalAnalysis.injectionScore
        };

        // Humanized AI probability
        const humanizationScore = Utils.weightedAverage(
            [scores.contractionArtifacts, scores.disfluencyInjection, scores.deliberateTypos,
             scores.registerMixing, scores.aiHumanCooccurrence, scores.rhetoricalInjection],
            [0.2, 0.15, 0.2, 0.15, 0.2, 0.1]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability: humanizationScore,
            humanizationScore,
            confidence: this.calculateConfidence(tokens.length, sentences.length),
            details: {
                contractionAnalysis,
                disfluencyAnalysis,
                typoAnalysis,
                registerMixing,
                cooccurrenceAnalysis,
                rhetoricalAnalysis
            },
            findings: this.generateFindings(scores, contractionAnalysis, disfluencyAnalysis, typoAnalysis, registerMixing, cooccurrenceAnalysis),
            scores
        };
    },

    /**
     * Analyze contraction distribution
     * Humanizers distribute contractions evenly; humans cluster them
     */
    analyzeContractionDistribution(text, sentences) {
        const lowerText = text.toLowerCase();
        
        // Count all contractions
        let totalContractions = 0;
        const allContractions = [...this.standardContractions, ...this.informalContractions];
        
        for (const c of allContractions) {
            const regex = new RegExp(`\\b${c.replace("'", "['']?")}\\b`, 'gi');
            const matches = lowerText.match(regex);
            if (matches) totalContractions += matches.length;
        }
        
        // Count informal contractions specifically
        let informalCount = 0;
        for (const c of this.informalContractions) {
            const regex = new RegExp(`\\b${c}\\b`, 'gi');
            const matches = lowerText.match(regex);
            if (matches) informalCount += matches.length;
        }
        
        // Analyze distribution across sentences
        const contractionsPerSentence = sentences.map(sent => {
            let count = 0;
            const lowerSent = sent.toLowerCase();
            for (const c of allContractions) {
                const regex = new RegExp(`\\b${c.replace("'", "['']?")}\\b`, 'gi');
                const matches = lowerSent.match(regex);
                if (matches) count += matches.length;
            }
            return count;
        });
        
        // Calculate distribution uniformity (low std = humanizer, high std = human)
        const mean = contractionsPerSentence.reduce((a, b) => a + b, 0) / contractionsPerSentence.length;
        const variance = contractionsPerSentence.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / contractionsPerSentence.length;
        const std = Math.sqrt(variance);
        const cv = mean > 0 ? std / mean : 0;
        
        // Low CV with high contraction count = suspicious (humanizer distributes evenly)
        const hasHighContractions = totalContractions > sentences.length * 0.3;
        const hasLowVariance = cv < 0.5;
        const artifactScore = hasHighContractions && hasLowVariance ? 
            Utils.normalize(1 - cv, 0, 1) * 0.7 + (informalCount > 2 ? 0.3 : 0) : 0;
        
        return {
            totalContractions,
            informalCount,
            distributionCV: cv,
            isUniformlyDistributed: hasLowVariance && hasHighContractions,
            artifactScore: Math.min(1, artifactScore)
        };
    },

    /**
     * Analyze sentence-start disfluencies
     * Humanizers often add "Well," "So," "I mean," to start sentences
     */
    analyzeSentenceStartDisfluencies(sentences) {
        let disfluencyCount = 0;
        const found = [];
        
        for (const sent of sentences) {
            const lowerSent = sent.toLowerCase().trim();
            for (const disfluency of this.sentenceStartDisfluencies) {
                if (lowerSent.startsWith(disfluency)) {
                    disfluencyCount++;
                    found.push(disfluency);
                    break;
                }
            }
        }
        
        const ratio = disfluencyCount / sentences.length;
        
        // High ratio of sentences starting with disfluencies = suspicious
        // Natural human text has ~5-15% sentences with these; humanizers often go to 30%+
        const isExcessive = ratio > 0.25;
        const injectionScore = ratio > 0.15 ? Utils.normalize(ratio, 0.15, 0.5) : 0;
        
        return {
            count: disfluencyCount,
            ratio,
            found: [...new Set(found)],
            isExcessive,
            injectionScore: Math.min(1, injectionScore)
        };
    },

    /**
     * Detect deliberate typos from common humanizer lists
     */
    analyzeDeliberateTypos(tokens) {
        let typoCount = 0;
        const foundTypos = [];
        
        for (const token of tokens) {
            const lower = token.toLowerCase();
            if (this.commonTypos.includes(lower)) {
                typoCount++;
                foundTypos.push(lower);
            }
        }
        
        const typoRate = typoCount / tokens.length * 100;
        
        // Any known humanizer typo is suspicious
        // Multiple known typos = very suspicious
        const suspicionScore = typoCount === 0 ? 0 :
            typoCount === 1 ? 0.4 :
            typoCount === 2 ? 0.7 :
            0.9;
        
        return {
            count: typoCount,
            typoRate,
            found: [...new Set(foundTypos)],
            suspicionScore
        };
    },

    /**
     * Analyze register mixing (formal + slang coexistence)
     * Natural text rarely mixes academic vocabulary with slang
     */
    analyzeRegisterMixing(tokens) {
        let formalCount = 0;
        let slangCount = 0;
        
        for (const token of tokens) {
            const lower = token.toLowerCase();
            if (this.formalWords.includes(lower)) formalCount++;
            if (this.slangWords.includes(lower)) slangCount++;
        }
        
        const formalRatio = formalCount / tokens.length;
        const slangRatio = slangCount / tokens.length;
        
        // High values of both = suspicious mixing
        const product = formalCount * slangCount;
        const hasBothRegisters = formalCount >= 2 && slangCount >= 2;
        
        const mixingScore = hasBothRegisters ? 
            Utils.normalize(product, 0, 20) : 0;
        
        return {
            formalCount,
            slangCount,
            formalRatio,
            slangRatio,
            hasBothRegisters,
            mixingScore: Math.min(1, mixingScore)
        };
    },

    /**
     * Detect AI + human marker co-occurrence
     * AI phrases remaining alongside injected human markers = humanized
     */
    analyzeCooccurrence(text, tokens) {
        const lowerText = text.toLowerCase();
        
        // Count AI phrases
        let aiPhraseCount = 0;
        const foundAiPhrases = [];
        for (const phrase of this.aiPhrases) {
            if (lowerText.includes(phrase)) {
                aiPhraseCount++;
                foundAiPhrases.push(phrase);
            }
        }
        
        // Count human markers (informal contractions + slang + disfluencies)
        let humanMarkerCount = 0;
        for (const token of tokens) {
            const lower = token.toLowerCase();
            if (this.informalContractions.includes(lower) || this.slangWords.includes(lower)) {
                humanMarkerCount++;
            }
        }
        
        // Count sentence-start disfluencies
        const sentences = Utils.splitSentences(text);
        for (const sent of sentences) {
            const lowerSent = sent.toLowerCase().trim();
            for (const d of this.sentenceStartDisfluencies) {
                if (lowerSent.startsWith(d)) {
                    humanMarkerCount++;
                    break;
                }
            }
        }
        
        // Both AI and human markers present in significant amounts = humanized
        const hasBoth = aiPhraseCount >= 1 && humanMarkerCount >= 3;
        const cooccurrenceScore = hasBoth ? 
            Utils.normalize(aiPhraseCount * humanMarkerCount, 0, 15) : 0;
        
        return {
            aiPhraseCount,
            humanMarkerCount,
            foundAiPhrases,
            hasSuspiciousCooccurrence: hasBoth,
            cooccurrenceScore: Math.min(1, cooccurrenceScore)
        };
    },

    /**
     * Analyze rhetorical question injection
     */
    analyzeRhetoricalQuestions(text, sentences) {
        const lowerText = text.toLowerCase();
        let count = 0;
        const found = [];
        
        for (const q of this.rhetoricalQuestions) {
            if (lowerText.includes(q)) {
                count++;
                found.push(q);
            }
        }
        
        const ratio = count / sentences.length;
        
        // Multiple rhetorical questions in short text = suspicious
        const injectionScore = count >= 2 ? Utils.normalize(count, 2, 5) : 0;
        
        return {
            count,
            ratio,
            found,
            injectionScore: Math.min(1, injectionScore)
        };
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(tokenCount, sentenceCount) {
        const lengthConfidence = Utils.normalize(tokenCount, 50, 500);
        const sentenceConfidence = Utils.normalize(sentenceCount, 5, 30);
        return (lengthConfidence + sentenceConfidence) / 2;
    },

    /**
     * Generate findings
     */
    generateFindings(scores, contractionAnalysis, disfluencyAnalysis, typoAnalysis, registerMixing, cooccurrenceAnalysis) {
        const findings = [];
        
        if (contractionAnalysis.isUniformlyDistributed) {
            findings.push({
                type: 'humanization',
                severity: 'medium',
                message: `Contractions unusually evenly distributed (CV: ${contractionAnalysis.distributionCV.toFixed(2)})`,
                indicator: 'uniformContractions'
            });
        }
        
        if (contractionAnalysis.informalCount > 2) {
            findings.push({
                type: 'humanization',
                severity: 'low',
                message: `${contractionAnalysis.informalCount} informal contractions (gonna, wanna, etc.)`,
                indicator: 'informalContractions'
            });
        }
        
        if (disfluencyAnalysis.isExcessive) {
            findings.push({
                type: 'humanization',
                severity: 'high',
                message: `${Math.round(disfluencyAnalysis.ratio * 100)}% of sentences start with disfluencies`,
                indicator: 'sentenceStartDisfluencies'
            });
        }
        
        if (typoAnalysis.count > 0) {
            findings.push({
                type: 'humanization',
                severity: typoAnalysis.count > 1 ? 'high' : 'medium',
                message: `Found ${typoAnalysis.count} known humanizer typo(s): ${typoAnalysis.found.join(', ')}`,
                indicator: 'deliberateTypos'
            });
        }
        
        if (registerMixing.hasBothRegisters) {
            findings.push({
                type: 'humanization',
                severity: 'medium',
                message: `Register mixing detected: ${registerMixing.formalCount} formal + ${registerMixing.slangCount} slang words`,
                indicator: 'registerMixing'
            });
        }
        
        if (cooccurrenceAnalysis.hasSuspiciousCooccurrence) {
            findings.push({
                type: 'humanization',
                severity: 'high',
                message: `AI phrases coexist with ${cooccurrenceAnalysis.humanMarkerCount} human markers`,
                indicator: 'aiHumanCooccurrence'
            });
        }
        
        return findings;
    },

    /**
     * Get empty result for insufficient text
     */
    getEmptyResult() {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            humanizationScore: 0,
            confidence: 0,
            details: {},
            findings: [],
            scores: {}
        };
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HumanizationAnalyzer;
}
