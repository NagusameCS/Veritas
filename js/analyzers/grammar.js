/**
 * VERITAS — Grammar & Error Pattern Analyzer
 * Category 1: Grammar Regularity, Correction Behavior, Human-Specific Signals
 */

const GrammarAnalyzer = {
    name: 'Grammar & Error Patterns',
    category: 1,
    weight: 1.2,

    // Common grammar error patterns
    errorPatterns: {
        articles: [
            { pattern: /\b(a)\s+([aeiou])/gi, type: 'article_before_vowel' },
            { pattern: /\b(an)\s+([^aeiou\s])/gi, type: 'an_before_consonant' },
            { pattern: /\bthe\s+the\b/gi, type: 'double_article' }
        ],
        subjectVerb: [
            { pattern: /\b(he|she|it)\s+(are|were|have)\b/gi, type: 'subject_verb_disagreement' },
            { pattern: /\b(they|we|you)\s+(is|was|has)\b/gi, type: 'subject_verb_disagreement' },
            { pattern: /\b(I)\s+(is|was)\b/gi, type: 'subject_verb_disagreement' }
        ],
        tense: [
            { pattern: /\b(yesterday|last\s+\w+)\s+\w+\s+(is|are|has)\b/gi, type: 'tense_inconsistency' },
            { pattern: /\b(tomorrow|next\s+\w+)\s+\w+\s+(was|were|had)\b/gi, type: 'tense_inconsistency' }
        ],
        prepositions: [
            { pattern: /\b(arrive)\s+(to)\b/gi, type: 'preposition_error' },
            { pattern: /\b(different)\s+(than)\b/gi, type: 'preposition_preference' },
            { pattern: /\b(comprised)\s+(of)\b/gi, type: 'preposition_error' }
        ],
        punctuation: [
            { pattern: /\s+[,.:;!?]/g, type: 'space_before_punctuation' },
            { pattern: /[,.:;!?]{2,}/g, type: 'repeated_punctuation' },
            { pattern: /[a-z]\.[A-Z]/g, type: 'missing_space_after_period' }
        ]
    },

    // Over-formal constructions (AI indicator)
    formalConstructions: [
        /\bit is important to note that\b/gi,
        /\bit should be noted that\b/gi,
        /\bit is worth mentioning that\b/gi,
        /\bone must consider\b/gi,
        /\bit is essential to\b/gi,
        /\bit is crucial to understand\b/gi,
        /\bthis serves to illustrate\b/gi,
        /\bit is imperative that\b/gi,
        /\bit bears mentioning\b/gi,
        /\bit is of paramount importance\b/gi
    ],

    // Self-correction patterns (human indicator)
    selfCorrectionPatterns: [
        /\b(I mean|that is|or rather|well actually|actually)\b/gi,
        /\b(no wait|scratch that|let me rephrase)\b/gi,
        /—\s*\w+/g, // em-dash interruptions
        /\.\.\./g   // trailing off
    ],

    // Repeated grammatical constructions
    repeatedConstructionPatterns: [
        /^(The\s+\w+)\s/gm,
        /^(It\s+is)\s/gm,
        /^(This\s+\w+)\s/gm,
        /^(There\s+are)\s/gm
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text);
        const tokenCount = tokens.length;
        
        if (tokenCount === 0) {
            return this.getEmptyResult();
        }

        // Analyze different aspects
        const errorAnalysis = this.analyzeErrors(text, tokenCount);
        const formalityAnalysis = this.analyzeFormalConstructions(text, sentences.length);
        const correctionAnalysis = this.analyzeSelfCorrections(text);
        const repetitionAnalysis = this.analyzeRepeatedConstructions(text, sentences);

        // Calculate overall AI probability for this category
        // High error rate = more human, Low error rate = more AI
        // High formality = more AI
        // Self-corrections = more human
        // Repeated constructions = more AI

        const scores = {
            errorRate: 1 - Utils.normalize(errorAnalysis.errorRate, 0, 10), // Low errors = AI-like
            formality: formalityAnalysis.formalityScore, // High formality = AI-like
            selfCorrection: 1 - correctionAnalysis.correctionScore, // No corrections = AI-like
            repetition: repetitionAnalysis.repetitionScore // High repetition = AI-like
        };

        const aiProbability = Utils.weightedAverage(
            [scores.errorRate, scores.formality, scores.selfCorrection, scores.repetition],
            [0.3, 0.25, 0.25, 0.2]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability: aiProbability,
            confidence: this.calculateConfidence(tokenCount),
            details: {
                errorAnalysis,
                formalityAnalysis,
                correctionAnalysis,
                repetitionAnalysis
            },
            findings: this.generateFindings(errorAnalysis, formalityAnalysis, correctionAnalysis, repetitionAnalysis),
            scores
        };
    },

    /**
     * Analyze grammar errors
     */
    analyzeErrors(text, tokenCount) {
        const errors = [];
        let totalErrors = 0;

        for (const [category, patterns] of Object.entries(this.errorPatterns)) {
            for (const { pattern, type } of patterns) {
                const matches = text.match(pattern);
                if (matches) {
                    errors.push({
                        category,
                        type,
                        count: matches.length,
                        examples: matches.slice(0, 3)
                    });
                    totalErrors += matches.length;
                }
            }
        }

        // Error rate per 1000 tokens
        const errorRate = tokenCount > 0 ? (totalErrors / tokenCount) * 1000 : 0;

        return {
            totalErrors,
            errorRate: errorRate,
            errorsByCategory: errors,
            hasIdiosyncraticErrors: errors.some(e => e.count === 1) // Unique errors are more human
        };
    },

    /**
     * Analyze formal/over-correct constructions
     */
    analyzeFormalConstructions(text, sentenceCount) {
        let formalCount = 0;
        const matches = [];

        for (const pattern of this.formalConstructions) {
            const found = text.match(pattern);
            if (found) {
                formalCount += found.length;
                matches.push(...found);
            }
        }

        // Score: frequency of formal constructions relative to sentence count
        const formalityScore = sentenceCount > 0 
            ? Utils.normalize(formalCount / sentenceCount, 0, 0.3)
            : 0;

        return {
            formalConstructionCount: formalCount,
            formalityScore,
            examples: [...new Set(matches)].slice(0, 5)
        };
    },

    /**
     * Analyze self-corrections (human indicator)
     */
    analyzeSelfCorrections(text) {
        let correctionCount = 0;
        const corrections = [];

        for (const pattern of this.selfCorrectionPatterns) {
            const matches = text.match(pattern);
            if (matches) {
                correctionCount += matches.length;
                corrections.push(...matches);
            }
        }

        // More corrections = more human-like
        const correctionScore = Utils.normalize(correctionCount, 0, 5);

        return {
            correctionCount,
            correctionScore,
            examples: corrections.slice(0, 5)
        };
    },

    /**
     * Analyze repeated grammatical constructions
     */
    analyzeRepeatedConstructions(text, sentences) {
        const sentenceStarts = sentences.map(s => {
            const words = s.split(/\s+/).slice(0, 3);
            return words.join(' ').toLowerCase();
        });

        const startFrequency = Utils.frequencyDistribution(sentenceStarts);
        const repeatedStarts = Object.entries(startFrequency)
            .filter(([_, count]) => count > 1)
            .sort((a, b) => b[1] - a[1]);

        const maxRepetition = repeatedStarts.length > 0 ? repeatedStarts[0][1] : 0;
        const repetitionRatio = sentences.length > 0 
            ? repeatedStarts.reduce((sum, [_, count]) => sum + count, 0) / sentences.length
            : 0;

        // High repetition of sentence structures = AI-like
        const repetitionScore = Utils.normalize(repetitionRatio, 0, 0.5);

        return {
            repeatedStarts: repeatedStarts.slice(0, 5),
            maxRepetition,
            repetitionRatio,
            repetitionScore
        };
    },

    /**
     * Generate human-readable findings
     */
    generateFindings(errorAnalysis, formalityAnalysis, correctionAnalysis, repetitionAnalysis) {
        const findings = [];

        // Error rate finding
        if (errorAnalysis.errorRate < 1) {
            findings.push({
                label: 'Error Rate',
                value: 'Very low grammatical error rate',
                note: 'AI-generated text typically has near-perfect grammar',
                indicator: 'ai'
            });
        } else if (errorAnalysis.errorRate > 5) {
            findings.push({
                label: 'Error Rate',
                value: 'Higher than average error rate',
                note: 'Minor errors may indicate human writing',
                indicator: 'human'
            });
        }

        // Formality finding
        if (formalityAnalysis.formalConstructionCount > 0) {
            findings.push({
                label: 'Formal Constructions',
                value: `${formalityAnalysis.formalConstructionCount} overly formal phrase(s) detected`,
                note: `Examples: ${formalityAnalysis.examples.slice(0, 2).join(', ')}`,
                indicator: 'ai'
            });
        }

        // Self-correction finding
        if (correctionAnalysis.correctionCount > 0) {
            findings.push({
                label: 'Self-Corrections',
                value: `${correctionAnalysis.correctionCount} self-correction(s) found`,
                note: 'Self-corrections suggest human writing process',
                indicator: 'human'
            });
        }

        // Repetition finding
        if (repetitionAnalysis.repetitionScore > 0.3) {
            findings.push({
                label: 'Structural Repetition',
                value: 'Repeated sentence structures detected',
                note: 'Uniform patterns may indicate AI generation',
                indicator: 'ai'
            });
        }

        return findings;
    },

    /**
     * Calculate confidence based on text length
     */
    calculateConfidence(tokenCount) {
        if (tokenCount < 50) return 0.3;
        if (tokenCount < 100) return 0.5;
        if (tokenCount < 200) return 0.7;
        if (tokenCount < 500) return 0.85;
        return 0.95;
    },

    /**
     * Return empty result for empty text
     */
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

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = GrammarAnalyzer;
}
