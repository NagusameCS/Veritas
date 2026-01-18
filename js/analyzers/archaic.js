/**
 * VERITAS — Archaic / Historical Grammar Analyzer
 * Category 5: Morphological Signals, Obsolete Forms, Consistency Checks
 */

const ArchaicAnalyzer = {
    name: 'Archaic / Historical Grammar',
    category: 5,
    weight: 0.7,

    // Archaic verb forms and conjugations
    archaicVerbs: {
        'doth': 'does',
        'hath': 'has',
        'art': 'are (thou)',
        'wilt': 'will (thou)',
        'shalt': 'shall (thou)',
        'hast': 'have (thou)',
        'hadst': 'had (thou)',
        'wouldst': 'would (thou)',
        'shouldst': 'should (thou)',
        'couldst': 'could (thou)',
        'didst': 'did (thou)',
        'dost': 'do (thou)',
        'knowest': 'know (thou)',
        'speaketh': 'speaks',
        'cometh': 'comes',
        'goeth': 'goes',
        'maketh': 'makes',
        'giveth': 'gives',
        'taketh': 'takes',
        'saith': 'says',
        'sayeth': 'says'
    },

    // Archaic pronouns
    archaicPronouns: ['thou', 'thee', 'thy', 'thine', 'ye', 'wherefore', 'whence', 'thence', 'hither', 'thither', 'hence'],

    // Archaic prepositions and syntax markers
    archaicSyntax: ['unto', 'amongst', 'whilst', 'betwixt', 'ere', 'nigh', 'oft', 'perchance', 'mayhap', 'forsooth', 'verily', 'methinks', 'prithee', 'anon'],

    // Anachronistic modern terms that shouldn't appear in archaic text
    modernAnachronisms: [
        'computer', 'internet', 'email', 'phone', 'television', 'radio',
        'airplane', 'automobile', 'electricity', 'technology', 'digital',
        'online', 'website', 'software', 'database', 'algorithm',
        'basically', 'literally', 'actually', 'totally', 'definitely',
        'awesome', 'amazing', 'incredible', 'fantastic'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const tokens = Utils.tokenize(text);
        
        if (tokens.length < 20) {
            return this.getEmptyResult();
        }

        // Detect if text attempts archaic style
        const archaicDetection = this.detectArchaicStyle(text);
        
        // If not attempting archaic style, return neutral
        if (!archaicDetection.isArchaicAttempt) {
            return {
                name: this.name,
                category: this.category,
                aiProbability: 0.5, // Neutral - not applicable
                confidence: 0.3,
                details: {
                    archaicDetection,
                    note: 'Text does not attempt archaic/historical style'
                },
                findings: [],
                scores: { applicable: false }
            };
        }

        // Analyze consistency of archaic style
        const consistencyAnalysis = this.analyzeArchaicConsistency(text, tokens);
        
        // Check for anachronisms
        const anachronismAnalysis = this.detectAnachronisms(text);
        
        // Analyze morphological consistency
        const morphologyAnalysis = this.analyzeMorphologicalConsistency(text);

        // Calculate AI probability
        // Surface-level imitation without deep consistency = AI-like
        const scores = {
            surfaceLevel: 1 - consistencyAnalysis.depthScore,
            hasAnachronisms: anachronismAnalysis.anachronismScore,
            morphologyInconsistent: 1 - morphologyAnalysis.consistencyScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.surfaceLevel, scores.hasAnachronisms, scores.morphologyInconsistent],
            [0.4, 0.3, 0.3]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(tokens.length, archaicDetection.archaicDensity),
            details: {
                archaicDetection,
                consistencyAnalysis,
                anachronismAnalysis,
                morphologyAnalysis
            },
            findings: this.generateFindings(consistencyAnalysis, anachronismAnalysis, morphologyAnalysis),
            scores
        };
    },

    /**
     * Detect if text attempts archaic style
     */
    detectArchaicStyle(text) {
        const lower = text.toLowerCase();
        let archaicCount = 0;
        const archaicFound = [];

        // Check for archaic verbs
        for (const verb of Object.keys(this.archaicVerbs)) {
            if (new RegExp(`\\b${verb}\\b`, 'gi').test(lower)) {
                archaicCount++;
                archaicFound.push(verb);
            }
        }

        // Check for archaic pronouns
        for (const pronoun of this.archaicPronouns) {
            if (new RegExp(`\\b${pronoun}\\b`, 'gi').test(lower)) {
                archaicCount++;
                archaicFound.push(pronoun);
            }
        }

        // Check for archaic syntax
        for (const word of this.archaicSyntax) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                archaicCount++;
                archaicFound.push(word);
            }
        }

        const tokens = Utils.tokenize(text);
        const archaicDensity = tokens.length > 0 ? archaicCount / tokens.length : 0;
        
        // Consider it an archaic attempt if density > 1%
        const isArchaicAttempt = archaicDensity > 0.01 || archaicCount >= 3;

        return {
            isArchaicAttempt,
            archaicCount,
            archaicDensity: (archaicDensity * 100).toFixed(2) + '%',
            archaicFound: [...new Set(archaicFound)]
        };
    },

    /**
     * Analyze consistency of archaic style
     */
    analyzeArchaicConsistency(text, tokens) {
        const sentences = Utils.splitSentences(text);
        const lower = text.toLowerCase();
        
        // Check if archaic elements are distributed throughout or just surface-level
        let sentencesWithArchaic = 0;
        for (const sentence of sentences) {
            const sentenceLower = sentence.toLowerCase();
            const hasArchaic = [...Object.keys(this.archaicVerbs), ...this.archaicPronouns, ...this.archaicSyntax]
                .some(word => new RegExp(`\\b${word}\\b`).test(sentenceLower));
            if (hasArchaic) sentencesWithArchaic++;
        }
        
        const distributionRatio = sentences.length > 0 ? sentencesWithArchaic / sentences.length : 0;
        
        // Check for consistent use of 'thou' forms
        const thouCount = (lower.match(/\bthou\b/g) || []).length;
        const theeCount = (lower.match(/\bthee\b/g) || []).length;
        const thyCount = (lower.match(/\bthy\b/g) || []).length;
        const secondPersonModern = (lower.match(/\byou\b/g) || []).length;
        
        // Mixing 'thou/thee' with 'you' is inconsistent
        const thouTotal = thouCount + theeCount + thyCount;
        const secondPersonMixed = thouTotal > 0 && secondPersonModern > 0;
        
        // Check for consistent verb endings (-eth, -est)
        const ethVerbs = (text.match(/\w+eth\b/gi) || []).length;
        const estVerbs = (text.match(/\w+est\b/gi) || []).length;
        const modernVerbEndings = (text.match(/\b(does|has|is|was|goes)\b/gi) || []).length;
        
        const verbMixed = (ethVerbs > 0 || estVerbs > 0) && modernVerbEndings > (ethVerbs + estVerbs);

        // Depth score: high if archaic is consistent throughout
        const depthScore = (
            (distributionRatio * 0.4) +
            (secondPersonMixed ? 0 : 0.3) +
            (verbMixed ? 0 : 0.3)
        );

        return {
            sentencesWithArchaic,
            totalSentences: sentences.length,
            distributionRatio: distributionRatio.toFixed(2),
            pronounConsistency: {
                archaicSecondPerson: thouTotal,
                modernSecondPerson: secondPersonModern,
                isMixed: secondPersonMixed
            },
            verbConsistency: {
                archaicEndings: ethVerbs + estVerbs,
                modernForms: modernVerbEndings,
                isMixed: verbMixed
            },
            depthScore
        };
    },

    /**
     * Detect anachronisms in archaic text
     */
    detectAnachronisms(text) {
        const lower = text.toLowerCase();
        const found = [];
        
        for (const word of this.modernAnachronisms) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                found.push(word);
            }
        }
        
        // Also check for modern contractions
        const modernContractions = ["don't", "doesn't", "didn't", "won't", "wouldn't", "couldn't", "shouldn't", "can't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"];
        for (const contraction of modernContractions) {
            if (lower.includes(contraction)) {
                found.push(contraction);
            }
        }
        
        const anachronismScore = Utils.normalize(found.length, 0, 3);

        return {
            anachronismsFound: found,
            count: found.length,
            anachronismScore
        };
    },

    /**
     * Analyze morphological consistency
     */
    analyzeMorphologicalConsistency(text) {
        const lower = text.toLowerCase();
        
        // Check for consistent grammatical patterns
        let consistencyPoints = 0;
        let totalChecks = 0;
        
        // 1. Check if 'thou' is followed by correct verb forms
        const thouPatterns = text.match(/\bthou\s+\w+/gi) || [];
        for (const pattern of thouPatterns) {
            totalChecks++;
            const verb = pattern.split(/\s+/)[1]?.toLowerCase();
            // Correct forms end in -st or -est, or are archaic forms
            if (verb && (verb.endsWith('st') || verb.endsWith('est') || 
                ['art', 'wilt', 'shalt', 'hast', 'dost', 'canst', 'mayest'].includes(verb))) {
                consistencyPoints++;
            }
        }
        
        // 2. Check if third person uses -eth/-th endings consistently
        const heShePatternsModern = (text.match(/\b(he|she|it)\s+(does|has|is|goes|says)\b/gi) || []).length;
        const heShePatternsArchaic = (text.match(/\b(he|she|it)\s+\w+[e]?th\b/gi) || []).length;
        
        if (heShePatternsArchaic > 0 || heShePatternsModern > 0) {
            totalChecks++;
            if (heShePatternsModern === 0 || heShePatternsArchaic === 0) {
                consistencyPoints++;
            }
        }
        
        // 3. Check article usage ('a' vs 'an' before archaic words starting with vowel sounds)
        // Not strictly archaic-specific, but errors here indicate surface imitation
        
        const consistencyScore = totalChecks > 0 ? consistencyPoints / totalChecks : 0.5;

        return {
            checksPerformed: totalChecks,
            consistencyPoints,
            consistencyScore,
            thouVerbPatterns: thouPatterns.slice(0, 5),
            thirdPersonMix: {
                modern: heShePatternsModern,
                archaic: heShePatternsArchaic
            }
        };
    },

    /**
     * Generate findings
     */
    generateFindings(consistencyAnalysis, anachronismAnalysis, morphologyAnalysis) {
        const findings = [];

        // Consistency
        if (consistencyAnalysis.pronounConsistency.isMixed) {
            findings.push({
                label: 'Pronoun Consistency',
                value: "Mixing 'thou/thee' with modern 'you'",
                note: 'Inconsistent second-person pronoun usage',
                indicator: 'ai'
            });
        }

        if (consistencyAnalysis.verbConsistency.isMixed) {
            findings.push({
                label: 'Verb Forms',
                value: 'Mixing archaic and modern verb endings',
                note: "Using both '-eth' forms and modern conjugations",
                indicator: 'ai'
            });
        }

        if (consistencyAnalysis.depthScore < 0.4) {
            findings.push({
                label: 'Style Depth',
                value: 'Surface-level archaic imitation',
                note: 'Archaic elements not consistently applied',
                indicator: 'ai'
            });
        }

        // Anachronisms
        if (anachronismAnalysis.count > 0) {
            findings.push({
                label: 'Anachronisms',
                value: `${anachronismAnalysis.count} modern term(s) in archaic text`,
                note: `Found: ${anachronismAnalysis.anachronismsFound.slice(0, 3).join(', ')}`,
                indicator: 'ai'
            });
        }

        // Morphology
        if (morphologyAnalysis.consistencyScore > 0.7 && consistencyAnalysis.depthScore > 0.6) {
            findings.push({
                label: 'Grammatical Consistency',
                value: 'Archaic grammar rules consistently applied',
                note: 'Deep understanding of historical grammar',
                indicator: 'human'
            });
        }

        return findings;
    },

    calculateConfidence(tokenCount, archaicDensity) {
        const densityValue = parseFloat(archaicDensity) || 0;
        const baseConfidence = tokenCount < 100 ? 0.3 : tokenCount < 300 ? 0.5 : 0.7;
        // Higher confidence if there's substantial archaic content to analyze
        return densityValue > 2 ? baseConfidence + 0.2 : baseConfidence;
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
    module.exports = ArchaicAnalyzer;
}
