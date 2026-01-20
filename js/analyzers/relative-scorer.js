/**
 * VERITAS — Relative Scoring System
 * Cohort-based comparison instead of absolute thresholds
 * 
 * Based on: Research indicating that absolute thresholds fail due to:
 *           - Model drift (newer AIs look more human)
 *           - Human overlap (advanced human writers resemble AI)
 *           - Domain variation (technical vs creative writing)
 * 
 * KEY INSIGHT: Instead of asking "Is this score high?", ask:
 * "How does this compare to expected human variance in this domain?"
 * 
 * Score deviation from expected human variance in cohort.
 * Flag only extreme outliers (>2σ from cohort baseline).
 */

const RelativeScorer = {
    name: 'Relative Scorer',

    // Baseline cohort statistics by domain/style
    // These represent expected ranges for HUMAN writers in each domain
    cohortBaselines: {
        academic: {
            description: 'Academic, research, and scholarly writing',
            patterns: [
                /\b(study|research|analysis|hypothesis|methodology|findings|conclusion|data|evidence)\b/i,
                /\b(et al|ibid|cf|viz|i\.e\.|e\.g\.)\b/i,
                /\b(significant|correlation|variables|participants|results)\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.55, stdDev: 0.12 }, // Academic has rich vocab
                sentenceComplexity: { mean: 0.65, stdDev: 0.15 }, // Long complex sentences
                formalityScore: { mean: 0.75, stdDev: 0.10 },     // Very formal
                passiveVoice: { mean: 0.25, stdDev: 0.10 },       // More passive
                firstPerson: { mean: 0.15, stdDev: 0.12 },        // Limited first person
                repetitionRate: { mean: 0.08, stdDev: 0.04 },     // Some technical repetition
                burstiness: { mean: 0.35, stdDev: 0.15 }          // Lower burstiness
            },
            threshold: 2.0 // Standard deviations for outlier
        },
        technical: {
            description: 'Technical documentation, tutorials, and guides',
            patterns: [
                /\b(function|method|class|variable|parameter|return|install|configure|setup)\b/i,
                /\b(step \d|first|then|next|finally|ensure|make sure)\b/i,
                /```|\bcode\b|\bcommand\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.40, stdDev: 0.10 },
                sentenceComplexity: { mean: 0.45, stdDev: 0.12 },
                formalityScore: { mean: 0.60, stdDev: 0.12 },
                passiveVoice: { mean: 0.20, stdDev: 0.08 },
                firstPerson: { mean: 0.25, stdDev: 0.15 },
                repetitionRate: { mean: 0.12, stdDev: 0.05 }, // More repetition expected
                burstiness: { mean: 0.45, stdDev: 0.15 }
            },
            threshold: 2.0
        },
        journalistic: {
            description: 'News articles, reporting, journalism',
            patterns: [
                /\b(said|according to|reported|announced|statement|official)\b/i,
                /\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b/i,
                /\b(percent|million|billion|government|company|president)\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.50, stdDev: 0.12 },
                sentenceComplexity: { mean: 0.50, stdDev: 0.15 },
                formalityScore: { mean: 0.65, stdDev: 0.12 },
                passiveVoice: { mean: 0.18, stdDev: 0.08 },
                firstPerson: { mean: 0.08, stdDev: 0.06 },
                repetitionRate: { mean: 0.06, stdDev: 0.03 },
                burstiness: { mean: 0.55, stdDev: 0.15 }
            },
            threshold: 2.0
        },
        creative: {
            description: 'Fiction, creative writing, narratives',
            patterns: [
                /\b(felt|thought|wondered|realized|seemed|appeared)\b/i,
                /["'][^"']*["']/g, // Dialogue
                /\b(heart|eyes|hands|voice|silence|darkness|light)\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.60, stdDev: 0.15 }, // High variance ok
                sentenceComplexity: { mean: 0.50, stdDev: 0.20 }, // Very variable
                formalityScore: { mean: 0.35, stdDev: 0.20 },     // Often informal
                passiveVoice: { mean: 0.10, stdDev: 0.08 },
                firstPerson: { mean: 0.35, stdDev: 0.25 },        // Highly variable
                repetitionRate: { mean: 0.10, stdDev: 0.08 },     // Stylistic repetition
                burstiness: { mean: 0.65, stdDev: 0.18 }          // High burstiness
            },
            threshold: 2.5 // More lenient for creative
        },
        conversational: {
            description: 'Casual writing, social media, messages',
            patterns: [
                /\b(lol|omg|btw|idk|tbh|imo|ngl)\b/i,
                /[!?]{2,}|\.{3,}|😀|😂|🤔/,
                /\b(like|literally|basically|honestly|actually)\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.35, stdDev: 0.12 },
                sentenceComplexity: { mean: 0.30, stdDev: 0.15 },
                formalityScore: { mean: 0.20, stdDev: 0.15 },
                passiveVoice: { mean: 0.05, stdDev: 0.05 },
                firstPerson: { mean: 0.45, stdDev: 0.20 },
                repetitionRate: { mean: 0.08, stdDev: 0.05 },
                burstiness: { mean: 0.70, stdDev: 0.15 }
            },
            threshold: 2.5
        },
        business: {
            description: 'Business correspondence, professional emails',
            patterns: [
                /\b(please|kindly|regards|sincerely|attached|meeting|schedule)\b/i,
                /\b(team|project|deadline|deliverable|stakeholder|budget)\b/i,
                /\b(looking forward|best regards|thank you for)\b/i
            ],
            expectedRanges: {
                vocabularyRichness: { mean: 0.42, stdDev: 0.10 },
                sentenceComplexity: { mean: 0.45, stdDev: 0.12 },
                formalityScore: { mean: 0.70, stdDev: 0.10 },
                passiveVoice: { mean: 0.15, stdDev: 0.08 },
                firstPerson: { mean: 0.30, stdDev: 0.15 },
                repetitionRate: { mean: 0.10, stdDev: 0.05 },
                burstiness: { mean: 0.40, stdDev: 0.12 }
            },
            threshold: 1.8 // More strict for business
        },
        general: {
            description: 'General writing, mixed style',
            patterns: [], // Default fallback
            expectedRanges: {
                vocabularyRichness: { mean: 0.50, stdDev: 0.15 },
                sentenceComplexity: { mean: 0.50, stdDev: 0.18 },
                formalityScore: { mean: 0.50, stdDev: 0.20 },
                passiveVoice: { mean: 0.15, stdDev: 0.10 },
                firstPerson: { mean: 0.25, stdDev: 0.18 },
                repetitionRate: { mean: 0.08, stdDev: 0.05 },
                burstiness: { mean: 0.50, stdDev: 0.18 }
            },
            threshold: 2.0
        }
    },

    /**
     * Detect the cohort/domain of the text
     */
    detectCohort(text) {
        const scores = {};
        
        for (const [cohort, data] of Object.entries(this.cohortBaselines)) {
            if (cohort === 'general') continue;
            
            let score = 0;
            for (const pattern of data.patterns) {
                const matches = text.match(pattern);
                if (matches) {
                    score += matches.length;
                }
            }
            scores[cohort] = score;
        }
        
        // Find best match
        let bestCohort = 'general';
        let bestScore = 0;
        
        for (const [cohort, score] of Object.entries(scores)) {
            if (score > bestScore) {
                bestScore = score;
                bestCohort = cohort;
            }
        }
        
        // Require minimum match strength
        if (bestScore < 3) {
            bestCohort = 'general';
        }
        
        return {
            cohort: bestCohort,
            confidence: Math.min(bestScore / 10, 1),
            scores
        };
    },

    /**
     * Calculate metrics for comparison
     */
    calculateMetrics(text) {
        const words = Utils.tokenize(text);
        const sentences = Utils.splitSentences(text);
        const uniqueWords = new Set(words.map(w => w.toLowerCase()));
        
        // Vocabulary richness (type-token ratio normalized)
        const vocabularyRichness = words.length > 0 ? 
            Math.min(uniqueWords.size / Math.sqrt(words.length) / 5, 1) : 0;
        
        // Sentence complexity (based on length and subordinate clauses)
        const avgSentenceLength = words.length / Math.max(sentences.length, 1);
        const subordinates = (text.match(/\b(because|although|while|if|when|since|unless|whereas)\b/gi) || []).length;
        const sentenceComplexity = Utils.normalize(avgSentenceLength, 5, 35) * 0.6 +
                                   Utils.normalize(subordinates / sentences.length, 0, 0.5) * 0.4;
        
        // Formality score
        const formalWords = (text.match(/\b(therefore|however|moreover|furthermore|consequently|thus|hence)\b/gi) || []).length;
        const informalWords = (text.match(/\b(like|basically|actually|literally|kinda|sorta|gonna|wanna)\b/gi) || []).length;
        const contractions = (text.match(/\b\w+[''](?:t|s|d|ll|ve|re|m)\b/gi) || []).length;
        const formalityScore = Utils.normalize(
            (formalWords * 2 - informalWords - contractions) / words.length * 100 + 50,
            0, 100
        ) / 100;
        
        // Passive voice ratio
        const passiveMatches = text.match(/\b(is|are|was|were|been|be|being)\s+\w+ed\b/gi) || [];
        const passiveVoice = passiveMatches.length / Math.max(sentences.length, 1);
        
        // First person usage
        const firstPersonMatches = text.match(/\b(I|me|my|mine|we|us|our|ours)\b/gi) || [];
        const firstPerson = firstPersonMatches.length / Math.max(words.length, 1);
        
        // Repetition rate
        const wordCounts = {};
        for (const word of words.map(w => w.toLowerCase())) {
            wordCounts[word] = (wordCounts[word] || 0) + 1;
        }
        const repeatedWords = Object.values(wordCounts).filter(c => c > 2).length;
        const repetitionRate = repeatedWords / Math.max(uniqueWords.size, 1);
        
        // Burstiness (variance in sentence lengths)
        const sentenceLengths = sentences.map(s => Utils.tokenize(s).length);
        const burstiness = sentenceLengths.length > 1 ? 
            Utils.normalize(Utils.variance(sentenceLengths), 0, 100) : 0.5;
        
        return {
            vocabularyRichness,
            sentenceComplexity,
            formalityScore,
            passiveVoice: Math.min(passiveVoice, 1),
            firstPerson: Math.min(firstPerson, 1),
            repetitionRate: Math.min(repetitionRate, 1),
            burstiness
        };
    },

    /**
     * Calculate z-scores for each metric relative to cohort
     */
    calculateZScores(metrics, cohort) {
        const baseline = this.cohortBaselines[cohort].expectedRanges;
        const zScores = {};
        
        for (const [metric, value] of Object.entries(metrics)) {
            if (baseline[metric]) {
                const { mean, stdDev } = baseline[metric];
                zScores[metric] = (value - mean) / stdDev;
            }
        }
        
        return zScores;
    },

    /**
     * Determine outliers
     */
    findOutliers(zScores, threshold) {
        const outliers = [];
        
        for (const [metric, zScore] of Object.entries(zScores)) {
            const absZ = Math.abs(zScore);
            if (absZ > threshold) {
                outliers.push({
                    metric,
                    zScore,
                    severity: absZ > threshold * 1.5 ? 'extreme' : 'moderate',
                    direction: zScore > 0 ? 'above' : 'below'
                });
            }
        }
        
        return outliers;
    },

    /**
     * Main scoring function
     */
    score(text, existingMetrics = null) {
        if (!text || text.length < 100) {
            return {
                cohort: 'general',
                relativeScore: 0.5,
                outliers: [],
                confidence: 0,
                message: 'Text too short for relative scoring'
            };
        }

        // Detect cohort
        const cohortResult = this.detectCohort(text);
        const cohort = cohortResult.cohort;
        const cohortData = this.cohortBaselines[cohort];
        
        // Calculate or use existing metrics
        const metrics = existingMetrics || this.calculateMetrics(text);
        
        // Calculate z-scores
        const zScores = this.calculateZScores(metrics, cohort);
        
        // Find outliers
        const outliers = this.findOutliers(zScores, cohortData.threshold);
        
        // Calculate relative AI score based on outlier severity
        // More outliers and more extreme = higher AI probability
        let outliersScore = 0;
        for (const outlier of outliers) {
            const weight = outlier.severity === 'extreme' ? 0.3 : 0.15;
            outliersScore += weight;
        }
        
        // Also consider unusual uniformity (all metrics close to 0 z-score)
        const avgAbsZ = Utils.mean(Object.values(zScores).map(Math.abs));
        const uniformityPenalty = avgAbsZ < 0.3 ? 0.2 : 0; // Suspiciously average
        
        const relativeScore = Math.min(outliersScore + uniformityPenalty, 1);
        
        // Calculate confidence based on cohort match and text length
        const confidence = cohortResult.confidence * 0.5 + 
                          Utils.normalize(text.length, 100, 2000) * 0.5;

        return {
            cohort,
            cohortDescription: cohortData.description,
            cohortConfidence: cohortResult.confidence,
            metrics,
            zScores,
            outliers,
            relativeScore,
            avgAbsZScore: avgAbsZ,
            uniformityPenalty,
            threshold: cohortData.threshold,
            confidence: Math.min(confidence, 1),
            interpretation: this.interpretResults(outliers, avgAbsZ, cohort)
        };
    },

    /**
     * Interpret results
     */
    interpretResults(outliers, avgAbsZ, cohort) {
        if (outliers.length === 0 && avgAbsZ < 0.3) {
            return `Text metrics are suspiciously average for ${cohort} writing. All measures fall within ±0.3σ of expected human baseline—natural writing typically shows more variation.`;
        }
        
        if (outliers.length === 0) {
            return `Text falls within expected human range for ${cohort} writing. No significant outliers detected.`;
        }
        
        if (outliers.length >= 3) {
            const metrics = outliers.map(o => o.metric).join(', ');
            return `Multiple metrics deviate significantly from ${cohort} baseline: ${metrics}. This pattern is unusual for human writing in this domain.`;
        }
        
        const outlier = outliers[0];
        return `${outlier.metric} is ${outlier.severity}ly ${outlier.direction} baseline for ${cohort} writing (z=${outlier.zScore.toFixed(2)}).`;
    },

    /**
     * Adjust raw AI probability using relative scoring
     * This is the key integration point
     */
    adjustProbability(rawProbability, relativeScore, cohortConfidence) {
        // If we're confident about the cohort, weight the relative score more
        const relativeWeight = 0.3 * cohortConfidence;
        const rawWeight = 1 - relativeWeight;
        
        // Blend raw and relative scores
        const blended = rawProbability * rawWeight + relativeScore * relativeWeight;
        
        return {
            adjusted: blended,
            rawProbability,
            relativeScore,
            adjustment: blended - rawProbability,
            relativeWeight
        };
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = RelativeScorer;
}
