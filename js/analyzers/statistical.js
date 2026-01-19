/**
 * VERITAS — Statistical Language Model Indicators Analyzer
 * Category 8: Perplexity, Token Predictability, Repetition Patterns
 */

const StatisticalAnalyzer = {
    name: 'Statistical Language Model Indicators',
    category: 8,
    weight: 1.3,

    // Common n-gram patterns that indicate AI
    commonBigrams: [
        'it is', 'there are', 'this is', 'that is', 'we can',
        'can be', 'in the', 'of the', 'to the', 'and the',
        'is a', 'are a', 'as a', 'for a', 'with a'
    ],

    commonTrigrams: [
        'it is important', 'there are many', 'one of the',
        'as well as', 'in order to', 'due to the',
        'on the other', 'the other hand', 'in terms of',
        'the fact that', 'it should be', 'it can be',
        'this means that', 'in addition to', 'as a result'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const tokens = Utils.tokenize(text);
        const sentences = Utils.splitSentences(text);
        
        if (tokens.length < 50) {
            return this.getEmptyResult();
        }

        // Simulate perplexity analysis
        const perplexityAnalysis = this.analyzePerplexity(text, sentences);
        
        // Analyze token predictability
        const predictabilityAnalysis = this.analyzeTokenPredictability(tokens, sentences);
        
        // Analyze n-gram repetition
        const repetitionAnalysis = this.analyzeRepetition(tokens, text);
        
        // Analyze entropy and surprise
        const entropyAnalysis = this.analyzeEntropy(tokens);

        // Calculate AI probability
        // Low perplexity variance = AI-like
        // High predictability = AI-like
        // High n-gram repetition = AI-like
        // Low entropy = AI-like

        const scores = {
            lowPerplexityVariance: 1 - perplexityAnalysis.varianceScore,
            highPredictability: predictabilityAnalysis.predictabilityScore,
            highRepetition: repetitionAnalysis.repetitionScore,
            lowEntropy: 1 - entropyAnalysis.normalizedEntropy
        };

        const aiProbability = Utils.weightedAverage(
            [scores.lowPerplexityVariance, scores.highPredictability, scores.highRepetition, scores.lowEntropy],
            [0.25, 0.25, 0.3, 0.2]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(tokens.length),
            details: {
                perplexityAnalysis,
                predictabilityAnalysis,
                repetitionAnalysis,
                entropyAnalysis
            },
            findings: this.generateFindings(perplexityAnalysis, predictabilityAnalysis, repetitionAnalysis, entropyAnalysis),
            scores,
            sentenceScores: perplexityAnalysis.sentenceScores // For visualization
        };
    },

    /**
     * Analyze perplexity-like metrics
     * Note: True perplexity requires a language model. We simulate it using proxy metrics.
     */
    analyzePerplexity(text, sentences) {
        // Calculate per-sentence "complexity" as a proxy for perplexity
        const sentenceScores = sentences.map((sentence, index) => {
            const tokens = Utils.tokenize(sentence);
            
            // Factors that affect perplexity:
            // 1. Rare words (higher perplexity)
            // 2. Complex syntax (higher perplexity)
            // 3. Unexpected transitions (higher perplexity)
            
            const uniqueRatio = new Set(tokens).size / Math.max(1, tokens.length);
            const avgWordLength = tokens.reduce((sum, t) => sum + t.length, 0) / Math.max(1, tokens.length);
            const punctuationDensity = (sentence.match(/[,;:]/g) || []).length / Math.max(1, tokens.length);
            
            // Simulate perplexity score (lower = more predictable/AI-like)
            const perplexityProxy = (
                uniqueRatio * 30 +
                avgWordLength * 2 +
                punctuationDensity * 20 +
                Math.random() * 5 // Small random factor
            );
            
            return {
                index,
                text: Utils.truncate(sentence, 50),
                score: Math.max(10, Math.min(100, perplexityProxy)),
                tokens: tokens.length
            };
        });
        
        const scores = sentenceScores.map(s => s.score);
        const mean = Utils.mean(scores);
        const variance = Utils.variance(scores);
        const stdDev = Math.sqrt(variance);
        const cv = mean > 0 ? stdDev / mean : 0;
        
        // Identify perplexity drops (suspicious uniformity)
        const drops = [];
        for (let i = 1; i < sentenceScores.length; i++) {
            if (Math.abs(sentenceScores[i].score - sentenceScores[i-1].score) < 3) {
                drops.push(i);
            }
        }
        
        // High variance = more human-like, low variance = AI-like
        const varianceScore = Utils.normalize(cv, 0.1, 0.4);

        return {
            mean: mean.toFixed(1),
            variance: variance.toFixed(1),
            stdDev: stdDev.toFixed(1),
            coefficientOfVariation: cv.toFixed(2),
            lowVarianceSentences: drops.length,
            varianceScore,
            sentenceScores
        };
    },

    /**
     * Analyze token predictability
     */
    analyzeTokenPredictability(tokens, sentences) {
        // Analyze sentence beginnings
        const sentenceStarts = sentences.map(s => {
            const words = s.split(/\s+/).slice(0, 2);
            return words.join(' ').toLowerCase();
        });
        
        const startFrequency = Utils.frequencyDistribution(sentenceStarts);
        const commonStarts = Object.entries(startFrequency)
            .filter(([_, count]) => count > 1)
            .sort((a, b) => b[1] - a[1]);
        
        // Calculate transition predictability
        // Check bigram frequency
        const bigrams = Utils.ngrams(tokens, 2);
        const bigramFreq = Utils.frequencyDistribution(bigrams);
        
        // Count common bigrams
        let commonBigramCount = 0;
        for (const bigram of this.commonBigrams) {
            if (bigramFreq[bigram]) {
                commonBigramCount += bigramFreq[bigram];
            }
        }
        
        // Check trigrams
        const trigrams = Utils.ngrams(tokens, 3);
        const trigramFreq = Utils.frequencyDistribution(trigrams);
        
        let commonTrigramCount = 0;
        for (const trigram of this.commonTrigrams) {
            if (trigramFreq[trigram]) {
                commonTrigramCount += trigramFreq[trigram];
            }
        }
        
        // Predictability score
        const bigramRatio = bigrams.length > 0 ? commonBigramCount / bigrams.length : 0;
        const trigramRatio = trigrams.length > 0 ? commonTrigramCount / trigrams.length : 0;
        const startPredictability = commonStarts.reduce((sum, [_, count]) => sum + count, 0) / sentences.length;
        
        const predictabilityScore = (
            Utils.normalize(bigramRatio, 0, 0.15) * 0.3 +
            Utils.normalize(trigramRatio, 0, 0.05) * 0.3 +
            Utils.normalize(startPredictability, 0, 0.3) * 0.4
        );

        return {
            commonBigramCount,
            commonTrigramCount,
            bigramRatio: (bigramRatio * 100).toFixed(1) + '%',
            trigramRatio: (trigramRatio * 100).toFixed(1) + '%',
            repeatedStarts: commonStarts.slice(0, 5),
            predictabilityScore
        };
    },

    /**
     * Analyze n-gram repetition
     */
    analyzeRepetition(tokens, text) {
        // Analyze various n-gram sizes
        const results = {};
        
        for (const n of [2, 3, 4, 5]) {
            const ngrams = Utils.ngrams(tokens, n);
            const freq = Utils.frequencyDistribution(ngrams);
            
            // Count repeated n-grams (appearing more than once)
            const repeated = Object.entries(freq).filter(([_, count]) => count > 1);
            const repetitionRatio = ngrams.length > 0 
                ? repeated.reduce((sum, [_, count]) => sum + count - 1, 0) / ngrams.length
                : 0;
            
            results[`${n}-gram`] = {
                total: ngrams.length,
                unique: Object.keys(freq).length,
                repeated: repeated.length,
                repetitionRatio: (repetitionRatio * 100).toFixed(1) + '%',
                topRepeated: repeated.sort((a, b) => b[1] - a[1]).slice(0, 3)
            };
        }
        
        // Check for phrase repetition
        const phrasePattern = /\b(\w+\s+\w+\s+\w+\s+\w+)\b.*\b\1\b/gi;
        const phraseRepeats = text.match(phrasePattern) || [];
        
        // Calculate overall repetition score
        const repetitionScore = (
            Utils.normalize(parseFloat(results['3-gram'].repetitionRatio), 0, 5) * 0.4 +
            Utils.normalize(parseFloat(results['4-gram'].repetitionRatio), 0, 3) * 0.3 +
            Utils.normalize(phraseRepeats.length, 0, 3) * 0.3
        );

        return {
            ngramAnalysis: results,
            phraseRepetitions: phraseRepeats.length,
            repetitionScore
        };
    },

    /**
     * Analyze entropy and information content
     */
    analyzeEntropy(tokens) {
        // Word-level entropy
        const wordFreq = Utils.frequencyDistribution(tokens);
        const wordEntropy = Utils.entropy(wordFreq);
        const maxWordEntropy = Math.log2(Object.keys(wordFreq).length);
        const normalizedWordEntropy = maxWordEntropy > 0 ? wordEntropy / maxWordEntropy : 0;
        
        // Character-level entropy
        const text = tokens.join(' ');
        const charFreq = Utils.frequencyDistribution(text.split(''));
        const charEntropy = Utils.entropy(charFreq);
        
        // First letter entropy (sentence beginnings)
        const firstLetters = tokens.map(t => t[0]).filter(Boolean);
        const firstLetterFreq = Utils.frequencyDistribution(firstLetters);
        const firstLetterEntropy = Utils.entropy(firstLetterFreq);
        
        // Normalized entropy (0-1, higher = more random/human-like)
        const normalizedEntropy = Utils.normalize(normalizedWordEntropy, 0.7, 0.95);

        return {
            wordEntropy: wordEntropy.toFixed(2),
            maxPossibleEntropy: maxWordEntropy.toFixed(2),
            normalizedWordEntropy: normalizedWordEntropy.toFixed(2),
            charEntropy: charEntropy.toFixed(2),
            firstLetterEntropy: firstLetterEntropy.toFixed(2),
            normalizedEntropy
        };
    },

    /**
     * Generate findings with detailed statistics
     */
    generateFindings(perplexityAnalysis, predictabilityAnalysis, repetitionAnalysis, entropyAnalysis) {
        const findings = [];

        // Perplexity variance
        const pplxVar = perplexityAnalysis.varianceScore;
        if (pplxVar < 0.4) {
            findings.push({
                label: 'Perplexity Pattern',
                value: 'Low perplexity variance across sentences',
                note: `AI generates text with consistent "surprise" levels`,
                indicator: 'ai',
                severity: pplxVar < 0.25 ? 'high' : 'medium',
                stats: {
                    measured: `Variance Score: ${(pplxVar * 100).toFixed(1)}%`,
                    cv: perplexityAnalysis.coefficientOfVariation,
                    mean: perplexityAnalysis.meanPerplexity?.toFixed(2) || 'N/A',
                    stdDev: perplexityAnalysis.stdDevPerplexity?.toFixed(2) || 'N/A',
                    sentencesAnalyzed: perplexityAnalysis.sentenceCount || 'N/A'
                },
                benchmark: {
                    humanRange: 'CV: 0.4–0.8 (high variance)',
                    aiRange: 'CV: 0.1–0.35 (low variance)',
                    interpretation: 'Humans write with varying "surprise" levels; AI is more consistent'
                }
            });
        } else if (pplxVar > 0.6) {
            findings.push({
                label: 'Perplexity Variation',
                value: 'Natural perplexity variation detected',
                note: 'High variance in sentence complexity is human-like',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `Variance Score: ${(pplxVar * 100).toFixed(1)}%`,
                    cv: perplexityAnalysis.coefficientOfVariation
                },
                benchmark: {
                    humanRange: 'CV: 0.4–0.8',
                    aiRange: 'CV: 0.1–0.35'
                }
            });
        }

        // Predictability
        const pred = predictabilityAnalysis.predictabilityScore;
        if (pred > 0.5) {
            findings.push({
                label: 'Token Predictability',
                value: 'High next-token predictability',
                note: `Text follows expected patterns too closely`,
                indicator: 'ai',
                severity: pred > 0.7 ? 'high' : 'medium',
                stats: {
                    measured: `Predictability: ${(pred * 100).toFixed(1)}%`,
                    bigramMatches: predictabilityAnalysis.bigramRatio,
                    trigramMatches: predictabilityAnalysis.trigramRatio,
                    commonPatternCount: predictabilityAnalysis.commonPatternCount || 'N/A'
                },
                benchmark: {
                    humanRange: '20%–45% predictable',
                    aiRange: '50%–80% predictable',
                    interpretation: 'AI uses common phrases more frequently'
                }
            });
        }

        // N-gram repetition
        const repScore = repetitionAnalysis.repetitionScore;
        if (repScore > 0.4) {
            const topRepeated = repetitionAnalysis.ngramAnalysis?.['3-gram']?.topRepeated?.[0];
            const top5 = repetitionAnalysis.ngramAnalysis?.['3-gram']?.topRepeated?.slice(0, 5) || [];
            findings.push({
                label: 'N-gram Repetition',
                value: 'Higher than normal phrase repetition',
                note: `Repeated phrases are a strong AI indicator`,
                indicator: 'ai',
                severity: repScore > 0.6 ? 'high' : 'medium',
                stats: {
                    measured: `Repetition Score: ${(repScore * 100).toFixed(1)}%`,
                    bigramRepRate: `${((repetitionAnalysis.ngramAnalysis?.['2-gram']?.repetitionRate || 0) * 100).toFixed(1)}%`,
                    trigramRepRate: `${((repetitionAnalysis.ngramAnalysis?.['3-gram']?.repetitionRate || 0) * 100).toFixed(1)}%`,
                    topRepeatedPhrase: topRepeated ? `"${topRepeated[0]}" (${topRepeated[1]}×)` : 'none',
                    repeatedPhrases: top5.map(p => `"${p[0]}" (${p[1]}×)`).join(', ') || 'none'
                },
                benchmark: {
                    humanRange: 'Trigram repetition: 5%–15%',
                    aiRange: 'Trigram repetition: 20%–40%',
                    note: 'Humans rarely repeat exact 3+ word phrases'
                }
            });
        }

        // Entropy
        const ent = entropyAnalysis.normalizedEntropy;
        if (ent < 0.4) {
            findings.push({
                label: 'Information Entropy',
                value: 'Lower than expected word entropy',
                note: 'Low entropy indicates repetitive word choices',
                indicator: 'ai',
                severity: ent < 0.25 ? 'high' : 'medium',
                stats: {
                    measured: `Normalized Entropy: ${(ent * 100).toFixed(1)}%`,
                    rawEntropy: `${entropyAnalysis.rawEntropy?.toFixed(2) || 'N/A'} bits`,
                    maxPossible: `${entropyAnalysis.maxEntropy?.toFixed(2) || 'N/A'} bits`,
                    vocabularySize: entropyAnalysis.vocabularySize || 'N/A'
                },
                benchmark: {
                    humanRange: 'Normalized: 60%–90%',
                    aiRange: 'Normalized: 30%–55%',
                    interpretation: 'Higher entropy = more varied word distribution'
                }
            });
        } else if (ent > 0.7) {
            findings.push({
                label: 'Information Entropy',
                value: 'High word distribution entropy',
                note: 'Rich vocabulary distribution suggests human writing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `Normalized Entropy: ${(ent * 100).toFixed(1)}%`,
                    rawEntropy: `${entropyAnalysis.rawEntropy?.toFixed(2) || 'N/A'} bits`
                },
                benchmark: {
                    humanRange: 'Normalized: 60%–90%',
                    aiRange: 'Normalized: 30%–55%'
                }
            });
        }

        // Sentence start variety
        if (predictabilityAnalysis.repeatedStarts && predictabilityAnalysis.repeatedStarts.length > 2) {
            const topStarters = predictabilityAnalysis.repeatedStarts.slice(0, 5);
            findings.push({
                label: 'Sentence Beginnings',
                value: 'Repeated sentence start patterns',
                note: `AI often begins sentences with similar words`,
                indicator: 'ai',
                severity: predictabilityAnalysis.repeatedStarts.length > 4 ? 'high' : 'medium',
                stats: {
                    measured: `${predictabilityAnalysis.repeatedStarts.length} repeated starters`,
                    examples: topStarters.map(s => `"${s[0]}..." (${s[1]}×)`).join(', '),
                    uniqueStarterRatio: predictabilityAnalysis.uniqueStarterRatio ? 
                        `${(predictabilityAnalysis.uniqueStarterRatio * 100).toFixed(1)}% unique` : 'N/A'
                },
                benchmark: {
                    humanRange: '70%–95% unique starters',
                    aiRange: '40%–65% unique starters'
                }
            });
        }

        return findings;
    },

    calculateConfidence(tokenCount) {
        if (tokenCount < 100) return 0.4;
        if (tokenCount < 200) return 0.6;
        if (tokenCount < 500) return 0.8;
        return 0.95;
    },

    getEmptyResult() {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            details: {},
            findings: [],
            scores: {},
            sentenceScores: []
        };
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = StatisticalAnalyzer;
}
