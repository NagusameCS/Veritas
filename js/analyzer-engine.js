/**
 * VERITAS — Analyzer Engine v2.0
 * Variance-Based Detection: Measures deviations from expected human variance
 * 
 * Philosophy: "Never grade features as 'AI-like' or 'human-like' in isolation.
 * Grade them as deviations from expected human variance."
 */

const AnalyzerEngine = {
    // Category weights based on detection importance
    categoryWeights: {
        syntaxVariance: 0.15,          // Syntax variance
        lexicalDiversity: 0.12,        // Lexical diversity  
        repetitionUniformity: 0.12,    // Repetition uniformity
        toneStability: 0.12,           // Tone stability
        grammarEntropy: 0.08,          // Grammar entropy
        perplexity: 0.08,              // Statistical perplexity
        authorshipDrift: 0.10,         // Authorship drift
        metadataFormatting: 0.25       // Metadata & formatting (strong signal for AI markers)
    },

    // Getter for analyzers - allows graceful handling of missing modules
    get analyzers() {
        const all = [];
        
        // Core analyzers (Categories 1-10)
        if (typeof GrammarAnalyzer !== 'undefined') all.push(GrammarAnalyzer);
        if (typeof SyntaxAnalyzer !== 'undefined') all.push(SyntaxAnalyzer);
        if (typeof LexicalAnalyzer !== 'undefined') all.push(LexicalAnalyzer);
        if (typeof DialectAnalyzer !== 'undefined') all.push(DialectAnalyzer);
        if (typeof ArchaicAnalyzer !== 'undefined') all.push(ArchaicAnalyzer);
        if (typeof DiscourseAnalyzer !== 'undefined') all.push(DiscourseAnalyzer);
        if (typeof SemanticAnalyzer !== 'undefined') all.push(SemanticAnalyzer);
        if (typeof StatisticalAnalyzer !== 'undefined') all.push(StatisticalAnalyzer);
        if (typeof AuthorshipAnalyzer !== 'undefined') all.push(AuthorshipAnalyzer);
        if (typeof MetaPatternsAnalyzer !== 'undefined') all.push(MetaPatternsAnalyzer);
        
        // New variance-based analyzers (Categories 11-14)
        if (typeof MetadataAnalyzer !== 'undefined') all.push(MetadataAnalyzer);
        if (typeof RepetitionAnalyzer !== 'undefined') all.push(RepetitionAnalyzer);
        if (typeof ToneAnalyzer !== 'undefined') all.push(ToneAnalyzer);
        if (typeof PartOfSpeechAnalyzer !== 'undefined') all.push(PartOfSpeechAnalyzer);
        
        return all;
    },

    // Fallback for missing analyzers (graceful degradation)
    safeGetAnalyzer(analyzerRef) {
        try {
            return typeof analyzerRef !== 'undefined' ? analyzerRef : null;
        } catch {
            return null;
        }
    },

    /**
     * Run full analysis on text
     * Now uses variance-based detection philosophy
     */
    analyze(text, metadata = null) {
        const startTime = performance.now();
        
        if (!text || text.trim().length === 0) {
            return this.getEmptyResult();
        }

        // Precompute common values
        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);

        // Basic stats
        const stats = {
            characters: text.length,
            words: tokens.length,
            sentences: sentences.length,
            paragraphs: paragraphs.length,
            avgWordsPerSentence: sentences.length > 0 ? (tokens.length / sentences.length).toFixed(1) : 0
        };

        // Advanced statistics
        const advancedStats = this.computeAdvancedStatistics(text, tokens, sentences);

        // Filter to only available analyzers
        const availableAnalyzers = this.analyzers.filter(a => this.safeGetAnalyzer(a));

        // Run all analyzers
        const categoryResults = [];
        for (const analyzer of availableAnalyzers) {
            if (!analyzer) continue;
            try {
                const result = analyzer.analyze(text, metadata);
                // Add variance scores if analyzer supports it
                if (result.uniformityScores) {
                    result.varianceAnalysis = this.computeVarianceMetrics(result.uniformityScores);
                }
                categoryResults.push(result);
            } catch (error) {
                console.error(`Error in ${analyzer?.name || 'unknown'}:`, error);
                categoryResults.push({
                    name: analyzer?.name || 'Unknown',
                    category: analyzer?.category || 0,
                    aiProbability: 0.5,
                    confidence: 0,
                    error: error.message
                });
            }
        }

        // Calculate overall AI probability using variance-based scoring
        const overallResult = this.calculateVarianceBasedProbability(categoryResults);
        
        // Generate sentence-level scores for highlighting
        const sentenceScores = this.scoreSentences(text, sentences, categoryResults);
        
        // Compile all findings
        const allFindings = categoryResults
            .flatMap(r => r.findings || [])
            .sort((a, b) => {
                const order = { ai: 0, mixed: 1, human: 2, neutral: 3 };
                return (order[a.indicator] || 3) - (order[b.indicator] || 3);
            });

        // Compute confidence interval
        const confidenceInterval = this.computeConfidenceInterval(
            overallResult.aiProbability,
            categoryResults.length,
            overallResult.confidence
        );

        // Assess false positive risk
        const falsePositiveRisk = this.assessFalsePositiveRisk(categoryResults, text);

        const endTime = performance.now();

        return {
            // Overall results
            aiProbability: overallResult.aiProbability,
            humanProbability: 1 - overallResult.aiProbability,
            mixedProbability: overallResult.mixedProbability,
            confidence: overallResult.confidence,
            confidenceInterval,
            falsePositiveRisk,
            verdict: this.getVerdict(overallResult.aiProbability, overallResult.confidence),
            
            // Variance-specific data
            varianceProfile: overallResult.varianceProfile,
            featureContributions: overallResult.featureContributions,
            
            // Per-category results
            categoryResults,
            
            // Sentence-level analysis
            sentences,
            sentenceScores,
            
            // All findings
            findings: allFindings,
            
            // Statistics
            stats,
            advancedStats,
            metadata: metadata || null,
            analysisTime: (endTime - startTime).toFixed(0) + 'ms'
        };
    },

    /**
     * Compute advanced statistics for comprehensive analysis
     */
    computeAdvancedStatistics(text, tokens, sentences) {
        const stats = {};

        // === Vocabulary Richness Metrics ===
        stats.vocabulary = {
            uniqueWords: new Set(tokens).size,
            typeTokenRatio: Utils.typeTokenRatio(tokens),
            rootTTR: Utils.rootTTR(tokens),
            logTTR: Utils.logTTR(tokens),
            msttr: Utils.msttr(tokens, 50),
            hapaxLegomenaRatio: Utils.hapaxLegomenaRatio(tokens),
            disLegomenaRatio: Utils.disLegomenaRatio(tokens),
            yulesK: Utils.yulesK(tokens),
            simpsonsD: Utils.simpsonsD(tokens),
            honoresR: Utils.honoresR(tokens),
            brunetsW: Utils.brunetsW(tokens),
            sichelsS: Utils.sichelsS(tokens)
        };

        // === Word Statistics ===
        stats.words = {
            avgLength: Utils.avgWordLength(tokens),
            lengthDistribution: Utils.wordLengthDistribution(tokens),
            entropy: Utils.entropy(Utils.frequencyDistribution(tokens))
        };

        // === Sentence Statistics ===
        const sentLengths = sentences.map(s => Utils.tokenize(s).length);
        stats.sentences = {
            mean: Utils.mean(sentLengths),
            median: Utils.median(sentLengths),
            stdDev: Utils.standardDeviation(sentLengths),
            variance: Utils.variance(sentLengths),
            min: Math.min(...sentLengths),
            max: Math.max(...sentLengths),
            range: Math.max(...sentLengths) - Math.min(...sentLengths),
            coefficientOfVariation: Utils.coefficientOfVariation(sentLengths),
            skewness: Utils.skewness(sentLengths),
            kurtosis: Utils.kurtosis(sentLengths),
            gini: Utils.giniCoefficient(sentLengths)
        };

        // === Zipf's Law Analysis ===
        stats.zipf = Utils.zipfAnalysis(tokens);

        // === N-gram Analysis ===
        const bigrams = Utils.ngrams(tokens, 2);
        const trigrams = Utils.ngrams(tokens, 3);
        stats.ngrams = {
            uniqueBigrams: new Set(bigrams).size,
            uniqueTrigrams: new Set(trigrams).size,
            bigramRepetitionRate: bigrams.length > 0 ? 1 - (new Set(bigrams).size / bigrams.length) : 0,
            trigramRepetitionRate: trigrams.length > 0 ? 1 - (new Set(trigrams).size / trigrams.length) : 0
        };

        // === Readability Approximations ===
        const syllableCount = this.estimateSyllables(tokens);
        const avgSyllablesPerWord = tokens.length > 0 ? syllableCount / tokens.length : 0;
        
        // Flesch Reading Ease approximation
        const fleschRE = 206.835 - 1.015 * (stats.sentences.mean || 15) - 84.6 * avgSyllablesPerWord;
        
        // Flesch-Kincaid Grade Level approximation
        const fleschKG = 0.39 * (stats.sentences.mean || 15) + 11.8 * avgSyllablesPerWord - 15.59;
        
        // Gunning Fog Index approximation
        const complexWords = tokens.filter(t => this.syllableCount(t) >= 3).length;
        const complexWordPct = tokens.length > 0 ? (complexWords / tokens.length) * 100 : 0;
        const gunningFog = 0.4 * ((stats.sentences.mean || 15) + complexWordPct);

        stats.readability = {
            avgSyllablesPerWord,
            fleschReadingEase: Math.max(0, Math.min(100, fleschRE)),
            fleschKincaidGrade: Math.max(0, fleschKG),
            gunningFogIndex: gunningFog,
            complexWordPercentage: complexWordPct
        };

        // === Burstiness Metrics ===
        stats.burstiness = {
            sentenceLength: VarianceUtils.burstiness(sentLengths),
            wordLength: VarianceUtils.burstiness(tokens.map(t => t.length)),
            overallUniformity: VarianceUtils.uniformityScore(sentLengths)
        };

        // === Function Word Analysis ===
        const functionWordCount = tokens.filter(t => Utils.functionWords.has(t.toLowerCase())).length;
        const contentWordCount = tokens.length - functionWordCount;
        stats.functionWords = {
            count: functionWordCount,
            ratio: tokens.length > 0 ? functionWordCount / tokens.length : 0,
            contentWordRatio: tokens.length > 0 ? contentWordCount / tokens.length : 0
        };

        // === Enhanced Statistical Analysis (v2.1) ===
        
        // Autocorrelation analysis - detects periodic patterns
        stats.autocorrelation = VarianceUtils.autocorrelation(sentLengths, 8);
        
        // N-gram perplexity approximation
        stats.perplexity = VarianceUtils.ngramPerplexity(tokens, 2);
        
        // Runs test for randomness
        stats.runsTest = VarianceUtils.runsTest(sentLengths);
        
        // Chi-squared uniformity test on sentence lengths (binned)
        const lengthBins = this.binValues(sentLengths, 5);
        stats.chiSquared = VarianceUtils.chiSquaredUniformity(lengthBins);
        
        // Variance stability across document
        stats.varianceStability = VarianceUtils.varianceStability(sentLengths, 5);
        
        // Length normalization factor
        stats.lengthNormalization = VarianceUtils.lengthNormalization(tokens.length);
        
        // Mahalanobis distance from human baseline
        const featureVector = {
            sentenceLengthCV: stats.sentences.coefficientOfVariation,
            hapaxRatio: stats.vocabulary.hapaxLegomenaRatio,
            burstiness: stats.burstiness.sentenceLength,
            zipfSlope: stats.zipf.slope,
            ttrNormalized: stats.vocabulary.typeTokenRatio
        };
        stats.mahalanobisDistance = VarianceUtils.mahalanobisDistance(featureVector);

        // === BELL CURVE / GAUSSIAN DEVIATION ANALYSIS (v2.2) ===
        // Core philosophy: Humans fall in a "reasonable middle" - neither too perfect nor too chaotic
        
        // How "natural" is the variance? (1 = human-like middle, 0 = extreme)
        stats.varianceNaturalness = VarianceUtils.varianceNaturalnessScore(sentLengths);
        
        // Does this text show extremes in either direction?
        stats.extremeVarianceIndicator = VarianceUtils.extremeVarianceScore(sentLengths);
        
        // Per-feature human likelihood scores
        stats.humanLikelihood = {
            sentenceLengthCV: VarianceUtils.humanLikelihoodScore(
                stats.sentences.coefficientOfVariation,
                VarianceUtils.HUMAN_BASELINES.sentenceLengthCV
            ),
            hapaxRatio: VarianceUtils.humanLikelihoodScore(
                stats.vocabulary.hapaxLegomenaRatio,
                VarianceUtils.HUMAN_BASELINES.hapaxRatio
            ),
            burstiness: VarianceUtils.humanLikelihoodScore(
                stats.burstiness.sentenceLength,
                VarianceUtils.HUMAN_BASELINES.burstiness
            ),
            zipfSlope: VarianceUtils.humanLikelihoodScore(
                stats.zipf.slope,
                VarianceUtils.HUMAN_BASELINES.zipfSlope
            ),
            ttr: VarianceUtils.humanLikelihoodScore(
                stats.vocabulary.typeTokenRatio,
                VarianceUtils.HUMAN_BASELINES.ttrNormalized
            )
        };
        
        // Average human likelihood (overall "normalcy" score)
        const hlScores = Object.values(stats.humanLikelihood);
        stats.overallHumanLikelihood = hlScores.reduce((a, b) => a + b, 0) / hlScores.length;

        // === AI Signature Metrics ===
        stats.aiSignatures = this.computeAISignatureMetrics(text, tokens, sentences);

        return stats;
    },

    /**
     * Bin continuous values into histogram buckets
     */
    binValues(values, numBins) {
        if (!values || values.length === 0) return [];
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / numBins || 1;
        
        const bins = new Array(numBins).fill(0);
        
        values.forEach(v => {
            const binIndex = Math.min(numBins - 1, Math.floor((v - min) / binWidth));
            bins[binIndex]++;
        });
        
        return bins;
    },
    /**
     * Compute AI-specific signature metrics
     */
    computeAISignatureMetrics(text, tokens, sentences) {
        const metrics = {};

        // Hedging word density
        const hedgingCount = Utils.hedgingWords.reduce((count, word) => {
            return count + (text.toLowerCase().match(new RegExp(`\\b${word}\\b`, 'gi')) || []).length;
        }, 0);
        metrics.hedgingDensity = tokens.length > 0 ? hedgingCount / tokens.length : 0;

        // Discourse marker density
        const discourseCount = Utils.discourseMarkers.reduce((count, marker) => {
            return count + (text.toLowerCase().includes(marker.toLowerCase()) ? 1 : 0);
        }, 0);
        metrics.discourseMarkerDensity = sentences.length > 0 ? discourseCount / sentences.length : 0;

        // Unicode anomalies (strong AI indicator)
        const unicodeAnomalies = (text.match(/[\u2014\u2013\u2018\u2019\u201C\u201D\u2026\u2E3A\u2E3B]/g) || []).length;
        metrics.unicodeAnomalyDensity = text.length > 0 ? unicodeAnomalies / text.length * 1000 : 0;

        // Decorative dividers (very strong AI indicator)
        const decorativeDividers = (text.match(/[⸻═━─]{3,}|[•●◦◆◇■□▪▫]{3,}/g) || []).length;
        metrics.decorativeDividerCount = decorativeDividers;

        // Perfect grammar indicators (lack of contractions)
        const contractionCount = (text.match(/\b(don't|won't|can't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|I'm|you're|we're|they're|he's|she's|it's|that's|there's|here's|what's|who's|let's)\b/gi) || []).length;
        metrics.contractionRate = sentences.length > 0 ? contractionCount / sentences.length : 0;

        // Sentence starter variety
        const starters = sentences.map(s => Utils.tokenize(s)[0]?.toLowerCase()).filter(Boolean);
        const uniqueStarters = new Set(starters).size;
        metrics.sentenceStarterVariety = starters.length > 0 ? uniqueStarters / starters.length : 0;

        // Passive voice indicators
        const passiveIndicators = (text.match(/\b(is|are|was|were|been|being)\s+(being\s+)?\w+ed\b/gi) || []).length;
        metrics.passiveVoiceRate = sentences.length > 0 ? passiveIndicators / sentences.length : 0;

        return metrics;
    },

    /**
     * Estimate total syllables in tokens
     */
    estimateSyllables(tokens) {
        return tokens.reduce((total, token) => total + this.syllableCount(token), 0);
    },

    /**
     * Estimate syllable count for a word
     */
    syllableCount(word) {
        word = word.toLowerCase().replace(/[^a-z]/g, '');
        if (word.length <= 3) return 1;
        
        // Count vowel groups
        const vowels = word.match(/[aeiouy]+/g);
        let count = vowels ? vowels.length : 1;
        
        // Subtract silent e
        if (word.endsWith('e') && !word.endsWith('le')) count--;
        
        // Ensure at least 1 syllable
        return Math.max(1, count);
    },

    /**
     * Compute variance metrics from uniformity scores
     */
    computeVarianceMetrics(uniformityScores) {
        const values = Object.values(uniformityScores).filter(v => typeof v === 'number');
        if (values.length === 0) return null;

        return {
            meanUniformity: Utils.mean(values),
            uniformityVariance: Utils.variance(values),
            maxUniformity: Math.max(...values),
            minUniformity: Math.min(...values),
            uniformityRange: Math.max(...values) - Math.min(...values)
        };
    },

    /**
     * Calculate overall AI probability using variance-based methodology
     * Key principle: High uniformity = AI, High variance = Human
     * 
     * ENHANCED v2.2: Uses Gaussian bell curve philosophy
     * - Humans fall in the "reasonable middle" of a normal distribution
     * - BOTH extremes (too perfect AND too chaotic) are suspicious
     * - Bayesian combination with calibrated confidence
     */
    calculateVarianceBasedProbability(categoryResults) {
        const validResults = categoryResults.filter(r => 
            r.confidence > 0.2 && !r.error && r.aiProbability !== undefined
        );

        if (validResults.length === 0) {
            return { 
                aiProbability: 0.5, 
                confidence: 0, 
                mixedProbability: 0,
                varianceProfile: null,
                featureContributions: [],
                combinationMethod: 'none'
            };
        }

        // Calculate feature contributions using category weights
        const featureContributions = [];
        const probabilities = [];
        const weights = [];

        for (const result of validResults) {
            // Map categories to weight keys
            const weightKey = this.getCategoryWeightKey(result.category, result.name);
            const baseWeight = this.categoryWeights[weightKey] || 0.1;
            
            // Apply confidence as a weight multiplier
            const effectiveWeight = baseWeight * result.confidence;
            
            probabilities.push(result.aiProbability);
            weights.push(effectiveWeight);
            
            featureContributions.push({
                category: result.category,
                name: result.name,
                aiProbability: result.aiProbability,
                weight: effectiveWeight,
                contribution: result.aiProbability * effectiveWeight,
                uniformityScores: result.uniformityScores || null
            });
        }

        // Use Bayesian combination for more accurate probability aggregation
        // This handles correlated signals better than weighted average
        const bayesianProb = VarianceUtils.bayesianCombine(probabilities, weights);
        
        // Also calculate weighted average for comparison
        const weightedAvgProb = weights.reduce((sum, w, i) => sum + w * probabilities[i], 0) / 
                               weights.reduce((sum, w) => sum + w, 0);
        
        // Blend methods: Bayesian for extremes, weighted for middle
        // This prevents over-confidence at the extremes
        const blendFactor = Math.abs(bayesianProb - 0.5) * 2; // 0 at center, 1 at extremes
        const aiProbability = bayesianProb * blendFactor + weightedAvgProb * (1 - blendFactor);
        
        // Calculate calibrated confidence based on multiple factors
        const confidence = this.calculateCalibratedConfidence(validResults, probabilities);
        
        // Mixed probability: higher when analyzers disagree
        const stdDev = Utils.standardDeviation(probabilities);
        const mixedProbability = Math.min(0.35, stdDev * 1.5);

        // Build variance profile
        const varianceProfile = this.buildVarianceProfile(validResults);

        return {
            aiProbability: Math.max(0, Math.min(1, aiProbability)),
            confidence,
            mixedProbability,
            varianceProfile,
            featureContributions: featureContributions.sort((a, b) => b.contribution - a.contribution),
            combinationMethod: 'bayesian-blend',
            rawBayesian: bayesianProb,
            rawWeighted: weightedAvgProb
        };
    },

    /**
     * Calculate calibrated confidence score
     * Based on: analyzer agreement, sample sizes, signal strength
     */
    calculateCalibratedConfidence(results, probabilities) {
        // Factor 1: Analyzer agreement (low std dev = high agreement)
        const stdDev = Utils.standardDeviation(probabilities);
        const agreementScore = 1 - Math.min(1, stdDev * 2.5);
        
        // Factor 2: Average analyzer confidence
        const avgAnalyzerConf = Utils.mean(results.map(r => r.confidence));
        
        // Factor 3: Signal strength (distance from 0.5)
        const avgProb = Utils.mean(probabilities);
        const signalStrength = Math.abs(avgProb - 0.5) * 2;
        
        // Factor 4: Number of analyzers (more = more confident)
        const analyzerCountFactor = Math.min(1, results.length / 10);
        
        // Factor 5: Consistency of direction (all pointing same way)
        const aiLeaning = probabilities.filter(p => p > 0.5).length / probabilities.length;
        const directionConsistency = Math.abs(aiLeaning - 0.5) * 2;
        
        // Weighted combination
        const calibratedConfidence = (
            agreementScore * 0.25 +
            avgAnalyzerConf * 0.25 +
            signalStrength * 0.15 +
            analyzerCountFactor * 0.15 +
            directionConsistency * 0.20
        );
        
        return Math.max(0, Math.min(1, calibratedConfidence));
    },

    /**
     * Map category number/name to weight key
     */
    getCategoryWeightKey(category, name) {
        const nameLower = (name || '').toLowerCase();
        
        if (nameLower.includes('syntax') || category === 2) return 'syntaxVariance';
        if (nameLower.includes('lexical') || category === 3) return 'lexicalDiversity';
        if (nameLower.includes('repetition') || category === 12) return 'repetitionUniformity';
        if (nameLower.includes('tone') || category === 13) return 'toneStability';
        if (nameLower.includes('grammar') || category === 1) return 'grammarEntropy';
        if (nameLower.includes('statistical') || nameLower.includes('perplexity') || category === 8) return 'perplexity';
        if (nameLower.includes('authorship') || category === 9) return 'authorshipDrift';
        if (nameLower.includes('metadata') || nameLower.includes('formatting') || category === 11) return 'metadataFormatting';
        
        return 'grammarEntropy'; // default
    },

    /**
     * Build variance profile summarizing key metrics
     */
    buildVarianceProfile(results) {
        const profile = {
            sentenceLengthVariance: null,
            vocabularyDiversity: null,
            toneStability: null,
            repetitionUniformity: null,
            overallUniformity: null
        };

        for (const result of results) {
            if (result.uniformityScores) {
                if (result.uniformityScores.sentenceLength !== undefined) {
                    profile.sentenceLengthVariance = result.uniformityScores.sentenceLength;
                }
                if (result.uniformityScores.vocabulary !== undefined) {
                    profile.vocabularyDiversity = 1 - result.uniformityScores.vocabulary;
                }
                if (result.uniformityScores.overall !== undefined) {
                    profile.overallUniformity = result.uniformityScores.overall;
                }
            }
            
            if (result.stability?.overall !== undefined) {
                profile.toneStability = result.stability.overall;
            }
        }

        // Calculate overall uniformity if not set
        const uniformityValues = Object.values(profile).filter(v => v !== null);
        if (profile.overallUniformity === null && uniformityValues.length > 0) {
            profile.overallUniformity = Utils.mean(uniformityValues);
        }

        return profile;
    },

    /**
     * Compute confidence interval for the AI probability estimate
     */
    computeConfidenceInterval(probability, sampleSize, confidence) {
        // Use Wilson score interval approximation
        const z = 1.96; // 95% confidence
        const n = Math.max(sampleSize, 1);
        const p = probability;
        
        const denominator = 1 + z * z / n;
        const center = (p + z * z / (2 * n)) / denominator;
        const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator;
        
        // Adjust based on confidence level
        const adjustedMargin = margin * (2 - confidence);
        
        return {
            lower: Math.max(0, center - adjustedMargin),
            upper: Math.min(1, center + adjustedMargin),
            center,
            margin: adjustedMargin
        };
    },

    /**
     * Assess false positive risk based on edge cases
     */
    assessFalsePositiveRisk(categoryResults, text) {
        const risks = [];
        
        // Check for academic/formal writing (often mistaken for AI)
        const formalIndicators = (text.match(/\b(furthermore|moreover|consequently|thus|hence|therefore)\b/gi) || []).length;
        if (formalIndicators > 3) {
            risks.push({
                type: 'formal_writing',
                message: 'Formal/academic writing style may increase false positive rate',
                severity: 'medium'
            });
        }

        // Check for short text (less reliable)
        const wordCount = Utils.tokenize(text).length;
        if (wordCount < 100) {
            risks.push({
                type: 'short_text',
                message: 'Short texts have lower detection accuracy',
                severity: 'high'
            });
        }

        // Check for technical/specialized content
        const technicalTerms = (text.match(/\b[A-Z]{2,}[a-z]*\b/g) || []).length;
        if (technicalTerms > 5) {
            risks.push({
                type: 'technical_content',
                message: 'Technical jargon may affect detection accuracy',
                severity: 'low'
            });
        }

        // Check for disagreement between analyzers
        const validResults = categoryResults.filter(r => r.confidence > 0.3);
        if (validResults.length > 3) {
            const probs = validResults.map(r => r.aiProbability);
            const spread = Math.max(...probs) - Math.min(...probs);
            if (spread > 0.4) {
                risks.push({
                    type: 'analyzer_disagreement',
                    message: 'High disagreement between detection methods',
                    severity: 'medium'
                });
            }
        }

        return {
            hasRisks: risks.length > 0,
            risks,
            overallRisk: risks.length === 0 ? 'low' : 
                         risks.some(r => r.severity === 'high') ? 'high' : 'medium'
        };
    },

    /**
     * Score individual sentences for highlighting
     */
    scoreSentences(text, sentences, categoryResults) {
        const scores = [];
        
        // Get statistical sentence scores if available
        const statResult = categoryResults.find(r => r.category === 8);
        const perplexityScores = statResult?.sentenceScores || [];

        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i];
            const tokens = Utils.tokenize(sentence);
            
            // Calculate multiple indicators for this sentence
            const indicators = {};
            
            // 1. Length regularity (compare to neighbors)
            if (i > 0 && i < sentences.length - 1) {
                const prevLen = Utils.tokenize(sentences[i-1]).length;
                const nextLen = Utils.tokenize(sentences[i+1]).length;
                const avgNeighbor = (prevLen + nextLen) / 2;
                const lengthDiff = Math.abs(tokens.length - avgNeighbor) / avgNeighbor;
                indicators.lengthRegularity = 1 - Math.min(1, lengthDiff);
            } else {
                indicators.lengthRegularity = 0.5;
            }

            // 2. Lexical diversity
            const uniqueRatio = tokens.length > 0 
                ? new Set(tokens).size / tokens.length 
                : 0.5;
            indicators.lexicalDiversity = uniqueRatio;

            // 3. Discourse markers
            const discourseMarkers = Utils.discourseMarkers.filter(m => 
                sentence.toLowerCase().includes(m.toLowerCase())
            );
            indicators.discourseMarkerDensity = Math.min(1, discourseMarkers.length * 0.3);

            // 4. Perplexity proxy (from statistical analyzer)
            if (perplexityScores[i]) {
                indicators.perplexity = perplexityScores[i].score / 100;
            }

            // 5. Personal pronouns (human indicator)
            const personalPronouns = (sentence.match(/\b(I|my|me|we|our|us)\b/gi) || []).length;
            indicators.personalLanguage = Math.min(1, personalPronouns * 0.2);

            // 6. Hedging language
            const hedgingWords = ['perhaps', 'maybe', 'possibly', 'might', 'could', 'may', 'seems', 'appears'];
            const hedgingCount = hedgingWords.filter(h => 
                sentence.toLowerCase().includes(h)
            ).length;
            indicators.hedging = Math.min(1, hedgingCount * 0.25);

            // Calculate sentence AI probability
            // Weight different indicators
            const sentenceAiProb = (
                (1 - indicators.lengthRegularity) * 0.15 + // High regularity = AI
                (1 - indicators.lexicalDiversity) * 0.15 + // Low diversity = AI
                indicators.discourseMarkerDensity * 0.25 + // Many markers = AI
                (1 - (indicators.perplexity || 0.5)) * 0.15 + // Low perplexity = AI
                (1 - indicators.personalLanguage) * 0.15 + // No personal = AI
                indicators.hedging * 0.15 // Hedging = AI
            );

            // Classify sentence
            let classification;
            if (sentenceAiProb < 0.35) {
                classification = 'human';
            } else if (sentenceAiProb > 0.6) {
                classification = 'ai';
            } else {
                classification = 'mixed';
            }

            scores.push({
                index: i,
                text: sentence,
                aiProbability: sentenceAiProb,
                classification,
                indicators
            });
        }

        return scores;
    },

    /**
     * Get verdict based on AI probability and confidence
     * Now includes probability bands and confidence qualifiers
     */
    getVerdict(aiProbability, confidence = 0.5) {
        // Confidence qualifier
        const confidenceLevel = confidence < 0.4 ? 'Low confidence: ' : 
                               confidence < 0.7 ? '' : 
                               'High confidence: ';

        // Probability bands with more nuanced thresholds
        if (aiProbability < 0.15) {
            return {
                label: confidenceLevel + 'Human Written',
                description: 'This text shows strong, consistent human-writing characteristics',
                level: 'human',
                band: 'definitely-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.30) {
            return {
                label: confidenceLevel + 'Likely Human',
                description: 'This text exhibits predominantly human patterns with minimal AI signals',
                level: 'probably-human',
                band: 'likely-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.45) {
            return {
                label: confidenceLevel + 'Possibly Human',
                description: 'This text appears mostly human but has some uncertain elements',
                level: 'leaning-human',
                band: 'possibly-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.55) {
            return {
                label: 'Inconclusive',
                description: 'This text shows mixed signals — could be human, AI, or a mixture',
                level: 'mixed',
                band: 'inconclusive',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.70) {
            return {
                label: confidenceLevel + 'Possibly AI',
                description: 'This text has notable AI-typical patterns but some human elements',
                level: 'leaning-ai',
                band: 'possibly-ai',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.85) {
            return {
                label: confidenceLevel + 'Likely AI',
                description: 'This text exhibits strong AI-generated characteristics',
                level: 'probably-ai',
                band: 'likely-ai',
                probability: aiProbability,
                confidence
            };
        } else {
            return {
                label: confidenceLevel + 'AI Generated',
                description: 'This text shows overwhelming AI-generated patterns',
                level: 'ai',
                band: 'definitely-ai',
                probability: aiProbability,
                confidence
            };
        }
    },

    /**
     * Get feature analysis summary for feature tab
     */
    getFeatureSummary(categoryResults) {
        return categoryResults.map(result => ({
            category: result.category,
            name: result.name,
            aiProbability: result.aiProbability,
            confidence: result.confidence,
            topFindings: (result.findings || []).slice(0, 3),
            scores: result.scores
        }));
    },

    /**
     * Generate detailed report
     */
    generateReport(analysisResult) {
        const sections = [];

        for (const category of analysisResult.categoryResults) {
            sections.push({
                number: category.category,
                name: category.name,
                aiScore: Math.round(category.aiProbability * 100),
                confidence: Math.round(category.confidence * 100),
                findings: category.findings || [],
                details: category.details || {}
            });
        }

        return {
            overall: {
                aiProbability: Math.round(analysisResult.aiProbability * 100),
                verdict: analysisResult.verdict,
                confidence: Math.round(analysisResult.confidence * 100)
            },
            sections,
            stats: analysisResult.stats,
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Get empty result for empty input
     */
    getEmptyResult() {
        return {
            aiProbability: 0.5,
            humanProbability: 0.5,
            mixedProbability: 0,
            confidence: 0,
            verdict: {
                label: 'No Input',
                description: 'Please enter text to analyze',
                level: 'none'
            },
            categoryResults: [],
            sentences: [],
            sentenceScores: [],
            findings: [],
            stats: {
                characters: 0,
                words: 0,
                sentences: 0,
                paragraphs: 0
            },
            analysisTime: '0ms'
        };
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnalyzerEngine;
}
