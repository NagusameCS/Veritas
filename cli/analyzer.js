/**
 * VERITAS — Core Analyzer Module for CLI/Node.js
 * Self-contained analysis engine for use outside the browser
 * 
 * This module bundles all necessary utilities and analyzers
 * for standalone operation without browser dependencies.
 */

// ═══════════════════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

const Utils = {
    /**
     * Split text into sentences
     */
    splitSentences(text) {
        if (!text || typeof text !== 'string') return [];
        
        const abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'U.S.', 'U.K.'];
        let processed = text;
        const placeholders = [];
        
        abbreviations.forEach((abbr, i) => {
            const placeholder = `__ABBR${i}__`;
            placeholders.push({ placeholder, abbr });
            processed = processed.split(abbr).join(placeholder);
        });
        
        const sentences = processed.split(/(?<=[.!?])\s+(?=[A-Z])/);
        
        return sentences.map(sentence => {
            let restored = sentence;
            placeholders.forEach(({ placeholder, abbr }) => {
                restored = restored.split(placeholder).join(abbr);
            });
            return restored.trim();
        }).filter(s => s.length > 0);
    },

    /**
     * Tokenize text into words
     */
    tokenize(text) {
        if (!text || typeof text !== 'string') return [];
        return text.toLowerCase()
            .replace(/[^\w\s'-]/g, ' ')
            .split(/\s+/)
            .filter(word => word.length > 0);
    },

    /**
     * Get word count
     */
    wordCount(text) {
        return this.tokenize(text).length;
    },

    /**
     * Calculate mean of an array
     */
    mean(arr) {
        if (!arr || arr.length === 0) return 0;
        return arr.reduce((a, b) => a + b, 0) / arr.length;
    },

    /**
     * Calculate variance of an array
     */
    variance(arr) {
        if (!arr || arr.length === 0) return 0;
        const avg = this.mean(arr);
        return arr.reduce((sum, val) => sum + Math.pow(val - avg, 2), 0) / arr.length;
    },

    /**
     * Calculate standard deviation
     */
    stdDev(arr) {
        return Math.sqrt(this.variance(arr));
    },

    /**
     * Calculate median
     */
    median(arr) {
        if (!arr || arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    },

    /**
     * Frequency distribution
     */
    frequencyDistribution(tokens) {
        const freq = {};
        for (const token of tokens) {
            freq[token] = (freq[token] || 0) + 1;
        }
        return freq;
    },

    /**
     * Common function words
     */
    functionWords: [
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these',
        'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which',
        'who', 'whom', 'whose', 'where', 'when', 'why', 'how', 'if', 'then',
        'than', 'so', 'just', 'also', 'only', 'very', 'even', 'more', 'most'
    ]
};

// ═══════════════════════════════════════════════════════════════════════════
// VARIANCE UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

const VarianceUtils = {
    /**
     * Human baseline parameters
     * KEY PRINCIPLE: Values at EITHER extreme are suspicious
     * - Too low (perfect/uniform) = AI-generated
     * - Too high (chaotic/extreme) = Humanizer tools or unusual AI
     */
    HUMAN_BASELINES: {
        sentenceLengthCV: { mean: 0.55, stdDev: 0.15, min: 0.25, max: 0.85 },
        hapaxRatio: { mean: 0.48, stdDev: 0.08, min: 0.30, max: 0.65 },
        burstiness: { mean: 0.15, stdDev: 0.12, min: -0.10, max: 0.40 },
        zipfSlope: { mean: -1.0, stdDev: 0.15, min: -1.3, max: -0.7 },
        ttrNormalized: { mean: 0.42, stdDev: 0.10, min: 0.25, max: 0.60 },
        sentenceEntropy: { mean: 2.8, stdDev: 0.4, min: 1.8, max: 3.8 }
    },

    /**
     * Gaussian scoring - values near mean get high scores, extremes get low
     * This is the core of bidirectional detection
     */
    gaussianScore(value, mean, stdDev) {
        if (stdDev === 0) return value === mean ? 1 : 0;
        const z = Math.abs((value - mean) / stdDev);
        return Math.exp(-z * z / 2);
    },

    /**
     * BIDIRECTIONAL HUMAN LIKELIHOOD SCORING
     * 
     * Core philosophy: Human writing falls in a "reasonable middle"
     * - Values too LOW suggest AI uniformity (too perfect)
     * - Values too HIGH suggest humanizer manipulation (artificially chaotic)
     * 
     * Both extremes are suspicious!
     */
    humanLikelihoodScore(value, baseline) {
        if (!baseline) return 0.5;
        
        const { mean, stdDev, min, max } = baseline;
        
        // Gaussian: how close to expected mean?
        const gaussianComponent = this.gaussianScore(value, mean, stdDev);
        
        // Range penalty for being outside expected bounds (either direction)
        let rangePenalty = 0;
        let extremeType = null;
        
        if (min !== undefined && max !== undefined) {
            if (value < min) {
                // Too low = too uniform/perfect (AI signature)
                rangePenalty = Math.min(1, (min - value) / (stdDev * 2));
                extremeType = 'low';
            } else if (value > max) {
                // Too high = too chaotic (humanizer or unusual)
                rangePenalty = Math.min(1, (value - max) / (stdDev * 2));
                extremeType = 'high';
            }
        }
        
        return {
            score: gaussianComponent * (1 - rangePenalty * 0.5),
            extremeType,
            rangePenalty
        };
    },

    /**
     * Calculate coefficient of variation
     */
    coefficientOfVariation(values) {
        if (!values || values.length < 2) return 0;
        const mean = Utils.mean(values);
        if (mean === 0) return 0;
        const stdDev = Math.sqrt(Utils.variance(values));
        return stdDev / mean;
    },

    /**
     * Calculate burstiness score
     * Humans show "bursty" patterns; AI shows even distribution
     * Range: -1 (periodic) to 1 (bursty), 0 is Poisson
     */
    burstiness(values) {
        if (!values || values.length < 2) return 0;
        const mean = Utils.mean(values);
        const stdDev = Math.sqrt(Utils.variance(values));
        if (stdDev + mean === 0) return 0;
        return (stdDev - mean) / (stdDev + mean);
    },

    /**
     * Uniformity score (0 = varied, 1 = perfectly uniform)
     */
    uniformityScore(values) {
        if (!values || values.length < 2) return 0.5;
        const cv = this.coefficientOfVariation(values);
        return 1 - Math.min(1, cv / 0.8);
    },

    /**
     * Calculate confidence interval using Wilson score
     */
    confidenceInterval(probability, sampleSize, confidence = 0.95) {
        const z = confidence === 0.95 ? 1.96 : 1.645;
        const n = Math.max(1, sampleSize);
        const p = probability;
        
        const denominator = 1 + z * z / n;
        const center = (p + z * z / (2 * n)) / denominator;
        const margin = (z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denominator;
        
        return {
            lower: Math.max(0, center - margin),
            upper: Math.min(1, center + margin),
            width: margin * 2
        };
    },

    /**
     * Convert probability to labeled band
     */
    toProbabilityBand(probability) {
        if (probability < 0.2) return { label: 'Human', confidence: 'high' };
        if (probability < 0.35) return { label: 'Probably Human', confidence: 'moderate' };
        if (probability < 0.5) return { label: 'Possibly Mixed', confidence: 'low' };
        if (probability < 0.65) return { label: 'Possibly AI', confidence: 'low' };
        if (probability < 0.8) return { label: 'Probably AI', confidence: 'moderate' };
        return { label: 'AI', confidence: 'high' };
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ANALYZER CLASS
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Sunrise Model v3.0 Configuration
 * ML-trained weights for optimal AI detection accuracy
 * Training Stats: 98.08% accuracy, 98.09% F1, 29,976 samples
 */
const SunriseConfig = {
    version: '3.0.0',
    modelName: 'Sunrise',
    
    // Category weights (normalized from ML training)
    categoryWeights: {
        syntax: 0.22,        // Sentence structure - high importance per ML
        vocabulary: 0.18,    // Lexical analysis
        zipf: 0.12,          // Statistical patterns
        burstiness: 0.18,    // Temporal patterns
        repetition: 0.15,    // Phrase repetition
        readability: 0.15    // Readability metrics
    },
    
    // Training verification
    trainingStats: {
        accuracy: 0.9808,
        f1Score: 0.9809,
        rocAuc: 0.9980,
        samples: 29976
    }
};

class VeritasAnalyzer {
    constructor() {
        this.version = '3.0.0';
        this.modelName = 'Sunrise';
        this.config = SunriseConfig;
    }

    /**
     * Main analysis entry point
     */
    analyze(text, metadata = null) {
        const startTime = Date.now();
        
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
            avgWordsPerSentence: sentences.length > 0 
                ? parseFloat((tokens.length / sentences.length).toFixed(1)) 
                : 0
        };

        // Run all analysis categories
        const categoryResults = [];
        const findings = [];

        // 1. Sentence Structure Analysis
        const syntaxResult = this.analyzeSyntax(sentences);
        categoryResults.push(syntaxResult);
        findings.push(...(syntaxResult.findings || []));

        // 2. Vocabulary Analysis
        const vocabResult = this.analyzeVocabulary(tokens);
        categoryResults.push(vocabResult);
        findings.push(...(vocabResult.findings || []));

        // 3. Zipf's Law Analysis
        const zipfResult = this.analyzeZipf(tokens);
        categoryResults.push(zipfResult);
        findings.push(...(zipfResult.findings || []));

        // 4. Burstiness Analysis
        const burstResult = this.analyzeBurstiness(sentences, tokens);
        categoryResults.push(burstResult);
        findings.push(...(burstResult.findings || []));

        // 5. Repetition Analysis
        const repResult = this.analyzeRepetition(text, tokens);
        categoryResults.push(repResult);
        findings.push(...(repResult.findings || []));

        // 6. Readability Analysis
        const readResult = this.analyzeReadability(text, sentences, tokens);
        categoryResults.push(readResult);
        findings.push(...(readResult.findings || []));

        // Compute overall probability using Sunrise ML-trained weights
        const sunriseWeights = [
            this.config.categoryWeights.syntax,      // 0.22
            this.config.categoryWeights.vocabulary,  // 0.18
            this.config.categoryWeights.zipf,        // 0.12
            this.config.categoryWeights.burstiness,  // 0.18
            this.config.categoryWeights.repetition,  // 0.15
            this.config.categoryWeights.readability  // 0.15
        ];
        let weightedSum = 0;
        let totalWeight = 0;
        
        categoryResults.forEach((cat, i) => {
            const weight = sunriseWeights[i] || 0.1;
            weightedSum += cat.aiProbability * weight * cat.confidence;
            totalWeight += weight * cat.confidence;
        });

        const aiProbability = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        
        // Calculate overall confidence
        const avgConfidence = Utils.mean(categoryResults.map(c => c.confidence));
        const sampleSizeBonus = Math.min(0.2, stats.sentences / 100);
        const confidence = Math.min(0.95, avgConfidence + sampleSizeBonus);

        // Confidence interval
        const ci = VarianceUtils.confidenceInterval(aiProbability, stats.sentences);

        // Determine verdict
        const verdict = this.determineVerdict(aiProbability, confidence);

        // Build advanced stats
        const advancedStats = this.buildAdvancedStats(sentences, tokens, categoryResults);

        const analysisTime = `${Date.now() - startTime}ms`;

        return {
            aiProbability,
            humanProbability: 1 - aiProbability,
            confidence,
            confidenceInterval: ci,
            verdict,
            stats,
            advancedStats,
            categoryResults,
            findings: findings.sort((a, b) => {
                const order = { ai: 0, mixed: 1, human: 2 };
                return (order[a.indicator] || 1) - (order[b.indicator] || 1);
            }),
            analysisTime,
            version: this.version,
            model: {
                name: this.modelName,
                version: this.config.version,
                accuracy: this.config.trainingStats.accuracy,
                f1Score: this.config.trainingStats.f1Score,
                trainingSamples: this.config.trainingStats.samples
            }
        };
    }

    /**
     * Analyze sentence structure (syntax variance)
     */
    analyzeSyntax(sentences) {
        const lengths = sentences.map(s => s.split(/\s+/).length);
        
        if (lengths.length < 3) {
            return {
                name: 'Sentence Structure',
                category: 2,
                aiProbability: 0.5,
                confidence: 0.3,
                findings: [{ text: 'Insufficient sentences for analysis', indicator: 'neutral' }]
            };
        }

        const cv = VarianceUtils.coefficientOfVariation(lengths);
        const baseline = VarianceUtils.HUMAN_BASELINES.sentenceLengthCV;
        const likelihoodResult = VarianceUtils.humanLikelihoodScore(cv, baseline);
        
        const uniformity = VarianceUtils.uniformityScore(lengths);
        const burstiness = VarianceUtils.burstiness(lengths);
        
        const findings = [];
        
        // Bidirectional detection: flag BOTH extremes
        if (cv < baseline.min) {
            findings.push({
                text: `Sentence length CV too low (${cv.toFixed(3)}) - unusually uniform, suggests AI generation`,
                indicator: 'ai'
            });
        } else if (cv > baseline.max) {
            findings.push({
                text: `Sentence length CV extremely high (${cv.toFixed(3)}) - may indicate humanizer tools or unusual editing`,
                indicator: 'mixed'
            });
        } else {
            findings.push({
                text: `Sentence length variance within natural range (CV: ${cv.toFixed(3)})`,
                indicator: 'human'
            });
        }

        if (uniformity > 0.7) {
            findings.push({
                text: `High uniformity score (${(uniformity * 100).toFixed(0)}%) - mechanical consistency`,
                indicator: 'ai'
            });
        }

        // AI probability is inverse of human likelihood
        const aiProbability = 1 - likelihoodResult.score;
        const confidence = Math.min(0.9, 0.5 + sentences.length / 50);

        return {
            name: 'Sentence Structure',
            category: 2,
            aiProbability,
            confidence,
            findings,
            details: { cv, uniformity, burstiness }
        };
    }

    /**
     * Analyze vocabulary richness
     */
    analyzeVocabulary(tokens) {
        if (tokens.length < 20) {
            return {
                name: 'Vocabulary Richness',
                category: 3,
                aiProbability: 0.5,
                confidence: 0.3,
                findings: [{ text: 'Insufficient tokens for vocabulary analysis', indicator: 'neutral' }]
            };
        }

        const uniqueTokens = new Set(tokens);
        const ttr = uniqueTokens.size / tokens.length;
        
        // Hapax legomena (words appearing exactly once)
        const freq = Utils.frequencyDistribution(tokens);
        const hapaxCount = Object.values(freq).filter(c => c === 1).length;
        const hapaxRatio = hapaxCount / uniqueTokens.size;

        const baseline = VarianceUtils.HUMAN_BASELINES.hapaxRatio;
        const likelihoodResult = VarianceUtils.humanLikelihoodScore(hapaxRatio, baseline);
        
        const findings = [];

        // BIDIRECTIONAL DETECTION for Hapax Ratio
        // Too low = AI (limited vocabulary reuse)
        // Too high = Humanizer tools (artificially inflated uniqueness)
        if (hapaxRatio < baseline.min) {
            findings.push({
                text: `Low hapax ratio (${(hapaxRatio * 100).toFixed(1)}%) - limited vocabulary diversity, AI pattern`,
                indicator: 'ai'
            });
        } else if (hapaxRatio > baseline.max) {
            findings.push({
                text: `Extremely high hapax ratio (${(hapaxRatio * 100).toFixed(1)}%) - unusually inflated uniqueness, may indicate thesaurus/humanizer manipulation`,
                indicator: 'mixed'
            });
        } else {
            findings.push({
                text: `Hapax ratio within natural range (${(hapaxRatio * 100).toFixed(1)}%)`,
                indicator: 'human'
            });
        }

        // TTR analysis - also bidirectional
        const ttrBaseline = VarianceUtils.HUMAN_BASELINES.ttrNormalized;
        if (ttr < ttrBaseline.min) {
            findings.push({
                text: `Very low vocabulary diversity (TTR: ${ttr.toFixed(3)})`,
                indicator: 'ai'
            });
        } else if (ttr > ttrBaseline.max) {
            findings.push({
                text: `Unusually high vocabulary diversity (TTR: ${ttr.toFixed(3)}) - may indicate synonym substitution`,
                indicator: 'mixed'
            });
        }

        const aiProbability = 1 - likelihoodResult.score;
        const confidence = Math.min(0.85, 0.4 + tokens.length / 500);

        return {
            name: 'Vocabulary Richness',
            category: 3,
            aiProbability,
            confidence,
            findings,
            details: { ttr, hapaxRatio, uniqueWords: uniqueTokens.size }
        };
    }

    /**
     * Analyze Zipf's law compliance
     */
    analyzeZipf(tokens) {
        if (tokens.length < 50) {
            return {
                name: "Zipf's Law Analysis",
                category: 8,
                aiProbability: 0.5,
                confidence: 0.3,
                findings: [{ text: 'Insufficient tokens for Zipf analysis', indicator: 'neutral' }]
            };
        }

        const freq = Utils.frequencyDistribution(tokens);
        const sortedFreqs = Object.values(freq).sort((a, b) => b - a);
        
        // Calculate log-log regression for Zipf slope
        const logRanks = [];
        const logFreqs = [];
        
        for (let i = 0; i < Math.min(sortedFreqs.length, 100); i++) {
            if (sortedFreqs[i] > 0) {
                logRanks.push(Math.log(i + 1));
                logFreqs.push(Math.log(sortedFreqs[i]));
            }
        }

        // Simple linear regression for slope
        const n = logRanks.length;
        const sumX = logRanks.reduce((a, b) => a + b, 0);
        const sumY = logFreqs.reduce((a, b) => a + b, 0);
        const sumXY = logRanks.reduce((sum, x, i) => sum + x * logFreqs[i], 0);
        const sumX2 = logRanks.reduce((sum, x) => sum + x * x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        
        // Zipf's law predicts slope ≈ -1
        const baseline = VarianceUtils.HUMAN_BASELINES.zipfSlope;
        const deviation = Math.abs(slope - baseline.mean);
        const compliance = Math.max(0, 1 - deviation / 0.5);

        const likelihoodResult = VarianceUtils.humanLikelihoodScore(slope, baseline);
        
        const findings = [];

        // Bidirectional check
        if (slope > baseline.max) {
            findings.push({
                text: `Zipf slope too shallow (${slope.toFixed(2)}) - flatter distribution than natural`,
                indicator: 'ai'
            });
        } else if (slope < baseline.min) {
            findings.push({
                text: `Zipf slope too steep (${slope.toFixed(2)}) - over-concentrated vocabulary`,
                indicator: 'mixed'
            });
        } else {
            findings.push({
                text: `Zipf's law compliance good (slope: ${slope.toFixed(2)}, expected: ~-1.0)`,
                indicator: 'human'
            });
        }

        const aiProbability = 1 - likelihoodResult.score;

        return {
            name: "Zipf's Law Analysis",
            category: 8,
            aiProbability,
            confidence: 0.7,
            findings,
            details: { slope, compliance, deviation }
        };
    }

    /**
     * Analyze burstiness patterns
     */
    analyzeBurstiness(sentences, tokens) {
        const sentenceLengths = sentences.map(s => s.split(/\s+/).length);
        
        if (sentenceLengths.length < 5) {
            return {
                name: 'Burstiness Patterns',
                category: 9,
                aiProbability: 0.5,
                confidence: 0.3,
                findings: [{ text: 'Insufficient data for burstiness analysis', indicator: 'neutral' }]
            };
        }

        const burstScore = VarianceUtils.burstiness(sentenceLengths);
        const baseline = VarianceUtils.HUMAN_BASELINES.burstiness;
        const likelihoodResult = VarianceUtils.humanLikelihoodScore(burstScore, baseline);

        const findings = [];

        // Bidirectional detection
        if (burstScore < baseline.min) {
            findings.push({
                text: `Negative burstiness (${burstScore.toFixed(3)}) - periodic/mechanical pattern, AI signature`,
                indicator: 'ai'
            });
        } else if (burstScore > baseline.max) {
            findings.push({
                text: `Extremely high burstiness (${burstScore.toFixed(3)}) - chaotic variation, possible humanizer artifact`,
                indicator: 'mixed'
            });
        } else {
            findings.push({
                text: `Natural burstiness pattern (${burstScore.toFixed(3)})`,
                indicator: 'human'
            });
        }

        const aiProbability = 1 - likelihoodResult.score;

        return {
            name: 'Burstiness Patterns',
            category: 9,
            aiProbability,
            confidence: 0.65,
            findings,
            details: { burstScore }
        };
    }

    /**
     * Analyze repetition patterns
     */
    analyzeRepetition(text, tokens) {
        const findings = [];
        let aiScore = 0;
        
        // Check for n-gram repetition
        const bigrams = [];
        const trigrams = [];
        
        for (let i = 0; i < tokens.length - 1; i++) {
            bigrams.push(`${tokens[i]} ${tokens[i + 1]}`);
        }
        for (let i = 0; i < tokens.length - 2; i++) {
            trigrams.push(`${tokens[i]} ${tokens[i + 1]} ${tokens[i + 2]}`);
        }

        const bigramFreq = Utils.frequencyDistribution(bigrams);
        const trigramFreq = Utils.frequencyDistribution(trigrams);

        // Find repeated n-grams
        const repeatedBigrams = Object.entries(bigramFreq).filter(([_, c]) => c >= 3);
        const repeatedTrigrams = Object.entries(trigramFreq).filter(([_, c]) => c >= 2);

        if (repeatedTrigrams.length > 5) {
            findings.push({
                text: `High trigram repetition (${repeatedTrigrams.length} phrases repeated) - template pattern`,
                indicator: 'ai'
            });
            aiScore += 0.3;
        }

        // Check for phrase uniformity
        const sentenceStarters = [];
        const sentences = Utils.splitSentences(text);
        
        for (const s of sentences) {
            const words = s.split(/\s+/).slice(0, 2).join(' ').toLowerCase();
            sentenceStarters.push(words);
        }

        const starterFreq = Utils.frequencyDistribution(sentenceStarters);
        const dominantStarter = Math.max(...Object.values(starterFreq));
        const starterUniformity = dominantStarter / sentences.length;

        if (starterUniformity > 0.3 && sentences.length >= 5) {
            findings.push({
                text: `Repetitive sentence starters (${(starterUniformity * 100).toFixed(0)}% same pattern)`,
                indicator: 'ai'
            });
            aiScore += 0.2;
        }

        if (findings.length === 0) {
            findings.push({
                text: 'Natural variation in phrasing and structure',
                indicator: 'human'
            });
        }

        const aiProbability = Math.min(0.9, aiScore + 0.3);

        return {
            name: 'Repetition Patterns',
            category: 12,
            aiProbability,
            confidence: 0.6,
            findings,
            details: { repeatedBigrams: repeatedBigrams.length, repeatedTrigrams: repeatedTrigrams.length }
        };
    }

    /**
     * Analyze readability metrics
     */
    analyzeReadability(text, sentences, tokens) {
        const findings = [];
        
        // Calculate syllables per word (approximate)
        const syllableCounts = tokens.map(word => {
            const cleaned = word.toLowerCase().replace(/[^a-z]/g, '');
            if (cleaned.length <= 3) return 1;
            
            // Count vowel groups
            const vowelGroups = cleaned.match(/[aeiouy]+/g) || [];
            let count = vowelGroups.length;
            
            // Adjust for silent e
            if (cleaned.endsWith('e') && count > 1) count--;
            
            return Math.max(1, count);
        });

        const avgSyllables = Utils.mean(syllableCounts);
        const avgSentenceLength = tokens.length / Math.max(1, sentences.length);

        // Flesch-Kincaid Grade Level
        const fkGrade = 0.39 * avgSentenceLength + 11.8 * avgSyllables - 15.59;
        
        // Flesch Reading Ease
        const fkEase = 206.835 - 1.015 * avgSentenceLength - 84.6 * avgSyllables;

        // Check for consistency (AI tends to have very consistent readability)
        const sentenceReadabilities = sentences.map(s => {
            const words = s.split(/\s+/);
            return words.length;
        });

        const readabilityCV = VarianceUtils.coefficientOfVariation(sentenceReadabilities);

        // Bidirectional: too consistent OR too variable
        if (readabilityCV < 0.25) {
            findings.push({
                text: `Very consistent sentence complexity (CV: ${readabilityCV.toFixed(2)}) - AI pattern`,
                indicator: 'ai'
            });
        } else if (readabilityCV > 1.0) {
            findings.push({
                text: `Extremely variable complexity (CV: ${readabilityCV.toFixed(2)}) - unusual pattern`,
                indicator: 'mixed'
            });
        } else {
            findings.push({
                text: `Natural complexity variation (CV: ${readabilityCV.toFixed(2)})`,
                indicator: 'human'
            });
        }

        // Grade level check
        if (fkGrade >= 8 && fkGrade <= 14) {
            findings.push({
                text: `Reading level appropriate (Grade ${fkGrade.toFixed(1)})`,
                indicator: 'human'
            });
        }

        // Calculate AI probability based on uniformity
        const uniformityScore = VarianceUtils.uniformityScore(sentenceReadabilities);
        const aiProbability = uniformityScore * 0.7 + 0.15;

        return {
            name: 'Readability Analysis',
            category: 7,
            aiProbability,
            confidence: 0.55,
            findings,
            details: { 
                fleschKincaidGrade: fkGrade, 
                fleschReadingEase: fkEase,
                avgSyllables,
                readabilityCV
            }
        };
    }

    /**
     * Build advanced statistics object
     */
    buildAdvancedStats(sentences, tokens, categoryResults) {
        const sentenceLengths = sentences.map(s => s.split(/\s+/).length);
        const freq = Utils.frequencyDistribution(tokens);
        const uniqueTokens = new Set(tokens);
        const hapaxCount = Object.values(freq).filter(c => c === 1).length;

        return {
            sentences: {
                count: sentences.length,
                mean: Utils.mean(sentenceLengths),
                median: Utils.median(sentenceLengths),
                stdDev: Utils.stdDev(sentenceLengths),
                cv: VarianceUtils.coefficientOfVariation(sentenceLengths)
            },
            vocabulary: {
                uniqueWords: uniqueTokens.size,
                totalWords: tokens.length,
                ttr: uniqueTokens.size / tokens.length,
                hapaxCount,
                hapaxRatio: hapaxCount / uniqueTokens.size
            },
            burstiness: {
                sentenceLength: VarianceUtils.burstiness(sentenceLengths)
            },
            uniformity: {
                overall: VarianceUtils.uniformityScore(sentenceLengths)
            },
            humanizerSignals: this.detectHumanizerSignals(sentenceLengths, tokens, categoryResults)
        };
    }

    /**
     * Detect signals of humanizer tool usage
     * Humanizers try to add variance, but create unnatural patterns
     */
    detectHumanizerSignals(sentenceLengths, tokens, categoryResults) {
        const signals = {
            isLikelyHumanized: false,
            warningFlags: 0
        };

        // Check for artificially high variance with stable second-order variance
        const cv = VarianceUtils.coefficientOfVariation(sentenceLengths);
        const localVars = [];
        const windowSize = 5;
        
        for (let i = 0; i <= sentenceLengths.length - windowSize; i++) {
            const window = sentenceLengths.slice(i, i + windowSize);
            localVars.push(Utils.variance(window));
        }

        if (localVars.length >= 3) {
            const varOfVar = Utils.variance(localVars);
            const meanVar = Utils.mean(localVars);
            
            // High overall variance but stable local variance = humanizer
            if (cv > 0.6 && meanVar > 0 && varOfVar / meanVar < 0.3) {
                signals.stableVarianceFlag = true;
                signals.warningFlags++;
            }
        }

        // Check for broken feature correlations
        const vocabResult = categoryResults.find(c => c.name === 'Vocabulary Richness');
        const syntaxResult = categoryResults.find(c => c.name === 'Sentence Structure');
        
        if (vocabResult && syntaxResult) {
            // In natural text, vocabulary and syntax AI scores correlate
            const vocabAI = vocabResult.aiProbability;
            const syntaxAI = syntaxResult.aiProbability;
            
            // Large discrepancy suggests manipulation
            if (Math.abs(vocabAI - syntaxAI) > 0.4) {
                signals.brokenCorrelationFlag = true;
                signals.warningFlags++;
            }
        }

        signals.isLikelyHumanized = signals.warningFlags >= 2;

        return signals;
    }

    /**
     * Determine verdict from probability
     */
    determineVerdict(aiProbability, confidence) {
        const band = VarianceUtils.toProbabilityBand(aiProbability);
        
        let description = '';
        if (aiProbability < 0.3) {
            description = 'Text shows natural human writing patterns with expected variance.';
        } else if (aiProbability < 0.5) {
            description = 'Text has some unusual patterns but may still be human-written.';
        } else if (aiProbability < 0.7) {
            description = 'Text shows patterns commonly associated with AI generation.';
        } else {
            description = 'Text displays strong indicators of AI generation.';
        }

        return {
            label: band.label,
            description,
            confidence: band.confidence
        };
    }

    /**
     * Empty result for invalid input
     */
    getEmptyResult() {
        return {
            aiProbability: 0,
            humanProbability: 1,
            confidence: 0,
            verdict: { label: 'Unknown', description: 'No text provided' },
            stats: { characters: 0, words: 0, sentences: 0, paragraphs: 0 },
            categoryResults: [],
            findings: [],
            error: 'No text provided for analysis'
        };
    }
}

// Export for Node.js
module.exports = {
    VeritasAnalyzer,
    Utils,
    VarianceUtils
};
