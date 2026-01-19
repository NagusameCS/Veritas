/**
 * VERITAS — Variance Utilities
 * Core statistical functions for variance-based detection
 * 
 * KEY PRINCIPLE: We measure DEVIATIONS FROM EXPECTED HUMAN VARIANCE,
 * not presence/absence of features. AI text is characterized by
 * unusual uniformity and predictability, not specific content.
 * 
 * ENHANCED v2.1: Added Bayesian combination, autocorrelation, 
 * n-gram perplexity approximation, and calibrated confidence scoring
 */

const VarianceUtils = {
    // ═══════════════════════════════════════════════════════════════════════════
    // Calibration Constants (empirically derived thresholds)
    // ═══════════════════════════════════════════════════════════════════════════
    
    // Expected human variance baselines (from literature)
    // KEY PRINCIPLE: Humans fall in a "reasonable middle" - neither too perfect nor too chaotic
    // These baselines define the center of a Gaussian; deviations in EITHER direction are suspicious
    HUMAN_BASELINES: {
        sentenceLengthCV: { mean: 0.55, stdDev: 0.15, min: 0.25, max: 0.85 },  // Too low = AI uniform; too high = chaotic
        hapaxRatio: { mean: 0.48, stdDev: 0.08, min: 0.30, max: 0.65 },        // Unique word ratio
        burstiness: { mean: 0.15, stdDev: 0.12, min: -0.10, max: 0.40 },       // Pattern of word usage
        zipfSlope: { mean: -1.0, stdDev: 0.15, min: -1.3, max: -0.7 },         // Natural language follows Zipf's law
        ttrNormalized: { mean: 0.42, stdDev: 0.10, min: 0.25, max: 0.60 },     // Vocabulary richness
        sentenceEntropy: { mean: 2.8, stdDev: 0.4, min: 1.8, max: 3.8 }        // Information content
    },
    
    // Minimum sample sizes for reliable statistics
    MIN_SAMPLES: {
        sentences: 5,
        words: 50,
        paragraphs: 2,
        ngrams: 20
    },
    /**
     * Calculate z-score for a value relative to expected distribution
     */
    zScore(value, mean, stdDev) {
        if (stdDev === 0) return 0;
        return (value - mean) / stdDev;
    },

    /**
     * GAUSSIAN DEVIATION SCORING
     * Core philosophy: There's a "normal curve" of reasonable values.
     * Humans fall somewhere in the reasonable middle.
     * Both extremes (too perfect AND too chaotic) are suspicious.
     * 
     * Returns 0-1 where:
     *   1 = value is at expected human mean (most natural)
     *   0 = value is far from expected (suspicious - either too perfect or too chaotic)
     */
    gaussianScore(value, mean, stdDev) {
        if (stdDev === 0) return value === mean ? 1 : 0;
        const z = Math.abs((value - mean) / stdDev);
        // Gaussian: e^(-z²/2) gives 1 at mean, decays to 0 at extremes
        return Math.exp(-z * z / 2);
    },

    /**
     * Score how "reasonably human" a value is based on expected range.
     * Uses soft boundaries - values outside range get low scores but not zero.
     * 
     * @param value - The observed value
     * @param baseline - Object with {mean, stdDev, min, max} from HUMAN_BASELINES
     * @returns 0-1 where 1 = typical human, 0 = highly unusual
     */
    humanLikelihoodScore(value, baseline) {
        if (!baseline) return 0.5;
        
        const { mean, stdDev, min, max } = baseline;
        
        // Gaussian component: how close to expected mean?
        const gaussianComponent = this.gaussianScore(value, mean, stdDev);
        
        // Range component: penalty for being outside expected bounds
        let rangePenalty = 0;
        if (min !== undefined && max !== undefined) {
            if (value < min) {
                // Too low (often too uniform/perfect = AI)
                rangePenalty = Math.min(1, (min - value) / (stdDev * 2));
            } else if (value > max) {
                // Too high (too chaotic, but could also be weird AI)
                rangePenalty = Math.min(1, (value - max) / (stdDev * 2));
            }
        }
        
        // Combined score: Gaussian weighted by range compliance
        return gaussianComponent * (1 - rangePenalty * 0.5);
    },

    /**
     * Compute AI probability from deviation from human baseline.
     * Values near human mean = low AI probability.
     * Values far from human mean (in either direction) = higher AI probability.
     */
    deviationToAIProbability(value, baseline) {
        const humanScore = this.humanLikelihoodScore(value, baseline);
        // Invert: high human score = low AI probability
        return 1 - humanScore;
    },

    /**
     * Calculate coefficient of variation (CV)
     * Lower CV = more uniform = more AI-like
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
        return (stdDev - mean) / (stdDev + mean);
    },

    /**
     * Calculate uniformity score (0 = varied, 1 = perfectly uniform)
     * OLD simple version - kept for backwards compatibility
     */
    uniformityScore(values) {
        if (!values || values.length < 2) return 0.5;
        const cv = this.coefficientOfVariation(values);
        // Typical human CV for sentence length is 0.4-0.8
        // AI tends toward 0.2-0.4
        // Normalize to 0-1 where higher = more uniform
        const normalized = 1 - Math.min(1, cv / 0.8);
        return normalized;
    },

    /**
     * Calculate how "natural" the variance is (bell curve approach).
     * Both too uniform (AI-like) AND too chaotic are flagged as unusual.
     * 
     * Returns:
     *   1 = variance is exactly what we'd expect from human writing
     *   0 = variance is suspiciously extreme (either direction)
     *   
     * This is the core of the "normal curve" philosophy.
     */
    varianceNaturalnessScore(values) {
        if (!values || values.length < 2) return 0.5;
        
        const cv = this.coefficientOfVariation(values);
        
        // Use Gaussian centered on expected human CV
        // Expected CV ~0.55, stdDev ~0.15
        return this.gaussianScore(cv, 0.55, 0.15);
    },

    /**
     * Combined AI indicator that flags BOTH extremes:
     * - Too perfect (low CV) = likely AI generated
     * - Too chaotic (high CV) = possibly AI with poor quality or unusual human
     * 
     * Returns 0-1 where higher = more likely AI
     */
    extremeVarianceScore(values) {
        if (!values || values.length < 2) return 0.5;
        
        const cv = this.coefficientOfVariation(values);
        const humanBaseline = this.HUMAN_BASELINES.sentenceLengthCV;
        
        // Distance from expected human mean in either direction
        const zScore = Math.abs((cv - humanBaseline.mean) / humanBaseline.stdDev);
        
        // Convert to probability: further from mean = higher AI probability
        // Using sigmoid to cap at reasonable bounds
        return 1 / (1 + Math.exp(-zScore + 1.5));
    },

    /**
     * Calculate local variance (variance within sliding windows)
     */
    localVariance(values, windowSize = 5) {
        if (!values || values.length < windowSize) return [];
        const localVars = [];
        for (let i = 0; i <= values.length - windowSize; i++) {
            const window = values.slice(i, i + windowSize);
            localVars.push(Utils.variance(window));
        }
        return localVars;
    },

    /**
     * Measure variance stability (how consistent variance is across document)
     * AI maintains consistent variance; humans have variable variance
     */
    varianceStability(values, windowSize = 5) {
        const localVars = this.localVariance(values, windowSize);
        if (localVars.length < 2) return 0.5;
        return this.uniformityScore(localVars);
    },

    /**
     * Calculate distribution divergence from expected human distribution
     * Uses simplified KL divergence approximation
     */
    distributionDivergence(observed, expected) {
        if (!observed || !expected || observed.length !== expected.length) return 0;
        
        let divergence = 0;
        for (let i = 0; i < observed.length; i++) {
            const p = Math.max(0.0001, observed[i]);
            const q = Math.max(0.0001, expected[i]);
            divergence += p * Math.log(p / q);
        }
        return divergence;
    },

    /**
     * Calculate inter-quartile range ratio
     * Humans typically have wider IQR relative to range
     */
    iqrRatio(values) {
        if (!values || values.length < 4) return 0.5;
        
        const sorted = [...values].sort((a, b) => a - b);
        const q1 = sorted[Math.floor(sorted.length * 0.25)];
        const q3 = sorted[Math.floor(sorted.length * 0.75)];
        const iqr = q3 - q1;
        const range = sorted[sorted.length - 1] - sorted[0];
        
        if (range === 0) return 0.5;
        return iqr / range;
    },

    /**
     * Detect periodicity in values (AI often shows regular patterns)
     * Returns periodicity strength (0 = random, 1 = perfectly periodic)
     */
    detectPeriodicity(values, maxPeriod = 10) {
        if (!values || values.length < maxPeriod * 2) return 0;
        
        let maxCorrelation = 0;
        
        for (let period = 2; period <= maxPeriod; period++) {
            let correlation = 0;
            let count = 0;
            
            for (let i = 0; i < values.length - period; i++) {
                const diff = Math.abs(values[i] - values[i + period]);
                const maxVal = Math.max(Math.abs(values[i]), Math.abs(values[i + period]), 1);
                correlation += 1 - (diff / maxVal);
                count++;
            }
            
            if (count > 0) {
                correlation /= count;
                maxCorrelation = Math.max(maxCorrelation, correlation);
            }
        }
        
        return maxCorrelation;
    },

    /**
     * Calculate drift score (how much a metric changes over document)
     * Humans tend to drift; AI maintains consistency
     */
    driftScore(values, windowSize = 10) {
        if (!values || values.length < windowSize * 2) return 0;
        
        const firstWindow = values.slice(0, windowSize);
        const lastWindow = values.slice(-windowSize);
        
        const firstMean = Utils.mean(firstWindow);
        const lastMean = Utils.mean(lastWindow);
        const overallMean = Utils.mean(values);
        
        if (overallMean === 0) return 0;
        
        const drift = Math.abs(lastMean - firstMean) / overallMean;
        return Math.min(1, drift);
    },

    /**
     * Calculate clustering score for repetitions
     * AI repeats at regular intervals; humans cluster repetitions
     */
    repetitionClustering(positions, totalLength) {
        if (!positions || positions.length < 2 || totalLength === 0) return 0.5;
        
        // Calculate distances between consecutive occurrences
        const distances = [];
        for (let i = 1; i < positions.length; i++) {
            distances.push(positions[i] - positions[i - 1]);
        }
        
        if (distances.length === 0) return 0.5;
        
        // Uniformity of distances (AI = uniform, human = clustered/varied)
        return 1 - this.uniformityScore(distances);
    },

    /**
     * Aggregate scores with category weights
     */
    weightedAggregate(scores, weights) {
        let totalWeight = 0;
        let weightedSum = 0;
        
        for (const key in scores) {
            if (weights[key] !== undefined && scores[key] !== undefined) {
                weightedSum += scores[key] * weights[key];
                totalWeight += weights[key];
            }
        }
        
        return totalWeight > 0 ? weightedSum / totalWeight : 0.5;
    },

    /**
     * Convert raw score to probability band
     */
    toProbabilityBand(score) {
        if (score < 0.15) return { band: 'very-low', label: 'Very Low', range: '0-15%' };
        if (score < 0.30) return { band: 'low', label: 'Low', range: '15-30%' };
        if (score < 0.45) return { band: 'moderate-low', label: 'Moderate-Low', range: '30-45%' };
        if (score < 0.55) return { band: 'uncertain', label: 'Uncertain', range: '45-55%' };
        if (score < 0.70) return { band: 'moderate-high', label: 'Moderate-High', range: '55-70%' };
        if (score < 0.85) return { band: 'high', label: 'High', range: '70-85%' };
        return { band: 'very-high', label: 'Very High', range: '85-100%' };
    },

    /**
     * Calculate confidence interval
     */
    confidenceInterval(probability, sampleSize, confidence = 0.95) {
        // Wilson score interval for proportions
        const z = confidence === 0.95 ? 1.96 : (confidence === 0.99 ? 2.576 : 1.645);
        const n = Math.max(1, sampleSize);
        const p = probability;
        
        const denominator = 1 + z * z / n;
        const center = (p + z * z / (2 * n)) / denominator;
        const margin = (z / denominator) * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n));
        
        return {
            lower: Math.max(0, center - margin),
            upper: Math.min(1, center + margin),
            margin: margin
        };
    },

    /**
     * Estimate false positive risk based on text characteristics
     */
    falsePositiveRisk(textStats, probability) {
        let risk = 0;
        const factors = [];

        // Short texts have higher FP risk
        if (textStats.words < 50) {
            risk += 0.3;
            factors.push('Short text length increases uncertainty');
        } else if (textStats.words < 100) {
            risk += 0.15;
            factors.push('Moderate text length adds some uncertainty');
        }

        // Technical/academic writing often triggers false positives
        if (textStats.avgWordsPerSentence > 25) {
            risk += 0.1;
            factors.push('Long sentences typical of academic writing');
        }

        // Very high probability often indicates genuine AI
        if (probability > 0.85) {
            risk -= 0.1;
            factors.push('Strong signal reduces false positive likelihood');
        }

        // Moderate probability has highest FP risk
        if (probability > 0.45 && probability < 0.65) {
            risk += 0.2;
            factors.push('Borderline scores have inherent uncertainty');
        }

        return {
            level: risk > 0.3 ? 'high' : (risk > 0.15 ? 'moderate' : 'low'),
            score: Math.min(1, Math.max(0, risk)),
            factors
        };
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // ENHANCED STATISTICAL METHODS v2.1
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Bayesian probability combination using log-odds
     * More accurate than simple weighted average for combining independent signals
     * 
     * Formula: Combined odds = product of individual odds
     * P(AI) = 1 / (1 + exp(-Σ log(p/(1-p))))
     */
    bayesianCombine(probabilities, weights = null) {
        if (!probabilities || probabilities.length === 0) return 0.5;
        
        // Filter valid probabilities (avoid 0 and 1 which cause log issues)
        const validProbs = probabilities.map(p => Math.max(0.001, Math.min(0.999, p)));
        const w = weights || validProbs.map(() => 1);
        
        // Convert to log-odds and sum
        let logOddsSum = 0;
        let totalWeight = 0;
        
        for (let i = 0; i < validProbs.length; i++) {
            const p = validProbs[i];
            const weight = w[i] || 1;
            const logOdds = Math.log(p / (1 - p));
            logOddsSum += logOdds * weight;
            totalWeight += weight;
        }
        
        // Normalize by weight and convert back to probability
        const avgLogOdds = logOddsSum / totalWeight;
        return 1 / (1 + Math.exp(-avgLogOdds));
    },

    /**
     * Calibrated probability using Platt scaling
     * Maps raw scores to calibrated probabilities using sigmoid
     */
    plattScale(rawScore, A = 1.0, B = 0.0) {
        // P(AI|score) = 1 / (1 + exp(A*score + B))
        // Default A=1, B=0 means no transformation
        return 1 / (1 + Math.exp(-(A * rawScore + B)));
    },

    /**
     * Isotonic regression approximation for probability calibration
     * Ensures monotonically increasing calibrated probabilities
     */
    isotonicCalibrate(rawScore, calibrationPoints = null) {
        // Default calibration curve based on empirical testing
        const defaultPoints = [
            { raw: 0.0, calibrated: 0.05 },
            { raw: 0.2, calibrated: 0.15 },
            { raw: 0.4, calibrated: 0.35 },
            { raw: 0.5, calibrated: 0.50 },
            { raw: 0.6, calibrated: 0.65 },
            { raw: 0.8, calibrated: 0.85 },
            { raw: 1.0, calibrated: 0.95 }
        ];
        
        const points = calibrationPoints || defaultPoints;
        
        // Find surrounding points and interpolate
        for (let i = 0; i < points.length - 1; i++) {
            if (rawScore >= points[i].raw && rawScore <= points[i + 1].raw) {
                const t = (rawScore - points[i].raw) / (points[i + 1].raw - points[i].raw);
                return points[i].calibrated + t * (points[i + 1].calibrated - points[i].calibrated);
            }
        }
        
        return rawScore; // Fallback
    },

    /**
     * Autocorrelation analysis - detects periodic patterns in sequences
     * AI text often shows higher autocorrelation (more predictable patterns)
     * 
     * @param values - Array of numeric values (e.g., sentence lengths)
     * @param maxLag - Maximum lag to compute (default 10)
     * @returns Object with autocorrelation coefficients and periodicity score
     */
    autocorrelation(values, maxLag = 10) {
        if (!values || values.length < maxLag + 5) {
            return { coefficients: [], avgAC: 0, maxAC: 0, periodicityScore: 0.5 };
        }
        
        const n = values.length;
        const mean = Utils.mean(values);
        const variance = Utils.variance(values);
        
        if (variance === 0) {
            return { coefficients: [], avgAC: 1, maxAC: 1, periodicityScore: 1 };
        }
        
        const coefficients = [];
        
        for (let lag = 1; lag <= maxLag; lag++) {
            let sum = 0;
            for (let i = 0; i < n - lag; i++) {
                sum += (values[i] - mean) * (values[i + lag] - mean);
            }
            const ac = sum / ((n - lag) * variance);
            coefficients.push({ lag, value: ac });
        }
        
        // Calculate summary statistics
        const acValues = coefficients.map(c => Math.abs(c.value));
        const avgAC = Utils.mean(acValues);
        const maxAC = Math.max(...acValues);
        
        // Periodicity score: higher autocorrelation = more periodic = more AI-like
        // Human text typically has low autocorrelation (random-like)
        const periodicityScore = Utils.normalize(avgAC, 0, 0.4);
        
        return { coefficients, avgAC, maxAC, periodicityScore };
    },

    /**
     * N-gram perplexity approximation using Markov chain entropy
     * Lower perplexity = more predictable = more AI-like
     * 
     * This approximates true neural LM perplexity using statistical methods
     */
    ngramPerplexity(tokens, n = 2) {
        if (!tokens || tokens.length < n + 10) {
            return { perplexity: 50, entropy: 4, predictability: 0.5 };
        }
        
        // Build n-gram frequency table
        const ngramCounts = {};
        const contextCounts = {};
        
        for (let i = 0; i <= tokens.length - n; i++) {
            const context = tokens.slice(i, i + n - 1).join(' ');
            const ngram = tokens.slice(i, i + n).join(' ');
            
            contextCounts[context] = (contextCounts[context] || 0) + 1;
            ngramCounts[ngram] = (ngramCounts[ngram] || 0) + 1;
        }
        
        // Calculate cross-entropy: H = -1/N * Σ log P(w_i | context)
        let logProbSum = 0;
        let count = 0;
        
        for (let i = 0; i <= tokens.length - n; i++) {
            const context = tokens.slice(i, i + n - 1).join(' ');
            const ngram = tokens.slice(i, i + n).join(' ');
            
            const contextCount = contextCounts[context] || 1;
            const ngramCount = ngramCounts[ngram] || 1;
            
            // Add-one smoothing
            const prob = (ngramCount + 1) / (contextCount + Object.keys(ngramCounts).length);
            logProbSum += Math.log2(prob);
            count++;
        }
        
        const crossEntropy = count > 0 ? -logProbSum / count : 10;
        const perplexity = Math.pow(2, crossEntropy);
        
        // Normalize perplexity to predictability score (0-1)
        // Lower perplexity = higher predictability = more AI-like
        // Typical human text: perplexity 100-300
        // AI text: perplexity 30-80
        const predictability = 1 - Utils.normalize(perplexity, 20, 200);
        
        return { 
            perplexity: Math.min(500, perplexity), 
            entropy: crossEntropy, 
            predictability 
        };
    },

    /**
     * Compute Mahalanobis distance from expected human distribution
     * Measures how "unusual" a text is compared to typical human writing
     */
    mahalanobisDistance(features, baseline = null) {
        const defaultBaseline = this.HUMAN_BASELINES;
        const base = baseline || defaultBaseline;
        
        let distanceSquared = 0;
        let count = 0;
        
        for (const [key, value] of Object.entries(features)) {
            if (base[key] && typeof value === 'number') {
                const { mean, stdDev } = base[key];
                if (stdDev > 0) {
                    const z = (value - mean) / stdDev;
                    distanceSquared += z * z;
                    count++;
                }
            }
        }
        
        return count > 0 ? Math.sqrt(distanceSquared / count) : 0;
    },

    /**
     * Calculate text length normalization factor
     * Short texts need different thresholds than long texts
     */
    lengthNormalization(wordCount) {
        // Logarithmic normalization: confidence increases with length but plateaus
        if (wordCount < 20) return 0.3;
        if (wordCount < 50) return 0.5;
        if (wordCount < 100) return 0.7;
        if (wordCount < 200) return 0.85;
        if (wordCount < 500) return 0.95;
        return 1.0;
    },

    /**
     * Compute Shannon entropy of a sequence with length normalization
     */
    normalizedEntropy(values) {
        if (!values || values.length < 2) return 0;
        
        // Build frequency distribution
        const freq = {};
        values.forEach(v => { freq[v] = (freq[v] || 0) + 1; });
        
        const n = values.length;
        let entropy = 0;
        
        for (const count of Object.values(freq)) {
            const p = count / n;
            if (p > 0) entropy -= p * Math.log2(p);
        }
        
        // Normalize by maximum possible entropy
        const maxEntropy = Math.log2(Object.keys(freq).length);
        return maxEntropy > 0 ? entropy / maxEntropy : 0;
    },

    /**
     * Jensen-Shannon divergence between two distributions
     * Symmetric measure of distribution difference
     */
    jensenShannonDivergence(p, q) {
        if (!p || !q || p.length !== q.length || p.length === 0) return 0;
        
        // Normalize distributions
        const sumP = p.reduce((a, b) => a + b, 0);
        const sumQ = q.reduce((a, b) => a + b, 0);
        const pNorm = p.map(x => x / sumP);
        const qNorm = q.map(x => x / sumQ);
        
        // Calculate midpoint distribution
        const m = pNorm.map((pi, i) => (pi + qNorm[i]) / 2);
        
        // Calculate KL divergences
        let klPM = 0, klQM = 0;
        for (let i = 0; i < p.length; i++) {
            if (pNorm[i] > 0 && m[i] > 0) {
                klPM += pNorm[i] * Math.log2(pNorm[i] / m[i]);
            }
            if (qNorm[i] > 0 && m[i] > 0) {
                klQM += qNorm[i] * Math.log2(qNorm[i] / m[i]);
            }
        }
        
        return (klPM + klQM) / 2;
    },

    /**
     * Semantic coherence score using sentence similarity decay
     * Measures how coherently ideas flow through text
     */
    coherenceScore(sentenceVectors) {
        if (!sentenceVectors || sentenceVectors.length < 3) return 0.5;
        
        const similarities = [];
        for (let i = 1; i < sentenceVectors.length; i++) {
            const sim = this.cosineSimilarity(sentenceVectors[i-1], sentenceVectors[i]);
            similarities.push(sim);
        }
        
        // Calculate statistics on adjacent sentence similarities
        const meanSim = Utils.mean(similarities);
        const varSim = Utils.variance(similarities);
        
        // AI text often has MORE consistent coherence (less variance)
        // Human text has natural fluctuations in coherence
        const coherenceUniformity = 1 - Utils.normalize(varSim, 0, 0.1);
        
        return {
            meanCoherence: meanSim,
            coherenceVariance: varSim,
            uniformityScore: coherenceUniformity
        };
    },

    /**
     * Cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        if (!a || !b || a.length !== b.length) return 0;
        
        let dotProduct = 0, normA = 0, normB = 0;
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        const denominator = Math.sqrt(normA) * Math.sqrt(normB);
        return denominator > 0 ? dotProduct / denominator : 0;
    },

    /**
     * Create simple sentence embedding using word frequency vectors
     * (Lightweight alternative to neural embeddings)
     */
    sentenceToVector(sentence, vocabulary) {
        const tokens = Utils.tokenize(sentence);
        const vector = new Array(vocabulary.length).fill(0);
        
        tokens.forEach(token => {
            const idx = vocabulary.indexOf(token);
            if (idx !== -1) vector[idx]++;
        });
        
        // Normalize
        const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
        return norm > 0 ? vector.map(v => v / norm) : vector;
    },

    /**
     * Chi-squared test for uniformity
     * Tests if a distribution is significantly different from uniform
     */
    chiSquaredUniformity(observed) {
        if (!observed || observed.length < 3) return { chiSquared: 0, pValue: 0.5 };
        
        const n = observed.length;
        const total = observed.reduce((a, b) => a + b, 0);
        const expected = total / n;
        
        let chiSquared = 0;
        for (const obs of observed) {
            chiSquared += Math.pow(obs - expected, 2) / expected;
        }
        
        // Approximate p-value using chi-squared distribution
        // (simplified - would need proper chi-squared CDF for exact p-value)
        const df = n - 1;
        const normalizedChi = chiSquared / df;
        
        // Higher chi-squared = less uniform = more human-like
        const uniformityScore = 1 / (1 + normalizedChi);
        
        return { chiSquared, degreesOfFreedom: df, uniformityScore };
    },

    /**
     * Runs test for randomness
     * Counts runs of consecutive values above/below median
     * Fewer runs than expected = non-random = potentially AI
     */
    runsTest(values) {
        if (!values || values.length < 10) return { runsRatio: 1, isRandom: true };
        
        const median = Utils.median(values);
        
        // Convert to binary (above/below median)
        const binary = values.map(v => v > median ? 1 : 0);
        
        // Count runs
        let runs = 1;
        for (let i = 1; i < binary.length; i++) {
            if (binary[i] !== binary[i-1]) runs++;
        }
        
        // Expected runs for random sequence
        const n = values.length;
        const n1 = binary.filter(b => b === 1).length;
        const n0 = n - n1;
        
        const expectedRuns = (2 * n0 * n1 / n) + 1;
        const stdRuns = Math.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n)) / (n * n * (n - 1)));
        
        const runsRatio = stdRuns > 0 ? (runs - expectedRuns) / stdRuns : 0;
        const isRandom = Math.abs(runsRatio) < 1.96; // 95% confidence
        
        return { 
            runs, 
            expectedRuns, 
            runsRatio, 
            isRandom,
            randomnessScore: 1 - Utils.normalize(Math.abs(runsRatio), 0, 3)
        };
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VarianceUtils;
}
