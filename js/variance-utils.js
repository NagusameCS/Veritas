/**
 * VERITAS — Variance Utilities
 * Core statistical functions for variance-based detection
 * 
 * KEY PRINCIPLE: We measure DEVIATIONS FROM EXPECTED HUMAN VARIANCE,
 * not presence/absence of features. AI text is characterized by
 * unusual uniformity and predictability, not specific content.
 */

const VarianceUtils = {
    /**
     * Calculate z-score for a value relative to expected distribution
     */
    zScore(value, mean, stdDev) {
        if (stdDev === 0) return 0;
        return (value - mean) / stdDev;
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
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VarianceUtils;
}
