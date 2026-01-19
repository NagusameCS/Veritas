/**
 * VERITAS — Utility Functions
 * Core utilities for text processing and analysis
 */

const Utils = {
    // ═══════════════════════════════════════════════════════════════════════════
    // Text Tokenization & Parsing
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Split text into sentences
     */
    splitSentences(text) {
        if (!text || typeof text !== 'string') return [];
        
        // Handle common abbreviations to avoid false splits
        const abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sr.', 'Jr.', 'vs.', 'etc.', 'e.g.', 'i.e.', 'U.S.', 'U.K.'];
        let processed = text;
        const placeholders = [];
        
        abbreviations.forEach((abbr, i) => {
            const placeholder = `__ABBR${i}__`;
            placeholders.push({ placeholder, abbr });
            processed = processed.split(abbr).join(placeholder);
        });
        
        // Split on sentence-ending punctuation
        const sentences = processed.split(/(?<=[.!?])\s+(?=[A-Z])/);
        
        // Restore abbreviations
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
     * Get character count (excluding spaces)
     */
    charCount(text) {
        return text.replace(/\s/g, '').length;
    },

    /**
     * Get sentence count
     */
    sentenceCount(text) {
        return this.splitSentences(text).length;
    },

    /**
     * Get paragraph count
     */
    paragraphCount(text) {
        return text.split(/\n\s*\n/).filter(p => p.trim().length > 0).length;
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // Statistical Utilities
    // ═══════════════════════════════════════════════════════════════════════════

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
    standardDeviation(arr) {
        return Math.sqrt(this.variance(arr));
    },

    /**
     * Calculate coefficient of variation
     */
    coefficientOfVariation(arr) {
        const avg = this.mean(arr);
        if (avg === 0) return 0;
        return this.standardDeviation(arr) / avg;
    },

    /**
     * Calculate entropy of a frequency distribution
     */
    entropy(frequencies) {
        const total = Object.values(frequencies).reduce((a, b) => a + b, 0);
        if (total === 0) return 0;
        
        let entropy = 0;
        for (const freq of Object.values(frequencies)) {
            if (freq > 0) {
                const p = freq / total;
                entropy -= p * Math.log2(p);
            }
        }
        return entropy;
    },

    /**
     * Normalize a value to 0-1 range
     */
    normalize(value, min, max) {
        if (max === min) return 0.5;
        return Math.max(0, Math.min(1, (value - min) / (max - min)));
    },

    /**
     * Calculate percentile
     */
    percentile(arr, p) {
        if (!arr || arr.length === 0) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const index = (p / 100) * (sorted.length - 1);
        const lower = Math.floor(index);
        const upper = Math.ceil(index);
        if (lower === upper) return sorted[lower];
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
    },

    /**
     * Calculate median
     */
    median(arr) {
        return this.percentile(arr, 50);
    },

    /**
     * Calculate skewness (asymmetry of distribution)
     * Positive skew = right tail, Negative skew = left tail
     */
    skewness(arr) {
        if (!arr || arr.length < 3) return 0;
        const n = arr.length;
        const mean = this.mean(arr);
        const stdDev = this.standardDeviation(arr);
        if (stdDev === 0) return 0;
        
        const sum = arr.reduce((acc, val) => acc + Math.pow((val - mean) / stdDev, 3), 0);
        return (n / ((n - 1) * (n - 2))) * sum;
    },

    /**
     * Calculate kurtosis (tailedness of distribution)
     * High kurtosis = heavy tails, Low kurtosis = light tails
     */
    kurtosis(arr) {
        if (!arr || arr.length < 4) return 0;
        const n = arr.length;
        const mean = this.mean(arr);
        const stdDev = this.standardDeviation(arr);
        if (stdDev === 0) return 0;
        
        const sum = arr.reduce((acc, val) => acc + Math.pow((val - mean) / stdDev, 4), 0);
        const excess = ((n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))) * sum;
        const correction = (3 * Math.pow(n - 1, 2)) / ((n - 2) * (n - 3));
        return excess - correction;
    },

    /**
     * Calculate Gini coefficient (inequality measure)
     * 0 = perfect equality, 1 = perfect inequality
     */
    giniCoefficient(arr) {
        if (!arr || arr.length < 2) return 0;
        const sorted = [...arr].sort((a, b) => a - b);
        const n = sorted.length;
        const mean = this.mean(sorted);
        if (mean === 0) return 0;
        
        let sum = 0;
        for (let i = 0; i < n; i++) {
            sum += (2 * (i + 1) - n - 1) * sorted[i];
        }
        return sum / (n * n * mean);
    },

    /**
     * Calculate Hapax Legomena ratio (words appearing exactly once)
     * Higher ratio suggests more natural/human text
     */
    hapaxLegomenaRatio(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        const freq = this.frequencyDistribution(tokens);
        const hapaxCount = Object.values(freq).filter(f => f === 1).length;
        return hapaxCount / tokens.length;
    },

    /**
     * Calculate Dis Legomena ratio (words appearing exactly twice)
     */
    disLegomenaRatio(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        const freq = this.frequencyDistribution(tokens);
        const disCount = Object.values(freq).filter(f => f === 2).length;
        return disCount / tokens.length;
    },

    /**
     * Calculate Yule's K characteristic (vocabulary richness)
     * Lower K = richer vocabulary
     */
    yulesK(tokens) {
        if (!tokens || tokens.length < 10) return 0;
        const freq = this.frequencyDistribution(tokens);
        const freqOfFreq = {};
        
        Object.values(freq).forEach(f => {
            freqOfFreq[f] = (freqOfFreq[f] || 0) + 1;
        });
        
        const N = tokens.length;
        let M2 = 0;
        
        for (const [r, Vr] of Object.entries(freqOfFreq)) {
            M2 += Vr * Math.pow(parseInt(r), 2);
        }
        
        const M1 = N;
        if (M1 === 0) return 0;
        
        return 10000 * (M2 - M1) / (M1 * M1);
    },

    /**
     * Calculate Simpson's D (diversity index)
     * Lower = more diverse
     */
    simpsonsD(tokens) {
        if (!tokens || tokens.length < 2) return 0;
        const freq = this.frequencyDistribution(tokens);
        const N = tokens.length;
        
        let sum = 0;
        for (const count of Object.values(freq)) {
            sum += count * (count - 1);
        }
        
        return sum / (N * (N - 1));
    },

    /**
     * Calculate Honore's R statistic (vocabulary richness)
     * Higher R = richer vocabulary
     */
    honoresR(tokens) {
        if (!tokens || tokens.length < 10) return 0;
        const freq = this.frequencyDistribution(tokens);
        const V = Object.keys(freq).length; // vocabulary size
        const V1 = Object.values(freq).filter(f => f === 1).length; // hapax legomena
        const N = tokens.length;
        
        if (V1 === V) return 0; // Avoid division by zero
        return (100 * Math.log(N)) / (1 - V1 / V);
    },

    /**
     * Calculate Type-Token Ratio (TTR)
     */
    typeTokenRatio(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        return new Set(tokens).size / tokens.length;
    },

    /**
     * Calculate Root TTR (RTTR) - more stable than TTR
     */
    rootTTR(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        return new Set(tokens).size / Math.sqrt(tokens.length);
    },

    /**
     * Calculate Log TTR (LTTR) - Herdan's C
     */
    logTTR(tokens) {
        if (!tokens || tokens.length < 2) return 0;
        const V = new Set(tokens).size;
        return Math.log(V) / Math.log(tokens.length);
    },

    /**
     * Calculate MSTTR (Mean Segmental TTR) - average TTR over segments
     */
    msttr(tokens, segmentSize = 50) {
        if (!tokens || tokens.length < segmentSize) return this.typeTokenRatio(tokens);
        
        const segments = [];
        for (let i = 0; i <= tokens.length - segmentSize; i += segmentSize) {
            const segment = tokens.slice(i, i + segmentSize);
            segments.push(this.typeTokenRatio(segment));
        }
        
        return this.mean(segments);
    },

    /**
     * Analyze Zipf's Law compliance
     * Returns how well word frequencies follow Zipf's distribution
     */
    zipfAnalysis(tokens) {
        if (!tokens || tokens.length < 20) return { compliance: 0.5, slope: 0, rSquared: 0 };
        
        const freq = this.frequencyDistribution(tokens);
        const sorted = Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 50); // Top 50 words
        
        // Calculate log-log regression
        const points = sorted.map((entry, i) => ({
            x: Math.log(i + 1), // log(rank)
            y: Math.log(entry[1]) // log(frequency)
        }));
        
        const n = points.length;
        const sumX = points.reduce((s, p) => s + p.x, 0);
        const sumY = points.reduce((s, p) => s + p.y, 0);
        const sumXY = points.reduce((s, p) => s + p.x * p.y, 0);
        const sumX2 = points.reduce((s, p) => s + p.x * p.x, 0);
        const sumY2 = points.reduce((s, p) => s + p.y * p.y, 0);
        
        // Linear regression slope
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // R-squared (coefficient of determination)
        const yMean = sumY / n;
        const ssTotal = points.reduce((s, p) => s + Math.pow(p.y - yMean, 2), 0);
        const ssResidual = points.reduce((s, p) => {
            const predicted = slope * p.x + intercept;
            return s + Math.pow(p.y - predicted, 2);
        }, 0);
        
        const rSquared = ssTotal > 0 ? 1 - ssResidual / ssTotal : 0;
        
        // Zipf's law predicts slope ≈ -1
        // Deviation from -1 indicates AI text (often slope < -1)
        const zipfCompliance = 1 - Math.min(1, Math.abs(slope + 1));
        
        return {
            compliance: zipfCompliance,
            slope: slope,
            rSquared: rSquared,
            expectedSlope: -1,
            deviation: slope + 1
        };
    },

    /**
     * Calculate Brunet's W (vocabulary richness index)
     * Typically ranges 10-20 for natural text
     */
    brunetsW(tokens) {
        if (!tokens || tokens.length < 10) return 0;
        const V = new Set(tokens).size;
        const N = tokens.length;
        return Math.pow(N, Math.pow(V, -0.172));
    },

    /**
     * Calculate Sichel's S (proportion of dis legomena)
     */
    sichelsS(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        const freq = this.frequencyDistribution(tokens);
        const V = Object.keys(freq).length;
        const V2 = Object.values(freq).filter(f => f === 2).length;
        return V > 0 ? V2 / V : 0;
    },

    /**
     * Calculate average word length
     */
    avgWordLength(tokens) {
        if (!tokens || tokens.length === 0) return 0;
        return tokens.reduce((sum, t) => sum + t.length, 0) / tokens.length;
    },

    /**
     * Calculate word length distribution
     */
    wordLengthDistribution(tokens) {
        if (!tokens || tokens.length === 0) return {};
        const dist = {};
        tokens.forEach(t => {
            const len = t.length;
            dist[len] = (dist[len] || 0) + 1;
        });
        return dist;
    },

    /**
     * Calculate sentence length distribution statistics
     */
    sentenceLengthStats(sentences) {
        if (!sentences || sentences.length === 0) {
            return { mean: 0, median: 0, stdDev: 0, min: 0, max: 0, cv: 0 };
        }
        
        const lengths = sentences.map(s => this.tokenize(s).length);
        return {
            mean: this.mean(lengths),
            median: this.median(lengths),
            stdDev: this.standardDeviation(lengths),
            min: Math.min(...lengths),
            max: Math.max(...lengths),
            cv: this.coefficientOfVariation(lengths),
            skewness: this.skewness(lengths),
            kurtosis: this.kurtosis(lengths)
        };
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // N-gram Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Generate n-grams from tokens
     */
    ngrams(tokens, n) {
        if (!tokens || tokens.length < n) return [];
        const result = [];
        for (let i = 0; i <= tokens.length - n; i++) {
            result.push(tokens.slice(i, i + n).join(' '));
        }
        return result;
    },

    /**
     * Get frequency distribution
     */
    frequencyDistribution(items) {
        const freq = {};
        items.forEach(item => {
            freq[item] = (freq[item] || 0) + 1;
        });
        return freq;
    },

    /**
     * Get top N items by frequency
     */
    topN(frequencies, n) {
        return Object.entries(frequencies)
            .sort((a, b) => b[1] - a[1])
            .slice(0, n);
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // Text Pattern Matching
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Count occurrences of a pattern in text
     */
    countPattern(text, pattern) {
        const regex = typeof pattern === 'string' ? new RegExp(pattern, 'gi') : pattern;
        const matches = text.match(regex);
        return matches ? matches.length : 0;
    },

    /**
     * Find all matches with positions
     */
    findPatternPositions(text, pattern) {
        const regex = typeof pattern === 'string' ? new RegExp(pattern, 'gi') : new RegExp(pattern.source, 'gi');
        const positions = [];
        let match;
        while ((match = regex.exec(text)) !== null) {
            positions.push({
                match: match[0],
                start: match.index,
                end: match.index + match[0].length
            });
        }
        return positions;
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // Word Lists & Dictionaries
    // ═══════════════════════════════════════════════════════════════════════════

    // Common function words (stop words)
    functionWords: new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
        'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
        'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
        'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same',
        'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'then', 'if', 'because', 'until', 'while', 'although', 'though',
        'after', 'before', 'since', 'unless', 'however', 'therefore', 'thus'
    ]),

    // Hedging words common in AI text
    hedgingWords: [
        'perhaps', 'maybe', 'possibly', 'probably', 'likely', 'unlikely',
        'somewhat', 'relatively', 'fairly', 'rather', 'quite', 'slightly',
        'generally', 'typically', 'usually', 'often', 'sometimes', 'occasionally',
        'tend to', 'seems to', 'appears to', 'may be', 'might be', 'could be',
        'it is possible', 'it is likely', 'in some cases', 'to some extent'
    ],

    // Common AI discourse markers
    discourseMarkers: [
        'furthermore', 'moreover', 'additionally', 'in addition',
        'however', 'nevertheless', 'nonetheless', 'on the other hand',
        'consequently', 'therefore', 'thus', 'hence', 'as a result',
        'in conclusion', 'to summarize', 'overall', 'in summary',
        'firstly', 'secondly', 'thirdly', 'finally', 'lastly',
        'for example', 'for instance', 'specifically', 'in particular',
        'it is important to note', 'it should be noted', 'notably',
        'essentially', 'fundamentally', 'basically', 'ultimately'
    ],

    // ═══════════════════════════════════════════════════════════════════════════
    // Scoring Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Sigmoid function for smooth scoring
     */
    sigmoid(x, steepness = 1, midpoint = 0.5) {
        return 1 / (1 + Math.exp(-steepness * (x - midpoint)));
    },

    /**
     * Weighted average of scores
     */
    weightedAverage(scores, weights) {
        let totalWeight = 0;
        let weightedSum = 0;
        
        for (let i = 0; i < scores.length; i++) {
            const weight = weights[i] || 1;
            weightedSum += scores[i] * weight;
            totalWeight += weight;
        }
        
        return totalWeight > 0 ? weightedSum / totalWeight : 0;
    },

    /**
     * Combine multiple probability scores
     */
    combineProbabilities(probabilities, method = 'average') {
        if (!probabilities || probabilities.length === 0) return 0.5;
        
        switch (method) {
            case 'average':
                return this.mean(probabilities);
            case 'max':
                return Math.max(...probabilities);
            case 'min':
                return Math.min(...probabilities);
            case 'geometric':
                return Math.pow(probabilities.reduce((a, b) => a * b, 1), 1 / probabilities.length);
            default:
                return this.mean(probabilities);
        }
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // Formatting Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Format number as percentage
     */
    formatPercent(value, decimals = 1) {
        return (value * 100).toFixed(decimals) + '%';
    },

    /**
     * Format large number with abbreviations
     */
    formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
        if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
        return num.toString();
    },

    /**
     * Truncate text with ellipsis
     */
    truncate(text, maxLength) {
        if (text.length <= maxLength) return text;
        return text.slice(0, maxLength - 3) + '...';
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // DOM Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Debounce function
     */
    debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    /**
     * Throttle function
     */
    throttle(func, limit) {
        let inThrottle;
        return function(...args) {
            if (!inThrottle) {
                func.apply(this, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    },

    /**
     * Generate unique ID
     */
    generateId() {
        return 'id_' + Math.random().toString(36).substr(2, 9);
    },

    /**
     * Deep clone object
     */
    deepClone(obj) {
        return JSON.parse(JSON.stringify(obj));
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // Storage Utilities
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Save to localStorage
     */
    saveToStorage(key, data) {
        try {
            localStorage.setItem(key, JSON.stringify(data));
            return true;
        } catch (e) {
            console.error('Failed to save to localStorage:', e);
            return false;
        }
    },

    /**
     * Load from localStorage
     */
    loadFromStorage(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (e) {
            console.error('Failed to load from localStorage:', e);
            return defaultValue;
        }
    },

    /**
     * Remove from localStorage
     */
    removeFromStorage(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (e) {
            console.error('Failed to remove from localStorage:', e);
            return false;
        }
    },

    // ═══════════════════════════════════════════════════════════════════════════
    // History Management
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Load analysis history from localStorage
     */
    loadHistory() {
        return this.loadFromStorage('veritas_history', []);
    },

    /**
     * Save analysis history to localStorage
     */
    saveHistory(history) {
        return this.saveToStorage('veritas_history', history);
    },

    /**
     * Clear analysis history
     */
    clearHistory() {
        return this.removeFromStorage('veritas_history');
    }
};

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Utils;
}
