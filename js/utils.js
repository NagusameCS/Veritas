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
