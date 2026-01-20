/**
 * FLARE Detection System - JavaScript Analyzer
 * =============================================
 * Specialized model for detecting humanized AI content vs genuine human writing.
 * 
 * Unlike other Veritas models that focus on AI vs Human classification,
 * Flare assumes the content appears human-like and specifically looks for
 * signs of AI humanization tools:
 * 
 * - Artificial variance injection
 * - Broken natural feature correlations  
 * - Synonym substitution patterns
 * - Mechanical contraction insertion
 * - Surface-level chaos with deep structural uniformity
 * 
 * This model runs AFTER initial classification and provides additional
 * warnings if humanization is detected.
 */

class FlareAnalyzer {
    constructor() {
        this.name = 'Flare';
        this.version = '1.0';
        this.type = 'humanization-detection';
        
        // Feature configuration - will be loaded from trained model
        this.config = null;
        this.isReady = false;
        
        // Synonym clusters for detection
        this.synonymClusters = {
            'good': ['excellent', 'great', 'wonderful', 'fantastic', 'superb', 'outstanding'],
            'bad': ['terrible', 'awful', 'horrible', 'dreadful', 'poor', 'disappointing'],
            'big': ['large', 'huge', 'enormous', 'massive', 'substantial', 'significant'],
            'small': ['tiny', 'little', 'minute', 'minor', 'slight', 'minimal'],
            'important': ['crucial', 'vital', 'essential', 'critical', 'significant', 'key'],
            'help': ['assist', 'aid', 'support', 'facilitate', 'enable', 'contribute'],
            'show': ['demonstrate', 'illustrate', 'reveal', 'indicate', 'display', 'exhibit'],
            'use': ['utilize', 'employ', 'leverage', 'apply', 'implement', 'harness'],
            'make': ['create', 'produce', 'generate', 'develop', 'construct', 'build'],
            'get': ['obtain', 'acquire', 'receive', 'gain', 'secure', 'attain'],
        };
        
        // AI hedging phrases
        this.hedgingPhrases = [
            'it is important to note', 'it should be noted', 'it is worth mentioning',
            'generally speaking', 'in many cases', 'it could be argued',
            'from this perspective', 'in this context', 'as we can see',
            'this suggests that', 'this indicates that', 'this demonstrates',
        ];
        
        // Transition markers
        this.transitionMarkers = [
            'furthermore', 'moreover', 'additionally', 'consequently', 'therefore',
            'however', 'nevertheless', 'nonetheless', 'subsequently', 'accordingly',
        ];
    }
    
    /**
     * Load trained model configuration
     */
    loadConfig(config) {
        this.config = config;
        this.isReady = true;
        console.log(`Flare model loaded: ${config.features?.length || 0} features`);
    }
    
    /**
     * Main analysis function - detect if text is humanized
     */
    analyze(text) {
        if (!text || text.length < 100) {
            return this.createEmptyResult('Text too short for humanization analysis');
        }
        
        const sentences = this.splitSentences(text);
        const words = this.tokenize(text);
        
        if (sentences.length < 3 || words.length < 30) {
            return this.createEmptyResult('Insufficient content for analysis');
        }
        
        // Extract all features
        const features = this.extractAllFeatures(text, sentences, words);
        
        // Calculate humanization probability
        const result = this.calculateProbability(features);
        
        // Generate detailed analysis
        result.features = features;
        result.analysis = this.generateAnalysis(features, result);
        result.flags = this.identifyFlags(features, result);
        
        return result;
    }
    
    /**
     * Extract all humanization detection features
     */
    extractAllFeatures(text, sentences, words) {
        const features = {};
        
        // 1. Variance Analysis
        Object.assign(features, this.varianceAnalysis(sentences, words));
        
        // 2. Autocorrelation Analysis
        Object.assign(features, this.autocorrelationAnalysis(sentences));
        
        // 3. Correlation Analysis
        Object.assign(features, this.correlationAnalysis(text, sentences, words));
        
        // 4. Synonym Analysis
        Object.assign(features, this.synonymAnalysis(words));
        
        // 5. Contraction Analysis
        Object.assign(features, this.contractionAnalysis(text, sentences));
        
        // 6. Structure Analysis
        Object.assign(features, this.structureAnalysis(sentences));
        
        // 7. Sophistication Analysis
        Object.assign(features, this.sophisticationAnalysis(words, sentences));
        
        // 8. N-gram Analysis
        Object.assign(features, this.ngramAnalysis(words));
        
        // 9. Punctuation Analysis
        Object.assign(features, this.punctuationAnalysis(text, sentences));
        
        // 10. Discourse Analysis
        Object.assign(features, this.discourseAnalysis(text, sentences));
        
        // 11. Word Length Analysis
        Object.assign(features, this.wordLengthAnalysis(words));
        
        // 12. Entropy Analysis
        Object.assign(features, this.entropyAnalysis(words, sentences));
        
        return features;
    }
    
    /**
     * Calculate humanization probability from features
     */
    calculateProbability(features) {
        if (!this.config || !this.config.weights) {
            // Fallback: use heuristic calculation
            return this.heuristicProbability(features);
        }
        
        // Use trained model weights
        let z = this.config.bias || 0;
        
        for (const [featureName, weight] of Object.entries(this.config.weights)) {
            let value = features[featureName] || 0.5;
            
            // Normalize using training stats
            if (this.config.featureStats) {
                const mean = this.config.featureStats.mean[featureName] || 0.5;
                const std = this.config.featureStats.std[featureName] || 1;
                value = (value - mean) / std;
            }
            
            z += value * weight;
        }
        
        // Sigmoid
        const probability = 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, z))));
        
        return {
            humanizedProbability: probability,
            isHumanized: probability >= 0.5,
            confidence: Math.abs(probability - 0.5) * 2,
            verdict: this.getVerdict(probability)
        };
    }
    
    /**
     * Heuristic probability when model weights aren't available
     */
    heuristicProbability(features) {
        let score = 0;
        let maxScore = 0;
        
        // Weights for each feature category
        const weights = {
            // Variance indicators (humanizers add too-uniform variance)
            variance_stability: 2.0,
            local_variance_consistency: 1.5,
            
            // Broken correlations (key indicator)
            feature_correlation_breaks: 2.5,
            length_complexity_corr: -1.5, // Inverted - low correlation = humanized
            
            // Synonym substitution
            rare_synonym_ratio: 2.0,
            sophistication_jumps: 1.5,
            
            // Artificial contractions
            artificial_contraction_score: 2.0,
            contraction_uniformity: 1.0,
            
            // Structure uniformity
            structure_template_score: 1.5,
            parallelism_score: 1.0,
            
            // Entropy patterns
            entropy_stability: 1.5,
            
            // Discourse markers (AI patterns left after humanization)
            hedging_density: 1.5,
            transition_density: 1.0,
        };
        
        for (const [feature, weight] of Object.entries(weights)) {
            const value = features[feature] || 0.5;
            if (weight > 0) {
                score += value * weight;
                maxScore += weight;
            } else {
                // Inverted features
                score += (1 - value) * Math.abs(weight);
                maxScore += Math.abs(weight);
            }
        }
        
        const probability = maxScore > 0 ? score / maxScore : 0.5;
        
        return {
            humanizedProbability: probability,
            isHumanized: probability >= 0.5,
            confidence: Math.abs(probability - 0.5) * 2,
            verdict: this.getVerdict(probability)
        };
    }
    
    /**
     * Get verdict string based on probability
     */
    getVerdict(probability) {
        if (probability >= 0.85) {
            return 'STRONGLY HUMANIZED';
        } else if (probability >= 0.70) {
            return 'LIKELY HUMANIZED';
        } else if (probability >= 0.55) {
            return 'POSSIBLY HUMANIZED';
        } else if (probability >= 0.45) {
            return 'UNCERTAIN';
        } else if (probability >= 0.30) {
            return 'LIKELY GENUINE';
        } else {
            return 'GENUINE HUMAN';
        }
    }
    
    // ===== FEATURE EXTRACTION METHODS =====
    
    varianceAnalysis(sentences, words) {
        const features = {};
        
        const sentLengths = sentences.map(s => this.tokenize(s).length);
        if (sentLengths.length < 3) {
            return { variance_of_variance: 0.5, variance_stability: 0.5, local_variance_consistency: 0.5 };
        }
        
        const mean = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
        const firstOrderVar = sentLengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / sentLengths.length;
        
        // Calculate local variances
        const windowSize = Math.max(3, Math.floor(sentLengths.length / 4));
        const localVariances = [];
        
        for (let i = 0; i <= sentLengths.length - windowSize; i++) {
            const window = sentLengths.slice(i, i + windowSize);
            const wMean = window.reduce((a, b) => a + b, 0) / window.length;
            const wVar = window.reduce((sum, l) => sum + Math.pow(l - wMean, 2), 0) / window.length;
            localVariances.push(wVar);
        }
        
        if (localVariances.length > 0) {
            const vovMean = localVariances.reduce((a, b) => a + b, 0) / localVariances.length;
            const vov = localVariances.reduce((sum, v) => sum + Math.pow(v - vovMean, 2), 0) / localVariances.length;
            
            features.variance_of_variance = Math.min(1, vov / (firstOrderVar + 1));
            
            const lvStd = Math.sqrt(localVariances.reduce((sum, v) => 
                sum + Math.pow(v - vovMean, 2), 0) / localVariances.length);
            features.variance_stability = 1 - Math.min(1, lvStd / (vovMean + 0.1));
        } else {
            features.variance_of_variance = 0.5;
            features.variance_stability = 0.5;
        }
        
        // Local variance consistency
        if (localVariances.length > 2) {
            const lvMean = localVariances.reduce((a, b) => a + b, 0) / localVariances.length;
            const lvStd = Math.sqrt(localVariances.reduce((sum, v) => 
                sum + Math.pow(v - lvMean, 2), 0) / localVariances.length);
            features.local_variance_consistency = 1 - Math.min(1, lvStd / (lvMean + 0.1));
        } else {
            features.local_variance_consistency = 0.5;
        }
        
        return features;
    }
    
    autocorrelationAnalysis(sentences) {
        const features = {};
        
        const sentLengths = sentences.map(s => this.tokenize(s).length);
        if (sentLengths.length < 5) {
            return { autocorr_decay_rate: 0.5, autocorr_flatness: 0.5, autocorr_periodicity: 0.5 };
        }
        
        const mean = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length;
        const variance = sentLengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / sentLengths.length;
        
        if (variance < 0.01) {
            return { autocorr_decay_rate: 0.5, autocorr_flatness: 1.0, autocorr_periodicity: 0.0 };
        }
        
        const autocorrs = [];
        const maxLag = Math.min(5, Math.floor(sentLengths.length / 2));
        
        for (let lag = 1; lag <= maxLag; lag++) {
            let cov = 0;
            for (let i = 0; i < sentLengths.length - lag; i++) {
                cov += (sentLengths[i] - mean) * (sentLengths[i + lag] - mean);
            }
            cov /= (sentLengths.length - lag);
            autocorrs.push(cov / variance);
        }
        
        if (autocorrs.length > 1) {
            const decayRate = (autocorrs[0] - autocorrs[autocorrs.length - 1]) / autocorrs.length;
            features.autocorr_decay_rate = Math.min(1, Math.max(0, decayRate + 0.5));
        } else {
            features.autocorr_decay_rate = 0.5;
        }
        
        // Flatness
        const acMean = autocorrs.reduce((a, b) => a + b, 0) / autocorrs.length;
        const acVar = autocorrs.reduce((sum, a) => sum + Math.pow(a - acMean, 2), 0) / autocorrs.length;
        features.autocorr_flatness = 1 - Math.min(1, acVar * 10);
        
        // Periodicity
        if (autocorrs.length >= 3) {
            let alternating = 0;
            for (let i = 0; i < autocorrs.length - 1; i++) {
                if (autocorrs[i] * autocorrs[i + 1] < 0) alternating++;
            }
            features.autocorr_periodicity = alternating / (autocorrs.length - 1);
        } else {
            features.autocorr_periodicity = 0;
        }
        
        return features;
    }
    
    correlationAnalysis(text, sentences, words) {
        const features = {};
        
        if (sentences.length < 5) {
            return { length_complexity_corr: 0.5, vocab_structure_corr: 0.5, feature_correlation_breaks: 0.5 };
        }
        
        // Length vs complexity correlation
        const sentData = sentences.map(s => {
            const sWords = this.tokenize(s);
            const complexity = sWords.length > 0 
                ? sWords.filter(w => w.length > 6).length / sWords.length 
                : 0;
            return { length: sWords.length, complexity };
        }).filter(d => d.length > 0);
        
        if (sentData.length >= 5) {
            const lengths = sentData.map(d => d.length);
            const complexities = sentData.map(d => d.complexity);
            const corr = this.pearsonCorrelation(lengths, complexities);
            features.length_complexity_corr = (corr + 1) / 2;
        } else {
            features.length_complexity_corr = 0.5;
        }
        
        // Vocab richness vs structure correlation
        const window = 3;
        const uniqueRatios = [];
        const structScores = [];
        
        for (let i = 0; i <= sentences.length - window; i++) {
            const chunk = sentences.slice(i, i + window);
            const chunkWords = chunk.flatMap(s => this.tokenize(s));
            if (chunkWords.length > 0) {
                uniqueRatios.push(new Set(chunkWords).size / chunkWords.length);
                const chunkLens = chunk.map(s => this.tokenize(s).length);
                const chunkMean = chunkLens.reduce((a, b) => a + b, 0) / chunkLens.length;
                const chunkStd = Math.sqrt(chunkLens.reduce((sum, l) => 
                    sum + Math.pow(l - chunkMean, 2), 0) / chunkLens.length);
                structScores.push(chunkStd / (chunkMean + 0.1));
            }
        }
        
        if (uniqueRatios.length >= 3) {
            const corr = this.pearsonCorrelation(uniqueRatios, structScores);
            features.vocab_structure_corr = (corr + 1) / 2;
        } else {
            features.vocab_structure_corr = 0.5;
        }
        
        // Correlation break score
        const expectedCorr = 0.3;
        features.feature_correlation_breaks = Math.abs((features.length_complexity_corr || 0.5) - expectedCorr - 0.5);
        
        return features;
    }
    
    synonymAnalysis(words) {
        const features = {};
        
        if (words.length < 20) {
            return { synonym_cluster_usage: 0.5, rare_synonym_ratio: 0.5, sophistication_jumps: 0.5 };
        }
        
        const wordFreq = {};
        words.forEach(w => { wordFreq[w] = (wordFreq[w] || 0) + 1; });
        
        let rareSynonymsUsed = 0;
        let commonWordsUsed = 0;
        
        for (const [baseWord, synonyms] of Object.entries(this.synonymClusters)) {
            if (wordFreq[baseWord]) commonWordsUsed += wordFreq[baseWord];
            for (const syn of synonyms) {
                if (wordFreq[syn]) rareSynonymsUsed += wordFreq[syn];
            }
        }
        
        const totalClusterWords = rareSynonymsUsed + commonWordsUsed;
        features.rare_synonym_ratio = totalClusterWords > 0 ? rareSynonymsUsed / totalClusterWords : 0;
        features.synonym_cluster_usage = totalClusterWords / words.length;
        
        // Sophistication jumps
        const sophScores = [];
        for (let i = 0; i < words.length - 5; i += 5) {
            const window = words.slice(i, i + 5);
            const avgLen = window.reduce((sum, w) => sum + w.length, 0) / window.length;
            sophScores.push(avgLen);
        }
        
        if (sophScores.length > 2) {
            const jumps = [];
            for (let i = 0; i < sophScores.length - 1; i++) {
                jumps.push(Math.abs(sophScores[i + 1] - sophScores[i]));
            }
            features.sophistication_jumps = jumps.reduce((a, b) => a + b, 0) / jumps.length / 3;
        } else {
            features.sophistication_jumps = 0;
        }
        
        return features;
    }
    
    contractionAnalysis(text, sentences) {
        const features = {};
        
        const contractions = (text.toLowerCase().match(/\b\w+'\w+\b/g) || []);
        const totalWords = (text.match(/\b\w+\b/g) || []).length;
        
        if (totalWords < 20) {
            return { contraction_rate: 0.5, contraction_uniformity: 0.5, artificial_contraction_score: 0.5 };
        }
        
        features.contraction_rate = Math.min(1, contractions.length / (totalWords / 20));
        
        // Contraction uniformity
        if (sentences.length >= 3 && contractions.length >= 2) {
            const contractionsPerSent = sentences.map(s => 
                (s.toLowerCase().match(/\b\w+'\w+\b/g) || []).length
            );
            const mean = contractionsPerSent.reduce((a, b) => a + b, 0) / contractionsPerSent.length;
            if (mean > 0) {
                const std = Math.sqrt(contractionsPerSent.reduce((sum, c) => 
                    sum + Math.pow(c - mean, 2), 0) / contractionsPerSent.length);
                features.contraction_uniformity = 1 - Math.min(1, std / (mean + 0.1));
            } else {
                features.contraction_uniformity = 0.5;
            }
        } else {
            features.contraction_uniformity = 0.5;
        }
        
        // Artificial contraction score
        if (features.contraction_rate > 0.3 && features.contraction_uniformity > 0.7) {
            features.artificial_contraction_score = (features.contraction_rate + features.contraction_uniformity) / 2;
        } else {
            features.artificial_contraction_score = 0;
        }
        
        return features;
    }
    
    structureAnalysis(sentences) {
        const features = {};
        
        if (sentences.length < 3) {
            return { sentence_start_diversity: 0.5, structure_template_score: 0.5, parallelism_score: 0.5 };
        }
        
        // Sentence start diversity
        const starts = sentences.map(s => {
            const words = this.tokenize(s);
            return words[0] || '';
        }).filter(s => s);
        
        features.sentence_start_diversity = starts.length > 0 
            ? new Set(starts).size / starts.length 
            : 0.5;
        
        // Structure template score
        const signatures = sentences.map(s => {
            const words = this.tokenize(s).slice(0, 3);
            return words.map(w => w[0] || '').join('_');
        });
        
        const sigCounts = {};
        signatures.forEach(sig => { sigCounts[sig] = (sigCounts[sig] || 0) + 1; });
        const repeatedSigs = Object.values(sigCounts).filter(c => c > 1).length;
        features.structure_template_score = repeatedSigs / sentences.length;
        
        // Parallelism score
        let similarPairs = 0;
        for (let i = 0; i < sentences.length - 1; i++) {
            const len1 = this.tokenize(sentences[i]).length;
            const len2 = this.tokenize(sentences[i + 1]).length;
            if (Math.abs(len1 - len2) <= 2) similarPairs++;
        }
        features.parallelism_score = similarPairs / (sentences.length - 1);
        
        return features;
    }
    
    sophisticationAnalysis(words, sentences) {
        const features = {};
        
        if (words.length < 20 || sentences.length < 3) {
            return { sophistication_variance: 0.5, sophistication_autocorr: 0.5, word_choice_consistency: 0.5 };
        }
        
        // Per-sentence sophistication
        const sentSoph = sentences.map(s => {
            const sWords = this.tokenize(s);
            return sWords.length > 0 
                ? sWords.reduce((sum, w) => sum + w.length, 0) / sWords.length 
                : 0;
        }).filter(s => s > 0);
        
        if (sentSoph.length >= 3) {
            const mean = sentSoph.reduce((a, b) => a + b, 0) / sentSoph.length;
            const variance = sentSoph.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / sentSoph.length;
            features.sophistication_variance = Math.min(1, variance / 4);
            
            if (sentSoph.length >= 4) {
                const pairs = sentSoph.slice(0, -1).map((s, i) => [s, sentSoph[i + 1]]);
                const corr = this.pearsonCorrelation(
                    pairs.map(p => p[0]),
                    pairs.map(p => p[1])
                );
                features.sophistication_autocorr = (corr + 1) / 2;
            } else {
                features.sophistication_autocorr = 0.5;
            }
        } else {
            features.sophistication_variance = 0.5;
            features.sophistication_autocorr = 0.5;
        }
        
        // Word choice consistency
        const windowRichness = [];
        for (let i = 0; i < words.length - 10; i += 10) {
            const window = words.slice(i, i + 10);
            windowRichness.push(new Set(window).size / 10);
        }
        
        if (windowRichness.length > 2) {
            const mean = windowRichness.reduce((a, b) => a + b, 0) / windowRichness.length;
            const std = Math.sqrt(windowRichness.reduce((sum, r) => 
                sum + Math.pow(r - mean, 2), 0) / windowRichness.length);
            features.word_choice_consistency = 1 - Math.min(1, std * 5);
        } else {
            features.word_choice_consistency = 0.5;
        }
        
        return features;
    }
    
    ngramAnalysis(words) {
        const features = {};
        
        if (words.length < 30) {
            return { bigram_predictability: 0.5, trigram_predictability: 0.5, ngram_surprise_variance: 0.5 };
        }
        
        // Bigram analysis
        const bigrams = [];
        for (let i = 0; i < words.length - 1; i++) {
            bigrams.push(`${words[i]}_${words[i + 1]}`);
        }
        
        const bigramCounts = {};
        bigrams.forEach(bg => { bigramCounts[bg] = (bigramCounts[bg] || 0) + 1; });
        const repeatedBigrams = Object.values(bigramCounts).filter(c => c > 1).length;
        features.bigram_predictability = repeatedBigrams / bigrams.length;
        
        // Trigram analysis
        const trigrams = [];
        for (let i = 0; i < words.length - 2; i++) {
            trigrams.push(`${words[i]}_${words[i + 1]}_${words[i + 2]}`);
        }
        
        const trigramCounts = {};
        trigrams.forEach(tg => { trigramCounts[tg] = (trigramCounts[tg] || 0) + 1; });
        const repeatedTrigrams = Object.values(trigramCounts).filter(c => c > 1).length;
        features.trigram_predictability = trigrams.length > 0 ? repeatedTrigrams / trigrams.length : 0.5;
        
        // N-gram surprise variance
        const wordGivenPrev = {};
        for (let i = 0; i < words.length - 1; i++) {
            const w1 = words[i];
            const w2 = words[i + 1];
            if (!wordGivenPrev[w1]) wordGivenPrev[w1] = {};
            wordGivenPrev[w1][w2] = (wordGivenPrev[w1][w2] || 0) + 1;
        }
        
        const surprises = [];
        for (let i = 1; i < words.length - 1; i++) {
            const w1 = words[i];
            const w2 = words[i + 1];
            if (wordGivenPrev[w1]) {
                const total = Object.values(wordGivenPrev[w1]).reduce((a, b) => a + b, 0);
                const prob = (wordGivenPrev[w1][w2] || 0) / total;
                surprises.push(-Math.log(prob + 0.01));
            }
        }
        
        if (surprises.length > 0) {
            const mean = surprises.reduce((a, b) => a + b, 0) / surprises.length;
            const variance = surprises.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / surprises.length;
            features.ngram_surprise_variance = Math.min(1, variance / 2);
        } else {
            features.ngram_surprise_variance = 0.5;
        }
        
        return features;
    }
    
    punctuationAnalysis(text, sentences) {
        const features = {};
        
        const wordCount = (text.match(/\b\w+\b/g) || []).length;
        if (wordCount < 20) {
            return { comma_density: 0.5, punctuation_variety: 0.5, punctuation_position_entropy: 0.5 };
        }
        
        // Comma density
        const commas = (text.match(/,/g) || []).length;
        features.comma_density = Math.min(1, commas / (wordCount / 10));
        
        // Punctuation variety
        const punctChars = text.match(/[^\w\s]/g) || [];
        features.punctuation_variety = punctChars.length > 0 
            ? Math.min(1, new Set(punctChars).size / 8) 
            : 0;
        
        // Punctuation position entropy
        const commaPositions = [];
        for (const s of sentences) {
            const sWords = s.split(/\s+/);
            sWords.forEach((w, i) => {
                if (w.includes(',')) {
                    commaPositions.push(i / sWords.length);
                }
            });
        }
        
        if (commaPositions.length >= 3) {
            const bins = [0, 0, 0, 0];
            commaPositions.forEach(pos => {
                const binIdx = Math.min(3, Math.floor(pos * 4));
                bins[binIdx]++;
            });
            const total = bins.reduce((a, b) => a + b, 0);
            const probs = bins.map(b => b / total);
            const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
            features.punctuation_position_entropy = Math.min(1, entropy / 1.4);
        } else {
            features.punctuation_position_entropy = 0.5;
        }
        
        return features;
    }
    
    discourseAnalysis(text, sentences) {
        const features = {};
        
        const textLower = text.toLowerCase();
        const wordCount = (text.match(/\b\w+\b/g) || []).length;
        
        if (wordCount < 20) {
            return { transition_density: 0.5, hedging_density: 0.5, discourse_marker_variety: 0.5 };
        }
        
        // Transition marker density
        const transitionCount = this.transitionMarkers.filter(t => textLower.includes(t)).length;
        features.transition_density = Math.min(1, transitionCount / (sentences.length / 3 + 0.1));
        
        // Hedging density
        const hedgingCount = this.hedgingPhrases.filter(h => textLower.includes(h)).length;
        features.hedging_density = Math.min(1, hedgingCount / (sentences.length / 5 + 0.1));
        
        // Discourse marker variety
        const markersFound = this.transitionMarkers.filter(t => textLower.includes(t));
        features.discourse_marker_variety = transitionCount > 0 
            ? new Set(markersFound).size / transitionCount 
            : 0.5;
        
        return features;
    }
    
    wordLengthAnalysis(words) {
        const features = {};
        
        if (words.length < 20) {
            return { word_length_variance: 0.5, word_length_entropy: 0.5, long_word_clustering: 0.5 };
        }
        
        const lengths = words.map(w => w.length);
        
        // Word length variance
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const variance = lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length;
        features.word_length_variance = Math.min(1, variance / 10);
        
        // Word length entropy
        const lengthCounts = {};
        lengths.forEach(l => { lengthCounts[l] = (lengthCounts[l] || 0) + 1; });
        const total = lengths.length;
        const probs = Object.values(lengthCounts).map(c => c / total);
        const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        features.word_length_entropy = Math.min(1, entropy / 3);
        
        // Long word clustering
        const longWordPositions = lengths.map((l, i) => l >= 8 ? i : -1).filter(i => i >= 0);
        if (longWordPositions.length >= 2) {
            const gaps = [];
            for (let i = 0; i < longWordPositions.length - 1; i++) {
                gaps.push(longWordPositions[i + 1] - longWordPositions[i]);
            }
            const gapMean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
            const gapVar = gaps.reduce((sum, g) => sum + Math.pow(g - gapMean, 2), 0) / gaps.length;
            features.long_word_clustering = 1 - Math.min(1, gapVar / 50);
        } else {
            features.long_word_clustering = 0.5;
        }
        
        return features;
    }
    
    entropyAnalysis(words, sentences) {
        const features = {};
        
        if (words.length < 20) {
            return { lexical_entropy: 0.5, sentence_entropy: 0.5, entropy_stability: 0.5 };
        }
        
        // Lexical entropy
        const wordCounts = {};
        words.forEach(w => { wordCounts[w] = (wordCounts[w] || 0) + 1; });
        const total = words.length;
        const probs = Object.values(wordCounts).map(c => c / total);
        const lexEntropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        const maxEntropy = Math.log(new Set(words).size + 1);
        features.lexical_entropy = maxEntropy > 0 ? lexEntropy / maxEntropy : 0.5;
        
        // Sentence length entropy
        const sentLengths = sentences.map(s => this.tokenize(s).length);
        const lengthCounts = {};
        sentLengths.forEach(l => { lengthCounts[l] = (lengthCounts[l] || 0) + 1; });
        const sentTotal = sentLengths.length;
        const sentProbs = Object.values(lengthCounts).map(c => c / sentTotal);
        const sentEntropy = -sentProbs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
        const sentMaxEntropy = Math.log(new Set(sentLengths).size + 1);
        features.sentence_entropy = sentMaxEntropy > 0 ? sentEntropy / sentMaxEntropy : 0.5;
        
        // Entropy stability
        const windowEntropies = [];
        const windowSize = 20;
        for (let i = 0; i < words.length - windowSize; i += Math.floor(windowSize / 2)) {
            const window = words.slice(i, i + windowSize);
            const wc = {};
            window.forEach(w => { wc[w] = (wc[w] || 0) + 1; });
            const wprobs = Object.values(wc).map(c => c / window.length);
            const went = -wprobs.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
            windowEntropies.push(went);
        }
        
        if (windowEntropies.length > 2) {
            const weMean = windowEntropies.reduce((a, b) => a + b, 0) / windowEntropies.length;
            const weStd = Math.sqrt(windowEntropies.reduce((sum, e) => 
                sum + Math.pow(e - weMean, 2), 0) / windowEntropies.length);
            features.entropy_stability = 1 - Math.min(1, weStd * 2);
        } else {
            features.entropy_stability = 0.5;
        }
        
        return features;
    }
    
    // ===== UTILITY METHODS =====
    
    splitSentences(text) {
        const normalized = text.replace(/\s+/g, ' ').trim();
        return normalized.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 5);
    }
    
    tokenize(text) {
        return (text.toLowerCase().match(/\b[a-z]+\b/g) || []);
    }
    
    pearsonCorrelation(x, y) {
        if (x.length !== y.length || x.length < 2) return 0;
        
        const n = x.length;
        const meanX = x.reduce((a, b) => a + b, 0) / n;
        const meanY = y.reduce((a, b) => a + b, 0) / n;
        
        let num = 0, denX = 0, denY = 0;
        for (let i = 0; i < n; i++) {
            const dx = x[i] - meanX;
            const dy = y[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        
        const den = Math.sqrt(denX * denY);
        return den > 0.0001 ? num / den : 0;
    }
    
    createEmptyResult(reason) {
        return {
            humanizedProbability: 0.5,
            isHumanized: false,
            confidence: 0,
            verdict: 'INSUFFICIENT DATA',
            reason,
            features: {},
            analysis: { summary: reason },
            flags: []
        };
    }
    
    /**
     * Generate detailed analysis from features
     */
    generateAnalysis(features, result) {
        const analysis = {
            summary: '',
            categories: {},
            recommendations: []
        };
        
        // Variance patterns
        const varianceScore = (features.variance_stability || 0.5) + 
                             (features.local_variance_consistency || 0.5);
        analysis.categories.variancePatterns = {
            score: varianceScore / 2,
            interpretation: varianceScore > 1.2 
                ? 'Unusually stable variance - possible artificial variance injection'
                : varianceScore > 0.8 
                    ? 'Normal variance patterns'
                    : 'Natural variance detected'
        };
        
        // Correlation integrity
        const corrScore = features.feature_correlation_breaks || 0.5;
        analysis.categories.correlationIntegrity = {
            score: 1 - corrScore,
            interpretation: corrScore > 0.4
                ? 'Feature correlations appear broken - sign of humanization'
                : 'Natural feature correlations intact'
        };
        
        // Linguistic patterns
        const lingScore = (features.rare_synonym_ratio || 0) + 
                         (features.hedging_density || 0) +
                         (features.transition_density || 0);
        analysis.categories.linguisticPatterns = {
            score: lingScore / 3,
            interpretation: lingScore > 1
                ? 'Elevated use of sophisticated synonyms and academic transitions'
                : 'Normal linguistic patterns'
        };
        
        // Contraction authenticity
        const contrScore = features.artificial_contraction_score || 0;
        analysis.categories.contractionAuthenticity = {
            score: 1 - contrScore,
            interpretation: contrScore > 0.5
                ? 'Contraction patterns appear artificially inserted'
                : 'Natural contraction usage'
        };
        
        // Generate summary
        if (result.humanizedProbability >= 0.7) {
            analysis.summary = 'Strong indicators of AI humanization detected. The text shows patterns ' +
                'consistent with AI-generated content that has been processed through humanization tools.';
            analysis.recommendations.push('Consider this text as potentially AI-generated with post-processing');
            analysis.recommendations.push('Look for the original AI-like structure beneath surface variations');
        } else if (result.humanizedProbability >= 0.5) {
            analysis.summary = 'Moderate indicators of possible humanization. Some patterns suggest ' +
                'artificial modifications, but evidence is not conclusive.';
            analysis.recommendations.push('Exercise caution - text may have been partially modified');
        } else {
            analysis.summary = 'Text appears to be genuine human writing. Natural patterns and correlations ' +
                'are intact, with no strong indicators of AI humanization.';
        }
        
        return analysis;
    }
    
    /**
     * Identify specific humanization flags
     */
    identifyFlags(features, result) {
        const flags = [];
        
        if ((features.variance_stability || 0) > 0.75) {
            flags.push({
                type: 'variance',
                severity: 'high',
                message: 'Suspiciously stable variance patterns',
                detail: 'Sentence length variance is artificially uniform across the text'
            });
        }
        
        if ((features.feature_correlation_breaks || 0) > 0.4) {
            flags.push({
                type: 'correlation',
                severity: 'high',
                message: 'Broken natural correlations',
                detail: 'Expected correlations between sentence length and complexity are disrupted'
            });
        }
        
        if ((features.rare_synonym_ratio || 0) > 0.5) {
            flags.push({
                type: 'vocabulary',
                severity: 'medium',
                message: 'Unusual synonym substitutions',
                detail: 'High usage of sophisticated synonyms in place of common words'
            });
        }
        
        if ((features.artificial_contraction_score || 0) > 0.5) {
            flags.push({
                type: 'contractions',
                severity: 'medium',
                message: 'Artificial contraction patterns',
                detail: 'Contractions appear mechanically inserted at regular intervals'
            });
        }
        
        if ((features.hedging_density || 0) > 0.3) {
            flags.push({
                type: 'discourse',
                severity: 'medium',
                message: 'Residual AI hedging patterns',
                detail: 'AI-typical hedging phrases remain despite humanization attempts'
            });
        }
        
        if ((features.entropy_stability || 0) > 0.8) {
            flags.push({
                type: 'entropy',
                severity: 'medium',
                message: 'Artificially stable entropy',
                detail: 'Text entropy is too consistent across sections'
            });
        }
        
        if ((features.structure_template_score || 0) > 0.3) {
            flags.push({
                type: 'structure',
                severity: 'low',
                message: 'Repeated structural patterns',
                detail: 'Sentence structures follow templates despite surface variation'
            });
        }
        
        return flags;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FlareAnalyzer };
}

// Global export for browser
if (typeof window !== 'undefined') {
    window.FlareAnalyzer = FlareAnalyzer;
}
