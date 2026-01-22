/**
 * VERITAS Flare V2 - Human vs Humanized Detection
 * ================================================
 * 
 * A specialized ML model that detects whether text classified as "human"
 * is genuinely human-written or AI-generated text that has been 
 * paraphrased/humanized to evade detection.
 * 
 * Architecture:
 * - XGBoost classifier (914 trees, depth 16)
 * - all-MiniLM-L6-v2 sentence embeddings (384 dimensions)
 * - 57 humanization-specific heuristic features
 * - Total: 441 features
 * 
 * Performance:
 * - Accuracy: 98.00%
 * - Precision: 97.16%
 * - Recall: 98.90%
 * - ROC AUC: 99.71%
 * 
 * Training Data:
 * - 100,000 genuine human samples (RAID dataset + Wikipedia)
 * - 100,000 humanized AI samples (RAID paraphrase attacks)
 * 
 * Usage:
 * - Designed as secondary layer for SUPERNOVA
 * - Only triggers when SUPERNOVA classifies text as "human"
 * - Provides additional check for humanization artifacts
 * 
 * @author VERITAS Team
 * @version 2.0.0
 * @license MIT
 */

const VERITAS_FLARE_V2 = {
    // Model state
    session: null,
    embedder: null,
    ready: false,
    loading: false,
    
    // Model configuration
    config: {
        modelName: 'Flare V2',
        version: '2.0.0',
        type: 'human_vs_humanized',
        accuracy: 0.98,
        threshold: 0.5,
        featureCount: 441,
        heuristicFeatures: 57,
        embeddingDim: 384,
        onnxPath: 'models/FlareV2/flare_v2.onnx',
        labels: {
            0: 'human',
            1: 'humanized'
        }
    },
    
    // Feature names (must match training order exactly)
    featureNames: [
        'chars', 'words', 'sents', 'avg_word_len',
        'sent_mean', 'sent_std', 'sent_cv', 'sent_range', 'sent_skew', 'sent_kurt',
        'ai_residue', 'ai_residue_pct',
        'paraphrase_markers', 'thesaurus_words', 'formal_synonyms',
        'human_informal', 'human_pct',
        'contractions', 'contraction_rate',
        'exclamations', 'questions', 'ellipses', 'dashes', 'parens', 'commas', 'semicolons', 'colons', 'quotes',
        'first_person', 'first_person_rate',
        'vocab_richness', 'hapax_ratio', 'top_word_freq',
        'bigram_rep', 'trigram_rep', 'quadgram_rep',
        'starter_diversity', 'starter_rep',
        'paragraphs', 'para_std', 'para_cv',
        'style_shift',
        'short_words', 'medium_words', 'long_words', 'very_long_words',
        'transitions', 'transition_rate',
        'passive_voice', 'passive_rate',
        'hedges', 'hedge_rate',
        'emphatics',
        'numbers', 'number_rate',
        'caps_ratio',
        'entropy'
    ],
    
    // Humanization detection patterns
    patterns: {
        aiResidue: [
            'it is important to note', 'it should be noted', 'in conclusion',
            'furthermore', 'moreover', 'additionally', 'consequently',
            'in summary', 'to summarize', 'in essence', 'fundamentally',
            'it is worth mentioning', 'it is essential', 'it is crucial',
            'comprehensive', 'robust', 'leverage', 'utilize', 'facilitate',
            'delve', 'crucial', 'tapestry', 'multifaceted', 'intricacies',
            'landscape', 'navigate', 'nuanced', 'holistic', 'overarching'
        ],
        
        humanInformal: [
            "i'm", "i've", "i'll", "i'd", "can't", "won't", "don't", "didn't",
            "wouldn't", "couldn't", "shouldn't", "isn't", "aren't", "wasn't",
            "weren't", "hasn't", "haven't", "hadn't", "gonna", "wanna", "gotta",
            "kinda", "sorta", "y'all", "ain't", "lemme", "gimme", "dunno",
            "tbh", "imo", "imho", "lol", "lmao", "omg", "wtf", "btw", "fyi",
            "haha", "hehe", "lmfao", "bruh", "bro", "dude", "yeah", "yep", "nope",
            "ugh", "meh", "idk", "rn", "ngl", "smh", "ikr", "ofc"
        ],
        
        paraphraseMarkers: [
            'in other words', 'put differently', 'to put it another way',
            'that is to say', 'namely', 'specifically', 'particularly',
            'in particular', 'especially', 'notably', 'as mentioned',
            'as stated', 'as noted', 'put simply', 'simply put'
        ],
        
        thesaurusWords: [
            'utilize', 'commence', 'terminate', 'endeavor', 'ascertain',
            'facilitate', 'implement', 'subsequent', 'prior', 'regarding',
            'pertaining', 'aforementioned', 'henceforth', 'thereby', 'whereby',
            'heretofore', 'notwithstanding', 'inasmuch', 'insofar', 'whilst',
            'amongst', 'towards', 'upon', 'thus', 'hence', 'furthermore',
            'moreover', 'nevertheless', 'nonetheless', 'whereas', 'whereby'
        ],
        
        formalSynonyms: {
            'big': ['large', 'substantial', 'significant', 'considerable'],
            'small': ['tiny', 'minor', 'diminutive', 'modest'],
            'good': ['excellent', 'beneficial', 'advantageous', 'favorable'],
            'bad': ['negative', 'detrimental', 'adverse', 'unfavorable'],
            'important': ['crucial', 'vital', 'essential', 'significant'],
            'show': ['demonstrate', 'illustrate', 'exhibit', 'display'],
            'use': ['utilize', 'employ', 'leverage', 'implement'],
            'make': ['create', 'produce', 'generate', 'construct'],
            'help': ['assist', 'aid', 'facilitate', 'support'],
            'get': ['obtain', 'acquire', 'receive', 'attain']
        },
        
        transitions: [
            'however', 'therefore', 'meanwhile', 'nevertheless', 'furthermore',
            'consequently', 'accordingly', 'hence', 'thus', 'besides', 'although',
            'whereas', 'while', 'despite', 'since', 'because', 'unless', 'until'
        ],
        
        hedges: [
            'perhaps', 'maybe', 'possibly', 'probably', 'might', 'could', 'may',
            'somewhat', 'relatively', 'fairly', 'rather', 'quite', 'seem', 'appear',
            'tend', 'suggest', 'indicate', 'likely', 'unlikely'
        ],
        
        emphatics: [
            'very', 'really', 'extremely', 'absolutely', 'definitely', 'certainly',
            'clearly', 'obviously', 'undoubtedly', 'surely', 'totally', 'completely'
        ]
    },
    
    /**
     * Initialize Flare V2 model
     */
    async initialize(progressCallback = null) {
        if (this.ready) return true;
        if (this.loading) {
            while (this.loading) {
                await new Promise(r => setTimeout(r, 100));
            }
            return this.ready;
        }
        
        this.loading = true;
        
        try {
            // Check for ONNX Runtime
            if (typeof ort === 'undefined') {
                throw new Error('ONNX Runtime not loaded');
            }
            
            if (progressCallback) progressCallback(10, 'Loading Flare V2 model...');
            
            // Load ONNX model
            const modelResponse = await fetch(this.config.onnxPath);
            if (!modelResponse.ok) {
                throw new Error(`Failed to fetch model: ${modelResponse.status}`);
            }
            
            if (progressCallback) progressCallback(40, 'Processing model...');
            
            const modelBuffer = await modelResponse.arrayBuffer();
            this.session = await ort.InferenceSession.create(modelBuffer);
            
            if (progressCallback) progressCallback(70, 'Checking embedding model...');
            
            // Use shared embedder from SUPERNOVA if available
            if (typeof VERITAS_SUPERNOVA !== 'undefined' && VERITAS_SUPERNOVA.embedder) {
                this.embedder = VERITAS_SUPERNOVA.embedder;
                console.log('Flare V2: Using shared SUPERNOVA embedder');
            } else if (typeof pipeline !== 'undefined') {
                // Load our own embedder
                this.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
                console.log('Flare V2: Loaded dedicated embedder');
            } else {
                console.warn('Flare V2: No embedder available, will use heuristics only');
            }
            
            if (progressCallback) progressCallback(100, 'Flare V2 ready');
            
            this.ready = true;
            this.loading = false;
            console.log('✅ Flare V2 initialized successfully');
            return true;
            
        } catch (error) {
            console.error('Flare V2 initialization error:', error);
            this.loading = false;
            return false;
        }
    },
    
    /**
     * Analyze text for humanization
     * @param {string} text - Text to analyze
     * @returns {Object} Analysis result
     */
    async analyze(text) {
        if (!this.ready) {
            const initialized = await this.initialize();
            if (!initialized) {
                return this.fallbackAnalysis(text);
            }
        }
        
        try {
            // Extract features
            const heuristicFeatures = this.extractFeatures(text);
            
            // Get embeddings
            let embeddings = new Array(384).fill(0);
            if (this.embedder) {
                try {
                    const output = await this.embedder(text.slice(0, 1000), {
                        pooling: 'mean',
                        normalize: true
                    });
                    embeddings = Array.from(output.data);
                } catch (e) {
                    console.warn('Flare V2: Embedding extraction failed, using zeros');
                }
            }
            
            // Combine features
            const featureArray = [
                ...this.featureNames.map(name => heuristicFeatures[name] || 0),
                ...embeddings
            ];
            
            // Run inference
            const inputTensor = new ort.Tensor('float32', new Float32Array(featureArray), [1, 441]);
            const results = await this.session.run({ float_input: inputTensor });
            
            // Parse results
            let humanizedProbability = 0.5;
            
            if (results.probabilities) {
                const probs = results.probabilities.data;
                humanizedProbability = probs[1]; // Index 1 = humanized
            } else if (results.label) {
                humanizedProbability = results.label.data[0] === 1 ? 0.8 : 0.2;
            }
            
            const humanProbability = 1 - humanizedProbability;
            const isHumanized = humanizedProbability >= this.config.threshold;
            
            // Generate flags for detected patterns
            const flags = this.generateFlags(text, heuristicFeatures);
            
            return {
                humanProbability,
                humanizedProbability,
                isHumanized,
                confidence: Math.abs(humanizedProbability - 0.5) * 2,
                confidenceLevel: this.getConfidenceLevel(humanizedProbability),
                verdict: isHumanized ? 'HUMANIZED AI' : 'GENUINE HUMAN',
                flags,
                heuristicFeatures,
                modelInfo: {
                    name: this.config.modelName,
                    version: this.config.version,
                    accuracy: this.config.accuracy
                }
            };
            
        } catch (error) {
            console.error('Flare V2 analysis error:', error);
            return this.fallbackAnalysis(text);
        }
    },
    
    /**
     * Extract heuristic features for humanization detection
     */
    extractFeatures(text) {
        if (!text || text.length < 10) {
            return this.emptyFeatures();
        }
        
        const textLower = text.toLowerCase();
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        const f = {};
        
        // Basic stats
        f.chars = text.length;
        f.words = words.length;
        f.sents = sentences.length;
        f.avg_word_len = words.length > 0 ? words.reduce((a, w) => a + w.length, 0) / words.length : 0;
        
        // Sentence analysis
        const sentLens = sentences.map(s => this.tokenize(s).length);
        f.sent_mean = this.mean(sentLens);
        f.sent_std = this.std(sentLens);
        f.sent_cv = f.sent_mean > 0 ? f.sent_std / f.sent_mean : 0;
        f.sent_range = sentLens.length > 0 ? Math.max(...sentLens) - Math.min(...sentLens) : 0;
        f.sent_skew = this.skewness(sentLens);
        f.sent_kurt = this.kurtosis(sentLens);
        
        // AI residue detection
        f.ai_residue = this.patterns.aiResidue.filter(p => textLower.includes(p)).length;
        f.ai_residue_pct = f.ai_residue / Math.max(sentences.length, 1) * 100;
        
        // Paraphrase markers
        f.paraphrase_markers = this.patterns.paraphraseMarkers.filter(p => textLower.includes(p)).length;
        f.thesaurus_words = this.patterns.thesaurusWords.filter(w => textLower.includes(w)).length;
        f.formal_synonyms = this.countFormalSynonyms(words);
        
        // Human informal markers
        f.human_informal = this.patterns.humanInformal.filter(m => textLower.includes(m)).length;
        f.human_pct = f.human_informal / Math.max(words.length, 1) * 100;
        
        // Contractions
        const contractions = textLower.match(/\b\w+[''](?:t|s|re|ve|ll|d|m)\b/g) || [];
        f.contractions = contractions.length;
        f.contraction_rate = contractions.length / Math.max(words.length, 1);
        
        // Punctuation
        f.exclamations = (text.match(/!/g) || []).length;
        f.questions = (text.match(/\?/g) || []).length;
        f.ellipses = (text.match(/\.\.\./g) || []).length;
        f.dashes = (text.match(/[—–]| - /g) || []).length;
        f.parens = (text.match(/\(/g) || []).length;
        f.commas = (text.match(/,/g) || []).length;
        f.semicolons = (text.match(/;/g) || []).length;
        f.colons = (text.match(/:/g) || []).length;
        f.quotes = (text.match(/["']/g) || []).length;
        
        // First person
        const firstPerson = textLower.match(/\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b/g) || [];
        f.first_person = firstPerson.length;
        f.first_person_rate = firstPerson.length / Math.max(words.length, 1);
        
        // Vocabulary
        const uniqueWords = new Set(words);
        f.vocab_richness = uniqueWords.size / Math.max(words.length, 1);
        f.hapax_ratio = [...uniqueWords].filter(w => words.filter(x => x === w).length === 1).length / Math.max(uniqueWords.size, 1);
        
        const wordFreq = {};
        words.forEach(w => wordFreq[w] = (wordFreq[w] || 0) + 1);
        const maxFreq = Math.max(...Object.values(wordFreq), 0);
        f.top_word_freq = maxFreq / Math.max(words.length, 1);
        
        // N-grams
        const bigrams = words.slice(0, -1).map((w, i) => `${w} ${words[i+1]}`);
        const trigrams = words.slice(0, -2).map((w, i) => `${w} ${words[i+1]} ${words[i+2]}`);
        const quadgrams = words.slice(0, -3).map((w, i) => `${w} ${words[i+1]} ${words[i+2]} ${words[i+3]}`);
        
        f.bigram_rep = bigrams.length > 0 ? 1 - new Set(bigrams).size / bigrams.length : 0;
        f.trigram_rep = trigrams.length > 0 ? 1 - new Set(trigrams).size / trigrams.length : 0;
        f.quadgram_rep = quadgrams.length > 0 ? 1 - new Set(quadgrams).size / quadgrams.length : 0;
        
        // Sentence starters
        const starters = sentences.map(s => {
            const words = this.tokenize(s);
            return words.length > 0 ? words[0] : '';
        });
        f.starter_diversity = starters.length > 0 ? new Set(starters).size / starters.length : 0;
        f.starter_rep = 1 - f.starter_diversity;
        
        // Paragraphs
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim());
        f.paragraphs = paragraphs.length;
        if (paragraphs.length > 1) {
            const paraLens = paragraphs.map(p => this.tokenize(p).length);
            f.para_std = this.std(paraLens);
            f.para_cv = this.mean(paraLens) > 0 ? this.std(paraLens) / this.mean(paraLens) : 0;
        } else {
            f.para_std = 0;
            f.para_cv = 0;
        }
        
        // Style shift
        f.style_shift = this.measureStyleShift(sentences);
        
        // Word length distribution
        const wordLens = words.map(w => w.length);
        f.short_words = wordLens.filter(l => l <= 3).length / Math.max(words.length, 1);
        f.medium_words = wordLens.filter(l => l >= 4 && l <= 7).length / Math.max(words.length, 1);
        f.long_words = wordLens.filter(l => l > 7).length / Math.max(words.length, 1);
        f.very_long_words = wordLens.filter(l => l > 10).length / Math.max(words.length, 1);
        
        // Transitions
        f.transitions = this.patterns.transitions.filter(t => textLower.includes(t)).length;
        f.transition_rate = f.transitions / Math.max(sentences.length, 1);
        
        // Passive voice proxy
        const passive = textLower.match(/\b(?:is|are|was|were|been|being)\s+\w+ed\b/g) || [];
        f.passive_voice = passive.length;
        f.passive_rate = passive.length / Math.max(sentences.length, 1);
        
        // Hedges
        f.hedges = this.patterns.hedges.filter(h => textLower.includes(h)).length;
        f.hedge_rate = f.hedges / Math.max(sentences.length, 1);
        
        // Emphatics
        f.emphatics = this.patterns.emphatics.filter(e => textLower.includes(e)).length;
        
        // Numbers
        const numbers = text.match(/\b\d+(?:\.\d+)?\b/g) || [];
        f.numbers = numbers.length;
        f.number_rate = numbers.length / Math.max(words.length, 1);
        
        // Caps
        const caps = (text.match(/[A-Z]/g) || []).length;
        f.caps_ratio = caps / Math.max(text.length, 1);
        
        // Entropy
        f.entropy = this.entropy(words);
        
        return f;
    },
    
    /**
     * Count formal synonyms (thesaurus swaps)
     */
    countFormalSynonyms(words) {
        let count = 0;
        const wordSet = new Set(words);
        for (const [simple, formal] of Object.entries(this.patterns.formalSynonyms)) {
            for (const f of formal) {
                if (wordSet.has(f)) count++;
            }
        }
        return count;
    },
    
    /**
     * Measure style consistency across sentences
     */
    measureStyleShift(sentences) {
        if (sentences.length < 3) return 0;
        
        const scores = sentences.map(sent => {
            const words = this.tokenize(sent.toLowerCase());
            if (words.length === 0) return 0;
            
            const formal = this.patterns.thesaurusWords.filter(w => words.includes(w)).length;
            const informal = this.patterns.humanInformal.filter(w => sent.toLowerCase().includes(w)).length;
            const contractions = (sent.match(/\w+[''](?:t|s|re|ve|ll|d|m)\b/gi) || []).length;
            
            return formal - informal - contractions * 0.5;
        });
        
        return this.std(scores);
    },
    
    /**
     * Generate flags for detected humanization patterns
     */
    generateFlags(text, features) {
        const flags = [];
        
        if (features.ai_residue > 2) {
            flags.push({
                type: 'ai_residue',
                severity: 'high',
                message: 'AI residue phrases detected',
                detail: `Found ${features.ai_residue} characteristic AI phrases that survived paraphrasing`
            });
        }
        
        if (features.paraphrase_markers > 1) {
            flags.push({
                type: 'paraphrase',
                severity: 'medium',
                message: 'Paraphrase tool markers',
                detail: 'Contains phrases commonly added by paraphrasing tools'
            });
        }
        
        if (features.thesaurus_words > 3) {
            flags.push({
                type: 'thesaurus',
                severity: 'medium',
                message: 'Thesaurus word substitutions',
                detail: `${features.thesaurus_words} uncommon formal synonyms detected (possible word swapping)`
            });
        }
        
        if (features.style_shift > 1.5) {
            flags.push({
                type: 'inconsistency',
                severity: 'high',
                message: 'Style inconsistency detected',
                detail: 'Writing style varies significantly between sentences (mixed editing)'
            });
        }
        
        if (features.sent_cv < 0.2 && features.sents > 3) {
            flags.push({
                type: 'uniformity',
                severity: 'low',
                message: 'Uniform sentence structure',
                detail: 'Sentence lengths are unusually consistent (AI baseline pattern)'
            });
        }
        
        if (features.human_informal > 3) {
            flags.push({
                type: 'human',
                severity: 'positive',
                message: 'Human informal markers',
                detail: `${features.human_informal} informal expressions typical of human writing`
            });
        }
        
        if (features.contraction_rate > 0.03) {
            flags.push({
                type: 'human',
                severity: 'positive',
                message: 'Natural contractions',
                detail: 'High contraction usage suggests genuine human writing'
            });
        }
        
        return flags;
    },
    
    /**
     * Fallback analysis when model is unavailable
     */
    fallbackAnalysis(text) {
        const features = this.extractFeatures(text);
        const flags = this.generateFlags(text, features);
        
        // Simple heuristic scoring
        let humanScore = 0;
        
        // Human indicators
        if (features.human_informal > 2) humanScore += 0.2;
        if (features.contraction_rate > 0.02) humanScore += 0.15;
        if (features.ellipses > 0) humanScore += 0.1;
        if (features.first_person_rate > 0.02) humanScore += 0.1;
        
        // Humanization indicators
        if (features.ai_residue > 2) humanScore -= 0.2;
        if (features.thesaurus_words > 3) humanScore -= 0.15;
        if (features.style_shift > 1.5) humanScore -= 0.2;
        if (features.paraphrase_markers > 1) humanScore -= 0.15;
        
        const humanProbability = Math.max(0, Math.min(1, 0.5 + humanScore));
        const humanizedProbability = 1 - humanProbability;
        
        return {
            humanProbability,
            humanizedProbability,
            isHumanized: humanizedProbability > 0.5,
            confidence: Math.abs(humanizedProbability - 0.5) * 2,
            confidenceLevel: 'low',
            verdict: humanizedProbability > 0.5 ? 'POSSIBLY HUMANIZED' : 'LIKELY HUMAN',
            flags,
            heuristicFeatures: features,
            fallback: true,
            modelInfo: {
                name: 'Flare V2 (Heuristic Fallback)',
                version: this.config.version,
                accuracy: 0.75
            }
        };
    },
    
    /**
     * Get confidence level string
     */
    getConfidenceLevel(probability) {
        const distance = Math.abs(probability - 0.5);
        if (distance >= 0.4) return 'high';
        if (distance >= 0.25) return 'medium';
        return 'low';
    },
    
    // Utility functions
    tokenize(text) {
        return (text.toLowerCase().match(/\b\w+\b/g) || []);
    },
    
    splitSentences(text) {
        return text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
    },
    
    mean(arr) {
        return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
    },
    
    std(arr) {
        if (arr.length < 2) return 0;
        const m = this.mean(arr);
        return Math.sqrt(arr.reduce((a, b) => a + (b - m) ** 2, 0) / arr.length);
    },
    
    skewness(arr) {
        if (arr.length < 3) return 0;
        const m = this.mean(arr);
        const s = this.std(arr);
        if (s === 0) return 0;
        return arr.reduce((a, x) => a + ((x - m) / s) ** 3, 0) / arr.length;
    },
    
    kurtosis(arr) {
        if (arr.length < 4) return 0;
        const m = this.mean(arr);
        const s = this.std(arr);
        if (s === 0) return 0;
        return arr.reduce((a, x) => a + ((x - m) / s) ** 4, 0) / arr.length - 3;
    },
    
    entropy(words) {
        if (words.length === 0) return 0;
        const freq = {};
        words.forEach(w => freq[w] = (freq[w] || 0) + 1);
        let h = 0;
        for (const count of Object.values(freq)) {
            const p = count / words.length;
            if (p > 0) h -= p * Math.log2(p);
        }
        return h;
    },
    
    emptyFeatures() {
        const f = {};
        this.featureNames.forEach(name => f[name] = 0);
        return f;
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_FLARE_V2;
}
