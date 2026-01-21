/**
 * SUPERNOVA - Neural-Enhanced AI Detection Model
 * 
 * The most advanced VERITAS model with 97%+ accuracy on high-confidence samples.
 * Uses sentence embeddings + heuristic features + XGBoost classifier.
 * 
 * Runs entirely in-browser using:
 * - Transformers.js for sentence embeddings
 * - ONNX Runtime Web for XGBoost inference
 */

const VERITAS_SUPERNOVA = {
    name: 'Supernova',
    version: '1.0',
    description: 'Neural-enhanced AI detection with 97%+ accuracy',
    
    // Model state
    ready: false,
    loading: false,
    embedder: null,
    onnxSession: null,
    scalerParams: null,
    
    // Performance metrics
    metrics: {
        accuracy: 0.9530,
        highConfAccuracy: 0.9728,
        highConfCoverage: 0.944,
        aucRoc: 0.9908
    },
    
    /**
     * Initialize SUPERNOVA model
     * Downloads embedding model (~30MB) and ONNX classifier (~65MB)
     */
    async initialize(progressCallback = null) {
        if (this.ready) return true;
        if (this.loading) {
            // Wait for loading to complete
            while (this.loading) {
                await new Promise(r => setTimeout(r, 100));
            }
            return this.ready;
        }
        
        this.loading = true;
        
        try {
            // Step 1: Load Transformers.js pipeline
            if (progressCallback) progressCallback('Loading embedding model...', 0.1);
            
            // Use pre-loaded Transformers.js if available, otherwise dynamic import
            let pipeline;
            if (window.TransformersJS && window.TransformersJS.pipeline) {
                pipeline = window.TransformersJS.pipeline;
            } else {
                const transformers = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1');
                pipeline = transformers.pipeline;
            }
            
            this.embedder = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
                progress_callback: (progress) => {
                    if (progressCallback && progress.status === 'progress') {
                        progressCallback(`Downloading embeddings: ${Math.round(progress.progress)}%`, 0.1 + progress.progress * 0.004);
                    }
                }
            });
            
            if (progressCallback) progressCallback('Loading classifier...', 0.5);
            
            // Step 2: Use pre-loaded ONNX Runtime (window.ort from CDN script)
            if (!window.ort) {
                throw new Error('ONNX Runtime not loaded. Ensure ort.min.js is included before supernova-model.js');
            }
            
            // Load the ONNX model
            const modelUrl = 'training/models/Supernova/supernova_xgb.onnx';
            this.onnxSession = await window.ort.InferenceSession.create(modelUrl);
            
            if (progressCallback) progressCallback('Loading scaler...', 0.9);
            
            // Step 3: Load scaler parameters
            const scalerResponse = await fetch('training/models/Supernova/scaler_params.json');
            this.scalerParams = await scalerResponse.json();
            
            this.ready = true;
            this.loading = false;
            
            if (progressCallback) progressCallback('SUPERNOVA ready!', 1.0);
            console.log('SUPERNOVA initialized successfully');
            
            return true;
        } catch (error) {
            console.error('Failed to initialize SUPERNOVA:', error);
            this.loading = false;
            throw error;
        }
    },
    
    /**
     * Extract heuristic features from text
     * These are the 31 hand-crafted features
     */
    extractHeuristicFeatures(text) {
        if (!text || text.length < 10) return null;
        
        const words = text.split(/\s+/);
        const wordCount = words.length || 1;
        
        const sentences = text.split(/[.!?]+/).filter(s => s.trim());
        const sentCount = sentences.length || 1;
        
        const features = {};
        
        // Top discriminators
        features.third_he_she = (text.match(/\b(he|she|him|her|his|hers)\b/gi) || []).length / wordCount;
        features.first_me = (text.match(/\b(me|my|mine|myself)\b/gi) || []).length / wordCount;
        features.answer_opener = /^(Yes|No|Sure|Certainly|Of course|Absolutely)\b/i.test(text.trim()) ? 1 : 0;
        features.ellipsis_count = (text.match(/\.{3}|…/g) || []).length;
        features.instruction_phrases = (text.match(/\b(first,|second,|finally,|step \d|for example)\b/gi) || []).length;
        features.attribution = (text.match(/\b(said|says|told|according to|noted|stated)\b/gi) || []).length;
        features.numbered_items = (text.match(/^\s*\d+[.)]\s+/gm) || []).length;
        features.helpful_phrases = (text.match(/\b(here is|let me|feel free|I hope this)\b/gi) || []).length;
        features.proper_nouns = (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g) || []).length / wordCount;
        
        // Structural features
        const sentLengths = sentences.map(s => s.trim().split(/\s+/).length);
        features.avg_sent_len = sentLengths.reduce((a, b) => a + b, 0) / sentLengths.length || 0;
        features.sent_len_std = Math.sqrt(sentLengths.reduce((sum, l) => sum + Math.pow(l - features.avg_sent_len, 2), 0) / sentLengths.length) || 0;
        features.sent_count = sentCount;
        features.min_sent_len = Math.min(...sentLengths) || 0;
        
        const paragraphs = text.split(/\n\n/).filter(p => p.trim());
        features.para_count = paragraphs.length;
        
        // Pronouns
        features.first_I = (text.match(/\bI\b/g) || []).length / wordCount;
        features.first_we = (text.match(/\b(we|us|our)\b/gi) || []).length / wordCount;
        features.second_you = (text.match(/\b(you|your)\b/gi) || []).length / wordCount;
        
        // Punctuation
        features.colon_rate = (text.match(/:/g) || []).length / sentCount;
        features.question_rate = (text.match(/\?/g) || []).length / sentCount;
        features.paren_count = (text.match(/\(/g) || []).length;
        
        // Temporal references
        features.month_mentions = (text.match(/\b(January|February|March|April|May|June|July|August|September|October|November|December)\b/g) || []).length;
        features.year_mentions = (text.match(/\b(19|20)\d{2}\b/g) || []).length;
        features.time_mentions = (text.match(/\b\d{1,2}:\d{2}\b/g) || []).length;
        features.specific_dates = (text.match(/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g) || []).length;
        
        // Style markers
        features.has_code = /```|def\s+\w+|function\s*\(/.test(text) ? 1 : 0;
        features.has_html = /<[a-z]+[^>]*>/i.test(text) ? 1 : 0;
        features.discourse_markers = (text.match(/\b(however|therefore|furthermore|moreover|consequently)\b/gi) || []).length;
        features.contraction_rate = (text.match(/\b\w+'(t|re|ve|ll|d|s|m)\b/gi) || []).length / wordCount;
        features.casual_words = (text.match(/\b(lol|haha|yeah|nah|ok|gonna|wanna)\b/gi) || []).length;
        
        features.word_count = wordCount;
        features.vocab_richness = new Set(words.map(w => w.toLowerCase())).size / wordCount;
        
        return features;
    },
    
    /**
     * Get sentence embedding using Transformers.js
     */
    async getEmbedding(text) {
        if (!this.embedder) {
            throw new Error('SUPERNOVA not initialized. Call initialize() first.');
        }
        
        // Truncate to ~512 tokens
        const truncated = text.split(/\s+/).slice(0, 400).join(' ');
        
        const output = await this.embedder(truncated, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
    },
    
    /**
     * Scale features using saved scaler parameters
     */
    scaleFeatures(features) {
        const scaled = [];
        for (let i = 0; i < features.length; i++) {
            const mean = this.scalerParams.mean[i] || 0;
            const scale = this.scalerParams.scale[i] || 1;
            scaled.push((features[i] - mean) / scale);
        }
        return scaled;
    },
    
    /**
     * Analyze text and return prediction
     */
    async analyze(text) {
        if (!this.ready) {
            throw new Error('SUPERNOVA not initialized. Call initialize() first.');
        }
        
        if (!text || text.length < 10) {
            return {
                error: 'Text too short (minimum 10 characters)',
                prediction: null
            };
        }
        
        const startTime = performance.now();
        
        // Step 1: Extract heuristic features
        const heuristicFeatures = this.extractHeuristicFeatures(text);
        const heuristicArray = Object.values(heuristicFeatures);
        
        // Step 2: Get embedding
        const embedding = await this.getEmbedding(text);
        
        // Step 3: Combine features
        const allFeatures = [...heuristicArray, ...embedding];
        
        // Step 4: Scale features
        const scaledFeatures = this.scaleFeatures(allFeatures);
        
        // Step 5: Run ONNX inference
        const inputTensor = new window.ort.Tensor('float32', new Float32Array(scaledFeatures), [1, scaledFeatures.length]);
        
        const feeds = { float_input: inputTensor };
        const results = await this.onnxSession.run(feeds);
        
        // Get prediction and probability
        const outputName = Object.keys(results)[0];
        const prediction = results[outputName].data[0];
        
        // Get probabilities if available
        let aiProbability = prediction > 0.5 ? prediction : 1 - prediction;
        if (results.probabilities) {
            aiProbability = results.probabilities.data[1];
        }
        
        // Determine confidence level
        let confidence, confidenceLabel;
        if (aiProbability > 0.9 || aiProbability < 0.1) {
            confidence = 'very_high';
            confidenceLabel = 'Very High';
        } else if (aiProbability > 0.8 || aiProbability < 0.2) {
            confidence = 'high';
            confidenceLabel = 'High';
        } else if (aiProbability > 0.7 || aiProbability < 0.3) {
            confidence = 'medium';
            confidenceLabel = 'Medium';
        } else {
            confidence = 'low';
            confidenceLabel = 'Low';
        }
        
        const inferenceTime = performance.now() - startTime;
        
        return {
            prediction: prediction > 0.5 ? 'ai' : 'human',
            aiProbability: aiProbability,
            humanProbability: 1 - aiProbability,
            confidence: confidence,
            confidenceLabel: confidenceLabel,
            confidenceScore: Math.max(aiProbability, 1 - aiProbability),
            flagForReview: confidence === 'low' || confidence === 'medium',
            modelVersion: this.version,
            inferenceTimeMs: Math.round(inferenceTime),
            
            // For VERITAS integration
            score: prediction > 0.5 ? aiProbability : 1 - aiProbability,
            classification: prediction > 0.5 ? 'AI-Generated' : 'Human-Written',
            indicators: this._getIndicators(heuristicFeatures, prediction > 0.5)
        };
    },
    
    /**
     * Generate human-readable indicators
     */
    _getIndicators(features, isAI) {
        const indicators = [];
        
        if (isAI) {
            if (features.helpful_phrases > 0) indicators.push('Contains AI helper phrases');
            if (features.instruction_phrases > 0) indicators.push('Instructional language detected');
            if (features.numbered_items > 0) indicators.push('Structured list format');
            if (features.discourse_markers > 2) indicators.push('Heavy use of discourse markers');
            if (features.answer_opener) indicators.push('Starts with typical AI response opener');
        } else {
            if (features.attribution > 0) indicators.push('Contains human attribution (said, told)');
            if (features.ellipsis_count > 0) indicators.push('Uses ellipses (informal trailing thoughts)');
            if (features.casual_words > 0) indicators.push('Contains casual language');
            if (features.contraction_rate > 0.02) indicators.push('Uses contractions naturally');
            if (features.first_me > 0.03) indicators.push('Personal first-person narrative');
        }
        
        return indicators.slice(0, 5);
    },
    
    /**
     * For VERITAS analyzer-engine integration
     */
    async analyzeForVeritas(text) {
        const result = await this.analyze(text);
        
        if (result.error) {
            return {
                score: 0.5,
                classification: 'Unknown',
                confidence: 0,
                explanation: result.error
            };
        }
        
        // Also extract heuristic features for detailed analysis
        const heuristicFeatures = this.extractHeuristicFeatures(text);
        
        return {
            score: result.score,
            classification: result.classification,
            confidence: result.confidenceScore,
            aiProbability: result.aiProbability,
            humanProbability: result.humanProbability,
            confidenceLevel: result.confidenceLabel,
            flagForReview: result.flagForReview,
            indicators: result.indicators,
            inferenceTimeMs: result.inferenceTimeMs,
            heuristicFeatures: heuristicFeatures
        };
    }
};

// Export for use in analyzer-engine
if (typeof window !== 'undefined') {
    window.VERITAS_SUPERNOVA = VERITAS_SUPERNOVA;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_SUPERNOVA;
}
