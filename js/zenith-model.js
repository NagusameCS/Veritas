/**
 * VERITAS Zenith Model Configuration v1.0
 * Perplexity-Based Detection with Entropy Analysis
 * 
 * Methodology:
 * This model uses entropy and burstiness as proxies for perplexity
 * (since we don't have access to GPT's logits).
 * 
 * Key Features:
 * - Paragraph uniformity detection (AI writes more uniform paragraphs)
 * - N-gram entropy analysis (AI has lower entropy = more predictable)
 * - Burstiness metrics (AI lacks natural "burstiness" in sentence lengths)
 * 
 * Training Statistics:
 * - Dataset: GPT-wiki-intro (19,912 samples)
 * - Test Accuracy: 99.57%
 * - ROC AUC: 0.9997
 * - Model Size: 5.8MB
 * 
 * Benchmark Results:
 * - Pure Human Detection: 100% (mean AI prob: 1.8%)
 * - Pure AI Detection: 100% (mean AI prob: 95.6%)
 * - Humanized AI Detection: 86.7%
 */

const VERITAS_ZENITH_CONFIG = {
    version: '1.0.0',
    modelName: 'Zenith',
    description: 'Perplexity-based entropy + burstiness detection',
    
    // Binary classification
    classes: ['Human', 'AI'],
    
    // Feature weights from training (percentages)
    featureWeights: {
        // === TOP FEATURES (>5%) ===
        paragraph_uniformity: 40.42,      // AI paragraphs are very uniform in length
        trigram_entropy: 14.05,           // AI has lower 3-word pattern entropy
        bigram_entropy: 9.66,             // AI has lower 2-word pattern entropy
        unigram_entropy: 5.71,            // AI vocabulary entropy
        
        // === MEDIUM IMPORTANCE (1-5%) ===
        vocabulary_entropy: 4.37,         // Overall vocabulary predictability
        sentence_start_diversity: 4.36,   // AI starts sentences similarly
        avg_sentence_length: 4.04,        // Sentence length patterns
        word_frequency_uniformity: 2.88,  // AI word frequency distribution
        period_rate: 1.90,                // Punctuation patterns
        trigram_repetition_rate: 1.49,    // 3-word phrase repetition
        trigram_predictability: 1.47,     // Perplexity proxy
        bigram_predictability: 1.10,      // Perplexity proxy
        
        // === LOWER IMPORTANCE (<1%) ===
        sentence_length_range_norm: 0.91,
        bigram_repetition_rate: 0.78,
        type_token_ratio: 0.70,
        sentence_length_cv: 0.69,
        word_repetition_rate: 0.55,
        yule_k: 0.54,
        word_predictability_score: 0.52,
        sentence_length_burstiness: 0.50,
        hapax_ratio: 0.37,
        comma_rate: 0.35,
        rare_word_ratio: 0.31,
        common_word_ratio: 0.31,
        short_word_ratio: 0.28,
        sentence_length_kurtosis: 0.22,
        punctuation_burstiness: 0.22,
        word_length_entropy: 0.21,
        sentence_length_skewness: 0.20,
        complexity_burstiness: 0.19,
        word_length_variance: 0.17,
        long_word_ratio: 0.17,
        word_length_burstiness: 0.16,
        avg_word_length: 0.16,
        semicolon_rate: 0.06,
        question_rate: 0.01,
    },
    
    // Feature explanations
    featureExplanations: {
        paragraph_uniformity: 'How uniform paragraph lengths are. AI produces very consistent paragraphs.',
        trigram_entropy: 'Entropy of 3-word patterns. Lower entropy = more predictable (AI-like).',
        bigram_entropy: 'Entropy of 2-word patterns. AI text has lower bigram entropy.',
        unigram_entropy: 'Vocabulary entropy. AI uses a more predictable vocabulary.',
        vocabulary_entropy: 'Overall vocabulary predictability measure.',
        sentence_start_diversity: 'Variety in sentence beginnings. AI often starts sentences similarly.',
        avg_sentence_length: 'Mean words per sentence. Affects pattern detection.',
        word_frequency_uniformity: 'How evenly distributed word frequencies are. AI is more uniform.',
        sentence_length_burstiness: 'Natural language has "bursty" patterns - AI is more uniform.',
        trigram_predictability: 'Proxy for perplexity - measures how predictable 3-word sequences are.',
        bigram_predictability: 'Proxy for perplexity - measures how predictable word pairs are.',
        word_predictability_score: 'Overall word predictability proxy.',
    },
    
    // Primary detection features (top 10)
    primaryFeatures: [
        "paragraph_uniformity",
        "trigram_entropy",
        "bigram_entropy",
        "unigram_entropy",
        "vocabulary_entropy",
        "sentence_start_diversity",
        "avg_sentence_length",
        "word_frequency_uniformity",
        "period_rate",
        "trigram_repetition_rate"
    ],
    
    // Scoring thresholds (calibrated from benchmark)
    scoring: {
        aiThreshold: 0.50,            // Above = AI (standard)
        humanThreshold: 0.50,         // Below = Human
        optimalThreshold: 0.10,       // From calibration - maximizes accuracy
        highConfidenceThreshold: 0.85,
        uncertainZone: [0.35, 0.65],
    },
    
    // Training verification
    trainingStats: {
        datasetsUsed: 1,
        datasetName: 'GPT-wiki-intro',
        totalSamples: 19912,
        humanSamples: 9969,
        aiSamples: 9943,
        testAccuracy: 0.9957,
        testF1: 0.9957,
        rocAuc: 0.9997,
        modelSize: '5.8MB',
    },
    
    // Humanization detection capability
    humanizationDetection: {
        // Sunset is better at detecting humanized text than Sunrise
        // Benchmark: 86.7% detection rate vs 66.7% for Sunrise
        capability: 'enhanced',
        detectionRate: 0.867,
        
        // Humanization signals
        signals: {
            entropyIncreaseWithStructure: 'Entropy increases but structure stays AI-like',
            burstinessIncrease: 'Artificial burstiness injected',
            uniformityPersists: 'Paragraph uniformity remains despite surface changes',
        }
    },
    
    // Expected ranges for anomaly detection
    expectedRanges: {
        human: {
            paragraph_uniformity: { min: 0.2, max: 0.7, typical: 0.45 },
            trigram_entropy: { min: 8, max: 14, typical: 11 },
            bigram_entropy: { min: 6, max: 11, typical: 8.5 },
            sentence_start_diversity: { min: 0.5, max: 1.0, typical: 0.75 },
        },
        ai: {
            paragraph_uniformity: { min: 0.7, max: 0.95, typical: 0.85 },
            trigram_entropy: { min: 5, max: 10, typical: 7 },
            bigram_entropy: { min: 4, max: 8, typical: 6 },
            sentence_start_diversity: { min: 0.3, max: 0.7, typical: 0.5 },
        }
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_ZENITH_CONFIG;
}
