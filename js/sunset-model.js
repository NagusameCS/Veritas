/**
 * VERITAS Sunset Model Configuration
 * Legacy 2-class classifier (Human vs AI) with high accuracy
 * 
 * This is the proven, stable detection model.
 * Use for definitive Human/AI classification.
 * 
 * Accuracy: 98.08% on 29,976 samples
 */

const VERITAS_SUNSET_CONFIG = {
    version: '2.0.0',
    modelName: 'Sunset',
    description: 'Binary classifier: Human vs AI (high accuracy)',
    
    // This model does NOT detect humanized text
    // It classifies: Human (0) or AI (1)
    classes: ['Human', 'AI'],
    
    // Feature weights from Sunrise v3.0 training
    featureWeights: {
        avg_paragraph_length: 16.85,
        paragraph_count: 14.66,
        hapax_count: 11.75,
        unique_word_count: 10.21,
        paragraph_length_cv: 7.36,
        sentence_length_range: 7.02,
        word_count: 6.48,
        sentence_length_max: 5.2,
        sentence_length_std: 3.46,
        avg_sentence_length: 2.18,
        sentence_count: 1.86,
        burstiness_sentence: 1.19,
        syllable_ratio: 1.07,
        question_rate: 0.86,
        sentence_similarity_avg: 0.83,
        automated_readability_index: 0.76,
        sentence_length_cv: 0.7,
        flesch_kincaid_grade: 0.69,
        trigram_repetition_rate: 0.67,
        zipf_slope: 0.65,
        overall_uniformity: 0.63,
        complexity_cv: 0.61,
        bigram_repetition_rate: 0.58,
        exclamation_rate: 0.56,
        type_token_ratio: 0.47,
        avg_word_length: 0.39,
        sentence_length_kurtosis: 0.34,
        comma_rate: 0.28,
        hapax_ratio: 0.26,
        sentence_length_min: 0.26,
        zipf_r_squared: 0.25,
        zipf_residual_std: 0.21,
        sentence_length_skewness: 0.18,
        burstiness_word_length: 0.18,
        dis_legomena_ratio: 0.17,
        word_length_cv: 0.15,
        semicolon_rate: 0.04,
    },
    
    // Scoring thresholds
    scoring: {
        aiThreshold: 0.65,      // Above this = AI
        humanThreshold: 0.35,   // Below this = Human
        highConfidence: 0.85,   // Very confident
        lowConfidence: 0.15,    // Uncertain
    },
    
    // Training stats for transparency
    trainingStats: {
        datasetsUsed: 3,
        totalSamples: 29976,
        testAccuracy: 0.9808,
        testF1: 0.9809,
        rocAuc: 0.9980,
    },
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_SUNSET_CONFIG;
}
