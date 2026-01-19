/**
 * VERITAS Sunrise Model Configuration v4.0
 * Enhanced 2-class detection with Humanization Indicator
 * 
 * Philosophy:
 * - Primary Classification: Human vs AI (98.08% accuracy)
 * - Secondary Analysis: Humanization probability (advisory, not definitive)
 * 
 * The 3-class problem (Human/AI/Humanized) is fundamentally harder because
 * good humanization makes AI text indistinguishable from human text.
 * Instead, we provide a PROBABILITY that detected AI may have been humanized.
 * 
 * Training Statistics:
 * - Datasets: 3
 * - Total Samples: 29,976
 * - Test Accuracy: 0.9808
 * - Test F1 Score: 0.9809
 * - ROC AUC: 0.9980
 */

const VERITAS_SUNRISE_CONFIG = {
    version: '4.0.0',
    modelName: 'Sunrise',
    description: 'Primary 2-class detection with humanization advisory',
    
    // Primary classification: Human (0) or AI (1)
    // Humanization is detected as a SECONDARY indicator
    classes: ['Human', 'AI'],
    
    // ML-derived feature weights (percentages, sum to 100)
    // These come from Random Forest feature importance on 29,976 samples
    featureWeights: {
        // === HIGH IMPORTANCE (>5%) ===
        avg_paragraph_length: 16.85,    // AI tends to have uniform paragraphs
        paragraph_count: 14.66,         // AI often produces predictable structure
        hapax_count: 11.75,             // Words used only once (humans use more)
        unique_word_count: 10.21,       // Vocabulary diversity
        paragraph_length_cv: 7.36,      // Coefficient of variation (humans vary more)
        sentence_length_range: 7.02,    // Max - Min sentence length
        word_count: 6.48,               // Document length affects patterns
        sentence_length_max: 5.2,       // Longest sentence length
        
        // === MEDIUM IMPORTANCE (1-5%) ===
        sentence_length_std: 3.46,      // Standard deviation of sentence lengths
        avg_sentence_length: 2.18,      // Mean words per sentence
        sentence_count: 1.86,           // Number of sentences
        burstiness_sentence: 1.19,      // Sentence length clustering
        syllable_ratio: 1.07,           // Polysyllabic word frequency
        
        // === LOWER IMPORTANCE (<1%) ===
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
    
    // Feature explanations for transparency
    featureExplanations: {
        avg_paragraph_length: 'Average number of words per paragraph. AI tends to produce more uniform paragraph lengths.',
        paragraph_count: 'Total paragraphs. AI often follows predictable structure patterns.',
        hapax_count: 'Words appearing exactly once. Humans typically use more unique vocabulary.',
        unique_word_count: 'Total distinct words. Higher diversity suggests human authorship.',
        paragraph_length_cv: 'Variability in paragraph lengths. Humans show more natural variation.',
        sentence_length_range: 'Difference between longest and shortest sentences. Humans vary more.',
        word_count: 'Total words. Length affects statistical reliability.',
        sentence_length_max: 'Longest sentence. Humans often produce more extreme outliers.',
        sentence_length_std: 'Standard deviation of sentence lengths. Measures consistency.',
        avg_sentence_length: 'Mean words per sentence. Affects readability patterns.',
        burstiness_sentence: 'How clustered sentence lengths are. AI tends to be more uniform.',
        syllable_ratio: 'Complex word frequency. AI often uses measured complexity.',
        type_token_ratio: 'Vocabulary diversity relative to length.',
        trigram_repetition_rate: 'Repeated 3-word phrases. AI may reuse patterns.',
        zipf_slope: 'Word frequency distribution. Natural language follows Zipf\'s law.',
    },
    
    // Primary detection features (top 10 by importance)
    primaryFeatures: [
        "avg_paragraph_length", 
        "paragraph_count", 
        "hapax_count", 
        "unique_word_count", 
        "paragraph_length_cv", 
        "sentence_length_range", 
        "word_count", 
        "sentence_length_max", 
        "sentence_length_std", 
        "avg_sentence_length"
    ],
    
    // Scoring thresholds
    scoring: {
        aiThreshold: 0.65,           // Above this = AI
        humanThreshold: 0.35,        // Below this = Human
        highConfidenceThreshold: 0.85,
        uncertainZone: [0.35, 0.65], // Gray zone
    },
    
    // === HUMANIZATION DETECTION (Advisory Only) ===
    // These patterns suggest AI text may have been modified by humanizer tools
    humanizationIndicators: {
        // Contradiction patterns: Some features say AI, others say Human
        contradictionThreshold: 0.3, // 30% disagreement triggers suspicion
        
        // Specific patterns that suggest humanization
        patterns: {
            // Artificially injected variance
            artificialVariance: {
                description: 'Sentence lengths vary TOO randomly (unnatural pattern)',
                check: 'sentence_length_cv > 0.8 AND kurtosis < -1',
                weight: 0.15
            },
            // Mixed formality
            mixedFormality: {
                description: 'Mix of formal and informal language (inconsistent)',
                check: 'formal_informal_ratio between 0.4 and 0.6',
                weight: 0.12
            },
            // Random punctuation changes
            punctuationAnomaly: {
                description: 'Unusual punctuation patterns',
                check: 'semicolon/comma ratio anomaly',
                weight: 0.08
            },
            // Vocabulary substitution artifacts
            vocabularyGaps: {
                description: 'Words that don\'t fit the register',
                check: 'word_sophistication_variance > expected',
                weight: 0.1
            },
            // Structure preserved, surface changed
            structureVsSurface: {
                description: 'AI structure with human-like surface features',
                check: 'paragraph_uniformity high BUT lexical_diversity high',
                weight: 0.2
            },
            // High analyzer disagreement
            analyzerDisagreement: {
                description: 'Statistical and stylistic analyzers disagree',
                check: 'max_analyzer_difference > 0.4',
                weight: 0.25
            },
            // Transition word manipulation
            transitionPatterns: {
                description: 'Transition words feel forced or over-used',
                check: 'transition_density anomaly',
                weight: 0.1
            }
        },
        
        // Thresholds for humanization advisory
        thresholds: {
            possible: 0.3,    // "May be humanized"
            likely: 0.5,      // "Likely humanized" 
            confident: 0.7    // "Strong humanization signals"
        }
    },
    
    // Training verification
    trainingStats: {
        datasetsUsed: 3,
        totalSamples: 29976,
        testAccuracy: 0.9808,
        testF1: 0.9809,
        rocAuc: 0.9980,
        dataHash: 'ba2084ca3001c09c',
        modelHash: '511efcb915e4f79f',
    },
    
    // Expected value ranges for features (for anomaly detection)
    expectedRanges: {
        human: {
            avg_paragraph_length: { min: 20, max: 200, typical: 75 },
            paragraph_length_cv: { min: 0.3, max: 0.9, typical: 0.5 },
            sentence_length_range: { min: 10, max: 50, typical: 25 },
            hapax_ratio: { min: 0.3, max: 0.7, typical: 0.5 },
            type_token_ratio: { min: 0.4, max: 0.8, typical: 0.6 },
        },
        ai: {
            avg_paragraph_length: { min: 50, max: 150, typical: 90 },
            paragraph_length_cv: { min: 0.1, max: 0.4, typical: 0.25 },
            sentence_length_range: { min: 5, max: 20, typical: 12 },
            hapax_ratio: { min: 0.2, max: 0.5, typical: 0.35 },
            type_token_ratio: { min: 0.3, max: 0.6, typical: 0.45 },
        }
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_SUNRISE_CONFIG;
}
