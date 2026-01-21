/**
 * SUPERNOVA Configuration for VERITAS
 * Neural-enhanced AI detection with 97%+ accuracy
 */

const VERITAS_SUPERNOVA_CONFIG = {
    name: 'Supernova',
    version: '1.0.0',
    type: 'neural-hybrid',
    
    // Model performance metrics
    metrics: {
        overallAccuracy: 0.9530,
        highConfidenceAccuracy: 0.9728,
        highConfidenceCoverage: 0.944,
        veryHighConfidenceAccuracy: 0.9813,
        veryHighConfidenceCoverage: 0.907,
        aucRoc: 0.9908,
        trainingSamples: 204028,
        testSamples: 36006
    },
    
    // Model architecture
    architecture: {
        classifier: 'XGBoost',
        nEstimators: 1000,
        maxDepth: 18,
        embeddingModel: 'all-MiniLM-L6-v2',
        embeddingDim: 384,
        heuristicFeatures: 31,
        totalFeatures: 415
    },
    
    // Confidence thresholds
    thresholds: {
        veryHigh: { min: 0.9, max: 0.1 },
        high: { min: 0.8, max: 0.2 },
        medium: { min: 0.7, max: 0.3 },
        low: { min: 0.4, max: 0.6 }
    },
    
    // Training data sources
    trainingSources: {
        human: [
            'OpenWebText', 'C4', 'WritingPrompts', 'IMDB', 'Yelp',
            'News', 'CC-News', 'Amazon-Reviews', 'StackExchange'
        ],
        ai: [
            'Anthropic-RLHF', 'Alpaca', 'OpenAssistant', 'GPT4All', 'UltraChat',
            'Dolly', 'WizardLM', 'ShareGPT', 'OpenOrca', 'WizardLM-V2'
        ]
    },
    
    // Per-source accuracy (from testing)
    sourceAccuracy: {
        'UltraChat': 0.993,
        'IMDB': 0.991,
        'StackExchange': 0.991,
        'WizardLM-V2': 0.991,
        'ShareGPT': 0.987,
        'WritingPrompts': 0.984,
        'Yelp': 0.982,
        'Anthropic-RLHF': 0.980,
        'WizardLM': 0.972,
        'OpenAssistant': 0.970,
        'Alpaca': 0.963,
        'OpenWebText': 0.961,
        'CC-News': 0.954,
        'OpenOrca': 0.946,
        'Amazon-Reviews': 0.937,
        'News': 0.936,
        'Dolly': 0.898,
        'GPT4All': 0.894,
        'C4': 0.848
    },
    
    // Feature importance (top features)
    featureImportance: {
        'third_he_she': 0.1012,
        'first_me': 0.0906,
        'answer_opener': 0.0302,
        'ellipsis_count': 0.0295,
        'instruction_phrases': 0.0209,
        'sent_count': 0.0208,
        'attribution': 0.0188,
        'numbered_items': 0.0178,
        'embedding_features': 0.3773
    },
    
    // Display configuration
    display: {
        icon: '🌟',
        color: '#FF6B35',
        gradientStart: '#FF6B35',
        gradientEnd: '#FFD700',
        description: 'Neural-enhanced detection with sentence embeddings',
        shortDescription: '97%+ accuracy with confidence thresholds',
        recommended: true
    },
    
    // Browser compatibility
    requirements: {
        transformersJs: '2.17.1',
        onnxRuntimeWeb: '1.17.0',
        minBrowserVersion: {
            chrome: 90,
            firefox: 85,
            safari: 15,
            edge: 90
        },
        estimatedModelDownload: '95MB',
        estimatedMemoryUsage: '500MB',
        estimatedInferenceTime: '1-3s'
    }
};

// Export
if (typeof window !== 'undefined') {
    window.VERITAS_SUPERNOVA_CONFIG = VERITAS_SUPERNOVA_CONFIG;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_SUPERNOVA_CONFIG;
}
