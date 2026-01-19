/**
 * VERITAS Enhanced Model Configuration v4.0
 * 45-Feature Enhanced Detection with Tone, Hedging, and Rhetorical Analysis
 * 
 * Accuracy: 99.24% | F1: 99.24% | ROC-AUC: 99.98%
 * 
 * Top 10 Feature Contributions:
 * 1. paragraph_uniformity: 39.33%
 * 2. trigram_entropy: 15.39%
 * 3. bigram_entropy: 14.93%
 * 4. avg_paragraph_length: 6.67%
 * 5. avg_sentence_length: 5.08%
 * 6. lexical_diversity: 4.08%
 * 7. comma_rhythm_variance: 3.49%
 * 8. paragraph_length_variance: 2.78%
 * 9. sentence_length_variance: 2.66%
 * 10. sentence_complexity: 1.56%
 */

const VERITAS_ENHANCED_CONFIG = {
    // Model identification
    modelName: "Veritas Enhanced",
    version: "4.0",
    type: "enhanced",
    
    // Training statistics
    trainingStats: {
        trainedAt: "2025-06-20T12:00:00Z",
        totalSamples: 29976,
        testAccuracy: 0.9924,
        testF1: 0.9924,
        testPrecision: 0.9969,
        testRecall: 0.9879,
        rocAuc: 0.9998,
        crossValidation: {
            folds: 5,
            mean: 0.9924,
            std: 0.002
        }
    },
    
    // Feature weights (top 15)
    featureWeights: {
        paragraph_uniformity: 0.3933,
        trigram_entropy: 0.1539,
        bigram_entropy: 0.1493,
        avg_paragraph_length: 0.0667,
        avg_sentence_length: 0.0508,
        lexical_diversity: 0.0408,
        comma_rhythm_variance: 0.0349,
        paragraph_length_variance: 0.0278,
        sentence_length_variance: 0.0266,
        sentence_complexity: 0.0156,
        avg_word_length: 0.0118,
        function_word_ratio: 0.0103,
        hapax_legomena_ratio: 0.0095,
        topic_drift_score: 0.0089,
        lexical_chain_strength: 0.0074
    },
    
    // Feature categories
    featureCategories: {
        structural: [
            "avg_sentence_length",
            "sentence_length_variance",
            "avg_paragraph_length",
            "paragraph_length_variance",
            "paragraph_uniformity",
            "sentence_complexity"
        ],
        entropy: [
            "bigram_entropy",
            "trigram_entropy"
        ],
        lexical: [
            "lexical_diversity",
            "avg_word_length",
            "function_word_ratio",
            "hapax_legomena_ratio"
        ],
        tone: [
            "formality_score",
            "emotional_intensity",
            "sentiment_variance",
            "tone_consistency"
        ],
        hedging: [
            "hedge_word_density",
            "qualifier_density",
            "uncertainty_markers"
        ],
        personal_voice: [
            "first_person_singular",
            "first_person_plural",
            "second_person",
            "opinion_marker_density",
            "personal_anecdote_markers"
        ],
        rhetorical: [
            "question_density",
            "rhetorical_question_rate",
            "exclamation_rate",
            "transitional_phrase_density"
        ],
        coherence: [
            "topic_drift_score",
            "lexical_chain_strength",
            "paragraph_topic_consistency"
        ],
        punctuation: [
            "comma_rhythm_variance",
            "semicolon_usage",
            "colon_usage",
            "dash_em_usage",
            "ellipsis_rate",
            "parenthetical_rate"
        ],
        style: [
            "passive_voice_ratio",
            "nominalization_rate",
            "sentence_starter_variety",
            "clause_depth"
        ]
    },
    
    // Thresholds for classification
    thresholds: {
        aiConfident: 0.85,
        aiLikely: 0.65,
        mixed: 0.5,
        humanLikely: 0.35,
        humanConfident: 0.15
    },
    
    // Feature extraction functions
    features: {
        /**
         * Calculate formality score based on contractions, slang, and formal markers
         */
        formalityScore(text) {
            const words = text.toLowerCase().split(/\s+/);
            const contractions = ["don't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", 
                                  "haven't", "hasn't", "hadn't", "couldn't", "wouldn't", "shouldn't",
                                  "i'm", "you're", "we're", "they're", "it's", "that's", "what's"];
            const formalMarkers = ["furthermore", "moreover", "consequently", "nevertheless", "accordingly",
                                   "subsequently", "therefore", "thus", "hence", "hitherto"];
            
            const contractionCount = words.filter(w => contractions.includes(w)).length;
            const formalCount = words.filter(w => formalMarkers.includes(w)).length;
            
            const contractionRatio = contractionCount / Math.max(words.length, 1);
            const formalRatio = formalCount / Math.max(words.length, 1);
            
            return Math.min(1, Math.max(0, 0.5 + formalRatio * 10 - contractionRatio * 5));
        },
        
        /**
         * Calculate emotional intensity from exclamations and emotional words
         */
        emotionalIntensity(text) {
            const words = text.toLowerCase().split(/\s+/);
            const emotionalWords = ["amazing", "terrible", "fantastic", "horrible", "wonderful",
                                    "awful", "excellent", "dreadful", "brilliant", "disgusting",
                                    "love", "hate", "excited", "furious", "thrilled"];
            
            const exclamations = (text.match(/!/g) || []).length;
            const emotionalCount = words.filter(w => emotionalWords.includes(w)).length;
            
            return (emotionalCount + exclamations * 0.5) / Math.max(words.length, 1);
        },
        
        /**
         * Calculate hedge word density
         */
        hedgeWordDensity(text) {
            const words = text.toLowerCase().split(/\s+/);
            const hedgeWords = ["perhaps", "maybe", "possibly", "probably", "might", "could",
                               "seemingly", "apparently", "somewhat", "relatively", "fairly",
                               "rather", "quite", "sort of", "kind of", "to some extent"];
            
            const hedgeCount = words.filter(w => hedgeWords.includes(w)).length;
            return hedgeCount / Math.max(words.length, 1);
        },
        
        /**
         * Calculate first-person pronoun usage
         */
        firstPersonUsage(text) {
            const words = text.toLowerCase().split(/\s+/);
            const firstPersonSingular = ["i", "me", "my", "mine", "myself"];
            const firstPersonPlural = ["we", "us", "our", "ours", "ourselves"];
            
            const singularCount = words.filter(w => firstPersonSingular.includes(w)).length;
            const pluralCount = words.filter(w => firstPersonPlural.includes(w)).length;
            
            return {
                singular: singularCount / Math.max(words.length, 1),
                plural: pluralCount / Math.max(words.length, 1)
            };
        },
        
        /**
         * Calculate question density
         */
        questionDensity(text) {
            const sentences = text.split(/[.!?]+/).filter(s => s.trim());
            const questions = (text.match(/\?/g) || []).length;
            return questions / Math.max(sentences.length, 1);
        },
        
        /**
         * Calculate passive voice ratio
         */
        passiveVoiceRatio(text) {
            const passivePatterns = [
                /\b(is|are|was|were|be|been|being)\s+\w+ed\b/gi,
                /\b(is|are|was|were|be|been|being)\s+\w+en\b/gi
            ];
            
            const sentences = text.split(/[.!?]+/).filter(s => s.trim());
            let passiveCount = 0;
            
            for (const pattern of passivePatterns) {
                const matches = text.match(pattern) || [];
                passiveCount += matches.length;
            }
            
            return passiveCount / Math.max(sentences.length, 1);
        },
        
        /**
         * Calculate paragraph uniformity (key AI indicator)
         */
        paragraphUniformity(text) {
            const paragraphs = text.split(/\n\n+/).filter(p => p.trim());
            if (paragraphs.length < 2) return 0.5;
            
            const lengths = paragraphs.map(p => p.split(/\s+/).length);
            const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
            const variance = lengths.reduce((sum, len) => sum + Math.pow(len - mean, 2), 0) / lengths.length;
            const cv = Math.sqrt(variance) / Math.max(mean, 1);
            
            // Lower CV = more uniform = more likely AI
            return Math.max(0, Math.min(1, 1 - cv));
        },
        
        /**
         * Calculate n-gram entropy
         */
        ngramEntropy(text, n = 2) {
            const words = text.toLowerCase().split(/\s+/);
            if (words.length < n) return 0;
            
            const ngrams = {};
            for (let i = 0; i <= words.length - n; i++) {
                const ngram = words.slice(i, i + n).join(' ');
                ngrams[ngram] = (ngrams[ngram] || 0) + 1;
            }
            
            const total = Object.values(ngrams).reduce((a, b) => a + b, 0);
            let entropy = 0;
            
            for (const count of Object.values(ngrams)) {
                const p = count / total;
                if (p > 0) {
                    entropy -= p * Math.log2(p);
                }
            }
            
            // Normalize by maximum possible entropy
            const maxEntropy = Math.log2(total);
            return maxEntropy > 0 ? entropy / maxEntropy : 0;
        }
    },
    
    // Analysis interpretation
    interpretation: {
        getVerdict(score) {
            if (score >= 0.85) return { label: "AI Generated", confidence: "Very High", color: "#e74c3c" };
            if (score >= 0.65) return { label: "Likely AI", confidence: "High", color: "#e67e22" };
            if (score >= 0.50) return { label: "Mixed/Uncertain", confidence: "Medium", color: "#f39c12" };
            if (score >= 0.35) return { label: "Likely Human", confidence: "High", color: "#3498db" };
            return { label: "Human Written", confidence: "Very High", color: "#27ae60" };
        },
        
        getTopIndicators(features) {
            const indicators = [];
            
            if (features.paragraph_uniformity > 0.8) {
                indicators.push({ feature: "Paragraph Uniformity", impact: "Strong AI indicator" });
            }
            if (features.trigram_entropy < 0.3) {
                indicators.push({ feature: "Low Trigram Entropy", impact: "Strong AI indicator" });
            }
            if (features.sentence_length_variance < 0.2) {
                indicators.push({ feature: "Consistent Sentence Length", impact: "AI indicator" });
            }
            if (features.first_person_singular > 0.03) {
                indicators.push({ feature: "Personal Voice", impact: "Human indicator" });
            }
            if (features.hedge_word_density > 0.02) {
                indicators.push({ feature: "Hedging Language", impact: "Human indicator" });
            }
            
            return indicators;
        }
    }
};

// Export for Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VERITAS_ENHANCED_CONFIG;
}
