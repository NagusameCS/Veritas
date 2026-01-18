/**
 * VERITAS — Analyzer Engine v2.0
 * Variance-Based Detection: Measures deviations from expected human variance
 * 
 * Philosophy: "Never grade features as 'AI-like' or 'human-like' in isolation.
 * Grade them as deviations from expected human variance."
 */

const AnalyzerEngine = {
    // Category weights based on detection importance
    categoryWeights: {
        syntaxVariance: 0.20,      // Syntax variance
        lexicalDiversity: 0.15,    // Lexical diversity  
        repetitionUniformity: 0.15, // Repetition uniformity
        toneStability: 0.15,       // Tone stability
        grammarEntropy: 0.10,      // Grammar entropy
        perplexity: 0.10,          // Statistical perplexity
        authorshipDrift: 0.15      // Authorship drift
    },

    // Getter for analyzers - allows graceful handling of missing modules
    get analyzers() {
        const all = [];
        
        // Core analyzers (Categories 1-10)
        if (typeof GrammarAnalyzer !== 'undefined') all.push(GrammarAnalyzer);
        if (typeof SyntaxAnalyzer !== 'undefined') all.push(SyntaxAnalyzer);
        if (typeof LexicalAnalyzer !== 'undefined') all.push(LexicalAnalyzer);
        if (typeof DialectAnalyzer !== 'undefined') all.push(DialectAnalyzer);
        if (typeof ArchaicAnalyzer !== 'undefined') all.push(ArchaicAnalyzer);
        if (typeof DiscourseAnalyzer !== 'undefined') all.push(DiscourseAnalyzer);
        if (typeof SemanticAnalyzer !== 'undefined') all.push(SemanticAnalyzer);
        if (typeof StatisticalAnalyzer !== 'undefined') all.push(StatisticalAnalyzer);
        if (typeof AuthorshipAnalyzer !== 'undefined') all.push(AuthorshipAnalyzer);
        if (typeof MetaPatternsAnalyzer !== 'undefined') all.push(MetaPatternsAnalyzer);
        
        // New variance-based analyzers (Categories 11-14)
        if (typeof MetadataAnalyzer !== 'undefined') all.push(MetadataAnalyzer);
        if (typeof RepetitionAnalyzer !== 'undefined') all.push(RepetitionAnalyzer);
        if (typeof ToneAnalyzer !== 'undefined') all.push(ToneAnalyzer);
        if (typeof PartOfSpeechAnalyzer !== 'undefined') all.push(PartOfSpeechAnalyzer);
        
        return all;
    },

    // Fallback for missing analyzers (graceful degradation)
    safeGetAnalyzer(analyzerRef) {
        try {
            return typeof analyzerRef !== 'undefined' ? analyzerRef : null;
        } catch {
            return null;
        }
    },

    /**
     * Run full analysis on text
     * Now uses variance-based detection philosophy
     */
    analyze(text, metadata = null) {
        const startTime = performance.now();
        
        if (!text || text.trim().length === 0) {
            return this.getEmptyResult();
        }

        // Precompute common values
        const sentences = Utils.splitSentences(text);
        const tokens = Utils.tokenize(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);

        // Basic stats
        const stats = {
            characters: text.length,
            words: tokens.length,
            sentences: sentences.length,
            paragraphs: paragraphs.length,
            avgWordsPerSentence: sentences.length > 0 ? (tokens.length / sentences.length).toFixed(1) : 0
        };

        // Filter to only available analyzers
        const availableAnalyzers = this.analyzers.filter(a => this.safeGetAnalyzer(a));

        // Run all analyzers
        const categoryResults = [];
        for (const analyzer of availableAnalyzers) {
            if (!analyzer) continue;
            try {
                const result = analyzer.analyze(text, metadata);
                // Add variance scores if analyzer supports it
                if (result.uniformityScores) {
                    result.varianceAnalysis = this.computeVarianceMetrics(result.uniformityScores);
                }
                categoryResults.push(result);
            } catch (error) {
                console.error(`Error in ${analyzer?.name || 'unknown'}:`, error);
                categoryResults.push({
                    name: analyzer?.name || 'Unknown',
                    category: analyzer?.category || 0,
                    aiProbability: 0.5,
                    confidence: 0,
                    error: error.message
                });
            }
        }

        // Calculate overall AI probability using variance-based scoring
        const overallResult = this.calculateVarianceBasedProbability(categoryResults);
        
        // Generate sentence-level scores for highlighting
        const sentenceScores = this.scoreSentences(text, sentences, categoryResults);
        
        // Compile all findings
        const allFindings = categoryResults
            .flatMap(r => r.findings || [])
            .sort((a, b) => {
                const order = { ai: 0, mixed: 1, human: 2, neutral: 3 };
                return (order[a.indicator] || 3) - (order[b.indicator] || 3);
            });

        // Compute confidence interval
        const confidenceInterval = this.computeConfidenceInterval(
            overallResult.aiProbability,
            categoryResults.length,
            overallResult.confidence
        );

        // Assess false positive risk
        const falsePositiveRisk = this.assessFalsePositiveRisk(categoryResults, text);

        const endTime = performance.now();

        return {
            // Overall results
            aiProbability: overallResult.aiProbability,
            humanProbability: 1 - overallResult.aiProbability,
            mixedProbability: overallResult.mixedProbability,
            confidence: overallResult.confidence,
            confidenceInterval,
            falsePositiveRisk,
            verdict: this.getVerdict(overallResult.aiProbability, overallResult.confidence),
            
            // Variance-specific data
            varianceProfile: overallResult.varianceProfile,
            featureContributions: overallResult.featureContributions,
            
            // Per-category results
            categoryResults,
            
            // Sentence-level analysis
            sentences,
            sentenceScores,
            
            // All findings
            findings: allFindings,
            
            // Statistics
            stats,
            metadata: metadata || null,
            analysisTime: (endTime - startTime).toFixed(0) + 'ms'
        };
    },

    /**
     * Compute variance metrics from uniformity scores
     */
    computeVarianceMetrics(uniformityScores) {
        const values = Object.values(uniformityScores).filter(v => typeof v === 'number');
        if (values.length === 0) return null;

        return {
            meanUniformity: Utils.mean(values),
            uniformityVariance: Utils.variance(values),
            maxUniformity: Math.max(...values),
            minUniformity: Math.min(...values),
            uniformityRange: Math.max(...values) - Math.min(...values)
        };
    },

    /**
     * Calculate overall AI probability using variance-based methodology
     * Key principle: High uniformity = AI, High variance = Human
     */
    calculateVarianceBasedProbability(categoryResults) {
        const validResults = categoryResults.filter(r => 
            r.confidence > 0.2 && !r.error && r.aiProbability !== undefined
        );

        if (validResults.length === 0) {
            return { 
                aiProbability: 0.5, 
                confidence: 0, 
                mixedProbability: 0,
                varianceProfile: null,
                featureContributions: []
            };
        }

        // Calculate feature contributions using category weights
        const featureContributions = [];
        let weightedSum = 0;
        let totalWeight = 0;

        for (const result of validResults) {
            // Map categories to weight keys
            const weightKey = this.getCategoryWeightKey(result.category, result.name);
            const baseWeight = this.categoryWeights[weightKey] || 0.1;
            
            // Apply confidence as a weight multiplier
            const effectiveWeight = baseWeight * result.confidence;
            
            // Calculate contribution
            const contribution = result.aiProbability * effectiveWeight;
            
            featureContributions.push({
                category: result.category,
                name: result.name,
                aiProbability: result.aiProbability,
                weight: effectiveWeight,
                contribution,
                uniformityScores: result.uniformityScores || null
            });

            weightedSum += contribution;
            totalWeight += effectiveWeight;
        }

        const aiProbability = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        
        // Calculate confidence based on result agreement
        const probabilities = validResults.map(r => r.aiProbability);
        const agreement = 1 - Utils.standardDeviation(probabilities);
        const avgConfidence = Utils.mean(validResults.map(r => r.confidence));
        const overallConfidence = (agreement * 0.4 + avgConfidence * 0.6);
        
        // Mixed probability increases with disagreement
        const mixedProbability = Math.min(0.3, Utils.standardDeviation(probabilities));

        // Build variance profile
        const varianceProfile = this.buildVarianceProfile(validResults);

        return {
            aiProbability: Math.max(0, Math.min(1, aiProbability)),
            confidence: overallConfidence,
            mixedProbability,
            varianceProfile,
            featureContributions: featureContributions.sort((a, b) => b.contribution - a.contribution)
        };
    },

    /**
     * Map category number/name to weight key
     */
    getCategoryWeightKey(category, name) {
        const nameLower = (name || '').toLowerCase();
        
        if (nameLower.includes('syntax') || category === 2) return 'syntaxVariance';
        if (nameLower.includes('lexical') || category === 3) return 'lexicalDiversity';
        if (nameLower.includes('repetition') || category === 12) return 'repetitionUniformity';
        if (nameLower.includes('tone') || category === 13) return 'toneStability';
        if (nameLower.includes('grammar') || category === 1) return 'grammarEntropy';
        if (nameLower.includes('statistical') || nameLower.includes('perplexity') || category === 8) return 'perplexity';
        if (nameLower.includes('authorship') || category === 9) return 'authorshipDrift';
        
        return 'grammarEntropy'; // default
    },

    /**
     * Build variance profile summarizing key metrics
     */
    buildVarianceProfile(results) {
        const profile = {
            sentenceLengthVariance: null,
            vocabularyDiversity: null,
            toneStability: null,
            repetitionUniformity: null,
            overallUniformity: null
        };

        for (const result of results) {
            if (result.uniformityScores) {
                if (result.uniformityScores.sentenceLength !== undefined) {
                    profile.sentenceLengthVariance = result.uniformityScores.sentenceLength;
                }
                if (result.uniformityScores.vocabulary !== undefined) {
                    profile.vocabularyDiversity = 1 - result.uniformityScores.vocabulary;
                }
                if (result.uniformityScores.overall !== undefined) {
                    profile.overallUniformity = result.uniformityScores.overall;
                }
            }
            
            if (result.stability?.overall !== undefined) {
                profile.toneStability = result.stability.overall;
            }
        }

        // Calculate overall uniformity if not set
        const uniformityValues = Object.values(profile).filter(v => v !== null);
        if (profile.overallUniformity === null && uniformityValues.length > 0) {
            profile.overallUniformity = Utils.mean(uniformityValues);
        }

        return profile;
    },

    /**
     * Compute confidence interval for the AI probability estimate
     */
    computeConfidenceInterval(probability, sampleSize, confidence) {
        // Use Wilson score interval approximation
        const z = 1.96; // 95% confidence
        const n = Math.max(sampleSize, 1);
        const p = probability;
        
        const denominator = 1 + z * z / n;
        const center = (p + z * z / (2 * n)) / denominator;
        const margin = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denominator;
        
        // Adjust based on confidence level
        const adjustedMargin = margin * (2 - confidence);
        
        return {
            lower: Math.max(0, center - adjustedMargin),
            upper: Math.min(1, center + adjustedMargin),
            center,
            margin: adjustedMargin
        };
    },

    /**
     * Assess false positive risk based on edge cases
     */
    assessFalsePositiveRisk(categoryResults, text) {
        const risks = [];
        
        // Check for academic/formal writing (often mistaken for AI)
        const formalIndicators = (text.match(/\b(furthermore|moreover|consequently|thus|hence|therefore)\b/gi) || []).length;
        if (formalIndicators > 3) {
            risks.push({
                type: 'formal_writing',
                message: 'Formal/academic writing style may increase false positive rate',
                severity: 'medium'
            });
        }

        // Check for short text (less reliable)
        const wordCount = Utils.tokenize(text).length;
        if (wordCount < 100) {
            risks.push({
                type: 'short_text',
                message: 'Short texts have lower detection accuracy',
                severity: 'high'
            });
        }

        // Check for technical/specialized content
        const technicalTerms = (text.match(/\b[A-Z]{2,}[a-z]*\b/g) || []).length;
        if (technicalTerms > 5) {
            risks.push({
                type: 'technical_content',
                message: 'Technical jargon may affect detection accuracy',
                severity: 'low'
            });
        }

        // Check for disagreement between analyzers
        const validResults = categoryResults.filter(r => r.confidence > 0.3);
        if (validResults.length > 3) {
            const probs = validResults.map(r => r.aiProbability);
            const spread = Math.max(...probs) - Math.min(...probs);
            if (spread > 0.4) {
                risks.push({
                    type: 'analyzer_disagreement',
                    message: 'High disagreement between detection methods',
                    severity: 'medium'
                });
            }
        }

        return {
            hasRisks: risks.length > 0,
            risks,
            overallRisk: risks.length === 0 ? 'low' : 
                         risks.some(r => r.severity === 'high') ? 'high' : 'medium'
        };
    },

    /**
     * Score individual sentences for highlighting
     */
    scoreSentences(text, sentences, categoryResults) {
        const scores = [];
        
        // Get statistical sentence scores if available
        const statResult = categoryResults.find(r => r.category === 8);
        const perplexityScores = statResult?.sentenceScores || [];

        for (let i = 0; i < sentences.length; i++) {
            const sentence = sentences[i];
            const tokens = Utils.tokenize(sentence);
            
            // Calculate multiple indicators for this sentence
            const indicators = {};
            
            // 1. Length regularity (compare to neighbors)
            if (i > 0 && i < sentences.length - 1) {
                const prevLen = Utils.tokenize(sentences[i-1]).length;
                const nextLen = Utils.tokenize(sentences[i+1]).length;
                const avgNeighbor = (prevLen + nextLen) / 2;
                const lengthDiff = Math.abs(tokens.length - avgNeighbor) / avgNeighbor;
                indicators.lengthRegularity = 1 - Math.min(1, lengthDiff);
            } else {
                indicators.lengthRegularity = 0.5;
            }

            // 2. Lexical diversity
            const uniqueRatio = tokens.length > 0 
                ? new Set(tokens).size / tokens.length 
                : 0.5;
            indicators.lexicalDiversity = uniqueRatio;

            // 3. Discourse markers
            const discourseMarkers = Utils.discourseMarkers.filter(m => 
                sentence.toLowerCase().includes(m.toLowerCase())
            );
            indicators.discourseMarkerDensity = Math.min(1, discourseMarkers.length * 0.3);

            // 4. Perplexity proxy (from statistical analyzer)
            if (perplexityScores[i]) {
                indicators.perplexity = perplexityScores[i].score / 100;
            }

            // 5. Personal pronouns (human indicator)
            const personalPronouns = (sentence.match(/\b(I|my|me|we|our|us)\b/gi) || []).length;
            indicators.personalLanguage = Math.min(1, personalPronouns * 0.2);

            // 6. Hedging language
            const hedgingWords = ['perhaps', 'maybe', 'possibly', 'might', 'could', 'may', 'seems', 'appears'];
            const hedgingCount = hedgingWords.filter(h => 
                sentence.toLowerCase().includes(h)
            ).length;
            indicators.hedging = Math.min(1, hedgingCount * 0.25);

            // Calculate sentence AI probability
            // Weight different indicators
            const sentenceAiProb = (
                (1 - indicators.lengthRegularity) * 0.15 + // High regularity = AI
                (1 - indicators.lexicalDiversity) * 0.15 + // Low diversity = AI
                indicators.discourseMarkerDensity * 0.25 + // Many markers = AI
                (1 - (indicators.perplexity || 0.5)) * 0.15 + // Low perplexity = AI
                (1 - indicators.personalLanguage) * 0.15 + // No personal = AI
                indicators.hedging * 0.15 // Hedging = AI
            );

            // Classify sentence
            let classification;
            if (sentenceAiProb < 0.35) {
                classification = 'human';
            } else if (sentenceAiProb > 0.6) {
                classification = 'ai';
            } else {
                classification = 'mixed';
            }

            scores.push({
                index: i,
                text: sentence,
                aiProbability: sentenceAiProb,
                classification,
                indicators
            });
        }

        return scores;
    },

    /**
     * Get verdict based on AI probability and confidence
     * Now includes probability bands and confidence qualifiers
     */
    getVerdict(aiProbability, confidence = 0.5) {
        // Confidence qualifier
        const confidenceLevel = confidence < 0.4 ? 'Low confidence: ' : 
                               confidence < 0.7 ? '' : 
                               'High confidence: ';

        // Probability bands with more nuanced thresholds
        if (aiProbability < 0.15) {
            return {
                label: confidenceLevel + 'Human Written',
                description: 'This text shows strong, consistent human-writing characteristics',
                level: 'human',
                band: 'definitely-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.30) {
            return {
                label: confidenceLevel + 'Likely Human',
                description: 'This text exhibits predominantly human patterns with minimal AI signals',
                level: 'probably-human',
                band: 'likely-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.45) {
            return {
                label: confidenceLevel + 'Possibly Human',
                description: 'This text appears mostly human but has some uncertain elements',
                level: 'leaning-human',
                band: 'possibly-human',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.55) {
            return {
                label: 'Inconclusive',
                description: 'This text shows mixed signals — could be human, AI, or a mixture',
                level: 'mixed',
                band: 'inconclusive',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.70) {
            return {
                label: confidenceLevel + 'Possibly AI',
                description: 'This text has notable AI-typical patterns but some human elements',
                level: 'leaning-ai',
                band: 'possibly-ai',
                probability: aiProbability,
                confidence
            };
        } else if (aiProbability < 0.85) {
            return {
                label: confidenceLevel + 'Likely AI',
                description: 'This text exhibits strong AI-generated characteristics',
                level: 'probably-ai',
                band: 'likely-ai',
                probability: aiProbability,
                confidence
            };
        } else {
            return {
                label: confidenceLevel + 'AI Generated',
                description: 'This text shows overwhelming AI-generated patterns',
                level: 'ai',
                band: 'definitely-ai',
                probability: aiProbability,
                confidence
            };
        }
    },

    /**
     * Get feature analysis summary for feature tab
     */
    getFeatureSummary(categoryResults) {
        return categoryResults.map(result => ({
            category: result.category,
            name: result.name,
            aiProbability: result.aiProbability,
            confidence: result.confidence,
            topFindings: (result.findings || []).slice(0, 3),
            scores: result.scores
        }));
    },

    /**
     * Generate detailed report
     */
    generateReport(analysisResult) {
        const sections = [];

        for (const category of analysisResult.categoryResults) {
            sections.push({
                number: category.category,
                name: category.name,
                aiScore: Math.round(category.aiProbability * 100),
                confidence: Math.round(category.confidence * 100),
                findings: category.findings || [],
                details: category.details || {}
            });
        }

        return {
            overall: {
                aiProbability: Math.round(analysisResult.aiProbability * 100),
                verdict: analysisResult.verdict,
                confidence: Math.round(analysisResult.confidence * 100)
            },
            sections,
            stats: analysisResult.stats,
            timestamp: new Date().toISOString()
        };
    },

    /**
     * Get empty result for empty input
     */
    getEmptyResult() {
        return {
            aiProbability: 0.5,
            humanProbability: 0.5,
            mixedProbability: 0,
            confidence: 0,
            verdict: {
                label: 'No Input',
                description: 'Please enter text to analyze',
                level: 'none'
            },
            categoryResults: [],
            sentences: [],
            sentenceScores: [],
            findings: [],
            stats: {
                characters: 0,
                words: 0,
                sentences: 0,
                paragraphs: 0
            },
            analysisTime: '0ms'
        };
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AnalyzerEngine;
}
