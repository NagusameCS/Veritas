/**
 * VERITAS — Analyzer Engine
 * Main analysis orchestrator combining all feature analyzers
 */

const AnalyzerEngine = {
    // All analyzers in order
    analyzers: [
        GrammarAnalyzer,      // Category 1
        SyntaxAnalyzer,       // Category 2
        LexicalAnalyzer,      // Category 3
        DialectAnalyzer,      // Category 4
        ArchaicAnalyzer,      // Category 5
        DiscourseAnalyzer,    // Category 6
        SemanticAnalyzer,     // Category 7
        StatisticalAnalyzer,  // Category 8
        AuthorshipAnalyzer,   // Category 9
        MetaPatternsAnalyzer  // Category 10
    ],

    /**
     * Run full analysis on text
     */
    analyze(text) {
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

        // Run all analyzers
        const categoryResults = [];
        for (const analyzer of this.analyzers) {
            try {
                const result = analyzer.analyze(text);
                categoryResults.push(result);
            } catch (error) {
                console.error(`Error in ${analyzer.name}:`, error);
                categoryResults.push({
                    name: analyzer.name,
                    category: analyzer.category,
                    aiProbability: 0.5,
                    confidence: 0,
                    error: error.message
                });
            }
        }

        // Calculate overall AI probability
        const overallResult = this.calculateOverallProbability(categoryResults);
        
        // Generate sentence-level scores for highlighting
        const sentenceScores = this.scoreSentences(text, sentences, categoryResults);
        
        // Compile all findings
        const allFindings = categoryResults
            .flatMap(r => r.findings || [])
            .sort((a, b) => {
                const order = { ai: 0, mixed: 1, human: 2, neutral: 3 };
                return (order[a.indicator] || 3) - (order[b.indicator] || 3);
            });

        const endTime = performance.now();

        return {
            // Overall results
            aiProbability: overallResult.aiProbability,
            humanProbability: 1 - overallResult.aiProbability,
            mixedProbability: overallResult.mixedProbability,
            confidence: overallResult.confidence,
            verdict: this.getVerdict(overallResult.aiProbability),
            
            // Per-category results
            categoryResults,
            
            // Sentence-level analysis
            sentences,
            sentenceScores,
            
            // All findings
            findings: allFindings,
            
            // Statistics
            stats,
            analysisTime: (endTime - startTime).toFixed(0) + 'ms'
        };
    },

    /**
     * Calculate overall AI probability from category results
     */
    calculateOverallProbability(categoryResults) {
        // Filter out categories with low confidence or errors
        const validResults = categoryResults.filter(r => 
            r.confidence > 0.2 && !r.error && r.aiProbability !== undefined
        );

        if (validResults.length === 0) {
            return { aiProbability: 0.5, confidence: 0, mixedProbability: 0 };
        }

        // Weight by both category weight and confidence
        let weightedSum = 0;
        let totalWeight = 0;

        for (const result of validResults) {
            const analyzer = this.analyzers.find(a => a.category === result.category);
            const weight = (analyzer?.weight || 1) * result.confidence;
            weightedSum += result.aiProbability * weight;
            totalWeight += weight;
        }

        const aiProbability = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        
        // Calculate overall confidence
        const avgConfidence = Utils.mean(validResults.map(r => r.confidence));
        
        // Calculate mixed probability (uncertainty)
        const probabilities = validResults.map(r => r.aiProbability);
        const variance = Utils.variance(probabilities);
        const mixedProbability = Math.min(0.3, Math.sqrt(variance));

        return {
            aiProbability: Math.max(0, Math.min(1, aiProbability)),
            confidence: avgConfidence,
            mixedProbability
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
     * Get verdict based on AI probability
     */
    getVerdict(aiProbability) {
        if (aiProbability < 0.25) {
            return {
                label: 'Likely Human',
                description: 'This text shows strong human-writing characteristics',
                level: 'human'
            };
        } else if (aiProbability < 0.4) {
            return {
                label: 'Possibly Human',
                description: 'This text appears mostly human with some uncertain elements',
                level: 'probably-human'
            };
        } else if (aiProbability < 0.6) {
            return {
                label: 'Uncertain',
                description: 'This text shows mixed signals - could be human or AI',
                level: 'mixed'
            };
        } else if (aiProbability < 0.75) {
            return {
                label: 'Possibly AI',
                description: 'This text shows several AI-typical patterns',
                level: 'probably-ai'
            };
        } else {
            return {
                label: 'Likely AI',
                description: 'This text exhibits strong AI-generated characteristics',
                level: 'ai'
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
