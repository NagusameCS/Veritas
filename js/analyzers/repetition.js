/**
 * VERITAS — Enhanced Repetition Analyzer
 * Variance-based repetition detection using distance, clustering, and semantic patterns
 * 
 * KEY PRINCIPLE: We measure HOW repetition is distributed, not just that it exists.
 * AI shows uniform repetition patterns; humans show clustered/bursty patterns.
 */

const RepetitionAnalyzer = {
    name: 'Repetition Patterns',
    category: 12,
    weight: 1.5,

    /**
     * Main analysis function
     */
    analyze(text) {
        if (!text || text.length < 100) {
            return this.getEmptyResult();
        }

        const tokens = Utils.tokenize(text);
        const sentences = Utils.splitSentences(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);

        // Lexical repetition analysis
        const lexicalAnalysis = this.analyzeLexicalRepetition(tokens, text);
        
        // Semantic repetition (paraphrase detection)
        const semanticAnalysis = this.analyzeSemanticRepetition(sentences, paragraphs);
        
        // Structural repetition (template detection)
        const structuralAnalysis = this.analyzeStructuralRepetition(sentences, paragraphs);

        // Calculate scores based on uniformity of repetition
        const scores = {
            lexicalUniformity: lexicalAnalysis.uniformityScore,
            repetitionClustering: lexicalAnalysis.clusteringScore,
            semanticRedundancy: semanticAnalysis.redundancyScore,
            structuralSimilarity: structuralAnalysis.similarityScore
        };

        // AI probability based on uniform repetition patterns
        const aiProbability = this.calculateAIProbability(scores, lexicalAnalysis, semanticAnalysis);
        
        const confidence = this.calculateConfidence(tokens.length, sentences.length);
        const findings = this.generateFindings(lexicalAnalysis, semanticAnalysis, structuralAnalysis);

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence,
            details: {
                lexical: lexicalAnalysis,
                semantic: semanticAnalysis,
                structural: structuralAnalysis
            },
            findings,
            scores
        };
    },

    /**
     * Analyze lexical repetition with distance and clustering metrics
     */
    analyzeLexicalRepetition(tokens, text) {
        const analysis = {
            ngramReuse: {},
            repetitionDistances: [],
            uniformityScore: 0.5,
            clusteringScore: 0.5,
            topRepeatedPhrases: []
        };

        // Analyze n-grams from 2 to 5
        for (let n = 2; n <= 5; n++) {
            const ngrams = Utils.ngrams(tokens, n);
            const ngramPositions = {};
            
            // Track positions of each n-gram
            // Note: Utils.ngrams returns already-joined strings
            ngrams.forEach((gram, index) => {
                const key = gram; // gram is already a string
                if (!ngramPositions[key]) {
                    ngramPositions[key] = [];
                }
                ngramPositions[key].push(index);
            });

            // Analyze repeated n-grams
            const repeated = {};
            for (const [key, positions] of Object.entries(ngramPositions)) {
                if (positions.length > 1) {
                    // Calculate distances between occurrences
                    const distances = [];
                    for (let i = 1; i < positions.length; i++) {
                        distances.push(positions[i] - positions[i - 1]);
                    }
                    
                    repeated[key] = {
                        count: positions.length,
                        positions,
                        distances,
                        avgDistance: Utils.mean(distances),
                        distanceVariance: Utils.variance(distances)
                    };
                }
            }

            analysis.ngramReuse[n] = {
                totalNgrams: ngrams.length,
                uniqueNgrams: Object.keys(ngramPositions).length,
                repeatedCount: Object.keys(repeated).length,
                reuseRate: Object.keys(ngramPositions).length > 0 
                    ? Object.keys(repeated).length / Object.keys(ngramPositions).length 
                    : 0,
                repeated
            };
        }

        // Calculate overall repetition statistics
        const allDistances = [];
        for (let n = 2; n <= 4; n++) {
            for (const phrase of Object.values(analysis.ngramReuse[n].repeated)) {
                allDistances.push(...phrase.distances);
            }
        }

        if (allDistances.length > 2) {
            // Uniformity of repetition distances
            // AI tends to repeat at regular intervals
            analysis.uniformityScore = VarianceUtils.uniformityScore(allDistances);
            
            // Clustering score - humans cluster repetitions, AI distributes evenly
            // Low variance in distances = uniform = AI-like
            const cv = VarianceUtils.coefficientOfVariation(allDistances);
            analysis.clusteringScore = Math.min(1, cv); // Higher = more clustered = more human
        }

        // Top repeated phrases
        const allRepeated = [];
        for (let n = 3; n <= 5; n++) {
            for (const [phrase, data] of Object.entries(analysis.ngramReuse[n].repeated)) {
                if (data.count >= 2) {
                    allRepeated.push({ phrase, ...data, n });
                }
            }
        }
        analysis.topRepeatedPhrases = allRepeated
            .sort((a, b) => b.count - a.count)
            .slice(0, 10);

        return analysis;
    },

    /**
     * Analyze semantic repetition (paraphrasing detection)
     */
    analyzeSemanticRepetition(sentences, paragraphs) {
        const analysis = {
            sentenceSimilarities: [],
            paragraphSimilarities: [],
            redundantConcepts: [],
            redundancyScore: 0.5
        };

        // Compare sentences for semantic similarity (using word overlap)
        if (sentences.length > 1) {
            for (let i = 0; i < sentences.length - 1; i++) {
                const tokens1 = new Set(Utils.tokenize(sentences[i].toLowerCase()));
                
                for (let j = i + 1; j < Math.min(i + 5, sentences.length); j++) {
                    const tokens2 = new Set(Utils.tokenize(sentences[j].toLowerCase()));
                    
                    // Jaccard similarity
                    const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
                    const union = new Set([...tokens1, ...tokens2]);
                    const similarity = union.size > 0 ? intersection.size / union.size : 0;
                    
                    if (similarity > 0.4) {
                        analysis.sentenceSimilarities.push({
                            sentence1: i,
                            sentence2: j,
                            similarity,
                            distance: j - i
                        });
                    }
                }
            }
        }

        // Compare paragraphs for semantic overlap
        if (paragraphs.length > 1) {
            for (let i = 0; i < paragraphs.length - 1; i++) {
                const tokens1 = new Set(Utils.tokenize(paragraphs[i].toLowerCase()));
                
                for (let j = i + 1; j < paragraphs.length; j++) {
                    const tokens2 = new Set(Utils.tokenize(paragraphs[j].toLowerCase()));
                    
                    const intersection = new Set([...tokens1].filter(x => tokens2.has(x)));
                    const union = new Set([...tokens1, ...tokens2]);
                    const similarity = union.size > 0 ? intersection.size / union.size : 0;
                    
                    if (similarity > 0.3) {
                        analysis.paragraphSimilarities.push({
                            para1: i,
                            para2: j,
                            similarity
                        });
                    }
                }
            }
        }

        // Detect redundant conclusions/summaries
        analysis.redundantConcepts = this.detectRedundantConcepts(sentences);

        // Calculate redundancy score
        const sentenceRedundancy = sentences.length > 0 
            ? analysis.sentenceSimilarities.length / sentences.length 
            : 0;
        const paraRedundancy = paragraphs.length > 1
            ? analysis.paragraphSimilarities.length / (paragraphs.length - 1)
            : 0;
            
        analysis.redundancyScore = Math.min(1, (sentenceRedundancy * 0.6 + paraRedundancy * 0.4) * 2);

        return analysis;
    },

    /**
     * Detect redundant concept restatements
     */
    detectRedundantConcepts(sentences) {
        const concepts = [];
        const redundantPhrases = [
            'in conclusion', 'to summarize', 'in summary', 'overall',
            'as mentioned', 'as stated', 'as discussed', 'as noted',
            'this means', 'this shows', 'this demonstrates', 'this indicates'
        ];

        for (let i = 0; i < sentences.length; i++) {
            const lower = sentences[i].toLowerCase();
            for (const phrase of redundantPhrases) {
                if (lower.includes(phrase)) {
                    concepts.push({
                        sentence: i,
                        phrase,
                        position: i / sentences.length
                    });
                }
            }
        }

        return concepts;
    },

    /**
     * Analyze structural repetition (sentence templates, paragraph shapes)
     */
    analyzeStructuralRepetition(sentences, paragraphs) {
        const analysis = {
            sentenceTemplates: [],
            paragraphShapes: [],
            similarityScore: 0.5,
            templateReuse: 0
        };

        // Extract sentence structures (POS-like patterns)
        const sentencePatterns = sentences.map(s => this.extractSentencePattern(s));
        
        // Find repeated patterns
        const patternCounts = {};
        sentencePatterns.forEach((pattern, i) => {
            if (!patternCounts[pattern]) {
                patternCounts[pattern] = [];
            }
            patternCounts[pattern].push(i);
        });

        // Templates used multiple times
        for (const [pattern, indices] of Object.entries(patternCounts)) {
            if (indices.length > 1) {
                analysis.sentenceTemplates.push({
                    pattern,
                    count: indices.length,
                    positions: indices
                });
            }
        }

        // Paragraph shape analysis (sentence count, avg length pattern)
        analysis.paragraphShapes = paragraphs.map(p => {
            const paraSentences = Utils.splitSentences(p);
            return {
                sentenceCount: paraSentences.length,
                avgLength: Utils.mean(paraSentences.map(s => s.length)),
                pattern: paraSentences.map(s => {
                    const len = s.split(/\s+/).length;
                    if (len < 10) return 'S';
                    if (len < 20) return 'M';
                    return 'L';
                }).join('')
            };
        });

        // Calculate structural similarity
        if (analysis.paragraphShapes.length > 1) {
            const patterns = analysis.paragraphShapes.map(s => s.pattern);
            const uniquePatterns = new Set(patterns);
            analysis.similarityScore = 1 - (uniquePatterns.size / patterns.length);
        }

        analysis.templateReuse = sentences.length > 0
            ? analysis.sentenceTemplates.reduce((sum, t) => sum + t.count - 1, 0) / sentences.length
            : 0;

        return analysis;
    },

    /**
     * Extract simplified sentence pattern
     */
    extractSentencePattern(sentence) {
        // Create a structural fingerprint
        const words = sentence.split(/\s+/);
        const pattern = [];
        
        // Categorize sentence start
        const firstWord = words[0]?.toLowerCase();
        if (['the', 'a', 'an', 'this', 'that'].includes(firstWord)) {
            pattern.push('DET');
        } else if (['i', 'we', 'you', 'he', 'she', 'they', 'it'].includes(firstWord)) {
            pattern.push('PRON');
        } else if (['however', 'therefore', 'moreover', 'furthermore', 'additionally'].includes(firstWord)) {
            pattern.push('TRANS');
        } else {
            pattern.push('OTHER');
        }

        // Categorize length
        if (words.length < 8) pattern.push('SHORT');
        else if (words.length < 15) pattern.push('MED');
        else if (words.length < 25) pattern.push('LONG');
        else pattern.push('VLONG');

        // Check for specific structures
        if (sentence.includes(',')) pattern.push('COMMA');
        if (sentence.match(/[:;]/)) pattern.push('COLON');
        if (sentence.includes('"') || sentence.includes("'")) pattern.push('QUOTE');

        return pattern.join('-');
    },

    /**
     * Calculate AI probability from scores
     */
    calculateAIProbability(scores, lexical, semantic) {
        // Higher uniformity = more AI-like
        // Lower clustering = more AI-like (AI distributes evenly)
        
        const uniformityContribution = scores.lexicalUniformity * 0.3;
        const clusteringContribution = (1 - scores.repetitionClustering) * 0.25;
        const redundancyContribution = scores.semanticRedundancy * 0.25;
        const structuralContribution = scores.structuralSimilarity * 0.2;

        let probability = uniformityContribution + clusteringContribution + 
                         redundancyContribution + structuralContribution;

        // Adjust for extreme repetition patterns
        if (lexical.topRepeatedPhrases.length > 5) {
            // Check if repetitions are evenly distributed (AI-like)
            const avgDistances = lexical.topRepeatedPhrases
                .filter(p => p.avgDistance)
                .map(p => p.avgDistance);
            
            if (avgDistances.length > 0) {
                const distanceUniformity = VarianceUtils.uniformityScore(avgDistances);
                if (distanceUniformity > 0.7) {
                    probability += 0.1; // Very uniform = AI
                }
            }
        }

        return Math.max(0, Math.min(1, probability));
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(tokenCount, sentenceCount) {
        let confidence = 0.3;
        
        if (tokenCount > 500) confidence += 0.3;
        else if (tokenCount > 200) confidence += 0.2;
        else if (tokenCount > 100) confidence += 0.1;
        
        if (sentenceCount > 20) confidence += 0.2;
        else if (sentenceCount > 10) confidence += 0.1;
        
        return Math.min(1, confidence);
    },

    /**
     * Generate findings
     */
    generateFindings(lexical, semantic, structural) {
        const findings = [];

        // Lexical findings
        if (lexical.uniformityScore > 0.7) {
            findings.push({
                text: `Repetition patterns show high uniformity (${(lexical.uniformityScore * 100).toFixed(0)}%) - AI tends to repeat at regular intervals`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        } else if (lexical.uniformityScore < 0.3) {
            findings.push({
                text: `Repetition patterns show natural clustering - typical of human writing`,
                category: this.name,
                indicator: 'human',
                severity: 'medium'
            });
        }

        // Top repeated phrases
        if (lexical.topRepeatedPhrases.length > 0) {
            const topPhrase = lexical.topRepeatedPhrases[0];
            findings.push({
                text: `"${topPhrase.phrase}" repeated ${topPhrase.count} times with average distance of ${topPhrase.avgDistance?.toFixed(0) || 'N/A'} tokens`,
                category: this.name,
                indicator: topPhrase.count > 4 ? 'ai' : 'neutral',
                severity: topPhrase.count > 4 ? 'medium' : 'low'
            });
        }

        // Semantic findings
        if (semantic.redundancyScore > 0.5) {
            findings.push({
                text: `High semantic redundancy (${(semantic.redundancyScore * 100).toFixed(0)}%) - similar ideas restated without new information`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (semantic.redundantConcepts.length > 3) {
            findings.push({
                text: `${semantic.redundantConcepts.length} redundant summary/transition phrases detected`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Structural findings
        if (structural.similarityScore > 0.6) {
            findings.push({
                text: `Paragraph structures are highly similar (${(structural.similarityScore * 100).toFixed(0)}%) - suggests template-based generation`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (structural.templateReuse > 0.3) {
            findings.push({
                text: `${(structural.templateReuse * 100).toFixed(0)}% of sentences follow reused structural templates`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        return findings;
    },

    /**
     * Empty result
     */
    getEmptyResult() {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            details: {},
            findings: [],
            scores: {}
        };
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RepetitionAnalyzer;
}
