/**
 * VERITAS — Sentence Structure & Syntax Analyzer
 * Category 2: Length Metrics, Structural Diversity, Coordination Patterns
 */

const SyntaxAnalyzer = {
    name: 'Sentence Structure & Syntax',
    category: 2,
    weight: 1.1,

    // Coordinating conjunctions
    coordinatingConjunctions: ['and', 'but', 'or', 'nor', 'for', 'yet', 'so'],
    
    // Subordinating conjunctions
    subordinatingConjunctions: [
        'although', 'because', 'since', 'unless', 'while', 'whereas',
        'if', 'when', 'whenever', 'where', 'wherever', 'before', 'after',
        'until', 'as', 'though', 'even though', 'even if', 'whether'
    ],

    // Transitional phrases
    transitionalPhrases: [
        'however', 'therefore', 'moreover', 'furthermore', 'consequently',
        'nevertheless', 'nonetheless', 'in addition', 'on the other hand',
        'as a result', 'in contrast', 'similarly', 'likewise', 'accordingly'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        
        if (sentences.length === 0) {
            return this.getEmptyResult();
        }

        // Analyze sentence lengths
        const lengthMetrics = this.analyzeSentenceLengths(sentences);
        
        // Analyze structural diversity
        const structuralDiversity = this.analyzeStructuralDiversity(sentences);
        
        // Analyze coordination patterns
        const coordinationPatterns = this.analyzeCoordinationPatterns(text, sentences);
        
        // Analyze parse tree depth (approximation)
        const syntacticComplexity = this.analyzeSyntacticComplexity(sentences);

        // Calculate AI probability
        // Low variance = AI-like
        // Low burstiness = AI-like
        // High conjunction uniformity = AI-like
        // Balanced structures = AI-like

        const scores = {
            lengthUniformity: 1 - Utils.normalize(lengthMetrics.coefficientOfVariation, 0.3, 0.8),
            burstinessLow: 1 - lengthMetrics.burstiness,
            structuralUniformity: 1 - structuralDiversity.diversityScore,
            conjunctionOveruse: coordinationPatterns.overuseScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.lengthUniformity, scores.burstinessLow, scores.structuralUniformity, scores.conjunctionOveruse],
            [0.25, 0.25, 0.25, 0.25]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(sentences.length),
            details: {
                lengthMetrics,
                structuralDiversity,
                coordinationPatterns,
                syntacticComplexity
            },
            findings: this.generateFindings(lengthMetrics, structuralDiversity, coordinationPatterns, syntacticComplexity),
            scores
        };
    },

    /**
     * Analyze sentence length metrics
     */
    analyzeSentenceLengths(sentences) {
        const lengths = sentences.map(s => Utils.tokenize(s).length);
        
        const mean = Utils.mean(lengths);
        const variance = Utils.variance(lengths);
        const stdDev = Math.sqrt(variance);
        const coefficientOfVariation = mean > 0 ? stdDev / mean : 0;
        
        // Burstiness: measure of irregularity
        // (σ - μ) / (σ + μ), higher = more bursty (human-like)
        const burstiness = (stdDev - mean) !== 0 
            ? (stdDev - mean) / (stdDev + mean + 0.001)
            : 0;
        
        // Normalize burstiness to 0-1 where 0 is very uniform
        const normalizedBurstiness = Utils.normalize(burstiness + 1, 0, 1);

        // Calculate length distribution
        const shortSentences = lengths.filter(l => l < 10).length;
        const mediumSentences = lengths.filter(l => l >= 10 && l < 25).length;
        const longSentences = lengths.filter(l => l >= 25).length;

        return {
            mean: mean,
            variance: variance,
            stdDev: stdDev,
            coefficientOfVariation,
            burstiness: normalizedBurstiness,
            min: Math.min(...lengths),
            max: Math.max(...lengths),
            distribution: {
                short: shortSentences,
                medium: mediumSentences,
                long: longSentences
            },
            lengths
        };
    },

    /**
     * Analyze structural diversity
     */
    analyzeStructuralDiversity(sentences) {
        const structures = sentences.map(s => this.classifySentenceStructure(s));
        
        const structureCounts = Utils.frequencyDistribution(structures);
        const totalStructures = Object.keys(structureCounts).length;
        
        // Calculate diversity using entropy
        const entropy = Utils.entropy(structureCounts);
        const maxEntropy = Math.log2(Math.max(4, sentences.length)); // Max possible entropy
        const diversityScore = maxEntropy > 0 ? entropy / maxEntropy : 0;

        // Check for repeated syntactic templates
        const firstWords = sentences.map(s => s.split(/\s+/)[0]?.toLowerCase());
        const firstWordFreq = Utils.frequencyDistribution(firstWords);
        const repeatedFirstWords = Object.entries(firstWordFreq)
            .filter(([_, count]) => count > sentences.length * 0.15);

        return {
            structureCounts,
            uniqueStructures: totalStructures,
            entropy: entropy.toFixed(2),
            diversityScore,
            repeatedFirstWords
        };
    },

    /**
     * Classify sentence structure
     */
    classifySentenceStructure(sentence) {
        const lower = sentence.toLowerCase();
        const hasSubordinate = this.subordinatingConjunctions.some(c => 
            lower.includes(c + ' ') || lower.includes(', ' + c)
        );
        const hasCoordinate = this.coordinatingConjunctions.some(c => 
            lower.includes(' ' + c + ' ')
        );
        const clauses = this.countClauses(sentence);

        if (clauses === 1 && !hasSubordinate && !hasCoordinate) {
            return 'simple';
        } else if (hasCoordinate && !hasSubordinate) {
            return 'compound';
        } else if (hasSubordinate && !hasCoordinate) {
            return 'complex';
        } else if (hasSubordinate && hasCoordinate) {
            return 'compound-complex';
        }
        return 'simple';
    },

    /**
     * Count approximate clauses in a sentence
     */
    countClauses(sentence) {
        // Approximate clause count by counting verbs and conjunctions
        const verbPatterns = /\b(is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|could|should|may|might|must|shall|can)\b/gi;
        const verbMatches = sentence.match(verbPatterns) || [];
        return Math.max(1, Math.ceil(verbMatches.length / 2));
    },

    /**
     * Analyze coordination patterns
     */
    analyzeCoordinationPatterns(text, sentences) {
        const lower = text.toLowerCase();
        const tokens = Utils.tokenize(text);
        
        // Count coordinating conjunctions
        const conjunctionCounts = {};
        let totalConjunctions = 0;
        
        for (const conj of this.coordinatingConjunctions) {
            const pattern = new RegExp(`\\b${conj}\\b`, 'gi');
            const count = (text.match(pattern) || []).length;
            if (count > 0) {
                conjunctionCounts[conj] = count;
                totalConjunctions += count;
            }
        }

        // Count transitional phrases
        let transitionalCount = 0;
        const transitionalUsed = [];
        
        for (const phrase of this.transitionalPhrases) {
            if (lower.includes(phrase.toLowerCase())) {
                transitionalCount++;
                transitionalUsed.push(phrase);
            }
        }

        // Check for parallel structures
        const parallelStructures = this.detectParallelStructures(sentences);

        // Calculate overuse score
        const conjunctionsPerSentence = sentences.length > 0 
            ? totalConjunctions / sentences.length 
            : 0;
        
        // AI tends to overuse certain conjunctions
        const andButRatio = (conjunctionCounts['and'] || 0) + (conjunctionCounts['but'] || 0);
        const overuseScore = Utils.normalize(conjunctionsPerSentence, 0, 2) * 0.5 +
                            Utils.normalize(transitionalCount / sentences.length, 0, 0.3) * 0.5;

        return {
            conjunctionCounts,
            totalConjunctions,
            conjunctionsPerSentence: conjunctionsPerSentence.toFixed(2),
            transitionalCount,
            transitionalUsed,
            parallelStructures,
            overuseScore
        };
    },

    /**
     * Detect parallel structures in sentences
     */
    detectParallelStructures(sentences) {
        let parallelCount = 0;
        const examples = [];

        for (const sentence of sentences) {
            // Check for lists with parallel structure
            const listPattern = /(\w+ing)[,\s]+(\w+ing)[,\s]+(and\s+)?(\w+ing)/gi;
            const infinitivePattern = /(to\s+\w+)[,\s]+(to\s+\w+)[,\s]+(and\s+)?(to\s+\w+)/gi;
            
            if (listPattern.test(sentence) || infinitivePattern.test(sentence)) {
                parallelCount++;
                if (examples.length < 3) {
                    examples.push(Utils.truncate(sentence, 60));
                }
            }
        }

        return {
            count: parallelCount,
            examples
        };
    },

    /**
     * Analyze syntactic complexity
     */
    analyzeSyntacticComplexity(sentences) {
        const complexityScores = sentences.map(s => this.estimateComplexity(s));
        
        return {
            meanComplexity: Utils.mean(complexityScores).toFixed(2),
            complexityVariance: Utils.variance(complexityScores).toFixed(2),
            complexityDistribution: {
                simple: complexityScores.filter(c => c < 0.3).length,
                moderate: complexityScores.filter(c => c >= 0.3 && c < 0.6).length,
                complex: complexityScores.filter(c => c >= 0.6).length
            }
        };
    },

    /**
     * Estimate syntactic complexity of a sentence
     */
    estimateComplexity(sentence) {
        const words = Utils.tokenize(sentence);
        const wordCount = words.length;
        
        // Factors that increase complexity
        const subordinates = this.subordinatingConjunctions.filter(c => 
            sentence.toLowerCase().includes(c)
        ).length;
        
        const commaCount = (sentence.match(/,/g) || []).length;
        const punctuationComplexity = (sentence.match(/[;:—]/g) || []).length;
        
        // Estimate depth based on nested clauses
        const depth = 1 + subordinates + (commaCount * 0.3) + punctuationComplexity;
        
        // Normalize to 0-1
        return Utils.normalize(depth, 1, 5);
    },

    /**
     * Generate findings with detailed statistics
     */
    generateFindings(lengthMetrics, structuralDiversity, coordinationPatterns, syntacticComplexity) {
        const findings = [];

        // Length uniformity
        const cv = lengthMetrics.coefficientOfVariation;
        if (cv < 0.35) {
            findings.push({
                label: 'Sentence Length Uniformity',
                value: 'Highly uniform sentence lengths detected',
                note: `AI typically produces consistent lengths with CV < 35%`,
                indicator: 'ai',
                severity: cv < 0.25 ? 'high' : 'medium',
                stats: {
                    measured: `CV: ${(cv * 100).toFixed(1)}%`,
                    mean: `${lengthMetrics.mean.toFixed(1)} words/sentence`,
                    stdDev: `±${lengthMetrics.stdDev.toFixed(1)} words`,
                    range: `${lengthMetrics.min}–${lengthMetrics.max} words`
                },
                benchmark: {
                    humanRange: 'CV: 40%–80%',
                    aiRange: 'CV: 15%–35%',
                    interpretation: 'Lower CV = more uniform = more AI-like'
                }
            });
        } else if (cv > 0.6) {
            findings.push({
                label: 'Sentence Length Variation',
                value: 'High natural variation in sentence lengths',
                note: 'Natural variation suggests human writing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `CV: ${(cv * 100).toFixed(1)}%`,
                    mean: `${lengthMetrics.mean.toFixed(1)} words/sentence`,
                    range: `${lengthMetrics.min}–${lengthMetrics.max} words`
                },
                benchmark: {
                    humanRange: 'CV: 40%–80%',
                    aiRange: 'CV: 15%–35%'
                }
            });
        }

        // Burstiness
        const burst = lengthMetrics.burstiness;
        if (burst < 0.3) {
            findings.push({
                label: 'Rhythm Pattern',
                value: 'Low burstiness detected',
                note: 'Text flows too evenly, lacking natural rhythm variation',
                indicator: 'ai',
                severity: burst < 0.15 ? 'high' : 'medium',
                stats: {
                    measured: `Burstiness: ${(burst * 100).toFixed(1)}%`,
                    interpretation: 'Measures irregularity in sentence lengths'
                },
                benchmark: {
                    humanRange: '30%–70%',
                    aiRange: '5%–25%',
                    formula: '(σ - μ) / (σ + μ)'
                }
            });
        } else if (burst > 0.5) {
            findings.push({
                label: 'Natural Rhythm',
                value: 'High burstiness detected',
                note: 'Natural irregular rhythm typical of human writing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `Burstiness: ${(burst * 100).toFixed(1)}%`
                },
                benchmark: {
                    humanRange: '30%–70%',
                    aiRange: '5%–25%'
                }
            });
        }

        // Structural diversity
        if (structuralDiversity.diversityScore < 0.4) {
            findings.push({
                label: 'Structural Diversity',
                value: 'Limited sentence structure variety',
                note: `Repetitive sentence patterns detected`,
                indicator: 'ai',
                severity: structuralDiversity.diversityScore < 0.25 ? 'high' : 'medium',
                stats: {
                    measured: `Diversity Score: ${(structuralDiversity.diversityScore * 100).toFixed(1)}%`,
                    uniqueStructures: structuralDiversity.uniqueStructures,
                    totalSentences: structuralDiversity.totalSentences,
                    structureRatio: `${((structuralDiversity.uniqueStructures / Math.max(1, structuralDiversity.totalSentences)) * 100).toFixed(1)}% unique`
                },
                benchmark: {
                    humanRange: '50%–90% diversity',
                    aiRange: '20%–45% diversity'
                }
            });
        }

        // Conjunction overuse
        if (coordinationPatterns.overuseScore > 0.5) {
            const topTransitions = coordinationPatterns.transitionalUsed.slice(0, 5);
            findings.push({
                label: 'Coordination Pattern',
                value: 'Excessive use of conjunctions/transitions',
                note: `Over-reliance on formal connectives is an AI signature`,
                indicator: 'ai',
                severity: coordinationPatterns.overuseScore > 0.7 ? 'high' : 'medium',
                stats: {
                    measured: `Overuse Score: ${(coordinationPatterns.overuseScore * 100).toFixed(1)}%`,
                    transitionsFound: topTransitions.length > 0 ? topTransitions.join(', ') : 'none',
                    perSentence: `${(coordinationPatterns.transitionalCount / Math.max(1, coordinationPatterns.sentenceCount)).toFixed(2)}/sentence`
                },
                benchmark: {
                    humanRange: '0.1–0.3 per sentence',
                    aiRange: '0.4–0.8 per sentence'
                }
            });
        }

        // Syntactic complexity
        if (syntacticComplexity && syntacticComplexity.avgDepth) {
            const depth = syntacticComplexity.avgDepth;
            if (depth < 2.0) {
                findings.push({
                    label: 'Syntactic Complexity',
                    value: 'Low clause depth detected',
                    note: 'Simple, shallow sentence structures',
                    indicator: 'neutral',
                    severity: 'low',
                    stats: {
                        measured: `Avg Depth: ${depth.toFixed(2)} levels`,
                        maxDepth: syntacticComplexity.maxDepth
                    },
                    benchmark: {
                        simple: '< 2.0 levels',
                        moderate: '2.0–3.5 levels',
                        complex: '> 3.5 levels'
                    }
                });
            }
        }

        return findings;
    },

    calculateConfidence(sentenceCount) {
        if (sentenceCount < 5) return 0.3;
        if (sentenceCount < 10) return 0.5;
        if (sentenceCount < 20) return 0.7;
        if (sentenceCount < 50) return 0.85;
        return 0.95;
    },

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

if (typeof module !== 'undefined' && module.exports) {
    module.exports = SyntaxAnalyzer;
}
