/**
 * VERITAS — Analyzer Engine v2.0
 * Variance-Based Detection: Measures deviations from expected human variance
 * 
 * Philosophy: "Never grade features as 'AI-like' or 'human-like' in isolation.
 * Grade them as deviations from expected human variance."
 * 
 * Powered by: Veritas Sun Suite (Helios, Zenith, Sunrise, Dawn)
 */

const AnalyzerEngine = {
    // Current model type (default: zenith for best overall performance)
    currentModel: 'zenith',
    
    // Flare integration settings (enabled by default for +28% humanized detection)
    flareEnabled: true,  // Whether to run Flare alongside other models
    flareThreshold: 0.40, // Minimum humanization probability to report
    
    // Binary mode settings (disabled by default for 3-class detection)
    binaryMode: false, // When true, only Human vs AI (ignores humanized classification)
    
    /**
     * Set the active model
     */
    setModel(modelType) {
        const validModels = ['helios', 'zenith', 'sunrise', 'dawn', 'flare'];
        if (validModels.includes(modelType)) {
            this.currentModel = modelType;
            console.log(`AnalyzerEngine: Model set to ${modelType}`);
        }
    },
    
    /**
     * Enable/disable Flare integration for humanization detection
     */
    setFlareEnabled(enabled) {
        this.flareEnabled = enabled;
        console.log(`AnalyzerEngine: Flare integration ${enabled ? 'enabled' : 'disabled'}`);
    },
    
    /**
     * Enable/disable binary mode (Human vs AI only)
     */
    setBinaryMode(enabled) {
        this.binaryMode = enabled;
        console.log(`AnalyzerEngine: Binary mode ${enabled ? 'enabled' : 'disabled'}`);
    },
    
    /**
     * Get the current model configuration
     */
    get modelConfig() {
        switch (this.currentModel) {
            case 'helios':
                return typeof VERITAS_HELIOS_CONFIG !== 'undefined' ? VERITAS_HELIOS_CONFIG : null;
            case 'zenith':
                return typeof VERITAS_ZENITH_CONFIG !== 'undefined' ? VERITAS_ZENITH_CONFIG : null;
            case 'sunrise':
                return typeof VERITAS_SUNRISE_CONFIG !== 'undefined' ? VERITAS_SUNRISE_CONFIG : null;
            case 'flare':
                return typeof FlareConfig !== 'undefined' ? FlareConfig : null;
            case 'dawn':
                return null; // Dawn uses rule-based defaults
            default:
                return typeof VERITAS_HELIOS_CONFIG !== 'undefined' ? VERITAS_HELIOS_CONFIG : null;
        }
    },
    
    /**
     * Get model-specific weights
     */
    get categoryWeights() {
        const config = this.modelConfig;
        
        // Use model-specific weights if available
        if (config?.featureWeights) {
            // Aggregate feature weights to category level based on model
            if (this.currentModel === 'zenith') {
                // Zenith focuses on entropy/perplexity
                return {
                    syntaxVariance: 0.10,
                    lexicalDiversity: 0.15,
                    repetitionUniformity: 0.05,
                    toneStability: 0.05,
                    grammarEntropy: 0.02,
                    perplexity: 0.25,  // Higher weight for entropy
                    authorshipDrift: 0.08,
                    metadataFormatting: 0.30
                };
            } else if (this.currentModel === 'helios') {
                // Helios uses full 45-feature analysis
                return {
                    syntaxVariance: 0.18,
                    lexicalDiversity: 0.20,
                    repetitionUniformity: 0.03,
                    toneStability: 0.08,  // Tone analysis
                    grammarEntropy: 0.02,
                    perplexity: 0.10,
                    authorshipDrift: 0.12,
                    metadataFormatting: 0.27
                };
            } else if (this.currentModel === 'flare') {
                // Flare focuses on humanization detection features
                return {
                    syntaxVariance: 0.15,
                    lexicalDiversity: 0.10,
                    repetitionUniformity: 0.08,
                    toneStability: 0.05,
                    grammarEntropy: 0.02,
                    perplexity: 0.20,  // Entropy stability important
                    authorshipDrift: 0.15,  // Variance patterns
                    metadataFormatting: 0.25
                };
            }
        }
        
        // Default weights (Sunrise/Dawn)
        return {
            syntaxVariance: 0.21,
            lexicalDiversity: 0.22,
            repetitionUniformity: 0.02,
            toneStability: 0.02,
            grammarEntropy: 0.01,
            perplexity: 0.02,
            authorshipDrift: 0.10,
            metadataFormatting: 0.40
        };
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
        
        // Humanization detection (Category 15)
        if (typeof HumanizationAnalyzer !== 'undefined') all.push(HumanizationAnalyzer);
        
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

        // Special handling for Flare model - dedicated humanization detection
        if (this.currentModel === 'flare') {
            return this.analyzeWithFlare(text, metadata, startTime);
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

        // Advanced statistics
        const advancedStats = this.computeAdvancedStatistics(text, tokens, sentences);

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
        
        // Run humanizer detection (second-order analysis for AI-written, human-modified text)
        const humanizerSignals = this.detectHumanizerSignalsWithCategories(text, sentences, advancedStats, categoryResults);
        
        // Run Flare analysis if enabled (enhanced humanization detection)
        let flareResult = null;
        if (this.flareEnabled && typeof FlareAnalyzer !== 'undefined') {
            try {
                const flareAnalyzer = new FlareAnalyzer();
                if (typeof FlareConfig !== 'undefined') {
                    flareAnalyzer.loadConfig(FlareConfig);
                }
                flareResult = flareAnalyzer.analyze(text);
                
                // Integrate Flare results with humanizer signals
                if (flareResult && flareResult.humanizedProbability) {
                    // Blend Flare's ML-based detection with existing heuristics
                    const flareWeight = 0.6; // Flare is more accurate (99.84%)
                    const heuristicWeight = 0.4;
                    
                    const blendedProbability = 
                        (flareResult.humanizedProbability * flareWeight) + 
                        (humanizerSignals.probability * heuristicWeight);
                    
                    // Update humanizer signals with Flare data
                    humanizerSignals.flareResult = flareResult;
                    humanizerSignals.flareEnabled = true;
                    humanizerSignals.originalProbability = humanizerSignals.probability;
                    humanizerSignals.probability = blendedProbability;
                    
                    // Merge flags from Flare
                    if (flareResult.flags && flareResult.flags.length > 0) {
                        humanizerSignals.flags = humanizerSignals.flags || [];
                        humanizerSignals.flags.push(...flareResult.flags.map(f => ({
                            ...f,
                            source: 'flare'
                        })));
                        humanizerSignals.flagCount = (humanizerSignals.flagCount || 0) + flareResult.flags.length;
                    }
                    
                    // Update isLikelyHumanized based on blended probability
                    humanizerSignals.isLikelyHumanized = blendedProbability >= this.flareThreshold;
                }
            } catch (e) {
                console.error('Flare analysis error:', e);
                humanizerSignals.flareEnabled = false;
                humanizerSignals.flareError = e.message;
            }
        } else {
            humanizerSignals.flareEnabled = false;
        }
        
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
        
        // Apply false positive risk adjustment to AI probability
        let adjustedAiProbability = overallResult.aiProbability;
        if (falsePositiveRisk.adjustmentWeight && falsePositiveRisk.adjustmentWeight < 1.0) {
            // Reduce AI probability based on false positive risk factors
            // The adjustment moves probability toward 0.5 (uncertain) proportionally
            const adjustment = (overallResult.aiProbability - 0.5) * (1 - falsePositiveRisk.adjustmentWeight);
            adjustedAiProbability = overallResult.aiProbability - adjustment;
        }
        
        // Human indicators can further reduce AI probability
        if (falsePositiveRisk.humanIndicatorCount > 0) {
            adjustedAiProbability = adjustedAiProbability * (1 - 0.05 * falsePositiveRisk.humanIndicatorCount);
        }
        
        // Clamp to valid range
        adjustedAiProbability = Math.max(0, Math.min(1, adjustedAiProbability));

        const endTime = performance.now();
        
        // Get verdict with humanizer detection factored in
        const verdict = this.getVerdictWithHumanizer(
            adjustedAiProbability, 
            overallResult.confidence, 
            humanizerSignals,
            overallResult.signalCounts
        );

        return {
            // Overall results
            aiProbability: adjustedAiProbability,
            rawAiProbability: overallResult.aiProbability, // Keep original for reference
            humanProbability: 1 - adjustedAiProbability,
            mixedProbability: overallResult.mixedProbability,
            confidence: overallResult.confidence,
            confidenceInterval,
            falsePositiveRisk,
            verdict,
            humanizerSignals,
            
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
            advancedStats,
            tokens, // For word frequency chart
            metadata: metadata || null,
            analysisTime: (endTime - startTime).toFixed(0) + 'ms'
        };
    },

    /**
     * Special analysis for Flare model - focused on humanization detection
     * Flare assumes the text could be human-written OR humanized AI
     * It doesn't care about raw AI vs human - only about detecting humanization
     */
    analyzeWithFlare(text, metadata, startTime) {
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

        // Run FlareAnalyzer if available
        let flareResult = null;
        if (typeof FlareAnalyzer !== 'undefined') {
            try {
                const analyzer = new FlareAnalyzer();
                if (typeof FlareConfig !== 'undefined') {
                    analyzer.loadConfig(FlareConfig);
                }
                flareResult = analyzer.analyze(text);
            } catch (e) {
                console.error('Flare analysis error:', e);
            }
        }

        // Also run standard humanization detection for comparison
        const advancedStats = this.computeAdvancedStatistics(text, tokens, sentences);
        
        // Run standard analyzers for additional context
        const categoryResults = [];
        for (const analyzer of this.analyzers) {
            if (!analyzer) continue;
            try {
                const result = analyzer.analyze(text, metadata);
                categoryResults.push(result);
            } catch (error) {
                console.error(`Error in ${analyzer?.name || 'unknown'}:`, error);
            }
        }

        // Get humanizer signals from existing system
        const humanizerSignals = this.detectHumanizerSignalsWithCategories(text, sentences, advancedStats, categoryResults);

        // Combine Flare result with humanizer signals
        const humanizedProbability = flareResult?.humanizedProbability ?? humanizerSignals.probability;
        const isHumanized = humanizedProbability >= 0.5 || humanizerSignals.isLikelyHumanized;
        
        // Build comprehensive flag list
        const flags = [];
        if (flareResult?.flags) {
            flags.push(...flareResult.flags);
        }
        if (humanizerSignals.flags) {
            flags.push(...humanizerSignals.flags);
        }

        // Determine verdict (in format expected by displayResults)
        let verdictLabel, verdictLevel, verdictDescription;
        if (humanizedProbability >= 0.85) {
            verdictLabel = 'HUMANIZED AI DETECTED';
            verdictLevel = 'ai';
            verdictDescription = 'Strong evidence of AI-generated text processed through humanization tools';
        } else if (humanizedProbability >= 0.65) {
            verdictLabel = 'LIKELY HUMANIZED AI';
            verdictLevel = 'likely-ai';
            verdictDescription = 'Significant humanization patterns detected in the text';
        } else if (humanizedProbability >= 0.45) {
            verdictLabel = 'POSSIBLE HUMANIZATION';
            verdictLevel = 'mixed';
            verdictDescription = 'Some humanization signals present but not conclusive';
        } else if (humanizedProbability >= 0.25) {
            verdictLabel = 'WEAK HUMANIZATION SIGNALS';
            verdictLevel = 'likely-human';
            verdictDescription = 'Minor patterns detected but likely authentic human writing';
        } else {
            verdictLabel = 'GENUINE HUMAN WRITING';
            verdictLevel = 'human';
            verdictDescription = 'No evidence of humanization tools or AI post-processing';
        }

        const verdict = {
            label: verdictLabel,
            level: verdictLevel,
            description: verdictDescription
        };

        // Generate findings
        const findings = [];
        if (humanizedProbability >= 0.5) {
            findings.push({
                indicator: 'ai',
                severity: 'high',
                message: 'Humanization artifacts detected',
                detail: 'Text shows patterns consistent with AI-to-human post-processing tools'
            });
        }
        
        for (const flag of flags.slice(0, 5)) {
            findings.push({
                indicator: flag.severity === 'high' ? 'ai' : 'mixed',
                severity: flag.severity || 'medium',
                message: flag.message,
                detail: flag.detail
            });
        }

        const endTime = performance.now();

        // Return Flare-specific result format
        return {
            // Overall results - Flare treats humanized probability as "AI probability"
            aiProbability: humanizedProbability,
            humanProbability: 1 - humanizedProbability,
            mixedProbability: 0,
            confidence: flareResult?.confidence ?? 0.8,
            verdict,
            
            // Flare-specific data
            flareResult,
            humanizedProbability,
            isHumanized,
            humanizerSignals,
            humanizationFlags: flags,
            
            // Feature analysis from Flare
            flareFeatures: flareResult?.features || {},
            flareAnalysis: flareResult?.analysis || '',
            
            // Standard data
            categoryResults,
            sentences,
            sentenceScores: this.scoreSentences(text, sentences, categoryResults),
            findings,
            stats,
            advancedStats,
            tokens,
            metadata: metadata || null,
            analysisTime: (endTime - startTime).toFixed(0) + 'ms',
            
            // Model info
            modelInfo: {
                id: 'flare',
                name: 'Flare',
                specialty: 'Humanization Detection',
                accuracy: '99.84%',
                description: 'Specialized model for detecting AI text that has been processed through humanization tools'
            }
        };
    },

    /**
     * Compute advanced statistics for comprehensive analysis
     */
    computeAdvancedStatistics(text, tokens, sentences) {
        const stats = {};

        // === Vocabulary Richness Metrics ===
        stats.vocabulary = {
            uniqueWords: new Set(tokens).size,
            typeTokenRatio: Utils.typeTokenRatio(tokens),
            rootTTR: Utils.rootTTR(tokens),
            logTTR: Utils.logTTR(tokens),
            msttr: Utils.msttr(tokens, 50),
            hapaxLegomenaRatio: Utils.hapaxLegomenaRatio(tokens),
            disLegomenaRatio: Utils.disLegomenaRatio(tokens),
            yulesK: Utils.yulesK(tokens),
            simpsonsD: Utils.simpsonsD(tokens),
            honoresR: Utils.honoresR(tokens),
            brunetsW: Utils.brunetsW(tokens),
            sichelsS: Utils.sichelsS(tokens)
        };

        // === Word Statistics ===
        stats.words = {
            avgLength: Utils.avgWordLength(tokens),
            lengthDistribution: Utils.wordLengthDistribution(tokens),
            entropy: Utils.entropy(Utils.frequencyDistribution(tokens))
        };

        // === Sentence Statistics ===
        const sentLengths = sentences.map(s => Utils.tokenize(s).length);
        stats.sentences = {
            mean: Utils.mean(sentLengths),
            median: Utils.median(sentLengths),
            stdDev: Utils.standardDeviation(sentLengths),
            variance: Utils.variance(sentLengths),
            min: Math.min(...sentLengths),
            max: Math.max(...sentLengths),
            range: Math.max(...sentLengths) - Math.min(...sentLengths),
            coefficientOfVariation: Utils.coefficientOfVariation(sentLengths),
            skewness: Utils.skewness(sentLengths),
            kurtosis: Utils.kurtosis(sentLengths),
            gini: Utils.giniCoefficient(sentLengths)
        };

        // === Zipf's Law Analysis ===
        stats.zipf = Utils.zipfAnalysis(tokens);

        // === N-gram Analysis ===
        const bigrams = Utils.ngrams(tokens, 2);
        const trigrams = Utils.ngrams(tokens, 3);
        const quadgrams = Utils.ngrams(tokens, 4);
        const pentagrams = Utils.ngrams(tokens, 5);

        // Find repeated phrases (4+ grams that appear more than once)
        const repeatedPhrases = this.findRepeatedPhrases(tokens);

        stats.ngrams = {
            uniqueBigrams: new Set(bigrams).size,
            uniqueTrigrams: new Set(trigrams).size,
            uniqueQuadgrams: new Set(quadgrams).size,
            bigramRepetitionRate: bigrams.length > 0 ? 1 - (new Set(bigrams).size / bigrams.length) : 0,
            trigramRepetitionRate: trigrams.length > 0 ? 1 - (new Set(trigrams).size / trigrams.length) : 0,
            quadgramRepetitionRate: quadgrams.length > 0 ? 1 - (new Set(quadgrams).size / quadgrams.length) : 0,
            // Research-backed: repeated higher-order n-grams are strong AI indicators
            repeatedPhraseCount: repeatedPhrases.count,
            repeatedPhrases: repeatedPhrases.phrases.slice(0, 10), // Top 10
            repeatedPhraseScore: repeatedPhrases.score,
            highOrderRepetition: (quadgrams.length > 0 && pentagrams.length > 0) 
                ? ((1 - new Set(quadgrams).size / quadgrams.length) + 
                   (1 - new Set(pentagrams).size / pentagrams.length)) / 2 
                : 0
        };

        // === Readability Approximations ===
        const syllableCount = this.estimateSyllables(tokens);
        const avgSyllablesPerWord = tokens.length > 0 ? syllableCount / tokens.length : 0;
        
        // Flesch Reading Ease approximation
        const fleschRE = 206.835 - 1.015 * (stats.sentences.mean || 15) - 84.6 * avgSyllablesPerWord;
        
        // Flesch-Kincaid Grade Level approximation
        const fleschKG = 0.39 * (stats.sentences.mean || 15) + 11.8 * avgSyllablesPerWord - 15.59;
        
        // Gunning Fog Index approximation
        const complexWords = tokens.filter(t => this.syllableCount(t) >= 3).length;
        const complexWordPct = tokens.length > 0 ? (complexWords / tokens.length) * 100 : 0;
        const gunningFog = 0.4 * ((stats.sentences.mean || 15) + complexWordPct);

        // Coleman-Liau Index (uses letter and sentence counts)
        const avgLettersPerWord = tokens.reduce((sum, t) => sum + t.replace(/[^a-zA-Z]/g, '').length, 0) / Math.max(1, tokens.length);
        const L = avgLettersPerWord * 100;  // letters per 100 words
        const S = (sentences.length / Math.max(1, tokens.length)) * 100;  // sentences per 100 words
        const colemanLiau = 0.0588 * L - 0.296 * S - 15.8;

        // SMOG Index (Simple Measure of Gobbledygook)
        const polysyllables = tokens.filter(t => this.syllableCount(t) >= 3).length;
        const smog = sentences.length >= 3 
            ? 1.0430 * Math.sqrt(polysyllables * (30 / Math.max(1, sentences.length))) + 3.1291
            : 0;

        // Automated Readability Index (ARI)
        const charCount = text.replace(/[^a-zA-Z0-9]/g, '').length;
        const ari = 4.71 * (charCount / Math.max(1, tokens.length)) + 
                   0.5 * (tokens.length / Math.max(1, sentences.length)) - 21.43;

        stats.readability = {
            avgSyllablesPerWord,
            fleschReadingEase: Math.max(0, Math.min(100, fleschRE)),
            fleschKincaidGrade: Math.max(0, fleschKG),
            gunningFogIndex: gunningFog,
            colemanLiauIndex: Math.max(0, colemanLiau),
            smogIndex: Math.max(0, smog),
            ariIndex: Math.max(0, ari),
            complexWordPercentage: complexWordPct,
            polysyllablePercentage: tokens.length > 0 ? (polysyllables / tokens.length) * 100 : 0
        };

        // === Burstiness Metrics ===
        stats.burstiness = {
            sentenceLength: VarianceUtils.burstiness(sentLengths),
            wordLength: VarianceUtils.burstiness(tokens.map(t => t.length)),
            overallUniformity: VarianceUtils.uniformityScore(sentLengths)
        };

        // === Function Word Analysis ===
        const functionWordCount = tokens.filter(t => Utils.functionWords.has(t.toLowerCase())).length;
        const contentWordCount = tokens.length - functionWordCount;
        stats.functionWords = {
            count: functionWordCount,
            ratio: tokens.length > 0 ? functionWordCount / tokens.length : 0,
            contentWordRatio: tokens.length > 0 ? contentWordCount / tokens.length : 0
        };

        // === Word Pattern Analysis (POS-like without external tools) ===
        // Research shows different POS distributions between human and AI text
        stats.wordPatterns = this.analyzeWordPatterns(tokens, text);

        // === Enhanced Statistical Analysis (v2.1) ===
        
        // Autocorrelation analysis - detects periodic patterns
        stats.autocorrelation = VarianceUtils.autocorrelation(sentLengths, 8);
        
        // N-gram perplexity approximation
        stats.perplexity = VarianceUtils.ngramPerplexity(tokens, 2);
        
        // Runs test for randomness
        stats.runsTest = VarianceUtils.runsTest(sentLengths);
        
        // Chi-squared uniformity test on sentence lengths (binned)
        const lengthBins = this.binValues(sentLengths, 5);
        stats.chiSquared = VarianceUtils.chiSquaredUniformity(lengthBins);
        
        // Variance stability across document
        stats.varianceStability = VarianceUtils.varianceStability(sentLengths, 5);
        
        // Length normalization factor
        stats.lengthNormalization = VarianceUtils.lengthNormalization(tokens.length);
        
        // Mahalanobis distance from human baseline
        const featureVector = {
            sentenceLengthCV: stats.sentences.coefficientOfVariation,
            hapaxRatio: stats.vocabulary.hapaxLegomenaRatio,
            burstiness: stats.burstiness.sentenceLength,
            zipfSlope: stats.zipf.slope,
            ttrNormalized: stats.vocabulary.typeTokenRatio
        };
        stats.mahalanobisDistance = VarianceUtils.mahalanobisDistance(featureVector);

        // === BELL CURVE / GAUSSIAN DEVIATION ANALYSIS (v2.2) ===
        // Core philosophy: Humans fall in a "reasonable middle" - neither too perfect nor too chaotic
        
        // How "natural" is the variance? (1 = human-like middle, 0 = extreme)
        stats.varianceNaturalness = VarianceUtils.varianceNaturalnessScore(sentLengths);
        
        // Does this text show extremes in either direction?
        stats.extremeVarianceIndicator = VarianceUtils.extremeVarianceScore(sentLengths);
        
        // Per-feature human likelihood scores
        stats.humanLikelihood = {
            sentenceLengthCV: VarianceUtils.humanLikelihoodScore(
                stats.sentences.coefficientOfVariation,
                VarianceUtils.HUMAN_BASELINES.sentenceLengthCV
            ),
            hapaxRatio: VarianceUtils.humanLikelihoodScore(
                stats.vocabulary.hapaxLegomenaRatio,
                VarianceUtils.HUMAN_BASELINES.hapaxRatio
            ),
            burstiness: VarianceUtils.humanLikelihoodScore(
                stats.burstiness.sentenceLength,
                VarianceUtils.HUMAN_BASELINES.burstiness
            ),
            zipfSlope: VarianceUtils.humanLikelihoodScore(
                stats.zipf.slope,
                VarianceUtils.HUMAN_BASELINES.zipfSlope
            ),
            ttr: VarianceUtils.humanLikelihoodScore(
                stats.vocabulary.typeTokenRatio,
                VarianceUtils.HUMAN_BASELINES.ttrNormalized
            )
        };
        
        // Average human likelihood (overall "normalcy" score)
        const hlScores = Object.values(stats.humanLikelihood);
        stats.overallHumanLikelihood = hlScores.reduce((a, b) => a + b, 0) / hlScores.length;

        // === HUMANIZER DETECTION (v2.3) ===
        // Detects AI text that has been post-processed to evade detection
        stats.humanizerSignals = this.detectHumanizerSignals(text, tokens, sentences, sentLengths, stats);

        // === AI Signature Metrics ===
        stats.aiSignatures = this.computeAISignatureMetrics(text, tokens, sentences);

        return stats;
    },

    /**
     * Detect signals that suggest text was humanized (AI + post-processing)
     * 
     * Key insight: Humanizers add first-order variance but fail to replicate
     * the natural second-order patterns (variance-of-variance, correlations,
     * autocorrelation decay) found in genuine human writing.
     */
    detectHumanizerSignals(text, tokens, sentences, sentLengths, stats) {
        const signals = {};
        
        // 1. VARIANCE STABILITY (Second-order variance)
        // Human: variable local variance. Humanized: stable variance despite high first-order variance
        const localVars = VarianceUtils.localVariance(sentLengths, 5);
        const varianceOfVariance = localVars.length > 1 ? Utils.coefficientOfVariation(localVars) : 0;
        signals.varianceOfVariance = varianceOfVariance;
        // Low variance-of-variance with high first-order variance = humanized
        signals.stableVarianceFlag = (stats.sentences.coefficientOfVariation > 0.4 && varianceOfVariance < 0.3);
        
        // 2. AUTOCORRELATION PATTERN
        // Human: gradual decay. Pure AI: periodic peaks. Humanized: flat (random noise)
        const acData = stats.autocorrelation || VarianceUtils.autocorrelation(sentLengths, 8);
        const acCoeffs = acData.coefficients?.map(c => Math.abs(c.value)) || [];
        const acVariance = acCoeffs.length > 1 ? Utils.variance(acCoeffs) : 0;
        signals.autocorrelationVariance = acVariance;
        // Very low AC variance with moderate AC values = random noise = humanized
        signals.flatAutocorrelationFlag = (acVariance < 0.01 && acData.avgAC > 0.05 && acData.avgAC < 0.25);
        
        // 3. FEATURE CORRELATION WEAKNESS
        // Human writing has correlated features (complex ideas → longer sentences → richer vocabulary)
        // Humanizers break these natural correlations
        const wordLengths = tokens.map(t => t.length);
        const localTTRs = [];
        const windowSize = Math.floor(tokens.length / 5);
        for (let i = 0; i < 5 && windowSize > 10; i++) {
            const windowTokens = tokens.slice(i * windowSize, (i + 1) * windowSize);
            localTTRs.push(new Set(windowTokens).size / windowTokens.length);
        }
        
        // Compute correlation between sentence length and local vocabulary richness
        if (localTTRs.length >= 5 && sentLengths.length >= 5) {
            const sentChunks = [];
            const chunkSize = Math.floor(sentLengths.length / 5);
            for (let i = 0; i < 5; i++) {
                sentChunks.push(Utils.mean(sentLengths.slice(i * chunkSize, (i + 1) * chunkSize)));
            }
            signals.sentenceTTRCorrelation = this.pearsonCorrelation(sentChunks, localTTRs);
        } else {
            signals.sentenceTTRCorrelation = 0.5; // neutral
        }
        // Low or negative correlation when variance is high = humanized
        signals.brokenCorrelationFlag = (stats.sentences.coefficientOfVariation > 0.4 && 
                                         Math.abs(signals.sentenceTTRCorrelation) < 0.2);
        
        // 4. WORD SOPHISTICATION CONSISTENCY
        // Humanizers do synonym substitution, creating word-level chaos in sophistication
        // Humans maintain paragraph-level sophistication consistency
        const wordRanks = tokens.map(t => this.getWordFrequencyRank(t.toLowerCase()));
        const sophVariance = Utils.coefficientOfVariation(wordRanks);
        signals.sophisticationVariance = sophVariance;
        // Very high word-level sophistication variance = synonym substitution = humanized
        signals.synonymSubstitutionFlag = sophVariance > 1.2;
        
        // 5. CONTRACTION PATTERN ANALYSIS
        // Humanizers often add contractions uniformly; humans use them contextually
        const contractionPositions = [];
        const contractionPattern = /\b(don't|won't|can't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|I'm|you're|we're|they're|he's|she's|it's|that's|there's|here's|what's|who's|let's)\b/gi;
        let match;
        while ((match = contractionPattern.exec(text)) !== null) {
            contractionPositions.push(match.index / text.length);
        }
        if (contractionPositions.length >= 3) {
            const contractionSpacing = [];
            for (let i = 1; i < contractionPositions.length; i++) {
                contractionSpacing.push(contractionPositions[i] - contractionPositions[i-1]);
            }
            signals.contractionUniformity = 1 - Utils.coefficientOfVariation(contractionSpacing);
        } else {
            signals.contractionUniformity = 0.5;
        }
        // High contraction uniformity = artificial insertion = humanized
        signals.artificialContractionFlag = (contractionPositions.length >= 3 && signals.contractionUniformity > 0.7);
        
        // 6. OVERALL HUMANIZER PROBABILITY
        const flagCount = [
            signals.stableVarianceFlag,
            signals.flatAutocorrelationFlag,
            signals.brokenCorrelationFlag,
            signals.synonymSubstitutionFlag,
            signals.artificialContractionFlag
        ].filter(Boolean).length;
        
        signals.humanizerProbability = Math.min(1, flagCount / 3); // 3+ flags = high probability
        signals.isLikelyHumanized = flagCount >= 2;
        signals.flagCount = flagCount;
        
        return signals;
    },

    /**
     * Enhanced humanizer detection that incorporates category results for contradiction analysis
     * Used in the main analyze() function after category analysis is complete
     */
    detectHumanizerSignalsWithCategories(text, sentences, advancedStats, categoryResults) {
        // Get base signals from advancedStats if available
        const baseSignals = advancedStats?.humanizerSignals || {};
        
        // Analyze category-level contradictions
        const contradictionAnalysis = this.analyzeContradictions(categoryResults);
        
        // Merge signals
        const signals = {
            ...baseSignals,
            ...contradictionAnalysis
        };
        
        // Enhanced flag detection including contradiction
        const flagCount = [
            signals.stableVarianceFlag,
            signals.flatAutocorrelationFlag,
            signals.brokenCorrelationFlag,
            signals.synonymSubstitutionFlag,
            signals.artificialContractionFlag,
            signals.categoryContradictionFlag
        ].filter(Boolean).length;
        
        // Higher probability when both structural and surface-level contradictions exist
        // But require more evidence to reduce false positives
        const contradictionBoost = contradictionAnalysis.contradictionScore > 0.7 ? 0.15 : 0;
        
        // More conservative humanizer probability calculation
        signals.humanizerProbability = Math.min(1, (flagCount / 5) + contradictionBoost);
        // Require 3+ flags to flag as humanized (was 2)
        signals.isLikelyHumanized = flagCount >= 3 && contradictionAnalysis.contradictionScore > 0.5;
        signals.flagCount = flagCount;
        
        return signals;
    },

    /**
     * Analyze contradictions in category results
     * Key patterns for humanized text:
     * - High uniformity in structure but high variance in surface features
     * - AI patterns in grammar/syntax but human patterns in personal markers
     * - Perfect mechanics but chaotic vocabulary
     */
    analyzeContradictions(categoryResults) {
        const analysis = {
            categoryContradictionFlag: false,
            contradictionScore: 0,
            contradictingCategories: []
        };
        
        if (!categoryResults || categoryResults.length < 4) {
            return analysis;
        }
        
        // Sort into AI-indicating and human-indicating categories
        const aiCategories = [];
        const humanCategories = [];
        
        for (const result of categoryResults) {
            if (result.aiProbability > 0.65) {
                aiCategories.push({ name: result.name, prob: result.aiProbability });
            } else if (result.aiProbability < 0.35) {
                humanCategories.push({ name: result.name, prob: result.aiProbability });
            }
        }
        
        // Contradiction: Having strong signals from both sides
        if (aiCategories.length >= 2 && humanCategories.length >= 2) {
            analysis.categoryContradictionFlag = true;
            
            // Score based on how polarized the results are
            const avgAiProb = aiCategories.reduce((s, c) => s + c.prob, 0) / aiCategories.length;
            const avgHumanProb = humanCategories.reduce((s, c) => s + (1 - c.prob), 0) / humanCategories.length;
            
            analysis.contradictionScore = (avgAiProb + avgHumanProb) / 2;
            analysis.contradictingCategories = {
                aiIndicators: aiCategories.map(c => c.name),
                humanIndicators: humanCategories.map(c => c.name)
            };
        }
        
        // Special contradiction patterns (highly indicative of humanization)
        const categoryMap = {};
        for (const result of categoryResults) {
            const nameLower = result.name?.toLowerCase() || '';
            categoryMap[nameLower] = result.aiProbability;
        }
        
        // Pattern 1: Perfect grammar/syntax (AI) but high personal markers (humanized)
        const grammarAI = (categoryMap['grammar'] || categoryMap['grammar patterns'] || 0.5) > 0.6;
        const authorshipHuman = (categoryMap['authorship'] || categoryMap['authorship markers'] || 0.5) < 0.4;
        
        // Pattern 2: Uniform structure (AI) but chaotic vocabulary (humanized)
        const syntaxAI = (categoryMap['syntax'] || categoryMap['syntax variance'] || 0.5) > 0.6;
        const lexicalHuman = (categoryMap['lexical'] || categoryMap['lexical diversity'] || 0.5) < 0.4;
        
        // Pattern 3: Stable tone (AI) but varied semantic fields (humanized)
        const toneAI = (categoryMap['tone'] || categoryMap['emotional tone'] || 0.5) > 0.6;
        const semanticHuman = (categoryMap['semantic'] || categoryMap['semantic analysis'] || 0.5) < 0.4;
        
        const patternFlags = [
            grammarAI && authorshipHuman,
            syntaxAI && lexicalHuman,
            toneAI && semanticHuman
        ].filter(Boolean).length;
        
        if (patternFlags >= 1) {
            analysis.categoryContradictionFlag = true;
            analysis.contradictionScore = Math.max(analysis.contradictionScore, 0.4 + patternFlags * 0.15);
        }
        
        return analysis;
    },

    /**
     * Pearson correlation coefficient
     */
    pearsonCorrelation(x, y) {
        if (!x || !y || x.length !== y.length || x.length < 2) return 0;
        
        const n = x.length;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const sumY2 = y.reduce((sum, yi) => sum + yi * yi, 0);
        
        const numerator = n * sumXY - sumX * sumY;
        const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
        
        return denominator === 0 ? 0 : numerator / denominator;
    },

    /**
     * Get approximate word frequency rank (lower = more common)
     * Uses a simplified frequency list based on common English words
     */
    getWordFrequencyRank(word) {
        // Top 100 most common English words get low ranks
        const commonWords = new Set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
            'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
            'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other',
            'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way',
            'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us'
        ]);
        
        if (commonWords.has(word)) return 50;
        if (word.length <= 3) return 100;
        if (word.length <= 5) return 200;
        if (word.length <= 7) return 400;
        return 600 + word.length * 50; // Longer/rarer words get higher ranks
    },

    /**
     * Find repeated phrases (n-grams of 4+ words appearing multiple times)
     * Research shows this is a strong AI indicator: AI often reuses exact phrases
     */
    findRepeatedPhrases(tokens) {
        const phrases = {
            count: 0,
            phrases: [],
            score: 0
        };
        
        if (tokens.length < 8) return phrases;
        
        // Check 4-grams, 5-grams, and 6-grams
        for (let n = 4; n <= 6; n++) {
            const ngrams = Utils.ngrams(tokens, n);
            const counts = {};
            
            // ngrams are already joined strings from Utils.ngrams
            ngrams.forEach(gram => {
                const key = gram.toLowerCase();
                counts[key] = (counts[key] || 0) + 1;
            });
            
            // Find n-grams appearing 2+ times
            Object.entries(counts).forEach(([phrase, count]) => {
                if (count >= 2) {
                    phrases.phrases.push({ 
                        phrase, 
                        count, 
                        length: n,
                        weight: n * count // Longer repeated phrases are more suspicious
                    });
                    phrases.count++;
                }
            });
        }
        
        // Sort by weight (longer and more frequent = more suspicious)
        phrases.phrases.sort((a, b) => b.weight - a.weight);
        
        // Calculate overall score (0-1): higher = more likely AI
        const totalWeight = phrases.phrases.reduce((sum, p) => sum + p.weight, 0);
        phrases.score = Math.min(1, totalWeight / (tokens.length * 0.02));
        
        return phrases;
    },

    /**
     * Analyze word patterns (POS-like analysis without external dependencies)
     * Research shows AI text has different verb/noun/adjective distributions
     */
    analyzeWordPatterns(tokens, text) {
        const patterns = {};
        const lowerTokens = tokens.map(t => t.toLowerCase());
        
        // Common word category sets (approximate POS without external tools)
        const verbEndings = ['ing', 'ed', 'ize', 'ise', 'ate', 'ify'];
        const adjEndings = ['ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ial', 'ic', 'ical'];
        const advEndings = ['ly'];
        const nounEndings = ['tion', 'sion', 'ment', 'ness', 'ity', 'ty', 'er', 'or', 'ist', 'ism'];
        
        // Definite articles and determiners
        const determiners = new Set(['the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'some', 'any', 'no', 'every', 'each', 'all', 'both', 'few', 'many', 'much', 'most']);
        
        // Personal pronouns
        const personalPronouns = new Set(['i', 'me', 'my', 'mine', 'myself', 'you', 'your', 'yours', 'yourself', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'we', 'us', 'our', 'ours', 'ourselves', 'they', 'them', 'their', 'theirs', 'themselves']);
        
        // First-person pronouns (humans use more)
        const firstPerson = new Set(['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves']);
        
        // Hedging words (AI uses more)
        const hedgingWords = new Set(['perhaps', 'maybe', 'possibly', 'potentially', 'likely', 'unlikely', 'probably', 'certainly', 'definitely', 'generally', 'typically', 'usually', 'often', 'sometimes', 'occasionally', 'rarely', 'seldom', 'apparently', 'seemingly', 'arguably']);
        
        // Count by category
        let verbLike = 0, adjLike = 0, advLike = 0, nounLike = 0;
        let determinerCount = 0, pronounCount = 0, firstPersonCount = 0, hedgingCount = 0;
        
        lowerTokens.forEach(token => {
            if (determiners.has(token)) determinerCount++;
            if (personalPronouns.has(token)) pronounCount++;
            if (firstPerson.has(token)) firstPersonCount++;
            if (hedgingWords.has(token)) hedgingCount++;
            
            for (const ending of verbEndings) {
                if (token.endsWith(ending) && token.length > ending.length + 2) { verbLike++; break; }
            }
            for (const ending of adjEndings) {
                if (token.endsWith(ending) && token.length > ending.length + 2) { adjLike++; break; }
            }
            for (const ending of advEndings) {
                if (token.endsWith(ending) && token.length > ending.length + 2) { advLike++; break; }
            }
            for (const ending of nounEndings) {
                if (token.endsWith(ending) && token.length > ending.length + 2) { nounLike++; break; }
            }
        });
        
        const total = Math.max(1, tokens.length);
        
        patterns.verbRatio = verbLike / total;
        patterns.adjectiveRatio = adjLike / total;
        patterns.adverbRatio = advLike / total;
        patterns.nounRatio = nounLike / total;
        patterns.determinerRatio = determinerCount / total;
        patterns.pronounRatio = pronounCount / total;
        patterns.firstPersonRatio = firstPersonCount / total;  // Low in AI text
        patterns.hedgingRatio = hedgingCount / total;  // High in AI text
        
        // Ratio of content words to function words approximation
        patterns.contentDensity = (verbLike + adjLike + nounLike) / total;
        
        // Sentence starters analysis
        const starters = Utils.splitSentences(text).map(s => {
            const words = Utils.tokenize(s);
            return words[0]?.toLowerCase();
        }).filter(Boolean);
        
        // Diversity of sentence starters (humans have more variety)
        const uniqueStarters = new Set(starters).size;
        patterns.starterDiversity = starters.length > 0 ? uniqueStarters / starters.length : 0;
        
        // Common AI starters
        const aiStarters = new Set(['the', 'this', 'it', 'in', 'as', 'there', 'when', 'while', 'although', 'however', 'moreover', 'furthermore', 'additionally']);
        const aiStarterCount = starters.filter(s => aiStarters.has(s)).length;
        patterns.aiStarterRatio = starters.length > 0 ? aiStarterCount / starters.length : 0;
        
        return patterns;
    },

    /**
     * Bin continuous values into histogram buckets
     */
    binValues(values, numBins) {
        if (!values || values.length === 0) return [];
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / numBins || 1;
        
        const bins = new Array(numBins).fill(0);
        
        values.forEach(v => {
            const binIndex = Math.min(numBins - 1, Math.floor((v - min) / binWidth));
            bins[binIndex]++;
        });
        
        return bins;
    },
    /**
     * Compute AI-specific signature metrics
     */
    computeAISignatureMetrics(text, tokens, sentences) {
        const metrics = {};

        // Hedging word density
        const hedgingCount = Utils.hedgingWords.reduce((count, word) => {
            return count + (text.toLowerCase().match(new RegExp(`\\b${word}\\b`, 'gi')) || []).length;
        }, 0);
        metrics.hedgingDensity = tokens.length > 0 ? hedgingCount / tokens.length : 0;

        // Discourse marker density
        const discourseCount = Utils.discourseMarkers.reduce((count, marker) => {
            return count + (text.toLowerCase().includes(marker.toLowerCase()) ? 1 : 0);
        }, 0);
        metrics.discourseMarkerDensity = sentences.length > 0 ? discourseCount / sentences.length : 0;

        // Unicode anomalies (strong AI indicator)
        const unicodeAnomalies = (text.match(/[\u2014\u2013\u2018\u2019\u201C\u201D\u2026\u2E3A\u2E3B]/g) || []).length;
        metrics.unicodeAnomalyDensity = text.length > 0 ? unicodeAnomalies / text.length * 1000 : 0;

        // Decorative dividers (very strong AI indicator)
        const decorativeDividers = (text.match(/[⸻═━─]{3,}|[•●◦◆◇■□▪▫]{3,}/g) || []).length;
        metrics.decorativeDividerCount = decorativeDividers;

        // Perfect grammar indicators (lack of contractions)
        const contractionCount = (text.match(/\b(don't|won't|can't|wouldn't|couldn't|shouldn't|isn't|aren't|wasn't|weren't|haven't|hasn't|hadn't|I'm|you're|we're|they're|he's|she's|it's|that's|there's|here's|what's|who's|let's)\b/gi) || []).length;
        metrics.contractionRate = sentences.length > 0 ? contractionCount / sentences.length : 0;

        // Sentence starter variety
        const starters = sentences.map(s => Utils.tokenize(s)[0]?.toLowerCase()).filter(Boolean);
        const uniqueStarters = new Set(starters).size;
        metrics.sentenceStarterVariety = starters.length > 0 ? uniqueStarters / starters.length : 0;

        // Passive voice indicators
        const passiveIndicators = (text.match(/\b(is|are|was|were|been|being)\s+(being\s+)?\w+ed\b/gi) || []).length;
        metrics.passiveVoiceRate = sentences.length > 0 ? passiveIndicators / sentences.length : 0;

        return metrics;
    },

    /**
     * Estimate total syllables in tokens
     */
    estimateSyllables(tokens) {
        return tokens.reduce((total, token) => total + this.syllableCount(token), 0);
    },

    /**
     * Estimate syllable count for a word
     */
    syllableCount(word) {
        word = word.toLowerCase().replace(/[^a-z]/g, '');
        if (word.length <= 3) return 1;
        
        // Count vowel groups
        const vowels = word.match(/[aeiouy]+/g);
        let count = vowels ? vowels.length : 1;
        
        // Subtract silent e
        if (word.endsWith('e') && !word.endsWith('le')) count--;
        
        // Ensure at least 1 syllable
        return Math.max(1, count);
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
     * 
     * ENHANCED v2.3: Improved signal aggregation
     * - Strong individual AI signals should have more impact
     * - Multiple agreeing signals compound their effect
     * - Human signals can only reduce AI probability proportionally
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
                featureContributions: [],
                combinationMethod: 'none'
            };
        }

        // Calculate feature contributions using category weights
        const featureContributions = [];
        const probabilities = [];
        const weights = [];
        
        // Track strong signals separately
        let strongAiSignalCount = 0;
        let strongHumanSignalCount = 0;
        let aiWeightedSum = 0;
        let humanWeightedSum = 0;
        let totalAiWeight = 0;
        let totalHumanWeight = 0;

        for (const result of validResults) {
            // Map categories to weight keys
            const weightKey = this.getCategoryWeightKey(result.category, result.name);
            const baseWeight = this.categoryWeights[weightKey] || 0.1;
            
            // Apply confidence as a weight multiplier
            const effectiveWeight = baseWeight * result.confidence;
            
            probabilities.push(result.aiProbability);
            weights.push(effectiveWeight);
            
            // Count strong signals and track weighted contributions
            if (result.aiProbability > 0.65) {
                strongAiSignalCount++;
                aiWeightedSum += result.aiProbability * effectiveWeight;
                totalAiWeight += effectiveWeight;
            } else if (result.aiProbability < 0.35) {
                strongHumanSignalCount++;
                humanWeightedSum += (1 - result.aiProbability) * effectiveWeight;
                totalHumanWeight += effectiveWeight;
            }
            
            featureContributions.push({
                category: result.category,
                name: result.name,
                aiProbability: result.aiProbability,
                weight: effectiveWeight,
                contribution: result.aiProbability * effectiveWeight,
                uniformityScores: result.uniformityScores || null
            });
        }

        // Calculate base weighted average
        const weightedAvgProb = weights.reduce((sum, w, i) => sum + w * probabilities[i], 0) / 
                               weights.reduce((sum, w) => sum + w, 0);
        
        // Apply signal majority adjustment
        // When AI signals dominate, boost the probability
        // When human signals dominate, reduce it
        let adjustedProbability = weightedAvgProb;
        
        const aiSignalRatio = validResults.length > 0 
            ? probabilities.filter(p => p > 0.55).length / validResults.length 
            : 0.5;
        
        // Strong AI indicator presence boost
        if (strongAiSignalCount >= 3 && strongAiSignalCount > strongHumanSignalCount) {
            // Multiple strong AI signals - compound effect
            const avgStrongAiProb = totalAiWeight > 0 ? aiWeightedSum / totalAiWeight : 0.5;
            const boostFactor = Math.min(0.25, strongAiSignalCount * 0.05);
            adjustedProbability = weightedAvgProb + (avgStrongAiProb - weightedAvgProb) * boostFactor * 2;
        }
        
        // When very high percentage of categories indicate AI, push toward higher probability
        if (aiSignalRatio > 0.7) {
            const pushFactor = (aiSignalRatio - 0.5) * 0.5;
            adjustedProbability = adjustedProbability + (1 - adjustedProbability) * pushFactor;
        }
        
        // Bayesian combination for additional signal
        const bayesianProb = VarianceUtils.bayesianCombine(probabilities, weights);
        
        // Use the higher of adjusted weighted or Bayesian when AI signals dominate
        let finalProbability;
        if (strongAiSignalCount > strongHumanSignalCount * 2) {
            // AI signals strongly dominate - use more aggressive estimate
            finalProbability = Math.max(adjustedProbability, bayesianProb);
        } else if (strongHumanSignalCount > strongAiSignalCount * 2) {
            // Human signals strongly dominate - use more conservative estimate
            finalProbability = Math.min(adjustedProbability, bayesianProb);
        } else {
            // Mixed signals - blend approaches
            finalProbability = (adjustedProbability + bayesianProb) / 2;
        }
        
        // Apply floor/ceiling based on strong signal counts
        // If 4+ strong AI signals with high confidence, minimum 50%
        if (strongAiSignalCount >= 4 && finalProbability < 0.5) {
            finalProbability = 0.5 + (strongAiSignalCount - 4) * 0.05;
        }
        
        // Calculate calibrated confidence based on multiple factors
        const confidence = this.calculateCalibratedConfidence(validResults, probabilities);
        
        // Mixed probability: higher when analyzers disagree
        const stdDev = Utils.standardDeviation(probabilities);
        const mixedProbability = Math.min(0.35, stdDev * 1.5);

        // Build variance profile
        const varianceProfile = this.buildVarianceProfile(validResults);

        return {
            aiProbability: Math.max(0, Math.min(1, finalProbability)),
            confidence,
            mixedProbability,
            varianceProfile,
            featureContributions: featureContributions.sort((a, b) => b.contribution - a.contribution),
            combinationMethod: 'signal-weighted',
            rawBayesian: bayesianProb,
            rawWeighted: weightedAvgProb,
            signalCounts: {
                strongAi: strongAiSignalCount,
                strongHuman: strongHumanSignalCount,
                aiRatio: aiSignalRatio
            }
        };
    },

    /**
     * Calculate calibrated confidence score
     * Based on: analyzer agreement, sample sizes, signal strength
     */
    calculateCalibratedConfidence(results, probabilities) {
        // Factor 1: Analyzer agreement (low std dev = high agreement)
        const stdDev = Utils.standardDeviation(probabilities);
        const agreementScore = 1 - Math.min(1, stdDev * 2.5);
        
        // Factor 2: Average analyzer confidence
        const avgAnalyzerConf = Utils.mean(results.map(r => r.confidence));
        
        // Factor 3: Signal strength (distance from 0.5)
        const avgProb = Utils.mean(probabilities);
        const signalStrength = Math.abs(avgProb - 0.5) * 2;
        
        // Factor 4: Number of analyzers (more = more confident)
        const analyzerCountFactor = Math.min(1, results.length / 10);
        
        // Factor 5: Consistency of direction (all pointing same way)
        const aiLeaning = probabilities.filter(p => p > 0.5).length / probabilities.length;
        const directionConsistency = Math.abs(aiLeaning - 0.5) * 2;
        
        // Weighted combination
        const calibratedConfidence = (
            agreementScore * 0.25 +
            avgAnalyzerConf * 0.25 +
            signalStrength * 0.15 +
            analyzerCountFactor * 0.15 +
            directionConsistency * 0.20
        );
        
        return Math.max(0, Math.min(1, calibratedConfidence));
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
        if (nameLower.includes('metadata') || nameLower.includes('formatting') || category === 11) return 'metadataFormatting';
        
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
        const lowerText = text.toLowerCase();
        
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
        
        // Check for instructional/educational tone (often false flagged)
        const instructionalMarkers = [
            'step 1', 'step 2', 'first,', 'second,', 'third,', 'finally,',
            'how to', 'in order to', 'you can', 'you should', 'make sure to',
            'remember to', 'don\'t forget', 'keep in mind', 'note that'
        ];
        const instructionalCount = instructionalMarkers.filter(m => lowerText.includes(m)).length;
        if (instructionalCount >= 3) {
            risks.push({
                type: 'instructional_tone',
                message: 'Instructional/how-to content naturally uses structured language similar to AI',
                severity: 'medium',
                adjustWeight: 0.85 // Reduce AI probability weight
            });
        }
        
        // Check for business/professional tone
        const businessMarkers = [
            'dear', 'sincerely', 'regards', 'best regards', 'thank you for',
            'please let me know', 'i am writing to', 'as discussed',
            'per our conversation', 'attached', 'following up'
        ];
        const businessCount = businessMarkers.filter(m => lowerText.includes(m)).length;
        if (businessCount >= 2) {
            risks.push({
                type: 'business_correspondence',
                message: 'Professional/business writing often has formal patterns similar to AI',
                severity: 'medium',
                adjustWeight: 0.85
            });
        }
        
        // Check for journalistic/news style
        const journalisticMarkers = [
            'according to', 'sources say', 'reported that', 'in a statement',
            'breaking:', 'update:', 'developing story'
        ];
        const journalisticCount = journalisticMarkers.filter(m => lowerText.includes(m)).length;
        if (journalisticCount >= 2) {
            risks.push({
                type: 'journalistic_style',
                message: 'News/journalistic writing has structured patterns that may trigger false positives',
                severity: 'low',
                adjustWeight: 0.9
            });
        }
        
        // Check for ESL/non-native speaker patterns
        // These often get flagged because of simpler vocabulary choices
        const sentences = Utils.splitSentences(text);
        const avgSentenceLength = sentences.length > 0 ? wordCount / sentences.length : 0;
        const uniqueWords = new Set(Utils.tokenize(lowerText)).size;
        const vocabDiversity = wordCount > 0 ? uniqueWords / wordCount : 0;
        
        if (avgSentenceLength < 12 && vocabDiversity < 0.5 && wordCount > 50) {
            risks.push({
                type: 'simple_vocabulary',
                message: 'Simple vocabulary and short sentences may indicate ESL writer, not AI',
                severity: 'medium',
                adjustWeight: 0.85
            });
        }
        
        // Check for creative/narrative writing
        const narrativeMarkers = [
            'i felt', 'i thought', 'i wondered', 'i remember', 'i realized',
            'she said', 'he said', 'they said', 'my heart', 'my mind'
        ];
        const narrativeCount = narrativeMarkers.filter(m => lowerText.includes(m)).length;
        if (narrativeCount >= 3) {
            risks.push({
                type: 'narrative_writing',
                message: 'Personal narrative/creative writing detected - reduces false positive likelihood',
                severity: 'low',
                isHumanIndicator: true
            });
        }

        // Check for disagreement between analyzers - may indicate humanized AI
        const validResults = categoryResults.filter(r => r.confidence > 0.3);
        if (validResults.length > 3) {
            const probs = validResults.map(r => r.aiProbability);
            const spread = Math.max(...probs) - Math.min(...probs);
            if (spread > 0.4) {
                risks.push({
                    type: 'analyzer_disagreement',
                    message: 'High disagreement between detection methods — this pattern is common in humanized AI text. See detailed report for more context.',
                    severity: 'high',
                    suggestsHumanized: true
                });
            }
        }
        
        // Calculate overall adjustment weight based on risks
        let adjustmentWeight = 1.0;
        let humanIndicatorCount = 0;
        for (const risk of risks) {
            if (risk.adjustWeight) {
                adjustmentWeight *= risk.adjustWeight;
            }
            if (risk.isHumanIndicator) {
                humanIndicatorCount++;
            }
        }

        return {
            hasRisks: risks.length > 0,
            risks,
            overallRisk: risks.length === 0 ? 'low' : 
                         risks.some(r => r.severity === 'high') ? 'high' : 'medium',
            adjustmentWeight,
            humanIndicatorCount
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
     * Enhanced verdict with humanizer detection
     * Identifies three states: Human, AI, or Humanized (AI modified by human or tool)
     * 
     * Key insight: Contradictory signals (AI structural patterns + human-like surface features)
     * often indicate AI text that has been run through a "humanizer" tool or manually edited.
     */
    getVerdictWithHumanizer(aiProbability, confidence = 0.5, humanizerSignals = null, signalCounts = null) {
        // Check for humanizer signals first - this takes precedence
        const isHumanized = humanizerSignals?.isLikelyHumanized || false;
        const humanizerProb = humanizerSignals?.humanizerProbability || 0;
        const flagCount = humanizerSignals?.flagCount || 0;
        
        // Detect contradiction: Strong AI signals in some areas but human-like in others
        const hasContradiction = signalCounts && 
            signalCounts.strongAi >= 2 && 
            signalCounts.strongHuman >= 2;
        
        // Calculate contradiction score (higher = more contradictory)
        let contradictionScore = 0;
        if (signalCounts) {
            const totalStrong = signalCounts.strongAi + signalCounts.strongHuman;
            if (totalStrong > 0) {
                // Maximum contradiction when equal strong signals from both sides
                const balance = Math.min(signalCounts.strongAi, signalCounts.strongHuman) / Math.max(1, totalStrong / 2);
                contradictionScore = balance * Math.min(1, totalStrong / 6);
            }
        }
        
        // Confidence qualifier
        const confidenceLevel = confidence < 0.4 ? 'Low confidence: ' : 
                               confidence < 0.7 ? '' : 
                               'High confidence: ';

        // HUMANIZED DETECTION
        // Only flag as humanized when there's strong evidence
        // Conditions for "Humanized" verdict:
        // 1. High humanizer probability (3+ flags from second-order analysis)
        // 2. Very strong contradiction (many strong AI signals AND many strong human signals)
        // 3. AI probability in the higher mixed range with strong contradiction evidence
        
        const humanizerThreshold = 0.6; // Requires 3+ flags out of 5 (was 0.4)
        const contradictionThreshold = 0.6; // Higher threshold (was 0.4)
        const isContradictory = contradictionScore > contradictionThreshold && hasContradiction;
        
        // Only flag humanized if: explicit strong humanizer signals AND supporting evidence
        // Be more conservative to avoid false "humanized" flags on normal AI text
        const shouldFlagHumanized = (
            (isHumanized && flagCount >= 3) ||
            (humanizerProb >= humanizerThreshold && aiProbability >= 0.50) ||
            (aiProbability >= 0.50 && aiProbability <= 0.70 && isContradictory && flagCount >= 2)
        );
        
        if (shouldFlagHumanized) {
            
            // Determine humanizer confidence level
            const humanizerConfidence = Math.max(humanizerProb, contradictionScore);
            const humanizerLevel = humanizerConfidence > 0.7 ? 'High confidence: ' : 
                                  humanizerConfidence > 0.5 ? '' : 'Possible: ';
            
            return {
                label: humanizerLevel + 'Possibly Humanized',
                description: 'This text shows patterns consistent with AI-generated content that may have been modified — consider verifying with additional analysis',
                level: 'humanized',
                band: 'humanized',
                probability: aiProbability,
                confidence,
                humanizerDetails: {
                    humanizerProbability: humanizerProb,
                    contradictionScore,
                    flagCount,
                    signals: {
                        stableVariance: humanizerSignals?.stableVarianceFlag || false,
                        flatAutocorrelation: humanizerSignals?.flatAutocorrelationFlag || false,
                        brokenCorrelation: humanizerSignals?.brokenCorrelationFlag || false,
                        synonymSubstitution: humanizerSignals?.synonymSubstitutionFlag || false,
                        artificialContraction: humanizerSignals?.artificialContractionFlag || false
                    }
                }
            };
        }
        
        // Standard probability bands
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
