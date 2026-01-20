/**
 * VERITAS — Lexical Choice & Vocabulary Analyzer
 * Category 3: Lexical Diversity, Word Frequency, Register Consistency
 */

const LexicalAnalyzer = {
    name: 'Lexical Choice & Vocabulary',
    category: 3,
    weight: 1.2,

    // Common mid-frequency words AI overuses
    aiOverusedWords: [
        'crucial', 'essential', 'fundamental', 'significant', 'substantial',
        'comprehensive', 'extensive', 'remarkable', 'notable', 'particularly',
        'specifically', 'effectively', 'efficiently', 'subsequently', 'consequently',
        'additionally', 'furthermore', 'moreover', 'therefore', 'thus',
        'various', 'numerous', 'multiple', 'diverse', 'distinct',
        'enhance', 'leverage', 'utilize', 'implement', 'facilitate',
        'demonstrate', 'illustrate', 'highlight', 'emphasize', 'underscore',
        'delve', 'navigate', 'foster', 'cultivate', 'streamline',
        'robust', 'seamless', 'holistic', 'innovative', 'dynamic'
    ],

    // Academic register words
    academicWords: [
        'methodology', 'paradigm', 'framework', 'discourse', 'context',
        'perspective', 'implications', 'phenomena', 'analysis', 'synthesis',
        'correlation', 'hypothesis', 'empirical', 'theoretical', 'conceptual'
    ],

    // Conversational words
    conversationalWords: [
        'kinda', 'gonna', 'wanna', 'gotta', 'yeah', 'nope', 'okay',
        'basically', 'literally', 'actually', 'honestly', 'seriously',
        'stuff', 'things', 'like', 'just', 'really', 'pretty', 'super'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const tokens = Utils.tokenize(text);
        const sentences = Utils.splitSentences(text);
        
        if (tokens.length < 10) {
            return this.getEmptyResult();
        }

        // Calculate lexical diversity metrics
        const diversityMetrics = this.calculateDiversityMetrics(tokens);
        
        // Analyze word frequency patterns
        const frequencyAnalysis = this.analyzeWordFrequency(tokens);
        
        // Check for AI-typical vocabulary
        const aiVocabAnalysis = this.analyzeAIVocabulary(text, tokens);
        
        // Analyze register consistency
        const registerAnalysis = this.analyzeRegister(text, tokens);

        // Calculate AI probability
        // Low diversity = AI-like
        // High mid-frequency usage = AI-like
        // AI vocabulary present = AI-like
        // Consistent (overly formal) register = AI-like

        const scores = {
            diversityLow: 1 - diversityMetrics.normalizedTTR,
            vocabularyFlattening: frequencyAnalysis.flatteningScore,
            aiVocabulary: aiVocabAnalysis.aiVocabScore,
            registerUniformity: registerAnalysis.uniformityScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.diversityLow, scores.vocabularyFlattening, scores.aiVocabulary, scores.registerUniformity],
            [0.25, 0.2, 0.35, 0.2]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(tokens.length),
            details: {
                diversityMetrics,
                frequencyAnalysis,
                aiVocabAnalysis,
                registerAnalysis
            },
            findings: this.generateFindings(text, diversityMetrics, frequencyAnalysis, aiVocabAnalysis, registerAnalysis),
            scores
        };
    },

    /**
     * Calculate lexical diversity metrics
     */
    calculateDiversityMetrics(tokens) {
        const uniqueTokens = new Set(tokens);
        const typeCount = uniqueTokens.size;
        const tokenCount = tokens.length;
        
        // Type-Token Ratio
        const ttr = tokenCount > 0 ? typeCount / tokenCount : 0;
        
        // Root TTR (Guiraud's R) - corrects for text length
        const rootTTR = tokenCount > 0 ? typeCount / Math.sqrt(tokenCount) : 0;
        
        // MTLD approximation (Mean Segmental TTR)
        const mtld = this.calculateMTLD(tokens);
        
        // Word entropy
        const wordFreq = Utils.frequencyDistribution(tokens);
        const entropy = Utils.entropy(wordFreq);
        const maxEntropy = Math.log2(typeCount);
        const normalizedEntropy = maxEntropy > 0 ? entropy / maxEntropy : 0;

        // Normalize TTR for scoring (higher = more diverse = more human)
        // Typical AI TTR: 0.3-0.5, Human: 0.4-0.7
        const normalizedTTR = Utils.normalize(ttr, 0.25, 0.65);

        return {
            typeCount,
            tokenCount,
            ttr: ttr.toFixed(3),
            rootTTR: rootTTR.toFixed(2),
            mtld: mtld.toFixed(1),
            entropy: entropy.toFixed(2),
            normalizedEntropy: normalizedEntropy.toFixed(2),
            normalizedTTR
        };
    },

    /**
     * Calculate MTLD (Measure of Textual Lexical Diversity)
     */
    calculateMTLD(tokens) {
        if (tokens.length < 10) return 0;
        
        const threshold = 0.72;
        let factors = 0;
        let factorCount = 0;
        let currentTypes = new Set();
        
        for (const token of tokens) {
            currentTypes.add(token);
            const currentTTR = currentTypes.size / (factors + 1);
            
            if (currentTTR <= threshold) {
                factorCount++;
                factors = 0;
                currentTypes = new Set();
            }
            factors++;
        }
        
        // Handle remainder
        if (factors > 0) {
            const partialFactor = (1 - (currentTypes.size / factors)) / (1 - threshold);
            factorCount += partialFactor;
        }
        
        return tokens.length / Math.max(1, factorCount);
    },

    /**
     * Analyze word frequency patterns
     */
    analyzeWordFrequency(tokens) {
        // Filter out function words for content word analysis
        const contentWords = tokens.filter(t => !Utils.functionWords.has(t) && t.length > 2);
        const wordFreq = Utils.frequencyDistribution(contentWords);
        
        // Get frequency distribution
        const frequencies = Object.values(wordFreq);
        const totalContentWords = contentWords.length;
        
        // Check for mid-frequency dominance (AI pattern)
        // AI tends to avoid very rare words and overuse mid-frequency words
        const hapaxLegomena = frequencies.filter(f => f === 1).length; // Words appearing once
        const hapaxRatio = Object.keys(wordFreq).length > 0 
            ? hapaxLegomena / Object.keys(wordFreq).length 
            : 0;
        
        // Low hapax ratio = vocabulary flattening = AI-like
        // Human text typically has 40-60% hapax legomena
        const flatteningScore = 1 - Utils.normalize(hapaxRatio, 0.2, 0.6);
        
        // Top words analysis
        const topWords = Utils.topN(wordFreq, 10);

        return {
            uniqueContentWords: Object.keys(wordFreq).length,
            totalContentWords,
            hapaxLegomena,
            hapaxRatio: hapaxRatio.toFixed(2),
            flatteningScore,
            topWords
        };
    },

    /**
     * Analyze AI-typical vocabulary
     */
    analyzeAIVocabulary(text, tokens) {
        const lower = text.toLowerCase();
        const foundAIWords = [];
        let aiWordCount = 0;
        
        for (const word of this.aiOverusedWords) {
            const pattern = new RegExp(`\\b${word}\\b`, 'gi');
            const matches = lower.match(pattern);
            if (matches) {
                aiWordCount += matches.length;
                foundAIWords.push({ word, count: matches.length });
            }
        }
        
        // Sort by frequency
        foundAIWords.sort((a, b) => b.count - a.count);
        
        // Score: ratio of AI words to total tokens
        const aiWordRatio = tokens.length > 0 ? aiWordCount / tokens.length : 0;
        const aiVocabScore = Utils.normalize(aiWordRatio, 0, 0.05);

        // Check for synonym rotation (AI pattern)
        const synonymGroups = this.detectSynonymRotation(text);

        return {
            aiWordCount,
            aiWordRatio: (aiWordRatio * 100).toFixed(2) + '%',
            aiVocabScore,
            foundAIWords: foundAIWords.slice(0, 10),
            synonymRotation: synonymGroups
        };
    },

    /**
     * Detect excessive synonym rotation
     */
    detectSynonymRotation(text) {
        const synonymGroups = [
            ['important', 'crucial', 'essential', 'vital', 'critical', 'significant'],
            ['show', 'demonstrate', 'illustrate', 'exhibit', 'display', 'reveal'],
            ['use', 'utilize', 'employ', 'leverage', 'harness', 'apply'],
            ['help', 'assist', 'aid', 'facilitate', 'support', 'enable'],
            ['improve', 'enhance', 'boost', 'strengthen', 'optimize', 'augment'],
            ['big', 'large', 'substantial', 'considerable', 'significant', 'extensive']
        ];
        
        const lower = text.toLowerCase();
        const rotationPatterns = [];
        
        for (const group of synonymGroups) {
            const found = group.filter(word => 
                new RegExp(`\\b${word}\\b`, 'i').test(lower)
            );
            if (found.length >= 3) {
                rotationPatterns.push({
                    group: group[0] + ' variants',
                    found: found,
                    count: found.length
                });
            }
        }
        
        return rotationPatterns;
    },

    /**
     * Analyze register consistency
     */
    analyzeRegister(text, tokens) {
        const lower = text.toLowerCase();
        
        // Count academic vs conversational markers
        let academicCount = 0;
        let conversationalCount = 0;
        const academicFound = [];
        const conversationalFound = [];
        
        for (const word of this.academicWords) {
            if (lower.includes(word)) {
                academicCount++;
                academicFound.push(word);
            }
        }
        
        for (const word of this.conversationalWords) {
            const pattern = new RegExp(`\\b${word}\\b`, 'gi');
            if (pattern.test(lower)) {
                conversationalCount++;
                conversationalFound.push(word);
            }
        }
        
        // Determine primary register
        const totalMarkers = academicCount + conversationalCount;
        let primaryRegister = 'neutral';
        let registerBalance = 0.5;
        
        if (totalMarkers > 0) {
            registerBalance = academicCount / totalMarkers;
            if (registerBalance > 0.7) primaryRegister = 'academic';
            else if (registerBalance < 0.3) primaryRegister = 'conversational';
            else primaryRegister = 'mixed';
        }
        
        // AI tends to be uniformly formal/academic
        // Lack of conversational elements + high academic = AI-like
        const uniformityScore = academicCount > 0 && conversationalCount === 0
            ? 0.8
            : (1 - Utils.normalize(conversationalCount, 0, 5)) * 0.5;

        return {
            academicCount,
            conversationalCount,
            academicFound,
            conversationalFound,
            primaryRegister,
            registerBalance: registerBalance.toFixed(2),
            uniformityScore
        };
    },

    /**
     * Generate findings with detailed statistics
     */
    generateFindings(text, diversityMetrics, frequencyAnalysis, aiVocabAnalysis, registerAnalysis) {
        const findings = [];
        
        // Helper to extract excerpts around a word
        const extractExcerpts = (searchWords, maxExcerpts = 5) => {
            const excerpts = [];
            let instanceNum = 0;
            
            for (const word of searchWords) {
                if (typeof word !== 'string') continue;
                const regex = new RegExp(`([^.!?]*\\b${word}\\b[^.!?]*[.!?]?)`, 'gi');
                let match;
                while ((match = regex.exec(text)) !== null && excerpts.length < maxExcerpts) {
                    instanceNum++;
                    const sentence = match[1].trim();
                    if (sentence.length > 10) {
                        excerpts.push({
                            word: word,
                            context: sentence.length > 100 ? sentence.substring(0, 100) + '...' : sentence,
                            explanation: `Usage of "${word}"`,
                            instance: instanceNum
                        });
                    }
                }
            }
            return excerpts;
        };

        // Lexical diversity
        const ttr = parseFloat(diversityMetrics.ttr);
        if (ttr < 0.35) {
            findings.push({
                label: 'Lexical Diversity',
                value: 'Low type-token ratio detected',
                note: `Limited vocabulary variety indicates potential AI generation`,
                indicator: 'ai',
                severity: ttr < 0.25 ? 'high' : 'medium',
                stats: {
                    measured: `TTR: ${(ttr * 100).toFixed(1)}%`,
                    uniqueWords: diversityMetrics.typeCount,
                    totalWords: diversityMetrics.tokenCount,
                    rootTTR: diversityMetrics.rootTTR,
                    mtld: diversityMetrics.mtld,
                    entropy: `${diversityMetrics.entropy} bits`
                },
                benchmark: {
                    humanRange: 'TTR: 40%–70%',
                    aiRange: 'TTR: 25%–40%',
                    interpretation: 'Higher TTR = more diverse vocabulary = more human-like'
                }
            });
        } else if (ttr > 0.55) {
            findings.push({
                label: 'Lexical Diversity',
                value: 'Rich vocabulary diversity detected',
                note: `High vocabulary variety suggests human authorship`,
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `TTR: ${(ttr * 100).toFixed(1)}%`,
                    uniqueWords: diversityMetrics.typeCount,
                    totalWords: diversityMetrics.tokenCount
                },
                benchmark: {
                    humanRange: 'TTR: 40%–70%',
                    aiRange: 'TTR: 25%–40%'
                }
            });
        }

        // Vocabulary flattening
        if (frequencyAnalysis.flatteningScore > 0.6) {
            findings.push({
                label: 'Vocabulary Flattening',
                value: 'Signs of vocabulary flattening detected',
                note: `AI tends to use mid-frequency words more evenly`,
                indicator: 'ai',
                severity: frequencyAnalysis.flatteningScore > 0.75 ? 'high' : 'medium',
                stats: {
                    measured: `Flattening Score: ${(frequencyAnalysis.flatteningScore * 100).toFixed(1)}%`,
                    hapaxRatio: `${(frequencyAnalysis.hapaxRatio * 100).toFixed(1)}%`,
                    interpretation: 'Low hapax ratio = fewer unique single-use words'
                },
                benchmark: {
                    humanRange: 'Hapax: 45%–65%',
                    aiRange: 'Hapax: 25%–40%',
                    note: 'Hapax legomena = words appearing exactly once'
                }
            });
        }

        // AI vocabulary
        if (aiVocabAnalysis.aiVocabScore > 0.4 && aiVocabAnalysis.foundAIWords.length > 0) {
            const topAI = aiVocabAnalysis.foundAIWords.slice(0, 5);
            const excerptWords = topAI.map(w => w.word);
            findings.push({
                label: 'AI-Typical Vocabulary',
                value: `${aiVocabAnalysis.aiWordCount} AI-associated terms detected`,
                note: `Words like "crucial", "leverage", "enhance" are overused by AI`,
                indicator: 'ai',
                severity: aiVocabAnalysis.aiVocabScore > 0.6 ? 'high' : 'medium',
                stats: {
                    measured: `AI Vocab Score: ${(aiVocabAnalysis.aiVocabScore * 100).toFixed(1)}%`,
                    wordsFound: aiVocabAnalysis.aiWordCount,
                    examples: topAI.map(w => `"${w.word}" (${w.count}×)`).join(', '),
                    density: `${((aiVocabAnalysis.aiWordCount / Math.max(1, aiVocabAnalysis.totalWords)) * 100).toFixed(2)}% of text`
                },
                benchmark: {
                    humanRange: '0%–2% AI-typical words',
                    aiRange: '3%–8% AI-typical words',
                    note: 'Based on 50+ known AI-overused terms'
                },
                excerpts: extractExcerpts(excerptWords)
            });
        }

        // Synonym rotation
        if (aiVocabAnalysis.synonymRotation && aiVocabAnalysis.synonymRotation.length > 0) {
            const rotations = aiVocabAnalysis.synonymRotation.slice(0, 3);
            findings.push({
                label: 'Synonym Rotation',
                value: 'Excessive synonym variety detected',
                note: 'Pattern of rotating similar words is common in AI text',
                indicator: 'ai',
                severity: 'medium',
                stats: {
                    measured: `${rotations.length} rotation patterns detected`,
                    examples: rotations.map(r => r.words ? r.words.join(', ') : r).join(' | ')
                },
                benchmark: {
                    note: 'AI often uses multiple synonyms where humans would repeat'
                }
            });
        }

        // Register uniformity
        if (registerAnalysis.uniformityScore > 0.6) {
            findings.push({
                label: 'Register Consistency',
                value: `Uniformly ${registerAnalysis.primaryRegister} register`,
                note: 'Lack of register variation may indicate AI generation',
                indicator: 'ai',
                severity: registerAnalysis.uniformityScore > 0.8 ? 'high' : 'medium',
                stats: {
                    measured: `Uniformity: ${(registerAnalysis.uniformityScore * 100).toFixed(1)}%`,
                    academicWords: registerAnalysis.academicCount,
                    conversationalWords: registerAnalysis.conversationalCount,
                    registerBalance: registerAnalysis.registerBalance
                },
                benchmark: {
                    humanRange: 'Mixed register with casual elements',
                    aiRange: 'Uniform formal/academic register',
                    note: 'Humans naturally mix formal and informal language'
                }
            });
        }

        // Conversational elements (human indicator)
        if (registerAnalysis.conversationalCount > 3) {
            const convWords = registerAnalysis.conversationalFound ? registerAnalysis.conversationalFound.slice(0, 5) : [];
            findings.push({
                label: 'Conversational Elements',
                value: 'Contains informal language markers',
                note: 'Natural conversational elements suggest human writing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `${registerAnalysis.conversationalCount} informal markers found`,
                    examples: registerAnalysis.conversationalFound ? registerAnalysis.conversationalFound.slice(0, 5).join(', ') : 'various'
                },
                benchmark: {
                    humanRange: '2+ conversational markers per 100 words',
                    aiRange: '0–1 conversational markers per 100 words'
                },
                excerpts: extractExcerpts(convWords)
            });
        }

        return findings;
    },

    calculateConfidence(tokenCount) {
        if (tokenCount < 50) return 0.3;
        if (tokenCount < 100) return 0.5;
        if (tokenCount < 200) return 0.7;
        if (tokenCount < 500) return 0.85;
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
    module.exports = LexicalAnalyzer;
}
