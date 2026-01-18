/**
 * VERITAS — Authorship Consistency Analyzer
 * Category 9: Intra-Document Consistency, Cross-Document Deviation, Style Drift
 */

const AuthorshipAnalyzer = {
    name: 'Authorship Consistency',
    category: 9,
    weight: 1.1,

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
        
        if (sentences.length < 10) {
            return this.getEmptyResult();
        }

        // Analyze intra-document style drift
        const styleDriftAnalysis = this.analyzeStyleDrift(sentences, paragraphs);
        
        // Analyze complexity shifts
        const complexityAnalysis = this.analyzeComplexityShifts(sentences);
        
        // Analyze tone consistency
        const toneAnalysis = this.analyzeToneConsistency(sentences, paragraphs);
        
        // Analyze vocabulary consistency
        const vocabAnalysis = this.analyzeVocabularyConsistency(paragraphs);

        // Calculate AI probability
        // Sudden style shifts can indicate both AI (inconsistent mimicry) or human (mood changes)
        // AI typically has LESS natural variation, more mechanical consistency
        // But sudden "fluency spikes" suggest AI assistance

        const scores = {
            styleDrift: styleDriftAnalysis.driftScore,
            complexitySpikes: complexityAnalysis.spikeScore,
            toneShift: toneAnalysis.shiftScore,
            vocabInconsistency: vocabAnalysis.inconsistencyScore
        };

        // This category is complex: 
        // - Too consistent = AI-like
        // - Sudden spikes = AI-like
        // - Natural gradual variation = human-like

        const aiProbability = Utils.weightedAverage(
            [scores.styleDrift, scores.complexitySpikes, scores.toneShift, scores.vocabInconsistency],
            [0.25, 0.3, 0.25, 0.2]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(sentences.length, paragraphs.length),
            details: {
                styleDriftAnalysis,
                complexityAnalysis,
                toneAnalysis,
                vocabAnalysis
            },
            findings: this.generateFindings(styleDriftAnalysis, complexityAnalysis, toneAnalysis, vocabAnalysis),
            scores
        };
    },

    /**
     * Analyze style drift across the document
     */
    analyzeStyleDrift(sentences, paragraphs) {
        if (paragraphs.length < 2) {
            return { driftScore: 0.5, note: 'Insufficient paragraphs' };
        }

        // Compare style metrics between first half and second half
        const midpoint = Math.floor(sentences.length / 2);
        const firstHalf = sentences.slice(0, midpoint);
        const secondHalf = sentences.slice(midpoint);
        
        // Calculate metrics for each half
        const firstMetrics = this.calculateStyleMetrics(firstHalf);
        const secondMetrics = this.calculateStyleMetrics(secondHalf);
        
        // Calculate drift as difference between halves
        const avgSentenceLengthDrift = Math.abs(firstMetrics.avgSentenceLength - secondMetrics.avgSentenceLength);
        const avgWordLengthDrift = Math.abs(firstMetrics.avgWordLength - secondMetrics.avgWordLength);
        const functionWordDrift = Math.abs(firstMetrics.functionWordRatio - secondMetrics.functionWordRatio);
        
        // Punctuation usage drift
        const punctuationDrift = Math.abs(firstMetrics.punctuationDensity - secondMetrics.punctuationDensity);
        
        // Calculate overall drift
        const totalDrift = (
            Utils.normalize(avgSentenceLengthDrift, 0, 10) * 0.3 +
            Utils.normalize(avgWordLengthDrift, 0, 1) * 0.2 +
            Utils.normalize(functionWordDrift, 0, 0.1) * 0.25 +
            Utils.normalize(punctuationDrift, 0, 0.05) * 0.25
        );
        
        // Some drift is natural, extreme drift or no drift are suspicious
        // AI tends to be too consistent (low drift) or have sudden changes (high drift)
        const driftScore = totalDrift < 0.2 ? 0.6 : (totalDrift > 0.6 ? 0.7 : 0.3);

        return {
            firstHalfMetrics: firstMetrics,
            secondHalfMetrics: secondMetrics,
            drifts: {
                sentenceLength: avgSentenceLengthDrift.toFixed(1),
                wordLength: avgWordLengthDrift.toFixed(2),
                functionWords: functionWordDrift.toFixed(3),
                punctuation: punctuationDrift.toFixed(3)
            },
            totalDrift: totalDrift.toFixed(2),
            driftScore
        };
    },

    /**
     * Calculate style metrics for a set of sentences
     */
    calculateStyleMetrics(sentences) {
        const text = sentences.join(' ');
        const tokens = Utils.tokenize(text);
        
        const sentenceLengths = sentences.map(s => Utils.tokenize(s).length);
        const avgSentenceLength = Utils.mean(sentenceLengths);
        
        const avgWordLength = tokens.length > 0 
            ? tokens.reduce((sum, t) => sum + t.length, 0) / tokens.length 
            : 0;
        
        const functionWordCount = tokens.filter(t => Utils.functionWords.has(t)).length;
        const functionWordRatio = tokens.length > 0 ? functionWordCount / tokens.length : 0;
        
        const punctuationCount = (text.match(/[.,;:!?]/g) || []).length;
        const punctuationDensity = tokens.length > 0 ? punctuationCount / tokens.length : 0;

        return {
            avgSentenceLength: avgSentenceLength.toFixed(1),
            avgWordLength: avgWordLength.toFixed(2),
            functionWordRatio: functionWordRatio.toFixed(3),
            punctuationDensity: punctuationDensity.toFixed(3)
        };
    },

    /**
     * Analyze complexity shifts (fluency spikes)
     */
    analyzeComplexityShifts(sentences) {
        // Calculate complexity for each sentence
        const complexities = sentences.map(sentence => {
            const tokens = Utils.tokenize(sentence);
            const avgWordLength = tokens.length > 0 
                ? tokens.reduce((sum, t) => sum + t.length, 0) / tokens.length 
                : 0;
            const clauseIndicators = (sentence.match(/[,;:]|which|that|because|although|while|if|when/gi) || []).length;
            
            return {
                length: tokens.length,
                avgWordLength,
                clauseCount: clauseIndicators,
                complexity: tokens.length * 0.3 + avgWordLength * 2 + clauseIndicators * 3
            };
        });
        
        const complexityValues = complexities.map(c => c.complexity);
        const mean = Utils.mean(complexityValues);
        const stdDev = Utils.standardDeviation(complexityValues);
        
        // Detect spikes (sentences significantly more complex than neighbors)
        const spikes = [];
        for (let i = 1; i < complexities.length - 1; i++) {
            const current = complexityValues[i];
            const prev = complexityValues[i - 1];
            const next = complexityValues[i + 1];
            const neighborAvg = (prev + next) / 2;
            
            if (current > neighborAvg + stdDev * 1.5) {
                spikes.push({
                    index: i,
                    text: Utils.truncate(sentences[i], 60),
                    complexity: current.toFixed(1),
                    deviation: (current - neighborAvg).toFixed(1)
                });
            }
        }
        
        // Also detect sudden drops
        const drops = [];
        for (let i = 1; i < complexities.length - 1; i++) {
            const current = complexityValues[i];
            const prev = complexityValues[i - 1];
            const next = complexityValues[i + 1];
            const neighborAvg = (prev + next) / 2;
            
            if (current < neighborAvg - stdDev * 1.5) {
                drops.push({
                    index: i,
                    text: Utils.truncate(sentences[i], 60),
                    complexity: current.toFixed(1),
                    deviation: (neighborAvg - current).toFixed(1)
                });
            }
        }
        
        // Spike score: more spikes or drops = more suspicious
        const spikeScore = Utils.normalize(spikes.length + drops.length, 0, sentences.length * 0.15);

        return {
            meanComplexity: mean.toFixed(1),
            stdDevComplexity: stdDev.toFixed(1),
            spikes,
            drops,
            spikeScore,
            complexityBySection: this.getComplexityBySection(complexityValues, 4)
        };
    },

    /**
     * Get complexity averaged by sections
     */
    getComplexityBySection(complexities, sections) {
        const sectionSize = Math.ceil(complexities.length / sections);
        const result = [];
        
        for (let i = 0; i < sections; i++) {
            const start = i * sectionSize;
            const end = Math.min(start + sectionSize, complexities.length);
            const sectionComplexities = complexities.slice(start, end);
            result.push({
                section: i + 1,
                avgComplexity: Utils.mean(sectionComplexities).toFixed(1)
            });
        }
        
        return result;
    },

    /**
     * Analyze tone consistency
     */
    analyzeToneConsistency(sentences, paragraphs) {
        // Define tone indicators
        const toneIndicators = {
            formal: ['therefore', 'consequently', 'furthermore', 'moreover', 'thus', 'hence', 'accordingly', 'whereby'],
            informal: ['basically', 'actually', 'like', 'just', 'really', 'pretty', 'kind of', 'sort of', 'gonna', 'wanna'],
            confident: ['clearly', 'obviously', 'certainly', 'definitely', 'undoubtedly', 'must', 'always', 'never'],
            uncertain: ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems', 'appears', 'somewhat']
        };
        
        // Analyze tone by paragraph
        const paragraphTones = paragraphs.map(para => {
            const lower = para.toLowerCase();
            const tone = {};
            
            for (const [toneType, indicators] of Object.entries(toneIndicators)) {
                tone[toneType] = indicators.filter(ind => lower.includes(ind)).length;
            }
            
            return tone;
        });
        
        // Check for tone shifts
        let shiftCount = 0;
        for (let i = 1; i < paragraphTones.length; i++) {
            const prev = paragraphTones[i - 1];
            const curr = paragraphTones[i];
            
            // Check for formal/informal shifts
            if ((prev.formal > 0 && curr.informal > 0) || (prev.informal > 0 && curr.formal > 0)) {
                shiftCount++;
            }
            // Check for confident/uncertain shifts
            if ((prev.confident > 0 && curr.uncertain > 1) || (prev.uncertain > 0 && curr.confident > 1)) {
                shiftCount++;
            }
        }
        
        // Calculate dominant tone
        const totalTones = paragraphTones.reduce((acc, t) => {
            for (const [key, value] of Object.entries(t)) {
                acc[key] = (acc[key] || 0) + value;
            }
            return acc;
        }, {});
        
        const dominantTone = Object.entries(totalTones)
            .sort((a, b) => b[1] - a[1])[0];
        
        // Shift score: some shifts are natural, too many are suspicious
        const shiftScore = Utils.normalize(shiftCount, 0, paragraphs.length * 0.3);

        return {
            paragraphTones,
            totalTones,
            dominantTone: dominantTone ? { type: dominantTone[0], count: dominantTone[1] } : null,
            toneShifts: shiftCount,
            shiftScore
        };
    },

    /**
     * Analyze vocabulary consistency across paragraphs
     */
    analyzeVocabularyConsistency(paragraphs) {
        if (paragraphs.length < 2) {
            return { inconsistencyScore: 0.5, note: 'Insufficient paragraphs' };
        }

        // Get content words for each paragraph
        const paragraphVocabs = paragraphs.map(para => {
            const tokens = Utils.tokenize(para);
            const contentWords = tokens.filter(t => 
                !Utils.functionWords.has(t) && t.length > 3
            );
            return new Set(contentWords);
        });
        
        // Calculate vocabulary overlap between adjacent paragraphs
        const overlaps = [];
        for (let i = 1; i < paragraphVocabs.length; i++) {
            const prev = paragraphVocabs[i - 1];
            const curr = paragraphVocabs[i];
            const overlap = [...prev].filter(w => curr.has(w)).length;
            const overlapRatio = Math.min(prev.size, curr.size) > 0 
                ? overlap / Math.min(prev.size, curr.size)
                : 0;
            overlaps.push(overlapRatio);
        }
        
        // Check for sudden vocabulary shifts
        const avgOverlap = Utils.mean(overlaps);
        const overlapVariance = Utils.variance(overlaps);
        
        // Low overlap with low variance = consistent but disconnected paragraphs
        // This can indicate AI-generated text with independent paragraphs
        const inconsistencyScore = avgOverlap < 0.2 && overlapVariance < 0.05 ? 0.7 : 
                                   avgOverlap < 0.1 ? 0.6 : 0.3;

        return {
            avgOverlap: avgOverlap.toFixed(2),
            overlapVariance: overlapVariance.toFixed(3),
            overlaps,
            inconsistencyScore
        };
    },

    /**
     * Generate findings
     */
    generateFindings(styleDriftAnalysis, complexityAnalysis, toneAnalysis, vocabAnalysis) {
        const findings = [];

        // Style drift
        if (styleDriftAnalysis.driftScore > 0.5 && parseFloat(styleDriftAnalysis.totalDrift) < 0.2) {
            findings.push({
                label: 'Style Consistency',
                value: 'Unusually consistent style throughout',
                note: 'Mechanical consistency may indicate AI',
                indicator: 'ai'
            });
        }

        if (parseFloat(styleDriftAnalysis.totalDrift) > 0.5) {
            findings.push({
                label: 'Style Shift',
                value: 'Significant style change detected',
                note: 'First and second half have different characteristics',
                indicator: 'mixed'
            });
        }

        // Complexity spikes
        if (complexityAnalysis.spikes.length > 0) {
            findings.push({
                label: 'Fluency Spikes',
                value: `${complexityAnalysis.spikes.length} sudden complexity increase(s)`,
                note: 'May indicate AI assistance in specific sections',
                indicator: 'ai'
            });
        }

        if (complexityAnalysis.drops.length > 0) {
            findings.push({
                label: 'Complexity Drops',
                value: `${complexityAnalysis.drops.length} sudden simplification(s)`,
                note: 'Inconsistent complexity across text',
                indicator: 'mixed'
            });
        }

        // Tone shifts
        if (toneAnalysis.toneShifts > 2) {
            findings.push({
                label: 'Tone Shifts',
                value: `${toneAnalysis.toneShifts} tone changes detected`,
                note: 'Inconsistent register throughout document',
                indicator: 'ai'
            });
        }

        // Vocabulary consistency
        if (vocabAnalysis.inconsistencyScore > 0.5) {
            findings.push({
                label: 'Vocabulary Cohesion',
                value: 'Low vocabulary overlap between paragraphs',
                note: 'Paragraphs may be independently generated',
                indicator: 'ai'
            });
        }

        return findings;
    },

    calculateConfidence(sentenceCount, paragraphCount) {
        if (paragraphCount < 3 || sentenceCount < 15) return 0.3;
        if (sentenceCount < 30) return 0.5;
        if (sentenceCount < 60) return 0.7;
        return 0.85;
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
    module.exports = AuthorshipAnalyzer;
}
