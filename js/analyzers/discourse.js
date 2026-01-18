/**
 * VERITAS — Discourse & Coherence Analyzer
 * Category 6: Paragraph Flow, Logical Progression, Discourse Markers
 */

const DiscourseAnalyzer = {
    name: 'Discourse & Coherence',
    category: 6,
    weight: 1.1,

    // Formulaic discourse markers (AI indicators)
    formulaicMarkers: {
        opening: [
            'in today\'s world', 'in the modern era', 'throughout history',
            'it is widely known', 'it is commonly understood', 'as we all know',
            'in recent years', 'with the advent of', 'in this day and age'
        ],
        transition: [
            'furthermore', 'moreover', 'additionally', 'in addition',
            'on the other hand', 'conversely', 'nevertheless', 'nonetheless',
            'consequently', 'as a result', 'therefore', 'thus', 'hence',
            'similarly', 'likewise', 'in the same vein', 'by the same token'
        ],
        conclusion: [
            'in conclusion', 'to conclude', 'in summary', 'to summarize',
            'overall', 'all in all', 'in the final analysis', 'ultimately',
            'to sum up', 'in essence', 'in short', 'all things considered'
        ],
        emphasis: [
            'it is important to note', 'it should be noted', 'notably',
            'significantly', 'importantly', 'crucially', 'essentially',
            'fundamentally', 'interestingly', 'remarkably', 'surprisingly'
        ]
    },

    // Redundant restatement patterns
    restatementPatterns: [
        /in other words,/gi,
        /that is to say,/gi,
        /put simply,/gi,
        /to put it another way,/gi,
        /this means that/gi,
        /what this means is/gi,
        /essentially,\s+this/gi
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
        
        if (sentences.length < 5) {
            return this.getEmptyResult();
        }

        // Analyze paragraph structure
        const paragraphAnalysis = this.analyzeParagraphStructure(paragraphs);
        
        // Analyze transitions and flow
        const transitionAnalysis = this.analyzeTransitions(text, sentences);
        
        // Analyze logical progression
        const logicalAnalysis = this.analyzeLogicalProgression(sentences, paragraphs);
        
        // Analyze discourse markers
        const markerAnalysis = this.analyzeDiscourseMarkers(text, sentences);

        // Calculate AI probability
        // Smooth transitions = AI-like
        // Uniform paragraphs = AI-like
        // Excessive discourse markers = AI-like
        // Predictable structure = AI-like

        const scores = {
            overSmooth: transitionAnalysis.smoothnessScore,
            uniformParagraphs: paragraphAnalysis.uniformityScore,
            markerOveruse: markerAnalysis.overuseScore,
            predictability: logicalAnalysis.predictabilityScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.overSmooth, scores.uniformParagraphs, scores.markerOveruse, scores.predictability],
            [0.25, 0.2, 0.3, 0.25]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(sentences.length, paragraphs.length),
            details: {
                paragraphAnalysis,
                transitionAnalysis,
                logicalAnalysis,
                markerAnalysis
            },
            findings: this.generateFindings(paragraphAnalysis, transitionAnalysis, logicalAnalysis, markerAnalysis),
            scores
        };
    },

    /**
     * Analyze paragraph structure
     */
    analyzeParagraphStructure(paragraphs) {
        if (paragraphs.length < 2) {
            return {
                count: paragraphs.length,
                uniformityScore: 0.5,
                note: 'Insufficient paragraphs for analysis'
            };
        }

        // Analyze paragraph lengths
        const lengths = paragraphs.map(p => Utils.wordCount(p));
        const mean = Utils.mean(lengths);
        const stdDev = Utils.standardDeviation(lengths);
        const cv = mean > 0 ? stdDev / mean : 0;
        
        // Check for topic sentence patterns
        let topicSentencePattern = 0;
        for (const para of paragraphs) {
            const sentences = Utils.splitSentences(para);
            if (sentences.length > 0) {
                const firstSentence = sentences[0].toLowerCase();
                // Check if first sentence is a clear topic statement
                if (firstSentence.length > 50 && !firstSentence.includes('?')) {
                    topicSentencePattern++;
                }
            }
        }
        const topicSentenceRatio = topicSentencePattern / paragraphs.length;
        
        // Uniformity score: high if paragraphs are too similar in length
        const uniformityScore = 1 - Utils.normalize(cv, 0.2, 0.6);

        return {
            count: paragraphs.length,
            meanLength: mean.toFixed(1),
            stdDev: stdDev.toFixed(1),
            coefficientOfVariation: cv.toFixed(2),
            topicSentenceRatio: topicSentenceRatio.toFixed(2),
            uniformityScore,
            lengths
        };
    },

    /**
     * Analyze transitions between sentences and paragraphs
     */
    analyzeTransitions(text, sentences) {
        const lower = text.toLowerCase();
        let transitionCount = 0;
        const transitionsFound = [];
        
        // Count all transition markers
        for (const markers of Object.values(this.formulaicMarkers)) {
            for (const marker of markers) {
                const pattern = new RegExp(marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                const matches = lower.match(pattern);
                if (matches) {
                    transitionCount += matches.length;
                    transitionsFound.push({ marker, count: matches.length });
                }
            }
        }
        
        // Transition density
        const transitionDensity = sentences.length > 0 ? transitionCount / sentences.length : 0;
        
        // Check for sentence-starting patterns
        const sentenceStarts = sentences.map(s => {
            const words = s.trim().split(/\s+/).slice(0, 3);
            return words.join(' ').toLowerCase();
        });
        
        // Count how many sentences start with transition words
        const transitionStarts = sentenceStarts.filter(start => 
            this.formulaicMarkers.transition.some(t => start.startsWith(t.toLowerCase()))
        ).length;
        
        const transitionStartRatio = transitionStarts / sentences.length;
        
        // Smoothness score: high if transitions are overused
        const smoothnessScore = Utils.normalize(transitionDensity, 0, 0.3);

        return {
            totalTransitions: transitionCount,
            transitionDensity: transitionDensity.toFixed(2),
            transitionStartRatio: transitionStartRatio.toFixed(2),
            topTransitions: transitionsFound.sort((a, b) => b.count - a.count).slice(0, 5),
            smoothnessScore
        };
    },

    /**
     * Analyze logical progression
     */
    analyzeLogicalProgression(sentences, paragraphs) {
        // Check for restatement patterns
        let restatementCount = 0;
        const text = sentences.join(' ');
        
        for (const pattern of this.restatementPatterns) {
            const matches = text.match(pattern);
            if (matches) restatementCount += matches.length;
        }
        
        // Check for circular reasoning (similar content at start and end)
        const firstParagraph = paragraphs[0] || '';
        const lastParagraph = paragraphs[paragraphs.length - 1] || '';
        
        const firstWords = new Set(Utils.tokenize(firstParagraph).filter(w => !Utils.functionWords.has(w)));
        const lastWords = new Set(Utils.tokenize(lastParagraph).filter(w => !Utils.functionWords.has(w)));
        
        const overlap = [...firstWords].filter(w => lastWords.has(w)).length;
        const overlapRatio = Math.min(firstWords.size, lastWords.size) > 0 
            ? overlap / Math.min(firstWords.size, lastWords.size)
            : 0;
        
        // Check for genuine digressions (human indicator)
        // Look for parenthetical asides, tangential comments
        const digressionPatterns = [
            /\(.*?\)/g,  // Parenthetical comments
            /—.*?—/g,    // Em-dash asides
            /by the way/gi,
            /incidentally/gi,
            /speaking of which/gi,
            /on a related note/gi
        ];
        
        let digressionCount = 0;
        for (const pattern of digressionPatterns) {
            const matches = text.match(pattern);
            if (matches) digressionCount += matches.length;
        }
        
        // Predictability score: high if text is too structured without digressions
        const predictabilityScore = (
            Utils.normalize(restatementCount, 0, 3) * 0.3 +
            Utils.normalize(overlapRatio, 0.3, 0.7) * 0.4 +
            (1 - Utils.normalize(digressionCount, 0, 3)) * 0.3
        );

        return {
            restatementCount,
            introOutroOverlap: (overlapRatio * 100).toFixed(1) + '%',
            digressionCount,
            predictabilityScore
        };
    },

    /**
     * Analyze discourse marker usage
     */
    analyzeDiscourseMarkers(text, sentences) {
        const lower = text.toLowerCase();
        const markersByCategory = {};
        let totalMarkers = 0;
        
        for (const [category, markers] of Object.entries(this.formulaicMarkers)) {
            markersByCategory[category] = { count: 0, examples: [] };
            for (const marker of markers) {
                const pattern = new RegExp(`\\b${marker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'gi');
                const matches = text.match(pattern);
                if (matches) {
                    markersByCategory[category].count += matches.length;
                    markersByCategory[category].examples.push(marker);
                    totalMarkers += matches.length;
                }
            }
        }
        
        // Check for formulaic paragraph endings
        const formulaicEndings = sentences.filter(s => {
            const lower = s.toLowerCase().trim();
            return lower.endsWith('in conclusion.') ||
                   lower.endsWith('moving forward.') ||
                   lower.endsWith('all things considered.') ||
                   lower.includes('it is clear that') ||
                   lower.includes('it is evident that');
        }).length;
        
        // Overuse score
        const markerDensity = sentences.length > 0 ? totalMarkers / sentences.length : 0;
        const overuseScore = Utils.normalize(markerDensity, 0, 0.4);

        return {
            totalMarkers,
            markerDensity: markerDensity.toFixed(2),
            markersByCategory,
            formulaicEndings,
            overuseScore
        };
    },

    /**
     * Generate findings
     */
    generateFindings(paragraphAnalysis, transitionAnalysis, logicalAnalysis, markerAnalysis) {
        const findings = [];

        // Paragraph uniformity
        if (paragraphAnalysis.uniformityScore > 0.6) {
            findings.push({
                label: 'Paragraph Structure',
                value: 'Highly uniform paragraph lengths',
                note: `CV: ${paragraphAnalysis.coefficientOfVariation} - suggests mechanical construction`,
                indicator: 'ai'
            });
        }

        // Over-smooth transitions
        if (transitionAnalysis.smoothnessScore > 0.5) {
            findings.push({
                label: 'Transition Density',
                value: 'Excessive use of transitional phrases',
                note: `${transitionAnalysis.totalTransitions} markers in ${transitionAnalysis.transitionDensity} per sentence`,
                indicator: 'ai'
            });
        }

        // Discourse marker overuse
        if (markerAnalysis.overuseScore > 0.5) {
            const topCategory = Object.entries(markerAnalysis.markersByCategory)
                .sort((a, b) => b[1].count - a[1].count)[0];
            findings.push({
                label: 'Discourse Markers',
                value: `Overuse of ${topCategory?.[0]} markers`,
                note: topCategory?.[1]?.examples?.slice(0, 3).join(', '),
                indicator: 'ai'
            });
        }

        // Restatement
        if (logicalAnalysis.restatementCount > 2) {
            findings.push({
                label: 'Redundancy',
                value: 'Multiple restatement patterns detected',
                note: 'Excessive clarification is common in AI text',
                indicator: 'ai'
            });
        }

        // Digressions (human indicator)
        if (logicalAnalysis.digressionCount > 1) {
            findings.push({
                label: 'Natural Digressions',
                value: 'Contains tangential thoughts or asides',
                note: 'Natural digressions suggest human writing',
                indicator: 'human'
            });
        }

        return findings;
    },

    calculateConfidence(sentenceCount, paragraphCount) {
        if (paragraphCount < 2 || sentenceCount < 10) return 0.3;
        if (sentenceCount < 20) return 0.5;
        if (sentenceCount < 50) return 0.7;
        return 0.9;
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
    module.exports = DiscourseAnalyzer;
}
