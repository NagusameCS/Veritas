/**
 * VERITAS — Meta-Patterns Unique to AI Analyzer
 * Category 10: Over-balanced Arguments, Hedging, Formatting, Rhetorical Arcs
 */

const MetaPatternsAnalyzer = {
    name: 'Meta-Patterns Unique to AI',
    category: 10,
    weight: 1.2,

    // Hedging phrases (expanded for newer models)
    hedgingPhrases: [
        'it could be argued', 'one might suggest', 'it is possible that',
        'there is potential for', 'in some cases', 'under certain circumstances',
        'to a certain extent', 'in many ways', 'arguably', 'presumably',
        'it seems that', 'it appears that', 'it would seem', 'it may be',
        'this suggests that', 'this indicates that', 'this implies that',
        'generally speaking', 'broadly speaking', 'for the most part',
        'with that said', 'that being said', 'having said that',
        // Gemini/newer model hedging
        'it is worth noting', 'it is important to note', 'it should be noted',
        'it is essential to understand', 'it is crucial to recognize',
        'while there are', 'there are several', 'there are many',
        'it is widely recognized', 'it is generally accepted',
        'can potentially', 'may potentially', 'could potentially'
    ],

    // Strong stance markers (absence indicates hedging)
    strongStanceMarkers: [
        'i strongly believe', 'without a doubt', 'absolutely', 'definitely',
        'certainly', 'undeniably', 'unquestionably', 'clearly', 'obviously',
        'it is clear that', 'there is no doubt', 'the fact is',
        'i am convinced', 'i firmly believe', 'make no mistake'
    ],

    // Gemini-specific structural patterns
    geminiStructuralPatterns: [
        /here('s| is) (a |the )?(breakdown|overview|summary|look)/gi,
        /let('s| us) (delve|explore|examine|consider|take a look)/gi,
        /(first|1\.|•)\s*(and |then )?(second|2\.|•)/gi,
        /key (takeaways?|points?|considerations?|factors?)/gi,
        /in (essence|summary|conclusion|short)/gi,
        /the (bottom line|key point|main takeaway)/gi,
        /what (this|that) means is/gi,
        /when it comes to/gi,
        /in terms of/gi
    ],

    // Balanced argument templates
    balancedTemplates: [
        /on one hand.*on the other hand/gi,
        /while.*however/gi,
        /although.*nevertheless/gi,
        /pros and cons/gi,
        /advantages and disadvantages/gi,
        /benefits and drawbacks/gi,
        /positive.*negative/gi,
        /both sides/gi,
        /there are arguments (for|both)/gi
    ],

    // Predictable rhetorical structures
    rhetoricalPatterns: {
        introduction: [
            /^(in (today's|the modern|our) (world|era|society))/i,
            /^(throughout history)/i,
            /^(it is (widely|commonly|generally) (known|accepted|understood))/i,
            /^(the (importance|significance|impact) of)/i
        ],
        body: [
            /(first(ly)?|second(ly)?|third(ly)?|finally)/gi,
            /(to begin with|moving on|furthermore|in addition)/gi,
            /(another (important|key|crucial) (point|aspect|factor))/gi
        ],
        conclusion: [
            /(in conclusion|to conclude|in summary|to summarize)/gi,
            /(all in all|overall|ultimately|in the end)/gi,
            /(taking everything into (account|consideration))/gi
        ]
    },

    /**
     * Main analysis function
     */
    analyze(text) {
        const sentences = Utils.splitSentences(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
        
        if (sentences.length < 5) {
            return this.getEmptyResult();
        }

        // Analyze argument balance
        const balanceAnalysis = this.analyzeArgumentBalance(text);
        
        // Analyze hedging patterns
        const hedgingAnalysis = this.analyzeHedging(text, sentences);
        
        // Analyze formatting symmetry
        const formattingAnalysis = this.analyzeFormatting(text, sentences, paragraphs);
        
        // Analyze rhetorical arc predictability
        const rhetoricalAnalysis = this.analyzeRhetoricalArc(text, paragraphs);
        
        // Analyze Gemini-specific structural patterns
        const geminiAnalysis = this.analyzeGeminiPatterns(text);

        // Calculate AI probability
        // All these patterns are strong AI indicators

        const scores = {
            overBalanced: balanceAnalysis.balanceScore,
            excessiveHedging: hedgingAnalysis.hedgingScore,
            formattingSymmetry: formattingAnalysis.symmetryScore,
            predictableArc: rhetoricalAnalysis.predictabilityScore,
            geminiPatterns: geminiAnalysis.geminiScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.overBalanced, scores.excessiveHedging, scores.formattingSymmetry, scores.predictableArc, scores.geminiPatterns],
            [0.2, 0.2, 0.15, 0.25, 0.2]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(sentences.length, paragraphs.length),
            details: {
                balanceAnalysis,
                hedgingAnalysis,
                formattingAnalysis,
                rhetoricalAnalysis,
                geminiAnalysis
            },
            findings: this.generateFindings(balanceAnalysis, hedgingAnalysis, formattingAnalysis, rhetoricalAnalysis, geminiAnalysis),
            scores
        };
    },

    /**
     * Analyze Gemini-specific structural patterns
     */
    analyzeGeminiPatterns(text) {
        let patternCount = 0;
        const patternsFound = [];
        
        for (const pattern of this.geminiStructuralPatterns) {
            const matches = text.match(pattern);
            if (matches) {
                patternCount += matches.length;
                patternsFound.push(matches[0].toLowerCase().trim());
            }
        }
        
        // Check for characteristic openings
        const lowerText = text.toLowerCase();
        const geminiOpenings = [
            'here is', 'here are', 'here\'s',
            'let me', 'let\'s',
            'i\'d be happy to',
            'absolutely!', 'great question!', 'that\'s a great',
            'i can help',
            'sure!', 'of course!'
        ];
        
        let openingMatches = 0;
        for (const opening of geminiOpenings) {
            if (lowerText.startsWith(opening) || lowerText.includes('\n' + opening)) {
                openingMatches++;
                patternsFound.push('characteristic opening: ' + opening);
            }
        }
        
        // Gemini score based on pattern density
        const geminiScore = Utils.normalize(patternCount + openingMatches, 0, 5);
        
        return {
            patternCount,
            openingMatches,
            patternsFound: [...new Set(patternsFound)].slice(0, 5),
            geminiScore
        };
    },

    /**
     * Analyze argument balance
     */
    analyzeArgumentBalance(text) {
        const lower = text.toLowerCase();
        
        // Check for balanced argument templates
        let templateMatches = 0;
        const templatesFound = [];
        
        for (const template of this.balancedTemplates) {
            if (template.test(text)) {
                templateMatches++;
                const match = text.match(template);
                if (match) templatesFound.push(match[0]);
            }
        }
        
        // Analyze pro/con balance
        const proWords = ['advantage', 'benefit', 'positive', 'good', 'helpful', 'useful', 'important', 'valuable'];
        const conWords = ['disadvantage', 'drawback', 'negative', 'bad', 'harmful', 'problematic', 'concerning', 'risk'];
        
        let proCount = 0;
        let conCount = 0;
        
        for (const word of proWords) {
            const matches = lower.match(new RegExp(`\\b${word}\\b`, 'gi'));
            if (matches) proCount += matches.length;
        }
        
        for (const word of conWords) {
            const matches = lower.match(new RegExp(`\\b${word}\\b`, 'gi'));
            if (matches) conCount += matches.length;
        }
        
        // Check if perfectly balanced (suspicious)
        const totalSides = proCount + conCount;
        const isBalanced = totalSides > 2 && Math.abs(proCount - conCount) <= 1;
        const isOverBalanced = templateMatches >= 2 || isBalanced;
        
        // Balance score: high if suspiciously balanced
        const balanceScore = (
            Utils.normalize(templateMatches, 0, 3) * 0.5 +
            (isBalanced ? 0.5 : 0)
        );

        return {
            templateMatches,
            templatesFound,
            proConBalance: {
                pro: proCount,
                con: conCount,
                isBalanced
            },
            isOverBalanced,
            balanceScore
        };
    },

    /**
     * Analyze hedging patterns
     */
    analyzeHedging(text, sentences) {
        const lower = text.toLowerCase();
        
        // Count hedging phrases
        let hedgingCount = 0;
        const hedgingFound = [];
        
        for (const phrase of this.hedgingPhrases) {
            if (lower.includes(phrase)) {
                hedgingCount++;
                hedgingFound.push(phrase);
            }
        }
        
        // Count strong stance markers
        let strongStanceCount = 0;
        for (const phrase of this.strongStanceMarkers) {
            if (lower.includes(phrase)) {
                strongStanceCount++;
            }
        }
        
        // Calculate hedging ratio
        const hedgingPerSentence = sentences.length > 0 ? hedgingCount / sentences.length : 0;
        const stanceRatio = (hedgingCount + strongStanceCount) > 0 
            ? hedgingCount / (hedgingCount + strongStanceCount)
            : 0.5;
        
        // Hedging score: high hedging + low strong stance = AI-like
        const hedgingScore = (
            Utils.normalize(hedgingPerSentence, 0, 0.2) * 0.5 +
            Utils.normalize(stanceRatio, 0.5, 0.9) * 0.5
        );

        return {
            hedgingCount,
            hedgingFound: hedgingFound.slice(0, 5),
            strongStanceCount,
            hedgingPerSentence: hedgingPerSentence.toFixed(2),
            stanceRatio: stanceRatio.toFixed(2),
            hedgingScore
        };
    },

    /**
     * Analyze formatting symmetry
     */
    analyzeFormatting(text, sentences, paragraphs) {
        // Check paragraph length symmetry
        const paragraphLengths = paragraphs.map(p => Utils.wordCount(p));
        const lengthCV = paragraphLengths.length > 1 
            ? Utils.standardDeviation(paragraphLengths) / Utils.mean(paragraphLengths)
            : 0.5;
        
        // Check for list/enumeration patterns
        const listPatterns = text.match(/^[\s]*[-•*]\s/gm) || [];
        const numberedPatterns = text.match(/^[\s]*\d+[.)]\s/gm) || [];
        const hasLists = listPatterns.length > 0 || numberedPatterns.length > 0;
        
        // Check for header-like patterns
        const headerPatterns = text.match(/^[A-Z][A-Za-z\s]+:[\s]*$/gm) || [];
        
        // Check sentence length symmetry within paragraphs
        let sentenceSymmetry = 0;
        for (const para of paragraphs) {
            const paraSentences = Utils.splitSentences(para);
            if (paraSentences.length >= 3) {
                const lengths = paraSentences.map(s => Utils.wordCount(s));
                const cv = Utils.standardDeviation(lengths) / Utils.mean(lengths);
                if (cv < 0.3) sentenceSymmetry++;
            }
        }
        
        // Check for consistent sentence counts per paragraph
        const sentenceCounts = paragraphs.map(p => Utils.splitSentences(p).length);
        const sentenceCountCV = sentenceCounts.length > 1
            ? Utils.standardDeviation(sentenceCounts) / Utils.mean(sentenceCounts)
            : 0.5;
        
        // Symmetry score: high if too regular/symmetric
        const symmetryScore = (
            (1 - Utils.normalize(lengthCV, 0.2, 0.6)) * 0.3 +
            (1 - Utils.normalize(sentenceCountCV, 0.3, 0.7)) * 0.3 +
            Utils.normalize(sentenceSymmetry, 0, paragraphs.length * 0.5) * 0.4
        );

        return {
            paragraphLengths,
            paragraphLengthCV: lengthCV.toFixed(2),
            hasLists,
            listItemCount: listPatterns.length + numberedPatterns.length,
            headerCount: headerPatterns.length,
            sentenceSymmetricParagraphs: sentenceSymmetry,
            sentenceCountCV: sentenceCountCV.toFixed(2),
            symmetryScore
        };
    },

    /**
     * Analyze rhetorical arc predictability
     */
    analyzeRhetoricalArc(text, paragraphs) {
        if (paragraphs.length < 3) {
            return { predictabilityScore: 0.5, note: 'Insufficient paragraphs' };
        }

        const lower = text.toLowerCase();
        let patternMatches = 0;
        const patternsFound = [];
        
        // Check introduction patterns in first paragraph
        const firstPara = paragraphs[0].toLowerCase();
        for (const pattern of this.rhetoricalPatterns.introduction) {
            if (pattern.test(firstPara)) {
                patternMatches++;
                patternsFound.push({ type: 'intro', location: 'first paragraph' });
            }
        }
        
        // Check body patterns in middle paragraphs
        const bodyPatternCount = {
            ordinal: 0,
            transitional: 0
        };
        for (const pattern of this.rhetoricalPatterns.body) {
            const matches = text.match(pattern);
            if (matches) {
                bodyPatternCount.ordinal += matches.length;
            }
        }
        if (bodyPatternCount.ordinal >= 3) {
            patternMatches++;
            patternsFound.push({ type: 'body', pattern: 'ordinal markers' });
        }
        
        // Check conclusion patterns in last paragraph
        const lastPara = paragraphs[paragraphs.length - 1].toLowerCase();
        for (const pattern of this.rhetoricalPatterns.conclusion) {
            if (pattern.test(lastPara)) {
                patternMatches++;
                patternsFound.push({ type: 'conclusion', location: 'last paragraph' });
            }
        }
        
        // Check for "five-paragraph essay" structure
        const isFiveParaStructure = paragraphs.length >= 4 && paragraphs.length <= 6 &&
            patternsFound.some(p => p.type === 'intro') &&
            patternsFound.some(p => p.type === 'conclusion');
        
        // Predictability score
        const predictabilityScore = (
            Utils.normalize(patternMatches, 0, 4) * 0.5 +
            (isFiveParaStructure ? 0.3 : 0) +
            Utils.normalize(bodyPatternCount.ordinal, 0, 5) * 0.2
        );

        return {
            patternMatches,
            patternsFound,
            bodyPatternCount,
            isFiveParaStructure,
            predictabilityScore
        };
    },

    /**
     * Generate findings
     */
    generateFindings(balanceAnalysis, hedgingAnalysis, formattingAnalysis, rhetoricalAnalysis, geminiAnalysis) {
        const findings = [];

        // Gemini-specific patterns (new)
        if (geminiAnalysis && geminiAnalysis.geminiScore > 0.3) {
            const total = geminiAnalysis.patternCount + geminiAnalysis.openingMatches;
            findings.push({
                text: `Detected ${total} Gemini/modern AI structural patterns. Examples: "${geminiAnalysis.patternsFound.slice(0, 3).join('", "')}". These patterns are characteristic of newer AI models like Gemini, Claude 3, and GPT-4.`,
                label: 'Modern AI Patterns',
                indicator: 'ai',
                severity: total >= 4 ? 'high' : 'medium'
            });
        }

        // Over-balanced arguments with specifics
        if (balanceAnalysis.isOverBalanced) {
            const proConStr = `${balanceAnalysis.proConBalance.pro} pros vs ${balanceAnalysis.proConBalance.con} cons`;
            findings.push({
                text: `Suspiciously balanced perspective: ${proConStr}. ${balanceAnalysis.templateMatches} balanced-argument templates detected. AI tends to present artificially even-handed views.`,
                label: 'Argument Balance',
                indicator: 'ai',
                severity: 'high'
            });
        }

        // Excessive hedging with metrics
        if (hedgingAnalysis.hedgingScore > 0.5) {
            findings.push({
                text: `Excessive hedging: ${hedgingAnalysis.hedgingCount} hedging phrases (${hedgingAnalysis.hedgingPerSentence}/sentence). Examples: "${hedgingAnalysis.hedgingFound.slice(0, 3).join('", "')}". Human avg: 0.1-0.2/sentence.`,
                label: 'Hedging Language',
                indicator: 'ai',
                severity: 'high'
            });
        }

        // Lack of strong stance
        if (hedgingAnalysis.strongStanceCount === 0 && hedgingAnalysis.hedgingCount > 2) {
            findings.push({
                text: `No strong stance markers found while using ${hedgingAnalysis.hedgingCount} hedging phrases. Stance ratio: ${hedgingAnalysis.stanceRatio} (1.0 = all hedging). AI avoids definitive positions.`,
                label: 'Position Strength',
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Formatting symmetry with detailed metrics
        if (formattingAnalysis.symmetryScore > 0.5) {
            findings.push({
                text: `Highly uniform paragraph structure: Length CV=${formattingAnalysis.paragraphLengthCV} (human avg: 0.4-0.6), Sentence count CV=${formattingAnalysis.sentenceCountCV}, ${formattingAnalysis.sentenceSymmetricParagraphs} paragraphs with uniform sentence lengths.`,
                label: 'Formatting Symmetry',
                indicator: 'ai',
                severity: 'high'
            });
        } else if (parseFloat(formattingAnalysis.paragraphLengthCV) > 0.5) {
            findings.push({
                text: `Natural paragraph length variation: CV=${formattingAnalysis.paragraphLengthCV} indicates human-like structural variety.`,
                label: 'Formatting Variety',
                indicator: 'human',
                severity: 'low'
            });
        }

        // Predictable rhetorical arc
        if (rhetoricalAnalysis.predictabilityScore > 0.5) {
            const types = rhetoricalAnalysis.patternsFound.map(p => p.type);
            findings.push({
                text: `Follows predictable essay template with ${rhetoricalAnalysis.patternMatches} rhetorical markers. Detected: ${[...new Set(types)].join(', ')}. Ordinal markers ("firstly", "secondly"): ${rhetoricalAnalysis.bodyPatternCount.ordinal}.`,
                label: 'Rhetorical Structure',
                indicator: 'ai',
                severity: 'high'
            });
        }

        // Five-paragraph essay structure
        if (rhetoricalAnalysis.isFiveParaStructure) {
            findings.push({
                text: `Classic five-paragraph essay format detected. Has intro pattern, body ordinals, and conclusion markers. This formulaic structure is common in AI-generated academic content.`,
                label: 'Essay Structure',
                indicator: 'ai',
                severity: 'high'
            });
        }

        return findings;
    },

    calculateConfidence(sentenceCount, paragraphCount) {
        if (paragraphCount < 3 || sentenceCount < 10) return 0.4;
        if (sentenceCount < 25) return 0.6;
        if (sentenceCount < 50) return 0.8;
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
    module.exports = MetaPatternsAnalyzer;
}
