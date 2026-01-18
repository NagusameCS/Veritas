/**
 * VERITAS — Dialect & Regional Language Analyzer
 * Category 4: Variant Detection, Lexical Regionalism, Temporal Drift
 */

const DialectAnalyzer = {
    name: 'Dialect & Regional Consistency',
    category: 4,
    weight: 0.9,

    // British vs American spelling variants
    spellingVariants: {
        american: {
            'color': 'colour',
            'honor': 'honour',
            'favor': 'favour',
            'labor': 'labour',
            'neighbor': 'neighbour',
            'behavior': 'behaviour',
            'organize': 'organise',
            'realize': 'realise',
            'analyze': 'analyse',
            'recognize': 'recognise',
            'center': 'centre',
            'theater': 'theatre',
            'meter': 'metre',
            'fiber': 'fibre',
            'defense': 'defence',
            'offense': 'offence',
            'license': 'licence',
            'practice': 'practise', // verb form
            'catalog': 'catalogue',
            'dialog': 'dialogue',
            'program': 'programme',
            'traveling': 'travelling',
            'canceled': 'cancelled',
            'jewelry': 'jewellery',
            'gray': 'grey',
            'aging': 'ageing',
            'judgment': 'judgement'
        }
    },

    // Regional vocabulary
    regionalVocab: {
        american: ['apartment', 'elevator', 'sidewalk', 'fall', 'truck', 'faucet', 'cookies', 'garbage', 'gasoline', 'vacation', 'soccer', 'pants', 'sneakers'],
        british: ['flat', 'lift', 'pavement', 'autumn', 'lorry', 'tap', 'biscuits', 'rubbish', 'petrol', 'holiday', 'football', 'trousers', 'trainers']
    },

    // Modern slang markers
    modernSlang: [
        'lowkey', 'highkey', 'slay', 'bussin', 'fire', 'lit', 'goat', 'vibe',
        'mood', 'flex', 'cap', 'no cap', 'bet', 'fam', 'bro', 'bruh',
        'sus', 'based', 'cringe', 'simp', 'stan', 'yeet', 'periodt',
        'snatched', 'tea', 'shade', 'ghosting', 'clout', 'drip'
    ],

    // Archaic/formal terms
    archaicFormal: [
        'hitherto', 'heretofore', 'whereby', 'wherein', 'thereof', 'therein',
        'henceforth', 'forthwith', 'whilst', 'amongst', 'amidst', 'unto',
        'shall', 'ought', 'endeavour', 'notwithstanding', 'aforementioned',
        'inasmuch', 'insofar', 'lest', 'whence', 'thence', 'hence'
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        const tokens = Utils.tokenize(text);
        
        if (tokens.length < 20) {
            return this.getEmptyResult();
        }

        // Analyze spelling variants
        const spellingAnalysis = this.analyzeSpellingVariants(text);
        
        // Analyze punctuation conventions
        const punctuationAnalysis = this.analyzePunctuation(text);
        
        // Analyze regional vocabulary
        const vocabAnalysis = this.analyzeRegionalVocab(text);
        
        // Analyze temporal consistency
        const temporalAnalysis = this.analyzeTemporalConsistency(text);

        // Calculate AI probability
        // Inconsistent dialect = AI-like (mixing variants)
        // Pure slang or pure formal without mixing = more natural

        const scores = {
            dialectInconsistency: spellingAnalysis.inconsistencyScore,
            punctuationMix: punctuationAnalysis.mixScore,
            vocabInconsistency: vocabAnalysis.inconsistencyScore,
            temporalMix: temporalAnalysis.mixScore
        };

        const aiProbability = Utils.weightedAverage(
            [scores.dialectInconsistency, scores.punctuationMix, scores.vocabInconsistency, scores.temporalMix],
            [0.3, 0.2, 0.25, 0.25]
        );

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence: this.calculateConfidence(tokens.length),
            details: {
                spellingAnalysis,
                punctuationAnalysis,
                vocabAnalysis,
                temporalAnalysis
            },
            findings: this.generateFindings(spellingAnalysis, punctuationAnalysis, vocabAnalysis, temporalAnalysis),
            scores
        };
    },

    /**
     * Analyze spelling variants
     */
    analyzeSpellingVariants(text) {
        const lower = text.toLowerCase();
        const americanFound = [];
        const britishFound = [];
        
        for (const [american, british] of Object.entries(this.spellingVariants.american)) {
            const amPattern = new RegExp(`\\b${american}\\b`, 'gi');
            const brPattern = new RegExp(`\\b${british}\\b`, 'gi');
            
            if (amPattern.test(lower)) {
                americanFound.push(american);
            }
            if (brPattern.test(lower)) {
                britishFound.push(british);
            }
        }
        
        const totalVariants = americanFound.length + britishFound.length;
        const isMixed = americanFound.length > 0 && britishFound.length > 0;
        
        let dominantVariant = 'unknown';
        if (americanFound.length > britishFound.length) dominantVariant = 'american';
        else if (britishFound.length > americanFound.length) dominantVariant = 'british';
        else if (totalVariants > 0) dominantVariant = 'mixed';
        
        // Inconsistency score: high if mixing variants
        const inconsistencyScore = isMixed 
            ? Utils.normalize(Math.min(americanFound.length, britishFound.length), 0, 3)
            : 0;

        return {
            americanFound,
            britishFound,
            dominantVariant,
            isMixed,
            inconsistencyScore
        };
    },

    /**
     * Analyze punctuation conventions
     */
    analyzePunctuation(text) {
        // Check quote styles
        const doubleQuotes = (text.match(/"/g) || []).length;
        const singleQuotes = (text.match(/'/g) || []).length;
        const curlyDoubleOpen = (text.match(/"/g) || []).length;
        const curlyDoubleClose = (text.match(/"/g) || []).length;
        const curlySingleOpen = (text.match(/'/g) || []).length;
        const curlySingleClose = (text.match(/'/g) || []).length;
        
        // Check Oxford comma usage
        const oxfordCommaPattern = /,\s+and\s+\w+/gi;
        const noOxfordPattern = /\w+,\s*\w+\s+and\s+\w+/gi;
        const oxfordUse = (text.match(oxfordCommaPattern) || []).length;
        const noOxfordUse = (text.match(noOxfordPattern) || []).length;
        
        // Check date formats
        const usDatePattern = /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g;
        const ukDatePattern = /\b\d{1,2}-\d{1,2}-\d{2,4}\b/g;
        const usDates = (text.match(usDatePattern) || []).length;
        const ukDates = (text.match(ukDatePattern) || []).length;
        
        // Mix score: high if inconsistent conventions
        const quotesMixed = doubleQuotes > 0 && singleQuotes > 5;
        const oxfordMixed = oxfordUse > 0 && noOxfordUse > 0;
        
        const mixScore = (quotesMixed ? 0.3 : 0) + (oxfordMixed ? 0.4 : 0) + 
                        ((usDates > 0 && ukDates > 0) ? 0.3 : 0);

        return {
            quoteUsage: {
                double: doubleQuotes,
                single: singleQuotes,
                curly: curlyDoubleOpen + curlyDoubleClose + curlySingleOpen + curlySingleClose
            },
            oxfordComma: {
                used: oxfordUse,
                notUsed: noOxfordUse,
                consistent: !(oxfordUse > 0 && noOxfordUse > 0)
            },
            dateFormats: {
                us: usDates,
                uk: ukDates
            },
            mixScore
        };
    },

    /**
     * Analyze regional vocabulary
     */
    analyzeRegionalVocab(text) {
        const lower = text.toLowerCase();
        const americanVocab = [];
        const britishVocab = [];
        
        for (const word of this.regionalVocab.american) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                americanVocab.push(word);
            }
        }
        
        for (const word of this.regionalVocab.british) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                britishVocab.push(word);
            }
        }
        
        const isMixed = americanVocab.length > 0 && britishVocab.length > 0;
        
        let dominantRegion = 'unknown';
        if (americanVocab.length > britishVocab.length) dominantRegion = 'american';
        else if (britishVocab.length > americanVocab.length) dominantRegion = 'british';
        
        const inconsistencyScore = isMixed 
            ? Utils.normalize(Math.min(americanVocab.length, britishVocab.length), 0, 2)
            : 0;

        return {
            americanVocab,
            britishVocab,
            dominantRegion,
            isMixed,
            inconsistencyScore
        };
    },

    /**
     * Analyze temporal consistency
     */
    analyzeTemporalConsistency(text) {
        const lower = text.toLowerCase();
        
        // Find modern slang
        const modernFound = [];
        for (const word of this.modernSlang) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                modernFound.push(word);
            }
        }
        
        // Find archaic/formal terms
        const archaicFound = [];
        for (const word of this.archaicFormal) {
            if (new RegExp(`\\b${word}\\b`, 'gi').test(lower)) {
                archaicFound.push(word);
            }
        }
        
        // Mix of modern slang with archaic formal is unusual
        const isMixed = modernFound.length > 0 && archaicFound.length > 0;
        
        let temporalTone = 'neutral';
        if (modernFound.length > archaicFound.length + 2) temporalTone = 'modern';
        else if (archaicFound.length > modernFound.length + 2) temporalTone = 'formal';
        else if (isMixed) temporalTone = 'mixed';
        
        const mixScore = isMixed 
            ? Utils.normalize(Math.min(modernFound.length, archaicFound.length), 0, 2)
            : 0;

        return {
            modernFound,
            archaicFound,
            temporalTone,
            isMixed,
            mixScore
        };
    },

    /**
     * Generate findings
     */
    generateFindings(spellingAnalysis, punctuationAnalysis, vocabAnalysis, temporalAnalysis) {
        const findings = [];

        // Spelling inconsistency
        if (spellingAnalysis.isMixed) {
            findings.push({
                label: 'Spelling Variants',
                value: 'Mixed American/British spelling detected',
                note: `American: ${spellingAnalysis.americanFound.slice(0, 2).join(', ')} | British: ${spellingAnalysis.britishFound.slice(0, 2).join(', ')}`,
                indicator: 'ai'
            });
        } else if (spellingAnalysis.dominantVariant !== 'unknown') {
            findings.push({
                label: 'Spelling Variants',
                value: `Consistent ${spellingAnalysis.dominantVariant} spelling`,
                note: 'Spelling consistency is normal for human writers',
                indicator: 'neutral'
            });
        }

        // Punctuation
        if (!punctuationAnalysis.oxfordComma.consistent) {
            findings.push({
                label: 'Oxford Comma',
                value: 'Inconsistent Oxford comma usage',
                note: 'Mixing comma conventions may indicate AI',
                indicator: 'ai'
            });
        }

        // Regional vocab
        if (vocabAnalysis.isMixed) {
            findings.push({
                label: 'Regional Vocabulary',
                value: 'Mixed regional vocabulary detected',
                note: `Using both ${vocabAnalysis.americanVocab[0]} (US) and ${vocabAnalysis.britishVocab[0]} (UK)`,
                indicator: 'ai'
            });
        }

        // Temporal mix
        if (temporalAnalysis.isMixed) {
            findings.push({
                label: 'Temporal Consistency',
                value: 'Mixed modern slang with formal language',
                note: 'Unusual combination of registers',
                indicator: 'ai'
            });
        }

        if (temporalAnalysis.modernFound.length > 3) {
            findings.push({
                label: 'Modern Language',
                value: 'Contains contemporary slang/idioms',
                note: 'Natural modern language use',
                indicator: 'human'
            });
        }

        return findings;
    },

    calculateConfidence(tokenCount) {
        if (tokenCount < 50) return 0.2;
        if (tokenCount < 100) return 0.4;
        if (tokenCount < 200) return 0.6;
        if (tokenCount < 500) return 0.8;
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
    module.exports = DialectAnalyzer;
}
