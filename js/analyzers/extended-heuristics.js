/**
 * VERITAS — Extended Heuristics Analyzer
 * =======================================
 * 
 * COMPREHENSIVE COLLECTION OF EVERY CONCEIVABLE TEXT HEURISTIC
 * For human reviewers to inspect - NOT used in ML model training
 * 
 * Categories:
 * 1. Linguistic Micro-Features
 * 2. Stylometric Signatures  
 * 3. Cognitive Load Indicators
 * 4. Emotional/Affective Signals
 * 5. Structural Patterns
 * 6. Error & Imperfection Analysis
 * 7. Domain-Specific Markers
 * 8. Temporal & Cultural Signals
 * 9. Discourse & Pragmatic Features
 * 10. Statistical Distributions
 */

const ExtendedHeuristicsAnalyzer = {
    name: 'Extended Heuristics (Human Review)',
    category: 'extended',
    
    /**
     * Main analysis - returns ALL heuristics for human review
     */
    analyze(text) {
        const results = {
            metadata: {
                analyzedAt: new Date().toISOString(),
                textLength: text.length,
                wordCount: this.countWords(text),
                sentenceCount: this.countSentences(text),
                paragraphCount: this.countParagraphs(text)
            },
            
            // === CATEGORY 1: LINGUISTIC MICRO-FEATURES ===
            linguisticMicro: this.analyzeLinguisticMicro(text),
            
            // === CATEGORY 2: STYLOMETRIC SIGNATURES ===
            stylometric: this.analyzeStylometric(text),
            
            // === CATEGORY 3: COGNITIVE LOAD INDICATORS ===
            cognitiveLoad: this.analyzeCognitiveLoad(text),
            
            // === CATEGORY 4: EMOTIONAL/AFFECTIVE SIGNALS ===
            emotional: this.analyzeEmotional(text),
            
            // === CATEGORY 5: STRUCTURAL PATTERNS ===
            structural: this.analyzeStructural(text),
            
            // === CATEGORY 6: ERROR & IMPERFECTION ANALYSIS ===
            errors: this.analyzeErrors(text),
            
            // === CATEGORY 7: DOMAIN-SPECIFIC MARKERS ===
            domain: this.analyzeDomain(text),
            
            // === CATEGORY 8: TEMPORAL & CULTURAL SIGNALS ===
            temporal: this.analyzeTemporal(text),
            
            // === CATEGORY 9: DISCOURSE & PRAGMATIC FEATURES ===
            discourse: this.analyzeDiscourse(text),
            
            // === CATEGORY 10: STATISTICAL DISTRIBUTIONS ===
            statistical: this.analyzeStatistical(text)
        };
        
        // Calculate summary scores
        results.summary = this.generateSummary(results);
        
        return results;
    },
    
    // ========================================================================
    // CATEGORY 1: LINGUISTIC MICRO-FEATURES
    // ========================================================================
    analyzeLinguisticMicro(text) {
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        return {
            // 1.1 Morphological Features
            morphological: {
                prefixUsage: this.countPrefixes(words),
                suffixUsage: this.countSuffixes(words),
                compoundWords: this.countCompoundWords(words),
                derivedWords: this.countDerivedWords(words),
                inflectedForms: this.countInflectedForms(words),
                rootWordRatio: this.calculateRootWordRatio(words),
                morphemeComplexity: this.calculateMorphemeComplexity(words)
            },
            
            // 1.2 Phonological Approximations
            phonological: {
                alliterationCount: this.countAlliteration(text),
                assonanceCount: this.countAssonance(text),
                consonanceCount: this.countConsonance(text),
                syllableVariation: this.calculateSyllableVariation(words),
                phoneticPatterns: this.detectPhoneticPatterns(text),
                rhythmScore: this.calculateRhythmScore(sentences)
            },
            
            // 1.3 Word-Level Features
            wordLevel: {
                averageWordLength: this.avgWordLength(words),
                wordLengthVariance: this.wordLengthVariance(words),
                monosyllabicRatio: this.monosyllabicRatio(words),
                polysyllabicRatio: this.polysyllabicRatio(words),
                wordLengthDistribution: this.wordLengthDistribution(words),
                shortWordChains: this.countShortWordChains(words),
                longWordClusters: this.countLongWordClusters(words)
            },
            
            // 1.4 Character-Level Features
            characterLevel: {
                vowelConsonantRatio: this.vowelConsonantRatio(text),
                doubleLetterFrequency: this.doubleLetterFrequency(text),
                tripleLetterOccurrence: this.tripleLetterOccurrence(text),
                characterNGrams: this.characterNGramAnalysis(text),
                letterFrequencyDeviation: this.letterFrequencyDeviation(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 2: STYLOMETRIC SIGNATURES
    // ========================================================================
    analyzeStylometric(text) {
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        return {
            // 2.1 Function Word Analysis
            functionWords: {
                functionWordRatio: this.functionWordRatio(words),
                functionWordDistribution: this.functionWordDistribution(words),
                articleUsage: this.articleUsage(words),
                prepositionPatterns: this.prepositionPatterns(words),
                conjunctionFrequency: this.conjunctionFrequency(words),
                auxiliaryVerbUsage: this.auxiliaryVerbUsage(words),
                modalVerbPatterns: this.modalVerbPatterns(words),
                pronounDistribution: this.pronounDistribution(words)
            },
            
            // 2.2 Vocabulary Richness
            vocabularyRichness: {
                typeTokenRatio: this.typeTokenRatio(words),
                rootTTR: this.rootTTR(words),
                correctedTTR: this.correctedTTR(words),
                hapaxLegomena: this.hapaxLegomena(words),
                hapaxDisLegomena: this.hapaxDisLegomena(words),
                yuleK: this.yuleK(words),
                sichelS: this.sichelS(words),
                honoréR: this.honoreR(words),
                brunetW: this.brunetW(words),
                masseA: this.masseA(words)
            },
            
            // 2.3 Sentence Complexity
            sentenceComplexity: {
                avgSentenceLength: this.avgSentenceLength(sentences),
                sentenceLengthStdDev: this.sentenceLengthStdDev(sentences),
                clauseCount: this.clauseCount(sentences),
                subordinateClauseRatio: this.subordinateClauseRatio(text),
                coordinateClauseRatio: this.coordinateClauseRatio(text),
                sentenceTypeDistribution: this.sentenceTypeDistribution(sentences),
                embeddingDepth: this.estimateEmbeddingDepth(sentences)
            },
            
            // 2.4 Punctuation Fingerprint
            punctuationFingerprint: {
                commaFrequency: this.punctuationRate(text, ','),
                semicolonFrequency: this.punctuationRate(text, ';'),
                colonFrequency: this.punctuationRate(text, ':'),
                dashFrequency: this.dashFrequency(text),
                parenthesisFrequency: this.parenthesisFrequency(text),
                quotationFrequency: this.quotationFrequency(text),
                ellipsisFrequency: this.ellipsisFrequency(text),
                exclamationFrequency: this.punctuationRate(text, '!'),
                questionFrequency: this.punctuationRate(text, '?'),
                punctuationVariety: this.punctuationVariety(text),
                punctuationDensity: this.punctuationDensity(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 3: COGNITIVE LOAD INDICATORS
    // ========================================================================
    analyzeCognitiveLoad(text) {
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        return {
            // 3.1 Readability Metrics
            readability: {
                fleschReadingEase: this.fleschReadingEase(text),
                fleschKincaidGrade: this.fleschKincaidGrade(text),
                gunningFogIndex: this.gunningFogIndex(text),
                smogIndex: this.smogIndex(text),
                colemanLiauIndex: this.colemanLiauIndex(text),
                automatedReadabilityIndex: this.automatedReadabilityIndex(text),
                daleChallScore: this.daleChallScore(text),
                lixReadability: this.lixReadability(text),
                rixReadability: this.rixReadability(text),
                spacheScore: this.spacheScore(text)
            },
            
            // 3.2 Processing Difficulty
            processingDifficulty: {
                averageParseTreeDepth: this.estimateParseTreeDepth(sentences),
                gardenPathSentences: this.detectGardenPath(sentences),
                centerEmbedding: this.detectCenterEmbedding(sentences),
                longDistanceDependencies: this.detectLongDependencies(sentences),
                ambiguityScore: this.estimateAmbiguity(sentences),
                negationComplexity: this.negationComplexity(text)
            },
            
            // 3.3 Information Density
            informationDensity: {
                contentWordRatio: this.contentWordRatio(words),
                propositionDensity: this.estimatePropositionDensity(sentences),
                ideaDensity: this.estimateIdeaDensity(text),
                conceptDensity: this.estimateConceptDensity(text),
                lexicalDensity: this.lexicalDensity(words),
                informationLoadPerSentence: this.informationLoadPerSentence(sentences)
            },
            
            // 3.4 Working Memory Load
            workingMemoryLoad: {
                maxClauseDepth: this.maxClauseDepth(sentences),
                pronounResolutionDifficulty: this.pronounResolutionDifficulty(text),
                referentDistance: this.averageReferentDistance(text),
                topicShiftFrequency: this.topicShiftFrequency(text),
                entityDensity: this.entityDensity(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 4: EMOTIONAL/AFFECTIVE SIGNALS
    // ========================================================================
    analyzeEmotional(text) {
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        return {
            // 4.1 Sentiment Features
            sentiment: {
                overallPolarity: this.estimateSentimentPolarity(text),
                sentimentVariance: this.sentimentVariance(sentences),
                sentimentTrajectory: this.sentimentTrajectory(sentences),
                positiveWordCount: this.countPositiveWords(words),
                negativeWordCount: this.countNegativeWords(words),
                neutralRatio: this.neutralWordRatio(words),
                sentimentIntensity: this.sentimentIntensity(text)
            },
            
            // 4.2 Emotional Lexicon
            emotionalLexicon: {
                joyWords: this.countEmotionWords(text, 'joy'),
                angerWords: this.countEmotionWords(text, 'anger'),
                fearWords: this.countEmotionWords(text, 'fear'),
                sadnessWords: this.countEmotionWords(text, 'sadness'),
                surpriseWords: this.countEmotionWords(text, 'surprise'),
                disgustWords: this.countEmotionWords(text, 'disgust'),
                trustWords: this.countEmotionWords(text, 'trust'),
                anticipationWords: this.countEmotionWords(text, 'anticipation'),
                emotionalDiversity: this.emotionalDiversity(text)
            },
            
            // 4.3 Affect Markers
            affectMarkers: {
                intensifiers: this.countIntensifiers(text),
                hedges: this.countHedges(text),
                boosters: this.countBoosters(text),
                downtoners: this.countDowntoners(text),
                expressivePunctuation: this.expressivePunctuation(text),
                emphasisMarkers: this.emphasisMarkers(text),
                exclamations: this.exclamationPatterns(text)
            },
            
            // 4.4 Subjectivity Indicators
            subjectivity: {
                opinionWords: this.countOpinionWords(text),
                subjectiveAdjectives: this.countSubjectiveAdjectives(text),
                evaluativeLanguage: this.countEvaluativeLanguage(text),
                personalOpinionMarkers: this.personalOpinionMarkers(text),
                objectivityScore: this.objectivityScore(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 5: STRUCTURAL PATTERNS
    // ========================================================================
    analyzeStructural(text) {
        const sentences = this.splitSentences(text);
        const paragraphs = this.splitParagraphs(text);
        
        return {
            // 5.1 Document Structure
            documentStructure: {
                paragraphCount: paragraphs.length,
                avgParagraphLength: this.avgParagraphLength(paragraphs),
                paragraphLengthVariance: this.paragraphLengthVariance(paragraphs),
                hasIntroduction: this.detectIntroduction(paragraphs),
                hasConclusion: this.detectConclusion(paragraphs),
                hasTransitions: this.detectTransitions(paragraphs),
                structuralBalance: this.structuralBalance(paragraphs)
            },
            
            // 5.2 Sentence Patterns
            sentencePatterns: {
                declarativeRatio: this.sentenceTypeRatio(sentences, 'declarative'),
                interrogativeRatio: this.sentenceTypeRatio(sentences, 'interrogative'),
                imperativeRatio: this.sentenceTypeRatio(sentences, 'imperative'),
                exclamatoryRatio: this.sentenceTypeRatio(sentences, 'exclamatory'),
                sentenceOpeningPatterns: this.sentenceOpeningPatterns(sentences),
                sentenceClosingPatterns: this.sentenceClosingPatterns(sentences),
                passiveVoiceRatio: this.passiveVoiceRatio(sentences)
            },
            
            // 5.3 List & Enumeration
            listEnumeration: {
                numberedListCount: this.numberedListCount(text),
                bulletListCount: this.bulletListCount(text),
                inlineEnumerationCount: this.inlineEnumerationCount(text),
                parallelStructureScore: this.parallelStructureScore(text),
                listItemConsistency: this.listItemConsistency(text)
            },
            
            // 5.4 Formatting Signals
            formattingSignals: {
                capitalizationPatterns: this.capitalizationPatterns(text),
                whitespacePatterns: this.whitespacePatterns(text),
                lineBreakPatterns: this.lineBreakPatterns(text),
                indentationConsistency: this.indentationConsistency(text),
                headingPresence: this.detectHeadings(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 6: ERROR & IMPERFECTION ANALYSIS
    // ========================================================================
    analyzeErrors(text) {
        const words = this.tokenize(text);
        
        return {
            // 6.1 Spelling Patterns
            spelling: {
                potentialMisspellings: this.detectMisspellings(words),
                typoPatterns: this.detectTypoPatterns(text),
                homophoneConfusion: this.detectHomophoneConfusion(text),
                spellingInconsistency: this.spellingInconsistency(text),
                britishAmericanMixing: this.detectBritishAmericanMix(text)
            },
            
            // 6.2 Grammar Patterns
            grammar: {
                subjectVerbDisagreement: this.detectSubjectVerbDisagreement(text),
                articleErrors: this.detectArticleErrors(text),
                prepositionErrors: this.detectPrepositionErrors(text),
                tenseInconsistency: this.detectTenseInconsistency(text),
                pronounErrors: this.detectPronounErrors(text),
                doubleNegatives: this.detectDoubleNegatives(text)
            },
            
            // 6.3 Punctuation Errors
            punctuationErrors: {
                missingPunctuation: this.detectMissingPunctuation(text),
                extraPunctuation: this.detectExtraPunctuation(text),
                commaErrors: this.detectCommaErrors(text),
                apostropheErrors: this.detectApostropheErrors(text),
                quotationErrors: this.detectQuotationErrors(text)
            },
            
            // 6.4 Disfluencies
            disfluencies: {
                repetitions: this.detectRepetitions(text),
                fillerWords: this.countFillerWords(text),
                falseStarts: this.detectFalseStarts(text),
                selfCorrections: this.detectSelfCorrections(text),
                incompleteThoughts: this.detectIncompleteThoughts(text),
                wordSearches: this.detectWordSearches(text)
            },
            
            // 6.5 ESL Markers
            eslMarkers: {
                articleOmission: this.detectArticleOmission(text),
                pluralErrors: this.detectPluralErrors(text),
                wordOrderIssues: this.detectWordOrderIssues(text),
                prepositionMisuse: this.detectPrepositionMisuse(text),
                collocationalErrors: this.detectCollocationalErrors(text),
                l1Interference: this.detectL1Interference(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 7: DOMAIN-SPECIFIC MARKERS
    // ========================================================================
    analyzeDomain(text) {
        return {
            // 7.1 Academic Writing
            academic: {
                citationPresence: this.detectCitations(text),
                citationStyle: this.detectCitationStyle(text),
                academicVocabulary: this.academicVocabularyRatio(text),
                hedgingLanguage: this.hedgingLanguageCount(text),
                methodologyMarkers: this.methodologyMarkers(text),
                researchTerms: this.researchTermCount(text),
                formalRegister: this.formalRegisterScore(text)
            },
            
            // 7.2 Technical Writing
            technical: {
                technicalTermDensity: this.technicalTermDensity(text),
                acronymUsage: this.acronymUsage(text),
                codePresence: this.detectCodePresence(text),
                numericalData: this.numericalDataDensity(text),
                unitMentions: this.unitMentionCount(text),
                specificationLanguage: this.specificationLanguage(text)
            },
            
            // 7.3 Creative Writing
            creative: {
                metaphorCount: this.detectMetaphors(text),
                simileCount: this.detectSimiles(text),
                personificationCount: this.detectPersonification(text),
                imageryDensity: this.imageryDensity(text),
                dialoguePresence: this.detectDialogue(text),
                narrativeMarkers: this.narrativeMarkers(text),
                descriptiveLanguage: this.descriptiveLanguageRatio(text)
            },
            
            // 7.4 Business/Professional
            business: {
                businessJargon: this.businessJargonCount(text),
                formalSalutation: this.detectFormalSalutation(text),
                professionalClosing: this.detectProfessionalClosing(text),
                actionableLanguage: this.actionableLanguageCount(text),
                stakeholderReferences: this.stakeholderReferences(text)
            },
            
            // 7.5 Social Media/Casual
            socialMedia: {
                hashtagCount: this.hashtagCount(text),
                mentionCount: this.mentionCount(text),
                emojiCount: this.emojiCount(text),
                abbreviationCount: this.abbreviationCount(text),
                slangUsage: this.slangUsage(text),
                internetisms: this.internetisms(text)
            },
            
            // 7.6 Legal Writing
            legal: {
                legalTerms: this.legalTermCount(text),
                latinPhrases: this.latinPhraseCount(text),
                formalDefinitions: this.formalDefinitionCount(text),
                conditionalClauses: this.conditionalClauseCount(text),
                disclaimerLanguage: this.disclaimerLanguage(text)
            },
            
            // 7.7 Medical/Scientific
            scientific: {
                scientificTerms: this.scientificTermCount(text),
                measurementMentions: this.measurementMentions(text),
                processDescriptions: this.processDescriptions(text),
                cautionaryLanguage: this.cautionaryLanguage(text),
                evidenceMarkers: this.evidenceMarkers(text)
            },
            
            // 7.8 Model UN/Debate/Speech
            formalSpeech: {
                delegateReferences: this.delegateReferences(text),
                countryReferences: this.countryReferences(text),
                proposalLanguage: this.proposalLanguage(text),
                resolutionTerms: this.resolutionTerms(text),
                diplomaticLanguage: this.diplomaticLanguage(text),
                speechMarkers: this.speechMarkers(text),
                audienceAddressing: this.audienceAddressing(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 8: TEMPORAL & CULTURAL SIGNALS
    // ========================================================================
    analyzeTemporal(text) {
        return {
            // 8.1 Temporal References
            temporalReferences: {
                pastReferences: this.countPastReferences(text),
                presentReferences: this.countPresentReferences(text),
                futureReferences: this.countFutureReferences(text),
                specificDates: this.detectSpecificDates(text),
                relativeTimeMarkers: this.relativeTimeMarkers(text),
                temporalConnectives: this.temporalConnectives(text),
                chronologicalMarkers: this.chronologicalMarkers(text)
            },
            
            // 8.2 Cultural Markers
            culturalMarkers: {
                geographicReferences: this.geographicReferences(text),
                culturalReferences: this.culturalReferences(text),
                idiomUsage: this.idiomUsage(text),
                proverbUsage: this.proverbUsage(text),
                popCultureReferences: this.popCultureReferences(text),
                historicalReferences: this.historicalReferences(text)
            },
            
            // 8.3 Dialect/Regional Signals
            dialectSignals: {
                regionalVocabulary: this.regionalVocabulary(text),
                dialectPatterns: this.dialectPatterns(text),
                spellingVariants: this.spellingVariants(text),
                idiomaticExpressions: this.idiomaticExpressions(text)
            },
            
            // 8.4 Generational Markers
            generationalMarkers: {
                slangGeneration: this.identifySlangGeneration(text),
                technologyReferences: this.technologyReferences(text),
                mediaReferences: this.mediaReferences(text),
                socialMediaIndicators: this.socialMediaIndicators(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 9: DISCOURSE & PRAGMATIC FEATURES
    // ========================================================================
    analyzeDiscourse(text) {
        const sentences = this.splitSentences(text);
        const paragraphs = this.splitParagraphs(text);
        
        return {
            // 9.1 Coherence Markers
            coherence: {
                topicConsistency: this.topicConsistency(paragraphs),
                lexicalCohesion: this.lexicalCohesion(sentences),
                referentialCohesion: this.referentialCohesion(text),
                conjunctionCohesion: this.conjunctionCohesion(text),
                thematicProgression: this.thematicProgression(paragraphs)
            },
            
            // 9.2 Discourse Connectives
            discourseConnectives: {
                additiveConnectives: this.additiveConnectives(text),
                adversativeConnectives: this.adversativeConnectives(text),
                causalConnectives: this.causalConnectives(text),
                temporalConnectives: this.temporalConnectivesCount(text),
                exemplificationMarkers: this.exemplificationMarkers(text),
                summaryMarkers: this.summaryMarkers(text)
            },
            
            // 9.3 Speech Acts
            speechActs: {
                assertives: this.countAssertives(text),
                directives: this.countDirectives(text),
                commissives: this.countCommissives(text),
                expressives: this.countExpressives(text),
                declaratives: this.countDeclaratives(text),
                questionTypes: this.questionTypeAnalysis(text)
            },
            
            // 9.4 Pragmatic Markers
            pragmaticMarkers: {
                politenessMarkers: this.politenessMarkers(text),
                mitigationDevices: this.mitigationDevices(text),
                emphasisDevices: this.emphasisDevices(text),
                evidentialMarkers: this.evidentialMarkers(text),
                epistemicMarkers: this.epistemicMarkers(text),
                attitudeMarkers: this.attitudeMarkers(text)
            },
            
            // 9.5 Argumentation Patterns
            argumentation: {
                claimMarkers: this.claimMarkers(text),
                evidenceMarkers: this.argumentEvidenceMarkers(text),
                counterargumentMarkers: this.counterargumentMarkers(text),
                concessionMarkers: this.concessionMarkers(text),
                reasoningMarkers: this.reasoningMarkers(text),
                conclusionMarkers: this.conclusionMarkers(text)
            }
        };
    },
    
    // ========================================================================
    // CATEGORY 10: STATISTICAL DISTRIBUTIONS
    // ========================================================================
    analyzeStatistical(text) {
        const words = this.tokenize(text);
        const sentences = this.splitSentences(text);
        
        return {
            // 10.1 Zipf's Law Analysis
            zipfAnalysis: {
                zipfSlope: this.calculateZipfSlope(words),
                zipfDeviation: this.calculateZipfDeviation(words),
                zipfRSquared: this.calculateZipfRSquared(words),
                mandelbrotFit: this.mandelbrotFit(words)
            },
            
            // 10.2 Entropy Measures
            entropyMeasures: {
                wordEntropy: this.wordEntropy(words),
                characterEntropy: this.characterEntropy(text),
                bigramEntropy: this.bigramEntropy(words),
                sentenceLengthEntropy: this.sentenceLengthEntropy(sentences),
                conditionalEntropy: this.conditionalEntropy(words),
                crossEntropy: this.estimateCrossEntropy(words)
            },
            
            // 10.3 Burstiness Analysis
            burstinessAnalysis: {
                wordBurstiness: this.wordBurstiness(words),
                topicBurstiness: this.topicBurstiness(text),
                keywordClustering: this.keywordClustering(words),
                interArrivalTimes: this.interArrivalTimeAnalysis(words)
            },
            
            // 10.4 Distribution Fits
            distributionFits: {
                wordLengthDistribution: this.wordLengthDistributionFit(words),
                sentenceLengthDistribution: this.sentenceLengthDistributionFit(sentences),
                vocabularyGrowthCurve: this.vocabularyGrowthCurve(words),
                heapsLawFit: this.heapsLawFit(words)
            },
            
            // 10.5 Sequence Analysis
            sequenceAnalysis: {
                markovTransitions: this.markovTransitionAnalysis(words),
                autocorrelation: this.autocorrelation(words),
                longRangeDependence: this.longRangeDependence(words),
                hurstExponent: this.hurstExponent(words)
            }
        };
    },

    // ========================================================================
    // HELPER METHODS - Basic utilities
    // ========================================================================
    
    tokenize(text) {
        return text.toLowerCase().match(/\b[a-z'-]+\b/g) || [];
    },
    
    splitSentences(text) {
        return text.split(/[.!?]+\s+/).filter(s => s.trim().length > 0);
    },
    
    splitParagraphs(text) {
        return text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
    },
    
    countWords(text) {
        return (text.match(/\b\w+\b/g) || []).length;
    },
    
    countSentences(text) {
        return this.splitSentences(text).length;
    },
    
    countParagraphs(text) {
        return this.splitParagraphs(text).length;
    },

    // Summary generation
    generateSummary(results) {
        return {
            totalHeuristicsAnalyzed: this.countTotalHeuristics(results),
            categoriesCovered: 10,
            analysisComplete: true,
            keyFindings: this.extractKeyFindings(results)
        };
    },
    
    countTotalHeuristics(results) {
        let count = 0;
        const countRecursive = (obj) => {
            for (const key in obj) {
                if (typeof obj[key] === 'object' && obj[key] !== null) {
                    countRecursive(obj[key]);
                } else {
                    count++;
                }
            }
        };
        countRecursive(results);
        return count;
    },
    
    extractKeyFindings(results) {
        // Extract notable findings for quick review
        return [];
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ExtendedHeuristicsAnalyzer;
}
