/**
 * VERITAS — Tone Stability Analyzer
 * Measures tone drift, stability, and variance patterns
 * 
 * KEY PRINCIPLE: Humans drift in tone; AI stabilizes tone.
 * We measure HOW tone changes, not WHAT the tone is.
 */

const ToneAnalyzer = {
    name: 'Tone Stability',
    category: 13,
    weight: 1.5,

    // Tone dimension lexicons
    lexicons: {
        firstPerson: ['i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'],
        secondPerson: ['you', 'your', 'yours', 'yourself', 'yourselves'],
        thirdPerson: ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'hers', 'its', 'their', 'theirs'],
        
        hedging: [
            'perhaps', 'maybe', 'possibly', 'probably', 'might', 'could', 'may', 'seems',
            'appears', 'suggests', 'indicates', 'somewhat', 'relatively', 'fairly', 'rather',
            'arguably', 'potentially', 'likely', 'unlikely', 'generally', 'typically',
            'tend to', 'in some ways', 'to some extent', 'it seems', 'it appears'
        ],
        
        certainty: [
            'definitely', 'certainly', 'absolutely', 'clearly', 'obviously', 'undoubtedly',
            'without doubt', 'unquestionably', 'surely', 'always', 'never', 'must', 'will',
            'proven', 'established', 'confirmed', 'demonstrated', 'evident', 'indisputable'
        ],
        
        positive: [
            'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 'love',
            'happy', 'pleased', 'excited', 'delighted', 'thrilled', 'fortunate', 'beneficial',
            'positive', 'successful', 'impressive', 'remarkable', 'outstanding', 'superb'
        ],
        
        negative: [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'disappointing',
            'sad', 'angry', 'frustrated', 'annoyed', 'unfortunate', 'problematic',
            'negative', 'failed', 'poor', 'weak', 'flawed', 'inadequate', 'concerning'
        ],
        
        formal: [
            'therefore', 'moreover', 'furthermore', 'consequently', 'nevertheless',
            'notwithstanding', 'hereby', 'thereof', 'wherein', 'whereas', 'pursuant',
            'accordingly', 'henceforth', 'thus', 'hence', 'thereby'
        ],
        
        informal: [
            'like', 'just', 'really', 'pretty', 'kind of', 'sort of', 'stuff', 'things',
            'gonna', 'wanna', 'gotta', 'yeah', 'okay', 'ok', 'cool', 'awesome', 'basically',
            'actually', 'literally', 'totally', 'super', 'kinda', 'anyways', 'btw', 'tbh'
        ],
        
        // Gemini/Claude/newer AI enthusiastic helper tone
        aiHelpfulTone: [
            'happy to help', 'glad to assist', 'here to help', 'let me help',
            'i can help', 'i\'d be happy', 'absolutely', 'great question',
            'excellent question', 'that\'s a great', 'wonderful question',
            'certainly', 'of course', 'sure thing', 'no problem',
            'hope this helps', 'feel free', 'don\'t hesitate', 'let me know',
            'happy to clarify', 'hope that helps', 'glad to help'
        ]
    },

    // Gemini-specific enthusiastic patterns (phrase-level)
    geminiEnthusiasticPatterns: [
        /absolutely!|great question!|excellent!|perfect!|wonderful!/gi,
        /i('d| would) be happy to/gi,
        /that's a (great|wonderful|excellent|fantastic) (question|point|idea)/gi,
        /i (hope|trust) (this|that) helps/gi,
        /(feel free|don't hesitate) to (ask|reach out|let me know)/gi,
        /let me (help|assist|explain|break (it |this )?down)/gi,
        /here('s| is) (a |the )?(quick |brief )?(summary|overview|breakdown)/gi
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        if (!text || text.length < 100) {
            return this.getEmptyResult();
        }

        const sentences = Utils.splitSentences(text);
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);

        // Measure tone dimensions per segment
        const sentenceTones = sentences.map(s => this.measureTone(s));
        const paragraphTones = paragraphs.map(p => this.measureTone(p));

        // Calculate stability metrics for each dimension
        const stability = this.calculateStability(sentenceTones);
        const drift = this.calculateDrift(paragraphTones);
        const consistency = this.calculateConsistency(sentenceTones);

        // Detect specific patterns
        const toneShifts = this.detectToneShifts(sentenceTones);
        const hedgingPattern = this.analyzeHedgingPattern(sentenceTones);
        const emotionalProfile = this.analyzeEmotionalVariance(sentenceTones);
        
        // Detect AI helper tone (Gemini, Claude, etc.)
        const aiHelperTone = this.detectAIHelperTone(text);

        // Calculate scores
        const scores = {
            overallStability: stability.overall,
            hedgingConsistency: hedgingPattern.consistency,
            emotionalFlatness: emotionalProfile.flatness,
            registerStability: stability.register,
            aiHelperTone: aiHelperTone.score
        };

        // AI probability based on excessive stability
        const aiProbability = this.calculateAIProbability(scores, stability, toneShifts, aiHelperTone);
        
        const confidence = this.calculateConfidence(sentences.length);
        const findings = this.generateFindings(stability, drift, toneShifts, hedgingPattern, emotionalProfile, aiHelperTone);

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence,
            details: {
                stability,
                drift,
                consistency,
                toneShifts,
                hedgingPattern,
                emotionalProfile,
                aiHelperTone,
                sentenceTones
            },
            findings,
            scores
        };
    },

    /**
     * Measure tone dimensions for a text segment
     */
    measureTone(text) {
        const tokens = Utils.tokenize(text.toLowerCase());
        const wordCount = tokens.length;
        
        if (wordCount === 0) {
            return this.getEmptyTone();
        }

        const counts = {};
        for (const dimension of Object.keys(this.lexicons)) {
            const words = this.lexicons[dimension];
            counts[dimension] = tokens.filter(t => words.includes(t)).length;
        }

        return {
            firstPersonRate: counts.firstPerson / wordCount,
            secondPersonRate: counts.secondPerson / wordCount,
            thirdPersonRate: counts.thirdPerson / wordCount,
            hedgingRate: counts.hedging / wordCount,
            certaintyRate: counts.certainty / wordCount,
            positiveRate: counts.positive / wordCount,
            negativeRate: counts.negative / wordCount,
            formalRate: counts.formal / wordCount,
            informalRate: counts.informal / wordCount,
            
            // Derived metrics
            emotionalValence: (counts.positive - counts.negative) / wordCount,
            modalityBalance: counts.certainty > 0 || counts.hedging > 0
                ? counts.certainty / (counts.certainty + counts.hedging)
                : 0.5,
            registerScore: counts.formal > 0 || counts.informal > 0
                ? counts.formal / (counts.formal + counts.informal)
                : 0.5,
            pronounBalance: (counts.firstPerson + counts.secondPerson) / 
                           (counts.firstPerson + counts.secondPerson + counts.thirdPerson + 1)
        };
    },

    /**
     * Empty tone measurement
     */
    getEmptyTone() {
        return {
            firstPersonRate: 0, secondPersonRate: 0, thirdPersonRate: 0,
            hedgingRate: 0, certaintyRate: 0,
            positiveRate: 0, negativeRate: 0,
            formalRate: 0, informalRate: 0,
            emotionalValence: 0, modalityBalance: 0.5, registerScore: 0.5, pronounBalance: 0
        };
    },

    /**
     * Calculate stability metrics across sentences
     */
    calculateStability(tones) {
        if (tones.length < 3) {
            return { overall: 0.5, register: 0.5, emotional: 0.5, modality: 0.5 };
        }

        // Extract dimension arrays
        const dimensions = {
            register: tones.map(t => t.registerScore),
            emotional: tones.map(t => t.emotionalValence),
            modality: tones.map(t => t.modalityBalance),
            hedging: tones.map(t => t.hedgingRate),
            pronoun: tones.map(t => t.pronounBalance)
        };

        // Calculate uniformity for each dimension
        const stabilities = {};
        for (const [name, values] of Object.entries(dimensions)) {
            stabilities[name] = VarianceUtils.uniformityScore(values);
        }

        // Overall stability is average of all dimensions
        stabilities.overall = Utils.mean(Object.values(stabilities));

        return stabilities;
    },

    /**
     * Calculate drift (change over document length)
     */
    calculateDrift(paragraphTones) {
        if (paragraphTones.length < 2) {
            return { hasDrift: false, magnitude: 0 };
        }

        const firstHalf = paragraphTones.slice(0, Math.floor(paragraphTones.length / 2));
        const secondHalf = paragraphTones.slice(Math.floor(paragraphTones.length / 2));

        const avgFirst = {
            register: Utils.mean(firstHalf.map(t => t.registerScore)),
            emotional: Utils.mean(firstHalf.map(t => t.emotionalValence)),
            hedging: Utils.mean(firstHalf.map(t => t.hedgingRate))
        };

        const avgSecond = {
            register: Utils.mean(secondHalf.map(t => t.registerScore)),
            emotional: Utils.mean(secondHalf.map(t => t.emotionalValence)),
            hedging: Utils.mean(secondHalf.map(t => t.hedgingRate))
        };

        const driftMagnitudes = {
            register: Math.abs(avgSecond.register - avgFirst.register),
            emotional: Math.abs(avgSecond.emotional - avgFirst.emotional),
            hedging: Math.abs(avgSecond.hedging - avgFirst.hedging)
        };

        const overallDrift = Utils.mean(Object.values(driftMagnitudes));

        return {
            hasDrift: overallDrift > 0.1,
            magnitude: overallDrift,
            dimensions: driftMagnitudes,
            direction: avgSecond.register > avgFirst.register ? 'more-formal' : 'less-formal'
        };
    },

    /**
     * Calculate consistency (how often tone stays similar)
     */
    calculateConsistency(tones) {
        if (tones.length < 2) {
            return { transitions: 0, smoothness: 1 };
        }

        let abruptChanges = 0;
        const changeThreshold = 0.15;

        for (let i = 1; i < tones.length; i++) {
            const registerChange = Math.abs(tones[i].registerScore - tones[i-1].registerScore);
            const emotionalChange = Math.abs(tones[i].emotionalValence - tones[i-1].emotionalValence);
            
            if (registerChange > changeThreshold || emotionalChange > changeThreshold) {
                abruptChanges++;
            }
        }

        return {
            transitions: abruptChanges,
            smoothness: 1 - (abruptChanges / (tones.length - 1))
        };
    },

    /**
     * Detect abrupt tone shifts
     */
    detectToneShifts(tones) {
        const shifts = [];
        
        for (let i = 1; i < tones.length; i++) {
            const prev = tones[i-1];
            const curr = tones[i];

            // Check for significant shifts
            if (Math.abs(curr.registerScore - prev.registerScore) > 0.3) {
                shifts.push({
                    position: i,
                    type: 'register',
                    from: prev.registerScore > 0.5 ? 'formal' : 'informal',
                    to: curr.registerScore > 0.5 ? 'formal' : 'informal',
                    magnitude: Math.abs(curr.registerScore - prev.registerScore)
                });
            }

            if (Math.abs(curr.emotionalValence - prev.emotionalValence) > 0.1) {
                shifts.push({
                    position: i,
                    type: 'emotional',
                    from: prev.emotionalValence > 0 ? 'positive' : 'negative',
                    to: curr.emotionalValence > 0 ? 'positive' : 'negative',
                    magnitude: Math.abs(curr.emotionalValence - prev.emotionalValence)
                });
            }

            if (Math.abs(curr.modalityBalance - prev.modalityBalance) > 0.3) {
                shifts.push({
                    position: i,
                    type: 'modality',
                    from: prev.modalityBalance > 0.5 ? 'certain' : 'hedged',
                    to: curr.modalityBalance > 0.5 ? 'certain' : 'hedged',
                    magnitude: Math.abs(curr.modalityBalance - prev.modalityBalance)
                });
            }
        }

        return {
            count: shifts.length,
            shifts,
            hasUnexplainedShifts: shifts.filter(s => s.magnitude > 0.4).length > 0
        };
    },

    /**
     * Analyze hedging pattern consistency
     */
    analyzeHedgingPattern(tones) {
        const hedgingRates = tones.map(t => t.hedgingRate);
        
        if (hedgingRates.every(r => r === 0)) {
            return { consistency: 1, pattern: 'none', avgRate: 0 };
        }

        const avgRate = Utils.mean(hedgingRates);
        const uniformity = VarianceUtils.uniformityScore(hedgingRates);
        
        // Check for periodicity (AI often hedges at regular intervals)
        const periodicity = VarianceUtils.detectPeriodicity(hedgingRates);

        return {
            consistency: uniformity,
            pattern: periodicity > 0.7 ? 'periodic' : (uniformity > 0.7 ? 'uniform' : 'natural'),
            avgRate,
            periodicity,
            isExcessivelyConsistent: uniformity > 0.8 && avgRate > 0.02
        };
    },

    /**
     * Analyze emotional variance
     */
    analyzeEmotionalVariance(tones) {
        const valences = tones.map(t => t.emotionalValence);
        const positiveRates = tones.map(t => t.positiveRate);
        const negativeRates = tones.map(t => t.negativeRate);

        const valenceVariance = Utils.variance(valences);
        const avgAbsValence = Utils.mean(valences.map(Math.abs));

        // "Flatness" - lack of emotional variation
        const flatness = 1 - Math.min(1, Math.sqrt(valenceVariance) * 10);
        
        // Over-neutral profile (AI tendency)
        const isOverNeutral = avgAbsValence < 0.01 && flatness > 0.8;

        return {
            variance: valenceVariance,
            flatness,
            isOverNeutral,
            avgValence: Utils.mean(valences),
            pattern: flatness > 0.85 ? 'flat' : (flatness > 0.6 ? 'stable' : 'varied')
        };
    },

    /**
     * Detect AI helper/assistant tone (Gemini, Claude, ChatGPT characteristic)
     */
    detectAIHelperTone(text) {
        const lower = text.toLowerCase();
        let patternMatches = 0;
        const patternsFound = [];
        
        // Check enthusiastic patterns
        for (const pattern of this.geminiEnthusiasticPatterns) {
            const matches = text.match(pattern);
            if (matches) {
                patternMatches += matches.length;
                patternsFound.push(matches[0].toLowerCase());
            }
        }
        
        // Check helper tone words
        const tokens = Utils.tokenize(lower);
        let helperWordCount = 0;
        for (const phrase of this.lexicons.aiHelpfulTone) {
            if (lower.includes(phrase)) {
                helperWordCount++;
                if (!patternsFound.includes(phrase)) {
                    patternsFound.push(phrase);
                }
            }
        }
        
        // Check for characteristic openings
        const aiOpenings = [
            'here is', 'here are', 'here\'s a',
            'let me', 'i\'d be happy', 'i can help',
            'absolutely', 'certainly', 'of course',
            'great question', 'that\'s a great'
        ];
        
        let openingScore = 0;
        for (const opening of aiOpenings) {
            if (lower.startsWith(opening) || lower.slice(0, 100).includes(opening)) {
                openingScore++;
                if (!patternsFound.includes(opening)) {
                    patternsFound.push('opening: ' + opening);
                }
            }
        }
        
        // Calculate overall helper tone score
        const totalSignals = patternMatches + helperWordCount + openingScore;
        const score = Utils.normalize(totalSignals, 0, 6);
        
        return {
            patternMatches,
            helperWordCount,
            openingScore,
            totalSignals,
            patternsFound: [...new Set(patternsFound)].slice(0, 5),
            score,
            isHelperTone: totalSignals >= 2
        };
    },

    /**
     * Calculate AI probability
     */
    calculateAIProbability(scores, stability, toneShifts, aiHelperTone) {
        // High stability = AI-like
        // Low emotional variance = AI-like
        // Consistent hedging = AI-like
        // AI helper tone = strongly AI-like
        
        let probability = 0;

        // Stability contributions
        probability += scores.overallStability * 0.2;
        probability += scores.hedgingConsistency * 0.15;
        probability += scores.emotionalFlatness * 0.2;
        probability += scores.registerStability * 0.1;
        
        // AI helper tone contribution (Gemini/Claude specific)
        probability += (scores.aiHelperTone || 0) * 0.2;

        // Adjustments
        if (stability.overall > 0.85) {
            probability += 0.1; // Very stable tone is AI signal
        }

        if (toneShifts.count === 0 && stability.overall > 0.7) {
            probability += 0.05; // No natural shifts
        }
        
        // Gemini/Claude helper tone bonus
        if (aiHelperTone && aiHelperTone.score > 0.5) {
            probability += 0.1; // Strong helper tone signature
        }

        // Human signals reduce probability
        if (toneShifts.hasUnexplainedShifts) {
            probability -= 0.1; // Unexpected shifts are human
        }

        return Math.max(0, Math.min(1, probability));
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(sentenceCount) {
        if (sentenceCount < 5) return 0.2;
        if (sentenceCount < 10) return 0.4;
        if (sentenceCount < 20) return 0.6;
        if (sentenceCount < 50) return 0.75;
        return 0.85;
    },

    /**
     * Generate findings with detailed statistics
     */
    generateFindings(stability, drift, toneShifts, hedging, emotional, aiHelperTone) {
        const findings = [];

        // AI Helper Tone findings (Gemini/Claude specific)
        if (aiHelperTone && aiHelperTone.isHelperTone) {
            findings.push({
                label: 'AI Assistant Tone',
                value: `Detected ${aiHelperTone.totalSignals} AI helper/assistant tone markers`,
                note: 'Characteristic of Gemini, Claude, and ChatGPT conversational responses',
                indicator: 'ai',
                severity: aiHelperTone.totalSignals >= 4 ? 'high' : 'medium',
                stats: {
                    enthusiasticPatterns: aiHelperTone.patternMatches,
                    helperPhrases: aiHelperTone.helperWordCount,
                    characteristicOpenings: aiHelperTone.openingScore,
                    examples: aiHelperTone.patternsFound.slice(0, 3).join(', ')
                },
                benchmark: {
                    humanRange: '0-1 helper markers',
                    aiRange: '2-10+ helper markers',
                    interpretation: 'Modern AI assistants have distinctive enthusiastic, helpful language'
                }
            });
        }

        // Stability findings
        if (stability.overall > 0.85) {
            findings.push({
                label: 'Tone Stability',
                value: 'Exceptionally stable tone throughout document',
                note: 'AI maintains consistent tone without natural variation',
                indicator: 'ai',
                severity: stability.overall > 0.92 ? 'high' : 'medium',
                stats: {
                    measured: `${(stability.overall * 100).toFixed(1)}% stability`,
                    formalityStability: stability.formality ? `${(stability.formality * 100).toFixed(1)}%` : 'N/A',
                    sentimentStability: stability.sentiment ? `${(stability.sentiment * 100).toFixed(1)}%` : 'N/A',
                    varianceDetected: `${((1 - stability.overall) * 100).toFixed(1)}%`
                },
                benchmark: {
                    humanRange: '40%–75% stability',
                    aiRange: '80%–98% stability',
                    interpretation: 'Humans naturally fluctuate in tone; AI is more consistent'
                }
            });
        } else if (stability.overall < 0.4) {
            findings.push({
                label: 'Natural Tone Variation',
                value: 'Healthy variation in tone detected',
                note: 'Natural tone fluctuation is characteristic of human writing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    measured: `${(stability.overall * 100).toFixed(1)}% stability`,
                    varianceDetected: `${((1 - stability.overall) * 100).toFixed(1)}% variance`
                },
                benchmark: {
                    humanRange: '40%–75% stability',
                    aiRange: '80%–98% stability'
                }
            });
        }

        // Drift findings
        if (drift.hasDrift) {
            findings.push({
                label: 'Tone Drift',
                value: `Tone drifts ${drift.direction} over the document`,
                note: 'Natural human tendency to shift tone as writing progresses',
                indicator: 'human',
                severity: 'low',
                stats: {
                    magnitude: `${(drift.magnitude * 100).toFixed(1)}% shift`,
                    direction: drift.direction,
                    startTone: drift.startTone || 'N/A',
                    endTone: drift.endTone || 'N/A'
                },
                benchmark: {
                    humanRange: '10%–40% drift common',
                    aiRange: '0%–10% drift typical',
                    note: 'AI maintains tone; humans naturally drift'
                }
            });
        }

        // Hedging findings
        if (hedging.isExcessivelyConsistent) {
            findings.push({
                label: 'Hedging Pattern',
                value: 'Hedging language used with unusual consistency',
                note: 'AI often uses hedging words ("perhaps", "may", "could") systematically',
                indicator: 'ai',
                severity: hedging.consistency > 0.9 ? 'high' : 'medium',
                stats: {
                    consistency: `${(hedging.consistency * 100).toFixed(1)}%`,
                    hedgeWordCount: hedging.count || 'N/A',
                    hedgeWordsFound: hedging.examples ? hedging.examples.slice(0, 5).join(', ') : 'N/A',
                    density: hedging.density ? `${(hedging.density * 100).toFixed(2)}% of words` : 'N/A'
                },
                benchmark: {
                    humanRange: 'Irregular hedging distribution',
                    aiRange: 'Even hedging distribution',
                    note: 'Humans hedge more in uncertain areas; AI hedges uniformly'
                }
            });
        }

        if (hedging.pattern === 'periodic') {
            findings.push({
                label: 'Periodic Hedging',
                value: 'Hedging appears at regular intervals',
                note: 'Suggests formulaic generation with systematic hedging insertion',
                indicator: 'ai',
                severity: 'medium',
                stats: {
                    pattern: 'Periodic/Regular',
                    interval: hedging.interval ? `Every ~${hedging.interval} sentences` : 'N/A'
                },
                benchmark: {
                    humanRange: 'Random hedging placement',
                    aiRange: 'Regular hedging intervals'
                }
            });
        }

        // Emotional findings
        if (emotional.isOverNeutral) {
            findings.push({
                label: 'Emotional Profile',
                value: 'Over-neutral emotional profile',
                note: 'AI tends toward emotional flattening and neutral tone',
                indicator: 'ai',
                severity: 'high',
                stats: {
                    neutrality: emotional.neutralScore ? `${(emotional.neutralScore * 100).toFixed(1)}%` : 'Very high',
                    emotionalRange: emotional.range ? `${(emotional.range * 100).toFixed(1)}% range` : 'Minimal',
                    positiveWords: emotional.positiveCount || 0,
                    negativeWords: emotional.negativeCount || 0
                },
                benchmark: {
                    humanRange: '30%–70% emotional variation',
                    aiRange: '80%–95% neutral/flat',
                    interpretation: 'Humans express more emotional peaks and valleys'
                }
            });
        }

        if (emotional.pattern === 'flat') {
            findings.push({
                label: 'Flat Emotional Tone',
                value: `Emotional tone is uniformly flat`,
                note: 'Lacks natural human expression variance',
                indicator: 'ai',
                severity: 'medium',
                stats: {
                    flatness: `${(emotional.flatness * 100).toFixed(1)}% uniformity`,
                    varianceScore: emotional.variance ? emotional.variance.toFixed(3) : 'Very low'
                },
                benchmark: {
                    humanRange: 'Variable emotional intensity',
                    aiRange: 'Consistent emotional level'
                }
            });
        }

        // Tone shift findings
        if (toneShifts.hasUnexplainedShifts) {
            findings.push({
                label: 'Abrupt Tone Shifts',
                value: `${toneShifts.count} sudden tone changes detected`,
                note: 'May indicate human emotion, multiple authors, or editing',
                indicator: 'human',
                severity: 'low',
                stats: {
                    shiftCount: toneShifts.count,
                    locations: toneShifts.positions ? `At sentences: ${toneShifts.positions.slice(0, 5).join(', ')}` : 'N/A',
                    avgMagnitude: toneShifts.avgMagnitude ? `${(toneShifts.avgMagnitude * 100).toFixed(1)}% avg shift` : 'N/A'
                },
                benchmark: {
                    humanRange: '2–6 tone shifts per 1000 words',
                    aiRange: '0–1 tone shifts per 1000 words',
                    note: 'Human writing has natural emotional rhythm'
                }
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
    module.exports = ToneAnalyzer;
}
