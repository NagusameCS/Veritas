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
        ]
    },

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

        // Calculate scores
        const scores = {
            overallStability: stability.overall,
            hedgingConsistency: hedgingPattern.consistency,
            emotionalFlatness: emotionalProfile.flatness,
            registerStability: stability.register
        };

        // AI probability based on excessive stability
        const aiProbability = this.calculateAIProbability(scores, stability, toneShifts);
        
        const confidence = this.calculateConfidence(sentences.length);
        const findings = this.generateFindings(stability, drift, toneShifts, hedgingPattern, emotionalProfile);

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
     * Calculate AI probability
     */
    calculateAIProbability(scores, stability, toneShifts) {
        // High stability = AI-like
        // Low emotional variance = AI-like
        // Consistent hedging = AI-like
        
        let probability = 0;

        // Stability contributions
        probability += scores.overallStability * 0.25;
        probability += scores.hedgingConsistency * 0.2;
        probability += scores.emotionalFlatness * 0.25;
        probability += scores.registerStability * 0.15;

        // Adjustments
        if (stability.overall > 0.85) {
            probability += 0.1; // Very stable tone is AI signal
        }

        if (toneShifts.count === 0 && stability.overall > 0.7) {
            probability += 0.05; // No natural shifts
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
     * Generate findings
     */
    generateFindings(stability, drift, toneShifts, hedging, emotional) {
        const findings = [];

        // Stability findings
        if (stability.overall > 0.85) {
            findings.push({
                text: `Tone is exceptionally stable (${(stability.overall * 100).toFixed(0)}%) across the document - AI maintains consistent tone`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        } else if (stability.overall < 0.4) {
            findings.push({
                text: `Natural tone variation detected (${(stability.overall * 100).toFixed(0)}% stability) - characteristic of human writing`,
                category: this.name,
                indicator: 'human',
                severity: 'medium'
            });
        }

        // Drift findings
        if (drift.hasDrift) {
            findings.push({
                text: `Tone drifts ${drift.direction} over the document (${(drift.magnitude * 100).toFixed(0)}% shift) - natural human tendency`,
                category: this.name,
                indicator: 'human',
                severity: 'medium'
            });
        }

        // Hedging findings
        if (hedging.isExcessivelyConsistent) {
            findings.push({
                text: `Hedging language used with unusual consistency (${(hedging.consistency * 100).toFixed(0)}%) - AI pattern`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (hedging.pattern === 'periodic') {
            findings.push({
                text: `Hedging appears at regular intervals - suggests formulaic generation`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Emotional findings
        if (emotional.isOverNeutral) {
            findings.push({
                text: `Over-neutral emotional profile with minimal variation - AI tendency toward emotional flattening`,
                category: this.name,
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (emotional.pattern === 'flat') {
            findings.push({
                text: `Emotional tone is flat (${(emotional.flatness * 100).toFixed(0)}% uniformity) - lacks natural human expression variance`,
                category: this.name,
                indicator: 'ai',
                severity: 'medium'
            });
        }

        // Tone shift findings
        if (toneShifts.hasUnexplainedShifts) {
            findings.push({
                text: `${toneShifts.count} abrupt tone shifts detected - may indicate human emotion or multiple authors`,
                category: this.name,
                indicator: 'human',
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
    module.exports = ToneAnalyzer;
}
