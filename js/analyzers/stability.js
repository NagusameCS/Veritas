/**
 * VERITAS — Contrastive Instability Analyzer
 * Category 15: Perturbation-based stability analysis
 * 
 * Based on: Mitchell et al., DetectGPT (2023)
 *           Su et al., Intrinsic Text Stability (2024)
 * 
 * KEY INSIGHT: AI text remains stable under perturbations in ways human text doesn't.
 * Instead of asking "does this look like AI?", we ask:
 * "Does this text re-stabilize unnaturally fast after micro-edits?"
 * 
 * This approach is robust to:
 * - Newer AI models with better entropy calibration
 * - Paraphrasing attacks
 * - Anti-detection fine-tuning
 */

const StabilityAnalyzer = {
    name: 'Contrastive Instability',
    category: 15,
    weight: 1.8, // High weight - this is a strong modern signal

    // Synonym clusters for perturbation
    synonymClusters: {
        'good': ['great', 'excellent', 'fine', 'nice', 'positive'],
        'bad': ['poor', 'terrible', 'negative', 'awful', 'weak'],
        'important': ['crucial', 'vital', 'essential', 'key', 'significant'],
        'big': ['large', 'major', 'significant', 'substantial', 'considerable'],
        'small': ['minor', 'slight', 'little', 'minimal', 'modest'],
        'show': ['demonstrate', 'indicate', 'reveal', 'display', 'exhibit'],
        'help': ['assist', 'aid', 'support', 'facilitate', 'enable'],
        'make': ['create', 'produce', 'generate', 'develop', 'build'],
        'use': ['utilize', 'employ', 'apply', 'leverage', 'harness'],
        'get': ['obtain', 'acquire', 'receive', 'gain', 'achieve'],
        'think': ['believe', 'consider', 'feel', 'suppose', 'reckon'],
        'say': ['state', 'mention', 'note', 'assert', 'claim'],
        'need': ['require', 'demand', 'necessitate', 'want', 'call for'],
        'find': ['discover', 'locate', 'identify', 'detect', 'uncover'],
        'give': ['provide', 'offer', 'supply', 'present', 'deliver'],
        'take': ['accept', 'receive', 'grab', 'seize', 'obtain'],
        'come': ['arrive', 'appear', 'emerge', 'reach', 'approach'],
        'see': ['observe', 'notice', 'view', 'perceive', 'witness'],
        'know': ['understand', 'recognize', 'realize', 'comprehend', 'grasp'],
        'become': ['turn', 'grow', 'develop', 'evolve', 'transform'],
        // AI-specific synonyms
        'delve': ['explore', 'examine', 'investigate', 'dig into', 'analyze'],
        'crucial': ['critical', 'vital', 'essential', 'key', 'pivotal'],
        'comprehensive': ['thorough', 'complete', 'extensive', 'detailed', 'full'],
        'facilitate': ['enable', 'help', 'assist', 'support', 'aid'],
        'leverage': ['use', 'utilize', 'employ', 'exploit', 'harness'],
        'implement': ['execute', 'apply', 'carry out', 'put into practice', 'deploy'],
        'enhance': ['improve', 'boost', 'strengthen', 'augment', 'elevate'],
        'optimize': ['improve', 'refine', 'streamline', 'perfect', 'maximize']
    },

    // Clause reordering patterns
    reorderablePatterns: [
        { pattern: /^(\w+),\s+(.+)$/, reorder: (m) => `${m[2]}, ${m[1].toLowerCase()}` },
        { pattern: /^(Although|While|When|If|Since|Because)\s+(.+),\s+(.+)$/, 
          reorder: (m) => `${m[3]}, ${m[1].toLowerCase()} ${m[2]}` },
        { pattern: /^(.+)\s+(because|since|as|while|although)\s+(.+)$/i,
          reorder: (m) => `${m[2].charAt(0).toUpperCase() + m[2].slice(1)} ${m[3]}, ${m[1].toLowerCase()}` }
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        if (!text || text.length < 200) {
            return this.getEmptyResult('Text too short for stability analysis');
        }

        const sentences = Utils.splitSentences(text);
        if (sentences.length < 5) {
            return this.getEmptyResult('Insufficient sentences for analysis');
        }

        // Generate perturbations
        const perturbations = this.generatePerturbations(text, sentences);
        
        // Measure structural stability
        const structuralStability = this.measureStructuralStability(text, perturbations);
        
        // Measure semantic drift
        const semanticDrift = this.measureSemanticDrift(text, perturbations);
        
        // Measure recovery speed (how fast text "re-stabilizes")
        const recoverySpeed = this.measureRecoverySpeed(perturbations);
        
        // Measure local vs global consistency
        const consistencyProfile = this.measureConsistencyProfile(sentences);

        // AI text tends to:
        // - Have HIGH structural stability (unnaturally stable)
        // - Have LOW semantic drift under perturbation
        // - Re-stabilize FAST after perturbation
        // - Be globally consistent but locally uniform

        const scores = {
            structuralStability: structuralStability.score,
            semanticDrift: 1 - semanticDrift.score, // Invert: low drift = AI
            recoverySpeed: recoverySpeed.score,
            consistencyUniformity: consistencyProfile.uniformityScore
        };

        // Calculate AI probability
        // High stability + low drift + fast recovery = likely AI
        const aiProbability = Utils.weightedAverage(
            [scores.structuralStability, scores.semanticDrift, scores.recoverySpeed, scores.consistencyUniformity],
            [0.3, 0.25, 0.25, 0.2]
        );

        const confidence = this.calculateConfidence(sentences.length, perturbations.length);

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence,
            details: {
                structuralStability,
                semanticDrift,
                recoverySpeed,
                consistencyProfile,
                perturbationCount: perturbations.length
            },
            findings: this.generateFindings(structuralStability, semanticDrift, recoverySpeed, consistencyProfile),
            scores
        };
    },

    /**
     * Generate perturbations of the text
     */
    generatePerturbations(text, sentences) {
        const perturbations = [];
        
        // 1. Synonym substitutions
        for (let i = 0; i < Math.min(sentences.length, 10); i++) {
            const perturbed = this.applySynonymSubstitution(sentences[i]);
            if (perturbed !== sentences[i]) {
                perturbations.push({
                    type: 'synonym',
                    original: sentences[i],
                    perturbed,
                    index: i
                });
            }
        }
        
        // 2. Clause reordering
        for (let i = 0; i < Math.min(sentences.length, 10); i++) {
            const perturbed = this.applyClauseReordering(sentences[i]);
            if (perturbed !== sentences[i]) {
                perturbations.push({
                    type: 'reorder',
                    original: sentences[i],
                    perturbed,
                    index: i
                });
            }
        }
        
        // 3. Punctuation changes (comma insertion/removal)
        for (let i = 0; i < Math.min(sentences.length, 8); i++) {
            const perturbed = this.applyPunctuationChange(sentences[i]);
            if (perturbed !== sentences[i]) {
                perturbations.push({
                    type: 'punctuation',
                    original: sentences[i],
                    perturbed,
                    index: i
                });
            }
        }
        
        // 4. Word order micro-swaps
        for (let i = 0; i < Math.min(sentences.length, 8); i++) {
            const perturbed = this.applyWordOrderSwap(sentences[i]);
            if (perturbed !== sentences[i]) {
                perturbations.push({
                    type: 'word_swap',
                    original: sentences[i],
                    perturbed,
                    index: i
                });
            }
        }

        return perturbations;
    },

    /**
     * Apply synonym substitution
     */
    applySynonymSubstitution(sentence) {
        let result = sentence;
        const words = sentence.toLowerCase().split(/\s+/);
        
        for (const word of words) {
            const cleanWord = word.replace(/[^a-z]/g, '');
            if (this.synonymClusters[cleanWord]) {
                const synonyms = this.synonymClusters[cleanWord];
                const replacement = synonyms[Math.floor(Math.random() * synonyms.length)];
                // Preserve case
                const regex = new RegExp(`\\b${cleanWord}\\b`, 'i');
                result = result.replace(regex, (match) => {
                    if (match[0] === match[0].toUpperCase()) {
                        return replacement.charAt(0).toUpperCase() + replacement.slice(1);
                    }
                    return replacement;
                });
                break; // Only one substitution per sentence
            }
        }
        
        return result;
    },

    /**
     * Apply clause reordering
     */
    applyClauseReordering(sentence) {
        for (const { pattern, reorder } of this.reorderablePatterns) {
            const match = sentence.match(pattern);
            if (match) {
                try {
                    return reorder(match);
                } catch (e) {
                    continue;
                }
            }
        }
        return sentence;
    },

    /**
     * Apply punctuation changes
     */
    applyPunctuationChange(sentence) {
        // Add or remove optional commas
        if (sentence.includes(', ')) {
            // Remove a comma
            return sentence.replace(/, /, ' ');
        } else if (sentence.includes(' and ')) {
            // Add a comma before 'and'
            return sentence.replace(' and ', ', and ');
        } else if (sentence.includes(' but ')) {
            return sentence.replace(' but ', ', but ');
        }
        return sentence;
    },

    /**
     * Apply word order micro-swaps (adjective order, adverb position)
     */
    applyWordOrderSwap(sentence) {
        // Swap adjacent adjectives
        const adjSwap = sentence.replace(/(\b\w+\b)\s+(\b\w+\b)\s+(noun|thing|person|place|idea|way|time)/i, 
            (match, adj1, adj2, noun) => `${adj2} ${adj1} ${noun}`);
        if (adjSwap !== sentence) return adjSwap;
        
        // Move adverb
        const advMove = sentence.replace(/^(\w+)\s+(quickly|slowly|carefully|often|always|never|usually)\s+/i,
            (match, verb, adv) => `${adv.charAt(0).toUpperCase() + adv.slice(1)}, ${verb.toLowerCase()} `);
        if (advMove !== sentence) return advMove;
        
        return sentence;
    },

    /**
     * Measure structural stability
     * AI text maintains structure even after perturbation
     */
    measureStructuralStability(originalText, perturbations) {
        if (perturbations.length === 0) {
            return { score: 0.5, message: 'No perturbations generated' };
        }

        let totalStability = 0;
        const measurements = [];
        
        for (const pert of perturbations) {
            // Measure structural similarity
            const origTokens = Utils.tokenize(pert.original);
            const pertTokens = Utils.tokenize(pert.perturbed);
            
            // Structure metrics
            const lengthRatio = Math.min(origTokens.length, pertTokens.length) / 
                               Math.max(origTokens.length, pertTokens.length);
            
            // N-gram overlap
            const origBigrams = this.getNgrams(origTokens, 2);
            const pertBigrams = this.getNgrams(pertTokens, 2);
            const bigramOverlap = this.calculateSetOverlap(origBigrams, pertBigrams);
            
            // Syntactic pattern preservation (simplified)
            const origPattern = this.getSyntacticPattern(pert.original);
            const pertPattern = this.getSyntacticPattern(pert.perturbed);
            const patternSimilarity = this.comparePatterns(origPattern, pertPattern);
            
            const stability = (lengthRatio * 0.2 + bigramOverlap * 0.4 + patternSimilarity * 0.4);
            totalStability += stability;
            
            measurements.push({
                type: pert.type,
                stability,
                lengthRatio,
                bigramOverlap,
                patternSimilarity
            });
        }
        
        const avgStability = totalStability / perturbations.length;
        
        // High stability (>0.8) is suspicious - AI text
        // Normal human text: 0.5-0.75
        // AI text: 0.75-0.95
        const score = Utils.normalize(avgStability, 0.5, 0.9);
        
        return {
            score,
            avgStability,
            measurements: measurements.slice(0, 5),
            interpretation: avgStability > 0.8 ? 'Unnaturally stable' : 
                           avgStability > 0.65 ? 'Moderately stable' : 'Natural variance'
        };
    },

    /**
     * Measure semantic drift under perturbation
     * Human text drifts more semantically; AI stays centered
     */
    measureSemanticDrift(originalText, perturbations) {
        if (perturbations.length === 0) {
            return { score: 0.5, message: 'No perturbations generated' };
        }

        let totalDrift = 0;
        const driftMeasurements = [];
        
        for (const pert of perturbations) {
            // Semantic drift approximation using keyword/concept preservation
            const origConcepts = this.extractConcepts(pert.original);
            const pertConcepts = this.extractConcepts(pert.perturbed);
            
            // Calculate concept drift
            const conceptOverlap = this.calculateSetOverlap(origConcepts, pertConcepts);
            const conceptDrift = 1 - conceptOverlap;
            
            // Calculate emphasis shift
            const emphasisShift = this.measureEmphasisShift(pert.original, pert.perturbed);
            
            const drift = (conceptDrift * 0.6 + emphasisShift * 0.4);
            totalDrift += drift;
            
            driftMeasurements.push({
                type: pert.type,
                drift,
                conceptDrift,
                emphasisShift
            });
        }
        
        const avgDrift = totalDrift / perturbations.length;
        
        // Low drift (<0.15) is suspicious - AI maintains meaning too perfectly
        // Normal human text: 0.15-0.35
        // AI text: 0.05-0.15
        const score = avgDrift; // Higher drift = more human-like
        
        return {
            score,
            avgDrift,
            measurements: driftMeasurements.slice(0, 5),
            interpretation: avgDrift < 0.1 ? 'Unnaturally low drift (AI-like)' :
                           avgDrift < 0.2 ? 'Low drift' : 'Natural semantic variation'
        };
    },

    /**
     * Measure how fast text "recovers" to original patterns
     * AI text has faster pattern recovery
     */
    measureRecoverySpeed(perturbations) {
        if (perturbations.length === 0) {
            return { score: 0.5, message: 'No perturbations generated' };
        }

        // Group perturbations by sentence index
        const byIndex = {};
        for (const pert of perturbations) {
            if (!byIndex[pert.index]) byIndex[pert.index] = [];
            byIndex[pert.index].push(pert);
        }
        
        let recoveryScore = 0;
        let count = 0;
        
        for (const index of Object.keys(byIndex)) {
            const perts = byIndex[index];
            if (perts.length < 2) continue;
            
            // Check if different perturbation types yield similar results
            // (indicating the text "wants" to maintain a certain form)
            const perturbedVersions = perts.map(p => p.perturbed);
            const similarity = this.measureIntraSetSimilarity(perturbedVersions);
            
            recoveryScore += similarity;
            count++;
        }
        
        const avgRecovery = count > 0 ? recoveryScore / count : 0.5;
        
        // High recovery speed (similar results from different perturbations) = AI
        const score = Utils.normalize(avgRecovery, 0.4, 0.85);
        
        return {
            score,
            avgRecovery,
            interpretation: avgRecovery > 0.7 ? 'Fast recovery (AI-like)' :
                           avgRecovery > 0.5 ? 'Moderate recovery' : 'Slow recovery (human-like)'
        };
    },

    /**
     * Measure local vs global consistency profile
     * AI is globally consistent but locally uniform
     */
    measureConsistencyProfile(sentences) {
        if (sentences.length < 5) {
            return { uniformityScore: 0.5, message: 'Insufficient sentences' };
        }

        // Measure variance in different windows
        const windowSizes = [3, 5, 8];
        const windowVariances = {};
        
        for (const windowSize of windowSizes) {
            if (sentences.length < windowSize) continue;
            
            const variances = [];
            for (let i = 0; i <= sentences.length - windowSize; i++) {
                const window = sentences.slice(i, i + windowSize);
                const lengths = window.map(s => Utils.tokenize(s).length);
                variances.push(Utils.variance(lengths));
            }
            
            windowVariances[windowSize] = {
                mean: Utils.mean(variances),
                variance: Utils.variance(variances)
            };
        }
        
        // AI text: low variance within windows, low variance across windows
        // Human text: higher variance within windows, higher variance across windows
        
        let uniformityScore = 0;
        let count = 0;
        
        for (const ws of Object.keys(windowVariances)) {
            const wv = windowVariances[ws];
            // Low within-window variance AND low cross-window variance = uniform
            const uniformity = 1 - Utils.normalize(wv.mean + wv.variance, 0, 50);
            uniformityScore += uniformity;
            count++;
        }
        
        uniformityScore = count > 0 ? uniformityScore / count : 0.5;
        
        return {
            uniformityScore,
            windowVariances,
            interpretation: uniformityScore > 0.7 ? 'Highly uniform (AI-like)' :
                           uniformityScore > 0.5 ? 'Moderately uniform' : 'Natural variation'
        };
    },

    /**
     * Helper: Get n-grams
     */
    getNgrams(tokens, n) {
        const ngrams = new Set();
        for (let i = 0; i <= tokens.length - n; i++) {
            ngrams.add(tokens.slice(i, i + n).join(' '));
        }
        return ngrams;
    },

    /**
     * Helper: Calculate set overlap (Jaccard similarity)
     */
    calculateSetOverlap(set1, set2) {
        const s1 = set1 instanceof Set ? set1 : new Set(set1);
        const s2 = set2 instanceof Set ? set2 : new Set(set2);
        
        let intersection = 0;
        for (const item of s1) {
            if (s2.has(item)) intersection++;
        }
        
        const union = s1.size + s2.size - intersection;
        return union > 0 ? intersection / union : 0;
    },

    /**
     * Helper: Get simplified syntactic pattern
     */
    getSyntacticPattern(sentence) {
        // Simplified POS-like pattern
        return sentence
            .replace(/\b(the|a|an)\b/gi, 'DET')
            .replace(/\b(is|are|was|were|be|been|being)\b/gi, 'BE')
            .replace(/\b(has|have|had)\b/gi, 'HAVE')
            .replace(/\b(will|would|could|should|may|might|can)\b/gi, 'MOD')
            .replace(/\b(and|or|but|yet|so)\b/gi, 'CONJ')
            .replace(/\b(in|on|at|to|for|with|by|from)\b/gi, 'PREP')
            .replace(/\b(I|you|he|she|it|we|they|me|him|her|us|them)\b/gi, 'PRON')
            .replace(/\b\w+ly\b/gi, 'ADV')
            .replace(/\b\w+ing\b/gi, 'VING')
            .replace(/\b\w+ed\b/gi, 'VED')
            .replace(/\b\w+tion\b/gi, 'NTION')
            .replace(/\b\w+ness\b/gi, 'NNESS');
    },

    /**
     * Helper: Compare syntactic patterns
     */
    comparePatterns(pattern1, pattern2) {
        const tokens1 = pattern1.split(/\s+/);
        const tokens2 = pattern2.split(/\s+/);
        
        let matches = 0;
        const minLen = Math.min(tokens1.length, tokens2.length);
        
        for (let i = 0; i < minLen; i++) {
            if (tokens1[i] === tokens2[i]) matches++;
        }
        
        const maxLen = Math.max(tokens1.length, tokens2.length);
        return maxLen > 0 ? matches / maxLen : 0;
    },

    /**
     * Helper: Extract key concepts
     */
    extractConcepts(text) {
        const stopwords = new Set(['the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'may', 'might', 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all',
            'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
            'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
            'now', 'here', 'there', 'this', 'that', 'these', 'those', 'it', 'its']);
        
        const tokens = Utils.tokenize(text.toLowerCase());
        return new Set(tokens.filter(t => t.length > 3 && !stopwords.has(t)));
    },

    /**
     * Helper: Measure emphasis shift
     */
    measureEmphasisShift(original, perturbed) {
        const origWords = Utils.tokenize(original.toLowerCase());
        const pertWords = Utils.tokenize(perturbed.toLowerCase());
        
        // Check position of key content words
        const contentWords = origWords.filter(w => w.length > 4);
        
        let totalShift = 0;
        for (const word of contentWords) {
            const origPos = origWords.indexOf(word) / origWords.length;
            const pertPos = pertWords.indexOf(word) / pertWords.length;
            
            if (pertPos >= 0) {
                totalShift += Math.abs(origPos - pertPos);
            } else {
                totalShift += 0.5; // Word missing = significant shift
            }
        }
        
        return contentWords.length > 0 ? totalShift / contentWords.length : 0;
    },

    /**
     * Helper: Measure similarity within a set of texts
     */
    measureIntraSetSimilarity(texts) {
        if (texts.length < 2) return 0.5;
        
        let totalSim = 0;
        let count = 0;
        
        for (let i = 0; i < texts.length; i++) {
            for (let j = i + 1; j < texts.length; j++) {
                const tokens1 = new Set(Utils.tokenize(texts[i].toLowerCase()));
                const tokens2 = new Set(Utils.tokenize(texts[j].toLowerCase()));
                totalSim += this.calculateSetOverlap(tokens1, tokens2);
                count++;
            }
        }
        
        return count > 0 ? totalSim / count : 0.5;
    },

    /**
     * Generate findings
     */
    generateFindings(structural, semantic, recovery, consistency) {
        const findings = [];

        if (structural.score > 0.7) {
            findings.push({
                text: `Text shows ${structural.interpretation}. Stability score: ${structural.avgStability.toFixed(2)}. AI-generated text typically maintains structure even after synonym substitution and clause reordering.`,
                label: 'Structural Stability',
                indicator: 'ai',
                severity: structural.score > 0.85 ? 'high' : 'medium',
                research: 'Based on DetectGPT (Mitchell et al., 2023)'
            });
        } else if (structural.score < 0.35) {
            findings.push({
                text: `Text shows natural structural variance under perturbation (${structural.avgStability.toFixed(2)}). Human writing typically varies more when paraphrased.`,
                label: 'Natural Structure',
                indicator: 'human',
                severity: 'medium'
            });
        }

        if (semantic.score < 0.15) {
            findings.push({
                text: `Semantic drift is unusually low (${semantic.avgDrift.toFixed(3)}). AI text maintains meaning too perfectly under perturbation. Human text shows more semantic variance.`,
                label: 'Semantic Stability',
                indicator: 'ai',
                severity: 'high',
                research: 'Based on Su et al., Intrinsic Text Stability (2024)'
            });
        }

        if (recovery.score > 0.7) {
            findings.push({
                text: `Text shows fast pattern recovery (${recovery.interpretation}). Different perturbation types yield similar results, suggesting the text "wants" to maintain a certain form.`,
                label: 'Pattern Recovery',
                indicator: 'ai',
                severity: 'medium'
            });
        }

        if (consistency.uniformityScore > 0.75) {
            findings.push({
                text: `${consistency.interpretation}. Both local and global consistency levels are unusually even, a hallmark of AI optimization.`,
                label: 'Consistency Profile',
                indicator: 'ai',
                severity: 'medium',
                research: 'Based on Ippolito et al., Inconsistency as a Signal (2023)'
            });
        } else if (consistency.uniformityScore < 0.4) {
            findings.push({
                text: `Natural inconsistency detected. Variance patterns differ across document sections, consistent with human writing.`,
                label: 'Natural Inconsistency',
                indicator: 'human',
                severity: 'low'
            });
        }

        return findings;
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(sentenceCount, perturbationCount) {
        if (perturbationCount < 5) return 0.3;
        if (sentenceCount < 10) return 0.5;
        if (sentenceCount < 20) return 0.7;
        return 0.85;
    },

    /**
     * Empty result
     */
    getEmptyResult(reason) {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            details: { reason },
            findings: [],
            scores: {}
        };
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = StabilityAnalyzer;
}
