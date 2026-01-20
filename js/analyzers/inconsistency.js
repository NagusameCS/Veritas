/**
 * VERITAS — Intra-document Inconsistency Analyzer
 * Category 16: Internal variance tracking
 * 
 * Based on: Ippolito et al., Inconsistency as a Signal (2023)
 *           Krishna et al., Paraphrasing Evades Detectors (2024)
 * 
 * KEY INSIGHT: Human writing naturally drifts over a document.
 * AI maintains unnatural consistency in:
 * - Sentence opening structures
 * - Idiom usage patterns  
 * - Argument depth vs lexical sophistication ratio
 * - Error types and distributions
 * 
 * This analyzer tracks VARIANCE, not presence.
 * Low variance = suspicious uniformity = likely AI
 */

const InconsistencyAnalyzer = {
    name: 'Intra-document Inconsistency',
    category: 16,
    weight: 1.6, // Strong modern signal

    // Sentence opening patterns
    openingPatterns: {
        pronoun: /^(I|We|You|He|She|It|They|One|This|That|These|Those)\b/i,
        article: /^(The|A|An)\b/i,
        conjunction: /^(And|But|Or|So|Yet|However|Therefore|Moreover|Furthermore|Additionally|Nevertheless)\b/i,
        adverb: /^(Quickly|Slowly|Carefully|Often|Always|Never|Usually|Generally|Typically|Specifically|Interestingly|Importantly)\b/i,
        preposition: /^(In|On|At|By|With|From|To|For|Through|During|Before|After|Among|Between)\b/i,
        gerund: /^\w+ing\b/i,
        question: /^(What|Why|How|When|Where|Who|Which|Is|Are|Do|Does|Can|Could|Would|Should)\b/i,
        conditional: /^(If|When|Although|While|Since|Because|Unless|Whether)\b/i,
        number: /^(\d+|One|Two|Three|Four|Five|First|Second|Third|Many|Several|Few)\b/i,
        quotation: /^["''"]/,
        name: /^[A-Z][a-z]+\s+[A-Z]/
    },

    // Idiom categories
    idiomPatterns: {
        temporal: ['at the end of the day', 'in the long run', 'time will tell', 'sooner or later'],
        comparative: ['on the other hand', 'in contrast', 'by comparison', 'as opposed to'],
        emphasis: ['in fact', 'as a matter of fact', 'to be sure', 'without a doubt'],
        causation: ['as a result', 'due to', 'owing to', 'on account of'],
        summary: ['in summary', 'to summarize', 'in conclusion', 'all in all'],
        clarification: ['in other words', 'that is to say', 'to put it simply', 'namely'],
        addition: ['in addition', 'furthermore', 'moreover', 'what is more'],
        concession: ['having said that', 'that being said', 'even so', 'be that as it may']
    },

    // AI-typical phrasings that suggest uniformity
    aiUniformPhrases: [
        /it is (important|essential|crucial|worth) (to note|noting|mentioning)/i,
        /this (highlights|demonstrates|shows|illustrates|underscores)/i,
        /in (today's|the modern|our contemporary) world/i,
        /plays a (vital|crucial|key|important|significant) role/i,
        /it (can|could|may|might) be (argued|said|noted)/i,
        /(ensuring|guaranteeing|making sure) that/i,
        /in (various|different|many|numerous) ways/i,
        /from (this|a|the) perspective/i,
        /(aims|seeks|strives|endeavors) to/i
    ],

    /**
     * Main analysis function
     */
    analyze(text) {
        if (!text || text.length < 300) {
            return this.getEmptyResult('Text too short for inconsistency analysis');
        }

        const sentences = Utils.splitSentences(text);
        if (sentences.length < 8) {
            return this.getEmptyResult('Insufficient sentences for analysis');
        }

        // Split document into segments for local analysis
        const segments = this.segmentDocument(sentences);

        // 1. Sentence opening variance
        const openingVariance = this.analyzeOpeningVariance(sentences);
        
        // 2. Idiom usage patterns
        const idiomVariance = this.analyzeIdiomVariance(text, segments);
        
        // 3. Argument depth vs lexical sophistication
        const complexityBalance = this.analyzeComplexityBalance(sentences, segments);
        
        // 4. Error type distribution
        const errorDistribution = this.analyzeErrorDistribution(text, segments);
        
        // 5. Stylistic drift across document
        const stylisticDrift = this.analyzeStyleDrift(segments);
        
        // 6. Cross-segment consistency (too much = AI)
        const crossSegmentConsistency = this.analyzeCrossSegmentConsistency(segments);

        // Calculate AI probability
        // Low variance in any of these = AI-like
        const varianceScores = {
            openings: 1 - openingVariance.uniformityScore, // High uniformity = AI
            idioms: 1 - idiomVariance.patternScore,
            complexity: 1 - complexityBalance.imbalanceScore,
            errors: errorDistribution.naturalScore,
            drift: stylisticDrift.driftScore,
            crossSegment: 1 - crossSegmentConsistency.consistencyScore
        };

        const aiProbability = this.calculateOverallAIProbability(varianceScores);
        const confidence = this.calculateConfidence(sentences.length, segments.length);

        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence,
            details: {
                openingVariance,
                idiomVariance,
                complexityBalance,
                errorDistribution,
                stylisticDrift,
                crossSegmentConsistency,
                segmentCount: segments.length
            },
            findings: this.generateFindings(
                openingVariance, idiomVariance, complexityBalance, 
                errorDistribution, stylisticDrift, crossSegmentConsistency
            ),
            scores: varianceScores
        };
    },

    /**
     * Segment document into analysis chunks
     */
    segmentDocument(sentences) {
        const segmentSize = Math.max(3, Math.floor(sentences.length / 4));
        const segments = [];
        
        for (let i = 0; i < sentences.length; i += segmentSize) {
            segments.push({
                sentences: sentences.slice(i, i + segmentSize),
                startIndex: i,
                endIndex: Math.min(i + segmentSize, sentences.length)
            });
        }
        
        return segments;
    },

    /**
     * Analyze sentence opening variance
     */
    analyzeOpeningVariance(sentences) {
        const openingCounts = {};
        const openingSequence = [];
        
        for (const pattern of Object.keys(this.openingPatterns)) {
            openingCounts[pattern] = 0;
        }
        openingCounts['other'] = 0;

        for (const sentence of sentences) {
            let matched = false;
            for (const [pattern, regex] of Object.entries(this.openingPatterns)) {
                if (regex.test(sentence)) {
                    openingCounts[pattern]++;
                    openingSequence.push(pattern);
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                openingCounts['other']++;
                openingSequence.push('other');
            }
        }

        // Calculate entropy of opening distribution
        const total = sentences.length;
        let entropy = 0;
        for (const count of Object.values(openingCounts)) {
            if (count > 0) {
                const p = count / total;
                entropy -= p * Math.log2(p);
            }
        }

        // Check for repetitive patterns
        const repetitions = this.countConsecutiveRepeats(openingSequence);
        
        // Calculate uniformity
        const maxEntropy = Math.log2(Object.keys(this.openingPatterns).length + 1);
        const normalizedEntropy = entropy / maxEntropy;
        
        // Low entropy or high repetition = uniform = AI-like
        const uniformityScore = (1 - normalizedEntropy) * 0.6 + 
                               (repetitions / sentences.length) * 0.4;

        return {
            uniformityScore,
            entropy,
            normalizedEntropy,
            openingCounts,
            repetitions,
            mostCommon: this.getMostCommon(openingCounts),
            interpretation: uniformityScore > 0.6 ? 'Highly uniform openings (AI-like)' :
                           uniformityScore > 0.4 ? 'Moderately uniform' : 'Natural variety'
        };
    },

    /**
     * Analyze idiom usage patterns
     */
    analyzeIdiomVariance(text, segments) {
        const textLower = text.toLowerCase();
        const foundIdioms = {};
        let totalIdioms = 0;
        
        for (const [category, idioms] of Object.entries(this.idiomPatterns)) {
            foundIdioms[category] = 0;
            for (const idiom of idioms) {
                const count = (textLower.match(new RegExp(idiom.replace(/\s+/g, '\\s+'), 'gi')) || []).length;
                foundIdioms[category] += count;
                totalIdioms += count;
            }
        }

        // Check for AI-uniform phrases
        let aiPhraseCount = 0;
        for (const pattern of this.aiUniformPhrases) {
            const matches = textLower.match(pattern);
            if (matches) aiPhraseCount += matches.length;
        }

        // Idiom distribution across segments
        const segmentIdiomCounts = segments.map(seg => {
            let count = 0;
            const segText = seg.sentences.join(' ').toLowerCase();
            for (const idioms of Object.values(this.idiomPatterns)) {
                for (const idiom of idioms) {
                    if (segText.includes(idiom)) count++;
                }
            }
            return count;
        });

        // Calculate variance of idiom usage
        const idiomVariance = Utils.variance(segmentIdiomCounts);
        
        // Evenly distributed idioms = AI-like
        // Natural writing has burst patterns
        const patternScore = 1 - Utils.normalize(idiomVariance, 0, 3);

        return {
            patternScore,
            foundIdioms,
            totalIdioms,
            aiPhraseCount,
            segmentDistribution: segmentIdiomCounts,
            idiomVariance,
            interpretation: aiPhraseCount > 3 ? 'High AI-phrase density' :
                           patternScore > 0.7 ? 'Uniformly distributed idioms (AI-like)' :
                           'Natural idiom patterns'
        };
    },

    /**
     * Analyze argument depth vs lexical sophistication balance
     * AI often has mismatched complexity levels
     */
    analyzeComplexityBalance(sentences, segments) {
        const segmentMetrics = segments.map(seg => {
            const text = seg.sentences.join(' ');
            const words = Utils.tokenize(text);
            
            // Lexical sophistication: average word length, syllables
            const avgWordLength = words.reduce((sum, w) => sum + w.length, 0) / words.length;
            const longWords = words.filter(w => w.length > 8).length / words.length;
            const lexicalSophistication = (avgWordLength / 10 + longWords) / 2;
            
            // Argument depth: subordinate clauses, connectives, nested structures
            const subordinates = (text.match(/\b(because|although|while|if|when|since|unless|whereas|whether)\b/gi) || []).length;
            const connectives = (text.match(/\b(therefore|however|moreover|furthermore|consequently|nevertheless)\b/gi) || []).length;
            const argumentDepth = (subordinates + connectives * 2) / seg.sentences.length;
            
            return { lexicalSophistication, argumentDepth, segmentIndex: seg.startIndex };
        });

        // Calculate balance ratio variance across segments
        const ratios = segmentMetrics.map(m => {
            if (m.argumentDepth === 0) return m.lexicalSophistication * 10;
            return m.lexicalSophistication / (m.argumentDepth + 0.1);
        });

        const ratioVariance = Utils.variance(ratios);
        const meanRatio = Utils.mean(ratios);

        // AI text: consistent ratio across segments
        // Human text: ratios vary as argument complexity changes
        const imbalanceScore = 1 - Utils.normalize(ratioVariance, 0, 2);

        // Also check for "sophisticated vocabulary + shallow argument" pattern
        const avgLexical = Utils.mean(segmentMetrics.map(m => m.lexicalSophistication));
        const avgArgument = Utils.mean(segmentMetrics.map(m => m.argumentDepth));
        const mismatch = avgLexical > 0.4 && avgArgument < 0.3;

        return {
            imbalanceScore,
            ratioVariance,
            meanRatio,
            avgLexicalSophistication: avgLexical,
            avgArgumentDepth: avgArgument,
            hasComplexityMismatch: mismatch,
            interpretation: imbalanceScore > 0.7 ? 'Uniformly balanced complexity (AI-like)' :
                           mismatch ? 'Sophisticated vocabulary with shallow arguments' :
                           'Natural complexity variation'
        };
    },

    /**
     * Analyze error type distribution
     * Humans make consistent error patterns; AI either perfect or random
     */
    analyzeErrorDistribution(text, segments) {
        const errorTypes = {
            typo: /\b\w*[aeiou]{3,}\w*\b/g, // Triple vowels often typos
            punctuation: /[,;:]{2,}|[.!?]{3,}/g,
            spacing: /\s{3,}|\w[.!?]\w/g,
            capitalization: /[.!?]\s+[a-z]/g,
            fragments: /^[A-Z][^.!?]{1,15}[.!?]$/gm // Very short sentences
        };

        const segmentErrors = segments.map(seg => {
            const text = seg.sentences.join(' ');
            const errors = {};
            for (const [type, pattern] of Object.entries(errorTypes)) {
                errors[type] = (text.match(pattern) || []).length;
            }
            return errors;
        });

        // Calculate error consistency across segments
        const errorVariances = {};
        for (const type of Object.keys(errorTypes)) {
            const counts = segmentErrors.map(e => e[type]);
            errorVariances[type] = Utils.variance(counts);
        }

        // Total errors
        const totalErrors = segmentErrors.reduce((sum, e) => 
            sum + Object.values(e).reduce((s, v) => s + v, 0), 0
        );

        // Natural writing: consistent error patterns
        // AI: either no errors or random distribution
        const avgVariance = Utils.mean(Object.values(errorVariances));
        const naturalScore = totalErrors === 0 ? 0.2 : 
                            Utils.normalize(avgVariance, 0, 2);

        return {
            naturalScore,
            errorVariances,
            totalErrors,
            segmentErrors,
            interpretation: totalErrors === 0 ? 'Suspiciously error-free' :
                           naturalScore > 0.5 ? 'Natural error patterns' : 
                           'Inconsistent error distribution'
        };
    },

    /**
     * Analyze stylistic drift across document
     * Human writing naturally evolves; AI maintains one style
     */
    analyzeStyleDrift(segments) {
        if (segments.length < 2) {
            return { driftScore: 0.5, message: 'Insufficient segments' };
        }

        const segmentStyles = segments.map(seg => {
            const text = seg.sentences.join(' ');
            const words = Utils.tokenize(text);
            
            return {
                avgSentenceLength: Utils.mean(seg.sentences.map(s => Utils.tokenize(s).length)),
                avgWordLength: words.reduce((sum, w) => sum + w.length, 0) / words.length,
                passiveRatio: (text.match(/\b(is|are|was|were|been|be)\s+\w+ed\b/gi) || []).length / seg.sentences.length,
                questionRatio: seg.sentences.filter(s => s.trim().endsWith('?')).length / seg.sentences.length,
                exclamationRatio: seg.sentences.filter(s => s.trim().endsWith('!')).length / seg.sentences.length,
                firstPersonRatio: (text.match(/\b(I|me|my|mine|we|us|our)\b/gi) || []).length / words.length
            };
        });

        // Calculate drift between consecutive segments
        let totalDrift = 0;
        for (let i = 1; i < segmentStyles.length; i++) {
            const prev = segmentStyles[i - 1];
            const curr = segmentStyles[i];
            
            let drift = 0;
            for (const key of Object.keys(prev)) {
                drift += Math.abs(prev[key] - curr[key]);
            }
            totalDrift += drift;
        }

        const avgDrift = totalDrift / (segments.length - 1);
        
        // Higher drift = more human-like
        const driftScore = Utils.normalize(avgDrift, 0, 3);

        return {
            driftScore,
            avgDrift,
            segmentStyles,
            interpretation: driftScore < 0.25 ? 'Minimal style drift (AI-like)' :
                           driftScore < 0.5 ? 'Low style drift' : 
                           'Natural stylistic evolution'
        };
    },

    /**
     * Analyze cross-segment consistency
     * Too much consistency across distant segments = AI
     */
    analyzeCrossSegmentConsistency(segments) {
        if (segments.length < 3) {
            return { consistencyScore: 0.5, message: 'Insufficient segments' };
        }

        // Compare first vs last segment
        const first = segments[0].sentences.join(' ');
        const last = segments[segments.length - 1].sentences.join(' ');

        const firstTokens = new Set(Utils.tokenize(first.toLowerCase()));
        const lastTokens = new Set(Utils.tokenize(last.toLowerCase()));

        // Vocabulary overlap
        let intersection = 0;
        for (const token of firstTokens) {
            if (lastTokens.has(token)) intersection++;
        }
        const vocabOverlap = intersection / Math.max(firstTokens.size, lastTokens.size);

        // Sentence structure similarity
        const firstStructures = segments[0].sentences.map(s => this.getSentenceStructure(s));
        const lastStructures = segments[segments.length - 1].sentences.map(s => this.getSentenceStructure(s));
        const structureOverlap = this.calculateArrayOverlap(firstStructures, lastStructures);

        // Combined consistency score
        const consistencyScore = vocabOverlap * 0.5 + structureOverlap * 0.5;

        return {
            consistencyScore,
            vocabOverlap,
            structureOverlap,
            interpretation: consistencyScore > 0.65 ? 'High cross-segment consistency (AI-like)' :
                           consistencyScore > 0.45 ? 'Moderate consistency' :
                           'Natural document progression'
        };
    },

    /**
     * Helper: Count consecutive repeats in a sequence
     */
    countConsecutiveRepeats(sequence) {
        let repeats = 0;
        for (let i = 1; i < sequence.length; i++) {
            if (sequence[i] === sequence[i - 1]) repeats++;
        }
        return repeats;
    },

    /**
     * Helper: Get most common item
     */
    getMostCommon(counts) {
        let maxKey = null;
        let maxVal = 0;
        for (const [key, val] of Object.entries(counts)) {
            if (val > maxVal) {
                maxVal = val;
                maxKey = key;
            }
        }
        return { key: maxKey, count: maxVal };
    },

    /**
     * Helper: Get simplified sentence structure
     */
    getSentenceStructure(sentence) {
        const words = Utils.tokenize(sentence);
        if (words.length < 3) return 'short';
        if (words.length < 8) return 'medium-simple';
        if (words.length < 15) return 'medium-complex';
        if (words.length < 25) return 'long';
        return 'very-long';
    },

    /**
     * Helper: Calculate array overlap
     */
    calculateArrayOverlap(arr1, arr2) {
        const set1 = new Set(arr1);
        const set2 = new Set(arr2);
        let intersection = 0;
        for (const item of set1) {
            if (set2.has(item)) intersection++;
        }
        return intersection / Math.max(set1.size, set2.size);
    },

    /**
     * Calculate overall AI probability
     */
    calculateOverallAIProbability(scores) {
        // Weight the different signals
        const weights = {
            openings: 0.15,
            idioms: 0.15,
            complexity: 0.2,
            errors: 0.15,
            drift: 0.2,
            crossSegment: 0.15
        };

        let weightedSum = 0;
        let totalWeight = 0;

        for (const [key, weight] of Object.entries(weights)) {
            if (scores[key] !== undefined) {
                // Invert scores where low = AI
                const score = scores[key];
                // For these, low score means high AI probability
                const aiScore = ['drift', 'errors'].includes(key) ? 1 - score : score;
                weightedSum += aiScore * weight;
                totalWeight += weight;
            }
        }

        return totalWeight > 0 ? weightedSum / totalWeight : 0.5;
    },

    /**
     * Generate findings
     */
    generateFindings(openingVariance, idiomVariance, complexityBalance, 
                     errorDistribution, stylisticDrift, crossSegmentConsistency) {
        const findings = [];

        if (openingVariance.uniformityScore > 0.6) {
            findings.push({
                text: `Sentence openings show ${openingVariance.interpretation}. Most common: "${openingVariance.mostCommon.key}" (${openingVariance.mostCommon.count} times). Low entropy: ${openingVariance.entropy.toFixed(2)}.`,
                label: 'Opening Uniformity',
                indicator: 'ai',
                severity: openingVariance.uniformityScore > 0.75 ? 'high' : 'medium',
                research: 'Based on Ippolito et al., Inconsistency as a Signal (2023)'
            });
        }

        if (idiomVariance.aiPhraseCount > 3) {
            findings.push({
                text: `Found ${idiomVariance.aiPhraseCount} AI-typical phrases (e.g., "it is important to note", "this highlights", "plays a crucial role"). These phrases appear uniformly distributed.`,
                label: 'AI Phrase Patterns',
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (complexityBalance.hasComplexityMismatch) {
            findings.push({
                text: `Complexity mismatch detected: sophisticated vocabulary (${(complexityBalance.avgLexicalSophistication * 100).toFixed(0)}%) with shallow argument depth (${(complexityBalance.avgArgumentDepth * 100).toFixed(0)}%). AI often produces "impressive-sounding" text without proportional argument complexity.`,
                label: 'Complexity Mismatch',
                indicator: 'ai',
                severity: 'high'
            });
        }

        if (errorDistribution.totalErrors === 0) {
            findings.push({
                text: `Text is suspiciously error-free. Natural human writing typically contains occasional typos, punctuation inconsistencies, or grammatical imperfections.`,
                label: 'Error-free Text',
                indicator: 'ai',
                severity: 'medium'
            });
        } else if (errorDistribution.naturalScore > 0.6) {
            findings.push({
                text: `Natural error patterns detected (${errorDistribution.totalErrors} errors with consistent distribution). This is characteristic of human writing.`,
                label: 'Human Error Patterns',
                indicator: 'human',
                severity: 'low'
            });
        }

        if (stylisticDrift.driftScore < 0.25) {
            findings.push({
                text: `${stylisticDrift.interpretation}. Writing style remains nearly constant across the document. Human writing naturally evolves as ideas develop.`,
                label: 'Style Consistency',
                indicator: 'ai',
                severity: 'high',
                research: 'Based on Krishna et al., Paraphrasing Evades Detectors (2024)'
            });
        } else if (stylisticDrift.driftScore > 0.6) {
            findings.push({
                text: `Writing style evolves naturally across the document (drift score: ${stylisticDrift.avgDrift.toFixed(2)}). This is characteristic of human writing.`,
                label: 'Natural Style Drift',
                indicator: 'human',
                severity: 'low'
            });
        }

        if (crossSegmentConsistency.consistencyScore > 0.65) {
            findings.push({
                text: `${crossSegmentConsistency.interpretation}. First and last sections of the document are unusually similar in vocabulary (${(crossSegmentConsistency.vocabOverlap * 100).toFixed(0)}%) and structure (${(crossSegmentConsistency.structureOverlap * 100).toFixed(0)}%).`,
                label: 'Cross-segment Consistency',
                indicator: 'ai',
                severity: 'medium'
            });
        }

        return findings;
    },

    /**
     * Calculate confidence
     */
    calculateConfidence(sentenceCount, segmentCount) {
        if (sentenceCount < 10) return 0.3;
        if (segmentCount < 3) return 0.4;
        if (sentenceCount < 20) return 0.6;
        if (sentenceCount < 40) return 0.75;
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
    module.exports = InconsistencyAnalyzer;
}
