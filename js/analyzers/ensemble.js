/**
 * VERITAS — Multi-Head Ensemble Architecture
 * 
 * Based on: Gehrmann et al., GLTR and Beyond
 *           OpenAI Detector Retrospective (2023)
 * 
 * KEY INSIGHT: Different signal types require specialized detection heads.
 * Combining via learned weighting outperforms simple averaging.
 * 
 * Heads:
 * 1. Syntax Variance Head - Sentence structure patterns
 * 2. Discourse Planning Head - Document-level organization
 * 3. Semantic Compression Head - Information density patterns
 * 4. Error Topology Head - Mistake patterns and distributions
 * 5. Stability Head - Perturbation response (new)
 * 6. Inconsistency Head - Intra-document variance (new)
 */

const EnsembleAnalyzer = {
    name: 'Multi-Head Ensemble',
    category: 17,
    weight: 2.0, // Highest weight - this is the meta-analyzer

    // Head configurations with learned weights
    heads: {
        syntaxVariance: {
            name: 'Syntax Variance',
            analyzers: ['SyntaxAnalyzer', 'GrammarAnalyzer'],
            weight: 0.15,
            description: 'Sentence structure and grammatical patterns'
        },
        discoursePlanning: {
            name: 'Discourse Planning',
            analyzers: ['DiscourseAnalyzer', 'MetaPatternsAnalyzer'],
            weight: 0.20,
            description: 'Document organization and rhetorical structure'
        },
        semanticCompression: {
            name: 'Semantic Compression',
            analyzers: ['LexicalAnalyzer', 'SemanticAnalyzer', 'StatisticalAnalyzer'],
            weight: 0.20,
            description: 'Information density and vocabulary patterns'
        },
        errorTopology: {
            name: 'Error Topology',
            analyzers: ['InconsistencyAnalyzer'],
            weight: 0.15,
            description: 'Mistake patterns and consistency'
        },
        stability: {
            name: 'Perturbation Stability',
            analyzers: ['StabilityAnalyzer'],
            weight: 0.15,
            description: 'Response to text perturbations'
        },
        authorship: {
            name: 'Authorship Signals',
            analyzers: ['ToneAnalyzer', 'AuthorshipAnalyzer'],
            weight: 0.15,
            description: 'Writing style and personal voice'
        }
    },

    // Deprecated/weak signals to downweight
    weakSignals: [
        'raw_perplexity',        // Now overlaps with humans
        'burstiness_alone',      // Easily gamed
        'repetition_counts',     // Not discriminative anymore
        'vocabulary_richness',   // ESL writers get false flagged
        'grammar_perfection'     // Academic writers get false flagged
    ],

    /**
     * Main analysis - combines all heads
     */
    analyze(text, categoryResults) {
        if (!text || text.length < 100) {
            return this.getEmptyResult('Text too short');
        }

        // Group analyzer results by head
        const headResults = this.groupByHead(categoryResults);
        
        // Calculate each head's score
        const headScores = {};
        const headDetails = {};
        
        for (const [headName, headConfig] of Object.entries(this.heads)) {
            const results = headResults[headName] || [];
            const headScore = this.calculateHeadScore(results, headConfig);
            headScores[headName] = headScore.score;
            headDetails[headName] = headScore;
        }
        
        // Apply learned weighting
        const weightedScore = this.applyLearnedWeights(headScores);
        
        // Calculate inter-head agreement
        const agreement = this.calculateAgreement(headScores);
        
        // Detect conflicting signals (suggests humanized AI)
        const conflicts = this.detectConflicts(headScores, headDetails);
        
        // Final probability with conflict adjustment
        let aiProbability = weightedScore.probability;
        
        // High conflict can indicate humanized AI
        if (conflicts.hasConflicts && conflicts.conflictScore > 0.3) {
            // Don't lower probability, but flag it
            conflicts.interpretation = 'Conflicting signals may indicate humanized AI or edge case';
        }
        
        // Calculate confidence based on head agreement
        const confidence = this.calculateConfidence(agreement, headScores);
        
        return {
            name: this.name,
            category: this.category,
            aiProbability,
            confidence,
            details: {
                headScores,
                headDetails,
                weightedScore,
                agreement,
                conflicts
            },
            findings: this.generateFindings(headScores, headDetails, agreement, conflicts),
            scores: headScores
        };
    },

    /**
     * Group category results by ensemble head
     */
    groupByHead(categoryResults) {
        const grouped = {};
        
        // Initialize empty arrays for each head
        for (const headName of Object.keys(this.heads)) {
            grouped[headName] = [];
        }
        
        if (!categoryResults) return grouped;
        
        // Map results to heads
        for (const result of categoryResults) {
            if (!result || result.error) continue;
            
            const analyzerName = result.name || '';
            
            // Find which head this analyzer belongs to
            for (const [headName, headConfig] of Object.entries(this.heads)) {
                const matchesHead = headConfig.analyzers.some(a => 
                    analyzerName.toLowerCase().includes(a.toLowerCase().replace('Analyzer', ''))
                );
                if (matchesHead) {
                    grouped[headName].push(result);
                    break;
                }
            }
        }
        
        return grouped;
    },

    /**
     * Calculate score for a single head
     */
    calculateHeadScore(results, headConfig) {
        if (!results || results.length === 0) {
            return {
                score: 0.5,
                confidence: 0,
                contributors: [],
                note: 'No analyzer results available'
            };
        }
        
        // Filter valid results
        const validResults = results.filter(r => 
            r.aiProbability !== undefined && 
            r.confidence !== undefined &&
            r.confidence > 0.1
        );
        
        if (validResults.length === 0) {
            return {
                score: 0.5,
                confidence: 0,
                contributors: [],
                note: 'No valid results with sufficient confidence'
            };
        }
        
        // Confidence-weighted average
        let totalWeight = 0;
        let weightedSum = 0;
        const contributors = [];
        
        for (const result of validResults) {
            const weight = result.confidence * (result.weight || 1);
            weightedSum += result.aiProbability * weight;
            totalWeight += weight;
            
            contributors.push({
                name: result.name,
                probability: result.aiProbability,
                confidence: result.confidence,
                weight: weight
            });
        }
        
        const score = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        const avgConfidence = validResults.reduce((s, r) => s + r.confidence, 0) / validResults.length;
        
        return {
            score,
            confidence: avgConfidence,
            contributors,
            resultCount: validResults.length
        };
    },

    /**
     * Apply learned weights to combine head scores
     */
    applyLearnedWeights(headScores) {
        let totalWeight = 0;
        let weightedSum = 0;
        const contributions = {};
        
        for (const [headName, score] of Object.entries(headScores)) {
            const headConfig = this.heads[headName];
            if (!headConfig) continue;
            
            const weight = headConfig.weight;
            const contribution = score * weight;
            
            weightedSum += contribution;
            totalWeight += weight;
            
            contributions[headName] = {
                rawScore: score,
                weight: weight,
                contribution: contribution
            };
        }
        
        const probability = totalWeight > 0 ? weightedSum / totalWeight : 0.5;
        
        return {
            probability,
            contributions,
            totalWeight
        };
    },

    /**
     * Calculate agreement between heads
     */
    calculateAgreement(headScores) {
        const scores = Object.values(headScores).filter(s => s !== undefined && s !== null);
        
        if (scores.length < 2) {
            return { level: 'unknown', score: 0, variance: 0 };
        }
        
        const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
        const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
        const stdDev = Math.sqrt(variance);
        
        // Agreement is inverse of standard deviation
        const agreementScore = 1 - Math.min(1, stdDev * 2);
        
        let level = 'low';
        if (agreementScore > 0.8) level = 'high';
        else if (agreementScore > 0.5) level = 'moderate';
        
        return {
            level,
            score: agreementScore,
            variance,
            stdDev,
            mean
        };
    },

    /**
     * Detect conflicting signals between heads
     */
    detectConflicts(headScores, headDetails) {
        const conflicts = [];
        const headNames = Object.keys(headScores);
        
        // Compare pairs of heads
        for (let i = 0; i < headNames.length; i++) {
            for (let j = i + 1; j < headNames.length; j++) {
                const head1 = headNames[i];
                const head2 = headNames[j];
                const score1 = headScores[head1];
                const score2 = headScores[head2];
                
                // Significant conflict if one says AI and other says human
                const diff = Math.abs(score1 - score2);
                if (diff > 0.4 && 
                    ((score1 > 0.6 && score2 < 0.4) || (score1 < 0.4 && score2 > 0.6))) {
                    conflicts.push({
                        head1,
                        head2,
                        score1,
                        score2,
                        difference: diff,
                        description: `${this.heads[head1].name} says ${score1 > 0.5 ? 'AI' : 'human'}, ` +
                                   `${this.heads[head2].name} says ${score2 > 0.5 ? 'AI' : 'human'}`
                    });
                }
            }
        }
        
        const conflictScore = conflicts.length > 0 
            ? conflicts.reduce((sum, c) => sum + c.difference, 0) / conflicts.length
            : 0;
        
        return {
            hasConflicts: conflicts.length > 0,
            conflicts,
            conflictScore,
            interpretation: conflicts.length > 0 
                ? 'Head disagreement detected - may indicate edge case or humanized AI'
                : 'Heads largely agree on classification'
        };
    },

    /**
     * Calculate overall confidence
     */
    calculateConfidence(agreement, headScores) {
        // Base confidence on agreement
        let confidence = agreement.score * 0.6;
        
        // Add confidence based on number of heads with strong signals
        const strongSignals = Object.values(headScores).filter(s => 
            s > 0.7 || s < 0.3
        ).length;
        confidence += (strongSignals / Object.keys(headScores).length) * 0.4;
        
        return Math.min(1, Math.max(0, confidence));
    },

    /**
     * Generate findings
     */
    generateFindings(headScores, headDetails, agreement, conflicts) {
        const findings = [];
        
        // Report on each head
        for (const [headName, score] of Object.entries(headScores)) {
            const headConfig = this.heads[headName];
            const details = headDetails[headName];
            
            if (score > 0.7) {
                findings.push({
                    label: headConfig.name,
                    text: `Strong AI signal from ${headConfig.description}. Score: ${(score * 100).toFixed(0)}%`,
                    indicator: 'ai',
                    severity: score > 0.85 ? 'high' : 'medium',
                    headScore: score,
                    contributors: details.contributors?.map(c => c.name).join(', ')
                });
            } else if (score < 0.3) {
                findings.push({
                    label: headConfig.name,
                    text: `Human-like signals from ${headConfig.description}. Score: ${(score * 100).toFixed(0)}%`,
                    indicator: 'human',
                    severity: 'low',
                    headScore: score
                });
            }
        }
        
        // Agreement finding
        if (agreement.level === 'high') {
            const direction = agreement.mean > 0.5 ? 'AI' : 'human';
            findings.push({
                label: 'Head Agreement',
                text: `High agreement across detection heads (${(agreement.score * 100).toFixed(0)}%). ` +
                      `All methods point toward ${direction}.`,
                indicator: direction === 'AI' ? 'ai' : 'human',
                severity: 'high'
            });
        } else if (agreement.level === 'low') {
            findings.push({
                label: 'Head Disagreement',
                text: `Low agreement between detection heads. This often indicates humanized AI or edge case writing.`,
                indicator: 'mixed',
                severity: 'medium'
            });
        }
        
        // Conflict findings
        if (conflicts.hasConflicts) {
            findings.push({
                label: 'Signal Conflicts',
                text: `${conflicts.conflicts.length} detection head conflicts detected. ` +
                      conflicts.conflicts.slice(0, 2).map(c => c.description).join('; '),
                indicator: 'mixed',
                severity: 'medium',
                suggestsHumanized: true
            });
        }
        
        return findings;
    },

    /**
     * Empty result template
     */
    getEmptyResult(note) {
        return {
            name: this.name,
            category: this.category,
            aiProbability: 0.5,
            confidence: 0,
            details: { note },
            findings: [],
            scores: {}
        };
    }
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnsembleAnalyzer;
}
