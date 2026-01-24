/**
 * VERITAS — Extended Heuristics Implementation Part 2
 * ====================================================
 * Stylometric, Cognitive Load, and Emotional Analysis implementations
 */

(function() {
    const E = typeof ExtendedHeuristicsAnalyzer !== 'undefined' ? ExtendedHeuristicsAnalyzer : 
              (typeof window !== 'undefined' ? window.ExtendedHeuristicsAnalyzer : {});

    // ========================================================================
    // STYLOMETRIC SIGNATURES IMPLEMENTATIONS
    // ========================================================================

    E.functionWordRatio = function(words) {
        const funcWords = words.filter(w => E.FUNCTION_WORDS.has(w.toLowerCase()));
        return funcWords.length / Math.max(words.length, 1);
    };

    E.functionWordDistribution = function(words) {
        const dist = {};
        words.forEach(w => {
            const lower = w.toLowerCase();
            if (E.FUNCTION_WORDS.has(lower)) {
                dist[lower] = (dist[lower] || 0) + 1;
            }
        });
        return Object.entries(dist).sort((a, b) => b[1] - a[1]).slice(0, 10);
    };

    E.articleUsage = function(words) {
        const articles = words.filter(w => ['a', 'an', 'the'].includes(w.toLowerCase()));
        return {
            count: articles.length,
            ratio: articles.length / Math.max(words.length, 1),
            theRatio: words.filter(w => w.toLowerCase() === 'the').length / Math.max(articles.length, 1),
            aAnRatio: words.filter(w => ['a', 'an'].includes(w.toLowerCase())).length / Math.max(articles.length, 1)
        };
    };

    E.prepositionPatterns = function(words) {
        const preps = ['in', 'on', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'out', 'off', 'over', 'under'];
        const found = {};
        words.forEach(w => {
            const lower = w.toLowerCase();
            if (preps.includes(lower)) {
                found[lower] = (found[lower] || 0) + 1;
            }
        });
        return {
            total: Object.values(found).reduce((a, b) => a + b, 0),
            variety: Object.keys(found).length,
            distribution: found
        };
    };

    E.conjunctionFrequency = function(words) {
        const conjs = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so', 'because', 'although', 'while', 'if', 'unless', 'until', 'since', 'when', 'where', 'whereas', 'whether'];
        const found = words.filter(w => conjs.includes(w.toLowerCase()));
        return {
            count: found.length,
            ratio: found.length / Math.max(words.length, 1)
        };
    };

    E.auxiliaryVerbUsage = function(words) {
        const aux = ['be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did'];
        const found = words.filter(w => aux.includes(w.toLowerCase()));
        return {
            count: found.length,
            ratio: found.length / Math.max(words.length, 1)
        };
    };

    E.modalVerbPatterns = function(words) {
        const modals = ['can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'];
        const dist = {};
        words.forEach(w => {
            const lower = w.toLowerCase();
            if (modals.includes(lower)) {
                dist[lower] = (dist[lower] || 0) + 1;
            }
        });
        return {
            total: Object.values(dist).reduce((a, b) => a + b, 0),
            distribution: dist
        };
    };

    E.pronounDistribution = function(words) {
        const pronouns = {
            firstSingular: ['i', 'me', 'my', 'mine', 'myself'],
            firstPlural: ['we', 'us', 'our', 'ours', 'ourselves'],
            secondPerson: ['you', 'your', 'yours', 'yourself', 'yourselves'],
            thirdSingular: ['he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself'],
            thirdPlural: ['they', 'them', 'their', 'theirs', 'themselves']
        };
        const result = {};
        for (const category in pronouns) {
            result[category] = words.filter(w => pronouns[category].includes(w.toLowerCase())).length;
        }
        return result;
    };

    E.typeTokenRatio = function(words) {
        const types = new Set(words.map(w => w.toLowerCase()));
        return types.size / Math.max(words.length, 1);
    };

    E.rootTTR = function(words) {
        const types = new Set(words.map(w => w.toLowerCase()));
        return types.size / Math.sqrt(words.length);
    };

    E.correctedTTR = function(words) {
        const types = new Set(words.map(w => w.toLowerCase()));
        return types.size / Math.sqrt(2 * words.length);
    };

    E.hapaxLegomena = function(words) {
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const hapax = Object.values(freq).filter(c => c === 1).length;
        return { count: hapax, ratio: hapax / Math.max(words.length, 1) };
    };

    E.hapaxDisLegomena = function(words) {
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const disLeg = Object.values(freq).filter(c => c === 2).length;
        return { count: disLeg, ratio: disLeg / Math.max(words.length, 1) };
    };

    E.yuleK = function(words) {
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const freqOfFreq = {};
        Object.values(freq).forEach(f => freqOfFreq[f] = (freqOfFreq[f] || 0) + 1);
        const N = words.length;
        let M1 = 0, M2 = 0;
        for (const [r, vr] of Object.entries(freqOfFreq)) {
            const rNum = parseInt(r);
            M1 += vr;
            M2 += vr * rNum * rNum;
        }
        if (M1 === 0) return 0;
        return 10000 * (M2 - M1) / (M1 * M1);
    };

    E.sichelS = function(words) {
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const V2 = Object.values(freq).filter(c => c === 2).length;
        const V = new Set(words.map(w => w.toLowerCase())).size;
        return V > 0 ? V2 / V : 0;
    };

    E.honoreR = function(words) {
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const V1 = Object.values(freq).filter(c => c === 1).length;
        const V = new Set(words.map(w => w.toLowerCase())).size;
        const N = words.length;
        if (V === V1) return 0;
        return 100 * Math.log(N) / (1 - V1/V);
    };

    E.brunetW = function(words) {
        const V = new Set(words.map(w => w.toLowerCase())).size;
        const N = words.length;
        if (N === 0 || V === 0) return 0;
        return Math.pow(N, Math.pow(V, -0.172));
    };

    E.masseA = function(words) {
        const V = new Set(words.map(w => w.toLowerCase())).size;
        const N = words.length;
        if (N === 0 || V === 0 || V === N) return 0;
        return (Math.log(N) - Math.log(V)) / Math.pow(Math.log(N), 2);
    };

    E.avgSentenceLength = function(sentences) {
        if (sentences.length === 0) return 0;
        const lengths = sentences.map(s => (s.match(/\b\w+/g) || []).length);
        return lengths.reduce((a, b) => a + b, 0) / sentences.length;
    };

    E.sentenceLengthStdDev = function(sentences) {
        if (sentences.length < 2) return 0;
        const lengths = sentences.map(s => (s.match(/\b\w+/g) || []).length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const variance = lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length;
        return Math.sqrt(variance);
    };

    E.clauseCount = function(sentences) {
        let total = 0;
        sentences.forEach(s => {
            total += (s.match(/,|;|:|and|but|or|because|although|while|when|if|unless/gi) || []).length + 1;
        });
        return total;
    };

    E.subordinateClauseRatio = function(text) {
        const subordinators = ['because', 'although', 'though', 'while', 'when', 'whenever', 'where', 'wherever', 'if', 'unless', 'until', 'since', 'after', 'before', 'as', 'that', 'which', 'who', 'whom', 'whose'];
        const matches = subordinators.reduce((count, sub) => {
            return count + (text.toLowerCase().match(new RegExp('\\b' + sub + '\\b', 'g')) || []).length;
        }, 0);
        const sentences = text.split(/[.!?]+/).length;
        return matches / Math.max(sentences, 1);
    };

    E.coordinateClauseRatio = function(text) {
        const coordinators = ['and', 'but', 'or', 'nor', 'for', 'yet', 'so'];
        const matches = coordinators.reduce((count, coord) => {
            return count + (text.toLowerCase().match(new RegExp('\\b' + coord + '\\b', 'g')) || []).length;
        }, 0);
        const sentences = text.split(/[.!?]+/).length;
        return matches / Math.max(sentences, 1);
    };

    E.sentenceTypeDistribution = function(sentences) {
        const dist = { declarative: 0, interrogative: 0, exclamatory: 0, imperative: 0 };
        sentences.forEach(s => {
            const trimmed = s.trim();
            if (trimmed.endsWith('?')) dist.interrogative++;
            else if (trimmed.endsWith('!')) dist.exclamatory++;
            else if (/^(please|let|do|don't|never|always)\b/i.test(trimmed)) dist.imperative++;
            else dist.declarative++;
        });
        return dist;
    };

    E.estimateEmbeddingDepth = function(sentences) {
        let maxDepth = 0;
        sentences.forEach(s => {
            let depth = 0;
            let currentMax = 0;
            for (const char of s) {
                if (char === '(' || char === '[' || char === '{') {
                    depth++;
                    currentMax = Math.max(currentMax, depth);
                } else if (char === ')' || char === ']' || char === '}') {
                    depth = Math.max(0, depth - 1);
                }
            }
            // Also count comma-separated clauses as a proxy
            const clauses = (s.match(/,/g) || []).length;
            currentMax = Math.max(currentMax, Math.floor(clauses / 2));
            maxDepth = Math.max(maxDepth, currentMax);
        });
        return maxDepth;
    };

    E.punctuationRate = function(text, punct) {
        const count = (text.match(new RegExp('\\' + punct, 'g')) || []).length;
        const sentences = text.split(/[.!?]+/).length;
        return count / Math.max(sentences, 1);
    };

    E.dashFrequency = function(text) {
        const dashes = (text.match(/[-–—]/g) || []).length;
        const words = (text.match(/\b\w+/g) || []).length;
        return dashes / Math.max(words, 1);
    };

    E.parenthesisFrequency = function(text) {
        const parens = (text.match(/[()]/g) || []).length / 2;
        const sentences = text.split(/[.!?]+/).length;
        return parens / Math.max(sentences, 1);
    };

    E.quotationFrequency = function(text) {
        const quotes = (text.match(/["'"']/g) || []).length / 2;
        const sentences = text.split(/[.!?]+/).length;
        return quotes / Math.max(sentences, 1);
    };

    E.ellipsisFrequency = function(text) {
        return (text.match(/\.{3}|…/g) || []).length;
    };

    E.punctuationVariety = function(text) {
        const types = new Set(text.match(/[.,;:!?'"()\-–—\[\]{}]/g) || []);
        return types.size;
    };

    E.punctuationDensity = function(text) {
        const punct = (text.match(/[.,;:!?'"()\-–—\[\]{}]/g) || []).length;
        return punct / Math.max(text.length, 1);
    };

    // ========================================================================
    // COGNITIVE LOAD IMPLEMENTATIONS
    // ========================================================================

    E.fleschReadingEase = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const syllables = words.reduce((sum, w) => sum + E.countSyllables(w), 0);
        return 206.835 - 1.015 * (words.length / sentences.length) - 84.6 * (syllables / words.length);
    };

    E.fleschKincaidGrade = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const syllables = words.reduce((sum, w) => sum + E.countSyllables(w), 0);
        return 0.39 * (words.length / sentences.length) + 11.8 * (syllables / words.length) - 15.59;
    };

    E.gunningFogIndex = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const complexWords = words.filter(w => E.countSyllables(w) >= 3).length;
        return 0.4 * ((words.length / sentences.length) + 100 * (complexWords / words.length));
    };

    E.smogIndex = function(text) {
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (sentences.length < 3) return 0;
        const words = text.match(/\b\w+/g) || [];
        const polysyllables = words.filter(w => E.countSyllables(w) >= 3).length;
        return 1.0430 * Math.sqrt(polysyllables * (30 / sentences.length)) + 3.1291;
    };

    E.colemanLiauIndex = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0) return 0;
        const letters = (text.match(/[a-zA-Z]/g) || []).length;
        const L = (letters / words.length) * 100;
        const S = (sentences.length / words.length) * 100;
        return 0.0588 * L - 0.296 * S - 15.8;
    };

    E.automatedReadabilityIndex = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const chars = (text.match(/[a-zA-Z0-9]/g) || []).length;
        return 4.71 * (chars / words.length) + 0.5 * (words.length / sentences.length) - 21.43;
    };

    E.daleChallScore = function(text) {
        // Simplified - would need Dale-Chall word list for accuracy
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const difficultWords = words.filter(w => w.length > 7).length;
        const pctDifficult = (difficultWords / words.length) * 100;
        return 0.1579 * pctDifficult + 0.0496 * (words.length / sentences.length);
    };

    E.lixReadability = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const longWords = words.filter(w => w.length > 6).length;
        return (words.length / sentences.length) + (longWords * 100 / words.length);
    };

    E.rixReadability = function(text) {
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (sentences.length === 0) return 0;
        const longWords = words.filter(w => w.length > 6).length;
        return longWords / sentences.length;
    };

    E.spacheScore = function(text) {
        // Simplified version
        const words = text.match(/\b\w+/g) || [];
        const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
        if (words.length === 0 || sentences.length === 0) return 0;
        const unfamiliar = words.filter(w => w.length > 5).length;
        return 0.141 * (words.length / sentences.length) + 0.086 * (unfamiliar / words.length * 100) + 0.839;
    };

    E.estimateParseTreeDepth = function(sentences) {
        // Estimate based on clause embedding
        let totalDepth = 0;
        sentences.forEach(s => {
            let depth = 1;
            const subClauses = (s.match(/\b(that|which|who|whom|where|when|because|although|while|if|unless)\b/gi) || []).length;
            depth += subClauses;
            totalDepth += depth;
        });
        return totalDepth / Math.max(sentences.length, 1);
    };

    E.detectGardenPath = function(sentences) {
        // Detect sentences that might cause parsing confusion
        let count = 0;
        sentences.forEach(s => {
            // Look for potentially ambiguous structures
            if (/\b(the|a)\s+\w+\s+\w+ed\s+\w+/.test(s)) count++;
            if (/\b(while|when)\s+\w+ing\b/.test(s.toLowerCase())) count++;
        });
        return count;
    };

    E.detectCenterEmbedding = function(sentences) {
        let count = 0;
        sentences.forEach(s => {
            const depth = (s.match(/\b(that|which|who)\b/gi) || []).length;
            if (depth >= 2) count++;
        });
        return count;
    };

    E.detectLongDependencies = function(sentences) {
        let count = 0;
        sentences.forEach(s => {
            const words = s.split(/\s+/);
            if (words.length > 20) {
                const subjects = (s.match(/\b(he|she|it|they|the \w+)\b/gi) || []);
                const verbs = (s.match(/\b(is|are|was|were|has|have|had|does|do|did)\b/gi) || []);
                if (subjects.length >= 2 && verbs.length >= 2) count++;
            }
        });
        return count;
    };

    E.estimateAmbiguity = function(sentences) {
        let ambiguityScore = 0;
        sentences.forEach(s => {
            // Pronouns without clear antecedents
            const pronouns = (s.match(/\b(it|this|that|these|those|they|them)\b/gi) || []).length;
            // Words with multiple common meanings
            const polysemous = (s.match(/\b(bank|run|set|get|take|make|put|turn|right|light|play|point|case)\b/gi) || []).length;
            ambiguityScore += pronouns * 0.1 + polysemous * 0.2;
        });
        return ambiguityScore / Math.max(sentences.length, 1);
    };

    E.negationComplexity = function(text) {
        const simpleNeg = (text.match(/\b(not|no|never|none|nothing|nobody|nowhere|neither)\b/gi) || []).length;
        const doubleNeg = (text.match(/\b(not\s+\w+\s+not|no\s+\w+\s+no|never\s+\w+\s+never)\b/gi) || []).length;
        const implicitNeg = (text.match(/\b(hardly|barely|scarcely|seldom|rarely)\b/gi) || []).length;
        return {
            simpleNegations: simpleNeg,
            doubleNegations: doubleNeg,
            implicitNegations: implicitNeg,
            total: simpleNeg + doubleNeg * 2 + implicitNeg
        };
    };

    E.contentWordRatio = function(words) {
        const contentWords = words.filter(w => !E.FUNCTION_WORDS.has(w.toLowerCase()));
        return contentWords.length / Math.max(words.length, 1);
    };

    E.estimatePropositionDensity = function(sentences) {
        // Estimate based on verb count per sentence
        let totalVerbs = 0;
        sentences.forEach(s => {
            const verbs = (s.match(/\b\w+(ed|ing|s|es)\b|\b(is|are|was|were|be|been|being|have|has|had)\b/gi) || []).length;
            totalVerbs += verbs;
        });
        return totalVerbs / Math.max(sentences.length, 1);
    };

    E.estimateIdeaDensity = function(text) {
        // Nouns + verbs / total words
        const words = text.match(/\b\w+/g) || [];
        const nouns = (text.match(/\b\w+(tion|ment|ness|ity|ism|er|or|ist|ance|ence)\b/gi) || []).length;
        const verbs = (text.match(/\b\w+(ed|ing|s)\b/gi) || []).length;
        return (nouns + verbs) / Math.max(words.length, 1);
    };

    E.estimateConceptDensity = function(text) {
        const words = text.match(/\b\w+/g) || [];
        // Count abstract nouns and complex terms
        const abstract = (text.match(/\b\w+(tion|ism|ity|ment|ness|ance|ence)\b/gi) || []).length;
        return abstract / Math.max(words.length, 1);
    };

    E.lexicalDensity = function(words) {
        // Lexical words / total words
        const lexical = words.filter(w => !E.FUNCTION_WORDS.has(w.toLowerCase()) && w.length > 2);
        return lexical.length / Math.max(words.length, 1);
    };

    E.informationLoadPerSentence = function(sentences) {
        const loads = sentences.map(s => {
            const words = s.match(/\b\w+/g) || [];
            const contentWords = words.filter(w => !E.FUNCTION_WORDS.has(w.toLowerCase()));
            return contentWords.length;
        });
        return loads.reduce((a, b) => a + b, 0) / Math.max(sentences.length, 1);
    };

    E.maxClauseDepth = function(sentences) {
        let max = 0;
        sentences.forEach(s => {
            const depth = (s.match(/\b(that|which|who|whom|when|where|because|although|if|unless)\b/gi) || []).length;
            max = Math.max(max, depth);
        });
        return max;
    };

    E.pronounResolutionDifficulty = function(text) {
        const pronouns = (text.match(/\b(it|this|that|these|those|they|them|he|she)\b/gi) || []).length;
        const nouns = (text.match(/\b[A-Z][a-z]+\b/g) || []).length;
        if (nouns === 0) return pronouns > 5 ? 1 : 0;
        return Math.min(1, pronouns / (nouns * 2));
    };

    E.averageReferentDistance = function(text) {
        // Simplified - average words between noun and pronoun
        const sentences = text.split(/[.!?]+/);
        let totalDistance = 0;
        let count = 0;
        sentences.forEach(s => {
            const words = s.split(/\s+/);
            let lastNounIndex = -1;
            words.forEach((w, i) => {
                if (/^[A-Z][a-z]+$/.test(w)) lastNounIndex = i;
                if (/\b(it|this|that|they|them|he|she)\b/i.test(w) && lastNounIndex >= 0) {
                    totalDistance += i - lastNounIndex;
                    count++;
                }
            });
        });
        return count > 0 ? totalDistance / count : 0;
    };

    E.topicShiftFrequency = function(text) {
        const paragraphs = text.split(/\n\s*\n/);
        let shifts = 0;
        for (let i = 1; i < paragraphs.length; i++) {
            const prevWords = new Set((paragraphs[i-1].match(/\b[A-Z][a-z]+\b/g) || []).map(w => w.toLowerCase()));
            const currWords = new Set((paragraphs[i].match(/\b[A-Z][a-z]+\b/g) || []).map(w => w.toLowerCase()));
            const overlap = [...prevWords].filter(w => currWords.has(w)).length;
            if (overlap < 2) shifts++;
        }
        return shifts / Math.max(paragraphs.length - 1, 1);
    };

    E.entityDensity = function(text) {
        const entities = (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*/g) || []).length;
        const words = (text.match(/\b\w+/g) || []).length;
        return entities / Math.max(words, 1);
    };

    // Export
    if (typeof window !== 'undefined') {
        window.ExtendedHeuristicsAnalyzer = E;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = E;
    }
})();
