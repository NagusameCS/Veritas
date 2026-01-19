/**
 * VERITAS — Metadata & Formatting Analyzer
 * Detects irregularities in Unicode, whitespace, tabs, and hidden formatting
 */

const MetadataAnalyzer = {
    name: 'Metadata & Formatting',
    category: 11,
    weight: 1.0,

    /**
     * Main analysis function
     */
    analyze(text, metadata = {}) {
        if (!text || text.length < 10) {
            return this.getEmptyResult();
        }

        const unicodeAnalysis = this.analyzeUnicode(text);
        const whitespaceAnalysis = this.analyzeWhitespace(text);
        const indentationAnalysis = this.analyzeIndentation(text);
        const hiddenCharsAnalysis = this.analyzeHiddenCharacters(text);
        const formattingConsistency = this.analyzeFormattingConsistency(text);
        const aiDecorativeMarkers = this.detectAIDecorativeMarkers(text);

        // Calculate uniformity scores
        const uniformityScores = {
            unicode: unicodeAnalysis.uniformityScore,
            whitespace: whitespaceAnalysis.uniformityScore,
            indentation: indentationAnalysis.uniformityScore,
            formatting: formattingConsistency.uniformityScore
        };

        // AI text typically has very consistent formatting
        const avgUniformity = Utils.mean(Object.values(uniformityScores));
        
        // Score based on unusual uniformity (AI) or inconsistency (human editing)
        let aiProbability = 0.5;
        
        // Very high uniformity suggests AI generation
        if (avgUniformity > 0.9) {
            aiProbability = 0.3 + avgUniformity * 0.4;
        }
        // Some inconsistency is human-like
        else if (avgUniformity > 0.6) {
            aiProbability = 0.4;
        }
        // More inconsistency suggests human
        else {
            aiProbability = 0.3;
        }

        // Suspicious patterns increase AI probability
        if (unicodeAnalysis.suspiciousPatterns.length > 0) {
            aiProbability += 0.15;
        }
        if (hiddenCharsAnalysis.count > 0) {
            aiProbability += 0.1;
        }
        
        // Strong AI decorative markers are a major indicator
        if (aiDecorativeMarkers.found) {
            aiProbability += aiDecorativeMarkers.weight;
        }

        const confidence = this.calculateConfidence(text.length, unicodeAnalysis, whitespaceAnalysis);
        const findings = this.generateFindings(unicodeAnalysis, whitespaceAnalysis, indentationAnalysis, hiddenCharsAnalysis, aiDecorativeMarkers);

        return {
            name: this.name,
            category: this.category,
            aiProbability: Math.max(0, Math.min(1, aiProbability)),
            confidence,
            details: {
                unicode: unicodeAnalysis,
                whitespace: whitespaceAnalysis,
                indentation: indentationAnalysis,
                hiddenChars: hiddenCharsAnalysis,
                formatting: formattingConsistency,
                aiMarkers: aiDecorativeMarkers,
                uniformityScores
            },
            findings,
            scores: uniformityScores
        };
    },
    
    /**
     * Detect AI-specific decorative markers (section dividers, horizontal rules)
     */
    detectAIDecorativeMarkers(text) {
        const markers = {
            found: false,
            weight: 0,
            patterns: []
        };
        
        // Three-em dash (⸻) - Very strong AI indicator
        const threeEmDash = /\u2E3B/g;
        const threeEmMatches = text.match(threeEmDash);
        if (threeEmMatches) {
            markers.found = true;
            markers.weight += 0.35;
            markers.patterns.push({
                type: 'three-em-dash',
                count: threeEmMatches.length,
                description: 'Three-em dash section divider (⸻) - Strong AI indicator'
            });
        }
        
        // Multiple em-dashes used as dividers (——— or ———)
        const multiEmDash = /[—\u2014]{3,}/g;
        const multiEmMatches = text.match(multiEmDash);
        if (multiEmMatches) {
            markers.found = true;
            markers.weight += 0.25;
            markers.patterns.push({
                type: 'multi-em-dash',
                count: multiEmMatches.length,
                description: 'Multiple em-dashes as section divider'
            });
        }
        
        // Decorative asterisk dividers (* * * or ***)
        const asteriskDivider = /^\s*[*\u2022\u2023\u25CF]{3,}\s*$|^\s*[*]\s+[*]\s+[*]\s*$/gm;
        const asteriskMatches = text.match(asteriskDivider);
        if (asteriskMatches) {
            markers.found = true;
            markers.weight += 0.15;
            markers.patterns.push({
                type: 'asterisk-divider',
                count: asteriskMatches.length,
                description: 'Decorative asterisk dividers'
            });
        }
        
        // Horizontal line characters
        const horizLine = /[\u2500-\u257F\u2580-\u259F]{3,}/g;
        const horizMatches = text.match(horizLine);
        if (horizMatches) {
            markers.found = true;
            markers.weight += 0.30;
            markers.patterns.push({
                type: 'box-drawing',
                count: horizMatches.length,
                description: 'Box drawing horizontal lines - AI formatting'
            });
        }
        
        // Emoji section headers 📌 🔹 ✨ etc at start of lines
        const emojiHeader = /^[\s]*[\u{1F300}-\u{1F9FF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]+\s+[A-Z]/gmu;
        const emojiMatches = text.match(emojiHeader);
        if (emojiMatches && emojiMatches.length >= 2) {
            markers.found = true;
            markers.weight += 0.20;
            markers.patterns.push({
                type: 'emoji-headers',
                count: emojiMatches.length,
                description: 'Emoji section headers - common in AI responses'
            });
        }
        
        // Cap the weight
        markers.weight = Math.min(0.4, markers.weight);
        
        return markers;
    },

    /**
     * Analyze Unicode character usage
     */
    analyzeUnicode(text) {
        const analysis = {
            totalChars: text.length,
            asciiCount: 0,
            extendedAsciiCount: 0,
            unicodeCount: 0,
            suspiciousPatterns: [],
            characterCategories: {},
            uniformityScore: 0.5
        };

        // Character category counts
        const categories = {
            basic_latin: 0,        // U+0000 - U+007F
            latin_extended: 0,     // U+0080 - U+024F
            punctuation: 0,        // Various punctuation blocks
            symbols: 0,            // Mathematical, currency, etc.
            emoji: 0,              // Emoji range
            other: 0
        };

        // Suspicious Unicode patterns (often from copy-paste or AI)
        const suspiciousChars = {
            '\u00A0': 'Non-breaking space',           // NBSP
            '\u2000': 'En quad',
            '\u2001': 'Em quad',
            '\u2002': 'En space',
            '\u2003': 'Em space',
            '\u2004': 'Three-per-em space',
            '\u2005': 'Four-per-em space',
            '\u2006': 'Six-per-em space',
            '\u2007': 'Figure space',
            '\u2008': 'Punctuation space',
            '\u2009': 'Thin space',
            '\u200A': 'Hair space',
            '\u200B': 'Zero-width space',
            '\u200C': 'Zero-width non-joiner',
            '\u200D': 'Zero-width joiner',
            '\u2028': 'Line separator',
            '\u2029': 'Paragraph separator',
            '\u202F': 'Narrow no-break space',
            '\u205F': 'Medium mathematical space',
            '\u3000': 'Ideographic space',
            '\uFEFF': 'BOM/Zero-width no-break space',
            '\u2018': 'Left single quote (curly)',
            '\u2019': 'Right single quote (curly)',
            '\u201C': 'Left double quote (curly)',
            '\u201D': 'Right double quote (curly)',
            '\u2013': 'En dash',
            '\u2014': 'Em dash',
            '\u2026': 'Horizontal ellipsis',
            '\u00B7': 'Middle dot',
            // Strong AI indicators - decorative/formatting characters
            '\u2E3B': 'Three-em dash (AI section divider)',
            '\u2E3A': 'Two-em dash',
            '\u2015': 'Horizontal bar',
            '\u2500': 'Box drawing horizontal',
            '\u2501': 'Box drawing heavy horizontal',
            '\u2504': 'Box drawing light triple dash',
            '\u2550': 'Box drawing double horizontal',
            '\u2594': 'Upper one eighth block',
            '\u23AF': 'Horizontal line extension'
        };

        const foundSuspicious = {};

        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            const code = text.charCodeAt(i);

            if (code <= 0x007F) {
                analysis.asciiCount++;
                categories.basic_latin++;
            } else if (code <= 0x00FF) {
                analysis.extendedAsciiCount++;
                categories.latin_extended++;
            } else {
                analysis.unicodeCount++;
                
                if (code >= 0x1F600 && code <= 0x1F64F) {
                    categories.emoji++;
                } else if (code >= 0x2000 && code <= 0x206F) {
                    categories.punctuation++;
                } else if (code >= 0x2100 && code <= 0x27FF) {
                    categories.symbols++;
                } else {
                    categories.other++;
                }
            }

            // Check for suspicious characters
            if (suspiciousChars[char]) {
                if (!foundSuspicious[char]) {
                    foundSuspicious[char] = { name: suspiciousChars[char], count: 0, positions: [] };
                }
                foundSuspicious[char].count++;
                foundSuspicious[char].positions.push(i);
            }
        }

        // Convert to array
        for (const char in foundSuspicious) {
            analysis.suspiciousPatterns.push({
                character: char,
                codePoint: 'U+' + char.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0'),
                ...foundSuspicious[char]
            });
        }

        analysis.characterCategories = categories;

        // Calculate uniformity (AI tends to use consistent character sets)
        const categoryValues = Object.values(categories).filter(v => v > 0);
        if (categoryValues.length > 1) {
            analysis.uniformityScore = 1 - VarianceUtils.coefficientOfVariation(categoryValues);
        } else {
            analysis.uniformityScore = 0.9; // Single category = very uniform
        }

        return analysis;
    },

    /**
     * Analyze whitespace patterns
     */
    analyzeWhitespace(text) {
        const analysis = {
            spaces: 0,
            tabs: 0,
            newlines: 0,
            carriageReturns: 0,
            mixedLineEndings: false,
            trailingWhitespace: 0,
            multipleSpaces: 0,
            uniformityScore: 0.5
        };

        const lines = text.split('\n');
        const lineEndingTypes = new Set();

        // Count whitespace
        for (let i = 0; i < text.length; i++) {
            const char = text[i];
            if (char === ' ') analysis.spaces++;
            else if (char === '\t') analysis.tabs++;
            else if (char === '\n') analysis.newlines++;
            else if (char === '\r') analysis.carriageReturns++;
        }

        // Check for mixed line endings
        if (text.includes('\r\n')) lineEndingTypes.add('CRLF');
        if (text.match(/[^\r]\n/)) lineEndingTypes.add('LF');
        if (text.match(/\r[^\n]/)) lineEndingTypes.add('CR');
        analysis.mixedLineEndings = lineEndingTypes.size > 1;

        // Check for trailing whitespace
        for (const line of lines) {
            if (line.match(/[ \t]+$/)) {
                analysis.trailingWhitespace++;
            }
        }

        // Check for multiple consecutive spaces
        const multiSpaceMatches = text.match(/  +/g);
        analysis.multipleSpaces = multiSpaceMatches ? multiSpaceMatches.length : 0;

        // Calculate uniformity
        // AI text typically has very consistent whitespace usage
        const spacesPerLine = lines.map(l => (l.match(/ /g) || []).length);
        if (spacesPerLine.length > 1) {
            const cv = VarianceUtils.coefficientOfVariation(spacesPerLine);
            analysis.uniformityScore = 1 - Math.min(1, cv);
        }

        return analysis;
    },

    /**
     * Analyze indentation patterns
     */
    analyzeIndentation(text) {
        const lines = text.split('\n').filter(l => l.trim().length > 0);
        const analysis = {
            usesSpaces: false,
            usesTabs: false,
            mixedIndentation: false,
            indentLevels: [],
            indentSizes: [],
            uniformityScore: 0.5
        };

        for (const line of lines) {
            const leadingWhitespace = line.match(/^(\s*)/)[1];
            
            if (leadingWhitespace.includes(' ')) analysis.usesSpaces = true;
            if (leadingWhitespace.includes('\t')) analysis.usesTabs = true;
            
            // Calculate indent level (normalize tabs to 4 spaces)
            const indentLevel = leadingWhitespace.replace(/\t/g, '    ').length;
            analysis.indentLevels.push(indentLevel);
            
            if (indentLevel > 0) {
                analysis.indentSizes.push(indentLevel);
            }
        }

        analysis.mixedIndentation = analysis.usesSpaces && analysis.usesTabs;

        // Calculate uniformity
        if (analysis.indentSizes.length > 1) {
            // Check if indent sizes follow a pattern (e.g., multiples of 2 or 4)
            const gcd = this.findGCD(analysis.indentSizes);
            const normalized = analysis.indentSizes.map(s => s / gcd);
            const cv = VarianceUtils.coefficientOfVariation(normalized);
            analysis.uniformityScore = 1 - Math.min(1, cv);
        } else if (analysis.indentLevels.every(l => l === 0)) {
            analysis.uniformityScore = 1; // No indentation = uniform
        }

        return analysis;
    },

    /**
     * Find GCD of array of numbers
     */
    findGCD(numbers) {
        const gcd = (a, b) => b === 0 ? a : gcd(b, a % b);
        return numbers.reduce((acc, n) => gcd(acc, n), numbers[0]);
    },

    /**
     * Analyze hidden/invisible characters
     */
    analyzeHiddenCharacters(text) {
        const hiddenPatterns = [
            { pattern: /\u200B/g, name: 'Zero-width space' },
            { pattern: /\u200C/g, name: 'Zero-width non-joiner' },
            { pattern: /\u200D/g, name: 'Zero-width joiner' },
            { pattern: /\uFEFF/g, name: 'Byte order mark' },
            { pattern: /\u00AD/g, name: 'Soft hyphen' },
            { pattern: /[\u2060-\u206F]/g, name: 'Invisible formatters' },
            { pattern: /[\uE000-\uF8FF]/g, name: 'Private use area' },
            { pattern: /[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, name: 'Control characters' }
        ];

        const found = [];
        let totalCount = 0;

        for (const { pattern, name } of hiddenPatterns) {
            const matches = text.match(pattern);
            if (matches) {
                found.push({ name, count: matches.length });
                totalCount += matches.length;
            }
        }

        return {
            count: totalCount,
            types: found,
            hasHiddenChars: totalCount > 0
        };
    },

    /**
     * Analyze overall formatting consistency
     */
    analyzeFormattingConsistency(text) {
        const paragraphs = text.split(/\n\s*\n/).filter(p => p.trim().length > 0);
        
        const analysis = {
            paragraphCount: paragraphs.length,
            paragraphLengths: paragraphs.map(p => p.length),
            sentenceSpacing: this.analyzeSentenceSpacing(text),
            punctuationStyle: this.analyzePunctuationStyle(text),
            uniformityScore: 0.5
        };

        // Calculate uniformity from paragraph lengths
        if (analysis.paragraphLengths.length > 1) {
            const cv = VarianceUtils.coefficientOfVariation(analysis.paragraphLengths);
            const uniformity = 1 - Math.min(1, cv);
            
            // Combine with sentence spacing uniformity
            analysis.uniformityScore = (uniformity + analysis.sentenceSpacing.uniformity) / 2;
        }

        return analysis;
    },

    /**
     * Analyze spacing after sentences
     */
    analyzeSentenceSpacing(text) {
        const oneSpace = (text.match(/[.!?] [A-Z]/g) || []).length;
        const twoSpaces = (text.match(/[.!?]  [A-Z]/g) || []).length;
        
        const total = oneSpace + twoSpaces;
        
        return {
            oneSpaceCount: oneSpace,
            twoSpaceCount: twoSpaces,
            consistent: total > 0 && (oneSpace === 0 || twoSpaces === 0),
            uniformity: total > 0 ? Math.max(oneSpace, twoSpaces) / total : 1
        };
    },

    /**
     * Analyze punctuation style consistency
     */
    analyzePunctuationStyle(text) {
        const straightQuotes = (text.match(/["']/g) || []).length;
        const curlyQuotes = (text.match(/[""'']/g) || []).length;
        
        const straightDash = (text.match(/--/g) || []).length;
        const emDash = (text.match(/—/g) || []).length;
        const enDash = (text.match(/–/g) || []).length;
        
        const threeDots = (text.match(/\.\.\./g) || []).length;
        const ellipsis = (text.match(/…/g) || []).length;

        return {
            quotes: { straight: straightQuotes, curly: curlyQuotes },
            dashes: { straight: straightDash, em: emDash, en: enDash },
            ellipsis: { dots: threeDots, symbol: ellipsis },
            consistent: this.isConsistentStyle(straightQuotes, curlyQuotes) &&
                       this.isConsistentStyle(threeDots, ellipsis)
        };
    },

    /**
     * Check if usage is consistent (one style dominates)
     */
    isConsistentStyle(count1, count2) {
        const total = count1 + count2;
        if (total === 0) return true;
        return count1 === 0 || count2 === 0 || Math.max(count1, count2) / total > 0.9;
    },

    /**
     * Calculate confidence based on text characteristics
     */
    calculateConfidence(textLength, unicodeAnalysis, whitespaceAnalysis) {
        let confidence = 0.5;
        
        if (textLength > 1000) confidence += 0.2;
        else if (textLength > 500) confidence += 0.1;
        
        if (unicodeAnalysis.suspiciousPatterns.length > 0) confidence += 0.15;
        if (whitespaceAnalysis.mixedLineEndings) confidence += 0.1;
        
        return Math.min(1, confidence);
    },

    /**
     * Generate human-readable findings with detailed statistics
     */
    generateFindings(unicode, whitespace, indentation, hidden, aiMarkers = null) {
        const findings = [];

        // AI Decorative Markers findings (highest priority)
        if (aiMarkers && aiMarkers.found) {
            for (const pattern of aiMarkers.patterns) {
                findings.push({
                    label: 'AI Decorative Pattern',
                    value: pattern.description,
                    note: 'Decorative Unicode patterns are a very strong AI generation signal',
                    indicator: 'ai',
                    severity: 'critical',
                    stats: {
                        occurrences: `${pattern.count}× found`,
                        pattern: pattern.example || pattern.description,
                        confidence: 'Very High (>95%)'
                    },
                    benchmark: {
                        humanRange: 'Rarely or never used',
                        aiRange: 'Common in ChatGPT, Claude outputs',
                        interpretation: 'These decorative elements are signature AI formatting'
                    }
                });
            }
        }

        // Unicode findings
        for (const pattern of unicode.suspiciousPatterns) {
            findings.push({
                label: 'Unusual Unicode Character',
                value: `Found ${pattern.name} characters`,
                note: 'Unusual characters may indicate copy-paste or AI generation',
                indicator: 'ai',
                severity: pattern.count > 5 ? 'high' : 'medium',
                stats: {
                    occurrences: `${pattern.count}× found`,
                    codePoint: pattern.codePoint,
                    characterName: pattern.name,
                    locations: pattern.positions ? `At positions: ${pattern.positions.slice(0, 5).join(', ')}${pattern.positions.length > 5 ? '...' : ''}` : 'N/A'
                },
                benchmark: {
                    humanRange: '0–1 occurrences typical',
                    aiRange: '2+ occurrences common',
                    note: 'AI often uses special dashes, quotes, and Unicode spaces'
                }
            });
        }

        // Whitespace findings
        if (whitespace.mixedLineEndings) {
            findings.push({
                label: 'Mixed Line Endings',
                value: 'Both CRLF and LF detected',
                note: 'Suggests text from multiple sources or platforms',
                indicator: 'mixed',
                severity: 'low',
                stats: {
                    crlfCount: whitespace.crlfCount || 'some',
                    lfCount: whitespace.lfCount || 'some',
                    interpretation: 'Windows uses CRLF, Unix/Mac uses LF'
                },
                benchmark: {
                    note: 'Single-source text typically has consistent line endings'
                }
            });
        }

        if (whitespace.trailingWhitespace > 5) {
            findings.push({
                label: 'Trailing Whitespace',
                value: `${whitespace.trailingWhitespace} lines with trailing spaces`,
                note: 'Unusual in carefully edited text',
                indicator: 'neutral',
                severity: 'low',
                stats: {
                    linesAffected: whitespace.trailingWhitespace,
                    percentage: whitespace.totalLines ? `${((whitespace.trailingWhitespace / whitespace.totalLines) * 100).toFixed(1)}% of lines` : 'N/A'
                }
            });
        }

        // Indentation findings
        if (indentation.mixedIndentation) {
            findings.push({
                label: 'Mixed Indentation',
                value: 'Both tabs and spaces used for indentation',
                note: 'Suggests multiple authors or copy-paste from different sources',
                indicator: 'human',
                severity: 'medium',
                stats: {
                    tabLines: indentation.tabLines || 'some',
                    spaceLines: indentation.spaceLines || 'some',
                    interpretation: 'Human editing often introduces inconsistencies'
                },
                benchmark: {
                    humanRange: 'Often inconsistent',
                    aiRange: 'Usually consistent (all spaces or all tabs)'
                }
            });
        }

        // Hidden character findings
        if (hidden.hasHiddenChars) {
            findings.push({
                label: 'Hidden Characters',
                value: `${hidden.count} invisible characters detected`,
                note: 'May indicate copy-paste from web or documents',
                indicator: 'ai',
                severity: 'high',
                stats: {
                    totalHidden: hidden.count,
                    types: hidden.types ? hidden.types.join(', ') : 'various',
                    examples: hidden.examples ? hidden.examples.slice(0, 3).join(', ') : 'N/A'
                },
                benchmark: {
                    humanRange: '0–2 hidden characters',
                    aiRange: '3+ hidden characters common',
                    note: 'Zero-width spaces and joiners often come from web sources'
                }
            });
        }

        // Overall paragraph uniformity (key metric)
        if (aiMarkers && typeof aiMarkers.paragraphUniformity === 'number') {
            const uniformity = aiMarkers.paragraphUniformity;
            if (uniformity > 0.7) {
                findings.push({
                    label: 'Paragraph Uniformity',
                    value: 'Highly uniform paragraph lengths',
                    note: 'AI tends to produce paragraphs of similar length',
                    indicator: 'ai',
                    severity: uniformity > 0.85 ? 'high' : 'medium',
                    stats: {
                        uniformityScore: `${(uniformity * 100).toFixed(1)}%`,
                        interpretation: 'Higher = more uniform = more AI-like'
                    },
                    benchmark: {
                        humanRange: '30%–60% uniformity',
                        aiRange: '70%–95% uniformity',
                        note: 'This is the #1 predictor in our Helios model (39% importance)'
                    }
                });
            }
        }

        return findings;
    },

    /**
     * Empty result for short text
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
    module.exports = MetadataAnalyzer;
}
