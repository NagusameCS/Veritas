/**
 * VERITAS — Extended Heuristics Implementation Part 3
 * ====================================================
 * Emotional, Structural, Error Analysis, and Domain implementations
 */

(function() {
    const E = typeof ExtendedHeuristicsAnalyzer !== 'undefined' ? ExtendedHeuristicsAnalyzer : 
              (typeof window !== 'undefined' ? window.ExtendedHeuristicsAnalyzer : {});

    // ========================================================================
    // EMOTIONAL/AFFECTIVE IMPLEMENTATIONS
    // ========================================================================

    E.estimateSentimentPolarity = function(text) {
        const words = text.toLowerCase().match(/\b[a-z]+/g) || [];
        let positive = 0, negative = 0;
        words.forEach(w => {
            if (E.POSITIVE_WORDS.has(w)) positive++;
            if (E.NEGATIVE_WORDS.has(w)) negative++;
        });
        const total = positive + negative;
        if (total === 0) return 0;
        return (positive - negative) / total; // -1 to 1
    };

    E.sentimentVariance = function(sentences) {
        const sentiments = sentences.map(s => {
            const words = s.toLowerCase().match(/\b[a-z]+/g) || [];
            let score = 0;
            words.forEach(w => {
                if (E.POSITIVE_WORDS.has(w)) score++;
                if (E.NEGATIVE_WORDS.has(w)) score--;
            });
            return score / Math.max(words.length, 1);
        });
        if (sentiments.length < 2) return 0;
        const mean = sentiments.reduce((a, b) => a + b, 0) / sentiments.length;
        return Math.sqrt(sentiments.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / sentiments.length);
    };

    E.sentimentTrajectory = function(sentences) {
        const scores = sentences.map(s => {
            const words = s.toLowerCase().match(/\b[a-z]+/g) || [];
            let score = 0;
            words.forEach(w => {
                if (E.POSITIVE_WORDS.has(w)) score++;
                if (E.NEGATIVE_WORDS.has(w)) score--;
            });
            return score;
        });
        if (scores.length < 2) return 'stable';
        const first = scores.slice(0, Math.floor(scores.length / 2));
        const second = scores.slice(Math.floor(scores.length / 2));
        const firstAvg = first.reduce((a, b) => a + b, 0) / first.length;
        const secondAvg = second.reduce((a, b) => a + b, 0) / second.length;
        if (secondAvg > firstAvg + 0.5) return 'improving';
        if (secondAvg < firstAvg - 0.5) return 'declining';
        return 'stable';
    };

    E.countPositiveWords = function(words) {
        return words.filter(w => E.POSITIVE_WORDS.has(w.toLowerCase())).length;
    };

    E.countNegativeWords = function(words) {
        return words.filter(w => E.NEGATIVE_WORDS.has(w.toLowerCase())).length;
    };

    E.neutralWordRatio = function(words) {
        const emotional = words.filter(w => E.POSITIVE_WORDS.has(w.toLowerCase()) || E.NEGATIVE_WORDS.has(w.toLowerCase())).length;
        return 1 - (emotional / Math.max(words.length, 1));
    };

    E.sentimentIntensity = function(text) {
        const intensified = (text.match(/\b(very|really|extremely|incredibly|absolutely|totally|completely)\s+\w+/gi) || []).length;
        const sentences = text.split(/[.!?]+/).length;
        return intensified / Math.max(sentences, 1);
    };

    E.countEmotionWords = function(text, emotion) {
        const words = E.EMOTION_LEXICON[emotion] || [];
        let count = 0;
        const textLower = text.toLowerCase();
        words.forEach(w => {
            const regex = new RegExp('\\b' + w + '\\b', 'g');
            const matches = textLower.match(regex);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.emotionalDiversity = function(text) {
        let emotionsFound = 0;
        for (const emotion in E.EMOTION_LEXICON) {
            if (E.countEmotionWords(text, emotion) > 0) emotionsFound++;
        }
        return emotionsFound / Object.keys(E.EMOTION_LEXICON).length;
    };

    E.countIntensifiers = function(text) {
        let count = 0;
        E.INTENSIFIERS.forEach(i => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + i + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countHedges = function(text) {
        let count = 0;
        E.HEDGES.forEach(h => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + h.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countBoosters = function(text) {
        const boosters = ['definitely', 'certainly', 'obviously', 'clearly', 'undoubtedly', 'surely', 'absolutely', 'always', 'never', 'completely', 'totally', 'entirely'];
        let count = 0;
        boosters.forEach(b => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + b + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countDowntoners = function(text) {
        const downtoners = ['slightly', 'somewhat', 'rather', 'fairly', 'a bit', 'a little', 'sort of', 'kind of', 'mildly', 'partly', 'marginally'];
        let count = 0;
        downtoners.forEach(d => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + d.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.expressivePunctuation = function(text) {
        const exclamations = (text.match(/!+/g) || []).length;
        const questions = (text.match(/\?+/g) || []).length;
        const multiPunct = (text.match(/[!?]{2,}/g) || []).length;
        const ellipses = (text.match(/\.{3,}|…/g) || []).length;
        return { exclamations, questions, multiPunct, ellipses, total: exclamations + questions + multiPunct + ellipses };
    };

    E.emphasisMarkers = function(text) {
        const caps = (text.match(/\b[A-Z]{2,}\b/g) || []).length;
        const asterisks = (text.match(/\*\w+\*/g) || []).length;
        const underscores = (text.match(/_\w+_/g) || []).length;
        return { allCaps: caps, asterisks, underscores, total: caps + asterisks + underscores };
    };

    E.exclamationPatterns = function(text) {
        const single = (text.match(/[^!]![^!]/g) || []).length;
        const double = (text.match(/!!/g) || []).length;
        const triple = (text.match(/!!!/g) || []).length;
        return { single, double, triple };
    };

    E.countOpinionWords = function(text) {
        const opinionMarkers = ['think', 'believe', 'feel', 'opinion', 'view', 'perspective', 'consider', 'argue', 'suggest', 'propose', 'claim', 'assert', 'maintain', 'contend'];
        let count = 0;
        opinionMarkers.forEach(w => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + w + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countSubjectiveAdjectives = function(text) {
        const subjective = ['good', 'bad', 'best', 'worst', 'beautiful', 'ugly', 'nice', 'terrible', 'wonderful', 'awful', 'great', 'horrible', 'amazing', 'disgusting', 'perfect', 'imperfect'];
        let count = 0;
        subjective.forEach(w => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + w + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countEvaluativeLanguage = function(text) {
        const evaluative = ['should', 'must', 'need to', 'ought to', 'important', 'necessary', 'essential', 'crucial', 'vital', 'wrong', 'right', 'correct', 'incorrect'];
        let count = 0;
        evaluative.forEach(e => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + e.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.personalOpinionMarkers = function(text) {
        const markers = ['i think', 'i believe', 'in my opinion', 'from my perspective', 'personally', 'i feel', 'to me', 'as i see it', 'i would say'];
        let count = 0;
        markers.forEach(m => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + m.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.objectivityScore = function(text) {
        const subjective = E.countOpinionWords(text) + E.countSubjectiveAdjectives(text);
        const words = (text.match(/\b\w+/g) || []).length;
        return 1 - (subjective / Math.max(words, 1));
    };

    // ========================================================================
    // STRUCTURAL PATTERN IMPLEMENTATIONS
    // ========================================================================

    E.avgParagraphLength = function(paragraphs) {
        if (paragraphs.length === 0) return 0;
        const lengths = paragraphs.map(p => (p.match(/\b\w+/g) || []).length);
        return lengths.reduce((a, b) => a + b, 0) / paragraphs.length;
    };

    E.paragraphLengthVariance = function(paragraphs) {
        if (paragraphs.length < 2) return 0;
        const lengths = paragraphs.map(p => (p.match(/\b\w+/g) || []).length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        return Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length);
    };

    E.detectIntroduction = function(paragraphs) {
        if (paragraphs.length === 0) return false;
        const first = paragraphs[0].toLowerCase();
        return /\b(introduction|overview|this (paper|essay|article)|in this|we will|i will)\b/.test(first);
    };

    E.detectConclusion = function(paragraphs) {
        if (paragraphs.length === 0) return false;
        const last = paragraphs[paragraphs.length - 1].toLowerCase();
        return /\b(conclusion|in summary|to summarize|in conclusion|finally|overall|to conclude|in closing)\b/.test(last);
    };

    E.detectTransitions = function(paragraphs) {
        const transitionWords = ['however', 'moreover', 'furthermore', 'additionally', 'nevertheless', 'consequently', 'therefore', 'meanwhile', 'similarly', 'conversely'];
        let count = 0;
        paragraphs.forEach(p => {
            const firstSentence = p.split(/[.!?]/)[0].toLowerCase();
            transitionWords.forEach(t => {
                if (firstSentence.includes(t)) count++;
            });
        });
        return count > 0;
    };

    E.structuralBalance = function(paragraphs) {
        if (paragraphs.length < 2) return 1;
        const lengths = paragraphs.map(p => (p.match(/\b\w+/g) || []).length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const cv = Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length) / mean;
        return 1 - Math.min(cv, 1);
    };

    E.sentenceTypeRatio = function(sentences, type) {
        const dist = E.sentenceTypeDistribution(sentences);
        const total = Object.values(dist).reduce((a, b) => a + b, 0);
        return total > 0 ? dist[type] / total : 0;
    };

    E.sentenceOpeningPatterns = function(sentences) {
        const patterns = {};
        sentences.forEach(s => {
            const firstWord = (s.match(/^\s*(\w+)/) || [, 'unknown'])[1].toLowerCase();
            patterns[firstWord] = (patterns[firstWord] || 0) + 1;
        });
        return Object.entries(patterns).sort((a, b) => b[1] - a[1]).slice(0, 10);
    };

    E.sentenceClosingPatterns = function(sentences) {
        const patterns = {};
        sentences.forEach(s => {
            const lastWord = (s.match(/(\w+)[.!?]*\s*$/) || [, 'unknown'])[1].toLowerCase();
            patterns[lastWord] = (patterns[lastWord] || 0) + 1;
        });
        return Object.entries(patterns).sort((a, b) => b[1] - a[1]).slice(0, 10);
    };

    E.passiveVoiceRatio = function(sentences) {
        let passive = 0;
        sentences.forEach(s => {
            if (/\b(was|were|is|are|been|being)\s+\w+ed\b/i.test(s)) passive++;
            if (/\b(was|were|is|are|been|being)\s+\w+en\b/i.test(s)) passive++;
        });
        return passive / Math.max(sentences.length, 1);
    };

    E.numberedListCount = function(text) {
        return (text.match(/^\s*\d+[.)]\s+/gm) || []).length;
    };

    E.bulletListCount = function(text) {
        return (text.match(/^\s*[-*•]\s+/gm) || []).length;
    };

    E.inlineEnumerationCount = function(text) {
        return (text.match(/\b(first|second|third|fourth|finally|lastly|firstly|secondly|thirdly)\b/gi) || []).length;
    };

    E.parallelStructureScore = function(text) {
        const sentences = text.split(/[.!?]+/);
        let parallel = 0;
        for (let i = 0; i < sentences.length - 1; i++) {
            const curr = sentences[i].trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();
            const next = sentences[i + 1].trim().split(/\s+/).slice(0, 3).join(' ').toLowerCase();
            if (curr === next && curr.length > 5) parallel++;
        }
        return parallel / Math.max(sentences.length - 1, 1);
    };

    E.listItemConsistency = function(text) {
        const items = text.match(/^\s*[-*•\d.]+\s+(.+)/gm) || [];
        if (items.length < 2) return 1;
        const lengths = items.map(i => i.length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const cv = Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length) / mean;
        return 1 - Math.min(cv, 1);
    };

    E.capitalizationPatterns = function(text) {
        return {
            allCaps: (text.match(/\b[A-Z]{2,}\b/g) || []).length,
            titleCase: (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b/g) || []).length,
            sentenceCase: (text.match(/^[A-Z][a-z]/gm) || []).length,
            lowercaseStart: (text.match(/^[a-z]/gm) || []).length
        };
    };

    E.whitespacePatterns = function(text) {
        return {
            doubleSpaces: (text.match(/  +/g) || []).length,
            tabCount: (text.match(/\t/g) || []).length,
            trailingSpaces: (text.match(/ +$/gm) || []).length,
            leadingSpaces: (text.match(/^ +/gm) || []).length
        };
    };

    E.lineBreakPatterns = function(text) {
        return {
            singleBreaks: (text.match(/[^\n]\n[^\n]/g) || []).length,
            doubleBreaks: (text.match(/\n\n/g) || []).length,
            tripleBreaks: (text.match(/\n\n\n/g) || []).length
        };
    };

    E.indentationConsistency = function(text) {
        const lines = text.split('\n');
        const indents = lines.map(l => (l.match(/^(\s*)/) || ['', ''])[1].length);
        const nonZero = indents.filter(i => i > 0);
        if (nonZero.length === 0) return 1;
        const uniqueIndents = new Set(nonZero);
        return 1 / uniqueIndents.size; // Fewer unique = more consistent
    };

    E.detectHeadings = function(text) {
        const patterns = [
            /^#+\s+.+$/gm,  // Markdown
            /^[A-Z][^.!?]*:$/gm,  // Colon headers
            /^[A-Z\s]+$/gm,  // All caps
            /^\d+\.\s+[A-Z]/gm  // Numbered
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    // ========================================================================
    // ERROR & IMPERFECTION IMPLEMENTATIONS
    // ========================================================================

    E.detectMisspellings = function(words) {
        // Heuristic detection - words with unusual patterns
        const suspicious = words.filter(w => {
            if (w.length < 4) return false;
            // Triple letters
            if (/(.)\1\1/.test(w)) return true;
            // No vowels in long word
            if (w.length > 4 && !/[aeiou]/i.test(w)) return true;
            // Unusual consonant clusters
            if (/[bcdfghjklmnpqrstvwxyz]{5,}/i.test(w)) return true;
            return false;
        });
        return { count: suspicious.length, examples: suspicious.slice(0, 5) };
    };

    E.detectTypoPatterns = function(text) {
        return {
            transpositions: (text.match(/\b\w*([a-z])([a-z])\2\1\w*/gi) || []).length,
            doubleLetters: (text.match(/\b\w*(.)\1{2,}\w*/gi) || []).length,
            adjacentKeyErrors: 0 // Would need keyboard layout
        };
    };

    E.detectHomophoneConfusion = function(text) {
        const homophones = [
            ['their', 'there', 'they\'re'],
            ['your', 'you\'re'],
            ['its', 'it\'s'],
            ['to', 'too', 'two'],
            ['then', 'than'],
            ['affect', 'effect'],
            ['accept', 'except'],
            ['loose', 'lose'],
            ['weather', 'whether'],
            ['principal', 'principle']
        ];
        const found = [];
        homophones.forEach(group => {
            const present = group.filter(w => new RegExp('\\b' + w + '\\b', 'i').test(text));
            if (present.length >= 2) {
                found.push(group);
            }
        });
        return { groups: found, count: found.length };
    };

    E.spellingInconsistency = function(text) {
        const variants = [
            ['color', 'colour'],
            ['organize', 'organise'],
            ['realize', 'realise'],
            ['center', 'centre'],
            ['theater', 'theatre'],
            ['analyze', 'analyse'],
            ['defense', 'defence'],
            ['traveled', 'travelled']
        ];
        let inconsistencies = 0;
        variants.forEach(([us, uk]) => {
            const hasUs = new RegExp('\\b' + us, 'i').test(text);
            const hasUk = new RegExp('\\b' + uk, 'i').test(text);
            if (hasUs && hasUk) inconsistencies++;
        });
        return inconsistencies;
    };

    E.detectBritishAmericanMix = function(text) {
        const british = (text.match(/\b(colour|favourite|honour|labour|centre|theatre|analyse|organise|realise|defence|licence|grey)\b/gi) || []).length;
        const american = (text.match(/\b(color|favorite|honor|labor|center|theater|analyze|organize|realize|defense|license|gray)\b/gi) || []).length;
        return {
            british,
            american,
            mixed: british > 0 && american > 0
        };
    };

    E.detectSubjectVerbDisagreement = function(text) {
        // Simple heuristic patterns
        const patterns = [
            /\b(he|she|it)\s+(are|were|have)\b/gi,
            /\b(they|we)\s+(is|was|has)\b/gi,
            /\b(I)\s+(is|was|has)\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectArticleErrors = function(text) {
        const aBeforeVowel = (text.match(/\ba\s+[aeiou]/gi) || []).length;
        const anBeforeConsonant = (text.match(/\ban\s+[^aeiou\s]/gi) || []).length;
        return { aBeforeVowel, anBeforeConsonant, total: aBeforeVowel + anBeforeConsonant };
    };

    E.detectPrepositionErrors = function(text) {
        // Common preposition errors
        const errors = [
            /\bon\s+the\s+internet/gi,  // Should be "on the internet" - actually correct
            /\bdifferent\s+than\b/gi,   // Could be "different from"
            /\bsame\s+like\b/gi,        // Should be "same as"
        ];
        let count = 0;
        errors.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectTenseInconsistency = function(text) {
        const sentences = text.split(/[.!?]+/);
        let inconsistencies = 0;
        sentences.forEach(s => {
            const past = (s.match(/\b\w+ed\b/g) || []).length;
            const present = (s.match(/\b(is|are|has|have|do|does)\b/gi) || []).length;
            if (past > 0 && present > 0) inconsistencies++;
        });
        return inconsistencies;
    };

    E.detectPronounErrors = function(text) {
        // Common pronoun case errors
        const patterns = [
            /\bme\s+and\s+\w+\s+(is|are|was|were)/gi,  // "Me and him are" instead of "He and I are"
            /\bbetween\s+you\s+and\s+I\b/gi,  // Should be "between you and me"
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectDoubleNegatives = function(text) {
        const patterns = [
            /\b(don't|doesn't|didn't|won't|wouldn't|can't|couldn't)\s+\w+\s+(no|nothing|nobody|nowhere|never)\b/gi,
            /\b(no|nothing|nobody)\s+\w+\s+(no|nothing|nobody)\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectMissingPunctuation = function(text) {
        // Sentences that seem to run on
        const longWithoutPunct = (text.match(/[^.!?,;:]{100,}/g) || []).length;
        return longWithoutPunct;
    };

    E.detectExtraPunctuation = function(text) {
        const double = (text.match(/[.,;:]{2,}/g) || []).length;
        const spaceBeforePunct = (text.match(/\s+[.,;:!?]/g) || []).length;
        return { double, spaceBeforePunct, total: double + spaceBeforePunct };
    };

    E.detectCommaErrors = function(text) {
        const commaSplice = (text.match(/[a-z]+,\s*[a-z]+\s+(is|are|was|were)\b/gi) || []).length;
        const missingOxford = (text.match(/\b\w+,\s*\w+\s+and\s+\w+\b/gi) || []).length;
        return { commaSplice, missingOxford };
    };

    E.detectApostropheErrors = function(text) {
        const itsError = (text.match(/\bits\s+(is|has)\b/gi) || []).length;
        const pluralApostrophe = (text.match(/\b\w+'s\s+(are|were)\b/gi) || []).length;
        return { itsError, pluralApostrophe, total: itsError + pluralApostrophe };
    };

    E.detectQuotationErrors = function(text) {
        const unmatchedSingle = (text.match(/'/g) || []).length % 2;
        const unmatchedDouble = (text.match(/"/g) || []).length % 2;
        return { unmatchedSingle, unmatchedDouble };
    };

    E.detectRepetitions = function(text) {
        const wordRepeat = (text.match(/\b(\w+)\s+\1\b/gi) || []).length;
        return { wordRepeat };
    };

    E.countFillerWords = function(text) {
        let count = 0;
        E.FILLER_WORDS.forEach(f => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + f.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectFalseStarts = function(text) {
        // "I mean", "well", "so" at sentence start
        return (text.match(/(^|\.\s+)(Well|So|I mean|Like)\b/gi) || []).length;
    };

    E.detectSelfCorrections = function(text) {
        return (text.match(/\b(I mean|that is|or rather|actually|well actually)\b/gi) || []).length;
    };

    E.detectIncompleteThoughts = function(text) {
        return (text.match(/[^.!?]\s*\.{3}|—\s*$/gm) || []).length;
    };

    E.detectWordSearches = function(text) {
        return (text.match(/\b(um|uh|er|ah)\b/gi) || []).length;
    };

    E.detectArticleOmission = function(text) {
        // Common in ESL: missing articles before nouns
        const patterns = /\b(is|are|was|were)\s+[a-z]+\s+[a-z]+(tion|ment|ness|ity)\b/gi;
        return (text.match(patterns) || []).length;
    };

    E.detectPluralErrors = function(text) {
        // Irregular plurals misused
        const patterns = [
            /\bmany\s+(information|advice|equipment|furniture)\b/gi,
            /\b(childs|mans|womans|foots|tooths|mouses|gooses)\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectWordOrderIssues = function(text) {
        // Adjective order issues, adverb placement
        const patterns = [
            /\balways\s+I\b/gi,  // "Always I do" instead of "I always do"
            /\bonly\s+I\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectPrepositionMisuse = function(text) {
        const patterns = [
            /\bon\s+the\s+\d{4}\b/gi,  // "on the 2024" instead of "in 2024"
            /\bsince\s+\d+\s+years?\b/gi,  // "since 5 years" instead of "for 5 years"
            /\bdepend\s+of\b/gi  // "depend of" instead of "depend on"
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectCollocationalErrors = function(text) {
        // Non-standard collocations
        const patterns = [
            /\bmake\s+a\s+decision\b/gi,  // correct
            /\bdo\s+a\s+mistake\b/gi,  // error - should be "make a mistake"
            /\bmake\s+homework\b/gi  // error - should be "do homework"
        ];
        let errors = 0;
        if (/\bdo\s+a\s+mistake\b/gi.test(text)) errors++;
        if (/\bmake\s+homework\b/gi.test(text)) errors++;
        return errors;
    };

    E.detectL1Interference = function(text) {
        // Common L1 interference patterns
        const patterns = [
            /\bthe\s+(my|your|his|her|their)\b/gi,  // "the my friend"
            /\bmore\s+\w+er\b/gi,  // "more better"
            /\bmost\s+\w+est\b/gi,  // "most biggest"
            /\bI\s+am\s+agree\b/gi  // "I am agree"
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    // Export
    if (typeof window !== 'undefined') {
        window.ExtendedHeuristicsAnalyzer = E;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = E;
    }
})();
