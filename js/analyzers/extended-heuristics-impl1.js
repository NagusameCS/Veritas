/**
 * VERITAS — Extended Heuristics Implementation Part 1
 * ====================================================
 * Implementation of Linguistic, Stylometric, and Cognitive Load heuristics
 */

// Extend the ExtendedHeuristicsAnalyzer with implementations
(function() {
    const E = typeof ExtendedHeuristicsAnalyzer !== 'undefined' ? ExtendedHeuristicsAnalyzer : {};

    // ========================================================================
    // WORD LISTS FOR ANALYSIS
    // ========================================================================
    
    E.FUNCTION_WORDS = new Set([
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
        'now', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself',
        'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
        'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'would', 'could', 'should',
        'might', 'must', 'shall', 'may'
    ]);

    E.POSITIVE_WORDS = new Set([
        'good', 'great', 'excellent', 'wonderful', 'fantastic', 'amazing',
        'beautiful', 'perfect', 'brilliant', 'outstanding', 'superb', 'love',
        'happy', 'joy', 'delighted', 'pleased', 'grateful', 'thankful', 'glad',
        'excited', 'thrilled', 'blessed', 'fortunate', 'lucky', 'awesome',
        'incredible', 'remarkable', 'extraordinary', 'magnificent', 'splendid',
        'terrific', 'marvelous', 'fabulous', 'phenomenal', 'spectacular',
        'exceptional', 'impressive', 'admirable', 'commendable', 'praiseworthy',
        'best', 'better', 'superior', 'finest', 'optimal', 'ideal', 'prime',
        'success', 'successful', 'triumph', 'victory', 'achievement', 'accomplish',
        'beneficial', 'advantage', 'positive', 'gain', 'improve', 'enhance',
        'progress', 'prosper', 'thrive', 'flourish', 'bloom', 'grow'
    ]);

    E.NEGATIVE_WORDS = new Set([
        'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'poor', 'worst',
        'hate', 'sad', 'angry', 'upset', 'disappointed', 'frustrated', 'annoyed',
        'irritated', 'furious', 'enraged', 'miserable', 'depressed', 'unhappy',
        'unfortunate', 'tragic', 'disastrous', 'catastrophic', 'devastating',
        'fail', 'failure', 'defeat', 'loss', 'lose', 'wrong', 'error', 'mistake',
        'problem', 'issue', 'trouble', 'difficulty', 'challenge', 'obstacle',
        'barrier', 'harm', 'damage', 'hurt', 'pain', 'suffer', 'struggle',
        'weak', 'weakness', 'flaw', 'defect', 'fault', 'blame', 'guilt',
        'shame', 'regret', 'sorry', 'worry', 'anxiety', 'fear', 'afraid',
        'scared', 'terrified', 'panic', 'stress', 'tension', 'conflict',
        'crisis', 'danger', 'threat', 'risk', 'hazard', 'peril'
    ]);

    E.EMOTION_LEXICON = {
        joy: ['happy', 'joy', 'delight', 'pleasure', 'cheerful', 'elated', 'ecstatic', 'jubilant', 'bliss', 'euphoria', 'merry', 'gleeful', 'content', 'satisfied', 'thrilled'],
        anger: ['angry', 'furious', 'rage', 'wrath', 'irritated', 'annoyed', 'outraged', 'incensed', 'enraged', 'livid', 'hostile', 'resentful', 'bitter', 'indignant', 'exasperated'],
        fear: ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panicked', 'dread', 'horror', 'alarmed', 'apprehensive', 'uneasy', 'tense', 'paranoid'],
        sadness: ['sad', 'unhappy', 'depressed', 'melancholy', 'grief', 'sorrow', 'miserable', 'gloomy', 'heartbroken', 'dejected', 'despondent', 'mournful', 'woeful', 'dismal', 'blue'],
        surprise: ['surprised', 'amazed', 'astonished', 'shocked', 'startled', 'stunned', 'astounded', 'bewildered', 'dumbfounded', 'flabbergasted', 'unexpected', 'sudden'],
        disgust: ['disgusted', 'repulsed', 'revolted', 'nauseated', 'sickened', 'appalled', 'loathing', 'abhorrent', 'detestable', 'vile', 'gross', 'repugnant'],
        trust: ['trust', 'faith', 'confidence', 'belief', 'reliable', 'dependable', 'honest', 'loyal', 'faithful', 'trustworthy', 'credible', 'sincere'],
        anticipation: ['anticipate', 'expect', 'await', 'hope', 'eager', 'excited', 'look forward', 'prepare', 'ready', 'watchful', 'vigilant']
    };

    E.INTENSIFIERS = ['very', 'really', 'extremely', 'incredibly', 'absolutely', 'completely', 'totally', 'utterly', 'highly', 'deeply', 'greatly', 'strongly', 'truly', 'quite', 'rather', 'somewhat', 'fairly', 'pretty', 'so', 'too', 'awfully', 'terribly', 'remarkably', 'exceptionally', 'extraordinarily'];

    E.HEDGES = ['maybe', 'perhaps', 'possibly', 'probably', 'might', 'could', 'may', 'seem', 'appear', 'tend', 'suggest', 'indicate', 'somewhat', 'rather', 'fairly', 'kind of', 'sort of', 'a bit', 'a little', 'in a way', 'to some extent', 'arguably', 'presumably', 'apparently', 'supposedly'];

    E.FILLER_WORDS = ['um', 'uh', 'er', 'ah', 'like', 'you know', 'i mean', 'basically', 'actually', 'literally', 'honestly', 'frankly', 'well', 'so', 'anyway', 'right', 'okay', 'just', 'really'];

    E.ACADEMIC_VOCABULARY = ['analysis', 'approach', 'assessment', 'assumption', 'authority', 'available', 'benefit', 'concept', 'consistent', 'constitutional', 'context', 'contract', 'create', 'data', 'definition', 'derived', 'distribution', 'economic', 'environment', 'established', 'estimate', 'evidence', 'export', 'factors', 'financial', 'formula', 'function', 'identified', 'income', 'indicate', 'individual', 'interpretation', 'involved', 'issues', 'labour', 'legal', 'legislation', 'major', 'method', 'occur', 'percent', 'period', 'policy', 'principle', 'procedure', 'process', 'required', 'research', 'response', 'role', 'section', 'sector', 'significant', 'similar', 'source', 'specific', 'structure', 'theory', 'variables'];

    E.SLANG_WORDS = ['gonna', 'wanna', 'gotta', 'kinda', 'sorta', 'dunno', 'lemme', 'gimme', 'cuz', 'cause', 'ya', 'yeah', 'yep', 'nope', 'nah', 'dude', 'bro', 'bruh', 'fam', 'lit', 'fire', 'slay', 'vibe', 'mood', 'lowkey', 'highkey', 'salty', 'shade', 'tea', 'stan', 'simp', 'cap', 'sus', 'bussin', 'bet', 'periodt', 'oof', 'yeet', 'cringe', 'based', 'woke', 'snatched', 'drip'];

    E.DISCOURSE_MARKERS = {
        additive: ['and', 'also', 'moreover', 'furthermore', 'additionally', 'besides', 'in addition', 'as well', 'too', 'likewise', 'similarly', 'equally', 'plus'],
        adversative: ['but', 'however', 'nevertheless', 'nonetheless', 'although', 'though', 'yet', 'still', 'whereas', 'while', 'on the other hand', 'in contrast', 'conversely', 'instead', 'rather'],
        causal: ['because', 'since', 'as', 'therefore', 'thus', 'hence', 'consequently', 'so', 'accordingly', 'as a result', 'due to', 'owing to', 'for this reason'],
        temporal: ['then', 'next', 'after', 'before', 'when', 'while', 'during', 'meanwhile', 'subsequently', 'previously', 'finally', 'eventually', 'firstly', 'secondly', 'lastly']
    };

    // ========================================================================
    // LINGUISTIC MICRO-FEATURES IMPLEMENTATIONS
    // ========================================================================

    E.countPrefixes = function(words) {
        const prefixes = ['un', 're', 'in', 'im', 'ir', 'il', 'dis', 'en', 'em', 'non', 'pre', 'mis', 'anti', 'de', 'over', 'under', 'sub', 'super', 'semi', 'mid', 'trans', 'inter', 'counter', 'ultra', 'mega', 'micro', 'mini', 'multi', 'mono', 'bi', 'tri', 'poly', 'auto', 'neo', 'post', 'pro', 'pseudo', 'quasi', 'self', 'co', 'ex', 'extra', 'hyper', 'infra', 'intra', 'macro', 'mal', 'meta', 'out', 'para', 'retro'];
        let count = 0;
        words.forEach(word => {
            prefixes.forEach(prefix => {
                if (word.startsWith(prefix) && word.length > prefix.length + 2) count++;
            });
        });
        return { count, ratio: count / Math.max(words.length, 1) };
    };

    E.countSuffixes = function(words) {
        const suffixes = ['ing', 'ed', 'ly', 'er', 'est', 'tion', 'sion', 'ness', 'ment', 'able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ial', 'ic', 'ical', 'ish', 'like', 'ward', 'wise', 'dom', 'ship', 'hood', 'ery', 'ry', 'ty', 'ity', 'ance', 'ence', 'ant', 'ent', 'ism', 'ist', 'ize', 'ise', 'fy', 'ate', 'en', 'ling', 'let', 'ette'];
        let count = 0;
        words.forEach(word => {
            suffixes.forEach(suffix => {
                if (word.endsWith(suffix) && word.length > suffix.length + 2) count++;
            });
        });
        return { count, ratio: count / Math.max(words.length, 1) };
    };

    E.countCompoundWords = function(words) {
        // Detect potential compound words by length and patterns
        const compounds = words.filter(w => w.length > 10 && /[a-z]{4,}[a-z]{4,}/.test(w));
        return { count: compounds.length, examples: compounds.slice(0, 5) };
    };

    E.countDerivedWords = function(words) {
        const derivationalSuffixes = ['tion', 'sion', 'ness', 'ment', 'ity', 'ism', 'ist', 'ize'];
        let count = 0;
        words.forEach(w => {
            if (derivationalSuffixes.some(s => w.endsWith(s))) count++;
        });
        return { count, ratio: count / Math.max(words.length, 1) };
    };

    E.countInflectedForms = function(words) {
        const inflectional = words.filter(w => /ed$|ing$|s$|er$|est$/.test(w));
        return { count: inflectional.length, ratio: inflectional.length / Math.max(words.length, 1) };
    };

    E.calculateRootWordRatio = function(words) {
        const shortWords = words.filter(w => w.length <= 5 && !/ed$|ing$|ly$/.test(w));
        return shortWords.length / Math.max(words.length, 1);
    };

    E.calculateMorphemeComplexity = function(words) {
        // Estimate average morphemes per word
        let totalMorphemes = 0;
        words.forEach(w => {
            let morphemes = 1;
            if (w.length > 3 && /^(un|re|in|dis|pre|mis)/.test(w)) morphemes++;
            if (w.length > 3 && /(tion|sion|ness|ment|able|ible|ful|less|ous|ive|ing|ed|ly|er|est)$/.test(w)) morphemes++;
            totalMorphemes += morphemes;
        });
        return totalMorphemes / Math.max(words.length, 1);
    };

    E.countAlliteration = function(text) {
        const sentences = text.split(/[.!?]+/);
        let count = 0;
        sentences.forEach(sent => {
            const words = sent.toLowerCase().match(/\b[a-z]+/g) || [];
            for (let i = 0; i < words.length - 1; i++) {
                if (words[i][0] === words[i+1][0]) count++;
            }
        });
        return count;
    };

    E.countAssonance = function(text) {
        const vowelPatterns = text.toLowerCase().match(/[aeiou]+/g) || [];
        const patternCounts = {};
        vowelPatterns.forEach(p => patternCounts[p] = (patternCounts[p] || 0) + 1);
        return Object.values(patternCounts).filter(c => c > 2).length;
    };

    E.countConsonance = function(text) {
        const consonantPatterns = text.toLowerCase().match(/[bcdfghjklmnpqrstvwxyz]{2,}/g) || [];
        const patternCounts = {};
        consonantPatterns.forEach(p => patternCounts[p] = (patternCounts[p] || 0) + 1);
        return Object.values(patternCounts).filter(c => c > 2).length;
    };

    E.calculateSyllableVariation = function(words) {
        const syllableCounts = words.map(w => this.countSyllables(w));
        if (syllableCounts.length < 2) return 0;
        const mean = syllableCounts.reduce((a, b) => a + b, 0) / syllableCounts.length;
        const variance = syllableCounts.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / syllableCounts.length;
        return Math.sqrt(variance);
    };

    E.countSyllables = function(word) {
        word = word.toLowerCase();
        if (word.length <= 3) return 1;
        word = word.replace(/(?:[^laeiouy]es|ed|[^laeiouy]e)$/, '');
        word = word.replace(/^y/, '');
        const matches = word.match(/[aeiouy]{1,2}/g);
        return matches ? matches.length : 1;
    };

    E.detectPhoneticPatterns = function(text) {
        return {
            sibilants: (text.match(/[sxz]|sh|ch/gi) || []).length,
            plosives: (text.match(/[bpdtgk]/gi) || []).length,
            nasals: (text.match(/[mn]/gi) || []).length,
            liquids: (text.match(/[lr]/gi) || []).length,
            fricatives: (text.match(/[fvszh]|th/gi) || []).length
        };
    };

    E.calculateRhythmScore = function(sentences) {
        if (sentences.length === 0) return 0;
        const lengths = sentences.map(s => (s.match(/\b\w+/g) || []).length);
        const diffs = [];
        for (let i = 1; i < lengths.length; i++) {
            diffs.push(Math.abs(lengths[i] - lengths[i-1]));
        }
        if (diffs.length === 0) return 0;
        const avgDiff = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        return Math.max(0, 1 - avgDiff / 10); // Normalize to 0-1
    };

    E.avgWordLength = function(words) {
        if (words.length === 0) return 0;
        return words.reduce((sum, w) => sum + w.length, 0) / words.length;
    };

    E.wordLengthVariance = function(words) {
        if (words.length < 2) return 0;
        const mean = this.avgWordLength(words);
        const variance = words.reduce((sum, w) => sum + Math.pow(w.length - mean, 2), 0) / words.length;
        return Math.sqrt(variance);
    };

    E.monosyllabicRatio = function(words) {
        const monosyllabic = words.filter(w => this.countSyllables(w) === 1);
        return monosyllabic.length / Math.max(words.length, 1);
    };

    E.polysyllabicRatio = function(words) {
        const polysyllabic = words.filter(w => this.countSyllables(w) >= 3);
        return polysyllabic.length / Math.max(words.length, 1);
    };

    E.wordLengthDistribution = function(words) {
        const dist = {};
        words.forEach(w => {
            const len = w.length;
            dist[len] = (dist[len] || 0) + 1;
        });
        return dist;
    };

    E.countShortWordChains = function(words) {
        let chains = 0;
        let currentChain = 0;
        words.forEach(w => {
            if (w.length <= 3) {
                currentChain++;
                if (currentChain >= 3) chains++;
            } else {
                currentChain = 0;
            }
        });
        return chains;
    };

    E.countLongWordClusters = function(words) {
        let clusters = 0;
        let currentCluster = 0;
        words.forEach(w => {
            if (w.length >= 8) {
                currentCluster++;
                if (currentCluster >= 2) clusters++;
            } else {
                currentCluster = 0;
            }
        });
        return clusters;
    };

    E.vowelConsonantRatio = function(text) {
        const vowels = (text.match(/[aeiou]/gi) || []).length;
        const consonants = (text.match(/[bcdfghjklmnpqrstvwxyz]/gi) || []).length;
        return consonants > 0 ? vowels / consonants : 0;
    };

    E.doubleLetterFrequency = function(text) {
        const doubles = (text.match(/(.)\1/gi) || []).length;
        return doubles / Math.max(text.length, 1);
    };

    E.tripleLetterOccurrence = function(text) {
        return (text.match(/(.)\1\1/gi) || []).length;
    };

    E.characterNGramAnalysis = function(text) {
        const cleanText = text.toLowerCase().replace(/[^a-z]/g, '');
        const bigrams = {};
        const trigrams = {};
        for (let i = 0; i < cleanText.length - 1; i++) {
            const bi = cleanText.substr(i, 2);
            bigrams[bi] = (bigrams[bi] || 0) + 1;
        }
        for (let i = 0; i < cleanText.length - 2; i++) {
            const tri = cleanText.substr(i, 3);
            trigrams[tri] = (trigrams[tri] || 0) + 1;
        }
        return {
            uniqueBigrams: Object.keys(bigrams).length,
            uniqueTrigrams: Object.keys(trigrams).length,
            topBigrams: Object.entries(bigrams).sort((a, b) => b[1] - a[1]).slice(0, 5),
            topTrigrams: Object.entries(trigrams).sort((a, b) => b[1] - a[1]).slice(0, 5)
        };
    };

    E.letterFrequencyDeviation = function(text) {
        // Expected English letter frequencies
        const expected = {e: 12.7, t: 9.1, a: 8.2, o: 7.5, i: 7.0, n: 6.7, s: 6.3, h: 6.1, r: 6.0, d: 4.3, l: 4.0, c: 2.8, u: 2.8, m: 2.4, w: 2.4, f: 2.2, g: 2.0, y: 2.0, p: 1.9, b: 1.5, v: 1.0, k: 0.8, j: 0.15, x: 0.15, q: 0.1, z: 0.07};
        const cleanText = text.toLowerCase().replace(/[^a-z]/g, '');
        const actual = {};
        for (const char of cleanText) {
            actual[char] = (actual[char] || 0) + 1;
        }
        let deviation = 0;
        for (const letter in expected) {
            const actualFreq = ((actual[letter] || 0) / cleanText.length) * 100;
            deviation += Math.abs(actualFreq - expected[letter]);
        }
        return deviation;
    };

    // Make methods available
    if (typeof window !== 'undefined') {
        window.ExtendedHeuristicsAnalyzer = E;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = E;
    }
})();
