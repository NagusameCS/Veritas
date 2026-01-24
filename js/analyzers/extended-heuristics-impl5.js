/**
 * VERITAS — Extended Heuristics Implementation Part 5
 * ====================================================
 * Temporal/Cultural, Discourse, and Statistical implementations
 */

(function() {
    const E = typeof ExtendedHeuristicsAnalyzer !== 'undefined' ? ExtendedHeuristicsAnalyzer : 
              (typeof window !== 'undefined' ? window.ExtendedHeuristicsAnalyzer : {});

    // ========================================================================
    // TEMPORAL & CULTURAL IMPLEMENTATIONS
    // ========================================================================

    E.countPastReferences = function(text) {
        const past = ['yesterday', 'last week', 'last month', 'last year', 'ago', 'previously', 'formerly', 'once', 'used to', 'back then', 'in the past', 'historically'];
        let count = 0;
        past.forEach(p => {
            if (text.toLowerCase().includes(p)) count++;
        });
        return count;
    };

    E.countPresentReferences = function(text) {
        const present = ['today', 'now', 'currently', 'presently', 'at present', 'nowadays', 'these days', 'right now', 'at the moment', 'this moment'];
        let count = 0;
        present.forEach(p => {
            if (text.toLowerCase().includes(p)) count++;
        });
        return count;
    };

    E.countFutureReferences = function(text) {
        const future = ['tomorrow', 'next week', 'next month', 'next year', 'soon', 'eventually', 'in the future', 'will', 'going to', 'plan to', 'intend to', 'expect to'];
        let count = 0;
        future.forEach(f => {
            if (text.toLowerCase().includes(f)) count++;
        });
        return count;
    };

    E.detectSpecificDates = function(text) {
        const patterns = [
            /\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(,?\s+\d{4})?\b/gi,
            /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g,
            /\b\d{4}-\d{2}-\d{2}\b/g,
            /\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.relativeTimeMarkers = function(text) {
        const relative = ['before', 'after', 'during', 'while', 'since', 'until', 'by the time', 'as soon as', 'whenever', 'every time', 'the moment'];
        let count = 0;
        relative.forEach(r => {
            if (text.toLowerCase().includes(r)) count++;
        });
        return count;
    };

    E.temporalConnectives = function(text) {
        const connectives = ['then', 'next', 'afterwards', 'subsequently', 'previously', 'meanwhile', 'simultaneously', 'eventually', 'finally', 'ultimately'];
        let count = 0;
        connectives.forEach(c => {
            if (new RegExp('\\b' + c + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.chronologicalMarkers = function(text) {
        const markers = ['first', 'second', 'third', 'fourth', 'fifth', 'firstly', 'secondly', 'thirdly', 'lastly', 'finally', 'initially', 'originally', 'at first', 'in the beginning', 'at last', 'in the end'];
        let count = 0;
        markers.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.geographicReferences = function(text) {
        const places = (text.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g) || []);
        // Filter to likely place names (simplified)
        return places.length;
    };

    E.culturalReferences = function(text) {
        const cultural = ['christmas', 'thanksgiving', 'easter', 'hanukkah', 'diwali', 'ramadan', 'eid', 'new year', 'halloween', 'valentine', 'independence day', 'memorial day', 'labor day'];
        let count = 0;
        cultural.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    E.idiomUsage = function(text) {
        const idioms = ['piece of cake', 'break a leg', 'hit the nail', 'bite the bullet', 'break the ice', 'call it a day', 'cut to the chase', 'get out of hand', 'hang in there', 'it takes two', 'kill two birds', 'let the cat out', 'miss the boat', 'once in a blue moon', 'piece of my mind', 'pull someone\'s leg', 'speak of the devil', 'spill the beans', 'take it with a grain', 'the ball is in your court', 'under the weather', 'when pigs fly'];
        let count = 0;
        idioms.forEach(i => {
            if (text.toLowerCase().includes(i)) count++;
        });
        return count;
    };

    E.proverbUsage = function(text) {
        const proverbs = ['a bird in the hand', 'actions speak louder', 'all that glitters', 'an apple a day', 'better late than never', 'birds of a feather', 'don\'t count your chickens', 'don\'t cry over spilled', 'don\'t judge a book', 'don\'t put all your eggs', 'early bird catches', 'every cloud has', 'fortune favors', 'great minds think', 'honesty is the best', 'practice makes perfect', 'the early bird', 'the pen is mightier', 'time heals all', 'two heads are better', 'when in rome', 'you can\'t teach an old dog'];
        let count = 0;
        proverbs.forEach(p => {
            if (text.toLowerCase().includes(p)) count++;
        });
        return count;
    };

    E.popCultureReferences = function(text) {
        // Would need current pop culture database - simplified
        const patterns = ['movie', 'film', 'tv show', 'series', 'album', 'song', 'celebrity', 'star wars', 'marvel', 'harry potter', 'game of thrones', 'netflix', 'spotify', 'tiktok', 'instagram', 'twitter', 'youtube'];
        let count = 0;
        patterns.forEach(p => {
            if (text.toLowerCase().includes(p)) count++;
        });
        return count;
    };

    E.historicalReferences = function(text) {
        const historical = ['world war', 'civil war', 'revolution', 'ancient', 'medieval', 'renaissance', 'industrial revolution', 'cold war', 'great depression', 'founding fathers', 'declaration of independence', 'constitution', 'civil rights', 'slavery', 'colonialism', 'empire'];
        let count = 0;
        historical.forEach(h => {
            if (text.toLowerCase().includes(h)) count++;
        });
        return count;
    };

    E.regionalVocabulary = function(text) {
        const british = ['flat', 'lift', 'lorry', 'biscuit', 'crisps', 'queue', 'holiday', 'mobile', 'post', 'rubbish'];
        const american = ['apartment', 'elevator', 'truck', 'cookie', 'chips', 'line', 'vacation', 'cell phone', 'mail', 'garbage'];
        let britishCount = 0, americanCount = 0;
        british.forEach(b => { if (new RegExp('\\b' + b + '\\b', 'i').test(text)) britishCount++; });
        american.forEach(a => { if (new RegExp('\\b' + a + '\\b', 'i').test(text)) americanCount++; });
        return { british: britishCount, american: americanCount };
    };

    E.dialectPatterns = function(text) {
        // Simplified dialect detection
        const southern = ['y\'all', 'fixin\' to', 'might could'];
        const aave = ['finna', 'ain\'t', 'bout to'];
        let southernCount = 0, aaveCount = 0;
        southern.forEach(s => { if (text.toLowerCase().includes(s)) southernCount++; });
        aave.forEach(a => { if (text.toLowerCase().includes(a)) aaveCount++; });
        return { southern: southernCount, aave: aaveCount };
    };

    E.spellingVariants = function(text) {
        return E.detectBritishAmericanMix(text);
    };

    E.idiomaticExpressions = function(text) {
        return E.idiomUsage(text) + E.proverbUsage(text);
    };

    E.identifySlangGeneration = function(text) {
        const genZ = ['sus', 'cap', 'bet', 'bussin', 'slay', 'periodt', 'no cap', 'lowkey', 'highkey', 'based', 'mid'];
        const millennial = ['adulting', 'bae', 'on fleek', 'squad goals', 'fomo', 'yolo', 'feels', 'can\'t even'];
        const older = ['groovy', 'far out', 'rad', 'gnarly', 'tubular'];
        
        let genZCount = 0, millennialCount = 0, olderCount = 0;
        genZ.forEach(g => { if (text.toLowerCase().includes(g)) genZCount++; });
        millennial.forEach(m => { if (text.toLowerCase().includes(m)) millennialCount++; });
        older.forEach(o => { if (text.toLowerCase().includes(o)) olderCount++; });
        
        if (genZCount > millennialCount && genZCount > olderCount) return 'Gen Z';
        if (millennialCount > genZCount && millennialCount > olderCount) return 'Millennial';
        if (olderCount > 0) return 'Older generation';
        return 'Neutral';
    };

    E.technologyReferences = function(text) {
        const tech = ['ai', 'artificial intelligence', 'machine learning', 'blockchain', 'cryptocurrency', 'bitcoin', 'nft', 'metaverse', 'virtual reality', 'vr', 'augmented reality', 'ar', 'cloud computing', 'saas', 'api', 'smartphone', 'app', 'social media', 'algorithm', '5g', 'iot', 'internet of things'];
        let count = 0;
        tech.forEach(t => {
            if (text.toLowerCase().includes(t)) count++;
        });
        return count;
    };

    E.mediaReferences = function(text) {
        const media = ['video', 'stream', 'podcast', 'blog', 'vlog', 'influencer', 'content creator', 'viral', 'trending', 'meme', 'gif', 'retweet', 'share', 'subscribe', 'follower', 'like'];
        let count = 0;
        media.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.socialMediaIndicators = function(text) {
        return E.hashtagCount(text) + E.mentionCount(text) + E.emojiCount(text);
    };

    // ========================================================================
    // DISCOURSE & PRAGMATIC IMPLEMENTATIONS
    // ========================================================================

    E.topicConsistency = function(paragraphs) {
        if (paragraphs.length < 2) return 1;
        let totalOverlap = 0;
        for (let i = 1; i < paragraphs.length; i++) {
            const prevWords = new Set((paragraphs[i-1].toLowerCase().match(/\b[a-z]{4,}\b/g) || []));
            const currWords = new Set((paragraphs[i].toLowerCase().match(/\b[a-z]{4,}\b/g) || []));
            const overlap = [...prevWords].filter(w => currWords.has(w)).length;
            totalOverlap += overlap / Math.max(Math.min(prevWords.size, currWords.size), 1);
        }
        return totalOverlap / (paragraphs.length - 1);
    };

    E.lexicalCohesion = function(sentences) {
        if (sentences.length < 2) return 1;
        let totalOverlap = 0;
        for (let i = 1; i < sentences.length; i++) {
            const prevWords = new Set((sentences[i-1].toLowerCase().match(/\b[a-z]+\b/g) || []));
            const currWords = new Set((sentences[i].toLowerCase().match(/\b[a-z]+\b/g) || []));
            const overlap = [...prevWords].filter(w => currWords.has(w) && !E.FUNCTION_WORDS.has(w)).length;
            totalOverlap += overlap;
        }
        return totalOverlap / (sentences.length - 1);
    };

    E.referentialCohesion = function(text) {
        const pronouns = (text.match(/\b(he|she|it|they|this|that|these|those|who|which)\b/gi) || []).length;
        const nouns = (text.match(/\b[A-Z][a-z]+\b/g) || []).length;
        return nouns > 0 ? Math.min(pronouns / nouns, 2) : 0;
    };

    E.conjunctionCohesion = function(text) {
        const conjunctions = (text.match(/\b(and|but|or|so|yet|because|although|while|when|if|since|therefore|however|moreover|furthermore|nevertheless|consequently)\b/gi) || []).length;
        const sentences = text.split(/[.!?]+/).length;
        return conjunctions / Math.max(sentences, 1);
    };

    E.thematicProgression = function(paragraphs) {
        // Simplified: check if key terms carry forward
        if (paragraphs.length < 2) return 1;
        const keyTerms = [];
        paragraphs.forEach(p => {
            const words = p.toLowerCase().match(/\b[a-z]{5,}\b/g) || [];
            const freq = {};
            words.forEach(w => freq[w] = (freq[w] || 0) + 1);
            const top = Object.entries(freq).sort((a, b) => b[1] - a[1]).slice(0, 3).map(e => e[0]);
            keyTerms.push(new Set(top));
        });
        let carried = 0;
        for (let i = 1; i < keyTerms.length; i++) {
            const overlap = [...keyTerms[i-1]].filter(t => keyTerms[i].has(t)).length;
            if (overlap > 0) carried++;
        }
        return carried / (paragraphs.length - 1);
    };

    E.additiveConnectives = function(text) {
        let count = 0;
        E.DISCOURSE_MARKERS.additive.forEach(c => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + c.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.adversativeConnectives = function(text) {
        let count = 0;
        E.DISCOURSE_MARKERS.adversative.forEach(c => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + c.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.causalConnectives = function(text) {
        let count = 0;
        E.DISCOURSE_MARKERS.causal.forEach(c => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + c.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.temporalConnectivesCount = function(text) {
        let count = 0;
        E.DISCOURSE_MARKERS.temporal.forEach(c => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + c + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.exemplificationMarkers = function(text) {
        const markers = ['for example', 'for instance', 'such as', 'like', 'including', 'e.g.', 'namely', 'specifically', 'in particular', 'to illustrate'];
        let count = 0;
        markers.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.summaryMarkers = function(text) {
        const markers = ['in summary', 'to summarize', 'in conclusion', 'to conclude', 'overall', 'in short', 'briefly', 'to sum up', 'in brief', 'all in all'];
        let count = 0;
        markers.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.countAssertives = function(text) {
        const assertives = ['claim', 'assert', 'state', 'maintain', 'argue', 'contend', 'affirm', 'declare', 'insist', 'believe'];
        let count = 0;
        assertives.forEach(a => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + a + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countDirectives = function(text) {
        const directives = ['must', 'should', 'need to', 'have to', 'require', 'demand', 'order', 'command', 'request', 'ask', 'suggest', 'recommend', 'advise', 'urge', 'warn'];
        let count = 0;
        directives.forEach(d => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + d.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countCommissives = function(text) {
        const commissives = ['promise', 'commit', 'pledge', 'vow', 'guarantee', 'assure', 'swear', 'undertake', 'will', 'shall'];
        let count = 0;
        commissives.forEach(c => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + c + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countExpressives = function(text) {
        const expressives = ['thank', 'congratulate', 'apologize', 'condole', 'welcome', 'praise', 'criticize', 'blame', 'complain', 'regret'];
        let count = 0;
        expressives.forEach(e => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + e + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.countDeclaratives = function(text) {
        const declaratives = ['declare', 'pronounce', 'announce', 'proclaim', 'decree', 'sentence', 'name', 'appoint', 'nominate', 'define'];
        let count = 0;
        declaratives.forEach(d => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + d + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.questionTypeAnalysis = function(text) {
        const questions = text.match(/[^.!]*\?/g) || [];
        const types = { yes_no: 0, wh: 0, rhetorical: 0, tag: 0 };
        questions.forEach(q => {
            if (/\b(who|what|where|when|why|how|which|whose)\b/i.test(q)) types.wh++;
            else if (/,\s*(isn't|aren't|wasn't|weren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't)\s+(it|he|she|they|you)\s*\?/i.test(q)) types.tag++;
            else if (/\b(do|does|did|is|are|was|were|can|could|will|would|should|have|has|had)\s+/i.test(q)) types.yes_no++;
            else types.rhetorical++;
        });
        return types;
    };

    E.politenessMarkers = function(text) {
        const polite = ['please', 'thank you', 'thanks', 'kindly', 'would you mind', 'could you', 'may i', 'excuse me', 'pardon', 'sorry', 'appreciate', 'grateful'];
        let count = 0;
        polite.forEach(p => {
            if (text.toLowerCase().includes(p)) count++;
        });
        return count;
    };

    E.mitigationDevices = function(text) {
        const mitigators = ['sort of', 'kind of', 'a bit', 'a little', 'somewhat', 'rather', 'fairly', 'quite', 'slightly', 'perhaps', 'maybe', 'possibly', 'might', 'could'];
        let count = 0;
        mitigators.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.emphasisDevices = function(text) {
        const emphasis = ['very', 'really', 'extremely', 'absolutely', 'definitely', 'certainly', 'indeed', 'in fact', 'actually', 'literally', 'totally', 'completely'];
        let count = 0;
        emphasis.forEach(e => {
            if (text.toLowerCase().includes(e)) count++;
        });
        return count;
    };

    E.evidentialMarkers = function(text) {
        const evidential = ['apparently', 'reportedly', 'allegedly', 'supposedly', 'according to', 'it is said', 'they say', 'i heard', 'it seems', 'it appears'];
        let count = 0;
        evidential.forEach(e => {
            if (text.toLowerCase().includes(e)) count++;
        });
        return count;
    };

    E.epistemicMarkers = function(text) {
        const epistemic = ['know', 'believe', 'think', 'suppose', 'assume', 'guess', 'imagine', 'suspect', 'doubt', 'certain', 'sure', 'confident', 'convinced'];
        let count = 0;
        epistemic.forEach(e => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + e + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.attitudeMarkers = function(text) {
        const attitude = ['unfortunately', 'fortunately', 'surprisingly', 'interestingly', 'importantly', 'significantly', 'remarkably', 'notably', 'sadly', 'happily', 'hopefully'];
        let count = 0;
        attitude.forEach(a => {
            if (text.toLowerCase().includes(a)) count++;
        });
        return count;
    };

    E.claimMarkers = function(text) {
        const claims = ['i argue', 'i claim', 'i contend', 'i maintain', 'i assert', 'the argument is', 'the claim is', 'this suggests', 'this shows', 'this demonstrates'];
        let count = 0;
        claims.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    E.argumentEvidenceMarkers = function(text) {
        const evidence = ['because', 'since', 'as', 'for', 'given that', 'due to', 'the evidence shows', 'research indicates', 'studies show', 'data suggests', 'statistics reveal'];
        let count = 0;
        evidence.forEach(e => {
            if (text.toLowerCase().includes(e)) count++;
        });
        return count;
    };

    E.counterargumentMarkers = function(text) {
        const counter = ['however', 'but', 'yet', 'although', 'though', 'while', 'whereas', 'on the other hand', 'in contrast', 'conversely', 'nevertheless', 'nonetheless', 'critics argue', 'some might say', 'opponents claim'];
        let count = 0;
        counter.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    E.concessionMarkers = function(text) {
        const concession = ['admittedly', 'granted', 'of course', 'certainly', 'to be sure', 'i acknowledge', 'i admit', 'while it is true', 'although', 'even though', 'despite'];
        let count = 0;
        concession.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    E.reasoningMarkers = function(text) {
        const reasoning = ['therefore', 'thus', 'hence', 'consequently', 'as a result', 'so', 'it follows that', 'this means', 'this implies', 'we can conclude'];
        let count = 0;
        reasoning.forEach(r => {
            if (text.toLowerCase().includes(r)) count++;
        });
        return count;
    };

    E.conclusionMarkers = function(text) {
        const conclusion = ['in conclusion', 'to conclude', 'in summary', 'to summarize', 'finally', 'ultimately', 'in the end', 'overall', 'all things considered', 'taking everything into account'];
        let count = 0;
        conclusion.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    // ========================================================================
    // STATISTICAL DISTRIBUTION IMPLEMENTATIONS
    // ========================================================================

    E.calculateZipfSlope = function(words) {
        if (words.length < 10) return 0;
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const sorted = Object.values(freq).sort((a, b) => b - a);
        const ranks = sorted.map((_, i) => i + 1);
        
        // Linear regression on log-log scale
        const logRanks = ranks.map(r => Math.log(r));
        const logFreqs = sorted.map(f => Math.log(f));
        
        const n = logRanks.length;
        const sumX = logRanks.reduce((a, b) => a + b, 0);
        const sumY = logFreqs.reduce((a, b) => a + b, 0);
        const sumXY = logRanks.reduce((sum, x, i) => sum + x * logFreqs[i], 0);
        const sumX2 = logRanks.reduce((sum, x) => sum + x * x, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return slope;
    };

    E.calculateZipfDeviation = function(words) {
        const slope = E.calculateZipfSlope(words);
        // Ideal Zipf slope is -1
        return Math.abs(slope + 1);
    };

    E.calculateZipfRSquared = function(words) {
        if (words.length < 10) return 0;
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const sorted = Object.values(freq).sort((a, b) => b - a);
        
        const logFreqs = sorted.map(f => Math.log(f));
        const ranks = sorted.map((_, i) => i + 1);
        const logRanks = ranks.map(r => Math.log(r));
        
        // Calculate R-squared
        const meanY = logFreqs.reduce((a, b) => a + b, 0) / logFreqs.length;
        const slope = E.calculateZipfSlope(words);
        const intercept = meanY - slope * (logRanks.reduce((a, b) => a + b, 0) / logRanks.length);
        
        const predicted = logRanks.map(x => slope * x + intercept);
        const ssRes = logFreqs.reduce((sum, y, i) => sum + Math.pow(y - predicted[i], 2), 0);
        const ssTot = logFreqs.reduce((sum, y) => sum + Math.pow(y - meanY, 2), 0);
        
        return ssTot > 0 ? 1 - ssRes / ssTot : 0;
    };

    E.mandelbrotFit = function(words) {
        // Simplified Mandelbrot-Zipf fit
        return { slope: E.calculateZipfSlope(words), rSquared: E.calculateZipfRSquared(words) };
    };

    E.wordEntropy = function(words) {
        if (words.length === 0) return 0;
        const freq = {};
        words.forEach(w => freq[w.toLowerCase()] = (freq[w.toLowerCase()] || 0) + 1);
        const probs = Object.values(freq).map(f => f / words.length);
        return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    };

    E.characterEntropy = function(text) {
        const chars = text.toLowerCase().replace(/\s/g, '');
        if (chars.length === 0) return 0;
        const freq = {};
        for (const c of chars) freq[c] = (freq[c] || 0) + 1;
        const probs = Object.values(freq).map(f => f / chars.length);
        return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    };

    E.bigramEntropy = function(words) {
        if (words.length < 2) return 0;
        const bigrams = [];
        for (let i = 0; i < words.length - 1; i++) {
            bigrams.push(words[i].toLowerCase() + ' ' + words[i+1].toLowerCase());
        }
        const freq = {};
        bigrams.forEach(b => freq[b] = (freq[b] || 0) + 1);
        const probs = Object.values(freq).map(f => f / bigrams.length);
        return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    };

    E.sentenceLengthEntropy = function(sentences) {
        if (sentences.length === 0) return 0;
        const lengths = sentences.map(s => (s.match(/\b\w+/g) || []).length);
        const freq = {};
        lengths.forEach(l => freq[l] = (freq[l] || 0) + 1);
        const probs = Object.values(freq).map(f => f / lengths.length);
        return -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
    };

    E.conditionalEntropy = function(words) {
        // H(X|Y) approximation using bigrams
        if (words.length < 2) return 0;
        const bigramEntropy = E.bigramEntropy(words);
        const wordEntropy = E.wordEntropy(words);
        return Math.max(0, bigramEntropy - wordEntropy);
    };

    E.estimateCrossEntropy = function(words) {
        // Simplified cross-entropy estimate
        return E.wordEntropy(words) * 1.1; // Approximate
    };

    E.wordBurstiness = function(words) {
        if (words.length < 2) return 0;
        const freq = {};
        words.forEach((w, i) => {
            const lower = w.toLowerCase();
            if (!freq[lower]) freq[lower] = [];
            freq[lower].push(i);
        });
        
        let totalBurst = 0;
        let count = 0;
        for (const positions of Object.values(freq)) {
            if (positions.length >= 2) {
                const gaps = [];
                for (let i = 1; i < positions.length; i++) {
                    gaps.push(positions[i] - positions[i-1]);
                }
                const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
                const variance = gaps.reduce((sum, g) => sum + Math.pow(g - mean, 2), 0) / gaps.length;
                const std = Math.sqrt(variance);
                if (mean > 0) {
                    totalBurst += (std - mean) / (std + mean);
                    count++;
                }
            }
        }
        return count > 0 ? totalBurst / count : 0;
    };

    E.topicBurstiness = function(text) {
        // Simplified - measure keyword clustering
        const paragraphs = text.split(/\n\s*\n/);
        if (paragraphs.length < 2) return 0;
        const keywordDensities = paragraphs.map(p => {
            const words = p.toLowerCase().match(/\b[a-z]{5,}\b/g) || [];
            return words.length / Math.max(p.length, 1);
        });
        const mean = keywordDensities.reduce((a, b) => a + b, 0) / keywordDensities.length;
        const variance = keywordDensities.reduce((sum, d) => sum + Math.pow(d - mean, 2), 0) / keywordDensities.length;
        return Math.sqrt(variance) / Math.max(mean, 0.001);
    };

    E.keywordClustering = function(words) {
        // Measure how clustered content words are
        const contentIndices = [];
        words.forEach((w, i) => {
            if (!E.FUNCTION_WORDS.has(w.toLowerCase()) && w.length > 3) {
                contentIndices.push(i);
            }
        });
        if (contentIndices.length < 2) return 0;
        const gaps = [];
        for (let i = 1; i < contentIndices.length; i++) {
            gaps.push(contentIndices[i] - contentIndices[i-1]);
        }
        const mean = gaps.reduce((a, b) => a + b, 0) / gaps.length;
        const variance = gaps.reduce((sum, g) => sum + Math.pow(g - mean, 2), 0) / gaps.length;
        return Math.sqrt(variance);
    };

    E.interArrivalTimeAnalysis = function(words) {
        // Analysis of word repeat intervals
        const arrivals = {};
        words.forEach((w, i) => {
            const lower = w.toLowerCase();
            if (!arrivals[lower]) arrivals[lower] = [];
            arrivals[lower].push(i);
        });
        const intervals = [];
        for (const positions of Object.values(arrivals)) {
            if (positions.length >= 2) {
                for (let i = 1; i < positions.length; i++) {
                    intervals.push(positions[i] - positions[i-1]);
                }
            }
        }
        if (intervals.length === 0) return { mean: 0, std: 0 };
        const mean = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        const std = Math.sqrt(intervals.reduce((sum, i) => sum + Math.pow(i - mean, 2), 0) / intervals.length);
        return { mean, std };
    };

    E.wordLengthDistributionFit = function(words) {
        const lengths = words.map(w => w.length);
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const std = Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length);
        return { mean, std, skewness: E.calculateSkewness(lengths) };
    };

    E.calculateSkewness = function(values) {
        if (values.length < 3) return 0;
        const mean = values.reduce((a, b) => a + b, 0) / values.length;
        const std = Math.sqrt(values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length);
        if (std === 0) return 0;
        const n = values.length;
        return values.reduce((sum, v) => sum + Math.pow((v - mean) / std, 3), 0) * n / ((n - 1) * (n - 2));
    };

    E.sentenceLengthDistributionFit = function(sentences) {
        const lengths = sentences.map(s => (s.match(/\b\w+/g) || []).length);
        if (lengths.length === 0) return { mean: 0, std: 0, skewness: 0 };
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const std = Math.sqrt(lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length);
        return { mean, std, skewness: E.calculateSkewness(lengths) };
    };

    E.vocabularyGrowthCurve = function(words) {
        // Track unique words as text progresses
        const seen = new Set();
        const growth = [];
        words.forEach((w, i) => {
            seen.add(w.toLowerCase());
            if ((i + 1) % 50 === 0 || i === words.length - 1) {
                growth.push({ position: i + 1, unique: seen.size });
            }
        });
        return growth;
    };

    E.heapsLawFit = function(words) {
        // Heaps' law: V = K * N^β
        const growth = E.vocabularyGrowthCurve(words);
        if (growth.length < 2) return { K: 0, beta: 0 };
        
        const logN = growth.map(g => Math.log(g.position));
        const logV = growth.map(g => Math.log(g.unique));
        
        const n = logN.length;
        const sumX = logN.reduce((a, b) => a + b, 0);
        const sumY = logV.reduce((a, b) => a + b, 0);
        const sumXY = logN.reduce((sum, x, i) => sum + x * logV[i], 0);
        const sumX2 = logN.reduce((sum, x) => sum + x * x, 0);
        
        const beta = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const logK = (sumY - beta * sumX) / n;
        
        return { K: Math.exp(logK), beta };
    };

    E.markovTransitionAnalysis = function(words) {
        // First-order Markov transition probabilities
        if (words.length < 2) return {};
        const transitions = {};
        for (let i = 0; i < words.length - 1; i++) {
            const curr = words[i].toLowerCase();
            const next = words[i+1].toLowerCase();
            if (!transitions[curr]) transitions[curr] = {};
            transitions[curr][next] = (transitions[curr][next] || 0) + 1;
        }
        // Calculate entropy of transitions
        let totalEntropy = 0;
        let count = 0;
        for (const from in transitions) {
            const total = Object.values(transitions[from]).reduce((a, b) => a + b, 0);
            const probs = Object.values(transitions[from]).map(c => c / total);
            const entropy = -probs.reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0);
            totalEntropy += entropy;
            count++;
        }
        return { averageTransitionEntropy: count > 0 ? totalEntropy / count : 0 };
    };

    E.autocorrelation = function(words) {
        // Word length autocorrelation
        const lengths = words.map(w => w.length);
        if (lengths.length < 10) return 0;
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const variance = lengths.reduce((sum, l) => sum + Math.pow(l - mean, 2), 0) / lengths.length;
        if (variance === 0) return 0;
        
        let autocorr = 0;
        for (let i = 0; i < lengths.length - 1; i++) {
            autocorr += (lengths[i] - mean) * (lengths[i+1] - mean);
        }
        return autocorr / ((lengths.length - 1) * variance);
    };

    E.longRangeDependence = function(words) {
        // Simplified long-range dependence measure
        return E.hurstExponent(words);
    };

    E.hurstExponent = function(words) {
        // Simplified Hurst exponent estimation using R/S analysis
        const lengths = words.map(w => w.length);
        if (lengths.length < 20) return 0.5;
        
        const mean = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const deviations = lengths.map(l => l - mean);
        const cumulative = [];
        let sum = 0;
        deviations.forEach(d => {
            sum += d;
            cumulative.push(sum);
        });
        
        const range = Math.max(...cumulative) - Math.min(...cumulative);
        const std = Math.sqrt(lengths.reduce((s, l) => s + Math.pow(l - mean, 2), 0) / lengths.length);
        
        if (std === 0) return 0.5;
        const rs = range / std;
        
        // H ≈ log(R/S) / log(n)
        return Math.log(rs) / Math.log(lengths.length);
    };

    // Export
    if (typeof window !== 'undefined') {
        window.ExtendedHeuristicsAnalyzer = E;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = E;
    }
})();
