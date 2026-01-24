/**
 * VERITAS — Extended Heuristics Implementation Part 4
 * ====================================================
 * Domain-Specific, Temporal/Cultural, Discourse, and Statistical implementations
 */

(function() {
    const E = typeof ExtendedHeuristicsAnalyzer !== 'undefined' ? ExtendedHeuristicsAnalyzer : 
              (typeof window !== 'undefined' ? window.ExtendedHeuristicsAnalyzer : {});

    // ========================================================================
    // DOMAIN-SPECIFIC IMPLEMENTATIONS
    // ========================================================================

    E.detectCitations = function(text) {
        const patterns = [
            /\([A-Z][a-z]+,?\s+\d{4}\)/g,  // (Author, 2024)
            /\([A-Z][a-z]+\s+et\s+al\.?,?\s+\d{4}\)/gi,  // (Author et al., 2024)
            /\[\d+\]/g,  // [1]
            /\[\d+,\s*\d+\]/g,  // [1, 2]
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectCitationStyle = function(text) {
        if (/\([A-Z][a-z]+,?\s+\d{4}\)/.test(text)) return 'APA/Harvard';
        if (/\[\d+\]/.test(text)) return 'IEEE/Numbered';
        if (/\d+\.\s+[A-Z][a-z]+,/.test(text)) return 'MLA';
        if (/[A-Z][a-z]+\s+\(\d{4}\)/.test(text)) return 'Chicago';
        return 'none';
    };

    E.academicVocabularyRatio = function(text) {
        const words = text.toLowerCase().match(/\b[a-z]+/g) || [];
        const academic = words.filter(w => E.ACADEMIC_VOCABULARY.includes(w));
        return academic.length / Math.max(words.length, 1);
    };

    E.hedgingLanguageCount = function(text) {
        const hedges = ['may', 'might', 'could', 'would', 'seem', 'appear', 'suggest', 'indicate', 'possible', 'likely', 'perhaps', 'probably', 'presumably', 'apparently', 'arguably', 'relatively', 'somewhat', 'tends to', 'appears to'];
        let count = 0;
        hedges.forEach(h => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + h.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.methodologyMarkers = function(text) {
        const markers = ['method', 'methodology', 'approach', 'procedure', 'technique', 'sample', 'participants', 'subjects', 'data', 'analysis', 'results', 'findings', 'hypothesis', 'variable', 'measure', 'instrument', 'questionnaire', 'survey', 'experiment', 'study'];
        let count = 0;
        markers.forEach(m => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + m + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.researchTermCount = function(text) {
        const terms = ['research', 'study', 'investigation', 'examination', 'exploration', 'inquiry', 'analysis', 'review', 'literature', 'theoretical', 'empirical', 'qualitative', 'quantitative', 'correlation', 'regression', 'significance', 'p-value', 'hypothesis', 'null', 'alternative'];
        let count = 0;
        terms.forEach(t => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + t + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.formalRegisterScore = function(text) {
        const formal = ['therefore', 'consequently', 'furthermore', 'moreover', 'nevertheless', 'notwithstanding', 'henceforth', 'whereby', 'thereof', 'herein', 'pursuant'];
        const informal = ['gonna', 'wanna', 'kinda', 'stuff', 'things', 'yeah', 'nope', 'cool', 'awesome', 'basically'];
        let formalCount = 0, informalCount = 0;
        formal.forEach(f => {
            if (new RegExp('\\b' + f + '\\b', 'i').test(text)) formalCount++;
        });
        informal.forEach(i => {
            if (new RegExp('\\b' + i + '\\b', 'i').test(text)) informalCount++;
        });
        if (formalCount + informalCount === 0) return 0.5;
        return formalCount / (formalCount + informalCount);
    };

    E.technicalTermDensity = function(text) {
        const words = text.match(/\b\w+/g) || [];
        // Heuristic: words with specific patterns
        const technical = words.filter(w => /[A-Z]{2,}|\d+/.test(w) || w.length > 10);
        return technical.length / Math.max(words.length, 1);
    };

    E.acronymUsage = function(text) {
        const acronyms = text.match(/\b[A-Z]{2,}\b/g) || [];
        return { count: acronyms.length, examples: [...new Set(acronyms)].slice(0, 10) };
    };

    E.detectCodePresence = function(text) {
        return {
            codeBlocks: (text.match(/```[\s\S]*?```/g) || []).length,
            inlineCode: (text.match(/`[^`]+`/g) || []).length,
            codeKeywords: (text.match(/\b(function|class|def|var|let|const|return|if|else|for|while|import|export|require)\b/g) || []).length
        };
    };

    E.numericalDataDensity = function(text) {
        const numbers = (text.match(/\b\d+\.?\d*\b/g) || []).length;
        const words = (text.match(/\b\w+/g) || []).length;
        return numbers / Math.max(words, 1);
    };

    E.unitMentionCount = function(text) {
        const units = (text.match(/\b\d+\.?\d*\s*(kg|g|mg|lb|oz|km|m|cm|mm|mi|ft|in|L|mL|gal|°C|°F|K|Hz|kHz|MHz|GHz|W|kW|MW|V|A|Ω|bit|byte|KB|MB|GB|TB|%)\b/gi) || []).length;
        return units;
    };

    E.specificationLanguage = function(text) {
        const specs = ['must', 'shall', 'should', 'required', 'mandatory', 'optional', 'recommended', 'specification', 'requirement', 'constraint', 'parameter', 'configuration', 'setting', 'option', 'default'];
        let count = 0;
        specs.forEach(s => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + s + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectMetaphors = function(text) {
        // Heuristic: "is a" patterns with abstract nouns
        const patterns = [
            /\b\w+\s+is\s+a\s+\w+\b/gi,
            /\blike\s+a\s+\w+/gi,
            /\bsea\s+of\b|\bocean\s+of\b|\bforest\s+of\b|\bmountain\s+of\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.detectSimiles = function(text) {
        return (text.match(/\b(like|as)\s+a\s+\w+/gi) || []).length;
    };

    E.detectPersonification = function(text) {
        const patterns = [
            /\b(sun|moon|wind|time|death|love|nature|earth)\s+(said|whispered|cried|laughed|danced|sang|smiled)/gi,
            /\bthe\s+\w+\s+(embraced|kissed|caressed|touched)\b/gi
        ];
        let count = 0;
        patterns.forEach(p => {
            const matches = text.match(p);
            if (matches) count += matches.length;
        });
        return count;
    };

    E.imageryDensity = function(text) {
        const sensory = ['see', 'saw', 'look', 'watch', 'hear', 'heard', 'listen', 'sound', 'smell', 'scent', 'taste', 'feel', 'felt', 'touch', 'warm', 'cold', 'hot', 'soft', 'hard', 'rough', 'smooth', 'bright', 'dark', 'loud', 'quiet', 'sweet', 'sour', 'bitter'];
        let count = 0;
        sensory.forEach(s => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + s + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        const words = (text.match(/\b\w+/g) || []).length;
        return count / Math.max(words, 1);
    };

    E.detectDialogue = function(text) {
        const quotedSpeech = (text.match(/"[^"]+"\s*(said|asked|replied|shouted|whispered)/gi) || []).length;
        const dialogueTags = (text.match(/\b(said|asked|replied|responded|exclaimed|muttered|whispered|shouted)\s+(he|she|they|I)\b/gi) || []).length;
        return { quotedSpeech, dialogueTags, total: quotedSpeech + dialogueTags };
    };

    E.narrativeMarkers = function(text) {
        const markers = ['once upon a time', 'long ago', 'one day', 'suddenly', 'meanwhile', 'later', 'finally', 'in the end', 'happily ever after', 'the end'];
        let count = 0;
        markers.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.descriptiveLanguageRatio = function(text) {
        const adjectives = (text.match(/\b\w+(ful|less|ous|ive|able|ible|al|ial|ic|ical|ish|like|ly)\b/gi) || []).length;
        const words = (text.match(/\b\w+/g) || []).length;
        return adjectives / Math.max(words, 1);
    };

    E.businessJargonCount = function(text) {
        const jargon = ['synergy', 'leverage', 'optimize', 'streamline', 'stakeholder', 'deliverable', 'actionable', 'bandwidth', 'circle back', 'touch base', 'moving forward', 'best practice', 'value add', 'game changer', 'paradigm shift', 'low-hanging fruit', 'win-win', 'think outside the box', 'deep dive', 'drill down'];
        let count = 0;
        jargon.forEach(j => {
            if (text.toLowerCase().includes(j)) count++;
        });
        return count;
    };

    E.detectFormalSalutation = function(text) {
        const salutations = ['dear sir', 'dear madam', 'dear mr', 'dear mrs', 'dear ms', 'dear dr', 'to whom it may concern', 'dear hiring manager', 'dear team'];
        for (const s of salutations) {
            if (text.toLowerCase().includes(s)) return true;
        }
        return false;
    };

    E.detectProfessionalClosing = function(text) {
        const closings = ['sincerely', 'best regards', 'kind regards', 'yours truly', 'respectfully', 'best wishes', 'thank you', 'warm regards'];
        for (const c of closings) {
            if (text.toLowerCase().includes(c)) return true;
        }
        return false;
    };

    E.actionableLanguageCount = function(text) {
        const actionable = ['please', 'kindly', 'would you', 'could you', 'can you', 'need to', 'must', 'should', 'require', 'request', 'ensure', 'confirm', 'verify', 'review', 'approve', 'submit', 'complete', 'finalize'];
        let count = 0;
        actionable.forEach(a => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + a.replace(/ /g, '\\s+') + '\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.stakeholderReferences = function(text) {
        const stakeholders = ['client', 'customer', 'user', 'stakeholder', 'partner', 'vendor', 'supplier', 'team', 'management', 'leadership', 'board', 'executive', 'employee', 'staff', 'colleague'];
        let count = 0;
        stakeholders.forEach(s => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + s + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.hashtagCount = function(text) {
        return (text.match(/#\w+/g) || []).length;
    };

    E.mentionCount = function(text) {
        return (text.match(/@\w+/g) || []).length;
    };

    E.emojiCount = function(text) {
        const emojiRegex = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu;
        return (text.match(emojiRegex) || []).length;
    };

    E.abbreviationCount = function(text) {
        const abbrevs = ['lol', 'lmao', 'rofl', 'btw', 'omg', 'idk', 'imo', 'imho', 'tbh', 'tho', 'rn', 'af', 'ngl', 'iirc', 'afaik', 'fwiw', 'tl;dr', 'asap', 'fyi', 'brb', 'gtg', 'ttyl'];
        let count = 0;
        abbrevs.forEach(a => {
            if (new RegExp('\\b' + a + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.slangUsage = function(text) {
        let count = 0;
        E.SLANG_WORDS.forEach(s => {
            if (new RegExp('\\b' + s + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.internetisms = function(text) {
        const patterns = ['lol', 'omg', 'wtf', 'smh', 'tbh', 'ngl', 'imo', 'icymi', 'tfw', 'mfw', 'irl', 'afk', 'dm', 'pm'];
        let count = 0;
        patterns.forEach(p => {
            if (new RegExp('\\b' + p + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.legalTermCount = function(text) {
        const legal = ['hereby', 'whereas', 'hereinafter', 'aforesaid', 'aforementioned', 'notwithstanding', 'pursuant', 'thereof', 'herein', 'thereto', 'hereunder', 'plaintiff', 'defendant', 'jurisdiction', 'litigation', 'arbitration', 'indemnify', 'liability', 'negligence', 'breach', 'damages', 'injunction', 'stipulation', 'deposition'];
        let count = 0;
        legal.forEach(l => {
            if (new RegExp('\\b' + l + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.latinPhraseCount = function(text) {
        const latin = ['et al', 'etc', 'i.e.', 'e.g.', 'ad hoc', 'de facto', 'per se', 'vice versa', 'pro bono', 'bona fide', 'prima facie', 'status quo', 'quid pro quo', 'modus operandi', 'habeas corpus', 'ipso facto'];
        let count = 0;
        latin.forEach(l => {
            if (text.toLowerCase().includes(l)) count++;
        });
        return count;
    };

    E.formalDefinitionCount = function(text) {
        return (text.match(/\b(means|refers to|is defined as|shall mean|constitutes)\b/gi) || []).length;
    };

    E.conditionalClauseCount = function(text) {
        return (text.match(/\b(if|unless|provided that|in the event that|subject to|notwithstanding)\b/gi) || []).length;
    };

    E.disclaimerLanguage = function(text) {
        const disclaimers = ['disclaimer', 'liability', 'not responsible', 'no warranty', 'as is', 'without warranty', 'limitation of liability', 'indemnification', 'hold harmless'];
        let count = 0;
        disclaimers.forEach(d => {
            if (text.toLowerCase().includes(d)) count++;
        });
        return count;
    };

    E.scientificTermCount = function(text) {
        const scientific = ['hypothesis', 'theory', 'experiment', 'observation', 'variable', 'control', 'specimen', 'sample', 'analysis', 'synthesis', 'compound', 'molecule', 'atom', 'cell', 'organism', 'species', 'genus', 'phylum', 'ecosystem', 'entropy', 'catalyst', 'reaction', 'solution', 'concentration', 'mass', 'volume', 'density', 'velocity', 'acceleration', 'force', 'energy', 'wavelength', 'frequency'];
        let count = 0;
        scientific.forEach(s => {
            const matches = text.toLowerCase().match(new RegExp('\\b' + s + '\\w*\\b', 'g'));
            if (matches) count += matches.length;
        });
        return count;
    };

    E.measurementMentions = function(text) {
        return (text.match(/\b\d+\.?\d*\s*(ml|L|mg|g|kg|mm|cm|m|km|°C|°F|K|pH|mol|M|mM|μM|nM|Hz|kHz|MHz|GHz|Pa|kPa|atm|psi)\b/gi) || []).length;
    };

    E.processDescriptions = function(text) {
        const process = ['first', 'second', 'third', 'then', 'next', 'finally', 'step', 'stage', 'phase', 'procedure', 'process', 'method', 'technique'];
        let count = 0;
        process.forEach(p => {
            if (new RegExp('\\b' + p + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.cautionaryLanguage = function(text) {
        const caution = ['caution', 'warning', 'danger', 'hazard', 'risk', 'avoid', 'do not', 'never', 'careful', 'precaution', 'safety', 'protective'];
        let count = 0;
        caution.forEach(c => {
            if (new RegExp('\\b' + c + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.evidenceMarkers = function(text) {
        const evidence = ['according to', 'research shows', 'studies indicate', 'evidence suggests', 'data shows', 'findings reveal', 'results demonstrate', 'statistics show', 'experts say', 'scientists found'];
        let count = 0;
        evidence.forEach(e => {
            if (text.toLowerCase().includes(e)) count++;
        });
        return count;
    };

    // Model UN / Debate / Speech markers
    E.delegateReferences = function(text) {
        return (text.match(/\b(delegate|delegates|delegation|ambassador|representative|distinguished|honorable|chair|committee)\b/gi) || []).length;
    };

    E.countryReferences = function(text) {
        const countries = ['united states', 'usa', 'china', 'russia', 'india', 'brazil', 'germany', 'france', 'uk', 'japan', 'canada', 'australia', 'mexico', 'italy', 'spain', 'south korea', 'indonesia', 'turkey', 'saudi arabia', 'argentina', 'united nations', 'un', 'nato', 'eu', 'european union', 'african union', 'asean'];
        let count = 0;
        countries.forEach(c => {
            if (text.toLowerCase().includes(c)) count++;
        });
        return count;
    };

    E.proposalLanguage = function(text) {
        const proposals = ['proposes', 'propose', 'proposal', 'resolution', 'recommend', 'recommends', 'urge', 'urges', 'call upon', 'calls upon', 'encourage', 'encourages', 'support', 'supports', 'endorse', 'endorses'];
        let count = 0;
        proposals.forEach(p => {
            if (new RegExp('\\b' + p + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.resolutionTerms = function(text) {
        const terms = ['resolution', 'clause', 'amendment', 'motion', 'vote', 'consensus', 'ratify', 'implement', 'enforce', 'sanction', 'treaty', 'accord', 'agreement', 'convention', 'protocol'];
        let count = 0;
        terms.forEach(t => {
            if (new RegExp('\\b' + t + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.diplomaticLanguage = function(text) {
        const diplomatic = ['cooperation', 'collaboration', 'dialogue', 'negotiation', 'peaceful', 'mutual', 'bilateral', 'multilateral', 'consensus', 'compromise', 'respect', 'sovereignty', 'territorial', 'humanitarian', 'international'];
        let count = 0;
        diplomatic.forEach(d => {
            if (new RegExp('\\b' + d + '\\b', 'i').test(text)) count++;
        });
        return count;
    };

    E.speechMarkers = function(text) {
        const markers = ['ladies and gentlemen', 'thank you', 'today i', 'we must', 'let us', 'i believe', 'we believe', 'our position', 'in conclusion', 'to conclude', 'in summary'];
        let count = 0;
        markers.forEach(m => {
            if (text.toLowerCase().includes(m)) count++;
        });
        return count;
    };

    E.audienceAddressing = function(text) {
        return (text.match(/\b(you|your|we|our|us|fellow|dear|everyone|all of you)\b/gi) || []).length;
    };

    // Export
    if (typeof window !== 'undefined') {
        window.ExtendedHeuristicsAnalyzer = E;
    }
    if (typeof module !== 'undefined' && module.exports) {
        module.exports = E;
    }
})();
