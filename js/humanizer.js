/**
 * VERITAS — Text Humanizer
 * Client-side humanization for testing AI detection
 */

const Humanizer = {
    // Contraction mappings
    contractions: {
        'do not': "don't",
        'does not': "doesn't",
        'did not': "didn't",
        'is not': "isn't",
        'are not': "aren't",
        'was not': "wasn't",
        'were not': "weren't",
        'has not': "hasn't",
        'have not': "haven't",
        'had not': "hadn't",
        'will not': "won't",
        'would not': "wouldn't",
        'could not': "couldn't",
        'should not': "shouldn't",
        'cannot': "can't",
        'can not': "can't",
        'it is': "it's",
        'it has': "it's",
        'that is': "that's",
        'there is': "there's",
        'here is': "here's",
        'what is': "what's",
        'who is': "who's",
        'I am': "I'm",
        'I have': "I've",
        'I will': "I'll",
        'I would': "I'd",
        'you are': "you're",
        'you have': "you've",
        'you will': "you'll",
        'we are': "we're",
        'we have': "we've",
        'they are': "they're",
        'they have': "they've",
        'let us': "let's",
        'going to': 'gonna',
        'want to': 'wanna',
        'got to': 'gotta',
    },

    // Disfluencies by position
    disfluencies: {
        start: ['Well, ', 'So, ', 'I mean, ', 'Basically, ', 'Honestly, ', 'Look, ', 'Okay, ', 'Actually, '],
        mid: [', you know,', ', like,', ', basically,', ', honestly,', ', I think,'],
        end: [', right?', ', you know?', ', I guess.', '.']
    },

    // AI phrases to remove or replace
    aiPhrases: {
        'it is important to note that': '',
        'it is worth noting that': '',
        'it is essential to': 'we need to',
        'furthermore': 'also',
        'moreover': 'plus',
        'additionally': 'also',
        'in conclusion': 'so basically',
        'to summarize': 'in short',
        'this demonstrates': 'this shows',
        'this illustrates': 'this shows',
        'plays a pivotal role': 'is really important',
        'plays a crucial role': 'matters a lot',
        'a myriad of': 'lots of',
        'a plethora of': 'tons of',
        'a multitude of': 'many',
        'comprehensive understanding': 'good understanding',
        'in the realm of': 'in',
        'in the context of': 'with',
        'subsequently': 'then',
        'consequently': 'so',
        'utilize': 'use',
        'facilitate': 'help',
        'implement': 'do',
        'demonstrate': 'show',
        'indicate': 'show',
        'leverage': 'use',
        'optimize': 'improve',
        'paradigm': 'way',
        'synergy': 'teamwork',
    },

    // Hedging phrases to add
    hedgingPhrases: [
        'I think ',
        'probably ',
        'maybe ',
        'it seems like ',
        'in my opinion, ',
        'from what I can tell, ',
    ],

    // Statistics tracking
    stats: {
        contractions: 0,
        disfluencies: 0,
        phraseRemovals: 0,
        hedging: 0,
        sentenceVariation: 0
    },

    /**
     * Main humanization function
     */
    humanize(text, intensity = 'medium', style = 'random') {
        this.resetStats();
        
        if (!text || text.trim().length === 0) {
            return { text: '', stats: this.stats };
        }

        let result = text;

        // Apply transformations based on intensity
        const config = this.getIntensityConfig(intensity);

        // Step 1: Remove/replace AI phrases
        result = this.removeAIPhrases(result, config.phraseRemovalRate);

        // Step 2: Add contractions
        result = this.addContractions(result, config.contractionRate);

        // Step 3: Add disfluencies
        result = this.addDisfluencies(result, config.disfluencyRate);

        // Step 4: Add hedging
        result = this.addHedging(result, config.hedgingRate);

        // Step 5: Vary sentence structure
        result = this.varySentenceStructure(result, config.variationRate);

        // Step 6: Add first-person references if style calls for it
        if (style === 'casual' || style === 'random') {
            result = this.addPersonalTouch(result, config.personalRate);
        }

        return { text: result, stats: { ...this.stats } };
    },

    getIntensityConfig(intensity) {
        const configs = {
            light: {
                contractionRate: 0.3,
                disfluencyRate: 0.1,
                phraseRemovalRate: 0.5,
                hedgingRate: 0.1,
                variationRate: 0.1,
                personalRate: 0.1
            },
            medium: {
                contractionRate: 0.6,
                disfluencyRate: 0.2,
                phraseRemovalRate: 0.8,
                hedgingRate: 0.2,
                variationRate: 0.2,
                personalRate: 0.2
            },
            heavy: {
                contractionRate: 0.9,
                disfluencyRate: 0.4,
                phraseRemovalRate: 1.0,
                hedgingRate: 0.3,
                variationRate: 0.3,
                personalRate: 0.3
            },
            stealth: {
                contractionRate: 0.5,
                disfluencyRate: 0.15,
                phraseRemovalRate: 1.0,
                hedgingRate: 0.15,
                variationRate: 0.25,
                personalRate: 0.15
            }
        };
        return configs[intensity] || configs.medium;
    },

    resetStats() {
        this.stats = {
            contractions: 0,
            disfluencies: 0,
            phraseRemovals: 0,
            hedging: 0,
            sentenceVariation: 0
        };
    },

    removeAIPhrases(text, rate) {
        let result = text;
        for (const [phrase, replacement] of Object.entries(this.aiPhrases)) {
            const regex = new RegExp(phrase, 'gi');
            const matches = result.match(regex);
            if (matches && Math.random() < rate) {
                result = result.replace(regex, replacement);
                this.stats.phraseRemovals += matches.length;
            }
        }
        return result;
    },

    addContractions(text, rate) {
        let result = text;
        for (const [full, contracted] of Object.entries(this.contractions)) {
            if (Math.random() < rate) {
                const regex = new RegExp('\\b' + full + '\\b', 'gi');
                const matches = result.match(regex);
                if (matches) {
                    result = result.replace(regex, contracted);
                    this.stats.contractions += matches.length;
                }
            }
        }
        return result;
    },

    addDisfluencies(text, rate) {
        const sentences = text.split(/(?<=[.!?])\s+/);
        const result = sentences.map((sentence, index) => {
            if (Math.random() > rate) return sentence;
            
            // Add at start of some sentences
            if (index > 0 && Math.random() < 0.3) {
                const disfluency = this.disfluencies.start[Math.floor(Math.random() * this.disfluencies.start.length)];
                sentence = disfluency + sentence.charAt(0).toLowerCase() + sentence.slice(1);
                this.stats.disfluencies++;
            }
            
            return sentence;
        });
        
        return result.join(' ');
    },

    addHedging(text, rate) {
        const sentences = text.split(/(?<=[.!?])\s+/);
        const result = sentences.map((sentence, index) => {
            if (Math.random() > rate || index === 0) return sentence;
            
            // Add hedging to statements
            if (!sentence.includes('?') && Math.random() < 0.3) {
                const hedging = this.hedgingPhrases[Math.floor(Math.random() * this.hedgingPhrases.length)];
                sentence = hedging + sentence.charAt(0).toLowerCase() + sentence.slice(1);
                this.stats.hedging++;
            }
            
            return sentence;
        });
        
        return result.join(' ');
    },

    varySentenceStructure(text, rate) {
        const sentences = text.split(/(?<=[.!?])\s+/);
        
        const result = sentences.map(sentence => {
            if (Math.random() > rate) return sentence;
            
            // Occasionally split long sentences
            if (sentence.length > 100 && sentence.includes(',')) {
                const parts = sentence.split(/,\s*/);
                if (parts.length >= 2) {
                    const splitPoint = Math.floor(parts.length / 2);
                    const first = parts.slice(0, splitPoint).join(', ');
                    const second = parts.slice(splitPoint).join(', ');
                    this.stats.sentenceVariation++;
                    return first + '. ' + second.charAt(0).toUpperCase() + second.slice(1);
                }
            }
            
            return sentence;
        });
        
        return result.join(' ');
    },

    addPersonalTouch(text, rate) {
        const sentences = text.split(/(?<=[.!?])\s+/);
        const personalPhrases = [
            'In my experience, ',
            'From what I\'ve seen, ',
            'I\'ve noticed that ',
            'Personally, I find that ',
        ];
        
        const result = sentences.map((sentence, index) => {
            if (index > 0 && Math.random() < rate && !sentence.toLowerCase().startsWith('i ')) {
                const phrase = personalPhrases[Math.floor(Math.random() * personalPhrases.length)];
                return phrase + sentence.charAt(0).toLowerCase() + sentence.slice(1);
            }
            return sentence;
        });
        
        return result.join(' ');
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Humanizer;
}
