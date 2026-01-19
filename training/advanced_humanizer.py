#!/usr/bin/env python3
"""
Advanced AI Text Humanizer - Randomized Multi-Technique Approach
=================================================================
Based on research into AI detector evasion techniques.

CRITICAL: Each humanization uses a RANDOM COMBINATION of techniques
with random parameters. This creates diverse training data that
teaches the model to detect ALL types of humanization, not just one.

Technique Categories:
1. Linguistic & Stylistic Variation (burstiness, idioms, questions)
2. Post-Processing (synonyms, paraphrasing, errors)
3. Technical Generation Simulation (perplexity, temperature effects)

Sources: TempParaphraser, MASH, QuillBot analysis, EssayDone patterns
"""

import random
import re
import string
from typing import List, Tuple, Optional, Set
from collections import defaultdict


class AdvancedHumanizer:
    """
    Humanizes AI text using randomized combinations of techniques.
    Each call to humanize() applies a DIFFERENT random subset of methods.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        
        self._init_linguistic_resources()
        self._init_synonym_database()
        self._init_error_patterns()
    
    def _init_linguistic_resources(self):
        """Initialize all linguistic modification resources"""
        
        # ══════════════════════════════════════════════════════════════════
        # AI PHRASES TO REMOVE/REPLACE (detectors look for these)
        # ══════════════════════════════════════════════════════════════════
        self.ai_phrases = {
            # Formal transitions AI overuses
            "Furthermore,": ["Also,", "Plus,", "And", "Besides,", ""],
            "Moreover,": ["Also,", "And", "Plus,", "What's more,", ""],
            "Additionally,": ["Also,", "Plus,", "And", "On top of that,", ""],
            "In conclusion,": ["So,", "Basically,", "To wrap up,", "All in all,", ""],
            "To summarize,": ["So basically,", "Long story short,", "In short,", ""],
            "It is important to note that": ["Thing is,", "Here's the deal:", "Worth noting:", ""],
            "It's worth mentioning that": ["Also,", "Oh, and", "By the way,", ""],
            "In today's world,": ["These days,", "Nowadays,", "Right now,", ""],
            "In the realm of": ["In", "When it comes to", "With", "For"],
            "It is essential to": ["You gotta", "You need to", "It helps to", "Better to"],
            "This demonstrates that": ["This shows", "See?", "Which means", "So basically"],
            "It can be observed that": ["You can see", "Clearly,", "Obviously,", ""],
            "There are several reasons why": ["Here's why:", "A few reasons:", "So,", ""],
            "One must consider": ["Think about", "Consider", "Look at", ""],
            "It is crucial to understand": ["The key thing is", "Important bit:", "Basically,", ""],
            "plays a pivotal role": ["really matters", "is huge", "is key", "makes a big difference"],
            "a myriad of": ["lots of", "tons of", "a bunch of", "many"],
            "utilize": ["use", "work with", "rely on", "go with"],
            "facilitate": ["help", "make easier", "enable", "allow"],
            "implement": ["do", "put in place", "set up", "use"],
            "comprehensive": ["full", "complete", "thorough", "detailed"],
            "subsequently": ["then", "after that", "next", "later"],
            "nevertheless": ["still", "but", "even so", "anyway"],
            "notwithstanding": ["despite", "even with", "regardless of", "still"],
            "henceforth": ["from now on", "going forward", "after this", ""],
            "thereby": ["so", "which", "and", "thus"],
            "wherein": ["where", "in which", "when", ""],
            "whereas": ["while", "but", "when", "though"],
            "insofar as": ["as far as", "to the extent", "inasmuch as", "since"],
        }
        
        # ══════════════════════════════════════════════════════════════════
        # IDIOMS AND COLLOQUIALISMS
        # ══════════════════════════════════════════════════════════════════
        self.idioms = [
            "at the end of the day", "when push comes to shove", "for what it's worth",
            "the thing is", "here's the deal", "long story short", "bottom line",
            "real talk", "no cap", "straight up", "low key", "high key",
            "not gonna lie", "to be honest", "honestly speaking", "truth be told",
            "if you ask me", "in my experience", "from what I've seen",
            "believe it or not", "funny enough", "interestingly enough",
            "here's the kicker", "plot twist", "spoiler alert", "heads up",
            "fair warning", "just saying", "food for thought", "devil's advocate",
            "on the flip side", "that said", "having said that", "all things considered",
            "when all is said and done", "at the risk of sounding", "call me crazy but",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # SPEECH DISFLUENCIES (humans use these, AI avoids them)
        # ══════════════════════════════════════════════════════════════════
        self.disfluencies = [
            "well,", "so,", "I mean,", "you know,", "like,", "basically,",
            "honestly,", "actually,", "look,", "see,", "thing is,", "okay so,",
            "right,", "anyway,", "anyhow,", "um,", "uh,", "hmm,",
            "let me think...", "how do I put this...", "what I mean is,",
            "the way I see it,", "if that makes sense,", "you get me?",
            "know what I mean?", "I guess,", "I suppose,", "I think,",
            "probably,", "maybe,", "sort of,", "kind of,", "more or less,",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # RHETORICAL QUESTIONS
        # ══════════════════════════════════════════════════════════════════
        self.rhetorical_questions = [
            "Right?", "You know?", "Make sense?", "See what I mean?",
            "Get it?", "Follow me?", "Isn't it?", "Don't you think?",
            "Wouldn't you agree?", "Know what I'm saying?", "Fair enough?",
            "Sound good?", "Cool?", "Yeah?", "No?", "Crazy, right?",
            "Wild, huh?", "Interesting, no?", "Funny how that works, right?",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # EMOTIONAL/EMPHATIC EXPRESSIONS
        # ══════════════════════════════════════════════════════════════════
        self.emphatics = [
            "seriously", "literally", "absolutely", "totally", "completely",
            "definitely", "certainly", "obviously", "clearly", "honestly",
            "frankly", "really", "truly", "genuinely", "actually",
            "for real", "no joke", "dead serious", "I swear", "trust me",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # HEDGING LANGUAGE (uncertainty markers humans use)
        # ══════════════════════════════════════════════════════════════════
        self.hedges = [
            "probably", "maybe", "perhaps", "possibly", "likely",
            "I think", "I believe", "I guess", "I suppose", "I imagine",
            "it seems", "apparently", "supposedly", "arguably", "presumably",
            "sort of", "kind of", "more or less", "in a way", "to some extent",
            "might be", "could be", "tends to", "seems to", "appears to",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # CONTRACTIONS (AI often avoids these)
        # ══════════════════════════════════════════════════════════════════
        self.contractions = {
            "I am": "I'm", "I have": "I've", "I will": "I'll", "I would": "I'd",
            "I had": "I'd", "you are": "you're", "you have": "you've",
            "you will": "you'll", "you would": "you'd", "he is": "he's",
            "he has": "he's", "he will": "he'll", "he would": "he'd",
            "she is": "she's", "she has": "she's", "she will": "she'll",
            "she would": "she'd", "it is": "it's", "it has": "it's",
            "it will": "it'll", "it would": "it'd", "we are": "we're",
            "we have": "we've", "we will": "we'll", "we would": "we'd",
            "they are": "they're", "they have": "they've", "they will": "they'll",
            "they would": "they'd", "that is": "that's", "that has": "that's",
            "that will": "that'll", "that would": "that'd", "who is": "who's",
            "who has": "who's", "who will": "who'll", "who would": "who'd",
            "what is": "what's", "what has": "what's", "what will": "what'll",
            "where is": "where's", "where has": "where's", "when is": "when's",
            "why is": "why's", "how is": "how's", "here is": "here's",
            "there is": "there's", "there has": "there's", "there will": "there'll",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't",
            "were not": "weren't", "has not": "hasn't", "have not": "haven't",
            "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "can not": "can't", "cannot": "can't", "could not": "couldn't",
            "should not": "shouldn't", "might not": "mightn't",
            "must not": "mustn't", "need not": "needn't", "ought not": "oughtn't",
            "let us": "let's", "that would": "that'd", "it would": "it'd",
            "could have": "could've", "would have": "would've",
            "should have": "should've", "might have": "might've",
            "must have": "must've", "going to": "gonna", "want to": "wanna",
            "got to": "gotta", "kind of": "kinda", "sort of": "sorta",
            "out of": "outta", "because": "'cause", "about": "'bout",
        }
        
        # ══════════════════════════════════════════════════════════════════
        # SENTENCE FRAGMENTS (natural in human speech/writing)
        # ══════════════════════════════════════════════════════════════════
        self.fragments = [
            "Crazy.", "Wild.", "Insane.", "Nuts.", "Ridiculous.",
            "Absolutely.", "Definitely.", "Obviously.", "Clearly.", "Exactly.",
            "For sure.", "No doubt.", "Big time.", "Huge.", "Massive.",
            "Not even close.", "Not really.", "Not quite.", "Sort of.", "Kind of.",
            "Makes sense.", "Fair point.", "Good question.", "Hard to say.",
            "Tough call.", "Long story.", "True story.", "Real talk.",
            "Bottom line.", "End of story.", "Case closed.", "Done deal.",
            "My bad.", "Their loss.", "Your call.", "No way.", "Hell no.",
            "For real though.", "Seriously though.", "Honestly though.",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # ANECDOTE/PERSONAL STARTERS
        # ══════════════════════════════════════════════════════════════════
        self.anecdote_starters = [
            "I remember when", "One time,", "Back in the day,", "Years ago,",
            "A friend of mine", "Someone I know", "I once", "There was this time",
            "Funny story:", "True story:", "Get this:", "So check it out:",
            "Here's the thing:", "Real quick:", "Quick example:", "Case in point:",
            "Speaking from experience,", "In my experience,", "From what I've seen,",
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # CULTURAL REFERENCES AND EXPRESSIONS
        # ══════════════════════════════════════════════════════════════════
        self.cultural_refs = [
            "it's like that meme where", "kind of like how", "similar to when",
            "reminds me of", "it's giving", "the vibes are", "major energy",
            "not the vibe", "hits different", "just hits", "slaps",
            "that's a whole mood", "big mood", "same energy as",
        ]
    
    def _init_synonym_database(self):
        """Initialize synonym mappings for word substitution"""
        self.synonyms = {
            # Common AI-overused words → more casual alternatives
            "significant": ["big", "major", "huge", "important", "notable", "key"],
            "important": ["big", "key", "major", "crucial", "vital", "essential"],
            "various": ["different", "lots of", "many", "several", "a bunch of"],
            "numerous": ["lots of", "many", "tons of", "a bunch of", "plenty of"],
            "demonstrate": ["show", "prove", "display", "reveal", "make clear"],
            "indicate": ["show", "suggest", "point to", "hint at", "mean"],
            "require": ["need", "call for", "demand", "take", "want"],
            "provide": ["give", "offer", "supply", "deliver", "bring"],
            "obtain": ["get", "grab", "score", "pick up", "land"],
            "sufficient": ["enough", "plenty", "adequate", "ample"],
            "approximately": ["about", "around", "roughly", "like", "or so"],
            "primarily": ["mainly", "mostly", "largely", "chiefly", "for the most part"],
            "frequently": ["often", "a lot", "regularly", "commonly", "usually"],
            "occasionally": ["sometimes", "now and then", "once in a while", "every so often"],
            "immediately": ["right away", "at once", "instantly", "straight away", "ASAP"],
            "extremely": ["super", "really", "very", "incredibly", "insanely"],
            "however": ["but", "though", "still", "yet", "although"],
            "therefore": ["so", "thus", "hence", "that's why", "which is why"],
            "consequently": ["so", "as a result", "because of this", "that's why"],
            "regarding": ["about", "on", "concerning", "as for", "when it comes to"],
            "assist": ["help", "aid", "support", "give a hand", "back up"],
            "attempt": ["try", "go for", "take a shot at", "have a go at"],
            "commence": ["start", "begin", "kick off", "get going", "launch"],
            "conclude": ["end", "finish", "wrap up", "close out", "wind down"],
            "construct": ["build", "make", "create", "put together", "set up"],
            "determine": ["figure out", "find out", "decide", "work out", "establish"],
            "enhance": ["improve", "boost", "upgrade", "make better", "level up"],
            "ensure": ["make sure", "guarantee", "secure", "confirm", "verify"],
            "establish": ["set up", "create", "build", "start", "found"],
            "examine": ["look at", "check out", "study", "review", "inspect"],
            "identify": ["find", "spot", "recognize", "pinpoint", "pick out"],
            "maintain": ["keep", "hold", "sustain", "preserve", "uphold"],
            "modify": ["change", "adjust", "tweak", "alter", "update"],
            "occur": ["happen", "take place", "go down", "come about", "pop up"],
            "perform": ["do", "carry out", "execute", "pull off", "handle"],
            "possess": ["have", "own", "hold", "carry", "keep"],
            "purchase": ["buy", "get", "pick up", "grab", "score"],
            "receive": ["get", "have", "obtain", "pick up", "be given"],
            "recommend": ["suggest", "advise", "propose", "say", "think"],
            "reside": ["live", "stay", "dwell", "hang out", "be based"],
            "retain": ["keep", "hold onto", "maintain", "preserve", "save"],
            "select": ["pick", "choose", "go with", "opt for", "settle on"],
            "terminate": ["end", "stop", "finish", "cut off", "wrap up"],
            "transmit": ["send", "pass on", "relay", "share", "forward"],
            "utilize": ["use", "employ", "work with", "make use of", "tap into"],
        }
    
    def _init_error_patterns(self):
        """Initialize realistic human error patterns"""
        
        # Keyboard neighbor typos
        self.keyboard_neighbors = {
            'a': 'sqwz', 'b': 'vghn', 'c': 'xdfv', 'd': 'erfcxs',
            'e': 'rdsw', 'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg',
            'i': 'uojk', 'j': 'uikmnh', 'k': 'ioljm', 'l': 'opk',
            'm': 'njk', 'n': 'bhjm', 'o': 'iplk', 'p': 'ol',
            'q': 'wa', 'r': 'etdf', 's': 'wedxza', 't': 'ryfg',
            'u': 'yihj', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc',
            'y': 'tugh', 'z': 'xsa',
        }
        
        # Common typos (intentional misspellings humans make)
        self.common_typos = {
            "the": ["teh", "hte", "th", "tha"],
            "and": ["adn", "nad", "an", "andd"],
            "that": ["taht", "tht", "thta"],
            "have": ["ahve", "hvae", "hav"],
            "with": ["wiht", "wtih", "wth"],
            "this": ["tihs", "thsi", "ths"],
            "from": ["form", "fomr", "frm"],
            "they": ["tehy", "thye", "tey"],
            "been": ["bene", "ben", "bean"],
            "their": ["thier", "ther", "theri"],
            "would": ["woudl", "owuld", "wuold"],
            "about": ["abuot", "abotu", "baout"],
            "could": ["coudl", "cuold", "ocould"],
            "which": ["whcih", "wich", "whihc"],
            "there": ["theer", "tehre", "ther"],
            "where": ["wehre", "wheer", "wher"],
            "because": ["becuase", "beacuse", "becasue", "b/c", "bc"],
            "something": ["somethign", "soemthing", "smth", "sth"],
            "really": ["realy", "raelly", "rly"],
            "definitely": ["definately", "definitly", "def"],
            "probably": ["porbably", "probabl", "prob"],
            "different": ["diferent", "diffrent", "diff"],
            "through": ["trough", "thru", "throuhg"],
            "thought": ["thougt", "thougth", "thoguht"],
            "actually": ["acutally", "actualy", "actally"],
            "immediately": ["immediatly", "imediately", "immedialty"],
            "necessary": ["neccessary", "necesary", "neccesary"],
            "received": ["recieved", "recevied", "receieved"],
            "separate": ["seperate", "seperete", "seprate"],
            "tomorrow": ["tommorow", "tomorow", "tmrw", "tmr"],
            "together": ["togehter", "togather", "togetehr"],
            "believe": ["beleive", "belive", "beleave"],
            "environment": ["enviroment", "enviornment", "enviorment"],
            "government": ["goverment", "govenment", "govt"],
            "beginning": ["begining", "beginnign", "beggining"],
        }
        
        # Missing/extra spaces humans make
        self.space_errors = [
            ("  ", " "),  # double space
            (" ,", ","),
            (" .", "."),
            ("( ", "("),
            (" )", ")"),
        ]
        
        # Punctuation variations
        self.punct_variations = {
            ".": [".", "..", "...", ". "],
            ",": [",", ", ", ",,"],
            "!": ["!", "!!", "!!!"],
            "?": ["?", "??", "?!"],
        }
    
    # ══════════════════════════════════════════════════════════════════════════
    # TECHNIQUE IMPLEMENTATIONS
    # ══════════════════════════════════════════════════════════════════════════
    
    def technique_remove_ai_phrases(self, text: str, intensity: float = 0.5) -> str:
        """Remove/replace AI-typical phrases"""
        for phrase, replacements in self.ai_phrases.items():
            if phrase.lower() in text.lower():
                if random.random() < intensity:
                    replacement = random.choice(replacements)
                    # Case-insensitive replace
                    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                    text = pattern.sub(replacement, text, count=1)
        return text
    
    def technique_add_contractions(self, text: str, intensity: float = 0.7) -> str:
        """Convert formal phrases to contractions"""
        for formal, contracted in self.contractions.items():
            if random.random() < intensity:
                pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
                text = pattern.sub(contracted, text)
        return text
    
    def technique_add_disfluencies(self, text: str, intensity: float = 0.3) -> str:
        """Add speech disfluencies at sentence starts"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for sent in sentences:
            if sent.strip() and random.random() < intensity:
                disfluency = random.choice(self.disfluencies)
                # Capitalize appropriately
                if sent[0].isupper():
                    disfluency = disfluency.capitalize()
                sent = disfluency + " " + sent[0].lower() + sent[1:]
            result.append(sent)
        
        return " ".join(result)
    
    def technique_add_idioms(self, text: str, intensity: float = 0.2) -> str:
        """Insert idioms and colloquialisms"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text
        
        # Insert idiom at random position
        num_insertions = max(1, int(len(sentences) * intensity * 0.3))
        for _ in range(num_insertions):
            if random.random() < intensity:
                pos = random.randint(1, len(sentences) - 1)
                idiom = random.choice(self.idioms)
                idiom_sentence = idiom.capitalize() + ", " if not idiom.endswith(",") else idiom.capitalize() + " "
                sentences[pos] = idiom_sentence + sentences[pos][0].lower() + sentences[pos][1:]
        
        return " ".join(sentences)
    
    def technique_add_rhetorical_questions(self, text: str, intensity: float = 0.25) -> str:
        """Add rhetorical questions"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for i, sent in enumerate(sentences):
            result.append(sent)
            # Add rhetorical question after some sentences
            if random.random() < intensity * 0.4 and i < len(sentences) - 1:
                question = random.choice(self.rhetorical_questions)
                result.append(question)
        
        return " ".join(result)
    
    def technique_add_emphatics(self, text: str, intensity: float = 0.3) -> str:
        """Add emphatic expressions"""
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            # Add emphatic before adjectives/verbs occasionally
            if random.random() < intensity * 0.1 and i > 0:
                emphatic = random.choice(self.emphatics)
                result.append(emphatic)
            result.append(word)
        
        return " ".join(result)
    
    def technique_add_hedges(self, text: str, intensity: float = 0.25) -> str:
        """Add hedging/uncertainty language"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        
        for sent in sentences:
            if sent.strip() and random.random() < intensity:
                hedge = random.choice(self.hedges)
                if hedge.startswith("I "):
                    sent = hedge + ", " + sent[0].lower() + sent[1:]
                else:
                    # Insert hedge into sentence
                    words = sent.split()
                    if len(words) > 3:
                        pos = random.randint(1, min(3, len(words) - 1))
                        words.insert(pos, hedge)
                        sent = " ".join(words)
            result.append(sent)
        
        return " ".join(result)
    
    def technique_synonym_substitution(self, text: str, intensity: float = 0.4) -> str:
        """Replace words with synonyms"""
        for word, synonyms in self.synonyms.items():
            if word in text.lower():
                if random.random() < intensity:
                    synonym = random.choice(synonyms)
                    pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                    text = pattern.sub(synonym, text, count=1)
        return text
    
    def technique_add_typos(self, text: str, intensity: float = 0.15) -> str:
        """Add realistic human typos"""
        words = text.split()
        result = []
        
        for word in words:
            clean_word = word.strip(string.punctuation)
            punct = word[len(clean_word):] if len(word) > len(clean_word) else ""
            
            # Check common typos first
            if clean_word.lower() in self.common_typos and random.random() < intensity:
                typo = random.choice(self.common_typos[clean_word.lower()])
                # Preserve capitalization
                if clean_word[0].isupper():
                    typo = typo.capitalize()
                result.append(typo + punct)
            # Then keyboard neighbor typos
            elif len(clean_word) > 3 and random.random() < intensity * 0.3:
                pos = random.randint(1, len(clean_word) - 2)
                char = clean_word[pos].lower()
                if char in self.keyboard_neighbors:
                    neighbor = random.choice(self.keyboard_neighbors[char])
                    typo = clean_word[:pos] + neighbor + clean_word[pos+1:]
                    result.append(typo + punct)
                else:
                    result.append(word)
            else:
                result.append(word)
        
        return " ".join(result)
    
    def technique_vary_punctuation(self, text: str, intensity: float = 0.2) -> str:
        """Add punctuation variations"""
        if random.random() < intensity:
            # Add ellipsis
            sentences = text.split(". ")
            if len(sentences) > 2:
                pos = random.randint(0, len(sentences) - 2)
                sentences[pos] = sentences[pos] + "..."
                text = ". ".join(sentences)
        
        if random.random() < intensity:
            # Add em-dash
            text = text.replace(" - ", " — ", 1)
        
        if random.random() < intensity * 0.5:
            # Double punctuation
            text = re.sub(r'!(?!\!)', '!!', text, count=1)
        
        return text
    
    def technique_fragment_sentences(self, text: str, intensity: float = 0.2) -> str:
        """Add sentence fragments"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text
        
        result = []
        for i, sent in enumerate(sentences):
            result.append(sent)
            if random.random() < intensity * 0.3 and i < len(sentences) - 1:
                fragment = random.choice(self.fragments)
                result.append(fragment)
        
        return " ".join(result)
    
    def technique_add_anecdote_hint(self, text: str, intensity: float = 0.15) -> str:
        """Add anecdote-style starters"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 4:
            return text
        
        if random.random() < intensity:
            pos = random.randint(1, len(sentences) - 2)
            starter = random.choice(self.anecdote_starters)
            sentences[pos] = starter + " " + sentences[pos][0].lower() + sentences[pos][1:]
        
        return " ".join(sentences)
    
    def technique_vary_sentence_length(self, text: str, intensity: float = 0.4) -> str:
        """Create burstiness - dramatic sentence length variation"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 4:
            return text
        
        result = []
        for sent in sentences:
            if random.random() < intensity * 0.3:
                # Split long sentence
                if len(sent) > 100:
                    # Find a comma to split at
                    comma_pos = sent.find(", ", len(sent) // 3, 2 * len(sent) // 3)
                    if comma_pos > 0:
                        part1 = sent[:comma_pos] + "."
                        part2 = sent[comma_pos + 2].upper() + sent[comma_pos + 3:]
                        result.append(part1)
                        result.append(part2)
                        continue
            
            if random.random() < intensity * 0.2:
                # Combine with next (if short enough)
                if len(sent) < 50 and result and len(result[-1]) < 50:
                    combined = result[-1].rstrip(".") + ", and " + sent[0].lower() + sent[1:]
                    result[-1] = combined
                    continue
            
            result.append(sent)
        
        return " ".join(result)
    
    def technique_add_cultural_refs(self, text: str, intensity: float = 0.1) -> str:
        """Add cultural references"""
        if random.random() > intensity:
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text
        
        pos = random.randint(0, len(sentences) - 1)
        ref = random.choice(self.cultural_refs)
        
        # Add as a comparison
        sentences[pos] = sentences[pos].rstrip(".") + " — " + ref + "."
        
        return " ".join(sentences)
    
    def technique_simulate_high_temperature(self, text: str, intensity: float = 0.3) -> str:
        """Simulate high-temperature generation (more randomness)"""
        # Replace some words with less predictable alternatives
        substitutions = {
            "very": ["super", "hella", "crazy", "mad", "wicked"],
            "good": ["solid", "fire", "dope", "sick", "legit"],
            "bad": ["rough", "trash", "weak", "mid", "wack"],
            "like": ["dig", "vibe with", "fw", "appreciate", "stan"],
            "said": ["was like", "goes", "dropped", "hit me with"],
            "think": ["reckon", "figure", "feel like", "lowkey think"],
            "know": ["get", "vibe", "realize", "clock"],
        }
        
        for word, alts in substitutions.items():
            if word in text.lower() and random.random() < intensity:
                alt = random.choice(alts)
                pattern = re.compile(r'\b' + word + r'\b', re.IGNORECASE)
                text = pattern.sub(alt, text, count=1)
        
        return text
    
    # ══════════════════════════════════════════════════════════════════════════
    # MAIN HUMANIZE METHOD - RANDOMIZED TECHNIQUE SELECTION
    # ══════════════════════════════════════════════════════════════════════════
    
    def humanize(self, text: str, style: str = "random") -> str:
        """
        Humanize text using a RANDOM combination of techniques.
        
        Styles:
        - "random": Completely random technique selection (default)
        - "light": Subtle changes, fewer techniques
        - "medium": Moderate changes
        - "heavy": Aggressive changes, many techniques
        - "stealth": Professional bypass style
        - "casual": Very informal/conversational
        - "academic": Subtle, maintains formality
        """
        
        if not text or len(text.strip()) < 20:
            return text
        
        original_text = text
        
        # ══════════════════════════════════════════════════════════════════
        # TECHNIQUE POOL with base probabilities
        # ══════════════════════════════════════════════════════════════════
        techniques = [
            ("remove_ai_phrases", self.technique_remove_ai_phrases, 0.7),
            ("add_contractions", self.technique_add_contractions, 0.65),
            ("add_disfluencies", self.technique_add_disfluencies, 0.4),
            ("add_idioms", self.technique_add_idioms, 0.35),
            ("add_rhetorical_questions", self.technique_add_rhetorical_questions, 0.3),
            ("add_emphatics", self.technique_add_emphatics, 0.35),
            ("add_hedges", self.technique_add_hedges, 0.4),
            ("synonym_substitution", self.technique_synonym_substitution, 0.5),
            ("add_typos", self.technique_add_typos, 0.25),
            ("vary_punctuation", self.technique_vary_punctuation, 0.35),
            ("fragment_sentences", self.technique_fragment_sentences, 0.3),
            ("add_anecdote_hint", self.technique_add_anecdote_hint, 0.2),
            ("vary_sentence_length", self.technique_vary_sentence_length, 0.45),
            ("add_cultural_refs", self.technique_add_cultural_refs, 0.15),
            ("simulate_high_temperature", self.technique_simulate_high_temperature, 0.25),
        ]
        
        # ══════════════════════════════════════════════════════════════════
        # STYLE-BASED PROBABILITY MODIFIERS
        # ══════════════════════════════════════════════════════════════════
        style_modifiers = {
            "random": {
                # Random intensity for each technique
                "base_prob_multiplier": (0.5, 1.5),
                "intensity_range": (0.2, 0.9),
                "min_techniques": 3,
                "max_techniques": 12,
            },
            "light": {
                "base_prob_multiplier": (0.3, 0.6),
                "intensity_range": (0.1, 0.3),
                "min_techniques": 2,
                "max_techniques": 5,
                "prefer": ["remove_ai_phrases", "add_contractions", "synonym_substitution"],
            },
            "medium": {
                "base_prob_multiplier": (0.6, 1.0),
                "intensity_range": (0.3, 0.6),
                "min_techniques": 4,
                "max_techniques": 8,
            },
            "heavy": {
                "base_prob_multiplier": (0.8, 1.3),
                "intensity_range": (0.5, 0.9),
                "min_techniques": 7,
                "max_techniques": 14,
            },
            "stealth": {
                "base_prob_multiplier": (0.5, 0.9),
                "intensity_range": (0.2, 0.5),
                "min_techniques": 5,
                "max_techniques": 9,
                "prefer": ["remove_ai_phrases", "add_contractions", "vary_sentence_length", 
                          "synonym_substitution", "add_hedges"],
                "avoid": ["add_typos", "add_cultural_refs", "simulate_high_temperature"],
            },
            "casual": {
                "base_prob_multiplier": (0.7, 1.2),
                "intensity_range": (0.4, 0.8),
                "min_techniques": 5,
                "max_techniques": 11,
                "prefer": ["add_disfluencies", "add_contractions", "add_rhetorical_questions",
                          "add_idioms", "simulate_high_temperature"],
            },
            "academic": {
                "base_prob_multiplier": (0.4, 0.7),
                "intensity_range": (0.15, 0.35),
                "min_techniques": 3,
                "max_techniques": 6,
                "prefer": ["remove_ai_phrases", "add_hedges", "vary_sentence_length"],
                "avoid": ["add_typos", "add_disfluencies", "add_cultural_refs", 
                         "simulate_high_temperature", "add_idioms"],
            },
        }
        
        # Get style config (default to random)
        if style not in style_modifiers:
            style = random.choice(list(style_modifiers.keys()))
        
        config = style_modifiers[style]
        
        # ══════════════════════════════════════════════════════════════════
        # RANDOMLY SELECT WHICH TECHNIQUES TO APPLY
        # ══════════════════════════════════════════════════════════════════
        selected_techniques = []
        
        for name, func, base_prob in techniques:
            # Skip if in avoid list
            if "avoid" in config and name in config["avoid"]:
                continue
            
            # Boost if in prefer list
            prob = base_prob
            if "prefer" in config and name in config["prefer"]:
                prob *= 1.5
            
            # Apply random multiplier
            mult_low, mult_high = config["base_prob_multiplier"]
            prob *= random.uniform(mult_low, mult_high)
            
            # Randomly decide to use this technique
            if random.random() < prob:
                # Random intensity within style range
                int_low, int_high = config["intensity_range"]
                intensity = random.uniform(int_low, int_high)
                selected_techniques.append((name, func, intensity))
        
        # Ensure we have enough techniques
        min_tech = config["min_techniques"]
        max_tech = config["max_techniques"]
        
        if len(selected_techniques) < min_tech:
            # Add more random techniques
            available = [(n, f, p) for n, f, p in techniques 
                        if n not in [t[0] for t in selected_techniques]
                        and ("avoid" not in config or n not in config["avoid"])]
            random.shuffle(available)
            while len(selected_techniques) < min_tech and available:
                n, f, _ = available.pop()
                int_low, int_high = config["intensity_range"]
                selected_techniques.append((n, f, random.uniform(int_low, int_high)))
        
        if len(selected_techniques) > max_tech:
            random.shuffle(selected_techniques)
            selected_techniques = selected_techniques[:max_tech]
        
        # ══════════════════════════════════════════════════════════════════
        # APPLY SELECTED TECHNIQUES IN RANDOM ORDER
        # ══════════════════════════════════════════════════════════════════
        random.shuffle(selected_techniques)
        
        for name, func, intensity in selected_techniques:
            try:
                text = func(text, intensity)
            except Exception:
                pass  # Skip if technique fails
        
        # Clean up any artifacts
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces
        text = re.sub(r'\s+([.,!?])', r'\1', text)  # Space before punctuation
        text = re.sub(r'([.,!?]){3,}', r'\1\1', text)  # Too much punctuation
        text = text.strip()
        
        # Ensure we actually changed something
        if text == original_text:
            # Force at least some changes
            text = self.technique_add_contractions(text, 0.8)
            text = self.technique_remove_ai_phrases(text, 0.8)
        
        return text
    
    def humanize_batch(self, texts: List[str], style_distribution: dict = None) -> List[str]:
        """
        Humanize a batch of texts with varied styles.
        
        Args:
            texts: List of texts to humanize
            style_distribution: Dict of style -> probability 
                               (default: equal distribution)
        """
        if style_distribution is None:
            style_distribution = {
                "random": 0.25,
                "light": 0.15,
                "medium": 0.20,
                "heavy": 0.10,
                "stealth": 0.15,
                "casual": 0.10,
                "academic": 0.05,
            }
        
        styles = list(style_distribution.keys())
        weights = list(style_distribution.values())
        
        results = []
        for text in texts:
            style = random.choices(styles, weights=weights)[0]
            results.append(self.humanize(text, style))
        
        return results


# ══════════════════════════════════════════════════════════════════════════════
# TESTING
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Test the humanizer
    humanizer = AdvancedHumanizer()
    
    test_text = """
    Furthermore, it is important to note that artificial intelligence has 
    demonstrated significant capabilities in various domains. The implementation 
    of these systems requires comprehensive understanding of the underlying 
    mechanisms. Additionally, one must consider the ethical implications that 
    subsequently arise from such technological advancements. In conclusion, 
    the utilization of AI will continue to facilitate numerous applications 
    in the foreseeable future.
    """
    
    print("=" * 70)
    print("ORIGINAL AI TEXT:")
    print("=" * 70)
    print(test_text.strip())
    
    styles = ["light", "medium", "heavy", "stealth", "casual", "academic", "random"]
    
    for style in styles:
        print("\n" + "=" * 70)
        print(f"HUMANIZED ({style.upper()}):")
        print("=" * 70)
        result = humanizer.humanize(test_text, style)
        print(result)
    
    # Show that random produces different results each time
    print("\n" + "=" * 70)
    print("RANDOM STYLE - 3 DIFFERENT OUTPUTS FROM SAME INPUT:")
    print("=" * 70)
    for i in range(3):
        print(f"\n--- Iteration {i+1} ---")
        print(humanizer.humanize(test_text, "random"))
