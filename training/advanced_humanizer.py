#!/usr/bin/env python3
"""
Advanced AI Text Humanizer - Randomized Multi-Technique Approach
Each humanization uses a DIFFERENT random combination of techniques.
"""

import random
import re
import string
from typing import List, Optional


class AdvancedHumanizer:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self._init_resources()
    
    def _init_resources(self):
        self.ai_phrases = {
            "Furthermore,": ["Also,", "Plus,", "And", ""],
            "Moreover,": ["Also,", "And", "Plus,", ""],
            "Additionally,": ["Also,", "Plus,", "And", ""],
            "In conclusion,": ["So,", "Basically,", "All in all,", ""],
            "To summarize,": ["Long story short,", "In short,", ""],
            "It is important to note that": ["Thing is,", "Here's the deal:", ""],
            "It's worth mentioning that": ["Also,", "Oh, and", ""],
            "In today's world,": ["These days,", "Nowadays,", ""],
            "It is essential to": ["You gotta", "You need to", "Better to"],
            "This demonstrates that": ["This shows", "See?", "So basically"],
            "plays a pivotal role": ["really matters", "is huge", "is key"],
            "a myriad of": ["lots of", "tons of", "many"],
            "utilize": ["use", "work with", "go with"],
            "facilitate": ["help", "make easier", "allow"],
            "comprehensive": ["full", "complete", "thorough"],
            "subsequently": ["then", "after that", "next"],
            "nevertheless": ["still", "but", "anyway"],
            "thereby": ["so", "which", "and"],
        }
        
        self.idioms = [
            "at the end of the day", "for what it's worth", "the thing is",
            "here's the deal", "long story short", "bottom line", "real talk",
            "not gonna lie", "to be honest", "truth be told", "if you ask me",
            "believe it or not", "funny enough", "just saying", "that said",
        ]
        
        self.disfluencies = [
            "well,", "so,", "I mean,", "you know,", "like,", "basically,",
            "honestly,", "actually,", "look,", "thing is,", "okay so,",
            "anyway,", "I guess,", "I think,", "probably,", "maybe,",
        ]
        
        self.rhetorical_questions = [
            "Right?", "You know?", "Make sense?", "Get it?", "Crazy, right?",
            "Wild, huh?", "Don't you think?", "Fair enough?",
        ]
        
        self.emphatics = [
            "seriously", "literally", "absolutely", "totally", "definitely",
            "obviously", "clearly", "honestly", "really", "actually",
        ]
        
        self.hedges = [
            "probably", "maybe", "perhaps", "I think", "I believe", "I guess",
            "it seems", "apparently", "sort of", "kind of", "might be",
        ]
        
        self.contractions = {
            "I am": "I'm", "I have": "I've", "I will": "I'll", "I would": "I'd",
            "you are": "you're", "you have": "you've", "you will": "you'll",
            "he is": "he's", "she is": "she's", "it is": "it's",
            "we are": "we're", "we have": "we've", "they are": "they're",
            "is not": "isn't", "are not": "aren't", "was not": "wasn't",
            "has not": "hasn't", "have not": "haven't", "will not": "won't",
            "do not": "don't", "does not": "doesn't", "did not": "didn't",
            "can not": "can't", "cannot": "can't", "could not": "couldn't",
            "should not": "shouldn't", "would not": "wouldn't",
            "could have": "could've", "would have": "would've",
            "should have": "should've", "going to": "gonna", "want to": "wanna",
            "got to": "gotta", "kind of": "kinda", "sort of": "sorta",
        }
        
        self.fragments = [
            "Crazy.", "Wild.", "Insane.", "Absolutely.", "Definitely.",
            "For sure.", "No doubt.", "Makes sense.", "Fair point.",
            "Seriously though.", "For real.", "No joke.",
        ]
        
        self.synonyms = {
            "significant": ["big", "major", "huge", "important"],
            "various": ["different", "lots of", "many", "several"],
            "demonstrate": ["show", "prove", "display"],
            "require": ["need", "call for", "demand"],
            "provide": ["give", "offer", "supply"],
            "obtain": ["get", "grab", "pick up"],
            "approximately": ["about", "around", "roughly"],
            "frequently": ["often", "a lot", "usually"],
            "extremely": ["super", "really", "very"],
            "however": ["but", "though", "still"],
            "therefore": ["so", "thus", "that's why"],
        }
        
        self.common_typos = {
            "the": ["teh", "hte"], "and": ["adn", "nad"],
            "that": ["taht", "tht"], "have": ["ahve", "hvae"],
            "with": ["wiht", "wtih"], "this": ["tihs", "thsi"],
            "from": ["form", "fomr"], "their": ["thier", "ther"],
            "because": ["becuase", "bc"], "really": ["realy", "rly"],
            "definitely": ["definately", "def"], "probably": ["prob", "prolly"],
        }
    
    def technique_remove_ai_phrases(self, text: str, intensity: float) -> str:
        for phrase, replacements in self.ai_phrases.items():
            if phrase.lower() in text.lower() and random.random() < intensity:
                replacement = random.choice(replacements)
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                text = pattern.sub(replacement, text, count=1)
        return text
    
    def technique_add_contractions(self, text: str, intensity: float) -> str:
        for formal, contracted in self.contractions.items():
            if random.random() < intensity:
                pattern = re.compile(r'\b' + re.escape(formal) + r'\b', re.IGNORECASE)
                text = pattern.sub(contracted, text)
        return text
    
    def technique_add_disfluencies(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for sent in sentences:
            if sent.strip() and random.random() < intensity:
                disfluency = random.choice(self.disfluencies)
                if sent[0].isupper():
                    disfluency = disfluency.capitalize()
                sent = disfluency + " " + sent[0].lower() + sent[1:] if len(sent) > 1 else sent
            result.append(sent)
        return " ".join(result)
    
    def technique_add_idioms(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text
        if random.random() < intensity:
            pos = random.randint(1, len(sentences) - 1)
            idiom = random.choice(self.idioms)
            sentences[pos] = idiom.capitalize() + ", " + sentences[pos][0].lower() + sentences[pos][1:] if len(sentences[pos]) > 1 else sentences[pos]
        return " ".join(sentences)
    
    def technique_add_rhetorical_questions(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for i, sent in enumerate(sentences):
            result.append(sent)
            if random.random() < intensity * 0.3 and i < len(sentences) - 1:
                result.append(random.choice(self.rhetorical_questions))
        return " ".join(result)
    
    def technique_add_emphatics(self, text: str, intensity: float) -> str:
        words = text.split()
        result = []
        for i, word in enumerate(words):
            if random.random() < intensity * 0.08 and i > 0:
                result.append(random.choice(self.emphatics))
            result.append(word)
        return " ".join(result)
    
    def technique_add_hedges(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = []
        for sent in sentences:
            if sent.strip() and random.random() < intensity:
                hedge = random.choice(self.hedges)
                sent = hedge + ", " + sent[0].lower() + sent[1:] if len(sent) > 1 else sent
            result.append(sent)
        return " ".join(result)
    
    def technique_synonym_substitution(self, text: str, intensity: float) -> str:
        for word, synonyms in self.synonyms.items():
            if word in text.lower() and random.random() < intensity:
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                text = pattern.sub(random.choice(synonyms), text, count=1)
        return text
    
    def technique_add_typos(self, text: str, intensity: float) -> str:
        words = text.split()
        result = []
        for word in words:
            clean = word.strip(string.punctuation)
            punct = word[len(clean):] if len(word) > len(clean) else ""
            if clean.lower() in self.common_typos and random.random() < intensity:
                typo = random.choice(self.common_typos[clean.lower()])
                if clean[0].isupper():
                    typo = typo.capitalize()
                result.append(typo + punct)
            else:
                result.append(word)
        return " ".join(result)
    
    def technique_vary_punctuation(self, text: str, intensity: float) -> str:
        if random.random() < intensity:
            text = text.replace(". ", "... ", 1)
        if random.random() < intensity:
            text = text.replace(" - ", " — ", 1)
        return text
    
    def technique_fragment_sentences(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 3:
            return text
        result = []
        for i, sent in enumerate(sentences):
            result.append(sent)
            if random.random() < intensity * 0.25 and i < len(sentences) - 1:
                result.append(random.choice(self.fragments))
        return " ".join(result)
    
    def technique_vary_sentence_length(self, text: str, intensity: float) -> str:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) < 4:
            return text
        result = []
        for sent in sentences:
            if random.random() < intensity * 0.25 and len(sent) > 80:
                comma_pos = sent.find(", ", len(sent)//3, 2*len(sent)//3)
                if comma_pos > 0:
                    result.append(sent[:comma_pos] + ".")
                    result.append(sent[comma_pos+2].upper() + sent[comma_pos+3:] if len(sent) > comma_pos+3 else "")
                    continue
            result.append(sent)
        return " ".join(result)
    
    def humanize(self, text: str, style: str = "random") -> str:
        if not text or len(text.strip()) < 20:
            return text
        
        original = text
        
        techniques = [
            (self.technique_remove_ai_phrases, 0.7),
            (self.technique_add_contractions, 0.65),
            (self.technique_add_disfluencies, 0.4),
            (self.technique_add_idioms, 0.35),
            (self.technique_add_rhetorical_questions, 0.3),
            (self.technique_add_emphatics, 0.35),
            (self.technique_add_hedges, 0.4),
            (self.technique_synonym_substitution, 0.5),
            (self.technique_add_typos, 0.25),
            (self.technique_vary_punctuation, 0.35),
            (self.technique_fragment_sentences, 0.3),
            (self.technique_vary_sentence_length, 0.45),
        ]
        
        style_config = {
            "random": {"prob_mult": (0.5, 1.5), "int_range": (0.2, 0.9), "min": 3, "max": 10},
            "light": {"prob_mult": (0.3, 0.6), "int_range": (0.1, 0.3), "min": 2, "max": 5},
            "medium": {"prob_mult": (0.6, 1.0), "int_range": (0.3, 0.6), "min": 4, "max": 7},
            "heavy": {"prob_mult": (0.8, 1.3), "int_range": (0.5, 0.9), "min": 6, "max": 11},
            "stealth": {"prob_mult": (0.5, 0.9), "int_range": (0.2, 0.5), "min": 4, "max": 8},
            "casual": {"prob_mult": (0.7, 1.2), "int_range": (0.4, 0.8), "min": 5, "max": 10},
            "academic": {"prob_mult": (0.4, 0.7), "int_range": (0.15, 0.35), "min": 3, "max": 6},
        }
        
        if style not in style_config:
            style = random.choice(list(style_config.keys()))
        
        cfg = style_config[style]
        selected = []
        
        for func, base_prob in techniques:
            prob = base_prob * random.uniform(*cfg["prob_mult"])
            if random.random() < prob:
                intensity = random.uniform(*cfg["int_range"])
                selected.append((func, intensity))
        
        while len(selected) < cfg["min"]:
            func, _ = random.choice(techniques)
            if func not in [s[0] for s in selected]:
                selected.append((func, random.uniform(*cfg["int_range"])))
        
        if len(selected) > cfg["max"]:
            random.shuffle(selected)
            selected = selected[:cfg["max"]]
        
        random.shuffle(selected)
        for func, intensity in selected:
            try:
                text = func(text, intensity)
            except:
                pass
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        if text == original:
            text = self.technique_add_contractions(text, 0.8)
            text = self.technique_remove_ai_phrases(text, 0.8)
        
        return text
    
    def humanize_batch(self, texts: List[str], style_distribution: dict = None) -> List[str]:
        if style_distribution is None:
            style_distribution = {
                "random": 0.25, "light": 0.15, "medium": 0.20, "heavy": 0.10,
                "stealth": 0.15, "casual": 0.10, "academic": 0.05,
            }
        styles = list(style_distribution.keys())
        weights = list(style_distribution.values())
        return [self.humanize(t, random.choices(styles, weights=weights)[0]) for t in texts]


if __name__ == "__main__":
    h = AdvancedHumanizer()
    test = """Furthermore, it is important to note that artificial intelligence has demonstrated significant capabilities. The implementation of these systems requires comprehensive understanding. Additionally, one must consider the ethical implications that subsequently arise."""
    
    print("ORIGINAL:", test)
    for style in ["light", "medium", "heavy", "stealth", "casual"]:
        print(f"\n{style.upper()}:", h.humanize(test, style))
