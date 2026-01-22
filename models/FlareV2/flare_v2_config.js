// Flare V2 Config
const FLARE_V2_CONFIG = {
    name: 'Flare V2',
    version: '2.0.0',
    type: 'human_vs_humanized',
    accuracy: 0.9800,
    features: 441,
    heuristicFeatures: 57,
    embeddingDim: 384,
    labels: { 0: 'human', 1: 'humanized' },
    featureNames: ["chars", "words", "sents", "avg_word_len", "sent_mean", "sent_std", "sent_cv", "sent_range", "sent_skew", "sent_kurt", "ai_residue", "ai_residue_pct", "paraphrase_markers", "thesaurus_words", "formal_synonyms", "human_informal", "human_pct", "contractions", "contraction_rate", "exclamations", "questions", "ellipses", "dashes", "parens", "commas", "semicolons", "colons", "quotes", "first_person", "first_person_rate", "vocab_richness", "hapax_ratio", "top_word_freq", "bigram_rep", "trigram_rep", "quadgram_rep", "starter_diversity", "starter_rep", "paragraphs", "para_std", "para_cv", "style_shift", "short_words", "medium_words", "long_words", "very_long_words", "transitions", "transition_rate", "passive_voice", "passive_rate", "hedges", "hedge_rate", "emphatics", "numbers", "number_rate", "caps_ratio", "entropy"]
};

if (typeof module !== 'undefined') module.exports = FLARE_V2_CONFIG;
