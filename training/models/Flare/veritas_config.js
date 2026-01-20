/**
 * FLARE Detection System Configuration
 * Trained: 2026-01-20 01:49:35
 * Accuracy: 99.84%
 * F1 Score: 0.9984
 */

const FlareConfig = {
    version: '2.0',
    type: 'humanization-detection',
    accuracy: 0.9984444444444445,
    precision: 1.0,
    recall: 0.9968888888888889,
    f1Score: 0.9984420209214333,
    
    features: [
        "variance_of_variance",
        "variance_stability",
        "local_var_consistency",
        "word_var_of_var",
        "syllable_variance",
        "length_distribution_uniformity",
        "autocorr_lag1",
        "autocorr_lag2",
        "autocorr_decay",
        "autocorr_flatness",
        "autocorr_periodicity",
        "autocorr_noise",
        "length_complexity_corr",
        "vocab_structure_corr",
        "position_length_corr",
        "correlation_break_score",
        "synonym_cluster_usage",
        "rare_synonym_ratio",
        "sophistication_jumps",
        "formal_informal_mix",
        "register_consistency",
        "word_choice_naturalness",
        "contraction_rate",
        "contraction_uniformity",
        "contraction_position_variance",
        "contraction_context_fit",
        "sentence_start_diversity",
        "sentence_start_entropy",
        "template_score",
        "parallelism_score",
        "clause_depth_variance",
        "embedding_naturalness",
        "sophistication_variance",
        "sophistication_autocorr",
        "word_choice_consistency",
        "formality_stability",
        "bigram_predictability",
        "trigram_predictability",
        "ngram_surprise_variance",
        "phrase_originality",
        "comma_density",
        "punctuation_variety",
        "punctuation_entropy",
        "semicolon_colon_ratio",
        "transition_density",
        "hedging_density",
        "discourse_variety",
        "ai_phrase_density",
        "lexical_entropy",
        "sentence_entropy",
        "entropy_stability",
        "char_entropy",
        "word_position_entropy",
        "perplexity_proxy",
        "char_repeat_ratio",
        "whitespace_consistency",
        "case_pattern_entropy",
        "special_char_density",
        "rhythm_variance",
        "stress_pattern_entropy",
        "syllable_rhythm",
        "reading_flow_score",
        "topic_consistency",
        "reference_density",
        "connector_appropriateness",
        "semantic_flow"
],
    
    featureStats: {
        "variance_of_variance": {
                "mean": 0.8997900252352792,
                "std": 0.2691471252852331,
                "importance": 0.0039173851416127795
        },
        "variance_stability": {
                "mean": 0.38020264915953755,
                "std": 0.2846018796065531,
                "importance": 0.005631587747879443
        },
        "local_var_consistency": {
                "mean": 0.38020264915953755,
                "std": 0.2846018796065531,
                "importance": 0.0055775816337872335
        },
        "word_var_of_var": {
                "mean": 0.6800600487685874,
                "std": 0.261612481662716,
                "importance": 0.005708619879871481
        },
        "syllable_variance": {
                "mean": 0.38673328867630025,
                "std": 0.1349823044589112,
                "importance": 0.050125069902481535
        },
        "length_distribution_uniformity": {
                "mean": 0.3237043000196604,
                "std": 0.27638800503774236,
                "importance": 0.004939940179968165
        },
        "autocorr_lag1": {
                "mean": 0.4666299524181895,
                "std": 0.133832751412572,
                "importance": 0.004966410895275088
        },
        "autocorr_lag2": {
                "mean": 0.4606877740555887,
                "std": 0.14204203301625395,
                "importance": 0.004760659146443601
        },
        "autocorr_decay": {
                "mean": 0.5139610879744224,
                "std": 0.21003683391772945,
                "importance": 0.004702044234928646
        },
        "autocorr_flatness": {
                "mean": 0.46323903437447106,
                "std": 0.3148171808756663,
                "importance": 0.005087670016044239
        },
        "autocorr_periodicity": {
                "mean": 0.5048518518518518,
                "std": 0.3118640338079888,
                "importance": 0.0020636446627121515
        },
        "autocorr_noise": {
                "mean": 0.4860621969175202,
                "std": 0.2191819662300171,
                "importance": 0.0057917076263557755
        },
        "length_complexity_corr": {
                "mean": 0.4962758215212294,
                "std": 0.18278153326471722,
                "importance": 0.006033762928241152
        },
        "vocab_structure_corr": {
                "mean": 0.48906764635137234,
                "std": 0.18062687331272204,
                "importance": 0.00937215624873859
        },
        "position_length_corr": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "correlation_break_score": {
                "mean": 0.2667614152520778,
                "std": 0.17471777532869867,
                "importance": 0.006318503878716689
        },
        "synonym_cluster_usage": {
                "mean": 0.013955535094720197,
                "std": 0.06052644046579105,
                "importance": 0.0033420015009991044
        },
        "rare_synonym_ratio": {
                "mean": 0.05937217372465683,
                "std": 0.21750860001859218,
                "importance": 0.002102858018787615
        },
        "sophistication_jumps": {
                "mean": 0.007244444444444445,
                "std": 0.059747303260595616,
                "importance": 0.0003034371169352637
        },
        "formal_informal_mix": {
                "mean": 0.5789186317645967,
                "std": 0.18803952748796438,
                "importance": 0.000985473458925318
        },
        "register_consistency": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "word_choice_naturalness": {
                "mean": 0.9927555555555555,
                "std": 0.059747303260595616,
                "importance": 0.0005304264505451205
        },
        "contraction_rate": {
                "mean": 0.11950005712206312,
                "std": 0.13119043964908078,
                "importance": 0.03777284826628875
        },
        "contraction_uniformity": {
                "mean": 0.1892267821662673,
                "std": 0.240360048413103,
                "importance": 0.013053328025467242
        },
        "contraction_position_variance": {
                "mean": 0.2136964270555318,
                "std": 0.2639459540736797,
                "importance": 0.04691524791895944
        },
        "contraction_context_fit": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "sentence_start_diversity": {
                "mean": 0.7344856118103013,
                "std": 0.19954638167216493,
                "importance": 0.07807347409466497
        },
        "sentence_start_entropy": {
                "mean": 0.7864310976972535,
                "std": 0.1952146205272528,
                "importance": 0.10848109869865986
        },
        "template_score": {
                "mean": 0.6007222153256869,
                "std": 0.21945618103665412,
                "importance": 0.03358913486820012
        },
        "parallelism_score": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "clause_depth_variance": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "embedding_naturalness": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "sophistication_variance": {
                "mean": 0.09781927811910868,
                "std": 0.11461886258445869,
                "importance": 0.007772796668627812
        },
        "sophistication_autocorr": {
                "mean": 0.44208802557302573,
                "std": 0.15721016441288496,
                "importance": 0.006287305276048519
        },
        "word_choice_consistency": {
                "mean": 0.9021807218808914,
                "std": 0.11461886258445869,
                "importance": 0.007097598430315715
        },
        "formality_stability": {
                "mean": 0.9021807218808914,
                "std": 0.11461886258445869,
                "importance": 0.007379509960185097
        },
        "bigram_predictability": {
                "mean": 0.09282919591624955,
                "std": 0.08484805175826658,
                "importance": 0.017063966797244777
        },
        "trigram_predictability": {
                "mean": 0.03904933168874887,
                "std": 0.07361407675552681,
                "importance": 0.0338904332480261
        },
        "ngram_surprise_variance": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "phrase_originality": {
                "mean": 0.960950668311251,
                "std": 0.07361407675552681,
                "importance": 0.03361661020346406
        },
        "comma_density": {
                "mean": 0.27916883399629605,
                "std": 0.14453530144485308,
                "importance": 0.05691369805573522
        },
        "punctuation_variety": {
                "mean": 0.49611717171717173,
                "std": 0.17505445000713138,
                "importance": 0.028168098506537634
        },
        "punctuation_entropy": {
                "mean": 0.556690289654715,
                "std": 0.13621766036861072,
                "importance": 0.016503551881772875
        },
        "semicolon_colon_ratio": {
                "mean": 0.05523776982977041,
                "std": 0.16373543916201944,
                "importance": 0.0017595131226105403
        },
        "transition_density": {
                "mean": 0.05015661137627886,
                "std": 0.1379796765929781,
                "importance": 0.003605037131705603
        },
        "hedging_density": {
                "mean": 0.01572553441998771,
                "std": 0.09196855761301558,
                "importance": 0.0007811455990958785
        },
        "discourse_variety": {
                "mean": 0.010443010752688171,
                "std": 0.04585863371323473,
                "importance": 0.0012752203506146162
        },
        "ai_phrase_density": {
                "mean": 0.01902561973579901,
                "std": 0.10417813322430965,
                "importance": 0.0010122186791432558
        },
        "lexical_entropy": {
                "mean": 0.9351497463867052,
                "std": 0.044359771643130555,
                "importance": 0.021847588577487604
        },
        "sentence_entropy": {
                "mean": 0.9731362615469183,
                "std": 0.05163154050676125,
                "importance": 0.007385799073119236
        },
        "entropy_stability": {
                "mean": 0.8381319781788528,
                "std": 0.13681956284051366,
                "importance": 0.015984066026335637
        },
        "char_entropy": {
                "mean": 0.8748826284655467,
                "std": 0.03535557242095772,
                "importance": 0.018926473351657418
        },
        "word_position_entropy": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "perplexity_proxy": {
                "mean": 0.9351497463867052,
                "std": 0.044359771643130555,
                "importance": 0.020565658410217263
        },
        "char_repeat_ratio": {
                "mean": 0.37279185448574126,
                "std": 0.14140640362090187,
                "importance": 0.011112163777352461
        },
        "whitespace_consistency": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "case_pattern_entropy": {
                "mean": 0.04260143005174673,
                "std": 0.04561415029113161,
                "importance": 0.028481716514280317
        },
        "special_char_density": {
                "mean": 0.33806137433778866,
                "std": 0.13450928972625026,
                "importance": 0.02194972458062653
        },
        "rhythm_variance": {
                "mean": 0.8369871063624871,
                "std": 0.2620866145640299,
                "importance": 0.05564230411418902
        },
        "stress_pattern_entropy": {
                "mean": 0.615442052671103,
                "std": 0.2366334563893437,
                "importance": 0.030689920974599372
        },
        "syllable_rhythm": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "reading_flow_score": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        },
        "topic_consistency": {
                "mean": 0.22511517973031514,
                "std": 0.09027954289602849,
                "importance": 0.013797021371306704
        },
        "reference_density": {
                "mean": 0.7617709546467726,
                "std": 0.11944646751049194,
                "importance": 0.007239467865215701
        },
        "connector_appropriateness": {
                "mean": 0.37670375005201634,
                "std": 0.23040426394911123,
                "importance": 0.07310531891099574
        },
        "semantic_flow": {
                "mean": 0.5,
                "std": 0.0,
                "importance": 0.0
        }
},
    
    thresholds: {
        "variance_of_variance": {
                "threshold": 0.1,
                "human_mean": 0.937562488136438,
                "humanized_mean": 0.8620175623341202
        },
        "variance_stability": {
                "threshold": 0.9,
                "human_mean": 0.3619670014558649,
                "humanized_mean": 0.3984382968632102
        },
        "local_var_consistency": {
                "threshold": 0.9,
                "human_mean": 0.3619670014558649,
                "humanized_mean": 0.3984382968632102
        },
        "word_var_of_var": {
                "threshold": 0.1,
                "human_mean": 0.6757585441984063,
                "humanized_mean": 0.6843615533387686
        },
        "syllable_variance": {
                "threshold": 0.1,
                "human_mean": 0.3326939885010312,
                "humanized_mean": 0.4407725888515694
        },
        "length_distribution_uniformity": {
                "threshold": 0.9,
                "human_mean": 0.3362703165075154,
                "humanized_mean": 0.3111382835318054
        },
        "autocorr_lag1": {
                "threshold": 0.9,
                "human_mean": 0.4729890819496978,
                "humanized_mean": 0.4602708228866811
        },
        "autocorr_lag2": {
                "threshold": 0.9,
                "human_mean": 0.46431691339480685,
                "humanized_mean": 0.4570586347163706
        },
        "autocorr_decay": {
                "threshold": 0.1,
                "human_mean": 0.5157540731257261,
                "humanized_mean": 0.5121681028231188
        },
        "autocorr_flatness": {
                "threshold": 0.9,
                "human_mean": 0.49420474007472304,
                "humanized_mean": 0.43227332867421897
        },
        "autocorr_periodicity": {
                "threshold": 0.1,
                "human_mean": 0.5008222222222222,
                "humanized_mean": 0.5088814814814814
        },
        "autocorr_noise": {
                "threshold": 0.1,
                "human_mean": 0.46262605862605743,
                "humanized_mean": 0.5094983352089829
        },
        "length_complexity_corr": {
                "threshold": 0.9,
                "human_mean": 0.505239060472014,
                "humanized_mean": 0.4873125825704449
        },
        "vocab_structure_corr": {
                "threshold": 0.9,
                "human_mean": 0.47155201999136154,
                "humanized_mean": 0.5065832727113831
        },
        "position_length_corr": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "correlation_break_score": {
                "threshold": 0.7000000000000001,
                "human_mean": 0.2599012229952272,
                "humanized_mean": 0.2736216075089284
        },
        "synonym_cluster_usage": {
                "threshold": 0.85,
                "human_mean": 0.009752322653505148,
                "humanized_mean": 0.018158747535935243
        },
        "rare_synonym_ratio": {
                "threshold": 0.8,
                "human_mean": 0.031081278666244876,
                "humanized_mean": 0.08766306878306877
        },
        "sophistication_jumps": {
                "threshold": 0.55,
                "human_mean": 0.007022222222222222,
                "humanized_mean": 0.007466666666666667
        },
        "formal_informal_mix": {
                "threshold": 0.1,
                "human_mean": 0.6061631894551194,
                "humanized_mean": 0.5516740740740741
        },
        "register_consistency": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "word_choice_naturalness": {
                "threshold": 0.1,
                "human_mean": 0.9929777777777777,
                "humanized_mean": 0.9925333333333334
        },
        "contraction_rate": {
                "threshold": 0.9,
                "human_mean": 0.17463952509882752,
                "humanized_mean": 0.0643605891452987
        },
        "contraction_uniformity": {
                "threshold": 0.9,
                "human_mean": 0.11439263786974099,
                "humanized_mean": 0.26406092646279355
        },
        "contraction_position_variance": {
                "threshold": 0.9,
                "human_mean": 0.3327746558939659,
                "humanized_mean": 0.09461819821709776
        },
        "contraction_context_fit": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "sentence_start_diversity": {
                "threshold": 0.1,
                "human_mean": 0.8319386714305995,
                "humanized_mean": 0.6370325521900032
        },
        "sentence_start_entropy": {
                "threshold": 0.1,
                "human_mean": 0.8852339784291015,
                "humanized_mean": 0.6876282169654059
        },
        "template_score": {
                "threshold": 0.1,
                "human_mean": 0.5247568257072983,
                "humanized_mean": 0.6766876049440755
        },
        "parallelism_score": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "clause_depth_variance": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "embedding_naturalness": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "sophistication_variance": {
                "threshold": 0.9,
                "human_mean": 0.10447377781895531,
                "humanized_mean": 0.09116477841926207
        },
        "sophistication_autocorr": {
                "threshold": 0.9,
                "human_mean": 0.45536387697346475,
                "humanized_mean": 0.42881217417258666
        },
        "word_choice_consistency": {
                "threshold": 0.1,
                "human_mean": 0.8955262221810447,
                "humanized_mean": 0.9088352215807379
        },
        "formality_stability": {
                "threshold": 0.1,
                "human_mean": 0.8955262221810447,
                "humanized_mean": 0.9088352215807379
        },
        "bigram_predictability": {
                "threshold": 0.75,
                "human_mean": 0.07126253049494045,
                "humanized_mean": 0.11439586133755862
        },
        "trigram_predictability": {
                "threshold": 0.7000000000000001,
                "human_mean": 0.02167130223133673,
                "humanized_mean": 0.05642736114616099
        },
        "ngram_surprise_variance": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "phrase_originality": {
                "threshold": 0.1,
                "human_mean": 0.9783286977686632,
                "humanized_mean": 0.943572638853839
        },
        "comma_density": {
                "threshold": 0.9,
                "human_mean": 0.23196341655092445,
                "humanized_mean": 0.32637425144166765
        },
        "punctuation_variety": {
                "threshold": 0.1,
                "human_mean": 0.5642464646464647,
                "humanized_mean": 0.42798787878787875
        },
        "punctuation_entropy": {
                "threshold": 0.9,
                "human_mean": 0.5989558184657585,
                "humanized_mean": 0.5144247608436715
        },
        "semicolon_colon_ratio": {
                "threshold": 0.9,
                "human_mean": 0.06835464820531605,
                "humanized_mean": 0.04212089145422478
        },
        "transition_density": {
                "threshold": 0.9,
                "human_mean": 0.039344434343020124,
                "humanized_mean": 0.0609687884095376
        },
        "hedging_density": {
                "threshold": 0.9,
                "human_mean": 0.026205124802170368,
                "humanized_mean": 0.005245944037805056
        },
        "discourse_variety": {
                "threshold": 0.55,
                "human_mean": 0.014106093189964157,
                "humanized_mean": 0.006779928315412185
        },
        "ai_phrase_density": {
                "threshold": 0.9,
                "human_mean": 0.027753374489712036,
                "humanized_mean": 0.010297864981885983
        },
        "lexical_entropy": {
                "threshold": 0.1,
                "human_mean": 0.9310622753558381,
                "humanized_mean": 0.9392372174175723
        },
        "sentence_entropy": {
                "threshold": 0.1,
                "human_mean": 0.9721903466887999,
                "humanized_mean": 0.9740821764050368
        },
        "entropy_stability": {
                "threshold": 0.1,
                "human_mean": 0.8618768709701421,
                "humanized_mean": 0.8143870853875634
        },
        "char_entropy": {
                "threshold": 0.1,
                "human_mean": 0.8692637529794224,
                "humanized_mean": 0.8805015039516711
        },
        "word_position_entropy": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "perplexity_proxy": {
                "threshold": 0.1,
                "human_mean": 0.9310622753558381,
                "humanized_mean": 0.9392372174175723
        },
        "char_repeat_ratio": {
                "threshold": 0.9,
                "human_mean": 0.3839135415909029,
                "humanized_mean": 0.36167016738057967
        },
        "whitespace_consistency": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "case_pattern_entropy": {
                "threshold": 0.85,
                "human_mean": 0.04017256426780253,
                "humanized_mean": 0.045030295835690924
        },
        "special_char_density": {
                "threshold": 0.1,
                "human_mean": 0.37846619067650616,
                "humanized_mean": 0.29765655799907115
        },
        "rhythm_variance": {
                "threshold": 0.1,
                "human_mean": 0.9304565349548609,
                "humanized_mean": 0.7435176777701132
        },
        "stress_pattern_entropy": {
                "threshold": 0.1,
                "human_mean": 0.6717529096617787,
                "humanized_mean": 0.5591311956804271
        },
        "syllable_rhythm": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "reading_flow_score": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        },
        "topic_consistency": {
                "threshold": 0.9,
                "human_mean": 0.2038203638738983,
                "humanized_mean": 0.24640999558673202
        },
        "reference_density": {
                "threshold": 0.1,
                "human_mean": 0.7650994755762232,
                "humanized_mean": 0.7584424337173219
        },
        "connector_appropriateness": {
                "threshold": 0.9,
                "human_mean": 0.47551674883497214,
                "humanized_mean": 0.27789075126906054
        },
        "semantic_flow": {
                "threshold": 0.1,
                "human_mean": 0.5,
                "humanized_mean": 0.5
        }
},
    
    topFeatures: [
        {
                "name": "sentence_start_entropy",
                "importance": 0.10848109869865986
        },
        {
                "name": "sentence_start_diversity",
                "importance": 0.07807347409466497
        },
        {
                "name": "connector_appropriateness",
                "importance": 0.07310531891099574
        },
        {
                "name": "comma_density",
                "importance": 0.05691369805573522
        },
        {
                "name": "rhythm_variance",
                "importance": 0.05564230411418902
        },
        {
                "name": "syllable_variance",
                "importance": 0.050125069902481535
        },
        {
                "name": "contraction_position_variance",
                "importance": 0.04691524791895944
        },
        {
                "name": "contraction_rate",
                "importance": 0.03777284826628875
        },
        {
                "name": "trigram_predictability",
                "importance": 0.0338904332480261
        },
        {
                "name": "phrase_originality",
                "importance": 0.03361661020346406
        },
        {
                "name": "template_score",
                "importance": 0.03358913486820012
        },
        {
                "name": "stress_pattern_entropy",
                "importance": 0.030689920974599372
        },
        {
                "name": "case_pattern_entropy",
                "importance": 0.028481716514280317
        },
        {
                "name": "punctuation_variety",
                "importance": 0.028168098506537634
        },
        {
                "name": "special_char_density",
                "importance": 0.02194972458062653
        },
        {
                "name": "lexical_entropy",
                "importance": 0.021847588577487604
        },
        {
                "name": "perplexity_proxy",
                "importance": 0.020565658410217263
        },
        {
                "name": "char_entropy",
                "importance": 0.018926473351657418
        },
        {
                "name": "bigram_predictability",
                "importance": 0.017063966797244777
        },
        {
                "name": "punctuation_entropy",
                "importance": 0.016503551881772875
        }
],
    
    // Normalization parameters
    scalerMean: [0.8997900252352782, 0.3802026491595379, 0.3802026491595379, 0.6800600487685855, 0.3867332886763013, 0.32370430001967476, 0.46662995241818817, 0.4606877740555857, 0.5139610879744196, 0.46323903437446823, 0.5048518518518528, 0.48606219691751873, 0.49627582152123245, 0.4890676463513713, 0.5, 0.2667614152520756, 0.01395553509472021, 0.05937217372465684, 0.007244444444444445, 0.5789186317645965, 0.5, 0.9927555555555555, 0.1195000571220632, 0.18922678216626712, 0.21369642705552577, 0.5, 0.7344856118102964, 0.786431097697263, 0.6007222153256792, 0.5, 0.5, 0.5, 0.09781927811910808, 0.44208802557302607, 0.9021807218808914, 0.9021807218808914, 0.09282919591624864, 0.039049331688748246, 0.5, 0.9609506683112591, 0.279168833996291, 0.49611717171719655, 0.5566902896547128, 0.05523776982977045, 0.05015661137627881, 0.015725534419987714, 0.010443010752688008, 0.019025619735799006, 0.9351497463867062, 0.9731362615469881, 0.8381319781788567, 0.8748826284655487, 0.5, 0.9351497463867062, 0.37279185448573615, 0.5, 0.04260143005174635, 0.3380613743377869, 0.8369871063624864, 0.6154420526711034, 0.5, 0.5, 0.22511517973031614, 0.7617709546467728, 0.37670375005201123, 0.5],
    scalerStd: [0.26914712528526175, 0.2846018796065505, 0.2846018796065505, 0.2616124816627054, 0.13498230445891043, 0.27638800503773164, 0.13383275141257794, 0.14204203301625853, 0.21003683391774056, 0.3148171808756747, 0.3118640338079966, 0.21918196623001088, 0.18278153326471377, 0.18062687331272548, 1.0, 0.17471777532869434, 0.060526440465800684, 0.21750860001859165, 0.05974730326058964, 0.18803952748795977, 1.0, 0.05974730326058964, 0.13119043964905883, 0.24036004841309289, 0.26394595407362464, 1.0, 0.1995463816721603, 0.19521462052725144, 0.2194561810366587, 1.0, 1.0, 1.0, 0.11461886258445803, 0.15721016441288585, 0.1146188625844581, 0.1146188625844581, 0.08484805175826664, 0.07361407675552874, 1.0, 0.0736140767555296, 0.14453530144485374, 0.17505445000715386, 0.13621766036861163, 0.1637354391620317, 0.13797967659300195, 0.09196855761299315, 0.045858633713245577, 0.10417813322429552, 0.04435977164313058, 0.05163154050675994, 0.13681956284051155, 0.035355572420957534, 1.0, 0.04435977164313058, 0.14140640362090134, 1.0, 0.045614150291131556, 0.13450928972624945, 0.26208661456401766, 0.23663345638934793, 1.0, 1.0, 0.09027954289602859, 0.11944646751049184, 0.23040426394910504, 1.0],
};

if (typeof module !== 'undefined' && module.exports) {
    module.exports = FlareConfig;
}
