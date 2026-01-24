/**
 * VERITAS — Public AI Model Benchmarks
 * =====================================
 * Comprehensive benchmark data for all major AI text generators
 * Sources: Official documentation, research papers, third-party evaluations
 * Last Updated: January 2025
 */

const AIModelBenchmarks = {
    version: '2.0.0',
    lastUpdated: '2025-01',
    
    // ========================================================================
    // OPENAI MODELS
    // ========================================================================
    openai: {
        'gpt-4-turbo': {
            name: 'GPT-4 Turbo',
            released: '2024-04',
            contextWindow: 128000,
            outputLimit: 4096,
            benchmarks: {
                mmlu: 86.4,
                humaneval: 87.1,
                math: 52.9,
                gpqa: 49.1,
                hellaswag: 95.3
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Highly coherent', 'Strong reasoning', 'Follows instructions well']
        },
        'gpt-4o': {
            name: 'GPT-4o',
            released: '2024-05',
            contextWindow: 128000,
            outputLimit: 16384,
            benchmarks: {
                mmlu: 87.2,
                humaneval: 90.2,
                math: 76.6,
                gpqa: 53.6,
                hellaswag: 96.0
            },
            detectionDifficulty: 'High',
            characteristics: ['Multimodal', 'Fast inference', 'Natural conversational style']
        },
        'gpt-4o-mini': {
            name: 'GPT-4o Mini',
            released: '2024-07',
            contextWindow: 128000,
            outputLimit: 16384,
            benchmarks: {
                mmlu: 82.0,
                humaneval: 87.0,
                math: 70.2
            },
            detectionDifficulty: 'Medium',
            characteristics: ['Cost-effective', 'Fast', 'Good for most tasks']
        },
        'gpt-3.5-turbo': {
            name: 'GPT-3.5 Turbo',
            released: '2023-03',
            contextWindow: 16385,
            outputLimit: 4096,
            benchmarks: {
                mmlu: 70.0,
                humaneval: 48.1,
                math: 34.1
            },
            detectionDifficulty: 'Low-Medium',
            characteristics: ['More formulaic', 'Occasional inconsistencies', 'Good baseline']
        },
        'o1': {
            name: 'o1',
            released: '2024-12',
            contextWindow: 200000,
            outputLimit: 100000,
            benchmarks: {
                mmlu: 91.8,
                gpqa: 78.0,
                aime: 83.3,
                math: 96.4
            },
            detectionDifficulty: 'Very High',
            characteristics: ['Reasoning model', 'Step-by-step thinking', 'Highly analytical']
        },
        'o1-mini': {
            name: 'o1-mini',
            released: '2024-09',
            contextWindow: 128000,
            outputLimit: 65536,
            benchmarks: {
                aime: 70.0,
                math: 90.0
            },
            detectionDifficulty: 'High',
            characteristics: ['Fast reasoning', 'Optimized for STEM', 'Concise outputs']
        }
    },
    
    // ========================================================================
    // ANTHROPIC MODELS
    // ========================================================================
    anthropic: {
        'claude-3.5-sonnet': {
            name: 'Claude 3.5 Sonnet',
            released: '2024-06',
            contextWindow: 200000,
            outputLimit: 8192,
            benchmarks: {
                mmlu: 88.7,
                humaneval: 92.0,
                math: 71.1,
                gpqa: 59.4
            },
            detectionDifficulty: 'High',
            characteristics: ['Balanced intelligence', 'Natural writing', 'Strong coding']
        },
        'claude-3-opus': {
            name: 'Claude 3 Opus',
            released: '2024-03',
            contextWindow: 200000,
            outputLimit: 4096,
            benchmarks: {
                mmlu: 86.8,
                humaneval: 84.9,
                math: 60.1,
                gpqa: 50.4
            },
            detectionDifficulty: 'High',
            characteristics: ['Deep analysis', 'Creative writing', 'Nuanced responses']
        },
        'claude-3-haiku': {
            name: 'Claude 3 Haiku',
            released: '2024-03',
            contextWindow: 200000,
            outputLimit: 4096,
            benchmarks: {
                mmlu: 75.2,
                humaneval: 75.9,
                math: 38.9
            },
            detectionDifficulty: 'Medium',
            characteristics: ['Fast', 'Concise', 'Cost-effective']
        }
    },
    
    // ========================================================================
    // GOOGLE MODELS
    // ========================================================================
    google: {
        'gemini-1.5-pro': {
            name: 'Gemini 1.5 Pro',
            released: '2024-02',
            contextWindow: 2000000,
            outputLimit: 8192,
            benchmarks: {
                mmlu: 85.9,
                humaneval: 84.1,
                math: 58.5,
                hellaswag: 92.5
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Long context', 'Multimodal', 'Strong reasoning']
        },
        'gemini-2.0-flash': {
            name: 'Gemini 2.0 Flash',
            released: '2024-12',
            contextWindow: 1000000,
            outputLimit: 8192,
            benchmarks: {
                mmlu: 82.0,
                humaneval: 82.0,
                livecodebench: 32.1
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Agentic capabilities', 'Native tool use', 'Fast']
        },
        'gemini-1.0-pro': {
            name: 'Gemini 1.0 Pro',
            released: '2023-12',
            contextWindow: 32000,
            outputLimit: 8192,
            benchmarks: {
                mmlu: 71.8,
                humaneval: 67.7
            },
            detectionDifficulty: 'Medium',
            characteristics: ['Reliable', 'Good general purpose', 'Efficient']
        }
    },
    
    // ========================================================================
    // META MODELS (LLAMA)
    // ========================================================================
    meta: {
        'llama-3.3-70b': {
            name: 'Llama 3.3 70B',
            released: '2024-12',
            contextWindow: 128000,
            parameters: '70B',
            benchmarks: {
                mmlu: 86.0,
                humaneval: 88.4,
                math: 77.0
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Open weights', 'Strong performance', 'Instruction tuned']
        },
        'llama-3.1-405b': {
            name: 'Llama 3.1 405B',
            released: '2024-07',
            contextWindow: 128000,
            parameters: '405B',
            benchmarks: {
                mmlu: 88.6,
                humaneval: 89.0,
                math: 73.8
            },
            detectionDifficulty: 'High',
            characteristics: ['Largest open model', 'State-of-the-art open', 'Multilingual']
        },
        'llama-3-8b': {
            name: 'Llama 3 8B',
            released: '2024-04',
            contextWindow: 8192,
            parameters: '8B',
            benchmarks: {
                mmlu: 66.6,
                humaneval: 62.2
            },
            detectionDifficulty: 'Low-Medium',
            characteristics: ['Small footprint', 'Fast', 'Local deployment']
        }
    },
    
    // ========================================================================
    // MISTRAL MODELS
    // ========================================================================
    mistral: {
        'mistral-large': {
            name: 'Mistral Large',
            released: '2024-02',
            contextWindow: 128000,
            benchmarks: {
                mmlu: 84.0,
                humaneval: 73.0,
                math: 45.0
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Multilingual', 'Strong reasoning', 'Function calling']
        },
        'mixtral-8x22b': {
            name: 'Mixtral 8x22B',
            released: '2024-04',
            contextWindow: 65536,
            parameters: '141B (22B active)',
            benchmarks: {
                mmlu: 77.8,
                humaneval: 75.0
            },
            detectionDifficulty: 'Medium',
            characteristics: ['MoE architecture', 'Open weights', 'Efficient']
        },
        'mistral-7b': {
            name: 'Mistral 7B',
            released: '2023-09',
            contextWindow: 32768,
            parameters: '7B',
            benchmarks: {
                mmlu: 62.5,
                humaneval: 26.2
            },
            detectionDifficulty: 'Low-Medium',
            characteristics: ['Very fast', 'Small', 'Apache license']
        }
    },
    
    // ========================================================================
    // COHERE MODELS
    // ========================================================================
    cohere: {
        'command-r-plus': {
            name: 'Command R+',
            released: '2024-04',
            contextWindow: 128000,
            benchmarks: {
                mmlu: 75.7,
                humaneval: 71.0
            },
            detectionDifficulty: 'Medium',
            characteristics: ['RAG optimized', 'Enterprise focus', 'Multilingual']
        },
        'command-r': {
            name: 'Command R',
            released: '2024-03',
            contextWindow: 128000,
            benchmarks: {
                mmlu: 68.2
            },
            detectionDifficulty: 'Medium',
            characteristics: ['Long context', 'Cost-effective', 'Business applications']
        }
    },
    
    // ========================================================================
    // DEEPSEEK MODELS
    // ========================================================================
    deepseek: {
        'deepseek-v3': {
            name: 'DeepSeek V3',
            released: '2024-12',
            contextWindow: 128000,
            parameters: '671B MoE',
            benchmarks: {
                mmlu: 88.5,
                humaneval: 82.6,
                math: 90.2,
                aime: 39.2
            },
            detectionDifficulty: 'High',
            characteristics: ['MoE architecture', 'Extremely cost-effective', 'Strong math/code']
        },
        'deepseek-coder-v2': {
            name: 'DeepSeek Coder V2',
            released: '2024-06',
            contextWindow: 128000,
            parameters: '236B MoE',
            benchmarks: {
                humaneval: 90.2,
                livecodebench: 43.4
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Code specialized', 'Open weights', 'Strong performance']
        }
    },
    
    // ========================================================================
    // ALIBABA MODELS (QWEN)
    // ========================================================================
    alibaba: {
        'qwen-2.5-72b': {
            name: 'Qwen 2.5 72B',
            released: '2024-09',
            contextWindow: 131072,
            parameters: '72B',
            benchmarks: {
                mmlu: 86.1,
                humaneval: 86.6,
                math: 83.1
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Multilingual', 'Long context', 'Open weights']
        },
        'qwen-2.5-coder-32b': {
            name: 'Qwen 2.5 Coder 32B',
            released: '2024-11',
            contextWindow: 131072,
            parameters: '32B',
            benchmarks: {
                humaneval: 92.7,
                livecodebench: 40.0
            },
            detectionDifficulty: 'Medium-High',
            characteristics: ['Code specialized', 'State-of-the-art coding', 'Open']
        }
    },
    
    // ========================================================================
    // XAI MODELS
    // ========================================================================
    xai: {
        'grok-2': {
            name: 'Grok 2',
            released: '2024-08',
            contextWindow: 128000,
            benchmarks: {
                mmlu: 87.5,
                humaneval: 88.4,
                math: 76.1
            },
            detectionDifficulty: 'High',
            characteristics: ['Real-time information', 'Conversational', 'Frontier-class']
        }
    },
    
    // ========================================================================
    // DETECTION DIFFICULTY GUIDE
    // ========================================================================
    detectionGuide: {
        'Very High': {
            description: 'Extremely difficult to detect with statistical methods alone',
            examples: ['o1', 'Claude 3.5 Sonnet', 'DeepSeek V3'],
            recommendedApproach: 'Combine multiple heuristics, look for reasoning patterns'
        },
        'High': {
            description: 'Challenging to detect, requires careful analysis',
            examples: ['GPT-4o', 'Llama 3.1 405B', 'Grok 2'],
            recommendedApproach: 'Statistical analysis combined with stylistic markers'
        },
        'Medium-High': {
            description: 'Detectable with good accuracy using multiple signals',
            examples: ['GPT-4 Turbo', 'Gemini 1.5 Pro', 'Mistral Large'],
            recommendedApproach: 'Standard detection methods work well'
        },
        'Medium': {
            description: 'Moderately detectable with standard approaches',
            examples: ['GPT-4o Mini', 'Claude 3 Haiku', 'Mixtral'],
            recommendedApproach: 'Basic statistical features often sufficient'
        },
        'Low-Medium': {
            description: 'Easier to detect, more obvious patterns',
            examples: ['GPT-3.5 Turbo', 'Llama 3 8B', 'Mistral 7B'],
            recommendedApproach: 'Simple statistical checks work'
        }
    },
    
    // ========================================================================
    // BENCHMARK DESCRIPTIONS
    // ========================================================================
    benchmarkInfo: {
        'mmlu': {
            name: 'MMLU (Massive Multitask Language Understanding)',
            description: '57 subjects including STEM, humanities, social sciences',
            maxScore: 100,
            humanBaseline: 89.8
        },
        'humaneval': {
            name: 'HumanEval',
            description: 'Python coding problems measuring code generation',
            maxScore: 100,
            humanBaseline: 'N/A'
        },
        'math': {
            name: 'MATH',
            description: 'Competition-level mathematics problems',
            maxScore: 100,
            humanBaseline: 40
        },
        'gpqa': {
            name: 'GPQA (Graduate-Level Google-Proof Q&A)',
            description: 'PhD-level science questions',
            maxScore: 100,
            humanBaseline: 65
        },
        'hellaswag': {
            name: 'HellaSwag',
            description: 'Sentence completion for commonsense reasoning',
            maxScore: 100,
            humanBaseline: 95.7
        },
        'aime': {
            name: 'AIME (American Invitational Mathematics Examination)',
            description: 'Challenging high school math competition',
            maxScore: 100,
            humanBaseline: 11.1
        }
    },
    
    // ========================================================================
    // UTILITY METHODS
    // ========================================================================
    
    /**
     * Get all models sorted by detection difficulty
     */
    getByDetectionDifficulty(difficulty) {
        const results = [];
        for (const [provider, models] of Object.entries(this)) {
            if (typeof models !== 'object' || !models) continue;
            for (const [id, model] of Object.entries(models)) {
                if (model.detectionDifficulty === difficulty) {
                    results.push({ provider, id, ...model });
                }
            }
        }
        return results;
    },
    
    /**
     * Get model by name
     */
    getModel(modelId) {
        for (const [provider, models] of Object.entries(this)) {
            if (typeof models !== 'object' || !models) continue;
            if (models[modelId]) {
                return { provider, ...models[modelId] };
            }
        }
        return null;
    },
    
    /**
     * Get all models as flat array
     */
    getAllModels() {
        const results = [];
        for (const [provider, models] of Object.entries(this)) {
            if (typeof models !== 'object' || !models) continue;
            if (['detectionGuide', 'benchmarkInfo', 'version', 'lastUpdated'].includes(provider)) continue;
            for (const [id, model] of Object.entries(models)) {
                results.push({ provider, id, ...model });
            }
        }
        return results;
    }
};

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AIModelBenchmarks;
}
if (typeof window !== 'undefined') {
    window.AIModelBenchmarks = AIModelBenchmarks;
}