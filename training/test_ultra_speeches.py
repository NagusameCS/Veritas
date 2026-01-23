#!/usr/bin/env python3
"""
Test SUPERNOVA ULTRA on Model UN / Debate speeches
These were causing false positives (95%+ AI) in v3
"""

import pickle
import numpy as np
import sys
sys.path.insert(0, '/workspaces/Veritas/training')
from feature_extractor_v3 import FeatureExtractorV3

print("=" * 70)
print("SUPERNOVA ULTRA - Model UN / Debate Speech Test")
print("=" * 70)

# Load model
print("\n📦 Loading SUPERNOVA ULTRA...")
with open('/workspaces/Veritas/training/models/SupernovaUltra/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('/workspaces/Veritas/training/models/SupernovaUltra/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
extractor = FeatureExtractorV3()
print("  ✓ Model loaded!")

# Test samples - formal human writing that often triggers false positives
test_samples = [
    {
        "name": "Model UN Opening Statement",
        "text": """Distinguished delegates, honorable chairs, and esteemed members of this assembly,

The delegation of the Federal Republic of Germany rises today to address the pressing matter before this committee. As we convene to discuss international climate policy, we must recognize that the challenge of climate change demands nothing less than unprecedented global cooperation.

Germany has demonstrated its commitment to environmental stewardship through concrete actions. Our Energiewende policy has positioned us as a leader in renewable energy adoption. We have committed to reducing greenhouse gas emissions by 65% by 2030 compared to 1990 levels, and achieving climate neutrality by 2045.

However, the delegation wishes to emphasize that climate action cannot come at the expense of developing nations. We must ensure that our policies incorporate principles of climate justice and provide adequate support for those most vulnerable to climate impacts.

Therefore, Germany proposes a three-pillar approach: first, enhanced technology transfer mechanisms; second, increased climate financing for adaptation and mitigation in developing countries; and third, strengthened monitoring and verification frameworks.

The delegation urges all member states to consider these proposals carefully. The time for incremental measures has passed. We must act with the urgency that this crisis demands.

Thank you, and Germany yields the floor back to the chair.""",
        "expected": "HUMAN"
    },
    {
        "name": "Academic Debate Rebuttal",
        "text": """Ladies and gentlemen, the opposition's argument fundamentally mischaracterizes our position. Allow me to address their points systematically.

First, they claim that economic development necessarily precedes environmental protection. This represents a false dichotomy that ignores decades of evidence. The Environmental Kuznets Curve, which they implicitly invoke, has been thoroughly debunked by recent scholarship. Stern's 2007 review demonstrated that the costs of inaction far exceed the costs of proactive environmental policy.

Second, the opposition suggests that developing nations cannot afford green technology. This ignores the precipitous decline in renewable energy costs. Solar photovoltaic prices have fallen 89% since 2010, according to IRENA data. Moreover, the health co-benefits of reduced air pollution alone justify immediate action—the Lancet Commission estimates that air pollution causes 9 million premature deaths annually.

Third, and most critically, the opposition's framework assumes we have time for gradual transition. We do not. The IPCC's 2021 report makes clear that limiting warming to 1.5°C requires halving global emissions by 2030. This is not alarmism; this is scientific consensus.

The motion must stand because the alternative—continued delay—is unconscionable. We cannot mortgage our children's future for short-term economic convenience. Thank you.""",
        "expected": "HUMAN"
    },
    {
        "name": "Student Essay Introduction",
        "text": """The question of whether artificial intelligence will fundamentally transform human society is no longer hypothetical—it is happening now. As I write this essay in my dorm room, algorithms are making decisions that affect millions: who gets a loan, who gets hired, who sees which news. The implications are profound and demand careful analysis.

In this paper, I will argue that while AI offers tremendous benefits, we are woefully unprepared for its societal implications. I'm not saying we should halt AI development—that would be both impossible and unwise. Rather, I'm arguing that we need robust governance frameworks that can keep pace with technological change.

My argument proceeds in three parts. First, I'll examine the current state of AI capabilities and their real-world applications, drawing on case studies from healthcare, criminal justice, and employment. Second, I'll analyze the regulatory landscape, highlighting critical gaps in existing frameworks. Finally, I'll propose a set of principles for AI governance that balance innovation with accountability.

Some might dismiss these concerns as premature or overblown. After all, we've navigated technological transitions before. But AI is different—it's the first technology that can improve itself, that can learn without explicit programming, that can in some domains exceed human cognitive capabilities. This demands a new approach to governance.

Let me begin with a story. Last summer, I interned at a tech company where I saw firsthand how algorithmic decisions affected real people...""",
        "expected": "HUMAN"
    },
    {
        "name": "Policy Brief Summary",
        "text": """Executive Summary

This policy brief examines the feasibility and implications of implementing a universal basic income (UBI) in the United States. Our analysis draws on evidence from pilot programs in Finland, Kenya, and Stockton, California, as well as economic modeling by researchers at the Roosevelt Institute.

Key Findings:

1. Existing pilot programs show modest positive effects on mental health and job searching behavior, with minimal impact on labor force participation.

2. A UBI of $1,000 per month for all adult citizens would cost approximately $3.5 trillion annually—roughly 16% of GDP.

3. Various funding mechanisms exist, including carbon taxes, value-added taxes, wealth taxes, and consolidation of existing transfer programs.

4. Implementation challenges include political feasibility, potential inflationary effects, and interaction with existing benefits programs.

Our assessment is that while UBI represents a promising approach to addressing economic insecurity in an era of technological disruption, significant questions remain about optimal program design and funding mechanisms. We recommend that policymakers pursue expanded pilot programs to generate additional evidence before considering national implementation.

This brief was prepared by the Policy Research Division at the request of the Economic Security Subcommittee.""",
        "expected": "HUMAN"
    },
    {
        "name": "ChatGPT Response",
        "text": """Artificial intelligence has become an increasingly important topic in today's rapidly evolving technological landscape. As we continue to make advancements in machine learning and natural language processing, it's essential to consider both the potential benefits and challenges that AI presents to society.

One of the primary advantages of AI is its ability to automate repetitive tasks and improve efficiency across various industries. From healthcare to finance, AI systems are helping organizations process large amounts of data and make more informed decisions. This technological transformation has the potential to revolutionize how we work and live.

However, it's important to acknowledge the ethical considerations surrounding AI development. Issues such as bias in algorithms, privacy concerns, and the potential displacement of workers require careful attention. As we move forward, it will be crucial to develop robust frameworks that ensure AI is developed and deployed responsibly.

In conclusion, while artificial intelligence offers tremendous opportunities for innovation and progress, we must approach its development thoughtfully. By balancing innovation with ethical considerations, we can harness the power of AI while minimizing potential risks. The future of AI depends on our collective commitment to responsible development and deployment.""",
        "expected": "AI"
    },
]

print("\n📊 Testing samples...")
print("-" * 70)

for sample in test_samples:
    # Extract features
    features = extractor.extract_features(sample["text"])
    X = np.array([list(features.values())])
    X_scaled = scaler.transform(X)
    
    # Predict
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]
    
    ai_prob = prob[1] * 100
    human_prob = prob[0] * 100
    
    prediction = "AI" if pred == 1 else "HUMAN"
    expected = sample["expected"]
    correct = "✅" if prediction == expected else "❌"
    
    print(f"\n📝 {sample['name']}")
    print(f"   Expected: {expected}")
    print(f"   Predicted: {prediction} {correct}")
    print(f"   AI Probability: {ai_prob:.1f}%")
    print(f"   Human Probability: {human_prob:.1f}%")

print("\n" + "=" * 70)
print("TEST COMPLETE!")
print("=" * 70)
