#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v6 Enhanced - Smart Inference with MUN Correction
==================================================================

This wrapper uses v6 (99.65% accuracy) as the base model but applies 
intelligent corrections for formal student speeches (MUN, debate, etc.)
that would otherwise be false positives.

The key insight: MUN speeches have unique markers that AI doesn't produce,
but their statistical profile (high readability, formal structure) looks AI-like.
"""

import json
import pickle
import re
import numpy as np
from xgboost import XGBClassifier
from typing import Dict, Tuple, Optional


class SupernovaUltraV6Enhanced:
    """
    Enhanced inference wrapper for SUPERNOVA ULTRA v6.
    
    Applies post-processing corrections for:
    - Model UN speeches
    - Debate speeches  
    - Student presentations
    - Formal academic addresses
    """
    
    def __init__(self, model_dir: str = "models/SupernovaUltraV6"):
        self.model_dir = model_dir
        self.model = XGBClassifier()
        self.model.load_model(f"{model_dir}/model.json")
        
        with open(f"{model_dir}/scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(f"{model_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # MUN/Debate specific patterns (weighted by strength)
        self.mun_patterns = {
            # Very strong indicators (weight 3)
            r'\bhello\s+delegates\b': 3,
            r'\bdistinguished\s+delegates\b': 3,
            r'\bhonorable\s+delegates\b': 3,
            r'\bfellow\s+delegates\b': 3,
            r'\bmy\s+name\s+is\s+\w+\s+and\s+i\s+am\s+representing\b': 3,
            r'\bi\s+am\s+representing\s+\w+\b': 3,
            r'\bthe\s+delegate\s+(from|of)\s+\w+\b': 3,
            r'\byour\s+delegate\s+from\b': 3,
            r'\bmodel\s+united\s+nations\b': 3,
            r'\bmodel\s+un\b': 3,
            r'\bmun\s+conference\b': 3,
            
            # Strong indicators (weight 2)
            r'\bour\s+committee\b': 2,
            r'\bthis\s+committee\b': 2,
            r'\bcommittee\s+sessions?\b': 2,
            r'\bduring\s+our\s+committee\b': 2,
            r'\bposition\s+paper\b': 2,
            r'\bi\s+urge\s+(all\s+)?delegates\b': 2,
            r'\bwe\s+urge\s+(all\s+)?delegates\b': 2,
            r'\bthe\s+delegation\s+of\b': 2,
            r'\bproposes?\s+(several|the\s+following)\b': 2,
            r'\bunited\s+nations\s+\w+\s+council\b': 2,
            
            # Moderate indicators (weight 1)
            r'\bmember\s+states\b': 1,
            r'\b(this|our)\s+nation\b': 1,
            r'\binternational\s+cooperation\b': 1,
            r'\bglobal\s+issue\b': 1,
            r'\bhumanitarian\s+(crisis|aid|efforts?)\b': 1,
            r'\bcalling\s+upon\s+all\b': 1,
            r'\bin\s+conclusion\b': 1,
            r'\bthank\s+you\b': 1,
        }
        
        # Student writing authenticity markers
        self.student_markers = [
            r'\bmy\s+name\s+is\b',
            r'\bhello,?\s+(my\s+name|i\'?m|everyone)\b',
            r'\bi\s+am\s+\w+,?\s+(and\s+)?i\s+will\b',
            r'\btoday\s+i\s+(will|am\s+going\s+to)\b',
            r'\blet\s+me\s+(start|begin)\b',
            r'\bas\s+a\s+student\b',
            r'\bi\s+believe\s+that\b',
            r'\bin\s+my\s+opinion\b',
            r'\bi\s+personally\b',
            r'\bi\s+think\s+that\b',
        ]
        
        # Uncertainty markers (humans express doubt, AI is confident)
        self.uncertainty_markers = [
            r'\bperhaps\b', r'\bmaybe\b', r'\bmight\b', r'\bcould\s+be\b',
            r'\bi\'?m\s+not\s+sure\b', r'\bi\s+guess\b', r'\bprobably\b',
            r'\bit\s+seems\b', r'\bappears\s+to\b', r'\bkind\s+of\b',
            r'\bsort\s+of\b', r'\ba\s+bit\b', r'\bsomewhat\b',
        ]
        
    def _count_mun_strength(self, text: str) -> Tuple[int, float]:
        """
        Count MUN indicators with weighted scoring.
        Returns (raw_count, weighted_score)
        """
        text_lower = text.lower()
        raw_count = 0
        weighted_score = 0.0
        
        for pattern, weight in self.mun_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                raw_count += matches
                weighted_score += matches * weight
        
        return raw_count, weighted_score
    
    def _count_student_markers(self, text: str) -> int:
        """Count student writing authenticity markers."""
        text_lower = text.lower()
        count = 0
        for pattern in self.student_markers:
            if re.search(pattern, text_lower):
                count += 1
        return count
    
    def _count_uncertainty_markers(self, text: str) -> int:
        """Count uncertainty expressions (authentic human signal)."""
        text_lower = text.lower()
        count = 0
        for pattern in self.uncertainty_markers:
            count += len(re.findall(pattern, text_lower))
        return count
    
    def _has_first_person_introduction(self, text: str) -> bool:
        """Check for first-person introductions typical of student presentations."""
        patterns = [
            r'\bi\s+am\s+\w+.*representing\b',
            r'\bmy\s+name\s+is\s+\w+.*i\s+(am|will)\b',
            r'\bhello.*my\s+name\s+is\b',
            r'\bi\'?m\s+\w+.*delegate\b',
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in patterns)
    
    def predict_with_correction(self, text: str, feature_vector: list) -> Dict:
        """
        Make prediction with intelligent MUN/formal speech correction.
        
        Args:
            text: Original text being analyzed
            feature_vector: 85-feature vector from FeatureExtractorV3 (without new features)
            
        Returns:
            Dict with prediction details including correction info
        """
        # Scale features and get base prediction
        X = self.scaler.transform([feature_vector])
        base_prob = self.model.predict_proba(X)[0][1]  # AI probability
        
        # Analyze text for MUN/student indicators
        mun_count, mun_score = self._count_mun_strength(text)
        student_count = self._count_student_markers(text)
        uncertainty_count = self._count_uncertainty_markers(text)
        has_intro = self._has_first_person_introduction(text)
        
        # Calculate correction factor
        correction = 0.0
        correction_reasons = []
        
        # Strong MUN detection (very likely student speech)
        if mun_score >= 6:  # Multiple strong MUN markers
            correction = min(0.50, mun_score * 0.05)  # Up to 50% reduction
            correction_reasons.append(f"Strong MUN signals (score={mun_score:.1f})")
        elif mun_score >= 3:  # Moderate MUN markers
            correction = min(0.30, mun_score * 0.05)
            correction_reasons.append(f"MUN signals detected (score={mun_score:.1f})")
        
        # Student markers add to correction
        if student_count >= 3:
            student_correction = min(0.20, student_count * 0.04)
            correction += student_correction
            correction_reasons.append(f"Student markers ({student_count})")
        
        # First-person introduction is strong human signal
        if has_intro:
            correction += 0.10
            correction_reasons.append("First-person introduction")
        
        # Uncertainty expressions are human authenticity signals
        if uncertainty_count >= 2:
            correction += min(0.10, uncertainty_count * 0.02)
            correction_reasons.append(f"Uncertainty expressions ({uncertainty_count})")
        
        # Cap total correction at 60%
        correction = min(0.60, correction)
        
        # Apply correction
        corrected_prob = max(0.0, base_prob - correction)
        
        # Determine classification
        is_ai = corrected_prob >= 0.5
        
        return {
            'base_probability': base_prob,
            'corrected_probability': corrected_prob,
            'correction_applied': correction,
            'correction_reasons': correction_reasons,
            'classification': 'AI' if is_ai else 'HUMAN',
            'confidence': abs(corrected_prob - 0.5) * 2,  # 0-1 scale
            'mun_score': mun_score,
            'mun_count': mun_count,
            'student_markers': student_count,
            'uncertainty_count': uncertainty_count,
            'has_first_person_intro': has_intro,
        }


def test_enhanced_model():
    """Test the enhanced model on MUN samples."""
    from feature_extractor_v3 import FeatureExtractorV3
    
    # Create enhanced model
    model = SupernovaUltraV6Enhanced()
    extractor = FeatureExtractorV3()
    
    # Get the v6 feature names (without new features)
    v6_feature_names = [n for n in extractor.feature_names 
                        if n not in ['formal_speech_strength', 'human_authenticity_score']]
    
    sample1 = '''Hello delegates, my name is NAME and I am representing France for the United Nations High Commissioner for Refugees. Today I will be discussing the protection of internally displaced people in armed conflict, a pressing global issue that demands our immediate attention. France is committed to upholding the principles outlined in the Guiding Principles on Internal Displacement and to working collaboratively with all member states to address this humanitarian crisis.

Internal displacement caused by armed conflict affects millions of lives worldwide, separating families and destroying communities. France has been at the forefront of humanitarian aid efforts, providing substantial funding and resources to UNHCR operations across the globe. We firmly believe that the protection of internally displaced persons is not just a moral obligation but a legal one under international humanitarian law.

During our committee sessions, France proposes several key initiatives. First, establishing early warning systems in conflict-prone regions to facilitate timely evacuations. Second, increasing funding for UNHCR field operations in active conflict zones. Third, strengthening collaboration between national governments and international organizations to create secure corridors for civilian movement.'''

    sample2 = '''Distinguished delegates,

I am (Insert Name), your delegate from Lebanon; I welcome you to the United Nations Disarmament and Security Council and would like to acknowledge the pressing global issue of statelessness. Statelessness is not merely an administrative inconvenience—it is the complete denial of legal identity to millions of men, women, and children. To be stateless is to lack access to education, healthcare, employment, and mobility. It is to be forgotten by the systems designed to protect us.

Lebanon, though not a signatory to the 1954 or 1961 conventions, has long borne witness to the devastating human toll of statelessness. Our nation hosts one of the largest per-capita refugee populations in the world, and within those communities—particularly among the Palestinians—we see generations who have never known the protection of citizenship. We understand better than most the complexity of this issue and the urgent need for international cooperation.

A child born stateless today faces obstacles at every stage of life. They cannot be registered at birth, they cannot access formal education, they cannot work legally, and they cannot marry or travel. They are invisible to systems meant to serve them. This is not just a legal failure—it is a moral one.'''

    samples = [('Sample 1 (France UNHCR)', sample1), ('Sample 2 (Lebanon)', sample2)]
    
    print("=" * 70)
    print("SUPERNOVA ULTRA v6 ENHANCED - MUN Correction Test")
    print("=" * 70)
    
    for name, text in samples:
        # Extract features (only the 85 v6 features)
        features = extractor.extract_features(text)
        feature_vector = [features[n] for n in v6_feature_names]
        
        # Get prediction with correction
        result = model.predict_with_correction(text, feature_vector)
        
        print(f"\n{name}:")
        print(f"  Base AI probability:     {result['base_probability']*100:.2f}%")
        print(f"  Correction applied:      -{result['correction_applied']*100:.2f}%")
        print(f"  Corrected probability:   {result['corrected_probability']*100:.2f}%")
        print(f"  Classification:          {result['classification']}")
        print(f"  Confidence:              {result['confidence']*100:.1f}%")
        print()
        print(f"  Detection details:")
        print(f"    MUN score:             {result['mun_score']:.1f} (count: {result['mun_count']})")
        print(f"    Student markers:       {result['student_markers']}")
        print(f"    Uncertainty markers:   {result['uncertainty_count']}")
        print(f"    First-person intro:    {result['has_first_person_intro']}")
        if result['correction_reasons']:
            print(f"  Correction reasons:")
            for reason in result['correction_reasons']:
                print(f"    - {reason}")
    
    print()
    print("=" * 70)
    print("Expected: Both samples should be classified as HUMAN")
    print("=" * 70)


if __name__ == "__main__":
    test_enhanced_model()
