#!/usr/bin/env python3
"""
SUPERNOVA ULTRA v6 Enhanced - Production Inference Wrapper
==========================================================

This is the production-ready inference system combining:
- SUPERNOVA ULTRA v6 (99.65% base accuracy)
- MUN/Formal Speech Correction Layer
- Confidence calibration

Usage:
    from supernova_ultra_inference import SupernovaUltraInference
    
    detector = SupernovaUltraInference()
    result = detector.analyze(text)
    
    print(f"Classification: {result['classification']}")
    print(f"AI Probability: {result['ai_probability']:.2%}")
"""

import json
import pickle
import re
import os
import numpy as np
from xgboost import XGBClassifier
from typing import Dict, List, Tuple, Optional

# Import the feature extractor - assumes it's in the same directory or on PYTHONPATH
try:
    from feature_extractor_v3 import FeatureExtractorV3
except ImportError:
    FeatureExtractorV3 = None


class SupernovaUltraInference:
    """
    Production inference wrapper for SUPERNOVA ULTRA v6 Enhanced.
    
    Features:
    - 99.65% base accuracy on general text
    - Intelligent MUN/debate/formal speech correction
    - Confidence calibration
    - Detailed analysis breakdown
    """
    
    VERSION = "6.0-enhanced"
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the inference engine.
        
        Args:
            model_dir: Path to model directory. Defaults to models/SupernovaUltraV6
        """
        if model_dir is None:
            # Find model directory relative to this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "models", "SupernovaUltraV6")
        
        self.model_dir = model_dir
        
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "model.json"))
        
        # Load scaler
        with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        with open(os.path.join(model_dir, "metadata.json"), 'r') as f:
            self.metadata = json.load(f)
        
        # Initialize feature extractor
        if FeatureExtractorV3 is not None:
            self.extractor = FeatureExtractorV3()
            # Get v6-compatible feature names (85 features)
            self.feature_names = [n for n in self.extractor.feature_names 
                                  if n not in ['formal_speech_strength', 'human_authenticity_score']]
        else:
            self.extractor = None
            self.feature_names = None
        
        # Initialize MUN detection patterns
        self._init_mun_patterns()
    
    def _init_mun_patterns(self):
        """Initialize MUN and formal speech detection patterns."""
        # MUN/Debate patterns with weights
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
        
        # Student authenticity markers
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
        
        # Uncertainty markers (authentic human signal)
        self.uncertainty_markers = [
            r'\bperhaps\b', r'\bmaybe\b', r'\bmight\b', r'\bcould\s+be\b',
            r'\bi\'?m\s+not\s+sure\b', r'\bi\s+guess\b', r'\bprobably\b',
            r'\bit\s+seems\b', r'\bappears\s+to\b', r'\bkind\s+of\b',
            r'\bsort\s+of\b', r'\ba\s+bit\b', r'\bsomewhat\b',
        ]
        
        # First-person intro patterns
        self.intro_patterns = [
            r'\bi\s+am\s+\w+.*representing\b',
            r'\bmy\s+name\s+is\s+\w+.*i\s+(am|will)\b',
            r'\bhello.*my\s+name\s+is\b',
            r'\bi\'?m\s+\w+.*delegate\b',
        ]
    
    def _count_mun_strength(self, text: str) -> Tuple[int, float]:
        """Count MUN indicators with weighted scoring."""
        text_lower = text.lower()
        raw_count = 0
        weighted_score = 0.0
        
        for pattern, weight in self.mun_patterns.items():
            matches = len(re.findall(pattern, text_lower))
            if matches > 0:
                raw_count += matches
                weighted_score += matches * weight
        
        return raw_count, weighted_score
    
    def _count_markers(self, text: str, patterns: List[str]) -> int:
        """Count occurrences of patterns in text."""
        text_lower = text.lower()
        count = 0
        for pattern in patterns:
            matches = re.findall(pattern, text_lower)
            count += len(matches) if isinstance(matches, list) else (1 if matches else 0)
        return count
    
    def _has_intro(self, text: str) -> bool:
        """Check for first-person introductions."""
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in self.intro_patterns)
    
    def _calculate_correction(self, text: str) -> Tuple[float, List[str]]:
        """
        Calculate correction factor for MUN/formal speech.
        
        Returns:
            Tuple of (correction_amount, reasons)
        """
        mun_count, mun_score = self._count_mun_strength(text)
        student_count = self._count_markers(text, self.student_markers)
        uncertainty_count = self._count_markers(text, self.uncertainty_markers)
        has_intro = self._has_intro(text)
        
        correction = 0.0
        reasons = []
        
        # MUN detection
        if mun_score >= 6:
            correction = min(0.50, mun_score * 0.05)
            reasons.append(f"Strong MUN signals (score={mun_score:.1f})")
        elif mun_score >= 3:
            correction = min(0.30, mun_score * 0.05)
            reasons.append(f"MUN signals (score={mun_score:.1f})")
        
        # Student markers
        if student_count >= 3:
            student_correction = min(0.20, student_count * 0.04)
            correction += student_correction
            reasons.append(f"Student markers ({student_count})")
        
        # First-person intro
        if has_intro:
            correction += 0.10
            reasons.append("First-person introduction")
        
        # Uncertainty expressions
        if uncertainty_count >= 2:
            correction += min(0.10, uncertainty_count * 0.02)
            reasons.append(f"Uncertainty expressions ({uncertainty_count})")
        
        # Cap correction
        correction = min(0.60, correction)
        
        return correction, reasons
    
    def analyze(self, text: str, return_features: bool = False) -> Dict:
        """
        Analyze text for AI-generated content.
        
        Args:
            text: Text to analyze
            return_features: If True, include feature values in response
            
        Returns:
            Dict with analysis results
        """
        if self.extractor is None:
            raise RuntimeError("FeatureExtractorV3 not available")
        
        # Extract features
        features = self.extractor.extract_features(text)
        feature_vector = [features[n] for n in self.feature_names]
        
        # Get base prediction
        X = self.scaler.transform([feature_vector])
        base_prob = float(self.model.predict_proba(X)[0][1])
        
        # Calculate correction
        correction, reasons = self._calculate_correction(text)
        
        # Apply correction
        corrected_prob = max(0.0, base_prob - correction)
        
        # Determine classification
        is_ai = corrected_prob >= 0.5
        
        # Calculate confidence
        confidence = abs(corrected_prob - 0.5) * 2
        
        # Confidence level labels
        if confidence >= 0.8:
            confidence_level = "Very High"
        elif confidence >= 0.6:
            confidence_level = "High"
        elif confidence >= 0.4:
            confidence_level = "Moderate"
        elif confidence >= 0.2:
            confidence_level = "Low"
        else:
            confidence_level = "Very Low"
        
        result = {
            'classification': 'AI' if is_ai else 'HUMAN',
            'ai_probability': corrected_prob,
            'human_probability': 1 - corrected_prob,
            'confidence': confidence,
            'confidence_level': confidence_level,
            
            # Correction details
            'correction_applied': correction > 0,
            'correction_amount': correction,
            'correction_reasons': reasons,
            
            # Raw model output
            'base_ai_probability': base_prob,
            
            # Metadata
            'model_version': self.VERSION,
            'features_used': len(self.feature_names),
        }
        
        if return_features:
            result['features'] = {n: features[n] for n in self.feature_names}
        
        return result
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'version': self.VERSION,
            'base_accuracy': self.metadata.get('accuracy', 0),
            'high_conf_accuracy': self.metadata.get('high_confidence_accuracy', 0),
            'features': len(self.feature_names) if self.feature_names else 0,
            'training_samples': self.metadata.get('training_samples', 0),
            'model_dir': self.model_dir,
        }


# Convenience function
def analyze_text(text: str, verbose: bool = False) -> Dict:
    """
    Convenience function to analyze text.
    
    Args:
        text: Text to analyze
        verbose: If True, print results
        
    Returns:
        Analysis result dict
    """
    detector = SupernovaUltraInference()
    result = detector.analyze(text)
    
    if verbose:
        print(f"Classification: {result['classification']}")
        print(f"AI Probability: {result['ai_probability']:.2%}")
        print(f"Confidence: {result['confidence_level']} ({result['confidence']:.2%})")
        if result['correction_applied']:
            print(f"Correction: -{result['correction_amount']:.2%}")
            for reason in result['correction_reasons']:
                print(f"  - {reason}")
    
    return result


if __name__ == "__main__":
    # Quick test
    detector = SupernovaUltraInference()
    
    print("=" * 60)
    print("SUPERNOVA ULTRA v6 Enhanced - Production Inference")
    print("=" * 60)
    
    info = detector.get_model_info()
    print(f"\nModel Version: {info['version']}")
    print(f"Base Accuracy: {info['base_accuracy']:.2%}")
    print(f"High-Confidence Accuracy: {info['high_conf_accuracy']:.2%}")
    print(f"Features: {info['features']}")
    
    # Test samples
    test_text = """Hello delegates, my name is Alex and I am representing France for 
    the United Nations High Commissioner for Refugees. Today I will be discussing 
    the protection of internally displaced people in armed conflict."""
    
    print("\n" + "-" * 60)
    print("Test Analysis:")
    result = analyze_text(test_text, verbose=True)
