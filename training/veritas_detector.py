#!/usr/bin/env python3
"""
VERITAS AI Detector - Production Inference
==========================================

Best model: SUPERNOVA ULTRA v11
- 93.30% accuracy on independent test set
- 3.70% false positive rate (humans incorrectly marked as AI)
- 9.70% false negative rate (AI incorrectly marked as human)
- 97.76% high-confidence accuracy

Usage:
    from veritas_detector import VeritasDetector
    
    detector = VeritasDetector()
    result = detector.analyze("Your text here...")
    
    print(f"Classification: {result['classification']}")
    print(f"AI Probability: {result['ai_probability']:.1%}")
    print(f"Confidence: {result['confidence_level']}")
"""

import json
import pickle
import os
import numpy as np
from xgboost import XGBClassifier
from typing import Dict, List, Optional

try:
    from feature_extractor_v3 import FeatureExtractorV3
except ImportError:
    FeatureExtractorV3 = None


class VeritasDetector:
    """
    Production AI text detector using SUPERNOVA ULTRA v11.
    """
    
    VERSION = "11.0"
    MODEL_NAME = "SUPERNOVA ULTRA v11"
    
    def __init__(self, model_dir: str = None):
        """
        Initialize the detector.
        
        Args:
            model_dir: Path to model directory. Defaults to models/SupernovaUltraV11
        """
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(script_dir, "models", "SupernovaUltraV11")
        
        self.model_dir = model_dir
        
        # Load model
        self.model = XGBClassifier()
        self.model.load_model(os.path.join(model_dir, "model.json"))
        
        # Load scaler
        with open(os.path.join(model_dir, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load metadata
        meta_path = os.path.join(model_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Initialize feature extractor
        if FeatureExtractorV3 is None:
            raise RuntimeError("FeatureExtractorV3 not found")
        
        self.extractor = FeatureExtractorV3()
        # Use 85 base features
        self.feature_names = [n for n in self.extractor.feature_names 
                              if n not in ['formal_speech_strength', 'human_authenticity_score']]
    
    def analyze(self, text: str) -> Dict:
        """
        Analyze text for AI-generated content.
        
        Args:
            text: Text to analyze (minimum ~50 words recommended)
            
        Returns:
            Dict with:
                - classification: 'AI' or 'HUMAN'
                - ai_probability: 0.0-1.0
                - human_probability: 0.0-1.0
                - confidence: 0.0-1.0
                - confidence_level: 'Very High'/'High'/'Moderate'/'Low'/'Very Low'
        """
        # Extract features
        features = self.extractor.extract_features(text)
        feature_vector = [features[n] for n in self.feature_names]
        
        # Scale and predict
        X = self.scaler.transform([feature_vector])
        ai_prob = float(self.model.predict_proba(X)[0][1])
        
        # Classification
        is_ai = ai_prob >= 0.5
        
        # Confidence (distance from decision boundary)
        confidence = abs(ai_prob - 0.5) * 2
        
        # Confidence level
        if confidence >= 0.8:
            conf_level = "Very High"
        elif confidence >= 0.6:
            conf_level = "High"
        elif confidence >= 0.4:
            conf_level = "Moderate"
        elif confidence >= 0.2:
            conf_level = "Low"
        else:
            conf_level = "Very Low"
        
        return {
            'classification': 'AI' if is_ai else 'HUMAN',
            'ai_probability': ai_prob,
            'human_probability': 1 - ai_prob,
            'confidence': confidence,
            'confidence_level': conf_level,
            'model_version': self.VERSION,
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Analyze multiple texts."""
        return [self.analyze(text) for text in texts]
    
    def get_info(self) -> Dict:
        """Get model information."""
        return {
            'name': self.MODEL_NAME,
            'version': self.VERSION,
            'accuracy': self.metadata.get('accuracy', 0.933),
            'fpr': self.metadata.get('fpr', 0.037),
            'fnr': self.metadata.get('fnr', 0.097),
            'features': len(self.feature_names),
        }


def analyze_text(text: str, verbose: bool = True) -> Dict:
    """
    Convenience function to analyze text.
    
    Args:
        text: Text to analyze
        verbose: Print results
        
    Returns:
        Analysis result
    """
    detector = VeritasDetector()
    result = detector.analyze(text)
    
    if verbose:
        print(f"Classification: {result['classification']}")
        print(f"AI Probability: {result['ai_probability']:.1%}")
        print(f"Confidence: {result['confidence_level']}")
    
    return result


if __name__ == "__main__":
    detector = VeritasDetector()
    info = detector.get_info()
    
    print("=" * 50)
    print(f"VERITAS Detector - {info['name']}")
    print("=" * 50)
    print(f"Version:  {info['version']}")
    print(f"Accuracy: {info['accuracy']*100:.1f}%")
    print(f"FPR:      {info['fpr']*100:.1f}%")
    print(f"FNR:      {info['fnr']*100:.1f}%")
    print(f"Features: {info['features']}")
    print()
    
    # Quick test
    test = "This is a test sentence to verify the detector is working properly."
    result = detector.analyze(test)
    print(f"Test: {result['classification']} ({result['ai_probability']:.1%} AI)")
