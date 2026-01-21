#!/usr/bin/env python3
"""
VERITAS Production Inference
Usage: python veritas_inference.py "Your text here"

Or import and use programmatically:
    from veritas_inference import VERITASDetector
    detector = VERITASDetector()
    result = detector.predict("Your text here")
"""

import sys
import json
import re
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class VERITASDetector:
    """Production AI text detection with confidence thresholds."""
    
    def __init__(self, model_path='veritas_production.pkl'):
        """Load the production model."""
        print("Loading VERITAS production model...")
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        
        self.model = bundle['model']
        self.scaler = bundle['scaler']
        self.feature_names = bundle['feature_names']
        self.metrics = bundle['metrics']
        
        print(f"Loading embedding model: {bundle['embed_model']}...")
        self.embed_model = SentenceTransformer(bundle['embed_model'])
        
        print(f"Model loaded. Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"High-confidence accuracy: {self.metrics['high_conf_accuracy']:.2%}")
    
    def extract_features(self, text):
        """Extract heuristic features from text."""
        if not text or len(text) < 10:
            return None
        
        words = text.split()
        word_count = len(words) if words else 1
        
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sent_count = len(sentences) if sentences else 1
        
        features = {}
        
        # Top discriminators
        features['third_he_she'] = len(re.findall(r'\b(he|she|him|her|his|hers)\b', text, re.I)) / word_count
        features['first_me'] = len(re.findall(r'\b(me|my|mine|myself)\b', text, re.I)) / word_count
        features['answer_opener'] = 1 if re.match(r'^(Yes|No|Sure|Certainly|Of course|Absolutely)\b', text.strip(), re.I) else 0
        features['ellipsis_count'] = len(re.findall(r'\.{3}|…', text))
        features['instruction_phrases'] = len(re.findall(r'\b(first,|second,|finally,|step \d|for example)\b', text, re.I))
        features['attribution'] = len(re.findall(r'\b(said|says|told|according to|noted|stated)\b', text, re.I))
        features['numbered_items'] = len(re.findall(r'^\s*\d+[.)]\s+', text, re.M))
        features['helpful_phrases'] = len(re.findall(r'\b(here is|let me|feel free|I hope this)\b', text, re.I))
        features['proper_nouns'] = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)) / word_count
        
        # Structural
        sent_lengths = [len(s.split()) for s in sentences]
        features['avg_sent_len'] = np.mean(sent_lengths) if sent_lengths else 0
        features['sent_len_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
        features['sent_count'] = sent_count
        features['min_sent_len'] = min(sent_lengths) if sent_lengths else 0
        
        paragraphs = text.split('\n\n')
        features['para_count'] = len([p for p in paragraphs if p.strip()])
        
        # Pronouns
        features['first_I'] = len(re.findall(r'\bI\b', text)) / word_count
        features['first_we'] = len(re.findall(r'\b(we|us|our)\b', text, re.I)) / word_count
        features['second_you'] = len(re.findall(r'\b(you|your)\b', text, re.I)) / word_count
        
        # Punctuation
        features['colon_rate'] = text.count(':') / sent_count
        features['question_rate'] = text.count('?') / sent_count
        features['paren_count'] = text.count('(')
        
        # Temporal references
        features['month_mentions'] = len(re.findall(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b', text))
        features['year_mentions'] = len(re.findall(r'\b(19|20)\d{2}\b', text))
        features['time_mentions'] = len(re.findall(r'\b\d{1,2}:\d{2}\b', text))
        features['specific_dates'] = len(re.findall(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', text))
        
        # Style
        features['has_code'] = 1 if re.search(r'```|def\s+\w+|function\s*\(', text) else 0
        features['has_html'] = 1 if re.search(r'<[a-z]+[^>]*>', text, re.I) else 0
        features['discourse_markers'] = len(re.findall(r'\b(however|therefore|furthermore|moreover|consequently)\b', text, re.I))
        features['contraction_rate'] = len(re.findall(r"\b\w+'(t|re|ve|ll|d|s|m)\b", text, re.I)) / word_count
        features['casual_words'] = len(re.findall(r'\b(lol|haha|yeah|nah|ok|gonna|wanna)\b', text, re.I))
        
        features['word_count'] = word_count
        features['vocab_richness'] = len(set(w.lower() for w in words)) / word_count
        
        return features
    
    def predict(self, text, return_proba=True):
        """
        Predict whether text is human or AI-generated.
        
        Args:
            text: Text to analyze
            return_proba: If True, return probability scores
        
        Returns:
            dict with prediction results including confidence level
        """
        # Extract features
        heuristic_features = self.extract_features(text)
        if heuristic_features is None:
            return {
                'error': 'Text too short (minimum 10 characters)',
                'prediction': None
            }
        
        # Get embedding
        embedding = self.embed_model.encode(text[:2000], show_progress_bar=False)
        
        # Combine features
        X = np.hstack([list(heuristic_features.values()), embedding])
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0).reshape(1, -1)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        
        # Calculate confidence
        ai_prob = proba[1]
        
        # Determine confidence level
        if ai_prob > 0.9 or ai_prob < 0.1:
            confidence = 'very_high'
            conf_pct = max(ai_prob, 1 - ai_prob)
        elif ai_prob > 0.8 or ai_prob < 0.2:
            confidence = 'high'
            conf_pct = max(ai_prob, 1 - ai_prob)
        elif ai_prob > 0.7 or ai_prob < 0.3:
            confidence = 'medium'
            conf_pct = max(ai_prob, 1 - ai_prob)
        else:
            confidence = 'low'
            conf_pct = max(ai_prob, 1 - ai_prob)
        
        result = {
            'prediction': 'ai' if prediction == 1 else 'human',
            'ai_probability': float(ai_prob),
            'human_probability': float(1 - ai_prob),
            'confidence': confidence,
            'confidence_score': float(conf_pct),
            'flag_for_review': confidence in ['low', 'medium'],
            'model_version': '1.0'
        }
        
        return result
    
    def predict_batch(self, texts):
        """Predict for multiple texts efficiently."""
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results


def main():
    """Command-line interface."""
    if len(sys.argv) < 2:
        print("Usage: python veritas_inference.py \"Your text here\"")
        print("\nExample:")
        print('  python veritas_inference.py "This is a sample text to analyze."')
        sys.exit(1)
    
    text = sys.argv[1]
    
    detector = VERITASDetector('veritas_production.pkl')
    result = detector.predict(text)
    
    print("\n" + "=" * 60)
    print("VERITAS DETECTION RESULT")
    print("=" * 60)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Prediction: {result['prediction'].upper()}")
        print(f"AI Probability: {result['ai_probability']:.1%}")
        print(f"Human Probability: {result['human_probability']:.1%}")
        print(f"Confidence: {result['confidence'].replace('_', ' ').upper()}")
        
        if result['flag_for_review']:
            print("\n⚠️  FLAGGED FOR HUMAN REVIEW (low confidence)")
        
        print("\n" + json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
