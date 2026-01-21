# VERITAS ML Detector - Quick Start Guide

## Installation

```bash
# Install dependencies
pip install scikit-learn xgboost sentence-transformers numpy

# Or from requirements file
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Detection

```python
from veritas_inference import VERITASDetector

# Load model
detector = VERITASDetector('veritas_production.pkl')

# Analyze text
text = "This is the text you want to analyze for AI detection."
result = detector.predict(text)

# Check result
if result['prediction'] == 'ai':
    print(f"⚠️  AI DETECTED ({result['ai_probability']:.1%} confidence)")
else:
    print(f"✅ HUMAN WRITTEN ({result['human_probability']:.1%} confidence)")

# Check if needs review
if result['flag_for_review']:
    print("⚠️  Low confidence - recommend human review")
```

### 2. Batch Processing

```python
# Analyze multiple texts
texts = [
    "First text to analyze...",
    "Second text to analyze...",
    "Third text to analyze..."
]

results = detector.predict_batch(texts)

for i, result in enumerate(results):
    print(f"Text {i+1}: {result['prediction']} ({result['confidence']})")
```

### 3. Understanding Results

```python
result = detector.predict(text)

# Result structure:
{
    'prediction': 'ai' or 'human',
    'ai_probability': 0.95,           # 0-1 scale
    'human_probability': 0.05,         # 0-1 scale
    'confidence': 'very_high',         # very_high, high, medium, low
    'confidence_score': 0.95,          # Max of ai/human probability
    'flag_for_review': False,          # True if confidence is low/medium
    'model_version': '1.0'
}
```

### 4. Confidence Levels

| Confidence Level | Probability Range | Expected Accuracy | Action |
|------------------|-------------------|-------------------|--------|
| `very_high` | >0.9 or <0.1 | 98.13% | Auto-classify |
| `high` | >0.8 or <0.2 | 97.28% | Auto-classify |
| `medium` | >0.7 or <0.3 | 96.69% | Consider review |
| `low` | 0.3-0.7 | ~50% | Flag for review |

## Command Line Usage

```bash
# Single text analysis
python veritas_inference.py "Your text here"

# From file
python veritas_inference.py "$(cat myfile.txt)"

# With output to JSON
python veritas_inference.py "Your text" > result.json
```

## Production Deployment Recommendations

### 1. Confidence-Based Routing

```python
result = detector.predict(text)

if result['confidence'] in ['very_high', 'high']:
    # Auto-classify (97-98% accuracy)
    return result['prediction']
else:
    # Flag for human review
    return 'needs_review', result
```

### 2. Batch Processing for Performance

```python
# Instead of:
for text in texts:
    result = detector.predict(text)  # Slow

# Use batch:
results = detector.predict_batch(texts)  # Fast
```

### 3. Caching

```python
import hashlib

def get_prediction(text, cache={}):
    text_hash = hashlib.md5(text.encode()).hexdigest()
    
    if text_hash not in cache:
        cache[text_hash] = detector.predict(text)
    
    return cache[text_hash]
```

## Performance Characteristics

### Inference Speed
- **Single text**: ~200-500ms (includes embedding)
- **Batch of 100**: ~5-8 seconds
- **Bottleneck**: Sentence-BERT embedding generation

### Memory Usage
- Model size: ~500MB loaded in memory
- Per prediction: ~10MB (embedding model)

### Optimization Tips
1. **Load model once** and reuse (don't reload per prediction)
2. **Use batch prediction** for multiple texts
3. **Cache embeddings** for repeated text
4. **Run on GPU** if available (2-3x faster embeddings)

## Interpreting Edge Cases

### Low Confidence (0.4-0.6)
- Text genuinely ambiguous
- Represents ~1.6% of samples
- Accuracy drops to ~50% (coin flip)
- **Action**: Always flag for human review

### Problem Domains
1. **Formal Web Content** (like C4):
   - Polished, formal human writing
   - Low/no contractions or casual language
   - May be misclassified as AI
   - Confidence: Usually medium-low

2. **Short Factual Text** (like Dolly):
   - Brief encyclopedic statements
   - No typical AI phrases
   - May be misclassified as human
   - Confidence: Usually medium

3. **Technical Q&A** (like GPT4All):
   - StackOverflow-style responses
   - Code snippets and technical language
   - Overlaps with forum posts
   - Confidence: Usually medium

### When Model Works Best
- ✅ Creative writing (99%+ accuracy)
- ✅ Movie/product reviews (98-99%)
- ✅ Conversational dialogue (99%)
- ✅ Instructional AI responses (97-98%)
- ✅ Forum posts (99%)

### When to Use Caution
- ⚠️ Very formal writing
- ⚠️ Short factual statements (<50 words)
- ⚠️ Technical documentation
- ⚠️ When confidence is "low" or "medium"

## Example Integration

### Flask REST API

```python
from flask import Flask, request, jsonify
from veritas_inference import VERITASDetector

app = Flask(__name__)
detector = VERITASDetector('veritas_production.pkl')

@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = detector.predict(text)
    return jsonify(result)

@app.route('/detect/batch', methods=['POST'])
def detect_batch():
    data = request.get_json()
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = detector.predict_batch(texts)
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### JavaScript Integration

```javascript
// Call API
async function detectAI(text) {
    const response = await fetch('http://localhost:5000/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
    });
    
    return await response.json();
}

// Usage
const result = await detectAI("Your text here");
console.log(`Prediction: ${result.prediction} (${result.confidence})`);

if (result.flag_for_review) {
    console.log("⚠️  Needs human review");
}
```

## Troubleshooting

### Model fails to load
```python
# Error: File not found
# Solution: Check path
detector = VERITASDetector('/full/path/to/veritas_production.pkl')
```

### Out of memory
```python
# Error: OOM when loading
# Solution: Reduce batch size
results = detector.predict_batch(texts[:50])  # Process in smaller batches
```

### Slow inference
```python
# Problem: Taking too long per prediction
# Solutions:
# 1. Use GPU for embeddings (set CUDA_VISIBLE_DEVICES)
# 2. Reduce text length: text = text[:2000]
# 3. Use batch processing
# 4. Cache results for repeated text
```

### Unexpected results
```python
# Always check confidence
result = detector.predict(text)
if result['confidence'] in ['low', 'medium']:
    print("⚠️  Result may be unreliable")
    # Consider getting human validation
```

## Support

For issues or questions:
1. Check confidence level - low confidence = ambiguous case
2. Review problem domains (formal web, short factual, technical Q&A)
3. Consider text length (works best on 50-500 words)
4. Check for special formatting (code, HTML) that may confuse model

## Model Updates

The model was trained on data up to the cutoff date. As AI systems evolve:
- Periodically retrain on new data
- Monitor performance on production samples
- Collect feedback on flagged cases
- Consider specialized models for new AI patterns
