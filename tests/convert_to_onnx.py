#!/usr/bin/env python3
"""
Convert SUPERNOVA XGBoost model to ONNX for browser deployment
"""

import pickle
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType as XGBFloatTensorType
import json

print("=" * 60)
print("SUPERNOVA - Converting to ONNX for Browser")
print("=" * 60)

# Load the production model
print("\nLoading production model...")
with open('veritas_production.pkl', 'rb') as f:
    bundle = pickle.load(f)

model = bundle['model']
scaler = bundle['scaler']
feature_names = bundle['feature_names']
metrics = bundle['metrics']

print(f"Model loaded: {metrics['accuracy']:.2%} accuracy")
print(f"Features: {len(feature_names)} heuristic + 384 embedding = 415 total")

# Convert XGBoost to ONNX
print("\nConverting XGBoost to ONNX...")
n_features = 415  # 31 heuristic + 384 embedding

initial_type = [('float_input', XGBFloatTensorType([None, n_features]))]
onnx_model = convert_xgboost(model, initial_types=initial_type, target_opset=12)

# Save ONNX model
onnx_path = '../training/models/Supernova/supernova_xgb.onnx'
import os
os.makedirs('../training/models/Supernova', exist_ok=True)
onnx.save_model(onnx_model, onnx_path)
print(f"Saved XGBoost ONNX model: {onnx_path}")

# Check model size
model_size = os.path.getsize(onnx_path) / (1024 * 1024)
print(f"Model size: {model_size:.2f} MB")

# Save scaler parameters for JavaScript
print("\nExporting scaler parameters...")
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'feature_names': feature_names
}

scaler_path = '../training/models/Supernova/scaler_params.json'
with open(scaler_path, 'w') as f:
    json.dump(scaler_params, f)
print(f"Saved scaler: {scaler_path}")

# Create model metadata
print("\nCreating metadata...")
metadata = {
    'name': 'Supernova',
    'version': '1.0',
    'accuracy': metrics['accuracy'],
    'high_conf_accuracy': metrics['high_conf_accuracy'],
    'high_conf_coverage': metrics['high_conf_coverage'],
    'auc_roc': metrics['auc_roc'],
    'n_heuristic_features': len(feature_names),
    'n_embedding_features': 384,
    'embedding_model': 'Xenova/all-MiniLM-L6-v2',  # Transformers.js compatible
    'training_samples': bundle['training_samples'],
    'description': 'SUPERNOVA - Neural-enhanced AI detection with 97%+ accuracy on high-confidence samples'
}

metadata_path = '../training/models/Supernova/metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata: {metadata_path}")

# Verify ONNX model
print("\nVerifying ONNX model...")
import onnxruntime as ort

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Test with random input
test_input = np.random.randn(1, n_features).astype(np.float32)
result = session.run([output_name], {input_name: test_input})
print(f"ONNX inference test: OK (output shape: {result[0].shape})")

print("\n" + "=" * 60)
print("SUPERNOVA ONNX CONVERSION COMPLETE")
print("=" * 60)
print(f"""
Files created:
  - supernova_xgb.onnx ({model_size:.2f} MB)
  - scaler_params.json
  - metadata.json

For browser deployment:
  1. Load embedding model with Transformers.js
  2. Extract heuristic features in JavaScript  
  3. Run ONNX model with ONNX Runtime Web
""")
