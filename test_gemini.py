import google.generativeai as genai
import json

genai.configure(api_key="AIzaSyCuFQyA5i421bB2PoRxf_knSNAjYiN2Rvo")

# List available models
print("Available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"  {m.name}")
