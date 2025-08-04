#!/usr/bin/env python3
"""Simple test for multi-mutation prompt"""
import requests
import json

OLLAMA_URL = "http://localhost:11434"
MODEL = "oncoscope-cancer:latest"

# Simple test prompt
prompt = """Analyze these mutations together:
- TP53:c.524G>A
- KRAS:c.35G>A

Return a simple JSON with:
{
  "pathogenic_count": 2,
  "risk": "high",
  "pathways": ["P53", "RAS"]
}"""

payload = {
    "model": MODEL,
    "prompt": prompt,
    "stream": False,
    "options": {
        "temperature": 0.1,
        "num_predict": 500
    }
}

print("Testing simple multi-mutation prompt...")
print(f"Prompt length: {len(prompt)} chars")

try:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=60
    )
    
    if response.status_code == 200:
        data = response.json()
        output = data.get("response", "")
        print(f"\nResponse length: {len(output)} chars")
        print(f"Response:\n{output}")
        
        # Try to parse as JSON
        try:
            parsed = json.loads(output)
            print("\n✅ Valid JSON!")
            print(json.dumps(parsed, indent=2))
        except:
            print("\n❌ Not valid JSON")
    else:
        print(f"Error: {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")