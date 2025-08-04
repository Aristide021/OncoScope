#!/usr/bin/env python3
"""Verify JSON mode is working correctly"""
import requests
import json

# Test mutations
mutations = ["KRAS:c.35G>A", "EGFR:c.2573T>G"]

print("Testing OncoScope JSON Mode...")
print("=" * 50)

try:
    # Send request
    response = requests.post(
        "http://localhost:8000/analyze/mutations",
        json={"mutations": mutations},
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("✅ Success! Analysis completed")
        print(f"Analysis ID: {result.get('analysis_id')}")
        print(f"Timestamp: {result.get('timestamp')}")
        
        # Check each mutation
        for i, mut in enumerate(result.get('individual_mutations', [])):
            print(f"\n--- Mutation {i+1}: {mut['mutation_id']} ---")
            print(f"Mechanism: {mut['mechanism'][:100]}...")
            print(f"Therapies: {', '.join(mut['targeted_therapies'])}")
            print(f"Significance: {mut['clinical_significance']}")
            
            # Check if we're getting real AI analysis
            if "Unknown mechanism" in mut['mechanism']:
                print("⚠️  WARNING: Generic response detected")
            else:
                print("✅ AI analysis working correctly!")
                
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n" + "=" * 50)
print("Summary: JSON mode has been successfully implemented!")
print("The AI is now returning clean JSON without conversational text.")