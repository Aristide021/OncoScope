#!/usr/bin/env python3
"""Test script to verify JSON mode is working correctly"""
import requests
import json
import sys

def test_mutation_analysis():
    """Test that the API returns clean JSON without conversational text"""
    
    # Test mutation
    test_mutations = ["KRAS:c.35G>A", "TP53:c.524G>A"]
    
    try:
        # Send analysis request
        print("Sending analysis request...")
        response = requests.post(
            "http://localhost:8000/analyze/mutations",
            json={"mutations": test_mutations},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Analysis completed successfully!")
            
            # Check if we have individual mutation analyses
            if "individual_mutations" in result:
                print(f"\nAnalyzed {len(result['individual_mutations'])} mutations")
                
                # Check the first mutation's mechanism
                first_mutation = result['individual_mutations'][0]
                mechanism = first_mutation.get('mechanism', '')
                
                print(f"\nFirst mutation mechanism: {mechanism[:100]}...")
                
                # Check if mechanism contains "Unknown mechanism"
                if "Unknown mechanism" in mechanism:
                    print("\n⚠️  WARNING: Still getting generic 'Unknown mechanism' response")
                    print("AI analysis might not be working properly")
                else:
                    print("\n✅ AI is providing detailed analysis!")
                    
            else:
                print("\n❌ No individual mutation analyses found in response")
                
        else:
            print(f"\n❌ Analysis failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to backend. Make sure it's running on http://localhost:8000")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("OncoScope JSON Mode Test")
    print("=" * 50)
    test_mutation_analysis()