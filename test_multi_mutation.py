#!/usr/bin/env python3
"""
Test multi-mutation analysis with the updated system
"""
import asyncio
import json
from ai.inference.ollama_client import OllamaClient
from ai.inference.prompts import GenomicAnalysisPrompts

async def test_multi_mutation():
    """Test the multi-mutation analysis"""
    
    # Initialize components
    client = OllamaClient()
    prompt_gen = GenomicAnalysisPrompts()
    
    # Test mutations
    mutations = [
        {"gene": "TP53", "variant": "c.524G>A"},
        {"gene": "KRAS", "variant": "c.35G>A"},
        {"gene": "EGFR", "variant": "c.2369C>T"},
        {"gene": "PIK3CA", "variant": "c.3140A>G"},
        {"gene": "BRAF", "variant": "c.1799T>A"}
    ]
    
    # Create prompt
    prompt = prompt_gen.create_multi_mutation_analysis_prompt(mutations)
    
    print("🧬 Testing Multi-Mutation Analysis")
    print("=" * 60)
    print(f"Mutations: {len(mutations)}")
    print(f"Token limit: 2048")
    print(f"Prompt length: {len(prompt)} chars")
    print()
    
    # Test the analysis
    try:
        print("📤 Sending to Ollama...")
        result = await client.analyze_multi_mutations(prompt)
        
        print("✅ Response received!")
        print(f"Response type: {type(result)}")
        
        # Check if we got a valid response
        if result and isinstance(result, dict):
            print("\n📊 Analysis Results:")
            
            # Check for the new format
            if "raw_analysis" in result:
                raw = result["raw_analysis"]
                print(f"  - Mutation profile: {raw.get('mutation_profile', {})}")
                print(f"  - Composite risk: {raw.get('composite_risk', {})}")
                print(f"  - Clinical summary: {raw.get('clinical_summary', 'N/A')}")
                print(f"  - Confidence: {raw.get('key_insights', {}).get('confidence', 'N/A')}")
            
            # Check for insights
            if "multi_mutation_insights" in result:
                print(f"\n  - Insights: {len(result['multi_mutation_insights'])} found")
                for insight in result['multi_mutation_insights'][:3]:
                    print(f"    • {insight}")
            
            # Check if individual analysis was triggered
            if result.get("individual_analyses"):
                print(f"\n  - Individual analyses: {len(result['individual_analyses'])} completed")
            
            print("\n✅ Multi-mutation analysis successful!")
            
        else:
            print("❌ Invalid response format")
            print(f"Result: {result}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_multi_mutation())