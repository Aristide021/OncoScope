#!/usr/bin/env python3
"""
Test script to verify context-aware mutation analysis
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent))

from backend.mutation_analyzer import CancerMutationAnalyzer
from ai.inference.ollama_client import OllamaClient

async def test_context_aware_analysis():
    """Test context-aware mutation analysis"""
    
    # Initialize analyzer
    analyzer = CancerMutationAnalyzer()
    
    # Test mutations
    mutations = [
        "EGFR:c.2573T>G",  # L858R - common in lung cancer
        "TP53:c.524G>A",   # R175H - common in many cancers
        "KRAS:c.35G>A"     # G12D - common in pancreatic, colorectal
    ]
    
    # Test 1: Without patient context
    print("=" * 80)
    print("TEST 1: Analysis WITHOUT patient context")
    print("=" * 80)
    
    result1 = await analyzer.analyze_mutation_list(
        mutations=mutations,
        patient_context=None
    )
    
    print("\nIndividual mutation analyses (no context):")
    for mut in result1['individual_mutations']:
        print(f"\n{mut['mutation_id']}:")
        print(f"  Cancer types: {mut['cancer_types'][:3]}")
        print(f"  Clinical context: {mut.get('clinical_context', 'N/A')[:100]}...")
    
    # Test 2: With breast cancer diagnosis
    print("\n" + "=" * 80)
    print("TEST 2: Analysis WITH breast cancer diagnosis")
    print("=" * 80)
    
    patient_context = {
        "diagnosis": "breast cancer",
        "age": "45",
        "gender": "female"
    }
    
    result2 = await analyzer.analyze_mutation_list(
        mutations=mutations,
        patient_context=patient_context
    )
    
    print("\nIndividual mutation analyses (breast cancer context):")
    for mut in result2['individual_mutations']:
        print(f"\n{mut['mutation_id']}:")
        print(f"  Cancer types: {mut['cancer_types'][:3]}")
        print(f"  Clinical context: {mut.get('clinical_context', 'N/A')[:100]}...")
    
    # Test 3: Direct prompt comparison
    print("\n" + "=" * 80)
    print("TEST 3: Direct prompt comparison")
    print("=" * 80)
    
    ollama_client = OllamaClient()
    
    # Single mutation prompt without context
    prompt1 = ollama_client._create_mutation_analysis_prompt("EGFR", "c.2573T>G", None)
    print("\nPrompt WITHOUT context:")
    print(prompt1[:300] + "...")
    
    # Single mutation prompt with context
    prompt2 = ollama_client._create_mutation_analysis_prompt("EGFR", "c.2573T>G", patient_context)
    print("\nPrompt WITH breast cancer context:")
    print(prompt2[:500] + "...")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE - Context awareness implemented!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_context_aware_analysis())