#!/usr/bin/env python3
"""Verify KRAS therapeutic associations in the regenerated dataset"""

import json
import sys
from pathlib import Path

# Load the 6k dataset
dataset_path = Path("training_data_regenerated/expert_curated_cancer_genomics_6k_20250730_152736.json")

with open(dataset_path, 'r') as f:
    dataset = json.load(f)

print(f"Loaded dataset with {len(dataset)} examples")
print("=" * 80)

# Find KRAS examples
kras_examples = [ex for ex in dataset if ex.get('gene') == 'KRAS']
print(f"Found {len(kras_examples)} KRAS examples")

# Check specific variants
variants_to_check = {
    'c.34G>T': 'G12C - Should have Sotorasib/Adagrasib',
    'c.35G>A': 'G12D - Should have MRTX1133 (investigational)',
    'c.35G>T': 'G12V - Should have limited options',
    'c.38G>A': 'G13D - Should have different profile'
}

print("\nChecking KRAS variant-specific therapies:")
print("-" * 80)

for variant, description in variants_to_check.items():
    print(f"\n{variant} ({description}):")
    
    # Find examples with this variant
    variant_examples = [ex for ex in kras_examples if ex.get('variant') == variant]
    
    if variant_examples:
        # Parse the first example's output
        example = variant_examples[0]
        output = json.loads(example['output'])
        
        # Navigate to therapeutic implications
        therapies = None
        if 'consensus_analysis' in output:
            therapies = output['consensus_analysis'].get('therapeutic_implications', {}).get('targeted_therapies', [])
        elif 'expert_panel_analysis' in output:
            therapies = output['expert_panel_analysis'].get('clinical_implications', {}).get('targeted_therapies', [])
        
        if therapies:
            print(f"  ✅ Found therapies: {therapies}")
            
            # Verify correctness
            if variant == 'c.34G>T':
                if any('Sotorasib' in str(t) for t in therapies):
                    print("  ✅ Correctly includes Sotorasib for G12C")
                else:
                    print("  ❌ ERROR: Missing Sotorasib for G12C")
                    
            elif variant == 'c.35G>A':
                if any('MRTX1133' in str(t) for t in therapies):
                    print("  ✅ Correctly includes MRTX1133 for G12D")
                else:
                    print("  ❌ ERROR: Missing MRTX1133 for G12D")
                    
                if any('Sotorasib' in str(t) for t in therapies):
                    print("  ❌ ERROR: Incorrectly includes G12C drugs for G12D")
                else:
                    print("  ✅ Correctly excludes G12C-specific drugs")
        else:
            print(f"  ⚠️  No therapeutic implications found in output")
    else:
        print(f"  ⚠️  No examples found for this variant")

print("\n" + "=" * 80)
print("Verification complete!")