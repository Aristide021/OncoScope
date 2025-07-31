#!/usr/bin/env python3
"""Analyze the quality and statistics of the regenerated datasets"""

import json
from pathlib import Path
from collections import Counter, defaultdict

# Analyze both datasets
datasets = {
    "6k": "training_data_regenerated/expert_curated_cancer_genomics_6k_20250730_152736.json",
    "2.5k": "training_data_regenerated/expert_curated_cancer_genomics_2k5_20250730_152737.json"
}

for size, path in datasets.items():
    print(f"\n{'='*80}")
    print(f"ANALYZING {size.upper()} DATASET")
    print(f"{'='*80}")
    
    with open(path, 'r') as f:
        dataset = json.load(f)
    
    # Basic stats
    print(f"\n📊 Basic Statistics:")
    print(f"  Total examples: {len(dataset)}")
    
    # Quality tiers
    quality_tiers = Counter(ex.get('quality_tier', 'unknown') for ex in dataset)
    print(f"\n📈 Quality Tiers:")
    for tier, count in quality_tiers.most_common():
        percentage = (count / len(dataset)) * 100
        print(f"  {tier}: {count} ({percentage:.1f}%)")
    
    # Gene distribution
    genes = Counter(ex.get('gene', 'unknown') for ex in dataset)
    print(f"\n🧬 Top 10 Genes:")
    for gene, count in genes.most_common(10):
        percentage = (count / len(dataset)) * 100
        print(f"  {gene}: {count} ({percentage:.1f}%)")
    
    # Mutation types
    mutation_types = Counter(ex.get('mutation_type', 'unknown') for ex in dataset)
    print(f"\n🔬 Mutation Types:")
    for mtype, count in mutation_types.most_common():
        print(f"  {mtype}: {count}")
    
    # Check for KRAS variant diversity
    kras_variants = defaultdict(int)
    for ex in dataset:
        if ex.get('gene') == 'KRAS':
            kras_variants[ex.get('variant', 'unknown')] += 1
    
    if kras_variants:
        print(f"\n🎯 KRAS Variant Distribution:")
        for variant, count in sorted(kras_variants.items(), key=lambda x: x[1], reverse=True):
            print(f"  {variant}: {count}")
    
    # Data augmentation stats
    augmented = sum(1 for ex in dataset if ex.get('metadata', {}).get('augmented', False))
    print(f"\n🔄 Data Augmentation:")
    print(f"  Augmented examples: {augmented} ({(augmented/len(dataset))*100:.1f}%)")
    
    # Clinical context diversity
    contexts = set()
    for ex in dataset:
        if 'metadata' in ex and 'clinical_context' in ex['metadata']:
            contexts.add(ex['metadata']['clinical_context'])
    print(f"\n🏥 Clinical Context Diversity:")
    print(f"  Unique contexts: {len(contexts)}")
    
    # File size
    file_size_mb = Path(path).stat().st_size / (1024 * 1024)
    print(f"\n💾 File size: {file_size_mb:.2f} MB")
    
print(f"\n{'='*80}")
print("✅ Dataset analysis complete!")