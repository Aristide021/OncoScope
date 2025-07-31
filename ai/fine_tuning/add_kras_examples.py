#!/usr/bin/env python3
"""
Add KRAS-specific training examples to clarify mechanism
"""
import json
import random
from datetime import datetime

# KRAS-specific training examples emphasizing correct mechanism
kras_examples = [
    # Direct mechanism explanations
    {
        "input": """Analyze the cancer mutation KRAS:G12C and provide structured clinical assessment.

Gene: KRAS
Variant: G12C

Provide your analysis in the following JSON format:
{
    "pathogenicity": <float between 0.0-1.0>,
    "cancer_types": ["<list of associated cancer types>"],
    "protein_change": "<predicted protein change notation>",
    "mechanism": "<biological mechanism of the mutation>",
    "significance": "<one of: PATHOGENIC, LIKELY_PATHOGENIC, UNCERTAIN, LIKELY_BENIGN, BENIGN>",
    "therapies": ["<list of potential targeted therapies if any>"],
    "prognosis": "<one of: poor, moderate, good, excellent_with_therapy, uncertain>",
    "clinical_context": "<brief clinical interpretation>",
    "confidence": <float between 0.0-1.0 indicating confidence in this analysis>
}

Focus on established clinical knowledge and cancer genomics databases. Clearly explain the molecular mechanism of this mutation.""",
        "output": """{
    "pathogenicity": 0.95,
    "cancer_types": ["non-small cell lung cancer", "colorectal cancer", "pancreatic cancer"],
    "protein_change": "p.Gly12Cys",
    "mechanism": "KRAS is a small GTPase protein that acts as a molecular switch. The G12C mutation locks KRAS in its active GTP-bound state by preventing GTP hydrolysis, leading to constitutive activation of downstream signaling pathways including RAF-MEK-ERK and PI3K-AKT. This is a gain-of-function mutation.",
    "significance": "PATHOGENIC",
    "therapies": ["Sotorasib (Lumakras)", "Adagrasib (Krazati)"],
    "prognosis": "excellent_with_therapy",
    "clinical_context": "KRAS G12C is a targetable driver mutation with FDA-approved specific inhibitors. KRAS is NOT a G protein-coupled receptor (GPCR) but rather a small GTPase. This mutation represents a major therapeutic breakthrough in previously undruggable KRAS.",
    "confidence": 0.98
}"""
    },
    
    {
        "input": """What is the molecular mechanism of KRAS mutations in cancer? Explain the difference between KRAS and GPCRs.""",
        "output": """KRAS Molecular Mechanism in Cancer:

1. **KRAS Identity**: KRAS is a small GTPase protein (21 kDa), NOT a G protein-coupled receptor (GPCR). This is a critical distinction:
   - GTPases like KRAS are small monomeric proteins that bind and hydrolyze GTP
   - GPCRs are large seven-transmembrane receptors that activate heterotrimeric G proteins

2. **Normal KRAS Function**:
   - Acts as a molecular switch cycling between GDP-bound (inactive) and GTP-bound (active) states
   - GTP hydrolysis is catalyzed by GTPase-activating proteins (GAPs)
   - Guanine nucleotide exchange factors (GEFs) promote GDP release and GTP binding

3. **KRAS Mutations are Gain-of-Function**:
   - Mutations at codons 12, 13, and 61 impair intrinsic GTPase activity
   - Mutations prevent GAP-mediated GTP hydrolysis
   - Result: KRAS remains locked in the active GTP-bound state
   - This causes constitutive activation of downstream pathways (RAF-MEK-ERK, PI3K-AKT)

4. **Clinical Significance**:
   - KRAS mutations are drivers in 30% of cancers
   - G12C mutation is now targetable with covalent inhibitors
   - Different KRAS mutations have distinct therapeutic vulnerabilities"""
    },
    
    {
        "input": """Explain why KRAS G12C mutation causes cancer and how the new drugs work.""",
        "output": """KRAS G12C Mutation and Targeted Therapy:

**Why KRAS G12C Causes Cancer:**

1. **Molecular Mechanism**:
   - KRAS is a small GTPase (not a GPCR) that regulates cell growth
   - G12C mutation replaces glycine-12 with cysteine
   - This mutation is a GAIN-OF-FUNCTION change that:
     - Impairs GTP hydrolysis
     - Locks KRAS in active GTP-bound conformation
     - Causes continuous growth signaling

2. **Downstream Effects**:
   - Constitutive activation of RAF-MEK-ERK pathway → proliferation
   - Activation of PI3K-AKT pathway → survival
   - Metabolic reprogramming → cancer cell adaptation

**How KRAS G12C Inhibitors Work:**

1. **Covalent Binding**:
   - Sotorasib and Adagrasib bind covalently to cysteine-12
   - They trap KRAS G12C in the inactive GDP-bound state
   - This is called "locking the off state"

2. **Specificity**:
   - Only work on G12C mutation (not G12D, G12V, etc.)
   - Exploit the reactive cysteine residue
   - Spare normal KRAS protein

3. **Clinical Impact**:
   - First successful targeting of "undruggable" KRAS
   - Significant responses in lung and colorectal cancers
   - Resistance mechanisms emerging, requiring combination strategies"""
    },
    
    {
        "input": """Compare KRAS G12C and G12D mutations in terms of mechanism and therapeutic options.""",
        "output": """KRAS G12C vs G12D Comparison:

**Shared Mechanism (Both are Gain-of-Function)**:
- KRAS is a small GTPase protein (not a GPCR)
- Both mutations occur at codon 12 in the P-loop
- Both impair GTPase activity and GAP-mediated hydrolysis
- Both lock KRAS in active GTP-bound state
- Both activate RAF-MEK-ERK and PI3K pathways

**Key Differences**:

1. **G12C (Glycine → Cysteine)**:
   - Creates reactive cysteine for covalent targeting
   - FDA-approved drugs: Sotorasib, Adagrasib
   - ~13% of KRAS mutations
   - More common in lung cancer

2. **G12D (Glycine → Aspartic acid)**:
   - No reactive cysteine for covalent binding
   - No FDA-approved targeted therapy yet
   - MRTX1133 in clinical trials
   - ~33% of KRAS mutations
   - More common in pancreatic cancer

**Therapeutic Implications**:
- G12C: Excellent prognosis with targeted therapy
- G12D: Currently relies on combination approaches
- Both: MEK inhibitors, SHP2 inhibitors under investigation
- Future: Pan-KRAS inhibitors in development

**Important**: KRAS mutations are gain-of-function alterations in a GTPase, fundamentally different from GPCR signaling."""
    },
    
    {
        "input": """Patient has KRAS G12V mutation. Explain the mechanism and why it's different from G12C.""",
        "output": """KRAS G12V Mutation Analysis:

**Molecular Mechanism**:
1. **KRAS Function**: KRAS is a small GTPase protein (21 kDa) - NOT a GPCR
2. **G12V Effect**: Glycine-12 → Valine substitution
   - Gain-of-function mutation
   - Impairs intrinsic GTPase activity
   - Prevents GAP-mediated GTP hydrolysis
   - Locks KRAS in active GTP-bound state

**Why G12V Differs from G12C**:

1. **Chemical Properties**:
   - G12V: Introduces hydrophobic valine
   - G12C: Introduces reactive cysteine
   - Key difference: No covalent binding site in G12V

2. **Drug Targeting**:
   - G12C: FDA-approved covalent inhibitors (Sotorasib, Adagrasib)
   - G12V: No specific inhibitors approved
   - G12V cannot be targeted by cysteine-reactive drugs

3. **Prevalence**:
   - G12V: ~20% of KRAS mutations
   - Common in pancreatic and colorectal cancers
   - Associated with smoking in lung cancer

**Current Treatment Approach**:
- No G12V-specific drugs yet
- Combination strategies: MEK + SHP2 inhibitors
- Clinical trials: RMC-6236 (pan-KRAS inhibitor)
- Immunotherapy resistance marker

**Prognosis**: Moderate - awaiting specific targeted therapies"""
    },
    
    # Clinical scenario examples
    {
        "input": """MULTI-LAB CONSENSUS CANCER VARIANT ANALYSIS

Patient: 58-year-old Male
Clinical Context: newly diagnosed lung adenocarcinoma
Family History: none significant

Genetic Finding:
Gene: KRAS
Variant: c.34G>T (p.G12C)
Mutation Context: somatic mutation in tumor

Provide detailed analysis including molecular mechanism.""",
        "output": """{
  "consensus_analysis": {
    "patient_context": {
      "demographics": {
        "age": 58,
        "sex": "M"
      },
      "clinical_context": "newly diagnosed lung adenocarcinoma"
    },
    "variant_summary": {
      "gene": "KRAS",
      "variant": "c.34G>T",
      "protein_change": "p.G12C",
      "clinical_significance": "PATHOGENIC",
      "pathogenicity_score": 0.95
    },
    "molecular_mechanism": {
      "protein_type": "KRAS is a small GTPase protein, NOT a G protein-coupled receptor (GPCR)",
      "mutation_effect": "Gain-of-function mutation that locks KRAS in active GTP-bound state",
      "functional_impact": "Impairs intrinsic GTPase activity and prevents GAP-mediated GTP hydrolysis",
      "downstream_signaling": "Constitutive activation of RAF-MEK-ERK and PI3K-AKT pathways"
    },
    "therapeutic_implications": {
      "targeted_therapies": [
        "Sotorasib (Lumakras) - FDA approved",
        "Adagrasib (Krazati) - FDA approved"
      ],
      "mechanism_of_action": "Covalent KRAS G12C inhibitors bind to cysteine-12 and lock KRAS in inactive GDP-bound state",
      "response_rate": "~40% objective response rate in NSCLC",
      "resistance_mechanisms": [
        "Secondary KRAS mutations",
        "MET amplification",
        "Bypass pathway activation"
      ]
    },
    "clinical_recommendations": [
      "Initiate Sotorasib or Adagrasib therapy",
      "Baseline molecular profiling for co-mutations",
      "Monitor for resistance with serial ctDNA",
      "Consider combination trials at progression"
    ]
  }
}"""
    },
    
    {
        "input": """Explain the biological difference between KRAS and RAS proteins in general.""",
        "output": """RAS Family Proteins - Biological Overview:

**What are RAS Proteins?**
- Small GTPase proteins (~21 kDa)
- NOT G protein-coupled receptors (GPCRs)
- Three main isoforms: KRAS, NRAS, HRAS
- Act as molecular switches in signal transduction

**Key Distinctions from GPCRs**:
1. **Size**: RAS proteins are small (188-189 amino acids) vs GPCRs (>300 amino acids)
2. **Structure**: RAS are cytoplasmic vs GPCRs span membrane 7 times
3. **Function**: RAS are effectors vs GPCRs are receptors
4. **Activation**: RAS activated by receptor tyrosine kinases, not by ligand binding

**KRAS Specifics**:
- Most frequently mutated RAS isoform in cancer (85% of RAS mutations)
- Two splice variants: KRAS4A and KRAS4B
- Membrane localization via C-terminal CAAX motif
- Critical for embryonic development (knockout is lethal)

**Mechanism of RAS Proteins**:
1. **GDP-bound (inactive)** → **GTP-bound (active)** cycle
2. GEFs (e.g., SOS1) promote GTP loading
3. GAPs (e.g., NF1) accelerate GTP hydrolysis
4. Mutations prevent GTP hydrolysis = gain-of-function

**Cancer Relevance**:
- 30% of all cancers have RAS mutations
- KRAS: pancreatic (90%), colorectal (40%), lung (30%)
- All RAS mutations are gain-of-function
- Lock protein in active GTP-bound state"""
    },
    
    {
        "input": """Why do KRAS mutations cause resistance to EGFR inhibitors?""",
        "output": """KRAS Mutations and EGFR Inhibitor Resistance:

**Molecular Basis of Resistance**:

1. **Normal EGFR-KRAS Signaling**:
   - EGFR (receptor) → RAS (GTPase) → RAF → MEK → ERK
   - EGFR activates KRAS through GEFs (e.g., SOS1)
   - KRAS is a small GTPase downstream of EGFR (NOT a GPCR)

2. **Why KRAS Mutations Cause Resistance**:
   - KRAS mutations are gain-of-function
   - Mutant KRAS is constitutively active (GTP-bound)
   - No longer requires upstream EGFR activation
   - Blocking EGFR becomes irrelevant

3. **Clinical Impact**:
   - Colorectal cancer: KRAS testing mandatory before anti-EGFR therapy
   - Only KRAS wild-type tumors respond to cetuximab/panitumumab
   - KRAS mutations predict primary resistance
   - ~40% of CRC has KRAS mutations

**Specific Mechanisms**:
- G12/G13 mutations: Strong resistance predictors
- Impaired GTPase activity locks KRAS "ON"
- Continuous downstream signaling despite EGFR blockade
- Bypass of growth factor dependence

**Therapeutic Implications**:
- Test for KRAS before EGFR inhibitors
- KRAS mutant → avoid anti-EGFR therapy
- Use KRAS-specific inhibitors (G12C) or combinations
- Target downstream MEK or ERK instead"""
    },
    
    {
        "input": """How do KRAS G12C inhibitors like Sotorasib actually work at the molecular level?""",
        "output": """Sotorasib Mechanism of Action - Molecular Details:

**Target Identification**:
- KRAS is a small GTPase protein (NOT a GPCR)
- G12C mutation creates a reactive cysteine at position 12
- Normal KRAS has glycine at position 12 (no cysteine)

**Sotorasib's Unique Mechanism**:

1. **GDP-State Targeting**:
   - Sotorasib binds KRAS G12C in GDP-bound (inactive) state
   - Forms covalent bond with cysteine-12
   - Occupies switch II pocket (S-IIP)
   - Prevents GTP binding and activation

2. **Covalent Chemistry**:
   - Acrylamide warhead reacts with cysteine thiol
   - Irreversible Michael addition reaction
   - Forms stable thioether bond
   - Mutation-specific (only G12C has cysteine)

3. **Functional Consequences**:
   - Locks KRAS G12C in inactive conformation
   - Prevents GEF-mediated nucleotide exchange
   - Blocks interaction with RAF and other effectors
   - Shuts down RAF-MEK-ERK signaling

**Why This Approach Works**:
- Exploits unique vulnerability of G12C mutation
- Overcomes high GTP affinity (picomolar)
- Achieves selectivity through covalent targeting
- Spares wild-type KRAS

**Clinical Efficacy**:
- ~40% response rate in NSCLC
- Median PFS ~6.8 months
- Well-tolerated compared to chemotherapy
- Resistance emerges through multiple mechanisms"""
    },
    
    {
        "input": """Compare the different KRAS mutations (G12C, G12D, G12V, G13D) in terms of prevalence and druggability.""",
        "output": """KRAS Mutation Landscape - Comprehensive Comparison:

**Important**: KRAS is a small GTPase protein (not a GPCR). All mutations are gain-of-function, locking KRAS in active GTP-bound state.

**1. KRAS G12C**
- Prevalence: ~13% of KRAS mutations
- Cancer types: Lung (40%), colorectal (10%)
- Druggability: EXCELLENT - FDA approved drugs
- Drugs: Sotorasib, Adagrasib (covalent inhibitors)
- Mechanism: Reactive cysteine enables covalent targeting

**2. KRAS G12D**
- Prevalence: ~33% of KRAS mutations (most common)
- Cancer types: Pancreatic (45%), colorectal (20%)
- Druggability: EMERGING - Clinical trials ongoing
- Drugs: MRTX1133 (non-covalent, Phase I/II)
- Challenge: No reactive cysteine, requires different approach

**3. KRAS G12V**
- Prevalence: ~20% of KRAS mutations
- Cancer types: Pancreatic (30%), colorectal (20%), lung (10%)
- Druggability: LIMITED - No specific inhibitors
- Approach: Combination therapies, pan-KRAS inhibitors
- Challenge: Hydrophobic valine, no chemical handle

**4. KRAS G13D**
- Prevalence: ~10% of KRAS mutations
- Cancer types: Colorectal (20%), endometrial (10%)
- Druggability: LIMITED - No specific inhibitors
- Clinical: May respond differently to MEK inhibitors
- Note: Position 13, slightly different structure

**Therapeutic Hierarchy**:
1. G12C: Targeted therapy available ✓
2. G12D: Promising drugs in trials
3. G12V/G13D: Combination approaches
4. All: Pan-KRAS inhibitors (RMC-6236) in development"""
    }
]

def add_kras_examples():
    """Add KRAS examples to training data"""
    
    # Load existing data
    with open('cancer_training_data.json', 'r') as f:
        data = json.load(f)
    
    print(f"Current training data size: {len(data)} examples")
    
    # Add KRAS examples with metadata
    for i, example in enumerate(kras_examples):
        # Add metadata
        example_with_metadata = {
            **example,
            "gene": "KRAS",
            "mutation_type": "mechanism_clarification",
            "quality_tier": "expert_curated",
            "quality_weight": 1.0,
            "confidence_score": 0.98,
            "augmentation_id": len(data) + i + 1,
            "metadata": {
                "purpose": "clarify_kras_mechanism",
                "emphasis": "GTPase_not_GPCR",
                "key_concepts": ["small_GTPase", "gain_of_function", "GTP_bound_state"],
                "added_date": datetime.now().isoformat(),
                "competition_tier": "critical_correction"
            }
        }
        data.append(example_with_metadata)
    
    # Backup original file
    import shutil
    shutil.copy('cancer_training_data.json', 'cancer_training_data_backup_before_kras.json')
    
    # Save updated data
    with open('cancer_training_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Added {len(kras_examples)} KRAS-specific examples")
    print(f"New training data size: {len(data)} examples")
    print("\nExample topics covered:")
    print("- KRAS is a small GTPase (not GPCR)")
    print("- Gain-of-function mutations")
    print("- GTP-bound state locking")
    print("- Specific inhibitor mechanisms")
    print("- Clinical implications")

if __name__ == "__main__":
    add_kras_examples()