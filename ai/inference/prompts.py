"""
Advanced Genomic Analysis Prompts
Sophisticated prompt engineering for cancer mutation analysis using Gemma 3n
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

class GenomicAnalysisPrompts:
    """Advanced prompts for genomic analysis with Gemma 3n"""
    
    def __init__(self):
        """Initialize genomic analysis prompts"""
        
        # Cancer gene classifications for context
        self.oncogenes = {
            'KRAS', 'EGFR', 'MYC', 'PIK3CA', 'BRAF', 'ERBB2', 'ALK', 'ROS1', 
            'MET', 'RET', 'FGFR1', 'FGFR2', 'FGFR3', 'PDGFRA', 'KIT'
        }
        
        self.tumor_suppressors = {
            'TP53', 'RB1', 'APC', 'BRCA1', 'BRCA2', 'PTEN', 'VHL', 'NF1',
            'CDKN2A', 'ATM', 'CHEK2', 'PALB2', 'CDH1', 'STK11'
        }
        
        self.dna_repair_genes = {
            'BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'PALB2', 'RAD51C', 'RAD51D',
            'BARD1', 'BRIP1', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM'
        }
        
        # Cancer pathway mappings
        self.cancer_pathways = {
            'PI3K_AKT': ['PIK3CA', 'PTEN', 'AKT1', 'TSC1', 'TSC2'],
            'RAS_RAF_MEK': ['KRAS', 'NRAS', 'BRAF', 'NF1', 'MAP2K1'],
            'P53_PATHWAY': ['TP53', 'MDM2', 'MDM4', 'CDKN2A', 'RB1'],
            'WNT_SIGNALING': ['APC', 'CTNNB1', 'AXIN1', 'AXIN2'],
            'RTK_SIGNALING': ['EGFR', 'ERBB2', 'MET', 'ALK', 'ROS1'],
            'DNA_DAMAGE_RESPONSE': ['BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'TP53']
        }
    
    def create_comprehensive_mutation_analysis_prompt(
        self,
        gene: str,
        variant: str,
        patient_context: Optional[Dict] = None,
        analysis_type: str = 'comprehensive'
    ) -> str:
        """Create comprehensive mutation analysis prompt equivalent to Bedrock implementation"""
        
        # Determine gene classification
        gene_classification = self._classify_gene(gene)
        pathway_info = self._get_pathway_information(gene)
        
        # Build context-aware prompt
        context_section = self._build_patient_context(patient_context) if patient_context else ""
        
        prompt = f"""You are a world-class computational genomicist and clinical cancer geneticist with expertise in precision oncology. Analyze the following cancer mutation with the depth and rigor of a molecular tumor board.

MUTATION TO ANALYZE:
Gene: {gene}
Variant: {variant}
Gene Classification: {gene_classification}
Pathway Involvement: {pathway_info}

{context_section}

ANALYSIS FRAMEWORK:
Provide a comprehensive analysis addressing each of the following domains:

1. FUNCTIONAL IMPACT PREDICTION:
   - Protein structure/function consequences
   - Domain-specific effects (if applicable)
   - Conservation analysis implications
   - Predicted biochemical changes

2. PATHOGENICITY ASSESSMENT:
   - Clinical significance classification (ACMG/AMP criteria)
   - Evidence supporting pathogenicity
   - Population frequency considerations
   - Computational prediction concordance

3. CANCER TYPE ASSOCIATIONS:
   - Primary cancer associations
   - Secondary cancer risks
   - Tissue-specific expression patterns
   - Penetrance estimates by cancer type

4. THERAPEUTIC IMPLICATIONS:
   - FDA-approved targeted therapies
   - Investigational agents
   - Drug resistance mechanisms
   - Combination therapy opportunities
   - Biomarker status for immunotherapy

5. PROGNOSTIC SIGNIFICANCE:
   - Overall survival impact
   - Disease-free survival associations
   - Treatment response patterns
   - Risk stratification implications

6. CLINICAL MANAGEMENT:
   - Screening recommendations
   - Surveillance protocols
   - Preventive interventions
   - Family screening considerations

CRITICAL ANALYSIS REQUIREMENTS:
- Base analysis on peer-reviewed literature and established databases
- Consider variant-specific evidence over gene-level associations
- Account for population ancestry and frequency data
- Integrate structural biology insights where available
- Address uncertainty and confidence levels explicitly

RESPONSE FORMAT:
Provide your analysis in the following structured JSON format:

{{
  "mutation_summary": {{
    "gene": "{gene}",
    "variant": "{variant}",
    "protein_change": "<predicted protein change>",
    "mutation_type": "<missense/nonsense/frameshift/splice/etc>",
    "domain_affected": "<protein domain if known>"
  }},
  
  "functional_impact": {{
    "predicted_effect": "<detailed functional prediction>",
    "mechanism": "<molecular mechanism of dysfunction>",
    "structural_impact": "<protein structure consequences>",
    "conservation_score": <float 0.0-1.0>,
    "functional_confidence": <float 0.0-1.0>
  }},
  
  "pathogenicity_assessment": {{
    "clinical_significance": "<PATHOGENIC/LIKELY_PATHOGENIC/UNCERTAIN/LIKELY_BENIGN/BENIGN>",
    "pathogenicity_score": <float 0.0-1.0>,
    "acmg_criteria": ["<list of applicable ACMG criteria>"],
    "evidence_summary": "<summary of supporting evidence>",
    "population_frequency": <frequency if known>,
    "confidence_level": <float 0.0-1.0>
  }},
  
  "cancer_associations": {{
    "primary_cancers": [
      {{
        "cancer_type": "<cancer type>",
        "lifetime_risk": <percentage if known>,
        "penetrance": "<high/moderate/low>",
        "evidence_level": "<strong/moderate/limited>"
      }}
    ],
    "secondary_cancers": ["<list of secondary cancer risks>"],
    "cancer_syndromes": ["<associated hereditary cancer syndromes>"]
  }},
  
  "therapeutic_implications": {{
    "targeted_therapies": [
      {{
        "drug": "<drug name>",
        "mechanism": "<mechanism of action>",
        "approval_status": "<FDA approved/investigational>",
        "cancer_types": ["<applicable cancer types>"],
        "evidence_level": "<strong/moderate/limited>"
      }}
    ],
    "resistance_mechanisms": ["<potential resistance mechanisms>"],
    "combination_opportunities": ["<combination therapy options>"],
    "immunotherapy_biomarker": "<positive/negative/unknown>",
    "actionability_score": <float 0.0-1.0>
  }},
  
  "prognostic_significance": {{
    "overall_survival_impact": "<favorable/unfavorable/neutral/unknown>",
    "disease_free_survival": "<favorable/unfavorable/neutral/unknown>",
    "treatment_response": "<enhanced/reduced/neutral/unknown>",
    "prognostic_category": "<good/intermediate/poor/unknown>",
    "supporting_studies": ["<key supporting studies>"]
  }},
  
  "clinical_recommendations": {{
    "screening_guidelines": ["<specific screening recommendations>"],
    "surveillance_frequency": "<recommended frequency>",
    "preventive_interventions": ["<risk-reducing options>"],
    "family_testing": "<recommended/consider/not_indicated>",
    "genetic_counseling": "<strongly_recommended/recommended/optional>",
    "specialist_referrals": ["<recommended specialist consultations>"]
  }},
  
  "confidence_metrics": {{
    "overall_confidence": <float 0.0-1.0>,
    "evidence_quality": "<high/moderate/low>",
    "data_completeness": <float 0.0-1.0>,
    "clinical_actionability": <float 0.0-1.0>,
    "uncertainty_factors": ["<sources of uncertainty>"]
  }},
  
  "scientific_rationale": "<detailed explanation of the analysis, citing key evidence and addressing limitations>"
}}

IMPORTANT GUIDELINES:
- Provide specific, actionable clinical guidance
- Clearly distinguish between established facts and predictions
- Include confidence intervals where appropriate
- Address ethnic/population-specific considerations
- Consider variant-level evidence over gene-level generalizations
- Maintain clinical conservatism while being informative

IMPORTANT: Your entire response must be only the JSON object, starting with '{{' and ending with '}}'. Do not include any introductory text, explanations, or markdown formatting outside of the JSON."""

        return prompt
    
    def create_multi_mutation_analysis_prompt(
        self,
        mutations: List[Dict[str, str]],
        patient_context: Optional[Dict] = None
    ) -> str:
        """Create prompt for analyzing multiple mutations together"""
        
        mutations_text = "\n".join([
            f"- {mut['gene']}:{mut['variant']}" for mut in mutations
        ])
        
        # Identify pathway overlaps
        pathway_analysis = self._analyze_pathway_convergence(mutations)
        
        context_section = self._build_patient_context(patient_context) if patient_context else ""
        
        prompt = f"""You are a computational genomicist analyzing a complex multi-mutation profile for precision cancer medicine. This analysis requires understanding mutation interactions, pathway convergence, and combinatorial effects.

MUTATION PROFILE:
{mutations_text}

PATHWAY CONVERGENCE ANALYSIS:
{pathway_analysis}

{context_section}

COMPREHENSIVE MULTI-MUTATION ANALYSIS:

1. MUTATION INTERACTION ANALYSIS:
   - Synergistic pathogenic effects
   - Pathway convergence patterns
   - Compensatory mechanisms
   - Tumor evolution implications

2. COMPOSITE RISK ASSESSMENT:
   - Combined pathogenicity score
   - Interaction-modified penetrance
   - Multi-hit hypothesis validation
   - Risk amplification factors

3. THERAPEUTIC STRATEGY OPTIMIZATION:
   - Multi-target therapy opportunities
   - Sequential treatment strategies
   - Resistance pathway analysis
   - Combination therapy rationale

4. PATHWAY-BASED INTERPRETATION:
   - Dominant pathway disruptions
   - Functional redundancy analysis
   - Synthetic lethality opportunities
   - Pathway crosstalk effects

RESPONSE FORMAT:
{{
  "mutation_profile": {{
    "total_mutations": {len(mutations)},
    "pathogenic_count": "<count of pathogenic variants>",
    "dominant_pathways": ["<list of most affected pathways>"],
    "interaction_pattern": "<synergistic/additive/antagonistic/independent>"
  }},
  
  "composite_risk": {{
    "overall_pathogenicity": <float 0.0-1.0>,
    "lifetime_cancer_risk": <percentage estimate>,
    "risk_modification": "<amplified/additive/neutral>",
    "cancer_spectrum": ["<cancer types in order of risk>"],
    "penetrance_estimate": "<high/moderate/low>"
  }},
  
  "pathway_analysis": {{
    "disrupted_pathways": [
      {{
        "pathway": "<pathway name>",
        "genes_affected": ["<genes in this pathway>"],
        "disruption_severity": "<complete/partial/minor>",
        "functional_impact": "<description>"
      }}
    ],
    "pathway_interactions": "<description of pathway crosstalk>",
    "compensatory_mechanisms": ["<potential compensatory pathways>"]
  }},
  
  "therapeutic_strategy": {{
    "combination_therapies": [
      {{
        "target_combination": ["<drug targets>"],
        "rationale": "<scientific rationale>",
        "expected_efficacy": "<high/moderate/low>",
        "evidence_level": "<strong/moderate/limited>"
      }}
    ],
    "sequential_strategies": ["<treatment sequence recommendations>"],
    "resistance_mitigation": ["<strategies to prevent resistance>"],
    "precision_medicine_score": <float 0.0-1.0>
  }},
  
  "clinical_implications": {{
    "surveillance_strategy": "<enhanced/standard/modified>",
    "screening_modifications": ["<specific modifications>"],
    "family_counseling": "<complex/standard/limited>",
    "specialist_coordination": ["<required specialist team>"]
  }},
  
  "research_opportunities": {{
    "clinical_trial_eligibility": ["<relevant trial types>"],
    "biomarker_studies": ["<relevant biomarker research>"],
    "functional_studies_needed": ["<recommended functional studies>"]
  }},
  
  "confidence_assessment": {{
    "analysis_confidence": <float 0.0-1.0>,
    "interaction_evidence": "<strong/moderate/limited/theoretical>",
    "clinical_actionability": <float 0.0-1.0>,
    "uncertainty_sources": ["<major sources of uncertainty>"]
  }},
  
  "comprehensive_interpretation": "<detailed clinical interpretation integrating all findings>"
}}

Focus on clinically actionable insights and provide clear rationale for all recommendations.

IMPORTANT: Your entire response must be only the JSON object, starting with '{{' and ending with '}}'. Do not include any introductory text, explanations, or markdown formatting outside of the JSON."""
        
        return prompt
    
    def create_risk_stratification_prompt(
        self,
        mutations: List[Dict],
        family_history: Optional[List[str]] = None,
        demographics: Optional[Dict] = None
    ) -> str:
        """Create prompt for comprehensive risk stratification"""
        
        mutations_summary = self._summarize_mutations(mutations)
        family_context = self._format_family_history(family_history) if family_history else "No family history provided"
        demo_context = self._format_demographics(demographics) if demographics else "Limited demographic information"
        
        prompt = f"""You are a cancer geneticist performing comprehensive risk stratification for personalized cancer prevention and early detection. Integrate genetic, familial, and demographic factors for precise risk assessment.

GENETIC PROFILE:
{mutations_summary}

FAMILY HISTORY:
{family_context}

DEMOGRAPHICS:
{demo_context}

COMPREHENSIVE RISK STRATIFICATION:

1. GENETIC RISK COMPONENT:
   - Individual mutation contributions
   - Mutation interaction effects
   - Penetrance calculations
   - Modifier gene considerations

2. FAMILIAL RISK INTEGRATION:
   - Family history impact
   - Inheritance pattern analysis
   - Shared environment factors
   - Polygenic risk considerations

3. DEMOGRAPHIC RISK FACTORS:
   - Age-specific risk curves
   - Ethnicity-specific considerations
   - Gender-related risk modifications
   - Geographic/environmental factors

4. COMPOSITE RISK MODELING:
   - Integrated risk score
   - Cancer-specific risks
   - Time-dependent risk projections
   - Confidence intervals

PROVIDE DETAILED RISK ASSESSMENT:
{{
  "genetic_risk_profile": {{
    "primary_genetic_risk": <float 0.0-1.0>,
    "high_penetrance_mutations": ["<list>"],
    "moderate_penetrance_mutations": ["<list>"],
    "polygenic_risk_estimate": <float 0.0-1.0>,
    "genetic_risk_category": "<very_high/high/moderate/average/low>"
  }},
  
  "familial_risk_assessment": {{
    "family_history_contribution": <float 0.0-1.0>,
    "inheritance_pattern": "<autosomal_dominant/recessive/polygenic/sporadic>",
    "shared_environment_risk": <float 0.0-1.0>,
    "family_risk_category": "<strong/moderate/weak/none>"
  }},
  
  "demographic_adjustments": {{
    "age_adjusted_risk": <float 0.0-1.0>,
    "ethnicity_specific_factors": ["<relevant factors>"],
    "gender_specific_considerations": ["<relevant factors>"],
    "environmental_modifiers": ["<relevant factors>"]
  }},
  
  "integrated_risk_assessment": {{
    "lifetime_cancer_risk": {{
      "overall": <percentage>,
      "breast": <percentage if applicable>,
      "ovarian": <percentage if applicable>,
      "colorectal": <percentage if applicable>,
      "prostate": <percentage if applicable>,
      "other_cancers": {{
        "<cancer_type>": <percentage>
      }}
    }},
    "age_specific_risks": {{
      "by_age_40": <percentage>,
      "by_age_50": <percentage>,
      "by_age_60": <percentage>,
      "by_age_70": <percentage>,
      "by_age_80": <percentage>
    }},
    "risk_category": "<very_high/high/moderate_high/moderate/average/below_average>",
    "risk_confidence": <float 0.0-1.0>
  }},
  
  "clinical_recommendations": {{
    "surveillance_protocol": {{
      "screening_intensity": "<enhanced/standard/reduced>",
      "screening_modalities": ["<recommended modalities>"],
      "screening_intervals": {{
        "<modality>": "<frequency>"
      }},
      "starting_age": "<age or immediate>"
    }},
    "prevention_strategies": {{
      "risk_reducing_surgeries": ["<if applicable>"],
      "chemoprevention": ["<options if applicable>"],
      "lifestyle_modifications": ["<specific recommendations>"],
      "environmental_modifications": ["<relevant modifications>"]
    }},
    "genetic_counseling": {{
      "urgency": "<immediate/routine/optional>",
      "focus_areas": ["<key counseling topics>"],
      "family_implications": ["<family testing recommendations>"]
    }}
  }},
  
  "monitoring_strategy": {{
    "biomarker_monitoring": ["<relevant biomarkers>"],
    "imaging_protocols": ["<specific imaging recommendations>"],
    "clinical_assessment_frequency": "<frequency>",
    "specialist_coordination": ["<required specialists>"]
  }},
  
  "risk_communication": {{
    "patient_friendly_summary": "<clear, non-technical explanation>",
    "key_takeaways": ["<3-5 key points>"],
    "action_items": ["<immediate action items>"],
    "long_term_planning": ["<long-term considerations>"]
  }}
}}

Ensure recommendations are evidence-based and clinically actionable.

IMPORTANT: Your entire response must be only the JSON object, starting with '{{' and ending with '}}'. Do not include any introductory text, explanations, or markdown formatting outside of the JSON."""
        
        return prompt
    
    def create_therapeutic_prioritization_prompt(
        self,
        mutations: List[Dict],
        cancer_type: Optional[str] = None,
        treatment_history: Optional[List[str]] = None
    ) -> str:
        """Create prompt for therapeutic option prioritization"""
        
        mutations_text = self._format_mutations_for_therapy(mutations)
        cancer_context = f"Cancer Type: {cancer_type}" if cancer_type else "Cancer type not specified"
        treatment_context = f"Previous treatments: {', '.join(treatment_history)}" if treatment_history else "Treatment-naive"
        
        prompt = f"""You are a precision oncology specialist developing personalized therapeutic strategies based on molecular profiling. Prioritize treatment options considering mutation profile, cancer type, and treatment history.

MOLECULAR PROFILE:
{mutations_text}

CLINICAL CONTEXT:
{cancer_context}
{treatment_context}

THERAPEUTIC STRATEGY DEVELOPMENT:

1. TARGETED THERAPY PRIORITIZATION:
   - Primary therapeutic targets
   - Evidence levels for each option
   - Expected response rates
   - Resistance mechanisms

2. COMBINATION THERAPY OPPORTUNITIES:
   - Synergistic combinations
   - Sequential therapy strategies
   - Biomarker-guided combinations
   - Safety considerations

3. CLINICAL TRIAL OPPORTUNITIES:
   - Matching clinical trials
   - Novel therapeutic approaches
   - Investigational combinations
   - Eligibility considerations

4. RESISTANCE MITIGATION:
   - Predicted resistance mechanisms
   - Monitoring strategies
   - Alternative pathway targeting
   - Combination approaches

THERAPEUTIC RECOMMENDATIONS:
{{
  "primary_targets": [
    {{
      "target": "<molecular target>",
      "drugs": ["<drug options>"],
      "evidence_level": "<FDA_approved/phase_3/phase_2/phase_1/preclinical>",
      "expected_response_rate": "<percentage or range>",
      "mutation_specificity": "<high/moderate/low>",
      "priority_ranking": <1-10>
    }}
  ],
  
  "combination_strategies": [
    {{
      "combination_type": "<targeted/immuno/chemo combination>",
      "components": ["<therapy components>"],
      "rationale": "<scientific rationale>",
      "evidence_level": "<strong/moderate/limited/theoretical>",
      "expected_benefit": "<high/moderate/low>",
      "safety_profile": "<favorable/acceptable/concerning>"
    }}
  ],
  
  "clinical_trial_opportunities": [
    {{
      "trial_type": "<drug/combination/biomarker study>",
      "intervention": "<investigational approach>",
      "eligibility_match": "<high/moderate/low>",
      "potential_benefit": "<high/moderate/low>",
      "trial_phase": "<phase_1/phase_2/phase_3>"
    }}
  ],
  
  "resistance_management": {{
    "predicted_resistance": ["<likely resistance mechanisms>"],
    "monitoring_biomarkers": ["<biomarkers to track>"],
    "backup_strategies": ["<alternative approaches>"],
    "combination_approaches": ["<resistance prevention strategies>"]
  }},
  
  "treatment_sequencing": {{
    "first_line_recommendation": "<specific recommendation>",
    "second_line_options": ["<backup options>"],
    "long_term_strategy": "<overall treatment approach>",
    "monitoring_plan": ["<key monitoring parameters>"]
  }},
  
  "precision_medicine_score": {{
    "actionability_score": <float 0.0-1.0>,
    "evidence_strength": "<strong/moderate/limited>",
    "clinical_utility": "<high/moderate/low>",
    "therapeutic_options": <number_of_options>
  }},
  
  "clinical_implementation": {{
    "immediate_actions": ["<immediate steps>"],
    "specialist_consultations": ["<required consultations>"],
    "timing_considerations": ["<timing factors>"],
    "patient_counseling_points": ["<key discussion points>"]
  }}
}}

Focus on evidence-based recommendations with clear rationale and practical implementation guidance.

IMPORTANT: Your entire response must be only the JSON object, starting with '{{' and ending with '}}'. Do not include any introductory text, explanations, or markdown formatting outside of the JSON."""
        
        return prompt
    
    # Helper methods
    def _classify_gene(self, gene: str) -> str:
        """Classify gene by function"""
        if gene in self.oncogenes:
            return "Oncogene"
        elif gene in self.tumor_suppressors:
            return "Tumor Suppressor"
        elif gene in self.dna_repair_genes:
            return "DNA Repair Gene"
        else:
            return "Other Cancer-Associated Gene"
    
    def _get_pathway_information(self, gene: str) -> str:
        """Get pathway information for gene"""
        pathways = []
        for pathway, genes in self.cancer_pathways.items():
            if gene in genes:
                pathways.append(pathway)
        
        if pathways:
            return f"Involved in: {', '.join(pathways)}"
        else:
            return "Pathway involvement: Variable/Unknown"
    
    def _build_patient_context(self, patient_context: Dict) -> str:
        """Build patient context section including clustering analysis"""
        context_lines = ["PATIENT CONTEXT:"]
        
        if patient_context.get('age'):
            context_lines.append(f"Age: {patient_context['age']}")
        
        if patient_context.get('sex'):
            context_lines.append(f"Sex: {patient_context['sex']}")
        
        if patient_context.get('cancer_type'):
            context_lines.append(f"Cancer Type: {patient_context['cancer_type']}")
        
        if patient_context.get('family_history'):
            context_lines.append(f"Family History: {', '.join(patient_context['family_history'])}")
        
        if patient_context.get('ethnicity'):
            context_lines.append(f"Ethnicity: {patient_context['ethnicity']}")
        
        # Include clustering analysis if present
        if patient_context.get('clustering_analysis'):
            cluster_data = patient_context['clustering_analysis']
            context_lines.append("\nCLUSTERING ANALYSIS RESULTS:")
            
            if cluster_data.get('pathway_convergence'):
                context_lines.append(f"Pathway Convergence: {cluster_data['pathway_convergence']}")
            
            if cluster_data.get('functional_groups'):
                context_lines.append(f"Functional Groups: {cluster_data['functional_groups']}")
            
            if cluster_data.get('interaction_patterns'):
                context_lines.append(f"Interaction Patterns: {cluster_data['interaction_patterns']}")
            
            if cluster_data.get('clustering_summary'):
                context_lines.append(f"Clustering Insights: {', '.join(cluster_data['clustering_summary'])}")
        
        return "\n".join(context_lines) + "\n"
    
    def _analyze_pathway_convergence(self, mutations: List[Dict]) -> str:
        """Analyze pathway convergence for multiple mutations"""
        pathway_counts = {}
        for mutation in mutations:
            gene = mutation['gene']
            for pathway, genes in self.cancer_pathways.items():
                if gene in genes:
                    pathway_counts[pathway] = pathway_counts.get(pathway, 0) + 1
        
        if pathway_counts:
            convergent_pathways = [p for p, count in pathway_counts.items() if count > 1]
            if convergent_pathways:
                return f"Pathway Convergence Detected: {', '.join(convergent_pathways)}"
            else:
                return f"Pathways Affected: {', '.join(pathway_counts.keys())}"
        else:
            return "No major pathway convergence identified"
    
    def _summarize_mutations(self, mutations: List[Dict]) -> str:
        """Summarize mutations for risk stratification"""
        summary_lines = []
        for mut in mutations:
            gene = mut.get('gene', 'Unknown')
            variant = mut.get('variant', 'Unknown')
            significance = mut.get('clinical_significance', 'Unknown')
            summary_lines.append(f"- {gene}:{variant} ({significance})")
        
        return "\n".join(summary_lines)
    
    def _format_family_history(self, family_history: List[str]) -> str:
        """Format family history information"""
        if not family_history:
            return "No family history of cancer reported"
        
        return f"Family cancer history: {', '.join(family_history)}"
    
    def _format_demographics(self, demographics: Dict) -> str:
        """Format demographic information"""
        demo_lines = []
        
        if demographics.get('age'):
            demo_lines.append(f"Age: {demographics['age']}")
        
        if demographics.get('sex'):
            demo_lines.append(f"Sex: {demographics['sex']}")
        
        if demographics.get('ethnicity'):
            demo_lines.append(f"Ethnicity: {demographics['ethnicity']}")
        
        if demographics.get('geographic_region'):
            demo_lines.append(f"Geographic Region: {demographics['geographic_region']}")
        
        return "\n".join(demo_lines) if demo_lines else "Limited demographic information available"
    
    def _format_mutations_for_therapy(self, mutations: List[Dict]) -> str:
        """Format mutations for therapeutic analysis"""
        therapy_lines = []
        
        for mut in mutations:
            gene = mut.get('gene', 'Unknown')
            variant = mut.get('variant', 'Unknown')
            therapies = mut.get('targeted_therapies', [])
            
            therapy_text = f"- {gene}:{variant}"
            if therapies:
                therapy_text += f" (Current targets: {', '.join(therapies[:3])})"
            
            therapy_lines.append(therapy_text)
        
        return "\n".join(therapy_lines)

# Example usage and template prompts for specific scenarios
ONCOLOGY_CONSULTATION_PROMPT = """
You are providing oncology consultation for a patient with complex genetic findings. 
Focus on:
1. Immediate clinical actions required
2. Treatment strategy modifications
3. Monitoring and surveillance needs
4. Prognosis and counseling points

Provide clear, actionable recommendations for the treating oncologist.
"""

GENETIC_COUNSELING_PROMPT = """
You are preparing content for genetic counseling session.
Focus on:
1. Risk communication in patient-friendly terms
2. Family implications and testing recommendations
3. Reproductive considerations
4. Psychological support needs
5. Long-term management planning

Provide empathetic, clear guidance for patient counseling.
"""

MOLECULAR_TUMOR_BOARD_PROMPT = """
You are presenting to a molecular tumor board.
Focus on:
1. Mutation functional significance
2. Therapeutic targeting opportunities
3. Clinical trial matching
4. Resistance mechanisms and monitoring
5. Novel therapeutic approaches

Provide comprehensive analysis for multidisciplinary team decision-making.
"""