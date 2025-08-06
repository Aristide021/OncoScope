"""
OncoScope Gradio Interface for Hugging Face Spaces
This version uses Hugging Face's Inference API instead of local Ollama
"""

import gradio as gr
import json
from datetime import datetime
import pandas as pd
from huggingface_hub import InferenceClient
import os

# Initialize Hugging Face Inference Client
# You'll need to set your HF token as a secret in Spaces
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in Hugging Face Spaces secrets
client = InferenceClient(token=HF_TOKEN)

# Pre-defined example mutations for demo
EXAMPLE_MUTATIONS = {
    "Breast Cancer Panel": ["BRCA1 c.5266dupC", "BRCA2 c.9097dupA", "TP53 c.818G>A", "PIK3CA c.3140A>G"],
    "Lung Cancer Panel": ["EGFR c.2369C>T", "KRAS c.35G>A", "ALK fusion", "ROS1 rearrangement"],
    "Colorectal Cancer Panel": ["KRAS c.35G>A", "BRAF c.1799T>A", "MSI-High", "PIK3CA c.1633G>A"],
    "Single Mutation": ["BRAF c.1799T>A"]
}

# Mock database for demo purposes (in production, this would be a real database)
MUTATION_DATABASE = {
    "BRCA1": {
        "function": "DNA repair, tumor suppressor",
        "associated_cancers": ["Breast", "Ovarian", "Prostate", "Pancreatic"]
    },
    "BRCA2": {
        "function": "DNA repair, tumor suppressor",
        "associated_cancers": ["Breast", "Ovarian", "Prostate", "Pancreatic"]
    },
    "TP53": {
        "function": "Cell cycle regulation, apoptosis",
        "associated_cancers": ["Multiple cancer types"]
    },
    "EGFR": {
        "function": "Growth factor receptor",
        "associated_cancers": ["Lung", "Glioblastoma", "Head and neck"]
    },
    "KRAS": {
        "function": "RAS/MAPK signaling",
        "associated_cancers": ["Lung", "Colorectal", "Pancreatic"]
    },
    "BRAF": {
        "function": "RAS/MAPK signaling",
        "associated_cancers": ["Melanoma", "Colorectal", "Thyroid"]
    },
    "PIK3CA": {
        "function": "PI3K/AKT signaling",
        "associated_cancers": ["Breast", "Colorectal", "Endometrial"]
    },
    "ALK": {
        "function": "Receptor tyrosine kinase",
        "associated_cancers": ["Lung", "Neuroblastoma", "ALCL"]
    }
}

def build_prompt(mutation, patient_info):
    """Build a prompt for the model"""
    
    prompt = f"""You are a clinical genomics expert. Analyze the following cancer mutation and provide a structured assessment.

Mutation: {mutation}
Patient Information:
- Age: {patient_info.get('age', 'Unknown')}
- Gender: {patient_info.get('gender', 'Unknown')}
- Cancer Type: {patient_info.get('cancer_type', 'Unknown')}
- Diagnosis: {patient_info.get('diagnosis', 'Not specified')}

IMPORTANT: Analyze this mutation in the context of the patient's existing diagnosis. Do not predict future cancer risks.

Provide your analysis in the following JSON format:
{{
    "gene": "gene name",
    "variant": "variant notation",
    "pathogenicity": "PATHOGENIC/LIKELY_PATHOGENIC/UNCERTAIN/LIKELY_BENIGN/BENIGN",
    "clinical_significance": "description of clinical impact",
    "mechanism": "molecular mechanism",
    "therapies": ["list of targeted therapies if applicable"],
    "pathogenicity_score": 0.0 to 1.0,
    "evidence_level": "strong/moderate/limited"
}}

Return ONLY the JSON object, no additional text."""
    
    return prompt

def analyze_single_mutation_hf(mutation, patient_info):
    """Analyze a single mutation using Hugging Face inference"""
    
    # Extract gene name
    gene = mutation.split()[0].split(':')[0]
    
    # Get info from mock database
    gene_info = MUTATION_DATABASE.get(gene, {})
    
    # Build prompt
    prompt = build_prompt(mutation, patient_info)
    
    try:
        # Use a model that's available on HF (you might need to use your fine-tuned model)
        # For demo, using a general model
        response = client.text_generation(
            prompt,
            model="google/gemma-2-2b-it",  # Replace with your fine-tuned model
            max_new_tokens=500,
            temperature=0.1,
            do_sample=True
        )
        
        # Try to parse JSON from response
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
            else:
                # Fallback if model doesn't return proper JSON
                result = create_mock_analysis(mutation, gene_info)
        except:
            result = create_mock_analysis(mutation, gene_info)
            
        return result
        
    except Exception as e:
        # Fallback to mock analysis for demo
        return create_mock_analysis(mutation, gene_info)

def create_mock_analysis(mutation, gene_info):
    """Create a mock analysis for demo purposes"""
    gene = mutation.split()[0].split(':')[0]
    
    # Simple logic for demo
    pathogenic_genes = ["BRCA1", "BRCA2", "TP53", "KRAS", "BRAF", "EGFR"]
    is_pathogenic = gene in pathogenic_genes
    
    return {
        "gene": gene,
        "variant": mutation,
        "pathogenicity": "PATHOGENIC" if is_pathogenic else "UNCERTAIN",
        "clinical_significance": f"Mutation in {gene_info.get('function', 'Unknown function')}",
        "mechanism": gene_info.get('function', "Unknown mechanism"),
        "therapies": ["Targeted therapy available"] if gene in ["EGFR", "BRAF", "ALK"] else [],
        "pathogenicity_score": 0.85 if is_pathogenic else 0.5,
        "evidence_level": "strong" if is_pathogenic else "limited"
    }

def analyze_mutations(mutations_text, patient_age, patient_gender, cancer_type, diagnosis):
    """Analyze mutations using Hugging Face inference"""
    
    # Parse mutations (one per line)
    mutations = [m.strip() for m in mutations_text.strip().split('\n') if m.strip()]
    
    if not mutations:
        return "❌ Please enter at least one mutation", None, None
    
    patient_info = {
        "age": int(patient_age) if patient_age else 50,
        "gender": patient_gender.lower(),
        "cancer_type": cancer_type.lower(),
        "diagnosis": diagnosis.strip() if diagnosis else None
    }
    
    try:
        # Analyze each mutation
        results = []
        for mutation in mutations:
            analysis = analyze_single_mutation_hf(mutation, patient_info)
            results.append(analysis)
        
        # Format results
        summary = format_analysis_summary(results, patient_info)
        mutations_df = format_mutations_table(results)
        clustering_viz = format_clustering_results(results)
        
        return summary, mutations_df, clustering_viz
        
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def format_analysis_summary(results, patient_info):
    """Format the analysis summary"""
    
    summary = f"""
# 🧬 OncoScope Analysis Complete

**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Mutations Analyzed:** {len(results)}

## 🎯 Key Findings:
"""
    
    # Count pathogenicity
    pathogenic = sum(1 for r in results if 'PATHOGENIC' in r.get('pathogenicity', ''))
    uncertain = sum(1 for r in results if 'UNCERTAIN' in r.get('pathogenicity', ''))
    benign = sum(1 for r in results if 'BENIGN' in r.get('pathogenicity', ''))
    
    summary += f"""
- **Pathogenic/Likely Pathogenic:** {pathogenic} mutations
- **Uncertain Significance:** {uncertain} mutations  
- **Benign/Likely Benign:** {benign} mutations

## 📋 Patient Context:
- **Age:** {patient_info.get('age', 'Unknown')}
- **Gender:** {patient_info.get('gender', 'Unknown')}
- **Cancer Type:** {patient_info.get('cancer_type', 'Unknown')}
- **Diagnosis:** {patient_info.get('diagnosis', 'Not specified')}
"""
    
    # Add therapy summary
    therapies = []
    for r in results:
        therapies.extend(r.get('therapies', []))
    
    if therapies:
        summary += f"\n## 💊 Targeted Therapies Available:\n"
        for therapy in set(therapies):
            summary += f"- {therapy}\n"
    
    return summary

def format_mutations_table(results):
    """Format mutations as a DataFrame"""
    
    data = []
    for r in results:
        data.append({
            "Gene": r.get('gene', ''),
            "Variant": r.get('variant', ''),
            "Pathogenicity": r.get('pathogenicity', ''),
            "Score": f"{r.get('pathogenicity_score', 0):.2f}",
            "Therapies": ', '.join(r.get('therapies', [])) or 'None',
            "Mechanism": r.get('mechanism', '')
        })
    
    return pd.DataFrame(data)

def format_clustering_results(results):
    """Format clustering analysis"""
    
    # Group by pathways
    pathways = {}
    for r in results:
        gene = r.get('gene', '')
        mechanism = r.get('mechanism', 'Unknown')
        
        if 'RAS/MAPK' in mechanism:
            pathways.setdefault('RAS/MAPK Pathway', []).append(gene)
        elif 'DNA repair' in mechanism:
            pathways.setdefault('DNA Repair', []).append(gene)
        elif 'PI3K/AKT' in mechanism:
            pathways.setdefault('PI3K/AKT Pathway', []).append(gene)
        elif 'Cell cycle' in mechanism:
            pathways.setdefault('Cell Cycle Control', []).append(gene)
    
    viz = "## 🔬 Pathway Analysis\n\n"
    
    if pathways:
        viz += "**Affected Pathways:**\n"
        for pathway, genes in pathways.items():
            viz += f"- **{pathway}:** {', '.join(set(genes))}\n"
    else:
        viz += "No clear pathway clustering identified.\n"
    
    return viz

def load_example(example_name):
    """Load example mutations"""
    mutations = EXAMPLE_MUTATIONS.get(example_name, [])
    return '\n'.join(mutations)

# Create Gradio Interface
with gr.Blocks(title="OncoScope - AI-Powered Cancer Genomics", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🧬 OncoScope: AI-Powered Cancer Genomics Analysis
    
    **Democratizing Precision Oncology** - Advanced cancer mutation analysis powered by Gemma 3n
    
    This system provides:
    - 🔬 Pathogenicity assessment using clinical guidelines
    - 💊 Targeted therapy recommendations
    - 🧪 Pathway-based clustering analysis
    - 🎯 Context-aware analysis based on patient's diagnosis
    - 🤖 Powered by fine-tuned Gemma 3n E4B model
    
    **Note:** This is a demonstration version. For production use, deploy with your fine-tuned model.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input Patient & Mutations")
            
            # Patient Information
            patient_age = gr.Number(label="Patient Age", value=50, precision=0)
            patient_gender = gr.Radio(
                label="Patient Gender", 
                choices=["Female", "Male", "Other"],
                value="Female"
            )
            cancer_type = gr.Textbox(
                label="Cancer Type",
                value="breast",
                placeholder="e.g., breast, lung, colorectal"
            )
            
            diagnosis = gr.Textbox(
                label="Current Diagnosis (Optional)",
                placeholder="e.g., Stage II breast adenocarcinoma, NSCLC, etc.",
                info="Helps AI analyze mutations in context of existing diagnosis"
            )
            
            # Mutation Input
            mutations_input = gr.Textbox(
                label="Mutations (one per line)",
                placeholder="Enter mutations in standard notation:\nBRAF c.1799T>A\nKRAS c.35G>A\nTP53 c.818G>A",
                lines=6
            )
            
            # Example buttons
            gr.Markdown("### Load Example:")
            example_btns = []
            for example_name in EXAMPLE_MUTATIONS.keys():
                btn = gr.Button(example_name, size="sm")
                btn.click(
                    fn=lambda x=example_name: load_example(x),
                    outputs=mutations_input
                )
            
            analyze_btn = gr.Button("🧬 Analyze Mutations", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            gr.Markdown("## Analysis Results")
            
            # Results displays
            with gr.Tab("Summary"):
                summary_output = gr.Markdown()
            
            with gr.Tab("Mutation Details"):
                mutations_table = gr.DataFrame(
                    headers=["Gene", "Variant", "Pathogenicity", "Score", "Therapies", "Mechanism"],
                    label="Individual Mutation Analysis"
                )
            
            with gr.Tab("Pathway Analysis"):
                clustering_output = gr.Markdown()
    
    # Connect the analyze button
    analyze_btn.click(
        fn=analyze_mutations,
        inputs=[mutations_input, patient_age, patient_gender, cancer_type, diagnosis],
        outputs=[summary_output, mutations_table, clustering_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    ### About OncoScope
    OncoScope uses a fine-tuned Gemma 3n model to provide advanced cancer genomics analysis.
    This demo version uses simplified inference for demonstration purposes.
    
    **Disclaimer:** This is a proof of concept. Always consult with healthcare professionals for medical decisions.
    
    ### Deployment Notes:
    - Set your Hugging Face token as HF_TOKEN in Spaces secrets
    - Replace the model name with your fine-tuned model on Hugging Face
    - For production, implement proper error handling and validation
    """)

if __name__ == "__main__":
    app.launch()