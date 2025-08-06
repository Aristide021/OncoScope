"""
OncoScope Gradio Interface
Molecular Tumor Board in a Box
"""

import gradio as gr
import requests
import json
from datetime import datetime
import pandas as pd

# API Configuration
API_URL = "http://localhost:8000"

# Pre-defined example mutations for demo
EXAMPLE_MUTATIONS = {
    "Breast Cancer Panel": ["BRCA1 c.5266dupC", "BRCA2 c.9097dupA", "TP53 c.818G>A", "PIK3CA c.3140A>G"],
    "Lung Cancer Panel": ["EGFR c.2369C>T", "KRAS c.35G>A", "ALK fusion", "ROS1 rearrangement"],
    "Colorectal Cancer Panel": ["KRAS c.35G>A", "BRAF c.1799T>A", "MSI-High", "PIK3CA c.1633G>A"],
    "Single Mutation": ["BRAF c.1799T>A"]
}

def analyze_mutations(mutations_text, patient_age, patient_gender, cancer_type, diagnosis):
    """Send mutations to OncoScope API for analysis"""
    
    # Parse mutations (one per line)
    mutations = [m.strip() for m in mutations_text.strip().split('\n') if m.strip()]
    
    if not mutations:
        return "❌ Please enter at least one mutation", None, None
    
    # Prepare request
    payload = {
        "mutations": mutations,
        "patient_info": {
            "age": int(patient_age) if patient_age else 50,
            "gender": patient_gender.lower(),
            "cancer_type": cancer_type.lower(),
            "diagnosis": diagnosis.strip() if diagnosis else None
        }
    }
    
    try:
        # Call API
        response = requests.post(f"{API_URL}/analyze/mutations", json=payload)
        response.raise_for_status()
        
        data = response.json()
        
        # Format results
        summary = format_analysis_summary(data)
        mutations_df = format_mutations_table(data)
        clustering_viz = format_clustering_results(data)
        
        return summary, mutations_df, clustering_viz
        
    except requests.exceptions.ConnectionError:
        return "❌ Error: OncoScope API is not running. Please start the server with 'uvicorn backend.main:app'", None, None
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

def format_analysis_summary(data):
    """Format the analysis summary"""
    
    if not data.get("success"):
        return "❌ Analysis failed"
    
    summary = f"""
# 🧬 OncoScope Analysis Complete

**Analysis ID:** `{data.get('analysis_id', 'N/A')}`  
**Timestamp:** {data.get('timestamp', 'N/A')}  
**Mutations Analyzed:** {len(data.get('individual_mutations', []))}

## 🎯 Key Findings:
"""
    
    # Add pathogenicity summary
    mutations = data.get('individual_mutations', [])
    if mutations:
        pathogenic = sum(1 for m in mutations if 'PATHOGENIC' in m.get('clinical_significance', ''))
        uncertain = sum(1 for m in mutations if 'UNCERTAIN' in m.get('clinical_significance', ''))
        benign = sum(1 for m in mutations if 'BENIGN' in m.get('clinical_significance', ''))
        
        summary += f"""
- **Pathogenic/Likely Pathogenic:** {pathogenic} mutations
- **Uncertain Significance:** {uncertain} mutations  
- **Benign/Likely Benign:** {benign} mutations
"""
    
    # Add clustering insights if available
    if data.get('clustering_results'):
        clusters = data['clustering_results'].get('clusters_identified', 0)
        summary += f"\n## 🔬 Clustering Analysis:\n**{clusters} functional cluster(s) identified**\n"
        
        insights = data['clustering_results'].get('clustering_insights', {})
        if insights:
            summary += f"\n{insights.get('summary', '')}\n"
    
    return summary

def format_mutations_table(data):
    """Format mutations as a pandas DataFrame for display"""
    
    mutations = data.get('individual_mutations', [])
    if not mutations:
        return pd.DataFrame()
    
    # Extract key fields for table
    table_data = []
    for mut in mutations:
        table_data.append({
            'Gene': mut.get('gene', ''),
            'Variant': mut.get('variant', ''),
            'Pathogenicity': mut.get('clinical_significance', '').replace('_', ' ').title(),
            'Score': f"{mut.get('pathogenicity_score', 0):.2f}",
            'Therapies': ', '.join(mut.get('targeted_therapies', [])) or 'None identified',
            'Mechanism': mut.get('mechanism', 'Unknown')[:50] + '...' if len(mut.get('mechanism', '')) > 50 else mut.get('mechanism', 'Unknown')
        })
    
    return pd.DataFrame(table_data)

def format_clustering_results(data):
    """Format clustering visualization"""
    
    clustering = data.get('clustering_results', {})
    if not clustering or clustering.get('clusters_identified', 0) == 0:
        return "No clustering performed (requires 3+ mutations)"
    
    # Create a simple text visualization
    viz = f"""
## 🧬 Mutation Clustering Results

**Clusters Identified:** {clustering.get('clusters_identified', 0)}

### Cluster Analysis:
"""
    
    analysis = clustering.get('cluster_analysis', {})
    if 'pathway_convergence' in analysis:
        viz += "\n**Pathway Convergence:**\n"
        for pathway, genes in analysis['pathway_convergence'].items():
            viz += f"- {pathway}: {', '.join(genes)}\n"
    
    return viz

def load_example(example_name):
    """Load example mutations"""
    mutations = EXAMPLE_MUTATIONS.get(example_name, [])
    return '\n'.join(mutations)

# Create Gradio Interface
with gr.Blocks(title="OncoScope - Molecular Tumor Board in a Box", theme=gr.themes.Soft()) as app:
    
    gr.Markdown("""
    # 🧬 OncoScope: Molecular Tumor Board in a Box
    
    **Democratizing Cancer Genomics Analysis** - What costs $10,000 in commercial testing, now runs free on your device.
    
    This system provides:
    - 🔬 Pathogenicity assessment using ACMG/AMP guidelines
    - 💊 Targeted therapy recommendations
    - 🧪 Multi-mutation clustering analysis
    - 🎯 Context-aware analysis based on patient's diagnosis
    - 🔒 Complete privacy - all analysis runs locally
    
    **Powered by Gemma 3n E4B** - Fine-tuned on 5,998 cancer genomics examples
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
            
            with gr.Tab("Clustering Analysis"):
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
    OncoScope uses a fine-tuned Gemma 3n model to provide institutional-grade cancer genomics analysis 
    that runs entirely on your local device. No patient data ever leaves your computer.
    
    **Note:** This is a proof of concept. Always consult with healthcare professionals for medical decisions.
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates a public link
        show_error=True
    )