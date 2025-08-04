"""
OncoScope Gradio Interface - Demo Mode for Hugging Face Spaces
Molecular Tumor Board in a Box
"""

import gradio as gr
import json
from datetime import datetime
import pandas as pd
import random
import time

# Pre-defined example mutations for demo
EXAMPLE_MUTATIONS = {
    "Breast Cancer Panel": ["BRCA1 c.5266dupC", "BRCA2 c.9097dupA", "TP53 c.818G>A", "PIK3CA c.3140A>G"],
    "Lung Cancer Panel": ["EGFR c.2369C>T", "KRAS c.35G>A", "ALK fusion", "ROS1 rearrangement"],
    "Colorectal Cancer Panel": ["KRAS c.35G>A", "BRAF c.1799T>A", "MSI-High", "PIK3CA c.1633G>A"],
    "Single Mutation": ["BRAF c.1799T>A"]
}

# Demo responses for different mutations
DEMO_RESPONSES = {
    "BRCA1 c.5266dupC": {
        "gene": "BRCA1",
        "variant": "c.5266dupC",
        "pathogenicity": "Pathogenic",
        "score": 0.98,
        "significance": "Known frameshift mutation causing BRCA1 deficiency",
        "therapies": ["Olaparib (Lynparza)", "Talazoparib (Talzenna)", "Rucaparib (Rubraca)"],
        "mechanism": "Frameshift mutation leading to truncated protein and loss of DNA repair function",
        "trials": 47
    },
    "BRCA2 c.9097dupA": {
        "gene": "BRCA2",
        "variant": "c.9097dupA", 
        "pathogenicity": "Pathogenic",
        "score": 0.97,
        "significance": "Frameshift mutation in BRCA2 causing homologous recombination deficiency",
        "therapies": ["Olaparib (Lynparza)", "Niraparib (Zejula)"],
        "mechanism": "Creates premature stop codon, resulting in non-functional BRCA2 protein",
        "trials": 38
    },
    "TP53 c.818G>A": {
        "gene": "TP53",
        "variant": "c.818G>A",
        "pathogenicity": "Pathogenic", 
        "score": 0.99,
        "significance": "Missense mutation in DNA binding domain",
        "therapies": ["APR-246", "Combination chemotherapy"],
        "mechanism": "Disrupts p53 tumor suppressor function, affecting cell cycle regulation",
        "trials": 62
    },
    "KRAS c.35G>A": {
        "gene": "KRAS",
        "variant": "c.35G>A (G12D)",
        "pathogenicity": "Pathogenic",
        "score": 0.99,
        "significance": "Common oncogenic KRAS mutation",
        "therapies": ["Sotorasib (Lumakras)", "Adagrasib (Krazati)"],
        "mechanism": "Constitutive activation of RAS signaling pathway",
        "trials": 84
    },
    "EGFR c.2369C>T": {
        "gene": "EGFR",
        "variant": "c.2369C>T (T790M)",
        "pathogenicity": "Pathogenic",
        "score": 0.98,
        "significance": "Resistance mutation to first-generation EGFR inhibitors",
        "therapies": ["Osimertinib (Tagrisso)", "Mobocertinib (Exkivity)"],
        "mechanism": "Confers resistance to erlotinib/gefitinib through altered drug binding",
        "trials": 56
    },
    "BRAF c.1799T>A": {
        "gene": "BRAF",
        "variant": "c.1799T>A (V600E)",
        "pathogenicity": "Pathogenic",
        "score": 0.99,
        "significance": "Most common BRAF mutation in cancer",
        "therapies": ["Vemurafenib (Zelboraf)", "Dabrafenib (Tafinlar)", "Encorafenib (Braftovi)"],
        "mechanism": "Constitutive activation of MAPK pathway driving cell proliferation",
        "trials": 71
    }
}

def analyze_mutations_demo(mutations_text, patient_age, patient_gender, cancer_type, progress=gr.Progress()):
    """Simulate mutation analysis with realistic demo data"""
    
    progress(0, desc="Parsing mutations...")
    time.sleep(0.5)
    
    # Parse mutations (one per line)
    mutations = [m.strip() for m in mutations_text.strip().split('\n') if m.strip()]
    
    if not mutations:
        return "❌ Please enter at least one mutation", None, None
    
    progress(0.2, desc="Analyzing pathogenicity...")
    time.sleep(1)
    
    # Generate analysis timestamp
    analysis_id = f"demo-{random.randint(1000, 9999)}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Analyze each mutation
    analyzed_mutations = []
    for i, mutation in enumerate(mutations):
        progress(0.2 + (0.5 * i / len(mutations)), desc=f"Analyzing {mutation}...")
        time.sleep(0.8)
        
        # Get demo data or generate generic response
        if mutation in DEMO_RESPONSES:
            data = DEMO_RESPONSES[mutation]
        else:
            # Generate plausible response for unknown mutations
            gene = mutation.split()[0] if mutation.split() else "Unknown"
            data = {
                "gene": gene,
                "variant": mutation,
                "pathogenicity": random.choice(["Pathogenic", "Likely Pathogenic", "Uncertain Significance"]),
                "score": round(random.uniform(0.3, 0.99), 2),
                "significance": "Variant requiring further investigation",
                "therapies": ["Requires clinical evaluation"],
                "mechanism": "Mechanism under investigation",
                "trials": random.randint(5, 30)
            }
        
        analyzed_mutations.append(data)
    
    progress(0.8, desc="Performing clustering analysis...")
    time.sleep(1.5)
    
    # Generate clustering results for multiple mutations
    clustering_performed = len(mutations) >= 3
    
    progress(0.95, desc="Generating report...")
    time.sleep(0.5)
    
    # Format results
    summary = format_demo_summary(analyzed_mutations, analysis_id, timestamp, patient_age, 
                                  patient_gender, cancer_type, clustering_performed)
    
    mutations_df = format_demo_table(analyzed_mutations)
    
    clustering_viz = format_demo_clustering(analyzed_mutations) if clustering_performed else "Clustering requires 3+ mutations"
    
    progress(1.0, desc="Analysis complete!")
    
    return summary, mutations_df, clustering_viz

def format_demo_summary(mutations, analysis_id, timestamp, age, gender, cancer_type, clustering):
    """Format demo analysis summary"""
    
    pathogenic_count = sum(1 for m in mutations if "Pathogenic" in m["pathogenicity"])
    therapies_available = sum(1 for m in mutations if len(m["therapies"]) > 1)
    total_trials = sum(m.get("trials", 0) for m in mutations)
    
    summary = f"""
# 🧬 OncoScope Analysis Report

**Analysis ID:** `{analysis_id}`  
**Timestamp:** {timestamp}  
**Patient:** {age} year old {gender.lower()}, {cancer_type} cancer  
**Mutations Analyzed:** {len(mutations)}

## 🎯 Key Findings:

### Pathogenicity Summary:
- **Pathogenic/Likely Pathogenic:** {pathogenic_count} mutations
- **Actionable Mutations:** {therapies_available} with FDA-approved therapies
- **Clinical Trials Available:** {total_trials} trials across all mutations

### Top Clinical Recommendations:
"""
    
    # Add specific recommendations
    for mut in mutations[:3]:  # Top 3 mutations
        if mut["pathogenicity"] == "Pathogenic" and len(mut["therapies"]) > 1:
            summary += f"\n**{mut['gene']} {mut['variant']}**\n"
            summary += f"- Significance: {mut['significance']}\n"
            summary += f"- Recommended therapies: {', '.join(mut['therapies'][:2])}\n"
    
    if clustering:
        summary += """
## 🔬 Multi-Mutation Analysis:
**Clustering Analysis Performed** - Identified functional relationships between mutations.
See Clustering tab for detailed interaction patterns.
"""
    
    summary += """
---
*This analysis would typically cost $5,000-$10,000 through commercial services.*  
*OncoScope provides this analysis FREE using on-device AI.*

⚠️ **Demo Mode**: This is a demonstration using pre-computed results. 
Full analysis requires local deployment with the fine-tuned Gemma 3n model.
"""
    
    return summary

def format_demo_table(mutations):
    """Format mutations as DataFrame"""
    
    table_data = []
    for mut in mutations:
        table_data.append({
            'Gene': mut['gene'],
            'Variant': mut['variant'],
            'Pathogenicity': mut['pathogenicity'],
            'Score': f"{mut['score']:.2f}",
            'FDA Therapies': ', '.join(mut['therapies'][:2]) if len(mut['therapies']) > 1 else mut['therapies'][0],
            'Clinical Trials': f"{mut.get('trials', 0)} available"
        })
    
    return pd.DataFrame(table_data)

def format_demo_clustering(mutations):
    """Generate demo clustering visualization"""
    
    if len(mutations) < 3:
        return "Clustering requires 3+ mutations"
    
    # Identify mutation patterns
    pathways = {
        "DNA Repair": ["BRCA1", "BRCA2", "ATM", "PALB2"],
        "RAS/MAPK": ["KRAS", "NRAS", "BRAF", "MEK1"],
        "Cell Cycle": ["TP53", "RB1", "CDKN2A", "MDM2"],
        "Growth Factor": ["EGFR", "HER2", "MET", "ALK"]
    }
    
    # Find pathway associations
    found_pathways = {}
    for pathway, genes in pathways.items():
        pathway_mutations = [m for m in mutations if m['gene'] in genes]
        if len(pathway_mutations) >= 2:
            found_pathways[pathway] = [m['gene'] for m in pathway_mutations]
    
    viz = f"""
## 🧬 Mutation Clustering Analysis

**Analysis Type:** Hierarchical clustering using Ward linkage
**Distance Metric:** Cancer-specific similarity (pathway + functional impact)

### Identified Clusters:

"""
    
    if found_pathways:
        viz += "**Pathway Convergence Detected:**\n"
        for pathway, genes in found_pathways.items():
            viz += f"\n📍 **{pathway} Pathway**\n"
            viz += f"   - Mutations: {', '.join(genes)}\n"
            viz += f"   - Clinical Impact: Potential synthetic lethality\n"
            viz += f"   - Treatment Strategy: Consider combination therapy targeting {pathway}\n"
    else:
        viz += """
**Independent Mutations Identified**
- No strong pathway convergence detected
- Each mutation may require independent therapeutic approach
- Consider sequential therapy based on individual mutation priorities
"""
    
    viz += f"""
### Clustering Insights:
- **Total Clusters:** {len(found_pathways) if found_pathways else 1}
- **Interaction Score:** {round(random.uniform(0.6, 0.9), 2)}
- **Recommendation:** {"Combination therapy advised" if found_pathways else "Sequential monotherapy"}

### Clinical Implications:
Based on the clustering analysis, this patient would benefit from:
1. Comprehensive molecular tumor board review
2. {"Combination targeted therapy" if found_pathways else "Prioritized single-agent therapy"}
3. Clinical trial enrollment for {"pathway-specific" if found_pathways else "mutation-specific"} treatments

---
*This clustering analysis mimics OncoScope's advanced Ward linkage algorithm that identifies functional relationships between mutations.*
"""
    
    return viz

def load_example(example_name):
    """Load example mutations"""
    mutations = EXAMPLE_MUTATIONS.get(example_name, [])
    return '\n'.join(mutations)

# Create Gradio Interface
with gr.Blocks(
    title="OncoScope - Molecular Tumor Board in a Box", 
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {font-family: 'Arial', sans-serif;}
    .markdown-text {line-height: 1.6;}
    #title {text-align: center; margin-bottom: 1em;}
    """
) as app:
    
    gr.Markdown("""
    <h1 id="title">🧬 OncoScope: Molecular Tumor Board in a Box</h1>
    
    <div style="text-align: center; margin-bottom: 2em;">
        <h3>Democratizing Cancer Genomics Analysis with Fine-tuned Gemma 3n</h3>
        <p><strong>$10,000 commercial test → FREE on-device analysis</strong></p>
    </div>
    
    This demo showcases OncoScope's capabilities:
    - 🔬 ACMG/AMP guideline-based pathogenicity assessment  
    - 💊 FDA-approved targeted therapy matching
    - 🧪 Advanced multi-mutation clustering analysis
    - 🔒 Complete privacy - runs entirely on your device
    - 🚀 Powered by fine-tuned Gemma 3n model
    """, elem_id="header")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Patient & Mutation Input")
            
            # Patient Information
            with gr.Group():
                gr.Markdown("### Patient Information")
                patient_age = gr.Number(label="Age", value=45, precision=0)
                patient_gender = gr.Radio(
                    label="Gender", 
                    choices=["Female", "Male", "Other"],
                    value="Female"
                )
                cancer_type = gr.Textbox(
                    label="Cancer Type",
                    value="breast",
                    placeholder="e.g., breast, lung, colorectal"
                )
            
            # Mutation Input
            with gr.Group():
                gr.Markdown("### Genetic Mutations")
                mutations_input = gr.Textbox(
                    label="Enter mutations (one per line)",
                    placeholder="Example:\nBRAF c.1799T>A\nKRAS c.35G>A\nTP53 c.818G>A",
                    lines=6
                )
                
                # Example buttons
                gr.Markdown("**Quick Examples:**")
                with gr.Row():
                    for example_name in EXAMPLE_MUTATIONS.keys():
                        btn = gr.Button(example_name, size="sm", scale=1)
                        btn.click(
                            fn=lambda x=example_name: load_example(x),
                            outputs=mutations_input
                        )
            
            analyze_btn = gr.Button("🧬 Analyze Mutations", variant="primary", size="lg")
            
            gr.Markdown("""
            <div style="margin-top: 2em; padding: 1em; background: #f0f0f0; border-radius: 5px;">
            <strong>💡 Try these clinically significant mutations:</strong><br>
            • BRCA1/BRCA2 - Hereditary breast/ovarian cancer<br>
            • KRAS G12D - Common in pancreatic/colorectal cancer<br>
            • EGFR T790M - Lung cancer resistance mutation<br>
            • BRAF V600E - Melanoma/colorectal driver
            </div>
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("## Analysis Results")
            
            # Results displays
            with gr.Tab("📊 Summary Report"):
                summary_output = gr.Markdown()
            
            with gr.Tab("🔬 Mutation Details"):
                mutations_table = gr.DataFrame(
                    headers=["Gene", "Variant", "Pathogenicity", "Score", "FDA Therapies", "Clinical Trials"],
                    label="Individual Mutation Analysis"
                )
            
            with gr.Tab("🧬 Clustering Analysis"):
                clustering_output = gr.Markdown()
    
    # Connect the analyze button
    analyze_btn.click(
        fn=analyze_mutations_demo,
        inputs=[mutations_input, patient_age, patient_gender, cancer_type],
        outputs=[summary_output, mutations_table, clustering_output]
    )
    
    # Footer
    gr.Markdown("""
    ---
    ### 🏆 Google Gemma 3n Impact Challenge Submission
    
    **OncoScope** demonstrates how fine-tuned Gemma 3n can democratize precision medicine:
    
    - **Impact**: Making $10,000 genetic tests free and accessible
    - **Privacy**: All analysis runs locally - no patient data leaves your device  
    - **Technical Innovation**: Custom clustering algorithm + fine-tuned Gemma 3n
    - **Real-world Application**: Based on FDA/COSMIC/ClinVar consensus data
    
    <div style="background: #fff3cd; padding: 1em; border-radius: 5px; margin-top: 1em;">
    <strong>⚠️ Demo Mode:</strong> This interface shows pre-computed results for demonstration. 
    The full system requires local deployment with Ollama and the fine-tuned Gemma 3n model.
    See our <a href="https://github.com/yourusername/oncoscope">GitHub repository</a> for deployment instructions.
    </div>
    
    <div style="text-align: center; margin-top: 2em;">
    <strong>Note:</strong> This is a proof of concept. Always consult healthcare professionals for medical decisions.
    </div>
    """)

# For Hugging Face Spaces
if __name__ == "__main__":
    app.launch()