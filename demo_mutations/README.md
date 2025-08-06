# OncoScope Demo: Patient Mutation Files

These demo files contain real clinical mutations for testing OncoScope's cancer genomics analysis. Each file represents mutations you'd typically receive from genomic testing labs.

## 📋 Demo Files

### 1. `single_mutation.txt`
- **Clinical Scenario**: Patient with Li-Fraumeni syndrome screening
- **Content**: TP53 hotspot mutation
- **Use Case**: Basic mutation analysis workflow

### 2. `common_mutations.txt`
- **Clinical Scenario**: Breast cancer patient with family history
- **Content**: BRCA1/2 mutations and associated variants
- **Use Case**: Hereditary cancer risk assessment

### 3. `comprehensive_panel.txt`
- **Clinical Scenario**: Advanced cancer patient requiring targeted therapy
- **Content**: 24 mutations across major cancer pathways
- **Use Case**: Complex multi-mutation analysis and drug targeting

### 4. `hereditary_mutations.txt`
- **Clinical Scenario**: High-risk patient from genetics clinic
- **Content**: 20 germline mutations across multiple cancer syndromes
- **Use Case**: Comprehensive hereditary risk evaluation

## 🩺 Clinical Workflow

### How Clinicians Use These Files:

1. **Receive Report**: Lab sends genomic analysis with identified mutations
2. **Extract Mutations**: Copy key variants in HGVS notation (already done in our demo files)
3. **Input to OncoScope**: Paste mutations into OncoScope for analysis
4. **Get Clinical Insights**: Receive pathogenicity, risk, and therapeutic recommendations

### Expected Analysis Output:
- **Risk Assessment**: Quantified cancer risk percentages
- **Pathogenicity Classification**: ACMG/AMP guidelines compliance
- **Therapeutic Options**: FDA-approved targeted treatments
- **Genetic Counseling**: Hereditary syndrome guidance
- **Family Screening**: Cascade testing recommendations

## 💼 Demo Scenarios

### Scenario 1: Tumor Board Preparation
**File**: `comprehensive_panel.txt`
**Setting**: Oncologist preparing for multidisciplinary review
**Goal**: Identify actionable mutations for treatment planning

### Scenario 2: Genetic Counseling Session
**File**: `hereditary_mutations.txt`
**Setting**: Genetic counselor with high-risk patient
**Goal**: Assess hereditary cancer syndromes and family risk

### Scenario 3: Treatment Selection
**File**: `common_mutations.txt`
**Setting**: Medical oncologist selecting targeted therapy
**Goal**: Match mutations to available treatments

## 🔬 Mutation Format

OncoScope accepts mutations in standard **HGVS notation**:

### Accepted Formats:
- `EGFR:p.L858R` (protein change)
- `TP53:c.818G>A` (DNA change)
- `BRCA1:c.68_69delAG` (deletion)
- `KRAS:p.Gly12Asp` (amino acid substitution)

### Clinical Examples:
```
# From a typical genomic report:
EGFR:p.L858R
TP53:p.R248Q
PIK3CA:p.E545K
BRAF:p.V600E
```

## 🎯 Testing Instructions

### Using OncoScope Desktop App:
1. Open OncoScope application
2. Click "Enter Patient Mutations" 
3. Copy/paste mutations from any demo file
4. Add patient demographics if prompted
5. Click "Analyze" for comprehensive report

### What You'll See:
- **Clinical Significance**: Pathogenic/Benign classification
- **Cancer Risk**: Lifetime risk percentages
- **Treatment Options**: Targeted therapy recommendations
- **Genetic Counseling**: Family screening guidance
- **Supporting Evidence**: Literature references and guidelines

## 📊 Key Mutations Included

### Tumor Suppressors:
- **TP53**: p.Arg273His (Li-Fraumeni syndrome)
- **BRCA1/2**: Frameshift mutations (HBOC syndrome)
- **PTEN**: Nonsense mutations (Cowden syndrome)

### Oncogenes:
- **EGFR**: p.L858R (lung cancer targetable)
- **BRAF**: p.Val600Glu (melanoma/colorectal targetable)
- **KRAS**: p.Gly12Asp (pancreatic/colorectal)

### DNA Repair Genes:
- **MSH2/MLH1**: Lynch syndrome mutations
- **PALB2/CHEK2**: Moderate penetrance breast cancer
- **RAD51D**: Ovarian cancer predisposition

## 🚀 Getting Started

1. **Choose Your Scenario**: Pick the clinical situation that matches your interest
2. **Open the File**: View the mutations in standard clinical notation
3. **Run Analysis**: Input into OncoScope and generate comprehensive report
4. **Review Results**: See how OncoScope provides actionable clinical insights

## 💡 Pro Tips

- **Real Workflow**: This mirrors how clinicians actually work with genomic data
- **Copy & Paste**: Just copy the mutations and paste into OncoScope
- **Multiple Patients**: Each file represents a different patient case
- **Clinical Context**: Note the scenario descriptions for realistic testing

---

**OncoScope transforms raw mutations into actionable clinical insights in seconds, not weeks.**