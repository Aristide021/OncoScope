"""
Clinical Report Generator
Professional-grade genomic analysis reports for clinical decision making
"""

import json
import io
from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import logging

from .models import MutationAnalysis, ClinicalSignificance, RiskLevel
from .risk_calculator import CancerRiskAssessment, PatientProfile, RiskFactors

logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    """Generate professional clinical reports for genomic analysis"""
    
    def __init__(self):
        """Initialize report generator"""
        self.report_templates = {
            'standard': self._generate_standard_report,
            'oncology': self._generate_oncology_report,
            'genetics': self._generate_genetics_report,
            'summary': self._generate_summary_report
        }
    
    def generate_clinical_report(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment,
        patient_profile: Optional[PatientProfile] = None,
        clustering_analysis: Optional[Dict] = None,
        report_type: str = 'standard',
        include_raw_data: bool = False
    ) -> Dict[str, Any]:
        """Generate comprehensive clinical report"""
        
        # Generate report based on type
        if report_type in self.report_templates:
            report_content = self.report_templates[report_type](
                mutations, risk_assessment, patient_profile, clustering_analysis
            )
        else:
            report_content = self._generate_standard_report(
                mutations, risk_assessment, patient_profile, clustering_analysis
            )
        
        # Create complete report structure
        report = {
            'report_metadata': self._generate_report_metadata(report_type),
            'executive_summary': self._generate_executive_summary(mutations, risk_assessment),
            'patient_information': self._format_patient_information(patient_profile),
            'content': report_content,
            'recommendations': self._generate_clinical_recommendations(mutations, risk_assessment),
            'disclaimers': self._generate_disclaimers(),
            'generated_at': datetime.now().isoformat(),
            'report_version': '1.0'
        }
        
        # Add raw data if requested
        if include_raw_data:
            report['raw_data'] = {
                'mutations': [asdict(m) for m in mutations],
                'risk_assessment': asdict(risk_assessment),
                'clustering_analysis': clustering_analysis
            }
        
        return report
    
    def _generate_report_metadata(self, report_type: str) -> Dict[str, Any]:
        """Generate report metadata"""
        return {
            'report_id': f"ONCO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'report_type': report_type,
            'system': 'OncoScope Genomic Analysis Platform',
            'version': '2.1.0',
            'generated_by': 'OncoScope AI Analysis Engine',
            'certification': 'Research Use Only - Not for Clinical Diagnosis',
            'laboratory': 'OncoScope Genomics Laboratory',
            'laboratory_director': 'Dr. OncoScope System',
            'contact': 'support@oncoscope.ai'
        }
    
    def _generate_executive_summary(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment
    ) -> Dict[str, Any]:
        """Generate executive summary"""
        
        # Key findings
        pathogenic_count = sum(
            1 for m in mutations 
            if m.clinical_significance in [ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC]
        )
        
        actionable_count = sum(1 for m in mutations if m.targeted_therapies)
        
        high_risk_genes = [
            m.gene for m in mutations 
            if m.pathogenicity_score > 0.7 and m.gene in ['BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2']
        ]
        
        return {
            'total_mutations_analyzed': len(mutations),
            'pathogenic_variants': pathogenic_count,
            'actionable_mutations': actionable_count,
            'overall_risk_level': risk_assessment.risk_level.value,
            'lifetime_risk_percentage': risk_assessment.lifetime_risk_percentage,
            'high_risk_genes_detected': high_risk_genes,
            'primary_cancer_risks': list(risk_assessment.cancer_type_risks.keys())[:3],
            'next_screening_recommended': risk_assessment.next_screening_date,
            'genetic_counseling_recommended': pathogenic_count > 0 or len(high_risk_genes) > 0,
            'key_recommendation': self._get_primary_recommendation(mutations, risk_assessment)
        }
    
    def _format_patient_information(self, patient_profile: Optional[PatientProfile]) -> Dict[str, Any]:
        """Format patient information for report"""
        if not patient_profile:
            return {
                'patient_id': 'PATIENT_001',
                'demographics': 'Not provided',
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'clinical_context': 'Genomic analysis requested'
            }
        
        return {
            'patient_id': 'PATIENT_001',  # Anonymized ID
            'age': patient_profile.age,
            'sex': patient_profile.sex,
            'ethnicity': patient_profile.ethnicity or 'Not specified',
            'family_history': 'Positive' if patient_profile.family_history else 'Negative',
            'analysis_date': datetime.now().strftime('%Y-%m-%d'),
            'clinical_indication': 'Cancer risk assessment and mutation analysis'
        }
    
    def _generate_standard_report(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment,
        patient_profile: Optional[PatientProfile],
        clustering_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate standard clinical report"""
        
        return {
            'mutation_analysis': self._format_mutation_analysis(mutations),
            'risk_assessment': self._format_risk_assessment(risk_assessment),
            'clustering_insights': self._format_clustering_analysis(clustering_analysis),
            'clinical_interpretation': self._generate_clinical_interpretation(mutations, risk_assessment),
            'therapeutic_implications': self._generate_therapeutic_implications(mutations),
            'screening_recommendations': self._generate_screening_recommendations(risk_assessment),
            'family_implications': self._generate_family_implications(mutations, patient_profile)
        }
    
    def _generate_oncology_report(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment,
        patient_profile: Optional[PatientProfile],
        clustering_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate oncology-focused report"""
        
        return {
            'tumor_risk_profile': self._generate_tumor_risk_profile(mutations, risk_assessment),
            'therapeutic_targets': self._identify_therapeutic_targets(mutations),
            'drug_resistance_markers': self._identify_resistance_markers(mutations),
            'prognostic_indicators': self._identify_prognostic_indicators(mutations),
            'clinical_trial_eligibility': self._assess_trial_eligibility(mutations),
            'molecular_tumor_board_summary': self._generate_mtb_summary(mutations, risk_assessment),
            'precision_medicine_recommendations': self._generate_precision_medicine_recs(mutations)
        }
    
    def _generate_genetics_report(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment,
        patient_profile: Optional[PatientProfile],
        clustering_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate genetics-focused report"""
        
        return {
            'inheritance_patterns': self._analyze_inheritance_patterns(mutations),
            'penetrance_analysis': self._analyze_penetrance(mutations),
            'variant_classification': self._classify_variants(mutations),
            'population_frequencies': self._analyze_population_frequencies(mutations),
            'functional_impact_prediction': self._predict_functional_impact(mutations),
            'cascade_testing_recommendations': self._recommend_cascade_testing(mutations, patient_profile),
            'reproductive_counseling': self._generate_reproductive_counseling(mutations, patient_profile)
        }
    
    def _generate_summary_report(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment,
        patient_profile: Optional[PatientProfile],
        clustering_analysis: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate concise summary report"""
        
        key_findings = []
        
        # High-impact mutations
        for mutation in mutations:
            if mutation.clinical_significance == ClinicalSignificance.PATHOGENIC:
                key_findings.append(f"Pathogenic variant in {mutation.gene}: {mutation.variant}")
        
        # Risk level
        risk_level = risk_assessment.risk_level.value
        lifetime_risk = risk_assessment.lifetime_risk_percentage
        
        return {
            'key_findings': key_findings,
            'risk_summary': f"{risk_level} cancer risk ({lifetime_risk:.1f}% lifetime risk)",
            'primary_recommendations': risk_assessment.recommendations[:3],
            'follow_up_required': len(key_findings) > 0,
            'genetic_counseling_indicated': any(
                m.gene in ['BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2'] 
                for m in mutations 
                if m.clinical_significance == ClinicalSignificance.PATHOGENIC
            )
        }
    
    def _format_mutation_analysis(self, mutations: List[MutationAnalysis]) -> List[Dict[str, Any]]:
        """Format mutation analysis for report"""
        
        formatted_mutations = []
        
        for mutation in mutations:
            formatted_mutation = {
                'gene': mutation.gene,
                'variant': mutation.variant,
                'protein_change': mutation.protein_change or 'Unknown',
                'clinical_significance': mutation.clinical_significance.value,
                'pathogenicity_score': f"{mutation.pathogenicity_score:.3f}",
                'cancer_associations': mutation.cancer_types,
                'targeted_therapies': mutation.targeted_therapies,
                'prognosis_impact': mutation.prognosis_impact.value,
                'mechanism': mutation.mechanism,
                'confidence_level': f"{mutation.confidence_score:.3f}",
                'evidence_sources': mutation.references,
                'clinical_interpretation': self._interpret_mutation(mutation)
            }
            
            formatted_mutations.append(formatted_mutation)
        
        return formatted_mutations
    
    def _format_risk_assessment(self, risk_assessment: CancerRiskAssessment) -> Dict[str, Any]:
        """Format risk assessment for report"""
        
        return {
            'overall_risk': {
                'score': f"{risk_assessment.overall_risk_score:.3f}",
                'level': risk_assessment.risk_level.value,
                'lifetime_risk_percentage': f"{risk_assessment.lifetime_risk_percentage:.1f}%",
                'five_year_risk_percentage': f"{risk_assessment.five_year_risk_percentage:.1f}%",
                'confidence_interval': f"{risk_assessment.confidence_interval[0]:.2f} - {risk_assessment.confidence_interval[1]:.2f}"
            },
            'risk_factors': {
                'genetic_mutations': f"{risk_assessment.risk_factors.mutation_risk:.3f}",
                'age_related': f"{risk_assessment.risk_factors.age_risk:.3f}",
                'family_history': f"{risk_assessment.risk_factors.family_history_risk:.3f}",
                'lifestyle_factors': f"{risk_assessment.risk_factors.lifestyle_risk:.3f}",
                'environmental_factors': f"{risk_assessment.risk_factors.environmental_risk:.3f}",
                'protective_factors': f"{risk_assessment.risk_factors.protective_factors:.3f}"
            },
            'cancer_type_risks': {
                cancer_type: f"{risk:.3f}" 
                for cancer_type, risk in risk_assessment.cancer_type_risks.items()
            },
            'risk_explanation': risk_assessment.risk_explanation
        }
    
    def _format_clustering_analysis(self, clustering_analysis: Optional[Dict]) -> Dict[str, Any]:
        """Format clustering analysis for report"""
        
        if not clustering_analysis:
            return {'clustering_performed': False, 'reason': 'Insufficient mutations for clustering analysis'}
        
        return {
            'clustering_performed': True,
            'clusters_identified': clustering_analysis.get('clusters_identified', 0),
            'high_risk_clusters': len(clustering_analysis.get('cluster_analysis', {}).get('high_risk_clusters', [])),
            'pathway_insights': clustering_analysis.get('cluster_analysis', {}).get('pathway_insights', {}),
            'therapeutic_opportunities': len(clustering_analysis.get('cluster_analysis', {}).get('therapeutic_opportunities', [])),
            'clinical_insights': clustering_analysis.get('clustering_insights', [])
        }
    
    def _generate_clinical_interpretation(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment
    ) -> str:
        """Generate overall clinical interpretation"""
        
        interpretation_parts = []
        
        # Mutation significance
        pathogenic_mutations = [
            m for m in mutations 
            if m.clinical_significance in [ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC]
        ]
        
        if pathogenic_mutations:
            genes = [m.gene for m in pathogenic_mutations]
            interpretation_parts.append(
                f"Analysis identified {len(pathogenic_mutations)} pathogenic/likely pathogenic variant(s) "
                f"in the following gene(s): {', '.join(genes)}."
            )
        
        # Risk level interpretation
        risk_level = risk_assessment.risk_level
        lifetime_risk = risk_assessment.lifetime_risk_percentage
        
        if risk_level == RiskLevel.HIGH:
            interpretation_parts.append(
                f"The patient has a HIGH cancer risk ({lifetime_risk:.1f}% lifetime risk), "
                "requiring immediate clinical attention and enhanced surveillance protocols."
            )
        elif risk_level == RiskLevel.MEDIUM_HIGH:
            interpretation_parts.append(
                f"The patient has a MODERATE-HIGH cancer risk ({lifetime_risk:.1f}% lifetime risk), "
                "warranting enhanced screening and genetic counseling."
            )
        else:
            interpretation_parts.append(
                f"The patient has a {risk_level.value} cancer risk ({lifetime_risk:.1f}% lifetime risk)."
            )
        
        # Actionable mutations
        actionable_mutations = [m for m in mutations if m.targeted_therapies]
        if actionable_mutations:
            therapy_count = sum(len(m.targeted_therapies) for m in actionable_mutations)
            interpretation_parts.append(
                f"Found {len(actionable_mutations)} actionable mutation(s) with {therapy_count} "
                "potential targeted therapy options."
            )
        
        return " ".join(interpretation_parts)
    
    def _generate_therapeutic_implications(self, mutations: List[MutationAnalysis]) -> Dict[str, Any]:
        """Generate therapeutic implications"""
        
        therapeutic_targets = []
        resistance_markers = []
        clinical_trials = []
        
        for mutation in mutations:
            if mutation.targeted_therapies:
                for therapy in mutation.targeted_therapies:
                    therapeutic_targets.append({
                        'gene': mutation.gene,
                        'variant': mutation.variant,
                        'therapy': therapy,
                        'evidence_level': 'FDA Approved' if therapy in ['osimertinib', 'sotorasib', 'olaparib'] else 'Clinical Evidence'
                    })
            
            # Check for resistance markers
            if 'resistance' in mutation.mechanism.lower():
                resistance_markers.append({
                    'gene': mutation.gene,
                    'variant': mutation.variant,
                    'resistance_to': 'Multiple therapies',  # Would be more specific in production
                    'mechanism': mutation.mechanism
                })
        
        return {
            'therapeutic_targets': therapeutic_targets,
            'resistance_markers': resistance_markers,
            'clinical_trial_opportunities': clinical_trials,
            'precision_medicine_score': len(therapeutic_targets) / max(len(mutations), 1)
        }
    
    def _generate_screening_recommendations(self, risk_assessment: CancerRiskAssessment) -> Dict[str, Any]:
        """Generate personalized screening recommendations"""
        
        screening_plan = {}
        
        # Risk-based screening intervals
        if risk_assessment.risk_level in [RiskLevel.HIGH, RiskLevel.MEDIUM_HIGH]:
            screening_plan['frequency'] = 'Enhanced (every 6-12 months)'
            screening_plan['modalities'] = ['MRI', 'Mammography', 'Ultrasound']
            screening_plan['start_age'] = 'Immediate'
        else:
            screening_plan['frequency'] = 'Standard (annual)'
            screening_plan['modalities'] = ['Standard screening protocols']
            screening_plan['start_age'] = 'Per guidelines'
        
        # Cancer-type specific recommendations
        cancer_specific = {}
        for cancer_type, risk_score in risk_assessment.cancer_type_risks.items():
            if risk_score > 0.3:
                cancer_specific[cancer_type] = {
                    'recommendation': 'Enhanced screening',
                    'frequency': 'Annual' if risk_score > 0.5 else 'Every 2 years',
                    'additional_tests': self._get_cancer_specific_tests(cancer_type)
                }
        
        return {
            'general_screening': screening_plan,
            'cancer_specific_screening': cancer_specific,
            'next_screening_date': risk_assessment.next_screening_date,
            'surveillance_program': 'High-risk surveillance' if risk_assessment.risk_level == RiskLevel.HIGH else 'Standard surveillance'
        }
    
    def _generate_family_implications(
        self,
        mutations: List[MutationAnalysis],
        patient_profile: Optional[PatientProfile]
    ) -> Dict[str, Any]:
        """Generate family implications and recommendations"""
        
        hereditary_mutations = [
            m for m in mutations 
            if m.gene in ['BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'APC', 'PALB2']
            and m.clinical_significance == ClinicalSignificance.PATHOGENIC
        ]
        
        if not hereditary_mutations:
            return {
                'cascade_testing_recommended': False,
                'inheritance_pattern': 'No hereditary cancer syndromes identified',
                'family_risk': 'Population risk',
                'genetic_counseling': 'Optional'
            }
        
        family_implications = {
            'cascade_testing_recommended': True,
            'family_members_at_risk': ['First-degree relatives', 'Second-degree relatives if indicated'],
            'inheritance_pattern': 'Autosomal dominant',
            'transmission_risk': '50% for each offspring',
            'genetic_counseling': 'Strongly recommended',
            'hereditary_syndromes_identified': []
        }
        
        # Identify specific syndromes
        for mutation in hereditary_mutations:
            if mutation.gene in ['BRCA1', 'BRCA2']:
                family_implications['hereditary_syndromes_identified'].append('Hereditary Breast and Ovarian Cancer (HBOC)')
            elif mutation.gene in ['MLH1', 'MSH2', 'MSH6', 'PMS2']:
                family_implications['hereditary_syndromes_identified'].append('Lynch Syndrome')
            elif mutation.gene == 'TP53':
                family_implications['hereditary_syndromes_identified'].append('Li-Fraumeni Syndrome')
        
        return family_implications
    
    def _generate_clinical_recommendations(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment
    ) -> List[Dict[str, Any]]:
        """Generate structured clinical recommendations"""
        
        recommendations = []
        
        # Priority recommendations based on findings
        pathogenic_count = sum(
            1 for m in mutations 
            if m.clinical_significance == ClinicalSignificance.PATHOGENIC
        )
        
        if pathogenic_count > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Genetic Counseling',
                'recommendation': 'Immediate genetic counseling consultation',
                'rationale': f'Pathogenic variant(s) identified in {pathogenic_count} gene(s)',
                'timeline': 'Within 2 weeks'
            })
        
        if risk_assessment.risk_level == RiskLevel.HIGH:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk Management',
                'recommendation': 'Enhanced cancer surveillance protocol',
                'rationale': 'High cancer risk identified',
                'timeline': 'Immediate'
            })
        
        # Therapeutic recommendations
        actionable_mutations = [m for m in mutations if m.targeted_therapies]
        if actionable_mutations:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Targeted Therapy',
                'recommendation': 'Molecular tumor board review for targeted therapy options',
                'rationale': f'{len(actionable_mutations)} actionable mutation(s) identified',
                'timeline': 'Within 4 weeks'
            })
        
        # Screening recommendations
        recommendations.extend(risk_assessment.recommendations[:3])
        
        return recommendations
    
    def _generate_disclaimers(self) -> List[str]:
        """Generate report disclaimers"""
        
        return [
            "This report is for research purposes only and should not be used for clinical diagnosis.",
            "Results should be interpreted by qualified healthcare professionals.",
            "Genetic counseling is recommended for interpretation of results.",
            "Technology limitations may affect variant detection and classification.",
            "Risk estimates are based on current scientific knowledge and may change.",
            "Family history and environmental factors should be considered in clinical decisions.",
            "This analysis does not detect all possible genetic variants.",
            "Consult with appropriate specialists for clinical management decisions."
        ]
    
    # Helper methods
    def _get_primary_recommendation(
        self,
        mutations: List[MutationAnalysis],
        risk_assessment: CancerRiskAssessment
    ) -> str:
        """Get primary recommendation"""
        
        if risk_assessment.risk_level == RiskLevel.HIGH:
            return "Immediate oncology consultation and enhanced surveillance"
        elif any(m.clinical_significance == ClinicalSignificance.PATHOGENIC for m in mutations):
            return "Genetic counseling and cascade family testing"
        elif any(m.targeted_therapies for m in mutations):
            return "Molecular tumor board review for targeted therapy"
        else:
            return "Continue standard screening protocols"
    
    def _interpret_mutation(self, mutation: MutationAnalysis) -> str:
        """Interpret individual mutation"""
        
        significance = mutation.clinical_significance.value
        pathogenicity = mutation.pathogenicity_score
        
        if significance == "PATHOGENIC":
            return f"Pathogenic variant with high disease risk (score: {pathogenicity:.2f})"
        elif significance == "LIKELY_PATHOGENIC":
            return f"Likely pathogenic variant with moderate disease risk (score: {pathogenicity:.2f})"
        elif significance == "VARIANT_OF_UNCERTAIN_SIGNIFICANCE":
            return f"Uncertain significance - insufficient evidence for classification (score: {pathogenicity:.2f})"
        else:
            return f"Benign or likely benign variant with low disease risk (score: {pathogenicity:.2f})"
    
    def _get_cancer_specific_tests(self, cancer_type: str) -> List[str]:
        """Get cancer-specific screening tests"""
        
        tests = {
            'breast': ['Mammography', 'MRI', 'Clinical breast exam'],
            'ovarian': ['Transvaginal ultrasound', 'CA-125', 'Clinical exam'],
            'colorectal': ['Colonoscopy', 'FIT test', 'Genetic testing'],
            'prostate': ['PSA', 'Digital rectal exam', 'MRI if indicated'],
            'lung': ['Low-dose CT', 'Smoking cessation counseling']
        }
        
        return tests.get(cancer_type, ['Standard screening protocols'])
    
    # Additional report generation methods for oncology and genetics reports would go here
    def _generate_tumor_risk_profile(self, mutations, risk_assessment):
        return {'tumor_types': 'Multiple', 'risk_drivers': 'Genetic'}
    
    def _identify_therapeutic_targets(self, mutations):
        return [m.gene for m in mutations if m.targeted_therapies]
    
    def _identify_resistance_markers(self, mutations):
        return []
    
    def _identify_prognostic_indicators(self, mutations):
        return []
    
    def _assess_trial_eligibility(self, mutations):
        return []
    
    def _generate_mtb_summary(self, mutations, risk_assessment):
        return {'recommendation': 'MTB review recommended'}
    
    def _generate_precision_medicine_recs(self, mutations):
        return []
    
    def _analyze_inheritance_patterns(self, mutations):
        return {'pattern': 'Autosomal dominant'}
    
    def _analyze_penetrance(self, mutations):
        return {'penetrance': 'High'}
    
    def _classify_variants(self, mutations):
        return [asdict(m) for m in mutations]
    
    def _analyze_population_frequencies(self, mutations):
        return {}
    
    def _predict_functional_impact(self, mutations):
        return {}
    
    def _recommend_cascade_testing(self, mutations, patient_profile):
        return {'recommended': True}
    
    def _generate_reproductive_counseling(self, mutations, patient_profile):
        return {'counseling_needed': False}

def export_report_to_pdf(report_data: Dict[str, Any]) -> bytes:
    """Export report to PDF format"""
    # This would use a library like reportlab or weasyprint
    # For now, return a placeholder
    pdf_content = f"""
ONCOSCOPE GENOMIC ANALYSIS REPORT
Generated: {report_data['generated_at']}
Report ID: {report_data['report_metadata']['report_id']}

EXECUTIVE SUMMARY:
{json.dumps(report_data['executive_summary'], indent=2)}

[Full PDF implementation would go here]
"""
    return pdf_content.encode('utf-8')

def export_report_to_csv(report_data: Dict[str, Any]) -> str:
    """Export key findings to CSV format"""
    
    csv_content = "Gene,Variant,Clinical_Significance,Pathogenicity_Score,Cancer_Types,Targeted_Therapies\n"
    
    if 'raw_data' in report_data and 'mutations' in report_data['raw_data']:
        for mutation in report_data['raw_data']['mutations']:
            csv_content += f"{mutation['gene']},{mutation['variant']},{mutation['clinical_significance']},{mutation['pathogenicity_score']},\"{';'.join(mutation['cancer_types'])}\",\"{';'.join(mutation['targeted_therapies'])}\"\n"
    
    return csv_content