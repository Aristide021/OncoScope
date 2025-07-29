"""
Core Cancer Mutation Analysis Engine
"""
import json
import re
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .models import (
    MutationAnalysis, ClinicalSignificance, Prognosis, RiskLevel,
    TherapyRecommendation, CancerTypeAssociation
)
from .config import settings
from .clustering_engine import CancerMutationClusterEngine

logger = logging.getLogger(__name__)

class CancerMutationAnalyzer:
    """Core cancer mutation analysis engine"""
    
    def __init__(self):
        self.mutation_db = {}
        self.drug_db = {}
        self.cancer_types_db = {}
        self.load_databases()
        
        # Import AI client lazily to avoid circular imports
        self._ollama_client = None
        
        # Initialize clustering engine
        self.clustering_engine = CancerMutationClusterEngine()
    
    @property
    def ollama_client(self):
        """Lazy load Ollama client"""
        if self._ollama_client is None:
            import sys
            from pathlib import Path
            # Add parent directory to path for imports
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from ai.inference.ollama_client import OllamaClient
            self._ollama_client = OllamaClient()
        return self._ollama_client
    
    def load_databases(self):
        """Load mutation, drug, and cancer type databases"""
        try:
            # Load COSMIC mutations
            cosmic_path = settings.data_dir / "cosmic_mutations.json"
            if cosmic_path.exists():
                with open(cosmic_path, 'r') as f:
                    self.mutation_db = json.load(f)
            else:
                logger.warning("COSMIC mutations database not found, using defaults")
                self._load_default_mutations()
            
            # Load drug associations
            drug_path = settings.data_dir / "drug_associations.json"
            if drug_path.exists():
                with open(drug_path, 'r') as f:
                    self.drug_db = json.load(f)
            else:
                self._load_default_drugs()
            
            # Load cancer types
            cancer_path = settings.data_dir / "cancer_types.json"
            if cancer_path.exists():
                with open(cancer_path, 'r') as f:
                    self.cancer_types_db = json.load(f)
            else:
                self._load_default_cancer_types()
                
        except Exception as e:
            logger.error(f"Failed to load databases: {e}")
            self._load_default_mutations()
            self._load_default_drugs()
            self._load_default_cancer_types()
    
    def _load_default_mutations(self):
        """Load default mutation database"""
        self.mutation_db = {
            "TP53": {
                "c.524G>A": {
                    "protein": "p.R175H",
                    "pathogenicity": 0.95,
                    "cancer_types": ["breast", "lung", "colorectal", "ovarian"],
                    "frequency": "11% of all cancers",
                    "mechanism": "DNA binding domain hotspot disrupts p53 tumor suppressor function",
                    "prognosis": "poor",
                    "targeted_therapy": ["APR-246", "PRIMA-1"]
                },
                "c.733G>A": {
                    "protein": "p.G245S",
                    "pathogenicity": 0.92,
                    "cancer_types": ["lung", "colorectal"],
                    "frequency": "3% of all cancers",
                    "mechanism": "Disrupts DNA binding leading to loss of tumor suppression",
                    "prognosis": "poor",
                    "targeted_therapy": []
                }
            },
            "KRAS": {
                "c.35G>A": {
                    "protein": "p.G12D",
                    "pathogenicity": 0.92,
                    "cancer_types": ["pancreatic", "colorectal", "lung"],
                    "frequency": "25% of colorectal cancers",
                    "mechanism": "Constitutive GTPase activation",
                    "prognosis": "poor_without_targeted_therapy",
                    "targeted_therapy": ["sotorasib", "adagrasib"]
                },
                "c.34G>T": {
                    "protein": "p.G12C",
                    "pathogenicity": 0.91,
                    "cancer_types": ["lung", "colorectal"],
                    "frequency": "13% of lung adenocarcinomas",
                    "mechanism": "Impaired GTPase activity",
                    "prognosis": "moderate",
                    "targeted_therapy": ["sotorasib", "adagrasib"]
                }
            },
            "EGFR": {
                "c.2573T>G": {
                    "protein": "p.L858R",
                    "pathogenicity": 0.90,
                    "cancer_types": ["lung_adenocarcinoma"],
                    "frequency": "15% of lung adenocarcinomas",
                    "mechanism": "Tyrosine kinase constitutive activation",
                    "prognosis": "excellent_with_therapy",
                    "targeted_therapy": ["erlotinib", "gefitinib", "osimertinib", "afatinib"]
                },
                "c.2369C>T": {
                    "protein": "p.T790M",
                    "pathogenicity": 0.88,
                    "cancer_types": ["lung_adenocarcinoma"],
                    "frequency": "50% of EGFR TKI resistance",
                    "mechanism": "Resistance mutation to first-gen TKIs",
                    "prognosis": "moderate",
                    "targeted_therapy": ["osimertinib"]
                }
            },
            "BRCA1": {
                "c.68_69delAG": {
                    "protein": "p.E23fs",
                    "pathogenicity": 0.95,
                    "cancer_types": ["breast", "ovarian"],
                    "frequency": "1% of Ashkenazi Jewish population",
                    "mechanism": "Frameshift leading to loss of DNA repair function",
                    "prognosis": "moderate",
                    "targeted_therapy": ["olaparib", "talazoparib", "rucaparib"]
                }
            },
            "PIK3CA": {
                "c.3140A>G": {
                    "protein": "p.H1047R",
                    "pathogenicity": 0.85,
                    "cancer_types": ["breast", "colorectal", "endometrial"],
                    "frequency": "15% of breast cancers",
                    "mechanism": "Constitutive PI3K activation",
                    "prognosis": "moderate",
                    "targeted_therapy": ["alpelisib"]
                }
            }
        }
    
    def _load_default_drugs(self):
        """Load default drug associations"""
        self.drug_db = {
            "erlotinib": {
                "targets": ["EGFR"],
                "class": "Tyrosine Kinase Inhibitor",
                "generation": "1st",
                "fda_approved": True,
                "indications": ["Non-small cell lung cancer"]
            },
            "osimertinib": {
                "targets": ["EGFR", "EGFR T790M"],
                "class": "Tyrosine Kinase Inhibitor", 
                "generation": "3rd",
                "fda_approved": True,
                "indications": ["EGFR-mutated NSCLC", "T790M resistance"]
            },
            "sotorasib": {
                "targets": ["KRAS G12C"],
                "class": "KRAS Inhibitor",
                "generation": "1st",
                "fda_approved": True,
                "indications": ["KRAS G12C mutated NSCLC"]
            },
            "olaparib": {
                "targets": ["BRCA1", "BRCA2"],
                "class": "PARP Inhibitor",
                "generation": "1st",
                "fda_approved": True,
                "indications": ["BRCA-mutated breast/ovarian cancer"]
            }
        }
    
    def _load_default_cancer_types(self):
        """Load default cancer type associations"""
        self.cancer_types_db = {
            "lung_adenocarcinoma": {
                "common_mutations": ["EGFR", "KRAS", "ALK", "ROS1"],
                "prevalence": "40% of lung cancers",
                "typical_age": "60-70",
                "risk_factors": ["smoking", "radon", "asbestos"]
            },
            "breast": {
                "common_mutations": ["BRCA1", "BRCA2", "PIK3CA", "TP53"],
                "prevalence": "12% lifetime risk",
                "typical_age": "50-70",
                "risk_factors": ["family history", "hormones", "age"]
            }
        }
    
    def parse_mutation_notation(self, mutation: str) -> Tuple[str, str]:
        """Parse mutation notation (e.g., 'TP53:c.524G>A')"""
        if ':' in mutation:
            gene, variant = mutation.split(':', 1)
            return gene.strip().upper(), variant.strip()
        else:
            # Try to infer from common patterns
            match = re.match(r'([A-Z0-9]+)[\s\-_]*(.+)', mutation, re.IGNORECASE)
            if match:
                return match.group(1).upper(), match.group(2)
            return "UNKNOWN", mutation
    
    async def analyze_single_mutation(self, mutation: str) -> MutationAnalysis:
        """Analyze individual mutation for clinical significance"""
        gene, variant = self.parse_mutation_notation(mutation)
        
        # Look up in database
        if gene in self.mutation_db and variant in self.mutation_db[gene]:
            data = self.mutation_db[gene][variant]
            
            return MutationAnalysis(
                mutation_id=f"{gene}:{variant}",
                gene=gene,
                variant=variant,
                protein_change=data.get('protein', ''),
                pathogenicity_score=data.get('pathogenicity', 0.5),
                cancer_types=data.get('cancer_types', []),
                clinical_significance=self.classify_significance(data.get('pathogenicity', 0.5)),
                targeted_therapies=data.get('targeted_therapy', []),
                prognosis_impact=Prognosis(data.get('prognosis', 'uncertain')),
                mechanism=data.get('mechanism', 'Unknown mechanism'),
                confidence_score=0.95,  # High confidence for known mutations
                references=["COSMIC", "ClinVar", "OncoKB"]
            )
        else:
            # Unknown mutation - use AI analysis
            return await self.ai_analyze_mutation(gene, variant)
    
    async def ai_analyze_mutation(self, gene: str, variant: str) -> MutationAnalysis:
        """Use AI model for unknown mutations"""
        try:
            ai_analysis = await self.ollama_client.analyze_cancer_mutation(gene, variant)
            
            return MutationAnalysis(
                mutation_id=f"{gene}:{variant}",
                gene=gene,
                variant=variant,
                protein_change=ai_analysis.get('protein_change', ''),
                pathogenicity_score=ai_analysis.get('pathogenicity', 0.5),
                cancer_types=ai_analysis.get('cancer_types', []),
                clinical_significance=ClinicalSignificance(
                    ai_analysis.get('significance', 'UNCERTAIN')
                ),
                targeted_therapies=ai_analysis.get('therapies', []),
                prognosis_impact=Prognosis(ai_analysis.get('prognosis', 'uncertain')),
                mechanism=ai_analysis.get('mechanism', 'AI-predicted mechanism'),
                confidence_score=ai_analysis.get('confidence', 0.7),
                references=["AI Analysis"]
            )
        except Exception as e:
            logger.error(f"AI analysis failed for {gene}:{variant}: {e}")
            return self.create_uncertain_analysis(gene, variant)
    
    def create_uncertain_analysis(self, gene: str, variant: str) -> MutationAnalysis:
        """Create uncertain analysis when both DB and AI fail"""
        return MutationAnalysis(
            mutation_id=f"{gene}:{variant}",
            gene=gene,
            variant=variant,
            protein_change="Unknown",
            pathogenicity_score=0.5,
            cancer_types=["unknown"],
            clinical_significance=ClinicalSignificance.UNCERTAIN,
            targeted_therapies=[],
            prognosis_impact=Prognosis.UNCERTAIN,
            mechanism="Insufficient data for analysis",
            confidence_score=0.3,
            references=[]
        )
    
    def classify_significance(self, pathogenicity: float) -> ClinicalSignificance:
        """Classify clinical significance based on pathogenicity score"""
        if pathogenicity >= 0.9:
            return ClinicalSignificance.PATHOGENIC
        elif pathogenicity >= 0.7:
            return ClinicalSignificance.LIKELY_PATHOGENIC
        elif pathogenicity <= 0.3:
            return ClinicalSignificance.BENIGN
        elif pathogenicity <= 0.5:
            return ClinicalSignificance.LIKELY_BENIGN
        else:
            return ClinicalSignificance.UNCERTAIN
    
    async def analyze_mutation_list(
        self,
        mutations: List[str],
        include_drug_interactions: bool = True,
        patient_context: Optional[Dict[str, Any]] = None,
        include_clustering: bool = True
    ) -> Dict[str, Any]:
        """Analyze list of mutations and calculate overall risk with clustering"""
        # Analyze individual mutations
        tasks = [self.analyze_single_mutation(mutation) for mutation in mutations]
        analyses = await asyncio.gather(*tasks)
        
        # Convert to dict format
        analyses_dict = [asdict(analysis) for analysis in analyses]
        
        # Calculate composite risk score
        overall_risk = self.calculate_composite_risk(analyses)
        
        # Generate clinical recommendations
        recommendations = self.generate_recommendations(
            analyses, overall_risk, patient_context
        )
        
        # Identify actionable mutations
        actionable = self.identify_actionable_mutations(analyses)
        
        # Predict tumor types
        tumor_predictions = self.predict_tumor_types(analyses)
        
        # Calculate confidence metrics
        confidence_metrics = self.calculate_confidence_metrics(analyses)
        
        # Perform clustering analysis if requested and sufficient mutations
        clustering_results = {}
        if include_clustering and len(analyses) >= 3:
            try:
                clusters, cluster_analysis = self.clustering_engine.cluster_mutations(analyses)
                clustering_results = {
                    "clusters_identified": len(clusters),
                    "cluster_analysis": cluster_analysis,
                    "clustering_insights": self.generate_clustering_insights(cluster_analysis)
                }
                
                # Enhance recommendations with clustering insights
                recommendations.extend(self.generate_clustering_recommendations(cluster_analysis))
                
            except Exception as e:
                logger.error(f"Clustering analysis failed: {e}")
                clustering_results = {
                    "clusters_identified": 0,
                    "error": "Clustering analysis unavailable"
                }
        
        result = {
            "individual_mutations": analyses_dict,
            "overall_risk_score": overall_risk,
            "risk_classification": self.classify_risk_level(overall_risk),
            "clinical_recommendations": recommendations,
            "actionable_mutations": actionable,
            "estimated_tumor_types": tumor_predictions,
            "confidence_metrics": confidence_metrics,
            "warnings": self.generate_warnings(analyses)
        }
        
        # Add clustering results if available
        if clustering_results:
            result["clustering_analysis"] = clustering_results
        
        return result
    
    def calculate_composite_risk(self, analyses: List[MutationAnalysis]) -> float:
        """Calculate overall risk score from multiple mutations"""
        if not analyses:
            return 0.0
        
        # Weighted average based on pathogenicity and clinical significance
        weights = {
            ClinicalSignificance.PATHOGENIC: 1.0,
            ClinicalSignificance.LIKELY_PATHOGENIC: 0.8,
            ClinicalSignificance.UNCERTAIN: 0.5,
            ClinicalSignificance.LIKELY_BENIGN: 0.2,
            ClinicalSignificance.BENIGN: 0.0
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for analysis in analyses:
            weight = weights.get(analysis.clinical_significance, 0.5)
            score = analysis.pathogenicity_score * weight
            
            # Boost score if targeted therapy available
            if analysis.targeted_therapies:
                score *= 0.9  # Slightly lower risk if treatable
            
            total_score += score
            total_weight += weight
        
        if total_weight > 0:
            # Normalize and apply sigmoid for better distribution
            raw_score = total_score / total_weight
            # Apply mild sigmoid to spread scores
            import math
            return 1 / (1 + math.exp(-4 * (raw_score - 0.5)))
        
        return 0.5
    
    def classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level based on score"""
        if risk_score >= 0.8:
            return RiskLevel.HIGH
        elif risk_score >= 0.65:
            return RiskLevel.MEDIUM_HIGH
        elif risk_score >= 0.5:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.35:
            return RiskLevel.LOW_MEDIUM
        else:
            return RiskLevel.LOW
    
    def generate_recommendations(
        self,
        analyses: List[MutationAnalysis],
        overall_risk: float,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate clinical recommendations based on analysis"""
        recommendations = []
        
        # High-risk recommendations
        if overall_risk >= 0.7:
            recommendations.append(
                "Immediate oncology consultation recommended due to high-risk mutation profile"
            )
        
        # Check for actionable mutations
        actionable_genes = set()
        for analysis in analyses:
            if analysis.targeted_therapies:
                actionable_genes.add(analysis.gene)
                therapies = ", ".join(analysis.targeted_therapies[:3])
                recommendations.append(
                    f"Consider targeted therapy for {analysis.gene} mutation: {therapies}"
                )
        
        # Check for hereditary cancer syndromes
        hereditary_genes = {"BRCA1", "BRCA2", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM"}
        found_hereditary = [a.gene for a in analyses if a.gene in hereditary_genes]
        if found_hereditary:
            recommendations.append(
                f"Genetic counseling recommended due to hereditary cancer gene mutations: "
                f"{', '.join(found_hereditary)}"
            )
            recommendations.append(
                "Consider cascade testing for family members"
            )
        
        # General recommendations based on significance
        pathogenic_count = sum(
            1 for a in analyses 
            if a.clinical_significance in [
                ClinicalSignificance.PATHOGENIC,
                ClinicalSignificance.LIKELY_PATHOGENIC
            ]
        )
        
        if pathogenic_count > 0:
            recommendations.append(
                f"Enhanced surveillance recommended due to {pathogenic_count} "
                f"pathogenic mutation{'s' if pathogenic_count > 1 else ''}"
            )
        
        # If no specific recommendations, provide general guidance
        if not recommendations:
            if overall_risk < 0.3:
                recommendations.append(
                    "Standard cancer screening protocols recommended"
                )
            else:
                recommendations.append(
                    "Consider enhanced screening based on family history and risk factors"
                )
        
        return recommendations
    
    def identify_actionable_mutations(
        self, analyses: List[MutationAnalysis]
    ) -> List[Dict[str, Any]]:
        """Identify mutations with therapeutic implications"""
        actionable = []
        
        for analysis in analyses:
            if analysis.targeted_therapies:
                actionable.append({
                    "mutation": analysis.mutation_id,
                    "gene": analysis.gene,
                    "therapies": analysis.targeted_therapies,
                    "therapy_class": self.get_therapy_classes(analysis.targeted_therapies),
                    "fda_approved": self.check_fda_approval(analysis.targeted_therapies),
                    "clinical_trials_available": True  # Placeholder
                })
        
        return actionable
    
    def get_therapy_classes(self, therapies: List[str]) -> List[str]:
        """Get drug classes for therapies"""
        classes = set()
        for therapy in therapies:
            if therapy in self.drug_db:
                classes.add(self.drug_db[therapy].get('class', 'Unknown'))
        return list(classes)
    
    def check_fda_approval(self, therapies: List[str]) -> bool:
        """Check if any therapy is FDA approved"""
        for therapy in therapies:
            if therapy in self.drug_db and self.drug_db[therapy].get('fda_approved', False):
                return True
        return False
    
    def predict_tumor_types(self, analyses: List[MutationAnalysis]) -> List[Dict[str, float]]:
        """Predict likely tumor types based on mutation profile"""
        tumor_scores = {}
        
        # Count mutations associated with each cancer type
        for analysis in analyses:
            for cancer_type in analysis.cancer_types:
                if cancer_type not in tumor_scores:
                    tumor_scores[cancer_type] = 0
                # Weight by pathogenicity
                tumor_scores[cancer_type] += analysis.pathogenicity_score
        
        # Normalize scores
        total_score = sum(tumor_scores.values())
        if total_score > 0:
            tumor_scores = {
                cancer: score / total_score 
                for cancer, score in tumor_scores.items()
            }
        
        # Sort by likelihood
        sorted_tumors = sorted(
            tumor_scores.items(), key=lambda x: x[1], reverse=True
        )
        
        # Return top predictions
        return [
            {"cancer_type": cancer, "likelihood": score}
            for cancer, score in sorted_tumors[:5]
        ]
    
    def calculate_confidence_metrics(self, analyses: List[MutationAnalysis]) -> Dict[str, float]:
        """Calculate confidence metrics for the analysis"""
        if not analyses:
            return {"overall_confidence": 0.0, "data_completeness": 0.0}
        
        # Average confidence across mutations
        avg_confidence = sum(a.confidence_score for a in analyses) / len(analyses)
        
        # Data completeness (how many mutations were in database vs AI)
        known_mutations = sum(1 for a in analyses if "COSMIC" in a.references)
        data_completeness = known_mutations / len(analyses)
        
        return {
            "overall_confidence": round(avg_confidence, 2),
            "data_completeness": round(data_completeness, 2),
            "known_mutations": known_mutations,
            "ai_predictions": len(analyses) - known_mutations
        }
    
    def generate_warnings(self, analyses: List[MutationAnalysis]) -> List[str]:
        """Generate warnings for the analysis"""
        warnings = []
        
        # Check for low confidence analyses
        low_confidence = [
            a for a in analyses if a.confidence_score < 0.5
        ]
        if low_confidence:
            warnings.append(
                f"{len(low_confidence)} mutation(s) had low confidence scores. "
                "Consider clinical validation."
            )
        
        # Check for contradictory results
        genes_with_multiple = {}
        for analysis in analyses:
            if analysis.gene not in genes_with_multiple:
                genes_with_multiple[analysis.gene] = []
            genes_with_multiple[analysis.gene].append(analysis)
        
        for gene, gene_analyses in genes_with_multiple.items():
            if len(gene_analyses) > 1:
                significances = {a.clinical_significance for a in gene_analyses}
                if len(significances) > 1:
                    warnings.append(
                        f"Multiple variants in {gene} with different clinical significances"
                    )
        
        return warnings
    
    def generate_clustering_insights(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate human-readable clustering insights"""
        insights = []
        
        if cluster_analysis.get('total_clusters', 0) == 0:
            return ["No significant mutation clusters identified"]
        
        total_clusters = cluster_analysis['total_clusters']
        insights.append(f"Identified {total_clusters} distinct mutation cluster{'s' if total_clusters > 1 else ''}")
        
        # High-risk cluster insights
        high_risk_clusters = cluster_analysis.get('high_risk_clusters', [])
        if high_risk_clusters:
            insights.append(f"{len(high_risk_clusters)} high-risk cluster{'s' if len(high_risk_clusters) > 1 else ''} requiring immediate attention")
        
        # Pathway insights
        pathway_insights = cluster_analysis.get('pathway_insights', {})
        dominant_pathways = pathway_insights.get('dominant_pathways', [])
        if dominant_pathways:
            pathways_str = ', '.join(dominant_pathways[:3])
            insights.append(f"Primary pathways affected: {pathways_str}")
        
        # Therapeutic opportunities
        therapeutic_opportunities = cluster_analysis.get('therapeutic_opportunities', [])
        if therapeutic_opportunities:
            high_priority = [t for t in therapeutic_opportunities if t.get('priority') == 'HIGH']
            if high_priority:
                insights.append(f"{len(high_priority)} cluster{'s' if len(high_priority) > 1 else ''} with high-priority therapeutic targets")
        
        return insights
    
    def generate_clustering_recommendations(self, cluster_analysis: Dict[str, Any]) -> List[str]:
        """Generate clinical recommendations based on clustering analysis"""
        recommendations = []
        
        if cluster_analysis.get('total_clusters', 0) == 0:
            return []
        
        # High-risk cluster recommendations
        high_risk_clusters = cluster_analysis.get('high_risk_clusters', [])
        for cluster in high_risk_clusters:
            pathways = cluster.get('pathways_involved', [])
            if 'DNA_REPAIR' in pathways:
                recommendations.append(
                    f"High-risk DNA repair cluster identified - consider PARP inhibitor therapy and genetic counseling"
                )
            elif 'P53_PATHWAY' in pathways:
                recommendations.append(
                    f"P53 pathway disruption cluster - consider aggressive treatment protocol and clinical trial enrollment"
                )
            elif 'RTK_SIGNALING' in pathways:
                recommendations.append(
                    f"Receptor tyrosine kinase cluster - multiple targeted therapy options available"
                )
        
        # Therapeutic opportunity recommendations
        therapeutic_opportunities = cluster_analysis.get('therapeutic_opportunities', [])
        high_priority_therapies = [t for t in therapeutic_opportunities if t.get('priority') == 'HIGH']
        if high_priority_therapies:
            recommendations.append(
                f"High-priority therapeutic targets identified in {len(high_priority_therapies)} cluster{'s' if len(high_priority_therapies) > 1 else ''} - prioritize molecular tumor board review"
            )
        
        # Pathway-specific recommendations
        pathway_insights = cluster_analysis.get('pathway_insights', {})
        dominant_pathways = pathway_insights.get('dominant_pathways', [])
        
        if 'MISMATCH_REPAIR' in dominant_pathways:
            recommendations.append(
                "Mismatch repair pathway involvement - consider microsatellite instability testing and immunotherapy"
            )
        
        if 'RAS_RAF' in dominant_pathways:
            recommendations.append(
                "RAS/RAF pathway activation - consider combination targeted therapy with MEK/ERK inhibitors"
            )
        
        return recommendations