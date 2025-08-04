"""
Core Cancer Mutation Analysis Engine
"""
import json
import re
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
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
            },
            "BRAF": {
                "c.1799T>A": {
                    "protein": "p.V600E",
                    "pathogenicity": 0.95,
                    "cancer_types": ["melanoma", "colorectal", "thyroid", "lung"],
                    "frequency": "50% of melanomas, 10% of colorectal cancers",
                    "mechanism": "Constitutive kinase activation in MAPK pathway",
                    "prognosis": "moderate",
                    "targeted_therapy": ["vemurafenib", "dabrafenib", "encorafenib"]
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
            },
            "vemurafenib": {
                "targets": ["BRAF V600E"],
                "class": "BRAF Inhibitor",
                "generation": "1st",
                "fda_approved": True,
                "indications": ["BRAF V600E melanoma", "Erdheim-Chester disease"]
            },
            "dabrafenib": {
                "targets": ["BRAF V600E", "BRAF V600K"],
                "class": "BRAF Inhibitor",
                "generation": "2nd",
                "fda_approved": True,
                "indications": ["BRAF V600E/K melanoma", "NSCLC", "anaplastic thyroid"]
            },
            "encorafenib": {
                "targets": ["BRAF V600E", "BRAF V600K"],
                "class": "BRAF Inhibitor",
                "generation": "2nd",
                "fda_approved": True,
                "indications": ["BRAF V600E/K melanoma", "colorectal cancer (with cetuximab)"]
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
        """Analyze individual mutation for clinical significance with caching"""
        logger.info(f"=== ANALYZING MUTATION: {mutation} ===")
        gene, variant = self.parse_mutation_notation(mutation)
        mutation_id = f"{gene}:{variant}"
        logger.info(f"Parsed as gene={gene}, variant={variant}")
        
        # Check cache first
        try:
            from .database import get_cached_mutation, cache_mutation_analysis
            cached_data = await get_cached_mutation(mutation_id)
            
            if cached_data:
                logger.info(f"Found cached analysis for {mutation_id}, but will use AI for Gemma 3n showcase")
                # Don't return cached data - always use AI for the competition
        except Exception as e:
            logger.warning(f"Cache lookup failed for {mutation_id}: {e}")
        
        # ALWAYS use AI analysis to showcase Gemma 3n capabilities
        # This ensures every mutation gets rich, AI-powered insights
        logger.info(f"Using Gemma 3n AI analysis for {mutation_id}")
        
        # Get database info if available
        db_data = None
        if gene in self.mutation_db and variant in self.mutation_db[gene]:
            db_data = self.mutation_db[gene][variant]
        
        # Call AI model for detailed analysis (showcasing Gemma 3n)
        analysis = await self.ai_analyze_mutation(gene, variant, db_data)
        
        # Cache the result
        try:
            await cache_mutation_analysis(
                mutation_id=mutation_id,
                gene=gene,
                variant=variant,
                analysis_data=analysis.model_dump()
            )
            logger.info(f"Cached analysis for {mutation_id}")
        except Exception as e:
            logger.warning(f"Failed to cache analysis for {mutation_id}: {e}")
        
        return analysis
    
    async def ai_analyze_mutation(self, gene: str, variant: str, db_data: Optional[Dict] = None) -> MutationAnalysis:
        """Use AI model for unknown mutations"""
        try:
            logger.info(f"Calling Gemma 3n AI model for {gene}:{variant}")
            ai_analysis = await self.ollama_client.analyze_cancer_mutation(gene, variant)
            
            if not ai_analysis:
                logger.warning(f"AI analysis returned None for {gene}:{variant}")
                return self.create_uncertain_analysis(gene, variant)
            
            logger.info(f"AI analysis result: {ai_analysis}")
            
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
        """Analyze list of mutations with CLUSTER-FIRST approach for integrated AI analysis"""
        
        # Extract analysis_id if present for progress tracking
        analysis_id = patient_context.get('analysis_id') if patient_context else None
        
        async def update_progress(progress: int, message: str):
            """Update analysis progress in global status"""
            if analysis_id:
                from .main import analysis_status
                if analysis_id in analysis_status:
                    analysis_status[analysis_id].update({
                        "progress": progress,
                        "message": message,
                        "status": "in_progress"
                    })
        
        # STEP 1: Parse mutations to get basic info for clustering
        await update_progress(10, "Parsing and validating mutations...")
        parsed_mutations = []
        for mutation in mutations:
            gene, variant = self.parse_mutation_notation(mutation)
            if gene and variant:
                # Get basic mutation info from database
                basic_info = self._get_basic_mutation_info(gene, variant)
                parsed_mutations.append({
                    "gene": gene,
                    "variant": variant,
                    "mutation_string": mutation,
                    **basic_info
                })
        
        # STEP 2: CLUSTER FIRST if we have enough mutations
        await update_progress(25, "Searching mutation databases...")
        clustering_results = {}
        cluster_context = None
        
        if include_clustering and len(parsed_mutations) >= 3:
            try:
                # Create minimal MutationAnalysis objects for clustering
                minimal_analyses = [
                    MutationAnalysis(
                        mutation_id=f"{mut['gene']}:{mut['variant']}",
                        gene=mut["gene"],
                        variant=mut["variant"],
                        protein_change=None,
                        pathogenicity_score=mut.get("pathogenicity_score", 0.5),
                        cancer_types=[],  # Will be filled by AI
                        clinical_significance=ClinicalSignificance.UNCERTAIN,
                        targeted_therapies=[],
                        prognosis_impact=Prognosis.UNCERTAIN,
                        mechanism="Unknown mechanism - pending analysis",
                        confidence_score=0.5,
                        references=[]
                    )
                    for mut in parsed_mutations
                ]
                
                clusters, cluster_analysis = self.clustering_engine.cluster_mutations(minimal_analyses)
                clustering_results = {
                    "clusters_identified": len(clusters),
                    "cluster_analysis": cluster_analysis,
                    "clustering_insights": self.generate_clustering_insights(cluster_analysis)
                }
                
                # Prepare cluster context for AI
                cluster_context = {
                    "pathway_convergence": cluster_analysis.get("pathway_convergence", {}),
                    "functional_groups": cluster_analysis.get("functional_clusters", {}),
                    "interaction_patterns": cluster_analysis.get("interaction_patterns", {}),
                    "clustering_summary": clustering_results["clustering_insights"]
                }
                
            except Exception as e:
                logger.error(f"Clustering analysis failed: {e}")
                cluster_context = None
        
        # STEP 3: Make SINGLE AI call with clustering context
        await update_progress(60, f"Running Gemma 3n AI analysis on {len(parsed_mutations)} mutations...")
        if len(parsed_mutations) >= 2:  # Use multi-mutation analysis for 2+ mutations
            # Add progress callback to patient context
            if patient_context:
                patient_context['progress_callback'] = lambda progress, msg: asyncio.create_task(update_progress(60 + int(progress * 0.25), msg))
            
            ai_analysis = await self._analyze_mutations_with_ai(
                parsed_mutations, 
                patient_context,
                cluster_context
            )
            
            # Parse AI response into individual analyses
            analyses = self._parse_multi_mutation_ai_response(ai_analysis, parsed_mutations)
        else:
            # Fallback to single mutation analysis
            tasks = [self.analyze_single_mutation(mutation) for mutation in mutations]
            analyses = await asyncio.gather(*tasks)
        
        # Convert to dict format
        # Use .dict() for Pydantic models instead of asdict() for dataclasses
        analyses_dict = [analysis.model_dump() for analysis in analyses]
        
        # Calculate composite risk score (enhanced with AI insights)
        await update_progress(85, "Calculating risk scores and generating recommendations...")
        overall_risk = self.calculate_composite_risk(analyses)
        
        # Generate clinical recommendations (now AI-informed)
        recommendations = self.generate_recommendations(
            analyses, overall_risk, patient_context
        )
        
        # If we have AI-generated multi-mutation insights, add them
        if len(parsed_mutations) >= 2 and "multi_mutation_insights" in ai_analysis:
            recommendations.extend(ai_analysis["multi_mutation_insights"])
        
        # Identify actionable mutations
        actionable = self.identify_actionable_mutations(analyses)
        
        # Predict tumor types
        tumor_predictions = self.predict_tumor_types(analyses)
        
        # Calculate confidence metrics
        confidence_metrics = self.calculate_confidence_metrics(analyses)
        
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
        
        # Save analysis to database
        try:
            from .database import save_analysis
            import uuid
            
            # Use provided analysis_id or generate new one
            analysis_id = (patient_context.get("analysis_id") if patient_context else None) or str(uuid.uuid4())
            
            # Add analysis ID and metadata for database storage
            analysis_data = {
                **result,
                "analysis_id": analysis_id,
                "mutations": mutations,
                "user_session": patient_context.get("session_id", "anonymous") if patient_context else "anonymous"
            }
            
            # Save to database asynchronously
            await save_analysis(analysis_data)
            logger.info(f"Analysis saved to database with ID: {analysis_id}")
            
            # Add the analysis_id to the result
            result["analysis_id"] = analysis_id
            
        except Exception as e:
            logger.error(f"Failed to save analysis to database: {e}")
            # Don't fail the analysis if database save fails
            result["database_save_error"] = str(e)
        
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
                    "clinical_trials_available": self.check_clinical_trials_availability(analysis.gene, analysis.targeted_therapies)
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
    
    def check_clinical_trials_availability(self, gene: str, therapies: List[str]) -> bool:
        """Check if clinical trials are likely available for this gene/therapy combination"""
        # High-priority cancer genes typically have active clinical trials
        high_priority_genes = {
            'TP53', 'KRAS', 'EGFR', 'BRCA1', 'BRCA2', 'PIK3CA', 'PTEN', 
            'APC', 'BRAF', 'MYC', 'RB1', 'VHL', 'MLH1', 'MSH2', 'MSH6',
            'ERBB2', 'ALK', 'ROS1', 'MET', 'KIT', 'PDGFRA'
        }
        
        # If gene is high priority, likely has trials
        if gene in high_priority_genes:
            return True
        
        # If any therapy is FDA approved, likely has ongoing trials for combinations
        if self.check_fda_approval(therapies):
            return True
        
        # Check if therapies are in clinical development
        investigational_therapies = {
            'APR-246', 'PRIMA-1MET', 'nutlin-3', 'adagrasib', 'alpelisib',
            'talazoparib', 'rucaparib', 'niraparib', 'veliparib'
        }
        
        for therapy in therapies:
            if therapy.lower() in [t.lower() for t in investigational_therapies]:
                return True
        
        # Default to false for unknown combinations
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
        
        # Count mutations with high confidence and pathogenic significance as "known"
        # These are well-characterized mutations in the literature
        known_mutations = sum(
            1 for a in analyses 
            if a.confidence_score >= 0.8 and 
            a.clinical_significance in [ClinicalSignificance.PATHOGENIC, ClinicalSignificance.LIKELY_PATHOGENIC]
        )
        data_completeness = known_mutations / len(analyses) if analyses else 0
        
        return {
            "overall_confidence": round(avg_confidence, 2),
            "data_completeness": round(data_completeness, 2),
            "known_mutations": known_mutations,
            "ai_predictions": len(analyses)
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
    
    def _get_basic_mutation_info(self, gene: str, variant: str) -> Dict[str, Any]:
        """Get basic mutation information from database for clustering"""
        # Check if mutation exists in database
        if gene in self.mutation_db and variant in self.mutation_db[gene]:
            mut_data = self.mutation_db[gene][variant]
            return {
                "mutation_type": mut_data.get("mutation_type", "unknown"),
                "pathogenicity_score": mut_data.get("pathogenicity_score", 0.5),
                "known_mutation": True
            }
        else:
            # Return defaults for unknown mutations
            return {
                "mutation_type": "unknown",
                "pathogenicity_score": 0.5,
                "known_mutation": False
            }
    
    async def _analyze_mutations_with_ai(
        self,
        mutations: List[Dict[str, Any]],
        patient_context: Optional[Dict[str, Any]],
        cluster_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Make single AI call with clustering context for multi-mutation analysis"""
        logger.info("=== CALLING GEMMA 3N AI FOR MULTI-MUTATION ANALYSIS ===")
        logger.info(f"Analyzing {len(mutations)} mutations with Gemma 3n model")
        
        # Import prompt generator
        # Add parent directory to path for imports
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ai.inference.prompts import GenomicAnalysisPrompts
        prompt_gen = GenomicAnalysisPrompts()
        
        # Prepare mutations for prompt
        mutation_list = [{"gene": m["gene"], "variant": m["variant"]} for m in mutations]
        
        # Create comprehensive prompt with clustering context
        if cluster_context:
            # Enhance patient context with clustering insights
            enhanced_context = patient_context or {}
            enhanced_context["clustering_analysis"] = cluster_context
            
            prompt = prompt_gen.create_multi_mutation_analysis_prompt(
                mutations=mutation_list,
                patient_context=enhanced_context
            )
        else:
            prompt = prompt_gen.create_multi_mutation_analysis_prompt(
                mutations=mutation_list,
                patient_context=patient_context
            )
        
        # Make AI call
        try:
            logger.info("Sending prompt to Gemma 3n via Ollama...")
            response = await self.ollama_client.analyze_multi_mutations(prompt)
            logger.info(f"Got response from Gemma 3n: {response}")
            return response
        except Exception as e:
            logger.error(f"AI multi-mutation analysis failed: {e}")
            # Return fallback structure
            return {
                "individual_analyses": {},
                "multi_mutation_insights": [],
                "pathway_interactions": {},
                "composite_risk": "uncertain"
            }
    
    def _parse_multi_mutation_ai_response(
        self,
        ai_response: Dict[str, Any],
        parsed_mutations: List[Dict[str, Any]]
    ) -> List[MutationAnalysis]:
        """Parse AI response into individual MutationAnalysis objects and cache results"""
        analyses = []
        
        # Import cache function
        from .database import cache_mutation_analysis
        
        # Get individual analyses from AI response
        individual_analyses = ai_response.get("individual_analyses", {})
        
        for mutation in parsed_mutations:
            mutation_key = f"{mutation['gene']}:{mutation['variant']}"
            
            # Check if AI provided analysis for this mutation
            if mutation_key in individual_analyses:
                ai_data = individual_analyses[mutation_key]
                
                # Create MutationAnalysis from AI response
                analysis = MutationAnalysis(
                    mutation_id=mutation_key,
                    gene=mutation["gene"],
                    variant=mutation["variant"],
                    protein_change=ai_data.get("protein_change"),
                    pathogenicity_score=ai_data.get("pathogenicity", 0.5),
                    cancer_types=ai_data.get("cancer_types", []),
                    clinical_significance=self._parse_significance(
                        ai_data.get("significance", "uncertain")
                    ),
                    targeted_therapies=ai_data.get("therapies", []),
                    prognosis_impact=self._parse_prognosis(ai_data.get("prognosis", "uncertain")),
                    mechanism=ai_data.get("mechanism", "Unknown mechanism"),
                    confidence_score=ai_data.get("confidence", 0.7),
                    references=ai_data.get("references", [])
                )
            else:
                # Fallback to basic analysis if AI didn't provide specific analysis
                analysis = MutationAnalysis(
                    mutation_id=f"{mutation['gene']}:{mutation['variant']}",
                    gene=mutation["gene"],
                    variant=mutation["variant"],
                    protein_change=None,
                    pathogenicity_score=mutation.get("pathogenicity_score", 0.5),
                    cancer_types=[],
                    clinical_significance=ClinicalSignificance.UNCERTAIN,
                    targeted_therapies=[],
                    prognosis_impact=Prognosis.UNCERTAIN,
                    mechanism="Unknown mechanism - pending analysis",
                    confidence_score=0.5,
                    references=[]
                )
            
            analyses.append(analysis)
            
            # Cache the analysis
            try:
                asyncio.create_task(cache_mutation_analysis(
                    mutation_id=mutation_key,
                    gene=mutation["gene"],
                    variant=mutation["variant"],
                    analysis_data=analysis.model_dump()
                ))
                logger.debug(f"Queued {mutation_key} for caching")
            except Exception as e:
                logger.warning(f"Failed to queue {mutation_key} for caching: {e}")
        
        return analyses
    
    def _parse_significance(self, significance_str: str) -> ClinicalSignificance:
        """Parse clinical significance from string"""
        # Handle both lowercase and uppercase variants
        sig_map = {
            "pathogenic": ClinicalSignificance.PATHOGENIC,
            "PATHOGENIC": ClinicalSignificance.PATHOGENIC,
            "likely_pathogenic": ClinicalSignificance.LIKELY_PATHOGENIC,
            "LIKELY_PATHOGENIC": ClinicalSignificance.LIKELY_PATHOGENIC,
            "uncertain": ClinicalSignificance.UNCERTAIN,
            "UNCERTAIN": ClinicalSignificance.UNCERTAIN,
            "VARIANT_OF_UNCERTAIN_SIGNIFICANCE": ClinicalSignificance.UNCERTAIN,
            "likely_benign": ClinicalSignificance.LIKELY_BENIGN,
            "LIKELY_BENIGN": ClinicalSignificance.LIKELY_BENIGN,
            "benign": ClinicalSignificance.BENIGN,
            "BENIGN": ClinicalSignificance.BENIGN
        }
        return sig_map.get(significance_str, ClinicalSignificance.UNCERTAIN)
    
    def _parse_prognosis(self, prognosis_str: str) -> Prognosis:
        """Parse prognosis from string"""
        prog_map = {
            "favorable": Prognosis.GOOD,
            "good": Prognosis.GOOD,
            "intermediate": Prognosis.MODERATE,
            "moderate": Prognosis.MODERATE,
            "poor": Prognosis.POOR,
            "excellent": Prognosis.EXCELLENT_WITH_THERAPY,
            "excellent_with_therapy": Prognosis.EXCELLENT_WITH_THERAPY,
            "moderate_with_targeted_therapy": Prognosis.MODERATE_WITH_TARGETED_THERAPY,
            "uncertain": Prognosis.UNCERTAIN
        }
        return prog_map.get(prognosis_str.lower(), Prognosis.UNCERTAIN)
    
    def _parse_therapy_recommendations(self, therapy_data: Dict[str, Any]) -> List[TherapyRecommendation]:
        """Parse therapy recommendations from AI response"""
        recommendations = []
        
        # Parse FDA-approved therapies
        for therapy in therapy_data.get("fda_approved", []):
            recommendations.append(TherapyRecommendation(
                drug_name=therapy.get("drug", ""),
                drug_class=therapy.get("class", ""),
                evidence_level=therapy.get("evidence", "FDA approved"),
                response_rate=therapy.get("response_rate", ""),
                fda_approved=True,
                combination_therapy=therapy.get("combination", False),
                biomarker_required=therapy.get("biomarker_required", True)
            ))
        
        # Parse investigational therapies
        for therapy in therapy_data.get("investigational", []):
            recommendations.append(TherapyRecommendation(
                drug_name=therapy.get("drug", ""),
                drug_class=therapy.get("class", ""),
                evidence_level=therapy.get("evidence", "Investigational"),
                response_rate=therapy.get("response_rate", ""),
                fda_approved=False,
                combination_therapy=therapy.get("combination", False),
                biomarker_required=therapy.get("biomarker_required", True)
            ))
        
        return recommendations