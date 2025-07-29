"""
OncoScope Data Models
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator
from enum import Enum

class ClinicalSignificance(str, Enum):
    """Clinical significance classifications"""
    PATHOGENIC = "PATHOGENIC"
    LIKELY_PATHOGENIC = "LIKELY_PATHOGENIC"
    UNCERTAIN = "VARIANT_OF_UNCERTAIN_SIGNIFICANCE"
    LIKELY_BENIGN = "LIKELY_BENIGN"
    BENIGN = "BENIGN"

class RiskLevel(str, Enum):
    """Risk level classifications"""
    HIGH = "HIGH"
    MEDIUM_HIGH = "MEDIUM-HIGH"
    MEDIUM = "MEDIUM"
    LOW_MEDIUM = "LOW-MEDIUM"
    LOW = "LOW"

class Prognosis(str, Enum):
    """Prognosis classifications"""
    POOR = "poor"
    MODERATE = "moderate"
    GOOD = "good"
    EXCELLENT_WITH_THERAPY = "excellent_with_therapy"
    UNCERTAIN = "uncertain"

class MutationInput(BaseModel):
    """Input mutation format"""
    mutation: str = Field(..., description="Mutation in format GENE:variant (e.g., TP53:c.524G>A)")
    
    @validator('mutation')
    def validate_mutation_format(cls, v):
        if ':' not in v and not any(sep in v for sep in [' ', '-', '_']):
            raise ValueError("Mutation must include gene and variant separated by ':', space, '-', or '_'")
        return v.strip()

class MutationAnalysis(BaseModel):
    """Individual mutation analysis result"""
    mutation_id: str
    gene: str
    variant: str
    protein_change: Optional[str] = None
    pathogenicity_score: float = Field(..., ge=0.0, le=1.0)
    cancer_types: List[str] = []
    clinical_significance: ClinicalSignificance
    targeted_therapies: List[str] = []
    prognosis_impact: Prognosis
    mechanism: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    references: List[str] = []

class AnalysisRequest(BaseModel):
    """Request for mutation analysis"""
    mutations: List[str] = Field(..., min_items=1, max_items=100)
    include_drug_interactions: bool = True
    include_clinical_trials: bool = False
    patient_context: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    """Complete analysis response"""
    success: bool
    analysis_id: str
    timestamp: datetime
    individual_mutations: List[MutationAnalysis]
    overall_risk_score: float = Field(..., ge=0.0, le=1.0)
    risk_classification: RiskLevel
    clinical_recommendations: List[str]
    actionable_mutations: List[Dict[str, Any]]
    estimated_tumor_types: List[Dict[str, float]]
    confidence_metrics: Dict[str, float]
    warnings: List[str] = []

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str
    ollama_connected: bool
    model_loaded: bool
    database_ready: bool
    version: str
    uptime_seconds: float

class ErrorResponse(BaseModel):
    """Error response format"""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ExportRequest(BaseModel):
    """Export request model"""
    analysis_id: str
    format: str = Field(default="pdf", pattern="^(pdf|json|csv|html)$")
    include_raw_data: bool = False

class TherapyRecommendation(BaseModel):
    """Therapy recommendation details"""
    drug_name: str
    drug_class: str
    mechanism_of_action: str
    efficacy_score: float = Field(..., ge=0.0, le=1.0)
    fda_approved: bool
    clinical_trials: List[str] = []
    contraindications: List[str] = []
    combination_therapies: List[str] = []

class CancerTypeAssociation(BaseModel):
    """Cancer type association details"""
    cancer_type: str
    frequency: float = Field(..., ge=0.0, le=1.0)
    subtype: Optional[str] = None
    stage_association: Optional[str] = None
    prognosis_modifier: Optional[str] = None