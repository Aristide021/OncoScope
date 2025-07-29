"""
Advanced Cancer Risk Calculator
Comprehensive risk scoring system for cancer mutations and patient profiles
"""

import math
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, date
import logging

from .models import MutationAnalysis, ClinicalSignificance, RiskLevel, Prognosis

logger = logging.getLogger(__name__)

@dataclass
class PatientProfile:
    """Patient demographic and clinical profile"""
    age: int
    sex: str  # 'M' or 'F'
    ethnicity: Optional[str] = None
    family_history: List[str] = None  # List of cancer types in family
    smoking_status: str = "unknown"  # never, former, current, unknown
    alcohol_use: str = "unknown"  # none, moderate, heavy, unknown
    bmi: Optional[float] = None
    prior_cancers: List[str] = None
    medications: List[str] = None
    
    def __post_init__(self):
        if self.family_history is None:
            self.family_history = []
        if self.prior_cancers is None:
            self.prior_cancers = []
        if self.medications is None:
            self.medications = []

@dataclass 
class RiskFactors:
    """Individual risk factor contributions"""
    mutation_risk: float
    age_risk: float
    family_history_risk: float
    lifestyle_risk: float
    environmental_risk: float
    protective_factors: float

@dataclass
class CancerRiskAssessment:
    """Comprehensive cancer risk assessment"""
    overall_risk_score: float
    risk_level: RiskLevel
    lifetime_risk_percentage: float
    five_year_risk_percentage: float
    risk_factors: RiskFactors
    cancer_type_risks: Dict[str, float]
    recommendations: List[str]
    confidence_interval: Tuple[float, float]
    risk_explanation: str
    next_screening_date: Optional[str]

class AdvancedRiskCalculator:
    """Advanced cancer risk calculator with population genetics and clinical guidelines"""
    
    def __init__(self):
        """Initialize risk calculator with population data and clinical guidelines"""
        
        # Population baseline cancer incidence rates (per 100,000)
        self.baseline_incidence = {
            'breast': {'F': 127.0, 'M': 0.9},
            'prostate': {'M': 104.0, 'F': 0.0},
            'lung': {'F': 51.5, 'M': 67.8},
            'colorectal': {'F': 38.1, 'M': 46.9},
            'ovarian': {'F': 11.2, 'M': 0.0},
            'pancreatic': {'F': 11.0, 'M': 13.8},
            'melanoma': {'F': 15.8, 'M': 25.6},
            'thyroid': {'F': 22.1, 'M': 7.6},
            'endometrial': {'F': 27.2, 'M': 0.0},
            'gastric': {'F': 7.0, 'M': 11.8}
        }
        
        # Age-specific risk multipliers
        self.age_multipliers = {
            'breast': {
                30: 0.1, 35: 0.2, 40: 0.4, 45: 0.7, 50: 1.0, 
                55: 1.4, 60: 1.8, 65: 2.1, 70: 2.3, 75: 2.4, 80: 2.3
            },
            'prostate': {
                40: 0.1, 45: 0.2, 50: 0.5, 55: 1.0, 60: 2.0,
                65: 3.5, 70: 5.0, 75: 6.0, 80: 6.5
            },
            'colorectal': {
                30: 0.1, 35: 0.2, 40: 0.3, 45: 0.5, 50: 1.0,
                55: 1.5, 60: 2.2, 65: 3.0, 70: 3.5, 75: 3.8, 80: 4.0
            }
        }
        
        # Mutation-specific relative risks
        self.mutation_relative_risks = {
            'BRCA1': {
                'breast': {'F': 72.0, 'M': 0.1},  # 72% lifetime risk for women
                'ovarian': {'F': 44.0, 'M': 0.0}
            },
            'BRCA2': {
                'breast': {'F': 69.0, 'M': 6.8},
                'ovarian': {'F': 17.0, 'M': 0.0},
                'prostate': {'M': 27.0, 'F': 0.0}
            },
            'TP53': {
                'breast': {'F': 85.0, 'M': 0.1},
                'colorectal': {'F': 25.0, 'M': 25.0},
                'lung': {'F': 15.0, 'M': 15.0}
            },
            'APC': {
                'colorectal': {'F': 87.0, 'M': 87.0}
            },
            'MLH1': {
                'colorectal': {'F': 54.0, 'M': 74.0},
                'endometrial': {'F': 71.0, 'M': 0.0}
            },
            'MSH2': {
                'colorectal': {'F': 52.0, 'M': 69.0},
                'endometrial': {'F': 57.0, 'M': 0.0}
            },
            'PALB2': {
                'breast': {'F': 53.0, 'M': 1.0}
            },
            'CHEK2': {
                'breast': {'F': 37.0, 'M': 0.1}
            }
        }
        
        # Family history relative risks
        self.family_history_risks = {
            'breast': {
                'one_first_degree': 2.1,
                'two_first_degree': 3.6,
                'one_second_degree': 1.5,
                'maternal_and_paternal': 2.9
            },
            'colorectal': {
                'one_first_degree': 2.2,
                'two_first_degree': 3.8,
                'one_second_degree': 1.6
            },
            'prostate': {
                'one_first_degree': 2.5,
                'two_first_degree': 4.3,
                'one_second_degree': 1.7
            }
        }
        
        # Lifestyle risk factors
        self.lifestyle_modifiers = {
            'smoking': {
                'lung': {'current': 15.0, 'former': 9.0, 'never': 1.0},
                'colorectal': {'current': 1.4, 'former': 1.2, 'never': 1.0},
                'pancreatic': {'current': 2.2, 'former': 1.7, 'never': 1.0}
            },
            'alcohol': {
                'breast': {'heavy': 1.6, 'moderate': 1.2, 'none': 1.0},
                'liver': {'heavy': 5.0, 'moderate': 2.0, 'none': 1.0},
                'colorectal': {'heavy': 1.4, 'moderate': 1.1, 'none': 1.0}
            },
            'bmi': {
                'breast_postmenopausal': {'obese': 1.3, 'overweight': 1.1, 'normal': 1.0},
                'colorectal': {'obese': 1.5, 'overweight': 1.2, 'normal': 1.0},
                'endometrial': {'obese': 3.4, 'overweight': 1.8, 'normal': 1.0}
            }
        }
    
    def calculate_comprehensive_risk(
        self,
        mutations: List[MutationAnalysis],
        patient_profile: PatientProfile,
        cancer_type: Optional[str] = None
    ) -> CancerRiskAssessment:
        """Calculate comprehensive cancer risk assessment"""
        
        # Calculate individual risk components
        mutation_risk = self._calculate_mutation_risk(mutations, patient_profile, cancer_type)
        age_risk = self._calculate_age_risk(patient_profile.age, cancer_type or 'breast')
        family_risk = self._calculate_family_history_risk(patient_profile, cancer_type)
        lifestyle_risk = self._calculate_lifestyle_risk(patient_profile, cancer_type)
        environmental_risk = self._calculate_environmental_risk(patient_profile)
        protective_factors = self._calculate_protective_factors(patient_profile)
        
        risk_factors = RiskFactors(
            mutation_risk=mutation_risk,
            age_risk=age_risk,
            family_history_risk=family_risk,
            lifestyle_risk=lifestyle_risk,
            environmental_risk=environmental_risk,
            protective_factors=protective_factors
        )
        
        # Calculate overall risk score
        overall_risk = self._combine_risk_factors(risk_factors)
        
        # Calculate cancer-type-specific risks
        cancer_type_risks = self._calculate_cancer_type_risks(
            mutations, patient_profile, risk_factors
        )
        
        # Convert to lifetime and 5-year risks
        lifetime_risk, five_year_risk = self._convert_to_clinical_risks(
            overall_risk, patient_profile.age, cancer_type or 'breast'
        )
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            risk_factors, cancer_type_risks, patient_profile
        )
        
        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(overall_risk, mutations)
        
        # Generate risk explanation
        risk_explanation = self._generate_risk_explanation(risk_factors, cancer_type_risks)
        
        # Determine next screening date
        next_screening = self._calculate_next_screening_date(
            overall_risk, patient_profile, cancer_type_risks
        )
        
        return CancerRiskAssessment(
            overall_risk_score=overall_risk,
            risk_level=self._classify_risk_level(overall_risk),
            lifetime_risk_percentage=lifetime_risk,
            five_year_risk_percentage=five_year_risk,
            risk_factors=risk_factors,
            cancer_type_risks=cancer_type_risks,
            recommendations=recommendations,
            confidence_interval=confidence_interval,
            risk_explanation=risk_explanation,
            next_screening_date=next_screening
        )
    
    def _calculate_mutation_risk(
        self,
        mutations: List[MutationAnalysis],
        patient_profile: PatientProfile,
        cancer_type: Optional[str]
    ) -> float:
        """Calculate risk contribution from genetic mutations"""
        if not mutations:
            return 0.0
        
        mutation_risk = 0.0
        high_penetrance_found = False
        
        for mutation in mutations:
            gene = mutation.gene
            pathogenicity = mutation.pathogenicity_score
            
            # High-penetrance mutations (BRCA1/2, TP53, etc.)
            if gene in self.mutation_relative_risks:
                high_penetrance_found = True
                gene_risks = self.mutation_relative_risks[gene]
                
                for cancer, sex_risks in gene_risks.items():
                    if cancer_type is None or cancer == cancer_type:
                        risk = sex_risks.get(patient_profile.sex, 0.0)
                        # Weight by pathogenicity and clinical significance
                        weighted_risk = risk * pathogenicity
                        
                        if mutation.clinical_significance == ClinicalSignificance.PATHOGENIC:
                            weighted_risk *= 1.0
                        elif mutation.clinical_significance == ClinicalSignificance.LIKELY_PATHOGENIC:
                            weighted_risk *= 0.8
                        else:
                            weighted_risk *= 0.5
                        
                        mutation_risk = max(mutation_risk, weighted_risk / 100.0)
            
            # Moderate-penetrance mutations
            else:
                moderate_risk = pathogenicity * 0.3  # Base moderate risk
                if mutation.clinical_significance == ClinicalSignificance.PATHOGENIC:
                    moderate_risk *= 1.5
                
                mutation_risk += moderate_risk * 0.1  # Additive for moderate risk
        
        # Cap mutation risk at 0.9 (90%)
        return min(mutation_risk, 0.9)
    
    def _calculate_age_risk(self, age: int, cancer_type: str) -> float:
        """Calculate age-related risk increase"""
        if cancer_type not in self.age_multipliers:
            # Generic age risk for other cancers
            if age < 40:
                return 0.1
            elif age < 50:
                return 0.3
            elif age < 60:
                return 0.5
            elif age < 70:
                return 0.7
            else:
                return 0.8
        
        age_curve = self.age_multipliers[cancer_type]
        
        # Find closest age bracket
        ages = sorted(age_curve.keys())
        for i, age_bracket in enumerate(ages):
            if age <= age_bracket:
                if i == 0:
                    return age_curve[age_bracket] * 0.1
                else:
                    # Linear interpolation
                    prev_age = ages[i-1]
                    age_factor = (age - prev_age) / (age_bracket - prev_age)
                    interpolated = age_curve[prev_age] + age_factor * (age_curve[age_bracket] - age_curve[prev_age])
                    return min(interpolated * 0.1, 0.8)
        
        # Age beyond maximum in table
        return min(age_curve[ages[-1]] * 0.1, 0.8)
    
    def _calculate_family_history_risk(
        self,
        patient_profile: PatientProfile,
        cancer_type: Optional[str]
    ) -> float:
        """Calculate family history risk contribution"""
        if not patient_profile.family_history:
            return 0.0
        
        family_risk = 0.0
        cancer_types = [cancer_type] if cancer_type else list(self.family_history_risks.keys())
        
        for cancer in cancer_types:
            if cancer in self.family_history_risks:
                cancer_count = sum(1 for fh_cancer in patient_profile.family_history if cancer in fh_cancer.lower())
                
                if cancer_count >= 2:
                    family_risk = max(family_risk, 0.4)  # High family history
                elif cancer_count == 1:
                    family_risk = max(family_risk, 0.2)  # Moderate family history
        
        return min(family_risk, 0.5)
    
    def _calculate_lifestyle_risk(
        self,
        patient_profile: PatientProfile,
        cancer_type: Optional[str]
    ) -> float:
        """Calculate lifestyle-related risk factors"""
        lifestyle_risk = 0.0
        
        # Smoking risk
        if patient_profile.smoking_status in ['current', 'former']:
            if cancer_type in ['lung', 'colorectal', 'pancreatic']:
                smoking_multiplier = 0.3 if patient_profile.smoking_status == 'current' else 0.2
                lifestyle_risk += smoking_multiplier
        
        # Alcohol risk
        if patient_profile.alcohol_use == 'heavy':
            if cancer_type in ['breast', 'liver', 'colorectal']:
                lifestyle_risk += 0.2
        elif patient_profile.alcohol_use == 'moderate':
            if cancer_type in ['breast', 'liver']:
                lifestyle_risk += 0.1
        
        # BMI risk
        if patient_profile.bmi:
            if patient_profile.bmi >= 30:  # Obese
                if cancer_type in ['breast', 'colorectal', 'endometrial']:
                    lifestyle_risk += 0.15
            elif patient_profile.bmi >= 25:  # Overweight
                if cancer_type in ['breast', 'colorectal', 'endometrial']:
                    lifestyle_risk += 0.08
        
        return min(lifestyle_risk, 0.4)
    
    def _calculate_environmental_risk(self, patient_profile: PatientProfile) -> float:
        """Calculate environmental risk factors"""
        # Placeholder for environmental factors
        # In production, this would consider geographic location, occupational exposures, etc.
        return 0.05  # Base environmental risk
    
    def _calculate_protective_factors(self, patient_profile: PatientProfile) -> float:
        """Calculate protective factors that reduce risk"""
        protective_benefit = 0.0
        
        # Check for protective medications
        if patient_profile.medications:
            # Aspirin for colorectal cancer
            if any('aspirin' in med.lower() for med in patient_profile.medications):
                protective_benefit += 0.1
            
            # Statins for various cancers
            if any('statin' in med.lower() for med in patient_profile.medications):
                protective_benefit += 0.05
        
        # Healthy lifestyle factors (if BMI is healthy)
        if patient_profile.bmi and 18.5 <= patient_profile.bmi < 25:
            protective_benefit += 0.05
        
        # Never smoker
        if patient_profile.smoking_status == 'never':
            protective_benefit += 0.05
        
        return min(protective_benefit, 0.2)
    
    def _combine_risk_factors(self, risk_factors: RiskFactors) -> float:
        """Combine individual risk factors into overall score"""
        # Use multiplicative model for most factors, additive for lifestyle
        mutation_component = risk_factors.mutation_risk
        
        # If high mutation risk, it dominates
        if mutation_component > 0.5:
            base_risk = mutation_component
            other_factors = (
                risk_factors.age_risk * 0.3 +
                risk_factors.family_history_risk * 0.2 +
                risk_factors.lifestyle_risk * 0.2 +
                risk_factors.environmental_risk * 0.1
            )
        else:
            # For lower mutation risk, combine more evenly
            base_risk = (
                mutation_component * 0.4 +
                risk_factors.age_risk * 0.25 +
                risk_factors.family_history_risk * 0.2 +
                risk_factors.lifestyle_risk * 0.15
            )
            other_factors = risk_factors.environmental_risk
        
        # Apply protective factors
        combined_risk = base_risk + other_factors - risk_factors.protective_factors
        
        # Ensure risk stays in valid range
        return max(0.0, min(combined_risk, 1.0))
    
    def _calculate_cancer_type_risks(
        self,
        mutations: List[MutationAnalysis],
        patient_profile: PatientProfile,
        risk_factors: RiskFactors
    ) -> Dict[str, float]:
        """Calculate risk for different cancer types"""
        cancer_risks = {}
        
        # Calculate risk for each major cancer type
        for cancer_type in ['breast', 'prostate', 'lung', 'colorectal', 'ovarian']:
            # Skip gender-specific cancers for wrong gender
            if cancer_type == 'prostate' and patient_profile.sex != 'M':
                continue
            if cancer_type in ['breast', 'ovarian'] and patient_profile.sex == 'M':
                continue
            
            # Calculate type-specific risk
            type_risk = self._calculate_mutation_risk(mutations, patient_profile, cancer_type)
            type_risk += self._calculate_age_risk(patient_profile.age, cancer_type)
            type_risk += self._calculate_family_history_risk(patient_profile, cancer_type)
            type_risk += self._calculate_lifestyle_risk(patient_profile, cancer_type)
            
            cancer_risks[cancer_type] = min(type_risk, 1.0)
        
        return cancer_risks
    
    def _convert_to_clinical_risks(
        self,
        risk_score: float,
        age: int,
        cancer_type: str
    ) -> Tuple[float, float]:
        """Convert risk score to lifetime and 5-year clinical risks"""
        
        # Get baseline incidence for cancer type and gender
        base_lifetime = 12.0  # Default 12% lifetime risk
        if cancer_type in self.baseline_incidence:
            # Use population baseline (convert from per 100,000 to percentage)
            base_lifetime = 8.0  # Simplified baseline
        
        # Calculate lifetime risk
        lifetime_risk = base_lifetime * (1 + risk_score * 4)  # Risk score amplifies baseline
        lifetime_risk = min(lifetime_risk, 85.0)  # Cap at 85%
        
        # Calculate 5-year risk (much lower)
        remaining_years = max(85 - age, 5)
        five_year_risk = lifetime_risk * (5.0 / remaining_years) * 0.8  # Conservative estimate
        five_year_risk = min(five_year_risk, 25.0)  # Cap at 25%
        
        return lifetime_risk, five_year_risk
    
    def _generate_risk_recommendations(
        self,
        risk_factors: RiskFactors,
        cancer_type_risks: Dict[str, float],
        patient_profile: PatientProfile
    ) -> List[str]:
        """Generate personalized risk management recommendations"""
        recommendations = []
        
        # High mutation risk recommendations
        if risk_factors.mutation_risk > 0.5:
            recommendations.append("Genetic counseling strongly recommended")
            recommendations.append("Consider enhanced screening protocols")
            recommendations.append("Discuss risk-reducing strategies with oncologist")
        
        # Age-based screening
        if patient_profile.age >= 50:
            if 'colorectal' in cancer_type_risks and cancer_type_risks['colorectal'] > 0.3:
                recommendations.append("Annual colonoscopy screening recommended")
        
        if patient_profile.sex == 'F' and patient_profile.age >= 40:
            if 'breast' in cancer_type_risks and cancer_type_risks['breast'] > 0.3:
                recommendations.append("Annual mammography with possible MRI")
        
        # Lifestyle modifications
        if risk_factors.lifestyle_risk > 0.2:
            recommendations.append("Lifestyle modifications to reduce cancer risk")
            
            if patient_profile.smoking_status in ['current', 'former']:
                recommendations.append("Smoking cessation counseling and support")
            
            if patient_profile.alcohol_use == 'heavy':
                recommendations.append("Reduce alcohol consumption")
            
            if patient_profile.bmi and patient_profile.bmi >= 25:
                recommendations.append("Weight management and regular exercise")
        
        # Family history considerations
        if risk_factors.family_history_risk > 0.2:
            recommendations.append("Family cancer history review with genetic counselor")
            recommendations.append("Consider earlier screening initiation")
        
        return recommendations
    
    def _calculate_confidence_interval(
        self,
        risk_score: float,
        mutations: List[MutationAnalysis]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for risk estimate"""
        
        # Base confidence on number of known vs unknown mutations
        known_mutations = sum(1 for m in mutations if m.confidence_score > 0.8)
        total_mutations = len(mutations)
        
        if total_mutations == 0:
            confidence_width = 0.3
        else:
            confidence_ratio = known_mutations / total_mutations
            confidence_width = 0.15 + 0.15 * (1 - confidence_ratio)
        
        lower_bound = max(0.0, risk_score - confidence_width)
        upper_bound = min(1.0, risk_score + confidence_width)
        
        return (lower_bound, upper_bound)
    
    def _generate_risk_explanation(
        self,
        risk_factors: RiskFactors,
        cancer_type_risks: Dict[str, float]
    ) -> str:
        """Generate human-readable risk explanation"""
        
        explanations = []
        
        # Primary risk drivers
        if risk_factors.mutation_risk > 0.4:
            explanations.append("High-risk genetic mutations are the primary risk factor")
        elif risk_factors.family_history_risk > 0.3:
            explanations.append("Strong family history significantly increases risk")
        elif risk_factors.age_risk > 0.4:
            explanations.append("Age-related risk is the predominant factor")
        
        # Contributing factors
        if risk_factors.lifestyle_risk > 0.2:
            explanations.append("Lifestyle factors contribute to elevated risk")
        
        if risk_factors.protective_factors > 0.1:
            explanations.append("Protective factors help reduce overall risk")
        
        # Specific cancer risks
        high_risk_cancers = [cancer for cancer, risk in cancer_type_risks.items() if risk > 0.5]
        if high_risk_cancers:
            explanations.append(f"Particularly elevated risk for: {', '.join(high_risk_cancers)}")
        
        return ". ".join(explanations) if explanations else "Risk assessment based on available factors"
    
    def _calculate_next_screening_date(
        self,
        overall_risk: float,
        patient_profile: PatientProfile,
        cancer_type_risks: Dict[str, float]
    ) -> Optional[str]:
        """Calculate recommended next screening date"""
        
        # Determine screening interval based on risk
        if overall_risk > 0.7:
            months = 6  # High risk - every 6 months
        elif overall_risk > 0.5:
            months = 12  # Moderate-high risk - annually
        elif overall_risk > 0.3:
            months = 18  # Moderate risk - every 1.5 years
        else:
            months = 24  # Low risk - every 2 years
        
        # Adjust for age
        if patient_profile.age >= 65:
            months = max(months - 6, 6)  # More frequent for elderly
        elif patient_profile.age < 40:
            months = min(months + 12, 36)  # Less frequent for young adults
        
        # Calculate next date
        from datetime import datetime, timedelta
        next_date = datetime.now() + timedelta(days=months * 30)
        return next_date.strftime("%Y-%m-%d")
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk level based on score"""
        if risk_score >= 0.7:
            return RiskLevel.HIGH
        elif risk_score >= 0.5:
            return RiskLevel.MEDIUM_HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        elif risk_score >= 0.15:
            return RiskLevel.LOW_MEDIUM
        else:
            return RiskLevel.LOW

def calculate_polygenic_risk_score(snp_data: Dict[str, str]) -> float:
    """Calculate polygenic risk score from SNP data"""
    
    # Example polygenic risk weights for breast cancer
    # In production, this would use validated PRS models
    prs_weights = {
        'rs2981582': {'T': 0.1, 'C': 0.0},  # FGFR2
        'rs3803662': {'T': 0.08, 'C': 0.0},  # TOX3
        'rs13281615': {'A': 0.06, 'G': 0.0},  # 8q24
        'rs4973768': {'T': 0.05, 'C': 0.0},  # SLC4A7
        'rs6504950': {'A': 0.04, 'G': 0.0}   # STXBP4
    }
    
    prs_score = 0.0
    counted_snps = 0
    
    for snp, alleles in snp_data.items():
        if snp in prs_weights:
            for allele in alleles:
                if allele in prs_weights[snp]:
                    prs_score += prs_weights[snp][allele]
                    counted_snps += 1
    
    # Normalize by number of counted SNPs
    if counted_snps > 0:
        return min(prs_score / counted_snps, 0.3)  # Cap PRS contribution
    
    return 0.0