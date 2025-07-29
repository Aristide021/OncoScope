"""
Advanced Cancer Mutation Clustering Engine
Adapted from OutScan's proven Ward linkage hierarchical clustering
Optimized for cancer genomics and clinical decision support
"""

import json
import math
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from collections import defaultdict, Counter
import logging
from dataclasses import dataclass

from .models import MutationAnalysis, ClinicalSignificance, Prognosis, RiskLevel

logger = logging.getLogger(__name__)

class Matrix:
    """Simple matrix implementation for distance calculations"""
    def __init__(self, size: int, default_value: float = 0.0):
        self.size = size
        self.data = [[default_value for _ in range(size)] for _ in range(size)]
    
    def get(self, i: int, j: int) -> float:
        return self.data[i][j]
    
    def set(self, i: int, j: int, value: float):
        self.data[i][j] = value
        self.data[j][i] = value  # Symmetric matrix
    
    def get_row(self, i: int) -> List[float]:
        return self.data[i][:]

@dataclass
class MutationCluster:
    """Represents a cluster of cancer mutations"""
    cluster_id: int
    mutation_indices: List[int]
    centroid_pathogenicity: float
    dominant_cancer_types: List[str]
    actionable_count: int
    clinical_significance: str
    risk_score: float
    size: int

class CancerMutationClusterEngine:
    """Advanced clustering engine optimized for cancer mutations"""
    
    def __init__(self):
        """Initialize the cancer clustering engine"""
        # Cancer-specific clustering parameters
        self.min_cluster_size = 3  # Smaller for cancer cohorts
        self.distance_threshold = 0.6
        self.max_clusters = 15
        
        # Cancer mutation importance weights
        self.significance_weights = {
            ClinicalSignificance.PATHOGENIC: 1.0,
            ClinicalSignificance.LIKELY_PATHOGENIC: 0.8,
            ClinicalSignificance.UNCERTAIN: 0.5,
            ClinicalSignificance.LIKELY_BENIGN: 0.2,
            ClinicalSignificance.BENIGN: 0.1
        }
        
        # Cancer gene importance (oncogenes and tumor suppressors)
        self.gene_weights = {
            # Tumor suppressors (high weight)
            'TP53': 1.0, 'RB1': 1.0, 'APC': 1.0, 'BRCA1': 1.0, 'BRCA2': 1.0,
            'VHL': 0.9, 'PTEN': 0.9, 'CDKN2A': 0.9,
            
            # Oncogenes (high weight)
            'KRAS': 1.0, 'EGFR': 1.0, 'MYC': 1.0, 'PIK3CA': 0.9,
            'BRAF': 0.9, 'ERBB2': 0.9, 'ALK': 0.8,
            
            # DNA repair genes
            'MLH1': 0.9, 'MSH2': 0.9, 'MSH6': 0.8, 'PMS2': 0.8,
            
            # Druggable targets
            'ERBB2': 0.9, 'KIT': 0.8, 'PDGFRA': 0.8, 'FLT3': 0.8
        }
        
        # Cancer pathway groupings
        self.pathway_groups = {
            'DNA_REPAIR': ['BRCA1', 'BRCA2', 'ATM', 'CHEK2', 'PALB2', 'RAD51C', 'RAD51D'],
            'P53_PATHWAY': ['TP53', 'MDM2', 'MDM4', 'CDKN2A', 'RB1'],
            'PI3K_AKT': ['PIK3CA', 'PTEN', 'AKT1', 'TSC1', 'TSC2'],
            'RAS_RAF': ['KRAS', 'NRAS', 'BRAF', 'NF1'],
            'RTK_SIGNALING': ['EGFR', 'ERBB2', 'MET', 'ALK', 'ROS1'],
            'WNT_SIGNALING': ['APC', 'CTNNB1', 'AXIN1', 'AXIN2'],
            'MISMATCH_REPAIR': ['MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM']
        }
        
        logger.info("Cancer mutation clustering engine initialized")

    def cluster_mutations(self, mutations: List[MutationAnalysis]) -> Tuple[List[MutationCluster], Dict[str, Any]]:
        """Main clustering function for cancer mutations"""
        logger.info(f"Starting cancer mutation clustering analysis for {len(mutations)} mutations...")
        
        if len(mutations) < self.min_cluster_size:
            logger.warning(f"Insufficient mutations for clustering: {len(mutations)} < {self.min_cluster_size}")
            return [], {'total_clusters': 0, 'cluster_details': {}}
        
        # Perform Ward linkage clustering
        flat_clusters = self.ward_linkage_clustering(mutations)
        
        # Convert to MutationCluster objects
        cluster_objects = self.create_cluster_objects(flat_clusters, mutations)
        
        # Perform advanced cluster analysis
        cluster_analysis = self.analyze_cancer_clusters(cluster_objects, mutations)
        
        logger.info(f"Cancer clustering complete: {len(cluster_objects)} clusters identified")
        return cluster_objects, cluster_analysis

    def ward_linkage_clustering(self, mutations: List[MutationAnalysis]) -> List[List[int]]:
        """Implement Ward linkage hierarchical clustering for cancer mutations"""
        logger.info("Implementing Ward linkage clustering for cancer mutations...")
        
        n = len(mutations)
        if n < 2:
            return []
        
        # Calculate cancer-specific distance matrix
        distance_matrix = self.calculate_cancer_distance_matrix(mutations)
        
        # Initialize clusters
        clusters = [[i] for i in range(n)]
        
        # Ward linkage algorithm
        while len(clusters) > 1:
            min_distance = float('inf')
            merge_i, merge_j = -1, -1
            
            # Find closest pair using Ward criterion
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    ward_dist = self.calculate_ward_distance(
                        clusters[i], clusters[j], distance_matrix, mutations
                    )
                    
                    if ward_dist < min_distance:
                        min_distance = ward_dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            merged_cluster = clusters[merge_i] + clusters[merge_j]
            clusters = [c for idx, c in enumerate(clusters) if idx not in [merge_i, merge_j]]
            clusters.append(merged_cluster)
            
            # Stop if we have enough large clusters
            if len(clusters) <= self.max_clusters:
                large_clusters = [c for c in clusters if len(c) >= self.min_cluster_size]
                if len(large_clusters) >= 2:
                    break
        
        # Return only clusters that meet minimum size
        return [c for c in clusters if len(c) >= self.min_cluster_size]

    def calculate_cancer_distance_matrix(self, mutations: List[MutationAnalysis]) -> Matrix:
        """Calculate cancer-specific distance matrix"""
        n = len(mutations)
        distance_matrix = Matrix(n)
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.calculate_cancer_mutation_distance(mutations[i], mutations[j])
                distance_matrix.set(i, j, distance)
        
        return distance_matrix

    def calculate_cancer_mutation_distance(self, mut1: MutationAnalysis, mut2: MutationAnalysis) -> float:
        """Calculate distance between cancer mutations"""
        
        # 1. Gene similarity (same gene = closer)
        gene_sim = 1.0 if mut1.gene == mut2.gene else 0.0
        
        # 2. Pathway similarity
        pathway_sim = self.calculate_pathway_similarity(mut1.gene, mut2.gene)
        
        # 3. Pathogenicity similarity
        path_diff = abs(mut1.pathogenicity_score - mut2.pathogenicity_score)
        path_sim = 1.0 - path_diff
        
        # 4. Clinical significance similarity
        sig_sim = self.calculate_significance_similarity(
            mut1.clinical_significance, mut2.clinical_significance
        )
        
        # 5. Cancer type overlap
        types1 = set(mut1.cancer_types)
        types2 = set(mut2.cancer_types)
        type_sim = self.jaccard_similarity(types1, types2)
        
        # 6. Therapy overlap (actionable mutations cluster together)
        therapies1 = set(mut1.targeted_therapies)
        therapies2 = set(mut2.targeted_therapies)
        therapy_sim = self.jaccard_similarity(therapies1, therapies2)
        
        # 7. Gene importance weighting
        gene1_weight = self.gene_weights.get(mut1.gene, 0.5)
        gene2_weight = self.gene_weights.get(mut2.gene, 0.5)
        importance_factor = (gene1_weight + gene2_weight) / 2.0
        
        # Weighted combination (cancer-optimized)
        similarity = (
            gene_sim * 0.25 +           # Same gene is very important
            pathway_sim * 0.20 +        # Pathway membership
            path_sim * 0.20 +           # Pathogenicity similarity
            sig_sim * 0.15 +            # Clinical significance
            type_sim * 0.15 +           # Cancer type overlap
            therapy_sim * 0.05          # Therapeutic similarity
        )
        
        # Boost similarity for important genes
        similarity = similarity * (0.7 + 0.3 * importance_factor)
        
        # Convert similarity to distance
        distance = 1.0 - similarity
        
        return max(0.0, min(1.0, distance))

    def calculate_pathway_similarity(self, gene1: str, gene2: str) -> float:
        """Calculate pathway membership similarity"""
        if gene1 == gene2:
            return 1.0
        
        # Check if genes are in the same pathway
        for pathway, genes in self.pathway_groups.items():
            if gene1 in genes and gene2 in genes:
                return 0.8  # High similarity for same pathway
        
        return 0.0

    def calculate_significance_similarity(self, sig1: ClinicalSignificance, sig2: ClinicalSignificance) -> float:
        """Calculate clinical significance similarity"""
        if sig1 == sig2:
            return 1.0
        
        # Define significance ordering
        sig_order = {
            ClinicalSignificance.PATHOGENIC: 5,
            ClinicalSignificance.LIKELY_PATHOGENIC: 4,
            ClinicalSignificance.UNCERTAIN: 3,
            ClinicalSignificance.LIKELY_BENIGN: 2,
            ClinicalSignificance.BENIGN: 1
        }
        
        order1 = sig_order.get(sig1, 3)
        order2 = sig_order.get(sig2, 3)
        
        # Calculate similarity based on distance in ordering
        max_diff = 4  # Maximum possible difference
        diff = abs(order1 - order2)
        
        return 1.0 - (diff / max_diff)

    def jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def calculate_ward_distance(self, cluster1: List[int], cluster2: List[int], 
                               distance_matrix: Matrix, mutations: List[MutationAnalysis]) -> float:
        """Calculate Ward distance between two clusters"""
        total_distance = 0.0
        comparisons = 0
        
        for i in cluster1:
            for j in cluster2:
                total_distance += distance_matrix.get(i, j)
                comparisons += 1
        
        if comparisons == 0:
            return float('inf')
        
        # Ward formula with cancer-specific adjustments
        avg_distance = total_distance / comparisons
        size_factor = (len(cluster1) * len(cluster2)) / (len(cluster1) + len(cluster2))
        
        # Adjust for clinical importance
        cluster1_importance = sum(self.gene_weights.get(mutations[i].gene, 0.5) for i in cluster1) / len(cluster1)
        cluster2_importance = sum(self.gene_weights.get(mutations[i].gene, 0.5) for i in cluster2) / len(cluster2)
        importance_factor = (cluster1_importance + cluster2_importance) / 2.0
        
        return avg_distance * size_factor * (0.8 + 0.2 * importance_factor)

    def create_cluster_objects(self, flat_clusters: List[List[int]], mutations: List[MutationAnalysis]) -> List[MutationCluster]:
        """Convert cluster indices to MutationCluster objects"""
        cluster_objects = []
        
        for cluster_id, indices in enumerate(flat_clusters):
            cluster_mutations = [mutations[i] for i in indices]
            
            # Calculate cluster characteristics
            avg_pathogenicity = sum(m.pathogenicity_score for m in cluster_mutations) / len(cluster_mutations)
            
            # Determine dominant cancer types
            all_cancer_types = []
            for m in cluster_mutations:
                all_cancer_types.extend(m.cancer_types)
            cancer_counter = Counter(all_cancer_types)
            dominant_types = [ct for ct, _ in cancer_counter.most_common(3)]
            
            # Count actionable mutations
            actionable_count = sum(1 for m in cluster_mutations if m.targeted_therapies)
            
            # Determine dominant clinical significance
            sig_counter = Counter(m.clinical_significance for m in cluster_mutations)
            dominant_significance = sig_counter.most_common(1)[0][0].value
            
            # Calculate cluster risk score
            risk_score = self.calculate_cluster_risk_score(cluster_mutations)
            
            cluster = MutationCluster(
                cluster_id=cluster_id,
                mutation_indices=indices,
                centroid_pathogenicity=avg_pathogenicity,
                dominant_cancer_types=dominant_types,
                actionable_count=actionable_count,
                clinical_significance=dominant_significance,
                risk_score=risk_score,
                size=len(indices)
            )
            
            cluster_objects.append(cluster)
        
        return cluster_objects

    def calculate_cluster_risk_score(self, cluster_mutations: List[MutationAnalysis]) -> float:
        """Calculate risk score for a cluster"""
        if not cluster_mutations:
            return 0.0
        
        # Component scores
        avg_pathogenicity = sum(m.pathogenicity_score for m in cluster_mutations) / len(cluster_mutations)
        actionable_ratio = sum(1 for m in cluster_mutations if m.targeted_therapies) / len(cluster_mutations)
        
        # Clinical significance weighting
        sig_scores = [self.significance_weights.get(m.clinical_significance, 0.5) for m in cluster_mutations]
        avg_sig_score = sum(sig_scores) / len(sig_scores)
        
        # Gene importance
        gene_scores = [self.gene_weights.get(m.gene, 0.5) for m in cluster_mutations]
        avg_gene_score = sum(gene_scores) / len(gene_scores)
        
        # Size factor (larger clusters may be more significant)
        size_factor = min(1.0, len(cluster_mutations) / 10.0)
        
        # Combined risk score
        risk_score = (
            avg_pathogenicity * 0.4 +
            avg_sig_score * 0.3 +
            avg_gene_score * 0.2 +
            size_factor * 0.1
        )
        
        # Slight reduction if many targeted therapies available
        if actionable_ratio > 0.5:
            risk_score *= 0.9
        
        return min(1.0, max(0.0, risk_score))

    def analyze_cancer_clusters(self, clusters: List[MutationCluster], mutations: List[MutationAnalysis]) -> Dict[str, Any]:
        """Perform advanced analysis of cancer clusters"""
        logger.info("Performing advanced cancer cluster analysis...")
        
        analysis = {
            'total_clusters': len(clusters),
            'cluster_details': {},
            'high_risk_clusters': [],
            'pathway_insights': {},
            'therapeutic_opportunities': []
        }
        
        for cluster in clusters:
            cluster_mutations = [mutations[i] for i in cluster.mutation_indices]
            
            # Pathway analysis
            pathways_involved = self.identify_pathways_in_cluster(cluster_mutations)
            
            # Therapeutic analysis
            therapy_analysis = self.analyze_cluster_therapies(cluster_mutations)
            
            # Clinical interpretation
            clinical_interpretation = self.generate_cluster_interpretation(cluster, cluster_mutations)
            
            cluster_detail = {
                'cluster_id': cluster.cluster_id,
                'size': cluster.size,
                'risk_score': round(cluster.risk_score, 3),
                'clinical_significance': cluster.clinical_significance,
                'dominant_cancer_types': cluster.dominant_cancer_types,
                'actionable_mutations': cluster.actionable_count,
                'pathways_involved': pathways_involved,
                'therapeutic_analysis': therapy_analysis,
                'clinical_interpretation': clinical_interpretation,
                'mutation_ids': [mutations[i].mutation_id for i in cluster.mutation_indices]
            }
            
            analysis['cluster_details'][str(cluster.cluster_id)] = cluster_detail
            
            # Flag high-risk clusters
            if cluster.risk_score > 0.7:
                analysis['high_risk_clusters'].append(cluster_detail)
            
            # Add therapeutic opportunities
            if cluster.actionable_count > 0:
                analysis['therapeutic_opportunities'].append({
                    'cluster_id': cluster.cluster_id,
                    'actionable_count': cluster.actionable_count,
                    'therapies': therapy_analysis['available_therapies'],
                    'priority': 'HIGH' if cluster.risk_score > 0.8 else 'MEDIUM'
                })
        
        # Overall pathway insights
        analysis['pathway_insights'] = self.generate_pathway_insights(clusters, mutations)
        
        return analysis

    def identify_pathways_in_cluster(self, cluster_mutations: List[MutationAnalysis]) -> List[str]:
        """Identify cancer pathways represented in cluster"""
        pathways = set()
        
        for mutation in cluster_mutations:
            for pathway, genes in self.pathway_groups.items():
                if mutation.gene in genes:
                    pathways.add(pathway)
        
        return list(pathways)

    def analyze_cluster_therapies(self, cluster_mutations: List[MutationAnalysis]) -> Dict[str, Any]:
        """Analyze therapeutic opportunities in cluster"""
        all_therapies = set()
        therapy_counts = Counter()
        
        for mutation in cluster_mutations:
            all_therapies.update(mutation.targeted_therapies)
            for therapy in mutation.targeted_therapies:
                therapy_counts[therapy] += 1
        
        return {
            'available_therapies': list(all_therapies),
            'therapy_frequency': dict(therapy_counts.most_common()),
            'multi_target_therapies': [t for t, c in therapy_counts.items() if c > 1]
        }

    def generate_cluster_interpretation(self, cluster: MutationCluster, cluster_mutations: List[MutationAnalysis]) -> str:
        """Generate clinical interpretation for cluster"""
        if cluster.size == 1:
            return "Single mutation - individual analysis recommended"
        
        # Pathway-based interpretation
        pathways = self.identify_pathways_in_cluster(cluster_mutations)
        
        if 'DNA_REPAIR' in pathways:
            interpretation = f"DNA repair pathway cluster (n={cluster.size}) - consider PARP inhibitors and hereditary cancer screening"
        elif 'P53_PATHWAY' in pathways:
            interpretation = f"P53 pathway disruption cluster (n={cluster.size}) - aggressive disease pattern, consider experimental therapies"
        elif 'RTK_SIGNALING' in pathways:
            interpretation = f"Receptor tyrosine kinase cluster (n={cluster.size}) - multiple targeted therapy options available"
        elif 'RAS_RAF' in pathways:
            interpretation = f"RAS/RAF pathway cluster (n={cluster.size}) - consider MEK/ERK inhibitors"
        else:
            interpretation = f"Mixed mutation cluster (n={cluster.size}) - individualized therapy approach recommended"
        
        # Add risk context
        if cluster.risk_score > 0.8:
            interpretation += " [HIGH PRIORITY]"
        elif cluster.actionable_count > cluster.size * 0.5:
            interpretation += " [ACTIONABLE]"
        
        return interpretation

    def generate_pathway_insights(self, clusters: List[MutationCluster], mutations: List[MutationAnalysis]) -> Dict[str, Any]:
        """Generate overall pathway insights across all clusters"""
        pathway_distribution = Counter()
        pathway_risk_scores = defaultdict(list)
        
        for cluster in clusters:
            cluster_mutations = [mutations[i] for i in cluster.mutation_indices]
            pathways = self.identify_pathways_in_cluster(cluster_mutations)
            
            for pathway in pathways:
                pathway_distribution[pathway] += cluster.size
                pathway_risk_scores[pathway].append(cluster.risk_score)
        
        # Calculate average risk per pathway
        pathway_avg_risk = {
            pathway: sum(scores) / len(scores) 
            for pathway, scores in pathway_risk_scores.items()
        }
        
        return {
            'pathway_distribution': dict(pathway_distribution.most_common()),
            'pathway_risk_scores': pathway_avg_risk,
            'dominant_pathways': [p for p, _ in pathway_distribution.most_common(3)]
        }