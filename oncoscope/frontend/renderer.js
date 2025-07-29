/**
 * OncoScope Frontend Application
 */

class OncoScopeApp {
    constructor() {
        this.apiBase = 'http://localhost:8000';
        this.currentAnalysis = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.checkSystemStatus();
        this.setupMenuHandlers();
    }
    
    initializeElements() {
        // Get DOM elements
        this.elements = {
            // Sections
            uploadSection: document.getElementById('upload-section'),
            progressSection: document.getElementById('progress-section'),
            resultsSection: document.getElementById('results-section'),
            
            // Upload elements
            uploadArea: document.getElementById('upload-area'),
            fileInput: document.getElementById('file-input'),
            mutationInput: document.getElementById('mutation-input'),
            analyzeButton: document.getElementById('analyze-button'),
            
            // Progress elements
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            steps: {
                parsing: document.getElementById('step-1'),
                database: document.getElementById('step-2'),
                ai: document.getElementById('step-3'),
                report: document.getElementById('step-4')
            },
            
            // Results elements
            overallRiskScore: document.getElementById('overall-risk-score'),
            riskClassification: document.getElementById('risk-classification'),
            riskCircle: document.getElementById('risk-circle'),
            confidenceScore: document.getElementById('confidence-score'),
            actionableCount: document.getElementById('actionable-count'),
            pathogenicCount: document.getElementById('pathogenic-count'),
            knownMutations: document.getElementById('known-mutations'),
            tumorList: document.getElementById('tumor-list'),
            mutationsResults: document.getElementById('mutations-results'),
            clinicalRecommendations: document.getElementById('clinical-recommendations'),
            warningsSection: document.getElementById('warnings-section'),
            warningsList: document.getElementById('warnings-list'),
            
            // Status
            connectionStatus: document.getElementById('connection-status'),
            statusDot: document.querySelector('.status-dot'),
            statusText: document.querySelector('.status-text'),
            
            // Modal
            errorModal: document.getElementById('error-modal'),
            errorMessage: document.getElementById('error-message')
        };
    }
    
    attachEventListeners() {
        // File upload
        this.elements.uploadArea.addEventListener('click', () => {
            this.elements.fileInput.click();
        });
        
        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files[0]);
        });
        
        // Drag and drop
        this.elements.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.add('dragover');
        });
        
        this.elements.uploadArea.addEventListener('dragleave', () => {
            this.elements.uploadArea.classList.remove('dragover');
        });
        
        this.elements.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.elements.uploadArea.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files[0]);
        });
        
        // Manual input
        this.elements.mutationInput.addEventListener('input', (e) => {
            const hasInput = e.target.value.trim().length > 0;
            this.elements.analyzeButton.disabled = !hasInput;
        });
        
        this.elements.analyzeButton.addEventListener('click', () => {
            this.analyzeMutations();
        });
        
        // Demo buttons
        document.querySelectorAll('.demo-button').forEach(button => {
            button.addEventListener('click', (e) => {
                this.loadDemoData(e.target.dataset.demo);
            });
        });
        
        // Export buttons
        document.getElementById('export-pdf').addEventListener('click', () => {
            this.exportToPDF();
        });
        
        document.getElementById('export-json').addEventListener('click', () => {
            this.exportToJSON();
        });
        
        document.getElementById('save-results').addEventListener('click', () => {
            this.saveResults();
        });
        
        // New analysis button
        document.getElementById('new-analysis-button').addEventListener('click', () => {
            this.resetAnalysis();
        });
    }
    
    setupMenuHandlers() {
        // Handle menu actions from main process
        if (window.api) {
            window.api.onMenuAction((event, action) => {
                switch (action) {
                    case 'menu-open-file':
                        this.elements.fileInput.click();
                        break;
                    case 'menu-save-results':
                        this.saveResults();
                        break;
                    case 'menu-export-pdf':
                        this.exportToPDF();
                        break;
                }
            });
        }
    }
    
    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const status = await response.json();
            
            this.updateConnectionStatus(status);
            
        } catch (error) {
            console.error('System status check failed:', error);
            this.updateConnectionStatus({ status: 'error', message: 'Connection failed' });
        }
    }
    
    updateConnectionStatus(status) {
        if (status.status === 'healthy' && status.ollama_connected && status.model_loaded) {
            this.elements.statusDot.classList.add('connected');
            this.elements.statusText.textContent = 'Ready for Analysis';
        } else if (status.status === 'healthy') {
            this.elements.statusText.textContent = 'System Ready (AI Loading...)';
        } else {
            this.elements.statusText.textContent = 'System Error';
            this.showError('Backend connection failed. Please ensure the OncoScope backend is running.');
        }
    }
    
    handleFileUpload(file) {
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const content = e.target.result;
            const mutations = this.parseFileContent(content, file.name);
            
            if (mutations.length > 0) {
                this.elements.mutationInput.value = mutations.join('\n');
                this.elements.analyzeButton.disabled = false;
            } else {
                this.showError('No valid mutations found in file. Please check the format.');
            }
        };
        
        reader.readAsText(file);
    }
    
    parseFileContent(content, filename) {
        const mutations = [];
        const lines = content.split('\n');
        
        for (const line of lines) {
            const trimmed = line.trim();
            if (trimmed && !trimmed.startsWith('#')) {
                // Simple parsing - look for gene:variant patterns
                const mutationMatch = trimmed.match(/([A-Z0-9]+)[:_\s]+([cp]\.[A-Z0-9>_]+)/i);
                if (mutationMatch) {
                    mutations.push(`${mutationMatch[1]}:${mutationMatch[2]}`);
                }
            }
        }
        
        return mutations;
    }
    
    loadDemoData(demoType) {
        const demoMutations = {
            high_risk: [
                'TP53:c.524G>A',
                'KRAS:c.35G>A',
                'EGFR:c.2369C>T',
                'PIK3CA:c.3140A>G',
                'BRAF:c.1799T>A'
            ],
            targetable: [
                'EGFR:c.2573T>G',
                'ALK:c.3522C>A',
                'BRCA1:c.68_69delAG',
                'PIK3CA:c.1633G>A'
            ],
            low_risk: [
                'TP53:c.215C>G',
                'EGFR:c.2361G>A',
                'MLH1:c.655A>G'
            ]
        };
        
        const mutations = demoMutations[demoType] || [];
        this.elements.mutationInput.value = mutations.join('\n');
        this.elements.analyzeButton.disabled = false;
    }
    
    async analyzeMutations() {
        const mutationText = this.elements.mutationInput.value.trim();
        if (!mutationText) return;
        
        const mutations = mutationText.split('\n')
            .map(m => m.trim())
            .filter(m => m.length > 0);
        
        // Show progress
        this.showSection('progress');
        this.startProgress();
        
        try {
            const response = await fetch(`${this.apiBase}/analyze/mutations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ mutations })
            });
            
            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }
            
            const result = await response.json();
            this.currentAnalysis = result;
            
            // Complete progress
            this.updateProgress(100, 'Analysis complete!');
            this.activateStep('report');
            
            // Show results after brief delay
            setTimeout(() => {
                this.displayResults(result.analysis);
                this.showSection('results');
            }, 1000);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
            this.showSection('upload');
        }
    }
    
    startProgress() {
        let progress = 0;
        const steps = ['parsing', 'database', 'ai', 'report'];
        let currentStep = 0;
        
        // Simulate progress
        const interval = setInterval(() => {
            progress += Math.random() * 15 + 5;
            
            if (progress >= 25 && currentStep === 0) {
                this.activateStep(steps[0]);
                this.updateProgress(25, 'Parsing mutations...');
                currentStep = 1;
            } else if (progress >= 50 && currentStep === 1) {
                this.activateStep(steps[1]);
                this.updateProgress(50, 'Searching mutation database...');
                currentStep = 2;
            } else if (progress >= 75 && currentStep === 2) {
                this.activateStep(steps[2]);
                this.updateProgress(75, 'AI analysis in progress...');
                currentStep = 3;
            }
            
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 500);
    }
    
    updateProgress(percent, message) {
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressText.textContent = message;
    }
    
    activateStep(stepName) {
        if (this.elements.steps[stepName]) {
            this.elements.steps[stepName].classList.add('active');
        }
    }
    
    displayResults(analysis) {
        // Update overall risk score with animation
        const riskScore = analysis.overall_risk_score;
        this.animateRiskScore(riskScore);
        
        // Update risk classification
        this.elements.riskClassification.textContent = analysis.risk_classification;
        this.updateRiskColor(riskScore);
        
        // Update metrics
        this.elements.confidenceScore.textContent = 
            `${(analysis.confidence_metrics.overall_confidence * 100).toFixed(0)}%`;
        this.elements.actionableCount.textContent = analysis.actionable_mutations.length;
        
        // Count pathogenic mutations
        const pathogenicCount = analysis.individual_mutations.filter(m => 
            m.clinical_significance === 'PATHOGENIC' || 
            m.clinical_significance === 'LIKELY_PATHOGENIC'
        ).length;
        this.elements.pathogenicCount.textContent = pathogenicCount;
        
        this.elements.knownMutations.textContent = 
            `${analysis.confidence_metrics.known_mutations}/${analysis.individual_mutations.length}`;
        
        // Display tumor predictions
        this.displayTumorPredictions(analysis.estimated_tumor_types);
        
        // Display individual mutations
        this.displayMutationResults(analysis.individual_mutations);
        
        // Display recommendations
        this.displayRecommendations(analysis.clinical_recommendations);
        
        // Display clustering analysis if available
        if (analysis.clustering_analysis && analysis.clustering_analysis.clusters_identified > 0) {
            this.displayClusteringResults(analysis.clustering_analysis);
        }
        
        // Display warnings if any
        if (analysis.warnings && analysis.warnings.length > 0) {
            this.displayWarnings(analysis.warnings);
        }
    }
    
    animateRiskScore(score) {
        // Animate the number
        let current = 0;
        const increment = score / 50;
        const timer = setInterval(() => {
            current += increment;
            if (current >= score) {
                current = score;
                clearInterval(timer);
            }
            this.elements.overallRiskScore.textContent = current.toFixed(2);
        }, 20);
        
        // Animate the circle
        const circumference = 2 * Math.PI * 70;
        const offset = circumference - (score * circumference);
        this.elements.riskCircle.style.strokeDashoffset = offset;
    }
    
    updateRiskColor(score) {
        let color;
        if (score >= 0.7) {
            color = 'var(--danger-color)';
        } else if (score >= 0.5) {
            color = 'var(--warning-color)';
        } else {
            color = 'var(--success-color)';
        }
        
        this.elements.riskCircle.style.stroke = color;
        this.elements.overallRiskScore.style.color = color;
    }
    
    displayTumorPredictions(predictions) {
        this.elements.tumorList.innerHTML = '';
        
        predictions.slice(0, 5).forEach(prediction => {
            const tumorItem = document.createElement('div');
            tumorItem.className = 'tumor-item';
            
            const likelihood = (prediction.likelihood * 100).toFixed(1);
            
            tumorItem.innerHTML = `
                <span class="tumor-name">${this.formatCancerType(prediction.cancer_type)}</span>
                <div class="tumor-likelihood">
                    <div class="likelihood-bar">
                        <div class="likelihood-fill" style="width: ${likelihood}%"></div>
                    </div>
                    <span class="likelihood-text">${likelihood}%</span>
                </div>
            `;
            
            this.elements.tumorList.appendChild(tumorItem);
        });
    }
    
    formatCancerType(type) {
        // Format cancer type names for display
        return type.replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    displayMutationResults(mutations) {
        this.elements.mutationsResults.innerHTML = '';
        
        mutations.forEach(mutation => {
            const mutationCard = document.createElement('div');
            mutationCard.className = 'mutation-card';
            
            const pathogenicityClass = this.getPathogenicityClass(mutation.clinical_significance);
            
            let therapiesHtml = '';
            if (mutation.targeted_therapies && mutation.targeted_therapies.length > 0) {
                therapiesHtml = `
                    <div class="detail-row">
                        <span class="detail-label">Targeted Therapies:</span>
                        <span class="detail-value therapy-list">${mutation.targeted_therapies.join(', ')}</span>
                    </div>
                `;
            }
            
            mutationCard.innerHTML = `
                <div class="mutation-header">
                    <h4>${mutation.mutation_id}</h4>
                    <span class="pathogenicity-badge ${pathogenicityClass}">
                        ${mutation.clinical_significance.replace(/_/g, ' ')}
                    </span>
                </div>
                
                <div class="mutation-details">
                    <div class="detail-row">
                        <span class="detail-label">Protein Change:</span>
                        <span class="detail-value">${mutation.protein_change || 'Unknown'}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Pathogenicity Score:</span>
                        <span class="detail-value">${mutation.pathogenicity_score.toFixed(2)}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Associated Cancers:</span>
                        <span class="detail-value">${mutation.cancer_types.map(t => this.formatCancerType(t)).join(', ')}</span>
                    </div>
                    
                    ${therapiesHtml}
                    
                    <div class="detail-row">
                        <span class="detail-label">Mechanism:</span>
                        <span class="detail-value">${mutation.mechanism}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Prognosis Impact:</span>
                        <span class="detail-value">${mutation.prognosis_impact.replace(/_/g, ' ')}</span>
                    </div>
                    
                    <div class="detail-row">
                        <span class="detail-label">Confidence:</span>
                        <span class="detail-value">${(mutation.confidence_score * 100).toFixed(0)}%</span>
                    </div>
                </div>
            `;
            
            this.elements.mutationsResults.appendChild(mutationCard);
        });
    }
    
    getPathogenicityClass(significance) {
        const classMap = {
            'PATHOGENIC': 'pathogenic',
            'LIKELY_PATHOGENIC': 'likely-pathogenic',
            'VARIANT_OF_UNCERTAIN_SIGNIFICANCE': 'uncertain',
            'LIKELY_BENIGN': 'likely-benign',
            'BENIGN': 'benign'
        };
        return classMap[significance] || 'uncertain';
    }
    
    displayRecommendations(recommendations) {
        this.elements.clinicalRecommendations.innerHTML = '';
        
        recommendations.forEach(rec => {
            const recElement = document.createElement('div');
            recElement.className = 'recommendation-item';
            recElement.innerHTML = `
                <div class="recommendation-icon">=�</div>
                <div class="recommendation-text">${rec}</div>
            `;
            this.elements.clinicalRecommendations.appendChild(recElement);
        });
    }
    
    displayWarnings(warnings) {
        this.elements.warningsSection.classList.remove('hidden');
        this.elements.warningsList.innerHTML = '';
        
        warnings.forEach(warning => {
            const warningElement = document.createElement('div');
            warningElement.className = 'warning-item';
            warningElement.innerHTML = `
                <span>�</span>
                <span>${warning}</span>
            `;
            this.elements.warningsList.appendChild(warningElement);
        });
    }
    
    showSection(section) {
        // Hide all sections
        this.elements.uploadSection.classList.add('hidden');
        this.elements.progressSection.classList.add('hidden');
        this.elements.resultsSection.classList.add('hidden');
        
        // Reset progress steps
        Object.values(this.elements.steps).forEach(step => {
            step.classList.remove('active');
        });
        
        // Show requested section
        switch (section) {
            case 'upload':
                this.elements.uploadSection.classList.remove('hidden');
                break;
            case 'progress':
                this.elements.progressSection.classList.remove('hidden');
                this.updateProgress(0, 'Initializing analysis...');
                break;
            case 'results':
                this.elements.resultsSection.classList.remove('hidden');
                break;
        }
    }
    
    resetAnalysis() {
        this.currentAnalysis = null;
        this.elements.mutationInput.value = '';
        this.elements.analyzeButton.disabled = true;
        this.elements.warningsSection.classList.add('hidden');
        this.showSection('upload');
    }
    
    showError(message) {
        this.elements.errorMessage.textContent = message;
        this.elements.errorModal.classList.remove('hidden');
    }
    
    displayClusteringResults(clusteringAnalysis) {
        // Create clustering section if it doesn't exist
        let clusteringSection = document.getElementById('clustering-section');
        if (!clusteringSection) {
            clusteringSection = document.createElement('div');
            clusteringSection.id = 'clustering-section';
            clusteringSection.className = 'results-section clustering-results';
            clusteringSection.innerHTML = `
                <h3>🧬 Mutation Clustering Analysis</h3>
                <div id="clustering-overview" class="clustering-overview"></div>
                <div id="clustering-insights" class="clustering-insights"></div>
                <div id="cluster-details" class="cluster-details"></div>
            `;
            
            // Insert before recommendations section
            const recommendationsSection = document.querySelector('.recommendations-section');
            if (recommendationsSection) {
                recommendationsSection.parentNode.insertBefore(clusteringSection, recommendationsSection);
            } else {
                document.getElementById('results-section').appendChild(clusteringSection);
            }
        }
        
        // Display clustering overview
        this.displayClusteringOverview(clusteringAnalysis);
        
        // Display clustering insights
        this.displayClusteringInsights(clusteringAnalysis);
        
        // Display detailed cluster information
        this.displayClusterDetails(clusteringAnalysis.cluster_analysis);
    }
    
    displayClusteringOverview(clusteringAnalysis) {
        const overviewContainer = document.getElementById('clustering-overview');
        const clustersFound = clusteringAnalysis.clusters_identified;
        
        overviewContainer.innerHTML = `
            <div class="clustering-stats">
                <div class="stat-item">
                    <div class="stat-number">${clustersFound}</div>
                    <div class="stat-label">Clusters Identified</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${clusteringAnalysis.cluster_analysis?.high_risk_clusters?.length || 0}</div>
                    <div class="stat-label">High-Risk Clusters</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">${clusteringAnalysis.cluster_analysis?.therapeutic_opportunities?.length || 0}</div>
                    <div class="stat-label">Therapeutic Opportunities</div>
                </div>
            </div>
        `;
    }
    
    displayClusteringInsights(clusteringAnalysis) {
        const insightsContainer = document.getElementById('clustering-insights');
        const insights = clusteringAnalysis.clustering_insights || [];
        
        if (insights.length === 0) {
            insightsContainer.innerHTML = '<p class="no-insights">No specific clustering insights available.</p>';
            return;
        }
        
        const insightsHTML = insights.map(insight => `
            <div class="insight-item">
                <span class="insight-icon">💡</span>
                <span class="insight-text">${insight}</span>
            </div>
        `).join('');
        
        insightsContainer.innerHTML = `
            <h4>Key Insights</h4>
            <div class="insights-list">
                ${insightsHTML}
            </div>
        `;
    }
    
    displayClusterDetails(clusterAnalysis) {
        const detailsContainer = document.getElementById('cluster-details');
        
        if (!clusterAnalysis?.cluster_details) {
            detailsContainer.innerHTML = '<p>No detailed cluster information available.</p>';
            return;
        }
        
        const clusters = Object.values(clusterAnalysis.cluster_details);
        
        const clustersHTML = clusters.map(cluster => {
            const riskColor = this.getRiskColor(cluster.risk_score);
            const pathwaysText = cluster.pathways_involved?.length > 0 
                ? cluster.pathways_involved.join(', ') 
                : 'Multiple pathways';
            
            return `
                <div class="cluster-card">
                    <div class="cluster-header">
                        <div class="cluster-title">
                            <span class="cluster-id">Cluster ${cluster.cluster_id + 1}</span>
                            <span class="cluster-size">${cluster.size} mutations</span>
                        </div>
                        <div class="cluster-risk">
                            <span class="risk-score" style="color: ${riskColor}">
                                Risk: ${(cluster.risk_score * 100).toFixed(0)}%
                            </span>
                        </div>
                    </div>
                    
                    <div class="cluster-body">
                        <div class="cluster-info">
                            <div class="info-item">
                                <strong>Pathways:</strong> ${pathwaysText}
                            </div>
                            <div class="info-item">
                                <strong>Clinical Significance:</strong> ${cluster.clinical_significance || 'Mixed'}
                            </div>
                            <div class="info-item">
                                <strong>Cancer Types:</strong> ${cluster.dominant_cancer_types?.join(', ') || 'Various'}
                            </div>
                            <div class="info-item">
                                <strong>Actionable Mutations:</strong> ${cluster.actionable_mutations || 0}
                            </div>
                        </div>
                        
                        ${cluster.clinical_interpretation ? `
                            <div class="cluster-interpretation">
                                <strong>Clinical Interpretation:</strong>
                                <p>${cluster.clinical_interpretation}</p>
                            </div>
                        ` : ''}
                        
                        ${cluster.therapeutic_analysis?.available_therapies?.length > 0 ? `
                            <div class="cluster-therapies">
                                <strong>Available Therapies:</strong>
                                <div class="therapy-tags">
                                    ${cluster.therapeutic_analysis.available_therapies.slice(0, 5).map(therapy => 
                                        `<span class="therapy-tag">${therapy}</span>`
                                    ).join('')}
                                </div>
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
        }).join('');
        
        detailsContainer.innerHTML = `
            <h4>Cluster Details</h4>
            <div class="clusters-container">
                ${clustersHTML}
            </div>
        `;
    }
    
    getRiskColor(riskScore) {
        if (riskScore >= 0.7) return 'var(--danger-color)';
        if (riskScore >= 0.5) return 'var(--warning-color)';
        return 'var(--success-color)';
    }

    exportToPDF() {
        window.print();
    }
    
    async exportToJSON() {
        if (!this.currentAnalysis) return;
        
        const dataStr = JSON.stringify(this.currentAnalysis, null, 2);
        const dataBlob = new Blob([dataStr], {type: 'application/json'});
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `oncoscope-analysis-${new Date().toISOString().split('T')[0]}.json`;
        link.click();
    }
    
    async saveResults() {
        if (!this.currentAnalysis) return;
        
        if (window.api) {
            const result = await window.api.showSaveDialog({
                title: 'Save Analysis Results',
                defaultPath: `oncoscope-analysis-${new Date().toISOString().split('T')[0]}.json`,
                filters: [
                    { name: 'JSON Files', extensions: ['json'] },
                    { name: 'All Files', extensions: ['*'] }
                ]
            });
            
            if (!result.canceled) {
                const fs = require('fs');
                fs.writeFileSync(result.filePath, JSON.stringify(this.currentAnalysis, null, 2));
            }
        } else {
            // Fallback for non-Electron environment
            this.exportToJSON();
        }
    }
}

// Global function for modal
function closeErrorModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new OncoScopeApp();
});