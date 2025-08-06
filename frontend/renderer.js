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
    
    // Utility functions for string formatting
    formatDisplayString(str) {
        /**
         * Format strings for display by replacing underscores with spaces
         * and capitalizing each word
         */
        return str.replace(/_/g, ' ')
            .split(' ')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }
    
    formatUnderscoreString(str) {
        /**
         * Simple replacement of underscores with spaces
         */
        return str.replace(/_/g, ' ');
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
            
            // Patient information elements
            patientAge: document.getElementById('patient-age'),
            patientGender: document.getElementById('patient-gender'),
            cancerType: document.getElementById('cancer-type'),
            cancerTypeOther: document.getElementById('cancer-type-other'),
            
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
        
        // Add keyboard navigation support
        this.setupKeyboardNavigation();
        
        // New analysis button
        document.getElementById('new-analysis-button').addEventListener('click', () => {
            this.resetAnalysis();
        });
        
        // Cancer type dropdown change handler
        this.elements.cancerType.addEventListener('change', (e) => {
            if (e.target.value === 'other') {
                this.elements.cancerTypeOther.classList.remove('hidden');
                this.elements.cancerTypeOther.focus();
            } else {
                this.elements.cancerTypeOther.classList.add('hidden');
                this.elements.cancerTypeOther.value = '';
            }
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
        console.log('Checking system status at:', `${this.apiBase}/health`);
        try {
            const response = await fetch(`${this.apiBase}/health`, {
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
                mode: 'cors',
                signal: AbortSignal.timeout(10000) // 10 second timeout for health check
            });
            
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const status = await response.json();
            console.log('System status:', status);
            
            this.updateConnectionStatus(status);
            
        } catch (error) {
            console.error('System status check failed:', error);
            console.error('Error details:', error.message, error.stack);
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
        const demoData = {
            high_risk: {
                mutations: [
                    'TP53:c.524G>A',
                    'KRAS:c.35G>A',
                    'EGFR:c.2369C>T',
                    'PIK3CA:c.3140A>G',
                    'BRAF:c.1799T>A'
                ],
                patient: {
                    age: 68,
                    gender: 'male',
                    cancerType: 'lung_adenocarcinoma'
                }
            },
            targetable: {
                mutations: [
                    'EGFR:c.2573T>G',
                    'ALK:c.3522C>A',
                    'BRCA1:c.68_69delAG',
                    'PIK3CA:c.1633G>A'
                ],
                patient: {
                    age: 52,
                    gender: 'female',
                    cancerType: 'breast_adenocarcinoma'
                }
            },
            low_risk: {
                mutations: [
                    'TP53:c.215C>G',
                    'EGFR:c.2361G>A',
                    'MLH1:c.655A>G'
                ],
                patient: {
                    age: 45,
                    gender: 'female',
                    cancerType: 'colorectal_adenocarcinoma'
                }
            }
        };
        
        const demo = demoData[demoType];
        if (demo) {
            // Load mutations
            this.elements.mutationInput.value = demo.mutations.join('\n');
            this.elements.analyzeButton.disabled = false;
            
            // Load patient information
            if (demo.patient) {
                this.elements.patientAge.value = demo.patient.age || '';
                this.elements.patientGender.value = demo.patient.gender || '';
                this.elements.cancerType.value = demo.patient.cancerType || '';
                // Hide the other input if we're setting a predefined cancer type
                if (demo.patient.cancerType && demo.patient.cancerType !== 'other') {
                    this.elements.cancerTypeOther.classList.add('hidden');
                    this.elements.cancerTypeOther.value = '';
                }
            }
        }
    }
    
    async analyzeMutations() {
        const mutationText = this.elements.mutationInput.value.trim();
        if (!mutationText) return;
        
        const mutations = mutationText.split('\n')
            .map(m => m.trim())
            .filter(m => m.length > 0);
        
        // Gather patient context
        const patientContext = {};
        
        // Add patient information if provided
        const age = this.elements.patientAge.value;
        if (age) {
            patientContext.age = parseInt(age);
        }
        
        const gender = this.elements.patientGender.value;
        if (gender) {
            patientContext.sex = gender;
        }
        
        const cancerTypeValue = this.elements.cancerType.value;
        if (cancerTypeValue === 'other') {
            const otherValue = this.elements.cancerTypeOther.value.trim();
            if (otherValue) {
                patientContext.cancer_type = otherValue;
            }
        } else if (cancerTypeValue) {
            patientContext.cancer_type = cancerTypeValue;
        }
        
        // Build request payload
        const requestPayload = {
            mutations: mutations
        };
        
        // Only add patient_context if we have any data
        if (Object.keys(patientContext).length > 0) {
            requestPayload.patient_context = patientContext;
        }
        
        // Show progress
        this.showSection('progress');
        this.startProgress();
        
        let analysisId = null;
        let statusPollInterval = null;
        
        try {
            console.log('Sending analysis request:', requestPayload);
            
            // Start the analysis
            const response = await fetch(`${this.apiBase}/analyze/mutations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestPayload),
                signal: AbortSignal.timeout(600000) // 10 minute timeout for complex analyses
            });
            
            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.statusText}`);
            }
            
            console.log('Analysis response received, parsing...');
            const result = await response.json();
            console.log('Analysis result:', result);
            this.currentAnalysis = result;
            
            // Extract analysis ID if available
            analysisId = result.analysis_id;
            
            // Start polling for status if we have an analysis ID
            if (analysisId) {
                statusPollInterval = setInterval(async () => {
                    try {
                        const statusResponse = await fetch(`${this.apiBase}/analysis/${analysisId}/status`);
                        if (statusResponse.ok) {
                            const status = await statusResponse.json();
                            if (status.progress !== undefined && status.message) {
                                this.updateProgress(status.progress, status.message);
                                
                                // Steps removed - using loading animation instead
                            }
                            
                            // Check if analysis failed
                            if (status.status === 'failed') {
                                clearInterval(statusPollInterval);
                                throw new Error(status.error || 'Analysis failed');
                            }
                        }
                    } catch (err) {
                        console.log('Status poll error:', err);
                    }
                }, 2000); // Poll every 2 seconds to reduce load
            }
            
            // Complete progress
            this.updateProgress(100, 'Analysis complete! Generating clinical recommendations...');
            
            // Clear status polling since analysis is complete
            if (statusPollInterval) {
                clearInterval(statusPollInterval);
                statusPollInterval = null;
            }
            
            // Clear fact rotation interval
            if (this.factInterval) {
                clearInterval(this.factInterval);
                this.factInterval = null;
            }
            
            // Show results after brief delay
            setTimeout(() => {
                console.log('Displaying results...');
                if (result && result.analysis) {
                    this.displayResults(result.analysis);
                    this.showSection('results');
                } else if (result) {
                    // Try using result directly if analysis is not nested
                    console.log('Using result directly:', result);
                    this.displayResults(result);
                    this.showSection('results');
                } else {
                    console.error('No analysis data in result');
                    this.showError('No analysis data received');
                }
            }, 1000);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
            this.showSection('upload');
        } finally {
            // Clean up status polling
            if (statusPollInterval) {
                clearInterval(statusPollInterval);
            }
        }
    }
    
    startProgress() {
        // Initialize progress display
        this.showCancerFacts();
        
        // Clear all step states
        Object.values(this.elements.steps).forEach(step => {
            step.classList.remove('active');
        });
    }
    
    showCancerFacts() {
        const facts = [
            {
                title: "Did you know?",
                text: "Precision oncology has improved 5-year survival rates by up to 30% for certain cancer types through targeted therapies."
            },
            {
                title: "Genomic Insight",
                text: "Over 400 genes have been identified as cancer drivers, with TP53 being mutated in over 50% of all cancers."
            },
            {
                title: "AI in Oncology",
                text: "Machine learning models can now predict drug response with 85% accuracy by analyzing mutation patterns."
            },
            {
                title: "Personalized Medicine",
                text: "Each tumor has an average of 4-5 driver mutations, making personalized treatment strategies essential."
            },
            {
                title: "Clinical Impact",
                text: "Targeted therapies based on genetic mutations have response rates 3x higher than traditional chemotherapy."
            },
            {
                title: "Early Detection",
                text: "Liquid biopsies can detect cancer mutations up to 4 years before traditional diagnostic methods."
            },
            {
                title: "Treatment Evolution",
                text: "Over 200 targeted cancer drugs have been approved since 2000, revolutionizing precision oncology."
            },
            {
                title: "Mutation Analysis",
                text: "OncoScope uses fine-tuned Gemma 3n to analyze complex mutation interactions and provide personalized insights."
            }
        ];
        
        let factIndex = 0;
        const factElement = document.getElementById('cancer-fact');
        
        const showNextFact = () => {
            const fact = facts[factIndex];
            factElement.style.opacity = '0';
            
            setTimeout(() => {
                factElement.innerHTML = `
                    <div class="cancer-fact-title">${fact.title}</div>
                    <div>${fact.text}</div>
                `;
                factElement.style.opacity = '1';
            }, 400);
            
            factIndex = (factIndex + 1) % facts.length;
        };
        
        // Show first fact immediately
        showNextFact();
        
        // Rotate facts every 6 seconds
        this.factInterval = setInterval(showNextFact, 6000);
    }
    
    updateProgress(percent, message) {
        // Progress bar removed - using loading animation instead
        console.log(`Progress: ${percent}% - ${message}`);
    }
    
    activateStep(stepName) {
        if (this.elements.steps[stepName]) {
            this.elements.steps[stepName].classList.add('active');
        }
    }
    
    displayResults(analysis) {
        console.log('displayResults called with:', analysis);
        
        if (!analysis) {
            console.error('No analysis data provided to displayResults');
            this.showError('No analysis data to display');
            return;
        }
        
        // Set timestamp
        const timestamp = new Date().toLocaleString('en-US', {
            month: 'long',
            day: 'numeric',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
        document.getElementById('analysis-timestamp').textContent = timestamp;
        
        // Update overall risk score with animation
        const riskScore = analysis.overall_risk_score || 0;
        console.log('Risk score:', riskScore);
        this.animateRiskScore(riskScore);
        
        // Update risk classification with color coding
        this.elements.riskClassification.textContent = analysis.risk_classification;
        this.elements.riskClassification.className = 'risk-label';
        if (riskScore >= 0.7) {
            this.elements.riskClassification.classList.add('risk-high');
        } else if (riskScore >= 0.5) {
            this.elements.riskClassification.classList.add('risk-medium');
        } else {
            this.elements.riskClassification.classList.add('risk-low');
        }
        this.updateRiskColor(riskScore);
        
        // Show risk alert based on classification
        this.showRiskAlert(analysis);
        
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
        
        // Display multi-mutation analysis if available
        console.log('Checking for multi-mutation analysis:', {
            hasMultiMutationAnalysis: !!analysis.multi_mutation_analysis,
            mutationCount: analysis.individual_mutations?.length,
            multiMutationData: analysis.multi_mutation_analysis
        });
        
        if (analysis.multi_mutation_analysis && analysis.individual_mutations.length > 1) {
            console.log('Displaying multi-mutation analysis section');
            this.displayMultiMutationAnalysis(analysis.multi_mutation_analysis);
        } else if (analysis.individual_mutations && analysis.individual_mutations.length > 1) {
            // Show section with mock data if we have multiple mutations but no proper analysis
            console.log('Showing multi-mutation section with placeholder data');
            console.log('Full analysis object:', analysis);
            this.displayMultiMutationAnalysis({
                mutation_profile: {
                    total_mutations: analysis.individual_mutations.length,
                    pathogenic_count: analysis.individual_mutations.filter(m => 
                        m.clinical_significance === 'PATHOGENIC' || 
                        m.clinical_significance === 'LIKELY_PATHOGENIC'
                    ).length,
                    dominant_pathways: ["Analysis pending..."],
                    interaction_pattern: "pending"
                },
                composite_risk: {
                    overall_pathogenicity: analysis.overall_risk_score || 0.5,
                    risk_modification: "pending",
                    penetrance_estimate: "pending"
                },
                therapeutic_strategy: {
                    precision_medicine_score: 0.5,
                    combination_therapies: []
                },
                comprehensive_interpretation: "Multi-mutation interaction analysis is pending. The AI model is currently analyzing the complex interactions between these mutations."
            });
        } else {
            console.log('Not displaying multi-mutation analysis:', {
                reason: !analysis.multi_mutation_analysis ? 'No multi_mutation_analysis data' : 'Not enough mutations'
            });
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
    
    showRiskAlert(analysis) {
        const alertElement = document.getElementById('risk-summary-alert');
        const mutationCountElement = document.getElementById('mutation-count');
        
        if (!alertElement) return;
        
        const riskScore = analysis.overall_risk_score || 0;
        const riskLevel = analysis.risk_classification || 'LOW';
        
        // Count total mutations analyzed
        const totalMutations = analysis.individual_mutations.length;
        
        // Count pathogenic mutations
        const pathogenicCount = analysis.individual_mutations.filter(m => 
            m.clinical_significance === 'PATHOGENIC' || 
            m.clinical_significance === 'LIKELY_PATHOGENIC'
        ).length;
        
        // Use total mutations if no pathogenic found (for uncertain variants)
        const mutationCount = pathogenicCount > 0 ? pathogenicCount : totalMutations;
        
        // Update mutation count
        mutationCountElement.textContent = mutationCount;
        
        // Remove all risk classes
        alertElement.classList.remove('moderate-risk', 'low-risk', 'hidden');
        
        if (riskLevel === 'HIGH' || riskLevel === 'MEDIUM-HIGH') {
            // Show high risk alert
            alertElement.classList.remove('hidden');
        } else if (riskLevel === 'MEDIUM' || riskLevel === 'LOW-MEDIUM') {
            // Show moderate risk
            alertElement.classList.remove('hidden');
            alertElement.classList.add('moderate-risk');
            alertElement.querySelector('.alert-title').textContent = 'MODERATE RISK PROFILE';
            alertElement.querySelector('.alert-action').textContent = 'Schedule oncology consultation';
        } else {
            // Low risk - hide alert
            alertElement.classList.add('hidden');
        }
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
        return this.formatDisplayString(type);
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
                        ${this.formatUnderscoreString(mutation.clinical_significance)}
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
                        <span class="detail-value">${this.formatUnderscoreString(mutation.prognosis_impact)}</span>
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
                <div class="recommendation-icon">•</div>
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
                this.updateProgress(0, 'Initializing OncoScope cancer genomics analysis...');
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
        
        // Clear patient information
        this.elements.patientAge.value = '';
        this.elements.patientGender.value = '';
        this.elements.cancerType.value = '';
        this.elements.cancerTypeOther.value = '';
        this.elements.cancerTypeOther.classList.add('hidden');
        
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
    
    displayMultiMutationAnalysis(multiMutationData) {
        console.log('Displaying multi-mutation analysis:', multiMutationData);
        
        // Show the multi-mutation section
        const multiMutationSection = document.getElementById('multi-mutation-section');
        multiMutationSection.classList.remove('hidden');
        
        // Display interaction pattern
        const interactionPattern = multiMutationData.mutation_profile?.interaction_pattern || 'Unknown';
        document.getElementById('interaction-pattern').textContent = 
            this.formatDisplayString(interactionPattern);
        
        // Display composite risk
        const compositeRisk = multiMutationData.composite_risk?.overall_pathogenicity || 0;
        const riskPercentage = (compositeRisk * 100).toFixed(0);
        document.getElementById('composite-risk').textContent = `${riskPercentage}%`;
        document.getElementById('composite-risk').style.color = this.getRiskColor(compositeRisk);
        
        // Display precision medicine score
        const precisionScore = multiMutationData.therapeutic_strategy?.precision_medicine_score || 0;
        document.getElementById('precision-score').textContent = `${(precisionScore * 100).toFixed(0)}%`;
        
        // Display pathway convergence
        this.displayPathwayConvergence(multiMutationData.pathway_analysis, multiMutationData.mutation_profile);
        
        // Display therapeutic strategies
        this.displayTherapeuticStrategies(multiMutationData.therapeutic_strategy);
        
        // Display clinical interpretation
        this.displayClinicalInterpretation(multiMutationData.comprehensive_interpretation);
    }
    
    displayPathwayConvergence(pathwayAnalysis, mutationProfile) {
        if (!pathwayAnalysis) return;
        
        const pathwayNetwork = document.getElementById('pathway-network');
        const pathwayDetails = document.getElementById('pathway-details');
        
        // Clear previous content
        pathwayNetwork.innerHTML = '';
        pathwayDetails.innerHTML = '';
        
        // Display pathway nodes
        const disruptedPathways = pathwayAnalysis.disrupted_pathways || [];
        const dominantPathways = mutationProfile?.dominant_pathways || [];
        
        // Create pathway visualization with network diagram
        if (dominantPathways.length > 0) {
            // Create a simple network visualization
            const networkHTML = `
                <div class="pathway-network-container">
                    <svg viewBox="0 0 440 320" class="pathway-svg">
                        <!-- Central node for mutations -->
                        <circle cx="220" cy="160" r="40" fill="#2563eb" opacity="0.2" stroke="#2563eb" stroke-width="2"/>
                        <text x="220" y="165" text-anchor="middle" class="mutation-count-text">${mutationProfile?.total_mutations || 0} Mutations</text>
                        
                        ${dominantPathways.map((pathway, index) => {
                            const angle = (index * 2 * Math.PI) / dominantPathways.length - Math.PI / 2;
                            const x = 220 + 110 * Math.cos(angle);
                            const y = 160 + 110 * Math.sin(angle);
                            const isConvergent = index === 0;
                            
                            return `
                                <!-- Pathway node -->
                                <line x1="220" y1="160" x2="${x}" y2="${y}" stroke="${isConvergent ? '#ef4444' : '#6b7280'}" stroke-width="${isConvergent ? '3' : '2'}" opacity="0.5"/>
                                <circle cx="${x}" cy="${y}" r="35" fill="${isConvergent ? '#fef2f2' : '#f9fafb'}" stroke="${isConvergent ? '#ef4444' : '#6b7280'}" stroke-width="2"/>
                                <text x="${x}" y="${y + 5}" text-anchor="middle" class="pathway-text ${isConvergent ? 'convergent' : ''}">${this.formatDisplayString(pathway)}</text>
                            `;
                        }).join('')}
                    </svg>
                    
                    <div class="pathway-legend">
                        <div class="legend-item">
                            <span class="legend-color convergent"></span>
                            <span>Primary Convergence</span>
                        </div>
                        <div class="legend-item">
                            <span class="legend-color normal"></span>
                            <span>Affected Pathway</span>
                        </div>
                    </div>
                </div>
            `;
            
            pathwayNetwork.innerHTML = networkHTML;
        } else {
            // Fallback if no dominant pathways
            pathwayNetwork.innerHTML = '<p class="no-pathway-data">No pathway convergence data available</p>';
        }
        
        // Display detailed pathway information
        disruptedPathways.forEach(pathway => {
            const detailItem = document.createElement('div');
            detailItem.className = 'pathway-detail-item';
            if (pathway.disruption_severity === 'complete') {
                detailItem.classList.add('high-impact');
            }
            
            detailItem.innerHTML = `
                <div class="pathway-name">${this.formatDisplayString(pathway.pathway)}</div>
                <div class="pathway-genes">Affected genes: ${pathway.genes_affected.join(', ')}</div>
                <div class="pathway-impact">
                    ${pathway.disruption_severity} disruption - ${pathway.functional_impact}
                </div>
            `;
            
            pathwayDetails.appendChild(detailItem);
        });
        
        // Add pathway interactions if available
        if (pathwayAnalysis.pathway_interactions) {
            const interactionDiv = document.createElement('div');
            interactionDiv.className = 'pathway-detail-item';
            interactionDiv.innerHTML = `
                <div class="pathway-name">Pathway Interactions</div>
                <div class="pathway-impact">${pathwayAnalysis.pathway_interactions}</div>
            `;
            pathwayDetails.appendChild(interactionDiv);
        }
    }
    
    displayTherapeuticStrategies(therapeuticStrategy) {
        if (!therapeuticStrategy) return;
        
        const combinationTherapies = document.getElementById('combination-therapies');
        combinationTherapies.innerHTML = '';
        
        const combinations = therapeuticStrategy.combination_therapies || [];
        
        combinations.forEach(combo => {
            const therapyDiv = document.createElement('div');
            therapyDiv.className = 'therapy-combination';
            
            const efficacyClass = combo.expected_efficacy?.toLowerCase() || 'moderate';
            
            therapyDiv.innerHTML = `
                <div class="combination-header">
                    <div class="combination-targets">
                        ${combo.target_combination.map(target => 
                            `<span class="target-pill">${target}</span>`
                        ).join('')}
                    </div>
                    <span class="efficacy-badge efficacy-${efficacyClass}">
                        ${combo.expected_efficacy} efficacy
                    </span>
                </div>
                <div class="combination-rationale">${combo.rationale}</div>
            `;
            
            combinationTherapies.appendChild(therapyDiv);
        });
        
        // Add resistance mitigation strategies if available
        if (therapeuticStrategy.resistance_mitigation && therapeuticStrategy.resistance_mitigation.length > 0) {
            const resistanceDiv = document.createElement('div');
            resistanceDiv.className = 'therapy-combination';
            resistanceDiv.innerHTML = `
                <div class="combination-header">
                    <div class="combination-targets">
                        <span class="target-pill">Resistance Prevention</span>
                    </div>
                </div>
                <div class="combination-rationale">
                    ${therapeuticStrategy.resistance_mitigation.join('; ')}
                </div>
            `;
            combinationTherapies.appendChild(resistanceDiv);
        }
    }
    
    displayClinicalInterpretation(interpretation) {
        if (!interpretation) return;
        
        const interpretationText = document.getElementById('interpretation-text');
        interpretationText.textContent = interpretation;
    }

    async exportToPDF() {
        if (!this.currentAnalysis) {
            this.showError('No analysis data available for export');
            return;
        }

        try {
            // Create a new window for PDF generation
            const printWindow = window.open('', '_blank');
            const pdfContent = this.generatePDFContent();
            
            printWindow.document.write(pdfContent);
            printWindow.document.close();
            
            // Wait for content to load then print
            printWindow.onload = () => {
                setTimeout(() => {
                    printWindow.print();
                    printWindow.close();
                }, 500);
            };
            
        } catch (error) {
            this.showError('Failed to generate PDF report: ' + error.message);
        }
    }
    
    generatePDFContent() {
        const analysis = this.currentAnalysis;
        const timestamp = new Date().toLocaleString();
        
        return `
        <!DOCTYPE html>
        <html>
        <head>
            <title>OncoScope Analysis Report</title>
            <style>
                @media print {
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
                    .header { border-bottom: 2px solid #2563eb; padding-bottom: 20px; margin-bottom: 30px; }
                    .logo { color: #2563eb; font-size: 24px; font-weight: bold; }
                    .timestamp { color: #666; font-size: 12px; margin-top: 10px; }
                    .section { margin-bottom: 25px; page-break-inside: avoid; }
                    .section-title { color: #2563eb; font-size: 18px; font-weight: bold; margin-bottom: 15px; border-bottom: 1px solid #e5e7eb; padding-bottom: 5px; }
                    .risk-summary { background: #f8fafc; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                    .risk-score { font-size: 36px; font-weight: bold; color: #2563eb; }
                    .mutation-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                    .mutation-table th, .mutation-table td { border: 1px solid #e5e7eb; padding: 10px; text-align: left; }
                    .mutation-table th { background: #f8fafc; font-weight: bold; }
                    .recommendations { background: #fef7cd; padding: 15px; border-left: 4px solid #f59e0b; margin: 10px 0; }
                    .warning { background: #fef2f2; padding: 15px; border-left: 4px solid #ef4444; margin: 10px 0; }
                    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e5e7eb; font-size: 12px; color: #666; }
                    .page-break { page-break-before: always; }
                }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo">OncoScope</div>
                <div style="font-size: 16px; color: #666;">Privacy-First Cancer Genomics Analysis Report</div>
                <div class="timestamp">Generated: ${timestamp}</div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Overall Risk Assessment</h2>
                <div class="risk-summary">
                    <div style="display: flex; align-items: center; gap: 20px;">
                        <div class="risk-score">${(analysis.overall_risk_score * 100).toFixed(1)}%</div>
                        <div>
                            <div style="font-size: 18px; font-weight: bold;">${analysis.risk_classification}</div>
                            <div style="color: #666;">Overall Risk Level</div>
                        </div>
                    </div>
                    <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                        <div>Confidence Score: ${(analysis.confidence_metrics?.overall_confidence * 100 || 'N/A')}%</div>
                        <div>Actionable Mutations: ${analysis.actionable_mutations?.length || 0}</div>
                        <div>Pathogenic Variants: ${analysis.individual_mutations?.filter(m => m.clinical_significance === 'PATHOGENIC').length || 0}</div>
                        <div>Known Mutations: ${analysis.confidence_metrics?.known_mutations || 0}</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2 class="section-title">Individual Mutation Analysis</h2>
                <table class="mutation-table">
                    <thead>
                        <tr>
                            <th>Gene</th>
                            <th>Variant</th>
                            <th>Protein Change</th>
                            <th>Pathogenicity</th>
                            <th>Clinical Significance</th>
                            <th>Targeted Therapies</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${analysis.individual_mutations?.map(mutation => `
                            <tr>
                                <td>${mutation.gene}</td>
                                <td>${mutation.variant}</td>
                                <td>${mutation.protein_change || 'Unknown'}</td>
                                <td>${(mutation.pathogenicity_score * 100).toFixed(1)}%</td>
                                <td>${this.formatUnderscoreString(mutation.clinical_significance)}</td>
                                <td>${mutation.targeted_therapies?.join(', ') || 'None'}</td>
                            </tr>
                        `).join('') || '<tr><td colspan="6">No mutation data available</td></tr>'}
                    </tbody>
                </table>
            </div>
            
            ${analysis.estimated_tumor_types?.length ? `
            <div class="section">
                <h2 class="section-title">Predicted Cancer Types</h2>
                <ul>
                    ${analysis.estimated_tumor_types.map(tumor => `
                        <li>${this.formatDisplayString(tumor.cancer_type)}: ${(tumor.likelihood * 100).toFixed(1)}% likelihood</li>
                    `).join('')}
                </ul>
            </div>
            ` : ''}
            
            ${analysis.clinical_recommendations?.length ? `
            <div class="section">
                <h2 class="section-title">Clinical Recommendations</h2>
                ${analysis.clinical_recommendations.map(rec => `
                    <div class="recommendations">${rec}</div>
                `).join('')}
            </div>
            ` : ''}
            
            ${analysis.actionable_mutations?.length ? `
            <div class="section">
                <h2 class="section-title">Actionable Mutations</h2>
                ${analysis.actionable_mutations.map(actionable => `
                    <div style="margin-bottom: 15px; padding: 15px; background: #f0f9ff; border-radius: 8px;">
                        <div style="font-weight: bold; color: #2563eb;">${actionable.gene} - ${actionable.mutation}</div>
                        <div>Therapies: ${actionable.therapies?.join(', ') || 'None'}</div>
                        <div>FDA Approved: ${actionable.fda_approved ? 'Yes' : 'No'}</div>
                        <div>Clinical Trials Available: ${actionable.clinical_trials_available ? 'Yes' : 'No'}</div>
                    </div>
                `).join('')}
            </div>
            ` : ''}
            
            ${analysis.warnings?.length ? `
            <div class="section">
                <h2 class="section-title">Warnings & Considerations</h2>
                ${analysis.warnings.map(warning => `
                    <div class="warning">${warning}</div>
                `).join('')}
            </div>
            ` : ''}
            
            <div class="footer">
                <div><strong>Disclaimer:</strong> This analysis is for research purposes only and should not be used for clinical decision-making without consulting a qualified healthcare professional.</div>
                <div style="margin-top: 10px;"><strong>Privacy Notice:</strong> All analysis performed locally. Your genetic data never leaves this device.</div>
                <div style="margin-top: 10px;">Generated by OncoScope v1.0.0 - Privacy-First Cancer Genomics Analysis Platform</div>
            </div>
        </body>
        </html>
        `;
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
    
    setupKeyboardNavigation() {
        // Global keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Skip if user is typing in an input field
            if (e.target.matches('input, textarea, select')) {
                return;
            }
            
            switch (e.key) {
                case 'Enter':
                case ' ':
                    // Activate focused button
                    if (e.target.matches('button')) {
                        e.preventDefault();
                        e.target.click();
                    }
                    break;
                    
                case 'Tab':
                    // Ensure proper tab order - browser handles this naturally
                    break;
                    
                case 'Escape':
                    // Close modals
                    if (!document.getElementById('error-modal').classList.contains('hidden')) {
                        closeErrorModal();
                    }
                    break;
                    
                case 'n':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.resetAnalysis();
                    }
                    break;
                    
                case 'o':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        this.elements.fileInput.click();
                    }
                    break;
                    
                case 'p':
                    if (e.ctrlKey || e.metaKey) {
                        e.preventDefault();
                        if (this.currentAnalysis) {
                            this.exportToPDF();
                        }
                    }
                    break;
            }
        });
        
        // Demo button keyboard navigation
        document.querySelectorAll('.demo-button').forEach(button => {
            button.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    this.loadDemoData(button.dataset.demo);
                }
            });
        });
        
        // Ensure all interactive elements are focusable
        this.ensureFocusableElements();
    }
    
    ensureFocusableElements() {
        // Make sure all clickable elements have proper tabindex
        const clickableElements = document.querySelectorAll(
            'button, [role="button"], .demo-button, .upload-area'
        );
        
        clickableElements.forEach(element => {
            if (!element.hasAttribute('tabindex')) {
                element.setAttribute('tabindex', '0');
            }
        });
        
        // Add keyboard support for upload area
        this.elements.uploadArea.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.elements.fileInput.click();
            }
        });
    }
}

// Global function for modal
function closeErrorModal() {
    document.getElementById('error-modal').classList.add('hidden');
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing OncoScope app...');
    // Add a small delay to ensure everything is ready
    setTimeout(() => {
        new OncoScopeApp();
    }, 500);
});