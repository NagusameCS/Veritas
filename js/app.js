/**
 * VERITAS — Main Application v3.0 (Sunrise)
 * Application initialization with enhanced file support and reporting
 */

const App = {
    currentResult: null,
    currentMetadata: null,
    isAnalyzing: false,

    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.bindNewEvents();
        this.loadHistory();
        this.initTheme();
        this.initTabs();
        console.log('VERITAS v3.0 (Sunrise) initialized');
    },

    /**
     * Bind all event listeners
     */
    bindEvents() {
        // Analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => this.analyze());
        }

        // Clear button
        const clearBtn = document.getElementById('clearBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearInput());
        }

        // Text input - enable/disable analyze button
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.addEventListener('input', () => this.updateAnalyzeButton());
            
            // Allow Ctrl+Enter to analyze
            textInput.addEventListener('keydown', (e) => {
                if (e.ctrlKey && e.key === 'Enter') {
                    this.analyze();
                }
            });
            
            // Handle paste events for clipboard detection
            textInput.addEventListener('paste', (e) => this.handlePaste(e));
        }

        // Sample text button
        const sampleBtn = document.getElementById('sampleBtn');
        if (sampleBtn) {
            sampleBtn.addEventListener('click', () => this.loadSampleText());
        }

        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const view = link.dataset.view;
                if (view) this.showView(view);
            });
        });

        // Theme toggle
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', () => this.toggleTheme());
        }

        // History clear
        const clearHistoryBtn = document.getElementById('clearHistoryBtn');
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', () => this.clearHistory());
        }

        // Copy results
        const copyBtn = document.getElementById('copyResultsBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyResults());
        }

        // Drag and drop file support
        this.initDragDrop();
    },

    /**
     * Initialize drag and drop for file uploads
     */
    initDragDrop() {
        const textInput = document.getElementById('textInput');
        const inputPanel = document.querySelector('.input-panel');
        if (!inputPanel) return;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            inputPanel.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        // Highlight drop area
        ['dragenter', 'dragover'].forEach(eventName => {
            inputPanel.addEventListener(eventName, () => {
                inputPanel.classList.add('drag-over');
            }, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            inputPanel.addEventListener(eventName, () => {
                inputPanel.classList.remove('drag-over');
            }, false);
        });

        // Handle dropped files
        inputPanel.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                const ext = file.name.split('.').pop().toLowerCase();
                
                // Determine file type
                let fileType = 'txt';
                if (ext === 'docx') fileType = 'docx';
                else if (ext === 'pdf') fileType = 'pdf';
                else if (['txt', 'md', 'text'].includes(ext)) fileType = 'txt';
                
                // Create a fake event for the file handler
                const fakeEvent = { target: { files: [file] } };
                this.handleFileUpload(fakeEvent, fileType);
            }
        }, false);
    },

    /**
     * Bind new events for enhanced features
     */
    bindNewEvents() {
        // Upload dropdown
        const uploadDropdownBtn = document.getElementById('uploadDropdownBtn');
        const uploadMenu = document.getElementById('uploadMenu');
        if (uploadDropdownBtn && uploadMenu) {
            uploadDropdownBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                uploadMenu.classList.toggle('active');
            });
            document.addEventListener('click', () => {
                uploadMenu.classList.remove('active');
            });
        }

        // File upload options
        document.querySelectorAll('.upload-option[data-type]').forEach(option => {
            const input = option.querySelector('input[type="file"]');
            if (input) {
                option.addEventListener('click', () => {});  // Label handles click
                input.addEventListener('change', (e) => this.handleFileUpload(e, option.dataset.type));
            }
        });

        // Paste from clipboard button
        const pasteBtn = document.getElementById('pasteClipboardBtn');
        if (pasteBtn) {
            pasteBtn.addEventListener('click', () => this.pasteFromClipboard());
        }

        // Google Docs button
        const gdocsBtn = document.getElementById('googleDocsBtn');
        if (gdocsBtn) {
            gdocsBtn.addEventListener('click', () => this.promptGoogleDocsUrl());
        }

        // View Full Report button
        const viewFullReportBtn = document.getElementById('viewFullReportBtn');
        if (viewFullReportBtn) {
            viewFullReportBtn.addEventListener('click', () => this.openFullReport());
        }
        
        // Humanizer events
        this.bindHumanizerEvents();
        
        // Model selector events
        this.bindModelSelectorEvents();
    },
    
    /**
     * Bind humanizer-related events
     */
    bindHumanizerEvents() {
        const humanizeBtn = document.getElementById('humanizeBtn');
        if (humanizeBtn) {
            humanizeBtn.addEventListener('click', () => this.humanizeText());
        }
        
        const copyHumanizedBtn = document.getElementById('copyHumanizedBtn');
        if (copyHumanizedBtn) {
            copyHumanizedBtn.addEventListener('click', () => this.copyHumanizedText());
        }
        
        const analyzeHumanizedBtn = document.getElementById('analyzeHumanizedBtn');
        if (analyzeHumanizedBtn) {
            analyzeHumanizedBtn.addEventListener('click', () => this.analyzeHumanizedText());
        }
        
        // Update stats as user types in humanizer input
        const humanizerInput = document.getElementById('humanizerInput');
        if (humanizerInput) {
            humanizerInput.addEventListener('input', () => this.updateHumanizerInputStats());
        }
    },
    
    /**
     * Bind model selector events - Carousel version
     */
    bindModelSelectorEvents() {
        // Model data for the carousel selector
        this.models = [
            { id: 'helios', name: 'Helios', accuracy: '99.24%', badge: 'Flagship', icon: 'flare', badgeClass: 'flagship', desc: '45 features · Tone + hedging · Best overall' },
            { id: 'zenith', name: 'Zenith', accuracy: '99.57%', badge: 'Perplexity', icon: 'brightness_high', badgeClass: 'perplexity', desc: 'Entropy analysis · 86.7% humanized detection' },
            { id: 'sunrise', name: 'Sunrise', accuracy: '98.08%', badge: 'Balanced', icon: 'wb_sunny', badgeClass: 'balanced', desc: 'Statistical variance · Fast · F1: 98.09%' },
            { id: 'dawn', name: 'Dawn', accuracy: '84.9%', badge: 'Legacy', icon: 'wb_twilight', badgeClass: 'legacy', desc: 'Rule-based heuristics · Lightweight' }
        ];
        
        // Load saved model or default to helios
        const savedModel = localStorage.getItem('veritas-model') || 'helios';
        this.currentModelIndex = this.models.findIndex(m => m.id === savedModel);
        if (this.currentModelIndex === -1) this.currentModelIndex = 0;
        
        // Navigation arrows
        const prevBtn = document.getElementById('modelPrev');
        const nextBtn = document.getElementById('modelNext');
        
        if (prevBtn) {
            prevBtn.addEventListener('click', () => {
                this.currentModelIndex = (this.currentModelIndex - 1 + this.models.length) % this.models.length;
                this.updateModelDisplay(true);
            });
        }
        
        if (nextBtn) {
            nextBtn.addEventListener('click', () => {
                this.currentModelIndex = (this.currentModelIndex + 1) % this.models.length;
                this.updateModelDisplay(true);
            });
        }
        
        // Initial display
        this.updateModelDisplay(false);
    },
    
    /**
     * Update the model carousel display
     */
    updateModelDisplay(showToast = true) {
        const model = this.models[this.currentModelIndex];
        const display = document.getElementById('modelDisplay');
        
        if (display) {
            display.innerHTML = `
                <div class="model-card-inner" data-model="${model.id}">
                    <div class="model-card-header">
                        <span class="material-icons model-card-icon">${model.icon}</span>
                        <div class="model-card-title">
                            <span class="model-card-name">${model.name}</span>
                            <span class="model-card-badge ${model.badgeClass}">${model.badge}</span>
                        </div>
                        <span class="model-card-accuracy">${model.accuracy}</span>
                    </div>
                    <p class="model-card-desc">${model.desc}</p>
                </div>
            `;
        }
        
        // Update hidden radio
        const radio = document.getElementById(`model-${model.id}`);
        if (radio) radio.checked = true;
        
        // Handle the change
        this.handleModelChange(model.id, showToast);
    },
    
    /**
     * Select a model by ID (used by About section links)
     */
    selectModel(modelType, showToast = true) {
        const index = this.models.findIndex(m => m.id === modelType);
        if (index !== -1) {
            this.currentModelIndex = index;
            this.updateModelDisplay(showToast);
        }
    },
    
    /**
     * Select model and navigate to analyze view
     */
    selectModelAndAnalyze(modelType) {
        this.selectModel(modelType, true);
        // Navigate to analyze view
        this.showView('analyze');
    },
    
    /**
     * Handle model type change
     */
    handleModelChange(modelType, showToast = true) {
        localStorage.setItem('veritas-model', modelType);
        
        // Model labels for all supported models
        const modelLabels = {
            'helios': 'Helios',
            'zenith': 'Zenith',
            'sunrise': 'Sunrise',
            'dawn': 'Dawn'
        };
        
        const modelLabel = modelLabels[modelType] || modelType;
        console.log(`Model changed to: ${modelLabel}`);
        
        // Update the analyzer engine's model
        if (typeof AnalyzerEngine !== 'undefined') {
            AnalyzerEngine.setModel(modelType);
        }
        
        // Update any result displays to indicate current model
        const resultModelIndicator = document.getElementById('resultModelIndicator');
        if (resultModelIndicator) {
            resultModelIndicator.textContent = modelLabel;
        }
        
        // Show toast notification
        if (showToast) {
            this.showToast(`Switched to ${modelLabel} Model`, 'info');
        }
    },
    
    /**
     * Get current model type
     */
    getCurrentModel() {
        const selected = document.querySelector('input[name="model"]:checked');
        return selected ? selected.value : 'helios';
    },
    
    /**
     * Update humanizer input stats
     */
    updateHumanizerInputStats() {
        const input = document.getElementById('humanizerInput');
        const statsEl = document.getElementById('humanizerInputStats');
        
        if (!input || !statsEl) return;
        
        const text = input.value;
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
        
        statsEl.textContent = `${chars.toLocaleString()} characters | ${words.toLocaleString()} words`;
    },
    
    /**
     * Humanize text using the Humanizer module
     */
    humanizeText() {
        const input = document.getElementById('humanizerInput');
        const output = document.getElementById('humanizerOutput');
        const intensity = document.getElementById('humanizeIntensity');
        const style = document.getElementById('humanizeStyle');
        const statsEl = document.getElementById('humanizeStats');
        const actionsEl = document.getElementById('humanizedActions');
        
        if (!input || !output) return;
        
        const text = input.value.trim();
        if (!text) {
            this.showToast('Please enter text to humanize', 'warning');
            return;
        }
        
        // Check if Humanizer is available
        if (typeof Humanizer === 'undefined') {
            this.showToast('Humanizer module not loaded', 'error');
            return;
        }
        
        try {
            const options = {
                intensity: intensity ? intensity.value : 'medium',
                style: style ? style.value : 'natural'
            };
            
            const result = Humanizer.humanize(text, options);
            
            // Display output
            output.value = result.text;
            
            // Show stats
            if (statsEl) {
                statsEl.innerHTML = `
                    <div class="humanize-stat">
                        <span class="stat-label">Contractions Added</span>
                        <span class="stat-value">${result.stats.contractionsAdded}</span>
                    </div>
                    <div class="humanize-stat">
                        <span class="stat-label">AI Phrases Removed</span>
                        <span class="stat-value">${result.stats.aiPhrasesRemoved}</span>
                    </div>
                    <div class="humanize-stat">
                        <span class="stat-label">Disfluencies Added</span>
                        <span class="stat-value">${result.stats.disfluenciesAdded}</span>
                    </div>
                    <div class="humanize-stat">
                        <span class="stat-label">Hedging Added</span>
                        <span class="stat-value">${result.stats.hedgingAdded}</span>
                    </div>
                    <div class="humanize-stat">
                        <span class="stat-label">Sentences Varied</span>
                        <span class="stat-value">${result.stats.sentencesVaried}</span>
                    </div>
                    <div class="humanize-stat">
                        <span class="stat-label">Overall Changes</span>
                        <span class="stat-value">${result.stats.totalChanges}</span>
                    </div>
                `;
                statsEl.style.display = 'grid';
            }
            
            // Show action buttons
            if (actionsEl) {
                actionsEl.style.display = 'flex';
            }
            
            this.showToast(`Text humanized with ${result.stats.totalChanges} changes`, 'success');
            
        } catch (error) {
            console.error('Humanization error:', error);
            this.showToast('Error humanizing text: ' + error.message, 'error');
        }
    },
    
    /**
     * Copy humanized text to clipboard
     */
    copyHumanizedText() {
        const output = document.getElementById('humanizerOutput');
        if (!output || !output.value) {
            this.showToast('No humanized text to copy', 'warning');
            return;
        }
        
        navigator.clipboard.writeText(output.value).then(() => {
            this.showToast('Humanized text copied to clipboard', 'success');
        }).catch(err => {
            console.error('Failed to copy:', err);
            this.showToast('Failed to copy text', 'error');
        });
    },
    
    /**
     * Analyze the humanized text
     */
    analyzeHumanizedText() {
        const output = document.getElementById('humanizerOutput');
        if (!output || !output.value) {
            this.showToast('No humanized text to analyze', 'warning');
            return;
        }
        
        // Switch to analyze view and populate input
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = output.value;
            this.updateAnalyzeButton();
        }
        
        this.showView('analyze');
        
        // Auto-trigger analysis
        setTimeout(() => this.analyze(), 100);
    },

    /**
     * Initialize tabs
     */
    initTabs() {
        document.querySelectorAll('.tab-btn').forEach(tab => {
            tab.addEventListener('click', () => {
                const tabGroup = tab.closest('.tabs');
                const tabId = tab.dataset.tab;

                // Update active tab
                tabGroup.querySelectorAll('.tab-btn').forEach(t => {
                    t.classList.remove('active');
                    t.setAttribute('aria-selected', 'false');
                });
                tab.classList.add('active');
                tab.setAttribute('aria-selected', 'true');

                // Update tab panels
                const container = tabGroup.closest('.results-tabs') || document;
                container.querySelectorAll('.tab-panel').forEach(panel => {
                    panel.classList.remove('active');
                    panel.hidden = true;
                });

                const activePanel = container.querySelector(`#${tabId}`);
                if (activePanel) {
                    activePanel.classList.add('active');
                    activePanel.hidden = false;
                }
            });
        });
    },

    /**
     * Initialize theme
     */
    initTheme() {
        const savedTheme = localStorage.getItem('veritas-theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    },

    /**
     * Toggle theme
     */
    toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('veritas-theme', next);
        this.updateThemeIcon(next);
    },

    /**
     * Update theme toggle icon
     */
    updateThemeIcon(theme) {
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.innerHTML = theme === 'dark' 
                ? '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2m0 18v2M4.22 4.22l1.42 1.42m12.72 12.72l1.42 1.42M1 12h2m18 0h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
                : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>';
        }
    },

    /**
     * Show a specific view
     */
    showView(viewName) {
        document.querySelectorAll('.view').forEach(view => {
            view.classList.remove('active');
            view.hidden = true;
        });

        const targetView = document.getElementById(`${viewName}View`);
        if (targetView) {
            targetView.classList.add('active');
            targetView.hidden = false;
        }

        // Update nav
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.toggle('active', link.dataset.view === viewName);
        });
    },

    /**
     * Update analyze button state
     */
    updateAnalyzeButton() {
        const textInput = document.getElementById('textInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (textInput && analyzeBtn) {
            const hasText = textInput.value.trim().length > 0;
            analyzeBtn.disabled = !hasText || this.isAnalyzing;
        }
        
        // Update text statistics
        this.updateTextStats();
        
        // Auto-expand textarea
        this.autoExpandTextarea();
    },

    /**
     * Update text statistics display (characters, words, sentences)
     */
    updateTextStats() {
        const textInput = document.getElementById('textInput');
        const charCountEl = document.getElementById('charCount');
        const wordCountEl = document.getElementById('wordCount');
        const sentenceCountEl = document.getElementById('sentenceCount');
        
        if (!textInput) return;
        
        const text = textInput.value;
        
        // Calculate stats
        const chars = text.length;
        const words = text.trim() ? text.trim().split(/\s+/).filter(w => w.length > 0).length : 0;
        const sentences = text.trim() ? (text.match(/[.!?]+(?:\s|$)/g) || []).length : 0;
        
        // Update display
        if (charCountEl) charCountEl.textContent = `${chars.toLocaleString()} character${chars !== 1 ? 's' : ''}`;
        if (wordCountEl) wordCountEl.textContent = `${words.toLocaleString()} word${words !== 1 ? 's' : ''}`;
        if (sentenceCountEl) sentenceCountEl.textContent = `${sentences.toLocaleString()} sentence${sentences !== 1 ? 's' : ''}`;
    },

    /**
     * Auto-expand textarea to fit content
     */
    autoExpandTextarea() {
        const textInput = document.getElementById('textInput');
        if (!textInput) return;
        
        // Reset height to auto to get the correct scrollHeight
        textInput.style.height = 'auto';
        
        // Set to scrollHeight with minimum
        const minHeight = 200;
        const newHeight = Math.max(minHeight, textInput.scrollHeight);
        textInput.style.height = newHeight + 'px';
    },

    /**
     * Clear input
     */
    clearInput() {
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = '';
            textInput.style.height = '200px'; // Reset to minimum
            textInput.focus();
        }
        
        // Reset metadata
        this.currentMetadata = null;
        const metadataBar = document.getElementById('metadataBar');
        if (metadataBar) {
            metadataBar.hidden = true;
        }
        
        this.updateAnalyzeButton();
        this.hideResults();
    },

    /**
     * Hide results panel
     */
    hideResults() {
        const resultsPanel = document.querySelector('.results-panel');
        if (resultsPanel) {
            resultsPanel.classList.remove('has-results');
        }
    },

    /**
     * Run analysis
     */
    async analyze() {
        const textInput = document.getElementById('textInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        if (!textInput || this.isAnalyzing) return;
        
        const text = textInput.value.trim();
        if (!text) return;

        // Check minimum length
        const words = text.split(/\s+/).length;
        if (words < 10) {
            this.showToast('Please enter at least 10 words for analysis', 'warning');
            return;
        }

        // Set loading state
        this.isAnalyzing = true;
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
        this.showLoadingState();

        try {
            // Small delay for UI feedback
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Run analysis with metadata if available
            const result = AnalyzerEngine.analyze(text, this.currentMetadata);
            this.currentResult = result;
            
            // Display results
            this.displayResults(result);
            
            // Render advanced visualizations
            this.renderAdvancedVisualizations(result);
            
            // Save to history
            this.saveToHistory(text, result);
            
            // Show results panel
            const resultsPanel = document.querySelector('.results-panel');
            if (resultsPanel) {
                resultsPanel.classList.add('has-results');
            }

            this.showToast('Analysis complete', 'success');

        } catch (error) {
            console.error('Analysis error:', error);
            this.showToast('An error occurred during analysis', 'error');
            // Clear skeleton loaders on error
            this.hideLoadingState();
        } finally {
            this.isAnalyzing = false;
            analyzeBtn.classList.remove('loading');
            this.updateAnalyzeButton();
        }
    },

    /**
     * Hide loading state / clear skeletons
     */
    hideLoadingState() {
        // Restore the primary-result structure if it was replaced
        const primaryResult = document.querySelector('.primary-result');
        if (primaryResult && primaryResult.querySelector('.skeleton')) {
            primaryResult.innerHTML = `
                <div class="score-ring-container"></div>
                <div class="result-text">
                    <span class="verdict-label"></span>
                    <span class="verdict-description"></span>
                    <div class="confidence-inline">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-value">-</span>
                    </div>
                </div>
            `;
        }

        const findingsContainer = document.getElementById('findingsList');
        if (findingsContainer) {
            findingsContainer.innerHTML = '<p class="no-findings">No findings available</p>';
        }
        
        // Clear transparency header
        const transparencyHeader = document.getElementById('modelTransparencyHeader');
        if (transparencyHeader) {
            transparencyHeader.innerHTML = '';
        }
    },

    /**
     * Render advanced visualizations in graphs tab
     */
    renderAdvancedVisualizations(result) {
        // Check if AdvancedVisualizations is available
        if (typeof AdvancedVisualizations === 'undefined') return;

        // Sentence length histogram
        const histogramContainer = document.getElementById('sentenceLengthHistogram');
        if (histogramContainer && result.sentences) {
            AdvancedVisualizations.createSentenceLengthHistogram(histogramContainer, result.sentences);
        }

        // N-gram heatmap - find repetition analyzer result
        const repetitionResult = result.categoryResults.find(r => 
            r.name?.toLowerCase().includes('repetition') || r.category === 12
        );
        const heatmapContainer = document.getElementById('ngramHeatmap');
        if (heatmapContainer && repetitionResult?.details) {
            AdvancedVisualizations.createNgramHeatmap(heatmapContainer, repetitionResult.details);
        }

        // Tone timeline - find tone analyzer result
        const toneResult = result.categoryResults.find(r => 
            r.name?.toLowerCase().includes('tone') || r.category === 13
        );
        const timelineContainer = document.getElementById('toneTimeline');
        if (timelineContainer && toneResult?.details) {
            AdvancedVisualizations.createToneTimeline(timelineContainer, toneResult.details);
        }

        // Zipf chart
        const textInput = document.getElementById('textInput');
        const zipfContainer = document.getElementById('zipfChart');
        if (zipfContainer && textInput) {
            const tokens = Utils.tokenize(textInput.value.toLowerCase());
            AdvancedVisualizations.createZipfChart(zipfContainer, tokens);
        }

        // Feature contribution chart
        const contributionContainer = document.getElementById('featureContributionChart');
        if (contributionContainer && result.categoryResults) {
            AdvancedVisualizations.createFeatureContributionChart(contributionContainer, result.categoryResults);
        }
    },

    /**
     * Show loading state
     */
    showLoadingState() {
        const resultsPanel = document.querySelector('.results-panel');
        if (!resultsPanel) return;

        resultsPanel.classList.add('has-results');
        
        // Show skeleton loaders
        const primaryResult = document.querySelector('.primary-result');
        if (primaryResult) {
            primaryResult.innerHTML = `
                <div class="skeleton skeleton-circle" style="width: 120px; height: 120px;"></div>
                <div class="result-text">
                    <div class="skeleton skeleton-text" style="width: 150px; height: 24px;"></div>
                    <div class="skeleton skeleton-text" style="width: 200px; height: 16px;"></div>
                </div>
            `;
        }

        const findingsContainer = document.getElementById('findingsList');
        if (findingsContainer) {
            findingsContainer.innerHTML = `
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text"></div>
                <div class="skeleton skeleton-text"></div>
            `;
        }
        
        // Clear model transparency header during loading
        const transparencyHeader = document.getElementById('modelTransparencyHeader');
        if (transparencyHeader) {
            transparencyHeader.innerHTML = `
                <div class="skeleton skeleton-text" style="width: 100%; height: 60px;"></div>
            `;
        }
    },

    /**
     * Display analysis results
     */
    displayResults(result) {
        // Render model transparency header first
        this.renderModelTransparencyHeader();
        
        // Restore primary-result structure if it was replaced by skeleton
        const primaryResult = document.querySelector('.primary-result');
        if (primaryResult && (primaryResult.querySelector('.skeleton') || !primaryResult.querySelector('.score-ring-container'))) {
            primaryResult.innerHTML = `
                <div class="score-ring-container"></div>
                <div class="result-text">
                    <span class="verdict-label"></span>
                    <span class="verdict-description"></span>
                    <div class="confidence-inline">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-value">-</span>
                    </div>
                </div>
            `;
        }

        // Overall score
        const scoreContainer = document.querySelector('.score-ring-container');
        if (scoreContainer) {
            Visualizations.createScoreRing(scoreContainer, result.aiProbability, result.confidence);
        }
        
        // Verdict - use the verdict from analyzer engine directly (don't override)
        const verdictEl = document.querySelector('.verdict-label');
        const verdictDescEl = document.querySelector('.verdict-description');
        if (verdictEl && result.verdict) {
            verdictEl.textContent = result.verdict.label;
            verdictEl.className = `verdict-label ${result.verdict.level}`;
        }
        if (verdictDescEl && result.verdict) {
            verdictDescEl.textContent = result.verdict.description;
        }

        // Confidence with interval
        const confidenceEl = document.querySelector('.confidence-value');
        if (confidenceEl) {
            if (result.confidenceInterval) {
                const ci = result.confidenceInterval;
                confidenceEl.innerHTML = `${Math.round(result.confidence * 100)}% 
                    <span class="confidence-range">(${Math.round(ci.lower * 100)}%—${Math.round(ci.upper * 100)}%)</span>`;
            } else {
                confidenceEl.textContent = Math.round(result.confidence * 100) + '%';
            }
        }
        
        // Render simplified probability bar
        this.renderProbabilityBar(result);
        
        // Render high severity alerts
        this.renderHighSeverityAlerts(result);

        // Render confidence interval bar
        this.renderConfidenceInterval(result);

        // Render false positive warnings
        this.renderFalsePositiveWarnings(result);

        // Render verbose conclusion
        this.renderVerboseConclusion(result);

        // Render humanization advisory
        this.renderHumanizationAdvisory(result);

        // Stats
        const statsContainer = document.querySelector('.text-stats');
        if (statsContainer && result.stats) {
            statsContainer.innerHTML = `
                <span class="stat">${result.stats.words} words</span>
                <span class="stat">${result.stats.sentences} sentences</span>
                <span class="stat">${result.analysisTime}</span>
            `;
        }

        // Findings
        const findingsContainer = document.getElementById('findingsList');
        if (findingsContainer) {
            Visualizations.renderFindings(findingsContainer, result.findings, 8);
        }

        // Tab content
        this.renderTabContent(result);

        // First tab active by default
        const firstTab = document.querySelector('.tab-btn[data-tab="tab-highlighted"]');
        if (firstTab) firstTab.click();
    },

    /**
     * Render tab content
     */
    renderTabContent(result) {
        // Highlighted text tab
        const highlightedContainer = document.getElementById('highlightedText');
        if (highlightedContainer) {
            Visualizations.renderHighlightedText(
                highlightedContainer, 
                result.sentences, 
                result.sentenceScores
            );
        }

        // Feature analysis tab
        const featuresContainer = document.getElementById('featureAnalysis');
        if (featuresContainer) {
            Visualizations.createCategoryBars(featuresContainer, result.categoryResults);
        }

        // Advanced statistics tab
        const statsContainer = document.getElementById('advancedStatistics');
        if (statsContainer && result.advancedStats) {
            this.renderAdvancedStatistics(statsContainer, result.advancedStats, result.stats, result);
        }

        // Probability graphs tab
        const graphsContainer = document.getElementById('probabilityGraphs');
        if (graphsContainer) {
            graphsContainer.innerHTML = `
                <div class="graph-section">
                    <h4>Sentence-by-Sentence Analysis</h4>
                    <div class="sentence-graph-container"></div>
                </div>
                <div class="graph-section">
                    <h4>Category Radar</h4>
                    <div class="radar-chart-container"></div>
                </div>
            `;
            
            const sentenceGraph = graphsContainer.querySelector('.sentence-graph-container');
            const radarChart = graphsContainer.querySelector('.radar-chart-container');
            
            if (sentenceGraph) {
                Visualizations.createSentenceGraph(sentenceGraph, result.sentenceScores);
            }
            if (radarChart) {
                Visualizations.createRadarChart(radarChart, result.categoryResults);
            }
        }

        // Detailed report tab
        const reportContainer = document.getElementById('detailedReport');
        if (reportContainer) {
            const report = AnalyzerEngine.generateReport(result);
            Visualizations.renderDetailedReport(reportContainer, report);
        }
    },

    /**
     * Render advanced statistics panel
     */
    renderAdvancedStatistics(container, advStats, basicStats, fullResult = null) {
        const formatNum = (n, decimals = 2) => {
            if (typeof n !== 'number' || isNaN(n)) return '—';
            return n.toFixed(decimals);
        };

        const formatPct = (n) => {
            if (typeof n !== 'number' || isNaN(n)) return '—';
            return (n * 100).toFixed(1) + '%';
        };

        const getIndicator = (value, thresholds, invert = false) => {
            // Returns 'ai', 'human', or 'neutral' based on value
            if (typeof value !== 'number' || isNaN(value)) return 'neutral';
            const [aiThresh, humanThresh] = thresholds;
            if (invert) {
                if (value < aiThresh) return 'ai';
                if (value > humanThresh) return 'human';
            } else {
                if (value > aiThresh) return 'ai';
                if (value < humanThresh) return 'human';
            }
            return 'neutral';
        };

        let html = `
            <div class="stats-grid">
                <!-- Basic Document Stats -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">description</span> Document Overview</h4>
                    <div class="stats-table">
                        <div class="stat-row"><span class="stat-label">Characters</span><span class="stat-value">${basicStats.characters?.toLocaleString() || 0}</span></div>
                        <div class="stat-row"><span class="stat-label">Words</span><span class="stat-value">${basicStats.words?.toLocaleString() || 0}</span></div>
                        <div class="stat-row"><span class="stat-label">Sentences</span><span class="stat-value">${basicStats.sentences?.toLocaleString() || 0}</span></div>
                        <div class="stat-row"><span class="stat-label">Paragraphs</span><span class="stat-value">${basicStats.paragraphs?.toLocaleString() || 0}</span></div>
                        <div class="stat-row"><span class="stat-label">Avg Words/Sentence</span><span class="stat-value">${basicStats.avgWordsPerSentence || 0}</span></div>
                    </div>
                </div>

                <!-- Vocabulary Richness -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">menu_book</span> Vocabulary Richness</h4>
                    <div class="stats-table">
                        <div class="stat-row">
                            <span class="stat-label">Unique Words</span>
                            <span class="stat-value">${advStats.vocabulary?.uniqueWords?.toLocaleString() || 0}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.vocabulary?.typeTokenRatio, [0.3, 0.5], true)}">
                            <span class="stat-label">Type-Token Ratio (TTR)</span>
                            <span class="stat-value">${formatPct(advStats.vocabulary?.typeTokenRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Root TTR (Guiraud's R)</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.rootTTR)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Log TTR (Herdan's C)</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.logTTR, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.vocabulary?.hapaxLegomenaRatio, [0.35, 0.5], true)}">
                            <span class="stat-label">Hapax Legomena Ratio</span>
                            <span class="stat-value">${formatPct(advStats.vocabulary?.hapaxLegomenaRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Dis Legomena Ratio</span>
                            <span class="stat-value">${formatPct(advStats.vocabulary?.disLegomenaRatio)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.vocabulary?.yulesK, [150, 100])}">
                            <span class="stat-label">Yule's K</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.yulesK, 1)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.vocabulary?.simpsonsD, [0.02, 0.01])}">
                            <span class="stat-label">Simpson's D</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.simpsonsD, 4)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Honore's R</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.honoresR, 0)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Brunet's W</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.brunetsW, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Sichel's S</span>
                            <span class="stat-value">${formatNum(advStats.vocabulary?.sichelsS, 3)}</span>
                        </div>
                    </div>
                </div>

                <!-- Sentence Statistics -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">notes</span> Sentence Analysis</h4>
                    <div class="stats-table">
                        <div class="stat-row">
                            <span class="stat-label">Mean Length</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.mean, 1)} words</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Median Length</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.median, 1)} words</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Std Deviation</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.stdDev, 2)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Min / Max</span>
                            <span class="stat-value">${advStats.sentences?.min || 0} / ${advStats.sentences?.max || 0}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.sentences?.coefficientOfVariation, [0.35, 0.5], true)}">
                            <span class="stat-label">Coeff. of Variation</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.coefficientOfVariation, 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Skewness</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.skewness, 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Kurtosis</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.kurtosis, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.sentences?.gini, [0.15, 0.25], true)}">
                            <span class="stat-label">Gini Coefficient</span>
                            <span class="stat-value">${formatNum(advStats.sentences?.gini, 3)}</span>
                        </div>
                    </div>
                </div>

                <!-- Zipf's Law -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">trending_up</span> Zipf's Law Analysis</h4>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.zipf?.compliance, [0.7, 0.85], true)}">
                            <span class="stat-label">Zipf Compliance</span>
                            <span class="stat-value">${formatPct(advStats.zipf?.compliance)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(Math.abs((advStats.zipf?.slope || 0) + 1), [0.3, 0.15])}">
                            <span class="stat-label">Log-Log Slope</span>
                            <span class="stat-value">${formatNum(advStats.zipf?.slope, 3)} (ideal: -1)</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">R² (Fit Quality)</span>
                            <span class="stat-value">${formatNum(advStats.zipf?.rSquared, 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Deviation from Ideal</span>
                            <span class="stat-value">${formatNum(advStats.zipf?.deviation, 3)}</span>
                        </div>
                    </div>
                </div>

                <!-- Readability -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">chrome_reader_mode</span> Readability Metrics</h4>
                    <div class="stats-table">
                        <div class="stat-row">
                            <span class="stat-label">Avg Syllables/Word</span>
                            <span class="stat-value">${formatNum(advStats.readability?.avgSyllablesPerWord, 2)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Flesch Reading Ease</span>
                            <span class="stat-value">${formatNum(advStats.readability?.fleschReadingEase, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Flesch-Kincaid Grade</span>
                            <span class="stat-value">${formatNum(advStats.readability?.fleschKincaidGrade, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Gunning Fog Index</span>
                            <span class="stat-value">${formatNum(advStats.readability?.gunningFogIndex, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Coleman-Liau Index</span>
                            <span class="stat-value">${formatNum(advStats.readability?.colemanLiauIndex, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">SMOG Index</span>
                            <span class="stat-value">${formatNum(advStats.readability?.smogIndex, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">ARI (Automated Readability)</span>
                            <span class="stat-value">${formatNum(advStats.readability?.ariIndex, 1)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Complex Word %</span>
                            <span class="stat-value">${formatNum(advStats.readability?.complexWordPercentage, 1)}%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Polysyllable %</span>
                            <span class="stat-value">${formatNum(advStats.readability?.polysyllablePercentage, 1)}%</span>
                        </div>
                    </div>
                </div>

                <!-- Burstiness -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">flash_on</span> Burstiness & Uniformity</h4>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.burstiness?.sentenceLength, [0.1, 0.25], true)}">
                            <span class="stat-label">Sentence Length Burstiness</span>
                            <span class="stat-value">${formatNum(advStats.burstiness?.sentenceLength, 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Word Length Burstiness</span>
                            <span class="stat-value">${formatNum(advStats.burstiness?.wordLength, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.burstiness?.overallUniformity, [0.7, 0.5])}">
                            <span class="stat-label">Overall Uniformity</span>
                            <span class="stat-value">${formatPct(advStats.burstiness?.overallUniformity)}</span>
                        </div>
                    </div>
                </div>

                <!-- N-gram Analysis -->
                <div class="stats-section ${advStats.ngrams?.repeatedPhraseScore > 0.3 ? 'humanizer-warning' : ''}">
                    <h4><span class="material-icons section-icon">link</span> N-gram & Phrase Analysis</h4>
                    <p class="section-note">Research shows repeated higher-order n-grams are strong AI indicators.</p>
                    <div class="stats-table">
                        <div class="stat-row">
                            <span class="stat-label">Unique Bigrams</span>
                            <span class="stat-value">${advStats.ngrams?.uniqueBigrams?.toLocaleString() || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Unique Trigrams</span>
                            <span class="stat-value">${advStats.ngrams?.uniqueTrigrams?.toLocaleString() || 0}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Unique Quadgrams</span>
                            <span class="stat-value">${advStats.ngrams?.uniqueQuadgrams?.toLocaleString() || 0}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.ngrams?.bigramRepetitionRate, [0.4, 0.25])}">
                            <span class="stat-label">Bigram Repetition Rate</span>
                            <span class="stat-value">${formatPct(advStats.ngrams?.bigramRepetitionRate)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.ngrams?.trigramRepetitionRate, [0.2, 0.1])}">
                            <span class="stat-label">Trigram Repetition Rate</span>
                            <span class="stat-value">${formatPct(advStats.ngrams?.trigramRepetitionRate)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.ngrams?.quadgramRepetitionRate, [0.1, 0.05])}">
                            <span class="stat-label">Quadgram Repetition Rate</span>
                            <span class="stat-value">${formatPct(advStats.ngrams?.quadgramRepetitionRate)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.ngrams?.repeatedPhraseScore, [0.3, 0.1])}">
                            <span class="stat-label"><strong>Repeated Phrase Score</strong></span>
                            <span class="stat-value"><strong>${formatPct(advStats.ngrams?.repeatedPhraseScore)}</strong></span>
                        </div>
                        <div class="stat-row indicator-${advStats.ngrams?.repeatedPhraseCount > 2 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Repeated Phrases (4+ words)</span>
                            <span class="stat-value">${advStats.ngrams?.repeatedPhraseCount || 0} found</span>
                        </div>
                    </div>
                    ${advStats.ngrams?.repeatedPhrases?.length > 0 ? `
                    <div class="repeated-phrases-list">
                        <p class="stat-label">Top repeated phrases:</p>
                        <ul class="phrase-list">
                            ${advStats.ngrams.repeatedPhrases.slice(0, 5).map(p => 
                                `<li>"${p.phrase}" (${p.count}x)</li>`
                            ).join('')}
                        </ul>
                    </div>
                    ` : ''}
                </div>

                <!-- Function Words -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">text_fields</span> Word Analysis</h4>
                    <div class="stats-table">
                        <div class="stat-row">
                            <span class="stat-label">Avg Word Length</span>
                            <span class="stat-value">${formatNum(advStats.words?.avgLength, 2)} chars</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Word Entropy</span>
                            <span class="stat-value">${formatNum(advStats.words?.entropy, 2)} bits</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Function Word Ratio</span>
                            <span class="stat-value">${formatPct(advStats.functionWords?.ratio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Content Word Ratio</span>
                            <span class="stat-value">${formatPct(advStats.functionWords?.contentWordRatio)}</span>
                        </div>
                    </div>
                </div>

                <!-- Word Pattern Analysis -->
                <div class="stats-section">
                    <h4><span class="material-icons section-icon">label</span> Word Pattern Analysis</h4>
                    <p class="section-note">POS-like analysis without external tools. Research shows AI has different word class distributions.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${advStats.wordPatterns?.firstPersonRatio < 0.01 ? 'ai' : (advStats.wordPatterns?.firstPersonRatio > 0.03 ? 'human' : 'neutral')}">
                            <span class="stat-label">First-Person Pronoun Ratio</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.firstPersonRatio)}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.wordPatterns?.hedgingRatio > 0.02 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Hedging Word Ratio</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.hedgingRatio)}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.wordPatterns?.starterDiversity < 0.4 ? 'ai' : (advStats.wordPatterns?.starterDiversity > 0.7 ? 'human' : 'neutral')}">
                            <span class="stat-label">Sentence Starter Diversity</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.starterDiversity)}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.wordPatterns?.aiStarterRatio > 0.5 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Common AI Starters Ratio</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.aiStarterRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Verb-like Words</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.verbRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Adjective-like Words</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.adjectiveRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Adverb-like Words</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.adverbRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Noun-like Words</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.nounRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Content Density</span>
                            <span class="stat-value">${formatPct(advStats.wordPatterns?.contentDensity)}</span>
                        </div>
                    </div>
                </div>

                <!-- Advanced Statistical Tests -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">science</span> Advanced Statistical Tests</h4>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.autocorrelation?.periodicityScore, [0.6, 0.3])}">
                            <span class="stat-label">Periodicity Score</span>
                            <span class="stat-value">${formatPct(advStats.autocorrelation?.periodicityScore)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.perplexity?.predictability, [0.6, 0.4])}">
                            <span class="stat-label">N-gram Predictability</span>
                            <span class="stat-value">${formatPct(advStats.perplexity?.predictability)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Perplexity (approx)</span>
                            <span class="stat-value">${formatNum(advStats.perplexity?.perplexity, 1)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.runsTest?.randomnessScore, [0.4, 0.6], true)}">
                            <span class="stat-label">Randomness Score</span>
                            <span class="stat-value">${formatPct(advStats.runsTest?.randomnessScore)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.chiSquared?.uniformityScore, [0.7, 0.4])}">
                            <span class="stat-label">χ² Uniformity</span>
                            <span class="stat-value">${formatPct(advStats.chiSquared?.uniformityScore)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.varianceStability, [0.7, 0.5])}">
                            <span class="stat-label">Variance Stability</span>
                            <span class="stat-value">${formatPct(advStats.varianceStability)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.mahalanobisDistance, [2.0, 1.0])}">
                            <span class="stat-label">Mahalanobis Distance</span>
                            <span class="stat-value">${formatNum(advStats.mahalanobisDistance, 2)}σ</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Length Normalization</span>
                            <span class="stat-value">${formatPct(advStats.lengthNormalization)}</span>
                        </div>
                    </div>
                </div>

                <!-- Human Likelihood (Bell Curve Analysis) -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">analytics</span> Human Likelihood (Bell Curve)</h4>
                    <p class="section-note">Measures how close features are to typical human writing. Values near 1.0 = normal human range.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.overallHumanLikelihood, [0.4, 0.6], true)}">
                            <span class="stat-label"><strong>Overall Human Likelihood</strong></span>
                            <span class="stat-value"><strong>${formatPct(advStats.overallHumanLikelihood)}</strong></span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.humanLikelihood?.sentenceLengthCV, [0.4, 0.7], true)}">
                            <span class="stat-label">Sentence Length Variance</span>
                            <span class="stat-value">${formatPct(advStats.humanLikelihood?.sentenceLengthCV)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.humanLikelihood?.hapaxRatio, [0.4, 0.7], true)}">
                            <span class="stat-label">Unique Word Distribution</span>
                            <span class="stat-value">${formatPct(advStats.humanLikelihood?.hapaxRatio)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.humanLikelihood?.burstiness, [0.4, 0.7], true)}">
                            <span class="stat-label">Word Usage Burstiness</span>
                            <span class="stat-value">${formatPct(advStats.humanLikelihood?.burstiness)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.humanLikelihood?.zipfSlope, [0.4, 0.7], true)}">
                            <span class="stat-label">Zipf's Law Compliance</span>
                            <span class="stat-value">${formatPct(advStats.humanLikelihood?.zipfSlope)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.humanLikelihood?.ttr, [0.4, 0.7], true)}">
                            <span class="stat-label">Vocabulary Richness</span>
                            <span class="stat-value">${formatPct(advStats.humanLikelihood?.ttr)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.varianceNaturalness, [0.4, 0.7], true)}">
                            <span class="stat-label">Variance Naturalness</span>
                            <span class="stat-value">${formatPct(advStats.varianceNaturalness)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.extremeVarianceIndicator, [0.6, 0.4])}">
                            <span class="stat-label">Extreme Variance Warning</span>
                            <span class="stat-value">${formatPct(advStats.extremeVarianceIndicator)}</span>
                        </div>
                    </div>
                </div>

                <!-- AI Signature Metrics -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">smart_toy</span> AI Signature Metrics</h4>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.hedgingDensity, [0.02, 0.01])}">
                            <span class="stat-label">Hedging Density</span>
                            <span class="stat-value">${formatPct(advStats.aiSignatures?.hedgingDensity)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.discourseMarkerDensity, [0.4, 0.2])}">
                            <span class="stat-label">Discourse Marker Density</span>
                            <span class="stat-value">${formatNum(advStats.aiSignatures?.discourseMarkerDensity, 2)}/sentence</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.unicodeAnomalyDensity, [1, 0.3])}">
                            <span class="stat-label">Unicode Anomaly Density</span>
                            <span class="stat-value">${formatNum(advStats.aiSignatures?.unicodeAnomalyDensity, 2)}/1000 chars</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.decorativeDividerCount, [1, 0])}">
                            <span class="stat-label">Decorative Dividers</span>
                            <span class="stat-value">${advStats.aiSignatures?.decorativeDividerCount || 0}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.contractionRate, [0.3, 0.5], true)}">
                            <span class="stat-label">Contraction Rate</span>
                            <span class="stat-value">${formatNum(advStats.aiSignatures?.contractionRate, 2)}/sentence</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.aiSignatures?.sentenceStarterVariety, [0.4, 0.6], true)}">
                            <span class="stat-label">Sentence Starter Variety</span>
                            <span class="stat-value">${formatPct(advStats.aiSignatures?.sentenceStarterVariety)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Passive Voice Rate</span>
                            <span class="stat-value">${formatNum(advStats.aiSignatures?.passiveVoiceRate, 2)}/sentence</span>
                        </div>
                    </div>
                </div>

                <!-- Humanizer Detection -->
                <div class="stats-section highlight-section ${advStats.humanizerSignals?.isLikelyHumanized ? 'humanizer-warning' : ''}">
                    <h4><span class="material-icons section-icon">sync_alt</span> Humanizer Detection</h4>
                    <p class="section-note">Detects AI text that has been post-processed to evade detection.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${advStats.humanizerSignals?.isLikelyHumanized ? 'ai' : 'neutral'}">
                            <span class="stat-label"><strong>Humanizer Probability</strong></span>
                            <span class="stat-value"><strong>${formatPct(advStats.humanizerSignals?.humanizerProbability)}</strong></span>
                        </div>
                        <div class="stat-row indicator-${advStats.humanizerSignals?.stableVarianceFlag ? 'ai' : 'neutral'}">
                            <span class="stat-label">Variance Stability (2nd order)</span>
                            <span class="stat-value">${advStats.humanizerSignals?.stableVarianceFlag ? 'Suspicious' : 'Normal'}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.humanizerSignals?.flatAutocorrelationFlag ? 'ai' : 'neutral'}">
                            <span class="stat-label">Autocorrelation Pattern</span>
                            <span class="stat-value">${advStats.humanizerSignals?.flatAutocorrelationFlag ? 'Random noise' : 'Natural'}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.humanizerSignals?.brokenCorrelationFlag ? 'ai' : 'neutral'}">
                            <span class="stat-label">Feature Correlations</span>
                            <span class="stat-value">${advStats.humanizerSignals?.brokenCorrelationFlag ? 'Broken' : 'Intact'}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.humanizerSignals?.synonymSubstitutionFlag ? 'ai' : 'neutral'}">
                            <span class="stat-label">Sophistication Consistency</span>
                            <span class="stat-value">${advStats.humanizerSignals?.synonymSubstitutionFlag ? 'Word-level chaos' : 'Consistent'}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.humanizerSignals?.artificialContractionFlag ? 'ai' : 'neutral'}">
                            <span class="stat-label">Contraction Pattern</span>
                            <span class="stat-value">${advStats.humanizerSignals?.artificialContractionFlag ? 'Artificial' : 'Natural'}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Warning Flags</span>
                            <span class="stat-value">${advStats.humanizerSignals?.flagCount || 0} / 5</span>
                        </div>
                    </div>
                </div>

                <!-- V3 Advanced Detection (Neural & Perplexity) -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">psychology</span> Neural & Perplexity Analysis (V3)</h4>
                    <p class="section-note">Advanced detection methods using embedding coherence and language model perplexity.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.neuralFeatures?.embeddingCoherence, [0.85, 0.7], false)}">
                            <span class="stat-label">Embedding Coherence</span>
                            <span class="stat-value">${formatNum(advStats.neuralFeatures?.embeddingCoherence, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.neuralFeatures?.embeddingDiversity, [0.2, 0.4], true)}">
                            <span class="stat-label">Embedding Diversity</span>
                            <span class="stat-value">${formatNum(advStats.neuralFeatures?.embeddingDiversity, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.neuralFeatures?.embeddingDrift, [0.1, 0.3], true)}">
                            <span class="stat-label">Topic Drift</span>
                            <span class="stat-value">${formatNum(advStats.neuralFeatures?.embeddingDrift, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.neuralFeatures?.embeddingUniformity, [0.8, 0.5], false)}">
                            <span class="stat-label">Embedding Uniformity</span>
                            <span class="stat-value">${formatNum(advStats.neuralFeatures?.embeddingUniformity, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.perplexityFeatures?.meanPerplexity, [60, 120], true)}">
                            <span class="stat-label">Mean Perplexity</span>
                            <span class="stat-value">${formatNum(advStats.perplexityFeatures?.meanPerplexity, 1)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.perplexityFeatures?.perplexityStd, [15, 40], true)}">
                            <span class="stat-label">Perplexity Variance</span>
                            <span class="stat-value">${formatNum(advStats.perplexityFeatures?.perplexityStd, 1)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.perplexityFeatures?.lowPerplexityRatio, [0.5, 0.2], false)}">
                            <span class="stat-label">Low Perplexity Ratio</span>
                            <span class="stat-value">${formatPct(advStats.perplexityFeatures?.lowPerplexityRatio)}</span>
                        </div>
                    </div>
                </div>

                <!-- V3 Watermark & Semantic Coherence -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">security</span> Watermark & Coherence Detection (V3)</h4>
                    <p class="section-note">Detects statistical watermarks and semantic structure anomalies.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${advStats.watermarkFeatures?.zeroWidthChars > 0 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Zero-Width Characters</span>
                            <span class="stat-value">${advStats.watermarkFeatures?.zeroWidthChars || 0}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.watermarkFeatures?.repeatedTransitions, [0.3, 0.1], false)}">
                            <span class="stat-label">Token Transition Bias</span>
                            <span class="stat-value">${formatNum(advStats.watermarkFeatures?.repeatedTransitions, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.watermarkFeatures?.wordLengthAutocorr, [0.4, 0.2], false)}">
                            <span class="stat-label">Word Length Autocorrelation</span>
                            <span class="stat-value">${formatNum(advStats.watermarkFeatures?.wordLengthAutocorr, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.semanticCoherence?.topicDrift, [0.1, 0.3], true)}">
                            <span class="stat-label">Semantic Topic Drift</span>
                            <span class="stat-value">${formatNum(advStats.semanticCoherence?.topicDrift, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.semanticCoherence?.coherenceVariance, [0.1, 0.25], true)}">
                            <span class="stat-label">Coherence Variance</span>
                            <span class="stat-value">${formatNum(advStats.semanticCoherence?.coherenceVariance, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.semanticCoherence?.semanticGaps, [0.1, 0.25], true)}">
                            <span class="stat-label">Semantic Gap Ratio</span>
                            <span class="stat-value">${formatPct(advStats.semanticCoherence?.semanticGaps)}</span>
                        </div>
                    </div>
                </div>

                <!-- V3 Null Combination Patterns -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">compare</span> Null Combination Patterns (V3)</h4>
                    <p class="section-note">Feature absence patterns for class discrimination (A ∧ ¬B logic).</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${advStats.nullPatterns?.formalNoContractions > 0.5 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Formal + No Contractions</span>
                            <span class="stat-value">${formatNum(advStats.nullPatterns?.formalNoContractions, 2)} <small>(AI signal)</small></span>
                        </div>
                        <div class="stat-row indicator-${advStats.nullPatterns?.aiPhrasesNoDisfluencies > 0.5 ? 'ai' : 'neutral'}">
                            <span class="stat-label">AI Phrases + No Disfluencies</span>
                            <span class="stat-value">${formatNum(advStats.nullPatterns?.aiPhrasesNoDisfluencies, 2)} <small>(AI signal)</small></span>
                        </div>
                        <div class="stat-row indicator-${advStats.nullPatterns?.contractionsNoAI > 0.5 ? 'human' : 'neutral'}">
                            <span class="stat-label">Contractions + No AI Phrases</span>
                            <span class="stat-value">${formatNum(advStats.nullPatterns?.contractionsNoAI, 2)} <small>(Human signal)</small></span>
                        </div>
                        <div class="stat-row indicator-${advStats.nullPatterns?.pureAINullScore > 0.7 ? 'ai' : 'neutral'}">
                            <span class="stat-label"><strong>Pure AI Null Score</strong></span>
                            <span class="stat-value"><strong>${formatNum(advStats.nullPatterns?.pureAINullScore, 2)}</strong></span>
                        </div>
                        <div class="stat-row indicator-${advStats.nullPatterns?.pureHumanNullScore > 0.5 ? 'human' : 'neutral'}">
                            <span class="stat-label"><strong>Pure Human Null Score</strong></span>
                            <span class="stat-value"><strong>${formatNum(advStats.nullPatterns?.pureHumanNullScore, 2)}</strong></span>
                        </div>
                        <div class="stat-row indicator-${advStats.nullPatterns?.humanizedNullScore > 0.5 ? 'ai' : 'neutral'}">
                            <span class="stat-label"><strong>Humanized Null Score</strong></span>
                            <span class="stat-value"><strong>${formatNum(advStats.nullPatterns?.humanizedNullScore, 2)}</strong> <small>(Mixed signals)</small></span>
                        </div>
                    </div>
                </div>

                <!-- V3 Partial Humanization Detection -->
                <div class="stats-section highlight-section">
                    <h4><span class="material-icons section-icon">tune</span> Partial Humanization Analysis (V3)</h4>
                    <p class="section-note">Detects selectively edited AI content with mixed authorship.</p>
                    <div class="stats-table">
                        <div class="stat-row indicator-${getIndicator(advStats.partialHumanization?.humanizationVariance, [0.3, 0.1], false)}">
                            <span class="stat-label">Humanization Variance</span>
                            <span class="stat-value">${formatNum(advStats.partialHumanization?.humanizationVariance, 3)}</span>
                        </div>
                        <div class="stat-row indicator-${getIndicator(advStats.partialHumanization?.segmentInconsistency, [0.3, 0.15], false)}">
                            <span class="stat-label">Segment Inconsistency</span>
                            <span class="stat-value">${formatNum(advStats.partialHumanization?.segmentInconsistency, 3)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">AI Segment Ratio</span>
                            <span class="stat-value">${formatPct(advStats.partialHumanization?.aiSegmentRatio)}</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Human Segment Ratio</span>
                            <span class="stat-value">${formatPct(advStats.partialHumanization?.humanSegmentRatio)}</span>
                        </div>
                        <div class="stat-row indicator-${advStats.partialHumanization?.mixedSegmentRatio > 0.3 ? 'ai' : 'neutral'}">
                            <span class="stat-label">Mixed Segment Ratio</span>
                            <span class="stat-value">${formatPct(advStats.partialHumanization?.mixedSegmentRatio)}</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Detection Transparency: Show the math behind the prediction -->
            ${this.renderDetectionTransparency(fullResult, formatNum, formatPct)}

            <div class="stats-legend">
                <h5>Indicator Key</h5>
                <div class="legend-items">
                    <span class="legend-item indicator-ai">Suggests AI</span>
                    <span class="legend-item indicator-human">Suggests Human</span>
                    <span class="legend-item indicator-neutral">Neutral</span>
                </div>
            </div>
        `;

        container.innerHTML = html;
    },

    /**
     * Render Detection Transparency section
     * Shows the actual math and feature values leading to the prediction
     */
    renderDetectionTransparency(result, formatNum, formatPct) {
        if (!result) return '';
        
        const featureContributions = result.featureContributions || [];
        const signalCounts = result.varianceProfile?.signalCounts || 
                            (result.aiProbability !== undefined ? { strongAi: 0, strongHuman: 0 } : null);
        
        // Get the config for explanations
        const config = typeof VERITAS_SUNRISE_CONFIG !== 'undefined' ? VERITAS_SUNRISE_CONFIG : null;
        const featureExplanations = config?.featureExplanations || {};
        const expectedRanges = config?.expectedRanges || {};
        const humanizationIndicators = config?.humanizationIndicators || {};
        
        // Calculate totals for transparency
        const totalWeight = featureContributions.reduce((sum, f) => sum + (f.weight || 0), 0);
        const totalContribution = featureContributions.reduce((sum, f) => sum + (f.contribution || 0), 0);
        const baseWeightedAvg = totalWeight > 0 ? totalContribution / totalWeight : 0.5;
        
        // Build feature contribution rows
        const topContributors = featureContributions.slice(0, 10); // Top 10 contributors
        const contributorRows = topContributors.map(f => {
            const indicator = f.aiProbability > 0.6 ? 'ai' : (f.aiProbability < 0.4 ? 'human' : 'neutral');
            const explanation = featureExplanations[f.name.toLowerCase().replace(/\s+/g, '_')] || '';
            const percentOfTotal = totalContribution > 0 ? (f.contribution / totalContribution * 100).toFixed(1) : 0;
            
            return `
                <div class="stat-row indicator-${indicator}">
                    <span class="stat-label" title="${explanation}">${f.name}</span>
                    <span class="stat-value">
                        <span class="feature-prob">${formatNum(f.aiProbability, 2)}</span>
                        <span class="feature-weight">× ${formatNum(f.weight, 3)}</span>
                        <span class="feature-contribution">= ${formatNum(f.contribution, 3)}</span>
                        <span class="feature-percent">(${percentOfTotal}%)</span>
                    </span>
                </div>
            `;
        }).join('');
        
        // Build humanization indicator rows if available
        let humanizationRows = '';
        if (result.humanizerSignals) {
            const signals = result.humanizerSignals;
            const indicators = [
                { name: 'Artificial Variance', flag: signals.stableVarianceFlag, desc: 'AI has unnaturally consistent sentence lengths' },
                { name: 'Flat Autocorrelation', flag: signals.flatAutocorrelationFlag, desc: 'Sentence patterns lack natural flow' },
                { name: 'Broken Correlations', flag: signals.brokenCorrelationFlag, desc: 'Features that should correlate don\'t' },
                { name: 'Synonym Substitution', flag: signals.synonymSubstitutionFlag, desc: 'Word-level edits break consistency' },
                { name: 'Artificial Contractions', flag: signals.artificialContractionFlag, desc: 'Contractions added post-hoc' }
            ];
            
            const flagged = indicators.filter(i => i.flag);
            if (flagged.length > 0) {
                humanizationRows = `
                    <div class="stats-section highlight-section warning-section">
                        <h4><span class="material-icons section-icon">warning</span> Humanization Flags Detected</h4>
                        <p class="section-note">These patterns suggest AI text may have been modified by humanizer tools. This is an <strong>advisory indicator</strong>, not a definitive classification.</p>
                        <div class="stats-table">
                            ${flagged.map(i => `
                                <div class="stat-row indicator-ai">
                                    <span class="stat-label" title="${i.desc}">${i.name}</span>
                                    <span class="stat-value flagged"><span class="material-icons" style="font-size:14px">warning</span> Flagged</span>
                                </div>
                            `).join('')}
                            <div class="stat-row">
                                <span class="stat-label"><strong>Total Flags</strong></span>
                                <span class="stat-value"><strong>${signals.flagCount || 0} / 5</strong></span>
                            </div>
                        </div>
                    </div>
                `;
            }
        }
        
        // Flagged characteristics from category results
        let flaggedCharacteristics = '';
        if (result.categoryResults) {
            const aiFindings = result.categoryResults
                .flatMap(c => (c.findings || []))
                .filter(f => f.indicator === 'ai' && f.text && f.text !== 'undefined')
                .slice(0, 8);
            
            const humanFindings = result.categoryResults
                .flatMap(c => (c.findings || []))
                .filter(f => f.indicator === 'human' && f.text && f.text !== 'undefined')
                .slice(0, 5);
            
            if (aiFindings.length > 0 || humanFindings.length > 0) {
                flaggedCharacteristics = `
                    <div class="stats-section highlight-section">
                        <h4><span class="material-icons section-icon">flag</span> Flagged Characteristics</h4>
                        <p class="section-note">Specific patterns detected that influenced the classification.</p>
                        <div class="flagged-findings">
                            ${aiFindings.length > 0 ? `
                                <div class="findings-group ai-findings">
                                    <h5 class="findings-title indicator-ai">AI Indicators</h5>
                                    <ul class="findings-list">
                                        ${aiFindings.map(f => `<li>${f.text}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            ${humanFindings.length > 0 ? `
                                <div class="findings-group human-findings">
                                    <h5 class="findings-title indicator-human">Human Indicators</h5>
                                    <ul class="findings-list">
                                        ${humanFindings.map(f => `<li>${f.text}</li>`).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
        }
        
        return `
            <div class="stats-section highlight-section transparency-section">
                <h4><span class="material-icons section-icon">calculate</span> Detection Transparency</h4>
                <p class="section-note">The actual math behind the prediction. Each analyzer contributes a probability (0-1) multiplied by its weight. The final score combines these using Bayesian methods.</p>
                
                <!-- Final Calculation Summary -->
                <div class="calculation-summary">
                    <div class="calc-row">
                        <span class="calc-label">Weighted Average (Σ prob × weight / Σ weight)</span>
                        <span class="calc-value">${formatNum(baseWeightedAvg, 3)}</span>
                    </div>
                    <div class="calc-row">
                        <span class="calc-label">Strong AI Signals</span>
                        <span class="calc-value">${signalCounts?.strongAi || 0} analyzers</span>
                    </div>
                    <div class="calc-row">
                        <span class="calc-label">Strong Human Signals</span>
                        <span class="calc-value">${signalCounts?.strongHuman || 0} analyzers</span>
                    </div>
                    <div class="calc-row final-score">
                        <span class="calc-label"><strong>Final AI Probability</strong></span>
                        <span class="calc-value"><strong>${formatPct(result.aiProbability)}</strong></span>
                    </div>
                </div>
                
                <!-- Feature Contributions Table -->
                <h5 class="sub-section-title">Top Feature Contributions</h5>
                <p class="section-note">Each row: Analyzer → AI Probability × Weight = Contribution (% of total)</p>
                <div class="stats-table feature-table">
                    ${contributorRows}
                    <div class="stat-row total-row">
                        <span class="stat-label"><strong>Total</strong></span>
                        <span class="stat-value"><strong>Σ = ${formatNum(totalContribution, 3)}</strong> (÷ ${formatNum(totalWeight, 3)} = ${formatPct(baseWeightedAvg)})</span>
                    </div>
                </div>
            </div>
            
            ${humanizationRows}
            ${flaggedCharacteristics}
        `;
    },

    /**
     * Render confidence interval visualization
     */
    renderConfidenceInterval(result) {
        const confidenceContainer = document.querySelector('.confidence');
        if (!confidenceContainer || !result.confidenceInterval) return;

        // Remove existing interval display
        const existing = confidenceContainer.querySelector('.confidence-interval');
        if (existing) existing.remove();

        const ci = result.confidenceInterval;
        const intervalHtml = `
            <div class="confidence-interval">
                <span class="ci-label">${Math.round(ci.lower * 100)}%</span>
                <div class="confidence-interval-bar">
                    <div class="confidence-interval-range" 
                         style="left: ${ci.lower * 100}%; width: ${(ci.upper - ci.lower) * 100}%"></div>
                    <div class="confidence-interval-center" 
                         style="left: ${result.aiProbability * 100}%"></div>
                </div>
                <span class="ci-label">${Math.round(ci.upper * 100)}%</span>
            </div>
        `;
        confidenceContainer.insertAdjacentHTML('beforeend', intervalHtml);
    },

    /**
     * Render false positive warnings
     */
    renderFalsePositiveWarnings(result) {
        const scoreCard = document.querySelector('.score-card');
        if (!scoreCard || !result.falsePositiveRisk?.hasRisks) return;

        // Remove existing warnings
        const existing = scoreCard.querySelector('.false-positive-warning');
        if (existing) existing.remove();

        const risks = result.falsePositiveRisk.risks;
        const warningHtml = `
            <div class="false-positive-warning">
                <div class="false-positive-warning-title">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z"/></svg>
                    Detection Caveats
                </div>
                <ul class="false-positive-warning-list">
                    ${risks.map(r => `<li>${r.message}</li>`).join('')}
                </ul>
            </div>
        `;
        scoreCard.insertAdjacentHTML('beforeend', warningHtml);
        
        // Add domain awareness note
        this.renderDomainAwarenessNote(scoreCard);
    },

    /**
     * Render domain awareness disclaimer
     * Research shows detection accuracy varies significantly by text domain
     */
    renderDomainAwarenessNote(container) {
        // Don't duplicate
        if (container.querySelector('.domain-awareness-note')) return;
        
        const noteHtml = `
            <div class="domain-awareness-note">
                <div class="note-title"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z"/></svg> Domain Notice</div>
                <p>Detection accuracy varies by text type. Academic, creative, and technical writing 
                may produce different results. No detector is 100% reliable—use as one data point, 
                not as definitive proof.</p>
            </div>
        `;
        container.insertAdjacentHTML('beforeend', noteHtml);
    },
    
    /**
     * Render model transparency header - shows which model is being used and how it works
     */
    renderModelTransparencyHeader() {
        const container = document.getElementById('modelTransparencyHeader');
        if (!container) return;
        
        const currentModel = this.getCurrentModel();
        
        const modelInfo = {
            helios: {
                name: 'Helios',
                icon: 'flare',
                version: 'v2.1 Flagship',
                accuracy: '99.24%',
                rocAuc: '99.98%',
                method: 'Comprehensive multi-analyzer fusion with 45 linguistic features. Combines lexical, syntactic, semantic, and statistical analysis using weighted Bayesian inference. Includes tone analysis, hedging detection, and second-order pattern recognition for humanized text detection.'
            },
            zenith: {
                name: 'Zenith',
                icon: 'brightness_high',
                version: 'v2.0 Perplexity-Based',
                accuracy: '99.57%',
                rocAuc: '99.72%',
                method: 'Entropy and perplexity-focused detection emphasizing burstiness patterns and information-theoretic measures. Optimized for detecting AI text that has been processed through humanization tools. Achieves 86.7% detection rate on bypassed content.'
            },
            sunrise: {
                name: 'Sunrise',
                icon: 'wb_sunny',
                version: 'v1.5 Balanced',
                accuracy: '98.08%',
                rocAuc: '98.45%',
                method: 'Balanced statistical variance analysis combining sentence structure patterns with vocabulary distribution metrics. Fast processing with well-rounded detection across multiple writing domains. F1 score: 98.09%.'
            },
            dawn: {
                name: 'Dawn',
                icon: 'wb_twilight',
                version: 'v1.0 Legacy',
                accuracy: '84.9%',
                rocAuc: '87.2%',
                method: 'Rule-based heuristic detection using foundational linguistic patterns. Lightweight and resource-efficient. Serves as a baseline detector and is useful for quick initial screening of content.'
            }
        };
        
        const info = modelInfo[currentModel] || modelInfo.helios;
        
        container.innerHTML = `
            <div class="model-transparency-content">
                <div class="model-identity">
                    <div class="model-identity-icon">
                        <span class="material-icons">${info.icon}</span>
                    </div>
                    <div class="model-identity-info">
                        <h4>${info.name}</h4>
                        <span class="model-version">${info.version}</span>
                    </div>
                </div>
                <div class="model-methodology">
                    <strong>Detection Method:</strong> ${info.method}
                </div>
                <div class="model-stats">
                    <div class="model-stat">
                        <div class="model-stat-value">${info.accuracy}</div>
                        <div class="model-stat-label">Accuracy</div>
                    </div>
                    <div class="model-stat">
                        <div class="model-stat-value">${info.rocAuc}</div>
                        <div class="model-stat-label">ROC-AUC</div>
                    </div>
                </div>
            </div>
        `;
    },
    
    /**
     * Render simplified probability bar - cleaner than certainty curve
     */
    renderProbabilityBar(result) {
        const container = document.getElementById('probabilityBar');
        if (!container) return;
        
        const prob = result.aiProbability;
        const ci = result.confidenceInterval || { lower: Math.max(0, prob - 0.1), upper: Math.min(1, prob + 0.1) };
        
        const humanPercent = Math.round((1 - prob) * 100);
        const aiPercent = Math.round(prob * 100);
        
        container.innerHTML = `
            <div class="probability-bar">
                <div class="probability-bar-header">
                    <span>Probability Spectrum</span>
                </div>
                <div class="probability-track">
                    <div class="probability-interval" style="left: ${ci.lower * 100}%; width: ${(ci.upper - ci.lower) * 100}%;"></div>
                    <div class="probability-marker" style="left: ${prob * 100}%;"></div>
                </div>
                <div class="probability-labels">
                    <span class="probability-label human">${humanPercent}% Human</span>
                    <span class="probability-label ai">${aiPercent}% AI</span>
                </div>
                <div class="probability-percentage">
                    Confidence interval: ${Math.round(ci.lower * 100)}% — ${Math.round(ci.upper * 100)}%
                </div>
            </div>
        `;
    },

    /**
     * Render high severity alerts section
     * Shows critical warnings that need attention with full context
     */
    renderHighSeverityAlerts(result) {
        const container = document.getElementById('highSeverityAlerts');
        if (!container) return;
        
        // Collect all high severity findings from all sources
        const highSeverityFindings = [];
        
        // From main findings
        if (result.findings && result.findings.length > 0) {
            result.findings.forEach((finding, idx) => {
                if (finding.severity === 'high' || finding.severity === 'critical') {
                    highSeverityFindings.push({
                        ...finding,
                        source: 'analysis',
                        sourceLabel: finding.label || 'Detection Analysis'
                    });
                }
            });
        }
        
        // From false positive risks
        if (result.falsePositiveRisk && result.falsePositiveRisk.risks) {
            result.falsePositiveRisk.risks.forEach(risk => {
                if (risk.severity === 'high' || risk.severity === 'critical') {
                    highSeverityFindings.push({
                        label: risk.type.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
                        value: risk.message,
                        note: risk.suggestsHumanized ? 'This pattern is commonly seen in humanized AI text' : 'This may affect detection accuracy',
                        severity: risk.severity,
                        source: 'risk',
                        sourceLabel: 'Risk Assessment'
                    });
                }
            });
        }
        
        // If no high severity alerts, hide the container
        if (highSeverityFindings.length === 0) {
            container.style.display = 'none';
            return;
        }
        
        container.style.display = 'block';
        
        // Severity explanations
        const severityInfo = {
            critical: { icon: 'error', label: 'Critical', color: '#dc2626' },
            high: { icon: 'warning', label: 'High Priority', color: '#ea580c' }
        };
        
        // Indicator info for context
        const indicatorInfo = {
            ai: { label: 'AI Pattern', class: 'ai' },
            human: { label: 'Human Pattern', class: 'human' },
            mixed: { label: 'Mixed Signal', class: 'mixed' },
            neutral: { label: 'Neutral', class: 'neutral' }
        };
        
        const alertsHTML = highSeverityFindings.map((finding, index) => {
            const sevInfo = severityInfo[finding.severity] || severityInfo.high;
            const indInfo = indicatorInfo[finding.indicator] || indicatorInfo.neutral;
            
            // Build stats display if available
            let statsHTML = '';
            if (finding.stats) {
                const statItems = Object.entries(finding.stats)
                    .filter(([k, v]) => v !== undefined && v !== null)
                    .slice(0, 4) // Limit to 4 stats
                    .map(([key, val]) => `<span class="alert-stat"><strong>${key.replace(/([A-Z])/g, ' $1').trim()}:</strong> ${val}</span>`)
                    .join('');
                if (statItems) {
                    statsHTML = `<div class="alert-stats">${statItems}</div>`;
                }
            }
            
            // Build benchmark display if available
            let benchmarkHTML = '';
            if (finding.benchmark) {
                benchmarkHTML = `
                    <div class="alert-benchmark">
                        <span class="benchmark-label">Expected Ranges:</span>
                        ${finding.benchmark.humanRange ? `<span class="benchmark-item human">Human: ${finding.benchmark.humanRange}</span>` : ''}
                        ${finding.benchmark.aiRange ? `<span class="benchmark-item ai">AI: ${finding.benchmark.aiRange}</span>` : ''}
                        ${finding.benchmark.interpretation ? `<span class="benchmark-interpretation">${finding.benchmark.interpretation}</span>` : ''}
                    </div>
                `;
            }
            
            return `
                <div class="high-severity-alert ${finding.severity}">
                    <div class="alert-header">
                        <div class="alert-icon-wrapper">
                            <span class="material-icons alert-icon">${sevInfo.icon}</span>
                        </div>
                        <div class="alert-title-section">
                            <span class="alert-label">${finding.label || 'Detection Alert'}</span>
                            <span class="alert-indicator ${indInfo.class}">${indInfo.label}</span>
                        </div>
                        <span class="alert-severity-badge">${sevInfo.label}</span>
                    </div>
                    <div class="alert-body">
                        <div class="alert-main-message">${finding.value || finding.note || 'High severity pattern detected'}</div>
                        ${finding.note && finding.value ? `<div class="alert-explanation">${finding.note}</div>` : ''}
                        ${statsHTML}
                        ${benchmarkHTML}
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = `
            <div class="high-severity-section">
                <div class="high-severity-header">
                    <span class="material-icons">priority_high</span>
                    <h3>Critical Findings <span class="alert-count">${highSeverityFindings.length}</span></h3>
                </div>
                <p class="high-severity-subtitle">These patterns require special attention and may significantly impact the analysis</p>
                <div class="high-severity-alerts-list">
                    ${alertsHTML}
                </div>
            </div>
        `;
    },

    /**
     * Render certainty curve visualization (legacy - kept for reference)
     * Shows where the detection result falls on a probability spectrum
     */
    renderCertaintyCurve(result) {
        const container = document.getElementById('certaintyCurve');
        if (!container) return;

        const prob = result.aiProbability;
        const conf = result.confidence;
        const ci = result.confidenceInterval || { lower: prob - 0.1, upper: prob + 0.1 };
        
        // Determine which zone the result falls in
        const zones = [
            { min: 0, max: 0.15, label: 'Definitely Human', color: 'var(--human-color)', description: 'Strong human writing patterns' },
            { min: 0.15, max: 0.30, label: 'Likely Human', color: 'var(--human-color-light)', description: 'Predominantly human patterns' },
            { min: 0.30, max: 0.45, label: 'Possibly Human', color: 'var(--neutral-color)', description: 'Mostly human with some uncertainty' },
            { min: 0.45, max: 0.55, label: 'Inconclusive', color: 'var(--gray-400)', description: 'Mixed signals, could be either' },
            { min: 0.55, max: 0.70, label: 'Possibly AI', color: 'var(--neutral-color)', description: 'Some AI patterns detected' },
            { min: 0.70, max: 0.85, label: 'Likely AI', color: 'var(--ai-color-light)', description: 'Strong AI patterns present' },
            { min: 0.85, max: 1.0, label: 'Definitely AI', color: 'var(--ai-color)', description: 'Overwhelming AI signatures' }
        ];

        const currentZone = zones.find(z => prob >= z.min && prob < z.max) || zones[zones.length - 1];
        
        container.innerHTML = `
            <div class="certainty-curve">
                <h4 class="certainty-title">Certainty Spectrum</h4>
                <div class="curve-container">
                    <div class="curve-gradient">
                        ${zones.map(z => `
                            <div class="curve-zone ${prob >= z.min && prob < z.max ? 'active' : ''}" 
                                 style="left: ${z.min * 100}%; width: ${(z.max - z.min) * 100}%;"
                                 title="${z.label}: ${z.description}">
                            </div>
                        `).join('')}
                        <div class="curve-marker" style="left: ${prob * 100}%">
                            <div class="marker-dot"></div>
                            <div class="marker-label">${Math.round(prob * 100)}%</div>
                        </div>
                        <div class="curve-interval" style="left: ${Math.max(0, ci.lower) * 100}%; width: ${Math.min(1, ci.upper - ci.lower) * 100}%"></div>
                    </div>
                    <div class="curve-labels">
                        <span class="curve-label-human">Human</span>
                        <span class="curve-label-uncertain">Uncertain</span>
                        <span class="curve-label-ai">AI</span>
                    </div>
                </div>
                <div class="curve-zone-indicator">
                    <span class="zone-badge" style="background: ${currentZone.color}">${currentZone.label}</span>
                    <span class="zone-description">${currentZone.description}</span>
                </div>
            </div>
        `;
    },

    /**
     * Render verbose conclusion with detailed analysis explanation
     */
    renderVerboseConclusion(result) {
        const container = document.getElementById('verboseConclusion');
        if (!container) return;

        const prob = result.aiProbability;
        const conf = result.confidence;
        const stats = result.stats || {};
        const featureContributions = result.featureContributions || [];
        const categoryResults = result.categoryResults || [];
        
        // Get top contributing factors
        const topAiFactors = featureContributions
            .filter(f => f.aiProbability > 0.6)
            .slice(0, 3)
            .map(f => f.name);
        const topHumanFactors = featureContributions
            .filter(f => f.aiProbability < 0.4)
            .slice(0, 3)
            .map(f => f.name);
        
        // Count analyzer agreement
        const aiLeaning = categoryResults.filter(c => c.aiProbability > 0.55).length;
        const humanLeaning = categoryResults.filter(c => c.aiProbability < 0.45).length;
        const totalAnalyzers = categoryResults.length;
        
        // Build detailed explanation
        let explanation = '';
        let confidenceNote = '';
        
        if (prob < 0.30) {
            explanation = `This text demonstrates characteristics strongly consistent with human authorship. `;
            if (topHumanFactors.length > 0) {
                explanation += `Key human indicators include: ${topHumanFactors.join(', ')}. `;
            }
            explanation += `The writing exhibits natural variation in sentence structure, authentic vocabulary choices, and organic flow patterns typical of human expression. `;
            if (humanLeaning > aiLeaning) {
                explanation += `${humanLeaning} out of ${totalAnalyzers} analysis modules agreed this appears human-written.`;
            }
        } else if (prob < 0.55) {
            explanation = `This text shows mixed characteristics that make definitive classification challenging. `;
            explanation += `While some patterns suggest human authorship (${topHumanFactors.join(', ') || 'natural flow'}), `;
            explanation += `other elements could indicate AI involvement (${topAiFactors.join(', ') || 'structural uniformity'}). `;
            explanation += `The analysis modules were split: ${humanLeaning} leaning human, ${aiLeaning} leaning AI, and ${totalAnalyzers - humanLeaning - aiLeaning} neutral. `;
            explanation += `This could indicate human writing with unusual patterns, AI text that was heavily edited, or collaborative human-AI content.`;
        } else if (prob < 0.75) {
            explanation = `This text exhibits notable patterns commonly associated with AI-generated content. `;
            if (topAiFactors.length > 0) {
                explanation += `Detected AI indicators include: ${topAiFactors.join(', ')}. `;
            }
            explanation += `However, some human-like elements are present, suggesting either the base AI output has been edited, or the human author's style happens to align with AI patterns. `;
            explanation += `${aiLeaning} out of ${totalAnalyzers} analysis modules flagged AI characteristics.`;
        } else {
            explanation = `This text displays strong characteristics typical of AI-generated content. `;
            if (topAiFactors.length > 0) {
                explanation += `Primary AI signatures detected: ${topAiFactors.join(', ')}. `;
            }
            explanation += `The writing shows patterns of uniform sentence construction, predictable vocabulary distribution, and structural regularity commonly observed in large language model outputs. `;
            explanation += `${aiLeaning} out of ${totalAnalyzers} analysis modules identified AI patterns with high confidence.`;
        }
        
        // Confidence qualifier
        if (conf < 0.4) {
            confidenceNote = `<span class="material-icons conf-icon">warning</span> <strong>Low Confidence:</strong> The analysis has limited certainty due to short text length, unusual writing style, or conflicting signals. Consider this result as indicative rather than definitive.`;
        } else if (conf < 0.7) {
            confidenceNote = `<span class="material-icons conf-icon">bar_chart</span> <strong>Moderate Confidence:</strong> The analysis has reasonable certainty, though some factors introduce ambiguity. The true origin is likely within the stated probability range.`;
        } else {
            confidenceNote = `<span class="material-icons conf-icon">check_circle</span> <strong>High Confidence:</strong> Multiple analyzers strongly agree on this assessment. The detected patterns are consistent and clear.`;
        }
        
        // Statistics summary
        const statsSummary = `
            <div class="conclusion-stats">
                <span class="stat-item"><strong>${stats.words || 0}</strong> words analyzed</span>
                <span class="stat-item"><strong>${stats.sentences || 0}</strong> sentences</span>
                <span class="stat-item"><strong>${totalAnalyzers}</strong> analysis modules</span>
                <span class="stat-item"><strong>${result.analysisTime || '?ms'}</strong> processing time</span>
            </div>
        `;
        
        container.innerHTML = `
            <div class="verbose-conclusion-content">
                <h4 class="conclusion-title"><span class="material-icons">assignment</span> Detailed Analysis</h4>
                ${statsSummary}
                <div class="conclusion-explanation">
                    <p>${explanation}</p>
                </div>
                <div class="conclusion-confidence">
                    <p>${confidenceNote}</p>
                </div>
                <div class="conclusion-methodology">
                    <details>
                        <summary>How was this determined?</summary>
                        <p>Veritas analyzes text using ${totalAnalyzers} specialized detection modules examining:</p>
                        <ul>
                            <li><strong>Lexical patterns:</strong> Vocabulary richness, word choice distribution, repetition</li>
                            <li><strong>Syntactic structure:</strong> Sentence length variation, complexity, grammar patterns</li>
                            <li><strong>Semantic coherence:</strong> Topic flow, logical connections, idea development</li>
                            <li><strong>Statistical signatures:</strong> Zipf's law deviation, burstiness, entropy measures</li>
                            <li><strong>Stylistic markers:</strong> Punctuation use, contractions, discourse markers</li>
                            <li><strong>Second-order patterns:</strong> Contradiction detection, humanization signals</li>
                        </ul>
                        <p>Each module produces a probability score, which are combined using weighted Bayesian inference based on module confidence and historical accuracy.</p>
                    </details>
                </div>
            </div>
        `;
    },

    /**
     * Render humanization advisory panel
     * Shows whether the text might be AI-generated but modified to appear human
     */
    renderHumanizationAdvisory(result) {
        const container = document.getElementById('humanizationAdvisory');
        if (!container) return;

        const humanizerSignals = result.humanizerSignals || {};
        const flagCount = humanizerSignals.flagCount || 0;
        const prob = result.aiProbability;
        
        // Check if analyzer disagreement suggests humanization
        const analyzerDisagreement = result.falsePositiveRisk?.risks?.some(r => r.suggestsHumanized) || false;
        
        // Calculate humanization likelihood
        const signals = {
            stableVariance: humanizerSignals.stableVarianceFlag,
            flatAutocorrelation: humanizerSignals.flatAutocorrelationFlag,
            brokenCorrelation: humanizerSignals.brokenCorrelationFlag,
            synonymSubstitution: humanizerSignals.synonymSubstitutionFlag,
            artificialContraction: humanizerSignals.artificialContractionFlag
        };
        
        const activeSignals = Object.entries(signals).filter(([k, v]) => v);
        
        // Effective flag count includes analyzer disagreement as an additional signal
        const effectiveFlagCount = analyzerDisagreement ? Math.max(flagCount, 2) : flagCount;
        
        // Determine advisory level
        let advisoryLevel = 'none';
        let advisoryColor = 'var(--text-tertiary)';
        let advisoryText = '';
        let advisoryIcon = 'check';
        
        if (effectiveFlagCount === 0 && prob < 0.4) {
            advisoryLevel = 'none';
            advisoryIcon = 'verified_user';
            advisoryColor = 'var(--human-color)';
            advisoryText = 'No humanization signals detected. This text appears to be authentically human-written.';
        } else if (effectiveFlagCount === 0 && prob >= 0.4) {
            advisoryLevel = 'none';
            advisoryIcon = 'smart_toy';
            advisoryColor = 'var(--ai-color)';
            advisoryText = 'No humanization signals detected. This appears to be unmodified AI-generated text.';
        } else if (analyzerDisagreement && flagCount === 0) {
            // Special case: analyzer disagreement but no explicit humanization signals
            advisoryLevel = 'possible';
            advisoryIcon = 'help_outline';
            advisoryColor = 'var(--warning-color, #f59e0b)';
            advisoryText = 'Inconsistent detection patterns observed. This could indicate humanized AI text, mixed authorship, or an unusual writing style.';
        } else if (effectiveFlagCount <= 2) {
            advisoryLevel = 'possible';
            advisoryIcon = 'search';
            advisoryColor = 'var(--warning-color, #f59e0b)';
            advisoryText = `Possible humanization detected (${flagCount}/5 signals). This could be AI text that was lightly edited or run through a paraphrasing tool.`;
        } else if (effectiveFlagCount <= 3) {
            advisoryLevel = 'likely';
            advisoryIcon = 'warning';
            advisoryColor = 'var(--warning-color, #f59e0b)';
            advisoryText = `Likely humanization detected (${flagCount}/5 signals). Strong indicators suggest this text originated from AI but was modified to appear more human.`;
        } else {
            advisoryLevel = 'confident';
            advisoryIcon = 'report';
            advisoryColor = 'var(--ai-color)';
            advisoryText = `High confidence humanization (${flagCount}/5 signals). Multiple clear indicators that AI-generated text was processed through humanization tools or extensive manual editing.`;
        }
        
        // Signal explanations
        const signalExplanations = {
            stableVariance: { name: 'Artificial Variance', desc: 'Sentence lengths are too uniform, lacking natural human variation' },
            flatAutocorrelation: { name: 'Flat Autocorrelation', desc: 'Sequential sentence patterns show random noise instead of natural flow' },
            brokenCorrelation: { name: 'Broken Correlations', desc: 'Features that normally correlate in human writing are disconnected' },
            synonymSubstitution: { name: 'Synonym Substitution', desc: 'Word sophistication varies chaotically, suggesting find-and-replace editing' },
            artificialContraction: { name: 'Artificial Contractions', desc: 'Contraction usage patterns suggest they were added artificially' }
        };
        
        const signalDetails = activeSignals.length > 0 ? `
            <div class="signal-details">
                <h5>Detected Signals:</h5>
                <ul class="signal-list">
                    ${activeSignals.map(([key, _]) => {
                        const info = signalExplanations[key];
                        return `<li><strong>${info.name}:</strong> ${info.desc}</li>`;
                    }).join('')}
                </ul>
            </div>
        ` : '';
        
        // Add analyzer disagreement to displayed signals if applicable
        const displayedSignals = analyzerDisagreement && flagCount === 0 ? 
            [...activeSignals, ['analyzerDisagreement', true]] : activeSignals;
        
        const allSignalExplanations = {
            ...signalExplanations,
            analyzerDisagreement: { name: 'Detection Inconsistency', desc: 'Different analysis methods produced conflicting results' }
        };
        
        const signalDetailsHtml = displayedSignals.length > 0 ? `
            <div class="signal-details">
                <h5>Detected Signals:</h5>
                <ul class="signal-list">
                    ${displayedSignals.map(([key, _]) => {
                        const info = allSignalExplanations[key];
                        return info ? `<li><strong>${info.name}:</strong> ${info.desc}</li>` : '';
                    }).join('')}
                </ul>
            </div>
        ` : '';
        
        container.innerHTML = `
            <div class="humanization-advisory-content ${advisoryLevel}">
                <h4 class="advisory-title">
                    <span class="material-icons advisory-icon" style="color: ${advisoryColor}">${advisoryIcon}</span>
                    Humanization Advisory
                </h4>
                <p class="advisory-text" style="border-left-color: ${advisoryColor}">${advisoryText}</p>
                ${signalDetailsHtml}
                <p class="advisory-disclaimer">
                    <em>Note: This is an advisory indicator, not a definitive classification. Some human-written text may trigger false positives, 
                    and sophisticated humanization may evade detection.</em>
                </p>
            </div>
        `;
    },

    /**
     * Handle file upload with format detection
     */
    async handleFileUpload(event, fileType) {
        const file = event.target.files[0];
        if (!file) return;

        // Check file size (max 5MB for PDFs/DOCX, 1MB for subtitles, 100KB for txt)
        const sizeMap = { 'txt': 100000, 'subtitle': 1000000, 'pdf': 5000000, 'docx': 5000000 };
        const maxSize = sizeMap[fileType] || 5000000;
        if (file.size > maxSize) {
            this.showToast(`File too large. Maximum size is ${maxSize / 1000000}MB`, 'error');
            return;
        }

        this.showToast('Processing file...', 'info');

        try {
            // Check if FileParser is available
            if (typeof FileParser !== 'undefined') {
                const result = await FileParser.parseFile(file);
                
                if (result.error) {
                    this.showToast(result.error, 'error');
                    return;
                }

                // Set text
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = result.text;
                    this.updateAnalyzeButton();
                }

                // Store metadata
                this.currentMetadata = result.metadata || {};
                this.currentMetadata.source = fileType;
                this.currentMetadata.filename = file.name;

                // Show metadata bar
                this.showMetadataBar(result);

                this.showToast(`Loaded ${file.name}`, 'success');
            } else {
                // Fallback to basic text reading
                const reader = new FileReader();
                reader.onload = (e) => {
                    const textInput = document.getElementById('textInput');
                    if (textInput) {
                        textInput.value = e.target.result;
                        this.updateAnalyzeButton();
                        this.showToast(`Loaded ${file.name}`, 'success');
                    }
                };
                reader.onerror = () => {
                    this.showToast('Error reading file', 'error');
                };
                reader.readAsText(file);
            }
        } catch (error) {
            console.error('File parsing error:', error);
            this.showToast('Error processing file: ' + error.message, 'error');
        }

        // Reset input
        event.target.value = '';
    },

    /**
     * Handle paste events for clipboard detection
     */
    async handlePaste(event) {
        // Check if FileParser is available for source detection
        if (typeof FileParser !== 'undefined') {
            const clipboardData = event.clipboardData || window.clipboardData;
            const html = clipboardData.getData('text/html');
            
            if (html) {
                const detection = FileParser.detectClipboardSource(html);
                if (detection.source !== 'unknown') {
                    // Delay to allow paste to complete
                    setTimeout(() => {
                        this.currentMetadata = {
                            source: 'clipboard',
                            detectedSource: detection.source,
                            confidence: detection.confidence
                        };
                        
                        // Check for hidden formatting
                        const textInput = document.getElementById('textInput');
                        if (textInput) {
                            const hiddenFormatting = FileParser.detectHiddenFormatting(textInput.value);
                            if (hiddenFormatting.hasIssues) {
                                this.currentMetadata.hiddenFormatting = hiddenFormatting;
                            }
                            this.showMetadataBar({
                                metadata: this.currentMetadata,
                                hiddenFormatting
                            });
                        }
                    }, 100);
                }
            }
        }
    },

    /**
     * Paste from clipboard button handler
     */
    async pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            if (text) {
                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = text;
                    this.updateAnalyzeButton();
                    this.showToast('Text pasted from clipboard', 'success');
                }
            }
        } catch (error) {
            this.showToast('Could not access clipboard. Please paste manually (Ctrl+V)', 'warning');
        }
    },

    /**
     * Prompt for Google Docs URL
     */
    async promptGoogleDocsUrl() {
        const url = prompt('Enter Google Docs URL:\n\n(Note: Document must be publicly accessible or you must be signed in)');
        
        if (!url) return;

        if (!url.includes('docs.google.com')) {
            this.showToast('Please enter a valid Google Docs URL', 'error');
            return;
        }

        this.showToast('Fetching document...', 'info');

        try {
            if (typeof FileParser !== 'undefined') {
                const result = await FileParser.parseGoogleDocsUrl(url);
                
                if (result.error) {
                    this.showToast(result.error, 'error');
                    return;
                }

                const textInput = document.getElementById('textInput');
                if (textInput) {
                    textInput.value = result.text;
                    this.updateAnalyzeButton();
                }

                this.currentMetadata = {
                    source: 'google_docs',
                    url: url
                };

                this.showToast('Google Doc loaded', 'success');
            } else {
                this.showToast('Google Docs parsing not available', 'error');
            }
        } catch (error) {
            console.error('Google Docs error:', error);
            this.showToast('Could not load Google Doc. Make sure it\'s publicly accessible.', 'error');
        }
    },

    /**
     * Show metadata bar with source info and warnings
     */
    showMetadataBar(result) {
        const metadataBar = document.getElementById('metadataBar');
        const sourceIndicator = document.getElementById('sourceIndicator');
        const warningsContainer = document.getElementById('metadataWarnings');
        
        if (!metadataBar) return;

        metadataBar.hidden = false;

        // Source indicator
        if (sourceIndicator) {
            const source = result.metadata?.source || result.metadata?.detectedSource || 'text';
            sourceIndicator.textContent = this.getSourceLabel(source);
            sourceIndicator.className = `metadata-source ${source}`;
        }

        // Warnings
        if (warningsContainer) {
            warningsContainer.innerHTML = '';
            
            const warnings = [];
            
            // Check hidden formatting
            if (result.hiddenFormatting?.hasIssues) {
                const hf = result.hiddenFormatting;
                if (hf.hasInvisibleChars) warnings.push('Invisible characters detected');
                if (hf.hasMixedSpacing) warnings.push('Mixed spacing/tabs');
                if (hf.hasUnusualUnicode) warnings.push('Unusual Unicode');
            }

            // Check metadata issues
            if (result.metadata?.detectedSource === 'chatgpt') {
                warnings.push('ChatGPT formatting detected');
            }
            if (result.metadata?.detectedSource === 'google_docs') {
                warnings.push('Google Docs formatting');
            }

            warnings.forEach(warning => {
                const badge = document.createElement('span');
                badge.className = 'metadata-warning';
                badge.innerHTML = '<span class="material-icons" style="font-size:12px;vertical-align:middle">warning</span> ' + warning;
                warningsContainer.appendChild(badge);
            });
        }
    },

    /**
     * Get human-readable source label
     */
    getSourceLabel(source) {
        const labels = {
            'txt': 'Plain Text',
            'docx': 'Word Document',
            'pdf': 'PDF Document',
            'clipboard': 'Clipboard',
            'google_docs': 'Google Docs',
            'chatgpt': 'ChatGPT',
            'ms_word': 'MS Word',
            'notion': 'Notion'
        };
        return labels[source] || 'Text';
    },

    /**
     * Load sample text
     */
    loadSampleText() {
        const samples = [
            `The implications of artificial intelligence on modern society are profound and far-reaching. As we navigate this technological landscape, it becomes increasingly important to consider both the opportunities and challenges that AI presents. This transformative technology has the potential to revolutionize various sectors, from healthcare to education, while simultaneously raising important ethical considerations that must be addressed.

In examining the current state of AI development, we observe a remarkable acceleration in capabilities across multiple domains. Natural language processing, computer vision, and machine learning algorithms have achieved unprecedented levels of sophistication. These advancements enable applications that were previously considered science fiction, now becoming practical realities that impact our daily lives.

However, it is essential to approach these developments with a balanced perspective. While the benefits are substantial, we must also acknowledge the potential risks and unintended consequences. Issues such as algorithmic bias, privacy concerns, and the displacement of human workers require thoughtful consideration and proactive solutions.`,

            `Yesterday I accidentally spilled coffee all over my laptop - total disaster! Had to rush to the repair shop, and the guy there was super helpful. He said the keyboard might be salvageable but no promises. I've been using my old tablet in the meantime which is honestly driving me crazy. The touchscreen just isn't the same, you know?

My cat has been acting weird lately too. She keeps staring at the wall for like, twenty minutes straight. My roommate thinks she's seeing ghosts lol. I think she's just being her usual dramatic self. Cats are weird.

Anyway, gonna try to get some work done on this ancient tablet. Wish me luck! Also need to remember to call mom - her birthday's coming up and I still haven't figured out a gift. Maybe those fancy chocolates she likes?`,

            `The study examined the effects of urban green spaces on mental health outcomes in metropolitan populations. A total of 2,847 participants were recruited from five major cities across the United States. Data collection occurred between March 2019 and December 2022, utilizing standardized psychological assessment tools including the Beck Depression Inventory and the General Anxiety Disorder-7 scale.

Results indicated a statistically significant correlation between proximity to green spaces and reduced symptoms of anxiety and depression. Participants residing within 500 meters of parks or natural areas demonstrated 23% lower average GAD-7 scores compared to those living in areas with minimal green coverage. Furthermore, frequency of green space visitation emerged as a moderating variable in this relationship.

These findings have important implications for urban planning and public health policy. Municipal authorities should consider incorporating green infrastructure into development projects, particularly in underserved neighborhoods with limited access to natural environments.`
        ];

        const textInput = document.getElementById('textInput');
        if (textInput) {
            const randomSample = samples[Math.floor(Math.random() * samples.length)];
            textInput.value = randomSample;
            this.updateAnalyzeButton();
            this.showToast('Sample text loaded', 'info');
        }
    },

    /**
     * Save to history
     */
    saveToHistory(text, result) {
        const history = Utils.loadHistory();
        const entry = {
            id: Date.now(),
            text: text.substring(0, 200) + (text.length > 200 ? '...' : ''),
            fullText: text,
            aiProbability: result.aiProbability,
            verdict: result.verdict.label,
            timestamp: new Date().toISOString(),
            stats: result.stats
        };
        
        history.unshift(entry);
        
        // Keep only last 50 entries
        if (history.length > 50) {
            history.pop();
        }
        
        Utils.saveHistory(history);
        this.renderHistoryList();
    },

    /**
     * Load and render history
     */
    loadHistory() {
        this.renderHistoryList();
    },

    /**
     * Render history list
     */
    renderHistoryList() {
        const historyList = document.getElementById('historyList');
        if (!historyList) return;

        const history = Utils.loadHistory();
        
        if (history.length === 0) {
            historyList.innerHTML = `
                <div class="history-empty">
                    <p>No analysis history yet</p>
                    <p class="text-muted">Your analyzed texts will appear here</p>
                </div>
            `;
            return;
        }

        historyList.innerHTML = history.map(entry => {
            const date = new Date(entry.timestamp);
            const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
            const level = entry.aiProbability > 0.6 ? 'ai' : entry.aiProbability < 0.4 ? 'human' : 'mixed';
            
            return `
                <div class="history-item" data-id="${entry.id}">
                    <div class="history-item-header">
                        <span class="history-verdict ${level}">${entry.verdict}</span>
                        <span class="history-score">${Math.round(entry.aiProbability * 100)}% AI</span>
                    </div>
                    <p class="history-text">${this.escapeHtml(entry.text)}</p>
                    <div class="history-item-footer">
                        <span class="history-date">${formattedDate}</span>
                        <span class="history-words">${entry.stats?.words || 0} words</span>
                    </div>
                    <div class="history-actions">
                        <button class="btn btn-sm btn-ghost history-load" data-id="${entry.id}">Load</button>
                        <button class="btn btn-sm btn-ghost history-delete" data-id="${entry.id}">Delete</button>
                    </div>
                </div>
            `;
        }).join('');

        // Bind history item events
        historyList.querySelectorAll('.history-load').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = parseInt(btn.dataset.id);
                this.loadFromHistory(id);
            });
        });

        historyList.querySelectorAll('.history-delete').forEach(btn => {
            btn.addEventListener('click', () => {
                const id = parseInt(btn.dataset.id);
                this.deleteFromHistory(id);
            });
        });
    },

    /**
     * Load analysis from history
     */
    loadFromHistory(id) {
        const history = Utils.loadHistory();
        const entry = history.find(h => h.id === id);
        
        if (entry && entry.fullText) {
            const textInput = document.getElementById('textInput');
            if (textInput) {
                textInput.value = entry.fullText;
                this.updateAnalyzeButton();
                this.showView('analyze');
                this.showToast('Text loaded from history', 'info');
            }
        }
    },

    /**
     * Delete from history
     */
    deleteFromHistory(id) {
        let history = Utils.loadHistory();
        history = history.filter(h => h.id !== id);
        Utils.saveHistory(history);
        this.renderHistoryList();
        this.showToast('Entry deleted', 'info');
    },

    /**
     * Clear all history
     */
    clearHistory() {
        if (confirm('Are you sure you want to clear all history?')) {
            Utils.saveHistory([]);
            this.renderHistoryList();
            this.showToast('History cleared', 'info');
        }
    },

    /**
     * Export report in specified format
     */
    async exportReport(format = 'markdown') {
        if (!this.currentResult) {
            this.showToast('No analysis to export', 'warning');
            return;
        }

        const textInput = document.getElementById('textInput');
        const originalText = textInput?.value || '';

        this.showToast('Generating report...', 'info');

        try {
            // Check if ReportExporter is available
            if (typeof ReportExporter !== 'undefined') {
                switch (format) {
                    case 'pdf':
                        await ReportExporter.exportPdf(this.currentResult, originalText);
                        this.showToast('PDF report generated', 'success');
                        break;
                    
                    case 'docx':
                        await ReportExporter.exportDocx(this.currentResult, originalText);
                        this.showToast('DOCX report downloaded', 'success');
                        break;
                    
                    case 'markdown':
                        const markdown = ReportExporter.exportMarkdown(this.currentResult, originalText);
                        this.downloadFile(markdown, `veritas-report-${Date.now()}.md`, 'text/markdown');
                        this.showToast('Markdown report downloaded', 'success');
                        break;
                    
                    case 'json':
                        const json = JSON.stringify({
                            generated: new Date().toISOString(),
                            result: this.currentResult,
                            originalText: originalText
                        }, null, 2);
                        this.downloadFile(json, `veritas-data-${Date.now()}.json`, 'application/json');
                        this.showToast('JSON data downloaded', 'success');
                        break;
                    
                    default:
                        this.showToast('Unknown export format', 'error');
                }
            } else {
                // Fallback to basic markdown export
                const report = AnalyzerEngine.generateReport(this.currentResult);
                const markdown = this.generateBasicMarkdown(report, originalText);
                this.downloadFile(markdown, `veritas-report-${Date.now()}.md`, 'text/markdown');
                this.showToast('Report exported', 'success');
            }
        } catch (error) {
            console.error('Export error:', error);
            this.showToast('Error exporting report: ' + error.message, 'error');
        }
    },

    /**
     * Open the full report in a new browser tab for printing/saving
     * This replaces the broken PDF/DOCX export with a more reliable approach
     */
    openFullReport() {
        if (!this.currentResult) {
            this.showToast('No analysis to view', 'warning');
            return;
        }

        const textInput = document.getElementById('textInput');
        const originalText = textInput?.value || '';
        
        // Get current model info
        const currentModel = this.getCurrentModel();
        const modelData = this.models?.find(m => m.id === currentModel) || { id: 'helios', name: 'Helios', accuracy: '99.24%' };

        try {
            if (typeof ReportExporter !== 'undefined') {
                // Generate the report content with model info
                const reportContent = ReportExporter.generateReportContent(this.currentResult, originalText, modelData);
                
                // Generate complete HTML report with model info
                const htmlContent = ReportExporter.generateHtmlReport(reportContent, this.currentResult, modelData);
                
                // Open in new tab
                const reportWindow = window.open('', '_blank');
                if (reportWindow) {
                    reportWindow.document.write(htmlContent);
                    reportWindow.document.close();
                    
                    // Add print/save instructions with print-hide class
                    const infoBar = reportWindow.document.createElement('div');
                    infoBar.className = 'print-info-bar';
                    infoBar.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#3b82f6;color:white;padding:10px 20px;font-family:system-ui,sans-serif;font-size:14px;display:flex;justify-content:space-between;align-items:center;z-index:9999;box-shadow:0 2px 10px rgba(0,0,0,0.2);';
                    infoBar.innerHTML = `
                        <span><span class="material-icons" style="font-size:14px;vertical-align:middle">description</span> VERITAS Analysis Report — Use <kbd style="background:#2563eb;padding:2px 6px;border-radius:3px;margin:0 3px;">Ctrl+P</kbd> / <kbd style="background:#2563eb;padding:2px 6px;border-radius:3px;margin:0 3px;">⌘P</kbd> to save as PDF or print</span>
                        <button onclick="this.parentElement.remove()" style="background:#2563eb;border:none;color:white;padding:5px 15px;border-radius:4px;cursor:pointer;font-size:12px;">Dismiss</button>
                    `;
                    reportWindow.document.body.insertBefore(infoBar, reportWindow.document.body.firstChild);
                    
                    // Add print styles to hide the info bar when printing
                    const printStyle = reportWindow.document.createElement('style');
                    printStyle.textContent = '@media print { .print-info-bar { display: none !important; } body { padding-top: 0 !important; } }';
                    reportWindow.document.head.appendChild(printStyle);
                    
                    // Add padding to body to account for fixed header
                    reportWindow.document.body.style.paddingTop = '50px';
                    
                    this.showToast('Report opened in new tab', 'success');
                } else {
                    this.showToast('Pop-up blocked. Please allow pop-ups for this site.', 'error');
                }
            } else {
                // Fallback: generate basic HTML report
                const report = AnalyzerEngine.generateReport(this.currentResult);
                const basicHtml = this.generateBasicHtmlReport(report, originalText);
                
                const reportWindow = window.open('', '_blank');
                if (reportWindow) {
                    reportWindow.document.write(basicHtml);
                    reportWindow.document.close();
                    this.showToast('Report opened in new tab', 'success');
                }
            }
        } catch (error) {
            console.error('Error opening report:', error);
            this.showToast('Error generating report: ' + error.message, 'error');
        }
    },

    /**
     * Generate basic HTML report (fallback)
     */
    generateBasicHtmlReport(report, originalText) {
        const prob = report.overall.aiProbability;
        const barColor = prob >= 60 ? '#ef4444' : (prob >= 40 ? '#f59e0b' : '#10b981');
        
        return `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VERITAS Analysis Report</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; color: #333; }
        h1 { color: #111; border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { color: #333; margin-top: 30px; }
        .verdict { font-size: 1.5em; font-weight: bold; color: ${barColor}; }
        .bar { height: 20px; background: #e5e5e5; border-radius: 10px; overflow: hidden; margin: 15px 0; }
        .fill { height: 100%; background: ${barColor}; }
        .stats { display: flex; gap: 20px; flex-wrap: wrap; }
        .stat { background: #f5f5f5; padding: 15px; border-radius: 8px; flex: 1; min-width: 120px; }
        .stat-label { font-size: 0.8em; color: #666; text-transform: uppercase; }
        .stat-value { font-size: 1.5em; font-weight: bold; }
        .text-sample { background: #f9fafb; border: 1px solid #e5e5e5; padding: 15px; border-radius: 8px; margin-top: 20px; white-space: pre-wrap; font-size: 0.9em; max-height: 400px; overflow-y: auto; }
        @media print { .info-bar { display: none !important; } body { padding-top: 0 !important; } }
    </style>
</head>
<body>
    <h1>◈ VERITAS Analysis Report</h1>
    <p>Generated: ${new Date().toLocaleString()}</p>
    
    <div class="verdict">${report.overall.verdict.label}</div>
    <div class="bar"><div class="fill" style="width: ${Math.max(1, prob)}%"></div></div>
    <p><strong>AI Probability:</strong> ${prob}% | <strong>Confidence:</strong> ${report.overall.confidence}%</p>
    
    <h2>Text Statistics</h2>
    <div class="stats">
        <div class="stat"><div class="stat-label">Words</div><div class="stat-value">${report.stats.words}</div></div>
        <div class="stat"><div class="stat-label">Sentences</div><div class="stat-value">${report.stats.sentences}</div></div>
        <div class="stat"><div class="stat-label">Paragraphs</div><div class="stat-value">${report.stats.paragraphs}</div></div>
    </div>
    
    <h2>Category Analysis</h2>
    ${report.sections.map(s => `
        <h3>${s.number}. ${s.name} — ${s.aiScore}% AI</h3>
        <p>Confidence: ${s.confidence}%</p>
        ${s.findings.length > 0 ? `<ul>${s.findings.slice(0, 5).map(f => `<li>${f.text}</li>`).join('')}</ul>` : ''}
    `).join('')}
    
    <h2>Analyzed Text</h2>
    <div class="text-sample">${originalText.substring(0, 2000)}${originalText.length > 2000 ? '...' : ''}</div>
    
    <p style="margin-top: 40px; color: #888; font-size: 0.9em; text-align: center;">
        Generated by VERITAS AI Detection System | Sunrise Model v3.0 | 98.08% Accuracy
    </p>
</body>
</html>`;
    },

    /**
     * Download file helper
     */
    downloadFile(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    },

    /**
     * Generate basic markdown report (fallback)
     */
    generateBasicMarkdown(report, originalText) {
        let markdown = `# VERITAS Analysis Report\n\n`;
        markdown += `**Generated:** ${new Date().toLocaleString()}\n\n`;
        markdown += `## Overall Result\n\n`;
        markdown += `- **Verdict:** ${report.overall.verdict.label}\n`;
        markdown += `- **AI Probability:** ${report.overall.aiProbability}%\n`;
        markdown += `- **Confidence:** ${report.overall.confidence}%\n\n`;
        markdown += `## Text Statistics\n\n`;
        markdown += `- Words: ${report.stats.words}\n`;
        markdown += `- Sentences: ${report.stats.sentences}\n`;
        markdown += `- Paragraphs: ${report.stats.paragraphs}\n\n`;
        markdown += `## Category Analysis\n\n`;
        
        for (const section of report.sections) {
            markdown += `### ${section.number}. ${section.name}\n\n`;
            markdown += `AI Score: ${section.aiScore}% | Confidence: ${section.confidence}%\n\n`;
            if (section.findings.length > 0) {
                markdown += `**Findings:**\n`;
                for (const finding of section.findings) {
                    const indicator = finding.indicator === 'ai' ? '[AI]' : finding.indicator === 'human' ? '[Human]' : '[Mixed]';
                    markdown += `- ${indicator} ${finding.text}\n`;
                }
            }
            markdown += `\n`;
        }

        markdown += `## Analyzed Text\n\n`;
        markdown += `\`\`\`\n${originalText}\n\`\`\`\n`;

        return markdown;
    },

    /**
     * Copy results to clipboard
     */
    async copyResults() {
        if (!this.currentResult) {
            this.showToast('No results to copy', 'warning');
            return;
        }

        const text = `VERITAS Analysis Result:
Verdict: ${this.currentResult.verdict.label}
AI Probability: ${Math.round(this.currentResult.aiProbability * 100)}%
Confidence: ${Math.round(this.currentResult.confidence * 100)}%`;

        try {
            await navigator.clipboard.writeText(text);
            this.showToast('Results copied to clipboard', 'success');
        } catch (err) {
            this.showToast('Failed to copy results', 'error');
        }
    },

    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        // Remove existing toast
        const existing = document.querySelector('.toast');
        if (existing) {
            existing.remove();
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        // Trigger animation
        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        // Auto remove
        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    },

    /**
     * Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = App;
}
