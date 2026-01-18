/**
 * VERITAS — Main Application
 * Application initialization and event handling
 */

const App = {
    currentResult: null,
    isAnalyzing: false,

    /**
     * Initialize the application
     */
    init() {
        this.bindEvents();
        this.loadHistory();
        this.initTheme();
        this.initTabs();
        console.log('VERITAS initialized');
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
        }

        // File upload
        const uploadBtn = document.getElementById('uploadBtn');
        const fileInput = document.getElementById('fileInput');
        if (uploadBtn && fileInput) {
            uploadBtn.addEventListener('click', () => fileInput.click());
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
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

        // Export button
        const exportBtn = document.getElementById('exportBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', () => this.exportReport());
        }

        // Copy results
        const copyBtn = document.getElementById('copyResultsBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyResults());
        }
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
    },

    /**
     * Clear input
     */
    clearInput() {
        const textInput = document.getElementById('textInput');
        if (textInput) {
            textInput.value = '';
            textInput.focus();
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
            
            // Run analysis
            const result = AnalyzerEngine.analyze(text);
            this.currentResult = result;
            
            // Display results
            this.displayResults(result);
            
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
        } finally {
            this.isAnalyzing = false;
            analyzeBtn.classList.remove('loading');
            this.updateAnalyzeButton();
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
        const scoreCard = document.querySelector('.overall-score');
        if (scoreCard) {
            scoreCard.innerHTML = `
                <div class="skeleton skeleton-circle" style="width: 150px; height: 150px; margin: 0 auto;"></div>
                <div class="skeleton skeleton-text" style="width: 60%; margin: 1rem auto;"></div>
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
    },

    /**
     * Display analysis results
     */
    displayResults(result) {
        // Overall score
        const scoreContainer = document.querySelector('.score-ring-container');
        if (scoreContainer) {
            Visualizations.createScoreRing(scoreContainer, result.aiProbability, result.confidence);
        }

        // Verdict
        const verdictEl = document.querySelector('.verdict-label');
        const verdictDescEl = document.querySelector('.verdict-description');
        if (verdictEl && result.verdict) {
            verdictEl.textContent = result.verdict.label;
            verdictEl.className = `verdict-label ${result.verdict.level}`;
        }
        if (verdictDescEl && result.verdict) {
            verdictDescEl.textContent = result.verdict.description;
        }

        // Confidence
        const confidenceEl = document.querySelector('.confidence-value');
        if (confidenceEl) {
            confidenceEl.textContent = Math.round(result.confidence * 100) + '%';
        }

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
     * Handle file upload
     */
    handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Check file type
        const validTypes = ['text/plain', 'text/markdown'];
        if (!validTypes.includes(file.type) && !file.name.endsWith('.txt') && !file.name.endsWith('.md')) {
            this.showToast('Please upload a .txt or .md file', 'error');
            return;
        }

        // Check file size (max 100KB)
        if (file.size > 100000) {
            this.showToast('File too large. Maximum size is 100KB', 'error');
            return;
        }

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

        // Reset input
        event.target.value = '';
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
     * Export report
     */
    exportReport() {
        if (!this.currentResult) {
            this.showToast('No analysis to export', 'warning');
            return;
        }

        const report = AnalyzerEngine.generateReport(this.currentResult);
        const textInput = document.getElementById('textInput');
        const originalText = textInput?.value || '';

        // Create markdown report
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
                    const indicator = finding.indicator === 'ai' ? '🤖' : finding.indicator === 'human' ? '👤' : '⚖️';
                    markdown += `- ${indicator} ${finding.text}\n`;
                }
            }
            markdown += `\n`;
        }

        markdown += `## Analyzed Text\n\n`;
        markdown += `\`\`\`\n${originalText}\n\`\`\`\n`;

        // Download file
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `veritas-report-${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(url);

        this.showToast('Report exported', 'success');
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
