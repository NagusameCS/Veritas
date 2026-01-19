/**
 * VERITAS — Visualizations
 * SVG-based charts and visual components
 */

const Visualizations = {
    colors: {
        ai: '#404040',        // Dark gray for AI
        human: '#a0a0a0',     // Light gray for human
        mixed: '#707070',     // Medium gray for mixed
        neutral: '#888888',   // Neutral gray
        bg: '#f5f5f5',        // Background
        border: '#d0d0d0',    // Border
        text: '#1a1a1a',      // Text
        textMuted: '#666666'  // Muted text
    },

    /**
     * Create the main score ring
     */
    createScoreRing(container, aiProbability, confidence) {
        const size = 200;
        const strokeWidth = 12;
        const radius = (size - strokeWidth) / 2;
        const circumference = 2 * Math.PI * radius;
        const aiOffset = circumference * (1 - aiProbability);
        const humanOffset = circumference * aiProbability;
        
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
        svg.setAttribute('class', 'score-ring-svg');
        
        // Background circle
        const bgCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        bgCircle.setAttribute('cx', size / 2);
        bgCircle.setAttribute('cy', size / 2);
        bgCircle.setAttribute('r', radius);
        bgCircle.setAttribute('fill', 'none');
        bgCircle.setAttribute('stroke', this.colors.bg);
        bgCircle.setAttribute('stroke-width', strokeWidth);
        
        // AI portion (dark)
        const aiCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        aiCircle.setAttribute('cx', size / 2);
        aiCircle.setAttribute('cy', size / 2);
        aiCircle.setAttribute('r', radius);
        aiCircle.setAttribute('fill', 'none');
        aiCircle.setAttribute('stroke', this.colors.ai);
        aiCircle.setAttribute('stroke-width', strokeWidth);
        aiCircle.setAttribute('stroke-linecap', 'round');
        aiCircle.setAttribute('stroke-dasharray', circumference);
        aiCircle.setAttribute('stroke-dashoffset', aiOffset);
        aiCircle.setAttribute('transform', `rotate(-90 ${size/2} ${size/2})`);
        aiCircle.classList.add('score-ring-progress');
        aiCircle.style.setProperty('--target-offset', aiOffset);
        
        // Human portion (light)
        const humanCircle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
        humanCircle.setAttribute('cx', size / 2);
        humanCircle.setAttribute('cy', size / 2);
        humanCircle.setAttribute('r', radius - 20);
        humanCircle.setAttribute('fill', 'none');
        humanCircle.setAttribute('stroke', this.colors.human);
        humanCircle.setAttribute('stroke-width', strokeWidth);
        humanCircle.setAttribute('stroke-linecap', 'round');
        humanCircle.setAttribute('stroke-dasharray', (radius - 20) * 2 * Math.PI);
        humanCircle.setAttribute('stroke-dashoffset', (radius - 20) * 2 * Math.PI * aiProbability);
        humanCircle.setAttribute('transform', `rotate(-90 ${size/2} ${size/2})`);
        humanCircle.classList.add('score-ring-progress');
        
        // Center text
        const centerGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        
        const percentText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        percentText.setAttribute('x', size / 2);
        percentText.setAttribute('y', size / 2 - 5);
        percentText.setAttribute('text-anchor', 'middle');
        percentText.setAttribute('class', 'score-ring-percent');
        percentText.textContent = Math.round(aiProbability * 100) + '%';
        
        const labelText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        labelText.setAttribute('x', size / 2);
        labelText.setAttribute('y', size / 2 + 20);
        labelText.setAttribute('text-anchor', 'middle');
        labelText.setAttribute('class', 'score-ring-label');
        labelText.textContent = 'AI Probability';
        
        centerGroup.appendChild(percentText);
        centerGroup.appendChild(labelText);
        
        svg.appendChild(bgCircle);
        svg.appendChild(humanCircle);
        svg.appendChild(aiCircle);
        svg.appendChild(centerGroup);
        
        container.innerHTML = '';
        container.appendChild(svg);
        
        // Animate after a frame
        requestAnimationFrame(() => {
            aiCircle.style.strokeDashoffset = aiOffset;
        });
    },

    /**
     * Create sentence probability graph
     */
    createSentenceGraph(container, sentenceScores) {
        if (!sentenceScores || sentenceScores.length === 0) {
            container.innerHTML = '<p class="no-data">Not enough sentences for visualization</p>';
            return;
        }

        const width = 600;
        const height = 200;
        const padding = { top: 20, right: 20, bottom: 40, left: 50 };
        const graphWidth = width - padding.left - padding.right;
        const graphHeight = height - padding.top - padding.bottom;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
        svg.setAttribute('class', 'sentence-graph-svg');
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        // Grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (graphHeight / 4) * i;
            const gridLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            gridLine.setAttribute('x1', padding.left);
            gridLine.setAttribute('x2', width - padding.right);
            gridLine.setAttribute('y1', y);
            gridLine.setAttribute('y2', y);
            gridLine.setAttribute('class', 'graph-grid-line');
            svg.appendChild(gridLine);

            // Y-axis labels
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', padding.left - 10);
            label.setAttribute('y', y + 4);
            label.setAttribute('text-anchor', 'end');
            label.setAttribute('class', 'graph-axis-label');
            label.textContent = (100 - i * 25) + '%';
            svg.appendChild(label);
        }

        // AI probability threshold line
        const thresholdY = padding.top + graphHeight * 0.4;
        const thresholdLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        thresholdLine.setAttribute('x1', padding.left);
        thresholdLine.setAttribute('x2', width - padding.right);
        thresholdLine.setAttribute('y1', thresholdY);
        thresholdLine.setAttribute('y2', thresholdY);
        thresholdLine.setAttribute('class', 'graph-threshold-line');
        svg.appendChild(thresholdLine);

        // Build path
        const points = sentenceScores.map((s, i) => {
            const x = padding.left + (graphWidth / (sentenceScores.length - 1 || 1)) * i;
            const y = padding.top + graphHeight * (1 - s.aiProbability);
            return { x, y, score: s };
        });

        if (points.length > 1) {
            // Area under curve
            let areaPath = `M ${points[0].x} ${padding.top + graphHeight} `;
            areaPath += `L ${points[0].x} ${points[0].y} `;
            for (let i = 1; i < points.length; i++) {
                areaPath += `L ${points[i].x} ${points[i].y} `;
            }
            areaPath += `L ${points[points.length-1].x} ${padding.top + graphHeight} Z`;

            const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            area.setAttribute('d', areaPath);
            area.setAttribute('class', 'graph-area');
            svg.appendChild(area);

            // Line
            let linePath = `M ${points[0].x} ${points[0].y} `;
            for (let i = 1; i < points.length; i++) {
                linePath += `L ${points[i].x} ${points[i].y} `;
            }

            const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            line.setAttribute('d', linePath);
            line.setAttribute('class', 'graph-line');
            svg.appendChild(line);
        }

        // Points
        points.forEach((point, i) => {
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', point.x);
            circle.setAttribute('cy', point.y);
            circle.setAttribute('r', 4);
            circle.setAttribute('class', `graph-point ${point.score.classification}`);
            circle.setAttribute('data-sentence-index', i);
            svg.appendChild(circle);
        });

        // X-axis label
        const xLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        xLabel.setAttribute('x', width / 2);
        xLabel.setAttribute('y', height - 5);
        xLabel.setAttribute('text-anchor', 'middle');
        xLabel.setAttribute('class', 'graph-axis-label');
        xLabel.textContent = 'Sentences';
        svg.appendChild(xLabel);

        // Y-axis label
        const yLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        yLabel.setAttribute('x', -height / 2);
        yLabel.setAttribute('y', 15);
        yLabel.setAttribute('text-anchor', 'middle');
        yLabel.setAttribute('transform', 'rotate(-90)');
        yLabel.setAttribute('class', 'graph-axis-label');
        yLabel.textContent = 'AI Probability';
        svg.appendChild(yLabel);

        container.innerHTML = '';
        container.appendChild(svg);
    },

    /**
     * Create feature radar chart
     */
    createRadarChart(container, categoryResults) {
        if (!categoryResults || categoryResults.length === 0) {
            container.innerHTML = '<p class="no-data">No feature data available</p>';
            return;
        }

        const size = 300;
        const center = size / 2;
        const maxRadius = size / 2 - 40;
        const numAxes = categoryResults.length;
        const angleStep = (2 * Math.PI) / numAxes;

        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('viewBox', `0 0 ${size} ${size}`);
        svg.setAttribute('class', 'radar-chart-svg');

        // Draw concentric circles (grid)
        for (let i = 1; i <= 4; i++) {
            const radius = (maxRadius / 4) * i;
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', center);
            circle.setAttribute('cy', center);
            circle.setAttribute('r', radius);
            circle.setAttribute('class', 'radar-grid-circle');
            svg.appendChild(circle);
        }

        // Draw axes
        categoryResults.forEach((_, i) => {
            const angle = angleStep * i - Math.PI / 2;
            const x = center + Math.cos(angle) * maxRadius;
            const y = center + Math.sin(angle) * maxRadius;
            
            const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            line.setAttribute('x1', center);
            line.setAttribute('y1', center);
            line.setAttribute('x2', x);
            line.setAttribute('y2', y);
            line.setAttribute('class', 'radar-axis');
            svg.appendChild(line);
        });

        // Draw data polygon
        const points = categoryResults.map((result, i) => {
            const angle = angleStep * i - Math.PI / 2;
            const radius = maxRadius * result.aiProbability;
            return {
                x: center + Math.cos(angle) * radius,
                y: center + Math.sin(angle) * radius
            };
        });

        const polygonPoints = points.map(p => `${p.x},${p.y}`).join(' ');
        
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', polygonPoints);
        polygon.setAttribute('class', 'radar-polygon');
        svg.appendChild(polygon);

        // Draw points and labels
        categoryResults.forEach((result, i) => {
            const angle = angleStep * i - Math.PI / 2;
            const dataRadius = maxRadius * result.aiProbability;
            const x = center + Math.cos(angle) * dataRadius;
            const y = center + Math.sin(angle) * dataRadius;
            
            // Data point
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', 5);
            circle.setAttribute('class', 'radar-point');
            svg.appendChild(circle);

            // Label
            const labelRadius = maxRadius + 20;
            const labelX = center + Math.cos(angle) * labelRadius;
            const labelY = center + Math.sin(angle) * labelRadius;
            
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', labelX);
            label.setAttribute('y', labelY);
            label.setAttribute('text-anchor', 'middle');
            label.setAttribute('dominant-baseline', 'middle');
            label.setAttribute('class', 'radar-label');
            label.textContent = result.category;
            svg.appendChild(label);
        });

        container.innerHTML = '';
        container.appendChild(svg);
    },

    /**
     * Create category bar chart
     */
    createCategoryBars(container, categoryResults) {
        if (!categoryResults || categoryResults.length === 0) {
            container.innerHTML = '<p class="no-data">No category data available</p>';
            return;
        }

        const barsHtml = categoryResults.map(result => {
            const percentage = Math.round(result.aiProbability * 100);
            const level = percentage > 60 ? 'high' : percentage > 40 ? 'medium' : 'low';
            
            return `
                <div class="category-bar-item">
                    <div class="category-bar-header">
                        <span class="category-bar-name">${result.category}. ${result.name}</span>
                        <span class="category-bar-value">${percentage}%</span>
                    </div>
                    <div class="category-bar-track">
                        <div class="category-bar-fill ${level}" style="width: ${percentage}%"></div>
                    </div>
                    <div class="category-bar-confidence">
                        Confidence: ${Math.round(result.confidence * 100)}%
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `<div class="category-bars">${barsHtml}</div>`;
    },

    /**
     * Render text with annotated trend observations instead of unreliable per-sentence highlighting
     */
    renderHighlightedText(container, sentences, sentenceScores, analysisResult = null) {
        if (!sentences || sentences.length === 0) {
            container.innerHTML = '<p class="no-text">No text to display</p>';
            return;
        }

        // Gather observed trends from analysis
        const trends = this.extractObservedTrends(sentences, sentenceScores, analysisResult);
        
        // Render clean text with trend annotations
        const textHtml = sentences.map((sentence, i) => {
            return `<span class="text-sentence" data-index="${i}">${this.escapeHtml(sentence)}</span>`;
        }).join(' ');

        // Render trend observations sidebar
        const trendsHtml = trends.length > 0 
            ? trends.map(trend => `
                <div class="trend-item ${trend.type}">
                    <span class="material-icons trend-icon">${trend.icon}</span>
                    <div class="trend-content">
                        <span class="trend-title">${this.escapeHtml(trend.title)}</span>
                        <span class="trend-detail">${this.escapeHtml(trend.detail)}</span>
                    </div>
                </div>
            `).join('')
            : '<p class="no-trends">No significant patterns detected</p>';

        container.innerHTML = `
            <div class="annotated-text-container">
                <div class="text-display">
                    <div class="text-content">${textHtml}</div>
                </div>
                <div class="trends-panel">
                    <h4>Observed Trends</h4>
                    <div class="trends-list">${trendsHtml}</div>
                </div>
            </div>
        `;
    },

    /**
     * Extract meaningful trends from the analysis
     */
    extractObservedTrends(sentences, sentenceScores, analysisResult) {
        const trends = [];
        
        if (!sentences || sentences.length === 0) return trends;
        
        // Calculate sentence length stats
        const lengths = sentences.map(s => s.split(/\s+/).length);
        const avgLength = lengths.reduce((a, b) => a + b, 0) / lengths.length;
        const lengthVariance = lengths.reduce((a, b) => a + Math.pow(b - avgLength, 2), 0) / lengths.length;
        const lengthCV = Math.sqrt(lengthVariance) / avgLength;
        
        // Sentence length uniformity
        if (lengthCV < 0.3) {
            trends.push({
                type: 'ai-signal',
                icon: 'straighten',
                title: 'Uniform sentence length',
                detail: `Low variance (CV: ${(lengthCV * 100).toFixed(0)}%) — typical of AI text`
            });
        } else if (lengthCV > 0.6) {
            trends.push({
                type: 'human-signal',
                icon: 'edit',
                title: 'Variable sentence length',
                detail: `High variance (CV: ${(lengthCV * 100).toFixed(0)}%) — typical of human writing`
            });
        }
        
        // Paragraph structure
        const paragraphs = sentences.join(' ').split(/\n\s*\n/).filter(p => p.trim());
        if (paragraphs.length === 1 && sentences.length > 5) {
            trends.push({
                type: 'neutral',
                icon: 'article',
                title: 'Single paragraph structure',
                detail: `${sentences.length} sentences in one block`
            });
        }
        
        // Word repetition patterns
        const allWords = sentences.join(' ').toLowerCase().split(/\s+/);
        const wordFreq = {};
        allWords.forEach(w => wordFreq[w] = (wordFreq[w] || 0) + 1);
        const repeatedWords = Object.entries(wordFreq).filter(([w, c]) => c >= 3 && w.length > 4);
        if (repeatedWords.length > 5) {
            trends.push({
                type: 'ai-signal',
                icon: 'loop',
                title: 'Repetitive vocabulary',
                detail: `${repeatedWords.length} words repeated 3+ times`
            });
        }
        
        // Check for AI-typical phrases
        const text = sentences.join(' ').toLowerCase();
        const aiPhrases = ['it is important to note', 'in conclusion', 'furthermore', 'moreover', 'it is worth noting', 'in summary', 'as such'];
        const foundPhrases = aiPhrases.filter(p => text.includes(p));
        if (foundPhrases.length >= 2) {
            trends.push({
                type: 'ai-signal',
                icon: 'format_quote',
                title: 'Formulaic transitions',
                detail: `Found: "${foundPhrases.slice(0, 2).join('", "')}"`
            });
        }
        
        // Personal pronouns
        const firstPersonCount = (text.match(/\b(i|me|my|myself|we|our|us)\b/gi) || []).length;
        const firstPersonRatio = firstPersonCount / allWords.length;
        if (firstPersonRatio > 0.02) {
            trends.push({
                type: 'human-signal',
                icon: 'person',
                title: 'Personal voice present',
                detail: `${firstPersonCount} first-person references found`
            });
        } else if (firstPersonRatio < 0.005 && sentences.length > 3) {
            trends.push({
                type: 'ai-signal',
                icon: 'smart_toy',
                title: 'Impersonal tone',
                detail: 'Minimal first-person pronouns'
            });
        }
        
        // Contractions
        const contractions = (text.match(/\b\w+'\w+\b/g) || []).length;
        if (contractions > 3) {
            trends.push({
                type: 'human-signal',
                icon: 'lightbulb',
                title: 'Natural contractions',
                detail: `${contractions} contractions used`
            });
        } else if (contractions === 0 && sentences.length > 5) {
            trends.push({
                type: 'ai-signal',
                icon: 'description',
                title: 'No contractions',
                detail: 'Formal style without contractions'
            });
        }
        
        // Questions and exclamations
        const questions = sentences.filter(s => s.trim().endsWith('?')).length;
        const exclamations = sentences.filter(s => s.trim().endsWith('!')).length;
        if (questions > 0) {
            trends.push({
                type: 'human-signal',
                icon: 'help',
                title: 'Rhetorical questions',
                detail: `${questions} question${questions > 1 ? 's' : ''} in text`
            });
        }
        if (exclamations > 0) {
            trends.push({
                type: 'human-signal',
                icon: 'priority_high',
                title: 'Expressive punctuation',
                detail: `${exclamations} exclamation${exclamations > 1 ? 's' : ''} used`
            });
        }
        
        return trends.slice(0, 8); // Limit to 8 trends
    },

    /**
     * Create findings list
     */
    renderFindings(container, findings, limit = 10) {
        if (!findings || findings.length === 0) {
            container.innerHTML = '<p class="no-findings">No significant findings</p>';
            return;
        }

        const displayFindings = findings.slice(0, limit);
        
        const html = displayFindings.map(finding => {
            const icon = this.getIndicatorIcon(finding.indicator);
            // Support both text format (legacy) and label/value format (new)
            const mainText = finding.text || finding.value || finding.label || 'Unknown finding';
            const category = finding.category || finding.label || '';
            return `
                <div class="finding-item ${finding.indicator || 'neutral'}">
                    <span class="finding-icon">${icon}</span>
                    <div class="finding-content">
                        <span class="finding-text">${this.escapeHtml(mainText)}</span>
                        ${category ? `<span class="finding-category">${this.escapeHtml(category)}</span>` : ''}
                    </div>
                </div>
            `;
        }).join('');

        let output = `<div class="findings-list">${html}</div>`;
        
        if (findings.length > limit) {
            output += `<p class="findings-more">+${findings.length - limit} more findings in detailed report</p>`;
        }

        container.innerHTML = output;
    },

    /**
     * Create detailed report accordion
     */
    renderDetailedReport(container, report) {
        if (!report || !report.sections) {
            container.innerHTML = '<p class="no-report">No detailed report available</p>';
            return;
        }

        // Count signals across all sections
        let totalAiFindings = 0;
        let totalHumanFindings = 0;
        
        const sectionsHtml = report.sections.map(section => {
            const aiFindings = section.findings.filter(f => f.indicator === 'ai');
            const humanFindings = section.findings.filter(f => f.indicator === 'human');
            totalAiFindings += aiFindings.length;
            totalHumanFindings += humanFindings.length;
            
            const signalSummary = aiFindings.length > 0 || humanFindings.length > 0
                ? `<span class="section-signal-count">🔴 ${aiFindings.length} AI | 🟢 ${humanFindings.length} Human</span>`
                : '';
            
            const findingsHtml = section.findings.length > 0 
                ? section.findings.map(f => {
                    const mainText = f.text || f.value || f.label || 'Unknown finding';
                    const severityClass = f.severity ? `severity-${f.severity}` : '';
                    return `
                        <div class="report-finding ${f.indicator || 'neutral'} ${severityClass}">
                            <span class="finding-icon">${this.getIndicatorIcon(f.indicator)}</span>
                            <div class="finding-content-wrapper">
                                <span class="finding-text">${this.escapeHtml(mainText)}</span>
                                ${f.note ? `<span class="finding-note">${this.escapeHtml(f.note)}</span>` : ''}
                            </div>
                        </div>
                    `;
                }).join('')
                : '<p class="no-findings">No significant findings in this category</p>';

            const barColor = section.aiScore >= 60 ? '#ef4444' : (section.aiScore >= 40 ? '#f59e0b' : '#10b981');
            
            return `
                <div class="accordion-item ${section.aiScore >= 60 ? 'high-signal' : ''}">
                    <button class="accordion-header">
                        <span class="accordion-title">
                            <span class="category-number">${section.number}</span>
                            ${section.name}
                            ${signalSummary}
                        </span>
                        <span class="accordion-meta">
                            <span class="accordion-score ${this.getScoreLevel(section.aiScore)}">${section.aiScore}%</span>
                            <svg class="accordion-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <path d="M6 9l6 6 6-6"/>
                            </svg>
                        </span>
                    </button>
                    <div class="accordion-content">
                        <div class="report-section-content">
                            <div class="report-meta">
                                <span>AI Score: <strong>${section.aiScore}%</strong></span>
                                <span>Confidence: <strong>${section.confidence}%</strong></span>
                                <div class="score-bar-mini">
                                    <div class="score-bar-fill" style="width:${section.aiScore}%;background:${barColor}"></div>
                                </div>
                            </div>
                            <div class="report-findings">
                                ${findingsHtml}
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = `
            <div class="detailed-report">
                <div class="report-header">
                    <h3>Detailed Analysis Report</h3>
                    <div class="report-overall">
                        <span class="report-verdict ${report.overall.verdict.level}">${report.overall.verdict.label}</span>
                        <span class="report-probability">${report.overall.aiProbability}% AI Probability</span>
                    </div>
                    <div class="report-signal-summary">
                        <span class="signal-badge ai">🔴 ${totalAiFindings} AI Indicators</span>
                        <span class="signal-badge human">🟢 ${totalHumanFindings} Human Indicators</span>
                    </div>
                    <p class="signal-note">Note: Indicator counts show patterns detected. Final probability uses ML-derived weights where high-weight categories (Metadata 40%, Lexical 22%, Syntax 21%) have more influence than low-weight categories.</p>
                </div>
                <div class="accordion">
                    ${sectionsHtml}
                </div>
                <div class="report-footer">
                    <p class="report-stats">
                        Analyzed: ${report.stats.words} words, ${report.stats.sentences} sentences
                    </p>
                    <p class="report-timestamp">
                        Generated: ${new Date(report.timestamp).toLocaleString()}
                    </p>
                </div>
            </div>
        `;

        // Add accordion functionality
        container.querySelectorAll('.accordion-header').forEach(header => {
            header.addEventListener('click', () => {
                const item = header.closest('.accordion-item');
                const isOpen = item.classList.contains('open');
                
                // Close all
                container.querySelectorAll('.accordion-item').forEach(i => {
                    i.classList.remove('open');
                });
                
                // Toggle current
                if (!isOpen) {
                    item.classList.add('open');
                }
            });
        });
    },

    /**
     * Helper: Get indicator icon (Material Icons)
     */
    getIndicatorIcon(indicator) {
        const icons = {
            ai: '<span class="material-icons finding-indicator ai">smart_toy</span>',
            human: '<span class="material-icons finding-indicator human">person</span>',
            mixed: '<span class="material-icons finding-indicator mixed">help_outline</span>',
            neutral: '<span class="material-icons finding-indicator neutral">radio_button_unchecked</span>'
        };
        return icons[indicator] || icons.neutral;
    },

    /**
     * Helper: Get score level class
     */
    getScoreLevel(score) {
        if (score >= 70) return 'high';
        if (score >= 40) return 'medium';
        return 'low';
    },

    /**
     * Helper: Escape HTML
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Visualizations;
}
