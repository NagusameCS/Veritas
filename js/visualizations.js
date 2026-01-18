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
     * Render highlighted text
     */
    renderHighlightedText(container, sentences, sentenceScores) {
        if (!sentences || sentences.length === 0) {
            container.innerHTML = '<p class="no-text">No text to display</p>';
            return;
        }

        const html = sentences.map((sentence, i) => {
            const score = sentenceScores[i] || { classification: 'mixed', aiProbability: 0.5 };
            const className = `highlight-${score.classification}`;
            const tooltip = `AI Probability: ${Math.round(score.aiProbability * 100)}%`;
            
            return `<span class="${className}" data-tooltip="${tooltip}" data-index="${i}">${this.escapeHtml(sentence)}</span>`;
        }).join(' ');

        container.innerHTML = `
            <div class="highlighted-text-content">${html}</div>
            <div class="highlight-legend">
                <div class="legend-item">
                    <span class="legend-color highlight-ai"></span>
                    <span class="legend-label">Likely AI</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color highlight-mixed"></span>
                    <span class="legend-label">Mixed/Uncertain</span>
                </div>
                <div class="legend-item">
                    <span class="legend-color highlight-human"></span>
                    <span class="legend-label">Likely Human</span>
                </div>
            </div>
        `;
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
            return `
                <div class="finding-item ${finding.indicator}">
                    <span class="finding-icon">${icon}</span>
                    <div class="finding-content">
                        <span class="finding-text">${this.escapeHtml(finding.text)}</span>
                        <span class="finding-category">${finding.category}</span>
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

        const sectionsHtml = report.sections.map(section => {
            const findingsHtml = section.findings.length > 0 
                ? section.findings.map(f => `
                    <div class="report-finding ${f.indicator}">
                        <span class="finding-icon">${this.getIndicatorIcon(f.indicator)}</span>
                        <span class="finding-text">${this.escapeHtml(f.text)}</span>
                    </div>
                `).join('')
                : '<p class="no-findings">No significant findings in this category</p>';

            return `
                <div class="accordion-item">
                    <button class="accordion-header">
                        <span class="accordion-title">
                            <span class="category-number">${section.number}</span>
                            ${section.name}
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
     * Helper: Get indicator icon
     */
    getIndicatorIcon(indicator) {
        const icons = {
            ai: '🤖',
            human: '👤',
            mixed: '⚖️',
            neutral: '○'
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
