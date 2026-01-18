/**
 * VERITAS — Advanced Visualizations
 * Enhanced charts including sentence distribution, n-gram heatmaps, 
 * tone timeline, Zipf curves, and network graphs
 */

const AdvancedVisualizations = {
    colors: {
        ai: '#2a2a2a',
        human: '#7a7a7a',
        mixed: '#5a5a5a',
        neutral: '#999999',
        grid: '#e0e0e0',
        accent: '#404040',
        bg: '#f8f8f8'
    },

    /**
     * Create sentence length distribution histogram
     */
    createSentenceLengthHistogram(container, sentences) {
        if (!sentences || sentences.length < 3) {
            container.innerHTML = '<p class="no-data">Insufficient sentences for histogram</p>';
            return;
        }

        const lengths = sentences.map(s => s.split(/\s+/).length);
        const bins = this.createHistogramBins(lengths, 10);
        
        const width = 500;
        const height = 250;
        const padding = { top: 30, right: 20, bottom: 50, left: 50 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        const maxCount = Math.max(...bins.map(b => b.count));
        const barWidth = chartWidth / bins.length - 2;

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="histogram-svg">`;
        
        // Title
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Sentence Length Distribution</text>`;

        // Grid lines
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartHeight / 4) * i;
            svg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" stroke="${this.colors.grid}" stroke-width="1"/>`;
        }

        // Bars
        bins.forEach((bin, i) => {
            const x = padding.left + i * (chartWidth / bins.length) + 1;
            const barHeight = maxCount > 0 ? (bin.count / maxCount) * chartHeight : 0;
            const y = padding.top + chartHeight - barHeight;
            
            // Calculate color based on expected human distribution
            // Middle lengths are most common for humans
            const expectedPeak = 15;
            const deviation = Math.abs(bin.center - expectedPeak) / expectedPeak;
            const intensity = Math.min(1, bin.count / maxCount);
            
            svg += `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" 
                    fill="${this.colors.accent}" opacity="${0.4 + intensity * 0.5}"
                    class="histogram-bar" data-count="${bin.count}" data-range="${bin.min}-${bin.max}"/>`;
            
            // X-axis labels
            if (i % 2 === 0) {
                svg += `<text x="${x + barWidth/2}" y="${height - 10}" text-anchor="middle" 
                        class="axis-label">${bin.center}</text>`;
            }
        });

        // Axes
        svg += `<line x1="${padding.left}" y1="${padding.top + chartHeight}" x2="${width - padding.right}" 
                y2="${padding.top + chartHeight}" stroke="${this.colors.accent}" stroke-width="1"/>`;
        svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" 
                y2="${padding.top + chartHeight}" stroke="${this.colors.accent}" stroke-width="1"/>`;

        // Axis labels
        svg += `<text x="${width/2}" y="${height - 5}" text-anchor="middle" class="axis-title">Words per Sentence</text>`;
        svg += `<text x="15" y="${height/2}" text-anchor="middle" transform="rotate(-90, 15, ${height/2})" 
                class="axis-title">Frequency</text>`;

        // Statistics annotation
        const mean = Utils.mean(lengths);
        const variance = Utils.variance(lengths);
        const cv = Math.sqrt(variance) / mean;
        
        svg += `<text x="${width - padding.right}" y="${padding.top + 15}" text-anchor="end" class="stat-text">
                μ = ${mean.toFixed(1)}, CV = ${cv.toFixed(2)}</text>`;

        svg += '</svg>';
        
        // Add description
        const uniformity = VarianceUtils.uniformityScore(lengths);
        const description = uniformity > 0.7 
            ? '<p class="chart-insight ai-signal">⚠️ Low variance in sentence length suggests AI-like uniformity</p>'
            : '<p class="chart-insight human-signal">✓ Natural variation in sentence length typical of human writing</p>';

        container.innerHTML = svg + description;
    },

    /**
     * Create histogram bins
     */
    createHistogramBins(values, numBins) {
        const min = Math.min(...values);
        const max = Math.max(...values);
        const binWidth = (max - min) / numBins || 1;
        
        const bins = [];
        for (let i = 0; i < numBins; i++) {
            const binMin = min + i * binWidth;
            const binMax = min + (i + 1) * binWidth;
            bins.push({
                min: Math.round(binMin),
                max: Math.round(binMax),
                center: Math.round((binMin + binMax) / 2),
                count: values.filter(v => v >= binMin && v < binMax).length
            });
        }
        // Include max value in last bin
        if (values.includes(max)) {
            bins[bins.length - 1].count += values.filter(v => v === max).length;
        }
        
        return bins;
    },

    /**
     * Create n-gram repetition heatmap
     */
    createNgramHeatmap(container, ngramData) {
        if (!ngramData || !ngramData.ngramReuse) {
            container.innerHTML = '<p class="no-data">No n-gram data available</p>';
            return;
        }

        const width = 500;
        const height = 200;
        const padding = { top: 30, right: 20, bottom: 40, left: 100 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="heatmap-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">N-gram Repetition Intensity</text>`;

        const nSizes = [2, 3, 4, 5];
        const rowHeight = chartHeight / nSizes.length;

        nSizes.forEach((n, rowIndex) => {
            const data = ngramData.ngramReuse[n];
            if (!data) return;

            const y = padding.top + rowIndex * rowHeight;
            
            // Row label
            svg += `<text x="${padding.left - 10}" y="${y + rowHeight/2 + 5}" text-anchor="end" 
                    class="row-label">${n}-gram</text>`;

            // Calculate reuse intensity across document positions
            const segments = 20;
            const segmentWidth = chartWidth / segments;
            
            for (let seg = 0; seg < segments; seg++) {
                const segStart = seg / segments;
                const segEnd = (seg + 1) / segments;
                
                // Count repetitions in this segment
                let intensity = 0;
                for (const [phrase, info] of Object.entries(data.repeated)) {
                    const normalizedPositions = info.positions.map(p => p / (data.totalNgrams || 1));
                    const inSegment = normalizedPositions.filter(p => p >= segStart && p < segEnd).length;
                    intensity += inSegment;
                }
                
                const maxIntensity = 10; // Normalize
                const normalizedIntensity = Math.min(1, intensity / maxIntensity);
                const opacity = 0.1 + normalizedIntensity * 0.8;

                svg += `<rect x="${padding.left + seg * segmentWidth}" y="${y + 2}" 
                        width="${segmentWidth - 1}" height="${rowHeight - 4}"
                        fill="${this.colors.accent}" opacity="${opacity}"
                        class="heatmap-cell"/>`;
            }
        });

        // X-axis
        svg += `<text x="${padding.left}" y="${height - 10}" class="axis-label">Start</text>`;
        svg += `<text x="${width - padding.right}" y="${height - 10}" text-anchor="end" class="axis-label">End</text>`;
        svg += `<text x="${width/2}" y="${height - 10}" text-anchor="middle" class="axis-label">Document Position</text>`;

        svg += '</svg>';

        // Legend
        const legend = `<div class="heatmap-legend">
            <span class="legend-label">Low repetition</span>
            <div class="legend-gradient"></div>
            <span class="legend-label">High repetition</span>
        </div>`;

        container.innerHTML = svg + legend;
    },

    /**
     * Create tone timeline chart
     */
    createToneTimeline(container, toneData) {
        if (!toneData || !toneData.sentenceTones || toneData.sentenceTones.length < 3) {
            container.innerHTML = '<p class="no-data">Insufficient data for tone timeline</p>';
            return;
        }

        const tones = toneData.sentenceTones;
        const width = 600;
        const height = 250;
        const padding = { top: 30, right: 20, bottom: 50, left: 60 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="timeline-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Tone Stability Timeline</text>`;

        // Grid
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartHeight / 4) * i;
            svg += `<line x1="${padding.left}" y1="${y}" x2="${width - padding.right}" y2="${y}" 
                    stroke="${this.colors.grid}" stroke-width="1"/>`;
        }

        // Center line (neutral)
        const centerY = padding.top + chartHeight / 2;
        svg += `<line x1="${padding.left}" y1="${centerY}" x2="${width - padding.right}" y2="${centerY}" 
                stroke="${this.colors.accent}" stroke-width="1" stroke-dasharray="4,4"/>`;

        // Dimensions to plot
        const dimensions = [
            { key: 'emotionalValence', label: 'Emotional', color: '#4a4a4a', scale: 0.1 },
            { key: 'registerScore', label: 'Register', color: '#7a7a7a', scale: 1 },
            { key: 'hedgingRate', label: 'Hedging', color: '#9a9a9a', scale: 10 }
        ];

        dimensions.forEach((dim, dimIndex) => {
            const values = tones.map(t => t[dim.key] * dim.scale);
            const points = values.map((v, i) => {
                const x = padding.left + (i / (values.length - 1)) * chartWidth;
                const y = centerY - v * (chartHeight / 2);
                return `${x},${Math.max(padding.top, Math.min(padding.top + chartHeight, y))}`;
            });

            // Line
            svg += `<polyline points="${points.join(' ')}" fill="none" stroke="${dim.color}" 
                    stroke-width="2" class="timeline-line"/>`;

            // Legend entry
            const legendX = padding.left + dimIndex * 100;
            svg += `<line x1="${legendX}" y1="${height - 15}" x2="${legendX + 20}" y2="${height - 15}" 
                    stroke="${dim.color}" stroke-width="2"/>`;
            svg += `<text x="${legendX + 25}" y="${height - 10}" class="legend-text">${dim.label}</text>`;
        });

        // Y-axis labels
        svg += `<text x="${padding.left - 5}" y="${padding.top + 5}" text-anchor="end" class="axis-label">High</text>`;
        svg += `<text x="${padding.left - 5}" y="${centerY + 5}" text-anchor="end" class="axis-label">Neutral</text>`;
        svg += `<text x="${padding.left - 5}" y="${padding.top + chartHeight}" text-anchor="end" class="axis-label">Low</text>`;

        // X-axis label
        svg += `<text x="${width/2}" y="${height - 30}" text-anchor="middle" class="axis-title">Sentences</text>`;

        svg += '</svg>';

        // Insight
        const stability = toneData.stability?.overall || 0.5;
        const insight = stability > 0.75
            ? '<p class="chart-insight ai-signal">⚠️ Tone remains unusually stable throughout - AI characteristic</p>'
            : '<p class="chart-insight human-signal">✓ Natural tone variation observed - human characteristic</p>';

        container.innerHTML = svg + insight;
    },

    /**
     * Create Zipf distribution chart (word frequency)
     */
    createZipfChart(container, tokens) {
        if (!tokens || tokens.length < 50) {
            container.innerHTML = '<p class="no-data">Insufficient tokens for Zipf analysis</p>';
            return;
        }

        const freq = Utils.frequencyDistribution(tokens);
        const sortedFreqs = Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 50);

        const width = 500;
        const height = 250;
        const padding = { top: 30, right: 20, bottom: 50, left: 60 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Use log scale
        const maxFreq = sortedFreqs[0][1];
        const logMax = Math.log(maxFreq);

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="zipf-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Word Frequency Distribution (Zipf)</text>`;

        // Expected Zipf curve (reference)
        let expectedPath = '';
        for (let i = 0; i < sortedFreqs.length; i++) {
            const rank = i + 1;
            const expectedFreq = maxFreq / rank; // Zipf's law
            const x = padding.left + (i / sortedFreqs.length) * chartWidth;
            const y = padding.top + chartHeight - (Math.log(expectedFreq) / logMax) * chartHeight;
            expectedPath += `${i === 0 ? 'M' : 'L'} ${x},${y}`;
        }
        svg += `<path d="${expectedPath}" fill="none" stroke="${this.colors.grid}" stroke-width="2" 
                stroke-dasharray="5,5" class="expected-line"/>`;

        // Actual distribution
        let actualPath = '';
        sortedFreqs.forEach(([word, count], i) => {
            const x = padding.left + (i / sortedFreqs.length) * chartWidth;
            const y = padding.top + chartHeight - (Math.log(count) / logMax) * chartHeight;
            actualPath += `${i === 0 ? 'M' : 'L'} ${x},${y}`;
        });
        svg += `<path d="${actualPath}" fill="none" stroke="${this.colors.accent}" stroke-width="2" class="actual-line"/>`;

        // Points
        sortedFreqs.forEach(([word, count], i) => {
            const x = padding.left + (i / sortedFreqs.length) * chartWidth;
            const y = padding.top + chartHeight - (Math.log(count) / logMax) * chartHeight;
            svg += `<circle cx="${x}" cy="${y}" r="3" fill="${this.colors.accent}" class="zipf-point" 
                    data-word="${word}" data-count="${count}"/>`;
        });

        // Axes
        svg += `<line x1="${padding.left}" y1="${padding.top + chartHeight}" x2="${width - padding.right}" 
                y2="${padding.top + chartHeight}" stroke="${this.colors.accent}" stroke-width="1"/>`;
        svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" 
                y2="${padding.top + chartHeight}" stroke="${this.colors.accent}" stroke-width="1"/>`;

        svg += `<text x="${width/2}" y="${height - 10}" text-anchor="middle" class="axis-title">Word Rank</text>`;
        svg += `<text x="15" y="${height/2}" text-anchor="middle" transform="rotate(-90, 15, ${height/2})" 
                class="axis-title">Log Frequency</text>`;

        // Legend
        svg += `<line x1="${width - 150}" y1="${padding.top + 10}" x2="${width - 130}" y2="${padding.top + 10}" 
                stroke="${this.colors.grid}" stroke-width="2" stroke-dasharray="5,5"/>`;
        svg += `<text x="${width - 125}" y="${padding.top + 14}" class="legend-text">Expected</text>`;
        svg += `<line x1="${width - 150}" y1="${padding.top + 25}" x2="${width - 130}" y2="${padding.top + 25}" 
                stroke="${this.colors.accent}" stroke-width="2"/>`;
        svg += `<text x="${width - 125}" y="${padding.top + 29}" class="legend-text">Actual</text>`;

        svg += '</svg>';

        container.innerHTML = svg;
    },

    /**
     * Create word co-occurrence network graph
     */
    createCooccurrenceGraph(container, tokens, windowSize = 3) {
        if (!tokens || tokens.length < 20) {
            container.innerHTML = '<p class="no-data">Insufficient tokens for network graph</p>';
            return;
        }

        // Build co-occurrence matrix
        const cooccurrences = {};
        const wordCounts = {};
        
        // Filter to content words only
        const contentWords = tokens.filter(t => 
            t.length > 3 && 
            !Utils.functionWords.includes(t.toLowerCase())
        );

        for (let i = 0; i < contentWords.length - windowSize; i++) {
            for (let j = 0; j < windowSize; j++) {
                for (let k = j + 1; k < windowSize; k++) {
                    const word1 = contentWords[i + j].toLowerCase();
                    const word2 = contentWords[i + k].toLowerCase();
                    
                    if (word1 === word2) continue;
                    
                    const key = [word1, word2].sort().join('|');
                    cooccurrences[key] = (cooccurrences[key] || 0) + 1;
                    wordCounts[word1] = (wordCounts[word1] || 0) + 1;
                    wordCounts[word2] = (wordCounts[word2] || 0) + 1;
                }
            }
        }

        // Get top connections
        const topConnections = Object.entries(cooccurrences)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 30);

        // Get unique words from top connections
        const words = new Set();
        topConnections.forEach(([key]) => {
            const [w1, w2] = key.split('|');
            words.add(w1);
            words.add(w2);
        });

        const wordArray = Array.from(words).slice(0, 15);
        
        const width = 500;
        const height = 400;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = Math.min(width, height) * 0.35;

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="network-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Word Co-occurrence Network</text>`;

        // Calculate node positions in a circle
        const nodePositions = {};
        wordArray.forEach((word, i) => {
            const angle = (2 * Math.PI * i) / wordArray.length - Math.PI / 2;
            nodePositions[word] = {
                x: centerX + radius * Math.cos(angle),
                y: centerY + radius * Math.sin(angle),
                count: wordCounts[word] || 1
            };
        });

        // Draw edges
        const maxWeight = Math.max(...topConnections.map(([, w]) => w));
        topConnections.forEach(([key, weight]) => {
            const [w1, w2] = key.split('|');
            if (nodePositions[w1] && nodePositions[w2]) {
                const opacity = 0.2 + (weight / maxWeight) * 0.6;
                const strokeWidth = 1 + (weight / maxWeight) * 3;
                svg += `<line x1="${nodePositions[w1].x}" y1="${nodePositions[w1].y}" 
                        x2="${nodePositions[w2].x}" y2="${nodePositions[w2].y}"
                        stroke="${this.colors.accent}" stroke-width="${strokeWidth}" 
                        opacity="${opacity}" class="network-edge"/>`;
            }
        });

        // Draw nodes
        const maxCount = Math.max(...Object.values(nodePositions).map(n => n.count));
        for (const [word, pos] of Object.entries(nodePositions)) {
            const nodeRadius = 8 + (pos.count / maxCount) * 15;
            svg += `<circle cx="${pos.x}" cy="${pos.y}" r="${nodeRadius}" 
                    fill="${this.colors.accent}" class="network-node"/>`;
            svg += `<text x="${pos.x}" y="${pos.y + nodeRadius + 12}" 
                    text-anchor="middle" class="node-label">${word}</text>`;
        }

        svg += '</svg>';

        container.innerHTML = svg;
    },

    /**
     * Create feature contribution bar chart
     */
    createFeatureContributionChart(container, categoryResults) {
        if (!categoryResults || categoryResults.length === 0) {
            container.innerHTML = '<p class="no-data">No category data available</p>';
            return;
        }

        const width = 500;
        const height = 300;
        const padding = { top: 30, right: 20, bottom: 30, left: 150 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        // Sort by contribution
        const sorted = [...categoryResults]
            .map(c => ({
                name: c.name,
                probability: c.aiProbability,
                confidence: c.confidence,
                contribution: c.aiProbability * c.confidence
            }))
            .sort((a, b) => b.contribution - a.contribution);

        const barHeight = chartHeight / sorted.length - 4;
        const maxContribution = Math.max(...sorted.map(c => c.contribution));

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="contribution-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Feature Contribution to AI Probability</text>`;

        sorted.forEach((cat, i) => {
            const y = padding.top + i * (barHeight + 4);
            const barWidth = maxContribution > 0 ? (cat.contribution / maxContribution) * chartWidth : 0;
            
            // Category name
            svg += `<text x="${padding.left - 5}" y="${y + barHeight/2 + 4}" 
                    text-anchor="end" class="category-label">${cat.name}</text>`;
            
            // Bar
            const opacity = 0.4 + cat.contribution * 0.6;
            svg += `<rect x="${padding.left}" y="${y}" width="${barWidth}" height="${barHeight}"
                    fill="${this.colors.accent}" opacity="${opacity}" rx="2" class="contribution-bar"/>`;
            
            // Value
            svg += `<text x="${padding.left + barWidth + 5}" y="${y + barHeight/2 + 4}" 
                    class="value-label">${Math.round(cat.probability * 100)}%</text>`;
        });

        // Axis
        svg += `<line x1="${padding.left}" y1="${padding.top}" x2="${padding.left}" 
                y2="${height - padding.bottom}" stroke="${this.colors.accent}" stroke-width="1"/>`;

        svg += '</svg>';

        container.innerHTML = svg;
    },

    /**
     * Create variance comparison chart
     */
    createVarianceChart(container, data) {
        if (!data) {
            container.innerHTML = '<p class="no-data">No variance data available</p>';
            return;
        }

        const metrics = [
            { label: 'Sentence Length', value: data.syntax?.sentenceLengthVariance || 0.5, expected: 0.6 },
            { label: 'Word Length', value: data.lexical?.wordLengthVariance || 0.5, expected: 0.5 },
            { label: 'Paragraph Size', value: data.structure?.paragraphVariance || 0.5, expected: 0.55 },
            { label: 'Tone Stability', value: data.tone?.stability || 0.5, expected: 0.4 },
            { label: 'Vocabulary Density', value: data.lexical?.ttrVariance || 0.5, expected: 0.5 }
        ];

        const width = 400;
        const height = 250;
        const padding = { top: 30, right: 30, bottom: 30, left: 120 };
        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;
        const barHeight = chartHeight / metrics.length - 8;

        let svg = `<svg viewBox="0 0 ${width} ${height}" class="variance-svg">`;
        svg += `<text x="${width/2}" y="20" text-anchor="middle" class="chart-title">Variance vs Expected (Human)</text>`;

        // Center line (expected)
        const centerX = padding.left + chartWidth / 2;
        svg += `<line x1="${centerX}" y1="${padding.top}" x2="${centerX}" y2="${height - padding.bottom}"
                stroke="${this.colors.grid}" stroke-width="2" stroke-dasharray="4,4"/>`;
        svg += `<text x="${centerX}" y="${height - 10}" text-anchor="middle" class="axis-label">Expected</text>`;

        metrics.forEach((metric, i) => {
            const y = padding.top + i * (barHeight + 8);
            
            // Label
            svg += `<text x="${padding.left - 5}" y="${y + barHeight/2 + 4}" 
                    text-anchor="end" class="metric-label">${metric.label}</text>`;

            // Deviation from expected
            const deviation = metric.value - metric.expected;
            const maxDev = 0.5;
            const deviationWidth = (Math.abs(deviation) / maxDev) * (chartWidth / 2);
            const barX = deviation >= 0 ? centerX : centerX - deviationWidth;
            
            const color = Math.abs(deviation) > 0.2 ? this.colors.ai : this.colors.human;
            
            svg += `<rect x="${barX}" y="${y}" width="${deviationWidth}" height="${barHeight}"
                    fill="${color}" rx="2"/>`;

            // Value indicator
            const valueX = padding.left + (metric.value / 1) * chartWidth;
            svg += `<circle cx="${valueX}" cy="${y + barHeight/2}" r="4" fill="${this.colors.accent}"/>`;
        });

        // Legend
        svg += `<text x="${padding.left}" y="${height - 5}" class="axis-label">Lower Variance</text>`;
        svg += `<text x="${width - padding.right}" y="${height - 5}" text-anchor="end" class="axis-label">Higher Variance</text>`;

        svg += '</svg>';

        container.innerHTML = svg;
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedVisualizations;
}
