/**
 * VERITAS — Report Exporter
 * Generates professional DOCX and PDF reports with dynamic template-based content
 */

const ReportExporter = {
    // Sunrise Model v3.0 - ML-derived category weights (98.08% accuracy)
    // Aggregated from 37 individual feature weights to 14 analyzer categories
    categoryWeightInfo: {
        1: { name: 'Grammar & Error Patterns', weight: 0.01, description: 'Low ML weight. Analyzes grammatical consistency (complexity_cv: 0.61%). Human errors are inconsistent; AI has near-perfect grammar.' },
        2: { name: 'Sentence Structure & Syntax', weight: 0.21, description: 'High ML weight (21%). Measures sentence_length_std, range, cv, burstiness. AI tends toward uniform lengths; humans show natural variance.' },
        3: { name: 'Lexical Choice & Vocabulary', weight: 0.22, description: 'Highest ML weight (22%). Evaluates hapax_count (11.75%), unique_word_count (10.21%), type_token_ratio. AI shows predictable vocabulary.' },
        4: { name: 'Dialect & Regional Consistency', weight: 0.01, description: 'Low ML weight. Checks for consistent regional spelling and terminology patterns.' },
        5: { name: 'Archaic / Historical Grammar', weight: 0.01, description: 'Low ML weight. Detects anachronistic language use and historical term misuse.' },
        6: { name: 'Discourse & Coherence', weight: 0.02, description: 'Low ML weight. Analyzes logical flow via sentence_similarity_avg (0.83%).' },
        7: { name: 'Semantic & Pragmatic Features', weight: 0.02, description: 'Low ML weight. Examines meaning depth and contextual appropriateness.' },
        8: { name: 'Statistical Language Model Indicators', weight: 0.02, description: 'Low ML weight. Zipf analysis (zipf_slope: 0.65%, zipf_r_squared: 0.25%).' },
        9: { name: 'Authorship Consistency', weight: 0.05, description: 'Moderate ML weight. Tracks stylistic drift via overall_uniformity (0.63%).' },
        10: { name: 'Meta-Patterns Unique to AI', weight: 0.02, description: 'Low ML weight. Detects AI-specific hedging phrases and balanced arguments.' },
        11: { name: 'Metadata & Formatting', weight: 0.40, description: 'HIGHEST ML weight (40%). avg_paragraph_length (16.85%), paragraph_count (14.66%), paragraph_length_cv (7.36%). Primary AI detection signal.' },
        12: { name: 'Repetition Patterns', weight: 0.02, description: 'Low ML weight. bigram_repetition_rate (0.58%), trigram_repetition_rate (0.67%).' },
        13: { name: 'Tone Stability', weight: 0.02, description: 'Low ML weight. burstiness_sentence (1.19%), burstiness_word_length (0.18%).' },
        14: { name: 'Part of Speech Patterns', weight: 0.01, description: 'Low ML weight. Examines verb/adverb patterns derived from other features.' }
    },

    /**
     * Generate and download DOCX report
     */
    async exportDocx(analysisResult, originalText) {
        const reportContent = this.generateReportContent(analysisResult, originalText);
        const docxBlob = await this.createDocxBlob(reportContent);
        
        // Download
        const filename = `veritas-report-${new Date().toISOString().slice(0, 10)}.docx`;
        this.downloadBlob(docxBlob, filename);
        
        return filename;
    },

    /**
     * Generate and download PDF report
     */
    async exportPdf(analysisResult, originalText) {
        const reportContent = this.generateReportContent(analysisResult, originalText);
        
        // Generate HTML for PDF conversion
        const htmlContent = this.generateHtmlReport(reportContent, analysisResult);
        
        // Check if html2pdf is available
        if (typeof html2pdf !== 'undefined') {
            const element = document.createElement('div');
            element.innerHTML = htmlContent;
            element.style.width = '190mm';
            element.style.position = 'absolute';
            element.style.left = '-9999px';
            element.style.top = '0';
            element.style.background = '#ffffff';
            document.body.appendChild(element);
            
            // Wait for images and fonts to load
            await new Promise(resolve => setTimeout(resolve, 100));
            
            const opt = {
                margin: [8, 8, 8, 8],
                filename: `veritas-report-${new Date().toISOString().slice(0, 10)}.pdf`,
                image: { type: 'jpeg', quality: 0.95 },
                html2canvas: { 
                    scale: 2, 
                    useCORS: true,
                    logging: false,
                    letterRendering: true,
                    allowTaint: true,
                    backgroundColor: '#ffffff',
                    removeContainer: true
                },
                jsPDF: { 
                    unit: 'mm', 
                    format: 'a4', 
                    orientation: 'portrait',
                    compress: true
                },
                pagebreak: { 
                    mode: ['avoid-all', 'css', 'legacy'],
                    before: '.page-break-before',
                    after: '.page-break-after',
                    avoid: '.category-card, .summary-box, table, .signal-summary'
                }
            };
            
            try {
                await html2pdf().set(opt).from(element).save();
            } finally {
                document.body.removeChild(element);
            }
        } else if (typeof jspdf !== 'undefined' || typeof jsPDF !== 'undefined') {
            // Fallback to jsPDF
            const { jsPDF } = window.jspdf || { jsPDF: window.jsPDF };
            const doc = new jsPDF();
            
            // Add content as text
            const plainText = this.generatePlainTextReport(reportContent);
            const lines = doc.splitTextToSize(plainText, 180);
            
            let y = 20;
            const pageHeight = doc.internal.pageSize.height;
            
            for (const line of lines) {
                if (y > pageHeight - 20) {
                    doc.addPage();
                    y = 20;
                }
                doc.text(line, 15, y);
                y += 6;
            }
            
            doc.save(`veritas-report-${new Date().toISOString().slice(0, 10)}.pdf`);
        } else {
            // Fallback: print-friendly HTML window
            const printWindow = window.open('', '_blank');
            printWindow.document.write(htmlContent);
            printWindow.document.close();
            printWindow.print();
        }
        
        return true;
    },

    /**
     * Generate HTML report for PDF export
     */
    generateHtmlReport(report, analysisResult) {
        const probability = report.summary.probability;
        
        // Check for humanized AI signals
        const humanizerSignals = analysisResult.humanizerSignals || {};
        const falsePositiveRisk = analysisResult.falsePositiveRisk || {};
        const hasHighDisagreement = falsePositiveRisk.risks?.some(r => r.type === 'analyzer_disagreement' && r.severity === 'high');
        const isLikelyHumanized = humanizerSignals.isLikelyHumanized || hasHighDisagreement;
        
        // Choose bar color - purple for humanized, red/yellow/green otherwise
        const barColor = isLikelyHumanized ? '#9333ea' : (probability >= 60 ? '#ef4444' : (probability >= 40 ? '#f59e0b' : '#10b981'));
        
        // Build verbose evidence summaries for each category
        const verboseEvidence = this.buildVerboseEvidence(report, analysisResult);
        
        let html = `<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>VERITAS AI Detection Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            font-size: 10pt; 
            line-height: 1.5; 
            color: #1a1a1a;
            padding: 12px 15px;
            max-width: 100%;
            margin: 0;
            background: #ffffff;
        }
        h1 { font-size: 18pt; margin-bottom: 3px; color: #111; }
        h2 { font-size: 13pt; margin: 10px 0 5px; color: #333; border-bottom: 1px solid #ddd; padding-bottom: 3px; page-break-after: avoid; }
        h3 { font-size: 11pt; margin: 6px 0 3px; color: #444; }
        h4 { font-size: 10pt; margin: 5px 0 2px; color: #555; }
        p { margin-bottom: 5px; }
        .header { text-align: center; margin-bottom: 12px; border-bottom: 2px solid #333; padding-bottom: 8px; }
        .meta { color: #666; font-size: 9pt; }
        .summary-box { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border: 1px solid #e5e5e5; border-radius: 8px; padding: 15px; margin: 12px 0; page-break-inside: avoid; }
        .verdict-container { display: flex; align-items: center; gap: 15px; margin: 10px 0; }
        .verdict-gauge { width: 80px; height: 80px; position: relative; flex-shrink: 0; }
        .verdict-gauge-circle { width: 100%; height: 100%; border-radius: 50%; background: conic-gradient(${barColor} calc(${probability} * 3.6deg), #e5e5e5 0deg); display: flex; align-items: center; justify-content: center; }
        .verdict-gauge-inner { width: 60px; height: 60px; border-radius: 50%; background: white; display: flex; align-items: center; justify-content: center; flex-direction: column; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .verdict-gauge-value { font-size: 16pt; font-weight: bold; color: ${barColor}; line-height: 1; }
        .verdict-gauge-label { font-size: 6pt; color: #666; text-transform: uppercase; }
        .verdict-info { flex: 1; }
        .verdict-badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-weight: bold; font-size: 10pt; margin-bottom: 6px; }
        .verdict-badge.high { background: #fee2e2; color: #b91c1c; }
        .verdict-badge.moderate { background: #fef3c7; color: #b45309; }
        .verdict-badge.low { background: #d1fae5; color: #047857; }
        .verdict-badge.humanized { background: #f3e8ff; color: #7c3aed; }
        .confidence-range { display: flex; align-items: center; gap: 8px; margin-top: 8px; font-size: 8pt; color: #666; }
        .confidence-bar { flex: 1; height: 6px; background: #e5e5e5; border-radius: 3px; position: relative; max-width: 200px; }
        .confidence-range-fill { position: absolute; height: 100%; background: linear-gradient(90deg, #d1d5db, ${barColor}); border-radius: 3px; }
        .confidence-marker { position: absolute; width: 2px; height: 10px; background: #333; top: -2px; transform: translateX(-50%); }
        .verdict { font-size: 11pt; font-weight: bold; margin: 5px 0; }
        .verdict.high { color: #ef4444; }
        .verdict.moderate { color: #f59e0b; }
        .verdict.low { color: #10b981; }
        .verdict.humanized { color: #9333ea; }
        .humanized-warning { background: #faf5ff; border: 1px solid #d8b4fe; border-radius: 4px; padding: 8px; margin: 8px 0; font-size: 9pt; }
        .humanized-warning h4 { color: #7c3aed; margin-bottom: 4px; }
        .probability-bar { height: 14px; background: #e5e5e5; border-radius: 7px; overflow: hidden; margin: 5px 0; }
        .probability-fill { height: 100%; background: ${barColor}; min-width: 1px; }
        .stat-grid { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0; }
        .stat-item { background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); padding: 10px 12px; border-radius: 6px; flex: 1; min-width: 100px; border: 1px solid #e5e5e5; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
        .stat-label { font-size: 7pt; color: #888; text-transform: uppercase; display: block; letter-spacing: 0.5px; margin-bottom: 2px; }
        .stat-value { font-size: 12pt; font-weight: bold; color: #333; }
        .category-card { border: 1px solid #e5e5e5; border-radius: 4px; padding: 8px; margin: 6px 0; page-break-inside: avoid; break-inside: avoid; }
        .category-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
        .category-name { font-weight: bold; font-size: 9pt; }
        .category-score { font-weight: bold; padding: 2px 6px; border-radius: 10px; font-size: 8pt; }
        .score-high { background: #fee2e2; color: #b91c1c; }
        .score-moderate { background: #fef3c7; color: #b45309; }
        .score-low { background: #d1fae5; color: #047857; }
        .category-bar { height: 5px; background: #e5e5e5; border-radius: 2px; overflow: hidden; margin: 3px 0; }
        .category-fill { height: 100%; border-radius: 2px; min-width: 1px; }
        .weight-info { font-size: 7pt; color: #888; margin-top: 4px; padding-top: 4px; border-top: 1px solid #eee; }
        .finding { padding: 2px 0 2px 10px; border-left: 2px solid #ddd; margin: 2px 0; font-size: 8pt; }
        .finding.ai { border-left-color: #ef4444; background: #fff5f5; }
        .finding.human { border-left-color: #10b981; background: #f0fdf4; }
        .finding.mixed { border-left-color: #f59e0b; background: #fffbeb; }
        .evidence-stats { font-size: 7pt; color: #666; margin-top: 3px; padding: 3px 6px; background: #f9fafb; border-radius: 2px; }
        .evidence-stats strong { color: #333; }
        .chart { margin: 8px 0; }
        .bar-chart { width: 100%; }
        .bar-row { display: flex; align-items: center; margin: 3px 0; }
        .bar-label { width: 160px; font-size: 8pt; text-align: right; padding-right: 8px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .bar-track { flex: 1; height: 14px; background: #e5e5e5; border-radius: 3px; overflow: hidden; }
        .bar-fill-ai { height: 100%; background: linear-gradient(90deg, #6b7280, #374151); border-radius: 3px; min-width: 2px; }
        .bar-value { width: 45px; text-align: right; font-size: 9pt; font-weight: bold; padding-left: 6px; }
        .disclaimer { background: #fff8e6; border: 1px solid #ffd666; border-radius: 4px; padding: 8px; margin: 10px 0; font-size: 8pt; page-break-inside: avoid; }
        .methodology { background: #f0f7ff; border-radius: 4px; padding: 8px; margin: 10px 0; font-size: 8pt; page-break-inside: avoid; }
        .signal-summary { background: #fafafa; border: 1px solid #e5e5e5; border-radius: 4px; padding: 8px; margin: 8px 0; page-break-inside: avoid; }
        .signal-summary h3 { margin-top: 0; font-size: 10pt; }
        .signal-grid { display: flex; flex-wrap: wrap; gap: 6px; }
        .signal-grid > div { flex: 1; min-width: 45%; }
        .signal-item { padding: 4px; border-radius: 3px; margin-bottom: 3px; }
        .signal-item.ai-signal { background: #fee2e2; border-left: 2px solid #ef4444; }
        .signal-item.human-signal { background: #d1fae5; border-left: 2px solid #10b981; }
        .signal-item .signal-label { font-size: 7pt; color: #666; }
        .signal-item .signal-value { font-size: 9pt; font-weight: bold; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 9pt; page-break-inside: avoid; }
        th, td { border: 1px solid #ddd; padding: 5px 6px; text-align: left; }
        th { background: #f5f5f5; font-weight: 600; font-size: 8pt; }
        td { font-size: 8pt; }
        .page-break { page-break-before: always; }
        .no-break { page-break-inside: avoid; break-inside: avoid; }
        @media print { 
            body { padding: 0; }
            .page-break { page-break-before: always; }
            .category-card, .summary-box, table, .signal-summary { page-break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>◈ VERITAS</h1>
        <p class="meta">AI Text Detection Analysis Report</p>
        <p class="meta">Powered by ${report.modelInfo.name} Model v${report.modelInfo.version} | ${(report.modelInfo.accuracy * 100).toFixed(1)}% Accuracy | ${report.modelInfo.trainingSamples.toLocaleString()} Training Samples</p>
        <p class="meta">Generated: ${new Date(report.generatedAt).toLocaleString()}</p>
    </div>

    <div class="summary-box">
        <h2 style="margin-top:0;border:none;font-size:13pt;margin-bottom:12px;">Executive Summary</h2>
        
        <div class="verdict-container">
            <div class="verdict-gauge">
                <div class="verdict-gauge-circle">
                    <div class="verdict-gauge-inner">
                        <span class="verdict-gauge-value">${probability}%</span>
                        <span class="verdict-gauge-label">AI Prob</span>
                    </div>
                </div>
            </div>
            <div class="verdict-info">
                <span class="verdict-badge ${isLikelyHumanized ? 'humanized' : (probability >= 60 ? 'high' : (probability >= 40 ? 'moderate' : 'low'))}">${isLikelyHumanized ? '🔄 Possibly Humanized AI' : report.verdict.label}</span>
                <p style="margin:4px 0;font-size:9pt;color:#555;">${isLikelyHumanized ? 'This text shows AI origin with humanization attempts — likely AI-generated then modified by tools or manual editing to appear more human-like.' : report.verdict.description}</p>
                <div class="confidence-range">
                    <span>Confidence: ${report.summary.confidence}%</span>
                    <div class="confidence-bar">
                        <div class="confidence-range-fill" style="left:${Math.round(report.verdict.confidenceInterval?.lower * 100 || 8)}%;width:${Math.round((report.verdict.confidenceInterval?.upper - report.verdict.confidenceInterval?.lower) * 100 || 68)}%;"></div>
                        <div class="confidence-marker" style="left:${probability}%;"></div>
                    </div>
                    <span>${Math.round(report.verdict.confidenceInterval?.lower * 100 || 8)}% — ${Math.round(report.verdict.confidenceInterval?.upper * 100 || 76)}%</span>
                </div>
            </div>
        </div>
        
        ${isLikelyHumanized ? `<div style="background:#faf5ff;border:1px solid #d8b4fe;border-radius:6px;padding:8px 12px;margin:10px 0;"><p style="font-size:9pt;color:#7c3aed;margin:0;"><strong>⚠️ Humanization Detected:</strong> High disagreement between detection categories suggests post-processing or editing of AI output. Manual review recommended.</p></div>` : ''}
        
        <p style="font-size:9pt;color:#444;line-height:1.5;margin-top:10px;">${report.summary.text}</p>
    </div>

    <div class="stat-grid">
        <div class="stat-item"><div class="stat-label">Words</div><div class="stat-value">${report.statistics.wordCount.toLocaleString()}</div></div>
        <div class="stat-item"><div class="stat-label">Sentences</div><div class="stat-value">${report.statistics.sentenceCount.toLocaleString()}</div></div>
        <div class="stat-item"><div class="stat-label">Paragraphs</div><div class="stat-value">${report.statistics.paragraphCount.toLocaleString()}</div></div>
        <div class="stat-item"><div class="stat-label">Analysis Time</div><div class="stat-value">${report.statistics.analysisTime}</div></div>
    </div>`;

        // Add detection caveats/warnings if any
        if (falsePositiveRisk.hasRisks && falsePositiveRisk.risks?.length > 0) {
            html += `
    <div class="disclaimer" style="background: ${isLikelyHumanized ? '#faf5ff' : '#fff8e6'}; border-color: ${isLikelyHumanized ? '#d8b4fe' : '#ffd666'};">
        <h4 style="margin: 0 0 5px 0; color: ${isLikelyHumanized ? '#7c3aed' : '#b45309'};">⚠️ Detection Caveats</h4>`;
            for (const risk of falsePositiveRisk.risks) {
                const riskColor = risk.severity === 'high' ? '#b91c1c' : (risk.severity === 'medium' ? '#b45309' : '#666');
                html += `
        <p style="font-size:8pt;margin:3px 0;color:${riskColor};">• ${risk.message}</p>`;
            }
            html += `
    </div>`;
        }

        // Add comprehensive statistics section matching the app's statistics page
        const advStats = analysisResult.advancedStats || {};
        const basicStats = analysisResult.stats || {};
        
        // Helper functions
        const formatNum = (n, decimals = 2) => n != null ? Number(n).toFixed(decimals) : 'N/A';
        const formatPct = (n) => n != null ? `${(n * 100).toFixed(1)}%` : 'N/A';
        const getIndicator = (val, thresholds, higherIsBetter = false) => {
            if (val == null) return '⚪';
            const [warn, good] = thresholds;
            if (higherIsBetter) {
                return val >= good ? '🟢' : (val >= warn ? '🟡' : '🔴');
            }
            return val <= good ? '🟢' : (val <= warn ? '🟡' : '🔴');
        };
        
        if (Object.keys(advStats).length > 0) {
            html += `
    <h2>Complete Statistical Analysis</h2>
    <p style="font-size:8pt;color:#666;margin-bottom:10px;">🔴 = AI-like | 🟡 = Uncertain | 🟢 = Human-like</p>
    
    <!-- Vocabulary Richness -->
    <h3>📚 Vocabulary Richness</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Unique Words</td><td>${advStats.vocabulary?.uniqueWords?.toLocaleString() || 0}</td><td>—</td></tr>
        <tr><td>Type-Token Ratio (TTR)</td><td>${formatPct(advStats.vocabulary?.typeTokenRatio)}</td><td>${getIndicator(advStats.vocabulary?.typeTokenRatio, [0.3, 0.5], true)}</td></tr>
        <tr><td>Root TTR (Guiraud's R)</td><td>${formatNum(advStats.vocabulary?.rootTTR)}</td><td>—</td></tr>
        <tr><td>Hapax Legomena Ratio</td><td>${formatPct(advStats.vocabulary?.hapaxLegomenaRatio)}</td><td>${getIndicator(advStats.vocabulary?.hapaxLegomenaRatio, [0.35, 0.5], true)}</td></tr>
        <tr><td>Dis Legomena Ratio</td><td>${formatPct(advStats.vocabulary?.disLegomenaRatio)}</td><td>—</td></tr>
        <tr><td>Yule's K</td><td>${formatNum(advStats.vocabulary?.yulesK, 1)}</td><td>${getIndicator(advStats.vocabulary?.yulesK, [150, 100])}</td></tr>
        <tr><td>Simpson's D</td><td>${formatNum(advStats.vocabulary?.simpsonsD, 4)}</td><td>${getIndicator(advStats.vocabulary?.simpsonsD, [0.02, 0.01])}</td></tr>
        <tr><td>Honore's R</td><td>${formatNum(advStats.vocabulary?.honoresR, 0)}</td><td>—</td></tr>
        <tr><td>Brunet's W</td><td>${formatNum(advStats.vocabulary?.brunetsW, 1)}</td><td>—</td></tr>
    </table>
    
    <!-- Sentence Analysis -->
    <h3>📝 Sentence Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Mean Length</td><td>${formatNum(advStats.sentences?.mean, 1)} words</td><td>—</td></tr>
        <tr><td>Median Length</td><td>${formatNum(advStats.sentences?.median, 1)} words</td><td>—</td></tr>
        <tr><td>Std Deviation</td><td>${formatNum(advStats.sentences?.stdDev, 2)}</td><td>—</td></tr>
        <tr><td>Min / Max</td><td>${advStats.sentences?.min || 0} / ${advStats.sentences?.max || 0}</td><td>—</td></tr>
        <tr><td>Coeff. of Variation</td><td>${formatNum(advStats.sentences?.coefficientOfVariation, 3)}</td><td>${getIndicator(advStats.sentences?.coefficientOfVariation, [0.35, 0.5], true)}</td></tr>
        <tr><td>Skewness</td><td>${formatNum(advStats.sentences?.skewness, 3)}</td><td>—</td></tr>
        <tr><td>Kurtosis</td><td>${formatNum(advStats.sentences?.kurtosis, 3)}</td><td>—</td></tr>
        <tr><td>Gini Coefficient</td><td>${formatNum(advStats.sentences?.gini, 3)}</td><td>${getIndicator(advStats.sentences?.gini, [0.15, 0.25], true)}</td></tr>
    </table>
    
    <!-- Zipf's Law -->
    <h3>📈 Zipf's Law Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Zipf Compliance</td><td>${formatPct(advStats.zipf?.compliance)}</td><td>${getIndicator(advStats.zipf?.compliance, [0.7, 0.85], true)}</td></tr>
        <tr><td>Log-Log Slope</td><td>${formatNum(advStats.zipf?.slope, 3)} (ideal: -1)</td><td>${getIndicator(Math.abs((advStats.zipf?.slope || 0) + 1), [0.3, 0.15])}</td></tr>
        <tr><td>R² (Fit Quality)</td><td>${formatNum(advStats.zipf?.rSquared, 3)}</td><td>—</td></tr>
        <tr><td>Deviation from Ideal</td><td>${formatNum(advStats.zipf?.deviation, 3)}</td><td>—</td></tr>
    </table>
    
    <!-- Readability -->
    <h3>📖 Readability Metrics</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Avg Syllables/Word</td><td>${formatNum(advStats.readability?.avgSyllablesPerWord, 2)}</td></tr>
        <tr><td>Flesch Reading Ease</td><td>${formatNum(advStats.readability?.fleschReadingEase, 1)}</td></tr>
        <tr><td>Flesch-Kincaid Grade</td><td>${formatNum(advStats.readability?.fleschKincaidGrade, 1)}</td></tr>
        <tr><td>Gunning Fog Index</td><td>${formatNum(advStats.readability?.gunningFogIndex, 1)}</td></tr>
        <tr><td>Coleman-Liau Index</td><td>${formatNum(advStats.readability?.colemanLiauIndex, 1)}</td></tr>
        <tr><td>SMOG Index</td><td>${formatNum(advStats.readability?.smogIndex, 1)}</td></tr>
        <tr><td>ARI</td><td>${formatNum(advStats.readability?.ariIndex, 1)}</td></tr>
        <tr><td>Complex Word %</td><td>${formatNum(advStats.readability?.complexWordPercentage, 1)}%</td></tr>
    </table>
    
    <!-- Burstiness & Uniformity -->
    <h3>⚡ Burstiness & Uniformity</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Sentence Length Burstiness</td><td>${formatNum(advStats.burstiness?.sentenceLength, 3)}</td><td>${getIndicator(advStats.burstiness?.sentenceLength, [0.1, 0.25], true)}</td></tr>
        <tr><td>Word Length Burstiness</td><td>${formatNum(advStats.burstiness?.wordLength, 3)}</td><td>—</td></tr>
        <tr><td>Overall Uniformity</td><td>${formatPct(advStats.burstiness?.overallUniformity)}</td><td>${getIndicator(advStats.burstiness?.overallUniformity, [0.7, 0.5])}</td></tr>
    </table>
    
    <!-- N-gram Analysis -->
    <h3>🔗 N-gram & Phrase Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Unique Bigrams</td><td>${advStats.ngrams?.uniqueBigrams?.toLocaleString() || 0}</td><td>—</td></tr>
        <tr><td>Unique Trigrams</td><td>${advStats.ngrams?.uniqueTrigrams?.toLocaleString() || 0}</td><td>—</td></tr>
        <tr><td>Bigram Repetition Rate</td><td>${formatPct(advStats.ngrams?.bigramRepetitionRate)}</td><td>${getIndicator(advStats.ngrams?.bigramRepetitionRate, [0.4, 0.25])}</td></tr>
        <tr><td>Trigram Repetition Rate</td><td>${formatPct(advStats.ngrams?.trigramRepetitionRate)}</td><td>${getIndicator(advStats.ngrams?.trigramRepetitionRate, [0.2, 0.1])}</td></tr>
        <tr><td>Quadgram Repetition Rate</td><td>${formatPct(advStats.ngrams?.quadgramRepetitionRate)}</td><td>${getIndicator(advStats.ngrams?.quadgramRepetitionRate, [0.1, 0.05])}</td></tr>
        <tr><td><strong>Repeated Phrase Score</strong></td><td><strong>${formatPct(advStats.ngrams?.repeatedPhraseScore)}</strong></td><td>${getIndicator(advStats.ngrams?.repeatedPhraseScore, [0.3, 0.1])}</td></tr>
        <tr><td>Repeated Phrases (4+ words)</td><td>${advStats.ngrams?.repeatedPhraseCount || 0} found</td><td>${advStats.ngrams?.repeatedPhraseCount > 2 ? '🔴' : '⚪'}</td></tr>
    </table>
    ${advStats.ngrams?.repeatedPhrases?.length > 0 ? `
    <p style="font-size:8pt;margin-top:5px;"><strong>Top repeated phrases:</strong></p>
    <ul style="font-size:8pt;margin:3px 0 10px 20px;">
        ${advStats.ngrams.repeatedPhrases.slice(0, 5).map(p => `<li>"${p.phrase}" (${p.count}x)</li>`).join('')}
    </ul>` : ''}
    
    <!-- Word Analysis -->
    <h3>🔤 Word Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Avg Word Length</td><td>${formatNum(advStats.words?.avgLength, 2)} chars</td></tr>
        <tr><td>Word Entropy</td><td>${formatNum(advStats.words?.entropy, 2)} bits</td></tr>
        <tr><td>Function Word Ratio</td><td>${formatPct(advStats.functionWords?.ratio)}</td></tr>
        <tr><td>Content Word Ratio</td><td>${formatPct(advStats.functionWords?.contentWordRatio)}</td></tr>
    </table>
    
    <!-- Word Pattern Analysis -->
    <h3>🏷️ Word Pattern Analysis</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>First-Person Pronoun Ratio</td><td>${formatPct(advStats.wordPatterns?.firstPersonRatio)}</td><td>${advStats.wordPatterns?.firstPersonRatio < 0.01 ? '🔴' : (advStats.wordPatterns?.firstPersonRatio > 0.03 ? '🟢' : '🟡')}</td></tr>
        <tr><td>Hedging Word Ratio</td><td>${formatPct(advStats.wordPatterns?.hedgingRatio)}</td><td>${advStats.wordPatterns?.hedgingRatio > 0.02 ? '🔴' : '⚪'}</td></tr>
        <tr><td>Sentence Starter Diversity</td><td>${formatPct(advStats.wordPatterns?.starterDiversity)}</td><td>${advStats.wordPatterns?.starterDiversity < 0.4 ? '🔴' : (advStats.wordPatterns?.starterDiversity > 0.7 ? '🟢' : '🟡')}</td></tr>
        <tr><td>Common AI Starters Ratio</td><td>${formatPct(advStats.wordPatterns?.aiStarterRatio)}</td><td>${advStats.wordPatterns?.aiStarterRatio > 0.5 ? '🔴' : '⚪'}</td></tr>
        <tr><td>Verb-like Words</td><td>${formatPct(advStats.wordPatterns?.verbRatio)}</td><td>—</td></tr>
        <tr><td>Adjective-like Words</td><td>${formatPct(advStats.wordPatterns?.adjectiveRatio)}</td><td>—</td></tr>
        <tr><td>Adverb-like Words</td><td>${formatPct(advStats.wordPatterns?.adverbRatio)}</td><td>—</td></tr>
        <tr><td>Content Density</td><td>${formatPct(advStats.wordPatterns?.contentDensity)}</td><td>—</td></tr>
    </table>
    
    <!-- Advanced Statistical Tests -->
    <h3>🔬 Advanced Statistical Tests</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Periodicity Score</td><td>${formatPct(advStats.autocorrelation?.periodicityScore)}</td><td>${getIndicator(advStats.autocorrelation?.periodicityScore, [0.6, 0.3])}</td></tr>
        <tr><td>N-gram Predictability</td><td>${formatPct(advStats.perplexity?.predictability)}</td><td>${getIndicator(advStats.perplexity?.predictability, [0.6, 0.4])}</td></tr>
        <tr><td>Perplexity (approx)</td><td>${formatNum(advStats.perplexity?.perplexity, 1)}</td><td>—</td></tr>
        <tr><td>Randomness Score</td><td>${formatPct(advStats.runsTest?.randomnessScore)}</td><td>${getIndicator(advStats.runsTest?.randomnessScore, [0.4, 0.6], true)}</td></tr>
        <tr><td>χ² Uniformity</td><td>${formatPct(advStats.chiSquared?.uniformityScore)}</td><td>${getIndicator(advStats.chiSquared?.uniformityScore, [0.7, 0.4])}</td></tr>
        <tr><td>Variance Stability</td><td>${formatPct(advStats.varianceStability)}</td><td>${getIndicator(advStats.varianceStability, [0.7, 0.5])}</td></tr>
        <tr><td>Mahalanobis Distance</td><td>${formatNum(advStats.mahalanobisDistance, 2)}σ</td><td>${getIndicator(advStats.mahalanobisDistance, [2.0, 1.0])}</td></tr>
    </table>
    
    <!-- Human Likelihood -->
    <h3>📊 Human Likelihood (Bell Curve Analysis)</h3>
    <p style="font-size:8pt;color:#666;margin-bottom:5px;">Measures how close features are to typical human writing. Values near 1.0 = normal human range.</p>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr style="background:#f5f5f5;"><td><strong>Overall Human Likelihood</strong></td><td><strong>${formatPct(advStats.overallHumanLikelihood)}</strong></td><td>${getIndicator(advStats.overallHumanLikelihood, [0.4, 0.6], true)}</td></tr>
        <tr><td>Sentence Length Variance</td><td>${formatPct(advStats.humanLikelihood?.sentenceLengthCV)}</td><td>${getIndicator(advStats.humanLikelihood?.sentenceLengthCV, [0.4, 0.7], true)}</td></tr>
        <tr><td>Unique Word Distribution</td><td>${formatPct(advStats.humanLikelihood?.hapaxRatio)}</td><td>${getIndicator(advStats.humanLikelihood?.hapaxRatio, [0.4, 0.7], true)}</td></tr>
        <tr><td>Word Usage Burstiness</td><td>${formatPct(advStats.humanLikelihood?.burstiness)}</td><td>${getIndicator(advStats.humanLikelihood?.burstiness, [0.4, 0.7], true)}</td></tr>
        <tr><td>Zipf's Law Compliance</td><td>${formatPct(advStats.humanLikelihood?.zipfSlope)}</td><td>${getIndicator(advStats.humanLikelihood?.zipfSlope, [0.4, 0.7], true)}</td></tr>
        <tr><td>Vocabulary Richness</td><td>${formatPct(advStats.humanLikelihood?.ttr)}</td><td>${getIndicator(advStats.humanLikelihood?.ttr, [0.4, 0.7], true)}</td></tr>
        <tr><td>Variance Naturalness</td><td>${formatPct(advStats.varianceNaturalness)}</td><td>${getIndicator(advStats.varianceNaturalness, [0.4, 0.7], true)}</td></tr>
        <tr><td>Extreme Variance Warning</td><td>${formatPct(advStats.extremeVarianceIndicator)}</td><td>${getIndicator(advStats.extremeVarianceIndicator, [0.6, 0.4])}</td></tr>
    </table>
    
    <!-- AI Signature Metrics -->
    <h3>🤖 AI Signature Metrics</h3>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Indicator</th></tr>
        <tr><td>Hedging Density</td><td>${formatPct(advStats.aiSignatures?.hedgingDensity)}</td><td>${getIndicator(advStats.aiSignatures?.hedgingDensity, [0.02, 0.01])}</td></tr>
        <tr><td>Discourse Marker Density</td><td>${formatNum(advStats.aiSignatures?.discourseMarkerDensity, 2)}/sentence</td><td>${getIndicator(advStats.aiSignatures?.discourseMarkerDensity, [0.4, 0.2])}</td></tr>
        <tr><td>Unicode Anomaly Density</td><td>${formatNum(advStats.aiSignatures?.unicodeAnomalyDensity, 2)}/1000 chars</td><td>${getIndicator(advStats.aiSignatures?.unicodeAnomalyDensity, [1, 0.3])}</td></tr>
        <tr><td>Decorative Dividers</td><td>${advStats.aiSignatures?.decorativeDividerCount || 0}</td><td>${getIndicator(advStats.aiSignatures?.decorativeDividerCount, [1, 0])}</td></tr>
        <tr><td>Contraction Rate</td><td>${formatNum(advStats.aiSignatures?.contractionRate, 2)}/sentence</td><td>${getIndicator(advStats.aiSignatures?.contractionRate, [0.3, 0.5], true)}</td></tr>
        <tr><td>Sentence Starter Variety</td><td>${formatPct(advStats.aiSignatures?.sentenceStarterVariety)}</td><td>${getIndicator(advStats.aiSignatures?.sentenceStarterVariety, [0.4, 0.6], true)}</td></tr>
        <tr><td>Passive Voice Rate</td><td>${formatNum(advStats.aiSignatures?.passiveVoiceRate, 2)}/sentence</td><td>—</td></tr>
    </table>
    
    <!-- Humanizer Detection -->
    <h3 style="${advStats.humanizerSignals?.isLikelyHumanized ? 'color:#7c3aed;' : ''}">🔄 Humanizer Detection</h3>
    <p style="font-size:8pt;color:#666;margin-bottom:5px;">Detects AI text that has been post-processed to evade detection.</p>
    <table>
        <tr><th>Metric</th><th>Value</th><th>Status</th></tr>
        <tr style="${advStats.humanizerSignals?.isLikelyHumanized ? 'background:#faf5ff;' : ''}"><td><strong>Humanizer Probability</strong></td><td><strong>${formatPct(advStats.humanizerSignals?.humanizerProbability)}</strong></td><td>${advStats.humanizerSignals?.isLikelyHumanized ? '🔴' : '⚪'}</td></tr>
        <tr><td>Variance Stability (2nd order)</td><td>${advStats.humanizerSignals?.stableVarianceFlag ? 'Suspicious' : 'Normal'}</td><td>${advStats.humanizerSignals?.stableVarianceFlag ? '🔴' : '⚪'}</td></tr>
        <tr><td>Autocorrelation Pattern</td><td>${advStats.humanizerSignals?.flatAutocorrelationFlag ? 'Random noise' : 'Natural'}</td><td>${advStats.humanizerSignals?.flatAutocorrelationFlag ? '🔴' : '⚪'}</td></tr>
        <tr><td>Feature Correlations</td><td>${advStats.humanizerSignals?.brokenCorrelationFlag ? 'Broken' : 'Intact'}</td><td>${advStats.humanizerSignals?.brokenCorrelationFlag ? '🔴' : '⚪'}</td></tr>
        <tr><td>Sophistication Consistency</td><td>${advStats.humanizerSignals?.synonymSubstitutionFlag ? 'Word-level chaos' : 'Consistent'}</td><td>${advStats.humanizerSignals?.synonymSubstitutionFlag ? '🔴' : '⚪'}</td></tr>
        <tr><td>Contraction Pattern</td><td>${advStats.humanizerSignals?.artificialContractionFlag ? 'Artificial' : 'Natural'}</td><td>${advStats.humanizerSignals?.artificialContractionFlag ? '🔴' : '⚪'}</td></tr>
        <tr><td>Warning Flags</td><td>${advStats.humanizerSignals?.flagCount || 0} / 5</td><td>—</td></tr>
    </table>`;
        }

        // Add Word Frequency Distribution Chart (Zipf)
        html += this.generateZipfChartHtml(analysisResult);

        html += `
    <h2>Category Weight Overview</h2>
    <p>Each detection category contributes to the overall AI probability with ML-derived weights (Sunrise Model v3.0):</p>
    <div class="chart bar-chart">`;

        // Add weight chart - show by AI probability, sorted
        const sortedCats = [...report.categoryAnalyses].sort((a, b) => b.aiProbability - a.aiProbability);
        
        for (const cat of sortedCats) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05 };
            const weightPct = Math.round(weightInfo.weight * 100);
            const barColor = cat.aiProbability >= 60 ? '#ef4444' : (cat.aiProbability >= 40 ? '#f59e0b' : '#10b981');
            html += `
        <div class="bar-row">
            <div class="bar-label">${cat.name} (${weightPct}%)</div>
            <div class="bar-track"><div class="bar-fill-ai" style="width: ${Math.max(2, cat.aiProbability)}%; background: ${barColor};"></div></div>
            <div class="bar-value">${cat.aiProbability}%</div>
        </div>`;
        }

        html += `
    </div>

    <h2>Detailed Category Analysis</h2>`;

        // All categories with full details
        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05, description: 'Analyzes various text characteristics.' };
            const weightPct = Math.round(weightInfo.weight * 100);
            const scoreClass = cat.aiProbability >= 60 ? 'score-high' : (cat.aiProbability >= 40 ? 'score-moderate' : 'score-low');
            const fillColor = cat.aiProbability >= 60 ? '#ef4444' : (cat.aiProbability >= 40 ? '#f59e0b' : '#10b981');
            
            html += `
    <div class="category-card">
        <div class="category-header">
            <span class="category-name">${cat.category}. ${cat.name}</span>
            <span class="category-score ${scoreClass}">${cat.aiProbability}% AI</span>
        </div>
        <div class="category-bar">
            <div class="category-fill" style="width: ${cat.aiProbability}%; background: ${fillColor};"></div>
        </div>
        <p><strong>Confidence:</strong> ${cat.confidence}%</p>
        <p>${cat.description}</p>`;

            if (cat.findings && cat.findings.length > 0) {
                html += `<h4>Key Findings:</h4>`;
                for (const f of cat.findings.slice(0, 5)) {
                    const findingText = f.text || f.value || f.label || 'Unknown finding';
                    const indicator = f.indicator || 'neutral';
                    html += `<div class="finding ${indicator}">${findingText}</div>`;
                }
            }

            html += `
        <div class="weight-info">
            <strong>Category Weight:</strong> ${weightPct}% of total score | ${weightInfo.description}
        </div>
    </div>`;
        }

        // Add verbose signal summary
        html += this.generateSignalSummaryHtml(analysisResult, verboseEvidence);

        // Add the original analyzed text
        html += `
    <h2 class="page-break">📄 Analyzed Text</h2>
    <div style="background:#fafafa;border:1px solid #e5e5e5;border-radius:6px;padding:12px;margin:10px 0;max-height:none;">
        <p style="font-size:8pt;color:#888;margin-bottom:8px;text-transform:uppercase;letter-spacing:0.5px;">Original Text (${report.statistics.wordCount.toLocaleString()} words)</p>
        <div style="font-size:9pt;line-height:1.6;color:#333;white-space:pre-wrap;word-wrap:break-word;font-family:'Georgia',serif;">${this.escapeHtml(report.analyzedText || '')}</div>
    </div>`;

        html += `
    <div class="methodology">
        <h3>${report.methodology.title}</h3>
        <p>${report.methodology.content.replace(/\n/g, '<br>')}</p>
    </div>

    <div class="disclaimer">
        <h3>${report.disclaimer.title}</h3>
        <p>${report.disclaimer.content.replace(/\n/g, '<br>')}</p>
    </div>

    <h2>Category Weight Reference Table</h2>
    <table>
        <tr><th>Category</th><th>Weight</th><th>What It Measures</th></tr>`;
        
        for (let i = 1; i <= 14; i++) {
            const info = this.categoryWeightInfo[i];
            if (info) {
                html += `
        <tr>
            <td>${i}. ${info.name}</td>
            <td>${Math.round(info.weight * 100)}%</td>
            <td>${info.description}</td>
        </tr>`;
            }
        }

        html += `
    </table>

    <div style="margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%); border-radius: 8px; border: 1px solid #e5e5e5; text-align: center;">
        <div style="font-size: 14pt; font-weight: bold; color: #333; margin-bottom: 8px;">◈ VERITAS</div>
        <p style="color: #666; font-size: 9pt; margin: 4px 0;">AI Text Detection Analysis System</p>
        <p style="color: #888; font-size: 8pt; margin: 4px 0;">Powered by Sunrise Model v3.0 | 98.08% Accuracy | 29,976 Training Samples</p>
        <div style="margin-top: 12px; padding-top: 12px; border-top: 1px dashed #ddd;">
            <p style="font-size: 7pt; color: #aaa; line-height: 1.6;">
                ⚠️ <strong>Important:</strong> No single metric is definitive • Context and domain matter significantly • Confidence varies with text length and complexity • AI detection methods continuously evolve • This report is for informational purposes only and should not be the sole basis for any decision.
            </p>
        </div>
    </div>
</body>
</html>`;

        return html;
    },

    /**
     * Build verbose evidence from analysis results with specific statistics
     */
    buildVerboseEvidence(report, analysisResult) {
        const evidence = {
            aiSignals: [],
            humanSignals: [],
            statistics: {}
        };

        const advStats = analysisResult.advancedStats || {};
        const stats = analysisResult.stats || {};

        // Sentence structure evidence
        if (advStats.sentences) {
            const cv = advStats.sentences.coefficientOfVariation;
            evidence.statistics.sentenceCV = cv;
            if (cv < 0.35) {
                evidence.aiSignals.push({
                    label: 'Low Sentence Variance',
                    value: `CV: ${(cv * 100).toFixed(1)}% (human avg: 45-60%)`,
                    severity: cv < 0.25 ? 'high' : 'medium'
                });
            } else if (cv > 0.5) {
                evidence.humanSignals.push({
                    label: 'Natural Sentence Variance',
                    value: `CV: ${(cv * 100).toFixed(1)}%`
                });
            }
        }

        // Burstiness evidence
        if (advStats.burstiness) {
            const burst = advStats.burstiness.sentenceLength;
            evidence.statistics.burstiness = burst;
            if (burst < 0.15) {
                evidence.aiSignals.push({
                    label: 'Low Burstiness',
                    value: `${(burst * 100).toFixed(1)}% (human avg: 25-40%)`,
                    severity: burst < 0.1 ? 'high' : 'medium'
                });
            } else if (burst > 0.25) {
                evidence.humanSignals.push({
                    label: 'Natural Burstiness',
                    value: `${(burst * 100).toFixed(1)}%`
                });
            }
        }

        // Vocabulary diversity
        if (advStats.vocabulary) {
            const hapax = advStats.vocabulary.hapaxLegomenaRatio;
            evidence.statistics.hapaxRatio = hapax;
            if (hapax < 0.35) {
                evidence.aiSignals.push({
                    label: 'Low Hapax Ratio',
                    value: `${(hapax * 100).toFixed(1)}% unique words (human avg: 40-55%)`,
                    severity: hapax < 0.3 ? 'high' : 'medium'
                });
            }
        }

        // Personal pronouns
        if (advStats.wordPatterns) {
            const fp = advStats.wordPatterns.firstPersonRatio;
            evidence.statistics.firstPersonRatio = fp;
            if (fp < 0.005) {
                evidence.aiSignals.push({
                    label: 'No Personal Pronouns',
                    value: `${(fp * 100).toFixed(2)}% first-person usage`,
                    severity: 'medium'
                });
            } else if (fp > 0.02) {
                evidence.humanSignals.push({
                    label: 'Personal Experience Markers',
                    value: `${(fp * 100).toFixed(2)}% first-person pronouns`
                });
            }

            const hedging = advStats.wordPatterns.hedgingRatio;
            if (hedging > 0.02) {
                evidence.aiSignals.push({
                    label: 'Excessive Hedging',
                    value: `${(hedging * 100).toFixed(2)}% hedging words (AI typically >2%)`,
                    severity: hedging > 0.03 ? 'high' : 'medium'
                });
            }
        }

        // Uniformity metrics
        if (advStats.burstiness?.overallUniformity) {
            const uniformity = advStats.burstiness.overallUniformity;
            evidence.statistics.overallUniformity = uniformity;
            if (uniformity > 0.7) {
                evidence.aiSignals.push({
                    label: 'High Uniformity Score',
                    value: `${(uniformity * 100).toFixed(1)}% (human avg: 40-60%)`,
                    severity: uniformity > 0.8 ? 'high' : 'medium'
                });
            }
        }

        // AI signatures
        if (advStats.aiSignatures) {
            if (advStats.aiSignatures.contractionRate < 0.05) {
                evidence.aiSignals.push({
                    label: 'No Contractions',
                    value: `${advStats.aiSignatures.contractionRate.toFixed(2)} per sentence (human avg: 0.2-0.5)`,
                    severity: 'low'
                });
            }
            if (advStats.aiSignatures.sentenceStarterVariety < 0.4) {
                evidence.aiSignals.push({
                    label: 'Repetitive Sentence Starters',
                    value: `${(advStats.aiSignatures.sentenceStarterVariety * 100).toFixed(1)}% variety`,
                    severity: 'medium'
                });
            }
        }

        // Category-specific evidence
        for (const cat of (analysisResult.categoryResults || [])) {
            if (cat.aiProbability > 0.65 && cat.confidence > 0.5) {
                const weightInfo = this.categoryWeightInfo[cat.category];
                if (weightInfo && weightInfo.weight > 0.08) {
                    evidence.aiSignals.push({
                        label: cat.name,
                        value: `${Math.round(cat.aiProbability * 100)}% AI probability, ${Math.round(cat.confidence * 100)}% confidence`,
                        severity: cat.aiProbability > 0.8 ? 'high' : 'medium'
                    });
                }
            } else if (cat.aiProbability < 0.35 && cat.confidence > 0.5) {
                evidence.humanSignals.push({
                    label: cat.name,
                    value: `${Math.round(cat.aiProbability * 100)}% AI probability`
                });
            }
        }

        return evidence;
    },

    /**
     * Generate Zipf distribution chart HTML
     */
    generateZipfChartHtml(analysisResult) {
        const tokens = analysisResult.tokens || [];
        if (!tokens || tokens.length < 50) {
            return '';
        }

        // Build frequency distribution
        const freq = {};
        for (const token of tokens) {
            const word = token.toLowerCase();
            freq[word] = (freq[word] || 0) + 1;
        }
        
        const sortedFreqs = Object.entries(freq)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 30);

        if (sortedFreqs.length < 10) {
            return '';
        }

        const maxFreq = sortedFreqs[0][1];
        const chartWidth = 400;
        const chartHeight = 180;
        const barWidth = Math.floor(chartWidth / sortedFreqs.length) - 2;

        let html = `
    <h3>📊 Word Frequency Distribution (Zipf Chart)</h3>
    <p style="font-size:8pt;color:#666;margin-bottom:10px;">Shows top 30 words by frequency. Natural text follows Zipf's law (rank × frequency ≈ constant).</p>
    <div style="background:#f9fafb;border:1px solid #e5e5e5;border-radius:4px;padding:10px;margin:10px 0;">
        <svg viewBox="0 0 ${chartWidth} ${chartHeight}" style="width:100%;max-width:${chartWidth}px;height:auto;">
            <!-- Title -->
            <text x="${chartWidth/2}" y="15" text-anchor="middle" style="font-size:10px;fill:#333;">Word Frequency (Top 30)</text>
            
            <!-- Y-axis label -->
            <text x="5" y="${chartHeight/2}" text-anchor="middle" transform="rotate(-90, 12, ${chartHeight/2})" style="font-size:8px;fill:#666;">Frequency</text>
            
            <!-- X-axis label -->
            <text x="${chartWidth/2}" y="${chartHeight - 5}" text-anchor="middle" style="font-size:8px;fill:#666;">Word Rank</text>
            
            <!-- Bars -->`;

        const plotTop = 25;
        const plotBottom = chartHeight - 25;
        const plotHeight = plotBottom - plotTop;
        const logMax = Math.log(maxFreq + 1);

        sortedFreqs.forEach(([word, count], i) => {
            const x = 30 + i * (barWidth + 2);
            const barHeight = (Math.log(count + 1) / logMax) * plotHeight;
            const y = plotBottom - barHeight;
            
            // Color gradient from high to low frequency
            const hue = 200 + (i / sortedFreqs.length) * 80; // Blue to purple
            const color = `hsl(${hue}, 60%, 50%)`;
            
            html += `
            <rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" fill="${color}" rx="1">
                <title>${word}: ${count}</title>
            </rect>`;
        });

        // Expected Zipf line (reference)
        let pathData = '';
        sortedFreqs.forEach(([word, count], i) => {
            const expectedFreq = maxFreq / (i + 1);
            const x = 30 + i * (barWidth + 2) + barWidth / 2;
            const y = plotBottom - (Math.log(expectedFreq + 1) / logMax) * plotHeight;
            pathData += `${i === 0 ? 'M' : 'L'} ${x},${y}`;
        });
        
        html += `
            <!-- Expected Zipf curve -->
            <path d="${pathData}" fill="none" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2" opacity="0.7"/>
            
            <!-- Legend -->
            <rect x="${chartWidth - 100}" y="20" width="10" height="10" fill="hsl(220, 60%, 50%)" rx="1"/>
            <text x="${chartWidth - 85}" y="28" style="font-size:7px;fill:#666;">Actual</text>
            <line x1="${chartWidth - 100}" y1="38" x2="${chartWidth - 90}" y2="38" stroke="#ef4444" stroke-width="1.5" stroke-dasharray="4,2"/>
            <text x="${chartWidth - 85}" y="41" style="font-size:7px;fill:#666;">Expected (Zipf)</text>
        </svg>
    </div>`;

        return html;
    },

    /**
     * Generate signal summary HTML section
     */
    generateSignalSummaryHtml(analysisResult, evidence) {
        if (!evidence || (evidence.aiSignals.length === 0 && evidence.humanSignals.length === 0)) {
            return '';
        }

        let html = `
    <div class="signal-summary">
        <h3>Detection Signal Summary</h3>
        <p style="font-size:9pt;color:#666;">Specific statistical evidence supporting the classification:</p>
        <div class="signal-grid">`;

        // AI signals column
        html += `<div>
            <h4 style="color:#b91c1c;font-size:10pt;margin-bottom:6px;">🔴 AI Indicators (${evidence.aiSignals.length})</h4>`;
        
        for (const signal of evidence.aiSignals.slice(0, 6)) {
            const severityColor = signal.severity === 'high' ? '#991b1b' : (signal.severity === 'medium' ? '#b91c1c' : '#dc2626');
            html += `
            <div class="signal-item ai-signal">
                <div class="signal-label">${signal.label}</div>
                <div class="signal-value" style="color:${severityColor}">${signal.value}</div>
            </div>`;
        }
        html += `</div>`;

        // Human signals column
        html += `<div>
            <h4 style="color:#047857;font-size:10pt;margin-bottom:6px;">🟢 Human Indicators (${evidence.humanSignals.length})</h4>`;
        
        for (const signal of evidence.humanSignals.slice(0, 6)) {
            html += `
            <div class="signal-item human-signal">
                <div class="signal-label">${signal.label}</div>
                <div class="signal-value" style="color:#047857">${signal.value}</div>
            </div>`;
        }
        
        if (evidence.humanSignals.length === 0) {
            html += `<p style="font-size:9pt;color:#888;padding:6px;">No significant human indicators found</p>`;
        }
        html += `</div>`;

        html += `
        </div>
    </div>`;

        return html;
    },

    /**
     * Generate structured report content
     */
    generateReportContent(result, originalText) {
        // Get model info if available
        const modelConfig = typeof VERITAS_SUNRISE_CONFIG !== 'undefined' ? VERITAS_SUNRISE_CONFIG : null;
        
        const report = {
            title: 'VERITAS AI Detection Analysis Report',
            generatedAt: new Date().toISOString(),
            modelInfo: modelConfig ? {
                name: modelConfig.modelName,
                version: modelConfig.version,
                accuracy: modelConfig.trainingStats?.testAccuracy || 0.98,
                f1Score: modelConfig.trainingStats?.testF1 || 0.98,
                trainingSamples: modelConfig.trainingStats?.totalSamples || 29976
            } : {
                name: 'Sunrise',
                version: '3.0.0',
                accuracy: 0.9808,
                f1Score: 0.9809,
                trainingSamples: 29976
            },
            summary: this.generateExecutiveSummary(result),
            verdict: this.generateVerdictSection(result),
            statistics: this.generateStatisticsSection(result),
            categoryAnalyses: this.generateCategoryAnalyses(result),
            keyFindings: this.generateKeyFindings(result),
            detailedBreakdown: this.generateDetailedBreakdown(result),
            methodology: this.generateMethodologySection(),
            disclaimer: this.generateDisclaimer(),
            analyzedText: originalText
        };

        return report;
    },

    /**
     * Generate executive summary
     */
    generateExecutiveSummary(result) {
        const probability = Math.round(result.aiProbability * 100);
        const band = VarianceUtils.toProbabilityBand(result.aiProbability);
        const confidence = Math.round(result.confidence * 100);
        
        let summaryText = '';
        
        if (probability < 30) {
            summaryText = `This analysis indicates a LOW probability (${probability}%) of AI-generated content. ` +
                `The text exhibits characteristics typically associated with human-authored writing, including ` +
                `natural variance in tone, sentence structure, and vocabulary usage. `;
        } else if (probability < 50) {
            summaryText = `This analysis indicates a MODERATE-LOW probability (${probability}%) of AI-generated content. ` +
                `While some patterns consistent with AI writing were detected, the majority of indicators suggest ` +
                `human authorship. `;
        } else if (probability < 60) {
            summaryText = `This analysis is INCONCLUSIVE (${probability}% probability). ` +
                `The text shows a mix of both human-like and AI-like characteristics. This could indicate ` +
                `human-edited AI content, AI-assisted writing, or naturally formal human prose. `;
        } else if (probability < 75) {
            summaryText = `This analysis indicates a MODERATE-HIGH probability (${probability}%) of AI-generated content. ` +
                `Several patterns characteristic of AI writing were detected, though some human-like elements are present. `;
        } else {
            summaryText = `This analysis indicates a HIGH probability (${probability}%) of AI-generated content. ` +
                `The text exhibits multiple strong indicators of AI generation, including unusual uniformity ` +
                `in structure, tone, and vocabulary patterns. `;
        }

        summaryText += `Analysis confidence: ${confidence}%.`;

        return {
            text: summaryText,
            probability,
            band: band.label,
            confidence
        };
    },

    /**
     * Generate verdict section
     */
    generateVerdictSection(result) {
        const fpRisk = VarianceUtils.falsePositiveRisk(result.stats, result.aiProbability);
        const ci = VarianceUtils.confidenceInterval(result.aiProbability, result.stats.sentences);

        return {
            label: result.verdict.label,
            description: result.verdict.description,
            probabilityBand: `${Math.round(ci.lower * 100)}% - ${Math.round(ci.upper * 100)}%`,
            confidenceInterval: ci,
            falsePositiveRisk: fpRisk
        };
    },

    /**
     * Generate statistics section
     */
    generateStatisticsSection(result) {
        return {
            wordCount: result.stats.words,
            sentenceCount: result.stats.sentences,
            paragraphCount: result.stats.paragraphs,
            characterCount: result.stats.characters,
            avgWordsPerSentence: result.stats.avgWordsPerSentence,
            analysisTime: result.analysisTime
        };
    },

    /**
     * Generate category analyses with template-based descriptions
     */
    generateCategoryAnalyses(result) {
        const analyses = [];

        for (const category of result.categoryResults) {
            const percentage = Math.round(category.aiProbability * 100);
            const detected = percentage > 50;
            const strength = percentage > 75 ? 'strong' : (percentage > 60 ? 'moderate' : 'weak');
            
            const analysis = {
                name: category.name,
                category: category.category,
                aiProbability: percentage,
                confidence: Math.round(category.confidence * 100),
                detected,
                strength,
                description: this.getCategoryDescription(category, detected, strength),
                findings: category.findings || []
            };

            analyses.push(analysis);
        }

        return analyses;
    },

    /**
     * Get template-based description for a category
     */
    getCategoryDescription(category, detected, strength) {
        const percentage = Math.round(category.aiProbability * 100);
        const templates = this.getDescriptionTemplates();
        
        const template = templates[category.name] || templates.default;
        
        if (detected) {
            return template.detected
                .replace('{percentage}', percentage)
                .replace('{strength}', strength)
                .replace('{category}', category.name);
        } else {
            return template.notDetected
                .replace('{percentage}', percentage)
                .replace('{category}', category.name);
        }
    },

    /**
     * Description templates for each category
     */
    getDescriptionTemplates() {
        return {
            'Grammar & Error Patterns': {
                detected: `Grammar analysis indicates a {strength} AI signal ({percentage}% probability). ` +
                    `The text shows unusually consistent grammatical patterns with minimal error variation, ` +
                    `which is characteristic of AI-generated content that lacks natural human inconsistencies.`,
                notDetected: `Grammar analysis shows natural human-like patterns ({percentage}% AI probability). ` +
                    `The text contains typical variation in grammatical structures and occasional inconsistencies ` +
                    `consistent with human writing.`
            },
            'Sentence Structure & Syntax': {
                detected: `Syntactic analysis reveals {strength} AI indicators ({percentage}% probability). ` +
                    `Sentence length variance is lower than typical human writing, suggesting systematic generation. ` +
                    `The uniformity of sentence structures indicates algorithmic production.`,
                notDetected: `Sentence structure analysis indicates human-like patterns ({percentage}% AI probability). ` +
                    `The text shows natural "burstiness" in sentence lengths, with appropriate variation ` +
                    `between short and long sentences typical of organic writing.`
            },
            'Lexical Choice & Vocabulary': {
                detected: `Vocabulary analysis shows {strength} AI patterns ({percentage}% probability). ` +
                    `Lexical diversity metrics indicate repetitive word choices or overuse of certain vocabulary ` +
                    `patterns commonly associated with language model outputs.`,
                notDetected: `Lexical analysis indicates human-typical vocabulary usage ({percentage}% AI probability). ` +
                    `Word choice shows natural variation with appropriate lexical diversity scores.`
            },
            'Tone Stability': {
                detected: `Tone analysis reveals {strength} AI characteristics ({percentage}% probability). ` +
                    `The emotional and stylistic tone remains unusually stable throughout the document, ` +
                    `lacking the natural drift and variation typical of human writing.`,
                notDetected: `Tone analysis indicates natural human patterns ({percentage}% AI probability). ` +
                    `The text shows appropriate variation in emotional valence and stylistic register.`
            },
            'Repetition Patterns': {
                detected: `Repetition analysis shows {strength} AI indicators ({percentage}% probability). ` +
                    `Repeated phrases and concepts appear at regular intervals, suggesting template-based generation. ` +
                    `This uniformity in repetition distribution is characteristic of AI outputs.`,
                notDetected: `Repetition patterns appear human-like ({percentage}% AI probability). ` +
                    `Any repetition shows natural clustering rather than uniform distribution.`
            },
            'Metadata & Formatting': {
                detected: `Metadata analysis reveals {strength} anomalies ({percentage}% probability). ` +
                    `The document contains formatting inconsistencies, unusual Unicode characters, or ` +
                    `hidden elements that may indicate copy-paste from AI sources.`,
                notDetected: `Metadata and formatting appear consistent ({percentage}% AI probability). ` +
                    `No significant formatting anomalies or suspicious hidden characters detected.`
            },
            'Statistical Language Model Indicators': {
                detected: `Statistical analysis shows {strength} AI signatures ({percentage}% probability). ` +
                    `Perplexity patterns and token predictability metrics are consistent with language model outputs. ` +
                    `The text shows lower-than-expected entropy in certain segments.`,
                notDetected: `Statistical indicators suggest human authorship ({percentage}% AI probability). ` +
                    `Perplexity and predictability metrics fall within expected human writing ranges.`
            },
            default: {
                detected: `Analysis of {category} shows {strength} AI indicators ({percentage}% probability). ` +
                    `Patterns detected are consistent with AI-generated content.`,
                notDetected: `Analysis of {category} indicates human-like patterns ({percentage}% AI probability). ` +
                    `No significant AI indicators detected in this category.`
            }
        };
    },

    /**
     * Generate key findings section
     */
    generateKeyFindings(result) {
        const findings = result.findings || [];
        
        // Sort by severity and indicator
        const sorted = [...findings].sort((a, b) => {
            const severityOrder = { high: 0, medium: 1, low: 2 };
            const indicatorOrder = { ai: 0, mixed: 1, human: 2, neutral: 3 };
            
            const severityDiff = (severityOrder[a.severity] || 2) - (severityOrder[b.severity] || 2);
            if (severityDiff !== 0) return severityDiff;
            
            return (indicatorOrder[a.indicator] || 3) - (indicatorOrder[b.indicator] || 3);
        });

        // Group by indicator
        const aiFindings = sorted.filter(f => f.indicator === 'ai');
        const humanFindings = sorted.filter(f => f.indicator === 'human');
        const mixedFindings = sorted.filter(f => f.indicator === 'mixed' || f.indicator === 'neutral');

        return {
            aiIndicators: aiFindings.slice(0, 10),
            humanIndicators: humanFindings.slice(0, 5),
            uncertainIndicators: mixedFindings.slice(0, 5),
            totalFindings: findings.length
        };
    },

    /**
     * Generate detailed breakdown
     */
    generateDetailedBreakdown(result) {
        const breakdown = [];

        // Sentence-level analysis summary
        if (result.sentenceScores && result.sentenceScores.length > 0) {
            const aiSentences = result.sentenceScores.filter(s => s.classification === 'ai').length;
            const humanSentences = result.sentenceScores.filter(s => s.classification === 'human').length;
            const mixedSentences = result.sentenceScores.filter(s => s.classification === 'mixed').length;
            
            breakdown.push({
                title: 'Sentence-Level Classification',
                content: `Of ${result.sentenceScores.length} sentences analyzed:\n` +
                    `• ${aiSentences} sentences (${Math.round(aiSentences/result.sentenceScores.length*100)}%) classified as likely AI-generated\n` +
                    `• ${humanSentences} sentences (${Math.round(humanSentences/result.sentenceScores.length*100)}%) classified as likely human-written\n` +
                    `• ${mixedSentences} sentences (${Math.round(mixedSentences/result.sentenceScores.length*100)}%) classified as uncertain/mixed`
            });
        }

        // Feature contribution breakdown
        const contributions = result.categoryResults
            .filter(c => c.confidence > 0.3)
            .map(c => ({
                name: c.name,
                contribution: c.aiProbability * c.confidence,
                weight: c.confidence
            }))
            .sort((a, b) => b.contribution - a.contribution);

        if (contributions.length > 0) {
            breakdown.push({
                title: 'Feature Contribution to AI Probability',
                items: contributions.map(c => ({
                    name: c.name,
                    value: `${Math.round(c.contribution * 100)}% weighted contribution`
                }))
            });
        }

        return breakdown;
    },

    /**
     * Generate methodology section
     */
    generateMethodologySection() {
        return {
            title: 'Analysis Methodology',
            content: `This analysis employs a variance-based detection approach that measures deviations from expected human writing patterns. ` +
                `Rather than flagging specific features as "AI-like" or "human-like," the system analyzes the distribution, uniformity, ` +
                `and predictability of linguistic features across the document.\n\n` +
                `Key analysis dimensions include:\n` +
                `• Grammar & Error Pattern Variance\n` +
                `• Sentence Structure Uniformity (Burstiness)\n` +
                `• Lexical Diversity & Vocabulary Distribution\n` +
                `• Tone Stability & Emotional Variance\n` +
                `• Repetition Clustering & Distribution\n` +
                `• Statistical Language Model Indicators\n` +
                `• Authorship Consistency Metrics\n` +
                `• Meta-Patterns Unique to AI Generation\n\n` +
                `Scores are normalized using z-scores relative to expected document characteristics ` +
                `and aggregated with weighted category contributions.`
        };
    },

    /**
     * Generate disclaimer
     */
    generateDisclaimer() {
        return {
            title: 'Important Disclaimer',
            content: `This analysis is provided for informational purposes only and should not be considered definitive proof ` +
                `of AI generation or human authorship.\n\n` +
                `• No AI detection system is 100% accurate\n` +
                `• False positives and false negatives can occur\n` +
                `• Technical, academic, or formal writing may trigger false positives\n` +
                `• Human-edited AI content may not be detected\n` +
                `• Results should be considered alongside other evidence\n\n` +
                `This report should be used as one factor among many in evaluating text authenticity. ` +
                `Human judgment remains essential in all final determinations.`
        };
    },

    /**
     * Create DOCX blob using docx.js library or fallback to basic format
     */
    async createDocxBlob(reportContent) {
        // Check if docx library is available
        if (typeof docx !== 'undefined') {
            return this.createDocxWithLibrary(reportContent);
        } else {
            // Fallback: create a simple DOCX-compatible format
            return this.createBasicDocx(reportContent);
        }
    },

    /**
     * Create DOCX using docx.js library
     */
    async createDocxWithLibrary(report) {
        const { Document, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, 
                WidthType, BorderStyle, AlignmentType, Packer } = docx;

        const children = [];

        // Title
        children.push(new Paragraph({
            children: [new TextRun({ text: report.title, bold: true, size: 48 })],
            heading: HeadingLevel.TITLE,
            alignment: AlignmentType.CENTER
        }));

        children.push(new Paragraph({
            children: [new TextRun({ text: `Generated: ${new Date(report.generatedAt).toLocaleString()}`, italics: true, size: 20 })],
            alignment: AlignmentType.CENTER
        }));

        children.push(new Paragraph({ text: '' })); // Spacer

        // Executive Summary
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Executive Summary', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        children.push(new Paragraph({
            children: [
                new TextRun({ text: `Overall AI Probability: `, bold: true }),
                new TextRun({ text: `${report.summary.probability}% (${report.summary.band})` })
            ]
        }));

        children.push(new Paragraph({ text: report.summary.text }));

        // Verdict
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Verdict', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        children.push(new Paragraph({
            children: [
                new TextRun({ text: report.verdict.label, bold: true, size: 28 })
            ]
        }));

        children.push(new Paragraph({ text: report.verdict.description }));

        children.push(new Paragraph({
            children: [
                new TextRun({ text: 'Probability Range: ', bold: true }),
                new TextRun({ text: report.verdict.probabilityBand })
            ]
        }));

        if (report.verdict.falsePositiveRisk.factors.length > 0) {
            children.push(new Paragraph({
                children: [new TextRun({ text: 'False Positive Considerations:', bold: true })]
            }));
            for (const factor of report.verdict.falsePositiveRisk.factors) {
                children.push(new Paragraph({ text: `• ${factor}` }));
            }
        }

        // Category Weight Overview
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Category Weight Overview', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        children.push(new Paragraph({
            text: 'Each detection category contributes to the overall AI probability with different weights based on reliability and signal strength:'
        }));

        // Category weights table
        const weightTableRows = [
            new TableRow({
                children: [
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Category', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Weight', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Score', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Description', bold: true })] })] })
                ]
            })
        ];

        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05, description: 'Analyzes text characteristics.' };
            weightTableRows.push(new TableRow({
                children: [
                    new TableCell({ children: [new Paragraph({ text: cat.name })] }),
                    new TableCell({ children: [new Paragraph({ text: `${Math.round(weightInfo.weight * 100)}%` })] }),
                    new TableCell({ children: [new Paragraph({ text: `${cat.aiProbability}%` })] }),
                    new TableCell({ children: [new Paragraph({ text: weightInfo.description.slice(0, 80) + '...' })] })
                ]
            }));
        }

        children.push(new Table({ rows: weightTableRows, width: { size: 100, type: WidthType.PERCENTAGE } }));

        children.push(new Paragraph({ text: '' })); // Spacer

        // Category Analyses
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Detailed Category Analysis', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05, description: 'Analyzes text characteristics.' };
            const weightPct = Math.round(weightInfo.weight * 100);
            
            // Visual bar using ASCII
            const barLength = 20;
            const filledLength = Math.round((cat.aiProbability / 100) * barLength);
            const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);

            children.push(new Paragraph({
                children: [new TextRun({ text: `${cat.category}. ${cat.name}`, bold: true, size: 26 })],
                heading: HeadingLevel.HEADING_2
            }));

            children.push(new Paragraph({
                children: [
                    new TextRun({ text: `AI Probability: ` }),
                    new TextRun({ text: `${cat.aiProbability}%`, bold: true }),
                    new TextRun({ text: ` | Confidence: ${cat.confidence}% | Weight: ${weightPct}%`, italics: true })
                ]
            }));

            children.push(new Paragraph({
                children: [
                    new TextRun({ text: `[${bar}]`, font: 'Courier New', size: 18 })
                ]
            }));

            children.push(new Paragraph({ text: cat.description }));

            children.push(new Paragraph({
                children: [new TextRun({ text: `What this measures: `, bold: true }), new TextRun({ text: weightInfo.description })]
            }));

            if (cat.findings.length > 0) {
                children.push(new Paragraph({
                    children: [new TextRun({ text: 'Key Findings:', bold: true })]
                }));
                for (const finding of cat.findings.slice(0, 5)) {
                    const findingText = finding.text || finding.value || finding.label || 'Unknown finding';
                    const indicator = finding.indicator || 'neutral';
                    const icon = indicator === 'ai' ? '[AI]' : (indicator === 'human' ? '[Human]' : '[Mixed]');
                    children.push(new Paragraph({ text: `${icon} ${findingText}` }));
                }
            }

            children.push(new Paragraph({ text: '' })); // Spacer
        }

        // Methodology
        children.push(new Paragraph({
            children: [new TextRun({ text: report.methodology.title, bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        for (const para of report.methodology.content.split('\n\n')) {
            children.push(new Paragraph({ text: para }));
        }

        // Category Weight Reference Table
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Category Weight Reference', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        children.push(new Paragraph({
            text: 'Complete reference of all evaluation categories and their contribution to the final score:'
        }));

        const refTableRows = [
            new TableRow({
                children: [
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: '#', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Category Name', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'Weight', bold: true })] })] }),
                    new TableCell({ children: [new Paragraph({ children: [new TextRun({ text: 'What It Measures', bold: true })] })] })
                ]
            })
        ];

        for (let i = 1; i <= 14; i++) {
            const info = this.categoryWeightInfo[i];
            if (info) {
                refTableRows.push(new TableRow({
                    children: [
                        new TableCell({ children: [new Paragraph({ text: `${i}` })] }),
                        new TableCell({ children: [new Paragraph({ text: info.name })] }),
                        new TableCell({ children: [new Paragraph({ text: `${Math.round(info.weight * 100)}%` })] }),
                        new TableCell({ children: [new Paragraph({ text: info.description })] })
                    ]
                }));
            }
        }

        children.push(new Table({ rows: refTableRows, width: { size: 100, type: WidthType.PERCENTAGE } }));

        children.push(new Paragraph({ text: '' })); // Spacer

        // Disclaimer
        children.push(new Paragraph({
            children: [new TextRun({ text: report.disclaimer.title, bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        for (const para of report.disclaimer.content.split('\n\n')) {
            children.push(new Paragraph({ text: para }));
        }

        const doc = new Document({
            sections: [{ children }]
        });

        return await Packer.toBlob(doc);
    },

    /**
     * Create basic DOCX without library (minimal format)
     */
    createBasicDocx(report) {
        // Create a simple Office Open XML document
        const content = this.generatePlainTextReport(report);
        
        // For now, export as RTF which Word can open
        const rtf = this.textToRtf(content);
        return new Blob([rtf], { type: 'application/rtf' });
    },

    /**
     * Generate plain text version of report
     */
    generatePlainTextReport(report) {
        let text = '';
        
        text += `${'='.repeat(60)}\n`;
        text += `${report.title}\n`;
        text += `${'='.repeat(60)}\n\n`;
        text += `Generated: ${new Date(report.generatedAt).toLocaleString()}\n\n`;

        text += `EXECUTIVE SUMMARY\n${'-'.repeat(40)}\n`;
        text += `Overall AI Probability: ${report.summary.probability}% (${report.summary.band})\n`;
        text += `Confidence: ${report.summary.confidence}%\n\n`;
        text += `${report.summary.text}\n\n`;

        text += `VERDICT\n${'-'.repeat(40)}\n`;
        text += `${report.verdict.label}\n`;
        text += `${report.verdict.description}\n`;
        text += `Probability Range: ${report.verdict.probabilityBand}\n\n`;

        text += `DOCUMENT STATISTICS\n${'-'.repeat(40)}\n`;
        text += `Words: ${report.statistics.wordCount}\n`;
        text += `Sentences: ${report.statistics.sentenceCount}\n`;
        text += `Paragraphs: ${report.statistics.paragraphCount}\n`;
        text += `Analysis Time: ${report.statistics.analysisTime}\n\n`;

        // Category Weight Overview
        text += `CATEGORY WEIGHT OVERVIEW\n${'-'.repeat(40)}\n`;
        text += `Each category contributes to the overall score with the following weights:\n\n`;
        
        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05 };
            const weightPct = Math.round(weightInfo.weight * 100);
            const barLength = 20;
            const filledLength = Math.round((cat.aiProbability / 100) * barLength);
            const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
            text += `  ${cat.name.padEnd(32)} [${bar}] ${cat.aiProbability}% (Weight: ${weightPct}%)\n`;
        }
        text += `\n`;

        text += `DETAILED CATEGORY ANALYSIS\n${'-'.repeat(40)}\n\n`;
        
        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05, description: 'Analyzes text characteristics.' };
            const weightPct = Math.round(weightInfo.weight * 100);
            
            text += `${cat.category}. ${cat.name}\n`;
            text += `${'─'.repeat(35)}\n`;
            text += `AI Probability: ${cat.aiProbability}% | Confidence: ${cat.confidence}% | Weight: ${weightPct}%\n`;
            text += `${cat.description}\n`;
            text += `\nWhat this measures: ${weightInfo.description}\n`;
            if (cat.findings.length > 0) {
                text += `\nKey Findings:\n`;
                for (const f of cat.findings.slice(0, 5)) {
                    const findingText = f.text || f.value || f.label || 'Unknown finding';
                    const indicator = f.indicator || 'neutral';
                    const icon = indicator === 'ai' ? '[AI]' : (indicator === 'human' ? '[HUMAN]' : '[?]');
                    text += `  ${icon} ${findingText}\n`;
                }
            }
            text += `\n`;
        }

        text += `KEY FINDINGS SUMMARY\n${'-'.repeat(40)}\n\n`;
        
        if (report.keyFindings.aiIndicators.length > 0) {
            text += `AI Indicators:\n`;
            for (const f of report.keyFindings.aiIndicators.slice(0, 5)) {
                const findingText = f.text || f.value || f.label || 'Unknown finding';
                text += `  • ${findingText}\n`;
            }
            text += `\n`;
        }

        if (report.keyFindings.humanIndicators.length > 0) {
            text += `Human Indicators:\n`;
            for (const f of report.keyFindings.humanIndicators.slice(0, 3)) {
                const findingText = f.text || f.value || f.label || 'Unknown finding';
                text += `  • ${findingText}\n`;
            }
            text += `\n`;
        }

        // Category Weight Reference
        text += `CATEGORY WEIGHT REFERENCE\n${'-'.repeat(40)}\n`;
        text += `Complete reference of all 14 evaluation categories:\n\n`;
        
        for (let i = 1; i <= 14; i++) {
            const info = this.categoryWeightInfo[i];
            if (info) {
                text += `${i}. ${info.name} (${Math.round(info.weight * 100)}%)\n`;
                text += `   ${info.description}\n\n`;
            }
        }

        text += `METHODOLOGY\n${'-'.repeat(40)}\n`;
        text += `${report.methodology.content}\n\n`;

        text += `DISCLAIMER\n${'-'.repeat(40)}\n`;
        text += `${report.disclaimer.content}\n\n`;

        text += `${'='.repeat(60)}\n`;
        text += `End of Report\n`;

        return text;
    },

    /**
     * Convert text to basic RTF format
     */
    textToRtf(text) {
        // Escape special RTF characters
        const escaped = text
            .replace(/\\/g, '\\\\')
            .replace(/\{/g, '\\{')
            .replace(/\}/g, '\\}')
            .replace(/\n/g, '\\par\n');

        return `{\\rtf1\\ansi\\deff0
{\\fonttbl{\\f0\\fswiss Arial;}}
\\viewkind4\\uc1\\pard\\f0\\fs20
${escaped}
}`;
    },

    /**
     * Download blob as file
     */
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    },

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        if (!text) return '';
        return text
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    },

    /**
     * Export as Markdown
     */
    exportMarkdown(analysisResult, originalText) {
        const report = this.generateReportContent(analysisResult, originalText);
        let md = '';

        md += `# ${report.title}\n\n`;
        md += `*Generated: ${new Date(report.generatedAt).toLocaleString()}*\n\n`;

        md += `## Executive Summary\n\n`;
        md += `**Overall AI Probability:** ${report.summary.probability}% (${report.summary.band})\n\n`;
        md += `${report.summary.text}\n\n`;

        md += `## Verdict\n\n`;
        md += `### ${report.verdict.label}\n\n`;
        md += `${report.verdict.description}\n\n`;
        md += `- **Probability Range:** ${report.verdict.probabilityBand}\n`;
        md += `- **False Positive Risk:** ${report.verdict.falsePositiveRisk.level}\n\n`;

        md += `## Document Statistics\n\n`;
        md += `| Metric | Value |\n|--------|-------|\n`;
        md += `| Words | ${report.statistics.wordCount} |\n`;
        md += `| Sentences | ${report.statistics.sentenceCount} |\n`;
        md += `| Paragraphs | ${report.statistics.paragraphCount} |\n`;
        md += `| Analysis Time | ${report.statistics.analysisTime} |\n\n`;

        // Category Weight Overview with visual bars
        md += `## Category Weight Overview\n\n`;
        md += `Each detection category contributes to the overall score with the following weights:\n\n`;
        md += `| Category | Score | Weight | Visual |\n|----------|-------|--------|--------|\n`;
        
        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05 };
            const weightPct = Math.round(weightInfo.weight * 100);
            const barLength = 10;
            const filledLength = Math.round((cat.aiProbability / 100) * barLength);
            const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
            md += `| ${cat.name} | ${cat.aiProbability}% | ${weightPct}% | ${bar} |\n`;
        }
        md += `\n`;

        md += `## Category Analysis\n\n`;
        for (const cat of report.categoryAnalyses) {
            const weightInfo = this.categoryWeightInfo[cat.category] || { weight: 0.05, description: 'Analyzes text characteristics.' };
            const weightPct = Math.round(weightInfo.weight * 100);
            const barLength = 20;
            const filledLength = Math.round((cat.aiProbability / 100) * barLength);
            const bar = '█'.repeat(filledLength) + '░'.repeat(barLength - filledLength);
            
            md += `### ${cat.category}. ${cat.name}\n\n`;
            md += `*AI Probability: ${cat.aiProbability}% | Confidence: ${cat.confidence}% | Weight: ${weightPct}%*\n\n`;
            md += `\`[${bar}]\`\n\n`;
            md += `${cat.description}\n\n`;
            md += `> **What this measures:** ${weightInfo.description}\n\n`;
            if (cat.findings.length > 0) {
                md += `**Findings:**\n`;
                for (const f of cat.findings.slice(0, 5)) {
                    const icon = f.indicator === 'ai' ? '[AI]' : (f.indicator === 'human' ? '[Human]' : '[Mixed]');
                    const findingText = f.text || f.value || f.label || 'Unknown finding';
                    md += `- ${icon} ${findingText}\n`;
                }
                md += `\n`;
            }
        }

        // Category Weight Reference Table
        md += `## Category Weight Reference\n\n`;
        md += `Complete reference of all 14 evaluation categories:\n\n`;
        md += `| # | Category | Weight | Description |\n|---|----------|--------|-------------|\n`;
        
        for (let i = 1; i <= 14; i++) {
            const info = this.categoryWeightInfo[i];
            if (info) {
                md += `| ${i} | ${info.name} | ${Math.round(info.weight * 100)}% | ${info.description} |\n`;
            }
        }
        md += `\n`;

        md += `## Methodology\n\n`;
        md += `${report.methodology.content}\n\n`;

        md += `## Disclaimer\n\n`;
        md += `${report.disclaimer.content}\n\n`;

        md += `---\n\n`;
        md += `## Analyzed Text\n\n`;
        md += `\`\`\`\n${originalText}\n\`\`\`\n`;

        return md;
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReportExporter;
}
