/**
 * VERITAS — Report Exporter
 * Generates professional DOCX reports with dynamic template-based content
 */

const ReportExporter = {
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
     * Generate structured report content
     */
    generateReportContent(result, originalText) {
        const report = {
            title: 'VERITAS AI Detection Analysis Report',
            generatedAt: new Date().toISOString(),
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

        // Category Analyses
        children.push(new Paragraph({
            children: [new TextRun({ text: 'Detailed Category Analysis', bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        for (const cat of report.categoryAnalyses) {
            children.push(new Paragraph({
                children: [new TextRun({ text: `${cat.category}. ${cat.name}`, bold: true, size: 26 })],
                heading: HeadingLevel.HEADING_2
            }));

            children.push(new Paragraph({
                children: [
                    new TextRun({ text: `AI Probability: ${cat.aiProbability}% | Confidence: ${cat.confidence}%`, italics: true })
                ]
            }));

            children.push(new Paragraph({ text: cat.description }));

            if (cat.findings.length > 0) {
                children.push(new Paragraph({
                    children: [new TextRun({ text: 'Findings:', bold: true })]
                }));
                for (const finding of cat.findings.slice(0, 3)) {
                    children.push(new Paragraph({ text: `• ${finding.text}` }));
                }
            }
        }

        // Methodology
        children.push(new Paragraph({
            children: [new TextRun({ text: report.methodology.title, bold: true, size: 32 })],
            heading: HeadingLevel.HEADING_1
        }));

        for (const para of report.methodology.content.split('\n\n')) {
            children.push(new Paragraph({ text: para }));
        }

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

        text += `DETAILED CATEGORY ANALYSIS\n${'-'.repeat(40)}\n\n`;
        
        for (const cat of report.categoryAnalyses) {
            text += `${cat.category}. ${cat.name}\n`;
            text += `AI Probability: ${cat.aiProbability}% | Confidence: ${cat.confidence}%\n`;
            text += `${cat.description}\n`;
            if (cat.findings.length > 0) {
                text += `Key Findings:\n`;
                for (const f of cat.findings.slice(0, 3)) {
                    text += `  • ${f.text}\n`;
                }
            }
            text += `\n`;
        }

        text += `KEY FINDINGS SUMMARY\n${'-'.repeat(40)}\n\n`;
        
        if (report.keyFindings.aiIndicators.length > 0) {
            text += `AI Indicators:\n`;
            for (const f of report.keyFindings.aiIndicators.slice(0, 5)) {
                text += `  • ${f.text}\n`;
            }
            text += `\n`;
        }

        if (report.keyFindings.humanIndicators.length > 0) {
            text += `Human Indicators:\n`;
            for (const f of report.keyFindings.humanIndicators.slice(0, 3)) {
                text += `  • ${f.text}\n`;
            }
            text += `\n`;
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

        md += `## Category Analysis\n\n`;
        for (const cat of report.categoryAnalyses) {
            md += `### ${cat.category}. ${cat.name}\n\n`;
            md += `*AI Probability: ${cat.aiProbability}% | Confidence: ${cat.confidence}%*\n\n`;
            md += `${cat.description}\n\n`;
            if (cat.findings.length > 0) {
                md += `**Findings:**\n`;
                for (const f of cat.findings.slice(0, 3)) {
                    const icon = f.indicator === 'ai' ? '🤖' : (f.indicator === 'human' ? '👤' : '⚖️');
                    md += `- ${icon} ${f.text}\n`;
                }
                md += `\n`;
            }
        }

        md += `## Methodology\n\n`;
        md += `${report.methodology.content}\n\n`;

        md += `## Disclaimer\n\n`;
        md += `${report.disclaimer.content}\n\n`;

        md += `---\n\n`;
        md += `## Analyzed Text\n\n`;
        md += `\`\`\`\n${originalText}\n\`\`\`\n`;

        // Download
        const blob = new Blob([md], { type: 'text/markdown' });
        this.downloadBlob(blob, `veritas-report-${Date.now()}.md`);
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ReportExporter;
}
