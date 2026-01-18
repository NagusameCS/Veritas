/**
 * VERITAS — File Parser
 * Handles parsing of various file formats: TXT, DOCX, PDF, and clipboard/web content
 */

const FileParser = {
    /**
     * Supported file types
     */
    supportedTypes: {
        'text/plain': 'txt',
        'text/markdown': 'md',
        'application/pdf': 'pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
        'application/msword': 'doc'
    },

    /**
     * Parse file based on type
     */
    async parseFile(file) {
        const type = file.type || this.getTypeFromExtension(file.name);
        
        switch (type) {
            case 'text/plain':
            case 'text/markdown':
                return await this.parseText(file);
            
            case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                return await this.parseDocx(file);
            
            case 'application/pdf':
                return await this.parsePdf(file);
            
            default:
                // Try as text
                return await this.parseText(file);
        }
    },

    /**
     * Get type from file extension
     */
    getTypeFromExtension(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const typeMap = {
            'txt': 'text/plain',
            'md': 'text/markdown',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'doc': 'application/msword',
            'pdf': 'application/pdf'
        };
        return typeMap[ext] || 'text/plain';
    },

    /**
     * Parse plain text file
     */
    async parseText(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                resolve({
                    text: e.target.result,
                    metadata: {
                        filename: file.name,
                        size: file.size,
                        type: 'text',
                        lastModified: new Date(file.lastModified).toISOString()
                    }
                });
            };
            reader.onerror = () => reject(new Error('Failed to read text file'));
            reader.readAsText(file);
        });
    },

    /**
     * Parse DOCX file
     * Uses a lightweight approach - extracts text from document.xml
     */
    async parseDocx(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    // DOCX is a zip file
                    const arrayBuffer = e.target.result;
                    const content = await this.extractDocxContent(arrayBuffer);
                    
                    resolve({
                        text: content.text,
                        metadata: {
                            filename: file.name,
                            size: file.size,
                            type: 'docx',
                            lastModified: new Date(file.lastModified).toISOString(),
                            ...content.metadata
                        }
                    });
                } catch (error) {
                    reject(new Error('Failed to parse DOCX: ' + error.message));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read DOCX file'));
            reader.readAsArrayBuffer(file);
        });
    },

    /**
     * Extract content from DOCX (ZIP) file
     */
    async extractDocxContent(arrayBuffer) {
        // Use JSZip if available, otherwise basic extraction
        if (typeof JSZip !== 'undefined') {
            return await this.extractDocxWithJSZip(arrayBuffer);
        } else {
            // Fallback: try to extract text patterns
            return this.extractDocxBasic(arrayBuffer);
        }
    },

    /**
     * Extract DOCX content using JSZip
     */
    async extractDocxWithJSZip(arrayBuffer) {
        const zip = await JSZip.loadAsync(arrayBuffer);
        
        // Get document.xml
        const documentXml = await zip.file('word/document.xml')?.async('string');
        if (!documentXml) {
            throw new Error('Invalid DOCX: missing document.xml');
        }

        // Parse XML and extract text
        const text = this.extractTextFromDocxXml(documentXml);
        
        // Try to get metadata
        const metadata = {};
        const coreXml = await zip.file('docProps/core.xml')?.async('string');
        if (coreXml) {
            metadata.core = this.parseDocxMetadata(coreXml);
        }

        return { text, metadata };
    },

    /**
     * Basic DOCX extraction without JSZip
     */
    extractDocxBasic(arrayBuffer) {
        // Convert to string and look for text patterns
        const bytes = new Uint8Array(arrayBuffer);
        let text = '';
        
        // DOCX XML contains text in <w:t> tags
        // This is a simplified extraction
        const decoder = new TextDecoder('utf-8', { fatal: false });
        const content = decoder.decode(bytes);
        
        // Extract text between <w:t> tags
        const matches = content.match(/<w:t[^>]*>([^<]*)<\/w:t>/g);
        if (matches) {
            text = matches
                .map(m => m.replace(/<[^>]+>/g, ''))
                .join(' ')
                .replace(/\s+/g, ' ')
                .trim();
        }

        return { 
            text: text || 'Unable to extract text from DOCX. Please try a different format.',
            metadata: {} 
        };
    },

    /**
     * Extract text from DOCX XML
     */
    extractTextFromDocxXml(xml) {
        const paragraphs = [];
        
        // Parse paragraphs
        const paraMatches = xml.match(/<w:p[^>]*>[\s\S]*?<\/w:p>/g) || [];
        
        for (const para of paraMatches) {
            const textMatches = para.match(/<w:t[^>]*>([^<]*)<\/w:t>/g) || [];
            const paraText = textMatches
                .map(m => m.replace(/<[^>]+>/g, ''))
                .join('');
            
            if (paraText.trim()) {
                paragraphs.push(paraText);
            }
        }

        return paragraphs.join('\n\n');
    },

    /**
     * Parse DOCX metadata from core.xml
     */
    parseDocxMetadata(xml) {
        const metadata = {};
        
        const fields = ['creator', 'created', 'modified', 'title', 'subject', 'lastModifiedBy'];
        for (const field of fields) {
            const match = xml.match(new RegExp(`<dc:${field}>([^<]*)</dc:${field}>`, 'i')) ||
                         xml.match(new RegExp(`<cp:${field}>([^<]*)</cp:${field}>`, 'i'));
            if (match) {
                metadata[field] = match[1];
            }
        }

        return metadata;
    },

    /**
     * Parse PDF file
     * Uses pdf.js if available, otherwise returns error message
     */
    async parsePdf(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = async (e) => {
                try {
                    const arrayBuffer = e.target.result;
                    
                    if (typeof pdfjsLib !== 'undefined') {
                        const content = await this.extractPdfContent(arrayBuffer);
                        resolve({
                            text: content.text,
                            metadata: {
                                filename: file.name,
                                size: file.size,
                                type: 'pdf',
                                lastModified: new Date(file.lastModified).toISOString(),
                                ...content.metadata
                            }
                        });
                    } else {
                        // No PDF.js available
                        resolve({
                            text: '',
                            metadata: {
                                filename: file.name,
                                size: file.size,
                                type: 'pdf',
                                error: 'PDF parsing requires pdf.js library. Please copy and paste the text instead.'
                            }
                        });
                    }
                } catch (error) {
                    reject(new Error('Failed to parse PDF: ' + error.message));
                }
            };
            reader.onerror = () => reject(new Error('Failed to read PDF file'));
            reader.readAsArrayBuffer(file);
        });
    },

    /**
     * Extract PDF content using pdf.js
     */
    async extractPdfContent(arrayBuffer) {
        const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
        const metadata = await pdf.getMetadata().catch(() => ({}));
        
        const textParts = [];
        
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const textContent = await page.getTextContent();
            const pageText = textContent.items
                .map(item => item.str)
                .join(' ');
            textParts.push(pageText);
        }

        return {
            text: textParts.join('\n\n'),
            metadata: {
                pageCount: pdf.numPages,
                info: metadata.info || {}
            }
        };
    },

    /**
     * Parse clipboard content
     */
    async parseClipboard(clipboardData) {
        const text = clipboardData.getData('text/plain');
        const html = clipboardData.getData('text/html');
        
        const result = {
            text: text,
            metadata: {
                type: 'clipboard',
                hasHtml: !!html,
                timestamp: new Date().toISOString()
            }
        };

        // If HTML is present, check for Google Docs markers
        if (html) {
            result.metadata.source = this.detectClipboardSource(html);
            result.metadata.hiddenFormatting = this.detectHiddenFormatting(html);
        }

        return result;
    },

    /**
     * Detect source of clipboard content
     */
    detectClipboardSource(html) {
        if (html.includes('docs-internal-guid') || html.includes('google-docs')) {
            return 'google-docs';
        }
        if (html.includes('MsoNormal') || html.includes('Microsoft')) {
            return 'microsoft-office';
        }
        if (html.includes('notion')) {
            return 'notion';
        }
        return 'unknown';
    },

    /**
     * Detect hidden formatting in HTML
     */
    detectHiddenFormatting(html) {
        const issues = [];
        
        if (html.includes('&nbsp;')) {
            issues.push('non-breaking-spaces');
        }
        if (html.match(/style\s*=\s*["'][^"']*font/i)) {
            issues.push('inline-font-styles');
        }
        if (html.includes('<!--')) {
            issues.push('html-comments');
        }
        if (html.match(/\u200B|\u200C|\u200D|\uFEFF/)) {
            issues.push('zero-width-characters');
        }

        return issues;
    },

    /**
     * Parse Google Docs URL (requires API access or public sharing)
     */
    async parseGoogleDocsUrl(url) {
        // Extract document ID from URL
        const match = url.match(/\/document\/d\/([a-zA-Z0-9-_]+)/);
        if (!match) {
            throw new Error('Invalid Google Docs URL');
        }

        const docId = match[1];
        
        // Try to fetch as exported text (only works for publicly shared docs)
        const exportUrl = `https://docs.google.com/document/d/${docId}/export?format=txt`;
        
        try {
            const response = await fetch(exportUrl);
            if (response.ok) {
                const text = await response.text();
                return {
                    text,
                    metadata: {
                        type: 'google-docs',
                        documentId: docId,
                        url: url
                    }
                };
            }
        } catch (error) {
            // Fetch failed, document may not be publicly accessible
        }

        return {
            text: '',
            metadata: {
                type: 'google-docs',
                documentId: docId,
                error: 'Unable to access document. Please ensure it is publicly shared or copy the text directly.'
            }
        };
    },

    /**
     * Validate and clean extracted text
     */
    cleanExtractedText(text, preserveFormatting = false) {
        if (!text) return '';

        let cleaned = text;

        // Normalize line endings
        cleaned = cleaned.replace(/\r\n/g, '\n').replace(/\r/g, '\n');

        // Remove null characters
        cleaned = cleaned.replace(/\u0000/g, '');

        if (!preserveFormatting) {
            // Collapse multiple spaces
            cleaned = cleaned.replace(/[ \t]+/g, ' ');
            
            // Collapse multiple newlines
            cleaned = cleaned.replace(/\n{3,}/g, '\n\n');
        }

        return cleaned.trim();
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FileParser;
}
