/**
 * VERITAS — File Parser
 * Handles parsing of various file formats: TXT, DOCX, PDF, VTT, SRT, and clipboard/web content
 */

const FileParser = {
    /**
     * Supported file types
     */
    supportedTypes: {
        'text/plain': 'txt',
        'text/markdown': 'md',
        'text/vtt': 'vtt',
        'application/x-subrip': 'srt',
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
            
            case 'text/vtt':
            case 'application/x-subrip':
                return await this.parseSubtitle(file);
            
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
            'vtt': 'text/vtt',
            'srt': 'application/x-subrip',
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
     * Parse VTT (WebVTT) or SRT subtitle file
     * Strips timestamps and cue identifiers, extracts only the spoken text
     */
    async parseSubtitle(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const rawContent = e.target.result;
                const ext = file.name.split('.').pop().toLowerCase();
                
                let text;
                let cueCount = 0;
                
                if (ext === 'vtt') {
                    const result = this.parseVTT(rawContent);
                    text = result.text;
                    cueCount = result.cueCount;
                } else {
                    const result = this.parseSRT(rawContent);
                    text = result.text;
                    cueCount = result.cueCount;
                }
                
                resolve({
                    text: text,
                    metadata: {
                        filename: file.name,
                        size: file.size,
                        type: ext,
                        lastModified: new Date(file.lastModified).toISOString(),
                        originalFormat: ext.toUpperCase(),
                        cueCount: cueCount,
                        note: `Extracted ${cueCount} subtitle cues, timestamps stripped`
                    }
                });
            };
            reader.onerror = () => reject(new Error('Failed to read subtitle file'));
            reader.readAsText(file);
        });
    },

    /**
     * Parse WebVTT format
     * Format:
     * WEBVTT
     * 
     * 00:00:00.000 --> 00:00:05.000
     * This is the subtitle text
     * 
     * 00:00:05.000 --> 00:00:10.000
     * More text here
     */
    parseVTT(content) {
        const lines = content.split('\n');
        const textLines = [];
        let cueCount = 0;
        let inCue = false;
        
        // VTT timestamp pattern: 00:00:00.000 --> 00:00:00.000
        const timestampPattern = /^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}/;
        // Also match HH:MM:SS,mmm format (sometimes used)
        const altTimestampPattern = /^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}/;
        // Short format: MM:SS.mmm --> MM:SS.mmm
        const shortTimestampPattern = /^\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}/;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Skip WEBVTT header and metadata
            if (line === 'WEBVTT' || line.startsWith('NOTE') || line.startsWith('STYLE')) {
                continue;
            }
            
            // Skip empty lines
            if (line === '') {
                inCue = false;
                continue;
            }
            
            // Skip cue identifiers (numbers or strings before timestamps)
            if (timestampPattern.test(line) || altTimestampPattern.test(line) || shortTimestampPattern.test(line)) {
                inCue = true;
                cueCount++;
                continue;
            }
            
            // Skip lines that look like cue identifiers (just numbers)
            if (/^\d+$/.test(line)) {
                continue;
            }
            
            // Skip positioning tags like align:start position:0%
            if (line.includes('-->') || /^(align|position|line|size):/.test(line)) {
                continue;
            }
            
            // This is actual subtitle text
            if (inCue || (i > 0 && !timestampPattern.test(line))) {
                // Strip VTT tags like <v Speaker>, <c.yellow>, <b>, <i>, etc.
                let cleanLine = line
                    .replace(/<v\s+[^>]+>/g, '')     // <v Speaker Name>
                    .replace(/<\/v>/g, '')           // </v>
                    .replace(/<c\.[^>]+>/g, '')      // <c.classname>
                    .replace(/<\/c>/g, '')           // </c>
                    .replace(/<[biu]>/g, '')         // <b>, <i>, <u>
                    .replace(/<\/[biu]>/g, '')       // </b>, </i>, </u>
                    .replace(/<ruby>/g, '')          // ruby text
                    .replace(/<\/ruby>/g, '')
                    .replace(/<rt>/g, '')
                    .replace(/<\/rt>/g, '')
                    .replace(/<lang[^>]*>/g, '')     // language tags
                    .replace(/<\/lang>/g, '')
                    .replace(/&nbsp;/g, ' ')         // HTML entities
                    .replace(/&amp;/g, '&')
                    .replace(/&lt;/g, '<')
                    .replace(/&gt;/g, '>')
                    .trim();
                
                if (cleanLine) {
                    textLines.push(cleanLine);
                }
            }
        }
        
        // Join lines intelligently - don't add extra space between continued sentences
        const text = this.joinSubtitleLines(textLines);
        
        return { text, cueCount };
    },

    /**
     * Parse SRT (SubRip) format
     * Format:
     * 1
     * 00:00:00,000 --> 00:00:05,000
     * This is the subtitle text
     * 
     * 2
     * 00:00:05,000 --> 00:00:10,000
     * More text here
     */
    parseSRT(content) {
        const lines = content.split('\n');
        const textLines = [];
        let cueCount = 0;
        let inCue = false;
        
        // SRT timestamp pattern: 00:00:00,000 --> 00:00:00,000
        const timestampPattern = /^\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}/;
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            
            // Skip empty lines
            if (line === '') {
                inCue = false;
                continue;
            }
            
            // Skip cue numbers
            if (/^\d+$/.test(line)) {
                continue;
            }
            
            // Skip timestamps
            if (timestampPattern.test(line)) {
                inCue = true;
                cueCount++;
                continue;
            }
            
            // This is subtitle text
            if (inCue) {
                // Strip common HTML-like tags in SRT
                let cleanLine = line
                    .replace(/<[^>]+>/g, '')         // Remove all HTML tags
                    .replace(/\{[^}]+\}/g, '')       // Remove ASS/SSA style tags {\\an8}
                    .replace(/&nbsp;/g, ' ')
                    .replace(/&amp;/g, '&')
                    .replace(/&lt;/g, '<')
                    .replace(/&gt;/g, '>')
                    .trim();
                
                if (cleanLine) {
                    textLines.push(cleanLine);
                }
            }
        }
        
        const text = this.joinSubtitleLines(textLines);
        
        return { text, cueCount };
    },

    /**
     * Intelligently join subtitle lines into flowing text
     * - Lines ending with punctuation get a space after
     * - Lines ending mid-word/sentence get joined directly or with space
     */
    joinSubtitleLines(lines) {
        if (lines.length === 0) return '';
        
        const result = [];
        
        for (let i = 0; i < lines.length; i++) {
            const line = lines[i];
            const prevLine = i > 0 ? lines[i - 1] : '';
            
            // Check if previous line ends with sentence-ending punctuation
            const prevEndsWithPunctuation = /[.!?]$/.test(prevLine);
            // Check if previous line ends with continuation punctuation
            const prevEndsWithContinuation = /[,;:\-–—]$/.test(prevLine);
            // Check if current line starts with lowercase (continuation)
            const startsWithLowercase = /^[a-z]/.test(line);
            
            if (i === 0) {
                result.push(line);
            } else if (prevEndsWithPunctuation) {
                // New sentence - add space
                result.push(' ' + line);
            } else if (prevEndsWithContinuation || startsWithLowercase) {
                // Continuation - add space
                result.push(' ' + line);
            } else {
                // Default - add space
                result.push(' ' + line);
            }
        }
        
        return result.join('').trim();
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
