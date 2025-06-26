// agents/tools/ragRetriever.js
export class RAGRetrieverTool {
    constructor({ chromaManager, embeddings }) {
        this.name = 'rag_retriever';
        this.description = 'ChromaDB üzerinden semantik arama yapar ve en alakalı dokümanları döndürür.';
        this.chromaManager = chromaManager;
        this.embeddings = embeddings;
    }

    /**
     * @param {Object|string} input - { query: string, nResults?: number, filter?: object }
     * @returns {string} - JSON stringified results or error
     */
    async call(input) {
        let parsed;
        try {
            parsed = typeof input === 'string' ? JSON.parse(input) : input;
        } catch {
            return JSON.stringify({ error: 'Input JSON formatında olmalı.' });
        }
        const { query, nResults = 5, filter = {} } = parsed;

        if (!query) {
            return JSON.stringify({ error: 'Query alanı zorunludur.' });
        }

        try {
            const results = await this.chromaManager.searchSimilar(query, { nResults, whereFilter: filter });
            return JSON.stringify({
                success: true,
                results: results.results
            });
        } catch (err) {
            return JSON.stringify({ error: 'ChromaDB arama hatası', details: err.message });
        }
    }
}