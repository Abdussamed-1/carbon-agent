// vectorstore/embeddings.js - Embedding işlemleri
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { MODEL_CONFIG } from '../config/models.js';
import fs from 'fs-extra';
import path from 'path';

export class EmbeddingManager {
    constructor(options = {}) {
        this.modelName = options.modelName || MODEL_CONFIG.EMBEDDINGS.modelName;
        this.maxConcurrency = options.maxConcurrency || MODEL_CONFIG.EMBEDDINGS.maxConcurrency;
        this.cacheFolder = options.cacheFolder || MODEL_CONFIG.EMBEDDINGS.cacheFolder;
        this.embeddings = null;
        this.isInitialized = false;
    }

    async initialize() {
        try {
            console.log('🔤 Embedding modeli yükleniyor...');
            
            // Cache klasörünü oluştur
            await fs.ensureDir(this.cacheFolder);
            
            this.embeddings = new HuggingFaceTransformersEmbeddings({
                modelName: this.modelName,
                maxConcurrency: this.maxConcurrency,
                cacheFolder: this.cacheFolder
            });

            // Test embedding oluştur
            const testEmbedding = await this.embeddings.embedQuery("test");
            console.log(`✅ Embedding modeli hazır! Boyut: ${testEmbedding.length}`);

            this.isInitialized = true;
            return true;

        } catch (error) {
            console.error('❌ Embedding modeli yükleme hatası:', error);
            throw error;
        }
    }

    async embedText(text) {
        if (!this.isInitialized) {
            throw new Error('Embedding modeli henüz başlatılmadı');
        }

        try {
            return await this.embeddings.embedQuery(text);
        } catch (error) {
            console.error('Embedding hatası:', error);
            throw error;
        }
    }

    async embedDocuments(documents) {
        if (!this.isInitialized) {
            throw new Error('Embedding modeli henüz başlatılmadı');
        }

        try {
            return await this.embeddings.embedDocuments(documents);
        } catch (error) {
            console.error('Doküman embedding hatası:', error);
            throw error;
        }
    }

    async embedBatch(texts, batchSize = 100) {
        if (!this.isInitialized) {
            throw new Error('Embedding modeli henüz başlatılmadı');
        }

        const embeddings = [];
        
        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            console.log(`📦 Batch ${Math.floor(i/batchSize) + 1}/${Math.ceil(texts.length/batchSize)} işleniyor...`);
            
            const batchEmbeddings = await this.embeddings.embedDocuments(batch);
            embeddings.push(...batchEmbeddings);
        }

        return embeddings;
    }

    // Benzerlik hesaplama
    calculateSimilarity(embedding1, embedding2) {
        if (embedding1.length !== embedding2.length) {
            throw new Error('Embedding boyutları eşleşmiyor');
        }

        const dotProduct = embedding1.reduce((sum, val, i) => sum + val * embedding2[i], 0);
        const magnitude1 = Math.sqrt(embedding1.reduce((sum, val) => sum + val * val, 0));
        const magnitude2 = Math.sqrt(embedding2.reduce((sum, val) => sum + val * val, 0));

        return dotProduct / (magnitude1 * magnitude2);
    }

    // Top-k benzer dokümanları bul
    async findSimilarDocuments(queryEmbedding, documentEmbeddings, k = 5) {
        const similarities = documentEmbeddings.map((docEmbedding, index) => ({
            index,
            similarity: this.calculateSimilarity(queryEmbedding, docEmbedding)
        }));

        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, k);
    }

    // Embedding cache yönetimi
    async saveEmbeddingCache(key, embedding) {
        const cacheFile = path.join(this.cacheFolder, `${key}.json`);
        await fs.writeJson(cacheFile, embedding);
    }

    async loadEmbeddingCache(key) {
        const cacheFile = path.join(this.cacheFolder, `${key}.json`);
        
        if (await fs.pathExists(cacheFile)) {
            return await fs.readJson(cacheFile);
        }
        
        return null;
    }

    async clearCache() {
        if (await fs.pathExists(this.cacheFolder)) {
            await fs.emptyDir(this.cacheFolder);
            console.log('🗑️ Embedding cache temizlendi');
        }
    }

    // Model bilgileri
    getModelInfo() {
        return {
            modelName: this.modelName,
            maxConcurrency: this.maxConcurrency,
            cacheFolder: this.cacheFolder,
            isInitialized: this.isInitialized
        };
    }

    // Performans metrikleri
    async getPerformanceMetrics() {
        if (!this.isInitialized) {
            return { error: 'Model henüz başlatılmadı' };
        }

        const startTime = Date.now();
        const testText = "ESG performance analysis test";
        
        try {
            await this.embedText(testText);
            const responseTime = Date.now() - startTime;

            return {
                responseTime,
                modelName: this.modelName,
                cacheSize: await this.getCacheSize(),
                status: 'healthy'
            };
        } catch (error) {
            return {
                error: error.message,
                status: 'error'
            };
        }
    }

    async getCacheSize() {
        if (!await fs.pathExists(this.cacheFolder)) {
            return 0;
        }

        const files = await fs.readdir(this.cacheFolder);
        return files.length;
    }
}

// Utility fonksiyonlar
export const embeddingUtils = {
    // Text chunking
    chunkText(text, chunkSize = MODEL_CONFIG.RAG.chunkSize, overlap = MODEL_CONFIG.RAG.chunkOverlap) {
        const chunks = [];
        let start = 0;

        while (start < text.length) {
            const end = Math.min(start + chunkSize, text.length);
            chunks.push(text.slice(start, end));
            start = end - overlap;
        }

        return chunks;
    },

    // Text preprocessing
    preprocessText(text) {
        return text
            .replace(/\s+/g, ' ') // Fazla boşlukları temizle
            .replace(/[^\w\s.,!?-]/g, '') // Özel karakterleri temizle
            .trim();
    },

    // Semantic search için query optimization
    optimizeQuery(query) {
        const keywords = [
            'ESG', 'environmental', 'social', 'governance',
            'sustainability', 'carbon', 'emissions', 'renewable',
            'social responsibility', 'corporate governance'
        ];

        const optimizedQuery = query.toLowerCase();
        const hasKeywords = keywords.some(keyword => 
            optimizedQuery.includes(keyword.toLowerCase())
        );

        if (!hasKeywords) {
            return `${query} ESG sustainability`;
        }

        return query;
    }
};

export default EmbeddingManager; 