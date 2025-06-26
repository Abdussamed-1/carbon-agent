// config/models.js - Model konfigürasyonları
import dotenv from 'dotenv';

dotenv.config();

export const MODEL_CONFIG = {
    // IBM Granite Model Ayarları
    GRANITE: {
        baseModel: 'ibm-granite/granite-3.0-8b-instruct',
        customModelPath: process.env.CUSTOM_MODEL_PATH || './models/granite-esg-finetuned',
        temperature: 0.1,
        maxTokens: 2048,
        topP: 0.9,
        frequencyPenalty: 0.1,
        presencePenalty: 0.1
    },

    // Embedding Model Ayarları
    EMBEDDINGS: {
        modelName: "sentence-transformers/all-MiniLM-L6-v2",
        maxConcurrency: 5,
        cacheFolder: "./cache/embeddings"
    },

    // Fine-tuning Ayarları
    FINE_TUNING: {
        learningRate: 2e-5,
        batchSize: 4,
        epochs: 3,
        warmupSteps: 100,
        weightDecay: 0.01,
        gradientAccumulationSteps: 4,
        maxGradNorm: 1.0,
        saveSteps: 500,
        evalSteps: 500,
        loggingSteps: 100
    },

    // RAG Ayarları
    RAG: {
        chunkSize: 1000,
        chunkOverlap: 200,
        similarityThreshold: 0.7,
        maxResults: 10,
        rerankTopK: 5
    },

    // Agent Ayarları
    AGENT: {
        maxIterations: 10,
        verbose: true,
        returnIntermediateSteps: true,
        maxConcurrency: 3
    },

    // ChromaDB Ayarları
    CHROMA_CONFIG: {
        path: process.env.CHROMA_DB_PATH || 'http://localhost:8000',
        collectionName: 'esg_documents'
    }
};

export const API_CONFIG = {
    port: process.env.PORT || 3000,
    rateLimit: {
        windowMs: 15 * 60 * 1000, // 15 dakika
        max: 100 // IP başına maksimum istek
    },
    cors: {
        origin: process.env.CORS_ORIGIN || '*',
        credentials: true
    }
};

export const LOGGING_CONFIG = {
    level: process.env.LOG_LEVEL || 'info',
    format: 'json',
    transports: ['file', 'console'],
    filename: './logs/app.log',
    maxSize: '10m',
    maxFiles: 5
};

export const DATA_CONFIG = {
    inputPath: './data/esg_data.csv',
    processedPath: './data/processed/',
    backupPath: './data/backup/',
    supportedFormats: ['csv', 'json', 'xlsx'],
    batchSize: 1000
};

// Model validation
export function validateModelConfig() {
    const requiredEnvVars = [
        'HF_TOKEN',
        'CHROMA_DB_PATH'
    ];

    const missing = requiredEnvVars.filter(varName => !process.env[varName]);
    
    if (missing.length > 0) {
        console.warn(`⚠️ Eksik environment variables: ${missing.join(', ')}`);
        console.warn('Bazı özellikler çalışmayabilir.');
    }

    return missing.length === 0;
}

// Model performance metrics
export const PERFORMANCE_METRICS = {
    responseTime: {
        target: 2000, // ms
        warning: 5000
    },
    accuracy: {
        target: 0.85,
        warning: 0.70
    },
    throughput: {
        target: 100, // requests/minute
        warning: 50
    }
};

export default {
    MODEL_CONFIG,
    API_CONFIG,
    LOGGING_CONFIG,
    DATA_CONFIG,
    validateModelConfig,
    PERFORMANCE_METRICS
}; 