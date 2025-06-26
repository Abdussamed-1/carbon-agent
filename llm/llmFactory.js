// llm/llmFactory.js
/**
 * LLM Factory - Granite fine-tuned model dahil
 */

import { OpenAI } from "@langchain/openai";
import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { OpenAIEmbeddings } from "@langchain/openai";
import { createGraniteLLM } from "./graniteAgent.js";

export function getLLM(config) {
    const llmType = config.LLM_TYPE?.toLowerCase() || "granite";
    
    console.log(`🤖 LLM başlatılıyor: ${llmType}`);
    
    switch (llmType) {
        case "granite":
        case "granite-fine-tuned":
            console.log("🚀 Fine-tuned Granite model yükleniyor...");
            return createGraniteLLM({
                GRANITE_MODEL_PATH: config.GRANITE_MODEL_PATH,
                MAX_TOKENS: config.MAX_TOKENS || 150,
                TEMPERATURE: config.TEMPERATURE || 0.7,
                PYTHON_PATH: config.PYTHON_PATH || "python"
            });
            
        case "openai":
            if (!config.OPENAI_API_KEY) {
                throw new Error("OpenAI API key gerekli");
            }
            return new OpenAI({
                openAIApiKey: config.OPENAI_API_KEY,
                modelName: config.OPENAI_MODEL || "gpt-3.5-turbo-instruct",
                temperature: config.TEMPERATURE || 0.7,
                maxTokens: config.MAX_TOKENS || 150
            });
            
        case "huggingface":
            if (!config.HF_TOKEN) {
                throw new Error("HuggingFace token gerekli");
            }
            return new HuggingFaceInference({
                model: config.HF_MODEL || "microsoft/DialoGPT-medium",
                apiKey: config.HF_TOKEN,
                temperature: config.TEMPERATURE || 0.7,
                maxTokens: config.MAX_TOKENS || 150
            });
            
        default:
            console.log("⚠️ Bilinmeyen LLM tipi, Granite kullanılıyor");
            return createGraniteLLM({
                GRANITE_MODEL_PATH: config.GRANITE_MODEL_PATH,
                MAX_TOKENS: config.MAX_TOKENS || 150,
                TEMPERATURE: config.TEMPERATURE || 0.7
            });
    }
}

export function getEmbeddings(config) {
    const embeddingType = config.EMBEDDING_TYPE?.toLowerCase() || "huggingface";
    
    console.log(`🔤 Embeddings başlatılıyor: ${embeddingType}`);
    
    switch (embeddingType) {
        case "openai":
            if (!config.OPENAI_API_KEY) {
                throw new Error("OpenAI API key gerekli");
            }
            return new OpenAIEmbeddings({
                openAIApiKey: config.OPENAI_API_KEY,
                modelName: config.EMBEDDING_MODEL || "text-embedding-ada-002"
            });
            
        case "huggingface":
        default:
            return new HuggingFaceTransformersEmbeddings({
                modelName: config.EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2"
            });
    }
}

// Test fonksiyonu
export async function testLLM(config) {
    try {
        console.log("🧪 LLM test ediliyor...");
        
        const llm = getLLM(config);
        const testPrompt = "Tesla'nın ESG performansı hakkında kısa bilgi ver";
        
        console.log("📝 Test prompt:", testPrompt);
        const response = await llm.call(testPrompt);
        
        console.log("✅ LLM testi başarılı");
        console.log("🤖 Yanıt:", response);
        
        return { success: true, response };
        
    } catch (error) {
        console.error("❌ LLM test hatası:", error);
        return { success: false, error: error.message };
    }
}

// Agent için optimized LLM
export function getESGOptimizedLLM(config) {
    const llm = getLLM(config);
    
    // ESG-specific system prompt wrapper
    const originalCall = llm.call;
    llm.call = async function(prompt, options) {
        const esgPrompt = `ESG (Environment, Social, Governance) uzmanı olarak aşağıdaki soruya kısa ve net yanıt ver:

${prompt}

Yanıtında şunları içer:
- Ana ESG faktörleri
- Somut öneriler
- Skor/değerlendirme (varsa)`;
        
        return await originalCall.call(this, esgPrompt, options);
    };
    
    return llm;
}