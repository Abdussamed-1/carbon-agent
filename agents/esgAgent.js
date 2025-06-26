// agents/esgAgent.js
import { HuggingFaceInference } from "@langchain/community/llms/hf";
import { AgentExecutor, createReactAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { ESGAnalyzerTool } from './tools/esgAnalyzer.js';
import { RAGRetrieverTool } from './tools/ragRetriever.js';
import { StrategyGeneratorTool } from './tools/strategyGenerator.js';

export class ESGAgent {
    constructor(options = {}) {
        this.modelPath = options.modelPath || 'microsoft/DialoGPT-medium';
        this.customModelPath = options.customModelPath;
        this.chromaManager = options.chromaManager;
        this.isReady = false;
        this.sessionMemory = new Map();
    }

    async initialize() {
        try {
            console.log('🤖 ESG Agent modelleri yükleniyor...');
            
            // Ana model (fine-tuned varsa onu kullan)
            const modelToUse = this.customModelPath && await this.checkCustomModel() 
                ? this.customModelPath 
                : this.modelPath;

            // HuggingFaceInference kullan
            this.llm = new HuggingFaceInference({
                model: modelToUse,
                apiKey: process.env.HUGGINGFACE_API_KEY || process.env.HF_TOKEN,
                temperature: 0.1,
                maxTokens: 2048
            });

            // Embedding modeli
            this.embeddings = new HuggingFaceTransformersEmbeddings({
                modelName: "sentence-transformers/all-MiniLM-L6-v2"
            });

            // Tools oluştur
            this.tools = [
                new ESGAnalyzerTool({ chromaManager: this.chromaManager }),
                new RAGRetrieverTool({ 
                    chromaManager: this.chromaManager,
                    embeddings: this.embeddings 
                }),
                new StrategyGeneratorTool({ llm: this.llm })
            ];

            // React agent oluştur
            const prompt = await pull("hwchase17/react");
            this.agent = await createReactAgent({
                llm: this.llm,
                tools: this.tools,
                prompt
            });

            this.agentExecutor = new AgentExecutor({
                agent: this.agent,
                tools: this.tools,
                verbose: true,
                maxIterations: 10
            });

            this.isReady = true;
            console.log('✅ ESG Agent hazır!');

        } catch (error) {
            console.error('❌ ESG Agent başlatma hatası:', error);
            throw error;
        }
    }

    async checkCustomModel() {
        try {
            const fs = await import('fs');
            return fs.existsSync(this.customModelPath);
        } catch {
            return false;
        }
    }

    async processMessage({ message, sessionId = 'default' }) {
        if (!this.isReady) {
            throw new Error('Agent henüz hazır değil');
        }

        try {
            // Session memory al veya oluştur
            let sessionContext = this.sessionMemory.get(sessionId) || {
                messages: [],
                context: {}
            };

            // ESG odaklı sistem promptu ekle
            const systemPrompt = `Sen ESG (Environmental, Social, Governance) konularında uzman bir danışmansın. 
            Şirketlere sürdürülebilirlik stratejileri konusunda veri odaklı öneriler sunuyorsun.
            
            Mevcut araçların:
            1. ESGAnalyzer: Şirket ESG skorlarını analiz eder
            2. RAGRetriever: ESG dokümanlarından bilgi getirir  
            3. StrategyGenerator: Veri temelli strateji önerileri oluşturur
            
            Her zaman veri temelli, objektif ve uygulanabilir öneriler sun.`;

            const fullMessage = `${systemPrompt}\n\nKullanıcı: ${message}`;

            // Agent'ı çalıştır
            const result = await this.agentExecutor.invoke({
                input: fullMessage,
                sessionId: sessionId
            });

            // Session memory güncelle
            sessionContext.messages.push({
                role: 'user',
                content: message,
                timestamp: new Date().toISOString()
            });
            sessionContext.messages.push({
                role: 'assistant', 
                content: result.output,
                timestamp: new Date().toISOString()
            });

            this.sessionMemory.set(sessionId, sessionContext);

            return {
                content: result.output,
                metadata: {
                    sessionId,
                    toolsUsed: result.intermediateSteps?.map(step => step.action?.tool) || [],
                    timestamp: new Date().toISOString()
                }
            };

        } catch (error) {
            console.error('Mesaj işleme hatası:', error);
            throw error;
        }
    }

    async analyzeESGScore({ companyName, timeRange, metrics }) {
        const tool = this.tools.find(t => t.name === 'esg_analyzer');
        
        const result = await tool.call({
            company: companyName,
            timeRange: timeRange,
            metrics: metrics || ['environment_score', 'social_score', 'governance_score']
        });

        return JSON.parse(result);
    }

    async generateStrategy({ companyData, benchmarkData, goals }) {
        const tool = this.tools.find(t => t.name === 'strategy_generator');
        
        const result = await tool.call({
            companyData,
            benchmarkData, 
            goals
        });

        return JSON.parse(result);
    }

    // Fine-tuning durumunu kontrol et
    async getModelInfo() {
        return {
            currentModel: this.customModelPath && await this.checkCustomModel() 
                ? 'Custom Fine-tuned Model' 
                : 'Base HuggingFace Model',
            modelPath: this.customModelPath || this.modelPath,
            isFineTuned: Boolean(this.customModelPath && await this.checkCustomModel()),
            toolsAvailable: this.tools.map(t => t.name),
            isReady: this.isReady
        };
    }

    // Session temizle
    clearSession(sessionId) {
        this.sessionMemory.delete(sessionId);
    }

    // Tüm session'ları temizle
    clearAllSessions() {
        this.sessionMemory.clear();
    }
}