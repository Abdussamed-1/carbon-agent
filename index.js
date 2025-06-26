// index.js - Ana server
import express from 'express';
import dotenv from 'dotenv';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import winston from 'winston';
import { ESGAgent } from './agents/esgAgent.js';
import { ChromaManager } from './vectorstore/chromaManager.js';
import { MODEL_CONFIG, API_CONFIG, LOGGING_CONFIG, validateModelConfig } from './config/models.js';
import chatRoutes from './routes/chat.js';
import analysisRoutes from './routes/analysis.js';
import marketRoutes from './routes/market.js';
import { getLLM, getEmbeddings } from "./llm/llmFactory.js";
import { getChromaClient } from "./vectorstore/chromaFactory.js";
import { createDynamicAgent } from "./agents/dynamicAgent.js";

dotenv.config();

const app = express();
const PORT = API_CONFIG.port;

// Logger konfigürasyonu
const logger = winston.createLogger({
    level: LOGGING_CONFIG.level,
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ 
            filename: LOGGING_CONFIG.filename,
            maxsize: LOGGING_CONFIG.maxSize,
            maxFiles: LOGGING_CONFIG.maxFiles
        }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Middleware
app.use(helmet());
app.use(cors(API_CONFIG.cors));
app.use(express.json({ limit: '10mb' }));
app.use(express.static('public'));

// Rate limiting
const limiter = rateLimit(API_CONFIG.rateLimit);
app.use('/api/', limiter);

// Global değişkenler
let esgAgent;
let chromaManager;
let agentExecutor;

// Uygulama başlatma
async function initializeApp() {
    try {
        logger.info('🚀 ESG RAG Agent başlatılıyor...');
        
        // Model konfigürasyonunu doğrula
        validateModelConfig();
        
        // LLM ve Embedding
        const config = {
            LLM_TYPE: process.env.LLM_TYPE || "granite",
            GRANITE_MODEL_PATH: process.env.GRANITE_MODEL_PATH,
            HF_TOKEN: process.env.HF_TOKEN,
            OPENAI_API_KEY: process.env.OPENAI_API_KEY,
            EMBEDDING_MODEL: process.env.EMBEDDING_MODEL || "sentence-transformers/all-MiniLM-L6-v2",
            CHROMA_DB_PATH: process.env.CHROMA_DB_PATH || "http://localhost:8000"
        };
        const llm = getLLM(config);
        const embeddings = getEmbeddings(config);

        // ChromaDB - Düzeltilmiş kısım
        const chromaClient = getChromaClient(config);
        chromaManager = new ChromaManager({ 
            client: chromaClient, 
            collectionName: "esg_documents",
            path: config.CHROMA_DB_PATH 
        });
        
        // ChromaDB başlatma
        await chromaManager.initialize();
        
        // Global değişkenleri route'larda kullanılabilir yap
        app.locals.chromaManager = chromaManager;
        
        // Agent
        agentExecutor = await createDynamicAgent({ llm, chromaManager, embeddings });
        
        // ESG Agent oluştur
        esgAgent = new ESGAgent({
            modelPath: MODEL_CONFIG.GRANITE.baseModel,
            customModelPath: MODEL_CONFIG.GRANITE.customModelPath,
            chromaManager: chromaManager,
            config: MODEL_CONFIG
        });
        await esgAgent.initialize();
        
        // ESG Agent'ı route'larda kullanılabilir yap
        app.locals.esgAgent = esgAgent;
        app.locals.agentExecutor = agentExecutor;
        
        logger.info('✅ ESG RAG Agent hazır!');
        
    } catch (error) {
        logger.error('❌ Başlatma hatası:', error);
        process.exit(1);
    }
}

// Ana sayfa
app.get('/', (req, res) => {
    res.json({
        name: 'ESG RAG Agent API',
        version: '1.0.0',
        status: 'running',
        timestamp: new Date(),
        services: {
            esgAgent: esgAgent?.isReady ? 'ready' : 'loading',
            chromaDB: chromaManager?.isReady ? 'connected' : 'connecting'
        },
        endpoints: {
            health: '/health',
            docs: '/api/docs',
            web: '/web',
            chat: '/api/chat/*',
            analysis: '/api/analysis/*',
            market: '/api/market/*',
            ask: '/api/ask'
        },
        usage: {
            example_chat: 'POST /api/ask with { "message": "your question" }',
            health_check: 'GET /health',
            documentation: 'GET /api/docs',
            web_interface: 'GET /web'
        }
    });
});

// Basit web arayüzü
app.get('/web', (req, res) => {
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>ESG RAG Agent</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 900px; margin: 0 auto; }
            .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; text-align: center; }
            .status { background: #27ae60; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .endpoint { background: #fff; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .method { font-weight: bold; color: #3498db; }
            .path { font-family: monospace; background: #ecf0f1; padding: 2px 6px; border-radius: 3px; }
            .test-area { margin: 20px 0; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            input, textarea, button { margin: 5px; padding: 10px; border-radius: 4px; border: 1px solid #ddd; }
            textarea { width: 100%; box-sizing: border-box; font-family: Arial, sans-serif; }
            button { background: #3498db; color: white; border: none; cursor: pointer; font-weight: bold; }
            button:hover { background: #2980b9; }
            button:disabled { background: #bdc3c7; cursor: not-allowed; }
            #response { background: #2c3e50; color: #ecf0f1; border: 1px solid #34495e; padding: 15px; border-radius: 4px; white-space: pre-wrap; font-family: monospace; max-height: 400px; overflow-y: auto; }
            .loading { color: #f39c12; }
            .success { color: #27ae60; }
            .error { color: #e74c3c; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🌱 ESG RAG Agent</h1>
                <p>AI-powered ESG analysis and RAG system with IBM Granite</p>
            </div>

            <div class="status">
                <strong>Status:</strong> ${esgAgent?.isReady ? '✅ Ready' : '⏳ Loading'} | 
                <strong>ChromaDB:</strong> ${chromaManager?.isReady ? '✅ Connected' : '⏳ Connecting'}
            </div>

            <div class="test-area">
                <h3>🧪 Test the Agent</h3>
                <p>ESG ile ilgili sorularınızı sorun. Örneğin:</p>
                <ul>
                    <li>"Tesla'nın ESG performansı nasıl?"</li>
                    <li>"Teknoloji sektöründe ESG trendleri neler?"</li>
                    <li>"En yüksek çevre skoruna sahip şirketler hangileri?"</li>
                </ul>
                
                <textarea id="messageInput" placeholder="ESG sorunuzu buraya yazın..." rows="4"></textarea><br>
                <button onclick="sendMessage()" id="sendBtn">Gönder</button>
                <button onclick="clearResponse()">Temizle</button>
                
                <h4>Response:</h4>
                <div id="response">Henüz sorgu gönderilmedi... (Ctrl+Enter ile de gönderebilirsiniz)</div>
            </div>

            <h3>📚 Available Endpoints</h3>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/health</span>
                <p>Sistem durumu kontrolü</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/ask</span>
                <p>ESG soruları sormak için ana endpoint</p>
                <code>{ "message": "your question here" }</code>
            </div>

            <div class="endpoint">
                <span class="method">GET</span> <span class="path">/api/docs</span>
                <p>Detaylı API dokümantasyonu</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/chat/*</span>
                <p>Chat session yönetimi</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/analysis/*</span>
                <p>ESG analiz endpoint'leri</p>
            </div>

            <div class="endpoint">
                <span class="method">POST</span> <span class="path">/api/market/*</span>
                <p>Pazar analizi endpoint'leri</p>
            </div>
        </div>

        <script>
            async function sendMessage() {
                const message = document.getElementById('messageInput').value;
                const responseDiv = document.getElementById('response');
                const sendBtn = document.getElementById('sendBtn');
                
                if (!message.trim()) {
                    alert('Lütfen bir mesaj girin');
                    return;
                }
                
                sendBtn.disabled = true;
                sendBtn.textContent = 'Gönderiliyor...';
                responseDiv.innerHTML = '<span class="loading">🤖 Agent düşünüyor...</span>';
                
                try {
                    const response = await fetch('/api/ask', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ message })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = '<span class="success">✅ Response:</span>\\n' + JSON.stringify(data, null, 2);
                    } else {
                        responseDiv.innerHTML = '<span class="error">❌ Error:</span>\\n' + JSON.stringify(data, null, 2);
                    }
                } catch (error) {
                    responseDiv.innerHTML = '<span class="error">❌ Network Error:</span>\\n' + error.message;
                } finally {
                    sendBtn.disabled = false;
                    sendBtn.textContent = 'Gönder';
                }
            }
            
            function clearResponse() {
                document.getElementById('response').innerHTML = 'Temizlendi...';
                document.getElementById('messageInput').value = '';
            }
            
            // Enter tuşu ile gönder
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && e.ctrlKey) {
                    sendMessage();
                }
            });
        </script>
    </body>
    </html>
    `);
});

// Routes
app.use('/api/chat', chatRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/market', marketRoutes);

// API endpoint
app.post("/api/ask", async (req, res) => {
    const { message } = req.body;
    
    if (!message) {
        return res.status(400).json({ 
            success: false, 
            error: 'Message is required',
            usage: 'POST /api/ask with { "message": "your question" }'
        });
    }

    try {
        if (!agentExecutor) {
            return res.status(503).json({ 
                success: false, 
                error: 'Agent is not ready yet',
                status: 'Agent is still initializing. Please try again in a few seconds.'
            });
        }

        logger.info(`📝 User question: ${message}`);
        const response = await agentExecutor.invoke({ input: message });
        
        logger.info(`🤖 Agent response generated`);
        res.json({ 
            success: true,
            response,
            timestamp: new Date(),
            agent_status: 'ready'
        });
    } catch (err) {
        logger.error(`❌ Agent error: ${err.message}`);
        res.status(500).json({ 
            success: false,
            error: err.message,
            timestamp: new Date()
        });
    }
});

// Sağlık kontrolü
app.get('/health', async (req, res) => {
    const health = {
        status: 'ok',
        timestamp: new Date(),
        services: {
            esgAgent: esgAgent?.isReady ? 'ready' : 'loading',
            chromaDB: chromaManager?.isReady ? 'connected' : 'connecting',
            model: esgAgent?.isReady ? 'loaded' : 'loading',
            agentExecutor: agentExecutor ? 'ready' : 'loading'
        },
        config: {
            model: MODEL_CONFIG.GRANITE.baseModel,
            port: PORT,
            environment: process.env.NODE_ENV || 'development',
            chromaPath: process.env.CHROMA_DB_PATH || 'http://localhost:8000'
        }
    };

    // ChromaDB stats ekle
    if (chromaManager?.isReady) {
        try {
            const stats = await chromaManager.getCollectionStats();
            health.chromaStats = stats;
        } catch (error) {
            health.chromaStats = { error: error.message };
        }
    }

    res.json(health);
});

// API dokümantasyonu
app.get('/api/docs', (req, res) => {
    res.json({
        name: 'ESG RAG Agent API',
        version: '1.0.0',
        description: 'AI-powered ESG analysis and RAG system with IBM Granite',
        baseUrl: `http://localhost:${PORT}`,
        endpoints: {
            main: {
                'GET /': 'API ana bilgileri',
                'GET /web': 'Web test arayüzü',
                'GET /health': 'Sistem durumu',
                'POST /api/ask': 'Ana chat endpoint'
            },
            chat: {
                'POST /api/chat/session': 'Chat session başlat',
                'POST /api/chat/message': 'Mesaj gönder',
                'GET /api/chat/session/:id/history': 'Session geçmişi',
                'DELETE /api/chat/session/:id': 'Session sil',
                'GET /api/chat/sessions': 'Aktif session\'lar',
                'POST /api/chat/stream': 'Streaming chat',
                'GET /api/chat/stats': 'Chat istatistikleri'
            },
            analysis: {
                'POST /api/analysis/esg-score': 'ESG skor analizi',
                'POST /api/analysis/strategy': 'Strateji önerisi',
                'POST /api/analysis/sector-comparison': 'Sektör karşılaştırması',
                'POST /api/analysis/trend-analysis': 'Trend analizi',
                'POST /api/analysis/benchmark': 'Benchmark analizi',
                'GET /api/analysis/stats': 'Analiz istatistikleri'
            },
            market: {
                'POST /api/market/sector-analysis': 'Sektör analizi',
                'POST /api/market/sector-comparison': 'Sektör karşılaştırması',
                'POST /api/market/trend-analysis': 'Trend analizi'
            }
        },
        examples: {
            askQuestion: {
                method: 'POST',
                url: '/api/ask',
                body: { message: "Tesla'nın ESG performansı nasıl?" }
            },
            healthCheck: {
                method: 'GET',
                url: '/health'
            }
        }
    });
});

// Hata yakalama middleware
app.use((error, req, res, next) => {
    logger.error('Genel hata:', error);
    res.status(500).json({
        success: false,
        error: 'Sunucu hatası',
        timestamp: new Date(),
        path: req.path
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        error: 'Endpoint bulunamadı',
        path: req.originalUrl,
        method: req.method,
        availableEndpoints: [
            'GET /',
            'GET /web',
            'GET /health',
            'GET /api/docs',
            'POST /api/ask'
        ]
    });
});

// Sunucu başlat
async function startServer() {
    await initializeApp();
    
    app.listen(PORT, () => {
        logger.info(`🌟 Server ${PORT} portunda çalışıyor`);
        logger.info(`📊 ESG Agent: http://localhost:${PORT}`);
        logger.info(`🌐 Web Interface: http://localhost:${PORT}/web`);
        logger.info(`📚 API Docs: http://localhost:${PORT}/api/docs`);
        logger.info(`🏥 Health Check: http://localhost:${PORT}/health`);
    });
}

// Graceful shutdown
process.on('SIGTERM', () => {
    logger.info('SIGTERM sinyali alındı, sunucu kapatılıyor...');
    process.exit(0);
});

process.on('SIGINT', () => {
    logger.info('SIGINT sinyali alındı, sunucu kapatılıyor...');
    process.exit(0);
});

startServer().catch(error => {
    logger.error('Sunucu başlatma hatası:', error);
    process.exit(1);
});

export { esgAgent, chromaManager, agentExecutor, logger };