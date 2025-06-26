// routes/chat.js - Chat endpoint'leri
import express from 'express';
import { v4 as uuidv4 } from 'uuid';

const router = express.Router();

// Session yönetimi için basit in-memory store
const sessions = new Map();

// Chat session başlat
router.post('/session', (req, res) => {
    const sessionId = uuidv4();
    const session = {
        id: sessionId,
        createdAt: new Date(),
        messages: [],
        metadata: {
            userAgent: req.headers['user-agent'],
            ip: req.ip
        }
    };

    sessions.set(sessionId, session);

    res.json({
        success: true,
        sessionId,
        message: 'Chat session başlatıldı'
    });
});

// Ana chat endpoint
router.post('/message', async (req, res) => {
    try {
        const { message, sessionId, context } = req.body;
        
        if (!message) {
            return res.status(400).json({
                success: false,
                error: 'Mesaj gerekli'
            });
        }

        // Session kontrolü
        let session = sessions.get(sessionId);
        if (!session) {
            sessionId = uuidv4();
            session = {
                id: sessionId,
                createdAt: new Date(),
                messages: [],
                metadata: {
                    userAgent: req.headers['user-agent'],
                    ip: req.ip
                }
            };
            sessions.set(sessionId, session);
        }

        // ESG Agent'dan yanıt al
        const esgAgent = req.app.locals.esgAgent;
        if (!esgAgent || !esgAgent.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ESG Agent henüz hazır değil'
            });
        }

        const startTime = Date.now();
        
        const response = await esgAgent.processMessage({
            message,
            sessionId,
            context: context || {}
        });

        const responseTime = Date.now() - startTime;

        // Mesajı session'a kaydet
        session.messages.push({
            id: uuidv4(),
            timestamp: new Date(),
            type: 'user',
            content: message,
            metadata: { responseTime: 0 }
        });

        session.messages.push({
            id: uuidv4(),
            timestamp: new Date(),
            type: 'assistant',
            content: response.content,
            metadata: { 
                responseTime,
                tools: response.metadata?.tools || [],
                sources: response.metadata?.sources || []
            }
        });

        // Session'ı güncelle
        sessions.set(sessionId, session);

        res.json({
            success: true,
            sessionId,
            response: {
                content: response.content,
                responseTime,
                metadata: response.metadata
            }
        });

    } catch (error) {
        console.error('Chat mesaj hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Mesaj işlenirken hata oluştu',
            details: error.message
        });
    }
});

// Session geçmişini getir
router.get('/session/:sessionId/history', (req, res) => {
    const { sessionId } = req.params;
    const session = sessions.get(sessionId);

    if (!session) {
        return res.status(404).json({
            success: false,
            error: 'Session bulunamadı'
        });
    }

    res.json({
        success: true,
        session: {
            id: session.id,
            createdAt: session.createdAt,
            messageCount: session.messages.length,
            messages: session.messages.slice(-50) // Son 50 mesaj
        }
    });
});

// Session'ı temizle
router.delete('/session/:sessionId', (req, res) => {
    const { sessionId } = req.params;
    const deleted = sessions.delete(sessionId);

    res.json({
        success: true,
        deleted,
        message: deleted ? 'Session silindi' : 'Session bulunamadı'
    });
});

// Aktif session'ları listele (admin)
router.get('/sessions', (req, res) => {
    const sessionList = Array.from(sessions.values()).map(session => ({
        id: session.id,
        createdAt: session.createdAt,
        messageCount: session.messages.length,
        lastActivity: session.messages.length > 0 
            ? session.messages[session.messages.length - 1].timestamp 
            : session.createdAt
    }));

    res.json({
        success: true,
        sessions: sessionList,
        totalSessions: sessionList.length
    });
});

// Streaming chat endpoint (WebSocket benzeri)
router.post('/stream', async (req, res) => {
    const { message, sessionId } = req.body;

    if (!message) {
        return res.status(400).json({
            success: false,
            error: 'Mesaj gerekli'
        });
    }

    // SSE headers
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    });

    try {
        const esgAgent = req.app.locals.esgAgent;
        
        // Streaming response (basit implementasyon)
        res.write(`data: ${JSON.stringify({
            type: 'start',
            message: 'Yanıt hazırlanıyor...'
        })}\n\n`);

        const response = await esgAgent.processMessage({
            message,
            sessionId,
            stream: true
        });

        // Yanıtı parçalara böl ve gönder
        const chunks = response.content.split(' ');
        
        for (const chunk of chunks) {
            res.write(`data: ${JSON.stringify({
                type: 'chunk',
                content: chunk + ' '
            })}\n\n`);
            
            // Küçük gecikme ekle
            await new Promise(resolve => setTimeout(resolve, 50));
        }

        res.write(`data: ${JSON.stringify({
            type: 'end',
            metadata: response.metadata
        })}\n\n`);

    } catch (error) {
        res.write(`data: ${JSON.stringify({
            type: 'error',
            error: error.message
        })}\n\n`);
    } finally {
        res.end();
    }
});

// Chat istatistikleri
router.get('/stats', (req, res) => {
    const totalSessions = sessions.size;
    const totalMessages = Array.from(sessions.values())
        .reduce((sum, session) => sum + session.messages.length, 0);
    
    const avgMessagesPerSession = totalSessions > 0 
        ? (totalMessages / totalSessions).toFixed(2) 
        : 0;

    res.json({
        success: true,
        stats: {
            totalSessions,
            totalMessages,
            avgMessagesPerSession,
            activeSessions: totalSessions
        }
    });
});

export default router; 