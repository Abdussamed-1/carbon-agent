import express from 'express';
import { MarketAnalyzerTool } from '../agents/tools/marketAnalyzer.js';

const router = express.Router();

// Pazar analizi endpoint
router.post('/sector-analysis', async (req, res) => {
    try {
        const { sector, metric, timeRange } = req.body;
        
        if (!sector || !metric) {
            return res.status(400).json({
                success: false,
                error: 'Sector ve metric parametreleri gerekli'
            });
        }

        // Market analyzer tool'u oluştur
        const marketTool = new MarketAnalyzerTool({ 
            chromaManager: req.app.locals.chromaManager 
        });

        const result = await marketTool.call(JSON.stringify({
            sector,
            metric,
            timeRange: timeRange || '2023'
        }));

        const analysis = JSON.parse(result);

        res.json({
            success: true,
            analysis
        });

    } catch (error) {
        console.error('Pazar analizi hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Pazar analizi yapılırken hata oluştu'
        });
    }
});

// Sektör karşılaştırması endpoint
router.post('/sector-comparison', async (req, res) => {
    try {
        const { sectors, metric, timeRange } = req.body;
        
        if (!sectors || !Array.isArray(sectors) || sectors.length < 2) {
            return res.status(400).json({
                success: false,
                error: 'En az 2 sektör belirtilmeli'
            });
        }

        const marketTool = new MarketAnalyzerTool({ 
            chromaManager: req.app.locals.chromaManager 
        });

        const comparisons = [];

        for (const sector of sectors) {
            const result = await marketTool.call(JSON.stringify({
                sector,
                metric: metric || 'overall_esg_score',
                timeRange: timeRange || '2023'
            }));

            const analysis = JSON.parse(result);
            comparisons.push({
                sector,
                averageScore: analysis.averageScore,
                topPerformers: analysis.topPerformers.slice(0, 3)
            });
        }

        // Sektörleri skorlarına göre sırala
        comparisons.sort((a, b) => parseFloat(b.averageScore) - parseFloat(a.averageScore));

        res.json({
            success: true,
            comparisons,
            bestPerformingSector: comparisons[0],
            worstPerformingSector: comparisons[comparisons.length - 1]
        });

    } catch (error) {
        console.error('Sektör karşılaştırma hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Sektör karşılaştırması yapılırken hata oluştu'
        });
    }
});

// Trend analizi endpoint
router.post('/trend-analysis', async (req, res) => {
    try {
        const { sector, metric, years } = req.body;
        
        if (!sector || !metric) {
            return res.status(400).json({
                success: false,
                error: 'Sector ve metric parametreleri gerekli'
            });
        }

        const timeRange = years || ['2020', '2021', '2022', '2023'];
        const marketTool = new MarketAnalyzerTool({ 
            chromaManager: req.app.locals.chromaManager 
        });

        const trends = [];

        for (const year of timeRange) {
            const result = await marketTool.call(JSON.stringify({
                sector,
                metric,
                timeRange: year
            }));

            const analysis = JSON.parse(result);
            trends.push({
                year,
                averageScore: analysis.averageScore,
                totalCompanies: analysis.totalCompanies
            });
        }

        // Trend hesaplama
        const scores = trends.map(t => parseFloat(t.averageScore));
        const trendDirection = scores[scores.length - 1] > scores[0] ? 'increasing' : 'decreasing';
        const trendPercentage = ((scores[scores.length - 1] - scores[0]) / scores[0] * 100).toFixed(2);

        res.json({
            success: true,
            sector,
            metric,
            trends,
            trendAnalysis: {
                direction: trendDirection,
                percentage: trendPercentage,
                startScore: scores[0],
                endScore: scores[scores.length - 1]
            }
        });

    } catch (error) {
        console.error('Trend analizi hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Trend analizi yapılırken hata oluştu'
        });
    }
});

export default router; 