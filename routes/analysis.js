// routes/analysis.js - Analiz endpoint'leri
import express from 'express';

const router = express.Router();

// ESG skor analizi
router.post('/esg-score', async (req, res) => {
    try {
        const { companyName, timeRange, metrics } = req.body;
        
        if (!companyName) {
            return res.status(400).json({
                success: false,
                error: 'Şirket adı gerekli'
            });
        }

        const esgAgent = req.app.locals.esgAgent;
        if (!esgAgent || !esgAgent.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ESG Agent henüz hazır değil'
            });
        }

        const analysis = await esgAgent.analyzeESGScore({
            companyName,
            timeRange: timeRange || '2020-2023',
            metrics: metrics || ['environment_score', 'social_score', 'governance_score', 'overall_esg_score']
        });

        res.json({
            success: true,
            analysis,
            metadata: {
                companyName,
                timeRange,
                metrics,
                timestamp: new Date()
            }
        });

    } catch (error) {
        console.error('ESG analiz hatası:', error);
        res.status(500).json({
            success: false,
            error: 'ESG analizi yapılırken hata oluştu',
            details: error.message
        });
    }
});

// Strateji önerisi
router.post('/strategy', async (req, res) => {
    try {
        const { companyData, benchmarkData, goals, timeframe } = req.body;
        
        if (!companyData || !goals) {
            return res.status(400).json({
                success: false,
                error: 'Şirket verisi ve hedefler gerekli'
            });
        }

        const esgAgent = req.app.locals.esgAgent;
        if (!esgAgent || !esgAgent.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ESG Agent henüz hazır değil'
            });
        }

        const strategy = await esgAgent.generateStrategy({
            companyData,
            benchmarkData,
            goals,
            timeframe: timeframe || '12 months'
        });

        res.json({
            success: true,
            strategy,
            metadata: {
                goals,
                timeframe,
                timestamp: new Date()
            }
        });

    } catch (error) {
        console.error('Strateji üretim hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Strateji önerisi üretilirken hata oluştu',
            details: error.message
        });
    }
});

// Sektör karşılaştırması
router.post('/sector-comparison', async (req, res) => {
    try {
        const { sector, metric, timeRange } = req.body;
        
        if (!sector || !metric) {
            return res.status(400).json({
                success: false,
                error: 'Sektör ve metrik gerekli'
            });
        }

        const chromaManager = req.app.locals.chromaManager;
        if (!chromaManager || !chromaManager.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ChromaDB henüz hazır değil'
            });
        }

        const sectorResults = await chromaManager.searchBySector(sector, {
            nResults: 50,
            whereFilter: {
                date: { $contains: timeRange || '2023' }
            }
        });

        // Sektör analizi
        const scores = sectorResults.results
            .map(r => parseFloat(r.metadata[metric]))
            .filter(score => !isNaN(score));

        const analysis = {
            sector,
            metric,
            timeRange: timeRange || '2023',
            averageScore: scores.length > 0 
                ? (scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(2)
                : 0,
            totalCompanies: scores.length,
            topPerformers: sectorResults.results
                .filter(r => !isNaN(parseFloat(r.metadata[metric])))
                .sort((a, b) => parseFloat(b.metadata[metric]) - parseFloat(a.metadata[metric]))
                .slice(0, 10)
                .map(r => ({
                    company: r.metadata.company_name,
                    score: parseFloat(r.metadata[metric]),
                    industry: r.metadata.industry
                }))
        };

        res.json({
            success: true,
            analysis
        });

    } catch (error) {
        console.error('Sektör karşılaştırma hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Sektör karşılaştırması yapılırken hata oluştu',
            details: error.message
        });
    }
});

// Trend analizi
router.post('/trend-analysis', async (req, res) => {
    try {
        const { companyName, metric, years } = req.body;
        
        if (!companyName || !metric) {
            return res.status(400).json({
                success: false,
                error: 'Şirket adı ve metrik gerekli'
            });
        }

        const chromaManager = req.app.locals.chromaManager;
        if (!chromaManager || !chromaManager.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ChromaDB henüz hazır değil'
            });
        }

        const timeRange = years || ['2020', '2021', '2022', '2023'];
        const trends = [];

        for (const year of timeRange) {
            const results = await chromaManager.searchByCompany(companyName, {
                nResults: 10,
                whereFilter: {
                    date: { $contains: year }
                }
            });

            if (results.results.length > 0) {
                const score = parseFloat(results.results[0].metadata[metric]);
                if (!isNaN(score)) {
                    trends.push({
                        year,
                        score,
                        dataPoint: 1
                    });
                }
            }
        }

        // Trend hesaplama
        const scores = trends.map(t => t.score);
        const trendAnalysis = {
            companyName,
            metric,
            trends,
            trendDirection: scores.length > 1 
                ? (scores[scores.length - 1] > scores[0] ? 'increasing' : 'decreasing')
                : 'stable',
            trendPercentage: scores.length > 1 
                ? ((scores[scores.length - 1] - scores[0]) / scores[0] * 100).toFixed(2)
                : 0,
            startScore: scores[0] || 0,
            endScore: scores[scores.length - 1] || 0
        };

        res.json({
            success: true,
            trendAnalysis
        });

    } catch (error) {
        console.error('Trend analizi hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Trend analizi yapılırken hata oluştu',
            details: error.message
        });
    }
});

// Benchmark analizi
router.post('/benchmark', async (req, res) => {
    try {
        const { companyName, benchmarkCompanies, metrics } = req.body;
        
        if (!companyName || !benchmarkCompanies) {
            return res.status(400).json({
                success: false,
                error: 'Şirket adı ve benchmark şirketleri gerekli'
            });
        }

        const chromaManager = req.app.locals.chromaManager;
        if (!chromaManager || !chromaManager.isReady) {
            return res.status(503).json({
                success: false,
                error: 'ChromaDB henüz hazır değil'
            });
        }

        const benchmarkMetrics = metrics || ['environment_score', 'social_score', 'governance_score', 'overall_esg_score'];
        const benchmarkResults = {};

        // Ana şirket verilerini al
        const companyResults = await chromaManager.searchByCompany(companyName, {
            nResults: 5,
            whereFilter: {
                date: { $contains: '2023' }
            }
        });

        if (companyResults.results.length > 0) {
            const companyData = companyResults.results[0].metadata;
            
            for (const metric of benchmarkMetrics) {
                const companyScore = parseFloat(companyData[metric]);
                
                if (!isNaN(companyScore)) {
                    // Benchmark şirketlerin skorlarını al
                    const benchmarkScores = [];
                    
                    for (const benchmarkCompany of benchmarkCompanies) {
                        const benchmarkResults = await chromaManager.searchByCompany(benchmarkCompany, {
                            nResults: 5,
                            whereFilter: {
                                date: { $contains: '2023' }
                            }
                        });

                        if (benchmarkResults.results.length > 0) {
                            const score = parseFloat(benchmarkResults.results[0].metadata[metric]);
                            if (!isNaN(score)) {
                                benchmarkScores.push({
                                    company: benchmarkCompany,
                                    score
                                });
                            }
                        }
                    }

                    const avgBenchmarkScore = benchmarkScores.length > 0
                        ? benchmarkScores.reduce((sum, b) => sum + b.score, 0) / benchmarkScores.length
                        : 0;

                    benchmarkResults[metric] = {
                        companyScore,
                        benchmarkScores,
                        averageBenchmarkScore: avgBenchmarkScore.toFixed(2),
                        performance: companyScore > avgBenchmarkScore ? 'above' : 'below',
                        difference: (companyScore - avgBenchmarkScore).toFixed(2)
                    };
                }
            }
        }

        res.json({
            success: true,
            benchmark: {
                companyName,
                benchmarkCompanies,
                metrics: benchmarkMetrics,
                results: benchmarkResults,
                timestamp: new Date()
            }
        });

    } catch (error) {
        console.error('Benchmark analizi hatası:', error);
        res.status(500).json({
            success: false,
            error: 'Benchmark analizi yapılırken hata oluştu',
            details: error.message
        });
    }
});

// Analiz istatistikleri
router.get('/stats', (req, res) => {
    const chromaManager = req.app.locals.chromaManager;
    
    if (!chromaManager || !chromaManager.isReady) {
        return res.status(503).json({
            success: false,
            error: 'ChromaDB henüz hazır değil'
        });
    }

    chromaManager.getCollectionStats()
        .then(stats => {
            res.json({
                success: true,
                stats: {
                    totalDocuments: stats.documentCount,
                    collectionName: stats.collectionName,
                    isReady: stats.isReady,
                    lastUpdated: new Date()
                }
            });
        })
        .catch(error => {
            res.status(500).json({
                success: false,
                error: 'İstatistikler alınırken hata oluştu',
                details: error.message
            });
        });
});

export default router; 