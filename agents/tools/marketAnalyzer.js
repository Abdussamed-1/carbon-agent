import { Tool } from "langchain/tools";

export class MarketAnalyzerTool extends Tool {
    constructor({ chromaManager }) {
        super();
        this.name = "market_analyzer";
        this.description = `Pazar analizi ve sektör karşılaştırması yapar.
        Input: JSON string with {sector, metric, timeRange}
        Örnek: {"sector": "Technology", "metric": "environment_score", "timeRange": "2023"}`;
        this.chromaManager = chromaManager;
    }

    async _call(input) {
        try {
            const params = JSON.parse(input);
            const { sector, metric, timeRange } = params;

            // Sektör bazlı arama
            const sectorResults = await this.chromaManager.searchBySector(sector, {
                nResults: 20,
                whereFilter: {
                    date: { $contains: timeRange }
                }
            });

            // Metrik analizi
            const analysis = this.analyzeSectorMetrics(sectorResults.results, metric);

            return JSON.stringify({
                sector,
                metric,
                timeRange,
                analysis,
                topPerformers: analysis.topPerformers.slice(0, 5),
                averageScore: analysis.averageScore,
                recommendations: analysis.recommendations
            });

        } catch (error) {
            return JSON.stringify({
                error: `Pazar analizi hatası: ${error.message}`,
                suggestion: "Parametreleri kontrol edin"
            });
        }
    }

    analyzeSectorMetrics(results, metric) {
        const scores = results
            .map(r => parseFloat(r.metadata[metric]))
            .filter(score => !isNaN(score));

        const averageScore = scores.length > 0 
            ? scores.reduce((a, b) => a + b, 0) / scores.length 
            : 0;

        const topPerformers = results
            .filter(r => !isNaN(parseFloat(r.metadata[metric])))
            .sort((a, b) => parseFloat(b.metadata[metric]) - parseFloat(a.metadata[metric]))
            .slice(0, 10)
            .map(r => ({
                company: r.metadata.company_name,
                score: parseFloat(r.metadata[metric]),
                sector: r.metadata.sector
            }));

        const recommendations = this.generateRecommendations(averageScore, metric);

        return {
            averageScore: averageScore.toFixed(2),
            topPerformers,
            recommendations,
            totalCompanies: scores.length
        };
    }

    generateRecommendations(averageScore, metric) {
        const recommendations = [];
        
        if (metric === 'environment_score') {
            if (averageScore < 50) {
                recommendations.push("Sektörde çevresel performans düşük. Karbon azaltma stratejileri öncelikli olmalı.");
            } else if (averageScore < 70) {
                recommendations.push("Çevresel performans orta seviyede. Yenilenebilir enerji geçişi hızlandırılmalı.");
            } else {
                recommendations.push("Çevresel performans yüksek. Sürdürülebilirlik liderliği için daha da geliştirilebilir.");
            }
        }

        if (metric === 'social_score') {
            if (averageScore < 50) {
                recommendations.push("Sosyal sorumluluk skorları düşük. Çalışan hakları ve toplumsal katkı artırılmalı.");
            } else if (averageScore < 70) {
                recommendations.push("Sosyal performans orta seviyede. Paydaş ilişkileri güçlendirilmeli.");
            } else {
                recommendations.push("Sosyal performans yüksek. Toplumsal etki projeleri genişletilebilir.");
            }
        }

        return recommendations;
    }
} 