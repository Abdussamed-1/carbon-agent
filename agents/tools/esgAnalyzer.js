// agents/tools/esgAnalyzer.js
import { Tool } from "langchain/tools";

export class ESGAnalyzerTool extends Tool {
    constructor({ chromaManager }) {
        super();
        this.name = "esg_analyzer";
        this.description = `ESG skorlarını analiz eder ve trend analizi yapar. 
        Input: JSON string with {company, timeRange, metrics}
        Örnek: {"company": "Apple", "timeRange": "2020-2023", "metrics": ["environment_score", "social_score"]}`;
        this.chromaManager = chromaManager;
    }

    async _call(input) {
        try {
            const params = JSON.parse(input);
            const { company, timeRange, metrics } = params;

            // ChromaDB'den şirket verilerini al
            const searchResults = await this.chromaManager.searchByCompany(company, {
                nResults: 50
            });

            // Veriyi analiz et
            const analysis = this.analyzeESGData(searchResults.results, metrics, timeRange);
            
            return JSON.stringify({
                company,
                timeRange,
                analysis,
                timestamp: new Date().toISOString()
            });

        } catch (error) {
            return JSON.stringify({
                error: `ESG analiz hatası: ${error.message}`,
                suggestion: "Şirket adını ve zaman aralığını kontrol edin"
            });
        }
    }

    analyzeESGData(results, metrics, timeRange) {
        if (!results || results.length === 0) {
            return {
                error: "Bu şirket için ESG verisi bulunamadı",
                suggestion: "Farklı bir şirket adı deneyin"
            };
        }

        const data = results.map(r => r.metadata).filter(m => m.record_type === 'esg_data');
        
        // Trend analizi
        const trends = this.calculateTrends(data, metrics);
        
        // Sektör karşılaştırması
        const sectorComparison = this.calculateSectorComparison(data);
        
        // Risk analizi
        const riskAssessment = this.assessRisks(data);

        return {
            dataPoints: data.length,
            trends,
            sectorComparison,
            riskAssessment,
            recommendations: this.generateRecommendations(trends, riskAssessment)
        };
    }

    calculateTrends(data, metrics) {
        const trends = {};
        
        metrics.forEach(metric => {
            const values = data.map(d => d[metric]).filter(v => v !== null && v !== undefined);
            
            if (values.length > 1) {
                const firstValue = values[0];
                const lastValue = values[values.length - 1];
                const change = lastValue - firstValue;
                const changePercent = (change / firstValue) * 100;
                
                trends[metric] = {
                    current: lastValue,
                    change: change.toFixed(2),
                    changePercent: changePercent.toFixed(2),
                    trend: change > 0 ? 'yükseliş' : change < 0 ? 'düşüş' : 'sabit',
                    average: (values.reduce((a, b) => a + b, 0) / values.length).toFixed(2)
                };
            }
        });
        
        return trends;
    }

    calculateSectorComparison(data) {
        const latestData = data[data.length - 1] || data[0];
        
        return {
            sector: latestData?.sector || 'Unknown',
            country: latestData?.country || 'Unknown',
            environmentScore: latestData?.environment_score || 0,
            socialScore: latestData?.social_score || 0,
            governanceScore: latestData?.governance_score || 0,
            overallScore: latestData?.overall_esg_score || 0
        };
    }

    assessRisks(data) {
        const latestData = data[data.length - 1] || data[0];
        const risks = [];
        
        if (latestData?.environment_score < 40) {
            risks.push({
                type: 'Çevresel Risk',
                level: 'Yüksek',
                description: 'Düşük çevresel performans skoru'
            });
        }
        
        if (latestData?.social_score < 40) {
            risks.push({
                type: 'Sosyal Risk',
                level: 'Yüksek', 
                description: 'Düşük sosyal sorumluluk skoru'
            });
        }
        
        if (latestData?.governance_score < 40) {
            risks.push({
                type: 'Yönetişim Riski',
                level: 'Yüksek',
                description: 'Düşük kurumsal yönetim skoru'
            });
        }

        return {
            riskLevel: risks.length > 2 ? 'Yüksek' : risks.length > 0 ? 'Orta' : 'Düşük',
            risks,
            totalRiskCount: risks.length
        };
    }

    generateRecommendations(trends, riskAssessment) {
        const recommendations = [];
        
        // Trend bazlı öneriler
        Object.entries(trends).forEach(([metric, trend]) => {
            if (parseFloat(trend.changePercent) < -10) {
                recommendations.push({
                    area: metric,
                    priority: 'Yüksek',
                    action: `${metric} alanında acil iyileştirme gerekli (${trend.changePercent}% düşüş)`
                });
            }
        });
        
        // Risk bazlı öneriler
        riskAssessment.risks.forEach(risk => {
            recommendations.push({
                area: risk.type,
                priority: risk.level,
                action: `${risk.type} için stratejik plan geliştirin`
            });
        });

        return recommendations;
    }
}

// agents/tools/ragRetriever.js
export class RAGRetrieverTool extends Tool {
    constructor({ chromaManager, embeddings }) {
        super();
        this.name = "rag_retriever";
        this.description = `ESG dokümanlarından bilgi getirir ve bağlamsal yanıtlar sağlar.
        Input: JSON string with {query, filters?, maxResults?}
        Örnek: {"query": "renewable energy transition strategies", "maxResults": 5}`;
        this.chromaManager = chromaManager;
        this.embeddings = embeddings;
    }

    async _call(input) {
        try {
            const params = JSON.parse(input);
            const { query, filters = {}, maxResults = 10 } = params;

            // Semantic search yap
            const searchResults = await this.chromaManager.searchSimilar(query, {
                nResults: maxResults,
                whereFilter: filters
            });

            // Sonuçları işle ve özetle
            const processedResults = this.processSearchResults(searchResults);
            
            return JSON.stringify({
                query,
                results: processedResults,
                summary: this.generateSummary(processedResults),
                sources: this.extractSources(searchResults.results)
            });

        } catch (error) {
            return JSON.stringify({
                error: `RAG retrieval hatası: ${error.message}`,
                suggestion: "Sorgu formatını kontrol edin"
            });
        }
    }

    processSearchResults(searchResults) {
        return searchResults.results.map(result => ({
            content: result.document,
            relevanceScore: result.similarity,
            metadata: {
                company: result.metadata.company_name,
                sector: result.metadata.sector,
                date: result.metadata.date,
                esgScores: {
                    environment: result.metadata.environment_score,
                    social: result.metadata.social_score,
                    governance: result.metadata.governance_score,
                    overall: result.metadata.overall_esg_score
                }
            }
        })).filter(result => result.relevanceScore > 0.7); // Yalnızca yüksek relevanslı sonuçlar
    }

    generateSummary(results) {
        if (results.length === 0) {
            return "İlgili dokümanta bulunamadı.";
        }

        const companies = [...new Set(results.map(r => r.metadata.company))];
        const sectors = [...new Set(results.map(r => r.metadata.sector))];
        const avgRelevance = (results.reduce((sum, r) => sum + r.relevanceScore, 0) / results.length).toFixed(2);

        return {
            resultCount: results.length,
            companies: companies.slice(0, 5), // İlk 5 şirket
            sectors: sectors.slice(0, 3), // İlk 3 sektör
            averageRelevance: avgRelevance,
            keyInsights: this.extractKeyInsights(results)
        };
    }

    extractKeyInsights(results) {
        const insights = [];
        
        // Yüksek performanslı şirketleri bul
        const highPerformers = results.filter(r => r.metadata.esgScores.overall > 70);
        if (highPerformers.length > 0) {
            insights.push(`${highPerformers.length} şirket yüksek ESG performansı gösteriyor`);
        }
        
        // Sektörel dağılım
        const sectorCounts = {};
        results.forEach(r => {
            sectorCounts[r.metadata.sector] = (sectorCounts[r.metadata.sector] || 0) + 1;
        });
        
        const topSector = Object.entries(sectorCounts).sort((a, b) => b[1] - a[1])[0];
        if (topSector) {
            insights.push(`En çok veri ${topSector[0]} sektöründen`);
        }

        return insights;
    }

    extractSources(results) {
        return results.map(result => ({
            company: result.metadata.company_name,
            sector: result.metadata.sector,
            date: result.metadata.date,
            relevance: result.similarity.toFixed(2)
        }));
    }
}

// agents/tools/strategyGenerator.js
export class StrategyGeneratorTool extends Tool {
    constructor({ llm }) {
        super();
        this.name = "strategy_generator";
        this.description = `ESG verilerine dayalı strateji önerileri oluşturur.
        Input: JSON string with {companyData, benchmarkData?, goals?}
        Örnek: {"companyData": {...}, "goals": ["carbon_neutral", "improve_social_score"]}`;
        this.llm = llm;
    }

    async _call(input) {
        try {
            const params = JSON.parse(input);
            const { companyData, benchmarkData, goals } = params;

            // Strateji prompt'u oluştur
            const strategyPrompt = this.buildStrategyPrompt(companyData, benchmarkData, goals);
            
            // LLM'den strateji oluştur
            const response = await this.llm.invoke(strategyPrompt);
            
            // Yapılandırılmış strateji yanıtı
            const strategy = this.parseStrategyResponse(response.content);
            
            return JSON.stringify({
                companyName: companyData.company_name,
                strategy,
                generatedAt: new Date().toISOString()
            });

        } catch (error) {
            return JSON.stringify({
                error: `Strateji üretim hatası: ${error.message}`,
                suggestion: "Şirket verilerini kontrol edin"
            });
        }
    }

    buildStrategyPrompt(companyData, benchmarkData, goals) {
        const prompt = `
ESG Strateji Danışmanlığı

Şirket Bilgileri:
- Ad: ${companyData.company_name || 'Unknown'}
- Sektör: ${companyData.sector || 'Unknown'}
- Çevresel Skor: ${companyData.environment_score || 'N/A'}/100
- Sosyal Skor: ${companyData.social_score || 'N/A'}/100  
- Yönetişim Skoru: ${companyData.governance_score || 'N/A'}/100
- Genel ESG Skoru: ${companyData.overall_esg_score || 'N/A'}/100

${benchmarkData ? `
Sektör Ortalamaları:
- Çevresel: ${benchmarkData.avg_environment || 'N/A'}
- Sosyal: ${benchmarkData.avg_social || 'N/A'}
- Yönetişim: ${benchmarkData.avg_governance || 'N/A'}
` : ''}

${goals ? `Hedefler: ${goals.join(', ')}` : ''}

Lütfen bu şirket için detaylı bir ESG iyileştirme stratejisi oluştur. Şunları içersin:

1. DURUM ANALİZİ
2. ÖNCELİKLİ İYİLEŞTİRME ALANLARI  
3. KISA VADELİ EYLEMLER (6-12 ay)
4. ORTA VADELİ STRATEJİ (1-3 yıl)
5. UZUN VADELİ VİZYON (3+ yıl)
6. KPI'LAR VE ÖLÇÜM YÖNTEMLERİ
7. KAYNAK İHTİYAÇLARI
8. RİSK DEĞERLENDİRMESİ

Her öneri spesifik, ölçülebilir ve uygulanabilir olsun.`;

        return prompt;
    }

    parseStrategyResponse(response) {
        // LLM yanıtını yapılandırılmış formata dönüştür
        const sections = this.extractSections(response);
        
        return {
            situationAnalysis: sections['DURUM ANALİZİ'] || '',
            priorityAreas: sections['ÖNCELİKLİ İYİLEŞTİRME ALANLARI'] || '',
            shortTermActions: sections['KISA VADELİ EYLEMLER'] || '',
            mediumTermStrategy: sections['ORTA VADELİ STRATEJİ'] || '',
            longTermVision: sections['UZUN VADELİ VİZYON'] || '',
            kpis: sections['KPI\'LAR VE ÖLÇÜM YÖNTEMLERİ'] || '',
            resources: sections['KAYNAK İHTİYAÇLARI'] || '',
            riskAssessment: sections['RİSK DEĞERLENDİRMESİ'] || '',
            fullResponse: response
        };
    }

    extractSections(text) {
        const sections = {};
        const sectionHeaders = [
            'DURUM ANALİZİ',
            'ÖNCELİKLİ İYİLEŞTİRME ALANLARI',
            'KISA VADELİ EYLEMLER',
            'ORTA VADELİ STRATEJİ', 
            'UZUN VADELİ VİZYON',
            'KPI\'LAR VE ÖLÇÜM YÖNTEMLERİ',
            'KAYNAK İHTİYAÇLARI',
            'RİSK DEĞERLENDİRMESİ'
        ];

        let currentSection = null;
        let currentContent = '';

        const lines = text.split('\n');
        
        lines.forEach(line => {
            const header = sectionHeaders.find(h => line.includes(h));
            
            if (header) {
                if (currentSection) {
                    sections[currentSection] = currentContent.trim();
                }
                currentSection = header;
                currentContent = '';
            } else if (currentSection) {
                currentContent += line + '\n';
            }
        });

        // Son section'ı da ekle
        if (currentSection) {
            sections[currentSection] = currentContent.trim();
        }

        return sections;
    }
}