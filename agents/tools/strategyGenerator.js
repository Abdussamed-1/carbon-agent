// agents/tools/strategyGenerator.js
import { DynamicTool } from "@langchain/core/tools";

export class StrategyGeneratorTool extends DynamicTool {
    constructor({ llm }) {
        // func fonksiyonunu closure ile llm'yi bağlayarak veriyoruz
        super({
            name: "strategy_generator",
            description: `ESG stratejisi oluşturur. Şirket verileri ve hedeflere göre sürdürülebilirlik stratejileri önerir.\nInput: JSON formatında {companyData, benchmarkData, goals}\n- companyData: Şirket ESG verileri\n- benchmarkData: Sektör benchmark verileri  \n- goals: Hedef ESG skorları veya amaçlar`,
            func: async (input) => {
                return await StrategyGeneratorTool.generateStrategy(llm, input);
            }
        });
        this.llm = llm;
    }

    // func fonksiyonunu static olarak tanımlıyoruz
    static async generateStrategy(llm, input) {
        try {
            let parsedInput;
            if (typeof input === 'string') {
                try {
                    parsedInput = JSON.parse(input);
                } catch {
                    parsedInput = { query: input };
                }
            } else {
                parsedInput = input;
            }
            const { companyData, benchmarkData, goals, query } = parsedInput;
            const strategyPrompt = StrategyGeneratorTool.buildStrategyPrompt(companyData, benchmarkData, goals, query);
            const response = await llm.invoke(strategyPrompt);
            const responseText = typeof response === 'string' ? response : response.content || response.text;
            const strategy = StrategyGeneratorTool.parseStrategyResponse(responseText, parsedInput);
            return JSON.stringify(strategy, null, 2);
        } catch (error) {
            console.error('Strateji oluşturma hatası:', error);
            return JSON.stringify({
                error: "Strateji oluşturulamadı",
                message: error.message,
                suggestions: [
                    "Şirket verilerini kontrol edin",
                    "Hedeflerin net olarak tanımlandığından emin olun",
                    "Benchmark verilerinin doğruluğunu kontrol edin"
                ]
            });
        }
    }

    static buildStrategyPrompt(companyData, benchmarkData, goals, query) {
        let prompt = `Sen ESG (Environmental, Social, Governance) stratejisti olarak, şirketlere sürdürülebilirlik stratejileri geliştiriyorsun.\n\nGÖREV: Aşağıdaki bilgilere dayanarak kapsamlı bir ESG stratejisi oluştur.\n\n`;
        if (companyData) {
            prompt += `ŞİRKET VERİLERİ:\n${JSON.stringify(companyData, null, 2)}\n\n`;
        }
        if (benchmarkData) {
            prompt += `SEKTÖR BENCHMARK VERİLERİ:\n${JSON.stringify(benchmarkData, null, 2)}\n\n`;
        }
        if (goals) {
            prompt += `HEDEFLER:\n${Array.isArray(goals) ? goals.join(', ') : JSON.stringify(goals)}\n\n`;
        }
        if (query) {
            prompt += `ÖZEL İSTEK:\n${query}\n\n`;
        }
        prompt += `STRATEJİ GEREKSİNİMLERİ:\n1. ENVIRONMENTAL (Çevresel) strateji önerileri\n2. SOCIAL (Sosyal) strateji önerileri  \n3. GOVERNANCE (Yönetişim) strateji önerileri\n4. Kısa vadeli aksiyonlar (6-12 ay)\n5. Orta vadeli hedefler (1-3 yıl)\n6. Uzun vadeli vizyonu (3-5 yıl)\n7. Ölçülebilir KPI'lar\n8. Risk analizi ve azaltma yöntemleri\n9. Maliyet tahmini ve ROI beklentisi\n10. Uygulama zamanlaması\n\nÇIKTI FORMATI: Yapılandırılmış, uygulanabilir ve ölçülebilir öneriler sun. Her önerinin gerekçesini belirt.`;
        return prompt;
    }

    static parseStrategyResponse(responseText, originalInput) {
        const strategy = {
            timestamp: new Date().toISOString(),
            companyInfo: originalInput.companyData?.company || "Belirtilmemiş",
            environmental: {
                title: "Çevresel Strateji",
                shortTerm: StrategyGeneratorTool.extractSection(responseText, "çevre", "kısa"),
                mediumTerm: StrategyGeneratorTool.extractSection(responseText, "çevre", "orta"),
                longTerm: StrategyGeneratorTool.extractSection(responseText, "çevre", "uzun"),
                kpis: StrategyGeneratorTool.extractKPIs(responseText, "environmental")
            },
            social: {
                title: "Sosyal Strateji",
                shortTerm: StrategyGeneratorTool.extractSection(responseText, "sosyal", "kısa"),
                mediumTerm: StrategyGeneratorTool.extractSection(responseText, "sosyal", "orta"),
                longTerm: StrategyGeneratorTool.extractSection(responseText, "sosyal", "uzun"),
                kpis: StrategyGeneratorTool.extractKPIs(responseText, "social")
            },
            governance: {
                title: "Yönetişim Stratejisi",
                shortTerm: StrategyGeneratorTool.extractSection(responseText, "yönetişim", "kısa"),
                mediumTerm: StrategyGeneratorTool.extractSection(responseText, "yönetişim", "orta"),
                longTerm: StrategyGeneratorTool.extractSection(responseText, "yönetişim", "uzun"),
                kpis: StrategyGeneratorTool.extractKPIs(responseText, "governance")
            },
            implementation: {
                timeline: StrategyGeneratorTool.extractTimeline(responseText),
                budget: StrategyGeneratorTool.extractBudget(responseText),
                risks: StrategyGeneratorTool.extractRisks(responseText),
                success_metrics: StrategyGeneratorTool.extractSuccessMetrics(responseText)
            },
            summary: StrategyGeneratorTool.extractSummary(responseText),
            rawResponse: responseText
        };
        return strategy;
    }

    static extractSection(text, category, timeframe) {
        const patterns = {
            environmental: ["çevresel", "environment", "enerji", "karbon", "atık"],
            social: ["sosyal", "social", "çalışan", "toplum", "insan"],
            governance: ["yönetişim", "governance", "etik", "şeffaflık", "risk"]
        };
        const timePatterns = {
            kısa: ["kısa", "6 ay", "12 ay", "bir yıl"],
            orta: ["orta", "1-3 yıl", "iki yıl", "üç yıl"],
            uzun: ["uzun", "3-5 yıl", "beş yıl", "vizyon"]
        };
        const lines = text.split('\n');
        const relevantLines = lines.filter(line => {
            const lowerLine = line.toLowerCase();
            const hasCategory = patterns[category]?.some(pattern => lowerLine.includes(pattern));
            const hasTimeframe = timePatterns[timeframe]?.some(pattern => lowerLine.includes(pattern));
            return hasCategory || hasTimeframe;
        });
        return relevantLines.slice(0, 3);
    }

    static extractKPIs(text, category) {
        const kpiPatterns = ["kpi", "metrik", "ölçüm", "hedef", "%", "azalma", "artış"];
        const lines = text.split('\n');
        return lines.filter(line =>
            kpiPatterns.some(pattern => line.toLowerCase().includes(pattern))
        ).slice(0, 5);
    }

    static extractTimeline(text) {
        const timelinePatterns = ["zaman", "süre", "ay", "yıl", "takvim"];
        const lines = text.split('\n');
        return lines.filter(line =>
            timelinePatterns.some(pattern => line.toLowerCase().includes(pattern))
        ).slice(0, 3);
    }

    static extractBudget(text) {
        const budgetPatterns = ["maliyet", "bütçe", "yatırım", "tl", "dolar", "euro"];
        const lines = text.split('\n');
        return lines.filter(line =>
            budgetPatterns.some(pattern => line.toLowerCase().includes(pattern))
        ).slice(0, 2);
    }

    static extractRisks(text) {
        const riskPatterns = ["risk", "tehdit", "sorun", "engel", "zorluk"];
        const lines = text.split('\n');
        return lines.filter(line =>
            riskPatterns.some(pattern => line.toLowerCase().includes(pattern))
        ).slice(0, 3);
    }

    static extractSuccessMetrics(text) {
        const successPatterns = ["başarı", "sonuç", "etki", "fayda", "kazanım"];
        const lines = text.split('\n');
        return lines.filter(line =>
            successPatterns.some(pattern => line.toLowerCase().includes(pattern))
        ).slice(0, 3);
    }

    static extractSummary(text) {
        const lines = text.split('\n').filter(line => line.trim().length > 0);
        return lines.slice(0, 3).join(' ');
    }
}