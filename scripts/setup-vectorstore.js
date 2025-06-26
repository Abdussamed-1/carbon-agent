// scripts/setup-vectorstore.js
import { ChromaManager } from '../vectorstore/chromaManager.js';
import fs from 'fs-extra';
import axios from 'axios';
import Papa from 'papaparse';
import dotenv from 'dotenv';

dotenv.config();

class VectorStoreSetup {
    constructor() {
        this.chromaManager = new ChromaManager({
            path: process.env.CHROMA_DB_PATH || './chroma_db',
            collectionName: 'esg_documents'
        });
        this.dataDir = './data';
        this.esgDatasetUrl = 'https://huggingface.co/datasets/nlp-esg-scoring/spx-sustainalytics-esg-scores';
    }

    async initialize() {
        try {
            console.log('🚀 VectorStore kurulumu başlıyor...');
            
            // Dizinleri oluştur
            await this.createDirectories();
            
            // ESG dataset'ini indir
            await this.downloadESGDataset();
            
            // ChromaDB başlat
            await this.chromaManager.initialize();
            
            // Verileri yükle
            await this.loadESGData();
            
            console.log('✅ VectorStore kurulumu tamamlandı!');
            
            // İstatistikleri göster
            await this.showStats();
            
        } catch (error) {
            console.error('❌ Kurulum hatası:', error);
            process.exit(1);
        }
    }

    async createDirectories() {
        const dirs = [
            this.dataDir,
            './data/processed',
            './models',
            './chroma_db',
            './logs'
        ];

        for (const dir of dirs) {
            await fs.ensureDir(dir);
            console.log(`📁 Dizin oluşturuldu: ${dir}`);
        }
    }

    async downloadESGDataset() {
        const esgFilePath = `${this.dataDir}/carbon-data/esg_scores.csv`;
        
        if (await fs.pathExists(esgFilePath)) {
            console.log('📊 ESG dataset zaten mevcut');
            return;
        }

        console.log('❌ ESG dataset bulunamadı. Lütfen data/carbon-data/esg_scores.csv dosyasını yükleyin.');
        throw new Error('ESG dataset bulunamadı');
    }

    convertToCSV(rows) {
        const headers = Object.keys(rows[0].row);
        const csvRows = [headers.join(',')];
        
        rows.forEach(item => {
            const row = headers.map(header => {
                const value = item.row[header];
                return typeof value === 'string' ? `"${value}"` : value;
            });
            csvRows.push(row.join(','));
        });
        
        return csvRows.join('\n');
    }

    async createDemoESGData(filePath) {
        console.log('🎭 Demo ESG verisi oluşturuluyor...');
        
        const companies = [
            { name: 'Apple Inc.', symbol: 'AAPL', sector: 'Technology', industry: 'Consumer Electronics', country: 'United States' },
            { name: 'Microsoft Corporation', symbol: 'MSFT', sector: 'Technology', industry: 'Software', country: 'United States' },
            { name: 'Alphabet Inc.', symbol: 'GOOGL', sector: 'Technology', industry: 'Internet Services', country: 'United States' },
            { name: 'Tesla Inc.', symbol: 'TSLA', sector: 'Automotive', industry: 'Electric Vehicles', country: 'United States' },
            { name: 'Amazon.com Inc.', symbol: 'AMZN', sector: 'Consumer Discretionary', industry: 'E-commerce', country: 'United States' },
            { name: 'Nestlé S.A.', symbol: 'NESN', sector: 'Consumer Staples', industry: 'Food & Beverages', country: 'Switzerland' },
            { name: 'Unilever PLC', symbol: 'UL', sector: 'Consumer Staples', industry: 'Personal Care', country: 'United Kingdom' },
            { name: 'Siemens AG', symbol: 'SIE', sector: 'Industrials', industry: 'Electrical Equipment', country: 'Germany' },
            { name: 'JPMorgan Chase & Co.', symbol: 'JPM', sector: 'Financials', industry: 'Banking', country: 'United States' },
            { name: 'Johnson & Johnson', symbol: 'JNJ', sector: 'Healthcare', industry: 'Pharmaceuticals', country: 'United States' }
        ];

        const demoData = [];
        const years = ['2020', '2021', '2022', '2023', '2024'];

        companies.forEach(company => {
            years.forEach(year => {
                // Rastgele ama mantıklı ESG skorları oluştur
                const baseEnv = Math.random() * 40 + 30; // 30-70 arası
                const baseSocial = Math.random() * 40 + 40; // 40-80 arası  
                const baseGov = Math.random() * 30 + 50; // 50-80 arası
                
                // Yıl bazlı trend ekle
                const yearTrend = (parseInt(year) - 2020) * 2; // Her yıl biraz iyileşme
                
                const envScore = Math.min(100, Math.max(0, baseEnv + yearTrend + (Math.random() - 0.5) * 10));
                const socialScore = Math.min(100, Math.max(0, baseSocial + yearTrend + (Math.random() - 0.5) * 10));
                const govScore = Math.min(100, Math.max(0, baseGov + yearTrend + (Math.random() - 0.5) * 10));
                const overallScore = (envScore + socialScore + govScore) / 3;

                demoData.push({
                    company_name: company.name,
                    symbol: company.symbol,
                    sector: company.sector,
                    industry: company.industry,
                    country: company.country,
                    environment_score: envScore.toFixed(1),
                    social_score: socialScore.toFixed(1),
                    governance_score: govScore.toFixed(1),
                    overall_esg_score: overallScore.toFixed(1),
                    date: `${year}-12-31`,
                    data_source: 'Demo Data',
                    last_updated: new Date().toISOString()
                });
            });
        });

        // CSV formatına dönüştür
        const csv = Papa.unparse(demoData);
        await fs.writeFile(filePath, csv);
        
        console.log(`✅ ${demoData.length} demo ESG kaydı oluşturuldu`);
    }

    async loadESGData() {
        const esgFilePath = `${this.dataDir}/carbon-data/esg_scores.csv`;
        
        if (!(await fs.pathExists(esgFilePath))) {
            throw new Error('ESG data dosyası bulunamadı: ' + esgFilePath);
        }

        console.log('📊 ESG verileri ChromaDB\'ye yükleniyor...');
        await this.chromaManager.addESGData(esgFilePath);
    }

    async showStats() {
        const stats = await this.chromaManager.getCollectionStats();
        
        console.log('\n📈 VectorStore İstatistikleri:');
        console.log('================================');
        console.log(`Collection: ${stats.collectionName}`);
        console.log(`Toplam Dokuman: ${stats.documentCount}`);
        console.log(`Durum: ${stats.isReady ? '✅ Hazır' : '❌ Hazır Değil'}`);
        console.log(`Veri Yolu: ${stats.path}`);
        
        // Test sorgusu yap
        if (stats.isReady && stats.documentCount > 0) {
            console.log('\n🔍 Test Sorgusu:');
            const testResult = await this.chromaManager.searchSimilar(
                'technology companies environmental performance', 
                { nResults: 3 }
            );
            
            console.log(`Test sonuçları: ${testResult.results.length} dokuman bulundu`);
            if (testResult.results.length > 0) {
                const topResult = testResult.results[0];
                console.log(`En iyi eşleşme: ${topResult.metadata.company_name} (${topResult.similarity.toFixed(2)} benzerlik)`);
            }
        }
    }
}

// Ana fonksiyon
async function main() {
    const setup = new VectorStoreSetup();
    await setup.initialize();
}

// Script olarak çalıştırıldığında
if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}

export { VectorStoreSetup };

// scripts/data-processor.js
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

export class ESGDataProcessor {
    constructor() {
        this.inputPath = './data/carbon-data/esg_scores.csv';
        this.outputPath = './data/processed/';
        this.embeddings = null;
    }

    async initialize() {
        this.embeddings = new HuggingFaceTransformersEmbeddings({
            modelName: "sentence-transformers/all-MiniLM-L6-v2"
        });
        
        await fs.ensureDir(this.outputPath);
    }

    async processESGData() {
        try {
            console.log('🔄 ESG verileri işleniyor...');
            
            // CSV dosyasını oku
            const csvContent = await fs.readFile(this.inputPath, 'utf8');
            const parsed = Papa.parse(csvContent, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true
            });

            const data = parsed.data;
            console.log(`📊 ${data.length} kayıt bulundu`);

            // Veri temizleme
            const cleanedData = this.cleanData(data);
            console.log(`🧹 ${cleanedData.length} kayıt temizlendi`);

            // Sektör analizi
            const sectorAnalysis = this.analyzeSectors(cleanedData);
            await fs.writeJson(`${this.outputPath}/sector_analysis.json`, sectorAnalysis, { spaces: 2 });

            // Trend analizi
            const trendAnalysis = this.analyzeTrends(cleanedData);
            await fs.writeJson(`${this.outputPath}/trend_analysis.json`, trendAnalysis, { spaces: 2 });

            // Şirket özetleri
            const companySummaries = this.generateCompanySummaries(cleanedData);
            await fs.writeJson(`${this.outputPath}/company_summaries.json`, companySummaries, { spaces: 2 });

            // İşlenmiş veriyi kaydet
            const processedCsv = Papa.unparse(cleanedData);
            await fs.writeFile(`${this.outputPath}/processed_esg_data.csv`, processedCsv);

            console.log('✅ Veri işleme tamamlandı!');
            
            return {
                totalRecords: cleanedData.length,
                sectors: Object.keys(sectorAnalysis),
                companies: Object.keys(companySummaries),
                outputFiles: [
                    'sector_analysis.json',
                    'trend_analysis.json', 
                    'company_summaries.json',
                    'processed_esg_data.csv'
                ]
            };

        } catch (error) {
            console.error('❌ Veri işleme hatası:', error);
            throw error;
        }
    }

    cleanData(data) {
        return data.filter(record => {
            // Gerekli alanları kontrol et
            return record.company_name && 
                   record.sector &&
                   typeof record.environment_score === 'number' &&
                   typeof record.social_score === 'number' &&
                   typeof record.governance_score === 'number';
        }).map(record => {
            // Veri tiplerini normalize et
            return {
                ...record,
                environment_score: parseFloat(record.environment_score) || 0,
                social_score: parseFloat(record.social_score) || 0,
                governance_score: parseFloat(record.governance_score) || 0,
                overall_esg_score: parseFloat(record.overall_esg_score) || 
                    (parseFloat(record.environment_score) + parseFloat(record.social_score) + parseFloat(record.governance_score)) / 3,
                date: this.normalizeDate(record.date)
            };
        });
    }

    normalizeDate(dateStr) {
        if (!dateStr) return new Date().toISOString().split('T')[0];
        
        try {
            const date = new Date(dateStr);
            return date.toISOString().split('T')[0];
        } catch {
            return new Date().toISOString().split('T')[0];
        }
    }

    analyzeSectors(data) {
        const sectors = {};
        
        data.forEach(record => {
            const sector = record.sector;
            if (!sectors[sector]) {
                sectors[sector] = {
                    companies: [],
                    avgScores: {
                        environment: 0,
                        social: 0,
                        governance: 0,
                        overall: 0
                    },
                    count: 0,
                    countries: new Set()
                };
            }
            
            sectors[sector].companies.push(record.company_name);
            sectors[sector].avgScores.environment += record.environment_score;
            sectors[sector].avgScores.social += record.social_score;
            sectors[sector].avgScores.governance += record.governance_score;
            sectors[sector].avgScores.overall += record.overall_esg_score;
            sectors[sector].count++;
            sectors[sector].countries.add(record.country);
        });

        // Ortalamaları hesapla
        Object.keys(sectors).forEach(sector => {
            const sectorData = sectors[sector];
            const count = sectorData.count;
            
            sectorData.avgScores.environment = (sectorData.avgScores.environment / count).toFixed(2);
            sectorData.avgScores.social = (sectorData.avgScores.social / count).toFixed(2);
            sectorData.avgScores.governance = (sectorData.avgScores.governance / count).toFixed(2);
            sectorData.avgScores.overall = (sectorData.avgScores.overall / count).toFixed(2);
            sectorData.companies = [...new Set(sectorData.companies)];
            sectorData.countries = [...sectorData.countries];
        });

        return sectors;
    }

    analyzeTrends(data) {
        const trends = {};
        
        // Şirket bazlı trendler
        data.forEach(record => {
            const company = record.company_name;
            const year = new Date(record.date).getFullYear();
            
            if (!trends[company]) {
                trends[company] = {};
            }
            
            trends[company][year] = {
                environment: record.environment_score,
                social: record.social_score,
                governance: record.governance_score,
                overall: record.overall_esg_score
            };
        });

        // Trend hesaplamaları
        Object.keys(trends).forEach(company => {
            const years = Object.keys(trends[company]).sort();
            if (years.length > 1) {
                const firstYear = years[0];
                const lastYear = years[years.length - 1];
                const first = trends[company][firstYear];
                const last = trends[company][lastYear];
                
                trends[company].trend = {
                    environment: (last.environment - first.environment).toFixed(2),
                    social: (last.social - first.social).toFixed(2),
                    governance: (last.governance - first.governance).toFixed(2),
                    overall: (last.overall - first.overall).toFixed(2),
                    years: `${firstYear}-${lastYear}`
                };
            }
        });

        return trends;
    }

    generateCompanySummaries(data) {
        const summaries = {};
        
        data.forEach(record => {
            const company = record.company_name;
            
            if (!summaries[company]) {
                summaries[company] = {
                    sector: record.sector,
                    industry: record.industry,
                    country: record.country,
                    symbol: record.symbol,
                    records: [],
                    latestScores: {},
                    averageScores: {
                        environment: 0,
                        social: 0, 
                        governance: 0,
                        overall: 0
                    }
                };
            }
            
            summaries[company].records.push({
                date: record.date,
                environment_score: record.environment_score,
                social_score: record.social_score,
                governance_score: record.governance_score,
                overall_esg_score: record.overall_esg_score
            });
        });

        // Her şirket için özetleri hesapla
        Object.keys(summaries).forEach(company => {
            const summary = summaries[company];
            const records = summary.records.sort((a, b) => new Date(b.date) - new Date(a.date));
            
            // En son skorlar
            summary.latestScores = records[0];
            
            // Ortalama skorlar
            const count = records.length;
            summary.averageScores.environment = (records.reduce((sum, r) => sum + r.environment_score, 0) / count).toFixed(2);
            summary.averageScores.social = (records.reduce((sum, r) => sum + r.social_score, 0) / count).toFixed(2);
            summary.averageScores.governance = (records.reduce((sum, r) => sum + r.governance_score, 0) / count).toFixed(2);
            summary.averageScores.overall = (records.reduce((sum, r) => sum + r.overall_esg_score, 0) / count).toFixed(2);
            
            summary.recordCount = count;
            summary.dateRange = {
                earliest: records[records.length - 1].date,
                latest: records[0].date
            };
        });

        return summaries;
    }
}

// Ana fonksiyon
async function processData() {
    const processor = new ESGDataProcessor();
    await processor.initialize();
    const result = await processor.processESGData();
    
    console.log('\n📈 İşleme Özeti:');
    console.log('================');
    console.log(`Toplam Kayıt: ${result.totalRecords}`);
    console.log(`Sektör Sayısı: ${result.sectors.length}`);
    console.log(`Şirket Sayısı: ${result.companies.length}`);
    console.log(`Çıktı Dosyaları: ${result.outputFiles.join(', ')}`);
}

// Script olarak çalıştırıldığında
if (import.meta.url === `file://${process.argv[1]}`) {
    processData().catch(console.error);
}