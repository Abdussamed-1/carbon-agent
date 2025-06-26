// vectorstore/chromaManager.js
import { ChromaClient } from 'chromadb';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import fs from 'fs-extra';
import Papa from 'papaparse';

export class ChromaManager {
    constructor(options = {}) {
        this.path = options.path || process.env.CHROMA_DB_PATH || 'http://localhost:8000';
        this.collectionName = options.collectionName || 'esg_documents';
        this.isReady = false;
        this.client = options.client || null;
        this.collection = null;
        this.embeddings = null;
    }

    async initialize() {
        try {
            console.log('🗄️ ChromaDB başlatılıyor...');

            // Eğer client dışarıdan verilmemişse oluştur
            if (!this.client) {
                console.log("ChromaClient path:", this.path);
                this.client = new ChromaClient({
                    path: this.path
                });
            }

            // Embedding fonksiyonu
            this.embeddings = new HuggingFaceTransformersEmbeddings({
                modelName: "sentence-transformers/all-MiniLM-L6-v2"
            });

            // Collection oluştur veya al
            await this.ensureCollection();

            this.isReady = true;
            console.log('✅ ChromaDB hazır!');

        } catch (error) {
            console.error('❌ ChromaDB başlatma hatası:', error);
            throw error;
        }
    }

    async ensureCollection() {
        try {
            console.log(`🔍 Collection kontrolü: ${this.collectionName}`);
            
            // Önce mevcut collection'ları listele
            const collections = await this.client.listCollections();
            console.log(`📋 Mevcut collection'lar:`, collections.map(c => c.name));
            
            // Collection'ın var olup olmadığını kontrol et
            const existingCollection = collections.find(c => c.name === this.collectionName);

            if (existingCollection) {
                console.log(`✅ Collection '${this.collectionName}' zaten mevcut, kullanılıyor...`);
                try {
                    this.collection = await this.client.getCollection({
                        name: this.collectionName
                    });
                    console.log(`📚 Collection başarıyla alındı: ${this.collectionName}`);
                } catch (getError) {
                    console.error(`❌ Collection alınırken hata:`, getError);
                    throw getError;
                }
            } else {
                console.log(`📦 Yeni collection oluşturuluyor: ${this.collectionName}`);
                try {
                    this.collection = await this.client.createCollection({
                        name: this.collectionName,
                        metadata: {
                            description: "ESG documents and data",
                            created_at: new Date().toISOString()
                        }
                    });
                    console.log(`✅ Collection başarıyla oluşturuldu: ${this.collectionName}`);
                } catch (createError) {
                    // Eğer collection oluşturma sırasında "already exists" hatası alırsak, getCollection ile dene
                    if (createError.message.includes('already exists') || createError.message.includes('resource already exists')) {
                        console.log(`🔄 Collection zaten var, getCollection ile denenecek...`);
                        try {
                            this.collection = await this.client.getCollection({
                                name: this.collectionName
                            });
                            console.log(`✅ Collection başarıyla alındı (retry): ${this.collectionName}`);
                        } catch (retryError) {
                            console.error(`❌ Collection retry hatası:`, retryError);
                            throw retryError;
                        }
                    } else {
                        console.error(`❌ Collection oluşturma hatası:`, createError);
                        throw createError;
                    }
                }
            }

            // Collection'ın başarıyla alındığını doğrula
            if (!this.collection) {
                throw new Error('Collection oluşturulamadı veya alınamadı');
            }

        } catch (error) {
            console.error('❌ Collection işlemi hatası:', error);
            throw error;
        }
    }

    async addESGData(csvPath) {
        if (!this.isReady) {
            throw new Error('ChromaDB henüz hazır değil');
        }

        try {
            console.log('📊 ESG verileri yükleniyor...');

            // CSV dosyasını oku
            const csvContent = await fs.readFile(csvPath, 'utf8');
            const parsed = Papa.parse(csvContent, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true
            });

            const data = parsed.data;
            console.log(`📈 ${data.length} ESG kaydı bulundu`);

            // Batch olarak işle (1000'li gruplar)
            const batchSize = 1000;
            for (let i = 0; i < data.length; i += batchSize) {
                const batch = data.slice(i, i + batchSize);
                await this.processBatch(batch, i);
            }

            console.log('✅ ESG verileri başarıyla yüklendi!');

        } catch (error) {
            console.error('ESG veri yükleme hatası:', error);
            throw error;
        }
    }

    async processBatch(batch, startIndex) {
        const documents = [];
        const metadatas = [];
        const ids = [];

        for (let i = 0; i < batch.length; i++) {
            const record = batch[i];
            
            // Her kayıt için text oluştur
            const text = this.createDocumentText(record);
            
            documents.push(text);
            metadatas.push({
                company: record.Company || 'Unknown',
                ticker: record.Ticker || '',
                peer_group: record.Peer_group_root || 'Unknown',
                region: record.Region || 'Unknown',
                country: record.Country || 'Unknown',
                date: record.Date || 0,
                total_esg_score: record.total_esg_score || 0,
                environment_score: record.environment_score || 0,
                social_score: record.social_score || 0,
                governance_score: record.governance_score || 0,
                scl_environment_score: record.scl_environment_score || 0,
                scl_social_score: record.scl_social_score || 0,
                scl_governance_score: record.scl_governance_score || 0,
                wght_environment_score: record.wght_environment_score || 0,
                wght_social_score: record.wght_social_score || 0,
                wght_governance_score: record.wght_governance_score || 0,
                record_type: 'esg_data'
            });
            ids.push(`esg_${record.Ticker || 'unknown'}_${record.Date || startIndex + i}`);
        }

        // Batch'i ChromaDB'ye ekle
        await this.collection.add({
            documents,
            metadatas,
            ids
        });

        console.log(`📊 Batch işlendi: ${startIndex + 1}-${startIndex + batch.length}`);
    }

    createDocumentText(record) {
        // ESG kaydını natural language text'e dönüştür
        return `Company: ${record.Company || 'Unknown'}
Ticker: ${record.Ticker || 'N/A'}
Peer Group: ${record.Peer_group_root || 'Unknown'}
Region: ${record.Region || 'Unknown'}
Country: ${record.Country || 'Unknown'}
Date: ${record.Date || 'Unknown'}
Total ESG Score: ${record.total_esg_score || 'N/A'}
Environmental Score: ${record.environment_score || 'N/A'}
Social Score: ${record.social_score || 'N/A'}
Governance Score: ${record.governance_score || 'N/A'}
Scaled Environmental Score: ${record.scl_environment_score || 'N/A'}
Scaled Social Score: ${record.scl_social_score || 'N/A'}
Scaled Governance Score: ${record.scl_governance_score || 'N/A'}
Weighted Environmental Score: ${record.wght_environment_score || 'N/A'}
Weighted Social Score: ${record.wght_social_score || 'N/A'}
Weighted Governance Score: ${record.wght_governance_score || 'N/A'}

This company ${record.Company} (${record.Ticker}) from the ${record.Country} in ${record.Region} region has a total ESG performance score of ${record.total_esg_score}. 
The environmental performance score is ${record.environment_score}, social responsibility score is ${record.social_score}, and governance score is ${record.governance_score}.
The company belongs to the ${record.Peer_group_root} peer group. The scaled scores are: Environmental ${record.scl_environment_score}, Social ${record.scl_social_score}, Governance ${record.scl_governance_score}.
The weighted scores are: Environmental ${record.wght_environment_score}, Social ${record.wght_social_score}, Governance ${record.wght_governance_score}.`;
    }

    async searchSimilar(query, options = {}) {
        if (!this.isReady) {
            throw new Error('ChromaDB henüz hazır değil');
        }

        try {
            const {
                nResults = 10,
                whereFilter = {},
                includeMetadata = true,
                includeDocuments = true
            } = options;

            const results = await this.collection.query({
                queryTexts: [query],
                nResults,
                where: whereFilter,
                include: ['metadatas', 'documents', 'distances']
            });

            return {
                query,
                results: results.documents[0].map((doc, i) => ({
                    document: doc,
                    metadata: results.metadatas[0][i],
                    distance: results.distances[0][i],
                    similarity: 1 - results.distances[0][i]
                }))
            };

        } catch (error) {
            console.error('Arama hatası:', error);
            throw error;
        }
    }

    async searchByCompany(companyName, options = {}) {
        return this.searchSimilar(`${companyName} ESG performance`, {
            ...options,
            whereFilter: {
                $or: [
                    { company: { $eq: companyName } },
                    { company: { $contains: companyName } },
                    { ticker: { $eq: companyName.toUpperCase() } }
                ]
            }
        });
    }

    async searchByTicker(ticker, options = {}) {
        return this.searchSimilar(`${ticker} stock ESG analysis`, {
            ...options,
            whereFilter: {
                ticker: { $eq: ticker.toUpperCase() }
            }
        });
    }

    async searchByRegion(region, options = {}) {
        return this.searchSimilar(`${region} region ESG analysis`, {
            ...options,
            whereFilter: {
                region: { $eq: region }
            }
        });
    }

    async searchByCountry(country, options = {}) {
        return this.searchSimilar(`${country} country ESG performance`, {
            ...options,
            whereFilter: {
                country: { $eq: country }
            }
        });
    }

    async searchBySector(sector, options = {}) {
        return this.searchSimilar(`${sector} sector ESG analysis`, {
            ...options,
            whereFilter: {
                peer_group: { $eq: sector }
            }
        });
    }

    async getESGTrends(filters = {}) {
        const results = await this.collection.get({
            where: filters,
            include: ['metadatas']
        });

        // Trend analizi için veriyi grupla
        const trends = {};
        results.metadatas.forEach(meta => {
            const key = `${meta.company}_${meta.date}`;
            trends[key] = {
                company: meta.company,
                ticker: meta.ticker,
                date: meta.date,
                country: meta.country,
                region: meta.region,
                peer_group: meta.peer_group,
                scores: {
                    total_esg: meta.total_esg_score,
                    environment: meta.environment_score,
                    social: meta.social_score,
                    governance: meta.governance_score,
                    scaled: {
                        environment: meta.scl_environment_score,
                        social: meta.scl_social_score,
                        governance: meta.scl_governance_score
                    },
                    weighted: {
                        environment: meta.wght_environment_score,
                        social: meta.wght_social_score,
                        governance: meta.wght_governance_score
                    }
                }
            };
        });

        return Object.values(trends);
    }

    async getCollectionStats() {
        if (!this.isReady) {
            return { error: 'ChromaDB not ready' };
        }

        try {
            const count = await this.collection.count();
            
            return {
                collectionName: this.collectionName,
                documentCount: count,
                isReady: this.isReady,
                path: this.path
            };

        } catch (error) {
            console.error('Stats hatası:', error);
            return { error: error.message };
        }
    }

    // Collection'ı temizle
    async clearCollection() {
        if (this.collection) {
            try {
                await this.client.deleteCollection({ name: this.collectionName });
                console.log(`🗑️ Collection silindi: ${this.collectionName}`);
            } catch (error) {
                console.log(`⚠️ Collection silme hatası (muhtemelen zaten yok):`, error.message);
            }
            await this.ensureCollection();
        }
    }

    // Collection'ı sıfırla (temizle ve yeniden oluştur)
    async resetCollection() {
        console.log(`🔄 Collection sıfırlanıyor: ${this.collectionName}`);
        
        try {
            // Önce collection'ı sil
            await this.client.deleteCollection({ name: this.collectionName });
            console.log(`✅ Collection silindi: ${this.collectionName}`);
        } catch (error) {
            console.log(`⚠️ Collection zaten yok veya silinemiyor:`, error.message);
        }

        // Yeniden oluştur
        await this.ensureCollection();
        console.log(`✅ Collection sıfırlandı: ${this.collectionName}`);
    }
}