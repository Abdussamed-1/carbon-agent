// scripts/load-esg-data.js
import { ChromaManager } from '../vectorstore/chromaManager.js';
import { getChromaClient } from '../vectorstore/chromaFactory.js';
import dotenv from 'dotenv';
import path from 'path';

dotenv.config();

async function loadESGData() {
    try {
        console.log('📊 ESG veri yükleme başlatılıyor...');
        
        // ChromaDB bağlantısı
        const config = {
            CHROMA_DB_PATH: process.env.CHROMA_DB_PATH || "http://localhost:8000"
        };
        
        const chromaClient = getChromaClient(config);
        const chromaManager = new ChromaManager({ 
            client: chromaClient, 
            collectionName: "esg_documents",
            path: config.CHROMA_DB_PATH 
        });
        
        await chromaManager.initialize();
        
        // ESG CSV dosyasının yolu
        const csvPath = process.env.ESG_DATASET_PATH || './data/carbon-data/esg_scores.csv';
        
        console.log(`📁 CSV dosyası yolu: ${csvPath}`);
        
        // Dosyanın var olup olmadığını kontrol et
        try {
            const fs = await import('fs-extra');
            const exists = await fs.pathExists(csvPath);
            if (!exists) {
                console.error(`❌ CSV dosyası bulunamadı: ${csvPath}`);
                console.log('💡 Lütfen ESG_DATASET_PATH environment variable\'ını kontrol edin.');
                console.log('💡 Mevcut .env dosyasında ESG_DATASET_PATH=./data/carbon-data/esg_scores.csv olarak ayarlanmış.');
                console.log('💡 CSV dosyasını doğru konuma kopyalayın veya path\'i güncelleyin.');
                process.exit(1);
            }
            
            // Dosya boyutunu kontrol et
            const stats = await fs.stat(csvPath);
            console.log(`📏 Dosya boyutu: ${(stats.size / 1024 / 1024).toFixed(2)} MB`);
            
        } catch (error) {
            console.error('❌ Dosya kontrol hatası:', error);
            process.exit(1);
        }
        
        // Mevcut collection'ı temizle (isteğe bağlı)
        const shouldReset = process.argv.includes('--reset');
        if (shouldReset) {
            console.log('🔄 Mevcut collection temizleniyor...');
            await chromaManager.resetCollection();
        }
        
        // Verileri yükle
        console.log('📥 Veri yükleme başlıyor...');
        const startTime = Date.now();
        
        await chromaManager.addESGData(csvPath);
        
        const endTime = Date.now();
        const duration = ((endTime - startTime) / 1000).toFixed(2);
        
        // İstatistikleri göster
        const stats = await chromaManager.getCollectionStats();
        console.log('🎉 Yükleme tamamlandı!');
        console.log(`⏱️ Süre: ${duration} saniye`);
        console.log('📊 İstatistikler:', stats);
        
        // Örnek arama testi
        console.log('\n🔍 Örnek arama testi yapılıyor...');
        const testResults = await chromaManager.searchSimilar('Apple Tesla ESG performance', { nResults: 3 });
        console.log('🎯 Test sonuçları:');
        testResults.results.forEach((result, i) => {
            console.log(`${i + 1}. ${result.metadata.company} (${result.metadata.ticker}) - Similarity: ${result.similarity.toFixed(3)}`);
        });
        
        process.exit(0);
        
    } catch (error) {
        console.error('❌ Veri yükleme hatası:', error);
        process.exit(1);
    }
}

// Ana fonksiyon
async function main() {
    const args = process.argv.slice(2);
    
    if (args.includes('--help')) {
        console.log(`
ESG Veri Yükleme Script'i

Kullanım:
  node scripts/load-esg-data.js                 # Verileri yükle
  node scripts/load-esg-data.js --reset         # Collection'ı sıfırla ve yükle
  node scripts/load-esg-data.js --help          # Bu yardımı göster

Environment Variables:
  ESG_DATASET_PATH - CSV dosyasının yolu (varsayılan: ./data/carbon-data/esg_scores.csv)
  CHROMA_DB_PATH   - ChromaDB URL'i (varsayılan: http://localhost:8000)

CSV Dosyası Formatı:
  Beklenen sütunlar: Date, Ticker, Company, Peer_group_root, Region, Country,
  total_esg_score, environment_score, social_score, governance_score,
  scl_environment_score, scl_social_score, scl_governance_score,
  wght_environment_score, wght_social_score, wght_governance_score, test

Örnek:
  13,237 satır ve 17 sütunlu ESG verisi yüklenecek.
        `);
        return;
    }
    
    await loadESGData();
}

main().catch(console.error);