// scripts/reset-chromadb.js
import { ChromaClient } from 'chromadb';
import dotenv from 'dotenv';

dotenv.config();

async function resetChromaDB() {
    try {
        console.log('🔄 ChromaDB sıfırlanıyor...');
        
        const client = new ChromaClient({
            path: process.env.CHROMA_DB_PATH || 'http://localhost:8000'
        });
        
        // Tüm collection'ları listele
        const collections = await client.listCollections();
        console.log('📋 Mevcut collection\'lar:', collections.map(c => c.name));
        
        // ESG collection'ını sil
        const esgCollection = collections.find(c => c.name === 'esg_documents');
        if (esgCollection) {
            await client.deleteCollection({ name: 'esg_documents' });
            console.log('✅ esg_documents collection silindi');
        } else {
            console.log('ℹ️ esg_documents collection zaten yok');
        }
        
        // Tekrar listele
        const remainingCollections = await client.listCollections();
        console.log('📋 Kalan collection\'lar:', remainingCollections.map(c => c.name));
        
        console.log('✅ ChromaDB sıfırlama tamamlandı!');
        
    } catch (error) {
        console.error('❌ ChromaDB sıfırlama hatası:', error);
    }
}

resetChromaDB();