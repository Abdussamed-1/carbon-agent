# Data Directory

Bu klasör ESG verilerini ve işlenmiş dosyaları içerir.

## Klasör Yapısı

```
data/
├── README.md                           # Bu dosya
├── esg_data.csv                        # Ana ESG dataset
├── carbon-data/                        # Karbon ayak izi verileri
│   └── genisletilmis_gida_karbon_ayakizi.csv
├── processed/                          # İşlenmiş veriler
│   ├── embeddings/                     # Embedding dosyaları
│   ├── chunks/                         # Text chunk'ları
│   └── metadata/                       # Metadata dosyaları
└── backup/                             # Yedek dosyalar
    ├── original/                       # Orijinal veriler
    └── processed/                      # İşlenmiş yedekler
```

## Veri Formatları

### ESG Data CSV Formatı
```csv
company_name,sector,industry,country,date,environment_score,social_score,governance_score,overall_esg_score
Apple Inc.,Technology,Consumer Electronics,United States,2023,75.2,82.1,88.5,81.9
Microsoft Corporation,Technology,Software,United States,2023,78.9,85.3,91.2,85.1
```

### Karbon Ayak İzi Verileri
```csv
food_item,carbon_footprint_kg_co2,production_method,region
beef,13.3,conventional,Global
chicken,2.9,conventional,Global
```

## Veri İşleme

Veriler `scripts/data-processor.js` scripti ile işlenir:

1. **Temizleme**: Eksik değerler ve tutarsızlıklar düzeltilir
2. **Normalizasyon**: Skorlar 0-100 aralığına normalize edilir
3. **Chunking**: Uzun metinler parçalara bölünür
4. **Embedding**: Text'ler vector'lara dönüştürülür

## Veri Güncelleme

Yeni veri eklemek için:

1. CSV dosyasını `data/` klasörüne koyun
2. `npm run process-data` komutunu çalıştırın
3. Vector store'u güncelleyin: `npm run setup`

## Veri Kalitesi

- **Completeness**: %95+ dolu alan
- **Accuracy**: Manuel doğrulama ile %90+
- **Consistency**: Standart format ve birimler
- **Timeliness**: Aylık güncelleme

## Backup Stratejisi

- Orijinal veriler `data/backup/original/` klasöründe saklanır
- İşlenmiş veriler `data/backup/processed/` klasöründe yedeklenir
- Otomatik backup her veri güncellemesinde çalışır 