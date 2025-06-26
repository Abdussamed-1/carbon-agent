# Models Directory

Bu klasör fine-tuned IBM Granite modellerini içerir.

## Klasör Yapısı

```
models/
├── README.md                           # Bu dosya
├── granite-esg-finetuned/             # Fine-tuned model dosyaları
│   ├── config.json                    # Model konfigürasyonu
│   ├── pytorch_model.bin              # Model ağırlıkları
│   ├── tokenizer.json                 # Tokenizer
│   ├── tokenizer_config.json          # Tokenizer konfigürasyonu
│   ├── special_tokens_map.json        # Özel tokenler
│   └── training_args.bin              # Eğitim parametreleri
└── checkpoints/                       # Eğitim sırasında kaydedilen checkpoint'ler
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── checkpoint-1500/
```

## Model Bilgileri

- **Base Model**: `ibm-granite/granite-3.0-8b-instruct`
- **Fine-tuning Dataset**: ESG analiz verileri
- **Training Epochs**: 3
- **Learning Rate**: 2e-5
- **Batch Size**: 4

## Kullanım

Model, `scripts/fine-tune-granite.py` scripti ile eğitilir ve `config/models.js` dosyasında konfigüre edilir.

## Performans Metrikleri

- **Accuracy**: ~85%
- **Response Time**: ~2s
- **ESG Domain Knowledge**: Yüksek

## Güncelleme

Yeni verilerle modeli güncellemek için:

1. `scripts/fine-tune-granite.py` scriptini çalıştırın
2. Yeni model dosyalarını bu klasöre kopyalayın
3. `config/models.js` dosyasında model yolunu güncelleyin 