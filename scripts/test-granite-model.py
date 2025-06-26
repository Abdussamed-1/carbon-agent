# scripts/test-granite-model.py
"""
RTX 4070 8GB için optimize edilmiş model test script
Fine-tuned Granite ESG model'ini test eder
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

def test_granite_esg_model():
    """Fine-tuned Granite ESG modelini test et"""
    
    print("🧪 RTX 4070 Granite ESG Model Test")
    print("=" * 50)
    
    # Paths
    base_model_name = 'ibm-granite/granite-3.0-8b-instruct'
    fine_tuned_path = './models/granite-esg-rtx4070'
    hf_token = os.getenv('HF_TOKEN')
    
    # GPU check
    if not torch.cuda.is_available():
        print("❌ CUDA bulunamadı!")
        return
    
    # Memory temizle
    torch.cuda.empty_cache()
    
    print("🔧 Model yükleniyor...")
    
    try:
        # Conservative quantization for testing
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True  # CPU offload enable
        )
        
        # Base model (CPU offload ile)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=hf_token,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Fine-tuned adapters
        model = PeftModel.from_pretrained(
            base_model, 
            fine_tuned_path,
            torch_dtype=torch.float16
        )
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            fine_tuned_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("✅ Model yüklendi!")
        
        # Test prompts
        test_cases = [
            {
                "name": "ESG Analizi",
                "prompt": """<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
Microsoft şirketinin ESG performansını analiz et
Şirket: Microsoft Corporation (MSFT)
Sektör: Technology
Ülke: USA<|assistant|>
"""
            },
            {
                "name": "İyileştirme Stratejisi", 
                "prompt": """<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
Tesla için ESG iyileştirme stratejisi öner
Mevcut ESG: 45/100
Zayıf alan: Sosyal<|assistant|>
"""
            },
            {
                "name": "Çevresel Değerlendirme",
                "prompt": """<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
Apple şirketinin çevresel performansını değerlendir
Şirket: Apple Inc (AAPL)
Çevresel skor: 75/100<|assistant|>
"""
            }
        ]
        
        # Her test case'i çalıştır
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n🔍 Test {i}: {test_case['name']}")
            print("-" * 40)
            
            # Tokenize
            inputs = tokenizer(
                test_case['prompt'],
                return_tensors="pt",
                truncation=True,
                max_length=200  # Short for memory
            )
            
            # GPU'ya gönder (device_map auto ile)
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,  # Short response
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            print(f"🤖 Model Yanıtı:")
            print(response.strip())
            print("\n" + "="*50)
            
            # Memory check
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"💾 VRAM: {allocated:.2f} GB")
        
        print("\n🎉 Model test tamamlandı!")
        print("✅ Fine-tuned model başarıyla ESG sorularını yanıtlıyor!")
        
    except Exception as e:
        print(f"❌ Test hatası: {e}")
        print("\n💡 Çözümler:")
        print("  - Diğer GPU programlarını kapatın")
        print("  - Windows'u yeniden başlatın") 
        print("  - Model dosyalarını kontrol edin")

if __name__ == "__main__":
    test_granite_esg_model()