# scripts/esg-inference-api.py
"""
Fine-tuned Granite ESG Model Inference API
RTX 4070 8GB için optimize edilmiş
"""

import os
import torch
import json
from datetime import datetime
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings('ignore')

class ESGGraniteInference:
    """Fine-tuned Granite ESG model için inference class"""
    
    def __init__(self, model_path: str = './models/granite-esg-rtx4070'):
        self.model_path = model_path
        self.base_model_name = 'ibm-granite/granite-3.0-8b-instruct'
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🎯 ESG Granite Inference başlatılıyor...")
        print(f"🔧 Device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Model ve tokenizer'ı yükle"""
        try:
            print("🤖 Model yükleniyor...")
            
            # Memory temizle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )
            
            # Base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.getenv('HF_TOKEN'),
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            # Fine-tuned model
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=torch.float16
            )
            
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("✅ Model başarıyla yüklendi!")
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def analyze_company_esg(self, company: str, ticker: str = "", 
                           sector: str = "", country: str = "") -> str:
        """Şirket ESG analizi yap"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company} şirketinin ESG performansını analiz et
Şirket: {company} ({ticker})
Sektör: {sector}
Ülke: {country}<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def suggest_esg_improvements(self, company: str, current_score: float, 
                                weak_area: str = "") -> str:
        """ESG iyileştirme önerileri"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company} için ESG iyileştirme stratejisi öner
Mevcut ESG: {current_score}/100
Zayıf alan: {weak_area}<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def evaluate_environmental_performance(self, company: str, 
                                         env_score: float) -> str:
        """Çevresel performans değerlendirmesi"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company} şirketinin çevresel performansını değerlendir
Çevresel skor: {env_score}/100<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def assess_social_responsibility(self, company: str, 
                                   social_score: float) -> str:
        """Sosyal sorumluluk değerlendirmesi"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company} şirketinin sosyal sorumluluk performansını değerlendir
Sosyal skor: {social_score}/100<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def evaluate_governance(self, company: str, gov_score: float) -> str:
        """Kurumsal yönetim değerlendirmesi"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company} şirketinin kurumsal yönetim kalitesini değerlendir
Yönetim skoru: {gov_score}/100<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def compare_companies(self, company1: str, company2: str,
                         score1: float, score2: float) -> str:
        """İki şirketi ESG açısından karşılaştır"""
        
        prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{company1} ve {company2} şirketlerini ESG performansı açısından karşılaştır
{company1} ESG skoru: {score1}/100
{company2} ESG skoru: {score2}/100<|assistant|>
"""
        
        return self._generate_response(prompt)
    
    def _generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Model yanıtı üret"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=300
            )
            
            # GPU'ya gönder
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    top_p=0.9
                )
            
            # Decode
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            return f"❌ Yanıt üretme hatası: {e}"
    
    def batch_analyze(self, companies: List[Dict]) -> List[Dict]:
        """Toplu şirket analizi"""
        results = []
        
        for company_data in companies:
            company = company_data.get('company', '')
            ticker = company_data.get('ticker', '')
            sector = company_data.get('sector', '')
            country = company_data.get('country', '')
            
            try:
                analysis = self.analyze_company_esg(company, ticker, sector, country)
                
                result = {
                    'company': company,
                    'ticker': ticker,
                    'analysis': analysis,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
            except Exception as e:
                result = {
                    'company': company,
                    'ticker': ticker,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'error'
                }
            
            results.append(result)
            print(f"✅ {company} analizi tamamlandı")
        
        return results
    
    def get_memory_usage(self) -> Dict:
        """GPU memory kullanımını al"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            return {
                'allocated_gb': round(allocated, 2),
                'reserved_gb': round(reserved, 2),
                'total_gb': round(total, 2),
                'free_gb': round(total - reserved, 2)
            }
        return {}


def main():
    """Demo fonksiyonu"""
    print("🎯 ESG Granite Inference Demo")
    print("=" * 50)
    
    # Inference engine başlat
    try:
        esg_ai = ESGGraniteInference()
        
        # Demo analizler
        print("\n🔍 Demo Analizler:")
        
        # 1. Company Analysis
        print("\n1️⃣ Şirket ESG Analizi:")
        analysis = esg_ai.analyze_company_esg(
            company="Microsoft Corporation",
            ticker="MSFT",
            sector="Technology",
            country="USA"
        )
        print(f"🤖 {analysis}")
        
        # 2. Improvement Suggestions
        print("\n2️⃣ İyileştirme Önerileri:")
        improvements = esg_ai.suggest_esg_improvements(
            company="Tesla Inc",
            current_score=45,
            weak_area="Sosyal"
        )
        print(f"🤖 {improvements}")
        
        # 3. Environmental Assessment
        print("\n3️⃣ Çevresel Değerlendirme:")
        env_assessment = esg_ai.evaluate_environmental_performance(
            company="Apple Inc",
            env_score=78
        )
        print(f"🤖 {env_assessment}")
        
        # Memory usage
        memory = esg_ai.get_memory_usage()
        if memory:
            print(f"\n💾 GPU Memory: {memory['allocated_gb']:.2f} GB / {memory['total_gb']:.2f} GB")
        
    except Exception as e:
        print(f"❌ Demo hatası: {e}")


if __name__ == "__main__":
    main()