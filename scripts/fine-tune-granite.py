# scripts/fine-tune-granite-rtx4070-final.py
"""
RTX 4070 8GB için optimize edilmiş Granite ESG Fine-tuning Script
Clean Code Architecture ile geliştirilmiş
"""

import os
import gc
import json
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Transformers & Training
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Warnings
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RTX4070Config:
    """RTX 4070 8GB için optimize edilmiş konfigürasyon"""
    
    # Model Configuration
    model_checkpoint: str = 'ibm-granite/granite-3.0-8b-instruct'
    output_dir: str = './models/granite-esg-rtx4070'
    dataset_path: str = './data/esg_training_data.csv'
    
    # Memory Management
    max_memory_gb: float = 7.5  # RTX 4070 8GB için güvenli limit
    memory_fraction: float = 0.95
     
    # Training Parameters
    num_train_epochs: int = 2
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1e-4
    warmup_steps: int = 50
    max_samples: int = 800  # 8GB için optimize
    
    # LoRA Configuration
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    
    # Logging
    logging_steps: int = 5
    save_steps: int = 200
    eval_steps: int = 200


class GPUMemoryManager:
    """GPU memory yönetimi için utility class"""
    
    @staticmethod
    def setup_gpu(memory_fraction: float = 0.95) -> bool:
        """GPU'yu kurula ve memory ayarlarını yap"""
        if not torch.cuda.is_available():
            print("❌ CUDA bulunamadı!")
            return False
        
        # Memory temizle
        torch.cuda.empty_cache()
        gc.collect()
        
        # Memory fraction ayarla
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
        
        # GPU bilgilerini göster
        gpu_name = torch.cuda.get_device_name()
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"🎮 GPU: {gpu_name}")
        print(f"💾 VRAM: {total_memory:.1f} GB")
        
        return True
    
    @staticmethod
    def check_memory() -> Dict[str, float]:
        """Güncel memory kullanımını kontrol et"""
        if not torch.cuda.is_available():
            return {}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free = total - reserved
        
        return {
            'allocated': allocated,
            'reserved': reserved,
            'total': total,
            'free': free
        }
    
    @staticmethod
    def print_memory_stats():
        """Memory istatistiklerini yazdır"""
        stats = GPUMemoryManager.check_memory()
        if stats:
            print(f"📊 GPU Memory:")
            print(f"  Allocated: {stats['allocated']:.2f} GB")
            print(f"  Reserved: {stats['reserved']:.2f} GB")
            print(f"  Total: {stats['total']:.2f} GB")
            print(f"  Free: {stats['free']:.2f} GB")


class ESGDataProcessor:
    """ESG verilerini işleme ve training dataset oluşturma"""
    
    def __init__(self, config: RTX4070Config):
        self.config = config
    
    def load_and_process_data(self) -> DatasetDict:
        """CSV'den veri yükle ve training için hazırla"""
        print("📊 ESG dataset yükleniyor...")
        
        # CSV yükle
        df = pd.read_csv(self.config.dataset_path)
        print(f"📈 Yüklenen: {len(df)} kayıt")
        
        # RTX 4070 için boyut sınırla
        if len(df) > self.config.max_samples:
            df = self._stratified_sampling(df)
            print(f"📉 Optimize edildi: {len(df)} kayıt")
        
        # Training examples oluştur
        training_examples = []
        for _, row in df.iterrows():
            examples = self._create_training_examples(row)
            training_examples.extend(examples)
        
        print(f"🎓 {len(training_examples)} eğitim örneği oluşturuldu")
        
        # Dataset oluştur ve böl
        dataset = Dataset.from_list(training_examples)
        split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        return DatasetDict({
            'train': split_dataset['train'],
            'test': split_dataset['test']
        })
    
    def _stratified_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """ESG skorlarına göre stratified sampling"""
        # ESG kategorileri oluştur
        df['esg_category'] = pd.cut(
            df['total_esg_score'], 
            bins=[0, 40, 60, 80, 100], 
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )
        
        # Her kategoriden örnek al
        sampled_dfs = []
        samples_per_category = self.config.max_samples // 4
        
        for category in ['Poor', 'Average', 'Good', 'Excellent']:
            category_data = df[df['esg_category'] == category]
            if len(category_data) > 0:
                sample_size = min(samples_per_category, len(category_data))
                sampled = category_data.sample(n=sample_size, random_state=42)
                sampled_dfs.append(sampled)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _create_training_examples(self, row: pd.Series) -> List[Dict[str, str]]:
        """Her şirket için training examples oluştur"""
        company = row.get('Company', 'Unknown Company')
        ticker = row.get('Ticker', 'N/A')
        peer_group = row.get('Peer_group_root', 'Unknown')
        country = row.get('Country', 'Unknown')
        
        total_esg = row.get('total_esg_score', 0)
        env_score = row.get('environment_score', 0)
        social_score = row.get('social_score', 0)
        gov_score = row.get('governance_score', 0)
        
        examples = []
        
        # 1. ESG Analizi
        instruction1 = f"{company} şirketinin ESG performansını analiz et"
        input1 = f"Şirket: {company} ({ticker})\nSektör: {peer_group}\nÜlke: {country}"
        output1 = self._generate_esg_analysis(company, total_esg, env_score, social_score, gov_score)
        
        # SFTTrainer için 'text' formatı
        text1 = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{instruction1}
{input1}<|assistant|>
{output1}<|endoftext|>"""
        
        examples.append({'text': text1})
        
        # 2. İyileştirme Önerisi (düşük skor için)
        if total_esg < 70:
            instruction2 = f"{company} için ESG iyileştirme stratejisi öner"
            input2 = f"Mevcut ESG: {total_esg:.1f}/100\nZayıf alan: {self._get_weakest_area(env_score, social_score, gov_score)}"
            output2 = self._generate_improvement_plan(company, total_esg, env_score, social_score, gov_score)
            
            text2 = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{instruction2}
{input2}<|assistant|>
{output2}<|endoftext|>"""
            
            examples.append({'text': text2})
        
        return examples
    
    def _generate_esg_analysis(self, company: str, total_esg: float, env: float, social: float, gov: float) -> str:
        """ESG analizi metni oluştur"""
        return f"""{company} ESG Skoru: {total_esg:.1f}/100

🔍 Detay Skorlar:
• Çevresel: {env:.1f}/100 ({self._get_level(env)})
• Sosyal: {social:.1f}/100 ({self._get_level(social)})
• Yönetim: {gov:.1f}/100 ({self._get_level(gov)})

📊 Değerlendirme: {self._get_assessment(total_esg)}
💡 Öneri: {self._get_recommendation(env, social, gov)}"""
    
    def _generate_improvement_plan(self, company: str, total_esg: float, env: float, social: float, gov: float) -> str:
        """İyileştirme planı oluştur"""
        target_score = min(total_esg + 20, 85)
        
        return f"""{company} İyileştirme Planı:

🎯 Öncelik: {self._get_priority_area(env, social, gov)}
⚡ Hızlı Adımlar:
{self._get_quick_actions(env, social, gov)}
📈 Hedef: {target_score:.1f}/100 (12-18 ay)"""
    
    def _get_level(self, score: float) -> str:
        """Skor seviyesi"""
        if score >= 80: return "Mükemmel"
        elif score >= 60: return "İyi"
        elif score >= 40: return "Orta"
        else: return "Zayıf"
    
    def _get_assessment(self, score: float) -> str:
        """Genel değerlendirme"""
        if score >= 80: return "ESG lideri"
        elif score >= 60: return "Güçlü performans"
        elif score >= 40: return "Gelişim gerekli"
        else: return "Acil aksiyon"
    
    def _get_recommendation(self, env: float, social: float, gov: float) -> str:
        """Temel öneri"""
        lowest = min(env, social, gov)
        if lowest == env: return "Çevresel etki azaltımına odaklanın"
        elif lowest == social: return "Sosyal sorumluluk artırın"
        else: return "Kurumsal şeffaflığı geliştirin"
    
    def _get_weakest_area(self, env: float, social: float, gov: float) -> str:
        """En zayıf alan"""
        scores = [('Çevresel', env), ('Sosyal', social), ('Yönetim', gov)]
        return min(scores, key=lambda x: x[1])[0]
    
    def _get_priority_area(self, env: float, social: float, gov: float) -> str:
        """Öncelik alanı"""
        return f"{self._get_weakest_area(env, social, gov)} alanda iyileştirme"
    
    def _get_quick_actions(self, env: float, social: float, gov: float) -> str:
        """Hızlı aksiyonlar"""
        actions = []
        if env < 60: actions.append("• Enerji verimliliği artırın")
        if social < 60: actions.append("• Çalışan memnuniyeti ölçün")
        if gov < 60: actions.append("• Etik politikalar oluşturun")
        return "\n".join(actions[:2])


class GraniteModelManager:
    """Granite model yükleme ve kurulum"""
    
    def __init__(self, config: RTX4070Config):
        self.config = config
        self.model = None
        self.tokenizer = None
    
    def setup_model(self) -> Tuple[any, any]:
        """Model ve tokenizer'ı kur"""
        print("🤖 RTX 4070 için Granite model hazırlanıyor...")
        
        # Memory kontrolü
        GPUMemoryManager.print_memory_stats()
        memory_stats = GPUMemoryManager.check_memory()
        
        if memory_stats.get('free', 0) < 6.0:
            print("⚠️ Yetersiz VRAM! Diğer programları kapatın.")
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_checkpoint,
            token=os.getenv('HF_TOKEN'),
            trust_remote_code=True,
            use_fast=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )
        
        # Model yükleme
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_checkpoint,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.getenv('HF_TOKEN'),
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_cache=False
        )
        
        # LoRA hazırlığı
        self.model = prepare_model_for_kbit_training(
            self.model,
            use_gradient_checkpointing=True
        )
        
        # LoRA konfigürasyonu
        peft_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, peft_config)
        
        # Final kontrol
        GPUMemoryManager.print_memory_stats()
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer


class RTX4070MemoryCallback(TrainerCallback):
    """RTX 4070 için memory monitoring callback"""
    
    def __init__(self, memory_limit_gb: float = 7.5):
        self.memory_limit_gb = memory_limit_gb
    
    def on_step_end(self, args, state, control, **kwargs):
        """Her step sonunda memory kontrolü"""
        if state.global_step % 10 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            if allocated > self.memory_limit_gb:
                print(f"⚠️ High VRAM usage: {allocated:.2f} GB")


class GraniteFineTuner:
    """Ana fine-tuning class"""
    
    def __init__(self, config: RTX4070Config):
        self.config = config
        self.data_processor = ESGDataProcessor(config)
        self.model_manager = GraniteModelManager(config)
    
    def run_training(self):
        """Complete training pipeline"""
        print("🎮 RTX 4070 Granite ESG Fine-tuning")
        print("=" * 50)
        
        # GPU kurulumu
        if not GPUMemoryManager.setup_gpu(self.config.memory_fraction):
            return False
        
        print("\n🎯 1. Dataset hazırlanıyor...")
        dataset = self.data_processor.load_and_process_data()
        
        print("\n🎯 2. Model yükleniyor...")
        model, tokenizer = self.model_manager.setup_model()
        
        print("\n🎯 3. Training başlatılıyor...")
        self._train_model(model, tokenizer, dataset)
        
        print("\n🎯 4. Model test ediliyor...")
        self._test_model()
        
        return True
    
    def _train_model(self, model, tokenizer, dataset):
        """Model training"""
        # Output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            optim="paged_adamw_8bit",
            max_grad_norm=1.0,
            dataloader_num_workers=0,
            save_total_limit=2,
            tf32=True if torch.cuda.is_available() else False,
            dataloader_drop_last=True
        )
        
        # Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=training_args
        )
        
        # Memory callback
        trainer.add_callback(RTX4070MemoryCallback(self.config.max_memory_gb))
        
        # Cache temizle
        torch.cuda.empty_cache()
        gc.collect()
        
        # Training
        start_time = datetime.now()
        print(f"⏰ Training başladı: {start_time}")
        print("💡 RTX 4070 8GB için optimize edilmiş ayarlar")
        
        try:
            trainer.train()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("❌ GPU Memory yetersiz!")
                print("💡 Çözümler:")
                print("  - Diğer GPU programlarını kapatın")
                print("  - gradient_accumulation_steps=32 yapın")
                print("  - num_train_epochs=1 yapın")
                raise
            else:
                raise
        
        # Save
        end_time = datetime.now()
        training_time = end_time - start_time
        
        trainer.save_model(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        # Stats
        print(f"✅ RTX 4070 Training tamamlandı!")
        print(f"⏱️ Süre: {training_time}")
        print(f"📁 Model: {self.config.output_dir}")
        
        # Config kaydet
        self._save_training_config(training_time, len(dataset["train"]))
        
        # Final memory
        GPUMemoryManager.print_memory_stats()
    
    def _save_training_config(self, training_time, dataset_size):
        """Training konfigürasyonunu kaydet"""
        config_info = {
            "base_model": self.config.model_checkpoint,
            "output_dir": self.config.output_dir,
            "training_time": str(training_time),
            "dataset_size": dataset_size,
            "gpu": "RTX 4070 8GB",
            "optimizations": [
                "4-bit quantization",
                f"LoRA r={self.config.lora_r}",
                f"batch_size={self.config.batch_size}",
                f"gradient_accumulation={self.config.gradient_accumulation_steps}",
                "minimal_sft_trainer"
            ],
            "completion_date": datetime.now().isoformat(),
            "config": {
                "epochs": self.config.num_train_epochs,
                "learning_rate": self.config.learning_rate,
                "max_samples": self.config.max_samples
            }
        }
        
        config_path = f"{self.config.output_dir}/rtx4070_training_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_info, f, indent=2, ensure_ascii=False)
    
    def _test_model(self):
        """Model testi"""
        print("🧪 Model test ediliyor...")
        
        test_prompts = [
            {
                "instruction": "Apple şirketinin ESG performansını analiz et",
                "input": "Şirket: Apple Inc (AAPL)\nSektör: Technology\nÜlke: USA"
            },
            {
                "instruction": "Tesla için ESG iyileştirme stratejisi öner",
                "input": "Mevcut ESG: 65/100\nZayıf alan: Çevresel"
            }
        ]
        
        try:
            from peft import PeftModel
            
            # Base model
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )
            
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_checkpoint,
                quantization_config=bnb_config,
                device_map="auto",
                token=os.getenv('HF_TOKEN'),
                torch_dtype=torch.float16
            )
            
            # Fine-tuned model
            model = PeftModel.from_pretrained(base_model, self.config.output_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.config.output_dir)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test
            for i, prompt_data in enumerate(test_prompts):
                print(f"\n🔍 Test {i+1}:")
                print(f"❓ Soru: {prompt_data['instruction']}")
                
                full_prompt = f"""<|system|>ESG uzmanısın. Kısa ve net yanıt ver.<|user|>
{prompt_data['instruction']}
{prompt_data['input']}<|assistant|>
"""
                
                inputs = tokenizer(
                    full_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(model.device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                response = tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                print(f"🤖 Yanıt: {response}")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Test hatası: {e}")


def main():
    """Ana fonksiyon"""
    # Konfigürasyon
    config = RTX4070Config()
    
    # Ön kontroller
    if not os.getenv('HF_TOKEN'):
        print("❌ HF_TOKEN bulunamadı!")
        print("💡 Çözüm: $env:HF_TOKEN = 'your_token_here'")
        return
    
    if not os.path.exists(config.dataset_path):
        print(f"❌ Dataset bulunamadı: {config.dataset_path}")
        print("💡 Önce veri analizi script'ini çalıştırın")
        return
    
    if not torch.cuda.is_available():
        print("❌ CUDA bulunamadı! RTX 4070 gerekli.")
        return
    
    # GPU kontrolü
    gpu_name = torch.cuda.get_device_name()
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if total_memory < 7:
        print("⚠️ VRAM 8GB altında! Sorun çıkabilir.")
        print(f"Mevcut VRAM: {total_memory:.1f} GB")
    
    # Fine-tuning başlat
    fine_tuner = GraniteFineTuner(config)
    
    try:
        success = fine_tuner.run_training()
        if success:
            print("\n🎉 RTX 4070 Fine-tuning başarılı!")
            print(f"📁 Model: {config.output_dir}")
        else:
            print("\n❌ Fine-tuning başarısız!")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()