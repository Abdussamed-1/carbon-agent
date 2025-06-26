# scripts/granite-langchain-adapter.py
"""
ESG Granite Model - LangChain Adapter
Mevcut ESGGraniteInference classını LangChain için uyarlar
"""

import argparse
import json
import sys
import os
import importlib.util

# Mevcut ESG Inference class'ını import et
spec = importlib.util.spec_from_file_location("esg_inference_api", "esg-inference-api.py")
esg_inference_api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esg_inference_api)
ESGGraniteInference = esg_inference_api.ESGGraniteInference

class GraniteLangChainAdapter:
    """LangChain için mevcut ESG model'ini adapter"""
    
    def __init__(self, model_path='./models/granite-esg-rtx4070'):
        self.esg_inference = ESGGraniteInference(model_path)
        
    def generate_response(self, prompt, max_tokens=150, temperature=0.7):
        """Generic prompt için yanıt üret"""
        try:
            # Mevcut _generate_response metodunu kullan
            response = self.esg_inference._generate_response(prompt, max_tokens)
            return response
        except Exception as e:
            raise Exception(f"Response generation error: {e}")
    
    def smart_esg_analysis(self, prompt):
        """Prompt'a göre uygun ESG analizi seç"""
        prompt_lower = prompt.lower()
        
        # Şirket analizi
        if any(word in prompt_lower for word in ['analiz', 'performans', 'değerlendir']):
            if any(company in prompt_lower for company in ['tesla', 'apple', 'microsoft', 'google']):
                company_name = self._extract_company_name(prompt)
                return self.esg_inference.analyze_company_esg(
                    company=company_name,
                    ticker="",
                    sector="Technology",
                    country="USA"
                )
        
        # İyileştirme önerisi
        if any(word in prompt_lower for word in ['iyileştir', 'öneri', 'strateji', 'geliştir']):
            company_name = self._extract_company_name(prompt)
            return self.esg_inference.suggest_esg_improvements(
                company=company_name,
                current_score=60,  # Default score
                weak_area="Genel"
            )
        
        # Çevresel değerlendirme
        if any(word in prompt_lower for word in ['çevre', 'environmental', 'karbon', 'emisyon']):
            company_name = self._extract_company_name(prompt)
            return self.esg_inference.evaluate_environmental_performance(
                company=company_name,
                env_score=70  # Default score
            )
        
        # Generic response
        return self.generate_response(prompt)
    
    def _extract_company_name(self, prompt):
        """Prompt'tan şirket ismini çıkar"""
        common_companies = {
            'tesla': 'Tesla Inc',
            'apple': 'Apple Inc',
            'microsoft': 'Microsoft Corporation',
            'google': 'Google LLC',
            'amazon': 'Amazon.com Inc',
            'meta': 'Meta Platforms Inc',
            'netflix': 'Netflix Inc'
        }
        
        prompt_lower = prompt.lower()
        for key, value in common_companies.items():
            if key in prompt_lower:
                return value
        
        # Şirket bulunamazsa generic
        return "Unknown Company"

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Granite ESG LangChain Adapter')
    parser.add_argument('--model-path', default='./models/granite-esg-rtx4070', help='Fine-tuned model path')
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=150, help='Max tokens')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser.add_argument('--smart-mode', action='store_true', help='Use smart ESG analysis')
    
    args = parser.parse_args()
    
    try:
        # Adapter oluştur
        adapter = GraniteLangChainAdapter(args.model_path)
        
        # Response üret
        if args.smart_mode:
            response = adapter.smart_esg_analysis(args.prompt)
        else:
            response = adapter.generate_response(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature
            )
        
        # JSON output (LangChain için)
        result = {
            "success": True,
            "response": response,
            "model_path": args.model_path,
            "prompt": args.prompt,
            "smart_mode": args.smart_mode
        }
        
        print(json.dumps(result, ensure_ascii=False))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "model_path": args.model_path,
            "prompt": args.prompt
        }
        
        print(json.dumps(error_result, ensure_ascii=False))
        sys.exit(1)

if __name__ == "__main__":
    main()