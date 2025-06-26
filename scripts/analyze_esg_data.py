# scripts/analyze_esg_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ESGDataAnalyzer:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.df = None
        
    def load_data(self):
        """CSV dosyasını yükle"""
        print("📊 ESG verisi yükleniyor...")
        self.df = pd.read_csv(self.csv_path)
        print(f"✅ {len(self.df)} satır, {len(self.df.columns)} sütun yüklendi")
        return self.df
    
    def analyze_data_quality(self):
        """Veri kalitesini analiz et"""
        print("\n🔍 VERİ KALİTESİ ANALİZİ")
        print("=" * 50)
        
        # Temel bilgiler
        print(f"📈 Toplam kayıt sayısı: {len(self.df):,}")
        print(f"📊 Sütun sayısı: {len(self.df.columns)}")
        print(f"📅 Tarih aralığı: {self.df['Date'].min()} - {self.df['Date'].max()}")
        
        # Eksik değerler
        print("\n❌ EKSIK DEĞERLER:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        for col in self.df.columns:
            if missing_data[col] > 0:
                print(f"  {col}: {missing_data[col]:,} ({missing_percent[col]:.1f}%)")
        
        # Benzersiz değerler
        print("\n🏢 BENZERSIZ DEĞERLER:")
        print(f"  Benzersiz şirket sayısı: {self.df['Company'].nunique():,}")
        print(f"  Benzersiz ticker sayısı: {self.df['Ticker'].nunique():,}")
        print(f"  Benzersiz ülke sayısı: {self.df['Country'].nunique()}")
        print(f"  Benzersiz bölge sayısı: {self.df['Region'].nunique()}")
        print(f"  Benzersiz peer group sayısı: {self.df['Peer_group_root'].nunique()}")
        
        # Skor dağılımları
        print("\n📊 SKOR İSTATİSTİKLERİ:")
        score_columns = ['total_esg_score', 'environment_score', 'social_score', 'governance_score']
        for col in score_columns:
            if col in self.df.columns:
                stats = self.df[col].describe()
                print(f"  {col}:")
                print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"    Ortalama: {stats['mean']:.2f}, Medyan: {stats['50%']:.2f}")
        
        return missing_data, missing_percent
    
    def detect_anomalies(self):
        """Anormal değerleri tespit et"""
        print("\n⚠️ ANOMALI TESPİTİ")
        print("=" * 50)
        
        anomalies = {}
        score_columns = ['total_esg_score', 'environment_score', 'social_score', 'governance_score']
        
        for col in score_columns:
            if col in self.df.columns:
                # Negatif değerler
                negative = self.df[self.df[col] < 0]
                if len(negative) > 0:
                    anomalies[f'{col}_negative'] = len(negative)
                    print(f"  {col}: {len(negative)} negatif değer")
                
                # 100'den büyük değerler (ESG skorları genelde 0-100 arası)
                over_100 = self.df[self.df[col] > 100]
                if len(over_100) > 0:
                    anomalies[f'{col}_over_100'] = len(over_100)
                    print(f"  {col}: {len(over_100)} değer 100'den büyük")
                
                # Aşırı değerler (IQR yöntemi)
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                if len(outliers) > 0:
                    anomalies[f'{col}_outliers'] = len(outliers)
                    print(f"  {col}: {len(outliers)} aşırı değer")
        
        # Mükerrer kayıtlar
        duplicates = self.df.duplicated(['Company', 'Date']).sum()
        if duplicates > 0:
            anomalies['duplicates'] = duplicates
            print(f"  Mükerrer kayıt: {duplicates}")
        
        return anomalies
    
    def clean_data(self):
        """Veriyi temizle"""
        print("\n🧹 VERİ TEMİZLEME")
        print("=" * 50)
        
        original_len = len(self.df)
        cleaned_df = self.df.copy()
        
        # 1. Eksik temel bilgilere sahip kayıtları kaldır
        essential_columns = ['Company', 'Ticker', 'total_esg_score']
        before_essential = len(cleaned_df)
        cleaned_df = cleaned_df.dropna(subset=essential_columns)
        removed_essential = before_essential - len(cleaned_df)
        if removed_essential > 0:
            print(f"  ❌ Temel bilgi eksik {removed_essential} kayıt kaldırıldı")
        
        # 2. Negatif ESG skorlarını kaldır
        score_columns = ['total_esg_score', 'environment_score', 'social_score', 'governance_score']
        for col in score_columns:
            if col in cleaned_df.columns:
                before_negative = len(cleaned_df)
                cleaned_df = cleaned_df[cleaned_df[col] >= 0]
                removed_negative = before_negative - len(cleaned_df)
                if removed_negative > 0:
                    print(f"  ❌ {col} negatif {removed_negative} kayıt kaldırıldı")
        
        # 3. 100'den büyük skorları 100 ile sınırla
        for col in score_columns:
            if col in cleaned_df.columns:
                over_100_count = (cleaned_df[col] > 100).sum()
                if over_100_count > 0:
                    cleaned_df[col] = cleaned_df[col].clip(upper=100)
                    print(f"  🔧 {col} {over_100_count} değer 100 ile sınırlandı")
        
        # 4. Mükerrer kayıtları kaldır
        before_duplicates = len(cleaned_df)
        cleaned_df = cleaned_df.drop_duplicates(['Company', 'Date'])
        removed_duplicates = before_duplicates - len(cleaned_df)
        if removed_duplicates > 0:
            print(f"  ❌ {removed_duplicates} mükerrer kayıt kaldırıldı")
        
        # 5. Eksik değerleri doldur
        # Categorical columns
        categorical_fills = {
            'Country': 'Unknown',
            'Region': 'Unknown', 
            'Peer_group_root': 'Other'
        }
        
        for col, fill_value in categorical_fills.items():
            if col in cleaned_df.columns:
                filled_count = cleaned_df[col].isnull().sum()
                if filled_count > 0:
                    cleaned_df[col] = cleaned_df[col].fillna(fill_value)
                    print(f"  🔧 {col} {filled_count} eksik değer '{fill_value}' ile dolduruldu")
        
        # Numerical columns - median ile doldur
        numerical_columns = ['environment_score', 'social_score', 'governance_score',
                           'scl_environment_score', 'scl_social_score', 'scl_governance_score',
                           'wght_environment_score', 'wght_social_score', 'wght_governance_score']
        
        for col in numerical_columns:
            if col in cleaned_df.columns:
                filled_count = cleaned_df[col].isnull().sum()
                if filled_count > 0:
                    median_value = cleaned_df[col].median()
                    cleaned_df[col] = cleaned_df[col].fillna(median_value)
                    print(f"  🔧 {col} {filled_count} eksik değer median ({median_value:.2f}) ile dolduruldu")
        
        # Sonuçları göster
        final_len = len(cleaned_df)
        removed_total = original_len - final_len
        print(f"\n✅ Temizleme tamamlandı:")
        print(f"  Başlangıç: {original_len:,} kayıt")
        print(f"  Bitiş: {final_len:,} kayıt")
        print(f"  Kaldırılan: {removed_total:,} kayıt ({(removed_total/original_len)*100:.1f}%)")
        
        return cleaned_df
    
    def create_training_dataset(self, cleaned_df, output_path):
        """Fine-tuning için optimize edilmiş dataset oluştur"""
        print("\n🎓 FINE-TUNING DATASET OLUŞTURMA")
        print("=" * 50)
        
        # Skorları kategorize et (daha iyi örnekler için)
        def categorize_score(score):
            if score >= 80: return "Excellent"
            elif score >= 60: return "Good"
            elif score >= 40: return "Average"
            else: return "Poor"
        
        # Her şirket için en son kaydı al (çoklu zaman serisi varsa)
        latest_data = cleaned_df.sort_values('Date').groupby('Company').tail(1).reset_index(drop=True)
        
        # Kategori sütunları ekle
        latest_data['esg_category'] = latest_data['total_esg_score'].apply(categorize_score)
        latest_data['env_category'] = latest_data['environment_score'].apply(categorize_score)
        latest_data['social_category'] = latest_data['social_score'].apply(categorize_score)
        latest_data['gov_category'] = latest_data['governance_score'].apply(categorize_score)
        
        # Çeşitlilik için stratified sampling
        # Her kategoriden örnek al
        training_samples = []
        
        for category in ['Excellent', 'Good', 'Average', 'Poor']:
            category_data = latest_data[latest_data['esg_category'] == category]
            if len(category_data) > 0:
                # Her kategoriden maksimum 500 örnek al
                sample_size = min(500, len(category_data))
                sampled = category_data.sample(n=sample_size, random_state=42)
                training_samples.append(sampled)
        
        final_training_data = pd.concat(training_samples, ignore_index=True)
        
        # Kaydet
        final_training_data.to_csv(output_path, index=False)
        
        print(f"✅ Fine-tuning dataset oluşturuldu:")
        print(f"  Dosya: {output_path}")
        print(f"  Toplam örnek: {len(final_training_data):,}")
        print(f"  Kategori dağılımı:")
        
        category_dist = final_training_data['esg_category'].value_counts()
        for cat, count in category_dist.items():
            print(f"    {cat}: {count:,}")
        
        return final_training_data
    
    def generate_sample_questions(self, df, n_samples=50):
        """Fine-tuning için örnek sorular oluştur"""
        questions = []
        
        # Farklı soru türleri
        question_templates = [
            "{company} şirketinin ESG performansını analiz eder misin?",
            "{company} için ESG iyileştirme stratejisi önerir misin?",
            "{company} şirketinin çevresel performansı nasıl?",
            "{company} şirketinin sosyal sorumluluk skoru hakkında bilgi verir misin?",
            "{company} şirketinin kurumsal yönetim kalitesi nasıl?",
            "{region} bölgesindeki ESG performansı nasıl?",
            "{country} ülkesindeki şirketlerin ESG durumu hakkında bilgi verir misin?",
            "{peer_group} sektöründe ESG liderleri kimler?",
        ]
        
        sample_data = df.sample(n=min(n_samples, len(df)), random_state=42)
        
        for _, row in sample_data.iterrows():
            template = np.random.choice(question_templates)
            
            if "{company}" in template:
                question = template.format(company=row['Company'])
            elif "{region}" in template:
                question = template.format(region=row['Region'])
            elif "{country}" in template:
                question = template.format(country=row['Country'])
            elif "{peer_group}" in template:
                question = template.format(peer_group=row['Peer_group_root'])
            
            questions.append({
                'question': question,
                'company': row['Company'],
                'ticker': row['Ticker'],
                'total_esg_score': row['total_esg_score'],
                'context': 'fine_tuning_sample'
            })
        
        return questions


def main():
    """Ana fonksiyon"""
    csv_path = "./data/carbon-data/esg_scores.csv"
    
    try:
        # Analyzer oluştur
        analyzer = ESGDataAnalyzer(csv_path)
        
        # Veriyi yükle
        df = analyzer.load_data()
        
        # Analiz et
        analyzer.analyze_data_quality()
        analyzer.detect_anomalies()
        
        # Temizle
        cleaned_df = analyzer.clean_data()
        
        # Fine-tuning dataset oluştur
        training_data = analyzer.create_training_dataset(
            cleaned_df, 
            "./data/esg_training_data.csv"
        )
        
        # Örnek sorular oluştur
        sample_questions = analyzer.generate_sample_questions(training_data)
        
        # Sonuçları kaydet
        import json
        with open("./data/sample_questions.json", "w", encoding="utf-8") as f:
            json.dump(sample_questions, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 Analiz tamamlandı!")
        print(f"📁 Temizlenmiş veri: ./data/esg_training_data.csv")
        print(f"📁 Örnek sorular: ./data/sample_questions.json")
        
    except Exception as e:
        print(f"❌ Hata: {e}")


if __name__ == "__main__":
    main()