"""
🛰️ Uydu Görüntü Değişiklik Tespit Sistemi - Final Demo
Tüm sistem bileşenlerinin entegre demosu
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

sys.path.append('src')

from preprocessor import ImagePreprocessor
from change_detector import ChangeDetector
from visualizer import ResultVisualizer
from utils import create_directory, get_image_files, print_progress

class SatelliteChangeDetectionSystem:
    """
    Uydu görüntü değişiklik tespit sistemi
    """
    
    def __init__(self, 
                 target_size=(1024, 1024),
                 detection_methods=['statistical', 'morphological', 'feature_based'],
                 confidence_threshold=0.8,
                 visualization_style='modern'):
        """
        Args:
            target_size: Hedef görüntü boyutu
            detection_methods: Kullanılacak tespit yöntemleri
            confidence_threshold: Güven eşiği
            visualization_style: Görselleştirme stili
        """
        
        print("🛰️ Satellite Change Detection System Başlatılıyor...")
        
        # Sistem bileşenlerini başlatma
        self.preprocessor = ImagePreprocessor(target_size=target_size)
        self.detector = ChangeDetector(methods=detection_methods, 
                                     confidence_threshold=confidence_threshold)
        self.visualizer = ResultVisualizer(style=visualization_style)
        
        # Ayarları kaydetme
        self.target_size = target_size
        self.detection_methods = detection_methods
        self.confidence_threshold = confidence_threshold
        
        # Sonuç klasörünü oluşturma
        create_directory('results')
        
        print("✅ Sistem başarıyla başlatıldı!")
        self._print_system_info()
    
    def _print_system_info(self):
        """Sistem bilgilerini yazdır"""
        print("\n📋 Sistem Konfigürasyonu:")
        print(f"   🖼️ Hedef boyut: {self.target_size}")
        print(f"   🧠 Tespit yöntemleri: {self.detection_methods}")
        print(f"   🎯 Güven eşiği: {self.confidence_threshold}")
        print(f"   🎨 Görselleştirme: {self.visualizer.style}")
    
    def process_image_pair(self, 
                          before_path: str, 
                          after_path: str,
                          output_name: str = None) -> dict:
        """
        İki görüntüyü işledim
        
        Args:
            before_path: Önceki görüntü yolu
            after_path: Sonraki görüntü yolu
            output_name: Çıktı dosya adı
            
        Returns:
            İşlem sonuçları
        """
        
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"analysis_{timestamp}"
        
        print(f"\n🔄 İşlem başlatıldı: {output_name}")
        print("=" * 50)
        
        # 1. Görüntüyü yükleme ve ön işleme
        print("📁 1/5 - Görüntüler yükleniyor...")
        try:
            img_before = self.preprocessor.load_and_preprocess(before_path)
            img_after = self.preprocessor.load_and_preprocess(after_path)
            print("✅ Görüntüler başarıyla yüklendi ve ön işlendi")
        except Exception as e:
            print(f"❌ Görüntü yükleme hatası: {e}")
            return {'error': str(e)}
        
        # 2. Görüntüyü hizalama
        print("🔧 2/5 - Görüntüler hizalanıyor...")
        try:
            img_before_aligned, img_after_aligned = self.preprocessor.align_images(
                img_before, img_after)
            
            # Hizalama kalitesini kontrol etme
            alignment_quality = self.preprocessor.calculate_alignment_quality(
                img_before_aligned, img_after_aligned)
            print(f"✅ Hizalama tamamlandı (Kalite: {alignment_quality:.3f})")
            
        except Exception as e:
            print(f"⚠️ Hizalama hatası, devam ediliyor: {e}")
            img_before_aligned, img_after_aligned = img_before, img_after
            alignment_quality = 0.0
        
        # 3. Değişiklik tespiti
        print("🔍 3/5 - Değişiklik tespiti yapılıyor...")
        try:
            results = self.detector.detect_changes(img_before_aligned, img_after_aligned)
            print("✅ Değişiklik tespiti tamamlandı")
            
            # Sonuçları özetleme
            self._print_detection_summary(results)
            
        except Exception as e:
            print(f"❌ Değişiklik tespiti hatası: {e}")
            return {'error': str(e)}
        
        # 4. Görselleştirme
        print("🎨 4/5 - Sonuçlar görselleştiriliyor...")
        try:
            report_title = f"Change Detection Report - {output_name}"
            report_path = self.visualizer.create_comprehensive_report(
                img_before_aligned, img_after_aligned, results, report_title)
            print(f"✅ Rapor oluşturuldu: {report_path}")
            
        except Exception as e:
            print(f"⚠️ Görselleştirme hatası: {e}")
            report_path = None
        
        # 5. Sonuçları kaydetme
        print("💾 5/5 - Sonuçlar kaydediliyor...")
        try:
            json_path = self.visualizer.save_results_json(results, f"{output_name}.json")
            print(f"✅ JSON kaydedildi: {json_path}")
            
        except Exception as e:
            print(f"⚠️ JSON kaydetme hatası: {e}")
            json_path = None
        
        # Final sonuçları
        final_results = {
            'input_files': {
                'before': before_path,
                'after': after_path
            },
            'preprocessing': {
                'target_size': self.target_size,
                'alignment_quality': alignment_quality
            },
            'detection_results': results,
            'output_files': {
                'report': report_path,
                'json': json_path
            },
            'timestamp': datetime.now().isoformat(),
            'success': True
        }
        
        print("\n🎉 İşlem başarıyla tamamlandı!")
        return final_results
    
    def _print_detection_summary(self, results):
        """Tespit sonuçlarının özetini yazdır"""
        if 'statistics' in results:
            stats = results['statistics']
            print("\n📊 Tespit Özeti:")
            print(f"   💫 Değişiklik yüzdesi: {stats.get('change_percentage', 0):.2f}%")
            print(f"   🎯 Ortalama güven: {stats.get('mean_confidence', 0):.3f}")
            print(f"   📍 Bölge sayısı: {stats.get('num_change_regions', 0)}")
            
            if results.get('change_regions'):
                print("   🏗️ Başlıca değişiklik tipleri:")
                type_counts = {}
                for region in results['change_regions']:
                    change_type = region['change_type']
                    type_counts[change_type] = type_counts.get(change_type, 0) + 1
                
                for change_type, count in sorted(type_counts.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]:
                    print(f"      - {change_type}: {count} bölge")
    
    def batch_process(self, before_dir: str, after_dir: str) -> list:
        """
        Toplu işlem - klasördeki tüm görüntü çiftlerini işle
        
        Args:
            before_dir: Önceki görüntüler klasörü
            after_dir: Sonraki görüntüler klasörü
            
        Returns:
            Tüm sonuçlar listesi
        """
        
        print(f"\n🗂️ Toplu İşlem Başlatıldı")
        print(f"📁 Kaynak klasörler: {before_dir} → {after_dir}")
        
        # Dosyaları listeleme
        before_files = get_image_files(before_dir)
        after_files = get_image_files(after_dir)
        
        if not before_files or not after_files:
            print("❌ Görüntü dosyası bulunamadı!")
            return []
        
        print(f"📄 {len(before_files)} önce, {len(after_files)} sonra görüntüsü bulundu")
        
        # Dosya çiftlerini eşleştirme
        pairs = self._match_image_pairs(before_files, after_files)
        
        if not pairs:
            print("❌ Eşleşen görüntü çifti bulunamadı!")
            return []
        
        print(f"🔗 {len(pairs)} görüntü çifti eşleştirildi")
        
        # Dosya çiftlerini eşleme
        all_results = []
        for i, (before_file, after_file) in enumerate(pairs):
            print(f"\n📊 İşleniyor: {i+1}/{len(pairs)}")
            
            output_name = f"batch_{i+1:03d}_{os.path.splitext(os.path.basename(before_file))[0]}"
            
            try:
                result = self.process_image_pair(before_file, after_file, output_name)
                all_results.append(result)
                
                print_progress(i+1, len(pairs), "Toplu İşlem")
                
            except Exception as e:
                print(f"❌ Çift {i+1} işlem hatası: {e}")
                all_results.append({'error': str(e), 'pair': (before_file, after_file)})
        
        print(f"\n✅ Toplu işlem tamamlandı! {len(all_results)} sonuç")
        return all_results
    
    def _match_image_pairs(self, before_files: list, after_files: list) -> list:
        """Görüntü dosyalarını eşleştir"""
        pairs = []
        
        # Basit eşleştirme: dosya adına göre
        before_names = {os.path.splitext(os.path.basename(f))[0]: f for f in before_files}
        after_names = {os.path.splitext(os.path.basename(f))[0]: f for f in after_files}
        
        for name in before_names:
            if name in after_names:
                pairs.append((before_names[name], after_names[name]))
        
        return pairs
    
    def create_sample_dataset(self, num_samples: int = 3):
        """Test için örnek veri seti oluştur"""
        
        print(f"🖼️ {num_samples} örnek görüntü çifti oluşturuluyor...")
        
        # Klasörleri oluşturma
        create_directory('data/before')
        create_directory('data/after')
        
        for i in range(num_samples):
            # Rastgele arka plan
            height, width = 400, 400
            
            # Before image
            before_img = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
            
            # Yapılar ekle
            cv2.rectangle(before_img, (50, 50), (150, 150), (200, 200, 200), -1)
            cv2.rectangle(before_img, (250, 100), (350, 200), (180, 180, 180), -1)
            cv2.circle(before_img, (200, 300), 40, (160, 160, 160), -1)
            
            # After image - değişikliklerle birlikte
            after_img = before_img.copy()
            
            if i == 0:  # İlk örnek - bina değişikliği
                cv2.rectangle(after_img, (50, 50), (150, 150), (255, 100, 100), -1)  # Kırmızı
                cv2.rectangle(after_img, (170, 170), (220, 220), (100, 255, 100), -1)  # Yeni yeşil
            elif i == 1:  # İkinci örnek - doğrusal özellik
                cv2.line(after_img, (0, 250), (400, 250), (100, 100, 255), 8)  # Mavi çizgi
                cv2.rectangle(after_img, (250, 100), (350, 200), (255, 255, 100), -1)  # Sarı
            else:  # Üçüncü örnek - alan geliştirme
                cv2.rectangle(after_img, (300, 250), (380, 330), (255, 150, 150), -1)
                cv2.circle(after_img, (200, 300), 40, (255, 200, 100), -1)  # Turuncu
            
            # Kaydet
            before_path = f"data/before/sample_{i+1:02d}.jpg"
            after_path = f"data/after/sample_{i+1:02d}.jpg"
            
            cv2.imwrite(before_path, before_img)
            cv2.imwrite(after_path, after_img)
            
            print(f"✅ Örnek {i+1}: {before_path} → {after_path}")
        
        print(f"🎉 {num_samples} örnek görüntü çifti oluşturuldu!")

def main():
    """Ana demo fonksiyonu"""
    
    parser = argparse.ArgumentParser(description='Satellite Change Detection System')
    parser.add_argument('--mode', choices=['demo', 'single', 'batch'], 
                       default='demo', help='Çalışma modu')
    parser.add_argument('--before', help='Önceki görüntü yolu')
    parser.add_argument('--after', help='Sonraki görüntü yolu')
    parser.add_argument('--before-dir', help='Önceki görüntüler klasörü')
    parser.add_argument('--after-dir', help='Sonraki görüntüler klasörü')
    parser.add_argument('--samples', type=int, default=3, help='Oluşturulacak örnek sayısı')
    
    args = parser.parse_args()
    
    print("🛰️ SATELLITE IMAGE CHANGE DETECTION SYSTEM")
    print("=" * 60)
    print("🎯 AI-Powered Change Detection for Satellite Imagery")
    print("👨‍💻 Developed by: Mehmet")
    print("📅 Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)
    
    # Sistemi başlatma
    system = SatelliteChangeDetectionSystem(
        target_size=(1024, 1024),
        detection_methods=['statistical', 'morphological', 'feature_based'],
        confidence_threshold=0.8,
        visualization_style='modern'
    )
    
    if args.mode == 'demo':
        # Demo modu - örnek veri oluşturma ve işleme
        print("\n🎬 DEMO MODU BAŞLATILIYOR...")
        
        # Örnek veri oluşturma
        system.create_sample_dataset(args.samples)
        
        # Örnekleri işleme
        print("\n🔄 Örnek görüntüler işleniyor...")
        results = system.batch_process('data/before', 'data/after')
        
        print(f"\n📈 DEMO SONUÇLARI:")
        successful = sum(1 for r in results if r.get('success', False))
        print(f"   ✅ Başarılı: {successful}/{len(results)}")
        print(f"   📁 Sonuçlar: results/ klasöründe")
        
    elif args.mode == 'single':
        # Tekli işleme
        if not args.before or not args.after:
            print("❌ --before ve --after parametreleri gerekli!")
            return
        
        print(f"\n🔍 TEK ÇİFT İŞLEME MODU")
        result = system.process_image_pair(args.before, args.after)
        
        if result.get('success'):
            print("\n✅ İşlem başarıyla tamamlandı!")
        else:
            print(f"\n❌ İşlem hatası: {result.get('error')}")
    
    elif args.mode == 'batch':
        # Toplu işleme
        if not args.before_dir or not args.after_dir:
            print("❌ --before-dir ve --after-dir parametreleri gerekli!")
            return
        
        print(f"\n📦 TOPLU İŞLEME MODU")
        results = system.batch_process(args.before_dir, args.after_dir)
        
        print(f"\n📊 TOPLU İŞLEM SONUÇLARI:")
        successful = sum(1 for r in results if r.get('success', False))
        print(f"   ✅ Başarılı: {successful}/{len(results)}")
        print(f"   ❌ Hatalı: {len(results) - successful}/{len(results)}")
    
    print("\n🎉 Program tamamlandı!")
    print("📂 Tüm sonuçlar 'results/' klasöründe!")
    print("\n💡 İpucu: Farklı modlar için --help parametresini kullanın")

if __name__ == "__main__":
    main()