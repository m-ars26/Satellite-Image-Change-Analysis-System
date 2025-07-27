"""
🧪 Change Detection Standalone Test
Import sorunlarını çözmek için ayrı test dosyası
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')

# Modülleri import et
from preprocessor import ImagePreprocessor
from change_detector import ChangeDetector

def main():
    """Ana test fonksiyonu"""
    print("🧪 ChangeDetector Test Başlatıldı...")
    
    # Test görüntülerinin yollarını kontrol ediyorum
    test_paths = [
        "data/before/test_image.jpg",
        "data/after/test_image.jpg"
    ]
    
    for path in test_paths:
        if not os.path.exists(path):
            print(f"❌ Test görüntüsü bulunamadı: {path}")
            print("Önce demo.py çalıştırarak test görüntülerini oluşturun!")
            return
    
    # Sistem bileşenlerini başlatıyorum
    print("🔧 Sistem bileşenleri başlatılıyor...")
    preprocessor = ImagePreprocessor(target_size=(400, 400))
    detector = ChangeDetector(methods=['statistical', 'morphological', 'feature_based'])
    
    # Görüntüleri yüklüyorum ve ön işlem uyguluyorum
    print("🔄 Görüntüler hazırlanıyor...")
    try:
        img_before = preprocessor.load_and_preprocess(test_paths[0])
        img_after = preprocessor.load_and_preprocess(test_paths[1])
        print("✅ Görüntüler yüklendi")
    except Exception as e:
        print(f"❌ Görüntü yükleme hatası: {e}")
        return
    
    # Görüntü hizalama
    print("🔄 Görüntüler hizalanıyor...")
    try:
        img_before_aligned, img_after_aligned = preprocessor.align_images(img_before, img_after)
        print("✅ Görüntüler hizalandı")
    except Exception as e:
        print(f"⚠️ Hizalama hatası (devam ediliyor): {e}")
        img_before_aligned, img_after_aligned = img_before, img_after
    
    # Değişiklik tespiti
    print("🔍 Değişiklik tespiti yapılıyor...")
    try:
        results = detector.detect_changes(img_before_aligned, img_after_aligned)
        print("✅ Değişiklik tespiti tamamlandı")
    except Exception as e:
        print(f"❌ Change detection hatası: {e}")
        return
    
    # Sonuçları yazdırma
    print("\n📊 Tespit Sonuçları:")
    if 'statistics' in results:
        stats = results['statistics']
        print(f"   💫 Değişiklik yüzdesi: {stats.get('change_percentage', 0):.2f}%")
        print(f"   🎯 Ortalama güven: {stats.get('mean_confidence', 0):.3f}")
        print(f"   📍 Değişiklik bölgesi sayısı: {stats.get('num_change_regions', 0)}")
    
    # Bölge analizleri
    if 'change_regions' in results and results['change_regions']:
        print("   🏗️ Tespit edilen değişiklik tipleri:")
        for region in results['change_regions'][:5]:  # İlk 5 bölge
            print(f"      - {region['change_type']}: {region['area']} piksel")
    
    # Görselleştirme
    print("🎨 Sonuçlar görselleştiriliyor...")
    try:
        create_visualization(img_before_aligned, img_after_aligned, results)
        print("✅ Görselleştirme tamamlandı")
    except Exception as e:
        print(f"⚠️ Görselleştirme hatası: {e}")
        # Hata durumunda basit görselleştirme yapıyorum
        simple_visualization(img_before_aligned, img_after_aligned, results)
    
    print("\n🎉 Test tamamlandı!")
    print("📂 Sonuçları results/ klasöründe kontrol edebilirsiniz.")

def create_visualization(img_before, img_after, results):
    """Detaylı görselleştirme"""
    plt.figure(figsize=(18, 12))
    
    # Orijinal görüntüler
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    plt.title("Before Image")
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    plt.title("After Image")
    plt.axis('off')
    
    # Ana sonuçlar
    if results.get('change_mask') is not None:
        plt.subplot(3, 4, 3)
        plt.imshow(results['change_mask'], cmap='hot')
        plt.title("Change Mask")
        plt.axis('off')
    
    if results.get('confidence_map') is not None:
        plt.subplot(3, 4, 4)
        im = plt.imshow(results['confidence_map'], cmap='viridis')
        plt.title("Confidence Map")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
    
    # Yöntem sonuçları
    if 'method_results' in results:
        method_names = list(results['method_results'].keys())
        for i, method in enumerate(method_names[:4]):
            if i + 5 <= 12:  # subplot sınırı
                plt.subplot(3, 4, 5 + i)
                plt.imshow(results['method_results'][method]['mask'], cmap='gray')
                plt.title(f"{method.replace('_', ' ').title()}")
                plt.axis('off')
    
    # Overlay görünüm
    if results.get('change_mask') is not None:
        plt.subplot(3, 4, 9)
        overlay = img_after.copy()
        overlay[results['change_mask'] > 0] = [0, 0, 255]  # Kırmızı
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Changes Overlay")
        plt.axis('off')
    
    # İstatistik grafiği
    if 'statistics' in results:
        plt.subplot(3, 4, 10)
        stats = results['statistics']
        stat_names = ['change_percentage', 'mean_confidence', 'num_change_regions']
        stat_values = [stats.get(name, 0) for name in stat_names]
        
        # Değerleri normalize et görselleştirme için
        normalized_values = []
        for i, val in enumerate(stat_values):
            if i == 0:  # change_percentage - olduğu gibi
                normalized_values.append(val)
            elif i == 1:  # mean_confidence - 100 ile çarp
                normalized_values.append(val * 100)
            else:  # num_change_regions - olduğu gibi
                normalized_values.append(val)
        
        plt.bar(range(len(stat_names)), normalized_values)
        plt.xticks(range(len(stat_names)), 
                  ['Change %', 'Confidence x100', 'Regions'], rotation=45)
        plt.title("Detection Statistics")
    
    # Bölge boyut dağılımı
    if results.get('change_regions'):
        plt.subplot(3, 4, 11)
        areas = [region['area'] for region in results['change_regions']]
        if areas:
            plt.hist(areas, bins=min(10, len(areas)), alpha=0.7)
            plt.title("Change Region Sizes")
            plt.xlabel("Area (pixels)")
            plt.ylabel("Count")
    
    # Güven dağılımı
    if results.get('confidence_map') is not None:
        plt.subplot(3, 4, 12)
        conf_flat = results['confidence_map'].flatten()
        conf_nonzero = conf_flat[conf_flat > 0.01]  # Çok küçük değerleri filtrele
        if len(conf_nonzero) > 0:
            plt.hist(conf_nonzero, bins=50, alpha=0.7)
            plt.title("Confidence Distribution")
            plt.xlabel("Confidence")
            plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("results/change_detection_test.png", dpi=150, bbox_inches='tight')
    plt.show()

def simple_visualization(img_before, img_after, results):
    """Basit görselleştirme (hata durumunda)"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB))
    plt.title("Before")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB))
    plt.title("After")
    plt.axis('off')
    
    if results.get('change_mask') is not None:
        plt.subplot(1, 3, 3)
        plt.imshow(results['change_mask'], cmap='hot')
        plt.title("Changes")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("results/simple_change_detection.png", dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()