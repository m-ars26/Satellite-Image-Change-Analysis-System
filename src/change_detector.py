"""
🛰️ Uydu Görüntü Değişiklik Tespit Modülü
Çoklu algoritma ile gelişmiş değişiklik tespiti
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy import ndimage
from typing import Tuple, Dict, List, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChangeDetector:
    """
    Gelişmiş değişiklik tespit sistemi
    Çoklu algoritma kullanarak güvenilir sonuçlar üretmeye çalıştım
    """
    
    def __init__(self, 
                 methods: List[str] = ['statistical', 'morphological', 'feature_based'],
                 confidence_threshold: float = 0.6):
        """
        Args:
            methods: Kullanılacak tespit yöntemleri
            confidence_threshold: Güven eşiği
        """
        self.methods = methods
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # CNN feature extractor (basit)
        self.feature_extractor = self._build_feature_extractor()
        
        logger.info(f"ChangeDetector başlatıldı. Methods: {methods}")
    
    def _build_feature_extractor(self) -> nn.Module:
        """Basit CNN feature extractor oluşturdum"""
        class SimpleFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                
            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = self.pool(x)
                x = F.relu(self.conv2(x))
                x = self.pool(x)
                x = F.relu(self.conv3(x))
                return x
        
        return SimpleFeatureExtractor().to(self.device)
    
    def detect_changes(self, 
                      image_before: np.ndarray, 
                      image_after: np.ndarray) -> Dict[str, Any]:
        """
        Ana değişiklik tespit fonksiyonu
        
        Args:
            image_before: Önceki görüntü
            image_after: Sonraki görüntü
            
        Returns:
            Tespit sonuçları
        """
        logger.info("Değişiklik tespiti başlatıldı...")
        
        results = {
            'change_mask': None,
            'confidence_map': None,
            'change_regions': [],
            'statistics': {},
            'method_results': {}
        }
        
        # Farklı yöntemlerle tespit yapıyorum
        method_masks = []
        method_confidences = []
        
        for method in self.methods:
            logger.info(f"Yöntem çalıştırılıyor: {method}")
            
            if method == 'statistical':
                mask, conf = self._statistical_change_detection(image_before, image_after)
            elif method == 'morphological':
                mask, conf = self._morphological_change_detection(image_before, image_after)
            elif method == 'feature_based':
                mask, conf = self._feature_based_change_detection(image_before, image_after)
            elif method == 'deep_learning':
                mask, conf = self._deep_learning_change_detection(image_before, image_after)
            else:
                continue
            
            method_masks.append(mask)
            method_confidences.append(conf)
            results['method_results'][method] = {'mask': mask, 'confidence': conf}
        
        # Sonuçları birleştiriyorum
        if method_masks:
            combined_mask, combined_confidence = self._combine_results(
                method_masks, method_confidences)
            
            results['change_mask'] = combined_mask
            results['confidence_map'] = combined_confidence
            
            # Değişiklik bölgelerini analiz ediyorum
            results['change_regions'] = self._analyze_change_regions(combined_mask)
            
            # İstatistikleri hesaplıyorum
            results['statistics'] = self._calculate_statistics(
                image_before, image_after, combined_mask, combined_confidence)
        
        logger.info("Değişiklik tespiti tamamlandı")
        return results
    
    def _statistical_change_detection(self, 
                                    img_before: np.ndarray, 
                                    img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """İstatistiksel değişiklik tespiti"""
        
        # Görüntüleri gri tonlamaya çevirdim
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        
        # Mutlak fark
        diff = cv2.absdiff(gray_before, gray_after)
        
        # Adaptive thresholding
        binary_mask = cv2.adaptiveThreshold(
            diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Morfolojik işlemler
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Güven haritası (fark yoğunluğuna göre)
        confidence_map = cv2.GaussianBlur(diff.astype(np.float32), (15, 15), 0)
        confidence_map = confidence_map / 255.0
        
        return cleaned, confidence_map
    
    def _morphological_change_detection(self, 
                                      img_before: np.ndarray, 
                                      img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Morfolojik değişiklik tespiti"""
        
        # RGB kanallarını ayrı ayrı işledim
        channels_before = cv2.split(img_before)
        channels_after = cv2.split(img_after)
        
        channel_masks = []
        
        for ch_before, ch_after in zip(channels_before, channels_after):
            # Fark hesapladım
            diff = cv2.absdiff(ch_before, ch_after)
            
            # Otsu thresholding
            _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morfolojik gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            
            channel_masks.append(gradient)
        
        # Kanalları birleştirdim
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(channel_masks[0], channel_masks[1]), 
                                      channel_masks[2])
        
        # Küçük gürültüleri temizledim
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        cleaned_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)
        
        # Güven haritası oluşturdum
        confidence_map = cv2.distanceTransform(cleaned_mask, cv2.DIST_L2, 5)
        confidence_map = cv2.normalize(confidence_map, None, 0, 1, cv2.NORM_MINMAX).astype(np.float32)
        
        return cleaned_mask, confidence_map
    
    def _feature_based_change_detection(self, 
                                       img_before: np.ndarray, 
                                       img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Özellik tabanlı değişiklik tespiti"""
        
        # SIFT özellikleri çıkardım
        sift = cv2.SIFT_create()
        
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        
        kp1, des1 = sift.detectAndCompute(gray_before, None)
        kp2, des2 = sift.detectAndCompute(gray_after, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            # Fallback: basit fark tespiti
            return self._simple_difference_detection(img_before, img_after)
        
        # Feature matching
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)
        
        # İyi eşleşmeleri filtreledim
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        
        # Değişiklik haritası oluşturdum
        change_map = np.zeros(gray_before.shape, dtype=np.uint8)
        confidence_map = np.zeros(gray_before.shape, dtype=np.float32)
        
        # Eşleşmeyen keypoint'ları işaretledim
        matched_kp1_idx = set([m.queryIdx for m in good_matches])
        matched_kp2_idx = set([m.trainIdx for m in good_matches])
        
        # Kayıp keypoint'lar (değişiklik göstergesi)
        for i, kp in enumerate(kp1):
            if i not in matched_kp1_idx:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(change_map, (x, y), 10, 255, -1)
                cv2.circle(confidence_map, (x, y), 10, kp.response, -1)
        
        # Yeni keypoint'lar
        for i, kp in enumerate(kp2):
            if i not in matched_kp2_idx:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                cv2.circle(change_map, (x, y), 10, 255, -1)
                cv2.circle(confidence_map, (x, y), 10, kp.response, -1)
        
        # Smooth yaparak bölgeleri genişlettim
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        change_map = cv2.morphologyEx(change_map, cv2.MORPH_DILATE, kernel)
        
        # Confidence map'i normalize ettim
        if np.max(confidence_map) > 0:
            confidence_map = confidence_map / np.max(confidence_map)
        
        return change_map, confidence_map
    
    def _deep_learning_change_detection(self, 
                                       img_before: np.ndarray, 
                                       img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Derin öğrenme tabanlı değişiklik tespiti"""
        
        # Görüntüleri tensor'a çevirdim
        tensor_before = self._image_to_tensor(img_before)
        tensor_after = self._image_to_tensor(img_after)
        
        with torch.no_grad():
            # Feature extraction
            features_before = self.feature_extractor(tensor_before)
            features_after = self.feature_extractor(tensor_after)
            
            # Feature farkları hesapladım
            feature_diff = torch.abs(features_before - features_after)
            
            # Farkları tek kanala indirdim
            change_features = torch.mean(feature_diff, dim=1, keepdim=True)
            
            # Orijinal boyuta geri getirdim
            original_size = (img_before.shape[0], img_before.shape[1])
            change_map = F.interpolate(change_features, size=original_size, 
                                     mode='bilinear', align_corners=False)
            
            # Numpy'a çevirdim
            change_map_np = change_map.squeeze().cpu().numpy()
        
        # Normalize ettim ve binary mask oluşturdum
        change_map_np = cv2.normalize(change_map_np, None, 0, 1, cv2.NORM_MINMAX)
        
        # Threshold uyguladım
        _, binary_mask = cv2.threshold((change_map_np * 255).astype(np.uint8), 
                                      127, 255, cv2.THRESH_BINARY)
        
        return binary_mask, change_map_np
    
    def _simple_difference_detection(self, 
                                   img_before: np.ndarray, 
                                   img_after: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Basit fark tespiti (fallback)"""
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray_before, gray_after)
        _, binary_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        confidence_map = diff.astype(np.float32) / 255.0
        
        return binary_mask, confidence_map
    
    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Görüntüyü tensor'a çevirme"""
        # BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize etmek
        image_norm = image_rgb.astype(np.float32) / 255.0
        
        # Tensor'a çevirme ve batch dimension ekleme
        tensor = torch.from_numpy(image_norm).permute(2, 0, 1).unsqueeze(0)
        
        return tensor.to(self.device)
    
    def _combine_results(self, 
                        masks: List[np.ndarray], 
                        confidences: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Farklı yöntemlerin sonuçlarını birleştirme"""
        
        if not masks:
            return np.zeros((100, 100), dtype=np.uint8), np.zeros((100, 100), dtype=np.float32)
        
        # Maskeleri normalize etmek
        normalized_masks = []
        normalized_confidences = []
        
        for mask, conf in zip(masks, confidences):
            # Binary mask'i 0-1 aralığına getirdim
            norm_mask = (mask > 0).astype(np.float32)
            
            # Confidence'ı normalize ettim
            if np.max(conf) > 0:
                norm_conf = conf / np.max(conf)
            else:
                norm_conf = conf
            
            normalized_masks.append(norm_mask)
            normalized_confidences.append(norm_conf)
        
        # Her yöntem için ağırlık
        weights = [1.0, 0.8, 0.9, 1.2]  
        weights = weights[:len(normalized_masks)]
        
        combined_confidence = np.zeros_like(normalized_confidences[0])
        combined_votes = np.zeros_like(normalized_masks[0])
        
        for mask, conf, weight in zip(normalized_masks, normalized_confidences, weights):
            combined_votes += mask * weight
            combined_confidence += conf * weight
        
        # Normalize etmek
        total_weight = sum(weights)
        combined_confidence = combined_confidence / total_weight
        combined_votes = combined_votes / total_weight
        
        # Final mask oluşturdum
        final_mask = (combined_votes > self.confidence_threshold).astype(np.uint8) * 255
        
        # Post-processing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        return final_mask, combined_confidence
    
    def _analyze_change_regions(self, change_mask: np.ndarray) -> List[Dict[str, Any]]:
        """Değişiklik bölgelerini analiz etme"""
        
        # Bağlı bileşenleri buldum
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_mask, connectivity=8)
        
        regions = []
        
        for i in range(1, num_labels):  # 0. label arka plan
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area < 50:  # Çok küçük bölgeleri filtreledim
                continue
            
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Bölge özelliklerini hesapladım
            aspect_ratio = w / h if h > 0 else 1.0
            compactness = area / (w * h) if (w * h) > 0 else 0.0
            
            region_info = {
                'id': i,
                'area': area,
                'bbox': (x, y, w, h),
                'centroid': centroids[i],
                'aspect_ratio': aspect_ratio,
                'compactness': compactness,
                'change_type': self._classify_change_type(area, aspect_ratio, compactness)
            }
            
            regions.append(region_info)
        
        # Alanına göre sıraladım
        regions.sort(key=lambda x: x['area'], reverse=True)
        
        return regions
    
    def _classify_change_type(self, area: float, aspect_ratio: float, compactness: float) -> str:
        """Değişiklik tipini sınıflandırma"""
        
        if area > 1000:
            if compactness > 0.7:
                return "major_structure"  # Büyük yapısal değişiklik
            else:
                return "area_development"  # Alan gelişimi
        elif area > 200:
            if aspect_ratio > 3.0:
                return "linear_feature"  # Doğrusal özellik (yol, kanal vb.)
            else:
                return "building_change"  # Bina değişikliği
        else:
            return "minor_change"  # Küçük değişiklik
    
    def _calculate_statistics(self, 
                            img_before: np.ndarray, 
                            img_after: np.ndarray,
                            change_mask: np.ndarray, 
                            confidence_map: np.ndarray) -> Dict[str, float]:
        """Değişiklik istatistiklerini hesaplama"""
        
        total_pixels = img_before.shape[0] * img_before.shape[1]
        change_pixels = np.sum(change_mask > 0)
        
        stats = {
            'change_percentage': (change_pixels / total_pixels) * 100,
            'total_change_area': change_pixels,
            'mean_confidence': np.mean(confidence_map),
            'max_confidence': np.max(confidence_map),
            'confidence_std': np.std(confidence_map),
            'num_change_regions': len(self._analyze_change_regions(change_mask))
        }
        
        return stats

def test_change_detector():
    """test fonksiyonunda değişiklik tespiti"""
    import matplotlib.pyplot as plt
    from src.preprocessor import ImagePreprocessor
    
    print("🧪 ChangeDetector Test Başlatıldı...")
    
    # Test görüntülerini kontrol etme
    test_paths = [
        "data/before/test_image.jpg",
        "data/after/test_image.jpg"
    ]
    
    for path in test_paths:
        if cv2.imread(path) is None:
            print(f"❌ Test görüntüsü bulunamadı: {path}")
            return
    
    # Sistem bileşenlerini başlatma
    preprocessor = ImagePreprocessor(target_size=(400, 400))
    detector = ChangeDetector(methods=['statistical', 'morphological', 'feature_based'])
    
    # Görüntüleri yükleme ve ön işleme
    print("🔄 Görüntüler hazırlanıyor...")
    img_before = preprocessor.load_and_preprocess(test_paths[0])
    img_after = preprocessor.load_and_preprocess(test_paths[1])
    
    # Hizalama
    img_before_aligned, img_after_aligned = preprocessor.align_images(img_before, img_after)
    
    # Değişiklik tespiti
    print("🔍 Değişiklik tespiti yapılıyor...")
    results = detector.detect_changes(img_before_aligned, img_after_aligned)
    
    # Sonuçları yazdırma
    print("\n📊 Tespit Sonuçları:")
    stats = results['statistics']
    print(f"   💫 Değişiklik yüzdesi: {stats['change_percentage']:.2f}%")
    print(f"   🎯 Ortalama güven: {stats['mean_confidence']:.3f}")
    print(f"   📍 Değişiklik bölgesi sayısı: {stats['num_change_regions']}")
    
    # Bölge analizleri
    if results['change_regions']:
        print("   🏗️ Tespit edilen değişiklik tipleri:")
        for region in results['change_regions'][:5]:  # İlk 5 bölge
            print(f"      - {region['change_type']}: {region['area']} piksel")
    
    # Görselleştirme
    plt.figure(figsize=(18, 12))
    
    # Orijinal görüntüler
    plt.subplot(3, 4, 1)
    plt.imshow(cv2.cvtColor(img_before_aligned, cv2.COLOR_BGR2RGB))
    plt.title("Before Image")
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(cv2.cvtColor(img_after_aligned, cv2.COLOR_BGR2RGB))
    plt.title("After Image")
    plt.axis('off')
    
    # Final sonuçları
    plt.subplot(3, 4, 3)
    plt.imshow(results['change_mask'], cmap='hot')
    plt.title("Final Change Mask")
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(results['confidence_map'], cmap='viridis')
    plt.title("Confidence Map")
    plt.axis('off')
    plt.colorbar()
    
    # Yöntem sonuçları
    method_names = list(results['method_results'].keys())
    for i, method in enumerate(method_names[:4]):
        plt.subplot(3, 4, 5 + i)
        plt.imshow(results['method_results'][method]['mask'], cmap='gray')
        plt.title(f"{method.replace('_', ' ').title()}")
        plt.axis('off')
    
    # Overlay görünüm
    plt.subplot(3, 4, 9)
    overlay = img_after_aligned.copy()
    overlay[results['change_mask'] > 0] = [0, 0, 255]  # Kırmızı
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Changes Overlay")
    plt.axis('off')
    
    # İstatistikler
    plt.subplot(3, 4, 10)
    methods = list(stats.keys())[:6]
    values = [stats[method] for method in methods]
    plt.bar(range(len(methods)), values)
    plt.xticks(range(len(methods)), methods, rotation=45)
    plt.title("Detection Statistics")
    
    # Bölge boyut dağılımı
    if results['change_regions']:
        plt.subplot(3, 4, 11)
        areas = [region['area'] for region in results['change_regions']]
        plt.hist(areas, bins=10, alpha=0.7)
        plt.title("Change Region Sizes")
        plt.xlabel("Area (pixels)")
        plt.ylabel("Count")
    
    # Güven dağılımı
    plt.subplot(3, 4, 12)
    conf_flat = results['confidence_map'].flatten()
    plt.hist(conf_flat[conf_flat > 0], bins=50, alpha=0.7)
    plt.title("Confidence Distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("results/change_detection_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Change detection test tamamlandı!")
    print("📊 Sonuç: results/change_detection_test.png")

if __name__ == "__main__":
    test_change_detector()