"""
🛰️ Uydu Görüntü Ön İşleme Modülü
Uydu görüntüleriyle çalışırken ilk adımda yaptığım işlemleri içeren modül. 
Boyutlandırma, gürültü temizleme, kontrast artırımı ve hizalama gibi temel ön işlemleri burada tanımladım.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

# Loglama ayarlarını baştan yapıyorum ki hata veya bilgi çıktıları terminale düzgün düşsün.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    Uydu görüntülerini işlerken kullandığım ön işleme adımlarını ve hizalama algoritmalarını içeren sınıf.
    Özellikle boyutlandırma, gürültü azaltma, kontrast iyileştirme ve hizalama gibi işlemleri kapsıyor.
    """

    def __init__(self, target_size: Tuple[int, int] = (1024, 1024)):
        """
        Başlangıçta hedef boyutu belirtiyorum. Bu boyuta göre görseller normalize ediliyor.
        CUDA destekliyse GPU’da çalışacak şekilde ayarlanıyor.
        """
        self.target_size = target_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"ImagePreprocessor başlatıldı. Device: {self.device}")
    
    def load_and_preprocess(self, image_path: str) -> np.ndarray:
        """
        Verilen görsel yolundan resmi okuyup tüm ön işleme adımlarını sırasıyla uyguluyor.
        Görüntü yoksa hata fırlatıyor.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Görüntü yüklenemedi: {image_path}")
        
        logger.info(f"Görüntü yüklendi: {image_path}, boyut: {image.shape}")
        return self._preprocess_pipeline(image)

    def _preprocess_pipeline(self, image: np.ndarray) -> np.ndarray:
        """
        Bütün ön işleme adımlarını sırayla uyguladığım fonksiyon. 
        Sırasıyla: yeniden boyutlandırma, gürültü azaltma, kontrast artırma, histogram eşitleme ve normalize etme.
        """
        resized = self._resize_image(image)
        denoised = self._denoise_image(resized)
        enhanced = self._enhance_contrast(denoised)
        equalized = self._histogram_equalization(enhanced)
        normalized = self._normalize_image(equalized)
        return normalized
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Görseli hedef boyuta göre yeniden boyutlandırıyorum."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Renkli görüntüdeki gürültüyü azaltmak için Non-Local Means filtresi kullanıyorum."""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Kontrastı artırmak için CLAHE (Adaptive Histogram Equalization) yöntemini sadece L (parlaklık) kanalına uyguluyorum.
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    def _histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        BGR kanallarını tek tek eşitleyerek daha homojen bir kontrast dağılımı sağlamaya çalışıyorum.
        """
        channels = cv2.split(image)
        eq_channels = [cv2.equalizeHist(c) for c in channels]
        return cv2.merge(eq_channels)
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Son olarak tüm piksel değerlerini 0-255 aralığına çekip uint8 formatına dönüştürüyorum."""
        return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    def align_images(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        İki görseli hizalamak için önce ORB (feature tabanlı), başarısız olursa ECC (yoğunluk tabanlı) yöntemini kullanıyorum.
        """
        logger.info("Görüntü hizalama başlatıldı...")
        aligned = self._orb_alignment(image1, image2)
        if aligned is None:
            logger.warning("ORB başarısız oldu, ECC denenecek.")
            aligned = self._ecc_alignment(image1, image2)
        
        if aligned is None:
            logger.warning("Hizalama başarısız oldu. Görseller orijinal halleriyle dönecek.")
            return image1, image2
        
        logger.info("Hizalama başarılı.")
        return image1, aligned
    
    def _orb_alignment(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        ORB algoritmasıyla anahtar nokta eşlemesi yapıp homografi matrisi ile hizalama denemesi yapıyorum.
        """
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            if des1 is None or des2 is None:
                return None
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = sorted(matcher.match(des1, des2), key=lambda x: x.distance)
            good = matches[:int(len(matches) * 0.15)]
            if len(good) < 4:
                return None
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            matrix, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if matrix is None:
                return None
            h, w = img1.shape[:2]
            return cv2.warpPerspective(img2, matrix, (w, h))
        except Exception as e:
            logger.error(f"ORB hizalama hatası: {e}")
            return None
    
    def _ecc_alignment(self, img1: np.ndarray, img2: np.ndarray) -> Optional[np.ndarray]:
        """
        Eğer ORB başarısız olursa, bu yedek yöntem olarak ECC (intensity tabanlı hizalama) kullanıyorum.
        Özellikle yapılar fazla değişmemişse daha başarılı sonuçlar verebiliyor.
        """
        try:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
            warp = np.eye(2, 3, dtype=np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-10)
            try:
                _, warp = cv2.findTransformECC(gray1, gray2, warp, cv2.MOTION_AFFINE, criteria)
                h, w = img1.shape[:2]
                return cv2.warpAffine(img2, warp, (w, h))
            except cv2.error:
                return None
        except Exception as e:
            logger.error(f"ECC hizalama hatası: {e}")
            return None
    
    def calculate_alignment_quality(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        İki görüntünün ne kadar örtüştüğünü ölçmek için normalize edilmiş korelasyon katsayısı hesaplıyorum.
        Değeri 0 ile 1 arasında çıkıyor.
        """
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g1 = (g1 - np.mean(g1)) / np.std(g1)
        g2 = (g2 - np.mean(g2)) / np.std(g2)
        corr = np.corrcoef(g1.flatten(), g2.flatten())[0, 1]
        return max(0, corr)
    
    def get_preprocessing_stats(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, Any]:
        """
        Görsellerin işlem öncesi ve sonrası bazı istatistiklerini (ortalama, std, boyut, kontrast farkı) kıyaslamak için hesaplıyorum.
        """
        return {
            'original_shape': original.shape,
            'processed_shape': processed.shape,
            'original_mean': np.mean(original),
            'processed_mean': np.mean(processed),
            'original_std': np.std(original),
            'processed_std': np.std(processed),
            'contrast_improvement': np.std(processed) / np.std(original) if np.std(original) > 0 else 1.0
        }

def test_preprocessor():
    """
    Preprocessing aşamasını test ettiğim, görselleri işleyip karşılaştırmalı bir çıktı üreten basit bir test fonksiyonu.
    Sonuçlar hem ekrana yazılıyor hem de PNG olarak kaydediliyor.
    """
    import matplotlib.pyplot as plt
    print("🧪 Preprocessor testi başlatıldı.")
    
    test_paths = [
        "data/before/test_image.jpg",
        "data/after/test_image.jpg"
    ]
    
    for path in test_paths:
        if not cv2.imread(path) is not None:
            print(f"❌ Test görseli bulunamadı: {path}")
            return
    
    preprocessor = ImagePreprocessor(target_size=(400, 400))
    
    img1_original = cv2.imread(test_paths[0])
    img2_original = cv2.imread(test_paths[1])
    
    print("🔄 Ön işleme uygulanıyor...")
    img1_processed = preprocessor._preprocess_pipeline(img1_original)
    img2_processed = preprocessor._preprocess_pipeline(img2_original)
    
    print("🔄 Görseller hizalanıyor...")
    img1_aligned, img2_aligned = preprocessor.align_images(img1_processed, img2_processed)
    
    quality = preprocessor.calculate_alignment_quality(img1_aligned, img2_aligned)
    print(f"📊 Hizalama skoru: {quality:.3f}")
    
    stats1 = preprocessor.get_preprocessing_stats(img1_original, img1_processed)
    stats2 = preprocessor.get_preprocessing_stats(img2_original, img2_processed)
    
    print("📈 İstatistikler:")
    print(f"Görsel 1 - Kontrast iyileşmesi: {stats1['contrast_improvement']:.2f}x")
    print(f"Görsel 2 - Kontrast iyileşmesi: {stats2['contrast_improvement']:.2f}x")
    
    # Görsel sonuçları çizip kaydediyorum
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(img1_original, cv2.COLOR_BGR2RGB))
    plt.title("Before - Original")
    plt.axis('off')

    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(img2_original, cv2.COLOR_BGR2RGB))
    plt.title("After - Original")
    plt.axis('off')

    plt.subplot(2, 4, 3)
    plt.imshow(cv2.cvtColor(img1_processed, cv2.COLOR_BGR2RGB))
    plt.title("Before - Processed")
    plt.axis('off')

    plt.subplot(2, 4, 4)
    plt.imshow(cv2.cvtColor(img2_processed, cv2.COLOR_BGR2RGB))
    plt.title("After - Processed")
    plt.axis('off')

    plt.subplot(2, 4, 5)
    plt.imshow(cv2.cvtColor(img1_aligned, cv2.COLOR_BGR2RGB))
    plt.title("Before - Aligned")
    plt.axis('off')

    plt.subplot(2, 4, 6)
    plt.imshow(cv2.cvtColor(img2_aligned, cv2.COLOR_BGR2RGB))
    plt.title("After - Aligned")
    plt.axis('off')

    diff = cv2.absdiff(img1_aligned, img2_aligned)
    plt.subplot(2, 4, 7)
    plt.imshow(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB))
    plt.title("Difference")
    plt.axis('off')

    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    plt.subplot(2, 4, 8)
    plt.hist(diff_gray.flatten(), bins=50, alpha=0.7)
    plt.title("Difference Histogram")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("results/preprocessing_test.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Test tamamlandı. Sonuç: results/preprocessing_test.png")

if __name__ == "__main__":
    test_preprocessor()
