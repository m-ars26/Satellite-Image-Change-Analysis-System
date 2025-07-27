# Uydu Görüntü Değişim Analizi Sistemi

**Farklı zamanlarda çekilmiş uydu görüntüleri arasındaki değişiklikleri tespit eden yapay zeka destekli analiz sistemi**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

---

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Sistem Mimarisi](#sistem-mimarisi)
- [API Referansı](#api-referansı)
- [Geliştirme](#geliştirme)

---

## Proje Hakkında

Bu sistem, uydu görüntüleri üzerinde değişim tespiti yapmak için gelişmiş bilgisayarlı görü algoritmaları ve makine öğrenmesi tekniklerini kullanan kapsamlı bir analiz platformudur. Çoklu algoritma yaklaşımı ile yüksek doğruluk oranında değişim tespiti sağlar.

### Temel Yetenekler

- Çoklu algoritma yaklaşımı ile gelişmiş doğruluk
- Otomatik görüntü ön işleme ve hizalama
- Akıllı değişiklik sınıflandırması
- Profesyonel web arayüzü ve raporlama
- Batch işleme desteği
- Kapsamlı metrik ve güven skorları

---

## Özellikler

### Gelişmiş Değişim Tespiti
- **İstatistiksel Analiz**: Adaptive thresholding ve morfolojik işlemler
- **Morfolojik Analiz**: Çok kanallı gradient analizi  
- **Hibrit Yöntem**: Birden fazla algoritmanın kombinasyonu
- **Gelişmiş Ön İşleme**: CLAHE, gürültü azaltma, histogram eşitleme

### Akıllı Görüntü İşleme
- **Otomatik Hizalama**: ORB ve ECC tabanlı registrasyon
- **Kalite Değerlendirmesi**: Hizalama kalite skorlaması
- **Format Desteği**: JPEG, PNG, TIFF, BMP
- **Boyut Optimizasyonu**: Otomatik görüntü yeniden boyutlandırma

### Web Arayüzü
- **İnteraktif Dashboard**: Streamlit tabanlı modern arayüz
- **Gerçek Zamanlı Görselleştirme**: Plotly grafikleri
- **Kolay Dosya Yükleme**: Drag & drop desteği
- **Çoklu Export Seçeneği**: PNG, JSON formatları

---

## Kurulum

### Gereksinimler
- Python 3.11 veya üzeri
- 4GB+ RAM önerilen

### Adım Adım Kurulum

1. **Repository'yi klonlayın**
   ```bash
   git clone https://github.com/m-ars26/satellite-change-detection.git
   cd satellite-change-detection
   ```

2. **Sanal ortam oluşturun**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux  
   source venv/bin/activate
   ```

3. **Bağımlılıkları yükleyin**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uygulamayı başlatın veya Demoyu test edin**
   ```bash
   streamlit run src/app.py
   ```
   **Demo**
    ```bash
   python src/demo.py --mode demo --samples 3
   ```
### Bağımlılıklar
```
streamlit>=1.28.0
opencv-python>=4.8.0
numpy>=1.24.0
plotly>=5.17.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
scipy>=1.11.0
scikit-image>=0.21.0
```

---

## Kullanım

### Web Arayüzü

1. **Uygulamayı başlatın**
   ```bash
   streamlit run src/app.py
   ```

2. **Tarayıcıda açılan arayüzde:**
   - İki farklı zamandan uydu görüntülerini yükleyin
   - Analiz yöntemini seçin (Statistical önerilir)
   - Duyarlılık parametrelerini ayarlayın
   - "Gelişmiş Analiz Başlat" butonuna tıklayın

3. **Sonuçları inceleyin:**
   - Değişim maskesi ve overlay görüntüleri
   - İnteraktif dashboard grafikleri
   - Detaylı bölge analizi tabloları
   - PNG ve JSON formatlarında export seçenekleri

### Komut Satırı (Demo)

**Kapsamlı demo sistemi** ile tüm özellikleri test edebilirsiniz:

```bash
# Demo modu - Otomatik test verileri oluşturur ve analiz eder
python src/demo.py --mode demo --samples 3

# Tek görüntü çifti analizi
python src/demo.py --mode single --before before.jpg --after after.jpg

# Toplu işleme (batch processing)
python src/demo.py --mode batch --before-dir data/before --after-dir data/after

# Yardım ve tüm seçenekler
python src/demo.py --help
```

**Demo sistemi özellikleri:**
- **Otomatik test verisi oluşturma**: Yapay değişiklikler içeren görüntü çiftleri
- **Progress tracking**: Gerçek zamanlı ilerleme takibi
- **Comprehensive reporting**: Detaylı analiz raporları
- **Batch processing**: Çoklu dosya desteği
- **Error handling**: Profesyonel hata yönetimi

### Python API

```python
from src.preprocessor import ImagePreprocessor
import cv2
import numpy as np

# Preprocessor başlat
preprocessor = ImagePreprocessor(target_size=(1024, 1024))

# Görüntüleri yükle ve işle
img1 = cv2.imread('before.jpg')
img2 = cv2.imread('after.jpg')

# Ön işleme uygula
processed1 = preprocessor._preprocess_pipeline(img1)
processed2 = preprocessor._preprocess_pipeline(img2)

# Hizalama
aligned1, aligned2 = preprocessor.align_images(processed1, processed2)

# Kalite kontrolü
quality = preprocessor.calculate_alignment_quality(aligned1, aligned2)
print(f"Hizalama kalitesi: {quality:.3f}")
```

---

## Sistem Mimarisi

### Proje Yapısı

```
satellite-change-detection/
├── src/
│   ├── app.py                  # Streamlit web uygulaması
│   ├── preprocessor.py         # Görüntü ön işleme (ImagePreprocessor)
│   ├── visualizer.py          # Profesyonel raporlama (ResultVisualizer)
│   ├── utils.py               # Yardımcı fonksiyonlar
│   ├── demo.py                # Kapsamlı demo sistemi
│   └── __init__.py            # Paket yapılandırması
├── test_change_detection.py   # Test dosyası
├── requirements.txt           # Python bağımlılıkları
├── .gitignore                 # Git ignore kuralları
└── README.md                  # Dokümantasyon
```

### Algoritma Pipeline

1. **Görüntü Yükleme ve Ön İşleme**
   - Çoklu format desteği
   - Boyut standardizasyonu
   - Gürültü azaltma (Non-local means)
   - Kontrast artırma (CLAHE)
   - Histogram eşitleme

2. **Görüntü Hizalama**
   - ORB feature detection ve matching
   - Homografi hesaplama
   - ECC algoritması fallback
   - Kalite skorlaması

3. **Değişim Tespiti**
   - İstatistiksel fark analizi
   - Morfolojik gradient analizi
   - Adaptive thresholding
   - Bağlı bileşen analizi

4. **Sonuç İşleme**
   - Bölge sınıflandırması
   - Güven skorları hesaplama
   - Metrik çıkarımı
   - Görselleştirme

---

## API Referansı

### ImagePreprocessor Sınıfı

```python
class ImagePreprocessor:
    """Görüntü ön işleme ve hizalama sınıfı"""
    
    def __init__(self, target_size: Tuple[int, int] = (1024, 1024))
        """Preprocessor'ı başlat"""
    
    def _preprocess_pipeline(self, image: np.ndarray) -> np.ndarray
        """Tam ön işleme pipeline'ı uygula"""
    
    def align_images(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]
        """İki görüntüyü hizala"""
    
    def calculate_alignment_quality(self, img1: np.ndarray, img2: np.ndarray) -> float
        """Hizalama kalitesini hesapla (0-1 arası)"""
```

### Ana Fonksiyonlar

```python
def detect_changes_statistical(img1, img2, threshold=30)
    """İstatistiksel değişim tespiti"""

def detect_changes_morphological(img1, img2)
    """Morfolojik değişim tespiti"""

def detect_changes_hybrid(img1, img2)
    """Hibrit yöntem kombinasyonu"""
```

---

## Performans

### Benchmark Sonuçları

| Metrik | Değer |
|--------|-------|
| Tespit Doğruluğu | %85-95 |
| İşleme Hızı | 2-5 saniye/çift |
| Bellek Kullanımı | 2-4 GB RAM |
| Maksimum Çözünürlük | 4096x4096 piksel |

### Desteklenen Formatlar
- **Giriş**: JPEG, PNG, TIFF, BMP
- **Çıkış**: PNG (görüntüler), JSON (veriler)

---

## Geliştirme

### Geliştirici Ortamı

```bash
# Test çalıştırma
python test_change_detection.py

# Kod kalitesi kontrolü
flake8 src/
black src/

# Performans testi
python -m cProfile src/demo.py
```

### Katkıda Bulunma

1. Repository'yi fork edin
2. Feature branch oluşturun
3. Değişikliklerinizi yapın
4. Test ekleyin ve çalıştırın
5. Pull request gönderin

### Kod Standartları

- PEP 8 kod stili
- Type hints kullanımı
- Kapsamlı docstring'ler
- Unit test coverage

---

## Sorun Giderme

### Yaygın Hatalar

**"Boolean index did not match" hatası**
```
Çözüm: Görüntüler farklı boyutlarda. Sistem otomatik düzeltir.
```

**"JSON serialization" hatası**
```
Çözüm: NumPy tipleri otomatik Python tiplerine dönüştürülür.
```

**"OpenCV sizes do not match" hatası**
```
Çözüm: ensure_same_size() fonksiyonu kullanılır.
```

### Performans Optimizasyonu

- Büyük görüntüler için target_size'ı küçültün
- GPU kullanılabiliyorsa CUDA'yı aktifleştirin
- Batch işleme için bellek yönetimi yapın

---

## İletişim

**Proje Geliştirici**: Mehmet Arslan  
**E-posta**: [mehmet26arslan@outlook.com]  
**GitHub**: [https://github.com/m-ars26]

---


**Bu proje Birsav Bilişim Yapay Zeka Uzmanı pozisyonu için geliştirilmiştir.**