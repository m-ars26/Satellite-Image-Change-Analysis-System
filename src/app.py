import streamlit as st
import sys
import os
import numpy as np
import cv2
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import io
import base64
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Mevcut src klasörünü Python path'ine ekler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Sayfa konfigürasyonu yapıyorum
st.set_page_config(
    page_title="Uydu Görüntü Değişim Analizi - Advanced",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


PREPROCESSOR_AVAILABLE = False
UTILS_AVAILABLE = False
VISUALIZER_AVAILABLE = False
TEST_MODULE_AVAILABLE = False

# Preprocessor modülünü import ediyorum
try:
    from preprocessor import ImagePreprocessor
    PREPROCESSOR_AVAILABLE = True
    if 'preprocessor_loaded' not in st.session_state:
        st.session_state.preprocessor_loaded = True
        st.success("✅ Gelişmiş ImagePreprocessor yüklendi!")
except ImportError as e:
    ImagePreprocessor = None
    if 'preprocessor_error' not in st.session_state:
        st.session_state.preprocessor_error = True
        st.warning(f"⚠️ preprocessor.py yüklenemedi: {e}")

# Utils modülünü import ediyorum
try:
    import utils
    UTILS_AVAILABLE = True
    if hasattr(utils, 'export_results'):
        export_results = utils.export_results
    else:
        export_results = None
except ImportError:
    export_results = None
    if 'utils_info_shown' not in st.session_state:
        st.session_state.utils_info_shown = True
        st.info("ℹ️ utils.py bulunamadı - temel export kullanılacak")

# Visualizer modülünü import ediyorum
try:
    import visualizer
    VISUALIZER_AVAILABLE = True
    if hasattr(visualizer, 'create_advanced_visualization'):
        create_advanced_visualization = visualizer.create_advanced_visualization
    else:
        create_advanced_visualization = None
except ImportError:
    create_advanced_visualization = None
    if 'visualizer_info_shown' not in st.session_state:
        st.session_state.visualizer_info_shown = True
        st.info("ℹ️ visualizer.py bulunamadı - temel görselleştirme kullanılacak")

# Custom CSS - Daha sade ve okunabilir bir tema ve Overlay sorununun çözümü
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 600;
        background: linear-gradient(90deg, #3498db, #2980b9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: none;
    }
    .metric-container {
        background: #ffffff;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e1e8ed;
    }
    .change-detected {
        background: #ffffff;
        color: #e74c3c;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(231,76,60,0.1);
    }
    .no-change {
        background: #ffffff;
        color: #27ae60;
        border: 2px solid #27ae60;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(39,174,96,0.1);
    }
    .status-card {
        background: #f8f9fa;
        color: #2c3e50;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        font-weight: 500;
    }
    .method-badge {
        background: #3498db;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .feature-highlight {
        background: #ffffff;
        color: #2c3e50;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #3498db;
        box-shadow: 0 2px 8px rgba(52,152,219,0.1);
    }
    .feature-highlight h4 {
        color: #3498db;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .feature-highlight p {
        color: #2c3e50;
        line-height: 1.6;
        margin: 0;
    }
    /* Streamlit özel sınıflarını geçersiz kıl */
    .stMarkdown {
        color: #2c3e50;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    /* Metric widget styling */
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* OVERLAY SORUNUNUN ÇÖZÜMÜ */
    /* Plotly chart container  */
    .js-plotly-plot {
        z-index: 1 !important;
        position: relative !important;
    }
    
    /* Streamlit plotly container */
    .stPlotlyChart {
        z-index: 1 !important;
        position: relative !important;
        overflow: visible !important;
    }
    
    /* Plotly modebar */
    .modebar {
        z-index: 10 !important;
        position: relative !important;
    }
    
    /* Container spacing  */
    .plotly-container {
        margin: 2rem 0 !important;
        padding: 1rem !important;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Chart height */
    .stPlotlyChart > div {
        height: 600px !important;
    }
    
    /* Sidebar metric */
    .css-1lcbmhc {
        z-index: 5 !important;
    }
</style>
""", unsafe_allow_html=True)

def ensure_same_size(img1, img2, target_size=None):
    """İki görüntüyü aynı boyuta getirdim"""
    if target_size is None:
        # İki görüntü arasında en küçük ortak boyutu bulma 
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        target_h = min(h1, h2)
        target_w = min(w1, w2)
        target_size = (target_w, target_h)
    else:
        target_w, target_h = target_size
    
    # Görüntüleri yeniden boyutlandırma
    img1_resized = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    img2_resized = cv2.resize(img2, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    return img1_resized, img2_resized

def convert_numpy_types(obj):
    """NumPy tiplerini Python tiplerine çevirme (JSON için)"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class EnhancedChangeDetectionApp:
    def __init__(self):
        self.before_image = None
        self.after_image = None
        self.results = {}
        self.available_methods = self._get_available_methods()
        
        # ImagePreprocessor'ı başlatıyorum
        if PREPROCESSOR_AVAILABLE and ImagePreprocessor:
            try:
                self.preprocessor = ImagePreprocessor(target_size=(1024, 1024))
                if 'advanced_preprocessing_status' not in st.session_state:
                    st.session_state.advanced_preprocessing_status = True
                    st.info("🚀 Gelişmiş ön işleme sistemi aktif!")
            except Exception as e:
                self.preprocessor = None
                st.error(f"ImagePreprocessor başlatılamadı: {e}")
        else:
            self.preprocessor = None
            if 'basic_preprocessing_status' not in st.session_state:
                st.session_state.basic_preprocessing_status = True
                st.info("📝 Temel ön işleme sistemi kullanılıyor")
        
    def _get_available_methods(self):
        """Kullanılabilir yöntemleri tespit et"""
        methods = {
            'Statistical': {'available': True, 'description': 'Temel istatistiksel fark analizi'},
            'Morphological': {'available': True, 'description': 'Morfolojik gradyan analizi'},
            'Advanced Preprocessing': {'available': PREPROCESSOR_AVAILABLE, 'description': 'Gelişmiş ön işleme + analiz'},
            'Hybrid Enhanced': {'available': True, 'description': 'Birleşik gelişmiş yöntem'}
        }
        return methods
    
    def apply_advanced_preprocessing(self, img1, img2):
        """Ön işleme uygula"""
        if self.preprocessor is None:
            return img1, img2, {'preprocessing_applied': False, 'error': 'Preprocessor not available'}
        
        try:
            # Önce görüntüleri aynı boyuta getiriyorum
            img1, img2 = ensure_same_size(img1, img2, target_size=(1024, 1024))
            
            # Görüntüleri BGR formatına çeviriyorum (OpenCV için)
            if len(img1.shape) == 3 and img1.shape[2] == 3:
                bgr1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
                bgr2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            else:
                bgr1, bgr2 = img1, img2
            
            # Ön işleme pipeline uyguluyorum
            processed1 = self.preprocessor._preprocess_pipeline(bgr1)
            processed2 = self.preprocessor._preprocess_pipeline(bgr2)
            
            # Görüntüleri tekrar aynı boyuta getiriyorum (ön işleme sonrası boyut değişebiliyor)
            processed1, processed2 = ensure_same_size(processed1, processed2)
            
            # Görüntüleri hizalıyorum
            aligned1, aligned2 = self.preprocessor.align_images(processed1, processed2)
            
            # Hizalama sonrası boyutları kontrol ediyorum
            if aligned1 is not None and aligned2 is not None:
                aligned1, aligned2 = ensure_same_size(aligned1, aligned2)
            else:
                aligned1, aligned2 = processed1, processed2
            
            # Hizalama kalitesini hesaplıyorum
            alignment_quality = self.preprocessor.calculate_alignment_quality(aligned1, aligned2)
            
            # RGB formatına geri çeviriyorum
            if len(aligned1.shape) == 3:
                rgb1 = cv2.cvtColor(aligned1, cv2.COLOR_BGR2RGB)
                rgb2 = cv2.cvtColor(aligned2, cv2.COLOR_BGR2RGB)
            else:
                rgb1, rgb2 = aligned1, aligned2
            
            # İstatistikleri hesaplıyorum
            stats1 = self.preprocessor.get_preprocessing_stats(bgr1, aligned1)
            stats2 = self.preprocessor.get_preprocessing_stats(bgr2, aligned2)
            
            return rgb1, rgb2, {
                'alignment_quality': float(alignment_quality),
                'stats1': convert_numpy_types(stats1),
                'stats2': convert_numpy_types(stats2),
                'preprocessing_applied': True,
                'success': True
            }
            
        except Exception as e:
            st.error(f"Gelişmiş ön işleme hatası: {e}")
            return img1, img2, {'preprocessing_applied': False, 'error': str(e)}
    
    def detect_changes_statistical(self, img1, img2, threshold=30, use_advanced=True):
        """İstatistiksel analiz"""
        preprocessing_info = {'preprocessing_applied': False}
        
        try:
            # Öncelikle görüntüleri aynı boyuta getiriyorum
            img1, img2 = ensure_same_size(img1, img2)
            
            # Preprocessor'i uyguluyorum
            if use_advanced and self.preprocessor is not None:
                try:
                    result = self.apply_advanced_preprocessing(img1, img2)
                    if len(result) == 3:
                        img1, img2, preprocessing_info = result
                    else:
                        img1, img2 = result
                except Exception as e:
                    st.warning(f"Gelişmiş ön işleme hatası: {e}, temel yöntem kullanılıyor")
            
            # Gri tonlama
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            # Boyut kontrolü ve eşitleme
            gray1, gray2 = ensure_same_size(gray1, gray2)
            h, w = gray1.shape[:2]
            
            # Gaussian blur ile gürültü azaltma
            gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)
            gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)
            
            # Fark hesaplama
            diff = cv2.absdiff(gray1, gray2)
            
            # Adaptive threshold
            change_mask = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)
            
            # Morfolojik temizleme
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
            
            # İstatistikleri hesaplama
            total_pixels = h * w
            changed_pixels = np.sum(change_mask > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            # Confidence haritası
            confidence_map = cv2.normalize(diff.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
            
            result = {
                'change_mask': change_mask,
                'difference_map': diff,
                'confidence_map': confidence_map,
                'change_percentage': float(change_percentage),
                'changed_pixels': int(changed_pixels),
                'total_pixels': int(total_pixels),
                'method': 'Statistical Enhanced',
                'preprocessing_info': convert_numpy_types(preprocessing_info)
            }
            
            return result
            
        except Exception as e:
            st.error(f"Statistical analiz hatası: {e}")
            raise
    
    def detect_changes_morphological(self, img1, img2):
        """Morfolojik analiz"""
        try:
            # Öncelikle görüntüleri aynı boyuta getiriyorum
            img1, img2 = ensure_same_size(img1, img2)
            
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
            
            # Boyut kontrolü
            gray1, gray2 = ensure_same_size(gray1, gray2)
            h, w = gray1.shape[:2]
            
            # Çoklu ölçek morfolojik analiz
            kernels = [
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            ]
            
            combined_diff = np.zeros_like(gray1, dtype=np.float32)
            
            for kernel in kernels:
                # Morfolojik gradyan
                grad1 = cv2.morphologyEx(gray1, cv2.MORPH_GRADIENT, kernel)
                grad2 = cv2.morphologyEx(gray2, cv2.MORPH_GRADIENT, kernel)
                
                # Fark hesapla
                diff = cv2.absdiff(grad1, grad2)
                combined_diff += diff.astype(np.float32)
            
            # Normalize et
            combined_diff = cv2.normalize(combined_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Threshold
            _, change_mask = cv2.threshold(combined_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Temizleme
            change_mask = cv2.medianBlur(change_mask, 5)
            
            total_pixels = h * w
            changed_pixels = np.sum(change_mask > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            return {
                'change_mask': change_mask,
                'difference_map': combined_diff,
                'confidence_map': combined_diff / 255.0,
                'change_percentage': float(change_percentage),
                'changed_pixels': int(changed_pixels),
                'total_pixels': int(total_pixels),
                'method': 'Morphological Enhanced'
            }
            
        except Exception as e:
            st.error(f"Morphological analiz hatası: {e}")
            raise
    
    def detect_changes_advanced_preprocessing(self, img1, img2):
        """Preprocessing yöntemi"""
        if not PREPROCESSOR_AVAILABLE:
            st.warning("Advanced Preprocessing için preprocessor.py gerekli!")
            return self.detect_changes_statistical(img1, img2, use_advanced=False)
        
        try:
            # Preprocessing uyguluyorum
            result = self.apply_advanced_preprocessing(img1, img2)
            if len(result) == 3:
                processed_img1, processed_img2, prep_info = result
            else:
                processed_img1, processed_img2 = result
                prep_info = {'preprocessing_applied': False}
            
            # Statistical analiz uyguluyorum
            analysis_result = self.detect_changes_statistical(processed_img1, processed_img2, use_advanced=False)
            analysis_result['method'] = 'Advanced Preprocessing + Statistical'
            analysis_result['preprocessing_info'] = convert_numpy_types(prep_info)
            
            return analysis_result
            
        except Exception as e:
            st.error(f"Advanced Preprocessing hatası: {e}")
            raise
    
    def detect_changes_hybrid(self, img1, img2):
        """Hibrit yöntem - Statistical + Morphological"""
        try:
            # Öncelikle görüntüleri aynı boyuta getiriyorum
            img1, img2 = ensure_same_size(img1, img2)
            
            # Statistical analiz
            stat_results = self.detect_changes_statistical(img1, img2, use_advanced=True)
            
            # Morphological analiz
            morph_results = self.detect_changes_morphological(img1, img2)
            
            # Boyut kontrolü - change mask'lerin aynı boyutta olduğundan emin oluyorum
            stat_mask = stat_results['change_mask']
            morph_mask = morph_results['change_mask']
            
            if stat_mask.shape != morph_mask.shape:
                # Daha küçük boyuta resize ediyorum
                target_h = min(stat_mask.shape[0], morph_mask.shape[0])
                target_w = min(stat_mask.shape[1], morph_mask.shape[1])
                
                stat_mask = cv2.resize(stat_mask, (target_w, target_h))
                morph_mask = cv2.resize(morph_mask, (target_w, target_h))
            
            # Sonuçları birleştiriyorum
            combined_mask = cv2.bitwise_or(stat_mask, morph_mask)
            
            # Confidence map'leri de aynı boyuta getiriyorum
            stat_conf = stat_results['confidence_map']
            morph_conf = morph_results['confidence_map']
            
            if stat_conf.shape != morph_conf.shape:
                target_h = min(stat_conf.shape[0], morph_conf.shape[0])
                target_w = min(stat_conf.shape[1], morph_conf.shape[1])
                
                stat_conf = cv2.resize(stat_conf, (target_w, target_h))
                morph_conf = cv2.resize(morph_conf, (target_w, target_h))
            
            # Confidence'ları ağırlıklı ortalama
            combined_confidence = (stat_conf * 0.6 + morph_conf * 0.4)
            
            total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
            changed_pixels = np.sum(combined_mask > 0)
            change_percentage = (changed_pixels / total_pixels) * 100
            
            return {
                'change_mask': combined_mask,
                'difference_map': stat_results['difference_map'],
                'confidence_map': combined_confidence,
                'change_percentage': float(change_percentage),
                'changed_pixels': int(changed_pixels),
                'total_pixels': int(total_pixels),
                'method': 'Hybrid Enhanced (Statistical + Morphological)',
                'individual_results': {
                    'statistical': convert_numpy_types(stat_results),
                    'morphological': convert_numpy_types(morph_results)
                },
                'preprocessing_info': convert_numpy_types(stat_results.get('preprocessing_info', {}))
            }
            
        except Exception as e:
            st.error(f"Hybrid analiz hatası: {e}")
            raise
    
    def analyze_change_regions(self, change_mask):
        """Değişim bölgelerini analiz etme"""
        try:
            # Bağlı bileşenleri buluyorum
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(change_mask, 8, cv2.CV_32S)
            
            regions = []
            for i in range(1, num_labels):  # 0 arka plan
                area = stats[i, cv2.CC_STAT_AREA]
                if area > 50:  # Minimum alan filtresi
                    regions.append({
                        'id': int(i),
                        'area': int(area),
                        'centroid': [float(centroids[i][0]), float(centroids[i][1])],
                        'bbox': [int(stats[i][j]) for j in range(5)]
                    })
            
            # Alan'a göre sıralama
            regions.sort(key=lambda x: x['area'], reverse=True)
            
            return regions[:10]  # En büyük 10 bölge
            
        except Exception as e:
            st.error(f"Bölge analizi hatası: {e}")
            return []
    
    def create_overlay_visualization(self, before_img, after_img, change_mask):
        """Overlay görselleştirmesi"""
        try:
            # Görüntüleri aynı boyuta getiriyorum
            before_img, after_img = ensure_same_size(before_img, after_img)
            
            # Change mask'i görüntü boyutuna getiriyorum
            if change_mask.shape[:2] != after_img.shape[:2]:
                change_mask = cv2.resize(change_mask, (after_img.shape[1], after_img.shape[0]))
            
            # Değişim maskesini renklendiriyorum
            colored_mask = np.zeros_like(after_img)
            colored_mask[change_mask > 0] = [255, 0, 0]  # Kırmızı
            
            # Overlay oluşturuyorum
            overlay = cv2.addWeighted(after_img, 0.7, colored_mask, 0.3, 0)
            
            return overlay
            
        except Exception as e:
            st.error(f"Overlay oluşturma hatası: {e}")
            return after_img

def main():
    st.markdown('<h1 class="main-header">🛰️ Gelişmiş Uydu Görüntü Değişim Analizi</h1>', unsafe_allow_html=True)
    
    app = EnhancedChangeDetectionApp()
    
    # Sistem durumu kontrolü
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "🟢 Aktif" if PREPROCESSOR_AVAILABLE else "🟡 Temel"
        st.markdown(f"""
        <div class="status-card">
            <strong>Preprocessor:</strong> {status}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        status = "🟢 Aktif" if UTILS_AVAILABLE else "🟡 Temel"
        st.markdown(f"""
        <div class="status-card">
            <strong>Utils:</strong> {status}
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        status = "🟢 Aktif" if VISUALIZER_AVAILABLE else "🟡 Temel"
        st.markdown(f"""
        <div class="status-card">
            <strong>Visualizer:</strong> {status}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="status-card">
            <strong>Sistem:</strong> 🟢 Hazır
        </div>
        """, unsafe_allow_html=True)
    
    # Özellik vurgusu 
    if PREPROCESSOR_AVAILABLE:
        st.markdown("""
        <div class="feature-highlight">
            <h4>🚀 Özellikler Aktif!</h4>
            <p>
            ✅ Otomatik görüntü hizalama<br>
            ✅ Gürültü azaltma ve kontrast artırma<br>
            ✅ Kalite metrikleri<br>
            ✅ Boyut uyumsuzlukları otomatik düzeltilir
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="feature-highlight">
            <h4>📝 Temel Analiz Modu</h4>
            <p>
            ✅ İstatistiksel değişim analizi<br>
            ✅ Morfolojik analiz<br>
            ✅ Hibrit yöntem<br>
            💡 preprocessor.py yükleyerek gelişmiş özellikleri aktifleştirin
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Analiz Parametreleri")
        
        # Yöntem seçimi
        available_methods = [name for name, info in app.available_methods.items() if info['available']]
        method = st.selectbox(
            "Analiz Yöntemi",
            available_methods,
            help="Kullanılacak değişim tespit algoritması"
        )
        
        # Seçilen yöntem bilgisi
        if method in app.available_methods:
            st.info(f"ℹ️ {app.available_methods[method]['description']}")
        
        st.markdown("---")
        
        # Temel parametreler
        sensitivity = st.slider(
            "Duyarlılık",
            min_value=1,
            max_value=10,
            value=5,
            help="1: Az hassas, 10: Çok hassas"
        )
        
        min_region_size = st.slider(
            "Min. Bölge Boyutu",
            min_value=10,
            max_value=1000,
            value=100,
            help="Göz ardı edilecek minimum alan (piksel)"
        )
        
        morphological_ops = st.checkbox(
            "Morfolojik Temizleme",
            value=True,
            help="Opening/Closing operasyonları"
        )
        
        # Sonuçları gösterme
        if 'results' in st.session_state:
            st.markdown("---")
            st.subheader("📊 Anlık Sonuçlar")
            results = st.session_state.results
            change_pct = results.get('change_percentage', 0)
            
            st.metric("Değişim Oranı", f"{change_pct:.2f}%")
            st.metric("Değişen Piksel", f"{results.get('changed_pixels', 0):,}")
            st.metric("Kullanılan Yöntem", results.get('method', 'N/A'))
    
    # Ana içerik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📸 Önceki Görüntü (Before)")
        before_file = st.file_uploader(
            "Önceki görüntüyü yükleyin",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="before",
            help="Karşılaştırmanın referans görüntüsü"
        )
        
        if before_file:
            app.before_image = np.array(Image.open(before_file))
            st.image(app.before_image, caption="Önceki Görüntü", use_container_width=True)
            st.info(f"📐 Boyut: {app.before_image.shape[1]}x{app.before_image.shape[0]} piksel")
    
    with col2:
        st.subheader("📸 Sonraki Görüntü (After)")
        after_file = st.file_uploader(
            "Sonraki görüntüyü yükleyin",
            type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
            key="after",
            help="Değişimlerin tespit edileceği görüntü"
        )
        
        if after_file:
            app.after_image = np.array(Image.open(after_file))
            st.image(app.after_image, caption="Sonraki Görüntü", use_container_width=True)
            st.info(f"📐 Boyut: {app.after_image.shape[1]}x{app.after_image.shape[0]} piksel")
    
    # Kontrol butonları
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        analyze_btn = st.button(
            "🔍 Gelişmiş Analiz Başlat",
            type="primary",
            use_container_width=True,
            disabled=(app.before_image is None or app.after_image is None)
        )
    
    with col2:
        clear_btn = st.button("🗑️ Temizle", use_container_width=True)
    
    with col3:
        help_btn = st.button("❓ Yardım", use_container_width=True)
    
    # Analiz işlemi
    if analyze_btn:
        with st.spinner('🔄 Gelişmiş analiz yapılıyor...'):
            try:
                # Threshold hesaplama (sensitivity'den)
                threshold = max(10, 50 - (sensitivity * 4))
                
                # Analiz 
                if method == "Statistical":
                    results = app.detect_changes_statistical(app.before_image, app.after_image)
                elif method == "Morphological":
                    results = app.detect_changes_morphological(app.before_image, app.after_image)
                elif method == "Advanced Preprocessing":
                    results = app.detect_changes_advanced_preprocessing(app.before_image, app.after_image)
                elif method == "Hybrid Enhanced":
                    results = app.detect_changes_hybrid(app.before_image, app.after_image)
                else:
                    results = app.detect_changes_statistical(app.before_image, app.after_image)
                
                # Bölge analizi
                if results['change_mask'] is not None:
                    regions = app.analyze_change_regions(results['change_mask'])
                    results['regions'] = regions
                
                # Overlay oluşturma
                if results['change_mask'] is not None:
                    overlay = app.create_overlay_visualization(
                        app.before_image, app.after_image, results['change_mask']
                    )
                    results['overlay'] = overlay
                
                st.session_state.results = results
                st.success("✅ Analiz başarıyla tamamlandı!")
                
            except Exception as e:
                st.error(f"❌ Analiz hatası: {str(e)}")
                st.info("💡 Lütfen farklı bir yöntem deneyin veya görüntü formatını kontrol edin.")
    
    # Temizleme
    if clear_btn:
        if 'results' in st.session_state:
            del st.session_state.results
        st.rerun()
    
    # Yardım göster
    if help_btn:
        st.info("""
        ### 🚀 Hızlı Kullanım Rehberi
        1. **Görüntü Yükleme**: İki farklı zamandan uydu görüntülerini yükleyin
        2. **Yöntem Seçimi**: 
           - **Statistical**: En hızlı ve güvenilir (önerilen)
           - **Morphological**: Yapısal değişimler için
           - **Advanced Preprocessing**: En kaliteli sonuç (preprocessor gerekli)
           - **Hybrid Enhanced**: Birleşik analiz
        3. **Analiz**: Mavi butona tıklayın
        4. **Sonuçları İnceleyin**: Maskeler, grafikler ve metrikleri görün
        5. **İndirin**: PNG veya JSON formatında sonuçları kaydedin
        
        ### ⚠️ Sorun Giderme
        - Görüntüler farklı boyutlarda olabilir (otomatik düzeltilir)
        - PNG/JPG formatları desteklenir
        - Büyük görüntüler otomatik küçültülür
        """)
    
    # Sonuçları gösterme
    if 'results' in st.session_state:
        results = st.session_state.results
        change_pct = results.get('change_percentage', 0)
        
        # Sonuç özeti
        if change_pct > 10:
            st.markdown(f"""
            <div class="change-detected">
                <h2>🚨 Kritik Değişim Tespit Edildi!</h2>
                <p><strong>{change_pct:.2f}%</strong> oranında değişim bulundu.</p>
                <p>Toplam <strong>{results.get('changed_pixels', 0):,}</strong> piksel etkilenmiş.</p>
                <p>Kullanılan yöntem: <strong>{results.get('method', 'N/A')}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        elif change_pct > 3:
            st.markdown(f"""
            <div class="no-change">
                <h2>⚠️ Orta Düzey Değişim</h2>
                <p><strong>{change_pct:.2f}%</strong> oranında değişim tespit edildi.</p>
                <p>Bu değişim izlenmesi gereken seviyede.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-change">
                <h2>✅ Minimal Değişim</h2>
                <p>Sadece <strong>{change_pct:.2f}%</strong> oranında değişim var.</p>
                <p>Normal varyasyon aralığında.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ön işleme bilgileri 
        if 'preprocessing_info' in results and results['preprocessing_info'].get('preprocessing_applied'):
            prep_info = results['preprocessing_info']
            st.markdown("### 🔬 Ön İşleme Analizi")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                alignment_quality = prep_info.get('alignment_quality', 0)
                delta_alignment = alignment_quality - 0.5 if alignment_quality > 0.5 else None
                st.metric(
                    "🎯 Hizalama Kalitesi",
                    f"{alignment_quality:.3f}",
                    delta=f"{delta_alignment:.3f}" if delta_alignment else None
                )
            
            with col2:
                if 'stats1' in prep_info:
                    contrast_improvement = prep_info['stats1'].get('contrast_improvement', 1.0)
                    delta_contrast = contrast_improvement - 1.0 if contrast_improvement > 1.0 else None
                    st.metric(
                        "📈 Kontrast İyileştirme",
                        f"{contrast_improvement:.2f}x",
                        delta=f"{delta_contrast:.2f}x" if delta_contrast else None
                    )
            
            with col3:
                processing_success = "✅ Başarılı" if prep_info.get('success', False) else "❌ Başarısız"
                st.metric("🔧 Ön İşleme Durumu", processing_success)
        
        # Görselleştirmeler
        st.subheader("📊 Analiz Sonuçları")
        
        # Ana görselleştirmeler
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔍 Değişim Maskesi")
            st.image(results['change_mask'], caption="Beyaz = Değişim Alanları", use_container_width=True)
        
        with col2:
            st.markdown("#### 🌡️ Fark Haritası")
            st.image(results['difference_map'], caption="Yoğunluk = Fark Büyüklüğü", use_container_width=True)
        
        with col3:
            st.markdown("#### 🎯 Overlay Görüntü")
            st.image(results['overlay'], caption="Kırmızı = Tespit Edilen Değişimler", use_container_width=True)
        
        # İnteraktif dashboard
        st.subheader("📈 İnteraktif Analiz Dashboard")
        
        # Dashboard container ile wrap ettim
        with st.container():
            st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
            
            try:
                # Dashboard oluşturma
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=(
                        'Confidence Dağılımı', 
                        'Değişim Bölgeleri',
                        'Piksel Değer Karşılaştırma',
                        'Ana Metrikler'
                    ),
                    specs=[
                        [{"type": "histogram"}, {"type": "scatter"}],
                        [{"type": "histogram"}, {"type": "bar"}]
                    ],
                    vertical_spacing=0.12,  # Alt-üst arası boşluk
                    horizontal_spacing=0.1   # Yan-yana arası boşluk
                )
                
                # Confidence histogram
                confidence_flat = results['confidence_map'].flatten()
                fig.add_histogram(x=confidence_flat, name="Güven Skoru", row=1, col=1, nbinsx=50)
                
                # Bölge scatter plot
                if 'regions' in results and results['regions']:
                    regions = results['regions']
                    fig.add_scatter(
                        x=[r['centroid'][0] for r in regions],
                        y=[r['centroid'][1] for r in regions],
                        mode='markers',
                        marker=dict(
                            size=[min(30, np.sqrt(r['area']/10)) for r in regions],  # Boyut küçültüldü
                            color=[r['area'] for r in regions],
                            colorscale='Reds',
                            showscale=True,
                            colorbar=dict(title="Alan (piksel)", x=0.48)  # Colorbar pozisyonu
                        ),
                        name="Değişim Bölgeleri",
                        text=[f"Bölge {i+1}: {r['area']} px" for i, r in enumerate(regions)],
                        row=1, col=2
                    )
                
                # Piksel değer karşılaştırma - sampling ile hızlandırma
                before_gray = cv2.cvtColor(app.before_image, cv2.COLOR_RGB2GRAY)
                after_gray = cv2.cvtColor(app.after_image, cv2.COLOR_RGB2GRAY)
                
                # Daha az sample almak
                sample_size = min(10000, before_gray.size)
                indices = np.random.choice(before_gray.size, sample_size, replace=False)
                
                fig.add_histogram(x=before_gray.flatten()[indices], name="Önceki", 
                                opacity=0.7, row=2, col=1, nbinsx=30)
                fig.add_histogram(x=after_gray.flatten()[indices], name="Sonraki", 
                                opacity=0.7, row=2, col=1, nbinsx=30)
                
                # Metrik barları
                metrics = ['Değişim %', 'Bölge Sayısı', 'Ortalama Güven']
                values = [
                    change_pct,
                    len(results.get('regions', [])),
                    np.mean(confidence_flat) * 100
                ]
                
                colors = ['#e74c3c', '#3498db', '#f39c12']
                fig.add_bar(x=metrics, y=values, name="Ana Metrikler", 
                           row=2, col=2, marker_color=colors)
                
                # Layout güncellemeleri
                fig.update_layout(
                    height=500,  # Yükseklik küçültüldü
                    showlegend=False,  # Legend kapatıldı (yer kazanmak için)
                    title_text="Kapsamlı Analiz Dashboard",
                    title_x=0.5,
                    title_font_size=16,
                    margin=dict(l=50, r=50, t=80, b=50)  # Margin'ler ayarlandı
                )
                
                # Subplot başlıklarını küçültme
                for annotation in fig['layout']['annotations']:
                    annotation['font'] = dict(size=12)
                
                st.plotly_chart(fig, use_container_width=True, key="dashboard_chart")
                
            except Exception as e:
                st.warning(f"Dashboard oluşturma hatası (görselleştirme atlandı): {e}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Değişim Oranı", f"{change_pct:.1f}%")
                with col2:
                    st.metric("Bölge Sayısı", len(results.get('regions', [])))
                with col3:
                    st.metric("Ortalama Güven", f"{np.mean(confidence_flat):.3f}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bölge detay analizi
        if 'regions' in results and results['regions']:
            st.subheader("📍 Tespit Edilen Değişim Bölgeleri")
            
            # Tablo verisi oluşturma
            table_data = []
            for i, region in enumerate(results['regions'][:10]):
                table_data.append({
                    'Bölge ID': f"R{i+1}",
                    'Alan (piksel)': f"{region['area']:,}",
                    'Alan (%)': f"{(region['area']/results['total_pixels']*100):.3f}%",
                    'Merkez X': f"{region['centroid'][0]:.1f}",
                    'Merkez Y': f"{region['centroid'][1]:.1f}",
                    'Genişlik': region['bbox'][2],
                    'Yükseklik': region['bbox'][3]
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        
        # İndirme seçenekleri
        st.subheader("💾 Sonuçları İndir")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # Değişim maskesi
            try:
                mask_bytes = cv2.imencode('.png', results['change_mask'])[1].tobytes()
                st.download_button(
                    "🖼️ Değişim Maskesi",
                    mask_bytes,
                    file_name=f"change_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Mask indirme hatası: {e}")
        
        with col2:
            # Overlay görüntü
            try:
                overlay_rgb = cv2.cvtColor(results['overlay'], cv2.COLOR_RGB2BGR)
                overlay_bytes = cv2.imencode('.png', overlay_rgb)[1].tobytes()
                st.download_button(
                    "🎯 Overlay Görüntü",
                    overlay_bytes,
                    file_name=f"overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Overlay indirme hatası: {e}")
        
        with col3:
            # Güven haritası
            try:
                confidence_img = (results['confidence_map'] * 255).astype(np.uint8)
                conf_bytes = cv2.imencode('.png', confidence_img)[1].tobytes()
                st.download_button(
                    "🌡️ Güven Haritası",
                    conf_bytes,
                    file_name=f"confidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Confidence indirme hatası: {e}")
        
        with col4:
            # JSON rapor
            try:
                report_data = {
                    'analiz_zamani': datetime.now().isoformat(),
                    'yontem': results.get('method', 'N/A'),
                    'degisim_yuzdesi': float(change_pct),
                    'degisen_piksel': int(results.get('changed_pixels', 0)),
                    'toplam_piksel': int(results.get('total_pixels', 0)),
                    'bolge_sayisi': len(results.get('regions', [])),
                    'en_buyuk_5_bolge': results.get('regions', [])[:5],
                    'parametreler': {
                        'duyarlilik': int(sensitivity),
                        'min_bolge_boyutu': int(min_region_size),
                        'morfolojik_temizleme': bool(morphological_ops)
                    },
                    'sistem_durumu': {
                        'gelismis_preprocessor': bool(PREPROCESSOR_AVAILABLE),
                        'utils_modul': bool(UTILS_AVAILABLE),
                        'visualizer_modul': bool(VISUALIZER_AVAILABLE)
                    }
                }
                
                # Ön işleme bilgilerini ekleme
                if 'preprocessing_info' in results:
                    report_data['on_isleme'] = convert_numpy_types(results['preprocessing_info'])
                
                # JSON'u güvenli şekilde serialize etmek
                json_str = json.dumps(report_data, indent=2, ensure_ascii=False, default=str)
                
                st.download_button(
                    "📊 JSON Rapor",
                    json_str,
                    file_name=f"change_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"JSON indirme hatası: {e}")

if __name__ == "__main__":
    main()