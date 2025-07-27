# config.py
"""
Streamlit uygulaması için konfigürasyon ve mevcut kod entegrasyonu
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np

class ProjectConfig:
    """Proje konfigürasyon yöneticisi"""
    
    def __init__(self, project_root: str = None):
        self.project_root = project_root or os.path.dirname(os.path.abspath(__file__))
        self.modules = {}
        self.config = self._load_default_config()
        self._setup_paths()
        
    def _setup_paths(self):
        """Python path'lerini ayarlama"""
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Varsayılan konfigürasyonu yükleme"""
        return {
            'project_structure': {
                'src_files': [
                    'preprocessor.py',    # gelişmiş ön işleme
                    'utils.py',          # yardımcı fonksiyonlar  
                    'visualizer.py',     # görselleştirme
                    'demo.py',           # demo fonksiyonları
                    'app.py',            # Streamlit uygulaması
                    'config.py'          # konfigürasyon
                ],
                'root_files': [
                    'test_change_detection.py',  # Ana dizinde - test dosyası
                    'run_app.py',                # Yeni - uygulama başlatıcı
                    'requirements.txt'           # Yeni - paket listesi
                ],
                'flexible_locations': {
                    'test_change_detection.py': ['/', '/src', '/tests'],
                    'demo.py': ['/src', '/'],
                    '__init__.py': ['/src']
                }
            },
            'analysis_methods': {
                'statistical': {
                    'enabled': True,
                    'default_threshold': 30,
                    'description': 'İstatistiksel piksel fark analizi'
                },
                'morphological': {
                    'enabled': True,
                    'kernel_sizes': [3, 5, 7],
                    'description': 'Morfolojik gradyan analizi'
                },
                'advanced_preprocessing': {
                    'enabled': False,  # preprocessor.py'ye bağlı
                    'description': 'Gelişmiş ön işleme + analiz'
                },
                'orb_feature_based': {
                    'enabled': False,  # preprocessor.py'ye bağlı
                    'description': 'ORB özellik tabanlı hizalama + analiz'
                },
                'hybrid_enhanced': {
                    'enabled': True,   # Her zaman kullanılabilir
                    'description': 'Çoklu algoritma hibrit yöntemi'
                }
            },
            'visualization': {
                'default_colormap': 'viridis',
                'overlay_alpha': 0.3,
                'change_color': [255, 50, 50],  # Kırmızı
                'confidence_colors': {
                    'low': [0, 255, 0],    # Yeşil
                    'medium': [255, 255, 0], # Sarı
                    'high': [255, 0, 0]     # Kırmızı
                }
            },
            'processing': {
                'max_image_size': (2048, 2048),
                'supported_formats': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'],
                'noise_reduction': True,
                'morphological_cleaning': True,
                'advanced_preprocessing_available': False  # Runtime'da güncellenecek
            },
            'export': {
                'formats': ['png', 'json', 'csv', 'txt', 'zip'],
                'include_metadata': True,
                'compression_quality': 95,
                'zip_package_enabled': True
            }
        }
    
    def load_existing_modules(self) -> Dict[str, bool]:
        """Mevcut modülleri yükleme ve durumlarını döndürme"""
        module_files = {
            'preprocessor': 'preprocessor.py',
            'utils': 'utils.py',
            'visualizer': 'visualizer.py',
            'demo': 'demo.py',
            'test_change_detection': 'test_change_detection.py'
        }
        
        status = {}
        
        for module_name, file_name in module_files.items():
            file_path = os.path.join(self.project_root, file_name)
            
            if os.path.exists(file_path):
                try:
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.modules[module_name] = module
                    status[module_name] = True
                    
                    # Module'e göre config güncelleme
                    self._update_config_for_module(module_name, module)
                    
                except Exception as e:
                    print(f"⚠️ {module_name} yüklenemedi: {e}")
                    status[module_name] = False
            else:
                status[module_name] = False
        
        return status
    
    def _update_config_for_module(self, module_name: str, module):
        """Modül varlığına göre config'i güncelleme"""
        if module_name == 'preprocessor':
            # Preprocessor'daki fonksiyonları kontrol etme
            if hasattr(module, 'feature_based_detection'):
                self.config['analysis_methods']['feature_based']['enabled'] = True
            
            if hasattr(module, 'deep_learning_detection'):
                self.config['analysis_methods']['deep_learning']['enabled'] = True
            
            if hasattr(module, 'preprocess_images'):
                self.config['processing']['preprocessing_available'] = True
        
        elif module_name == 'visualizer':
            if hasattr(module, 'create_advanced_report'):
                self.config['visualization']['advanced_plots'] = True
        
        elif module_name == 'utils':
            if hasattr(module, 'export_results'):
                self.config['export']['advanced_export'] = True
    
    def get_available_methods(self) -> Dict[str, Dict]:
        """Kullanılabilir analiz yöntemlerini döndürme"""
        return {
            name: method for name, method in self.config['analysis_methods'].items() 
            if method['enabled']
        }
    
    def get_module_function(self, module_name: str, function_name: str):
        """Belirli bir modülden fonksiyon çağırma"""
        if module_name in self.modules:
            module = self.modules[module_name]
            if hasattr(module, function_name):
                return getattr(module, function_name)
        return None
    
    def call_existing_function(self, module_name: str, function_name: str, *args, **kwargs):
        """Mevcut koddaki fonksiyonu çağırma"""
        func = self.get_module_function(module_name, function_name)
        if func:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Fonksiyon çağırma hatası ({module_name}.{function_name}): {e}")
                return None
        return None

class AnalysisIntegrator:
    """Analiz kodlarıyla entegrasyon"""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
    
    def detect_changes_with_existing_code(self, img1: np.ndarray, img2: np.ndarray, 
                                        method: str = 'statistical', **params) -> Dict[str, Any]:

        # Preprocessing uygulama
        processed_img1, processed_img2 = self._apply_preprocessing(img1, img2)
        
        # Yöntem'e göre analiz yapma
        if method.lower() == 'feature_based':
            return self._feature_based_analysis(processed_img1, processed_img2, **params)
        
        elif method.lower() == 'deep_learning':
            return self._deep_learning_analysis(processed_img1, processed_img2, **params)
        
        elif method.lower() == 'morphological':
            return self._morphological_analysis(processed_img1, processed_img2, **params)
        
        else:  # statistical (default)
            return self._statistical_analysis(processed_img1, processed_img2, **params)
    
    def _apply_preprocessing(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocessing uygula"""
        # Mevcut preprocessing fonksiyonunuzu çağırma
        result = self.config.call_existing_function('preprocessor', 'preprocess_images', img1, img2)
        
        if result is not None:
            return result
        
        # Fallback preprocessing
        return self._basic_preprocessing(img1, img2)
    
    def _basic_preprocessing(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Temel preprocessing"""
        import cv2
        
        # Boyut eşitleme
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        target_h, target_w = min(h1, h2), min(w1, w2)
        
        img1_resized = cv2.resize(img1, (target_w, target_h))
        img2_resized = cv2.resize(img2, (target_w, target_h))
        
        # Gürültü azaltma
        if self.config.config['processing']['noise_reduction']:
            img1_resized = cv2.GaussianBlur(img1_resized, (3, 3), 0)
            img2_resized = cv2.GaussianBlur(img2_resized, (3, 3), 0)
        
        return img1_resized, img2_resized
    
    def _feature_based_analysis(self, img1: np.ndarray, img2: np.ndarray, **params) -> Dict[str, Any]:
        """Feature-based analiz"""
        result = self.config.call_existing_function('preprocessor', 'feature_based_detection', img1, img2)
        
        if result is not None:
            return self._standardize_result(result, 'Feature Based')
        
        # Fallback: ORB eşleşme
        return self._orb_feature_analysis(img1, img2)
    
    def _deep_learning_analysis(self, img1: np.ndarray, img2: np.ndarray, **params) -> Dict[str, Any]:
        """Deep learning analiz"""
        result = self.config.call_existing_function('preprocessor', 'deep_learning_detection', img1, img2)
        
        if result is not None:
            return self._standardize_result(result, 'Deep Learning')
        
        # Fallback: istatistik analiz
        return self._statistical_analysis(img1, img2)
    
    def _statistical_analysis(self, img1: np.ndarray, img2: np.ndarray, **params) -> Dict[str, Any]:
        """İstatistiksel analiz"""
        import cv2
        
        threshold = params.get('threshold', self.config.config['analysis_methods']['statistical']['default_threshold'])
        
        # Gri tonlama
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        # Fark hesaplama
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold
        _, change_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Temizleme
        if self.config.config['processing']['morphological_cleaning']:
            kernel = np.ones((3, 3), np.uint8)
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        return self._create_result_dict(change_mask, diff, gray1, gray2, 'Statistical Enhanced')
    
    def _morphological_analysis(self, img1: np.ndarray, img2: np.ndarray, **params) -> Dict[str, Any]:
        """Morfolojik analiz"""
        import cv2
        
        kernel_sizes = params.get('kernel_sizes', self.config.config['analysis_methods']['morphological']['kernel_sizes'])
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        combined_diff = np.zeros_like(gray1, dtype=np.float32)
        
        for size in kernel_sizes:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            grad1 = cv2.morphologyEx(gray1, cv2.MORPH_GRADIENT, kernel)
            grad2 = cv2.morphologyEx(gray2, cv2.MORPH_GRADIENT, kernel)
            diff = cv2.absdiff(grad1, grad2)
            combined_diff += diff.astype(np.float32)
        
        combined_diff = cv2.normalize(combined_diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, change_mask = cv2.threshold(combined_diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return self._create_result_dict(change_mask, combined_diff, gray1, gray2, 'Morphological Enhanced')
    
    def _orb_feature_analysis(self, img1: np.ndarray, img2: np.ndarray) -> Dict[str, Any]:
        """ORB feature tabanlı analiz (fallback)"""
        import cv2
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY) if len(img1.shape) == 3 else img1
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY) if len(img2.shape) == 3 else img2
        
        # ORB detector
        orb = cv2.ORB_create()
        
        # Keypoint ve descriptor bulma
        kp1, des1 = orb.detectAndCompute(gray1, None)
        kp2, des2 = orb.detectAndCompute(gray2, None)
        
        if des1 is not None and des2 is not None:
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            
            matched_kp1 = set([m.queryIdx for m in matches])
            matched_kp2 = set([m.trainIdx for m in matches])
            
            unmatched_kp1 = [kp for i, kp in enumerate(kp1) if i not in matched_kp1]
            unmatched_kp2 = [kp for i, kp in enumerate(kp2) if i not in matched_kp2]
            
            change_mask = np.zeros_like(gray1)
            
            for kp in unmatched_kp1 + unmatched_kp2:
                cv2.circle(change_mask, (int(kp.pt[0]), int(kp.pt[1])), 10, 255, -1)
            
            diff = cv2.absdiff(gray1, gray2)
            
            return self._create_result_dict(change_mask, diff, gray1, gray2, 'ORB Feature Based')
        
        
        return self._statistical_analysis(img1, img2)
    
    def _create_result_dict(self, change_mask: np.ndarray, diff_map: np.ndarray, 
                          gray1: np.ndarray, gray2: np.ndarray, method: str) -> Dict[str, Any]:
        """Standart sonuç sözlüğü oluşturma"""
        total_pixels = change_mask.shape[0] * change_mask.shape[1]
        changed_pixels = np.sum(change_mask > 0)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        confidence_map = cv2.normalize(diff_map.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)
        
        return {
            'change_mask': change_mask,
            'difference_map': diff_map,
            'confidence_map': confidence_map,
            'change_percentage': change_percentage,
            'changed_pixels': changed_pixels,
            'total_pixels': total_pixels,
            'method': method,
            'gray1': gray1,
            'gray2': gray2
        }
    
    def _standardize_result(self, result: Any, method: str) -> Dict[str, Any]:
        """Mevcut koddan gelen sonucu standartlaştırma"""
        # Eğer result zaten dict formatındaysa ve gerekli alanları varsa
        if isinstance(result, dict):
            if all(key in result for key in ['change_mask', 'difference_map']):
                result['method'] = method
                return result
        
        # Aksi halde statistical analiz yapar
        return {'method': method, 'error': 'Standardization failed'}

# Kullanım örneği
def setup_streamlit_integration():
    """Streamlit uygulaması için entegrasyonu hazırlama"""
    config = ProjectConfig()
    module_status = config.load_existing_modules()
    integrator = AnalysisIntegrator(config)
    
    return config, integrator, module_status

# Ana export
__all__ = ['ProjectConfig', 'AnalysisIntegrator', 'setup_streamlit_integration']