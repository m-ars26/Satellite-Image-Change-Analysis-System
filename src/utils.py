"""
🛰️ Uydu Görüntü Değişiklik Tespit Utility Modülü
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Union
import numpy as np
import cv2


def create_directory(path: Union[str, Path], verbose: bool = True) -> Path:
    """
    Verilen dizin yoksa oluşturur.

    Argümanlar:
        path: Dizin yolu
        verbose: True ise oluşturulan dizini terminale yazar

    Dönüş:
        Path nesnesi olarak dizin yolu
    """
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"📁 Dizin oluşturuldu: {path}")
    return path


def get_image_files(directory: Union[str, Path], 
                   extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'],
                   recursive: bool = False) -> List[str]:
    """
    Belirtilen klasördeki görüntü dosyalarını listeler.

    Argümanlar:
        directory: Hedef klasör
        extensions: Dikkate alınacak uzantılar
        recursive: Alt klasörler de taransın mı

    Dönüş:
        Görüntü dosyalarının yollarını içeren liste
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    image_files = []
    
    if recursive:
        for ext in extensions:
            image_files.extend(directory.rglob(f'*{ext}'))
            image_files.extend(directory.rglob(f'*{ext.upper()}'))
    else:
        for ext in extensions:
            image_files.extend(directory.glob(f'*{ext}'))
            image_files.extend(directory.glob(f'*{ext.upper()}'))
    
    return list(set([str(f) for f in image_files]))


def print_progress(current: int, total: int, prefix: str = "İlerleme", 
                  bar_length: int = 50) -> None:
    """
    Konsola bir ilerleme çubuğu basar.

    Argümanlar:
        current: Şu ana kadar işlenen öğe sayısı
        total: Toplam öğe sayısı
        prefix: Başlık
        bar_length: Çubuğun uzunluğu
    """
    percent = float(current) / total
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    sys.stdout.write(f'\r{prefix}: |{bar}| {percent*100:.1f}% ({current}/{total})')
    sys.stdout.flush()
    
    if current == total:
        print()  # Bittiğinde yeni satıra geç


class AdvancedUtils:
    """Gelişmiş yardımcı işlevlerin yer aldığı sınıf"""
    
    @staticmethod
    def create_directory(path: Union[str, Path], verbose: bool = True) -> Path:
        """Dizin oluşturur ve Path nesnesi döner"""
        return create_directory(path, verbose)
    
    @staticmethod
    def get_image_files(directory: Union[str, Path], 
                       recursive: bool = False,
                       filter_corrupted: bool = True) -> List[Path]:
        """
        Görüntü dosyalarını Path formatında döner. İstenirse bozuk dosyaları ayıklar.

        Argümanlar:
            directory: Dizin
            recursive: Alt dizinlere de bakılsın mı
            filter_corrupted: Bozuk dosyalar ayıklansın mı

        Dönüş:
            Geçerli görüntü dosyalarının listesi
        """
        files = get_image_files(directory, recursive=recursive)
        path_files = [Path(f) for f in files]
        
        if filter_corrupted:
            valid_files = []
            for file_path in path_files:
                try:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        valid_files.append(file_path)
                except:
                    print(f"⚠️ Bozuk dosya atlandı: {file_path.name}")
            return valid_files
        
        return path_files
    
    @staticmethod
    def save_metadata(metadata: Dict[str, Any], path: Union[str, Path]) -> None:
        """
        Verilen sözlüğü JSON dosyası olarak kaydeder.

        Argümanlar:
            metadata: Kaydedilecek veriler
            path: Çıktı dosyası
        """
        path = Path(path)
        
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        metadata_converted = convert_types(metadata)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(metadata_converted, f, indent=2, ensure_ascii=False, default=str)
    
    @staticmethod
    def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
        """
        JSON dosyasından metadata yükler.

        Argümanlar:
            path: Girdi dosya yolu

        Dönüş:
            Yüklenen sözlük
        """
        path = Path(path)
        if not path.exists():
            return {}
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def print_progress_advanced(current: int, total: int, prefix: str = "İlerleme",
                              suffix: str = "", decimals: int = 1, 
                              bar_length: int = 50, fill: str = '█',
                              print_end: str = "\r") -> None:
        """
        Daha özelleştirilebilir bir ilerleme çubuğu bastırır.

        Argümanlar:
            current: Şu anki ilerleme
            total: Toplam sayı
            prefix: Başlık
            suffix: Sonuna eklenecek yazı
            decimals: Yüzdelik hassasiyet
            bar_length: Çubuğun uzunluğu
            fill: Dolgu karakteri
            print_end: Satır sonu karakteri
        """
        percent = f"{100 * (current / float(total)):.{decimals}f}"
        filled_length = int(bar_length * current // total)
        bar = fill * filled_length + '-' * (bar_length - filled_length)
        
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
        
        if current == total:
            print()
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path], 
                          algorithm: str = 'md5') -> str:
        """
        Dosyanın hash (özeti) alınır, bütünlük kontrolü için kullanılabilir.

        Argümanlar:
            file_path: Dosya yolu
            algorithm: Kullanılacak algoritma ('md5' ya da 'sha256')

        Dönüş:
            Hash string’i
        """
        import hashlib
        
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        
        hash_func = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    @staticmethod
    def format_bytes(bytes_size: int) -> str:
        """
        Byte cinsinden gelen değeri okunabilir hale getirir.

        Argümanlar:
            bytes_size: Byte değeri

        Dönüş:
            İnsan okunabilir string
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.2f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.2f} PB"
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Sistem ile ilgili bazı temel bilgileri döner"""
        import platform
        
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'opencv_version': cv2.__version__
        }
        
        try:
            import psutil
            memory = psutil.virtual_memory()
            info['memory_total'] = AdvancedUtils.format_bytes(memory.total)
            info['memory_available'] = AdvancedUtils.format_bytes(memory.available)
            info['memory_percent'] = f"{memory.percent}%"
        except ImportError:
            pass
        
        try:
            import torch
            info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info['cuda_device'] = torch.cuda.get_device_name(0)
        except ImportError:
            pass
        
        return info
    
    @staticmethod
    def validate_image_pair(img1_path: Union[str, Path], 
                          img2_path: Union[str, Path]) -> Dict[str, Any]:
        """
        İki görüntü arasında işlenebilirlik kontrolü yapar.

        Argümanlar:
            img1_path: İlk görüntü
            img2_path: İkinci görüntü

        Dönüş:
            Geçerlilik, uyarılar ve hata mesajlarını içeren sözlük
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'img1_info': {},
            'img2_info': {}
        }
        
        for idx, img_path in enumerate([img1_path, img2_path]):
            img_path = Path(img_path)
            info_key = f'img{idx+1}_info'
            
            if not img_path.exists():
                results['valid'] = False
                results['errors'].append(f"Görüntü {idx+1} mevcut değil: {img_path}")
                continue
            
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    results['valid'] = False
                    results['errors'].append(f"Görüntü {idx+1} okunamıyor: {img_path}")
                    continue
                
                results[info_key] = {
                    'path': str(img_path),
                    'shape': img.shape,
                    'dtype': str(img.dtype),
                    'size': AdvancedUtils.format_bytes(img_path.stat().st_size),
                    'channels': img.shape[2] if len(img.shape) == 3 else 1
                }
                
            except Exception as e:
                results['valid'] = False
                results['errors'].append(f"Görüntü {idx+1} okunurken hata: {str(e)}")
        
        if results['valid'] and results['img1_info'] and results['img2_info']:
            shape1 = results['img1_info']['shape']
            shape2 = results['img2_info']['shape']
            
            if shape1[:2] != shape2[:2]:
                results['warnings'].append(
                    f"Görüntü boyutları farklı: {shape1[:2]} vs {shape2[:2]}. "
                    "İşlem öncesi yeniden boyutlandırılacak."
                )
            
            if shape1[2:] != shape2[2:]:
                results['warnings'].append(
                    f"Kanal sayıları farklı: {shape1[2:]} vs {shape2[2:]}. "
                    "İşlem için uyumlu hale getirilecek."
                )
        
        return results
