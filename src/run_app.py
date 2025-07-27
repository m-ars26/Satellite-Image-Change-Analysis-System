#!/usr/bin/env python3
"""
Uydu Görüntü Değişim Analizi - Streamlit Uygulaması Başlatıcı
Bu dosyayı projenizin ana klasöründe çalıştırın
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def check_requirements():
    """Gerekli paketlerin yüklü olup olmadığını kontrol eder"""
    required_packages = [
        'streamlit',
        'opencv-python', 
        'pillow',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    return missing_packages

def install_requirements(missing_packages):
    """Eksik paketleri yükler"""
    if missing_packages:
        print("📦 Eksik paketler tespit edildi. Yükleniyor...")
        for package in missing_packages:
            print(f"  - Yükleniyor: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ Tüm paketler başarıyla yüklendi!\n")

def check_project_structure():
    """Proje yapısını kontrol eder"""
    current_dir = Path.cwd()
    src_dir = current_dir / "src"
    
    print("🔍 Proje yapısı kontrol ediliyor...")
    
    # src klasörünü kontrol etme
    if not src_dir.exists():
        print("❌ 'src' klasörü bulunamadı!")
        print("📁 Lütfen aşağıdaki yapıyı oluşturun:")
        print("   project_root/")
        print("   ├── src/")
        print("   │   ├── app.py")
        print("   │   ├── config.py") 
        print("   │   ├── preprocessor.py  (mevcut)")
        print("   │   ├── utils.py         (mevcut)")
        print("   │   ├── visualizer.py    (mevcut)")
        print("   │   ├── demo.py          (mevcut)")
        print("   │   └── test_change_detection.py (mevcut)")
        print("   ├── run_app.py")
        print("   └── requirements.txt")
        return False
    
    # Gerekli dosyaları kontrol etme
    required_files = ['app.py']
    existing_files = ['preprocessor.py', 'utils.py', 'visualizer.py', 'demo.py', 'test_change_detection.py']
    
    missing_files = []
    found_existing = []
    
    for file in required_files:
        if not (src_dir / file).exists():
            missing_files.append(file)
    
    for file in existing_files:
        if (src_dir / file).exists():
            found_existing.append(file)
    
    print(f"📂 src/ klasörü bulundu: {src_dir}")
    
    if found_existing:
        print("✅ Mevcut dosyalar:")
        for file in found_existing:
            print(f"   - {file}")
    
    if missing_files:
        print("⚠️ Eksik dosyalar:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ Proje yapısı uygun!\n")
    return True

def create_missing_files():
    """Eksik dosyaları oluşturma"""
    src_dir = Path.cwd() / "src"
    
    # src klasörünü oluşturur
    src_dir.mkdir(exist_ok=True)
    
    # __init__.py oluşturur
    init_file = src_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
        print("📄 __init__.py oluşturuldu")

def show_run_instructions():
    """Çalıştırma talimatlarını göster"""
    print("🚀 UYGULAMA BAŞLATMA TALİMATLARI")
    print("=" * 50)
    print()
    print("1️⃣ Terminal/Komut İstemi'ni açın")
    print("2️⃣ Proje klasörünüze gidin:")
    print("   cd /path/to/your/project")
    print()
    print("3️⃣ Streamlit uygulamasını başlatın:")
    print("   streamlit run src/app.py")
    print()
    print("4️⃣ Tarayıcınızda otomatik olarak açılacak adres:")
    print("   http://localhost:8501")
    print()
    print("💡 İPUÇLARI:")
    print("   - İlk çalıştırmada tarayıcı otomatik açılır")
    print("   - Uygulamayı durdurmak için Ctrl+C")
    print("   - Kodda değişiklik yaptığınızda otomatik yenilenir")
    print("   - Farklı port kullanmak için: streamlit run src/app.py --server.port 8502")
    print()

def run_streamlit_app():
    """Streamlit uygulamasını başlatır"""
    src_dir = Path.cwd() / "src"
    app_file = src_dir / "app.py"
    
    if not app_file.exists():
        print("❌ src/app.py dosyası bulunamadı!")
        return False
    
    print("🚀 Streamlit uygulaması başlatılıyor...")
    print("📍 Dosya konumu:", app_file)
    print("🌐 Tarayıcınızda http://localhost:8501 adresinde açılacak")
    print("⏹️ Durdurmak için Ctrl+C tuşlarına basın")
    print("=" * 50)
    
    try:
        # Streamlit uygulamasını çalıştırır
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_file),
            "--server.headless", "false",
            "--server.fileWatcherType", "auto",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Uygulama başlatılamadı: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Uygulama durduruldu.")
        return True

def main():
    """Ana fonksiyon"""
    print("🛰️ UYDU GÖRÜNTÜ DEĞİŞİM ANALİZİ - STREAMLIT UYGULAMASI")
    print("=" * 60)
    print()
    
    # Gereksinimler kontrolü
    print("1️⃣ Gerekli paketler kontrol ediliyor...")
    missing = check_requirements()
    
    if missing:
        response = input(f"📦 Eksik paketler: {', '.join(missing)}\n   Yüklemek istiyor musunuz? (y/n): ")
        if response.lower() in ['y', 'yes', 'evet', 'e']:
            install_requirements(missing)
        else:
            print("❌ Gerekli paketler yüklenmeden uygulama çalışmaz!")
            return
    else:
        print("✅ Tüm gerekli paketler yüklü!")
    
    print()
    
    # Proje yapısı kontrolü
    print("2️⃣ Proje yapısı kontrol ediliyor...")
    structure_ok = check_project_structure()
    
    if not structure_ok:
        response = input("📁 Eksik klasör/dosyalar var. Oluşturmaya çalışayım mı? (y/n): ")
        if response.lower() in ['y', 'yes', 'evet', 'e']:
            create_missing_files()
            print("📁 Temel yapı oluşturuldu. Lütfen app.py dosyasını src/ klasörüne kopyalayın.")
            show_run_instructions()
        else:
            print("❌ Proje yapısı uygun değil!")
        return
    
    print()
    
    # Uygulama başlatma seçenekleri
    print("3️⃣ Uygulama başlatma seçenekleri:")
    print("   [1] Streamlit uygulamasını başlat")
    print("   [2] Sadece talimatları göster")
    print("   [3] Çıkış")
    
    choice = input("Seçiminiz (1/2/3): ").strip()
    
    if choice == "1":
        run_streamlit_app()
    elif choice == "2":
        show_run_instructions()
    elif choice == "3":
        print("👋 Görüşmek üzere!")
    else:
        print("❌ Geçersiz seçim!")

def quick_start():
    """Hızlı başlatma (parametresiz)"""
    missing = check_requirements()
    if missing:
        print(f"❌ Eksik paketler: {', '.join(missing)}")
        print("Lütfen önce: pip install streamlit opencv-python pillow plotly pandas numpy")
        return
    
    if not check_project_structure():
        print("❌ Proje yapısı uygun değil!")
        return
    
    run_streamlit_app()

if __name__ == "__main__":
    # Komut satırı argümanlarını kontrol eder
    if len(sys.argv) > 1:
        if sys.argv[1] == "--quick" or sys.argv[1] == "-q":
            quick_start()
        elif sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print("Kullanım:")
            print("  python run_app.py          # İnteraktif menü")
            print("  python run_app.py --quick  # Hızlı başlatma")
            print("  python run_app.py --help   # Bu yardım")
        else:
            print("❌ Bilinmeyen parametre. --help kullanın.")
    else:
        main()