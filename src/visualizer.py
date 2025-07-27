"""
🎨 Uydu Görüntü Sonuç Görselleştirme Modülü
Profesyonel raporlama ve görselleştirme araçları
"""

# Gerekli kütüphaneleri içe aktarıyorum
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime
import json

class ResultVisualizer:
    """
    Değişiklik tespit sonuçlarını görselleştirmek için yazdığım ana sınıf
    """
    
    def __init__(self, style: str = 'modern', save_path: str = 'results'):
        """
        Başlangıçta stil seçimini yapıyor ve gerekli ayarları yüklüyorum
        """
        self.style = style
        self.save_path = save_path
        self.setup_style()
        
        # Renk paletini burada belirledim. Her renk, raporda farklı anlam taşıyor.
        self.colors = {
            'change': '#FF4444',      # Değişiklik bölgeleri için kırmızı
            'stable': '#44FF44',      # Sabit kalan alanlar için yeşil
            'uncertain': '#FFAA44',   # Belirsiz alanlar için turuncu
            'background': '#2E2E2E',  # Arka plan rengi (grafikler için)
            'text': '#FFFFFF',        # Yazı rengi
            'accent': '#00AAFF'       # Vurgulayıcı mavi
        }
        
        # Değişiklik türlerine özel renkler
        self.change_type_colors = {
            'major_structure': '#FF0000',
            'area_development': '#FF8800',
            'building_change': '#FFFF00',
            'linear_feature': '#00FF00',
            'minor_change': '#0088FF'
        }
        
        print(f"🎨 ResultVisualizer başlatıldı. Stil: {style}")
    
    def setup_style(self):
        """Matplotlib stilini burada ayarlıyorum"""
        if self.style == 'modern':
            plt.style.use('dark_background')
            sns.set_palette("husl")  # Daha renkli bir tema
        elif self.style == 'scientific':
            plt.style.use('seaborn-v0_8')  # Bilimsel görünüm
        else:
            plt.style.use('default')
        
        # Grafiklerin yazı boyutlarını ve diğer stil detaylarını buradan güncelliyorum
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14
        })

    def create_comprehensive_report(self, 
                                    image_before: np.ndarray,
                                    image_after: np.ndarray,
                                    results: Dict[str, Any],
                                    title: str = "Satellite Image Change Detection Report") -> str:
        """
        Burası modülün ana fonksiyonu. Tüm sonuçları toplayıp kapsamlı bir grafiksel rapor üretiyor.
        """
        print("📊 Kapsamlı rapor oluşturuluyor...")
        
        # Büyük boyutlu bir figür başlatıyorum
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # 4x5'lik grid yapısı kuruyorum. Bu yapıya göre tüm grafikleri yerleştiriyorum.
        gs = fig.add_gridspec(4, 5, hspace=0.3, wspace=0.3)
        
        # İlk satırda önceki ve sonraki görüntüler + maske + güven haritası
        self._plot_main_images(fig, gs, image_before, image_after, results)
        
        # İkinci satırda yöntem karşılaştırmaları var
        self._plot_method_comparison(fig, gs, results)
        
        # Üçüncü satırda çeşitli analiz grafikleri
        self._plot_analysis_charts(fig, gs, results)
        
        # Dördüncü satırda istatistik tablo + bölge haritası
        self._plot_detailed_stats(fig, gs, results, image_before.shape)
        
        # En alta timestamp yazıyorum
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8, alpha=0.7)
        
        # Son olarak dosyayı .png olarak kaydediyorum
        filename = f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print(f"✅ Rapor kaydedildi: {filepath}")
        return filepath

    def _plot_main_images(self, fig, gs, image_before, image_after, results):
        """Önceki, sonraki görüntü, maske, güven haritası ve bindirme görsellerini çiziyorum"""
        
        # Önceki uydu görüntüsünü gösteriyorum
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image_before, cv2.COLOR_BGR2RGB))
        ax1.set_title("Before Image", fontweight='bold')
        ax1.axis('off')
        
        # Sonraki görüntüyü ekliyorum
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(cv2.cvtColor(image_after, cv2.COLOR_BGR2RGB))
        ax2.set_title("After Image", fontweight='bold')
        ax2.axis('off')
        
        # Değişiklik maskesini renklendirip gösteriyorum
        ax3 = fig.add_subplot(gs[0, 2])
        if results.get('change_mask') is not None:
            change_colored = self._colorize_change_mask(results['change_mask'])
            ax3.imshow(change_colored)
        ax3.set_title("Detected Changes", fontweight='bold')
        ax3.axis('off')
        
        # Güven haritası varsa ekliyorum
        ax4 = fig.add_subplot(gs[0, 3])
        if results.get('confidence_map') is not None:
            im = ax4.imshow(results['confidence_map'], cmap='plasma', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
        ax4.set_title("Confidence Map", fontweight='bold')
        ax4.axis('off')
        
        # Değişiklikleri görüntü üzerine bindirip gösteriyorum
        ax5 = fig.add_subplot(gs[0, 4])
        if results.get('change_mask') is not None:
            overlay = self._create_overlay(image_after, results['change_mask'])
            ax5.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax5.set_title("Changes Overlay", fontweight='bold')
        ax5.axis('off')

    def _plot_method_comparison(self, fig, gs, results):
        """Farklı yöntemlerle elde edilen değişiklik maskelerini yan yana karşılaştırıyorum"""
        if 'method_results' not in results:
            return
        
        methods = list(results['method_results'].keys())
        for i, method in enumerate(methods[:4]):
            ax = fig.add_subplot(gs[1, i])
            mask = results['method_results'][method]['mask']
            ax.imshow(mask, cmap='hot')
            ax.set_title(f"{method.replace('_', ' ').title()}", fontweight='bold')
            ax.axis('off')
        
        # Son kutucukta performans çubuğunu gösteriyorum
        if len(methods) < 4:
            ax = fig.add_subplot(gs[1, len(methods)])
            self._plot_method_performance(ax, results)

    def _plot_analysis_charts(self, fig, gs, results):
        """3. satırdaki grafiksel analizleri çiziyorum"""
        self._plot_region_distribution(fig.add_subplot(gs[2, 0]), results)
        self._plot_change_types_pie(fig.add_subplot(gs[2, 1]), results)
        self._plot_confidence_histogram(fig.add_subplot(gs[2, 2]), results)
        self._plot_area_analysis(fig.add_subplot(gs[2, 3]), results)
        self._plot_summary_metrics(fig.add_subplot(gs[2, 4]), results)

    def _plot_detailed_stats(self, fig, gs, results, image_shape):
        """İstatistik tablo ve değişiklik bölgesi haritasını en alt satıra yerleştiriyorum"""
        self._create_stats_table(fig.add_subplot(gs[3, :3]), results, image_shape)
        self._plot_change_regions_map(fig.add_subplot(gs[3, 3:]), results)

    def _colorize_change_mask(self, change_mask: np.ndarray) -> np.ndarray:
        """Değişiklik maskesini kırmızıya boyuyorum (görsellik için)"""
        colored = np.zeros((*change_mask.shape, 3), dtype=np.uint8)
        colored[change_mask > 0] = [255, 68, 68]
        return colored

    def _create_overlay(self, base_image: np.ndarray, change_mask: np.ndarray) -> np.ndarray:
        """Orijinal görüntünün üzerine değişiklikleri kırmızı olarak bindiriyorum"""
        overlay = base_image.copy()
        overlay[change_mask > 0] = [0, 0, 255]
        alpha = 0.7
        return cv2.addWeighted(base_image, alpha, overlay, 1-alpha, 0)

    def _plot_region_distribution(self, ax, results):
        """Değişiklik bölgelerinin büyüklük dağılımını histogram olarak çiziyorum"""
        if not results.get('change_regions'):
            ax.text(0.5, 0.5, 'No regions detected', ha='center', va='center')
            ax.set_title("Region Size Distribution")
            return
        areas = [region['area'] for region in results['change_regions']]
        ax.hist(areas, bins=min(10, len(areas)), alpha=0.7, color=self.colors['change'])
        ax.set_xlabel("Area (pixels)")
        ax.set_ylabel("Count")
        ax.set_title("Region Size Distribution")
        ax.grid(True, alpha=0.3)

    def _plot_change_types_pie(self, ax, results):
        """Değişiklik türlerini pasta grafiği şeklinde gösteriyorum"""
        if not results.get('change_regions'):
            ax.text(0.5, 0.5, 'No regions detected', ha='center', va='center')
            ax.set_title("Change Types Distribution")
            return
        type_counts = {}
        for region in results['change_regions']:
            change_type = region['change_type']
            type_counts[change_type] = type_counts.get(change_type, 0) + 1
        if type_counts:
            colors = [self.change_type_colors.get(t, '#888888') for t in type_counts.keys()]
            ax.pie(type_counts.values(), labels=type_counts.keys(),
                   colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title("Change Types Distribution")

    def _plot_confidence_histogram(self, ax, results):
        """Güven skoru dağılımını histogram olarak çiziyorum"""
        if results.get('confidence_map') is None:
            ax.text(0.5, 0.5, 'No confidence data', ha='center', va='center')
            ax.set_title("Confidence Distribution")
            return
        conf_flat = results['confidence_map'].flatten()
        conf_nonzero = conf_flat[conf_flat > 0.01]
        if len(conf_nonzero) > 0:
            ax.hist(conf_nonzero, bins=50, alpha=0.7, color=self.colors['accent'])
            ax.set_xlabel("Confidence")
            ax.set_ylabel("Frequency")
            ax.set_title("Confidence Distribution")
            ax.grid(True, alpha=0.3)
            mean_conf = np.mean(conf_nonzero)
            ax.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.3f}')
            ax.legend()

    def _plot_area_analysis(self, ax, results):
        """En büyük 10 değişiklik bölgesini çubuk grafikte gösteriyorum"""
        if not results.get('change_regions'):
            ax.text(0.5, 0.5, 'No regions detected', ha='center', va='center')
            ax.set_title("Area Analysis")
            return
        top_regions = sorted(results['change_regions'], key=lambda x: x['area'], reverse=True)[:10]
        areas = [r['area'] for r in top_regions]
        labels = [f"{r['change_type'][:8]}_{r['id']}" for r in top_regions]
        colors = [self.change_type_colors.get(r['change_type'], '#888888') for r in top_regions]
        bars = ax.bar(range(len(areas)), areas, color=colors)
        ax.set_xlabel("Region ID")
        ax.set_ylabel("Area (pixels)")
        ax.set_title("Top 10 Largest Regions")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)

    def _plot_summary_metrics(self, ax, results):
        """Genel metrikleri yatay çubuk grafik olarak sunuyorum"""
        if 'statistics' not in results:
            ax.text(0.5, 0.5, 'No statistics available', ha='center', va='center')
            ax.set_title("Summary Metrics")
            return
        stats = results['statistics']
        metrics = {
            'Change %': stats.get('change_percentage', 0),
            'Avg Confidence': stats.get('mean_confidence', 0) * 100,
            'Regions': stats.get('num_change_regions', 0),
            'Max Confidence': stats.get('max_confidence', 0) * 100
        }
        y_pos = np.arange(len(metrics))
        values = list(metrics.values())
        bars = ax.barh(y_pos, values, color=[
            self.colors['change'],
            self.colors['accent'],
            self.colors['uncertain'],
            self.colors['stable']
        ])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metrics.keys())
        ax.set_xlabel("Value")
        ax.set_title("Summary Metrics")
        ax.grid(True, alpha=0.3)
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.01, i, f'{v:.1f}', va='center')

    def _plot_method_performance(self, ax, results):
        """Her yöntemin ne kadar değişiklik tespit ettiğini görselleştiriyorum"""
        if 'method_results' not in results:
            return
        methods = list(results['method_results'].keys())
        performances = []
        for method in methods:
            mask = results['method_results'][method]['mask']
            change_percent = (np.sum(mask > 0) / mask.size) * 100
            performances.append(change_percent)
        bars = ax.bar(methods, performances, color=self.colors['accent'])
        ax.set_ylabel("Change % Detected")
        ax.set_title("Method Performance")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        for bar, perf in zip(bars, performances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(performances)*0.01,
                    f'{perf:.1f}%', ha='center', va='bottom')

    def _create_stats_table(self, ax, results, image_shape):
        """İstatistik verilerini tabloya çevirip gösteriyorum"""
        ax.axis('tight')
        ax.axis('off')
        stats_data = []
        if 'statistics' in results:
            stats = results['statistics']
            stats_data.extend([
                ['Total Image Size', f"{image_shape[0]} x {image_shape[1]} pixels"],
                ['Total Pixels', f"{image_shape[0] * image_shape[1]:,}"],
                ['Change Percentage', f"{stats.get('change_percentage', 0):.2f}%"],
                ['Changed Pixels', f"{stats.get('total_change_area', 0):,}"],
                ['Number of Regions', f"{stats.get('num_change_regions', 0)}"],
                ['Mean Confidence', f"{stats.get('mean_confidence', 0):.3f}"],
                ['Max Confidence', f"{stats.get('max_confidence', 0):.3f}"],
                ['Confidence Std', f"{stats.get('confidence_std', 0):.3f}"]
            ])
        if results.get('change_regions'):
            top_3 = sorted(results['change_regions'], key=lambda x: x['area'], reverse=True)[:3]
            stats_data.append(['', ''])
            stats_data.append(['TOP REGIONS', ''])
            for i, region in enumerate(top_3, 1):
                stats_data.append([f"Region {i}", f"{region['change_type']} ({region['area']} px)"])
        table = ax.table(cellText=stats_data, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center', colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else 'white')
        ax.set_title("Detailed Statistics", fontweight='bold', pad=20)

    def _plot_change_regions_map(self, ax, results):
        """Değişiklik bölgelerinin konumlarını harita gibi çiziyorum"""
        if not results.get('change_regions') or results.get('change_mask') is None:
            ax.text(0.5, 0.5, 'No regions to display', ha='center', va='center')
            ax.set_title("Change Regions Map")
            return
        ax.imshow(results['change_mask'], cmap='gray', alpha=0.7)
        for region in results['change_regions'][:10]:
            x, y, w, h = region['bbox']
            color = self.change_type_colors.get(region['change_type'], '#888888')
            rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                     edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x + w/2, y - 5, f"{region['id']}", ha='center', va='bottom',
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.2",
                    facecolor=color, alpha=0.7))
        ax.set_title("Change Regions Map (Top 10)")
        ax.axis('off')

    def save_results_json(self, results: Dict[str, Any], filename: str = None) -> str:
        """Sonuçları JSON formatında kaydediyorum"""
        if filename is None:
            filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.save_path, filename)
        json_results = self._convert_for_json(results)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"📁 JSON sonuçları kaydedildi: {filepath}")
        return filepath

    def _convert_for_json(self, obj):
        """Numpy veri tiplerini JSON'a uygun hale çeviriyorum"""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj

def test_visualizer():
    """Modülü test etmek için örnek veriyle demo raporu oluşturuyorum"""
    print("🧪 ResultVisualizer Test Başlatıldı...")
    
    # Sahte test verileri üretiyorum
    dummy_results = {
        'change_mask': np.random.randint(0, 256, (300, 300), dtype=np.uint8),
        'confidence_map': np.random.rand(300, 300),
        'statistics': {
            'change_percentage': 25.5,
            'mean_confidence': 0.75,
            'max_confidence': 0.95,
            'confidence_std': 0.12,
            'num_change_regions': 8,
            'total_change_area': 22950
        },
        'change_regions': [
            {'id': 1, 'area': 5000, 'bbox': (50, 50, 100, 80), 'change_type': 'major_structure'},
            {'id': 2, 'area': 3000, 'bbox': (200, 100, 60, 60), 'change_type': 'building_change'},
            {'id': 3, 'area': 1500, 'bbox': (80, 200, 120, 40), 'change_type': 'linear_feature'}
        ],
        'method_results': {
            'statistical': {'mask': np.random.randint(0, 256, (300, 300))},
            'morphological': {'mask': np.random.randint(0, 256, (300, 300))},
            'feature_based': {'mask': np.random.randint(0, 256, (300, 300))}
        }
    }
    
    # Rastgele görüntüler üretiyorum
    img_before = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img_after = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Visualizer sınıfını başlatıp raporu oluşturuyorum
    visualizer = ResultVisualizer(style='modern')
    report_path = visualizer.create_comprehensive_report(
        img_before, img_after, dummy_results, "Test Report - Visualizer Demo")
    json_path = visualizer.save_results_json(dummy_results, "test_results.json")
    
    print("✅ Visualizer test tamamlandı!")
    print(f"📊 Rapor: {report_path}")
    print(f"📁 JSON: {json_path}")

# Kod direkt çalıştırıldığında test fonksiyonunu başlatıyorum
if __name__ == "__main__":
    test_visualizer()
