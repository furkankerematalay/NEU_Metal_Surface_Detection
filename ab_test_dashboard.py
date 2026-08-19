import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluator:
    """Endüstriyel Yapay Zeka projeleri için statik A/B test görselleştirme sınıfı."""

    def __init__(self):
        # Kurumsal renk paleti ve Sınıf İsimleri
        self.class_names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
        self.colors = {'cnn': '#E66101', 'resnet': '#5E3C99'}
        self.models = ['Custom CNN (V1)', 'ResNet50 (V2)']
        self._apply_style()

    def _apply_style(self):
        """Matplotlib için modern ve temiz bir global tema ayarlar."""
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'figure.titlesize': 16
        })

    def generate_dashboard(self, data, save_path=None):
        """Hazır metrikleri alır ve 2x2 kurumsal bir dashboard üzerinde birleştirir."""
        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Industrial Steel Defect Detection: Custom CNN vs ResNet50 A/B Test',
                     fontweight='bold', color='#2C3E50', y=0.98)

        # 1. Grafiği Çiz: Model Convergence
        self._plot_convergence(axs[0, 0], data['epochs'], data['val_acc_cnn'], data['val_acc_resnet'])

        # 2. Grafiği Çiz: Inference Time
        self._plot_hardware_bar(axs[0, 1], data['inference_time'], 'Hardware Load: Single Frame Inference Time',
                                'Milliseconds (ms)', 'ms')

        # 3. Grafiği Çiz: Classification Report (F1-Score)
        self._plot_f1_comparison(axs[1, 0], data['f1_cnn'], data['f1_resnet'])

        # 4. Grafiği Çiz: Model Complexity
        self._plot_hardware_bar(axs[1, 1], data['parameters'], 'RAM Consumption: Model Complexity',
                                'Trainable Parameters (Millions)', 'M')

        plt.tight_layout(pad=2.0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[BİLGİ] Dashboard başarıyla kaydedildi: {save_path}")

        plt.show()

    def _plot_convergence(self, ax, epochs, val_cnn, val_resnet):
        ax.plot(epochs, val_resnet, marker='o', linestyle='-', color=self.colors['resnet'], label='ResNet50 (V2)')
        ax.plot(epochs, val_cnn, marker='s', linestyle='--', color=self.colors['cnn'], label='Custom CNN (V1)')
        ax.set_title('Model Convergence (Validation Accuracy)', fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')

    def _plot_hardware_bar(self, ax, values, title, ylabel, unit):
        bars = ax.bar(self.models, values, width=0.4, color=[self.colors['cnn'], self.colors['resnet']], alpha=0.85)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f} {unit}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        ax.set_ylim(0, max(values) * 1.15)

    def _plot_f1_comparison(self, ax, f1_cnn, f1_resnet):
        x = np.arange(len(self.class_names))
        width = 0.35

        ax.bar(x - width / 2, f1_cnn, width, label='Custom CNN', color=self.colors['cnn'], alpha=0.85)
        ax.bar(x + width / 2, f1_resnet, width, label='ResNet50', color=self.colors['resnet'], alpha=0.85)

        ax.set_title('Defect-Level Quality Metrics (F1-Score)', fontweight='bold')
        ax.set_ylabel('F1-Score')
        ax.set_xticks(x)
        ax.set_xticklabels(self.class_names, rotation=30, ha='right')
        ax.set_ylim(0, 1.2)
        ax.legend(frameon=True, facecolor='white', edgecolor='none')


# ==========================================
# ÇALIŞTIRMA BLOĞU (HIZLI VE YALIN)
# ==========================================
if __name__ == "__main__":
    import os  # Klasör oluşturma motoru

    epoch_sayisi = np.arange(1, 16)
    np.random.seed(42)

    statik_veriler = {
        'epochs': epoch_sayisi,
        'val_acc_cnn': np.linspace(0.40, 0.82, 15) + np.random.normal(0, 0.02, 15),
        'val_acc_resnet': np.linspace(0.60, 0.96, 15) + np.random.normal(0, 0.01, 15),
        'inference_time': [12.0, 45.0],
        'parameters': [2.5, 23.5],
        'f1_cnn': [0.75, 0.82, 0.78, 0.85, 0.80, 0.72],
        'f1_resnet': [0.92, 0.97, 0.94, 1.00, 0.98, 0.96]
    }

    # 1. GÜVENLİK PROTOKOLÜ: "reports" klasörü yoksa otonom olarak oluştur
    os.makedirs("reports", exist_ok=True)

    # 2. KAYIT EMRİ: save_path parametresi ile resmi fiziksel diske yaz
    evaluator = ModelEvaluator()
    evaluator.generate_dashboard(statik_veriler, save_path="reports/ab_test_dashboard.png")