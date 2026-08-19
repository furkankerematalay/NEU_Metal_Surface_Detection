import os
import builtins # Python'un Çekirdek DNA motoru
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

builtins.preprocess_input = preprocess_input

from data_loader import val_ds, SINIF_ISIMLERI

# ==========================================
# 1. GÜVENLİK VE HAZIRLIK
# ==========================================
# Çıktıların kaydedileceği klasörü otonom olarak oluştur
RAPOR_KLASORU = "reports"
os.makedirs(RAPOR_KLASORU, exist_ok=True)

KAYITLI_MODEL = "model_resnet50_fine_tuned.keras"

print(f"\n[DONANIM] '{KAYITLI_MODEL}' RAM'e yükleniyor...")
model = load_model(
    KAYITLI_MODEL,
    custom_objects={'preprocess_input': preprocess_input},
    compile=False,
    safe_mode=False
)

y_true = []
y_pred = []

print("[VERİ] Test seti üzerinden tahminler (Inference) çalıştırılıyor. Lütfen bekleyin...")
for images, labels in val_ds:
    y_true.extend(labels.numpy())
    tahmin_olasiliklari = model.predict(images, verbose=0)
    tahmin_edilen_siniflar = np.argmax(tahmin_olasiliklari, axis=1)
    y_pred.extend(tahmin_edilen_siniflar)

# ==========================================
# 2. KARMAŞIKLIK MATRİSİ (CONFUSION MATRIX) GÖRSELLEŞTİRME VE KAYIT
# ==========================================
print("\n[MATEMATİK] Karışıklık Matrisi (Isı Haritası) çiziliyor ve kaydediliyor...")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=SINIF_ISIMLERI,
    yticklabels=SINIF_ISIMLERI
)
plt.title('ResNet50 Industrial Defect Detection - Confusion Matrix', fontweight='bold', pad=15)
plt.ylabel('True Label (Ground Truth)', fontweight='bold')
plt.xlabel('Predicted Label', fontweight='bold')
plt.tight_layout()

# Diske kaydet
cm_kayit_yolu = os.path.join(RAPOR_KLASORU, "resnet_confusion_matrix.png")
plt.savefig(cm_kayit_yolu, dpi=300, bbox_inches='tight')
print(f"[BİLGİ] Karışıklık matrisi mühürlendi: {cm_kayit_yolu}")
plt.show()

# ==========================================
# 3. ALTI SİGMA SINIFLANDIRMA RAPORU (METİN + GÖRSEL)
# ==========================================
print("\n" + "="*60)
print("ALTI SİGMA SINIFLANDIRMA RAPORU (METİN ÖZETİ)")
print("="*60)

# Metin raporunu yazdır
metin_rapor = classification_report(y_true, y_pred, target_names=SINIF_ISIMLERI)
print(metin_rapor)

# Raporu işlemcinin anlayacağı Sözlük (Dict) formatına çevir
veri_raporu = classification_report(y_true, y_pred, target_names=SINIF_ISIMLERI, output_dict=True)

print("\n[MATEMATİK] Sınıflandırma Raporu (Bar Grafik) çiziliyor ve kaydediliyor...")

# Metrikleri listelere ayıkla
precision = [veri_raporu[sinif]['precision'] for sinif in SINIF_ISIMLERI]
recall = [veri_raporu[sinif]['recall'] for sinif in SINIF_ISIMLERI]
f1_score = [veri_raporu[sinif]['f1-score'] for sinif in SINIF_ISIMLERI]

# Grafik çizim mimarisi
x = np.arange(len(SINIF_ISIMLERI))  # Sınıf sayısı kadar X ekseni noktası
width = 0.25                        # Sütun kalınlığı

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(14, 7))

# Sütunları yan yana diz (Matris kaydırma)
bar1 = ax.bar(x - width, precision, width, label='Precision ', color='#2C3E50', alpha=0.9)
bar2 = ax.bar(x, recall, width, label='Recall ', color='#E66101', alpha=0.9)
bar3 = ax.bar(x + width, f1_score, width, label='F1-Score', color='#5E3C99', alpha=0.9)

# Estetik ve etiketler
ax.set_title('Defect-Level Quality Control Metrics', fontsize=15, fontweight='bold', pad=20)
ax.set_ylabel('Metric Score (0.0 - 1.0)', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(SINIF_ISIMLERI, rotation=30, ha='right', fontsize=11, fontweight='bold')
ax.set_ylim(0, 1.15)
ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')

# Değerleri sütunların üstüne yaz (Okunabilirliği artır)
def etiket_ekle(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

etiket_ekle(bar1)
etiket_ekle(bar2)
etiket_ekle(bar3)

plt.tight_layout()

# Diske kaydet
rapor_kayit_yolu = os.path.join(RAPOR_KLASORU, "resnet_classification_report.png")
plt.savefig(rapor_kayit_yolu, dpi=300, bbox_inches='tight')
plt.show()