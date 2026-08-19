import os
import shutil

kaynak_klasor = "NEU-CLS"
hedef_ana_klasör = "organize_veri_Seti"

sınıf = {
    "cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled-in_scale",
    "Sc": "scratches"
}
if not os.path.exists(hedef_ana_klasör):
    for sınıf_adı in sınıf.values():
        yeni_klasör_yolu = os.path.join(hedef_ana_klasör,sınıf_adı)
        os.makedirs(yeni_klasör_yolu,exist_ok=True)


taşınan_sayısı = 0

if  os.path.exists(kaynak_klasor) and os.listdir(kaynak_klasor):
    for dosya_adi in os.listdir(kaynak_klasor):
        if dosya_adi.lower().endswith(('.bmp')):

            dosya_adi_lowered = dosya_adi.lower()

            for kod,sınıf_adı in sınıf.items():
                if kod.lower() in dosya_adi_lowered:
                    eski_konum = os.path.join(kaynak_klasor, dosya_adi)
                    yeni_konum = os.path.join(hedef_ana_klasör,sınıf_adı,dosya_adi)

                    shutil.move(eski_konum,yeni_konum)
                    taşınan_sayısı +=1
                    break
