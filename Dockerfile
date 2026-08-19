# 1. TEMEL DONANIM: En hafif Python işletim sistemi
FROM python:3.11-slim

# 2. ÇALIŞMA ALANI: Konteynerin içindeki fabrikamızı kuruyoruz
WORKDIR /app

# (DİKKAT: apt-get ile C++ kütüphaneleri kurma aşamasını sildik. Artık çok daha hızlıyız!)

# 3. REÇETEYİ İÇERİ AL VE KUR
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. TÜM KODLARI VE BEYNİ (Ağırlıkları) İÇERİ KOPYALA
COPY . .

# 5. DIŞ DÜNYAYA AÇILAN KAPI
EXPOSE 8000

# 6. MOTORU ATEŞLEME EMRİ
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]