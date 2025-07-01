# 🔧 CodeSandbox Kurulum Talimatları

## 🚨 **HIZLI ÇÖZÜM - Terminal'de Çalıştır:**

### **Yöntem 1: Otomatik Kurulum (Önerilen)**
```bash
bash codesandbox_setup.sh
```

### **Yöntem 2: Manuel Adımlar**

#### **1. Dependencies Kur:**
```bash
pip install flask gunicorn requests flask-caching pytz
```

#### **2. API Key Güncelle:**
Aşağıdaki dosyalarda eski API key'i değiştir:

**api_routes.py (satır 38):**
```python
# Değiştir:
API_FOOTBALL_KEY = os.environ.get('APIFOOTBALL_API_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# Şununla:
API_FOOTBALL_KEY = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

**main.py (satır 69 ve 287):**
```python
# Değiştir:
api_key = os.environ.get('APIFOOTBALL_API_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# Şununla:
api_key = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

**match_prediction.py (satır 42):**
```python
# Değiştir:
self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# Şununla:
self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

#### **3. Uygulamayı Başlat:**
```bash
# Güvenli başlama (önerilen):
python start_codesandbox.py

# Veya direkt:
python main.py
```

---

## 🔧 **Import Hatası Çözüldü**

Ana problem olan import hatalarını çözdük:
- ✅ `optimized_http_client` import'u güvenli hale getirildi
- ✅ `aiohttp` dependency'si opsiyonel yapıldı
- ✅ Fallback mekanizmaları eklendi
- ✅ Tüm prediction modülleri safe import'la sarıldı

---

## 🔑 **API Key Bilgileri**

### **Yeni API Key (Kullanılacak):**
```
908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
```

### **Environment Variable (Opsiyonel):**
```bash
export APIFOOTBALL_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
```

---

## 🛠️ **Sorun Giderme**

### **Problem: Import Hatası**
**Çözüm:** `main.py` artık tüm import'ları güvenli try-catch ile yapıyor

### **Problem: aiohttp bulunamıyor**
**Çözüm:** Artık opsiyonel, basic requests kullanılıyor

### **Problem: API Key Çalışmıyor**
**Çözüm:** Yukarıdaki adımlarla API key'i güncelle

### **Problem: 502 Bad Gateway**
**Çözüm:** 
1. Terminal'de Ctrl+C ile durdur
2. `python start_codesandbox.py` ile başlat

---

## ✅ **Başarı Kontrol Listesi**

- [ ] Dependencies kuruldu
- [ ] API key'ler 3 dosyada güncellendi  
- [ ] Import hataları yok
- [ ] Uygulama başlatıldı
- [ ] 502 hatası gitti
- [ ] Preview çalışıyor

---

## 🚀 **Hızlı Başlatma Komutları**

```bash
# Tek komutla her şeyi yap:
bash codesandbox_setup.sh && python start_codesandbox.py

# Manuel kontrol:
python -c "import requests; print('✅ Requests OK')"
python -c "import flask; print('✅ Flask OK')"
python -c "from main import app; print('✅ App OK')"
```

---

**Artık CodeSandbox'ta uygulaman çalışmalı! 🎉**