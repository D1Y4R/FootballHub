# 🚀 Football Prediction App - Render Deployment

## 📋 İçindekiler
- [🌐 Render'da Deployment](#render-deployment)
- [⚙️ Kurulum Adımları](#kurulum-adımları)
- [🔧 Konfigürasyon](#konfigürasyon)
- [🐛 Sorun Giderme](#sorun-giderme)

---

## 🌐 Render Deployment

### **Adım 1: GitHub Repository Hazırla**

1. **GitHub'a projeyi yükle:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit for Render deployment"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/football-prediction-app.git
   git push -u origin main
   ```

### **Adım 2: Render Hesabı Oluştur**

1. **[render.com](https://render.com)** sitesine git
2. **"Get Started for Free"** tıkla
3. **GitHub ile giriş yap**

### **Adım 3: Web Service Oluştur**

1. Render dashboard'da **"New +"** → **"Web Service"**
2. **"Build and deploy from a Git repository"** seç
3. GitHub repository'nizi seç
4. Aşağıdaki ayarları yap:

#### **⚙️ Service Ayarları:**
```
Name: football-prediction-app
Environment: Python 3
Region: Oregon (US West)
Branch: main
Build Command: pip install -r requirements.txt
Start Command: python run.py
```

#### **🔧 Environment Variables:**
```
FLASK_ENV = production
APIFOOTBALL_API_KEY = 908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
```

5. **"Create Web Service"** tıkla

### **Adım 4: Deployment İzle**

- Build logları otomatik açılacak
- **"Deploy"** işlemi 5-10 dakika sürer
- Başarılı olunca yeşil **"Live"** durumu göreceksin

---

## 🔗 **Canlı URL**

Deployment tamamlandıktan sonra:
```
https://football-prediction-app.onrender.com
```
*(URL kendi service adınıza göre değişir)*

---

## 📁 **Dosya Yapısı**

```
football-prediction-app/
├── main.py                 # Ana Flask uygulaması
├── run.py                  # Başlatma script'i
├── requirements.txt        # Python bağımlılıkları
├── render.yaml            # Render konfigürasyonu
├── render_start.sh        # Başlatma script'i (alternatif)
├── .env.example           # Environment variables örneği
├── static/                # CSS, JS, resim dosyaları
├── templates/             # HTML şablonları
└── README_RENDER.md       # Bu dosya
```

---

## 🐛 **Sorun Giderme**

### **❌ Build Hatası**
```bash
# Eğer build fail olursa:
1. Render dashboard → Logs sekmesi
2. Build logs'u incele
3. requirements.txt'deki versiyon uyumluluğunu kontrol et
```

### **❌ Application Error**
```bash
# Uygulama başlamazsa:
1. Environment Variables kontrol et
2. PORT variable'ının doğru set edildiğini kontrol et
3. Logs'ta Python error'larını ara
```

### **❌ Slow Loading**
```bash
# Render free tier bazen yavaş olabilir:
- İlk request 30 saniye sürebilir (cold start)
- Bu normal bir durum
- Premium plan ile çözülür
```

---

## 🔧 **Production Optimizasyonları**

### **Gunicorn ile Çalıştırma (Daha İyi Performance):**

`render.yaml` dosyasında:
```yaml
startCommand: gunicorn --bind 0.0.0.0:$PORT --workers 2 main:app
```

### **Auto-Deploy Ayarı:**
- GitHub'a her push'ta otomatik deploy olur
- `render.yaml` içinde `autoDeploy: true` ayarı aktif

### **Custom Domain (İsteğe Bağlı):**
1. Render dashboard → Settings → Custom Domains
2. Domain'inizi ekle
3. DNS ayarlarını güncelle

---

## 📊 **Monitoring**

### **Logs İzleme:**
```bash
# Render dashboard'da:
1. Service seç → Logs
2. Real-time logları izle
3. Error tracking
```

### **Performance:**
```bash
# Metrics sekmesinde:
- CPU kullanımı
- Memory kullanımı  
- Response time
- Request count
```

---

## 🆙 **Güncelleme**

```bash
# Kod değişikliği sonrası:
git add .
git commit -m "Update: new features"
git push origin main

# Render otomatik olarak yeni deploy başlatır
```

---

## 💡 **Pro Tips**

1. **Environment Variables** sensitive bilgiler için kullan
2. **Logs'u sürekli izle** deployment sırasında
3. **Free tier** sleep mode'a geçebilir (inactivity sonrası)
4. **Health check endpoint** ekle production için
5. **Database** gerekirse Render PostgreSQL kullan

---

## 🎯 **Başarılı Deployment Sonrası**

✅ **Şunları test et:**
- Ana sayfa yükleniyor mu?
- API endpoint'leri çalışıyor mu?
- Static dosyalar (CSS/JS) yükleniyor mu?
- Mobile responsive çalışıyor mu?

**🎉 Tebrikler! Uygulamanız canlıda!**

URL: `https://YOUR-APP-NAME.onrender.com`