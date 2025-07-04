# Futbol Tahmin Uygulaması - CodeSandbox Kurulumu

Bu uygulama, futbol maçları için gelişmiş tahminler yapan bir Flask web uygulamasıdır.

## CodeSandbox'ta Çalıştırma

### 1. Adım: Dependency'leri Yükle
**Minimal kurulum (CodeSandbox için önerilen):**
```bash
pip install -r requirements.txt
```

**Eğer disk alanınız varsa ve ML özelliklerini istiyorsanız:**
```bash
pip install -r requirements-full.txt
```

**Not**: CodeSandbox'ta disk alanı sınırlı olduğu için minimal kurulum önerilir.

### 2. Adım: Uygulamayı Başlat

**🚀 En Kolay Yol (Önerilen):**
```bash
bash start.sh
```

**⚡ Hızlı Başlatma Seçenekleri:**

1. **Gunicorn (CPU optimized):**
```bash
gunicorn -c gunicorn.conf.py main:app
```

2. **Python Development:**
```bash
python3 main.py
```

3. **Manuel Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 2 main:app
```

### 3. Adım: Uygulama Erişimi
Uygulama başlatıldıktan sonra, CodeSandbox preview penceresinde uygulamanıza erişebilirsiniz.

## Özellikler

- **Maç Tahminleri**: Gelişmiş makine öğrenmesi modelleri ile maç sonucu tahminleri
- **Canlı Maçlar**: Günlük maç programı ve canlı skorlar
- **İstatistikler**: Takım performans analizleri
- **API Endpoints**: REST API ile tahmin servisleri
- **Önbellek**: Hızlı erişim için tahmin önbelleği

## API Anahtarları

Uygulama, apifootball.com API'sini kullanmaktadır. API anahtarı kodda tanımlıdır, ancak production ortamında environment variable olarak ayarlanmalıdır:

```bash
export APIFOOTBALL_API_KEY="your_api_key_here"
```

## Temel Endpoint'ler

- `/` - Ana sayfa (günlük maçlar)
- `/api/predict-match/<home_id>/<away_id>` - Maç tahmini
- `/predictions` - Tüm tahminler
- `/leagues` - Lig puan durumları

## Dosya Yapısı

- `main.py` - Ana uygulama dosyası
- `match_prediction.py` - Tahmin algoritmaları
- `api_routes.py` - API endpoint'leri
- `templates/` - HTML şablonları
- `static/` - CSS, JS ve resim dosyaları

## 🔧 CPU Optimizasyonları

**CPU %97 Sorunu Çözüldü!** Aşağıdaki optimizasyonlar uygulandı:

### ✅ Yapılan İyileştirmeler:
- **Lazy Loading**: Servisler ihtiyaç anında yüklenir
- **Minimal Startup**: Gereksiz işlemler başlangıçta atlanır  
- **Resource Limits**: CPU ve memory kullanımı sınırlandırıldı
- **Smart Fallbacks**: ML kütüphaneleri yoksa basit algoritmalar kullanılır
- **Environment Detection**: CodeSandbox otomatik algılanır

### ⚡ Performans Ayarları:
```bash
# CPU kullanımını kontrol et
top -p $(pgrep -f "python\|gunicorn")

# Memory kullanımını kontrol et  
free -h
```

## 🛠️ Veritabanı & Caching

### Built-in Caching:
- **Flask-Caching**: Otomatik olarak aktif
- **Prediction Cache**: Tahminler 10 dakika önbelleğe alınır
- **Route Cache**: API yanıtları önbelleğe alınır

### SQLite Veritabanı:
```python
# Otomatik olarak oluşturulur:
- team_performance.db
- predictions_cache.json
```

## 🚨 Sorun Giderme

### CPU Yüksek Kullanım:
1. `bash start.sh` kullanın (optimize edilmiş)
2. Gunicorn ile başlatın: `gunicorn -c gunicorn.conf.py main:app`
3. Debug mode'u kapatın: `python3 main.py` yerine production mode

### Import Hataları:
1. `pip install -r requirements.txt --no-cache-dir` çalıştırın
2. Bireysel paket kurulumu: `pip install Flask requests pytz gunicorn`
3. Python 3.8+ kullandığınızdan emin olun

### Memory Hataları:
1. `bash start.sh` optimize edilmiş ayarlarla başlatır
2. Worker sayısını azaltın: `--workers 1`
3. Cache'i temizleyin: `/api/clear-cache` endpoint'ini çağırın

### Setup Failed (3/3):
1. `requirements.txt` dosyasının mevcut olduğunu kontrol edin
2. `bash start.sh` ile otomatik kurulum yapın
3. Manual kurulum: `pip install Flask==3.0.0 requests==2.31.0`