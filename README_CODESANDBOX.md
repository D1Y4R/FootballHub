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
Terminal'de aşağıdaki komutu çalıştırın:
```bash
python3 main.py
```

Alternatif olarak Gunicorn ile:
```bash
gunicorn --bind 0.0.0.0:5000 main:app
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

## Sorun Giderme

Eğer kütüphane import hataları alırsanız:
1. `pip install -r requirements.txt` komutunu tekrar çalıştırın
2. Python 3.8+ kullandığınızdan emin olun
3. CodeSandbox terminal'inde `python3 --version` ile Python versiyonunu kontrol edin