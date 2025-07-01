# 🚀 CodeSandbox - Basit Kurulum

## ⚡ En Basit Yöntem (Önerilen)

1. Tüm dosyaları CodeSandbox'a yükleyin
2. Terminal'de bu komutu çalıştırın:

```bash
python3 run.py
```

## 🔧 Manuel Kurulum

Eğer yukarıdaki çalışmazsa:

```bash
# Environment variables'ları yükle
export APIFOOTBALL_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
export API_FOOTBALL_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"

# Flask app'i direkt çalıştır
python3 main.py
```

## 🎯 Dosya Yapısı

Şu dosyaların CodeSandbox'ta olduğundan emin olun:
- ✅ `main.py` (ana uygulama)
- ✅ `run.py` (basit çalıştırma script'i)
- ✅ `.env` (API key'ler)
- ✅ `requirements.txt` (bağımlılıklar)

## 🌐 Erişim

Uygulama çalıştıktan sonra:
- Ana sayfa: Browser preview'da görünecek
- Port: 5000 (otomatik)

## 🔍 Sorun Giderme

### Hata: "Module not found"
```bash
pip install flask requests --user
python3 run.py
```

### Hata: "Permission denied"
```bash
chmod +x run.py start.sh
python3 run.py
```

### API Key kontrol
```bash
python3 -c "import os; print('API Key:', os.environ.get('APIFOOTBALL_API_KEY', 'NOT SET')[:20] + '...')"
```

## 📊 API Key Bilgisi

✅ **API Key**: `908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485`
✅ **Lokasyon**: `.env` dosyasında
✅ **Otomatik yüklenme**: `run.py` ile

## 🎮 CodeSandbox'ta Çalıştırma

1. **Upload all files** to CodeSandbox
2. **Open terminal**
3. **Run**: `python3 run.py`
4. **Open preview** in browser

That's it! 🎉