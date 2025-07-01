# 🔄 GitHub ve CodeSandbox API Key Senkronizasyon Rehberi

## 📋 Mevcut Durum
- ✅ **Yerel Dosyalar**: Yeni API key ile güncellendi
- ✅ **Git Commit**: Değişiklikler commit edildi
- ❌ **GitHub**: Push edilmesi gerekiyor
- ❌ **CodeSandbox**: Senkronizasyon gerekiyor

---

## 🚀 **ADIM 1: GitHub'a Push İşlemi**

### Terminal'de çalıştır:
```bash
# 1. Mevcut branch'i kontrol et
git branch

# 2. Değişiklikleri GitHub'a push et
git push origin cursor/analyze-and-optimize-code-performance-0aba

# 3. Ana branch'e merge için (opsiyonel)
git checkout main
git merge cursor/analyze-and-optimize-code-performance-0aba
git push origin main
```

---

## 🔧 **ADIM 2: CodeSandbox Senkronizasyonu**

### Yöntem 1: Otomatik Senkronizasyon
1. **CodeSandbox'ı aç**: https://codesandbox.io
2. **GitHub'dan projeyi aç**
3. **Sync butonuna tıkla** (sağ üst köşe)
4. **Pull from GitHub** seçeneğini seç

### Yöntem 2: Manuel Güncelleme
1. **CodeSandbox'ta dosyaları aç**:
   - `api_routes.py`
   - `main.py` 
   - `match_prediction.py`

2. **API key'leri manuel olarak değiştir**:
   ```python
   # ESKİ KEY (Değiştirilecek)
   'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df'
   
   # YENİ KEY (Kullanılacak)
   '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485'
   ```

---

## 📁 **ADIM 3: Güncellenecek Dosyalar**

### 1. `api_routes.py` - Satır 38
```python
# ÖNCE
API_FOOTBALL_KEY = os.environ.get('APIFOOTBALL_API_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# SONRA  
API_FOOTBALL_KEY = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

### 2. `main.py` - Satır 69 ve 287
```python
# ÖNCE
api_key = os.environ.get('APIFOOTBALL_API_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# SONRA
api_key = os.environ.get('APIFOOTBALL_API_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

### 3. `match_prediction.py` - Satır 42
```python
# ÖNCE
self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')

# SONRA
self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
```

---

## 🔍 **ADIM 4: CodeSandbox'ta Doğrulama**

### Kontrol Edilecek Noktalar:
1. **Dosya İçerikleri**: Yeni API key'in doğru yerleştirildiğini kontrol et
2. **Konsol Hatalar**: CodeSandbox konsolu API hatalarını kontrol et  
3. **API Bağlantısı**: Maç verileri çekilip çekilmediğini test et

### Test Komutu (CodeSandbox Terminal):
```bash
# API key'i kontrol et
grep -r "908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485" .

# Eski key'in kalmadığını kontrol et  
grep -r "aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df" .
```

---

## 🎯 **ADIM 5: Environment Variables (Opsiyonel)**

### CodeSandbox'ta Environment Variables Ayarlama:
1. **Settings** → **Environment Variables**
2. **Yeni değişken ekle**:
   ```
   Name: APIFOOTBALL_API_KEY
   Value: 908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   
   Name: APIFOOTBALL_PREMIUM_KEY  
   Value: 908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
   ```

---

## ⚡ **Hızlı Senkronizasyon Adımları**

### Yerel'den GitHub'a:
```bash
git push origin cursor/analyze-and-optimize-code-performance-0aba
```

### CodeSandbox'ta:
1. **Sync** butonuna tıkla
2. **Pull from GitHub** seç
3. **Değişiklikleri onayla**
4. **Sandbox'ı restart et**

---

## 🚨 **Olası Sorunlar ve Çözümleri**

### Problem 1: CodeSandbox Senkronizasyon Hatası
**Çözüm**: Manuel dosya düzenleme yapın

### Problem 2: API Key Çalışmıyor
**Çözüm**: 
1. APIfootball.com'da key'in aktif olduğunu kontrol edin
2. Quota limitlerini kontrol edin
3. Network bağlantısını test edin

### Problem 3: Eski Key Hala Görünüyor  
**Çözüm**:
1. Browser cache'ini temizleyin
2. CodeSandbox'ı hard refresh yapın (Ctrl+F5)
3. Sandbox'ı restart edin

---

## ✅ **Senkronizasyon Kontrol Listesi**

- [ ] GitHub'a push yapıldı
- [ ] CodeSandbox sync edildi  
- [ ] `api_routes.py` güncellendi
- [ ] `main.py` güncellendi
- [ ] `match_prediction.py` güncellendi
- [ ] API bağlantısı test edildi
- [ ] Konsol hataları kontrol edildi
- [ ] Environment variables ayarlandı (opsiyonel)

---

## 📞 **Destek Bilgileri**

### GitHub Repository URL:
```
https://github.com/[username]/[repository-name]
```

### CodeSandbox URL:
```
https://codesandbox.io/s/github/[username]/[repository-name]
```

### Yeni API Key:
```
908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485
```

---

## 🎉 **Senkronizasyon Tamamlandığında**

1. **Restart** CodeSandbox'ı
2. **Test** API bağlantılarını
3. **Verify** maç verilerinin çekildiğini
4. **Monitor** konsol loglarını

**Başarılı senkronizasyon sonrası her iki platformda da yeni API key aktif olacaktır!** 🚀