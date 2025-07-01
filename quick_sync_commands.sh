#!/bin/bash

# GitHub ve CodeSandbox API Key Senkronizasyon Komutları
# Bu dosyayı CodeSandbox terminal'inde çalıştırabilirsiniz

echo "🔄 GitHub ve CodeSandbox API Key Senkronizasyonu"
echo "================================================"

# Yeni API Key
NEW_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
OLD_API_KEY="aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df"

echo "🔍 1. Mevcut API key'leri kontrol ediliyor..."

# Eski key kontrolü
OLD_COUNT=$(grep -r "$OLD_API_KEY" . --include="*.py" | wc -l)
echo "   Eski API key bulundu: $OLD_COUNT dosyada"

# Yeni key kontrolü  
NEW_COUNT=$(grep -r "$NEW_API_KEY" . --include="*.py" | wc -l)
echo "   Yeni API key bulundu: $NEW_COUNT dosyada"

if [ $OLD_COUNT -gt 0 ]; then
    echo ""
    echo "⚠️ 2. ESKİ API KEY'LER BULUNDU - Değiştiriliyor..."
    
    # api_routes.py güncellemesi
    if [ -f "api_routes.py" ]; then
        sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" api_routes.py
        echo "   ✅ api_routes.py güncellendi"
    fi
    
    # main.py güncellemesi
    if [ -f "main.py" ]; then
        sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" main.py
        echo "   ✅ main.py güncellendi"
    fi
    
    # match_prediction.py güncellemesi
    if [ -f "match_prediction.py" ]; then
        sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" match_prediction.py
        echo "   ✅ match_prediction.py güncellendi"
    fi
    
    echo ""
    echo "🔄 3. Kontrol ediliyor..."
    
    # Son kontrol
    FINAL_OLD_COUNT=$(grep -r "$OLD_API_KEY" . --include="*.py" | wc -l)
    FINAL_NEW_COUNT=$(grep -r "$NEW_API_KEY" . --include="*.py" | wc -l)
    
    echo "   Son durum - Eski key: $FINAL_OLD_COUNT, Yeni key: $FINAL_NEW_COUNT"
    
    if [ $FINAL_OLD_COUNT -eq 0 ] && [ $FINAL_NEW_COUNT -gt 0 ]; then
        echo "   ✅ API key güncellemesi BAŞARILI!"
    else
        echo "   ❌ API key güncellemesi BAŞARISIZ!"
        echo "   Manuel kontrol gerekebilir."
    fi
    
else
    echo ""
    echo "✅ 2. API key'ler zaten güncel!"
fi

echo ""
echo "📁 3. Güncellenen dosyalar:"
echo "   - api_routes.py (satır 38)"
echo "   - main.py (satır 69, 287)"  
echo "   - match_prediction.py (satır 42)"

echo ""
echo "🎯 4. Test komutları:"
echo "   # Yeni key'i kontrol et:"
echo "   grep -n '$NEW_API_KEY' *.py"
echo ""
echo "   # Eski key kaldı mı kontrol et:"  
echo "   grep -n '$OLD_API_KEY' *.py"

echo ""
echo "🚀 5. CodeSandbox restart önerisi:"
echo "   Değişikliklerin etkili olması için sandbox'ı restart edin!"

echo ""
echo "✨ Senkronizasyon tamamlandı!"
echo "================================================"