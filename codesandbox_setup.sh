#!/bin/bash

echo "🚀 CodeSandbox Kurulum ve API Key Güncelleme"
echo "=============================================="

# Yeni API Key
NEW_API_KEY="908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"
OLD_API_KEY="aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df"

echo "📦 1. Dependencies kuruluyor..."
pip install flask gunicorn requests flask-caching pytz

echo "🔧 2. API key'ler güncelleniyor..."

# API key güncellemeleri
if [ -f "api_routes.py" ]; then
    sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" api_routes.py
    echo "   ✅ api_routes.py güncellendi"
fi

if [ -f "main.py" ]; then
    sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" main.py
    echo "   ✅ main.py güncellendi"
fi

if [ -f "match_prediction.py" ]; then
    sed -i "s/$OLD_API_KEY/$NEW_API_KEY/g" match_prediction.py
    echo "   ✅ match_prediction.py güncellendi"
fi

echo "🔍 3. Kontrol ediliyor..."

# Eski key kontrolü
OLD_COUNT=$(grep -r "$OLD_API_KEY" . --include="*.py" 2>/dev/null | wc -l)
NEW_COUNT=$(grep -r "$NEW_API_KEY" . --include="*.py" 2>/dev/null | wc -l)

echo "   Eski API key: $OLD_COUNT dosyada"
echo "   Yeni API key: $NEW_COUNT dosyada"

if [ $OLD_COUNT -eq 0 ] && [ $NEW_COUNT -gt 0 ]; then
    echo "   ✅ API key güncellemesi BAŞARILI!"
else
    echo "   ⚠️ Manuel kontrol gerekebilir"
fi

echo "🎯 4. Environment variable ayarlama (opsiyonel):"
echo "   export APIFOOTBALL_API_KEY=\"$NEW_API_KEY\""

echo "🚀 5. Uygulamayı başlatmak için:"
echo "   python start_codesandbox.py"
echo "   # veya"
echo "   python main.py"

echo ""
echo "✨ Kurulum tamamlandı!"
echo "=============================================="