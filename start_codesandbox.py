#!/usr/bin/env python3
"""
CodeSandbox Başlatma Scripti
Bu script CodeSandbox'ta uygulamayı güvenli bir şekilde başlatır.
"""

import os
import sys
import logging

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def install_dependencies():
    """Gerekli dependencies'i kur"""
    try:
        import subprocess
        logger.info("Dependencies kuruluyor...")
        
        # Basic dependencies
        basic_deps = [
            'flask==2.3.3',
            'gunicorn==21.2.0', 
            'requests==2.31.0',
            'flask-caching==2.1.0',
            'pytz==2023.3'
        ]
        
        for dep in basic_deps:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
                logger.info(f"✅ {dep} kuruldu")
            except Exception as e:
                logger.warning(f"⚠️ {dep} kurulamadı: {e}")
        
        logger.info("✅ Temel dependencies kuruldu")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dependency kurulumu başarısız: {e}")
        return False

def check_api_keys():
    """API key'leri kontrol et"""
    logger.info("API key'ler kontrol ediliyor...")
    
    # Environment variables kontrol
    api_key = os.environ.get('APIFOOTBALL_API_KEY')
    if api_key:
        logger.info(f"✅ APIFOOTBALL_API_KEY bulundu: {api_key[:20]}...")
    else:
        logger.warning("⚠️ APIFOOTBALL_API_KEY environment variable bulunamadı")
    
    return True

def start_application():
    """Uygulamayı başlat"""
    try:
        logger.info("🚀 Uygulama başlatılıyor...")
        
        # main.py'yi import et ve çalıştır
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        import main
        
        # Flask app'i al
        app = main.app
        
        # Port'u belirle
        port = int(os.environ.get('PORT', 5000))
        host = os.environ.get('HOST', '0.0.0.0')
        
        logger.info(f"✅ Uygulama başlatıldı: http://{host}:{port}")
        
        # Debug mode'da çalıştır
        app.run(host=host, port=port, debug=True)
        
    except Exception as e:
        logger.error(f"❌ Uygulama başlatma hatası: {e}")
        
        # Fallback: Basic Flask app
        logger.info("🔄 Fallback modu başlatılıyor...")
        start_fallback()

def start_fallback():
    """Fallback Flask uygulaması"""
    from flask import Flask, jsonify
    
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return jsonify({
            'status': 'success',
            'message': 'Football Prediction App - Fallback Mode',
            'api_key_status': 'Configured' if os.environ.get('APIFOOTBALL_API_KEY') else 'Missing'
        })
    
    @app.route('/health')
    def health():
        return jsonify({'status': 'healthy', 'mode': 'fallback'})
    
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"✅ Fallback uygulaması başlatıldı: http://{host}:{port}")
    app.run(host=host, port=port, debug=True)

if __name__ == '__main__':
    logger.info("🔄 CodeSandbox başlatma scripti çalışıyor...")
    
    # 1. Dependencies kontrol ve kurulum
    install_dependencies()
    
    # 2. API key kontrol
    check_api_keys()
    
    # 3. Uygulamayı başlat
    start_application()