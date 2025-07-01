#!/usr/bin/env python3
"""
Ana main.py dosyasının basitleştirilmiş versiyonu
CodeSandbox için optimize edilmiştir
"""

from flask import Flask, jsonify, render_template, request
import os
import requests
import logging
from datetime import datetime
import pytz

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# API Key - YENİ KEY
API_KEY = "908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"

# Ana sayfa route
@app.route('/')
def index():
    """Ana sayfa - Günün maçları"""
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    try:
        matches_data = get_matches(selected_date)
        
        # Eğer template varsa kullan, yoksa JSON döndür
        try:
            return render_template('index.html', matches=matches_data, selected_date=selected_date)
        except:
            return jsonify({
                'status': 'success',
                'message': 'Template bulunamadı, JSON response döndürülüyor',
                'date': selected_date,
                'matches': matches_data
            })
            
    except Exception as e:
        logger.error(f"Ana sayfa hatası: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'date': selected_date
        })

def get_matches(selected_date=None):
    """API'den maçları çek"""
    try:
        if not selected_date:
            selected_date = datetime.now().strftime('%Y-%m-%d')

        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'APIkey': API_KEY,
            'from': selected_date,
            'to': selected_date,
            'timezone': 'Europe/Istanbul'
        }
        
        logger.info(f"API çağrısı yapılıyor: {selected_date}")
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                matches = []
                for match in data:
                    match_obj = {
                        'id': match.get('match_id', ''),
                        'competition': {
                            'name': match.get('league_name', '')
                        },
                        'homeTeam': {
                            'name': match.get('match_hometeam_name', '')
                        },
                        'awayTeam': {
                            'name': match.get('match_awayteam_name', '')
                        },
                        'turkish_time': match.get('match_time', ''),
                        'status': match.get('match_status', 'SCHEDULED'),
                        'score': {
                            'fullTime': {
                                'home': match.get('match_hometeam_score', 0),
                                'away': match.get('match_awayteam_score', 0)
                            }
                        }
                    }
                    matches.append(match_obj)
                
                return {'leagues': [{'name': 'Tüm Ligler', 'matches': matches}]}
            else:
                logger.error(f"API hata döndü: {data}")
                return {'leagues': []}
        else:
            logger.error(f"API HTTP hatası: {response.status_code}")
            return {'leagues': []}
            
    except Exception as e:
        logger.error(f"Maç alma hatası: {str(e)}")
        return {'leagues': []}

# Health check
@app.route('/health')
def health():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'healthy',
        'app': 'football-prediction',
        'version': 'simple',
        'api_key': 'configured' if API_KEY else 'missing'
    })

# API endpoint
@app.route('/api/matches')
def api_matches():
    """API endpoint - maçlar"""
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    matches_data = get_matches(selected_date)
    return jsonify(matches_data)

# Test endpoint
@app.route('/test')
def test():
    """Test endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Test başarılı!',
        'timestamp': datetime.now().isoformat(),
        'api_key_preview': API_KEY[:20] + "..." if API_KEY else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info("🚀 Football Prediction App (Simple Mode) başlatılıyor...")
    logger.info(f"Port: {port}")
    logger.info(f"API Key: {API_KEY[:20]}..." if API_KEY else "API Key yok!")
    
    app.run(host=host, port=port, debug=True)