#!/usr/bin/env python3
"""
Basit Flask Uygulaması - CodeSandbox Test
"""

from flask import Flask, jsonify, render_template, request
import os
import requests
import logging
from datetime import datetime

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# API Key
API_KEY = "908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485"

@app.route('/')
def home():
    """Ana sayfa"""
    return jsonify({
        'status': 'success',
        'message': 'Football Prediction App - Simple Mode',
        'timestamp': datetime.now().isoformat(),
        'api_key_status': 'Configured' if API_KEY else 'Missing',
        'endpoints': [
            '/',
            '/health',
            '/test-api',
            '/matches?date=2024-01-01'
        ]
    })

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'mode': 'simple',
        'dependencies': {
            'flask': True,
            'requests': True
        }
    })

@app.route('/test-api')
def test_api():
    """API bağlantı testi"""
    try:
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'APIkey': API_KEY,
            'from': '2024-01-01',
            'to': '2024-01-01',
            'timezone': 'Europe/Istanbul'
        }
        
        response = requests.get(url, params=params, timeout=10)
        
        return jsonify({
            'status': 'success',
            'api_status': response.status_code,
            'api_working': response.status_code == 200,
            'response_size': len(response.content) if response.content else 0
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'api_working': False
        })

@app.route('/matches')
def matches():
    """Basit maç listesi"""
    selected_date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))
    
    try:
        url = "https://apiv3.apifootball.com/"
        params = {
            'action': 'get_events',
            'APIkey': API_KEY,
            'from': selected_date,
            'to': selected_date,
            'timezone': 'Europe/Istanbul'
        }
        
        response = requests.get(url, params=params, timeout=15)
        
        if response.status_code == 200:
            matches = response.json()
            if isinstance(matches, list):
                simplified_matches = []
                for match in matches[:10]:  # İlk 10 maç
                    simplified_matches.append({
                        'home_team': match.get('match_hometeam_name', 'N/A'),
                        'away_team': match.get('match_awayteam_name', 'N/A'),
                        'league': match.get('league_name', 'N/A'),
                        'time': match.get('match_time', 'N/A'),
                        'status': match.get('match_status', 'N/A')
                    })
                
                return jsonify({
                    'status': 'success',
                    'date': selected_date,
                    'total_matches': len(matches),
                    'showing': len(simplified_matches),
                    'matches': simplified_matches
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'API returned non-list data',
                    'data': matches
                })
        else:
            return jsonify({
                'status': 'error',
                'message': f'API returned status {response.status_code}'
            })
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"🚀 Basit uygulama başlatılıyor: http://{host}:{port}")
    logger.info(f"API Key: {API_KEY[:20]}...")
    
    app.run(host=host, port=port, debug=True)