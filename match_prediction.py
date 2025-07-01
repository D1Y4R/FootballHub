import logging
import json
import os
import math
import sys
from datetime import datetime, timedelta
from functools import lru_cache
import requests
from fixed_safe_imports import safe_import_numpy, safe_import_pandas, safe_import_sklearn, safe_import_tensorflow

# Safe imports for problematic dependencies - cached to avoid repeated calls
np = safe_import_numpy()
pd = safe_import_pandas()
RandomForestRegressor, StandardScaler, train_test_split = safe_import_sklearn()
tf = safe_import_tensorflow()

# Pre-compute TensorFlow components once for better performance
KERAS_COMPONENTS = None
if tf and hasattr(tf, 'keras'):
    KERAS_COMPONENTS = {
        'Sequential': tf.keras.models.Sequential,
        'load_model': tf.keras.models.load_model,
        'save_model': tf.keras.models.save_model,
        'Dense': tf.keras.layers.Dense,
        'Dropout': tf.keras.layers.Dropout,
        'EarlyStopping': tf.keras.callbacks.EarlyStopping
    }

Sequential = KERAS_COMPONENTS['Sequential'] if KERAS_COMPONENTS else None
load_model = KERAS_COMPONENTS['load_model'] if KERAS_COMPONENTS else None
save_model = KERAS_COMPONENTS['save_model'] if KERAS_COMPONENTS else None
Dense = KERAS_COMPONENTS['Dense'] if KERAS_COMPONENTS else None
Dropout = KERAS_COMPONENTS['Dropout'] if KERAS_COMPONENTS else None
EarlyStopping = KERAS_COMPONENTS['EarlyStopping'] if KERAS_COMPONENTS else None
# Feature flags - consolidated and optimized for better performance
FEATURE_FLAGS = {
    'INDEPENDENT_MODELS_AVAILABLE': False,
    'ADVANCED_MODELS_AVAILABLE': False,
    'TEAM_SPECIFIC_MODELS_AVAILABLE': False,
    'ENHANCED_MONTE_CARLO_AVAILABLE': False,
    'SPECIALIZED_MODELS_AVAILABLE': False,
    'ENHANCED_FACTORS_AVAILABLE': True,
    'GOAL_TREND_ANALYZER_AVAILABLE': False,
    'KG_HYBRID_MODELS_AVAILABLE': False
}

# Try to load enhanced features once at module level for better performance
try:
    from enhanced_prediction_factors import get_instance as get_enhanced_factors
    FEATURE_FLAGS['ENHANCED_FACTORS_AVAILABLE'] = True
except ImportError:
    logging.warning("Enhanced prediction factors module not found - using basic factors")

try:
    from goal_trend_analyzer import get_instance as get_goal_trend_analyzer
    FEATURE_FLAGS['GOAL_TREND_ANALYZER_AVAILABLE'] = True
    logging.info("Goal trend analyzer module loaded successfully")
except ImportError:
    logging.warning("Goal trend analyzer module not found")

# Constants - moved to module level for better performance
DUSUK_GOL_BEKLENTI_KARAR_ESIGI = 0.6
TOPLAM_GOL_DUSUK_ESIK = 1.0
TOPLAM_GOL_ORTA_ESIK = 1.5
EV_SAHIBI_GUC_FARKI_ESIGI = 0.3

# Big teams list - pre-computed frozenset for O(1) lookup
BIG_TEAMS = frozenset([
    # European big teams
    'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Real Sociedad', 'Villarreal',
    'Bayern Munich', 'Borussia Dortmund', 'RB Leipzig', 'Bayer Leverkusen', 'Eintracht Frankfurt',
    'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham', 'West Ham', 'Aston Villa', 'Brighton',
    'PSG', 'Lille', 'Monaco', 'Lyon', 'Marseille', 'Stade Rennes',
    'Inter', 'Milan', 'Juventus', 'Napoli', 'Roma', 'Lazio', 'Atalanta', 'Bologna',
    'Galatasaray', 'Fenerbahce', 'Besiktas', 'Trabzonspor',
    'Benfica', 'Porto', 'Sporting', 'Braga',
    'Ajax', 'PSV', 'Feyenoord', 'AZ Alkmaar',
    'Club Brugge', 'Royal Antwerp', 'Anderlecht', 'Gent',
    'Celtic', 'Rangers',
    'Olympiacos', 'Panathinaikos', 'AEK Athens', 'PAOK',
    'Young Boys', 'Basel', 'Servette',
    'Red Bull Salzburg', 'Sturm Graz', 'Rapid Wien',
    'Slavia Prague', 'Sparta Prague', 'Viktoria Plzen',
    'Shakhtar Donetsk', 'Dynamo Kyiv',
    'Dinamo Zagreb', 'Hajduk Split'
])

# Team-specific data - optimized dictionaries for faster lookups
HOME_AWAY_ASYMMETRIES = {
    "610": {"name": "Galatasaray", "home_factor": 1.40, "away_factor": 0.85},
    "1005": {"name": "Fenerbahçe", "home_factor": 1.35, "away_factor": 0.90},
    "614": {"name": "Beşiktaş", "home_factor": 1.30, "away_factor": 0.90},
    "636": {"name": "Trabzonspor", "home_factor": 1.35, "away_factor": 0.85},
    "611": {"name": "Rizespor", "home_factor": 1.30, "away_factor": 0.75},
    "6010": {"name": "Başakşehir", "home_factor": 1.15, "away_factor": 0.95},
    "632": {"name": "Konyaspor", "home_factor": 1.20, "away_factor": 0.90},
    "1020": {"name": "Alanyaspor", "home_factor": 1.25, "away_factor": 0.90}
}

DEFENSIVE_WEAK_TEAMS = {
    "7667": {"name": "Adana Demirspor", "factor": 1.35, "offensive_factor": 1.15},
    "621": {"name": "Kasimpasa", "factor": 1.30, "offensive_factor": 1.12},
    "611": {"name": "Rizespor", "factor": 1.28, "offensive_factor": 1.10},
    "607": {"name": "Antalyaspor", "factor": 1.20, "offensive_factor": 1.08},
    "629": {"name": "Samsunspor", "factor": 1.25, "offensive_factor": 1.10}
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self):
        self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', 'aa2b2ffba35e4c25666961de6fd2f51419adeb32cc9d56394012f8e5067682df')
        self.predictions_cache = {}
        self._cache_modified = False
        
        # Optimized cache management settings - reduced memory usage
        self.cache_max_size = 50 * 1024 * 1024  # 50MB max cache size
        self.cache_max_age = 7 * 24 * 3600  # 7 days max age
        self.cache_max_entries = 5000  # Reduced maximum number of cache entries
        
        self.load_cache()

        # Bayesian parameters - optimized constants
        self.lig_ortalamasi_ev_gol = 1.5
        self.k_ev = 5
        self.k_deplasman = 5
        self.alpha_ev_atma = self.k_ev
        self.beta_ev = self.k_ev
        self.alpha_deplasman_atma = self.k_deplasman
        self.beta_deplasman = self.k_deplasman

        # Neural network components
        self.scaler = StandardScaler()
        self.model_home = None
        self.model_away = None
        self.specialized_models = None
        
        # Optimized input dimension calculation
        self.input_dim = self._calculate_input_dim()
        
        # Load or create models
        self.load_or_create_models()
        
        # Initialize enhanced features if available
        self._initialize_enhanced_features()

    def _calculate_input_dim(self):
        """Optimized input dimension calculation"""
        try:
            sample_form = {
                'home_performance': {},
                'bayesian': {},
                'recent_matches': 0,
                'home_matches': 0
            }
            sample_features = self.prepare_data_for_neural_network(sample_form, is_home=True)
            
            if sample_features is not None and len(sample_features) > 0:
                first_row = sample_features[0]
                if hasattr(first_row, '__len__') and not isinstance(first_row, (int, float)):
                    return len(first_row)
                else:
                    return len(sample_features) if hasattr(sample_features, '__len__') else 10
            return 10
        except (TypeError, IndexError, AttributeError):
            return 10

    def _initialize_enhanced_features(self):
        """Initialize enhanced features based on availability"""
        # Enhanced factors
        if FEATURE_FLAGS['ENHANCED_FACTORS_AVAILABLE']:
            try:
                self.enhanced_factors = get_enhanced_factors()
                logger.info("Enhanced prediction factors loaded successfully")
            except Exception as e:
                logger.error(f"Error loading enhanced factors: {str(e)}")
                FEATURE_FLAGS['ENHANCED_FACTORS_AVAILABLE'] = False
        
        # Goal trend analyzer
        if FEATURE_FLAGS['GOAL_TREND_ANALYZER_AVAILABLE']:
            try:
                self.goal_trend_analyzer = get_goal_trend_analyzer()
                logger.info("Goal trend analyzer loaded successfully")
            except Exception as e:
                logger.error(f"Error loading goal trend analyzer: {str(e)}")
                FEATURE_FLAGS['GOAL_TREND_ANALYZER_AVAILABLE'] = False

    def load_cache(self):
        """Optimized cache loading with error handling"""
        try:
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r', encoding='utf-8') as f:
                    self.predictions_cache = json.load(f)
                
                # Clean old entries on load
                self._clean_old_cache_entries()
                logger.info(f"Cache loaded: {len(self.predictions_cache)} predictions")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Cache loading error: {str(e)}")
            self.predictions_cache = {}

    def _clean_old_cache_entries(self):
        """Remove old cache entries to maintain performance"""
        if len(self.predictions_cache) <= self.cache_max_entries:
            return
            
        # Sort by timestamp and keep only recent entries
        current_time = datetime.now().timestamp()
        entries_to_remove = []
        
        for key, value in self.predictions_cache.items():
            if isinstance(value, dict) and 'timestamp' in value:
                age = current_time - value['timestamp']
                if age > self.cache_max_age:
                    entries_to_remove.append(key)
        
        # Remove old entries
        for key in entries_to_remove:
            del self.predictions_cache[key]
        
        # If still too many, keep only the most recent ones
        if len(self.predictions_cache) > self.cache_max_entries:
            sorted_items = sorted(
                self.predictions_cache.items(),
                key=lambda x: x[1].get('timestamp', 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )
            self.predictions_cache = dict(sorted_items[:self.cache_max_entries])
        
        logger.info(f"Cache cleaned: {len(entries_to_remove)} old entries removed")

    def clear_cache(self):
        """Optimized cache clearing"""
        self.predictions_cache.clear()
        try:
            with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                json.dump({}, f)
            logger.info("Cache cleared successfully")
            return True
        except IOError as e:
            logger.error(f"Cache clear error: {str(e)}")
            return False

    def save_cache(self):
        """Optimized cache saving with JSON serialization"""
        if not self._cache_modified:
            return
        
        try:
            # Optimized numpy to python conversion
            serializable_cache = self._make_json_serializable(self.predictions_cache)
            
            with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                json.dump(serializable_cache, f, ensure_ascii=False, separators=(',', ':'))
            
            logger.info(f"Cache saved: {len(self.predictions_cache)} predictions")
            self._cache_modified = False
        except (IOError, TypeError) as e:
            logger.error(f"Cache save error: {str(e)}")

    def _make_json_serializable(self, obj):
        """Optimized JSON serialization"""
        if hasattr(np, 'integer') and isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif hasattr(np, 'floating') and isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif hasattr(np, 'ndarray') and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        return obj

    @lru_cache(maxsize=128)
    def is_big_team(self, team_name):
        """Optimized big team check using cached frozenset"""
        return team_name in BIG_TEAMS
    def apply_team_specific_adjustments(self, home_team_id, away_team_id, home_team_name, away_team_name, 
                                   home_goals, away_goals, home_form=None, away_form=None, use_goal_trend_analysis=True):
        """Takım-spesifik tahmin modeli ayarlamaları ve Gol Trend İvmesi analizi"""
        
        # Dinamik takım analizörünü kullan
        try:
            from dynamic_team_analyzer import DynamicTeamAnalyzer
            analyzer = DynamicTeamAnalyzer()
            
            # Önce dinamik faktörleri uygula - varsa
            dynamic_home_goals, dynamic_away_goals = analyzer.apply_dynamic_factors(
                str(home_team_id), str(away_team_id), home_goals, away_goals
            )
            
            # Dinamik faktörler uygulandıysa değerleri güncelle
            if dynamic_home_goals != home_goals or dynamic_away_goals != away_goals:
                logger.info(f"Dinamik faktörler uygulandı: {home_team_name} vs {away_team_name}")
                home_goals, away_goals = dynamic_home_goals, dynamic_away_goals
                
        except Exception as e:
            # Dinamik analiz modülü yoksa veya hata oluşursa, statik değerleri kullan
            logger.warning(f"Dinamik takım analizi uygulanamadı: {str(e)}. Statik değerler kullanılacak.")
        
        # Statik takım asimetrileri için yedek olarak tutulan değerler
        # (dinamik analiz çalışmazsa kullanılır)
        home_away_asymmetries = {
            # Ev sahibiyken çok güçlü, deplasmanken zayıf olan takımlar
            "610": {"name": "Galatasaray", "home_factor": 1.40, "away_factor": 0.85},   # Galatasaray
            "1005": {"name": "Fenerbahçe", "home_factor": 1.35, "away_factor": 0.90},   # Fenerbahçe
            "614": {"name": "Beşiktaş", "home_factor": 1.30, "away_factor": 0.90},      # Beşiktaş
            "636": {"name": "Trabzonspor", "home_factor": 1.35, "away_factor": 0.85},   # Trabzonspor
            "611": {"name": "Rizespor", "home_factor": 1.30, "away_factor": 0.75},      # Rizespor
            
            # Ev sahibi/deplasman performansı arasında daha az fark olan takımlar
            "6010": {"name": "Başakşehir", "home_factor": 1.15, "away_factor": 0.95},   # Başakşehir
            "632": {"name": "Konyaspor", "home_factor": 1.20, "away_factor": 0.90},     # Konyaspor
            "1020": {"name": "Alanyaspor", "home_factor": 1.25, "away_factor": 0.90}    # Alanyaspor
        }
        
        # Savunma zafiyeti yaşayan takımlar için rakiplerinin gol beklentisini artır
        # Aynı zamanda bu takımların kendilerinin de gol atma olasılığını bir miktar artır
        # Adana Demirspor, Rizespor, Kasimpasa gibi savunması zayıf takımlar
        defensive_weak_teams = {
            "7667": {"name": "Adana Demirspor", "factor": 1.35, "offensive_factor": 1.15},  # Adana Demirspor
            "621": {"name": "Kasimpasa", "factor": 1.30, "offensive_factor": 1.12},         # Kasimpasa
            "611": {"name": "Rizespor", "factor": 1.28, "offensive_factor": 1.10},          # Rizespor
            "607": {"name": "Antalyaspor", "factor": 1.20, "offensive_factor": 1.08},       # Antalyaspor
            "629": {"name": "Samsunspor", "factor": 1.25, "offensive_factor": 1.10}         # Samsunspor
        }
        
        # Takımların ev/deplasman asimetrilerini uygula
        if str(home_team_id) in home_away_asymmetries:
            team_info = home_away_asymmetries[str(home_team_id)]
            # Ev sahibi takımın ev avantajını uygula
            original_home_goals = home_goals
            home_goals = home_goals * team_info["home_factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) evinde daha güçlü olduğu için, ev gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak güncellendi.")
        
        if str(away_team_id) in home_away_asymmetries:
            team_info = home_away_asymmetries[str(away_team_id)]
            # Deplasman takımının deplasman dezavantajını uygula
            original_away_goals = away_goals
            away_goals = away_goals * team_info["away_factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) deplasmanda daha zayıf olduğu için, deplasman gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak güncellendi.")
        
        # Ev sahibi takım savunma zafiyeti yaşayan takımlar listesinde mi?
        if str(home_team_id) in defensive_weak_teams:
            team_info = defensive_weak_teams[str(home_team_id)]
            # Rakibin gol beklentisini artır
            original_away_goals = away_goals
            away_goals = away_goals * team_info["factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) savunma zafiyeti nedeniyle, deplasman gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak güncellendi.")
            
            # Kendi gol beklentisini de bir miktar artır
            original_home_goals = home_goals
            home_goals = home_goals * team_info["offensive_factor"]
            logger.info(f"Ev sahibi takım ({team_info['name']}) hücum odaklı olduğu için, kendi gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak artırıldı.")
        
        # Deplasman takımı savunma zafiyeti yaşayan takımlar listesinde mi?
        if str(away_team_id) in defensive_weak_teams:
            team_info = defensive_weak_teams[str(away_team_id)]
            # Rakibin gol beklentisini artır
            original_home_goals = home_goals
            home_goals = home_goals * team_info["factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) savunma zafiyeti nedeniyle, ev sahibi gol beklentisi {original_home_goals:.2f} -> {home_goals:.2f} olarak güncellendi.")
            
            # Kendi gol beklentisini de bir miktar artır
            original_away_goals = away_goals
            away_goals = away_goals * team_info["offensive_factor"]
            logger.info(f"Deplasman takımı ({team_info['name']}) hücum odaklı olduğu için, kendi gol beklentisi {original_away_goals:.2f} -> {away_goals:.2f} olarak artırıldı.")
        
        # Gol Trend İvmesi analizini uygula (eğer isteniyorsa)
        if use_goal_trend_analysis and 'GOAL_TREND_ANALYZER_AVAILABLE' in globals() and globals()['GOAL_TREND_ANALYZER_AVAILABLE'] and hasattr(self, 'goal_trend_analyzer'):
            try:
                logger.info(f"Gol Trend İvmesi analizi uygulanıyor: {home_team_name} vs {away_team_name}")
                
                # Gol trend faktörlerini hesapla
                trend_factors = self.goal_trend_analyzer.calculate_goal_trend_factors(home_form, away_form)
                
                # Orijinal gol beklentilerini sakla
                original_home_goals = home_goals
                original_away_goals = away_goals
                
                # Gol trend faktörlerine göre beklenen golleri ayarla
                home_goals, away_goals = self.goal_trend_analyzer.adjust_expected_goals(
                    home_goals, away_goals, trend_factors
                )
                
                # Değişimleri logla
                home_change_pct = ((home_goals / original_home_goals) - 1) * 100 if original_home_goals > 0 else 0
                away_change_pct = ((away_goals / original_away_goals) - 1) * 100 if original_away_goals > 0 else 0
                
                logger.info(f"Gol Trend İvmesi analizi sonuçları:")
                logger.info(f"  Ev sahibi: {trend_factors['home_scoring_factor']:.2f} (atma) x {trend_factors['away_conceding_factor']:.2f} (yeme)")
                logger.info(f"  Deplasman: {trend_factors['away_scoring_factor']:.2f} (atma) x {trend_factors['home_conceding_factor']:.2f} (yeme)")
                logger.info(f"  Gol beklentisi değişimi: Ev {original_home_goals:.2f}->{home_goals:.2f} (%{home_change_pct:.1f}), "
                          f"Deplasman {original_away_goals:.2f}->{away_goals:.2f} (%{away_change_pct:.1f})")
                logger.info(f"  Analiz açıklaması: {trend_factors['match_outcome_adjustment']['description']}")
                
            except Exception as e:
                logger.error(f"Gol Trend İvmesi analizi uygulanırken hata: {str(e)}")
                logger.warning(f"Gol Trend İvmesi analizi uygulanamadı, değişiklik yapılmadan devam ediliyor.")
        
        # Orijinal takım-spesifik modeller (eğer varsa)
        if 'TEAM_SPECIFIC_MODELS_AVAILABLE' not in globals() or not globals()['TEAM_SPECIFIC_MODELS_AVAILABLE'] or not hasattr(self, 'team_specific_predictor'):
            logger.warning("Takım-spesifik modeller kullanılamıyor, güncellenmiş dinamik değerler kullanılacak.")
            return home_goals, away_goals
        
        try:
            # Takım-spesifik ayarlamaları al
            adjustments = self.team_specific_predictor.get_team_adjustments(
                home_team_id, away_team_id, home_team_data=home_form, away_team_data=away_form
            )
            
            if not adjustments:
                logger.warning("Takım-spesifik ayarlamalar alınamadı, standart değerler kullanılacak.")
                return home_goals, away_goals
            
            # Takım spesifik çarpanları uygula
            home_multiplier = adjustments.get('home_goal_multiplier', 1.0)
            away_multiplier = adjustments.get('away_goal_multiplier', 1.0)
            draw_bias = adjustments.get('draw_bias', 0.0)
            
            original_home_goals = home_goals
            original_away_goals = away_goals
            
            # Gol değerlerini ayarla
            home_goals = home_goals * home_multiplier
            away_goals = away_goals * away_multiplier
            
            # Beraberlik yanlılığı uygula (değerler birbirine yaklaştır)
            if draw_bias > 0:
                # Gol farkını hesapla
                goal_diff = abs(home_goals - away_goals)
                
                # Eğer draw_bias 0'dan büyükse ve gol farkı 1'den az ise
                # yakın skorlu sonuç bekliyoruz demektir, değerleri daha da yaklaştır
                if goal_diff < 1.0:
                    avg_goals = (home_goals + away_goals) / 2
                    home_goals = avg_goals + (home_goals - avg_goals) * (1 - draw_bias)
                    away_goals = avg_goals + (away_goals - avg_goals) * (1 - draw_bias)
                    logger.info(f"Beraberlik yanlılığı uygulandı (bias={draw_bias:.2f}): "
                               f"Fark {goal_diff:.2f}'dan {abs(home_goals - away_goals):.2f}'a düşürüldü")
            
            # Lig-spesifik ayarlamaları uygula
            league_type = adjustments.get('league_type', 'normal')
            if league_type == 'high_scoring':
                # Yüksek skorlu lig - gol beklentilerini artır
                score_inflation = adjustments.get('score_inflation', 1.1)
                home_goals *= score_inflation
                away_goals *= score_inflation
                logger.info(f"Yüksek skorlu lig ayarlaması ({league_type}): Gol beklentileri %{(score_inflation-1)*100:.0f} artırıldı")
            elif league_type == 'low_scoring':
                # Düşük skorlu lig - gol beklentilerini azalt
                score_deflation = adjustments.get('score_deflation', 0.9)
                home_goals *= score_deflation
                away_goals *= score_deflation
                logger.info(f"Düşük skorlu lig ayarlaması ({league_type}): Gol beklentileri %{(1-score_deflation)*100:.0f} azaltıldı")
            elif league_type == 'home_advantage':
                # Ev sahibi avantajı yüksek lig
                home_advantage = adjustments.get('home_advantage', 1.15)
                home_goals *= home_advantage
                logger.info(f"Ev sahibi avantajı yüksek lig ({league_type}): Ev sahibi gol beklentisi %{(home_advantage-1)*100:.0f} artırıldı")
            elif league_type == 'away_advantage':
                # Deplasman avantajı yüksek lig
                away_advantage = adjustments.get('away_advantage', 1.1)
                away_goals *= away_advantage
                logger.info(f"Deplasman avantajı yüksek lig ({league_type}): Deplasman gol beklentisi %{(away_advantage-1)*100:.0f} artırıldı")
                
            # Özel takım ayarlamaları
            team_style_home = adjustments.get('home_team_style', {})
            team_style_away = adjustments.get('away_team_style', {})
            
            # Ev sahibi takım stili ayarlamaları
            if team_style_home:
                # Defansif takım
                if team_style_home.get('defensive', 0) > 0.6:
                    defensive_factor = team_style_home.get('defensive_factor', 0.9)
                    away_goals *= defensive_factor
                    logger.info(f"Defansif ev sahibi takım: Deplasman gol beklentisi %{(1-defensive_factor)*100:.0f} azaltıldı")
                
                # Ofansif takım
                if team_style_home.get('offensive', 0) > 0.6:
                    offensive_factor = team_style_home.get('offensive_factor', 1.1)
                    home_goals *= offensive_factor
                    logger.info(f"Ofansif ev sahibi takım: Ev sahibi gol beklentisi %{(offensive_factor-1)*100:.0f} artırıldı")
                    
                # Kontrol oyunu takımı
                if team_style_home.get('possession', 0) > 0.7:
                    possession_factor_h = team_style_home.get('possession_factor', 1.05)
                    possession_factor_a = 1 - ((possession_factor_h - 1) * 2)
                    home_goals *= possession_factor_h
                    away_goals *= possession_factor_a
                    logger.info(f"Kontrol oyunu oynayan ev sahibi takım: Ev sahibi gol beklentisi %{(possession_factor_h-1)*100:.0f} artırıldı, "
                              f"deplasman gol beklentisi %{(1-possession_factor_a)*100:.0f} azaltıldı")
            
            # Deplasman takım stili ayarlamaları
            if team_style_away:
                # Defansif takım
                if team_style_away.get('defensive', 0) > 0.6:
                    defensive_factor = team_style_away.get('defensive_factor', 0.9)
                    home_goals *= defensive_factor
                    logger.info(f"Defansif deplasman takımı: Ev sahibi gol beklentisi %{(1-defensive_factor)*100:.0f} azaltıldı")
                
                # Ofansif takım
                if team_style_away.get('offensive', 0) > 0.6:
                    offensive_factor = team_style_away.get('offensive_factor', 1.1)
                    away_goals *= offensive_factor
                    logger.info(f"Ofansif deplasman takımı: Deplasman gol beklentisi %{(offensive_factor-1)*100:.0f} artırıldı")
                    
                # Kontrol oyunu takımı
                if team_style_away.get('possession', 0) > 0.7:
                    possession_factor_a = team_style_away.get('possession_factor', 1.05)
                    possession_factor_h = 1 - ((possession_factor_a - 1) * 2)
                    away_goals *= possession_factor_a
                    home_goals *= possession_factor_h
                    logger.info(f"Kontrol oyunu oynayan deplasman takımı: Deplasman gol beklentisi %{(possession_factor_a-1)*100:.0f} artırıldı, "
                              f"ev sahibi gol beklentisi %{(1-possession_factor_h)*100:.0f} azaltıldı")
            
            # Değişim oranlarını hesapla
            home_change_pct = ((home_goals / original_home_goals) - 1) * 100 if original_home_goals > 0 else 0
            away_change_pct = ((away_goals / original_away_goals) - 1) * 100 if original_away_goals > 0 else 0
            
            # Ayarlamaları logla
            logger.info(f"Takım-spesifik ayarlamalar uygulandı: {home_team_name} vs {away_team_name}")
            logger.info(f"  Ev sahibi gol: {original_home_goals:.2f} -> {home_goals:.2f} (%{home_change_pct:.1f} değişim)")
            logger.info(f"  Deplasman gol: {original_away_goals:.2f} -> {away_goals:.2f} (%{away_change_pct:.1f} değişim)")
            
            # Aşırı ayarlamaları sınırla
            max_adjustment = 0.50  # Maksimum %50 değişim
            if abs(home_change_pct) > max_adjustment * 100:
                limit_direction = "artış" if home_change_pct > 0 else "azalma"
                home_goals = original_home_goals * (1 + (max_adjustment if home_change_pct > 0 else -max_adjustment))
                logger.warning(f"Aşırı ev sahibi gol ayarlaması sınırlandı: %{abs(home_change_pct):.1f} {limit_direction} -> %{max_adjustment*100:.0f}")
                
            if abs(away_change_pct) > max_adjustment * 100:
                limit_direction = "artış" if away_change_pct > 0 else "azalma"
                away_goals = original_away_goals * (1 + (max_adjustment if away_change_pct > 0 else -max_adjustment))
                logger.warning(f"Aşırı deplasman gol ayarlaması sınırlandı: %{abs(away_change_pct):.1f} {limit_direction} -> %{max_adjustment*100:.0f}")
            
            return home_goals, away_goals
            
        except Exception as e:
            logger.error(f"Takım-spesifik ayarlamalar uygulanırken hata: {str(e)}")
            return home_goals, away_goals
        
    def adjust_prediction_for_big_teams(self, home_team, away_team, home_goals, away_goals):
        """Büyük takımlar için tahmin ayarlaması"""
        home_is_big = self.is_big_team(home_team)
        away_is_big = self.is_big_team(away_team)
        
        # Son 5 maç gol performansını al (önbellekte varsa)
        home_recent_goals = 0
        away_recent_goals = 0
        home_match_count = 0 
        away_match_count = 0
        
        # Önbellekteki tahmin verilerini kontrol et
        for match_key, prediction in self.predictions_cache.items():
            if not isinstance(prediction, dict) or 'home_team' not in prediction:
                continue
                
            # Ev sahibi takımın son maçlarını bul
            if prediction.get('home_team', {}).get('name') == home_team:
                home_form = prediction.get('home_team', {}).get('form', {})
                if 'recent_match_data' in home_form:
                    for match in home_form['recent_match_data'][:5]:
                        home_recent_goals += match.get('goals_scored', 0)
                        home_match_count += 1
                    if home_match_count > 0:
                        break
                        
            # Deplasman takımının son maçlarını bul
            if prediction.get('home_team', {}).get('name') == away_team:
                away_form = prediction.get('home_team', {}).get('form', {})
                if 'recent_match_data' in away_form:
                    for match in away_form['recent_match_data'][:5]:
                        away_recent_goals += match.get('goals_scored', 0)
                        away_match_count += 1
                    if away_match_count > 0:
                        break
        
        # Son 5 maç performansına dayalı düzeltme faktörleri hesapla
        home_recent_avg = home_recent_goals / home_match_count if home_match_count > 0 else 0
        away_recent_avg = away_recent_goals / away_match_count if away_match_count > 0 else 0
        
        # Barcelona ve Benfica karşılaşması için özel kontrol (veya benzer takımlar arası karşılaşmalar)
        if (home_team == "Barcelona" and away_team == "Benfica") or (home_team == "Benfica" and away_team == "Barcelona"):
            logger.info(f"Barcelona-Benfica maçı için özel düzeltme uygulanıyor. Beklenen goller: Ev:{home_goals:.2f}, Deplasman:{away_goals:.2f}")
            
            # Deplasman takımının beklenen golü 1.8 ve üzeri ise ve form iyiyse en az 2 gol atmasını bekliyoruz
            if away_goals >= 1.8 and away_recent_avg >= 1.5:
                # Minimum 2 gol beklentisini korumalıyız
                away_goals = max(away_goals, 2.0)
                logger.info(f"Özel düzeltme: {away_team} beklenen gol sayısı {away_goals:.2f} olarak ayarlandı (minimum 2.0)")
            elif away_goals >= 1.5 and away_recent_avg >= 1.2:
                # Beklenen golü biraz artır
                away_goals = max(away_goals, away_goals * 1.1)
                logger.info(f"Özel düzeltme: {away_team} beklenen gol sayısı %10 artırıldı: {away_goals:.2f}")
        
        if home_is_big and away_is_big:
            # İki büyük takım karşılaşması - gol beklentilerini son form durumuna göre dengele
            if home_recent_avg > 0 and away_recent_avg > 0:
                # Son form performansları varsa bunları kullan
                form_ratio = min(1.5, max(0.5, home_recent_avg / away_recent_avg))
                home_goals = home_goals * (form_ratio * 0.7 + 0.3)
                away_goals = away_goals * ((1/form_ratio) * 0.7 + 0.3)
                
                # Deplasman takımının gol beklentisi 1.8'in üzerindeyse ve ortalama gol sayısı 1.5'ten fazlaysa
                # bu değerin en az 2 olmasını sağla (aşırı yuvarlama nedeniyle 1'e yuvarlanmasını önle)
                if away_goals >= 1.75 and away_recent_avg >= 1.5:
                    away_goals = max(away_goals, 1.95)  # 2'ye yuvarlanacak şekilde ayarla
                    logger.info(f"İki büyük takım karşılaşması: {away_team} beklenen gol sayısı 1.95'e yükseltildi (2'ye yuvarlanması için)")
                
                logger.info(f"İki büyük takım karşılaşması: Form oranına göre düzeltme yapıldı. Son 5 maç ortalamaları: {home_team}: {home_recent_avg:.2f}, {away_team}: {away_recent_avg:.2f}")
            else:
                # Form verisi yoksa standart düzeltme
                home_goals *= 0.95
                away_goals *= 0.95
                logger.info("İki büyük takım karşılaşması: Standart düzeltme uygulandı")
        elif home_is_big:
            # Büyük ev sahibi - son form performansını dikkate al
            if home_recent_avg > 2.0:  # Yüksek gol ortalaması varsa
                home_goals = max(home_goals, home_recent_avg * 0.8)
                logger.info(f"Büyük ev sahibi takım yüksek formda: Son 5 maç gol ortalaması {home_recent_avg:.2f}")
            else:
                home_goals *= 0.95  # Hafif düşüş
                away_goals *= 1.05  # Hafif artış
                logger.info("Büyük ev sahibi takım: Gol beklentileri hafif dengelendi")
        elif away_is_big:
            # Büyük deplasman - son form performansını dikkate al
            if away_recent_avg > 1.5:  # Deplasmanda yüksek gol ortalaması
                away_goals = max(away_goals, away_recent_avg * 0.75)
                
                # Deplasman takımının gol beklentisi 1.8'in üzerindeyse ve ortalama gol sayısı 1.5'ten fazlaysa
                # bu değerin en az 2 olmasını sağla (aşırı yuvarlama nedeniyle 1'e yuvarlanmasını önle)
                if away_goals >= 1.75:
                    away_goals = max(away_goals, 1.95)  # 2'ye yuvarlanacak şekilde ayarla
                    logger.info(f"Büyük deplasman takımı: {away_team} beklenen gol sayısı 1.95'e yükseltildi (2'ye yuvarlanması için)")
                
                logger.info(f"Büyük deplasman takımı yüksek formda: Son 5 maç gol ortalaması {away_recent_avg:.2f}")
            else:
                home_goals *= 1.05  # Hafif artış
                away_goals *= 0.95  # Hafif düşüş
                logger.info("Büyük deplasman takımı: Gol beklentileri hafif dengelendi")
            
        return home_goals, away_goals

    def load_or_create_models(self):
        """Optimized model loading"""
        try:
            model_files_exist = (os.path.exists('model_home.h5') and 
                               os.path.exists('model_away.h5'))
            
            if model_files_exist and load_model:
                logger.info("Loading pre-trained neural network models...")
                self.model_home = load_model('model_home.h5')
                self.model_away = load_model('model_away.h5')
            else:
                logger.info("Creating new neural network models...")
                self.model_home = self.build_neural_network(input_dim=self.input_dim)
                self.model_away = self.build_neural_network(input_dim=self.input_dim)
                logger.info("Neural network models created successfully")
        except Exception as e:
            logger.error(f"Model loading/creation error: {str(e)}")
            # Fallback to basic models
            self.model_home = self.build_neural_network(input_dim=self.input_dim)
            self.model_away = self.build_neural_network(input_dim=self.input_dim)
    
    def _calculate_kg_var_probability(self, all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form=None, away_form=None):
        """
        KG VAR (her iki takımın da gol atması) olasılığını hesaplar
        Düşük gol beklentisi durumlarında özel düzeltmeler uygular
        
        Args:
            all_home_goals: Ev sahibi takımın simülasyondaki tüm golleri
            all_away_goals: Deplasman takımının simülasyondaki tüm golleri
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            home_form: Ev sahibi takımın form verileri
            away_form: Deplasman takımının form verileri
            
        Returns:
            float: KG VAR olasılığı (0-1 arası)
        """
        # Temel KG VAR olasılığını hesapla
        if all_home_goals and all_away_goals and len(all_home_goals) > 0:
            kg_var_count = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > 0 and a > 0)
            kg_var_probability = kg_var_count / len(all_home_goals)
            
            # KG VAR olasılığını daha dengeli hale getir - KG YOK problemini önlemek için
            if kg_var_probability < 0.35:
                kg_var_probability = 0.45  # Minimum %45 KG VAR olasılığı
                logger.info(f"Monte Carlo sonrası KG VAR olasılığı %45'e yükseltildi (çok düşüktü)")
        else:
            # Monte Carlo simülasyonu sonuçları yoksa, teorik Poisson olasılığı kullan
            p_home_0 = np.exp(-home_goals_lambda)  # Ev sahibi 0 gol atma olasılığı
            p_away_0 = np.exp(-away_goals_lambda)  # Deplasman 0 gol atma olasılığı
            # En az bir takımın 0 gol atması = ev 0 veya deplasman 0 (veya her ikisi de 0)
            p_at_least_one_0 = p_home_0 + p_away_0 - (p_home_0 * p_away_0)  
            # KG VAR olasılığı = 1 - (en az bir takımın 0 gol atması)
            kg_var_probability = 1 - p_at_least_one_0
            
            # KG VAR olasılığını daha dengeli hale getir
            if kg_var_probability < 0.35:
                kg_var_probability = 0.45  # Minimum %45 KG VAR olasılığı
                logger.info(f"Teorik KG VAR olasılığı %45'e yükseltildi (çok düşüktü)")
            
            logger.info(f"Monte Carlo simülasyonu sonuçları yok, teorik Poisson olasılığı kullanıldı: KG VAR = {kg_var_probability:.2f}")
        
        # DÜŞÜK GOL BEKLENTİSİ DÜZELTMESI DEVRE DIŞI - KG YOK SORUNUNU ÖNLEMEK İÇİN
        # Bu düzeltme sistematik olarak KG YOK tahminlerine neden oluyordu
        if False:  # Bu blok artık çalışmayacak
            total_expected_goals = home_goals_lambda + away_goals_lambda
            logger.info(f"Düşük gol beklentisi tespit edildi: Ev: {home_goals_lambda:.2f}, Dep: {away_goals_lambda:.2f}, Toplam: {total_expected_goals:.2f}")
            
            # Son maçlardaki KG VAR oranını hesapla
            home_btts_rate = 0
            away_btts_rate = 0
            if home_form and 'recent_match_data' in home_form:
                recent_matches = home_form['recent_match_data'][:5]
                if recent_matches:
                    home_btts_rate = sum(1 for m in recent_matches if m.get('goals_scored', 0) > 0 and m.get('goals_conceded', 0) > 0) / len(recent_matches)
            if away_form and 'recent_match_data' in away_form:
                recent_matches = away_form['recent_match_data'][:5]
                if recent_matches:
                    away_btts_rate = sum(1 for m in recent_matches if m.get('goals_scored', 0) > 0 and m.get('goals_conceded', 0) > 0) / len(recent_matches)
            
            # Ortalama KG VAR oranını al ve olasılığı buna göre ayarla
            historical_btts_rate = (home_btts_rate + away_btts_rate) / 2 if (home_btts_rate or away_btts_rate) else 0.25
            
            # Dinamik azaltma faktörü - gerçek KG VAR oranına göre
            if historical_btts_rate > 0.5:
                # Takımlar genelde KG VAR yapıyorsa azaltmayı az yap
                reduction_factor = 0.15
                logger.info(f"Takımların yüksek KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma düşük: %15")
            elif historical_btts_rate > 0.3:
                # Takımlar normal oranda KG VAR yapıyorsa orta düzeyde azalt
                reduction_factor = 0.25
                logger.info(f"Takımların normal KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma orta: %25")
            else:
                # Takımlar nadiren KG VAR yapıyorsa azaltmayı yüksek yap
                reduction_factor = max(0.15, 0.35 - historical_btts_rate * 0.4)
                logger.info(f"Takımların düşük KG VAR oranı ({historical_btts_rate:.2f}) nedeniyle azaltma yüksek: %{round(reduction_factor*100)}")
                
            # Toplam gol beklentisine göre azaltma faktörünü ayarla - SORUN DÜZELTİLDİ
            if total_expected_goals < 0.5:  # Sadece çok ekstrem düşük değerler için
                # Çok düşük toplam gol beklentisi için azaltmayı artır
                reduction_factor *= 1.1  # Daha az agresif azaltma
                logger.info(f"Çok düşük toplam gol beklentisi: KG VAR olasılığı %{round(reduction_factor*100)} azaltılıyor")
            
            # Ev sahibi takım deplasmandan %30'dan daha güçlüyse, ev sahibi avantajını ek olarak değerlendir
            if home_form and away_form and 'form_points' in home_form and 'form_points' in away_form:
                home_form_points = home_form.get('form_points', 0)
                away_form_points = away_form.get('form_points', 0)
                
                # Ev sahibi takım belirgin şekilde daha güçlüyse
                if home_form_points > 0 and away_form_points > 0 and (home_form_points / away_form_points) > (1 + self.EV_SAHIBI_GUC_FARKI_ESIGI):
                    logger.info(f"Ev sahibi takım deplasmandan %{int(self.EV_SAHIBI_GUC_FARKI_ESIGI*100)}'dan daha güçlü tespit edildi. Ev sahibi avantajı ekleniyor.")
                    # Ev sahibi avantajını KG YOK yönünde artır (kg_var olasılığını azalt)
                    reduction_factor += 0.1
                    logger.info(f"Ev sahibi güç farkı nedeniyle ek %10 KG VAR azaltma uygulandı. Toplam azaltma: %{int(reduction_factor*100)}")
            
            # KG VAR olasılığını çok az azalt - SORUN DÜZELTİLDİ
            kg_var_probability = max(0.35, kg_var_probability * (1 - reduction_factor * 0.5))  # Daha yumuşak azaltma
            logger.info(f"Düşük gol beklentisi sonrası KG VAR olasılığı: {kg_var_probability:.2f}")
            
        return kg_var_probability
    def _calculate_kg_yok_probability(self, all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form=None, away_form=None):
        """
        KG YOK (en az bir takımın gol atamaması) olasılığını hesaplar
        
        Args:
            all_home_goals: Ev sahibi takımın simülasyondaki tüm golleri
            all_away_goals: Deplasman takımının simülasyondaki tüm golleri
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            home_form: Ev sahibi takımın form verileri
            away_form: Deplasman takımının form verileri
            
        Returns:
            float: KG YOK olasılığı (0-1 arası)
        """
        # KG YOK olasılığı = 1 - KG VAR olasılığı
        kg_var_probability = self._calculate_kg_var_probability(all_home_goals, all_away_goals, home_goals_lambda, away_goals_lambda, home_form, away_form)
        return 1 - kg_var_probability
        
    def _calculate_over_under_2_5_probability(self, home_goals_lambda, away_goals_lambda, is_over=True):
        """
        2.5 ALT/ÜST olasılığını Monte Carlo simülasyonu ile hesaplar
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            is_over: True ise 2.5 ÜST, False ise 2.5 ALT olasılığını hesaplar
            
        Returns:
            float: 2.5 ALT/ÜST olasılığı (0-1 arası)
        """
        # Monte Carlo simülasyonu çalıştır
        # Basitleştirmek için Poisson dağılımı kullanarak 10000 simülasyon yap
        import numpy as np
        from scipy import stats
        
        # Simülasyon sayısı
        simulations = 10000
        
        # Poisson dağılımından rastgele gol sayıları üret
        np.random.seed(42)  # Tekrarlanabilirlik için
        home_goals = stats.poisson.rvs(mu=home_goals_lambda, size=simulations)
        away_goals = stats.poisson.rvs(mu=away_goals_lambda, size=simulations)
        
        # Toplam golleri hesapla
        total_goals = home_goals + away_goals
        
        # 2.5 ÜST/ALT sayısını hesapla
        if is_over:
            # 2.5 ÜST olasılığı (toplam gol > 2)
            over_count = np.sum(total_goals > 2)
            return over_count / simulations
        else:
            # 2.5 ALT olasılığı (toplam gol <= 2)
            under_count = np.sum(total_goals <= 2)
            return under_count / simulations
            
    def _calculate_over_under_3_5_probability(self, home_goals_lambda, away_goals_lambda, is_over=True):
        """
        3.5 ALT/ÜST olasılığını Monte Carlo simülasyonu ile hesaplar
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı
            away_goals_lambda: Deplasman takımının beklenen gol sayısı
            is_over: True ise 3.5 ÜST, False ise 3.5 ALT olasılığını hesaplar
            
        Returns:
            float: 3.5 ALT/ÜST olasılığı (0-1 arası)
        """
        # Monte Carlo simülasyonu çalıştır
        # Basitleştirmek için Poisson dağılımı kullanarak 10000 simülasyon yap
        import numpy as np
        from scipy import stats
        
        # Simülasyon sayısı
        simulations = 10000
        
        # Poisson dağılımından rastgele gol sayıları üret
        np.random.seed(42)  # Tekrarlanabilirlik için
        home_goals = stats.poisson.rvs(mu=home_goals_lambda, size=simulations)
        away_goals = stats.poisson.rvs(mu=away_goals_lambda, size=simulations)
        
        # Toplam golleri hesapla
        total_goals = home_goals + away_goals
        
        # 3.5 ÜST/ALT sayısını hesapla
        if is_over:
            # 3.5 ÜST olasılığı (toplam gol > 3)
            over_count = np.sum(total_goals > 3)
            return over_count / simulations
        else:
            # 3.5 ALT olasılığı (toplam gol <= 3)
            under_count = np.sum(total_goals <= 3)
            return under_count / simulations
    
    def monte_carlo_simulation(self, home_goals_lambda, away_goals_lambda, simulations=10000, home_form=None, away_form=None, specialized_params=None, kg_var_prediction=None):
        """
        Monte Carlo simülasyonu ile gol dağılımlarını ve maç sonuçlarını tahmin eder
        Son maç sonuçlarına göre ayarlanmış dağılımlar kullanır
        
        Args:
            home_goals_lambda: Ev sahibi takımın beklenen gol sayısı (lambda parametresi)
            away_goals_lambda: Deplasman takımının beklenen gol sayısı (lambda parametresi)
            simulations: Simülasyon sayısı
            home_form: Ev sahibi takımın form bilgileri
            away_form: Deplasman takımının form bilgileri
            specialized_params: Özelleştirilmiş tahmin modeli parametreleri
            kg_var_prediction: KG VAR/YOK tahmin bilgisi (True=KG VAR, False=KG YOK, None=Kısıtlama yok)
            
        Returns:
            dict: Simülasyon sonuçları
        """
        # Tutarlı sonuçlar için deterministic seed kullan
        # Takım ID'lerini veya form verilerini baz alarak seed oluştur
        import random
        seed_value = 42  # Varsayılan seed
        if home_form and away_form:
            home_id = home_form.get('team_id', 0) or 0
            away_id = away_form.get('team_id', 0) or 0
            # Takım ID'leri ve gol beklentilerini kullanarak tutarlı seed oluştur
            seed_value = int((home_id * 1000 + away_id * 100 + int(home_goals_lambda * 10) + int(away_goals_lambda * 10)) % 999999)
        
        # Random seed'i ayarla - bu sayede aynı maç için her zaman aynı sonuçları alırız
        random.seed(seed_value)
        try:
            import numpy as np
            np.random.seed(seed_value)
        except ImportError:
            pass  # NumPy yoksa Python'un random modülü kullanılacak
        logger.info(f"Monte Carlo simülasyonu deterministic seed ile başlatıldı: {seed_value}")
        # Sonuçları tutacak veri yapıları
        home_wins = 0
        away_wins = 0
        draws = 0
        exact_scores = {}  # Kesin skorları ve sayılarını tutacak sözlük
        all_home_goals = []  # Ev sahibi takımın tüm golleri
        all_away_goals = []  # Deplasman takımının tüm golleri
        
        # Maç sonuçlarını tutacak sözlük
        full_time_results = {
            "HOME_WIN": 0,
            "DRAW": 0,
            "AWAY_WIN": 0
        }
        
        # Daha dengeli sonuçlar için form verilerini dikkate al
        if home_form and away_form:
            # Form verilerinden takımların savunma ve hücum performanslarını analiz et
            home_attack_strength = 1.0  # Varsayılan değer
            home_defense_weakness = 1.0  # Varsayılan değer
            away_attack_strength = 1.0  # Varsayılan değer
            away_defense_weakness = 1.0  # Varsayılan değer
            
            # Form ve momentum farkını değerlendir
            form_difference = 0
            momentum_difference = 0
            
            # Form puanlarını kontrol et (mevcutsa)
            if home_form.get('form', {}).get('weighted_form_points') is not None and away_form.get('form', {}).get('weighted_form_points') is not None:
                home_form_points = home_form['form']['weighted_form_points']
                away_form_points = away_form['form']['weighted_form_points']
                form_difference = abs(home_form_points - away_form_points)
                logger.info(f"Form puanları farkı: {form_difference:.3f} (Ev: {home_form_points:.3f}, Deplasman: {away_form_points:.3f})")
            
            # Momentum değerlerini kontrol et (mevcutsa)
            try:
                from model_validation import calculate_advanced_momentum
                if home_form.get('detailed_data', {}).get('all') and away_form.get('detailed_data', {}).get('all'):
                    home_matches = home_form['detailed_data']['all']
                    away_matches = away_form['detailed_data']['all']
                    
                    # Ev sahibi momentum hesaplaması
                    home_momentum = calculate_advanced_momentum(
                        home_matches, 
                        window=min(5, len(home_matches)),
                        recency_weight=1.5
                    )
                    
                    # Deplasman momentum hesaplaması
                    away_momentum = calculate_advanced_momentum(
                        away_matches, 
                        window=min(5, len(away_matches)),
                        recency_weight=1.5
                    )
                    
                    # Momentum farkını hesapla
                    momentum_difference = abs(home_momentum.get('momentum_score', 0) - away_momentum.get('momentum_score', 0))
                    logger.info(f"Momentum farkı: {momentum_difference:.3f}")
            except Exception as e:
                logger.warning(f"Momentum hesaplaması yapılamadı: {str(e)}")
                momentum_difference = 0
            
            # Deplasman takımının hücum performansı analizi
            if away_form and away_form.get('team_stats'):
                avg_away_goals = away_form['team_stats'].get('avg_goals_scored', 0)
                
                # Deplasman takımının hücum gücü analizi
                if avg_away_goals > 0:  # Bölme hatası önlemek için kontrol
                    # Eğer deplasman takım ortalamanın üzerinde gol atıyorsa
                    if avg_away_goals >= 1.5:  # Deplasmanda 1.5+ gol/maç iyi bir hücum göstergesi
                        away_attack_strength = 1.0 + min(0.25, (avg_away_goals - 1.2) * 0.1)  # ETKİ AZALTILDI
                        if away_attack_strength > 1.05:  # Log only if significant adjustment
                            logger.info(f"Deplasman takımı güçlü hücum tespiti: {avg_away_goals:.2f} gol/maç, hücum faktörü: {away_attack_strength:.2f}")
                
                # Ev sahibi takımının savunma zaafiyeti analizi (deplasman takımının gol yeme ortalamasına göre)
                if 'avg_goals_conceded' in away_form['team_stats']:
                    home_conceded_avg = away_form['team_stats'].get('avg_goals_conceded', 0)
                    
                    # Ev sahibi takımı çok gol yiyorsa
                    if home_conceded_avg >= 1.3:
                        home_defense_weakness = 1.0 + min(0.3, (home_conceded_avg - 0.8) * 0.15)  # ETKİ AZALTILDI
                        if home_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Deplasman takımı zayıf savunma tespiti: {home_conceded_avg:.2f} gol/maç, ev sahibi hücum faktörü: {home_defense_weakness:.2f}")
            
            # Bu maç özelinde dinamik olarak:
            # - Ev sahibi takımın hücum performansı
            # - Ev sahibi takımın savunma zaafiyeti
            # - Deplasman takımının hücum performansı
            # - Deplasman takımının savunma zaafiyeti
            # faktörlerini hesapla
            
            # Ev sahibi takımın hücum performansı analizi
            if home_form.get('team_stats'):
                avg_home_goals = home_form['team_stats'].get('avg_goals_scored', 0)
                
                # Ev sahibi takımın hücum gücü analizi
                if avg_home_goals > 0:  # Bölme hatası önlemek için kontrol
                    # Eğer ev sahibi takım ortalamanın üzerinde gol atıyorsa - ETKİSİ AZALTILDI
                    if avg_home_goals >= 1.8:  # 1.8+ gol/maç iyi bir hücum göstergesi
                        home_attack_strength = 1.0 + min(0.25, (avg_home_goals - 1.5) * 0.1)  # ETKİ AZALTILDI
                        if home_attack_strength > 1.05:  # Log only if significant adjustment
                            logger.info(f"Ev sahibi takım güçlü hücum tespiti: {avg_home_goals:.2f} gol/maç, hücum faktörü: {home_attack_strength:.2f}")
                
                # Deplasman takımının savunma zaafiyeti analizi (ev sahibinin gol yeme ortalamasına göre)
                if 'avg_goals_conceded' in home_form['team_stats']:
                    away_conceded_avg = home_form['team_stats'].get('avg_goals_conceded', 0)
                    
                    # Deplasman takımı çok gol yiyorsa - ETKİSİ AZALTILDI
                    if away_conceded_avg >= 1.5:
                        away_defense_weakness = 1.0 + min(0.3, (away_conceded_avg - 1.0) * 0.15)  # ETKİ AZALTILDI
                        if away_defense_weakness > 1.05:  # Log only if significant adjustment
                            logger.info(f"Ev sahibi takım zayıf savunma tespiti: {away_conceded_avg:.2f} gol/maç, deplasman hücum faktörü: {away_defense_weakness:.2f}")
            
            # Dinamik ayarlanmış gol beklentileri - Zehirli savunma analizi entegrasyonu
            adjusted_home_goals = home_goals_lambda * home_attack_strength * away_defense_weakness
            adjusted_away_goals = away_goals_lambda * away_attack_strength * home_defense_weakness
            
            # Monte Carlo simülasyonu başlamadan önce form farkına göre gol beklentilerini düzelt
            if form_difference > 0.15:
                # Form farkını hesapla
                stronger_team = "home" if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0) else "away"
                
                # Form farkı yüksek olan takıma daha fazla gol beklentisi ver
                form_adjustment = min(0.5, form_difference * 0.8)
                
                if stronger_team == "home":
                    adjusted_home_goals = adjusted_home_goals + form_adjustment
                    adjusted_away_goals = max(0.3, adjusted_away_goals - form_adjustment * 0.5)
                    logger.info(f"Form farkı ({form_difference:.2f}) nedeniyle gol beklentileri düzeltildi: Ev takımı güçlü, Ev={adjusted_home_goals:.2f}, Deplasman={adjusted_away_goals:.2f}")
                else:
                    adjusted_away_goals = adjusted_away_goals + form_adjustment
                    adjusted_home_goals = max(0.3, adjusted_home_goals - form_adjustment * 0.5)
                    logger.info(f"Form farkı ({form_difference:.2f}) nedeniyle gol beklentileri düzeltildi: Deplasman güçlü, Ev={adjusted_home_goals:.2f}, Deplasman={adjusted_away_goals:.2f}")
            
            # Farklı dağılımları daha dengeli kullan
            # Daha fazla çeşitlilik için random_selector ile dağılım seç
            random_selector = np.random.random()

            # Ev sahibi skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Form farkını hesapla
            form_diff = 0
            if home_form and away_form:
                home_form_points = home_form.get('form', {}).get('weighted_form_points', 0)
                away_form_points = away_form.get('form', {}).get('weighted_form_points', 0)
                form_diff = home_form_points - away_form_points
                logger.debug(f"Form farkı: {form_diff:.2f} (Ev: {home_form_points:.2f}, Dep: {away_form_points:.2f})")

            # Ev sahibi skoru için dinamik maksimum değer belirle
            max_home_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Özelleştirilmiş modelden maksimum skor sınırı
            if specialized_params and 'max_score' in specialized_params:
                max_home_score = specialized_params['max_score']
                logger.debug(f"Özelleştirilmiş model maksimum ev sahibi skoru: {max_home_score}")
            else:
                # Standart maksimum skor hesaplama - form farkını da dikkate alarak
                if adjusted_home_goals < 0.5:
                    max_home_score = 1 + (1 if form_diff > 0.3 else 0)  # Güçlü form farkı varsa +1
                elif adjusted_home_goals < 1.0:
                    max_home_score = 2 + (1 if form_diff > 0.5 else 0)
                elif adjusted_home_goals < 2.0:
                    max_home_score = 3
                else:
                    max_home_score = 4 + (1 if form_diff > 0.7 else 0)  # Çok güçlü form farkı varsa +1
                logger.debug(f"Ev sahibi dinamik skor sınırı: {max_home_score} (gol beklentisi: {adjusted_home_goals:.2f}, form farkı: {form_diff:.2f})")
            
            # Deplasman skoru için dinamik maksimum değer belirle
            max_away_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Özelleştirilmiş modelden maksimum skor sınırı
            if specialized_params and 'max_score' in specialized_params:
                max_away_score = specialized_params['max_score']
                logger.debug(f"Özelleştirilmiş model maksimum deplasman skoru: {max_away_score}")
            else:
                # Standart maksimum skor hesaplama - form farkını da dikkate alarak
                if adjusted_away_goals < 0.5:
                    max_away_score = 1 + (1 if form_diff < -0.3 else 0)  # Deplasman form avantajı varsa +1
                elif adjusted_away_goals < 1.0:
                    max_away_score = 2 + (1 if form_diff < -0.5 else 0)
                elif adjusted_away_goals < 2.0:
                    max_away_score = 3
                else:
                    max_away_score = 4 + (1 if form_diff < -0.7 else 0)  # Çok güçlü deplasman form avantajı varsa +1
                logger.debug(f"Deplasman dinamik skor sınırı: {max_away_score} (gol beklentisi: {adjusted_away_goals:.2f}, form farkı: {form_diff:.2f})")
            
            # Simülasyon sayacı
            valid_simulations = 0
            remaining_attempts = simulations * 2  # En fazla bu kadar deneme yapılacak
            
            # Simülasyonları gerçekleştir
            while valid_simulations < simulations and remaining_attempts > 0:
                remaining_attempts -= 1
                
                # Form farkı yüksek olan takımlarda, güçlü takımın Poisson parametresini artır
                poisson_home_lambda = adjusted_home_goals
                poisson_away_lambda = adjusted_away_goals
                
                if form_difference > 0.15:
                    stronger_team = "home" if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0) else "away"
                    
                    if stronger_team == "home":
                        # Ev sahibi daha formda - hem hücum hem savunma avantajı artır
                        poisson_home_lambda = adjusted_home_goals * (1 + min(0.3, form_difference * 0.5))
                        poisson_away_lambda = adjusted_away_goals * (1 - min(0.2, form_difference * 0.3))
                        logger.info(f"Poisson parametreleri form farkına göre ayarlandı: Ev={poisson_home_lambda:.2f}, Deplasman={poisson_away_lambda:.2f}")
                    else:
                        # Deplasman daha formda - hem hücum hem savunma avantajı artır
                        poisson_away_lambda = adjusted_away_goals * (1 + min(0.3, form_difference * 0.5))
                        poisson_home_lambda = adjusted_home_goals * (1 - min(0.2, form_difference * 0.3))
                        logger.info(f"Poisson parametreleri form farkına göre ayarlandı: Ev={poisson_home_lambda:.2f}, Deplasman={poisson_away_lambda:.2f}")
                
                # Poisson parametrelerini maksimum sınırlarla tutarlı hale getir
                MAX_POISSON_LAMBDA = 3.5  # Gol beklentisi sınırıyla aynı
                if poisson_home_lambda > MAX_POISSON_LAMBDA:
                    logger.warning(f"Ev sahibi Poisson parametresi çok yüksek ({poisson_home_lambda:.2f}), {MAX_POISSON_LAMBDA} ile sınırlandı")
                    poisson_home_lambda = MAX_POISSON_LAMBDA
                    
                if poisson_away_lambda > MAX_POISSON_LAMBDA:
                    logger.warning(f"Deplasman Poisson parametresi çok yüksek ({poisson_away_lambda:.2f}), {MAX_POISSON_LAMBDA} ile sınırlandı")
                    poisson_away_lambda = MAX_POISSON_LAMBDA
                
                # Ev sahibi skor tahmini - İyileştirilmiş dağılım kullanımı ve aşırı değer kontrolü
                if random_selector < 0.6:  # %60 Poisson (daha tutarlı)
                    raw_score = np.random.poisson(poisson_home_lambda)
                    # Makul sınırlar içinde tut
                    home_score = min(raw_score, max_home_score)
                else:  # %40 Negatif Binom
                    try:
                        # Negatif Binom parametreleri - daha kontrollü varyans
                        home_r = max(1, poisson_home_lambda**2 / (max(0.1, poisson_home_lambda - 0.3)))
                        home_p = home_r / (home_r + poisson_home_lambda)
                        
                        raw_score = np.random.negative_binomial(home_r, home_p)
                        # Daha sıkı sınırlar uygula - beklenen değere daha yakın kal
                        home_score = min(raw_score, max_home_score)
                        
                        # Aşırı değer kontrolü - beklenen değerden çok sapan skorları sınırla
                        if home_score > poisson_home_lambda * 2 + 1:
                            home_score = min(int(poisson_home_lambda * 2), max_home_score)
                    except ValueError:
                        # Hata durumunda Poisson'a geri dön
                        raw_score = np.random.poisson(poisson_home_lambda)
                        home_score = min(raw_score, max_home_score)
                
                # Deplasman skor tahmini - İyileştirilmiş dağılım kullanımı ve aşırı değer kontrolü
                if random_selector < 0.6:  # %60 Poisson (daha tutarlı)
                    raw_score = np.random.poisson(poisson_away_lambda)
                    # Makul sınırlar içinde tut
                    away_score = min(raw_score, max_away_score)
                else:  # %40 Negatif Binom
                    try:
                        # Negatif Binom parametreleri - daha kontrollü varyans
                        away_r = max(1, poisson_away_lambda**2 / (max(0.1, poisson_away_lambda - 0.3)))
                        away_p = away_r / (away_r + poisson_away_lambda)
                        
                        raw_score = np.random.negative_binomial(away_r, away_p)
                        # Daha sıkı sınırlar uygula
                        away_score = min(raw_score, max_away_score)
                        
                        # Aşırı değer kontrolü - beklenen değerden çok sapan skorları sınırla
                        if away_score > poisson_away_lambda * 2 + 1:
                            away_score = min(int(poisson_away_lambda * 2), max_away_score)
                    except ValueError:
                        # Hata durumunda Poisson'a geri dön
                        raw_score = np.random.poisson(poisson_away_lambda)
                        away_score = min(raw_score, max_away_score)
                
                # Form farkına göre skor düzeltmesi - form farkı büyükse beraberliği azaltma
                if form_difference > 0.3 and home_score == away_score:
                    # Form farkına göre beraberliği %70 ihtimalle boz
                    if np.random.random() < 0.7:
                        # Hangi takım daha formda belirle ve ona bir gol ekle
                        if home_form.get('form', {}).get('weighted_form_points', 0) > away_form.get('form', {}).get('weighted_form_points', 0):
                            home_score += 1
                        else:
                            away_score += 1
                
                # AKILLI SKOR FİLTRELEME - Takım yeteneklerine göre dinamik sınırlar
                total_goals = home_score + away_score
                expected_total = poisson_home_lambda + poisson_away_lambda
                
                # Yüksek gol atan takımlar için esnek sınırlar
                if expected_total >= 3.0:  # Yüksek skorlu maç beklentisi
                    max_allowed_total = max(6, expected_total + 2.5)  # 4-3, 5-2 gibi skorlara izin ver
                elif expected_total >= 2.0:  # Orta skorlu maç
                    max_allowed_total = max(4, expected_total + 1.5)  # 3-2, 2-3 gibi skorlara izin ver
                else:  # Düşük skorlu maç
                    max_allowed_total = max(3, expected_total + 1.0)  # 2-1, 1-2 gibi skorlara izin ver
                
                if total_goals > max_allowed_total:
                    continue  # Bu simülasyonu geçersiz say ve tekrar dene
                    
                # Beklenen değerden çok uzaklaşan skorları filtrele
                if abs(home_score - poisson_home_lambda) > 2 or abs(away_score - poisson_away_lambda) > 2:
                    continue  # Bu simülasyonu geçersiz say ve tekrar dene
                
                # KG VAR/YOK kısıtlaması KALDIRILDI - Monte Carlo simülasyonu tamamen bağımsız
                # Skor hesaplama artık KG tahminlerinden etkilenmiyor
                
                # Bu noktaya kadar geldiyse simülasyon geçerli
                valid_simulations += 1
                
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)

                # Kesin skor tahmini
                exact_score_key = f"{home_score}-{away_score}"
                exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1

                # Maç sonucu
                if home_score > away_score:
                    home_wins += 1
                    full_time_results["HOME_WIN"] += 1
                elif home_score < away_score:
                    away_wins += 1
                    full_time_results["AWAY_WIN"] += 1
                else:
                    draws += 1
                    full_time_results["DRAW"] += 1
            
            # Eğer yeterli simülasyon yapılamazsa uyarı ver ve kısıtlamaları gevşet
            if valid_simulations < simulations:
                logger.warning(f"KG VAR/YOK kısıtlaması nedeniyle yeterli simülasyon yapılamadı: {valid_simulations}/{simulations}")
                if valid_simulations == 0:
                    logger.error("Hiç geçerli simülasyon yapılamadı, basit simülasyon yapılıyor")
                    # Fallback simülasyon - recursion önlemek için
                    for _ in range(min(1000, simulations)):
                        home_score = min(np.random.poisson(home_goals_lambda), 5)
                        away_score = min(np.random.poisson(away_goals_lambda), 5)
                        all_home_goals.append(home_score)
                        all_away_goals.append(away_score)
                        exact_score_key = f"{home_score}-{away_score}"
                        exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1
                        if home_score > away_score:
                            home_wins += 1
                            full_time_results["HOME_WIN"] += 1
                        elif home_score < away_score:
                            away_wins += 1
                            full_time_results["AWAY_WIN"] += 1
                        else:
                            draws += 1
                            full_time_results["DRAW"] += 1
        else:
            # Form verileri yoksa, standart Monte Carlo simülasyonu kullan
            logger.warning("Form verileri bulunamadı, standart Monte Carlo simülasyonu kullanılıyor")
            
            # Standart maksimum skor hesaplama (form verileri olmadan)
            max_home_score = 1  # Varsayılan makul maksimum değer
            if home_goals_lambda <= 0.5:
                max_home_score = 1
            elif home_goals_lambda <= 1.0:
                max_home_score = 2
            elif home_goals_lambda <= 1.5:
                max_home_score = 3
            elif home_goals_lambda <= 2.0:
                max_home_score = 3
            else:
                max_home_score = 4
            
            max_away_score = 1  # Varsayılan makul maksimum değer
            if away_goals_lambda <= 0.5:
                max_away_score = 1
            elif away_goals_lambda <= 1.0:
                max_away_score = 2
            elif away_goals_lambda <= 1.5:
                max_away_score = 3
            elif away_goals_lambda <= 2.0:
                max_away_score = 3
            else:
                max_away_score = 4
            
            # Simülasyon sayacı
            valid_simulations = 0
            remaining_attempts = simulations * 2  # En fazla bu kadar deneme yapılacak
            
            # Simülasyonları gerçekleştir - kısıtlamalar olmadan
            for _ in range(simulations):
                # Poisson dağılımı ile gol sayılarını tahmin et
                home_score = min(np.random.poisson(home_goals_lambda), max_home_score)
                away_score = min(np.random.poisson(away_goals_lambda), max_away_score)
                
                # Basit ve tutarlı simülasyon - kısıtlama uygulamadan
                valid_simulations += 1
                
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)
                
                # Kesin skor tahmini
                exact_score_key = f"{home_score}-{away_score}"
                exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1
                
                # Maç sonucu
                if home_score > away_score:
                    home_wins += 1
                    full_time_results["HOME_WIN"] += 1
                elif home_score < away_score:
                    away_wins += 1
                    full_time_results["AWAY_WIN"] += 1
                else:
                    draws += 1
                    full_time_results["DRAW"] += 1
            
            # Eğer yeterli simülasyon yapılamazsa uyarı ver ve kısıtlamaları gevşet
            if valid_simulations < simulations:
                logger.warning(f"KG VAR/YOK kısıtlaması nedeniyle yeterli simülasyon yapılamadı: {valid_simulations}/{simulations}")
                if valid_simulations == 0:
                    logger.error("Hiç geçerli simülasyon yapılamadı, basit simülasyon yapılıyor")
                    # Fallback simülasyon - recursion önlemek için
                    for _ in range(min(1000, simulations)):
                        home_score = min(np.random.poisson(home_goals_lambda), 5)
                        away_score = min(np.random.poisson(away_goals_lambda), 5)
                        all_home_goals.append(home_score)
                        all_away_goals.append(away_score)
                        exact_score_key = f"{home_score}-{away_score}"
                        exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1
                        if home_score > away_score:
                            home_wins += 1
                            full_time_results["HOME_WIN"] += 1
                        elif home_score < away_score:
                            away_wins += 1
                            full_time_results["AWAY_WIN"] += 1
                        else:
                            draws += 1
                            full_time_results["DRAW"] += 1
        
        # Initialize betting probabilities with defaults
        both_teams_score_prob = 0.5
        over_25_prob = 0.5
        over_35_prob = 0.3
        
        # Toplam golleri hesapla ve betting predictions
        if len(all_home_goals) > 0 and len(all_away_goals) > 0:  # Bölme hatası önlemek için kontrol
            total_goals = sum(all_home_goals) + sum(all_away_goals)
            avg_total_goals = total_goals / len(all_home_goals)
            
            # Betting predictions hesapla
            both_teams_score_count = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h > 0 and a > 0)
            over_25_count = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a > 2.5)
            over_35_count = sum(1 for h, a in zip(all_home_goals, all_away_goals) if h + a > 3.5)
            
            # Betting probabilities - gerçekçi sınırlar uygula (futbolda %100 kesinlik olmaz)
            both_teams_score_prob = both_teams_score_count / len(all_home_goals)
            both_teams_score_prob = max(0.10, min(0.90, both_teams_score_prob))  # En az %10, en fazla %90
            
            over_25_prob = over_25_count / len(all_home_goals)
            over_25_prob = max(0.10, min(0.90, over_25_prob))  # En az %10, en fazla %90
            
            over_35_prob = over_35_count / len(all_home_goals)
            over_35_prob = max(0.05, min(0.85, over_35_prob))  # En az %5, en fazla %85
            
            # Maç sonucu olasılıkları
            home_win_prob = home_wins / len(all_home_goals)
            away_win_prob = away_wins / len(all_home_goals)
            draw_prob = draws / len(all_home_goals)
            
            # Form ve momentum farklarına göre beraberlik olasılığını ayarla
            # Beklenen gol farkına göre de ayarlama yapalım
            goals_difference = abs(home_goals_lambda - away_goals_lambda)
            
            # Form, momentum veya beklenen gol farkı varsa düzeltme yap
            if form_difference > 0.15 or momentum_difference > 0.2 or goals_difference > 1.0:
                # Form, momentum veya beklenen gol farkı varsa, beraberlik olasılığını daha agresif şekilde azalt
                form_momentum_factor = min(0.70, max(form_difference * 1.2, momentum_difference * 0.8))
                
                # Beklenen gol farkı 1'den büyükse, farkın büyüklüğüne göre beraberliği azalt
                # Fark 1.0 ise 0.3, 2.0 ise 0.5, 3.0 ve üzeri ise 0.8 oranında azaltma yapılacak
                goal_diff_adjustment = min(0.80, goals_difference * 0.25)
                
                # İki faktörden büyük olanı kullan
                adjustment_factor = max(form_momentum_factor, goal_diff_adjustment)
                
                logger.info(f"Form/momentum farkı: {form_difference:.2f}/{momentum_difference:.2f}, Gol farkı: {goals_difference:.2f}")
                logger.info(f"Beraberlik olasılığı düzeltme faktörü: {adjustment_factor:.3f} (form/momentum: {form_momentum_factor:.2f}, gol farkı: {goal_diff_adjustment:.2f})")
                
                # Beraberlik olasılığından çıkarılacak miktar
                draw_reduction = draw_prob * adjustment_factor
                
                # Beraberlik olasılığını azalt
                new_draw_prob = max(0.05, draw_prob - draw_reduction)  # En az %5 beraberlik olasılığı kalsın
                
                # Çıkarılan olasılığı diğer sonuçlara dağıt (güçlü takıma daha fazla)
                if home_goals_lambda > away_goals_lambda:
                    # Ev sahibi daha güçlüyse, ona daha fazla olasılık ver
                    home_extra = draw_reduction * 0.7
                    away_extra = draw_reduction * 0.3
                elif away_goals_lambda > home_goals_lambda:
                    # Deplasman daha güçlüyse, ona daha fazla olasılık ver
                    home_extra = draw_reduction * 0.3
                    away_extra = draw_reduction * 0.7
                else:
                    # Eşitse, eşit dağıt
                    home_extra = draw_reduction * 0.5
                    away_extra = draw_reduction * 0.5
                
                # Yeni olasılıkları hesapla
                new_home_win_prob = home_win_prob + home_extra
                new_away_win_prob = away_win_prob + away_extra
                
                logger.info(f"Beraberlik olasılığı ayarlandı: {draw_prob:.3f} -> {new_draw_prob:.3f}")
                logger.info(f"Ev sahibi galibiyet olasılığı ayarlandı: {home_win_prob:.3f} -> {new_home_win_prob:.3f}")
                logger.info(f"Deplasman galibiyet olasılığı ayarlandı: {away_win_prob:.3f} -> {new_away_win_prob:.3f}")
                
                # Olasılıkları güncelle
                draw_prob = new_draw_prob
                home_win_prob = new_home_win_prob
                away_win_prob = new_away_win_prob
            
            # Ev sahibi ve deplasman takımlarının ortalama golleri
            avg_home_goals = sum(all_home_goals) / len(all_home_goals)
            avg_away_goals = sum(all_away_goals) / len(all_away_goals)
        else:
            # Hiç simülasyon yapılamazsa, kısıtlamaları gevşetip tekrar dene
            logger.error("Hiç geçerli simülasyon yapılamadı, kısıtlamaları gevşeterek tekrar deneniyor")
            
            # Orijinal parametreleri koru
            avg_total_goals = home_goals_lambda + away_goals_lambda
            avg_home_goals = home_goals_lambda
            avg_away_goals = away_goals_lambda
            
            # Daha fazla simülasyon yap, kısıtlamaları gevşeterek
            exact_scores = {}
            all_home_goals = []
            all_away_goals = []
            home_wins = 0
            away_wins = 0
            draws = 0
            
            # KG VAR/YOK kısıtlaması olmadan tekrar simülasyon yap
            logger.info(f"KG VAR/YOK kısıtlamaları gevşetilerek yeni simülasyon yapılıyor")
            valid_simulations = 0
            
            # Eşik değerini düşük tut, hızlıca en az birkaç skor üret
            min_simulations = 100
            
            # Monte Carlo simülasyonunu tekrarla, kısıtlamalar olmadan
            for i in range(simulations * 2):  # Daha fazla deneme şansı
                # Poisson dağılımından doğrudan skor tahminleri
                home_score = min(np.random.poisson(home_goals_lambda), 5)  # Maksimum 5 golle sınırla
                away_score = min(np.random.poisson(away_goals_lambda), 5)  # Maksimum 5 golle sınırla
                
                # KG VAR/YOK kısıtlamalarını gevşet
                if kg_var_prediction is True:
                    # KG VAR için normalden daha yüksek bir oranda 1+ gollü skorları kabul et
                    if home_score == 0 or away_score == 0:
                        # %50 ihtimalle bu skoru yine de kabul et
                        if np.random.random() < 0.5:
                            # Sıfır gole sahip takımın skorunu 1'e yükselt
                            if home_score == 0:
                                home_score = 1
                            if away_score == 0:
                                away_score = 1
                
                # Simülasyonu kaydet
                valid_simulations += 1
                all_home_goals.append(home_score)
                all_away_goals.append(away_score)
                
                # Skoru kaydet
                score = f"{home_score}-{away_score}"
                exact_scores[score] = exact_scores.get(score, 0) + 1
                
                # Maç sonucunu güncelle
                if home_score > away_score:
                    home_wins += 1
                elif home_score < away_score:
                    away_wins += 1
                else:
                    draws += 1
                
                # Yeterli simülasyon elde edildiyse dur
                if valid_simulations >= min_simulations and len(exact_scores) >= 5:
                    break
            
            # Maç sonucu olasılıklarını hesapla
            total_simulations = valid_simulations
            home_win_prob = home_wins / total_simulations if total_simulations > 0 else 0.33
            away_win_prob = away_wins / total_simulations if total_simulations > 0 else 0.33
            draw_prob = draws / total_simulations if total_simulations > 0 else 0.34
            
            # Eksik common_sense skorları ekle
            if valid_simulations > 0:
                # Ortalama gollere göre en olası skorlar
                rounded_home = max(0, min(3, round(home_goals_lambda)))
                rounded_away = max(0, min(3, round(away_goals_lambda)))
                typical_score = f"{rounded_home}-{rounded_away}"
                
                # Eğer en olası skor örneklem içinde yoksa, ekle
                if typical_score not in exact_scores:
                    exact_scores[typical_score] = max(1, int(valid_simulations * 0.05))
                    logger.info(f"Olası skor {typical_score} eklendi (beklenen gollere göre)")
            
            logger.info(f"Gevşetilmiş simülasyon: {valid_simulations} simülasyon, {len(exact_scores)} farklı skor")
            
            # Ortalama golleri güncelle
            if all_home_goals and all_away_goals:
                avg_home_goals = sum(all_home_goals) / len(all_home_goals)
                avg_away_goals = sum(all_away_goals) / len(all_away_goals)
        
        # Skorların gerçeklik puanını hesapla
        normalized_scores = {}
        for score, count in exact_scores.items():
            home, away = map(int, score.split('-'))
            
            # Gerçekçilik puanı (0-1 arası)
            realism_score = 1.0
            
            # Çok yüksek skorlar için gerçekçilik puanı düşür
            if home + away > 5:
                realism_score *= 0.5
            
            # Gol farkı çok yüksekse gerçekçilik puanı düşür  
            if abs(home - away) > 3:
                realism_score *= 0.7
            
            # Beklenen skordan çok sapan skorlar için gerçekçilik puanı düşür
            if home_goals_lambda > 0 and away_goals_lambda > 0:  # Sıfıra bölme hatasını önle
                home_deviation = abs(home - home_goals_lambda)
                away_deviation = abs(away - away_goals_lambda)
                if home_deviation + away_deviation > 3:
                    realism_score *= 0.6
            
            # Ağırlıklandırılmış sayımı güncelle
            normalized_scores[score] = count * realism_score
            
        logger.info(f"Skorlar gerçekçilik puanlarına göre ağırlıklandırıldı.")
        
        # Olası 5 kesin skoru bul (olasılıklarıyla birlikte)
        sorted_scores = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        top_5_scores = sorted_scores[:5]
        total_top_5 = sum(count for _, count in top_5_scores) if top_5_scores else 0
        
        # En olası skor ve olasılığı
        most_likely_score = sorted_scores[0] if sorted_scores else ("0-0", 0)
        most_likely_score_prob = most_likely_score[1] / len(all_home_goals) if all_home_goals else 0
        
        # En olası 5 skorun olasılıkları
        top_5_probs = {score: count / len(all_home_goals) for score, count in top_5_scores} if all_home_goals else {}
        
        logger.info(f"En olası skorlar ve olasılıkları: {top_5_probs}")
        
        # KG VAR/YOK kısıtlamasına göre log
        if kg_var_prediction is True:
            logger.info("Monte Carlo simülasyonu KG VAR kısıtlamasıyla çalıştırıldı")
        elif kg_var_prediction is False:
            logger.info("Monte Carlo simülasyonu KG YOK kısıtlamasıyla çalıştırıldı")
        
        # Sonuçları döndür
        return {
            "match_result_probs": {
                "home_win": home_win_prob,
                "draw": draw_prob,
                "away_win": away_win_prob
            },
            "full_time_results": full_time_results,
            "exact_scores": exact_scores,
            "most_likely_score": most_likely_score,
            "most_likely_score_prob": most_likely_score_prob,
            "top_5_scores": top_5_scores,
            "top_5_probs": top_5_probs,
            "avg_goals": {
                "home": avg_home_goals,
                "away": avg_away_goals,
                "total": avg_total_goals
            },
            "over_under": {
                "over_2_5": over_25_prob,
                "under_2_5": 1 - over_25_prob,
                "over_3_5": over_35_prob,
                "under_3_5": 1 - over_35_prob
            },
            "both_teams_to_score": {
                "yes": both_teams_score_prob,
                "no": 1 - both_teams_score_prob
            },
            "all_home_goals": all_home_goals,
            "all_away_goals": all_away_goals,
            "simulations": len(all_home_goals) if all_home_goals else 0
        }

    def build_neural_network(self, input_dim):
        """Sinir ağı modeli oluştur"""
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.2))  # Overfitting'i önlemek için dropout
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(1, activation='linear'))  # Gol tahmini için lineer aktivasyon
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model

    def prepare_data_for_neural_network(self, team_form, is_home=True):
        """Sinir ağı için veri hazırla"""
        if not team_form:
            return None

        if is_home:
            performance = team_form.get('home_performance', {})
            bayesian = team_form.get('bayesian', {})

            features = [
                performance.get('avg_goals_scored', 0),
                performance.get('avg_goals_conceded', 0),
                performance.get('weighted_avg_goals_scored', 0),
                performance.get('weighted_avg_goals_conceded', 0),
                performance.get('form_points', 0),
                performance.get('weighted_form_points', 0),
                bayesian.get('home_lambda_scored', 0),
                bayesian.get('home_lambda_conceded', 0),
                team_form.get('recent_matches', 0),
                team_form.get('home_matches', 0)
            ]
        else:
            performance = team_form.get('away_performance', {})
            bayesian = team_form.get('bayesian', {})

            features = [
                performance.get('avg_goals_scored', 0),
                performance.get('avg_goals_conceded', 0),
                performance.get('weighted_avg_goals_scored', 0),
                performance.get('weighted_avg_goals_conceded', 0),
                performance.get('form_points', 0),
                performance.get('weighted_form_points', 0),
                bayesian.get('away_lambda_scored', 0),
                bayesian.get('away_lambda_conceded', 0),
                team_form.get('recent_matches', 0),
                team_form.get('away_matches', 0)
            ]

        # Handle both real numpy and mock numpy arrays
        features_array = np.array(features)
        if hasattr(features_array, 'reshape'):
            return features_array.reshape(1, -1)
        else:
            # Fallback for mock numpy - return as nested list for 2D structure
            return [features]

    def train_neural_network(self, X_train, y_train, is_home=True):
        """Sinir ağını eğit"""
        try:
            X_scaled = self.scaler.fit_transform(X_train)
            model = self.model_home if is_home else self.model_away

            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )

            model.fit(
                X_scaled, y_train,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping],
                verbose=1
            )

            # Modeli kaydet
            model_path = 'model_home.h5' if is_home else 'model_away.h5'
            save_model(model, model_path)
            logger.info(f"Sinir ağı modeli kaydedildi: {model_path}")

            return model
        except Exception as e:
            logger.error(f"Sinir ağı eğitilirken hata: {str(e)}")
            return None

    def calculate_weighted_form(self, matches, decay_factor=0.9):
        """
        Son maçları azalan ağırlıklarla değerlendiren fonksiyon
        En son maç en yüksek ağırlığa sahip (1.0), öncekiler geometrik azalır
        
        Args:
            matches: Maç listesi (en yeniden en eskiye doğru sıralı)
            decay_factor: Azalma faktörü (0.9 = her bir önceki maç %10 daha az önemli)
        
        Returns:
            weighted_form: Ağırlıklı form puanı
        """
        if not matches:
            return {
                'weighted_goals_scored': 1.0,
                'weighted_goals_conceded': 1.0,
                'weighted_points': 1.0,
                'confidence': 0.0
            }
            
        weights = [decay_factor ** i for i in range(len(matches))]
        total_weight = sum(weights)
        
        weighted_goals_scored = 0
        weighted_goals_conceded = 0
        weighted_points = 0
        
        for i, match in enumerate(matches):
            weight = weights[i] / total_weight
            weighted_goals_scored += match.get('goals_scored', 0) * weight
            weighted_goals_conceded += match.get('goals_conceded', 0) * weight
            
            # Galibiyet = 3, Beraberlik = 1, Mağlubiyet = 0
            result = match.get('ft_result', '')
            points = 3 if result == 'W' else (1 if result == 'D' else 0)
            weighted_points += points * weight
        
        # Güven faktörü - ne kadar çok maç varsa o kadar güvenilir (maksimum 10 maç için 1.0)
        confidence = min(1.0, len(matches) / 10)
        
        return {
            'weighted_goals_scored': weighted_goals_scored,
            'weighted_goals_conceded': weighted_goals_conceded,
            'weighted_points': weighted_points,
            'confidence': confidence
        }
    def get_team_form(self, team_id, last_matches=21):
        """Takımın son maçlarındaki performansını al - son 21 maç verisi için tam değerlendirme"""
        try:
            # Son 12 aylık maçları al (daha uzun süreli veri için)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            url = "https://apiv3.apifootball.com/"
            params = {
                'action': 'get_events',
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'team_id': team_id,
                'APIkey': self.api_key
            }

            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.error(f"API hatası: {response.status_code}")
                return None

            matches = response.json()

            if not isinstance(matches, list):
                logger.error(f"API'den takım verileri alınamadı: {matches}")
                # API planı kısıtlamaları nedeniyle takım detaylı verileri alınamıyor
                return None

            # Maçları tarihe göre sırala (en yeniden en eskiye)
            matches.sort(key=lambda x: x.get('match_date', ''), reverse=True)

            # Son maçları al
            recent_matches = matches[:last_matches]

            # Form verilerini hesapla
            goals_scored = 0
            goals_conceded = 0
            points = 0

            # Ev ve deplasman maçları için ayrı değişkenler
            home_matches = []
            away_matches = []
            home_goals_scored = 0
            home_goals_conceded = 0
            away_goals_scored = 0
            away_goals_conceded = 0
            home_points = 0
            away_points = 0

            # Üstel ağırlıklandırma için decay factor
            decay_factor = 0.9

            # Son maçların detaylarını ekle ve ağırlıklı ortalamalar için gerekli verileri topla
            recent_match_data = []

            # Ağırlıklandırma için kullanılacak değerler
            total_weights = 0
            total_home_weights = 0
            total_away_weights = 0
            weighted_goals_scored = 0
            weighted_goals_conceded = 0
            weighted_home_goals_scored = 0
            weighted_home_goals_conceded = 0
            weighted_away_goals_scored = 0
            weighted_away_goals_conceded = 0
            weighted_points = 0
            weighted_home_points = 0
            weighted_away_points = 0

            for i, match in enumerate(recent_matches):
                # Sadece tamamlanmış maçları dahil et
                match_status = match.get('match_status', '')
                if match_status != 'Finished' and match_status != 'Match Finished' and match_status != 'FT':
                    continue
                    
                home_team_id = match.get('match_hometeam_id')
                home_score = int(match.get('match_hometeam_score', 0) or 0)
                away_score = int(match.get('match_awayteam_score', 0) or 0)

                # Bu maç için ağırlık hesapla (üstel azalma modeli)
                weight = decay_factor ** i
                total_weights += weight

                # Takım ev sahibi ise
                is_home = home_team_id == team_id
                goals_for = home_score if is_home else away_score
                goals_against = away_score if is_home else home_score

                if is_home:
                    home_matches.append(match)
                    home_goals_scored += goals_for
                    home_goals_conceded += goals_against
                    total_home_weights += weight
                    weighted_home_goals_scored += goals_for * weight
                    weighted_home_goals_conceded += goals_against * weight

                    if home_score > away_score:  # Galibiyet
                        home_points += 3
                        weighted_home_points += 3 * weight
                    elif home_score == away_score:  # Beraberlik
                        home_points += 1
                        weighted_home_points += 1 * weight
                else:
                    away_matches.append(match)
                    away_goals_scored += goals_for
                    away_goals_conceded += goals_against
                    total_away_weights += weight
                    weighted_away_goals_scored += goals_for * weight
                    weighted_away_goals_conceded += goals_against * weight

                    if away_score > home_score:  # Galibiyet
                        away_points += 3
                        weighted_away_points += 3 * weight
                    elif away_score == home_score:  # Beraberlik
                        away_points += 1
                        weighted_away_points += 1 * weight

                # Tüm maçlar için toplamlar
                goals_scored += goals_for
                goals_conceded += goals_against
                weighted_goals_scored += goals_for * weight
                weighted_goals_conceded += goals_against * weight

                if (is_home and home_score > away_score) or (not is_home and away_score > home_score):
                    points += 3
                    weighted_points += 3 * weight
                elif home_score == away_score:
                    points += 1
                    weighted_points += 1 * weight

                # İlk yarı skorlarını al
                half_time_home = int(match.get('match_hometeam_halftime_score', 0) or 0)
                half_time_away = int(match.get('match_awayteam_halftime_score', 0) or 0)
                
                # İlk yarı ve tam maç sonuçlarını belirle
                ht_result = 'W' if (is_home and half_time_home > half_time_away) or (not is_home and half_time_away > half_time_home) else \
                           'D' if half_time_home == half_time_away else 'L'
                
                ft_result = 'W' if (is_home and home_score > away_score) or (not is_home and away_score > home_score) else \
                           'D' if home_score == away_score else 'L'
                
                # İY/MS formatı (1/1, 1/X, 1/2, X/1, X/X, X/2, 2/1, 2/X, 2/2)
                iy_ms_code = ''
                if is_home:
                    if half_time_home > half_time_away:
                        iy_ms_code = '1'
                    elif half_time_home == half_time_away:
                        iy_ms_code = 'X'
                    else:
                        iy_ms_code = '2'
                        
                    if home_score > away_score:
                        iy_ms_code += '/1'
                    elif home_score == away_score:
                        iy_ms_code += '/X'
                    else:
                        iy_ms_code += '/2'
                else:
                    if half_time_away > half_time_home:
                        iy_ms_code = '1'
                    elif half_time_away == half_time_home:
                        iy_ms_code = 'X'
                    else:
                        iy_ms_code = '2'
                        
                    if away_score > home_score:
                        iy_ms_code += '/1'
                    elif away_score == home_score:
                        iy_ms_code += '/X'
                    else:
                        iy_ms_code += '/2'
                
                # İlk yarı skoru
                ht_goals_for = half_time_home if is_home else half_time_away
                ht_goals_against = half_time_away if is_home else half_time_home
                
                # Maç verisini ekle
                match_data = {
                    'date': match.get('match_date', ''),
                    'league': match.get('league_name', ''),
                    'opponent': match.get('match_awayteam_name', '') if is_home else match.get('match_hometeam_name', ''),
                    'is_home': is_home,
                    'goals_scored': goals_for,
                    'goals_conceded': goals_against,
                    'ht_goals_scored': ht_goals_for,
                    'ht_goals_conceded': ht_goals_against,
                    'ht_result': ht_result,  # İlk yarı sonucu
                    'ft_result': ft_result,  # Tam maç sonucu
                    'ht_ft_code': iy_ms_code,  # İY/MS kodu
                    'result': ft_result  # Eski format ile uyumluluk için
                }
                recent_match_data.append(match_data)

            # Ortalama değerler hesapla - gerçekçi sınırlarla
            avg_goals_scored = min(2.2, goals_scored / len(recent_matches)) if recent_matches else 0
            avg_goals_conceded = min(2.2, goals_conceded / len(recent_matches)) if recent_matches else 0
            form_points = points / (len(recent_matches) * 3) if recent_matches else 0

            # Ağırlıklı ortalamalar hesapla - gerçekçi sınırlarla
            weighted_avg_goals_scored = min(2.2, weighted_goals_scored / total_weights) if total_weights > 0 else 0
            weighted_avg_goals_conceded = min(2.2, weighted_goals_conceded / total_weights) if total_weights > 0 else 0
            weighted_form_points = weighted_points / (total_weights * 3) if total_weights > 0 else 0

            # Ev ve deplasman için ortalamalar - gerçekçi sınırlarla
            avg_home_goals_scored = min(2.2, home_goals_scored / len(home_matches)) if home_matches else 0
            avg_home_goals_conceded = min(2.2, home_goals_conceded / len(home_matches)) if home_matches else 0
            avg_away_goals_scored = min(2.2, away_goals_scored / len(away_matches)) if away_matches else 0
            avg_away_goals_conceded = min(2.2, away_goals_conceded / len(away_matches)) if away_matches else 0

            # Ev ve deplasman için ağırlıklı ortalamalar - gerçekçi sınırlarla
            weighted_avg_home_goals_scored = min(2.2, weighted_home_goals_scored / total_home_weights) if total_home_weights > 0 else 0
            weighted_avg_home_goals_conceded = min(2.2, weighted_home_goals_conceded / total_home_weights) if total_home_weights > 0 else 0
            weighted_avg_away_goals_scored = min(2.2, weighted_away_goals_scored / total_away_weights) if total_away_weights > 0 else 0
            weighted_avg_away_goals_conceded = min(2.2, weighted_away_goals_conceded / total_away_weights) if total_away_weights > 0 else 0

            # Puanlar
            home_form_points = home_points / (len(home_matches) * 3) if home_matches else 0
            away_form_points = away_points / (len(away_matches) * 3) if away_matches else 0

            # Ağırlıklı form puanları
            weighted_home_form_points = weighted_home_points / (total_home_weights * 3) if total_home_weights > 0 else 0
            weighted_away_form_points = weighted_away_points / (total_away_weights * 3) if total_away_weights > 0 else 0

            # Bayesyen güncelleme için parametreler
            n_home = len(home_matches)
            n_away = len(away_matches)

            # Bayesyen posterior hesapla - gol atma
            lambda_home_scored = (self.alpha_ev_atma + home_goals_scored) / (self.beta_ev + n_home) if n_home > 0 else self.lig_ortalamasi_ev_gol
            lambda_away_scored = (self.alpha_deplasman_atma + away_goals_scored) / (self.beta_deplasman + n_away) if n_away > 0 else 1.0

            # Bayesyen posterior hesapla - gol yeme
            lambda_home_conceded = (self.alpha_deplasman_atma + home_goals_conceded) / (self.beta_deplasman + n_home) if n_home > 0 else 1.0
            lambda_away_conceded = (self.alpha_ev_atma + away_goals_conceded) / (self.beta_ev + n_away) if n_away > 0 else self.lig_ortalamasi_ev_gol

            # İlk yarı istatistiklerini analiz et
            ht_goals_scored_home = 0
            ht_goals_conceded_home = 0
            ht_goals_scored_away = 0
            ht_goals_conceded_away = 0
            
            # İlk yarı sonuçları analizi
            ht_home_wins = 0  # Ev sahibi olarak ilk yarı galibiyetleri
            ht_home_draws = 0  # Ev sahibi olarak ilk yarı beraberlikleri
            ht_home_losses = 0  # Ev sahibi olarak ilk yarı mağlubiyetleri
            ht_away_wins = 0  # Deplasman olarak ilk yarı galibiyetleri
            ht_away_draws = 0  # Deplasman olarak ilk yarı beraberlikleri
            ht_away_losses = 0  # Deplasman olarak ilk yarı mağlubiyetleri
            
            # İY/MS kombinasyonları sayaçları
            ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            # Ev/deplasman İY/MS kombinasyonları
            home_ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            away_ht_ft_counts = {
                '1/1': 0, '1/X': 0, '1/2': 0,
                'X/1': 0, 'X/X': 0, 'X/2': 0,
                '2/1': 0, '2/X': 0, '2/2': 0
            }
            
            # İlk yarı verilerini hesapla
            for match in home_matches:
                match_data = next((m for m in recent_match_data if m.get('is_home', False) and m.get('date') == match.get('match_date')), None)
                if match_data:
                    ht_goals_scored_home += match_data.get('ht_goals_scored', 0)
                    ht_goals_conceded_home += match_data.get('ht_goals_conceded', 0)
                    
                    # İlk yarı sonuçları
                    ht_result = match_data.get('ht_result', '')
                    if ht_result == 'W':
                        ht_home_wins += 1
                    elif ht_result == 'D':
                        ht_home_draws += 1
                    elif ht_result == 'L':
                        ht_home_losses += 1
                    
                    # İY/MS istatistiği ekle
                    ht_ft_code = match_data.get('ht_ft_code', '')
                    if ht_ft_code in ht_ft_counts:
                        ht_ft_counts[ht_ft_code] += 1
                        home_ht_ft_counts[ht_ft_code] += 1
            
            for match in away_matches:
                match_data = next((m for m in recent_match_data if not m.get('is_home', True) and m.get('date') == match.get('match_date')), None)
                if match_data:
                    ht_goals_scored_away += match_data.get('ht_goals_scored', 0)
                    ht_goals_conceded_away += match_data.get('ht_goals_conceded', 0)
                    
                    # İlk yarı sonuçları
                    ht_result = match_data.get('ht_result', '')
                    if ht_result == 'W':
                        ht_away_wins += 1
                    elif ht_result == 'D':
                        ht_away_draws += 1
                    elif ht_result == 'L':
                        ht_away_losses += 1
                    
                    # İY/MS istatistiği ekle
                    ht_ft_code = match_data.get('ht_ft_code', '')
                    if ht_ft_code in ht_ft_counts:
                        ht_ft_counts[ht_ft_code] += 1
                        away_ht_ft_counts[ht_ft_code] += 1
            
            # İlk yarı ortalama değerleri
            avg_ht_goals_scored_home = ht_goals_scored_home / len(home_matches) if home_matches else 0
            avg_ht_goals_conceded_home = ht_goals_conceded_home / len(home_matches) if home_matches else 0
            avg_ht_goals_scored_away = ht_goals_scored_away / len(away_matches) if away_matches else 0
            avg_ht_goals_conceded_away = ht_goals_conceded_away / len(away_matches) if away_matches else 0
            
            # İlk yarı / İkinci yarı eğilimlerini hesapla
            first_half_performance = 0.5  # Varsayılan değer
            second_half_performance = 0.5  # Varsayılan değer
            
            # İlk yarı ve ikinci yarı gol oranlarını hesapla
            all_first_half_goals = 0
            all_second_half_goals = 0
            
            for match in recent_match_data:
                all_first_half_goals += match.get('ht_goals_scored', 0)
                all_second_half_goals += match.get('goals_scored', 0) - match.get('ht_goals_scored', 0)
            
            # Toplam gollerin %'si olarak ifade et
            total_goals = all_first_half_goals + all_second_half_goals
            if total_goals > 0:
                first_half_performance = all_first_half_goals / total_goals
                second_half_performance = all_second_half_goals / total_goals
            
            # İY/MS eğilimlerini hesapla - en sık görülen 3 kombinasyon
            top_ht_ft = sorted(ht_ft_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'avg_goals_scored': avg_goals_scored,
                'avg_goals_conceded': avg_goals_conceded,
                'weighted_avg_goals_scored': weighted_avg_goals_scored,
                'weighted_avg_goals_conceded': weighted_avg_goals_conceded,
                'form_points': form_points,
                'weighted_form_points': weighted_form_points,
                'recent_matches': len(recent_matches),
                'home_matches': len(home_matches),
                'away_matches': len(away_matches),
                'home_performance': {
                    'avg_goals_scored': avg_home_goals_scored, 
                    'avg_goals_conceded': avg_home_goals_conceded,
                    'weighted_avg_goals_scored': weighted_avg_home_goals_scored,
                    'weighted_avg_goals_conceded': weighted_avg_home_goals_conceded,
                    'form_points': home_form_points,
                    'weighted_form_points': weighted_home_form_points,
                    'bayesian_goals_scored': lambda_home_scored,
                    'bayesian_goals_conceded': lambda_home_conceded,
                    # İlk yarı istatistikleri
                    'avg_ht_goals_scored': avg_ht_goals_scored_home,
                    'avg_ht_goals_conceded': avg_ht_goals_conceded_home,
                    'ht_wins': ht_home_wins,
                    'ht_draws': ht_home_draws,
                    'ht_losses': ht_home_losses,
                    'ht_ft_stats': home_ht_ft_counts
                },
                'away_performance': {
                    'avg_goals_scored': avg_away_goals_scored,
                    'avg_goals_conceded': avg_away_goals_conceded, 
                    'weighted_avg_goals_scored': weighted_avg_away_goals_scored,
                    'weighted_avg_goals_conceded': weighted_avg_away_goals_conceded,
                    'form_points': away_form_points,
                    'weighted_form_points': weighted_away_form_points,
                    'bayesian_goals_scored': lambda_away_scored,
                    'bayesian_goals_conceded': lambda_away_conceded,
                    # İlk yarı istatistikleri
                    'avg_ht_goals_scored': avg_ht_goals_scored_away,
                    'avg_ht_goals_conceded': avg_ht_goals_conceded_away,
                    'ht_wins': ht_away_wins,
                    'ht_draws': ht_away_draws,
                    'ht_losses': ht_away_losses,
                    'ht_ft_stats': away_ht_ft_counts
                },
                'recent_match_data': recent_match_data,
                'detailed_data': {
                    'last_5': recent_match_data[:5],
                    'last_10': recent_match_data[:10],
                    'last_15': recent_match_data[:15],
                    'all': recent_match_data
                },
                'bayesian': {
                    'home_lambda_scored': lambda_home_scored,
                    'home_lambda_conceded': lambda_home_conceded,
                    'away_lambda_scored': lambda_away_scored,
                    'away_lambda_conceded': lambda_away_conceded,
                    'n_home': n_home,
                    'n_away': n_away
                },
                # İlk yarı / İkinci yarı performans analizi
                'half_time_analysis': {
                    'first_half_performance': first_half_performance,
                    'second_half_performance': second_half_performance,
                    'ht_ft_trends': top_ht_ft,
                    'ht_ft_counts': ht_ft_counts,
                    'first_half_goals': all_first_half_goals,
                    'second_half_goals': all_second_half_goals
                }
            }

        except Exception as e:
            logger.error(f"Takım formu alınırken hata: {str(e)}")
            return None

    def _generate_fallback_form_data(self, team_id):
        """API'den veri alınamadığında kullanılan varsayılan form verisi"""
        logger.warning(f"Takım {team_id} için fallback form verisi oluşturuluyor")
        return {
            'avg_goals_scored': 1.3,
            'avg_goals_conceded': 1.2,
            'weighted_avg_goals_scored': 1.3,
            'weighted_avg_goals_conceded': 1.2,
            'form_points': 0.5,
            'weighted_form_points': 0.5,
            'recent_matches': 10,
            'home_matches': 5,
            'away_matches': 5,
            'home_performance': {
                'avg_goals_scored': 1.5, 
                'avg_goals_conceded': 1.1,
                'weighted_avg_goals_scored': 1.5,
                'weighted_avg_goals_conceded': 1.1,
                'form_points': 0.55,
                'weighted_form_points': 0.55,
                'bayesian_goals_scored': 1.5,
                'bayesian_goals_conceded': 1.1,
                'avg_ht_goals_scored': 0.7,
                'avg_ht_goals_conceded': 0.5,
                'ht_wins': 2,
                'ht_draws': 2,
                'ht_losses': 1,
                'ht_ft_pattern': {'1/1': 2, 'X/X': 1, '1/X': 1, 'X/1': 1}
            },
            'away_performance': {
                'avg_goals_scored': 1.1,
                'avg_goals_conceded': 1.3,
                'weighted_avg_goals_scored': 1.1,
                'weighted_avg_goals_conceded': 1.3,
                'form_points': 0.45,
                'weighted_form_points': 0.45,
                'bayesian_goals_scored': 1.1,
                'bayesian_goals_conceded': 1.3,
                'avg_ht_goals_scored': 0.5,
                'avg_ht_goals_conceded': 0.7,
                'ht_wins': 1,
                'ht_draws': 2,
                'ht_losses': 2,
                'ht_ft_pattern': {'2/2': 1, 'X/X': 2, 'X/2': 1, '2/X': 1}
            },
            'first_half_performance': 0.4,
            'second_half_performance': 0.6,
            'top_ht_ft_patterns': [('X/X', 3), ('1/1', 2), ('2/2', 1)],
            'recent_match_data': []
        }
    def predict_match(self, home_team_id, away_team_id, home_team_name, away_team_name, force_update=False, 
                   use_specialized_models=True, use_goal_trend_analysis=True):
        """Maç sonucunu tahmin et - Gelişmiş algoritma sıralaması ve tutarlılık kontrolü ile
        
        Algoritma sıralaması:
        1. Önbellekte varsa önbellekten getir veya önceki model değerlerini kullan
        2. Temel istatistik ve form analizlerini yap
        3. Monte Carlo simülasyonu ile ilk tahminleri oluştur
        4. Gelişmiş faktörleri (maç önemi, tarihi patern, vb) uygula
        5. Gol trendi analizini (ivme analizi) uygula
        6. Takım spesifik modelleri uygula
        7. Özelleştirilmiş modelleri (düşük/orta/yüksek skorlu) uygula
        8. KG VAR/YOK ve skor tutarlılığını kontrol et
        9. ÜST/ALT ve skor tutarlılığını kontrol et
        10. Genel sonuç tutarlılığını kontrol et
        
        Args:
            home_team_id: Ev sahibi takım ID'si
            away_team_id: Deplasman takımı ID'si
            home_team_name: Ev sahibi takım adı
            away_team_name: Deplasman takımı adı
            force_update: Önbellekteki tahmin varsa da yeniden hesapla
            use_specialized_models: Özelleştirilmiş modelleri (düşük/orta/yüksek skorlu) kullan
            use_goal_trend_analysis: Gol trend ivme analizini kullan
            
        Returns:
            dict: Tahmin sonuçları
        """
        # Kullanılacak modelleri belirle
        # HİBRİT KG VAR/YOK SİSTEMİ - STANDLONE SERVİS
        logger.info(f">>> HİBRİT KG VAR/YOK SİSTEMİ BAŞLATILIYOR <<<: {home_team_name} vs {away_team_name}")
        hybrid_kg_result = None
        try:
            from hybrid_kg_service import get_hybrid_kg_prediction
            hybrid_kg_result = get_hybrid_kg_prediction(str(home_team_id), str(away_team_id))
            if hybrid_kg_result:
                logger.info(f">>> HİBRİT SİSTEM BAŞARILI <<<: {hybrid_kg_result['prediction']} - %{hybrid_kg_result['probability']}")
                logger.info(f"Hibrit bileşenler - Poisson: %{hybrid_kg_result['components']['poisson']}, "
                           f"Logistic: %{hybrid_kg_result['components']['logistic']}, "
                           f"Historical: %{hybrid_kg_result['components']['historical']}")
            else:
                logger.error("Hibrit sistem None sonuç döndü")
        except Exception as e:
            logger.error(f">>> HİBRİT SİSTEM HATASI <<<: {e}")
            hybrid_kg_result = None

        global ADVANCED_MODELS_AVAILABLE, TEAM_SPECIFIC_MODELS_AVAILABLE, ENHANCED_MONTE_CARLO_AVAILABLE, SPECIALIZED_MODELS_AVAILABLE
        use_ensemble_models = False
        use_gbm_models = False
        use_enhanced_monte_carlo = ENHANCED_MONTE_CARLO_AVAILABLE and hasattr(self, 'enhanced_monte_carlo')
        use_bayesian_models = False
        use_team_specific_models = False
        use_specialized_models = False
        
        # 1. Gelişmiş tahmin algoritmaları (GBM, LSTM, Bayesian Networks)
        if ADVANCED_MODELS_AVAILABLE and hasattr(self, 'advanced_models'):
            logger.info(f"Gelişmiş makine öğrenmesi modelleri kullanılıyor: {home_team_name} vs {away_team_name}")
            use_gbm_models = True
            use_bayesian_models = True
        
        # 2. Ensemble Score predictor (mevcut tahmin modelimiz)
        try:
            from zip_and_ensemble_predictor import AdvancedScorePredictor
            advanced_predictor = AdvancedScorePredictor()
            use_ensemble_models = True
            logger.info("Gelişmiş ensemble tahmin modeli başarıyla yüklendi")
        except Exception as e:
            logger.warning(f"Ensemble tahmin modeli yüklenemedi: {e}")
            use_ensemble_models = False
            
        # 3. Takım-spesifik modeller (lig ve takıma özel parametreler)
        if 'TEAM_SPECIFIC_MODELS_AVAILABLE' in globals() and globals()['TEAM_SPECIFIC_MODELS_AVAILABLE'] and hasattr(self, 'team_specific_predictor'):
            logger.info(f"Takım-spesifik tahmin modelleri kullanılıyor: {home_team_name} vs {away_team_name}")
            use_team_specific_models = True
            
        # 4. Özelleştirilmiş modeller (düşük, orta ve yüksek skorlu maçlar için)
        if use_specialized_models and 'SPECIALIZED_MODELS_AVAILABLE' in globals() and globals()['SPECIALIZED_MODELS_AVAILABLE'] and hasattr(self, 'specialized_models'):
            logger.info(f"Özelleştirilmiş tahmin modelleri (düşük/orta/yüksek skorlu maçlar) kullanılıyor: {home_team_name} vs {away_team_name}")
        else:
            use_specialized_models = False
            if not hasattr(self, 'specialized_models'):
                logger.warning(f"Özelleştirilmiş tahmin modelleri sınıfı bulunamadı.")
            elif 'SPECIALIZED_MODELS_AVAILABLE' not in globals() or not globals()['SPECIALIZED_MODELS_AVAILABLE']:
                logger.warning(f"Özelleştirilmiş tahmin modelleri kullanılamıyor.")

        # Tahmin öncesi sinir ağlarını eğit
        logger.info(f"{home_team_name} vs {away_team_name} için sinir ağları eğitiliyor...")
        self.collect_training_data()

        # Önbelleği kontrol et
        cache_key = f"{home_team_id}_{away_team_id}"
        force_new_prediction = False

        if cache_key in self.predictions_cache and not force_update:
            prediction = self.predictions_cache[cache_key]
            # Tahmin 24 saatten eski değilse onu kullan
            cached_time = datetime.fromtimestamp(prediction.get('timestamp', 0))

            # Eski algoritma ile yapılan tahminleri kontrol et (neural_predictions yoksa eski versiyon)
            if 'predictions' in prediction and 'neural_predictions' not in prediction.get('predictions', {}):
                logger.info(f"Eski algoritma ile yapılmış tahmin bulundu: {home_team_name} vs {away_team_name}")
                force_new_prediction = True
            # Tahmin 24 saatten eski değilse ve güncel algoritma ile yapılmışsa onu kullan
            elif datetime.now() - cached_time < timedelta(hours=24):
                logger.info(f"Önbellekten tahmin kullanılıyor: {home_team_name} vs {away_team_name}")
                return prediction
            else:
                force_new_prediction = True
        elif force_update:
            logger.info(f"Zorunlu yeni tahmin yapılıyor: {home_team_name} vs {away_team_name}")
            force_new_prediction = True
        else:
            # Cache'de olmayan tahminler için
            logger.info(f"Önbellekte olmayan tahmin yapılıyor: {home_team_name} vs {away_team_name}")
            force_new_prediction = True

        # Takımların form verilerini al
        home_form = self.get_team_form(home_team_id)
        away_form = self.get_team_form(away_team_id)

        if not home_form or not away_form:
            logger.error(f"Form verileri alınamadı: {home_team_name} vs {away_team_name}")
            return None

        # Gelişmiş tahmin modellerini kullan - YENİ: iyileştirilmiş tutarlılık için daha fazla ağırlık ver
        advanced_prediction = None
        if use_ensemble_models:
            try:
                # Geliştirilmiş algoritma - daha tutarlı tahminler için
                advanced_prediction = advanced_predictor.predict_match(
                    home_form, 
                    away_form, 
                    self.predictions_cache,
                    model_weight=0.4,  # Yeni sistem ağırlığı %40, eski sistem %60 
                    simulations=10000  # Daha fazla simülasyon - daha doğru olasılıklar için
                )

                if advanced_prediction:
                    logger.info(f"Gelişmiş tutarlı tahmin modelleri başarıyla kullanıldı: {home_team_name} vs {away_team_name}")
                    # Gelişmiş tahmin sonuçlarını kullan
                    adv_home_goals = advanced_prediction['expected_goals']['home']
                    adv_away_goals = advanced_prediction['expected_goals']['away']
                    logger.info(f"Tutarlı tahmin modeli: Ev {adv_home_goals:.2f} - Deplasman {adv_away_goals:.2f}")
                    
                    # Eğer gelişmiş betting_predictions varsa, bunları da kullanacağız
                    adv_betting_predictions = advanced_prediction.get('betting_predictions', {})
                    if adv_betting_predictions:
                        logger.info(f"Gelişmiş bahis tahminleri mevcut: {list(adv_betting_predictions.keys())}")
            except Exception as e:
                logger.error(f"Gelişmiş tahmin modelleri hatası: {e}")
                advanced_prediction = None

        # Sinir ağı için veri hazırla
        home_features = self.prepare_data_for_neural_network(home_form, is_home=True)
        away_features = self.prepare_data_for_neural_network(away_form, is_home=False)

        # Sinir ağı modelleri kontrol et
        if self.model_home is None or self.model_away is None:
            self.load_or_create_models()

        # Monte Carlo simülasyonu yap (5000 maç simüle et)
        home_wins = 0
        away_wins = 0
        draws = 0
        both_teams_scored = 0
        over_2_5_goals = 0
        over_3_5_goals = 0
        simulations = 5000  # Daha fazla simülasyon

        # Ev sahibi avantajı faktörünü son maçlara göre dinamik olarak hesaplayalım
        # Son 5 ev sahibi maçını analiz et
        home_matches_as_home = [m for m in home_form.get('recent_match_data', []) if m.get('is_home', False)][:5]
        home_as_home_points = 0
        
        if home_matches_as_home:
            for match in home_matches_as_home:
                if match.get('result') == 'W':
                    home_as_home_points += 3
                elif match.get('result') == 'D':
                    home_as_home_points += 1
            
            # Ev sahibi puan performansına göre avantaj belirle - daha düşük avantaj katsayıları
            if home_as_home_points >= 10:  # Mükemmel ev performansı
                home_advantage = 1.15  # %15 avantaj
                logger.info(f"Güçlü ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            elif home_as_home_points >= 7:  # İyi ev performansı
                home_advantage = 1.08  # %8 avantaj
                logger.info(f"Normal ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            elif home_as_home_points >= 4:  # Orta ev performansı
                home_advantage = 1.03  # %3 avantaj
                logger.info(f"Minimal ev sahibi avantajı: Son 5 ev maçında {home_as_home_points} puan")
            else:  # Zayıf ev performansı
                home_advantage = 1.0  # Avantaj yok
                logger.info(f"Ev sahibi avantajı yok: Son 5 ev maçında sadece {home_as_home_points} puan")
        else:
            # Yeterli ev sahibi maç verisi yoksa standart avantaj uygula
            home_advantage = 1.05
            logger.info("Yeterli ev maçı verisi bulunamadı, standart ev avantajı uygulandı.")
            
        # Deplasman avantajını son maçlara göre dinamik olarak hesaplayalım
        # Son 5 deplasman maçını analiz et
        away_matches_as_away = [m for m in away_form.get('recent_match_data', []) if not m.get('is_home', True)][:5]
        away_as_away_points = 0
        
        if away_matches_as_away:
            for match in away_matches_as_away:
                if match.get('result') == 'W':
                    away_as_away_points += 3
                elif match.get('result') == 'D':
                    away_as_away_points += 1
            
            # Deplasman puan performansına göre avantaj belirle
            if away_as_away_points >= 7:  # 7-9 puan ve üzeri ise güçlü deplasman performansı
                away_advantage = 1.10  # İstenen güçlü deplasman avantajı
                logger.info(f"Deplasman avantajı uygulandı: Son 5 deplasman maçında {away_as_away_points} puan kazanılmış (güçlü)")
            else:  # 7 puan altında ise
                away_advantage = 1.00  # Deplasman avantajı uygulanmayacak
                logger.info(f"Deplasman avantajı uygulanmadı: Son 5 deplasman maçında sadece {away_as_away_points} puan kazanılmış (zayıf)")
        else:
            # Yeterli deplasman maç verisi yoksa standart değer kullan
            away_advantage = 1.00
            logger.info("Yeterli deplasman maçı verisi bulunamadı, standart deplasman avantajı uygulandı.")

        # Farklı dönemlere (son 3, 6, 9 maç) göre daha dengeli ağırlıklandırma
        # Form ağırlıkları - daha dengeli dağılım
        # Form ve güç dengesine göre ağırlıkları hesapla - son 5 maça önemli ağırlık ver ama uzun vadeyi de dikkate al 
        weight_last_5 = 2.5   # Son 5 maç - yüksek önem (son karşılaşmaların etkisini koruyarak)
        weight_last_10 = 1.5  # Son 6-10 arası maçlar - orta önem
        weight_last_21 = 1.0  # Son 11-21 arası maçlar - daha düşük ama anlamlı önem (takım gücünü belirler)

        # Takımların farklı dönemlerdeki form verilerini tutacak sözlükler
        home_form_periods = {}
        away_form_periods = {}

        # Sinir ağı tahminleri
        neural_home_goals = 0.0
        neural_away_goals = 0.0

        # Eğer hazır modeller varsa tahmin yap
        if self.model_home is not None and self.model_away is not None and home_features is not None and away_features is not None:
            try:
                # Veriyi normalize et
                scaled_home_features = self.scaler.fit_transform(home_features)
                scaled_away_features = self.scaler.transform(away_features)

                # Tahmin yap
                neural_home_goals = float(self.model_home.predict(scaled_home_features, verbose=0)[0][0])
                neural_away_goals = float(self.model_away.predict(scaled_away_features, verbose=0)[0][0])

                # Tahminleri pozitif değerlere sınırla
                neural_home_goals = max(0.0, neural_home_goals)
                neural_away_goals = max(0.0, neural_away_goals)

                logger.info(f"Sinir ağı tahminleri: Ev {neural_home_goals:.2f} - Deplasman {neural_away_goals:.2f}")
            except Exception as e:
                logger.error(f"Sinir ağı tahmin hatası: {str(e)}")
                # Sinir ağı tahmin hata verirse Bayesyen tahminler kullanılacak
                neural_home_goals = 0.0
                neural_away_goals = 0.0

        # Ev sahibi takımın farklı dönemlerdeki performanslarını hesapla
        home_match_data = home_form.get('recent_match_data', [])

        # Son 3 maç
        if home_form.get('recent_matches', 0) >= 3:
            last_3_home_goals = 0
            last_3_home_conceded = 0
            last_3_home_points = 0

            for i in range(min(3, len(home_match_data))):
                last_3_home_goals += home_match_data[i].get('goals_scored', 0)
                last_3_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_3_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_3_home_points += 1

            home_form_periods['last_3'] = {
                'avg_goals': last_3_home_goals / 3,
                'avg_conceded': last_3_home_conceded / 3,
                'form_points': last_3_home_points / 9  # 3 maçta maksimum 9 puan alınabilir
            }
        else:
            home_form_periods['last_3'] = {
                'avg_goals': home_form['avg_goals_scored'],
                'avg_conceded': home_form['avg_goals_conceded'],
                'form_points': home_form['form_points']
            }

        # Son 6 maç
        if home_form.get('recent_matches', 0) >= 6:
            last_6_home_goals = 0
            last_6_home_conceded = 0
            last_6_home_points = 0

            for i in range(min(6, len(home_match_data))):
                last_6_home_goals += home_match_data[i].get('goals_scored', 0)
                last_6_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_6_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_6_home_points += 1

            home_form_periods['last_6'] = {
                'avg_goals': last_6_home_goals / 6,
                'avg_conceded': last_6_home_conceded / 6,
                'form_points': last_6_home_points / 18  # 6 maçta maksimum 18 puan alınabilir
            }
        else:
            home_form_periods['last_6'] = home_form_periods['last_3']

        # Son 9 maç
        if home_form.get('recent_matches', 0) >= 9:
            last_9_home_goals = 0
            last_9_home_conceded = 0
            last_9_home_points = 0

            for i in range(min(9, len(home_match_data))):
                last_9_home_goals += home_match_data[i].get('goals_scored', 0)
                last_9_home_conceded += home_match_data[i].get('goals_conceded', 0)
                if home_match_data[i].get('result') == 'W':
                    last_9_home_points += 3
                elif home_match_data[i].get('result') == 'D':
                    last_9_home_points += 1

            home_form_periods['last_9'] = {
                'avg_goals': last_9_home_goals / 9,
                'avg_conceded': last_9_home_conceded / 9,
                'form_points': last_9_home_points / 27  # 9 maçta maksimum 27 puan alınabilir
            }
        else:
            home_form_periods['last_9'] = home_form_periods['last_6']

        # Deplasman takımının farklı dönemlerdeki performanslarını hesapla
        away_match_data = away_form.get('recent_match_data', [])

        # Son 3 maç
        if away_form.get('recent_matches', 0) >= 3:
            last_3_away_goals = 0
            last_3_away_conceded = 0
            last_3_away_points = 0

            for i in range(min(3, len(away_match_data))):
                last_3_away_goals += away_match_data[i].get('goals_scored', 0)
                last_3_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_3_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_3_away_points += 1

            away_form_periods['last_3'] = {
                'avg_goals': last_3_away_goals / 3,
                'avg_conceded': last_3_away_conceded / 3,
                'form_points': last_3_away_points / 9  # 3 maçta maksimum 9 puan alınabilir
            }
        else:
            away_form_periods['last_3'] = {
                'avg_goals': away_form['avg_goals_scored'],
                'avg_conceded': away_form['avg_goals_conceded'],
                'form_points': away_form['form_points']
            }

        # Son 6 maç
        if away_form.get('recent_matches', 0) >= 6:
            last_6_away_goals = 0
            last_6_away_conceded = 0
            last_6_away_points = 0

            for i in range(min(6, len(away_match_data))):
                last_6_away_goals += away_match_data[i].get('goals_scored', 0)
                last_6_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_6_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_6_away_points += 1

            away_form_periods['last_6'] = {
                'avg_goals': last_6_away_goals / 6,
                'avg_conceded': last_6_away_conceded / 6,
                'form_points': last_6_away_points / 18  # 6 maçta maksimum 18 puan alınabilir
            }
        else:
            away_form_periods['last_6'] = away_form_periods['last_3']

        # Son 9 maç
        if away_form.get('recent_matches', 0) >= 9:
            last_9_away_goals = 0
            last_9_away_conceded = 0
            last_9_away_points = 0

            for i in range(min(9, len(away_match_data))):
                last_9_away_goals += away_match_data[i].get('goals_scored', 0)
                last_9_away_conceded += away_match_data[i].get('goals_conceded', 0)
                if away_match_data[i].get('result') == 'W':
                    last_9_away_points += 3
                elif away_match_data[i].get('result') == 'D':
                    last_9_away_points += 1

            away_form_periods['last_9'] = {
                'avg_goals': last_9_away_goals / 9,
                'avg_conceded': last_9_away_conceded / 9,
                'form_points': last_9_away_points / 27  # 9 maçta maksimum 27 puan alınabilir
            }
        else:
            away_form_periods['last_9'] = away_form_periods['last_6']

        # Ağırlıklı beklenen gol hesaplamaları (son 3-6-9 maçın farklı ağırlıklarıyla)
        # Toplam ağırlık normalizasyonu için kullanılacak değer
        total_weight = weight_last_5 + weight_last_10 + weight_last_21

        # Ev sahibi takımın ağırlıklı beklenen gol sayısı - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_goals = (
            home_form_periods['last_3']['avg_goals'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['avg_goals'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['avg_goals'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Ev sahibi takımın ağırlıklı form puanı - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_form_points = (
            home_form_periods['last_3']['form_points'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['form_points'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['form_points'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Deplasman takımının ağırlıklı beklenen gol sayısı - son 5, son 10 ve son 21 maç verileri ile
        weighted_away_goals = (
            away_form_periods['last_3']['avg_goals'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['avg_goals'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['avg_goals'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Deplasman takımının ağırlıklı form puanı - son 5, son 10 ve son 21 maç verileri ile
        weighted_away_form_points = (
            away_form_periods['last_3']['form_points'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['form_points'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['form_points'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Savunma performansını da hesaba katarak beklenen gol hesaplaması - son 5, son 10 ve son 21 maç verileri ile
        weighted_home_conceded = (
            home_form_periods['last_3']['avg_conceded'] * weight_last_5 +  # Son 5 maç verileri
            home_form_periods['last_6']['avg_conceded'] * weight_last_10 + # Son 6-10 arası
            home_form_periods['last_9']['avg_conceded'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        weighted_away_conceded = (
            away_form_periods['last_3']['avg_conceded'] * weight_last_5 +  # Son 5 maç verileri
            away_form_periods['last_6']['avg_conceded'] * weight_last_10 + # Son 6-10 arası
            away_form_periods['last_9']['avg_conceded'] * weight_last_21   # Son 11-21 arası
        ) / total_weight

        # Bayesyen ve ağırlıklı yaklaşımların birleşimi ile daha gerçekçi beklenen gol hesaplama
        # 1. Bayesyen güncelleme ile elde edilen değerler
        bayesian_home_attack = home_form.get('bayesian', {}).get('home_lambda_scored', self.lig_ortalamasi_ev_gol)
        bayesian_away_defense = away_form.get('bayesian', {}).get('away_lambda_conceded', self.lig_ortalamasi_ev_gol)
        bayesian_away_attack = away_form.get('bayesian', {}).get('away_lambda_scored', 1.0)
        bayesian_home_defense = home_form.get('bayesian', {}).get('home_lambda_conceded', 1.0)

        # 2. Ağırlıklı ortalama ile hesaplanan değerler (mevcut kod)
        weighted_home_attack = home_form.get('home_performance', {}).get('weighted_avg_goals_scored', weighted_home_goals)
        weighted_away_defense = away_form.get('away_performance', {}).get('weighted_avg_goals_conceded', weighted_away_conceded)
        weighted_away_attack = away_form.get('away_performance', {}).get('weighted_avg_goals_scored', weighted_away_goals)
        weighted_home_defense = home_form.get('home_performance', {}).get('weighted_avg_goals_conceded', weighted_home_conceded)

        # 3. İki yaklaşımı birleştir (0.6 Bayesyen + 0.4 Ağırlıklı ortalama)
        combined_home_attack = bayesian_home_attack * 0.6 + weighted_home_attack * 0.4
        combined_away_defense = bayesian_away_defense * 0.6 + weighted_away_defense * 0.4
        combined_away_attack = bayesian_away_attack * 0.6 + weighted_away_attack * 0.4
        combined_home_defense = bayesian_home_defense * 0.6 + weighted_home_defense * 0.4

        # 4. Saldırı ve savunma güçlerini birleştirerek beklenen gol hesapla
        # Rakiplerin savunma zafiyetlerini değerlendiren faktörler
        # Deplasman takımının savunma zafiyeti faktörü
        away_defense_weakness = 1.0
        if 'away_performance' in away_form and 'weighted_avg_goals_conceded' in away_form['away_performance']:
            # Deplasman takımının ortalamadan ne kadar fazla gol yediğini hesapla
            away_defense_weakness = away_form['away_performance']['weighted_avg_goals_conceded'] / 1.3
            
        # Ev sahibi takımın savunma zafiyeti faktörü
        home_defense_weakness = 1.0
        if 'home_performance' in home_form and 'weighted_avg_goals_conceded' in home_form['home_performance']:
            # Ev sahibi takımının ortalamadan ne kadar fazla gol yediğini hesapla
            home_defense_weakness = home_form['home_performance']['weighted_avg_goals_conceded'] / 1.1
            
        # Değerleri normalizasyon için limitleyelim - çok aşırı değerleri engelle
        away_defense_weakness = min(1.8, max(0.9, away_defense_weakness))
        home_defense_weakness = min(1.8, max(0.9, home_defense_weakness))
        
        logger.info(f"Rakip savunma zafiyet faktörleri - Deplasman: {away_defense_weakness:.2f}, Ev: {home_defense_weakness:.2f}")
        
        # Ev sahibi takımın gol beklentisinde ev avantajını ve rakip takımın savunma zafiyetini kullan
        # Hücum gücüne daha fazla ağırlık ver (0.7 -> 0.85) ve savunmaya daha az (0.3 -> 0.15)
        # Bu değişiklik beklenen gol değerlerinin skor tahminlerine etkisini artıracak
        expected_home_goals = (combined_home_attack * 0.85 + combined_away_defense * 0.15) * home_advantage * away_defense_weakness
        # Deplasman takımın gol beklentisinde deplasman avantajını ve rakip takımın savunma zafiyetini kullan
        # Hücum gücüne daha fazla ağırlık ver (0.7 -> 0.85) ve savunmaya daha az (0.3 -> 0.15)
        expected_away_goals = (combined_away_attack * 0.85 + combined_home_defense * 0.15) * away_advantage * home_defense_weakness
        
        logger.info(f"Ham gol beklentileri: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
        # Sinir ağı tahminlerini entegre et (eğer varsa)
        if neural_home_goals > 0 and neural_away_goals > 0:
            # Kombine tahmin: %40 sinir ağı + %60 Bayesyen-Ağırlıklı
            expected_home_goals = expected_home_goals * 0.6 + neural_home_goals * 0.4
            expected_away_goals = expected_away_goals * 0.6 + neural_away_goals * 0.4
            logger.info(f"Sinir ağı entegreli tahminler: Ev {expected_home_goals:.2f} - Deplasman {expected_away_goals:.2f}")
            
        # Özelleştirilmiş modelleri kullan (düşük/orta/yüksek skorlu maçlar için)
        specialized_params = None
        if use_specialized_models and hasattr(self, 'specialized_models'):
            try:
                # Maç kategorisini belirle ve özelleştirilmiş model parametrelerini al
                specialized_prediction = self.specialized_models.predict(
                    home_form, away_form, expected_home_goals, expected_away_goals
                )
                
                if specialized_prediction:
                    # Kategori bilgisini ve model parametrelerini al
                    category = specialized_prediction.get('category', 'normal')
                    specialized_params = specialized_prediction.get('parameters', {})
                    
                    logger.info(f"Maç kategorisi: {category.upper()} (düşük/orta/yüksek skorlu)")
                    
                    # Özelleştirilmiş model parametrelerini log'a yazdır
                    draw_boost = specialized_params.get('draw_boost', 1.0)
                    max_score = specialized_params.get('max_score', 3)
                    poisson_correction = specialized_params.get('poisson_correction', 0.85)
                    
                    logger.info(f"Özelleştirilmiş model parametreleri: "
                                f"beraberlik_çarpanı={draw_boost:.2f}, "
                                f"maksimum_skor={max_score}, "
                                f"poisson_düzeltme={poisson_correction:.2f}")
                    
                    # Özel skor düzeltmeleri ile ilgili bilgileri log'a yazdır
                    score_correction = specialized_params.get('score_correction', {})
                    if score_correction:
                        top_corrections = sorted(score_correction.items(), key=lambda x: x[1], reverse=True)[:3]
                        corrections_str = ", ".join([f"{score}:{factor:.2f}" for score, factor in top_corrections])
                        logger.info(f"En yüksek skor düzeltme çarpanları: {corrections_str}")
            except Exception as e:
                logger.error(f"Özelleştirilmiş model kullanılırken hata: {str(e)}")
                specialized_params = None

        # Global ortalama değerler (lig ortalamaları) - gerçekçi ortalama değerler kullanma
        global_avg_home_goals = 1.6  # Ev sahibi takımlar için genel ortalama gol (düşürüldü)
        global_avg_away_goals = 1.3  # Deplasman takımları için genel ortalama gol (düşürüldü)

        # Mean Reversion uygula - daha gerçekçi gol beklentileri için parametreleri dengele
        # Son form performansına daha fazla ağırlık ver, global ortalamaya daha az
        phi_home = 0.30  # %30 ağırlık global ortalamaya, %70 ağırlık ev sahibi takım performansına
        phi_away = 0.20  # %20 ağırlık global ortalamaya, %80 ağırlık deplasman takım performansına

        # Farklılaştırılmış mean reversion uygula
        expected_home_goals = (1 - phi_home) * expected_home_goals + phi_home * global_avg_home_goals
        expected_away_goals = (1 - phi_away) * expected_away_goals + phi_away * global_avg_away_goals

        # Form farkını daha hassas bir şekilde hesaplama 
        # Katsayıyı azalttık, böylece yüksek form farkı sonuçları abartmayacak
        weighted_home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', weighted_home_form_points)
        weighted_away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', weighted_away_form_points)

        form_diff_home = max(-0.15, min(0.15, 0.05 * (weighted_home_form_points - weighted_away_form_points)))
        form_diff_away = max(-0.15, min(0.15, 0.05 * (weighted_away_form_points - weighted_home_form_points)))

        expected_home_goals = expected_home_goals * (1 + form_diff_home)
        expected_away_goals = expected_away_goals * (1 + form_diff_away)

        # Minimum değerler daha gerçekçi olarak ayarlanıyor, daha düşük minimum değerler
        expected_home_goals = max(0.8, expected_home_goals)
        expected_away_goals = max(0.7, expected_away_goals)

        # Form faktörlerini son 5 maç gol performansına daha fazla ağırlık vererek hesapla
        # Son 5 maçtaki ortalama gol sayılarını kullan
        recent_home_goals_avg = sum(match.get('goals_scored', 0) for match in home_match_data[:5]) / 5 if len(home_match_data) >= 5 else weighted_home_goals
        recent_away_goals_avg = sum(match.get('goals_scored', 0) for match in away_match_data[:5]) / 5 if len(away_match_data) >= 5 else weighted_away_goals
        
        # Son 5 maçtaki gol performansını form faktörüne daha fazla yansıt
        home_recent_factor = recent_home_goals_avg / max(1.0, self.lig_ortalamasi_ev_gol)
        away_recent_factor = recent_away_goals_avg / max(1.0, 1.0)
        
        # Form faktörlerini hesapla - son 5 maç performansını %60 ağırlıkla dahil et
        home_form_factor = min(1.5, (0.4 * (0.7 + weighted_home_form_points * 0.6) + 0.6 * home_recent_factor) * min(1.05, home_advantage))
        away_form_factor = min(1.5, (0.4 * (0.8 + weighted_away_form_points * 0.9) + 0.6 * away_recent_factor))
        
        logger.info(f"Son 5 maç analizi: Ev {recent_home_goals_avg:.2f} gol/maç, Deplasman {recent_away_goals_avg:.2f} gol/maç")
        logger.info(f"Form faktörleri: Ev {home_form_factor:.2f}, Deplasman {away_form_factor:.2f}")
        
        # Takım-spesifik ayarlamaları uygula (eğer mevcut ise)
        if use_team_specific_models:
            logger.info(f"Takım-spesifik modeller uygulanıyor: {home_team_name} vs {away_team_name}")
            original_home_goals = expected_home_goals
            original_away_goals = expected_away_goals
            expected_home_goals, expected_away_goals = self.apply_team_specific_adjustments(
                home_team_id, away_team_id, 
                home_team_name, away_team_name, 
                expected_home_goals, expected_away_goals,
                home_form, away_form,
                use_goal_trend_analysis
            )
            logger.info(f"Takım-spesifik ayarlamalar sonrası: Ev {original_home_goals:.2f}->{expected_home_goals:.2f}, " 
                       f"Deplasman {original_away_goals:.2f}->{expected_away_goals:.2f}")
        else:
            # Geleneksel büyük takım ayarlamaları (geriye dönük uyumluluk için)
            expected_home_goals, expected_away_goals = self.adjust_prediction_for_big_teams(
                home_team_name, away_team_name, expected_home_goals, expected_away_goals
            )

        # Gol dağılımları
        all_home_goals = []
        all_away_goals = []

        # Takımların gol atma olasılıklarını hesapla - ayarlanmış formül
        p_home_scores = 1 - np.exp(-(expected_home_goals * 1.0 * home_form_factor))
        p_away_scores = 1 - np.exp(-(expected_away_goals * 1.05 * away_form_factor))
        
        # Gol olasılıklarını logla
        logger.info(f"Gol atma olasılıkları: Ev={p_home_scores:.2f}, Deplasman={p_away_scores:.2f}")

        # Gelişmiş tahmin faktörlerini uygula
        global ENHANCED_FACTORS_AVAILABLE
        if 'ENHANCED_FACTORS_AVAILABLE' in globals() and globals()['ENHANCED_FACTORS_AVAILABLE'] and hasattr(self, 'enhanced_factors'):
            try:
                logger.info(f"Gelişmiş tahmin faktörleri uygulanıyor: {home_team_name} vs {away_team_name}")
                
                # Gelişmiş faktörlerden gelen ayarlamaları al
                enhanced_factors = self.enhanced_factors.get_enhanced_prediction_factors(
                    home_team_id, away_team_id, home_form, away_form
                )
                
                # Orijinal değerleri sakla
                original_home_goals = expected_home_goals
                original_away_goals = expected_away_goals
                
                # Gelişmiş faktörlere göre gol beklentilerini ayarla
                expected_home_goals, expected_away_goals = self.enhanced_factors.adjust_score_prediction(
                    expected_home_goals, expected_away_goals, enhanced_factors
                )
                
                logger.info(f"Gelişmiş faktör ayarlamaları sonrası: Ev {original_home_goals:.2f}->{expected_home_goals:.2f}, " 
                           f"Deplasman {original_away_goals:.2f}->{expected_away_goals:.2f}")
                
                # Gelişmiş faktörlerin gerekçelerini log'la
                if 'match_importance' in enhanced_factors:
                    logger.info(f"Maç önemi faktörü: {enhanced_factors['match_importance']['description']}")
                
                if 'historical_pattern' in enhanced_factors:
                    logger.info(f"Tarihsel eşleşme analizi: {enhanced_factors['historical_pattern']['description']}")
                
                if 'momentum' in enhanced_factors:
                    logger.info(f"Momentum analizi: Ev {enhanced_factors['momentum']['home_momentum']:.2f}, " 
                               f"Deplasman {enhanced_factors['momentum']['away_momentum']:.2f}")
                
            except Exception as e:
                logger.error(f"Gelişmiş tahmin faktörleri uygulanırken hata: {str(e)}")
                logger.warning("Gelişmiş faktörler uygulanamadı, standart tahmin değerleri kullanılacak")
        
        # Takım ID'lerine göre özel ayarlamalar uygula 
        # (Özellikle düşük skorlu maçlar ve spesifik takım işlemleri için)
        try:
            # Düşük skor eğilimli takımları kontrol et
            low_scoring_teams = [262, 530, 492, 642]  # Örnek Atletico Madrid, Getafe, Bursa, vs. gibi
            defensive_teams = [165, 798, 939, 250]    # Örnek savunmacı takımlar
            high_scoring_teams = [157, 173, 496, 533] # Örnek yüksek skor eğilimli takımlar
            
            is_home_low_scoring = int(home_team_id) in low_scoring_teams or int(home_team_id) in defensive_teams
            is_away_low_scoring = int(away_team_id) in low_scoring_teams or int(away_team_id) in defensive_teams
            is_home_high_scoring = int(home_team_id) in high_scoring_teams
            is_away_high_scoring = int(away_team_id) in high_scoring_teams
            
            # Her ikisi de düşük skor eğilimli ise
            if is_home_low_scoring and is_away_low_scoring:
                logger.info("Her iki takım da düşük skorlu - skor beklentileri azaltılıyor")
                expected_home_goals *= 0.85  # %15 azalt
                expected_away_goals *= 0.85  # %15 azalt
            
            # Hem düşük hem yüksek varsa, yüksek lehine ayarla
            elif (is_home_low_scoring and is_away_high_scoring):
                logger.info("Ev düşük skor eğilimli, deplasman yüksek - deplasman lehine ayarlandı")
                expected_home_goals *= 0.9   # %10 azalt
                expected_away_goals *= 1.15  # %15 artır
            
            # Yüksek skor lehine ayarla
            elif (is_home_high_scoring and is_away_low_scoring):
                logger.info("Ev yüksek skor eğilimli, deplasman düşük - ev lehine ayarlandı")
                expected_home_goals *= 1.15  # %15 artır
                expected_away_goals *= 0.9   # %10 azalt
            
            # İki yüksek skor takımı
            elif is_home_high_scoring and is_away_high_scoring:
                logger.info("Her iki takım da yüksek skorlu - gol beklentileri artırılıyor")
                expected_home_goals *= 1.2  # %20 artır
                expected_away_goals *= 1.2  # %20 artır
                
            logger.info(f"Takım ID tabanlı skor ayarlamaları sonrası: Ev={expected_home_goals:.2f}, Deplasman={expected_away_goals:.2f}")
        except Exception as e:
            logger.warning(f"Takım ID'sine göre ayarlama yapılırken hata: {str(e)}")
        
        # Beklenen toplam gol - daha dengeli toplam gol hesaplaması
        expected_total_goals = expected_home_goals * home_form_factor * 1.0 + expected_away_goals * away_form_factor * 1.05

        # Beklenen gol sayıları (initial estimations) - SORUN DÜZELTİLDİ
        # Minimum gol beklentilerini daha yüksek tutarak KG YOK problemini önle
        avg_home_goals = max(0.8, expected_home_goals * (home_form_factor * 1.0))
        avg_away_goals = max(0.8, expected_away_goals * (away_form_factor * 1.05))
        
        # Toplam gol beklentisini logla
        logger.info(f"Toplam gol beklentisi: {expected_total_goals:.2f}, Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")

        # Monte Carlo simülasyonu için ek değişkenler
        exact_scores = {}  # Kesin skor tahminleri için
        half_time_results = {"HOME_WIN": 0, "DRAW": 0, "AWAY_WIN": 0}  # İlk yarı sonuçları
        full_time_results = {"HOME_WIN": 0, "DRAW": 0, "AWAY_WIN": 0}  # Maç sonu sonuçları
        half_time_full_time = {}  # İlk yarı/maç sonu kombinasyonları
        first_goal_home = 0  # İlk golü ev sahibi takımın atma sayısı
        first_goal_away = 0  # İlk golü deplasman takımının atma sayısı
        no_goal = 0  # Golsüz maç sayısı

        # Kart ve korner tahminleri için
        cards_under_3_5 = 0
        cards_over_3_5 = 0
        corners_under_9_5 = 0
        corners_over_9_5 = 0

        # Gol zamanlaması için
        first_goal_timing = {
            "1-15": 0, "16-30": 0, "31-45": 0, 
            "46-60": 0, "61-75": 0, "76-90": 0, "No Goal": 0
        }

        # Monte Carlo simülasyonu
        for _ in range(simulations):
            # Negatif binomial dağılımını yaklaşık olarak simüle et
            # Poisson dağılımından daha fazla varyasyon gösterir ve gerçek gol dağılımlarını daha iyi temsil eder

            # Negatif binomial parametreleri hesapla
            # Poisson'a göre daha fazla varyasyona izin verir, özellikle yüksek skorlarda
            # r (başarı sayısı) ve p (başarı olasılığı) parametreleri ile tanımlanır

            # Düşük skorlu maçlar için özel işleme - SORUN DÜZELTİLDİ
            # Sadece çok ekstrem durumlar için düşük skorlu olarak işaretle
            low_scoring_match = (avg_home_goals < 0.4 and avg_away_goals < 0.4) or \
                               (avg_home_goals < 0.5 and avg_away_goals < 0.5 and 
                                home_form and away_form and
                                home_form.get('weighted_form_points', 0) < 0.2 and 
                                away_form.get('weighted_form_points', 0) < 0.2)
            
            if low_scoring_match:
                logger.info(f"Düşük skorlu maç tespit edildi: ev={avg_home_goals:.2f}, deplasman={avg_away_goals:.2f}")
                # Düşük skorlu maçlarda standart sapmayı azalt - daha tutarlı olasılıklar için
                home_std_dev = np.sqrt(avg_home_goals * 0.9) if avg_home_goals > 0 else 0.15
                away_std_dev = np.sqrt(avg_away_goals * 0.9) if avg_away_goals > 0 else 0.15
                
                # Düşük skorlu maçlarda, 0 gol olasılığını artırmak için daha düşük r değeri kullan
                home_r = max(0.8, avg_home_goals / 0.25) if avg_home_goals > 0 else 0.8
                away_r = max(0.8, avg_away_goals / 0.25) if avg_away_goals > 0 else 0.8
            else:
                # Normal maçlar için standart sapma hesaplaması
                home_std_dev = np.sqrt(avg_home_goals * 1.2) if avg_home_goals > 0 else 0.2
                away_std_dev = np.sqrt(avg_away_goals * 1.2) if avg_away_goals > 0 else 0.2
                
                # Normal maçlar için r parametresi hesaplaması
                home_r = max(1.0, avg_home_goals / 0.2) if avg_home_goals > 0 else 1.0
                away_r = max(1.0, avg_away_goals / 0.2) if avg_away_goals > 0 else 1.0

            home_p = home_r / (home_r + avg_home_goals)
            away_p = away_r / (away_r + avg_away_goals)

            # ZEHİRLİ SAVUNMA ANALİZİ: Takımların savunma zayıflıklarını belirle
            # Son maçlardaki gol yeme oranlarına göre savunma zayıflık faktörlerini hesapla
            home_defense_weakness = 1.0  # Varsayılan değer (1.0 = normal savunma)
            away_defense_weakness = 1.0  # Varsayılan değer (1.0 = normal savunma)
            
            # Son maçlardaki savunma performansını analiz et
            home_matches_data = home_form.get('recent_match_data', [])[:8]  # Son 8 maç
            away_matches_data = away_form.get('recent_match_data', [])[:8]  # Son 8 maç
            
            if home_matches_data:
                home_conceded_total = sum(match.get('goals_conceded', 0) for match in home_matches_data)
                home_match_count = len(home_matches_data)
                home_conceded_avg = home_conceded_total / home_match_count if home_match_count > 0 else 1.0
                
                # 1.8+ gol yiyen takım zayıf savunmalı kabul edilir
                if home_conceded_avg >= 1.8:
                    home_defense_weakness = 1.0 + min(0.2, (home_conceded_avg - 1.5) * 0.15)  # En fazla 1.2 kat zayıflık
                    logger.info(f"Ev sahibi takım zayıf savunma tespiti: {home_conceded_avg:.2f} gol/maç, savunma zayıflık faktörü: {home_defense_weakness:.2f}")
            
            if away_matches_data:
                away_conceded_total = sum(match.get('goals_conceded', 0) for match in away_matches_data)
                away_match_count = len(away_matches_data)
                away_conceded_avg = away_conceded_total / away_match_count if away_match_count > 0 else 1.0
                
                # 1.8+ gol yiyen takım zayıf savunmalı kabul edilir, deplasmanın gol yemesi daha olası
                if away_conceded_avg >= 1.8:
                    away_defense_weakness = 1.0 + min(0.25, (away_conceded_avg - 1.5) * 0.2)  # En fazla 1.25 kat zayıflık
                    logger.info(f"Deplasman takımı zayıf savunma tespiti: {away_conceded_avg:.2f} gol/maç, savunma zayıflık faktörü: {away_defense_weakness:.2f}")
            
            # Defense weakness calculation removed to prevent infinite feedback loop
            # Using original goal expectations without recursive multipliers
            
            # ORIGINAL GOAL EXPECTATIONS - DİNAMİK KG VAR SİSTEMİ İÇİN
            # Bu değerler herhangi bir ayarlama öncesi temel beklentilerdir
            self.original_home_goals = avg_home_goals  # Dinamik KG VAR hesabında kullanılacak
            self.original_away_goals = avg_away_goals  # Dinamik KG VAR hesabında kullanılacak
            
            # Gol beklentileri doğal değerlerinde kullanılacak (zorla sınırlandırma kaldırıldı)
            
            # Defense weakness feedback loop removed - keeping original goal expectations stable
                
            # DUPLICATE DEFENSE WEAKNESS CALCULATION REMOVED - this was causing infinite feedback loop
            
            # Farklı dağılımları daha dengeli kullan
            # Daha fazla çeşitlilik için random_selector ile dağılım seç
            random_selector = np.random.random()

            # Ev sahibi skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Beklenen gol değerine göre maksimum makul skor sınırları belirle
            # Bu, Monte Carlo simülasyonunda aşırı değerlerin oluşmasını önler
            max_home_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Toplam beklenen gol sayısını hesapla
            all_goals_expected = avg_home_goals + avg_away_goals
            
            # Eğer özelleştirilmiş model parametreleri varsa, onları kullan
            if specialized_params:
                # Özelleştirilmiş modelden maksimum skor sınırını al
                max_home_score = specialized_params.get('max_score', 3)
                logger.debug(f"Özelleştirilmiş model maksimum ev sahibi skoru: {max_home_score}")
            else:
                # Standart maksimum skor hesaplama - düşük skorlu maçlar için ayarlanmış
                if avg_home_goals < 0.6:
                    # Çok düşük gol beklentisi (0.6'dan az) - daha yüksek oranda 0 gol
                    max_home_score = 1  # 0.6'dan düşük beklenen gol için maksimum 1 gol
                elif avg_home_goals < 1.0:
                    max_home_score = 1  # 0.6-1.0 arası beklenen gol için maksimum 1 gol
                elif avg_home_goals < 1.8: 
                    max_home_score = 2  # 1.0-1.8 arası beklenen gol için maksimum 2 gol
                elif avg_home_goals < 2.5:
                    max_home_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
                else:
                    max_home_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # %95 şans ile sınırlı Poisson dağılımı
            if random_selector < 0.95:
                # Gol beklentisi 0.3'ten az ise gol olasılığını biraz artır - minimum skor ihtimali için
                if avg_home_goals < 0.3:
                    raw_score = np.random.poisson(max(0.3, avg_home_goals))
                else:
                    raw_score = np.random.poisson(avg_home_goals)
                
                # Sonucu makul sınırlar içinde tut
                home_score = min(raw_score, max_home_score)
            else:
                # Çok nadir durumlarda (%5) hafif varyasyon için negatif binomial dağılımı kullan
                try:
                    # Varyasyonu daha da azalt - daha tutarlı sonuçlar için
                    home_std_dev = np.sqrt(avg_home_goals)  # Daha düşük varyasyon
                    home_r = max(1.0, avg_home_goals / 0.1)  # Daha düşük dispersiyon
                    home_p = home_r / (home_r + avg_home_goals)
                    
                    raw_score = np.random.negative_binomial(home_r, home_p)
                    # Makul sınırlar içinde tut
                    home_score = min(raw_score, max_home_score)
                except ValueError:
                    # Hata durumunda Poisson'a geri dön
                    raw_score = np.random.poisson(avg_home_goals)
                    home_score = min(raw_score, max_home_score)

            # Deplasman skoru dağılımı - Beklenen gol değerine çok daha yakın sonuçlar üretmek için iyileştirilmiş Poisson dağılımı
            
            # Beklenen gol değerine göre maksimum makul skor sınırları belirle
            # Bu, Monte Carlo simülasyonunda aşırı değerlerin oluşmasını önler
            max_away_score = 1  # Varsayılan makul maksimum değer (düşük beklenen goller için)
            
            # Eğer özelleştirilmiş model parametreleri varsa, onları kullan
            if specialized_params:
                # Özelleştirilmiş modelden maksimum skor sınırını al
                max_away_score = specialized_params.get('max_score', 3)
                logger.debug(f"Özelleştirilmiş model maksimum deplasman skoru: {max_away_score}")
            else:
                # Standart maksimum skor hesaplama - düşük skorlu maçlar için ayarlanmış
                if avg_away_goals < 0.6:
                    # Çok düşük gol beklentisi (0.6'dan az) - daha yüksek oranda 0 gol
                    max_away_score = 1  # 0.6'dan düşük beklenen gol için maksimum 1 gol
                elif avg_away_goals < 1.0:
                    max_away_score = 1  # 0.6-1.0 arası beklenen gol için maksimum 1 gol
                elif avg_away_goals < 1.8: 
                    max_away_score = 2  # 1.0-1.8 arası beklenen gol için maksimum 2 gol
                elif avg_away_goals < 2.5:
                    max_away_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
                else:
                    max_away_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # %95 şans ile sınırlı Poisson dağılımı
            if random_selector < 0.95:
                # Gol beklentisi 0.3'ten az ise gol olasılığını biraz artır - minimum skor ihtimali için
                if avg_away_goals < 0.3:
                    raw_score = np.random.poisson(max(0.3, avg_away_goals))
                else:
                    raw_score = np.random.poisson(avg_away_goals)
                
                # Sonucu makul sınırlar içinde tut
                away_score = min(raw_score, max_away_score)
            else:
                # Çok nadir durumlarda (%5) hafif varyasyon için negatif binomial dağılımı kullan
                try:
                    # Varyasyonu daha da azalt - daha tutarlı sonuçlar için
                    away_std_dev = np.sqrt(avg_away_goals)  # Daha düşük varyasyon
                    away_r = max(1.0, avg_away_goals / 0.1)  # Daha düşük dispersiyon
                    away_p = away_r / (away_r + avg_away_goals)
                    
                    raw_score = np.random.negative_binomial(away_r, away_p)
                    # Makul sınırlar içinde tut
                    away_score = min(raw_score, max_away_score)
                except ValueError:
                    # Hata durumunda Poisson'a geri dön
                    raw_score = np.random.poisson(avg_away_goals)
                    away_score = min(raw_score, max_away_score)

            all_home_goals.append(home_score)
            all_away_goals.append(away_score)

            # Kesin skor tahmini
            exact_score_key = f"{home_score}-{away_score}"
            exact_scores[exact_score_key] = exact_scores.get(exact_score_key, 0) + 1

            # Maç sonucu
            if home_score > away_score:
                home_wins += 1
                full_time_results["HOME_WIN"] += 1
            elif home_score < away_score:
                away_wins += 1
                full_time_results["AWAY_WIN"] += 1
            else:
                draws += 1
                full_time_results["DRAW"] += 1

            # İlk yarı simülasyonu - ilk yarı gollerinin yaklaşık %40'ı atılır
            first_half_home_mean = avg_home_goals * 0.4
            first_half_away_mean = avg_away_goals * 0.4

            first_half_home = np.random.poisson(first_half_home_mean)
            first_half_away = np.random.poisson(first_half_away_mean)

            # İlk yarı sonucu
            if first_half_home > first_half_away:
                half_time_results["HOME_WIN"] += 1
                half_time_key = "HOME_WIN"
            elif first_half_home < first_half_away:
                half_time_results["AWAY_WIN"] += 1
                half_time_key = "AWAY_WIN"
            else:
                half_time_results["DRAW"] += 1
                half_time_key = "DRAW"

            # Maç sonu sonucu
            if home_score > away_score:
                full_time_key = "HOME_WIN"
            elif home_score < away_score:
                full_time_key = "AWAY_WIN"
            else:
                full_time_key = "DRAW"

            # İlk yarı/maç sonu kombinasyonu
            ht_ft_key = f"{half_time_key}/{full_time_key}"
            half_time_full_time[ht_ft_key] = half_time_full_time.get(ht_ft_key, 0) + 1

            # İlk golü kim attı
            total_goals = home_score + away_score
            if total_goals == 0:
                no_goal += 1
            else:
                # İlk golü atma olasılığı hesapla
                p_home_first = avg_home_goals / (avg_home_goals + avg_away_goals) if (avg_home_goals + avg_away_goals) > 0 else 0.5

                if np.random.random() < p_home_first and home_score > 0:
                    first_goal_home += 1
                elif away_score > 0:
                    first_goal_away += 1

            # Gol zamanlaması simülasyonu
            if total_goals == 0:
                first_goal_timing["No Goal"] += 1
            else:
                # Gol zamanlamasını simüle et - genellikle ikinci yarıda daha fazla gol olur
                timing_weights = [0.15, 0.15, 0.15, 0.17, 0.18, 0.20]  # Zamanlamalar için ağırlıklar
                timing_ranges = ["1-15", "16-30", "31-45", "46-60", "61-75", "76-90"]

                first_goal_timing[np.random.choice(timing_ranges, p=[w/sum(timing_weights) for w in timing_weights])] += 1

            # İki takım da gol attı mı
            if home_score > 0 and away_score > 0:
                both_teams_scored += 1

            # Toplam gol sayısı 2.5'tan fazla mı
            total_goals = home_score + away_score
            if total_goals > 2.5:
                over_2_5_goals += 1
            # Toplam gol sayısı 3.5'tan fazla mı
            if total_goals > 3.5:
                over_3_5_goals += 1

            # Kart sayısı simülasyonu
            # Kart sayısı maçın gerginliğine ve gol farkına bağlıdır
            tension_factor = 1.0
            if abs(home_score - away_score) <= 1:  # Yakın maçlarda daha fazla kart
                tension_factor = 1.3
            elif total_goals > 3:  # Çok gollü maçlarda genelde daha az kart
                tension_factor = 0.9

            # Ortalama kart sayısı yaklaşık 3.5
            avg_cards = 3.5 * tension_factor
            cards = np.random.poisson(avg_cards)

            if cards <= 3.5:
                cards_under_3_5 += 1
            else:
                cards_over_3_5 += 1

            # Korner sayısı simülasyonu
            # Korner sayısı takımların hücum gücüne bağlıdır
            attack_factor = (avg_home_goals + avg_away_goals) / 2.5  # Lig ortalamasına göre normalizasyon
            # Korner sayısı için üst sınır kontrolü
            avg_corners = min(15.0, 10 * attack_factor)  # En fazla 15 ortalama korner
            corners = np.random.poisson(avg_corners)

            if corners <= 9.5:
                corners_under_9_5 += 1
            else:
                corners_over_9_5 += 1

        # Olasılıkları hesapla
        home_win_prob = home_wins / simulations
        away_win_prob = away_wins / simulations
        draw_prob = draws / simulations
        
        # Özelleştirilmiş model beraberlik çarpanını uygula
        if specialized_params:
            # Beraberlik çarpanını al ve uygula
            draw_boost = specialized_params.get('draw_boost', 1.0)
            if draw_boost != 1.0:
                original_draw_prob = draw_prob
                # Beraberlik olasılığını çarpana göre ayarla (maksimum 0.95)
                draw_prob = min(0.95, draw_prob * draw_boost)
                
                # Diğer olasılıkları (homewin ve awaywin) azalt ve normalizasyon yap
                if draw_prob > original_draw_prob:
                    # Toplam galibiyet olasılığı
                    total_win_prob = home_win_prob + away_win_prob
                    if total_win_prob > 0:
                        # Beraberlik için alan açmak üzere galibiyet olasılıklarını azalt
                        reduction_factor = (1 - draw_prob) / total_win_prob
                        home_win_prob *= reduction_factor
                        away_win_prob *= reduction_factor
                    
                    logger.info(f"Özelleştirilmiş model beraberlik düzeltmesi: {original_draw_prob:.2f} -> {draw_prob:.2f} (çarpan: {draw_boost:.2f})")
        
        # Olasılıklar eşit paylaşılmış mı diye kontrol et (33-34-33 gibi)
        # Eğer öyleyse, gol beklentilerine göre yeniden hesapla
        if abs(home_win_prob - 0.33) < 0.03 and abs(away_win_prob - 0.33) < 0.03 and abs(draw_prob - 0.34) < 0.03:
            logger.warning("Olasılıklar çok dengeli dağılmış (varsayılan değerler), form verilerine göre ayarlanıyor!")
            # Monte Carlo dışı alternatif hesaplama - doğrudan gol beklentilerini kullan
            exp_total = avg_home_goals + avg_away_goals
            
            # Dixon-Coles benzeri hesaplama modeli
            # Poisson olasılıklarını hesapla
            p_home_win = 0.0
            p_draw = 0.0
            p_away_win = 0.0
            
            max_goals = 5  # Hesaplama için maksimum gol sayısı
            
            for h in range(max_goals+1):
                home_poisson = np.exp(-avg_home_goals) * (avg_home_goals**h) / np.math.factorial(h)
                
                for a in range(max_goals+1):
                    away_poisson = np.exp(-avg_away_goals) * (avg_away_goals**a) / np.math.factorial(a)
                    
                    # Düşük skorlu maçlar için tau düzeltmesi (Dixon-Coles)
                    if h <= 1 and a <= 1:
                        correction = 1.0
                        if h == 0 and a == 0:
                            correction = 1.2  # 0-0 skoru için artış
                        elif h == 1 and a == 1:
                            correction = 1.1  # 1-1 skoru için artış
                        joint_prob = home_poisson * away_poisson * correction
                    else:
                        joint_prob = home_poisson * away_poisson
                    
                    # Sonucu hesapla
                    if h > a:
                        p_home_win += joint_prob
                    elif h == a:
                        p_draw += joint_prob
                    else:
                        p_away_win += joint_prob
            
            # Olasılıkları normalize et
            total_prob = p_home_win + p_draw + p_away_win
            if total_prob > 0:
                p_home_win /= total_prob
                p_draw /= total_prob
                p_away_win /= total_prob
                
                # Monte Carlo sonuçlarıyla harmanla - Monte Carlo %40, Dixon-Coles %60 ağırlık
                home_win_prob = home_win_prob * 0.4 + p_home_win * 0.6
                draw_prob = draw_prob * 0.4 + p_draw * 0.6
                away_win_prob = away_win_prob * 0.4 + p_away_win * 0.6
                
                logger.info(f"Olasılıklar yeniden hesaplandı: Ev={home_win_prob:.2f}, Beraberlik={draw_prob:.2f}, Deplasman={away_win_prob:.2f}")
        
        # Diğer bahis olasılıkları
        both_teams_scored_prob = both_teams_scored / simulations
        over_2_5_goals_prob = over_2_5_goals / simulations
        over_3_5_goals_prob = over_3_5_goals / simulations
        
        # Düşük skorlu maçlar için özel KG YOK düzeltmesi (iyileştirilmiş 0-0, 1-0, 0-1 olasılıkları)
        # Her iki takımın da gol beklentisi düşükse KG VAR olasılığını azalt
        if avg_home_goals < 1.0 and avg_away_goals < 1.0:
            logger.info(f"Düşük skorlu maç tespit edildi. KG VAR düzeltmesi öncesi: %{both_teams_scored_prob*100:.2f}")
            # Çok düşük gol beklentisi (toplam 1.5 altı) - KG YOK olasılığını büyük ölçüde artır
            if avg_home_goals + avg_away_goals < 1.5:
                both_teams_scored_prob = both_teams_scored_prob * 0.65  # KG VAR olasılığını %35 azalt
                logger.info(f"Çok düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")
                
                # Düşük skorda hangi takımın kazanma olasılığının daha yüksek olduğunu form ve h2h verilerine göre belirle
                home_advantage_factor = self._calculate_low_scoring_advantage(home_form, away_form, home_team_id, away_team_id)
                
                # 0-0 yerine 1-0 veya 0-1 skoru için kalıtım mekanizması
                if home_advantage_factor > 0.2:  # Ev sahibi avantajlı
                    # 1-0 skorunun olasılığını artır
                    score_key_10 = "1-0"
                    exact_scores[score_key_10] = exact_scores.get(score_key_10, 0) + int(simulations * 0.08)
                    logger.info(f"Form ve H2H avantajı ev sahibinde: 1-0 skoru olasılığı artırıldı")
                elif home_advantage_factor < -0.2:  # Deplasman avantajlı
                    # 0-1 skorunun olasılığını artır
                    score_key_01 = "0-1"
                    exact_scores[score_key_01] = exact_scores.get(score_key_01, 0) + int(simulations * 0.08)
                    logger.info(f"Form ve H2H avantajı deplasmanda: 0-1 skoru olasılığı artırıldı")
                else:
                    # Her iki takımın da şansı denk, 0-0 skorunun olasılığını artır
                    score_key_00 = "0-0"
                    exact_scores[score_key_00] = exact_scores.get(score_key_00, 0) + int(simulations * 0.05)
                    logger.info(f"İki takım da denk güçte: 0-0 skoru olasılığı artırıldı")
                
            elif avg_home_goals + avg_away_goals < 1.8:
                both_teams_scored_prob = both_teams_scored_prob * 0.75  # KG VAR olasılığını %25 azalt
                logger.info(f"Düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")
            else:
                both_teams_scored_prob = both_teams_scored_prob * 0.85  # KG VAR olasılığını %15 azalt
                logger.info(f"Orta-düşük gol beklentili maç: KG VAR olasılığı %{both_teams_scored_prob*100:.2f}'e düşürüldü")

        # Gelişmiş tahminler için olasılıklar
        cards_over_3_5_prob = cards_over_3_5 / simulations
        corners_over_9_5_prob = corners_over_9_5 / simulations

        # Beraberlik olasılığını yükseltme - kesin skor dağılımına göre ayarlama
        # Hesaplanan en olası kesin skor X-X formunda ise (berabere) beraberlik olasılığını artır
        top_exact_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Özelleştirilmiş modelden beraberlik çarpanını al
        draw_boost = 1.0  # Varsayılan çarpan
        if specialized_params:
            draw_boost = specialized_params.get('draw_boost', 1.0)
            # Eğer özel bir beraberlik çarpanı varsa, direkt olarak uygula
            if draw_boost != 1.0:
                original_draw_prob = draw_prob
                # Beraberlik olasılığını artır ancak üst limiti 0.95 olarak belirle
                draw_prob = min(0.95, draw_prob * draw_boost)
                
                # Diğer olasılıkları azalt ve normalizasyon yap
                if draw_prob > original_draw_prob:
                    total_win_prob = home_win_prob + away_win_prob
                    if total_win_prob > 0:
                        reduction_factor = (1 - draw_prob) / total_win_prob
                        home_win_prob *= reduction_factor
                        away_win_prob *= reduction_factor
                    logger.info(f"Özelleştirilmiş model beraberlik düzeltmesi: {original_draw_prob:.2f} -> {draw_prob:.2f} (çarpan: {draw_boost:.2f})")

        # Kesin skor bazlı beraberlik ayarlaması
        for score, count in top_exact_scores:
            if '-' in score:
                home_score, away_score = map(int, score.split('-'))
                if home_score == away_score:  # Berabere skor
                    # Skor berabere ve ilk 3 olası skor içindeyse beraberlik olasılığını yükselt
                    score_prob = count / simulations
                    if score_prob > 0.05:  # %5'ten fazla olasılıkla gözüken beraberlik skoru
                        # Beraberlik olasılığını artır - skor olasılığına göre ağırlıklandır
                        adjustment = min(0.25, score_prob * 2)  # Max %25 artış
                        draw_prob = min(0.95, draw_prob * (1 + adjustment))
                        # Diğer olasılıkları azalt ve yeniden normalize et
                        total_win_prob = home_win_prob + away_win_prob
                        if total_win_prob > 0:
                            reduction_factor = (1 - draw_prob) / total_win_prob
                            home_win_prob *= reduction_factor
                            away_win_prob *= reduction_factor
                        logger.info(f"Skor bazlı düzeltme: {score} skoru için beraberlik olasılığı artırıldı")

        # En olası kesin skor - skorları çeşitlendirme
        # En yüksek olasılıklı 3 skoru al ve bunlardan birini seç
        top_3_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:3]

        logger.info("DEBUG: KG VAR/YOK hesaplama bölümüne ulaşıldı")
        # HİBRİT SİSTEM KULLANIMI - İLK ÖNCE HİBRİT SİSTEM ÇALIŞACAK
        logger.info(f"Hibrit sistem başlatılıyor: home_team_id={home_team_id}, away_team_id={away_team_id}")
        hybrid_system_success = False
        forced_correction_applied = False
        try:
            # Direkt hibrit sistem import ve kullanım
            import kg_prediction_models
            from kg_prediction_models import kg_predictor
            
            logger.info("Hibrit sistem import başarılı")
            kg_predictor.load_team_data()  # Takım verilerini yükle
            logger.info("Takım verileri yüklendi")
            
            hybrid_result = kg_predictor.predict_kg_var_yok(str(home_team_id), str(away_team_id))
            logger.info(f"Hibrit tahmin tamamlandı: {hybrid_result}")
            
            kg_var_adjusted_prob = hybrid_result['probability'] / 100  # Yüzdeyi oran'a çevir
            logger.info(f"HİBRİT SİSTEM SONUCU: {hybrid_result['prediction']} - %{hybrid_result['probability']}")
            logger.info(f"Bileşenler - Poisson: %{hybrid_result['components']['poisson']}, "
                       f"Logistic: %{hybrid_result['components']['logistic']}, "
                       f"Historical: %{hybrid_result['components']['historical']}")
            
            hybrid_system_success = True
            forced_correction_applied = True
            logger.info("HİBRİT SİSTEM BAŞARILI - Zorla düzeltme sistemleri atlanacak")
                
        except Exception as e:
            logger.error(f"Hibrit sistem hatası: {e}")
            import traceback
            logger.error(f"Hibrit sistem detaylı hata: {traceback.format_exc()}")
            hybrid_system_success = False
            
        # Eğer hibrit sistem başarısız olduysa, fallback sistem
        if not hybrid_system_success:
            logger.info("Hibrit sistem başarısız, fallback sistema geçiliyor")
            # KG VAR/YOK temel hesaplaması - matematiksel yaklaşım
            import math
            p_home_scores = 1 - math.exp(-expected_home_goals)
            p_away_scores = 1 - math.exp(-expected_away_goals)
            p_both_teams_score = p_home_scores * p_away_scores  # İki takımın da gol atma olasılığı
            
            # BASIT VE ETKİLİ KG VAR/YOK SİSTEMİ
            # Düşük gol beklentili takımlar için agresif düzeltme
            kg_var_adjusted_prob = p_both_teams_score
        
        # Hibrit sistem başarılı ise zorla düzeltme sistemlerini atla
        if not hybrid_system_success:
            # Minimum gol beklentisine göre agresif düzeltme
            min_goal_expectation = min(expected_home_goals, expected_away_goals)
            
            if min_goal_expectation < 0.5:
                kg_var_adjusted_prob *= 0.08  # %92 azalt
                logger.info(f"Çok düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.5) - %92 azalma")
            elif min_goal_expectation < 0.7:
                kg_var_adjusted_prob *= 0.12  # %88 azalt
                logger.info(f"Düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.7) - %88 azalma")
            elif min_goal_expectation < 0.85:
                kg_var_adjusted_prob *= 0.18  # %82 azalt
                logger.info(f"Orta-düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.85) - %82 azalma")
            elif min_goal_expectation < 1.0:
                kg_var_adjusted_prob *= 0.25  # %75 azalt
                logger.info(f"1'in altı minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 1.0) - %75 azalma")
            
            # Toplam gol beklentisi de düşükse ek düzeltme
            total_goals = expected_home_goals + expected_away_goals
            if total_goals < 1.5:
                kg_var_adjusted_prob *= 0.7  # Ek %30 azalt
                logger.info(f"Düşük toplam gol beklentisi ek düzeltmesi ({total_goals:.2f} < 1.5)")
            elif total_goals < 2.0:
                kg_var_adjusted_prob *= 0.85  # Ek %15 azalt
                logger.info(f"Orta-düşük toplam gol beklentisi ek düzeltmesi ({total_goals:.2f} < 2.0)")
        else:
            logger.info("HİBRİT SİSTEM AKTIF - Zorla düzeltme sistemleri atlandı")
        
        logger.info(f"Basit KG VAR sistemi sonucu: {kg_var_adjusted_prob:.3f}")
        
        # DÜŞÜK GOL BEKLENTİSİ DÜZELTMESI DEVRE DIŞI - KG YOK SORUNUNU TAMAMEN ÇÖZ
        # Bu kod sistematik olarak KG YOK tahminlerine neden oluyordu
        if False:  # Bu blok artık çalışmayacak
            kg_var_adjusted_prob = min(kg_var_adjusted_prob, 0.25)  # Bu satır artık çalışmaz
            logger.info(f"Bu log artık görünmeyecek")
        
        # Hibrit sistem başarılı ise zorla düzeltme sistemlerini atla  
        if not hybrid_system_success:
            # Son 5 maçtaki gol istatistiklerini de dikkate al
            home_recent_scored = sum(match.get('goals_scored', 0) for match in home_match_data[:5] if 'goals_scored' in match)
            home_recent_conceded = sum(match.get('goals_conceded', 0) for match in home_match_data[:5] if 'goals_conceded' in match)
            away_recent_scored = sum(match.get('goals_scored', 0) for match in away_match_data[:5] if 'goals_scored' in match)
            away_recent_conceded = sum(match.get('goals_conceded', 0) for match in away_match_data[:5] if 'goals_conceded' in match)
            
            # İki takım da son 5 maçta ortalama 1+ gol attıysa, KG VAR olasılığını yükselt
            if home_recent_scored >= 5 and away_recent_scored >= 5:
                kg_var_adjusted_prob = max(kg_var_adjusted_prob, 0.75)
                logger.info(f"İki takım da son 5 maçta ortalama 1+ gol attı: KG VAR olasılığı %{kg_var_adjusted_prob*100:.2f}'e yükseltildi")
        
        # SAVUNMA TABANI KG YOK DÜZELTMESI DEVRE DIŞI - KG YOK SORUNUNU ÇÖZ
        # Bu kod da sistematik olarak KG YOK tahminlerine neden oluyordu
        if False:  # Bu blok artık çalışmayacak
            kg_var_adjusted_prob = min(kg_var_adjusted_prob, 0.35)
            logger.info(f"Bu log artık görünmeyecek")
        
        # Duplicate hybrid system block removed - using the primary one above
            
            # Düşük gol beklentisi için matematiksel ayarlama (zorla değil!)
            min_goal_exp = min(expected_home_goals, expected_away_goals)
            total_goals_exp = expected_home_goals + expected_away_goals
            
            if min_goal_exp <= 0.5:
                adjustment_factor = 0.15 * math.exp(min_goal_exp * 2)
            elif min_goal_exp <= 0.8:
                adjustment_factor = 0.2 + (min_goal_exp - 0.5) * 0.4
            elif min_goal_exp <= 1.0:
                adjustment_factor = 0.35 + (min_goal_exp - 0.8) * 0.75
            else:
                adjustment_factor = 0.7 + min(0.3, (min_goal_exp - 1.0) * 0.15)
            
            if total_goals_exp < 1.5:
                total_adjustment = 0.8
            elif total_goals_exp < 2.0:
                total_adjustment = 0.9
            else:
                total_adjustment = 1.0
                
            kg_var_adjusted_prob = kg_var_adjusted_prob * adjustment_factor * total_adjustment
            kg_var_adjusted_prob = max(0.05, min(0.95, kg_var_adjusted_prob))
            
            logger.info(f"Matematiksel ayarlama: min_gol={min_goal_exp:.2f}, faktör={adjustment_factor:.3f}")
            forced_correction_applied = False
        
        kg_var_prediction = kg_var_adjusted_prob >= 0.5  # True = KG VAR, False = KG YOK
        logger.info(f"Final KG VAR sonucu: {kg_var_adjusted_prob:.3f} -> {'KG VAR' if kg_var_prediction else 'KG YOK'}")
        
        # RANA FK MAÇI İÇİN ÖZEL LOG KAYDI
        if "Rana" in str(away_team_name) or "5697" in str(home_team_id) or "10970" in str(away_team_id):
            logger.warning(f"=== RANA FK MAÇI KG VAR/YOK ANALİZİ ===")
            logger.warning(f"Ev takım: {home_team_name} (ID: {home_team_id})")
            logger.warning(f"Deplasman: {away_team_name} (ID: {away_team_id})")
            logger.warning(f"Beklenen goller - Ev: {expected_home_goals:.2f}, Deplasman: {expected_away_goals:.2f}")
            logger.warning(f"İlk KG VAR düzeltmesi geçirildi - düşük gol beklentisi bloku devre dışı")
            logger.warning(f"Savunma tabanlı düzeltme geçirildi - defensive team bloku devre dışı")
            logger.warning(f"Ayarlanmış KG VAR olasılığı: {kg_var_adjusted_prob:.2f}")
            logger.warning(f"KG VAR tahmin (>=0.5): {kg_var_prediction}")
            logger.warning(f"=== RANA FK ANALİZ SONU ===")
        
        # Skorlar içinde en çok KG VAR ve KG YOK olan skorları bul
        kg_var_scores = [(score, count) for score, count in exact_scores.items() if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0]
        kg_yok_scores = [(score, count) for score, count in exact_scores.items() if int(score.split('-')[0]) == 0 or int(score.split('-')[1]) == 0]
        
        # KG VAR/YOK skorlarının toplam olasılıkları
        kg_var_total_prob = sum(count for _, count in kg_var_scores) / simulations if kg_var_scores else 0
        kg_yok_total_prob = sum(count for _, count in kg_yok_scores) / simulations if kg_yok_scores else 0
        
        # Simülasyon sonuçları ile KG VAR/YOK tahmini tutarsız değilse kaydet
        if (kg_var_prediction and kg_var_total_prob < 0.5) or (not kg_var_prediction and kg_yok_total_prob < 0.5):
            logger.warning(f"Uyarı: KG VAR/YOK tahmini ({kg_var_prediction}) simülasyon sonuçlarıyla tutarsız! KG VAR olasılığı: %{kg_var_total_prob*100:.2f}, KG YOK olasılığı: %{kg_yok_total_prob*100:.2f}")
        
        # KG Var/Yok tahminine göre en olası skoru belirle
        if kg_var_prediction:  # KG VAR tahmini yapıldıysa
            # Her iki takımın da gol attığı skorlar arasından en olası olanı seç
            if kg_var_scores:
                most_likely_score = max(kg_var_scores, key=lambda x: x[1])  # KG VAR skorları içinde en olası olanı
                logger.info(f"KG VAR tahmini nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                # Tahmin edilen skordan takımların gol sayılarını al
                home_score = int(most_likely_score[0].split('-')[0])
                away_score = int(most_likely_score[0].split('-')[1])
                # Eğer her iki takım da gol atmıyorsa düzelt (bu bir tutarsızlık olurdu)
                if home_score == 0 or away_score == 0:
                    logger.warning(f"KG VAR ile tutarsız skor bulundu: {most_likely_score[0]}, düzeltiliyor...")
                    if home_score == 0:
                        home_score = 1
                    if away_score == 0:
                        away_score = 1
                    most_likely_score = (f"{home_score}-{away_score}", most_likely_score[1])
                    logger.info(f"KG VAR tutarlılığı için skor {most_likely_score[0]} olarak güncellendi")
            else:
                # KG VAR tahmini yapıldı ama uygun skor bulunamadı (teorik olarak mümkün değil ama önlem amaçlı)
                logger.warning(f"KG VAR tahmini yapıldı ama hiçbir KG VAR skoru bulunamadı! Varsayılan 1-1 kullanılıyor.")
                most_likely_score = ('1-1', 1)  # Varsayılan 1-1 skoru
        else:  # KG YOK tahmini yapıldıysa
            # En az bir takımın gol atmadığı skorlar arasından en olası olanı seç
            if kg_yok_scores:
                most_likely_score = max(kg_yok_scores, key=lambda x: x[1])  # KG YOK skorları içinde en olası olanı
                
                # Tahmin edilen skordan takımların gol sayılarını al
                home_score = int(most_likely_score[0].split('-')[0])
                away_score = int(most_likely_score[0].split('-')[1])
                
                # KG YOK tutarlılığı kontrolü - her iki takım da gol atıyorsa düzelt
                if home_score > 0 and away_score > 0:
                    logger.warning(f"KG YOK ile tutarsız skor bulundu: {most_likely_score[0]}, düzeltiliyor...")
                    
                    # Beklenen gollere göre hangi takımın skor kaydetme olasılığının daha yüksek olduğunu belirle
                    if expected_home_goals > expected_away_goals + 0.3:
                        # Ev sahibi takımın gol beklentisi daha yüksek, 0'lı skorlarda ev sahibini seç
                        home_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                         if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) == 0]
                        if home_win_kg_yok:
                            most_likely_score = max(home_win_kg_yok, key=lambda x: x[1])
                            logger.info(f"KG YOK + Ev sahibi üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                        else:
                            most_likely_score = ('1-0', 1)  # Tutarlılık için varsayılan değer
                            logger.warning("KG YOK + Ev sahibi üstünlüğü için uygun skor bulunamadı, varsayılan 1-0 kullanılıyor")
                    elif expected_away_goals > expected_home_goals + 0.3:
                        # Deplasman takımının gol beklentisi daha yüksek, 0'lı skorlarda deplasmanı seç
                        away_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                         if int(score.split('-')[0]) == 0 and int(score.split('-')[1]) > 0]
                        if away_win_kg_yok:
                            most_likely_score = max(away_win_kg_yok, key=lambda x: x[1])
                            logger.info(f"KG YOK + Deplasman üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                        else:
                            most_likely_score = ('0-1', 1)  # Tutarlılık için varsayılan değer
                            logger.warning("KG YOK + Deplasman üstünlüğü için uygun skor bulunamadı, varsayılan 0-1 kullanılıyor")
                    else:
                        # Takımlar denk, olası 0-0
                        zero_zero_scores = [(score, count) for score, count in kg_yok_scores if score == '0-0']
                        if zero_zero_scores:
                            most_likely_score = zero_zero_scores[0]
                            logger.info(f"KG YOK + dengeli maç nedeniyle kesin skor 0-0 olarak güncellendi")
                        else:
                            # Eğer 0-0 yoksa, en olası KG YOK skorunu seç
                            if kg_yok_scores:
                                most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                            else:
                                most_likely_score = ('0-0', 1)
                                logger.warning("KG YOK için uygun skor bulunamadı, varsayılan 0-0 kullanılıyor")
                elif expected_away_goals > expected_home_goals + 0.3:
                    # Deplasman takımının gol beklentisi daha yüksek, 0'lı skorlarda deplasmanı seç
                    away_win_kg_yok = [(score, count) for score, count in kg_yok_scores 
                                     if int(score.split('-')[0]) == 0 and int(score.split('-')[1]) > 0]
                    if away_win_kg_yok:
                        most_likely_score = max(away_win_kg_yok, key=lambda x: x[1])
                        logger.info(f"KG YOK + Deplasman üstünlüğü nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
                    else:
                        most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                else:
                    # Takımlar denk, en olası KG YOK skorunu seç
                    most_likely_score = max(kg_yok_scores, key=lambda x: x[1])
                
                logger.info(f"KG YOK tahmini nedeniyle kesin skor {most_likely_score[0]} olarak güncellendi")
            else:
                # KG YOK tahmini yapıldı ama hiç KG YOK skoru bulunamadı (teorik olarak mümkün değil)
                logger.warning(f"KG YOK tahmini yapıldı ama hiçbir KG YOK skoru bulunamadı! Varsayılan 0-0 veya 1-0 kullanılıyor.")
                if expected_home_goals > expected_away_goals:
                    most_likely_score = ('1-0', 1)  # Varsayılan 1-0 skoru
                else:
                    most_likely_score = ('0-0', 1)  # Varsayılan 0-0 skoru

        most_likely_score_prob = most_likely_score[1] / simulations

        # İlk yarı/maç sonu en olası kombinasyon
        most_likely_ht_ft = max(half_time_full_time.items(), key=lambda x: x[1]) if half_time_full_time else ("DRAW/DRAW", 0)
        most_likely_ht_ft_prob = most_likely_ht_ft[1] / simulations if half_time_full_time else 0

        # İlk golün zamanlaması
        most_likely_first_goal_time = max(first_goal_timing.items(), key=lambda x: x[1])
        most_likely_first_goal_time_prob = most_likely_first_goal_time[1] / simulations

        # İlk golü atan takım
        first_goal_home_prob = first_goal_home / simulations if (first_goal_home + first_goal_away + no_goal) > 0 else 0
        first_goal_away_prob = first_goal_away / simulations if (first_goal_home + first_goal_away + no_goal) > 0 else 0
        no_goal_prob = no_goal / simulations

        # Beklenen gol sayıları (final estimations) - form faktörünün etkisini daha da azalt
        # avg_home_goals ve avg_away_goals zaten daha önce tanımlandı, burada sadece güncelleniyor
        avg_home_goals = expected_home_goals * (home_form_factor * 0.85)
        avg_away_goals = expected_away_goals * (away_form_factor * 0.85)

        # Aşırı yüksek tahminleri düzeltmek için gelişmiş yöntemler

        # Son 3, 5 ve 10 maçın gerçek gol ortalamalarını hesapla
        home_recent_avg_goals = {}
        away_recent_avg_goals = {}
        periods = [3, 5, 10]

        for period in periods:
            # Ev sahibi için
            home_matches_count = min(period, len(home_match_data))
            if home_matches_count > 0:
                home_recent_avg_goals[period] = sum(match.get('goals_scored', 0) for match in home_match_data[:home_matches_count]) / home_matches_count
            else:
                home_recent_avg_goals[period] = self.lig_ortalamasi_ev_gol

            # Deplasman için
            away_matches_count = min(period, len(away_match_data))
            if away_matches_count > 0:
                away_recent_avg_goals[period] = sum(match.get('goals_scored', 0) for match in away_match_data[:away_matches_count]) / away_matches_count
            else:
                away_recent_avg_goals[period] = 1.0

        # Son maçların ortalaması ile genel lig ortalamasını karşılaştır
        home_avg_deviation = (home_recent_avg_goals[3] / self.lig_ortalamasi_ev_gol) * 0.5 + \
                            (home_recent_avg_goals[5] / self.lig_ortalamasi_ev_gol) * 0.3 + \
                            (home_recent_avg_goals[10] / self.lig_ortalamasi_ev_gol) * 0.2
        away_avg_deviation = (away_recent_avg_goals[3] / 1.0) * 0.5 + \
                            (away_recent_avg_goals[5] / 1.0) * 0.3 + \
                            (away_recent_avg_goals[10] / 1.0) * 0.2

        # Sapma değerini sınırla (çok aşırı değerleri engelle)
        home_avg_deviation = min(1.5, max(0.7, home_avg_deviation))
        away_avg_deviation = min(1.5, max(0.7, away_avg_deviation))

        # Z-skor bazlı normalizasyon için takım gol dağılımlarını hesapla
        # Standart sapma hesapla (son 10 maçta)
        home_std_dev = np.std([match.get('goals_scored', 0) for match in home_match_data[:10]]) if len(home_match_data) >= 10 else 1.0
        away_std_dev = np.std([match.get('goals_scored', 0) for match in away_match_data[:10]]) if len(away_match_data) >= 10 else 0.8

        # Savunma gücü değerlendirmesi - rakip takımın savunma istatistikleri
        home_defense_strength = away_form.get('home_performance', {}).get('weighted_avg_goals_conceded', weighted_away_conceded)
        away_defense_strength = home_form.get('away_performance', {}).get('weighted_avg_goals_conceded', weighted_home_conceded)

        # Savunma gücünü lig ortalamasıyla karşılaştır
        home_defense_factor = home_defense_strength / 1.0
        away_defense_factor = away_defense_strength / self.lig_ortalamasi_ev_gol

        # Gol tahminlerini sapma oranı ile düzelt
        avg_home_goals = avg_home_goals * home_avg_deviation * (1.0 + 0.2 * (1.0 - min(1.5, away_defense_factor)))
        avg_away_goals = avg_away_goals * away_avg_deviation * (1.0 + 0.2 * (1.0 - min(1.5, home_defense_factor)))

        # Limit fonksiyonu - logaritmik düzeltme
        def limit_high_values(value, threshold, scaling_factor=0.3):
            if value <= threshold:
                return value
            else:
                return threshold + scaling_factor * np.log1p(value - threshold)

        # Ortalama gol performansına göre takımları sınıflandır (aşırı yüksek tahminleri daha sıkı sınırla)
        home_is_high_scoring = home_recent_avg_goals[5] > 2.0
        away_is_high_scoring = away_recent_avg_goals[5] > 1.5

        # Yüksek gol atan takımlar için daha esnek, düşük gol atan takımlar için daha sıkı sınırlar
        # Ancak genel olarak daha yüksek tahminlere izin ver
        home_threshold = 3.0 if home_is_high_scoring else 2.7
        away_threshold = 2.5 if away_is_high_scoring else 2.2

        # Ev sahibi gol tahminlerini sınırla - daha yumuşak sınırlama için scaling factor artırıldı
        avg_home_goals = limit_high_values(avg_home_goals, home_threshold, 0.5)

        # Deplasman gol tahminlerini sınırla - daha yumuşak sınırlama için scaling factor artırıldı
        avg_away_goals = limit_high_values(avg_away_goals, away_threshold, 0.45)

        # Gerçek dünya istatistiklerine göre maksimum sınırlar - daha dengeli değerler
        home_max = 3.2 if home_is_high_scoring else 3.0
        away_max = 3.2 if away_is_high_scoring else 2.8

        if avg_home_goals > home_max:
            avg_home_goals = home_max + ((avg_home_goals - home_max) * 0.25)

        if avg_away_goals > away_max:
            avg_away_goals = away_max + ((avg_away_goals - away_max) * 0.25)

        # Minimum değerler için daha dengeli alt sınırlar belirle
        # Ev sahibi için daha düşük minimum değer kullanarak zayıf takımları daha doğru yansıt
        avg_home_goals = max(0.8, avg_home_goals)
        avg_away_goals = max(0.7, avg_away_goals)

        # Standart sapma hesapla - Poisson dağılımında standart sapma, ortalamanın kareköküdür
        std_dev_home = np.sqrt(avg_home_goals)  
        std_dev_away = np.sqrt(avg_away_goals)

        # KG VAR/YOK mantığını daha akıllı bir şekilde hesaplayalım
        # Eğer iki takımın da beklenen gol sayısı yüksekse, KG VAR VAR olasılığı daha yüksek olmalı
        kg_var_theoretical_prob = p_home_scores * p_away_scores  # Bağımsız olasılıklar çarpımı

        # HİBRİT SİSTEM AKTİF - Tüm zorla düzeltme sistemleri devre dışı
        logger.info(f"Hibrit sistem sonucu korunuyor: {kg_var_adjusted_prob:.3f} (forced_correction_applied: {forced_correction_applied})")

        # 2.5 ve 3.5 gol için teorik olasılıklar (Poisson kümülatif dağılım fonksiyonu)
        # P(X > 2.5) = 1 - P(X ≤ 2) where X ~ Poisson(lambda)
        lambda_total = expected_total_goals
        p_under_25_theoretical = np.exp(-lambda_total) * (1 + lambda_total + (lambda_total**2)/2)
        p_over_25_theoretical = 1 - p_under_25_theoretical

        # 3.5 gol için
        p_under_35_theoretical = p_under_25_theoretical + np.exp(-lambda_total) * (lambda_total**3)/6
        p_over_35_theoretical = 1 - p_under_35_theoretical

        # Simülasyon ve teorik hesaplamalar arasında dengeli karışım
        # Daha konservatif ağırlıklanmış - teorik hesaplamalara daha fazla ağırlık
        over_25_adjusted_prob = 0.4 * over_2_5_goals_prob + 0.6 * p_over_25_theoretical
        over_35_adjusted_prob = 0.4 * over_3_5_goals_prob + 0.6 * p_over_35_theoretical

        # Bahis tahminlerini hazırla - korner ve kart tahminlerini çıkararak basitleştir
        # İlk olarak kesin skordan maç sonucunu türet (tutarlılık için)
        temp_exact_score = most_likely_score[0]
        
        # Kesin skordan maç sonucunu türet (tutarlılık için)
        score_based_outcome = self._get_outcome_from_score(temp_exact_score)
        
        bet_predictions = {
            'match_result': 'MS1' if score_based_outcome == 'HOME_WIN' else
                           'X' if score_based_outcome == 'DRAW' else 'MS2',
            'both_teams_to_score': 'KG VAR' if kg_var_prediction else 'KG YOK',  # KG VAR/YOK formatında - tutarlılık için kg_var_prediction kullan
            'over_2_5_goals': '2.5 ÜST' if over_2_5_goals_prob > 0.5 else '2.5 ALT',  # Gerçek Monte Carlo sonucu
            'over_3_5_goals': '3.5 ÜST' if over_3_5_goals_prob > 0.5 else '3.5 ALT',  # Gerçek Monte Carlo sonucu
            'exact_score': temp_exact_score,  # Tutarlılık için aynı skoru kullan
            'half_time_full_time': most_likely_ht_ft[0].replace('HOME_WIN', 'MS1').replace('DRAW', 'X').replace('AWAY_WIN', 'MS2'),
            'first_goal_time': most_likely_first_goal_time[0],
            'first_goal_team': 'EV' if first_goal_home_prob > first_goal_away_prob and first_goal_home_prob > no_goal_prob else
                              'DEP' if first_goal_away_prob > first_goal_home_prob and first_goal_away_prob > no_goal_prob else 'GOL YOK'
            # Korner ve kart tahminleri kaldırıldı
        }
        
        # KONSEPTÜEL TUTARLILIK KONTROLÜ:
        # Kesin skordan toplam gol sayısını hesapla ve ÜST/ALT kararlarını güncelle
        if '-' in bet_predictions['exact_score']:
            try:
                home_goals, away_goals = map(int, bet_predictions['exact_score'].split('-'))
                total_goals = home_goals + away_goals
                
                # 2.5 ve 3.5 ÜST/ALT tahminleri Monte Carlo simülasyonu sonuçlarından korunuyor
                # Kesin skordan zorla düzeltme kaldırıldı - gerçek olasılıklar Monte Carlo'dan geliyor
                logger.info(f"Kesin skor {bet_predictions['exact_score']} (toplam: {total_goals} gol) - Monte Carlo sonuçları korunuyor") 
                
                # KG VAR/YOK kontrolü - Monte Carlo simülasyonu sonuçlarını koruyoruz
                # Kesin skor sadece en olası sonuç, KG VAR/YOK kararını simülasyon sonuçları belirler
                logger.info(f"Kesin skor ({bet_predictions['exact_score']}) - KG VAR/YOK kararı Monte Carlo simülasyonuna göre: {bet_predictions['both_teams_to_score']}")
                # bet_predictions['both_teams_to_score'] değeri zaten Monte Carlo simülasyonundan geldi, değiştirmiyoruz
                
                # Maç sonucu kontrolü
                if home_goals > away_goals:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) ev sahibi önde, MS1 olarak güncellendi")
                    bet_predictions['match_result'] = 'MS1'
                elif away_goals > home_goals:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) deplasman önde, MS2 olarak güncellendi")
                    bet_predictions['match_result'] = 'MS2'
                else:
                    logger.info(f"Skordan ({bet_predictions['exact_score']}) eşitlik, X olarak güncellendi")
                    bet_predictions['match_result'] = 'X'
            except Exception as e:
                logger.error(f"Skor analizi yapılırken hata oluştu: {bet_predictions['exact_score']} - Hata: {str(e)}")
        
        # Ek kontrol - bet_predictions içinde 'exact_score' kesin olarak ayarlandığından emin ol
        logger.info(f"Kesin skor tahmini: {most_likely_score[0]} (KG {'VAR' if kg_var_prediction else 'YOK'})")

        # Son maçların gol ortalamalarını sadece gerekli değerlerle kaydet
        recent_goals_average = {
            'home': home_recent_avg_goals.get(5, 0),
            'away': away_recent_avg_goals.get(5, 0)
        }

        # Gol beklentilerine göre KG VAR/YOK, ÜST/ALT tahminlerini ayarla ve tutarlılığı sağla
        expected_total_goals = avg_home_goals + avg_away_goals
        
        # KG VAR/YOK tahmini - geliştirilmiş mantık
        # Takımların gol beklentileri, gol atma olasılıkları ve son maçlardaki performanslarını hesaba kat
        p_home_scores_at_least_one = 1 - np.exp(-avg_home_goals)
        p_away_scores_at_least_one = 1 - np.exp(-avg_away_goals)
        
        # İki takımın da en az 1 gol atma olasılığı
        p_both_teams_score = p_home_scores_at_least_one * p_away_scores_at_least_one
        
        # Son maçlardaki KG VAR/YOK oranlarını analiz et
        kg_var_rate_home = 0
        kg_var_rate_away = 0
        kg_var_matches_home = 0
        kg_var_matches_away = 0
        
        # Ev sahibi takımın son maçlarında KG VAR oranı
        for match in home_match_data[:min(10, len(home_match_data))]:
            if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0:
                kg_var_matches_home += 1
        
        # Deplasman takımının son maçlarında KG VAR oranı
        for match in away_match_data[:min(10, len(away_match_data))]:
            if match.get('goals_scored', 0) > 0 and match.get('goals_conceded', 0) > 0:
                kg_var_matches_away += 1
        
        if home_match_data and len(home_match_data) > 0:
            kg_var_rate_home = kg_var_matches_home / min(10, len(home_match_data))
        
        if away_match_data and len(away_match_data) > 0:
            kg_var_rate_away = kg_var_matches_away / min(10, len(away_match_data))
        
        # İki takımın geçmiş maçlarındaki KG VAR oranı
        kg_var_historical_rate = (kg_var_rate_home + kg_var_rate_away) / 2
        
        logger.info(f"KG VAR geçmiş oranları - Ev: {kg_var_rate_home:.2f}, Deplasman: {kg_var_rate_away:.2f}, Ortalama: {kg_var_historical_rate:.2f}")
        
        # Teorik ve geçmiş verileri birleştirerek daha doğru tahmin yap
        p_kg_var_combined = 0.6 * p_both_teams_score + 0.4 * kg_var_historical_rate
        
        logger.info(f"KG VAR olasılığı - Teorik: {p_both_teams_score:.2f}, Geçmiş: {kg_var_historical_rate:.2f}, Birleşik: {p_kg_var_combined:.2f}")
        
        # İlk olarak varsayılan tahmin yap (başlangıçta nötr)
        kg_var_prediction = p_kg_var_combined > 0.5  # Başlangıç tahmini
        
        # HİBRİT SİSTEM AKTİF - Ek düzeltme yapılmayacak, hibrit sonucu korunacak
        if not forced_correction_applied:
            logger.info(f"Hibrit sistem aktif - ek düzeltme yapılmadı, sonuç korundu: {kg_var_adjusted_prob:.3f}")
        else:
            logger.warning(f"ZORLA DÜZELTME FLAGİ - Gol beklentisi artırımları atlandı")
        
        # KG VAR/YOK tahminini bağımsız olarak simülasyon sonuçlarından belirle
        # Simülasyon zaten yapıldı, sonuçlarını kullan
        btts_yes_prob = both_teams_scored_prob
        btts_no_prob = 1 - both_teams_scored_prob
        
        # HİBRİT SİSTEM SONUCUNU KULLAN
        if hybrid_kg_result is not None:
            # Hibrit sistem sonucunu direkt kullan
            hybrid_prob = hybrid_kg_result['probability'] / 100
            bet_predictions['both_teams_to_score'] = hybrid_kg_result['prediction']
            kg_var_adjusted_prob = hybrid_prob  # Override the forced correction value
            logger.info(f">>> HİBRİT SİSTEM SONUCU KULLANILIYOR <<<: {hybrid_kg_result['prediction']} - %{hybrid_kg_result['probability']}")
            logger.info(f"Hibrit bileşenler - Poisson: %{hybrid_kg_result['components']['poisson']}, "
                       f"Logistic: %{hybrid_kg_result['components']['logistic']}, "
                       f"Historical: %{hybrid_kg_result['components']['historical']}")
        else:
            # Fallback sistem - KG VAR/YOK tahminini zorla düzeltme sistemi ile uyumlu hale getir
            if kg_var_adjusted_prob > 0.5:
                bet_predictions['both_teams_to_score'] = 'KG VAR'
                logger.info(f"Fallback sistem KG VAR tahmini: %{kg_var_adjusted_prob*100:.1f}")
            else:
                bet_predictions['both_teams_to_score'] = 'KG YOK' 
                logger.info(f"Fallback sistem KG YOK tahmini: %{(1-kg_var_adjusted_prob)*100:.1f}")
        
        # Simülasyon sonuçları ile uyumsuzluk kontrolü (sadece bilgi amaçlı)
        if btts_yes_prob > btts_no_prob:
            logger.info(f"Not: Simülasyon KG VAR öneriyor (%{btts_yes_prob*100:.1f}) ama hesaplanan değer kullanılıyor")
        else:
            logger.info(f"Not: Simülasyon KG YOK öneriyor (%{btts_no_prob*100:.1f}) ama hesaplanan değer kullanılıyor")
        
        # 2.5 ve 3.5 ÜST/ALT tahminleri zaten satır 3808-3809'da Monte Carlo temel alınarak belirlendi
        # Burada tekrar override edilmeyecek - gerçek Monte Carlo sonuçları korunacak
        logger.info(f"2.5 ve 3.5 Alt/Üst tahminleri Monte Carlo simülasyonu sonuçlarından korunuyor (satır 3808-3809)")
        logger.info(f"Monte Carlo sonuçları: 2.5 ÜST olasılığı %{over_2_5_goals_prob*100:.1f}, 3.5 ÜST olasılığı %{over_3_5_goals_prob*100:.1f}")
            
        # Bahis tahminleri artık bağımsız - kesin skor ile tutarlılık kontrolü yok
        # Her tahmin kendi Monte Carlo simülasyonu sonuçlarına dayanıyor
        logger.info("Tüm bahis tahminleri Monte Carlo simülasyonu sonuçlarından bağımsız olarak belirlendi")

        # En yüksek olasılıklı tahmini belirle - korner, kart, ilk gol ve İY/MS tahminlerini çıkardık
        # Gerçekçi olasılık sınırları uygula - aşırı yüksek değerleri önle
        def normalize_probability(prob, min_val=0.15, max_val=0.85):
            """Olasılığı gerçekçi sınırlar içinde normalize et"""
            return max(min_val, min(max_val, prob))
        
        # Normalized olasılıkları hesapla
        normalized_kg_var = normalize_probability(kg_var_adjusted_prob)
        logger.warning(f"NORMALIZE KONTROLÜ: kg_var_adjusted_prob={kg_var_adjusted_prob:.3f} -> normalized_kg_var={normalized_kg_var:.3f}")
        normalized_over25 = normalize_probability(over_25_adjusted_prob)
        normalized_over35 = normalize_probability(over_35_adjusted_prob, min_val=0.10, max_val=0.75)
        
        # Monte Carlo simülasyonundan gelen gerçek olasılıkları kullan
        bet_probabilities = {
            'match_result': max(home_win_prob, draw_prob, away_win_prob),
            'both_teams_to_score': normalized_kg_var if bet_predictions['both_teams_to_score'] == 'KG VAR' else (1 - normalized_kg_var),
            'over_2_5_goals': normalized_over25 if bet_predictions['over_2_5_goals'] == '2.5 ÜST' else (1 - normalized_over25),
            'over_3_5_goals': normalized_over35 if bet_predictions['over_3_5_goals'] == '3.5 ÜST' else (1 - normalized_over35),
            'exact_score': most_likely_score_prob
            # İY/MS, ilk gol tahminleri kaldırıldı (half_time_full_time, first_goal_time ve first_goal_team)
        }

        # Tahminler arasındaki mantık tutarlılığını kontrol et - geliştirilmiş versiyon
        
        # İlk adım: Gol beklentilerine göre olasılıkları ayarlama
        # Doğrudan gol beklentisi farkından MS olasılıklarını hesapla
        goal_diff = avg_home_goals - avg_away_goals
        
        # Gol beklentileri ve maç sonucu arasındaki ilişkiyi daha doğru kur
        # Gol farkı formülü: sigmoid benzeri bir yaklaşım kullan
        def sigmoid_like(x, scale=1.5):
            return 1 / (1 + np.exp(-scale * x))
        
        # Gol farkına göre ev sahibi ve deplasman kazanma olasılıklarını hesapla
        base_home_win = sigmoid_like(goal_diff)
        base_away_win = sigmoid_like(-goal_diff)
        
        # Beraberlik olasılığını gol farkının mutlak değerine göre ayarla
        # Gol farkı az ise beraberlik olasılığı yüksek olmalı
        base_draw = 1 - (sigmoid_like(abs(goal_diff), scale=2.5))
        
        # Olasılıkları normalize et
        total = base_home_win + base_draw + base_away_win
        norm_home_win = base_home_win / total
        norm_draw = base_draw / total  
        norm_away_win = base_away_win / total
        
        # Mevcut simülasyon olasılıkları ile hesaplanan olasılıklar arasında denge kur
        # %70 simülasyon sonuçlarına, %30 gol beklentisi temelli matematiksel hesaplamaya güven
        blend_factor = 0.3
        home_win_prob = (1 - blend_factor) * home_win_prob + blend_factor * norm_home_win
        draw_prob = (1 - blend_factor) * draw_prob + blend_factor * norm_draw  
        away_win_prob = (1 - blend_factor) * away_win_prob + blend_factor * norm_away_win
        
        logger.info(f"Gol beklentilerine göre MS olasılıkları ayarlandı: MS1={home_win_prob:.2f}, X={draw_prob:.2f}, MS2={away_win_prob:.2f}")
        
        # İkinci adım: Kesin skor ile diğer tahminleri uyumlu hale getir
        # Önce en olası kesin skoru belirle
        top_3_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # En olası skorları analiz et
        score_probabilities = {}
        for score, count in top_3_scores:
            score_probabilities[score] = count / simulations
            
        logger.info(f"En olası skorlar ve olasılıkları: {score_probabilities}")
        
        # Doğrudan beklenen gol değerlerinden kesin skoru hesapla
        most_expected_home_score = round(avg_home_goals)
        most_expected_away_score = round(avg_away_goals)
        expected_score = f"{most_expected_home_score}-{most_expected_away_score}"
        
        # Top 5 skorları alarak olasılık dağılımını daha iyi anla
        top_5_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"En olası 5 skor ve olasılıkları: {[(s, round(c/simulations*100, 2)) for s, c in top_5_scores]}")
        logger.info(f"Beklenen goller: Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")
        
        # Kesin skor olasılıklarını daha doğru değerlendirmek için histogram
        score_histogram = {}
        
        # Düşük skorlu maçlar için olasılık artışı faktörleri - daha güçlü artış
        low_score_boost = {
            "0-0": 3.0,  # %200 artış
            "1-0": 2.5,  # %150 artış
            "0-1": 2.5,  # %150 artış
            "1-1": 1.8,  # %80 artış
            "2-0": 1.5,  # %50 artış
            "0-2": 1.5   # %50 artış
        }
        
        # Her iki takımın da gol beklentisi düşükse, düşük skorların olasılığını önemli ölçüde artır
        # Sınırı 1.2'ye yükselttik - daha fazla maç düşük skorlu olarak değerlendirilecek
        is_low_scoring_match = avg_home_goals < 1.2 and avg_away_goals < 1.2
        
        # Skoru bir takımın lehine değiştirmek için form farklarını kontrol et
        home_stronger = False
        away_stronger = False
        
        if 'home_performance' in home_form and 'away_performance' in away_form:
            home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', 0.5)
            away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', 0.5)
            
            # Form puanları arasında önemli fark varsa
            if home_form_points > away_form_points + 0.2:
                home_stronger = True
                logger.info(f"Ev sahibi takım form olarak daha güçlü: {home_form_points:.2f} > {away_form_points:.2f}")
            elif away_form_points > home_form_points + 0.2:
                away_stronger = True
                logger.info(f"Deplasman takımı form olarak daha güçlü: {away_form_points:.2f} > {home_form_points:.2f}")
        
        # Takım form farklarına göre düşük skor boost faktörlerini ayarla
        if home_stronger:
            low_score_boost["1-0"] = 3.0  # %200 artış
            low_score_boost["2-0"] = 2.0  # %100 artış
            low_score_boost["2-1"] = 1.5  # %50 artış
            logger.info(f"Ev sahibi takım daha formda olduğu için 1-0, 2-0, 2-1 skorlarının olasılıkları artırıldı")
        elif away_stronger:
            low_score_boost["0-1"] = 3.0  # %200 artış
            low_score_boost["0-2"] = 2.0  # %100 artış
            low_score_boost["1-2"] = 1.5  # %50 artış
            logger.info(f"Deplasman takımı daha formda olduğu için 0-1, 0-2, 1-2 skorlarının olasılıkları artırıldı")
        
        # Skorları histogram şeklinde oluştur ve düşük skorlu maçlar için özel işlem yap
        for h in range(6):  # 0-5 gol
            for a in range(6):  # 0-5 gol
                score = f"{h}-{a}"
                base_prob = exact_scores.get(score, 0) / simulations
                
                # Düşük skorlu maç kontrolü ve ek kontroller
                if is_low_scoring_match:
                    # Düşük skorlu bir maçta her iki takım da 1.0'dan az gol beklentisine sahipse
                    # ve skorda toplam gol sayısı 3 veya daha fazlaysa, bu skor daha az olası
                    if h + a >= 3:
                        base_prob = base_prob * 0.5  # %50 azalt, düşük skorlu maçta yüksek skoru daha agresif azalt
                        logger.info(f"Düşük skorlu maçta yüksek skor {score} azaltıldı: %{round(base_prob*100, 2)}")
                    
                    # 0-0, 0-1, 1-0 gibi skorları özel olarak artır
                    if score in ["0-0", "1-0", "0-1"]:
                        special_boost = 3.5  # Çok düşük skorlu maçlar için ekstra artış
                        boosted_prob = base_prob * special_boost
                        score_histogram[score] = boosted_prob
                        logger.info(f"Çok düşük skorlu maç tespiti: {score} skoru olasılığı %{round(base_prob*100, 2)}'den %{round(boosted_prob*100, 2)}'e yükseltildi")
                    # Boost uygulanacak skor ise olasılığı artır
                    elif score in low_score_boost:
                        boosted_prob = base_prob * low_score_boost[score]
                        score_histogram[score] = boosted_prob
                        logger.info(f"Düşük skorlu maç tespiti: {score} skoru olasılığı %{round(base_prob*100, 2)}'den %{round(boosted_prob*100, 2)}'e yükseltildi")
                    else:
                        score_histogram[score] = base_prob
                else:
                    # Düşük skorlu maç değilse normal olasılık kullan
                    score_histogram[score] = base_prob
                    
        # Skor tahminlerini normalize et - toplam olasılık 1.0 olsun
        total_prob = sum(score_histogram.values())
        if total_prob > 0:
            for score in score_histogram:
                score_histogram[score] = score_histogram[score] / total_prob
                
        # En yüksek olasılıklı skorları gruplandırarak analiz et
        same_outcome_scores = {
            "HOME_WIN": {},
            "DRAW": {},
            "AWAY_WIN": {}
        }
        
        # Gerçekçi skor sınırları - beklenen gollere göre belirleme
        max_reasonable_home = 4
        max_reasonable_away = 4
        
        if avg_home_goals < 1.0:
            max_reasonable_home = 1
        elif avg_home_goals < 2.0:
            max_reasonable_home = 2
        elif avg_home_goals < 3.0:
            max_reasonable_home = 3
            
        if avg_away_goals < 1.0:
            max_reasonable_away = 1
        elif avg_away_goals < 2.0:
            max_reasonable_away = 2
        elif avg_away_goals < 3.0:
            max_reasonable_away = 3
        
        # Skorları gruplandır ve gerçekçi olmayan skorları filtrele
        for score, prob in score_histogram.items():
            if '-' in score:
                h, a = map(int, score.split('-'))
                
                # Gerçekçi olmayan yüksek skorları filtrele
                if h > max_reasonable_home * 1.5 or a > max_reasonable_away * 1.5:
                    logger.info(f"Gerçekçi olmayan skor filtrelendi: {score} (beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f})")
                    continue
                
                if h > a:
                    same_outcome_scores["HOME_WIN"][score] = prob
                elif h == a:
                    # Yüksek beraberlik skorlarını filtrele
                    if h <= max_reasonable_home:
                        same_outcome_scores["DRAW"][score] = prob
                    else:
                        logger.info(f"Gerçekçi olmayan beraberlik skoru filtrelendi: {score} (beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f})")
                else:
                    same_outcome_scores["AWAY_WIN"][score] = prob
        
        # Maç sonucu olasılıklarına göre en olası skoru belirlemek - beklenen gollere göre
        most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
        
        # En olası maç sonucu için bir skor listesi oluştur
        if most_likely_outcome in same_outcome_scores and same_outcome_scores[most_likely_outcome]:
            top_scores_by_outcome = sorted(same_outcome_scores[most_likely_outcome].items(), key=lambda x: x[1], reverse=True)
        else:
            # Eğer en olası sonuç için skor bulunamazsa (filtrelerden dolayı boş kalmış olabilir)
            # alternatif bir sonuç kullan veya beklenen gollere göre bir skor oluştur
            logger.warning(f"En olası sonuç {most_likely_outcome} için skor bulunamadı, alternatif kullanılıyor")
            
            # Beklenen gollere dayanarak alternatif bir maç sonucu belirle
            if avg_home_goals > avg_away_goals + 0.5:
                alt_outcome = "HOME_WIN"
            elif avg_away_goals > avg_home_goals + 0.5:
                alt_outcome = "AWAY_WIN"
            else:
                alt_outcome = "DRAW"
                
            # Alternatif sonuç için skorlar var mı?
            if alt_outcome in same_outcome_scores and same_outcome_scores[alt_outcome]:
                top_scores_by_outcome = sorted(same_outcome_scores[alt_outcome].items(), key=lambda x: x[1], reverse=True)
                logger.info(f"Alternatif sonuç {alt_outcome} kullanıldı, beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f}")
            else:
                # Herhangi bir uygun skor bulunamazsa, beklenen gollere göre oluştur
                rounded_home = round(avg_home_goals)
                rounded_away = round(avg_away_goals)
                score = f"{rounded_home}-{rounded_away}"
                top_scores_by_outcome = [(score, 1.0)]
                logger.info(f"Hiçbir uygun skor bulunamadı, beklenen gollere göre skor oluşturuldu: {score}")
        
        # Eğer en olası maç sonucu için skorlar varsa, bunları değerlendir
        if top_scores_by_outcome:
            logger.info(f"En olası maç sonucu {most_likely_outcome} için olası skorlar: {[(s, round(p*100, 2)) for s, p in top_scores_by_outcome[:3]]}")
        # Düşük gol beklentisinde (1'in altında) form durumuna göre karar ver
        # Ev sahibi takım için
        if avg_home_goals < 1.0:
            # Form ve ev sahibi avantajını değerlendir
            home_form_points = home_form.get('home_performance', {}).get('weighted_form_points', 0.3)
            
            # Son 3 maçtaki gol ortalamasına da bak
            recent_goals_avg = 0
            goals_in_last_matches = 0
            match_count = 0
            
            for match in home_match_data[:3]:
                if match.get('is_home', False):  # Sadece ev sahibi maçlarını dikkate al
                    goals_in_last_matches += match.get('goals_scored', 0)
                    match_count += 1
            
            if match_count > 0:
                recent_goals_avg = goals_in_last_matches / match_count
            
            # Form iyi, ev sahibi avantajı varsa veya son maçlarda gol attıysa
            if (home_form_points > 0.5 and home_advantage > 1.02) or weighted_home_form_points > 0.7 or recent_goals_avg > 0.7:
                # Form iyiyse veya son maçlarda gol attıysa daha yüksek gol olasılığı
                rounded_home_goals = 1
                logger.info(f"Ev sahibi gol beklentisi 1'in altında ({avg_home_goals:.2f}) ama form iyi veya son maçlarda gol ortalaması {recent_goals_avg:.2f}, 1 gol veriliyor")
            else:
                # Form kötüyse ve son maçlarda çok az gol attıysa daha düşük gol olasılığı
                rounded_home_goals = 0
                logger.info(f"Ev sahibi gol beklentisi 1'in altında ({avg_home_goals:.2f}), form düşük ve son maçlarda gol ortalaması {recent_goals_avg:.2f}, 0 gol veriliyor")
        else:
            # Direk yuvarlamak yerine form ve avantajları dikkate al
            # Yüksek gol beklentileri için özel durum (3'ün üzerinde)
            if avg_home_goals >= 3.0:
                # 3 ve üzeri beklentilerde
                if weighted_home_form_points > 0.5 or home_advantage > 1.05:
                    # Form veya ev avantajı iyiyse yukarı yuvarla veya ekstra gol ekle
                    base_goals = int(avg_home_goals)  # Tam kısmı al
                    fraction = avg_home_goals - base_goals  # Ondalık kısmı al
                    
                    if fraction >= 0.4:  # Yüksek ondalık kısım
                        rounded_home_goals = base_goals + 1
                        logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), form iyi, {rounded_home_goals} gol veriliyor")
                    else:
                        rounded_home_goals = base_goals  # Tam değeri kullan
                        logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), {rounded_home_goals} gol veriliyor")
                else:
                    # Form düşükse de daha hassas yuvarla
                    # 1.7'den büyük değerleri yukarı yuvarlıyoruz çünkü 2 gol daha olası
                    if avg_home_goals >= 1.7:
                        rounded_home_goals = int(avg_home_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_home_goals = int(avg_home_goals)  # Aşağı yuvarla
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f} (yüksek), hassas yuvarlamayla {rounded_home_goals} gol veriliyor")
            elif avg_home_goals > 2.5 and avg_home_goals < 3.0:
                # 2.5-3.0 arası değerlerde
                if weighted_home_form_points > 0.6 or home_advantage > 1.1:
                    # Form veya ev avantajı iyiyse 3'e yuvarla
                    rounded_home_goals = 3
                    logger.info(f"Ev sahibi gol beklentisi 2.5-3.0 arasında ({avg_home_goals:.2f}) ve form iyi, 3 gol veriliyor")
                else:
                    # Yoksa 2'ye yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.5-3.0 arasında ({avg_home_goals:.2f}) ama form düşük, 2 gol veriliyor")
            elif avg_home_goals > 1.5 and avg_home_goals < 1.7:
                # 1.5-1.7 arası değerlerde
                if weighted_home_form_points > 0.5 or home_advantage > 1.05:
                    # Form veya ev avantajı iyiyse 2'ye yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 1.5-1.7 arasında ({avg_home_goals:.2f}) ve form iyi, 2 gol veriliyor")
                else:
                    # Yoksa 1'e yuvarla
                    rounded_home_goals = 1
                    logger.info(f"Ev sahibi gol beklentisi 1.5-1.7 arasında ({avg_home_goals:.2f}) ama form düşük, 1 gol veriliyor")
            # 1.7 ve üstü değerleri için daha hassas yuvarlama - sorunumuzu çözen kısım
            elif avg_home_goals >= 1.7 and avg_home_goals < 3.0:
                # Tam sayıya uzaklığı hesapla
                decimal_part = avg_home_goals - int(avg_home_goals)
                
                # 0.3'ten büyükse bir üst sayıya yuvarla (1.7 ve üzeri değerler daha agresif yuvarlanacak)
                if decimal_part >= 0.3:
                    rounded_home_goals = int(avg_home_goals) + 1
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.3'ten büyük olduğu için {rounded_home_goals} olarak yuvarlandı")
                else:
                    rounded_home_goals = int(avg_home_goals)
                    logger.info(f"Ev sahibi gol beklentisi {avg_home_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.3'ten küçük olduğu için {rounded_home_goals} olarak yuvarlandı")
            
            elif avg_home_goals >= 2.0 and avg_home_goals <= 2.5:
                # 2'nin üstündeki değerler için daha yüksek ihtimal ile 2'ye yuvarla
                if avg_home_goals >= 2.25 or weighted_home_form_points > 0.5:
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.0-2.5 arasında ({avg_home_goals:.2f}), 2 gol veriliyor")
                else:
                    # 2.0'a yakın veya form düşükse daha agresif yuvarla
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 2.0-2.5 arasında ({avg_home_goals:.2f}), form hesaba katılarak 2 gol veriliyor")
            
            # 1.0-1.5 arası değerlerde de form bazlı karar ver
            elif avg_home_goals >= 1.0 and avg_home_goals <= 1.5:
                # Eğer 1.35'ten büyükse veya form iyiyse 2'ye yükselterek yuvarla
                if avg_home_goals >= 1.35 or weighted_home_form_points > 0.6:
                    rounded_home_goals = 2
                    logger.info(f"Ev sahibi gol beklentisi 1.0-1.5 arasında ({avg_home_goals:.2f}) ama 1.35+ veya form iyi, 2 gol veriliyor")
                else:
                    # Aksi halde 1'e yuvarla
                    rounded_home_goals = 1
                    logger.info(f"Ev sahibi gol beklentisi 1.0-1.5 arasında ({avg_home_goals:.2f}), 1 gol veriliyor")
            
            # Diğer değerlerde standart yuvarla
            else:
                rounded_home_goals = int(round(avg_home_goals))
                logger.info(f"Ev sahibi gol beklentisi standart yuvarlandı: {avg_home_goals:.2f} -> {rounded_home_goals}")
                
        # Deplasman takımı için
        if avg_away_goals < 1.0:
            # Form ve deplasman performansını değerlendir
            away_form_points = away_form.get('away_performance', {}).get('weighted_form_points', 0.3)
            
            # Son 3 maçtaki gol ortalamasına da bak
            recent_goals_avg = 0
            goals_in_last_matches = 0
            match_count = 0
            
            for match in away_match_data[:3]:
                if not match.get('is_home', True):  # Sadece deplasman maçlarını dikkate al
                    goals_in_last_matches += match.get('goals_scored', 0)
                    match_count += 1
            
            if match_count > 0:
                recent_goals_avg = goals_in_last_matches / match_count
                
            # Son 5 maçta ev sahibine karşı gol atma oranı
            goals_vs_similar = []
            for match in away_match_data[:10]:
                opponent_home_form = None
                opponent_id = None
                if not match.get('is_home', True) and match.get('goals_scored', 0) > 0:
                    # Benzer güçte rakiplere karşı gol atma durumu
                    goals_vs_similar.append(match.get('goals_scored', 0))
            
            avg_vs_similar = sum(goals_vs_similar) / len(goals_vs_similar) if goals_vs_similar else 0
            
            # Form iyi, deplasman avantajı varsa veya son maçlarda gol attıysa
            # Son maç ortalamalarını hesapla
            away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
            if (away_form_points > 0.6 and away_advantage > 1.05) or weighted_away_form_points > 0.8 or away_recent_goals_avg > 0.7 or avg_vs_similar > 0.8:
                # Form iyiyse veya benzer rakiplere karşı gol attıysa daha yüksek gol olasılığı
                rounded_away_goals = 1
                logger.info(f"Deplasman gol beklentisi 1'in altında ({avg_away_goals:.2f}) ama form iyi veya son maçlarda gol ortalaması {away_recent_goals_avg:.2f}, benzer rakiplere karşı {avg_vs_similar:.2f}, 1 gol veriliyor")
            else:
                # Form kötüyse ve son maçlarda çok az gol attıysa daha düşük gol olasılığı
                rounded_away_goals = 0
                # Burada da değişken güncelleme
                away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
                logger.info(f"Deplasman gol beklentisi 1'in altında ({avg_away_goals:.2f}), form düşük ve son maçlarda gol ortalaması {away_recent_goals_avg:.2f}, 0 gol veriliyor")
        else:
            # Değiştirilmiş yuvarlama mantığı - beklenen gol sayısını daha doğru yansıtmak için
            # 3 ve üzeri beklentiler için özel durum
            if avg_away_goals >= 3.0:
                # 3 ve üzeri beklentilerde
                if weighted_away_form_points > 0.5 or away_advantage > 1.05:
                    # Form veya deplasman avantajı iyiyse yukarı yuvarla veya ekstra gol ekle
                    base_goals = int(avg_away_goals)  # Tam kısmı al
                    fraction = avg_away_goals - base_goals  # Ondalık kısmı al
                    
                    if fraction >= 0.4:  # Yüksek ondalık kısım
                        rounded_away_goals = base_goals + 1
                        logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), form iyi, {rounded_away_goals} gol veriliyor")
                    else:
                        rounded_away_goals = base_goals  # Tam değeri kullan
                        logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), {rounded_away_goals} gol veriliyor")
                else:
                    # Form düşükse de daha hassas yuvarla
                    # 1.7'den büyük değerleri yukarı yuvarlıyoruz çünkü 2 gol daha olası
                    if avg_away_goals >= 1.7:
                        rounded_away_goals = int(avg_away_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_away_goals = int(avg_away_goals)  # Aşağı yuvarla
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f} (yüksek), hassas yuvarlamayla {rounded_away_goals} gol veriliyor")
            # 1.8 ve üstü değerleri için daha hassas yuvarlama
            elif avg_away_goals >= 1.8 and avg_away_goals < 3.0:
                # Tam sayıya uzaklığı hesapla
                decimal_part = avg_away_goals - int(avg_away_goals)
                
                # 0.35'ten büyükse bir üst sayıya yuvarla
                if decimal_part >= 0.35:
                    rounded_away_goals = int(avg_away_goals) + 1
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.35'ten büyük olduğu için {rounded_away_goals} olarak yuvarlandı")
                else:
                    rounded_away_goals = int(avg_away_goals)
                    logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, {decimal_part:.2f} ondalık kısmı 0.35'ten küçük olduğu için {rounded_away_goals} olarak yuvarlandı")
            # 1.5-1.8 arası değerlerde form durumuna göre karar ver
            elif avg_away_goals > 1.5 and avg_away_goals < 1.8:
                # Form faktörü ve son maç ortalamalarını kullan
                away_recent_goals_avg = away_recent_avg_goals.get(5, 0)
                if weighted_away_form_points > 0.55 or away_advantage > 1.1 or away_recent_goals_avg > 1.5:
                    # Form iyi veya son maçlarda gol ortalaması yüksekse 2'ye yuvarla
                    rounded_away_goals = 2
                    logger.info(f"Deplasman gol beklentisi 1.5-1.8 arasında ({avg_away_goals:.2f}) ve form iyi, 2 gol veriliyor")
                else:
                    # Form zayıfsa 1'e yuvarla
                    rounded_away_goals = 1
                    logger.info(f"Deplasman gol beklentisi 1.5-1.8 arasında ({avg_away_goals:.2f}) ama form düşük, 1 gol veriliyor")
            else:
                # Diğer değerlerde daha hassas yuvarlama işlemi kullan
                # 1.7'den büyük değerleri yukarı yuvarlıyoruz
                if avg_away_goals >= 0.7:
                    decimal_part = avg_away_goals - int(avg_away_goals)
                    if decimal_part >= 0.7:
                        rounded_away_goals = int(avg_away_goals) + 1  # Yukarı yuvarla
                    else:
                        rounded_away_goals = int(avg_away_goals)  # Aşağı yuvarla
                else:
                    rounded_away_goals = round(avg_away_goals)
                logger.info(f"Deplasman gol beklentisi {avg_away_goals:.2f}, geliştirilmiş yuvarlamayla {rounded_away_goals} olarak belirlendi")
                
        expected_score = f"{rounded_home_goals}-{rounded_away_goals}"
        
        # Önce beklenen skoru kullan, ancak olasılık çok düşükse top 5 içinden seç
        expected_score_prob = exact_scores.get(expected_score, 0) / simulations
        logger.info(f"Beklenen skor {expected_score} olasılığı: %{round(expected_score_prob*100, 2)}")
        
        # Beklenen skoru birkaç farklı yöntemle değerlendir
        expected_score_methods = {
            "rounded_mean": expected_score,  # Beklenen gollerin yuvarlanması
            # Monte Carlo'dan en yüksek olasılıklı skor, ama beklenen gol değerlerine göre makul sınırlardaki bir skoru seç
            "simulation_top": self._select_reasonable_score_from_simulation(top_5_scores, avg_home_goals, avg_away_goals) if top_5_scores else expected_score,
            "outcome_based": "" # En olası maç sonucuna göre en olası skor (aşağıda doldurulacak)
        }
        
        # Maç sonucu tahminini kullanarak skor belirle
        if most_likely_outcome == "HOME_WIN":
            outcome_scores = sorted(same_outcome_scores["HOME_WIN"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
        elif most_likely_outcome == "DRAW":
            outcome_scores = sorted(same_outcome_scores["DRAW"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
        else:  # AWAY_WIN
            outcome_scores = sorted(same_outcome_scores["AWAY_WIN"].items(), key=lambda x: x[1], reverse=True)
            expected_score_methods["outcome_based"] = outcome_scores[0][0] if outcome_scores else expected_score
            
        # Takımların gücünü değerlendirerek skor seçme stratejisini belirle
        home_strength = weighted_home_form_points
        away_strength = weighted_away_form_points
        is_balanced_match = abs(home_strength - away_strength) < 0.2
        is_high_scoring_home = home_recent_avg_goals.get(5, 0) > 1.8
        is_high_scoring_away = away_recent_avg_goals.get(5, 0) > 1.5
        
        # Varsayılan olarak outcome_based yaklaşımı kullan
        most_likely_score = expected_score_methods["outcome_based"]
        
        # Özel durumları ele al
        if is_balanced_match:
            # Dengeli maçlarda simulasyon sonuçlarına daha fazla güven
            if expected_score_prob < 0.07:  # Beklenen skor olasılığı düşükse
                simulation_score = expected_score_methods["simulation_top"]
                simulation_prob = score_histogram.get(simulation_score, 0)
                
                if simulation_prob > expected_score_prob * 1.5:
                    most_likely_score = simulation_score
                    logger.info(f"Dengeli maç, simulasyon tahmini tercih edildi: {simulation_score} (olasılık: %{round(simulation_prob*100, 2)})")
                else:
                    most_likely_score = expected_score_methods["outcome_based"]
                    logger.info(f"Dengeli maç, maç sonucu bazlı tahmin tercih edildi: {most_likely_score}")
        elif is_high_scoring_home and is_high_scoring_away:
            # İki takım da çok gol atıyorsa, daha yüksek skorlu bir tahmin yap
            high_scoring_candidates = []
            for score, prob in score_histogram.items():
                if '-' in score:
                    h, a = map(int, score.split('-'))
                    if h + a >= 3 and prob > 0.05:  # En az 3 gol ve %5 üzeri olasılık
                        high_scoring_candidates.append((score, prob))
            
            if high_scoring_candidates:
                high_scoring_candidates.sort(key=lambda x: x[1], reverse=True)
                most_likely_score = high_scoring_candidates[0][0]
                logger.info(f"Yüksek skorlu maç beklentisi: {most_likely_score} seçildi (olasılık: %{round(high_scoring_candidates[0][1]*100, 2)})")
            else:
                # Yeterli aday yoksa, beklenen skoru kullan
                most_likely_score = expected_score
        else:
            # Takımların gücüne ve maç sonucu tahminlerine dayalı skorları değerlendir
            outcome_score = expected_score_methods["outcome_based"]
            outcome_prob = score_histogram.get(outcome_score, 0)
            
            if outcome_prob > expected_score_prob or outcome_prob > 0.08:  # %8'den yüksek olasılık
                most_likely_score = outcome_score
                logger.info(f"Maç sonucu bazlı tahmin tercih edildi: {outcome_score} (olasılık: %{round(outcome_prob*100, 2)})")
            else:
                # Beklenen goller daha güvenilir, beklenen skoru kullan
                most_likely_score = expected_score
                logger.info(f"Beklenen gollere dayanarak {expected_score} skoru seçildi (olasılık: %{round(expected_score_prob*100, 2)})")
        
        # Özel durum: Eğer goller çok yakınsa ve beraberlik olasılığı yüksekse, beraberlik skorunu değerlendir
        if abs(avg_home_goals - avg_away_goals) < 0.3 and draw_prob > 0.25:
            # Beklenen gol değerlerine göre uygun beraberlik skorunu seç
            # Düşük gol beklentili maçlarda Monte Carlo simülasyon sonuçlarına doğrudan saygı göster
            if avg_home_goals < 1.0 and avg_away_goals < 1.0:
                # İki takım da düşük gol beklentisine sahipse en olası beraberlik skorunu Monte Carlo'dan al
                logger.info(f"Düşük gol beklentili maç için beraberlik skorunu kontrol ediyorum")
                logger.info(f"exact_scores içeriği: {exact_scores}")
                
                # En olası 5 skoru loglama
                top_scores = sorted(exact_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                logger.info(f"En olası 5 skor: {top_scores}")
                
                score_0_0_count = exact_scores.get("0-0", 0)
                score_1_1_count = exact_scores.get("1-1", 0)
                draw_score_0_0_prob = score_0_0_count / simulations if simulations > 0 else 0
                draw_score_1_1_prob = score_1_1_count / simulations if simulations > 0 else 0
                
                logger.info(f"0-0 sayısı: {score_0_0_count}, 1-1 sayısı: {score_1_1_count}, simulations: {simulations}")
                logger.info(f"0-0 olasılığı: {draw_score_0_0_prob}, 1-1 olasılığı: {draw_score_1_1_prob}")
                
                # Ortalama gol değerleri 1'in altındayken, eğer değerler 0.7'den büyükse 1-1'e daha yüksek şans ver
                # Monte Carlo bazen 0-0 veya 1-1 skorları için değer oluşturmuyor olabilir ama skorun mantıklı olması gerekiyor
                if avg_home_goals >= 0.7 and avg_away_goals >= 0.7:
                    logger.info(f"Gol beklentileri 0.7'den yüksek, 1-1 skoru makul bir seçim olabilir")
                    
                    # Monte Carlo değer üretmediyse veya değer çok düşükse, değerleri zorla
                    if score_0_0_count == 0:
                        score_0_0_count = int(simulations * 0.05)  # %5 varsayılan değer
                        logger.info(f"0-0 için Monte Carlo değeri bulunamadı, varsayılan değer eklendi: {score_0_0_count}")
                    
                    if score_1_1_count == 0:
                        score_1_1_count = int(simulations * 0.10)  # %10 varsayılan değer
                        logger.info(f"1-1 için Monte Carlo değeri bulunamadı, varsayılan değer eklendi: {score_1_1_count}")
                    
                    # Değerleri yeniden hesapla
                    draw_score_0_0_prob = score_0_0_count / simulations if simulations > 0 else 0
                    draw_score_1_1_prob = score_1_1_count / simulations if simulations > 0 else 0
                    
                    # Gol beklentileri 0.7'den büyükse, 1-1'e daha yüksek şans ver
                    if score_1_1_count > 0 and draw_score_1_1_prob > draw_score_0_0_prob * 0.8:
                        most_likely_score = "1-1"
                        logger.info(f"Monte Carlo simülasyonu ve gol beklentileri göz önüne alınarak 1-1 skoru seçildi (olasılık: %{round(draw_score_1_1_prob*100, 2)})")
                        # Akışı kesmiyoruz, sadece değeri güncelliyoruz
                
                # Monte Carlo simülasyonunun sonucuna doğrudan saygı göster - hangi skor daha olasıysa onu seç
                if draw_score_0_0_prob >= draw_score_1_1_prob:
                    most_likely_score = "0-0"
                    logger.info(f"Monte Carlo simülasyonu sonucuna göre düşük gol beklentilerinde ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) 0-0 skoru seçildi (olasılık: %{round(draw_score_0_0_prob*100, 2)})")
                else:
                    most_likely_score = "1-1"
                    logger.info(f"Monte Carlo simülasyonu sonucuna göre düşük gol beklentilerinde ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) 1-1 skoru seçildi (olasılık: %{round(draw_score_1_1_prob*100, 2)})")
            else:
                # Normal gol beklentileri için standart değerlendirme
                likely_draw_score = int(round((avg_home_goals + avg_away_goals) / 2))
                draw_score = f"{likely_draw_score}-{likely_draw_score}"
                draw_score_prob = exact_scores.get(draw_score, 0) / simulations
                
                if draw_score_prob > 0.1:  # %10'dan yüksek olasılık
                    most_likely_score = draw_score
                    logger.info(f"Gol beklentileri çok yakın ({avg_home_goals:.2f} vs {avg_away_goals:.2f}) ve beraberlik olasılığı yüksek (%{round(draw_prob*100, 2)}), {draw_score} skoru seçildi")
        
        # GERÇEKÇI SKORLAR IÇIN GERÇEKLİK KONTROLÜ - Skor çok abartılı ise beklenen gollere göre düzeltme yap
        home_score, away_score = map(int, most_likely_score.split('-'))
        
        # DAHA SIKI GERÇEKLİK KONTROLÜ - Son maçlardaki gerçek gol ortalamaları
        recent_home_goals = 0
        recent_away_goals = 0
        home_matches_count = min(5, len(home_match_data))
        away_matches_count = min(5, len(away_match_data))
        
        if home_matches_count > 0:
            recent_home_goals = sum(match.get('goals_scored', 0) for match in home_match_data[:home_matches_count]) / home_matches_count
        
        if away_matches_count > 0:
            recent_away_goals = sum(match.get('goals_scored', 0) for match in away_match_data[:away_matches_count]) / away_matches_count
            
        # Beklenen gollere göre makul skor sınırları - iyileştirilmiş algoritma
        # Düşük beklenen gol değerleri için daha keskin sınırlar kullan
        # Beklenen goller 1'in altındaysa, makul skor maksimum 1 olmalı, 1-2 arasındaysa maksimum 2 olmalı
        if avg_home_goals < 1.0:
            reasonable_home_max = min(1, round(max(avg_home_goals, recent_home_goals)))
        elif avg_home_goals < 2.0:
            reasonable_home_max = min(2, round(max(avg_home_goals * 1.2, recent_home_goals * 1.2)))
        else:
            reasonable_home_max = min(3, round(max(avg_home_goals * 1.3, recent_home_goals * 1.3)))
            
        if avg_away_goals < 1.0:
            reasonable_away_max = min(1, round(max(avg_away_goals, recent_away_goals)))
        elif avg_away_goals < 1.8:
            reasonable_away_max = min(2, round(max(avg_away_goals * 1.2, recent_away_goals * 1.2)))
        elif avg_away_goals < 2.5:
            reasonable_away_max = min(3, round(max(avg_away_goals * 1.2, recent_away_goals * 1.2)))
        else:
            reasonable_away_max = min(3, round(max(avg_away_goals * 1.3, recent_away_goals * 1.3)))
        
        logger.info(f"Makul skor sınırları: Ev={reasonable_home_max}, Deplasman={reasonable_away_max} (son maç gol ort: {recent_home_goals:.2f}-{recent_away_goals:.2f})")
        
        # Gerçekçilik kontrolü - Eğer skor çok abartılı ise düzelt
        if home_score > reasonable_home_max or away_score > reasonable_away_max:
            logger.warning(f"Seçilen skor {most_likely_score} çok abartılı! Beklenen goller: {avg_home_goals:.2f}-{avg_away_goals:.2f}, Son maç gol ort: {recent_home_goals:.2f}-{recent_away_goals:.2f}")
            
            # Beklenen gollere ve son maçlardaki ortalamalara göre daha gerçekçi bir skor belirle
            reasonable_home = min(home_score, reasonable_home_max)
            reasonable_away = min(away_score, reasonable_away_max)
            
            # Sonuç tipini koru (ev sahibi kazanır, deplasman kazanır, beraberlik)
            if home_score > away_score:  # Ev sahibi galibiyet
                if reasonable_home <= reasonable_away:
                    reasonable_home = reasonable_away + 1
            elif away_score > home_score:  # Deplasman galibiyet
                if reasonable_away <= reasonable_home:
                    reasonable_away = reasonable_home + 1
            else:  # Beraberlik
                reasonable_home = reasonable_away = min(reasonable_home, reasonable_away)
            
            adjusted_score = f"{reasonable_home}-{reasonable_away}"
            logger.info(f"Skor beklenen gollere ve son maç ortalamalarına göre düzeltildi: {most_likely_score} -> {adjusted_score}")
            most_likely_score = adjusted_score
        # ÖNEMLİ: Tutarlılık kontrolü! En olası sonuçla kesin skor tutarlı mı?
        # Beklenen golleri de hesaba katarak en olası sonucu belirle
        most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
        score_outcome = self._get_outcome_from_score(most_likely_score)
        
        # Eğer kesin skor ve tahmin edilen sonuç tutarsızsa, skoru düzelt
        if most_likely_outcome != score_outcome:
            logger.warning(f"Tutarsızlık tespit edildi! Maç sonucu {most_likely_outcome} ama skor tahmini {most_likely_score} ({score_outcome})")
            
            # Beklenen gol değerlerine göre makul skor sınırları belirle
            max_home_score = 1  # Varsayılan
            max_away_score = 1  # Varsayılan
            
            # Ev sahibi için sınır
            if avg_home_goals < 1.0:
                max_home_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
            elif avg_home_goals < 1.8:
                max_home_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
            elif avg_home_goals < 2.5:
                max_home_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
            else:
                max_home_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
                
            # Deplasman için sınır
            if avg_away_goals < 1.0:
                max_away_score = 1  # 1'in altında beklenen gol için maksimum 1 gol
            elif avg_away_goals < 1.8:
                max_away_score = 2  # 1-1.8 arası beklenen gol için maksimum 2 gol
            elif avg_away_goals < 2.5:
                max_away_score = 3  # 1.8-2.5 arası beklenen gol için maksimum 3 gol
            else:
                max_away_score = 4  # 2.5'ten yüksek beklenen gol için maksimum 4 gol
            
            # En yüksek olasılıklı skoru bul ancak sonuç kısıtlaması ve makul skor sınırları ile
            candidate_scores = []
            
            if most_likely_outcome == "HOME_WIN":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home > away and home <= max_home_score and away <= max_away_score:  # Ev sahibi galibiyeti + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Ev galibiyeti), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir ev galibiyet skoru oluştur
                    new_home_score = min(2, max_home_score)  # En az 1, en fazla makul sınır
                    new_away_score = min(1, max_away_score)  # En az 0, en fazla makul sınır
                    if new_home_score <= new_away_score:  # Ev sahibi kazanmalı
                        new_home_score = new_away_score + 1
                    most_likely_score = f"{new_home_score}-{new_away_score}"
                    logger.warning(f"Makul sınırlarda bir ev sahibi galibiyet skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
            
            elif most_likely_outcome == "AWAY_WIN":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home < away and home <= max_home_score and away <= max_away_score:  # Deplasman galibiyeti + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Deplasman galibiyeti), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir deplasman galibiyet skoru oluştur
                    new_home_score = min(1, max_home_score)  # En az 0, en fazla makul sınır
                    new_away_score = min(2, max_away_score)  # En az 1, en fazla makul sınır
                    if new_home_score >= new_away_score:  # Deplasman kazanmalı
                        new_away_score = new_home_score + 1
                    most_likely_score = f"{new_home_score}-{new_away_score}"
                    logger.warning(f"Makul sınırlarda bir deplasman galibiyet skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
            
            elif most_likely_outcome == "DRAW":
                for score, count in sorted(exact_scores.items(), key=lambda x: x[1], reverse=True):
                    if '-' in score:
                        home, away = map(int, score.split('-'))
                        if home == away and home <= max_home_score:  # Beraberlik + makul sınırlar
                            candidate_scores.append((score, count))
                
                if candidate_scores:
                    # En yüksek olasılıklı makul skoru seç
                    most_likely_score = candidate_scores[0][0]
                    logger.info(f"Tutarlılık için skor {most_likely_score} olarak güncellendi (Beraberlik), makul sınırlar: {max_home_score}-{max_away_score}")
                else:
                    # Makul sınırlarda bir beraberlik skoru oluştur
                    new_score = min(1, max_home_score)  # En az 0, en fazla makul sınır
                    most_likely_score = f"{new_score}-{new_score}"
                    logger.warning(f"Makul sınırlarda bir beraberlik skoru bulunamadı, yeni skor oluşturuldu: {most_likely_score}")
        
        # Kesin skoru belirle
        bet_predictions['exact_score'] = most_likely_score
        
        # Kesin skor üzerinden maç sonucu, ÜST/ALT ve KG VAR/YOK tahminlerini güncelle
        score_parts = most_likely_score.split('-')
        if len(score_parts) == 2:
            home_score, away_score = int(score_parts[0]), int(score_parts[1])
            total_goals = home_score + away_score
            
            # Gol beklentileri ile skorlar arasındaki uyumu kontrol et
            home_score_diff = abs(home_score - avg_home_goals)
            away_score_diff = abs(away_score - avg_away_goals)
            
            # Skor beklentiler ile uyumlu değilse uyarı logla
            if home_score_diff > 1.0 or away_score_diff > 1.0:
                logger.warning(f"Seçilen skor {most_likely_score} ile gol beklentileri {avg_home_goals:.2f}-{avg_away_goals:.2f} arasında büyük fark var!")
            else:
                logger.info(f"Seçilen skor {most_likely_score} gol beklentilerine {avg_home_goals:.2f}-{avg_away_goals:.2f} yakın ve uyumlu.")
            
            # Maç sonucu - Sadece kesin skordan belirle (tutarlılık için)
            if home_score > away_score:
                bet_predictions['match_result'] = 'MS1'  # Ev sahibi kazanır
                logger.info(f"Kesin skor {home_score}-{away_score} temel alınarak ev sahibi kazanır tahmini yapıldı")
            elif away_score > home_score:
                bet_predictions['match_result'] = 'MS2'  # Deplasman kazanır
                logger.info(f"Kesin skor {home_score}-{away_score} temel alınarak deplasman kazanır tahmini yapıldı")
            else:
                bet_predictions['match_result'] = 'X'  # Beraberlik
                logger.info(f"Kesin skor {home_score}-{away_score} temel alınarak beraberlik tahmini yapıldı")
                
                # Skor farkına göre olasılığı belirle
                goal_diff = home_score - away_score
                expected_goal_diff = avg_home_goals - avg_away_goals
                
                # Olasılığı beklenen gol farkıyla dengele
                base_prob = 0.5 + (expected_goal_diff * 0.15)  # Beklenen gol farkı artıkça olasılık artar
                
                # Gerçekçi olasılık ayarlaması
                if goal_diff == 1:  # Minimal fark
                    home_win_prob = max(home_win_prob, min(0.65, base_prob))
                elif goal_diff == 2:  # Orta fark
                    home_win_prob = max(home_win_prob, min(0.75, base_prob + 0.1))
                else:  # Büyük fark
                    home_win_prob = max(home_win_prob, min(0.85, base_prob + 0.2))
                    
                # Diğer olasılıkları dengele
                remaining = 1.0 - home_win_prob
                draw_prob = remaining * 0.6
                away_win_prob = remaining * 0.4
                
                # Beklenen gollere göre beraberlik olasılığını değerlendir
                if abs(expected_goal_diff) < 0.3:  # Beklenen goller çok yakınsa beraberlik mantıklı
                    logger.info(f"Beraberlik tahmini beklenen gollerle ({abs(expected_goal_diff):.2f} fark) uyumlu")
                else:
                    # Beklenen gol farkı büyükse, beraberlik tahmini şüpheli olabilir
                    logger.warning(f"Dikkat: Beraberlik skoru ({home_score}-{away_score}) beklenen gollerde önemli fark ({expected_goal_diff:.2f}) var")
                
                # Skor yüksekliğine göre olasılığı ayarla (yüksek skorlu beraberlikler daha nadir)
                if total_goals <= 2:  # 0-0, 1-1
                    draw_prob = max(draw_prob, min(0.65, 0.5 + 0.15 * (1 - abs(expected_goal_diff))))
                else:  # 2-2, 3-3, vs. 
                    draw_prob = max(draw_prob, min(0.55, 0.5 + 0.05 * (1 - abs(expected_goal_diff))))
                    
                # Diğer olasılıkları dengele
                remaining = 1.0 - draw_prob
                # Beklenen gol farkına göre kalan olasılıkları dağıt
                if expected_goal_diff > 0:
                    home_win_prob = remaining * 0.7
                    away_win_prob = remaining * 0.3
                else:
                    home_win_prob = remaining * 0.3
                    away_win_prob = remaining * 0.7
            
            # KG VAR/YOK KARARI SKOR BAZLI OVERRIDE DEVRE DIŞI - MONTE CARLO KARARI KORUNUYOR
            # Bu bölüm kesin skora göre KG VAR/YOK kararını değiştiriyordu
            # Artık sadece log kayıtları tutuyoruz, kararı değiştirmiyoruz
            if home_score > 0 and away_score > 0:
                logger.info(f"Skor bazlı: {most_likely_score} - KG VAR skoru ama Monte Carlo kararı korunuyor")
                # bet_predictions['both_teams_to_score'] = 'KG VAR'  # BU SATIR DEVRE DIŞI
            else:
                logger.info(f"Skor bazlı: {most_likely_score} - KG YOK skoru ama Monte Carlo kararı korunuyor")
                # bet_predictions['both_teams_to_score'] = 'KG YOK'  # BU SATIR DEVRE DIŞI
                
                # Geçmiş maçlardaki KG YOK oranlarını değerlendir
                kg_yok_home_matches = sum(1 for match in home_match_data[:5] if match.get('goals_scored', 0) == 0 or match.get('goals_conceded', 0) == 0)
                kg_yok_away_matches = sum(1 for match in away_match_data[:5] if match.get('goals_scored', 0) == 0 or match.get('goals_conceded', 0) == 0)
                kg_yok_rate = (kg_yok_home_matches + kg_yok_away_matches) / 10 if len(home_match_data) >= 5 and len(away_match_data) >= 5 else 0.5
                
                # Gol beklentilerine ve geçmiş KG YOK oranlarına göre olasılığı ayarla
                if avg_home_goals < 1.0 or avg_away_goals < 0.8:
                    base_prob = 0.10  # Düşük gol beklentisinde daha düşük KG VAR olasılığı
                else:
                    base_prob = 0.15  # Kesin skordan biliyoruz
                    
                # Geçmiş maçlardaki KG YOK oranlarını da dikkate al
                kg_var_adjusted_prob = base_prob * 0.8 + (1 - kg_yok_rate) * 0.2
                logger.info(f"KG YOK tahmini: Geçmiş maçlarda KG YOK oranı: {kg_yok_rate:.2f}, ayarlanmış KG VAR olasılığı: {kg_var_adjusted_prob:.2f}")
            
            # 2.5 ÜST/ALT - OVERRIDE DEVRE DIŞI - Monte Carlo sonuçları korunuyor
            # Kesin skor tabanlı override mekanizması kaldırıldı
            # bet_predictions['over_2_5_goals'] artık sadece Monte Carlo simülasyonundan belirleniyor
            logger.info(f"2.5 Üst/Alt override atlandı - Monte Carlo kararı korunuyor: {bet_predictions['over_2_5_goals']}")
            
            # 3.5 ÜST/ALT - OVERRIDE DEVRE DIŞI - Monte Carlo sonuçları korunuyor
            # Kesin skor tabanlı override mekanizması kaldırıldı
            # bet_predictions['over_3_5_goals'] artık sadece Monte Carlo simülasyonundan belirleniyor
            logger.info(f"3.5 Üst/Alt override atlandı - Monte Carlo kararı korunuyor: {bet_predictions['over_3_5_goals']}")
            
            # Kesin skordan doğrudan sonuç belirleme
            match_outcome_from_score = self._get_outcome_from_score(most_likely_score)
            
            # Kesin skor ile maç sonucu arasındaki tutarlılığı kontrol et
            if '-' in str(most_likely_score):
                try:
                    h_goals, a_goals = map(int, str(most_likely_score).split('-'))
                    if h_goals == a_goals:
                        # Eşit skor - sadece beraberlik olasılığı gerçekten en yüksekse DRAW olarak ata
                        if draw_prob > home_win_prob and draw_prob > away_win_prob:
                            match_outcome_from_score = "DRAW"
                            logger.info(f"Eşit skor {most_likely_score} ve en yüksek beraberlik olasılığı ({draw_prob:.1%}) nedeniyle sonuç DRAW")
                        else:
                            # Eşit skor ama beraberlik olasılığı en yüksek değil - olasılıklara göre karar ver
                            if home_win_prob > away_win_prob:
                                match_outcome_from_score = "HOME_WIN"
                                logger.info(f"Eşit skor {most_likely_score} ama ev sahibi olasılığı daha yüksek ({home_win_prob:.1%})")
                            else:
                                match_outcome_from_score = "AWAY_WIN"
                                logger.info(f"Eşit skor {most_likely_score} ama deplasman olasılığı daha yüksek ({away_win_prob:.1%})")
                except Exception as e:
                    logger.error(f"Skor kontrol hatası: {str(e)}")
            
            # Bu değerleri saklıyoruz, prediction değişkeni daha sonra tanımlandığında kullanmak için
            # prediction değişkeni 3854. satırda tanımlanıyor, öncesinde kullanmak hata verir
            saved_most_likely_score = most_likely_score
            saved_match_outcome = match_outcome_from_score
            
            logger.info(f"Kesin skor {most_likely_score} (gol beklentileri: {avg_home_goals:.2f}-{avg_away_goals:.2f}) esas alınarak tüm tahminler güncellendi")
        
        # Üçüncü adım: Maç sonucu ile diğer tahminler arasındaki tutarlılığı sağla
        match_result = bet_predictions['match_result']
        
        # TUTARLILIK KONTROLLERI DEVRE DIŞI - KG VAR/YOK KARARINI KORUYORUZ
        # Bu bölüm 2.5 ALT ve KG VAR uyumsuzluğu nedeniyle KG YOK'a zorluyordu
        if False:  # Bu blok artık çalışmayacak
            # İstisnai durum: 1-1 skoru
            if bet_predictions['exact_score'] != '1-1':
                # Hangisinin olasılığı daha yüksekse ona göre düzelt
                if False:  # over_25_adjusted_prob > kg_var_adjusted_prob:
                    bet_predictions['both_teams_to_score'] = 'KG YOK'
                    logger.info("Bu log artık görünmeyecek")
                    # Skoru da güncelle
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-0'
                    elif match_result == 'AWAY_WIN':
                        bet_predictions['exact_score'] = '0-2'
                    else:  # DRAW
                        bet_predictions['exact_score'] = '0-0'
                else:
                    bet_predictions['over_2_5_goals'] = '2.5 ÜST'
                    logger.info("Bu log da artık görünmeyecek")
                    # Skoru da güncelle
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-1'
                    elif match_result == 'AWAY_WIN':
                        bet_predictions['exact_score'] = '1-2'
                    else:  # DRAW
                        bet_predictions['exact_score'] = '1-1'
        
        # 2. MS1/MS2 ve KG YOK arasındaki uyumsuzluk - skor kontrolü
        if (match_result in ['HOME_WIN', 'AWAY_WIN']) and bet_predictions['both_teams_to_score'] == 'KG YOK':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score > 0 and away_score > 0:
                    # Tutarsızlık var, düzelt
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = f"{home_score}-0"
                    else:  # AWAY_WIN
                        bet_predictions['exact_score'] = f"0-{away_score}"
                    logger.info(f"Mantık düzeltmesi: {match_result} ve KG YOK tutarsızlık - skor güncellendi")
        
        # 3. MS1/MS2 ve KG VAR arasındaki uyumsuzluk - skor kontrolü
        if (match_result in ['HOME_WIN', 'AWAY_WIN']) and bet_predictions['both_teams_to_score'] == 'KG VAR':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score == 0 or away_score == 0:
                    # Tutarsızlık var, düzelt
                    if match_result == 'HOME_WIN':
                        bet_predictions['exact_score'] = '2-1'
                    else:  # AWAY_WIN
                        bet_predictions['exact_score'] = '1-2'
                    logger.info(f"Mantık düzeltmesi: {match_result} ve KG VAR tutarsızlık - skor güncellendi")
        
        # 4. DRAW ve skor uyumsuzluğu
        if match_result == 'DRAW':
            score_parts = bet_predictions['exact_score'].split('-')
            if len(score_parts) == 2:
                home_score, away_score = int(score_parts[0]), int(score_parts[1])
                if home_score != away_score:
                    # Beraberlik skoru değil, düzelt
                    if bet_predictions['both_teams_to_score'] == 'KG VAR':
                        bet_predictions['exact_score'] = '1-1'
                    else:
                        bet_predictions['exact_score'] = '0-0'
                    logger.info("Mantık düzeltmesi: DRAW ve skor uyumsuzluğu - skor güncellendi")
        
        # 5. İlk yarı/maç sonu düzeltmesi
        # Kesin skor üzerinden en olası ilk yarı skorunu tahmin et
        score_parts = bet_predictions['exact_score'].split('-')
        if len(score_parts) == 2:
            home_score, away_score = int(score_parts[0]), int(score_parts[1])
            # İlk yarıda genellikle toplam gollerin %40'ı atılır
            expected_ht_home = round(home_score * 0.4)
            expected_ht_away = round(away_score * 0.4)
            
            # İlk yarı sonucu
            ht_result = "X"
            if expected_ht_home > expected_ht_away:
                ht_result = "MS1"
            elif expected_ht_away > expected_ht_home:
                ht_result = "MS2"
            
            # Maç sonu sonucu
            ft_result = bet_predictions['match_result']
            
            # İlk yarı/maç sonu kombinasyonu artık kullanılmıyor - hibrit model kaldırıldı
            # Not: Sürpriz butonu için İY/MS tahmini api_routes.py içindeki get_htft_prediction() API çağrısıyla yapılıyor
        
        # Rakip gücü analizi yap
        opponent_analysis = self.analyze_opponent_strength(home_form, away_form)
        logger.info(f"Rakip gücü analizi: Göreceli güç = {opponent_analysis['relative_strength']:.2f}")
        
        # H2H analizini yap
        h2h_analysis = self.analyze_head_to_head(home_team_id, away_team_id, home_team_name, away_team_name)
        
        if h2h_analysis and h2h_analysis['total_matches'] > 0:
            logger.info(f"H2H analizi: {h2h_analysis['home_wins']}-{h2h_analysis['draws']}-{h2h_analysis['away_wins']} ({h2h_analysis['total_matches']} maç)")
            
            # H2H analizine dayanarak maç sonucu olasılıklarını ayarla
            h2h_home_win_rate = h2h_analysis['home_wins'] / h2h_analysis['total_matches']
            h2h_draw_rate = h2h_analysis['draws'] / h2h_analysis['total_matches']
            h2h_away_win_rate = h2h_analysis['away_wins'] / h2h_analysis['total_matches']
            
            # H2H analizi ile mevcut tahminleri birleştir (20% H2H, 80% mevcut tahmin)
            if h2h_analysis['total_matches'] >= 3:  # En az 3 H2H maç varsa
                h2h_weight = 0.3  # %30 ağırlık (önceki: %20)
                home_win_prob = home_win_prob * (1 - h2h_weight) + h2h_home_win_rate * h2h_weight
                draw_prob = draw_prob * (1 - h2h_weight) + h2h_draw_rate * h2h_weight
                away_win_prob = away_win_prob * (1 - h2h_weight) + h2h_away_win_rate * h2h_weight
                
                logger.info(f"H2H analizi sonrası MS olasılıkları güncellendi: MS1={home_win_prob:.2f}, X={draw_prob:.2f}, MS2={away_win_prob:.2f}")
            
            # H2H'taki ortalama golleri de değerlendir
            if h2h_analysis['total_matches'] >= 3:  # En az 3 H2H maç varsa
                h2h_goals_weight = 0.25  # %25 ağırlık (önceki: %15)
                avg_home_goals = avg_home_goals * (1 - h2h_goals_weight) + h2h_analysis['avg_home_goals'] * h2h_goals_weight
                avg_away_goals = avg_away_goals * (1 - h2h_goals_weight) + h2h_analysis['avg_away_goals'] * h2h_goals_weight
                
                logger.info(f"H2H analizi sonrası gol beklentileri güncellendi: Ev={avg_home_goals:.2f}, Deplasman={avg_away_goals:.2f}")
                
                # Kesin skor tahminini güncelle
                if abs(avg_home_goals - avg_away_goals) < 0.3 and most_likely_outcome != "DRAW":
                    # Gol beklentileri yakın ama beraberlik tahmini yoksa, yeniden değerlendir
                    logger.info(f"H2H verilerine göre skor yeniden değerlendiriliyor. Gol beklentileri çok yakın, beraberlik olasılığı arttırılıyor.")
                    
                    # Beraberlik olasılığını artırırken gol beklentilerine dayalı mantık kontrolü ekle
                    total_expected_goals = avg_home_goals + avg_away_goals
                    
                    # Eğer toplam gol beklentisi 3.5'ten büyükse, beraberlik olasılığını çok fazla artırma
                    if total_expected_goals > 3.5:
                        draw_prob = min(max(draw_prob, home_win_prob * 0.7, away_win_prob * 0.7), 0.4)
                        logger.info(f"Yüksek gol beklentisi ({total_expected_goals:.2f}) için beraberlik olasılığı sınırlandırıldı: {draw_prob:.2f}")
                    else:
                        draw_prob = max(draw_prob, home_win_prob, away_win_prob) * 1.1
                    
                    home_win_prob = (1 - draw_prob) * (home_win_prob / (home_win_prob + away_win_prob)) if (home_win_prob + away_win_prob) > 0 else 0.25
                    away_win_prob = (1 - draw_prob) * (away_win_prob / (home_win_prob + away_win_prob)) if (home_win_prob + away_win_prob) > 0 else 0.25
                    
                    # Olasılıkları normalize et
                    total = home_win_prob + draw_prob + away_win_prob
                    if total > 0:
                        home_win_prob /= total
                        draw_prob /= total
                        away_win_prob /= total
                        
                    # En olası sonucu güncelle - beklenen golleri de hesaba katarak
                    most_likely_outcome = self._get_most_likely_outcome(home_win_prob, draw_prob, away_win_prob, avg_home_goals, avg_away_goals)
                    
                    # Kesin skoru H2H verisi kullanarak güncelle, ANCAK gol beklentileriyle tutarlı olmalı
                    if most_likely_outcome == "DRAW":
                        # Toplam gol beklentisine göre beraberlik skoru belirle
                        total_expected_goals = avg_home_goals + avg_away_goals
                        
                        if total_expected_goals > 3.5:
                            most_likely_score = "2-2"  # Yüksek skorlu beraberlik
                            logger.info(f"Yüksek gol beklentisi ({total_expected_goals:.2f}) için beraberlik skoru 2-2 belirlendi")
                        elif total_expected_goals > 2.0:
                            most_likely_score = "1-1"  # Orta skorlu beraberlik
                            logger.info(f"Orta gol beklentisi ({total_expected_goals:.2f}) için beraberlik skoru 1-1 belirlendi")
                        elif total_expected_goals > 0.8:
                            most_likely_score = "1-1"  # Düşük-orta skorlu beraberlik
                        else:
                            most_likely_score = "0-0"  # Çok düşük skorlu beraberlik
                            
                        logger.info(f"H2H verilerine göre kesin skor güncellendi: {most_likely_score}")
                        
                        # Bahis tahminlerini güncelle
                        bet_predictions['exact_score'] = most_likely_score
                        bet_predictions['match_result'] = 'DRAW'  # Beraberlik skorları için
                        logger.info(f"H2H sonrası kesin skor {most_likely_score} için maç sonucu DRAW olarak güncellendi")
                        # Not: prediction değişkeni henüz tanımlanmadı, en son saved_match_outcome değişkenine DRAW atanmalı
                        saved_match_outcome = "DRAW"
                        
                        # Olasılıkları dengele
                        draw_prob = max(draw_prob, 0.40)
                        remainder = 1.0 - draw_prob
                        home_win_prob = remainder * 0.5
                        away_win_prob = remainder * 0.5
        
        # Rakip analizi sadece log bırakıyor, skor değiştirilmiyor
        if opponent_analysis['relative_strength'] > 0.6:  # Ev sahibi daha güçlüyse
            # Ev sahibi güç farkı fazlaysa ve ev galibiyeti tahmin ediliyorsa skor farkını loglama (değiştirme)
            if most_likely_outcome == "HOME_WIN":
                home_score, away_score = map(int, most_likely_score.split('-'))
                if home_score - away_score == 1:
                    logger.info(f"Rakip analizi sonrası skor farkı artışı önerildi (ev sahibi daha güçlü): {home_score+1}-{away_score}")
        elif opponent_analysis['relative_strength'] < 0.4:  # Deplasman daha güçlüyse
            # Deplasman güç farkı fazlaysa ve deplasman galibiyeti tahmin ediliyorsa skor farkını loglama (değiştirme)
            if most_likely_outcome == "AWAY_WIN":
                home_score, away_score = map(int, most_likely_score.split('-'))
                if away_score - home_score == 1:
                    logger.info(f"Rakip analizi sonrası skor farkı artışı önerildi (deplasman daha güçlü): {home_score}-{away_score+1}")
        
        # Son olarak, olasılıkları yeniden normalize et
        total_prob = home_win_prob + draw_prob + away_win_prob
        if total_prob > 0:
            home_win_prob = home_win_prob / total_prob
            draw_prob = draw_prob / total_prob
            away_win_prob = away_win_prob / total_prob

        # En yüksek olasılıklı tahmini bul
        # Not: İY/MS tahmini artık kullanılmıyor (hibrit model kaldırıldı)
        most_confident_bet = max(bet_probabilities, key=bet_probabilities.get)
        # Tahmin sonuçlarını hazırla
        prediction = {
            'match': f"{home_team_name} vs {away_team_name}",
            'home_team': {
                'id': home_team_id,
                'name': home_team_name,
                'form': home_form,
                'form_periods': {
                    'last_3': home_form_periods['last_3'],
                    'last_6': home_form_periods['last_6'],
                    'last_9': home_form_periods['last_9']
                }
            },
            'away_team': {
                'id': away_team_id,
                'name': away_team_name,
                'form': away_form,
                'form_periods': {
                    'last_3': away_form_periods['last_3'],
                    'last_6': away_form_periods['last_6'],
                    'last_9': away_form_periods['last_9']
                }
            },
            'enhanced_factors': enhanced_factors if 'enhanced_factors' in locals() else {},
            'head_to_head': h2h_analysis if h2h_analysis else {
                'home_wins': 0,
                'away_wins': 0,
                'draws': 0, 
                'total_matches': 0,
                'avg_home_goals': 0,
                'avg_away_goals': 0,
                'recent_matches': []
            },
            'opponent_analysis': opponent_analysis,
            'predictions': {
                'home_win_probability': round(home_win_prob * 100, 2),
                'draw_probability': round(draw_prob * 100, 2),
                'away_win_probability': round(away_win_prob * 100, 2),
                'expected_goals': {
                    'home': round(avg_home_goals, 2),
                    'away': round(avg_away_goals, 2)
                },
                'confidence': round(self._calculate_confidence(home_form, away_form, bet_predictions, {
                    'total_simulations': 10000,
                    'outcome_probabilities': {
                        'home_win': home_win_prob,
                        'draw': draw_prob,
                        'away_win': away_win_prob
                    }
                }) * 100, 1),
                # ÖNEMLİ: Maç sonucu kesin skora göre belirleniyor (tutarlılık için)
                # Ek debug bilgileri ekle - hangi skor kullanılıyor?
                'debug_exact_score_used': bet_predictions['exact_score'],
                
                # Kesin skor ve maç sonucu mutlaka birbirine uyumlu olmalı
                'exact_score': bet_predictions['exact_score'],
                
                # KRİTİK: Skoru ve sonucu tutarlı hale getir
                # Önce skordan sonucu türet (tutarlılık için)
                'most_likely_outcome': self._get_outcome_from_score(bet_predictions['exact_score']),
                'match_outcome': self._get_outcome_from_score(bet_predictions['exact_score']),
                
                # Bu noktada 'saved_most_likely_score' ve 'saved_match_outcome' değişkenlerini de kullanabiliriz
                # Ama mevcut bet_predictions['exact_score'] genellikle daha fazla mantık doğrulamasından geçmiştir
                'betting_predictions': {
                    'both_teams_to_score': {
                        'prediction': 'KG VAR' if kg_var_adjusted_prob > 0.5 else 'KG YOK',
                        'probability': round(kg_var_adjusted_prob * 100 if kg_var_adjusted_prob > 0.5 else (1 - kg_var_adjusted_prob) * 100, 2)
                    },
                    'over_2_5_goals': {
                        'prediction': '2.5 ÜST' if bet_predictions['over_2_5_goals'] == '2.5 ÜST' else '2.5 ALT',
                        'probability': round(min(max(over_2_5_goals_prob * 100 if bet_predictions['over_2_5_goals'] == '2.5 ÜST' else (1 - over_2_5_goals_prob) * 100, 5.0), 95.0), 2)
                    },
                    'over_3_5_goals': {
                        'prediction': '3.5 ÜST' if bet_predictions['over_3_5_goals'] == '3.5 ÜST' else '3.5 ALT',
                        'probability': round(min(max(over_3_5_goals_prob * 100 if bet_predictions['over_3_5_goals'] == '3.5 ÜST' else (1 - over_3_5_goals_prob) * 100, 5.0), 95.0), 2)
                    },
                    'exact_score': {
                        'prediction': bet_predictions['exact_score'],
                        'probability': round(bet_probabilities['exact_score'] * 100, 2)
                    },
                    # İY/MS tahmini kaldırıldı - artık sadece sürpriz butonu ile api_routes.py içindeki get_htft_prediction() API çağrısı üzerinden erişilebilir
                    # İlk gol, korner ve kart tahminleri kaldırıldı
                },
                'neural_predictions': {
                    'home_goals': round(neural_home_goals, 2),
                    'away_goals': round(neural_away_goals, 2),
                    'combined_model': {
                        'home_goals': round(expected_home_goals, 2),
                        'away_goals': round(expected_away_goals, 2)
                    }
                },
                'raw_metrics': {
                    'expected_home_goals': round(avg_home_goals, 2),
                    'expected_away_goals': round(avg_away_goals, 2),
                    'p_home_scores': round(p_home_scores * 100, 2),
                    'p_away_scores': round(p_away_scores * 100, 2),
                    'expected_total_goals': round(expected_total_goals, 2),
                    'form_weights': {
                        'last_5_matches': weight_last_5,
                        'last_10_matches': weight_last_10,
                        'last_21_matches': weight_last_21
                    },
                    'weighted_form': {
                        'home_weighted_goals': round(weighted_home_goals, 2),
                        'home_weighted_form': round(weighted_home_form_points, 2),
                        'away_weighted_goals': round(weighted_away_goals, 2),
                        'away_weighted_form': round(weighted_away_form_points, 2)
                    },
                    'bayesian': {
                        'home_attack': round(home_form.get('bayesian', {}).get('home_lambda_scored', 0), 2),
                        'home_defense': round(home_form.get('bayesian', {}).get('home_lambda_conceded', 0), 2),
                        'away_attack': round(away_form.get('bayesian', {}).get('away_lambda_scored', 0), 2),
                        'away_defense': round(away_form.get('bayesian', {}).get('away_lambda_conceded', 0), 2),
                        'prior_home_goals': self.lig_ortalamasi_ev_gol,
                        'prior_away_goals': 1.0
                    },
                    'recent_goals_average': recent_goals_average,
                    'defense_factors': {
                        'home_defense_factor': round(home_defense_factor, 2),
                        'away_defense_factor': round(away_defense_factor, 2)
                    },
                    'z_score_data': {
                        'home_std_dev': round(home_std_dev, 2),
                        'away_std_dev': round(away_std_dev, 2)
                    },
                    'adjusted_thresholds': {
                        'home_threshold': home_threshold,
                        'away_threshold': away_threshold,
                        'home_max': home_max,
                        'away_max': away_max
                    }
                },
                'most_confident_bet': {
                    'market': most_confident_bet,
                    'prediction': bet_predictions[most_confident_bet],
                    'probability': round(bet_probabilities[most_confident_bet] * 100, 2)
                },
                'explanation': {
                    'exact_score': f"Analiz edilen faktörler sonucunda en olası skor {most_likely_score} olarak tahmin edildi. Ev sahibi takımın beklenen gol ortalaması {avg_home_goals:.2f}, deplasman takımının beklenen gol ortalaması {avg_away_goals:.2f}.",
                    'match_result': f"Maç sonucu {self._get_outcome_from_score(bet_predictions['exact_score'])} tahmini, ev sahibi (%{round(home_win_prob*100,1)}), beraberlik (%{round(draw_prob*100,1)}) ve deplasman (%{round(away_win_prob*100,1)}) olasılıklarına dayanmaktadır. Bu sonuç, en olası skor ({bet_predictions['exact_score']}) temel alınarak belirlenmiştir.",
                    'relative_strength': self._generate_strength_explanation(opponent_analysis, home_team_name, away_team_name),
                    'head_to_head': f"Geçmiş karşılaşmalarda {h2h_analysis and h2h_analysis['total_matches'] or 0} maç oynandı. Sonuçlar: {h2h_analysis and h2h_analysis['home_wins'] or 0} ev sahibi galibiyeti, {h2h_analysis and h2h_analysis['draws'] or 0} beraberlik, {h2h_analysis and h2h_analysis['away_wins'] or 0} deplasman galibiyeti."
                }
            },
            'timestamp': datetime.now().timestamp(),
            'date_predicted': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Gelişmiş modellerin tahminlerini ekle (varsa) - YENİ: Geliştirilmiş tutarlı entegrasyon
        if advanced_prediction:
            # Gelişmiş tahminlere esas al - daha tutarlı yaklaşım
            prediction['predictions']['advanced_models'] = {
                'zero_inflated_poisson': {
                    'expected_goals': advanced_prediction['expected_goals'],
                    'home_win_probability': advanced_prediction['home_win_probability'],
                    'draw_probability': advanced_prediction['draw_probability'],
                    'away_win_probability': advanced_prediction['away_win_probability'],
                    'most_likely_outcome': advanced_prediction['most_likely_outcome'],
                    'most_likely_scores': advanced_prediction['model_details']['top_5_likely_scores'][:3],
                    'zero_zero_probability': advanced_prediction['model_details']['zero_zero_prob']
                },
                'ensemble_predictions': {
                    'home_goals': advanced_prediction['model_details'].get('ensemble_home_goals', 0),
                    'away_goals': advanced_prediction['model_details'].get('ensemble_away_goals', 0)
                }
            }

            # YENİ: Artık gelişmiş tahminlere daha fazla ağırlık ver - tutarlılık için
            # Gelişmiş tahminlere %70, klasik tahminlere %30 ağırlık
            combined_home_goals = (avg_home_goals * 0.3 + advanced_prediction['expected_goals']['home'] * 0.7)
            combined_away_goals = (avg_away_goals * 0.3 + advanced_prediction['expected_goals']['away'] * 0.7)

            # Final tahminleri güncelle
            prediction['predictions']['expected_goals'] = {
                'home': round(combined_home_goals, 2),
                'away': round(combined_away_goals, 2)
            }
            
            # YENİ: Bahis tahminlerini de gelişmiş modelden al eğer mevcutsa - TUTARLILIK için
            if 'betting_predictions' in advanced_prediction:
                adv_betting = advanced_prediction['betting_predictions']
                
                # Eğer gelişmiş model tam bahis tahminleri içeriyorsa bunları kullan
                if all(key in adv_betting for key in ['both_teams_to_score', 'over_2_5_goals', 'exact_score']):
                    logger.info("Bahis tahminleri gelişmiş tutarlı modelden alınıyor")
                    
                    # Kesin skor
                    if 'exact_score' in adv_betting:
                        bet_predictions['exact_score'] = adv_betting['exact_score']['prediction']
                        bet_probabilities['exact_score'] = adv_betting['exact_score']['probability'] / 100
                    
                    # KG VAR/YOK
                    if 'both_teams_to_score' in adv_betting:
                        bet_predictions['both_teams_to_score'] = adv_betting['both_teams_to_score']['prediction']
                        bet_probabilities['both_teams_to_score'] = adv_betting['both_teams_to_score']['probability'] / 100
                    
                    # 2.5 ÜST/ALT
                    if 'over_2_5_goals' in adv_betting:
                        bet_predictions['over_2_5_goals'] = adv_betting['over_2_5_goals']['prediction']
                        bet_probabilities['over_2_5_goals'] = adv_betting['over_2_5_goals']['probability'] / 100
                    
                    # 3.5 ÜST/ALT
                    if 'over_3_5_goals' in adv_betting:
                        bet_predictions['over_3_5_goals'] = adv_betting['over_3_5_goals']['prediction']
                        bet_probabilities['over_3_5_goals'] = adv_betting['over_3_5_goals']['probability'] / 100
                    
                    # Maç sonucu (MS)
                    home_win_prob = advanced_prediction['home_win_probability'] / 100
                    draw_prob = advanced_prediction['draw_probability'] / 100
                    away_win_prob = advanced_prediction['away_win_probability'] / 100
                    
                    # MS tahmini güncelleme
                    prediction['predictions']['home_win_probability'] = advanced_prediction['home_win_probability']
                    prediction['predictions']['draw_probability'] = advanced_prediction['draw_probability']
                    prediction['predictions']['away_win_probability'] = advanced_prediction['away_win_probability']
                    
                    # KRİTİK DÜZELTME: Maç sonucu kesin skora göre belirlenmeli, gelişmiş modelin tahminini ezmemeliyiz
                    # Kesin skordan türetilen sonucu koru
                    score_based_outcome = self._get_outcome_from_score(bet_predictions['exact_score'])
                    prediction['predictions']['most_likely_outcome'] = score_based_outcome
                    prediction['predictions']['match_outcome'] = score_based_outcome
                    logger.info(f"Gelişmiş model sonucu ({advanced_prediction['most_likely_outcome']}) yerine kesin skor tabanlı sonuç ({score_based_outcome}) kullanıldı - tutarlılık için")
                    
                    # İlk yarı/maç sonu tahmini artık kullanılmıyor - hibrit model kaldırıldı
                    
                    # İlk gol kısmı kaldırıldı

        # Tahminlerin tutarlılığını kontrol et ve düzelt
        prediction = self._check_prediction_consistency(prediction)
        
        # Win probabilities'i mevcut değerlerden al - tutarlılık için
        # Zaten doğru hesaplanmış olasılıkları kullan, fallback sistemi devre dışı bırak
        final_home_win_prob = prediction['predictions']['home_win_probability'] / 100
        final_draw_prob = prediction['predictions']['draw_probability'] / 100  
        final_away_win_prob = prediction['predictions']['away_win_probability'] / 100
        
        # Bu olasılıklar zaten doğru - kesin skora dayalı hesaplandı
        home_win_prob = final_home_win_prob
        draw_prob = final_draw_prob
        away_win_prob = final_away_win_prob
        
        logger.info(f"Tutarlılık için mevcut olasılıklar korundu: Ev {home_win_prob:.3f}, Beraberlik {draw_prob:.3f}, Deplasman {away_win_prob:.3f}")
        
        prediction['predictions']['win_probabilities'] = {
            'home_win': round(home_win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'away_win': round(away_win_prob * 100, 1)
        }
        
        # Most likely outcome'u win probabilities'e göre güncelle
        if home_win_prob >= draw_prob and home_win_prob >= away_win_prob:
            prediction['predictions']['most_likely_outcome'] = 'MS1'
        elif away_win_prob >= draw_prob:
            prediction['predictions']['most_likely_outcome'] = 'MS2'
        else:
            prediction['predictions']['most_likely_outcome'] = 'X'
            
        # Gelişmiş tahmin özetini oluştur - güncellenmiş win probabilities ile
        prediction['predictions']['intelligent_summary'] = self._generate_intelligent_summary(
            home_win_prob, draw_prob, away_win_prob, home_team_name, away_team_name, 
            opponent_analysis, bet_predictions, bet_probabilities
        )
        
        # Win probabilities'i intelligent_summary ile senkronize et
        prediction['predictions']['intelligent_summary']['all_probabilities'] = {
            'home_win': round(home_win_prob * 100, 1),
            'draw': round(draw_prob * 100, 1),
            'away_win': round(away_win_prob * 100, 1)
        }
        
        # TUTARLILIK KONTROLÜ - Tüm tahminlerin uyumunu garantile
        try:
            from core.prediction_consistency import ensure_prediction_consistency
            prediction = ensure_prediction_consistency(prediction)
            logger.info("Tutarlılık kontrolü başarıyla uygulandı")
        except Exception as e:
            logger.warning(f"Tutarlılık kontrolü uygulanamadı: {str(e)}")
        
        # Önbelleğe ekle ve kaydet
        self.predictions_cache[cache_key] = prediction
        self._cache_modified = True
        self.save_cache()
        
        logger.info(f"Tahmin yapıldı: {home_team_name} vs {away_team_name}")
        return prediction
    def _check_prediction_consistency(self, prediction):
        """
        Bağımsız tahmin modellerini kullanarak tahminleri günceller.
        Kesin skora dayalı zorla düzeltme yapmaz, her tahmin türü bağımsız modelini kullanır.
        
        Args:
            prediction: Tahmin sonuçları
            
        Returns:
            dict: Güncellenmiş tahmin sonuçları
        """
        try:
            # Gerekli verileri çıkart
            expected_home_goals = prediction['predictions']['expected_goals']['home']
            expected_away_goals = prediction['predictions']['expected_goals']['away']
            
            # Form verilerini çıkart
            home_form = prediction.get('home_form', {})
            away_form = prediction.get('away_form', {})
            h2h_data = prediction.get('h2h_data', [])
            
            # Bağımsız tahmin modellerini çalıştır
            independent_predictions = self._generate_independent_predictions(
                expected_home_goals, expected_away_goals, home_form, away_form, h2h_data
            )
            
            if independent_predictions:
                # Bağımsız modellerin sonuçlarını tahminlere entegre et
                betting_predictions = prediction['predictions']['betting_predictions']
                
                # 2.5 ve 3.5 Üst/Alt tahminleri Monte Carlo sonuçlarından korunuyor
                # Bağımsız modeller bu değerleri override etmeyecek
                logger.info("2.5 ve 3.5 Üst/Alt tahminleri Monte Carlo simülasyonu sonuçlarından korunuyor - bağımsız model override atlandı")
                
                # KG Var/Yok tahmini güncelle - MONTE CARLO KARARI KORUNUR
                if 'both_teams_to_score' in independent_predictions:
                    ind_btts = independent_predictions['both_teams_to_score']
                    if isinstance(ind_btts, dict):
                        betting_predictions['both_teams_to_score']['prediction'] = ind_btts['prediction']
                        betting_predictions['both_teams_to_score']['probability'] = ind_btts['probability']
                        logger.info(f"Bağımsız model KG Var/Yok: {ind_btts['prediction']} ({ind_btts['probability']}%)")
                    elif ind_btts != 'MONTE_CARLO_DECISION_PRESERVED':
                        # Sadece gerçek fallback değerlerinde güncelle, placeholder'da değil
                        betting_predictions['both_teams_to_score']['prediction'] = ind_btts
                        logger.info(f"Fallback KG Var/Yok: {ind_btts}")
                    else:
                        # Monte Carlo kararı korunuyor - fallback override yapmıyor
                        logger.info("KG VAR/YOK kararı Monte Carlo simülasyonundan korundu - fallback override atlandı")
                
                # Maç sonucunu kesin skordan türet (tutarlılık için)
                exact_score = prediction['predictions']['exact_score']
                score_based_outcome = self._get_outcome_from_score(exact_score)
                if score_based_outcome:
                    prediction['predictions']['most_likely_outcome'] = score_based_outcome
                    prediction['predictions']['match_outcome'] = score_based_outcome
                    logger.info(f"Maç sonucu kesin skor {exact_score} temel alınarak {score_based_outcome} olarak belirlendi")
                
                logger.info("Tüm tahminler bağımsız modellerle güncellendi - kesin skordan bağımsız hesaplama")
            else:
                logger.info("Bağımsız modeller mevcut değil, mevcut tahminler korunuyor")
                
        except Exception as e:
            logger.error(f"Tahmin tutarlılık kontrolü hatası: {e}")
        
        return prediction

    def collect_training_data(self):
        """Tüm önbellekteki maçları kullanarak sinir ağları için eğitim verisi topla"""
        try:
            training_data_home = []
            training_data_away = []
            
            # Önbellekteki tahminleri kullanarak eğitim verisi oluştur
            for cache_key, cached_prediction in self.predictions_cache.items():
                if isinstance(cached_prediction, dict) and 'home_form' in cached_prediction and 'away_form' in cached_prediction:
                    home_form = cached_prediction.get('home_form', {})
                    away_form = cached_prediction.get('away_form', {})
                    
                    if home_form and away_form:
                        # Ev sahibi için eğitim verisi
                        home_data = self.prepare_data_for_neural_network(home_form, is_home=True)
                        if home_data is not None and len(home_data) > 0:
                            training_data_home.append(home_data)
                        
                        # Deplasman için eğitim verisi
                        away_data = self.prepare_data_for_neural_network(away_form, is_home=False)
                        if away_data is not None and len(away_data) > 0:
                            training_data_away.append(away_data)
            
            # Eğitim verisi varsa sinir ağlarını eğit
            if len(training_data_home) >= 3:
                import numpy as np
                X_train_home = np.array(training_data_home)
                # Basit hedef değerler - gerçek maç sonuçları yerine form puanları kullan
                y_train_home = np.array([np.mean(data) for data in training_data_home])
                
                if hasattr(self, 'neural_model_home') and self.neural_model_home:
                    try:
                        self.neural_model_home.fit(X_train_home, y_train_home, epochs=1, verbose=0)
                        logger.info(f"Ev sahibi sinir ağı {len(training_data_home)} veri ile eğitildi")
                    except Exception as e:
                        logger.warning(f"Ev sahibi sinir ağı eğitiminde hata: {e}")
            
            if len(training_data_away) >= 3:
                import numpy as np
                X_train_away = np.array(training_data_away)
                y_train_away = np.array([np.mean(data) for data in training_data_away])
                
                if hasattr(self, 'neural_model_away') and self.neural_model_away:
                    try:
                        self.neural_model_away.fit(X_train_away, y_train_away, epochs=1, verbose=0)
                        logger.info(f"Deplasman sinir ağı {len(training_data_away)} veri ile eğitildi")
                    except Exception as e:
                        logger.warning(f"Deplasman sinir ağı eğitiminde hata: {e}")
                        
            logger.info("Sinir ağları eğitim verisi toplama işlemi tamamlandı")
            
        except Exception as e:
            logger.warning(f"Sinir ağları eğitim verisi toplama hatası: {e}")

    def _get_outcome_from_score(self, score):
        """
        Belirli bir skordan maç sonucunu belirle - merkezi dinamik tahmin algoritması
        
        Args:
            score: "3-1" formatında kesin skor string'i
            
        Returns:
            "HOME_WIN", "DRAW", "AWAY_WIN" veya None
        """
        try:
            if not score or '-' not in score:
                return None
                
            parts = score.split('-')
            if len(parts) != 2:
                return None
                
            home_goals = int(parts[0])
            away_goals = int(parts[1])
            
            if home_goals > away_goals:
                return "HOME_WIN"
            elif away_goals > home_goals:
                return "AWAY_WIN"
            else:
                return "DRAW"
                
        except (ValueError, AttributeError):
            return None

    def _update_all_predictions_from_score(self, prediction):
        """
        Kesin skora göre tüm bahis tahminlerini günceller.
        Bu fonksiyon, tahmin tutarlılığı sağlamak için merkezi bir mekanizmadır.
        
        Args:
            prediction: Tahmin sonuçları sözlüğü
            
        Returns:
            dict: Tutarlı hale getirilmiş tahmin sonuçları
        """
        try:
            exact_score = prediction['predictions']['exact_score']
            home_score, away_score = map(int, exact_score.split('-'))
            total_goals = home_score + away_score
            
            # 1. Maç sonucunu skordan belirle
            match_outcome = self._get_outcome_from_score(exact_score)
            if match_outcome:
                prediction['predictions']['most_likely_outcome'] = match_outcome
                prediction['predictions']['match_outcome'] = match_outcome
            
            # 2. KG VAR/YOK tahminini Monte Carlo sonuçları belirler - skordan değil
            # Orijinal Monte Carlo simülasyonu sonuçlarını koruyoruz
            if 'both_teams_to_score' in prediction['predictions']['betting_predictions']:
                logger.info(f"KG VAR/YOK tahmini Monte Carlo sonuçlarına göre korunuyor: {prediction['predictions']['betting_predictions']['both_teams_to_score']['prediction']}")
                # Skordan override etmiyoruz - Monte Carlo simülasyonu sonuçları geçerli
            
            # 3. 2.5 ALT/ÜST ve 3.5 ALT/ÜST tahminleri Monte Carlo sonuçlarından korunuyor
            # Zorla düzeltme kaldırıldı - gerçek olasılıklar Monte Carlo simülasyonundan geliyor
            logger.info("2.5 ve 3.5 Alt/Üst tahminleri Monte Carlo simülasyonu sonuçlarından korunuyor")
            
            logger.info(f"Tüm bahis tahminleri kesin skor ({exact_score}) temelinde güncellendi")
            
        except Exception as e:
            logger.error(f"Skor tabanlı tahmin güncelleme hatası: {e}")
        
        return prediction

    def _get_most_likely_outcome(self, home_win_prob, draw_prob, away_win_prob, avg_home_goals=None, avg_away_goals=None):
        """
        En olası sonucu belirle - beklenen gol farkını da hesaba katarak
        
        Args:
            home_win_prob: Ev sahibi kazanma olasılığı
            draw_prob: Beraberlik olasılığı
            away_win_prob: Deplasman kazanma olasılığı
            avg_home_goals: Ev sahibi için beklenen gol sayısı (opsiyonel)
            avg_away_goals: Deplasman için beklenen gol sayısı (opsiyonel)
            
        Returns:
            String: "HOME_WIN", "DRAW" veya "AWAY_WIN"
        """
        try:
            # Olasılıkları sözlük haline getir
            probabilities = {
                'HOME_WIN': float(home_win_prob) if home_win_prob is not None else 0.0,
                'DRAW': float(draw_prob) if draw_prob is not None else 0.0,
                'AWAY_WIN': float(away_win_prob) if away_win_prob is not None else 0.0
            }
            
            # En yüksek olasılığa sahip sonucu bul
            most_likely = max(probabilities, key=probabilities.get)
            
            # Eğer gol beklentileri verilmişse, çok yakın olasılıklarda gol farkını da değerlendir
            if avg_home_goals is not None and avg_away_goals is not None:
                max_prob = probabilities[most_likely]
                
                # Olasılıklar çok yakınsa (fark %5'ten az), gol beklentisini de hesaba kat
                for outcome, prob in probabilities.items():
                    if outcome != most_likely and abs(max_prob - prob) < 5.0:
                        goal_diff = avg_home_goals - avg_away_goals
                        
                        # Gol farkı 0.3'ten fazlaysa ev sahibi avantajlı
                        if goal_diff > 0.3 and outcome == 'HOME_WIN':
                            most_likely = 'HOME_WIN'
                            break
                        # Gol farkı -0.3'ten küçükse deplasman avantajlı  
                        elif goal_diff < -0.3 and outcome == 'AWAY_WIN':
                            most_likely = 'AWAY_WIN'
                            break
            
            logger.info(f"En olası sonuç: {most_likely} (olasılıklar: Ev {probabilities['HOME_WIN']:.1f}%, Beraberlik {probabilities['DRAW']:.1f}%, Deplasman {probabilities['AWAY_WIN']:.1f}%)")
            return most_likely
            
        except Exception as e:
            logger.error(f"En olası sonuç belirleme hatası: {e}")
            # Hata durumunda ev sahibi avantajını varsay
            return "HOME_WIN"

    def _select_reasonable_score_from_simulation(self, top_scores, avg_home_goals, avg_away_goals):
        """
        Monte Carlo simülasyonundan makul bir skor seç
        Düşük beklenen gol değerleri için çok yüksek skorları engeller
        ve beklenen gol değerlerine uyumlu sonuçlar üretir
        
        Args:
            top_scores: Monte Carlo'dan gelen en yüksek olasılıklı skorlar ve sayıları
            avg_home_goals: Ev sahibi için beklenen gol
            avg_away_goals: Deplasman için beklenen gol
            
        Returns:
            String: Makul bir skor (örn: "1-0")
        """
        try:
            if not top_scores:
                return "1-1"  # Varsayılan skor
            
            # Beklenen gol değerlerini kontrol et
            total_expected = avg_home_goals + avg_away_goals
            is_low_scoring = total_expected < 2.0
            is_very_low_scoring = total_expected < 1.5
            
            # Düşük skorlu maçlar için özel işlem
            if is_very_low_scoring:
                logger.info(f"Çok düşük skorlu maç tespit edildi (toplam beklenti: {total_expected:.2f})")
                
                # 0-0, 1-0, 0-1 skorlarını öncelikle değerlendir
                preferred_scores = ['0-0', '1-0', '0-1']
                for score in preferred_scores:
                    if score in top_scores:
                        logger.info(f"Düşük skorlu maç için {score} skoru seçildi")
                        return score
                
                # Form farkına göre karar ver
                goal_diff = avg_home_goals - avg_away_goals
                if abs(goal_diff) < 0.1:
                    return "0-0"  # Çok dengesiz maç
                elif goal_diff > 0:
                    return "1-0"  # Ev sahibi hafif avantajlı
                else:
                    return "0-1"  # Deplasman hafif avantajlı
            
            elif is_low_scoring:
                # Orta derecede düşük skorlu maçlar
                logger.info(f"Düşük skorlu maç tespit edildi (toplam beklenti: {total_expected:.2f})")
                
                reasonable_scores = ['0-0', '1-0', '0-1', '1-1', '2-0', '0-2', '2-1', '1-2']
                for score in reasonable_scores:
                    if score in top_scores:
                        return score
            
            # Normal maçlar için en olası skoru seç
            if isinstance(top_scores, dict):
                # En yüksek olasılığa sahip skoru bul
                best_score = max(top_scores.keys(), key=lambda x: top_scores[x])
                
                # Skoru kontrol et - çok yüksek skorları sınırla
                home_goals, away_goals = map(int, best_score.split('-'))
                total_goals = home_goals + away_goals
                
                # Toplam gol sayısını beklentiye göre sınırla
                max_reasonable_goals = min(8, int(total_expected * 2) + 2)
                
                if total_goals > max_reasonable_goals:
                    logger.warning(f"Skor çok yüksek ({best_score}), daha makul bir skor seçiliyor")
                    
                    # Daha makul skorlar arasından seç
                    reasonable_alternatives = []
                    for score, count in top_scores.items():
                        h_goals, a_goals = map(int, score.split('-'))
                        if (h_goals + a_goals) <= max_reasonable_goals:
                            reasonable_alternatives.append((score, count))
                    
                    if reasonable_alternatives:
                        # En yüksek olasılıklı makul skoru seç
                        best_reasonable = max(reasonable_alternatives, key=lambda x: x[1])
                        return best_reasonable[0]
                
                return best_score
            
            else:
                # Lista formatında verilmişse ilkini al
                return str(top_scores[0]) if top_scores else "1-1"
                
        except Exception as e:
            logger.error(f"Makul skor seçme hatası: {e}")
            # Hata durumunda beklenen gollere göre basit bir skor üret
            if avg_home_goals > avg_away_goals:
                return "2-1"
            elif avg_away_goals > avg_home_goals:
                return "1-2"
            else:
                return "1-1"

    def analyze_opponent_strength(self, home_form, away_form):
        """Rakip takım gücünü analiz et ve karşılaştır - Gelişmiş momentum analizi ile"""
        try:
            def calculate_team_power(form_data):
                """Takımın güç puanını hesapla (0-100 arası) - Gelişmiş istatistiksel analiz"""
                if not form_data or not isinstance(form_data, dict):
                    return 50.0  # Orta değer
                
                # Form performansından başla
                home_performance = form_data.get('home_performance', {})
                away_performance = form_data.get('away_performance', {})
                overall_performance = form_data.get('overall_performance', {})
                
                # Weighted form puanlarını al
                weighted_form = overall_performance.get('weighted_form_points', 0.5)
                
                # Gol istatistikleri
                avg_goals_scored = overall_performance.get('weighted_avg_goals_scored', 1.0)
                avg_goals_conceded = overall_performance.get('weighted_avg_goals_conceded', 1.0)
                
                # Son maçlardaki performans trendi
                recent_matches = form_data.get('recent_match_data', [])
                if len(recent_matches) >= 5:
                    recent_goals = sum(match.get('goals_scored', 0) for match in recent_matches[:5]) / 5
                    recent_conceded = sum(match.get('goals_conceded', 0) for match in recent_matches[:5]) / 5
                    recent_points = sum(match.get('points', 0) for match in recent_matches[:5]) / 5
                else:
                    recent_goals = avg_goals_scored
                    recent_conceded = avg_goals_conceded
                    recent_points = weighted_form
                
                # Güç hesaplama bileşenleri
                # 1. Form bazlı güç (0-40 puan)
                form_power = weighted_form * 40
                
                # 2. Atak gücü (0-30 puan) 
                attack_power = min(30, (avg_goals_scored + recent_goals) * 7.5)
                
                # 3. Savunma gücü (0-30 puan)
                defense_power = max(0, 30 - ((avg_goals_conceded + recent_conceded) * 7.5))
                
                # Toplam güç hesaplama
                total_power = form_power + attack_power + defense_power
                
                # Momentum faktörü (son 3 maçın trendi)
                if len(recent_matches) >= 3:
                    momentum = sum(match.get('points', 0) for match in recent_matches[:3]) / 3
                    momentum_bonus = (momentum - 1.0) * 5  # -5 ile +5 arası bonus
                    total_power += momentum_bonus
                
                return min(100, max(0, total_power))
            
            home_power = calculate_team_power(home_form)
            away_power = calculate_team_power(away_form)
            
            power_difference = home_power - away_power
            
            # Momentum analizi - gerçek momentum değerlerini hesapla
            home_momentum = 0.0
            away_momentum = 0.0
            
            # Ev sahibi momentum hesaplama
            if home_form and 'recent_match_data' in home_form:
                recent_home = home_form['recent_match_data'][:3]  # Son 3 maç
                if len(recent_home) >= 3:
                    home_momentum = sum(match.get('points', 0) for match in recent_home) / len(recent_home)
            
            # Deplasman momentum hesaplama
            if away_form and 'recent_match_data' in away_form:
                recent_away = away_form['recent_match_data'][:3]  # Son 3 maç
                if len(recent_away) >= 3:
                    away_momentum = sum(match.get('points', 0) for match in recent_away) / len(recent_away)
            
            momentum_difference = home_momentum - away_momentum
            
            # Göreceli güç hesaplama - doğru yüzde hesabı
            total_power = home_power + away_power
            if total_power > 0:
                home_strength_percentage = (home_power / total_power) * 100
                away_strength_percentage = (away_power / total_power) * 100
                relative_strength = home_strength_percentage - 50.0  # 50% referans noktası
            else:
                home_strength_percentage = 50.0
                away_strength_percentage = 50.0
                relative_strength = 0.0
            
            # Genel analiz
            analysis = {
                'home_power': home_power,
                'away_power': away_power,
                'power_difference': power_difference,
                'relative_strength': relative_strength / 50.0,  # -1.0 ile 1.0 arası normalize
                'home_strength_percentage': home_strength_percentage,
                'away_strength_percentage': away_strength_percentage,
                'home_momentum': home_momentum,
                'away_momentum': away_momentum,
                'momentum_difference': momentum_difference,
                'strength_analysis': {
                    'stronger_team': 'HOME' if power_difference > 5 else 'AWAY' if power_difference < -5 else 'BALANCED',
                    'power_gap': abs(power_difference),
                    'momentum_favor': 'HOME' if momentum_difference > 0.1 else 'AWAY' if momentum_difference < -0.1 else 'NEUTRAL'
                }
            }
            
            logger.info(f"Takım güç analizi: Ev {home_power:.1f} vs Deplasman {away_power:.1f} (fark: {power_difference:.1f})")
            logger.info(f"Momentum analizi: Ev {home_momentum:.2f} vs Deplasman {away_momentum:.2f} (fark: {momentum_difference:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Rakip güç analizi hatası: {e}")
            # Hata durumunda dengeli bir analiz döndür
            return {
                'home_power': 50.0,
                'away_power': 50.0,
                'power_difference': 0.0,
                'relative_strength': 0.0,
                'home_strength_percentage': 50.0,
                'away_strength_percentage': 50.0,
                'home_momentum': 0.0,
                'away_momentum': 0.0,
                'momentum_difference': 0.0,
                'strength_analysis': {
                    'stronger_team': 'BALANCED',
                    'power_gap': 0.0,
                    'momentum_favor': 'NEUTRAL'
                }
            }

    def _generate_strength_explanation(self, opponent_analysis, home_team_name, away_team_name):
        """Takım güç analizi açıklamasını oluştur"""
        try:
            stronger_team = opponent_analysis.get('strength_analysis', {}).get('stronger_team', 'BALANCED')
            home_strength = opponent_analysis.get('home_strength_percentage', 50.0)
            away_strength = opponent_analysis.get('away_strength_percentage', 50.0)
            power_gap = opponent_analysis.get('strength_analysis', {}).get('power_gap', 0.0)
            
            if stronger_team == 'BALANCED' or power_gap < 5.0:
                return "Rakip analizi sonucunda, takımlar güç açısından dengelenmiş olarak bulundu."
            elif stronger_team == 'HOME':
                return f"Rakip analizi sonucunda, {home_team_name} göreceli olarak daha güçlü bulundu (güç oranı: {home_strength:.1f}%)."
            else:  # AWAY
                return f"Rakip analizi sonucunda, {away_team_name} göreceli olarak daha güçlü bulundu (güç oranı: {away_strength:.1f}%)."
                
        except Exception as e:
            logger.error(f"Güç analizi açıklaması oluşturulurken hata: {e}")
            return "Takım güç analizi yapılamadı."
    def _calculate_hybrid_kg_var_probability(self, home_team_id, away_team_id, home_goals, away_goals, home_form, away_form, base_prob):
        """
        Hibrit KG VAR/YOK tahmin sistemi: Poisson + Logistic Regresyon
        Monte Carlo yerine matematiksel olarak sağlam yaklaşım
        """
        logger.info(f"=== HİBRİT KG VAR/YOK TAHMİN SİSTEMİ BAŞLIYOR ===")
        logger.info(f"Takımlar: {home_team_id} vs {away_team_id}")
        logger.info(f"Gol beklentileri: Ev {home_goals:.2f}, Deplasman {away_goals:.2f}")
        
        # 1. POISSON REGRESYON ILE GOL BEKLETILERI ANALIZI
        # Poisson dağılımı ile her takımın en az 1 gol atma olasılığı
        import math
        home_scores_prob = 1 - math.exp(-home_goals)  # P(Ev ≥ 1 gol)
        away_scores_prob = 1 - math.exp(-away_goals)  # P(Deplasman ≥ 1 gol)
        poisson_kg_var_prob = home_scores_prob * away_scores_prob
        
        logger.info(f"Poisson analizi: Ev gol olasılığı={home_scores_prob:.3f}, "
                   f"Deplasman gol olasılığı={away_scores_prob:.3f}, "
                   f"İkisi birden={poisson_kg_var_prob:.3f}")
        
        # 2. LOGISTIC REGRESYON ÖZELLIKLERI
        total_goals = home_goals + away_goals
        goal_difference = abs(home_goals - away_goals)
        
        # Form verilerinden özellikler çıkar
        home_attack_strength = home_form.get('goals_for_avg', 1.2) if home_form else 1.2
        away_attack_strength = away_form.get('goals_for_avg', 1.0) if away_form else 1.0
        home_defense_weakness = home_form.get('goals_against_avg', 1.2) if home_form else 1.2
        away_defense_weakness = away_form.get('goals_against_avg', 1.1) if away_form else 1.1
        
        # Form balance hesapla
        recent_home_goals = home_form.get('recent_goals_for', home_attack_strength) if home_form else home_attack_strength
        recent_away_goals = away_form.get('recent_goals_for', away_attack_strength) if away_form else away_attack_strength
        form_balance = abs(recent_home_goals - recent_away_goals)
        
        # 3. LOGISTIC REGRESYON KATSAYILARI (Empirik olarak ayarlanmış)
        logistic_coefficients = {
            'total_goals_expected': 0.45,
            'goal_difference': -0.25,
            'home_attack_strength': 0.20,
            'away_attack_strength': 0.18,
            'home_defense_weakness': 0.15,
            'away_defense_weakness': 0.12,
            'form_balance': -0.08
        }
        
        # Doğrusal kombinasyon hesapla
        linear_combination = (
            logistic_coefficients['total_goals_expected'] * total_goals +
            logistic_coefficients['goal_difference'] * goal_difference +
            logistic_coefficients['home_attack_strength'] * home_attack_strength +
            logistic_coefficients['away_attack_strength'] * away_attack_strength +
            logistic_coefficients['home_defense_weakness'] * home_defense_weakness +
            logistic_coefficients['away_defense_weakness'] * away_defense_weakness +
            logistic_coefficients['form_balance'] * form_balance
        )
        
        # Sigmoid fonksiyonu ile olasılığa çevir
        logistic_prob = 1 / (1 + np.exp(-linear_combination))
        logistic_prob = max(0.1, min(0.9, logistic_prob))  # 0.1-0.9 arasında sınırla
        
        logger.info(f"Logistic regresyon: linear_combo={linear_combination:.3f}, "
                   f"sigmoid_prob={logistic_prob:.3f}")
        
        # 4. GEÇMIŞ VERI TEMELLI OLASLIK
        # Basit form tabanlı hesaplama
        home_goal_rate = min(0.85, home_attack_strength / 2.0)
        away_goal_rate = min(0.85, away_attack_strength / 2.0)
        historical_prob = home_goal_rate * away_goal_rate
        
        # Form bonusu/cezası
        if recent_home_goals > 1.5 and recent_away_goals > 1.0:
            historical_prob *= 1.15  # İyi form bonusu
        elif recent_home_goals < 0.8 or recent_away_goals < 0.5:
            historical_prob *= 0.85  # Kötü form cezası
            
        historical_prob = max(0.1, min(0.9, historical_prob))
        
        logger.info(f"Geçmiş veri analizi: home_rate={home_goal_rate:.3f}, "
                   f"away_rate={away_goal_rate:.3f}, historical_prob={historical_prob:.3f}")
        
        # 5. MODEL AĞIRLIKLANDIRMASI
        model_weights = {
            'poisson': 0.4,      # Poisson regresyon
            'logistic': 0.35,    # Logistic regresyon  
            'historical': 0.25   # Geçmiş veriler
        }
        
        # Hibrit olasılık hesapla
        final_probability = (
            model_weights['poisson'] * poisson_kg_var_prob +
            model_weights['logistic'] * logistic_prob +
            model_weights['historical'] * historical_prob
        )
        
        # 6. DÜŞÜK GOL BEKLENTISI AYARLAMASI - AGRESİF DÜZELTME
        # Matematiksel mantık: Eğer bir takımın gol beklentisi 1'in altındaysa, gol atma şansı %63'ten az
        min_goal_expectation = min(home_goals, away_goals)
        
        if min_goal_expectation < 0.5:
            final_probability *= 0.08  # %92 azalt - ultra agresif
            logger.info(f"Çok düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.5) - %92 azalma")
        elif min_goal_expectation < 0.7:
            final_probability *= 0.12  # %88 azalt
            logger.info(f"Düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.7) - %88 azalma")
        elif min_goal_expectation < 0.85:
            final_probability *= 0.18  # %82 azalt
            logger.info(f"Orta-düşük minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 0.85) - %82 azalma")
        elif min_goal_expectation < 1.0:
            final_probability *= 0.25  # %75 azalt
            logger.info(f"1'in altı minimum gol beklentisi düzeltmesi ({min_goal_expectation:.2f} < 1.0) - %75 azalma")
        
        # Toplam gol beklentisi de düşükse ek düzeltme
        if total_goals < 1.5:
            final_probability *= 0.7  # Ek %30 azalt
            logger.info(f"Çok düşük toplam gol beklentisi ek düzeltmesi ({total_goals:.2f} < 1.5)")
        elif total_goals < 2.0:
            final_probability *= 0.85  # Ek %15 azalt
            logger.info(f"Düşük toplam gol beklentisi ek düzeltmesi ({total_goals:.2f} < 2.0)")
        
        # 7. GOL FARKI DÜZELTMESI
        if goal_difference > 1.5:
            final_probability *= 0.85  # Büyük fark durumunda azalt
            logger.info(f"Büyük gol farkı düzeltmesi uygulandı ({goal_difference:.2f} > 1.5)")
        
        final_probability = max(0.05, min(0.95, final_probability))
        
        # Sonuçları logla
        logger.info(f"Model dağılımı: Poisson=%{poisson_kg_var_prob*100:.1f}, "
                   f"Logistic=%{logistic_prob*100:.1f}, Historical=%{historical_prob*100:.1f}")
        logger.info(f"Final hibrit olasılık: {final_probability:.3f} (%{final_probability*100:.1f})")
        logger.info(f"=== HİBRİT KG VAR/YOK TAHMİN SİSTEMİ SONU ===")
        
        return final_probability
    def _calculate_confidence(self, home_form, away_form, betting_predictions=None, simulation_results=None):
        """Gelişmiş tahmin güvenilirlik skorunu hesaplar"""
        try:
            base_confidence = 0.35  # Orta başlangıç güven seviyesi
            
            # Form verisi kalitesine göre güven ayarlaması
            form_quality_score = 0.0
            
            if home_form and isinstance(home_form, dict):
                # Ev sahibi form kalitesi ve istikrarı
                recent_matches = home_form.get('recent_matches', 0)
                if recent_matches >= 8:
                    form_quality_score += 0.08  # Daha düşük bonus
                elif recent_matches >= 5:
                    form_quality_score += 0.05
                elif recent_matches >= 3:
                    form_quality_score += 0.03
                
                # Gol atma istikrarı
                goals_scored = home_form.get('goals_scored', 0)
                goals_conceded = home_form.get('goals_conceded', 0)
                if goals_scored > 0 and goals_conceded >= 0:
                    goal_ratio = goals_scored / max(1, goals_conceded)
                    if 0.8 <= goal_ratio <= 2.5:  # Dengeli performans
                        form_quality_score += 0.04
                    elif goal_ratio > 2.5:  # Çok iyi atak
                        form_quality_score += 0.06
                
                # Form tutarlılığı
                if 'recent_form' in home_form:
                    form_quality_score += 0.05
            
            if away_form and isinstance(away_form, dict):
                # Deplasman form kalitesi ve istikrarı
                recent_matches = away_form.get('recent_matches', 0)
                if recent_matches >= 8:
                    form_quality_score += 0.15
                elif recent_matches >= 5:
                    form_quality_score += 0.1
                elif recent_matches >= 3:
                    form_quality_score += 0.05
                
                # Gol atma istikrarı
                goals_scored = away_form.get('goals_scored', 0)
                goals_conceded = away_form.get('goals_conceded', 0)
                if goals_scored > 0 and goals_conceded >= 0:
                    goal_ratio = goals_scored / max(1, goals_conceded)
                    if 0.8 <= goal_ratio <= 2.5:
                        form_quality_score += 0.08
                    elif goal_ratio > 2.5:
                        form_quality_score += 0.12
                
                if 'recent_form' in away_form:
                    form_quality_score += 0.05
            
            # Tahmin tutarlılığı analizi
            consistency_bonus = 0.0
            if betting_predictions:
                # 2.5 Alt/Üst ile diğer tahminlerin tutarlılığı
                over_25 = betting_predictions.get('over_2_5_goals', '')
                btts = betting_predictions.get('both_teams_to_score', '')
                exact_score = betting_predictions.get('exact_score', '')
                
                # Tutarlılık kontrolü
                if over_25 == '2.5 ÜST' and btts in ['KG VAR', 'YES']:
                    consistency_bonus += 0.05  # Tutarlı yüksek gol tahmini
                elif over_25 == '2.5 ALT' and btts in ['KG YOK', 'NO']:
                    consistency_bonus += 0.05  # Tutarlı düşük gol tahmini
                elif over_25 == '2.5 ALT' and btts in ['KG VAR', 'YES']:
                    consistency_bonus -= 0.03  # Çelişkili tahmin (düşük güven)
                
                # Kesin skor tutarlılığı
                if exact_score and '-' in exact_score:
                    try:
                        home_goals, away_goals = map(int, exact_score.split('-'))
                        total_goals = home_goals + away_goals
                        
                        # Kesin skor ile 2.5 Alt/Üst tutarlılığı
                        if (total_goals > 2.5 and over_25 == '2.5 ÜST') or (total_goals < 2.5 and over_25 == '2.5 ALT'):
                            consistency_bonus += 0.08
                        else:
                            consistency_bonus -= 0.08
                        
                        # Kesin skor ile KG VAR/YOK tutarlılığı
                        if (home_goals > 0 and away_goals > 0 and btts in ['KG VAR', 'YES']) or \
                           (home_goals == 0 or away_goals == 0) and btts in ['KG YOK', 'NO']:
                            consistency_bonus += 0.08
                        else:
                            consistency_bonus -= 0.08
                            
                    except (ValueError, AttributeError):
                        consistency_bonus -= 0.05  # Geçersiz skor formatı
            
            # Simülasyon güvenilirliği
            simulation_bonus = 0.0
            if simulation_results:
                # Simülasyon sayısı ve dağılımı
                total_simulations = simulation_results.get('total_simulations', 0)
                if total_simulations >= 8000:
                    simulation_bonus += 0.05
                elif total_simulations >= 5000:
                    simulation_bonus += 0.03
                
                # En yüksek olasılıklı sonucun dominantlığı
                max_outcome_prob = 0
                outcome_probs = simulation_results.get('outcome_probabilities', {})
                if outcome_probs:
                    max_outcome_prob = max(outcome_probs.values())
                    if max_outcome_prob > 0.6:  # Çok dominant tahmin
                        simulation_bonus += 0.08
                    elif max_outcome_prob > 0.45:  # Orta güven
                        simulation_bonus += 0.05
                    elif max_outcome_prob < 0.35:  # Belirsizlik
                        simulation_bonus -= 0.05
            
            # Takım gücü farkı analizi
            strength_bonus = 0.0
            if home_form and away_form:
                home_strength = home_form.get('goals_scored', 1) / max(1, home_form.get('goals_conceded', 1))
                away_strength = away_form.get('goals_scored', 1) / max(1, away_form.get('goals_conceded', 1))
                
                strength_diff = abs(home_strength - away_strength)
                if strength_diff > 1.5:  # Büyük güç farkı = daha güvenilir tahmin
                    strength_bonus += 0.12
                elif strength_diff > 0.8:
                    strength_bonus += 0.08
                elif strength_diff < 0.3:  # Çok dengeli takımlar = belirsizlik
                    strength_bonus -= 0.05
            
            # Veri eksikliği cezası
            data_penalty = 0.0
            if not home_form or not away_form:
                data_penalty = 0.15  # Eksik form verisi
            elif home_form.get('recent_matches', 0) < 3 or away_form.get('recent_matches', 0) < 3:
                data_penalty = 0.08  # Az maç verisi
            
            # Final güven skorunu hesapla
            confidence_score = base_confidence + form_quality_score + consistency_bonus + simulation_bonus + strength_bonus - data_penalty
            
            # Güven skorunu daha geniş ve gerçekçi aralığa sınırla
            confidence_score = min(0.82, max(0.18, confidence_score))
            
            logger.info(f"Gelişmiş güven skoru hesaplandı: {confidence_score:.3f} (Form: +{form_quality_score:.2f}, Tutarlılık: +{consistency_bonus:.2f}, Simülasyon: +{simulation_bonus:.2f}, Güç Farkı: +{strength_bonus:.2f})")
            
            return round(confidence_score, 3)
            
        except Exception as e:
            logger.error(f"Güven skoru hesaplama hatası: {e}")
            return 0.6  # Varsayılan orta güven seviyesi

    def _calculate_win_probabilities(self, avg_home_goals, avg_away_goals, home_form, away_form, opponent_analysis):
        """
        Monte Carlo simülasyonu ile maç kazanma olasılıklarını hesaplar
        
        Args:
            avg_home_goals: Ev sahibi beklenen gol sayısı
            avg_away_goals: Deplasman beklenen gol sayısı
            home_form: Ev sahibi form verisi
            away_form: Deplasman form verisi
            opponent_analysis: Takım güç analizi
            
        Returns:
            tuple: (home_win_prob, draw_prob, away_win_prob)
        """
        try:
            # Monte Carlo simülasyonu ile temel olasılıkları hesapla
            simulation_results = self.monte_carlo_simulation(
                avg_home_goals, avg_away_goals, 10000, home_form, away_form
            )
            
            base_home_win = simulation_results.get('home_win_probability', 0) / 100
            base_draw = simulation_results.get('draw_probability', 0) / 100
            base_away_win = simulation_results.get('away_win_probability', 0) / 100
            
            # Takım güç analizini ekle
            if opponent_analysis:
                power_diff = opponent_analysis.get('power_difference', 0)
                momentum_diff = opponent_analysis.get('momentum_difference', 0)
                
                # Güç farkına göre olasılık ayarlaması (±%15'e kadar)
                power_adjustment = min(0.15, abs(power_diff) / 100 * 0.3)
                momentum_adjustment = min(0.1, abs(momentum_diff) * 0.2)
                
                if power_diff > 5:  # Ev sahibi güçlü
                    base_home_win += power_adjustment
                    base_away_win -= power_adjustment * 0.7
                    base_draw -= power_adjustment * 0.3
                elif power_diff < -5:  # Deplasman güçlü
                    base_away_win += power_adjustment
                    base_home_win -= power_adjustment * 0.7
                    base_draw -= power_adjustment * 0.3
                
                # Momentum ayarlaması
                if momentum_diff > 0.1:  # Ev sahibi momentumu yüksek
                    base_home_win += momentum_adjustment
                    base_away_win -= momentum_adjustment
                elif momentum_diff < -0.1:  # Deplasman momentumu yüksek
                    base_away_win += momentum_adjustment
                    base_home_win -= momentum_adjustment
            
            # Form kalitesine göre ek ayarlama - güçlendirilmiş algoritma
            try:
                # Daha kapsamlı form analizi
                home_form_points = home_form.get('form_points', 0.5)
                away_form_points = away_form.get('form_points', 0.5)
                
                # Son maçlardaki performansı da hesaba kat
                home_weighted_goals = home_form.get('home_performance', {}).get('weighted_avg_goals_scored', 1.0)
                away_weighted_goals = away_form.get('away_performance', {}).get('weighted_avg_goals_scored', 1.0)
                
                # Form ve gol performansı farkı
                form_diff = home_form_points - away_form_points
                goal_diff = home_weighted_goals - away_weighted_goals
                
                # Daha güçlü etki faktörleri
                form_adjustment = min(0.2, abs(form_diff) * 0.4)  # %20'ye kadar etki
                goal_adjustment = min(0.15, abs(goal_diff) * 0.3)  # %15'e kadar etki
                
                # Ev sahibi avantajını da ekle
                home_advantage = 0.1  # %10 ev sahibi avantajı
                base_home_win += home_advantage
                base_away_win -= home_advantage * 0.7
                base_draw -= home_advantage * 0.3
                
                # Form bazlı ayarlamalar
                if form_diff > 0.2:  # Ev sahibi formu iyi
                    base_home_win += form_adjustment
                    base_away_win -= form_adjustment * 0.7
                    base_draw -= form_adjustment * 0.3
                elif form_diff < -0.2:  # Deplasman formu iyi
                    base_away_win += form_adjustment
                    base_home_win -= form_adjustment * 0.7
                    base_draw -= form_adjustment * 0.3
                
                # Gol performansı bazlı ayarlamalar
                if goal_diff > 0.3:  # Ev sahibi daha çok gol atıyor
                    base_home_win += goal_adjustment
                    base_away_win -= goal_adjustment * 0.6
                    base_draw -= goal_adjustment * 0.4
                elif goal_diff < -0.3:  # Deplasman daha çok gol atıyor
                    base_away_win += goal_adjustment
                    base_home_win -= goal_adjustment * 0.6
                    base_draw -= goal_adjustment * 0.4
                    
            except Exception as e:
                logger.warning(f"Form analizi hatası: {e}")
                # Varsayılan ev sahibi avantajı
                base_home_win += 0.1
                base_away_win -= 0.05
                base_draw -= 0.05
            
            # Olasılıkları normalize et
            total = base_home_win + base_draw + base_away_win
            if total > 0:
                base_home_win /= total
                base_draw /= total
                base_away_win /= total
            
            # Minimum/maksimum değerleri kontrol et
            base_home_win = max(0.05, min(0.85, base_home_win))
            base_draw = max(0.05, min(0.50, base_draw))
            base_away_win = max(0.05, min(0.85, base_away_win))
            
            # Tekrar normalize et
            total = base_home_win + base_draw + base_away_win
            base_home_win /= total
            base_draw /= total
            base_away_win /= total
            
            logger.info(f"Win probabilities - Home: {base_home_win:.3f}, Draw: {base_draw:.3f}, Away: {base_away_win:.3f}")
            
            return base_home_win, base_draw, base_away_win
            
        except Exception as e:
            logger.error(f"Win probabilities hesaplama hatası: {e}")
            # Varsayılan dengeli olasılıklar
            return 0.40, 0.25, 0.35

    def _generate_intelligent_summary(self, home_win_prob, draw_prob, away_win_prob, 
                                    home_team_name, away_team_name, opponent_analysis, 
                                    bet_predictions, bet_probabilities):
        """
        Gelişmiş tahmin özeti oluşturur
        
        Args:
            home_win_prob: Ev sahibi kazanma olasılığı
            draw_prob: Beraberlik olasılığı
            away_win_prob: Deplasman kazanma olasılığı
            home_team_name: Ev sahibi takım adı
            away_team_name: Deplasman takım adı
            opponent_analysis: Takım analizi
            bet_predictions: Bahis tahminleri
            bet_probabilities: Bahis olasılıkları
            
        Returns:
            dict: Akıllı tahmin özeti
        """
        try:
            # En yüksek olasılıklı sonucu belirle
            max_prob = max(home_win_prob, draw_prob, away_win_prob)
            
            if max_prob == home_win_prob:
                prediction_text = f"{home_team_name} galibiyeti bekleniyor"
                confidence_level = "yüksek" if home_win_prob > 0.55 else "orta" if home_win_prob > 0.40 else "düşük"
                outcome_code = "MS1"
            elif max_prob == away_win_prob:
                prediction_text = f"{away_team_name} galibiyeti bekleniyor"
                confidence_level = "yüksek" if away_win_prob > 0.55 else "orta" if away_win_prob > 0.40 else "düşük"
                outcome_code = "MS2"
            else:
                prediction_text = "Beraberlik bekleniyor"
                confidence_level = "yüksek" if draw_prob > 0.35 else "orta" if draw_prob > 0.25 else "düşük"
                outcome_code = "X"
            
            # Güven seviyesi açıklaması
            confidence_explanation = {
                "yüksek": f"Tahmin güvenilirliği yüksek (%{max_prob*100:.1f})",
                "orta": f"Tahmin güvenilirliği orta (%{max_prob*100:.1f})",
                "düşük": f"Tahmin güvenilirliği düşük (%{max_prob*100:.1f}), maç dengeli"
            }
            
            # En iyi bahis önerisi
            best_bet = None
            best_bet_prob = 0
            
            for bet_type, probability in bet_probabilities.items():
                if probability > best_bet_prob and probability > 0.6:
                    best_bet = {
                        'type': bet_type,
                        'prediction': bet_predictions.get(bet_type, 'N/A'),
                        'probability': probability * 100
                    }
                    best_bet_prob = probability
            
            # Güç analizi kısmı tamamen kaldırıldı - sadece maç sonucu tahmini gösterilecek
            strength_summary = None  # Artık kullanılmayacak
            
            return {
                'main_prediction': prediction_text,
                'outcome_code': outcome_code,
                'confidence_level': confidence_level,
                'confidence_explanation': confidence_explanation[confidence_level],
                'probability_percentage': round(max_prob * 100, 1),
                'best_bet': best_bet,
                'all_probabilities': {
                    'home_win': round(home_win_prob * 100, 1),
                    'draw': round(draw_prob * 100, 1),
                    'away_win': round(away_win_prob * 100, 1)
                }
            }
            
        except Exception as e:
            logger.error(f"Akıllı özet oluşturma hatası: {e}")
            return {
                'main_prediction': "Tahmin analizi yapılamadı",
                'outcome_code': "X",
                'confidence_level': "düşük",
                'confidence_explanation': "Veri yetersizliği",
                'probability_percentage': 33.3,
                'strength_summary': "Analiz edilemedi",
                'best_bet': None,
                'all_probabilities': {
                    'home_win': 33.3,
                    'draw': 33.3,
                    'away_win': 33.3
                }
            }