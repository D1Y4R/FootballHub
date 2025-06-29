"""
KG VAR/YOK Tahmin Modelleri
Poisson Regresyon + Logistic Regresyon hibrit yaklaşımı

Bu modül futbol maçları için "Her İki Takım da Gol Atar mı?" sorusunu
matematiksel olarak sağlam yöntemlerle yanıtlar.
"""

import logging
from collections import defaultdict
import json
import os
import math

# NumPy ve SciPy yerine basit matematik kullanacağız
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # NumPy olmadan basit exponential hesaplama
    def exp(x):
        return math.exp(x)

logger = logging.getLogger(__name__)

class PoissonLogisticPredictor:
    """
    Poisson Regresyon ile gol beklentilerini hesaplar,
    Logistic Regresyon ile KG VAR/YOK kararı verir.
    """
    
    def __init__(self, cache_file="predictions_cache.json"):
        """
        Hibrit tahmin modelini başlat
        
        Args:
            cache_file: Tahmin önbellek dosyası
        """
        self.cache_file = cache_file
        self.team_stats = {}
        self.league_averages = {}
        self.model_weights = {
            'poisson': 0.4,
            'logistic': 0.35,
            'historical': 0.25
        }
        
        # Poisson parametreleri
        self.home_advantage = 1.15
        self.league_avg_goals = 2.5
        
        # Logistic regresyon parametreleri
        self.logistic_coefficients = {
            'total_goals_expected': 0.45,
            'goal_difference': -0.25,
            'home_attack_strength': 0.20,
            'away_attack_strength': 0.18,
            'home_defense_weakness': 0.15,
            'away_defense_weakness': 0.12,
            'form_balance': 0.08,
            'h2h_goals_avg': 0.35
        }
        
        logger.info("PoissonLogisticPredictor başlatıldı")
    
    def load_team_data(self):
        """Önbellek dosyasından takım verilerini yükle"""
        if not os.path.exists(self.cache_file):
            logger.warning(f"Önbellek dosyası bulunamadı: {self.cache_file}")
            return
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            self._process_team_statistics(cache_data)
            logger.info(f"{len(self.team_stats)} takım verisi yüklendi")
            
        except Exception as e:
            logger.error(f"Takım verileri yüklenirken hata: {e}")
    
    def _process_team_statistics(self, cache_data):
        """Önbellek verilerinden takım istatistiklerini hesapla"""
        team_matches = defaultdict(list)
        
        for match_key, match_data in cache_data.items():
            if not isinstance(match_data, dict) or 'home_team_id' not in match_data:
                continue
                
            home_id = match_data.get('home_team_id')
            away_id = match_data.get('away_team_id')
            
            if 'actual_result' in match_data:
                home_goals = match_data['actual_result'].get('home_goals', 0)
                away_goals = match_data['actual_result'].get('away_goals', 0)
                
                team_matches[home_id].append({
                    'is_home': True,
                    'goals_for': home_goals,
                    'goals_against': away_goals,
                    'opponent_id': away_id
                })
                
                team_matches[away_id].append({
                    'is_home': False,
                    'goals_for': away_goals,
                    'goals_against': home_goals,
                    'opponent_id': home_id
                })
        
        for team_id, matches in team_matches.items():
            if len(matches) >= 3:
                self.team_stats[team_id] = self._calculate_team_stats(matches)
    
    def _calculate_team_stats(self, matches):
        """Takımın detaylı istatistiklerini hesapla"""
        home_matches = [m for m in matches if m['is_home']]
        away_matches = [m for m in matches if not m['is_home']]
        
        stats = {
            'total_matches': len(matches),
            'goals_for_total': sum(m['goals_for'] for m in matches),
            'goals_against_total': sum(m['goals_against'] for m in matches),
            'avg_goals_for': sum(m['goals_for'] for m in matches) / len(matches),
            'avg_goals_against': sum(m['goals_against'] for m in matches) / len(matches),
        }
        
        if home_matches:
            stats['avg_goals_for_home'] = sum(m['goals_for'] for m in home_matches) / len(home_matches)
            stats['avg_goals_against_home'] = sum(m['goals_against'] for m in home_matches) / len(home_matches)
        else:
            stats['avg_goals_for_home'] = stats['avg_goals_for']
            stats['avg_goals_against_home'] = stats['avg_goals_against']
            
        if away_matches:
            stats['avg_goals_for_away'] = sum(m['goals_for'] for m in away_matches) / len(away_matches)
            stats['avg_goals_against_away'] = sum(m['goals_against'] for m in away_matches) / len(away_matches)
        else:
            stats['avg_goals_for_away'] = stats['avg_goals_for']
            stats['avg_goals_against_away'] = stats['avg_goals_against']
        
        recent_matches = matches[-5:] if len(matches) >= 5 else matches
        stats['recent_goals_for'] = sum(m['goals_for'] for m in recent_matches) / len(recent_matches)
        stats['recent_goals_against'] = sum(m['goals_against'] for m in recent_matches) / len(recent_matches)
        
        return stats
    
    def calculate_poisson_expectations(self, home_team_id, away_team_id):
        """
        Poisson regresyon ile gol beklentilerini hesapla
        
        Args:
            home_team_id, away_team_id: Takım ID'leri
            
        Returns:
            tuple: (home_expected, away_expected, confidence)
        """
        home_stats = self.team_stats.get(home_team_id, {})
        away_stats = self.team_stats.get(away_team_id, {})
        
        # Temel gol oranları
        home_attack = home_stats.get('avg_goals_for_home', self.league_avg_goals / 2)
        away_attack = away_stats.get('avg_goals_for_away', self.league_avg_goals / 2) 
        home_defense = home_stats.get('avg_goals_against_home', self.league_avg_goals / 2)
        away_defense = away_stats.get('avg_goals_against_away', self.league_avg_goals / 2)
        
        # Poisson gol beklentileri
        home_expected = (home_attack * away_defense / self.league_avg_goals) * self.home_advantage
        away_expected = (away_attack * home_defense / self.league_avg_goals)
        
        # Güven skoru (veri kalitesine göre)
        home_matches = home_stats.get('total_matches', 0)
        away_matches = away_stats.get('total_matches', 0)
        confidence = min(1.0, (home_matches + away_matches) / 20)
        
        return home_expected, away_expected, confidence
    
    def calculate_logistic_features(self, home_team_id, away_team_id, home_expected, away_expected):
        """
        Logistic regresyon için özellik vektörünü hesapla
        
        Args:
            home_team_id, away_team_id: Takım ID'leri
            home_expected, away_expected: Poisson'dan gelen beklentiler
            
        Returns:
            dict: Logistic regresyon özellikleri
        """
        home_stats = self.team_stats.get(home_team_id, {})
        away_stats = self.team_stats.get(away_team_id, {})
        
        features = {
            'total_goals_expected': home_expected + away_expected,
            'goal_difference': abs(home_expected - away_expected),
            'home_attack_strength': home_stats.get('avg_goals_for_home', 1.2),
            'away_attack_strength': away_stats.get('avg_goals_for_away', 1.0),
            'home_defense_weakness': home_stats.get('avg_goals_against_home', 1.2),
            'away_defense_weakness': away_stats.get('avg_goals_against_away', 1.1),
            'form_balance': abs(home_stats.get('recent_goals_for', 1.2) - away_stats.get('recent_goals_for', 1.0)),
            'h2h_goals_avg': (home_expected + away_expected) / 2
        }
        
        return features
    
    def predict_logistic_probability(self, features):
        """
        Logistic regresyon ile KG VAR olasılığını hesapla
        
        Args:
            features: Özellik vektörü
            
        Returns:
            float: KG VAR olasılığı (0-1 arası)
        """
        linear_combination = 0.0
        
        for feature_name, coefficient in self.logistic_coefficients.items():
            feature_value = features.get(feature_name, 0.0)
            linear_combination += coefficient * feature_value
        
        if NUMPY_AVAILABLE:
            probability = 1 / (1 + np.exp(-linear_combination))
        else:
            probability = 1 / (1 + exp(-linear_combination))
        probability = max(0.1, min(0.9, probability))
        
        return probability
    
    def calculate_historical_probability(self, home_team_id, away_team_id):
        """
        Geçmiş maç verilerine dayalı KG VAR olasılığı
        
        Args:
            home_team_id, away_team_id: Takım ID'leri
            
        Returns:
            float: Geçmiş verilere dayalı KG VAR olasılığı
        """
        home_stats = self.team_stats.get(home_team_id, {})
        away_stats = self.team_stats.get(away_team_id, {})
        
        home_score_rate = min(0.9, home_stats.get('avg_goals_for_home', 1.2) / 2.0)
        away_score_rate = min(0.9, away_stats.get('avg_goals_for_away', 1.0) / 2.0)
        
        both_score_prob = home_score_rate * away_score_rate
        
        home_form = home_stats.get('recent_goals_for', 1.2)
        away_form = away_stats.get('recent_goals_for', 1.0)
        
        if home_form > 1.5 and away_form > 1.0:
            both_score_prob *= 1.2
        elif home_form < 0.8 or away_form < 0.5:
            both_score_prob *= 0.8
        
        return min(0.9, max(0.1, both_score_prob))
    
    def apply_low_goal_mathematical_adjustment(self, base_probability, home_expected, away_expected):
        """
        Düşük gol beklentisi için matematiksel algoritma (zorla düzeltme değil!)
        
        Args:
            base_probability: Hibrit sistemden gelen temel olasılık
            home_expected: Ev sahibi gol beklentisi
            away_expected: Deplasman gol beklentisi
            
        Returns:
            float: Matematiksel olarak ayarlanmış olasılık
        """
        min_expected = min(home_expected, away_expected)
        total_expected = home_expected + away_expected
        
        if min_expected <= 0.5:
            adjustment_factor = 0.15 * math.exp(min_expected * 2)
        elif min_expected <= 0.8:
            adjustment_factor = 0.2 + (min_expected - 0.5) * 0.4
        elif min_expected <= 1.0:
            adjustment_factor = 0.35 + (min_expected - 0.8) * 0.75
        else:
            adjustment_factor = 0.7 + min(0.3, (min_expected - 1.0) * 0.15)
        
        if total_expected < 1.5:
            total_adjustment = 0.8
        elif total_expected < 2.0:
            total_adjustment = 0.9
        else:
            total_adjustment = 1.0
        
        adjusted_probability = base_probability * adjustment_factor * total_adjustment
        
        logger.info(f"Matematiksel ayarlama: min_gol={min_expected:.2f}, total_gol={total_expected:.2f}")
        logger.info(f"Faktörler: adj_factor={adjustment_factor:.3f}, total_adj={total_adjustment:.3f}")
        logger.info(f"Olasılık: {base_probability:.3f} -> {adjusted_probability:.3f}")
        
        return max(0.05, min(0.95, adjusted_probability))

    def predict_kg_var_yok(self, home_team_id, away_team_id):
        """
        Hibrit model ile KG VAR/YOK tahmini yap
        
        Args:
            home_team_id: Ev sahibi takım ID
            away_team_id: Deplasman takım ID
            
        Returns:
            dict: Detaylı tahmin sonuçları
        """
        logger.info(f"=== KG VAR/YOK HİBRİT TAHMİN BAŞLIYOR ===")
        logger.info(f"Ev takım ID: {home_team_id}, Deplasman takım ID: {away_team_id}")
        
        # 1. Poisson ile gol beklentilerini hesapla
        home_expected, away_expected, poisson_confidence = self.calculate_poisson_expectations(home_team_id, away_team_id)
        
        # 2. Logistic regresyon özellikleri
        features = self.calculate_logistic_features(home_team_id, away_team_id, home_expected, away_expected)
        logistic_prob = self.predict_logistic_probability(features)
        
        # 3. Geçmiş veriler analizi
        historical_prob = self.calculate_historical_probability(home_team_id, away_team_id)
        
        # 4. Ağırlıklı hibrit kombinasyon
        base_probability = (
            self.model_weights['poisson'] * self.calculate_poisson_btts_probability(home_expected, away_expected) +
            self.model_weights['logistic'] * logistic_prob +
            self.model_weights['historical'] * historical_prob
        )
        
        logger.info(f"Hibrit bileşenler - Poisson: {self.calculate_poisson_btts_probability(home_expected, away_expected):.3f}, "
                   f"Logistic: {logistic_prob:.3f}, Historical: {historical_prob:.3f}")
        logger.info(f"Temel hibrit olasılık: {base_probability:.3f}")
        
        # 5. Matematiksel düşük gol ayarlaması (zorla düzeltme değil!)
        final_probability = self.apply_low_goal_mathematical_adjustment(base_probability, home_expected, away_expected)
        
        # 6. Tahmin kararı
        prediction = 'KG VAR' if final_probability > 0.5 else 'KG YOK'
        confidence_score = abs(final_probability - 0.5) * 2
        
        logger.info(f"=== HİBRİT TAHMİN SONUCU ===")
        logger.info(f"Final olasılık: {final_probability:.3f}")
        logger.info(f"Tahmin: {prediction}")
        logger.info(f"Güven skoru: {confidence_score:.3f}")
        
        return {
            'prediction': prediction,
            'probability': round(final_probability * 100, 1),
            'confidence': round(confidence_score * 100, 1),
            'components': {
                'poisson': round(self.calculate_poisson_btts_probability(home_expected, away_expected) * 100, 1),
                'logistic': round(logistic_prob * 100, 1),
                'historical': round(historical_prob * 100, 1),
                'base_hybrid': round(base_probability * 100, 1)
            },
            'goal_expectations': {
                'home': round(home_expected, 2),
                'away': round(away_expected, 2)
            }
        }
    
    def calculate_poisson_btts_probability(self, home_expected, away_expected):
        """
        Poisson dağılımı kullanarak BTTS olasılığını hesapla
        
        Args:
            home_expected, away_expected: Gol beklentileri
            
        Returns:
            float: Poisson tabanlı BTTS olasılığı
        """
        if NUMPY_AVAILABLE:
            home_scores_prob = 1 - np.exp(-home_expected)
            away_scores_prob = 1 - np.exp(-away_expected)
        else:
            home_scores_prob = 1 - exp(-home_expected)
            away_scores_prob = 1 - exp(-away_expected)
        
        return home_scores_prob * away_scores_prob


# Global hibrit model instance
kg_predictor = PoissonLogisticPredictor()