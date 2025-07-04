"""
Tutarlılık Motoru - Tüm tahminlerin birbirine uyumlu olmasını garantiler
Professional Fullstack Implementation
"""

import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

class PredictionConsistencyEngine:
    """
    Tüm tahmin türleri arasında tutarlılığı sağlayan merkezi motor
    - Kesin skor ↔ Maç sonucu tutarlılığı
    - Kesin skor ↔ Alt/Üst tutarlılığı
    - Kesin skor ↔ KG VAR/YOK tutarlılığı
    - Olasılık toplamları = %100 kontrolü
    """
    
    def __init__(self):
        self.logger = logger
        
    def ensure_consistency(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ana tutarlılık kontrolü ve düzeltme fonksiyonu
        
        Args:
            predictions: Tahmin verilerini içeren dictionary
            
        Returns:
            Tutarlılığı garantilenmiş tahmin verisi
        """
        self.logger.info("Tutarlılık kontrolü başlatıldı")
        
        # 1. Kesin skor validasyonu
        exact_score = self._extract_exact_score(predictions)
        if not exact_score:
            self.logger.warning("Kesin skor bulunamadı, tutarlılık kontrolü atlanıyor")
            return predictions
            
        # 2. Skor-tabanlı tutarlılık düzeltmeleri
        predictions = self._ensure_score_match_result_consistency(predictions, exact_score)
        predictions = self._ensure_score_over_under_consistency(predictions, exact_score)
        predictions = self._ensure_score_btts_consistency(predictions, exact_score)
        
        # 3. Olasılık normalizasyonu
        predictions = self._normalize_probabilities(predictions)
        
        # 4. Final validasyon
        self._validate_final_consistency(predictions, exact_score)
        
        self.logger.info("Tutarlılık kontrolü tamamlandı")
        return predictions
    
    def _extract_exact_score(self, predictions: Dict[str, Any]) -> Optional[str]:
        """Tahmin verilerinden kesin skoru çıkarır"""
        try:
            # Farklı yapılardaki exact_score lokasyonlarını kontrol et
            if 'betting_predictions' in predictions.get('predictions', {}):
                exact_score_data = predictions['predictions']['betting_predictions'].get('exact_score', {})
                if isinstance(exact_score_data, dict):
                    return exact_score_data.get('prediction')
                return exact_score_data
                
            if 'exact_score' in predictions.get('predictions', {}):
                exact_score_data = predictions['predictions']['exact_score']
                if isinstance(exact_score_data, dict):
                    return exact_score_data.get('prediction')
                return exact_score_data
                
            return None
        except Exception as e:
            self.logger.error(f"Kesin skor çıkarılırken hata: {str(e)}")
            return None
    
    def _ensure_score_match_result_consistency(self, predictions: Dict[str, Any], exact_score: str) -> Dict[str, Any]:
        """Kesin skor ile maç sonucu arasındaki tutarlılığı garantiler"""
        try:
            if '-' not in exact_score:
                return predictions
                
            home_goals, away_goals = map(int, exact_score.split('-'))
            
            # Skordan maç sonucunu türet
            if home_goals > away_goals:
                correct_outcome = 'HOME_WIN'
                correct_ms_code = 'MS1'
            elif away_goals > home_goals:
                correct_outcome = 'AWAY_WIN'
                correct_ms_code = 'MS2'
            else:
                correct_outcome = 'DRAW'
                correct_ms_code = 'X'
            
            # Maç sonucu tahminlerini güncelle
            if 'predictions' in predictions:
                # Betting predictions
                if 'betting_predictions' in predictions['predictions']:
                    if 'match_result' in predictions['predictions']['betting_predictions']:
                        old_result = predictions['predictions']['betting_predictions']['match_result']
                        predictions['predictions']['betting_predictions']['match_result'] = correct_ms_code
                        self.logger.info(f"Maç sonucu tutarlılık: {old_result} → {correct_ms_code} (skor: {exact_score})")
                
                # Most likely outcome
                if 'most_likely_outcome' in predictions['predictions']:
                    predictions['predictions']['most_likely_outcome'] = correct_outcome
                
                if 'match_outcome' in predictions['predictions']:
                    predictions['predictions']['match_outcome'] = correct_outcome
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Skor-sonuç tutarlılığı kontrolünde hata: {str(e)}")
            return predictions
    
    def _ensure_score_over_under_consistency(self, predictions: Dict[str, Any], exact_score: str) -> Dict[str, Any]:
        """Kesin skor ile alt/üst tahminleri arasındaki tutarlılığı garantiler"""
        try:
            if '-' not in exact_score:
                return predictions
                
            home_goals, away_goals = map(int, exact_score.split('-'))
            total_goals = home_goals + away_goals
            
            # 2.5 ve 3.5 alt/üst için tutarlılık - SADECE MANTIK HATALARINI DÜZELT
            if 'betting_predictions' in predictions.get('predictions', {}):
                bp = predictions['predictions']['betting_predictions']
                
                # 2.5 Alt/Üst tutarlılığı - sadece açık mantık hatalarını düzelt
                if 'over_2_5_goals' in bp:
                    current_25_pred = bp['over_2_5_goals'].get('prediction', '')
                    
                    # Mantık hatası: 4 gol var ama 2.5 ALT tahmini
                    if total_goals >= 4 and '2.5 ALT' in current_25_pred:
                        bp['over_2_5_goals']['prediction'] = '2.5 ÜST'
                        # Olasılığı da uygun şekilde ayarla
                        current_prob = bp['over_2_5_goals'].get('probability', 50)
                        bp['over_2_5_goals']['probability'] = max(current_prob, 60.0)
                        self.logger.info(f"2.5 Alt/Üst tutarlılık düzeltmesi: {total_goals} gol için ÜST'e çevrildi")
                    
                    # Mantık hatası: 1 gol var ama 2.5 ÜST tahmini
                    elif total_goals <= 1 and '2.5 ÜST' in current_25_pred:
                        bp['over_2_5_goals']['prediction'] = '2.5 ALT'
                        current_prob = bp['over_2_5_goals'].get('probability', 50)
                        bp['over_2_5_goals']['probability'] = max(current_prob, 60.0)
                        self.logger.info(f"2.5 Alt/Üst tutarlılık düzeltmesi: {total_goals} gol için ALT'a çevrildi")
                
                # 3.5 Alt/Üst tutarlılığı - sadece açık mantık hatalarını düzelt
                if 'over_3_5_goals' in bp:
                    current_35_pred = bp['over_3_5_goals'].get('prediction', '')
                    
                    # Mantık hatası: 5+ gol var ama 3.5 ALT tahmini
                    if total_goals >= 5 and '3.5 ALT' in current_35_pred:
                        bp['over_3_5_goals']['prediction'] = '3.5 ÜST'
                        current_prob = bp['over_3_5_goals'].get('probability', 50)
                        bp['over_3_5_goals']['probability'] = max(current_prob, 70.0)
                        self.logger.info(f"3.5 Alt/Üst tutarlılık düzeltmesi: {total_goals} gol için ÜST'e çevrildi")
                    
                    # Mantık hatası: 2 gol var ama 3.5 ÜST tahmini  
                    elif total_goals <= 2 and '3.5 ÜST' in current_35_pred:
                        bp['over_3_5_goals']['prediction'] = '3.5 ALT'
                        current_prob = bp['over_3_5_goals'].get('probability', 50)
                        bp['over_3_5_goals']['probability'] = max(current_prob, 70.0)
                        self.logger.info(f"3.5 Alt/Üst tutarlılık düzeltmesi: {total_goals} gol için ALT'a çevrildi")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Skor-alt/üst tutarlılığı kontrolünde hata: {str(e)}")
            return predictions
    
    def _ensure_score_btts_consistency(self, predictions: Dict[str, Any], exact_score: str) -> Dict[str, Any]:
        """Kesin skor ile KG VAR/YOK arasındaki tutarlılığı garantiler"""
        try:
            if '-' not in exact_score:
                return predictions
                
            home_goals, away_goals = map(int, exact_score.split('-'))
            
            # KG VAR/YOK tutarlılığı - SADECE MANTIK HATALARINI DÜZELT
            if 'betting_predictions' in predictions.get('predictions', {}):
                bp = predictions['predictions']['betting_predictions']
                
                if 'both_teams_to_score' in bp:
                    current_btts_pred = bp['both_teams_to_score'].get('prediction', '')
                    
                    # Mantık hatası: Her iki takım gol attı ama KG YOK tahmini
                    if home_goals > 0 and away_goals > 0 and 'KG YOK' in current_btts_pred:
                        bp['both_teams_to_score']['prediction'] = 'KG VAR'
                        current_prob = bp['both_teams_to_score'].get('probability', 50)
                        bp['both_teams_to_score']['probability'] = max(current_prob, 60.0)
                        self.logger.info(f"KG VAR/YOK tutarlılık düzeltmesi: {exact_score} için KG VAR'a çevrildi")
                    
                    # Mantık hatası: En az bir takım gol atmadı ama KG VAR tahmini
                    elif (home_goals == 0 or away_goals == 0) and 'KG VAR' in current_btts_pred:
                        bp['both_teams_to_score']['prediction'] = 'KG YOK'
                        current_prob = bp['both_teams_to_score'].get('probability', 50)
                        bp['both_teams_to_score']['probability'] = max(current_prob, 60.0)
                        self.logger.info(f"KG VAR/YOK tutarlılık düzeltmesi: {exact_score} için KG YOK'a çevrildi")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Skor-KG tutarlılığı kontrolünde hata: {str(e)}")
            return predictions
    
    def _normalize_probabilities(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Maç sonucu olasılıklarının toplamının %100 olmasını garantiler"""
        try:
            if 'predictions' not in predictions:
                return predictions
            
            pred = predictions['predictions']
            
            # Maç sonucu olasılıklarını kontrol et
            home_prob = pred.get('home_win_probability', 0)
            draw_prob = pred.get('draw_probability', 0)
            away_prob = pred.get('away_win_probability', 0)
            
            total_prob = home_prob + draw_prob + away_prob
            
            # Normalizasyon gerekli mi?
            if abs(total_prob - 100.0) > 0.1:
                if total_prob > 0:
                    # Oransal olarak normalleştir
                    factor = 100.0 / total_prob
                    pred['home_win_probability'] = round(home_prob * factor, 2)
                    pred['draw_probability'] = round(draw_prob * factor, 2)
                    pred['away_win_probability'] = round(away_prob * factor, 2)
                    
                    self.logger.info(f"Olasılık normalizasyonu: {total_prob:.1f}% → 100.0%")
                else:
                    # Eşit dağıt
                    pred['home_win_probability'] = 33.33
                    pred['draw_probability'] = 33.33
                    pred['away_win_probability'] = 33.34
                    self.logger.warning("Sıfır olasılık tespit edildi, eşit dağıtım uygulandı")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Olasılık normalizasyonunda hata: {str(e)}")
            return predictions
    
    def _validate_final_consistency(self, predictions: Dict[str, Any], exact_score: str) -> bool:
        """Final tutarlılık validasyonu"""
        try:
            inconsistencies = []
            
            if '-' in exact_score:
                home_goals, away_goals = map(int, exact_score.split('-'))
                total_goals = home_goals + away_goals
                
                # Maç sonucu tutarlılığı
                if 'predictions' in predictions:
                    pred = predictions['predictions']
                    
                    # Kesin skor vs maç sonucu
                    if home_goals > away_goals:
                        expected_outcome = 'HOME_WIN'
                    elif away_goals > home_goals:
                        expected_outcome = 'AWAY_WIN'
                    else:
                        expected_outcome = 'DRAW'
                    
                    actual_outcome = pred.get('most_likely_outcome', '')
                    if actual_outcome != expected_outcome:
                        inconsistencies.append(f"Skor {exact_score} → {expected_outcome}, tahmin {actual_outcome}")
                    
                    # KG VAR/YOK tutarlılığı
                    if 'betting_predictions' in pred:
                        btts_pred = pred['betting_predictions'].get('both_teams_to_score', {}).get('prediction', '')
                        
                        if home_goals > 0 and away_goals > 0 and 'KG YOK' in btts_pred:
                            inconsistencies.append(f"Skor {exact_score} KG VAR olmalı, tahmin KG YOK")
                        elif (home_goals == 0 or away_goals == 0) and 'KG VAR' in btts_pred:
                            inconsistencies.append(f"Skor {exact_score} KG YOK olmalı, tahmin KG VAR")
            
            if inconsistencies:
                self.logger.warning(f"Tutarlılık uyarıları: {'; '.join(inconsistencies)}")
                return False
            else:
                self.logger.info("Final tutarlılık kontrolü başarılı")
                return True
                
        except Exception as e:
            self.logger.error(f"Final tutarlılık kontrolünde hata: {str(e)}")
            return False

def ensure_prediction_consistency(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dışarıdan kullanım için basit wrapper fonksiyon
    
    Args:
        predictions: Tahmin verilerini içeren dictionary
        
    Returns:
        Tutarlılığı garantilenmiş tahmin verisi
    """
    engine = PredictionConsistencyEngine()
    return engine.ensure_consistency(predictions)