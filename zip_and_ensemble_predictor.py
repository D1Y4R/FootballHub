"""
ZIP and Ensemble Predictor Module
Combines multiple prediction models using ensemble techniques
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class ZIPEnsemblePredictor:
    """
    Zero-Inflated Poisson (ZIP) and Ensemble Prediction Model
    Combines multiple prediction approaches for better accuracy
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {
            'poisson': 0.3,
            'neural_network': 0.3,
            'statistical': 0.2,
            'historical': 0.2
        }
        self.prediction_cache = {}
        
    def register_model(self, name: str, model_func, weight: float = 1.0):
        """Register a prediction model with the ensemble"""
        self.models[name] = {
            'function': model_func,
            'weight': weight
        }
        logger.info(f"Model registered: {name} (weight: {weight})")
    
    def predict_match_outcome(self, home_team_id: str, away_team_id: str, 
                            home_form: Dict = None, away_form: Dict = None) -> Dict:
        """
        Predict match outcome using ensemble of models
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            home_form: Home team form data
            away_form: Away team form data
            
        Returns:
            Dict containing ensemble prediction
        """
        try:
            cache_key = f"{home_team_id}_{away_team_id}"
            
            # Check cache first
            if cache_key in self.prediction_cache:
                logger.debug(f"Using cached ensemble prediction for {cache_key}")
                return self.prediction_cache[cache_key]
            
            predictions = {}
            total_weight = 0
            
            # Collect predictions from all registered models
            for model_name, model_info in self.models.items():
                try:
                    model_func = model_info['function']
                    weight = model_info['weight']
                    
                    # Get prediction from model
                    prediction = model_func(home_team_id, away_team_id, home_form, away_form)
                    
                    if prediction:
                        predictions[model_name] = {
                            'prediction': prediction,
                            'weight': weight
                        }
                        total_weight += weight
                        logger.debug(f"Model {model_name} prediction collected (weight: {weight})")
                        
                except Exception as e:
                    logger.warning(f"Error getting prediction from model {model_name}: {str(e)}")
                    continue
            
            if not predictions:
                logger.warning("No ensemble predictions available, using fallback")
                return self._get_fallback_prediction(home_team_id, away_team_id)
            
            # Combine predictions using weighted average
            ensemble_result = self._combine_predictions(predictions, total_weight)
            
            # Cache the result
            self.prediction_cache[cache_key] = ensemble_result
            
            logger.info(f"Ensemble prediction completed for {home_team_id} vs {away_team_id}")
            return ensemble_result
            
        except Exception as e:
            logger.error(f"Ensemble prediction error: {str(e)}")
            return self._get_fallback_prediction(home_team_id, away_team_id)
    
    def _combine_predictions(self, predictions: Dict, total_weight: float) -> Dict:
        """Combine multiple predictions using weighted ensemble"""
        
        # Initialize combined values
        combined_home_goals = 0
        combined_away_goals = 0
        combined_home_win_prob = 0
        combined_draw_prob = 0
        combined_away_win_prob = 0
        combined_over_2_5_prob = 0
        combined_btts_prob = 0
        
        # Weight and combine all predictions
        for model_name, model_data in predictions.items():
            pred = model_data['prediction']
            weight = model_data['weight'] / total_weight  # Normalize weight
            
            # Extract values safely
            home_goals = self._safe_get_numeric(pred, 'home_goals', 1.2)
            away_goals = self._safe_get_numeric(pred, 'away_goals', 1.0)
            home_win_prob = self._safe_get_numeric(pred, 'home_win_prob', 40.0)
            draw_prob = self._safe_get_numeric(pred, 'draw_prob', 30.0)
            away_win_prob = self._safe_get_numeric(pred, 'away_win_prob', 30.0)
            over_2_5_prob = self._safe_get_numeric(pred, 'over_2_5_prob', 50.0)
            btts_prob = self._safe_get_numeric(pred, 'btts_prob', 45.0)
            
            # Weight and add to combined values
            combined_home_goals += home_goals * weight
            combined_away_goals += away_goals * weight
            combined_home_win_prob += home_win_prob * weight
            combined_draw_prob += draw_prob * weight
            combined_away_win_prob += away_win_prob * weight
            combined_over_2_5_prob += over_2_5_prob * weight
            combined_btts_prob += btts_prob * weight
        
        # Create ensemble result
        ensemble_result = {
            'home_goals': round(combined_home_goals, 2),
            'away_goals': round(combined_away_goals, 2),
            'home_win_prob': round(combined_home_win_prob, 1),
            'draw_prob': round(combined_draw_prob, 1),
            'away_win_prob': round(combined_away_win_prob, 1),
            'over_2_5_prob': round(combined_over_2_5_prob, 1),
            'btts_prob': round(combined_btts_prob, 1),
            'prediction_type': 'ensemble',
            'models_used': list(predictions.keys()),
            'confidence': self._calculate_ensemble_confidence(predictions)
        }
        
        # Normalize probabilities to sum to 100%
        total_outcome_prob = ensemble_result['home_win_prob'] + ensemble_result['draw_prob'] + ensemble_result['away_win_prob']
        if total_outcome_prob > 0:
            ensemble_result['home_win_prob'] = round((ensemble_result['home_win_prob'] / total_outcome_prob) * 100, 1)
            ensemble_result['draw_prob'] = round((ensemble_result['draw_prob'] / total_outcome_prob) * 100, 1)
            ensemble_result['away_win_prob'] = round((ensemble_result['away_win_prob'] / total_outcome_prob) * 100, 1)
        
        return ensemble_result
    
    def _safe_get_numeric(self, data: Dict, key: str, default: float) -> float:
        """Safely extract numeric value from prediction data"""
        try:
            value = data.get(key, default)
            if isinstance(value, (int, float)):
                return float(value)
            elif isinstance(value, str):
                return float(value)
            else:
                return default
        except (ValueError, TypeError):
            return default
    
    def _calculate_ensemble_confidence(self, predictions: Dict) -> float:
        """Calculate confidence score based on model agreement"""
        if len(predictions) < 2:
            return 70.0  # Default confidence for single model
        
        # Calculate variance in predictions to measure agreement
        home_goals_values = []
        away_goals_values = []
        
        for model_data in predictions.values():
            pred = model_data['prediction']
            home_goals_values.append(self._safe_get_numeric(pred, 'home_goals', 1.2))
            away_goals_values.append(self._safe_get_numeric(pred, 'away_goals', 1.0))
        
        # Calculate coefficient of variation (lower = more agreement = higher confidence)
        home_cv = np.std(home_goals_values) / np.mean(home_goals_values) if np.mean(home_goals_values) > 0 else 1.0
        away_cv = np.std(away_goals_values) / np.mean(away_goals_values) if np.mean(away_goals_values) > 0 else 1.0
        
        avg_cv = (home_cv + away_cv) / 2
        
        # Convert to confidence score (lower variance = higher confidence)
        confidence = max(50.0, min(95.0, 90.0 - (avg_cv * 100)))
        
        return round(confidence, 1)
    
    def _get_fallback_prediction(self, home_team_id: str, away_team_id: str) -> Dict:
        """Fallback prediction when ensemble fails"""
        return {
            'home_goals': 1.2,
            'away_goals': 1.0,
            'home_win_prob': 40.0,
            'draw_prob': 30.0,
            'away_win_prob': 30.0,
            'over_2_5_prob': 50.0,
            'btts_prob': 45.0,
            'prediction_type': 'fallback',
            'models_used': ['fallback'],
            'confidence': 60.0
        }
    
    def predict_zero_inflated_poisson(self, home_lambda: float, away_lambda: float) -> Dict:
        """
        Zero-Inflated Poisson prediction for low-scoring scenarios
        """
        try:
            import math
            
            def zip_probability(lam: float, k: int, pi: float = 0.1) -> float:
                """Zero-inflated Poisson probability"""
                if k == 0:
                    return pi + (1 - pi) * math.exp(-lam)
                else:
                    poisson_prob = ((lam ** k) * math.exp(-lam)) / math.factorial(k)
                    return (1 - pi) * poisson_prob
            
            # Calculate probabilities for different score combinations
            score_probs = {}
            max_goals = 6
            
            for home_goals in range(max_goals + 1):
                for away_goals in range(max_goals + 1):
                    home_prob = zip_probability(home_lambda, home_goals)
                    away_prob = zip_probability(away_lambda, away_goals)
                    score_probs[f"{home_goals}-{away_goals}"] = home_prob * away_prob
            
            # Find most likely score
            most_likely_score = max(score_probs.items(), key=lambda x: x[1])
            
            # Calculate outcome probabilities
            home_win_prob = sum(prob for score, prob in score_probs.items() 
                              if int(score.split('-')[0]) > int(score.split('-')[1]))
            draw_prob = sum(prob for score, prob in score_probs.items() 
                           if int(score.split('-')[0]) == int(score.split('-')[1]))
            away_win_prob = sum(prob for score, prob in score_probs.items() 
                              if int(score.split('-')[0]) < int(score.split('-')[1]))
            
            # Calculate other betting probabilities
            over_2_5_prob = sum(prob for score, prob in score_probs.items() 
                               if (int(score.split('-')[0]) + int(score.split('-')[1])) > 2)
            btts_prob = sum(prob for score, prob in score_probs.items() 
                           if int(score.split('-')[0]) > 0 and int(score.split('-')[1]) > 0)
            
            return {
                'home_goals': round(home_lambda, 2),
                'away_goals': round(away_lambda, 2),
                'most_likely_score': most_likely_score[0],
                'score_probability': round(most_likely_score[1] * 100, 1),
                'home_win_prob': round(home_win_prob * 100, 1),
                'draw_prob': round(draw_prob * 100, 1),
                'away_win_prob': round(away_win_prob * 100, 1),
                'over_2_5_prob': round(over_2_5_prob * 100, 1),
                'btts_prob': round(btts_prob * 100, 1),
                'prediction_type': 'zero_inflated_poisson'
            }
            
        except Exception as e:
            logger.error(f"ZIP prediction error: {str(e)}")
            return self._get_fallback_prediction("", "")
    
    def clear_cache(self):
        """Clear prediction cache"""
        self.prediction_cache.clear()
        logger.info("Ensemble prediction cache cleared")

# Global ensemble predictor instance
ensemble_predictor = ZIPEnsemblePredictor()

def get_ensemble_prediction(home_team_id: str, away_team_id: str, 
                          home_form: Dict = None, away_form: Dict = None) -> Dict:
    """
    Main function to get ensemble prediction
    """
    return ensemble_predictor.predict_match_outcome(home_team_id, away_team_id, home_form, away_form)

def register_prediction_model(name: str, model_func, weight: float = 1.0):
    """
    Register a new prediction model with the ensemble
    """
    ensemble_predictor.register_model(name, model_func, weight)

def predict_zip(home_lambda: float, away_lambda: float) -> Dict:
    """
    Get Zero-Inflated Poisson prediction
    """
    return ensemble_predictor.predict_zero_inflated_poisson(home_lambda, away_lambda)