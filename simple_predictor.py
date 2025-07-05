"""
Simple Fallback Predictor for FootballHub
Works without external ML dependencies as a backup when main predictor fails
"""

import logging
import json
import os
import math
import random
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SimpleFallbackPredictor:
    """
    A simplified predictor that generates reasonable predictions 
    without requiring external ML libraries or API calls
    """
    
    def __init__(self):
        self.predictions_cache = {}
        self._cache_modified = False
        self.load_cache()
        
        # Simple team ratings based on common team names
        self.team_ratings = {
            # Big European teams
            'Real Madrid': 92, 'Barcelona': 90, 'Manchester City': 91,
            'Liverpool': 89, 'Bayern Munich': 91, 'PSG': 88,
            'Chelsea': 87, 'Manchester United': 86, 'Arsenal': 85,
            'Tottenham': 83, 'Atletico Madrid': 86, 'Inter': 85,
            'Milan': 84, 'Juventus': 83, 'Napoli': 84,
            
            # Turkish teams
            'Galatasaray': 78, 'Fenerbahce': 77, 'Besiktas': 76,
            'Trabzonspor': 74, 'Basaksehir': 72, 'Konyaspor': 70,
            'Antalyaspor': 68, 'Alanyaspor': 69, 'Rizespor': 66,
            'Kasimpasa': 67, 'Samsunspor': 68,
            
            # Default rating for unknown teams
            'default': 70
        }
    
    def load_cache(self):
        """Load previous predictions"""
        try:
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r', encoding='utf-8') as f:
                    self.predictions_cache = json.load(f)
                logger.info(f"Simple predictor cache loaded: {len(self.predictions_cache)} predictions")
        except Exception as e:
            logger.error(f"Error loading simple predictor cache: {str(e)}")
    
    def clear_cache(self):
        """Clear prediction cache"""
        try:
            self.predictions_cache = {}
            with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)
            logger.info("Simple predictor cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing simple predictor cache: {str(e)}")
            return False
    
    def get_team_rating(self, team_name):
        """Get team rating, with fallback for unknown teams"""
        return self.team_ratings.get(team_name, self.team_ratings['default'])
    
    def predict_match(self, home_team_id, away_team_id, home_team_name, away_team_name, force_update=False):
        """
        Generate a simple but reasonable prediction for a match
        """
        try:
            logger.info(f"Simple predictor generating prediction for {home_team_name} vs {away_team_name}")
            
            # Check cache first
            cache_key = f"{home_team_id}_{away_team_id}"
            if cache_key in self.predictions_cache and not force_update:
                cached_prediction = self.predictions_cache[cache_key]
                cached_time = datetime.fromtimestamp(cached_prediction.get('timestamp', 0))
                if datetime.now() - cached_time < timedelta(hours=24):
                    logger.info(f"Returning cached simple prediction for {home_team_name} vs {away_team_name}")
                    return cached_prediction
            
            # Get team ratings
            home_rating = self.get_team_rating(home_team_name)
            away_rating = self.get_team_rating(away_team_name)
            
            # Calculate strength difference
            strength_diff = home_rating - away_rating
            
            # Add home advantage
            home_advantage = 5  # 5 point home advantage
            adjusted_home_rating = home_rating + home_advantage
            
            # Calculate expected goals based on ratings
            home_expected = max(0.5, min(4.0, 1.5 + (adjusted_home_rating - away_rating) / 25))
            away_expected = max(0.5, min(4.0, 1.3 + (away_rating - adjusted_home_rating) / 30))
            
            # Add some randomness for realism
            home_expected += random.uniform(-0.2, 0.2)
            away_expected += random.uniform(-0.2, 0.2)
            
            # Calculate win probabilities
            total_strength = adjusted_home_rating + away_rating
            home_win_prob = max(0.2, min(0.7, (adjusted_home_rating / total_strength) * 1.3))
            away_win_prob = max(0.1, min(0.6, (away_rating / total_strength) * 0.9))
            draw_prob = max(0.15, 1.0 - home_win_prob - away_win_prob)
            
            # Normalize probabilities
            total_prob = home_win_prob + draw_prob + away_win_prob
            home_win_prob /= total_prob
            draw_prob /= total_prob
            away_win_prob /= total_prob
            
            # Calculate betting predictions
            total_goals_expected = home_expected + away_expected
            
            # Both teams to score
            btts_prob = max(0.3, min(0.8, 0.6 if min(home_expected, away_expected) > 0.8 else 0.4))
            btts_prediction = "KG VAR" if btts_prob > 0.5 else "KG YOK"
            
            # Over/Under 2.5
            over_25_prob = max(0.2, min(0.8, 0.7 if total_goals_expected > 2.5 else 0.3))
            over_25_prediction = "2.5 ÜST" if over_25_prob > 0.5 else "2.5 ALT"
            
            # Over/Under 3.5
            over_35_prob = max(0.1, min(0.7, 0.5 if total_goals_expected > 3.5 else 0.2))
            over_35_prediction = "3.5 ÜST" if over_35_prob > 0.5 else "3.5 ALT"
            
            # Most likely score
            home_score = max(0, round(home_expected))
            away_score = max(0, round(away_expected))
            exact_score = f"{home_score}-{away_score}"
            
            # Generate simple form data
            home_form = self._generate_simple_form_data(home_team_name, home_rating)
            away_form = self._generate_simple_form_data(away_team_name, away_rating)
            
            # Create prediction response
            prediction = {
                "home_team": {
                    "id": home_team_id,
                    "name": home_team_name,
                    "form": home_form
                },
                "away_team": {
                    "id": away_team_id,
                    "name": away_team_name,
                    "form": away_form
                },
                "predictions": {
                    "expected_goals": {
                        "home": round(home_expected, 2),
                        "away": round(away_expected, 2)
                    },
                    "match_winner": {
                        "home_win": round(home_win_prob, 3),
                        "draw": round(draw_prob, 3),
                        "away_win": round(away_win_prob, 3)
                    },
                    "betting_predictions": {
                        "both_teams_to_score": {
                            "prediction": btts_prediction,
                            "probability": round(btts_prob * 100, 1)
                        },
                        "over_2_5_goals": {
                            "prediction": over_25_prediction,
                            "probability": round(over_25_prob * 100, 1)
                        },
                        "over_3_5_goals": {
                            "prediction": over_35_prediction,
                            "probability": round(over_35_prob * 100, 1)
                        },
                        "exact_score": {
                            "prediction": exact_score,
                            "probability": round(15.0 + random.uniform(-5, 5), 1)
                        }
                    },
                    "home_win_probability": round(home_win_prob * 100, 1),
                    "draw_probability": round(draw_prob * 100, 1),
                    "away_win_probability": round(away_win_prob * 100, 1),
                    "exact_score": exact_score,
                    "most_likely_outcome": (
                        "HOME_WIN" if home_win_prob > max(draw_prob, away_win_prob)
                        else "AWAY_WIN" if away_win_prob > draw_prob
                        else "DRAW"
                    ),
                    "confidence": round(max(home_win_prob, draw_prob, away_win_prob) * 100, 1),
                    "explanation": {
                        "exact_score": f"Based on team strengths, {exact_score} is the most likely scoreline.",
                        "match_result": f"Team ratings: {home_team_name} ({home_rating}) vs {away_team_name} ({away_rating})",
                        "relative_strength": f"Strength difference: {strength_diff} points in favor of {'home' if strength_diff > 0 else 'away' if strength_diff < 0 else 'neither'} team",
                        "head_to_head": "Historical data not available in simple mode"
                    },
                    "intelligent_summary": {
                        "main_prediction": (
                            f"{home_team_name} favored to win" if home_win_prob > 0.5
                            else f"{away_team_name} favored to win" if away_win_prob > 0.5
                            else "Close match, could go either way"
                        ),
                        "probability_percentage": round(max(home_win_prob, draw_prob, away_win_prob) * 100, 1)
                    }
                },
                "head_to_head": {
                    "total_matches": 0,
                    "home_wins": 0,
                    "draws": 0,
                    "away_wins": 0,
                    "recent_matches": []
                },
                "timestamp": datetime.now().timestamp(),
                "from_cache": False,
                "predictor_type": "simple_fallback",
                "note": "Generated by simple fallback predictor due to main predictor unavailability"
            }
            
            # Cache the prediction
            self.predictions_cache[cache_key] = prediction
            self._cache_modified = True
            
            logger.info(f"Simple prediction generated successfully for {home_team_name} vs {away_team_name}")
            return prediction
            
        except Exception as e:
            logger.error(f"Error in simple predictor: {str(e)}")
            return {
                "error": f"Simple prediction failed: {str(e)}",
                "match": f"{home_team_name} vs {away_team_name}",
                "timestamp": datetime.now().timestamp()
            }
    
    def _generate_simple_form_data(self, team_name, rating):
        """Generate simple form data based on team rating"""
        # Convert rating to performance metrics
        avg_goals_scored = max(0.5, min(3.0, (rating - 50) / 15))
        avg_goals_conceded = max(0.5, min(3.0, (100 - rating) / 20))
        form_points = max(0.2, min(1.0, rating / 100))
        
        # Generate some fake recent matches
        recent_matches = []
        for i in range(5):
            if rating > 75:
                result = random.choices(['W', 'D', 'L'], weights=[0.6, 0.3, 0.1])[0]
            elif rating > 60:
                result = random.choices(['W', 'D', 'L'], weights=[0.4, 0.4, 0.2])[0]
            else:
                result = random.choices(['W', 'D', 'L'], weights=[0.2, 0.3, 0.5])[0]
                
            goals_scored = random.randint(0, 3) if result != 'L' else random.randint(0, 2)
            goals_conceded = random.randint(0, 2) if result == 'W' else random.randint(1, 3)
            
            recent_matches.append({
                "result": result,
                "goals_scored": goals_scored,
                "goals_conceded": goals_conceded,
                "opponent": f"Opponent {i+1}",
                "date": (datetime.now() - timedelta(days=(i+1)*7)).strftime("%Y-%m-%d"),
                "is_home": random.choice([True, False])
            })
        
        return {
            "avg_goals_scored": round(avg_goals_scored, 2),
            "avg_goals_conceded": round(avg_goals_conceded, 2),
            "form_points": round(form_points, 2),
            "recent_matches": 5,
            "recent_match_data": recent_matches,
            "home_performance": {
                "avg_goals_scored": round(avg_goals_scored * 1.1, 2),
                "avg_goals_conceded": round(avg_goals_conceded * 0.9, 2)
            },
            "away_performance": {
                "avg_goals_scored": round(avg_goals_scored * 0.9, 2),
                "avg_goals_conceded": round(avg_goals_conceded * 1.1, 2)
            }
        }