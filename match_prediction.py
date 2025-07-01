#!/usr/bin/env python3
"""
Optimized Match Prediction Module
Streamlined version with improved performance and reduced complexity.
"""

import logging
import json
import os
import math
import sys
from datetime import datetime, timedelta

# Use optimized imports
from lazy_ml_imports import safe_import_numpy, safe_import_pandas, safe_import_sklearn, safe_import_tensorflow
from optimized_http_client import http_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for ML dependencies
np = safe_import_numpy()
pd = safe_import_pandas()
RandomForestRegressor, StandardScaler, train_test_split = safe_import_sklearn()
tf = safe_import_tensorflow()

# TensorFlow components (with fallbacks)
Sequential = getattr(tf.keras.models, 'Sequential', None) if hasattr(tf, 'keras') else None
load_model = getattr(tf.keras.models, 'load_model', None) if hasattr(tf, 'keras') else None
save_model = getattr(tf.keras.models, 'save_model', None) if hasattr(tf, 'keras') else None
Dense = getattr(tf.keras.layers, 'Dense', None) if hasattr(tf, 'keras') else None
Dropout = getattr(tf.keras.layers, 'Dropout', None) if hasattr(tf, 'keras') else None
EarlyStopping = getattr(tf.keras.callbacks, 'EarlyStopping', None) if hasattr(tf, 'keras') else None

class MatchPredictor:
    """
    Optimized football match prediction system.
    Simplified and streamlined for better performance.
    """
    
    def __init__(self):
        self.api_key = os.environ.get('APIFOOTBALL_PREMIUM_KEY', '908ca1caaca4f5470f8c9d7f01a02d66fa06d149e77627804796c4f12568a485')
        
        # Use optimized cache manager
        try:
            from cache_optimizer import cache_manager
            self.cache_manager = cache_manager
            self.predictions_cache = self.cache_manager._cache_data
            self._cache_modified = False
        except ImportError:
            logger.warning("Optimized cache not available, using basic cache")
            self.predictions_cache = {}
            self._cache_modified = False
            self.load_cache()

        # Core prediction parameters
        self.lig_ortalamasi_ev_gol = 1.5
        self.k_ev = 5
        self.k_deplasman = 5
        self.alpha_ev_atma = self.k_ev
        self.beta_ev = self.k_ev
        self.alpha_deplasman_atma = self.k_deplasman
        self.beta_deplasman = self.k_deplasman

        # ML Models
        self.scaler = StandardScaler() if StandardScaler else None
        self.model_home = None
        self.model_away = None
        self.input_dim = 10  # Default dimension
        
        # Initialize models
        self.load_or_create_models()
        
        # Team data for optimizations
        self.big_teams = self._load_big_teams()
        self.team_adjustments = self._load_team_adjustments()

        logger.info("MatchPredictor initialized with optimized configuration")

    def _load_big_teams(self):
        """Load big teams list (optimized)"""
        return {
            'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Bayern Munich', 'Borussia Dortmund',
            'Manchester City', 'Manchester United', 'Liverpool', 'Chelsea', 'Arsenal', 'Tottenham',
            'PSG', 'Monaco', 'Lyon', 'Marseille', 'Inter', 'Milan', 'Juventus', 'Napoli', 'Roma',
            'Galatasaray', 'Fenerbahce', 'Besiktas', 'Trabzonspor', 'Benfica', 'Porto', 'Sporting',
            'Ajax', 'PSV', 'Feyenoord', 'Celtic', 'Rangers'
        }

    def _load_team_adjustments(self):
        """Load team-specific adjustment factors (optimized)"""
        return {
            # Home/Away asymmetries
            "610": {"name": "Galatasaray", "home_factor": 1.40, "away_factor": 0.85},
            "1005": {"name": "Fenerbahçe", "home_factor": 1.35, "away_factor": 0.90},
            "614": {"name": "Beşiktaş", "home_factor": 1.30, "away_factor": 0.90},
            "636": {"name": "Trabzonspor", "home_factor": 1.35, "away_factor": 0.85},
            
            # Defensive weaknesses
            "defensive_weak": {
                "7667": {"name": "Adana Demirspor", "factor": 1.35, "offensive_factor": 1.15},
                "621": {"name": "Kasimpasa", "factor": 1.30, "offensive_factor": 1.12},
                "611": {"name": "Rizespor", "factor": 1.28, "offensive_factor": 1.10},
            }
        }

    def load_cache(self):
        """Load prediction cache (fallback method)"""
        try:
            if os.path.exists('predictions_cache.json'):
                with open('predictions_cache.json', 'r', encoding='utf-8') as f:
                    self.predictions_cache = json.load(f)
                logger.info(f"Cache loaded: {len(self.predictions_cache)} predictions")
        except Exception as e:
            logger.error(f"Cache loading error: {e}")
            self.predictions_cache = {}

    def clear_cache(self):
        """Clear prediction cache"""
        try:
            if hasattr(self, 'cache_manager'):
                return self.cache_manager.clear()
            else:
                self.predictions_cache = {}
                with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False

    def save_cache(self):
        """Save prediction cache"""
        if not self._cache_modified:
            return
            
        try:
            if hasattr(self, 'cache_manager'):
                self.cache_manager.save_cache()
            else:
                with open('predictions_cache.json', 'w', encoding='utf-8') as f:
                    json.dump(self.predictions_cache, f, ensure_ascii=False, indent=2)
            
            self._cache_modified = False
            logger.debug("Cache saved successfully")
        except Exception as e:
            logger.error(f"Cache save error: {e}")

    def is_big_team(self, team_name):
        """Check if team is considered a big team"""
        return team_name in self.big_teams

    def load_or_create_models(self):
        """Load or create neural network models"""
        try:
            if os.path.exists('model_home.h5') and os.path.exists('model_away.h5') and load_model:
                logger.info("Loading pre-trained models...")
                self.model_home = load_model('model_home.h5')
                self.model_away = load_model('model_away.h5')
            else:
                logger.info("Creating new models...")
                self.model_home = self.build_neural_network(self.input_dim)
                self.model_away = self.build_neural_network(self.input_dim)
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            self.model_home = self.build_neural_network(self.input_dim)
            self.model_away = self.build_neural_network(self.input_dim)

    def build_neural_network(self, input_dim):
        """Build optimized neural network model"""
        if not Sequential or not Dense:
            logger.warning("TensorFlow not available, using mock model")
            return None
            
        try:
            model = Sequential([
                Dense(64, activation='relu', input_shape=(input_dim,)),
                Dropout(0.3) if Dropout else Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dropout(0.2) if Dropout else Dense(32, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return model
        except Exception as e:
            logger.error(f"Neural network creation error: {e}")
            return None

    def get_team_form(self, team_id, last_matches=15):
        """Get team form data (optimized)"""
        try:
            url = "https://apiv3.apifootball.com/"
            params = {
                'action': 'get_events',
                'from': (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'team_id': team_id,
                'APIkey': self.api_key
            }
            
            # Use optimized HTTP client
            response = http_client.get(url, params=params)
            
            if response.status_code == 200:
                matches = response.json()
                return self._process_team_form(matches, team_id, last_matches)
            else:
                logger.warning(f"API error for team {team_id}: {response.status_code}")
                return self._generate_fallback_form_data(team_id)
                
        except Exception as e:
            logger.error(f"Error fetching team form for {team_id}: {e}")
            return self._generate_fallback_form_data(team_id)

    def _process_team_form(self, matches, team_id, last_matches):
        """Process team form data efficiently"""
        if not matches or len(matches) == 0:
            return self._generate_fallback_form_data(team_id)

        # Sort matches by date (most recent first)
        matches.sort(key=lambda x: x.get('match_date', ''), reverse=True)
        recent_matches = matches[:last_matches]

        # Calculate basic statistics
        total_matches = len(recent_matches)
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        home_matches = away_matches = 0

        for match in recent_matches:
            home_team_id = str(match.get('match_hometeam_id', ''))
            away_team_id = str(match.get('match_awayteam_id', ''))
            
            home_score = int(match.get('match_hometeam_score', 0) or 0)
            away_score = int(match.get('match_awayteam_score', 0) or 0)

            if home_team_id == str(team_id):
                # Home match
                home_matches += 1
                goals_scored += home_score
                goals_conceded += away_score
                
                if home_score > away_score:
                    wins += 1
                elif home_score == away_score:
                    draws += 1
                else:
                    losses += 1
                    
            elif away_team_id == str(team_id):
                # Away match
                away_matches += 1
                goals_scored += away_score
                goals_conceded += home_score
                
                if away_score > home_score:
                    wins += 1
                elif away_score == home_score:
                    draws += 1
                else:
                    losses += 1

        # Calculate averages
        avg_goals_scored = goals_scored / total_matches if total_matches > 0 else 0
        avg_goals_conceded = goals_conceded / total_matches if total_matches > 0 else 0
        
        return {
            'matches_played': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_scored': goals_scored,
            'goals_conceded': goals_conceded,
            'avg_goals_scored': avg_goals_scored,
            'avg_goals_conceded': avg_goals_conceded,
            'home_matches': home_matches,
            'away_matches': away_matches,
            'win_percentage': wins / total_matches if total_matches > 0 else 0,
            'points': (wins * 3 + draws),
            'recent_match_data': recent_matches[:5],  # Last 5 matches for detailed analysis
            'form_strength': self._calculate_form_strength(wins, draws, losses, avg_goals_scored),
            'bayesian': self._calculate_bayesian_stats(avg_goals_scored, avg_goals_conceded)
        }

    def _calculate_form_strength(self, wins, draws, losses, avg_goals):
        """Calculate team form strength (simplified)"""
        total_matches = wins + draws + losses
        if total_matches == 0:
            return 0.5
            
        win_rate = wins / total_matches
        point_rate = (wins * 3 + draws) / (total_matches * 3)
        goal_factor = min(1.0, avg_goals / 1.5)  # Normalize around 1.5 goals
        
        return (win_rate * 0.4 + point_rate * 0.4 + goal_factor * 0.2)

    def _calculate_bayesian_stats(self, avg_goals_scored, avg_goals_conceded):
        """Calculate Bayesian statistics (simplified)"""
        return {
            'home_lambda_scored': avg_goals_scored * 1.2,  # Home advantage
            'away_lambda_scored': avg_goals_scored * 0.9,  # Away disadvantage
            'home_lambda_conceded': avg_goals_conceded * 0.9,
            'away_lambda_conceded': avg_goals_conceded * 1.1
        }

    def _generate_fallback_form_data(self, team_id):
        """Generate fallback form data when API fails"""
        return {
            'matches_played': 10,
            'wins': 4,
            'draws': 3,
            'losses': 3,
            'goals_scored': 15,
            'goals_conceded': 12,
            'avg_goals_scored': 1.5,
            'avg_goals_conceded': 1.2,
            'home_matches': 5,
            'away_matches': 5,
            'win_percentage': 0.4,
            'points': 15,
            'recent_match_data': [],
            'form_strength': 0.5,
            'bayesian': {
                'home_lambda_scored': 1.8,
                'away_lambda_scored': 1.35,
                'home_lambda_conceded': 1.08,
                'away_lambda_conceded': 1.32
            }
        }

    def apply_team_adjustments(self, home_team_id, away_team_id, home_goals, away_goals):
        """Apply team-specific adjustments (simplified)"""
        try:
            # Apply home/away asymmetries
            if str(home_team_id) in self.team_adjustments:
                team_info = self.team_adjustments[str(home_team_id)]
                home_goals *= team_info["home_factor"]
                
            if str(away_team_id) in self.team_adjustments:
                team_info = self.team_adjustments[str(away_team_id)]
                away_goals *= team_info["away_factor"]

            # Apply defensive weakness adjustments
            defensive_weak = self.team_adjustments.get("defensive_weak", {})
            
            if str(home_team_id) in defensive_weak:
                team_info = defensive_weak[str(home_team_id)]
                away_goals *= team_info["factor"]
                home_goals *= team_info["offensive_factor"]
                
            if str(away_team_id) in defensive_weak:
                team_info = defensive_weak[str(away_team_id)]
                home_goals *= team_info["factor"]
                away_goals *= team_info["offensive_factor"]

            return home_goals, away_goals
            
        except Exception as e:
            logger.error(f"Team adjustment error: {e}")
            return home_goals, away_goals

    def monte_carlo_simulation(self, home_lambda, away_lambda, simulations=10000):
        """Optimized Monte Carlo simulation"""
        try:
            if not np or not hasattr(np, 'random'):
                # Fallback simulation without numpy
                return self._basic_simulation(home_lambda, away_lambda, simulations)
            
            # Generate random samples
            home_goals = np.random.poisson(home_lambda, simulations)
            away_goals = np.random.poisson(away_lambda, simulations)
            
            # Count outcomes
            home_wins = np.sum(home_goals > away_goals)
            draws = np.sum(home_goals == away_goals)
            away_wins = np.sum(home_goals < away_goals)
            
            # Over/Under 2.5
            total_goals = home_goals + away_goals
            over_2_5 = np.sum(total_goals > 2.5)
            
            # Both teams to score
            btts = np.sum((home_goals > 0) & (away_goals > 0))
            
            # Most common scores
            scores = {}
            for h, a in zip(home_goals, away_goals):
                score = f"{h}-{a}"
                scores[score] = scores.get(score, 0) + 1
            
            most_likely_score = max(scores.items(), key=lambda x: x[1])
            
            return {
                'home_win_prob': home_wins / simulations,
                'draw_prob': draws / simulations,
                'away_win_prob': away_wins / simulations,
                'over_2_5_prob': over_2_5 / simulations,
                'btts_prob': btts / simulations,
                'most_likely_score': most_likely_score[0],
                'avg_home_goals': float(np.mean(home_goals)),
                'avg_away_goals': float(np.mean(away_goals)),
                'total_simulations': simulations
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return self._basic_simulation(home_lambda, away_lambda, simulations)

    def _basic_simulation(self, home_lambda, away_lambda, simulations):
        """Basic simulation without numpy (fallback)"""
        import random
        
        home_wins = draws = away_wins = 0
        over_2_5 = btts = 0
        total_home_goals = total_away_goals = 0
        scores = {}
        
        for _ in range(simulations):
            # Simple Poisson approximation
            h_goals = max(0, int(random.expovariate(1/home_lambda)) if home_lambda > 0 else 0)
            a_goals = max(0, int(random.expovariate(1/away_lambda)) if away_lambda > 0 else 0)
            
            total_home_goals += h_goals
            total_away_goals += a_goals
            
            if h_goals > a_goals:
                home_wins += 1
            elif h_goals == a_goals:
                draws += 1
            else:
                away_wins += 1
                
            if h_goals + a_goals > 2.5:
                over_2_5 += 1
                
            if h_goals > 0 and a_goals > 0:
                btts += 1
                
            score = f"{h_goals}-{a_goals}"
            scores[score] = scores.get(score, 0) + 1
        
        most_likely_score = max(scores.items(), key=lambda x: x[1]) if scores else ("1-1", 1)
        
        return {
            'home_win_prob': home_wins / simulations,
            'draw_prob': draws / simulations,
            'away_win_prob': away_wins / simulations,
            'over_2_5_prob': over_2_5 / simulations,
            'btts_prob': btts / simulations,
            'most_likely_score': most_likely_score[0],
            'avg_home_goals': total_home_goals / simulations,
            'avg_away_goals': total_away_goals / simulations,
            'total_simulations': simulations
        }

    def predict_match(self, home_team_id, away_team_id, home_team_name, away_team_name, force_update=False):
        """Main prediction method (optimized)"""
        cache_key = f"prediction_{home_team_id}_{away_team_id}_{home_team_name}_{away_team_name}"
        
        # Check cache first
        if not force_update and cache_key in self.predictions_cache:
            cached_prediction = self.predictions_cache[cache_key]
            if self._is_cache_valid(cached_prediction):
                logger.info(f"Using cached prediction for {home_team_name} vs {away_team_name}")
                cached_prediction['from_cache'] = True
                return cached_prediction

        logger.info(f"Generating new prediction: {home_team_name} vs {away_team_name}")
        
        try:
            # Get team forms
            home_form = self.get_team_form(home_team_id)
            away_form = self.get_team_form(away_team_id)
            
            # Calculate base goal expectations
            home_lambda = home_form['bayesian']['home_lambda_scored']
            away_lambda = away_form['bayesian']['away_lambda_scored']
            
            # Apply team adjustments
            home_lambda, away_lambda = self.apply_team_adjustments(
                home_team_id, away_team_id, home_lambda, away_lambda
            )
            
            # Run Monte Carlo simulation
            simulation_results = self.monte_carlo_simulation(home_lambda, away_lambda)
            
            # Generate final prediction
            prediction = {
                'timestamp': datetime.now().isoformat(),
                'home_team': {
                    'id': home_team_id,
                    'name': home_team_name,
                    'form': home_form
                },
                'away_team': {
                    'id': away_team_id,
                    'name': away_team_name,
                    'form': away_form
                },
                'predictions': {
                    'match_winner': {
                        'home_win': simulation_results['home_win_prob'],
                        'draw': simulation_results['draw_prob'],
                        'away_win': simulation_results['away_win_prob']
                    },
                    'exact_score': {
                        'score': simulation_results['most_likely_score'],
                        'confidence': 0.8
                    },
                    'betting_predictions': {
                        'over_2_5_goals': {
                            'prediction': 'YES' if simulation_results['over_2_5_prob'] > 0.5 else 'NO',
                            'probability': simulation_results['over_2_5_prob'],
                            'display_value': f"{simulation_results['over_2_5_prob']:.2f}"
                        },
                        'btts': {
                            'prediction': 'YES' if simulation_results['btts_prob'] > 0.5 else 'NO',
                            'probability': simulation_results['btts_prob'],
                            'display_value': f"{simulation_results['btts_prob']:.2f}"
                        }
                    }
                },
                'expected_goals': {
                    'home': simulation_results['avg_home_goals'],
                    'away': simulation_results['avg_away_goals']
                },
                'confidence': self._calculate_confidence(home_form, away_form),
                'from_cache': False
            }
            
            # Cache the prediction
            self.predictions_cache[cache_key] = prediction
            self._cache_modified = True
            
            logger.info(f"Prediction generated: {home_team_name} vs {away_team_name}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error for {home_team_name} vs {away_team_name}: {e}")
            return self._generate_fallback_prediction(home_team_name, away_team_name)

    def _is_cache_valid(self, cached_prediction, max_age_hours=24):
        """Check if cached prediction is still valid"""
        try:
            timestamp = cached_prediction.get('timestamp')
            if not timestamp:
                return False
                
            cache_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.now() - cache_time.replace(tzinfo=None)
            
            return age.total_seconds() < (max_age_hours * 3600)
        except Exception:
            return False

    def _calculate_confidence(self, home_form, away_form):
        """Calculate prediction confidence (simplified)"""
        try:
            home_matches = home_form.get('matches_played', 0)
            away_matches = away_form.get('matches_played', 0)
            
            # Base confidence on data quality
            data_quality = min(1.0, (home_matches + away_matches) / 20)
            
            # Adjust for form consistency
            home_consistency = home_form.get('form_strength', 0.5)
            away_consistency = away_form.get('form_strength', 0.5)
            
            form_factor = (home_consistency + away_consistency) / 2
            
            return max(0.6, min(0.95, data_quality * 0.7 + form_factor * 0.3))
            
        except Exception:
            return 0.75  # Default confidence

    def _generate_fallback_prediction(self, home_team_name, away_team_name):
        """Generate fallback prediction when main prediction fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'home_team': {'name': home_team_name, 'form': self._generate_fallback_form_data('0')},
            'away_team': {'name': away_team_name, 'form': self._generate_fallback_form_data('0')},
            'predictions': {
                'match_winner': {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30},
                'exact_score': {'score': '1-1', 'confidence': 0.6},
                'betting_predictions': {
                    'over_2_5_goals': {'prediction': 'NO', 'probability': 0.45, 'display_value': '0.45'},
                    'btts': {'prediction': 'YES', 'probability': 0.55, 'display_value': '0.55'}
                }
            },
            'expected_goals': {'home': 1.3, 'away': 1.1},
            'confidence': 0.6,
            'from_cache': False,
            'fallback': True
        }

# Global instance for backward compatibility
try:
    from optimized_http_client import api_manager
    # Use optimized HTTP client if available
    match_predictor = MatchPredictor()
    logger.info("Optimized MatchPredictor instance created")
except ImportError:
    # Fallback to basic version
    match_predictor = MatchPredictor()
    logger.warning("Using basic MatchPredictor (optimization modules not available)")