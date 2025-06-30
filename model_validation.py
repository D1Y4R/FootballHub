#!/usr/bin/env python3
"""
Optimized Model Validation and Evaluation Module
Streamlined version with improved performance and reduced complexity.
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta

# Use optimized imports
from lazy_ml_imports import safe_import_numpy, safe_import_pandas, safe_import_sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Safe imports for ML dependencies
np = safe_import_numpy()
pd = safe_import_pandas()

# Sklearn components (with fallbacks)
try:
    sklearn_components = safe_import_sklearn()
    if sklearn_components:
        from sklearn.model_selection import KFold, TimeSeriesSplit
        from sklearn.metrics import accuracy_score, mean_squared_error
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        SKLEARN_AVAILABLE = True
    else:
        raise ImportError("Sklearn not available")
except ImportError:
    logger.warning("Sklearn not available, using fallback implementations")
    SKLEARN_AVAILABLE = False
    
    # Fallback implementations
    class KFold:
        def __init__(self, n_splits=5, *args, **kwargs):
            self.n_splits = n_splits
        def split(self, X, y=None):
            n = len(X)
            fold_size = n // self.n_splits
            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else n
                test_idx = list(range(start, end))
                train_idx = list(range(0, start)) + list(range(end, n))
                yield train_idx, test_idx

    class TimeSeriesSplit:
        def __init__(self, n_splits=5, *args, **kwargs):
            self.n_splits = n_splits
        def split(self, X, y=None):
            n = len(X)
            fold_size = n // (self.n_splits + 1)
            for i in range(1, self.n_splits + 1):
                train_end = i * fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, n)
                train_idx = list(range(0, train_end))
                test_idx = list(range(test_start, test_end))
                yield train_idx, test_idx

    def accuracy_score(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0
        return sum(1 for i in range(len(y_true)) if abs(y_true[i] - y_pred[i]) < 0.5) / len(y_true)

    def mean_squared_error(y_true, y_pred):
        if len(y_true) == 0:
            return 0.0
        return sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true))) / len(y_true)

    class RandomForestRegressor:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [1.5] * len(X)  # Default prediction

    class GradientBoostingRegressor:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [1.5] * len(X)  # Default prediction

    class LinearRegression:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X, y):
            return self
        def predict(self, X):
            return [1.5] * len(X)  # Default prediction

    class StandardScaler:
        def __init__(self, *args, **kwargs):
            pass
        def fit(self, X):
            return self
        def transform(self, X):
            return X
        def fit_transform(self, X):
            return X

# Utility functions
def numpy_to_python(obj):
    """Convert NumPy values to Python native types for JSON serialization"""
    if not np:
        return obj
        
    try:
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [numpy_to_python(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: numpy_to_python(value) for key, value in obj.items()}
    except Exception:
        pass
    return obj

def calculate_time_weight(match_date, reference_date=None, max_days=180, min_weight=0.3):
    """Calculate time-based weight for matches (simplified)"""
    try:
        if reference_date is None:
            reference_date = datetime.now()
        elif isinstance(reference_date, str):
            reference_date = datetime.strptime(reference_date, "%Y-%m-%d")
            
        if isinstance(match_date, str):
            match_date = datetime.strptime(match_date.split(" ")[0], "%Y-%m-%d")
            
        days_diff = (reference_date - match_date).days
        
        if days_diff < 0:
            return 1.0
            
        normalized_diff = min(1.0, days_diff / max_days)
        weight = max(min_weight, 1.0 - normalized_diff)
        
        return weight
    except Exception as e:
        logger.error(f"Time weight calculation error: {e}")
        return 1.0

def calculate_form_trend(results, window=5):
    """Calculate form trend (simplified)"""
    if not results or len(results) < window:
        return 0.0
        
    recent = results[-window:]
    mid = len(recent) // 2
    recent_half = recent[mid:]
    older_half = recent[:mid]
    
    recent_avg = sum(recent_half) / len(recent_half)
    older_avg = sum(older_half) / len(older_half)
    
    if older_avg == 0:
        return 1.0 if recent_avg > 0 else 0.0
    
    change = (recent_avg - older_avg) / max(older_avg, 3)
    return max(min(change, 1.0), -1.0)

class OptimizedEnsembleModel:
    """Simplified ensemble model for predictions"""
    
    def __init__(self, models=None):
        if models is None:
            self.models = [
                ('lr', LinearRegression()),
                ('rf', RandomForestRegressor(n_estimators=50)),
                ('gbm', GradientBoostingRegressor(n_estimators=50))
            ]
        else:
            self.models = models
        self.weights = [1.0 / len(self.models)] * len(self.models)
        self.fitted_models = {}

    def fit(self, X, y):
        """Train all models in the ensemble"""
        try:
            for name, model in self.models:
                logger.debug(f"Training {name} model...")
                model.fit(X, y)
                self.fitted_models[name] = model
        except Exception as e:
            logger.error(f"Ensemble training error: {e}")
        return self

    def predict(self, X):
        """Make weighted predictions from all models"""
        try:
            if not self.fitted_models:
                return [1.5] * len(X)  # Default prediction
                
            predictions = []
            for name, model in self.fitted_models.items():
                pred = model.predict(X)
                predictions.append(pred)
            
            if not predictions:
                return [1.5] * len(X)
            
            # Weighted average
            ensemble_pred = []
            for i in range(len(X)):
                weighted_sum = sum(pred[i] * weight for pred, weight in zip(predictions, self.weights))
                ensemble_pred.append(weighted_sum)
            
            return ensemble_pred
        except Exception as e:
            logger.error(f"Ensemble prediction error: {e}")
            return [1.5] * len(X)

    def update_weights(self, predictions, actual_results):
        """Update model weights based on performance"""
        try:
            if len(predictions) != len(self.models):
                return
                
            errors = []
            for pred in predictions:
                error = mean_squared_error(actual_results, pred)
                errors.append(error)
            
            # Inverse error weighting
            if all(e > 0 for e in errors):
                inv_errors = [1.0 / e for e in errors]
                total_inv = sum(inv_errors)
                self.weights = [ie / total_inv for ie in inv_errors]
            
        except Exception as e:
            logger.error(f"Weight update error: {e}")

class ModelValidator:
    """Optimized model validation system"""
    
    def __init__(self, predictor, cache_file='predictions_cache.json'):
        self.predictor = predictor
        self.cache_file = cache_file
        self.validation_results = {}
        self.scaler = StandardScaler()
        
        # Load existing validation results
        self._load_validation_results()

    def _load_validation_results(self):
        """Load previous validation results"""
        try:
            if os.path.exists('validation_results.json'):
                with open('validation_results.json', 'r', encoding='utf-8') as f:
                    self.validation_results = json.load(f)
                logger.info(f"Loaded {len(self.validation_results)} validation results")
        except Exception as e:
            logger.error(f"Error loading validation results: {e}")
            self.validation_results = {}

    def save_validation_results(self):
        """Save validation results"""
        try:
            with open('validation_results.json', 'w', encoding='utf-8') as f:
                json.dump(numpy_to_python(self.validation_results), f, 
                         ensure_ascii=False, indent=2)
            logger.info("Validation results saved")
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")

    def _prepare_data_from_cache(self, use_time_weights=True, max_days=180):
        """Prepare training data from prediction cache"""
        try:
            if not os.path.exists(self.cache_file):
                logger.warning("Cache file not found")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            if not cache_data:
                logger.warning("Empty cache data")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Extract features and targets
            features = []
            home_goals = []
            away_goals = []
            dates = []

            for prediction in cache_data.values():
                if not isinstance(prediction, dict):
                    continue

                # Extract basic features
                home_form = prediction.get('home_team', {}).get('form', {})
                away_form = prediction.get('away_team', {}).get('form', {})
                
                feature_row = self._extract_basic_features(home_form, away_form)
                if feature_row is None:
                    continue

                # Extract targets
                expected_goals = prediction.get('expected_goals', {})
                h_goals = expected_goals.get('home', 1.5)
                a_goals = expected_goals.get('away', 1.2)

                features.append(feature_row)
                home_goals.append(h_goals)
                away_goals.append(a_goals)
                dates.append(prediction.get('timestamp', datetime.now().isoformat()))

            if not features:
                logger.warning("No valid features extracted")
                return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

            # Create DataFrames
            feature_names = [f'feature_{i}' for i in range(len(features[0]))]
            df_features = pd.DataFrame(features, columns=feature_names)
            df_home_goals = pd.DataFrame({'home_goals': home_goals, 'date': dates})
            df_away_goals = pd.DataFrame({'away_goals': away_goals, 'date': dates})

            # Apply time weights if requested
            if use_time_weights:
                weights = [calculate_time_weight(date, max_days=max_days) for date in dates]
                df_features['weight'] = weights
                df_home_goals['weight'] = weights
                df_away_goals['weight'] = weights

            logger.info(f"Prepared {len(df_features)} samples for training")
            return df_features, df_home_goals, df_away_goals

        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def _extract_basic_features(self, home_form, away_form):
        """Extract basic features from team form data"""
        try:
            features = []
            
            # Home team features
            features.extend([
                home_form.get('avg_goals_scored', 1.5),
                home_form.get('avg_goals_conceded', 1.2),
                home_form.get('wins', 0) / max(home_form.get('matches_played', 1), 1),
                home_form.get('form_strength', 0.5),
                home_form.get('home_matches', 0) / max(home_form.get('matches_played', 1), 1)
            ])
            
            # Away team features
            features.extend([
                away_form.get('avg_goals_scored', 1.2),
                away_form.get('avg_goals_conceded', 1.4),
                away_form.get('wins', 0) / max(away_form.get('matches_played', 1), 1),
                away_form.get('form_strength', 0.5),
                away_form.get('away_matches', 0) / max(away_form.get('matches_played', 1), 1)
            ])
            
            # Relative features
            home_attack = features[0]
            away_defense = features[6]
            away_attack = features[5]
            home_defense = features[1]
            
            features.extend([
                home_attack / max(away_defense, 0.5),  # Home attack vs Away defense
                away_attack / max(home_defense, 0.5),  # Away attack vs Home defense
                (features[2] - features[7]),           # Win rate difference
                (features[3] - features[8])            # Form difference
            ])

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def cross_validate(self, k_folds=5, use_time_weights=True):
        """Perform cross-validation (simplified)"""
        try:
            logger.info(f"Starting {k_folds}-fold cross-validation...")
            
            # Prepare data
            X, y_home, y_away = self._prepare_data_from_cache(use_time_weights=use_time_weights)
            
            if X.empty:
                logger.warning("No data available for cross-validation")
                return {'error': 'No data available'}

            # Remove weight column for training
            weights = X.get('weight', [1.0] * len(X))
            X_features = X.drop(['weight'], axis=1, errors='ignore')

            # Initialize results
            home_scores = []
            away_scores = []
            
            # Perform cross-validation
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_features)):
                logger.debug(f"Processing fold {fold + 1}/{k_folds}")
                
                # Split data
                X_train, X_test = X_features.iloc[train_idx], X_features.iloc[test_idx]
                y_home_train, y_home_test = y_home.iloc[train_idx]['home_goals'], y_home.iloc[test_idx]['home_goals']
                y_away_train, y_away_test = y_away.iloc[train_idx]['away_goals'], y_away.iloc[test_idx]['away_goals']
                
                # Train models
                home_model = OptimizedEnsembleModel()
                away_model = OptimizedEnsembleModel()
                
                home_model.fit(X_train, y_home_train)
                away_model.fit(X_train, y_away_train)
                
                # Make predictions
                home_pred = home_model.predict(X_test)
                away_pred = away_model.predict(X_test)
                
                # Calculate scores
                home_mse = mean_squared_error(y_home_test, home_pred)
                away_mse = mean_squared_error(y_away_test, away_pred)
                
                home_scores.append(home_mse)
                away_scores.append(away_mse)

            # Calculate average scores
            avg_home_mse = sum(home_scores) / len(home_scores)
            avg_away_mse = sum(away_scores) / len(away_scores)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'method': 'cross_validation',
                'k_folds': k_folds,
                'samples': len(X),
                'home_goal_mse': avg_home_mse,
                'away_goal_mse': avg_away_mse,
                'overall_mse': (avg_home_mse + avg_away_mse) / 2,
                'home_scores': home_scores,
                'away_scores': away_scores
            }
            
            # Save results
            self.validation_results[f'cv_{datetime.now().strftime("%Y%m%d_%H%M%S")}'] = results
            self.save_validation_results()
            
            logger.info(f"Cross-validation completed. Overall MSE: {results['overall_mse']:.4f}")
            return results

        except Exception as e:
            logger.error(f"Cross-validation error: {e}")
            return {'error': str(e)}

    def backtesting(self, days_back=90, test_ratio=0.3):
        """Perform backtesting (simplified)"""
        try:
            logger.info(f"Starting backtesting for {days_back} days...")
            
            # Prepare data
            X, y_home, y_away = self._prepare_data_from_cache()
            
            if X.empty:
                logger.warning("No data available for backtesting")
                return {'error': 'No data available'}

            # Time-based split
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            # Simple split by index (last test_ratio of data)
            n_samples = len(X)
            n_test = int(n_samples * test_ratio)
            n_train = n_samples - n_test
            
            X_features = X.drop(['weight'], axis=1, errors='ignore')
            
            X_train = X_features.iloc[:n_train]
            X_test = X_features.iloc[n_train:]
            y_home_train = y_home.iloc[:n_train]['home_goals']
            y_home_test = y_home.iloc[n_train:]['home_goals']
            y_away_train = y_away.iloc[:n_train]['away_goals']
            y_away_test = y_away.iloc[n_train:]['away_goals']
            
            # Train models
            home_model = OptimizedEnsembleModel()
            away_model = OptimizedEnsembleModel()
            
            home_model.fit(X_train, y_home_train)
            away_model.fit(X_train, y_away_train)
            
            # Make predictions
            home_pred = home_model.predict(X_test)
            away_pred = away_model.predict(X_test)
            
            # Calculate metrics
            home_mse = mean_squared_error(y_home_test, home_pred)
            away_mse = mean_squared_error(y_away_test, away_pred)
            
            # Calculate accuracy (within 0.5 goals)
            home_accuracy = accuracy_score(y_home_test, home_pred)
            away_accuracy = accuracy_score(y_away_test, away_pred)
            
            results = {
                'timestamp': datetime.now().isoformat(),
                'method': 'backtesting',
                'days_back': days_back,
                'train_samples': n_train,
                'test_samples': n_test,
                'home_goal_mse': home_mse,
                'away_goal_mse': away_mse,
                'overall_mse': (home_mse + away_mse) / 2,
                'home_accuracy': home_accuracy,
                'away_accuracy': away_accuracy,
                'overall_accuracy': (home_accuracy + away_accuracy) / 2
            }
            
            # Save results
            self.validation_results[f'bt_{datetime.now().strftime("%Y%m%d_%H%M%S")}'] = results
            self.save_validation_results()
            
            logger.info(f"Backtesting completed. Overall MSE: {results['overall_mse']:.4f}, "
                       f"Accuracy: {results['overall_accuracy']:.3f}")
            return results

        except Exception as e:
            logger.error(f"Backtesting error: {e}")
            return {'error': str(e)}

    def generate_validation_report(self):
        """Generate a comprehensive validation report"""
        try:
            if not self.validation_results:
                return {
                    'error': 'No validation results available',
                    'suggestion': 'Run cross_validate() or backtesting() first'
                }

            # Get latest results
            latest_cv = None
            latest_bt = None
            
            for key, result in self.validation_results.items():
                if result.get('method') == 'cross_validation':
                    if latest_cv is None or result.get('timestamp', '') > latest_cv.get('timestamp', ''):
                        latest_cv = result
                elif result.get('method') == 'backtesting':
                    if latest_bt is None or result.get('timestamp', '') > latest_bt.get('timestamp', ''):
                        latest_bt = result

            report = {
                'timestamp': datetime.now().isoformat(),
                'total_validations': len(self.validation_results),
                'latest_cross_validation': latest_cv,
                'latest_backtesting': latest_bt,
                'summary': {},
                'recommendations': []
            }

            # Generate summary
            if latest_cv:
                report['summary']['cv_mse'] = latest_cv.get('overall_mse', 'N/A')
                report['summary']['cv_samples'] = latest_cv.get('samples', 'N/A')

            if latest_bt:
                report['summary']['bt_mse'] = latest_bt.get('overall_mse', 'N/A')
                report['summary']['bt_accuracy'] = latest_bt.get('overall_accuracy', 'N/A')

            # Generate recommendations
            if latest_cv and latest_cv.get('overall_mse', float('inf')) > 0.5:
                report['recommendations'].append(
                    "High cross-validation error detected. Consider feature engineering or model tuning."
                )

            if latest_bt and latest_bt.get('overall_accuracy', 0) < 0.6:
                report['recommendations'].append(
                    "Low backtesting accuracy. Consider more recent data or different model approach."
                )

            if not latest_cv:
                report['recommendations'].append("Consider running cross-validation for better model assessment.")

            if not latest_bt:
                report['recommendations'].append("Consider running backtesting for time-series performance evaluation.")

            logger.info("Validation report generated")
            return report

        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return {'error': str(e)}

    def get_latest_results(self, result_type='all', count=5):
        """Get latest validation results"""
        try:
            if not self.validation_results:
                return []

            # Sort by timestamp
            sorted_results = sorted(
                self.validation_results.items(),
                key=lambda x: x[1].get('timestamp', ''),
                reverse=True
            )

            # Filter by type if specified
            if result_type != 'all':
                sorted_results = [
                    (k, v) for k, v in sorted_results
                    if v.get('method') == result_type
                ]

            # Return latest count results
            return [
                {'key': k, **v} for k, v in sorted_results[:count]
            ]

        except Exception as e:
            logger.error(f"Error getting latest results: {e}")
            return []

# Global instance for backward compatibility
def create_validator(predictor):
    """Create a validator instance"""
    return ModelValidator(predictor)

# Export functions for compatibility
__all__ = [
    'ModelValidator',
    'OptimizedEnsembleModel',
    'numpy_to_python',
    'calculate_time_weight',
    'calculate_form_trend',
    'create_validator'
]