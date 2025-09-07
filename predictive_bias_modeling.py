#!/usr/bin/env python3
"""
Predictive Bias Modeling for LLM Risk Fairness Experiments

This module provides advanced predictive capabilities including:
- Bias prediction for new demographics and scenarios
- Model performance forecasting across different conditions
- Optimal sampling size calculation for bias detection
- Power analysis for statistical significance
- Adaptive experiment design recommendations
- Bias severity risk assessment
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from scipy import stats
import warnings

# Machine learning libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin

# Advanced modeling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Bayesian optimization
try:
    from bayes_opt import BayesianOptimization
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    BAYESIAN_OPT_AVAILABLE = False


@dataclass
class PredictiveModelConfig:
    """Configuration for predictive bias modeling."""
    # Model selection
    model_type: str = "ensemble"  # linear, random_forest, gradient_boosting, xgboost, ensemble
    use_cross_validation: bool = True
    cv_folds: int = 5
    
    # Feature engineering
    include_interaction_terms: bool = True
    polynomial_features: bool = False
    feature_selection: bool = True
    
    # Hyperparameter optimization
    hyperparameter_tuning: bool = True
    optimization_trials: int = 100
    optimization_timeout: int = 300  # seconds
    
    # Validation
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Prediction confidence
    prediction_intervals: bool = True
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


@dataclass
class BiasRiskProfile:
    """Risk profile for bias prediction."""
    demographic_combination: Dict[str, Any]
    predicted_bias_score: float
    confidence_interval: Tuple[float, float]
    risk_level: str  # low, moderate, high, critical
    contributing_factors: List[str]
    mitigation_recommendations: List[str]
    statistical_power: float
    sample_size_needed: int


@dataclass
class ExperimentDesignRecommendation:
    """Recommendations for optimal experiment design."""
    recommended_sample_size: int
    optimal_demographic_distribution: Dict[str, float]
    critical_comparisons: List[Tuple[str, str]]
    expected_effect_sizes: Dict[str, float]
    statistical_power_analysis: Dict[str, float]
    cost_benefit_analysis: Dict[str, float]
    timeline_estimate: str


class BiasPredictor(BaseEstimator, RegressorMixin):
    """Custom bias prediction model with domain-specific features."""
    
    def __init__(self, base_model=None, include_interactions=True):
        self.base_model = base_model or RandomForestRegressor(n_estimators=100, random_state=42)
        self.include_interactions = include_interactions
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
    
    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific features for bias prediction."""
        
        features = X.copy()
        
        # Demographic interaction features
        if self.include_interactions and 'gender' in features.columns and 'age_group' in features.columns:
            # Gender-age interactions
            for gender in features['gender'].unique():
                for age in features['age_group'].unique():
                    feature_name = f'gender_{gender}_age_{age}'
                    features[feature_name] = ((features['gender'] == gender) & 
                                            (features['age_group'] == age)).astype(int)
        
        # Model complexity features
        if 'model_name' in features.columns:
            # Model family indicators
            features['is_gpt'] = features['model_name'].str.contains('gpt', case=False).astype(int)
            features['is_claude'] = features['model_name'].str.contains('claude', case=False).astype(int)
            features['is_gemini'] = features['model_name'].str.contains('gemini', case=False).astype(int)
        
        # Statistical power indicators
        if 'sample_size' in features.columns:
            features['log_sample_size'] = np.log(features['sample_size'] + 1)
            features['sample_size_squared'] = features['sample_size'] ** 2
        
        # Temporal features
        if 'timestamp' in features.columns:
            features['timestamp'] = pd.to_datetime(features['timestamp'])
            features['month'] = features['timestamp'].dt.month
            features['day_of_week'] = features['timestamp'].dt.dayofweek
            features['hour'] = features['timestamp'].dt.hour
        
        return features
    
    def fit(self, X, y):
        """Fit the bias prediction model."""
        
        # Create features
        X_features = self._create_features(X)
        
        # Handle categorical variables
        categorical_columns = X_features.select_dtypes(include=['object']).columns
        X_encoded = X_features.copy()
        
        for col in categorical_columns:
            if col != 'timestamp':  # Skip timestamp
                # One-hot encode categorical variables
                dummies = pd.get_dummies(X_features[col], prefix=col)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded = X_encoded.drop(col, axis=1)
        
        # Remove timestamp if present
        if 'timestamp' in X_encoded.columns:
            X_encoded = X_encoded.drop('timestamp', axis=1)
        
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X_encoded)
        
        # Store feature names
        self.feature_names = X_encoded.columns.tolist()
        
        # Fit base model
        self.base_model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make bias predictions."""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Create features
        X_features = self._create_features(X)
        
        # Handle categorical variables (match training encoding)
        categorical_columns = X_features.select_dtypes(include=['object']).columns
        X_encoded = X_features.copy()
        
        for col in categorical_columns:
            if col != 'timestamp':
                dummies = pd.get_dummies(X_features[col], prefix=col)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded = X_encoded.drop(col, axis=1)
        
        # Remove timestamp if present
        if 'timestamp' in X_encoded.columns:
            X_encoded = X_encoded.drop('timestamp', axis=1)
        
        # Ensure consistent feature set
        missing_features = set(self.feature_names) - set(X_encoded.columns)
        for feature in missing_features:
            X_encoded[feature] = 0
        
        X_encoded = X_encoded[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_encoded)
        
        # Make predictions
        return self.base_model.predict(X_scaled)
    
    def predict_proba_intervals(self, X, confidence_level=0.95):
        """Predict with confidence intervals using bootstrap."""
        
        if not hasattr(self.base_model, 'estimators_'):
            # For models without built-in uncertainty, use simple prediction
            predictions = self.predict(X)
            std_pred = np.std(predictions) if len(predictions) > 1 else 0.1
            
            alpha = 1 - confidence_level
            z_score = stats.norm.ppf(1 - alpha/2)
            margin = z_score * std_pred
            
            lower_bounds = predictions - margin
            upper_bounds = predictions + margin
            
            return predictions, lower_bounds, upper_bounds
        
        # For ensemble models, use individual estimator predictions
        individual_predictions = []
        
        # Process features
        X_features = self._create_features(X)
        categorical_columns = X_features.select_dtypes(include=['object']).columns
        X_encoded = X_features.copy()
        
        for col in categorical_columns:
            if col != 'timestamp':
                dummies = pd.get_dummies(X_features[col], prefix=col)
                X_encoded = pd.concat([X_encoded, dummies], axis=1)
                X_encoded = X_encoded.drop(col, axis=1)
        
        if 'timestamp' in X_encoded.columns:
            X_encoded = X_encoded.drop('timestamp', axis=1)
        
        missing_features = set(self.feature_names) - set(X_encoded.columns)
        for feature in missing_features:
            X_encoded[feature] = 0
        
        X_encoded = X_encoded[self.feature_names]
        X_scaled = self.scaler.transform(X_encoded)
        
        # Get predictions from individual estimators
        for estimator in self.base_model.estimators_:
            pred = estimator.predict(X_scaled)
            individual_predictions.append(pred)
        
        individual_predictions = np.array(individual_predictions)
        
        # Compute statistics
        mean_predictions = np.mean(individual_predictions, axis=0)
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bounds = np.percentile(individual_predictions, lower_percentile, axis=0)
        upper_bounds = np.percentile(individual_predictions, upper_percentile, axis=0)
        
        return mean_predictions, lower_bounds, upper_bounds


class PredictiveBiasModeling:
    """Main class for predictive bias modeling and experiment optimization."""
    
    def __init__(self, config: PredictiveModelConfig = None):
        self.config = config or PredictiveModelConfig()
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.trained_models = {}
        self.feature_importance = {}
        self.training_data = None
        self.scaler = StandardScaler()
        
        # Historical data for pattern learning
        self.historical_experiments = []
        self.bias_patterns = {}
    
    def load_historical_data(self, experiment_directories: List[str]):
        """Load historical experiment data for training predictive models."""
        
        self.logger.info(f"Loading historical data from {len(experiment_directories)} experiments")
        
        all_data = []
        
        for exp_dir in experiment_directories:
            try:
                exp_path = Path(exp_dir)
                
                # Load results
                results_file = exp_path / "results.csv"
                if not results_file.exists():
                    continue
                
                df = pd.read_csv(results_file)
                
                # Load metadata
                metadata = {}
                metadata_file = exp_path / "experiment_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                
                # Load experiment report
                report = {}
                report_file = exp_path / "experiment_report.json"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        report = json.load(f)
                
                # Extract features and outcomes for each demographic group
                for model in df['model'].unique():
                    model_data = df[df['model'] == model]
                    
                    for demographic_cols in [['gender'], ['age_group'], ['gender', 'age_group']]:
                        if all(col in model_data.columns for col in demographic_cols):
                            
                            # Compute bias metrics for each demographic combination
                            if len(demographic_cols) == 1:
                                groups = model_data.groupby(demographic_cols[0])
                            else:
                                groups = model_data.groupby(demographic_cols)
                            
                            group_stats = []
                            for group_name, group_data in groups:
                                if len(group_data) >= 5:  # Minimum sample size
                                    group_accuracy = group_data['risk_band_correct'].mean() if 'risk_band_correct' in group_data.columns else 0.5
                                    group_stats.append((group_name, group_accuracy, len(group_data)))
                            
                            if len(group_stats) >= 2:
                                # Compute demographic parity violation
                                accuracies = [stat[1] for stat in group_stats]
                                bias_score = max(accuracies) - min(accuracies)
                                
                                # Create feature vector
                                feature_data = {
                                    'model_name': model,
                                    'sample_size': len(model_data),
                                    'num_demographic_groups': len(group_stats),
                                    'experiment_id': exp_path.name,
                                    'bias_score': bias_score,  # Target variable
                                    'timestamp': report.get('timestamp', datetime.now().isoformat())
                                }
                                
                                # Add demographic features
                                if len(demographic_cols) == 1:
                                    feature_data[f'primary_demographic'] = demographic_cols[0]
                                else:
                                    feature_data['primary_demographic'] = '_'.join(demographic_cols)
                                
                                # Add model characteristics
                                if 'gpt' in model.lower():
                                    feature_data['model_family'] = 'gpt'
                                elif 'claude' in model.lower():
                                    feature_data['model_family'] = 'claude'
                                elif 'gemini' in model.lower():
                                    feature_data['model_family'] = 'gemini'
                                else:
                                    feature_data['model_family'] = 'other'
                                
                                # Add experimental conditions
                                feature_data['total_cost'] = report.get('total_cost', 0)
                                feature_data['duration_hours'] = report.get('duration_hours', 1)
                                feature_data['success_rate'] = 1 - model_data['error'].notna().mean() if 'error' in model_data.columns else 1.0
                                
                                all_data.append(feature_data)
                
            except Exception as e:
                self.logger.warning(f"Failed to load data from {exp_dir}: {str(e)}")
                continue
        
        if all_data:
            self.training_data = pd.DataFrame(all_data)
            self.logger.info(f"Loaded {len(self.training_data)} training examples")
        else:
            self.logger.warning("No valid training data found")
            self.training_data = pd.DataFrame()
    
    def train_bias_prediction_model(self) -> Dict[str, float]:
        """Train predictive models for bias forecasting."""
        
        if self.training_data is None or len(self.training_data) < 10:
            raise ValueError("Insufficient training data. Need at least 10 examples.")
        
        self.logger.info("Training bias prediction models...")
        
        # Prepare features and target
        feature_columns = [col for col in self.training_data.columns 
                          if col not in ['bias_score', 'experiment_id', 'timestamp']]
        
        X = self.training_data[feature_columns].copy()
        y = self.training_data['bias_score'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        models_to_train = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=self.config.random_state),
            'gradient_boosting': GradientBoostingRegressor(random_state=self.config.random_state),
            'custom_bias_predictor': BiasPredictor()
        }
        
        if XGBOOST_AVAILABLE:
            models_to_train['xgboost'] = xgb.XGBRegressor(random_state=self.config.random_state)
        
        results = {}
        
        for model_name, model in models_to_train.items():
            try:
                self.logger.info(f"Training {model_name}...")
                
                # Hyperparameter tuning if enabled
                if self.config.hyperparameter_tuning and model_name != 'linear':
                    model = self._tune_hyperparameters(model, X_train, y_train, model_name)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_pred = model.predict(X_train)
                test_pred = model.predict(X_test)
                
                train_r2 = r2_score(y_train, train_pred)
                test_r2 = r2_score(y_test, test_pred)
                test_mae = mean_absolute_error(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Cross-validation
                if self.config.use_cross_validation:
                    cv_scores = cross_val_score(model, X_train, y_train, 
                                              cv=self.config.cv_folds, scoring='r2')
                    cv_mean = np.mean(cv_scores)
                    cv_std = np.std(cv_scores)
                else:
                    cv_mean = cv_std = 0
                
                # Store model and results
                self.trained_models[model_name] = model
                
                results[model_name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'cv_mean_r2': cv_mean,
                    'cv_std_r2': cv_std
                }
                
                # Feature importance for tree-based models
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    self.feature_importance[model_name] = dict(zip(feature_columns, importance))
                
                self.logger.info(f"{model_name} - Test R²: {test_r2:.3f}, MAE: {test_mae:.3f}")
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_name}: {str(e)}")
                continue
        
        # Select best model
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
            self.logger.info(f"Best model: {best_model_name} (R² = {results[best_model_name]['test_r2']:.3f})")
        
        return results
    
    def _tune_hyperparameters(self, model, X_train, y_train, model_name: str):
        """Tune hyperparameters using grid search or Bayesian optimization."""
        
        if not OPTUNA_AVAILABLE:
            # Fallback to simple grid search
            return self._grid_search_tuning(model, X_train, y_train, model_name)
        
        def objective(trial):
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
                tuned_model = RandomForestRegressor(**params, random_state=self.config.random_state)
                
            elif model_name == 'gradient_boosting':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10)
                }
                tuned_model = GradientBoostingRegressor(**params, random_state=self.config.random_state)
                
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
                }
                tuned_model = xgb.XGBRegressor(**params, random_state=self.config.random_state)
            
            else:
                return float('-inf')  # Skip tuning for unsupported models
            
            # Cross-validation score
            scores = cross_val_score(tuned_model, X_train, y_train, cv=3, scoring='r2')
            return np.mean(scores)
        
        try:
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=self.config.optimization_trials, 
                          timeout=self.config.optimization_timeout)
            
            # Return model with best parameters
            best_params = study.best_params
            
            if model_name == 'random_forest':
                return RandomForestRegressor(**best_params, random_state=self.config.random_state)
            elif model_name == 'gradient_boosting':
                return GradientBoostingRegressor(**best_params, random_state=self.config.random_state)
            elif model_name == 'xgboost' and XGBOOST_AVAILABLE:
                return xgb.XGBRegressor(**best_params, random_state=self.config.random_state)
            
        except Exception as e:
            self.logger.warning(f"Hyperparameter tuning failed for {model_name}: {str(e)}")
            return model  # Return original model
    
    def _grid_search_tuning(self, model, X_train, y_train, model_name: str):
        """Fallback grid search hyperparameter tuning."""
        
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        if model_name in param_grids:
            try:
                grid_search = GridSearchCV(
                    model, param_grids[model_name], 
                    cv=3, scoring='r2', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                return grid_search.best_estimator_
            except Exception as e:
                self.logger.warning(f"Grid search failed for {model_name}: {str(e)}")
        
        return model  # Return original model if tuning fails
    
    def predict_bias_for_demographics(self, demographic_combinations: List[Dict[str, Any]], 
                                    model_name: str = "gpt-4o", 
                                    sample_size: int = 100) -> List[BiasRiskProfile]:
        """Predict bias risk for specific demographic combinations."""
        
        if not self.trained_models:
            raise ValueError("No trained models available. Train models first.")
        
        # Use best available model
        best_model_name = max(self.trained_models.keys(), 
                             key=lambda x: getattr(self.trained_models[x], 'score', lambda: 0)())
        model = self.trained_models[best_model_name]
        
        predictions = []
        
        for demo_combo in demographic_combinations:
            try:
                # Create feature vector
                feature_data = {
                    'model_name': model_name,
                    'sample_size': sample_size,
                    'num_demographic_groups': len(demo_combo),
                    'total_cost': 0,  # Will be estimated
                    'duration_hours': 1,
                    'success_rate': 0.95,
                    'primary_demographic': '_'.join(demo_combo.keys()),
                }
                
                # Add model family
                if 'gpt' in model_name.lower():
                    feature_data['model_family'] = 'gpt'
                elif 'claude' in model_name.lower():
                    feature_data['model_family'] = 'claude'
                elif 'gemini' in model_name.lower():
                    feature_data['model_family'] = 'gemini'
                else:
                    feature_data['model_family'] = 'other'
                
                # Create DataFrame for prediction
                feature_df = pd.DataFrame([feature_data])
                
                # Make prediction
                if hasattr(model, 'predict_proba_intervals'):
                    pred_mean, pred_lower, pred_upper = model.predict_proba_intervals(
                        feature_df, confidence_level=self.config.confidence_level
                    )
                    predicted_bias = pred_mean[0]
                    confidence_interval = (pred_lower[0], pred_upper[0])
                else:
                    predicted_bias = model.predict(feature_df)[0]
                    # Estimate confidence interval
                    std_error = 0.05  # Conservative estimate
                    z_score = stats.norm.ppf(1 - (1 - self.config.confidence_level) / 2)
                    margin = z_score * std_error
                    confidence_interval = (predicted_bias - margin, predicted_bias + margin)
                
                # Determine risk level
                if predicted_bias < 0.05:
                    risk_level = "low"
                elif predicted_bias < 0.1:
                    risk_level = "moderate"
                elif predicted_bias < 0.2:
                    risk_level = "high"
                else:
                    risk_level = "critical"
                
                # Generate contributing factors based on feature importance
                contributing_factors = self._identify_contributing_factors(
                    feature_data, best_model_name, predicted_bias
                )
                
                # Generate mitigation recommendations
                mitigation_recommendations = self._generate_mitigation_recommendations(
                    demo_combo, predicted_bias, risk_level
                )
                
                # Estimate required sample size for reliable detection
                sample_size_needed = self._estimate_required_sample_size(predicted_bias)
                
                # Estimate statistical power
                statistical_power = self._estimate_statistical_power(predicted_bias, sample_size)
                
                # Create risk profile
                risk_profile = BiasRiskProfile(
                    demographic_combination=demo_combo,
                    predicted_bias_score=predicted_bias,
                    confidence_interval=confidence_interval,
                    risk_level=risk_level,
                    contributing_factors=contributing_factors,
                    mitigation_recommendations=mitigation_recommendations,
                    statistical_power=statistical_power,
                    sample_size_needed=sample_size_needed
                )
                
                predictions.append(risk_profile)
                
            except Exception as e:
                self.logger.error(f"Failed to predict bias for {demo_combo}: {str(e)}")
                continue
        
        return predictions
    
    def _identify_contributing_factors(self, feature_data: Dict, model_name: str, 
                                     predicted_bias: float) -> List[str]:
        """Identify factors contributing to predicted bias."""
        
        factors = []
        
        # Check feature importance if available
        if model_name in self.feature_importance:
            importance = self.feature_importance[model_name]
            
            # Top contributing features
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:3]
            
            for feature, importance_score in top_features:
                if importance_score > 0.1:  # Significant contribution
                    factors.append(f"{feature} (importance: {importance_score:.2f})")
        
        # Model-specific factors
        model_family = feature_data.get('model_family', 'unknown')
        if model_family == 'gpt' and predicted_bias > 0.1:
            factors.append("GPT models show higher bias sensitivity")
        elif model_family == 'claude' and predicted_bias > 0.05:
            factors.append("Claude models demonstrate moderate bias patterns")
        
        # Sample size factors
        sample_size = feature_data.get('sample_size', 0)
        if sample_size < 50:
            factors.append("Small sample size increases uncertainty")
        elif sample_size < 100:
            factors.append("Moderate sample size may limit bias detection power")
        
        return factors if factors else ["Insufficient data for factor analysis"]
    
    def _generate_mitigation_recommendations(self, demo_combo: Dict, 
                                           predicted_bias: float, 
                                           risk_level: str) -> List[str]:
        """Generate bias mitigation recommendations."""
        
        recommendations = []
        
        if risk_level in ["high", "critical"]:
            recommendations.extend([
                "Implement adversarial debiasing during model training",
                "Apply post-processing fairness constraints",
                "Increase sample size for affected demographic groups",
                "Consider demographic-aware prompting strategies"
            ])
        
        if risk_level in ["moderate", "high", "critical"]:
            recommendations.extend([
                "Monitor model outputs continuously for bias drift",
                "Implement threshold adjustment based on demographic group",
                "Conduct regular bias audits with expanded test cases"
            ])
        
        # Demographic-specific recommendations
        if 'gender' in demo_combo:
            recommendations.append("Ensure balanced gender representation in training examples")
        
        if 'age' in demo_combo or 'age_group' in demo_combo:
            recommendations.append("Validate age-related assumptions in risk assessment logic")
        
        if predicted_bias > 0.15:
            recommendations.append("Consider alternative models with lower bias propensity")
        
        return recommendations
    
    def _estimate_required_sample_size(self, expected_effect_size: float, 
                                     alpha: float = 0.05, power: float = 0.8) -> int:
        """Estimate required sample size for bias detection."""
        
        # Cohen's guidelines for effect sizes
        if expected_effect_size < 0.02:
            effect_size_category = "very_small"
        elif expected_effect_size < 0.05:
            effect_size_category = "small"
        elif expected_effect_size < 0.1:
            effect_size_category = "medium"
        else:
            effect_size_category = "large"
        
        # Conservative sample size estimates
        sample_size_table = {
            "very_small": 1000,
            "small": 500,
            "medium": 200,
            "large": 100
        }
        
        base_sample_size = sample_size_table.get(effect_size_category, 200)
        
        # Adjust for multiple comparisons (Bonferroni correction)
        num_comparisons = 3  # Typical number of demographic comparisons
        adjusted_alpha = alpha / num_comparisons
        
        # Simple adjustment factor
        adjustment_factor = np.log(alpha / adjusted_alpha)
        adjusted_sample_size = int(base_sample_size * adjustment_factor)
        
        return max(50, adjusted_sample_size)  # Minimum 50 samples
    
    def _estimate_statistical_power(self, effect_size: float, sample_size: int, 
                                   alpha: float = 0.05) -> float:
        """Estimate statistical power for bias detection."""
        
        # Simplified power calculation for two-sample t-test
        # This is an approximation - more sophisticated power analysis would require
        # specific assumptions about variance and distribution
        
        # Effect size (Cohen's d approximation)
        cohens_d = effect_size / 0.1  # Normalize by typical standard deviation
        
        # Critical t-value
        df = 2 * sample_size - 2  # Degrees of freedom for two-sample t-test
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Non-centrality parameter
        ncp = cohens_d * np.sqrt(sample_size / 2)
        
        # Power calculation
        power = 1 - stats.t.cdf(t_critical, df, loc=ncp)
        
        return min(max(power, 0.0), 1.0)  # Clamp between 0 and 1
    
    def recommend_experiment_design(self, target_demographics: List[str], 
                                  available_budget: float = 1000.0,
                                  required_power: float = 0.8) -> ExperimentDesignRecommendation:
        """Recommend optimal experiment design for bias detection."""
        
        # Predict bias for common demographic combinations
        demographic_combinations = []
        
        # Generate all possible combinations of target demographics
        if 'gender' in target_demographics:
            if 'age' in target_demographics:
                demographic_combinations.extend([
                    {'gender': 'Male', 'age_group': 'Young'},
                    {'gender': 'Female', 'age_group': 'Young'},
                    {'gender': 'Male', 'age_group': 'Senior'},
                    {'gender': 'Female', 'age_group': 'Senior'}
                ])
            else:
                demographic_combinations.extend([
                    {'gender': 'Male'},
                    {'gender': 'Female'}
                ])
        
        # Predict bias for these combinations
        bias_predictions = self.predict_bias_for_demographics(demographic_combinations)
        
        # Find combination with highest predicted bias
        highest_risk_combo = max(bias_predictions, key=lambda x: x.predicted_bias_score)
        critical_effect_size = highest_risk_combo.predicted_bias_score
        
        # Calculate required sample size for reliable detection
        required_sample_size = self._estimate_required_sample_size(
            critical_effect_size, power=required_power
        )
        
        # Optimize demographic distribution
        optimal_distribution = {}
        total_budget_per_group = available_budget / len(demographic_combinations)
        estimated_cost_per_sample = 0.05  # Estimated cost per API call
        
        for combo in demographic_combinations:
            combo_key = '_'.join(f"{k}:{v}" for k, v in combo.items())
            max_samples = int(total_budget_per_group / estimated_cost_per_sample)
            optimal_distribution[combo_key] = min(required_sample_size, max_samples)
        
        # Identify critical comparisons
        critical_comparisons = []
        for i, combo1 in enumerate(demographic_combinations):
            for combo2 in demographic_combinations[i+1:]:
                # Find predicted bias difference
                pred1 = next(p for p in bias_predictions if p.demographic_combination == combo1)
                pred2 = next(p for p in bias_predictions if p.demographic_combination == combo2)
                
                bias_diff = abs(pred1.predicted_bias_score - pred2.predicted_bias_score)
                if bias_diff > 0.02:  # Significant difference threshold
                    combo1_key = '_'.join(f"{k}:{v}" for k, v in combo1.items())
                    combo2_key = '_'.join(f"{k}:{v}" for k, v in combo2.items())
                    critical_comparisons.append((combo1_key, combo2_key))
        
        # Expected effect sizes
        expected_effect_sizes = {}
        for prediction in bias_predictions:
            combo_key = '_'.join(f"{k}:{v}" for k, v in prediction.demographic_combination.items())
            expected_effect_sizes[combo_key] = prediction.predicted_bias_score
        
        # Power analysis
        statistical_power_analysis = {}
        for prediction in bias_predictions:
            combo_key = '_'.join(f"{k}:{v}" for k, v in prediction.demographic_combination.items())
            statistical_power_analysis[combo_key] = prediction.statistical_power
        
        # Cost-benefit analysis
        total_estimated_cost = sum(optimal_distribution.values()) * estimated_cost_per_sample
        expected_bias_detection_value = sum(expected_effect_sizes.values()) * 1000  # Arbitrary value unit
        
        cost_benefit_analysis = {
            'estimated_total_cost': total_estimated_cost,
            'expected_detection_value': expected_bias_detection_value,
            'cost_effectiveness_ratio': expected_bias_detection_value / max(total_estimated_cost, 1)
        }
        
        # Timeline estimate
        estimated_duration_hours = (sum(optimal_distribution.values()) * 0.1) / 60  # Assume 0.1 minutes per sample
        timeline_estimate = f"{estimated_duration_hours:.1f} hours"
        
        return ExperimentDesignRecommendation(
            recommended_sample_size=required_sample_size,
            optimal_demographic_distribution=optimal_distribution,
            critical_comparisons=critical_comparisons,
            expected_effect_sizes=expected_effect_sizes,
            statistical_power_analysis=statistical_power_analysis,
            cost_benefit_analysis=cost_benefit_analysis,
            timeline_estimate=timeline_estimate
        )
    
    def generate_bias_forecast_report(self, output_path: str) -> str:
        """Generate comprehensive bias forecasting report."""
        
        report_lines = [
            "# Predictive Bias Modeling Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Model Performance Summary",
            ""
        ]
        
        if self.trained_models:
            for model_name in self.trained_models:
                # Add model performance details
                report_lines.append(f"### {model_name.title()} Model")
                report_lines.append(f"- Training completed successfully")
                
                if model_name in self.feature_importance:
                    importance = self.feature_importance[model_name]
                    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    report_lines.append("- Top predictive features:")
                    for feature, importance_score in top_features:
                        report_lines.append(f"  - {feature}: {importance_score:.3f}")
                
                report_lines.append("")
        else:
            report_lines.append("No trained models available.")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        
        return output_path


def main():
    """CLI interface for predictive bias modeling."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Predictive Bias Modeling for LLM Experiments")
    parser.add_argument('command', choices=['train', 'predict', 'recommend', 'report'])
    parser.add_argument('--data-dirs', nargs='+', help='Historical experiment directories')
    parser.add_argument('--output-dir', default='predictive_modeling_output', help='Output directory')
    parser.add_argument('--demographics', nargs='+', default=['gender', 'age'], 
                       help='Demographics to analyze')
    parser.add_argument('--model-name', default='gpt-4o', help='Model to analyze')
    parser.add_argument('--budget', type=float, default=1000.0, help='Available budget')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize predictive modeling
    modeler = PredictiveBiasModeling()
    
    if args.command == 'train' and args.data_dirs:
        # Load data and train models
        modeler.load_historical_data(args.data_dirs)
        results = modeler.train_bias_prediction_model()
        
        print("Model Training Results:")
        for model_name, metrics in results.items():
            print(f"{model_name}: R² = {metrics['test_r2']:.3f}, MAE = {metrics['test_mae']:.3f}")
    
    elif args.command == 'predict':
        if args.data_dirs:
            modeler.load_historical_data(args.data_dirs)
            modeler.train_bias_prediction_model()
        
        # Generate predictions
        demo_combinations = [
            {'gender': 'Male', 'age_group': 'Young'},
            {'gender': 'Female', 'age_group': 'Young'},
            {'gender': 'Male', 'age_group': 'Senior'},
            {'gender': 'Female', 'age_group': 'Senior'}
        ]
        
        predictions = modeler.predict_bias_for_demographics(demo_combinations, args.model_name)
        
        print("Bias Risk Predictions:")
        for pred in predictions:
            print(f"{pred.demographic_combination}: {pred.predicted_bias_score:.3f} ({pred.risk_level})")
    
    elif args.command == 'recommend':
        if args.data_dirs:
            modeler.load_historical_data(args.data_dirs)
            modeler.train_bias_prediction_model()
        
        recommendation = modeler.recommend_experiment_design(
            args.demographics, args.budget
        )
        
        print("Experiment Design Recommendations:")
        print(f"Recommended sample size: {recommendation.recommended_sample_size}")
        print(f"Timeline estimate: {recommendation.timeline_estimate}")
        print("Optimal distribution:", recommendation.optimal_demographic_distribution)
    
    elif args.command == 'report':
        report_path = Path(args.output_dir) / 'bias_forecast_report.md'
        modeler.generate_bias_forecast_report(str(report_path))
        print(f"Report generated: {report_path}")


if __name__ == "__main__":
    main()