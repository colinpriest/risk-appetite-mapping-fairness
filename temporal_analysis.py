#!/usr/bin/env python3
"""
Temporal Bias Analysis Module

Track how model bias changes over time, across model versions,
and in response to different conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path

# Statistical analysis
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Change point detection
try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False

@dataclass
class TemporalTrend:
    """Container for temporal trend analysis."""
    slope: float
    p_value: float
    r_squared: float
    confidence_interval: Tuple[float, float]
    trend_direction: str  # "increasing", "decreasing", "stable"
    changepoints: Optional[List[int]] = None
    seasonal_component: Optional[np.ndarray] = None

@dataclass
class ModelVersionComparison:
    """Compare bias across model versions."""
    version_metrics: Dict[str, Dict[str, float]]
    significant_changes: List[Dict[str, Any]]
    regression_discontinuity: Optional[Dict[str, float]] = None
    version_timeline: Optional[List[Tuple[str, datetime]]] = None

@dataclass
class BiasEvolutionResults:
    """Complete temporal bias evolution analysis."""
    overall_trend: TemporalTrend
    by_group_trends: Dict[str, TemporalTrend]
    model_comparisons: Optional[ModelVersionComparison] = None
    stability_metrics: Optional[Dict[str, float]] = None
    drift_detection: Optional[Dict[str, Any]] = None

class TemporalBiasAnalyzer:
    """Analyze how bias evolves over time."""
    
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.scaler = StandardScaler()
    
    def analyze_temporal_evolution(self, 
                                 df: pd.DataFrame,
                                 time_col: str,
                                 bias_metric_col: str,
                                 group_col: Optional[str] = None,
                                 model_version_col: Optional[str] = None) -> BiasEvolutionResults:
        """Comprehensive temporal bias analysis."""
        
        # Convert time column to datetime if needed
        if df[time_col].dtype == 'object':
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Sort by time
        df_sorted = df.sort_values(time_col).copy()
        df_sorted['time_numeric'] = (df_sorted[time_col] - df_sorted[time_col].min()).dt.days
        
        # Overall temporal trend
        overall_trend = self._analyze_trend(
            df_sorted['time_numeric'].values,
            df_sorted[bias_metric_col].values
        )
        
        # By-group trends
        by_group_trends = {}
        if group_col:
            for group in df_sorted[group_col].unique():
                group_data = df_sorted[df_sorted[group_col] == group]
                if len(group_data) >= 3:  # Minimum points for trend analysis
                    by_group_trends[group] = self._analyze_trend(
                        group_data['time_numeric'].values,
                        group_data[bias_metric_col].values
                    )
        
        # Model version comparisons
        model_comparisons = None
        if model_version_col:
            model_comparisons = self._compare_model_versions(
                df_sorted, time_col, bias_metric_col, model_version_col, group_col
            )
        
        # Stability metrics
        stability_metrics = self._calculate_stability_metrics(
            df_sorted[bias_metric_col].values
        )
        
        # Drift detection
        drift_detection = self._detect_drift(
            df_sorted['time_numeric'].values,
            df_sorted[bias_metric_col].values
        )
        
        return BiasEvolutionResults(
            overall_trend=overall_trend,
            by_group_trends=by_group_trends,
            model_comparisons=model_comparisons,
            stability_metrics=stability_metrics,
            drift_detection=drift_detection
        )
    
    def _analyze_trend(self, time_values: np.ndarray, metric_values: np.ndarray) -> TemporalTrend:
        """Analyze temporal trend using linear regression."""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(time_values) | np.isnan(metric_values))
        time_clean = time_values[valid_mask]
        metric_clean = metric_values[valid_mask]
        
        if len(time_clean) < 3:
            return TemporalTrend(
                slope=0,
                p_value=1.0,
                r_squared=0,
                confidence_interval=(0, 0),
                trend_direction="insufficient_data"
            )
        
        # Linear regression
        X = time_clean.reshape(-1, 1)
        y = metric_clean
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        slope = reg.coef_[0]
        r_squared = r2_score(y, reg.predict(X))
        
        # Statistical significance test
        n = len(time_clean)
        
        # Calculate standard error of slope
        y_pred = reg.predict(X)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        se_slope = np.sqrt(mse / np.sum((time_clean - np.mean(time_clean))**2))
        
        # t-test for slope significance
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        
        # Confidence interval for slope
        t_critical = stats.t.ppf(1 - self.significance_level/2, n - 2)
        ci_lower = slope - t_critical * se_slope
        ci_upper = slope + t_critical * se_slope
        
        # Determine trend direction
        if p_value < self.significance_level:
            if slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # Change point detection
        changepoints = None
        if RUPTURES_AVAILABLE and len(metric_clean) >= 10:
            try:
                algo = rpt.Pelt(model="rbf").fit(metric_clean)
                changepoints = algo.predict(pen=10)
            except:
                pass
        
        return TemporalTrend(
            slope=slope,
            p_value=p_value,
            r_squared=r_squared,
            confidence_interval=(ci_lower, ci_upper),
            trend_direction=trend_direction,
            changepoints=changepoints
        )
    
    def _compare_model_versions(self, df: pd.DataFrame, time_col: str, 
                               metric_col: str, version_col: str,
                               group_col: Optional[str] = None) -> ModelVersionComparison:
        """Compare bias across different model versions."""
        
        version_metrics = {}
        versions = sorted(df[version_col].unique())
        
        # Calculate metrics for each version
        for version in versions:
            version_data = df[df[version_col] == version]
            
            metrics = {
                "mean_bias": version_data[metric_col].mean(),
                "std_bias": version_data[metric_col].std(),
                "median_bias": version_data[metric_col].median(),
                "n_observations": len(version_data)
            }
            
            # By-group metrics if specified
            if group_col:
                group_metrics = {}
                for group in version_data[group_col].unique():
                    group_data = version_data[version_data[group_col] == group]
                    if len(group_data) > 0:
                        group_metrics[group] = {
                            "mean_bias": group_data[metric_col].mean(),
                            "n": len(group_data)
                        }
                metrics["by_group"] = group_metrics
            
            version_metrics[version] = metrics
        
        # Detect significant changes between versions
        significant_changes = []
        
        for i in range(len(versions) - 1):
            v1, v2 = versions[i], versions[i + 1]
            v1_data = df[df[version_col] == v1][metric_col].values
            v2_data = df[df[version_col] == v2][metric_col].values
            
            if len(v1_data) > 0 and len(v2_data) > 0:
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(v1_data, v2_data)
                
                if p_value < self.significance_level:
                    effect_size = (np.mean(v2_data) - np.mean(v1_data)) / np.sqrt(
                        (np.var(v1_data) + np.var(v2_data)) / 2
                    )
                    
                    significant_changes.append({
                        "from_version": v1,
                        "to_version": v2,
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "mean_change": np.mean(v2_data) - np.mean(v1_data),
                        "change_direction": "increase" if effect_size > 0 else "decrease"
                    })
        
        # Version timeline
        version_timeline = []
        for version in versions:
            version_data = df[df[version_col] == version]
            if len(version_data) > 0:
                first_occurrence = version_data[time_col].min()
                version_timeline.append((version, first_occurrence))
        
        return ModelVersionComparison(
            version_metrics=version_metrics,
            significant_changes=significant_changes,
            version_timeline=version_timeline
        )
    
    def _calculate_stability_metrics(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate various stability/volatility metrics."""
        
        values_clean = values[~np.isnan(values)]
        
        if len(values_clean) < 2:
            return {"error": "insufficient_data"}
        
        # Basic volatility measures
        volatility = np.std(values_clean)
        coefficient_variation = volatility / abs(np.mean(values_clean)) if np.mean(values_clean) != 0 else np.inf
        
        # Rolling volatility (if enough points)
        if len(values_clean) >= 10:
            window = min(10, len(values_clean) // 3)
            rolling_std = pd.Series(values_clean).rolling(window=window).std()
            avg_rolling_volatility = rolling_std.mean()
            volatility_of_volatility = rolling_std.std()
        else:
            avg_rolling_volatility = volatility
            volatility_of_volatility = 0
        
        # Stability ratio (inverse of coefficient of variation)
        stability_ratio = 1 / coefficient_variation if coefficient_variation != 0 and np.isfinite(coefficient_variation) else 0
        
        return {
            "volatility": volatility,
            "coefficient_variation": coefficient_variation,
            "stability_ratio": stability_ratio,
            "avg_rolling_volatility": avg_rolling_volatility,
            "volatility_of_volatility": volatility_of_volatility,
            "range": np.max(values_clean) - np.min(values_clean),
            "interquartile_range": np.percentile(values_clean, 75) - np.percentile(values_clean, 25)
        }
    
    def _detect_drift(self, time_values: np.ndarray, metric_values: np.ndarray) -> Dict[str, Any]:
        """Detect concept drift in bias metrics."""
        
        # Remove NaN values
        valid_mask = ~(np.isnan(time_values) | np.isnan(metric_values))
        time_clean = time_values[valid_mask]
        metric_clean = metric_values[valid_mask]
        
        if len(time_clean) < 10:
            return {"error": "insufficient_data"}
        
        drift_results = {}
        
        # CUSUM test for drift detection
        mean_baseline = np.mean(metric_clean[:len(metric_clean)//3])  # First third as baseline
        std_baseline = np.std(metric_clean[:len(metric_clean)//3])
        
        if std_baseline > 0:
            cusum_pos = np.zeros(len(metric_clean))
            cusum_neg = np.zeros(len(metric_clean))
            
            threshold = 3 * std_baseline  # 3-sigma threshold
            
            for i in range(1, len(metric_clean)):
                cusum_pos[i] = max(0, cusum_pos[i-1] + metric_clean[i] - mean_baseline - threshold/2)
                cusum_neg[i] = max(0, cusum_neg[i-1] - metric_clean[i] + mean_baseline - threshold/2)
            
            # Detect drift points
            drift_points = []
            if np.max(cusum_pos) > threshold:
                drift_points.append({
                    "type": "upward_drift",
                    "index": np.argmax(cusum_pos > threshold),
                    "magnitude": np.max(cusum_pos)
                })
            
            if np.max(cusum_neg) > threshold:
                drift_points.append({
                    "type": "downward_drift", 
                    "index": np.argmax(cusum_neg > threshold),
                    "magnitude": np.max(cusum_neg)
                })
            
            drift_results["cusum_drift_detected"] = len(drift_points) > 0
            drift_results["drift_points"] = drift_points
        
        # Page-Hinkley test for drift
        try:
            # Simple Page-Hinkley implementation
            ph_stat = 0
            ph_threshold = 10
            ph_drift_point = None
            
            for i in range(1, len(metric_clean)):
                ph_stat = max(0, ph_stat + metric_clean[i] - mean_baseline - std_baseline)
                if ph_stat > ph_threshold and ph_drift_point is None:
                    ph_drift_point = i
            
            drift_results["page_hinkley_drift"] = ph_drift_point is not None
            drift_results["page_hinkley_point"] = ph_drift_point
        except:
            pass
        
        return drift_results

class ModelVersionTracker:
    """Track and manage different model versions for comparison."""
    
    def __init__(self, storage_path: str = "model_versions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.version_registry = self._load_registry()
    
    def register_model_version(self, version_id: str, metadata: Dict[str, Any]):
        """Register a new model version with metadata."""
        self.version_registry[version_id] = {
            "registration_time": datetime.now().isoformat(),
            "metadata": metadata
        }
        self._save_registry()
    
    def log_experiment_result(self, version_id: str, experiment_data: Dict[str, Any]):
        """Log experimental results for a specific model version."""
        result_file = self.storage_path / f"{version_id}_results.jsonl"
        
        # Add timestamp
        experiment_data["timestamp"] = datetime.now().isoformat()
        experiment_data["version_id"] = version_id
        
        # Append to results file
        with open(result_file, 'a') as f:
            f.write(json.dumps(experiment_data) + '\n')
    
    def load_version_results(self, version_id: str) -> pd.DataFrame:
        """Load all results for a specific model version."""
        result_file = self.storage_path / f"{version_id}_results.jsonl"
        
        if not result_file.exists():
            return pd.DataFrame()
        
        results = []
        with open(result_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        return pd.DataFrame(results)
    
    def compare_versions(self, version_ids: List[str], metric_col: str) -> Dict[str, Any]:
        """Compare multiple model versions on a specific metric."""
        version_data = {}
        
        for version_id in version_ids:
            df = self.load_version_results(version_id)
            if not df.empty and metric_col in df.columns:
                version_data[version_id] = df[metric_col].values
        
        if len(version_data) < 2:
            return {"error": "insufficient_versions"}
        
        # Statistical comparisons
        comparisons = {}
        from itertools import combinations
        
        for v1, v2 in combinations(version_ids, 2):
            if v1 in version_data and v2 in version_data:
                data1, data2 = version_data[v1], version_data[v2]
                
                # Two-sample t-test
                t_stat, p_value = stats.ttest_ind(data1, data2)
                
                # Effect size (Cohen's d)
                pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
                cohens_d = (np.mean(data2) - np.mean(data1)) / pooled_std if pooled_std > 0 else 0
                
                comparisons[f"{v1}_vs_{v2}"] = {
                    "mean_diff": np.mean(data2) - np.mean(data1),
                    "t_statistic": t_stat,
                    "p_value": p_value,
                    "cohens_d": cohens_d,
                    "significant": p_value < 0.05
                }
        
        return {
            "version_summaries": {
                vid: {
                    "mean": np.mean(data),
                    "std": np.std(data),
                    "n": len(data)
                } for vid, data in version_data.items()
            },
            "pairwise_comparisons": comparisons
        }
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load the model version registry."""
        registry_file = self.storage_path / "version_registry.json"
        
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _save_registry(self):
        """Save the model version registry."""
        registry_file = self.storage_path / "version_registry.json"
        
        with open(registry_file, 'w') as f:
            json.dump(self.version_registry, f, indent=2)

# Utility functions
def detect_seasonal_patterns(time_values: np.ndarray, metric_values: np.ndarray, 
                           period: Optional[int] = None) -> Dict[str, Any]:
    """Detect seasonal patterns in bias metrics."""
    
    if len(time_values) < 12:  # Need minimum data for seasonality
        return {"error": "insufficient_data"}
    
    # Convert to pandas for easier time series analysis
    df = pd.DataFrame({
        'time': pd.to_datetime(time_values),
        'metric': metric_values
    }).sort_values('time')
    
    # Remove NaN values
    df = df.dropna()
    
    if len(df) < 12:
        return {"error": "insufficient_clean_data"}
    
    # Simple seasonal decomposition
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Set frequency based on data
        if period is None:
            # Try to infer period from data frequency
            time_diff = df['time'].diff().median()
            if time_diff.days <= 1:
                period = 7  # Weekly pattern
            elif time_diff.days <= 7:
                period = 4  # Monthly pattern
            else:
                period = 12  # Yearly pattern
        
        # Ensure we have enough periods
        if len(df) < 2 * period:
            period = len(df) // 2
        
        if period >= 2:
            decomposition = seasonal_decompose(
                df.set_index('time')['metric'], 
                model='additive', 
                period=period,
                extrapolate_trend='freq'
            )
            
            return {
                "seasonal_component": decomposition.seasonal.values,
                "trend_component": decomposition.trend.values,
                "residual_component": decomposition.resid.values,
                "seasonal_strength": np.var(decomposition.seasonal) / np.var(df['metric']),
                "period_detected": period
            }
    except Exception as e:
        return {"error": f"decomposition_failed: {e}"}
    
    return {"error": "analysis_failed"}

# Export all classes and functions
__all__ = [
    'TemporalTrend',
    'ModelVersionComparison', 
    'BiasEvolutionResults',
    'TemporalBiasAnalyzer',
    'ModelVersionTracker',
    'detect_seasonal_patterns'
]