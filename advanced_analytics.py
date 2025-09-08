#!/usr/bin/env python3
"""
Advanced Analytics Module for LLM Risk Fairness Experiments

This module provides cutting-edge statistical analysis capabilities including:
- Bayesian uncertainty quantification
- Bootstrap confidence intervals
- Advanced fairness metrics
- Model calibration analysis
- Individual fairness assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import warnings
from datetime import datetime
import json

# Statistical libraries
from scipy import stats
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
import pingouin as pg

# Bootstrap and resampling
try:
    from scikits.bootstrap import bootstrap
except ImportError:
    # Fallback bootstrap implementation
    def bootstrap(data, statfunc, n_samples=1000, alpha=0.05):
        """Simple bootstrap implementation."""
        n = len(data)
        boot_samples = []
        for _ in range(n_samples):
            sample = np.random.choice(data, size=n, replace=True)
            boot_samples.append(statfunc(sample))
        
        boot_samples = np.array(boot_samples)
        lower = np.percentile(boot_samples, 100 * alpha/2)
        upper = np.percentile(boot_samples, 100 * (1 - alpha/2))
        return boot_samples, (lower, upper)

# Advanced fairness libraries
try:
    import fairlearn.metrics as fl_metrics
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    warnings.warn("Fairlearn not available - some advanced fairness metrics disabled")

try:
    from aif360.datasets import StandardDataset
    from aif360.metrics import ClassificationMetric
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    warnings.warn("AIF360 not available - some fairness metrics disabled")

# Bayesian analysis
try:
    import pymc as pm
    import arviz as az
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False
    warnings.warn("PyMC/ArviZ not available - Bayesian analysis disabled")

@dataclass
class UncertaintyResults:
    """Container for uncertainty quantification results."""
    point_estimate: float
    confidence_interval: Tuple[float, float]
    credible_interval: Optional[Tuple[float, float]] = None
    bootstrap_samples: Optional[np.ndarray] = None
    bayesian_samples: Optional[np.ndarray] = None
    method: str = "bootstrap"
    confidence_level: float = 0.95

@dataclass 
class CalibrationResults:
    """Container for model calibration analysis."""
    brier_score: float
    reliability_score: float
    resolution_score: float
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    calibration_curve: Tuple[np.ndarray, np.ndarray]
    isotonic_calibrator: Any = None
    platt_calibrator: Any = None

@dataclass
class IndividualFairnessResults:
    """Container for individual fairness metrics."""
    lipschitz_constant: float
    consistency_score: float
    individual_disparities: np.ndarray
    similarity_matrix: Optional[np.ndarray] = None
    method: str = "euclidean"

class UncertaintyQuantifier:
    """Advanced uncertainty quantification for fairness metrics."""
    
    def __init__(self, n_bootstrap=1000, confidence_level=0.95, random_state=42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state = random_state
        np.random.seed(random_state)
    
    def bootstrap_metric(self, data: np.ndarray, metric_func, **kwargs) -> UncertaintyResults:
        """Calculate bootstrap confidence intervals for any metric."""
        
        def statfunc(x):
            return metric_func(x, **kwargs)
        
        try:
            boot_samples, ci = bootstrap(data, statfunc, 
                                       n_samples=self.n_bootstrap,
                                       alpha=1-self.confidence_level)
            
            point_est = metric_func(data, **kwargs)
            
            return UncertaintyResults(
                point_estimate=point_est,
                confidence_interval=ci,
                bootstrap_samples=boot_samples,
                method="bootstrap",
                confidence_level=self.confidence_level
            )
        except Exception as e:
            # Fallback to simple percentile method
            boot_samples = []
            n = len(data)
            
            for _ in range(self.n_bootstrap):
                sample = np.random.choice(data, size=n, replace=True)
                boot_samples.append(metric_func(sample, **kwargs))
            
            boot_samples = np.array(boot_samples)
            alpha = 1 - self.confidence_level
            lower = np.percentile(boot_samples, 100 * alpha/2)
            upper = np.percentile(boot_samples, 100 * (1 - alpha/2))
            
            return UncertaintyResults(
                point_estimate=np.mean(boot_samples),
                confidence_interval=(lower, upper),
                bootstrap_samples=boot_samples,
                method="bootstrap_fallback"
            )
    
    def bayesian_metric(self, data: np.ndarray, prior_params: Dict) -> UncertaintyResults:
        """Bayesian analysis of metrics with uncertainty."""
        if not BAYESIAN_AVAILABLE:
            return self.bootstrap_metric(data, np.mean)
        
        try:
            with pm.Model() as model:
                # Simple Bayesian model for proportion/mean
                if "beta" in prior_params:
                    # Beta prior for proportions
                    alpha, beta = prior_params["beta"]
                    theta = pm.Beta("theta", alpha=alpha, beta=beta)
                    obs = pm.Bernoulli("obs", p=theta, observed=data)
                else:
                    # Normal prior for means
                    mu_prior = prior_params.get("mu", 0)
                    sigma_prior = prior_params.get("sigma", 1)
                    theta = pm.Normal("theta", mu=mu_prior, sigma=sigma_prior)
                    obs = pm.Normal("obs", mu=theta, sigma=1, observed=data)
                
                # Sample from posterior
                trace = pm.sample(1000, return_inferencedata=True, random_seed=self.random_state)
            
            # Extract results
            posterior_samples = trace.posterior["theta"].values.flatten()
            point_est = np.mean(posterior_samples)
            
            alpha = 1 - self.confidence_level
            lower = np.percentile(posterior_samples, 100 * alpha/2)
            upper = np.percentile(posterior_samples, 100 * (1 - alpha/2))
            
            return UncertaintyResults(
                point_estimate=point_est,
                confidence_interval=(lower, upper),
                credible_interval=(lower, upper),
                bayesian_samples=posterior_samples,
                method="bayesian"
            )
            
        except Exception as e:
            warnings.warn(f"Bayesian analysis failed: {e}. Falling back to bootstrap.")
            return self.bootstrap_metric(data, np.mean)

class ModelCalibrator:
    """Advanced model calibration analysis."""
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    
    def analyze_calibration(self, y_true: np.ndarray, y_prob: np.ndarray) -> CalibrationResults:
        """Comprehensive calibration analysis."""
        
        # Basic calibration metrics
        brier = brier_score_loss(y_true, y_prob)
        
        # Calibration curve
        fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=self.n_bins)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += prop_in_bin * bin_error
                mce = max(mce, bin_error)
        
        # Reliability and resolution
        reliability = np.sum((fraction_pos - mean_pred) ** 2 * 
                           np.histogram(y_prob, bins=self.n_bins)[0] / len(y_prob))
        
        resolution = np.sum((fraction_pos - np.mean(y_true)) ** 2 * 
                          np.histogram(y_prob, bins=self.n_bins)[0] / len(y_prob))
        
        # Calibration methods
        isotonic_cal = IsotonicRegression(out_of_bounds="clip")
        isotonic_cal.fit(y_prob, y_true)
        
        return CalibrationResults(
            brier_score=brier,
            reliability_score=reliability,
            resolution_score=resolution,
            ece=ece,
            mce=mce,
            calibration_curve=(fraction_pos, mean_pred),
            isotonic_calibrator=isotonic_cal
        )

class IndividualFairnessAnalyzer:
    """Individual fairness analysis beyond group fairness."""
    
    def __init__(self, distance_metric="euclidean"):
        self.distance_metric = distance_metric
    
    def compute_individual_fairness(self, X: np.ndarray, y_pred: np.ndarray, 
                                  sensitive_features: np.ndarray) -> IndividualFairnessResults:
        """Analyze individual fairness using Lipschitz continuity."""
        
        n_samples = len(X)
        
        # Compute pairwise distances in feature space
        from scipy.spatial.distance import pdist, squareform
        
        if self.distance_metric == "euclidean":
            distances = squareform(pdist(X, metric='euclidean'))
        elif self.distance_metric == "manhattan":
            distances = squareform(pdist(X, metric='manhattan'))
        else:
            distances = squareform(pdist(X, metric=self.distance_metric))
        
        # Compute prediction differences
        pred_diffs = np.abs(y_pred.reshape(-1, 1) - y_pred.reshape(1, -1))
        
        # Lipschitz constant (worst-case ratio)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = pred_diffs / distances
            ratios[distances == 0] = 0  # Handle identical points
            ratios[np.isnan(ratios)] = 0
            ratios[np.isinf(ratios)] = 0
        
        lipschitz_constant = np.max(ratios)
        
        # Individual disparities
        individual_disparities = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Find similar individuals (in same sensitive group)
            same_group = sensitive_features == sensitive_features[i]
            if np.sum(same_group) > 1:
                same_group_idx = np.where(same_group)[0]
                same_group_distances = distances[i][same_group_idx]
                same_group_pred_diffs = pred_diffs[i][same_group_idx]
                
                # Average disparity for similar individuals
                if len(same_group_distances) > 0:
                    weights = 1 / (same_group_distances + 1e-8)  # Inverse distance weighting
                    individual_disparities[i] = np.average(same_group_pred_diffs, weights=weights)
        
        # Consistency score (1 - normalized disparity)
        consistency_score = 1 - np.mean(individual_disparities)
        
        return IndividualFairnessResults(
            lipschitz_constant=lipschitz_constant,
            consistency_score=consistency_score,
            individual_disparities=individual_disparities,
            similarity_matrix=distances,
            method=self.distance_metric
        )

class IntersectionalBiasAnalyzer:
    """Analyze bias across multiple intersecting demographic dimensions."""
    
    def __init__(self):
        self.results = {}
    
    def analyze_intersectional_bias(self, df: pd.DataFrame, 
                                  demographic_cols: List[str],
                                  outcome_col: str,
                                  prediction_col: str) -> Dict[str, Any]:
        """Comprehensive intersectional bias analysis."""
        
        results = {
            "single_dimension": {},
            "pairwise_interactions": {},
            "higher_order_interactions": {},
            "disparity_amplification": {},
            "intersectional_metrics": {}
        }
        
        # Single dimension analysis
        for col in demographic_cols:
            results["single_dimension"][col] = self._single_dimension_analysis(
                df, col, outcome_col, prediction_col
            )
        
        # Pairwise interactions
        from itertools import combinations
        for col1, col2 in combinations(demographic_cols, 2):
            key = f"{col1}_{col2}"
            results["pairwise_interactions"][key] = self._pairwise_analysis(
                df, col1, col2, outcome_col, prediction_col
            )
        
        # Higher-order interactions (if computationally feasible)
        if len(demographic_cols) <= 4:
            for cols in combinations(demographic_cols, 3):
                key = "_".join(cols)
                results["higher_order_interactions"][key] = self._higher_order_analysis(
                    df, cols, outcome_col, prediction_col
                )
        
        # Disparity amplification analysis
        results["disparity_amplification"] = self._analyze_disparity_amplification(
            df, demographic_cols, outcome_col, prediction_col
        )
        
        return results
    
    def _single_dimension_analysis(self, df, demo_col, outcome_col, pred_col):
        """Analyze bias for a single demographic dimension."""
        groups = df[demo_col].unique()
        group_metrics = {}
        
        for group in groups:
            mask = df[demo_col] == group
            group_data = df[mask]
            
            if len(group_data) > 0:
                accuracy = (group_data[outcome_col] == group_data[pred_col]).mean()
                pred_positive_rate = (group_data[pred_col] == 1).mean()
                true_positive_rate = (group_data[outcome_col] == 1).mean()
                
                group_metrics[group] = {
                    "n": len(group_data),
                    "accuracy": accuracy,
                    "pred_positive_rate": pred_positive_rate,
                    "true_positive_rate": true_positive_rate,
                    "selection_rate": pred_positive_rate
                }
        
        return group_metrics
    
    def _pairwise_analysis(self, df, col1, col2, outcome_col, pred_col):
        """Analyze bias for pairwise demographic intersections."""
        intersection_metrics = {}
        
        for val1 in df[col1].unique():
            for val2 in df[col2].unique():
                mask = (df[col1] == val1) & (df[col2] == val2)
                group_data = df[mask]
                
                if len(group_data) > 5:  # Minimum sample size
                    key = f"{val1}_{val2}"
                    accuracy = (group_data[outcome_col] == group_data[pred_col]).mean()
                    pred_positive_rate = (group_data[pred_col] == 1).mean()
                    
                    intersection_metrics[key] = {
                        "n": len(group_data),
                        "accuracy": accuracy,
                        "pred_positive_rate": pred_positive_rate
                    }
        
        return intersection_metrics
    
    def _higher_order_analysis(self, df, cols, outcome_col, pred_col):
        """Analyze higher-order demographic intersections."""
        # Create intersection column
        intersection_col = df[cols].apply(lambda x: "_".join(x.astype(str)), axis=1)
        
        intersection_metrics = {}
        for intersection in intersection_col.unique():
            mask = intersection_col == intersection
            group_data = df[mask]
            
            if len(group_data) > 5:
                accuracy = (group_data[outcome_col] == group_data[pred_col]).mean()
                intersection_metrics[intersection] = {
                    "n": len(group_data),
                    "accuracy": accuracy
                }
        
        return intersection_metrics
    
    def _analyze_disparity_amplification(self, df, demo_cols, outcome_col, pred_col):
        """Analyze if intersections amplify or diminish disparities."""
        
        # Calculate single-dimension disparities
        single_disparities = {}
        for col in demo_cols:
            groups = df[col].unique()
            group_rates = []
            for group in groups:
                mask = df[col] == group
                if mask.sum() > 0:
                    rate = (df[mask][pred_col] == 1).mean()
                    group_rates.append(rate)
            
            if len(group_rates) > 1:
                single_disparities[col] = max(group_rates) - min(group_rates)
        
        # Calculate intersection disparities
        from itertools import combinations
        intersection_disparities = {}
        
        for col1, col2 in combinations(demo_cols, 2):
            intersection_rates = []
            for val1 in df[col1].unique():
                for val2 in df[col2].unique():
                    mask = (df[col1] == val1) & (df[col2] == val2)
                    if mask.sum() > 5:
                        rate = (df[mask][pred_col] == 1).mean()
                        intersection_rates.append(rate)
            
            if len(intersection_rates) > 1:
                key = f"{col1}_{col2}"
                intersection_disparities[key] = max(intersection_rates) - min(intersection_rates)
        
        return {
            "single_dimension_disparities": single_disparities,
            "intersection_disparities": intersection_disparities,
            "amplification_detected": any(
                intersection_disparities.get(k, 0) > max(
                    single_disparities.get(k.split('_')[0], 0),
                    single_disparities.get(k.split('_')[1], 0)
                ) for k in intersection_disparities
            )
        }

# Convenience functions
def calculate_uncertainty_for_fairness_metrics(df: pd.DataFrame, 
                                             sensitive_attr: str,
                                             outcome_col: str,
                                             prediction_col: str) -> Dict[str, UncertaintyResults]:
    """Calculate uncertainty for all fairness metrics."""
    
    quantifier = UncertaintyQuantifier()
    results = {}
    
    # Demographic parity with uncertainty
    def demographic_parity_metric(data_indices):
        subset = df.iloc[data_indices]
        # Calculate demographic parity for this bootstrap sample
        groups = subset[sensitive_attr].unique()
        if len(groups) < 2:
            return 0
        
        rates = []
        for group in groups:
            group_mask = subset[sensitive_attr] == group
            if group_mask.sum() > 0:
                rate = (subset[group_mask][prediction_col] == 1).mean()
                rates.append(rate)
        
        return max(rates) - min(rates) if len(rates) > 1 else 0
    
    # Bootstrap for demographic parity
    indices = np.arange(len(df))
    results["demographic_parity"] = quantifier.bootstrap_metric(
        indices, demographic_parity_metric
    )
    
    # Accuracy by group with uncertainty
    for group in df[sensitive_attr].unique():
        group_mask = df[sensitive_attr] == group
        if group_mask.sum() > 0:
            group_predictions = df[group_mask][prediction_col].values
            group_outcomes = df[group_mask][outcome_col].values
            
            def accuracy_metric(data):
                return (data == group_outcomes[:len(data)]).mean()
            
            results[f"accuracy_{group}"] = quantifier.bootstrap_metric(
                group_predictions, accuracy_metric
            )
    
    return results

# Export all classes and functions
__all__ = [
    'UncertaintyResults',
    'CalibrationResults', 
    'IndividualFairnessResults',
    'UncertaintyQuantifier',
    'ModelCalibrator',
    'IndividualFairnessAnalyzer',
    'IntersectionalBiasAnalyzer',
    'calculate_uncertainty_for_fairness_metrics'
]