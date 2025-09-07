#!/usr/bin/env python3
"""
Meta-Analysis Capabilities for LLM Risk Fairness Experiments

This module provides sophisticated meta-analysis tools including:
- Systematic literature review integration
- Effect size meta-analysis across studies
- Publication bias detection (funnel plots)
- Forest plot visualization
- Bayesian meta-analysis with hierarchical models
- Cross-experiment statistical synthesis
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Meta-analysis specific libraries
try:
    import pymc as pm
    import arviz as az
    import bambi as bmb
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Publication bias detection
from sklearn.linear_model import LinearRegression


@dataclass
class StudyData:
    """Data structure for individual study in meta-analysis."""
    study_id: str
    study_name: str
    effect_size: float
    standard_error: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    p_value: float
    
    # Study characteristics
    models_tested: List[str]
    demographic_groups: List[str]
    risk_assessment_domain: str
    publication_year: int
    study_quality_score: float
    
    # Effect size details
    effect_type: str  # 'cohen_d', 'cramers_v', 'odds_ratio', 'difference_in_means'
    outcome_measure: str  # 'demographic_parity', 'equalized_odds', 'accuracy_difference'
    
    # Additional metadata
    geographic_region: str
    sample_characteristics: Dict[str, Any]
    methodology_notes: str


@dataclass
class MetaAnalysisConfig:
    """Configuration for meta-analysis procedures."""
    # Effect size settings
    effect_size_metric: str = "standardized_mean_difference"
    confidence_level: float = 0.95
    heterogeneity_threshold: float = 0.75  # I² threshold
    
    # Statistical methods
    fixed_effects_model: bool = False
    random_effects_model: bool = True
    bayesian_analysis: bool = True
    publication_bias_tests: bool = True
    
    # Subgroup analysis
    perform_subgroup_analysis: bool = True
    subgroup_variables: List[str] = None
    meta_regression: bool = True
    
    # Sensitivity analysis
    leave_one_out_analysis: bool = True
    influence_diagnostics: bool = True
    outlier_detection: bool = True
    
    # Visualization
    forest_plot: bool = True
    funnel_plot: bool = True
    bayesian_posterior_plots: bool = True
    
    def __post_init__(self):
        if self.subgroup_variables is None:
            self.subgroup_variables = ['models_tested', 'demographic_groups', 'geographic_region']


class MetaAnalysisEngine:
    """Core engine for conducting meta-analyses of LLM fairness experiments."""
    
    def __init__(self, config: MetaAnalysisConfig = None):
        self.config = config or MetaAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        self.studies: List[StudyData] = []
        
        # Results storage
        self.meta_analysis_results = {}
        self.heterogeneity_stats = {}
        self.publication_bias_results = {}
        self.subgroup_results = {}
    
    def add_study_from_experiment(self, experiment_dir: str, study_name: str = None) -> str:
        """Add study data from experiment results directory."""
        
        try:
            exp_path = Path(experiment_dir)
            
            # Load experiment data
            results_df = pd.read_csv(exp_path / "results.csv")
            experiment_report = {}
            stats_summary = {}
            
            report_file = exp_path / "experiment_report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    experiment_report = json.load(f)
            
            stats_file = exp_path / "stats_summary.json"
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats_summary = json.load(f)
            
            # Extract study characteristics
            study_id = exp_path.name
            study_name = study_name or experiment_report.get('experiment_name', study_id)
            
            # Compute effect sizes and statistics
            effect_size, standard_error, sample_size = self._compute_study_effect_size(results_df)
            
            # Compute confidence interval
            ci_lower = effect_size - 1.96 * standard_error
            ci_upper = effect_size + 1.96 * standard_error
            
            # Compute p-value (two-tailed test)
            z_score = effect_size / standard_error if standard_error > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            # Extract metadata
            models_tested = results_df['model'].unique().tolist() if 'model' in results_df.columns else []
            demographic_groups = []
            for col in ['gender', 'age_group', 'ethnicity', 'location']:
                if col in results_df.columns:
                    demographic_groups.extend(results_df[col].unique().tolist())
            
            # Create study data
            study = StudyData(
                study_id=study_id,
                study_name=study_name,
                effect_size=effect_size,
                standard_error=standard_error,
                sample_size=sample_size,
                confidence_interval=(ci_lower, ci_upper),
                p_value=p_value,
                models_tested=models_tested,
                demographic_groups=demographic_groups,
                risk_assessment_domain="superannuation",
                publication_year=datetime.now().year,
                study_quality_score=self._assess_study_quality(results_df, experiment_report),
                effect_type="standardized_mean_difference",
                outcome_measure="demographic_parity_violation",
                geographic_region="Australia",
                sample_characteristics={
                    'total_responses': len(results_df),
                    'unique_subjects': results_df['subject_id'].nunique() if 'subject_id' in results_df.columns else 0,
                    'models_count': len(models_tested)
                },
                methodology_notes=f"Experiment conducted using {len(models_tested)} LLM models"
            )
            
            self.studies.append(study)
            self.logger.info(f"Added study: {study_name} (Effect size: {effect_size:.3f} ± {standard_error:.3f})")
            
            return study_id
            
        except Exception as e:
            self.logger.error(f"Failed to add study from {experiment_dir}: {str(e)}")
            return None
    
    def _compute_study_effect_size(self, df: pd.DataFrame) -> Tuple[float, float, int]:
        """Compute effect size for demographic bias from experiment results."""
        
        if 'gender' not in df.columns or 'risk_label' not in df.columns:
            return 0.0, 0.1, len(df)
        
        try:
            # Compute demographic parity violation as effect size
            gender_groups = df['gender'].unique()
            if len(gender_groups) < 2:
                return 0.0, 0.1, len(df)
            
            # Calculate risk label distribution by gender
            risk_distributions = {}
            for gender in gender_groups:
                gender_data = df[df['gender'] == gender]
                risk_dist = gender_data['risk_label'].value_counts(normalize=True)
                risk_distributions[gender] = risk_dist
            
            # Compute maximum difference across risk labels (demographic parity violation)
            all_risk_labels = set()
            for dist in risk_distributions.values():
                all_risk_labels.update(dist.index)
            
            max_difference = 0.0
            for risk_label in all_risk_labels:
                rates = [risk_distributions[gender].get(risk_label, 0) for gender in gender_groups]
                difference = max(rates) - min(rates)
                max_difference = max(max_difference, difference)
            
            # Estimate standard error using bootstrap or analytical approximation
            n_male = len(df[df['gender'] == gender_groups[0]])
            n_female = len(df[df['gender'] == gender_groups[1]])
            
            # Approximate standard error for proportion difference
            p_male = risk_distributions[gender_groups[0]].iloc[0] if len(risk_distributions[gender_groups[0]]) > 0 else 0.5
            p_female = risk_distributions[gender_groups[1]].iloc[0] if len(risk_distributions[gender_groups[1]]) > 0 else 0.5
            
            se = np.sqrt((p_male * (1 - p_male) / n_male) + (p_female * (1 - p_female) / n_female))
            
            return max_difference, max(se, 0.01), len(df)
            
        except Exception as e:
            self.logger.error(f"Error computing effect size: {str(e)}")
            return 0.0, 0.1, len(df)
    
    def _assess_study_quality(self, df: pd.DataFrame, report: Dict) -> float:
        """Assess study quality on 0-1 scale."""
        
        quality_score = 0.0
        max_score = 0.0
        
        # Sample size adequacy (20 points)
        sample_size = len(df)
        if sample_size >= 100:
            quality_score += 20
        elif sample_size >= 50:
            quality_score += 15
        elif sample_size >= 20:
            quality_score += 10
        max_score += 20
        
        # Multiple models tested (15 points)
        models_count = df['model'].nunique() if 'model' in df.columns else 1
        if models_count >= 3:
            quality_score += 15
        elif models_count >= 2:
            quality_score += 10
        elif models_count >= 1:
            quality_score += 5
        max_score += 15
        
        # Demographic diversity (15 points)
        demographic_cols = ['gender', 'age_group', 'ethnicity', 'location']
        diversity_score = sum(1 for col in demographic_cols if col in df.columns and df[col].nunique() > 1)
        quality_score += (diversity_score / 4) * 15
        max_score += 15
        
        # Statistical rigor (20 points)
        if 'risk_band_correct' in df.columns:
            quality_score += 10  # Has accuracy measures
        if 'response_time_seconds' in df.columns:
            quality_score += 5   # Has performance measures
        if len(df['risk_label'].unique()) >= 5:
            quality_score += 5   # Good outcome diversity
        max_score += 20
        
        # Methodological completeness (15 points)
        if report.get('total_cost', 0) > 0:
            quality_score += 5   # Cost tracking
        if report.get('duration_hours', 0) > 0:
            quality_score += 5   # Time tracking
        if 'stratified_sampling' in str(report):
            quality_score += 5   # Proper sampling
        max_score += 15
        
        # Reproducibility (15 points)
        if 'temperature' in str(report) and '0.0' in str(report):
            quality_score += 5   # Deterministic settings
        if 'cache' in str(report):
            quality_score += 5   # Caching for consistency
        if 'seed' in str(report) or 'random' in str(report):
            quality_score += 5   # Random seed control
        max_score += 15
        
        return quality_score / max_score if max_score > 0 else 0.5
    
    def run_meta_analysis(self) -> Dict[str, Any]:
        """Run comprehensive meta-analysis of all studies."""
        
        if len(self.studies) < 2:
            raise ValueError("Meta-analysis requires at least 2 studies")
        
        self.logger.info(f"Running meta-analysis on {len(self.studies)} studies")
        
        results = {}
        
        # 1. Fixed and random effects models
        if self.config.fixed_effects_model:
            results['fixed_effects'] = self._run_fixed_effects_model()
        
        if self.config.random_effects_model:
            results['random_effects'] = self._run_random_effects_model()
        
        # 2. Heterogeneity assessment
        results['heterogeneity'] = self._assess_heterogeneity()
        
        # 3. Publication bias assessment
        if self.config.publication_bias_tests:
            results['publication_bias'] = self._assess_publication_bias()
        
        # 4. Bayesian meta-analysis
        if self.config.bayesian_analysis and BAYESIAN_AVAILABLE:
            results['bayesian'] = self._run_bayesian_meta_analysis()
        
        # 5. Subgroup analysis
        if self.config.perform_subgroup_analysis:
            results['subgroup_analysis'] = self._run_subgroup_analysis()
        
        # 6. Sensitivity analysis
        if self.config.leave_one_out_analysis:
            results['sensitivity_analysis'] = self._run_sensitivity_analysis()
        
        # Store results
        self.meta_analysis_results = results
        
        return results
    
    def _run_fixed_effects_model(self) -> Dict[str, float]:
        """Run fixed effects meta-analysis."""
        
        # Extract data
        effect_sizes = np.array([study.effect_size for study in self.studies])
        standard_errors = np.array([study.standard_error for study in self.studies])
        weights = 1 / (standard_errors ** 2)
        
        # Compute weighted mean effect size
        weighted_effect = np.sum(weights * effect_sizes) / np.sum(weights)
        weighted_se = np.sqrt(1 / np.sum(weights))
        
        # Confidence interval
        ci_lower = weighted_effect - 1.96 * weighted_se
        ci_upper = weighted_effect + 1.96 * weighted_se
        
        # Z-test
        z_score = weighted_effect / weighted_se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'effect_size': weighted_effect,
            'standard_error': weighted_se,
            'confidence_interval': [ci_lower, ci_upper],
            'z_score': z_score,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _run_random_effects_model(self) -> Dict[str, float]:
        """Run random effects meta-analysis using DerSimonian-Laird method."""
        
        # Fixed effects first
        fixed_results = self._run_fixed_effects_model()
        
        # Extract data
        effect_sizes = np.array([study.effect_size for study in self.studies])
        standard_errors = np.array([study.standard_error for study in self.studies])
        weights_fixed = 1 / (standard_errors ** 2)
        
        # Compute Q statistic
        weighted_mean_fixed = fixed_results['effect_size']
        Q = np.sum(weights_fixed * (effect_sizes - weighted_mean_fixed) ** 2)
        
        # Estimate between-study variance (tau²)
        k = len(self.studies)
        df = k - 1
        C = np.sum(weights_fixed) - np.sum(weights_fixed ** 2) / np.sum(weights_fixed)
        tau_squared = max(0, (Q - df) / C) if C > 0 else 0
        
        # Random effects weights
        weights_random = 1 / (standard_errors ** 2 + tau_squared)
        
        # Random effects estimate
        weighted_effect = np.sum(weights_random * effect_sizes) / np.sum(weights_random)
        weighted_se = np.sqrt(1 / np.sum(weights_random))
        
        # Confidence interval
        ci_lower = weighted_effect - 1.96 * weighted_se
        ci_upper = weighted_effect + 1.96 * weighted_se
        
        # Z-test
        z_score = weighted_effect / weighted_se if weighted_se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return {
            'effect_size': weighted_effect,
            'standard_error': weighted_se,
            'confidence_interval': [ci_lower, ci_upper],
            'z_score': z_score,
            'p_value': p_value,
            'tau_squared': tau_squared,
            'Q_statistic': Q,
            'Q_p_value': 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0,
            'significant': p_value < 0.05
        }
    
    def _assess_heterogeneity(self) -> Dict[str, float]:
        """Assess heterogeneity between studies."""
        
        # Get Q statistic from random effects
        random_results = self._run_random_effects_model()
        Q = random_results['Q_statistic']
        df = len(self.studies) - 1
        
        # I² statistic
        I_squared = max(0, ((Q - df) / Q) * 100) if Q > 0 else 0
        
        # H statistic
        H = np.sqrt(Q / df) if df > 0 else 1.0
        
        # Tau (between-study standard deviation)
        tau = np.sqrt(random_results['tau_squared'])
        
        # Interpretation
        if I_squared < 25:
            heterogeneity_level = "low"
        elif I_squared < 50:
            heterogeneity_level = "moderate"
        elif I_squared < 75:
            heterogeneity_level = "substantial"
        else:
            heterogeneity_level = "considerable"
        
        return {
            'Q_statistic': Q,
            'Q_df': df,
            'Q_p_value': random_results['Q_p_value'],
            'I_squared': I_squared,
            'H_statistic': H,
            'tau': tau,
            'tau_squared': random_results['tau_squared'],
            'heterogeneity_level': heterogeneity_level,
            'significant_heterogeneity': random_results['Q_p_value'] < 0.10
        }
    
    def _assess_publication_bias(self) -> Dict[str, Any]:
        """Assess publication bias using multiple methods."""
        
        results = {}
        
        # Extract data
        effect_sizes = np.array([study.effect_size for study in self.studies])
        standard_errors = np.array([study.standard_error for study in self.studies])
        
        # 1. Egger's test (linear regression of effect size on standard error)
        if len(self.studies) >= 3:
            # Regress standardized effect size on precision
            precision = 1 / standard_errors
            standardized_effects = effect_sizes / standard_errors
            
            reg = LinearRegression().fit(precision.reshape(-1, 1), standardized_effects)
            intercept = reg.intercept_
            
            # T-test for intercept
            residuals = standardized_effects - reg.predict(precision.reshape(-1, 1))
            se_intercept = np.sqrt(np.sum(residuals ** 2) / (len(self.studies) - 2)) / np.sqrt(np.sum((precision - precision.mean()) ** 2))
            t_stat = intercept / se_intercept if se_intercept > 0 else 0
            p_value_egger = 2 * (1 - stats.t.cdf(abs(t_stat), len(self.studies) - 2))
            
            results['eggers_test'] = {
                'intercept': intercept,
                't_statistic': t_stat,
                'p_value': p_value_egger,
                'significant_bias': p_value_egger < 0.05
            }
        
        # 2. Begg's test (rank correlation between effect sizes and variances)
        if len(self.studies) >= 3:
            variances = standard_errors ** 2
            rank_corr, p_value_begg = stats.spearmanr(effect_sizes, variances)
            
            results['beggs_test'] = {
                'rank_correlation': rank_corr,
                'p_value': p_value_begg,
                'significant_bias': p_value_begg < 0.05
            }
        
        # 3. Fail-safe N (number of null studies needed to make result non-significant)
        if len(self.studies) >= 3:
            # Fixed effects z-score
            fixed_results = self._run_fixed_effects_model()
            z_fixed = fixed_results['z_score']
            
            # Critical z for alpha = 0.05
            z_critical = 1.96
            
            if abs(z_fixed) > z_critical:
                fail_safe_n = ((abs(z_fixed) - z_critical) ** 2) * len(self.studies) / (z_critical ** 2)
                results['fail_safe_n'] = {
                    'n_studies_needed': fail_safe_n,
                    'robust_to_bias': fail_safe_n > 5 * len(self.studies) + 10
                }
        
        return results
    
    def _run_bayesian_meta_analysis(self) -> Dict[str, Any]:
        """Run Bayesian meta-analysis with hierarchical modeling."""
        
        if not BAYESIAN_AVAILABLE:
            return {'error': 'PyMC not available for Bayesian analysis'}
        
        try:
            # Prepare data
            effect_sizes = [study.effect_size for study in self.studies]
            standard_errors = [study.standard_error for study in self.studies]
            
            with pm.Model() as model:
                # Hyperpriors for population parameters
                mu = pm.Normal('mu', mu=0, sigma=1)  # Population effect
                tau = pm.HalfNormal('tau', sigma=0.5)  # Between-study heterogeneity
                
                # Study-specific true effects
                theta = pm.Normal('theta', mu=mu, sigma=tau, shape=len(self.studies))
                
                # Observed effect sizes
                y = pm.Normal('y', mu=theta, sigma=standard_errors, observed=effect_sizes)
                
                # Sample from posterior
                trace = pm.sample(2000, tune=1000, chains=4, return_inferencedata=True)
            
            # Extract results
            summary = az.summary(trace, var_names=['mu', 'tau'])
            
            # Population effect estimate
            mu_mean = summary.loc['mu', 'mean']
            mu_hdi = az.hdi(trace, var_names=['mu'])['mu'].values
            
            # Heterogeneity estimate
            tau_mean = summary.loc['tau', 'mean']
            
            # Posterior probability of positive effect
            mu_samples = trace.posterior['mu'].values.flatten()
            prob_positive = np.mean(mu_samples > 0)
            
            return {
                'population_effect': mu_mean,
                'population_effect_hdi': mu_hdi.tolist(),
                'between_study_sd': tau_mean,
                'probability_positive_effect': prob_positive,
                'model_summary': summary.to_dict(),
                'n_samples': len(mu_samples)
            }
            
        except Exception as e:
            self.logger.error(f"Bayesian meta-analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _run_subgroup_analysis(self) -> Dict[str, Any]:
        """Run subgroup analysis based on study characteristics."""
        
        results = {}
        
        for variable in self.config.subgroup_variables:
            if variable == 'models_tested':
                # Group by primary model used
                subgroups = defaultdict(list)
                for study in self.studies:
                    primary_model = study.models_tested[0] if study.models_tested else 'unknown'
                    subgroups[primary_model].append(study)
                
            elif variable == 'geographic_region':
                subgroups = defaultdict(list)
                for study in self.studies:
                    subgroups[study.geographic_region].append(study)
                
            elif variable == 'study_quality':
                # High vs low quality studies
                median_quality = np.median([study.study_quality_score for study in self.studies])
                subgroups = {'high_quality': [], 'low_quality': []}
                for study in self.studies:
                    if study.study_quality_score >= median_quality:
                        subgroups['high_quality'].append(study)
                    else:
                        subgroups['low_quality'].append(study)
            
            else:
                continue
            
            # Analyze each subgroup
            subgroup_results = {}
            for subgroup_name, subgroup_studies in subgroups.items():
                if len(subgroup_studies) >= 2:
                    # Temporarily set studies to subgroup
                    original_studies = self.studies
                    self.studies = subgroup_studies
                    
                    # Run meta-analysis on subgroup
                    subgroup_ma = self._run_random_effects_model()
                    subgroup_results[subgroup_name] = {
                        'n_studies': len(subgroup_studies),
                        'effect_size': subgroup_ma['effect_size'],
                        'confidence_interval': subgroup_ma['confidence_interval'],
                        'p_value': subgroup_ma['p_value']
                    }
                    
                    # Restore original studies
                    self.studies = original_studies
            
            # Test for subgroup differences (Q_between)
            if len(subgroup_results) >= 2:
                # Compute Q_between statistic
                overall_effect = self._run_random_effects_model()['effect_size']
                Q_between = 0
                for subgroup_name, subgroup_result in subgroup_results.items():
                    n_studies = subgroup_result['n_studies']
                    effect_diff = subgroup_result['effect_size'] - overall_effect
                    Q_between += n_studies * (effect_diff ** 2)
                
                df_between = len(subgroup_results) - 1
                p_between = 1 - stats.chi2.cdf(Q_between, df_between)
                
                subgroup_results['test_for_differences'] = {
                    'Q_between': Q_between,
                    'df': df_between,
                    'p_value': p_between,
                    'significant_differences': p_between < 0.05
                }
            
            results[variable] = subgroup_results
        
        return results
    
    def _run_sensitivity_analysis(self) -> Dict[str, Any]:
        """Run sensitivity analysis including leave-one-out and influence diagnostics."""
        
        results = {}
        
        # Leave-one-out analysis
        leave_one_out_results = []
        original_studies = self.studies.copy()
        
        for i, excluded_study in enumerate(self.studies):
            # Remove one study
            self.studies = [study for j, study in enumerate(original_studies) if j != i]
            
            # Run meta-analysis
            if len(self.studies) >= 2:
                ma_result = self._run_random_effects_model()
                leave_one_out_results.append({
                    'excluded_study': excluded_study.study_name,
                    'effect_size': ma_result['effect_size'],
                    'confidence_interval': ma_result['confidence_interval'],
                    'p_value': ma_result['p_value']
                })
        
        # Restore original studies
        self.studies = original_studies
        
        # Compute influence statistics
        overall_effect = self._run_random_effects_model()['effect_size']
        influences = []
        
        for result in leave_one_out_results:
            influence = abs(result['effect_size'] - overall_effect)
            influences.append(influence)
        
        results['leave_one_out'] = {
            'results': leave_one_out_results,
            'max_influence': max(influences) if influences else 0,
            'influential_studies': [
                result['excluded_study'] for i, result in enumerate(leave_one_out_results)
                if influences[i] > np.std(influences) * 2  # Studies with influence > 2 SD
            ]
        }
        
        # Outlier detection based on standardized residuals
        outliers = []
        for study in self.studies:
            # Compute standardized residual
            residual = study.effect_size - overall_effect
            standardized_residual = residual / study.standard_error
            
            if abs(standardized_residual) > 2.58:  # 99% confidence level
                outliers.append({
                    'study_name': study.study_name,
                    'effect_size': study.effect_size,
                    'standardized_residual': standardized_residual
                })
        
        results['outlier_detection'] = {
            'outliers': outliers,
            'n_outliers': len(outliers)
        }
        
        return results
    
    def generate_forest_plot(self, output_path: str = None) -> str:
        """Generate forest plot visualization."""
        
        if not self.studies:
            raise ValueError("No studies available for forest plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.studies) * 0.8)))
        
        # Extract data
        study_names = [study.study_name[:30] + "..." if len(study.study_name) > 30 else study.study_name 
                      for study in self.studies]
        effect_sizes = [study.effect_size for study in self.studies]
        ci_lowers = [study.confidence_interval[0] for study in self.studies]
        ci_uppers = [study.confidence_interval[1] for study in self.studies]
        weights = [1 / (study.standard_error ** 2) for study in self.studies]
        
        # Normalize weights for point sizes
        max_weight = max(weights)
        point_sizes = [50 + (w / max_weight) * 200 for w in weights]
        
        y_positions = range(len(self.studies))
        
        # Plot individual studies
        colors = ['blue' if study.p_value < 0.05 else 'gray' for study in self.studies]
        
        # Plot confidence intervals
        for i, (lower, upper) in enumerate(zip(ci_lowers, ci_uppers)):
            ax.plot([lower, upper], [i, i], color=colors[i], linewidth=2)
        
        # Plot effect sizes
        ax.scatter(effect_sizes, y_positions, s=point_sizes, c=colors, alpha=0.7, zorder=5)
        
        # Add meta-analysis result if available
        if hasattr(self, 'meta_analysis_results') and 'random_effects' in self.meta_analysis_results:
            meta_result = self.meta_analysis_results['random_effects']
            meta_effect = meta_result['effect_size']
            meta_ci = meta_result['confidence_interval']
            
            # Add diamond for meta-analysis result
            y_meta = len(self.studies)
            diamond_x = [meta_ci[0], meta_effect, meta_ci[1], meta_effect, meta_ci[0]]
            diamond_y = [y_meta, y_meta + 0.2, y_meta, y_meta - 0.2, y_meta]
            
            ax.plot(diamond_x, diamond_y, color='red', linewidth=2)
            ax.fill(diamond_x, diamond_y, color='red', alpha=0.3)
            
            study_names.append("Meta-Analysis")
        
        # Vertical line at null effect
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_yticks(range(len(study_names)))
        ax.set_yticklabels(study_names)
        ax.set_xlabel('Effect Size (Demographic Parity Violation)')
        ax.set_title('Forest Plot: LLM Risk Assessment Bias Meta-Analysis')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Studies from top to bottom
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                      markersize=8, label='Significant (p < 0.05)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                      markersize=8, label='Not Significant'),
            plt.Line2D([0], [0], marker='D', color='red', markersize=8, 
                      label='Meta-Analysis Result')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = f"forest_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Forest plot saved to: {output_path}")
        return output_path
    
    def generate_funnel_plot(self, output_path: str = None) -> str:
        """Generate funnel plot for publication bias assessment."""
        
        if not self.studies:
            raise ValueError("No studies available for funnel plot")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        effect_sizes = np.array([study.effect_size for study in self.studies])
        standard_errors = np.array([study.standard_error for study in self.studies])
        
        # Plot studies
        colors = ['red' if study.p_value < 0.05 else 'blue' for study in self.studies]
        ax.scatter(effect_sizes, standard_errors, c=colors, alpha=0.7, s=60)
        
        # Add funnel (pseudo confidence limits)
        if hasattr(self, 'meta_analysis_results') and 'random_effects' in self.meta_analysis_results:
            meta_effect = self.meta_analysis_results['random_effects']['effect_size']
        else:
            meta_effect = np.mean(effect_sizes)
        
        # Create funnel lines
        max_se = max(standard_errors) * 1.1
        se_range = np.linspace(0, max_se, 100)
        
        # 95% confidence limits
        funnel_left_95 = meta_effect - 1.96 * se_range
        funnel_right_95 = meta_effect + 1.96 * se_range
        
        # 99% confidence limits
        funnel_left_99 = meta_effect - 2.58 * se_range
        funnel_right_99 = meta_effect + 2.58 * se_range
        
        ax.plot(funnel_left_95, se_range, 'k--', alpha=0.5, label='95% CI')
        ax.plot(funnel_right_95, se_range, 'k--', alpha=0.5)
        ax.plot(funnel_left_99, se_range, 'k:', alpha=0.5, label='99% CI')
        ax.plot(funnel_right_99, se_range, 'k:', alpha=0.5)
        
        # Vertical line at meta-analysis estimate
        ax.axvline(x=meta_effect, color='red', linestyle='-', alpha=0.7, label='Meta-Analysis Effect')
        
        # Formatting
        ax.set_xlabel('Effect Size (Demographic Parity Violation)')
        ax.set_ylabel('Standard Error')
        ax.set_title('Funnel Plot: Publication Bias Assessment')
        ax.invert_yaxis()  # Smaller SE at top
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add text annotation for interpretation
        if hasattr(self, 'meta_analysis_results') and 'publication_bias' in self.meta_analysis_results:
            bias_results = self.meta_analysis_results['publication_bias']
            if 'eggers_test' in bias_results:
                egger_p = bias_results['eggers_test']['p_value']
                bias_text = f"Egger's test p = {egger_p:.3f}"
                ax.text(0.05, 0.95, bias_text, transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        
        # Save plot
        if output_path is None:
            output_path = f"funnel_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Funnel plot saved to: {output_path}")
        return output_path
    
    def generate_meta_analysis_report(self, output_path: str) -> str:
        """Generate comprehensive meta-analysis report."""
        
        if not self.meta_analysis_results:
            self.run_meta_analysis()
        
        report_lines = [
            "# Meta-Analysis Report: LLM Risk Assessment Fairness",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Study Overview",
            f"- Number of studies: {len(self.studies)}",
            f"- Total sample size: {sum(study.sample_size for study in self.studies)}",
            f"- Effect size range: {min(study.effect_size for study in self.studies):.3f} to {max(study.effect_size for study in self.studies):.3f}",
            "",
            "## Studies Included"
        ]
        
        for i, study in enumerate(self.studies, 1):
            report_lines.extend([
                f"{i}. **{study.study_name}**",
                f"   - Effect size: {study.effect_size:.3f} (SE: {study.standard_error:.3f})",
                f"   - Sample size: {study.sample_size}",
                f"   - Models: {', '.join(study.models_tested)}",
                f"   - Quality score: {study.study_quality_score:.2f}",
                ""
            ])
        
        # Add meta-analysis results
        if 'random_effects' in self.meta_analysis_results:
            re_results = self.meta_analysis_results['random_effects']
            report_lines.extend([
                "## Meta-Analysis Results (Random Effects Model)",
                f"- **Pooled effect size**: {re_results['effect_size']:.3f}",
                f"- **95% Confidence interval**: [{re_results['confidence_interval'][0]:.3f}, {re_results['confidence_interval'][1]:.3f}]",
                f"- **Z-score**: {re_results['z_score']:.3f}",
                f"- **p-value**: {re_results['p_value']:.4f}",
                f"- **Statistical significance**: {'Yes' if re_results['significant'] else 'No'}",
                ""
            ])
        
        # Add heterogeneity assessment
        if 'heterogeneity' in self.meta_analysis_results:
            het_results = self.meta_analysis_results['heterogeneity']
            report_lines.extend([
                "## Heterogeneity Assessment",
                f"- **I² statistic**: {het_results['I_squared']:.1f}%",
                f"- **Heterogeneity level**: {het_results['heterogeneity_level'].title()}",
                f"- **Q-statistic**: {het_results['Q_statistic']:.3f} (p = {het_results['Q_p_value']:.4f})",
                f"- **Between-study variance (τ²)**: {het_results['tau_squared']:.4f}",
                ""
            ])
        
        # Add publication bias assessment
        if 'publication_bias' in self.meta_analysis_results:
            pb_results = self.meta_analysis_results['publication_bias']
            report_lines.extend([
                "## Publication Bias Assessment",
                ""
            ])
            
            if 'eggers_test' in pb_results:
                egger = pb_results['eggers_test']
                report_lines.extend([
                    f"- **Egger's test**: p = {egger['p_value']:.4f} ({'Significant bias detected' if egger['significant_bias'] else 'No significant bias'})",
                ])
            
            if 'beggs_test' in pb_results:
                begg = pb_results['beggs_test']
                report_lines.extend([
                    f"- **Begg's test**: p = {begg['p_value']:.4f} ({'Significant bias detected' if begg['significant_bias'] else 'No significant bias'})",
                ])
            
            if 'fail_safe_n' in pb_results:
                fsn = pb_results['fail_safe_n']
                report_lines.extend([
                    f"- **Fail-safe N**: {fsn['n_studies_needed']:.0f} studies needed ({'Robust' if fsn['robust_to_bias'] else 'Not robust'})",
                ])
            
            report_lines.append("")
        
        # Add interpretation
        if 'random_effects' in self.meta_analysis_results:
            re_results = self.meta_analysis_results['random_effects']
            effect_size = re_results['effect_size']
            
            report_lines.extend([
                "## Interpretation",
                ""
            ])
            
            if effect_size < 0.05:
                interpretation = "The meta-analysis suggests **minimal demographic bias** in LLM risk assessment."
            elif effect_size < 0.1:
                interpretation = "The meta-analysis suggests **low demographic bias** in LLM risk assessment."
            elif effect_size < 0.2:
                interpretation = "The meta-analysis suggests **moderate demographic bias** in LLM risk assessment."
            else:
                interpretation = "The meta-analysis suggests **substantial demographic bias** in LLM risk assessment."
            
            report_lines.append(interpretation)
            
            if re_results['significant']:
                report_lines.append("This result is **statistically significant**, indicating reliable evidence of bias.")
            else:
                report_lines.append("This result is **not statistically significant**, suggesting insufficient evidence for bias.")
        
        # Save report
        report_text = "\n".join(report_lines)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        self.logger.info(f"Meta-analysis report saved to: {output_path}")
        return output_path


def main():
    """CLI interface for meta-analysis."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Analysis for LLM Fairness Experiments")
    parser.add_argument('command', choices=['add-study', 'run-analysis', 'generate-plots', 'generate-report'])
    parser.add_argument('--experiment-dirs', nargs='+', help='Experiment directories to include')
    parser.add_argument('--output-dir', default='meta_analysis_output', help='Output directory')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize meta-analysis engine
    engine = MetaAnalysisEngine()
    
    if args.command == 'add-study' and args.experiment_dirs:
        for exp_dir in args.experiment_dirs:
            study_id = engine.add_study_from_experiment(exp_dir)
            print(f"Added study: {study_id}")
    
    elif args.command == 'run-analysis':
        if args.experiment_dirs:
            for exp_dir in args.experiment_dirs:
                engine.add_study_from_experiment(exp_dir)
        
        results = engine.run_meta_analysis()
        
        # Save results
        results_file = Path(args.output_dir) / 'meta_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Meta-analysis completed. Results saved to: {results_file}")
    
    elif args.command == 'generate-plots':
        if args.experiment_dirs:
            for exp_dir in args.experiment_dirs:
                engine.add_study_from_experiment(exp_dir)
        
        # Generate plots
        forest_plot = engine.generate_forest_plot(Path(args.output_dir) / 'forest_plot.png')
        funnel_plot = engine.generate_funnel_plot(Path(args.output_dir) / 'funnel_plot.png')
        
        print(f"Plots generated: {forest_plot}, {funnel_plot}")
    
    elif args.command == 'generate-report':
        if args.experiment_dirs:
            for exp_dir in args.experiment_dirs:
                engine.add_study_from_experiment(exp_dir)
        
        report_file = engine.generate_meta_analysis_report(Path(args.output_dir) / 'meta_analysis_report.md')
        print(f"Report generated: {report_file}")


if __name__ == "__main__":
    main()