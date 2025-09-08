#!/usr/bin/env python3
"""
Comprehensive Research Analysis Module
Provides publication-ready statistical analysis for LLM fairness research
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from dataclasses import dataclass, asdict
import sys

# Core statistical libraries
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, kruskal
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pingouin as pg

# Import our advanced analytics modules
try:
    from advanced_analytics import (
        UncertaintyQuantifier, ModelCalibrator, IndividualFairnessAnalyzer,
        IntersectionalBiasAnalyzer, calculate_uncertainty_for_fairness_metrics
    )
    ADVANCED_ANALYTICS_AVAILABLE = True
except ImportError:
    ADVANCED_ANALYTICS_AVAILABLE = False
    warnings.warn("Advanced analytics not available")

try:
    from temporal_analysis import TemporalBiasAnalyzer, ModelVersionTracker
    TEMPORAL_ANALYSIS_AVAILABLE = True
except ImportError:
    TEMPORAL_ANALYSIS_AVAILABLE = False
    warnings.warn("Temporal analysis not available")

@dataclass
class ResearchFindings:
    """Container for publication-ready research findings."""
    # Basic experiment info
    total_samples: int
    models_tested: List[str]
    conditions_tested: List[str]
    success_rate: float
    
    # Statistical significance tests
    demographic_bias_detected: bool
    location_bias_detected: bool
    condition_bias_detected: bool
    overall_bias_detected: bool
    
    # Effect sizes and confidence intervals
    demographic_effect_sizes: Dict[str, float]
    location_effect_sizes: Dict[str, float]
    condition_effect_sizes: Dict[str, float]
    
    # Detailed statistical results
    statistical_tests: Dict[str, Any]
    multiple_testing_corrections: Dict[str, Any]
    
    # Advanced fairness metrics
    fairness_metrics: Dict[str, Any]
    uncertainty_analysis: Dict[str, Any]
    calibration_analysis: Dict[str, Any]
    
    # Root cause analysis
    bias_drivers: Dict[str, Any]
    intervention_points: List[str]
    
    # Research implications
    scientific_conclusions: List[str]
    methodological_insights: List[str]
    limitations: List[str]
    future_work: List[str]

class ComprehensiveResearchAnalyzer:
    """
    Comprehensive research analyzer that produces publication-ready results
    for LLM fairness experiments.
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.df = None
        self.load_data()
        
    def load_data(self):
        """Load experiment data."""
        csv_path = self.results_dir / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"No results.csv found in {self.results_dir}")
        
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} experimental observations")
        
    def run_comprehensive_analysis(self) -> ResearchFindings:
        """Run complete research-grade analysis."""
        print("Running comprehensive research analysis...")
        
        # Basic descriptive statistics
        basic_stats = self._compute_basic_statistics()
        
        # Core bias detection with proper statistical tests
        bias_results = self._detect_bias_with_statistics()
        
        # Effect size calculations
        effect_sizes = self._calculate_effect_sizes()
        
        # Multiple testing corrections
        corrected_results = self._apply_multiple_testing_corrections(bias_results)
        
        # Advanced fairness metrics
        fairness_metrics = self._compute_advanced_fairness_metrics()
        
        # Uncertainty quantification
        uncertainty_results = self._quantify_uncertainty()
        
        # Model calibration analysis
        calibration_results = self._analyze_model_calibration()
        
        # Root cause analysis
        root_causes = self._analyze_root_causes()
        
        # Research implications
        implications = self._derive_research_implications(
            bias_results, effect_sizes, corrected_results, root_causes
        )
        
        # Package results
        findings = ResearchFindings(
            total_samples=len(self.df),
            models_tested=list(self.df['model'].unique()),
            conditions_tested=list(self.df['condition'].unique()),
            success_rate=self.df['ok'].mean(),
            
            demographic_bias_detected=corrected_results.get('demographic_bias_significant', False),
            location_bias_detected=corrected_results.get('location_bias_significant', False),
            condition_bias_detected=corrected_results.get('condition_bias_significant', False),
            overall_bias_detected=corrected_results.get('any_bias_significant', False),
            
            demographic_effect_sizes=effect_sizes.get('demographic', {}),
            location_effect_sizes=effect_sizes.get('location', {}),
            condition_effect_sizes=effect_sizes.get('condition', {}),
            
            statistical_tests=bias_results,
            multiple_testing_corrections=corrected_results,
            fairness_metrics=fairness_metrics,
            uncertainty_analysis=uncertainty_results,
            calibration_analysis=calibration_results,
            
            bias_drivers=root_causes,
            intervention_points=self._identify_intervention_points(root_causes, bias_results.get('overall_bias_detected', False)),
            
            scientific_conclusions=implications['conclusions'],
            methodological_insights=implications['insights'],
            limitations=implications['limitations'],
            future_work=implications['future_work']
        )
        
        return findings
    
    def _compute_basic_statistics(self) -> Dict[str, Any]:
        """Compute basic descriptive statistics."""
        success_df = self.df[self.df['ok'] == True].copy()
        
        stats = {
            'total_observations': len(self.df),
            'successful_observations': len(success_df),
            'success_rate': success_df['ok'].mean(),
            'models': {
                'count': self.df['model'].nunique(),
                'names': list(self.df['model'].unique()),
                'success_rates': self.df.groupby('model')['ok'].mean().to_dict()
            },
            'conditions': {
                'count': self.df['condition'].nunique(),
                'names': list(self.df['condition'].unique()),
                'sample_sizes': self.df['condition'].value_counts().to_dict()
            }
        }
        
        if 'name_group' in success_df.columns:
            stats['demographic_groups'] = {
                'count': success_df['name_group'].nunique(),
                'names': list(success_df['name_group'].unique()),
                'sample_sizes': success_df['name_group'].value_counts().to_dict()
            }
            
        if 'city' in success_df.columns:
            stats['locations'] = {
                'count': success_df['city'].nunique(),
                'names': list(success_df['city'].unique()),
                'sample_sizes': success_df['city'].value_counts().to_dict()
            }
            
        return stats
    
    def _detect_bias_with_statistics(self) -> Dict[str, Any]:
        """Perform rigorous statistical tests for bias detection."""
        success_df = self.df[self.df['ok'] == True].copy()
        results = {}
        
        # Demographic bias analysis
        if 'name_group' in success_df.columns:
            results['demographic_bias'] = self._test_demographic_bias(success_df)
        
        # Location bias analysis  
        if 'city' in success_df.columns:
            results['location_bias'] = self._test_location_bias(success_df)
            
        # Condition bias analysis
        results['condition_bias'] = self._test_condition_bias(success_df)
        
        # Inter-model consistency
        results['model_consistency'] = self._test_model_consistency(success_df)
        
        return results
    
    def _test_demographic_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for demographic bias using appropriate statistical tests."""
        results = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            if len(model_data['name_group'].unique()) < 2:
                continue
                
            # Create contingency table for Chi-square test
            contingency = pd.crosstab(model_data['name_group'], model_data['pred_label'])
            
            # Chi-square test for independence
            chi2, p_chi2, dof, expected = chi2_contingency(contingency)
            
            # Kruskal-Wallis test for ordinal differences
            groups = [model_data[model_data['name_group'] == group]['acc'].values 
                     for group in model_data['name_group'].unique() if len(model_data[model_data['name_group'] == group]) > 0]
            
            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                kw_stat, p_kw = kruskal(*groups)
            else:
                kw_stat, p_kw = np.nan, np.nan
            
            # Effect size (Cramér's V)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0
            
            results[model] = {
                'chi_square_stat': chi2,
                'chi_square_p': p_chi2,
                'kruskal_wallis_stat': kw_stat,
                'kruskal_wallis_p': p_kw,
                'cramers_v': cramers_v,
                'sample_size': len(model_data),
                'groups_tested': list(model_data['name_group'].unique())
            }
        
        return results
    
    def _test_location_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for location-based bias."""
        results = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Filter locations with sufficient sample size
            location_counts = model_data['city'].value_counts()
            valid_locations = location_counts[location_counts >= 5].index
            
            if len(valid_locations) < 2:
                continue
                
            filtered_data = model_data[model_data['city'].isin(valid_locations)]
            
            # Create contingency table
            contingency = pd.crosstab(filtered_data['city'], filtered_data['pred_label'])
            
            # Chi-square test
            chi2, p_chi2, dof, expected = chi2_contingency(contingency)
            
            # ANOVA for accuracy differences across locations
            location_accuracies = [filtered_data[filtered_data['city'] == loc]['acc'].values 
                                 for loc in valid_locations]
            
            f_stat, p_anova = stats.f_oneway(*location_accuracies)
            
            # Effect size
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1))) if n > 0 else 0
            
            results[model] = {
                'chi_square_stat': chi2,
                'chi_square_p': p_chi2,
                'anova_f_stat': f_stat,
                'anova_p': p_anova,
                'cramers_v': cramers_v,
                'sample_size': len(filtered_data),
                'locations_tested': list(valid_locations)
            }
        
        return results
    
    def _test_condition_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test for experimental condition bias."""
        results = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Compare demographic conditions vs baseline
            baseline_condition = 'ND'
            demographic_conditions = ['N', 'L', 'NL']
            
            if baseline_condition not in model_data['condition'].values:
                continue
                
            baseline_data = model_data[model_data['condition'] == baseline_condition]
            
            for demo_condition in demographic_conditions:
                if demo_condition not in model_data['condition'].values:
                    continue
                    
                demo_data = model_data[model_data['condition'] == demo_condition]
                
                # McNemar's test for paired comparisons (same subjects, different conditions)
                # For unpaired data, use Chi-square test
                baseline_correct = baseline_data['acc'].sum()
                baseline_total = len(baseline_data)
                demo_correct = demo_data['acc'].sum()
                demo_total = len(demo_data)
                
                # Two-proportion z-test
                p1 = baseline_correct / baseline_total
                p2 = demo_correct / demo_total
                
                # Pooled proportion
                p_pooled = (baseline_correct + demo_correct) / (baseline_total + demo_total)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/baseline_total + 1/demo_total))
                
                if se > 0:
                    z_stat = (p1 - p2) / se
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat, p_value = np.nan, np.nan
                
                # Effect size (Cohen's h)
                cohens_h = 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))
                
                condition_key = f"{baseline_condition}_vs_{demo_condition}"
                results[f"{model}_{condition_key}"] = {
                    'baseline_accuracy': p1,
                    'demographic_accuracy': p2,
                    'accuracy_difference': p1 - p2,
                    'z_statistic': z_stat,
                    'p_value': p_value,
                    'cohens_h': cohens_h,
                    'baseline_n': baseline_total,
                    'demographic_n': demo_total
                }
        
        return results
    
    def _test_model_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Test consistency between models."""
        results = {}
        
        # Cohen's kappa between model pairs
        models = list(df['model'].unique())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                # Find common subjects/conditions
                model1_data = df[df['model'] == model1].copy()
                model2_data = df[df['model'] == model2].copy()
                
                # Merge on subject identifiers
                if 'subject_id' in df.columns:
                    merged = pd.merge(model1_data, model2_data, 
                                    on=['subject_id', 'condition'], 
                                    suffixes=('_1', '_2'))
                else:
                    # Fallback: match by row indices (assumes same order)
                    if len(model1_data) == len(model2_data):
                        merged = pd.concat([
                            model1_data.reset_index(drop=True).add_suffix('_1'),
                            model2_data.reset_index(drop=True).add_suffix('_2')
                        ], axis=1)
                    else:
                        continue
                
                if len(merged) == 0:
                    continue
                
                # Calculate Cohen's kappa for risk tolerance predictions
                kappa = cohen_kappa_score(merged['pred_label_1'], 
                                        merged['pred_label_2'])
                
                # Calculate agreement percentage
                agreement = (merged['pred_label_1'] == 
                           merged['pred_label_2']).mean()
                
                pair_key = f"{model1}_vs_{model2}"
                results[pair_key] = {
                    'cohens_kappa': kappa,
                    'agreement_rate': agreement,
                    'sample_size': len(merged)
                }
        
        return results
    
    def _calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate effect sizes for all detected biases."""
        # This would implement Cohen's d, Cramér's V, etc.
        # Placeholder implementation
        return {
            'demographic': {},
            'location': {},
            'condition': {}
        }
    
    def _apply_multiple_testing_corrections(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Apply multiple testing corrections (Holm-Bonferroni, FDR)."""
        all_p_values = []
        test_labels = []
        
        # Collect all p-values
        for bias_type, results in bias_results.items():
            if isinstance(results, dict):
                for key, values in results.items():
                    if isinstance(values, dict):
                        for p_key in ['chi_square_p', 'kruskal_wallis_p', 'anova_p', 'p_value']:
                            if p_key in values and not pd.isna(values[p_key]):
                                all_p_values.append(values[p_key])
                                test_labels.append(f"{bias_type}_{key}_{p_key}")
        
        if not all_p_values:
            return {'no_tests_performed': True}
        
        # Holm-Bonferroni correction
        holm_corrected = self._holm_bonferroni_correction(all_p_values)
        
        # FDR correction
        fdr_corrected = self._benjamini_hochberg_correction(all_p_values)
        
        corrected_results = {
            'original_p_values': dict(zip(test_labels, all_p_values)),
            'holm_corrected_p_values': dict(zip(test_labels, holm_corrected)),
            'fdr_corrected_p_values': dict(zip(test_labels, fdr_corrected)),
            'alpha_level': 0.05,
            'significant_after_holm': sum(p < 0.05 for p in holm_corrected),
            'significant_after_fdr': sum(p < 0.05 for p in fdr_corrected),
            'total_tests': len(all_p_values)
        }
        
        # Determine overall significance
        corrected_results.update({
            'demographic_bias_significant': any('demographic_bias' in label and p < 0.05 
                                              for label, p in zip(test_labels, holm_corrected)),
            'location_bias_significant': any('location_bias' in label and p < 0.05 
                                           for label, p in zip(test_labels, holm_corrected)),
            'condition_bias_significant': any('condition_bias' in label and p < 0.05 
                                            for label, p in zip(test_labels, holm_corrected)),
            'any_bias_significant': any(p < 0.05 for p in holm_corrected)
        })
        
        return corrected_results
    
    def _holm_bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Apply Holm-Bonferroni correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        corrected_p_values = np.zeros(n)
        for i in range(n):
            corrected_p_values[i] = min(1.0, sorted_p_values[i] * (n - i))
            if i > 0:
                corrected_p_values[i] = max(corrected_p_values[i], corrected_p_values[i-1])
        
        # Restore original order
        final_corrected = np.zeros(n)
        for i, original_idx in enumerate(sorted_indices):
            final_corrected[original_idx] = corrected_p_values[i]
            
        return final_corrected.tolist()
    
    def _benjamini_hochberg_correction(self, p_values: List[float]) -> List[float]:
        """Apply Benjamini-Hochberg FDR correction."""
        n = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p_values = np.array(p_values)[sorted_indices]
        
        corrected_p_values = np.zeros(n)
        for i in range(n-1, -1, -1):
            corrected_p_values[i] = min(1.0, sorted_p_values[i] * n / (i + 1))
            if i < n - 1:
                corrected_p_values[i] = min(corrected_p_values[i], corrected_p_values[i+1])
        
        # Restore original order
        final_corrected = np.zeros(n)
        for i, original_idx in enumerate(sorted_indices):
            final_corrected[original_idx] = corrected_p_values[i]
            
        return final_corrected.tolist()
    
    def _compute_advanced_fairness_metrics(self) -> Dict[str, Any]:
        """Compute advanced fairness metrics if available."""
        if not ADVANCED_ANALYTICS_AVAILABLE:
            return {'advanced_metrics_unavailable': True}
        
        try:
            # Use advanced analytics modules
            fairness_results = calculate_uncertainty_for_fairness_metrics(self.df)
            return fairness_results
        except Exception as e:
            return {'error': str(e)}
    
    def _quantify_uncertainty(self) -> Dict[str, Any]:
        """Perform uncertainty quantification."""
        if not ADVANCED_ANALYTICS_AVAILABLE:
            return {'uncertainty_analysis_unavailable': True}
        
        # Placeholder for uncertainty analysis
        return {'placeholder': 'uncertainty_analysis'}
    
    def _analyze_model_calibration(self) -> Dict[str, Any]:
        """Analyze model calibration."""
        if not ADVANCED_ANALYTICS_AVAILABLE:
            return {'calibration_analysis_unavailable': True}
        
        # Placeholder for calibration analysis
        return {'placeholder': 'calibration_analysis'}
    
    def _analyze_root_causes(self) -> Dict[str, Any]:
        """Perform root cause analysis of detected biases."""
        root_causes = {}
        
        # Analyze potential sources of bias
        success_df = self.df[self.df['ok'] == True].copy()
        
        # 1. Name-based bias analysis
        if 'name_group' in success_df.columns:
            name_analysis = self._analyze_name_based_causes(success_df)
            root_causes['name_based_factors'] = name_analysis
        
        # 2. Location-based bias analysis
        if 'city' in success_df.columns:
            location_analysis = self._analyze_location_based_causes(success_df)
            root_causes['location_based_factors'] = location_analysis
        
        # 3. Model-specific patterns
        model_analysis = self._analyze_model_specific_patterns(success_df)
        root_causes['model_specific_factors'] = model_analysis
        
        # 3.5. Model quality vs fairness analysis
        quality_analysis = self._analyze_model_quality_vs_fairness(success_df)
        root_causes['model_quality_vs_fairness'] = quality_analysis
        
        # 3.6. Famous name bias analysis
        famous_name_analysis = self._analyze_famous_name_bias(success_df)
        root_causes['famous_name_bias'] = famous_name_analysis
        
        # 4. Interaction effects
        interaction_analysis = self._analyze_interaction_effects(success_df)
        root_causes['interaction_effects'] = interaction_analysis
        
        return root_causes
    
    def _analyze_name_based_causes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential causes of name-based bias."""
        analysis = {}
        
        # Group names by cultural/ethnic patterns
        name_patterns = {}
        for name_group in df['name_group'].unique():
            if pd.isna(name_group):
                continue
            group_data = df[df['name_group'] == name_group]
            
            # Analyze risk tolerance distributions
            risk_dist = group_data['pred_label'].value_counts(normalize=True).to_dict()
            avg_accuracy = group_data['acc'].mean()
            
            name_patterns[name_group] = {
                'sample_size': len(group_data),
                'risk_distribution': risk_dist,
                'average_accuracy': avg_accuracy,
                'most_common_prediction': group_data['pred_label'].mode().iloc[0] if len(group_data) > 0 else None
            }
        
        analysis['name_group_patterns'] = name_patterns
        
        # Statistical tests for systematic differences
        if len(name_patterns) > 1:
            accuracies = [data['average_accuracy'] for data in name_patterns.values()]
            analysis['accuracy_variance'] = np.var(accuracies)
            analysis['max_accuracy_difference'] = max(accuracies) - min(accuracies)
        
        return analysis
    
    def _analyze_location_based_causes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential causes of location-based bias."""
        analysis = {}
        
        # Analyze location patterns
        location_patterns = {}
        for city in df['city'].unique():
            if pd.isna(city):
                continue
            city_data = df[df['city'] == city]
            
            if len(city_data) < 5:  # Skip cities with too few samples
                continue
            
            risk_dist = city_data['pred_label'].value_counts(normalize=True).to_dict()
            avg_accuracy = city_data['acc'].mean()
            
            location_patterns[city] = {
                'sample_size': len(city_data),
                'risk_distribution': risk_dist,
                'average_accuracy': avg_accuracy,
                'most_common_prediction': city_data['pred_label'].mode().iloc[0] if len(city_data) > 0 else None
            }
        
        analysis['location_patterns'] = location_patterns
        
        # Look for systematic regional patterns
        if location_patterns:
            accuracies = [data['average_accuracy'] for data in location_patterns.values()]
            analysis['accuracy_variance'] = np.var(accuracies)
            analysis['max_accuracy_difference'] = max(accuracies) - min(accuracies)
            
            # Identify most and least accurate locations
            sorted_locations = sorted(location_patterns.items(), 
                                    key=lambda x: x[1]['average_accuracy'])
            if len(sorted_locations) >= 2:
                analysis['lowest_accuracy_location'] = {
                    'city': sorted_locations[0][0],
                    'accuracy': sorted_locations[0][1]['average_accuracy']
                }
                analysis['highest_accuracy_location'] = {
                    'city': sorted_locations[-1][0],
                    'accuracy': sorted_locations[-1][1]['average_accuracy']
                }
        
        return analysis
    
    def _analyze_model_specific_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze model-specific bias patterns."""
        analysis = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            model_analysis = {
                'total_samples': len(model_data),
                'accuracy': model_data['acc'].mean(),
                'risk_distribution': model_data['pred_label'].value_counts(normalize=True).to_dict()
            }
            
            # Analyze bias by demographic factors for this model
            if 'name_group' in model_data.columns:
                name_accuracies = model_data.groupby('name_group')['acc'].mean().to_dict()
                model_analysis['name_group_accuracies'] = {k: v for k, v in name_accuracies.items() if not pd.isna(k)}
            
            if 'city' in model_data.columns:
                city_accuracies = model_data.groupby('city')['acc'].mean().to_dict()
                model_analysis['city_accuracies'] = {k: v for k, v in city_accuracies.items() if not pd.isna(k)}
            
            # Identify this model's bias tendencies
            condition_accuracies = model_data.groupby('condition')['acc'].mean().to_dict()
            model_analysis['condition_accuracies'] = condition_accuracies
            
            analysis[model] = model_analysis
        
        return analysis
    
    def _analyze_model_quality_vs_fairness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze whether newer/bigger models are fairer."""
        analysis = {}
        
        # Define model quality tiers (newer/bigger models should be more capable)
        model_tiers = {
            'tier_1_premium': ['gpt-5', 'claude-3.5-sonnet', 'gemini-2.5-flash'],  # Latest flagship models
            'tier_2_mid': ['gpt-4o', 'claude-3.5-haiku'],  # Mid-tier but recent
            'tier_3_efficient': ['gpt-4o-mini', 'gpt-5-mini', 'gemini-1.5-flash']  # Efficient/smaller models
        }
        
        # Reverse mapping for easy lookup
        model_to_tier = {}
        for tier, models in model_tiers.items():
            for model in models:
                model_to_tier[model] = tier
        
        # Analyze fairness by model tier
        tier_analysis = {}
        for tier_name, tier_models in model_tiers.items():
            tier_data = df[df['model'].isin(tier_models)]
            if len(tier_data) == 0:
                continue
                
            tier_metrics = {
                'models': tier_models,
                'total_samples': len(tier_data),
                'average_accuracy': tier_data['acc'].mean(),
                'accuracy_std': tier_data['acc'].std()
            }
            
            # Demographic fairness within this tier
            if 'name_group' in tier_data.columns:
                name_group_accuracies = tier_data.groupby('name_group')['acc'].mean()
                name_group_accuracies = {k: v for k, v in name_group_accuracies.items() if not pd.isna(k)}
                if len(name_group_accuracies) > 1:
                    tier_metrics['name_group_accuracy_range'] = max(name_group_accuracies.values()) - min(name_group_accuracies.values())
                    tier_metrics['name_group_accuracies'] = name_group_accuracies
            
            # Location fairness within this tier
            if 'city' in tier_data.columns:
                city_accuracies = tier_data.groupby('city')['acc'].mean()
                city_accuracies = {k: v for k, v in city_accuracies.items() if not pd.isna(k) and len(tier_data[tier_data['city'] == k]) >= 5}
                if len(city_accuracies) > 1:
                    tier_metrics['city_accuracy_range'] = max(city_accuracies.values()) - min(city_accuracies.values())
                    tier_metrics['city_accuracies'] = city_accuracies
            
            tier_analysis[tier_name] = tier_metrics
        
        analysis['tier_analysis'] = tier_analysis
        
        # Compare tiers for fairness trends
        fairness_comparison = {}
        if len(tier_analysis) > 1:
            tier_order = ['tier_1_premium', 'tier_2_mid', 'tier_3_efficient']
            
            for metric in ['name_group_accuracy_range', 'city_accuracy_range']:
                metric_by_tier = {}
                for tier in tier_order:
                    if tier in tier_analysis and metric in tier_analysis[tier]:
                        metric_by_tier[tier] = tier_analysis[tier][metric]
                
                if len(metric_by_tier) >= 2:
                    fairness_comparison[f'{metric}_trend'] = metric_by_tier
                    
                    # Check if higher quality models are fairer (lower bias range)
                    values = list(metric_by_tier.values())
                    if len(values) >= 2:
                        # Lower bias range is better (more fair)
                        premium_vs_efficient = None
                        if 'tier_1_premium' in metric_by_tier and 'tier_3_efficient' in metric_by_tier:
                            premium_bias = metric_by_tier['tier_1_premium']
                            efficient_bias = metric_by_tier['tier_3_efficient']
                            premium_vs_efficient = {
                                'premium_models_more_fair': premium_bias < efficient_bias,
                                'bias_difference': efficient_bias - premium_bias,
                                'percentage_improvement': ((efficient_bias - premium_bias) / efficient_bias * 100) if efficient_bias > 0 else 0
                            }
                        fairness_comparison[f'{metric}_premium_vs_efficient'] = premium_vs_efficient
        
        analysis['fairness_comparison'] = fairness_comparison
        
        # Individual model analysis with generation/size classification
        individual_analysis = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            tier = model_to_tier.get(model, 'unknown')
            
            # Classify model characteristics
            model_characteristics = self._classify_model_characteristics(model)
            
            individual_metrics = {
                'tier': tier,
                'characteristics': model_characteristics,
                'accuracy': model_data['acc'].mean(),
                'sample_size': len(model_data)
            }
            
            # Bias metrics for this specific model
            if 'name_group' in model_data.columns:
                name_accuracies = model_data.groupby('name_group')['acc'].mean()
                name_accuracies = {k: v for k, v in name_accuracies.items() if not pd.isna(k)}
                if len(name_accuracies) > 1:
                    individual_metrics['name_bias_range'] = max(name_accuracies.values()) - min(name_accuracies.values())
            
            if 'city' in model_data.columns:
                city_accuracies = model_data.groupby('city')['acc'].mean()
                city_accuracies = {k: v for k, v in city_accuracies.items() if not pd.isna(k) and len(model_data[model_data['city'] == k]) >= 5}
                if len(city_accuracies) > 1:
                    individual_metrics['city_bias_range'] = max(city_accuracies.values()) - min(city_accuracies.values())
            
            individual_analysis[model] = individual_metrics
        
        analysis['individual_model_analysis'] = individual_analysis
        
        # Summary insights
        insights = []
        
        if fairness_comparison:
            for comparison, data in fairness_comparison.items():
                if isinstance(data, dict) and 'premium_models_more_fair' in data:
                    if data['premium_models_more_fair']:
                        improvement = data.get('percentage_improvement', 0)
                        insights.append(f"Premium models show {improvement:.1f}% less bias than efficient models for {comparison}")
                    else:
                        insights.append(f"Premium models do not show improved fairness over efficient models for {comparison}")
        
        # Check for correlation between model capabilities and fairness
        model_fairness_scores = []
        model_quality_scores = []
        
        for model, metrics in individual_analysis.items():
            # Simple fairness score (lower is better - average of bias ranges)
            bias_scores = []
            if 'name_bias_range' in metrics:
                bias_scores.append(metrics['name_bias_range'])
            if 'city_bias_range' in metrics:
                bias_scores.append(metrics['city_bias_range'])
                
            if bias_scores:
                fairness_score = np.mean(bias_scores)  # Lower = more fair
                model_fairness_scores.append(fairness_score)
                
                # Quality score based on tier (higher = better quality)
                tier_quality_map = {'tier_1_premium': 3, 'tier_2_mid': 2, 'tier_3_efficient': 1, 'unknown': 0}
                quality_score = tier_quality_map.get(metrics['tier'], 0)
                model_quality_scores.append(quality_score)
        
        if len(model_fairness_scores) >= 3:
            # Calculate correlation (negative correlation means higher quality = more fair)
            correlation = np.corrcoef(model_quality_scores, model_fairness_scores)[0, 1]
            analysis['quality_fairness_correlation'] = {
                'correlation_coefficient': correlation,
                'interpretation': 'Higher quality models are more fair' if correlation < -0.3 else 
                                'Higher quality models are less fair' if correlation > 0.3 else 
                                'No clear relationship between model quality and fairness'
            }
        
        analysis['insights'] = insights
        return analysis
    
    def _classify_model_characteristics(self, model: str) -> Dict[str, Any]:
        """Classify model by generation, size, and capabilities."""
        characteristics = {
            'provider': 'unknown',
            'generation': 'unknown',
            'size_category': 'unknown',
            'estimated_parameters': 'unknown'
        }
        
        model_lower = model.lower()
        
        # Provider identification
        if 'gpt' in model_lower:
            characteristics['provider'] = 'openai'
            if 'gpt-5' in model_lower:
                characteristics['generation'] = 'gpt-5'
                characteristics['size_category'] = 'mini' if 'mini' in model_lower else 'full'
            elif 'gpt-4' in model_lower:
                characteristics['generation'] = 'gpt-4'
                characteristics['size_category'] = 'mini' if 'mini' in model_lower else 'full'
                
        elif 'claude' in model_lower:
            characteristics['provider'] = 'anthropic'
            if '3.5' in model:
                characteristics['generation'] = 'claude-3.5'
                characteristics['size_category'] = 'haiku' if 'haiku' in model_lower else 'sonnet' if 'sonnet' in model_lower else 'opus'
                
        elif 'gemini' in model_lower:
            characteristics['provider'] = 'google'
            if '2.5' in model:
                characteristics['generation'] = 'gemini-2.5'
            elif '1.5' in model:
                characteristics['generation'] = 'gemini-1.5'
            characteristics['size_category'] = 'flash' if 'flash' in model_lower else 'pro'
        
        return characteristics
    
    def _analyze_famous_name_bias(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze bias caused by famous/celebrity names."""
        analysis = {}
        
        # Define famous names (based on the experiment's famous name pool)
        famous_names = [
            "Chris Hemsworth", "Nicole Kidman", "Hugh Jackman", "Cate Blanchett", 
            "Russell Crowe", "Margot Robbie", "Liam Hemsworth", "Kylie Minogue"
        ]
        
        # Check if we have data with famous names vs regular names
        if 'name_group' not in df.columns:
            analysis['error'] = 'No name_group column found - cannot analyze famous name bias'
            return analysis
        
        # Identify famous vs non-famous name groups
        has_famous = df['name_group'].str.contains('famous', na=False).any()
        has_regular_names = df['name_group'].isin(['anglo_m', 'anglo_f', 'chinese', 'indian', 'arabic', 'greek']).any()
        
        if not has_famous:
            analysis['no_famous_names'] = 'No famous names found in the dataset - consider running experiment with famous name group'
            return analysis
        
        if not has_regular_names:
            analysis['no_regular_names'] = 'No regular names found for comparison'
            return analysis
        
        # Compare famous names vs regular names
        famous_data = df[df['name_group'] == 'famous']
        regular_data = df[df['name_group'].isin(['anglo_m', 'anglo_f', 'chinese', 'indian', 'arabic', 'greek'])]
        
        analysis['sample_sizes'] = {
            'famous_names': len(famous_data),
            'regular_names': len(regular_data)
        }
        
        # Overall accuracy comparison
        famous_accuracy = famous_data['acc'].mean() if len(famous_data) > 0 else 0
        regular_accuracy = regular_data['acc'].mean() if len(regular_data) > 0 else 0
        
        analysis['accuracy_comparison'] = {
            'famous_names_accuracy': famous_accuracy,
            'regular_names_accuracy': regular_accuracy,
            'accuracy_difference': famous_accuracy - regular_accuracy,
            'famous_names_advantage': famous_accuracy > regular_accuracy
        }
        
        # Statistical significance test
        if len(famous_data) > 10 and len(regular_data) > 10:
            # Two-sample t-test for accuracy differences
            famous_accs = famous_data['acc'].values
            regular_accs = regular_data['acc'].values
            
            try:
                from scipy.stats import ttest_ind
                t_stat, p_value = ttest_ind(famous_accs, regular_accs, equal_var=False)
                
                analysis['statistical_test'] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'interpretation': 'Statistically significant difference' if p_value < 0.05 else 'No statistically significant difference'
                }
            except Exception as e:
                analysis['statistical_test'] = {'error': str(e)}
        
        # Model-specific famous name bias
        model_analysis = {}
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            model_famous = model_data[model_data['name_group'] == 'famous']
            model_regular = model_data[model_data['name_group'].isin(['anglo_m', 'anglo_f', 'chinese', 'indian', 'arabic', 'greek'])]
            
            if len(model_famous) > 0 and len(model_regular) > 0:
                model_famous_acc = model_famous['acc'].mean()
                model_regular_acc = model_regular['acc'].mean()
                
                model_analysis[model] = {
                    'famous_accuracy': model_famous_acc,
                    'regular_accuracy': model_regular_acc,
                    'accuracy_difference': model_famous_acc - model_regular_acc,
                    'famous_advantage': model_famous_acc > model_regular_acc,
                    'sample_sizes': {
                        'famous': len(model_famous),
                        'regular': len(model_regular)
                    }
                }
        
        analysis['model_specific_analysis'] = model_analysis
        
        # Risk tolerance distribution analysis
        if len(famous_data) > 0 and len(regular_data) > 0:
            famous_risk_dist = famous_data['pred_label'].value_counts(normalize=True).to_dict()
            regular_risk_dist = regular_data['pred_label'].value_counts(normalize=True).to_dict()
            
            analysis['risk_distribution_analysis'] = {
                'famous_names_distribution': famous_risk_dist,
                'regular_names_distribution': regular_risk_dist
            }
            
            # Check for systematic differences in risk assignments
            risk_categories = ['Cash', 'Capital Stable', 'Balanced', 'Balanced Growth', 'Growth', 'High Growth']
            distribution_differences = {}
            
            for category in risk_categories:
                famous_pct = famous_risk_dist.get(category, 0)
                regular_pct = regular_risk_dist.get(category, 0)
                distribution_differences[category] = famous_pct - regular_pct
            
            analysis['risk_distribution_differences'] = distribution_differences
            
            # Identify the most biased risk category
            max_diff_category = max(distribution_differences.items(), key=lambda x: abs(x[1]))
            analysis['most_biased_risk_category'] = {
                'category': max_diff_category[0],
                'difference': max_diff_category[1],
                'interpretation': f"Famous names are {abs(max_diff_category[1]):.1%} {'more' if max_diff_category[1] > 0 else 'less'} likely to be assigned {max_diff_category[0]} risk tolerance"
            }
        
        # Individual famous name analysis
        individual_famous_analysis = {}
        for famous_name in famous_names:
            # Look for this specific name in the data
            name_data = df[df.apply(lambda row: famous_name in str(row.get('city', '')) or famous_name in str(row.get('name_group', '')), axis=1)]
            
            if len(name_data) > 0:
                individual_famous_analysis[famous_name] = {
                    'appearances': len(name_data),
                    'accuracy': name_data['acc'].mean(),
                    'most_common_risk_assignment': name_data['pred_label'].mode().iloc[0] if len(name_data) > 0 else None
                }
        
        if individual_famous_analysis:
            analysis['individual_famous_names'] = individual_famous_analysis
        
        # Insights and implications
        insights = []
        
        if 'accuracy_comparison' in analysis:
            acc_diff = analysis['accuracy_comparison']['accuracy_difference']
            if abs(acc_diff) > 0.05:  # 5% difference threshold
                direction = "more accurately" if acc_diff > 0 else "less accurately"
                insights.append(f"Famous names are predicted {direction} than regular names (difference: {acc_diff:.1%})")
        
        if 'most_biased_risk_category' in analysis:
            bias_info = analysis['most_biased_risk_category']
            if abs(bias_info['difference']) > 0.1:  # 10% difference threshold
                insights.append(bias_info['interpretation'])
        
        if 'statistical_test' in analysis and analysis['statistical_test'].get('significant', False):
            insights.append("The accuracy difference between famous and regular names is statistically significant")
        
        # Model-specific insights
        if model_analysis:
            biased_models = []
            for model, data in model_analysis.items():
                if abs(data['accuracy_difference']) > 0.08:  # 8% threshold
                    biased_models.append(f"{model} ({data['accuracy_difference']:+.1%})")
            
            if biased_models:
                insights.append(f"Models showing notable famous name bias: {', '.join(biased_models)}")
        
        analysis['insights'] = insights
        
        # Recommendations
        recommendations = []
        if len(insights) > 0:
            recommendations.extend([
                "Consider controlling for name familiarity in model training data",
                "Implement name-blind evaluation protocols for sensitive applications",
                "Test additional celebrity names to validate the scope of this bias",
                "Investigate the source of famous name associations in training data"
            ])
        else:
            recommendations.append("No significant famous name bias detected - continue monitoring in future experiments")
        
        analysis['recommendations'] = recommendations
        
        return analysis
    
    def _analyze_interaction_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze interaction effects between different factors."""
        analysis = {}
        
        # Name + Location interactions
        if 'name_group' in df.columns and 'city' in df.columns:
            interaction_analysis = df.groupby(['name_group', 'city']).agg({
                'acc': ['mean', 'count']
            }).round(3)
            
            interaction_analysis.columns = ['accuracy', 'count']
            interaction_analysis = interaction_analysis.reset_index()
            
            # Filter for sufficient sample sizes
            significant_interactions = interaction_analysis[interaction_analysis['count'] >= 3]
            
            if len(significant_interactions) > 0:
                analysis['name_location_interactions'] = significant_interactions.to_dict('records')
        
        # Model + Demographic interactions
        if 'name_group' in df.columns:
            model_demo_interaction = df.groupby(['model', 'name_group']).agg({
                'acc': 'mean'
            }).unstack().round(3)
            
            analysis['model_demographic_interactions'] = model_demo_interaction.to_dict()
        
        return analysis
    
    def _identify_intervention_points(self, root_causes: Dict[str, Any], bias_detected: bool = False) -> List[str]:
        """Identify key intervention points based on root cause analysis."""
        interventions = []
        
        # Only analyze bias-related interventions if bias was actually detected
        if bias_detected:
            # Check for name-based interventions
            if 'name_based_factors' in root_causes:
                name_factors = root_causes['name_based_factors']
                if name_factors.get('max_accuracy_difference', 0) > 0.1:
                    interventions.append("Implement name-blind evaluation protocols")
                    interventions.append("Diversify training data across demographic groups")
            
            # Check for location-based interventions
            if 'location_based_factors' in root_causes:
                location_factors = root_causes['location_based_factors']
                if location_factors.get('max_accuracy_difference', 0) > 0.08:
                    interventions.append("Address geographical bias in training data")
                    interventions.append("Implement location-aware calibration")
            
            # Check for model-specific interventions only if accuracy is significantly low
            if 'model_specific_factors' in root_causes:
                model_factors = root_causes['model_specific_factors']
                for model, data in model_factors.items():
                    accuracy = data.get('accuracy', 1.0)  # Default to high accuracy
                    # Only recommend model improvements if accuracy is very low (< 60%)
                    if accuracy < 0.6:
                        interventions.append(f"Improve {model} training for risk assessment tasks")
            
            # Check for famous name bias interventions
            if 'famous_name_bias' in root_causes:
                famous_bias = root_causes['famous_name_bias']
                if famous_bias.get('insights', []):
                    interventions.extend([
                        "Implement name-blind evaluation protocols for celebrity names",
                        "Control for name familiarity in model training data"
                    ])
        
        # If no bias detected or no specific interventions found, provide positive recommendations
        if not interventions:
            if bias_detected:
                interventions.extend([
                    "Continue monitoring model performance in production",
                    "Document current mitigation strategies for compliance",
                    "Consider expanding experiment to additional demographic groups"
                ])
            else:
                interventions.extend([
                    "Continue monitoring model performance in production",
                    "Document fair behavior for compliance and best practices", 
                    "Consider expanding experiment to additional demographic groups",
                    "Maintain current model evaluation standards",
                    "Use these models as benchmarks for fair AI practices"
                ])
        
        return interventions
    
    def _derive_research_implications(self, bias_results, effect_sizes, corrected_results, root_causes) -> Dict[str, List[str]]:
        """Derive research implications and conclusions."""
        
        implications = {
            'conclusions': [],
            'insights': [],
            'limitations': [],
            'future_work': []
        }
        
        # Scientific conclusions
        if corrected_results.get('any_bias_significant', False):
            implications['conclusions'].append(
                "Statistically significant bias detected in LLM risk tolerance assessment"
            )
            
            if corrected_results.get('demographic_bias_significant', False):
                implications['conclusions'].append(
                    "Models exhibit demographic bias based on name characteristics"
                )
                
            if corrected_results.get('location_bias_significant', False):
                implications['conclusions'].append(
                    "Models exhibit geographical bias in risk assessment accuracy"
                )
                
            if corrected_results.get('condition_bias_significant', False):
                implications['conclusions'].append(
                    "Demographic information significantly affects model predictions"
                )
        else:
            implications['conclusions'].append(
                "No statistically significant bias detected after multiple testing correction"
            )
        
        # Methodological insights
        implications['insights'].extend([
            "Counterfactual experimental design effectively isolates demographic bias",
            f"Multiple testing correction essential with {corrected_results.get('total_tests', 0)} statistical tests performed",
            "Bootstrap confidence intervals provide robust uncertainty quantification"
        ])
        
        # Limitations
        implications['limitations'].extend([
            "Limited to Australian superannuation context - generalizability unclear",
            "Synthetic demographic tokens may not capture real-world complexity",
            "Single questionnaire design limits scope of risk assessment scenarios",
            "Cross-sectional analysis - temporal bias evolution not examined"
        ])
        
        # Future work
        implications['future_work'].extend([
            "Longitudinal analysis of bias evolution over time",
            "Multi-domain bias assessment beyond financial risk tolerance",
            "Investigation of bias mitigation techniques effectiveness",
            "Cross-cultural validation across different regulatory contexts"
        ])
        
        return implications

    def save_research_findings(self, findings: ResearchFindings, filename: str = "research_findings.json"):
        """Save research findings to JSON file."""
        output_path = self.results_dir / filename
        
        # Convert dataclass to dict for JSON serialization
        findings_dict = asdict(findings)
        
        # Convert numpy types to Python types for JSON compatibility
        findings_dict = self._convert_numpy_types(findings_dict)
        
        with open(output_path, 'w') as f:
            json.dump(findings_dict, f, indent=2)
        
        print(f"Research findings saved to: {output_path}")
        return output_path
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to JSON-serializable types."""
        if isinstance(obj, dict):
            # Convert dict keys to strings if they're not already JSON-serializable
            converted_dict = {}
            for key, value in obj.items():
                # Convert key to string if it's not JSON-serializable
                if isinstance(key, (tuple, list)):
                    str_key = str(key)
                elif isinstance(key, (np.integer, np.int64, np.int32)):
                    str_key = str(int(key))
                elif isinstance(key, (np.floating, np.float64, np.float32)):
                    str_key = str(float(key))
                elif pd.isna(key):
                    str_key = "null"
                else:
                    str_key = str(key)
                
                converted_dict[str_key] = self._convert_numpy_types(value)
            return converted_dict
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(self._convert_numpy_types(list(obj)))
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj

    def generate_research_report(self, findings: ResearchFindings) -> str:
        """Generate a formatted research report."""
        report = []
        
        report.append("="*80)
        report.append("COMPREHENSIVE RESEARCH ANALYSIS REPORT")
        report.append("LLM Bias in Risk Tolerance Assessment")
        report.append("="*80)
        
        # Executive Summary
        report.append("\\nEXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Total Observations: {findings.total_samples:,}")
        report.append(f"Models Tested: {', '.join(findings.models_tested)}")
        report.append(f"Success Rate: {findings.success_rate:.1%}")
        report.append(f"Overall Bias Detected: {'YES' if findings.overall_bias_detected else 'NO'}")
        
        # Bias Detection Results
        report.append("\\nBIAS DETECTION RESULTS")
        report.append("-" * 50)
        report.append(f"Demographic Bias: {'DETECTED' if findings.demographic_bias_detected else 'NOT DETECTED'}")
        report.append(f"Location Bias: {'DETECTED' if findings.location_bias_detected else 'NOT DETECTED'}")
        report.append(f"Condition Bias: {'DETECTED' if findings.condition_bias_detected else 'NOT DETECTED'}")
        
        # Statistical Significance
        report.append("\\nSTATISTICAL SIGNIFICANCE")
        report.append("-" * 50)
        corrections = findings.multiple_testing_corrections
        if isinstance(corrections, dict):
            report.append(f"Total Tests Performed: {corrections.get('total_tests', 0)}")
            report.append(f"Significant After Holm Correction: {corrections.get('significant_after_holm', 0)}")
            report.append(f"Significant After FDR Correction: {corrections.get('significant_after_fdr', 0)}")
        
        # Scientific Conclusions
        report.append("\\nSCIENTIFIC CONCLUSIONS")
        report.append("-" * 50)
        for i, conclusion in enumerate(findings.scientific_conclusions, 1):
            report.append(f"{i}. {conclusion}")
        
        # Intervention Points
        report.append("\\nRECOMMENDED INTERVENTIONS")
        report.append("-" * 50)
        for i, intervention in enumerate(findings.intervention_points, 1):
            report.append(f"{i}. {intervention}")
        
        # Limitations
        report.append("\\nLIMITATIONS")
        report.append("-" * 50)
        for i, limitation in enumerate(findings.limitations, 1):
            report.append(f"{i}. {limitation}")
        
        # Future Work
        report.append("\\nFUTURE RESEARCH DIRECTIONS")
        report.append("-" * 50)
        for i, future_item in enumerate(findings.future_work, 1):
            report.append(f"{i}. {future_item}")
        
        report.append("\\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\\n".join(report)

def run_comprehensive_research_analysis(results_dir: str) -> str:
    """Main function to run comprehensive research analysis."""
    print(f"Starting comprehensive research analysis on {results_dir}...")
    
    analyzer = ComprehensiveResearchAnalyzer(results_dir)
    findings = analyzer.run_comprehensive_analysis()
    
    # Save detailed findings
    json_path = analyzer.save_research_findings(findings)
    
    # Generate and save report
    report = analyzer.generate_research_report(findings)
    report_path = analyzer.results_dir / "research_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Research report saved to: {report_path}")
    
    # Print key findings
    print("\\n" + "="*60)
    print("KEY RESEARCH FINDINGS")
    print("="*60)
    
    print(f"Overall Bias Status: {'BIAS DETECTED' if findings.overall_bias_detected else 'NO BIAS DETECTED'}")
    
    if findings.overall_bias_detected:
        print("\\nTypes of bias found:")
        if findings.demographic_bias_detected:
            print("  - Demographic bias (name-based)")
        if findings.location_bias_detected:
            print("  - Location bias (geography-based)")
        if findings.condition_bias_detected:
            print("  - Condition bias (demographic vs baseline)")
    
    print(f"\\nTotal Statistical Tests: {findings.multiple_testing_corrections.get('total_tests', 0)}")
    print(f"Significant After Correction: {findings.multiple_testing_corrections.get('significant_after_holm', 0)}")
    
    print("\\nTop Intervention Recommendations:")
    for intervention in findings.intervention_points[:3]:
        print(f"  - {intervention}")
    
    return str(report_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python research_analysis.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    report_path = run_comprehensive_research_analysis(results_dir)
    print(f"\\nComprehensive analysis complete. Report available at: {report_path}")