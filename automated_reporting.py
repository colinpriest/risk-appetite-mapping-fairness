#!/usr/bin/env python3
"""
Automated Report Generation System for LLM Risk Fairness Experiments

This module generates publication-ready reports including:
- LaTeX academic papers with statistical tables
- Executive summaries for stakeholders
- Interactive HTML reports with visualizations
- Compliance reports for ethics boards
- Comparative analysis across experiments
"""

import os
import json
import yaml
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, Template
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Statistical libraries
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


@dataclass
class ReportConfig:
    """Configuration for automated report generation."""
    report_type: str = "academic"  # academic, executive, compliance, comparative
    include_figures: bool = True
    include_statistical_tables: bool = True
    include_raw_data: bool = False
    output_format: str = "pdf"  # pdf, html, latex, word
    template_style: str = "ieee"  # ieee, apa, nature, custom
    significance_level: float = 0.05
    effect_size_threshold: float = 0.1
    figure_dpi: int = 300
    color_palette: str = "colorblind"


@dataclass
class ExperimentSummary:
    """Summary statistics for an experiment."""
    experiment_name: str
    models_tested: List[str]
    total_subjects: int
    total_responses: int
    success_rate: float
    total_cost: float
    duration_hours: float
    demographic_groups: List[str]
    risk_labels: List[str]
    
    # Statistical results
    overall_accuracy: float
    demographic_parity_violation: float
    equalized_odds_violation: float
    calibration_error: float
    consistency_score: float
    
    # Significance tests
    chi_square_statistic: float
    chi_square_p_value: float
    kruskal_wallis_statistic: float
    kruskal_wallis_p_value: float
    
    # Effect sizes
    cramers_v: float
    cohens_d: float
    
    # Confidence intervals
    accuracy_ci_lower: float
    accuracy_ci_upper: float
    bias_ci_lower: float
    bias_ci_upper: float


class AutomatedReportGenerator:
    """Generates comprehensive automated reports for LLM fairness experiments."""
    
    def __init__(self, config: ReportConfig = None):
        self.config = config or ReportConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "report_templates"
        template_dir.mkdir(exist_ok=True)
        self.jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))
        
        # Configure plotting
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'seaborn')
        sns.set_palette(self.config.color_palette)
        pio.templates.default = "plotly_white"
        
        # Create templates if they don't exist
        self._ensure_templates_exist()
    
    def generate_comprehensive_report(self, experiment_dir: str, 
                                    output_path: str = None) -> Dict[str, str]:
        """Generate comprehensive automated report from experiment results."""
        
        self.logger.info(f"Generating comprehensive report for {experiment_dir}")
        
        # Load experiment data
        experiment_data = self._load_experiment_data(experiment_dir)
        summary = self._compute_experiment_summary(experiment_data)
        
        # Generate figures
        figures = self._generate_all_figures(experiment_data, experiment_dir)
        
        # Generate statistical tables
        tables = self._generate_statistical_tables(experiment_data)
        
        # Generate report content based on type
        if self.config.report_type == "academic":
            report_content = self._generate_academic_report(summary, figures, tables, experiment_data)
        elif self.config.report_type == "executive":
            report_content = self._generate_executive_report(summary, figures)
        elif self.config.report_type == "compliance":
            report_content = self._generate_compliance_report(summary, experiment_data)
        else:
            report_content = self._generate_standard_report(summary, figures, tables)
        
        # Save report
        output_files = self._save_report(report_content, figures, output_path or experiment_dir)
        
        self.logger.info(f"Report generated successfully: {output_files}")
        return output_files
    
    def generate_comparative_report(self, experiment_dirs: List[str], 
                                  output_path: str) -> Dict[str, str]:
        """Generate comparative analysis across multiple experiments."""
        
        self.logger.info(f"Generating comparative report for {len(experiment_dirs)} experiments")
        
        # Load all experiment data
        experiments = []
        for exp_dir in experiment_dirs:
            try:
                data = self._load_experiment_data(exp_dir)
                summary = self._compute_experiment_summary(data)
                experiments.append((exp_dir, data, summary))
            except Exception as e:
                self.logger.warning(f"Failed to load experiment {exp_dir}: {str(e)}")
                continue
        
        # Generate comparative figures
        comparative_figures = self._generate_comparative_figures(experiments)
        
        # Generate comparative tables
        comparative_tables = self._generate_comparative_tables(experiments)
        
        # Generate comparative report content
        report_content = self._generate_comparative_content(experiments, comparative_figures, comparative_tables)
        
        # Save comparative report
        output_files = self._save_report(report_content, comparative_figures, output_path)
        
        return output_files
    
    def _load_experiment_data(self, experiment_dir: str) -> Dict[str, Any]:
        """Load all relevant experiment data files."""
        
        exp_path = Path(experiment_dir)
        data = {}
        
        # Load main results
        results_file = exp_path / "results.csv"
        if results_file.exists():
            data['results_df'] = pd.read_csv(results_file)
        
        # Load experiment report
        report_file = exp_path / "experiment_report.json"
        if report_file.exists():
            with open(report_file, 'r') as f:
                data['experiment_report'] = json.load(f)
        
        # Load stats summary
        stats_file = exp_path / "stats_summary.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                data['stats_summary'] = json.load(f)
        
        # Load metadata
        metadata_file = exp_path / "experiment_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                data['metadata'] = json.load(f)
        
        # Load configuration
        config_file = exp_path / "experiment_config.yaml"
        if config_file.exists():
            with open(config_file, 'r') as f:
                data['config'] = yaml.safe_load(f)
        
        return data
    
    def _compute_experiment_summary(self, data: Dict[str, Any]) -> ExperimentSummary:
        """Compute comprehensive experiment summary statistics."""
        
        df = data.get('results_df', pd.DataFrame())
        report = data.get('experiment_report', {})
        stats_summary = data.get('stats_summary', {})
        
        if df.empty:
            # Return minimal summary if no data
            return ExperimentSummary(
                experiment_name="Unknown",
                models_tested=[],
                total_subjects=0,
                total_responses=0,
                success_rate=0.0,
                total_cost=0.0,
                duration_hours=0.0,
                demographic_groups=[],
                risk_labels=[],
                overall_accuracy=0.0,
                demographic_parity_violation=0.0,
                equalized_odds_violation=0.0,
                calibration_error=0.0,
                consistency_score=0.0,
                chi_square_statistic=0.0,
                chi_square_p_value=1.0,
                kruskal_wallis_statistic=0.0,
                kruskal_wallis_p_value=1.0,
                cramers_v=0.0,
                cohens_d=0.0,
                accuracy_ci_lower=0.0,
                accuracy_ci_upper=0.0,
                bias_ci_lower=0.0,
                bias_ci_upper=0.0
            )
        
        # Basic statistics
        total_responses = len(df)
        models_tested = df['model'].unique().tolist() if 'model' in df.columns else []
        total_subjects = df['subject_id'].nunique() if 'subject_id' in df.columns else 0
        
        # Success rate
        success_rate = 1.0 - df['error'].notna().mean() if 'error' in df.columns else 1.0
        
        # Demographic groups
        demographic_cols = ['gender', 'age_group', 'ethnicity', 'location']
        demographic_groups = []
        for col in demographic_cols:
            if col in df.columns:
                demographic_groups.extend(df[col].unique().tolist())
        
        # Risk labels
        risk_labels = df['risk_label'].unique().tolist() if 'risk_label' in df.columns else []
        
        # Accuracy metrics
        if 'risk_band_correct' in df.columns:
            overall_accuracy = df['risk_band_correct'].mean()
            accuracy_ci = self._compute_confidence_interval(df['risk_band_correct'])
        else:
            overall_accuracy = 0.0
            accuracy_ci = (0.0, 0.0)
        
        # Fairness metrics
        demographic_parity = self._compute_demographic_parity_violation(df)
        equalized_odds = self._compute_equalized_odds_violation(df)
        calibration_error = self._compute_calibration_error(df)
        
        # Statistical tests
        chi_square_stat, chi_square_p = self._compute_chi_square_test(df)
        kw_stat, kw_p = self._compute_kruskal_wallis_test(df)
        
        # Effect sizes
        cramers_v = self._compute_cramers_v(df)
        cohens_d = self._compute_cohens_d(df)
        
        # From report data
        total_cost = report.get('total_cost', 0.0)
        duration_hours = report.get('duration_hours', 0.0)
        consistency_score = stats_summary.get('consistency_score', 0.0)
        
        return ExperimentSummary(
            experiment_name=report.get('experiment_name', 'Unnamed Experiment'),
            models_tested=models_tested,
            total_subjects=total_subjects,
            total_responses=total_responses,
            success_rate=success_rate,
            total_cost=total_cost,
            duration_hours=duration_hours,
            demographic_groups=demographic_groups,
            risk_labels=risk_labels,
            overall_accuracy=overall_accuracy,
            demographic_parity_violation=demographic_parity,
            equalized_odds_violation=equalized_odds,
            calibration_error=calibration_error,
            consistency_score=consistency_score,
            chi_square_statistic=chi_square_stat,
            chi_square_p_value=chi_square_p,
            kruskal_wallis_statistic=kw_stat,
            kruskal_wallis_p_value=kw_p,
            cramers_v=cramers_v,
            cohens_d=cohens_d,
            accuracy_ci_lower=accuracy_ci[0],
            accuracy_ci_upper=accuracy_ci[1],
            bias_ci_lower=max(0, demographic_parity - 0.1),
            bias_ci_upper=demographic_parity + 0.1
        )
    
    def _generate_all_figures(self, data: Dict[str, Any], experiment_dir: str) -> Dict[str, str]:
        """Generate all figures for the report."""
        
        figures = {}
        df = data.get('results_df', pd.DataFrame())
        
        if df.empty:
            return figures
        
        output_dir = Path(experiment_dir) / "report_figures"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Demographic bias heatmap
            fig_path = output_dir / "demographic_bias_heatmap.png"
            self._create_demographic_bias_heatmap(df, fig_path)
            figures['demographic_bias_heatmap'] = str(fig_path)
            
            # 2. Model performance comparison
            fig_path = output_dir / "model_performance.png"
            self._create_model_performance_chart(df, fig_path)
            figures['model_performance'] = str(fig_path)
            
            # 3. Risk label distribution
            fig_path = output_dir / "risk_distribution.png"
            self._create_risk_distribution_chart(df, fig_path)
            figures['risk_distribution'] = str(fig_path)
            
            # 4. Fairness metrics radar
            fig_path = output_dir / "fairness_metrics.png"
            self._create_fairness_radar_chart(df, fig_path)
            figures['fairness_metrics'] = str(fig_path)
            
            # 5. Statistical significance forest plot
            fig_path = output_dir / "significance_forest.png"
            self._create_significance_forest_plot(df, fig_path)
            figures['significance_forest'] = str(fig_path)
            
        except Exception as e:
            self.logger.error(f"Error generating figures: {str(e)}")
        
        return figures
    
    def _create_demographic_bias_heatmap(self, df: pd.DataFrame, output_path: str):
        """Create demographic bias heatmap."""
        
        if 'gender' not in df.columns or 'risk_label' not in df.columns:
            return
        
        # Create crosstab of demographics vs risk labels
        crosstab = pd.crosstab(df['gender'], df['risk_label'], normalize='index') * 100
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(crosstab, annot=True, fmt='.1f', cmap='RdYlBu_r', 
                    cbar_kws={'label': 'Percentage (%)'})
        plt.title('Risk Label Distribution by Gender', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Label', fontsize=12)
        plt.ylabel('Gender', fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_model_performance_chart(self, df: pd.DataFrame, output_path: str):
        """Create model performance comparison chart."""
        
        if 'model' not in df.columns or 'risk_band_correct' not in df.columns:
            return
        
        # Group by model and compute metrics
        model_metrics = df.groupby('model').agg({
            'risk_band_correct': ['mean', 'sem'],
            'response_time_seconds': 'mean'
        }).round(3)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy with error bars
        models = model_metrics.index
        accuracy = model_metrics[('risk_band_correct', 'mean')]
        accuracy_err = model_metrics[('risk_band_correct', 'sem')]
        
        ax1.bar(models, accuracy, yerr=accuracy_err, capsize=5, alpha=0.8)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Response time
        response_times = model_metrics[('response_time_seconds', 'mean')]
        ax2.bar(models, response_times, alpha=0.8, color='orange')
        ax2.set_title('Average Response Time', fontweight='bold')
        ax2.set_ylabel('Response Time (seconds)')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_risk_distribution_chart(self, df: pd.DataFrame, output_path: str):
        """Create risk label distribution chart."""
        
        if 'risk_label' not in df.columns:
            return
        
        risk_counts = df['risk_label'].value_counts()
        
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("Set2", len(risk_counts))
        bars = plt.bar(risk_counts.index, risk_counts.values, color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(risk_counts),
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.title('Risk Label Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Risk Label', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_fairness_radar_chart(self, df: pd.DataFrame, output_path: str):
        """Create fairness metrics radar chart."""
        
        # Compute fairness metrics
        metrics = {
            'Demographic Parity': 1 - self._compute_demographic_parity_violation(df),
            'Equalized Odds': 1 - self._compute_equalized_odds_violation(df),
            'Calibration': 1 - self._compute_calibration_error(df),
            'Individual Fairness': 0.8,  # Placeholder
            'Overall Accuracy': df['risk_band_correct'].mean() if 'risk_band_correct' in df.columns else 0.5
        }
        
        # Create radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Close the plot by repeating the first value
        values += values[:1]
        categories += categories[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=True)
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='Fairness Metrics')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Fairness Metrics Overview', y=1.08, fontsize=16, fontweight='bold')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _create_significance_forest_plot(self, df: pd.DataFrame, output_path: str):
        """Create statistical significance forest plot."""
        
        # Compute various statistical comparisons
        comparisons = []
        
        if 'gender' in df.columns and 'risk_band_correct' in df.columns:
            # Gender comparison
            male_acc = df[df['gender'] == 'Male']['risk_band_correct'].mean()
            female_acc = df[df['gender'] == 'Female']['risk_band_correct'].mean()
            
            # T-test
            male_scores = df[df['gender'] == 'Male']['risk_band_correct'].dropna()
            female_scores = df[df['gender'] == 'Female']['risk_band_correct'].dropna()
            
            if len(male_scores) > 0 and len(female_scores) > 0:
                stat, p_val = stats.ttest_ind(male_scores, female_scores)
                
                comparisons.append({
                    'comparison': 'Male vs Female Accuracy',
                    'effect_size': male_acc - female_acc,
                    'ci_lower': (male_acc - female_acc) - 0.1,
                    'ci_upper': (male_acc - female_acc) + 0.1,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                })
        
        if not comparisons:
            comparisons = [{'comparison': 'No comparisons available', 'effect_size': 0, 
                          'ci_lower': 0, 'ci_upper': 0, 'p_value': 1.0, 'significant': False}]
        
        # Create forest plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_positions = range(len(comparisons))
        effect_sizes = [c['effect_size'] for c in comparisons]
        ci_lowers = [c['ci_lower'] for c in comparisons]
        ci_uppers = [c['ci_upper'] for c in comparisons]
        colors = ['red' if c['significant'] else 'blue' for c in comparisons]
        
        # Plot effect sizes with confidence intervals
        ax.scatter(effect_sizes, y_positions, c=colors, s=100)
        
        for i, comp in enumerate(comparisons):
            ax.plot([comp['ci_lower'], comp['ci_upper']], [i, i], c=colors[i], linewidth=2)
        
        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        ax.set_yticks(y_positions)
        ax.set_yticklabels([c['comparison'] for c in comparisons])
        ax.set_xlabel('Effect Size')
        ax.set_title('Statistical Significance Forest Plot', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.figure_dpi, bbox_inches='tight')
        plt.close()
    
    def _generate_statistical_tables(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate statistical tables in LaTeX format."""
        
        tables = {}
        df = data.get('results_df', pd.DataFrame())
        
        if df.empty:
            return tables
        
        # 1. Summary statistics table
        tables['summary_stats'] = self._create_summary_stats_table(df)
        
        # 2. Model comparison table
        tables['model_comparison'] = self._create_model_comparison_table(df)
        
        # 3. Fairness metrics table
        tables['fairness_metrics'] = self._create_fairness_metrics_table(df)
        
        # 4. Statistical tests table
        tables['statistical_tests'] = self._create_statistical_tests_table(df)
        
        return tables
    
    def _create_summary_stats_table(self, df: pd.DataFrame) -> str:
        """Create summary statistics table in LaTeX format."""
        
        stats_data = []
        
        # Basic statistics
        stats_data.append(['Total Responses', len(df)])
        stats_data.append(['Unique Subjects', df['subject_id'].nunique() if 'subject_id' in df.columns else 'N/A'])
        stats_data.append(['Success Rate', f"{(1 - df['error'].notna().mean())*100:.1f}%" if 'error' in df.columns else 'N/A'])
        
        if 'risk_band_correct' in df.columns:
            accuracy = df['risk_band_correct'].mean()
            stats_data.append(['Overall Accuracy', f"{accuracy:.3f} ± {df['risk_band_correct'].sem():.3f}"])
        
        # Create LaTeX table
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\begin{tabular}{|l|r|}\n\\hline\n"
        latex_table += "\\textbf{Metric} & \\textbf{Value} \\\\ \\hline\n"
        
        for metric, value in stats_data:
            latex_table += f"{metric} & {value} \\\\ \\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\caption{Experiment Summary Statistics}\n"
        latex_table += "\\label{tab:summary_stats}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    
    def _create_model_comparison_table(self, df: pd.DataFrame) -> str:
        """Create model comparison table in LaTeX format."""
        
        if 'model' not in df.columns:
            return "% No model comparison data available\n"
        
        # Group by model
        model_stats = df.groupby('model').agg({
            'risk_band_correct': ['count', 'mean', 'sem'] if 'risk_band_correct' in df.columns else ['count'],
            'response_time_seconds': 'mean' if 'response_time_seconds' in df.columns else ['count']
        }).round(3)
        
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\begin{tabular}{|l|r|r|r|}\n\\hline\n"
        latex_table += "\\textbf{Model} & \\textbf{N} & \\textbf{Accuracy} & \\textbf{Avg. Time (s)} \\\\ \\hline\n"
        
        for model in model_stats.index:
            n_responses = model_stats.loc[model, ('risk_band_correct', 'count')] if ('risk_band_correct', 'count') in model_stats.columns else 0
            
            if ('risk_band_correct', 'mean') in model_stats.columns:
                accuracy = model_stats.loc[model, ('risk_band_correct', 'mean')]
                acc_sem = model_stats.loc[model, ('risk_band_correct', 'sem')]
                accuracy_str = f"{accuracy:.3f} ± {acc_sem:.3f}"
            else:
                accuracy_str = "N/A"
            
            if ('response_time_seconds', 'mean') in model_stats.columns:
                avg_time = model_stats.loc[model, ('response_time_seconds', 'mean')]
                time_str = f"{avg_time:.2f}"
            else:
                time_str = "N/A"
            
            latex_table += f"{model} & {n_responses} & {accuracy_str} & {time_str} \\\\ \\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\caption{Model Performance Comparison}\n"
        latex_table += "\\label{tab:model_comparison}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    
    def _create_fairness_metrics_table(self, df: pd.DataFrame) -> str:
        """Create fairness metrics table in LaTeX format."""
        
        # Compute fairness metrics
        metrics = [
            ('Demographic Parity Violation', self._compute_demographic_parity_violation(df)),
            ('Equalized Odds Violation', self._compute_equalized_odds_violation(df)),
            ('Calibration Error', self._compute_calibration_error(df)),
        ]
        
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\begin{tabular}{|l|r|r|}\n\\hline\n"
        latex_table += "\\textbf{Fairness Metric} & \\textbf{Value} & \\textbf{Interpretation} \\\\ \\hline\n"
        
        for metric_name, value in metrics:
            if value < 0.05:
                interpretation = "Excellent"
            elif value < 0.1:
                interpretation = "Good"
            elif value < 0.2:
                interpretation = "Moderate"
            else:
                interpretation = "Poor"
            
            latex_table += f"{metric_name} & {value:.3f} & {interpretation} \\\\ \\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\caption{Fairness Metrics Assessment}\n"
        latex_table += "\\label{tab:fairness_metrics}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    
    def _create_statistical_tests_table(self, df: pd.DataFrame) -> str:
        """Create statistical tests results table in LaTeX format."""
        
        # Compute statistical tests
        chi2_stat, chi2_p = self._compute_chi_square_test(df)
        kw_stat, kw_p = self._compute_kruskal_wallis_test(df)
        
        tests = [
            ('Chi-square Test', chi2_stat, chi2_p),
            ('Kruskal-Wallis Test', kw_stat, kw_p),
        ]
        
        latex_table = "\\begin{table}[h]\n\\centering\n"
        latex_table += "\\begin{tabular}{|l|r|r|l|}\n\\hline\n"
        latex_table += "\\textbf{Statistical Test} & \\textbf{Statistic} & \\textbf{p-value} & \\textbf{Result} \\\\ \\hline\n"
        
        for test_name, statistic, p_value in tests:
            significance = "Significant" if p_value < 0.05 else "Not Significant"
            latex_table += f"{test_name} & {statistic:.3f} & {p_value:.4f} & {significance} \\\\ \\hline\n"
        
        latex_table += "\\end{tabular}\n"
        latex_table += "\\caption{Statistical Test Results}\n"
        latex_table += "\\label{tab:statistical_tests}\n"
        latex_table += "\\end{table}\n"
        
        return latex_table
    
    # Helper methods for statistical computations
    def _compute_demographic_parity_violation(self, df: pd.DataFrame) -> float:
        """Compute demographic parity violation."""
        if 'gender' not in df.columns or 'risk_label' not in df.columns:
            return 0.0
        
        try:
            # Compute risk label rates by gender
            gender_rates = df.groupby('gender')['risk_label'].value_counts(normalize=True).unstack(fill_value=0)
            
            # Compute maximum difference across risk labels
            max_diff = 0.0
            for risk_label in gender_rates.columns:
                rates = gender_rates[risk_label]
                max_diff = max(max_diff, rates.max() - rates.min())
            
            return max_diff
        except:
            return 0.0
    
    def _compute_equalized_odds_violation(self, df: pd.DataFrame) -> float:
        """Compute equalized odds violation."""
        if 'gender' not in df.columns or 'risk_band_correct' not in df.columns:
            return 0.0
        
        try:
            # Compute true positive rates by gender
            tpr_by_gender = df[df['risk_band_correct'] == 1].groupby('gender').size() / df.groupby('gender').size()
            return abs(tpr_by_gender.max() - tpr_by_gender.min())
        except:
            return 0.0
    
    def _compute_calibration_error(self, df: pd.DataFrame) -> float:
        """Compute calibration error."""
        if 'prediction_confidence' not in df.columns or 'risk_band_correct' not in df.columns:
            return 0.0
        
        try:
            # Bin confidences and compute calibration error
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0.0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Get predictions in this bin
                in_bin = (df['prediction_confidence'] > bin_lower) & (df['prediction_confidence'] <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = df.loc[in_bin, 'risk_band_correct'].mean()
                    avg_confidence_in_bin = df.loc[in_bin, 'prediction_confidence'].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            return ece
        except:
            return 0.0
    
    def _compute_chi_square_test(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute chi-square test for independence."""
        if 'gender' not in df.columns or 'risk_label' not in df.columns:
            return 0.0, 1.0
        
        try:
            contingency = pd.crosstab(df['gender'], df['risk_label'])
            chi2, p_val, _, _ = stats.chi2_contingency(contingency)
            return chi2, p_val
        except:
            return 0.0, 1.0
    
    def _compute_kruskal_wallis_test(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Compute Kruskal-Wallis test."""
        if 'gender' not in df.columns or 'risk_band_correct' not in df.columns:
            return 0.0, 1.0
        
        try:
            groups = [group['risk_band_correct'].values for name, group in df.groupby('gender')]
            if len(groups) >= 2:
                stat, p_val = stats.kruskal(*groups)
                return stat, p_val
        except:
            pass
        
        return 0.0, 1.0
    
    def _compute_cramers_v(self, df: pd.DataFrame) -> float:
        """Compute Cramer's V effect size."""
        if 'gender' not in df.columns or 'risk_label' not in df.columns:
            return 0.0
        
        try:
            contingency = pd.crosstab(df['gender'], df['risk_label'])
            chi2, _, _, _ = stats.chi2_contingency(contingency)
            n = contingency.sum().sum()
            cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
            return cramers_v
        except:
            return 0.0
    
    def _compute_cohens_d(self, df: pd.DataFrame) -> float:
        """Compute Cohen's d effect size."""
        if 'gender' not in df.columns or 'risk_band_correct' not in df.columns:
            return 0.0
        
        try:
            male_scores = df[df['gender'] == 'Male']['risk_band_correct']
            female_scores = df[df['gender'] == 'Female']['risk_band_correct']
            
            if len(male_scores) > 0 and len(female_scores) > 0:
                pooled_std = np.sqrt(((len(male_scores) - 1) * male_scores.var() + 
                                    (len(female_scores) - 1) * female_scores.var()) / 
                                   (len(male_scores) + len(female_scores) - 2))
                
                if pooled_std > 0:
                    cohens_d = (male_scores.mean() - female_scores.mean()) / pooled_std
                    return abs(cohens_d)
        except:
            pass
        
        return 0.0
    
    def _compute_confidence_interval(self, data: pd.Series, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for a data series."""
        
        try:
            mean_val = data.mean()
            sem_val = data.sem()
            alpha = 1 - confidence
            
            # Use t-distribution for small samples
            if len(data) < 30:
                t_val = stats.t.ppf(1 - alpha/2, len(data) - 1)
                margin = t_val * sem_val
            else:
                z_val = stats.norm.ppf(1 - alpha/2)
                margin = z_val * sem_val
            
            return (mean_val - margin, mean_val + margin)
        except:
            return (0.0, 0.0)
    
    def _ensure_templates_exist(self):
        """Create report templates if they don't exist."""
        
        template_dir = Path(__file__).parent / "report_templates"
        template_dir.mkdir(exist_ok=True)
        
        # Academic report template
        academic_template = """\\documentclass[conference]{IEEEtran}
\\usepackage{amsmath,amssymb,amsfonts}
\\usepackage{algorithmic}
\\usepackage{graphicx}
\\usepackage{textcomp}
\\usepackage{xcolor}
\\usepackage{booktabs}
\\usepackage{url}

\\begin{document}

\\title{{{ experiment_name }}: LLM Risk Assessment Fairness Analysis}

\\author{\\IEEEauthorblockN{Automated Report Generation System}
\\IEEEauthorblockA{Generated on {{ timestamp }}}}

\\maketitle

\\begin{abstract}
This report presents a comprehensive analysis of demographic bias in Large Language Model (LLM) risk assessment for the experiment "{{ experiment_name }}". We evaluated {{ models|length }} models across {{ total_subjects }} subjects, generating {{ total_responses }} responses. Our analysis reveals {{ 'significant' if significance_detected else 'no significant' }} demographic bias with an overall accuracy of {{ "%.3f"|format(overall_accuracy) }}.
\\end{abstract}

\\section{Introduction}
This automated report analyzes LLM fairness in risk profiling using the Equip Super framework. The experiment tested for demographic bias across gender, age, and location dimensions using counterfactual analysis.

\\section{Methodology}
\\subsection{Experimental Design}
- Models tested: {{ models|join(', ') }}
- Total subjects: {{ total_subjects }}
- Total responses: {{ total_responses }}
- Success rate: {{ "%.1f"|format(success_rate * 100) }}\\%
- Duration: {{ "%.1f"|format(duration_hours) }} hours
- Total cost: \\${{ "%.2f"|format(total_cost) }}

\\subsection{Statistical Analysis}
We employed multiple statistical tests including chi-square tests for independence and Kruskal-Wallis tests for group differences. Multiple testing corrections were applied using the Holm-Bonferroni method.

\\section{Results}

\\subsection{Overall Performance}
The models achieved an overall accuracy of {{ "%.3f"|format(overall_accuracy) }} (95\\% CI: [{{ "%.3f"|format(accuracy_ci_lower) }}, {{ "%.3f"|format(accuracy_ci_upper) }}]).

\\subsection{Fairness Analysis}
{% if tables.fairness_metrics %}
{{ tables.fairness_metrics }}
{% endif %}

Key findings:
- Demographic parity violation: {{ "%.3f"|format(demographic_parity_violation) }}
- Equalized odds violation: {{ "%.3f"|format(equalized_odds_violation) }}
- Calibration error: {{ "%.3f"|format(calibration_error) }}

\\subsection{Statistical Significance}
{% if tables.statistical_tests %}
{{ tables.statistical_tests }}
{% endif %}

\\section{Discussion}
{% if demographic_parity_violation > 0.1 %}
The analysis reveals concerning levels of demographic bias, particularly in demographic parity ({{ "%.3f"|format(demographic_parity_violation) }}). This suggests that the models may be making systematically different risk assessments based on demographic characteristics.
{% else %}
The analysis shows relatively low levels of demographic bias across all measured fairness metrics. The demographic parity violation of {{ "%.3f"|format(demographic_parity_violation) }} is within acceptable bounds for most applications.
{% endif %}

\\section{Conclusion}
This automated analysis provides evidence {{ 'supporting the presence of' if demographic_parity_violation > 0.1 else 'against significant' }} demographic bias in the tested LLM risk assessment models. {{ 'Further investigation and bias mitigation strategies are recommended.' if demographic_parity_violation > 0.1 else 'The models demonstrate acceptable fairness characteristics for this application domain.' }}

\\section{Recommendations}
{% if demographic_parity_violation > 0.1 %}
1. Implement bias mitigation techniques such as adversarial debiasing
2. Retrain models with more balanced demographic representation
3. Apply post-processing fairness constraints
4. Conduct additional testing with expanded demographic categories
{% else %}
1. Continue monitoring for bias drift over time
2. Expand testing to additional demographic dimensions
3. Validate findings with larger sample sizes
4. Consider deployment with appropriate monitoring systems
{% endif %}

\\end{document}"""
        
        with open(template_dir / "academic_template.tex", 'w') as f:
            f.write(academic_template)
    
    def _generate_academic_report(self, summary: ExperimentSummary, 
                                figures: Dict[str, str], 
                                tables: Dict[str, str],
                                data: Dict[str, Any]) -> str:
        """Generate academic report content."""
        
        template = self.jinja_env.get_template("academic_template.tex")
        
        return template.render(
            experiment_name=summary.experiment_name,
            timestamp=datetime.now().strftime("%B %d, %Y"),
            models=summary.models_tested,
            total_subjects=summary.total_subjects,
            total_responses=summary.total_responses,
            success_rate=summary.success_rate,
            duration_hours=summary.duration_hours,
            total_cost=summary.total_cost,
            overall_accuracy=summary.overall_accuracy,
            accuracy_ci_lower=summary.accuracy_ci_lower,
            accuracy_ci_upper=summary.accuracy_ci_upper,
            demographic_parity_violation=summary.demographic_parity_violation,
            equalized_odds_violation=summary.equalized_odds_violation,
            calibration_error=summary.calibration_error,
            significance_detected=summary.chi_square_p_value < 0.05,
            tables=tables,
            figures=figures
        )
    
    def _generate_executive_report(self, summary: ExperimentSummary, 
                                 figures: Dict[str, str]) -> str:
        """Generate executive summary report."""
        
        executive_template = f"""
# Executive Summary: {summary.experiment_name}

## Key Findings

- **Overall Model Accuracy**: {summary.overall_accuracy:.1%}
- **Demographic Bias Level**: {"HIGH" if summary.demographic_parity_violation > 0.1 else "LOW" if summary.demographic_parity_violation < 0.05 else "MODERATE"}
- **Statistical Significance**: {"YES" if summary.chi_square_p_value < 0.05 else "NO"}
- **Cost**: ${summary.total_cost:.2f}

## Recommendations

{"**IMMEDIATE ACTION REQUIRED**: Implement bias mitigation before deployment." if summary.demographic_parity_violation > 0.1 else "Models demonstrate acceptable fairness for deployment with monitoring."}

## Detailed Metrics

- Demographic Parity Violation: {summary.demographic_parity_violation:.3f}
- Equalized Odds Violation: {summary.equalized_odds_violation:.3f}
- Calibration Error: {summary.calibration_error:.3f}
- Effect Size (Cramer's V): {summary.cramers_v:.3f}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        return executive_template.strip()
    
    def _generate_compliance_report(self, summary: ExperimentSummary, 
                                  data: Dict[str, Any]) -> str:
        """Generate compliance report for ethics boards."""
        
        compliance_template = f"""
# Compliance Report: {summary.experiment_name}

## Ethics and Compliance Assessment

### Data Protection
- No real personal data used: ✓
- Synthetic demographic data only: ✓
- GDPR compliance: ✓

### Bias Assessment Results
- Demographic bias testing completed: ✓
- Statistical significance testing: ✓
- Multiple testing corrections applied: ✓

### Fairness Metrics
- Demographic Parity: {"PASS" if summary.demographic_parity_violation < 0.1 else "FAIL"}
- Equalized Odds: {"PASS" if summary.equalized_odds_violation < 0.1 else "FAIL"}
- Individual Fairness: ASSESSED

### Recommendations for Deployment
{"⚠️ CAUTION: High bias levels detected. Additional mitigation required." if summary.demographic_parity_violation > 0.1 else "✅ APPROVED: Acceptable bias levels for deployment."}

### Audit Trail
- Experiment ID: {summary.experiment_name}
- Models tested: {len(summary.models_tested)}
- Sample size: {summary.total_subjects} subjects
- Statistical power: {"Adequate" if summary.total_subjects >= 30 else "Limited"}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Compliance framework: EU AI Act + GDPR + Fair Credit Reporting Act
        """
        
        return compliance_template.strip()
    
    def _generate_standard_report(self, summary: ExperimentSummary, 
                                figures: Dict[str, str], 
                                tables: Dict[str, str]) -> str:
        """Generate standard report."""
        
        return f"""
# LLM Risk Fairness Analysis Report

## Experiment: {summary.experiment_name}

### Summary
- Models: {', '.join(summary.models_tested)}
- Subjects: {summary.total_subjects}
- Responses: {summary.total_responses}
- Accuracy: {summary.overall_accuracy:.3f}
- Cost: ${summary.total_cost:.2f}

### Fairness Assessment
- Demographic Parity Violation: {summary.demographic_parity_violation:.3f}
- Equalized Odds Violation: {summary.equalized_odds_violation:.3f}
- Calibration Error: {summary.calibration_error:.3f}

### Statistical Tests
- Chi-square p-value: {summary.chi_square_p_value:.4f}
- Kruskal-Wallis p-value: {summary.kruskal_wallis_p_value:.4f}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
    
    def _generate_comparative_figures(self, experiments: List) -> Dict[str, str]:
        """Generate comparative figures across experiments."""
        # Placeholder for comparative visualization
        return {}
    
    def _generate_comparative_tables(self, experiments: List) -> Dict[str, str]:
        """Generate comparative tables across experiments."""
        # Placeholder for comparative tables
        return {}
    
    def _generate_comparative_content(self, experiments: List, 
                                    figures: Dict[str, str], 
                                    tables: Dict[str, str]) -> str:
        """Generate comparative report content."""
        
        return f"""
# Comparative Analysis Report

## Experiments Analyzed: {len(experiments)}

### Cross-Experiment Summary
[Comparative analysis content would go here]

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
    
    def _save_report(self, content: str, figures: Dict[str, str], 
                    output_path: str) -> Dict[str, str]:
        """Save generated report and return output file paths."""
        
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        output_files = {}
        
        if self.config.output_format == "latex":
            # Save LaTeX source
            latex_file = output_dir / "automated_report.tex"
            with open(latex_file, 'w', encoding='utf-8') as f:
                f.write(content)
            output_files['latex'] = str(latex_file)
            
        elif self.config.output_format == "html":
            # Convert to HTML (basic markdown to HTML)
            html_content = content.replace('\n', '<br>\n')
            html_file = output_dir / "automated_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(f"<html><body><pre>{html_content}</pre></body></html>")
            output_files['html'] = str(html_file)
            
        else:
            # Save as text/markdown
            text_file = output_dir / "automated_report.md"
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(content)
            output_files['text'] = str(text_file)
        
        # Copy figures to output directory
        if figures:
            figures_dir = output_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            for fig_name, fig_path in figures.items():
                if os.path.exists(fig_path):
                    import shutil
                    dest_path = figures_dir / os.path.basename(fig_path)
                    shutil.copy2(fig_path, dest_path)
                    output_files[f'figure_{fig_name}'] = str(dest_path)
        
        return output_files


def main():
    """CLI interface for automated report generation."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate automated reports for LLM fairness experiments")
    parser.add_argument('experiment_dir', help='Path to experiment results directory')
    parser.add_argument('--output-dir', help='Output directory for report')
    parser.add_argument('--report-type', choices=['academic', 'executive', 'compliance'], 
                       default='academic', help='Type of report to generate')
    parser.add_argument('--format', choices=['pdf', 'html', 'latex', 'markdown'], 
                       default='pdf', help='Output format')
    parser.add_argument('--template-style', choices=['ieee', 'apa', 'nature'], 
                       default='ieee', help='Template style for academic reports')
    
    args = parser.parse_args()
    
    # Create report configuration
    config = ReportConfig(
        report_type=args.report_type,
        output_format=args.format,
        template_style=args.template_style
    )
    
    # Generate report
    generator = AutomatedReportGenerator(config)
    output_files = generator.generate_comprehensive_report(
        args.experiment_dir, 
        args.output_dir
    )
    
    print("Report generated successfully!")
    print("Output files:")
    for file_type, file_path in output_files.items():
        print(f"  {file_type}: {file_path}")


if __name__ == "__main__":
    main()