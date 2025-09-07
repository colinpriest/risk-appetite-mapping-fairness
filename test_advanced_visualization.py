#!/usr/bin/env python3
"""
Test Suite for Advanced Visualization System

Tests all components of the advanced visualization suite including 3D bias landscapes,
correlation networks, temporal animations, publication figures, custom grammar,
and interactive dashboards.

Author: LLM Fairness Research Platform
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os
from datetime import datetime, timedelta
import json
from unittest.mock import patch, MagicMock
import warnings

# Suppress warnings during testing
warnings.filterwarnings('ignore')

try:
    from advanced_visualization import (
        BiasLandscapeVisualizer,
        CorrelationNetworkVisualizer, 
        TemporalAnimationVisualizer,
        PublicationFigureGenerator,
        CustomGrammarVisualizer,
        InteractiveDashboard,
        AdvancedVisualizationSuite,
        VisualizationConfig
    )
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import visualization modules: {e}")
    print("Some visualization tests will be skipped")
    IMPORTS_AVAILABLE = False


class TestVisualizationConfig(unittest.TestCase):
    """Test visualization configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        config = VisualizationConfig()
        self.assertEqual(config.figure_size, (12, 8))
        self.assertEqual(config.dpi, 300)
        self.assertEqual(config.color_palette, "viridis")
        self.assertEqual(config.font_family, "Arial")
        self.assertTrue(config.interactive_mode)
        self.assertIn('png', config.export_formats)
    
    def test_custom_config(self):
        """Test custom configuration"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        config = VisualizationConfig(
            figure_size=(16, 10),
            dpi=600,
            color_palette="plasma",
            export_formats=['svg', 'pdf']
        )
        self.assertEqual(config.figure_size, (16, 10))
        self.assertEqual(config.dpi, 600)
        self.assertEqual(config.color_palette, "plasma")
        self.assertEqual(config.export_formats, ['svg', 'pdf'])


class TestBiasLandscapeVisualizer(unittest.TestCase):
    """Test 3D bias landscape visualization"""
    
    def setUp(self):
        """Set up test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], 100),
            'age_group': np.random.choice(['Young', 'Senior'], 100),
            'location': np.random.choice(['Urban', 'Rural'], 100),
            'demographic_parity_difference': np.random.normal(0, 0.1, 100),
            'equalized_odds_difference': np.random.normal(0, 0.08, 100),
            'risk_category': np.random.choice(['Low', 'Medium', 'High'], 100)
        })
        
        self.visualizer = BiasLandscapeVisualizer()
    
    def test_2d_surface_creation(self):
        """Test 2D bias surface creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_3d_bias_surface(
            self.test_data,
            demographic_cols=['gender', 'age_group'],
            bias_metric='demographic_parity_difference'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        self.assertIn('Bias Landscape', fig.layout.title.text)
    
    def test_1d_surface_creation(self):
        """Test 1D bias surface (elevated line) creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_3d_bias_surface(
            self.test_data,
            demographic_cols=['gender'],
            bias_metric='demographic_parity_difference'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
    
    def test_bias_heatmap_matrix(self):
        """Test bias metrics correlation heatmap"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_bias_heatmap_matrix(self.test_data)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertIn('Correlation Matrix', fig.layout.title.text)
    
    def test_risk_category_filtering(self):
        """Test risk category filtering"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_3d_bias_surface(
            self.test_data,
            demographic_cols=['gender', 'age_group'],
            bias_metric='demographic_parity_difference',
            risk_category='High'
        )
        
        self.assertIsInstance(fig, go.Figure)


class TestCorrelationNetworkVisualizer(unittest.TestCase):
    """Test correlation network visualization"""
    
    def setUp(self):
        """Set up test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'gender': np.random.choice(['Male', 'Female'], 200),
            'age': np.random.randint(18, 80, 200),
            'location': np.random.choice(['Urban', 'Rural'], 200),
            'demographic_parity_difference': np.random.normal(0, 0.1, 200),
            'equalized_odds_difference': np.random.normal(0, 0.08, 200),
            'individual_fairness_score': np.random.beta(2, 3, 200),
            'model_bias_score': np.random.normal(0, 0.05, 200)
        })
        
        self.visualizer = CorrelationNetworkVisualizer()
    
    def test_bias_correlation_network(self):
        """Test bias correlation network creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_bias_correlation_network(
            self.test_data, threshold=0.1
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertIn('Network', fig.layout.title.text)
    
    def test_demographic_influence_network(self):
        """Test demographic influence network"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_demographic_influence_network(self.test_data)
        
        self.assertIsInstance(fig, go.Figure)
        self.assertIn('Network', fig.layout.title.text)
    
    def test_empty_network_handling(self):
        """Test handling of empty networks"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Create data with no correlations
        empty_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        
        fig = self.visualizer.create_bias_correlation_network(
            empty_data, threshold=0.9
        )
        
        self.assertIsInstance(fig, go.Figure)
    
    def test_network_threshold_adjustment(self):
        """Test network creation with different thresholds"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig_low = self.visualizer.create_bias_correlation_network(
            self.test_data, threshold=0.1
        )
        fig_high = self.visualizer.create_bias_correlation_network(
            self.test_data, threshold=0.8
        )
        
        self.assertIsInstance(fig_low, go.Figure)
        self.assertIsInstance(fig_high, go.Figure)


class TestTemporalAnimationVisualizer(unittest.TestCase):
    """Test temporal animation visualization"""
    
    def setUp(self):
        """Set up temporal test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        self.temporal_data = pd.DataFrame({
            'timestamp': np.tile(dates, 3)[:150],
            'demographic_group': np.repeat(['Group_A', 'Group_B', 'Group_C'], 50),
            'model': np.random.choice(['gpt-4o', 'claude-3-5', 'gemini'], 150),
            'demographic_parity_difference': np.random.normal(0, 0.1, 150),
            'bias_trend': np.sin(np.arange(150) * 0.1) + np.random.normal(0, 0.05, 150)
        })
        
        self.visualizer = TemporalAnimationVisualizer()
    
    def test_bias_evolution_animation(self):
        """Test bias evolution animation creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = self.visualizer.create_bias_evolution_animation(
            self.temporal_data,
            time_col='timestamp',
            bias_metric='demographic_parity_difference',
            group_col='demographic_group'
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.frames), 0)
        self.assertIn('Evolution', fig.layout.title.text)
    
    def test_model_comparison_animation(self):
        """Test model comparison animation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        model_data = pd.DataFrame({
            'model': ['gpt-4o', 'claude-3-5', 'gemini'] * 20,
            'accuracy': np.random.beta(8, 2, 60),
            'bias_score': np.random.normal(0, 0.1, 60),
            'fairness_metric': np.random.beta(5, 3, 60)
        })
        
        fig = self.visualizer.create_model_comparison_animation(
            model_data, ['accuracy', 'bias_score', 'fairness_metric']
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.frames), 0)
    
    def test_missing_time_column(self):
        """Test handling of missing time column"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        data_no_time = self.temporal_data.drop('timestamp', axis=1)
        
        fig = self.visualizer.create_bias_evolution_animation(
            data_no_time,
            time_col='nonexistent_time',
            bias_metric='demographic_parity_difference',
            group_col='demographic_group'
        )
        
        self.assertIsInstance(fig, go.Figure)


class TestPublicationFigureGenerator(unittest.TestCase):
    """Test publication-ready figure generation"""
    
    def setUp(self):
        """Set up test data and temporary directory"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'model': np.random.choice(['gpt-4o', 'claude-3-5', 'gemini'], 200),
            'gender': np.random.choice(['Male', 'Female'], 200),
            'age_group': np.random.choice(['Young', 'Senior'], 200),
            'demographic_parity_difference': np.random.normal(0, 0.1, 200),
            'equalized_odds_difference': np.random.normal(0, 0.08, 200),
            'individual_fairness_score': np.random.beta(2, 3, 200),
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='H')
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.generator = PublicationFigureGenerator()
    
    def tearDown(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')  # Close any open matplotlib figures
    
    def test_bias_comparison_figure(self):
        """Test bias comparison figure generation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        saved_files = self.generator.generate_bias_comparison_figure(
            self.test_data, output_dir=self.temp_dir
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 0)
        
        # Check that files were actually created
        for format_type, filepath in saved_files.items():
            self.assertTrue(os.path.exists(filepath), f"File not created: {filepath}")
    
    def test_statistical_summary_figure(self):
        """Test statistical summary figure generation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        saved_files = self.generator.generate_statistical_summary_figure(
            self.test_data, output_dir=self.temp_dir
        )
        
        self.assertIsInstance(saved_files, dict)
        self.assertGreater(len(saved_files), 0)
        
        # Check file creation
        for format_type, filepath in saved_files.items():
            self.assertTrue(os.path.exists(filepath))
    
    def test_custom_config(self):
        """Test generator with custom configuration"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        config = VisualizationConfig(
            export_formats=['png'],
            figure_size=(10, 6),
            dpi=150
        )
        
        generator = PublicationFigureGenerator(config)
        saved_files = generator.generate_bias_comparison_figure(
            self.test_data, output_dir=self.temp_dir
        )
        
        self.assertIn('png', saved_files)
        self.assertTrue(os.path.exists(saved_files['png']))


class TestCustomGrammarVisualizer(unittest.TestCase):
    """Test custom grammar of graphics implementation"""
    
    def setUp(self):
        """Set up test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'x_var': np.random.normal(0, 1, 100),
            'y_var': np.random.normal(0, 1, 100),
            'color_var': np.random.choice(['A', 'B', 'C'], 100),
            'size_var': np.random.uniform(1, 10, 100)
        })
        
        self.visualizer = CustomGrammarVisualizer()
    
    def test_ggplot_chaining(self):
        """Test ggplot-style method chaining"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        fig = (self.visualizer
               .ggplot(self.test_data)
               .aes(x='x_var', y='y_var', color='color_var')
               .geom_point(size=8)
               .theme_minimal()
               .labs(title='Test Plot', x='X Variable', y='Y Variable')
               .build())
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
        self.assertEqual(fig.layout.title.text, 'Test Plot')
    
    def test_geom_line(self):
        """Test line geometry"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Sort data for line plot
        sorted_data = self.test_data.sort_values('x_var')
        
        fig = (self.visualizer
               .ggplot(sorted_data)
               .aes(x='x_var', y='y_var', color='color_var')
               .geom_line(width=3)
               .build())
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
    
    def test_geom_bar(self):
        """Test bar geometry"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        bar_data = self.test_data.groupby('color_var')['y_var'].mean().reset_index()
        bar_data.columns = ['category', 'value']
        
        fig = (self.visualizer
               .ggplot(bar_data)
               .aes(x='category', y='value')
               .geom_bar()
               .build())
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
    
    def test_aesthetics_mapping(self):
        """Test aesthetic mappings"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Test that aesthetics are stored correctly
        viz = (self.visualizer
               .ggplot(self.test_data)
               .aes(x='x_var', y='y_var', color='color_var'))
        
        self.assertEqual(viz.aesthetics['x'], 'x_var')
        self.assertEqual(viz.aesthetics['y'], 'y_var')
        self.assertEqual(viz.aesthetics['color'], 'color_var')


class TestAdvancedVisualizationSuite(unittest.TestCase):
    """Test the main visualization suite orchestrator"""
    
    def setUp(self):
        """Set up test data and temporary directory"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'model': np.random.choice(['gpt-4o', 'claude-3-5', 'gemini'], 300),
            'gender': np.random.choice(['Male', 'Female'], 300),
            'age': np.random.randint(18, 80, 300),
            'location': np.random.choice(['Urban', 'Rural'], 300),
            'demographic_parity_difference': np.random.normal(0, 0.1, 300),
            'equalized_odds_difference': np.random.normal(0, 0.08, 300),
            'individual_fairness_score': np.random.beta(2, 3, 300),
            'bias_metric_1': np.random.normal(0, 0.05, 300),
            'bias_metric_2': np.random.normal(0, 0.03, 300),
            'timestamp': pd.date_range(start='2024-01-01', periods=300, freq='H')
        })
        
        self.temp_dir = tempfile.mkdtemp()
        self.viz_suite = AdvancedVisualizationSuite()
    
    def tearDown(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_complete_analysis_suite(self):
        """Test complete visualization suite creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        results = self.viz_suite.create_complete_analysis_suite(
            self.test_data, output_dir=self.temp_dir
        )
        
        # Check that all expected components are present
        self.assertIn('interactive_figures', results)
        self.assertIn('publication_figures', results)
        self.assertIn('networks', results)
        self.assertIn('saved_files', results)
        self.assertIn('dashboard_app', results)
        
        # Check that files were created
        self.assertGreater(len(results['saved_files']), 0)
        
        # Verify files exist on disk
        for file_key, file_path in results['saved_files'].items():
            if isinstance(file_path, str):
                self.assertTrue(os.path.exists(file_path), 
                              f"File not found: {file_path}")
    
    def test_bias_metrics_identification(self):
        """Test bias metrics identification"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        bias_cols = self.viz_suite._identify_bias_metrics(self.test_data)
        
        expected_bias_cols = [
            'demographic_parity_difference',
            'equalized_odds_difference', 
            'individual_fairness_score',
            'bias_metric_1',
            'bias_metric_2'
        ]
        
        for col in expected_bias_cols:
            self.assertIn(col, bias_cols)
    
    def test_demographic_cols_identification(self):
        """Test demographic columns identification"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        demo_cols = self.viz_suite._identify_demographic_cols(self.test_data)
        
        expected_demo_cols = ['gender', 'age', 'location']
        
        for col in expected_demo_cols:
            self.assertIn(col, demo_cols)
    
    def test_visualization_summary_generation(self):
        """Test visualization summary generation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        mock_results = {
            'interactive_figures': {'fig1': 'mock_fig'},
            'publication_figures': {'fig2': 'mock_fig'},
            'animations': {},
            'networks': {'net1': 'mock_net'},
            'saved_files': {'file1': 'path1', 'file2': 'path2'}
        }
        
        summary = self.viz_suite._generate_visualization_summary(
            self.test_data, mock_results
        )
        
        self.assertIn('dataset_info', summary)
        self.assertIn('visualizations_created', summary)
        self.assertIn('recommendations', summary)
        self.assertIn('timestamp', summary)
        
        # Check dataset info
        dataset_info = summary['dataset_info']
        self.assertEqual(dataset_info['total_rows'], len(self.test_data))
        self.assertEqual(dataset_info['total_columns'], len(self.test_data.columns))
        self.assertTrue(dataset_info['has_temporal_data'])
        self.assertTrue(dataset_info['has_model_data'])
    
    def test_analysis_recommendations(self):
        """Test analysis recommendations generation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        recommendations = self.viz_suite._generate_analysis_recommendations(
            self.test_data
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check for expected recommendations based on test data
        rec_text = ' '.join(recommendations).lower()
        self.assertIn('temporal', rec_text)  # Should recommend temporal analysis
        self.assertIn('model', rec_text)     # Should recommend model comparison


@patch('dash.Dash')
class TestInteractiveDashboard(unittest.TestCase):
    """Test interactive dashboard creation"""
    
    def setUp(self):
        """Set up test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        self.test_data = pd.DataFrame({
            'model': np.random.choice(['gpt-4o', 'claude-3-5'], 200),
            'gender': np.random.choice(['Male', 'Female'], 200),
            'demographic_parity_difference': np.random.normal(0, 0.1, 200),
            'bias_metric': np.random.normal(0, 0.05, 200),
            'timestamp': pd.date_range(start='2024-01-01', periods=200, freq='H')
        })
    
    def test_dashboard_app_creation(self, mock_dash):
        """Test dashboard app creation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Create mock Dash app
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        dashboard = InteractiveDashboard()
        app = dashboard.create_dashboard_app(self.test_data)
        
        # Verify Dash was called
        mock_dash.assert_called_once()
        self.assertEqual(app, mock_app)
    
    def test_option_generation(self, mock_dash):
        """Test dropdown option generation"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        mock_app = MagicMock()
        mock_dash.return_value = mock_app
        
        dashboard = InteractiveDashboard()
        dashboard.results_data = self.test_data
        
        # Test bias metric options
        bias_options = dashboard._get_bias_metric_options()
        self.assertIsInstance(bias_options, list)
        self.assertGreater(len(bias_options), 0)
        
        # Test demographic options
        demo_options = dashboard._get_demographic_options()
        self.assertIsInstance(demo_options, list)
        
        # Test model options
        model_options = dashboard._get_model_options()
        self.assertIsInstance(model_options, list)
        self.assertGreater(len(model_options), 0)
        
        # Test date range
        date_range = dashboard._get_date_range()
        self.assertIsInstance(date_range, tuple)
        self.assertEqual(len(date_range), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete visualization system"""
    
    def setUp(self):
        """Set up comprehensive test data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        np.random.seed(42)
        n_samples = 500
        
        self.comprehensive_data = pd.DataFrame({
            'model': np.random.choice(['gpt-4o', 'claude-3-5-sonnet', 'gemini-pro'], n_samples),
            'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
            'age': np.random.randint(18, 80, n_samples),
            'location': np.random.choice(['Urban', 'Rural', 'Suburban'], n_samples),
            'indigenous_status': np.random.choice(['Indigenous', 'Non-Indigenous'], n_samples),
            'demographic_parity_difference': np.random.normal(0, 0.1, n_samples),
            'equalized_odds_difference': np.random.normal(0, 0.08, n_samples),
            'individual_fairness_score': np.random.beta(2, 3, n_samples),
            'statistical_parity_difference': np.random.normal(0, 0.05, n_samples),
            'disparate_impact_ratio': np.random.beta(8, 2, n_samples),
            'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
            'experiment_id': np.random.randint(1, 10, n_samples),
            'confidence_score': np.random.beta(3, 2, n_samples)
        })
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        plt.close('all')
    
    def test_end_to_end_visualization_pipeline(self):
        """Test complete end-to-end visualization pipeline"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Create visualization suite with custom config
        config = VisualizationConfig(
            figure_size=(10, 8),
            export_formats=['png', 'html'],
            animation_duration=100  # Faster for testing
        )
        
        viz_suite = AdvancedVisualizationSuite(config)
        
        # Run complete analysis
        results = viz_suite.create_complete_analysis_suite(
            self.comprehensive_data,
            output_dir=self.temp_dir
        )
        
        # Comprehensive validation
        self.assertNotIn('error', results, "Pipeline should complete without errors")
        
        # Check all major components were created
        expected_components = [
            'interactive_figures',
            'publication_figures', 
            'networks',
            'saved_files',
            'dashboard_app'
        ]
        
        for component in expected_components:
            self.assertIn(component, results, f"Missing component: {component}")
        
        # Validate file creation
        self.assertGreater(len(results['saved_files']), 5, 
                          "Should create multiple output files")
        
        # Check specific file types were created
        saved_files_str = str(results['saved_files'])
        self.assertIn('.html', saved_files_str, "Should create HTML files")
        self.assertIn('.png', saved_files_str, "Should create PNG files")
        
        # Validate summary file creation and content
        summary_path = results['saved_files'].get('summary')
        if summary_path:
            self.assertTrue(os.path.exists(summary_path))
            with open(summary_path, 'r') as f:
                summary_data = json.load(f)
                self.assertIn('dataset_info', summary_data)
                self.assertIn('visualizations_created', summary_data)
    
    def test_robustness_with_missing_data(self):
        """Test system robustness with missing/incomplete data"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Create data with missing values and limited columns
        sparse_data = pd.DataFrame({
            'model': ['gpt-4o'] * 50,
            'some_bias_metric': np.random.normal(0, 0.1, 50),
            'numeric_col': np.random.normal(0, 1, 50)
        })
        
        # Add some NaN values
        sparse_data.loc[10:15, 'some_bias_metric'] = np.nan
        
        viz_suite = AdvancedVisualizationSuite()
        
        # Should handle sparse data gracefully
        results = viz_suite.create_complete_analysis_suite(
            sparse_data,
            output_dir=self.temp_dir
        )
        
        # Should still produce some results, even if limited
        self.assertIn('saved_files', results)
        self.assertNotIn('error', results)
    
    def test_large_dataset_performance(self):
        """Test performance with larger datasets"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Visualization imports not available")
        
        # Create larger dataset
        np.random.seed(42)
        large_data = pd.DataFrame({
            'model': np.random.choice(['gpt-4o', 'claude-3-5'], 2000),
            'demographic_col': np.random.choice(['A', 'B', 'C', 'D'], 2000),
            'bias_metric': np.random.normal(0, 0.1, 2000),
            'timestamp': pd.date_range(start='2024-01-01', periods=2000, freq='30min')
        })
        
        viz_suite = AdvancedVisualizationSuite()
        
        import time
        start_time = time.time()
        
        results = viz_suite.create_complete_analysis_suite(
            large_data,
            output_dir=self.temp_dir
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(processing_time, 120, "Processing should complete within 2 minutes")
        self.assertNotIn('error', results)


def run_visualization_tests():
    """
    Run all visualization tests with comprehensive reporting.
    """
    print("="*80)
    print("ADVANCED VISUALIZATION SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    if not IMPORTS_AVAILABLE:
        print("WARNING: Visualization imports not available. Tests will be skipped.")
        print("Please install required dependencies:")
        print("pip install plotly matplotlib networkx dash umap-learn")
        print()
        return False
    
    # Create test suite
    test_classes = [
        TestVisualizationConfig,
        TestBiasLandscapeVisualizer,
        TestCorrelationNetworkVisualizer,
        TestTemporalAnimationVisualizer,
        TestPublicationFigureGenerator,
        TestCustomGrammarVisualizer,
        TestAdvancedVisualizationSuite,
        TestInteractiveDashboard,
        TestIntegration
    ]
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes to the suite
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print()
    
    if result.failures:
        print("FAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
        print()
    
    if result.errors:
        print("ERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
        print()
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print()
        print("Advanced Visualization System is ready for use!")
        print("Features tested and validated:")
        print("- 3D Bias Landscape Visualization")
        print("- Correlation Network Analysis")
        print("- Temporal Animation Generation")
        print("- Publication-Ready Figure Creation")
        print("- Custom Grammar of Graphics")
        print("- Interactive Dashboard Components")
        print("- End-to-End Integration Pipeline")
    else:
        print("❌ Some tests failed. Please review the failures above.")
    
    print()
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_visualization_tests()
    exit(0 if success else 1)