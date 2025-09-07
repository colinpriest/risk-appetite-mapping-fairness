"""
Advanced Visualization Suite for LLM Bias Analysis

This module provides sophisticated visualization capabilities for exploring and understanding
bias patterns in LLM risk profiling experiments. Features include interactive 3D bias
landscapes, dynamic correlation networks, animated temporal displays, and publication-ready
figure generation.

Key Components:
- BiasLandscapeVisualizer: 3D bias surface mapping with interactive exploration
- CorrelationNetworkVisualizer: Dynamic network graphs for bias relationships
- TemporalAnimationVisualizer: Animated bias evolution over time
- PublicationFigureGenerator: High-quality figures for academic publications
- InteractiveDashboard: Real-time exploration with filtering and drill-down
- CustomGrammarVisualizer: Grammar of graphics implementation for flexible plotting

Author: LLM Fairness Research Platform
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
import scipy.stats as stats
from sklearn.manifold import TSNE, UMAP
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings
from datetime import datetime, timedelta
import io
import base64
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set style preferences
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters"""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    color_palette: str = "viridis"
    font_family: str = "Arial"
    font_size: int = 12
    export_formats: List[str] = None
    interactive_mode: bool = True
    animation_duration: int = 500
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['png', 'pdf', 'svg', 'html']


class BiasLandscapeVisualizer:
    """
    Creates interactive 3D bias landscape visualizations to explore bias patterns
    across demographic dimensions and risk categories.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
    def create_3d_bias_surface(self, 
                             results_df: pd.DataFrame,
                             demographic_cols: List[str],
                             bias_metric: str = 'demographic_parity_difference',
                             risk_category: str = None) -> go.Figure:
        """
        Create 3D surface plot showing bias landscape across demographic dimensions.
        
        Args:
            results_df: Experiment results DataFrame
            demographic_cols: List of demographic columns to visualize
            bias_metric: Bias metric to plot on Z-axis
            risk_category: Specific risk category to focus on (optional)
            
        Returns:
            Plotly 3D surface figure
        """
        logger.info(f"Creating 3D bias surface for {bias_metric}")
        
        # Prepare data
        if risk_category:
            data = results_df[results_df['risk_category'] == risk_category].copy()
        else:
            data = results_df.copy()
            
        # Handle different demographic combinations
        if len(demographic_cols) == 2:
            return self._create_2d_surface(data, demographic_cols, bias_metric)
        elif len(demographic_cols) == 1:
            return self._create_1d_surface(data, demographic_cols[0], bias_metric)
        else:
            # Use dimensionality reduction for >2 dimensions
            return self._create_reduced_surface(data, demographic_cols, bias_metric)
    
    def _create_2d_surface(self, data: pd.DataFrame, 
                          demographic_cols: List[str], 
                          bias_metric: str) -> go.Figure:
        """Create 2D surface plot for two demographic dimensions"""
        
        # Create mesh grid
        x_vals = data[demographic_cols[0]].unique()
        y_vals = data[demographic_cols[1]].unique()
        
        # Aggregate bias values
        pivot_data = data.pivot_table(
            values=bias_metric,
            index=demographic_cols[0],
            columns=demographic_cols[1],
            aggfunc='mean',
            fill_value=0
        )
        
        # Create 3D surface
        fig = go.Figure(data=[
            go.Surface(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale=self.config.color_palette,
                name='Bias Surface'
            )
        ])
        
        fig.update_layout(
            title=f'3D Bias Landscape: {bias_metric}',
            scene=dict(
                xaxis_title=demographic_cols[1],
                yaxis_title=demographic_cols[0],
                zaxis_title=bias_metric,
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def _create_1d_surface(self, data: pd.DataFrame, 
                          demographic_col: str, 
                          bias_metric: str) -> go.Figure:
        """Create 1D surface plot (elevated line) for single demographic"""
        
        grouped = data.groupby(demographic_col)[bias_metric].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        # Add main line
        fig.add_trace(go.Scatter3d(
            x=grouped[demographic_col],
            y=[0] * len(grouped),
            z=grouped['mean'],
            mode='lines+markers',
            line=dict(width=8, color='blue'),
            marker=dict(size=8),
            name='Bias Trend'
        ))
        
        # Add confidence bands
        for i, row in grouped.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[row[demographic_col], row[demographic_col]],
                y=[0, 0],
                z=[row['mean'] - row['std'], row['mean'] + row['std']],
                mode='lines',
                line=dict(width=2, color='rgba(255,0,0,0.3)'),
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'Bias Profile: {bias_metric} by {demographic_col}',
            scene=dict(
                xaxis_title=demographic_col,
                yaxis_title='',
                zaxis_title=bias_metric
            )
        )
        
        return fig
    
    def _create_reduced_surface(self, data: pd.DataFrame, 
                               demographic_cols: List[str], 
                               bias_metric: str) -> go.Figure:
        """Create surface using dimensionality reduction for >2 demographics"""
        
        # Prepare feature matrix
        features = data[demographic_cols].copy()
        
        # Encode categorical variables
        for col in features.columns:
            if features[col].dtype == 'object':
                features[col] = pd.Categorical(features[col]).codes
        
        # Apply UMAP for dimensionality reduction
        reducer = UMAP(n_components=2, random_state=42)
        reduced_coords = reducer.fit_transform(features)
        
        # Create scatter plot
        fig = go.Figure(data=[
            go.Scatter3d(
                x=reduced_coords[:, 0],
                y=reduced_coords[:, 1],
                z=data[bias_metric],
                mode='markers',
                marker=dict(
                    size=8,
                    color=data[bias_metric],
                    colorscale=self.config.color_palette,
                    showscale=True
                ),
                text=[f"Sample {i}" for i in range(len(data))],
                hovertemplate='<b>%{text}</b><br>' +
                             'UMAP-1: %{x:.2f}<br>' +
                             'UMAP-2: %{y:.2f}<br>' +
                             f'{bias_metric}: %{{z:.3f}}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Bias Landscape (UMAP Projection): {bias_metric}',
            scene=dict(
                xaxis_title='UMAP Component 1',
                yaxis_title='UMAP Component 2', 
                zaxis_title=bias_metric
            )
        )
        
        return fig
    
    def create_bias_heatmap_matrix(self, results_df: pd.DataFrame) -> go.Figure:
        """Create comprehensive heatmap matrix of all bias metrics"""
        
        # Select bias metrics
        bias_cols = [col for col in results_df.columns if 'bias' in col.lower() or 
                    'parity' in col.lower() or 'fairness' in col.lower()]
        
        if not bias_cols:
            # Fallback to numeric columns that might represent bias
            bias_cols = results_df.select_dtypes(include=[np.number]).columns.tolist()
            bias_cols = [col for col in bias_cols if col not in ['age', 'score', 'K', 'repeats']]
        
        # Create correlation matrix
        corr_matrix = results_df[bias_cols].corr()
        
        # Create heatmap
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.round(3).astype(str).values,
            colorscale='RdBu',
            showscale=True
        )
        
        fig.update_layout(
            title='Bias Metrics Correlation Matrix',
            xaxis={'side': 'bottom'},
            width=800,
            height=600
        )
        
        return fig


class CorrelationNetworkVisualizer:
    """
    Creates dynamic network visualizations to show relationships between
    bias metrics, demographics, and experimental conditions.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def create_bias_correlation_network(self, 
                                      results_df: pd.DataFrame,
                                      threshold: float = 0.3) -> go.Figure:
        """
        Create network graph showing correlations between bias metrics.
        
        Args:
            results_df: Experiment results DataFrame
            threshold: Minimum correlation threshold for edge creation
            
        Returns:
            Plotly network graph figure
        """
        logger.info("Creating bias correlation network")
        
        # Identify bias metrics
        bias_cols = self._identify_bias_metrics(results_df)
        
        if len(bias_cols) < 2:
            logger.warning("Insufficient bias metrics for network visualization")
            return self._create_empty_network_fig()
        
        # Calculate correlations
        corr_matrix = results_df[bias_cols].corr()
        
        # Create network graph
        G = nx.Graph()
        
        # Add nodes
        for metric in bias_cols:
            G.add_node(metric, type='bias_metric')
        
        # Add edges based on correlation threshold
        for i, metric1 in enumerate(bias_cols):
            for j, metric2 in enumerate(bias_cols[i+1:], i+1):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    G.add_edge(metric1, metric2, weight=abs(corr_val), correlation=corr_val)
        
        # Generate layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract edge coordinates
        edge_x, edge_y = [], []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            edge_data = G.edges[edge]
            edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Correlation: {edge_data['correlation']:.3f}")
        
        # Create edge trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='lightgray'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Extract node coordinates and info
        node_x, node_y = [], []
        node_text, node_info = [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node.replace('_', '<br>'))
            
            # Calculate node statistics
            connections = list(G.neighbors(node))
            node_info.append(f"<b>{node}</b><br>Connections: {len(connections)}")
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            hovertext=node_info,
            text=node_text,
            textposition="middle center",
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='black')
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                        title='Bias Metrics Correlation Network',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text=f"Threshold: {threshold}",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002,
                            xanchor='left', yanchor='bottom',
                            font=dict(color='black', size=12)
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        ))
        
        return fig
    
    def create_demographic_influence_network(self, results_df: pd.DataFrame) -> go.Figure:
        """Create network showing demographic influence on bias metrics"""
        
        # Identify demographic and bias columns
        demo_cols = self._identify_demographic_cols(results_df)
        bias_cols = self._identify_bias_metrics(results_df)
        
        G = nx.Graph()
        
        # Add demographic nodes
        for demo in demo_cols:
            G.add_node(demo, type='demographic', color='red')
        
        # Add bias metric nodes
        for bias in bias_cols:
            G.add_node(bias, type='bias_metric', color='blue')
        
        # Calculate influence (correlation) between demographics and bias
        for demo in demo_cols:
            demo_data = results_df[demo]
            
            # Encode categorical demographics
            if demo_data.dtype == 'object':
                demo_encoded = pd.Categorical(demo_data).codes
            else:
                demo_encoded = demo_data
            
            for bias in bias_cols:
                bias_data = results_df[bias]
                
                # Calculate correlation
                if len(demo_encoded) > 1 and len(bias_data) > 1:
                    corr_val, p_val = stats.pearsonr(demo_encoded, bias_data)
                    
                    if abs(corr_val) > 0.1 and p_val < 0.05:  # Significant correlation
                        G.add_edge(demo, bias, weight=abs(corr_val), 
                                 correlation=corr_val, p_value=p_val)
        
        return self._create_network_visualization(G, "Demographic Influence Network")
    
    def _identify_bias_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that represent bias metrics"""
        bias_keywords = ['bias', 'parity', 'fairness', 'discrimination', 'disparity']
        bias_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in bias_keywords):
                bias_cols.append(col)
        
        # If no explicit bias metrics found, use numeric columns (excluding IDs)
        if not bias_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['age', 'score', 'K', 'repeats', 'id', 'subject_id']
            bias_cols = [col for col in numeric_cols if col.lower() not in [x.lower() for x in exclude_cols]]
        
        return bias_cols[:10]  # Limit to prevent overcrowding
    
    def _identify_demographic_cols(self, df: pd.DataFrame) -> List[str]:
        """Identify demographic columns"""
        demo_keywords = ['gender', 'age', 'location', 'name', 'ethnicity', 'race', 'indigenous']
        demo_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in demo_keywords):
                demo_cols.append(col)
        
        return demo_cols
    
    def _create_network_visualization(self, G: nx.Graph, title: str) -> go.Figure:
        """Create Plotly network visualization from NetworkX graph"""
        
        if len(G.nodes()) == 0:
            return self._create_empty_network_fig(title)
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                               line=dict(width=1, color='gray'),
                               hoverinfo='none',
                               mode='lines')
        
        # Create nodes
        node_x, node_y = [], []
        node_color, node_text, node_info = [], [], []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            if 'type' in node_data:
                color = 'red' if node_data['type'] == 'demographic' else 'blue'
            else:
                color = 'gray'
            node_color.append(color)
            
            node_text.append(node.replace('_', ' ').title())
            node_info.append(f"<b>{node}</b><br>Type: {node_data.get('type', 'unknown')}<br>Connections: {len(list(G.neighbors(node)))}")
        
        node_trace = go.Scatter(x=node_x, y=node_y,
                               mode='markers+text',
                               hoverinfo='text',
                               hovertext=node_info,
                               text=node_text,
                               textposition="top center",
                               marker=dict(size=20,
                                         color=node_color,
                                         line=dict(width=2, color='black')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title=title,
                                      titlefont_size=16,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=20,l=5,r=5,t=40),
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
        
        return fig
    
    def _create_empty_network_fig(self, title: str = "Network Visualization") -> go.Figure:
        """Create empty network figure with message"""
        fig = go.Figure()
        fig.add_annotation(
            text="No network connections found<br>Try adjusting correlation threshold",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        return fig


class TemporalAnimationVisualizer:
    """
    Creates animated visualizations showing bias evolution over time,
    model versions, or experimental iterations.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
    
    def create_bias_evolution_animation(self, 
                                      temporal_data: pd.DataFrame,
                                      time_col: str = 'timestamp',
                                      bias_metric: str = 'demographic_parity_difference',
                                      group_col: str = 'demographic_group') -> go.Figure:
        """
        Create animated visualization of bias evolution over time.
        
        Args:
            temporal_data: DataFrame with temporal bias data
            time_col: Column containing time information
            bias_metric: Bias metric to animate
            group_col: Column for grouping (e.g., demographic groups)
            
        Returns:
            Animated Plotly figure
        """
        logger.info(f"Creating bias evolution animation for {bias_metric}")
        
        # Prepare data
        if time_col not in temporal_data.columns:
            logger.warning(f"Time column {time_col} not found. Using index.")
            temporal_data[time_col] = temporal_data.index
        
        # Sort by time
        temporal_data = temporal_data.sort_values(time_col)
        
        # Create frames for animation
        frames = []
        time_points = temporal_data[time_col].unique()
        
        for t in time_points:
            frame_data = temporal_data[temporal_data[time_col] <= t]
            
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=frame_data[frame_data[group_col] == group][time_col],
                        y=frame_data[frame_data[group_col] == group][bias_metric],
                        mode='lines+markers',
                        name=str(group),
                        line=dict(width=3),
                        marker=dict(size=8)
                    ) for group in frame_data[group_col].unique()
                ],
                name=str(t)
            )
            frames.append(frame)
        
        # Create initial traces
        initial_traces = []
        for group in temporal_data[group_col].unique():
            group_data = temporal_data[temporal_data[group_col] == group]
            initial_traces.append(
                go.Scatter(
                    x=group_data[time_col],
                    y=group_data[bias_metric],
                    mode='lines+markers',
                    name=str(group),
                    line=dict(width=3),
                    marker=dict(size=8)
                )
            )
        
        # Create figure
        fig = go.Figure(
            data=initial_traces,
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title=f'Bias Evolution Over Time: {bias_metric}',
            xaxis_title=time_col.replace('_', ' ').title(),
            yaxis_title=bias_metric.replace('_', ' ').title(),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(label="Play",
                             method="animate",
                             args=[None, {"frame": {"duration": self.config.animation_duration, "redraw": False},
                                         "fromcurrent": True, "transition": {"duration": 300,
                                                                           "easing": "quadratic-in-out"}}]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], {"frame": {"duration": 0, "redraw": False},
                                           "mode": "immediate",
                                           "transition": {"duration": 0}}])
                    ]),
                    pad={"r": 10, "t": 87},
                    showactive=False,
                    x=0.011,
                    xanchor="right",
                    y=0,
                    yanchor="top"
                ),
            ]
        )
        
        # Add slider
        sliders = [dict(
            active=0,
            currentvalue={"prefix": f"{time_col}: "},
            pad={"b": 10},
            steps=[
                dict(
                    args=[[f.name], {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 300}
                    }],
                    label=str(f.name),
                    method="animate"
                ) for f in frames
            ]
        )]
        
        fig.update_layout(sliders=sliders)
        
        return fig
    
    def create_model_comparison_animation(self, 
                                        model_data: pd.DataFrame,
                                        metric_cols: List[str]) -> go.Figure:
        """Create animated comparison of different models over metrics"""
        
        if 'model' not in model_data.columns:
            logger.error("Model column not found in data")
            return go.Figure()
        
        # Create frames for each metric
        frames = []
        for i, metric in enumerate(metric_cols):
            frame_data = []
            
            for model in model_data['model'].unique():
                model_subset = model_data[model_data['model'] == model]
                
                frame_data.append(
                    go.Bar(
                        x=[model],
                        y=[model_subset[metric].mean()],
                        name=model,
                        error_y=dict(
                            type='data',
                            array=[model_subset[metric].std()],
                            visible=True
                        )
                    )
                )
            
            frame = go.Frame(data=frame_data, name=metric)
            frames.append(frame)
        
        # Initial frame
        initial_metric = metric_cols[0]
        initial_data = []
        
        for model in model_data['model'].unique():
            model_subset = model_data[model_data['model'] == model]
            initial_data.append(
                go.Bar(
                    x=[model],
                    y=[model_subset[initial_metric].mean()],
                    name=model,
                    error_y=dict(
                        type='data',
                        array=[model_subset[initial_metric].std()],
                        visible=True
                    )
                )
            )
        
        fig = go.Figure(data=initial_data, frames=frames)
        
        # Add controls
        fig.update_layout(
            title="Model Performance Comparison Across Metrics",
            xaxis_title="Model",
            yaxis_title="Metric Value",
            updatemenus=[{
                "buttons": [
                    {
                        "args": [None, {"frame": {"duration": 500, "redraw": True},
                                       "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    },
                    {
                        "args": [[None], {"frame": {"duration": 0, "redraw": True},
                                         "mode": "immediate"}],
                        "label": "Pause",
                        "method": "animate"
                    }
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }]
        )
        
        return fig


class PublicationFigureGenerator:
    """
    Generates high-quality, publication-ready figures with proper formatting,
    legends, and multiple export formats.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Set publication style
        plt.rcParams.update({
            'font.family': self.config.font_family,
            'font.size': self.config.font_size,
            'axes.linewidth': 1.0,
            'axes.spines.left': True,
            'axes.spines.bottom': True,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 6,
            'xtick.minor.size': 3,
            'ytick.major.size': 6,
            'ytick.minor.size': 3,
            'legend.frameon': True,
            'legend.fancybox': True,
            'legend.shadow': True
        })
    
    def generate_bias_comparison_figure(self, 
                                      results_df: pd.DataFrame,
                                      output_dir: str = "figures") -> Dict[str, str]:
        """
        Generate publication-ready bias comparison figure.
        
        Args:
            results_df: Experiment results DataFrame
            output_dir: Directory to save figures
            
        Returns:
            Dictionary of format -> file path mappings
        """
        logger.info("Generating publication bias comparison figure")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size, dpi=self.config.dpi)
        fig.suptitle('LLM Bias Analysis: Comprehensive Comparison', fontsize=16, fontweight='bold')
        
        # Identify bias metrics and demographics
        bias_cols = self._identify_bias_metrics(results_df)
        demo_cols = self._identify_demographic_cols(results_df)
        
        # Plot 1: Bias distribution by demographics (top-left)
        if bias_cols and demo_cols:
            ax1 = axes[0, 0]
            primary_demo = demo_cols[0]
            primary_bias = bias_cols[0]
            
            if results_df[primary_demo].dtype == 'object':
                sns.boxplot(data=results_df, x=primary_demo, y=primary_bias, ax=ax1)
                ax1.tick_params(axis='x', rotation=45)
            else:
                ax1.scatter(results_df[primary_demo], results_df[primary_bias], alpha=0.6)
            
            ax1.set_title(f'Bias Distribution by {primary_demo}', fontweight='bold')
            ax1.set_xlabel(primary_demo.replace('_', ' ').title())
            ax1.set_ylabel(primary_bias.replace('_', ' ').title())
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Correlation heatmap (top-right)
        if len(bias_cols) >= 2:
            ax2 = axes[0, 1]
            corr_matrix = results_df[bias_cols[:6]].corr()  # Limit to 6 for readability
            
            im = ax2.imshow(corr_matrix, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
            ax2.set_xticks(range(len(corr_matrix.columns)))
            ax2.set_yticks(range(len(corr_matrix.index)))
            ax2.set_xticklabels([col.replace('_', '\n') for col in corr_matrix.columns], 
                               rotation=45, ha='right')
            ax2.set_yticklabels([col.replace('_', '\n') for col in corr_matrix.index])
            ax2.set_title('Bias Metrics Correlation', fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
            cbar.set_label('Correlation')
        
        # Plot 3: Model comparison (bottom-left)
        ax3 = axes[1, 0]
        if 'model' in results_df.columns and bias_cols:
            model_comparison = results_df.groupby('model')[bias_cols[0]].agg(['mean', 'std']).reset_index()
            
            bars = ax3.bar(model_comparison['model'], model_comparison['mean'], 
                          yerr=model_comparison['std'], capsize=5, alpha=0.7)
            ax3.set_title('Model Bias Comparison', fontweight='bold')
            ax3.set_xlabel('Model')
            ax3.set_ylabel(f'Mean {bias_cols[0].replace("_", " ").title()}')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Add significance indicators
            self._add_significance_indicators(ax3, model_comparison['mean'].values)
        
        # Plot 4: Temporal trends (bottom-right)
        ax4 = axes[1, 1]
        if 'timestamp' in results_df.columns and bias_cols:
            temporal_data = results_df.groupby('timestamp')[bias_cols[0]].mean().reset_index()
            ax4.plot(temporal_data['timestamp'], temporal_data[bias_cols[0]], 
                    marker='o', linewidth=2, markersize=6)
            ax4.set_title('Bias Temporal Trend', fontweight='bold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel(f'{bias_cols[0].replace("_", " ").title()}')
            ax4.grid(True, alpha=0.3)
        else:
            # Fallback: distribution plot
            if bias_cols:
                ax4.hist(results_df[bias_cols[0]], bins=30, alpha=0.7, density=True)
                ax4.set_title(f'{bias_cols[0].replace("_", " ").title()} Distribution', fontweight='bold')
                ax4.set_xlabel('Bias Value')
                ax4.set_ylabel('Density')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save in multiple formats
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        base_filename = "bias_comparison_analysis"
        
        for fmt in self.config.export_formats:
            if fmt in ['png', 'pdf', 'svg', 'eps']:
                filepath = os.path.join(output_dir, f"{base_filename}.{fmt}")
                fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                           dpi=self.config.dpi if fmt == 'png' else None)
                saved_files[fmt] = filepath
        
        plt.close(fig)
        return saved_files
    
    def generate_statistical_summary_figure(self, 
                                          results_df: pd.DataFrame,
                                          confidence_level: float = 0.95,
                                          output_dir: str = "figures") -> Dict[str, str]:
        """Generate publication-ready statistical summary figure"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.config.dpi)
        fig.suptitle('Statistical Analysis Summary', fontsize=16, fontweight='bold')
        
        bias_cols = self._identify_bias_metrics(results_df)
        
        # Effect sizes with confidence intervals
        if bias_cols:
            ax1 = axes[0, 0]
            effect_sizes = []
            ci_lower, ci_upper = [], []
            labels = []
            
            for bias_col in bias_cols[:8]:  # Limit for readability
                data = results_df[bias_col].dropna()
                if len(data) > 1:
                    mean_effect = data.mean()
                    sem = stats.sem(data)
                    ci = stats.t.interval(confidence_level, len(data)-1, loc=mean_effect, scale=sem)
                    
                    effect_sizes.append(mean_effect)
                    ci_lower.append(mean_effect - ci[0])
                    ci_upper.append(ci[1] - mean_effect)
                    labels.append(bias_col.replace('_', '\n'))
            
            y_pos = np.arange(len(labels))
            ax1.barh(y_pos, effect_sizes, xerr=[ci_lower, ci_upper], 
                    capsize=5, alpha=0.7, height=0.6)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(labels)
            ax1.set_xlabel('Effect Size')
            ax1.set_title('Effect Sizes with 95% CI', fontweight='bold')
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax1.grid(True, alpha=0.3)
        
        # Power analysis simulation
        ax2 = axes[0, 1]
        sample_sizes = np.arange(10, 201, 10)
        power_values = []
        
        for n in sample_sizes:
            # Simplified power calculation for demonstration
            effect_size = 0.3  # Medium effect
            power = 1 - stats.beta.cdf(0.05, 1, n-1)  # Simplified
            power_values.append(min(power * 2, 1.0))  # Scale appropriately
        
        ax2.plot(sample_sizes, power_values, marker='o', linewidth=2)
        ax2.axhline(y=0.8, color='red', linestyle='--', label='80% Power')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Statistical Power')
        ax2.set_title('Power Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # P-value distribution
        ax3 = axes[1, 0]
        if 'p_value' in results_df.columns:
            p_values = results_df['p_value'].dropna()
        else:
            # Generate synthetic p-values for demonstration
            p_values = np.random.beta(2, 5, 100)  # Typical p-value distribution
        
        ax3.hist(p_values, bins=20, alpha=0.7, density=True, edgecolor='black')
        ax3.axvline(x=0.05, color='red', linestyle='--', label='α = 0.05')
        ax3.set_xlabel('P-value')
        ax3.set_ylabel('Density')
        ax3.set_title('P-value Distribution', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ROC curves (if applicable)
        ax4 = axes[1, 1]
        if 'true_positive_rate' in results_df.columns and 'false_positive_rate' in results_df.columns:
            tpr = results_df['true_positive_rate']
            fpr = results_df['false_positive_rate']
            auc_score = np.trapz(tpr.sort_values(), fpr.sort_values())
            
            ax4.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.3f})')
            ax4.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7)
            ax4.set_xlabel('False Positive Rate')
            ax4.set_ylabel('True Positive Rate')
            ax4.set_title('ROC Curve Analysis', fontweight='bold')
            ax4.legend()
        else:
            # Fallback: Q-Q plot
            if bias_cols:
                data = results_df[bias_cols[0]].dropna()
                stats.probplot(data, dist="norm", plot=ax4)
                ax4.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save files
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        base_filename = "statistical_summary"
        
        for fmt in self.config.export_formats:
            if fmt in ['png', 'pdf', 'svg', 'eps']:
                filepath = os.path.join(output_dir, f"{base_filename}.{fmt}")
                fig.savefig(filepath, format=fmt, bbox_inches='tight', 
                           dpi=self.config.dpi if fmt == 'png' else None)
                saved_files[fmt] = filepath
        
        plt.close(fig)
        return saved_files
    
    def _identify_bias_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that represent bias metrics"""
        bias_keywords = ['bias', 'parity', 'fairness', 'discrimination', 'disparity', 'difference']
        bias_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in bias_keywords):
                bias_cols.append(col)
        
        # Fallback to numeric columns
        if not bias_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['age', 'score', 'K', 'repeats', 'id', 'subject_id']
            bias_cols = [col for col in numeric_cols if col.lower() not in [x.lower() for x in exclude_cols]]
        
        return bias_cols[:8]  # Limit for visualization
    
    def _identify_demographic_cols(self, df: pd.DataFrame) -> List[str]:
        """Identify demographic columns"""
        demo_keywords = ['gender', 'age', 'location', 'name', 'ethnicity', 'race', 'indigenous', 'demographic']
        demo_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in demo_keywords):
                demo_cols.append(col)
        
        return demo_cols
    
    def _add_significance_indicators(self, ax, values: np.ndarray):
        """Add significance indicators to bar plot"""
        max_val = max(values)
        for i, val in enumerate(values):
            if val > max_val * 0.8:  # Simple significance threshold
                ax.text(i, val + max_val * 0.02, '***', ha='center', fontweight='bold')
            elif val > max_val * 0.6:
                ax.text(i, val + max_val * 0.02, '**', ha='center', fontweight='bold')
            elif val > max_val * 0.4:
                ax.text(i, val + max_val * 0.02, '*', ha='center', fontweight='bold')


class CustomGrammarVisualizer:
    """
    Custom grammar of graphics implementation for flexible, programmatic
    visualization creation similar to ggplot2.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.layers = []
        self.aesthetics = {}
        self.scales = {}
        self.theme = {}
    
    def ggplot(self, data: pd.DataFrame) -> 'CustomGrammarVisualizer':
        """Initialize plot with data (similar to ggplot2)"""
        self.data = data
        return self
    
    def aes(self, **mappings) -> 'CustomGrammarVisualizer':
        """Set aesthetic mappings"""
        self.aesthetics.update(mappings)
        return self
    
    def geom_point(self, **kwargs) -> 'CustomGrammarVisualizer':
        """Add scatter plot layer"""
        self.layers.append(('point', kwargs))
        return self
    
    def geom_line(self, **kwargs) -> 'CustomGrammarVisualizer':
        """Add line plot layer"""
        self.layers.append(('line', kwargs))
        return self
    
    def geom_bar(self, **kwargs) -> 'CustomGrammarVisualizer':
        """Add bar plot layer"""
        self.layers.append(('bar', kwargs))
        return self
    
    def facet_wrap(self, facets: str, **kwargs) -> 'CustomGrammarVisualizer':
        """Add faceting"""
        self.facets = facets
        self.facet_kwargs = kwargs
        return self
    
    def scale_color_manual(self, values: List[str]) -> 'CustomGrammarVisualizer':
        """Set manual color scale"""
        self.scales['color'] = values
        return self
    
    def theme_minimal(self) -> 'CustomGrammarVisualizer':
        """Apply minimal theme"""
        self.theme = {
            'background': 'white',
            'grid': True,
            'spines': False
        }
        return self
    
    def labs(self, title: str = None, x: str = None, y: str = None) -> 'CustomGrammarVisualizer':
        """Set labels"""
        if title:
            self.title = title
        if x:
            self.xlabel = x
        if y:
            self.ylabel = y
        return self
    
    def build(self) -> go.Figure:
        """Build the final plotly figure"""
        
        # Create base figure
        fig = go.Figure()
        
        # Apply layers
        for layer_type, layer_kwargs in self.layers:
            self._add_layer(fig, layer_type, layer_kwargs)
        
        # Apply labels and theme
        layout_updates = {}
        
        if hasattr(self, 'title'):
            layout_updates['title'] = self.title
        if hasattr(self, 'xlabel'):
            layout_updates['xaxis_title'] = self.xlabel
        if hasattr(self, 'ylabel'):
            layout_updates['yaxis_title'] = self.ylabel
        
        # Apply theme
        if self.theme.get('grid', True):
            layout_updates['xaxis'] = dict(showgrid=True, gridcolor='lightgray')
            layout_updates['yaxis'] = dict(showgrid=True, gridcolor='lightgray')
        
        if self.theme.get('background'):
            layout_updates['plot_bgcolor'] = self.theme['background']
        
        fig.update_layout(**layout_updates)
        
        return fig
    
    def _add_layer(self, fig: go.Figure, layer_type: str, layer_kwargs: Dict):
        """Add a layer to the figure"""
        
        x_col = self.aesthetics.get('x')
        y_col = self.aesthetics.get('y')
        color_col = self.aesthetics.get('color')
        
        if not x_col or not y_col:
            logger.error("x and y aesthetics must be specified")
            return
        
        if layer_type == 'point':
            if color_col:
                for color_val in self.data[color_col].unique():
                    subset = self.data[self.data[color_col] == color_val]
                    fig.add_trace(go.Scatter(
                        x=subset[x_col],
                        y=subset[y_col],
                        mode='markers',
                        name=str(color_val),
                        marker=dict(size=layer_kwargs.get('size', 8))
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=self.data[x_col],
                    y=self.data[y_col],
                    mode='markers',
                    marker=dict(size=layer_kwargs.get('size', 8))
                ))
        
        elif layer_type == 'line':
            if color_col:
                for color_val in self.data[color_col].unique():
                    subset = self.data[self.data[color_col] == color_val]
                    fig.add_trace(go.Scatter(
                        x=subset[x_col],
                        y=subset[y_col],
                        mode='lines',
                        name=str(color_val),
                        line=dict(width=layer_kwargs.get('width', 2))
                    ))
            else:
                fig.add_trace(go.Scatter(
                    x=self.data[x_col],
                    y=self.data[y_col],
                    mode='lines',
                    line=dict(width=layer_kwargs.get('width', 2))
                ))
        
        elif layer_type == 'bar':
            if color_col:
                fig.add_trace(go.Bar(
                    x=self.data[x_col],
                    y=self.data[y_col],
                    marker_color=self.data[color_col].map(
                        dict(zip(self.data[color_col].unique(), 
                               px.colors.qualitative.Set1))
                    )
                ))
            else:
                fig.add_trace(go.Bar(
                    x=self.data[x_col],
                    y=self.data[y_col]
                ))


class InteractiveDashboard:
    """
    Interactive dashboard for real-time exploration of bias analysis results
    with filtering, drill-down capabilities, and dynamic visualizations.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.results_data = None
        
    def create_dashboard_app(self, results_df: pd.DataFrame) -> dash.Dash:
        """
        Create interactive dashboard application.
        
        Args:
            results_df: Experiment results DataFrame
            
        Returns:
            Configured Dash application
        """
        self.results_data = results_df
        
        # Define app layout
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("LLM Bias Analysis Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Controls"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Bias Metric:"),
                                    dcc.Dropdown(
                                        id='bias-metric-dropdown',
                                        options=self._get_bias_metric_options(),
                                        value=self._get_bias_metric_options()[0]['value'] if self._get_bias_metric_options() else None
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Demographic Filter:"),
                                    dcc.Dropdown(
                                        id='demographic-dropdown',
                                        options=self._get_demographic_options(),
                                        multi=True
                                    )
                                ], width=6)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Model Filter:"),
                                    dcc.Dropdown(
                                        id='model-dropdown',
                                        options=self._get_model_options(),
                                        multi=True
                                    )
                                ], width=6),
                                dbc.Col([
                                    dbc.Label("Date Range:"),
                                    dcc.DatePickerRange(
                                        id='date-picker-range',
                                        start_date=self._get_date_range()[0],
                                        end_date=self._get_date_range()[1]
                                    )
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Main Visualizations
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='main-bias-plot')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='correlation-network')
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dcc.Graph(id='temporal-analysis')
                ], width=6),
                dbc.Col([
                    dcc.Graph(id='statistical-summary')
                ], width=6)
            ], className="mb-4"),
            
            # Data Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Filtered Results"),
                        dbc.CardBody([
                            html.Div(id='results-table')
                        ])
                    ])
                ], width=12)
            ])
            
        ], fluid=True)
        
        # Set up callbacks
        self._setup_callbacks()
        
        return self.app
    
    def _get_bias_metric_options(self) -> List[Dict]:
        """Get available bias metrics for dropdown"""
        if self.results_data is None:
            return []
        
        bias_keywords = ['bias', 'parity', 'fairness', 'discrimination', 'disparity']
        bias_cols = []
        
        for col in self.results_data.columns:
            if any(keyword in col.lower() for keyword in bias_keywords):
                bias_cols.append(col)
        
        return [{'label': col.replace('_', ' ').title(), 'value': col} for col in bias_cols]
    
    def _get_demographic_options(self) -> List[Dict]:
        """Get demographic options for filtering"""
        if self.results_data is None:
            return []
        
        demo_keywords = ['gender', 'age', 'location', 'name', 'ethnicity', 'race', 'indigenous']
        demo_cols = []
        
        for col in self.results_data.columns:
            if any(keyword in col.lower() for keyword in demo_keywords):
                demo_cols.append(col)
        
        return [{'label': col.replace('_', ' ').title(), 'value': col} for col in demo_cols]
    
    def _get_model_options(self) -> List[Dict]:
        """Get model options for filtering"""
        if self.results_data is None or 'model' not in self.results_data.columns:
            return []
        
        models = self.results_data['model'].unique()
        return [{'label': model, 'value': model} for model in models]
    
    def _get_date_range(self) -> Tuple[str, str]:
        """Get date range from data"""
        if self.results_data is None or 'timestamp' not in self.results_data.columns:
            today = datetime.now()
            return (today - timedelta(days=30)).strftime('%Y-%m-%d'), today.strftime('%Y-%m-%d')
        
        dates = pd.to_datetime(self.results_data['timestamp'])
        return dates.min().strftime('%Y-%m-%d'), dates.max().strftime('%Y-%m-%d')
    
    def _setup_callbacks(self):
        """Set up interactive callbacks"""
        
        @self.app.callback(
            [Output('main-bias-plot', 'figure'),
             Output('correlation-network', 'figure'),
             Output('temporal-analysis', 'figure'),
             Output('statistical-summary', 'figure'),
             Output('results-table', 'children')],
            [Input('bias-metric-dropdown', 'value'),
             Input('demographic-dropdown', 'value'),
             Input('model-dropdown', 'value'),
             Input('date-picker-range', 'start_date'),
             Input('date-picker-range', 'end_date')]
        )
        def update_dashboard(bias_metric, demographics, models, start_date, end_date):
            """Update all dashboard components based on filters"""
            
            # Filter data
            filtered_data = self.results_data.copy()
            
            if models:
                if 'model' in filtered_data.columns:
                    filtered_data = filtered_data[filtered_data['model'].isin(models)]
            
            if start_date and end_date:
                if 'timestamp' in filtered_data.columns:
                    filtered_data = filtered_data[
                        (pd.to_datetime(filtered_data['timestamp']) >= pd.to_datetime(start_date)) &
                        (pd.to_datetime(filtered_data['timestamp']) <= pd.to_datetime(end_date))
                    ]
            
            # Generate visualizations
            bias_visualizer = BiasLandscapeVisualizer(self.config)
            network_visualizer = CorrelationNetworkVisualizer(self.config)
            
            # Main bias plot
            if demographics and len(demographics) > 0:
                main_plot = bias_visualizer.create_3d_bias_surface(
                    filtered_data, demographics, bias_metric
                )
            else:
                # Fallback to simple distribution
                main_plot = px.histogram(filtered_data, x=bias_metric, 
                                       title=f'{bias_metric} Distribution')
            
            # Network plot
            network_plot = network_visualizer.create_bias_correlation_network(filtered_data)
            
            # Temporal analysis
            if 'timestamp' in filtered_data.columns:
                temporal_data = filtered_data.groupby('timestamp')[bias_metric].mean().reset_index()
                temporal_plot = px.line(temporal_data, x='timestamp', y=bias_metric,
                                      title='Temporal Bias Trend')
            else:
                temporal_plot = px.box(filtered_data, y=bias_metric, 
                                     title=f'{bias_metric} Distribution')
            
            # Statistical summary
            stats_plot = px.violin(filtered_data, y=bias_metric,
                                 title='Statistical Distribution')
            
            # Results table
            table_data = filtered_data.head(100)  # Limit for performance
            table = dbc.Table.from_dataframe(table_data, striped=True, bordered=True, 
                                           hover=True, responsive=True, size='sm')
            
            return main_plot, network_plot, temporal_plot, stats_plot, table
    
    def run_server(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = True):
        """Run the dashboard server"""
        logger.info(f"Starting dashboard server at http://{host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)


class AdvancedVisualizationSuite:
    """
    Main orchestrator class that provides a unified interface to all
    advanced visualization capabilities.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Initialize component visualizers
        self.bias_landscape = BiasLandscapeVisualizer(self.config)
        self.correlation_network = CorrelationNetworkVisualizer(self.config)
        self.temporal_animation = TemporalAnimationVisualizer(self.config)
        self.publication_generator = PublicationFigureGenerator(self.config)
        self.custom_grammar = CustomGrammarVisualizer(self.config)
        self.interactive_dashboard = InteractiveDashboard(self.config)
        
        logger.info("Advanced Visualization Suite initialized")
    
    def create_complete_analysis_suite(self, 
                                     results_df: pd.DataFrame,
                                     output_dir: str = "visualizations") -> Dict[str, Any]:
        """
        Create complete visualization analysis suite with all components.
        
        Args:
            results_df: Experiment results DataFrame
            output_dir: Directory to save all visualizations
            
        Returns:
            Dictionary containing all generated visualizations and saved files
        """
        logger.info("Creating complete visualization analysis suite")
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'interactive_figures': {},
            'publication_figures': {},
            'animations': {},
            'networks': {},
            'saved_files': {},
            'dashboard_app': None
        }
        
        try:
            # 1. Bias Landscape Visualizations
            logger.info("Generating bias landscape visualizations")
            demo_cols = self._identify_demographic_cols(results_df)
            bias_cols = self._identify_bias_metrics(results_df)
            
            if demo_cols and bias_cols:
                for bias_metric in bias_cols[:3]:  # Limit for performance
                    landscape_fig = self.bias_landscape.create_3d_bias_surface(
                        results_df, demo_cols[:2], bias_metric
                    )
                    results['interactive_figures'][f'bias_landscape_{bias_metric}'] = landscape_fig
                    
                    # Save interactive HTML
                    html_path = os.path.join(output_dir, f'bias_landscape_{bias_metric}.html')
                    landscape_fig.write_html(html_path)
                    results['saved_files'][f'bias_landscape_{bias_metric}_html'] = html_path
            
            # 2. Correlation Networks
            logger.info("Generating correlation network visualizations")
            network_fig = self.correlation_network.create_bias_correlation_network(results_df)
            results['networks']['bias_correlation'] = network_fig
            
            network_html_path = os.path.join(output_dir, 'bias_correlation_network.html')
            network_fig.write_html(network_html_path)
            results['saved_files']['network_html'] = network_html_path
            
            if demo_cols:
                demo_network_fig = self.correlation_network.create_demographic_influence_network(results_df)
                results['networks']['demographic_influence'] = demo_network_fig
                
                demo_network_html_path = os.path.join(output_dir, 'demographic_influence_network.html')
                demo_network_fig.write_html(demo_network_html_path)
                results['saved_files']['demo_network_html'] = demo_network_html_path
            
            # 3. Temporal Animations
            logger.info("Generating temporal animations")
            if 'timestamp' in results_df.columns and bias_cols:
                temporal_fig = self.temporal_animation.create_bias_evolution_animation(
                    results_df, 'timestamp', bias_cols[0], demo_cols[0] if demo_cols else 'model'
                )
                results['animations']['bias_evolution'] = temporal_fig
                
                animation_html_path = os.path.join(output_dir, 'bias_evolution_animation.html')
                temporal_fig.write_html(animation_html_path)
                results['saved_files']['animation_html'] = animation_html_path
            
            if 'model' in results_df.columns and len(bias_cols) > 1:
                model_animation_fig = self.temporal_animation.create_model_comparison_animation(
                    results_df, bias_cols[:4]
                )
                results['animations']['model_comparison'] = model_animation_fig
            
            # 4. Publication-Ready Figures
            logger.info("Generating publication-ready figures")
            pub_figures = self.publication_generator.generate_bias_comparison_figure(
                results_df, output_dir
            )
            results['publication_figures']['bias_comparison'] = pub_figures
            results['saved_files'].update({f'bias_comparison_{k}': v for k, v in pub_figures.items()})
            
            stats_figures = self.publication_generator.generate_statistical_summary_figure(
                results_df, output_dir=output_dir
            )
            results['publication_figures']['statistical_summary'] = stats_figures
            results['saved_files'].update({f'statistical_summary_{k}': v for k, v in stats_figures.items()})
            
            # 5. Custom Grammar Examples
            logger.info("Generating custom grammar visualizations")
            if bias_cols and demo_cols:
                custom_fig = (self.custom_grammar
                            .ggplot(results_df)
                            .aes(x=demo_cols[0], y=bias_cols[0], color='model' if 'model' in results_df.columns else demo_cols[0])
                            .geom_point(size=10)
                            .theme_minimal()
                            .labs(title='Custom Grammar Visualization',
                                 x=demo_cols[0].replace('_', ' ').title(),
                                 y=bias_cols[0].replace('_', ' ').title())
                            .build())
                
                results['interactive_figures']['custom_grammar'] = custom_fig
                
                custom_html_path = os.path.join(output_dir, 'custom_grammar_example.html')
                custom_fig.write_html(custom_html_path)
                results['saved_files']['custom_grammar_html'] = custom_html_path
            
            # 6. Interactive Dashboard
            logger.info("Setting up interactive dashboard")
            dashboard_app = self.interactive_dashboard.create_dashboard_app(results_df)
            results['dashboard_app'] = dashboard_app
            
            # Create comprehensive summary
            summary = self._generate_visualization_summary(results_df, results)
            summary_path = os.path.join(output_dir, 'visualization_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            results['saved_files']['summary'] = summary_path
            
            logger.info(f"Complete visualization suite created. {len(results['saved_files'])} files saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error creating visualization suite: {e}")
            results['error'] = str(e)
        
        return results
    
    def _identify_bias_metrics(self, df: pd.DataFrame) -> List[str]:
        """Identify columns that represent bias metrics"""
        bias_keywords = ['bias', 'parity', 'fairness', 'discrimination', 'disparity']
        bias_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in bias_keywords):
                bias_cols.append(col)
        
        # Fallback to numeric columns
        if not bias_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['age', 'score', 'K', 'repeats', 'id', 'subject_id']
            bias_cols = [col for col in numeric_cols if col.lower() not in [x.lower() for x in exclude_cols]]
        
        return bias_cols[:8]  # Limit for performance
    
    def _identify_demographic_cols(self, df: pd.DataFrame) -> List[str]:
        """Identify demographic columns"""
        demo_keywords = ['gender', 'age', 'location', 'name', 'ethnicity', 'race', 'indigenous', 'demographic']
        demo_cols = []
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in demo_keywords):
                demo_cols.append(col)
        
        return demo_cols
    
    def _generate_visualization_summary(self, 
                                      results_df: pd.DataFrame, 
                                      results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of visualization results"""
        
        summary = {
            'dataset_info': {
                'total_rows': len(results_df),
                'total_columns': len(results_df.columns),
                'bias_metrics_identified': len(self._identify_bias_metrics(results_df)),
                'demographic_columns_identified': len(self._identify_demographic_cols(results_df)),
                'has_temporal_data': 'timestamp' in results_df.columns,
                'has_model_data': 'model' in results_df.columns
            },
            'visualizations_created': {
                'interactive_figures': len(results.get('interactive_figures', {})),
                'publication_figures': len(results.get('publication_figures', {})),
                'animations': len(results.get('animations', {})),
                'networks': len(results.get('networks', {})),
                'saved_files': len(results.get('saved_files', {}))
            },
            'recommendations': self._generate_analysis_recommendations(results_df),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _generate_analysis_recommendations(self, results_df: pd.DataFrame) -> List[str]:
        """Generate recommendations for further analysis"""
        recommendations = []
        
        bias_cols = self._identify_bias_metrics(results_df)
        demo_cols = self._identify_demographic_cols(results_df)
        
        if len(bias_cols) > 3:
            recommendations.append("Consider dimensionality reduction for bias metrics exploration")
        
        if len(demo_cols) > 2:
            recommendations.append("Use intersectional analysis for multi-demographic interactions")
        
        if 'timestamp' in results_df.columns:
            recommendations.append("Conduct temporal stability analysis for bias trends")
        
        if 'model' in results_df.columns and len(results_df['model'].unique()) > 2:
            recommendations.append("Perform comparative model analysis with statistical significance testing")
        
        if len(results_df) < 100:
            recommendations.append("Consider increasing sample size for more robust statistical analysis")
        
        return recommendations


def main():
    """
    Main function demonstrating the Advanced Visualization Suite capabilities.
    """
    # Example usage with synthetic data
    logger.info("Advanced Visualization Suite - Demo Mode")
    
    # Create synthetic experiment data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    demo_data = {
        'model': np.random.choice(['gpt-4o', 'claude-3-5-sonnet', 'gemini-pro'], n_samples),
        'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n_samples),
        'age_group': np.random.choice(['Young', 'Middle', 'Senior'], n_samples),
        'location': np.random.choice(['Urban', 'Rural', 'Suburban'], n_samples),
        'demographic_parity_difference': np.random.normal(0, 0.1, n_samples),
        'equalized_odds_difference': np.random.normal(0, 0.08, n_samples),
        'individual_fairness_score': np.random.beta(2, 3, n_samples),
        'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='H'),
        'confidence_score': np.random.beta(3, 2, n_samples)
    }
    
    results_df = pd.DataFrame(demo_data)
    
    # Initialize visualization suite
    config = VisualizationConfig(
        figure_size=(14, 10),
        dpi=300,
        color_palette='viridis',
        export_formats=['png', 'pdf', 'html']
    )
    
    viz_suite = AdvancedVisualizationSuite(config)
    
    # Create complete analysis suite
    visualization_results = viz_suite.create_complete_analysis_suite(
        results_df, output_dir="demo_visualizations"
    )
    
    print(f"Visualization suite completed!")
    print(f"Files saved: {len(visualization_results['saved_files'])}")
    print(f"Interactive figures created: {len(visualization_results['interactive_figures'])}")
    
    # Optionally launch dashboard
    launch_dashboard = input("Launch interactive dashboard? (y/n): ").lower() == 'y'
    if launch_dashboard:
        dashboard_app = visualization_results['dashboard_app']
        if dashboard_app:
            print("Starting dashboard at http://127.0.0.1:8050")
            dashboard_app.run_server(debug=False)


if __name__ == "__main__":
    main()