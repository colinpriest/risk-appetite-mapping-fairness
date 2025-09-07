#!/usr/bin/env python3
"""
Interactive Web Interface for LLM Risk Fairness Experiments

This module provides a web-based interface for designing, configuring, and monitoring
LLM risk fairness experiments. Built with Flask and Dash for interactive visualization.
"""

import os
import json
import yaml
import threading
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from flask import Flask, request, jsonify, render_template_string
import flask_cors

# Import experiment modules
from llm_risk_fairness_experiment import ExperimentConfig, ExperimentProgress
from advanced_analytics import UncertaintyQuantifier, ModelCalibrator, IndividualFairnessAnalyzer, IntersectionalBiasAnalyzer
from temporal_analysis import TemporalBiasAnalyzer, ModelVersionTracker


class ExperimentWebInterface:
    """Web interface for experiment design and monitoring."""
    
    def __init__(self, host="127.0.0.1", port=8050):
        self.host = host
        self.port = port
        self.flask_app = Flask(__name__)
        self.flask_app.config['SECRET_KEY'] = 'experiment-web-interface-secret'
        flask_cors.CORS(self.flask_app)
        
        # Initialize Dash app
        self.dash_app = dash.Dash(
            __name__,
            server=self.flask_app,
            url_base_pathname='/dashboard/',
            external_stylesheets=[dbc.themes.BOOTSTRAP]
        )
        
        self.running_experiments = {}  # Track running experiments
        self.experiment_configs = {}   # Store experiment configurations
        self.setup_routes()
        self.setup_dashboard()
    
    def setup_routes(self):
        """Setup Flask routes for API endpoints."""
        
        @self.flask_app.route('/')
        def index():
            """Main interface page."""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>LLM Risk Fairness Experiment Interface</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
                <style>
                    .experiment-card { margin: 10px 0; }
                    .status-running { color: #28a745; }
                    .status-completed { color: #007bff; }
                    .status-failed { color: #dc3545; }
                </style>
            </head>
            <body>
                <div class="container mt-4">
                    <h1>LLM Risk Fairness Experiment Interface</h1>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Quick Actions</h3>
                                </div>
                                <div class="card-body">
                                    <a href="/dashboard/" class="btn btn-primary btn-lg">
                                        📊 Live Dashboard
                                    </a>
                                    <a href="/config" class="btn btn-success btn-lg ms-2">
                                        ⚙️ Configure Experiment
                                    </a>
                                    <a href="/results" class="btn btn-info btn-lg ms-2">
                                        📈 View Results
                                    </a>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="card">
                                <div class="card-header">
                                    <h3>Running Experiments</h3>
                                </div>
                                <div class="card-body" id="experiments-list">
                                    <p class="text-muted">No experiments currently running</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
                <script>
                    // Auto-refresh experiment list every 5 seconds
                    setInterval(() => {
                        fetch('/api/experiments')
                            .then(response => response.json())
                            .then(data => updateExperimentsList(data));
                    }, 5000);
                    
                    function updateExperimentsList(experiments) {
                        const container = document.getElementById('experiments-list');
                        if (Object.keys(experiments).length === 0) {
                            container.innerHTML = '<p class="text-muted">No experiments currently running</p>';
                            return;
                        }
                        
                        let html = '';
                        for (const [id, exp] of Object.entries(experiments)) {
                            html += `<div class="experiment-card">
                                <strong>${exp.name}</strong> 
                                <span class="status-${exp.status}">${exp.status}</span>
                                <br><small>${exp.progress}% complete</small>
                            </div>`;
                        }
                        container.innerHTML = html;
                    }
                </script>
            </body>
            </html>
            """)
        
        @self.flask_app.route('/config')
        def config_page():
            """Experiment configuration page."""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Configure Experiment</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <h1>Configure Experiment</h1>
                    <form id="config-form">
                        <div class="row">
                            <div class="col-md-6">
                                <h3>Basic Settings</h3>
                                <div class="mb-3">
                                    <label class="form-label">Experiment Name</label>
                                    <input type="text" class="form-control" name="name" value="experiment_{{ timestamp }}">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Number of Subjects (K)</label>
                                    <input type="number" class="form-control" name="K" value="20" min="1" max="1000">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Repeats per Subject</label>
                                    <input type="number" class="form-control" name="repeats" value="5" min="1" max="20">
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Max Total Cost ($)</label>
                                    <input type="number" class="form-control" name="max_cost" value="50.0" step="0.01">
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h3>Model Selection</h3>
                                <div class="mb-3">
                                    <label class="form-label">Models to Test</label>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="models" value="gpt-4o" checked>
                                        <label class="form-check-label">GPT-4o</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="models" value="gpt-4o-mini">
                                        <label class="form-check-label">GPT-4o Mini</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="models" value="claude-3-5-sonnet-20241022">
                                        <label class="form-check-label">Claude 3.5 Sonnet</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="models" value="claude-3-5-haiku-20241022">
                                        <label class="form-check-label">Claude 3.5 Haiku</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="models" value="gemini-1.5-pro-002">
                                        <label class="form-check-label">Gemini 1.5 Pro</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <h3>Demographics</h3>
                                <div class="mb-3">
                                    <label class="form-label">Gender Distribution</label>
                                    <input type="range" class="form-range" name="gender_balance" min="0" max="100" value="50">
                                    <small class="form-text">Male/Female balance (%)</small>
                                </div>
                                <div class="mb-3">
                                    <label class="form-label">Age Range</label>
                                    <select class="form-select" name="age_range">
                                        <option value="all">All ages (25-65)</option>
                                        <option value="young">Young (25-35)</option>
                                        <option value="middle">Middle (35-50)</option>
                                        <option value="senior">Senior (50-65)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h3>Advanced Options</h3>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" name="stratified_sampling" checked>
                                    <label class="form-check-label">Use Stratified Sampling</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" name="validate_responses" checked>
                                    <label class="form-check-label">Validate Model Responses</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" name="use_cache">
                                    <label class="form-check-label">Use Response Cache</label>
                                </div>
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" name="temporal_analysis">
                                    <label class="form-check-label">Enable Temporal Analysis</label>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mt-4">
                            <button type="submit" class="btn btn-success btn-lg">Start Experiment</button>
                            <button type="button" class="btn btn-secondary btn-lg ms-2" onclick="previewConfig()">Preview Config</button>
                        </div>
                    </form>
                    
                    <div id="config-preview" class="mt-4" style="display: none;">
                        <h3>Configuration Preview</h3>
                        <pre id="config-yaml" class="bg-light p-3"></pre>
                    </div>
                </div>
                
                <script>
                    document.getElementById('config-form').addEventListener('submit', async (e) => {
                        e.preventDefault();
                        const formData = new FormData(e.target);
                        const config = Object.fromEntries(formData.entries());
                        
                        // Handle checkboxes
                        config.models = Array.from(document.querySelectorAll('input[name="models"]:checked')).map(cb => cb.value);
                        config.stratified_sampling = document.querySelector('input[name="stratified_sampling"]').checked;
                        config.validate_responses = document.querySelector('input[name="validate_responses"]').checked;
                        config.use_cache = document.querySelector('input[name="use_cache"]').checked;
                        config.temporal_analysis = document.querySelector('input[name="temporal_analysis"]').checked;
                        
                        const response = await fetch('/api/start_experiment', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify(config)
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            alert('Experiment started! ID: ' + result.experiment_id);
                            window.location.href = '/dashboard/';
                        } else {
                            alert('Error starting experiment');
                        }
                    });
                    
                    function previewConfig() {
                        const formData = new FormData(document.getElementById('config-form'));
                        const config = Object.fromEntries(formData.entries());
                        config.models = Array.from(document.querySelectorAll('input[name="models"]:checked')).map(cb => cb.value);
                        
                        document.getElementById('config-yaml').textContent = JSON.stringify(config, null, 2);
                        document.getElementById('config-preview').style.display = 'block';
                    }
                    
                    // Set timestamp in experiment name
                    document.querySelector('input[name="name"]').value = 'experiment_' + Date.now();
                </script>
            </body>
            </html>
            """)
        
        @self.flask_app.route('/api/experiments')
        def get_experiments():
            """Get current experiments status."""
            return jsonify(self.running_experiments)
        
        @self.flask_app.route('/api/start_experiment', methods=['POST'])
        def start_experiment():
            """Start a new experiment."""
            try:
                config_data = request.json
                experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Create experiment config
                config = ExperimentConfig(
                    K=int(config_data.get('K', 20)),
                    repeats=int(config_data.get('repeats', 5)),
                    models=config_data.get('models', ['gpt-4o']),
                    max_total_cost=float(config_data.get('max_cost', 50.0)),
                    stratified_sampling=config_data.get('stratified_sampling', True),
                    validate_responses=config_data.get('validate_responses', True),
                    use_cache=config_data.get('use_cache', True)
                )
                
                # Save config
                config_path = f"web_experiments/{experiment_id}_config.yaml"
                os.makedirs("web_experiments", exist_ok=True)
                config.save_yaml(config_path)
                
                # Start experiment in background
                self._start_experiment_process(experiment_id, config_path, config_data)
                
                return jsonify({
                    'success': True,
                    'experiment_id': experiment_id,
                    'message': 'Experiment started successfully'
                })
            
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)}), 500
        
        @self.flask_app.route('/results')
        def results_page():
            """Results browser page."""
            return render_template_string("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Experiment Results</title>
                <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            </head>
            <body>
                <div class="container mt-4">
                    <h1>Experiment Results</h1>
                    <div id="results-list">
                        <p>Loading results...</p>
                    </div>
                </div>
                
                <script>
                    // Load results on page load
                    fetch('/api/results')
                        .then(response => response.json())
                        .then(data => displayResults(data));
                    
                    function displayResults(results) {
                        const container = document.getElementById('results-list');
                        if (results.length === 0) {
                            container.innerHTML = '<p class="text-muted">No completed experiments found</p>';
                            return;
                        }
                        
                        let html = '<div class="row">';
                        results.forEach(result => {
                            html += `
                                <div class="col-md-6 mb-3">
                                    <div class="card">
                                        <div class="card-header">
                                            <h5>${result.name}</h5>
                                            <small class="text-muted">${result.date}</small>
                                        </div>
                                        <div class="card-body">
                                            <p><strong>Models:</strong> ${result.models.join(', ')}</p>
                                            <p><strong>Subjects:</strong> ${result.subjects}</p>
                                            <p><strong>Cost:</strong> $${result.cost.toFixed(2)}</p>
                                            <a href="/dashboard/?experiment=${result.id}" class="btn btn-primary">View Analysis</a>
                                        </div>
                                    </div>
                                </div>
                            `;
                        });
                        html += '</div>';
                        container.innerHTML = html;
                    }
                </script>
            </body>
            </html>
            """)
        
        @self.flask_app.route('/api/results')
        def get_results():
            """Get completed experiment results."""
            results = []
            results_dirs = ['runs', 'web_experiments']
            
            for results_dir in results_dirs:
                if os.path.exists(results_dir):
                    for exp_dir in os.listdir(results_dir):
                        exp_path = os.path.join(results_dir, exp_dir)
                        if os.path.isdir(exp_path):
                            result_file = os.path.join(exp_path, 'experiment_report.json')
                            if os.path.exists(result_file):
                                try:
                                    with open(result_file, 'r') as f:
                                        report = json.load(f)
                                    results.append({
                                        'id': exp_dir,
                                        'name': report.get('experiment_name', exp_dir),
                                        'date': report.get('timestamp', 'Unknown'),
                                        'models': report.get('models_tested', []),
                                        'subjects': report.get('total_subjects', 0),
                                        'cost': report.get('total_cost', 0)
                                    })
                                except:
                                    continue
            
            return jsonify(sorted(results, key=lambda x: x['date'], reverse=True))
    
    def _start_experiment_process(self, experiment_id: str, config_path: str, config_data: dict):
        """Start experiment in background process."""
        def run_experiment():
            try:
                self.running_experiments[experiment_id] = {
                    'name': config_data.get('name', experiment_id),
                    'status': 'running',
                    'progress': 0,
                    'start_time': datetime.now(timezone.utc).isoformat()
                }
                
                # Run experiment using subprocess
                outdir = f"web_experiments/{experiment_id}"
                cmd = [
                    'python', 'llm_risk_fairness_experiment.py', 'run',
                    '--config', config_path,
                    '--outdir', outdir,
                    '--log-level', 'INFO'
                ]
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Monitor progress
                for line in process.stdout:
                    if 'Progress:' in line:
                        # Extract progress percentage
                        try:
                            pct = float(line.split('%')[0].split()[-1])
                            self.running_experiments[experiment_id]['progress'] = pct
                        except:
                            pass
                
                process.wait()
                
                if process.returncode == 0:
                    self.running_experiments[experiment_id]['status'] = 'completed'
                    self.running_experiments[experiment_id]['progress'] = 100
                else:
                    self.running_experiments[experiment_id]['status'] = 'failed'
                
            except Exception as e:
                self.running_experiments[experiment_id]['status'] = 'failed'
                self.running_experiments[experiment_id]['error'] = str(e)
        
        thread = threading.Thread(target=run_experiment)
        thread.daemon = True
        thread.start()
    
    def setup_dashboard(self):
        """Setup Dash dashboard for real-time monitoring."""
        self.dash_app.layout = dbc.Container([
            dcc.Store(id='current-experiment'),
            dcc.Interval(id='update-interval', interval=5000, n_intervals=0),
            
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("LLM Risk Fairness Experiment Dashboard", className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Experiment selector
            dbc.Row([
                dbc.Col([
                    html.Label("Select Experiment:", className="form-label"),
                    dcc.Dropdown(
                        id='experiment-selector',
                        placeholder="Choose an experiment to analyze...",
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Div(id='experiment-info', className="mt-2")
                ], width=6)
            ]),
            
            # Main content tabs
            dbc.Tabs([
                dbc.Tab(label="📊 Overview", tab_id="overview"),
                dbc.Tab(label="🧠 Model Comparison", tab_id="models"),
                dbc.Tab(label="⚖️ Fairness Analysis", tab_id="fairness"),
                dbc.Tab(label="📈 Temporal Analysis", tab_id="temporal"),
                dbc.Tab(label="🔍 Detailed Results", tab_id="detailed")
            ], id="main-tabs", active_tab="overview", className="mb-3"),
            
            # Tab content
            html.Div(id='tab-content'),
            
        ], fluid=True)
        
        # Callbacks
        @self.dash_app.callback(
            [Output('experiment-selector', 'options'),
             Output('experiment-selector', 'value')],
            [Input('update-interval', 'n_intervals')]
        )
        def update_experiment_list(n):
            """Update list of available experiments."""
            options = []
            current_value = None
            
            # Get completed experiments
            for results_dir in ['runs', 'web_experiments']:
                if os.path.exists(results_dir):
                    for exp_dir in os.listdir(results_dir):
                        exp_path = os.path.join(results_dir, exp_dir)
                        if os.path.isdir(exp_path):
                            result_file = os.path.join(exp_path, 'experiment_report.json')
                            if os.path.exists(result_file):
                                options.append({'label': exp_dir, 'value': exp_path})
                                if current_value is None:
                                    current_value = exp_path
            
            return options, current_value
        
        @self.dash_app.callback(
            Output('tab-content', 'children'),
            [Input('main-tabs', 'active_tab'),
             Input('experiment-selector', 'value')]
        )
        def render_tab_content(active_tab, selected_experiment):
            """Render content for the active tab."""
            if not selected_experiment or not os.path.exists(selected_experiment):
                return html.Div("Please select a valid experiment.", className="alert alert-warning")
            
            try:
                # Load experiment data
                report_file = os.path.join(selected_experiment, 'experiment_report.json')
                results_file = os.path.join(selected_experiment, 'results.csv')
                
                if not os.path.exists(report_file):
                    return html.Div("Experiment report not found.", className="alert alert-danger")
                
                with open(report_file, 'r') as f:
                    report = json.load(f)
                
                if os.path.exists(results_file):
                    df = pd.read_csv(results_file)
                else:
                    df = pd.DataFrame()
                
                if active_tab == "overview":
                    return self._render_overview_tab(report, df)
                elif active_tab == "models":
                    return self._render_models_tab(report, df)
                elif active_tab == "fairness":
                    return self._render_fairness_tab(report, df)
                elif active_tab == "temporal":
                    return self._render_temporal_tab(report, df)
                elif active_tab == "detailed":
                    return self._render_detailed_tab(report, df)
            
            except Exception as e:
                return html.Div(f"Error loading experiment data: {str(e)}", className="alert alert-danger")
            
            return html.Div("Tab content not implemented yet.")
    
    def _render_overview_tab(self, report: Dict, df: pd.DataFrame) -> html.Div:
        """Render experiment overview tab."""
        if df.empty:
            return html.Div("No data available for visualization.", className="alert alert-warning")
        
        # Summary statistics
        summary_cards = dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{report.get('total_subjects', 0)}", className="card-title"),
                        html.P("Total Subjects", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(report.get('models_tested', []))}", className="card-title"),
                        html.P("Models Tested", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"${report.get('total_cost', 0):.2f}", className="card-title"),
                        html.P("Total Cost", className="card-text")
                    ])
                ])
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4(f"{len(df)} " if not df.empty else "0", className="card-title"),
                        html.P("Total Responses", className="card-text")
                    ])
                ])
            ], width=3),
        ], className="mb-4")
        
        # Risk label distribution
        if not df.empty and 'risk_label' in df.columns:
            risk_dist = df['risk_label'].value_counts()
            fig_risk = px.pie(
                values=risk_dist.values,
                names=risk_dist.index,
                title="Risk Label Distribution"
            )
            fig_risk.update_layout(height=400)
            risk_chart = dcc.Graph(figure=fig_risk)
        else:
            risk_chart = html.Div("No risk label data available.", className="alert alert-info")
        
        # Model performance comparison
        if not df.empty and 'model' in df.columns:
            model_stats = df.groupby('model').agg({
                'risk_label': 'count',
                'response_time_seconds': 'mean'
            }).reset_index()
            model_stats.columns = ['Model', 'Responses', 'Avg Response Time']
            
            fig_models = px.bar(
                model_stats,
                x='Model',
                y='Responses',
                title="Responses by Model"
            )
            fig_models.update_layout(height=400)
            models_chart = dcc.Graph(figure=fig_models)
        else:
            models_chart = html.Div("No model data available.", className="alert alert-info")
        
        return html.Div([
            summary_cards,
            dbc.Row([
                dbc.Col([risk_chart], width=6),
                dbc.Col([models_chart], width=6)
            ])
        ])
    
    def _render_models_tab(self, report: Dict, df: pd.DataFrame) -> html.Div:
        """Render model comparison tab."""
        if df.empty or 'model' not in df.columns:
            return html.Div("No model comparison data available.", className="alert alert-warning")
        
        # Model accuracy comparison
        if 'risk_band_correct' in df.columns:
            accuracy_by_model = df.groupby('model')['risk_band_correct'].mean().reset_index()
            accuracy_by_model.columns = ['Model', 'Accuracy']
            
            fig_accuracy = px.bar(
                accuracy_by_model,
                x='Model',
                y='Accuracy',
                title="Model Accuracy Comparison",
                labels={'Accuracy': 'Accuracy Rate'}
            )
            fig_accuracy.update_layout(height=400)
            fig_accuracy.update_traces(marker_color='lightblue')
        else:
            fig_accuracy = go.Figure().add_annotation(
                text="No accuracy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Response time comparison
        if 'response_time_seconds' in df.columns:
            fig_time = px.box(
                df,
                x='model',
                y='response_time_seconds',
                title="Response Time Distribution by Model"
            )
            fig_time.update_layout(height=400)
        else:
            fig_time = go.Figure().add_annotation(
                text="No response time data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Cost per response
        if 'cost' in df.columns:
            cost_by_model = df.groupby('model')['cost'].mean().reset_index()
            cost_by_model.columns = ['Model', 'Avg Cost']
            
            fig_cost = px.bar(
                cost_by_model,
                x='Model',
                y='Avg Cost',
                title="Average Cost per Response by Model"
            )
            fig_cost.update_layout(height=400)
            fig_cost.update_traces(marker_color='lightcoral')
        else:
            fig_cost = go.Figure().add_annotation(
                text="No cost data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_accuracy)], width=6),
                dbc.Col([dcc.Graph(figure=fig_time)], width=6)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_cost)], width=12)
            ])
        ])
    
    def _render_fairness_tab(self, report: Dict, df: pd.DataFrame) -> html.Div:
        """Render fairness analysis tab."""
        if df.empty:
            return html.Div("No fairness analysis data available.", className="alert alert-warning")
        
        # Gender bias analysis
        if 'gender' in df.columns and 'risk_label' in df.columns:
            gender_risk = pd.crosstab(df['gender'], df['risk_label'], normalize='index') * 100
            
            fig_gender = px.imshow(
                gender_risk.values,
                x=gender_risk.columns,
                y=gender_risk.index,
                title="Risk Label Distribution by Gender (%)",
                aspect="auto",
                color_continuous_scale="RdYlBu"
            )
            fig_gender.update_layout(height=300)
        else:
            fig_gender = go.Figure().add_annotation(
                text="No gender bias data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Age bias analysis
        if 'age' in df.columns and 'risk_label' in df.columns:
            # Create age groups
            df_copy = df.copy()
            df_copy['age_group'] = pd.cut(df_copy['age'], bins=[20, 35, 50, 70], labels=['Young', 'Middle', 'Senior'])
            age_risk = pd.crosstab(df_copy['age_group'], df_copy['risk_label'], normalize='index') * 100
            
            fig_age = px.imshow(
                age_risk.values,
                x=age_risk.columns,
                y=age_risk.index,
                title="Risk Label Distribution by Age Group (%)",
                aspect="auto",
                color_continuous_scale="RdYlBu"
            )
            fig_age.update_layout(height=300)
        else:
            fig_age = go.Figure().add_annotation(
                text="No age bias data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Demographic Bias Analysis", className="mb-3"),
                    dcc.Graph(figure=fig_gender)
                ], width=6),
                dbc.Col([
                    html.H4("Age Group Analysis", className="mb-3"),
                    dcc.Graph(figure=fig_age)
                ], width=6)
            ])
        ])
    
    def _render_temporal_tab(self, report: Dict, df: pd.DataFrame) -> html.Div:
        """Render temporal analysis tab."""
        if df.empty or 'timestamp' not in df.columns:
            return html.Div("No temporal analysis data available.", className="alert alert-warning")
        
        # Convert timestamps
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy = df_copy.sort_values('timestamp')
        
        # Accuracy over time
        if 'risk_band_correct' in df_copy.columns:
            # Rolling accuracy
            df_copy['rolling_accuracy'] = df_copy['risk_band_correct'].rolling(window=10, min_periods=1).mean()
            
            fig_temporal = px.line(
                df_copy,
                x='timestamp',
                y='rolling_accuracy',
                title="Model Accuracy Over Time (Rolling Average)",
                labels={'rolling_accuracy': 'Accuracy Rate', 'timestamp': 'Time'}
            )
            fig_temporal.update_layout(height=400)
        else:
            fig_temporal = go.Figure().add_annotation(
                text="No temporal accuracy data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        # Response distribution over time
        if 'risk_label' in df_copy.columns:
            # Hourly risk label counts
            df_copy['hour'] = df_copy['timestamp'].dt.hour
            hourly_dist = pd.crosstab(df_copy['hour'], df_copy['risk_label'])
            
            fig_hourly = px.bar(
                hourly_dist.reset_index(),
                x='hour',
                y=hourly_dist.columns.tolist(),
                title="Risk Label Distribution by Hour of Day"
            )
            fig_hourly.update_layout(height=400)
        else:
            fig_hourly = go.Figure().add_annotation(
                text="No hourly distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        
        return html.Div([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_temporal)], width=12)
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_hourly)], width=12)
            ])
        ])
    
    def _render_detailed_tab(self, report: Dict, df: pd.DataFrame) -> html.Div:
        """Render detailed results tab."""
        if df.empty:
            return html.Div("No detailed data available.", className="alert alert-warning")
        
        # Data table with key columns
        display_columns = []
        for col in ['subject_id', 'model', 'risk_label', 'true_risk_band', 'risk_band_correct', 'gender', 'age', 'timestamp']:
            if col in df.columns:
                display_columns.append(col)
        
        if display_columns:
            display_df = df[display_columns].head(100)  # Limit to first 100 rows for performance
            
            # Create HTML table
            table = dbc.Table.from_dataframe(
                display_df,
                striped=True,
                bordered=True,
                hover=True,
                responsive=True,
                size='sm',
                class_name="table-sm"
            )
            
            return html.Div([
                html.H4(f"Detailed Results (showing first {len(display_df)} of {len(df)} rows)"),
                html.P("Key experiment data for analysis and debugging."),
                table
            ])
        else:
            return html.Div("No data columns available for display.", className="alert alert-warning")
    
    def run(self, debug=False):
        """Start the web interface server."""
        print(f"Starting LLM Risk Fairness Experiment Web Interface...")
        print(f"Main interface: http://{self.host}:{self.port}/")
        print(f"Dashboard: http://{self.host}:{self.port}/dashboard/")
        print(f"Configuration: http://{self.host}:{self.port}/config")
        
        self.flask_app.run(
            host=self.host,
            port=self.port,
            debug=debug,
            threaded=True
        )


def main():
    """CLI entry point for web interface."""
    import argparse
    parser = argparse.ArgumentParser(description="LLM Risk Fairness Experiment Web Interface")
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    interface = ExperimentWebInterface(host=args.host, port=args.port)
    interface.run(debug=args.debug)


if __name__ == "__main__":
    main()