# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This repository implements a fairness experiment for risk appetite mapping in Australian superannuation contexts, testing whether LLMs exhibit demographic bias when assigning risk profiles.

## Key Commands

### Running Experiments
```bash
# Execute main experiment with stratified sampling (default, recommended)
python llm_risk_fairness_experiment.py run \
  --models gpt-4o,claude-opus-4-1,gemini-2.5-pro \
  --K 60 --repeats 2 --nl-sample 12 \
  --outdir runs/equip_v1

# Disable stratified sampling (use random sampling)
python llm_risk_fairness_experiment.py run --no-stratified \
  --K 40 --repeats 2 --nl-sample 12 \
  --outdir runs/equip_v1

# Launch dashboard for results visualization
streamlit run llm_risk_fairness_experiment.py -- --dashboard --indir runs/equip_v1

# Test stratified sampling implementation
python test_stratified_sampling.py

# Test enhanced statistical features (ordinal bias, multiple testing, caching)
python test_enhanced_features.py

# Test LLM response caching system
python test_caching.py

# Test all enhanced features (config, cost tracking, progress, etc.)
python test_enhanced_experiment.py

# Test all advanced features (uncertainty, calibration, fairness, etc.)
python test_advanced_features.py
```

### Enhanced Features

```bash
# Run with configuration file
python llm_risk_fairness_experiment.py run --config config_minimal.yaml

# Resume interrupted experiment  
python llm_risk_fairness_experiment.py run --resume --outdir runs/interrupted_run

# Run with cost limits and enhanced logging
python llm_risk_fairness_experiment.py run --max-cost 50.0 --log-level DEBUG --K 30

# Run with custom retry settings (via config file)
# See config_example.yaml for all available options
```

### Advanced Analytics & Interfaces

```bash
# Launch interactive web interface for experiment design
python web_interface.py
# Main interface: http://localhost:8050/
# Dashboard: http://localhost:8050/dashboard/
# Configuration: http://localhost:8050/config

# Start distributed experiment workers (requires Redis)
python distributed_execution.py workers --backend celery --num-workers 4

# Run distributed experiment
python distributed_execution.py run \
  --config config_example.yaml \
  --outdir runs/distributed_run \
  --backend celery --workers 4 --batch-size 10

# Run sophisticated experimental design enhancements
python experimental_enhancements.py  # Example ablation study

# Test all experimental enhancements
python test_experimental_enhancements.py
```

### Development
```bash
# Install all dependencies (includes dashboard)
pip install -r requirements.txt

# Or install minimal dependencies (no dashboard)
pip install -r requirements-minimal.txt

# Check environment variables are set
python -c "import os; print('Keys:', 'OPENAI' if os.getenv('OPENAI_API_KEY') else 'X', 'ANTHROPIC' if os.getenv('ANTHROPIC_API_KEY') else 'X', 'GOOGLE' if os.getenv('GOOGLE_API_KEY') else 'X')"
```

## Architecture

### Core Components

1. **llm_risk_fairness_experiment.py**: Production-ready experiment runner
   - Implements Equip Super risk-profile rubric digitization
   - Manages experimental conditions: No-demographics (ND), Name (N), Location (L), Name+Location (NL)
   - **Stratified sampling**: Ensures balanced representation across all 6 risk bands
   - **LLM response caching**: Stores responses in `llm_responses/` to avoid redundant API calls
   - **Enhanced statistics**: Multiple testing corrections (Holm-Bonferroni/FDR), ordinal bias analysis
   - **Production features**: Config files, progress tracking, resume capability, cost estimation
   - **Robust error handling**: Retry logic, graceful shutdown, response validation
   - **Advanced logging**: Structured logs, experiment metadata, system monitoring
   - Uses `instructor` library for structured LLM outputs across providers with explicit temperature=0
   - Computes fairness metrics: demographic parity, equalized odds, calibration

2. **advanced_analytics.py**: Advanced statistical analysis and fairness metrics
   - **UncertaintyQuantifier**: Bootstrap confidence intervals, Bayesian uncertainty estimation
   - **ModelCalibrator**: Expected/Maximum Calibration Error, Platt scaling, reliability diagrams
   - **IndividualFairnessAnalyzer**: Lipschitz continuity, similarity-based fairness, counterfactual analysis
   - **IntersectionalBiasAnalyzer**: Multi-dimensional bias detection, subgroup discovery, disparity measurement
   - Supports advanced metrics beyond traditional group fairness

3. **temporal_analysis.py**: Temporal bias tracking and model evolution
   - **TemporalBiasAnalyzer**: Change point detection, concept drift analysis, seasonal patterns
   - **ModelVersionTracker**: Version comparison, performance tracking, evolution monitoring
   - CUSUM/Page-Hinkley tests for drift detection, seasonal decomposition
   - Tracks bias evolution over time and model updates

4. **web_interface.py**: Interactive web interface for experiment design and monitoring
   - Flask/Dash-based web application with real-time dashboard
   - Visual experiment configuration with form-based parameter selection
   - Live monitoring of running experiments with progress tracking
   - Interactive results visualization with filtering and analysis tools
   - API endpoints for experiment management and status monitoring

5. **distributed_execution.py**: Scalable distributed experiment execution
   - Support for Celery, Ray, and Dask distributed computing backends
   - Batch processing with configurable worker pools and retry logic
   - Real-time progress monitoring and fault tolerance
   - Hierarchical result aggregation and distributed checkpointing
   - Scales experiments across multiple machines and cloud environments

6. **experimental_enhancements.py**: Advanced experimental design and diagnostics
   - **Two-mode ablation**: Mapping-only vs full pipeline to isolate bias sources
   - **Ground-truth validation**: Automatic checking against computed risk labels and asset mixes
   - **Boundary probe generation**: Tests edge cases around risk threshold boundaries
   - **Refusal tracking**: Detects and classifies safety/policy responses with pattern matching
   - **Consistency testing**: Measures stability with temperature=0 repeats
   - **Prompt minimization**: Tests bias persistence with minimal context (TOI/TH only)

### Data Flow & Processing

1. **Core Experiment Pipeline**:
   - Synthetic subject generation with fixed questionnaire answers
   - LLM calls via unified interface (OpenAI, Anthropic, Google)
   - Results stored as JSONL + CSV with derived metrics
   - Statistical analysis: accuracy tests, McNemar flip-rate, Cohen's κ

2. **Advanced Analytics Pipeline**:
   - **Uncertainty analysis**: Bootstrap resampling, prediction intervals, Bayesian inference
   - **Calibration assessment**: ECE/MCE computation, reliability diagrams, Platt scaling
   - **Individual fairness**: Lipschitz continuity, counterfactual analysis
   - **Intersectional analysis**: Multi-dimensional bias detection across demographic groups
   - **Temporal tracking**: Drift detection, change point analysis, seasonal decomposition

3. **Web Interface Workflow**:
   - Browser-based experiment configuration with real-time validation
   - Background experiment execution with progress monitoring
   - Interactive dashboard with live result visualization
   - API-driven experiment management and status updates

4. **Distributed Processing**:
   - Batch creation with configurable subject grouping
   - Worker pool management across multiple backends (Celery/Ray/Dask)
   - Result aggregation with fault tolerance and retry logic
   - Distributed checkpointing for long-running experiments

### Key Configuration

- API keys loaded from `.env` file
- Model presets in `MODEL_PRESETS` dict  
- Equip rubric thresholds in `RUBRIC` dict
- Name pools and locations for demographic conditions
- Advanced analytics parameters in config files (uncertainty levels, temporal windows)
- Distributed execution settings (worker counts, batch sizes, retry policies)
- Web interface customization (ports, authentication, visualization themes)

### Experimental Design

- **K** base subjects with questionnaire answers mapped to ground-truth risk labels
- **R** repeats per condition to measure consistency
- Counterfactual design: identical answers, only demographic tokens change
- Fairness tests against Australian anti-discrimination context

## Advanced Feature Dependencies

### Required for Core Features
```bash
# Install basic requirements
pip install -r requirements-minimal.txt
```

### Optional for Advanced Features
```bash
# Web interface and visualization
pip install streamlit plotly dash flask flask-cors

# Distributed computing (choose one or more)
pip install celery redis  # For Celery backend
pip install ray          # For Ray backend  
pip install dask         # For Dask backend

# Advanced statistical analysis
pip install pymc arviz bayesian-optimization
pip install fairlearn aif360 lime shap
pip install scikit-bootstrap pingouin

# Development and testing
pip install pytest pytest-cov coverage black pre-commit
```

### Backend Services
```bash
# For distributed execution with Celery
docker run -p 6379:6379 redis:alpine  # or install Redis locally

# For Ray distributed computing  
ray start --head                      # starts Ray cluster

# For Dask distributed computing
# No additional setup needed (uses local cluster by default)
```

## File Structure & Components

```
risk-appetite-mapping-fairness/
├── llm_risk_fairness_experiment.py    # Core experiment runner
├── advanced_analytics.py               # Advanced statistical analysis
├── temporal_analysis.py                # Temporal bias tracking  
├── web_interface.py                     # Interactive web interface
├── distributed_execution.py            # Distributed experiment execution
├── config_example.yaml                 # Full configuration template
├── config_minimal.yaml                 # Minimal configuration template
├── requirements.txt                     # All dependencies
├── requirements-minimal.txt             # Core dependencies only
├── experimental_enhancements.py         # Advanced experimental design features
├── test_*.py                           # Test suites for all features  
├── ENHANCEMENTS.md                     # Detailed feature documentation
└── CLAUDE.md                           # This file
```

## Usage Patterns

### For Researchers
1. **Standard Experiments**: Use `llm_risk_fairness_experiment.py` with config files
2. **Advanced Analysis**: Import modules from `advanced_analytics.py` and `temporal_analysis.py`
3. **Interactive Exploration**: Launch `web_interface.py` for visual experiment design
4. **Large-Scale Studies**: Use `distributed_execution.py` for multi-machine experiments

### For Developers
1. **Testing**: Run `test_advanced_features.py` to validate all components
2. **Extension**: Add new analytics to `advanced_analytics.py`
3. **Customization**: Modify web interface templates in `web_interface.py`
4. **Integration**: Use distributed execution as a library for custom workflows

## Important Notes

- Results are for research only, not financial advice
- Indigenous identification handled via explicit statement, not name fabrication (per AIATSIS ethics)
- Rubric digitization based on Equip Super V2 01.11.24 questionnaire
- Advanced features require additional dependencies - see requirements.txt
- Distributed execution requires Redis/Ray/Dask setup depending on backend choice
- Web interface runs on localhost by default - configure for production use
- All advanced analytics support both real-time and batch processing modes