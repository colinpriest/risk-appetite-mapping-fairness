# Installation Guide for LLM Risk Fairness Experiment Platform

This guide provides step-by-step installation instructions for different use cases.

## Quick Installation (Recommended)

### Step 1: Install Core Dependencies
```bash
pip install -r requirements-core.txt
```

### Step 2: Test Basic Functionality
```bash
python llm_risk_fairness_experiment.py --help
```

### Step 3: Add Advanced Features (Optional)
```bash
# For advanced visualization
pip install networkx>=3.1.0 umap-learn>=0.5.3 kaleido>=0.2.1 dash-cytoscape>=0.3.0

# For web interface
pip install streamlit>=1.20.0 dash>=2.14.0 dash-bootstrap-components>=1.5.0

# For distributed computing (choose one)
pip install celery>=5.3.0 redis>=4.6.0  # Celery option
pip install dask[complete]>=2023.8.0     # Dask option

# For advanced ML features
pip install xgboost>=1.7.0 optuna>=3.0.0 fairlearn>=0.9.0

# For advanced analytics (large dependencies)
pip install pymc>=5.0.0 arviz>=0.15.0

# For report generation
pip install jinja2>=3.1.0 reportlab>=4.0.0 weasyprint>=60.0
```

## Full Installation (All Features)

### Option 1: Modified requirements.txt
```bash
pip install -r requirements.txt
```

### Option 2: Manual installation of problematic packages
If the above fails, install problematic packages separately:

```bash
# Install core first
pip install -r requirements-core.txt

# Install advanced packages one by one
pip install bayesian-optimization
pip install networkx umap-learn kaleido
pip install dash dash-bootstrap-components
pip install celery redis
pip install xgboost optuna
pip install fairlearn lime shap

# Optional: Large ML packages (install if needed)
pip install pymc arviz  # Bayesian analysis
pip install ray         # Distributed computing (complex on Windows)
pip install aif360      # Fairness toolkit (complex dependencies)
```

## Platform-Specific Instructions

### Windows
```bash
# Some packages may require Visual C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# For Ray (optional, complex on Windows)
pip install ray[default]  # May require additional setup

# For LaTeX report generation
# Install MikTeX: https://miktex.org/download
```

### Linux/Mac
```bash
# Install system dependencies for some packages
sudo apt-get update
sudo apt-get install build-essential python3-dev

# For LaTeX report generation
sudo apt-get install texlive-latex-base texlive-latex-extra

# Full installation should work directly
pip install -r requirements.txt
```

## Verification

### Test Core Functionality
```bash
python test_enhanced_experiment.py
```

### Test Advanced Features
```bash
# Test what's installed
python -c "import pandas, numpy, openai, anthropic; print('Core packages OK')"

# Test visualization (if installed)
python -c "import plotly, matplotlib, networkx; print('Visualization packages OK')"

# Test advanced analytics (if installed)
python -c "import sklearn, scipy, statsmodels; print('Analytics packages OK')"
```

## Troubleshooting

### Common Issues

**1. scikit-bootstrap not found**
- This package doesn't exist. Use `arch>=5.3.0` instead (already fixed in requirements.txt)

**2. Ray installation fails on Windows**
- Ray has complex Windows dependencies. Skip it initially:
  ```bash
  pip install -r requirements.txt --ignore-requires-ray
  ```
  Or install manually later if distributed computing is needed.

**3. aif360 installation fails**
- This package has complex dependencies. Skip it initially:
  ```bash
  # Comment out aif360 in requirements.txt or install separately
  pip install aif360 --no-deps  # Install without dependencies first
  ```

**4. PyMC installation is slow**
- PyMC is a large package. Install separately:
  ```bash
  pip install pymc arviz  # Takes 5-10 minutes
  ```

**5. Memory issues during installation**
- Install packages in smaller batches:
  ```bash
  pip install -r requirements-core.txt
  pip install networkx umap-learn plotly
  pip install dash streamlit
  # etc.
  ```

### Minimal Working Setup

If you encounter many issues, use this minimal setup that definitely works:

```bash
pip install pandas numpy scipy scikit-learn
pip install openai anthropic google-genai
pip install plotly matplotlib seaborn
pip install tqdm pyyaml python-dotenv
pip install tiktoken instructor pydantic
pip install pytest
```

Then test:
```bash
python llm_risk_fairness_experiment.py run --models gpt-4o --K 10 --repeats 1 --outdir test_results
```

## Feature Dependencies

| Feature | Required Packages |
|---------|------------------|
| Core Experiments | pandas, numpy, openai, anthropic |
| Visualization | plotly, matplotlib, seaborn |
| Advanced Analytics | scikit-learn, scipy, statsmodels |
| 3D Visualizations | networkx, umap-learn, plotly |
| Web Dashboard | streamlit, dash |
| Distributed Computing | celery+redis OR dask OR ray |
| ML Prediction | xgboost, optuna, fairlearn |
| Bayesian Analysis | pymc, arviz |
| Report Generation | jinja2, reportlab, weasyprint |

## Next Steps

After successful installation:

1. Set up API keys in `.env` file
2. Run a test experiment: `python llm_risk_fairness_experiment.py run --K 10 --models gpt-4o`
3. Check the experimental workflow guide in README.md
4. Start with core features, add advanced components as needed

## Getting Help

If installation issues persist:

1. Check the error messages for specific package names
2. Install packages individually to isolate issues
3. Use `pip install --verbose` for detailed error information
4. Consider using a virtual environment: `python -m venv fairness_env`
5. Check package compatibility: `pip check`