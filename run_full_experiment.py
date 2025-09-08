#!/usr/bin/env python3
"""
Full Fairness Experiment Runner
Runs comprehensive fairness analysis with all models and statistical tests
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
import subprocess
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_api_keys():
    """Check that all required API keys are set."""
    from dotenv import load_dotenv
    load_dotenv()
    
    keys_status = {
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'ANTHROPIC_API_KEY': bool(os.getenv('ANTHROPIC_API_KEY')),
        'GOOGLE_API_KEY': bool(os.getenv('GOOGLE_API_KEY'))
    }
    
    logger.info("API Key Status:")
    for key, present in keys_status.items():
        status = "OK - Present" if present else "ERROR - Missing"
        logger.info(f"  {key}: {status}")
    
    if not all(keys_status.values()):
        logger.warning("Some API keys are missing. Those models will be skipped.")
    
    return keys_status

def estimate_experiment_stats():
    """Estimate experiment statistics and requirements."""
    import yaml
    
    with open('full_experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    K = config['K']
    repeats = config['repeats']
    nl_sample = config['nl_sample']
    num_models = len(config['models'])
    
    # Calculate total calls
    # Conditions: 1 ND + 7 name groups + 9 locations + nl_sample NL pairs
    conditions_per_subject = 1 + 7 + 9 + nl_sample
    total_calls = num_models * K * repeats * conditions_per_subject
    
    # Estimate time (assuming ~2s per call with caching)
    estimated_hours = (total_calls * 2) / 3600
    
    # Estimate cost (rough approximation)
    avg_cost_per_call = 0.02  # Average across all models
    estimated_cost = total_calls * avg_cost_per_call
    
    stats = {
        'subjects': K,
        'repeats': repeats,
        'models': num_models,
        'conditions_per_subject': conditions_per_subject,
        'total_calls': total_calls,
        'estimated_hours': estimated_hours,
        'estimated_cost_usd': estimated_cost
    }
    
    logger.info("\nExperiment Statistics:")
    logger.info(f"  Subjects: {stats['subjects']} (stratified across 6 risk bands)")
    logger.info(f"  Repeats: {stats['repeats']}")
    logger.info(f"  Models: {stats['models']}")
    logger.info(f"  Conditions per subject/repeat: {stats['conditions_per_subject']}")
    logger.info(f"  Total API calls: {stats['total_calls']:,}")
    logger.info(f"  Estimated runtime: {stats['estimated_hours']:.1f} hours")
    logger.info(f"  Estimated cost: ${stats['estimated_cost_usd']:.2f} USD")
    
    return stats

def run_experiment(output_dir: str, threads: int = 5):
    """Run the full experiment with monitoring."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"{output_dir}/full_run_{timestamp}"
    
    logger.info(f"\nStarting experiment with output directory: {outdir}")
    
    # Build command
    cmd = [
        sys.executable,
        "llm_risk_fairness_experiment.py",
        "run",
        "--config", "full_experiment_config.yaml",
        "--outdir", outdir,
        "--threads", str(threads),
        "--log-level", "INFO"
    ]
    
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Run experiment with real-time output
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"\n[SUCCESS] Experiment completed successfully!")
            return outdir
        else:
            logger.error(f"\n[ERROR] Experiment failed with return code: {process.returncode}")
            return None
            
    except KeyboardInterrupt:
        logger.info("\n\n[WARNING] Experiment interrupted by user")
        logger.info("Partial results saved. Use --resume flag to continue.")
        return outdir
    except Exception as e:
        logger.error(f"\n[ERROR] Error running experiment: {e}")
        return None

def analyze_results(outdir: str):
    """Run analysis on completed experiment."""
    logger.info(f"\nAnalyzing results in: {outdir}")
    
    # Run statistical analysis
    cmd = [
        sys.executable,
        "-c",
        f"import llm_risk_fairness_experiment as exp; summary = exp.stats_report('{outdir}'); print(summary)"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Analysis failed: {e}")
        
    # Launch dashboard
    logger.info("\nLaunching dashboard for visualization...")
    logger.info("Dashboard will be available at: http://localhost:8501")
    logger.info("Press Ctrl+C to stop the dashboard")
    
    cmd = [
        "streamlit", "run", "llm_risk_fairness_experiment.py",
        "--", "--dashboard", "--indir", outdir
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("\nDashboard stopped")

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run full fairness experiment")
    parser.add_argument("--output-dir", default="results", help="Base output directory")
    parser.add_argument("--threads", type=int, default=5, help="Number of concurrent threads")
    parser.add_argument("--analyze-only", help="Only analyze existing results", metavar="DIR")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip analysis after experiment")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("FULL FAIRNESS EXPERIMENT RUNNER")
    logger.info("=" * 60)
    
    if args.analyze_only:
        analyze_results(args.analyze_only)
        return
    
    # Pre-flight checks
    check_api_keys()
    stats = estimate_experiment_stats()
    
    # Confirmation prompt for large experiments
    if stats['total_calls'] > 10000:
        logger.info("\nWARNING: This is a large experiment!")
        response = input("\nProceed with experiment? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Experiment cancelled.")
            return
    
    # Run experiment
    logger.info("\n" + "=" * 60)
    logger.info("STARTING EXPERIMENT")
    logger.info("=" * 60)
    
    start_time = time.time()
    outdir = run_experiment(args.output_dir, args.threads)
    elapsed = time.time() - start_time
    
    if outdir:
        logger.info(f"\nExperiment completed in {elapsed/3600:.1f} hours")
        logger.info(f"Results saved to: {outdir}")
        
        if not args.skip_analysis:
            analyze_results(outdir)
    else:
        logger.error("\nExperiment failed or was interrupted")
        
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT RUNNER FINISHED")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()