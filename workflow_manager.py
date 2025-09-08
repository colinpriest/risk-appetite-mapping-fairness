#!/usr/bin/env python3
"""
Unified Workflow Manager for Risk Fairness Experiments
Intelligently manages the entire experiment pipeline from setup to analysis
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import yaml

# Color codes for terminal output (Windows compatible)
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_status(status: str, message: str):
    """Print a status message with color."""
    symbols = {
        'success': f"{Colors.GREEN}[OK]{Colors.ENDC}",
        'error': f"{Colors.RED}[X]{Colors.ENDC}",
        'warning': f"{Colors.YELLOW}[!]{Colors.ENDC}",
        'info': f"{Colors.BLUE}[i]{Colors.ENDC}",
        'pending': f"{Colors.YELLOW}[?]{Colors.ENDC}"
    }
    print(f"{symbols.get(status, '[.]')} {message}")

def check_environment() -> Dict[str, bool]:
    """Check if the environment is properly set up."""
    from dotenv import load_dotenv
    load_dotenv()
    
    checks = {
        'python': True,  # We're running Python
        'api_keys': {
            'openai': bool(os.getenv('OPENAI_API_KEY')),
            'anthropic': bool(os.getenv('ANTHROPIC_API_KEY')),
            'google': bool(os.getenv('GOOGLE_API_KEY'))
        },
        'dependencies': True,
        'config_files': {
            'full_experiment': os.path.exists('full_experiment_config.yaml'),
            'test_models': os.path.exists('test_all_models.yaml')
        }
    }
    
    # Check key dependencies
    try:
        import pandas
        import numpy
        import llm_risk_fairness_experiment
        import streamlit
    except ImportError as e:
        checks['dependencies'] = False
    
    return checks

def find_experiment_runs() -> List[Dict]:
    """Find all experiment runs and their status."""
    runs = []
    
    # Check common output directories
    for base_dir in ['results', 'runs']:
        if not os.path.exists(base_dir):
            continue
            
        for item in Path(base_dir).iterdir():
            if item.is_dir():
                run_info = analyze_run(item)
                if run_info:
                    runs.append(run_info)
    
    # Sort by timestamp (most recent first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs

def analyze_run(run_dir: Path) -> Optional[Dict]:
    """Analyze a single experiment run directory."""
    info = {
        'path': str(run_dir),
        'name': run_dir.name,
        'timestamp': run_dir.stat().st_mtime
    }
    
    # Check for key files
    files = {
        'results_jsonl': run_dir / 'results.jsonl',
        'results_csv': run_dir / 'results.csv',
        'stats_summary': run_dir / 'stats_summary.json',
        'bias_summary': run_dir / 'bias_summary.json',
        'research_findings': run_dir / 'research_findings.json',
        'research_report': run_dir / 'research_report.txt',
        'visualizations': run_dir / 'visualizations',
        'dashboard_state': run_dir / '.dashboard_state'
    }
    
    info['has_results'] = files['results_jsonl'].exists()
    info['has_csv'] = files['results_csv'].exists()
    info['has_analysis'] = files['stats_summary'].exists()
    info['has_bias_analysis'] = files['bias_summary'].exists()
    info['has_research_analysis'] = files['research_findings'].exists() and files['research_report'].exists()
    info['has_visualizations'] = files['visualizations'].exists() and any(files['visualizations'].iterdir()) if files['visualizations'].exists() else False
    info['dashboard_viewed'] = files['dashboard_state'].exists()
    
    # Count results if available
    if info['has_results']:
        try:
            with open(files['results_jsonl'], 'r') as f:
                info['total_calls'] = sum(1 for _ in f)
        except:
            info['total_calls'] = 0
    else:
        info['total_calls'] = 0
    
    # Get models used if CSV exists
    if info['has_csv']:
        try:
            import pandas as pd
            df = pd.read_csv(files['results_csv'])
            info['models'] = df['model'].unique().tolist() if 'model' in df.columns else []
            info['success_rate'] = (df['ok'].sum() / len(df) * 100) if 'ok' in df.columns else 0
        except:
            info['models'] = []
            info['success_rate'] = 0
    
    return info if info['has_results'] else None

def display_run_status(run: Dict):
    """Display the status of a single run."""
    timestamp = datetime.fromtimestamp(run['timestamp'])
    age = datetime.now() - timestamp
    
    print(f"\n{Colors.BOLD}Run: {run['name']}{Colors.ENDC}")
    print(f"  Path: {run['path']}")
    print(f"  Age: {format_timedelta(age)}")
    print(f"  Total calls: {run.get('total_calls', 0):,}")
    
    if run.get('models'):
        print(f"  Models: {', '.join(run['models'])}")
    if 'success_rate' in run:
        print(f"  Success rate: {run['success_rate']:.1f}%")
    
    print("\n  Status:")
    statuses = [
        ('Results collected', run['has_results'], 'success' if run['has_results'] else 'error'),
        ('CSV generated', run['has_csv'], 'success' if run['has_csv'] else 'warning'),
        ('Statistical analysis', run['has_analysis'], 'success' if run['has_analysis'] else 'pending'),
        ('Bias detection analysis', run.get('has_bias_analysis', False), 'success' if run.get('has_bias_analysis', False) else 'pending'),
        ('Research-grade analysis', run.get('has_research_analysis', False), 'success' if run.get('has_research_analysis', False) else 'pending'),
        ('Visualizations created', run['has_visualizations'], 'success' if run['has_visualizations'] else 'pending'),
        ('Dashboard viewed', run['dashboard_viewed'], 'info' if run['dashboard_viewed'] else 'pending')
    ]
    
    for label, status, status_type in statuses:
        if status:
            print_status(status_type, label)
        else:
            print_status('pending', f"{label} - Not completed")

def format_timedelta(td: timedelta) -> str:
    """Format a timedelta in human-readable form."""
    if td.days > 0:
        return f"{td.days} day{'s' if td.days != 1 else ''} ago"
    elif td.seconds > 3600:
        hours = td.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif td.seconds > 60:
        minutes = td.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"

def run_statistical_analysis(run_dir: str) -> bool:
    """Run statistical analysis on experiment results."""
    print_status('info', f"Running statistical analysis on {run_dir}...")
    
    try:
        cmd = [
            sys.executable, "-c",
            f"""
import json
import llm_risk_fairness_experiment as exp
summary = exp.stats_report('{run_dir}')
with open('{run_dir}/stats_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("Analysis complete!")
print(summary)
"""
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print_status('success', "Statistical analysis completed")
            print(result.stdout)
            return True
        else:
            print_status('error', f"Analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print_status('error', f"Error running analysis: {e}")
        return False

def run_bias_detection(run_dir: str) -> bool:
    """Run bias detection analysis with clear yes/no findings."""
    print_status('info', f"Running bias detection analysis on {run_dir}...")
    
    try:
        cmd = [
            sys.executable,
            "bias_summary.py",
            run_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print_status('success', "Bias detection analysis completed")
            print(result.stdout)
            return True
        else:
            print_status('error', f"Bias analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print_status('error', f"Error running bias analysis: {e}")
        return False

def run_comprehensive_research_analysis(run_dir: str) -> bool:
    """Run comprehensive research-grade statistical analysis."""
    print_status('info', f"Running comprehensive research analysis on {run_dir}...")
    print("This performs publication-ready statistical analysis including:")
    print("  - Rigorous statistical tests with multiple testing correction")
    print("  - Effect size calculations and confidence intervals")
    print("  - Root cause analysis of detected biases")
    print("  - Research implications and intervention recommendations")
    
    try:
        cmd = [
            sys.executable,
            "research_analysis.py",
            run_dir
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print_status('success', "Comprehensive research analysis completed")
            print(result.stdout)
            
            # Show key findings from the analysis
            findings_path = Path(run_dir) / "research_findings.json"
            if findings_path.exists():
                print("\\n" + "="*50)
                print("RESEARCH ANALYSIS SUMMARY")
                print("="*50)
                try:
                    with open(findings_path, 'r') as f:
                        findings = json.load(f)
                    
                    print(f"Bias Status: {'DETECTED' if findings.get('overall_bias_detected') else 'NOT DETECTED'}")
                    if findings.get('overall_bias_detected'):
                        print("\\nBias Types Found:")
                        if findings.get('demographic_bias_detected'):
                            print("  ✓ Demographic bias (name-based)")
                        if findings.get('location_bias_detected'):
                            print("  ✓ Location bias (geography-based)")  
                        if findings.get('condition_bias_detected'):
                            print("  ✓ Condition bias (demographic vs baseline)")
                    
                    corrections = findings.get('multiple_testing_corrections', {})
                    if isinstance(corrections, dict):
                        total_tests = corrections.get('total_tests', 0)
                        sig_tests = corrections.get('significant_after_holm', 0)
                        print(f"\\nStatistical Tests: {sig_tests}/{total_tests} significant after correction")
                    
                    interventions = findings.get('intervention_points', [])
                    if interventions:
                        print("\\nTop Intervention Recommendations:")
                        for i, intervention in enumerate(interventions[:3], 1):
                            print(f"  {i}. {intervention}")
                            
                except Exception as e:
                    print(f"Could not parse research findings: {e}")
            
            return True
        else:
            print_status('error', f"Research analysis failed: {result.stderr}")
            return False
    except Exception as e:
        print_status('error', f"Error running research analysis: {e}")
        return False

def generate_visualizations(run_dir: str) -> bool:
    """Generate advanced visualizations."""
    print_status('info', f"Generating visualizations for {run_dir}...")
    
    try:
        viz_dir = Path(run_dir) / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        cmd = [
            sys.executable,
            "advanced_visualization.py",
            "--indir", run_dir,
            "--outdir", str(viz_dir),
            "--no-dashboard"  # Don't launch interactive dashboard
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print_status('success', f"Visualizations saved to {viz_dir}")
            # List generated files
            viz_files = list(viz_dir.glob('*.html')) + list(viz_dir.glob('*.png'))
            if viz_files:
                print("  Generated files:")
                for f in viz_files[:5]:  # Show first 5
                    print(f"    - {f.name}")
                if len(viz_files) > 5:
                    print(f"    ... and {len(viz_files)-5} more")
            return True
        else:
            print_status('error', f"Visualization failed: {result.stderr}")
            return False
    except Exception as e:
        print_status('error', f"Error generating visualizations: {e}")
        return False

def launch_dashboard(run_dir: str):
    """Launch the interactive dashboard."""
    print_status('info', "Launching interactive dashboard...")
    print("  Dashboard will open in your browser at http://localhost:8501")
    print("  Press Ctrl+C to stop the dashboard and return to this menu")
    
    # Mark dashboard as viewed
    dashboard_state = Path(run_dir) / '.dashboard_state'
    dashboard_state.write_text(str(datetime.now()))
    
    cmd = [
        "streamlit", "run", "llm_risk_fairness_experiment.py",
        "--", "--dashboard", "--indir", run_dir
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n")
        print_status('info', "Dashboard closed")

def run_new_experiment() -> Optional[str]:
    """Run a new experiment with user configuration."""
    print_header("NEW EXPERIMENT SETUP")
    
    # Show available configurations
    configs = {
        '1': ('Quick test (3 subjects, 1 repeat)', 'test_all_models.yaml'),
        '2': ('Full experiment (30 subjects, 3 repeats)', 'full_experiment_config.yaml'),
        '3': ('Custom configuration', None)
    }
    
    print("\nAvailable configurations:")
    for key, (desc, _) in configs.items():
        print(f"  {key}. {desc}")
    
    choice = input("\nSelect configuration (1-3): ").strip()
    
    if choice not in configs:
        print_status('error', "Invalid choice")
        return None
    
    if choice == '3':
        # Custom configuration
        K = input("Number of subjects (default 20): ").strip() or "20"
        repeats = input("Number of repeats (default 3): ").strip() or "3"
        models = input("Models (comma-separated, default gpt-4o,claude-3.5-sonnet): ").strip() or "gpt-4o,claude-3.5-sonnet"
        threads = input("Number of threads (default 5): ").strip() or "5"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"results/custom_run_{timestamp}"
        
        cmd = [
            sys.executable, "llm_risk_fairness_experiment.py", "run",
            "--models", models,
            "--K", K,
            "--repeats", repeats,
            "--threads", threads,
            "--outdir", outdir
        ]
    else:
        config_file = configs[choice][1]
        threads = input("Number of threads (default 5): ").strip() or "5"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = f"results/run_{timestamp}"
        
        cmd = [
            sys.executable, "llm_risk_fairness_experiment.py", "run",
            "--config", config_file,
            "--threads", threads,
            "--outdir", outdir
        ]
    
    print_status('info', f"Starting experiment...")
    print(f"  Command: {' '.join(cmd)}")
    
    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print_status('warning', "Experiment cancelled")
        return None
    
    try:
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print_status('success', f"Experiment completed! Results in {outdir}")
            return outdir
        else:
            print_status('error', "Experiment failed")
            return None
    except KeyboardInterrupt:
        print_status('warning', "Experiment interrupted")
        return outdir  # Partial results may be available

def main_menu():
    """Main interactive menu."""
    while True:
        print_header("RISK FAIRNESS EXPERIMENT WORKFLOW MANAGER")
        
        # Check environment
        env_status = check_environment()
        
        print("\n" + Colors.BOLD + "Environment Status:" + Colors.ENDC)
        print_status('success' if env_status['python'] else 'error', "Python environment")
        
        api_status = env_status['api_keys']
        for provider, has_key in api_status.items():
            print_status('success' if has_key else 'warning', f"{provider.upper()} API key")
        
        print_status('success' if env_status['dependencies'] else 'error', "Required dependencies")
        
        # Find and display runs
        runs = find_experiment_runs()
        
        print(f"\n{Colors.BOLD}Experiment Runs Found: {len(runs)}{Colors.ENDC}")
        
        if runs:
            # Show most recent run
            print("\nMost recent run:")
            display_run_status(runs[0])
            
            # Determine next steps for most recent run
            current_run = runs[0]
            next_steps = []
            
            if not current_run['has_analysis']:
                next_steps.append(('Run statistical analysis', 'analysis', current_run['path']))
            if not current_run.get('has_bias_analysis', False):
                next_steps.append(('Run bias detection analysis', 'bias', current_run['path']))
            if not current_run.get('has_research_analysis', False):
                next_steps.append(('Run comprehensive research analysis', 'research', current_run['path']))
            if not current_run['has_visualizations']:
                next_steps.append(('Generate visualizations', 'visualize', current_run['path']))
            if not current_run['dashboard_viewed'] or True:  # Always allow dashboard
                next_steps.append(('Launch interactive dashboard', 'dashboard', current_run['path']))
        else:
            print_status('info', "No experiment runs found")
            next_steps = []
        
        # Menu options
        print(f"\n{Colors.BOLD}Available Actions:{Colors.ENDC}")
        
        menu_items = []
        
        # Add next steps for current run
        for i, (desc, action, path) in enumerate(next_steps, 1):
            menu_items.append((str(i), desc, action, path))
            print(f"  {i}. {desc}")
        
        # Always available options
        base_option = len(next_steps) + 1
        
        menu_items.append((str(base_option), 'Run new experiment', 'new', None))
        print(f"  {base_option}. Run new experiment")
        
        if len(runs) > 1:
            menu_items.append((str(base_option + 1), 'Select different run', 'select', None))
            print(f"  {base_option + 1}. Select different run")
        
        menu_items.append((str(base_option + 2), 'Refresh', 'refresh', None))
        print(f"  {base_option + 2}. Refresh")
        
        menu_items.append(('q', 'Quit', 'quit', None))
        print(f"  q. Quit")
        
        # Get user choice
        choice = input(f"\n{Colors.BOLD}Select action:{Colors.ENDC} ").strip().lower()
        
        # Find matching action
        selected = None
        for key, desc, action, path in menu_items:
            if choice == key:
                selected = (action, path)
                break
        
        if not selected:
            print_status('error', "Invalid choice")
            input("\nPress Enter to continue...")
            continue
        
        action, path = selected
        
        # Execute action
        if action == 'quit':
            print_status('info', "Goodbye!")
            break
        elif action == 'refresh':
            continue
        elif action == 'new':
            new_run = run_new_experiment()
            if new_run:
                input("\nPress Enter to continue...")
        elif action == 'analysis':
            run_statistical_analysis(path)
            input("\nPress Enter to continue...")
        elif action == 'bias':
            run_bias_detection(path)
            input("\nPress Enter to continue...")
        elif action == 'research':
            run_comprehensive_research_analysis(path)
            input("\nPress Enter to continue...")
        elif action == 'visualize':
            generate_visualizations(path)
            input("\nPress Enter to continue...")
        elif action == 'dashboard':
            launch_dashboard(path)
        elif action == 'select':
            print("\nAvailable runs:")
            for i, run in enumerate(runs, 1):
                timestamp = datetime.fromtimestamp(run['timestamp'])
                print(f"  {i}. {run['name']} ({format_timedelta(datetime.now() - timestamp)})")
            
            run_choice = input("\nSelect run number: ").strip()
            try:
                selected_idx = int(run_choice) - 1
                if 0 <= selected_idx < len(runs):
                    display_run_status(runs[selected_idx])
                    # Update current run for next iteration
                    runs.insert(0, runs.pop(selected_idx))
                else:
                    print_status('error', "Invalid run number")
            except ValueError:
                print_status('error', "Invalid input")
            
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    try:
        # Enable ANSI colors on Windows
        if sys.platform == 'win32':
            os.system('color')
        
        main_menu()
    except KeyboardInterrupt:
        print("\n")
        print_status('info', "Workflow manager terminated")
    except Exception as e:
        print_status('error', f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()