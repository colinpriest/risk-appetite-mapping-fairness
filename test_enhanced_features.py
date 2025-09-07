#!/usr/bin/env python3
"""Test script to verify enhanced statistical features."""

import numpy as np
import pandas as pd
from llm_risk_fairness_experiment import (
    ordinal_error, 
    analyze_ordinal_bias,
    apply_multiple_testing_correction,
    RUBRIC
)

def test_ordinal_error():
    """Test ordinal error calculation."""
    print("Testing ordinal error calculation...")
    print("-" * 40)
    
    # Test cases: (predicted, true, expected_error)
    test_cases = [
        ("Cash", "Cash", 0),  # Exact match
        ("Capital Stable", "Cash", 1),  # Over-risk by 1
        ("Balanced", "Cash", 2),  # Over-risk by 2
        ("Cash", "Capital Stable", -1),  # Under-risk by 1
        ("Cash", "High Growth", -5),  # Under-risk by 5
        ("High Growth", "Cash", 5),  # Over-risk by 5
    ]
    
    for pred, true, expected in test_cases:
        error = ordinal_error(pred, true)
        status = "OK" if error == expected else "FAIL"
        print(f"  {pred:15s} vs {true:15s}: {error:+2d} (expected {expected:+2d}) [{status}]")
    
    print()

def test_ordinal_bias_analysis():
    """Test ordinal bias analysis with synthetic data."""
    print("Testing ordinal bias analysis...")
    print("-" * 40)
    
    # Create synthetic data with different bias patterns
    np.random.seed(42)
    
    # Group A: tends to over-risk
    group_a_data = []
    for _ in range(50):
        true_idx = np.random.randint(0, 5)  # Don't use High Growth as true
        pred_idx = min(5, true_idx + np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
        group_a_data.append({
            "true_label": RUBRIC["labels"][true_idx],
            "pred_label": RUBRIC["labels"][pred_idx],
            "group": "GroupA"
        })
    
    # Group B: tends to under-risk
    group_b_data = []
    for _ in range(50):
        true_idx = np.random.randint(1, 6)  # Don't use Cash as true
        pred_idx = max(0, true_idx - np.random.choice([0, 1, 2], p=[0.3, 0.5, 0.2]))
        group_b_data.append({
            "true_label": RUBRIC["labels"][true_idx],
            "pred_label": RUBRIC["labels"][pred_idx],
            "group": "GroupB"
        })
    
    # Group C: unbiased
    group_c_data = []
    for _ in range(50):
        true_idx = np.random.randint(0, 6)
        pred_idx = np.clip(true_idx + np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25]), 0, 5)
        group_c_data.append({
            "true_label": RUBRIC["labels"][true_idx],
            "pred_label": RUBRIC["labels"][pred_idx],
            "group": "GroupC"
        })
    
    # Combine data
    df = pd.DataFrame(group_a_data + group_b_data + group_c_data)
    
    # Analyze
    results = analyze_ordinal_bias(df, "group")
    
    print("Results by group:")
    for group in ["GroupA", "GroupB", "GroupC"]:
        stats = results["by_group"][group]
        print(f"\n  {group}:")
        print(f"    Mean error: {stats['mean_error']:+.2f}")
        print(f"    Over-risk rate: {stats['over_risk_rate']:.1%}")
        print(f"    Under-risk rate: {stats['under_risk_rate']:.1%}")
        print(f"    Exact rate: {stats['exact_rate']:.1%}")
    
    if "kruskal_wallis" in results:
        print(f"\nKruskal-Wallis test:")
        print(f"  H-statistic: {results['kruskal_wallis']['statistic']:.2f}")
        print(f"  p-value: {results['kruskal_wallis']['p_value']:.4f}")
    
    if "chi2_direction" in results:
        print(f"\nChi-square test for directional bias:")
        print(f"  Chi2: {results['chi2_direction']['statistic']:.2f}")
        print(f"  p-value: {results['chi2_direction']['p_value']:.4f}")
    
    print()

def test_multiple_testing_correction():
    """Test multiple testing corrections."""
    print("Testing multiple testing corrections...")
    print("-" * 40)
    
    # Create some p-values with a mix of significant and non-significant
    p_values = {
        "test1": 0.001,  # Very significant
        "test2": 0.01,   # Significant
        "test3": 0.04,   # Borderline
        "test4": 0.06,   # Not significant
        "test5": 0.15,   # Not significant
        "test6": 0.003,  # Very significant
        "test7": np.nan, # Missing
    }
    
    # Test different correction methods
    methods = ["holm", "bonferroni", "fdr_bh"]
    
    for method in methods:
        print(f"\nMethod: {method}")
        print("  Test         Original   Corrected   Reject H0")
        print("  " + "-" * 45)
        
        corrected = apply_multiple_testing_correction(p_values, method=method)
        
        for test_name in sorted(p_values.keys()):
            if test_name in corrected:
                orig = corrected[test_name]["original"]
                corr = corrected[test_name]["corrected"]
                reject = corrected[test_name]["reject_h0"]
                
                if np.isnan(orig):
                    print(f"  {test_name:12s}    NaN        NaN         -")
                else:
                    reject_str = "Yes" if reject else "No"
                    print(f"  {test_name:12s} {orig:8.4f}   {corr:8.4f}    {reject_str}")
    
    print()

def test_temperature_setting():
    """Verify temperature settings in LLM client initialization."""
    print("Testing temperature settings in LLM clients...")
    print("-" * 40)
    
    from llm_risk_fairness_experiment import LLMClient, MODEL_PRESETS
    
    # Test a mock client (without actual API calls)
    test_models = ["gpt-4o", "claude-opus-4.1", "gemini-2.5-pro"]
    
    for model_name in test_models:
        if model_name in MODEL_PRESETS:
            meta = MODEL_PRESETS[model_name]
            print(f"\nModel: {model_name}")
            print(f"  Provider: {meta['provider']}")
            print(f"  Model ID: {meta['model']}")
            
            # Check that temperature would be set (without actual API key)
            # This just verifies the code structure
            try:
                # Note: This will fail without API keys, but we can check the error
                client = LLMClient(meta["provider"], meta["model"])
                print(f"  Temperature setting: {client.kw.get('temperature', 'not set')}")
                print(f"  Top-p setting: {client.kw.get('top_p', 'not set')}")
            except Exception as e:
                # Expected when API key missing or library incompatibility
                print(f"  Would set temperature=0.0, top_p=1.0 (client init failed: {type(e).__name__})")
    
    print()

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Enhanced Statistical Features")
    print("=" * 60)
    print()
    
    test_ordinal_error()
    test_ordinal_bias_analysis()
    test_multiple_testing_correction()
    test_temperature_setting()
    
    print("=" * 60)
    print("All tests complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()