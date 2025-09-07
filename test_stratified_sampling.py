#!/usr/bin/env python3
"""Test script to verify stratified sampling for risk bands."""

import sys
from collections import Counter
from llm_risk_fairness_experiment import make_stratified_subjects, equip_label_for, RUBRIC

def test_stratified_sampling():
    """Test that stratified sampling produces balanced risk label distribution."""
    
    print("Testing stratified sampling for risk bands...")
    print("=" * 60)
    
    # Test with different K values
    test_sizes = [6, 12, 30, 60, 120]
    
    for K in test_sizes:
        print(f"\nTest with K={K} subjects:")
        print("-" * 40)
        
        # Generate subjects
        subjects = make_stratified_subjects(K)
        
        # Count labels
        label_counts = Counter(s.true_label for s in subjects)
        
        # Expected per label
        expected_per_label = K // 6
        remainder = K % 6
        
        print(f"Expected per label: {expected_per_label} (+ {remainder} distributed)")
        print("\nActual distribution:")
        
        for label in RUBRIC["labels"]:
            count = label_counts.get(label, 0)
            percentage = (count / K) * 100
            expected = expected_per_label + (1 if RUBRIC["labels"].index(label) < remainder else 0)
            status = "OK" if count == expected else "MISMATCH"
            print(f"  {label:20s}: {count:3d} ({percentage:5.1f}%) - Expected: {expected} {status}")
        
        # Verify all subjects have correct labels
        errors = []
        for s in subjects:
            computed_label = equip_label_for(s.toi, s.th)
            if computed_label != s.true_label:
                errors.append(f"Subject {s.subject_id}: TOI={s.toi}, TH={s.th}, "
                            f"Expected={s.true_label}, Got={computed_label}")
        
        if errors:
            print("\nWARNING: Label computation errors:")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(errors) > 5:
                print(f"  ... and {len(errors) - 5} more errors")
        else:
            print("\n[PASS] All labels computed correctly")
        
        # Check TOI and TH ranges
        toi_values = [s.toi for s in subjects]
        th_values = [s.th for s in subjects]
        
        print(f"\nTOI range: {min(toi_values)} - {max(toi_values)} (valid: 8-40)")
        print(f"TH range: {min(th_values)} - {max(th_values)} (valid: 2-10)")
        
        # Verify questionnaire answers
        for s in subjects[:3]:  # Check first 3 subjects
            toi_sum = sum(s.answers[f"1.{i}"] for i in range(1, 9))
            th_sum = sum(s.answers[f"2.{i}"] for i in range(1, 3))
            
            if toi_sum != s.toi or th_sum != s.th:
                print(f"\nWARNING: Sum mismatch for subject {s.subject_id}:")
                print(f"  TOI: answers sum to {toi_sum}, expected {s.toi}")
                print(f"  TH: answers sum to {th_sum}, expected {s.th}")
    
    print("\n" + "=" * 60)
    print("Stratified sampling test complete!")

if __name__ == "__main__":
    test_stratified_sampling()