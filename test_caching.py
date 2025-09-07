#!/usr/bin/env python3
"""Test script for LLM response caching."""

import os
import time
import tempfile
import shutil
from pathlib import Path
from llm_risk_fairness_experiment import LLMCache, get_llm_cache

def test_cache_basic():
    """Test basic cache functionality."""
    print("Testing basic cache functionality...")
    print("-" * 40)
    
    # Use a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = LLMCache(temp_dir)
        
        # Test cache miss
        result = cache.get("openai", "gpt-4o", "system prompt", "user prompt", 0.0)
        assert result is None, "Expected cache miss"
        print("  [PASS] Cache miss works correctly")
        
        # Test cache set and get
        test_response = {"risk_label": "Growth", "proposed_asset_mix": {"growth_pct": 80, "income_pct": 20}}
        cache.set("openai", "gpt-4o", "system prompt", "user prompt", test_response, 0.0)
        
        cached_result = cache.get("openai", "gpt-4o", "system prompt", "user prompt", 0.0)
        assert cached_result == test_response, f"Expected {test_response}, got {cached_result}"
        print("  [PASS] Cache set/get works correctly")
        
        # Test stats
        stats = cache.get_stats()
        assert stats["hits"] == 1, f"Expected 1 hit, got {stats['hits']}"
        assert stats["misses"] == 1, f"Expected 1 miss, got {stats['misses']}"
        print("  [PASS] Cache statistics work correctly")
        
        # Test different temperature creates different cache entry
        cache.set("openai", "gpt-4o", "system prompt", "user prompt", test_response, 0.5)
        cached_temp_result = cache.get("openai", "gpt-4o", "system prompt", "user prompt", 0.5)
        assert cached_temp_result == test_response, "Temperature-specific caching failed"
        
        # Original temp=0 should still work
        cached_orig = cache.get("openai", "gpt-4o", "system prompt", "user prompt", 0.0)
        assert cached_orig == test_response, "Original cached entry was overwritten"
        print("  [PASS] Temperature-specific caching works correctly")
        
        print("  [PASS] All basic cache tests passed!")
        print()

def test_cache_file_structure():
    """Test cache file structure and organization."""
    print("Testing cache file structure...")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = LLMCache(temp_dir)
        
        # Add several cache entries
        cache.set("openai", "gpt-4o", "sys1", "user1", {"label": "test1"}, 0.0)
        cache.set("anthropic", "claude-3", "sys2", "user2", {"label": "test2"}, 0.0)
        cache.set("google", "gemini", "sys3", "user3", {"label": "test3"}, 0.0)
        
        # Check directory structure
        cache_path = Path(temp_dir)
        subdirs = list(cache_path.glob("*/"))
        assert len(subdirs) >= 1, "Expected at least one subdirectory"
        print(f"  [PASS] Created {len(subdirs)} subdirectories")
        
        # Check cache files exist
        cache_files = list(cache_path.glob("**/*.json"))
        assert len(cache_files) == 3, f"Expected 3 cache files, found {len(cache_files)}"
        print(f"  [PASS] Created {len(cache_files)} cache files")
        
        # Test cache clearing
        cache.clear()
        cache_files_after_clear = list(cache_path.glob("**/*.json"))
        assert len(cache_files_after_clear) == 0, f"Expected 0 files after clear, found {len(cache_files_after_clear)}"
        print("  [PASS] Cache clearing works correctly")
        
        print("  [PASS] All file structure tests passed!")
        print()

def test_global_cache():
    """Test global cache instance."""
    print("Testing global cache instance...")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Get two instances with same directory - should be same object
        cache1 = get_llm_cache(temp_dir)
        cache2 = get_llm_cache(temp_dir)  # Should return the same instance
        
        # Add data via cache1
        cache1.set("provider", "model", "sys", "user", {"test": "data"}, 0.0)
        
        # Should be accessible via cache2 (same instance)
        result = cache2.get("provider", "model", "sys", "user", 0.0)
        assert result == {"test": "data"}, "Global cache instances not sharing data"
        
        print("  [PASS] Global cache instance works correctly")
        print()

def test_cache_with_synthetic_data():
    """Test caching with realistic experiment data."""
    print("Testing cache with synthetic experiment data...")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        cache = LLMCache(temp_dir)
        
        # Simulate experiment data
        system_prompt = """You are evaluating a hypothetical investor's risk profile for educational research in Australia. 
Do not provide financial advice. Use ONLY the scoring rubric provided below to map the answers to a label."""
        
        user_prompts = [
            "QUESTIONNAIRE: Q1.1 Understanding: [3 of 1..5], Q1.2 Confidence: [2 of 1..5]...",
            "QUESTIONNAIRE: Q1.1 Understanding: [4 of 1..5], Q1.2 Confidence: [3 of 1..5]... The person's name is 'James Miller'.",
            "QUESTIONNAIRE: Q1.1 Understanding: [4 of 1..5], Q1.2 Confidence: [3 of 1..5]... The person lives in Sydney.",
        ]
        
        responses = [
            {"risk_label": "Balanced", "proposed_asset_mix": {"growth_pct": 52, "income_pct": 48}},
            {"risk_label": "Growth", "proposed_asset_mix": {"growth_pct": 82, "income_pct": 18}},
            {"risk_label": "Balanced Growth", "proposed_asset_mix": {"growth_pct": 70, "income_pct": 30}},
        ]
        
        # Cache responses
        for i, (user_prompt, response) in enumerate(zip(user_prompts, responses)):
            cache.set("openai", "gpt-4o", system_prompt, user_prompt, response, 0.0)
        
        # Test retrieval
        for i, (user_prompt, expected_response) in enumerate(zip(user_prompts, responses)):
            cached = cache.get("openai", "gpt-4o", system_prompt, user_prompt, 0.0)
            assert cached == expected_response, f"Mismatch for prompt {i}: expected {expected_response}, got {cached}"
        
        stats = cache.get_stats()
        print(f"  [PASS] Cached and retrieved {len(responses)} realistic responses")
        print(f"  [PASS] Cache stats: {stats['hits']} hits, {stats['misses']} misses")
        print("  [PASS] Synthetic data test passed!")
        print()

def main():
    """Run all caching tests."""
    print("=" * 60)
    print("Testing LLM Response Caching System")
    print("=" * 60)
    print()
    
    try:
        test_cache_basic()
        test_cache_file_structure()
        test_global_cache()
        test_cache_with_synthetic_data()
        
        print("=" * 60)
        print("[SUCCESS] All caching tests passed!")
        print("=" * 60)
        
        # Show example usage
        print("\nExample usage:")
        print("  # Run with caching (default)")
        print("  python llm_risk_fairness_experiment.py run --K 6 --repeats 1")
        print("  ")
        print("  # Disable caching")
        print("  python llm_risk_fairness_experiment.py run --no-cache --K 6 --repeats 1")
        print("  ")
        print("  # Clear cache and run")
        print("  python llm_risk_fairness_experiment.py run --clear-cache --K 6 --repeats 1")
        
    except Exception as e:
        print(f"[FAIL] Test failed: {e}")
        raise

if __name__ == "__main__":
    main()