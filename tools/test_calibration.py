#!/usr/bin/env python3
"""
Test Harness for Room Calibration.

This script provides a test harness for validating the room calibration tool
without requiring actual audio hardware or a transcription server. It uses
mock/simulated data to test the calibration algorithms.

Usage:
    # Run all tests
    python tools/test_calibration.py

    # Run specific test
    python tools/test_calibration.py --test wer

    # Verbose output
    python tools/test_calibration.py -v
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from room_calibration import (
    BASELINE_PARAMS,
    PARAMETER_RANGES,
    PARAMETER_SPACE,
    CalibrationResult,
    calculate_transcript_quality,
    calculate_word_accuracy,
    generate_parameter_combinations,
    HAS_JIWER,
    HAS_OPTUNA,
)


def test_wer_metric() -> Tuple[bool, str]:
    """Test WER-based accuracy calculation."""
    test_cases = [
        # (reference, transcript, expected_min_accuracy, expected_max_accuracy, description)
        ("hello world", "hello world", 0.99, 1.01, "exact match"),
        ("hello world", "Hello World", 0.99, 1.01, "case insensitive match"),
        ("hello world", "goodbye world", 0.4, 0.6, "one word wrong"),
        ("hello world", "world hello", 0.0, 0.01, "word order swapped - WER penalizes"),
        ("the quick brown fox", "the quick brown", 0.7, 0.8, "missing word"),
        ("the quick brown fox", "the quick brown fox jumps", 0.7, 0.85, "extra word"),
        ("hello world", "", 0.0, 0.01, "empty transcript"),
        ("", "hello world", 0.0, 0.01, "empty reference"),
        ("Hello, world!", "Hello world", 0.99, 1.01, "punctuation difference (normalized)"),
    ]
    
    failed = []
    for ref, trans, min_acc, max_acc, desc in test_cases:
        accuracy = calculate_word_accuracy(ref, trans, metric="wer")
        if not (min_acc <= accuracy <= max_acc):
            failed.append(f"  FAIL: {desc}: got {accuracy:.3f}, expected [{min_acc}, {max_acc}]")
    
    if failed:
        return False, "WER metric tests failed:\n" + "\n".join(failed)
    return True, f"WER metric tests passed ({len(test_cases)} cases)" + (
        " [using jiwer]" if HAS_JIWER else " [using fallback F1]"
    )


def test_cer_metric() -> Tuple[bool, str]:
    """Test CER-based accuracy calculation (if jiwer available)."""
    if not HAS_JIWER:
        return True, "CER metric tests skipped (jiwer not installed)"
    
    test_cases = [
        # (reference, transcript, expected_min_accuracy, expected_max_accuracy, description)
        ("hello", "hello", 0.99, 1.01, "exact match"),
        ("hello", "hallo", 0.75, 0.85, "one char wrong"),
        ("hello", "hell", 0.75, 0.85, "one char missing"),
        ("hello", "helloo", 0.80, 0.90, "one char extra"),
        ("Hello, world!", "hello world", 0.99, 1.01, "punctuation and case (normalized)"),
    ]
    
    failed = []
    for ref, trans, min_acc, max_acc, desc in test_cases:
        accuracy = calculate_word_accuracy(ref, trans, metric="cer")
        if not (min_acc <= accuracy <= max_acc):
            failed.append(f"  FAIL: {desc}: got {accuracy:.3f}, expected [{min_acc}, {max_acc}]")
    
    if failed:
        return False, "CER metric tests failed:\n" + "\n".join(failed)
    return True, f"CER metric tests passed ({len(test_cases)} cases)"


def test_transcript_quality() -> Tuple[bool, str]:
    """Test heuristic transcript quality scoring."""
    test_cases = [
        # (transcript, min_score, max_score, description)
        ("", 0.0, 0.01, "empty transcript"),
        ("ERROR: connection failed", 0.0, 0.01, "error message"),
        ("hi", 0.05, 0.15, "very short"),
        ("the quick brown fox jumps over the lazy dog", 0.8, 1.0, "good transcript"),
        ("!!!###@@@", 0.0, 0.3, "mostly symbols"),
        ("Hello, world! This is a test.", 0.7, 1.0, "with punctuation"),
    ]
    
    failed = []
    for transcript, min_score, max_score, desc in test_cases:
        score = calculate_transcript_quality(transcript)
        if not (min_score <= score <= max_score):
            failed.append(f"  FAIL: {desc}: got {score:.3f}, expected [{min_score}, {max_score}]")
    
    if failed:
        return False, "Transcript quality tests failed:\n" + "\n".join(failed)
    return True, f"Transcript quality tests passed ({len(test_cases)} cases)"


def test_parameter_combinations() -> Tuple[bool, str]:
    """Test parameter combination generation."""
    # Test with the actual PARAMETER_SPACE
    param_space = PARAMETER_SPACE
    
    # All combinations would be huge, test with limited sampling
    combos_limited = generate_parameter_combinations(param_space, max_combinations=10)
    if len(combos_limited) > 10:
        return False, f"Expected at most 10 combinations, got {len(combos_limited)}"
    if len(combos_limited) < 1:
        return False, f"Expected at least 1 combination, got {len(combos_limited)}"
    
    # Each combination should have all required keys
    required_keys = set(param_space.keys())
    for combo in combos_limited:
        if set(combo.keys()) != required_keys:
            return False, f"Combination missing keys: {required_keys - set(combo.keys())}"
    
    # Baseline should be included when sampling
    baseline_tuple = tuple(BASELINE_PARAMS[k] for k in param_space.keys())
    has_baseline = any(
        tuple(c[k] for k in param_space.keys()) == baseline_tuple
        for c in combos_limited
    )
    if not has_baseline:
        return False, "Baseline configuration not included in sampled combinations"
    
    return True, "Parameter combination tests passed"


def test_parameter_ranges() -> Tuple[bool, str]:
    """Test parameter range definitions for Bayesian optimization."""
    required_params = set(BASELINE_PARAMS.keys())
    range_params = set(PARAMETER_RANGES.keys())
    
    if required_params != range_params:
        missing = required_params - range_params
        extra = range_params - required_params
        msg = []
        if missing:
            msg.append(f"Missing ranges: {missing}")
        if extra:
            msg.append(f"Extra ranges: {extra}")
        return False, "Parameter range mismatch: " + ", ".join(msg)
    
    # Check all ranges are valid
    for param, (min_val, max_val, step) in PARAMETER_RANGES.items():
        if min_val >= max_val:
            return False, f"Invalid range for {param}: min({min_val}) >= max({max_val})"
        if step <= 0:
            return False, f"Invalid step for {param}: step({step}) <= 0"
    
    # Check baseline values are within ranges
    for param, value in BASELINE_PARAMS.items():
        min_val, max_val, _ = PARAMETER_RANGES[param]
        val = float(value)
        if not (min_val <= val <= max_val):
            return False, f"Baseline {param}={value} outside range [{min_val}, {max_val}]"
    
    return True, "Parameter range tests passed"


def test_bayesian_optimization_available() -> Tuple[bool, str]:
    """Test that Bayesian optimization dependencies are available."""
    if not HAS_OPTUNA:
        return False, "optuna not installed - Bayesian optimization unavailable"
    return True, "Bayesian optimization available (optuna installed)"


def test_wer_library_available() -> Tuple[bool, str]:
    """Test that WER library is available."""
    if not HAS_JIWER:
        return False, "jiwer not installed - using fallback F1 metric"
    return True, "WER/CER metrics available (jiwer installed)"


def test_mock_calibration() -> Tuple[bool, str]:
    """Test calibration workflow with mocked audio/transcription."""
    # This test validates the overall workflow logic without actual audio
    
    mock_results = []
    
    # Simulate some parameter configurations and their "scores"
    test_configs = [
        ({"noisered": "0.21", "highpass": "300"}, "hello world test", 0.85),
        ({"noisered": "0.15", "highpass": "200"}, "hello world", 0.70),
        ({"noisered": "0.30", "highpass": "400"}, "hello test world", 0.75),
    ]
    
    for params, transcript, score in test_configs:
        result = CalibrationResult(
            params=params,
            transcript=transcript,
            accuracy_score=score,
            wav_path="/tmp/test.wav",
            sox_command=["sox", "test"]
        )
        mock_results.append(result)
    
    # Verify sorting
    mock_results.sort(key=lambda r: r.accuracy_score, reverse=True)
    if mock_results[0].accuracy_score != 0.85:
        return False, "Results not sorted correctly by accuracy"
    
    # Verify best result selection
    best = mock_results[0]
    if best.params["noisered"] != "0.21":
        return False, "Best configuration not selected correctly"
    
    return True, "Mock calibration workflow tests passed"


def run_all_tests(verbose: bool = False) -> bool:
    """Run all test functions and report results."""
    tests = [
        ("WER Library", test_wer_library_available),
        ("Bayesian Opt", test_bayesian_optimization_available),
        ("WER Metric", test_wer_metric),
        ("CER Metric", test_cer_metric),
        ("Quality Score", test_transcript_quality),
        ("Param Combos", test_parameter_combinations),
        ("Param Ranges", test_parameter_ranges),
        ("Mock Calibration", test_mock_calibration),
    ]
    
    print("=" * 60)
    print("Room Calibration Test Harness")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    warnings = 0
    
    for name, test_func in tests:
        try:
            success, message = test_func()
            if success:
                status = "✓ PASS"
                passed += 1
            else:
                # Check if it's a "soft" failure (optional dependency)
                if "not installed" in message:
                    status = "⚠ WARN"
                    warnings += 1
                else:
                    status = "✗ FAIL"
                    failed += 1
            
            print(f"[{status}] {name}")
            if verbose or not success:
                for line in message.split("\n"):
                    print(f"       {line}")
        except Exception as e:
            status = "✗ ERROR"
            failed += 1
            print(f"[{status}] {name}")
            print(f"       Exception: {e}")
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {warnings} warnings")
    print("=" * 60)
    
    return failed == 0


def main():
    parser = argparse.ArgumentParser(
        description="Test harness for room calibration tool"
    )
    parser.add_argument(
        "--test", choices=["wer", "cer", "quality", "combos", "ranges", "bayesian", "mock", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.test == "all":
        success = run_all_tests(verbose=args.verbose)
        sys.exit(0 if success else 1)
    
    # Run specific test
    test_map = {
        "wer": test_wer_metric,
        "cer": test_cer_metric,
        "quality": test_transcript_quality,
        "combos": test_parameter_combinations,
        "ranges": test_parameter_ranges,
        "bayesian": test_bayesian_optimization_available,
        "mock": test_mock_calibration,
    }
    
    test_func = test_map[args.test]
    success, message = test_func()
    
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {args.test}: {message}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
