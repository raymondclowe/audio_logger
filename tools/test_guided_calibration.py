#!/usr/bin/env python3
"""
Tests for Guided Calibration Tool.

This test module validates the guided calibration workflow components
without requiring actual audio hardware or transcription services.

Usage:
    python tools/test_guided_calibration.py
    python tools/test_guided_calibration.py -v
"""

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import patch, MagicMock

# Import from the guided_calibration module
from guided_calibration import (
    SAMPLE_TEXT,
    SAMPLE_TEXT_FILE,
    load_sample_text,
    detect_audio_device,
    display_sample_text,
)

# Import shared components from room_calibration
from room_calibration import (
    BASELINE_PARAMS,
    CALIBRATION_CONFIG_FILE,
    calculate_word_accuracy,
    HAS_JIWER,
    HAS_OPTUNA,
)


def test_sample_text_exists() -> Tuple[bool, str]:
    """Test that sample text is defined and reasonable."""
    if not SAMPLE_TEXT:
        return False, "SAMPLE_TEXT is empty"
    
    if len(SAMPLE_TEXT) < 100:
        return False, f"SAMPLE_TEXT too short ({len(SAMPLE_TEXT)} chars)"
    
    # Should contain multiple sentences
    sentences = SAMPLE_TEXT.count('.')
    if sentences < 3:
        return False, f"SAMPLE_TEXT has too few sentences ({sentences})"
    
    # Should have enough words for meaningful WER calculation
    words = len(SAMPLE_TEXT.split())
    if words < 50:
        return False, f"SAMPLE_TEXT has too few words ({words})"
    
    return True, f"SAMPLE_TEXT valid: {len(SAMPLE_TEXT)} chars, {words} words, {sentences} sentences"


def test_sample_text_from_file() -> Tuple[bool, str]:
    """Test that sample text is loaded from the shared file."""
    if not SAMPLE_TEXT_FILE.exists():
        return False, f"Sample text file not found: {SAMPLE_TEXT_FILE}"
    
    # Verify loaded text matches file content
    file_content = SAMPLE_TEXT_FILE.read_text().strip()
    if SAMPLE_TEXT != file_content:
        return False, "SAMPLE_TEXT does not match file content"
    
    return True, f"SAMPLE_TEXT correctly loaded from {SAMPLE_TEXT_FILE.name}"


def test_sample_text_content() -> Tuple[bool, str]:
    """Test that sample text matches the Darwin's Voyage excerpt."""
    expected_start = "AFTER having been twice driven back"
    expected_contains = ["Beagle", "Captain Fitz Roy", "Patagonia", "Teneriffe"]
    
    if not SAMPLE_TEXT.startswith(expected_start):
        return False, f"SAMPLE_TEXT doesn't start with expected text"
    
    missing = [word for word in expected_contains if word not in SAMPLE_TEXT]
    if missing:
        return False, f"SAMPLE_TEXT missing expected words: {missing}"
    
    return True, "SAMPLE_TEXT contains expected Darwin's Voyage content"


def test_device_detection_format() -> Tuple[bool, str]:
    """Test that device detection returns proper format."""
    # Mock arecord -l output
    mock_output = """**** List of CAPTURE Hardware Devices ****
card 0: PCH [HDA Intel PCH], device 0: ALC892 Analog [ALC892 Analog]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
card 3: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
"""
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout=mock_output,
            returncode=0
        )
        
        device = detect_audio_device()
        
        # Should return plughw format
        if not device.startswith("plughw:"):
            return False, f"Device format invalid: {device}"
        
        # Should have card and device numbers
        parts = device.replace("plughw:", "").split(",")
        if len(parts) != 2:
            return False, f"Device format should be plughw:X,Y: {device}"
        
        return True, f"Device detection returns valid format: {device}"


def test_device_detection_filters_hdmi() -> Tuple[bool, str]:
    """Test that device detection filters out HDMI and loopback devices."""
    mock_output = """**** List of CAPTURE Hardware Devices ****
card 0: HDMI [HDA Intel HDMI], device 0: HDMI 0 [HDMI 0]
  Subdevices: 1/1
card 1: Loopback [Loopback], device 0: Loopback PCM [Loopback PCM]
  Subdevices: 1/1
card 2: Device [USB Audio Device], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
"""
    
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            stdout=mock_output,
            returncode=0
        )
        
        device = detect_audio_device()
        
        # Should prefer USB device over HDMI/Loopback
        if "plughw:2,0" not in device:
            return False, f"Should prefer USB device, got: {device}"
        
        return True, "Device detection correctly filters HDMI and Loopback"


def test_wer_with_sample_text() -> Tuple[bool, str]:
    """Test WER calculation with sample text."""
    # Exact match should give 1.0
    score = calculate_word_accuracy(SAMPLE_TEXT, SAMPLE_TEXT)
    if score < 0.99:
        return False, f"Exact match should be ~1.0, got {score}"
    
    # Partial match
    partial = " ".join(SAMPLE_TEXT.split()[:20])
    score_partial = calculate_word_accuracy(SAMPLE_TEXT, partial)
    if not (0.0 < score_partial < 1.0):
        return False, f"Partial match should be between 0 and 1, got {score_partial}"
    
    # Empty should be 0
    score_empty = calculate_word_accuracy(SAMPLE_TEXT, "")
    if score_empty != 0.0:
        return False, f"Empty transcript should be 0, got {score_empty}"
    
    return True, "WER calculation works correctly with sample text"


def test_calibration_config_path() -> Tuple[bool, str]:
    """Test that calibration config file path is valid."""
    if not isinstance(CALIBRATION_CONFIG_FILE, Path):
        return False, f"Config file should be Path, got {type(CALIBRATION_CONFIG_FILE)}"
    
    # Should be in repository root
    expected_parent = Path(__file__).parent.parent
    if CALIBRATION_CONFIG_FILE.parent != expected_parent:
        return False, f"Config should be in {expected_parent}, got {CALIBRATION_CONFIG_FILE.parent}"
    
    if CALIBRATION_CONFIG_FILE.name != "room_calibration_config.json":
        return False, f"Config should be 'room_calibration_config.json', got {CALIBRATION_CONFIG_FILE.name}"
    
    return True, f"Calibration config path valid: {CALIBRATION_CONFIG_FILE}"


def test_baseline_params_complete() -> Tuple[bool, str]:
    """Test that baseline params contain all required keys."""
    required_keys = [
        "noisered", "highpass", "lowpass",
        "compand_attack", "compand_decay",
        "eq1_freq", "eq1_width", "eq1_gain",
        "eq2_freq", "eq2_width", "eq2_gain"
    ]
    
    missing = [k for k in required_keys if k not in BASELINE_PARAMS]
    if missing:
        return False, f"BASELINE_PARAMS missing keys: {missing}"
    
    return True, f"BASELINE_PARAMS has all {len(required_keys)} required keys"


def test_dependencies_available() -> Tuple[bool, str]:
    """Test that required Python dependencies are available."""
    issues = []
    
    if not HAS_JIWER:
        issues.append("jiwer not installed (WER calculation uses fallback)")
    
    if not HAS_OPTUNA:
        issues.append("optuna not installed (Bayesian optimization unavailable)")
    
    if issues:
        return False, "Missing optional dependencies: " + ", ".join(issues)
    
    return True, "All dependencies available (jiwer, optuna)"


def run_all_tests(verbose: bool = False) -> bool:
    """Run all test functions."""
    tests = [
        ("Sample Text Exists", test_sample_text_exists),
        ("Sample Text From File", test_sample_text_from_file),
        ("Sample Text Content", test_sample_text_content),
        ("Device Detection Format", test_device_detection_format),
        ("Device Detection Filters", test_device_detection_filters_hdmi),
        ("WER with Sample Text", test_wer_with_sample_text),
        ("Calibration Config Path", test_calibration_config_path),
        ("Baseline Params Complete", test_baseline_params_complete),
        ("Dependencies Available", test_dependencies_available),
    ]
    
    print("=" * 60)
    print("Guided Calibration Test Suite")
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
                if "not installed" in message:
                    status = "⚠ WARN"
                    warnings += 1
                else:
                    status = "✗ FAIL"
                    failed += 1
            
            print(f"[{status}] {name}")
            if verbose or not success:
                print(f"       {message}")
                
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
        description="Test suite for guided calibration tool"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
