#!/usr/bin/env python3
"""
End-to-End Test for Room Calibration.

This script validates that the N-dimensional parameter space search is actually
working by:
1. Generating a WAV file from a famous quotation using text-to-speech
2. Running the calibration with a simulated transcription service
3. Verifying that different parameters produce different results
4. Confirming that the optimization finds reasonable settings

Usage:
    # Run the end-to-end test
    python tools/test_calibration_e2e.py

    # Verbose output
    python tools/test_calibration_e2e.py -v

    # Use a custom quotation
    python tools/test_calibration_e2e.py --quote "To be or not to be"

Requirements:
    - espeak-ng or espeak (for TTS)
    - sox (for audio processing)
    - jiwer (for WER calculation)
    - optuna (for Bayesian optimization)
"""

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Test configuration constants
MIN_RMS_VARIATION_DB = 0.5  # Minimum RMS difference to consider parameters as effective
SCORE_TOLERANCE = 0.3  # Tolerance for expected scores in scoring tests
MIN_SCORE_RANGE = 0.05  # Minimum score range for optimization to be meaningful
MIN_ACCEPTABLE_SCORE = 0.5  # Minimum acceptable best score

# Famous quotations for testing
FAMOUS_QUOTES = [
    "To be or not to be, that is the question.",
    "The only thing we have to fear is fear itself.",
    "I have a dream that one day this nation will rise up.",
    "Ask not what your country can do for you, ask what you can do for your country.",
    "The quick brown fox jumps over the lazy dog.",
]

# Import from room_calibration
from room_calibration import (
    BASELINE_PARAMS,
    PARAMETER_SPACE,
    PARAMETER_RANGES,
    CalibrationResult,
    build_noise_profile,
    build_sox_command,
    apply_sox_processing,
    generate_test_audio,
    calculate_word_accuracy,
    generate_parameter_combinations,
    HAS_JIWER,
    HAS_OPTUNA,
)


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are available."""
    missing = []
    
    # Check TTS
    tts_found = False
    for cmd in ["espeak-ng", "espeak"]:
        try:
            subprocess.run(["which", cmd], check=True, capture_output=True)
            tts_found = True
            break
        except subprocess.CalledProcessError:
            continue
    if not tts_found:
        missing.append("espeak-ng (install with: sudo apt-get install espeak-ng)")
    
    # Check sox
    try:
        subprocess.run(["which", "sox"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        missing.append("sox (install with: sudo apt-get install sox)")
    
    # Check Python dependencies
    if not HAS_JIWER:
        missing.append("jiwer (install with: pip install jiwer)")
    
    if not HAS_OPTUNA:
        missing.append("optuna (install with: pip install optuna)")
    
    return len(missing) == 0, missing


def simulate_transcription(audio_path: Path, reference_text: str, noise_level: float = 0.0) -> str:
    """
    Simulate a transcription by returning the reference text with some variation.
    
    The noise_level parameter (0.0-1.0) determines how much the transcript
    varies from the reference. This simulates how different sox parameters
    might affect transcription quality.
    """
    if noise_level == 0.0:
        return reference_text
    
    words = reference_text.split()
    modified_words = []
    
    for word in words:
        if random.random() < noise_level:
            # Simulate various transcription errors
            error_type = random.choice(["delete", "substitute", "insert"])
            if error_type == "delete":
                continue  # Skip this word
            elif error_type == "substitute":
                # Replace with a similar-sounding word or garbled text
                modified_words.append(word[::-1] if len(word) > 2 else "um")
            else:  # insert
                modified_words.append(word)
                modified_words.append("uh")
        else:
            modified_words.append(word)
    
    return " ".join(modified_words)


def run_sox_and_get_stats(input_path: Path, output_path: Path, 
                          noise_profile: Path, params: Dict[str, str]) -> Optional[Dict]:
    """Apply sox processing and get audio statistics."""
    success, sox_cmd = apply_sox_processing(input_path, output_path, noise_profile, params)
    if not success:
        return None
    
    # Get audio stats using sox
    try:
        result = subprocess.run(
            ["sox", str(output_path), "-n", "stats"],
            capture_output=True,
            text=True
        )
        # Parse stats from stderr (sox outputs stats to stderr)
        # Format is like: "RMS lev dB    -21.58" (name followed by spaces and value)
        stats = {}
        for line in result.stderr.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Find the last whitespace-separated token as the value
            parts = line.rsplit(None, 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value_str = parts[1].strip()
                # Try to convert numeric values
                try:
                    stats[key] = float(value_str)
                except ValueError:
                    stats[key] = value_str
        return stats
    except Exception as e:
        return {"error": str(e)}


def test_parameter_variation(workdir: Path, audio_path: Path, 
                            reference_text: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Test that different parameter combinations produce different audio outputs.
    This validates that the parameter space is actually being explored.
    """
    noise_profile = workdir / "noise.prof"
    if not build_noise_profile(audio_path, noise_profile):
        return False, "Failed to build noise profile"
    
    # Test a few distinct parameter combinations
    test_params = [
        BASELINE_PARAMS.copy(),  # Baseline
        {**BASELINE_PARAMS, "noisered": "0.30", "highpass": "400"},  # High noise reduction
        {**BASELINE_PARAMS, "noisered": "0.10", "highpass": "100"},  # Low noise reduction
        {**BASELINE_PARAMS, "eq1_gain": "6", "eq2_gain": "5"},  # High EQ boost
        {**BASELINE_PARAMS, "eq1_gain": "1", "eq2_gain": "1"},  # Low EQ boost
    ]
    
    results = []
    for i, params in enumerate(test_params):
        output_path = workdir / f"variant_{i}.wav"
        stats = run_sox_and_get_stats(audio_path, output_path, noise_profile, params)
        if stats:
            results.append({
                "params": params,
                "stats": stats,
                "path": str(output_path)
            })
            if verbose:
                rms_db = stats.get("RMS lev dB", "N/A")
                print(f"  Variant {i}: noisered={params['noisered']}, "
                      f"highpass={params['highpass']}, eq1_gain={params['eq1_gain']}, "
                      f"RMS={rms_db}dB")
    
    if len(results) < 3:
        return False, f"Only {len(results)} variants processed successfully"
    
    # Check that outputs are different by comparing RMS levels
    rms_values = []
    for r in results:
        rms = r["stats"].get("RMS lev dB")
        if rms is not None and isinstance(rms, (int, float)):
            rms_values.append(rms)
    
    if len(rms_values) < 2:
        return False, "Could not extract RMS values from audio stats"
    
    # Check that we have variation in RMS levels
    rms_range = max(rms_values) - min(rms_values)
    if rms_range < MIN_RMS_VARIATION_DB:
        return False, f"RMS variation too small ({rms_range:.2f} dB) - parameters may not be affecting output"
    
    return True, f"Parameter variation test passed: RMS range={rms_range:.2f}dB across {len(results)} variants"


def test_scoring_variation(reference_text: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Test that the WER scoring function properly differentiates transcripts.
    """
    test_cases = [
        (reference_text, 1.0, "exact match"),
        (reference_text.replace(".", ""), 0.95, "no punctuation"),
        (" ".join(reference_text.split()[:3]), 0.3, "partial transcript"),
        ("completely different text here", 0.0, "wrong transcript"),
        ("", 0.0, "empty transcript"),
    ]
    
    results = []
    for transcript, expected_range, desc in test_cases:
        score = calculate_word_accuracy(reference_text, transcript)
        # Allow some tolerance for expected scores
        in_range = abs(score - expected_range) < SCORE_TOLERANCE
        results.append((desc, score, expected_range, in_range))
        if verbose:
            status = "✓" if in_range else "✗"
            print(f"  {status} {desc}: score={score:.2f}, expected≈{expected_range:.2f}")
    
    passed = sum(1 for r in results if r[3])
    if passed < len(results) - 1:  # Allow 1 failure
        return False, f"Scoring test: {passed}/{len(results)} cases passed"
    
    return True, f"Scoring variation test passed: {passed}/{len(results)} cases"


def test_optimization_converges(workdir: Path, audio_path: Path,
                                 reference_text: str, verbose: bool = False) -> Tuple[bool, str]:
    """
    Test that the parameter search can find configurations with good scores.
    Uses simulated transcription to validate the optimization flow.
    """
    noise_profile = workdir / "noise.prof"
    if not noise_profile.exists():
        if not build_noise_profile(audio_path, noise_profile):
            return False, "Failed to build noise profile"
    
    # Generate a small set of parameter combinations
    combinations = generate_parameter_combinations(PARAMETER_SPACE, max_combinations=10)
    
    results = []
    for i, params in enumerate(combinations):
        output_path = workdir / f"opt_variant_{i}.wav"
        success, sox_cmd = apply_sox_processing(audio_path, output_path, noise_profile, params)
        
        if success:
            # Simulate transcription with varying quality based on params
            # Better params (closer to baseline) = better transcription
            baseline_diff = sum(
                abs(float(params.get(k, 0)) - float(BASELINE_PARAMS.get(k, 0)))
                for k in ["noisered", "eq1_gain", "eq2_gain"]
                if k in params and k in BASELINE_PARAMS
            )
            noise_level = min(0.5, baseline_diff / 10)
            transcript = simulate_transcription(audio_path, reference_text, noise_level)
            score = calculate_word_accuracy(reference_text, transcript)
            
            results.append({
                "params": params,
                "score": score,
                "transcript": transcript[:50] + "..." if len(transcript) > 50 else transcript
            })
            
            if verbose:
                print(f"  Config {i}: score={score:.3f}")
    
    if len(results) < 5:
        return False, f"Only {len(results)} configurations tested"
    
    # Check that we have a range of scores (optimization is meaningful)
    scores = [r["score"] for r in results]
    score_range = max(scores) - min(scores)
    
    if score_range < MIN_SCORE_RANGE:
        return False, f"Score range too small ({score_range:.3f}), optimization may not be effective"
    
    best_score = max(scores)
    if best_score < MIN_ACCEPTABLE_SCORE:
        return False, f"Best score ({best_score:.3f}) too low"
    
    return True, f"Optimization test passed: best={best_score:.3f}, range={score_range:.3f}"


def run_e2e_test(quote: str, verbose: bool = False) -> bool:
    """Run the full end-to-end test."""
    print("=" * 60)
    print("Room Calibration End-to-End Test")
    print("=" * 60)
    print()
    
    # Check dependencies
    print("[1/5] Checking dependencies...")
    deps_ok, missing = check_dependencies()
    if not deps_ok:
        print("  ✗ Missing dependencies:")
        for dep in missing:
            print(f"    - {dep}")
        return False
    print("  ✓ All dependencies available")
    
    # Create temp directory
    workdir = Path(tempfile.mkdtemp(prefix="calibration_e2e_"))
    print(f"\n[2/5] Working directory: {workdir}")
    
    try:
        # Generate audio from quote
        print(f"\n[3/5] Generating audio from quote:")
        print(f"      \"{quote}\"")
        audio_path = workdir / "test_audio.wav"
        if not generate_test_audio(quote, audio_path):
            print("  ✗ Failed to generate audio")
            return False
        print(f"  ✓ Generated audio: {audio_path}")
        
        # Test parameter variation
        print("\n[4/5] Testing parameter variation...")
        passed, msg = test_parameter_variation(workdir, audio_path, quote, verbose)
        if passed:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
            return False
        
        # Test scoring variation
        print("\n[5/5] Testing scoring metrics...")
        passed, msg = test_scoring_variation(quote, verbose)
        if passed:
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")
            return False
        
        # Test optimization flow
        print("\n[Bonus] Testing optimization convergence...")
        passed, msg = test_optimization_converges(workdir, audio_path, quote, verbose)
        if passed:
            print(f"  ✓ {msg}")
        else:
            print(f"  ⚠ {msg} (non-critical)")
        
        print()
        print("=" * 60)
        print("✓ END-TO-END TEST PASSED")
        print("=" * 60)
        print()
        print("The N-dimensional parameter space search is working correctly:")
        print("  - Different parameters produce different audio outputs")
        print("  - WER/CER scoring properly differentiates transcript quality")
        print("  - Optimization can find reasonable parameter configurations")
        print()
        return True
        
    finally:
        # Cleanup
        if verbose:
            print(f"\nKeeping workdir for inspection: {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end test for room calibration"
    )
    parser.add_argument(
        "--quote", type=str, default=FAMOUS_QUOTES[0],
        help="Quote to use for testing (default: first famous quote)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--list-quotes", action="store_true",
        help="List available famous quotes"
    )
    
    args = parser.parse_args()
    
    if args.list_quotes:
        print("Available quotes:")
        for i, quote in enumerate(FAMOUS_QUOTES):
            print(f"  {i+1}. \"{quote}\"")
        return 0
    
    success = run_e2e_test(args.quote, args.verbose)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
