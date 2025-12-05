#!/usr/bin/env python3
"""
Room Calibration Tool for Audio Logger.

This tool helps find optimal sox audio processing parameters for a specific room
by testing different configurations against a reference audio sample and comparing
transcription accuracy.

Features:
- Can record a speech sample or use an existing audio file
- Can generate test audio using text-to-speech (if available)
- Explores a configurable space of sox parameters (noise reduction, filters, etc.)
- Transcribes each variant and compares to find the most accurate settings
- Outputs recommended sox settings for main.py

Usage:
    # Record a speech sample and calibrate
    python tools/room_calibration.py --record --device plughw:3,0

    # Use an existing audio file with known transcript
    python tools/room_calibration.py --input sample.wav --reference "expected transcript text"

    # Generate test audio (requires espeak)
    python tools/room_calibration.py --generate --device plughw:3,0

    # Quick calibration with fewer parameter combinations
    python tools/room_calibration.py --record --quick
"""

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import wave

DEFAULT_URL = "http://192.168.0.142:8085/transcribe"
DEFAULT_DEVICE = "plughw:0,0"
DEFAULT_MODEL = "base"

# Parameter space for calibration
# Each parameter has a name and list of values to try
PARAMETER_SPACE = {
    "noisered": ["0.15", "0.18", "0.21", "0.25", "0.30"],
    "highpass": ["100", "200", "300", "400"],
    "lowpass": ["3000", "3400", "3800", "4000"],
    "compand_attack": ["0.03", "0.05", "0.1"],
    "compand_decay": ["0.15", "0.2", "0.3"],
    "eq1_freq": ["800", "1000", "1200"],
    "eq1_width": ["400", "500", "600"],
    "eq1_gain": ["2", "3", "4"],
    "eq2_freq": ["2500", "3000", "3500"],
    "eq2_width": ["800", "1000", "1200"],
    "eq2_gain": ["2", "3"],
}

# Quick mode uses a reduced parameter space
QUICK_PARAMETER_SPACE = {
    "noisered": ["0.18", "0.21", "0.25"],
    "highpass": ["200", "300"],
    "lowpass": ["3400", "3800"],
    "compand_attack": ["0.05"],
    "compand_decay": ["0.2"],
    "eq1_freq": ["1000"],
    "eq1_width": ["500"],
    "eq1_gain": ["3"],
    "eq2_freq": ["3000"],
    "eq2_width": ["1000"],
    "eq2_gain": ["2"],
}

# Baseline sox chain (current main.py settings)
BASELINE_PARAMS = {
    "noisered": "0.21",
    "highpass": "300",
    "lowpass": "3400",
    "compand_attack": "0.03",
    "compand_decay": "0.15",
    "eq1_freq": "800",
    "eq1_width": "400",
    "eq1_gain": "4",
    "eq2_freq": "2500",
    "eq2_width": "800",
    "eq2_gain": "3",
}

# Test phrase for generated audio
TEST_PHRASE = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a test of the audio calibration system. "
    "One two three four five six seven eight nine ten."
)


@dataclass
class CalibrationResult:
    """Result from testing a sox parameter configuration."""

    params: Dict[str, str]
    transcript: str
    accuracy_score: float
    wav_path: str
    sox_command: List[str]


def run_command(cmd: List[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def record_audio(device: str, output_path: Path, duration: int = 10) -> bool:
    """Record audio from the specified device."""
    print(f"Recording {duration}s from {device}...")
    cmd = [
        "arecord",
        "-D", device,
        "-d", str(duration),
        "-f", "cd",
        "-t", "wav",
        str(output_path)
    ]
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Recording failed: {e}")
        return False


def generate_test_audio(text: str, output_path: Path) -> bool:
    """Generate speech audio using espeak (if available)."""
    try:
        # Check if espeak is available
        run_command(["which", "espeak"])
    except subprocess.CalledProcessError:
        print("espeak not found. Install with: sudo apt-get install espeak")
        return False

    print("Generating test audio with espeak...")
    cmd = [
        "espeak",
        "-w", str(output_path),
        "-s", "150",  # Speed
        "-p", "50",   # Pitch
        text
    ]
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Audio generation failed: {e}")
        return False


def playback_and_record(
    source_audio: Path,
    device: str,
    output_path: Path,
    playback_device: str = "default"
) -> bool:
    """Play audio through speakers and record it (for room acoustics testing)."""
    print(f"Playing audio and recording room response...")

    # Get duration of source audio
    try:
        with wave.open(str(source_audio), 'rb') as wf:
            duration = wf.getnframes() / wf.getframerate()
    except Exception as e:
        print(f"Failed to read source audio: {e}")
        return False

    # Add some padding for room acoustics
    record_duration = int(duration) + 2

    # Start recording in background
    record_cmd = [
        "arecord",
        "-D", device,
        "-d", str(record_duration),
        "-f", "cd",
        "-t", "wav",
        str(output_path)
    ]

    try:
        record_proc = subprocess.Popen(record_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Brief delay to ensure recording has started
        time.sleep(0.5)

        # Play the audio
        play_cmd = ["aplay", str(source_audio)]
        subprocess.run(play_cmd, check=True, capture_output=True)

        # Wait for recording to complete
        record_proc.wait(timeout=record_duration + 5)
        return record_proc.returncode == 0

    except Exception as e:
        print(f"Playback/record failed: {e}")
        return False


def build_noise_profile(audio_path: Path, noise_profile_path: Path) -> bool:
    """Build a noise profile from the first 0.5 seconds of audio."""
    cmd = [
        "sox", str(audio_path), "-n",
        "trim", "0", "0.5",
        "noiseprof", str(noise_profile_path)
    ]
    try:
        run_command(cmd)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to build noise profile: {e}")
        return False


def build_sox_command(
    input_path: Path,
    output_path: Path,
    noise_profile: Path,
    params: Dict[str, str]
) -> List[str]:
    """Build the sox command from parameters."""
    cmd = [
        "sox", str(input_path), str(output_path),
        "noisered", str(noise_profile), params["noisered"],
        "highpass", params["highpass"],
        "lowpass", params["lowpass"],
        "compand", f"{params['compand_attack']},{params['compand_decay']}", "6:-70,-65,-40", "-5", "-90", "0.05",
        "equalizer", params["eq1_freq"], params["eq1_width"], params["eq1_gain"],
        "equalizer", params["eq2_freq"], params["eq2_width"], params["eq2_gain"],
        "norm", "-3",
        "rate", "16k",
        "channels", "1"
    ]
    return cmd


def apply_sox_processing(
    input_path: Path,
    output_path: Path,
    noise_profile: Path,
    params: Dict[str, str]
) -> Tuple[bool, List[str]]:
    """Apply sox processing with given parameters."""
    cmd = build_sox_command(input_path, output_path, noise_profile, params)
    try:
        run_command(cmd)
        return True, cmd
    except subprocess.CalledProcessError as e:
        return False, cmd


def transcribe(url: str, model: str, wav_path: Path) -> str:
    """Transcribe audio file using the transcription service."""
    import requests
    with open(wav_path, 'rb') as f:
        files = {'file': f}
        params = {'model': model}
        try:
            r = requests.post(url, files=files, params=params, timeout=60)
            r.raise_for_status()
            return r.json().get('text', '').strip()
        except Exception as e:
            return f"ERROR: {e}"


def calculate_word_accuracy(reference: str, transcript: str) -> float:
    """
    Calculate word-level accuracy between reference and transcript.
    Uses a simple word overlap metric.
    """
    if not reference or not transcript:
        return 0.0

    # Normalize texts
    ref_words = set(reference.lower().split())
    trans_words = set(transcript.lower().split())

    if not ref_words:
        return 0.0

    # Calculate overlap
    common = ref_words.intersection(trans_words)
    precision = len(common) / len(trans_words) if trans_words else 0.0
    recall = len(common) / len(ref_words)

    # F1 score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def calculate_transcript_quality(transcript: str) -> float:
    """
    Calculate transcript quality score based on heuristics.
    Higher is better.
    """
    if not transcript or transcript.startswith("ERROR:"):
        return 0.0

    # Penalize very short transcripts
    word_count = len(transcript.split())
    if word_count < 3:
        return 0.1

    # Calculate character quality
    total_chars = len(transcript)
    alnum_chars = sum(c.isalnum() or c.isspace() for c in transcript)
    punct_chars = sum(c in ",.;:!?-'\"" for c in transcript)
    quality_ratio = (alnum_chars + punct_chars) / total_chars if total_chars > 0 else 0.0

    # Bonus for longer transcripts (more content captured)
    length_bonus = min(1.0, word_count / 20)

    return quality_ratio * 0.7 + length_bonus * 0.3


def generate_parameter_combinations(param_space: Dict[str, List[str]], max_combinations: int = 100) -> List[Dict[str, str]]:
    """Generate parameter combinations to test."""
    keys = list(param_space.keys())
    values = [param_space[k] for k in keys]

    # Generate all combinations
    all_combinations = list(itertools.product(*values))

    # If too many, sample a subset
    if len(all_combinations) > max_combinations:
        # Always include baseline
        combinations = [tuple(BASELINE_PARAMS[k] for k in keys)]

        # Sample additional combinations
        import random
        random.seed(42)  # For reproducibility
        sampled = random.sample(all_combinations, max_combinations - 1)
        combinations.extend(sampled)
    else:
        combinations = all_combinations

    # Convert to dicts
    return [dict(zip(keys, combo)) for combo in combinations]


def format_sox_settings(params: Dict[str, str]) -> str:
    """Format parameters as a Python dict string for main.py."""
    lines = [
        f"# Room-calibrated sox settings",
        f"# Generated by tools/room_calibration.py",
        f"NOISERED_AMOUNT = \"{params['noisered']}\"",
        f"HIGHPASS_FREQ = \"{params['highpass']}\"",
        f"LOWPASS_FREQ = \"{params['lowpass']}\"",
        f"COMPAND_ATTACK = \"{params['compand_attack']}\"",
        f"COMPAND_DECAY = \"{params['compand_decay']}\"",
        f"EQ1_FREQ = \"{params['eq1_freq']}\"",
        f"EQ1_WIDTH = \"{params['eq1_width']}\"",
        f"EQ1_GAIN = \"{params['eq1_gain']}\"",
        f"EQ2_FREQ = \"{params['eq2_freq']}\"",
        f"EQ2_WIDTH = \"{params['eq2_width']}\"",
        f"EQ2_GAIN = \"{params['eq2_gain']}\"",
    ]
    return "\n".join(lines)


def run_calibration(
    input_audio: Path,
    reference_text: Optional[str],
    url: str,
    model: str,
    param_space: Dict[str, List[str]],
    max_combinations: int,
    workdir: Path,
    verbose: bool = False
) -> List[CalibrationResult]:
    """Run calibration by testing different parameter combinations."""
    results = []

    # Build noise profile
    noise_profile = workdir / "noise.prof"
    if not build_noise_profile(input_audio, noise_profile):
        print("Failed to build noise profile")
        return results

    # Generate parameter combinations
    combinations = generate_parameter_combinations(param_space, max_combinations)
    total = len(combinations)

    print(f"\nTesting {total} parameter combinations...")

    for i, params in enumerate(combinations, 1):
        output_wav = workdir / f"variant_{i:04d}.wav"

        # Apply sox processing
        success, sox_cmd = apply_sox_processing(input_audio, output_wav, noise_profile, params)
        if not success:
            if verbose:
                print(f"  [{i}/{total}] FAILED: sox processing error")
            continue

        # Transcribe
        transcript = transcribe(url, model, output_wav)

        # Calculate score
        if reference_text:
            score = calculate_word_accuracy(reference_text, transcript)
        else:
            score = calculate_transcript_quality(transcript)

        result = CalibrationResult(
            params=params,
            transcript=transcript,
            accuracy_score=score,
            wav_path=str(output_wav),
            sox_command=sox_cmd
        )
        results.append(result)

        if verbose:
            print(f"  [{i}/{total}] Score: {score:.3f} | Transcript: {transcript[:50]}...")
        else:
            # Progress indicator
            if i % 10 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({100*i//total}%)")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Room calibration tool for finding optimal sox audio processing parameters."
    )

    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", type=Path,
        help="Path to existing audio file to use for calibration"
    )
    input_group.add_argument(
        "--record", action="store_true",
        help="Record a speech sample for calibration"
    )
    input_group.add_argument(
        "--generate", action="store_true",
        help="Generate test audio using text-to-speech and record room playback"
    )

    # Recording options
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"Audio recording device (default: {DEFAULT_DEVICE})"
    )
    parser.add_argument(
        "--duration", type=int, default=10,
        help="Recording duration in seconds (default: 10)"
    )

    # Reference text for accuracy comparison
    parser.add_argument(
        "--reference", type=str,
        help="Reference transcript text for accuracy comparison"
    )

    # Transcription service options
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Transcription service URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )

    # Calibration options
    parser.add_argument(
        "--quick", action="store_true",
        help="Use reduced parameter space for faster calibration"
    )
    parser.add_argument(
        "--max-combinations", type=int, default=50,
        help="Maximum number of parameter combinations to test (default: 50)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed progress"
    )

    # Output options
    parser.add_argument(
        "--output", type=Path,
        help="Output file for calibration results (JSON)"
    )
    parser.add_argument(
        "--keep-workdir", action="store_true",
        help="Keep working directory with all generated files"
    )

    args = parser.parse_args()

    # Create working directory
    workdir = Path(tempfile.mkdtemp(prefix='room_calibration_'))
    print(f"Working directory: {workdir}")

    try:
        # Get or create input audio
        if args.input:
            if not args.input.exists():
                print(f"Input file not found: {args.input}")
                sys.exit(1)
            input_audio = args.input
            print(f"Using input audio: {input_audio}")

        elif args.record:
            input_audio = workdir / "recorded_sample.wav"
            print("\n" + "=" * 50)
            print("RECORDING MODE")
            print("=" * 50)
            print(f"Please speak clearly for {args.duration} seconds.")
            print("For best results, read a known passage aloud.")
            print("Press Enter to start recording...")
            input()

            if not record_audio(args.device, input_audio, args.duration):
                print("Recording failed")
                sys.exit(1)
            print(f"Recorded to: {input_audio}")

        elif args.generate:
            print("\n" + "=" * 50)
            print("GENERATED AUDIO MODE")
            print("=" * 50)
            print("This mode generates speech and records it through your room.")
            print("1. Generating test speech audio...")

            tts_audio = workdir / "tts_audio.wav"
            if not generate_test_audio(TEST_PHRASE, tts_audio):
                print("Failed to generate test audio")
                sys.exit(1)

            print("2. Playing audio and recording room response...")
            print("   Make sure speakers are on and microphone is positioned.")
            print("Press Enter to start playback and recording...")
            input()

            input_audio = workdir / "room_recording.wav"
            if not playback_and_record(tts_audio, args.device, input_audio):
                print("Playback/recording failed")
                sys.exit(1)

            # Use the test phrase as reference
            if not args.reference:
                args.reference = TEST_PHRASE

        # Select parameter space
        param_space = QUICK_PARAMETER_SPACE if args.quick else PARAMETER_SPACE

        # Run calibration
        print("\n" + "=" * 50)
        print("CALIBRATION")
        print("=" * 50)

        results = run_calibration(
            input_audio=input_audio,
            reference_text=args.reference,
            url=args.url,
            model=args.model,
            param_space=param_space,
            max_combinations=args.max_combinations,
            workdir=workdir,
            verbose=args.verbose
        )

        if not results:
            print("No successful calibration results")
            sys.exit(1)

        # Sort by accuracy score (descending)
        results.sort(key=lambda r: r.accuracy_score, reverse=True)

        # Print results summary
        print("\n" + "=" * 50)
        print("CALIBRATION RESULTS")
        print("=" * 50)

        print("\nTop 5 configurations:")
        for i, result in enumerate(results[:5], 1):
            print(f"\n{i}. Score: {result.accuracy_score:.3f}")
            print(f"   Transcript: {result.transcript[:80]}...")
            print(f"   Key params: noisered={result.params['noisered']}, "
                  f"highpass={result.params['highpass']}, "
                  f"lowpass={result.params['lowpass']}")

        # Best result
        best = results[0]
        print("\n" + "=" * 50)
        print("RECOMMENDED SETTINGS")
        print("=" * 50)
        print(f"\nBest configuration (score: {best.accuracy_score:.3f}):\n")
        print(format_sox_settings(best.params))

        print("\n\nSox command:")
        print(" ".join(best.sox_command))

        # Save results
        output_file = args.output or workdir / "calibration_results.json"
        results_data = {
            "best_params": best.params,
            "best_score": best.accuracy_score,
            "best_transcript": best.transcript,
            "sox_command": best.sox_command,
            "all_results": [
                {
                    "params": r.params,
                    "score": r.accuracy_score,
                    "transcript": r.transcript
                }
                for r in results[:10]  # Save top 10
            ]
        }
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {output_file}")

        # Cleanup
        if not args.keep_workdir and args.output:
            # Copy output file if specified
            if args.output != output_file:
                shutil.copy(output_file, args.output)

    finally:
        if not args.keep_workdir:
            print(f"\nCleaning up working directory: {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"\nWorking directory preserved: {workdir}")


if __name__ == "__main__":
    main()
