#!/usr/bin/env python3
"""
Guided Room Calibration Tool for Audio Logger.

This tool provides an interactive, guided workflow for room calibration that:
1. Prompts the user with sample text to read
2. Records the audio at 16kHz mono WAV
3. Uses Bayesian optimization with sox parameters to find the best Word Error Rate
4. Saves optimal parameters to a config file loaded by the main transcriber

This is a streamlined alternative to the full room_calibration.py tool,
designed for ease of use from the command line.

Usage:
    # Run guided calibration (will auto-detect device)
    python tools/guided_calibration.py

    # Specify audio device
    python tools/guided_calibration.py --device plughw:3,0

    # Use a different number of optimization trials
    python tools/guided_calibration.py --trials 100

    # Verbose output
    python tools/guided_calibration.py -v
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Import from room_calibration module
from room_calibration import (
    BASELINE_PARAMS,
    CALIBRATION_CONFIG_FILE,
    DEFAULT_DEVICE,
    DEFAULT_MODEL,
    DEFAULT_URL,
    build_noise_profile,
    calculate_word_accuracy,
    run_bayesian_optimization,
    run_calibration,
    PARAMETER_SPACE,
    HAS_OPTUNA,
)

# Sample text for guided calibration - Darwin's Voyage excerpt
# This is the same text used in room_calibration-example.txt
SAMPLE_TEXT = """AFTER having been twice driven back by heavy southwestern gales, Her Majesty's ship Beagle, a ten-gun brig, under the command of Captain Fitz Roy, R. N., sailed from Devonport on the 27th of December, 1831. The object of the expedition was to complete the survey of Patagonia and Tierra del Fuego, commenced under Captain King in 1826 to 1830, to survey the shores of Chile, Peru, and of some islands in the Pacific, and to carry a chain of chronometrical measurements round the World. On the 6th of January we reached Teneriffe, but were prevented landing, by fears of our bringing the cholera: the next morning we saw the sun rise behind the rugged outline of the Grand Canary island, and suddenly illuminate the Peak of Teneriffe, whilst the lower parts were veiled in fleecy clouds. This was the first of many delightful days never to be forgotten. On the 16th of January, 1832, we anchored at Porto Praya, in St. Jago, the chief island of the Cape de Verd archipelago."""


def print_banner():
    """Print welcome banner."""
    print()
    print("=" * 70)
    print("          GUIDED ROOM CALIBRATION FOR AUDIO LOGGER")
    print("=" * 70)
    print()


def print_instructions():
    """Print calibration instructions."""
    print("This tool will help you find the optimal audio processing settings")
    print("for your room and microphone setup.")
    print()
    print("WHAT YOU NEED TO DO:")
    print("  1. Position yourself in your normal speaking location")
    print("  2. Ensure your microphone is properly connected")
    print("  3. Read the sample text aloud when prompted")
    print("  4. Wait while the system finds the best settings")
    print()


def detect_audio_device() -> str:
    """Auto-detect audio recording device."""
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        devices = []
        for line in result.stdout.split('\n'):
            if line.startswith('card '):
                parts = line.split(':')
                if len(parts) >= 2:
                    card_info = parts[0].strip()
                    card_num = card_info.split()[1]
                    device_info = parts[1].strip().split(',')
                    if len(device_info) >= 2:
                        device_name = device_info[0].strip()
                        device_num_str = device_info[1].strip().split()[1]
                        devices.append((card_num, device_num_str, device_name))
        
        if not devices:
            return DEFAULT_DEVICE
        
        # Filter out non-microphone devices
        filtered = []
        for card, device, name in devices:
            name_lower = name.lower()
            if any(skip in name_lower for skip in ["hdmi", "loopback", "sof-hda-dsp"]):
                continue
            filtered.append((card, device, name))
        
        if filtered:
            card, device, name = filtered[0]
            return f"plughw:{card},{device}"
        elif devices:
            card, device, _ = devices[0]
            return f"plughw:{card},{device}"
        
        return DEFAULT_DEVICE
        
    except Exception:
        return DEFAULT_DEVICE


def record_audio_16k_mono(device: str, output_path: Path, duration: int = 30) -> bool:
    """
    Record audio at 16kHz mono WAV format.
    
    This is the format optimized for speech transcription.
    """
    print(f"\n🎤 Recording for {duration} seconds...")
    print("   (Read the sample text aloud now)")
    print()
    
    cmd = [
        "arecord",
        "-D", device,
        "-d", str(duration),
        "-f", "S16_LE",      # 16-bit signed little endian
        "-r", "16000",       # 16kHz sample rate
        "-c", "1",           # Mono
        "-t", "wav",
        str(output_path)
    ]
    
    try:
        # Show countdown while recording
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        for remaining in range(duration, 0, -1):
            print(f"   Recording... {remaining} seconds remaining", end="\r")
            time.sleep(1)
            if process.poll() is not None:
                break
        
        process.wait(timeout=5)
        print(" " * 50, end="\r")  # Clear countdown line
        
        if process.returncode == 0 and output_path.exists():
            print("✓ Recording complete!")
            return True
        else:
            print("✗ Recording failed")
            return False
            
    except Exception as e:
        print(f"✗ Recording error: {e}")
        return False


def display_sample_text():
    """Display the sample text for the user to read."""
    print("\n" + "=" * 70)
    print("PLEASE READ THE FOLLOWING TEXT ALOUD:")
    print("=" * 70)
    print()
    
    # Format text with line wrapping for readability
    words = SAMPLE_TEXT.split()
    line = ""
    for word in words:
        if len(line) + len(word) + 1 > 68:
            print(f"  {line}")
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        print(f"  {line}")
    
    print()
    print("=" * 70)
    print()


def run_guided_calibration(
    device: str,
    url: str,
    model: str,
    trials: int,
    verbose: bool = False,
    recording_duration: int = 30
) -> bool:
    """Run the guided calibration workflow."""
    
    print_banner()
    print_instructions()
    
    # Show detected device
    print(f"Audio device: {device}")
    print(f"Transcription URL: {url}")
    print(f"Model: {model}")
    print(f"Optimization trials: {trials}")
    print()
    
    # Check if Bayesian optimization is available
    if not HAS_OPTUNA:
        print("⚠ Warning: optuna not installed. Using random sampling instead.")
        print("  For better results, install with: pip install optuna")
        print()
    
    # Create working directory
    workdir = Path(__file__).parent / "calibration_output"
    workdir.mkdir(exist_ok=True)
    
    # Prompt user to start
    print("-" * 70)
    input("Press ENTER when you are ready to begin recording...")
    print()
    
    # Display sample text
    display_sample_text()
    
    # Countdown before recording
    print("Starting in...")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print()
    
    # Record audio
    audio_path = workdir / "guided_calibration_sample.wav"
    if not record_audio_16k_mono(device, audio_path, recording_duration):
        print("\n✗ Failed to record audio. Please check your microphone.")
        return False
    
    # Build noise profile
    print("\n📊 Analyzing audio...")
    noise_profile = workdir / "noise.prof"
    if not build_noise_profile(audio_path, noise_profile):
        print("✗ Failed to build noise profile")
        return False
    
    # Run optimization
    print("\n🔍 Finding optimal settings using Bayesian optimization...")
    print("   This may take several minutes depending on transcription service speed.")
    print()
    
    if HAS_OPTUNA:
        results = run_bayesian_optimization(
            input_audio=audio_path,
            reference_text=SAMPLE_TEXT,
            url=url,
            model=model,
            noise_profile=noise_profile,
            workdir=workdir,
            n_trials=trials,
            metric="wer",
            verbose=verbose
        )
    else:
        # Fall back to random sampling
        results = run_calibration(
            input_audio=audio_path,
            reference_text=SAMPLE_TEXT,
            url=url,
            model=model,
            param_space=PARAMETER_SPACE,
            max_combinations=trials,
            workdir=workdir,
            noise_profile=noise_profile,
            metric="wer",
            verbose=verbose
        )
    
    if not results:
        print("\n✗ Calibration failed - no results produced")
        return False
    
    # Sort results by accuracy
    results.sort(key=lambda r: r.accuracy_score, reverse=True)
    best = results[0]
    
    # Display results
    print("\n" + "=" * 70)
    print("                    CALIBRATION RESULTS")
    print("=" * 70)
    print()
    print(f"Best Word Error Rate Score: {best.accuracy_score:.3f}")
    print(f"Best Transcription: {best.transcript[:100]}...")
    print()
    
    # Show parameter comparison with baseline
    print("Optimal Parameters (compared to baseline):")
    print("-" * 50)
    for key in BASELINE_PARAMS:
        baseline_val = BASELINE_PARAMS[key]
        new_val = best.params.get(key, baseline_val)
        if baseline_val != new_val:
            print(f"  {key}: {baseline_val} → {new_val} (changed)")
        else:
            print(f"  {key}: {new_val}")
    print()
    
    # Save results to config file
    results_data = {
        "best_params": best.params,
        "best_score": best.accuracy_score,
        "best_transcript": best.transcript,
        "sox_command": best.sox_command,
        "reference_text": SAMPLE_TEXT[:100] + "...",
        "calibration_device": device,
        "all_results": [
            {
                "params": r.params,
                "score": r.accuracy_score,
                "transcript": r.transcript
            }
            for r in results[:10]
        ]
    }
    
    try:
        with open(CALIBRATION_CONFIG_FILE, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"✓ Configuration saved to: {CALIBRATION_CONFIG_FILE}")
        print()
        print("=" * 70)
        print("                    CALIBRATION COMPLETE!")
        print("=" * 70)
        print()
        print("The main transcriber will automatically use these optimized settings")
        print("on its next run.")
        print()
        return True
        
    except Exception as e:
        print(f"✗ Failed to save configuration: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Guided room calibration tool for finding optimal audio settings."
    )
    
    parser.add_argument(
        "--device",
        help="Audio recording device (auto-detected if not specified)"
    )
    parser.add_argument(
        "--url", default=DEFAULT_URL,
        help=f"Transcription service URL (default: {DEFAULT_URL})"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Whisper model to use (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of optimization trials (default: 50)"
    )
    parser.add_argument(
        "--duration", type=int, default=60,
        help="Recording duration in seconds (default: 60)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device or detect_audio_device()
    
    try:
        success = run_guided_calibration(
            device=device,
            url=args.url,
            model=args.model,
            trials=args.trials,
            verbose=args.verbose,
            recording_duration=args.duration
        )
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
