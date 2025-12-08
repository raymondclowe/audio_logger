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

    # Use custom text file
    python tools/guided_calibration.py --text my_calibration_text.txt

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

# Sample text file path (shared with room_calibration-example.txt)
SAMPLE_TEXT_FILE = Path(__file__).parent / "room_calibration-example.txt"

# Load sample text at module level for testing
SAMPLE_TEXT = """AFTER having been twice driven back by heavy southwestern gales, Her Majesty's ship Beagle, a ten-gun brig, under the command of Captain Fitz Roy, R. N., sailed from Devonport on the 27th of December, 1831. The object of the expedition was to complete the survey of Patagonia and Tierra del Fuego, commenced under Captain King in 1826 to 1830,--to survey the shores of Chile, Peru, and of some islands in the Pacific--and to carry a chain of chronometrical measurements round the World. On the 6th of January we reached Teneriffe, but were prevented landing, by fears of our bringing the cholera: the next morning we saw the sun rise behind the rugged outline of the Grand Canary island, and suddenly illuminate the Peak of Teneriffe, whilst the lower parts were veiled in fleecy clouds. This was the first of many delightful days never to be forgotten. On the 16th of January, 1832, we anchored at Porto Praya, in St. Jago, the chief island of the Cape de Verd archipelago."""

if SAMPLE_TEXT_FILE.exists():
    try:
        SAMPLE_TEXT = SAMPLE_TEXT_FILE.read_text().strip()
    except Exception:
        pass  # Use fallback text above


def load_sample_text(text_file: Optional[Path] = None) -> str:
    """Load sample text from file for consistency with other tools."""
    # Use custom text file if provided
    if text_file is not None:
        if text_file.exists():
            try:
                return text_file.read_text().strip()
            except Exception as e:
                print(f"⚠ Warning: Could not read custom text file: {e}")
                print(f"  Falling back to default sample text")
        else:
            print(f"⚠ Warning: Custom text file not found: {text_file}")
            print(f"  Falling back to default sample text")
    
    # Try default sample text file
    if SAMPLE_TEXT_FILE.exists():
        try:
            return SAMPLE_TEXT_FILE.read_text().strip()
        except Exception:
            pass
    
    # Fallback text if file is not available
    return """AFTER having been twice driven back by heavy southwestern gales, Her Majesty's ship Beagle, a ten-gun brig, under the command of Captain Fitz Roy, R. N., sailed from Devonport on the 27th of December, 1831. The object of the expedition was to complete the survey of Patagonia and Tierra del Fuego, commenced under Captain King in 1826 to 1830,--to survey the shores of Chile, Peru, and of some islands in the Pacific--and to carry a chain of chronometrical measurements round the World. On the 6th of January we reached Teneriffe, but were prevented landing, by fears of our bringing the cholera: the next morning we saw the sun rise behind the rugged outline of the Grand Canary island, and suddenly illuminate the Peak of Teneriffe, whilst the lower parts were veiled in fleecy clouds. This was the first of many delightful days never to be forgotten. On the 16th of January, 1832, we anchored at Porto Praya, in St. Jago, the chief island of the Cape de Verd archipelago."""


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
    print("  3. Stay SILENT during the 3-second countdown (captures ambient noise)")
    print("  4. Read the sample text aloud clearly")
    print("  5. Wait while the system finds the best settings")
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


def calculate_recording_duration(text: str, words_per_minute: int = 150) -> int:
    """
    Calculate recording duration based on text length.
    
    Args:
        text: The text to be read
        words_per_minute: Average speaking speed (default 150 wpm, which is moderate pace)
    
    Returns:
        Recommended recording duration in seconds with buffer time
    """
    word_count = len(text.split())
    # Calculate base time needed
    base_seconds = int((word_count / words_per_minute) * 60)
    # Add 30 second buffer for pauses and slower speakers
    return base_seconds + 30


def record_audio_16k_mono(device: str, output_path: Path, duration: int = 60) -> bool:
    """
    Record audio at 16kHz mono WAV format.
    
    This is the format optimized for speech transcription.
    User can press Enter to stop recording early.
    Starts with 3-second countdown for ambient noise capture.
    """
    print(f"\n🎤 Starting recording...")
    
    cmd = [
        "arecord",
        "-D", device,
        "-f", "S16_LE",      # 16-bit signed little endian
        "-r", "16000",       # 16kHz sample rate
        "-c", "1",           # Mono
        "-t", "wav",
        str(output_path)
    ]
    
    try:
        import threading
        import select
        
        # Start recording process BEFORE countdown to capture ambient noise
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Brief delay to ensure recording has started
        time.sleep(0.2)
        
        print("   Stay SILENT during countdown (capturing ambient noise)...")
        
        # 3-second countdown for ambient noise capture - recording is already happening
        for i in range(3, 0, -1):
            print(f"   {i}...")
            time.sleep(1)
        
        print(f"   🔴 NOW READING (up to {duration} seconds)")
        print("   (Read the sample text aloud now)")
        print("   Press ENTER when you're done reading to stop early")
        print()
        
        stop_recording = threading.Event()
        
        def wait_for_enter():
            """Wait for user to press Enter in a separate thread."""
            try:
                # Use select to check if input is available (works on Unix systems)
                import sys
                if select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()  # Clear any buffered input
                input()  # Wait for Enter
                stop_recording.set()
            except:
                pass
        
        # Start thread to wait for Enter key
        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()
        
        # Show countdown while recording (account for 3 seconds already elapsed during countdown)
        for elapsed in range(3, duration):
            remaining = duration - elapsed
            print(f"   Recording... {elapsed}s elapsed, {remaining}s remaining (or press ENTER to finish)", end="\r")
            time.sleep(1)
            
            if stop_recording.is_set() or process.poll() is not None:
                break
        
        # Stop the recording
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        print(" " * 80, end="\r")  # Clear countdown line
        
        if output_path.exists() and output_path.stat().st_size > 1000:
            print("✓ Recording complete!")
            return True
        else:
            print("✗ Recording failed or file too small")
            return False
            
    except Exception as e:
        print(f"✗ Recording error: {e}")
        if process.poll() is None:
            process.kill()
        return False


def display_sample_text(text: str):
    """Display the sample text for the user to read."""
    print("\n" + "=" * 70)
    print("PLEASE READ THE FOLLOWING TEXT ALOUD:")
    print("=" * 70)
    print()
    
    # Format text with line wrapping for readability
    words = text.split()
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
    recording_duration: Optional[int] = None,
    text_file: Optional[Path] = None,
    adaptive_ranges: bool = False
) -> bool:
    """Run the guided calibration workflow."""
    
    print_banner()
    print_instructions()
    
    # Load the sample text (custom or default)
    sample_text = load_sample_text(text_file)
    
    # Calculate recording duration if not explicitly provided
    if recording_duration is None:
        recording_duration = calculate_recording_duration(sample_text)
    
    # Show detected device
    print(f"Audio device: {device}")
    print(f"Transcription URL: {url}")
    print(f"Model: {model}")
    print(f"Optimization trials: {trials}")
    print(f"Text length: {len(sample_text.split())} words")
    print(f"Recommended recording time: {recording_duration} seconds")
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
    print("When recording starts:")
    print("  • Stay SILENT during the 3-second countdown (for noise profiling)")
    print("  • Then read the text clearly")
    print("  • Press ENTER when finished to stop early")
    print()
    input("Press ENTER when you are ready to begin recording...")
    print()
    
    # Display sample text
    display_sample_text(sample_text)
    
    # Record audio (includes countdown)
    audio_path = workdir / "guided_calibration_sample.wav"
    if not record_audio_16k_mono(device, audio_path, recording_duration):
        print("\n✗ Failed to record audio. Please check your microphone.")
        return False
    
    # Build noise profile from the initial silence
    print("\n📊 Building noise profile from ambient sound...")
    noise_profile = workdir / "noise.prof"
    if not build_noise_profile(audio_path, noise_profile):
        print("✗ Failed to build noise profile")
        return False
    
    # Run optimization
    print("\n🔍 Finding optimal settings using Bayesian optimization...")
    if adaptive_ranges:
        print("   Using adaptive range expansion (3 phases)")
    print("   This may take several minutes depending on transcription service speed.")
    print()
    
    if HAS_OPTUNA:
        results = run_bayesian_optimization(
            input_audio=audio_path,
            reference_text=sample_text,
            url=url,
            model=model,
            noise_profile=noise_profile,
            workdir=workdir,
            n_trials=trials,
            metric="wer",
            verbose=verbose,
            adaptive_ranges=adaptive_ranges
        )
    else:
        # Fall back to random sampling
        results = run_calibration(
            input_audio=audio_path,
            reference_text=sample_text,
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
    
    # Sort results by composite score: 90% WER + 10% CER
    # This prioritizes WER but uses CER as a tiebreaker
    results.sort(key=lambda r: (0.9 * r.accuracy_score + 0.1 * r.cer_score), reverse=True)
    best = results[0]
    
    # Display results
    print("\n" + "=" * 70)
    print("                    CALIBRATION RESULTS")
    print("=" * 70)
    print()
    composite = 0.9 * best.accuracy_score + 0.1 * best.cer_score
    print(f"Best WER Score: {best.accuracy_score:.3f}")
    print(f"Best CER Score: {best.cer_score:.3f}")
    print(f"Combined Score: {composite:.3f} (90% WER + 10% CER)")
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
        "best_wer_score": best.accuracy_score,
        "best_cer_score": best.cer_score,
        "best_transcript": best.transcript,
        "sox_command": best.sox_command,
        "reference_text": sample_text[:100] + "...",
        "calibration_device": device,
        "all_results": [
            {
                "params": r.params,
                "wer_score": r.accuracy_score,
                "cer_score": r.cer_score,
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
        "--adaptive-ranges", action="store_true",
        help="Automatically expand parameter ranges when boundaries are hit (runs in 3 phases)"
    )
    parser.add_argument(
        "--duration", type=int,
        help="Recording duration in seconds (auto-calculated based on text length if not specified)"
    )
    parser.add_argument(
        "--text",
        help="Path to custom text file to read (uses default sample text if not specified)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed progress"
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified
    device = args.device or detect_audio_device()
    
    # Normalize URL - append /transcribe if not present
    url = args.url
    if not url.endswith('/transcribe'):
        url = url.rstrip('/') + '/transcribe'
    
    # Parse text file path if provided
    text_file = Path(args.text) if args.text else None
    
    try:
        success = run_guided_calibration(
            device=device,
            url=url,
            model=args.model,
            trials=args.trials,
            verbose=args.verbose,
            recording_duration=args.duration,
            text_file=text_file,
            adaptive_ranges=args.adaptive_ranges
        )
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nCalibration cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
