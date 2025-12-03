#!/usr/bin/env python3
"""
Continuous audio logger with transcription.
Records 1-minute segments, cleans audio, transcribes, and logs to daily files.
"""

import os
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import psutil
import wave

# Configuration
AUDIO_DEVICE = "plughw:3,0"
RECORD_DURATION = 60  # seconds
TRANSCRIBE_URL = "http://192.168.0.142:8085/transcribe"
TRANSCRIBE_MODEL = "base"
LOG_DIR = Path(__file__).parent / "logs"
TEMP_DIR = Path(__file__).parent / "temp"

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 2
SILENCE_THRESHOLD = 500  # RMS threshold for silence detection
VLC_CPU_THRESHOLD = 5.0  # percent; if VLC above this, assume playing
VLC_CHECK_INTERVAL = 0.5  # seconds for cpu sampling


def setup_directories():
    """Create necessary directories."""
    LOG_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)


def get_daily_log_path():
    """Get the path for today's log file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"audio_log_{date_str}.txt"


def record_audio(output_path: Path, duration: int) -> bool:
    """Record audio to a WAV file."""
    try:
        cmd = [
            "arecord",
            "-D", AUDIO_DEVICE,
            "-d", str(duration),
            "-f", "cd",
            "-t", "wav",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Recording failed: {e}")
        return False


def is_audio_silent(wav_path: Path) -> bool:
    """Check if audio file is mostly silent using RMS energy."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            # Read audio data
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16)
            
            # Calculate RMS (Root Mean Square) energy
            rms = np.sqrt(np.mean(audio_data.astype(float) ** 2))
            
            # Return True if RMS is below threshold
            return rms < SILENCE_THRESHOLD
    except Exception as e:
        print(f"Error checking silence: {e}")
        return False


def clean_audio(input_path: Path, output_path: Path) -> bool:
    """Clean audio using sox: noise reduction, filters, normalization."""
    try:
        # Create noise profile
        noise_prof = TEMP_DIR / f"noise_{time.time()}.prof"
        
        # Generate noise profile from first 0.5 seconds
        cmd1 = [
            "sox", str(input_path), "-n",
            "trim", "0", "0.5",
            "noiseprof", str(noise_prof)
        ]
        subprocess.run(cmd1, check=True, capture_output=True)
        
        # Apply noise reduction and filters
        cmd2 = [
            "sox", str(input_path), str(output_path),
            "noisered", str(noise_prof), "0.21",
            "highpass", "200",
            "treble", "6",
            "norm", "-3"
        ]
        subprocess.run(cmd2, check=True, capture_output=True)
        
        # Clean up noise profile
        noise_prof.unlink(missing_ok=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Audio cleaning failed: {e}")
        return False


def is_vlc_playing() -> bool:
    """Detect if VLC is running and actively playing by CPU usage."""
    try:
        vlc_procs = []
        for p in psutil.process_iter(attrs=["name"]):
            if p.info.get("name", "").lower() == "vlc":
                vlc_procs.append(p)
        if not vlc_procs:
            return False

        # Prime cpu_percent measurements
        for p in vlc_procs:
            try:
                p.cpu_percent(interval=None)
            except Exception:
                pass
        # Sample over interval
        time.sleep(VLC_CHECK_INTERVAL)
        for p in vlc_procs:
            try:
                cpu = p.cpu_percent(interval=None)
                if cpu >= VLC_CPU_THRESHOLD:
                    return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def transcribe_audio(audio_path: Path) -> str:
    """Transcribe audio file using the transcription service."""
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            params = {'model': TRANSCRIBE_MODEL}
            response = requests.post(TRANSCRIBE_URL, files=files, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('text', '').strip()
            
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""


def log_transcript(text: str):
    """Append transcript with timestamp to daily log file."""
    if not text:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_daily_log_path()
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {text}\n")
        print(f"[{timestamp}] Logged: {text}")
    except Exception as e:
        print(f"Failed to write log: {e}")


def cleanup_temp_files(max_age_hours: int = 1):
    """Remove old temporary files."""
    try:
        cutoff_time = time.time() - (max_age_hours * 3600)
        for temp_file in TEMP_DIR.glob("*"):
            if temp_file.stat().st_mtime < cutoff_time:
                temp_file.unlink()
    except Exception as e:
        print(f"Cleanup failed: {e}")


def main():
    """Main loop: record, clean, transcribe, log."""
    print("Starting audio logger service...")
    setup_directories()
    
    while True:
        try:
            # Pause logging if VLC is actively playing
            if is_vlc_playing():
                print("VLC activity detected; pausing recording for this cycle.")
                time.sleep(5)
                continue
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_audio = TEMP_DIR / f"raw_{timestamp}.wav"
            clean_audio_path = TEMP_DIR / f"clean_{timestamp}.wav"
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Recording for {RECORD_DURATION} seconds...")
            
            # Record audio
            if not record_audio(raw_audio, RECORD_DURATION):
                print("Recording failed, retrying...")
                time.sleep(5)
                continue
            
            # Check if audio is silent
            if is_audio_silent(raw_audio):
                print("Audio is silent, skipping transcription.")
                raw_audio.unlink()
                continue
            
            # Clean audio
            print("Cleaning audio...")
            if not clean_audio(raw_audio, clean_audio_path):
                print("Audio cleaning failed, skipping transcription.")
                raw_audio.unlink()
                continue
            
            # Transcribe
            print("Transcribing...")
            transcript = transcribe_audio(clean_audio_path)
            
            # Log the result
            if transcript:
                log_transcript(transcript)
            else:
                print("No transcript generated.")
            
            # Cleanup temporary files
            raw_audio.unlink(missing_ok=True)
            clean_audio_path.unlink(missing_ok=True)
            cleanup_temp_files()
            
        except KeyboardInterrupt:
            print("\nStopping audio logger...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
