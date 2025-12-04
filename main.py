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
OVERLAP_DURATION = 2  # seconds - overlap between chunks to avoid word breaks
TRANSCRIBE_URL = "http://192.168.0.142:8085/transcribe"
TRANSCRIBE_MODEL = "small"
LOG_DIR = Path(__file__).parent / "logs"
TEMP_DIR = Path(__file__).parent / "temp"
NOISE_PROFILE = Path(__file__).parent / "temp" / "ambient_noise.prof"

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 2
SILENCE_THRESHOLD = 600  # RMS peak threshold for silence detection
# Windowed RMS config: speech is intermittent; use peak of window RMS
RMS_WINDOW_SECS = 0.05  # 50ms window
RMS_PERCENTILE = 90     # percentile of window RMS (for stats only)
VLC_CPU_THRESHOLD = 5.0  # percent; if VLC above this, assume playing
VLC_CHECK_INTERVAL = 0.5  # seconds for cpu sampling

# Context tracking for transcription continuity
LAST_TRANSCRIPT = None
CONTEXT_WORDS = 30  # Number of words to include from previous transcript
PREVIOUS_AUDIO = None  # Store path to previous recording for overlap


def setup_directories():
    """Create necessary directories."""
    LOG_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)


def get_daily_log_path():
    """Get the path for today's log file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"audio_log_{date_str}.txt"


def create_overlapped_audio(current_path: Path, output_path: Path) -> bool:
    """Create audio with overlap from previous recording."""
    global PREVIOUS_AUDIO
    
    try:
        if PREVIOUS_AUDIO and PREVIOUS_AUDIO.exists():
            # Extract last N seconds from previous audio
            overlap_temp = TEMP_DIR / f"overlap_{time.time()}.wav"
            cmd_extract = [
                "sox", str(PREVIOUS_AUDIO), str(overlap_temp),
                "trim", f"-{OVERLAP_DURATION}"
            ]
            subprocess.run(cmd_extract, check=True, capture_output=True)
            
            # Concatenate overlap + current audio
            cmd_concat = [
                "sox", str(overlap_temp), str(current_path), str(output_path)
            ]
            subprocess.run(cmd_concat, check=True, capture_output=True)
            
            # Cleanup temp overlap file
            overlap_temp.unlink(missing_ok=True)
            return True
        else:
            # No overlap available, just copy current
            import shutil
            shutil.copy(current_path, output_path)
            return True
            
    except Exception as e:
        print(f"Failed to create overlap, using current audio only: {e}")
        import shutil
        shutil.copy(current_path, output_path)
        return True


def record_audio(output_path: Path, duration: int) -> bool:
    """Record audio to a WAV file with gain boost."""
    try:
        # First set ALSA capture volume to maximum for the device
        try:
            subprocess.run(
                ["amixer", "-c", "3", "set", "Capture", "100%"],
                capture_output=True, timeout=2
            )
        except Exception:
            pass  # Continue even if amixer fails
        
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


def is_audio_silent(wav_path: Path) -> tuple[bool, dict]:
    """Check if audio is silent using windowed RMS stats.
    Returns (is_silent, stats) where stats contains mean/peak/percentile RMS."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        # 16-bit PCM expected; downmix to mono for RMS calc
        dtype = np.int16 if sampwidth == 2 else np.int16
        audio = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

        if audio.size == 0:
            return True, {"mean": 0.0, "peak": 0.0, "perc": 0.0}

        # Windowed RMS
        win_samples = max(1, int(framerate * RMS_WINDOW_SECS))
        # Pad to full windows
        pad = (-audio.size) % win_samples
        if pad:
            audio = np.pad(audio, (0, pad), mode='constant')
        windows = audio.reshape(-1, win_samples).astype(np.float64)
        rms_windows = np.sqrt(np.mean(windows ** 2, axis=1))

        mean_rms = float(np.mean(rms_windows))
        peak_rms = float(np.max(rms_windows))
        perc_rms = float(np.percentile(rms_windows, RMS_PERCENTILE))

        stats = {"mean": mean_rms, "peak": peak_rms, "perc": perc_rms}
        # Decide speech presence using peak RMS against threshold
        is_silent = peak_rms < SILENCE_THRESHOLD
        return is_silent, stats
    except Exception as e:
        print(f"Error checking silence: {e}")
        return False, {"mean": 0.0, "peak": 0.0, "perc": 0.0}


def update_noise_profile(audio_path: Path) -> bool:
    """Update the ambient noise profile from a silent recording."""
    try:
        # Use entire silent recording to build comprehensive noise profile
        cmd = [
            "sox", str(audio_path), "-n",
            "noiseprof", str(NOISE_PROFILE)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"Updated noise profile from {audio_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to update noise profile: {e}")
        return False


def clean_audio(input_path: Path, output_path: Path) -> bool:
    """Clean audio using sox: TEST 19 pipeline (noise reduction, filters, compand, EQ, norm, resample)."""
    try:
        # Use adaptive noise profile if available, otherwise create from first 0.5s
        if NOISE_PROFILE.exists():
            noise_prof = NOISE_PROFILE
            cleanup_prof = False
        else:
            noise_prof = TEMP_DIR / f"noise_{time.time()}.prof"
            cleanup_prof = True
            # Generate noise profile from first 0.5 seconds
            cmd1 = [
                "sox", str(input_path), "-n",
                "trim", "0", "0.5",
                "noiseprof", str(noise_prof)
            ]
            subprocess.run(cmd1, check=True, capture_output=True)
        
        # Apply enhanced pipeline: noise reduction, tighter bandpass for speech, compand, dual EQ, normalize, resample to 16kHz mono
        cmd2 = [
            "sox", str(input_path), str(output_path),
            "noisered", str(noise_prof), "0.21",
            "highpass", "300",           # Higher to remove more room rumble/echo
            "lowpass", "3400",            # Slightly brighter, still within speech band
            "compand", "0.03,0.15", "6:-70,-65,-40", "-5", "-90", "0.05",  # Faster attack/release, more aggressive
            "equalizer", "800", "400", "4",   # Boost lower speech frequencies more
            "equalizer", "2500", "800", "3",  # Boost upper speech frequencies
            "norm", "-3",
            "rate", "16k",
            "channels", "1"
        ]
        subprocess.run(cmd2, check=True, capture_output=True)
        
        # Clean up temporary noise profile if created
        if cleanup_prof:
            noise_prof.unlink(missing_ok=True)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Audio cleaning failed: {e}")
        return False


def is_vlc_playing() -> bool:
    """Detect if VLC is running and actively playing by CPU usage."""
    try:
        vlc_procs = []
        for p in psutil.process_iter():
            try:
                if p.name().lower() == "vlc":
                    vlc_procs.append(p)
            except Exception:
                continue
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
    """Transcribe audio file using the transcription service with context."""
    global LAST_TRANSCRIPT
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            params = {'model': TRANSCRIBE_MODEL}
            
            # Add context from previous transcription if available
            if LAST_TRANSCRIPT:
                words = LAST_TRANSCRIPT.split()
                if len(words) > CONTEXT_WORDS:
                    context = ' '.join(words[-CONTEXT_WORDS:])
                else:
                    context = LAST_TRANSCRIPT
                params['prompt'] = f"Continuing conversation: ...{context}"
            
            response = requests.post(TRANSCRIBE_URL, files=files, params=params, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            transcript = result.get('text', '').strip()
            
            # Update context for next transcription
            if transcript:
                LAST_TRANSCRIPT = transcript
            
            return transcript
            
    except Exception as e:
        print(f"Transcription failed: {e}")
        return ""


def log_transcript(text: str, stats: dict = None, filename: str = None):
    """Append transcript with timestamp and audio stats to daily log file."""
    if not text:
        return
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = get_daily_log_path()
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            # Write audio stats for debugging
            if stats:
                f.write(f"[{timestamp}] RMS(mean={stats['mean']:.1f}, peak={stats['peak']:.1f}, p90={stats['perc']:.1f})")
                if filename:
                    f.write(f" file={filename}")
                f.write(f"\n")
            f.write(f"[{timestamp}] {text}\n")
        print(f"[{timestamp}] Logged: {text}")
    except Exception as e:
        print(f"Failed to write log: {e}")


def cleanup_temp_files(max_age_hours: int = 12, keep_last_n_wavs: int = 5):
    """Remove old temporary files, but keep the last N wavs for debugging."""
    try:
        # Prune generic old files by age
        cutoff_time = time.time() - (max_age_hours * 3600)
        for temp_file in TEMP_DIR.glob("*"):
            try:
                # Skip the permanent adaptive noise profile
                if temp_file == NOISE_PROFILE:
                    continue
                # Remove old temp noise profiles immediately
                if temp_file.suffix == ".prof":
                    temp_file.unlink()
                    continue
                # Remove other old temp files by age
                if temp_file.suffix not in {".wav"} and temp_file.stat().st_mtime < cutoff_time:
                    temp_file.unlink()
            except Exception:
                continue

        # Keep last N wav files (raw and clean) by mtime, delete older ones
        wavs = sorted(TEMP_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in wavs[keep_last_n_wavs:]:
            try:
                old.unlink()
            except Exception:
                continue
    except Exception as e:
        print(f"Cleanup failed: {e}")


def main():
    """Main loop: record, clean, transcribe, log."""
    global PREVIOUS_AUDIO
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
            overlapped_audio = TEMP_DIR / f"overlapped_{timestamp}.wav"
            clean_audio_path = TEMP_DIR / f"clean_{timestamp}.wav"
            
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Recording for {RECORD_DURATION} seconds...")
            
            # Record audio
            if not record_audio(raw_audio, RECORD_DURATION):
                print("Recording failed, retrying...")
                time.sleep(5)
                continue
            
            # Create overlapped version with previous recording
            create_overlapped_audio(raw_audio, overlapped_audio)
            
            # Update previous audio reference for next iteration
            PREVIOUS_AUDIO = raw_audio
            
            # Check if audio is silent and log RMS value (using overlapped audio)
            is_silent, stats = is_audio_silent(overlapped_audio)
            print(f"RMS_mean={stats['mean']:.2f} RMS_peak={stats['peak']:.2f} RMS_p{RMS_PERCENTILE}={stats['perc']:.2f} (threshold {SILENCE_THRESHOLD})")
            if is_silent:
                print("Audio considered silent, skipping transcription.")
                # Update noise profile from this silent recording for adaptive noise reduction
                update_noise_profile(overlapped_audio)
                # Keep file for debugging; cleanup will prune older ones
                continue
            
            # Clean audio (use overlapped version)
            print("Cleaning audio...")
            if not clean_audio(overlapped_audio, clean_audio_path):
                print("Audio cleaning failed, skipping transcription.")
                raw_audio.unlink()
                continue
            
            # Transcribe
            print("Transcribing...")
            transcript = transcribe_audio(clean_audio_path)
            
            # Log the result with audio stats for debugging
            if transcript:
                log_transcript(transcript, stats, raw_audio.name)
            else:
                print("No transcript generated.")
            
            # Cleanup temporary files (keep last few wavs for debugging)
            cleanup_temp_files()
            
        except KeyboardInterrupt:
            print("\nStopping audio logger...")
            break
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
