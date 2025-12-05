import http.server
import socketserver
# Health check globals
LAST_LOG_TIME = None
LAST_ERROR = None
#!/usr/bin/env python3
"""
Continuous audio logger with transcription.
Records 1-minute segments, cleans audio, transcribes, and logs to daily files.
"""

import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import psutil
import wave
import threading
import queue

# Debug flag - check CLI argument and .debug file in working directory
DEBUG = "--debug" in sys.argv or Path(".debug").exists()

# Configuration
AUDIO_DEVICE = None  # Will be auto-detected at startup
RECORD_DURATION = 60  # seconds
OVERLAP_DURATION = 2  # seconds - overlap between chunks to avoid word breaks
TRANSCRIBE_URL = "http://192.168.0.142:8085/transcribe"
TRANSCRIBE_MODEL = "small"
LOG_DIR = Path(__file__).parent / "logs"
TEMP_DIR = Path(__file__).parent / "temp"
NOISE_PROFILE = Path(__file__).parent / "temp" / "ambient_noise.prof"
BASELINE_FILE = Path(__file__).parent / "temp" / "audio_baseline.json"

# Audio settings
SAMPLE_RATE = 44100
CHANNELS = 2
# Adaptive thresholds - will be learned from silent periods
SILENCE_BASELINE_SAMPLES = 10  # Collect this many silent recordings before stabilizing
BASELINE_MARGIN = 1.3  # 30% margin above silence baseline for "definitely quiet" (lowered for testing)
SPEECH_MULTIPLIER = 2.0  # Speech must be this many times above silence baseline (lowered for testing)
SPEECH_VARIANCE_THRESHOLD = 0.25  # Coefficient of variation for speech pattern detection (lowered for testing)
ENVIRONMENT_UPDATE_THRESHOLD = 0.5  # If no speech found this many times, re-baseline
ABSOLUTE_SPEECH_MINIMUM = 80  # Conservative absolute minimum for speech (lowered for portability)
# Spike filtering - reject audio with isolated loud peaks (clicks, pops)
# Relaxed to reduce false negatives on real speech with brief transients
PEAK_TO_MEAN_MAX_RATIO = 15.0  # If peak/mean >15 AND peak/P90 > threshold, treat as spike
PEAK_TO_P90_MAX_RATIO = 10.0   # If peak/P90 >10 AND peak/mean > threshold, treat as spike
# Windowed RMS config: speech is intermittent; use P90 of window RMS for decisions (more stable than peak)
RMS_WINDOW_SECS = 0.05  # 50ms window
RMS_PERCENTILE = 90     # percentile of window RMS (use for decisions, more stable than peak)
VLC_CPU_THRESHOLD = 5.0  # percent; if VLC above this, assume playing
VLC_CHECK_INTERVAL = 0.5  # seconds for cpu sampling

# Context tracking for transcription continuity
LAST_TRANSCRIPT = None
CONTEXT_WORDS = 30  # Number of words to include from previous transcript
PREVIOUS_AUDIO = None  # Store path to previous recording for overlap


class AudioBaseline:
    """Adaptive baseline threshold manager.
    
    Learns from silent periods to establish dynamic thresholds that adapt to
    environment (room, microphone, ambient noise level).
    """
    
    def __init__(self, baseline_file: Path):
        self.baseline_file = baseline_file
        self.silent_peaks = []  # RMS peaks from confirmed silent periods
        self.no_speech_count = 0  # Consecutive segments with no speech pattern
        self.is_learning = True  # In learning mode until we have enough samples
        self.load()
    
    def load(self):
        """Load baseline from file if it exists."""
        try:
            if self.baseline_file.exists():
                import json
                with open(self.baseline_file, 'r') as f:
                    data = json.load(f)
                    self.silent_peaks = data.get('silent_peaks', [])
                    self.is_learning = len(self.silent_peaks) < SILENCE_BASELINE_SAMPLES
                    if not self.is_learning and DEBUG:
                        print(f"✓ Loaded baseline from {self.baseline_file.name}")
                        print(f"  Silence baseline (P10): {self.get_silence_baseline():.1f}")
        except Exception as e:
            if DEBUG:
                print(f"⚠ Could not load baseline: {e}")
    
    def save(self):
        """Save baseline to file."""
        try:
            import json
            TEMP_DIR.mkdir(exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump({'silent_peaks': self.silent_peaks}, f)
        except Exception as e:
            if DEBUG:
                print(f"⚠ Could not save baseline: {e}")
    
    def add_silent_sample(self, p90_rms: float):
        """Record a P90 RMS from a confirmed silent period."""
        self.silent_peaks.append(p90_rms)
        # Keep only recent samples (last 20) to adapt to environment changes
        if len(self.silent_peaks) > 20:
            self.silent_peaks = self.silent_peaks[-20:]
        self.is_learning = len(self.silent_peaks) < SILENCE_BASELINE_SAMPLES
        self.no_speech_count = 0  # Reset counter
        self.save()
        
        if DEBUG:
            if self.is_learning:
                print(f"  Learning baseline ({len(self.silent_peaks)}/{SILENCE_BASELINE_SAMPLES})")
            else:
                print(f"  Baseline updated: {self.get_silence_baseline():.1f}")
    
    def record_no_speech(self):
        """Record that we detected high RMS but no speech pattern."""
        self.no_speech_count += 1
        if self.no_speech_count >= 3 and DEBUG:
            print(f"⚠ Background noise increased ({self.no_speech_count}x no-speech detections)")
            print(f"  Consider environment change (new AC, window open, etc.)")
    
    def reset_no_speech_counter(self):
        """Speech was detected, reset counter."""
        self.no_speech_count = 0
    
    def get_silence_baseline(self) -> float:
        """Get baseline threshold (P10 of silent peaks)."""
        if not self.silent_peaks:
            return 150  # Conservative default if no data yet
        return float(np.percentile(self.silent_peaks, 10))
    
    def get_speech_minimum(self) -> float:
        """Get minimum RMS needed to consider as potential speech."""
        baseline = self.get_silence_baseline()
        return baseline * SPEECH_MULTIPLIER
    
    def get_definitely_quiet_threshold(self) -> float:
        """Get threshold below which audio is definitely quiet."""
        baseline = self.get_silence_baseline()
        return baseline * BASELINE_MARGIN


# Initialize global baseline manager
BASELINE = AudioBaseline(BASELINE_FILE)


def setup_directories():
    """Create necessary directories."""
    LOG_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)


def set_file_permissions(file_path: Path):
    """Set file permissions to 0640 (rw-r-----) for security."""
    try:
        file_path.chmod(0o640)
    except Exception:
        pass  # Continue if permission setting fails


def detect_audio_devices():
    """Detect available audio capture devices.
    Returns list of tuples: (card_num, device_num, device_name)
    """
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        devices = []
        for line in result.stdout.split('\n'):
            # Parse lines like: "card 3: Device [USB Audio Device], device 0: USB Audio [USB Audio]"
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
        
        return devices
    except Exception as e:
        if DEBUG:
            print(f"Failed to detect audio devices: {e}")
        return []


def test_audio_device(card: str, device: str, duration: float = 1.0):
    """Test if an audio device is active by recording a short sample.
    Returns (success, peak_rms) tuple.
    """
    try:
        test_file = TEMP_DIR / f"test_card{card}_dev{device}.wav"
        cmd = [
            "arecord",
            "-D", f"plughw:{card},{device}",
            "-d", str(duration),
            "-f", "cd",
            "-t", "wav",
            str(test_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 2)
        
        if result.returncode == 0 and test_file.exists():
            # Check RMS level
            try:
                with wave.open(str(test_file), 'rb') as wf:
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16)
                    if wf.getnchannels() > 1:
                        audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1)
                    peak_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                    test_file.unlink(missing_ok=True)
                    return (True, peak_rms)
            except Exception:
                test_file.unlink(missing_ok=True)
                return (False, 0.0)
        
        test_file.unlink(missing_ok=True)
        return (False, 0.0)
        
    except Exception as e:
        if DEBUG:
            print(f"Failed to test device card{card},device{device}: {e}")
        return (False, 0.0)


def auto_detect_audio_device():
    """Auto-detect the best audio capture device.
    
    Strategy:
    1. If only one device found, use it
    2. If multiple devices, test each for 1 second and pick the one with highest RMS
    3. Fall back to plughw:0,0 if detection fails
    """
    global AUDIO_DEVICE
    
    devices = detect_audio_devices()
    
    if not devices:
        if DEBUG:
            print("⚠ No audio devices detected, using default plughw:0,0")
        AUDIO_DEVICE = "plughw:0,0"
        return
    
    if len(devices) == 1:
        card, device, name = devices[0]
        AUDIO_DEVICE = f"plughw:{card},{device}"
        print(f"✓ Auto-detected single audio device: {AUDIO_DEVICE} ({name})")
        return
    
    # Multiple devices - test each and pick the most active one
    print(f"Found {len(devices)} audio devices, testing...")
    best_device = None
    best_rms = 0.0
    
    for card, device, name in devices:
        device_id = f"plughw:{card},{device}"
        if DEBUG:
            print(f"  Testing {device_id} ({name})...")
        success, rms = test_audio_device(card, device, duration=1.0)
        if success:
            if DEBUG:
                print(f"    RMS: {rms:.1f}")
            if rms > best_rms:
                best_rms = rms
                best_device = (device_id, name)
    
    if best_device:
        AUDIO_DEVICE = best_device[0]
        print(f"✓ Selected most active device: {AUDIO_DEVICE} ({best_device[1]}) with RMS {best_rms:.1f}")
    else:
        # All tests failed, use first device
        card, device, name = devices[0]
        AUDIO_DEVICE = f"plughw:{card},{device}"
        print(f"⚠ All device tests failed, using first device: {AUDIO_DEVICE} ({name})")


def get_daily_log_path():
    """Get the path for today's log file."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"audio_log_{date_str}.txt"


def ensure_daily_log_file():
    """Ensure today's log file exists so we roll over correctly at midnight."""
    log_path = get_daily_log_path()
    if not log_path.exists():
        log_path.touch()
        set_file_permissions(log_path)


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
            set_file_permissions(output_path)
            
            # Cleanup temp overlap file
            overlap_temp.unlink(missing_ok=True)
            return True
        else:
            # No overlap available, just copy current
            import shutil
            shutil.copy(current_path, output_path)
            set_file_permissions(output_path)
            return True
            
    except Exception as e:
        if DEBUG:
            print(f"Failed to create overlap, using current audio only: {e}")
        import shutil
        shutil.copy(current_path, output_path)
        set_file_permissions(output_path)
        return True


def record_audio(output_path: Path, duration: int) -> bool:
    """Record audio to a WAV file with gain boost."""
    try:
        # Set ALSA capture volume to maximum for the device
        # Extract card number from AUDIO_DEVICE (e.g., "plughw:3,0" -> "3")
        try:
            if AUDIO_DEVICE and ':' in AUDIO_DEVICE:
                card_num = AUDIO_DEVICE.split(':')[1].split(',')[0]
                subprocess.run(
                    ["amixer", "-c", card_num, "set", "Capture", "100%"],
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
        set_file_permissions(output_path)
        return True
    except subprocess.CalledProcessError as e:
        if DEBUG:
            print(f"Recording failed: {e}")
        return False


def is_audio_silent(wav_path: Path) -> tuple[bool, dict]:
    """Multi-algorithm speech detection with voting.
    
    Uses 4 independent algorithms that each vote on whether audio contains speech:
    1. Energy-based: P90 RMS vs adaptive baseline threshold
    2. Spike filter: Rejects isolated transient peaks (peak/mean ratio)
    3. Acoustic pattern: High variance indicates natural speech vs flat noise
    4. Spectral richness: Multiple frequency components vs single tone
    
    Decision: Speech requires 2+ votes. This is much more robust than any single algorithm.
    
    Returns (is_silent, stats) dict with votes and analysis."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            frames = wf.readframes(n_frames)

        dtype = np.int16
        audio = np.frombuffer(frames, dtype=dtype)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1).astype(np.int16)

        if audio.size == 0:
            return True, {
                "mean": 0.0, "peak": 0.0, "perc": 0.0, "variance": 0.0,
                "is_speech": False, "reason": "empty", "votes": {"energy": False, "spike_filter": True, "pattern": False, "spectral": False}
            }

        # Windowed RMS analysis
        win_samples = max(1, int(framerate * RMS_WINDOW_SECS))
        pad = (-audio.size) % win_samples
        if pad:
            audio_padded = np.pad(audio, (0, pad), mode='constant')
        else:
            audio_padded = audio
        windows = audio_padded.reshape(-1, win_samples).astype(np.float64)
        rms_windows = np.sqrt(np.mean(windows ** 2, axis=1))

        mean_rms = float(np.mean(rms_windows))
        peak_rms = float(np.max(rms_windows))
        perc_rms = float(np.percentile(rms_windows, RMS_PERCENTILE))
        std_rms = float(np.std(rms_windows))
        cv = std_rms / mean_rms if mean_rms > 0 else 0.0
        
        # Get adaptive thresholds based on learned silence baseline
        silence_baseline = BASELINE.get_silence_baseline()
        definitely_quiet = BASELINE.get_definitely_quiet_threshold()
        speech_minimum = max(BASELINE.get_speech_minimum(), ABSOLUTE_SPEECH_MINIMUM)
        
        # ===== VOTE 1: Energy-based detection =====
        # Speech should have P90 RMS or peak well above baseline
        vote_energy = (
            (perc_rms >= speech_minimum and perc_rms >= definitely_quiet) or
            (peak_rms >= speech_minimum * 1.2)
        )
        
        # ===== VOTE 2: Spike filter (vote AGAINST speech if transient) =====
        # Reject only if BOTH ratios are extremely high (isolated click/pop)
        peak_to_mean_ratio = peak_rms / mean_rms if mean_rms > 0 else 1.0
        peak_to_p90_ratio = peak_rms / perc_rms if perc_rms > 0 else 1.0
        # More lenient: require both ratios to exceed limits to call it a spike
        vote_spike_filter = not (peak_to_mean_ratio > PEAK_TO_MEAN_MAX_RATIO and peak_to_p90_ratio > PEAK_TO_P90_MAX_RATIO)
        
        # ===== VOTE 3: Acoustic pattern (variance/CV) =====
        # Speech has natural variation in amplitude (peaks and valleys)
        vote_pattern = cv > SPEECH_VARIANCE_THRESHOLD
        
        # ===== VOTE 4: Spectral richness =====
        # Speech has energy across multiple frequency bands, not single tone
        # Analyze high frequencies vs low frequencies
        try:
            # Simple frequency-domain check: compute RMS in speech frequency ranges
            # Low frequencies (200-800 Hz), mid (800-2000 Hz), high (2000-3400 Hz)
            n_fft = min(2048, len(audio_padded))
            # Normalize audio
            audio_norm = audio_padded.astype(np.float64) / (np.max(np.abs(audio_padded)) + 1e-10)
            
            # Compute spectrum (simple approach: energy in different bands)
            freq_bins = np.abs(np.fft.rfft(audio_norm, n=n_fft))
            freq_bins_smooth = np.convolve(freq_bins, np.ones(5)/5, mode='same')  # Smooth to reduce noise
            
            # Define band indices (very approximate based on sampling rate)
            bin_per_hz = n_fft / (2 * framerate)
            low_band = freq_bins_smooth[int(200 * bin_per_hz):int(800 * bin_per_hz)]
            mid_band = freq_bins_smooth[int(800 * bin_per_hz):int(2000 * bin_per_hz)]
            high_band = freq_bins_smooth[int(2000 * bin_per_hz):int(3400 * bin_per_hz)]
            
            # Speech should have energy across bands (not dominated by one)
            if len(low_band) > 0 and len(mid_band) > 0 and len(high_band) > 0:
                total_energy = np.sum(low_band) + np.sum(mid_band) + np.sum(high_band)
                # If any single band has > 70% of energy, it's likely not speech (single tone or noise)
                max_band_ratio = max(np.sum(low_band), np.sum(mid_band), np.sum(high_band)) / (total_energy + 1e-10)
                vote_spectral = max_band_ratio < 0.7  # Speech spreads energy across bands
            else:
                vote_spectral = False  # Not enough data
        except Exception:
            vote_spectral = False  # If spectral analysis fails, don't vote
        
        # ===== VOTING DECISION =====
        votes = {
            "energy": vote_energy,
            "spike_filter": vote_spike_filter,
            "pattern": vote_pattern,
            "spectral": vote_spectral
        }
        
        # Count votes FOR speech (all must pass spike filter, then need 2+ other votes)
        speech_votes = sum([vote_energy, vote_pattern, vote_spectral])
        
        # Speech requires:
        # 1. Pass the spike filter (not a transient)
        # 2. At least 2 votes from energy/pattern/spectral
        is_likely_speech = vote_spike_filter and speech_votes >= 2
        
        # Determine reason and update baseline
        if perc_rms < definitely_quiet:
            reason = "quiet"
            is_silent = True
            is_speech = False
            BASELINE.add_silent_sample(perc_rms)
        elif not vote_spike_filter:
            reason = f"transient_spike (peak_ratio={peak_to_mean_ratio:.1f}x)"
            is_silent = True
            is_speech = False
            BASELINE.record_no_speech()
        elif is_likely_speech:
            reason = f"speech_detected ({speech_votes}_votes)"
            is_silent = False
            is_speech = True
            BASELINE.reset_no_speech_counter()
        else:
            reason = f"rejected ({speech_votes}_votes)"
            is_silent = True
            is_speech = False
            BASELINE.record_no_speech()
        
        stats = {
            "mean": mean_rms,
            "peak": peak_rms,
            "perc": perc_rms,
            "variance": cv,
            "is_speech": is_speech,
            "reason": reason,
            "votes": votes,
            "vote_count": speech_votes,
            "peak_ratio": peak_to_mean_ratio,
            "baselines": {
                "silence": silence_baseline,
                "quiet": definitely_quiet,
                "speech_min": speech_minimum
            },
            "learning": BASELINE.is_learning
        }
        if DEBUG:
            print(f"[is_audio_silent] mean={mean_rms:.1f} peak={peak_rms:.1f} perc={perc_rms:.1f} var={cv:.3f}")
            print(f"  Baselines: silence<{silence_baseline:.1f} quiet<{definitely_quiet:.1f} speech>{speech_minimum:.1f}")
            print(f"  Votes: energy={votes['energy']} spike_filter={votes['spike_filter']} pattern={votes['pattern']} spectral={votes['spectral']}")
            print(f"  Decision: {reason} (is_speech={is_speech})")
        # .force_transcribe override
        if Path('.force_transcribe').exists():
            if DEBUG:
                print("[FORCE TRANSCRIBE OVERRIDE] Forcing is_silent=False for this segment.")
            return False, stats
        return is_silent, stats
    except Exception as e:
        if DEBUG:
            print(f"Error checking silence: {e}")
        return False, {
            "mean": 0.0, "peak": 0.0, "perc": 0.0, "variance": 0.0,
            "is_speech": False, "reason": "error", "votes": {"energy": False, "spike_filter": True, "pattern": False, "spectral": False}
        }



def update_noise_profile(audio_path: Path) -> bool:
    """Update the ambient noise profile from a silent recording."""
    try:
        # Use entire silent recording to build comprehensive noise profile
        cmd = [
            "sox", str(audio_path), "-n",
            "noiseprof", str(NOISE_PROFILE)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if DEBUG:
            print(f"Updated noise profile from {audio_path.name}")
        set_file_permissions(NOISE_PROFILE)
        return True
    except subprocess.CalledProcessError as e:
        if DEBUG:
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
        set_file_permissions(noise_prof)
        
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
        set_file_permissions(output_path)
        
        # Clean up temporary noise profile if created
        if cleanup_prof:
            noise_prof.unlink(missing_ok=True)
        return True
        
    except subprocess.CalledProcessError as e:
        if DEBUG:
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
        if DEBUG:
            print(f"[transcribe_audio] Attempting transcription for {audio_path}")
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
            if DEBUG:
                print(f"[transcribe_audio] Transcript: '{transcript}'")
            # Update context for next transcription
            if transcript:
                LAST_TRANSCRIPT = transcript
            return transcript
    except Exception as e:
        if DEBUG:
            print(f"[transcribe_audio] Transcription failed: {e}")
        return ""


def log_transcript(text: str, stats: dict = None, filename: str = None):
    """Append transcript with timestamp and audio stats to daily log file."""
    global LAST_LOG_TIME, LAST_ERROR
    if not text:
        if DEBUG:
            print("[log_transcript] No text to log.")
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
        set_file_permissions(log_path)
        LAST_LOG_TIME = timestamp
        if DEBUG:
            print(f"[log_transcript] Logged: {text}")
    except Exception as e:
        LAST_ERROR = str(e)
        if DEBUG:
            print(f"[log_transcript] Failed to write log: {e}")


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
        if DEBUG:
            print(f"Cleanup failed: {e}")


def process_audio_worker(task_queue):
    """Background worker thread that processes audio files without blocking recording."""
    global PREVIOUS_AUDIO
    
    while True:
        try:
            task = task_queue.get()
            if task is None:  # Shutdown signal
                break
                
            raw_audio, overlapped_audio, clean_audio_path, timestamp = task
            
            # Check if audio is silent and log RMS value (using overlapped audio)
            is_silent, stats = is_audio_silent(overlapped_audio)
            
            # Format detailed diagnostic output (debug only)
            if DEBUG:
                baselines = stats.get('baselines', {})
                learning_status = " [LEARNING]" if stats.get('learning') else ""
                print(f"[{timestamp}] RMS peak={stats['peak']:.1f} mean={stats['mean']:.1f} var={stats['variance']:.3f}{learning_status}")
                print(f"  Thresholds: quiet<{baselines.get('quiet', 0):.0f} speech>{baselines.get('speech_min', 0):.0f} Reason: {stats['reason']}")
            
            if is_silent:
                update_noise_profile(overlapped_audio)
                # Keep file for debugging; cleanup will prune older ones
                task_queue.task_done()
                continue
            
            # Clean audio (use overlapped version)
            if DEBUG:
                print(f"[{timestamp}] Cleaning audio...")
            if not clean_audio(overlapped_audio, clean_audio_path):
                if DEBUG:
                    print(f"[{timestamp}] Audio cleaning failed, skipping transcription.")
                try:
                    raw_audio.unlink()
                except Exception:
                    pass
                task_queue.task_done()
                continue
            
            # Transcribe
            if DEBUG:
                print(f"[{timestamp}] Transcribing...")
            transcript = transcribe_audio(clean_audio_path)
            
            # Log the result
            if transcript:
                log_transcript(transcript, stats, raw_audio.name)
            elif DEBUG:
                print(f"[{timestamp}] No transcript generated.")
            
            # Cleanup temporary files (keep last few wavs for debugging)
            cleanup_temp_files()
            
            task_queue.task_done()
            
        except Exception as e:
            global LAST_ERROR
            LAST_ERROR = str(e)
            if DEBUG:
                print(f"Error in processing worker: {e}")
            task_queue.task_done()


def main():
    # Start health check HTTP server in background
    def health_check_server():
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                global LAST_LOG_TIME, LAST_ERROR
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                import json
                resp = {
                    'status': 'running',
                    'last_log_time': LAST_LOG_TIME,
                    'last_error': LAST_ERROR
                }
                self.wfile.write(json.dumps(resp).encode())
            def log_message(self, format, *args):
                return  # Suppress default logging
        with socketserver.TCPServer(("0.0.0.0", 8089), Handler) as httpd:
            httpd.serve_forever()
    threading.Thread(target=health_check_server, daemon=True).start()

    """Main loop: continuous recording with background processing."""
    global PREVIOUS_AUDIO
    if DEBUG:
        print("Starting audio logger service...")
    setup_directories()

    # Auto-detect audio device if not configured
    if AUDIO_DEVICE is None:
        auto_detect_audio_device()
    else:
        print(f"Using configured audio device: {AUDIO_DEVICE}")

    # Start background processing thread
    processing_queue = queue.Queue(maxsize=5)  # Limit queue to prevent memory buildup
    worker_thread = threading.Thread(target=process_audio_worker, args=(processing_queue,), daemon=True)
    worker_thread.start()

    while True:
        try:
            # Ensure a fresh daily log file exists, even if we're silent
            ensure_daily_log_file()
            # Pause logging if VLC is actively playing
            if is_vlc_playing():
                if DEBUG:
                    print("VLC activity detected; pausing recording for this cycle.")
                time.sleep(5)
                continue

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            raw_audio = TEMP_DIR / f"raw_{timestamp}.wav"
            overlapped_audio = TEMP_DIR / f"overlapped_{timestamp}.wav"
            clean_audio_path = TEMP_DIR / f"clean_{timestamp}.wav"

            if DEBUG:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Recording for {RECORD_DURATION} seconds...")

            # Record audio (BLOCKS for 60 seconds - this is unavoidable)
            if not record_audio(raw_audio, RECORD_DURATION):
                if DEBUG:
                    print("Recording failed, retrying...")
                time.sleep(5)
                continue

            # Create overlapped version with previous recording (fast, <1s)
            create_overlapped_audio(raw_audio, overlapped_audio)

            # Update previous audio reference for next iteration
            PREVIOUS_AUDIO = raw_audio

            # Submit to background worker for processing
            # This is non-blocking! Recording will continue immediately
            try:
                processing_queue.put((raw_audio, overlapped_audio, clean_audio_path, timestamp), block=False)
                if DEBUG:
                    print(f"[{timestamp}] Queued for processing (queue size: {processing_queue.qsize()})")
            except queue.Full:
                if DEBUG:
                    print(f"[{timestamp}] Processing queue full, skipping this segment")

        except KeyboardInterrupt:
            if DEBUG:
                print("\nStopping audio logger...")
            processing_queue.put(None)  # Signal worker to shut down
            worker_thread.join(timeout=5)
            break
        except Exception as e:
            if DEBUG:
                print(f"Error in main loop: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()
