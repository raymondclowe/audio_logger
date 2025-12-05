#!/usr/bin/env python3
"""
Audio device testing harness.
Tests all available audio devices and provides detailed diagnostic information.
"""

import subprocess
import sys
import tempfile
import wave
from pathlib import Path
import numpy as np


def detect_audio_devices():
    """Detect available ALSA audio capture devices."""
    devices = []
    try:
        result = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print(f"Error running arecord -l: {result.stderr}")
            return devices
        
        # Parse output: card X: ... device Y: ...
        for line in result.stdout.splitlines():
            if line.startswith("card"):
                try:
                    parts = line.split()
                    card = parts[1].rstrip(':')
                    device_idx = parts.index("device") + 1
                    device = parts[device_idx].rstrip(':')
                    # Extract name
                    name_start = line.find('[')
                    name_end = line.find(']', name_start)
                    if name_start != -1 and name_end != -1:
                        name = line[name_start+1:name_end]
                    else:
                        name = "Unknown"
                    devices.append((card, device, name))
                except (ValueError, IndexError) as e:
                    print(f"Failed to parse line: {line} - {e}")
                    continue
    except subprocess.TimeoutExpired:
        print("Timeout while detecting audio devices")
    except Exception as e:
        print(f"Failed to detect audio devices: {e}")
    
    return devices


def test_audio_device_verbose(card: str, device: str, duration: float = 0.5):
    """Test an audio device with verbose output.
    Returns (success, peak_rms, error_message) tuple.
    """
    device_id = f"plughw:{card},{device}"
    test_file = Path(tempfile.gettempdir()) / f"audio_test_{card}_{device}.wav"
    
    try:
        # Remove any existing test file
        test_file.unlink(missing_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Testing: {device_id}")
        print(f"{'='*60}")
        
        cmd = [
            "arecord",
            "-D", device_id,
            "-d", str(int(duration)),
            "-f", "S16_LE",  # 16-bit signed little endian
            "-r", "16000",    # 16kHz sample rate (faster, sufficient for speech)
            "-c", "1",        # Mono (faster)
            "-t", "wav",
            str(test_file)
        ]
        
        print(f"Command: {' '.join(cmd)}")
        print(f"Recording for {duration} seconds...")
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            timeout=duration + 5
        )
        
        print(f"Return code: {result.returncode}")
        
        if result.stdout:
            print(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"STDERR:\n{result.stderr}")
        
        # Check if file was created
        if not test_file.exists():
            error_msg = "Recording file was not created"
            print(f"❌ FAILED: {error_msg}")
            return (False, 0.0, error_msg)
        
        file_size = test_file.stat().st_size
        print(f"File created: {test_file} ({file_size} bytes)")
        
        if file_size == 0:
            error_msg = "Recording file is empty (0 bytes)"
            print(f"❌ FAILED: {error_msg}")
            test_file.unlink(missing_ok=True)
            return (False, 0.0, error_msg)
        
        # Analyze audio content
        try:
            with wave.open(str(test_file), 'rb') as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                
                print(f"Audio properties:")
                print(f"  Channels: {channels}")
                print(f"  Sample rate: {sample_rate} Hz")
                print(f"  Frames: {frames}")
                print(f"  Sample width: {sample_width} bytes")
                print(f"  Duration: {frames/sample_rate:.2f} seconds")
                
                # Read and analyze audio data
                audio_data = wf.readframes(frames)
                audio = np.frombuffer(audio_data, dtype=np.int16)
                
                # Convert to mono if stereo
                if channels > 1:
                    audio = audio.reshape(-1, channels).mean(axis=1)
                
                # Calculate statistics
                mean_rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
                max_amplitude = float(np.max(np.abs(audio)))
                min_val = float(np.min(audio))
                max_val = float(np.max(audio))
                
                # Calculate variance/dynamics to detect real audio vs static noise
                # Split audio into 100ms windows and check variation
                window_size = int(sample_rate * 0.1)  # 100ms windows
                num_windows = len(audio) // window_size
                
                if num_windows > 0:
                    windows = audio[:num_windows * window_size].reshape(num_windows, window_size)
                    window_rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=1))
                    rms_variance = float(np.std(window_rms))
                    rms_mean = float(np.mean(window_rms))
                    coefficient_of_variation = rms_variance / rms_mean if rms_mean > 0 else 0
                else:
                    rms_variance = 0
                    coefficient_of_variation = 0
                
                print(f"Audio analysis:")
                print(f"  RMS (overall): {mean_rms:.1f}")
                print(f"  RMS (windowed mean): {rms_mean:.1f}")
                print(f"  RMS variance: {rms_variance:.1f}")
                print(f"  Coefficient of variation: {coefficient_of_variation:.3f}")
                print(f"  Max amplitude: {max_amplitude:.1f}")
                print(f"  Range: [{min_val:.0f}, {max_val:.0f}]")
                
                # Quality scoring for device selection
                # Key insight: Real ambient room noise is STEADY (low CV < 0.5)
                # Electrical/digital noise is SPIKY (high CV > 1.0)
                # A real microphone in a quiet room has consistent low-level ambient noise
                
                is_likely_real_mic = False
                reason = ""
                
                if mean_rms < 10:
                    reason = "Too quiet - no input"
                    quality_score = 0
                elif coefficient_of_variation > 1.0:
                    reason = "High variance - electrical noise/interference"
                    quality_score = -1000  # Penalize heavily
                elif mean_rms >= 50 and coefficient_of_variation < 0.5:
                    reason = "Steady signal - real microphone"
                    is_likely_real_mic = True
                    quality_score = mean_rms
                else:
                    reason = "Moderate quality"
                    quality_score = mean_rms * 0.5  # Lower priority
                
                # Cleanup
                test_file.unlink(missing_ok=True)
                
                if result.returncode == 0:
                    if is_likely_real_mic:
                        print(f"✅ REAL MICROPHONE: Steady ambient signal (RMS: {mean_rms:.1f}, CV: {coefficient_of_variation:.3f})")
                    elif quality_score < 0:
                        print(f"❌ REJECTED: {reason} (RMS: {mean_rms:.1f}, CV: {coefficient_of_variation:.3f})")
                    elif quality_score == 0:
                        print(f"⚠️  {reason} (RMS: {mean_rms:.1f}, CV: {coefficient_of_variation:.3f})")
                    else:
                        print(f"⚠️  {reason} (RMS: {mean_rms:.1f}, CV: {coefficient_of_variation:.3f})")
                    
                    return (True, quality_score, reason if quality_score <= 0 else None)
                else:
                    error_msg = f"arecord returned non-zero exit code: {result.returncode}"
                    print(f"❌ FAILED: {error_msg}")
                    return (True, -1000, error_msg)
                    
        except Exception as e:
            error_msg = f"Failed to analyze audio file: {e}"
            print(f"❌ FAILED: {error_msg}")
            test_file.unlink(missing_ok=True)
            return (False, 0.0, error_msg)
    
    except subprocess.TimeoutExpired:
        error_msg = "Recording timed out"
        print(f"❌ FAILED: {error_msg}")
        test_file.unlink(missing_ok=True)
        return (False, 0.0, error_msg)
    except Exception as e:
        error_msg = f"Unexpected error: {e}"
        print(f"❌ FAILED: {error_msg}")
        test_file.unlink(missing_ok=True)
        return (False, 0.0, error_msg)


def main():
    """Test all available audio devices."""
    print("Audio Device Testing Harness")
    print("="*60)
    
    # Detect devices
    devices = detect_audio_devices()
    
    if not devices:
        print("❌ No audio devices found!")
        sys.exit(1)
    
    print(f"\nFound {len(devices)} audio device(s):")
    for i, (card, device, name) in enumerate(devices, 1):
        print(f"  {i}. plughw:{card},{device} - {name}")
    
    # Apply name-based heuristics to skip known bad devices
    # These can be overridden by setting AUDIO_DEVICE environment variable
    filtered_devices = []
    skipped = []
    
    for card, device, name in devices:
        name_lower = name.lower()
        skip = False
        reason = ""
        
        # Skip known internal/chipset audio that isn't real microphone input
        if "sof-hda-dsp" in name_lower:
            skip = True
            reason = "Intel Smart Sound chipset (not external mic)"
        elif "hdmi" in name_lower:
            skip = True
            reason = "HDMI audio output"
        elif "loopback" in name_lower:
            skip = True
            reason = "Virtual loopback device"
        
        if skip:
            skipped.append((card, device, name, reason))
        else:
            filtered_devices.append((card, device, name))
    
    if skipped:
        print(f"\n⏭️  Skipped {len(skipped)} device(s) by heuristic:")
        for card, device, name, reason in skipped:
            print(f"  • plughw:{card},{device} ({name}) - {reason}")
    
    if not filtered_devices:
        print(f"\n⚠️  All devices skipped by heuristics. Testing all devices anyway...")
        filtered_devices = devices
    
    # Test each remaining device
    results = []
    for card, device, name in filtered_devices:
        success, rms, error = test_audio_device_verbose(card, device, duration=1.0)
        results.append({
            'card': card,
            'device': device,
            'name': name,
            'success': success,
            'rms': rms,
            'error': error
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    working_devices = [r for r in results if r['success']]
    failed_devices = [r for r in results if not r['success']]
    
    if working_devices:
        print(f"\n✅ Working devices ({len(working_devices)}):")
        real_mics = []
        rejected = []
        
        for r in working_devices:
            device_id = f"plughw:{r['card']},{r['device']}"
            quality_str = f"Quality: {r['rms']:.1f}"
            
            if r['rms'] > 0:
                real_mics.append(r)
                icon = "🎤"
            elif r['rms'] == 0:
                icon = "🔇"
            else:
                rejected.append(r)
                icon = "❌"
            
            print(f"  {icon} {device_id} ({r['name']}) - {quality_str}")
            if r['error']:
                print(f"      Reason: {r['error']}")
        
        # Recommend best device
        if real_mics:
            # Prefer USB devices first, then highest quality score
            usb_real_mics = [r for r in real_mics if "USB" in r['name'].upper()]
            if usb_real_mics:
                best = max(usb_real_mics, key=lambda x: x['rms'])
            else:
                best = max(real_mics, key=lambda x: x['rms'])
            
            best_id = f"plughw:{best['card']},{best['device']}"
            print(f"\n🎯 RECOMMENDED: {best_id} ({best['name']}) - Quality: {best['rms']:.1f}")
        else:
            print(f"\n❌ No suitable microphone found!")
            if rejected:
                print(f"   All devices rejected due to electrical noise patterns")
    
    if failed_devices:
        print(f"\n❌ Failed devices ({len(failed_devices)}):")
        for r in failed_devices:
            device_id = f"plughw:{r['card']},{r['device']}"
            print(f"  • {device_id} ({r['name']})")
            if r['error']:
                print(f"    Error: {r['error']}")
    
    if not working_devices:
        print("\n❌ No working audio devices found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
