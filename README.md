# Audio Logger

Continuous audio recording and transcription service.

## Features

- Records audio in 1-minute segments
- Cleans audio using sox (noise reduction, filters)
- Detects silent audio and skips transcription
- Detects VLC playback via CPU usage and pauses recording
- Transcribes using Whisper API
- Logs transcripts to daily log files with timestamps
- Adaptive noise profiling for better speech detection
- Secure file permissions (group-readable only)

## Quick Setup

### Automated Setup (Recommended)

Run the setup script to automatically detect hardware and configure everything:

```bash
# Clone the repository
git clone https://github.com/raymondclowe/audio_logger.git
cd audio_logger

# Basic setup (configure only, no service install)
bash setup.sh

# Or install as a service immediately
bash setup.sh --install-service

# Or full production install to /opt
bash setup.sh --production
```

The setup script will:
- Install required system packages (sox, alsa-utils, curl)
- Install Python dependencies (requests, numpy, psutil)
- Auto-detect your audio capture device
- Configure microphone gain to maximum
- Test audio recording and levels
- Update configuration files
- Optionally create system user and install service

### Manual Setup

If you prefer manual configuration:

1. Install dependencies:
```bash
sudo apt-get install sox libsox-fmt-all alsa-utils
```

2. Install Python packages:
```bash
# With uv (recommended)
uv sync

# Or with pip
python3 -m venv .venv
.venv/bin/pip install requests numpy psutil
```

3. Configure your audio device in `main.py`:
```python
AUDIO_DEVICE = "plughw:0,0"  # Update card number as needed
```

4. Configure transcription service in `audio_logger.json`:
```json
{
  "transcription": {
    "url": "http://your-server:8085/transcribe",
    "model": "small"
  }
}
```

5. Test manually:
```bash
.venv/bin/python main.py
```

## Production Deployment

Deploy to `/opt` with dedicated user and systemd service:

```bash
bash setup.sh --production
```

Or use the deploy script directly after setup:

```bash
bash deploy.sh
```

This will:
- Create `audiologger` user and group
- Copy application to `/opt/audio_logger`
- Set proper permissions (files are group-readable only)
- Install and start systemd service
- Add current user to `audiologger` group for log access

## Service Management

After installation:

```bash
# Check status
sudo systemctl status audio_logger

# View live logs
sudo journalctl -u audio_logger -f

# Stop/Start/Restart
sudo systemctl stop audio_logger
sudo systemctl start audio_logger
sudo systemctl restart audio_logger
```

## Configuration

Edit `main.py` to change:
- `AUDIO_DEVICE`: Audio input device (auto-detected by setup.sh)
- `RECORD_DURATION`: Recording duration in seconds (default: 60)
- `SILENCE_THRESHOLD`: RMS peak threshold for silence detection (default: 200)

Edit `audio_logger.json` to configure transcription service:
- `transcription.url`: Transcription service URL (default: http://localhost:8085/transcribe)
- `transcription.model`: Whisper model to use (default: "small")
- `VLC_CPU_THRESHOLD`: Minimum CPU% to consider VLC "playing" (default: 5.0)

## Log Files

Daily log files are saved in the `logs/` directory:
- Format: `audio_log_YYYY-MM-DD.txt`
- Each entry includes timestamp, RMS statistics, and transcript
- Files are `rw-r-----` (640) - readable by user and group only

**Production location**: `/opt/audio_logger/logs/`

**Note**: You must be a member of the `audiologger` group to read logs. Log out and back in after installation for group membership to take effect.

## Troubleshooting

### Low Audio Levels

If audio is not being detected:
1. Check microphone is not muted
2. Verify correct device: `arecord -l`
3. Test recording: `arecord -D plughw:0,0 -d 5 test.wav`
4. Increase ALSA capture volume: `amixer -c 0 cset numid=3 16`
5. Lower `SILENCE_THRESHOLD` in `main.py`

### Transcription Service Unreachable

1. Verify service is running: `curl http://your-server:8085/transcribe -X OPTIONS`
2. Check network connectivity
3. Update transcription URL in `audio_logger.json`

### Permission Denied

After installation, log out and back in to refresh group membership:
```bash
# Check if you're in the audiologger group
groups

# Should show: ... audiologger
```

## Hardware Requirements

- USB audio capture device (or built-in microphone)
- Network access to Whisper transcription service
- Sufficient disk space for audio files and logs

## Room Calibration

The audio logger includes a room calibration tool to find optimal sox audio processing parameters for your specific environment. Different rooms have different acoustic properties (reverb, ambient noise, etc.) that affect transcription accuracy.

### Running Room Calibration

```bash
# Record a speech sample and calibrate (recommended)
python tools/room_calibration.py --record --device plughw:0,0

# Use an existing audio file with known transcript
python tools/room_calibration.py --input sample.wav --reference "expected transcript text"

# Generate test audio (requires espeak) and calibrate
python tools/room_calibration.py --generate --device plughw:0,0

# Quick calibration with fewer parameter combinations
python tools/room_calibration.py --record --quick
```

### What It Does

1. Records or generates test audio
2. Tests multiple sox parameter combinations (noise reduction, filters, EQ)
3. Transcribes each variant using the Whisper API
4. Compares transcripts to find the most accurate settings
5. Outputs recommended sox parameters for your room

### Applying Calibration Results

After running calibration, update the `clean_audio()` function in `main.py` with the recommended parameters, or use the generated sox command.

## Tools

Additional tools are available in the `tools/` directory:

- `room_calibration.py` - Find optimal sox settings for your room
- `ab_compare.py` - Interactive A/B testing for sox cleaning chains
- `ab_test_sox.py` - Batch A/B testing with transcription comparison

## TODO

- [ ] **Room Calibration Enhancements**:
  - [ ] Auto-generate standardized test audio for calibration
  - [ ] Record real speech samples for more realistic calibration
  - [ ] Expand parameter search space for more fine-tuned results
  - [ ] Add support for saving/loading room profiles
  - [ ] Integrate calibration results directly into main.py configuration

## Security Notes

- All data files (audio recordings, transcripts, noise profiles) are restricted to `rw-r-----` (640) permissions
- Only the `audiologger` user can write files
- Members of the `audiologger` group can read files
- Files are NOT world-readable for privacy
