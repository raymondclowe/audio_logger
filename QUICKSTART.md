# Audio Logger - Quick Start Guide

## Installation (One Command)

For a complete production installation:

```bash
git clone https://github.com/raymondclowe/audio_logger.git
cd audio_logger
bash setup.sh --production
```

That's it! The service will be running and logging audio transcripts to `/opt/audio_logger/logs/`

## Room Calibration (Recommended for Better Performance)

**For best transcription accuracy, run room calibration after installation.** This optimizes audio processing parameters for your specific room acoustics and microphone:

```bash
# Stop the service temporarily
sudo systemctl stop audio_logger

# Run guided calibration (interactive, easy to use)
cd /opt/audio_logger
python tools/guided_calibration.py

# Restart the service with optimized settings
sudo systemctl start audio_logger
```

The calibration will prompt you to read sample text aloud, then automatically find the best settings for your environment.

## What the Setup Does

1. ✓ Installs system dependencies (sox, alsa-utils)
2. ✓ Installs Python packages (requests, numpy, psutil)
3. ✓ Auto-detects your USB audio device
4. ✓ Sets microphone gain to maximum
5. ✓ Tests audio recording
6. ✓ Creates `audiologger` user and group
7. ✓ Deploys to `/opt/audio_logger`
8. ✓ Installs and starts systemd service
9. ✓ Adds you to `audiologger` group for log access

## Quick Commands

```bash
# Check if service is running
sudo systemctl status audio_logger

# View live transcript logs
sudo journalctl -u audio_logger -f

# Read today's log file (after logging out and back in)
cat /opt/audio_logger/logs/audio_log_$(date +%Y-%m-%d).txt

# Restart service
sudo systemctl restart audio_logger

# Stop service
sudo systemctl stop audio_logger
```

## Configuration

Update transcription server in `/opt/audio_logger/audio_logger.json`:

```json
{
  "transcription": {
    "url": "http://your-whisper-server:8085/transcribe",
    "model": "small"
  }
}
```

Then restart:
```bash
sudo systemctl restart audio_logger
```

## Important Notes

- **Log out and back in** after installation to access log files
- Logs are saved daily: `audio_log_YYYY-MM-DD.txt`
- Only group members can read logs (secure by default)
- Service starts automatically on boot

## Troubleshooting

**No transcripts appearing?**
- Check audio levels are high enough (speak loudly or adjust `SILENCE_THRESHOLD`)
- Verify transcription service is reachable: `curl http://your-server:8085/transcribe -X OPTIONS`

**Permission denied reading logs?**
- Log out and back in (group membership requires new session)
- Check you're in the group: `groups | grep audiologger`

**Service not starting?**
- Check logs: `sudo journalctl -u audio_logger -n 50`
- Verify audio device: `arecord -l`

## Testing Before Installation

To test without installing the service:

```bash
bash setup.sh
.venv/bin/python main.py
```

Press Ctrl+C to stop. If recording works, install with:

```bash
bash setup.sh --production
```

## Advanced Options

```bash
# Setup only (no service install)
bash setup.sh

# Install service from current directory
bash setup.sh --install-service

# Full production install to /opt
bash setup.sh --production
```
