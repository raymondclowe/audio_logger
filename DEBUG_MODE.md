# Debug Mode

All verbose diagnostic output is now gated behind a debug flag. Production deployments run silently while development gets full visibility.

## Enabling Debug Mode

Choose one of two methods:

### Method 1: Command-Line Argument (Temporary)

Run with `--debug` flag:

```bash
python main.py --debug
```

Or for systemd service (temporary):
```bash
sudo systemctl stop audio_logger
python /opt/audio_logger/main.py --debug
```

### Method 2: Debug File (Persistent)

Create a `.debug` file in the working directory:

```bash
# For local development
touch ~/.debug

# For production deployment
touch /opt/audio_logger/.debug
```

The service will automatically enable debug mode if this file exists. No service restart needed!

To disable: `rm /opt/audio_logger/.debug`

## What Gets Hidden

In production mode (no debug flag), these are suppressed:

- Service startup message
- Recording start timestamps
- RMS peak/mean/variance analysis
- Threshold calculations
- Learning progress indicators `[LEARNING]`
- Baseline updates
- Environment change warnings
- Transcription status ("Cleaning audio...", "Transcribing...")
- Audio statistics in logs
- Error messages (except critical failures)

## What Always Appears

- Transcription results (what was recorded and logged)
- Critical systemd errors (if service fails to start)

## Debug Output Example

```
Starting audio logger service...
✓ Loaded baseline from audio_baseline.json
  Silence baseline (P10): 87.5

[04:15:22] Recording for 60 seconds...
RMS peak=156.3 mean=92.1 var=0.087 [LEARNING]
  Thresholds: quiet<114 speech>219 Reason: quiet
  Learning baseline (3/10)

[04:16:22] Recording for 60 seconds...
RMS peak=2841.5 mean=612.3 var=0.391
  Thresholds: quiet<114 speech>219 Reason: speech_detected
Cleaning audio...
Transcribing...
[04:16:52] Logged: Continuing to log the audio that's being recorded...
```

## Production Mode Output (No Debug)

```
[04:16:52] Logged: Continuing to log the audio that's being recorded...
```

Much cleaner! Only the actual results, nothing else.

## Monitoring in Production

Even without debug output, check the service status:

```bash
# See last 20 log lines (only transcriptions)
sudo journalctl -u audio_logger -n 20

# Watch for transcriptions as they come in
sudo journalctl -u audio_logger -f

# Check if service is running
systemctl status audio_logger
```

## Troubleshooting

If the service isn't working and you need diagnostics:

```bash
# Enable debug mode
touch /opt/audio_logger/.debug

# Watch the logs in real-time
sudo journalctl -u audio_logger -f

# You'll now see all diagnostic output
# [Look for error messages and decision logic]

# When done, disable debug
rm /opt/audio_logger/.debug
```

## Implementation Notes

The debug flag is checked at module load time:

```python
DEBUG = "--debug" in sys.argv or Path(".debug").exists()
```

This is evaluated once when the script starts, so:
- Changing the `.debug` file requires service restart
- But adding/removing it before startup affects immediate behavior
- The `--debug` CLI arg works at any point during execution

For all wrapped output, the check is simple:
```python
if DEBUG:
    print(...)
```

Zero performance impact when disabled - the check is minimal.
