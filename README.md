# Audio Logger

Continuous audio recording and transcription service.

## Features

- Records audio in 1-minute segments
- Cleans audio using sox (noise reduction, filters)
- Detects silent audio and skips transcription
- Transcribes using Whisper API at http://192.168.0.142:8085
- Logs transcripts to daily log files with timestamps

## Setup

1. Install dependencies:
```bash
cd /home/pi/audio_logger
source .venv/bin/activate
uv pip install requests numpy
```

2. Ensure sox is installed:
```bash
sudo apt-get install sox
```

3. Test the script:
```bash
python main.py
```

4. Install as a system service:
```bash
sudo cp audio_logger.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable audio_logger
sudo systemctl start audio_logger
```

5. Check service status:
```bash
sudo systemctl status audio_logger
sudo journalctl -u audio_logger -f
```

## Configuration

Edit `main.py` to change:
- `AUDIO_DEVICE`: Audio input device (default: "plughw:3,0")
- `RECORD_DURATION`: Recording duration in seconds (default: 60)
- `TRANSCRIBE_URL`: Transcription service URL
- `TRANSCRIBE_MODEL`: Whisper model to use (default: "base")
- `SILENCE_THRESHOLD`: RMS threshold for silence detection (default: 500)

## Log Files

Daily log files are saved in the `logs/` directory:
- Format: `audio_log_YYYY-MM-DD.txt`
- Each entry includes timestamp and transcript
