# Platform Expansion: Windows, Android, and Mobile Web

## Goal
Record continuous voice notes throughout the day from multiple devices (desktop/laptop, phone) into a unified searchable transcript log.

## Current System Architecture
- **Recording**: ALSA (`arecord`) for Linux audio capture
- **Processing**: SoX for audio cleanup and filtering
- **Transcription**: External HTTP API (Whisper at `192.168.0.142:8085`)
- **Storage**: Local daily text log files
- **Dependencies**: Linux-specific audio stack, systemd service

---

## Option 1: Windows Native Port

### Challenges
1. **Audio Capture**: Replace `arecord` (ALSA)
   - Windows alternatives: `ffmpeg`, `sounddevice` Python library, or Windows Audio Session API
   - Need device enumeration (like `Get-WmiObject Win32_SoundDevice`)
   
2. **Audio Processing**: Replace SoX
   - SoX **does** have Windows builds available
   - Or use `pydub` + `ffmpeg` for Python-native solution
   - Or use `scipy.signal` for Python-only DSP (no external deps)

3. **Service Management**: Replace systemd
   - Windows Service using `pywin32` or NSSM (Non-Sucking Service Manager)
   - Or Task Scheduler for simpler deployment
   - Or just run as startup application

4. **File Paths**: Adapt Path handling
   - Already using `pathlib.Path` which is cross-platform ✓
   - Need to handle Windows AppData for logs in production

5. **Process Detection**: VLC detection via `psutil`
   - `psutil` works on Windows ✓ (already cross-platform)

### Implementation Effort: **MEDIUM**
- Core logic is Python and portable
- Main work is audio I/O layer abstraction
- Could create platform-specific backends

### Recommended Approach for Windows
```python
# Abstract audio interface
class AudioBackend:
    def list_devices(self): pass
    def record(self, device, duration, output_path): pass
    def set_gain(self, device, level): pass

class LinuxAudioBackend(AudioBackend):
    # Current arecord/amixer implementation
    
class WindowsAudioBackend(AudioBackend):
    # Use sounddevice + soundfile or ffmpeg
```

**Pros:**
- Full control over recording
- Works offline
- Can use same transcription server

**Cons:**
- Requires porting/testing effort
- Sox on Windows less tested
- Need separate installers per platform

---

## Option 2: Android Native App

### Challenges
1. **Language**: Would need to rewrite in Kotlin/Java or use Python port (Kivy/BeeWare)
2. **Background Recording**: Android restricts background audio access
   - Need Foreground Service with persistent notification
   - Battery optimization issues
3. **Audio Processing**: Limited SoX support on Android
   - Would need to use Android AudioRecord API
   - Could skip local processing, send raw audio to server
4. **Network**: Phone on cellular vs WiFi to local server
   - Need transcription server accessible from internet (security concern)
   - Or only work on local WiFi
5. **Storage**: Android scoped storage restrictions
6. **Battery Life**: Continuous recording drains battery fast

### Implementation Effort: **HIGH**
- Essentially a ground-up rewrite
- Different platform APIs and constraints
- App store submission complexity

**Pros:**
- True mobile recording throughout the day
- Can use phone's always-with-you nature

**Cons:**
- Battery drain significant
- Complex background service management
- Need mobile-optimized transcription or server access
- Harder to maintain sync with desktop version

---

## Option 3: Progressive Web App (PWA) - **RECOMMENDED**

### Architecture
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Phone     │         │   Server     │         │   Desktop   │
│  (Browser)  │────────▶│   (Central)  │◀────────│   (Linux)   │
│             │  HTTPS  │              │  Local  │             │
└─────────────┘         └──────────────┘         └─────────────┘
     Web Audio API         • Receives audio          arecord
     MediaRecorder          • Transcribes
     IndexedDB cache        • Stores logs
                           • Web UI for viewing
```

### Components

#### 1. Central Server (New)
- **Web Interface**: Upload recordings from any device
- **Unified Log Storage**: All devices → one timeline
- **Authentication**: Simple token or login
- **REST API**:
  - `POST /upload` - Accept audio chunks from mobile
  - `POST /transcribe` - Already exists (Whisper)
  - `GET /logs` - View/search transcripts
  - `GET /logs/today` - Today's timeline

#### 2. Desktop Client (Modified)
- Keep current daemon running
- Add: POST audio to central server after transcription
- Or: POST cleaned audio + let server transcribe
- Fallback: Still log locally if server down

#### 3. Mobile Web App
- **No app store needed**
- **Web Audio API**: `navigator.mediaDevices.getUserMedia()`
- **MediaRecorder**: Capture audio in browser
- **Chunked Upload**: Record 30-60s, upload to server
- **Offline Queue**: Store in IndexedDB if offline, sync when online
- **Add to Home Screen**: iOS/Android both support PWAs
- **Background**: Limited, but can work with screen off on Android

### Implementation Effort: **MEDIUM-LOW**
- Reuse existing transcription service
- Web tech is well-documented
- No app store approval
- Can start simple and iterate

### PWA Pros
✅ **Cross-platform**: Works on Android, iOS, desktop  
✅ **No app stores**: Just visit URL, add to home screen  
✅ **Easy updates**: Server-side updates instantly available  
✅ **Familiar tech**: HTML/JS/Python backend  
✅ **Gradual adoption**: Can start with just web UI for viewing logs  
✅ **Unified logs**: All devices feed same timeline  
✅ **Searchable**: Can add full-text search across all transcripts  

### PWA Cons
❌ **Background recording limited**: iOS especially restrictive  
❌ **Battery**: Keeping browser tab active drains battery  
❌ **Network required**: Need server accessible from phone  
❌ **No true always-on**: User must manually start recording  

---

## Option 4: Hybrid Approach (PWA + Desktop Sync)

### Best of Both Worlds

1. **Desktop/Laptop**: Native daemon (current system)
   - Continuous recording while at desk
   - Posts to central server

2. **Mobile**: PWA for ad-hoc voice notes
   - User initiates recording when needed
   - "Record voice note" button
   - Like voice memo but auto-transcribed
   - Upload to central server

3. **Central Server**: Unified log timeline
   - Merge all sources chronologically
   - Web UI to view/search all transcripts
   - Tag by source (desktop/phone)

### User Workflow
- **At desk**: Automatic recording from USB mic
- **Away from desk**: Pull out phone, open PWA, tap "Record", speak thoughts, stop
- **Later**: Open web UI, see chronological transcript from both sources
- **Search**: "What did I say about the database schema yesterday?"

---

## Recommended Implementation Plan

### Phase 1: Central Server + Web UI (2-3 days)
```python
# Simple Flask/FastAPI server
@app.post("/api/upload")
async def upload_audio(file: UploadFile, device_id: str):
    # Save audio
    # Transcribe via existing Whisper
    # Store in database with timestamp + device_id
    
@app.get("/api/logs")
async def get_logs(date: str = None, device: str = None):
    # Return transcripts, optionally filtered
    
@app.get("/")
async def web_ui():
    # Simple HTML page showing timeline of transcripts
```

**Database**: SQLite is fine initially
```sql
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME,
    device_id TEXT,
    audio_path TEXT,
    transcript TEXT,
    rms_stats JSON
);
```

### Phase 2: Modify Desktop Client (1 day)
```python
# Add to main.py after successful transcription:
def upload_to_server(audio_path, transcript, stats):
    try:
        response = requests.post(
            f"{CENTRAL_SERVER}/api/upload",
            files={"audio": open(audio_path, "rb")},
            data={
                "device_id": DEVICE_ID,
                "transcript": transcript,
                "stats": json.dumps(stats)
            }
        )
    except Exception as e:
        # Log locally as fallback
        pass
```

### Phase 3: Mobile PWA (2-3 days)
```html
<!-- Simple HTML page -->
<button id="record">🎤 Record Voice Note</button>
<div id="status"></div>

<script>
let mediaRecorder;
let audioChunks = [];

document.getElementById('record').onclick = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    
    mediaRecorder.onstop = async () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('audio', blob);
        formData.append('device_id', 'phone');
        
        await fetch('/api/upload', { method: 'POST', body: formData });
        audioChunks = [];
    };
    
    mediaRecorder.start();
    // Stop after 60s or on user click
};
</script>
```

### Phase 4: Enhanced Features (ongoing)
- Full-text search across all transcripts
- Date range filtering
- Export to markdown/PDF
- Speaker identification (if multiple people)
- Automatic topic extraction
- Voice command: "Remember this" → flag for review

---

## Network Architecture Considerations

### Security
- **VPN**: Run server on home network, VPN from phone
- **Tailscale**: Zero-config VPN, works great for this
- **CloudFlare Tunnel**: Expose local server securely
- **Authentication**: Simple token in request header

### Transcription Server
**Current**: Local server at `192.168.0.142:8085`

**Options for mobile access**:
1. **VPN** (Tailscale/WireGuard): Phone joins home network
2. **Cloud VM**: Run Whisper on cloud (costs $$, privacy concern)
3. **On-device**: Use phone's Whisper.cpp or Web GPU (slower, drains battery)
4. **Hybrid**: Desktop server available when at home, fallback to cloud/on-device when away

---

## Cost-Benefit Analysis

| Approach | Implementation Time | Maintenance | Battery Impact | Coverage |
|----------|-------------------|-------------|----------------|----------|
| **Windows Port** | 1-2 weeks | Medium | Low | Desktop only |
| **Android Native** | 4-6 weeks | High | High | Mobile only |
| **PWA** | 1 week | Low | Medium | All devices |
| **Hybrid (PWA + Desktop)** | 1.5 weeks | Low-Medium | Low-Medium | Complete |

---

## My Recommendation: **Hybrid PWA Approach**

### Why?
1. **Fastest time to value**: Web UI in days, not weeks
2. **Cross-platform**: Works on Android, iOS, Windows, Mac, Linux
3. **No app stores**: Skip approval process
4. **Leverages existing code**: Transcription service reused
5. **Incremental**: Can build in stages
6. **Realistic usage**: True continuous recording drains battery; manual voice notes more practical on mobile

### Next Steps
1. Set up simple Flask/FastAPI server with `/upload` and `/logs` endpoints
2. Create basic web UI showing transcript timeline
3. Modify desktop client to post to server (with local fallback)
4. Create mobile PWA with "Record" button
5. Add Tailscale for secure mobile access

### Proof of Concept (1-2 hours)
I could create a minimal working demo:
- 100-line Flask server
- 50-line HTML mobile page
- Desktop client modification
- Test end-to-end: phone → server → transcript → web view

Would you like me to build this proof of concept now?

---

## Critical Multi-Device Coordination Issues

### Problem 1: Concurrent Microphones & Log Organization

**Scenario**: Multiple devices recording simultaneously:
- Desktop mic in office
- Phone in pocket walking around
- Laptop mic in meeting room
- Partner's phone in kitchen

**Questions**:
1. How to organize logs to avoid interleaving?
2. How to identify source/person/location?
3. How to handle transcript conflicts (two people talking in different rooms)?

### Solution Options

#### Option A: Per-Device Log Files
```
logs/
  device_office-desktop_2025-12-04.txt
  device_phone-raymond_2025-12-04.txt
  device_laptop-meeting_2025-12-04.txt
  device_phone-partner_2025-12-04.txt
```

**Pros**:
- Simple, no interleaving
- Clear source attribution
- Easy to review by location/person

**Cons**:
- Fragmented timeline
- Hard to search across all devices
- Lose chronological context

#### Option B: Single Log with Device Tags
```
[2025-12-04 09:15:23] [office-desktop] Checking email about database schema
[2025-12-04 09:16:45] [phone-raymond] Walking to meeting, thinking about API design
[2025-12-04 09:17:10] [office-desktop] Found that email, it mentions PostgreSQL
[2025-12-04 09:18:30] [laptop-meeting] In the meeting now, discussing architecture
```

**Pros**:
- Unified timeline
- See full context chronologically
- Single search across everything

**Cons**:
- Can be confusing if multiple people/locations active
- Harder to filter just one source

#### Option C: Hierarchical Organization (RECOMMENDED)
```
logs/
  unified/
    audio_log_2025-12-04.txt          # All devices chronologically
  by_device/
    office-desktop_2025-12-04.txt
    phone-raymond_2025-12-04.txt
    laptop-meeting_2025-12-04.txt
  by_person/
    raymond_2025-12-04.txt            # All Raymond's devices
    partner_2025-12-04.txt
  by_location/
    office_2025-12-04.txt
    meeting-room_2025-12-04.txt
```

**Database Schema**:
```sql
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    device_id TEXT NOT NULL,           -- "office-desktop"
    person_id TEXT,                    -- "raymond" (from login)
    location TEXT,                     -- "office", "meeting-room"
    audio_path TEXT,
    transcript TEXT,
    rms_stats JSON,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_device (device_id),
    INDEX idx_person (person_id),
    INDEX idx_location (location)
);
```

**Queries**:
```sql
-- Everything chronologically
SELECT * FROM transcripts ORDER BY timestamp;

-- Just my office mic
SELECT * FROM transcripts WHERE device_id = 'office-desktop';

-- Everything I said today (all my devices)
SELECT * FROM transcripts WHERE person_id = 'raymond' AND DATE(timestamp) = '2025-12-04';

-- Everything said in office today (any device in that room)
SELECT * FROM transcripts WHERE location = 'office' AND DATE(timestamp) = '2025-12-04';
```

### Device Identification Strategy

#### 1. Device Registration
When client starts, register with server:
```python
# On desktop daemon startup
DEVICE_ID = "office-desktop"  # Configured per installation
PERSON_ID = "raymond"          # From config or login
LOCATION = "office"            # From config

# On mobile PWA
# Show login screen first time
person_id = prompt("Your name?")  # Or Google Sign-In
location = prompt("Where are you?") or detect_location()
device_id = f"phone-{person_id}"
```

#### 2. Configuration Files
```yaml
# /opt/audio_logger/config.yaml (desktop)
device:
  id: office-desktop
  person: raymond
  location: office
  
server:
  url: https://transcribe.yourdomain.com
```

#### 3. Mobile PWA Login
```javascript
// First time user opens PWA
async function initialize() {
    let config = localStorage.getItem('audio_logger_config');
    
    if (!config) {
        // Show setup dialog
        const person = await promptOrGoogleSignIn();
        const deviceName = await prompt("Device name (e.g., 'phone', 'tablet')");
        
        config = {
            device_id: `${deviceName}-${person}`,
            person_id: person
        };
        
        localStorage.setItem('audio_logger_config', JSON.stringify(config));
    }
    
    return config;
}
```

### Location Detection Options

1. **Manual prompt**: "Where are you?" → "office", "home", "car"
2. **GPS + geofencing**: Auto-detect "office" if at work GPS coords
3. **WiFi SSID**: Detect location by network name
4. **Last used**: Remember last location per device

---

### Problem 2: Concurrent Transcription Server Limits

**Issue**: Mini Transcriber might not handle multiple concurrent requests

**Current Status**: Unknown capacity
- Single Whisper model loads ~2-8GB RAM
- Processing time: 10-30 seconds per 60s audio clip
- Can it queue? Does it crash on concurrent calls?

#### Solution A: Client-Side Queuing with Retry (RECOMMENDED for Phase 1)
```python
# In desktop client
import queue
import threading
from audio_logger_config import get_config_manager

config_manager = get_config_manager()
transcription_queue = queue.Queue()

def transcription_worker():
    """Background thread with exponential backoff retry."""
    while True:
        task = transcription_queue.get()
        audio_path, callback = task
        
        max_retries = 5
        base_delay = 2
        url = config_manager.get_transcription_url()
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    files={'file': open(audio_path, 'rb')},
                    timeout=60
                )
                
                if response.status_code == 503:  # Service busy
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"Server busy, retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                    
                response.raise_for_status()
                transcript = response.json().get('text', '')
                callback(transcript)
                break
                
            except Exception as e:
                if attempt == max_retries - 1:
                    callback(None)  # Failed
                else:
                    time.sleep(base_delay * (2 ** attempt))
        
        transcription_queue.task_done()

# Start worker
threading.Thread(target=transcription_worker, daemon=True).start()
```

#### Solution B: Multiple Transcription Workers (for scale)
```yaml
# docker-compose.yml
services:
  transcriber-1:
    image: minitranscriber
    ports: ["8085:8085"]
  transcriber-2:
    image: minitranscriber
    ports: ["8086:8085"]
  
  nginx:
    image: nginx
    # Round-robin load balance
```

#### Testing Concurrent Load
```python
# test_concurrent.py - simulate 3 devices uploading simultaneously
import asyncio
import aiohttp

async def send_audio(device_id):
    from audio_logger_config import get_config_manager
    config_manager = get_config_manager()
    url = config_manager.get_transcription_url()
    
    async with aiohttp.ClientSession() as session:
        for i in range(5):
            audio = open("test.wav", "rb")
            data = aiohttp.FormData()
            data.add_field('audio', audio)
            
            start = time.time()
            async with session.post(url, data=data) as resp:
                elapsed = time.time() - start
                print(f"[{device_id}] {i}: {elapsed:.1f}s - {resp.status}")
            
            await asyncio.sleep(60)

# Run 3 devices concurrently
asyncio.run(asyncio.gather(
    send_audio("office"),
    send_audio("phone"),
    send_audio("laptop")
))
```

---

### Recommended Architecture Summary

```
┌─────────────────┐
│  Desktop Daemon │
│  device_id:     │
│  office-desktop │──┐
│  person: raymond│  │
│  location:office│  │
└─────────────────┘  │
                     │
┌─────────────────┐  │    ┌──────────────────┐      ┌─────────────────┐
│  Phone PWA      │  ├───▶│  Central Server  │─────▶│ Mini Transcriber│
│  device_id:     │  │    │                  │      │  (queue/retry)  │
│  phone-raymond  │  │    │  • Upload API    │      └─────────────────┘
│  person: raymond│  │    │  • Database      │
│  location: GPS  │  │    │  • Web UI        │      ┌─────────────────┐
└─────────────────┘  │    └──────────────────┘      │   SQLite/       │
                     │                               │   PostgreSQL    │
┌─────────────────┐  │                               │                 │
│  Laptop Daemon  │  │                               │  transcripts    │
│  device_id:     │──┘                               │  devices        │
│  laptop-meeting │                                  │  users          │
│  person: raymond│                                  └─────────────────┘
│  location:conf-B│
└─────────────────┘
```

**Database Schema**:
```sql
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    name TEXT,
    created_at DATETIME
);

CREATE TABLE devices (
    id TEXT PRIMARY KEY,           -- "office-desktop"
    user_id TEXT REFERENCES users(id),
    device_type TEXT,              -- "daemon", "pwa"
    default_location TEXT
);

CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    device_id TEXT REFERENCES devices(id),
    user_id TEXT REFERENCES users(id),
    location TEXT,
    audio_path TEXT,
    transcript TEXT,
    rms_stats JSON,
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_device (device_id),
    INDEX idx_user (user_id),
    INDEX idx_location (location)
);
```

---

### Implementation Priority

1. **Phase 1**: Single device ✓ (current)
2. **Phase 2**: Add device_id tagging, test concurrent with retry logic
3. **Phase 3**: Central server + database for multi-device
4. **Phase 4**: User identification (config or login)
5. **Phase 5**: Location tagging (manual → auto)
6. **Phase 6**: Web UI with filtering by device/user/location

**Next Step**: Test current Mini Transcriber with concurrent requests to determine if we need immediate queuing solution.
