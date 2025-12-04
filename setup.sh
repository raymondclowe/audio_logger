#!/bin/bash
# Audio Logger Setup Script
# Automatically detects audio hardware and configures the system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_PY="$SCRIPT_DIR/main.py"
SERVICE_FILE="$SCRIPT_DIR/audio_logger.service"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy.sh"

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== Audio Logger Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then 
   echo -e "${RED}Error: Please run this script as a normal user (not with sudo)${NC}"
   echo "The script will prompt for sudo when needed."
   exit 1
fi

# Parse command line arguments
INSTALL_SERVICE=false
PRODUCTION_INSTALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --install-service)
            INSTALL_SERVICE=true
            shift
            ;;
        --production)
            PRODUCTION_INSTALL=true
            INSTALL_SERVICE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --install-service    Install and start systemd service"
            echo "  --production         Full production install to /opt"
            echo "  --help              Show this help message"
            echo ""
            echo "Without options, performs setup and configuration only."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# 1. Install system dependencies
echo "Step 1/8: Checking and installing system dependencies..."

# Check for required commands
MISSING_DEPS=()

if ! command -v sox &> /dev/null; then
    MISSING_DEPS+=("sox" "libsox-fmt-all")
fi

if ! command -v arecord &> /dev/null; then
    MISSING_DEPS+=("alsa-utils")
fi

if ! command -v amixer &> /dev/null && ! echo "${MISSING_DEPS[@]}" | grep -q "alsa-utils"; then
    MISSING_DEPS+=("alsa-utils")
fi

if ! command -v curl &> /dev/null; then
    MISSING_DEPS+=("curl")
fi

if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo -e "${YELLOW}Installing missing dependencies: ${MISSING_DEPS[*]}${NC}"
    sudo apt-get update -qq
    sudo apt-get install -y "${MISSING_DEPS[@]}"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ All system dependencies present${NC}"
fi

# Verify installations
for cmd in sox arecord amixer curl; do
    if ! command -v $cmd &> /dev/null; then
        echo -e "${RED}✗ Failed to install $cmd${NC}"
        exit 1
    fi
done

# 2. Install Python dependencies
echo ""
echo "Step 2/8: Installing Python dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv for package management..."
    uv sync
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    echo -e "${GREEN}✓ Python packages installed with uv${NC}"
else
    echo "uv not found, using pip..."
    if [ ! -d ".venv" ]; then
        python3 -m venv .venv
    fi
    .venv/bin/pip install -q requests numpy psutil
    PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"
    echo -e "${GREEN}✓ Python packages installed with pip${NC}"
fi

# Verify Python dependencies
if ! $PYTHON_BIN -c "import requests, numpy, psutil" 2>/dev/null; then
    echo -e "${RED}✗ Failed to install Python dependencies${NC}"
    exit 1
fi

# 3. Detect audio devices
echo ""
echo "Step 3/8: Detecting audio capture devices..."
echo ""
arecord -l
echo ""

# Find USB audio device
USB_DEVICE=$(arecord -l | grep -i "USB\|usb" | head -1 | grep -oP 'card \K[0-9]+' || echo "")

if [ -z "$USB_DEVICE" ]; then
    echo -e "${YELLOW}No USB audio device found automatically.${NC}"
    echo "Available devices:"
    arecord -l
    read -p "Enter card number to use: " USB_DEVICE
    if [ -z "$USB_DEVICE" ]; then
        echo -e "${RED}Error: No card number provided${NC}"
        exit 1
    fi
fi

AUDIO_DEVICE="plughw:${USB_DEVICE},0"
echo -e "${GREEN}✓ Selected audio device: $AUDIO_DEVICE (card $USB_DEVICE)${NC}"

# 4. Test and configure audio gain
echo ""
echo "Step 4/8: Testing audio device and configuring gain..."

# Find microphone controls
MIC_NUMID=""
AUTO_GAIN_NUMID=""

# Try to find Mic Capture Volume control
MIC_NUMID=$(amixer -c "$USB_DEVICE" controls | grep -i "Mic.*Capture.*Volume" | head -1 | grep -oP 'numid=\K[0-9]+' || echo "")
if [ -n "$MIC_NUMID" ]; then
    # Get max value for this control
    MAX_GAIN=$(amixer -c "$USB_DEVICE" cget "numid=$MIC_NUMID" | grep -oP 'max=\K[0-9]+' || echo "16")
    echo "Found microphone volume control (numid=$MIC_NUMID), setting to maximum ($MAX_GAIN)..."
    amixer -c "$USB_DEVICE" cset "numid=$MIC_NUMID" "$MAX_GAIN" > /dev/null
    echo -e "${GREEN}✓ Microphone gain set to maximum${NC}"
else
    echo -e "${YELLOW}⚠ Could not find microphone volume control${NC}"
fi

# Check for Auto Gain Control
AUTO_GAIN_NUMID=$(amixer -c "$USB_DEVICE" controls | grep -i "Auto.*Gain" | head -1 | grep -oP 'numid=\K[0-9]+' || echo "")
if [ -n "$AUTO_GAIN_NUMID" ]; then
    echo "Found auto gain control (numid=$AUTO_GAIN_NUMID), enabling..."
    amixer -c "$USB_DEVICE" cset "numid=$AUTO_GAIN_NUMID" on > /dev/null 2>&1 || true
    echo -e "${GREEN}✓ Auto gain control enabled${NC}"
fi

# Test recording
echo "Testing audio capture (3 seconds) - please make some noise..."
TEST_FILE="/tmp/audio_test_$$.wav"
if arecord -D "$AUDIO_DEVICE" -d 3 -f cd -t wav "$TEST_FILE" 2>/dev/null; then
    echo -e "${GREEN}✓ Audio capture working!${NC}"
    
    # Analyze RMS levels
    RMS_PEAK=$($PYTHON_BIN << EOF
import wave
import numpy as np
try:
    with wave.open('$TEST_FILE', 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        if wf.getnchannels() > 1:
            audio = audio.reshape(-1, wf.getnchannels()).mean(axis=1).astype(np.int16)
        win_samples = max(1, int(wf.getframerate() * 0.05))
        pad = (-audio.size) % win_samples
        if pad:
            audio = np.pad(audio, (0, pad), mode='constant')
        windows = audio.reshape(-1, win_samples).astype(np.float64)
        rms_windows = np.sqrt(np.mean(windows ** 2, axis=1))
        print(f"{float(np.max(rms_windows)):.1f}")
except:
    print("0")
EOF
)
    echo "  Peak RMS level: $RMS_PEAK"
    if [ -n "$RMS_PEAK" ] && [ "$RMS_PEAK" != "0" ]; then
        if (( $(echo "$RMS_PEAK < 100" | bc -l 2>/dev/null || echo 0) )); then
            echo -e "${YELLOW}  ⚠ Warning: Audio levels are very low. Consider:${NC}"
            echo "    - Speaking louder or moving closer to microphone"
            echo "    - Checking microphone is not muted"
            echo "    - Verifying correct input device is selected"
        elif (( $(echo "$RMS_PEAK > 200" | bc -l 2>/dev/null || echo 1) )); then
            echo -e "${GREEN}  ✓ Good audio levels detected${NC}"
        fi
    fi
    rm -f "$TEST_FILE"
else
    echo -e "${RED}✗ Audio capture failed!${NC}"
    exit 1
fi

# 5. Update configuration in main.py
echo ""
echo "Step 5/8: Updating configuration..."

# Update AUDIO_DEVICE
if grep -q "^AUDIO_DEVICE = " "$MAIN_PY"; then
    sed -i "s|^AUDIO_DEVICE = .*|AUDIO_DEVICE = \"$AUDIO_DEVICE\"|" "$MAIN_PY"
    echo -e "${GREEN}✓ Updated AUDIO_DEVICE to $AUDIO_DEVICE${NC}"
fi

# Update amixer command if we found the control
if [ -n "$MIC_NUMID" ]; then
    # Update the amixer command in record_audio function
    if grep -q "amixer.*-c.*set.*Capture" "$MAIN_PY" || grep -q "amixer.*-c.*cset.*numid" "$MAIN_PY"; then
        sed -i "s|\"amixer\", \"-c\", \"[0-9]*\"|\"amixer\", \"-c\", \"$USB_DEVICE\"|g" "$MAIN_PY"
        sed -i "s|\"cset\", \"numid=[0-9]*\"|\"cset\", \"numid=$MIC_NUMID\"|g" "$MAIN_PY"
        sed -i "s|, \"[0-9]*\"\]|, \"$MAX_GAIN\"]|g" "$MAIN_PY"
        echo -e "${GREEN}✓ Updated microphone gain control (card=$USB_DEVICE, numid=$MIC_NUMID, max=$MAX_GAIN)${NC}"
    fi
fi

# 6. Test the transcription service
echo ""
echo "Step 6/8: Testing transcription service..."
TRANSCRIBE_URL=$(grep "^TRANSCRIBE_URL = " "$MAIN_PY" | grep -oP '".*?"' | tr -d '"' || echo "")

if [ -n "$TRANSCRIBE_URL" ]; then
    echo "Testing connection to $TRANSCRIBE_URL..."
    if curl -s --connect-timeout 5 "$TRANSCRIBE_URL" -X OPTIONS > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Transcription service is reachable${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Cannot reach transcription service at $TRANSCRIBE_URL${NC}"
        echo "  You may need to update TRANSCRIBE_URL in main.py"
    fi
else
    echo -e "${YELLOW}⚠ Could not find TRANSCRIBE_URL in main.py${NC}"
fi

# 7. Create user and group (if installing service)
if [ "$INSTALL_SERVICE" = true ]; then
    echo ""
    echo "Step 7/8: Setting up system user and permissions..."
    
    # Create audiologger group
    if ! getent group audiologger > /dev/null 2>&1; then
        sudo groupadd audiologger
        echo -e "${GREEN}✓ Created audiologger group${NC}"
    else
        echo "  audiologger group already exists"
    fi
    
    # Create audiologger user
    if ! id audiologger > /dev/null 2>&1; then
        if [ "$PRODUCTION_INSTALL" = true ]; then
            sudo useradd -r -g audiologger -s /usr/sbin/nologin -m -d /opt/audio_logger audiologger
        else
            sudo useradd -r -g audiologger -s /usr/sbin/nologin audiologger
        fi
        echo -e "${GREEN}✓ Created audiologger user${NC}"
    else
        echo "  audiologger user already exists"
    fi
    
    # Add current user to audiologger group
    if ! groups "$USER" | grep -q audiologger; then
        sudo usermod -a -G audiologger "$USER"
        echo -e "${GREEN}✓ Added $USER to audiologger group${NC}"
        echo -e "${YELLOW}  Note: Log out and back in for group membership to take effect${NC}"
    else
        echo "  $USER already in audiologger group"
    fi
    
    # Add audiologger user to audio group for device access
    if ! groups audiologger | grep -q audio; then
        sudo usermod -a -G audio,input audiologger
        echo -e "${GREEN}✓ Added audiologger to audio and input groups${NC}"
    fi
else
    echo ""
    echo "Step 7/8: Skipping user creation (use --install-service to enable)"
fi

# 8. Install service
if [ "$INSTALL_SERVICE" = true ]; then
    echo ""
    echo "Step 8/8: Installing systemd service..."
    
    if [ "$PRODUCTION_INSTALL" = true ]; then
        # Run deploy script for production
        if [ -x "$DEPLOY_SCRIPT" ]; then
            echo "Running production deployment..."
            bash "$DEPLOY_SCRIPT"
        else
            echo -e "${RED}✗ Deploy script not found or not executable${NC}"
            exit 1
        fi
    else
        # Install service from current directory
        sudo cp "$SERVICE_FILE" /etc/systemd/system/
        sudo systemctl daemon-reload
        sudo systemctl enable audio_logger
        sudo systemctl restart audio_logger
        echo -e "${GREEN}✓ Service installed and started${NC}"
    fi
else
    echo ""
    echo "Step 8/8: Skipping service installation (use --install-service to enable)"
fi

# Summary
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Configuration Summary:"
echo "  Audio Device: $AUDIO_DEVICE (card $USB_DEVICE)"
[ -n "$MIC_NUMID" ] && echo "  Mic Control: numid=$MIC_NUMID, max gain=$MAX_GAIN"
echo "  Transcriber: ${TRANSCRIBE_URL:-Not configured}"
echo ""

if [ "$INSTALL_SERVICE" = true ]; then
    echo "Service Status:"
    if systemctl is-active --quiet audio_logger; then
        echo -e "  ${GREEN}✓ Service is running${NC}"
    else
        echo -e "  ${YELLOW}⚠ Service is not running${NC}"
    fi
    echo ""
    echo "Useful Commands:"
    echo "  Check status:    sudo systemctl status audio_logger"
    echo "  View logs:       sudo journalctl -u audio_logger -f"
    echo "  Stop service:    sudo systemctl stop audio_logger"
    echo "  Start service:   sudo systemctl start audio_logger"
    echo "  Restart service: sudo systemctl restart audio_logger"
    if [ "$PRODUCTION_INSTALL" = true ]; then
        echo ""
        echo "Log files: /opt/audio_logger/logs/"
        echo "Temp files: /opt/audio_logger/temp/"
    fi
else
    echo "Next Steps:"
    echo "  1. Test manually:        $PYTHON_BIN main.py"
    echo "  2. Install service:      bash setup.sh --install-service"
    echo "  3. Production install:   bash setup.sh --production"
fi

echo ""
echo "Configuration files updated:"
echo "  - $MAIN_PY"
echo ""
