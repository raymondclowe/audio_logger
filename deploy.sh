#!/bin/bash
# Audio Logger Service Deployment Script
# Deploys the audio logger to /opt and sets up systemd service

set -e

AUDIO_LOGGER_HOME="/opt/audio_logger"
SERVICE_FILE="/etc/systemd/system/audio_logger.service"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Audio Logger Deployment ==="

# Stop existing service if running
if systemctl is-active --quiet audio_logger; then
    echo "Stopping existing audio_logger service..."
    sudo systemctl stop audio_logger
fi

# Create audiologger user and group
echo "Creating audiologger user and group..."
sudo groupadd -f audiologger || true
sudo useradd -r -g audiologger -s /usr/sbin/nologin -m -d "$AUDIO_LOGGER_HOME" audiologger 2>/dev/null || true

# Create /opt/audio_logger directory
echo "Creating $AUDIO_LOGGER_HOME directory..."
sudo mkdir -p "$AUDIO_LOGGER_HOME"

# Copy project files
echo "Copying project files to $AUDIO_LOGGER_HOME..."
sudo cp -r "$SOURCE_DIR"/.venv "$AUDIO_LOGGER_HOME/" || sudo cp -r "$SOURCE_DIR"/.venv "$AUDIO_LOGGER_HOME/"
sudo cp "$SOURCE_DIR"/main.py "$AUDIO_LOGGER_HOME/"
sudo cp "$SOURCE_DIR"/audio_logger_config.py "$AUDIO_LOGGER_HOME/"
sudo cp "$SOURCE_DIR"/pyproject.toml "$AUDIO_LOGGER_HOME/"
sudo cp "$SOURCE_DIR"/README.md "$AUDIO_LOGGER_HOME/"
# Copy config file if it exists
[ -f "$SOURCE_DIR/audio_logger.json" ] && sudo cp "$SOURCE_DIR"/audio_logger.json "$AUDIO_LOGGER_HOME/"
# Copy room calibration config if it exists
[ -f "$SOURCE_DIR/room_calibration_config.json" ] && sudo cp "$SOURCE_DIR"/room_calibration_config.json "$AUDIO_LOGGER_HOME/"
sudo mkdir -p "$AUDIO_LOGGER_HOME/logs"
sudo mkdir -p "$AUDIO_LOGGER_HOME/temp"

# Set ownership and permissions
echo "Setting ownership and permissions..."
sudo chown -R audiologger:audiologger "$AUDIO_LOGGER_HOME"
sudo chmod 755 "$AUDIO_LOGGER_HOME"
sudo chmod 755 "$AUDIO_LOGGER_HOME/logs"
sudo chmod 755 "$AUDIO_LOGGER_HOME/temp"
sudo chmod 755 "$AUDIO_LOGGER_HOME/.venv/bin/python"
sudo chmod 644 "$AUDIO_LOGGER_HOME/main.py"

# Add current user to audiologger group
CURRENT_USER="${SUDO_USER:-$USER}"
echo "Adding $CURRENT_USER to audiologger group..."
sudo usermod -a -G audiologger "$CURRENT_USER"

# Install systemd service
echo "Installing systemd service..."
sudo cp "$SOURCE_DIR/audio_logger.service" "$SERVICE_FILE"
sudo chmod 644 "$SERVICE_FILE"

# Reload systemd and enable service
echo "Enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable audio_logger

# Start service
echo "Starting audio_logger service..."
sudo systemctl start audio_logger

echo ""
echo "=== Deployment Complete ==="
echo "Service file: $SERVICE_FILE"
echo "Application home: $AUDIO_LOGGER_HOME"
echo ""
echo "Check status with: sudo systemctl status audio_logger"
echo "View logs with: sudo journalctl -u audio_logger -f"
echo ""
echo "Note: You are now a member of the 'audiologger' group and can read logs:"
echo "  logs are at: $AUDIO_LOGGER_HOME/logs/"
echo "  temp files at: $AUDIO_LOGGER_HOME/temp/"
