#!/usr/bin/env python3
"""
Configuration management for Audio Logger.
Handles loading, validation, and hostname resolution for services.
"""

import json
import socket
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""
    url: str
    model: str
    resolved_ip: Optional[str] = None
    resolved_port: Optional[int] = None
    is_localhost: bool = False

    def get_resolved_url(self) -> str:
        """Get the URL with hostname resolved to IP address."""
        if self.resolved_ip and self.resolved_port:
            return f"http://{self.resolved_ip}:{self.resolved_port}/transcribe"
        return self.url


@dataclass
class AudioLoggerConfig:
    """Main configuration for Audio Logger."""
    transcription: TranscriptionConfig
    audio_device: Optional[str] = None
    debug: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioLoggerConfig':
        """Create config from dictionary."""
        trans_data = data.get('transcription', {})
        transcription = TranscriptionConfig(
            url=trans_data.get('url', 'http://localhost:8085/transcribe'),
            model=trans_data.get('model', 'small')
        )

        return cls(
            transcription=transcription,
            audio_device=data.get('audio_device'),
            debug=data.get('debug', False)
        )


class ConfigManager:
    """Manages loading and validation of Audio Logger configuration."""

    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file or self._find_config_file()
        self.config: Optional[AudioLoggerConfig] = None
        self.last_resolution_time = 0
        self.resolution_cache_ttl = 300  # 5 minutes

    def _find_config_file(self) -> Path:
        """Find the configuration file in standard locations."""
        search_paths = [
            Path.cwd() / 'audio_logger.json',
            Path.cwd() / 'config.json',
            Path.home() / '.audio_logger.json',
            Path.home() / '.config' / 'audio_logger.json',
            Path('/etc/audio_logger/config.json'),
        ]

        for path in search_paths:
            if path.exists():
                return path

        # Default to current directory
        return Path.cwd() / 'audio_logger.json'

    def load_config(self) -> AudioLoggerConfig:
        """Load and validate configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                self.config = AudioLoggerConfig.from_dict(data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_file}: {e}")
                print("Using default configuration.")
                self.config = self._get_default_config()
        else:
            print(f"Config file not found at {self.config_file}")
            print("Using default configuration.")
            self.config = self._get_default_config()

        # Resolve hostnames and check localhost
        self._resolve_transcription_service()
        return self.config

    def _get_default_config(self) -> AudioLoggerConfig:
        """Get default configuration."""
        return AudioLoggerConfig(
            transcription=TranscriptionConfig(
                url='http://localhost:8085/transcribe',
                model='small'
            )
        )

    def _resolve_transcription_service(self):
        """Resolve hostname to IP and check for localhost preference."""
        if not self.config:
            return

        trans = self.config.transcription
        current_time = time.time()

        # Check if we need to re-resolve (cache expired)
        if current_time - self.last_resolution_time > self.resolution_cache_ttl:
            try:
                # Parse URL to extract host and port
                if trans.url.startswith('http://'):
                    url_part = trans.url[7:]  # Remove 'http://'
                else:
                    url_part = trans.url

                if '/' in url_part:
                    host_port = url_part.split('/')[0]
                else:
                    host_port = url_part

                if ':' in host_port:
                    host, port_str = host_port.split(':', 1)
                    try:
                        port = int(port_str)
                    except ValueError:
                        port = 8085
                else:
                    host = host_port
                    port = 8085

                # Check if localhost is available and preferred
                localhost_available = self._check_localhost_service(port)
                if localhost_available and host != 'localhost' and host != '127.0.0.1':
                    print(f"✓ Local transcription service found at localhost:{port}, using it instead of {host}:{port}")
                    trans.resolved_ip = '127.0.0.1'
                    trans.resolved_port = port
                    trans.is_localhost = True
                else:
                    # Resolve the configured hostname
                    try:
                        resolved_ip = socket.gethostbyname(host)
                        trans.resolved_ip = resolved_ip
                        trans.resolved_port = port
                        trans.is_localhost = (resolved_ip == '127.0.0.1')
                        print(f"✓ Resolved {host}:{port} to {resolved_ip}:{port}")
                    except socket.gaierror as e:
                        print(f"⚠ Failed to resolve hostname {host}: {e}")
                        print(f"  Will use hostname directly: {host}:{port}")
                        trans.resolved_ip = None
                        trans.resolved_port = None

                self.last_resolution_time = current_time

            except Exception as e:
                print(f"⚠ Error resolving transcription service: {e}")
                trans.resolved_ip = None
                trans.resolved_port = None

    def _check_localhost_service(self, port: int) -> bool:
        """Check if localhost transcription service is running."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get_transcription_url(self) -> str:
        """Get the resolved transcription URL."""
        if not self.config:
            self.load_config()

        return self.config.transcription.get_resolved_url()

    def get_transcription_model(self) -> str:
        """Get the transcription model."""
        if not self.config:
            self.load_config()

        return self.config.transcription.model

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        if not self.config:
            self.load_config()

        return self.config.debug

    def get_audio_device(self) -> Optional[str]:
        """Get the configured audio device."""
        if not self.config:
            self.load_config()

        return self.config.audio_device


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get the global config manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config() -> AudioLoggerConfig:
    """Load configuration (convenience function)."""
    return get_config_manager().load_config()


def get_transcription_url() -> str:
    """Get transcription URL (convenience function)."""
    return get_config_manager().get_transcription_url()


def get_transcription_model() -> str:
    """Get transcription model (convenience function)."""
    return get_config_manager().get_transcription_model()