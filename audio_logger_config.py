#!/usr/bin/env python3
"""
Configuration management for Audio Logger.
Handles loading, validation, and hostname resolution for services.
"""

import json
import socket
import time
import requests
from requests.exceptions import RequestException
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class TranscriptionConfig:
    """Configuration for transcription service."""
    url: str
    model: str
    language: str = "en"
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
    record_duration: int = 60
    overlap_duration: int = 5

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AudioLoggerConfig':
        """Create config from dictionary."""
        trans_data = data.get('transcription', {})
        transcription = TranscriptionConfig(
            url=trans_data.get('url', 'http://localhost:8085/transcribe'),
            model=trans_data.get('model', 'small'),
            language=trans_data.get('language', 'en')
        )

        return cls(
            transcription=transcription,
            audio_device=data.get('audio_device'),
            debug=data.get('debug', False),
            record_duration=data.get('record_duration', 60),
            overlap_duration=data.get('overlap_duration', 5)
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

    def get_record_duration(self) -> int:
        """Get the record duration."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config, 'record_duration'):
            return self.config.record_duration
        raise AttributeError("record_duration is not defined in the configuration")

    def get_overlap_duration(self) -> int:
        """Get the overlap duration."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config, 'overlap_duration'):
            return self.config.overlap_duration
        raise AttributeError("overlap_duration is not defined in the configuration")

    def get_transcription_url(self) -> str:
        """Get the resolved transcription URL."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config.transcription, 'get_resolved_url'):
            return self.config.transcription.get_resolved_url()
        raise AttributeError("transcription URL is not defined in the configuration")

    def get_transcription_model(self) -> str:
        """Get the transcription model."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config.transcription, 'model'):
            return self.config.transcription.model
        raise AttributeError("transcription model is not defined in the configuration")

    def get_transcription_language(self) -> str:
        """Get the transcription language code."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config.transcription, 'language'):
            return self.config.transcription.language
        raise AttributeError("transcription language is not defined in the configuration")

    def is_debug_enabled(self) -> bool:
        """Check if debug mode is enabled."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config, 'debug'):
            return self.config.debug
        raise AttributeError("debug is not defined in the configuration")

    def get_audio_device(self) -> Optional[str]:
        """Get the configured audio device."""
        if self.config is None:
            self.load_config()
        if self.config and hasattr(self.config, 'audio_device'):
            return self.config.audio_device
        raise AttributeError("audio_device is not defined in the configuration")


class TranscriptionService:
    """Handles interaction with the Mini Transcriber service."""

    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url
        self.timeout = timeout

    def transcribe(self, audio_file_path: str, language: str = "en", model: str = "small") -> Optional[Dict[str, Any]]:
        """Send an audio file for transcription and wait for the result."""
        url = f"{self.base_url}/transcribe"
        files = {"audio": open(audio_file_path, "rb")}
        data = {"language": language, "model": model}

        try:
            response = requests.post(url, files=files, data=data, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Error during transcription request: {e}")
            return None
        finally:
            files["audio"].close()

    def handle_transcription(self, audio_file_path: str, retries: int = 3, backoff_factor: int = 2) -> Optional[Dict[str, Any]]:
        """Handle transcription with retries and backoff."""
        attempt = 0
        while attempt < retries:
            result = self.transcribe(audio_file_path)
            if result:
                return result

            attempt += 1
            wait_time = backoff_factor ** attempt
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)

        print("Transcription failed after multiple attempts.")
        return None


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


def get_transcription_language() -> str:
    """Get transcription language (convenience function)."""
    return get_config_manager().get_transcription_language()


__all__ = ["get_config_manager", "load_config"]