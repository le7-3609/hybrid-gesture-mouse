"""
BaseVoiceListener — Abstract Base Class for voice input sources.

Concrete implementations:
  WhisperVoiceListener  — real microphone + local Whisper model
  NullVoiceListener     — no-op implementation for --no-voice mode

The controller always depends on this interface, never on a concrete class.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from domain.voice_command import VoiceCommand


class BaseVoiceListener(ABC):

    @abstractmethod
    def start(self) -> None:
        """Begin listening for voice input (may start a background thread)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop listening and release all audio resources."""

    @abstractmethod
    def get_latest_command(self) -> VoiceCommand | None:
        """
        Return the most recently recognised command and clear the internal
        buffer, or None if no new command is available since the last call.
        Thread-safe; safe to call from the camera loop.
        """

    @abstractmethod
    def is_recording(self) -> bool:
        """Return True while audio is actively being captured."""
