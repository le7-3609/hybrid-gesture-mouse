"""
NullVoiceListener — no-op implementation of BaseVoiceListener.

Used when the application is started with --no-voice.
Satisfies the Liskov Substitution Principle: the controller never needs
an 'if voice_enabled' branch — it always calls the same interface.
"""
from __future__ import annotations
from interfaces.voice_listener import BaseVoiceListener
from domain.voice_command import VoiceCommand


class NullVoiceListener(BaseVoiceListener):
    """
    Silent voice listener that does absolutely nothing.
    Implements the Null Object pattern — safe to use everywhere a
    BaseVoiceListener is expected.
    """

    def start(self) -> None:
        """No-op."""

    def stop(self) -> None:
        """No-op."""

    def get_latest_command(self) -> VoiceCommand | None:
        """Always returns None — no commands ever arrive."""
        return None

    def is_recording(self) -> bool:
        """Always False — no audio is ever captured."""
        return False
