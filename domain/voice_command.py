"""
VoiceCommand — immutable data object carrying a recognised speech transcript.
Produced by any BaseVoiceListener implementation.
"""
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class VoiceCommand:
    """
    Represents a single recognised voice utterance.

    Attributes
    ----------
    transcript : str
        Raw text returned by the speech recognition model (lower-cased).
    confidence : float
        Recognition confidence in [0, 1].  Use 1.0 when the backend does
        not expose a confidence score (e.g. Whisper).
    timestamp  : float
        Unix epoch seconds when the command was captured.
    """
    transcript : str
    confidence : float = 1.0
    timestamp  : float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f'VoiceCommand("{self.transcript}", conf={self.confidence:.2f})'
