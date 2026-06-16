"""
WhisperVoiceListener — real microphone + local Whisper speech recognition.

Architecture
------------
A dedicated daemon thread runs the audio capture / VAD / transcription loop
so the main camera loop is NEVER blocked.  Thread-safe communication is
via a queue.Queue (capacity 1, newest wins).

Processing pipeline
-------------------
1. sounddevice records audio chunks at 16 kHz (Whisper's native sample rate).
2. A simple RMS energy VAD detects speech start and silence end.
3. When a speech segment ends, the accumulated numpy array is sent to
   whisper.model.transcribe() — runs entirely locally, zero network calls.
4. The transcript is wrapped in a VoiceCommand and placed in the queue.
5. The camera loop calls get_latest_command() which drains the queue.

Lazy loading
------------
Whisper and sounddevice are imported inside the class to allow the rest of the
application to start (and be tested) without these heavy dependencies installed.
"""
from __future__ import annotations
import queue
import threading
import time
import numpy as np
from interfaces.voice_listener import BaseVoiceListener
from domain.voice_command import VoiceCommand
from config.settings import (
    VOICE_SAMPLE_RATE,
    VOICE_SILENCE_THRESHOLD,
    VOICE_MIN_DURATION,
    VOICE_MAX_DURATION,
    VOICE_WHISPER_MODEL,
)


class WhisperVoiceListener(BaseVoiceListener):
    """
    Real-time voice listener using sounddevice + Whisper tiny (local).

    Parameters
    ----------
    model_name : str
        Whisper model size: 'tiny' | 'base' | 'small'.
        'tiny' (~75 MB) runs in <1 s on most CPUs.
    sample_rate : int
        Audio sample rate in Hz (must match Whisper's expectation: 16 000).
    silence_threshold : float
        RMS energy level below which audio is considered silence.
    min_duration : float
        Minimum speech duration (seconds) before transcribing.
    max_duration : float
        Hard cap on recording length before forced transcription.
    """

    def __init__(
        self,
        model_name: str        = VOICE_WHISPER_MODEL,
        sample_rate: int       = VOICE_SAMPLE_RATE,
        silence_threshold: float = VOICE_SILENCE_THRESHOLD,
        min_duration: float    = VOICE_MIN_DURATION,
        max_duration: float    = VOICE_MAX_DURATION,
    ) -> None:
        self._model_name        = model_name
        self._sample_rate       = sample_rate
        self._silence_threshold = silence_threshold
        self._min_duration      = min_duration
        self._max_duration      = max_duration

        self._command_queue: queue.Queue[VoiceCommand] = queue.Queue(maxsize=1)
        self._recording        = threading.Event()   # set = currently recording audio
        self._stop_event       = threading.Event()
        self._thread: threading.Thread | None = None
        self._model            = None               # lazy-loaded

    # ------------------------------------------------------------------
    # BaseVoiceListener interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load the Whisper model (once) and start the listener thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="WhisperVoiceListener",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the listener thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_latest_command(self) -> VoiceCommand | None:
        """
        Return the most recent VoiceCommand (and remove it from the queue),
        or None if no new command has arrived.
        Thread-safe.
        """
        try:
            return self._command_queue.get_nowait()
        except queue.Empty:
            return None

    def is_recording(self) -> bool:
        """True while a speech segment is being captured."""
        return self._recording.is_set()

    # ------------------------------------------------------------------
    # Private — listener thread
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy-load Whisper model on first use."""
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_name)
        return self._model

    def _listen_loop(self) -> None:
        """
        Main capture / VAD / transcription loop running in the daemon thread.
        Uses sounddevice's InputStream for low-latency chunk-by-chunk reading.
        """
        try:
            import sounddevice as sd
        except ImportError:
            return   # sounddevice not installed → silent failure

        model = self._load_model()

        chunk_size   = int(self._sample_rate * 0.1)   # 100 ms chunks
        audio_buffer: list[np.ndarray] = []
        recording    = False
        silence_dur  = 0.0
        record_dur   = 0.0

        def _rms(data: np.ndarray) -> float:
            return float(np.sqrt(np.mean(data ** 2)))

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=1,
            dtype="float32",
            blocksize=chunk_size,
        ) as stream:
            while not self._stop_event.is_set():
                chunk, _ = stream.read(chunk_size)
                chunk = chunk.flatten()
                energy = _rms(chunk)

                if not recording:
                    if energy > self._silence_threshold:
                        recording = True
                        record_dur = 0.0
                        audio_buffer = [chunk]
                        self._recording.set()
                else:
                    audio_buffer.append(chunk)
                    record_dur += 0.1

                    if energy < self._silence_threshold:
                        silence_dur += 0.1
                    else:
                        silence_dur = 0.0

                    ended = (
                        silence_dur > 0.6 and record_dur >= self._min_duration
                    ) or record_dur >= self._max_duration

                    if ended:
                        self._recording.clear()
                        recording = False
                        silence_dur = 0.0

                        # Transcribe in-thread (blocks ~0.5–1 s for 'tiny')
                        audio_np = np.concatenate(audio_buffer).astype(np.float32)
                        try:
                            result = model.transcribe(
                                audio_np,
                                fp16=False,
                                language="en",
                            )
                            text = result.get("text", "").strip().lower()
                            if text:
                                cmd = VoiceCommand(transcript=text)
                                # Overwrite old command if queue is full
                                try:
                                    self._command_queue.put_nowait(cmd)
                                except queue.Full:
                                    try:
                                        self._command_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    self._command_queue.put_nowait(cmd)
                        except Exception:
                            pass   # transcription errors are non-fatal

                        audio_buffer = []
