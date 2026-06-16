"""
FPSCounter — lightweight exponential moving average FPS tracker.
"""
import time


class FPSCounter:
    """Tracks frame rate using EMA smoothing."""

    def __init__(self, alpha: float = 0.1) -> None:
        self._alpha    = alpha
        self._fps      = 0.0
        self._prev_ts  = time.monotonic()

    def tick(self) -> float:
        """Call once per frame. Returns the current smoothed FPS."""
        now   = time.monotonic()
        delta = now - self._prev_ts
        self._prev_ts = now
        if delta > 0:
            instant = 1.0 / delta
            self._fps = self._alpha * instant + (1 - self._alpha) * self._fps
        return round(self._fps, 1)
