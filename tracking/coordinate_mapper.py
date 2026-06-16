"""
CoordinateMapper — maps raw pixel positions to smoothed screen coordinates.

Responsibilities:
  1. Active-zone cropping: only the inner portion of the webcam frame
     is used for control, reducing jitter at the edges.
  2. Linear mapping: crops coordinates to full-screen range.
  3. EMA smoothing: exponential moving average reduces cursor jitter.

This class contains NO business logic — it is a pure math utility.
"""
from __future__ import annotations


# Active-zone margins as fraction of frame (crop this much from each edge)
_MARGIN = 0.15


class CoordinateMapper:
    """
    Maps a raw (x, y) pixel position inside the webcam frame to a smoothed
    absolute screen position.

    Parameters
    ----------
    screen_width, screen_height : int
        Monitor resolution returned by the OS mouse service.
    smoothing : float
        EMA factor α ∈ (0, 1].  Lower = more smoothing, higher = more responsive.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        smoothing: float = 0.25,
    ) -> None:
        self.screen_width  = screen_width
        self.screen_height = screen_height
        self.smoothing     = smoothing

        self.prev_x: float | None = None
        self.prev_y: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, x: float, y: float) -> None:
        """Seed the EMA with a known position (e.g. current cursor pos)."""
        self.prev_x = float(x)
        self.prev_y = float(y)

    def map_and_smooth(
        self,
        raw_x: int,
        raw_y: int,
        frame_width: int,
        frame_height: int,
    ) -> tuple[int, int]:
        """
        Convert a raw pixel position to a smoothed screen position.

        Returns
        -------
        (screen_x, screen_y) : tuple[int, int]
            Clamped, smoothed absolute screen coordinates.
        """
        # 1. Compute active-zone pixel boundaries
        x_min = int(frame_width  * _MARGIN)
        x_max = int(frame_width  * (1 - _MARGIN))
        y_min = int(frame_height * _MARGIN)
        y_max = int(frame_height * (1 - _MARGIN))

        # 2. Clamp raw position to active zone
        clamped_x = max(x_min, min(x_max, raw_x))
        clamped_y = max(y_min, min(y_max, raw_y))

        # 3. Linear map to screen coordinates
        mapped_x = int((clamped_x - x_min) / max(x_max - x_min, 1) * self.screen_width)
        mapped_y = int((clamped_y - y_min) / max(y_max - y_min, 1) * self.screen_height)

        # 4. EMA smoothing
        if self.prev_x is None or self.prev_y is None:
            smooth_x, smooth_y = float(mapped_x), float(mapped_y)
        else:
            alpha   = self.smoothing
            smooth_x = alpha * mapped_x + (1 - alpha) * self.prev_x
            smooth_y = alpha * mapped_y + (1 - alpha) * self.prev_y

        self.prev_x = smooth_x
        self.prev_y = smooth_y

        # 5. Clamp to screen bounds
        final_x = max(0, min(self.screen_width  - 1, int(smooth_x)))
        final_y = max(0, min(self.screen_height - 1, int(smooth_y)))
        return final_x, final_y
