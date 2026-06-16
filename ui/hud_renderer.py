"""
HudRenderer — all OpenCV drawing code for the Gesture+ heads-up display.

Responsibilities (one class, one concern: drawing):
  - Glassmorphic semi-transparent overlay panels
  - Per-state colour-coded status bar
  - Active-zone boundary rectangle
  - Real-time FPS and screen target position
  - Majority-voting history queue visualisation
  - Skeletal hand landmark overlay
  - Voice activity indicator (pulsing mic icon when recording)
  - Last voice command feedback (fades after VOICE_DISPLAY_SECONDS)

No business logic lives here — this class only reads data and draws.
"""
from __future__ import annotations
import time
import cv2
import numpy as np
from domain.gesture_state import GestureState
from config.settings import STATE_COLORS, VOICE_DISPLAY_SECONDS

# Colours (BGR)
_WHITE  = (255, 255, 255)
_BLACK  = (0,   0,   0)
_GREEN  = (0,   220, 100)
_RED    = (0,   60,  220)
_YELLOW = (0,   220, 220)
_GREY   = (80,  80,  80)

# Active-zone margin (fraction of frame)
_MARGIN = 0.15

# Font
_FONT       = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SMALL = 0.5
_FONT_MED   = 0.65
_FONT_LARGE = 0.85


class HudRenderer:
    """OpenCV-based heads-up display renderer for the Gesture+ controller."""

    def __init__(self) -> None:
        self._voice_text      = ""
        self._voice_routed    = False
        self._voice_expire_ts = 0.0

    # ------------------------------------------------------------------
    # Public API called once per frame
    # ------------------------------------------------------------------

    def set_voice_feedback(self, transcript: str, routed: bool) -> None:
        """Store the latest voice command to display for VOICE_DISPLAY_SECONDS."""
        self._voice_text      = transcript
        self._voice_routed    = routed
        self._voice_expire_ts = time.monotonic() + VOICE_DISPLAY_SECONDS

    def draw_hand_landmarks(self, frame, hand_landmarks) -> None:
        """Draw the skeletal hand joints/connections with neon overlay."""
        try:
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_hands   = mp.solutions.hands
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 120), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 150, 255), thickness=2),
            )
        except (AttributeError, ModuleNotFoundError):
            from utils.mediapipe_shim import draw_custom_landmarks
            draw_custom_landmarks(frame, hand_landmarks)

    def draw_hud(
        self,
        frame,
        state:        GestureState,
        confidence:   float,
        screen_x:     int,
        screen_y:     int,
        fps:          float,
        vote_history: list,
        is_recording: bool = False,
    ) -> None:
        """Composite all HUD elements onto the frame (in-place)."""
        h, w = frame.shape[:2]

        self._draw_active_zone(frame, w, h)
        self._draw_status_panel(frame, state, confidence, fps, screen_x, screen_y, w)
        self._draw_vote_history(frame, vote_history, w, h)
        self._draw_voice_indicator(frame, is_recording, w)
        self._draw_voice_feedback(frame, w, h)

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_active_zone(self, frame, w: int, h: int) -> None:
        x1 = int(w * _MARGIN)
        y1 = int(h * _MARGIN)
        x2 = int(w * (1 - _MARGIN))
        y2 = int(h * (1 - _MARGIN))
        cv2.rectangle(frame, (x1, y1), (x2, y2), _YELLOW, 2)
        cv2.putText(frame, "ACTIVE ZONE", (x1 + 5, y1 - 8),
                    _FONT, _FONT_SMALL, _YELLOW, 1)

    def _draw_status_panel(
        self, frame, state: GestureState, confidence: float,
        fps: float, sx: int, sy: int, w: int
    ) -> None:
        """Semi-transparent status bar across the top."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), _BLACK, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        color = STATE_COLORS.get(state.name.lower(), _WHITE)

        # State label
        cv2.putText(frame, f"STATE: {state.label}", (10, 28),
                    _FONT, _FONT_LARGE, color, 2)

        # Confidence bar
        bar_x, bar_y, bar_w, bar_h = 10, 38, 200, 14
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _GREY, -1)
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + int(bar_w * confidence), bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{confidence:.0%}", (bar_x + bar_w + 5, bar_y + bar_h),
                    _FONT, _FONT_SMALL, _WHITE, 1)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 22),
                    _FONT, _FONT_SMALL, _WHITE, 1)

        # Screen target
        cv2.putText(frame, f"({sx}, {sy})", (w - 120, 50),
                    _FONT, _FONT_SMALL, _GREY, 1)

    def _draw_vote_history(self, frame, history: list, w: int, h: int) -> None:
        """Compact vote queue dots at the bottom of the frame."""
        if not history:
            return
        dot_r, gap = 8, 5
        total_w    = len(history) * (dot_r * 2 + gap)
        start_x    = (w - total_w) // 2
        y          = h - 18
        for i, vote in enumerate(history):
            cx    = start_x + i * (dot_r * 2 + gap) + dot_r
            color = STATE_COLORS.get(vote.name.lower(), _GREY)
            cv2.circle(frame, (cx, y), dot_r, color, -1)
            cv2.circle(frame, (cx, y), dot_r, _WHITE, 1)

    def _draw_voice_indicator(self, frame, is_recording: bool, w: int) -> None:
        """Pulsing red mic dot (top-right) when voice is being captured."""
        cx, cy = w - 30, 110
        if is_recording:
            pulse = int(abs(np.sin(time.monotonic() * 5)) * 6)
            cv2.circle(frame, (cx, cy), 10 + pulse, (0, 0, 220), -1)
            cv2.putText(frame, "REC", (cx - 20, cy + 25),
                        _FONT, _FONT_SMALL, (0, 0, 220), 1)
        else:
            cv2.circle(frame, (cx, cy), 10, _GREY, 2)

    def _draw_voice_feedback(self, frame, w: int, h: int) -> None:
        """Show last voice transcript for VOICE_DISPLAY_SECONDS with fade."""
        if not self._voice_text:
            return
        remaining = self._voice_expire_ts - time.monotonic()
        if remaining <= 0:
            return
        alpha   = min(1.0, remaining / VOICE_DISPLAY_SECONDS)
        color   = _GREEN if self._voice_routed else _RED
        label   = f"\"{self._voice_text}\""
        cv2.putText(frame, label, (10, h - 40),
                    _FONT, _FONT_MED, tuple(int(c * alpha) for c in color), 1)
