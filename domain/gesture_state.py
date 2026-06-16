"""
GestureState — domain enum that replaces all magic integers.

Extended from the original 6 states to 8, adding:
  RIGHT_CLICK  (6) — pinky-only extended
  DOUBLE_CLICK (7) — rapid repeated pinch
"""
from enum import IntEnum


class GestureState(IntEnum):
    IDLE         = 0
    MOVE         = 1
    CLICK        = 2
    DRAG         = 3
    SCROLL_UP    = 4
    SCROLL_DOWN  = 5
    RIGHT_CLICK  = 6
    DOUBLE_CLICK = 7

    @property
    def label(self) -> str:
        """Human-readable label used in HUD display."""
        return self.name.replace("_", " ").title()
