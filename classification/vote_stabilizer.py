"""
VoteStabilizer — sliding-window majority-voting filter for gesture states.

Problem: the ML classifier may flicker between adjacent states on ambiguous
frames. A raw per-frame output makes the cursor jittery and clicks unreliable.

Solution: maintain a fixed-size deque of the last N predictions.
The stabilised state is the mode (most frequent element) of the window.
If there is a tie, the last entry wins.

Thread safety: not needed — always called from the single camera loop thread.
"""
from __future__ import annotations
from collections import deque
from domain.gesture_state import GestureState


class VoteStabilizer:
    """
    Sliding-window majority-voting stabiliser.

    Parameters
    ----------
    window_size : int
        Number of recent predictions to consider (default 7).
    """

    def __init__(self, window_size: int = 7) -> None:
        self._window_size = window_size
        self._history: deque[GestureState] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_vote(self, state: GestureState) -> None:
        """Push a new prediction into the sliding window."""
        self._history.append(state)

    def get_stabilized_state(self) -> GestureState:
        """
        Return the majority-vote winner of the current window.
        Falls back to GestureState.IDLE when the window is empty.
        """
        if not self._history:
            return GestureState.IDLE
        return max(set(self._history), key=self._history.count)

    def get_history(self) -> list[GestureState]:
        """Return a snapshot of the current window (newest last)."""
        return list(self._history)

    def clear(self) -> None:
        """Reset the window (e.g. when the hand leaves the frame)."""
        self._history.clear()
