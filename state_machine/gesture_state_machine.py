"""
GestureStateMachine — translates stabilised GestureState values into
concrete OS mouse actions via the injected BaseMouseService.

Extends the original 6-state machine to support 8 states:
  RIGHT_CLICK  — debounced secondary click
  DOUBLE_CLICK — debounced double-click

Design notes
------------
* All state is held inside this class — the controller loop is stateless.
* Transitions release drag safely when a conflicting state arrives.
* Fail-safe: raises FailSafeException when the cursor reaches a screen corner.
* The class depends on BaseMouseService (injected) — never on a concrete impl.
"""
from __future__ import annotations
import time
import pyautogui
from domain.gesture_state import GestureState
from interfaces.mouse_service import BaseMouseService


class GestureStateMachine:
    """
    Manages OS-level mouse state, transitions, and fail-safe triggers.

    Parameters
    ----------
    mouse_service : BaseMouseService
        Injected OS mouse service (Windows, Mac, or Mock).
    debounce : float
        Minimum seconds between successive click events.
    scroll_sens : float
        Multiplier applied to each scroll tick.
    scroll_step : int
        Base scroll unit (number of lines per tick).
    """

    def __init__(
        self,
        mouse_service: BaseMouseService,
        debounce: float = 0.4,
        scroll_sens: float = 1.5,
        scroll_step: int = 2,
    ) -> None:
        self._svc          = mouse_service
        self._debounce     = debounce
        self._scroll_sens  = scroll_sens
        self._scroll_step  = scroll_step

        # Internal state
        self._is_dragging    = False
        self._last_click_ts  = 0.0
        self._click_latched  = False   # prevents hold-gesture multi-fire

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, state: GestureState, x: int, y: int) -> None:
        """
        Execute the action corresponding to the stabilised state.

        Parameters
        ----------
        state : GestureState
            Current stabilised gesture.
        x, y : int
            Target screen coordinates.

        Raises
        ------
        pyautogui.FailSafeException
            When the cursor reaches a screen corner (emergency stop).
        """
        sw, sh = self._svc.get_screen_size()
        if x <= 1 or y <= 1 or x >= sw - 2 or y >= sh - 2:
            raise pyautogui.FailSafeException(
                "FailSafe: cursor reached screen corner — shutting down safely."
            )

        now = time.monotonic()

        # ── Reset click latch when leaving a click-type state ──────────
        if state not in (GestureState.CLICK, GestureState.RIGHT_CLICK,
                         GestureState.DOUBLE_CLICK):
            self._click_latched = False

        # ── State dispatch ────────────────────────────────────────────
        if state == GestureState.IDLE:
            self._safe_release_drag("IDLE")

        elif state == GestureState.MOVE:
            self._safe_release_drag("MOVE")
            self._svc.move_to(x, y)

        elif state == GestureState.CLICK:
            self._safe_release_drag("CLICK")
            if not self._click_latched and self._debounce_ok(now):
                self._svc.move_to(x, y)
                self._svc.click()
                self._last_click_ts = now
                self._click_latched = True

        elif state == GestureState.DRAG:
            if not self._is_dragging:
                self._svc.move_to(x, y)
                self._svc.press_down()
                self._is_dragging = True
            else:
                self._svc.move_to(x, y)

        elif state == GestureState.SCROLL_UP:
            self._safe_release_drag("SCROLL_UP")
            self._svc.scroll(int(self._scroll_step * self._scroll_sens))

        elif state == GestureState.SCROLL_DOWN:
            self._safe_release_drag("SCROLL_DOWN")
            self._svc.scroll(-int(self._scroll_step * self._scroll_sens))

        elif state == GestureState.RIGHT_CLICK:
            self._safe_release_drag("RIGHT_CLICK")
            if not self._click_latched and self._debounce_ok(now):
                self._svc.move_to(x, y)
                self._svc.right_click()
                self._last_click_ts = now
                self._click_latched = True

        elif state == GestureState.DOUBLE_CLICK:
            self._safe_release_drag("DOUBLE_CLICK")
            if not self._click_latched and self._debounce_ok(now):
                self._svc.move_to(x, y)
                self._svc.double_click()
                self._last_click_ts = now
                self._click_latched = True

    def shutdown(self) -> None:
        """Ensure all buttons are released before the application exits."""
        if self._is_dragging:
            try:
                self._svc.release_up()
            except Exception:
                pass
            self._is_dragging = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_release_drag(self, incoming_state_name: str) -> None:
        """Release an active drag if any other state takes over."""
        if self._is_dragging:
            self._svc.release_up()
            self._is_dragging = False

    def _debounce_ok(self, now: float) -> bool:
        return (now - self._last_click_ts) >= self._debounce
