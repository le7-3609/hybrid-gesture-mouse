"""
Tests for GestureStateMachine — verifies all 8 state transitions using
a MockMouseService (no OS calls, no hardware required).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import time
import pyautogui
from domain.gesture_state import GestureState
from interfaces.mouse_service import BaseMouseService
from state_machine.gesture_state_machine import GestureStateMachine


# ─── Re-use MockMouseService from test_services ───────────────────────────────

class MockMouseService(BaseMouseService):
    def __init__(self, screen=(1920, 1080)):
        self.calls: list[str] = []
        self._x = 500
        self._y = 500
        self._screen = screen

    def move_to(self, x, y):
        self._x, self._y = x, y
        self.calls.append(f"move({x},{y})")

    def click(self):           self.calls.append("click")
    def right_click(self):     self.calls.append("right_click")
    def double_click(self):    self.calls.append("double_click")
    def press_down(self):      self.calls.append("press_down")
    def release_up(self):      self.calls.append("release_up")
    def scroll(self, amount):  self.calls.append(f"scroll({amount})")
    def get_screen_size(self): return self._screen
    def get_position(self):    return self._x, self._y


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _machine(debounce=0.0) -> tuple[GestureStateMachine, MockMouseService]:
    svc = MockMouseService()
    sm  = GestureStateMachine(svc, debounce=debounce)
    return sm, svc


MID = (960, 540)   # safe screen centre position


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestIdleState(unittest.TestCase):
    def test_idle_does_not_move(self):
        sm, svc = _machine()
        sm.execute(GestureState.IDLE, *MID)
        self.assertNotIn("move(960,540)", svc.calls)

    def test_idle_releases_active_drag(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)       # start drag
        sm.execute(GestureState.IDLE, *MID)       # should release
        self.assertIn("release_up", svc.calls)
        self.assertFalse(sm._is_dragging)


class TestMoveState(unittest.TestCase):
    def test_move_calls_move_to(self):
        sm, svc = _machine()
        sm.execute(GestureState.MOVE, 100, 200)
        self.assertIn("move(100,200)", svc.calls)

    def test_move_releases_active_drag(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)
        sm.execute(GestureState.MOVE, *MID)
        self.assertIn("release_up", svc.calls)


class TestClickState(unittest.TestCase):
    def test_click_fires_once(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.CLICK, *MID)
        self.assertIn("click", svc.calls)

    def test_click_latches_on_hold(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.CLICK, *MID)
        sm.execute(GestureState.CLICK, *MID)   # held — should NOT fire again
        self.assertEqual(svc.calls.count("click"), 1)

    def test_click_debounce_blocks(self):
        sm, svc = _machine(debounce=10.0)   # 10s debounce
        sm.execute(GestureState.CLICK, *MID)
        sm._click_latched = False           # reset latch manually
        sm.execute(GestureState.CLICK, *MID)
        self.assertEqual(svc.calls.count("click"), 1)


class TestDragState(unittest.TestCase):
    def test_drag_press_down(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)
        self.assertIn("press_down", svc.calls)
        self.assertTrue(sm._is_dragging)

    def test_drag_continues_without_extra_press(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)
        sm.execute(GestureState.DRAG, 900, 500)
        self.assertEqual(svc.calls.count("press_down"), 1)

    def test_drag_releases_on_idle(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)
        sm.execute(GestureState.IDLE, *MID)
        self.assertIn("release_up", svc.calls)


class TestScrollStates(unittest.TestCase):
    def test_scroll_up(self):
        sm, svc = _machine()
        sm.execute(GestureState.SCROLL_UP, *MID)
        self.assertTrue(any(c.startswith("scroll(") and not c.startswith("scroll(-") for c in svc.calls))

    def test_scroll_down(self):
        sm, svc = _machine()
        sm.execute(GestureState.SCROLL_DOWN, *MID)
        self.assertTrue(any(c.startswith("scroll(-") for c in svc.calls))


class TestRightClickState(unittest.TestCase):
    def test_right_click_fires(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.RIGHT_CLICK, *MID)
        self.assertIn("right_click", svc.calls)

    def test_right_click_latches(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.RIGHT_CLICK, *MID)
        sm.execute(GestureState.RIGHT_CLICK, *MID)
        self.assertEqual(svc.calls.count("right_click"), 1)


class TestDoubleClickState(unittest.TestCase):
    def test_double_click_fires(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.DOUBLE_CLICK, *MID)
        self.assertIn("double_click", svc.calls)

    def test_double_click_latches(self):
        sm, svc = _machine(debounce=0.0)
        sm.execute(GestureState.DOUBLE_CLICK, *MID)
        sm.execute(GestureState.DOUBLE_CLICK, *MID)
        self.assertEqual(svc.calls.count("double_click"), 1)


class TestFailSafe(unittest.TestCase):
    def test_corner_top_left_raises(self):
        sm, _ = _machine()
        with self.assertRaises(pyautogui.FailSafeException):
            sm.execute(GestureState.MOVE, 0, 0)

    def test_corner_bottom_right_raises(self):
        sm, _ = _machine()
        with self.assertRaises(pyautogui.FailSafeException):
            sm.execute(GestureState.MOVE, 1919, 1079)

    def test_center_does_not_raise(self):
        sm, _ = _machine()
        sm.execute(GestureState.MOVE, *MID)   # must not raise


class TestShutdown(unittest.TestCase):
    def test_shutdown_releases_drag(self):
        sm, svc = _machine()
        sm.execute(GestureState.DRAG, *MID)
        sm.shutdown()
        self.assertIn("release_up", svc.calls)
        self.assertFalse(sm._is_dragging)

    def test_shutdown_safe_when_not_dragging(self):
        sm, svc = _machine()
        sm.shutdown()   # must not raise or call release_up
        self.assertNotIn("release_up", svc.calls)


if __name__ == "__main__":
    unittest.main()
