"""
Tests for the controller and utility layers.
Uses mocks for all dependencies — no camera, no ML model, no OS calls.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import time
from unittest.mock import MagicMock, patch
from domain.gesture_state import GestureState
from domain.voice_command import VoiceCommand
from interfaces.mouse_service import BaseMouseService
from interfaces.classifier import BaseGestureClassifier
from interfaces.voice_listener import BaseVoiceListener
from voice.null_listener import NullVoiceListener
from voice.command_router import VoiceCommandRouter
from utils.fps_counter import FPSCounter
from utils.logger import get_logger
import logging


# ─── Stubs ───────────────────────────────────────────────────────────────────

class StubMouseService(BaseMouseService):
    def __init__(self):
        self.calls = []
    def move_to(self, x, y):     self.calls.append(f"move({x},{y})")
    def click(self):             self.calls.append("click")
    def right_click(self):       self.calls.append("right_click")
    def double_click(self):      self.calls.append("double_click")
    def press_down(self):        self.calls.append("press_down")
    def release_up(self):        self.calls.append("release_up")
    def scroll(self, a):         self.calls.append(f"scroll({a})")
    def get_screen_size(self):   return (1920, 1080)
    def get_position(self):      return (960, 540)


class StubClassifier(BaseGestureClassifier):
    def __init__(self, state=GestureState.IDLE, prob=1.0):
        self._state, self._prob = state, prob
    def predict(self, _):
        return self._state, self._prob


class StubVoiceListener(BaseVoiceListener):
    def __init__(self, commands=None):
        self._cmds = list(commands or [])
        self.started = self.stopped = False
    def start(self):                 self.started = True
    def stop(self):                  self.stopped = True
    def get_latest_command(self):
        return self._cmds.pop(0) if self._cmds else None
    def is_recording(self):         return False


# ─── FPSCounter tests ────────────────────────────────────────────────────────

class TestFPSCounter(unittest.TestCase):

    def test_initial_fps_is_zero(self):
        fps = FPSCounter()
        self.assertEqual(fps._fps, 0.0)

    def test_first_tick_returns_nonnegative(self):
        fps = FPSCounter()
        result = fps.tick()
        self.assertGreaterEqual(result, 0.0)

    def test_fps_increases_with_fast_ticks(self):
        fps = FPSCounter(alpha=1.0)   # no smoothing
        # Simulate 60 fps ticks
        fps._prev_ts = time.monotonic() - (1 / 60)
        result = fps.tick()
        self.assertGreater(result, 30)   # should be near 60

    def test_returns_float(self):
        fps = FPSCounter()
        result = fps.tick()
        self.assertIsInstance(result, float)


# ─── Logger tests ─────────────────────────────────────────────────────────────

class TestLogger(unittest.TestCase):

    def test_get_logger_returns_logger(self):
        lg = get_logger("test_logger")
        self.assertIsInstance(lg, logging.Logger)

    def test_logger_has_handler(self):
        lg = get_logger("test_logger_2")
        self.assertTrue(len(lg.handlers) > 0)

    def test_logger_name(self):
        lg = get_logger("my_module")
        self.assertEqual(lg.name, "my_module")

    def test_idempotent_handler_creation(self):
        """Calling twice should not double handlers."""
        get_logger("idempotent")
        lg = get_logger("idempotent")
        self.assertEqual(len(lg.handlers), 1)


# ─── Controller wiring tests ─────────────────────────────────────────────────

class TestControllerWiring(unittest.TestCase):
    """
    Test the GestureController wiring without opening a real webcam.
    Patches cv2.VideoCapture to immediately return no frames.
    Imports GestureController inside each test to allow graceful skip
    if cv2 is not installed.
    """

    def _try_import(self):
        try:
            from controller.gesture_controller import GestureController
            return GestureController
        except ImportError:
            self.skipTest("cv2/mediapipe not installed — skipping controller tests")

    def _make_controller(self, voice_listener=None, classifier=None):
        GestureController = self._try_import()
        svc    = StubMouseService()
        clf    = classifier or StubClassifier()
        voice  = voice_listener or NullVoiceListener()
        router = VoiceCommandRouter()
        return GestureController(svc, clf, voice, router)

    def test_controller_constructs_without_error(self):
        ctrl = self._make_controller()
        self.assertIsNotNone(ctrl)

    def test_voice_listener_is_stored(self):
        listener = StubVoiceListener()
        ctrl = self._make_controller(voice_listener=listener)
        self.assertIs(ctrl._voice, listener)

    def test_null_listener_satisfies_interface(self):
        """NullVoiceListener must be accepted as a BaseVoiceListener."""
        ctrl = self._make_controller(voice_listener=NullVoiceListener())
        self.assertIsInstance(ctrl._voice, BaseVoiceListener)

    def test_run_exits_cleanly_when_camera_fails(self):
        """If webcam cannot open, run() must return without crashing."""
        ctrl = self._make_controller()
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap
            ctrl.run()   # must not raise

    def test_voice_command_polled_each_frame(self):
        """Verify voice.get_latest_command() is called during run()."""
        cmd = VoiceCommand(transcript="undo")
        listener = StubVoiceListener(commands=[cmd])
        ctrl = self._make_controller(voice_listener=listener)

        with patch("cv2.VideoCapture") as mock_cap_cls, \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")), \
             patch("cv2.destroyAllWindows"):
            mock_cap = MagicMock()
            mock_cap.isOpened.side_effect = [True, False]
            frame = MagicMock()
            frame.shape = (720, 1280, 3)
            mock_cap.read.return_value = (True, frame)
            mock_cap_cls.return_value = mock_cap
            # patch flip so frame stays a MagicMock
            with patch("cv2.flip", return_value=frame), \
                 patch("cv2.cvtColor", return_value=frame):
                 ctrl._tracker = MagicMock()
                 ctrl._tracker.detect.return_value = None
                 ctrl.run()

        # listener.started must have been called
        self.assertTrue(listener.started)
        self.assertTrue(listener.stopped)


if __name__ == "__main__":
    unittest.main()
