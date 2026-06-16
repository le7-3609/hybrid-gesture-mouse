"""
Tests for the voice layer.
All tests avoid Whisper / sounddevice / microphone — they test the
interfaces, Null Object, and routing logic only.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import time
from domain.voice_command import VoiceCommand
from domain.gesture_state import GestureState
from interfaces.voice_listener import BaseVoiceListener
from voice.null_listener import NullVoiceListener


# ─── NullVoiceListener tests ─────────────────────────────────────────────────

class TestNullVoiceListener(unittest.TestCase):

    def setUp(self):
        self.listener = NullVoiceListener()

    def test_implements_base_interface(self):
        self.assertIsInstance(self.listener, BaseVoiceListener)

    def test_start_does_not_raise(self):
        self.listener.start()   # must not raise

    def test_stop_does_not_raise(self):
        self.listener.stop()    # must not raise

    def test_get_latest_returns_none(self):
        self.assertIsNone(self.listener.get_latest_command())

    def test_is_recording_always_false(self):
        self.assertFalse(self.listener.is_recording())

    def test_multiple_calls_safe(self):
        for _ in range(10):
            self.listener.start()
            self.listener.stop()
            self.assertIsNone(self.listener.get_latest_command())


# ─── VoiceCommandRouter tests (without OS side-effects) ──────────────────────

class MockRouter:
    """
    A VoiceCommandRouter with the pyautogui calls replaced by a call recorder.
    Allows testing routing logic without triggering real keyboard events.
    """

    # Reproduce the same COMMAND_MAP and logic, but record actions
    _COMMAND_MAP = {
        "copy":         ("ctrl", "c"),
        "paste":        ("ctrl", "v"),
        "undo":         ("ctrl", "z"),
        "redo":         ("ctrl", "y"),
        "save":         ("ctrl", "s"),
        "close":        ("alt",  "f4"),
        "new tab":      ("ctrl", "t"),
        "scroll up":    "pageup",
        "scroll down":  "pagedown",
        "screenshot":   ("win", "shift", "s"),
        "volume up":    "volumeup",
        "volume down":  "volumedown",
        "mute":         "volumemute",
        "select all":   ("ctrl", "a"),
    }

    def __init__(self):
        self.executed: list[tuple | str] = []

    def route(self, transcript: str) -> bool:
        text = transcript.strip().lower()
        matched_key = None
        for keyword in sorted(self._COMMAND_MAP, key=len, reverse=True):
            if keyword in text:
                matched_key = keyword
                break
        if matched_key is None:
            return False
        self.executed.append(self._COMMAND_MAP[matched_key])
        return True


class TestVoiceCommandRouter(unittest.TestCase):

    def setUp(self):
        self.router = MockRouter()

    def test_exact_keyword_matches(self):
        self.assertTrue(self.router.route("copy"))
        self.assertIn(("ctrl", "c"), self.router.executed)

    def test_keyword_in_sentence_matches(self):
        self.router.route("please undo my last change")
        self.assertIn(("ctrl", "z"), self.router.executed)

    def test_unknown_command_returns_false(self):
        result = self.router.route("xyzzy flibbertigibbet")
        self.assertFalse(result)

    def test_empty_transcript_returns_false(self):
        result = self.router.route("")
        self.assertFalse(result)

    def test_longest_match_wins(self):
        # "scroll down" should win over "scroll" if both existed
        self.router.route("scroll down please")
        self.assertIn("pagedown", self.router.executed)

    def test_scroll_up_matches(self):
        self.router.route("scroll up")
        self.assertIn("pageup", self.router.executed)

    def test_case_insensitive(self):
        self.router.route("COPY")
        self.assertIn(("ctrl", "c"), self.router.executed)

    def test_save_matches(self):
        self.router.route("save the file")
        self.assertIn(("ctrl", "s"), self.router.executed)

    def test_new_tab_matches(self):
        self.router.route("open new tab")
        self.assertIn(("ctrl", "t"), self.router.executed)

    def test_screenshot_matches(self):
        self.router.route("take a screenshot")
        self.assertIn(("win", "shift", "s"), self.router.executed)

    def test_mute_matches(self):
        self.router.route("mute please")
        self.assertIn("volumemute", self.router.executed)

    def test_select_all_matches(self):
        self.router.route("select all items")
        self.assertIn(("ctrl", "a"), self.router.executed)

    def test_multiple_commands_in_sequence(self):
        cmds = ["copy", "paste", "undo"]
        for cmd in cmds:
            self.router.route(cmd)
        self.assertEqual(len(self.router.executed), 3)


# ─── VoiceCommand integration with NullListener ───────────────────────────────

class TestVoiceCommandFlow(unittest.TestCase):

    def test_null_listener_never_produces_command(self):
        listener = NullVoiceListener()
        listener.start()
        for _ in range(5):
            cmd = listener.get_latest_command()
            self.assertIsNone(cmd)
        listener.stop()

    def test_voice_command_routing_with_mock(self):
        """Simulate a complete voice → route flow."""
        router = MockRouter()
        cmd = VoiceCommand(transcript="please save the document", confidence=0.95)
        matched = router.route(cmd.transcript)
        self.assertTrue(matched)
        self.assertIn(("ctrl", "s"), router.executed)


if __name__ == "__main__":
    unittest.main()
