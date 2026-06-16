"""
Tests for domain layer: GestureState enum and VoiceCommand dataclass.
No external dependencies required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import time
from domain.gesture_state import GestureState
from domain.voice_command import VoiceCommand


class TestGestureState(unittest.TestCase):

    def test_all_eight_states_exist(self):
        states = list(GestureState)
        self.assertEqual(len(states), 8)

    def test_state_values(self):
        self.assertEqual(GestureState.IDLE,         0)
        self.assertEqual(GestureState.MOVE,         1)
        self.assertEqual(GestureState.CLICK,        2)
        self.assertEqual(GestureState.DRAG,         3)
        self.assertEqual(GestureState.SCROLL_UP,    4)
        self.assertEqual(GestureState.SCROLL_DOWN,  5)
        self.assertEqual(GestureState.RIGHT_CLICK,  6)
        self.assertEqual(GestureState.DOUBLE_CLICK, 7)

    def test_label_property(self):
        self.assertEqual(GestureState.RIGHT_CLICK.label,  "Right Click")
        self.assertEqual(GestureState.DOUBLE_CLICK.label, "Double Click")
        self.assertEqual(GestureState.IDLE.label,          "Idle")

    def test_int_comparison(self):
        self.assertTrue(GestureState.MOVE > GestureState.IDLE)
        self.assertFalse(GestureState.CLICK == GestureState.DRAG)

    def test_from_int(self):
        self.assertEqual(GestureState(6), GestureState.RIGHT_CLICK)
        self.assertEqual(GestureState(7), GestureState.DOUBLE_CLICK)


class TestVoiceCommand(unittest.TestCase):

    def test_creation_with_defaults(self):
        cmd = VoiceCommand(transcript="copy")
        self.assertEqual(cmd.transcript, "copy")
        self.assertEqual(cmd.confidence, 1.0)
        self.assertAlmostEqual(cmd.timestamp, time.time(), delta=1.0)

    def test_creation_with_all_fields(self):
        ts = 1_000_000.0
        cmd = VoiceCommand(transcript="paste", confidence=0.92, timestamp=ts)
        self.assertEqual(cmd.transcript, "paste")
        self.assertEqual(cmd.confidence, 0.92)
        self.assertEqual(cmd.timestamp,  ts)

    def test_immutability(self):
        cmd = VoiceCommand(transcript="undo")
        with self.assertRaises(Exception):
            cmd.transcript = "redo"  # type: ignore[misc]

    def test_str_representation(self):
        cmd = VoiceCommand(transcript="save", confidence=0.85)
        s = str(cmd)
        self.assertIn("save", s)
        self.assertIn("0.85", s)


if __name__ == "__main__":
    unittest.main()
