"""
Tests for the classification layer.
Uses a synthetic sklearn model — no trained .pkl file required.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
import pickle
import tempfile
import numpy as np
from unittest.mock import MagicMock
from domain.gesture_state import GestureState
from classification.gesture_classifier import GestureClassifier
from classification.vote_stabilizer import VoteStabilizer
from classification.model_loader import load_pickle_model


# ─── GestureClassifier tests ─────────────────────────────────────────────────

def _make_mock_model(predicted_idx: int, num_classes: int = 8):
    """Return a mock sklearn estimator that always predicts predicted_idx."""
    probs = [0.0] * num_classes
    probs[predicted_idx] = 1.0
    model = MagicMock()
    model.predict_proba.return_value = np.array([probs])
    return model


class TestGestureClassifier(unittest.TestCase):

    def test_predict_returns_correct_state(self):
        for idx in range(8):
            with self.subTest(idx=idx):
                clf = GestureClassifier(_make_mock_model(idx))
                state, prob = clf.predict([0.0] * 63)
                self.assertEqual(state, GestureState(idx))

    def test_predict_returns_float_confidence(self):
        clf = GestureClassifier(_make_mock_model(0))
        _, prob = clf.predict([0.0] * 63)
        self.assertIsInstance(prob, float)

    def test_predict_confidence_between_0_and_1(self):
        clf = GestureClassifier(_make_mock_model(2))
        _, prob = clf.predict([0.0] * 63)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_predict_right_click_state(self):
        clf = GestureClassifier(_make_mock_model(6))
        state, prob = clf.predict([0.0] * 63)
        self.assertEqual(state, GestureState.RIGHT_CLICK)
        self.assertAlmostEqual(prob, 1.0)

    def test_predict_double_click_state(self):
        clf = GestureClassifier(_make_mock_model(7))
        state, _ = clf.predict([0.0] * 63)
        self.assertEqual(state, GestureState.DOUBLE_CLICK)


# ─── VoteStabilizer tests ─────────────────────────────────────────────────────

class TestVoteStabilizer(unittest.TestCase):

    def test_empty_returns_idle(self):
        vs = VoteStabilizer()
        self.assertEqual(vs.get_stabilized_state(), GestureState.IDLE)

    def test_majority_wins(self):
        vs = VoteStabilizer(window_size=5)
        for _ in range(3):
            vs.add_vote(GestureState.MOVE)
        for _ in range(2):
            vs.add_vote(GestureState.CLICK)
        self.assertEqual(vs.get_stabilized_state(), GestureState.MOVE)

    def test_window_evicts_old_votes(self):
        vs = VoteStabilizer(window_size=3)
        vs.add_vote(GestureState.CLICK)
        vs.add_vote(GestureState.CLICK)
        vs.add_vote(GestureState.MOVE)
        vs.add_vote(GestureState.MOVE)   # evicts first CLICK
        vs.add_vote(GestureState.MOVE)   # evicts second CLICK → all MOVE
        self.assertEqual(vs.get_stabilized_state(), GestureState.MOVE)

    def test_clear_resets_to_idle(self):
        vs = VoteStabilizer()
        vs.add_vote(GestureState.DRAG)
        vs.clear()
        self.assertEqual(vs.get_stabilized_state(), GestureState.IDLE)

    def test_get_history_length(self):
        vs = VoteStabilizer(window_size=5)
        for state in [GestureState.IDLE, GestureState.MOVE, GestureState.CLICK]:
            vs.add_vote(state)
        self.assertEqual(len(vs.get_history()), 3)

    def test_right_click_stabilizes(self):
        vs = VoteStabilizer(window_size=5)
        for _ in range(4):
            vs.add_vote(GestureState.RIGHT_CLICK)
        vs.add_vote(GestureState.IDLE)
        self.assertEqual(vs.get_stabilized_state(), GestureState.RIGHT_CLICK)


# ─── model_loader tests ───────────────────────────────────────────────────────

class TestModelLoader(unittest.TestCase):

    def test_loads_valid_pickle(self):
        model_obj = {"key": "test_model"}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model_obj, f)
            path = f.name
        try:
            loaded = load_pickle_model(path)
            self.assertEqual(loaded, model_obj)
        finally:
            os.unlink(path)

    def test_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_pickle_model("nonexistent_model.pkl")

    def test_raises_value_error_on_corrupt_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            f.write(b"not_a_pickle")
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pickle_model(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
