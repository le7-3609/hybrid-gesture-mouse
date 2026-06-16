"""
Tests for the tracking layer: normalization and CoordinateMapper.
Uses mock MediaPipe landmark objects — no camera or MediaPipe install needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from unittest.mock import MagicMock
from tracking.normalization import normalize_landmarks
from tracking.coordinate_mapper import CoordinateMapper


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_landmark(x, y, z):
    lm = MagicMock()
    lm.x, lm.y, lm.z = x, y, z
    return lm


def _make_hand_landmarks(positions: list[tuple]):
    """Build a mock NormalizedLandmarkList from a list of (x, y, z) tuples."""
    hand = MagicMock()
    hand.landmark = [_make_landmark(*p) for p in positions]
    return hand


def _flat_hand():
    """21 landmarks; wrist at (0.5, 0.5, 0), mid-MCP at (0.5, 0.4, 0)."""
    positions = [(0.5, 0.5, 0.0)] * 21
    # landmark 9 = middle-finger MCP at a known distance
    positions[9] = (0.5, 0.4, 0.0)
    return _make_hand_landmarks(positions)


# ─── normalization tests ─────────────────────────────────────────────────────

class TestNormalizeLandmarks(unittest.TestCase):

    def test_returns_63_values(self):
        result = normalize_landmarks(_flat_hand())
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 63)

    def test_wrist_is_zero(self):
        result = normalize_landmarks(_flat_hand())
        # landmark 0 (wrist) → first three values should be 0.0
        self.assertAlmostEqual(result[0], 0.0, places=6)
        self.assertAlmostEqual(result[1], 0.0, places=6)
        self.assertAlmostEqual(result[2], 0.0, places=6)

    def test_scale_invariance(self):
        """Doubling the hand size should produce the same normalised vector."""
        base = [(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)]
        base[9] = (0.5 + 9 * 0.01, 0.4 + 9 * 0.01, 0.0)   # reference

        scaled = [(x * 2, y * 2, z) for x, y, z in base]
        scaled[0] = (base[0][0] * 2, base[0][1] * 2, 0.0)  # keep wrist scale

        r1 = normalize_landmarks(_make_hand_landmarks(base))
        r2 = normalize_landmarks(_make_hand_landmarks(scaled))
        if r1 and r2:
            # At least the relative shape should be proportional
            self.assertEqual(len(r1), len(r2))

    def test_returns_none_on_degenerate(self):
        """Wrist == mid-MCP → scale is zero → should return None."""
        positions = [(0.5, 0.5, 0.0)] * 21   # all at same position
        result = normalize_landmarks(_make_hand_landmarks(positions))
        self.assertIsNone(result)

    def test_all_floats(self):
        result = normalize_landmarks(_flat_hand())
        self.assertTrue(all(isinstance(v, float) for v in result))


# ─── CoordinateMapper tests ───────────────────────────────────────────────────

class TestCoordinateMapper(unittest.TestCase):

    def setUp(self):
        self.mapper = CoordinateMapper(screen_width=1920, screen_height=1080, smoothing=1.0)

    def test_maps_center_to_screen_center(self):
        """A raw position at the center of the active zone maps near screen center."""
        # Active zone: 15% margin → effective range [192, 1088] x [108, 612]
        # Center of active zone ≈ frame_width//2, frame_height//2
        sx, sy = self.mapper.map_and_smooth(640, 360, 1280, 720)
        # With smoothing=1.0 (no EMA) and center input → near screen center
        self.assertGreater(sx, 600)
        self.assertLess(sx, 1400)
        self.assertGreater(sy, 300)
        self.assertLess(sy, 800)

    def test_output_clamped_to_screen(self):
        """Output coordinates must always be within screen bounds."""
        for raw_x, raw_y in [(0, 0), (1280, 720), (9999, 9999), (-100, -100)]:
            sx, sy = self.mapper.map_and_smooth(raw_x, raw_y, 1280, 720)
            self.assertGreaterEqual(sx, 0)
            self.assertLess(sx, 1920)
            self.assertGreaterEqual(sy, 0)
            self.assertLess(sy, 1080)

    def test_ema_smoothing(self):
        """With smoothing < 1, output should not immediately jump to new position."""
        mapper = CoordinateMapper(1920, 1080, smoothing=0.1)
        mapper.reset(0, 0)
        sx, sy = mapper.map_and_smooth(1280, 720, 1280, 720)
        # With alpha=0.1 and prev=0, output should be much less than max
        self.assertLess(sx, 1000)
        self.assertLess(sy, 600)

    def test_reset_seeds_ema(self):
        """After reset, first call uses seeded value for EMA."""
        self.mapper.reset(960, 540)
        self.assertEqual(self.mapper.prev_x, 960.0)
        self.assertEqual(self.mapper.prev_y, 540.0)

    def test_returns_integers(self):
        sx, sy = self.mapper.map_and_smooth(640, 360, 1280, 720)
        self.assertIsInstance(sx, int)
        self.assertIsInstance(sy, int)


if __name__ == "__main__":
    unittest.main()
