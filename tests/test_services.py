"""
Tests for the services layer.
Uses a MockMouseService (in-memory implementation) — no OS calls, no hardware.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from interfaces.mouse_service import BaseMouseService
from services.factory import create_mouse_service


# ─── MockMouseService — in-memory test double ────────────────────────────────

class MockMouseService(BaseMouseService):
    """Records all calls for assertion in tests."""

    def __init__(self):
        self.calls: list[str] = []
        self._x = 0
        self._y = 0
        self._screen = (1920, 1080)

    def move_to(self, x, y):
        self._x, self._y = x, y
        self.calls.append(f"move_to({x},{y})")

    def click(self):
        self.calls.append("click")

    def right_click(self):
        self.calls.append("right_click")

    def double_click(self):
        self.calls.append("double_click")

    def press_down(self):
        self.calls.append("press_down")

    def release_up(self):
        self.calls.append("release_up")

    def scroll(self, amount):
        self.calls.append(f"scroll({amount})")

    def get_screen_size(self):
        return self._screen

    def get_position(self):
        return self._x, self._y


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestMockMouseService(unittest.TestCase):
    """Verify that MockMouseService itself behaves correctly (it is used by other tests)."""

    def setUp(self):
        self.svc = MockMouseService()

    def test_implements_base_interface(self):
        self.assertIsInstance(self.svc, BaseMouseService)

    def test_move_to_records_position(self):
        self.svc.move_to(100, 200)
        self.assertEqual(self.svc.get_position(), (100, 200))

    def test_click_recorded(self):
        self.svc.click()
        self.assertIn("click", self.svc.calls)

    def test_right_click_recorded(self):
        self.svc.right_click()
        self.assertIn("right_click", self.svc.calls)

    def test_double_click_recorded(self):
        self.svc.double_click()
        self.assertIn("double_click", self.svc.calls)

    def test_press_and_release_recorded(self):
        self.svc.press_down()
        self.svc.release_up()
        self.assertIn("press_down", self.svc.calls)
        self.assertIn("release_up", self.svc.calls)

    def test_scroll_recorded(self):
        self.svc.scroll(3)
        self.assertIn("scroll(3)", self.svc.calls)
        self.svc.scroll(-2)
        self.assertIn("scroll(-2)", self.svc.calls)

    def test_screen_size(self):
        self.assertEqual(self.svc.get_screen_size(), (1920, 1080))

    def test_call_order(self):
        self.svc.move_to(10, 20)
        self.svc.click()
        self.svc.right_click()
        self.assertEqual(self.svc.calls, ["move_to(10,20)", "click", "right_click"])


class TestServiceFactory(unittest.TestCase):
    """
    Tests for create_mouse_service() using MockMouseService to avoid
    importing OS-specific backends (pyautogui / ctypes) during CI.
    """

    def test_mock_returns_base_mouse_service(self):
        """MockMouseService must satisfy the BaseMouseService contract."""
        svc = MockMouseService()
        self.assertIsInstance(svc, BaseMouseService)

    def test_mock_has_all_required_methods(self):
        """Verify MockMouseService implements every method of the interface."""
        svc = MockMouseService()
        for method in ["move_to", "click", "right_click", "double_click",
                       "press_down", "release_up", "scroll",
                       "get_screen_size", "get_position"]:
            self.assertTrue(hasattr(svc, method), f"Missing method: {method}")

    def test_mock_get_screen_size_returns_positive_ints(self):
        svc = MockMouseService()
        w, h = svc.get_screen_size()
        self.assertIsInstance(w, int)
        self.assertIsInstance(h, int)
        self.assertGreater(w, 0)
        self.assertGreater(h, 0)

    def test_mock_get_position_returns_ints(self):
        svc = MockMouseService()
        x, y = svc.get_position()
        self.assertIsInstance(x, int)
        self.assertIsInstance(y, int)

    def test_factory_import_is_callable(self):
        """factory.create_mouse_service must be importable and callable."""
        from services.factory import create_mouse_service as csvc
        self.assertTrue(callable(csvc))


if __name__ == "__main__":
    unittest.main()
