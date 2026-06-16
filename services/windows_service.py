"""
WindowsMouseService — Win32 ctypes implementation of BaseMouseService.

Uses direct Win32 API calls via ctypes for zero-dependency, zero-latency
hardware mouse events on Windows.  No pyautogui overhead for move/click —
pyautogui is used only for scroll (SendInput scroll is niche to implement).

New in Gesture+:
  right_click()  — MOUSEEVENTF_RIGHTDOWN + MOUSEEVENTF_RIGHTUP
  double_click() — two rapid MOUSEEVENTF_LEFTDOWN/UP pairs
"""
from __future__ import annotations
import ctypes
import time
import pyautogui
from interfaces.mouse_service import BaseMouseService

# Win32 mouse event flags
_MOUSEEVENTF_MOVE      = 0x0001
_MOUSEEVENTF_LEFTDOWN  = 0x0002
_MOUSEEVENTF_LEFTUP    = 0x0004
_MOUSEEVENTF_RIGHTDOWN = 0x0008
_MOUSEEVENTF_RIGHTUP   = 0x0010
_MOUSEEVENTF_ABSOLUTE  = 0x8000

# Normalised coordinate space for Win32 SetCursorPos
_WIN32_NORM = 65535


def _send_mouse_event(flags: int, dx: int = 0, dy: int = 0) -> None:
    ctypes.windll.user32.mouse_event(flags, dx, dy, 0, 0)


class WindowsMouseService(BaseMouseService):
    """
    Windows-only, direct Win32 mouse service.
    Offers the lowest latency path on Windows without any extra dependencies.
    """

    def move_to(self, x: int, y: int) -> None:
        ctypes.windll.user32.SetCursorPos(x, y)

    def click(self) -> None:
        _send_mouse_event(_MOUSEEVENTF_LEFTDOWN)
        _send_mouse_event(_MOUSEEVENTF_LEFTUP)

    def right_click(self) -> None:
        _send_mouse_event(_MOUSEEVENTF_RIGHTDOWN)
        _send_mouse_event(_MOUSEEVENTF_RIGHTUP)

    def double_click(self) -> None:
        _send_mouse_event(_MOUSEEVENTF_LEFTDOWN)
        _send_mouse_event(_MOUSEEVENTF_LEFTUP)
        time.sleep(0.05)
        _send_mouse_event(_MOUSEEVENTF_LEFTDOWN)
        _send_mouse_event(_MOUSEEVENTF_LEFTUP)

    def press_down(self) -> None:
        _send_mouse_event(_MOUSEEVENTF_LEFTDOWN)

    def release_up(self) -> None:
        _send_mouse_event(_MOUSEEVENTF_LEFTUP)

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount)

    def get_screen_size(self) -> tuple[int, int]:
        w = ctypes.windll.user32.GetSystemMetrics(0)
        h = ctypes.windll.user32.GetSystemMetrics(1)
        return int(w), int(h)

    def get_position(self) -> tuple[int, int]:
        pt = ctypes.wintypes.POINT()
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        return int(pt.x), int(pt.y)
