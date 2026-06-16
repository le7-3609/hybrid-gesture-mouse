"""
MacMouseService — PyAutoGUI-based implementation for macOS / generic platforms.

Falls back to pyautogui which works on macOS and Linux.
Used when the OS is not Windows.
"""
from __future__ import annotations
import pyautogui
from interfaces.mouse_service import BaseMouseService

# Disable pyautogui fail-safe for this service — the state machine owns failsafe logic
pyautogui.PAUSE = 0


class MacMouseService(BaseMouseService):
    """PyAutoGUI-backed mouse service for macOS and Linux."""

    def move_to(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y, duration=0)

    def click(self) -> None:
        pyautogui.click()

    def right_click(self) -> None:
        pyautogui.rightClick()

    def double_click(self) -> None:
        pyautogui.doubleClick()

    def press_down(self) -> None:
        pyautogui.mouseDown()

    def release_up(self) -> None:
        pyautogui.mouseUp()

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount)

    def get_screen_size(self) -> tuple[int, int]:
        w, h = pyautogui.size()
        return int(w), int(h)

    def get_position(self) -> tuple[int, int]:
        x, y = pyautogui.position()
        return int(x), int(y)
