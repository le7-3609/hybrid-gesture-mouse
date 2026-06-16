"""
BaseMouseService — Abstract Base Class for OS-level mouse control.

Decouples the core ML / tracking engine from any concrete OS API.
Concrete implementations: WindowsMouseService, MacMouseService.
"""
from abc import ABC, abstractmethod


class BaseMouseService(ABC):

    @abstractmethod
    def move_to(self, x: int, y: int) -> None:
        """Move cursor to absolute screen coordinates (x, y)."""

    @abstractmethod
    def click(self) -> None:
        """Fire a primary (left) click at the current cursor position."""

    @abstractmethod
    def right_click(self) -> None:
        """Fire a secondary (right) click at the current cursor position."""

    @abstractmethod
    def double_click(self) -> None:
        """Fire two rapid left-clicks at the current cursor position."""

    @abstractmethod
    def press_down(self) -> None:
        """Hold the primary button down (start drag)."""

    @abstractmethod
    def release_up(self) -> None:
        """Release the primary button (end drag)."""

    @abstractmethod
    def scroll(self, amount: int) -> None:
        """Scroll vertically.  Positive = up, negative = down."""

    @abstractmethod
    def get_screen_size(self) -> tuple[int, int]:
        """Return (width, height) of the primary monitor in pixels."""

    @abstractmethod
    def get_position(self) -> tuple[int, int]:
        """Return current absolute cursor (x, y)."""
