"""
factory — OS-detection factory for mouse service creation.

The factory is the only place in the codebase that checks platform.
All other modules depend on the BaseMouseService interface.
"""
from __future__ import annotations
import sys
from interfaces.mouse_service import BaseMouseService


def create_mouse_service() -> BaseMouseService:
    """
    Detect the current OS and return the appropriate concrete mouse service.

    Returns
    -------
    BaseMouseService
        WindowsMouseService on Windows, MacMouseService on macOS/Linux.
    """
    if sys.platform == "win32":
        from services.windows_service import WindowsMouseService
        return WindowsMouseService()
    else:
        from services.mac_service import MacMouseService
        return MacMouseService()
