"""
VoiceCommandRouter — maps raw Whisper transcripts to OS-level hotkey actions.

Design
------
* Keyword-to-action mapping is declared as a plain dict — easy to extend.
* Matching is fuzzy: checks if any keyword appears as a substring of the
  transcript, so "please copy this" still triggers "copy".
* Actions are pyautogui.hotkey / pyautogui.press calls.
* The router depends on nothing — it is a pure function object.

Injected into GestureController so the controller never knows about
hotkeys directly (Single Responsibility, Dependency Inversion).
"""
from __future__ import annotations
import pyautogui


# Command dictionary: keyword → (hotkey_args or press_key)
# Format:
#   str → tuple  = pyautogui.hotkey(*tuple)
#   str → str    = pyautogui.press(str)
_COMMAND_MAP: dict[str, tuple | str] = {
    # Editing
    "copy":         ("ctrl", "c"),
    "paste":        ("ctrl", "v"),
    "cut":          ("ctrl", "x"),
    "undo":         ("ctrl", "z"),
    "redo":         ("ctrl", "y"),
    "save":         ("ctrl", "s"),
    "select all":   ("ctrl", "a"),
    "find":         ("ctrl", "f"),

    # Window / app control
    "close":        ("alt",  "f4"),
    "new tab":      ("ctrl", "t"),
    "close tab":    ("ctrl", "w"),
    "switch tab":   ("ctrl", "tab"),
    "new window":   ("ctrl", "n"),
    "minimize":     ("win",  "down"),
    "maximize":     ("win",  "up"),

    # Navigation / scrolling
    "scroll up":    "pageup",
    "scroll down":  "pagedown",
    "home":         "home",
    "end":          "end",
    "go back":      ("alt", "left"),
    "go forward":   ("alt", "right"),

    # Screenshots
    "screenshot":   ("win", "shift", "s"),

    # Media
    "volume up":    "volumeup",
    "volume down":  "volumedown",
    "mute":         "volumemute",
    "play":         "playpause",
    "pause":        "playpause",
    "next":         "nexttrack",
    "previous":     "prevtrack",
}


class VoiceCommandRouter:
    """
    Translates a VoiceCommand transcript into an OS action.

    Usage
    -----
    router = VoiceCommandRouter()
    if router.route(voice_command):
        pass   # action was executed
    """

    def route(self, transcript: str) -> bool:
        """
        Attempt to match the transcript to a known command and execute it.

        Parameters
        ----------
        transcript : str
            Raw (lowercased) transcript from the voice listener.

        Returns
        -------
        bool
            True if a matching command was found and executed, False otherwise.
        """
        text = transcript.strip().lower()

        # Longest-match wins (avoids 'copy' matching before 'copy all')
        matched_key = None
        for keyword in sorted(_COMMAND_MAP, key=len, reverse=True):
            if keyword in text:
                matched_key = keyword
                break

        if matched_key is None:
            return False

        action = _COMMAND_MAP[matched_key]
        try:
            if isinstance(action, tuple):
                pyautogui.hotkey(*action)
            else:
                pyautogui.press(action)
        except Exception:
            return False

        return True

    @staticmethod
    def list_commands() -> list[str]:
        """Return all recognised keywords (useful for README / HUD help)."""
        return sorted(_COMMAND_MAP.keys())
