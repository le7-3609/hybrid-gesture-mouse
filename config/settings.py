"""
Centralized configuration and constants for Gesture+.
All magic values live here — no other module hardcodes numbers or paths.
"""

# ─────────────────────────────────────────────────────────────
# File Paths (relative to project root)
# ─────────────────────────────────────────────────────────────
DATASET_PATH         = "training/gestures_dataset.csv"
MODEL_PATH           = "models/gesture_model.pkl"
HAND_LANDMARKER_PATH = "models/hand_landmarker.task"

# ─────────────────────────────────────────────────────────────
# Gesture class map  (index → name)
# ─────────────────────────────────────────────────────────────
CLASSES: dict[int, str] = {
    0: "idle",
    1: "move",
    2: "click",
    3: "drag",
    4: "scroll_up",
    5: "scroll_down",
    6: "right_click",
    7: "double_click",
}

# ─────────────────────────────────────────────────────────────
# HUD color palette  (BGR format for OpenCV)
# ─────────────────────────────────────────────────────────────
STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "idle":         (128, 128, 128),   # Gray
    "move":         (0,   255,   0),   # Green
    "click":        (255,   0,   0),   # Blue
    "drag":         (0,   165, 255),   # Orange
    "scroll_up":    (255,   0, 255),   # Magenta
    "scroll_down":  (255, 255,   0),   # Cyan
    "right_click":  (0,   100, 255),   # Gold
    "double_click": (180,   0, 255),   # Purple
}

# ─────────────────────────────────────────────────────────────
# Controller defaults
# ─────────────────────────────────────────────────────────────
SMOOTHING          = 0.25   # EMA factor (0 = static, 1 = raw)
CONFIDENCE         = 0.75   # Minimum ML probability to accept
HISTORY_SIZE       = 7      # Majority-voting window (frames)
CLICK_DEBOUNCE     = 0.4    # Seconds between successive clicks
SCROLL_SENSITIVITY = 1.5    # Scroll multiplier
SCROLL_STEP        = 2      # Discrete scroll unit

# ─────────────────────────────────────────────────────────────
# Voice listener defaults
# ─────────────────────────────────────────────────────────────
VOICE_ENABLED          = True
VOICE_SAMPLE_RATE      = 16_000    # Hz — Whisper expects 16 kHz
VOICE_SILENCE_THRESHOLD = 0.015   # RMS energy below this = silence
VOICE_MIN_DURATION     = 0.5      # Seconds of speech before transcribing
VOICE_MAX_DURATION     = 5.0      # Hard cap on recording length (seconds)
VOICE_WHISPER_MODEL    = "tiny"   # tiny / base / small (local, no API key)
VOICE_DISPLAY_SECONDS  = 3.0      # How long to show last command on HUD
