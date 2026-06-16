import os
import argparse
from utils.logger import get_logger
from services.factory import create_mouse_service
from config.settings import (
    MODEL_PATH,
    SMOOTHING,
    CONFIDENCE,
    HISTORY_SIZE,
    CLICK_DEBOUNCE,
    SCROLL_SENSITIVITY,
    SCROLL_STEP,
)
from classification.model_loader import load_pickle_model
from classification.gesture_classifier import GestureClassifier
from voice.null_listener import NullVoiceListener
from voice.whisper_listener import WhisperVoiceListener
from voice.command_router import VoiceCommandRouter
from controller.gesture_controller import GestureController

logger = get_logger("main")


def main():
    parser = argparse.ArgumentParser(
        description="Gesture+ Real-Time Hybrid Hand Gesture & Voice Controller"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help="Path to trained ML model pickle",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=SMOOTHING,
        help="EMA smoothing factor (0 = static, 1 = raw jittery)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=CONFIDENCE,
        help="Min probability to accept predicted state changes",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=HISTORY_SIZE,
        help="Queue size for majority voting filter",
    )
    parser.add_argument(
        "--debounce",
        type=float,
        default=CLICK_DEBOUNCE,
        help="Cooldown in seconds to trigger subsequent clicks",
    )
    parser.add_argument(
        "--scroll-sens",
        type=float,
        default=SCROLL_SENSITIVITY,
        help="Scroll vertical sensitivity multiplier",
    )
    parser.add_argument(
        "--scroll-step",
        type=int,
        default=SCROLL_STEP,
        help="Discrete scroll step size",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable local Whisper voice listener (runs gestures only)",
    )

    args = parser.parse_args()

    # 1. Verify model file exists
    if not os.path.exists(args.model):
        logger.error(f"Trained ML model file '{args.model}' not found!")
        logger.info("Please train a model first using your training scripts.")
        logger.info(
            "Or generate a mock model for testing with: python training/train.py --synthetic"
        )
        return

    try:
        # 2. Load ML Model
        raw_model = load_pickle_model(args.model)
        gesture_classifier = GestureClassifier(raw_model)

        # 3. Resolve OS Mouse Service dependency
        mouse_service = create_mouse_service()

        # 4. Resolve Voice Listener & Command Router dependencies
        if args.no_voice:
            logger.info("Voice listener disabled by user (using NullVoiceListener).")
            voice_listener = NullVoiceListener()
        else:
            logger.info("Initializing Whisper voice listener (offline local VAD)...")
            voice_listener = WhisperVoiceListener()

        command_router = VoiceCommandRouter()

        # 5. Instantiate and run the Orchestrator Controller
        controller = GestureController(
            mouse_service=mouse_service,
            gesture_classifier=gesture_classifier,
            voice_listener=voice_listener,
            command_router=command_router,
            smoothing=args.smoothing,
            confidence=args.confidence,
            history_size=args.history,
            debounce=args.debounce,
            scroll_sens=args.scroll_sens,
            scroll_step=args.scroll_step,
        )

        controller.run()
    except Exception as e:
        logger.exception(f"Unhandled critical exception in main execution: {e}")


if __name__ == "__main__":
    main()
