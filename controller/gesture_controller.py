"""
GestureController — top-level orchestrator for the real-time processing loop.

Dependency graph (all injected via constructor):
  BaseMouseService       → GestureStateMachine
  BaseGestureClassifier  → predict()
  BaseVoiceListener      → get_latest_command()
  VoiceCommandRouter     → route(transcript)

The controller owns NO business logic of its own — it wires and drives
the other layers in a single OpenCV frame loop.

Raises pyautogui.FailSafeException (from GestureStateMachine) to trigger
an emergency stop — caught here and logged cleanly.
"""
from __future__ import annotations
import cv2
import pyautogui
from interfaces.mouse_service import BaseMouseService
from interfaces.classifier import BaseGestureClassifier
from interfaces.voice_listener import BaseVoiceListener
from voice.command_router import VoiceCommandRouter
from tracking.hand_tracker import HandTracker
from tracking.normalization import normalize_landmarks
from tracking.coordinate_mapper import CoordinateMapper
from classification.vote_stabilizer import VoteStabilizer
from state_machine.gesture_state_machine import GestureStateMachine
from domain.gesture_state import GestureState
from ui.hud_renderer import HudRenderer
from utils.fps_counter import FPSCounter
from utils.logger import get_logger

logger = get_logger("controller")


class GestureController:
    """
    Wires and drives all processing layers in a real-time camera loop.

    Parameters — all injected, none created internally
    ----------
    mouse_service : BaseMouseService
    gesture_classifier : BaseGestureClassifier
    voice_listener : BaseVoiceListener
        Pass NullVoiceListener for --no-voice mode.
    command_router : VoiceCommandRouter
    smoothing, confidence, history_size, debounce,
    scroll_sens, scroll_step : floats/ints forwarded to child layers.
    """

    def __init__(
        self,
        mouse_service:       BaseMouseService,
        gesture_classifier:  BaseGestureClassifier,
        voice_listener:      BaseVoiceListener,
        command_router:      VoiceCommandRouter,
        smoothing:    float = 0.25,
        confidence:   float = 0.75,
        history_size: int   = 7,
        debounce:     float = 0.4,
        scroll_sens:  float = 1.5,
        scroll_step:  int   = 2,
    ) -> None:
        self._svc        = mouse_service
        self._clf        = gesture_classifier
        self._voice      = voice_listener
        self._router     = command_router
        self._confidence = confidence

        # Store construction params for child layers created in run()
        self._smoothing    = smoothing
        self._history_size = history_size
        self._debounce     = debounce
        self._scroll_sens  = scroll_sens
        self._scroll_step  = scroll_step

        # Layers that don't require mediapipe/cv2 at construction time
        self._stabilizer  = VoteStabilizer(window_size=history_size)
        sw, sh            = mouse_service.get_screen_size()
        self._mapper      = CoordinateMapper(sw, sh, smoothing)
        self._state_machine = GestureStateMachine(
            mouse_service, debounce, scroll_sens, scroll_step
        )
        self._hud  = HudRenderer()
        self._fps  = FPSCounter()
        # HandTracker deferred to run() — avoids mediapipe init at import time
        self._tracker: HandTracker | None = None


    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the webcam and start the real-time orchestration loop."""
        init_x, init_y = self._svc.get_position()
        self._mapper.reset(init_x, init_y)

        if self._tracker is None:
            self._tracker = HandTracker()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam — aborting.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._voice.start()
        logger.info("Gesture+ engine running. Press Q or move cursor to corner to stop.")

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    continue

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]
                fps   = self._fps.tick()

                # ── 1. Hand tracking ──────────────────────────────────
                rgb          = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks    = self._tracker.detect(rgb)
                predicted    = GestureState.IDLE
                pred_prob    = 1.0
                raw_x = raw_y = None

                if landmarks:
                    self._hud.draw_hand_landmarks(frame, landmarks)
                    tip    = landmarks.landmark[8]
                    raw_x  = int(tip.x * w)
                    raw_y  = int(tip.y * h)
                    norm   = normalize_landmarks(landmarks)
                    if norm is not None:
                        predicted, pred_prob = self._clf.predict(norm)
                        if pred_prob >= self._confidence:
                            self._stabilizer.add_vote(predicted)
                    stabilized = self._stabilizer.get_stabilized_state()
                else:
                    stabilized = GestureState.IDLE
                    self._stabilizer.clear()

                # ── 2. Coordinate mapping ─────────────────────────────
                if raw_x is not None:
                    sx, sy = self._mapper.map_and_smooth(raw_x, raw_y, w, h)
                else:
                    prev = (self._mapper.prev_x, self._mapper.prev_y)
                    if None in prev:
                        sx, sy = self._svc.get_position()
                        self._mapper.reset(sx, sy)
                    else:
                        sx, sy = int(prev[0]), int(prev[1])

                # ── 3. State machine execution ────────────────────────
                try:
                    self._state_machine.execute(stabilized, sx, sy)
                except pyautogui.FailSafeException:
                    logger.warning("Fail-safe triggered — shutting down.")
                    break

                # ── 4. Voice command polling ──────────────────────────
                cmd = self._voice.get_latest_command()
                if cmd:
                    routed = self._router.route(cmd.transcript)
                    logger.info(
                        f"Voice: '{cmd.transcript}' → {'executed' if routed else 'unknown'}"
                    )
                    self._hud.set_voice_feedback(cmd.transcript, routed)

                # ── 5. HUD rendering ──────────────────────────────────
                self._hud.draw_hud(
                    frame=frame,
                    state=stabilized,
                    confidence=pred_prob,
                    screen_x=sx,
                    screen_y=sy,
                    fps=fps,
                    vote_history=self._stabilizer.get_history(),
                    is_recording=self._voice.is_recording(),
                )

                cv2.imshow("Gesture+ | Hand Gesture + Voice Controller", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            self._state_machine.shutdown()
            if self._tracker is not None:
                self._tracker.close()
            self._voice.stop()
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Gesture+ engine stopped.")
