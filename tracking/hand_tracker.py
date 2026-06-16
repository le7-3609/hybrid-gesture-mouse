"""
HandTracker — wraps MediaPipe Hands for single-hand landmark detection.

Responsibility: convert raw BGR/RGB video frames into a structured
MediaPipe NormalizedLandmarkList (or None when no hand is found).
All MediaPipe configuration lives here; nothing else touches mp.solutions.
"""
try:
    import mediapipe.solutions.hands as mp_hands
except (ModuleNotFoundError, AttributeError):
    import utils.mediapipe_shim as mp_hands


class HandTracker:
    """
    Thin wrapper around MediaPipe Hands.

    Parameters
    ----------
    max_num_hands : int
        Maximum number of hands to detect (default 1 for performance).
    min_detection_confidence : float
        Minimum detection confidence threshold.
    min_tracking_confidence : float
        Minimum tracking confidence threshold.
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._hands = mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb_frame):
        """
        Run hand detection on an RGB frame.

        Parameters
        ----------
        rgb_frame : np.ndarray
            H×W×3 uint8 array in RGB colour order.

        Returns
        -------
        NormalizedLandmarkList | None
            The first detected hand's landmarks, or None if no hand found.
        """
        results = self._hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]
        return None

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()
