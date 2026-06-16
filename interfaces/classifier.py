"""
BaseGestureClassifier — Abstract Base Class for ML gesture prediction.

Decouples the controller from any specific ML library (scikit-learn, TFLite, …).
Concrete implementation: GestureClassifier (wraps a scikit-learn model).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from domain.gesture_state import GestureState


class BaseGestureClassifier(ABC):

    @abstractmethod
    def predict(self, normalized_landmarks: list[float]) -> tuple[GestureState, float]:
        """
        Run inference on a normalized landmark vector.

        Parameters
        ----------
        normalized_landmarks : list[float]
            63-dimensional vector produced by normalization.normalize_landmarks().

        Returns
        -------
        (GestureState, float)
            The predicted state and the associated probability in [0, 1].
        """
