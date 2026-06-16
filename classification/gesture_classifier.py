"""
GestureClassifier — scikit-learn model adapter implementing BaseGestureClassifier.

Responsibility: wrap a pre-trained sklearn estimator and expose a clean
predict() method that returns a typed (GestureState, confidence) pair.
The controller depends on BaseGestureClassifier — never on this concrete class.
"""
from __future__ import annotations
import numpy as np
from interfaces.classifier import BaseGestureClassifier
from domain.gesture_state import GestureState


class GestureClassifier(BaseGestureClassifier):
    """
    Wraps a scikit-learn classifier (e.g. RandomForestClassifier) that was
    trained on 63-dimensional normalised landmark vectors.

    Parameters
    ----------
    model : sklearn estimator
        Any estimator that implements predict_proba(X).
    """

    def __init__(self, model) -> None:
        self._model = model

    def predict(self, normalized_landmarks: list[float]) -> tuple[GestureState, float]:
        """
        Run inference and return the most-probable gesture state.

        Parameters
        ----------
        normalized_landmarks : list[float]
            63-dim vector from tracking.normalization.normalize_landmarks().

        Returns
        -------
        (GestureState, float)
            Predicted state and its probability score in [0, 1].
        """
        X = np.array([normalized_landmarks])
        probs = self._model.predict_proba(X)[0]
        idx   = int(np.argmax(probs))
        return GestureState(idx), float(probs[idx])
