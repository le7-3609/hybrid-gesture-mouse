"""
model_loader — deserialises a trained scikit-learn model from a pickle file.

Kept as a standalone function (not a class) because loading is a one-shot
operation at startup.  The controller receives an already-loaded model via DI.
"""
from __future__ import annotations
import pickle
from pathlib import Path


def load_pickle_model(model_path: str | Path):
    """
    Load and return a scikit-learn model from a pickle file.

    Parameters
    ----------
    model_path : str | Path
        Absolute or relative path to the .pkl file.

    Returns
    -------
    sklearn estimator
        The deserialised model object.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file cannot be unpickled.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")

    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to load model from '{path}': {exc}") from exc
