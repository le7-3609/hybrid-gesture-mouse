"""
normalize_landmarks — geometric preprocessing for the gesture classifier.

Applies two invariances so the ML model is robust regardless of where
in the frame the hand appears or how large it looks:

  Translation invariance: wrist landmark is moved to the origin (0, 0, 0).
  Scale invariance: all coordinates are divided by the distance from the
                    wrist to the middle-finger MCP joint.

Returns a flat 63-dimensional vector  [x0, y0, z0, x1, y1, z1, … x20, y20, z20]
or None if the reference distance is zero (degenerate frame).
"""
from __future__ import annotations
import math


def normalize_landmarks(hand_landmarks) -> list[float] | None:
    """
    Normalise a MediaPipe NormalizedLandmarkList to a translation- and
    scale-invariant 63-dim float vector.

    Parameters
    ----------
    hand_landmarks : mediapipe NormalizedLandmarkList
        Raw output from HandTracker.detect().

    Returns
    -------
    list[float] | None
        63 normalised coordinates, or None on degenerate input.
    """
    landmarks = hand_landmarks.landmark

    # Wrist (landmark 0) becomes the origin
    wrist = landmarks[0]
    wrist_x, wrist_y, wrist_z = wrist.x, wrist.y, wrist.z

    # Middle-finger MCP (landmark 9) defines the scale reference
    mid_mcp = landmarks[9]
    scale = math.sqrt(
        (mid_mcp.x - wrist_x) ** 2
        + (mid_mcp.y - wrist_y) ** 2
        + (mid_mcp.z - wrist_z) ** 2
    )

    if scale < 1e-6:
        return None  # hand too close / degenerate frame

    coords: list[float] = []
    for lm in landmarks:
        coords.append((lm.x - wrist_x) / scale)
        coords.append((lm.y - wrist_y) / scale)
        coords.append((lm.z - wrist_z) / scale)

    return coords  # length == 63
