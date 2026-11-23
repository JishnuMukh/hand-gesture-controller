"""Stateless per-frame hand pose classification.

Finger states use MediaPipe's metric 3-D *world* landmarks, which makes them
invariant to camera distance and far less sensitive to hand rotation than 2-D
pixel heuristics. Whether the palm or the back of the hand faces the camera is
decided from the 2-D image landmarks together with the handedness label.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import GestureThresholds

# Landmark indices (MediaPipe hand model)
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# (tip, pip) for index, middle, ring, pinky
_FINGERS = ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))


class Pose(str, Enum):
    NONE = "none"              # no hand
    PALM_FRONT = "palm_front"  # five fingers, palm towards the camera
    PALM_BACK = "palm_back"    # five fingers, back of the hand towards the camera
    OPEN_HAND = "open_hand"    # five fingers, orientation unclear (edge-on)
    PEACE = "peace"            # index + middle
    PINCH = "pinch"            # thumb + index active, other fingers folded
    FIST = "fist"
    THUMB_UP = "thumb_up"
    OTHER = "other"


POSE_LABELS = {
    Pose.NONE: "No hand",
    Pose.PALM_FRONT: "Palm (front)",
    Pose.PALM_BACK: "Back of hand",
    Pose.OPEN_HAND: "Open hand (turn it)",
    Pose.PEACE: "Peace sign",
    Pose.PINCH: "Pinch",
    Pose.FIST: "Fist",
    Pose.THUMB_UP: "Thumb only",
    Pose.OTHER: "Unrecognised",
}


@dataclass(frozen=True)
class HandFeatures:
    pose: Pose
    extension: tuple[float, float, float, float]       # index..pinky ratios
    extended: tuple[bool, bool, bool, bool, bool]      # thumb..pinky
    thumb_spread: float                                # thumb tip <-> pinky MCP / palm width
    pinch_ratio: float                                 # thumb tip <-> index tip / palm length
    volume_hold_ok: bool                               # pose still valid for an engaged pinch
    facing: float = 0.0                                # >0 palm towards camera, <0 back, ~0 edge-on


def _dist(p: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(p[a] - p[b]))


def palm_facing(image_points: np.ndarray, handedness: str, sign: float = 1.0) -> float:
    """Signed palm orientation in [-1, 1]: positive = palm faces the camera.

    The 2-D cross product of wrist->index-MCP and wrist->pinky-MCP flips sign
    when the hand turns over. Left and right hands are mirror images, so the
    handedness label decides which sign means "palm". Its magnitude is the sine
    of the angle between the two vectors, so it is near zero when the hand is
    seen edge-on. Invariant to in-plane rotation and to scale.
    """
    p = np.asarray(image_points, dtype=np.float64)[:, :2]
    v1, v2 = p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST]
    norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if norm < 1e-12:
        return 0.0
    cross = float(v1[0] * v2[1] - v1[1] * v2[0]) / norm   # image y axis points down
    label = handedness.strip().lower()
    if label not in ("left", "right"):
        return 0.0
    # Calibrated on the mirrored preview: MediaPipe labels a palm-forward hand
    # whose thumb is on the left as "Left", and that gives cross > 0.
    return sign * (cross if label == "left" else -cross)


def analyze_hand(points: np.ndarray, th: GestureThresholds,
                 image_points: np.ndarray | None = None,
                 handedness: str | None = None) -> HandFeatures:
    """Classify a hand from (21, 3) world landmarks (+ image landmarks for orientation)."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape != (21, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {p.shape}")

    eps = 1e-9
    palm_length = max(_dist(p, WRIST, MIDDLE_MCP), eps)
    palm_width = max(_dist(p, INDEX_MCP, PINKY_MCP), eps)

    extension = tuple(
        _dist(p, WRIST, tip) / max(_dist(p, WRIST, pip), eps) for tip, pip in _FINGERS
    )
    finger_ext = tuple(r > th.finger_extended_ratio for r in extension)
    thumb_spread = _dist(p, THUMB_TIP, PINKY_MCP) / palm_width
    pinch_ratio = _dist(p, THUMB_TIP, INDEX_TIP) / palm_length

    idx, mid, ring, pinky = finger_ext
    others_folded = not (mid or ring or pinky)
    index_active = extension[0] > th.index_curled_ratio
    # The thumb is "in play" when it is out to the side or touching the index tip.
    thumb_active = thumb_spread > th.thumb_open_ratio or pinch_ratio < th.pinch_closed_ratio

    facing = 0.0
    if image_points is not None and handedness:
        facing = palm_facing(image_points, handedness, th.palm_sign)

    if idx and mid and ring and pinky and thumb_spread > th.thumb_open_ratio:
        if facing > th.palm_facing_margin:
            pose = Pose.PALM_FRONT
        elif facing < -th.palm_facing_margin:
            pose = Pose.PALM_BACK
        else:
            pose = Pose.OPEN_HAND
    elif idx and mid and not ring and not pinky:
        pose = Pose.PEACE
    elif others_folded and index_active and thumb_active:
        pose = Pose.PINCH
    elif not any(finger_ext):
        pose = Pose.THUMB_UP if thumb_spread > th.thumb_out_ratio else Pose.FIST
    else:
        pose = Pose.OTHER

    # While volume mode is engaged, stay engaged as long as middle/ring/pinky
    # remain folded and the index has not curled all the way into a fist.
    volume_hold_ok = others_folded and index_active

    return HandFeatures(
        pose=pose,
        extension=extension,  # type: ignore[arg-type]
        extended=(thumb_spread > th.thumb_open_ratio, *finger_ext),  # type: ignore[arg-type]
        thumb_spread=thumb_spread,
        pinch_ratio=pinch_ratio,
        volume_hold_ok=volume_hold_ok,
        facing=facing,
    )
