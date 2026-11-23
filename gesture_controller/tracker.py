"""Thin wrapper around the MediaPipe Tasks HandLandmarker (VIDEO mode)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .config import MODEL_SHA256, MODEL_URL, TrackerConfig

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
)


@dataclass(frozen=True)
class HandObservation:
    image_points: np.ndarray   # (21, 3) normalised image coords (x, y in 0..1)
    world_points: np.ndarray   # (21, 3) metric coords centred on the hand
    handedness: str
    score: float

    def pixel_points(self, width: int, height: int) -> np.ndarray:
        pts = self.image_points[:, :2] * (width, height)
        return pts.astype(np.int32)


class HandTracker:
    def __init__(self, cfg: TrackerConfig) -> None:
        model = Path(cfg.model_path)
        if not model.is_file():
            raise FileNotFoundError(
                f"Hand landmark model not found at {model}.\n"
                f"Run ./run.sh (it downloads and verifies the model), or download\n"
                f"  {MODEL_URL}\n"
                f"and check that its SHA-256 is {MODEL_SHA256}."
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=cfg.min_detection_confidence,
            min_hand_presence_confidence=cfg.min_presence_confidence,
            min_tracking_confidence=cfg.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> HandObservation | None:
        # VIDEO mode requires strictly increasing timestamps.
        ts = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, ts)
        if not result.hand_landmarks:
            return None
        img = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float64)
        if result.hand_world_landmarks:
            world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]],
                             dtype=np.float64)
        else:  # fall back to aspect-correct image coordinates
            world = img * (frame_bgr.shape[1], frame_bgr.shape[0], frame_bgr.shape[1])
        handed, score = "?", 0.0
        if result.handedness and result.handedness[0]:
            cat = result.handedness[0][0]
            handed, score = cat.category_name or "?", float(cat.score)
        return HandObservation(img, world, handed, score)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
