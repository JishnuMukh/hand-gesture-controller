#!/usr/bin/env python3
"""Run the gesture classifier on still images (handy for tuning thresholds).

    .venv/bin/python scripts/classify_images.py photo1.jpg photo2.jpg ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

from gesture_controller.config import DEFAULT_MODEL_PATH, GestureThresholds  # noqa: E402
from gesture_controller.gestures import analyze_hand  # noqa: E402


def main(paths: list[str]) -> int:
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(DEFAULT_MODEL_PATH)), num_hands=2,
        min_hand_detection_confidence=0.3)
    th = GestureThresholds()
    with vision.HandLandmarker.create_from_options(opts) as det:
        for path in paths:
            res = det.detect(mp.Image.create_from_file(path))
            if not res.hand_world_landmarks:
                print(f"{Path(path).name:45s} no hand")
            for i, hand in enumerate(res.hand_world_landmarks):
                f = analyze_hand(np.array([[p.x, p.y, p.z] for p in hand]), th)
                ext = "".join("TIMRP"[j] if e else "." for j, e in enumerate(f.extended))
                print(f"{Path(path).name:45s} hand{i} {f.pose.value:10s} fingers={ext} "
                      f"thumb={f.thumb_spread:.2f} pinch={f.pinch_ratio:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
