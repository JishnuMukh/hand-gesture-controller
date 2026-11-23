"""Pick a camera by name (e.g. the MacBook's built-in camera, not an iPhone).

OpenCV's AVFoundation backend numbers cameras by sorting the devices on their
``uniqueID``. ``system_profiler`` reports each camera's name and uniqueID, so
sorting its list the same way gives the index OpenCV will use for each name.
This matters because an iPhone Continuity Camera often sorts *before* the
built-in camera and silently becomes index 0.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Tried in order when no name is given: built-in cameras on MacBooks / iMacs.
BUILTIN_HINTS = ("MacBook", "FaceTime", "Built-in")


@dataclass(frozen=True)
class Camera:
    index: int
    name: str
    unique_id: str


def list_cameras() -> list[Camera]:
    """Cameras in OpenCV AVFoundation index order."""
    try:
        out = subprocess.run(["system_profiler", "SPCameraDataType", "-json"],
                             capture_output=True, text=True, timeout=15).stdout
        items = json.loads(out).get("SPCameraDataType", [])
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    devices = sorted(
        ((it.get("spcamera_unique-id", ""), it.get("_name", "?")) for it in items),
        key=lambda d: d[0],
    )
    return [Camera(i, name, uid) for i, (uid, name) in enumerate(devices)]


def resolve_camera(index: int | None, name: str | None) -> tuple[int, str]:
    """Return (OpenCV index, description). An explicit index always wins."""
    cams = list_cameras()
    if index is not None:
        match = next((c for c in cams if c.index == index), None)
        return index, match.name if match else f"camera {index}"
    hints = (name,) if name else BUILTIN_HINTS
    for hint in hints:
        for cam in cams:
            if hint.lower() in cam.name.lower():
                return cam.index, cam.name
    if cams:
        log.warning("no camera matching %s; available: %s - using index 0",
                    " / ".join(hints), ", ".join(f"{c.index}: {c.name}" for c in cams))
    return 0, cams[0].name if cams else "camera 0"
