"""All tunable parameters in one place.

Ratios are dimensionless: they are computed from MediaPipe's 3-D *world*
landmarks and normalised by hand size, so they do not depend on how far the
hand is from the camera or how it is rotated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
# Pinned model build (version 1, float16) and its SHA-256. run.sh downloads and
# verifies it; keep these identical to the values in run.sh (a test checks).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


@dataclass
class GestureThresholds:
    # dist(wrist, tip) / dist(wrist, PIP). Measured on real hands:
    # straight fingers 1.20-1.86, curled fingers 0.60-0.99.
    finger_extended_ratio: float = 1.10
    # While volume mode is engaged the index finger may bend toward the
    # thumb, but if it curls into the palm (a fist, ~0.7) we disengage.
    index_curled_ratio: float = 0.85
    # dist(thumb tip, pinky MCP) / palm width. Tucked thumb 0.79-0.98,
    # relaxed open hand 1.19+, thumb clearly out 1.5-2.5.
    thumb_open_ratio: float = 1.05     # enough for "open palm"
    thumb_out_ratio: float = 1.30      # thumb clearly out (thumbs-up vs fist)
    # Thumb tip <-> index tip / palm length below this counts as "touching".
    pinch_closed_ratio: float = 0.45
    # Palm orientation: |sin(angle)| between wrist->index-MCP and wrist->pinky-MCP
    # must exceed this, otherwise the hand is edge-on and neither side fires.
    palm_facing_margin: float = 0.25
    palm_sign: float = 1.0             # set to -1 if palm/back come out swapped


@dataclass
class TimingConfig:
    hold_s: float = 0.30               # a discrete gesture must be held this long to fire
    release_gap_s: float = 0.25        # gaps shorter than this count as tracking flicker
    action_cooldown_s: float = 1.50    # per-gesture cooldown after it fires
    global_cooldown_s: float = 0.80    # after any discrete action, block all others
    volume_engage_s: float = 0.25      # hold the pinch pose this long to enter volume mode
    volume_disengage_s: float = 0.15   # grace period before leaving volume mode


@dataclass
class VolumeConfig:
    # Thumb-index tip distance / palm length mapped linearly onto 0-100 %.
    pinch_min_ratio: float = 0.18      # at or below -> 0 %
    pinch_max_ratio: float = 1.05      # at or above -> 100 %
    # One-Euro filter on the target level (jitter removal).
    filter_min_cutoff_hz: float = 0.5  # lower = smoother when the hand is still
    filter_beta: float = 0.01          # higher = less lag when the hand moves fast
    max_rate_per_s: float = 150.0      # slew limit in % per second
    # How finger distance becomes a volume level (neither ever jumps on engage):
    #  "relative" - only the *change* in gap matters: start with the fingers
    #               together and spread them -> volume rises by the amount you
    #               spread; start apart and bring them together -> it falls by
    #               that amount. Full closed-to-wide travel = volume_gain * 100 %.
    #  "absolute" - a fixed gap always means the same level (pinch_min..max ->
    #               0..100 %). Soft takeover: the volume is only "grabbed" once
    #               the hand's level reaches the current volume.
    mapping: str = "relative"
    volume_gain: float = 1.0
    pickup_tolerance: float = 4.0
    deadband: int = 2                  # ignore changes smaller than this (hysteresis)
    release_rollback_s: float = 0.10   # on release, restore the level from this long ago


@dataclass
class CameraConfig:
    index: int | None = None           # explicit OpenCV index; overrides name
    name: str | None = None            # substring of the camera name; None = built-in
    width: int = 1280
    height: int = 720
    mirror: bool = True


@dataclass
class TrackerConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    thresholds: GestureThresholds = field(default_factory=GestureThresholds)
    timing: TimingConfig = field(default_factory=TimingConfig)
    volume: VolumeConfig = field(default_factory=VolumeConfig)
    dry_run: bool = False
    show_window: bool = True
    max_seconds: float | None = None
    snapshot_dir: Path | None = None
    players: tuple[str, ...] = ("Spotify", "Music")
