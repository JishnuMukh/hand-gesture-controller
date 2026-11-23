"""Main loop and command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

from .cameras import list_cameras, resolve_camera
from .config import AppConfig
from .engine import ActionType, GestureEngine
from .gestures import POSE_LABELS, analyze_hand
from .macos import MacController
from .overlay import Overlay
from .tracker import HandTracker

log = logging.getLogger("gesture_controller")
WINDOW = "Hand Gesture Controller"


def open_camera(index: int, width: int, height: int,
                permission_wait_s: float = 30.0) -> cv2.VideoCapture:
    # On first use OpenCV triggers macOS's camera prompt but fails immediately
    # instead of waiting for the answer, so keep retrying while the user decides.
    deadline = time.monotonic() + permission_wait_s
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        log.info("waiting for camera access - click Allow if macOS asks...")
    while not cap.isOpened() and time.monotonic() < deadline:
        time.sleep(1.0)
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {index}. Grant camera access to your terminal app in "
            "System Settings > Privacy & Security > Camera (or pick another with --list-cameras)."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame
    # The first frames after opening can be empty while the camera warms up.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size and frame.mean() > 1.0:
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(
        f"Camera {index} opened but delivered no images. macOS may be blocking camera access "
        "for this terminal (System Settings > Privacy & Security > Camera)."
    )


def run(cfg: AppConfig) -> int:
    overlay = Overlay(dry_run=cfg.dry_run)
    controller = MacController(cfg.players, dry_run=cfg.dry_run, on_message=overlay.notify)
    engine = GestureEngine(cfg.timing, cfg.volume)

    cam_index, cam_name = resolve_camera(cfg.camera.index, cfg.camera.name)
    cap = open_camera(cam_index, cfg.camera.width, cfg.camera.height)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("camera %d (%s) opened at %dx%d", cam_index, cam_name, w, h)

    controller.start()
    log.info("system volume %s%%, muted=%s (backend: %s)", controller.state.volume,
             controller.state.muted, controller.backend)

    if cfg.show_window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, min(w, 1280), min(h, 720))

    controls_enabled = True
    last_pose = None
    was_volume_mode = False
    snap_events: list[str] = []
    if cfg.snapshot_dir is not None:
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
    fps, frames, fps_t0 = 0.0, 0, time.monotonic()
    t_start = time.monotonic()
    read_failures = 0
    frame = None

    try:
        with HandTracker(cfg.tracker) as tracker:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    read_failures += 1
                    if read_failures > 30:
                        log.error("camera stopped delivering frames")
                        return 1
                    continue
                read_failures = 0
                if cfg.camera.mirror:
                    frame = cv2.flip(frame, 1)

                now = time.monotonic()
                hand = tracker.process(frame, int((now - t_start) * 1000))
                feat = (analyze_hand(hand.world_points, cfg.thresholds, hand.image_points,
                                     hand.handedness) if hand else None)
                out = engine.update(feat, now, controller.state.volume)

                if out.pose != last_pose:
                    log.debug("pose: %s", POSE_LABELS.get(out.pose, out.pose))
                    last_pose = out.pose
                if out.volume_mode != was_volume_mode:
                    level = out.smoothed_volume
                    log.info("volume mode %s (level %s%%)", "ON" if out.volume_mode else "OFF",
                             "?" if level is None else round(level))
                    snap_events.append("volume_on" if out.volume_mode else "volume_off")
                    was_volume_mode = out.volume_mode

                for action in out.actions:
                    if not controls_enabled:
                        continue
                    if action.type is ActionType.SET_VOLUME:
                        controller.set_volume(action.value or 0)
                    elif action.type is ActionType.MUTE:
                        log.info("gesture: palm (front) -> mute  [facing %+.2f]", feat.facing)
                        controller.mute()
                        snap_events.append("mute")
                    elif action.type is ActionType.UNMUTE:
                        log.info("gesture: back of hand -> unmute  [facing %+.2f]", feat.facing)
                        controller.unmute()
                        snap_events.append("unmute")
                    elif action.type is ActionType.PLAY_PAUSE:
                        log.info("gesture: peace sign -> play/pause")
                        controller.play_pause()
                        snap_events.append("playpause")

                frames += 1
                if now - fps_t0 >= 0.5:
                    fps = frames / (now - fps_t0)
                    frames, fps_t0 = 0, now

                overlay.render(frame, hand, out, engine, controller.state, fps, controls_enabled)
                if cfg.snapshot_dir is not None and snap_events:
                    name = f"{now - t_start:06.2f}s_{'_'.join(snap_events)}.jpg"
                    cv2.imwrite(str(cfg.snapshot_dir / name), frame)
                    snap_events.clear()

                if cfg.show_window:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord(" "):
                        controls_enabled = not controls_enabled
                        overlay.notify("Controls " + ("resumed" if controls_enabled else "paused"))
                    elif key == ord("h"):
                        overlay.show_help = not overlay.show_help
                    # Closing the window with the red button ends the app.
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                if cfg.max_seconds is not None and now - t_start >= cfg.max_seconds:
                    log.info("max run time reached")
                    break
    finally:
        if cfg.snapshot_dir is not None and frame is not None:
            cv2.imwrite(str(cfg.snapshot_dir / "last.jpg"), frame)
            log.info("snapshots saved in %s", cfg.snapshot_dir)
        cap.release()
        controller.stop()
        if cfg.show_window:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
    return 0


def parse_args(argv: list[str] | None = None) -> AppConfig:
    p = argparse.ArgumentParser(description="Control macOS volume and media with hand gestures.")
    p.add_argument("--camera", type=int, help="camera index (overrides --camera-name)")
    p.add_argument("--camera-name",
                   help="use the camera whose name contains this text (default: the built-in "
                        "MacBook/FaceTime camera, never an iPhone Continuity Camera)")
    p.add_argument("--list-cameras", action="store_true", help="list cameras and exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--no-mirror", action="store_true", help="don't mirror the preview")
    p.add_argument("--dry-run", action="store_true",
                   help="recognise gestures but don't change volume or playback")
    p.add_argument("--players", default="Spotify,Music",
                   help="media players to control, in priority order (default Spotify,Music)")
    p.add_argument("--volume-gain", type=float,
                   help="volume change for a full closed-to-wide pinch, as a multiple of 100%% "
                        "(default 1.0)")
    p.add_argument("--flip-palm", action="store_true",
                   help="swap palm/back detection if mute and unmute come out reversed")
    p.add_argument("--absolute-volume", action="store_true",
                   help="fixed gap -> fixed level mapping (with soft takeover) instead of the "
                        "default relative mapping")
    p.add_argument("--pinch-min", type=float, help="pinch ratio mapped to 0%% (default 0.18)")
    p.add_argument("--pinch-max", type=float, help="pinch ratio mapped to 100%% (default 1.05)")
    p.add_argument("--cooldown", type=float, help="seconds between repeated actions (default 1.5)")
    p.add_argument("--headless", action="store_true", help="no preview window (for testing)")
    p.add_argument("--max-seconds", type=float, help="quit automatically after N seconds")
    p.add_argument("--snapshot-dir", type=Path,
                   help="save annotated frames here whenever a gesture fires (for debugging)")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    a = p.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    cfg = AppConfig()
    if a.list_cameras:
        for cam in list_cameras():
            print(f"{cam.index}: {cam.name}")
        raise SystemExit(0)
    cfg.camera.index, cfg.camera.width, cfg.camera.height = a.camera, a.width, a.height
    cfg.camera.name = a.camera_name
    if a.volume_gain is not None:
        cfg.volume.volume_gain = a.volume_gain
    if a.flip_palm:
        cfg.thresholds.palm_sign = -1.0
    cfg.camera.mirror = not a.no_mirror
    cfg.dry_run = a.dry_run
    cfg.players = tuple(x.strip() for x in a.players.split(",") if x.strip())
    if a.absolute_volume:
        cfg.volume.mapping = "absolute"
    if a.pinch_min is not None:
        cfg.volume.pinch_min_ratio = a.pinch_min
    if a.pinch_max is not None:
        cfg.volume.pinch_max_ratio = a.pinch_max
    if cfg.volume.pinch_max_ratio <= cfg.volume.pinch_min_ratio:
        p.error("--pinch-max must be greater than --pinch-min")
    if a.cooldown is not None:
        cfg.timing.action_cooldown_s = a.cooldown
    cfg.show_window = not a.headless
    cfg.max_seconds = a.max_seconds
    cfg.snapshot_dir = a.snapshot_dir
    return cfg


def main(argv: list[str] | None = None) -> int:
    if sys.platform != "darwin":
        print("This controller uses macOS AppleScript and only runs on macOS.", file=sys.stderr)
        return 2
    cfg = parse_args(argv)
    try:
        return run(cfg)
    except (RuntimeError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 0
