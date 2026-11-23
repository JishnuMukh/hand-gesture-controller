"""HUD rendering with OpenCV: skeleton, gesture panel, volume bar, toasts."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from .engine import ActionType, EngineOutput, GestureEngine
from .gestures import POSE_LABELS, Pose
from .macos import SystemState
from .tracker import HAND_CONNECTIONS, HandObservation

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# BGR colours
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (25, 25, 25)
GREEN = (110, 220, 90)
AMBER = (40, 190, 250)
RED = (70, 70, 235)
CYAN = (230, 200, 60)
PANEL = (30, 30, 30)

FINGER_COLOURS = {  # by landmark range
    "thumb": (90, 180, 255), "index": (110, 220, 90), "middle": (230, 200, 60),
    "ring": (220, 120, 200), "pinky": (120, 140, 255),
}


def _finger_of(idx: int) -> str:
    if 1 <= idx <= 4:
        return "thumb"
    if 5 <= idx <= 8:
        return "index"
    if 9 <= idx <= 12:
        return "middle"
    if 13 <= idx <= 16:
        return "ring"
    if 17 <= idx <= 20:
        return "pinky"
    return "palm"


GESTURE_HINTS = {
    Pose.PEACE: "Play / Pause",
    Pose.PALM_FRONT: "Mute",
    Pose.PALM_BACK: "Unmute",
    Pose.OPEN_HAND: "Face palm or back to the camera",
    Pose.PINCH: "Hold still to start volume control",
}


@dataclass
class Toast:
    text: str
    at: float
    duration: float = 2.0


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.6,
           colour: tuple[int, int, int] = PANEL) -> None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    overlay = np.full_like(roi, colour)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def _text(img, text, org, scale=0.6, colour=WHITE, thick=1):
    # Outline via offsets at the same thickness: in OpenCV a thicker stroke also
    # widens glyph advance, which would make a thick outline drift from the fill.
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 2), (2, 0)):
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, DARK, thick, AA)
    cv2.putText(img, text, org, FONT, scale, colour, thick, AA)


def _lerp_colour(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


class Overlay:
    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.toast: Toast | None = None
        self.show_help = True

    def notify(self, text: str) -> None:
        self.toast = Toast(text, time.monotonic())

    # ------------------------------------------------------------------ render
    def render(self, frame: np.ndarray, hand: HandObservation | None, out: EngineOutput,
               engine: GestureEngine, system: SystemState, fps: float,
               controls_enabled: bool) -> np.ndarray:
        img = frame
        H, W = img.shape[:2]
        now = time.monotonic()
        scale = max(0.6, W / 1280)

        if hand is not None:
            pts = hand.pixel_points(W, H)
            self._draw_skeleton(img, pts, scale)
            if out.volume_mode or out.pose == Pose.PINCH:
                self._draw_pinch(img, pts, out, scale)

        self._draw_gesture_panel(img, out, hand, scale)
        self._draw_volume_bar(img, out, system, scale)
        self._draw_cooldowns(img, engine, now, scale)
        self._draw_footer(img, fps, controls_enabled, scale)
        if self.show_help:
            self._draw_help(img, scale)
        self._draw_toast(img, now, scale)
        return img

    def _draw_skeleton(self, img, pts, s):
        for a, b in HAND_CONNECTIONS:
            same_finger = _finger_of(a) == _finger_of(b) or (a, b) == (0, 1)
            colour = FINGER_COLOURS[_finger_of(b)] if same_finger else (200, 200, 200)
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), DARK, int(6 * s), AA)
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), colour, int(3 * s), AA)
        for i, p in enumerate(pts):
            tip = i in (4, 8, 12, 16, 20)
            r = int((7 if tip else 4) * s)
            cv2.circle(img, tuple(p), r + 2, DARK, -1, AA)
            cv2.circle(img, tuple(p), r, FINGER_COLOURS.get(_finger_of(i), WHITE), -1, AA)

    def _draw_pinch(self, img, pts, out: EngineOutput, s):
        a, b = tuple(pts[4]), tuple(pts[8])
        level = (out.target_volume if out.target_volume is not None else 0) / 100
        colour = _lerp_colour(RED, GREEN, level) if out.grabbed else GREY
        cv2.line(img, a, b, DARK, int(7 * s), AA)
        cv2.line(img, a, b, colour, int(4 * s), AA)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        cv2.circle(img, mid, int(10 * s), colour, -1, AA)
        if out.volume_mode and out.target_volume is not None:
            label = f"{out.smoothed_volume:.0f}%" if out.grabbed else f"{out.target_volume:.0f}%"
            _text(img, label, (mid[0] + int(16 * s), mid[1] - int(10 * s)), 0.8 * s, colour, 2)
        elif out.engage_progress > 0:
            # Ring that fills while the L-shape is held long enough to engage.
            cv2.ellipse(img, mid, (int(22 * s), int(22 * s)), -90, 0,
                        360 * out.engage_progress, AMBER, int(4 * s), AA)

    def _draw_gesture_panel(self, img, out: EngineOutput, hand, s):
        x, y, w, h = int(16 * s), int(16 * s), int(360 * s), int(128 * s)
        _panel(img, x, y, w, h)
        if out.volume_mode and out.grabbed:
            title, colour = "Pinch: volume", GREEN
            sub = "Spread = louder, close = quieter. Drop hand to set"
        elif out.volume_mode:
            title, colour = "Pinch: volume", AMBER
            cur = "?" if out.smoothed_volume is None else f"{out.smoothed_volume:.0f}%"
            sub = f"Move to current level ({cur}) to grab"
        else:
            title = POSE_LABELS.get(out.pose, out.pose.value)
            colour = AMBER if out.pending else (WHITE if out.pose != Pose.NONE else GREY)
            sub = GESTURE_HINTS.get(out.pose, "")
            if out.pose == Pose.PINCH:
                sub = "Hold still to start volume control"
            if out.blocked_by_cooldown:
                sub += "  (cooling down)"
        _text(img, "GESTURE", (x + int(14 * s), y + int(28 * s)), 0.5 * s, GREY)
        _text(img, title, (x + int(14 * s), y + int(64 * s)), 0.95 * s, colour, 2)
        _text(img, sub, (x + int(14 * s), y + int(92 * s)), 0.52 * s, WHITE)

        bar_x, bar_y, bar_w = x + int(14 * s), y + int(106 * s), w - int(28 * s)
        progress = out.hold_progress if out.pending else out.engage_progress
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + int(8 * s)), (70, 70, 70), -1)
        if out.volume_mode:
            progress, pcol = 1.0, (GREEN if out.grabbed else AMBER)
        else:
            pcol = RED if out.blocked_by_cooldown else AMBER
        if progress > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + int(8 * s)),
                          pcol, -1)
        if hand is not None:
            _text(img, f"{hand.handedness} hand", (x + w - int(110 * s), y + int(28 * s)),
                  0.45 * s, GREY)

    def _draw_volume_bar(self, img, out: EngineOutput, system: SystemState, s):
        H, W = img.shape[:2]
        bw, bh = int(46 * s), int(300 * s)
        x, y = W - bw - int(30 * s), int(H / 2 - bh / 2)
        _panel(img, x - int(14 * s), y - int(44 * s), bw + int(28 * s), bh + int(96 * s))
        vol = system.volume
        if out.volume_mode and out.grabbed and out.smoothed_volume is not None:
            vol = round(out.smoothed_volume)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (70, 70, 70), -1)
        if vol is not None:
            fill = int(bh * vol / 100)
            colour = GREY if system.muted else (GREEN if out.grabbed else CYAN)
            cv2.rectangle(img, (x, y + bh - fill), (x + bw, y + bh), colour, -1)
        if out.volume_mode and out.target_volume is not None:
            # Marker for where the hand currently points.
            ty = y + bh - int(bh * out.target_volume / 100)
            mcol = WHITE if out.grabbed else AMBER
            cv2.line(img, (x - int(10 * s), ty), (x + bw + int(10 * s), ty), mcol, 2, AA)
            tri = np.array([[x - int(10 * s), ty - int(7 * s)], [x - int(10 * s), ty + int(7 * s)],
                            [x - int(2 * s), ty]], np.int32)
            cv2.fillPoly(img, [tri], mcol, AA)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), WHITE, 1)
        _text(img, "VOL", (x + int(6 * s), y - int(16 * s)), 0.55 * s, GREY)
        label = "--" if vol is None else f"{vol}%"
        _text(img, label, (x - int(4 * s), y + bh + int(28 * s)), 0.65 * s, WHITE, 2)
        if system.muted:
            _panel(img, x - int(18 * s), y + bh // 2 - int(18 * s), bw + int(36 * s), int(34 * s),
                   0.85, (40, 40, 160))
            _text(img, "MUTED", (x - int(10 * s), y + bh // 2 + int(6 * s)), 0.6 * s, WHITE, 2)

    def _draw_cooldowns(self, img, engine: GestureEngine, now: float, s):
        x, y = int(16 * s), int(160 * s)
        for label, action in (("Play/Pause", ActionType.PLAY_PAUSE), ("Mute", ActionType.MUTE),
                              ("Unmute", ActionType.UNMUTE)):
            remaining = engine.cooldown_remaining(action, now)
            w = int(170 * s)
            _panel(img, x, y, w, int(30 * s), 0.5)
            if remaining > 0:
                frac = remaining / max(engine.t.action_cooldown_s, 1e-6)
                cv2.rectangle(img, (x, y + int(26 * s)), (x + int(w * min(frac, 1)), y + int(30 * s)),
                              RED, -1)
                status, colour = f"{remaining:.1f}s", RED
            else:
                status, colour = "ready", GREEN
            _text(img, label, (x + int(8 * s), y + int(20 * s)), 0.48 * s, WHITE)
            _text(img, status, (x + w - int(56 * s), y + int(20 * s)), 0.48 * s, colour)
            y += int(36 * s)

    def _draw_footer(self, img, fps, enabled, s):
        H, W = img.shape[:2]
        _panel(img, 0, H - int(32 * s), W, int(32 * s), 0.55)
        txt = f"{fps:4.1f} FPS"
        if self.dry_run:
            txt += "   DRY RUN (no system changes)"
        if not enabled:
            txt += "   CONTROLS PAUSED (space to resume)"
        _text(img, txt, (int(12 * s), H - int(10 * s)), 0.5 * s, AMBER if not enabled else WHITE)
        keys = "q quit   space pause   h help"
        _text(img, keys, (W - int(290 * s), H - int(10 * s)), 0.5 * s, GREY)

    def _draw_help(self, img, s):
        H, W = img.shape[:2]
        lines = ["Pinch (other fingers folded), hold, then spread = louder / close = quieter",
                 "Peace sign -> play / pause",
                 "Palm facing camera -> mute      Back of hand -> unmute"]
        x, y = int(16 * s), H - int(32 * s) - int(24 * s) * len(lines) - int(18 * s)
        _panel(img, x, y, int(640 * s), int(24 * s) * len(lines) + int(12 * s), 0.5)
        for i, line in enumerate(lines):
            _text(img, line, (x + int(10 * s), y + int(22 * s) + i * int(24 * s)), 0.5 * s, WHITE)

    def _draw_toast(self, img, now, s):
        if self.toast is None:
            return
        age = now - self.toast.at
        if age > self.toast.duration:
            self.toast = None
            return
        H, W = img.shape[:2]
        alpha = 0.75 if age < self.toast.duration - 0.4 else 0.75 * (self.toast.duration - age) / 0.4
        (tw, th), _ = cv2.getTextSize(self.toast.text, FONT, 0.9 * s, 2)
        x, y = (W - tw) // 2, int(70 * s)
        _panel(img, x - int(20 * s), y - th - int(16 * s), tw + int(40 * s), th + int(32 * s),
               alpha, (60, 60, 60))
        _text(img, self.toast.text, (x, y), 0.9 * s, WHITE, 2)
