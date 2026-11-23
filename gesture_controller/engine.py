"""Stateful gesture engine: turns per-frame poses into debounced actions.

Responsibilities
* Discrete gestures (peace sign, palm front, back of hand) must be held for ``hold_s`` before
  they fire, fire only once per "press", and respect per-gesture and global
  cooldowns. Short tracking dropouts (< ``release_gap_s``) do not count as a
  release, so a flickering detection can never double-fire.
* Volume mode is engaged by holding the pinch pose (thumb + index, other
  fingers folded) for ``volume_engage_s``. While engaged, thumb-index distance
  drives the volume through a One-Euro filter and a slew limiter. Entering
  volume mode never makes the volume jump: in "relative" mapping only the
  change in gap since engaging counts (spread = louder by that amount, close =
  quieter by that amount); in "absolute" mapping the volume is only grabbed
  (soft takeover) once the hand's level reaches the current volume.
  On release, the level from ``release_rollback_s`` earlier is restored so the
  motion of letting go does not change the volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .config import TimingConfig, VolumeConfig
from .gestures import HandFeatures, Pose
from .smoothing import OneEuroFilter, VolumeSmoother


class ActionType(str, Enum):
    SET_VOLUME = "set_volume"
    MUTE = "mute"
    UNMUTE = "unmute"
    PLAY_PAUSE = "play_pause"


@dataclass(frozen=True)
class Action:
    type: ActionType
    value: int | None = None


DISCRETE_ACTIONS = {
    Pose.PEACE: ActionType.PLAY_PAUSE,
    Pose.PALM_FRONT: ActionType.MUTE,
    Pose.PALM_BACK: ActionType.UNMUTE,
}


@dataclass
class _Press:
    start: float
    last_seen: float
    fired: bool = False


@dataclass
class EngineOutput:
    pose: Pose
    volume_mode: bool = False
    grabbed: bool = False                 # volume follows the hand (pickup reached)
    engage_progress: float = 0.0          # 0..1 while holding the pinch pose
    hold_progress: float = 0.0            # 0..1 while holding a discrete gesture
    pending: Pose | None = None           # discrete gesture currently being held
    blocked_by_cooldown: bool = False
    pinch_ratio: float | None = None
    target_volume: float | None = None    # raw mapped level
    smoothed_volume: float | None = None
    actions: list[Action] = field(default_factory=list)


class GestureEngine:
    def __init__(self, timing: TimingConfig, volume: VolumeConfig) -> None:
        self.t = timing
        self.v = volume
        self.smoother = VolumeSmoother(volume.max_rate_per_s)
        self.target_filter = OneEuroFilter(volume.filter_min_cutoff_hz, volume.filter_beta)
        self._grabbed = False
        self._pickup_sign = 0.0
        self._anchor = (50.0, 50.0)   # (hand level, volume) pinned at engage time
        self._presses: dict[Pose, _Press] = {}
        self._last_fire: dict[ActionType, float] = {}
        self._last_any_fire = float("-inf")
        self._engaged = False
        self._l_since: float | None = None
        self._fail_since: float | None = None
        self._last_sent: int | None = None

    # ------------------------------------------------------------------ public
    @property
    def volume_mode(self) -> bool:
        return self._engaged

    def cooldown_remaining(self, action: ActionType, now: float) -> float:
        last = self._last_fire.get(action, float("-inf"))
        per = self.t.action_cooldown_s - (now - last)
        glob = self.t.global_cooldown_s - (now - self._last_any_fire)
        return max(0.0, per, glob)

    def update(self, feat: HandFeatures | None, now: float,
               current_volume: float | None) -> EngineOutput:
        pose = feat.pose if feat else Pose.NONE
        out = EngineOutput(pose=pose)

        if self._engaged:
            self._update_volume_mode(feat, now, out)
            if self._engaged:
                # Discrete gestures are ignored while adjusting volume.
                self._presses.clear()
                return out

        self._update_engagement(feat, now, current_volume, out)
        if not self._engaged:
            self._update_discrete(pose, now, out)
        return out

    # ----------------------------------------------------------------- volume
    def _update_engagement(self, feat: HandFeatures | None, now: float,
                           current_volume: float | None, out: EngineOutput) -> None:
        if feat is None or feat.pose != Pose.PINCH:
            self._l_since = None
            return
        if self._l_since is None:
            self._l_since = now
        held = now - self._l_since
        out.engage_progress = min(1.0, held / self.t.volume_engage_s) if self.t.volume_engage_s else 1.0
        out.pinch_ratio = feat.pinch_ratio
        if held >= self.t.volume_engage_s:
            self._engaged = True
            self._fail_since = None
            self.target_filter.reset()
            hand = self.target_filter(self._hand_level(feat), now)
            start = current_volume if current_volume is not None else min(100.0, max(0.0, hand))
            self.smoother.reset(start, now)
            self._last_sent = round(start)
            self._anchor = (hand, start)
            if self.v.mapping == "relative":
                self._grabbed = True
            else:
                gap = self._map(hand) - start
                self._grabbed = current_volume is None or abs(gap) <= self.v.pickup_tolerance
                self._pickup_sign = math.copysign(1.0, gap)
            self._apply_volume(feat, now, out, prefiltered=hand)

    def _update_volume_mode(self, feat: HandFeatures | None, now: float,
                            out: EngineOutput) -> None:
        out.volume_mode = True
        if feat is not None and feat.volume_hold_ok:
            self._fail_since = None
            self._apply_volume(feat, now, out)
            return

        # Pose broken (or hand lost): freeze, and leave after a grace period.
        if self._fail_since is None:
            self._fail_since = now
        out.smoothed_volume = self.smoother.value
        if feat is None or now - self._fail_since >= self.t.volume_disengage_s:
            self._disengage(now, out)

    def _apply_volume(self, feat: HandFeatures, now: float, out: EngineOutput,
                      prefiltered: float | None = None) -> None:
        hand = prefiltered if prefiltered is not None else self.target_filter(self._hand_level(feat), now)
        if self.v.mapping == "relative":
            # Ratchet: once pushed past 0 % or 100 %, re-anchor at the limit so
            # reversing direction responds immediately (no dead travel).
            h0, v0 = self._anchor
            raw = v0 + self.v.volume_gain * (hand - h0)
            if raw > 100.0:
                self._anchor = (hand, 100.0)
            elif raw < 0.0:
                self._anchor = (hand, 0.0)
        target = self._map(hand)
        out.volume_mode = True
        out.pinch_ratio = feat.pinch_ratio
        out.target_volume = target
        if not self._grabbed:
            held = self.smoother.value if self.smoother.value is not None else target
            gap = target - held
            # Grab when the hand's level reaches (or sweeps past) the current level.
            if abs(gap) <= self.v.pickup_tolerance or math.copysign(1.0, gap) != self._pickup_sign:
                self._grabbed = True
                self.smoother.reset(held, now)
            else:
                out.smoothed_volume = held
                return
        out.grabbed = True
        out.smoothed_volume = self.smoother.update(target, now)
        self._emit_volume(out.smoothed_volume, out)

    def _disengage(self, now: float, out: EngineOutput) -> None:
        # Restore the level from just before the hand started to let go.
        cutoff = (self._fail_since if self._fail_since is not None else now) - self.v.release_rollback_s
        restored = self.smoother.value_at(cutoff)
        if restored is not None:
            self._emit_volume(restored, out, force=True)
            out.smoothed_volume = restored
        self._engaged = False
        self._grabbed = False
        self._fail_since = None
        self._l_since = None
        out.volume_mode = False

    def _emit_volume(self, value: float, out: EngineOutput, force: bool = False) -> None:
        level = int(round(value))
        changed = level != self._last_sent
        if self._last_sent is None or abs(level - self._last_sent) >= self.v.deadband or (
            changed and (force or level in (0, 100))   # extremes must always be reachable
        ):
            self._last_sent = level
            out.actions.append(Action(ActionType.SET_VOLUME, level))

    def _hand_level(self, feat: HandFeatures) -> float:
        """Thumb-index gap on the 0-100 scale (unclamped, so relative mode sees all travel)."""
        lo, hi = self.v.pinch_min_ratio, self.v.pinch_max_ratio
        return 100.0 * (feat.pinch_ratio - lo) / (hi - lo)

    def _map(self, hand: float) -> float:
        """Hand level -> volume level, according to the configured mapping."""
        if self.v.mapping != "relative":
            return min(100.0, max(0.0, hand))
        h0, v0 = self._anchor
        return min(100.0, max(0.0, v0 + self.v.volume_gain * (hand - h0)))

    # --------------------------------------------------------------- discrete
    def _update_discrete(self, pose: Pose, now: float, out: EngineOutput) -> None:
        # Forget presses whose gesture has been absent longer than the gap.
        for g in list(self._presses):
            if g != pose and now - self._presses[g].last_seen > self.t.release_gap_s:
                del self._presses[g]

        action = DISCRETE_ACTIONS.get(pose)
        if action is None:
            return
        press = self._presses.get(pose)
        if press is None:
            press = self._presses[pose] = _Press(start=now, last_seen=now)
        press.last_seen = now

        out.pending = pose
        held = now - press.start
        out.hold_progress = 1.0 if press.fired else min(1.0, held / self.t.hold_s if self.t.hold_s else 1.0)
        if press.fired or held < self.t.hold_s:
            return
        if self.cooldown_remaining(action, now) > 0:
            out.blocked_by_cooldown = True
            return
        press.fired = True
        self._last_fire[action] = now
        self._last_any_fire = now
        out.actions.append(Action(action))
