import numpy as np

from gesture_controller.config import GestureThresholds, TimingConfig, VolumeConfig
from gesture_controller.engine import ActionType, GestureEngine
from gesture_controller.gestures import HandFeatures, Pose, analyze_hand
from gesture_controller.smoothing import OneEuroFilter, VolumeSmoother, map_ratio_to_volume
from tests.synthetic_hand import CURLED, make_hand, pinch_hand
from tests.test_gestures import image_of

TH = GestureThresholds()
VC = VolumeConfig()
DT = 1 / 30
OPEN = make_hand(thumb="out")
PEACE = analyze_hand(make_hand(ring=CURLED, pinky=CURLED, thumb="tucked"), TH)
PALM = analyze_hand(OPEN, TH, image_of(OPEN), "Left")
BACK = analyze_hand(OPEN, TH, image_of(OPEN, mirror=True), "Left")
FIST = analyze_hand(make_hand(CURLED, CURLED, CURLED, CURLED, thumb="tucked"), TH)
CLOSED = analyze_hand(pinch_hand(0.0), TH)          # fingertips together
assert PALM.pose == Pose.PALM_FRONT and BACK.pose == Pose.PALM_BACK and CLOSED.pose == Pose.PINCH


def at_level(level):
    """Pinch features whose thumb-index gap sits at ``level`` on the 0-100 hand scale."""
    ratio = VC.pinch_min_ratio + level / 100 * (VC.pinch_max_ratio - VC.pinch_min_ratio)
    return HandFeatures(Pose.PINCH, CLOSED.extension, CLOSED.extended, CLOSED.thumb_spread,
                        ratio, True)


def drive(engine, frames, t0=0.0, volume=50.0):
    """Feed (features, n_frames) pairs; return (actions, end_time, last_output)."""
    actions, t, out = [], t0, None
    for feat, n in frames:
        for _ in range(n):
            out = engine.update(feat, t, volume)
            actions.extend(out.actions)
            t += DT
    return actions, t, out


def ramp(a, b, n):
    return [(at_level(a + (b - a) * i / (n - 1)), 1) for i in range(n)]


def kinds(actions):
    return [a.type for a in actions if a.type is not ActionType.SET_VOLUME]


def volumes(actions):
    return [a.value for a in actions if a.type is ActionType.SET_VOLUME]


def new_engine(volume_cfg=None, **timing):
    return GestureEngine(TimingConfig(**timing), volume_cfg or VolumeConfig())


# ------------------------------------------------------------ discrete gestures
def test_peace_requires_hold_and_toggles_once():
    e = new_engine()
    acts, t, _ = drive(e, [(PEACE, 5)])            # ~0.17 s < hold
    assert kinds(acts) == []
    acts, _, _ = drive(e, [(PEACE, 120)], t0=t)    # keep holding 4 s
    assert kinds(acts) == [ActionType.PLAY_PAUSE]


def test_palm_front_mutes_back_unmutes():
    e = new_engine()
    acts, t, _ = drive(e, [(PALM, 20)])
    assert kinds(acts) == [ActionType.MUTE]
    acts, _, _ = drive(e, [(BACK, 40)], t0=t)      # turn the hand over
    assert kinds(acts) == [ActionType.UNMUTE]


def test_turning_hand_over_quickly_still_fires_after_global_cooldown():
    e = new_engine(global_cooldown_s=0.8)
    acts, _, _ = drive(e, [(PALM, 12), (BACK, 12)])  # back shown right after mute
    assert kinds(acts) == [ActionType.MUTE]          # blocked by the global cooldown...
    acts, _, _ = drive(e, [(PALM, 12), (BACK, 30)])
    assert ActionType.UNMUTE in kinds(acts)          # ...but fires once it expires


def test_flicker_does_not_refire():
    e = new_engine(action_cooldown_s=0.0, global_cooldown_s=0.0)
    seq = [(PEACE, 15), (None, 3), (PEACE, 15), (FIST, 2), (PEACE, 15)]  # gaps < 0.25 s
    acts, _, _ = drive(e, seq)
    assert kinds(acts) == [ActionType.PLAY_PAUSE]


def test_cooldown_blocks_quick_repeat_then_allows():
    e = new_engine()
    acts, t, out = drive(e, [(PEACE, 12), (None, 12), (PEACE, 12)])
    assert kinds(acts) == [ActionType.PLAY_PAUSE]
    assert out.blocked_by_cooldown
    acts, _, _ = drive(e, [(None, 30), (PEACE, 12)], t0=t)
    assert kinds(acts) == [ActionType.PLAY_PAUSE]


# ---------------------------------------------------------- relative volume
def engaged(level, volume, cfg=None):
    e = new_engine(cfg)
    _, t, out = drive(e, [(at_level(level), 12)], volume=volume)
    assert out.volume_mode
    return e, t


def test_engaging_never_changes_volume():
    e = new_engine()
    acts, _, out = drive(e, [(CLOSED, 45)], volume=30)
    assert out.volume_mode and out.grabbed and volumes(acts) == []


def test_fingers_together_then_apart_raises_by_amount_spread():
    e, t = engaged(0, volume=30)
    acts, t, out = drive(e, ramp(0, 40, 20) + [(at_level(40), 40)], t0=t, volume=30)
    vols = volumes(acts)
    assert vols == sorted(vols)                    # only ever goes up
    assert abs(vols[-1] - 70) <= 2                 # 30 % + 40 % of travel


def test_fingers_apart_then_together_lowers_by_amount_closed():
    e, t = engaged(80, volume=60)
    acts, t, out = drive(e, ramp(80, 55, 20) + [(at_level(55), 40)], t0=t, volume=60)
    vols = volumes(acts)
    assert vols == sorted(vols, reverse=True)
    assert abs(vols[-1] - 35) <= 2                 # 60 % - 25 % of travel


def test_volume_gain_scales_change():
    e, t = engaged(0, volume=20, cfg=VolumeConfig(volume_gain=0.5))
    acts, _, _ = drive(e, ramp(0, 40, 20) + [(at_level(40), 40)], t0=t, volume=20)
    assert abs(volumes(acts)[-1] - 40) <= 2


def test_ratchet_at_limit_reverses_immediately():
    e, t = engaged(0, volume=90)
    acts, t, _ = drive(e, ramp(0, 60, 20) + [(at_level(60), 30)], t0=t)
    assert volumes(acts)[-1] == 100
    acts, _, _ = drive(e, ramp(60, 50, 10) + [(at_level(50), 40)], t0=t)
    assert abs(volumes(acts)[-1] - 90) <= 2        # 10 % back from where it hit the limit


def test_changes_are_smooth():
    e, t = engaged(0, volume=0)
    acts, _, _ = drive(e, [(at_level(100), 60)], t0=t, volume=0)   # instant jump of the hand
    vols = volumes(acts)
    assert vols[-1] == 100
    assert max(b - a for a, b in zip([0] + vols, vols)) <= 6        # slew-limited


def test_release_restores_level_from_before_letting_go():
    e, t = engaged(20, volume=50)
    acts, t, out = drive(e, ramp(20, 40, 15) + [(at_level(40), 45)], t0=t)
    level = out.smoothed_volume
    # Letting go: the fingers fly apart for a couple of frames, then the pose breaks.
    acts, t, out = drive(e, [(at_level(90), 2), (PALM, 10)], t0=t)
    assert not e.volume_mode
    assert abs(volumes(acts)[-1] - level) <= 2
    assert ActionType.MUTE not in kinds(acts)      # palm right after release needs a full hold


def test_hand_lost_exits_volume_mode():
    e, t = engaged(0, volume=50)
    _, _, out = drive(e, [(None, 1)], t0=t)
    assert not out.volume_mode


def test_discrete_gestures_ignored_in_volume_mode():
    e, t = engaged(0, volume=50)
    acts, _, _ = drive(e, [(PEACE, 3)], t0=t)
    assert kinds(acts) == []


def test_jitter_is_suppressed_when_hand_is_still():
    e, t = engaged(50, volume=50)
    rng = np.random.default_rng(0)
    noisy = [(at_level(50 + rng.normal(0, 4)), 1) for _ in range(90)]   # sigma 4 %
    _, t, _ = drive(e, noisy[:30], t0=t)
    acts, _, _ = drive(e, noisy[30:], t0=t)
    vols = volumes(acts)
    assert not vols or max(vols) - min(vols) <= 4, vols


def test_extremes_reachable_despite_deadband():
    e, t = engaged(0, volume=99)
    acts, t, _ = drive(e, [(at_level(30), 60)], t0=t)
    assert e._last_sent == 100
    acts, _, _ = drive(e, [(at_level(-200), 120)], t0=t)
    assert volumes(acts)[-1] == 0


# ---------------------------------------------------------- absolute mode
ABSOLUTE = VolumeConfig(mapping="absolute")


def test_absolute_pickup_engaging_never_jumps_volume():
    e = new_engine(ABSOLUTE)
    acts, _, out = drive(e, [(at_level(95), 40)], volume=30)
    assert out.volume_mode and not out.grabbed and volumes(acts) == []


def test_absolute_pickup_grabs_when_level_is_crossed_then_follows():
    e = new_engine(ABSOLUTE)
    _, t, _ = drive(e, [(at_level(95), 12)], volume=30)
    acts, _, out = drive(e, [(CLOSED, 45)], t0=t, volume=30)
    assert out.grabbed
    vols = volumes(acts)
    assert vols[0] >= 25 and vols[-1] == 0


# ---------------------------------------------------------- filters
def test_one_euro_tracks_fast_motion_with_little_lag():
    f = OneEuroFilter(min_cutoff_hz=0.5, beta=0.01)
    t, x = 0.0, 0.0
    for i in range(30):                            # sweep 0 -> 100 in 0.5 s
        x = f(min(100.0, i * 100 / 15), t)
        t += DT
    assert x > 90


def test_smoother_slew_limit_and_mapping():
    s = VolumeSmoother(max_rate_per_s=100)
    s.reset(0, 0.0)
    assert s.update(100, 0.1) == 10.0
    assert map_ratio_to_volume(0.0, 0.2, 1.0) == 0
    assert map_ratio_to_volume(2.0, 0.2, 1.0) == 100
    assert abs(map_ratio_to_volume(0.6, 0.2, 1.0) - 50) < 1e-9
