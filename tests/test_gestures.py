import numpy as np
import pytest

from gesture_controller.config import GestureThresholds
from gesture_controller.gestures import Pose, analyze_hand, palm_facing
from tests.synthetic_hand import CURLED, HALF, make_hand, pinch_hand, random_rotation

TH = GestureThresholds()


def image_of(world, mirror=False):
    """Project the synthetic hand into image coords (y down); mirror = turn it over."""
    img = np.c_[world[:, 0], -world[:, 1], world[:, 2]]
    if mirror:
        img[:, 0] *= -1
    return img


def classify(world, label="Left", mirror=False):
    return analyze_hand(world, TH, image_of(world, mirror), label)


OPEN = make_hand(thumb="out")
POSES = {
    Pose.PEACE: make_hand(ring=CURLED, pinky=CURLED, thumb="tucked"),
    Pose.PINCH: make_hand(middle=CURLED, ring=CURLED, pinky=CURLED, thumb="out"),
    Pose.FIST: make_hand(CURLED, CURLED, CURLED, CURLED, thumb="tucked"),
    Pose.THUMB_UP: make_hand(CURLED, CURLED, CURLED, CURLED, thumb="out"),
}


@pytest.mark.parametrize("expected", list(POSES))
def test_basic_poses(expected):
    assert classify(POSES[expected]).pose == expected


def test_palm_front_vs_back():
    # Thumb on the left with label "Left" = palm towards the camera (calibrated live).
    assert classify(OPEN, "Left").pose == Pose.PALM_FRONT
    assert classify(OPEN, "Left", mirror=True).pose == Pose.PALM_BACK
    # The other hand is the mirror image, so the same picture means the opposite.
    assert classify(OPEN, "Right").pose == Pose.PALM_BACK
    assert classify(OPEN, "Right", mirror=True).pose == Pose.PALM_FRONT


def test_palm_orientation_invariant_to_in_plane_rotation_and_scale():
    img = image_of(OPEN)
    for deg in range(0, 360, 30):
        a = np.radians(deg)
        r = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        rotated = img @ r.T * 0.01 + 0.5
        assert palm_facing(rotated, "Left") > 0.25
        assert palm_facing(rotated * [-1, 1, 1], "Left") < -0.25


def test_edge_on_hand_triggers_neither():
    img = image_of(OPEN)
    img[:, 0] *= 0.05                       # hand turned ~87 degrees: seen edge-on
    f = analyze_hand(OPEN, TH, img, "Left")
    assert f.pose == Pose.OPEN_HAND


def test_flip_palm_setting_swaps_sides():
    th = GestureThresholds(palm_sign=-1.0)
    assert analyze_hand(OPEN, th, image_of(OPEN), "Left").pose == Pose.PALM_BACK


def test_unknown_handedness_never_mutes_or_unmutes():
    assert analyze_hand(OPEN, TH).pose == Pose.OPEN_HAND


def test_peace_with_thumb_out_is_still_peace():
    assert classify(make_hand(ring=CURLED, pinky=CURLED, thumb="out")).pose == Pose.PEACE


def test_four_fingers_with_tucked_thumb_is_not_palm():
    assert classify(make_hand(thumb="tucked")).pose not in (Pose.PALM_FRONT, Pose.PALM_BACK)


def test_closed_pinch_is_pinch_pose():
    # Fingers start together: tips touching, index bent, others folded.
    assert classify(pinch_hand(0.0)).pose == Pose.PINCH
    assert classify(pinch_hand(1.0)).pose == Pose.PINCH


def test_pointing_with_tucked_thumb_is_not_pinch():
    hand = make_hand(middle=CURLED, ring=CURLED, pinky=CURLED, thumb="tucked")
    assert classify(hand).pose == Pose.OTHER


@pytest.mark.parametrize("expected", list(POSES))
def test_invariant_to_rotation_translation_and_scale(expected):
    rng = np.random.default_rng(len(expected.value))
    for _ in range(20):
        r = random_rotation(rng)
        hand = POSES[expected] @ r.T * rng.uniform(0.005, 0.05) + rng.normal(size=3)
        assert analyze_hand(hand, TH).pose == expected


def test_pinch_ratio_is_monotonic_in_gap():
    ratios = [analyze_hand(pinch_hand(g), TH).pinch_ratio for g in (0.0, 1.0, 3.0, 6.0, 9.0)]
    assert ratios == sorted(ratios)
    assert ratios[0] < 0.05


def test_volume_hold_rules():
    assert analyze_hand(pinch_hand(0.0), TH).volume_hold_ok
    assert analyze_hand(make_hand(index=HALF, middle=CURLED, ring=CURLED, pinky=CURLED), TH).volume_hold_ok
    assert not analyze_hand(POSES[Pose.FIST], TH).volume_hold_ok
    assert not analyze_hand(POSES[Pose.PEACE], TH).volume_hold_ok


def test_rejects_bad_shape():
    with pytest.raises(ValueError):
        analyze_hand(np.zeros((20, 3)), TH)
