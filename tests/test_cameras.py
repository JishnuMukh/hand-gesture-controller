import json
from unittest import mock

from gesture_controller import cameras

PROFILE = json.dumps({"SPCameraDataType": [
    {"_name": "MacBook Pro Camera", "spcamera_unique-id": "6C707041-05AC-0011-0009-000000000001"},
    {"_name": "iPhone (2) Camera", "spcamera_unique-id": "34498D80-2EFE-4659-88A2-B5E800000001"},
]})


def fake_run(*_a, **_k):
    return mock.Mock(stdout=PROFILE)


def test_builtin_camera_chosen_even_when_iphone_sorts_first():
    with mock.patch.object(cameras.subprocess, "run", fake_run):
        assert [c.name for c in cameras.list_cameras()] == ["iPhone (2) Camera", "MacBook Pro Camera"]
        assert cameras.resolve_camera(None, None) == (1, "MacBook Pro Camera")
        assert cameras.resolve_camera(None, "iphone") == (0, "iPhone (2) Camera")
        assert cameras.resolve_camera(0, None) == (0, "iPhone (2) Camera")


def test_no_cameras_listed_falls_back_to_index_0():
    with mock.patch.object(cameras.subprocess, "run", side_effect=OSError):
        assert cameras.resolve_camera(None, None) == (0, "camera 0")
