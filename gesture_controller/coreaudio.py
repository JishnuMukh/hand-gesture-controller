"""Direct CoreAudio access via ctypes for fast volume / mute control.

An ``osascript`` call costs ~200 ms (process start-up + AppleScript), which
caps volume updates at ~5 Hz and makes the slider feel steppy. Talking to
CoreAudio directly takes microseconds. Everything here operates on the
*current* default output device, re-resolved on every call so switching to
headphones/AirPods just works. If anything is unsupported the caller falls
back to ``osascript``.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from ctypes import POINTER, Structure, byref, c_float, c_uint32, c_void_p, sizeof


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


kAudioObjectSystemObject = 1
kAudioHardwarePropertyDefaultOutputDevice = _fourcc("dOut")
kAudioObjectPropertyScopeGlobal = _fourcc("glob")
kAudioDevicePropertyScopeOutput = _fourcc("outp")
kAudioObjectPropertyElementMain = 0
kAudioHardwareServiceDeviceProperty_VirtualMainVolume = _fourcc("vmvc")
kAudioDevicePropertyMute = _fourcc("mute")


class AudioObjectPropertyAddress(Structure):
    _fields_ = [("mSelector", c_uint32), ("mScope", c_uint32), ("mElement", c_uint32)]


class CoreAudioError(RuntimeError):
    pass


class CoreAudio:
    def __init__(self) -> None:
        ca_path = ctypes.util.find_library("CoreAudio")
        tb_path = ctypes.util.find_library("AudioToolbox")
        if not ca_path or not tb_path:
            raise CoreAudioError("CoreAudio frameworks not found")
        ca = ctypes.CDLL(ca_path)
        tb = ctypes.CDLL(tb_path)

        addr_p = POINTER(AudioObjectPropertyAddress)
        sig_get = [c_uint32, addr_p, c_uint32, c_void_p, POINTER(c_uint32), c_void_p]
        sig_set = [c_uint32, addr_p, c_uint32, c_void_p, c_uint32, c_void_p]

        self._obj_get = ca.AudioObjectGetPropertyData
        self._obj_get.argtypes, self._obj_get.restype = sig_get, ctypes.c_int32
        self._obj_set = ca.AudioObjectSetPropertyData
        self._obj_set.argtypes, self._obj_set.restype = sig_set, ctypes.c_int32
        self._obj_has = ca.AudioObjectHasProperty
        self._obj_has.argtypes, self._obj_has.restype = [c_uint32, addr_p], ctypes.c_ubyte

        # The "virtual main volume" is what the menu-bar slider and
        # `set volume output volume` control; it lives in AudioToolbox.
        self._svc_get = tb.AudioHardwareServiceGetPropertyData
        self._svc_get.argtypes, self._svc_get.restype = sig_get, ctypes.c_int32
        self._svc_set = tb.AudioHardwareServiceSetPropertyData
        self._svc_set.argtypes, self._svc_set.restype = sig_set, ctypes.c_int32
        self._svc_has = tb.AudioHardwareServiceHasProperty
        self._svc_has.argtypes, self._svc_has.restype = [c_uint32, addr_p], ctypes.c_ubyte

        self.default_output_device()  # fail fast if unusable

    # ------------------------------------------------------------------ device
    def default_output_device(self) -> int:
        addr = AudioObjectPropertyAddress(kAudioHardwarePropertyDefaultOutputDevice,
                                          kAudioObjectPropertyScopeGlobal,
                                          kAudioObjectPropertyElementMain)
        dev, size = c_uint32(0), c_uint32(sizeof(c_uint32))
        status = self._obj_get(kAudioObjectSystemObject, byref(addr), 0, None, byref(size), byref(dev))
        if status != 0 or dev.value == 0:
            raise CoreAudioError(f"no default output device (status {status})")
        return dev.value

    @staticmethod
    def _out_addr(selector: int) -> AudioObjectPropertyAddress:
        return AudioObjectPropertyAddress(selector, kAudioDevicePropertyScopeOutput,
                                          kAudioObjectPropertyElementMain)

    # ------------------------------------------------------------------ volume
    def has_volume(self) -> bool:
        addr = self._out_addr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume)
        return bool(self._svc_has(self.default_output_device(), byref(addr)))

    def get_volume(self) -> int:
        addr = self._out_addr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume)
        val, size = c_float(0), c_uint32(sizeof(c_float))
        status = self._svc_get(self.default_output_device(), byref(addr), 0, None, byref(size), byref(val))
        if status != 0:
            raise CoreAudioError(f"get volume failed (status {status})")
        return int(round(val.value * 100))

    def set_volume(self, level: int) -> None:
        addr = self._out_addr(kAudioHardwareServiceDeviceProperty_VirtualMainVolume)
        val = c_float(max(0, min(100, int(level))) / 100.0)
        status = self._svc_set(self.default_output_device(), byref(addr), 0, None, sizeof(c_float), byref(val))
        if status != 0:
            raise CoreAudioError(f"set volume failed (status {status})")

    # -------------------------------------------------------------------- mute
    def has_mute(self) -> bool:
        addr = self._out_addr(kAudioDevicePropertyMute)
        return bool(self._obj_has(self.default_output_device(), byref(addr)))

    def get_mute(self) -> bool:
        addr = self._out_addr(kAudioDevicePropertyMute)
        val, size = c_uint32(0), c_uint32(sizeof(c_uint32))
        status = self._obj_get(self.default_output_device(), byref(addr), 0, None, byref(size), byref(val))
        if status != 0:
            raise CoreAudioError(f"get mute failed (status {status})")
        return bool(val.value)

    def set_mute(self, muted: bool) -> None:
        addr = self._out_addr(kAudioDevicePropertyMute)
        val = c_uint32(1 if muted else 0)
        status = self._obj_set(self.default_output_device(), byref(addr), 0, None, sizeof(c_uint32), byref(val))
        if status != 0:
            raise CoreAudioError(f"set mute failed (status {status})")


def load() -> CoreAudio | None:
    try:
        return CoreAudio()
    except (OSError, AttributeError, CoreAudioError):
        return None
