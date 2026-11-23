"""macOS system control.

* Volume and mute: CoreAudio (via ctypes, ~5 ms per call) when available,
  otherwise ``osascript`` (~200 ms per call).
* Play/Pause: ``osascript`` talking to Spotify or Apple Music.

All calls run on one background worker thread so they never stall the video
loop. Volume requests are coalesced (latest value wins); discrete commands are
queued in order.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from . import coreaudio

log = logging.getLogger(__name__)

OSASCRIPT_TIMEOUT_S = 5.0
POLL_INTERVAL_S = 0.5          # refresh volume/mute so changes made elsewhere show up
POLL_QUIET_AFTER_SET_S = 1.0
# The first command to a player can wait on macOS's Automation permission
# prompt; give the user time to answer it (this runs off the video thread).
PLAYER_TIMEOUT_S = 60.0   # don't poll right after we set the volume ourselves


class AppleScriptError(RuntimeError):
    pass


def run_osascript(script: str, timeout: float = OSASCRIPT_TIMEOUT_S) -> str:
    try:
        proc = subprocess.run(
            ["osascript", "-e", script], capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise AppleScriptError(f"osascript timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise AppleScriptError(proc.stderr.strip() or f"osascript exited {proc.returncode}")
    return proc.stdout.strip()


# ----------------------------------------------------------------- primitives
def get_volume_state() -> tuple[int | None, bool]:
    """Return (output volume 0-100 or None if the device has no volume, muted)."""
    out = run_osascript(
        'set s to (get volume settings)\n'
        'return ((output volume of s) as text) & "|" & ((output muted of s) as text)'
    )
    vol_s, _, muted_s = out.partition("|")
    volume = int(vol_s) if vol_s.strip().lstrip("-").isdigit() else None
    return volume, muted_s.strip() == "true"


def set_volume(level: int) -> None:
    level = max(0, min(100, int(level)))
    run_osascript(f"set volume output volume {level}")


def _is_running(app: str) -> bool:
    # pgrep needs no Automation permission and never launches the app.
    return subprocess.run(["pgrep", "-xq", app], capture_output=True).returncode == 0


def _player_state(app: str) -> str:
    return run_osascript(f'tell application "{app}" to return (player state as text)',
                         timeout=PLAYER_TIMEOUT_S)


def play_pause(players: tuple[str, ...]) -> str:
    """Pause whatever is playing; if nothing is playing, resume the preferred player.

    Uses explicit ``play``/``pause`` commands chosen from the state read first,
    so the result is predictable even if the player updates its state slowly.
    Never launches a player. Returns a status message.
    """
    running = [p for p in players if _is_running(p)]
    if not running:
        return f"No media player running ({' / '.join(players)})"
    playing = []
    for app in running:
        try:
            if _player_state(app) == "playing":
                playing.append(app)
        except AppleScriptError as exc:
            if "-1743" in str(exc) or "Not authorized" in str(exc):
                raise
    if playing:
        for app in playing:
            run_osascript(f'tell application "{app}" to pause', timeout=PLAYER_TIMEOUT_S)
        return f"{' & '.join(playing)}: paused"
    target = running[0]
    run_osascript(f'tell application "{target}" to play', timeout=PLAYER_TIMEOUT_S)
    return f"{target}: playing"


# --------------------------------------------------------------------- worker
@dataclass
class SystemState:
    volume: int | None = None
    muted: bool = False
    available: bool = True     # False if osascript keeps failing


class MacController:
    def __init__(self, players: tuple[str, ...] = ("Spotify", "Music"), dry_run: bool = False,
                 on_message: Callable[[str], None] | None = None,
                 use_coreaudio: bool = True) -> None:
        self.players = players
        self.audio = coreaudio.load() if use_coreaudio and not dry_run else None
        self.backend = "CoreAudio" if self.audio else "osascript"
        self.dry_run = dry_run
        self.on_message = on_message or (lambda msg: None)
        self.state = SystemState()
        self._lock = threading.Condition()
        self._pending_volume: int | None = None
        self._commands: deque[str] = deque()
        self._last_set = 0.0
        self._last_poll = 0.0
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="mac-controller", daemon=True)

    # --------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.dry_run:
            self.state = SystemState(volume=50, muted=False)
        else:
            self._refresh()
        self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop = True
            self._lock.notify()
        self._thread.join(timeout=OSASCRIPT_TIMEOUT_S + 1)

    # ------------------------------------------------------------------ public
    def set_volume(self, level: int) -> None:
        with self._lock:
            self._pending_volume = level
            # Update the HUD immediately; the worker confirms shortly after.
            self.state.volume = level
            self._lock.notify()

    def mute(self) -> None:
        self._enqueue("mute")

    def unmute(self) -> None:
        self._enqueue("unmute")

    def play_pause(self) -> None:
        self._enqueue("playpause")

    # ---------------------------------------------------------------- internal
    def _enqueue(self, cmd: str) -> None:
        with self._lock:
            self._commands.append(cmd)
            self._lock.notify()

    def _run(self) -> None:
        while True:
            with self._lock:
                self._lock.wait_for(
                    lambda: self._stop or self._commands or self._pending_volume is not None,
                    timeout=POLL_INTERVAL_S,
                )
                if self._stop:
                    return
                cmd = self._commands.popleft() if self._commands else None
                vol = None
                if cmd is None and self._pending_volume is not None:
                    vol, self._pending_volume = self._pending_volume, None
            try:
                if cmd in ("mute", "unmute"):
                    self._do_mute(cmd == "mute")
                elif cmd == "playpause":
                    self._do_play_pause()
                elif vol is not None:
                    self._do_set_volume(vol)
                else:
                    self._maybe_poll()
            except AppleScriptError as exc:
                log.warning("AppleScript failed: %s", exc)
                self.on_message(_explain_error(str(exc)))
            except Exception:  # keep the worker alive no matter what
                log.exception("controller error")

    def _do_set_volume(self, level: int) -> None:
        if not self.dry_run:
            if self.audio is not None:
                try:
                    self.audio.set_volume(level)
                except coreaudio.CoreAudioError:
                    set_volume(level)  # e.g. device without a volume control
            else:
                set_volume(level)
        self._last_set = time.monotonic()
        log.debug("volume -> %d", level)

    def _do_mute(self, muted: bool) -> None:
        if self._current_muted() == muted:
            msg = "Already muted" if muted else "Already unmuted"
        else:
            self._set_muted(muted)
            msg = "Muted" if muted else "Unmuted"
        log.info("ACTION %s -> %s", "mute" if muted else "unmute", msg)
        self.on_message(msg)

    def _do_play_pause(self) -> None:
        msg = "Play/Pause (dry run)" if self.dry_run else play_pause(self.players)
        log.info("ACTION play_pause -> %s", msg)
        self.on_message(msg)

    def _maybe_poll(self) -> None:
        now = time.monotonic()
        if self.dry_run or now - self._last_set < POLL_QUIET_AFTER_SET_S:
            return
        if now - self._last_poll >= POLL_INTERVAL_S:
            self._refresh()

    def _refresh(self) -> None:
        self._last_poll = time.monotonic()
        try:
            vol, muted = self._read_state()
        except AppleScriptError as exc:
            log.warning("could not read volume: %s", exc)
            self.state.available = False
            return
        with self._lock:
            if self._pending_volume is None:
                self.state.volume = vol
            self.state.muted = muted
            self.state.available = True


    def _current_muted(self) -> bool:
        if self.dry_run:
            return self.state.muted
        if self.audio is not None and self._try(self.audio.has_mute):
            try:
                return self.audio.get_mute()
            except coreaudio.CoreAudioError:
                pass
        return get_volume_state()[1]

    def _set_muted(self, muted: bool) -> None:
        if not self.dry_run:
            done = False
            if self.audio is not None and self._try(self.audio.has_mute):
                try:
                    self.audio.set_mute(muted)
                    done = True
                except coreaudio.CoreAudioError:
                    pass
            if not done:
                run_osascript(f"set volume output muted {'true' if muted else 'false'}")
        self.state.muted = muted

    def _read_state(self) -> tuple[int | None, bool]:
        if self.audio is not None:
            try:
                vol = self.audio.get_volume() if self.audio.has_volume() else None
                muted = self.audio.get_mute() if self.audio.has_mute() else False
                return vol, muted
            except coreaudio.CoreAudioError:
                pass
        return get_volume_state()

    @staticmethod
    def _try(fn) -> bool:
        try:
            return bool(fn())
        except coreaudio.CoreAudioError:
            return False


def _explain_error(err: str) -> str:
    if "-1743" in err or "Not authorized" in err:
        return "Automation permission denied - allow it in System Settings > Privacy > Automation"
    if "-600" in err:
        return "Player is not running"
    return f"AppleScript error: {err[:60]}"
