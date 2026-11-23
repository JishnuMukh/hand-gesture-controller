# Hand Gesture Controller for macOS

Control your Mac's volume, mute and music with hand gestures in front of the
built-in webcam. Built with Python, OpenCV and MediaPipe.

| Gesture | Action |
|---|---|
| 🤏 **Pinch**: thumb + index, other fingers folded. Hold still about ¼ s until the panel turns green, then move. | **Volume, relative.** Start with the fingertips together and spread them: the volume rises by the amount you spread. Start apart and bring them together: it falls by the amount you close. Drop your hand to keep the level. |
| ✌️ **Peace sign** (index + middle) | **Play / Pause** Spotify or Apple Music. |
| 🖐 **Palm facing the camera** (all five fingers) | **Mute.** |
| 🔙 **Back of the hand facing the camera** (all five fingers) | **Unmute.** |

## Launching

From Terminal:

```bash
cd hand-gesture-controller
./run.sh
```

Or double-click **`Launch Gesture Controller.command`** in Finder.

The first time it runs, `run.sh` creates a `.venv` virtual environment,
installs the requirements, and downloads the MediaPipe hand model to `models/`.
Later launches start immediately. To run it by hand instead:

```bash
source .venv/bin/activate
python main.py
```

**Keys in the window:** `q`/`Esc` quit · `space` pause or resume gesture control · `h` show or hide the gesture help.

### Camera selection

By default the app uses the **Mac's built-in camera**, matched by the name
"MacBook", "FaceTime" or "Built-in". It never silently uses an iPhone
Continuity Camera. This matters because OpenCV numbers cameras by sorting
their device IDs, so a nearby iPhone often becomes camera 0.

```bash
./run.sh --list-cameras             # show cameras with OpenCV's index
./run.sh --camera-name iphone       # pick a camera by (part of) its name
./run.sh --camera 2                 # or by index
```

### macOS permissions (first run only)

1. **Camera.** macOS asks whether *Terminal* (or iTerm, etc.) may use the
   camera. Click **Allow**. The app waits up to 30 s for your answer. If you
   clicked "Don't Allow" earlier, turn it on in **System Settings → Privacy &
   Security → Camera**.
2. **Automation.** The first Play/Pause asks whether Terminal may control
   *Spotify* or *Music*. Click **OK**. To change it later, go to **System
   Settings → Privacy & Security → Automation**.

Launch from Terminal.app rather than from inside an IDE or another app's
built-in terminal. Some host apps can't show the camera permission prompt.

## How it behaves

* **Palm vs back of hand.** The app compares the direction from the wrist to
  the index knuckle with the direction to the pinky knuckle, and combines that
  with MediaPipe's left/right-hand label. This tells palm from back for either
  hand, at any rotation. A hand seen edge-on (mid-turn) triggers nothing. If
  mute and unmute ever come out reversed, run with `--flip-palm`.
* **Mute and unmute are explicit.** Palm always mutes and back of hand always
  unmutes; repeating one is harmless ("Already muted"). Changing the volume by
  pinching does *not* unmute.
* **Play / Pause.** If anything is playing, it is paused. Otherwise the first
  running player (Spotify, then Music) is resumed. Neither app is ever
  launched for you.
* **No accidental triggers.** A gesture must be held for 0.3 s before it fires.
  It fires once per gesture: lower your hand to fire it again. Each action has
  a 1.5 s cooldown, plus a 0.8 s global cooldown between different actions.
  Tracking flicker shorter than 0.25 s is ignored. The HUD shows cooldowns
  counting down.
* **Volume never jumps.** Only the *change* in thumb-index distance since the
  pinch engaged counts. A full closed-to-wide spread equals 100%; adjust with
  `--volume-gain`. If you push past 0% or 100%, reversing direction responds
  immediately, with no dead travel. Pinch can't be confused with the other
  gestures, because it needs middle, ring and pinky folded, and other gestures
  are ignored while volume mode is active.
* **Smooth volume.** A One-Euro filter removes landmark jitter (heavy smoothing
  when your hand is still, light smoothing when it moves). A slew limiter caps
  changes at 150% per second, and a 2% hysteresis stops flicker between
  neighbouring values. When you let go, the level from 0.1 s before release is
  kept, so the motion of releasing doesn't change the volume.

### Options

```text
./run.sh --help
  --camera N / --camera-name TEXT / --list-cameras   camera selection (default: built-in)
  --volume-gain G     volume change for a full closed-to-wide pinch, x100 % (default 1.0)
  --absolute-volume   fixed gap = fixed level instead of relative (with soft takeover)
  --flip-palm         swap palm/back detection if mute and unmute are reversed
  --cooldown S        seconds between repeated actions (default 1.5)
  --players LIST      players in priority order (default Spotify,Music)
  --dry-run           recognise gestures but don't change anything
  --no-mirror         don't mirror the preview
  --snapshot-dir DIR  save an annotated frame whenever a gesture fires (debugging)
  --headless --max-seconds N   run without a window for N seconds (testing)
  -v                  debug logging (every pose change and volume write)
```

Every threshold and timing value is in `gesture_controller/config.py`.

## Project layout

```text
main.py                       entry point
run.sh                        sets up .venv + model on first run, then launches
Launch Gesture Controller.command   double-clickable launcher
requirements.txt              mediapipe, opencv-contrib-python, numpy
gesture_controller/
  config.py      all tunables (thresholds, timings, smoothing)
  cameras.py     find the built-in camera's OpenCV index by name
  tracker.py     MediaPipe HandLandmarker wrapper (VIDEO mode)
  gestures.py    stateless per-frame pose classification + palm orientation
  engine.py      hold/cooldown/latching state machine, volume mode, mappings
  smoothing.py   One-Euro filter + slew limiter with rollback history
  coreaudio.py   ctypes CoreAudio volume/mute (~5 ms per call)
  macos.py       background worker: CoreAudio, with osascript fallback; Spotify/Music via osascript
  overlay.py     HUD: skeleton, gesture panel, volume bar, cooldowns, toasts
  app.py         camera loop + CLI
tests/           unit tests with anatomically modelled synthetic hands
scripts/classify_images.py    run the classifier on photos (for tuning)
```

### Design notes

* **Rotation- and distance-invariant recognition.** Finger states come from
  MediaPipe's 3-D *world* landmarks, using ratios rather than pixel positions:
  * whether a finger is extended: distance from wrist to tip compared with
    wrist to the middle knuckle;
  * whether the thumb is out: thumb tip to pinky knuckle compared with palm
    width;
  * pinch size: thumb tip to index tip compared with palm length.

  Thresholds were calibrated on real hand photos. Measured values: straight
  fingers 1.20–1.86, curled fingers 0.60–0.99.
* **Volume via CoreAudio.** An `osascript` call takes about 200 ms, which would
  limit volume updates to about 5 per second. The app calls CoreAudio's
  virtual main volume (the same control as the menu-bar slider) through
  `ctypes` instead, which takes milliseconds and needs no extra dependency.
  `osascript` is the automatic fallback, and it's used for Spotify and Music.
  All system calls run on a worker thread, so the video loop never stalls.

## Tests

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

42 tests cover:
* pose classification, including invariance under random 3-D rotation and
  scale;
* palm vs back for both hands, at any in-plane rotation, with edge-on
  rejection;
* hold, latch, flicker and cooldown behaviour;
* relative volume: raising by the amount spread, lowering by the amount
  closed, gain, ratchet at the limits, slew limiting, jitter suppression,
  release rollback;
* absolute mode with soft takeover;
* picking the built-in camera when an iPhone sorts first.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not open camera" | Allow the camera for your terminal app in Privacy & Security, then relaunch. |
| Wrong camera | `./run.sh --list-cameras`, then `--camera-name` or `--camera`. |
| Mute and unmute reversed | `./run.sh --flip-palm` |
| "Automation permission denied" toast | Enable Spotify/Music for your terminal under Privacy & Security → Automation. |
| "No media player running" | Open Spotify or Music first; the app won't launch them. |
| Gestures flicker | Improve lighting and keep the whole hand in frame, about 0.4–1 m from the camera. |
