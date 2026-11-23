# Hand Gesture Controller for macOS

Control your Mac's volume, mute and music with hand gestures in front of the
built-in webcam. Built with Python, OpenCV and MediaPipe.

| Gesture | Action |
|---|---|
| 🤏 **Pinch**: thumb + index, other fingers folded. Hold still about ¼ s until the panel turns green, then move. | **Volume, relative.** Start with the fingertips together and spread them: the volume rises by the amount you spread. Start apart and bring them together: it falls by the amount you close. Drop your hand to keep the level. |
| ✌️ **Peace sign** (index + middle) | **Play / Pause** Spotify or Apple Music. |
| 🖐 **Palm facing the camera** (all five fingers) | **Mute.** |
| 🔙 **Back of the hand facing the camera** (all five fingers) | **Unmute.** |

## Requirements

| | Supported | Tested |
|---|---|---|
| Mac | Apple silicon (arm64). MediaPipe 1.2.3 publishes no Intel macOS wheel. | MacBook Pro, Apple silicon |
| macOS | 15 or newer (the pinned OpenCV wheel targets macOS 15+) | 27.0.1 |
| Python | 3.12 or newer (the pinned NumPy needs 3.12+) | 3.14.6 |

Only the "Tested" column has been run. Other versions in the supported range
should install, but nobody has checked them.

Dependencies are pinned to the tested versions: direct ones in
`requirements.txt`, the full set in `constraints.txt`. The MediaPipe model is
pinned to a versioned URL (`float16/1`), and its SHA-256 is checked on every
launch.

## Launching

From Terminal:

```bash
cd hand-gesture-controller
./run.sh
```

Or double-click **`Launch Gesture Controller.command`** in Finder.

On first run, `run.sh` does three things:
1. creates a `.venv` virtual environment;
2. installs the pinned dependencies;
3. downloads the hand model to `models/` and checks its SHA-256.

Later launches skip whatever is already done. Each step is safe to retry:

* If the dependency install fails or is interrupted, the next launch retries it.
  It's only marked done after pip succeeds.
* Dependencies are reinstalled whenever `requirements.txt` or `constraints.txt`
  change.
* If `.venv` is missing its interpreter, or the interpreter no longer runs
  (for example, the Python it was built from was uninstalled), `.venv` is
  rebuilt.
* The model is downloaded to a temporary `.part` file and only moved into place
  if its checksum matches. A wrong checksum aborts setup and the file is
  deleted. An existing model that doesn't match is downloaded again.

```bash
./run.sh --setup-only                  # set up / repair without starting the app
PYTHON=/path/to/python3.13 ./run.sh    # use a specific Python for a new .venv
```

To run the app by hand after setup:

```bash
source .venv/bin/activate
python main.py
```

**Keys in the window:** `q`/`Esc` quit · `space` pause or resume gesture control · `h` show or hide the gesture help.

### Camera selection

By default the app looks for the **Mac's built-in camera** by name: the first
camera whose name contains "MacBook", "FaceTime" or "Built-in". This matters
because OpenCV numbers cameras by sorting their device IDs, so a nearby iPhone
Continuity Camera often becomes camera 0. If no camera matches those names,
the app logs a warning listing the cameras it found and falls back to index 0.

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
  with MediaPipe's left/right-hand label. In principle this works for either
  hand at any in-plane rotation. A hand seen nearly edge-on falls inside a
  dead zone and triggers neither action.
  * The sign convention was calibrated live, with one hand held upright in
    front of the MacBook camera.
  * The other hand and strong tilts are covered only by the synthetic tests,
    not by live testing.
  * If mute and unmute come out reversed, run with `--flip-palm`.
* **Mute and unmute are explicit.** Palm always mutes and back of hand always
  unmutes; repeating one is harmless ("Already muted"). Changing the volume by
  pinching does *not* unmute.
* **Play / Pause.** If anything is playing, it is paused. Otherwise the first
  running player (Spotify, then Music) is resumed. Neither app is ever
  launched for you.
* **Guarding against accidental triggers:**
  * a gesture must be held for 0.3 s before it fires;
  * it fires once per gesture: lower your hand to fire it again;
  * each action has a 1.5 s cooldown, plus a 0.8 s global cooldown between
    different actions;
  * tracking dropouts shorter than 0.25 s are ignored;
  * the HUD shows cooldowns counting down.

  These reduce false triggers but can't rule them out. For example, a hand
  passing through a peace-sign shape for longer than 0.3 s will toggle
  playback.
* **Volume doesn't jump on engage.** Only the *change* in thumb-index distance
  since the pinch engaged counts.
  * A full closed-to-wide spread equals 100%; adjust with `--volume-gain`.
  * If you push past 0% or 100%, reversing direction responds immediately,
    with no dead travel.
  * Pinch requires middle, ring and pinky folded, which separates it from the
    other gestures, and discrete gestures are ignored while volume mode is
    active.
* **Volume smoothing:**
  * a One-Euro filter reduces landmark jitter (heavy smoothing when your hand
    is still, light smoothing when it moves);
  * a slew limiter caps changes at 150% per second;
  * a 2% hysteresis stops flicker between neighbouring values;
  * when you let go, the level from 0.1 s before release is kept, so the motion
    of releasing doesn't change the volume.

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

`./run.sh --setup-only` is handled by `run.sh` itself; every other option is
passed through to the app. Every threshold and timing value is in
`gesture_controller/config.py`.

## Project layout

```text
main.py                       entry point
run.sh                        sets up / repairs .venv + model, then launches
Launch Gesture Controller.command   double-clickable launcher
requirements.txt              direct dependencies (pinned)
constraints.txt               full tested dependency set (pinned)
requirements-dev.txt          test dependencies (pinned)
gesture_controller/
  config.py      all tunables (thresholds, timings, smoothing) + pinned model URL/SHA-256
  cameras.py     find the built-in camera's OpenCV index by name
  tracker.py     MediaPipe HandLandmarker wrapper (VIDEO mode)
  gestures.py    stateless per-frame pose classification + palm orientation
  engine.py      hold/cooldown/latching state machine, volume mode, mappings
  smoothing.py   One-Euro filter + slew limiter with rollback history
  coreaudio.py   ctypes CoreAudio volume/mute
  macos.py       background worker: CoreAudio, with osascript fallback; Spotify/Music via osascript
  overlay.py     HUD: skeleton, gesture panel, volume bar, cooldowns, toasts
  app.py         camera loop + CLI
tests/           automated tests (see Testing)
scripts/classify_images.py    run the classifier on photos (for tuning)
```

### Design notes

* **Size- and rotation-tolerant features.** Finger states come from
  MediaPipe's 3-D *world* landmarks, using ratios rather than pixel positions:
  * whether a finger is extended: distance from wrist to tip compared with
    wrist to the middle knuckle;
  * whether the thumb is out: thumb tip to pinky knuckle compared with palm
    width;
  * pinch size: thumb tip to index tip compared with palm length.

  The ratios don't change with hand size or with a rigid rotation of the
  landmarks. The synthetic tests check exactly that. On real video they're
  only as stable as MediaPipe's landmark estimates, which get worse when the
  fingers are hidden from the camera.
* **Where the thresholds come from.** They were set from a small set of
  MediaPipe's public sample gesture photos, where straight fingers measured
  1.20–1.86 and curled fingers 0.60–0.99. They were then checked in live
  sessions with one person. They haven't been tuned across different hands,
  lighting or camera positions.
* **Volume via CoreAudio.** On the development Mac, an `osascript` call took
  about 200 ms, which would limit volume updates to about 5 per second. So the
  app calls CoreAudio's virtual main volume (the same control as the menu-bar
  slider) through `ctypes`, which took a few milliseconds there. `osascript` is
  the automatic fallback and is always used for Spotify and Music. All system
  calls run on a worker thread, so the video loop never waits on them.

## Testing

### Automated tests (no camera, no audio, no media players)

```bash
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
```

These tests don't touch the webcam, the system volume, or Spotify/Music.

* **Gesture tests (`test_gestures.py`, `test_engine.py`)** feed in synthetic
  hands, landmarks generated from a simple anatomical model, instead of
  MediaPipe output. They show the classification and state-machine logic is
  correct for those inputs. They don't show it works on real camera footage.
  Covered:
  * pose classification, including random 3-D rotation and scaling of the
    synthetic hand;
  * palm vs back for both handedness labels at any in-plane rotation, with the
    edge-on dead zone;
  * hold, latch, flicker and cooldown timing;
  * relative volume: raising and lowering, gain, ratchet at the limits, slew
    limiting, jitter suppression, release rollback;
  * absolute mode with soft takeover.
* **Camera tests (`test_cameras.py`)** check name-based camera selection
  against a canned `system_profiler` listing, including an iPhone sorting
  first.
* **Setup tests (`test_setup.py`)** check the installer and repository
  configuration:
  * the model URL and SHA-256 in `run.sh` match `config.py`, and the
    downloaded model matches the pinned checksum;
  * the requirements are exactly pinned and agree with `constraints.txt`;
  * `.gitignore` excludes env files, keys and credentials but keeps
    `*.example` files;
  * the real `run.sh` runs in a temporary copy of the project, with an empty
    requirements file and a fake `curl`. Scenarios: fresh setup, retry after a
    failed install, reinstall after the requirements change, rebuilding a
    broken venv, rejecting a checksum mismatch, recovering from a failed or
    corrupt download, refusing a too-old Python.

  These need macOS on Apple silicon and are skipped elsewhere. They don't
  install the real packages or download the real model.

### Manual hardware testing

Hardware behaviour can only be checked by hand, with a person in front of a
real camera. On 2026-11-22/23, on one MacBook Pro running macOS 27.0.1 with
one person, the following was observed:

| Checked live | Result |
|---|---|
| Built-in MacBook camera chosen while an iPhone Continuity Camera was connected (and sorted first) | ✅ |
| Camera access from Terminal.app (the app logged `camera 1 (MacBook Pro Camera) opened`) | ✅ |
| Palm → mute, back of hand → unmute (logged facing values about ±0.43) | ✅ |
| Peace sign → Spotify paused, then resumed | ✅ |
| Relative pinch volume, both directions, with the CoreAudio backend | ✅ |
| Clean install: `./run.sh --setup-only` in a fresh copy (real pip install of the pinned versions, real model download, checksum verified, model loads) | ✅ |

**Not yet tested on real hardware:**
* Apple **Music** (only Spotify was used);
* the **osascript fallback** for volume and mute (CoreAudio was always
  available);
* the **other hand**, and other people's hands;
* poor or back lighting;
* external webcams, and Macs other than the one above;
* double-clicking **`Launch Gesture Controller.command`** in Finder (live
  sessions started the app from Terminal);
* the camera-name fallback when there's no built-in camera;
* the 30 s wait for the first-run camera permission prompt (it never ran
  live: Terminal already had camera access each time);
* other Python and macOS versions;
* the gestures again after the setup and pinning changes. Application code
  didn't change, apart from the pinned model URL and checksum constants and
  one error message. The model file itself is byte-identical.

**Manual checklist** (about 5 minutes; start Spotify or Music first):
1. `./run.sh --list-cameras`, then `./run.sh`. The log line
   `camera N (…) opened` names the built-in camera.
2. Show your palm: the toast says "Muted" and the menu-bar volume shows muted.
   Show the back of your hand: "Unmuted". If they're swapped, use
   `--flip-palm`.
3. Peace sign: playback pauses. Peace sign again (after about 1.5 s): it
   resumes.
4. Pinch with fingertips together, wait for green, then spread: the volume
   rises. Start apart and close: it falls. Drop your hand: the level stays.
5. Repeat steps 2–4 with your other hand and in dimmer light.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `run.sh: Apple silicon (arm64) is required` | MediaPipe has no Intel macOS build at the pinned version. |
| `run.sh: Python 3.12 or newer is required` | Install a newer Python and run `PYTHON=/path/to/python3.x ./run.sh`. |
| `dependency installation failed` | Fix the pip error shown (often no network), then run `./run.sh` again; it retries. |
| `downloaded model has SHA-256 …` | The download was corrupted or the file changed upstream; nothing was installed. Retry; if it persists, open an issue. |
| "Could not open camera" | Allow the camera for your terminal app in Privacy & Security, then relaunch. |
| Wrong camera | `./run.sh --list-cameras`, then `--camera-name` or `--camera`. |
| Mute and unmute reversed | `./run.sh --flip-palm` |
| "Automation permission denied" toast | Enable Spotify/Music for your terminal under Privacy & Security → Automation. |
| "No media player running" | Open Spotify or Music first; the app won't launch them. |
| Gestures flicker | Improve lighting and keep the whole hand in frame, about 0.4–1 m from the camera. |
