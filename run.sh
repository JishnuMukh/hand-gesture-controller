#!/usr/bin/env bash
# Launch the gesture controller, setting up (or repairing) its environment first.
#
# Every step is safe to re-run after an interruption or failure:
#   * the virtual environment is recreated if its interpreter is missing or broken;
#   * dependencies are (re)installed until an install completes, and again
#     whenever requirements.txt or constraints.txt change;
#   * the model is downloaded to a temporary file, checked against a pinned
#     SHA-256 and only then moved into place; a corrupt copy is replaced.
#
# Usage: ./run.sh [--setup-only] [app options...]
#   PYTHON=/path/to/python3.12 ./run.sh   use a specific interpreter for the venv
set -euo pipefail
cd "$(dirname "$0")"

# Keep in sync with gesture_controller/config.py (a test checks this).
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_SHA256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
MODEL_PATH="models/hand_landmarker.task"
MIN_PYTHON="3.12"

VENV=".venv"
VPY="$VENV/bin/python"
STAMP="$VENV/.requirements.sha256"

die() { echo "run.sh: $*" >&2; exit 1; }
sha256() { shasum -a 256 "$1" | awk '{print $1}'; }

setup_only=0
if [ "${1:-}" = "--setup-only" ]; then
  setup_only=1
  shift
fi

# --------------------------------------------------------------- platform checks
[ "$(uname -s)" = "Darwin" ] || die "this app only runs on macOS."
# MediaPipe publishes macOS wheels for Apple silicon only.
[ "$(uname -m)" = "arm64" ] || die "Apple silicon (arm64) is required; MediaPipe has no Intel macOS wheel."

# ------------------------------------------------------------ virtual environment
venv_ok() {
  [ -x "$VPY" ] && "$VPY" -c 'import sys' >/dev/null 2>&1
}

if ! venv_ok; then
  base="${PYTHON:-python3}"
  command -v "$base" >/dev/null 2>&1 || die "'$base' not found. Install Python $MIN_PYTHON+ (e.g. from python.org)."
  "$base" -c "import sys; sys.exit(0 if sys.version_info >= tuple(map(int, '$MIN_PYTHON'.split('.'))) else 1)" \
    || die "Python $MIN_PYTHON or newer is required; '$base' is $("$base" -c 'import platform; print(platform.python_version())'). Set PYTHON=/path/to/python3.x."
  if [ -e "$VENV" ]; then
    echo "Virtual environment is incomplete or broken; recreating it..."
  else
    echo "Creating virtual environment..."
  fi
  rm -f "$STAMP"
  "$base" -m venv --clear "$VENV"
fi

# ------------------------------------------------------------------ dependencies
# The stamp is written only after pip succeeds, so an interrupted or failed
# install is retried on the next launch instead of being skipped.
want="$(cat requirements.txt constraints.txt | shasum -a 256 | awk '{print $1}')"
have="$(cat "$STAMP" 2>/dev/null || true)"
if [ "$want" != "$have" ]; then
  echo "Installing dependencies..."
  rm -f "$STAMP"
  # A venv whose creation was interrupted can lack pip.
  "$VPY" -m pip --version >/dev/null 2>&1 || "$VPY" -m ensurepip --upgrade \
    || die "could not install pip into $VENV. Delete $VENV and run ./run.sh again."
  "$VPY" -m pip install --disable-pip-version-check -r requirements.txt -c constraints.txt \
    || die "dependency installation failed. Fix the error above and run ./run.sh again; it will retry."
  echo "$want" > "$STAMP"
fi

# ------------------------------------------------------------------------- model
if [ -f "$MODEL_PATH" ] && [ "$(sha256 "$MODEL_PATH")" != "$MODEL_SHA256" ]; then
  echo "Existing hand model does not match the pinned checksum; downloading it again..."
  rm -f "$MODEL_PATH"
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading MediaPipe hand landmark model..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  tmp="$MODEL_PATH.part"
  rm -f "$tmp"
  curl -fL --retry 2 -o "$tmp" "$MODEL_URL" || { rm -f "$tmp"; die "model download failed; run ./run.sh again to retry."; }
  got="$(sha256 "$tmp")"
  if [ "$got" != "$MODEL_SHA256" ]; then
    rm -f "$tmp"
    die "downloaded model has SHA-256 $got, expected $MODEL_SHA256. Not using it."
  fi
  mv -f "$tmp" "$MODEL_PATH"
fi

if [ "$setup_only" = "1" ]; then
  echo "Setup complete."
  exit 0
fi
exec "$VPY" main.py "$@"
