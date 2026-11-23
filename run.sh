#!/usr/bin/env bash
# Launch the gesture controller, creating the virtual environment on first run.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -r requirements.txt
fi

if [ ! -f models/hand_landmarker.task ]; then
  echo "Downloading MediaPipe hand landmark model..."
  mkdir -p models
  curl -fL -o models/hand_landmarker.task \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
fi

exec .venv/bin/python main.py "$@"
