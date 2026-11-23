#!/usr/bin/env bash
# Double-click in Finder to start the gesture controller in Terminal.
cd "$(dirname "$0")"
./run.sh "$@"
status=$?
echo
read -n 1 -s -r -p "Gesture controller exited (status $status). Press any key to close."
