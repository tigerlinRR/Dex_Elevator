#!/bin/bash
# Run a press and STREAM its output.
#
# Waiting for the process to exit before looking means the whole press — 40 to 70 s —
# happens with nothing on the terminal, and the waiting command then times out and
# goes to the background, adding another layer of delay on top. tail -f costs nothing
# and makes the log current the moment anyone looks at it.
cd "$(dirname "$0")/.."
LOG=/tmp/press_stream.log
: > "$LOG"
PYTHONPATH=. nohup timeout 300 /usr/bin/python3 -u initialization/press_buttons.py "$@" \
    >> "$LOG" 2>&1 &
PID=$!
tail -n +1 -f "$LOG" &
TAIL=$!
wait $PID
EXIT=$?
sleep 1
kill $TAIL 2>/dev/null
echo "=== press exited $EXIT ==="
