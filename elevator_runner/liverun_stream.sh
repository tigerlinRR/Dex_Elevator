#!/bin/bash
# Start a live run and stream its log, so the terminal follows the robot instead of
# showing snapshots minutes apart. Measured: SSH itself costs 0.16 s per call once the
# connection is reused, so the lag was never the link — it was fetching the log every
# few minutes instead of following it.
cd "$(dirname "$0")"
LOG=${1:-/tmp/live_stream.log}
rm -f "$LOG" /tmp/dex_press_go
nohup timeout 1500 python3 -u liverun.py > "$LOG" 2>&1 &
echo "run pid $!"
# Follow until the run exits, prefixing each line with seconds since start so the
# terminal shows WHEN, not just what.
T0=$(date +%s)
tail -n +1 -f "$LOG" &
TAIL=$!
while pgrep -f liverun.py > /dev/null; do sleep 1; done
sleep 2
kill $TAIL 2>/dev/null
echo "=== run finished after $(( $(date +%s) - T0 ))s ==="
