#!/bin/bash
# Start the elevator_runner server detached, and wait until it is actually listening.
pkill -f "python3 server.py" 2>/dev/null
sleep 2
cd "$HOME/Dex_Elevator/elevator_runner" || exit 1
rm -f /tmp/runner.log
setsid /usr/bin/python3 server.py > /tmp/runner.log 2>&1 < /dev/null &
for i in $(seq 1 20); do
  sleep 1
  if ss -tln 2>/dev/null | grep -q 8765; then
    echo "listening on 8765 (pid $(pgrep -f 'python3 server.py' | head -1))"
    exit 0
  fi
done
echo "FAILED to start"; tail -6 /tmp/runner.log
exit 1
