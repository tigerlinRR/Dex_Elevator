#!/usr/bin/env bash
# The ONLY pre-approved way to drive the chassis from the dev Mac.
#
# Why it exists: a Bash permission rule matches by PREFIX, so it cannot target a
# command buried inside `ssh host '...'`. A rule broad enough to match that would
# have to allow every command on the robot. This script is the narrow alternative:
# it passes its arguments to ONE program and can do nothing else.
#
# It deliberately drives through `drive_straight.py`, never `goto_pose.py`:
#   * drive_straight pins angular velocity to 0 and runs a LIDAR CLEARANCE GATE
#     before it emits any twist, plus a current-based stall abort.
#   * goto_pose hands the path to the chassis's own planner, which we cannot gate
#     once it starts — on 2026-09-28 three moves posted that way ended with the
#     chassis's overcurrent protection firing and the operator physically blocking
#     the robot.
#
#   ./initialization/drive_remote.sh move --speed 0.18 --tol 0.03 -- 1.509
#   ./initialization/drive_remote.sh clearance
#   ./initialization/drive_remote.sh state
#
# Put flags BEFORE the `--` separator: argparse treats everything after `--` as
# positional, so `move -- -1.49 --speed 0.18` exits with "unrecognized arguments".
set -euo pipefail

HOST="${DEX_HOST:-dex5-ll}"

if [ "$#" -eq 0 ]; then
    echo "usage: $(basename "$0") <state|clearance|move|turn|align|calibrate|stop> [args...]" >&2
    exit 2
fi

# Quote each argument so the remote shell receives them exactly as given.
remote_args=""
for a in "$@"; do
    remote_args+=" $(printf '%q' "$a")"
done

exec ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" \
    "cd Dex_Elevator && exec python3 -u initialization/drive_straight.py${remote_args}"
