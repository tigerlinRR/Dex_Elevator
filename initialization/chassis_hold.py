#!/usr/bin/env python3
"""Stop the chassis from driving off on its own — with a battery floor.

The chassis's `control_unit` posts its own `charge` moves whenever it feels like it,
and holding `remote` does not stop it: it takes the mode back to run one. Measured
2026-09-14 — driven 30 cm off its dock, it posted a charge move 30 s later and drove
itself back to within 1 cm of where it started, silently undoing a positioning step.
That is what this cancels.

**Why it is not a blanket block.** On 2026-09-11 the chassis parked 0.66 m short of
its dock, never charged, ran the battery to 4 % and shut itself down — costing three
days. A watchdog that refuses every charge move would cause exactly that. So below
``--min-battery`` the robot is left alone to go and charge; the block only applies
while it has enough charge that a trip to the dock is discretionary.

Only `control_unit`-created moves are touched. Anything this project posts (creator
`dex_elevator*`) is never cancelled — the watchdog must not fight our own navigation.

    python3 initialization/chassis_hold.py                  # run until Ctrl-C
    python3 initialization/chassis_hold.py --min-battery 15 # allow charging later
    python3 initialization/chassis_hold.py --minutes 90     # stop by itself
    python3 initialization/chassis_hold.py --install-cron   # survive a reboot
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import requests

HOST = "192.168.25.25"
API = 8000
WS_API = 8090
OURS = "dex_elevator"


def battery_pct() -> float | None:
    """Battery percentage, or None if unreadable.

    Read from the websocket topic, NOT from ``/battery-state`` on port 8000 — that
    path serves an HTML page, so a JSON parse there fails in a way that looks like a
    dead battery reading. Only consulted when a charge move actually shows up, so the
    once-per-call connection cost does not matter.
    """
    try:
        from websocket import create_connection
        ws = create_connection(f"ws://{HOST}:{WS_API}/ws/v2/topics", timeout=5)
        try:
            ws.send(json.dumps({"enable_topic": "/battery_state"}))
            deadline = time.time() + 6.0
            while time.time() < deadline:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    continue
                d = json.loads(msg)
                if d.get("topic") != "/battery_state":
                    continue
                v = d.get("percentage")
                if v is None:
                    continue
                return float(v) * 100.0 if float(v) <= 1.0 else float(v)
        finally:
            ws.close()
    except Exception:
        pass
    return None


def install_cron() -> int:
    """Re-launch at boot. User crontab — no sudo, and easy for the operator to remove."""
    # ABSOLUTE path: cron does not run from the repo directory, so a relative path
    # here installs an entry that silently does nothing at boot — which is the worst
    # kind of watchdog, one that looks installed and is not.
    me = os.path.abspath(sys.argv[0])
    line = (f"@reboot sleep 60 && /usr/bin/python3 {me} "
            f">> /tmp/chassis_hold.log 2>&1\n")
    try:
        cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    except FileNotFoundError:
        print("no crontab on this machine")
        return 1
    if me in cur:
        print("already installed in the user crontab:")
        print("  " + [l for l in cur.splitlines() if me in l][0])
        return 0
    subprocess.run(["crontab", "-"], input=cur + line, text=True, check=True)
    print("installed at-boot entry in the USER crontab (no sudo, remove with `crontab -e`):")
    print("  " + line.strip())
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-battery", type=float, default=25.0,
                    help="below this %%, let the robot go and charge (default 25). The "
                         "chassis once ran itself flat and shut down; this is the guard "
                         "against causing that")
    ap.add_argument("--minutes", type=float, default=0.0, help="0 = run until stopped")
    ap.add_argument("--poll", type=float, default=1.0, help="seconds between checks")
    ap.add_argument("--install-cron", action="store_true", help="re-launch at boot, then exit")
    args = ap.parse_args()

    if args.install_cron:
        return install_cron()

    deadline = time.time() + args.minutes * 60 if args.minutes else float("inf")
    print(f"holding the chassis still (cancelling control_unit moves) — "
          f"battery floor {args.min_battery:.0f}%", flush=True)
    cancels, allowed, last_state = 0, 0, None
    warn_key, warn_at, warn_n = None, 0.0, 0
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{HOST}:{API}/chassis/moves/current", timeout=4)
            if r.status_code == 200:
                mv = r.json()
                creator = str(mv.get("creator", ""))
                if creator.startswith(OURS):
                    pass                                  # never fight our own navigation
                elif creator == "control_unit":
                    pct = battery_pct()
                    # AN UNREADABLE BATTERY MUST ALLOW, NOT CANCEL. This read
                    # `pct is not None and pct < floor`, so every failure of
                    # `battery_pct()` — a websocket timeout, no /battery_state message
                    # inside its 6 s window, a missing field — fell through to the
                    # cancel branch and BLOCKED the robot from going to charge. That is
                    # exactly the blanket block this module's docstring exists to
                    # prevent, and it is the shape of failure this project keeps
                    # meeting: the dangerous answer arriving as an absence of data
                    # rather than as an error.
                    #
                    # The two mistakes are not symmetric. A wrong ALLOW costs a trip to
                    # the dock, recoverable by repositioning. A wrong CANCEL costs the
                    # battery: the robot shut down at 4 % on 2026-09-11 (three days
                    # lost), and was found at 4 %, offline and `moveState: cancelled`
                    # again on 2026-09-18.
                    if pct is None or pct < args.min_battery:
                        why = ("battery UNREADABLE — allowing the move rather than risk "
                               "blocking a charge" if pct is None else
                               f"battery {pct:.0f}% < {args.min_battery:.0f}% — "
                               f"ALLOWING it to charge")
                        if last_state != "allow":
                            print(f"{time.strftime('%H:%M:%S')}  {why}", flush=True)
                            last_state = "allow"
                        allowed += 1
                    else:
                        requests.patch(f"http://{HOST}:{API}/chassis/moves/current",
                                       json={"state": "cancelled"}, timeout=4)
                        cancels += 1
                        last_state = "cancel"
                        print(f"{time.strftime('%H:%M:%S')}  cancelled {mv.get('type')} "
                              f"move id={mv.get('id')} (battery "
                              f"{'?' if pct is None else f'{pct:.0f}%'}, total {cancels})",
                              flush=True)
        except Exception as e:
            # Throttle a REPEATING fault. With the chassis powered off this loop logs
            # the same ConnectionError every poll: measured 36,704 identical lines in
            # one run, which fills /tmp and buries anything worth reading. The first
            # occurrence still prints immediately — a watchdog that goes quiet about a
            # new fault is worse than a noisy one — and a continuing one repeats every
            # 10 minutes with a count, so "how long has this been broken" stays legible.
            key = type(e).__name__
            now = time.time()
            if key != warn_key or now - warn_at >= 600:
                extra = (f" (same fault, {warn_n} more since)" if key == warn_key else "")
                print(f"{time.strftime('%H:%M:%S')}  warn: {key}: {e}{extra}", flush=True)
                warn_key, warn_at, warn_n = key, now, 0
            else:
                warn_n += 1
        time.sleep(args.poll)
    print(f"done — cancelled {cancels}, allowed {allowed} while low", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
