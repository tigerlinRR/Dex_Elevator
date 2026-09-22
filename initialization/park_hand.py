#!/usr/bin/env python3
"""Close the right hand into a fist. Meant to run at boot, and safe to.

WHY THIS EXISTS. The LinkerHand loses power across an emergency stop and across any
controller restart, and its fingers relax OPEN. So "it was a fist last time" is not a
state that survives a restart — the robot comes back up with the hand splayed, which is
both what the operator sees as "the arm is open" and a real hazard: the fingertips sit
172.87 mm from the flange against the plunger's ~154 mm, so with the fingers extended
the HAND is the front-most part of the end effector and reaches a panel before the
plunger does.

WHY ONLY THE HAND, AND NOT THE ARM. Closing the fingers folds them inward and sweeps
nothing, so it is safe with nobody watching. Moving the ARM at boot is not: the
controller's self-collision model covers the arm's own links and end effector only —
not the torso, the lift column, the camera mount, the other arm, or the room — and an
unattended re-positioning sweep is exactly the motion that once took this arm toward a
wall. The arm does not move on its own across a restart anyway; it comes back where it
was left. Park it deliberately with:

    python3 initialization/move_arm_staged.py --to="<joints>"        # check only
    python3 initialization/move_arm_staged.py --to="<joints>" --go   # then execute

Install at boot (USER crontab, no sudo):

    python3 initialization/park_hand.py --install-cron
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import load_pipeline  # noqa: E402
from core.robot.realman import RealmanArm  # noqa: E402


def install_cron() -> int:
    """Re-run at boot from the USER crontab — no sudo, easy for the operator to remove.

    The path must be ABSOLUTE: cron does not run from the repo directory, so a relative
    entry installs cleanly and then silently does nothing, which is the worst kind of
    startup job — one that looks installed and is not.
    """
    me = os.path.abspath(sys.argv[0])
    # 90 s, not 60: the arm controller has to be up and answering before the hand can be
    # reached at all, because the hand is a Modbus slave on the arm's own tool-side bus.
    line = f"@reboot sleep 90 && /usr/bin/python3 {me} >> /tmp/park_hand.log 2>&1"
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout
    if me in cur:
        print("already installed:")
        print("  " + next(l for l in cur.splitlines() if me in l))
        return 0
    new = (cur.rstrip("\n") + "\n" + line + "\n") if cur.strip() else line + "\n"
    p = subprocess.run(["crontab", "-"], input=new, text=True)
    if p.returncode != 0:
        print("!! could not write the crontab")
        return 1
    print("installed:\n  " + line)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install-cron", action="store_true",
                    help="add a @reboot entry to the USER crontab and exit")
    ap.add_argument("--retries", type=int, default=6,
                    help="the arm controller may not be up yet at boot (default 6, "
                         "10 s apart)")
    args = ap.parse_args()
    if args.install_cron:
        return install_cron()

    cfg = load_pipeline()["arm"]
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(1, args.retries + 1):
        arm = RealmanArm(side=cfg.get("side", "right"), ip=cfg.get("ip"),
                         port=cfg.get("port", 8080))
        try:
            arm.connect()
            from core.hand.linkerhand import LinkerHand  # noqa: E402
            hand = LinkerHand(arm)
            hand.power_on()
            hand.set_speed()
            hand.pose("fist")
            # VERIFY by reading back, and test "nothing is sticking out" rather than
            # per-joint target matching: in a fist the index bottoms out against the
            # thumb at ~70 instead of 0, measured on both hands, so an exact test would
            # fail every time.
            deadline, joints = time.time() + 5.0, None
            while time.time() < deadline:
                joints = hand.read_joints()
                if joints is not None and max(joints) <= 100:
                    print(f"{stamp}  fist confirmed {joints}")
                    return 0
                time.sleep(0.4)
            print(f"{stamp}  attempt {attempt}: hand did not close (read back {joints})")
        except Exception as e:
            print(f"{stamp}  attempt {attempt}: {type(e).__name__}: {e}")
        finally:
            try:
                arm.disconnect()
            except Exception:
                pass
        if attempt < args.retries:
            time.sleep(10.0)
    # Report the failure rather than exiting 0 quietly: a startup job that fails
    # silently is indistinguishable from one that is not installed.
    print(f"{stamp}  GAVE UP after {args.retries} attempts — the hand is NOT a fist")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
