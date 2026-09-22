#!/usr/bin/env python3
"""Send the chassis to a pose using its OWN local planner, then measure where it landed.

For repositioning in open space this beats the straight-line driver, and the difference
is not speed: a straight leg can only travel along the robot's heading, so it cannot fix
a LATERAL offset at all, and the aim angle that would fix one is often below the ~3 deg
this chassis can resolve. A `standard` move handles position and heading together, in
the chassis's own planner, with its own obstacle avoidance.

Measured before: it lands ~5 cm from the pose it is given, three consecutive returns
agreeing to the millimetre — biased and repeatable rather than noisy. **Do not read the
target as where the robot will be**; this script measures afterwards and reports both.

Two things that bite, both recorded:
  * a local move runs in the chassis's planner, so killing THIS process does not stop
    the robot. Every abandoning path here cancels the move.
  * `chassis_hold.py` cancels moves whose creator is `control_unit` and never ours, so
    the creator must stay in the `dex_elevator*` namespace or the watchdog will fight
    this and win.

    python3 initialization/goto_pose.py --station approach   # the registered outside point
    python3 initialization/goto_pose.py --x 21.8 --y -15.0 --ori 2.85
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import requests
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

HOST, PORT = "192.168.25.25", 8000
CREATOR = "dex_elevator_goto"          # must stay in the namespace chassis_hold spares


def current():
    r = requests.get(f"http://{HOST}:{PORT}/chassis/moves/current", timeout=6)
    return r.json() if r.status_code == 200 else None


def cancel() -> None:
    try:
        requests.patch(f"http://{HOST}:{PORT}/chassis/moves/current",
                       json={"state": "cancelled"}, timeout=6)
    except Exception:
        pass


def _drive_to(tgt: dict, args) -> tuple[float, float, float, float, float] | None:
    """Post ONE `standard` move and wait for it.

    Returns (mm off, deg off, achieved x, achieved y, achieved ori) — the achieved pose
    is what lets the caller mirror the target instead of repeating it.

    Returns None if the move was cancelled or never finished, so the caller can tell
    "it landed badly" (retryable) apart from "it did not run" (not).
    """
    from initialization.round_trip import settled_pose
    body = {"creator": CREATOR, "type": "standard",
            "target_x": float(tgt["x"]), "target_y": float(tgt["y"]),
            "target_ori": float(tgt["ori"])}
    r = requests.post(f"http://{HOST}:{PORT}/chassis/moves", json=body, timeout=10)
    if r.status_code not in (200, 201):
        print(f"!! POST /chassis/moves -> {r.status_code}: {r.text[:200]}")
        return None
    mid = (r.json() or {}).get("id")
    print(f"posted move id={mid}, waiting…")
    # Read the start pose HERE, not from the caller: on a retry the caller's `before`
    # is the pose from the previous attempt, and the stall watchdog would then compare
    # motion against a place the robot has already left.
    before, _ = settled_pose()

    # Wait on MOTION, not on a budget. The pose feed is 1 Hz and repeats itself while
    # moving, so "has it moved" is asked over a window rather than between two samples.
    from initialization.drive_straight import Chassis

    t0 = time.time()
    ch = Chassis()
    ch.open_stream()
    try:
        last_state, last_pose, last_move, last_seq = None, before, time.time(), -1
        while True:
            ch.pump(0.3)
            if ch.pose and ch.pose_seq != last_seq:
                last_seq = ch.pose_seq
                if (abs(ch.pose[0] - last_pose[0]) > 0.01
                        or abs(ch.pose[1] - last_pose[1]) > 0.01
                        or abs(ch.pose[2] - last_pose[2]) > 0.021):
                    last_pose, last_move = ch.pose, time.time()
            m = current()
            state = (m or {}).get("state")
            if state != last_state:
                print(f"    t+{time.time() - t0:5.1f}s  state={state}")
                last_state = state
            if m is None or state in ("succeeded", "failed", "cancelled"):
                break
            still = time.time() - last_move
            if still > args.stall:
                print(f"!! no motion for {still:.0f} s — treating that as stuck and "
                      f"cancelling (the move runs in the chassis, not here)")
                cancel()
                return None
            if time.time() - t0 > args.hard_cap:
                print(f"!! hard cap {args.hard_cap:.0f} s reached — cancelling")
                cancel()
                return None
    except KeyboardInterrupt:
        print("\n!! interrupted — cancelling the move")
        cancel()
        return None
    finally:
        # `Chassis` has no close method — the websocket is the thing to release, and
        # the attribute name is what to close. Calling a method that does not exist
        # inside a bare `except` is invisible: it looked like cleanup for a whole
        # session and never ran once.
        try:
            if ch._ws is not None:
                ch._ws.close()
        except Exception:
            pass

    after, ok = settled_pose()
    dx, dy = after[0] - tgt["x"], after[1] - tgt["y"]
    dori = (after[2] - tgt["ori"] + math.pi) % (2 * math.pi) - math.pi
    print(f"after  : x={after[0]:.4f} y={after[1]:.4f} ori={after[2]:.4f} settled={ok}")
    print(f"landed {math.hypot(dx, dy) * 1000:.0f} mm and {math.degrees(dori):+.2f} deg "
          f"from the target")
    return math.hypot(dx, dy) * 1000, math.degrees(dori), after[0], after[1], after[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", choices=["approach", "press"],
                    help="a pose from configs/stations.yaml")
    ap.add_argument("--x", type=float)
    ap.add_argument("--y", type=float)
    ap.add_argument("--ori", type=float)
    ap.add_argument("--tol-mm", type=float, default=25.0, metavar="MM",
                    help="retry until the robot lands within this of the target")
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--stall", type=float, default=45.0, metavar="S",
                    help="cancel only after this long with NO MOTION (default 45). A "
                         "budget measured from dispatch conflates 'slower than usual' "
                         "with 'stuck', and those want opposite responses — this "
                         "project has the measurement: one drive took 316 s where the "
                         "same route normally took 56, and a 300 s budget gave up 6 s "
                         "before it finished. Progress is MOTION, never elapsed time")
    ap.add_argument("--hard-cap", type=float, default=300.0, metavar="S")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.station:
        with open(os.path.join(REPO, "configs", "stations.yaml")) as fh:
            st = yaml.safe_load(fh)["stations"][0]
        tgt = st["approach_pose"] if args.station == "approach" else st["chassis_pose"]
    elif None not in (args.x, args.y, args.ori):
        tgt = {"x": args.x, "y": args.y, "ori": args.ori}
    else:
        print("!! give --station, or all of --x --y --ori")
        return 2

    from initialization.round_trip import settled_pose

    # The arm rides with the robot and nothing models the room, so refuse to drive with
    # it extended — the same gate the round-trip harness applies.
    from core.config import load_pipeline
    from core.robot.realman import RealmanArm

    c = load_pipeline()["arm"]
    arm = RealmanArm(side=c.get("side", "right"), ip=c.get("ip"), port=c.get("port", 8080))
    arm.connect()
    try:
        q = [float(v) for v in arm.get_joint_angles()]
    finally:
        arm.disconnect()
    travel = [float(v) for v in c["home_joints_deg"]]
    off = max(abs(a - b) for a, b in zip(q, travel))
    print(f"arm is {off:.3f} deg from the travel pose")
    if off > 2.0:
        print("!! the arm is not stowed — refusing to drive")
        return 1

    before, ok = settled_pose()
    print(f"before : x={before[0]:.4f} y={before[1]:.4f} ori={before[2]:.4f} settled={ok}")
    print(f"target : x={tgt['x']:.4f} y={tgt['y']:.4f} ori={tgt['ori']:.4f}")
    print(f"         {math.hypot(tgt['x'] - before[0], tgt['y'] - before[1]) * 1000:.0f} mm away")
    if args.dry:
        print("dry run — nothing sent")
        return 0

    # RETRY TO A TOLERANCE, because one shot is not accurate enough for what comes
    # next. Measured 2026-09-18: single moves landed 18 / 22 / 43 mm from the same
    # target in clear space. The round-trip's drive-in then ADDS to that — a 43 mm
    # residual with -0.57 deg of heading came out as 69 mm at the station against a
    # 60 mm gate, and the cycle aborted. The planner is biased rather than noisy, so a
    # second attempt from much closer lands closer; this asks for a number instead of
    # hoping for one.
    best, moved = None, None
    for attempt in range(1, args.tries + 1):
        got = _drive_to(tgt, args)
        if got is None:
            return 1
        best = got
        if best[0] <= args.tol_mm:
            break
        # STOP if the robot did not actually move: this planner has a DEAD ZONE of
        # roughly 7 cm and simply declares arrival inside it. Measured 2026-09-18:
        # three moves to the same target landed on the identical pose to the
        # millimetre 35 mm out, and three more to DIFFERENT (mirrored) targets did not
        # move it either — so neither repeating nor aiming past the target converges,
        # and both were tried before this was written. Retrying is not a tool for the
        # last few centimetres here; nothing in this file can close them.
        if moved is not None and math.hypot(best[2] - moved[0], best[3] - moved[1]) < 0.005:
            print(f"!! the robot did not move ({best[0]:.0f} mm off): this planner "
                  f"has a ~7 cm arrival dead zone and will not close a gap smaller "
                  f"than that. Reposition by hand, or drive from further away.")
            break
        moved = (best[2], best[3])
    else:
        print(f"!! still {best[0]:.0f} mm off after {args.tries} attempts — "
              f"reporting it rather than pretending it converged")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
