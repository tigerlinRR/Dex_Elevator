#!/usr/bin/env python3
"""Run the whole elevator manoeuvre as ONE process: in, turn, press, aim, out.

    python3 initialization/mock_elevator_run.py --back 2.56 --turn -25
    python3 initialization/mock_elevator_run.py --back 2.56 --turn -25 --press "1 4"
    python3 initialization/mock_elevator_run.py --back 1.0 --turn 0 --dry

WHY ONE PROCESS. Each `drive_straight` command restores `auto` when it exits, and in that
gap the chassis's OWN control_unit issues charge moves and drives the robot away —
measured 2026-09-04: a turn-and-turn-back test came back +40.7 deg for a commanded +25
and the base had translated 0.53 m between the two turns, because move 646
(`creator: control_unit, type: charge`) was running underneath it. Any multi-step
measurement taken across separate invocations near the dock is contaminated. This holds
`remote` from the first leg to the last, which takes the wheels away from that planner,
and restores `auto` on every exit path.

WHAT "SUCCESS" MEANS, which is not what it first looks like. Returning to the exact
starting pose is neither achievable nor required. Not achievable: the base cannot turn
finer than ~3 deg (stiction — commands under 3 deg often move it not at all), so a
remembered angle cannot be replayed. Not required: after leaving the car the robot
relocalises through navigation anyway. The exit's real requirement is to clear the
doorway, so the return heading is MEASURED against the opening with the lidar
(`heading_sweep`) rather than replayed from the turn that went in. Where there is no
constriction — an open room, as in the lab mock — there is nothing to aim at and the
outbound turn is simply undone; the closure error is then reported for what it is, a
measurement of the dead-reckoning drift, not a pass/fail.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from drive_straight import (  # noqa: E402
    HALF_WIDTH_M, HOST, HULL_M, MAX_SPEED, PORT, TURN_FLOOR_DEG, Chassis, _leg,
    _settled_pose, _wrap_deg, clear_band, heading_sweep,
)

REPO = Path(__file__).resolve().parent.parent
OURS = "dex_elevator_mock"
# Where the panel has to sit, in the ARM BASE frame, for the press to have room. Measured
# 2026-09-04 on this panel: x 0.649 and 0.667 both planned all four buttons at 24/24 rolls
# with 63-69 deg of margin and 34-48 mm of path clearance; x 0.695 pressed 4/4 but two
# buttons were down to 1/24 and 2/24; x 0.731 left two buttons with NO IK solution at all
# and the other two with 5-6 mm of clearance against a 5 mm limit. So the usable window is
# a few centimetres wide, and a single 2.7 m leg scatters by ~5 % — 14 cm — which is why
# the distance cannot simply be hard-coded.
TARGET_X_M = 0.67
APPROACH_TOL_M = 0.03
# A straight leg sends angular 0, so its heading cannot change by much. When one does,
# something else has the wheels — measured 2026-09-04, an exit leg came back 0.646 m of a
# commanded 2.50 with the heading 36.67 deg round, because move 647
# (`creator: control_unit, type: charge`) was running underneath it.
CONTAMINATED_DEG = 5.0


def _active_move() -> dict | None:
    try:
        r = requests.get(f"http://{HOST}:{PORT}/chassis/moves/current", timeout=6)
    except Exception:                                              # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    m = r.json()
    return m if isinstance(m, dict) and m.get("state") == "moving" else None


def _cancel_move() -> bool:
    try:
        r = requests.patch(f"http://{HOST}:{PORT}/chassis/moves/current",
                           json={"state": "cancelled"}, timeout=6)
        return r.status_code < 300
    except Exception:                                              # noqa: BLE001
        return False


def _clear_foreign(tag: str) -> None:
    """The chassis's own control_unit issues charge moves whenever it feels like it, and
    holding `remote` does not stop it — it takes the mode back to execute one. So every
    phase starts by looking for a move we did not create and cancelling it."""
    m = _active_move()
    if m is None:
        return
    if m.get("creator") == OURS:
        return
    print(f"  [{tag}] FOREIGN MOVE {m.get('id')} by {m.get('creator')!r} "
          f"({m.get('type')}) is running — cancelling it")
    ok = _cancel_move()
    print(f"  [{tag}] cancel {'accepted' if ok else 'FAILED'}")


def _measure_panel(cam, det, base_T_cam, frames: int = 3):
    """Panel centre in the ARM BASE frame, from the same perception the press uses.

    Returns (x, y, n) or None. Costs ~200 ms once the engine and camera are warm, which
    is why the warm-up is done OUTSIDE the car, before the entry leg: inside, the car may
    already be moving for somebody else and the seconds are not ours to spend.
    """
    import numpy as np
    from core.transforms import deproject_pixel, transform_point
    acc = []
    for _ in range(frames):
        f = cam.capture()
        rgb, depth, intr = f.rgb, f.depth, f.intrinsics
        for d in det.detect(rgb):
            x1, y1, x2, y2 = d.bbox_xyxy
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            patch = depth[int(y1):int(y2), int(x1):int(x2)]
            good = patch[patch > 0.05]
            if not good.size:
                continue
            z = float(np.median(good))
            if z > 2.0:
                continue
            bx, by, bz = transform_point(base_T_cam, deproject_pixel(intr, cx, cy, z))
            if bz < 0.3:            # floor clutter is not a button
                continue
            acc.append((bx, by))
    if len(acc) < 3:
        return None
    a = np.array(acc)
    return float(np.median(a[:, 0])), float(np.median(a[:, 1])), len(acc)


def _report(tag: str, ch: Chassis, start) -> tuple:
    p = _settled_pose(ch) or ch.wait_pose()
    print(f"  [{tag}] pose x={p[0]:.3f} y={p[1]:.3f} ori={p[2]:+.4f}"
          f"   from start: {math.hypot(p[0] - start[0], p[1] - start[1]) * 100:.1f} cm, "
          f"{_wrap_deg(p[2] - start[2]):+.1f} deg")
    return p


def _turn(ch: Chassis, degrees_: float, rate: float, corrections: int) -> float:
    """Damped closed-loop turn, inline so `remote` is never handed back mid-run."""
    from drive_straight import CORR_GAIN, _spin_leg
    total = 0.0
    for attempt in range(corrections + 1):
        remaining = degrees_ - total
        if abs(remaining) <= (TURN_FLOOR_DEG if attempt else 0.5):
            break
        want = remaining if attempt == 0 else remaining * CORR_GAIN
        if attempt and abs(want) < TURN_FLOOR_DEG:
            want = math.copysign(TURN_FLOOR_DEG, want)
        ch.enter_remote()
        got, drift, reason = _spin_leg(ch, want, rate, verbose=False)
        total += got
        print(f"    turn {attempt + 1}: {want:+.1f} -> {got:+.1f} deg "
              f"(remaining {degrees_ - total:+.1f}, drift {drift * 1000:.0f} mm)")
        if reason != "turn complete":
            print(f"    stopped: {reason}")
            break
    return total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--back", type=float, required=True, metavar="M",
                    help="reverse this far to get in (positive = the base's own front, "
                         "which is the arm's rear — the direction it backs in)")
    ap.add_argument("--turn", type=float, required=True, metavar="DEG",
                    help="turn this much once inside to face the panel")
    ap.add_argument("--press", metavar="FLOORS", default=None,
                    help="button labels to press once turned, e.g. \"1 4\"")
    ap.add_argument("--press-args", default="--lift --no-verify",
                    help="extra arguments for press_buttons.py")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--rate", type=float, default=0.20, metavar="RADPS")
    ap.add_argument("--corrections", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.05, metavar="M")
    ap.add_argument("--window", type=float, default=1.20, metavar="M")
    ap.add_argument("--sweep", type=float, default=40.0, metavar="DEG")
    ap.add_argument("--target-x", type=float, default=TARGET_X_M, metavar="M",
                    help="where the panel should sit along the arm's forward axis")
    ap.add_argument("--approach-tol", type=float, default=APPROACH_TOL_M, metavar="M",
                    help="do not move at all if the panel is already this close to target")
    ap.add_argument("--no-approach", action="store_true",
                    help="skip the measured final approach and press from wherever the "
                         "dead-reckoned legs land. Faster by ~2 s and occasionally "
                         "unpressable — see TARGET_X_M for what that costs.")
    ap.add_argument("--hold", action="store_true",
                    help="stop after the turn and leave the robot there. For calibrating "
                         "--back/--turn: measure the panel, adjust, repeat. Without it "
                         "the run continues to press, aim and drive back out.")
    ap.add_argument("--dry", action="store_true",
                    help="plan and print, move nothing")
    args = ap.parse_args()

    ch = Chassis()
    ch.check_identity()
    ch.open_stream()
    try:
        start = _settled_pose(ch) or ch.wait_pose()
        print(f"START pose x={start[0]:.3f} y={start[1]:.3f} ori={start[2]:+.4f}")
        half = HALF_WIDTH_M + args.margin

        # --- 1. is the way in clear, and is it clear straight? -------------------
        sweep_in, _ = heading_sweep(ch, args.back, args.window, args.margin, args.sweep)
        if sweep_in is None:
            return print("ABORT: no lidar returns near the entry path") or 2
        straight_in = dict(sweep_in)[0]
        print(f"entry: {_fmt(straight_in)} of side clearance straight ahead "
              f"(need {half:.2f} m)")
        if straight_in < half:
            band = clear_band(sweep_in, half)
            hint = "" if band is None else f"; {band[0]:+d}..{band[1]:+d} deg would clear"
            return print(f"ABORT: the entry path is blocked{hint}") or 2
        if args.dry:
            print("dry run: would reverse %.2f m, turn %+.1f deg, press %s, "
                  "aim by lidar, drive back %.2f m"
                  % (args.back, args.turn, args.press or "nothing", args.back))
            return 0

        # Warm the perception up HERE, outside, where the seconds are free: loading the
        # TensorRT engine and opening the camera is ~5 s, against ~200 ms for the
        # measurement itself once warm. Inside the car that 5 s would come out of the
        # ride, so it is paid before the robot has gone anywhere.
        cam = det = base_T_cam = None
        if not (args.no_approach or args.hold):
            try:
                import numpy as np
                sys.path.insert(0, str(REPO))
                from core.camera.orbbec import OrbbecCamera
                from core.config import REPO_ROOT
                from yolo.trt_detector import TrtButtonDetector
                base_T_cam = np.load(REPO_ROOT / "data/calibration/cam_chest.npy")
                det = TrtButtonDetector(conf=0.25)
                cam = OrbbecCamera(camera_id="cam_chest", match_name="335",
                                   exposure=156, gain=16)
                cam.start()
                print("perception warm (engine + camera up before entering)")
            except Exception as e:                                 # noqa: BLE001
                print(f"perception unavailable ({type(e).__name__}: {e}) — "
                      f"continuing without the measured approach")
                cam = det = None

        # remote is taken ONCE, here, and given back only in the finally block
        ch.enter_remote()
        ch.pump(0.3)

        # --- 2. in ---------------------------------------------------------------
        print(f"IN: reversing {args.back:.2f} m")
        _clear_foreign("in")
        ch.gate_clearance(args.back)
        got, d_ori, reason = _leg(ch, args.back, min(args.speed, MAX_SPEED), 4.0,
                                  verbose=True)
        print(f"  achieved {got:+.3f} m ({reason}), heading {d_ori:+.2f} deg")
        if abs(d_ori) > CONTAMINATED_DEG:
            _clear_foreign("in")
            return print(f"ABORT: the entry leg turned {d_ori:+.1f} deg — a straight leg "
                         f"sends angular 0, so something else had the wheels") or 3
        _report("in", ch, start)

        # --- 3. face the panel ----------------------------------------------------
        if abs(args.turn) >= 0.5:
            print(f"TURN: {args.turn:+.1f} deg to face the panel")
            _clear_foreign("turn")
            _turn(ch, args.turn, args.rate, args.corrections)
        at_panel = _report("at panel", ch, start)
        if args.hold:
            print("HOLD: stopping here as asked. Measure the panel, adjust --back/--turn, "
                  "and re-run; `elevator_runner/goto.py start` drives back to the point.")
            return 0

        # --- 3b. close the last few centimetres by MEASURING, not by remembering -----
        if cam is not None and det is not None:
            m = _measure_panel(cam, det, base_T_cam)
            if m is None:
                print("APPROACH: panel not located — leaving the position as it is")
            else:
                px, py, n = m
                err = px - args.target_x
                print(f"APPROACH: panel at x {px:+.3f} y {py:+.3f} ({n} obs); "
                      f"target x {args.target_x:+.3f}, off by {err * 100:+.1f} cm")
                if abs(err) <= args.approach_tol:
                    print(f"  already inside {args.approach_tol * 100:.0f} cm — not moving")
                else:
                    # Moving along the arm's own axis is the most accurate primitive we
                    # have: short legs land within 2 mm. Negative `move` closes on the
                    # panel. y is left alone — a straight leg cannot fix it, and the
                    # measured window in y is wide (-0.15 to -0.40 all pressed).
                    print(f"  nudging {-err:+.3f} m along the arm axis")
                    _clear_foreign("approach")
                    ch.enter_remote()
                    got, d_ori, reason = _leg(ch, -err, min(args.speed, MAX_SPEED), 4.0,
                                              verbose=False)
                    print(f"  achieved {got:+.3f} m ({reason})")
                    m2 = _measure_panel(cam, det, base_T_cam)
                    if m2:
                        print(f"  now at x {m2[0]:+.3f} y {m2[1]:+.3f}"
                              f"   ({(m2[0] - args.target_x) * 100:+.1f} cm from target)")
        if cam is not None:
            # Release the device BEFORE the press starts: two processes on one Orbbec is
            # how the chest camera gets into the state where it enumerates but never
            # delivers colour. The method is stop(), not close() — and this must NOT be
            # swallowed, because a silent failure here leaves the press contending for
            # the device and the symptom appears one step later, as "no color frame".
            try:
                cam.stop()
                print("camera released before the press")
            except Exception as e:                                 # noqa: BLE001
                print(f"  !! could NOT release the camera ({type(e).__name__}: {e}) — "
                      f"the press may fail with 'no color frame'")
            cam = None

        # --- 4. press -------------------------------------------------------------
        if args.press:
            cmd = [sys.executable, str(REPO / "initialization" / "press_buttons.py")]
            cmd += args.press.split() + args.press_args.split() + ["--go"]
            print(f"PRESS: {' '.join(cmd[1:])}")
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            tail = [l for l in r.stdout.splitlines() if l.strip()][-3:]
            for l in tail:
                print("  " + l)
            if r.returncode != 0:
                print(f"  press exited {r.returncode} — continuing to the exit anyway")
            # The press blocks for ~60 s with nothing pumping the topic stream, and the
            # chassis closes an idle socket. Reconnect and re-assert remote before the
            # exit, or the next wait_pose dies after the press has already succeeded.
            ch.reconnect()
            ch.enter_remote()
            ch.pump(0.3)
            print("  chassis stream reconnected after the press")

        # --- 5. aim the way out, by MEASURING it ----------------------------------
        out_leg = -args.back
        sweep_out, _ = heading_sweep(ch, out_leg, args.window, args.margin, args.sweep)
        aim = 0.0
        if sweep_out is None:
            print("EXIT: no lidar returns near the exit path — undoing the turn instead")
            aim = -args.turn
        else:
            straight_out = dict(sweep_out)[0]
            band = clear_band(sweep_out, half)
            resid = math.radians(TURN_FLOOR_DEG) * abs(out_leg)
            print(f"EXIT: {_fmt(straight_out)} straight out (need {half:.2f} m), "
                  f"unresolvable drift {resid * 100:.1f} cm over this leg")
            if band is None:
                print("  nothing clears — undoing the turn and letting the gate refuse")
                aim = -args.turn
            else:
                lo, hi, _best = band
                mid = (lo + hi) / 2.0
                if straight_out >= half and (straight_out - half) > resid:
                    print(f"  clear band {lo:+d}..{hi:+d} deg; straight out already has "
                          f"{(straight_out - half) * 100:.0f} cm of slack — no constriction "
                          f"to aim at, so undoing the turn")
                    aim = -args.turn
                else:
                    # +mid, not -mid: the travel frame negates both axes for a
                    # backward leg, which preserves handedness, so a clear heading t is
                    # reached by turning +t either way. The sign was wrong here first and
                    # the exit turned 27.5 deg the wrong way; the clearance gate refused
                    # the resulting path, which is the only reason nothing was hit.
                    aim = mid
                    print(f"  clear band {lo:+d}..{hi:+d} deg; aiming at its middle "
                          f"({mid:+.1f} deg in the travel frame)")
        if abs(aim) >= 0.5:
            print(f"AIM: {aim:+.1f} deg")
            _clear_foreign("aim")
            _turn(ch, aim, args.rate, args.corrections)
        aimed = _report("aimed", ch, start)

        # --- 6. out ---------------------------------------------------------------
        print(f"OUT: driving {out_leg:+.2f} m")
        _clear_foreign("out")
        ch.gate_clearance(out_leg)
        # verbose, deliberately: _leg only reports "wheel_slipping went true during this
        # leg" when verbose, and suppressing it cost the one explanation available when
        # an exit leg issued its full 2.70 m of velocity and moved 1.564 m (2026-09-04).
        got, d_ori, reason = _leg(ch, out_leg, min(args.speed, MAX_SPEED), 4.0,
                                  verbose=True)
        print(f"  achieved {got:+.3f} m ({reason}), heading {d_ori:+.2f} deg")
        if abs(d_ori) > CONTAMINATED_DEG:
            _clear_foreign("out")
            print(f"  WARNING: the exit leg turned {d_ori:+.1f} deg — something else had "
                  f"the wheels, so the closure below is not a measurement of our driving")
        end = _report("out", ch, start)

        print(f"\nCLOSURE  {math.hypot(end[0] - start[0], end[1] - start[1]) * 100:.1f} cm"
              f"   heading {_wrap_deg(end[2] - start[2]):+.1f} deg")
        print("(closure is a measurement of dead-reckoning drift, not a pass/fail — the "
              "exit's requirement is clearing the doorway, see the module docstring)")
        return 0
    finally:
        ch.close()


def _fmt(v: float) -> str:
    return "inf" if v == float("inf") else f"{v:.2f} m"


if __name__ == "__main__":
    raise SystemExit(main())
