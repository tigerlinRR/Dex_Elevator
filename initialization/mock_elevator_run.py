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

        # remote is taken ONCE, here, and given back only in the finally block
        ch.enter_remote()
        ch.pump(0.3)

        # --- 2. in ---------------------------------------------------------------
        print(f"IN: reversing {args.back:.2f} m")
        _clear_foreign("in")
        ch.gate_clearance(args.back)
        got, d_ori, reason = _leg(ch, args.back, min(args.speed, MAX_SPEED), 4.0,
                                  verbose=False)
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
                    aim = -mid          # heading_sweep is in the travel frame
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
        got, d_ori, reason = _leg(ch, out_leg, min(args.speed, MAX_SPEED), 4.0,
                                  verbose=False)
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
