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
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from drive_straight import (  # noqa: E402
    HALF_WIDTH_M, HOST, HULL_M, MAX_SPEED, PORT, TURN_FLOOR_DEG, Chassis, _settled_pose,
    _wrap_deg, clear_band, drive, heading_sweep,
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


_T0 = None
_MARKS = []


def _mark(label: str) -> None:
    """Timestamp a phase boundary. Measure before optimising: this project has produced a
    step-by-step profile whose parts summed to 10 ms for a call that took 26."""
    global _T0
    now = time.time()
    if _T0 is None:
        _T0 = now
        _MARKS.append((label, now, 0.0))
        print(f"  [t+  0.0s] {label}")
        return
    prev = _MARKS[-1][1]
    _MARKS.append((label, now, now - prev))
    print(f"  [t+{now - _T0:6.1f}s] {label}   (+{now - prev:.1f}s)")


def _profile() -> None:
    if len(_MARKS) < 2:
        return
    print("\n=== where the time went ===")
    for label, _, dt in _MARKS[1:]:
        print(f"  {dt:6.1f}s  {label}")
    print(f"  {_MARKS[-1][1] - _T0:6.1f}s  TOTAL")


class Halted(Exception):
    """The emergency stop went on mid-run. Marching through the remaining phases with a
    dead base — which this harness did once, all the way to launching the press — turns
    one clear cause into a page of consequences."""


def _drive(ch: Chassis, metres: float, args) -> None:
    """Every straight leg goes through drive(), never a bare _leg.

    A single uncorrected leg is NOT reliable in the arm-ward (-ori) direction: measured
    2026-09-04, a 0.099 m nudge moved 0.045 and an exit leg issued its full 3.00 m of
    velocity over 15.3 s and travelled 1.553. It is a start-up lag rather than a speed
    cap — 1.553 m is exactly 7.8 s at the commanded 0.20 m/s, so the base spends the first
    ~7.5 s of a leg not moving. Legs the other way OVER-deliver by ~5 %. drive()'s
    correction legs absorb both, which is why every accurate figure recorded until now
    came from that path and why calling _leg directly here quietly produced a press at
    +10 cm, where the arm had no IK solution at all.
    """
    # tol 0.03, not 0.02: below ~3 cm this base does not respond — the same stiction
    # floor the turns have. With the per-direction leg model in place the first leg lands
    # within a couple of centimetres, and a 2 cm tolerance then spent two or three further
    # legs commanding 21-24 mm and achieving 0.000, which is the visible "stop and shuffle
    # again" with nothing to show for it.
    drive(ch, metres, min(args.speed, MAX_SPEED), 4.0, 0.03, 3)
    ch.pump(0.1)
    if ch.estop:
        raise Halted("emergency stop pressed during a leg")


def _prewarm_press(floors: str, extra: str, token: Path, log: Path):
    """Start the press NOW and let it block on a token, so its startup overlaps the drive.

    Measured 2026-09-04: from the turn finishing to the arm moving was ~14 s, of which
    ~7.3 s is startup that does not depend on where the robot is — 2.8 s loading the
    TensorRT engine, 2.5 s opening the camera, ~1.5 s ramping the GPU off its 306 MHz idle
    clock, 0.4 s connecting the arm. `elevator_runner` already pays that during its drive
    (8.3 s -> 1.6 s); this harness was paying it after arriving, every run.

    Nothing about WHEN to press moves into the child — it blocks on a file this process
    writes after the turn. Letting the press decide arrival for itself is what drove the
    arm into the panel once.
    """
    try:
        token.unlink()
    except FileNotFoundError:
        pass
    cmd = [sys.executable, str(REPO / "initialization" / "press_buttons.py")]
    cmd += floors.split() + extra.split() + ["--go", "--wait-go", str(token)]
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT,
                            text=True)
    return proc, fh


def _wait_prewarm(proc, log: Path, timeout: float = 40.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        try:
            if "PREWARM_READY" in log.read_text():
                return True
        except OSError:
            pass
        time.sleep(0.2)
    return False


def _report(tag: str, ch: Chassis, start) -> tuple:
    # `ch.pose` is already SETTLED here: every leg and every turn ends by waiting for two
    # consecutive agreeing samples. Re-settling cost a measured 2.0 s per report at the
    # 1 Hz pose feed, purely to print a line.
    p = ch.pose or ch.wait_pose()
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
    proc = fh = None
    try:
        ch.pump(1.0)
        if ch.estop:
            return print("ABORT: the emergency stop is pressed — nothing here can move, "
                         "and the arm controllers are unpowered too") or 4
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

        # PRE-WARM the press here, outside, where the seconds are free — it holds the
        # camera and the engine from now until it presses, so the harness does NOT open a
        # camera of its own. Two processes on one Orbbec is how the chest 335 gets into
        # the state where it enumerates but delivers no colour.
        # The measured approach moved to a FALLBACK for the same reason: it needs the
        # camera. Since the per-direction leg model went in, four consecutive runs landed
        # the panel 0.3-4.2 cm from target and the correction chose not to move every
        # time, so paying 4.9 s of measurement on every run to catch a case that has not
        # occurred is the wrong trade. If the press refuses, the camera is free by then
        # and the harness measures, nudges and retries.
        token = Path(f"/tmp/dex_mock_go_{os.getpid()}")
        log = Path(f"/tmp/dex_mock_press_{os.getpid()}.log")
        if args.press and not args.hold:
            proc, fh = _prewarm_press(args.press, args.press_args, token, log)
            ready = _wait_prewarm(proc, log)
            print(f"press pre-warm {'ready' if ready else 'NOT ready (continuing)'} "
                  f"— its ~7 s of startup now overlaps the drive")

        # remote is taken ONCE, here, and given back only in the finally block
        ch.enter_remote()
        ch.pump(0.3)

        # --- 2. in ---------------------------------------------------------------
        _mark("entry leg starts")
        print(f"IN: reversing {args.back:.2f} m")
        _clear_foreign("in")
        _drive(ch, args.back, args)
        _mark("entry leg done")
        after = _report("in", ch, start)
        d_ori = _wrap_deg(after[2] - start[2])
        if abs(d_ori) > CONTAMINATED_DEG:
            _clear_foreign("in")
            return print(f"ABORT: the entry leg turned {d_ori:+.1f} deg — a straight leg "
                         f"sends angular 0, so something else had the wheels") or 3

        # --- 3. face the panel ----------------------------------------------------
        if abs(args.turn) >= 0.5:
            _mark("turn starts")
            print(f"TURN: {args.turn:+.1f} deg to face the panel")
            _clear_foreign("turn")
            _turn(ch, args.turn, args.rate, args.corrections)
        _mark("turn done")
        at_panel = _report("at panel", ch, start)
        if args.hold:
            print("HOLD: stopping here as asked. Measure the panel, adjust --back/--turn, "
                  "and re-run; `elevator_runner/goto.py start` drives back to the point.")
            return 0

        # --- 4. press: release the pre-warmed child ------------------------------
        _mark("press released")
        press_rc = 0
        if args.press and proc is not None:
            print(f"PRESS: releasing the pre-warmed child ({args.press})")
            token.touch()
            proc.wait()
            press_rc = proc.returncode
            out = log.read_text()
            for l in [x for x in out.splitlines() if x.strip()][-3:]:
                print("  " + l)
            if fh:
                fh.close()
            proc = None
        elif args.press:
            cmd = [sys.executable, str(REPO / "initialization" / "press_buttons.py")]
            cmd += args.press.split() + args.press_args.split() + ["--go"]
            print(f"PRESS (cold, no pre-warm): {' '.join(cmd[1:])}")
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            press_rc = r.returncode
            for l in [x for x in r.stdout.splitlines() if x.strip()][-3:]:
                print("  " + l)

        if args.press:
            # The press blocks for ~60 s with nothing pumping the topic stream, and the
            # chassis closes an idle socket. Reconnect and re-assert remote before the
            # exit, or the next wait_pose dies after the press has already succeeded.
            ch.reconnect()
            ch.enter_remote()
            ch.pump(0.3)

        # --- 4b. FALLBACK: only if the press could not reach ----------------------
        # The camera is free now, so this costs nothing on a good run. It exists because
        # a panel outside the reachable window produces "no IK at standoff" for every
        # roll — measured at x 0.731 — and the cure is a few centimetres of base motion,
        # which the press cannot do for itself.
        if args.press and press_rc != 0 and not args.no_approach:
            print(f"  press exited {press_rc} — measuring and retrying once")
            _mark("fallback approach")
            cam = det = base_T_cam = None
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
                m = _measure_panel(cam, det, base_T_cam)
            except Exception as e:                                 # noqa: BLE001
                print(f"  fallback perception unavailable ({type(e).__name__}: {e})")
                m = None
            if m is not None:
                px, py, n = m
                err = px - args.target_x
                print(f"  panel at x {px:+.3f} y {py:+.3f} ({n} obs); off by "
                      f"{err * 100:+.1f} cm")
                if abs(err) > args.approach_tol:
                    print(f"  nudging {-err:+.3f} m along the arm axis")
                    _clear_foreign("approach")
                    _drive(ch, -err, args)
            if cam is not None:
                # stop(), not close() — and never swallowed: a camera left open makes the
                # retry fail with "no color frame", one step away from the real cause.
                try:
                    cam.stop()
                except Exception as e:                             # noqa: BLE001
                    print(f"  !! could NOT release the camera ({type(e).__name__}: {e})")
            cmd = [sys.executable, str(REPO / "initialization" / "press_buttons.py")]
            cmd += args.press.split() + args.press_args.split() + ["--go"]
            r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
            for l in [x for x in r.stdout.splitlines() if x.strip()][-3:]:
                print("  " + l)
            ch.reconnect()
            ch.enter_remote()
            ch.pump(0.3)

        # --- 5. aim the way out, by MEASURING it ----------------------------------
        _mark("press done")
        out_leg = -args.back
        sweep_out, _ = heading_sweep(ch, out_leg, args.window, args.margin, args.sweep)
        # The target heading is the one we CAME IN ON, not the middle of whatever the
        # lidar finds open. Aiming at the band's middle over-turned by 5.7 deg in the lab
        # (2026-09-04) and cost 24 cm of closure: in a car the middle of the clear band IS
        # the doorway, but in an open room it is merely the roomiest direction, which is a
        # different thing. So the entry heading is the objective and the lidar band is a
        # CONSTRAINT on it — which also stays correct in a car, where the entry heading is
        # perpendicular to the door and therefore inside the band anyway.
        at = ch.pose or start
        want = _wrap_deg(start[2] - at[2])
        aim = want
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
                print(f"  nothing clears — going back to the entry heading ({want:+.1f} "
                      f"deg) and letting the gate refuse if it is wrong")
            else:
                lo, hi, _best = band
                # A clear heading t is reached by turning +t whichever way the leg points:
                # the travel frame negates BOTH axes for a backward leg, which is a 180 deg
                # rotation and preserves handedness. Getting that sign wrong once turned
                # the exit 27.5 deg the wrong way, and only the clearance gate saved it.
                if lo <= want <= hi:
                    print(f"  clear band {lo:+d}..{hi:+d} deg contains the entry heading "
                          f"({want:+.1f} deg) — going back to it")
                else:
                    aim = float(min(max(want, lo + 1), hi - 1))
                    print(f"  entry heading {want:+.1f} deg is OUTSIDE the clear band "
                          f"{lo:+d}..{hi:+d}; clamping to {aim:+.1f} deg")
                    print(f"  clear band {lo:+d}..{hi:+d} deg; aiming at its middle "
                          f"({mid:+.1f} deg in the travel frame)")
        _mark("exit aim computed")
        if abs(aim) >= 0.5:
            print(f"AIM: {aim:+.1f} deg")
            _clear_foreign("aim")
            _turn(ch, aim, args.rate, args.corrections)
        aimed = _report("aimed", ch, start)

        # --- 6. out ---------------------------------------------------------------
        _mark("exit aim done")
        print(f"OUT: driving {out_leg:+.2f} m")
        _clear_foreign("out")
        before = ch.pose
        _drive(ch, out_leg, args)
        end = _report("out", ch, start)
        if before is not None and abs(_wrap_deg(end[2] - before[2])) > CONTAMINATED_DEG:
            print(f"  WARNING: the exit leg turned "
                  f"{_wrap_deg(end[2] - before[2]):+.1f} deg — something else had the "
                  f"wheels, so the closure below is not a measurement of our driving")

        print(f"\nCLOSURE  {math.hypot(end[0] - start[0], end[1] - start[1]) * 100:.1f} cm"
              f"   heading {_wrap_deg(end[2] - start[2]):+.1f} deg")
        print("(closure is a measurement of dead-reckoning drift, not a pass/fail — the "
              "exit's requirement is clearing the doorway, see the module docstring)")
        _mark("exit leg done")
        _profile()
        return 0
    except Halted as e:
        print(f"ABORT: {e} — stopping here rather than running the rest on a dead base")
        return 4
    finally:
        # A pre-warmed child that is never released keeps the camera open, and the NEXT
        # attempt then fails with "no color frame after 40 tries" — a symptom one step
        # away from the cause. Every path that abandons the press has to kill it.
        if proc is not None and proc.poll() is None:
            print("  killing the un-released pre-warm child (it holds the camera)")
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except Exception:                                      # noqa: BLE001
                proc.kill()
        if fh is not None:
            try:
                fh.close()
            except Exception:                                      # noqa: BLE001
                pass
        ch.close()


def _fmt(v: float) -> str:
    return "inf" if v == float("inf") else f"{v:.2f} m"


if __name__ == "__main__":
    raise SystemExit(main())
