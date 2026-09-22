#!/usr/bin/env python3
"""Drive out, drive in, extend, verify, press, retract, drive back — one cycle, measured.

Runs the arm-camera flow the operator asked for, end to end, as many times as asked, and
MEASURES at every stage instead of assuming. It calls the existing tools rather than
reimplementing them (`drive_straight.py`, `move_arm_staged.py`, `press_buttons.py`), so
there is one implementation of each behaviour and this file only owns the ORDER and the
gates between steps.

It exists as a file, not as a shell one-liner, for a reason: two runs were lost on
2026-09-18 to command construction alone — quotes eaten by a nested ssh, and `--`
swallowing the flags that followed it. Neither moved the robot, both cost a cycle.

    python3 initialization/round_trip.py --cycles 2
    python3 initialization/round_trip.py --dry        # plan and measure, never move

THE GATES, and why each one is there:
  * the arm must be at the TRAVEL pose before any drive. The camera and the plunger ride
    the arm, and nothing models the room.
  * after landing, the base must be within `--land-tol` of the station IN BOTH AXES
    before the arm extends. `move_arm_staged` checks the tip against a panel plane at a
    FIXED x in the base frame; land a long way closer and that constant is optimistic in
    the dangerous direction.
  * the press is left to refuse on its own. Identity verification, the aim-offset pose
    check and the obstacle flag are its business, not this file's.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import yaml  # noqa: E402

PY = sys.executable or "/usr/bin/python3"


def sh(args: list[str], timeout: float = 600.0) -> tuple[int, str]:
    """Run a repo tool and stream its output, returning (code, captured text)."""
    p = subprocess.Popen(args, cwd=REPO, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    out = []
    try:
        for line in p.stdout:
            out.append(line)
            sys.stdout.write("    " + line)
            sys.stdout.flush()
        p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        return 124, "".join(out)
    return p.returncode, "".join(out)


class Prewarm:
    """Start the press's position-independent startup while the base is still driving.

    Measured on this robot 2026-09-18: of a one-button press's 18.3 s, **9.6 s is
    reaction and only 8.5 s is motion**, and 7.5 s of the reaction (python import,
    arm connect, closing the hand, the TensorRT engine and opening the camera) does
    not depend on where the robot is standing. `press_buttons.py --wait-go` already
    does exactly this for `elevator_runner`; this harness launched it cold.

    Two things this must NOT do, both learned the hard way:
      * it must never infer arrival. The child blocks on a token file that only this
        file writes, after the landing gate and the arm extension have both passed.
        Letting the press decide for itself when the robot had arrived is what drove
        the arm into the panel.
      * it must be killed on EVERY abandoning path. The child holds the camera from
        launch, and one that is never released makes the next attempt fail with
        "no color frame after 40 tries" — a symptom one step away from its cause.

    Holding an arm connection while `move_arm_staged` moves the arm was verified safe
    before this was wired in: two concurrent RealmanArm connections both read joints
    fine, and the child issues nothing while blocked.
    """

    def __init__(self, floors: list[str], speed: float, contact: float,
                 push: float | None):
        self.token = Path(tempfile.gettempdir()) / f"dex_press_go_{os.getpid()}"
        self.token.unlink(missing_ok=True)
        self.p = subprocess.Popen(
            [PY, "initialization/press_buttons.py", *floors,
             "--speed", str(speed), "--press-speed", str(contact),
             *(["--push", str(push)] if push is not None else []),
             "--camera", "cam_arm", "--panel", "mock_cabinet",
             "--no-obstacle-check", "--go", "--wait-go", str(self.token)],
            cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self.out: list[str] = []
        self.ready = False

    def wait_ready(self, timeout: float = 60.0) -> bool:
        """Read until the child says it has finished the startup it can do early."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            line = self.p.stdout.readline()
            if not line:
                break
            self.out.append(line)
            sys.stdout.write("    (prewarm) " + line)
            sys.stdout.flush()
            if "PREWARM_READY" in line:
                self.ready = True
                return True
        return False

    def release(self, timeout: float = 400.0) -> tuple[int, str]:
        """Let it press, then stream the rest of its output."""
        self.token.touch()
        try:
            for line in self.p.stdout:
                self.out.append(line)
                sys.stdout.write("    " + line)
                sys.stdout.flush()
            self.p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            return 124, "".join(self.out)
        finally:
            self.token.unlink(missing_ok=True)
        return self.p.returncode, "".join(self.out)

    def kill(self) -> None:
        self.token.unlink(missing_ok=True)
        if self.p.poll() is None:
            self.p.kill()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass


def busy() -> str | None:
    """Another of our tools already driving or moving the arm? Name it, do not guess.

    Matched on the interpreter plus the script path, never with `pgrep -f` on a pattern
    that would also match this process's own command line — that self-match once stopped
    a pre-flight guard from ever passing.
    """
    ps = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    for line in ps.splitlines()[1:]:
        if "python3" not in line:
            continue
        if any(t in line for t in ("drive_straight.py", "press_buttons.py",
                                   "move_arm_staged.py")):
            return line.strip()
    return None


def station() -> dict:
    with open(os.path.join(REPO, "configs", "stations.yaml")) as fh:
        return yaml.safe_load(fh)["stations"][0]


def settled_pose(timeout: float = 20.0):
    """Chassis pose, once two consecutive FRESH samples agree.

    Freshness is the SEQUENCE, not the value: a stationary robot republishes the same
    pose, so a "wait for it to change" test collects one sample and never agrees. The ori
    tolerance sits above the feed's own 0.573 deg quantisation for the same reason.
    """
    from initialization.drive_straight import Chassis

    ch = Chassis()
    ch.check_identity()
    ch.open_stream()
    try:
        seen, last_seq = [], -1
        t0 = time.time()
        while time.time() - t0 < timeout and len(seen) < 4:
            ch.pump(0.2)
            if ch.pose and ch.pose_seq != last_seq:
                seen.append(ch.pose)
                last_seq = ch.pose_seq
        for a, b in zip(seen, seen[1:]):
            if (abs(a[0] - b[0]) < 0.01 and abs(a[1] - b[1]) < 0.01
                    and abs(a[2] - b[2]) < 0.021):
                return b, True
        return (seen[-1] if seen else None), False
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


def in_robot_frame(target: dict, pose) -> tuple[float, float, float]:
    """Where `target` sits in the ROBOT's own frame: along its heading, and lateral.

    The robot can only travel along its heading, so this is the split that decides
    whether a straight leg can reach a point at all — one combined distance hides the
    lateral component, which is the sensitive axis here.
    """
    x, y, o = pose
    dx, dy = target["x"] - x, target["y"] - y
    fwd = dx * math.cos(o) + dy * math.sin(o)
    lat = -dx * math.sin(o) + dy * math.cos(o)
    dori = (target["ori"] - o + math.pi) % (2 * math.pi) - math.pi
    return fwd, lat, math.degrees(dori)


def arm_joints():
    from core.config import load_pipeline
    from core.robot.realman import RealmanArm

    c = load_pipeline()["arm"]
    a = RealmanArm(side=c.get("side", "right"), ip=c.get("ip"), port=c.get("port", 8080))
    a.connect()
    try:
        return [float(v) for v in a.get_joint_angles()]
    finally:
        a.disconnect()


def worst(a, b) -> float:
    return max(abs(x - y) for x, y in zip(a, b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=1)
    ap.add_argument("--floors", default="1 open 4")
    ap.add_argument("--speed", type=float, default=0.18, metavar="MPS",
                    help="chassis speed for the straight legs")
    ap.add_argument("--arm-speed", type=float, default=0.50, metavar="FRAC",
                    help="arm speed for the extend/retract legs. Raised 0.20 -> 0.50 on "
                         "2026-09-18 by the operator's explicit call, worth ~6 s a "
                         "cycle. It was held back until then for a reason worth keeping "
                         "in view: these legs are a ~264 deg sweep ACROSS THE ROOM, and "
                         "move_arm_staged checks self-collision, the panel and the "
                         "torso column but has NO MODEL OF THE ROOM — so the 50 % that "
                         "is right inside the small checked volume in front of the "
                         "panel was not automatically right here. Lower it again for an "
                         "unattended run or a cell that has been rearranged. (0.08 on "
                         "the first watched run cost about as much as the whole press.)")
    # A COMMA LIST runs the press once per value from the SAME station visit, which
    # is the only way to compare speeds without the station, the localisation and the
    # room light changing underneath the comparison. The drive is the expensive part;
    # a second press costs seconds.
    ap.add_argument("--press-speed", default="0.50", metavar="FRAC[,FRAC...]",
                    help="press free-air joint speed(s); a list runs the press once each")
    ap.add_argument("--no-prewarm", action="store_true",
                    help="launch the press cold instead of starting its ~7.5 s of "
                         "position-independent startup under the drive")
    ap.add_argument("--press-push", default="", metavar="MM[,MM...]",
                    help="push depth(s) in mm; a list runs the press once each. "
                         "Empty = whatever configs/pipeline.yaml says. Force at the "
                         "button is only judgeable by the operator, so offering two "
                         "depths in ONE station visit is the cheap way to choose.")
    ap.add_argument("--press-contact-speed", type=float, default=0.15, metavar="FRAC",
                    help="press straight-line contact speed; raising this turns "
                         "speed into impact force, so change it deliberately")
    ap.add_argument("--land-tol", type=float, default=0.06, metavar="M",
                    help="the arm will not extend if the base landed further than this "
                         "from the station in either axis (default 0.06)")
    ap.add_argument("--dry", action="store_true",
                    help="measure and plan, never move")
    args = ap.parse_args()

    from core.config import load_pipeline

    cfg = load_pipeline()["arm"]
    st = station()
    view = [float(v) for v in cfg["view_joints_deg"]]
    travel = [float(v) for v in cfg["home_joints_deg"]]
    vs, ts = ",".join(str(v) for v in view), ",".join(str(v) for v in travel)

    running = busy()
    if running:
        print(f"!! already running: {running}")
        return 1

    results = []
    for n in range(1, args.cycles + 1):
        print(f"\n{'=' * 64}\nCYCLE {n} of {args.cycles}\n{'=' * 64}")
        rec = {"cycle": n}

        q = arm_joints()
        print(f"[1] arm is {worst(q, travel):.3f} deg from the travel pose")
        if worst(q, travel) > 1.0:
            if args.dry:
                print("    dry run: would retract before driving")
            else:
                print("    retracting before any drive")
                if sh([PY, "initialization/move_arm_staged.py", f"--to={ts}",
                       "--go", "--speed", str(args.arm_speed)])[0] != 0:
                    print("!! retract failed"); return 1

        pose, ok = settled_pose()
        if pose is None:
            print("!! no chassis pose"); return 1
        fwd, lat, dori = in_robot_frame(st["chassis_pose"], pose)
        print(f"[2] at x={pose[0]:.4f} y={pose[1]:.4f} ori={pose[2]:.4f} settled={ok}")
        print(f"    station is {fwd:+.3f} m along the heading, {lat * 100:+.1f} cm LATERAL, "
              f"heading {dori:+.2f} deg")
        rec["in_lat_cm"] = lat * 100

        if args.dry:
            print("    dry run: stopping before the drive")
            return 0

        # Start the press's position-independent startup NOW, so its ~7.5 s runs
        # under the drive instead of after it. It cannot act: it blocks on a token
        # this file writes only once the landing gate and the arm extension pass.
        speeds = [float(v) for v in str(args.press_speed).split(",") if v.strip()]
        pushes = [float(v) for v in str(args.press_push).split(",") if v.strip()] or [None]
        runs = [(sp, pu) for sp in speeds for pu in pushes]
        warm = None
        if not args.no_prewarm:
            print(f"[3a] pre-warming the press ({runs[0][0] * 100:.0f}% free-air"
                  + (f", push {runs[0][1]:.1f} mm" if runs[0][1] is not None else "")
                  + ") while the base drives")
            warm = Prewarm(args.floors.split(), runs[0][0],
                           args.press_contact_speed, runs[0][1])

        print(f"[3] driving in {fwd:+.3f} m")
        if sh([PY, "initialization/drive_straight.py", "move", "--speed", str(args.speed),
               "--tol", "0.03", "--corrections", "2", "--", f"{fwd:.3f}"])[0] != 0:
            print("!! drive in failed")
            if warm: warm.kill()
            return 1

        pose, ok = settled_pose()
        fwd2, lat2, dori2 = in_robot_frame(st["chassis_pose"], pose)
        print(f"[4] landed {math.hypot(fwd2, lat2) * 1000:.0f} mm from the station "
              f"({fwd2 * 1000:+.0f} mm along, {lat2 * 1000:+.0f} mm lateral, "
              f"{dori2:+.2f} deg)")
        rec["land_along_mm"], rec["land_lat_mm"] = fwd2 * 1000, lat2 * 1000
        if max(abs(fwd2), abs(lat2)) > args.land_tol:
            print(f"!! landed further than {args.land_tol * 100:.0f} cm from the station "
                  f"— NOT extending the arm. The staged mover checks the tip against a "
                  f"panel at a fixed distance, and that constant is optimistic once the "
                  f"base is closer than it assumes.")
            if warm: warm.kill()
            return 1

        if warm is not None:
            # Drain and confirm BEFORE the arm moves. Draining is not optional: an
            # unread pipe fills and blocks the child mid-startup, which would look
            # like the pre-warm simply being slow.
            if warm.wait_ready():
                print("[4a] pre-warm ready — its startup ran under the drive")
            else:
                print("[4a] pre-warm did NOT report ready; falling back to a cold press")
                warm.kill()
                warm = None

        print("[5] extending the arm to the viewing pose")
        if sh([PY, "initialization/move_arm_staged.py", f"--to={vs}",
               "--go", "--speed", str(args.arm_speed)])[0] != 0:
            print("!! extend failed")
            if warm: warm.kill()
            return 1
        q = arm_joints()
        d = worst(q, view)
        print(f"    on the registered viewing vector to {d:.3f} deg "
              f"-> aim offset {'APPLIES' if d <= 1.0 else 'WILL BE REFUSED'}")
        rec["view_err_deg"] = d

        rec["pressed"], rec["press_ok"] = 0, False
        for i, (sp, pu) in enumerate(runs, 1):
            print(f"[6.{i}] pressing {args.floors} at free-air {sp * 100:.0f}% / "
                  f"contact {args.press_contact_speed * 100:.0f}%"
                  + (f" / push {pu:.1f} mm" if pu is not None else "")
                  + ("  [pre-warmed]" if (i == 1 and warm and warm.ready) else ""))
            if i == 1 and warm is not None:
                # Only the first run can be pre-warmed; the rest of a sweep is a
                # measurement convenience and pays its own startup.
                t_go = time.time()
                code, out = warm.release()
                warm = None
                print(f"    released -> press finished {time.time() - t_go:.1f} s later")
            else:
                code, out = sh([PY, "initialization/press_buttons.py",
                                *args.floors.split(),
                                "--speed", str(sp),
                                "--press-speed", str(args.press_contact_speed),
                                *(["--push", str(pu)] if pu is not None else []),
                                "--camera", "cam_arm", "--panel", "mock_cabinet",
                                "--no-obstacle-check", "--go"], timeout=400)
            pressed = [l for l in out.splitlines() if "pressed" in l and "mm" in l]
            rec[f"pressed_{sp:g}" + (f"_{pu:g}" if pu is not None else "")] = len(pressed)
            rec["pressed"] += len(pressed)
            rec["press_ok"] = rec["press_ok"] or ("pressed in" in out)

        print("[7] retracting")
        if sh([PY, "initialization/move_arm_staged.py", f"--to={ts}",
               "--go", "--speed", str(args.arm_speed)])[0] != 0:
            print("!! retract failed"); return 1

        pose, ok = settled_pose()
        back, blat, bdori = in_robot_frame(st["approach_pose"], pose)
        print(f"[8] driving back {back:+.3f} m to the approach pose")
        if sh([PY, "initialization/drive_straight.py", "move", "--speed", str(args.speed),
               "--tol", "0.03", "--corrections", "2", "--", f"{back:.3f}"])[0] != 0:
            print("!! drive back failed"); return 1
        pose, ok = settled_pose()
        f3, l3, d3 = in_robot_frame(st["approach_pose"], pose)
        print(f"[9] back at {math.hypot(f3, l3) * 1000:.0f} mm from the approach pose, "
              f"heading {d3:+.2f} deg")
        rec["back_mm"], rec["back_heading_deg"] = math.hypot(f3, l3) * 1000, d3
        results.append(rec)

    print(f"\n{'=' * 64}\nSUMMARY\n{'=' * 64}")
    print("  cyc | lat in | landed (along/lat) | view err | pressed | back | heading")
    for r in results:
        print("  %3d | %+5.1f cm | %+5.0f / %+5.0f mm   | %6.3f d | %5s   | %4.0f mm | %+5.2f d"
              % (r["cycle"], r["in_lat_cm"], r["land_along_mm"], r["land_lat_mm"],
                 r["view_err_deg"], r["pressed"], r["back_mm"], r["back_heading_deg"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
