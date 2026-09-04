#!/usr/bin/env python3
"""Measure the straight-leg ratio SEPARATELY FOR EACH DIRECTION.

    python3 initialization/calibrate_legs.py                 # both directions
    python3 initialization/calibrate_legs.py --only back     # arm-ward only
    python3 initialization/calibrate_legs.py --dists 0.5,1.0,1.5

WHY. `SPEED_RATIO` is one number applied to both directions, and this chassis is not
symmetric. Measured over three complete runs on 2026-09-04, the FIRST leg of a commanded
−1.400 m arm-ward move achieved −0.842, −0.851 and −0.836 m: **59.7 / 60.1 / 60.8 %**,
three runs in a row. The other way it OVER-delivers — 2.826 m of a commanded 2.70, +4.7 %.
A single shared constant cannot describe both, so the correction legs quietly make up the
difference on every move, which is the "it stopped and then shuffled again" the operator
noticed.

That deficit is a CONSTANT, not noise, which is the whole point: a repeatable error can be
divided out, and then the first leg lands close enough that the correction rarely fires.

METHOD. Each measurement is ONE uncorrected leg, so it shows what the base actually does
rather than what the correction loop papers over. After each measurement the robot is
driven back WITH corrections, so the test stays in one place instead of walking across the
room — the restore legs are deliberately not measured. Everything runs in a single process
holding `remote`, because the chassis's own control_unit issues charge moves in the gaps
between separate invocations and drives the robot away underneath the measurement.

It fits `achieved = k * commanded - c` per direction, the same fixed-loss-plus-proportional
shape that the ramp deficit turned out to have. Report both terms: a pure ratio would hide
a fixed loss, and a fixed loss is exactly what makes short legs behave differently from
long ones.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drive_straight import (  # noqa: E402
    MAX_SPEED, SPEED_RATIO, Chassis, _leg, _settled_pose, drive,
)


def _fit(points):
    """Least squares on achieved = k * commanded - c. Returns (k, c, rms)."""
    n = len(points)
    if n < 2:
        return None
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sxx = sum(p[0] * p[0] for p in points)
    sxy = sum(p[0] * p[1] for p in points)
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return None
    k = (n * sxy - sx * sy) / den
    b = (sy - k * sx) / n
    rms = math.sqrt(sum((k * x + b - y) ** 2 for x, y in points) / n)
    return k, -b, rms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dists", default="0.4,0.8,1.2,1.6",
                    help="comma-separated leg lengths, metres")
    ap.add_argument("--only", choices=("fwd", "back", "both"), default="both",
                    help="fwd = +ori (the base's own front); back = -ori (arm-ward)")
    ap.add_argument("--speed", type=float, default=0.20)
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    dists = [float(d) for d in args.dists.split(",") if d.strip()]
    dirs = []
    if args.only in ("fwd", "both"):
        dirs.append(("+ori (base front)", +1.0))
    if args.only in ("back", "both"):
        dirs.append(("-ori (arm-ward)", -1.0))

    ch = Chassis()
    ch.check_identity()
    ch.open_stream()
    results = {name: [] for name, _ in dirs}
    try:
        ch.pump(1.0)
        if ch.estop:
            return print("ABORT: emergency stop is pressed") or 4
        ch.enter_remote()
        ch.pump(0.3)
        print(f"speed {args.speed:.2f} m/s, current SPEED_RATIO {SPEED_RATIO} "
              f"(shared by both directions today)")

        for name, sign in dirs:
            print(f"\n=== {name} ===")
            for d in dists:
                for rep in range(args.repeats):
                    metres = sign * d
                    try:
                        ch.gate_clearance(metres)
                    except SystemExit as e:
                        print(f"  {metres:+.2f} m: skipped — {e}")
                        continue
                    ch.enter_remote()
                    got, d_ori, reason = _leg(ch, metres, min(args.speed, MAX_SPEED),
                                              4.0, verbose=False)
                    ratio = got / metres if metres else float("nan")
                    print(f"  commanded {metres:+.2f} -> achieved {got:+.3f} m "
                          f"({ratio * 100:.1f} %), heading {d_ori:+.2f} deg ({reason})")
                    if reason == "leg complete":
                        results[name].append((abs(metres), abs(got)))
                    # go back WITH corrections so the test stays put; not measured
                    try:
                        ch.gate_clearance(-metres)
                        drive(ch, -metres, min(args.speed, MAX_SPEED), 4.0, 0.02, 3)
                    except SystemExit as e:
                        print(f"    (restore refused: {e}) — stopping this direction")
                        break
                    ch.pump(0.1)
                    if ch.estop:
                        return print("ABORT: emergency stop pressed mid-run") or 4

        print("\n=== fit: achieved = k * commanded - c ===")
        for name, _ in dirs:
            pts = results[name]
            f = _fit(pts)
            if f is None:
                print(f"  {name}: not enough clean legs ({len(pts)})")
                continue
            k, c, rms = f
            print(f"  {name}: k = {k:.3f}, c = {c * 1000:+.0f} mm, rms {rms * 1000:.0f} mm "
                  f"({len(pts)} legs)")
            print(f"      -> to land on target, issue  commanded / {k:.3f} + "
                  f"{c * 1000:+.0f} mm")
        return 0
    finally:
        ch.close()


if __name__ == "__main__":
    raise SystemExit(main())
