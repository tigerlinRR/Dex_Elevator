#!/usr/bin/env python3
"""Solve eye-in-hand from SEVERAL sample directories, with outlier rejection.

Capture rarely happens in one clean run. A cable fails, a round turns out to be
degenerate and needs topping up with different rotation axes, the operator stops
for the day. Each of those leaves a separate sample directory, and
``run_calibration_arm_cam.py --from-samples`` only takes one.

Merging is legitimate **only while ``gripper_T_camera`` has not changed** — the
camera must not have been moved on its bracket between rounds, and the board must
have stayed put. Neither is checkable from the files, so it is the operator's call;
what this script CAN do is show whether the rounds agree once merged, which is what
the per-round residual table is for. A round that was captured after the camera
shifted shows up there as a whole block of large residuals.

It also reports what each round contributes to the CONDITIONING, because that is
usually why a second round exists: one round of pure yaw plus one round of roll is
solvable, while two rounds of yaw are not, no matter how many samples they hold.

    python3 initialization/solve_eye_in_hand_merged.py \
        data/calibration/cam_arm_samples_round1 \
        data/calibration/cam_arm_samples_round2a \
        data/calibration/cam_arm_samples --out cam_arm.npy

    # see what it would do without writing:
    ... --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from calibration.board import CharucoBoard
from calibration.eye_in_hand import (
    EyeInHandSession,
    consistency_residual,
    rotation_diversity,
    save_eye_in_hand,
    save_eye_in_hand_record,
    select_best,
    solve_eye_in_hand_all_methods,
    validate_eye_in_hand,
)
from calibration.io import CALIB_DIR
from calibration.validate import save_report


def _load(dirs: list[str], board: CharucoBoard):
    """Load every round, remembering which round each sample came from."""
    bg, ct, tags = [], [], []
    for d in dirs:
        p = Path(d)
        if not p.is_absolute():
            p = Path.cwd() / p
        s = EyeInHandSession.from_samples(p, board=board)
        bg += s.base_T_gripper
        ct += s.cam_T_target
        tags += [p.name] * s.num_samples
    return bg, ct, tags


def _report(bg, ct, tags, label: str):
    """Solve and print the per-round breakdown."""
    cand = solve_eye_in_hand_all_methods(bg, ct)
    method, X = select_best(cand, bg, ct)
    res = consistency_residual(X, bg, ct)
    div = rotation_diversity(bg)
    print(f"\n=== {label}: {len(bg)} samples, method={method} ===")
    print(f"  base_T_board consistency : {res['trans_m']['mean']*1000:7.2f} mm mean, "
          f"{res['trans_m']['max']*1000:.2f} max")
    print(f"  rotation consistency     : {res['rot_deg']['mean']:7.3f} deg mean, "
          f"{res['rot_deg']['max']:.3f} max")
    print(f"  rotation-axis spread     : {div['axis_ratio']:7.3f}  (need > 0.15)")
    print(f"  largest relative rotation: {div['max_angle_deg']:7.1f} deg")

    # Per-round: does one round disagree with the rest? That is what a camera that
    # moved between rounds looks like — a whole block of large residuals, not a
    # scatter of individual bad samples.
    boards = [bg[i] @ X @ ct[i] for i in range(len(bg))]
    P = np.array([T[:3, 3] for T in boards])
    err = np.linalg.norm(P - P.mean(0), axis=1) * 1000
    print("  per-round residual (mm):")
    for t in sorted(set(tags)):
        m = np.array([i for i, x in enumerate(tags) if x == t])
        print(f"    {t:<34} n={len(m):3d}  median {np.median(err[m]):6.1f}  "
              f"max {err[m].max():6.1f}")
    return X, method, cand, res, div, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", help="sample directories to merge, in any order")
    ap.add_argument("--out", default="cam_arm.npy", help="output .npy under data/calibration/")
    ap.add_argument("--reject", type=float, default=2.5,
                    help="drop samples whose residual exceeds FACTOR x the median (default 2.5); "
                         "0 disables rejection")
    ap.add_argument("--min-samples", type=int, default=12, help="never go below this many")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--squares-x", type=int, default=14)
    ap.add_argument("--squares-y", type=int, default=9)
    ap.add_argument("--square-len", type=float, default=0.040)
    ap.add_argument("--marker-len", type=float, default=0.030)
    ap.add_argument("--dictionary", default="DICT_5X5_100")
    args = ap.parse_args()

    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )
    bg, ct, tags = _load(args.dirs, board)
    if len(bg) < 6:
        print(f"only {len(bg)} samples across {len(args.dirs)} dirs — need more")
        return 1

    # What each round contributes to conditioning, on its own. This is the number
    # that explains why merging helps: two rounds of the same axis do not add up.
    print("\n=== conditioning per round (on its own) ===")
    for t in sorted(set(tags)):
        m = [i for i, x in enumerate(tags) if x == t]
        d = rotation_diversity([bg[i] for i in m])
        print(f"  {t:<34} n={len(m):3d}  axis_ratio {d['axis_ratio']:.3f}  "
              f"max_rot {d['max_angle_deg']:5.1f} deg")

    X, method, cand, res, div, err = _report(bg, ct, tags, "MERGED, all samples")

    if args.reject > 0:
        limit = args.reject * float(np.median(err))
        keep = [i for i, e in enumerate(err) if e <= limit]
        if len(keep) >= args.min_samples and len(keep) < len(err):
            print(f"\ndropping {len(err) - len(keep)} samples above "
                  f"{args.reject}x median ({limit:.1f} mm)")
            bg = [bg[i] for i in keep]
            ct = [ct[i] for i in keep]
            tags = [tags[i] for i in keep]
            X, method, cand, res, div, err = _report(bg, ct, tags, "after rejection")
        else:
            print(f"\nnothing to reject at {args.reject}x median")

    t = X[:3, 3] * 1000.0
    print(f"\ngripper_T_camera: camera at [{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}] mm from the TCP "
          f"frame — check that against a tape measure before trusting it")

    rep = validate_eye_in_hand(X, bg, ct, all_methods=cand)
    print(rep.summary())

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0 if rep.status != "fail" else 1

    path = save_eye_in_hand(args.out, X)
    save_eye_in_hand_record(args.out, X, method, cand, res, len(bg), div)
    save_report(rep, CALIB_DIR / f"{Path(args.out).stem}_extrinsic_validation.json")
    print(f"\nsaved gripper_T_camera -> {path}")
    return 0 if rep.status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
