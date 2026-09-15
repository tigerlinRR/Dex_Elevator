#!/usr/bin/env python3
"""Capture eye-in-hand samples by SWEEPING A WRIST JOINT, not by hand-guiding.

Why this exists. Three hand-guided rounds on this robot (56 samples) all came out
with 81-92% of their rotation energy on a single axis — the base-frame Z — and the
solve is unobservable in that direction no matter how many samples are added. The
cause is mechanical, not operator error: when a person hand-guides an arm they hold
it and move it to a new *place*, and the orientation change that comes with that is
dominated by J1, the base rotation, which IS the Z axis. Deliberately spinning the
last joint while holding everything else still is not a motion a hand makes.

A program makes it trivially. Measured on this cell, the camera's optical axis sits
2 degrees off the TCP's z axis, i.e. **J6 rotates the camera about its own line of
sight**. That is exactly the rotation the AX = XB translation term needs, and it has
two properties that make it safe to automate:

* the arm does not translate — only the last joint turns, so nothing approaches the
  board, the robot, or the cabling envelope beyond a known twist;
* the optical axis does not move, so the board stays in view for every sample. It
  simply spins in the image.

A second, smaller sweep of J5 (wrist pitch) adds a third axis. That one DOES swing
the line of sight, so its range is kept small and any pose where the board is not
detected is skipped rather than forced.

Cable note: J6 twists whatever is routed through the wrist. The default range is
deliberately modest and the sweep is stepped with a settle between points so a human
watching can stop it. Widen it only after watching one full pass.

    python3 initialization/sweep_eye_in_hand_poses.py --out-dir cam_arm_samples_j6scan
    python3 initialization/sweep_eye_in_hand_poses.py --j6-range 45 --j6-step 15   # gentler
    python3 initialization/sweep_eye_in_hand_poses.py --dry-run                    # no motion
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.board import CharucoBoard
from calibration.eye_in_hand import EyeInHandSession, rotation_diversity
from calibration.io import CALIB_DIR
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm

JOINT_LIMITS_DEG = [178, 130, 135, 178, 128, 360]   # read from this controller


def _clamp(joints: np.ndarray) -> tuple[np.ndarray, bool]:
    """Clamp to the controller's real joint limits; report whether anything was clipped."""
    out = joints.copy()
    clipped = False
    for i, lim in enumerate(JOINT_LIMITS_DEG):
        if abs(out[i]) > lim - 2.0:                  # keep 2 deg off the hard stop
            out[i] = np.sign(out[i]) * (lim - 2.0)
            clipped = True
    return out, clipped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_arm")
    ap.add_argument("--out-dir", default="cam_arm_samples_j6scan",
                    help="sample dir under data/calibration/ (must be empty or new)")
    ap.add_argument("--j6-range", type=float, default=75.0,
                    help="+/- degrees to spin the wrist about the optical axis (default 75)")
    ap.add_argument("--j6-step", type=float, default=25.0, help="degrees between samples")
    ap.add_argument("--j5-range", type=float, default=15.0,
                    help="+/- degrees of wrist pitch, a smaller secondary axis (default 15)")
    ap.add_argument("--j5-step", type=float, default=7.5, help="degrees between samples")
    ap.add_argument("--speed", type=float, default=0.06,
                    help="joint speed fraction, deliberately slow (default 0.06)")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="seconds to wait after each move before capturing (default 1.5)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, move nothing")
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

    arm_cfg = load_pipeline().get("arm", {})
    arm = RealmanArm(side=arm_cfg.get("side", "right"), ip=arm_cfg.get("ip"),
                     port=arm_cfg.get("port", 8080),
                     max_speed_pct=arm_cfg.get("max_speed_pct", 20))
    arm.connect()
    home = np.asarray(arm.get_joint_angles(), dtype=float)
    print("start joints (deg):", np.array2string(home, precision=1))

    # Build the plan. J6 spins the camera about its own optical axis, which is what
    # the translation term needs — EXCEPT along that axis itself, where a pure J6
    # sweep is blind: (R_A - I) is singular along the rotation axis, so t_z never
    # enters the equations. J5 (wrist pitch) turns about an axis PERPENDICULAR to
    # the optical axis, and that is what pins t_z down. Measured on this cell: two
    # J6-dominated sweeps agreed to 0.1 mm in x and y and disagreed by 12 mm in z.
    #
    # J5's usable range is asymmetric in practice — the wrist often sits near one of
    # its limits — so it is computed from the limit rather than assumed symmetric.
    # An asymmetric 60 deg of real travel beats a symmetric +/-15 that half fits.
    plan: list[tuple[str, np.ndarray]] = []
    j6_offsets = np.arange(-args.j6_range, args.j6_range + 1e-6, args.j6_step)
    for d in j6_offsets:
        j = home.copy(); j[5] += d
        plan.append((f"J6 {d:+.0f}", j))

    lim5 = JOINT_LIMITS_DEG[4] - 2.0
    lo5 = max(home[4] - args.j5_range, -lim5)
    hi5 = min(home[4] + args.j5_range, lim5)
    n5 = max(2, int(round((hi5 - lo5) / args.j5_step)) + 1)
    j5_targets = np.linspace(lo5, hi5, n5)
    print(f"J5 usable range: {lo5:+.1f}..{hi5:+.1f} deg "
          f"({hi5 - lo5:.0f} deg of travel, {n5} points) around home {home[4]:+.1f}")
    for t5 in j5_targets:
        if abs(t5 - home[4]) < 1e-6:
            continue
        j = home.copy(); j[4] = t5
        plan.append((f"J5 ->{t5:+.0f}", j))

    # Combined poses: two axes moving at once breaks the degeneracy fastest. The J5
    # component uses the real usable span, not a fixed small number.
    span5 = (hi5 - lo5) / 2.0
    for f6, f5 in ((-0.7, 0.8), (0.7, -0.8), (-0.35, -0.4), (0.35, 0.4)):
        j = home.copy()
        j[5] += f6 * args.j6_range
        j[4] = np.clip(home[4] + f5 * span5, lo5, hi5)
        plan.append((f"J6 {f6 * args.j6_range:+.0f} J5 ->{j[4]:+.0f}", j))

    print(f"\nplan: {len(plan)} poses "
          f"(J6 {j6_offsets[0]:+.0f}..{j6_offsets[-1]:+.0f}, J5 +/-{args.j5_range:.0f}, "
          f"speed {args.speed}, settle {args.settle}s)")
    for name, j in plan:
        clamped, hit = _clamp(j)
        if hit:
            print(f"   {name:<18} WOULD HIT A JOINT LIMIT — will be clamped")
    if args.dry_run:
        print("--dry-run: no motion, no capture")
        arm.disconnect()
        return 0

    sample_dir = CALIB_DIR / args.out_dir
    if sample_dir.exists() and list(sample_dir.glob("sample*.npz")):
        print(f"!! {sample_dir} already holds samples — pick an empty --out-dir")
        arm.disconnect()
        return 1
    session = EyeInHandSession(board=board, sample_dir=sample_dir,
                               min_angle_deg=0.0, min_trans_m=0.0)  # the plan IS the diversity

    handle = CameraManager().build().get(args.camera)
    handle.camera.start()
    taken, skipped = 0, 0
    try:
        for idx, (name, target) in enumerate(plan, 1):
            target, hit = _clamp(target)
            if hit:
                print(f"[{idx}/{len(plan)}] {name}: clamped to a joint limit — skipped")
                skipped += 1
                continue
            print(f"[{idx}/{len(plan)}] {name}: moving...", flush=True)
            ok = arm.move_joints_sync(list(target), speed=args.speed)
            if not ok:
                # False means arrival was not OBSERVED, not that the arm stopped —
                # rm_movej keeps executing in the background. Stop it before deciding.
                print("   !! arrival not confirmed — stopping the arm and aborting")
                try:
                    arm._require().rm_set_arm_stop()
                    arm._require().rm_set_delete_current_trajectory()
                except Exception:
                    pass
                break
            time.sleep(args.settle)
            frame = handle.camera.capture()
            cam_T_target = session.detect_board(frame.rgb, frame.intrinsics)
            if cam_T_target is None:
                print("   board not detected here — skipped (not an error)")
                skipped += 1
                continue
            base_T_gripper = arm.get_tcp_pose()
            session.add_sample(base_T_gripper, cam_T_target, rgb=frame.rgb)
            taken += 1
            div = rotation_diversity(session.base_T_gripper)
            print(f"   captured {taken}  |  axis spread {div['axis_ratio']:.3f}  "
                  f"max rot {div['max_angle_deg']:.0f} deg")
    except KeyboardInterrupt:
        print("\ninterrupted — stopping the arm")
        try:
            arm._require().rm_set_arm_stop()
        except Exception:
            pass
    finally:
        print(f"\nreturning to the start pose")
        try:
            arm.move_joints_sync(list(home), speed=args.speed)
        except Exception as e:
            print(f"!! could not return to start: {e}")
        handle.camera.stop()
        arm.disconnect()

    print(f"\n{taken} samples captured ({skipped} skipped) -> {sample_dir}")
    if taken:
        div = rotation_diversity(session.base_T_gripper)
        print(f"this sweep on its own: axis spread {div['axis_ratio']:.3f}, "
              f"max rotation {div['max_angle_deg']:.0f} deg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
