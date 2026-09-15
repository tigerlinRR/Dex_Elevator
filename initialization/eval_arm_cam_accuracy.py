#!/usr/bin/env python3
"""End-to-end localisation accuracy for the arm-mounted camera, on HELD-OUT poses.

The question a press depends on: when this camera sees something and we compute its
position in the arm base frame, how far off is it? The calibration's own consistency
residual does not answer that — it is measured on the samples the solve minimised.

The board stays where it was for calibration, so its base-frame position is a
constant. At each pose we compute

    base_T_board = base_T_gripper @ gripper_T_camera @ cam_T_board

and measure the spread. Unlike the fixed chest camera, this number contains the arm's
forward-kinematics and joint repeatability error as well as the calibration's, because
the camera now rides through them — which is exactly what a press has to tolerate.

**Two numbers, and they differ a lot:**

* *Repeatability at one pose* — return to the SAME pose repeatedly and measure the
  spread of those readings **about their own mean**, not about the global reference.
  Those are different questions and the distinction is not academic: measured here,
  four returns to one pose read 3.50 / 2.94 / 3.30 / 3.09 mm from the global
  reference — a 3.2 mm BIAS belonging to that pose, while the repeatability hiding
  inside it is 0.56 mm of spread. Reporting the first as "repeatability" makes a very
  steady camera look ten times worse than it is.
* *Accuracy across poses* — different poses. This is what governs looking from
  wherever the arm happens to be.

The poses here are deliberately NOT the ones the sweeps used: different J6/J5 values
plus J4, which no calibration sweep touched at all.

    python3 initialization/eval_arm_cam_accuracy.py
    python3 initialization/eval_arm_cam_accuracy.py --dry-run
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.eye_in_hand import EyeInHandSession
from calibration.io import CALIB_DIR, load_extrinsic
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm

JOINT_LIMITS_DEG = [178, 130, 135, 178, 128, 360]
CAL_ROUNDS = ("cam_arm_samples_j6scan", "cam_arm_samples_j6scan2", "cam_arm_samples_j5scan")


def _safe(j: np.ndarray) -> bool:
    return all(abs(j[i]) <= JOINT_LIMITS_DEG[i] - 2.0 for i in range(6))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_arm")
    ap.add_argument("--repeats", type=int, default=4,
                    help="how many times to return to the reference pose (repeatability)")
    ap.add_argument("--speed", type=float, default=0.08)
    ap.add_argument("--settle", type=float, default=1.5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--square-len", type=float, default=0.040)
    ap.add_argument("--marker-len", type=float, default=0.030)
    args = ap.parse_args()

    board = CharucoBoard(squares_x=14, squares_y=9, square_length=args.square_len,
                         marker_length=args.marker_len, dictionary=cv2.aruco.DICT_5X5_100)
    det = BoardDetector(board=board, accurate=True)
    X = load_extrinsic(f"{args.camera}.npy")

    # Reference: where the board sat according to the calibration samples themselves.
    pts = []
    for r in CAL_ROUNDS:
        try:
            s = EyeInHandSession.from_samples(CALIB_DIR / r, board=board)
        except FileNotFoundError:
            continue
        pts += [(bg @ X @ ct)[:3, 3] for bg, ct in zip(s.base_T_gripper, s.cam_T_target)]
    if not pts:
        print("no calibration samples found — cannot establish a reference")
        return 1
    ref = np.mean(pts, axis=0)
    cal_spread = np.linalg.norm(np.array(pts) - ref, axis=1) * 1000
    print(f"reference from {len(pts)} calibration samples: "
          f"{np.array2string(ref * 1000, precision=1)} mm")
    print(f"  (their own spread: median {np.median(cal_spread):.2f} mm, "
          f"max {cal_spread.max():.2f} — the floor for what follows)")

    arm_cfg = load_pipeline().get("arm", {})
    arm = RealmanArm(side=arm_cfg.get("side", "right"), ip=arm_cfg.get("ip"),
                     port=arm_cfg.get("port", 8080),
                     max_speed_pct=arm_cfg.get("max_speed_pct", 20))
    arm.connect()
    home = np.asarray(arm.get_joint_angles(), dtype=float)
    print("start joints:", np.array2string(home, precision=1))

    # HELD-OUT poses: angles the sweeps did not visit, including J4 which none touched.
    plan: list[tuple[str, np.ndarray]] = []
    for k in range(args.repeats):
        plan.append((f"REPEAT {k + 1}", home.copy()))
        if k < args.repeats - 1:                     # move away and come back
            j = home.copy(); j[5] += 35.0
            plan.append((f"(away {k + 1})", j))
    for name, d in (
        ("J6 +11", (5, 11.0)), ("J6 -33", (5, -33.0)), ("J6 +57", (5, 57.0)),
        ("J5 +22", (4, 22.0)), ("J5 -9", (4, -9.0)),
        ("J4 +18", (3, 18.0)), ("J4 -18", (3, -18.0)),
        ("J2 +6", (1, 6.0)), ("J3 -7", (2, -7.0)), ("J1 +5", (0, 5.0)),
    ):
        j = home.copy(); j[d[0]] += d[1]
        plan.append((name, j))

    print(f"\n{len(plan)} poses ({args.repeats} repeats + "
          f"{len(plan) - 2 * args.repeats + 1} held-out)")
    if args.dry_run:
        for n, j in plan:
            print(f"  {n:<12} {'OK' if _safe(j) else 'UNSAFE'}")
        arm.disconnect()
        return 0

    handle = CameraManager().build().get(args.camera)
    handle.camera.start()
    rep_err, new_err, rep_pos = [], [], []
    try:
        for idx, (name, target) in enumerate(plan, 1):
            if not _safe(target):
                print(f"[{idx}/{len(plan)}] {name}: outside joint limits — skipped")
                continue
            if not arm.move_joints_sync(list(target), speed=args.speed):
                print("   !! arrival not confirmed — stopping")
                try:
                    arm._require().rm_set_arm_stop()
                    arm._require().rm_set_delete_current_trajectory()
                except Exception:
                    pass
                break
            if name.startswith("(away"):
                continue
            time.sleep(args.settle)
            frame = handle.camera.capture()
            P = det.detect(frame.rgb, frame.intrinsics.K, frame.intrinsics.D)
            if P is None:
                print(f"[{idx}/{len(plan)}] {name}: board not detected — skipped")
                continue
            here = (arm.get_tcp_pose() @ X @ P)[:3, 3]
            e = float(np.linalg.norm(here - ref)) * 1000
            if name.startswith("REPEAT"):
                rep_pos.append(here)                 # keep the 3D point, not the distance
                rep_err.append(e)
            else:
                new_err.append(e)
            print(f"[{idx}/{len(plan)}] {name:<12} error {e:6.2f} mm")
    except KeyboardInterrupt:
        print("\ninterrupted")
        try:
            arm._require().rm_set_arm_stop()
        except Exception:
            pass
    finally:
        print("\nreturning to the start pose")
        try:
            arm.move_joints_sync(list(home), speed=args.speed)
        except Exception as e:
            print(f"!! could not return: {e}")
        handle.camera.stop()
        arm.disconnect()

    def _stats(a, label):
        if not a:
            print(f"  {label}: no measurements")
            return
        v = np.array(a)
        print(f"  {label}: n={len(v)}  mean {v.mean():.2f}  median {np.median(v):.2f}  "
              f"max {v.max():.2f}  min {v.min():.2f} mm")

    print("\n=== end-to-end localisation error (arm base frame) ===")
    _stats(new_err, "accuracy ACROSS poses (vs reference)")
    if len(rep_pos) >= 2:
        # Repeatability is the spread about the repeats' OWN mean. Their distance to
        # the global reference is a different quantity — that pose's systematic bias —
        # and conflating the two makes a steady camera look far worse than it is.
        Pm = np.array(rep_pos)
        centre = Pm.mean(0)
        spread = np.linalg.norm(Pm - centre, axis=1) * 1000
        bias = float(np.linalg.norm(centre - ref)) * 1000
        print(f"  repeatability at ONE pose          : n={len(Pm)}  "
              f"spread about own mean: mean {spread.mean():.2f}, max {spread.max():.2f} mm")
        print(f"  that pose's systematic bias        : {bias:.2f} mm "
              f"(constant there, so a fixed viewing pose could calibrate it out)")
    if rep_pos and new_err:
        print("\n  What governs a press: the ACROSS-pose figure, unless the press always")
        print("  looks from one fixed pose — then the bias above can be measured once")
        print("  and subtracted, leaving the repeatability figure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
