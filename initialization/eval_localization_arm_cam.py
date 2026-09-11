#!/usr/bin/env python3
"""End-to-end localization accuracy for an ARM-MOUNTED camera (eye-in-hand).

The counterpart to ``eval_localization.py``, which stays as it is for the fixed
chest camera. It answers the same question — when the camera sees something and
we compute its 3D position in the arm base frame, how far off is it? — but for
this mount the answer depends on the ARM POSE, which is the whole point of
measuring it separately.

The ChArUco board stays FIXED in the scene (as during eye-in-hand calibration),
so its base-frame pose is a constant. At each fresh pose we compute

    base_T_board = base_T_gripper @ gripper_T_camera @ cam_T_board

and measure how far that lands from the reference. Unlike the fixed-camera case
this error contains the arm's forward-kinematics and joint repeatability error
as well as the calibration's, because the camera now rides through them — which
is exactly the number a press has to tolerate, so it is the honest one.

**Two things worth measuring separately, because they differ a lot:**

* *Repeatability at one viewing pose* — capture several samples after returning
  to the SAME pose. A systematic FK error is constant there, so it is absorbed
  and the accuracy approaches the fixed-camera case. If the press will always
  look from one or two fixed viewing poses, this is the number that governs it.
* *Accuracy across poses* — capture from many different poses. This is the
  number that governs looking from wherever the arm happens to be.

Reference pose: by default the mean ``base_T_board`` from the calibration
samples, which measures ABSOLUTE accuracy and assumes the board has not moved
since. Pass ``--self-reference`` if the board HAS been moved — the run then
measures spread about this session's own mean, i.e. repeatability only.

Run on the robot:
    python3 initialization/eval_localization_arm_cam.py --camera cam_arm --web
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from calibration.board import CharucoBoard
from calibration.capture_ui import live_capture_loop, web_capture_loop
from calibration.eye_in_hand import EyeInHandSession
from calibration.io import CALIB_DIR, load_extrinsic
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm
from core.transforms import make_transform, matrix_to_quat, quat_to_matrix


def _mean_transform(Ts):
    """Average a list of near-identical 4x4 transforms (mean translation + quaternion)."""
    ts = np.array([T[:3, 3] for T in Ts]).mean(0)
    qs = np.array([matrix_to_quat(T[:3, :3]) for T in Ts])
    qs = qs * np.sign(qs[:, :1] + 1e-9)          # hemisphere-align before averaging
    q = qs.mean(0)
    q = q / np.linalg.norm(q)
    return make_transform(quat_to_matrix(*q), ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_arm")
    ap.add_argument("--samples-dir", default=None,
                    help="calibration samples dir (default data/calibration/<camera>_samples)")
    ap.add_argument("--self-reference", action="store_true",
                    help="measure spread about THIS session's mean instead of the calibration "
                         "board position (use when the board has been moved since calibrating — "
                         "this measures repeatability, not absolute accuracy)")
    ap.add_argument("--web", action="store_true", help="browser MJPEG preview (headless robot)")
    ap.add_argument("--web-port", type=int, default=8010)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--squares-x", type=int, default=14)
    ap.add_argument("--squares-y", type=int, default=9)
    ap.add_argument("--square-len", type=float, default=0.020)
    ap.add_argument("--marker-len", type=float, default=0.015)
    ap.add_argument("--dictionary", default="DICT_5X5_100")
    args = ap.parse_args()

    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )

    X = load_extrinsic(f"{args.camera}.npy")     # gripper_T_camera
    ref = None
    if not args.self_reference:
        sdir = args.samples_dir or str(CALIB_DIR / f"{args.camera}_samples")
        cal = EyeInHandSession.from_samples(sdir, board=board)
        boards = [bg @ X @ ct for bg, ct in zip(cal.base_T_gripper, cal.cam_T_target)]
        ref = _mean_transform(boards)
        bt = np.array([b[:3, 3] for b in boards]) * 1000.0
        print(f"base_T_board from {len(boards)} calib samples; translation spread (mm): "
              f"{np.array2string((bt.max(0) - bt.min(0)), precision=1)} "
              "(this is the calibration's own consistency — the floor for what follows)")

    manager = CameraManager().build()
    handle = manager.get(args.camera)
    if handle.mount != "arm":
        raise SystemExit(f"✗ {args.camera} is `mount: {handle.mount}` — use eval_localization.py "
                         f"for a fixed camera.")
    if handle.extrinsic is None or handle.intrinsics is None:
        raise SystemExit(f"{args.camera} missing intrinsics/extrinsic — calibrate first.")

    arm_cfg = load_pipeline().get("arm", {})
    arm = RealmanArm(side=arm_cfg.get("side", "right"), ip=arm_cfg.get("ip"),
                     port=arm_cfg.get("port", 8080), max_speed_pct=arm_cfg.get("max_speed_pct", 20))
    arm.connect()
    print(f"Connected to RealMan {arm.side} arm at {arm.ip}.")

    det = EyeInHandSession(board=board)          # for detect_board (accurate)
    measured: list[np.ndarray] = []              # base-frame board positions (m)
    errs: list[float] = []

    def _recompute_self_reference() -> None:
        """Self-reference mode: the reference is the running mean, so errors re-derive."""
        if not args.self_reference or not measured:
            return
        centre = np.mean(np.array(measured), axis=0)
        errs[:] = [float(np.linalg.norm(p - centre)) * 1000.0 for p in measured]

    def on_capture(frame, visible: bool) -> str:
        if not visible:
            return "board not visible — reposition"
        cam_T_board = det.detect_board(frame.rgb, frame.intrinsics)
        if cam_T_board is None:
            return "board not solved — reposition"
        try:
            base_T_gripper = arm.get_tcp_pose()
        except RuntimeError as e:
            return f"arm read failed: {e}"
        # The one measurement that matters: where the camera says the board is,
        # in the arm base frame, through the LIVE arm pose.
        p = (base_T_gripper @ X @ cam_T_board)[:3, 3]
        measured.append(p)
        if args.self_reference:
            _recompute_self_reference()
            e_mm = errs[-1] if errs else 0.0
        else:
            e_mm = float(np.linalg.norm(p - ref[:3, 3])) * 1000.0
            errs.append(e_mm)
        depth_mm = float(cam_T_board[2, 3]) * 1000.0
        a = np.array(errs)
        return (f"pose {len(errs)}: ERROR {e_mm:.1f} mm (board {depth_mm:.0f} mm from cam) | "
                f"mean {a.mean():.1f}, max {a.max():.1f} over {len(errs)}")

    def on_delete() -> str:
        if measured:
            measured.pop()
            errs.pop()
            _recompute_self_reference()
            return f"dropped last (now {len(errs)})"
        return "nothing to drop"

    def get_hud() -> list:
        if not errs:
            return ["measurements: 0 — move to a fresh pose & Capture"]
        a = np.array(errs)
        return [f"n={len(errs)}  mean={a.mean():.1f}mm  median={np.median(a):.1f}  "
                f"max={a.max():.1f}  min={a.min():.1f}"]

    try:
        input("\n⚠  SAFETY: the camera and its cable ride the arm. Support the arm, clear the "
              "workspace, then press Enter to enable drag-teach... ")
        arm.set_manual_mode(True)
        print("Drag-teach ON — guide the arm to fresh poses. Do panel-like distances, and take "
              "several samples at ONE pose too (that separates repeatability from accuracy).")
        handle.camera.start()
        try:
            if args.web:
                web_capture_loop(camera=handle.camera, board=board, on_capture=on_capture,
                                 on_delete=on_delete, get_hud=get_hud, port=args.web_port)
            else:
                live_capture_loop(camera=handle.camera, board=board, on_capture=on_capture,
                                  on_delete=on_delete, get_hud=get_hud,
                                  window=f"{args.camera} — eye-in-hand localization eval",
                                  force_headless=args.headless)
        finally:
            handle.camera.stop()
    finally:
        try:
            arm.set_manual_mode(False)
            print("Drag-teach OFF — servos re-locked.")
        except Exception:
            pass
        arm.disconnect()

    if errs:
        a = np.array(errs)
        mode = "spread about this session's mean (repeatability)" if args.self_reference \
            else "error against the calibration board position (absolute)"
        print(f"\n=== localization {mode} over {len(errs)} poses ===")
        print(f"  mean {a.mean():.1f} mm | median {np.median(a):.1f} mm | "
              f"max {a.max():.1f} mm | min {a.min():.1f} mm")
        print("  (this is the real camera->base-frame accuracy your press must tolerate; it now "
              "includes the arm's own FK error, unlike the fixed-camera number)")
    else:
        print("no measurements taken.")


if __name__ == "__main__":
    main()
