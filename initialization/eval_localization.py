#!/usr/bin/env python3
"""End-to-end localization accuracy check.

Answers the real question for pressing: when the chest camera sees something and
we compute its 3D position in the RIGHT-arm base frame, how far off is it?

The ChArUco board stays rigidly on the flange (as in extrinsic calibration). We
recover the fixed ``gripper_T_board`` from the calibration samples, then at each
FRESH drag-teach pose compare two independent estimates of the board's base-frame
position:

    P_cam = (base_T_camera @ cam_T_board)[:3, 3]      # camera says the board is here
    P_arm = (base_T_gripper @ gripper_T_board)[:3, 3]  # the arm (FK) says it is here

Their translation difference is the "camera -> base-frame point" error a button
press depends on. No fingertip/probe TCP needed; no autonomous motion (drag-teach).
Move to poses at a PANEL-LIKE distance/position to test the operating region.

Run on the robot (richtech-v3 env):
    python initialization/eval_localization.py --camera cam_chest --web
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from calibration.board import CharucoBoard
from calibration.capture_ui import live_capture_loop, web_capture_loop
from calibration.extrinsic import ExtrinsicCalibrationSession
from calibration.io import CALIB_DIR, load_extrinsic
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm
from core.transforms import invert, make_transform, matrix_to_quat, quat_to_matrix


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
    ap.add_argument("--camera", default="cam_chest")
    ap.add_argument("--samples-dir", default=None,
                    help="calibration samples dir (default data/calibration/<camera>_samples)")
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

    X = load_extrinsic(f"{args.camera}.npy")     # base_T_camera
    sdir = args.samples_dir or str(CALIB_DIR / f"{args.camera}_samples")
    cal = ExtrinsicCalibrationSession.from_samples(sdir, board=board)
    grips = [invert(bg) @ X @ ct for bg, ct in zip(cal.base_T_gripper, cal.cam_T_target)]
    gripper_T_board = _mean_transform(grips)
    gt = np.array([g[:3, 3] for g in grips]) * 1000.0
    print(f"gripper_T_board from {len(grips)} calib samples; translation spread "
          f"(mm): {np.array2string((gt.max(0) - gt.min(0)), precision=1)} "
          "(small = board didn't move on the flange since calibration)")

    manager = CameraManager().build()
    handle = manager.get(args.camera)
    if handle.extrinsic is None or handle.intrinsics is None:
        raise SystemExit(f"{args.camera} missing intrinsics/extrinsic — calibrate first.")

    arm_cfg = load_pipeline().get("arm", {})
    arm = RealmanArm(side=arm_cfg.get("side", "right"), ip=arm_cfg.get("ip"),
                     port=arm_cfg.get("port", 8080), max_speed_pct=arm_cfg.get("max_speed_pct", 20))
    arm.connect()
    print(f"Connected to RealMan {arm.side} arm at {arm.ip}.")

    det = ExtrinsicCalibrationSession(board=board)   # for detect_board (accurate)
    errs: list[float] = []

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
        p_cam = (X @ cam_T_board)[:3, 3]
        p_arm = (base_T_gripper @ gripper_T_board)[:3, 3]
        e_mm = float(np.linalg.norm(p_cam - p_arm)) * 1000.0
        depth_mm = float(cam_T_board[2, 3]) * 1000.0
        errs.append(e_mm)
        a = np.array(errs)
        return (f"pose {len(errs)}: ERROR {e_mm:.1f} mm (board {depth_mm:.0f} mm from cam) | "
                f"mean {a.mean():.1f}, max {a.max():.1f} over {len(errs)}")

    def on_delete() -> str:
        if errs:
            errs.pop()
            return f"dropped last (now {len(errs)})"
        return "nothing to drop"

    def get_hud() -> list:
        if not errs:
            return ["measurements: 0 — move to a fresh pose & Capture"]
        a = np.array(errs)
        return [f"n={len(errs)}  mean={a.mean():.1f}mm  median={np.median(a):.1f}  "
                f"max={a.max():.1f}  min={a.min():.1f}"]

    try:
        input("\n⚠  SAFETY: support the arm, clear the workspace, then press Enter to enable "
              "drag-teach mode... ")
        arm.set_manual_mode(True)
        print("Drag-teach ON — guide the arm to fresh poses (aim for panel-like distance/positions).")
        handle.camera.start()
        try:
            if args.web:
                web_capture_loop(camera=handle.camera, board=board, on_capture=on_capture,
                                 on_delete=on_delete, get_hud=get_hud, port=args.web_port)
            else:
                live_capture_loop(camera=handle.camera, board=board, on_capture=on_capture,
                                  on_delete=on_delete, get_hud=get_hud,
                                  window=f"{args.camera} — localization eval",
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
        print(f"\n=== localization error over {len(errs)} fresh poses ===")
        print(f"  mean {a.mean():.1f} mm | median {np.median(a):.1f} mm | "
              f"max {a.max():.1f} mm | min {a.min():.1f} mm")
        print("  (this is the real camera->base-frame accuracy your press must tolerate)")
    else:
        print("no measurements taken.")


if __name__ == "__main__":
    main()
