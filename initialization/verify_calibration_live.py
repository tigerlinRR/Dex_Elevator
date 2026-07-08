#!/usr/bin/env python3
"""Live, hardware-in-the-loop accuracy check for a finished extrinsic calibration.

The internal validation (consistency / hold-out / cross-method in
``validate_calibration.py``) only proves the AX=XB solve is self-consistent — it
cannot catch a wrong board size, a unit slip, or a base-frame mistake. This
script gives the real, physical number: **how far the camera's prediction of a
point in the robot base frame is from where the robot actually is.**

Method (no autonomous motion — the arm only ever reports its pose):
  1. The ChArUco board is rigid to the gripper. Recover the fixed mount
     ``flange_T_board`` from the saved calibration samples (its spread across
     samples = how rigid the mount is).
  2. With the arm parked at ANY pose, read the true flange pose from the
     controller (``get_position``, TCP zeroed to match calibration) and detect
     the board in a fresh camera frame -> ``cam_T_board``.
  3. Predict the flange pose purely from vision:
         base_T_flange_pred = base_T_camera @ cam_T_board @ inv(flange_T_board)
     The translation gap vs the reported flange pose is the calibration's
     absolute positioning error at that pose.

For a TRUE accuracy figure the arm must sit at a pose the calibration never saw
(otherwise step 1 makes it circular) — the script warns if the current pose
coincides with a calibration sample. Run it at several fresh poses; each result
is appended to ``<cam>_live_check.json`` so the poses accumulate.

Usage:
    python initialization/verify_calibration_live.py --camera cam_right            # single shot
    python initialization/verify_calibration_live.py --camera cam_right --live     # real-time readout
    python initialization/verify_calibration_live.py --camera cam_right --tol-mm 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.capture_ui import display_available
from calibration.extrinsic import ExtrinsicCalibrationSession
from calibration.io import CALIB_DIR, load_extrinsic
from core.camera.manager import CameraManager
from core.config import load_cameras, load_pipeline
from core.transforms import invert, pose_from_rpy, rot_angle_deg

try:
    from xarm.wrapper import XArmAPI
except ImportError as _e:  # SDK is an optional runtime dep
    raise SystemExit(
        "xArm-Python-SDK is required for the live check. "
        "Install it with: pip install xArm-Python-SDK"
    ) from _e


def _camera_cfg(camera_id: str) -> dict:
    for cam in load_cameras().get("cameras", []):
        if cam["id"] == camera_id:
            return cam
    return {}


def _mean_se3(transforms: list[np.ndarray]) -> np.ndarray:
    """Average a set of 4x4 rigid transforms (translation mean + SVD rotation mean)."""
    t = np.mean([T[:3, 3] for T in transforms], axis=0)
    R_sum = np.sum([T[:3, :3] for T in transforms], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection -> nearest proper rotation
        U[:, -1] *= -1
        R = U @ Vt
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def _connect_arm(host: str, read_timeout: float) -> XArmAPI:
    """Open the single master control connection, retrying while it's contended."""
    last_err = None
    for _ in range(60):
        try:
            arm = XArmAPI(host, is_radian=False)
            break
        except Exception as e:  # SDK raises if the control socket can't open
            last_err = e
            time.sleep(0.5)
    else:
        raise SystemExit(
            f"Could not open a control connection to the xArm at {host} ({last_err}). "
            "It allows only one master client — close xArm Studio and retry."
        )
    arm.clean_warn()
    arm.clean_error()
    arm.motion_enable(enable=True)
    arm.set_mode(0)   # position control (we never command motion, only read)
    arm.set_state(0)
    arm.set_timeout(read_timeout)
    # Match calibration: zero the TCP so get_position reports the FLANGE pose.
    arm.set_tcp_offset([0, 0, 0, 0, 0, 0], is_radian=False)
    arm.set_state(0)
    return arm


def _read_flange_pose(arm: XArmAPI) -> np.ndarray:
    code, pose = arm.get_position(is_radian=False)
    if code != 0 or pose is None:
        raise SystemExit(
            f"get_position returned code={code} (3 = reply timeout: controller not "
            "servicing this session — close xArm Studio / power-cycle the box)"
        )
    x, y, z, roll, pitch, yaw = pose  # mm, deg
    return pose_from_rpy(x / 1000.0, y / 1000.0, z / 1000.0, roll, pitch, yaw, degrees=True)


def _error(base_T_cam, cam_T_board, flange_T_board, g_live):
    """Vision-predicted flange pose vs the reported one -> (pred, dxyz_mm, |err|mm, err_deg)."""
    pred = base_T_cam @ cam_T_board @ invert(flange_T_board)
    d_trans = (pred[:3, 3] - g_live[:3, 3]) * 1000.0
    return pred, d_trans, float(np.linalg.norm(d_trans)), rot_angle_deg(invert(g_live) @ pred)


def _nearest_calib_mm(base_T_gripper: list[np.ndarray], g_live: np.ndarray) -> float:
    return min(np.linalg.norm((bg[:3, 3] - g_live[:3, 3])) * 1000.0 for bg in base_T_gripper)


def _run_live(session, board, base_T_cam, flange_T_board, arm, handle, args) -> int:
    """Continuously detect the board + read the flange, printing the error in real time.

    Shows an OpenCV window (corner overlay + error HUD) when a display is
    available; otherwise refreshes a single terminal line. Move the board around
    and watch the error update; press Esc/q (window) or Ctrl-C (terminal) to quit.
    """
    overlay_det = BoardDetector(board=board, accurate=False)  # fast, for the dot overlay
    use_gui = display_available()
    win = f"{args.camera} — live calib check"
    period = 1.0 / max(args.hz, 0.5)
    handle.camera.start()
    if use_gui:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    else:
        print("(no display — refreshing terminal line) Ctrl-C to stop")
    try:
        while True:
            frame = handle.camera.capture()
            g_live = _read_flange_pose(arm)
            nearest = _nearest_calib_mm(session.base_T_gripper, g_live)
            cam_T_board = session.detect_board(frame.rgb, frame.intrinsics)
            ok = cam_T_board is not None
            if ok:
                _, d_trans, err_mm, err_deg = _error(base_T_cam, cam_T_board, flange_T_board, g_live)
            indep = nearest >= 2.0

            if use_gui:
                vis = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                found = overlay_det.detect_corners(frame.rgb)
                if found is not None:
                    for c in found[0].reshape(-1, 2):
                        cv2.circle(vis, (int(round(c[0])), int(round(c[1]))), 5, (0, 200, 0), -1)
                if ok:
                    color = (0, 200, 0) if (err_mm <= args.tol_mm and indep) else (0, 165, 255)
                    cv2.putText(vis, f"POS ERROR {err_mm:5.2f} mm", (15, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
                    cv2.putText(vis, f"ROT ERROR {err_deg:5.2f} deg   dxyz {np.round(d_trans,1)} mm",
                                (15, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(vis, "no board", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
                tag = f"nearest calib pose {nearest:5.1f} mm " + ("(independent)" if indep else "(CIRCULAR)")
                cv2.putText(vis, tag, (15, frame.rgb.shape[0] - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(win, vis)
                if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if (cv2.waitKey(max(1, int(period * 1000))) & 0xFF) in (27, ord("q")):
                    break
            else:
                flag = "indep " if indep else "CIRCUL"
                line = (f"  POS {err_mm:6.2f} mm | ROT {err_deg:5.2f} deg | nearest {nearest:6.1f} mm {flag}"
                        if ok else f"  no board                              nearest {nearest:6.1f} mm {flag}")
                print("\r" + line + "   ", end="", flush=True)
                time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        if use_gui:
            cv2.destroyWindow(win)
        else:
            print()
        handle.camera.stop()
        arm.disconnect()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="cam_right", help="camera id from cameras.yaml")
    parser.add_argument("--extrinsic", help="extrinsic .npy (default: from cameras.yaml or <cam>.npy)")
    parser.add_argument("--samples", help="extrinsic samples dir (default: <cam>_samples/)")
    parser.add_argument("--squares-x", type=int, default=14)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-len", type=float, default=0.020)
    parser.add_argument("--marker-len", type=float, default=0.015)
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="cv2.aruco DICT_* name")
    parser.add_argument("--read-timeout", type=float, default=5.0, help="xArm reply timeout (s)")
    parser.add_argument("--tol-mm", type=float, default=5.0,
                        help="position error at/under this passes (default 5 mm)")
    parser.add_argument("--live", action="store_true",
                        help="continuously detect + print the error in real time "
                             "(OpenCV window if a display is available, else a terminal line)")
    parser.add_argument("--hz", type=float, default=5.0, help="live refresh rate (default 5 Hz)")
    args = parser.parse_args()

    cfg = _camera_cfg(args.camera)
    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )

    extr_file = args.extrinsic or cfg.get("extrinsic_file") or f"{args.camera}.npy"
    extr_path = Path(extr_file)
    if not extr_path.is_absolute():
        extr_path = CALIB_DIR / extr_path
    samples_dir = Path(args.samples) if args.samples else CALIB_DIR / f"{extr_path.stem}_samples"
    if not extr_path.exists() or not samples_dir.exists():
        print(f"✗ need {extr_path} and {samples_dir} — run the calibration first.")
        return 2

    base_T_cam = load_extrinsic(extr_path)

    # 1) Recover the fixed flange->board mount from the calibration samples.
    session = ExtrinsicCalibrationSession.from_samples(samples_dir, board=board)
    fTb = [invert(bg) @ base_T_cam @ ct
           for bg, ct in zip(session.base_T_gripper, session.cam_T_target)]
    flange_T_board = _mean_se3(fTb)
    t_std = np.std([T[:3, 3] for T in fTb], axis=0) * 1000.0
    r_std = float(np.std([rot_angle_deg(invert(flange_T_board) @ T) for T in fTb]))
    print(f"flange_T_board from {len(fTb)} samples — mount rigidity: "
          f"translation std {np.round(t_std, 2)} mm, rotation std {r_std:.3f} deg")

    host = load_pipeline().get("robot", {}).get("host", "192.168.10.101")
    manager = CameraManager().build()
    handle = manager.get(args.camera)

    # Live mode: hold the arm connection open and stream the error continuously.
    if args.live:
        arm = _connect_arm(host, args.read_timeout)
        return _run_live(session, board, base_T_cam, flange_T_board, arm, handle, args)

    # 2) Read the true flange pose (arm parked) and detect the board (single shot).
    arm = _connect_arm(host, args.read_timeout)
    try:
        g_live = _read_flange_pose(arm)
    finally:
        arm.disconnect()

    # Warn if this pose is (near) a calibration pose — then the check is circular.
    nearest_mm = _nearest_calib_mm(session.base_T_gripper, g_live)

    handle.camera.start()
    try:
        for _ in range(5):  # drop warm-up frames
            frame = handle.camera.capture()
    finally:
        handle.camera.stop()
    cam_T_board = session.detect_board(frame.rgb, frame.intrinsics)
    if cam_T_board is None:
        print("✗ board not detected in the live frame — reposition so it's fully visible.")
        return 2

    # 3) Vision-predicted flange pose vs the controller-reported one.
    pred, d_trans, err_mm, err_deg = _error(base_T_cam, cam_T_board, flange_T_board, g_live)

    print("\n--- LIVE CHECK (arm stationary, no motion commanded) ---")
    print(f"  robot-reported flange (mm):  {np.round(g_live[:3, 3] * 1000.0, 1)}")
    print(f"  vision-predicted flange (mm):{np.round(pred[:3, 3] * 1000.0, 1)}")
    print(f"  >> POSITION ERROR: {np.round(d_trans, 2)} mm   |error| = {err_mm:.2f} mm")
    print(f"  >> ORIENTATION ERROR: {err_deg:.3f} deg")
    if nearest_mm < 2.0:
        print(f"  ⚠ this pose is {nearest_mm:.1f} mm from a CALIBRATION pose — result is "
              "NOT independent (circular). Move to a fresh pose and re-run.")
    independent = nearest_mm >= 2.0
    passed = err_mm <= args.tol_mm and independent
    print(f"  verdict: {'PASS' if passed else 'FAIL' if independent else 'INCONCLUSIVE'} "
          f"(tol {args.tol_mm:.1f} mm; nearest calib pose {nearest_mm:.1f} mm away)")

    # Append this pose's result so repeated runs accumulate a record.
    log_path = CALIB_DIR / f"{extr_path.stem}_live_check.json"
    record = {
        "flange_reported_mm": [round(float(v), 2) for v in g_live[:3, 3] * 1000.0],
        "flange_predicted_mm": [round(float(v), 2) for v in pred[:3, 3] * 1000.0],
        "position_error_mm": round(err_mm, 3),
        "orientation_error_deg": round(float(err_deg), 4),
        "nearest_calib_pose_mm": round(float(nearest_mm), 2),
        "independent": bool(independent),
    }
    history = []
    if log_path.exists():
        history = json.loads(log_path.read_text())
    history.append(record)
    log_path.write_text(json.dumps(history, indent=2))
    print(f"  appended to {log_path} ({len(history)} pose(s) recorded)")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
