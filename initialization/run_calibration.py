#!/usr/bin/env python3
"""Calibration — PART 2: camera extrinsics (eye-to-hand) for the chest camera.

Run once, AFTER Part 1 (intrinsics). The chest Orbbec 335 is fixed to the torso;
a ChArUco board is attached rigidly to the RIGHT arm's end-effector. A live
preview shows the camera; hand-guide the arm (drag-teach) to varied poses and
press [c] to record a sample (the script grabs the frame, detects the board using
the shared Part 1 intrinsics, and reads the RealMan TCP pose). Each accepted
sample is written immediately to a gitignored dir so a crash loses nothing. After
>= 10 varied samples press [Esc] and it solves ``base_T_camera``.

Because the chest camera and the arm base ride the torso lift together, the result
``base_T_camera`` is a fixed constant — calibrate once, share at runtime.

Controls (live window):  [c] capture   [d] delete last   [Esc] finish + solve

Default board: 14x9 ChArUco, 20 mm squares, 15 mm markers, DICT_5X5_100.

Run this ON the robot (ssh dex4) with the RealMan SDK env, e.g.:
    ~/miniconda3/envs/richtech-v3/bin/python initialization/run_calibration.py --camera cam_chest
    # re-solve offline from saved samples (no camera/robot needed):
    python initialization/run_calibration.py --camera cam_chest \
        --from-samples data/calibration/cam_chest_samples

NOTE: make sure no other client (the richtech stack) is commanding the right arm
during capture, and do NOT change the arm's active tool frame mid-capture — the
eye-to-hand solve only needs the recorded pose to be *consistent* across samples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration.board import CharucoBoard
from calibration.capture_ui import live_capture_loop, web_capture_loop
from calibration.extrinsic import ExtrinsicCalibrationSession
from calibration.io import CALIB_DIR
from calibration.validate import save_report, validate_extrinsics
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm


def _validate(session, base_T_camera, out: str) -> None:
    """Validate the solved extrinsic and write a report next to it."""
    report = validate_extrinsics(
        base_T_camera, session.base_T_gripper, session.cam_T_target,
        all_methods=session.last_candidates,
    )
    print(report.summary())
    rpath = save_report(report, CALIB_DIR / f"{Path(out).stem}_extrinsic_validation.json")
    print(f"  report -> {rpath}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="cam_chest",
                        help="camera id from cameras.yaml (default: cam_chest)")
    parser.add_argument("--out", default=None,
                        help="output .npy under data/calibration/ (default: <camera>.npy)")
    parser.add_argument("--squares-x", type=int, default=14, help="ChArUco columns")
    parser.add_argument("--squares-y", type=int, default=9, help="ChArUco rows")
    parser.add_argument("--square-len", type=float, default=0.020, help="square side (m)")
    parser.add_argument("--marker-len", type=float, default=0.015, help="marker side (m)")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="cv2.aruco DICT_* name")
    parser.add_argument("--target-samples", type=int, default=15, help="suggested # of poses")
    parser.add_argument("--from-samples", metavar="DIR",
                        help="re-solve offline from previously saved samples (no camera/robot)")
    parser.add_argument("--headless", action="store_true", help="no preview window (prompt-based)")
    parser.add_argument("--web", action="store_true",
                        help="serve a browser preview (MJPEG) instead of a local window — for a "
                             "headless robot; open http://<robot-ip>:<web-port>/ on any machine on the LAN")
    parser.add_argument("--web-port", type=int, default=8010, help="port for the --web preview")
    parser.add_argument("--no-validate", action="store_true", help="skip post-calibration validation")
    parser.add_argument("--manual", action=argparse.BooleanOptionalAction, default=True,
                        help="enable the arm's drag-teach (zero-gravity) mode to hand-guide it "
                             "to each pose. Default ON; pass --no-manual to keep the servos locked.")
    args = parser.parse_args()

    # Default the output name to the camera id (cam_chest -> cam_chest.npy).
    if args.out is None:
        args.out = f"{args.camera}.npy"

    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )

    # Offline re-solve: rebuild the session from saved samples and solve. No
    # hardware needed — handy after a crash or to compare methods.
    if args.from_samples:
        session = ExtrinsicCalibrationSession.from_samples(args.from_samples, board=board)
        X = session.solve(save_as=args.out)
        if not args.no_validate:
            _validate(session, X, args.out)
        return

    manager = CameraManager().build()
    handle = manager.get(args.camera)
    if handle.intrinsics is None:
        print(
            "⚠ no shared intrinsics for this camera (Part 1). Run "
            "initialization/calibrate_intrinsics.py first; falling back to SDK intrinsics."
        )

    # Connect to the RealMan arm through the adapter (the single place that talks
    # to the SDK). Which arm/IP comes from configs/pipeline.yaml `arm:`.
    arm_cfg = load_pipeline().get("arm", {})
    arm = RealmanArm(
        side=arm_cfg.get("side", "right"),
        ip=arm_cfg.get("ip"),
        port=arm_cfg.get("port", 8080),
        max_speed_pct=arm_cfg.get("max_speed_pct", 20),
    )
    arm.connect()
    print(f"Connected to RealMan {arm.side} arm at {arm.ip}.")

    def read_pose():
        """base_T_gripper (4x4, meters) from the RealMan SDK."""
        return arm.get_tcp_pose()

    # Pre-flight: prove the controller answers BEFORE entering drag-teach.
    try:
        t = read_pose()[:3, 3] * 1000.0
        print(f"Arm read OK — TCP at x={t[0]:.1f} y={t[1]:.1f} z={t[2]:.1f} mm.")
    except RuntimeError as e:
        print(f"✗ Arm not responding right after connect — aborting: {e}.")
        arm.disconnect()
        return

    # Persist each accepted sample (gitignored dir) so the run can be re-solved /
    # audited offline; a crash mid-capture loses nothing.
    sample_dir = CALIB_DIR / f"{Path(args.out).stem}_samples"
    session = ExtrinsicCalibrationSession(board=board, sample_dir=sample_dir)

    def on_capture(frame, board_visible: bool) -> str:
        # Uses the shared Part 1 intrinsics (K + distortion) carried on the frame.
        if not board_visible:
            return "board not visible — reposition and retry"
        cam_T_target = session.detect_board(frame.rgb, frame.intrinsics)
        if cam_T_target is None:
            return "board not solved — reposition and retry"
        try:
            base_T_gripper = read_pose()
        except RuntimeError as e:
            # A transient read failure must NOT crash the whole capture — samples
            # so far are already persisted. Skip this one and continue.
            return f"arm read failed — sample skipped ({e})"
        if not session.is_diverse(base_T_gripper):
            return "pose too similar to the last one — move/rotate more"
        session.add_sample(base_T_gripper, cam_T_target, rgb=frame.rgb)
        return f"captured sample {session.num_samples} -> {session.sample_dir}"

    def on_delete() -> str:
        return f"deleted last sample (now {session.num_samples})" if session.pop_sample() \
            else "nothing to delete"

    def get_hud() -> list:
        return [f"samples {session.num_samples} / {args.target_samples}"]

    try:
        try:
            if args.manual:
                input("\n⚠  SAFETY: support the arm, clear the workspace, then press Enter "
                      "to enable drag-teach mode (the arm becomes hand-movable)... ")
                arm.set_manual_mode(True)
                print("Drag-teach ON — guide the arm by hand to each pose.")
            else:
                print("Arm servos are LOCKED (--no-manual). Use --manual to hand-guide the arm, "
                      "or jog it from another client between captures.")

            # Calibration is one camera at a time. Start ONLY the requested camera.
            handle.camera.start()
            try:
                print(f"Guide the arm to varied poses (aim for {args.target_samples}+).")
                if args.web:
                    web_capture_loop(
                        camera=handle.camera, board=board, on_capture=on_capture, on_delete=on_delete,
                        get_hud=get_hud, port=args.web_port,
                    )
                else:
                    live_capture_loop(
                        camera=handle.camera, board=board, on_capture=on_capture, on_delete=on_delete,
                        get_hud=get_hud, window=f"{args.camera} — extrinsics", force_headless=args.headless,
                    )
            finally:
                handle.camera.stop()
        finally:
            # Always leave the arm servo-locked again, never in freedrive.
            if args.manual:
                try:
                    arm.set_manual_mode(False)
                    print("Drag-teach OFF — arm servos re-locked.")
                except Exception as e:  # pragma: no cover - hardware dependent
                    print(f"⚠ could not leave drag-teach: {e}")

        if session.num_samples < 3:
            print(f"Only {session.num_samples} samples — need >= 3 (10+ recommended). "
                  f"Samples kept at {sample_dir}; re-run with --from-samples to resume.")
            return
        X = session.solve(save_as=args.out)
        if not args.no_validate:
            _validate(session, X, args.out)
    finally:
        # ALWAYS release the connection, even on a crash or early return.
        arm.disconnect()


if __name__ == "__main__":
    main()
