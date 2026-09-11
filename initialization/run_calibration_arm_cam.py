#!/usr/bin/env python3
"""Calibration — PART 2b: extrinsics for an ARM-MOUNTED camera (eye-IN-hand).

The counterpart to ``run_calibration.py``, which stays exactly as it is for the
fixed chest camera. Use THIS one when the camera is bolted to the arm.

**The setup is the mirror image of the eye-to-hand one, and getting it backwards
produces a clean-looking, completely wrong answer:**

    eye-to-hand (run_calibration.py)     eye-in-hand (this script)
    ---------------------------------    ---------------------------------
    board BOLTED TO THE FLANGE           board FIXED IN THE SCENE
    camera fixed on the torso            camera rides the arm
    solves base_T_camera (a constant)    solves gripper_T_camera (a constant)

So: clamp/tape the ChArUco board somewhere rigid where the arm-mounted camera can
see it from many angles — a wall, the panel cabinet, a stand. It must NOT move at
any point during capture, and nothing may move it between samples. Then
hand-guide the arm to varied poses and press [c] at each one.

Three requirements that are specific to this geometry:

1. **Rotate about more than one axis.** ``AX = XB`` recovers the camera's
   TRANSLATION only from the rotation between poses, so a set that is mostly
   translation — the natural thing to do when hand-guiding an arm — leaves it
   ill-conditioned, and a set whose rotations all share one axis leaves the
   translation along that axis unobservable. Neither announces itself in the
   solved matrix. The HUD shows a live ``axis`` ratio (0 = one axis only); keep
   capturing until it is comfortably above 0.15, and aim for relative rotations
   of 30 deg and more.
2. **Let the arm SETTLE before capturing.** The camera is now ON the arm, so a
   pose read while it is still micro-swinging corrupts the camera pose itself,
   not just the board's. The one bad sample in the 2026-08 chest calibration
   (34 mm, degrading the whole solve 3x) was exactly this, in the milder
   geometry. Let go, wait, then capture.
3. **Do not change the tool frame, ever again.** ``gripper_T_camera`` is
   expressed relative to whatever pose ``get_tcp_pose()`` reports, which on this
   robot already carries the controller's 130 mm tool z. Re-configuring the tool
   frame (or moving the plunger) after calibrating invalidates the extrinsic
   silently. The active tool frame is printed at startup — write it down.

Also unchanged from the fixed-camera procedure: run Part 1 (intrinsics) FIRST,
keep the torso lift still (it moves the base frame), and make sure nothing else
is commanding the arm.

Controls (live preview):  [c] capture   [d] delete last   [Esc] finish + solve

Run ON the robot with the system python3:
    python3 initialization/run_calibration_arm_cam.py --camera cam_arm --web
    # re-solve offline from saved samples (no camera/robot needed):
    python3 initialization/run_calibration_arm_cam.py --camera cam_arm \
        --from-samples data/calibration/cam_arm_samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration.board import CharucoBoard
from calibration.capture_ui import live_capture_loop, web_capture_loop
from calibration.eye_in_hand import EyeInHandSession, rotation_diversity, validate_eye_in_hand
from calibration.io import CALIB_DIR
from calibration.validate import save_report
from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.robot.realman import RealmanArm


def _validate(session, gripper_T_camera, out: str) -> None:
    """Validate the solved extrinsic and write a report next to it."""
    report = validate_eye_in_hand(
        gripper_T_camera, session.base_T_gripper, session.cam_T_target,
        all_methods=session.last_candidates,
    )
    print(report.summary())
    rpath = save_report(report, CALIB_DIR / f"{Path(out).stem}_extrinsic_validation.json")
    print(f"  report -> {rpath}")


def _print_tool_frame(arm) -> None:
    """Best-effort: print the controller's active tool frame, which the result is tied to.

    Not part of the solve — it is the thing that invalidates the solve later if
    anyone changes it, so it belongs in the operator's notes and in the log.
    """
    try:
        handle = arm._require()                      # the SDK object; read-only call
        ret = handle.rm_get_current_tool_frame()
        print(f"  active tool frame (record this!): {ret}")
    except Exception as e:  # pragma: no cover - SDK/firmware dependent
        print(f"  ⚠ could not read the active tool frame ({e}) — record it by hand. "
              f"gripper_T_camera is only valid while the tool frame stays as it is now.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default="cam_arm",
                        help="camera id from cameras.yaml (default: cam_arm)")
    parser.add_argument("--out", default=None,
                        help="output .npy under data/calibration/ (default: <camera>.npy)")
    parser.add_argument("--squares-x", type=int, default=14, help="ChArUco columns")
    parser.add_argument("--squares-y", type=int, default=9, help="ChArUco rows")
    parser.add_argument("--square-len", type=float, default=0.020, help="square side (m)")
    parser.add_argument("--marker-len", type=float, default=0.015, help="marker side (m)")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="cv2.aruco DICT_* name")
    parser.add_argument("--target-samples", type=int, default=20,
                        help="suggested # of poses (more than eye-to-hand: the arm's own "
                             "FK error is now inside the measurement)")
    parser.add_argument("--from-samples", metavar="DIR",
                        help="re-solve offline from previously saved samples (no camera/robot)")
    parser.add_argument("--headless", action="store_true", help="no preview window (prompt-based)")
    parser.add_argument("--web", action="store_true",
                        help="serve a browser preview (MJPEG) instead of a local window — for a "
                             "headless robot; open http://<robot-ip>:<web-port>/ on the LAN")
    parser.add_argument("--web-port", type=int, default=8010, help="port for the --web preview")
    parser.add_argument("--no-validate", action="store_true", help="skip post-calibration validation")
    parser.add_argument("--manual", action=argparse.BooleanOptionalAction, default=True,
                        help="enable the arm's drag-teach (zero-gravity) mode to hand-guide it "
                             "to each pose. Default ON; pass --no-manual to keep the servos locked.")
    args = parser.parse_args()

    if args.out is None:
        args.out = f"{args.camera}.npy"

    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )

    # Offline re-solve from saved samples — no hardware needed.
    if args.from_samples:
        session = EyeInHandSession.from_samples(args.from_samples, board=board)
        X = session.solve(save_as=args.out)
        if not args.no_validate:
            _validate(session, X, args.out)
        return

    manager = CameraManager().build()
    handle = manager.get(args.camera)
    if handle.mount != "arm":
        raise SystemExit(
            f"✗ {args.camera} is configured `mount: {handle.mount}` in configs/cameras.yaml. "
            f"This script solves the ARM-mounted geometry; set `mount: arm` for it, or use "
            f"initialization/run_calibration.py for a fixed camera."
        )
    if handle.intrinsics is None:
        print("⚠ no shared intrinsics for this camera (Part 1). Run "
              "initialization/calibrate_intrinsics.py first; falling back to SDK intrinsics.")

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
    _print_tool_frame(arm)

    sample_dir = CALIB_DIR / f"{Path(args.out).stem}_samples"
    session = EyeInHandSession(board=board, sample_dir=sample_dir)

    def on_capture(frame, board_visible: bool) -> str:
        if not board_visible:
            return "board not visible — move the ARM so the board comes into view"
        cam_T_target = session.detect_board(frame.rgb, frame.intrinsics)
        if cam_T_target is None:
            return "board not solved — reposition and retry"
        try:
            base_T_gripper = read_pose()
        except RuntimeError as e:
            # A transient read failure must NOT crash the capture — samples so far
            # are already persisted. Skip this one and continue.
            return f"arm read failed — sample skipped ({e})"
        if not session.is_diverse(base_T_gripper):
            return "pose too similar to the last one — move/rotate more"
        session.add_sample(base_T_gripper, cam_T_target, rgb=frame.rgb)
        div = rotation_diversity(session.base_T_gripper)
        return (f"captured sample {session.num_samples} | axis spread {div['axis_ratio']:.2f} "
                f"(want > 0.15), largest rotation {div['max_angle_deg']:.0f} deg")

    def on_delete() -> str:
        return f"deleted last sample (now {session.num_samples})" if session.pop_sample() \
            else "nothing to delete"

    def get_hud() -> list:
        # The conditioning number belongs ON SCREEN, not in the post-hoc report:
        # a degenerate pose set can only be fixed while the operator is still at
        # the robot with drag-teach on.
        if session.num_samples < 2:
            return [f"samples {session.num_samples} / {args.target_samples}"]
        div = rotation_diversity(session.base_T_gripper)
        flag = "OK" if div["axis_ratio"] >= 0.15 else "ROTATE ABOUT ANOTHER AXIS"
        return [f"samples {session.num_samples} / {args.target_samples}",
                f"axis spread {div['axis_ratio']:.2f} ({flag})  max rot {div['max_angle_deg']:.0f} deg"]

    try:
        try:
            if args.manual:
                input("\n⚠  SAFETY: the camera and its cable now ride the arm. Support the arm, "
                      "clear the workspace, check the cable has slack, then press Enter to enable "
                      "drag-teach (the arm becomes hand-movable)... ")
                arm.set_manual_mode(True)
                print("Drag-teach ON — guide the arm by hand to each pose.")
            else:
                print("Arm servos are LOCKED (--no-manual). Use --manual to hand-guide the arm, "
                      "or jog it from another client between captures.")

            handle.camera.start()
            try:
                print(f"Board must be FIXED in the scene and stay put. Guide the arm to varied "
                      f"poses (aim for {args.target_samples}+), rotating about DIFFERENT axes, "
                      f"and let it settle before each capture.")
                if args.web:
                    web_capture_loop(
                        camera=handle.camera, board=board, on_capture=on_capture,
                        on_delete=on_delete, get_hud=get_hud, port=args.web_port,
                    )
                else:
                    live_capture_loop(
                        camera=handle.camera, board=board, on_capture=on_capture,
                        on_delete=on_delete, get_hud=get_hud,
                        window=f"{args.camera} — eye-in-hand extrinsics",
                        force_headless=args.headless,
                    )
            finally:
                handle.camera.stop()
        finally:
            if args.manual:
                try:
                    arm.set_manual_mode(False)
                    print("Drag-teach OFF — arm servos re-locked.")
                except Exception as e:  # pragma: no cover - hardware dependent
                    print(f"⚠ could not leave drag-teach: {e}")

        if session.num_samples < 3:
            print(f"Only {session.num_samples} samples — need >= 3 (15+ recommended). "
                  f"Samples kept at {sample_dir}; re-run with --from-samples to resume.")
            return
        X = session.solve(save_as=args.out)
        if not args.no_validate:
            _validate(session, X, args.out)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
