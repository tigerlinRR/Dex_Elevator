#!/usr/bin/env python3
"""One-command readiness check for the robot, to run BEFORE any calibration.

Calibration needs four things to be true at once: the geometry libs are usable,
the configs point at the hardware actually plugged in, the chest camera delivers
frames, and the pressing arm answers. Each of those fails in its own way and
some (a wrong serial, a colour profile that never yields a frame) only surface
minutes into a capture session. This checks all of them in ~10 seconds.

It only ever READS: the arm is asked for its pose, never commanded to move.

Usage:
    python initialization/bringup_check.py                    # chest camera + right arm
    python initialization/bringup_check.py --camera cam_head --arm-ip 192.168.11.32
"""
from __future__ import annotations

import argparse
import sys

REPO_HINT = "run from the repo root, or `pip install --user -e .` so the packages resolve"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="cam_chest", help="camera id in configs/cameras.yaml")
    ap.add_argument("--arm-ip", default=None, help="override the arm IP from configs/pipeline.yaml")
    ap.add_argument("--skip-arm", action="store_true", help="camera-only check")
    args = ap.parse_args()

    failures: list[str] = []

    def check(label, fn):
        try:
            print(f"  [OK] {label:<24} {fn()}")
        except Exception as exc:  # noqa: BLE001 - report every failure, never abort early
            failures.append(label)
            print(f"  [!!] {label:<24} {type(exc).__name__}: {exc}")

    print("=== Dex_Elevator bring-up check ===")

    def _libs():
        import cv2
        import numpy
        if not hasattr(cv2.aruco, "CharucoDetector"):
            raise RuntimeError("cv2.aruco.CharucoDetector missing — need an opencv-contrib build")
        if not hasattr(cv2, "calibrateHandEye"):
            raise RuntimeError("cv2.calibrateHandEye missing")
        return f"numpy {numpy.__version__}, cv2 {cv2.__version__} (aruco + handeye OK)"

    check("numpy / cv2", _libs)

    def _config():
        from core.config import load_cameras, load_pipeline
        cams = load_cameras()["cameras"]
        entry = next((c for c in cams if c["id"] == args.camera), None)
        if entry is None:
            raise RuntimeError(f"{args.camera!r} not in configs/cameras.yaml")
        arm_ip = args.arm_ip or load_pipeline()["arm"]["ip"]
        return f"{args.camera} serial={entry.get('serial')}, arm ip={arm_ip}"

    check("configs", _config)

    def _board():
        from calibration.board import CharucoBoard
        b = CharucoBoard()
        return f"{b.squares_x}x{b.squares_y}, square {b.square_length * 1000:.0f} mm"

    check("ChArUco board", _board)

    def _camera():
        from core.camera.manager import CameraManager
        handle = CameraManager().build().get(args.camera)
        cam = handle.camera
        cam.start()
        try:
            frame = cam.capture()
        finally:
            cam.stop()
        depth_pct = 100.0 * (frame.depth > 0).mean() if frame.depth is not None else 0.0
        k = frame.intrinsics
        return (f"RGB {frame.rgb.shape}, depth valid {depth_pct:.0f}%, "
                f"fx={k.fx:.1f} cx={k.cx:.1f} cy={k.cy:.1f}")

    check(f"camera {args.camera}", _camera)

    if not args.skip_arm:
        def _arm():
            from core.config import load_pipeline
            from core.robot.realman import RealmanArm
            ip = args.arm_ip or load_pipeline()["arm"]["ip"]
            arm = RealmanArm(ip=ip)
            arm.connect()
            try:
                pose = arm.get_tcp_pose()
            finally:
                arm.disconnect()
            t = pose[:3, 3]
            return f"{ip} TCP xyz = [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m"

        check("arm (read-only)", _arm)

    print()
    if failures:
        print(f"=== {len(failures)} check(s) FAILED: {', '.join(failures)} ===")
        print(f"(if imports failed: {REPO_HINT})")
        return 1
    print("=== all checks passed — ready to calibrate ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
