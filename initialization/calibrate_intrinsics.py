#!/usr/bin/env python3
"""Calibration — PART 1: camera intrinsics (K + distortion).

Run once per camera. Saves ``data/calibration/<cam>_intrinsics.npz`` (shared by
the runtime and Part 2). A live preview shows the camera; each capture is written
immediately to ``data/calibration/<cam>_intrinsics_images/`` — which is gitignored
(``data/**``) — so a crash mid-capture loses nothing. The result is validated
right after (reprojection residuals, per the lensboy guide).

Controls (live window):  [c] capture   [d] delete last   [Esc] finish

Default board: 14x9 ChArUco, 20 mm squares, 15 mm markers, DICT_5X5_100.

Two modes:
  * default — interactive ChArUco calibration -> cv2.calibrateCamera.
  * --from-sdk — just store the Orbbec factory intrinsics (fast; no validation,
    since reprojection validation needs board images).

Usage:
    python initialization/calibrate_intrinsics.py --camera cam_right
    python initialization/calibrate_intrinsics.py --camera cam_right --from-sdk
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from calibration.board import CharucoBoard
from calibration.capture_ui import live_capture_loop, web_capture_loop
from calibration.intrinsic import (
    calibrate_intrinsics_from_images,
    intrinsics_from_camera,
)
from calibration.io import CALIB_DIR
from calibration.validate import save_report, validate_intrinsics
from core.camera.orbbec import OrbbecCamera
from core.config import load_cameras


def _camera_cfg(camera_id: str) -> dict:
    for cam in load_cameras().get("cameras", []):
        if cam["id"] == camera_id:
            return cam
    raise SystemExit(f"camera {camera_id!r} not found in configs/cameras.yaml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="camera id from cameras.yaml")
    parser.add_argument("--out", help="output .npz (default: <camera>_intrinsics.npz)")
    parser.add_argument("--from-sdk", action="store_true",
                        help="store the SDK factory intrinsics instead of a ChArUco calibration")
    parser.add_argument("--squares-x", type=int, default=14, help="ChArUco columns")
    parser.add_argument("--squares-y", type=int, default=9, help="ChArUco rows")
    parser.add_argument("--square-len", type=float, default=0.020, help="square side (m)")
    parser.add_argument("--marker-len", type=float, default=0.015, help="marker side (m)")
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="cv2.aruco DICT_* name")
    parser.add_argument("--target-images", type=int, default=20, help="suggested # of views")
    parser.add_argument("--headless", action="store_true", help="no preview window (prompt-based)")
    parser.add_argument("--web", action="store_true",
                        help="serve a browser preview (MJPEG) instead of a local window — for a "
                             "headless robot; open http://<robot-ip>:<web-port>/ on any machine on the LAN")
    parser.add_argument("--web-port", type=int, default=8010, help="port for the --web preview")
    parser.add_argument("--no-validate", action="store_true", help="skip post-calibration validation")
    parser.add_argument("--resume", action="store_true",
                        help="load the views already in the capture dir and keep shooting from "
                             "there, instead of wiping them. Use this to add coverage to an "
                             "existing set, or to recover a session whose process died.")
    args = parser.parse_args()

    out = args.out or f"{args.camera}_intrinsics.npz"
    cfg = _camera_cfg(args.camera)
    # Build the camera directly (no intrinsics override) so Part 1 never depends
    # on a pre-existing intrinsics file.
    camera = OrbbecCamera(
        camera_id=args.camera, serial=cfg.get("serial"), match_name=cfg.get("match_name"),
        width=cfg.get("width", 1280), height=cfg.get("height", 720), fps=cfg.get("fps", 30),
    )
    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )
    # Gitignored (data/**). Captured views land here as they are taken.
    images_dir = CALIB_DIR / f"{args.camera}_intrinsics_images"

    frames: list = []        # in-memory RGB views
    saved: list = []         # parallel saved file paths (for 'delete last')

    def on_capture(frame, board_visible: bool) -> str:
        if not board_visible:
            return "no board — reposition and try again"
        images_dir.mkdir(parents=True, exist_ok=True)
        p = images_dir / f"img{len(frames):03d}.png"
        cv2.imwrite(str(p), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
        frames.append(frame.rgb.copy())
        saved.append(p)
        return f"captured view {len(frames)} -> {p.name}"

    def on_delete() -> str:
        if not frames:
            return "nothing to delete"
        frames.pop()
        p = saved.pop()
        try:
            p.unlink()
        except OSError:
            pass
        return f"deleted last view (now {len(frames)})"

    def get_hud() -> list:
        return [f"captured {len(frames)} / {args.target_images}",
                f"board {args.squares_x}x{args.squares_y} {args.dictionary}"]

    with camera:
        if args.from_sdk:
            calib = intrinsics_from_camera(camera)
            print("Using Orbbec factory intrinsics.")
        else:
            if args.resume and images_dir.exists():
                # Re-load previous views so more can be added to them. Disk holds
                # BGR (see on_capture); frames[] is RGB. New views keep numbering
                # from len(frames), so nothing is overwritten.
                for p in sorted(images_dir.glob("img*.png")):
                    bgr = cv2.imread(str(p))
                    if bgr is None:
                        continue
                    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    saved.append(p)
                print(f"resumed {len(frames)} existing view(s) from {images_dir}")
            elif images_dir.exists():
                # Start fresh so a re-run doesn't mix with stale images.
                for old in images_dir.glob("img*.png"):
                    old.unlink()
            print("Move the board to cover the frame (corners + tilts; mostly angled "
                  f"close-ups), aim for {args.target_images}+ views.")
            if args.web:
                web_capture_loop(
                    camera=camera, board=board, on_capture=on_capture, on_delete=on_delete,
                    get_hud=get_hud, port=args.web_port,
                )
            else:
                live_capture_loop(
                    camera=camera, board=board, on_capture=on_capture, on_delete=on_delete,
                    get_hud=get_hud, window=f"{args.camera} — intrinsics", force_headless=args.headless,
                )
            if len(frames) < 5:
                print(f"Only {len(frames)} views — need >= 5. Aborting (nothing saved).")
                return
            calib = calibrate_intrinsics_from_images(frames, board=board)
            print(f"Calibrated from {calib.num_images} views, RMS reprojection = {calib.rms:.4f} px")

    path = calib.save(out)
    print(f"  K =\n{calib.K}")
    print(f"  dist = {calib.dist}")
    print(f"✓ saved intrinsics -> {path}")
    if frames:
        print(f"  views saved (gitignored) -> {images_dir}")

    if not args.no_validate and frames:
        print("\nValidating intrinsics...")
        report = validate_intrinsics(
            calib.K, calib.dist, frames, board=board,
            image_size=(calib.width, calib.height),
            plot_dir=CALIB_DIR / f"{args.camera}_intrinsics_validation",
        )
        print(report.summary())
        rpath = save_report(report, CALIB_DIR / f"{args.camera}_intrinsics_validation/report.json")
        print(f"  report -> {rpath}")
    elif args.from_sdk:
        print("\n(skip validation: --from-sdk has no board images; capture views or run "
              "initialization/validate_calibration.py with saved images.)")


if __name__ == "__main__":
    main()
