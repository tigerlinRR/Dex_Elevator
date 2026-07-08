#!/usr/bin/env python3
"""Validate a finished calibration — standalone, no hardware needed.

Re-runs the validation checks (reprojection residuals for intrinsics; consistency
/ hold-out / cross-method for extrinsics) against the artifacts saved under
``data/calibration/`` by the calibration scripts. Use it any time to re-check a
calibration, or after editing thresholds.

Usage:
    python initialization/validate_calibration.py --camera cam_right
    python initialization/validate_calibration.py --camera cam_right --no-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from calibration.board import CharucoBoard
from calibration.extrinsic import ExtrinsicCalibrationSession, solve_eye_to_hand_all_methods
from calibration.io import CALIB_DIR, load_camera_intrinsics, load_extrinsic
from calibration.validate import save_report, validate_extrinsics, validate_intrinsics
from core.config import load_cameras


def _camera_cfg(camera_id: str) -> dict:
    for cam in load_cameras().get("cameras", []):
        if cam["id"] == camera_id:
            return cam
    return {}


def _load_images(images_dir: Path) -> list[np.ndarray]:
    files = sorted([p for p in images_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    imgs = []
    for p in files:
        bgr = cv2.imread(str(p))
        if bgr is not None:
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return imgs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", required=True, help="camera id from cameras.yaml")
    parser.add_argument("--intrinsics", help="intrinsics .npz (default: <cam>_intrinsics.npz)")
    parser.add_argument("--images", help="intrinsics images dir (default: <cam>_intrinsics_images/)")
    parser.add_argument("--extrinsic", help="extrinsic .npy (default: from cameras.yaml or <cam>.npy)")
    parser.add_argument("--samples", help="extrinsic samples dir (default: <cam>_samples/)")
    parser.add_argument("--squares-x", type=int, default=14)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-len", type=float, default=0.020)
    parser.add_argument("--marker-len", type=float, default=0.015)
    parser.add_argument("--dictionary", default="DICT_5X5_100", help="cv2.aruco DICT_* name")
    parser.add_argument("--no-plots", action="store_true", help="skip matplotlib plots")
    args = parser.parse_args()

    cfg = _camera_cfg(args.camera)
    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )
    reports = []

    # --- Part 1: intrinsics ---
    intr_file = args.intrinsics or cfg.get("intrinsics_file") or f"{args.camera}_intrinsics.npz"
    images_dir = Path(args.images) if args.images else CALIB_DIR / f"{args.camera}_intrinsics_images"
    intr_path = Path(intr_file)
    if not intr_path.is_absolute():
        intr_path = CALIB_DIR / intr_path
    if intr_path.exists() and images_dir.exists():
        intr = load_camera_intrinsics(intr_path)
        images = _load_images(images_dir)
        if images:
            print(f"Validating intrinsics ({len(images)} views from {images_dir})...")
            plot_dir = None if args.no_plots else CALIB_DIR / f"{args.camera}_intrinsics_validation"
            rep = validate_intrinsics(
                intr.K, intr.D, images, board=board,
                image_size=(intr.width, intr.height), plot_dir=plot_dir,
            )
            print(rep.summary())
            save_report(rep, CALIB_DIR / f"{args.camera}_intrinsics_validation/report.json")
            reports.append(rep)
        else:
            print(f"⚠ no images found in {images_dir}; skipping intrinsics validation")
    else:
        print(f"⚠ intrinsics validation skipped (need {intr_path} and {images_dir})")

    # --- Part 2: extrinsics ---
    extr_file = args.extrinsic or cfg.get("extrinsic_file") or f"{args.camera}.npy"
    extr_path = Path(extr_file)
    if not extr_path.is_absolute():
        extr_path = CALIB_DIR / extr_path
    samples_dir = Path(args.samples) if args.samples else CALIB_DIR / f"{extr_path.stem}_samples"
    if extr_path.exists() and samples_dir.exists():
        base_T_camera = load_extrinsic(extr_path)
        session = ExtrinsicCalibrationSession.from_samples(samples_dir, board=board)
        all_methods = solve_eye_to_hand_all_methods(session.base_T_gripper, session.cam_T_target)
        print(f"\nValidating extrinsics ({session.num_samples} samples from {samples_dir})...")
        rep = validate_extrinsics(
            base_T_camera, session.base_T_gripper, session.cam_T_target, all_methods=all_methods,
        )
        print(rep.summary())
        save_report(rep, CALIB_DIR / f"{extr_path.stem}_extrinsic_validation.json")
        reports.append(rep)
    else:
        print(f"⚠ extrinsics validation skipped (need {extr_path} and {samples_dir})")

    if not reports:
        print("\nNothing to validate — run the calibration scripts first.")
        return 2
    # Non-zero exit if anything failed (useful for CI / scripted checks).
    return 1 if any(r.status == "fail" for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
