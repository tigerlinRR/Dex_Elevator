#!/usr/bin/env python3
"""Dump live button 3D coordinates (base frame) to JSON — perception only, NO arm.

Runs the SAME perception the press pipeline uses (detector -> panel layout -> live
plane fit -> ray-plane), but talks to nothing but the chest camera: it never
connects to or moves the arm. Output feeds boundary_sweep.py --buttons-json so the
reachability / lift table is computed against the REAL panel instead of a nominal one.

    python3 initialization/dump_button_coords.py                 # -> button_coords.json
    python3 initialization/dump_button_coords.py --out /tmp/b.json --circles

Mirrors initialization/press_buttons.py's "locate the panel and every button once"
block (detector path, with a Hough-circle fallback), so the coordinates match what
a real press would aim at. Prints the fitted plane too, for the record.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

REPO = __file__.rsplit("/", 2)[0]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from core.camera.manager import CameraManager  # noqa: E402
from core.config import load_pipeline  # noqa: E402
from core.press import fit_panel_plane_from_depth, ray_plane_intersection  # noqa: E402
from yolo.button_circles import detect_buttons  # noqa: E402
from yolo.panel_layout import assign, load_panel, panel_roi, tight_roi  # noqa: E402

CIRCLE_LABELS = [("A", "dot"), ("5", "6"), ("3", "4"), ("1", "2"), ("open", "close")]
CIRCLE_ROI = (955, 145, 1130, 480)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="button_coords.json")
    ap.add_argument("--camera", default="cam_chest")
    ap.add_argument("--panel", default="mock_cabinet")
    ap.add_argument("--circles", action="store_true",
                    help="Hough-circle finder + fixed grid instead of the detector")
    args = ap.parse_args()

    cfg = load_pipeline()
    protrusion = float(cfg["elevator"]["press"]["button_protrusion"])
    handle = CameraManager().build().get(args.camera)
    if handle.extrinsic is None:
        print("!! camera has no eye-to-hand extrinsic — run run_calibration.py first")
        return 1
    base_T_cam = handle.extrinsic
    cam = handle.camera

    detector = layout = None
    if not args.circles:
        from yolo.trt_detector import TrtButtonDetector  # noqa: E402
        detector = TrtButtonDetector()
        layout = load_panel(args.panel)
        print(f"panel {layout.id!r}: {layout.shape} grid, engine {detector.engine_path.name}")
    cam.start()

    buttons3d = plane = None
    for attempt in range(6):
        frame = cam.capture()
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        if args.circles:
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=CIRCLE_ROI)
            if fit is None:
                print(f"  [{attempt + 1}/6] plane fit failed")
                continue
            grid, p2 = detect_buttons(bgr, CIRCLE_ROI, CIRCLE_LABELS)
            if grid is None:
                print(f"  [{attempt + 1}/6] no circles")
                continue
            print(f"  [{attempt + 1}/6] {len(grid)} buttons (Hough, param2={p2})")
        else:
            coarse = panel_roi(bgr, detector, verbose=False)
            if coarse is None:
                print(f"  [{attempt + 1}/6] no buttons in frame")
                continue
            grid, rep = assign(bgr, coarse, detector, layout, verify=True)
            for line in rep.lines():
                print(f"  [{attempt + 1}/6] {line}")
            if not rep.ok:
                continue
            roi = tight_roi(rep.found, bgr.shape[:2])
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=roi)
            if fit is None:
                print(f"  [{attempt + 1}/6] plane fit failed in {roi}")
                continue

        origin, normal = fit
        buttons3d = {}
        for name, px in grid.items():
            pt = ray_plane_intersection(px, frame.intrinsics, base_T_cam,
                                        origin + protrusion * normal, normal)
            if pt is not None:
                buttons3d[str(name)] = [float(v) for v in pt]
        plane = (origin, normal)
        break

    if buttons3d is None:
        print("!! could not locate the panel/buttons (view blocked, or wrong panel)")
        return 1

    origin, normal = plane
    out = {
        "buttons": buttons3d,
        "plane_point": [float(v) for v in origin],
        "plane_normal": [float(v) for v in (normal / np.linalg.norm(normal))],
        "note": "base-frame (RIGHT arm) button-face coords; perception-only dump",
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {len(buttons3d)} buttons -> {args.out}")
    print("plane_point :", [round(v, 4) for v in out["plane_point"]])
    print("plane_normal:", [round(v, 4) for v in out["plane_normal"]])
    for name, xyz in buttons3d.items():
        print(f"  {name:>6}: [{xyz[0]:+.4f}, {xyz[1]:+.4f}, {xyz[2]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
