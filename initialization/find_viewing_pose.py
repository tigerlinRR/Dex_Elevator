#!/usr/bin/env python3
"""Find and record a good VIEWING POSE for an arm-mounted camera.

With the camera on the arm, "where does the robot look from" becomes a real
parameter, and it is not the same thing as "where can the arm reach". This searches
a small neighbourhood of the current pose and scores each one on what the press
actually depends on, using the SAME detector the press uses — a pose that scores well
here is one the press will succeed from, not one that merely looks nice.

**Why a search and not a live hunt at run time.** The panel does not move and the
dock does not move, so the viewing pose is a constant that can be measured once and
replayed — the same argument `panels.yaml` makes for the button layout. Measured on
this cell, the base docks to 1.5-2.3 cm, which at 0.5 m is ~2.3 degrees of bearing
against a camera with an 85 degree field of view: there is nothing to search for at
run time. Searching belongs here, once, at commissioning.

Four things are scored, all of which have bitten this project:

* **buttons found** — the whole point; a pose that sees 6 of 10 is not a viewing pose.
* **button size in pixels** — the classifier degrades with it, measured: 44-50 px reads
  labels correctly, 28 px scores 2-3 correct out of 10. Too large is bad too (the panel
  stops fitting in frame).
* **centring** — a panel against the frame edge loses buttons the moment the base docks
  a centimetre differently, and the ROI derivation needs room to grow into.
* **squareness** — the angle between the optical axis and the panel normal, from the
  live depth fit. A steeply oblique view costs both button-centre precision and plane
  fit quality.

    python3 initialization/find_viewing_pose.py                 # search, report, save nothing
    python3 initialization/find_viewing_pose.py --go-to-best    # also move there at the end
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from core.camera.manager import CameraManager
from core.config import load_pipeline
from core.press import fit_panel_plane_from_depth
from core.robot.realman import RealmanArm
from yolo.panel_layout import load_panel, panel_roi

JOINT_LIMITS_DEG = [178, 130, 135, 178, 128, 360]


def _safe(j) -> bool:
    return all(abs(j[i]) <= JOINT_LIMITS_DEG[i] - 2.0 for i in range(6))


def score_view(bgr, frame, detector, base_T_cam, layout_shape):
    """Return (score, detail) for one view. Higher is better; -1 means unusable."""
    dets = detector.detect(bgr)
    n = len(dets)
    if n < 4:
        return -1.0, {"buttons": n, "why": "too few buttons"}
    xs = np.array([d.center[0] for d in dets])
    ys = np.array([d.center[1] for d in dets])
    ws = np.array([abs(d.bbox_xyxy[2] - d.bbox_xyxy[0]) for d in dets])
    h, w = bgr.shape[:2]

    size_px = float(np.median(ws))
    # Peak at ~46 px, the band where the classifier was measured to read labels
    # correctly; falls off both ways.
    size_term = float(np.exp(-((size_px - 46.0) / 22.0) ** 2))

    cx, cy = float(xs.mean()), float(ys.mean())
    off = np.hypot((cx - w / 2) / (w / 2), (cy - h / 2) / (h / 2))
    centre_term = float(max(0.0, 1.0 - off))

    # Squareness from the live plane fit, over the detections' own box.
    x0, y0 = int(max(0, xs.min() - 40)), int(max(0, ys.min() - 40))
    x1, y1 = int(min(w, xs.max() + 40)), int(min(h, ys.max() + 40))
    square_term, ang = 0.0, float("nan")
    fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=(x0, y0, x1, y1))
    if fit is not None:
        _, normal = fit
        optical = base_T_cam[:3, 2]
        ang = float(np.degrees(np.arccos(np.clip(abs(normal @ optical), -1, 1))))
        square_term = float(max(0.0, 1.0 - ang / 60.0))

    found_term = min(1.0, n / max(1, int(np.prod(layout_shape)) if layout_shape else 10))
    score = 2.0 * found_term + 1.0 * size_term + 1.0 * centre_term + 1.5 * square_term
    return score, {"buttons": n, "size_px": size_px, "off_centre": off,
                   "obliquity_deg": ang, "score": score}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="cam_arm")
    ap.add_argument("--panel", default="mock_cabinet")
    ap.add_argument("--speed", type=float, default=0.08)
    ap.add_argument("--settle", type=float, default=1.2)
    ap.add_argument("--back-off", type=str, default="",
                    help="comma-separated cm to retreat ALONG THE OPTICAL AXIS from the "
                         "start pose, e.g. 0,10,20,25. Joint tweaks cannot fix framing "
                         "that is wrong because the camera is simply too close: button "
                         "size in pixels is set by distance, and the classifier wants "
                         "44-50 px. Uses a Cartesian move, so the view direction is kept")
    ap.add_argument("--go-to-best", action="store_true",
                    help="drive to the best pose found at the end (default: return to start)")
    args = ap.parse_args()

    cfg = load_pipeline()
    arm_cfg = cfg.get("arm", {})
    arm = RealmanArm(side=arm_cfg.get("side", "right"), ip=arm_cfg.get("ip"),
                     port=arm_cfg.get("port", 8080),
                     max_speed_pct=arm_cfg.get("max_speed_pct", 20))
    arm.connect()
    home = np.asarray(arm.get_joint_angles(), dtype=float)
    print("start joints:", np.array2string(home, precision=1))

    handle = CameraManager().build().get(args.camera)
    if handle.mount != "arm":
        print(f"!! {args.camera} is mount={handle.mount}; this tool is for an arm-mounted camera")
        arm.disconnect()
        return 1

    from yolo.trt_detector import TrtButtonDetector
    detector = TrtButtonDetector()
    try:
        layout = load_panel(args.panel)
        shape = layout.shape
    except Exception:
        shape = ()

    # A small neighbourhood: enough to fix framing, not enough to drive into anything.
    offsets = [("start", (0, 0, 0, 0, 0, 0))]
    for d1 in (-8, 8):
        offsets.append((f"J1 {d1:+d}", (d1, 0, 0, 0, 0, 0)))
    for d2 in (-6, 6):
        offsets.append((f"J2 {d2:+d}", (0, d2, 0, 0, 0, 0)))
    for d3 in (-6, 6):
        offsets.append((f"J3 {d3:+d}", (0, 0, d3, 0, 0, 0)))
    for d5 in (-10, -5, 5, 10):
        offsets.append((f"J5 {d5:+d}", (0, 0, 0, 0, d5, 0)))
    for d1, d5 in ((-6, -6), (6, -6), (-6, 6), (6, 6)):
        offsets.append((f"J1 {d1:+d} J5 {d5:+d}", (d1, 0, 0, 0, d5, 0)))

    # Distance sweep: retreat along the camera's own optical axis (TCP -z), which
    # keeps the panel centred while changing only how much of it fits in frame.
    if args.back_off:
        T0 = arm.get_tcp_pose()
        handle.camera.start()
        results = []
        try:
            for idx, cm in enumerate([float(c) for c in args.back_off.split(",")], 1):
                T = T0.copy()
                T[:3, 3] = T0[:3, 3] - T0[:3, 2] * (cm / 100.0)
                print(f"[{idx}] back {cm:.0f} cm along the optical axis...", flush=True)
                if not arm.move_line_sync(T, speed=0.05):
                    print("   move refused (no IK on this straight line) — skipped")
                    continue
                time.sleep(args.settle)
                frame = handle.camera.capture()
                base_T_cam = handle.base_T_camera(arm.get_tcp_pose())
                bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
                sc, det = score_view(bgr, frame, detector, base_T_cam, shape)
                j_here = np.asarray(arm.get_joint_angles(), dtype=float)
                results.append((sc, f"back {cm:.0f}cm", j_here, det))
                if sc < 0:
                    print(f"   unusable ({det.get('why')}, {det['buttons']} buttons)")
                else:
                    print(f"   score {sc:5.2f}  buttons {det['buttons']:2d}  "
                          f"size {det['size_px']:5.1f}px  off-centre {det['off_centre']:.2f}  "
                          f"oblique {det['obliquity_deg']:5.1f}deg")
        finally:
            handle.camera.stop()
        results.sort(key=lambda r: -r[0])
        if results and results[0][0] > 0:
            bs, bn, bj, bd = results[0]
            print(f"\n=== best: {bn}  score {bs:.2f} ===")
            print(f"  buttons {bd['buttons']}, {bd['size_px']:.1f} px, "
                  f"off-centre {bd['off_centre']:.2f}, oblique {bd['obliquity_deg']:.1f} deg")
            print("  view_joints_deg_" + args.panel + ": [" +
                  ", ".join(f"{v:.2f}" for v in bj) + "]")
            dest = bj if args.go_to_best else home
        else:
            print("\nno usable distance found")
            dest = home
        print(f"\nmoving to {'the best pose' if args.go_to_best and results else 'the start pose'}")
        try:
            arm.move_joints_sync(list(dest), speed=args.speed)
        except Exception as e:
            print(f"!! could not move: {e}")
        arm.disconnect()
        return 0

    handle.camera.start()
    results = []
    try:
        for idx, (name, off) in enumerate(offsets, 1):
            target = home + np.asarray(off, dtype=float)
            if not _safe(target):
                print(f"[{idx}/{len(offsets)}] {name}: outside joint limits — skipped")
                continue
            if not arm.move_joints_sync(list(target), speed=args.speed):
                print("   !! arrival not confirmed — stopping")
                try:
                    arm._require().rm_set_arm_stop()
                    arm._require().rm_set_delete_current_trajectory()
                except Exception:
                    pass
                break
            time.sleep(args.settle)
            frame = handle.camera.capture()
            base_T_cam = handle.base_T_camera(arm.get_tcp_pose())
            bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
            s, det = score_view(bgr, frame, detector, base_T_cam, shape)
            results.append((s, name, target.copy(), det))
            if s < 0:
                print(f"[{idx}/{len(offsets)}] {name:<14} unusable ({det.get('why')}, "
                      f"{det['buttons']} buttons)")
            else:
                print(f"[{idx}/{len(offsets)}] {name:<14} score {s:5.2f}  "
                      f"buttons {det['buttons']:2d}  size {det['size_px']:5.1f}px  "
                      f"off-centre {det['off_centre']:.2f}  oblique {det['obliquity_deg']:5.1f}deg")
    except KeyboardInterrupt:
        print("\ninterrupted")
        try:
            arm._require().rm_set_arm_stop()
        except Exception:
            pass
    finally:
        handle.camera.stop()

    results.sort(key=lambda r: -r[0])
    if not results or results[0][0] < 0:
        print("\nno usable viewing pose found in this neighbourhood — move the arm closer "
              "to facing the panel and run again")
        arm.disconnect()
        return 1

    best_s, best_name, best_j, best_det = results[0]
    print(f"\n=== best: {best_name}  score {best_s:.2f} ===")
    print(f"  buttons {best_det['buttons']}, {best_det['size_px']:.1f} px, "
          f"off-centre {best_det['off_centre']:.2f}, oblique {best_det['obliquity_deg']:.1f} deg")
    print(f"  joints: {np.array2string(best_j, precision=2, separator=', ')}")
    print("\nTo register it, put this under `arm:` in configs/pipeline.yaml:")
    print("  view_joints_deg_" + args.panel + ": [" +
          ", ".join(f"{v:.2f}" for v in best_j) + "]")

    dest = best_j if args.go_to_best else home
    print(f"\nmoving to {'the best pose' if args.go_to_best else 'the start pose'}")
    try:
        arm.move_joints_sync(list(dest), speed=args.speed)
    except Exception as e:
        print(f"!! could not move: {e}")
    arm.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
