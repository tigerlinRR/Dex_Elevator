#!/usr/bin/env python3
"""Press a sequence of elevator buttons autonomously, returning home between each.

    python3 initialization/press_buttons.py 1 4 2 5          # plan only
    python3 initialization/press_buttons.py 1 4 2 5 --go     # execute
    python3 initialization/press_buttons.py 1 --push=4.0 --go

Verified on hardware 2026-08-20: 4/4 buttons lit, depth error <=0.05 mm, lateral
<=0.24 mm, ~48 s for four buttons.

Per button: home -> movej to standoff -> movel through contact into the button
-> movel retract -> movej home. Every motion confirms arrival before the next
step is planned (see RealmanArm.move_joints_sync / move_line_sync for why).

Two decisions in here were bought with debugging time:

* Button 3D coordinates are located ONCE at the start and cached. The base and
  panel do not move during a sequence, so those coordinates are constants;
  re-detecting per button only adds chances to fail (an earlier version lost 2 of
  3 buttons to intermittent circle-detection misses).
* The panel plane is fitted live rather than read from config, because the base
  docks with centimetres of error. Between two sessions on the same day the panel
  moved 16 cm — a stored plane would have been silently wrong.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

REPO = __file__.rsplit("/", 2)[0]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2  # noqa: E402

from core.camera.manager import CameraManager  # noqa: E402
from core.config import load_pipeline  # noqa: E402
from core.press import (  # noqa: E402
    _orthonormal_frame, fit_panel_plane_from_depth, ray_plane_intersection,
)
from core.robot.realman import RealmanArm  # noqa: E402
from core.transforms import make_transform, matrix_to_rpy, rpy_to_matrix  # noqa: E402
from yolo.button_circles import detect_buttons  # noqa: E402
from yolo.panel_layout import assign, load_panel, panel_roi, tight_roi  # noqa: E402

# Fallback layout for --circles (Hough circles cannot read labels, so the mapping
# has to be supplied). The detector path gets its layout from configs/panels.yaml.
CIRCLE_LABELS = [("A", "dot"), ("5", "6"), ("3", "4"), ("1", "2"), ("open", "close")]
CIRCLE_ROI = (955, 145, 1130, 480)  # faceplate only: the wall behind is a second plane
STANDOFF = 0.050                     # m. Cannot exceed ~50 mm — beyond that the target
                                     # sits inside the arm's inner unreachable region.


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("buttons", nargs="+", help="button labels, e.g. 1 4 2 5")
    ap.add_argument("--go", action="store_true", help="execute (default: plan only)")
    ap.add_argument("--push", type=float, default=None, help="override push depth (mm)")
    ap.add_argument("--camera", default="cam_chest")
    ap.add_argument("--panel", default="mock_cabinet",
                    help="registered layout id from configs/panels.yaml")
    ap.add_argument("--circles", action="store_true",
                    help="use the old Hough-circle finder + hard-coded grid instead "
                         "of the detector (fallback / A-B comparison)")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip anchor verification. UNSAFE: a shifted grid then "
                         "presses the wrong floor silently")
    args = ap.parse_args()

    cfg = load_pipeline()
    tcp = np.array(cfg["end_effector"]["tcp_offset"], dtype=np.float64)
    protrusion = cfg["elevator"]["press"]["button_protrusion"]
    push = args.push / 1000.0 if args.push is not None else cfg["elevator"]["press"]["push_depth"]
    home = cfg["arm"]["home_joints_deg"]

    arm = RealmanArm(side=cfg["arm"]["side"])
    arm.connect()
    sdk = arm._require()
    handle = CameraManager().build().get(args.camera)
    base_T_cam = handle.extrinsic
    cam = handle.camera

    # ---- locate the panel and every button once ----
    # Detector path (default): a full-frame pass gives the button boxes, their union
    # becomes the ROI, and a second pass on that crop gives the positions. Deriving
    # the ROI from the detections rather than hard-coding it is what keeps the
    # robot's own arm out of the plane fit — with the hand in a fixed ROI the fit
    # was dragged 23 mm. It also removes the last panel-specific constant.
    # Build the detector BEFORE starting the camera. Loading a 44 MB TensorRT
    # engine and allocating its buffers takes a second or two, and doing that
    # between start() and the first capture() starved the colour stream: the 335
    # emits depth-only framesets until colour syncs in, and it burned through all
    # 40 retries. Symptom is "no color frame after 40 tries", which reads like a
    # broken camera.
    detector = layout = None
    if not args.circles:
        from yolo.trt_detector import TrtButtonDetector  # noqa: E402
        detector = TrtButtonDetector()
        layout = load_panel(args.panel)
        print(f"panel {layout.id!r}: {layout.shape} grid, "
              f"{len(layout.anchors)} anchors, engine {detector.engine_path.name}")
    cam.start()

    buttons3d = plane = None
    for attempt in range(6):
        frame = cam.capture()
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        if args.circles:
            roi = CIRCLE_ROI
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=roi)
            if fit is None:
                continue
            grid, p2 = detect_buttons(bgr, roi, CIRCLE_LABELS)
            if grid is None:
                continue
            print(f"located panel + {len(grid)} buttons (Hough, param2={p2})")
        else:
            # Order matters: buttons first, THEN the plane. The coarse ROI is
            # deliberately generous and can reach past the faceplate onto the
            # cabinet — a second, roughly parallel plane that would bias the fit.
            # Once the buttons are known, the fit gets a box around them alone.
            coarse = panel_roi(bgr, detector, verbose=True)
            if coarse is None:
                print(f"  [{attempt + 1}/6] no buttons in the frame at all")
                continue
            grid, rep = assign(bgr, coarse, detector, layout,
                               verify=not args.no_verify)
            for line in rep.lines():
                print(f"  [{attempt + 1}/6] {line}")
            if not rep.ok:
                continue
            roi = tight_roi(rep.found, bgr.shape[:2])
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=roi)
            if fit is None:
                print(f"  [{attempt + 1}/6] plane fit failed in ROI {roi}")
                continue
            print(f"  [{attempt + 1}/6] plane fitted in {roi} "
                  f"(coarse was {coarse})")

        origin, normal = fit
        buttons3d = {
            name: ray_plane_intersection(px, frame.intrinsics, base_T_cam,
                                         origin + protrusion * normal, normal)
            for name, px in grid.items()
        }
        plane = (origin, normal)
        break
    if buttons3d is None:
        print("!! could not locate the panel/buttons — is the arm blocking the view, "
              "or is this a different panel than the registered layout?")
        return 1
    origin, normal = plane
    inward = -normal

    # roll basis about the approach axis
    e1 = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(inward, e1)

    from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t  # noqa: E402

    def plan(target_tip, seed):
        """Best (least joint travel) reachable approach orientation, or None."""
        best = None
        for deg in range(0, 360, 15):
            a = np.radians(deg)
            R = _orthonormal_frame(inward, up_hint=np.cos(a) * e1 + np.sin(a) * e2)
            cmd = make_transform(R, target_tip) @ make_transform(np.eye(3), -tcp)
            params = rm_inverse_kinematics_params_t(
                seed, list(cmd[:3, 3]) + list(matrix_to_rpy(R, degrees=False)), 1)
            code, sol = sdk.rm_algo_inverse_kinematics(params)
            if code != 0:
                continue
            sol = [float(x) for x in sol]
            travel = max(abs(x - y) for x, y in zip(sol, seed))
            if best is None or travel < best[0]:
                best = (travel, deg, sol)
        return best

    def clearance(seed, goal):
        """Closest the plunger tip gets to the panel along the joint-interpolated path."""
        worst = 1e9
        for i in range(21):
            q = [x + (y - x) * (i / 20.0) for x, y in zip(seed, goal)]
            fk = sdk.rm_algo_forward_kinematics(q, 1)
            p = np.array(fk[:3])
            R = rpy_to_matrix(*fk[3:], degrees=False)
            worst = min(worst, float((p + R @ tcp - origin) @ normal) * 1000)
        return worst

    results = []
    print(f"sequence {' -> '.join(args.buttons)}   push={push * 1000:.1f} mm")
    started = time.time()
    for name in args.buttons:
        print(f">>> {name}")
        if name not in buttons3d:
            print(f"    unknown button (have {sorted(buttons3d)})")
            results.append((name, False))
            continue
        button = buttons3d[name]
        seed = list(arm.get_joint_angles())
        if max(abs(a - b) for a, b in zip(seed, home)) > 3.0:
            if not arm.move_joints_sync(home):
                print("    cannot return home, aborting")
                results.append((name, False))
                break
            seed = list(arm.get_joint_angles())
        best = plan(button + normal * STANDOFF, seed)
        if best is None:
            print("    no IK solution")
            results.append((name, False))
            continue
        travel, deg, goal = best
        clear = clearance(seed, goal)
        print(f"    roll={deg} travel={travel:.0f} deg clearance={clear:.0f} mm")
        if clear <= 5:
            print("    path would hit the panel, refusing")
            results.append((name, False))
            continue
        if not args.go:
            results.append((name, None))
            continue
        if not arm.move_joints_sync(goal):
            print("    failed to reach standoff")
            results.append((name, False))
            continue
        R = arm.get_tcp_pose()[:3, :3]
        pressed = arm.move_line_sync(make_transform(R, button - normal * push)
                                     @ make_transform(np.eye(3), -tcp))
        tip = (arm.get_tcp_pose() @ np.append(tcp, 1.0))[:3]
        delta = button - tip
        along = float(delta @ inward) * 1000
        lateral = float(np.linalg.norm(delta - (along / 1000) * inward)) * 1000
        arm.move_line_sync(make_transform(R, button + normal * STANDOFF)
                           @ make_transform(np.eye(3), -tcp))
        went_home = arm.move_joints_sync(home)
        if pressed:
            print(f"    pressed {along:+.2f} mm, lateral {lateral:.2f} mm"
                  + ("" if went_home else "  [home NOT confirmed]"))
        else:
            print("    press motion did not complete (retracted)")
        results.append((name, pressed))

    cam.stop()
    arm.disconnect()
    if args.go:
        ok = sum(1 for _, r in results if r)
        print(f"=== {ok}/{len(results)} pressed in {time.time() - started:.1f} s ===")
        return 0 if ok == len(results) else 1
    print("=== plan only; pass --go to execute ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
