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
# Hard stops read from the controller (rm_get_joint_max_pos) on the RIGHT RM-65, not
# guessed. J3 is the one that binds for every button on this panel.
J_LIMIT = np.array([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])


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
    ap.add_argument("--auto", action="store_true",
                    help="watch until the panel is visible AND has stopped moving, "
                         "then press the sequence ONCE and exit. Re-running is the "
                         "manual re-trigger; it never re-arms itself.")
    ap.add_argument("--settle-frames", type=int, default=5,
                    help="consecutive stable frames required by --auto (default 5)")
    ap.add_argument("--settle-mm", type=float, default=2.0,
                    help="max panel movement between frames to count as still (mm)")
    ap.add_argument("--auto-timeout", type=float, default=300.0,
                    help="give up if nothing settles within this many seconds")
    ap.add_argument("--countdown", type=int, default=3,
                    help="seconds of warning before --auto starts moving the arm")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip anchor verification. UNSAFE: a shifted grid then "
                         "presses the wrong floor silently")
    ap.add_argument("--lift", action="store_true",
                    help="LOOK LOW, PRESS HIGH: detect once at the current (visible) "
                         "height, then per button raise the torso lift to bring that "
                         "button to arm.lift.target_relative_z (best joint margin) and "
                         "press from the compensated coord. Needs arm.lift in config. "
                         "Restores the lift afterwards. No single height reaches all "
                         "rows with safe margin, so this is required for full coverage.")
    args = ap.parse_args()

    cfg = load_pipeline()
    tcp = np.array(cfg["end_effector"]["tcp_offset"], dtype=np.float64)
    protrusion = cfg["elevator"]["press"]["button_protrusion"]
    push = args.push / 1000.0 if args.push is not None else cfg["elevator"]["press"]["push_depth"]
    home = cfg["arm"]["home_joints_deg"]
    LIM = cfg["arm"]["limits"]

    arm = RealmanArm(side=cfg["arm"]["side"])
    arm.connect()
    sdk = arm._require()
    handle = CameraManager().build().get(args.camera)
    base_T_cam = handle.extrinsic
    cam = handle.camera

    # ---- torso lift: SEARCHED, not computed --------------------------------
    # The lift is a degree of freedom, so it is searched exactly like the approach
    # roll, and for the same reason. An earlier version computed one height per button
    # from a fixed `target_relative_z`, and that target had been measured at ONE
    # docking distance. Re-docked 12 cm further out it picked the worst heights
    # available: button `1` went to 12 of 24 rolls at the computed height and 0 of 24
    # at the next one up, while EVERY height from 444 to 944 gave 24/24 with 41-56 deg
    # of margin. Reachability is a function of distance AND height; only the height
    # axis was measured, and the conclusion was over-generalised.
    #
    # Searching removes the need for that map. It also prefers NOT to move: the torso
    # only rises when the current height cannot deliver `prefer_margin_deg`, which at
    # a comfortable docking distance means no torso motion at all.
    #
    # Mechanics that bite: the lift hangs off the LEFT controller (the right one
    # reports a meaningless pos=0), it reports TWICE the real travel, and the blocking
    # form of rm_set_lift_height hangs forever past the travel limit. The base frame
    # rides the lift, so raising by `rise` metres lowers a button's base-frame z by the
    # same `rise` (x and y unchanged) — and the panel PLANE moves with it too.
    lift_arm = lift_cfg = None
    L0 = None
    if args.lift:
        lift_cfg = cfg["arm"].get("lift")
        if lift_cfg is None:
            print("!! --lift needs `arm.lift` in configs/pipeline.yaml")
            return 1
        lift_arm = RealmanArm(side=lift_cfg.get("controller", "left"))
        lift_arm.connect()
        L0 = lift_arm.get_lift_height()          # command units at detection time
        print(f"lift: on the {lift_cfg.get('controller','left')} controller, "
              f"currently at command {L0:.0f}")

    def move_lift(cmd_target, speed=30, timeout=25.0, tol=3.0):
        """Move the lift to a command value; poll until it arrives (mode != 2)."""
        cmd_target = float(np.clip(cmd_target, lift_cfg["command_min"],
                                   lift_cfg["command_max"]))
        ls = lift_arm._require()
        ls.rm_set_lift_height(int(speed), int(round(cmd_target)), 0)  # non-blocking
        t0 = time.time()
        while time.time() - t0 < timeout:
            code, st = ls.rm_get_lift_state()
            if code == 0 and st and st.get("mode") != 2 \
                    and abs(st.get("pos", 0) - cmd_target) <= tol:
                return cmd_target
            time.sleep(0.3)
        return None

    def lift_options():
        """Lift commands to try, current height FIRST so a tie prefers not moving."""
        lo, hi = lift_cfg["command_min"], lift_cfg["command_max"]
        step = int(lift_cfg.get("search_step", 50))
        out, seen = [], set()
        for c in [L0] + list(range(int(lo), int(hi) + 1, step)):
            c = float(np.clip(c, lo, hi))
            if round(c) not in seen:
                seen.add(round(c))
                out.append(c)
        return out

    def rise_of(cmd):
        """Real vertical rise, in metres, of going from L0 to `cmd`."""
        return (cmd - L0) / lift_cfg["command_per_mm"] / 1000.0

    def plan_over_lift(button0, seed):
        """Search (lift height, approach roll) together.

        Returns ``(cmd, rise, best, reasons, tried)``. Selection: among heights whose
        best pose clears `prefer_margin_deg`, take the one needing the LEAST torso
        motion (ties to the larger margin); if none clears it, take the largest margin
        available. Preferring stillness matters — moving the torso costs seconds, and
        it drops the panel out of the camera's view, so it should happen only when the
        arm genuinely cannot do the job from where it is.
        """
        prefer = float(lift_cfg.get("prefer_margin_deg", 20.0))
        feasible, last_reasons, tried = [], {}, 0
        for cmd in lift_options():
            rise = rise_of(cmd)
            b = np.asarray(button0, dtype=np.float64) - np.array([0.0, 0.0, rise])
            best, reasons = plan(b, seed, origin - np.array([0.0, 0.0, rise]))
            tried += 1
            if best is None:
                last_reasons = reasons or last_reasons
                continue
            feasible.append((cmd, rise, best))
        if not feasible:
            return None, 0.0, None, last_reasons, tried
        good = [f for f in feasible if f[2][3] >= prefer]
        if good:
            cmd, rise, best = min(good, key=lambda f: (abs(f[0] - L0), -f[2][3]))
        else:
            cmd, rise, best = max(feasible, key=lambda f: f[2][3])
        return cmd, rise, best, {}, tried

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

    def locate(verbose=True, tag=""):
        """One attempt at locating the panel and every button. None if it failed.

        Returns ``(buttons3d, (origin, normal))`` in the arm base frame.
        """
        frame = cam.capture()
        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)

        if args.circles:
            roi = CIRCLE_ROI
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=roi)
            if fit is None:
                return None
            grid, p2 = detect_buttons(bgr, roi, CIRCLE_LABELS)
            if grid is None:
                return None
            if verbose:
                print(f"{tag}located panel + {len(grid)} buttons (Hough, param2={p2})")
        else:
            # Order matters: buttons first, THEN the plane. The coarse ROI is
            # deliberately generous and can reach past the faceplate onto the
            # cabinet — a second, roughly parallel plane that would bias the fit.
            # Once the buttons are known, the fit gets a box around them alone.
            coarse = panel_roi(bgr, detector, verbose=verbose)
            if coarse is None:
                if verbose:
                    print(f"{tag}no buttons in the frame at all")
                return None
            grid, rep = assign(bgr, coarse, detector, layout,
                               verify=not args.no_verify)
            if verbose:
                for line in rep.lines():
                    print(f"{tag}{line}")
            if not rep.ok:
                return None
            roi = tight_roi(rep.found, bgr.shape[:2])
            fit = fit_panel_plane_from_depth(frame, base_T_cam, roi=roi)
            if fit is None:
                if verbose:
                    print(f"{tag}plane fit failed in ROI {roi}")
                return None
            if verbose:
                print(f"{tag}plane fitted in {roi} (coarse was {coarse})")

        org, nrm = fit
        pts = {name: ray_plane_intersection(px, frame.intrinsics, base_T_cam,
                                            org + protrusion * nrm, nrm)
               for name, px in grid.items()}
        return pts, (org, nrm)


    def centroid(pts):
        return np.mean(np.array(list(pts.values()), dtype=np.float64), axis=0)


    def wait_until_settled():
        """Watch until the panel is visible AND has stopped moving, then return it.

        The trigger is the PANEL's pose in the arm base frame, not the base's own
        odometry — which is deliberate. The chassis reporting "stopped" is not the same
        as the panel being still relative to the arm (the body rocks after a stop), and
        a chassis nudge that does not move the panel is not worth waiting out. The
        camera measures the quantity the press actually depends on. It also needs no
        access to the base at all: the only candidate found on the wired
        network is an unidentified service on port 9090 with no client library here.
        """
        need = args.settle_frames
        tol = args.settle_mm / 1000.0
        deadline = time.time() + args.auto_timeout
        stable, prev, last = 0, None, None
        misses = 0
        while time.time() < deadline:
            got = locate(verbose=False)
            if got is None:
                misses += 1
                if stable:
                    print(f"    lost the panel after {stable} stable frame(s)")
                stable, prev = 0, None
                if misses % 10 == 0:
                    print(f"    waiting: panel not located ({misses} frames)")
                continue
            pts, pl = got
            c = centroid(pts)
            if prev is not None:
                moved = float(np.linalg.norm(c - prev)) * 1000
                if moved <= args.settle_mm:
                    stable += 1
                else:
                    if stable:
                        print(f"    moving again ({moved:.1f} mm) — resetting")
                    stable = 0
            prev, last = c, (pts, pl)
            if stable >= need:
                print(f"    settled: {need} consecutive frames within "
                      f"{args.settle_mm} mm")
                return last
            if stable == 1:
                print(f"    panel visible, checking it is still…")
        print(f"!! gave up after {args.auto_timeout:.0f} s without a settled panel")
        return None


    if args.auto:
        print(f"AUTO: waiting for the panel to be visible and still "
              f"({args.settle_frames} frames within {args.settle_mm} mm), then pressing "
              f"{' '.join(args.buttons)} ONCE and exiting.")
        got = wait_until_settled()
        if got is None:
            return 1
        buttons3d, plane = got
        if args.go:
            for k in range(args.countdown, 0, -1):
                print(f"    pressing in {k}…  (Ctrl+C to cancel)")
                time.sleep(1.0)
    else:
        buttons3d = plane = None
        # 15 attempts, not 6. Localisation succeeds on roughly a quarter of frames at a
        # re-docked distance (the detector drops a button or two and the lattice needs
        # 70 % of them), and three consecutive runs used attempts 4, 1 and 5 of 6 — one
        # bad frame away from failing outright. Each attempt costs ~200 ms, so the whole
        # budget is 3 s against a 60 s sequence.
        for attempt in range(15):
            got = locate(tag=f"  [{attempt + 1}/15] ")
            if got is not None:
                buttons3d, plane = got
                break
        if buttons3d is None:
            print("!! could not locate the panel/buttons — is the arm blocking the "
                  "view, or is this a different panel than the registered layout?")
            return 1
    origin, normal = plane
    inward = -normal

    # roll basis about the approach axis
    e1 = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(inward, e1)

    from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t  # noqa: E402

    def ik(target_tip, R, seed):
        cmd = make_transform(R, target_tip) @ make_transform(np.eye(3), -tcp)
        params = rm_inverse_kinematics_params_t(
            seed, list(cmd[:3, 3]) + list(matrix_to_rpy(R, degrees=False)), 1)
        code, sol = sdk.rm_algo_inverse_kinematics(params)
        return [float(x) for x in sol] if code == 0 else None

    def joint_margin(q):
        """Degrees the closest joint has left before its hard stop."""
        q = np.asarray(q, dtype=np.float64)
        return float(np.min(np.minimum(J_LIMIT - q, q + J_LIMIT)))

    def self_collides(seed, q_off, q_on):
        """True if the arm self-collides anywhere along seed -> standoff -> contact.

        Uses the controller's OWN model via rm_algo_safety_robot_self_collision_detection
        (0 = clear, nonzero = collision OR joint-limit exceeded). Joints are already
        margin-checked before this runs, so a hit here is a genuine self-collision.
        Pure computation, no motion. NOTE this covers the arm's own links + the
        end-effector per the controller model only — NOT the other arm, the chassis,
        or the door frame (those need virtual walls). Sampled densely because a
        configuration can pass through a collision between two clear endpoints.
        """
        path = np.vstack([np.linspace(seed, q_off, 25), np.linspace(q_off, q_on, 6)])
        for q in path:
            if sdk.rm_algo_safety_robot_self_collision_detection(
                    [float(x) for x in q[:6]]) != 0:
                return True
        return False

    def plan(button, seed, org):
        """Choose an approach roll that satisfies the boundaries, or explain why not.

        The roll is SEARCHED, not hard-coded — what is fixed is which poses are
        disallowed (`arm.limits` in configs/pipeline.yaml). Two things this fixes:

        * The old rule took the least-joint-travel solution, which on this panel
          meant poses sitting on a hard stop: 0.1 deg of J3 margin for `2`, 0.7 for
          `4`. J3 binds for every button here and the controller's self-collision
          check is off, so an IK solution alone is no guarantee.
        * The contact pose was never IK-checked at all — only the standoff was, and
          the press itself was a Cartesian `movel` into whatever lay 50 mm further
          in. Driving a straight line into a pose with no good solution is exactly
          where the arm jams.
        * The controller's self-collision check is off, so the whole seed -> standoff
          -> contact path is now checked here at planning time via
          `rm_algo_safety_robot_self_collision_detection` (arm's own links + EE only;
          the other arm / chassis / door still need virtual walls).

        Returns ``(travel, deg, q_standoff, margin, reasons)``; ``reasons`` counts why
        rolls were rejected, so a refusal says WHICH boundary bit.
        """
        cands, reasons = [], {}

        def reject(why):
            reasons[why] = reasons.get(why, 0) + 1

        for deg in range(0, 360, 15):
            a = np.radians(deg)
            R = _orthonormal_frame(inward, up_hint=np.cos(a) * e1 + np.sin(a) * e2)
            q_off = ik(button + normal * STANDOFF, R, seed)
            if q_off is None:
                reject("no IK at standoff")
                continue
            q_on = ik(button - normal * push, R, q_off)
            if q_on is None:
                reject("no IK at contact")
                continue
            margin = min(joint_margin(q_off), joint_margin(q_on))
            if margin < LIM["min_joint_margin_deg"]:
                reject(f"joint margin < {LIM['min_joint_margin_deg']} deg")
                continue
            wrist = min(abs(q_off[4]), abs(q_on[4]))
            if wrist < LIM["min_wrist_deg"]:
                reject(f"wrist singularity (|J5| < {LIM['min_wrist_deg']} deg)")
                continue
            steps = np.linspace(seed, q_off, 21)
            jump = max(float(np.abs(np.diff(steps, axis=0)).max()),
                       float(np.abs(np.array(q_on) - np.array(q_off)).max()))
            if jump > LIM["max_path_jump_deg"]:
                reject(f"path jump > {LIM['max_path_jump_deg']} deg "
                       "(configuration flip)")
                continue
            clear = clearance(seed, q_off, org)
            if clear < LIM["min_clearance_mm"]:
                reject(f"clearance < {LIM['min_clearance_mm']} mm")
                continue
            if self_collides(seed, q_off, q_on):
                reject("self-collision on path")
                continue
            travel = max(abs(x - y) for x, y in zip(q_off, seed))
            cands.append((travel, deg, q_off, margin, clear))
        if not cands:
            return None, reasons
        # Largest joint-limit margin wins; joint travel only breaks ties. Maximising
        # margin alone would swing the path into the panel (measured -12.0 mm for
        # `dot`), which is what the clearance boundary above is there to stop.
        cands.sort(key=lambda c: (-c[3], c[0]))
        travel, deg, q_off, margin, clear = cands[0]
        return (travel, deg, q_off, margin, clear, len(cands)), reasons

    def clearance(seed, goal, org):
        """Closest the plunger tip gets to the panel along the joint-interpolated path.

        ``org`` must be the plane origin AS SEEN FROM THE CURRENT lift height. With
        --lift the base frame rides the lift, so a plane measured before the move is
        stale by the rise; the distance is taken along the panel normal, so the error
        is only ``rise * normal_z`` — measured at 4.7-6.4 mm for rises of 242-328 mm,
        which flipped no verdict here (real clearances are 45-52 mm against a 5 mm
        threshold) and errs on the conservative side. Passed explicitly anyway,
        because the day a path does graze the faceplate is not the day to discover it.
        """
        worst = 1e9
        for i in range(21):
            q = [x + (y - x) * (i / 20.0) for x, y in zip(seed, goal)]
            fk = sdk.rm_algo_forward_kinematics(q, 1)
            p = np.array(fk[:3])
            R = rpy_to_matrix(*fk[3:], degrees=False)
            worst = min(worst, float((p + R @ tcp - org) @ normal) * 1000)
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
        origin_now = origin

        # Home FIRST: the seed decides which IK branch comes back, so planning has to
        # start from the pose the arm will actually depart from.
        seed = list(arm.get_joint_angles())
        if max(abs(a - b) for a, b in zip(seed, home)) > 3.0:
            if not arm.move_joints_sync(home):
                print("    cannot return home, aborting")
                results.append((name, False))
                break
            seed = list(arm.get_joint_angles())

        if args.lift:
            cmd, rise, best, reasons, tried = plan_over_lift(button, seed)
            if best is None:
                print(f"    no pose satisfies the boundaries at ANY of {tried} lift "
                      "heights:")
                for why, n in sorted((reasons or {}).items(), key=lambda kv: -kv[1]):
                    print(f"      {n:>2}/24 rolls  {why}")
                results.append((name, False))
                continue
            if abs(cmd - L0) < 1.0:
                print(f"    lift stays at {L0:.0f} (no torso motion needed)")
            else:
                print(f"    lift -> command {cmd:.0f} (raise {rise * 1000:+.0f} mm), "
                      f"chosen from {tried} heights")
                if args.go and move_lift(cmd) is None:
                    print("    lift move did not confirm; skipping")
                    results.append((name, False))
                    continue
            # Both the button and the plane ride the lift: raising by `rise` lowers
            # their base-frame z by `rise` (x, y unchanged).
            button = np.asarray(button, dtype=np.float64) - np.array([0.0, 0.0, rise])
            origin_now = origin - np.array([0.0, 0.0, rise])
        else:
            best, reasons = plan(button, seed, origin_now)
        if best is None:
            print("    no pose satisfies the boundaries:")
            for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
                print(f"      {n:>2}/24 rolls  {why}")
            results.append((name, False))
            continue
        travel, deg, goal, margin, clear, n_ok = best
        print(f"    roll={deg} travel={travel:.0f} deg  joint margin={margin:.1f} deg  "
              f"clearance={clear:.0f} mm  ({n_ok}/24 rolls passed)")
        if reasons:
            worst = sorted(reasons.items(), key=lambda kv: -kv[1])[:2]
            print("      rejected: " + ", ".join(f"{n}x {why}" for why, n in worst))
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

    if args.lift and lift_arm is not None:
        if args.go and L0 is not None:
            print(f"restoring lift to command {L0:.0f}")
            move_lift(L0)
        lift_arm.disconnect()
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
