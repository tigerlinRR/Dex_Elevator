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
from pathlib import Path

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
    ap.add_argument("--camera", default="cam_chest",
                    help="camera that LOCATES the buttons. cam_chest (fixed) is the "
                         "default; cam_arm rides the arm and recomputes base_T_camera "
                         "per frame")
    ap.add_argument("--obstacle-camera", default="cam_chest",
                    help="camera that checks the path is clear. Must be a FIXED mount — "
                         "a camera on the arm moves with the limb it would be watching, "
                         "so it cannot do this job. Only consulted when --camera is "
                         "arm-mounted; otherwise the button camera does both")
    ap.add_argument("--aim-offset", default=None, metavar="X,Y,Z",
                    help="shift every button target by this much in the BASE frame, in "
                         "millimetres. For a bias in WHERE THE BUTTONS ARE, as opposed to "
                         "where the tip is: a tcp_offset error rotates with the approach "
                         "roll, so it cannot produce a bias that stays in the same "
                         "world direction across different rolls — that one comes from "
                         "the hand-eye calibration or the button-centre pixel, and has "
                         "to be corrected here instead")
    ap.add_argument("--hold", type=float, default=0.0, metavar="SEC",
                    help="stay at the contact pose for SEC seconds before retracting. "
                         "For diagnosing WHERE the tip lands: the press itself is too "
                         "quick to photograph, and the press log cannot see an aiming "
                         "error at all (it compares the command against the same assumed "
                         "TCP the command was built from)")
    ap.add_argument("--tcp-offset", default=None, metavar="X,Y,Z",
                    help="override end_effector.tcp_offset, in MILLIMETRES, for this run "
                         "only. The configured value was measured once by hand-holding "
                         "the tip against a button and has no independent cross-check; "
                         "the press log cannot detect an error in it, because the "
                         "commanded and measured sides of its own comparison both use "
                         "this same number. Only the lamps and an outside measurement can")
    ap.add_argument("--home-joints", default=None, metavar="J1,..,J6",
                    help="override the pose the arm departs from and returns to. The "
                         "configured home was chosen to keep the arm OUT of the chest "
                         "camera's view; for an arm-mounted camera the natural home is "
                         "the VIEWING pose, and travelling between the two is a large "
                         "sweep past the panel that nothing in this program models")
    ap.add_argument("--view-joints", default=None, metavar="J1,..,J6",
                    help="joint angles to LOOK from, before locating the panel. Only "
                         "meaningful for an arm-mounted camera, where the arm carries "
                         "the lens: the home pose was chosen to keep the arm OUT of the "
                         "chest camera's view, which is the opposite of what an arm "
                         "camera needs. Defaults to arm.view_joints_deg in pipeline.yaml")
    ap.add_argument("--no-obstacle-check", action="store_true",
                    help="proceed even when no fixed camera can watch the path. This "
                         "removes the only sensor that sees the arm's workspace — the "
                         "base's own obstacle sensors face 180 degrees the other way")
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
    ap.add_argument("--arm-after-move", type=float, default=0.0, metavar="MM",
                    help="with --auto, do not arm until the panel has first gone away "
                         "(not locatable, or moved more than this many mm). Needed when "
                         "the sequence STARTS parked at the panel: otherwise the panel "
                         "is already visible and still, and the press fires before the "
                         "robot has driven anywhere. 0 = arm immediately.")
    ap.add_argument("--countdown", type=int, default=3,
                    help="seconds of warning before --auto starts moving the arm")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip anchor verification. UNSAFE: a shifted grid then "
                         "presses the wrong floor silently")
    ap.add_argument("--lift-objective", choices=("margin", "still"), default=None,
                    help="how the torso height is chosen per button (default: "
                         "arm.lift.objective in the config). `margin` moves to the "
                         "height with the best joint margin for THAT button, so the "
                         "torso visibly tracks the panel row by row; `still` keeps it "
                         "where it is whenever the current height is good enough.")
    ap.add_argument("--no-fist", action="store_true",
                    help="skip closing the LinkerHand before moving. ONLY for a robot "
                         "with no hand fitted — with the hand on, the fingers are the "
                         "front-most part of the end-effector and hit the panel first.")
    ap.add_argument("--wait-go", metavar="PATH", default=None,
                    help="PRE-WARM MODE: do all the position-independent startup "
                         "(connect the arm, open the camera, load the TensorRT engine, "
                         "ramp the GPU clocks), print PREWARM_READY, then block until "
                         "PATH appears before locating the panel. Lets a caller start "
                         "this process while the base is still driving and pay the ~7 s "
                         "of startup in parallel with the drive. The caller still "
                         "decides WHEN to press — nothing here infers arrival, which is "
                         "the mistake that drove the arm into the panel once already.")
    ap.add_argument("--wait-go-timeout", type=float, default=600.0,
                    help="give up waiting for --wait-go PATH after this many seconds")
    ap.add_argument("--lift", action="store_true",
                    help="LOOK LOW, PRESS HIGH: detect once at the current (visible) "
                         "height, then per button raise the torso lift to bring that "
                         "button to arm.lift.target_relative_z (best joint margin) and "
                         "press from the compensated coord. Needs arm.lift in config. "
                         "Restores the lift afterwards. No single height reaches all "
                         "rows with safe margin, so this is required for full coverage.")
    args = ap.parse_args()

    cfg = load_pipeline()
    if args.tcp_offset:
        mm = [float(v) for v in args.tcp_offset.split(",")]
        cfg["end_effector"]["tcp_offset"] = [v / 1000.0 for v in mm]
        print(f"tcp_offset overridden to {mm} mm (config had "
              f"{[round(v*1000, 2) for v in load_pipeline()['end_effector']['tcp_offset']]})")

    tcp = np.array(cfg["end_effector"]["tcp_offset"], dtype=np.float64)
    protrusion = cfg["elevator"]["press"]["button_protrusion"]
    push = args.push / 1000.0 if args.push is not None else cfg["elevator"]["press"]["push_depth"]
    home = cfg["arm"]["home_joints_deg"]
    if args.home_joints:
        home = [float(v) for v in args.home_joints.split(",")]
        print(f"home overridden to {['%.1f' % v for v in home]}")
    LIM = cfg["arm"]["limits"]

    arm = RealmanArm(side=cfg["arm"]["side"])
    arm.connect()
    sdk = arm._require()

    # ---- close the hand BEFORE anything moves -----------------------------
    # This is a prerequisite of pressing, not tidiness. The LinkerHand's fingertips
    # sit 172.87 mm from the flange while the plunger tip is at ~154 mm, so with the
    # fingers extended the HAND is the front-most part of the end-effector and reaches
    # the panel before the plunger does. A fist folds them behind it.
    #
    # It lives here, in the press, because it used to be done out-of-band by whoever
    # was driving — and an out-of-band prerequisite is one someone eventually forgets.
    # The hand also loses power across an emergency stop, so "it was a fist last time"
    # is not a state that survives.
    if not args.no_fist:
        from core.hand.linkerhand import LinkerHand, POSES  # noqa: E402
        hand = LinkerHand(arm)
        try:
            hand.power_on()
            hand.set_speed()
            hand.pose("fist")
            # Verify by READING it back rather than trusting the write. The check is
            # "nothing is sticking out" rather than "each joint hit its target": in a
            # fist the index finger bottoms out against the thumb at ~70 instead of 0,
            # measured on both hands, so an exact-match test would fail every time.
            deadline = time.time() + 4.0
            joints = None
            while time.time() < deadline:
                joints = hand.read_joints()
                if joints is not None and max(joints) <= 100:
                    break
                time.sleep(0.3)
            if joints is None:
                print("!! the hand did not answer — refusing to move. Power-cycle it, "
                      "or pass --no-fist if no hand is fitted.")
                arm.disconnect()
                return 1
            if max(joints) > 100:
                print(f"!! the hand did not close (joints {joints}) — refusing to move; "
                      "extended fingers reach the panel before the plunger.")
                arm.disconnect()
                return 1
            print(f"hand: fist confirmed {joints}")
        except Exception as e:  # noqa: BLE001
            print(f"!! could not close the hand ({type(e).__name__}: {e}) — refusing "
                  "to move. Pass --no-fist only if no hand is fitted.")
            arm.disconnect()
            return 1
    manager = CameraManager().build()
    handle = manager.get(args.camera)
    cam = handle.camera

    # base_T_camera is a CONSTANT only for a torso-mounted camera. With the camera on
    # the arm it is a function of the arm pose, so it must be read at the instant of
    # the frame, with the arm stationary — every caller below takes it from cam_pose()
    # rather than closing over a value.
    if handle.mount == "arm":
        def cam_pose():
            return handle.base_T_camera(arm.get_tcp_pose())
        print(f"camera {args.camera}: mount=arm — base_T_camera recomputed per frame")
    else:
        _fixed_T_cam = handle.extrinsic
        def cam_pose():
            return _fixed_T_cam

    # THE OBSTACLE CHECK NEEDS A CAMERA THAT DOES NOT MOVE WITH THE ARM. It works by
    # asking what sits in front of the panel plane, and that is only meaningful while
    # the robot's own limbs are outside the view — which the home pose guarantees for
    # the chest camera and cannot guarantee for a camera bolted to the arm (the arm IS
    # the mount). So an arm-mounted button camera keeps using the chest camera here.
    obs_handle, obs_cam_pose = handle, cam_pose
    if handle.mount == "arm":
        try:
            obs_handle = manager.get(args.obstacle_camera)
        except KeyError:
            obs_handle = None
        if obs_handle is None or obs_handle.extrinsic is None or obs_handle.mount != "fixed":
            print(f"!! {args.camera} rides the arm, so the path check needs a FIXED camera; "
                  f"'{args.obstacle_camera}' is missing or uncalibrated. Refusing to move — "
                  f"pass --no-obstacle-check to accept that nothing watches the path.")
            if not args.no_obstacle_check:
                arm.disconnect()
                return 1
        else:
            _obs_T = obs_handle.extrinsic
            def obs_cam_pose():
                return _obs_T
            print(f"path check uses {args.obstacle_camera} (fixed) — the arm camera cannot "
                  f"do this job, it moves with the limb it is meant to watch")

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

    def plan_over_lift(button0, seed, l_now):
        """Search (lift height, approach roll) together.

        Returns ``(cmd, rise, best, reasons, tried)``. Two objectives:

        `still`  — among heights whose best pose clears `prefer_margin_deg`, take the
          one needing the LEAST torso motion. Keeps the torso still at a comfortable
          docking distance; moving it costs seconds and drops the panel out of view.

        `margin` — take the height with the BEST joint margin for this button. The
          torso then tracks the panel row by row, which is what an operator watching
          the robot expects to see, and it is not merely cosmetic: the margins it
          selects are the largest available. If that height is where the torso already
          is, the next-best height at least `min_move_command` away is taken instead —
          but only from poses that ALREADY cleared the boundaries, so the visible
          motion never costs safety, only optimality.

        Either way the candidates come from `plan`, which has already rejected
        anything on a joint stop, through a self-collision, or inside the panel.
        """
        prefer = float(lift_cfg.get("prefer_margin_deg", 20.0))
        objective = (args.lift_objective
                     or str(lift_cfg.get("objective", "still")).lower())
        min_move = float(lift_cfg.get("min_move_command", 0.0))
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
        if objective == "margin":
            pool = good or feasible
            cmd, rise, best = max(pool, key=lambda f: f[2][3])
            if min_move and abs(cmd - l_now) < min_move:
                moved = [f for f in pool if abs(f[0] - l_now) >= min_move]
                if moved:
                    cmd, rise, best = max(moved, key=lambda f: f[2][3])
        elif good:
            cmd, rise, best = min(good, key=lambda f: (abs(f[0] - l_now), -f[2][3]))
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

    if args.wait_go is not None:
        # Warm the GPU before releasing the barrier, not after. The Orin idles its
        # GPU at 306 MHz under `schedutil`, so the first inference in a process pays
        # the clock ramp — and here that ramp would otherwise land AFTER the go
        # signal, i.e. inside the time we are trying to remove.
        _warm = cv2.cvtColor(cam.capture().rgb, cv2.COLOR_RGB2BGR)
        if detector is not None:
            for _ in range(30):
                detector.detect(_warm)
        go = Path(args.wait_go)
        if go.exists():
            go.unlink()          # a stale token from a previous run is not a signal
        print("PREWARM_READY", flush=True)
        _t0 = time.time()
        while not go.exists():
            if time.time() - _t0 > args.wait_go_timeout:
                print(f"!! no go signal at {go} after {args.wait_go_timeout:.0f} s")
                return 1
            time.sleep(0.1)
        go.unlink()
        print(f"go signal received after {time.time() - _t0:.1f} s of waiting")

    # Set by locate() when a refusal was about the LAYOUT rather than the frame; read
    # by the retry loop, which must not retry past it.
    _last_reason: dict[str, bool] = {}

    def locate(verbose=True, tag=""):
        """One attempt at locating the panel and every button. None if it failed.

        Returns ``(buttons3d, (origin, normal))`` in the arm base frame.
        """
        frame = cam.capture()
        # Read the arm pose immediately after the frame, with the arm stationary: for
        # an arm-mounted camera these two are one measurement, and any motion between
        # them corrupts the camera pose itself, not just the target's.
        base_T_cam = cam_pose()
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
            coarse = panel_roi(bgr, detector, shape=layout.shape,
                               verbose=verbose)
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
                # Classify the refusal: an alignment doubt is about WHICH BUTTON IS
                # WHICH and must not be retried away; everything else is about this
                # frame and may be.
                r = (rep.reason or "").lower()
                if "shift" in r or "anchor" in r:
                    _last_reason["alignment"] = True
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


    def path_obstructed(org_now, nrm_now, tag=""):
        """Is anything between the robot and the panel? None if clear, else a reason.

        THE CHEST CAMERA IS THE ONLY SENSOR THAT LOOKS WHERE THE ARM GOES. The base's
        own obstacle sensors face its direction of travel, which on this robot is
        **180 degrees away** from the panel — so `hasObstruction` structurally cannot
        see the arm's workspace, and it is False anyway whenever the base is parked
        (verified: an operator blocked the robot for 70 s and not one of the 35 fields
        in `robot_state` moved). There is no other source.

        The test is geometric, not learned: deproject the depth image into the base
        frame and ask whether anything sits IN FRONT of the fitted panel plane, on the
        robot's side. With the space empty that number is essentially zero — measured
        over 56,813 sampled pixels, the 99.9th percentile is +4 mm and the maximum
        +21 mm (the buttons' own protrusion), with **0.000%** past 30 mm, while the
        wall sits 127 mm BEHIND the plane. A person or a chair in the gap reads
        hundreds of mm. So the threshold has enormous margin either way.

        Called with the arm at HOME, where it is deliberately outside the camera's
        view, so the robot's own limbs are not what gets detected.

        LIMITS, stated rather than papered over: it sees only the camera's field of
        view, and it is a check BEFORE the motion — the arm is out for seconds
        afterwards and nothing here watches that window.
        """
        gd = cfg["elevator"].get("press", {}).get("obstacle", {})
        min_mm = float(gd.get("min_mm", 40.0))
        min_frac = float(gd.get("min_frac", 0.002))
        try:
            fr = obs_handle.camera.capture()
        except Exception as e:  # noqa: BLE001
            return f"could not capture to check the path ({e})"
        d = fr.depth[::4, ::4]
        K = fr.intrinsics
        ys, xs = np.mgrid[0:fr.depth.shape[0]:4, 0:fr.depth.shape[1]:4]
        m = (d > 0.15) & (d < 4.0)
        if m.sum() < 1000:
            return "depth image is mostly invalid — cannot verify the path is clear"
        z = d[m].astype(np.float64)
        x = (xs[m] - K.cx) / K.fx * z
        y = (ys[m] - K.cy) / K.fy * z
        T_obs = obs_cam_pose()
        pts = np.stack([x, y, z], 1) @ T_obs[:3, :3].T + T_obs[:3, 3]
        front = (pts - org_now) @ nrm_now * 1000.0
        frac = float((front > min_mm).mean())
        if frac > min_frac:
            return (f"{100 * frac:.1f}% of the view is more than {min_mm:.0f} mm in "
                    f"front of the panel (max {front.max():.0f} mm) — something is "
                    "between the robot and the panel")
        return None

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
        deadline = time.time() + args.auto_timeout
        stable, prev, last = 0, None, None
        misses = 0
        # When the run STARTS parked at the panel, the panel is already visible and
        # still, so arming immediately would press before the robot drove anywhere.
        # Require it to have gone away first: either not locatable, or displaced by
        # more than --arm-after-move.
        armed = args.arm_after_move <= 0
        first_c = None
        while time.time() < deadline:
            got = locate(verbose=False)
            if got is None:
                misses += 1
                if not armed:
                    armed = True
                    print("    panel left the view — armed")
                if stable:
                    print(f"    lost the panel after {stable} stable frame(s)")
                stable, prev = 0, None
                if misses % 10 == 0:
                    print(f"    waiting: panel not located ({misses} frames)")
                continue
            pts, pl = got
            c = centroid(pts)
            if first_c is None:
                first_c = c
            if not armed:
                moved = float(np.linalg.norm(c - first_c)) * 1000
                if moved > args.arm_after_move:
                    armed = True
                    print(f"    panel moved {moved:.0f} mm — armed")
                else:
                    prev, last = c, (pts, pl)
                    continue
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
        # An arm-mounted camera has to be AIMED before it can locate anything. The
        # panel and the dock are both fixed, so this pose is a registered constant
        # (see initialization/find_viewing_pose.py), not something to search for at
        # run time. A fixed camera ignores all of this and keeps localising from home.
        view = args.view_joints or cfg["arm"].get("view_joints_deg")
        if view is not None and handle.mount == "arm":
            view = [float(v) for v in (view.split(",") if isinstance(view, str) else view)]
            cur = list(arm.get_joint_angles())
            if max(abs(a - b) for a, b in zip(cur, view)) > 1.0:
                print(f"moving to the viewing pose {['%.1f' % v for v in view]}")
                if not arm.move_joints_sync(view):
                    print("!! could not reach the viewing pose")
                    return 1
                time.sleep(0.6)
        elif handle.mount == "arm":
            print("!! camera rides the arm but no viewing pose is registered "
                  "(arm.view_joints_deg or --view-joints). Locating from wherever the "
                  "arm happens to be, which is unlikely to see the panel.")

        buttons3d = plane = None
        # 15 attempts, not 6. Localisation succeeds on roughly a quarter of frames at a
        # re-docked distance (the detector drops a button or two and the lattice needs
        # 70 % of them), and three consecutive runs used attempts 4, 1 and 5 of 6 — one
        # bad frame away from failing outright. Each attempt costs ~200 ms, so the whole
        # budget is 3 s against a 60 s sequence.
        # A refusal about the FRAME may be retried; a refusal about the LAYOUT may not.
        # "only 6 of 10 buttons detected" is about this image and the next one may be
        # better. "a shifted alignment explains the anchors at least as well" is about
        # which button is which, and retrying it is rolling dice until one comes up.
        # Measured 2026-09-16 against a deliberately shifted layout: it was refused on
        # 4 frames and accepted on the 5th, so 15 retries turned a working safety check
        # into "refuse only if fifteen frames in a row refuse". An alignment contradicted
        # once in a run stays contradicted: later frames cannot un-see it.
        contradicted = False
        for attempt in range(15):
            got = locate(tag=f"  [{attempt + 1}/15] ")
            if _last_reason.get("alignment"):
                contradicted = True
            if got is not None:
                if contradicted:
                    print("!! an earlier frame in this run found the layout ambiguous, so "
                          "this acceptance is not trusted — refusing. Re-run once the view "
                          "is better, or fix the layout; do not retry past this.")
                    return 1
                buttons3d, plane = got
                break
        if buttons3d is None:
            print("!! could not locate the panel/buttons — is the arm blocking the "
                  "view, or is this a different panel than the registered layout?")
            return 1
    aim_mm = cfg["elevator"]["press"].get("aim_offset_mm")
    if args.aim_offset:
        aim_mm = [float(v) for v in args.aim_offset.split(",")]
    if aim_mm and any(abs(float(v)) > 1e-9 for v in aim_mm):
        off = np.array(aim_mm, dtype=float) / 1000.0
        buttons3d = {k: np.asarray(v, dtype=float) + off for k, v in buttons3d.items()}
        src = "--aim-offset" if args.aim_offset else "elevator.press.aim_offset_mm"
        print(f"aim offset applied: {[round(float(v), 1) for v in aim_mm]} mm in the base "
              f"frame (from {src}) — this is a recorded bias patch, not a calibration")

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
    # Where the torso actually is right now. `L0` is where it was when the buttons
    # were localised and stays the reference for compensating their coordinates; the
    # lift is NOT returned to it between buttons, so "how far would this move the
    # torso" has to be measured from here, not from L0.
    L_now = L0
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
            cmd, rise, best, reasons, tried = plan_over_lift(button, seed, L_now)
            if best is None:
                print(f"    no pose satisfies the boundaries at ANY of {tried} lift "
                      "heights:")
                for why, n in sorted((reasons or {}).items(), key=lambda kv: -kv[1]):
                    print(f"      {n:>2}/24 rolls  {why}")
                results.append((name, False))
                continue
            if abs(cmd - L_now) < 1.0:
                print(f"    lift stays at {L_now:.0f} (margin "
                      f"{best[3]:.1f} deg)")
            else:
                print(f"    lift {L_now:.0f} -> {cmd:.0f} "
                      f"(body {(cmd - L_now) / lift_cfg['command_per_mm']:+.0f} mm, "
                      f"{rise * 1000:+.0f} mm from where it was localised), "
                      f"margin {best[3]:.1f} deg, chosen from {tried} heights")
                if args.go and move_lift(cmd) is None:
                    print("    lift move did not confirm; skipping")
                    results.append((name, False))
                    continue
                if args.go:
                    L_now = cmd
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

        # Last thing before the arm moves. The arm is at home here, i.e. outside the
        # camera's view, so what this sees is the world and not the robot.
        # --no-obstacle-check does not merely tolerate a missing camera; it SKIPS the
        # check. Running it with a camera that cannot see the panel is worse than not
        # running it: the answer is meaningless either way, but a "clear" verdict
        # reads like protection that is not there.
        blocked = None if args.no_obstacle_check else path_obstructed(origin_now, normal)
        if args.no_obstacle_check:
            print("    path check SKIPPED (--no-obstacle-check): nothing is watching the "
                  "space between the arm and the panel")
        if blocked is not None:
            print(f"    REFUSING to move: {blocked}")
            results.append((name, False))
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
        if args.hold > 0:
            print(f"    holding at contact for {args.hold:.0f} s — photograph the tip now",
                  flush=True)
            time.sleep(args.hold)
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
