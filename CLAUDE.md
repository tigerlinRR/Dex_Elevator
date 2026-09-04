# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

**Current build status, results, and next steps: @PROGRESS.md** (auto-imported — read it first to see where we are).

## What this is

Vision-guided pipeline for the Richtech **DEX** robot to **press elevator buttons**
and ride ordinary elevators between floors. A chest **Orbbec Gemini 335** detects
the buttons (YOLO), the button pixel is turned into a 3D contact point on the panel
via an **eye-to-hand** calibration (`base_T_camera` to the right arm), and the
**RealMan right arm** presses it (standoff → press → retract). US regulations don't
allow commanding the elevator's floor dispatch, so physical pressing is the task.

The mobile base docks **roughly** in front of the panel; the camera + arm + torso
lift absorb the docking error, so the base doesn't move during a press.

## Environment / hardware (important, non-obvious)

The robot is an NVIDIA **Jetson AGX Orin Developer Kit** (JetPack 6.2 / R36.4.4, CUDA
12.6, 61 GB RAM), reachable at **`ssh dex5-wired`** (`192.168.11.31`, user `jetson`).
An older **Jetson Thor** unit (`dex4`, `192.168.11.41`) is retired — ignore it, and do
not reuse its calibration artifacts. Hardware, verified on the device — do NOT assume an xArm:

- **Arms: dual RealMan RM 65** (SDK imports as `Robotic_Arm`, class `RoboticArm`).
  LEFT `192.168.11.32`, RIGHT `192.168.11.33`, port `8080`. **Pressing uses the
  right arm.** Left/right **VERIFIED 2026-08-14**: hand-pushing the right arm while
  polling both moved `.33` by 34.21° and `.32` by 0.01°.
- **Python is the SYSTEM `python3` (3.10.12), not conda.** It has **numpy 1.26.4** and
  **cv2 4.8.0 with `aruco.CharucoDetector` + `calibrateHandEye`**, and pyyaml. Everything
  else is `pip install --user`: `Robotic_Arm` 1.1.6, `pyorbbecsdk` 2.1.2 (built from
  source in `~/pyorbbecsdk`), `cuda-python` 12.6.2, and this repo (`pip install --user -e .`).
  This machine's `richtech-v3` env is NOT usable here (no RealMan/Orbbec SDK, CPU-only
  torch) — leave it alone.
- **The robot is FULLY STANDALONE.** Camera, arm, hand, pressing and GPU button
  detection all run on the Orin with nothing on the dev Mac involved. The Mac is only
  a terminal (and the only route to it is the Ethernet cable — see the network note).
- **YOLO inference is TensorRT, not torch** (`yolo/trt_detector.py`). `torch` IS present
  (2.10.0, ultralytics 8.4.121) but it is the generic aarch64 **CPU-only** wheel, so
  Ultralytics gets 491 ms/frame. **Do not swap in NVIDIA's JetPack torch**: that install
  lives in the shared `~/.local` and `~/mmdetection` imports it. JetPack's own TensorRT
  10.3 (+ `cuda-python`) reaches 43 ms/frame end-to-end without touching torch at all.
- **`onnx` lives OUTSIDE the package path on purpose** — unpacked into
  `~/Dex_Elevator/.pydeps-onnx` and put on `PYTHONPATH` only for the one-off ONNX export.
  It needs protobuf >= 3.20 while the system has 3.12.4 in `/usr/lib/python3/dist-packages`,
  and installing a newer protobuf into `~/.local` would shadow it for **every** project on
  this shared machine. Verified isolated: from a clean cwd, and inside `~/mmdetection`,
  `protobuf` still reports 3.12.4.
- **This is a SHARED machine.** The home dir holds unrelated projects (ZED, VR teleop,
  LinkerHand gripper endurance). Touch only `~/Dex_Elevator`, `~/pyorbbecsdk`, `~/.local`.
- **The Jetson HAS internet, and is reachable without the cable — both were wrong in earlier
  notes.** Re-verified 2026-08-21:
  - Its WiFi is associated with `Brother HL-L3275` (`192.222.10.133/24`) — an access point
    named after a printer, which nonetheless **routes to the internet** (`curl https://pypi.org`
    returns 200, and it is the default route at metric 600, beating the dead wired gateway
    `192.168.11.1` at 20100). So **`pip install` works directly on the robot**; the old
    "download on the dev Mac and rsync it over" workaround is obsolete.
  - **Tailscale is up and has been all along** (`100.122.187.11`, node `ubuntu`, key valid to
    2026-12-27). The "offline, last seen 199d" lines in `tailscale status` belong to OTHER
    people's nodes — misreading them is what produced the "Tailscale is offline" claim.
  - Two SSH routes, both in the Mac's `~/.ssh/config`: **`dex5-wired`** (cable, 54 MB/s) and
    **`dex5-ts`** (Tailscale, 6 MB/s, ~110 ms, works from any network). Verified cable-free by
    pressing two buttons with `en0` down. Prefer the cable for bulk rsync and `--web` preview.
  - The `192.168.10.146` / `Richtech_Tech` address in older notes is **stale**. The robot also
    scans only that one SSID and `nmcli dev wifi rescan` returns `not authorized`.
  - Careful reading routes on the Mac: its `en0` is `192.168.11.50` with a **/16** netmask,
    so `route get 192.168.10.x` cheerfully answers "via en0" for addresses that are not
    reachable at all. Test with an actual connection, not a route lookup — and for Tailscale
    check the peer's actual endpoint (`tailscale ping`), since it will happily use the LAN
    path when the cable IS plugged in and then look cable-independent when it is not.
- **`sudo` requires a password**, so any root step has to be handed to the user.
- **The arm and the chest camera face 180 degrees away from the base's front.** The
  base drives with its own front (green light, obstacle sensors, face camera) pointing
  directly AWAY from the panel the arm reaches for. Two consequences that are easy to
  get wrong: the base's `hasObstruction` can never see the arm's workspace — it watches
  the opposite hemisphere — and blocking one side is invisible to the other. Verified
  by blocking each in turn and watching the other report nothing.
- **Torso lift column** (RealMan lift API, mm) raises/lowers the upper body. The
  chest camera and arm bases ride it together, so `base_T_camera` stays constant.
- **LinkerHand** dexterous hand as end-effector (driven by its own bridge, not the
  RealMan gripper) — pressing uses a fixed "pointing" hand pose set out-of-band.
- **Two Orbbec** cameras: **Gemini 335** (chest, button camera) + **335L** (head,
  scene). Distinguish them by device name suffix ("335" vs "335L") or serial.
  - **The head 335L is aimed steeply DOWN** (verified 2026-08-21 by looking at a frame): it
    sees the robot's own two arms, the base deck, and the near floor — NOT the elevator door
    or people at standing height. It catches only the bottom 4 of the 10 buttons, very
    obliquely, and has no calibration files. The roughly level, forward-facing camera is the
    CHEST one. Conclusion: the head unit earns nothing as mounted; leave it unused rather than
    inventing a job for it, and revisit only if "is the door open / is the car full" turns out
    to need a high forward view. Re-aiming it would cost a mechanical change plus a fresh
    hand-eye calibration.
  - **Its auto-exposure needs ~30 frames to settle** (46 -> 121/255, plateauing at frame ~32).
    A single `capture()` right after `start()` is 3x too dark. `cam_chest` never showed this
    because its exposure is locked.
- SDKs (`Robotic_Arm`, `pyorbbecsdk`, `ultralytics`) are **optional at import
  time** — every module guards the import and degrades, so the repo imports on a
  laptop and hardware code runs on the robot.

## Common commands

Run hardware commands **on the robot** (`ssh dex5-wired`) with the plain system
`python3` — there is no env to activate.

```bash
# Readiness self-test FIRST: libs, configs, board, camera capture, arm pose (read-only).
python3 initialization/bringup_check.py

# list connected Orbbec devices (name + serial) — confirms 335 vs 335L
python3 -m core.camera.orbbec

# Calibration (once, per camera, in order): intrinsics -> extrinsics -> validate.
# The robot is HEADLESS: add --web to stream an MJPEG preview to a browser on the
# LAN (http://192.168.11.31:8010/); capture via the page buttons or keys c/d.
python3 initialization/calibrate_intrinsics.py --camera cam_chest --web   # 1. intrinsics (hold board)
python3 initialization/run_calibration.py      --camera cam_chest --web   # 2. eye-to-hand (board on flange, drag-teach)
python3 initialization/validate_calibration.py --camera cam_chest         # 3. re-validate saved artifacts
python3 initialization/eval_localization.py    --camera cam_chest --web   # end-to-end localization accuracy (fresh poses)

# No test suite or linter is configured yet.
```

**Stream long robot commands; do not wait for them to exit.** `press_stream.sh` and
`elevator_runner`'s `liverun_stream.sh` run the job and `tail -f` its log. Waiting for
the process instead means the whole 40-70 s press happens with nothing on the terminal,
and the waiting command then times out and goes to the background, adding another layer
on top — measured, that turned a 2 s view of the robot into a 2 min 40 s blind spot.
SSH itself was never the problem: 108 ms RTT over Tailscale, and 0.16 s per call once
`ControlMaster` reuses the connection (0.6 s without). The floor is ~2 s, because the
runner's own data comes from the cloud and the base reports event-driven.

**The `--web` preview only works if the browser can route to the Jetson.** The dev Mac is
on `192.168.40.x` and cannot reach the Jetson's WiFi subnet (`192.168.10.x`), so today
calibration needs the **Ethernet cable** between Mac and robot (Mac becomes `192.168.11.50`).

Button detector (YOLO) — train on the Orin's GPU in the isolated venv
`~/venv-dex-train` (created with `venv --without-pip` + `get-pip.py`, because Ubuntu
strips `ensurepip` and installing it needs sudo). It holds **torch 2.8.0 + torchvision
0.23.0 CUDA builds** from `https://pypi.jetson-ai-lab.io/jp6/cu126` — note the `.io`
domain, `.dev` is unreachable — plus ultralytics installed `--no-deps`. **Do not use
torch 2.10 there**: its CUDA build needs `libcudss.so.0`, which JetPack does not ship.
Nothing in `~/.local` is touched, so the CPU torch that `~/mmdetection` imports is
unaffected. Run training with `PYTHONPATH=~/Dex_Elevator` (the venv has no editable
install of this repo).

```bash
PYU=~/miniconda3/envs/ultralytics/bin/python
PYTHONPATH=~/Dex_Elevator $PYU yolo/prepare_dataset.py --src "<roboflow_export_dir>"        # multi-class -> data/datasets/buttons/
PYTHONPATH=~/Dex_Elevator $PYU yolo/train_buttons.py --model yolo11m.pt --imgsz 640         # -> data/weights/buttons.pt
PYTHONPATH=~/Dex_Elevator $PYU yolo/predict_server.py --port 8011                           # browser check: http://<robot-ip>:8011/
```

## Architecture

**Interface-first.** Each stage depends on an abstract base so hardware is
swappable and the orchestration is testable without it: `core/camera/base.py`
(`Camera`), `core/robot/base.py` (`RobotArm`). `core/robot/sim.py` (`SimRobot`)
records moves and never touches hardware, for headless runs/tests.

**Data types flow through `core/types.py`** (plain dataclasses + numpy):
`CameraFrame` (RGB uint8 + metric depth + `CameraIntrinsics`), `Detection`.
**Convention: `CameraFrame.rgb` is RGB** (not BGR); depth is meters.

**SE3 / geometry lives only in `core/transforms.py`** (numpy-only): `make_transform`,
`invert`, `transform_point(s)`, `deproject_pixel`, `rpy_to_matrix`/`matrix_to_rpy`,
`pose_from_rpy`, and **`quat_to_matrix`/`matrix_to_quat`** (used to convert RealMan
poses without hard-coding an Euler order). Keep the calibration package depending
only on these (no scipy, no vendored helpers).

**Press geometry is pure geometry** (`core/press.py`, pure functions — no learned
model). A button is a point target on the known **vertical panel plane**:
- `press_target_from_pixel`: back-project the button pixel to a ray in the base
  frame and intersect it with the panel plane → 3D contact point (depth-independent
  ray∩plane; robust to small/low-contrast buttons where depth is noisy). Approach
  axis (tool +Z) = inward panel normal.
- `press_waypoints`: STANDOFF → PRESS (advance `push_depth` past the face) → RETRACT.
- `fit_panel_plane_from_depth`: **measures the panel plane LIVE** from the depth ROI
  (RANSAC + least-squares refit, ~5 ms, repeats to ~1 mm). **Always fit; never trust
  the plane in the config** — the base docks with centimetres of error, so a stored
  plane is wrong the moment it stops anywhere else. Pass an ROI covering the
  faceplate only: the wall behind is a second, parallel plane, and **anything else in
  the ROI (notably the robot's own hand) drags the fit** — measured 23 mm of error
  with the hand in frame. Once YOLO is wired, use the button detections' bounding
  box as the ROI, which excludes the hand automatically.

**Nothing may be between the robot and the panel when the arm moves**
(`path_obstructed` in `press_buttons.py`, `elevator.press.obstacle`, 2026-08-28).
Checked from a fresh depth frame immediately before EVERY button's motion, with the arm
at home — which is deliberately outside the camera's view, so what it sees is the world
and not the robot's own limbs.

**The chest camera is the only sensor that looks where the arm goes.** The base's
obstacle sensors face its direction of travel, 180 degrees away from the panel, and
`hasObstruction` is False whenever the base is parked in any case — verified by
blocking the robot for 70 s and watching all 35 scalar fields of `robot_state`, none of
which moved. A guard built on `hasObstruction` was written first and believed for an
hour before that was measured; it could never have worked.

The test is geometric, not learned: deproject the depth image into the base frame and
ask what sits IN FRONT of the fitted panel plane, on the robot's side. Empty, that is
essentially nothing — over 56,813 sampled pixels the 99.9th percentile is **+4 mm** and
the maximum **+21 mm** (the buttons' own protrusion), with **0.000%** past 30 mm, while
the wall behind sits **127 mm** the other way. A hand in the gap measured **197 mm**
and 20.8 % of the view. So the 40 mm threshold sits in a wide gap between the two
cases rather than on a tuned edge. Verified on hardware: button `1` pressed normally,
a hand went in, button `4` was REFUSED with IK, clearance and self-collision all
passing — the arm did not move.

Two limits, stated rather than papered over: it sees only the camera's field of view,
and it is a check BEFORE the motion — the arm is out for seconds afterwards and nothing
watches that window.

**Dexterous hand (`core/hand/linkerhand.py`)** — LinkerHand O6 as a Modbus RTU slave on
the RIGHT arm's tool-side RS485, sharing the arm's connection. Bring-up:
`rm_set_tool_voltage(3)` (24 V) then `rm_set_modbus_mode(1, 115200, timeout)`.
Registers: read 0..5 = current joint positions, **write 0..5 = TARGET positions**
(same addresses, different meaning), 12..17 = speed; 0 = closed, 255 = open.
`POSES["point"]` (index extended, rest curled) is the pressing posture. `probe()`
sweeps ports/slaves read-only to locate the device — verified `port=1, slave=0x27`.
The fingertip TCP is `hand.fingertip_offset` in `configs/pipeline.yaml`, derived from
LinkerBot's official O6 URDF + STL meshes (github.com/linker-bot/linkerhand-urdf).

**Choosing a press pose is constrained, not scripted** (`plan()` in
`initialization/press_buttons.py`, thresholds in `arm.limits`). The approach roll is
still searched over all 24 directions; what is fixed is which poses are DISALLOWED.
Measured against the controller's real limits (J1 +-178, J2 +-130, **J3 +-135**,
J4 +-178, J5 +-128, J6 +-360 — read, not guessed):
- The old objective, least joint travel, was picking poses sitting ON a hard stop:
  **0.1 deg** of J3 margin for button `2`, 0.7 deg for `4`. J3 binds for every button
  on this panel, and the controller's own self-collision check is **off**, so "IK
  returned a solution" is not "safe to execute".
- Maximising margin instead swings the path INTO the panel (measured clearance
  -12.0 mm for `dot`, -3.4 mm for `A`). So: reject the unsafe, then take the largest
  margin among survivors, breaking ties on travel. Margins went 0.7 -> 3.6 deg for
  `4`, 4.1 -> 9.2 for `6`, 3.9 -> 7.6 for `3`.
- **The contact pose is now IK-checked too.** It never was: only the standoff was
  solved, and the press itself was a Cartesian `movel` 50 mm further in. Driving a
  straight line into a pose with no good solution is exactly where the arm jams.
- Button `2` is now REFUSED from the current lift height, because its best achievable
  margin is 1.0 deg. It used to "work" on 0.1 deg of margin, i.e. by luck. The fix is
  the lift (below), not a looser threshold.

**Panel height is handled by the lift, panel-independently** (`arm.lift`). What
governs reachability is the button's height in the base frame, and the lift moves that
frame — so `required_lift = command_min + command_per_mm * (z_measured - target_z)`
works on any elevator instead of a per-panel constant. Three measured facts that all
look different from what they are:
- **The lift is on the LEFT controller.** The right one answers `rm_get_lift_state`
  with `pos = 0`, which reads exactly like "fully down".
- **Reported position is 2x the real travel** (ratio 0.5016 / 0.5021 / 0.5008 over
  three different moves, checked against the camera). Command and readback agree to
  1 mm and `err_flag` stays clean, so nothing warns you — only measuring the geometry
  reveals that the body rose half as far.
- **The blocking form of `rm_set_lift_height` hangs forever past the travel limit**
  (1100 is fine, 1200 never returns). Use the non-blocking form and poll for
  `mode != 2`.
Repeatability is 0.09 mm at command 444 and 0.24 mm at 600 (spread of the panel's
measured z over four cycles), which is what makes **look low, press high** viable: the
camera rides the lift, so past ~175 mm of real rise the bottom button row leaves the
frame, while the arm wants another 200 mm on top of that.

**The height is chosen per button by `arm.lift.objective`** (2026-08-27). `still` keeps
the torso where it is whenever the current height already clears `prefer_margin_deg`;
**`margin` (the default now) moves to the height with the BEST joint margin for that
button**, so the body visibly tracks the panel row by row. That was asked for as a
presentation choice — a robot that stands perfectly still while only its arm works
looks broken — but it is not merely cosmetic: the heights it picks carry the largest
margins on offer rather than the first sufficient one. Both objectives choose only
from poses that already passed the joint-limit, wrist, self-collision and
panel-clearance checks, so this trades optimality and time, never safety. Two buttons
on the same row share a best height, so `min_move_command` (100 units = 50 mm of body
travel) makes the second one take the next-best SAFE height instead, purely so there
is motion to see. Measured: `1 4 2 5` went 51.2 s -> **71.2 s**, still 4/4 lit,
depth -3.10 mm, lateral 0.36 mm (0.08 mm when the torso did not move — worth watching,
since every lift move re-compensates the cached coordinates).

**`press_buttons.py --lift` implements that** (contributed 2026-08-26): detect once at
the current visible height, then per button move the lift, compensate the cached 3D
coord by the achieved rise, press, and restore the lift at the end. **The height is
SEARCHED, not computed** — the first version computed it from a fixed
`target_relative_z` measured at ONE docking distance, and re-docked 12 cm further out
that constant selected the worst heights available: button `1` got 12 of 24 rolls at
the computed height and 0 of 24 one step up, while every height from 444 to 944 gave
24/24 with 41-56 deg of margin. Reachability depends on distance AND height; only the
height axis had been measured. Searching needs no such map, and it prefers NOT to move
(`prefer_margin_deg`), so at a comfortable docking distance the torso stays still and
the sequence runs ~11 s faster. Per-button targeting beats one height for the whole panel:
button `2`, which the boundary check refuses outright at the viewing height (best
margin 1.0 deg), plans at **52.8 deg of margin with all 24 rolls passing** — and it
lights. The plane origin must be compensated along with the button (`origin_now`),
since the panel rides the lift too; getting that wrong understates clearance by
`rise * normal_z`, measured at 4.7-6.4 mm.

**Self-collision is checked at planning time** via
`rm_algo_safety_robot_self_collision_detection` (also contributed 2026-08-26) — pure
computation against the controller's own model, sampled along the whole
seed -> standoff -> contact path, because a configuration can pass through a collision
between two clear endpoints. This matters because the controller's *runtime*
self-collision check is **off**. It covers the arm's own links and end-effector only:
the other arm, the chassis and the door frame still need virtual walls.

**Autonomous pressing** (`initialization/press_buttons.py`) — the executable that
ties everything together: `python3 initialization/press_buttons.py 1 4 2 5 --go`.
Per button: home -> `movej` to standoff -> `movel` through contact into the button
-> `movel` retract -> `movej` home. Verified 2026-08-20: **4/4 buttons lit**, depth
error <=0.05 mm, lateral <=0.28 mm, ~48 s for four. Two structural decisions:
- **Button 3D coordinates are located once per sequence and cached.** The base and
  panel don't move mid-sequence, so they're constants; re-detecting per button only
  adds failure chances (an earlier version lost 2 of 3 buttons to intermittent
  circle-detection misses).
- **`standoff` cannot exceed ~50 mm.** Backing further off moves the target TOWARD
  the arm base and into its inner unreachable region: 0/30/50 mm are 24/24
  reachable, 80 mm+ is 0/24. So "retreat far, then approach" is not an option here.

**Where the buttons are, and which is which** (`yolo/panel_layout.py` +
`configs/panels.yaml`) — the default path since 2026-08-25. Positions come from the
detector; the floor label for each position comes from that elevator's **registered
layout**; and the buttons the model *does* read confidently anchor the grid so a
misalignment is caught rather than pressed. Measured, and each number changed the design:
- **Positions are reliable, labels are not.** 10/10 buttons found on the panel crop at
  imgsz 640 (centre error 0.3-6.3 px, matching Hough), but only 5 of 8 markings read
  correctly. So positions are trusted and labels are not.
- **Run detection on a CROP, not the full frame.** Full-frame at conf 0.25 misses one
  button of ten; the crop finds all ten with one spurious box.
- **A whole-panel pass classifies far worse than per-button crops** (`3` reads `empty`
  0.46 in the panel pass, `3` 0.48 alone), so anchors are re-classified from their own
  crops. `classify_solo`'s `pad_scale` is NOT a free parameter — 44 px crops score
  1/10, 68 px 3/10, 104 px 4/10; 1.6 button widths made every anchor mismatch and
  looked exactly like a scrambled grid. 2.1 reproduces the best case.
- **The ROI is derived, not hard-coded** (`panel_roi`): the union of ALL detections is
  too crude, because a low-confidence pass also fires on the cabinet's keyhole and
  stretched the ROI 145 px past the faceplate. Single-linkage clustering on the
  centres, keeping the largest cluster, separates the grid from those strays. The link
  distance is **2.2** button widths, not 3.0: the keyhole sits ~2.25 widths from the
  nearest button, so at 3.0 it merged on some frames and not others — and when it
  merged the ROI stretched 100 px, the refined pass returned 11 "buttons", and the
  lattice fit failed. Neighbouring buttons are only ~1.6 widths apart. `assign` also
  drops the detection furthest from the group and retries when the fit fails, rather
  than discarding the frame. Together these took localisation from 1 of 4 frames to
  7 of 8. Then
  `tight_roi` re-boxes the fit around the confirmed buttons only — the wall and cabinet
  behind are separate parallel planes and bias the depth fit.
- **The ROI must be able to look where the first pass MISSED, or the failure is
  self-sealing** (`panel_roi(shape=...)`, 2026-08-28). The full-frame pass scales
  1280x720 into the model's 640, so a 44 px button becomes ~22 px and a whole edge ROW
  can drop out: measured live, `open`/`close` scored **0.16 and 0.06** full-frame while
  the same two buttons scored **0.91 and 0.96** on a crop of that row alone. Since the
  ROI is derived FROM those detections, a missed bottom row put the ROI's bottom edge
  above it, the refined pass never looked there, and the lattice fit got four rows for
  a five-row layout — residual 11-12.5 px against a 6 px threshold, **refused 15 of 15
  attempts with the panel plainly in view**. Fix: when fewer rows (or columns) are
  found than the layout registers, grow the ROI by the MISSING count times the measured
  pitch, both ways along that axis. Bounded by what is missing rather than a bigger
  blanket margin — a complete grid grows by nothing, so working frames are unchanged,
  and it cannot run away. On the failing frame it took the bottom edge 410 -> 463 px
  (still 81 px clear of the cabinet keyhole at 544 that `link` exists to exclude) and
  the residual 11.3 -> **1.7 px**. Note the first hypothesis was WRONG and measuring
  killed it: the scene was blown out, but the missed row's mean/std were 192.8/47.9
  against 190.4/49.1 for a row that WAS found — brightness was not the variable.
- **Verification is a LATTICE FIT, not a row-shape match** (`fit_lattice`). Requiring
  the detected grid to equal the registered shape exactly threw away good frames: at a
  re-docked distance 12 cm further out the buttons shrank from 50 to 44 px and one or
  two per frame go missing, so the shape check refused 6 of 6 attempts while all ten
  buttons were plainly visible. Fitting `pixel(i,j) = origin + i*row + j*col` fills the
  gaps AND catches a shift, because rows/cols come from the registered layout: a
  missing row cannot be fitted without the residual blowing up. Measured residual on a
  good frame is ~1.5 px.
- **Anchors are a RELATIVE test, not an absolute one.** They used to require "2 of 4
  read correctly", and that refused a perfectly correct grid once the classifier
  degraded at the greater distance — a false refusal whose consequence is the robot
  standing in the lobby unable to act. Now the unshifted alignment merely has to score
  better than every shifted alternative, which keeps the protection (a shift breaks all
  anchors at once) without depending on absolute classifier confidence.
- **Refuse, don't guess.** A shifted grid yields perfectly plausible coordinates, so a
  wrong label is a SILENT wrong-floor press. Verified by feeding a deliberately
  inverted layout: anchors went 0/4 and the press was refused.

**Button centres via Hough circles** (`yolo/button_circles.py`) — the previous
stand-in, kept behind `press_buttons.py --circles` as a fallback and A/B reference.
Head to head on 20 live frames while the showroom light drifted the faceplate to 223
(blown out): **Hough 1/20, detector 12/20**. At the designed brightness both work and
the press results are indistinguishable (lateral 0.17/0.20 mm vs 0.06/0.23 mm), so the
reason to switch is not accuracy — it is labels, the derived ROI, and light tolerance.
Detector localisation repeats to **±0.3 px**; it costs 201 ms per attempt against
Hough's 41 ms, which is 1 % of a 23 s two-button sequence.

**Button detection** (`yolo/button_detector.py`): `ButtonDetector` (Ultralytics
YOLO, single class `button`) → `Detection` list; `centroid_pixel` gives the press
pixel; `read_floor_label` (**stub**) is the "which floor" reader — the real open
problem, deliberately decoupled from the geometry so it can be swapped freely.

**GPU button detection** (`yolo/trt_detector.py`) — `TrtButtonDetector` returns the same
`Detection` list as `ButtonDetector`, but runs the model through a **TensorRT FP16 engine**
instead of Ultralytics, because this Orin's torch is CPU-only (see the environment note).
Measured: **43 ms/frame end-to-end (23 FPS)** vs 491 ms on CPU; the GPU part alone is 6.4 ms.
That is fast enough for continuous perception (tracking the panel while the base moves,
visual servoing), so being restricted to a parked base is no longer a performance limit.
Building the engine is a two-step, **per-machine** offline job — engines are tuned for the
specific GPU and are NOT portable, and it takes ~10 min:

```bash
PYTHONPATH=~/Dex_Elevator/.pydeps-onnx python3 -c \
  'from ultralytics import YOLO; YOLO("data/weights/buttons.pt").export(
       format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)'
/usr/src/tensorrt/bin/trtexec --onnx=data/weights/buttons.onnx \
    --saveEngine=data/weights/buttons_fp16.engine --fp16
```

Class names do not survive into the engine — `export_names()` writes them to
`data/weights/buttons_names.json` alongside it.

**Button-YOLO training tooling** (`yolo/prepare_dataset.py`, `yolo/train_buttons.py`,
`yolo/buttons.yaml`): the detector is trained from **public** datasets — five of them, merged.
**Multi-class by default**: each floor symbol keeps its own label so the model **detects AND
identifies** which floor (`--collapse` is an opt-in single-`button` mode). `train_buttons.py`
trains YOLO11 → `data/weights/buttons.pt` — **use `--model yolo11m.pt`**; nano is far too weak
for this many classes. Attribution in `DATASETS.md`.

Merging several sources needed two fixes, both of which failed *quietly* before:
- **Sources are merged by class NAME, never by class id.** Public sets do not agree on what to
  call a button — floor 3 is `3`, `three` and `button-3` across the three taxonomies we have —
  so an id-based merge relabels every box. `CLASS_ALIASES` canonicalises names and `CLASS_DROP`
  removes classes whose meaning is ambiguous (`floor-1` next to `button-1`) or which are not
  buttons at all (`closed-door` is a door STATE). Drops are counted and printed; a silent drop
  is indistinguishable from "that class was never in the data".
- **The val split is carved on a FRACTION, not on absence.** The old rule ("carve only if no
  source has a val split") was defeated by sources shipping 5 and 10 val images: the merge kept
  **194 val images for 369 classes** — under one example per class — so every mAP would have
  been noise that still looked like a measurement. Now anything below half of `--val-frac` is
  topped up (194 → 1076).

Licence discipline matters here because the goal is a product: only CC BY 4.0 and MIT sources
are used (both permit commercial use with attribution); NC or unlicensed data is worse than no
data. The largest academic set (CUHK, 3,718 images / 35,100 labels, arXiv 2103.09030) is **not**
used — its download link is dead and it states no licence.

**Driving the base LOCALLY, in a straight line** (`initialization/drive_straight.py`,
2026-09-01) — the chassis has its own LAN API, independent of AutoXing's cloud:

```
POST http://192.168.25.25:8090/services/wheel_control/set_control_mode
     {"control_mode":"remote","mode":"remote"}
ws://192.168.25.25:8090/ws/v2/topics
     -> {"enable_topic":"/tracked_pose"}
     -> {"topic":"/twist","linear_velocity":v,"angular_velocity":0}
     <- /tracked_pose{pos,ori}, /scan_matched_points2, /slam/state, /wheel_state,
        /battery_state, /alerts, /planning_state
```

**Why it exists**: the cloud API's only abstraction is "go to this point", and its
planner decides how — it prefers to turn. Entering and leaving an elevator has to be
straight in and straight out to make the door window, and a planner that happened not
to turn this time is not a guarantee. Here the angular velocity is pinned to 0 with no
planner in the loop, so straight is structural. Measured `+-2 m`: **1.5-2.9 cm of
distance error, 9-43 mm lateral, under 1.2 deg of heading change**.

**The agreed elevator flow** (operator's decision, 2026-09-02). Orientation is named
with the ARM and CHEST CAMERA as the robot's front; the base's own front (green light,
obstacle sensors, face camera) is its back. Outside the car the robot stands with its
BACK to the doors, so it **reverses straight in** and ends up facing the doors from
inside — which is also where the car's own button panel is. It then **turns once**,
presses, turns back, and **drives forward straight out**. Two constraints fall out of
this and both are hard:
- **At most one turn inside the car, and it must be quick.** The car may already be
  travelling for someone else, so a floor pressed late is a floor missed. Only the
  straight legs are inside the door's open window; the turn and press are not, but they
  are inside the car's own travel time.
- **Lateral position must be set BEFORE entering.** It cannot be corrected inside
  without more turns, and 5.4 cm of it costs almost all the approach-roll redundancy
  (see PROGRESS 2026-09-02). Outside the door the time is free, so that is where the
  entry line's lateral placement belongs — from the lidar, which can see into an open
  car, since the chest camera faces away from the doors until the robot is inside.

**In-place TURNS are a separate primitive** (`drive_straight.py turn`, 2026-09-02).
`move` sends angular 0 and `turn` sends linear 0, in two different functions, so no
caller can emit an arc and "straight is structural" survives: a path through a doorway
stays a pure translation and any re-aiming is its own gated, measured leg. Turns exist
because the flow needs them — outside the door the pose that lets the arm press the
call panel is not the pose that backs into the car, and inside, a panel that is not
square to the robot cannot be reached by translating.
- **The swept radius is 0.476 m**, the furthest of the chassis's own 19 `/robot/footprint`
  points from the pose origin. The 0.80 x 0.76 m bounding box would have said 0.55 m and
  refused turns in gaps that fit. The gate is nearest-return-in-ANY-direction, since the
  corridor test that gates a straight leg says nothing about a turn.
- **Measured achieved/commanded**: 0.688 at 10 deg, 0.874 at 23, 0.880 at 28, **0.957 and
  0.963 at 90** — the same fixed-loss-plus-proportional shape as the straight legs, so
  short turns lose most. Position drift 0-6 mm, i.e. it really does turn in place.
  `TURN_RATIO` is still 1.0 in the file: the correction legs absorb the difference and
  every run prints the ratio it achieved.
- **Right turn = NEGATIVE degrees = ori decreasing**, verified against the camera: a
  commanded -10 deg moved the panel's bearing +31.5 -> +23.5.
- A whole re-position — turn -90, translate 0.265 m, turn +90 — came back to within
  **0.57 deg** of the original heading, which is the pose feed's own quantisation.

**Legs are driven on the INTEGRAL of what was commanded, and the ramp is capped**
(2026-09-04). Both loops used to run for `target / speed` seconds; both also ramp down
over the last stretch, and that was not in the sum, so every leg lost the ramp's own
deficit — a FIXED loss, which is why the turn "ratio" looked like it wandered between
0.69 and 0.96 (it fitted `achieved = 0.969 * commanded - 2.8 deg`). Worse, a leg shorter
than the ramp spent its whole duration inside it, below the wheels' breakaway. Now each
loop accumulates the velocity it sends and stops when that integral reaches the target,
and the ramp is capped at 40 % of the leg. Straight legs measured after: 1.00 m -> 0.988,
0.40 -> 0.390, 0.15 -> 0.152, heading 0.00 deg.

**~3 degrees is this chassis's angular resolution — treat it as a constraint.** With the
ramp deficit gone, single turns of 3-12 deg still came back 2-4 deg off, and commands
below ~3 deg often moved the base NOT AT ALL (0.0 deg three times running, then a
break-away to 2.9). It is the plant, not the measurement: `ori` held its value for 20 s
after a turn and the feed quantises at 0.573 deg. So `turn` closes the loop and DAMPS each
correction to 60 % of what remains, stopping at a 3 deg floor. Consequence worth stating
before designing anything on top: 3 deg over the ~1.2 m it takes to clear a doorway is
~6 cm of lateral drift, which is the entire margin of a 0.9 m door against a 0.8 m robot.

**The chassis rotation centre is 0.281 m BEHIND the arm base origin** (arm-frame
(-0.281, +0.030), solved from one measured turn). So a turn changes the panel's RANGE as
well as its bearing — which is what makes "one straight leg plus one turn" able to satisfy
both, and why a turn is not a pure re-aim.

**The chassis has a LOCAL navigation API, not just twist** (`/chassis/moves`, found
2026-09-04). This corrects an assumption the straight-line driver was built under — that
anything beyond a straight line had to go through AutoXing's cloud. It does not:

    GET  /chassis/               -> Move Actions, Current Map, Pose, Status
    GET  /chassis/moves          -> history; each has type, target pose, state
    POST /chassis/moves          -> {"creator","type","target_x","target_y","target_ori"}
    GET  /chassis/moves/current  -> 404 when nothing is running
    GET  /services               -> the full service list (estop, wheels, lidar power, …)

Types seen in this robot's own history: **`charge`, `standard`, `along_given_route`**.
Verified by sending one: a `charge` move to the pose the control_unit itself last used
(12.955, -17.617, ori -0.219) docked the robot and it drew -2.1 A within **24 s**.
- **A local move runs in the chassis's planner, exactly like a cloud task** — so killing
  the process that posted it does NOT stop the robot. Cancel it, or take the wheels away
  with `set_control_mode remote` (which is what the straight-line driver does anyway).
- **The charge target is an approach pose, not where the robot ends up.** Docked, the
  tracked pose read (13.348, -18.25, ori -0.0) against that target — 0.74 m and 12 deg
  away — because the chassis does its own contact-seeking at the end. Do not read a
  completed charge move's target as the robot's position.
- Worth revisiting what this unlocks: `elevator_runner` drives through the cloud today,
  which is where the ~24 s of arrival lag and the read-only map both come from.

**Aligning to a doorway is a HEADING SWEEP, not a gap measurement**
(`drive_straight.py align`, 2026-09-04). The first version found the nearest return on
each side and called the space between them the opening; facing a blank wall 1.66 m away
it duly reported "an opening 0.041 m wide", which is true and useless. What the robot can
act on is which HEADINGS let its whole swept rectangle through, so that is what is
computed: for each candidate heading the scan is rotated into it and the swept path
checked, and the clear headings are reported as a band. A doorway is then found by the
robot fitting through it rather than by recognising a door.
- Validated by making the problem harder in each dimension separately: a 2.0 m leg had a
  51 deg clear band, a 3.5 m leg only 11 deg (the far wall enters the path), and demanding
  a 1.9 m corridor turned straight-ahead BLOCKED while still finding a clear band at
  -16..-3 deg — i.e. the doorway case, "straight will not fit, 3 deg will".
- It reports in metres, not degrees, where it matters: the verdict compares the slack
  going straight against the drift left over from the ~3 deg the base cannot resolve.
- Read-only, and cheap enough for dead time — the scan is 2.00 Hz with 74 ms of latency,
  so two confirming reads cost about a second against a 60 s press. Take it while the car
  is moving and leave only the straight leg for the door's open window.

**Read the pose only after it has SETTLED** (`_settled_pose`, 2026-09-02). Waiting for
"a couple of fresh samples" after braking was not enough: SLAM settles late, so the
sample arriving right after the brake still describes a place the robot has left. A leg
commanded 0.40 m read back **0.127 m** that way, the correction legs fired to make up
the difference, and the move ended at **0.589 m — a 47 % overshoot of the request**.
The test is now agreement, not arrival: keep pumping zero twist until two consecutive
fresh poses agree to 1 cm and 0.5 deg. After the fix, 0.80 m commanded achieved 0.753 m
with 1 mm of lateral. Cross-checked against an independent camera fiducial (the panel's
own measured distance): a 0.50 m leg moved 0.401 m by camera and 0.384 m by pose, so the
settled pose is trustworthy — it was only ever the mid-move samples that were not.

**Where the chassis actually is, because three earlier answers were wrong.** It is
`192.168.25.25`, on the robot's OWN WIRED network — the same cable as the arms, but a
**different subnet**, only reachable once the Jetson's `eno1` also carries
`192.168.25.46`. Things that are NOT the chassis, each of which cost a detour:
- `192.168.11.35` is a **`LubanCat-4` / EmbedFire Android 12 board** doing iFlytek AIUI
  voice + face HRI (ports 9070/9080/9090 audio-video, 19199 AIUI control, 5555 ADB).
  All 65535 ports scanned: no control interface. Do not look for 8090 here.
- `192.168.11.34` is an lwIP device with HTTP Basic auth on port 80 — still unidentified.
- Three AutoXing chassis DO answer `/device/info` on the WiFi subnet
  (`192.222.10.102/.161/.235`) and **none of them are ours** — they are other people's
  `longjack` and `hawk_longtray` robots. `drive_straight.py` checks the serial
  (`1352512k06960Og`) before it will send anything, and that check is not optional.

So this robot's parts sit on **three mutually unreachable networks** (Jetson wired,
Jetson WiFi, chassis subnet), which is why everything local had to go through the cloud
until now.

**Four measured facts the driver is built around:**
- **twist keepalive must be >=20 Hz.** At ~4 Hz the base's watchdog stopped the wheels
  after 2.3 cm, which reads exactly like "the command was rejected".
- **`/tracked_pose` is 1.07 Hz AND repeats the same value while moving** (SLAM settles
  late). See the gotcha below — this single fact produced five wrong conclusions.
- **achieved/commanded distance is 0.90**, so the leg duration is divided by it.
- **braking coasts ~11 cm at 0.2 m/s**, so the last 18 cm ramps down to a 0.06 m/s creep.

**Safety, in the order it applies**: serial check -> lidar clearance gate (before any
twist) -> current-based stall abort -> heading/estop/timeout -> brake and restore
`auto` on EVERY exit path. Leaving the base in `remote` silently disables AutoXing
navigation for whatever runs next, so the restore is retried three times and shouted
about if it fails. Note the base's own move controller is NOT in the loop: nothing
avoids obstacles *during* the motion, the clearance gate is a check *before* it.

**Obstacle sensing for the BASE is the lidar, not a camera** (`/scan_matched_points2`).
360 degrees, ~870 points/frame, map coordinates, **1.87 Hz with 74 ms median transport
latency**. Transformed into the body frame it gives forward AND back clearance in one
shot — which the chest camera structurally cannot do, since it faces only the arm side;
reasoning about the other direction from it produced a wrong "it hit a wall". Verified
against a person standing 0.9 m away: **two clusters, 14-15 cm wide, 32 cm apart** —
two shins, and the gate refused to move.

**The three sensors see different HEIGHTS and cannot substitute for each other:**

| sensor | sees | owns |
|---|---|---|
| chassis lidar | z ~ 0, a **7 cm** thick plane (14,800 points spanned 0.07 m) | base obstacles: people, chairs, walls, door frames |
| **arm-mounted camera** (planned) | the panel, from wherever the arm puts it | button detection and press verification |
| chest camera | the arm's workspace at 0.7-1.2 m | assisting: workspace clearance before the arm moves |

The lidar cannot see a hand reaching toward the panel (half a metre above its plane),
nor low objects or overhangs outside that plane. The chest camera cannot see behind the
robot. Neither gap is fixable by configuration.

**Driving to the panel** (`elevator_runner/`, a Flask tool on the robot) — loads the
robot's waypoints from **AutoXing's cloud API**, drives a chosen route, and when the
robot arrives within tolerance of the point marked as the elevator, runs
`press_buttons.py <floors> --go --lift`. Dry run is the default. It binds to
`127.0.0.1:8765`, so it is reachable only by someone already SSH'd in (deliberate on a
shared box); credentials live in a gitignored `.env`.
- **AutoXing returns a POI's position as `coordinate: [x, y]`, NOT top-level `x`/`y`** —
  while `robot_state` DOES use top-level `x`/`y`. Reading `poi["x"]` silently defaulted
  every waypoint to (0, 0), so the arrival gate compared the robot's distance to the MAP
  ORIGIN (2927 cm here) against an 8 cm tolerance: a live run would have driven to the
  elevator, refused to press as "out of reach", backed off and retried until it gave up.
  The dry run hides this because it substitutes a fixed 3.0 cm simulated error. Fixed
  and verified by computing a real distance: 2.8 cm to the POI named `elevator test`.
- This map uses none of the AutoXing elevator POI types (`[6, 28]`) — the elevator point
  is type 11 like any other waypoint — so the UI's auto-highlight never fires and the
  operator picks the point by name.
- **Our AutoXing credentials are READ-ONLY for the map.** `/map/v1.1/poi/list` and
  `/map/v2.0/poi/list` answer 200; fourteen plausible write paths (`poi/update`,
  `poi/edit`, `poi/save`, `poi/create`, `poi/delete`, `feature/update`, PUT `/poi`, …)
  all return 404. So waypoints can be listed and driven to, but not moved or created —
  that is done in the AutoXing app. Probe such an API by sending a record's EXISTING
  values: on a shared production map, discovering that a write endpoint works by
  having changed something is the wrong way to find out.
- **Arrival is gated on the BASE, not on the cloud's task status** (`base_arrived` +
  `wait_for_arrival`, 2026-08-27). `robot_state` is a level below the task: it carries
  the base's own `moveState`, `speed`, `hasPersonAhead`, `locQuality` and `battery`,
  and the base reports it **event-driven** — parked, its timestamp goes 60 s without
  moving, but the report that follows a change arrives ~2 s old. Measured side by side
  on one drive: the base said "arrived" at **51 s**, the task-status poll at **75 s**,
  and the two positions differed by **0.6 mm**, i.e. the base really had stopped.
  Four conditions, all required, plus two guards:
  - timestamp after dispatch (not last run's stale report), `moveState` done AND
    `speed` 0, and within tolerance of the elevator point (filters intermediate
    route points, where a multi-point task also completes a move);
  - **departure must be observed first.** Every loop after the first STARTS parked at
    the elevator point with `moveState` already "succeeded", which satisfies all four
    at once — without this guard the press fires as the robot is about to drive away;
  - **position must not drift between the two confirming reports** (>1 cm = still
    travelling). This is the only way the signal can be early, and it is exactly the
    failure that put the arm into the panel when the camera was the trigger.
  The task status stays in the loop as the BACKSTOP — it catches a cancelled task, or
  one that ended somewhere the fast gate never accepts. `fast_arrival: "observe"`
  computes the fast signal and logs when it WOULD have fired while still waiting for
  the cloud: one loop in that mode measures the real saving at no risk, which is how
  the 24 s above was obtained.
- **Careful what "the cloud is 20-55 s late" means.** An earlier note said that, from
  comparing a task's `endTime` to when the runner acted. Wrong decomposition: `endTime`
  is only **6-8 s** after the base itself reports arrival (57 vs 51; 99 vs 91). The lag
  is between `endTime` and when a poll can SEE `taskStatus == 4` — 18 s on the measured
  run. The operationally honest number is the one measured end to end: **the base
  signal beats the task-status poll by ~24 s.**
- **The press process is PRE-WARMED during the drive** (`press_prewarm` /
  `press_release`, `press_buttons.py --wait-go PATH`). ~7.3 s of the press's startup
  does not depend on where the robot is — 2.8 s loading the TensorRT engine, 2.5 s
  opening the camera, ~1.5 s ramping the GPU off its 306 MHz idle clock, 0.4 s
  connecting the arm — so it is paid in parallel with the drive. The child does
  everything position-independent, prints `PREWARM_READY`, and BLOCKS on a token file
  that only the runner writes, after arrival is confirmed. **Nothing about the arrival
  decision moves into the press**; letting the press infer arrival from the camera is
  what drove the arm into the panel once. Measured: 8.3 s -> **1.6 s** between arrival
  and the first motion. Every path that abandons an attempt must call `press_kill`,
  or the child keeps the camera open and the next attempt fails with "no color frame
  after 40 tries".
- **The full press transcript goes to `/tmp/dex_press_last.log`** (`press_log`). Only a
  three-line tail reaches the UI, and when a press failed mid-sequence that tail was
  the camera's startup banner — which says nothing about why. The per-button lift
  height, joint margin, clearance and residual are in the transcript.
- **`poll_timeout_sec` is a STALL timeout, and progress means MOTION** (2026-08-28).
  Measured from dispatch it conflates "slower than usual" with "stuck", which want
  opposite responses: one drive took **316 s** where the same route normally takes
  56 s — 9.30 m against 8.77 m, i.e. nearly the same path at a fifth of the speed — and
  a 300 s budget gave up **6 s** before that task completed, with the robot already
  3 cm from the target, killing the pre-warmed press with it. So the clock now measures
  time since the base last moved.
  **Progress is position change or non-zero speed, never what the base says about its
  situation.** A first version counted `hasObstruction`/`hasPersonAhead` as progress,
  reasoning that a base with a reason to wait is not stalled. That is wrong in the
  worst direction: measured, the base finished its 7.22 m route, stopped 5 cm from the
  elevator point, and held `moveState=moving, speed=0, hasObstruction=True` for over
  **six minutes** without moving — which would have waited out the whole `hard_cap`.
  Motion is a fact; "there is an obstruction" is an opinion, and they come apart
  exactly when it matters.
- **A stall AT the target is not a failure — but cancel the task before pressing.**
  That six-minute case was the base parked exactly where it was asked and simply never
  saying "succeeded". When the stall fires inside tolerance, the task is CANCELLED
  first (so nothing can command a late docking nudge while the arm is out), stillness
  is re-confirmed, and that counts as arrival. This is not the camera-based guess that
  drove the arm into the panel: it is the base's own odometry showing no motion over a
  long window, with nothing left that could move it.
- **The base's situation is LOGGED whenever it changes** during a drive — moveState,
  moving/stopped, personAhead/obstruction, distance to go. A slow drive has to be
  diagnosable afterwards: with only mileage and duration to go on, a slow drive and a
  stuck one are indistinguishable. A handful of lines per drive.
- **A person or object blocking the path shows up as `hasObstruction`, NOT
  `hasPersonAhead`.** Operator-confirmed: the obstructions seen on 2026-08-28 were
  people and chairs put in the robot's way deliberately, to test that it stops and
  re-routes. So `hasPersonAhead` is a different signal (the HRI/face path), and
  `hasObstruction` is the one that means "something is in the way right now". Worth
  stating because the false inference runs both ways — a `hasPersonAhead` of False does
  NOT mean no person, and reading it that way is how a correct first guess got
  retracted for the wrong reason.
- **Obstacle handling is verified, and the runner must not fight it.** Blocked, the base
  stops, waits, re-routes and still reaches the point: measured across several drives —
  a block at 17 cm from the point held it for 63 s before it escaped and completed to
  **0.4 cm**; another produced 9.30 m of travel against a nominal 8.77 m and still
  arrived. This is normal operation on a real floor, not an exception, which is exactly
  why the stall timeout has to key on MOTION rather than on elapsed time.
- **The base can report a task "succeeded" while still ~1 m short.** Observed: it
  declared success at **95 cm** from the elevator point, and the corrective drive
  closed it to 7.9 cm in 25 s. This is what `corrective_drives` exists for; it also
  goes through the same `wait_for_arrival`, so it gets the fast gate too.
- **Only the ELEVATOR point's docking accuracy matters; a route's other waypoints are
  places to drive FROM** (measured 2026-08-31). `BBB` docks badly and inconsistently —
  **3.0 / 7.6 / 8.6 / 26 / 68 cm** from its point over five navigated trips, 2 of 5
  inside 8 cm — while `elevator test` closed to **1.5-2.3 cm on nine of nine arrivals**
  the same day. It changes nothing: a run that stopped ~17 cm from `BBB` drove on,
  reached the elevator point at **0.9 cm** and pressed **4/4**. Do not treat a start
  waypoint's docking error as a blocker, and do not measure it as if it were.
  Docking quality is a property of the SPOT: at `BBB` the base stops wherever it is,
  declares `moveState: succeeded` and never converges — no `hasObstruction`, no
  `hasPersonAhead`, no `failed` — and re-recording the waypoint onto a reached pose did
  not fix it. Yaw there IS repeatable (4.43-4.45), so orientation is not the variable.
  **The arrival gate is not what fails**: the fast base gate correctly refused the 26 cm
  and 68 cm cases, and what returned "finished" was the cloud task-status backstop, which
  does not look at distance. `run_loop` re-reads the position afterwards and gates on the
  measured error, so those go to `corrective_drives` rather than to the arm.

**Live previews for positioning the robot by hand** (both read-only, both safe to leave
running — neither commands anything):
- `initialization/lidar_preview.py` — top-down chassis lidar in a browser (port 8011):
  forward/back clearance inside the base's own width, corridor width, and **which side
  it is off-centre**, which a single "nearest obstacle" number hides.
- `initialization/aim_preview.py` — chest camera with live button detection, bearing off
  the camera axis, distance, and the panel's position in the ARM BASE frame (port 8010),
  plus a READY/NOT READY banner. Built because aiming the robot at a panel by hand
  otherwise meant grab a still, copy it off, look, nudge, repeat.
  Both are reachable from the dev Mac now that it shares the robot's WiFi subnet, so
  `--web`-style previews no longer need the Ethernet cable.

**Test harnesses for the drive + press loop** (`elevator_runner/`, added 2026-08-31 —
they existed only on the robot before, which meant re-deriving them after any reset,
and stray `.py` files at the repo root shadow the stdlib):
- `liverun.py [start] [floors]` — ONE live loop: drive via `start` (default `BBB`, `-`
  skips it) to `elevator test`, then press `floors` (default `1 4 2 5`). Verified
  2026-08-31: `BBB` -> elevator -> `1 4 2 5` ran **3 of 3, 4/4 buttons each**, arrival
  0.9-2.3 cm, 66.9-68.1 s per press. Pre-flight checks e-stop, both arm controllers and
  the base being online BEFORE dispatching, because an earlier attempt spent two route
  dispatches and ten minutes discovering from a timeout that the e-stop was engaged.
- `goto.py <waypoint>` — drive to a named waypoint. Uses the same `wait_for_arrival`,
  so it behaves like a real run rather than being a second, differently-behaved path.
  **Read the cm figure it prints, never its `arrived=` flag**: it passes a 15 cm
  tolerance where production uses 8 cm, and it prints `wait_for_arrival`'s return value
  without the position gate that follows it in `run_loop` — so it reported
  `arrived=True` while parked 68 cm from the point.
- `liverun_stream.sh` / `initialization/press_stream.sh` — run the job and `tail -f` its
  log. Use these, not "wait for it to exit": see the streaming note under Common
  commands.
- `start_runner.sh` — start the Flask runner detached and wait until 8765 is listening.

**Pressing on its own** (`press_buttons.py --auto`) — waits until the panel is both
visible AND still (5 consecutive frames within 2 mm), then presses ONCE and exits;
re-running is the manual re-trigger, it never re-arms itself. The trigger is the
PANEL's pose in the arm base frame rather than the base's odometry, which needs no
access to the base at all. Not needed when the runner drives, since the runner already
gates on arrival; it exists for hand-positioned or non-AutoXing deployments.

**Orchestrator** (`core/elevator_pipeline.py`): capture → detect → match the button
whose label == requested floor → `press_target_from_pixel` → arm standoff/press/retract.

**RealMan adapter (`core/robot/realman.py`)** is the only code that imports the SDK.
TCP pose is a 4×4 `base_T_gripper` in **meters**. The SDK reports/consumes a pose as
`[x, y, z, rx, ry, rz]` with **position in meters, orientation in radians**;
orientation is converted via the SDK's own euler↔quaternion helpers, so the Euler
axis order is never guessed. Also exposes joint read, `move_line_to_pose`
(`rm_movel`, for the straight press), lift get/set, and drag-teach via
`set_manual_mode` (`rm_start_drag_teach`/`rm_stop_drag_teach`). `open/close_gripper`
are no-ops (the LinkerHand is separate).

**Calibration** (`calibration/`, self-contained: `cv2`/`numpy`/`core` only):
- Part 1 `intrinsic.py` → `data/calibration/<cam>_intrinsics.npz` (K + distortion).
- Part 2 `extrinsic.py` → `data/calibration/<cam>.npy` (`base_T_camera`, eye-to-hand).
  Part 2 detects the board using Part 1's intrinsics, so **order matters**.
- The solver takes `base_T_gripper` + `cam_T_target` pose lists (arm-agnostic). All
  OpenCV methods run; `select_best` picks by residual — **PARK is the reliable one**
  for angled/down-looking cameras. Uses the modern `cv2.aruco.CharucoDetector`.
  Default board: **14×9, 20 mm squares, 15 mm markers, `DICT_5X5_100`**.
- `core/camera/manager.py` loads both files once per camera and shares them; nothing
  recomputes calibration at runtime.

**Orbbec driver (`core/camera/orbbec.py`)** selects a device by `serial` (exact,
preferred) or `match_name` (name suffix, e.g. "335"/"335L") so two units don't
collide; picks the best decodable color profile at the requested resolution and
decodes to RGB. `python -m core.camera.orbbec` lists connected devices.

**Config**: `configs/cameras.yaml` (per-camera `serial`/`match_name`/resolution +
`intrinsics_file`/`extrinsic_file`) and `configs/pipeline.yaml` (`arm` = RealMan
side/ip, `button_yolo` weights, `elevator` = panel plane + press params). Loaded via
`core/config.py` (`REPO_ROOT`-relative). **The `elevator.panel` plane and press
poses are PLACEHOLDERS — measure them on the real cell before running on hardware.**

## Gotchas

- **RealMan pose convention**: verified units are m/rad; the adapter uses SDK
  euler↔quaternion to build the matrix. Sanity-check the first solved extrinsic's
  residual (should be a few mm) — a wrong convention/units shows up as a big residual.
- **Arm contention**: drag-teach and motion contend with whatever else drives the
  arm. Make sure the richtech stack isn't commanding the right arm during
  calibration; don't change the active tool frame mid-capture.
- **Two Orbbecs**: with both plugged in, always select by `match_name`/`serial`
  (else the SDK opens whichever it enumerates first).
- **Lift look-then-move**: settle the lift → measure with the camera → press. Don't
  move the lift/base between measuring and pressing (the measurement goes stale).
- **Headless robot / preview**: the robot has no monitor and the Mac has no XQuartz,
  so use `--web` (MJPEG to a browser on the LAN, `http://192.168.11.31:8010/`) for
  any capture — a local OpenCV window or `ssh -Y` shows nothing.
- **Calibration board mount**: bolt the ChArUco board to the bare **flange** (remove
  the LinkerHand — its fingertip TCP isn't defined in the controller). `base_T_camera`
  is end-effector-independent, so the hand is remounted afterward with no re-calibration.
- **`rm_movej`/`rm_movel` need arrival confirmation — fire-and-forget misreports the
  cause.** Two failure modes stack: they return `false` while still finishing the
  move in the background, AND they reject a new command while the previous one is
  settling. Symptoms point somewhere else entirely: the NEXT button came back "no
  IK solution" (the IK seed was still the previous pose), and a press reported
  "pressed at +50 mm" when the `movel` had simply been dropped. Both looked like the
  button was unreachable — single-button retests proved otherwise. Use
  `RealmanArm.move_joints_sync` / `move_line_sync`, which poll joints/TCP until
  arrival, retry 3x, and settle 0.35 s between moves.
- **`/tracked_pose` is 1 Hz and REPEATS ITSELF while the base is moving — never read
  it as live feedback.** Measured 1.07 Hz (median gap 1.03 s), and consecutive messages
  carry the same value during motion because SLAM settles late. At 0.2 m/s that is a
  20 cm position resolution, against an elevator reachable window of +-8 cm. Reading it
  as if it were live produced FIVE wrong conclusions in one session: a 0.20 m move
  "overshooting" to 0.31 m; three separate "STALLED — something is in the way" aborts
  while the robot was driving at 0.18 m/s; and a "wheel slip" that cut a move short.
  The clean measurement that settled it — 100 twists over 5 s, **120 `/twist_feedback`
  replies, 0.903 m travelled, 0.181 m/s** — showed the chassis had been executing
  perfectly the whole time. Drive distance **open-loop on time** with the calibrated
  0.90 ratio; use the pose only for a settled before/after measurement (0.0 mm of
  scatter while stationary) and a correction leg.
- **`/slam/state`'s `wheel_slipping` is not a stall signal.** It goes true during
  ordinary driving — it cut a 1.00 m move short at 0.695 m while the current sat at a
  healthy 5.7 A. Like the pose, it is unreliable *while moving*. Log it, never abort on
  it; the distance loop reads before/after pose rather than odometry, so a slip cannot
  corrupt the measurement anyway.
- **Stall detection uses battery CURRENT.** Driving into the charging dock drew
  **30.4 A** against ~4.4 A idle and ~5 A driving normally, with zero displacement. It
  is a physical quantity with no SLAM lag. 20 A held for 0.8 s. (`/battery_state` is
  also only ~1 Hz, so pair it with the lidar gate rather than relying on it alone.)
- **Restore `control_mode` to `auto` on every exit path.** Leaving the chassis in
  `remote` silently disables AutoXing navigation for whatever runs next, and nothing
  warns you — the next cloud task simply does not move. `/wheel_state` reports the mode,
  so it can be verified rather than assumed.
- **Re-assert `remote` and keep sending zero twist between legs.** A 2-3 s gap with no
  twist at all appears to drop the base out of its remote-driving state, which showed up
  as "the first leg always moves and every later one stops after a few centimetres".
- **A chassis reboot clears some alerts and a stuck action, but not all.** It cleared
  `6007` (head_unit link) and a `charge / failed / platform_alert_error` action that had
  been stuck in `/planning_state`, leaving the base `idle`; `6010 System down
  unexpectedly!` and `9502 Debugging config file exists` survived, and `6007` came back
  later. Confirm a reboot actually happened by watching `planning_state.action_id` reset
  — the first time, it did not (`action_id` stayed 643) and nothing had changed.
- **The lidar point cloud can come back EMPTY, and the cause is not the emergency stop.**
  Seen 2026-09-04 morning: `/scan_matched_points2` publishing at 1.33 Hz with
  `npoints = 0` while `/slam/state` still reported `lidar_reliable: true,
  lidar_matched: true`, so nothing said the lidar was unavailable. It recovered to 2.00 Hz
  and ~900 points after the e-stop was released, and I wrote that down as cause and
  effect. **That was wrong** — measured again the same afternoon with the e-stop ENGAGED,
  the scan was publishing 878-938 points per frame quite happily. One co-occurrence is not
  a mechanism. The real trigger is still unknown; a chassis that has just booted and not
  yet localised is the leading candidate. What matters operationally is unchanged: an
  empty scan reads exactly like "nothing in the way", so anything built on it must fail
  CLOSED, and `drive_straight` aborting with "no lidar scan" is correct behaviour that
  should not be relaxed. (The e-stop DOES unpower both arm controllers, which come back a
  few seconds after release — that part was independently verified.)
- **`initialization/boundary_sweep.py` no longer runs against the current config.** It
  reads `arm.lift.usable_relative_z_m` and `target_relative_z_m`, both of which were
  removed when the lift height became SEARCHED rather than computed, so it dies with a
  `KeyError` before doing anything. Fixing it means teaching it to search the lift the
  way `plan()` does — it is not a missing-key patch, because the fixed-height model it
  encodes is the superseded one.
- **Killing the local process does NOT stop the robot.** An AutoXing task is executed
  by the cloud, not by our process: after `kill -9` on the runner the base carried on
  and drove the rest of its route by itself. Stopping means cancelling the task
  (`AX.cancel_task`) or going through `/api/stop`, which does. `kill` alone leaves the
  robot driving.
- **`move_joints_sync` returning False does NOT mean the arm stopped.** It means
  arrival was not observed within the timeout; `rm_movej` keeps executing in the
  background. Reading that False as "it did not move" and going off to plan a retry is
  how an arm ends up crossing the room while nobody thinks anything is happening —
  confirmed by watching the joints creep toward the target between two status reads. On
  a timeout, call `rm_set_arm_stop` + `rm_set_delete_current_trajectory` FIRST, then
  decide.
- **The self-collision check does not include the robot's own body.**
  `rm_algo_safety_robot_self_collision_detection` covers the arm's links and end
  effector only — not the torso, the lift column, the camera mount or the chassis. A
  path that passes it, and whose plunger tip stays clear of the panel, can still hit
  the robot itself. There is no model for that, so a large re-positioning move needs
  eyes on it or hand-guiding; two computed checks passing is not "it will not collide".
- **The press log's own depth/lateral figures CANNOT detect a wrong TCP.** They compare
  the commanded tip against `get_tcp_pose() @ tcp_offset` — the controller servos to
  the assumed TCP and both sides of the comparison use the same offset, so an error in
  it cancels out. Measured: after the plunger was adjusted by hand the offset was
  5.57 mm out and the reported lateral error stayed under 1 mm. The only signals that
  see it are the button lighting up and an independent re-measurement (touch the tip to
  a button, then `offset = inv(base_T_tool) @ camera_measured_button_point`).
- **Close the hand into a fist BEFORE moving the arm.** The LinkerHand's fingertips sit
  172.87 mm from the flange while the plunger tip is at ~154 mm, so with the fingers
  extended the HAND is the front-most part and reaches the panel first. A fist folds
  them behind the plunger. This is a pressing prerequisite, not tidiness.
  **`press_buttons.py` now does it itself**, as the first thing after connecting and
  before any motion (2026-08-27) — it used to be done out-of-band by whoever was
  driving, and an out-of-band prerequisite is one someone eventually forgets. The hand
  also loses power across an emergency stop, so "it was a fist last time" is not a
  state that survives. The fist is VERIFIED by reading the joints back, and the test is
  "nothing is sticking out" (all six <= 100) rather than per-joint target matching: in
  a fist the index bottoms out against the thumb at ~70, not 0 (measured 67 here, 69/75
  on the two hands earlier), so an exact test would fail every time. If the hand does
  not answer or does not close, the press REFUSES to move. `--no-fist` is for a robot
  with no hand fitted.
- **The base's port 9090 is the FACE camera, not navigation.** It is a binary push
  stream (`a5` magic, 1 byte version, 1 byte type, 4-byte LE length, 2-byte sequence),
  and everything on it is HRI: a one-off map-metadata JSON, MJPEG frames at 1.8 Hz, and
  a per-frame `hasFace`/`wakeup`/`hasHand` list. **No pose, no odometry** — so it is not
  a local arrival signal, and it is not rosbridge despite the port number. Port 5555 is
  also open (almost certainly ADB, the base's Android side); left alone. Recorded so
  nobody re-derives this: the only local-looking route to the base does not carry what
  a local arrival signal would need.
- **An interrupted tool call may already have started the remote process.** A
  `rsync && ssh ... nohup` that was cancelled mid-flight had ALREADY launched a live
  run, which then kept driving and pressing while the next command was being composed;
  worse, the `pkill` sent to stop it returned SSH exit 255 and that was not retried.
  Before starting anything on this robot, check `pgrep -af "liverun|press_buttons"` AND
  the cloud task, and confirm the previous one is actually gone — "I cancelled it" is
  not the same as "it stopped".
- **A camera that enumerates but never delivers colour is fixed by
  `device.reboot()`, no root and no replug.** The chest 335 got into this state after a
  USB re-enumeration: `lsusb` and `query_devices()` both listed it, and every
  `capture()` died with "no color frame after 40 tries". The head 335L worked
  throughout, which is what proved it was device-state and not the code — test the
  OTHER camera before touching anything. `pyorbbecsdk`'s `Device.reboot()` cleared it
  in ~25 s. Note the misleading first hypothesis: the failure appeared right after
  TensorRT was added to the same process, so it looked like engine init was starving
  the camera stream; reordering changed nothing, and the camera failed on its own too.
  **It costs a whole drive when it hits mid-run, and nothing recovers automatically**
  (seen 2026-08-31): the pre-warm died on it, the runner fell back to a serial press, and
  the press then died the same way — `successes=0 failures=1` after driving the full
  route. Two API details for the reboot: `Context` and `DeviceList` must be held in LIVE
  references (temporaries fail with `NULL pointer passed for argument "deviceMgr"`), and
  the device re-appears in `query_devices()` within ~2 s while still delivering nothing,
  so poll an actual `capture()` rather than enumeration.
- **`pip install --extra-index-url` will happily pick the WRONG index.** The
  jetson-ai-lab index and PyPI both offer `torch==2.10.0`; with both indexes visible pip
  chose PyPI's **CPU** build and `torch.cuda.is_available()` stayed False. Install GPU
  torch with `--index-url <jetson index> --no-deps` only, then add the pure-python deps
  from PyPI separately.
- **Stray `.py` files at the repo root on the robot shadow the stdlib.** The July
  deployment left a flattened copy of the whole package there, including `types.py` and
  `config.py`; any python started from `~/Dex_Elevator` then died with a circular-import
  error out of `enum`/`dataclasses`. Moved to `~/Dex_Elevator_stray_root_backup_*`. The
  repo root legitimately contains NO `.py` files — if one appears, an rsync flattened
  something. **That same flatten also left duplicate `camera/`, `hand/`, `robot/` packages
  and root-level `pipeline.yaml`/`cameras.yaml`/`buttons.yaml` on the robot** (still there
  2026-09-02). Nothing imports them — `core/config.py` reads only `configs/` — so they are
  inert, but the root `pipeline.yaml` is a DIVERGENT older copy (no left-arm home pose, no
  `lift.objective`), which is exactly the kind of thing someone eventually reads by mistake.
- **`rsync` to the robot must exclude `data/` — the model is trained ON the robot.**
  `data/weights/buttons.pt` exists on both machines and the robot's is the newer one, so
  a sync that includes it pushes the Mac's stale copy the WRONG WAY. Measured 2026-09-02:
  the documented command (which excluded only `data/calibration/`) silently replaced the
  5-source merge from 2026-08-25 with the Mac's July single-source build. It is easy to
  miss because rsync reports only that a file differed, never which side was right, and
  because nothing breaks — `trt_detector.py` loads the `.engine`, not the `.pt`, so
  inference kept working with the correct model while its source was gone. Recover from
  `runs/detect/buttons_merged5/weights/best.pt`, which is what `buttons.pt` was copied from.
- **Never trust a Jetson benchmark that has not been warmed up.** The CPU governor is
  `schedutil` and the GPU idles at 306 MHz, so **whatever is measured first in a process
  pays for the clock ramp and everything after it looks faster**. This produced two wrong
  numbers in one session, including a step-by-step profile whose parts summed to 10 ms for
  a call that measured 26 ms. Discard 120-150 iterations before recording, and if the parts
  do not add up to the whole, instrument the real call's own lines rather than timing the
  steps in isolation — isolated timings are also systematically optimistic, because the
  real loop's 4.9 MB/frame of writes evicts the cache that the isolated test kept warm
  (`letterbox` measured 0.8 ms alone, 2.6 ms in place).
- **Zero-copy is NOT automatically right on a Jetson.** Host and device share physical
  memory, so mapped host buffers look like a free win — but the GPU then reads and writes
  **uncached** memory, and its compute went 6.4 -> 13.2 ms. Against `cudaMalloc` + pinned
  staging copies it is a near-tie (43.2 vs 44.9 ms end-to-end, A/B'd twice interleaved);
  plain pageable `cudaMemcpy` is the only clearly bad option (42 ms just in copies for
  17 MB of tensors). Both paths are kept behind `zero_copy=` in `trt_detector.py`.
- **The public-data YOLO baseline is not "bad on our panel" — it is bad on LOW-CONTRAST
  markings specifically.** Measured per button on a live cam_chest frame: `alarm` 0.97,
  `close` 0.98, `open` 0.93, `2` 0.86, and the unmarked disc correctly `empty` 0.98 — but the
  etched `5/6/3/4` come back as `empty`/`12`/`15`/`37`, and `1` (a green star overlapping the
  digit) as `15`. **Whatever a human can read in the frame, the model reads correctly**; the
  failures are buttons whose marking is barely present in the image. So do NOT describe the
  baseline as "labels our buttons empty" — that misdiagnoses it as a metal-vs-plastic domain
  gap when the real variable is marking contrast.
  - **Exposure is NOT the lever; the LIGHT DIRECTION is.** Swept 50-190 (gain 16, room
    light on): per-button contrast rises monotonically with exposure (11.0 -> 19.8) but the
    correct count never exceeds 3/8, and the LOW end is worst (0/8 at exposure 50). Then
    **turning the room light off** and re-sweeping: contrast **18.2 -> 26.0 (+43 %)** and
    4/8 correct, with `5`/`6`/`3`/`4` becoming legible to a human for the first time. The
    digits are laser-etched into brushed steel — a shadow feature, not an albedo one — so a
    broad overhead source fills the etch from every direction and erases them, and no global
    exposure can put them back. Note "turn the room light off" is a DIAGNOSTIC, not a
    deployment option — a real lobby's ceiling lighting is exactly the bad case and we do not
    control it.
    **A robot-mounted grazing light was considered and RULED OUT by the operator (2026-08-21)** —
    do not propose it again. So the etched digits will never be reliably legible, and the design
    must not depend on reading them. The chosen direction instead:
    - button POSITIONS come from YOLO (reliable — 10/10), which also removes the OpenCV
      Hough-circle stand-in from the pipeline;
    - the label for each position comes from that elevator's REGISTERED LAYOUT, captured once at
      commissioning. This is not the same as hard-coding the panel's position in space (that
      must stay live-measured — the base docks with centimetres of error); the arrangement of
      buttons on a given faceplate is a physical property of that elevator;
    - the layout's alignment is VERIFIED using the buttons the model does read confidently
      (`open`/`close`/`alarm`, 0.93-0.98) as anchors, since their position in the layout is known;
    - if the anchors disagree with the layout, REFUSE to press rather than press something.
      This is the part that fixes the real hazard: a mislabelled button is a SILENT failure —
      the robot quietly presses the wrong floor with no error anywhere.
    Fine-tuning still happens, but its goal drops from "read every digit" to "make positions and
    the high-contrast symbols rock solid" — which the existing lighting already supports.
  - Also not levers, all measured: upscaling the crop (imgsz 640/1280/1920 on the same
    175x335 crop — 1920 makes it *worse*, 6 detections -> 2), CLAHE (clip 2/4: digits still
    wrong), and unsharp masking (turns every button into `empty` at 0.45-0.95).
  - When testing crops, remember Ultralytics resizes to `imgsz`: upscaling a crop 4x and then
    passing `imgsz=640` scales it straight back down. An early test "compared" native vs 4x
    and got identical numbers for exactly this reason.
- **Aim at the button CENTRE before touching `push_depth`.** A press that fails to
  actuate looks like "not enough force" but is usually "off centre": 2 mm on the
  chamfer does nothing, 2 mm on the centre nearly works. Fix the aim first, then the
  depth — otherwise you tune depth to compensate for an aiming error.
- **`push_depth` by hand-feel is ~3x too small.** The operator felt 0.5-1 mm of
  switch travel; the real figure is **3 mm** (1 and 2 mm both failed to light the
  button, 3 mm lit it repeatably). A finger pushes the switch directly, whereas the
  plunger must ALSO compress its own spring, and the spring eats most of the stroke.
  Expect the same bias with any compliant press tool.
- **ALWAYS outlier-reject after calibrating — a single bad view/pose costs 3x accuracy.**
  Measured on the 2026-08 run: intrinsics RMS 0.698 → **0.383 px** by dropping 25 of 88 views;
  extrinsic consistency 1.85 → **0.67 mm** and leave-one-out worst 33.6 → **1.81 mm** by dropping
  ONE pose out of 41 (`sample002`, captured before the arm had settled). The danger is that the
  headline numbers still look acceptable, because the mean hides the tail — always look at the
  *max* and the per-item distribution, not just RMS/consistency.
  - Intrinsics: `cv2.calibrateCameraExtended` returns `perViewErrors`; drop views above
    ~2x the median, then re-run. Cache the detected corners first — detection is the slow part
    (~8 min for 88 views), after which trying different rejection thresholds is instant.
  - Extrinsics: reconstruct `flange_T_board = inv(base_T_gripper) @ X @ cam_T_target` for every
    sample. The board is rigid, so these must all agree; bad poses stick out by an order of
    magnitude (0.6 mm median vs 34 mm for the bad one). This doubles as a measurement of how
    rigidly the board is actually bolted on.
  - Verify with a *held-out* metric (leave-one-out, or train/test split), never RMS alone —
    rejecting views always improves RMS, so RMS cannot tell you whether you improved or overfit.
  - Keep the rejects (`*_rejected/` dirs) and pre-rejection backups for traceability.
- **Capture only after the arm has fully settled.** Both outliers in the 2026-08 run
  (`sample002`, and the 8.3 mm point in the localization eval) were poses captured while the arm
  was still micro-swinging after being hand-guided. Let go, wait, then hit Capture.
- **This is a GEN-3 RealMan controller** (`rm_get_robot_info()` → `robot_controller_version: 3`).
  The gen-4 API family returns `-4` ("三代控制器不支持该接口"): use `rm_set_modbus_mode` /
  `rm_write_registers` / `rm_read_multiple_holding_registers` (which take
  `rm_peripheral_read_write_params_t`), NOT `rm_set_tool_rs485_mode` /
  `rm_write_modbus_rtu_registers`. Also: the multi-register write `rm_write_registers`
  **returns 0 but silently does nothing** here — write registers one at a time with
  `rm_write_single_register`. And reads come back as raw BYTES (num*2, big-endian per
  register), not register values.
- **`get_tcp_pose()` is NOT the flange.** A tool frame is already configured on this
  robot (`rm_get_current_tool_frame()` → z = 130 mm, payload 0.356 kg ≈ the hand), so
  poses are reported at a point 130 mm beyond the flange. Also the `base` frame is not
  the arm's own base — the current TCP reads 1317 mm from the origin, far beyond the
  RM-65's ~650 mm reach — so **never sanity-check reach by taking the norm of a base-frame
  position**; ask the controller with `rm_algo_inverse_kinematics` instead.
- **Depth must be aligned with AlignFilter, not `Config.set_align_mode()`.** On
  pyorbbecsdk v2 the latter is silently ignored: depth keeps coming out at the depth
  sensor's native 848x480 while colour is 1280x720, so indexing the depth map with a
  colour pixel silently reads a completely different part of the scene (this produced
  a plausible-looking but 2x-wrong TCP measurement). `core/camera/orbbec.py` now builds
  `AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)` and runs every frameset
  through it; depth comes back at 1280x720, pixel-aligned.
- **Never `np.linalg.svd` an Nx3 matrix to get a plane normal.** For N ~ 10k numpy
  builds the full NxN left-singular matrix: it cost 125 ms and was 96% of
  `fit_panel_plane_from_depth`'s runtime. Use `np.linalg.eigh` on the 3x3 covariance —
  same answer, microseconds (the whole fit went 649 ms → 4.8 ms).
- **Orbbec colour format: MJPG, not RGB.** The Gemini 335 advertises uncompressed RGB at
  1280x720@30 but never delivers a frame on it — `wait_for_frames` just times out, which
  looks exactly like a broken camera. `_FORMAT_PREFERENCE` in `core/camera/orbbec.py`
  therefore puts `MJPG` (the device's own default) first. Don't "optimise" it back.
- **`capture()` needs a lot of retries.** With depth enabled the 335 emits ~11
  depth-only framesets before colour syncs in (only ~0.5 s total — it is a frameset
  *count*, not a timeout), so `retries` defaults to 40. At 10 it fails intermittently.
- **Building `pyorbbecsdk` has two traps.** (1) `pip install .` fails until CMake has
  been run first (`mkdir build && cmake -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) ..
  && make -j && make install`). (2) The install drops the package's `__init__.py`, and
  that file calls `version("pyorbbecsdk2")` which raises because the local build registers
  no such distribution — both are patched in place under `~/.local/.../pyorbbecsdk/`.
- **Beware `UNKNOWN-0.0.0` collisions.** The system setuptools is 59.6 (< 61) so it
  ignores `pyproject.toml`'s `[project]` table and registers editable installs as
  `UNKNOWN-0.0.0`. pyorbbecsdk landed under the same name, so installing this repo
  editable *uninstalled pyorbbecsdk's `.so`*. If the camera suddenly reports
  `No module named 'pyorbbecsdk.pyorbbecsdk'`, restore it with
  `cp -r ~/pyorbbecsdk/install/lib/pyorbbecsdk/. ~/.local/lib/python3.10/site-packages/pyorbbecsdk/`.

## Stubs / not-yet-wired (marked in-code with `# TODO`)

- **Camera moving to the ARM** (decided 2026-09-01, not built). The chest camera is
  fixed, so the positions where it can SEE the panel and where the arm can REACH it must
  overlap — and at the new site they do not: the panel sits at `y = -0.55, z = +0.69` in
  the arm base frame, where **336 combinations (14 lift heights x 24 approach rolls)
  have no IK solution**, while detection was perfectly healthy (lattice residual
  0.7-0.9 px). Mounting the camera on the arm decouples viewing from standing position.
  What it costs, to be planned together: a different calibration problem
  (`gripper_T_camera`; the existing `cam_chest.npy` and its 41 samples are void), a
  geometry chain that can no longer treat `base_T_camera` as constant, cabling along the
  arm, measurements only while stationary, and — most importantly — **the obstacle check
  breaks**: it currently works because the home pose puts the arm entirely outside the
  chest camera's view, so the depth image shows the world and not the robot's own limb.
  With the camera on the arm that property is gone.
- **Which floor the car is on** — unsolved, and the next perception problem after the
  above. Three candidate signals, to be COMBINED and chosen per site rather than picked
  blindly, because they fail in different ways:
  - **button lamps** (lit/unlit) — occluded by the plunger during a press, and the
    locked exposure that makes digits legible saturates the lamp;
  - **the elevator's own floor display** — may not exist, may be outside any camera's
    view, and its font/format varies per building;
  - **the robot's IMU** — vertical acceleration integrates to which way and how far the
    car moved, needs no view of anything, but drifts and cannot give an absolute floor
    without a known starting point.
  Whichever is used, the system should state which signal it is relying on rather than
  fusing them into an unattributable answer.

- Button YOLO: tooling BUILT + a **multi-class** baseline trained (yolo11m — detects AND
  identifies each floor) → `data/weights/buttons.pt`, and **running on the Orin's GPU** via
  `yolo/trt_detector.py` (TensorRT FP16, 43 ms/frame). Speed is DONE. Accuracy is not:
  **measured on our panel 2026-08-20**, it is correct on every high-contrast marking and
  useless on the low-contrast etched digits. Fine-tuning on our own cam_chest captures is the
  remaining work; whether a grazing light is added first is undecided (see the gotcha).
- Implement `read_floor_label` (OCR / multi-class / template) — the identification step.
- **Wrist orientation for pressing.** The fingertip must approach along the panel's
  inward normal; parked by hand it sat 46.9° off, which would skid instead of press.
  Reachability is also orientation-limited: at the panel, sweeping the roll about the
  approach axis found only 3 of 12 directions solvable by IK — so the press pose has
  to be chosen from what IK accepts, not assumed.
- **Press works** (2026-08-20, 4/4 buttons lit) with a rigid **spring plunger** on the
  flange, not the LinkerHand: no joints to damage, and the spring supplies the
  compliance force control would otherwise have to. The hand is still fitted but is
  not what presses. `end_effector.tcp_offset` in `configs/pipeline.yaml`.
- Force-limited press (`rm_force_position_move_pose`) — now optional, since the
  spring provides mechanical compliance.
- **Verifying a press visually is unsolved.** The plunger occludes the button while
  pressed, and the locked exposure that makes digits legible blows out the panel, so
  the indicator lamp is lost in saturation. Options: a second capture at lower
  exposure after retracting, the head 335L from another angle, or skip it (pressing
  a lit button again is harmless). Note this means **"read the digits" and "see the
  lamp" need different exposures** — one setting cannot do both.

Runtime artifacts are gitignored (weights, calibration outputs, captures under
`data/**`). Don't commit them.
