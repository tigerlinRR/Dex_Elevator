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

**`press_buttons.py --lift` implements that** (contributed 2026-08-26): detect once at
the current visible height, then PER BUTTON raise the lift so that button sits at
`target_relative_z_m`, compensate the cached 3D coord by the achieved rise, press, and
restore the lift at the end. Per-button targeting beats one height for the whole panel:
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
  centres, keeping the largest cluster, separates the grid from those strays. Then
  `tight_roi` re-boxes the fit around the confirmed buttons only — the wall and cabinet
  behind are separate parallel planes and bias the depth fit.
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
- **A camera that enumerates but never delivers colour is fixed by
  `device.reboot()`, no root and no replug.** The chest 335 got into this state after a
  USB re-enumeration: `lsusb` and `query_devices()` both listed it, and every
  `capture()` died with "no color frame after 40 tries". The head 335L worked
  throughout, which is what proved it was device-state and not the code — test the
  OTHER camera before touching anything. `pyorbbecsdk`'s `Device.reboot()` cleared it
  in ~25 s. Note the misleading first hypothesis: the failure appeared right after
  TensorRT was added to the same process, so it looked like engine init was starving
  the camera stream; reordering changed nothing, and the camera failed on its own too.
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
  something.
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
