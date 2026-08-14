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
- **Python is the SYSTEM `python3` (3.10.12), not conda.** It already has numpy 1.21.5,
  **cv2 4.8.0 with `aruco.CharucoDetector` + `calibrateHandEye`**, and pyyaml. Everything
  else is `pip install --user`: `Robotic_Arm` 1.1.6, `pyorbbecsdk` 2.1.2 (built from
  source in `~/pyorbbecsdk`), and this repo (`pip install --user -e .`). This machine's
  `richtech-v3` env is NOT usable here (no RealMan/Orbbec SDK, CPU-only torch) — leave it alone.
- **This is a SHARED machine.** The home dir holds unrelated projects (ZED, VR teleop,
  LinkerHand gripper endurance). Touch only `~/Dex_Elevator`, `~/pyorbbecsdk`, `~/.local`.
- **The Jetson has NO internet** (default route → dead gateway `192.168.11.1`; its WiFi
  reaches DNS but no external host). `pip`/`apt`/`git clone` fail on the robot — download
  on the dev Mac and rsync the artifacts over.
- **`sudo` requires a password**, so any root step has to be handed to the user.
- **Torso lift column** (RealMan lift API, mm) raises/lowers the upper body. The
  chest camera and arm bases ride it together, so `base_T_camera` stays constant.
- **LinkerHand** dexterous hand as end-effector (driven by its own bridge, not the
  RealMan gripper) — pressing uses a fixed "pointing" hand pose set out-of-band.
- **Two Orbbec** cameras: **Gemini 335** (chest, button camera) + **335L** (head,
  scene). Distinguish them by device name suffix ("335" vs "335L") or serial.
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

Button detector (YOLO) — train in the robot's dedicated **`ultralytics`** conda env
(torch + CUDA); **not** `richtech-v3` (no torch there, and installing it would break
its pinned `numpy<2`). Bootstrap from public **CC BY** Roboflow exports.

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
- The panel plane `(point, outward_normal)` in the base frame is measured once for
  the docked pose (config); `fit_panel_plane_from_depth` is a stub for a live fit.

**Button detection** (`yolo/button_detector.py`): `ButtonDetector` (Ultralytics
YOLO, single class `button`) → `Detection` list; `centroid_pixel` gives the press
pixel; `read_floor_label` (**stub**) is the "which floor" reader — the real open
problem, deliberately decoupled from the geometry so it can be swapped freely.

**Button-YOLO training tooling** (`yolo/prepare_dataset.py`, `yolo/train_buttons.py`,
`yolo/buttons.yaml`): no in-house dataset exists yet, so the detector is bootstrapped from
public **CC BY** Roboflow exports. **Multi-class by default** — each floor symbol keeps its own
label so the model **detects AND identifies** which floor (`prepare_dataset.py` keeps the source
labels + pHash-dedups into `data/datasets/buttons/`; `--collapse` is an opt-in single-`button`
mode). `train_buttons.py` trains YOLO11 → `data/weights/buttons.pt` — **use `--model yolo11m.pt`**;
nano is far too weak for 368-class floor ID (floor mAP50 0.1–0.37 vs yolo11m 0.7–0.85). Run both in
the robot's **`ultralytics`** conda env (torch+cuda; `richtech-v3` has no torch). Attribution in
`DATASETS.md`. Fine-tune on our own cam_chest captures later for higher accuracy.

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
  identifies each floor) → `data/weights/buttons.pt` (already deployed on the Orin).
  **Not runnable on the Orin yet**: torch/ultralytics are not installed there and the robot
  has no internet, so the wheels must be fetched on the dev Mac and rsync-ed over. Expect to
  need a TensorRT engine (exported on the Orin itself — engines are not portable) or a drop
  to `yolo11s` for real-time. Then: fine-tune on our own cam_chest captures.
- Implement `read_floor_label` (OCR / multi-class / template) — the identification step.
- Measure `elevator.panel` + press poses; consider a live plane fit later.
- Force-limited press (`rm_force_position_move_pose`) and the LinkerHand pointing pose.

Runtime artifacts are gitignored (weights, calibration outputs, captures under
`data/**`). Don't commit them.
