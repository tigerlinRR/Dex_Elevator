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

The robot is an NVIDIA **Jetson Thor**, reachable at `ssh dex4` (`192.168.11.41`,
user `jetson`). Hardware, verified on the device — do NOT assume an xArm:

- **Arms: dual RealMan RM** (SDK imports as `Robotic_Arm`, class `RoboticArm`).
  LEFT `192.168.11.42`, RIGHT `192.168.11.43`, port `8080`. **Pressing uses the
  right arm.** The RealMan SDK lives in the robot's `richtech-v3` conda env
  (`~/miniconda3/envs/richtech-v3/bin/python`).
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

Run hardware commands **on the robot** (`ssh dex4`) with the RealMan env.

```bash
# list connected Orbbec devices (name + serial) — confirms 335 vs 335L
python -m core.camera.orbbec

# Calibration (once, per camera, in order): intrinsics -> extrinsics -> validate.
# The robot is HEADLESS: add --web to stream an MJPEG preview to a browser on the
# LAN (http://192.168.11.41:8010/); capture via the page buttons or keys c/d.
python initialization/calibrate_intrinsics.py --camera cam_chest --web   # 1. intrinsics (hold board)
python initialization/run_calibration.py      --camera cam_chest --web   # 2. eye-to-hand (board on flange, drag-teach)
python initialization/validate_calibration.py --camera cam_chest         # 3. re-validate saved artifacts
python initialization/eval_localization.py    --camera cam_chest --web   # end-to-end localization accuracy (fresh poses)

# No test suite or linter is configured yet.
```

Button detector (YOLO) — train in the robot's dedicated **`ultralytics`** conda env
(torch + CUDA); **not** `richtech-v3` (no torch there, and installing it would break
its pinned `numpy<2`). Bootstrap from public **CC BY** Roboflow exports.

```bash
PYU=~/miniconda3/envs/ultralytics/bin/python
PYTHONPATH=~/Dex_Elevator $PYU yolo/prepare_dataset.py --src "<roboflow_export_dir>"        # multi-class -> data/datasets/buttons/
PYTHONPATH=~/Dex_Elevator $PYU yolo/train_buttons.py --model yolo11m.pt --imgsz 640         # -> data/weights/buttons.pt
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
  so use `--web` (MJPEG to a browser on the LAN, `http://192.168.11.41:8010/`) for
  any capture — a local OpenCV window or `ssh -Y` shows nothing.
- **Calibration board mount**: bolt the ChArUco board to the bare **flange** (remove
  the LinkerHand — its fingertip TCP isn't defined in the controller). `base_T_camera`
  is end-effector-independent, so the hand is remounted afterward with no re-calibration.

## Stubs / not-yet-wired (marked in-code with `# TODO`)

- Button YOLO: tooling BUILT + a **multi-class** baseline trained (yolo11m — detects AND
  identifies each floor) → `data/weights/buttons.pt`. Remaining: fine-tune on our own
  cam_chest captures for higher accuracy.
- Implement `read_floor_label` (OCR / multi-class / template) — the identification step.
- Measure `elevator.panel` + press poses; consider a live plane fit later.
- Force-limited press (`rm_force_position_move_pose`) and the LinkerHand pointing pose.

Runtime artifacts are gitignored (weights, calibration outputs, captures under
`data/**`). Don't commit them.
