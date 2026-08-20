# Dex_Elevator

Vision-guided **elevator button pressing** for the Richtech **DEX** robot. A
chest-mounted camera detects the floor buttons, and the right arm physically
presses the requested one — so the robot can ride ordinary elevators between
floors, the way a person does.

> Why physical pressing (not a cloud elevator API): in the US a service robot is
> not permitted to command an elevator's central floor dispatch, so the arm must
> press the buttons itself.

## Pipeline

```
  chest Orbbec 335 (RGB + depth)
        │
        ▼
  YOLO button detect ─ single class `button` → boxes/masks in pixels
        │        └─ read floor label (OCR / classifier)  ── which floor is which
        ▼
  select the button whose label == requested floor
        │
        ▼
  ray ∩ panel-plane ─ button pixel → 3D contact point on the panel (base frame)
        │        └─ approach axis = inward panel normal (depth-independent)
        ▼
  eye-to-hand extrinsic  base_T_camera ── contact point → RIGHT-arm base frame
        │
        ▼
  RealMan right arm ── standoff · press · retract   (base does NOT move; lift sets height)
        │
        ▼
  spring plunger on the flange ── presses the button (3 mm past the face)
```

The mobile base docks **roughly** in front of the panel (repeatable, not precise);
the chest camera + right arm + torso lift absorb the docking error, so the base
stays put during a press. Localization accuracy comes from geometry + calibration;
the vision only needs to give a reliable button pixel and its floor label.

## Hardware

| Component | Choice | Notes |
|---|---|---|
| Arm (pressing) | **RealMan RM 65**, right arm `192.168.11.33` | `Robotic_Arm` SDK; `core/robot/realman.py` |
| Torso lift | vertical column (RealMan lift API) | reaches buttons at different heights |
| End-effector (pressing) | **spring plunger** bolted to the flange | rigid, no joints to damage; the spring gives compliance without force control |
| End-effector (other) | LinkerHand O6 dexterous hand | still fitted and controllable (`core/hand/linkerhand.py`), but not what presses |
| Camera (buttons) | **Orbbec Gemini 335** (chest) | `core/camera/orbbec.py` (pyorbbecsdk v2) |
| Camera (scene) | Orbbec Gemini 335L (head) | navigation; not used for pressing yet |
| Compute | NVIDIA Jetson AGX Orin (`ssh dex5-wired`) | JetPack 6.2 / CUDA 12.6; runs the SDKs + inference |
| Detection | Ultralytics YOLO11m | multi-class (per floor: `1`,`2`,`B1`,`G`…) — detects AND identifies; trained from CC BY data (`yolo/`, `DATASETS.md`) |
| Calibration | intrinsics + eye-to-hand (ChArUco) | once per camera, shared (`calibration/`) |
| Press geometry | ray ∩ panel-plane | pure geometry, no learned model (`core/press.py`) |

## Repository layout

```
Dex_Elevator/
├── core/
│   ├── types.py            # CameraFrame, CameraIntrinsics, Detection
│   ├── transforms.py       # SE3 helpers, pixel deprojection, quaternion <-> matrix
│   ├── config.py           # YAML config loading
│   ├── press.py            # button-press geometry (ray ∩ panel-plane, waypoints)
│   ├── elevator_pipeline.py# orchestrator: capture → detect → match floor → press
│   ├── camera/             # Camera interface + Orbbec driver (serial/name select) + manager
│   └── robot/              # RobotArm interface + RealMan adapter + Sim (headless)
│   └── hand/               # LinkerHand O6 over the arm's tool-side Modbus RS485
├── yolo/                   # ButtonDetector + Hough-circle button finder + dataset prep & training (YOLO11)
├── calibration/            # one-time intrinsics + eye-to-hand base_T_camera (ChArUco)
├── initialization/         # bring-up check, calibration + eval scripts, and press_buttons.py (the presser)
├── configs/                # cameras.yaml, pipeline.yaml
└── data/                   # weights/, calibration/ (gitignored)
```

**Interface-first**: each stage depends on an abstract base (`Camera`, `RobotArm`),
so hardware is swappable and the logic is testable without a robot
(`core/robot/sim.py`). Press geometry is pure functions in `core/press.py`.

## Install

```bash
pip install -e .                 # core + opencv
pip install -e ".[vision]"       # + ultralytics (YOLO)
# On the robot (Jetson AGX Orin) there is NO conda env — the system python3 already
# has numpy + cv2 4.8 (contrib: aruco + calibrateHandEye) + pyyaml, and the rest is
# `pip install --user`: Robotic_Arm 1.1.6, pyorbbecsdk 2.1.2 (built from source at
# https://github.com/orbbec/pyorbbecsdk), and this repo via `pip install --user -e .`.
# The robot has no internet: download wheels/sources on the dev Mac and rsync them over.
```

The RealMan / Orbbec / Ultralytics SDKs are optional at import time — the modules
degrade cleanly when a SDK is absent (so this imports on a dev laptop), and
hardware code runs on the robot.

## Usage — hand-eye calibration

Run on the robot (`ssh dex5-wired`) with the plain system `python3` — no env to
activate. The robot is **headless** (no monitor), so add `--web` to stream a live
MJPEG preview and open it in a browser on the same LAN — from the Mac at
`http://192.168.11.31:8010/`. Capture with the page's Capture / Delete / Finish
buttons (or keys `c`/`d`).

> The Mac reaches that address over the **Ethernet cable** to the robot. There is no
> route from the Mac's subnet to the Jetson's WiFi subnet, so keep the cable plugged
> in while calibrating.

```bash
# 0. readiness self-test: libs, configs, board, camera capture, arm pose (read-only)
python3 initialization/bringup_check.py

# 0b. list connected cameras / confirm which is 335 vs 335L
python3 -m core.camera.orbbec

# 1. intrinsics for the chest camera (hold the ChArUco board, cover the frame)
python3 initialization/calibrate_intrinsics.py --camera cam_chest --web

# 2. eye-to-hand extrinsic: board bolted to the bare flange (LinkerHand removed),
#    drag-teach to 10+ varied poses
python3 initialization/run_calibration.py --camera cam_chest --web

# 3. (optional) end-to-end localization accuracy at fresh poses
python3 initialization/eval_localization.py --camera cam_chest --web
```

See `calibration/README.md` and `initialization/README.md` for details, and
`CLAUDE.md` for the architecture and gotchas.

## Usage — pressing buttons

Calibration must be done first. Without `--go` the script only locates the panel and
plans, printing every target — always dry-run before letting it move.

```bash
python3 initialization/press_buttons.py 1 4 2 5            # plan only, no motion
python3 initialization/press_buttons.py 1 4 2 5 --go       # press the sequence
python3 initialization/press_buttons.py 1 --go --push=2    # override push depth (mm)
```

Each press is `home → standoff (50 mm) → linear approach through contact → retract → home`,
with the joint-interpolated path checked for panel clearance beforehand. The panel plane and
the button 3D positions are **measured live** from the current camera frame every run — nothing
about the panel's position is stored, because the base docks with centimetres of error.

## Usage — button detector (YOLO)

No in-house elevator dataset exists yet, so the detector is bootstrapped from public **CC BY**
datasets (Roboflow Universe) — **multi-class** (each floor keeps its own label, so it detects
AND identifies which floor), then fine-tuned on our own cam_chest captures later. Train on the
robot in a dedicated env with torch + CUDA. Attribution: `DATASETS.md`.

> **Not yet runnable on the AGX Orin unit**: torch/ultralytics are not installed there and the
> robot has no internet, so the wheels have to be fetched on the dev Mac and rsync-ed over. The
> commands below are from the retired Thor unit and are kept as the reference recipe.
> `data/weights/buttons.pt` is already deployed. On Orin, `yolo11m` will likely need a TensorRT
> engine (exported on the Orin itself) or a drop to `yolo11s` for real-time.

```bash
PY=~/miniconda3/envs/ultralytics/bin/python
# 1. keep each floor label + pHash de-dup  (add --collapse for a single `button` class instead)
PYTHONPATH=~/Dex_Elevator $PY yolo/prepare_dataset.py --src "<export_dir>"   # or --zip a.zip
# 2. train (use yolo11m — nano is too weak for 368-class floor ID) → data/weights/buttons.pt
PYTHONPATH=~/Dex_Elevator $PY yolo/train_buttons.py --model yolo11m.pt --imgsz 640
# 3. (optional) browser check — upload an image, see boxes + floor labels
PYTHONPATH=~/Dex_Elevator $PY yolo/predict_server.py --port 8011   # open http://<robot-ip>:8011/
```

## Status

**Pressing works (2026-08-20).** From a fixed home pose the robot locates the panel and its
buttons, then presses a requested sequence. Running `press_buttons.py 1 4 2 5 --go` lit **4 of
4** buttons in 47.9 s, with depth error ≤0.05 mm and lateral error ≤0.28 mm against the
commanded contact point — the whole chain (intrinsics, hand-eye extrinsic, live plane fit,
plunger TCP, IK) agreeing at once. Reproduced twice.

Scope is deliberately fixed there: the base does not move, and the button pixels come from
Hough circles rather than YOLO. Both were held constant so that a failed press could only be a
geometry or motion problem.

Measured, not assumed:

| quantity | value |
|---|---|
| button protrusion above the faceplate | 2.3 mm (from depth) |
| push depth | 3 mm (1 and 2 mm failed to light the button; 3 mm lit it repeatably) |
| usable standoff | ≤50 mm — beyond that the target falls inside the arm's unreachable inner region |
| plunger TCP | `[26.0, −1.9, 24.7] mm`, two independent methods agreeing to 0.7 mm |

**Hand-eye calibration DONE & validated (2026-08-14)** (chest Gemini 335 ↔ right arm
`192.168.11.33`):

| | result |
|---|---|
| Intrinsics | RMS **0.383 px** (63 of 88 views kept), mean reprojection 0.253 px |
| Eye-to-hand extrinsic | method PARK, consistency **0.67 mm** / rot 0.290°, validation **PASS** |
| End-to-end localization | **median 0.9 mm, mean 1.9 mm** over 8 fresh poses |

Better than the retired Thor robot on every comparable metric (its baseline: 0.31 px,
2.05 mm, 2.3 mm mean). Outlier rejection was worth a lot — a single unsettled pose was
degrading the extrinsic solve 3x; rejected views/samples are kept alongside for traceability.
`base_T_camera` is end-effector-independent, so the end-effector was swapped from the
LinkerHand to the plunger **with no re-calibration** — but keep the ChArUco board for future
remounts and spot-checks.

Also working: `initialization/bringup_check.py` (10 s readiness self-test), live panel-plane
fitting (4.8 ms, repeats to 1.1 mm), LinkerHand O6 control on both arms, Orbbec serial/name
selection, the `--web` browser capture UI.

Left/right arm **verified 2026-08-14**: `.32` and `.33` are identical RM_65s, so the
mapping was confirmed by hand-pushing the right arm while polling both — `.33` moved
34.21°, `.32` moved 0.01°. RIGHT = `192.168.11.33`.

Still to do:
- **Fine-tune the button YOLO on our own cam_chest captures.** The CC BY baseline (yolo11m,
  multi-class, common-floor val mAP50 ≈ 0.7–0.85) labels our embossed brushed-steel buttons
  `empty`, so the label→button mapping in `press_buttons.py` is still a hard-coded grid. That
  mapping is the last hard-coded thing and it needs to go.
- Implement **`read_floor_label`** (the "which floor" reader) — currently a stub.
- **Press after driving and re-docking** — everything is already measured live per approach, but
  it has never been tried.
- Visual confirmation that a press registered: the plunger occludes the button, and the exposure
  that makes digits legible saturates the indicator lamp, so the two need different exposures.
- Force-limited press (RealMan `rm_force_position_move_pose`) — optional now that the spring
  provides the compliance.
