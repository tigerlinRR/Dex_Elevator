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
```

The mobile base docks **roughly** in front of the panel (repeatable, not precise);
the chest camera + right arm + torso lift absorb the docking error, so the base
stays put during a press. Localization accuracy comes from geometry + calibration;
the vision only needs to give a reliable button pixel and its floor label.

## Hardware

| Component | Choice | Notes |
|---|---|---|
| Arm (pressing) | **RealMan RM**, right arm `192.168.11.43` | `Robotic_Arm` SDK; `core/robot/realman.py` |
| Torso lift | vertical column (RealMan lift API) | reaches buttons at different heights |
| End-effector | LinkerHand dexterous hand | driven separately; a fixed "pointing" pose presses |
| Camera (buttons) | **Orbbec Gemini 335** (chest) | `core/camera/orbbec.py` (pyorbbecsdk v2) |
| Camera (scene) | Orbbec Gemini 335L (head) | navigation; not used for pressing yet |
| Compute | NVIDIA Jetson Thor (`ssh dex4`) | runs the SDKs + inference |
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
├── yolo/                   # ButtonDetector + floor-label reader (stub) + dataset prep & training (YOLO11)
├── calibration/            # one-time intrinsics + eye-to-hand base_T_camera (ChArUco)
├── initialization/         # calibration + eval scripts (intrinsics, extrinsics, validate, localization eval)
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
# On the robot (Jetson): the RealMan SDK (`Robotic_Arm`) and pyorbbecsdk v2 are
# already installed in the `richtech-v3` env. pyorbbecsdk is built from source:
#   https://github.com/orbbec/pyorbbecsdk
```

The RealMan / Orbbec / Ultralytics SDKs are optional at import time — the modules
degrade cleanly when a SDK is absent (so this imports on a dev laptop), and
hardware code runs on the robot.

## Usage — hand-eye calibration

Run on the robot (`ssh dex4`) with the RealMan SDK env
(`~/miniconda3/envs/richtech-v3/bin/python`). The robot is **headless** (no
monitor), so add `--web` to stream a live MJPEG preview and open it in a browser
on the same LAN — e.g. from the Mac at `http://<robot-ip>:8010/`. Capture with the
page's Capture / Delete / Finish buttons (or keys `c`/`d`).

```bash
# 0. list connected cameras / confirm which is 335 vs 335L
python -m core.camera.orbbec

# 1. intrinsics for the chest camera (hold the ChArUco board, cover the frame)
python initialization/calibrate_intrinsics.py --camera cam_chest --web

# 2. eye-to-hand extrinsic: board bolted to the bare flange (LinkerHand removed),
#    drag-teach to 10+ varied poses
python initialization/run_calibration.py --camera cam_chest --web

# 3. (optional) end-to-end localization accuracy at fresh poses
python initialization/eval_localization.py --camera cam_chest --web
```

See `calibration/README.md` and `initialization/README.md` for details, and
`CLAUDE.md` for the architecture and gotchas.

## Usage — button detector (YOLO)

No in-house elevator dataset exists yet, so the detector is bootstrapped from public **CC BY**
datasets (Roboflow Universe) — **multi-class** (each floor keeps its own label, so it detects
AND identifies which floor), then fine-tuned on our own cam_chest captures later. Train on the
robot in its dedicated `ultralytics` conda env (torch + CUDA; `richtech-v3` has no torch).
Attribution: `DATASETS.md`.

```bash
PY=~/miniconda3/envs/ultralytics/bin/python
# 1. keep each floor label + pHash de-dup  (add --collapse for a single `button` class instead)
PYTHONPATH=~/Dex_Elevator $PY yolo/prepare_dataset.py --src "<export_dir>"   # or --zip a.zip
# 2. train (use yolo11m — nano is too weak for 368-class floor ID) → data/weights/buttons.pt
PYTHONPATH=~/Dex_Elevator $PY yolo/train_buttons.py --model yolo11m.pt --imgsz 640
```

## Status

**Hand-eye calibration DONE & validated** (chest 335 ↔ right arm): intrinsics
(RMS 0.31 px) + eye-to-hand extrinsic (consistency 2.05 mm, validation PASS). An
end-to-end check (`eval_localization.py`) measured the camera→base-frame position
error at **~2.3 mm mean (3.6 mm max)** over fresh poses — well within button
tolerance. Saved matrices are in `data/calibration/` (see
`cam_chest_calibration_summary.txt`).

Also working: RealMan adapter, press geometry, Orbbec serial/name selection, the
`--web` browser capture UI, the orchestrator skeleton — all import + checks pass headless.

Still to do (marked in-code):
- **Fine-tune the button YOLO** on our own cam_chest captures for higher accuracy. A
  **multi-class** baseline (yolo11m, detects + identifies each floor; common-floor val mAP50 ≈ 0.7–0.85)
  is already trained → `data/weights/buttons.pt` (see `yolo/`, `DATASETS.md`).
- Implement **`read_floor_label`** (the "which floor" reader) — currently a stub.
- **Measure the panel plane** and press poses in `configs/pipeline.yaml` (placeholders).
- Force-limited press (RealMan `rm_force_position_move_pose`); the LinkerHand pointing pose + its fingertip TCP.
