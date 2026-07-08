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
| Detection | Ultralytics YOLO | button detector — needs trained weights |
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
├── yolo/                   # ButtonDetector + floor-label reader (stub)
├── calibration/            # one-time intrinsics + eye-to-hand base_T_camera (ChArUco)
├── initialization/         # calibration entry scripts (intrinsics, extrinsics, validate)
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

## Usage — hand-eye calibration (the first real step)

Run on the robot (`ssh dex4`) with the RealMan SDK env:

```bash
# 0. list connected cameras / confirm which is 335 vs 335L
python -m core.camera.orbbec

# 1. intrinsics for the chest camera
python initialization/calibrate_intrinsics.py --camera cam_chest

# 2. eye-to-hand extrinsic: ChArUco board on the right hand, drag-teach 10+ poses
python initialization/run_calibration.py --camera cam_chest
```

See `calibration/README.md` and `initialization/README.md` for details, and
`CLAUDE.md` for the architecture and gotchas.

## Status

Working: RealMan adapter, eye-to-hand calibration flow, press geometry, Orbbec
serial/name device selection, the orchestrator skeleton — all import + unit-checks
pass headless.

Still to do (marked in-code):
- Train the **button YOLO** on captured data → `data/weights/buttons.pt`.
- Implement **`read_floor_label`** (the "which floor" reader) — currently a stub.
- **Measure the panel plane** and press poses in `configs/pipeline.yaml` (placeholders).
- Force-limited press on hardware (RealMan `rm_force_position_move_pose`).
- Set the LinkerHand "pointing" pose.
