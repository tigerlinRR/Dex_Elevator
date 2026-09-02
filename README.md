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
| Camera (scene) | Orbbec Gemini 335L (head) | aimed steeply DOWN — sees the robot's own arms and the near floor, not the door or standing people; uncalibrated and currently unused |
| Compute | NVIDIA Jetson AGX Orin (`ssh dex5-wired`) | JetPack 6.2 / CUDA 12.6; runs the SDKs + inference |
| Detection | Ultralytics YOLO11m | multi-class (per floor: `1`,`2`,`B1`,`G`…) — detects AND identifies; trained from CC BY data (`yolo/`, `DATASETS.md`) |
| Inference | **TensorRT FP16** on the Orin's GPU | 43 ms/frame (23 FPS) via `yolo/trt_detector.py`; the machine's torch is CPU-only and is deliberately left alone |
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
├── yolo/                   # TensorRT GPU detector, panel-layout matcher, dataset prep & training
├── calibration/            # one-time intrinsics + eye-to-hand base_T_camera (ChArUco)
├── initialization/         # bring-up check, calibration + eval scripts, and press_buttons.py (the presser)
├── configs/                # cameras.yaml, pipeline.yaml, panels.yaml
├── elevator_runner/        # drive-to-the-panel tool (AutoXing cloud API + press)
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

## Reaching the robot

Two routes, both to the same machine (`~/.ssh/config` on the dev Mac):

| alias | path | throughput | when |
|---|---|---|---|
| `dex5-wired` | Ethernet cable, `192.168.11.31` | 54 MB/s | bulk rsync, `--web` MJPEG preview |
| `dex5-ts` | **Tailscale**, `100.122.187.11` | 6 MB/s | everything else — works from any network, no cable |

Tailscale was verified cable-free on 2026-08-21 (`en0` down, wired alias timing out) by
running a real two-button press over it. The path is the peer's public endpoint, not the
`192.168.11.x` link. Latency (~110 ms) does not affect motion safety: `move_joints_sync` /
`move_line_sync` poll for arrival **on the robot**, so the control loop is local — only the
command and the log cross the network. That would NOT hold for anything closing a visual
servo loop on the Mac.

The robot **does have internet** (via its WiFi), so `pip install` works directly on it — the
old "download on the Mac and rsync it over" workaround is no longer needed.

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
python3 initialization/press_buttons.py 1 4 2 5 --go --lift  # let the torso track the panel
python3 initialization/press_buttons.py 1 --go --push=2    # override push depth (mm)
```

Each press is `home → standoff (50 mm) → linear approach through contact → retract → home`,
with the joint-interpolated path checked for panel clearance beforehand. The panel plane and
the button 3D positions are **measured live** from the current camera frame every run — nothing
about the panel's position is stored, because the base docks with centimetres of error.

Before anything moves, the script closes the LinkerHand into a fist and **verifies it by
reading the joints back** — with the fingers extended the hand, not the plunger, is the
front-most part of the end-effector. If the hand does not answer or does not close, the press
refuses to move (`--no-fist` is for a robot with no hand fitted).

With `--lift`, the torso goes to each button's **best-margin height** rather than staying put
(`arm.lift.objective: margin`). The body visibly tracks the panel row by row, and the heights
it picks carry the largest joint margins available. Both objectives choose only from poses that
already passed the joint-limit, wrist, self-collision and clearance checks.

## Usage — driving the base in a straight line

The chassis has a local LAN API (`192.168.25.25:8090`) that takes wheel velocities
directly, bypassing AutoXing's cloud planner. That matters because the cloud's only
abstraction is "go to this point" and its planner prefers to turn, while entering and
leaving an elevator has to be straight in and straight out. Here the angular velocity is
pinned to zero, so straight is structural rather than lucky.

```bash
python3 initialization/drive_straight.py state        # read-only: mode, estop, alerts
python3 initialization/drive_straight.py clearance    # read-only: lidar corridor
python3 initialization/drive_straight.py move 2.0     # forward 2 m (signed metres)
python3 initialization/drive_straight.py move -- -2.0 # back 2 m
python3 initialization/drive_straight.py turn 30      # turn 30 deg in place (signed)
python3 initialization/drive_straight.py turn --corrections 1 -- -90
python3 initialization/drive_straight.py calibrate    # achieved vs commanded distance
```

Measured over `+-2 m`: **1.5–2.9 cm of distance error, 9–43 mm lateral, under 1.2° of
heading change**. Before any motion it checks the chassis serial number (this network
carries other people's robots) and gates on the **lidar** clearance in the direction of
travel; a stall aborts on battery current (30 A into an obstacle vs ~5 A driving); and
every exit path brakes and restores `auto` mode, because leaving the base in `remote`
silently disables AutoXing navigation.

Two live browser previews help position the robot by hand, both read-only:

```bash
python3 initialization/lidar_preview.py   # top-down lidar, port 8011
python3 initialization/aim_preview.py     # chest camera + button detection, port 8010
```

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

## Usage — GPU inference (TensorRT)

On the Orin, `torch` is the generic aarch64 **CPU-only** wheel, so Ultralytics needs 491 ms per
frame. It is deliberately not replaced — that install is shared with another project on the
machine. Instead the model runs through JetPack's own TensorRT, which does not involve torch:

```python
from yolo.trt_detector import TrtButtonDetector
det = TrtButtonDetector()          # data/weights/buttons_fp16.engine
dets = det.detect(bgr)             # 43 ms/frame — same Detection list as ButtonDetector
```

Build the engine once **per machine** (engines are tuned for the specific GPU and are not
portable; ~10 min):

```bash
PYTHONPATH=~/Dex_Elevator/.pydeps-onnx python3 -c \
  'from ultralytics import YOLO; YOLO("data/weights/buttons.pt").export(
       format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)'
/usr/src/tensorrt/bin/trtexec --onnx=data/weights/buttons.onnx \
    --saveEngine=data/weights/buttons_fp16.engine --fp16
```

## Status

**Pressing works (2026-08-20).** From a fixed home pose the robot locates the panel and its
buttons, then presses a requested sequence. Running `press_buttons.py 1 4 2 5 --go` lit **4 of
4** buttons in 47.9 s, with depth error ≤0.05 mm and lateral error ≤0.28 mm against the
commanded contact point — the whole chain (intrinsics, hand-eye extrinsic, live plane fit,
plunger TCP, IK) agreeing at once. Reproduced twice.

Scope is deliberately fixed there: the base does not move. The button pixels came from Hough
circles at the time, so a failed press could only be a geometry or motion problem — since
**2026-08-25 they come from the detector** (see below).

Measured, not assumed:

| quantity | value |
|---|---|
| button protrusion above the faceplate | 2.3 mm (from depth) |
| push depth | 3 mm (1 and 2 mm failed to light the button; 3 mm lit it repeatably) |
| usable standoff | ≤50 mm — beyond that the target falls inside the arm's unreachable inner region |
| plunger TCP | `[26.0, −1.9, 24.7] mm`, two independent methods agreeing to 0.7 mm |

**Driving and pressing, end to end (2026-08-27).** `elevator_runner/` drives the robot to the
elevator point and presses. Two things that used to make it look broken are fixed, both
measured on real drives:

| | before | after |
|---|---|---|
| arrival declared by | the cloud's task status | the base's own `moveState` + `speed` + position |
| ...on one side-by-side drive | 75 s | **51 s** |
| startup after arrival | 8.3 s | **1.6 s** — the press is pre-warmed during the drive |
| torso while pressing | still | **moves to each button's best-margin height** |
| `1 4 2 5` after driving | 51.2 s, 4/4 lit | **71.2 s, 4/4 lit** |

The base reports its state **event-driven**: parked, its timestamp goes 60 s without moving,
but the report following a change arrives ~2 s old. The two positions taken 24 s apart differed
by 0.6 mm, so the base really had stopped before the cloud admitted it. Arrival still requires
four conditions *and* two guards — departure must be observed first (every loop after the first
starts parked at the elevator with a stale "succeeded"), and the position must not drift between
confirmations. The cloud task status remains the backstop.

The pre-warm pays the press's ~7.3 s of position-independent startup (TensorRT engine, camera,
GPU clock ramp, arm connect) in parallel with the drive; the child then BLOCKS until the runner
says go. **The arrival decision does not move into the press** — inferring arrival from the
camera is what drove the arm into the panel once.

**Two self-sealing failures fixed (2026-08-28).** Both looked like bad luck and neither
was; in both, the instrumentation needed to tell the difference did not exist until it
was added.

*The ROI could not look where the detector had missed.* The full-frame pass scales
1280x720 into the model's 640, so a 44 px button becomes ~22 px and a whole edge row can
drop out — `open`/`close` scored **0.16 and 0.06** full-frame against **0.91 and 0.96**
on a crop of that row alone. Because the ROI is derived FROM those detections, the
refined pass never looked at the missing row, and the lattice fit got four rows for a
five-row layout: refused on 15 of 15 attempts with the panel plainly in view. Now, when
fewer rows or columns are found than the layout registers, the ROI grows by the missing
count times the measured pitch — bounded by what is missing, so a complete grid grows by
nothing. Residual on the failing frame: 11.3 -> **1.7 px**.

*"Slower than usual" and "stuck" were the same condition.* One drive took 316 s where the
same route takes 56 s (9.30 m against 8.77 m — nearly the same path at a fifth of the
speed), and a 300 s wall-clock budget gave up 6 s before the task completed with the robot
3 cm from the target. `poll_timeout_sec` is now a STALL timeout, and **progress means
motion** — position change or non-zero speed, never what the base says about its
situation. That distinction is not academic: the base once held
`moveState=moving, speed=0, hasObstruction=True` for over six minutes, parked 5 cm from
the goal, without moving. A stall inside tolerance is treated as arrival, but the task is
cancelled first so nothing can nudge the base while the arm is out.

The base's situation is now logged whenever it changes during a drive, which immediately
caught it declaring a task "succeeded" while still **95 cm** short — closed by a
corrective drive in 25 s.

*The arm refuses to move into anything (2026-08-28).* Before every button's motion, a
fresh depth frame is checked for anything sitting in front of the fitted panel plane on
the robot's side. **The chest camera is the only sensor that looks where the arm goes**:
the base's obstacle sensors face its direction of travel, 180° away from the panel, and
parked, the cloud API reports nothing about the surroundings at all — blocking the robot
for 70 s moved none of `robot_state`'s 35 scalar fields. The separation is wide: an empty
scene has **0.000%** of pixels past 30 mm (99.9th percentile +4 mm, max +21 mm — the
buttons themselves), the wall behind sits at −127 mm, and a hand in the gap reads
**+197 mm over 20.8% of the view**. Verified on hardware: button `1` pressed normally, a
hand went in, button `4` was refused with IK, clearance and self-collision all passing.
It sees only the camera's field of view, and it checks *before* the motion — not during.

*Obstacle handling verified.* Deliberately blocked with people and chairs, the base stops,
waits, re-routes and still reaches the point: a block 17 cm from the goal held it for 63 s,
after which it completed the route and arrived at **0.4 cm** (4/4 lit). A blocked path
reports as `hasObstruction`, not `hasPersonAhead`. This is normal operation on a real
floor, which is exactly why the stall timeout keys on motion rather than elapsed time — a
run that gives up because the drive was slow is one that gives up whenever anyone walks
past.

**Plunger TCP re-measured after a collision (2026-08-27).** The configured offset was 5.57 mm
out — 5.14 mm of it along the approach axis, so a commanded 3 mm push was really pressing about
8 mm. The button still lit, which is why this had to be *measured*: the press log's own
depth/lateral figures cannot see a TCP error, because the controller servos to the assumed offset
and it cancels out of both sides of that comparison. Re-measuring (tip held on a button, then the
camera measuring that button over 8 frames) and pressing `open 1 4 5 close` gave 5/5 lit at a true
3 mm push. Detection also went from 1 of 4 frames to 7 of 8 by tightening the ROI clustering and
retrying the lattice fit without its worst outlier.

**Pressing after driving — working (2026-08-26).** Re-docked three times at three
different stopping positions and pressed 4 of 4 each time; the worst docking error absorbed was
123 mm further out, 44 mm sideways and 4.5° of yaw, with no constant changed. Getting there took
two fixes: the grid check became a **lattice fit** that infers buttons the detector missed (at
that distance it drops one or two per frame), and the anchor test became **relative** — the
unshifted alignment only has to beat every shifted one, because an absolute "2 of 4 must read
correctly" refused a correct grid once the buttons shrank from 50 to 44 px.

**The lift height is searched, not computed.** A target height measured at one docking distance
picked the worst heights available once the robot parked 12 cm further out — button `1` had 0 of
24 approach rolls there while every height from 444 to 944 gave 24/24. Searching lift × roll,
preferring the least torso motion that still clears the margin, restored it to 24/24 and cut the
four-button sequence from 63 s to 51 s.

**Driving to the panel** lives in `elevator_runner/` — a Flask tool on the robot that loads
waypoints from AutoXing's cloud API, drives a route, and presses on arrival within tolerance.
Dry run by default, bound to localhost, credentials in a gitignored `.env`.

**Look low, press high — working (2026-08-26).** `press_buttons.py --lift` detects once at a
height where the camera can see the panel, then per button raises the torso so that button sits
at the arm's best-margin height, compensates the cached 3D coordinate by the achieved rise, and
presses. Button `2`, which the boundary check refuses outright at the viewing height (1.0° of
joint margin), plans at **52.8°** with all 24 approach rolls passing — and lights. Planning also
runs the controller's own self-collision model along the whole path, which matters because its
runtime check is off.

**Press poses are bounded, and the lift makes reach panel-independent (2026-08-25).**
The approach roll is still searched; `arm.limits` says which poses are disallowed. Against the
controller's real joint limits the old "least joint travel" rule been choosing poses sitting
on a hard stop — 0.1° of J3 margin for button `2`, 0.7° for `4` — and the contact pose was never
IK-checked at all. Margins are now 3.6–19.8° where they were 0.7–13.2°, verified with a 2/2 press
at the newly chosen rolls.

Panel height is absorbed by the torso lift rather than a per-panel constant:
`required_lift = command_min + 2 × (z_measured − 0.240 m)`. That makes **button centres
0.71–1.23 m off the floor** pressable, covering the ADA range of 0.89–1.22 m. Three measured
surprises are recorded in the config: the lift is on the LEFT controller, its reported position is
**twice** the real travel, and the blocking move call hangs forever past the travel limit.

**Detection drives the press (2026-08-25).** `press_buttons.py` gets button positions from the
detector and floor labels from a registered layout in `configs/panels.yaml`; both panel-specific
constants (the label grid and the hand-measured ROI) are gone. The ROI is now derived from the
detections by clustering, which also keeps the robot's own arm out of the depth plane fit.

The buttons the model reads confidently (`open`/`close`/`alarm`/`empty`) anchor the grid, and a
mismatch **refuses the press** — verified by feeding an inverted layout (anchors 0/4, refused).
That guard exists because the failure it catches is silent: a shifted grid yields perfectly
plausible coordinates and the wrong floor gets pressed with no error anywhere.

Result on hardware: 2/2 pressed, depth error 0.09 mm, lateral 0.06 / 0.23 mm, and the same
roll/travel/clearance the Hough version planned. Head to head over 20 live frames the detector
localises to ±0.3 px and survived a lighting drift that took Hough from working to 1/20.

**Model retrained on five merged public datasets (2026-08-25)** — 7,178 de-duplicated images
(CC BY 4.0 and MIT only, so commercially usable), all-class mAP50 **0.30 → 0.676**, common
floors 0.89–0.97, and on our own panel **3 of 8 markings correct → 5 of 8**. Trained on the
Orin's GPU in ~8.9 h inside an isolated venv, leaving the machine's shared CPU torch untouched.

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

**Button detection runs on the GPU (2026-08-20)** — 43 ms/frame (23 FPS) end-to-end through a
TensorRT FP16 engine, against 491 ms for Ultralytics on the machine's CPU-only torch. At 23 FPS
against a 30 FPS camera, continuous perception is no longer performance-bound; the parked-base
restriction now exists only because of detection *accuracy*, not speed.

The robot is **fully standalone** — camera, arm, hand, pressing and detection all run on the
Orin, with the dev Mac acting only as a terminal.

Also working: `initialization/bringup_check.py` (10 s readiness self-test), live panel-plane
fitting (4.8 ms, repeats to 1.1 mm), LinkerHand O6 control on both arms, Orbbec serial/name
selection, the `--web` browser capture UI.

Left/right arm **verified 2026-08-14**: `.32` and `.33` are identical RM_65s, so the
mapping was confirmed by hand-pushing the right arm while polling both — `.33` moved
34.21°, `.32` moved 0.01°. RIGHT = `192.168.11.33`.

Still to do:
- **Fine-tune on our own cam_chest captures** — the remaining accuracy work.
  Measured per button on our panel, the CC BY baseline is right on every marking that is
  legible (`alarm` 0.97, `close` 0.98, `open` 0.93, `2` 0.86, blank disc correctly `empty`)
  and wrong only where the marking is barely in the image (the laser-etched `5`/`6`/`3`/`4`).
  Exposure is not the lever — **light direction is**: killing the room light raised per-button
  contrast 18.2 → 26.0 and made those digits human-legible for the first time, because an
  etched digit is a shadow feature that broad overhead light fills in. **Adding a light to the
  robot was ruled out (2026-08-21)**, so nothing may depend on reading those digits. The design
  instead: positions from YOLO (reliable), floor labels from each elevator's layout registered
  once at commissioning, and the buttons the model *does* read confidently (`open`/`close`/
  `alarm`) used as anchors to verify that layout is aligned — with a refusal to press if they
  disagree, because a mislabelled button is otherwise a **silent** wrong-floor press.
  Registering a layout is not the same as hard-coding the panel's position in space, which
  stays live-measured.
- Implement **`read_floor_label`** (the "which floor" reader) — currently a stub.
- ~~**Press after driving and re-docking**~~ — DONE. Verified 2026-08-26 at three different
  stopping positions (worst docking error absorbed: 123 mm further out, 44 mm sideways,
  4.5 deg of yaw), and again 2026-08-31 as one command end to end: `elevator_runner/liverun.py
  BBB` drives `BBB` -> `elevator test` and presses `1 4 2 5`, **3 of 3 runs with 4/4 buttons**,
  arriving 0.9-2.3 cm from the elevator point. Only the elevator point's docking accuracy
  matters — the same runs stopped 3-68 cm from the *start* waypoint with no effect on the press.
- Visual confirmation that a press registered: the plunger occludes the button, and the exposure
  that makes digits legible saturates the indicator lamp, so the two need different exposures.
- Force-limited press (RealMan `rm_force_position_move_pose`) — optional now that the spring
  provides the compliance.
