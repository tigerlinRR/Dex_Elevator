# PROGRESS

Build status of Dex_Elevator. Read with `CLAUDE.md` (which explains how the code
works) to pick up where we are. Engineering status only — keep it current; not a work log.

## ▶ Migrated to the new AGX Orin DEX (2026-08-13) — READY TO CALIBRATE
The old DEX (Jetson **Thor**, right arm fault 4104) is retired. Everything is now brought up on
the **new AGX Orin DEX** and `initialization/bringup_check.py` passes end to end, so the next
action is the hand-eye calibration itself.

**New machine.** Jetson **AGX Orin Developer Kit**, JetPack **6.2** (R36.4.4), CUDA 12.6, 61 GB
RAM, 915 GB NVMe. Host `ubuntu`, user `jetson`. It is a **shared machine** — the home dir holds
unrelated projects (ZED, VR teleop, LinkerHand gripper endurance). Touch only `~/Dex_Elevator`,
`~/pyorbbecsdk`, and `~/.local`.

**Addressing changed from the Thor unit** (`4x` → `3x`):

| | Thor (old) | Orin (new) |
|---|---|---|
| Jetson | `192.168.11.41` | `192.168.11.31` (wired `eno1`), WiFi `192.168.10.146` |
| Left arm / Right arm | `.42` / `.43` | `.32` / **`.33`** (right = pressing arm) |
| Chest 335 / Head 335L serial | `CP0E8530000V` / `CP2G853000BS` | **`CP0BB5300041`** / **`CP2G8530000W`** |

SSH aliases are in the dev Mac's `~/.ssh/config`: **`dex5`** (WiFi) and **`dex5-wired`** (wired).

**Runtime environment — NOT conda.** Unlike the Thor unit (`richtech-v3`), this machine's
`richtech-v3` has no RealMan/Orbbec SDK, CPU-only torch, and numpy 2.x. The elevator stack runs
on the **system `python3` (3.10.12)** instead, which already ships numpy 1.21.5 + **cv2 4.8.0
(aruco `CharucoDetector` + `calibrateHandEye` present)** + pyyaml. Extra packages go in with
`pip install --user` — no sudo, no conda, and `richtech-v3` is left untouched.

| Component | How it is installed |
|---|---|
| `Robotic_Arm` 1.1.6 (RealMan) | `pip install --user` from the PyPI wheel (`py3-none-any`) |
| `pyorbbecsdk` **2.1.2** (OrbbecSDK 2.9.3) | built from source with CMake in `~/pyorbbecsdk`, then `pip install --user` |
| `dex_elevator` (this repo) | `pip install --user -e .` → `easy-install.pth` |

**⚠ The Jetson has NO internet.** Its default route points at `192.168.11.1`, which does not
answer, and the WiFi (`Richtech_Tech`, gateway `192.168.10.1`) reaches DNS but no external
host. So `pip`/`apt`/`git clone` all fail on the robot. Everything above was installed by
**downloading on the dev Mac and rsync-ing the artifacts over**. Fix the route (see below)
before trying to install torch/ultralytics.

**HAND-EYE CALIBRATION IS DONE on this machine (2026-08-14)** — see the section below for numbers.

**Stop point (2026-08-14): everything is in place for a first press except the wrist
orientation.** Resuming Monday. Next actions, in order:

1. **Wrist orientation.** Parked by hand, the fingertip pointed 46.9° off the panel's
   inward normal — pressing like that would skid across the face rather than push.
   Reachability is orientation-limited too: sweeping roll about the approach axis at
   the panel, only 3 of 12 directions had an IK solution, so the press orientation has
   to be picked from what IK accepts. Turn the wrist (by hand is safest) to within
   ~15° of the normal, then the remaining correction is a **31.5 mm translation** —
   small and low-risk.
2. **First press, in stages.** standoff 50 → 30 → 10 mm with `push_depth=0`,
   photographing at each step, then add depth. Never been done; biggest untested risk.
   The photo at 10 mm also resolves the last open number: the fingertip TCP's z has a
   17 mm disagreement between the URDF derivation and the visual measurement (x/y agree
   to 0.5–1.8 mm). Take the LARGER z until settled — too small overshoots into the panel.
3. **YOLO** — deliberately last. It only automates "which pixel is the button", and
   that pixel can be supplied by hand today. Wiring it earlier would confound
   perception errors with geometry errors. Needs torch/ultralytics fetched on the Mac
   and rsync-ed over (robot has no internet), then likely a TensorRT engine or `yolo11s`.
4. Route fix + Tailscale (the installed one belongs to `tony.h@` and is offline).
5. Force-limited press, full pipeline, base docking.

**Practical notes for Monday:**
- The mock panel is a real elevator faceplate on a fire-extinguisher cabinet; the `1`
  button is the green one, 4th row left. Panel sits ~0.68 m in front of / 0.25 m right
  of the arm base, vertical to within 1.4°.
- **Don't let the hand into the plane-fit ROI** — it dragged the fit by 23 mm. Use the
  faceplate's left half until YOLO supplies a proper button-box ROI.
- Lighting matters: the first captures came out at mean brightness 40–50 and the button
  labels were unreadable. With the room light on it was fine.

**Left/right arm VERIFIED (2026-08-14).** `.32`/`.33` are identical RM_65s, so the mapping was
confirmed physically: with both arms polled, hand-pushing the right arm moved `.33` by 34.21°
and `.32` by 0.01°. RIGHT = `192.168.11.33`, as configured. (It also showed the right arm is
not servo-locked — it hand-drags freely, which suits the drag-teach capture in step 2.)

## Done & validated
- **Framework** (interface-first; imports on a laptop with SDKs absent, runs on the robot):
  - `core/robot/realman.py` — RealMan RM adapter: connect, `get_tcp_pose`, `move_to_pose`,
    `move_line_to_pose`, lift get/set, drag-teach (`set_manual_mode`). Right arm `.43`.
  - `core/camera/orbbec.py` — Orbbec driver + `serial`/`match_name` device selection
    (chest 335 / head 335L). `python -m core.camera.orbbec` lists connected devices.
  - `core/transforms.py` — SE3 helpers + `quat_to_matrix`/`matrix_to_quat` (RealMan pose conv).
  - `core/press.py` — button-press geometry (pixel → ray ∩ panel-plane → contact pose → standoff/press/retract waypoints). Pure functions.
  - `yolo/button_detector.py` — `ButtonDetector` + `read_floor_label` (**STUB**) + `centroid_pixel`.
  - `core/elevator_pipeline.py` — orchestrator (capture → detect → match floor → press).
  - `calibration/` — intrinsics + eye-to-hand, driven off the RealMan adapter; `--web`
    browser MJPEG capture (`capture_ui.web_capture_loop`) for the headless robot.
- **HAND-EYE CALIBRATION DONE & VALIDATED on the AGX Orin unit (2026-08-14)** — `cam_chest`
  (chest Gemini 335 `CP0BB5300041`) ↔ **right arm** `192.168.11.33`:
  - **Intrinsics**: 88 views captured, **63 kept** after outlier rejection → RMS **0.383 px**,
    mean reprojection 0.253 px. `fx 692.76  fy 692.81  cx 641.44  cy 366.29`,
    `dist [0.01626, -0.07119, 0.00056, 0.00034, 0.04941]` → `cam_chest_intrinsics.npz`.
    Cross-checks against the camera's own factory intrinsics (689.3/689.3/640.5/367.1) to 0.9%.
  - **Eye-to-hand extrinsic**: 41 drag-teach poses, **40 kept** (one bad sample, see below),
    method **PARK**, consistency **0.67 mm** / rot 0.290°, validation **PASS** → `cam_chest.npy`.
  - **End-to-end localization** (`eval_localization.py`, 8 fresh poses): **median 0.9 mm,
    mean 1.9 mm**, min 0.3, max 8.3 (a single outlier; 7 of 8 sat around 1 mm).
  - Board rigidity: `flange_T_board` spread only **0.61 mm median** across the kept samples —
    the board did not shift during capture.
  - **Better than the retired Thor unit on every comparable metric** (its baseline: intrinsics
    RMS 0.31 px, extrinsic consistency 2.05 mm, end-to-end 2.3 mm mean / 3.6 mm max).
  - Camera↔arm-base is rigid ⇒ calibration holds across base motion + hand remount; the ChArUco
    board can be removed and the LinkerHand refitted with NO re-calibration. **Keep the board** —
    it is needed again after any camera/arm remount or collision, and for spot-checks via
    `verify_calibration_live.py`.
  - **Outlier rejection mattered a lot.** Rejected artifacts are kept for traceability in
    `cam_chest_intrinsics_rejected/`, `cam_chest_samples_rejected/`, plus pre-rejection backups
    `cam_chest_intrinsics_before_outlier_reject.npz` and `cam_chest_all41.{npy,calib.npz}`:

    | | before | after |
    |---|---|---|
    | intrinsics RMS | 0.698 px (88 views) | **0.383 px** (63 views) |
    | extrinsic consistency | 1.85 mm (41 poses) | **0.67 mm** (40 poses) |
    | extrinsic leave-one-out worst | 33.6 mm | **1.81 mm** |

    A *single* bad pose (`sample002`, 34 mm off — captured before the arm settled) was degrading
    the whole extrinsic solve by 3x. Find them by reconstructing `flange_T_board` per sample and
    looking at the spread; the bad ones stick out by an order of magnitude.
- **LinkerHand O6 under control (2026-08-14)** — `core/hand/linkerhand.py`, driven as a
  Modbus RTU slave on the right arm's tool-side RS485 (`port=1, slave=0x27`, 115200,
  24 V via `rm_set_tool_voltage(3)`). `POSES["point"]` verified on hardware: index
  extended, other four curled, reaches target in ~2 s. Fingertip TCP taken from
  LinkerBot's official O6 URDF + STL meshes → `[11.88, 28.08, 172.87] mm` from the
  flange face; the same computation reproduces the manual's 177 mm middle-fingertip
  figure to 0.1 mm. Note the fingertip is 28 mm OFF the hand's centre axis — a pure-z
  TCP would miss a 20 mm button by more than its own diameter.
- **Panel plane now fitted LIVE** — `fit_panel_plane_from_depth` implemented (was a
  stub): RANSAC + least-squares over the depth ROI, **4.8 ms**, repeats to **1.1 mm**
  frame-to-frame, 93 % inliers, 1.0 mm residual. The stored plane in
  `configs/pipeline.yaml` is now reference-only.
- **Full pixel → 3D geometry chain verified on the real panel.** Projecting the 9
  button pixels through `ray ∩ plane` gives a **column-to-column spacing of 56.6 mm
  that is identical across all five rows** (row pitch 43.4 mm). Consistency at that
  level across independent rows is only possible if the hand-eye extrinsic, the plane
  fit and the ray-plane maths are all correct simultaneously.
- **Deployed on the Orin unit** at `~/Dex_Elevator` (rsync from the Mac — repo is private),
  editable-installed into the **system python3** via `pip install --user -e .`.
  `initialization/bringup_check.py` passes: libs, configs, ChArUco board, chest-camera capture
  (RGB 1280x720 + 93 % valid depth), and a read-only pose from the right arm.
- Cameras: chest 335 `CP0BB5300041`, head 335L `CP2G8530000W` (pinned in `configs/cameras.yaml`).

## Not done yet (stubs / TODO)
- [~] **Button YOLO baseline (MULTI-CLASS)** — tooling BUILT + RUN on dex4 (`yolo/prepare_dataset.py`,
      `yolo/train_buttons.py`, `yolo/buttons.yaml`, `DATASETS.md`):
      prepared the CC BY sun-moon export (2019 imgs @416×416, 368 classes, labels **KEPT**) → pHash
      de-dup → `data/datasets/buttons/` (1408/403/200); trained **YOLO11m** in the robot's
      **`ultralytics`** conda env (torch+cuda) → `data/weights/buttons.pt`. **Detects AND identifies
      each floor.** Common-floor val mAP50 ≈ 0.7–0.85 (`1`.85 `2`.85 `3`.78 `4`.85 `5`.74 `G`.70);
      all-class mean (0.30) is dragged down by ~230 long-tail classes with 1–2 samples. Single test
      image (L/3/2/1) all identified correctly. imgsz 640; `--collapse` = old single-`button` mode.
      NOTE: yolo11n was far too weak here (floor mAP50 0.1–0.37) — use yolo11m+.
      TODO: fine-tune on our own cam_chest captures for higher accuracy; optionally add `yolov7ncku` for diversity.
      NOTE: `~/dataset` on the Jetson is an **unrelated** task — do NOT train on it, do NOT delete it.
- [ ] Implement `read_floor_label` (OCR / template / multi-class) — the "which floor" reader.
- [ ] LinkerHand "pointing" pose + register its **fingertip TCP** (needed to command a press point).
- [ ] First autonomous press of a KNOWN point (slow, guarded) — biggest untested risk.
- [ ] Force-limited press (`rm_force_position_move_pose`).
- [ ] Measure `elevator.panel` plane + press poses in `configs/pipeline.yaml` (currently placeholders).
- [ ] Wire the full pipeline end-to-end; base docking (AutoXing) to a repeatable pose.

## Next-step order
1. **Motion/press first** (arm is ready, needs no elevator): LinkerHand fingertip TCP →
   known-point press → force-limited press → measure a panel plane.
2. **Perception** (bootstrap now, no elevator needed): download public CC BY button exports →
   `yolo/prepare_dataset.py` → `yolo/train_buttons.py` → baseline `buttons.pt` → `read_floor_label`.
   Fine-tune on our own cam_chest captures once a (mock/real) panel exists.
   (`~/dataset` on the robot is a different task — not this.)
3. **Integrate**: full pipeline + base docking.

## Testing note — no real elevator panel yet (on order)
Almost everything can proceed on a **mock panel**: a rigid flat board, mounted vertically at
button height, with cheap arcade/momentary push buttons (real press feel) and/or printed
button labels. The `ray ∩ plane` geometry doesn't care if it's a real elevator. Only the real
panel's plane re-measurement + force/lighting fine-tuning need the actual hardware.

## How to run (on the robot)
Plain `python3` — no conda, no env activation (see the environment table at the top).
```
ssh dex5-wired && cd ~/Dex_Elevator
python3 initialization/bringup_check.py                                   # ~10 s readiness self-test
python3 -m core.camera.orbbec                                             # list cameras + serials
python3 initialization/calibrate_intrinsics.py --camera cam_chest --web   # 1. intrinsics
python3 initialization/run_calibration.py      --camera cam_chest --web   # 2. eye-to-hand
python3 initialization/validate_calibration.py --camera cam_chest         # 3. re-validate
python3 initialization/eval_localization.py    --camera cam_chest --web   # 4. accuracy at fresh poses
```
`--web` serves a browser preview at **`http://192.168.11.31:8010/`** (the robot has no monitor).
**The browser has to reach that address**: the dev Mac sits on `192.168.40.x` and there is no
route to the Jetson's WiFi subnet (`192.168.10.x`), so calibration currently requires the
**Ethernet cable** to the Mac (which puts the Mac on `192.168.11.50`). Fix the default route or
Tailscale to drop the cable.

Button YOLO is **not runnable on this machine yet** — torch/ultralytics are not installed and
the robot has no internet. `data/weights/buttons.pt` is already deployed for when they are.

Push code to the robot: rsync from the Mac (repo is private, so `git clone` on the robot fails).
`data/calibration/` is excluded on purpose — the Thor unit's artifacts must NOT be reused here.
```
rsync -az --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='data/calibration/' --exclude='乘梯相关接口.pdf' \
  ~/Desktop/Dex_Elevator/ dex5-wired:Dex_Elevator/
```

### Sudo-only steps (the `jetson` user needs a password for these)
```
# Orbbec USB permissions — WITHOUT THIS THE CAMERA CANNOT BE OPENED (already done 2026-08-13)
sudo sh ~/pyorbbecsdk/install/lib/pyorbbecsdk/shared/install_udev_rules.sh
sudo udevadm control --reload-rules && sudo udevadm trigger

# Give the Jetson internet: its default route points at the dead gateway 192.168.11.1
sudo nmcli con mod "Wired connection 1" ipv4.never-default yes
sudo nmcli con up "Wired connection 1"

# Tailscale: the installed node belongs to tony.h@ and is offline
sudo tailscale logout && sudo tailscale up
```
