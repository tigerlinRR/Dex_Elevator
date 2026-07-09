# PROGRESS

Build status of Dex_Elevator. Read with `CLAUDE.md` (which explains how the code
works) to pick up where we are. Engineering status only — keep it current; not a work log.

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
- **Hand-eye calibration DONE** (`cam_chest` chest 335 ↔ right arm base):
  - Intrinsics: 80 views, RMS 0.31 px → `data/calibration/cam_chest_intrinsics.npz`.
  - Eye-to-hand extrinsic: 28 drag-teach poses, `base_T_camera`, consistency 2.05 mm, validation PASS
    → `data/calibration/cam_chest.npy` (readable: `cam_chest_calibration_summary.txt`).
  - End-to-end localization (`initialization/eval_localization.py`): ~2.3 mm mean / 3.6 mm max at fresh poses.
  - Camera↔arm-base is rigid ⇒ calibration holds across base motion + hand remount; no re-calibration.
- **Deployed**: on `dex4` at `~/Dex_Elevator` (rsync from the Mac — repo is private),
  editable-installed in the `richtech-v3` conda env (has RealMan + pyorbbecsdk + cv2 + numpy<2).
- Cameras: chest 335 serial `CP0E8530000V`, head 335L `CP2G853000BS` (pinned in `configs/cameras.yaml`).

## Not done yet (stubs / TODO)
- [ ] Train button YOLO (single class `button`) on the robot's `~/dataset` → `data/weights/buttons.pt`.
- [ ] Implement `read_floor_label` (OCR / template / multi-class) — the "which floor" reader.
- [ ] LinkerHand "pointing" pose + register its **fingertip TCP** (needed to command a press point).
- [ ] First autonomous press of a KNOWN point (slow, guarded) — biggest untested risk.
- [ ] Force-limited press (`rm_force_position_move_pose`).
- [ ] Measure `elevator.panel` plane + press poses in `configs/pipeline.yaml` (currently placeholders).
- [ ] Wire the full pipeline end-to-end; base docking (AutoXing) to a repeatable pose.

## Next-step order
1. **Motion/press first** (arm is ready, needs no elevator): LinkerHand fingertip TCP →
   known-point press → force-limited press → measure a panel plane.
2. **Perception in parallel** (uses existing `~/dataset`): train button YOLO → `read_floor_label`.
3. **Integrate**: full pipeline + base docking.

## Testing note — no real elevator panel yet (on order)
Almost everything can proceed on a **mock panel**: a rigid flat board, mounted vertically at
button height, with cheap arcade/momentary push buttons (real press feel) and/or printed
button labels. The `ray ∩ plane` geometry doesn't care if it's a real elevator. Only the real
panel's plane re-measurement + force/lighting fine-tuning need the actual hardware.

## How to run (on the robot)
```
ssh dex4 && cd ~/Dex_Elevator
PY=~/miniconda3/envs/richtech-v3/bin/python
$PY -m core.camera.orbbec                                   # list cameras
$PY initialization/calibrate_intrinsics.py --camera cam_chest --web   # browser preview (headless)
$PY initialization/run_calibration.py      --camera cam_chest --web
$PY initialization/eval_localization.py    --camera cam_chest --web
```
`--web` serves a browser preview at `http://192.168.11.41:8010/` (robot has no monitor).
Push code to the robot: rsync from the Mac (repo is private, so `git clone` on the robot fails).
```
rsync -az --exclude='.git' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='README.zh-CN.md' --exclude='乘梯相关接口.pdf' \
  ~/Desktop/Dex_Elevator/ dex4:Dex_Elevator/
```
