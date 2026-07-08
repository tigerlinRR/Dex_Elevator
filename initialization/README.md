# Initialization

One-time setup for the **chest camera + right arm**, done in order before the
press pipeline can turn a button pixel into a base-frame motion:

```
1. Intrinsics  ─┐
2. Extrinsics   ├─ calibration (initialization/*calibration*.py)
3. Validate    ─┘
```

Order matters: extrinsics detects the board **using the intrinsics from step 1**,
so calibrate intrinsics first.

## What's being calibrated

| Camera | Role | Selected by | Extrinsic is to |
|---|---|---|---|
| `cam_chest` | torso Gemini 335, faces the panel — **button camera** | `match_name: "335"` (or a serial) in `cameras.yaml` | the **RIGHT** arm base |
| `cam_head`  | head Gemini 335L — navigation/scene | `match_name: "335L"` | (not calibrated for pressing yet) |

Because the chest camera and the arm base ride the torso lift **together**,
`base_T_camera` is a fixed constant — calibrate once, share at runtime.

## Prerequisites

- Run **on the robot** (`ssh dex4`) with the RealMan SDK env, e.g.
  `~/miniconda3/envs/richtech-v3/bin/python`. Cameras plugged in (Orbbec udev
  rules installed).
- A **ChArUco board** (default 14×9, 20 mm squares, 15 mm markers, `DICT_5X5_100`)
  attached rigidly to the **right hand**.
- The richtech stack must **not** be commanding the right arm during extrinsic
  capture (it uses drag-teach). Don't change the arm's active tool frame mid-capture.
- A **DISPLAY** for the live preview, or add `--headless` for a prompt-based loop over SSH.
- List connected cameras / confirm selection: `python -m core.camera.orbbec`.

---

## 1. Intrinsics — `calibrate_intrinsics.py`

Per-camera **K + distortion**. Run first (Part 2 uses it).

```bash
python initialization/calibrate_intrinsics.py --camera cam_chest     # add --headless over SSH
```
Live preview: `[c]` capture, `[d]` delete last, `[Esc]` finish.
→ `data/calibration/cam_chest_intrinsics.npz`

## 2. Extrinsics (eye-to-hand) — `run_calibration.py`

Eye-to-hand `base_T_camera` (chest camera → right-arm base). Attach the board to
the right hand; hand-guide the arm (drag-teach) to ≥10 varied poses, press `c` at
each to pair `base_T_gripper` (robot) with `cam_T_target` (board via step-1 intrinsics).
The solver feeds the *inverted* gripper poses into `cv2.calibrateHandEye`.

```bash
python initialization/run_calibration.py --camera cam_chest              # drag-teach ON by default
python initialization/run_calibration.py --camera cam_chest --no-manual  # keep servos locked
```
→ `data/calibration/cam_chest.npy` (PARK solve, reliable for down-/angled cameras),
plus a `cam_chest.calib.npz` sidecar (every method + residuals). Each accepted
sample is saved under `data/calibration/cam_chest_samples/`, so a crashed run can
be re-solved offline (no hardware):

```bash
python initialization/run_calibration.py --camera cam_chest \
    --from-samples data/calibration/cam_chest_samples
```
Aim for a translation consistency residual of a few mm.

> The adapter converts the RealMan pose (meters + radians) to a matrix via the
> SDK's own euler↔quaternion helpers, so the Euler axis order is never guessed. A
> large residual on the first solve is the tell-tale of a convention/units problem.

## 3. Validate (recommended)

```bash
python initialization/validate_calibration.py    --camera cam_chest        # reprojection / consistency
python initialization/verify_calibration_live.py  --camera cam_chest --live # motion-free vision-vs-FK check
```

## Outputs at a glance

| Step | Artifact |
|---|---|
| Intrinsics | `data/calibration/cam_chest_intrinsics.npz` |
| Extrinsics | `data/calibration/cam_chest.npy` (+ `.calib.npz` sidecar, `_samples/`) |
| Validation | `data/calibration/cam_chest_extrinsic_validation.json` |
