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
| `cam_arm` | **planned** — rides the right arm | `mount: arm` in `cameras.yaml` (commented out until fitted) | the right arm's **TCP** (`gripper_T_camera`) |

Because the chest camera and the arm base ride the torso lift **together**,
`base_T_camera` is a fixed constant — calibrate once, share at runtime.

An **arm-mounted** camera is the opposite: there is no constant `base_T_camera`
at all, so step 2 is replaced by `run_calibration_arm_cam.py` (eye-IN-hand) and
the runtime composes `base_T_gripper(t) @ gripper_T_camera` per frame. Which of
the two a saved `.npy` holds is recorded in a `<cam>.frame` sidecar and checked
at load — see §2b.

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

## 2b. Extrinsics (eye-IN-hand) — `run_calibration_arm_cam.py`

**Only for a camera mounted ON the arm** — the chest camera keeps using step 2,
unchanged. The setup is the mirror image: the board is **fixed in the scene** and
the arm carries the camera around it, solving `gripper_T_camera`.

```bash
python3 initialization/selftest_eye_in_hand.py                              # synthetic check, no hardware
python3 initialization/run_calibration_arm_cam.py  --camera cam_arm --web   # capture + solve
python3 initialization/eval_localization_arm_cam.py --camera cam_arm --web  # end-to-end accuracy
```
→ `data/calibration/cam_arm.npy` + `cam_arm.frame` (the tag that stops the runtime
reading it as a fixed camera's `base_T_camera`).

Watch the HUD's **`axis spread`** while capturing: `AX = XB` gets the camera's
translation only from the ROTATION between poses, so a set of mostly-translation
poses — the natural way to hand-guide an arm — is ill-conditioned, and one whose
rotations share an axis leaves that direction unobservable. The solve still
returns a tidy matrix; only the conditioning check sees it, and it **fails**
rather than warns. Keep `axis spread` above 0.15 and rotate about different axes.

Also specific to this mount: let the arm **settle** before each capture (the pose
error now corrupts the camera pose itself), and never change the tool frame
afterwards — `gripper_T_camera` is relative to whatever `get_tcp_pose()` reports.
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
