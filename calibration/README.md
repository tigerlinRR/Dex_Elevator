# calibration

Camera calibration is done **once per camera** and the resulting matrices are
**shared by the whole runtime** (loaded once by `CameraManager`, reused by the
press pipeline). It has two parts:

| Part | Output | Produced by | Loaded by |
|------|--------|-------------|-----------|
| **1 — Intrinsics** | `data/calibration/<cam>_intrinsics.npz` (K + distortion) | `initialization/calibrate_intrinsics.py` | the Orbbec driver (authoritative over SDK factory values) |
| **2 — Extrinsics** (fixed camera) | `data/calibration/<cam>.npy` (4×4 `base_T_camera`) | `initialization/run_calibration.py` | `CameraManager` → `handle.extrinsic`, used by the press pipeline |
| **2b — Extrinsics** (arm-mounted camera) | `data/calibration/<cam>.npy` (4×4 `gripper_T_camera`) + `<cam>.frame` tag | `initialization/run_calibration_arm_cam.py` | `CameraManager` → `handle.base_T_camera(arm_pose)` |

Part 2 **depends on** Part 1: the extrinsic solve detects the board with Part 1's
K + distortion. So calibrate intrinsics first, then extrinsics.

```
calibration/
├── board.py        # shared ChArUco detection (CharucoDetector API) for both parts
├── intrinsic.py    # PART 1: IntrinsicCalibration (SDK factory OR cv2.calibrateCamera)
├── extrinsic.py    # PART 2: eye-to-hand solver + capture session (camera FIXED)
├── eye_in_hand.py  # PART 2b: eye-IN-hand solver + conditioning checks (camera ON THE ARM)
├── frames.py       # which frame a saved .npy is in (base vs gripper) — checked at load
├── io.py           # shared persistence (intrinsics npz, extrinsic npy + sidecar)
└── validate.py     # post-calibration validation (reprojection residuals, etc.)
```
Self-contained — depends only on `cv2`, `numpy` and `core/` (plots optionally use
`matplotlib`; numbers work headless). Arm-agnostic: the solver takes
`base_T_gripper` + `cam_T_target` pose lists, so it is unchanged across arms.

**Default board:** 14×9 ChArUco, 20 mm squares, 15 mm markers, `DICT_5X5_100`
(matches the lensboy guide). Override with `--squares-x/--squares-y/--square-len/
--marker-len/--dictionary` if yours differs.

**Live capture controls** (a preview window opens showing the camera + detected
board): **`c`** capture · **`d`** delete last · **`Esc`** finish. Over SSH / no
display, add `--headless` for a prompt-based fallback. Captured frames/samples are
written immediately to a **gitignored** dir under `data/calibration/`, so a crash
mid-capture loses nothing.

## Part 1 — intrinsics (once, first)
```bash
# proper ChArUco calibration (cv2.calibrateCamera) — best distortion estimate
python initialization/calibrate_intrinsics.py --camera cam_chest
# or just store the Orbbec factory intrinsics (fast, no validation)
python initialization/calibrate_intrinsics.py --camera cam_chest --from-sdk
```
Views are saved to `data/calibration/<cam>_intrinsics_images/`; result to
`<cam>_intrinsics.npz`. `intrinsics_file` in `configs/cameras.yaml` is already
wired so the camera loads it instead of re-reading the SDK each start.

## Part 2 — extrinsics / eye-to-hand (once, after Part 1)
Setup: camera fixed on the torso, ChArUco board **rigidly on the right hand**.
Hand-guide the arm to ≥10 varied poses; press `c` at each to pair `base_T_gripper`
(robot) with `cam_T_target` (board pose via Part 1 intrinsics). The solver feeds
the *inverted* gripper poses into `cv2.calibrateHandEye`, yielding `base_T_camera`.
```bash
python initialization/run_calibration.py --camera cam_chest
```
The chosen extrinsic is saved as a 4×4 `.npy` (auto-loaded by `CameraManager`),
with a `<name>.calib.npz` sidecar (every method + residuals). Each accepted sample
is written to `data/calibration/<name>_samples/` so a crashed run isn't lost and
the solver can be re-run offline:
```bash
python initialization/run_calibration.py --camera cam_chest --from-samples data/calibration/cam_chest_samples
```
Aim for a translation consistency residual of a few mm.

> Detection uses OpenCV's modern `cv2.aruco.CharucoDetector` API (the older
> `interpolateCornersCharuco` / `estimatePoseCharucoBoard` were removed in 4.7+).
> PARK is the default hand-eye method — the reliable one for angled/down-looking
> cameras; the multi-method table is a diagnostic and `select_best` guards it.

**Measure your board.** `--square-len` / `--marker-len` (meters) must match the
printed board or the whole calibration scales wrong.

## Part 2b — extrinsics / eye-IN-hand (only for an ARM-MOUNTED camera)
Use this **instead of** Part 2 when the camera is bolted to the arm. Part 1 is
unchanged (intrinsics are a property of the lens, not the mount).

The setup is the **mirror image** of Part 2, and swapping the two produces a
clean-looking answer that is wrong by the length of the arm:

| | Part 2 — eye-to-hand | Part 2b — eye-IN-hand |
|---|---|---|
| board | bolted to the **flange** | **fixed in the scene** |
| camera | fixed on the torso | rides the arm |
| solved | `base_T_camera` (a constant) | `gripper_T_camera` (a constant) |
| runtime | use it directly | `base_T_gripper(t) @ gripper_T_camera`, **every frame** |
| OpenCV | gripper poses **inverted** first | gripper poses passed **as-is** |
| must be rigid | `gripper_T_board` | `base_T_board` |

```bash
python3 initialization/run_calibration_arm_cam.py --camera cam_arm --web
# offline re-solve, no hardware:
python3 initialization/run_calibration_arm_cam.py --camera cam_arm \
    --from-samples data/calibration/cam_arm_samples
```

Three things this geometry needs that the fixed one does not:

- **Rotate about more than one axis.** `AX = XB` recovers the camera's
  *translation* only from the rotation between poses, so a pose set that is mostly
  translation is ill-conditioned and one whose rotations share a single axis
  leaves the translation along that axis **unobservable** — and the solver still
  returns a tidy matrix. `rotation_diversity()` scores it, the capture HUD shows
  it live (`axis spread`, want > 0.15), and `validate_eye_in_hand` **fails** a
  near-collinear set rather than warning.
- **Let the arm settle before every capture.** The camera is now on the arm, so a pose
  read while it is still micro-swinging corrupts the camera pose itself.
- **Never change the tool frame afterwards.** `gripper_T_camera` is relative to
  whatever `get_tcp_pose()` reports (which already carries the controller's 130 mm
  tool z). The active tool frame is printed at startup — record it.

**Frame tags.** Both parts save a bare 4×4, so `<cam>.frame` records which one it
is and `CameraManager` refuses a file whose tag disagrees with `mount:` in
`cameras.yaml`. Files with no tag are read as `base` (every existing eye-to-hand
calibration), so nothing needs migrating.

**Check the maths before the hardware exists:**
```bash
python3 initialization/selftest_eye_in_hand.py   # synthetic, needs cv2, no camera/arm
```

## Validation (always run after calibrating)
Both calibration scripts validate automatically at the end and write a report
under `data/calibration/`. To re-validate later (no hardware needed):
```bash
python initialization/validate_calibration.py --camera cam_chest
```
It reuses the saved artifacts — intrinsic views in `<cam>_intrinsics_images/` and
extrinsic samples in `<cam>_samples/` — so re-checking is reproducible. Exit code
is non-zero if any check **fails** (handy for CI).

**Intrinsics checks** (from the [lensboy guide](https://robertleoj.github.io/lensboy/calibration_guide.html)):
mean/worst **reprojection error** (px) after 5σ outlier filtering, per-frame RMS,
Gaussian **MAD σ**, 2D residual **isotropy**, **spatial edge-vs-center bias**,
image **coverage**, and **split-half cross-validation**. Plots are written when
matplotlib is installed. Default pass band ≤ 0.5 px (warn), > 1.0 px (fail) — tune
in `IntrinsicThresholds`.

**Extrinsics checks:** `gripper_T_target` **consistency** residual (mm + deg),
**leave-one-out** hold-out, and **cross-method spread**
(TSAI/PARK/HORAUD/DANIILIDIS agreement). Default pass band ≤ 5 mm (warn), > 15 mm
(fail) — tune in `ExtrinsicThresholds`.

**Eye-in-hand checks** (`validate_eye_in_hand`, `EyeInHandThresholds`): the same
three against `base_T_board` (the constant in this geometry), **plus** the
rotation-axis conditioning check above. Note its consistency residual is **not
comparable** to the fixed-camera one — it also absorbs the arm's forward-kinematics
and joint repeatability error, because the camera now rides through them. Expect it
to be the larger of the two, and re-tune the thresholds against the first real
calibration instead of reading a warning as a defect.
