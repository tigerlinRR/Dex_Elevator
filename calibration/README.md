# calibration

Camera calibration is done **once per camera** and the resulting matrices are
**shared by the whole runtime** (loaded once by `CameraManager`, reused by the
press pipeline). It has two parts:

| Part | Output | Produced by | Loaded by |
|------|--------|-------------|-----------|
| **1 — Intrinsics** | `data/calibration/<cam>_intrinsics.npz` (K + distortion) | `initialization/calibrate_intrinsics.py` | the Orbbec driver (authoritative over SDK factory values) |
| **2 — Extrinsics** | `data/calibration/<cam>.npy` (4×4 `base_T_camera`) | `initialization/run_calibration.py` | `CameraManager` → `handle.extrinsic`, used by the press pipeline |

Part 2 **depends on** Part 1: the extrinsic solve detects the board with Part 1's
K + distortion. So calibrate intrinsics first, then extrinsics.

```
calibration/
├── board.py        # shared ChArUco detection (CharucoDetector API) for both parts
├── intrinsic.py    # PART 1: IntrinsicCalibration (SDK factory OR cv2.calibrateCamera)
├── extrinsic.py    # PART 2: eye-to-hand solver + capture session
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
