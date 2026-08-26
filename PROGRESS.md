# PROGRESS

Build status of Dex_Elevator. Read with `CLAUDE.md` (which explains how the code
works) to pick up where we are. Engineering status only — keep it current; not a work log.

## ▶ PRESSING WORKS (2026-08-20) — 4 buttons in a row, all lit
The robot now presses real elevator buttons autonomously. From a fixed home pose it locates the
panel and the buttons from the chest camera, then presses a requested sequence:

```
python3 initialization/press_buttons.py 1 4 2 5 --go     # 4/4 lit, 47.9 s
```

| button | approach roll | joint travel | path clearance | depth error | lateral error |
|---|---|---|---|---|---|
| `1` | 45° | 129° | 47 mm | −3.05 mm (target −3.00) | 0.23 mm |
| `4` | 45° | 142° | 37 mm | −3.00 mm | 0.23 mm |
| `2` | 105° | 138° | 34 mm | −2.99 mm | 0.08 mm |
| `5` | 15° | 154° | 36 mm | −3.01 mm | 0.28 mm |

Depth error ≤0.05 mm and lateral error ≤0.28 mm against the commanded contact point, which is
the whole chain — intrinsics, hand-eye extrinsic, live plane fit, plunger TCP, IK — agreeing at
once. Repeated twice with identical results.

**Scope deliberately fixed here**: the base does NOT move (parked in front of the panel), and
the button pixels come from Hough circles, not YOLO. Both were held constant on purpose so that
a failed press could only be a geometry or motion problem. That paid off — every failure this
session was diagnosable.

### Pressing after driving — WORKING, and the lift is now searched (2026-08-26)

**Re-docked and pressed, three times, at three different stopping positions.** The
worst docking error absorbed was **123 mm further out, 44 mm sideways, 4.5 deg of
yaw** — all taken up by the live measurement chain with no constant changed. Depth
error stayed <=0.08 mm and lateral 0.05-0.50 mm, the same as from a fixed spot.

Two things had to be fixed first, and the first attempt failed 6 of 6 on both:

| symptom | cause | fix |
|---|---|---|
| anchors 1/4, press refused | the classifier degrades at 44 px (was 50 px), and the anchor test was ABSOLUTE ("2 of 4 must read correctly"). It refused a grid that a look at the frame showed to be perfectly correct | anchors became a RELATIVE test: the unshifted alignment only has to beat every shifted one |
| grid shape != (2,2,2,2,2) | at that distance the detector drops one or two buttons per frame | fit the button LATTICE and infer the missed cells (residual ~1.5 px); rows/cols come from the layout, so a real shift still blows the residual up |

The anchor false-positive is the one worth remembering: its failure direction is the
worst available — it refuses a CORRECT detection, and the consequence is a robot
standing in a lobby unable to act.

**The lift height is now searched, not computed.** A `target_relative_z_m: 0.240`
constant, measured at one docking distance, was picking the worst heights on offer once
the robot parked 12 cm further out: button `1` had 12 of 24 approach rolls at the
computed height and **0 of 24** one step above (so the press failed, 3/4), while EVERY
height from 444 to 944 gave 24/24 with 41-56 deg of margin. Reachability is a function
of distance AND height, and only the height axis had been measured — the same mistake
as the old "least joint travel" rule, one level up: **a variable that should be
searched had been pinned to a constant.**

Searching (lift x roll, preferring the least torso motion that still clears
`prefer_margin_deg`) fixed it and is strictly better:

| | computed height | searched |
|---|---|---|
| `1` | 0/24 rolls — FAILED | 24/24, margin 45.4 deg |
| `4` | 7/24, margin 35 deg | 24/24, margin 44.1 deg |
| `2` | 7/24, margin 35 deg | 23/24, margin 41.6 deg |
| `5` | 15/24, margin 34.5 deg | 24/24, margin 49.0 deg |
| sequence time | 62-64 s | **51.1 s** (no torso motion needed at this dock) |

Also: localisation retries raised 6 -> 15 (three runs used attempts 4, 1 and 5 of 6 —
one bad frame from failing outright, and each attempt costs 200 ms), and
`press_buttons.py --auto` added: wait until the panel is visible AND still, press once,
exit. Verified on hardware, 4/4.

### Driving to the panel: `elevator_runner/` (2026-08-26)

A Flask tool on the robot (contributed by a colleague) that loads waypoints from
**AutoXing's cloud API**, drives a route, and presses when the robot arrives within
tolerance of the point marked as the elevator. Dry run by default; binds to
`127.0.0.1:8765` so only someone SSH'd in can reach it; credentials in a gitignored
`.env`.

Tested on the robot before any live drive, which caught a defect the dry run cannot
show: **AutoXing returns a POI's position as `coordinate: [x, y]`, not top-level
`x`/`y`** — although `robot_state` DOES use top-level `x`/`y`. Every waypoint therefore
parsed to (0, 0), and the arrival gate compared the robot's distance to the map origin,
**2927 cm, against an 8 cm tolerance**. A live run would have driven to the elevator,
refused to press as "out of reach", backed off, retried and given up — with a log line
that reads like a docking-accuracy problem. Fixed, then verified by computing a real
distance: **2.8 cm** from the robot to the POI named `elevator test`.

Still to note: this map uses none of the AutoXing elevator POI types (`[6, 28]`) — the
elevator point is type 11 like any other waypoint — so the tool's auto-highlight never
fires and the operator selects the point by name.

### Look low, press high — WORKING (2026-08-26)

Contributed by a colleague; the lift is now part of the press loop. `--lift` detects
once at the current visible height, then **per button** raises the torso so that button
sits at the arm's best-margin height, compensates the cached 3D coordinate by the
achieved rise, presses, and restores the lift afterwards.

Result: button `2` — refused entirely at the viewing height (best joint margin
1.0 deg) — plans at **52.8 deg of margin with 24/24 rolls passing, and lights**.
Per-button targeting is why it beats the ~28 deg a single panel-wide height would give.

Also added: **self-collision checking at planning time**
(`rm_algo_safety_robot_self_collision_detection`, pure computation against the
controller's own model, sampled along the whole path because a configuration can pass
through a collision between two clear endpoints). The controller's runtime
self-collision check is off, so this closes that gap for the arm's own links and
end-effector — the other arm, chassis and door frame still need virtual walls.

Reviewed and fixed one defect: `clearance()` kept using the plane origin measured
BEFORE the lift moved. The panel rides the lift too, so the distance along the normal
was off by `rise * normal_z` — measured **4.7-6.4 mm** for rises of 242-328 mm.
Verified numerically before changing anything: it flipped **no** verdict (real
clearances are 45-52 mm against a 5 mm threshold) and erred conservatively, so it was
inert rather than dangerous — but it is now compensated (`origin_now`), and clearance
for `2` reads 52 mm where it read 46.

**The dominant failure mode is now detection flakiness, not reach.** In that same dry
run, 4 of 6 attempts were refused because one button was missed — grids like
`(2,2,2,1,2)`. The buttons sit on a very regular lattice (row pitch 43.4 mm / 68 px,
column 56.6 mm / 79 px, consistent to ~1 px across all five rows), so fitting that
lattice and filling in the missed cells should recover them. Not built yet.

### Press poses are now bounded, and the lift makes it panel-independent (2026-08-25)

**Boundaries, not a recipe.** The approach roll is still searched over all 24
directions; `arm.limits` in `configs/pipeline.yaml` says which poses are disallowed.
Read from the controller, the real joint limits are J1 +-178, J2 +-130, **J3 +-135**,
J4 +-178, J5 +-128, J6 +-360 — and J3 binds for every button on this panel:

| button | old margin to J3's stop | new |
|---|---|---|
| `2` | **0.1 deg** | refused (see below) |
| `4` | 0.7 | **3.6** |
| `6` | 4.1 | **9.2** |
| `3` | 3.9 | **7.6** |
| `5` | 7.0 | **12.9** |
| `A` | 13.2 | **19.8** |

Verified on hardware after the change: 2/2 pressed, depth error 0.09 mm, lateral
0.18 / 0.35 mm — at different rolls than before (`1` moved 30 -> 105, `4` 30 -> 180).

Three things this exposed:
- The old objective (least joint travel) was selecting poses **on a hard stop**.
  Button `2` had been working on 0.1 deg of margin, i.e. by luck; it is now refused,
  which is a real functional loss and the correct call.
- Maximising margin instead is not the answer — it swings the path into the panel
  (clearance -12.0 mm for `dot`). Hence constraints plus an objective.
- **The contact pose was never IK-checked.** Only the standoff was solved; the press
  was a Cartesian `movel` 50 mm further in. That is precisely where a jam comes from,
  and the controller's own self-collision check is **off** (verified).

**The lift is the fix for the thin-margin buttons, and it generalises.** What governs
reachability is the button's height in the base frame; the lift moves that frame, so
`required_lift = 444 + 2 * (z_measured - 0.240 m)` works on any elevator rather than
being a constant tuned to this faceplate. The arm is 10/10 reachable for panel-centre
heights of 0.19-0.39 m, but the margin across that band runs 2.8 deg at 0.39 to
41.4 deg at 0.19.

| lift command | base-frame origin above floor | pressable button heights |
|---|---|---|
| 444 (as found) | 516 mm | **706 - 906 mm** |
| 1100 (verified max) | 844 mm | **1034 - 1234 mm** |

So **button centres 0.71 - 1.23 m off the floor are pressable**, which covers the ADA
range of 0.89 - 1.22 m — though the top of that is reached at the thin-margin end of
the band, so high panels have less margin than low ones.

Measured facts behind this, each of which looks like something else:
- **The lift hangs off the LEFT controller.** The right one returns `pos = 0`, which
  reads exactly like "fully down" — it is actually at 444.
- **Reported position is 2x the real vertical travel** (0.5016 / 0.5021 / 0.5008 over
  three moves, measured against the camera). Command and readback agree to 1 mm and
  `err_flag` stays clean, so nothing flags it; only measuring the geometry shows the
  body moved half as far. Trusting the number would have put the compensation out by
  78 mm, on a press where 1 mm decides button versus chamfer.
- **The blocking `rm_set_lift_height` hangs forever past the limit** — 1100 holds,
  1200 never returns. Use the non-blocking form and poll until `mode != 2`.
- **Repeatability 0.09 mm at 444 and 0.24 mm at 600**, as the spread of the panel's
  measured z over four cycles — better than the press's own lateral error, which is
  what makes "look low, press high" viable.
- The base frame's origin is ~525 mm BELOW the arm's own mounting flange (a tape put
  the flange at 41 in while the origin computes to 515.9 mm), so it is not the arm
  base — but it does ride the lift, which is all the model needs. Its height was
  derived from the camera's z for the `3`/`4` row (+568.4 mm) against a tape reading
  of 42-11/16 in, and cross-checked by predicting the row spacing (86.3 mm) before
  measuring it (3-3/8 to 3-1/2 in).

Not yet done: actually moving the lift as part of a press ("look low, press high").

### Detection replaced the OpenCV stand-in (2026-08-25)

`press_buttons.py` no longer uses Hough circles or a hard-coded label grid. Both of the
panel-specific constants are gone:

| was | is now |
|---|---|
| `PANEL_LABELS` — a grid literal in the source | `configs/panels.yaml` — a registered layout, verified at run time |
| `PANEL_ROI = (955,145,1130,480)` — measured by hand | derived from the detections by clustering (`panel_roi`) |

Verified on hardware: **2/2 pressed**, depth error 0.09 mm, lateral 0.06 / 0.23 mm — and the
planner produced the *same* roll/travel/clearance as the Hough version, which cross-validates
the new positions. The auto-derived ROI came out `(955,124)-(1131,498)` against the hand-measured
`(955,145,1130,480)`.

The safety mechanism is real, not paper. Buttons the model reads confidently
(`open`/`close`/`alarm`/`empty`) anchor the grid, and a mismatch REFUSES the press:

- fed a deliberately inverted layout → anchors 0/4 → refused
- a spurious extra detection made the grid `(2,2,2,2,2,1)` → refused, retried, passed
- one anchor (`A`) misreads consistently, and `min_anchors: 2` is what tolerates it

This matters because the failure being guarded against is **silent**: a shifted grid produces
perfectly plausible coordinates and the robot would press the wrong floor with no error anywhere.

**Hough vs detector, 20 live frames each, same frames.** The showroom light drifted the faceplate
to 223 (blown out) mid-test, which turned into a robustness measurement:

| | success | at panel brightness | position repeatability | per attempt |
|---|---|---|---|---|
| Hough circles | 1/20 | succeeded at 199, failed at 223 | — (one success) | 41 ms |
| **detector** | **12/20** | succeeded at 221 | **±0.3 px** | 201 ms |

At the designed brightness both work and press accuracy is indistinguishable, so the reason to
switch is not accuracy — it is labels, the derived ROI, and light tolerance. 201 ms is 1 % of a
23 s two-button sequence. `--circles` is kept as a fallback and A/B reference.

### Model retrained on five merged datasets (2026-08-25)

| | before | after |
|---|---|---|
| sources | 1 | **5** (CC BY 4.0 + MIT only — all commercially usable) |
| images after de-dup | 2,019 | **7,178** (2,634 duplicates dropped) |
| train / val | 1,412 / 405 | **6,084 / 1,076** |
| all-class mAP50 | 0.30 | **0.676** |
| common floors mAP50 | 0.70–0.85 | **`1` .965 `2` .964 `3` .955 `4` .954 `5` .894 `6` .898 `G` .937** |
| **on OUR panel** | **3/8 markings** | **5/8** |

Trained on the Orin's GPU in ~8.9 h (yolo11m, imgsz 640, 100 epochs). 27 % of the nominal 9,812
images were duplicates — two of the five sources are the same upstream set at different versions,
and two more are another shared upstream.

The gain on our own panel is real but partial: `3` and `4` became correct (0.48 / 0.81) while
`5`, `6` and `1` are still wrong. **This softens an earlier claim of mine** — "more data cannot
help our panel because the marking is not in the image" was too strong. It holds for the worst
buttons; for `3` and `4` the information was there and only the old model could not extract it.

**GPU training on the Orin, without disturbing the shared torch**: an isolated venv
`~/venv-dex-train` holds torch 2.8.0 + torchvision 0.23.0 CUDA builds from
`pypi.jetson-ai-lab.io/jp6/cu126`. Three traps: `--extra-index-url` let pip pick PyPI's CPU
wheel instead (use `--index-url` + `--no-deps`); torch **2.10**'s CUDA build needs
`libcudss.so.0`, which JetPack does not ship, so 2.8.0 it is; and Ubuntu strips `ensurepip`, so
the venv is made with `--without-pip` and bootstrapped via `get-pip.py`.

### The robot is standalone, and YOLO now runs on its GPU (2026-08-20)

Nothing on the dev Mac is involved at run time — camera, arm, hand, pressing and button
detection all execute on the Orin. The Mac is only a terminal.

Button inference was moved off Ultralytics onto a **TensorRT FP16 engine**
(`yolo/trt_detector.py`), which changed it from "offline check only" to real-time:

| | per frame | |
|---|---|---|
| Ultralytics + the Orin's CPU-only torch | 491 ms | 2.0 FPS |
| **TensorRT FP16, end to end** | **43 ms** | **23 FPS** |
| — of which GPU compute | 6.4 ms | 157 FPS |

23 FPS against a 30 FPS camera means **continuous perception is no longer performance-bound**:
tracking the panel while the base moves, and visual servoing, are both open now. The parked-base
restriction stays only because the perception ACCURACY problem below is unsolved.

**The point of the design: torch was never touched.** `torch` on the Orin is the generic
aarch64 **CPU-only** wheel, and it lives in the shared `~/.local` where `~/mmdetection`
(a colleague's project) imports the same install — swapping in NVIDIA's JetPack build would
change another project's environment underneath it. JetPack already ships TensorRT 10.3 with
working Python bindings, and TensorRT does not involve torch, so the route is
`buttons.pt -> ONNX -> .engine` with torch used only for the one-off export.

| added | where | effect on other projects |
|---|---|---|
| `onnx` + `protobuf 5.28` | unpacked (not pip-installed) into `~/Dex_Elevator/.pydeps-onnx`, on `PYTHONPATH` only when exporting | **none** — verified that a clean cwd and `~/mmdetection` still see protobuf 3.12.4 |
| `cuda-python` 12.6.2 | `~/.local` (`pip install --user --no-deps`) | none — new package, no conflicts |
| numpy 2.2.6 | **NOT installed** | `onnx` wanted it; it would have broken cv2 4.8 and pyorbbecsdk |

The engine (`data/weights/buttons_fp16.engine`, 44.5 MB, 612 s to build) is **per-GPU and not
portable** — rebuild it after any machine or TensorRT change. Commands are in the module docstring.

The old DEX (Jetson **Thor**, right arm fault 4104) is retired; everything below runs on the
**AGX Orin DEX**.

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

**⚠ The Jetson has NO internet, and the Ethernet cable is the ONLY way in.** Re-checked
2026-08-20: the default route points at `192.168.11.1`, which does not answer, **and its WiFi
is associated with a printer's access point** (`Brother HL-L3275`, `192.222.10.133/24`) — that
is the actual reason there is no internet. The `Richtech_Tech` / `192.168.10.146` address in
earlier notes is **stale and does not respond**. The robot scans only that one SSID, and
`nmcli dev wifi rescan` returns `not authorized`, so re-pointing it needs the user's password.

So `pip`/`apt`/`git clone` all fail on the robot; everything above was installed by
**downloading on the dev Mac and rsync-ing the artifacts over**. And **pulling the Mac's cable
cuts off all remote access** — the robot itself keeps running, but no scripts, captures or
`--web` preview. To go cable-free, either put the robot's WiFi on the same network as the Mac
or fix Tailscale (both need sudo — see the bottom of this file).

Watch out when checking this from the Mac: its `en0` is `192.168.11.50` with a **/16** netmask,
so `route -n get 192.168.10.x` reports "via en0" for addresses that are not reachable at all.
Test with a real connection attempt, never a route lookup.

**HAND-EYE CALIBRATION IS DONE on this machine (2026-08-14)** — see the section below for numbers.

### What the press actually does

The end-effector is a **rigid spring plunger bolted to the flange**, NOT a LinkerHand finger —
switched 2026-08-20 because the hand's finger joints are a damage risk on contact, and because
future end-effectors may be other grippers entirely. The spring supplies the compliance that
force control would otherwise have to. TCP `[26.0, −1.9, 24.7] mm` relative to the reported TCP
frame (which already carries the controller's 130 mm tool z), calibrated two independent ways
that agree to 0.7 mm axially / 2.5 mm radially.

Per button, from `configs/pipeline.yaml`:
`home → movej to standoff 50 mm → movel through contact into the button → movel retract → movej home`,
with the joint-interpolated path checked for panel clearance (refuses anything under 5 mm).

Numbers that were **measured, not assumed**:

| quantity | value | how |
|---|---|---|
| button protrusion above faceplate | **2.3 mm** | depth ROI (faceplate −0.2, buttons +2.0…+2.3) |
| `push_depth` | **3 mm** | walked up on hardware: 1 mm and 2 mm failed to light, 3 mm lit 2/2 |
| `standoff` usable range | **≤50 mm** | 0/30/50 mm are 24/24 reachable, 80 mm+ is 0/24 |
| home pose | `[39.76, −100.13, −79.28, −131.63, 117.0, 68.12]°` | picked so the arm does not occlude the panel AND all 10 buttons stay reachable |

The home pose is one pose for both driving and pressing (operator-confirmed safe to drive with).
It was chosen over two other candidates because the arm projects entirely OUTSIDE the camera
image there — a pose that blocks the faceplate makes the perception step impossible no matter how
good the geometry is. Price paid: 117–166° of joint travel per press, vs 33–143° for the rejected
candidate.

### Next, in order

1. **Fine-tune on our own cam_chest captures** — the remaining accuracy item. The five-source
   public model now reads 5 of 8 markings on our panel (was 3/8); `5`, `6` and `1` still fail
   because their etched marks are barely in the image, and a grazing light was ruled out. The
   pipeline no longer *depends* on reading them — positions plus a registered layout carry the
   press, with the readable buttons as anchors — so this is now about tightening the safety
   margin rather than unblocking the feature. **Mix, do not replace**: training on our panel
   alone would destroy the generality that makes the anchors work anywhere else.
2. **Press after driving.** Everything is already live-measured per approach (plane fit + button
   3D), so re-docking should work without code changes — but it has never been tried.
3. **Fit the button lattice and infer missed cells.** Detection flakiness is now the
   dominant failure mode: 4 of 6 localisation attempts get refused because one button of ten
   was missed. The lattice is regular to ~1 px, so the missing cells can be predicted from
   the ones that were found. Anchors must stay real detections (verifying inferences with
   inferences is circular), inferred cells must be logged as inferred, and a poor lattice fit
   must refuse rather than extrapolate.
4. Force-limited press (`rm_force_position_move_pose`) — now optional, the spring is the
   compliance.
5. Route fix + Tailscale (the installed node belongs to `tony.h@` and is offline).

**Open limitation: a press cannot yet be confirmed visually.** The plunger occludes the button
while pressed, and the locked exposure that makes the digits legible saturates the indicator
lamp. Reading digits and seeing the lamp need *different* exposures. Options: a second lower-
exposure capture after retracting, or the head 335L from another angle. Pressing an already-lit
button is harmless, so this is not blocking.

**Practical notes on the cell:**
- The mock panel is a real elevator faceplate on a fire-extinguisher cabinet, ~0.68 m in front of
  and 0.25 m right of the arm base, vertical to within 1.4°. Layout, top to bottom:
  `A/dot`, `5/6`, `3/4`, `1/2`, `open/close`.
- **Don't let the hand or plunger into the plane-fit ROI** — it dragged the fit by 23 mm.
- Lock the camera exposure (`exposure: 156`, `gain: 16`). On auto, AE meters the mostly-dark wall
  and crushes the faceplate to ~34/255; the digits disappear entirely.
- **The exposure constant WILL have to be re-tuned on site.** The lab is a showroom and is lit
  far more strongly than a real lobby (operator-confirmed), and even within one session its light
  drifted enough to take the faceplate from 199 to 223 — where Hough collapsed to 1/20. A value
  tuned here is therefore almost certainly too dark in the field, in the opposite direction. The
  durable fix is auto-exposure metered on the panel ROI only (follows the environment without
  being dragged by the wall), which is not built yet.

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
- **AUTONOMOUS BUTTON PRESS WORKING (2026-08-20)** — `initialization/press_buttons.py`,
  4/4 of `1 4 2 5` lit in 47.9 s, depth error ≤0.05 mm, lateral ≤0.28 mm. Reproduced twice.
  See the top section for the per-button table and the measured constants.
  - **Spring plunger on the flange** replaces the LinkerHand finger as the press tool
    (no joints to damage; the spring is the compliance). TCP `[26.0, −1.9, 24.7] mm`,
    calibrated by touching the tip to a button and solving `inv(base_T_tool) @ tip_3d`,
    cross-checked against a tape measurement to 0.7 mm axial / 2.5 mm radial.
  - **Motion made synchronous** — `RealmanArm.move_joints_sync` / `move_line_sync` poll for
    arrival instead of trusting the return code (`rm_movej`/`rm_movel` return `false` while
    still executing, and silently reject a command issued mid-settle). Without this, two
    failures were misattributed entirely: a dropped `movel` reported "pressed at +50 mm",
    and a stale IK seed reported "no IK solution" for a button that is perfectly reachable.
  - **Button centres from Hough circles** (`yolo/button_circles.py`) — eyeballing the pixel was
    only ~2.5 px off but that landed the plunger on the button chamfer, which would not actuate
    even at 2 mm push. Circle centres fixed it at 2 mm; 3 mm is the production value.
  - Buttons are located ONCE per sequence and cached (base and panel are static mid-sequence);
    re-detecting per button only added failure chances.
- **GPU button inference (2026-08-20)** — `yolo/trt_detector.py`, TensorRT FP16, 43 ms/frame
  (23 FPS) vs 491 ms on the CPU-only torch, with no change to the shared torch install.
  Two results worth keeping, both of which contradicted a reasonable prediction:
  - **Zero-copy is not automatically right on a Jetson.** Mapped host buffers remove the
    copies, but the GPU then reads/writes *uncached* memory and its compute goes 6.4 -> 13.2 ms.
    Against `cudaMalloc` + pinned staging it is a near-tie (43.2 vs 44.9 ms, A/B'd twice
    interleaved) — the prediction that pinned+device would win clearly was wrong. Only plain
    pageable `cudaMemcpy` is clearly bad (42 ms of pure copying for 17 MB of tensors).
  - **Jetson benchmarks must be warmed up.** The CPU governor is `schedutil` and the GPU idles
    at 306 MHz, so the first thing measured in a process pays for the clock ramp — this
    produced a step-by-step profile summing to 10 ms for a call that took 26 ms. Discard
    120-150 iterations, and instrument the real call rather than timing steps in isolation
    (isolated timings are optimistic: the real loop's per-frame writes evict the cache).
- **Detection wired into the press path (2026-08-25)** — `yolo/panel_layout.py` +
  `configs/panels.yaml` replaced Hough circles and the hard-coded label grid. 2/2 pressed,
  0.09 mm depth error; a deliberately inverted layout is refused. See the top section.
- **Model retrained on 5 merged public datasets (2026-08-25)** — 7,178 images, all-class mAP50
  0.30 → 0.676, and 3/8 → 5/8 markings correct on our own panel.
- **Camera recovery**: the chest 335 can enumerate normally yet never deliver a colour frame
  (every `capture()` fails after 40 retries). `pyorbbecsdk`'s `Device.reboot()` clears it in
  ~25 s — no root, no replug. The head 335L working throughout is what identified it as
  device state rather than a code fault.
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
      **Does not transfer to our panel as-is**: it labels our embossed brushed-steel buttons
      `empty` (the public data is backlit plastic). Fine-tuning on cam_chest captures is now the
      top open item, since it is the only thing standing between us and dropping the hard-coded
      label→button grid in `press_buttons.py`.
- [x] ~~First autonomous press~~ — DONE 2026-08-20, see above.
- [x] ~~LinkerHand pointing pose + fingertip TCP~~ — superseded: pressing uses the spring plunger.
- [x] ~~Measure the panel plane~~ — fitted LIVE every approach; the config value is reference-only.
- [ ] Fine-tune the button model on our own cam_chest captures, then replace the hard-coded
      `PANEL_LABELS` grid in `initialization/press_buttons.py` with real detections.
- [ ] Implement `read_floor_label` (OCR / template / multi-class) — the "which floor" reader.
- [ ] Press after the base has driven and re-docked (geometry is already live-measured, untried).
- [ ] Two-stage path for `close` (14/24 rolls) and `2` (8/24) — working but thin on margin.
- [ ] Confirm a press visually (needs a second, lower-exposure capture — see the top section).
- [ ] Force-limited press (`rm_force_position_move_pose`) — optional now the spring is compliant.
- [ ] Wire `core/elevator_pipeline.py` end-to-end; base docking (AutoXing) to a repeatable pose.

## Next-step order
Motion is done; the remaining work is perception, then integration.
1. **Perception** — fine-tune on our own cam_chest captures so the model reads OUR buttons,
   then feed real detections into `press_buttons.py` in place of the hard-coded grid.
   (`~/dataset` on the robot is a different task — not this.)
2. **Integrate** — press after driving/re-docking, then the full `core/elevator_pipeline.py`.
3. **Harden** — two-stage paths for the thin-margin buttons, visual press confirmation,
   optionally force-limited press.

## Testing note — mock panel
Testing runs against a **real elevator faceplate mounted on a fire-extinguisher cabinet**, which
is enough for everything except riding an actual elevator: the buttons are the real parts with
real switch travel, and `ray ∩ plane` does not care what is behind the plane. A real installed
panel would only change the plane measurement (done live anyway) and the lighting.

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

# PRESS BUTTONS (needs calibration done). Without --go it only plans and prints targets.
python3 initialization/press_buttons.py 1 4 2 5            # dry run: locate + plan, no motion
python3 initialization/press_buttons.py 1 4 2 5 --go       # actually press
python3 initialization/press_buttons.py 1 --go --push=2    # override push depth, mm
```
`--web` serves a browser preview at **`http://192.168.11.31:8010/`** (the robot has no monitor).
**The browser has to reach that address**: the dev Mac sits on `192.168.40.x` and there is no
route to the Jetson's WiFi subnet (`192.168.10.x`), so calibration currently requires the
**Ethernet cable** to the Mac (which puts the Mac on `192.168.11.50`). Fix the default route or
Tailscale to drop the cable.

Button YOLO runs **on the GPU** through TensorRT directly (`yolo/trt_detector.py`),
43 ms/frame:

```
python3 -c "import sys; sys.path.insert(0,'.'); import cv2
from yolo.trt_detector import TrtButtonDetector
d = TrtButtonDetector(); print(d.detect(cv2.imread('some_frame.png')))"
```

The system torch/ultralytics are present but **CPU-only** (491 ms/frame) — they are used only
for the offline `buttons.pt -> ONNX` export, never in the loop. Rebuild the engine after any
machine change (it is not portable); commands are in the module docstring.

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

# Give the Jetson internet AND let it be reached without the Ethernet cable.
# Its WiFi is currently on a PRINTER's access point (Brother HL-L3275, 192.222.10.133),
# it scans only that SSID, and a user-level rescan is refused ("not authorized").
sudo nmcli dev wifi list                                    # can it even see the office WiFi?
sudo nmcli dev wifi connect "<SSID>" password "<password>"
# and stop the dead wired gateway from being the default route:
sudo nmcli con mod "Wired connection 1" ipv4.never-default yes
sudo nmcli con up "Wired connection 1"

# Tailscale: the installed node belongs to tony.h@ and is offline
sudo tailscale logout && sudo tailscale up
```
