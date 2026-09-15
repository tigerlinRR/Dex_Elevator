# PROGRESS

Build status of Dex_Elevator. Read with `CLAUDE.md` (which explains how the code
works) to pick up where we are. Engineering status only — keep it current; not a work log.
## ▶ THE ARM CAMERA IS CALIBRATED, AND HAND-GUIDED CAPTURE IS THE WRONG METHOD (2026-09-14)

Both parts done on the arm-mounted Gemini 335. Intrinsics beat the in-service chest
camera on every metric; the eye-in-hand extrinsic passes validation. The result that
generalises beyond this robot is **how** the extrinsic had to be captured.

| | intrinsics | extrinsic |
|---|---|---|
| kept | **58** of 80 views | **42** samples (3 program sweeps) |
| residual | RMS **0.177 px**, mean reproj **0.152 px** | **1.60 mm** / **0.382 deg** |
| conditioning | coverage 0.979 | axis spread **0.256** (need 0.15) |
| cross-check | 4 methods agree to **0.07 mm** | z from two independent subsets: **-90.6 / -89.1 mm** |
| result | `gripper_T_camera = [15.7, -48.9, -89.1] mm` | ✓ PASS |

Against the chest camera that is flying today (0.9 mm end-to-end, 0.3 mm presses):
mean reprojection **0.152 vs 0.253 px**, cross-val **1.63 vs 3.29 px**, i.e. better on
every comparable number.

### Hand-guiding an arm produces a DEGENERATE pose set, structurally

Three hand-guided rounds — 56 samples over ~40 minutes — all failed, and not because
the operator did it badly:

| round | samples | axis spread | dominant axis (base frame) |
|---|---|---|---|
| 1 | 30 | 0.089 | [-0.08, -0.01, 1.00] — 86.7 % of the energy |
| 2a | 5 | 0.062 | [ 0.17, -0.06, 0.98] — 91.8 % |
| 3 | 21 | 0.124 | [-0.13, 0.26, -0.96] — 82.7 % |

All three are the SAME axis, base-frame Z. The cause is mechanical: **a person holds
the arm and moves it to a new place, and the orientation change that comes with that
is dominated by J1, the base rotation — which is Z.** Deliberately spinning the last
joint while holding everything else still is not a motion a hand makes. Merging the
rounds made it worse, not better: 56 single-axis samples DILUTE the one round that
had a second axis (merged ratio 0.135 < the good round's 0.243 on its own).

Program-driven joint sweeps fixed it in one pass, and the data is an order of
magnitude cleaner — per-round residual **median 2.4 mm / max 5.4** against the hand
rounds' **median 3.7-5.8 / max 33-53**. Etc. the arm settles properly when a program
waits for it.

### A pure J6 sweep cannot see along the optical axis — and the residual hides it

The camera's optical axis sits 2 deg off the TCP's z, so J6 spins the camera about its
own line of sight. That is exactly what the translation term needs, EXCEPT along that
axis: in `(R_A - I) t_X = ...` the matrix is singular along the rotation axis, so `t_z`
never enters the equations. Measured:

```
j6scan  alone  z = -81.1 mm   consistency 1.10 mm, PASS   <- both "pass"
j6scan2 alone  z = -93.1 mm   consistency 1.35 mm, PASS   <- and differ by 12 mm
j5scan  alone  z = -90.6 mm   axis spread 0.554           <- J5 turns PERPENDICULAR to it
merged         z = -89.1 mm
```

Two sweeps agreed to **0.1 mm in x and y** and disagreed by **12 mm in z**, while both
reported excellent residuals — the residual is blind to the unobservable direction.
J5 (wrist pitch) turns about an axis perpendicular to the line of sight and pins `t_z`
down; its usable range is asymmetric (the wrist sits near a limit), so the sweep
computes it from the joint limit rather than assuming symmetry — 66 deg of real travel
where a symmetric +/-15 would have been used.

### NaN passes every threshold check

A degenerate pose set makes `cv2.calibrateHandEye` return an **all-NaN** transform
rather than raising. Every validation test is a comparison and `nan > limit` is False,
so all of them fall through to "pass" and the calibration reports clean. Found by the
solver self-test; only the rotation-axis conditioning check caught it. Now refused at
the solve, in `select_best` (whose `min()` would rank a NaN first), and in the
validator independently. **The same pattern exists unfixed in the eye-to-hand path**
and is left alone deliberately — that is the calibration currently flying — but noted.

### Also
- **Outlier rejection must be judged on held-out data, not RMS.** Sweeping the
  threshold: RMS falls monotonically 0.487 -> 0.166 px as views are dropped 80 -> 50,
  but cross-validation bottoms out at **58 views (1.63 px)** and degrades after. RMS
  alone would have over-rejected by 8 views. `reject_intrinsic_outliers.py` caches the
  corner detection (the slow part) so thresholds can be tried in seconds.
- **Three cables to get a working link.** Two had dead SuperSpeed pairs (480 Mbps, half
  the bandwidth, colour dropping to 15 fps) and one failed outright mid-session. Proved
  it was the cable and not the robot: the same hub port runs 5000 Mbps with the third.
  150 s soak, 0 failures.
- The arm camera is the SAME MODEL as the chest one, so `match_name: "335"` now matches
  both — **both are pinned by serial**.

### Still open
- **End-to-end localisation accuracy has not been measured** (`eval_localization_arm_cam.py`,
  ~10 min). It separates repeatability at one viewing pose from accuracy across poses,
  which decides whether the press looks from a fixed pose or from wherever the arm is.
- **Not wired into the press.** One line in `press_buttons.py`; `core/press.py` needs no
  change. The real question behind it is where the arm STANDS to look.
- **The arm obstacle check breaks** with the camera on the arm — it works today only
  because the home pose puts the arm outside the chest camera's view. Keeping the chest
  camera for that job is the cheap answer; both mounts coexist in `cameras.yaml`.
- `gripper_T_camera`'s z has not been cross-checked against a tape measure.

## ▶ THE ARM CAMERA IS ON THE ROBOT AND SEEN; CALIBRATION IS NEXT (2026-09-14)

The camera is fitted and wired. Enumerated, captured from, and registered in the config;
the solver self-test now passes on the robot's own cv2 rather than only on the Mac's
synthetic run. **Not calibrated yet** — that needs a person with the board.

| | |
|---|---|
| arm camera | Gemini 335, serial **`CP0T263000FK`** |
| first capture | 1280x720 MJPG, depth **73-84 %** valid, median range 0.315 m |
| SDK intrinsics | fx 691.5 / cx 643.5 / cy 363.0 (chest unit: 692.76 calibrated — same model) |
| `bringup_check.py --camera cam_arm` | **passes** |
| solver self-test on the robot | exact recovery; **0.58 mm** X error at 0.5 mm / 0.1 deg detection noise; degenerate set refused |

**`match_name` stopped being enough the moment this camera went on.** It is the SAME
MODEL as the chest one, so `"335"` now matches both and the SDK would open whichever it
enumerated first — silently, and not necessarily the same one twice. `cam_arm` is pinned
by serial, and so is `cam_chest` already.

**The first frame settles a design question before calibration starts**: the plunger and
the hand occupy the lower third of the image. So the tool is inside this camera's field
of view — a plane-fit ROI must exclude it (the hand in frame dragged a chest-camera fit
by 23 mm), and the board must not be occluded by it while capturing samples.

### The self-test found a real defect, and it is the project's classic shape

A degenerate pose set makes OpenCV's `calibrateHandEye` return an **all-NaN** transform
instead of raising. That is worse than an error: every validation test is a threshold
comparison, and `nan > limit` is **False**, so all of them fall through to their "pass"
branch and the calibration reports clean. Only the rotation-axis conditioning check
caught it. Non-finite results are now refused at the solve, in `select_best` (whose
`min()` would otherwise rank a NaN first), and in the validator independently, since an
extrinsic can arrive from a saved file.

The same pattern exists in the eye-to-hand path and was **left alone deliberately** —
that is the calibration currently flying — but it is the same hazard if that solve ever
degenerates. Worth a decision rather than a silent fix.

### Next, and the first two need a person
1. Intrinsics for `cam_arm` (hold the board, cover the frame).
2. `run_calibration_arm_cam.py` — board FIXED in the scene this time, arm carries the
   camera. Watch the HUD's axis spread; keep it above 0.15.
3. `eval_localization_arm_cam.py` — measure repeatability at one pose separately from
   accuracy across poses. That number decides whether the press looks from a fixed
   viewing pose or from wherever the arm is.
4. Exposure is on **auto** for now (mean 107.9, 0.00 % saturated). Re-tune and lock it
   against the real panel before trusting button detection.

## ▶ THE ARM-MOUNTED CAMERA HAS A CALIBRATION PATH; THE FIXED ONE IS UNTOUCHED (2026-09-11)

Groundwork for moving the button camera onto the arm, which is still an **undecided**
option — so it is built as a second path that coexists with the chest camera rather than
a migration. `mount: fixed` is the default, the eye-to-hand code is byte-for-byte
unchanged, and the existing `cam_chest.npy` loads exactly as before.

**No hardware was involved. Nothing here has run on the robot.** The camera is not
fitted; what exists is the maths, the capture tooling and a synthetic proof.

### It is a different problem, not a variant of the same one

| | eye-to-hand (chest, unchanged) | eye-IN-hand (arm, new) |
|---|---|---|
| board | bolted to the flange | **fixed in the scene** |
| solved | `base_T_camera`, a constant | `gripper_T_camera`, a constant |
| runtime | use it directly | `base_T_gripper(t) @ gripper_T_camera`, **per frame** |
| OpenCV call | gripper poses **inverted** | gripper poses **as-is** |
| must stay rigid | `gripper_T_board` | `base_T_board` |

`cv2.calibrateHandEye` solves eye-in-hand natively, so the new path is the one that
*drops* a step. Everything else — board detection, pose-diversity gating, per-sample
persistence, offline re-solve — is reused by subclassing the existing session.

**`core/press.py` needs no change at all**: it is pure geometry and takes the 4x4 as an
argument, so it does not care where the matrix came from. The integration is one line in
`press_buttons.py` plus the real question behind it — where the arm stands to look.

### What the synthetic check proved (and one thing it killed)

| check | result |
|---|---|
| noiseless recovery of a known `gripper_T_camera` | exact |
| the same data through the eye-to-hand routine | **1398 mm** residual vs 0.000 mm |
| yaw-only pose set | solver returns a tidy matrix; validator **fails** it |
| `base_T_camera(X, pose) @ cam_T_target == base_T_board`, every pose | holds |

The 1398 mm is the important one. Both extrinsics are a bare 4×4 on disk and **nothing
in the file says which it is**, so a mix-up gives confident coordinates wrong by the
length of the arm. Hence `data/calibration/<cam>.frame`, checked against `mount:` at
load, failing closed. Files with no tag read as `base`, so there is nothing to migrate.

The second important one is the yaw-only set. `AX = XB` recovers the camera's
*translation* only from the ROTATION between poses — so a mostly-translation pose set
(the natural way to hand-guide an arm) is ill-conditioned, and one whose rotations share
an axis leaves that direction **unobservable while the solve still looks fine**. The
capture HUD shows the axis spread live, because that is a defect only fixable while the
operator is still standing at the robot with drag-teach on.

### Still open — and these are decisions, not ports
- **Where the arm stands to look.** The measurement now passes through the arm's FK and
  joint repeatability, which the chest camera's 0.9 mm end-to-end median never included.
  Looking from ONE OR TWO FIXED poses should absorb most of it as a systematic error;
  `eval_localization_arm_cam.py` reports repeatability-at-one-pose separately from
  accuracy-across-poses so that is settled with numbers.
- **The obstacle check breaks.** It works only because the home pose puts the arm outside
  the chest camera's view, so the depth image shows the world and not the robot's own
  limb. Keeping the chest camera fitted for that job is the cheap answer, and the two
  mounts already coexist in `cameras.yaml`.
- Exposure and working distance have to be re-tuned: both change when the camera moves.
- `EyeInHandThresholds` are engineering judgement, not measurement. Re-tune them against
  the first real calibration instead of reading a warning as a defect.

## ▶ THE LEG MODEL WAS DIRECTIONAL ALL ALONG, AND THE DEAD TIME IS GONE (2026-09-04, evening)

Seven consecutive end-to-end runs from a registered start point, **4/4 buttons every
time**. Closure came down from 37.3 cm / 9.2° in the morning to **2.8 cm / 1.1°**, and the
two waits the operator could see between phases are now **0.0 s**.

### The chassis is not symmetric, and one shared constant was hiding it

`SPEED_RATIO` was a single number applied both ways. Measured with `calibrate_legs.py`
(new: single uncorrected legs, restored between measurements, all in one process):

| commanded | +ori, base front | −ori, arm-ward |
|---|---|---|
| 0.4 m | 82.5 % | 73.0 % |
| 0.8 m | 93.2 % | 62.5 % |
| 1.2 m | 98.2 % | 61.6 % |
| fit `achieved = k·cmd − c` | k **1.061**, c +97 mm, rms 4 mm | k **0.559**, c −63 mm, rms 7 mm |

Residuals of 4 and 7 mm — a clean model, not scatter. **A slope of 1.06 against 0.56
cannot be described by one constant**, and what it hid was paid for by the correction
legs on every single move: the "it stops and then shuffles again" the operator asked
about. Corroborating it, the exit leg's FIRST attempt had come back 0.842 / 0.851 /
0.836 m of a commanded 1.400 across three earlier runs — 59.7 / 60.1 / 60.8 %.

With a per-direction model in `_leg`, single uncorrected arm-ward legs measure **100.1 /
101.2 / 100.7 %** (k 1.011, rms 3 mm). Entry and exit now each take ONE leg: −1.400 m
commanded, −1.404 and −1.405 achieved.

**The model is only applied where it was measured** (0.30–1.60 m). Extrapolated to zero
the arm-ward line claims 63 mm of travel for a commanded 0, which is nonsense, and at
2.7 m it predicted 2.768 against 2.826 measured. Shorter legs keep the old behaviour and
lean on the measure-move-measure loop, which is what absorbs centimetre corrections anyway.

Correction tolerance also went 2 → 3 cm: with the model in place the first leg lands
within a couple of centimetres, and a 2 cm tolerance then spent two or three further legs
commanding 21–24 mm and achieving **0.000** — the same stiction floor the turns have.

### The dead time between phases is gone

Measured with the new `_mark`/`_profile` instrumentation rather than guessed — and the
guess was half wrong: "entry leg done → turn starts" was **1.0 s**, not the 6 s of
redundant settles predicted. The real cost was elsewhere:

| gap | before | after |
|---|---|---|
| entry done → turn starts | 1.0 s | **0.0 s** |
| turn done → arm moves | 2.0 + 4.9 + ~7.3 ≈ **14 s** | **0.0 s** |

Three changes: `_report` stopped re-settling the pose (every leg and turn already ends by
waiting for two agreeing samples — 2.0 s to print a line); the harness stopped opening a
camera of its own; and **the press is PRE-WARMED**, started before the entry leg so its
~7.3 s of engine load, camera open, GPU ramp and arm connect overlaps the drive. That
mechanism already existed for `elevator_runner` (8.3 → 1.6 s); this harness had never
used it. Releasing it after the turn is one `token.touch()`.

The measured approach became a FALLBACK for the same reason — it needs the camera, which
the pre-warm now holds. That is a trade, stated plainly: since the leg model went in,
five consecutive runs landed the panel **0.3–4.2 cm** from target and the correction chose
not to move every time, so paying 4.9 s per run to catch a case that has not occurred is
the wrong side of it. If the press refuses, the camera is free by then and the harness
measures, nudges and retries once.

**A pre-warmed child that is never released keeps the camera open**, and the next attempt
then fails with "no color frame after 40 tries" — one step away from the cause. Every
abandoning path kills it.

### Press depth raised again, 4.5 → 6.0 mm, and where that stops being the lever

The operator still reported the plunger reaching buttons that did not light. Measured
depth is now −5.94 to −6.06 mm against a commanded −6.00, 4/4 on every run since.

**Note the limit this crosses.** The config's own comment says that past ~5 mm the
suspicion should shift from stroke to the SPRING being too soft — compressing without
transmitting force. If 6 mm still leaves buttons unlit, more depth is the wrong lever and
a stiffer spring is the fix; deeper travel only compresses the spring further and the
force at the button stops rising with it.

### The exit aim targets the ENTRY heading, not the middle of the clear band

Aiming at the middle of the lidar's clear band over-turned by **5.7°** and cost 24 cm of
closure. In a car that middle IS the doorway; in an open room it is merely the roomiest
direction, which is a different thing. The entry heading is now the objective and the
lidar band is a CONSTRAINT on it — inside the band, go back to it; outside, clamp to the
nearest edge. That stays correct in a car, where the entry heading is perpendicular to
the door and therefore inside the band anyway. Measured after: heading residual **−0.6 /
+1.1 / +1.1 / −0.6°** against +5.7° before.

### Still open
- **The 3° dead zone costs 4–8 s per turn when the remainder lands in 3–4°.** Seen twice
  in one run: `+3.0 → +0.0` then `+3.0 → +3.4`. The recommendation is a 3° tolerance —
  stop at the plant's own resolution instead of chasing it. 3° over 1.4 m is 7 cm against
  a 1.1 m door's 15 cm of margin, and in a car the seconds are the scarcer resource.
  Operator has not decided; nothing changed yet.
- **The entry turn is still HARD-CODED** (`--turn`), while the exit turn is computed. It
  cannot be measured before turning — the chest camera faces away from the panel until
  the robot is inside. It CAN be computed without a camera by registering the panel's
  MAP position once (the same argument `panels.yaml` makes for the layout) and taking the
  bearing from the robot's live pose. That would make the turn follow where the entry leg
  actually landed; today a drifted entry leaves −25° wrong and the fallback corrects only
  distance, never bearing.
- **The local `standard` move lands ~5 cm from the pose it is given** — three consecutive
  returns measured 5.1 cm, identical to the millimetre, so it is repeatable and biased
  rather than noisy. Useful for repositioning (12–15 s, no cloud), but do not read its
  target as where the robot will be.
- Heading drift on the straight legs (+1.1 to +1.7° over 1.4 m, ~26–32 mm lateral) is now
  the largest remaining error. Angular is pinned to zero, so heading is fully open loop.
- Identity verification at the reachable distance (anchors still 0/4; every press today
  used `--no-verify`).
- The press itself is 60–62 s of the ~103 s run. Suspects unchanged — the return to home
  between every button, the lift moving serially with the arm, the 0.35 s settle between
  moves — but instrument it before touching it.

## ▶ THE WHOLE MANOEUVRE RUNS AS ONE PROCESS, AND VISION CLOSES THE LAST FEW CM (2026-09-04, later)

`mock_elevator_run.py` drives the flow end to end from a registered start point: reverse
in, turn once, **measure the panel and close the remaining centimetres**, press, aim the
exit off the lidar, drive out. Three consecutive presses, **4/4 every time** (61.9 s,
59.8 s, 58.7 s), depth −3.06 to −3.18 mm, lateral 0.16–0.62 mm.

### The measured final approach is what makes the recipe repeatable

A hard-coded distance cannot work here, and the numbers say why: a single 2.7 m leg
scatters by about 5 % — 14 cm — while the window that decides whether the panel can be
pressed at all is a few centimetres wide. Measured on this panel: at x 0.649 and 0.667
all four buttons plan at 24/24 rolls with 63–69° of margin; at 0.695 they still press but
two are down to 1/24 and 2/24; at **0.731 two buttons have no IK solution at all** and the
other two are left with 5–6 mm of path clearance against a 5 mm limit.

So the recipe only has to get close, and a camera measurement closes the rest. Across
three runs the dead-reckoned legs put the panel at **x 0.661, 0.683, 0.696** against a
0.670 target — 0.9, 1.3 and 2.6 cm out, all inside tolerance, so the correction correctly
moved **nothing**. Same shape as the lift's `objective: still`: measure always, move only
when it buys something. The measurement costs ~200 ms because the engine and camera are
warmed up outside, before the robot enters, where the seconds are free.

### Three defects this shook out, one of which the safety layer caught

- **The exit aim had a sign error.** `heading_sweep` builds its travel frame by negating
  both axes for a backward leg, which is a 180° rotation and therefore preserves
  handedness — a clear heading `t` is reached by turning `+t` whichever way the leg
  points. Multiplying by the direction sign sent the robot the wrong way by twice the
  angle: the exit aimed −27.5° instead of +27.5°. **Nothing was hit, because the lidar
  clearance gate refused the resulting path** — the guard fired on a real fault for the
  first time.
- **The chassis closes an idle websocket.** The press blocks for ~60 s with nothing
  pumping the topic stream, so the next `wait_pose` died with "Connection to remote host
  was lost" *after the press had already succeeded*. The stream is now reconnected and
  `remote` re-asserted before the exit.
- **`OrbbecCamera` releases with `stop()`, not `close()`**, and the wrong call sat inside
  a bare `except: pass` — so the camera would have stayed open while the press tried to
  open it, which is exactly how the chest 335 gets into the state where it enumerates but
  delivers no colour. The failure would have surfaced one step later, wearing a different
  face.

### The base drives itself away between commands

The chassis's own `control_unit` issues `charge` moves whenever it feels like it, and
**holding `remote` does not stop it** — it takes the mode back to run one. Across separate
`drive_straight` invocations this silently corrupted measurements: a turn-and-turn-back
came back +40.7° for a commanded +25 with 0.53 m of translation, and an exit leg made
0.646 m of a commanded 2.50 with 36.67° of heading change. A straight leg sends angular 0,
so that heading change is the tell; the harness now aborts on it and cancels foreign moves
before each phase (`PATCH /chassis/moves/current {"state":"cancelled"}`, verified).

### `/scan_matched_points2` goes empty when the robot sits still

It publishes the points from the last SLAM matching update, so a long-stationary robot
produces no update and the topic returns `npoints = 0` at full rate — while `/slam/state`
still reports `lidar_reliable: true, lidar_matched: true`. Moving 0.3 m restored it
immediately (878–918 points, constraints 1 → 3). **This is the third explanation offered
today and the first that survives the evidence**: it was blamed on the emergency stop in
both directions before being measured properly, and both of those notes have been
retracted. The operational rule is unchanged and matters more than the cause: an empty
scan reads exactly like "nothing in the way", so everything built on it must fail closed.

### Press depth raised 3.0 → 4.5 mm

Operator reports the press is often too light — the plunger reaches the button and it does
not light. 3 mm was the first value that worked in a walk-up (1 and 2 failed), which is
not the same as a value with margin. 4.5 mm stays well inside what this tool has already
survived: when the plunger TCP was found 5.57 mm stale a commanded 3 mm was really
pressing about 8 mm, and the button lit with no damage. **Verify by the lamps, not the
log** — the press log compares the command against the same assumed TCP on both sides, so
it cannot see this class of error at all.

### Still open
- **Heading drift on long straight legs is the last systematic error**: +2.29° and +2.86°
  over ~2.8 m, i.e. ~11 cm of lateral, while other legs run 0.00°. Angular is pinned to
  zero, so heading is fully open loop and nothing corrects it. Closing that loop touches
  the file's core promise and should be a deliberate decision, not a quiet change.
- One exit leg issued its full 2.70 m of velocity and moved **1.564 m**. It did not
  reproduce (a repeat leg landed 1.265 m of 1.27), and a `2008 Wheel is major slipping`
  alert was standing — but the harness had `verbose=False` and so suppressed the one line
  that would have confirmed it. Now un-suppressed.
- Identity verification at the reachable distance (anchors still 0/4; every press this
  session used `--no-verify`).
- The lidar exit alignment cannot be validated in an open room — it needs a real doorway.

## ▶ ONE TURN IS ENOUGH TO REACH THE PANEL; COMING BACK BY ODOMETRY IS NOT (2026-09-04)

The flow the operator wants — park at a fixed point, reverse in, turn once, press, turn
back, drive straight out — was run end to end. **The first half works and the second half
does not**, and both halves produced a number worth keeping.

| | |
|---|---|
| recipe from the fixed point | reverse **2.558 m**, turn **−25.2°** |
| result | four buttons **24/24 rolls**, margins **60.7–69.5°** |
| press | **4/4 in 62.8 s**, depth −3.01…−3.05 mm, lateral 0.11–0.39 mm |
| round trip back to the fixed point | **0.373 m and 9.2° out** |

One turn really is enough, and this pose was better than the one that pressed on 09-02
(which had buttons at 1/24 and 2/24). Worth knowing why: the turn moves the arm base as
well as aiming it, because **the chassis rotation centre sits 0.281 m behind the arm base
origin** (solved from a measured turn: arm-frame (−0.281, +0.030)). So a turn changes the
panel's range as well as its bearing, which is what makes distance and bearing solvable
with one straight leg plus one turn.

### The return is the part that fails, and it is not fixable by tuning

Driving back out on dead reckoning missed by 37 cm and 9.2°. Against a 0.9 m car door and
a 0.8 m robot — 5 cm of margin per side — that does not fit. **So the exit cannot be
dead-reckoned; it has to be referenced to the door itself**, and the lidar is the only
sensor that can see it (the chest camera faces the doors only from inside, and the exit
is the one leg where being wrong is expensive). The measurement is cheap and can be taken
during the ride, when the robot is otherwise idle: a lidar frame is 2.00 Hz with 74 ms of
latency, i.e. ~1 s for two confirming reads, against a 62.8 s press. Only the straight leg
itself has to happen inside the door window.

### Legs now integrate what they command, and short legs stopped being wrong

Both loops used to run for `target / speed` seconds. Both also ramp DOWN over the last
stretch, and that was not in the sum — so every leg lost roughly the ramp's own deficit, a
**fixed** loss independent of leg length. It fitted `achieved = 0.969 × commanded − 2.8°`
across three clean turns, which is 28 % of a 10° turn and 3 % of a 90° one: exactly why the
"ratio" looked like it wandered between 0.69 and 0.96 with no pattern. Worse, a leg shorter
than the ramp spent its whole duration inside it.

The loop now accumulates the velocity it actually sends and stops when that integral
reaches the target, and the ramp is capped at 40 % of the leg. Straight legs, measured
after:

| commanded | achieved | error | before the fix |
|---|---|---|---|
| 1.00 m | 0.988 m | 12 mm | — |
| 0.40 m | 0.390 m | 10 mm | 0.80 → 0.676 (124 mm) |
| 0.15 m | 0.152 m | 2 mm | 0.075 → 0.083 |

### The chassis cannot turn finer than about 3°, and that is the plant

With the ramp deficit gone, single turns of 3–12° still came back +2.2, +3.0, −2.1, −2.9
and −4.0° off, and **commands below ~3° frequently moved the base not at all** — 0.0°
three times running, then a break-away to 2.9°. That is stiction, not measurement: `ori`
held its value for **20 s** after a turn, so nothing was still converging, and the feed's
own quantisation is 0.573°.

So `turn` closes the loop instead of trusting a model, damping each correction to 60 % of
what remains (a single step can be 70 % out, and damping converges monotonically anyway),
and stops at a 3° floor rather than chasing noise. Measured: +15° requested → 16.0°, +5° →
5.2°.

**The consequence is a design constraint, not a tuning target.** ±3° of heading over the
~1.2 m it takes to clear a doorway is ~6 cm of lateral drift, which is the whole margin of
a 0.9 m door. Either the target elevator's doors are wider than that, or exiting needs
something better than in-place turns. **Measure the real door width before designing the
exit** — it decides whether this approach works at all.

### Aligning to the door is built (`align`), and it is a heading sweep
`drive_straight.py align 2.0` reports which headings let the whole robot through, rather
than measuring a gap — the first version, facing a blank wall, reported "an opening
0.041 m wide", which is true and useless. Validated by tightening each dimension on its
own: a 2.0 m leg cleared over a 51 deg band, a 3.5 m leg over only 11 deg, and demanding a
1.9 m corridor made straight-ahead BLOCKED while still finding -16..-3 deg clear — the
doorway case, "straight will not fit, 3 deg will". Read-only, ~1 s for two confirming
reads.

Taking the operator's 1.1 m door: 30 cm of total margin, 15 cm a side, against the ~6 cm
of drift that 3 deg of unresolvable heading costs over the ~1.2 m of a doorway. It fits,
with room — which is what makes this approach viable at all.

### The chassis turns out to have a LOCAL navigation API

Looking for a way to send the robot back to its charger surfaced `/chassis/moves` — a
local move-action endpoint with types `charge`, `standard` and `along_given_route`. That
contradicts the assumption the straight-line driver was built under, that anything beyond
a straight line had to go through AutoXing's cloud. Verified by using it: a `charge` move
docked the robot and it drew -2.1 A within 24 s.

Two cautions, both learned elsewhere and both applying here: a local move runs in the
chassis's own planner, so killing the process that posted it does not stop the robot; and
a charge move's target is an approach pose, not where the robot ends up — docked, the
pose read 0.74 m and 12 deg away from the target it was given.

Worth revisiting properly, because `elevator_runner` drives through the cloud today, and
that is where the ~24 s of arrival lag and the read-only map both come from.

### A claim from this morning, retracted

I recorded that an engaged emergency stop empties the lidar point cloud, from seeing
`npoints = 0` while it was pressed and ~900 points after it was released. Measured again
the same afternoon with the e-stop engaged, the scan published 878-938 points per frame.
One co-occurrence is not a mechanism, and the note has been corrected rather than left
standing. What actually causes the empty scan is still unknown — a chassis that has just
booted and not yet localised is the leading candidate. The operational rule is unaffected:
an empty scan looks exactly like "nothing in the way", so it must fail closed.

### Still open
- Profile the press per phase before optimising it: 62.8 s for four buttons is ~15 s each,
  and the suspects are the return to home between every button, the lift moving serially
  with the arm, and the 0.35 s settle between moves — but this project has been burned by
  step-by-step timings before, so measure the real call.
- Identity verification at the reachable distance (anchors 0/4) — still the arm-camera
  argument.
- The fixed point outside the door should be an AutoXing waypoint, not a dead-reckoned
  one: `elevator test` docks to 1.5–2.3 cm, which is what makes a recorded recipe
  replayable at all.

## ▶ THE BASE TURNS IN PLACE, AND PRESSED 4/4 AT A NEW SITE IT DROVE ITSELF TO (2026-09-02)

The robot went from the charging dock to a pressable pose under its own power — straight
legs plus the new in-place turn — and pressed four buttons. **4/4 in 62.8 s**, depth
−3.00 to −3.10 mm against a commanded −3.00, lateral 0.16–0.34 mm, which is the same
accuracy as the best runs at the old cell.

| button | lift | joint margin | rolls passing | depth | lateral |
|---|---|---|---|---|---|
| `1` | 441 → 644 | 54.0° | 14/24 | −3.00 mm | 0.30 mm |
| `4` | 644 → 544 | 50.9° | **1/24** | −3.05 mm | 0.28 mm |
| `open` | 544 → 644 | 54.0° | 14/24 | −3.00 mm | 0.34 mm |
| `close` | 644 → 794 | 50.4° | **2/24** | −3.10 mm | 0.16 mm |

### The pose it worked from — recorded so it need not be re-derived

| | value |
|---|---|
| chassis pose (SLAM map) | **x = 16.492, y = −18.225, ori = −0.73 rad** |
| panel centre, ARM BASE frame | **x = +0.695, y = −0.171 m** |
| panel button heights, base frame | z = **+0.588 … +0.765 m** (≈ 1.10–1.28 m off the floor) |
| camera distance / bearing | 0.610 m / +15.9° |
| lift at localisation | command 441 |

Getting there from the dock: back **3.1 m** → turn **−27°** to aim → forward **1.06 m** →
turn −90°, translate **0.265 m**, turn +90° (the lateral fix) → forward **7.6 cm**. The
lateral leg is an artefact of starting at the charging dock, whose axis misses the panel
by 0.95 m; a designed approach would not need it.

### The reachable window is THIN, and that is the number that governs elevator entry

Two measurements, both from the same panel on the same afternoon:

| change | effect |
|---|---|
| **7.5 cm** closer along the approach | all four buttons **0/24 rolls → all four solvable** |
| **5.4 cm** of lateral offset (y −0.233 → −0.179) | button `4` **24/24 → 1/24 rolls**, margins 65–68° → 50–54° |

So "park within a few centimetres" is not a rule of thumb here — lateral position is the
sensitive axis, and it cannot be corrected after entering a car if the flow allows only
one turn. It has to be built into the entry line, outside the door, where time is free.

### In-place turns, and a 47 % overshoot that was really a stale pose

`drive_straight.py turn` — `move` sends angular 0, `turn` sends linear 0, two separate
functions so no caller can emit an arc. Swept radius **0.476 m**, taken from the chassis's
own 19-point footprint (the bounding box would have said 0.55 m). Measured
achieved/commanded: **0.688 at 10°, 0.874 at 23°, 0.880 at 28°, 0.957/0.963 at 90°**,
position drift 0–6 mm. A turn −90 / translate / turn +90 round trip came back within
**0.57°** — the pose feed's own quantisation.

The straight legs got a real fix on the way. Correction legs were overshooting badly
(0.40 m commanded → **0.589 m**) because the end-of-leg pose was read before SLAM had
settled: that leg reported 0.127 m, i.e. 68 % short, so the corrections chased a
difference that was not there. Reading the pose only once two consecutive fresh samples
agree took 0.80 m commanded to **0.753 m achieved, 1 mm lateral**. An independent camera
fiducial confirms the settled pose is sound: a 0.50 m leg measured 0.401 m by camera and
0.384 m by pose.

### The press had to bypass the identity check, and that is the arm-camera argument

At the pose the arm can reach, **all four anchors read `empty`** — 0/4 agreement — so the
press was refused 15 of 15 times and only ran with `--no-verify`. Not exposure (faceplate
mean 116–173, **0.0 % saturated** at every value swept from 60 to 200) and not detection
(9–10 of 10 buttons, lattice residual 2.0–2.4 px against a 6 px threshold). It is pixel
size: buttons are **28.4 px** here, so `classify_solo` sees a 60 px crop, and the measured
curve for that crop size is 2–3 correct out of 10.

This reproduces, at a second site and from the opposite direction, the 2026-09-01 finding
that the positions where the camera can SEE and where the arm can REACH do not overlap.
There it was "detection healthy, 336 combinations with no IK"; here it is "reach solved,
identity unverifiable". Same gap, and it is what an arm-mounted camera is for.

### Still open
- **Identity verification at the reachable distance.** Unsolved. Next cheap test: classify
  all ten cells individually and see whether any survives at 28 px — the green star on `1`
  is the highest-contrast marking on the faceplate.
- **Door open/closed and floor arrival** — neither built. Note the chest camera
  structurally cannot see the door before entry (the robot backs in, so it faces away),
  which leaves the lidar as the only sensor for the door, with no cross-check.
- **In-car time budget.** One turn plus one button is roughly 18–25 s on today's numbers
  (4 buttons + 5 lift moves took 62.8 s; a 27° turn ~3 s, a 90° turn ~10 s). The 0.30 rad/s
  yaw cap is a conservative choice of ours, not a chassis limit, and is the obvious lever.
- `boundary_sweep.py` is dead against the current config (see the gotcha).
- Second independent cross-check of the re-measured plunger TCP (still one touch).

## ▶ THE BASE CAN NOW BE DRIVEN IN A STRAIGHT LINE, LOCALLY (2026-09-01)

`+-2 m` straight, commanded directly to the chassis, with the angular velocity pinned
to zero. Measured, both directions:

| | back 2 m | forward 2 m |
|---|---|---|
| achieved | −1.971 m | **+1.985 m** |
| distance error | +2.9 cm | **−1.5 cm** |
| lateral deviation | 9 mm | 43 mm |
| heading change | −0.57° | −1.15° |
| correction legs needed | 2 | **0** |

### Why this was needed at all

Everything up to now drove through AutoXing's cloud, whose only abstraction is "go to
this point" — the planner decides how, and **it prefers to turn**. Entering and leaving
an elevator has to be straight in and straight out to make the door window, and "it did
not turn this time" is not a guarantee you can build on. Operator's call, and it is the
right one: the requirement is a hard constraint, not a preference.

Two facts made this urgent rather than nice-to-have. The panel at the new position sits
at `y = −0.55 m, z = +0.69 m` in the arm base frame, where **336 combinations (14 lift
heights × 24 approach rolls) have no IK solution** — the arm simply cannot reach from
where the base can conveniently stop. And the robot cannot park as close as it used to,
because it must not block the doorway. So the gap between "where it can see" and "where
it can reach" has to be closed by a short, exact, straight move.

### Finding the chassis took most of the session, and the map was wrong three times

The chassis is **not** on any network we had documented. What it took to find it:

- `chassis_ctl.py` (a colleague's script for another DEX) documented the protocol but
  not the address: its `KNOWN_HOSTS` are `192.168.25.25` / `192.168.12.1`, and its
  `IFACE` is the **WiFi** adapter. Neither host was reachable.
- Scanned the base's presumed address `192.168.11.35` end to end (**all 65535 ports**):
  only ADB and iFlytek AIUI audio/video. **That device is not the chassis at all** — ADB
  says `LubanCat-4 / EmbedFire`, an Android 12 board doing voice and face HRI. Chasing
  8090 on it was wasted from the start.
- Swept the whole WiFi subnet: three AutoXing chassis answered `/device/info`, **all of
  them other people's robots** (`longjack`, two `hawk_longtray`). Identified by serial,
  never touched.
- The chassis turned out to be at **`192.168.25.25`**, on the robot's own **wired**
  network — same cable as the arms, a *different subnet*. It only became reachable when
  the Jetson's `eno1` gained a second address, `192.168.25.46`.

Worth stating plainly: this robot's parts sit on **three mutually unreachable
networks** (Jetson wired, Jetson WiFi, chassis subnet), which is why every local
interaction so far had to go through the cloud. Straight-line driving is just the first
feature to hit that.

### Protocol, and the four things that had to be measured to make it work

```
POST /services/wheel_control/set_control_mode {"control_mode":"remote"}
ws://192.168.25.25:8090/ws/v2/topics
    -> {"enable_topic":"/tracked_pose"}
    -> {"topic":"/twist","linear_velocity":v,"angular_velocity":0}
    <- /tracked_pose {pos,ori}, /scan_matched_points2, /slam/state, /wheel_state
```

| measured | value | consequence |
|---|---|---|
| twist keepalive | must be **≥20 Hz** | at ~4 Hz the watchdog stopped the wheels after 2.3 cm |
| **`/tracked_pose` rate** | **1.07 Hz**, and it REPEATS the same value while moving | see below — this caused every false failure |
| achieved / commanded | **0.90** | leg duration is divided by it |
| coast after braking | ~11 cm at 0.2 m/s | ramp down over the last 18 cm |

### The pose feed is 1 Hz, and reading it as if it were live produced five wrong conclusions

This is the lesson of the session. At 0.2 m/s a 1 Hz feed is a **20 cm** position
resolution, and worse, consecutive messages repeat the same value while the base is
moving (SLAM settles late). Every one of these was a measurement artefact, not a fault:

| what I concluded | what was actually happening |
|---|---|
| "0.20 m move overshot to 0.31 m" | pose lagged; the loop saw the target late |
| "STALLED — something is in the way" (×3) | robot was driving at 0.18 m/s the whole time |
| "wheel slip" cut a move short | `wheel_slipping` goes true during ordinary driving |
| "it hit a wall" | the chest camera saw a wall on the ARM side, which says nothing about the other direction of travel |
| "the base is being cut off by its alert state" | it was not; alerts were a red herring |

The clean measurement that settled it: **100 twists over 5 s, 120 `/twist_feedback`
replies, 0.903 m travelled, 0.181 m/s average.** The chassis had been executing
perfectly the entire time.

So distance is now driven **open-loop on time** at the calibrated ratio, and the pose is
used only for what it is good at: a settled before/after measurement (0.0 mm of scatter
while stationary, lidar-matched, no drift) plus a correction leg when short.

### Stall detection moved to CURRENT, obstacle sensing to the LIDAR

Two replacements, both because the original signal was unusable while moving:

- **Stall → battery current.** Driving into the charging dock drew **30.4 A** against a
  ~4.4 A idle and ~5 A while driving normally, with zero displacement. A physical
  quantity, no SLAM lag. Threshold 20 A held for 0.8 s.
- **Obstacles → `/scan_matched_points2`.** 360°, ~870 points/frame, in map coordinates,
  **1.87 Hz with 74 ms median transport latency**. Transformed into the body frame it
  gives forward and back clearance in one shot — which the chest camera structurally
  cannot do, since it faces only the arm side. That limitation was the operator's
  observation, and it was correct.

Verified against a person: someone standing 0.9 m in front produced **two clusters,
14–15 cm wide, 32 cm apart** — two shins. The clearance gate would have refused to move.

### New tools

- `initialization/drive_straight.py` — `state` / `clearance` / `move` / `calibrate` /
  `stop`. Serial-number check (this network carries other robots), lidar clearance gate
  before any twist, current-based stall abort, and every exit path brakes and restores
  `auto` (leaving the base in `remote` silently disables AutoXing navigation).
- `initialization/lidar_preview.py` — live top-down lidar in a browser (port 8011):
  forward/back clearance, corridor width, and **which side the robot is off-centre**,
  which a single "nearest obstacle" number hides.
- `initialization/aim_preview.py` — live chest camera with button detection, bearing,
  distance and base-frame position (port 8010), for aiming the robot at a panel by hand.

### The sensor split, decided from measurements

Not redundancy — the three sensors see **different heights** and cannot substitute for
each other:

| sensor | sees | owns |
|---|---|---|
| chassis lidar | z ≈ 0, a **7 cm** thick plane (measured: 14,800 points span 0.07 m) | base obstacles — people, chairs, walls, door frames |
| **arm-mounted camera** (planned) | the panel, from wherever the arm puts it | button detection and press verification |
| chest camera | the arm's workspace, 0.7–1.2 m | assisting: workspace clearance before the arm moves |

The lidar cannot see a hand reaching toward the panel — that happens half a metre above
its scan plane. The chest camera cannot see behind the robot. Neither gap is fixable by
configuration.

### Still open
- **Leg ratio is not stable with distance**: 0.91 at 1 m, 0.55 on the first leg of a 2 m
  back move, 0.99 on a 2 m forward move. The correction legs absorb it, but the model
  should be fixed loss + proportional, not a single ratio. Short correction legs are
  worst hit (0.09 m commanded → 0.039 m).
- Lateral deviation grows with distance (9 mm at 2 m one way, 43 mm the other).
- The chassis carries a standing `6010 System down unexpectedly!` alert and a
  `9502 Debugging config file exists(.param.yaml)` warning. A reboot cleared `6007`
  (head_unit link) and a stuck failed charge action, but `6007` came back.
- **Floor detection is unsolved and next.** Three candidate signals, to be combined
  rather than chosen blindly: the button lamps (lit/unlit), the elevator's own floor
  display, and the robot's **IMU** (vertical acceleration integrates to which way and
  how far the car moved). Each fails differently — lamps are occluded by the plunger and
  wash out at press exposure, a display may not exist or be unreadable, IMU drifts —
  so the design should pick per site, and say which it is relying on.
- Second independent cross-check of the re-measured plunger TCP (still one touch).

## ▶ THE WHOLE LOOP RUNS FROM `BBB`: DRIVE, DOCK, PRESS 4/4 (2026-08-31)

`BBB` -> `elevator test` -> press `1 4 2 5`, three consecutive live runs, **3 of 3 with
4/4 buttons pressed**. This is the first time the drive and the press have been exercised
together from a second start point.

| run | arrival error at the elevator point | buttons | press time | lateral |
|---|---|---|---|---|
| 1 | **0.9 cm** | **4/4** | 66.9 s | 0.34 mm |
| 2 | **2.3 cm** | **4/4** | 68.1 s | 0.31 mm |
| 3 | **2.0 cm** | **4/4** | 67.2 s | 0.18 mm |

Per button, run 1: joint margin 46.6-53.5 deg, path clearance 52 mm, **24/24 approach
rolls passing** on every button, depth -2.97 to -3.03 mm against a commanded -3.00. The
torso tracked the panel 444 -> 744 -> 944 -> 844 -> 944 and restored to 444. Pre-warm
came up in 9.4-9.6 s during the drive, so the arm starts moving ~1.6 s after arrival.

`liverun.py` now takes the start waypoint and floors as arguments
(`python3 liverun.py BBB "1 4"`, `-` skips the start point) instead of hard-coding `AAA`.

### The start waypoint's own docking accuracy does not matter

Measured over five navigated round trips, `BBB` docks badly and inconsistently — **3.0 /
7.6 / 8.6 / 26 / 68 cm** from its point, i.e. 2 of 5 inside 8 cm — while `elevator test`
closed to **1.5-2.3 cm on every one of nine arrivals** today. At `BBB` the base simply
stops wherever it is, declares `moveState: succeeded`, and never converges: no
`hasObstruction`, no `hasPersonAhead`, no `failed`. Moving the waypoint onto a
reached pose did not change this, so it is the spot, not the coordinates.

**And it is irrelevant to the task.** Run 1 stopped ~17 cm from `BBB`, drove on, and
still reached the elevator point at 0.9 cm and pressed 4/4. Only the elevator point's
accuracy feeds the press, and that one is repeatable. `BBB` is a place to drive FROM.
This was chased as a blocker first and it was not one — the open item from 2026-08-28
asked whether `BBB` docks cleanly, which measured the wrong thing.

Worth knowing anyway, because it means our arrival gate gets exercised: the fast base
gate correctly REFUSED the 26 cm and 68 cm cases and kept waiting; what returned
"finished" there was the cloud task-status backstop, which does not look at distance.
`run_loop` re-reads the position afterwards and gates on the measured error, so those go
to `corrective_drives`, never to the arm.

### The only real failure was the camera, and it took the whole run down

The first attempt failed with `[cam_chest] no color frame after 40 tries` — the chest 335
enumerating normally while delivering no colour, the known device-state fault. It hit
**twice in one run**: the pre-warm died on it (the runner fell back to a serial press),
and the press itself then died the same way, so the loop ended `successes=0 failures=1`
having driven the whole route.

Diagnosed the recorded way — the head 335L captured fine throughout, which is what
identifies device state rather than a code fault — and cleared with `Device.reboot()`, no
root and no replug. Two API details worth keeping: the reboot needs `Context` and
`DeviceList` held in **live references** (temporaries fail with `NULL pointer passed for
argument "deviceMgr"`), and the device re-appears in `query_devices()` within ~2 s while
still not delivering frames, so poll an actual `capture()` rather than enumeration.

### Still open
- **Nothing recovers from the camera fault automatically.** It is a known,
  self-clearing-with-a-reboot condition that costs a full drive when it hits, and both
  the pre-warm and the press fail on it independently. A `Device.reboot()` + retry inside
  the capture path would turn a failed loop into a ~30 s delay.
- **Anchor agreement is thin.** Runs today read **1/4, 0/4 and 2/4** anchors; the 0/4
  frame was correctly REFUSED ("a shifted alignment explains the anchors at least as
  well") and the retry passed at 2/4. The relative test is doing its job and the 15
  retries absorb it, but there is little margin, and it is the same classifier
  degradation recorded at greater docking distances. Detection also found 8-9 of 10
  buttons per frame, with the lattice filling the rest at 1.6-1.7 px residual.
- **`BBB` needs a different pose, not a nudge**, if it is ever wanted as a dock rather
  than a start point. Yaw there IS repeatable (4.43-4.45 across all runs), so orientation
  is not the variable. Re-recording it needs the AutoXing app — our credentials are
  read-only for the map.
- Second independent cross-check of the re-measured plunger TCP (still one touch).
- Why the base reports an obstruction at the docking point at all. Handled is not
  understood; the cabinet the faceplate is mounted on is the obvious candidate.

## ▶ THE ARM WON'T MOVE INTO ANYTHING, AND A DETECTION DEAD END IS FIXED (2026-08-28)

Three things: an obstacle guard for the arm, a detection failure that could not recover
from itself, and the terminal finally keeping up with the robot.

| | |
|---|---|
| arm obstacle guard | depth from the chest camera, before every button — verified by hand |
| detection dead end | 15 of 15 refusals with the panel in view -> residual 11.3 px to **1.7 px** |
| terminal lag behind the robot | minutes -> **2 s** |
| `1 4 2 5` after driving | 4/4 lit, lateral 0.09 mm |

### The arm now refuses to move into anything

**The base cannot see the arm's workspace, and never could.** The arm and chest camera
face **180 degrees away** from the base's front, so its obstacle sensors watch the
opposite hemisphere. Worse, parked, the cloud API reports nothing about the
surroundings at all: blocking the robot for 70 s moved **none of `robot_state`'s 35
scalar fields**. I built a guard on `hasObstruction` first, said it would stop the arm,
and believed it for an hour before measuring. It could never have fired.

What works is the chest camera's depth, which points exactly where the arm goes.
Deproject it into the base frame and ask what sits IN FRONT of the fitted panel plane:

| | mm in front of the plane |
|---|---|
| the wall behind | −127 |
| empty scene, 99.9th pct of 56,813 px | **+4** |
| empty scene, max | +21 (the buttons' own protrusion) |
| a hand in the gap | **+197**, over 20.8 % of the view |

0.000 % of an empty scene passes 30 mm, so the 40 mm threshold is the middle of a wide
band, not a tuned edge. Checked from a fresh frame before EVERY button, with the arm at
home — outside the camera's view by design, so it detects the world and not the robot.
Verified on hardware: `1` pressed normally, a hand went in, `4` was refused with IK,
52 mm clearance and self-collision all passing. Limits stated rather than hidden: it
sees only the camera's field of view, and it checks *before* the motion, not during.

### The ROI could not look where the detector had missed

The full-frame pass scales 1280x720 into the model's 640, so a 44 px button becomes
~22 px and a whole edge row can drop out — `open`/`close` scored **0.16 and 0.06** full
frame against **0.91 and 0.96** on a crop of that row alone. The ROI is derived FROM
those detections, so the missed row put the ROI's edge above it, the refined pass never
looked there, and the lattice got four rows for a five-row layout: **refused 15 of 15
attempts with the panel plainly in view**. A failure that seals itself.

Fix: when fewer rows or columns are found than the layout registers, grow the ROI by
the MISSING count times the measured pitch. Bounded by what is missing, so a complete
grid grows by nothing and working frames are untouched. Bottom edge 410 -> 463 px,
still 81 px clear of the cabinet keyhole the clustering exists to exclude.

**The first hypothesis was wrong and measuring killed it.** The scene was blown out and
the light was the obvious suspect, but the missed row measured mean/std 192.8/47.9
against 190.4/49.1 for a row that WAS found. Brightness was not the variable.

### The terminal follows the robot now

Measured, SSH was never the cause: 108 ms RTT, 0.16 s per call with `ControlMaster`
(0.6 s without). The cause was waiting for a job to EXIT before reading its log — which
turned a 40-70 s press into a blank terminal, and the waiting command then timed out
into the background on top. One measured gap was **2 min 40 s** while the robot drove
and pressed. `press_stream.sh` and `liverun_stream.sh` `tail -f` instead: **2 s**, which
is the floor, since the data comes from the cloud and the base reports event-driven.

### Also
- **`poll_timeout_sec` is now a STALL timeout, and progress means MOTION.** Measured
  from dispatch it conflated "slow" with "stuck": a blocked drive took 316 s against a
  normal 56 s, and the 300 s budget gave up 6 s before it finished, 3 cm from target. A
  first fix counting `hasObstruction` as progress was worse — the base held that flag
  for **six minutes** parked 5 cm from its goal without moving. A stall inside tolerance
  now cancels the task first, re-confirms stillness, and counts as arrival.
- **Obstacle handling verified.** Blocked deliberately with people and chairs, the base
  stops, re-routes and still arrives: a block 17 cm from the goal held it 63 s, after
  which it completed and docked at **0.4 cm**. Blocking reports as `hasObstruction`, not
  `hasPersonAhead` — I got the causation wrong in both directions before measuring it.
- **The base's situation is logged whenever it changes** during a drive. It immediately
  caught the base declaring a task "succeeded" while still **95 cm** short; a corrective
  drive closed that to 7.9 cm in 25 s.
- **Per-button lift heights, recorded at last**: 443 -> 744 -> 894 -> 794 -> 894, margins
  49.6-55.6 deg, 24/24 rolls, 52 mm clearance throughout. The torso moves both ways.
- **Our AutoXing credentials are read-only for the map.** Both POI list endpoints answer
  200; fourteen plausible write paths all 404. Probed by sending a record's existing
  values, so nothing changed while finding out.
- **Docking quality is a property of the SPOT.** Same ~2.5 m: `BBB` took 121 s and four
  attempts (`moveState: failed`, a value not seen before); moved 50 cm, 78 s, stopping
  21 cm short; `elevator test` took **20 s** straight in to 2 cm.

### Still open
- ~~Navigate to `BBB`'s new position and watch whether it docks cleanly.~~ Answered
  2026-08-31, and it was the wrong question: `BBB` docks badly (2 of 5 inside 8 cm) but
  that does not matter, because it is only a place to drive FROM. See the top section —
  the whole loop runs from it, 3/3.
- Second independent cross-check of the re-measured plunger TCP (still one touch).
- Why the base reports an obstruction at the docking point at all. Handled is not
  understood; the cabinet the faceplate is mounted on is the obvious candidate.

## ▶ THE WAIT AT THE ELEVATOR IS GONE, AND THE TORSO NOW MOVES (2026-08-27)

Two complaints, both fixed and both measured. The robot used to park at the elevator
and then stand there for the better part of a minute before the arm moved; and while
pressing, only the arm moved, which reads as a fault to anyone watching.

| | before | after |
|---|---|---|
| arrival declared | cloud task-status poll | the base's own `moveState` |
| ...measured on one drive | 75 s | **51 s** |
| startup after arrival | 8.3 s | **1.6 s** (pre-warmed during the drive) |
| torso during a press | still | **moves to each button's best-margin height** |
| `1 4 2 5` | 51.2 s, 4/4 lit | **71.2 s, 4/4 lit** |

Net: ~31 s less standing about, and the extra 20 s is deliberate motion.

### Arrival now comes from the base, not the cloud

`robot_state` sits a level below the navigation task and carries the base's own
`moveState`, `speed`, `hasPersonAhead`, `locQuality` and `battery`. The base reports it
**event-driven**: parked, its timestamp went 60 s without moving; the report that
followed a change was **1.9 s** old. Measured side by side on one drive — the base said
"arrived" at **51 s**, the task poll at **75 s**, and the two positions differed by
**0.6 mm**, so the base really had stopped 24 s before the cloud admitted it.

Four conditions, all required, plus two guards, each covering a different way of being
wrong:

| condition | what it stops |
|---|---|
| timestamp after dispatch | last run's stale report |
| `moveState` done **and** `speed` 0 | guessing from outside; this is the base's own word |
| within tolerance of the elevator point | a multi-point task also completes moves at intermediate points |
| **departure observed first** | every loop after the first STARTS parked at the elevator with `moveState` already "succeeded" — all four pass at once, and the press would fire as the robot is about to drive away |
| **no position drift between confirmations** | the base still creeping to its goal — the one way this signal can be early, and exactly the failure that put the arm into the panel when the camera was the trigger |

The cloud task status stays as the BACKSTOP (it catches a cancelled task, or one that
ends where the fast gate never accepts). `fast_arrival: "observe"` computes the fast
signal and logs when it WOULD have fired while still waiting for the cloud — one loop
in that mode measured the 24 s at no risk, which is how this was validated before
being switched on.

**Correcting an earlier claim of mine.** I wrote that the cloud was "20-55 s late",
from comparing a task's `endTime` to when the runner acted. Wrong decomposition:
`endTime` is only **6-8 s** behind the base's own arrival (57 vs 51; 99 vs 91). The lag
is between `endTime` and when a poll can SEE `taskStatus == 4` — 18 s on the measured
run. The honest number is the end-to-end one: **~24 s.**

Ruled out along the way: the base's port **9090 is the face camera** — a binary push
stream carrying MJPEG and `hasFace`/`hasHand` JSON, no pose, and not rosbridge despite
the port. There is no local navigation interface we can reach.

### The press is pre-warmed during the drive

~7.3 s of the press's startup does not depend on where the robot is:

| | |
|---|---|
| load the 44 MB TensorRT engine | 2.77 s |
| open the camera | 2.54 s |
| ramp the GPU off its 306 MHz idle clock | ~1.5 s |
| connect the arm | 0.4 s |

`press_buttons.py --wait-go PATH` does all of it, prints `PREWARM_READY`, and BLOCKS on
a token file the runner writes only after arrival is confirmed. **The arrival decision
does not move into the press** — inferring arrival from the camera is what drove the
arm into the panel once. A/B, twice each: **8.3 / 8.4 s cold vs 1.59 / 1.70 s warm.**

### The hand now closes itself, and refuses to move if it cannot

`press_buttons.py` never touched the hand — the fist was done out-of-band by whoever
was driving, so a run that skipped it drove the arm at the panel with the fingers
extended (the fingertips are 172.87 mm from the flange against the plunger's ~154 mm).
It is now the first thing after connecting, before any motion, and it is VERIFIED by
reading the joints back. The test is "nothing is sticking out" (all six <= 100), not
per-joint matching: in a fist the index bottoms out against the thumb at ~70, measured
67 on this run. No answer or no closure -> the press refuses to move. `--no-fist` is
for a robot with no hand fitted.

An out-of-band prerequisite is one someone eventually forgets — and the hand loses
power across an emergency stop, so "it was a fist last time" is not a state that
survives.

### The torso tracks the panel (`arm.lift.objective: margin`)

The lift used to keep still whenever the current height already cleared
`prefer_margin_deg`, so at a comfortable dock it never moved. It now goes to the height
with the BEST joint margin for each button. Asked for as presentation — a robot that
stands perfectly still while only its arm works looks broken — but the heights it picks
carry the largest margins available rather than the first sufficient one. Both
objectives choose only from poses that already passed the joint-limit, wrist,
self-collision and panel-clearance checks, so this trades optimality and time, never
safety. Two buttons on the same row share a best height, so `min_move_command` (100
units = 50 mm of body travel) sends the second to the next-best SAFE height purely so
there is motion to see.

Result: 4/4 lit, depth -3.10 mm, lateral 0.36 mm, 71.2 s. Worth watching: lateral was
0.08 mm on the run where the torso did not move, so every lift move costs a little
accuracy through the coordinate re-compensation.

### Also
- The **full press transcript** now goes to `/tmp/dex_press_last.log`. Only a
  three-line tail reaches the UI, and when a press failed mid-sequence that tail was
  the camera's startup banner — the failure was undiagnosable afterwards.
- **Corrective drives** merged from the robot: up to N single-point tasks to close the
  remaining distance when the arrival gate fails, instead of stopping dead. An
  avoidance manoeuvre left the robot 73 cm short with the task still reporting
  "completed"; re-dispatching closed it first time.
- `wait_for_arrival` is ONE implementation, shared by every wait. An arrival gate with
  two copies is an arrival gate with two behaviours.

### Not done
- The per-button lift height table — the run that would have produced it predates the
  transcript logging, so the heights `1 4 2 5` actually selected are not recorded yet.
- Second independent cross-check of the re-measured plunger TCP (still one touch).

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

### A collision, and what it cost to understand (2026-08-27)

**The arm collided with the panel.** Cause: the arrival trigger had been changed from
"AutoXing reports the task finished" to "the camera sees the panel visible and still",
to save the 20-55 s of cloud latency. Those are not the same condition. Localisation
takes ~200 ms per frame, so during the base's final deceleration consecutive frames
fall inside the 2 mm stillness threshold while the robot is still travelling. The press
locked coordinates and moved; the panel then kept moving. `--auto` had only ever been
tested with the robot stationary, where this failure cannot appear — the module was
fine, the mistake was wiring it into a scenario that was never re-verified.

Three things I got wrong during the recovery, all the same shape — treating "nothing is
happening on my side" as "the robot will not move":

| assumed | actually |
|---|---|
| `kill -9` on the runner stops the robot | the task runs in AutoXing's cloud; the base drove the rest of its route by itself |
| `move_joints_sync` returning False means the arm stopped | `rm_movej` keeps executing in the background; the joints were creeping toward the target between two of my status reads, which is what was colliding while I analysed paths |
| self-collision clear + tip clear of the panel = safe | the controller's model covers the arm's own links only, not the torso, lift column or camera mount |

Recovery: stop + delete trajectory + clear joint errors, then a two-stage return
(retract 80-100 mm along -x, then home in verified legs). The direct path home would
have taken the tip to 73 mm from the panel; retracting first kept it 190 mm clear.
No hardware damage — every button still lights.

### Plunger TCP re-measured, and it was 5.57 mm out (2026-08-27)

After the plunger was adjusted by hand, the configured offset was stale:

| | value |
|---|---|
| configured | `[26.00, -1.91, 24.74]` mm |
| measured | `[24.15, -2.99, 29.88]` mm |
| error | `[-1.85, -1.08, +5.14]` mm, norm **5.57 mm** |

The +5.14 mm is along the approach axis, so a commanded 3 mm push was really pressing
about 8 mm, with the contact point ~2.1 mm off centre. **The button still lit** — which
is the whole reason this had to be measured rather than inferred.

**The press log cannot detect this.** Its depth/lateral figures compare the commanded
tip against `get_tcp_pose() @ tcp_offset`; the controller servos to the assumed TCP, so
an error in it cancels out of both sides. Reported lateral stayed under 1 mm while the
offset was 5.57 mm wrong. Only the button lighting and an independent measurement see it.

Method: the operator held the tip on the centre of button `1`, that pose was read from
the controller, the arm retracted, and the camera measured button 1's surface over 8
frames (spread 0.27 / 0.55 / 0.65 mm). `offset = inv(base_T_tool) @ tip`. The camera
side is solid; the uncertain part is the hand placement, and unlike the original
calibration this has no second, independent cross-check yet.

Validated by pressing `open 1 4 5 close`: **5/5 lit at a true 3 mm push**, and every
button's lateral residual improved:

| button | stale TCP (2 runs) | re-measured |
|---|---|---|
| `open` | 0.72 / 0.80 | **0.16** |
| `1` | 0.87 / 0.39 | **0.28** |
| `4` | 0.24 / 0.23 | **0.06** |
| `5` | 0.63 / 0.65 | **0.39** |
| `close` | 0.17 / 0.17 | **0.08** |

Careful with that table though: it is corroboration, not proof. Changing the offset
changes the commanded flange pose and therefore the IK solution and the roll chosen
(`open` went 330 deg -> 0, `5` 270 -> 255), and the residual it measures is
configuration-dependent servo error. The proof is the lights at 3 mm.

### Detection got much more reliable (2026-08-27)

Localisation went from **1 of 4 frames to 7 of 8**. Two causes, both found by looking at
why frames were refused rather than by retrying harder:
- The ROI clustering link distance was 2.2 button widths too generous at 3.0. The
  cabinet's keyhole sits ~2.25 widths from the nearest button, so it merged on some
  frames; when it did, the ROI stretched 100 px, the refined pass returned 11
  "buttons", and the lattice fit failed outright. Neighbouring buttons are 1.6 widths
  apart, so 2.2 keeps the grid and excludes the keyhole.
- When the fit still fails, `assign` now drops the detection furthest from the group and
  retries, instead of throwing the frame away.

### Both arms have a symmetric resting pose (2026-08-27)

`arm.home_joints_deg_left` = `[-39.76, -100.13, -79.28, 131.63, 117.0, -68.12]`. The
mirror is "negate J1, J4, J6" — established by trying all five plausible sign patterns
and checking with FK which one lands the left TCP on the mirror of the right TCP. Only
that one does, to **0.0 mm**; the others miss by 324-488 mm, and four of the five look
plausible. Both hands rest in a fist, and the fist goes on BEFORE the arm moves: the
LinkerHand's fingertips are 172.87 mm from the flange against the plunger's ~154 mm, so
extended fingers are the front-most part and hit the panel first.

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
**`data/` is excluded WHOLESALE, and both halves of that matter.** `data/calibration/` holds the
Thor unit's artifacts, which must never be reused here. `data/weights/` is the opposite problem:
the model that matters is TRAINED ON THE ROBOT and does not exist on the Mac, so syncing that
directory sends the Mac's stale copy the wrong way. Measured 2026-09-02 — a run of the old
command overwrote the robot's `buttons.pt` (the 5-source merge, 2026-08-25) with the Mac's
single-source July build, silently, because rsync only reports that a file changed and not which
direction was right. Inference survived it (`trt_detector.py` loads the `.engine`, not the `.pt`),
and the source was recoverable from `runs/detect/buttons_merged5/weights/best.pt`, but nothing
warned at the time.
```
rsync -az --exclude='.git/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.DS_Store' \
  --exclude='data/' --exclude='乘梯相关接口.pdf' \
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
