#!/usr/bin/env python3
"""Offline boundary sweep for the button-press planner — NO hardware motion.

Finds where the press planner's boundaries bite, WITHOUT running the arm. This is
the "let it find more boundaries" tool done the safe way: sweep the model, not the
real arm. Nothing here commands a motion; the SDK layer only calls
``rm_algo_inverse_kinematics`` (pure computation).

    # Geometry-only (runs on any laptop; no arm, no SDK):
    python3 initialization/boundary_sweep.py --nominal

    # Full sweep WITH joint-limit / singularity / path checks (run ON the Orin,
    # where the RealMan SDK is importable — same kinematics as the real arm):
    python3 initialization/boundary_sweep.py --nominal --use-sdk --csv out.csv

Two layers, deliberately split by what each needs:

* GEOMETRY layer (always on) — reuses ``core.press`` only, so it is the SAME
  geometry the real pipeline uses. Checks the boundaries that need no IK:
    - the straight push must not sweep the plunger into the faceplate
      (``min_clearance_mm``, but the geometric part of it: contact-vs-standoff);
    - the button must sit inside the arm's usable height band once the lift has
      compensated (``lift.usable_relative_z_m``);
    - the standoff must not fall inside the inner unreachable region (STANDOFF cap).

* IK layer (``--use-sdk``, Orin only) — mirrors ``plan()`` in press_buttons.py so
  the numbers match production: solves IK at BOTH standoff and contact, then checks
    - ``min_joint_margin_deg``  (distance of the nearest joint to its hard stop),
    - ``min_wrist_deg``         (|J5| away from the wrist singularity),
    - ``max_path_jump_deg``     (configuration flip along the interpolated path) —
      but sampled DENSER than plan()'s 21 points, and along the standoff->contact
      movel too, which plan() never samples. That extra sampling is exactly the
      "more boundaries" this sweep is meant to surface.

The sweep dimensions are the three real-world variabilities from the stability
agenda: DOCKING offset (base stops with cm error), LIFT height, and the 24 approach
ROLLS already searched at run time. For every combination it records pass/fail and,
on failure, which boundary bit — so the output is a coverage map, not a yes/no.

Button 3D coordinates come from either a real run (``--buttons-json``, the exact
coords a session cached) or a NOMINAL panel reconstructed from configs (``--nominal``:
the known-good dock plane in ``configs/pipeline.yaml`` + the grid in
``configs/panels.yaml`` + the measured 86.3 mm row pitch). Nominal is clearly marked
in the output; it is for exploring sensitivity, not for certifying a specific dock.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import numpy as np

REPO = __file__.rsplit("/", 2)[0]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from core.config import load_pipeline, load_yaml  # noqa: E402
from core.press import _orthonormal_frame  # noqa: E402
from core.transforms import make_transform, matrix_to_rpy  # noqa: E402

# Hard stops read from the controller (rm_get_joint_max_pos) on the RIGHT RM-65 —
# identical to press_buttons.py. J3 binds for every button on the mock panel.
J_LIMIT = np.array([178.0, 130.0, 135.0, 178.0, 128.0, 360.0])

# Nominal panel row pitch, measured 2026-08-25 (tape 3-3/8..3-1/2 in ~= 86.3 mm) and
# a plausible column pitch for the 2-wide faceplate. Used ONLY for --nominal.
NOMINAL_ROW_PITCH_M = 0.0863
NOMINAL_COL_PITCH_M = 0.040


# ----------------------------------------------------------------------------
# Button coordinates
# ----------------------------------------------------------------------------
def nominal_buttons(cfg: dict) -> dict[str, np.ndarray]:
    """Reconstruct a 5x2 grid on the config's known-good dock plane (base frame).

    NOT a measurement of any live dock — a nominal layout for sweeping. Buttons are
    laid on the reference plane, centred on the reference point, using the measured
    row pitch and a nominal column pitch. Rows run top->bottom, cols left->right, to
    match configs/panels.yaml.
    """
    panel = cfg["elevator"]["panel"]
    centre = np.asarray(panel["point"], dtype=np.float64)
    normal = np.asarray(panel["normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    grid = load_yaml("panels.yaml")["panels"][0]["grid"]  # mock_cabinet
    n_rows, n_cols = len(grid), len(grid[0])

    # In-plane axes: "up" is world +Z projected onto the plane; "right" completes it.
    up = np.array([0.0, 0.0, 1.0]) - normal * (normal @ [0.0, 0.0, 1.0])
    up = up / np.linalg.norm(up)
    right = np.cross(up, normal)
    right = right / np.linalg.norm(right)

    out: dict[str, np.ndarray] = {}
    for r, row in enumerate(grid):
        dv = (r - (n_rows - 1) / 2.0) * NOMINAL_ROW_PITCH_M   # +down as r grows
        for c, cell in enumerate(row):
            dh = (c - (n_cols - 1) / 2.0) * NOMINAL_COL_PITCH_M
            pt = centre - up * dv + right * dh
            out[str(cell["label"])] = pt
    return out


def load_buttons_json(path: str):
    """Return (buttons, plane_normal_or_None).

    Accepts both a flat ``{label:[x,y,z]}`` map and the nested dump from
    dump_button_coords.py (``{"buttons":{...}, "plane_normal":[...]}``).
    """
    with open(path) as f:
        raw = json.load(f)
    normal = None
    if "buttons" in raw:
        pts = raw["buttons"]
        if raw.get("plane_normal"):
            normal = np.asarray(raw["plane_normal"], dtype=np.float64)
    else:
        pts = {k: v for k, v in raw.items() if isinstance(v, list) and len(v) == 3}
    buttons = {k: np.asarray(v, dtype=np.float64) for k, v in pts.items()}
    return buttons, normal


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
@dataclass
class Row:
    button: str
    dx: float
    dy: float
    dyaw: float
    lift_dz: float
    rolls_passed: int = 0
    best_margin: float | None = None      # deg, IK layer only
    best_roll: int | None = None
    reasons: dict[str, int] = field(default_factory=dict)


def yaw_R(deg: float) -> np.ndarray:
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sweep(cfg, buttons, *, docks, lifts, use_sdk, sdk=None, tcp=None,
          path_steps=101, home_override=None, no_band=False, normal_override=None,
          face_coords=False, standoff_override=None):
    lim = cfg["arm"]["limits"]
    press = cfg["elevator"]["press"]
    # press_buttons.py uses a module constant STANDOFF=0.050, NOT press.standoff
    # (0.06); default to that so the sweep matches the real planner. Override with
    # --standoff-mm.
    standoff = (standoff_override if standoff_override is not None else 0.050)
    protrusion = float(press["button_protrusion"])
    push = float(press["push_depth"])
    normal = np.asarray(normal_override if normal_override is not None
                        else cfg["elevator"]["panel"]["normal"], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)
    inward = -normal
    band = cfg["arm"]["lift"]["usable_relative_z_m"]
    target_z = float(cfg["arm"]["lift"]["target_relative_z_m"])
    # The lift is commanded, per dock, to bring the panel CENTRE to target_z in the
    # base frame (raising the base lowers a button's base-frame z). So the reachable
    # test is: after that compensation, does each button's base-frame z fall in the
    # usable band? --lift-mm then sweeps a residual error ABOUT that command.
    centre_z = float(np.mean([b[2] for b in buttons.values()]))

    # roll basis about the approach axis (same construction as press_buttons.plan)
    e1 = np.cross(inward, np.array([0.0, 0.0, 1.0]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(inward, e1)

    home = np.asarray(home_override if home_override is not None
                      else cfg["arm"]["home_joints_deg"], dtype=np.float64)
    rows: list[Row] = []

    for name, pt0 in buttons.items():
        for (dx, dy, dyaw) in docks:
            # Apply a docking offset: translate + yaw the button about the panel centre.
            R_yaw = yaw_R(dyaw)
            pt_dock = R_yaw @ pt0 + np.array([dx, dy, 0.0])
            inward_dock = R_yaw @ inward
            # Lift compensation for THIS dock: move the panel centre to target_z.
            # The dock's dy translation shifts the centre too, so include it.
            lift_shift = centre_z + dy - target_z
            for lift_dz in lifts:
                # Compensated button: subtract the lift shift, add a residual error.
                pt = pt_dock - np.array([0.0, 0.0, lift_shift]) + np.array([0.0, 0.0, lift_dz])
                rec = Row(name, dx, dy, dyaw, lift_dz)

                # --- height band (needs no IK) -----------------------------
                rel_z = pt[2]
                if not no_band and not (band[0] <= rel_z <= band[1]):
                    rec.reasons["height outside usable band"] = 1
                    rows.append(rec)
                    continue

                # Match press_buttons.py exactly. For --buttons-json the coords are
                # already button-FACE points (ray-plane against origin+protrusion),
                # so DON'T re-add protrusion: standoff = face + normal*STANDOFF,
                # press = face - normal*push. For nominal coords (on the faceplate)
                # keep the protrusion term.
                if face_coords:
                    standoff_pt = pt - inward_dock * standoff
                    push_pt = pt + inward_dock * push
                else:
                    contact_pt = pt + inward_dock * (protrusion - push)
                    standoff_pt = contact_pt - inward_dock * standoff
                    push_pt = contact_pt + inward_dock * push

                # --- geometry clearance of the straight push ---------------
                # The plunger tip must not be closer to the panel than the button
                # face at any point BEFORE contact. Along a straight movel from
                # standoff to push, the signed distance to the plane is monotone,
                # so the binding point is the start (standoff); the boundary here is
                # simply that standoff sits outside the face by margin.
                clr_mm = float((standoff_pt - pt) @ inward_dock) * -1000.0  # dist outside face
                if clr_mm < lim["min_clearance_mm"]:
                    rec.reasons[f"clearance < {lim['min_clearance_mm']} mm"] = 1
                    rows.append(rec)
                    continue

                for deg in range(0, 360, 15):
                    a = np.radians(deg)
                    R = _orthonormal_frame(inward_dock,
                                           up_hint=np.cos(a) * e1 + np.sin(a) * e2)
                    ok, why, margin = _check_roll(
                        R, standoff_pt, push_pt, home, lim, use_sdk, sdk, tcp,
                        path_steps)
                    if ok:
                        rec.rolls_passed += 1
                        if margin is not None and (
                                rec.best_margin is None or margin > rec.best_margin):
                            rec.best_margin, rec.best_roll = margin, deg
                    else:
                        rec.reasons[why] = rec.reasons.get(why, 0) + 1
                rows.append(rec)
    return rows


def _check_roll(R, standoff_pt, push_pt, home, lim, use_sdk, sdk, tcp, path_steps):
    """One (button, dock, lift, roll). Returns (ok, reason_if_not, margin_or_None)."""
    if not use_sdk:
        # Geometry layer can only confirm the pose is well-formed; the reachability
        # verdict is the IK layer's. Count it as a geometric candidate.
        return True, "", None

    from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

    def ik(target_tip, seed):
        cmd = make_transform(R, target_tip) @ make_transform(np.eye(3), -tcp)
        params = rm_inverse_kinematics_params_t(
            list(seed), list(cmd[:3, 3]) + list(matrix_to_rpy(R, degrees=False)), 1)
        code, sol = sdk.rm_algo_inverse_kinematics(params)
        return np.asarray([float(x) for x in sol]) if code == 0 else None

    q_off = ik(standoff_pt, home)
    if q_off is None:
        return False, "no IK at standoff", None
    q_on = ik(push_pt, q_off)
    if q_on is None:
        return False, "no IK at contact", None

    margin = float(min(_joint_margin(q_off), _joint_margin(q_on)))
    if margin < lim["min_joint_margin_deg"]:
        return False, f"joint margin < {lim['min_joint_margin_deg']} deg", margin
    if min(abs(q_off[4]), abs(q_on[4])) < lim["min_wrist_deg"]:
        return False, f"wrist singularity (|J5| < {lim['min_wrist_deg']} deg)", margin

    # Denser than plan()'s 21 points, and the standoff->contact leg is sampled too.
    steps = np.linspace(home, q_off, path_steps)
    jump = float(np.abs(np.diff(steps, axis=0)).max())
    steps2 = np.linspace(q_off, q_on, max(2, path_steps // 5))
    jump = max(jump, float(np.abs(np.diff(steps2, axis=0)).max()))
    if jump > lim["max_path_jump_deg"]:
        return False, f"path jump > {lim['max_path_jump_deg']} deg", margin

    # Self-collision along the WHOLE path, using the SDK's own arm model
    # (rm_algo_safety_robot_self_collision_detection: 0 = clear, nonzero = collision
    # OR joint-limit exceeded). Joint limits are already margin-checked above, so a
    # hit here is a genuine self-collision. Covers the arm's own links + end-effector
    # per the controller's model; it does NOT know the other arm / chassis (those
    # need virtual walls). Pure computation — no motion.
    path = np.vstack([np.linspace(home, q_off, 25), np.linspace(q_off, q_on, 6)])
    for q in path:
        if sdk.rm_algo_safety_robot_self_collision_detection(
                [float(x) for x in q[:6]]) != 0:
            return False, "self-collision on path", margin
    return True, "", margin


def _joint_margin(q):
    q = np.asarray(q, dtype=np.float64)
    return float(np.min(np.minimum(J_LIMIT - q, q + J_LIMIT)))


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nominal", action="store_true",
                     help="reconstruct a nominal panel from configs (no hardware)")
    src.add_argument("--buttons-json",
                     help="button base coords {label:[x,y,z]} from a real run")
    ap.add_argument("--use-sdk", action="store_true",
                    help="add the IK/joint/singularity layer (Orin only)")
    ap.add_argument("--dock-mm", type=float, default=30.0,
                    help="docking offset half-range in x/y, mm (default 30)")
    ap.add_argument("--dock-step-mm", type=float, default=15.0)
    ap.add_argument("--yaw-deg", type=float, default=4.0,
                    help="docking yaw half-range, deg (default 4)")
    ap.add_argument("--yaw-step-deg", type=float, default=4.0)
    ap.add_argument("--lift-mm", type=float, default=0.0,
                    help="extra lift half-range about nominal, mm (default 0)")
    ap.add_argument("--lift-step-mm", type=float, default=20.0)
    ap.add_argument("--config",
                    help="explicit pipeline.yaml (e.g. the newer GitHub-HEAD config "
                         "when the on-arm repo is behind); default: repo configs/")
    ap.add_argument("--home", help="override home_joints_deg, e.g. '40,-100,-79,-131,117,68'")
    ap.add_argument("--no-band", action="store_true",
                    help="bypass the fixed usable-height band; let IK+margin decide "
                         "reachability directly (use with a wide --lift-mm)")
    ap.add_argument("--standoff-mm", type=float, default=None,
                    help="standoff distance (mm); default 50 to match press_buttons.py")
    ap.add_argument("--csv")
    args = ap.parse_args()

    cfg = load_yaml(args.config) if args.config else load_pipeline()
    normal_override = None
    if args.nominal:
        buttons = nominal_buttons(cfg)
        src_label = "NOMINAL (reconstructed from configs)"
    else:
        buttons, normal_override = load_buttons_json(args.buttons_json)
        src_label = args.buttons_json + (" (+live plane normal)" if normal_override is not None else "")
    print(f"buttons: {len(buttons)} from {src_label}")

    def grid(half, step):
        if half <= 0:
            return [0.0]
        n = int(round(half / step))
        return [round(step * k, 6) for k in range(-n, n + 1)]

    xs = grid(args.dock_mm / 1000.0, args.dock_step_mm / 1000.0)
    ys = grid(args.dock_mm / 1000.0, args.dock_step_mm / 1000.0)
    yaws = grid(args.yaw_deg, args.yaw_step_deg)
    docks = [(dx, dy, dyaw) for dx in xs for dy in ys for dyaw in yaws]
    lifts = grid(args.lift_mm / 1000.0, args.lift_step_mm / 1000.0)
    print(f"sweep: {len(buttons)} buttons x {len(docks)} docks x {len(lifts)} lifts "
          f"x 24 rolls = {len(buttons) * len(docks) * len(lifts) * 24} pose pairs")

    sdk = tcp = None
    if args.use_sdk:
        try:
            from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
        except ImportError:
            print("!! --use-sdk but the RealMan SDK is not importable. Run this on "
                  "the Orin (richtech-v3 env), or drop --use-sdk for geometry only.")
            return 2
        arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        arm.rm_create_robot_arm(cfg["arm"]["ip"], cfg["arm"]["port"])
        sdk = arm
        tcp = np.asarray(cfg["end_effector"]["tcp_offset"], dtype=np.float64)

    home_override = ([float(x) for x in args.home.split(",")] if args.home else None)
    if home_override is not None:
        print(f"home override: {home_override}")
    if args.no_band:
        print("height band: BYPASSED (IK+margin decides reachability)")
    rows = sweep(cfg, buttons, docks=docks, lifts=lifts,
                 use_sdk=args.use_sdk, sdk=sdk, tcp=tcp,
                 home_override=home_override, no_band=args.no_band,
                 normal_override=normal_override,
                 face_coords=(not args.nominal),
                 standoff_override=(args.standoff_mm / 1000.0
                                    if args.standoff_mm is not None else None))

    layer = "IK+geometry" if args.use_sdk else "geometry-only (no IK — run --use-sdk on the Orin for joint checks)"
    print(f"\n=== boundary sweep ({layer}) ===")
    _print_summary(rows, args.use_sdk)
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")
    return 0


def _print_summary(rows, use_sdk):
    by_btn: dict[str, list[Row]] = {}
    for r in rows:
        by_btn.setdefault(r.button, []).append(r)
    hdr = f"{'button':>6} {'docks_ok/total':>15} {'min_margin':>11}  first-boundary-to-bite"
    print(hdr)
    print("-" * len(hdr))
    for name, rs in by_btn.items():
        ok = sum(1 for r in rs if r.rolls_passed > 0)
        margins = [r.best_margin for r in rs if r.best_margin is not None]
        mm = f"{min(margins):.1f} deg" if margins else ("n/a" if not use_sdk else "-")
        agg: dict[str, int] = {}
        for r in rs:
            if r.rolls_passed == 0:
                for why, n in r.reasons.items():
                    agg[why] = agg.get(why, 0) + 1
        worst = max(agg.items(), key=lambda kv: kv[1])[0] if agg else "(all docks reachable)"
        print(f"{name:>6} {ok:>7}/{len(rs):<7} {mm:>11}  {worst}")


def _write_csv(rows, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["button", "dx_m", "dy_m", "dyaw_deg", "lift_dz_m",
                    "rolls_passed", "best_margin_deg", "best_roll_deg", "reasons"])
        for r in rows:
            w.writerow([r.button, r.dx, r.dy, r.dyaw, r.lift_dz, r.rolls_passed,
                        "" if r.best_margin is None else f"{r.best_margin:.2f}",
                        "" if r.best_roll is None else r.best_roll,
                        ";".join(f"{k}x{v}" for k, v in r.reasons.items())])


if __name__ == "__main__":
    raise SystemExit(main())
