"""Validate a finished calibration — run this AFTER calibrating.

Implements the reprojection-residual checks from the lensboy calibration guide
(https://robertleoj.github.io/lensboy/calibration_guide.html) for **intrinsics**,
plus a complementary set of geometric checks for the **eye-to-hand extrinsics**
(which that guide does not cover). Pure numpy + cv2; plots are optional
(matplotlib, guarded) so the numeric report always works headless.

Intrinsics checks (per the guide):
  * mean / worst reprojection error (px), after 5-sigma outlier filtering
  * per-frame RMS residuals
  * Gaussian MAD sigma of the residual components (histogram should be ~Gaussian)
  * 2D residual isotropy (std_x vs std_y)
  * spatial residual grid: edge-vs-center bias (growth toward edges = underfitting)
  * image coverage of detected corners
  * split-half cross-validation: refit on each half, compare projections
    (widespread differences > 2x MAD sigma => over/under-fitting)

Extrinsics checks:
  * gripper_T_target consistency residual (translation mm + rotation deg)
  * leave-one-out hold-out consistency (does the solve generalize?)
  * cross-method spread (TSAI/PARK/HORAUD/DANIILIDIS agreement)

Caveat: reprojection validation re-solves each view's pose with PnP, so for a
near-planar board a *uniform* focal error is largely absorbed by pose depth (the
focal/depth ambiguity) and shows up only weakly. Capturing strongly-angled
close-ups (as the guide recommends) is what makes focal length observable; the
residual checks here still catch principal-point, anisotropic-focal and
distortion errors cleanly.

NOT implemented: the guide's Legendre "target warp" estimation (a lensboy-specific
near-planar deflection fit) — out of scope; noted so it isn't mistaken for done.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.extrinsic import consistency_residual, solve_eye_to_hand
from core.transforms import invert

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


# ==========================================================================
# Thresholds (guide-derived defaults; tune per camera/resolution)
# ==========================================================================
@dataclass
class IntrinsicThresholds:
    # The guide's high-end rig reached ~0.1-0.17 px; these are practical pass/warn
    # bands for a 1280x720 industrial RGB stream.
    mean_reproj_warn_px: float = 0.5
    mean_reproj_fail_px: float = 1.0
    worst_reproj_warn_px: float = 2.0
    outlier_frac_warn: float = 0.05      # guide: >5% per-frame is a red flag
    isotropy_warn_ratio: float = 2.0     # max(std_x/std_y, std_y/std_x)
    edge_center_warn_ratio: float = 3.0  # edge residual mean / center residual mean
    crossval_factor: float = 2.0         # concern if diffs > factor x MAD sigma
    crossval_min_sigma_px: float = 0.1   # floor for the cross-val bound (guide ref sigma)
    residual_tail_warn_ratio: float = 1.4  # std/MAD-sigma; >1 => heavy-tailed (non-Gaussian)
    per_frame_rms_warn_factor: float = 3.0  # warn if a frame RMS > factor x median frame RMS
    min_coverage_frac: float = 0.6       # fraction of an 8x6 grid hit by corners
    outlier_sigma: float = 5.0           # 5-sigma filtering (guide default)


@dataclass
class ExtrinsicThresholds:
    trans_resid_warn_m: float = 0.005    # 5 mm
    trans_resid_fail_m: float = 0.015    # 15 mm
    rot_resid_warn_deg: float = 1.0
    rot_resid_fail_deg: float = 3.0
    crossmethod_warn_m: float = 0.010    # 10 mm spread across methods
    holdout_warn_m: float = 0.010        # hold-out consistency degradation


@dataclass
class ValidationReport:
    part: str                            # "intrinsics" | "extrinsics"
    status: str = "pass"                 # "pass" | "warn" | "fail"
    metrics: dict = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)

    def _bump(self, level: str) -> None:
        order = {"pass": 0, "warn": 1, "fail": 2}
        if order[level] > order[self.status]:
            self.status = level

    def check(self, ok_level: str, msg: str) -> None:
        """Record a message and raise the overall status to ok_level if worse."""
        self._bump(ok_level)
        self.messages.append(f"[{ok_level.upper()}] {msg}")

    def to_dict(self) -> dict:
        return {"part": self.part, "status": self.status,
                "metrics": self.metrics, "messages": self.messages, "plots": self.plots}

    def summary(self) -> str:
        icon = {"pass": "✓", "warn": "⚠", "fail": "✗"}[self.status]
        lines = [f"{icon} {self.part} validation: {self.status.upper()}"]
        lines += [f"    {m}" for m in self.messages]
        return "\n".join(lines)


def save_report(report: ValidationReport, path: str | Path) -> Path:
    """Write a validation report to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    return path


# ==========================================================================
# Shared residual computation
# ==========================================================================
def _detect_views(images: Sequence[np.ndarray], board: CharucoBoard, accurate: bool,
                   image_is_rgb: bool):
    """Detect ChArUco corners in each image -> list of (objp(M,3), imgp(M,1,2))."""
    det = BoardDetector(board=board, accurate=accurate, image_is_rgb=image_is_rgb)
    views = []
    for img in images:
        found = det.detect_corners(img)
        if found is None:
            continue
        corners, ids, ocv_board = found
        objp = ocv_board.getChessboardCorners()[ids.flatten()].astype(np.float32)
        views.append((objp, np.asarray(corners, dtype=np.float32)))
    return views


def _view_residuals(K: np.ndarray, dist: np.ndarray, objp: np.ndarray, imgp: np.ndarray):
    """Signed reprojection residual vectors (M,2) for one view, or None."""
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
    if not ok:
        return None
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    return imgp.reshape(-1, 2) - proj.reshape(-1, 2)


def _mad_sigma(values: np.ndarray) -> float:
    """Gaussian-consistent sigma from the median absolute deviation."""
    med = np.median(values)
    return float(1.4826 * np.median(np.abs(values - med)))


# ==========================================================================
# Intrinsics validation
# ==========================================================================
def validate_intrinsics(
    K: np.ndarray,
    dist: np.ndarray,
    images: Sequence[np.ndarray],
    board: Optional[CharucoBoard] = None,
    image_size: Optional[tuple[int, int]] = None,
    thresholds: Optional[IntrinsicThresholds] = None,
    accurate: bool = True,
    image_is_rgb: bool = True,
    plot_dir: Optional[str | Path] = None,
) -> ValidationReport:
    """Validate intrinsics (K, dist) against the calibration images. See module docs."""
    if cv2 is None:
        raise ImportError("opencv-contrib-python is required for validation.")
    board = board or CharucoBoard()
    th = thresholds or IntrinsicThresholds()
    K = np.asarray(K, dtype=np.float64)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    rep = ValidationReport(part="intrinsics")

    views = _detect_views(images, board, accurate, image_is_rgb)
    if len(views) < 3:
        rep.check("fail", f"only {len(views)} board views detected (need >= 3)")
        return rep
    if image_size is None:
        h, w = images[0].shape[:2]
        image_size = (w, h)
    w, h = image_size

    # Per-view + pooled residuals.
    per_frame_rms, all_res, all_pts = [], [], []
    for objp, imgp in views:
        res = _view_residuals(K, dist, objp, imgp)
        if res is None:
            continue
        norms = np.linalg.norm(res, axis=1)
        per_frame_rms.append(float(np.sqrt(np.mean(norms ** 2))))
        all_res.append(res)
        all_pts.append(imgp.reshape(-1, 2))
    if not all_res:
        rep.check("fail", f"{len(views)} views detected but PnP failed on all of them")
        return rep
    res = np.vstack(all_res)                 # (N,2) signed
    pts = np.vstack(all_pts)                 # (N,2) pixel locations
    norms = np.linalg.norm(res, axis=1)

    # 5-sigma outlier filtering on pooled components (guide default).
    comps = res.reshape(-1)
    sigma = _mad_sigma(comps)
    keep = np.all(np.abs(res) <= th.outlier_sigma * sigma, axis=1) if sigma > 0 else np.ones(len(res), bool)
    outlier_frac = float(1.0 - keep.mean())
    inl = res[keep]
    inl_norms = np.linalg.norm(inl, axis=1)

    mean_reproj = float(inl_norms.mean())
    worst_reproj = float(inl_norms.max())
    mad_sigma = _mad_sigma(inl.reshape(-1))
    std_x, std_y = float(inl[:, 0].std()), float(inl[:, 1].std())
    isotropy = max(std_x, std_y) / max(min(std_x, std_y), 1e-9)
    # Heavy-tail / non-Gaussian proxy: for Gaussian residuals std == MAD-sigma, so
    # std/MAD-sigma ~ 1.0; a larger ratio means heavy tails (detector systematics /
    # underfitting). This is the numeric stand-in for the guide's "histogram should
    # be ~Gaussian" check so it works headless (the histogram is only a plot).
    tail_ratio = float(inl.reshape(-1).std()) / max(mad_sigma, 1e-9)

    # Edge-vs-center bias: residual norm mean in outer vs inner radial band.
    cx, cy = K[0, 2], K[1, 2]
    r = np.linalg.norm(pts[keep] - np.array([cx, cy]), axis=1)
    r_norm = r / (0.5 * np.hypot(w, h))
    center_mask, edge_mask = r_norm < 0.5, r_norm >= 0.5
    center_mean = float(inl_norms[center_mask].mean()) if center_mask.any() else 0.0
    edge_mean = float(inl_norms[edge_mask].mean()) if edge_mask.any() else 0.0
    # Only meaningful when both bands are populated; else neutral (no false warn).
    edge_center_ratio = (edge_mean / center_mean) if (center_mask.any() and edge_mask.any()) else 1.0

    # Coverage over an 8x6 grid.
    gx, gy = 8, 6
    cells = {(int(min(px / w * gx, gx - 1)), int(min(py / h * gy, gy - 1))) for px, py in pts}
    coverage = len(cells) / (gx * gy)

    rep.metrics.update({
        "num_views": len(views), "num_points": int(len(res)),
        "mean_reproj_px": round(mean_reproj, 4), "worst_reproj_px": round(worst_reproj, 4),
        "mad_sigma_px": round(mad_sigma, 4), "outlier_frac": round(outlier_frac, 4),
        "isotropy_ratio": round(isotropy, 3), "std_x_px": round(std_x, 4), "std_y_px": round(std_y, 4),
        "residual_tail_ratio": round(tail_ratio, 3),
        "edge_center_ratio": round(edge_center_ratio, 3),
        "coverage_frac": round(coverage, 3),
        "per_frame_rms_px": [round(v, 4) for v in per_frame_rms],
        "worst_frame_rms_px": round(max(per_frame_rms), 4),
    })

    # --- acceptance checks ---
    if mean_reproj > th.mean_reproj_fail_px:
        rep.check("fail", f"mean reprojection {mean_reproj:.3f} px > {th.mean_reproj_fail_px} px")
    elif mean_reproj > th.mean_reproj_warn_px:
        rep.check("warn", f"mean reprojection {mean_reproj:.3f} px > {th.mean_reproj_warn_px} px")
    else:
        rep.check("pass", f"mean reprojection {mean_reproj:.3f} px (MAD sigma {mad_sigma:.3f} px)")

    if worst_reproj > th.worst_reproj_warn_px:
        rep.check("warn", f"worst reprojection {worst_reproj:.3f} px > {th.worst_reproj_warn_px} px (inspect that frame)")
    if outlier_frac > th.outlier_frac_warn:
        rep.check("warn", f"{outlier_frac*100:.1f}% outliers > {th.outlier_frac_warn*100:.0f}% (data quality)")
    if isotropy > th.isotropy_warn_ratio:
        rep.check("warn", f"anisotropic residuals (ratio {isotropy:.2f}) — detector/target issue")
    if tail_ratio > th.residual_tail_warn_ratio:
        rep.check("warn", f"non-Gaussian residuals (std/MAD-sigma {tail_ratio:.2f} > "
                          f"{th.residual_tail_warn_ratio}) — heavy tails / detector systematics")
    if edge_center_ratio > th.edge_center_warn_ratio:
        rep.check("warn", f"residuals grow toward edges (x{edge_center_ratio:.1f}) — possible underfitting")
    # Per-frame RMS: flag a single uniformly-bad frame (defocus/motion blur) even
    # if no single point is a 5-sigma outlier. Relative (robust across resolutions).
    med_rms = float(np.median(per_frame_rms))
    worst_idx = int(np.argmax(per_frame_rms))
    if med_rms > 0 and max(per_frame_rms) > th.per_frame_rms_warn_factor * med_rms:
        rep.check("warn", f"frame {worst_idx} RMS {max(per_frame_rms):.3f} px > "
                          f"{th.per_frame_rms_warn_factor}x median ({med_rms:.3f} px) — inspect that frame")
    if coverage < th.min_coverage_frac:
        rep.check("warn", f"image coverage {coverage*100:.0f}% < {th.min_coverage_frac*100:.0f}% — add edge/corner views")

    # Split-half cross-validation (refit + compare projections).
    cv_diff = _cross_validate_intrinsics(views, image_size)
    if cv_diff is not None:
        rep.metrics["crossval_median_px"] = round(cv_diff["median"], 4)
        rep.metrics["crossval_p95_px"] = round(cv_diff["p95"], 4)
        # Floor the bound: a tiny/degenerate MAD sigma must not make a clean fit fail
        # this trivially (the guide's "2x MAD sigma" assumes a normal-magnitude sigma).
        concern = th.crossval_factor * max(mad_sigma, th.crossval_min_sigma_px)
        if cv_diff["p95"] > concern:
            rep.check("warn", f"cross-val p95 diff {cv_diff['p95']:.3f} px > {concern:.3f} px "
                              f"(2x MAD sigma floored at {th.crossval_min_sigma_px}) — over/under-fitting")
        else:
            rep.check("pass", f"cross-val p95 diff {cv_diff['p95']:.3f} px <= {concern:.3f} px")

    if plot_dir is not None:
        rep.plots = _plot_intrinsics(rep, inl, pts[keep], (w, h), per_frame_rms, Path(plot_dir))
    return rep


def _cross_validate_intrinsics(views, image_size) -> Optional[dict]:
    """Refit on each half, project a normalized grid with both, compare (px)."""
    if len(views) < 6:
        return None
    half = len(views) // 2
    fits = []
    for subset in (views[:half], views[half:]):
        objs = [v[0] for v in subset]
        imgs = [v[1] for v in subset]
        try:
            _rms, K, d, _r, _t = cv2.calibrateCamera(objs, imgs, image_size, None, None)
        except cv2.error:
            return None
        fits.append((np.asarray(K, np.float64), np.asarray(d, np.float64).reshape(-1)))
    # Grid of camera-frame rays at z=1 spanning a generous FOV.
    xs = np.linspace(-0.6, 0.6, 25)
    ys = np.linspace(-0.45, 0.45, 25)
    gx, gy = np.meshgrid(xs, ys)
    pts3d = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1).astype(np.float64)
    zero = np.zeros(3)
    p1, _ = cv2.projectPoints(pts3d, zero, zero, fits[0][0], fits[0][1])
    p2, _ = cv2.projectPoints(pts3d, zero, zero, fits[1][0], fits[1][1])
    d = np.linalg.norm(p1.reshape(-1, 2) - p2.reshape(-1, 2), axis=1)
    return {"median": float(np.median(d)), "p95": float(np.percentile(d, 95))}


# ==========================================================================
# Extrinsics validation
# ==========================================================================
def validate_extrinsics(
    base_T_camera: np.ndarray,
    base_T_gripper: list[np.ndarray],
    cam_T_target: list[np.ndarray],
    all_methods: Optional[dict[str, np.ndarray]] = None,
    thresholds: Optional[ExtrinsicThresholds] = None,
) -> ValidationReport:
    """Validate the eye-to-hand extrinsic. See module docs."""
    th = thresholds or ExtrinsicThresholds()
    rep = ValidationReport(part="extrinsics")
    n = len(base_T_gripper)
    if n < 3:
        rep.check("fail", f"only {n} samples (need >= 3)")
        return rep

    res = consistency_residual(base_T_camera, base_T_gripper, cam_T_target)
    tmean, tmax = res["trans_m"]["mean"], res["trans_m"]["max"]
    rmean, rmax = res["rot_deg"]["mean"], res["rot_deg"]["max"]
    rep.metrics.update({
        "num_samples": n,
        "trans_resid_mean_mm": round(tmean * 1000, 3), "trans_resid_max_mm": round(tmax * 1000, 3),
        "rot_resid_mean_deg": round(rmean, 4), "rot_resid_max_deg": round(rmax, 4),
    })
    if tmean > th.trans_resid_fail_m:
        rep.check("fail", f"translation consistency {tmean*1000:.2f} mm > {th.trans_resid_fail_m*1000:.0f} mm")
    elif tmean > th.trans_resid_warn_m:
        rep.check("warn", f"translation consistency {tmean*1000:.2f} mm > {th.trans_resid_warn_m*1000:.0f} mm")
    else:
        rep.check("pass", f"translation consistency {tmean*1000:.2f} mm (rot {rmean:.3f} deg)")
    if rmean > th.rot_resid_fail_deg:
        rep.check("fail", f"rotation consistency {rmean:.3f} deg > {th.rot_resid_fail_deg} deg")
    elif rmean > th.rot_resid_warn_deg:
        rep.check("warn", f"rotation consistency {rmean:.3f} deg > {th.rot_resid_warn_deg} deg")

    # Leave-one-out hold-out: does the solve GENERALIZE to a pose it never saw?
    # For each i, refit X on the OTHER n-1 samples, build the gripper_T_target
    # reference from those training samples, then measure how far the HELD-OUT
    # sample's predicted gripper_T_target lands from that reference. This isolates
    # sample i (unlike a full-set consistency, which would dilute it).
    if n >= 6:
        worst_holdout = 0.0
        for i in range(n):
            idx = [k for k in range(n) if k != i]
            bg = [base_T_gripper[k] for k in idx]
            ct = [cam_T_target[k] for k in idx]
            try:
                X = solve_eye_to_hand(bg, ct)
            except Exception:
                continue
            ref = np.mean([(invert(bg[j]) @ X @ ct[j])[:3, 3] for j in range(len(bg))], axis=0)
            held = invert(base_T_gripper[i]) @ X @ cam_T_target[i]
            worst_holdout = max(worst_holdout, float(np.linalg.norm(held[:3, 3] - ref)))
        rep.metrics["holdout_worst_mm"] = round(worst_holdout * 1000, 3)
        if worst_holdout > th.holdout_warn_m:
            rep.check("warn", f"leave-one-out hold-out worst {worst_holdout*1000:.2f} mm "
                              f"> {th.holdout_warn_m*1000:.0f} mm — solve may not generalize / too few poses")

    # Cross-method spread — only among CONVERGED methods. Some OpenCV methods
    # (TSAI/HORAUD/DANIILIDIS) diverge by hundreds of mm for down-looking cameras;
    # including them would always trip the check. Keep methods whose own
    # consistency residual is sane, then measure how much those agree.
    if all_methods and len(all_methods) > 1:
        method_res = {
            name: consistency_residual(X, base_T_gripper, cam_T_target)["trans_m"]["mean"]
            for name, X in all_methods.items()
        }
        best = min(method_res.values())
        cap = max(3.0 * best, 0.02)  # within 3x the best, or 20 mm absolute
        converged = {n: all_methods[n] for n, r in method_res.items() if r <= cap}
        rep.metrics["crossmethod_converged"] = sorted(converged)
        if len(converged) >= 2:
            ts = np.array([X[:3, 3] for X in converged.values()])
            spread = float(np.linalg.norm(ts - ts.mean(axis=0), axis=1).max())
            rep.metrics["crossmethod_spread_mm"] = round(spread * 1000, 3)
            if spread > th.crossmethod_warn_m:
                rep.check("warn", f"converged methods disagree by {spread*1000:.1f} mm "
                                  f"> {th.crossmethod_warn_m*1000:.0f} mm — poorly conditioned pose set")
            else:
                rep.check("pass", f"{len(converged)} methods agree within {spread*1000:.2f} mm")
        else:
            rep.metrics["crossmethod_spread_mm"] = 0.0
            rep.check("pass", f"only {sorted(converged)} converged; cross-method check skipped")
    return rep


# ==========================================================================
# Plotting (optional)
# ==========================================================================
def _plot_intrinsics(rep, res, pts, image_size, per_frame_rms, out_dir: Path) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        rep.messages.append("[INFO] matplotlib not installed — skipping plots")
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = image_size
    norms = np.linalg.norm(res, axis=1)
    saved = []

    # 1) residual histogram (should be ~Gaussian) + 2) 2D scatter (isotropy)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.hist(res.reshape(-1), bins=60, color="#4C78A8")
    a1.set_title("Residual components (px) — should be ~Gaussian")
    a2.scatter(res[:, 0], res[:, 1], s=2, alpha=0.3, color="#E45756")
    a2.set_aspect("equal"); a2.set_title("2D residuals — should be isotropic")
    a2.axhline(0, lw=0.5, color="k"); a2.axvline(0, lw=0.5, color="k")
    p = out_dir / "intrinsics_residuals.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    saved.append(str(p))

    # 3) spatial residual quiver + coverage
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.quiver(pts[:, 0], pts[:, 1], res[:, 0], res[:, 1], norms, cmap="viridis",
              angles="xy", scale_units="xy")
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.set_title("Residual field (bias toward edges = underfit)")
    p = out_dir / "intrinsics_residual_field.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    saved.append(str(p))

    # 4) per-frame RMS bar
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(range(len(per_frame_rms)), per_frame_rms, color="#54A24B")
    ax.set_xlabel("frame"); ax.set_ylabel("RMS (px)"); ax.set_title("Per-frame RMS residual")
    p = out_dir / "intrinsics_per_frame_rms.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    saved.append(str(p))
    return saved
