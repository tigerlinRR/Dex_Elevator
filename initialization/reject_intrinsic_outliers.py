#!/usr/bin/env python3
"""Re-solve intrinsics with per-view OUTLIER REJECTION, verified on held-out data.

`calibrate_intrinsics.py` fits every view it captured. That is the right default —
it should not silently discard the operator's data — but a single bad view (motion
blur, a half-seen board, the arm still settling) degrades the whole solve, and the
headline RMS hides it because the mean is dominated by the good views. Measured on
the chest camera's 2026-08 run: dropping 25 of 88 views took RMS 0.698 -> **0.383 px**
and the leave-one-out worst case 33.6 -> 1.81 mm.

So this is a separate, RE-RUNNABLE step over the SAME captured images.

Three things it does that a plain re-fit does not:

1. **Caches the corner detection.** That is the slow part (~8-12 min for 80 views of
   a 14x9 board); the fit itself is seconds. Cached, trying another threshold is
   instant, so the threshold can be chosen by looking at the data rather than guessed
   once and lived with.
2. **Verifies on HELD-OUT views.** Rejecting views always improves RMS — it is the
   same data the fit minimised — so RMS cannot say whether the solve got better or
   merely over-fitted. Each pass reports a split-half cross-validation: refit on each
   half, project a common grid of rays with both, and compare in pixels. That number
   is allowed to get WORSE, and if it does the rejection went too far.
3. **Keeps everything.** The pre-rejection `.npz` is backed up and the rejected views
   are listed, so the decision is auditable and reversible.

    python3 initialization/reject_intrinsic_outliers.py --camera cam_arm
    python3 initialization/reject_intrinsic_outliers.py --camera cam_arm --factor 1.5
    python3 initialization/reject_intrinsic_outliers.py --camera cam_arm --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from calibration.board import BoardDetector, CharucoBoard
from calibration.io import CALIB_DIR, save_intrinsics
from calibration.validate import save_report, validate_intrinsics


def _detect_all(paths: list[Path], board: CharucoBoard, cache: Path) -> dict:
    """Detect ChArUco corners in every image, caching the result (the slow step)."""
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        if int(data["num"]) == len(paths):
            print(f"[detect] reusing cached corners for {len(paths)} views ({cache.name})")
            return {"obj": list(data["obj"]), "img": list(data["img"]),
                    "names": [str(n) for n in data["names"]]}
        print("[detect] cache is for a different view count — re-detecting")

    detector = BoardDetector(board=board, accurate=True, image_is_rgb=False)
    obj, img, names = [], [], []
    for i, p in enumerate(paths, 1):
        im = cv2.imread(str(p))
        if im is None:
            continue
        found = detector.detect_corners(im)
        if found is None:
            print(f"[detect] {p.name}: board not found — dropped")
            continue
        corners, ids, ocv_board = found
        obj.append(ocv_board.getChessboardCorners()[ids.flatten()].astype(np.float32))
        img.append(np.asarray(corners, dtype=np.float32))
        names.append(p.name)
        if i % 10 == 0:
            print(f"[detect] {i}/{len(paths)} views", flush=True)
    np.savez(cache, obj=np.array(obj, dtype=object), img=np.array(img, dtype=object),
             names=np.array(names), num=len(paths))
    print(f"[detect] {len(obj)} usable views cached -> {cache.name}")
    return {"obj": obj, "img": img, "names": names}


def _fit(obj, img, size):
    """calibrateCameraExtended -> (rms, K, dist, per-view errors)."""
    rms, K, dist, _rv, _tv, _si, _se, per_view = cv2.calibrateCameraExtended(
        obj, img, size, None, None
    )
    return rms, K, np.asarray(dist).reshape(-1), np.asarray(per_view).reshape(-1)


def _crossval_px(obj, img, size) -> float | None:
    """Split-half cross-validation: refit on each half, compare projections (px, p95).

    The held-out metric. RMS always improves when views are dropped, so it cannot tell
    a better solve from an over-fitted one; this can, because the two halves are fitted
    independently.
    """
    n = len(obj)
    if n < 8:
        return None
    half = n // 2
    fits = []
    for lo, hi in ((0, half), (half, n)):
        try:
            _r, K, d, _pv = _fit(obj[lo:hi], img[lo:hi], size)
        except cv2.error:
            return None
        fits.append((K, d))
    xs = np.linspace(-0.6, 0.6, 25)
    ys = np.linspace(-0.45, 0.45, 25)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 1).astype(np.float64)
    zero = np.zeros(3)
    p1, _ = cv2.projectPoints(pts, zero, zero, fits[0][0], fits[0][1])
    p2, _ = cv2.projectPoints(pts, zero, zero, fits[1][0], fits[1][1])
    d = np.linalg.norm(p1.reshape(-1, 2) - p2.reshape(-1, 2), axis=1)
    return float(np.percentile(d, 95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, help="camera id (e.g. cam_arm)")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="drop views whose per-view error exceeds FACTOR x the median (default 2.0)")
    ap.add_argument("--passes", type=int, default=2,
                    help="rejection passes; the median moves after the first cut (default 2)")
    ap.add_argument("--min-views", type=int, default=20, help="never go below this many views")
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--squares-x", type=int, default=14)
    ap.add_argument("--squares-y", type=int, default=9)
    ap.add_argument("--square-len", type=float, default=0.040)
    ap.add_argument("--marker-len", type=float, default=0.030)
    ap.add_argument("--dictionary", default="DICT_5X5_100")
    args = ap.parse_args()

    board = CharucoBoard(
        squares_x=args.squares_x, squares_y=args.squares_y,
        square_length=args.square_len, marker_length=args.marker_len,
        dictionary=getattr(cv2.aruco, args.dictionary),
    )

    images_dir = CALIB_DIR / f"{args.camera}_intrinsics_images"
    paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if not paths:
        print(f"no captured views under {images_dir}")
        return 1
    size_img = cv2.imread(str(paths[0]))
    size = (size_img.shape[1], size_img.shape[0])
    print(f"{len(paths)} views, {size[0]}x{size[1]}")

    det = _detect_all(paths, board, CALIB_DIR / f"{args.camera}_intrinsics_corners.npz")
    obj, img, names = det["obj"], det["img"], det["names"]

    rms, K, dist, per_view = _fit(obj, img, size)
    cv0 = _crossval_px(obj, img, size)
    print(f"\nbaseline: {len(obj)} views  RMS {rms:.4f} px  "
          f"(per-view median {np.median(per_view):.3f}, max {per_view.max():.3f})"
          + (f"  cross-val p95 {cv0:.3f} px" if cv0 else ""))

    kept = list(range(len(obj)))
    dropped: list[tuple[str, float]] = []
    for p in range(args.passes):
        pv = per_view
        med = float(np.median(pv))
        limit = args.factor * med
        keep_now = [i for i, e in enumerate(pv) if e <= limit]
        if len(keep_now) < args.min_views or len(keep_now) == len(pv):
            print(f"pass {p + 1}: nothing further to drop at {args.factor}x median "
                  f"({limit:.3f} px)")
            break
        for i, e in enumerate(pv):
            if e > limit:
                dropped.append((names[kept[i]], float(e)))
        kept = [kept[i] for i in keep_now]
        obj = [obj[i] for i in keep_now]
        img = [img[i] for i in keep_now]
        rms, K, dist, per_view = _fit(obj, img, size)
        cvn = _crossval_px(obj, img, size)
        print(f"pass {p + 1}: dropped {len(pv) - len(keep_now)} above {limit:.3f} px "
              f"-> {len(obj)} views  RMS {rms:.4f} px"
              + (f"  cross-val p95 {cvn:.3f} px" if cvn else ""))

    cv_final = _crossval_px(obj, img, size)
    print(f"\nfinal: {len(obj)} views  RMS {rms:.4f} px")
    print(f"  K =\n{K}")
    print(f"  dist = {dist}")
    if dropped:
        print(f"  dropped {len(dropped)} views:")
        for n, e in sorted(dropped, key=lambda t: -t[1]):
            print(f"    {n}  per-view {e:.3f} px")
    # The held-out number is the one that decides whether this helped. RMS improving
    # is not evidence — it is the objective the fit just minimised over fewer views.
    if cv0 is not None and cv_final is not None:
        verdict = "BETTER" if cv_final < cv0 else "WORSE — rejection went too far"
        print(f"  cross-val p95: {cv0:.3f} -> {cv_final:.3f} px  ({verdict})")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    out = CALIB_DIR / f"{args.camera}_intrinsics.npz"
    backup = CALIB_DIR / f"{args.camera}_intrinsics_before_outlier_reject.npz"
    if out.exists() and not backup.exists():
        shutil.copy2(out, backup)
        print(f"\nbacked up the pre-rejection solve -> {backup.name}")
    save_intrinsics(out, K, dist, size[0], size[1], rms=rms)
    print(f"saved -> {out}")

    frames = [cv2.imread(str(images_dir / n)) for n in [names[i] for i in kept]]
    report = validate_intrinsics(K, dist, frames, board=board, image_size=size,
                                 image_is_rgb=False,
                                 plot_dir=CALIB_DIR / f"{args.camera}_intrinsics_validation")
    print(report.summary())
    save_report(report, CALIB_DIR / f"{args.camera}_intrinsics_validation/report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
