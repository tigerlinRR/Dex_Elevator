"""Persistence for both calibration parts — write once, load everywhere.

Layout under ``data/calibration/`` (the single shared source of truth that the
runtime ``CameraManager`` loads for every camera):

* Part 1 — intrinsics:  ``<cam>_intrinsics.npz``  (camera_matrix, dist_coeffs, width, height)
* Part 2 — extrinsics:  ``<cam>.npy``             (4x4 base_T_camera)
                        ``<cam>.calib.npz``        (all methods + residuals, audit)

Self-contained: depends only on ``numpy`` and ``core``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.config import REPO_ROOT
from core.types import CameraIntrinsics

CALIB_DIR = REPO_ROOT / "data" / "calibration"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else CALIB_DIR / path


# --------------------------------------------------------------------------
# Part 1 — intrinsics (K + distortion)
# --------------------------------------------------------------------------
def save_intrinsics(
    path: str | Path,
    K: np.ndarray,
    dist: np.ndarray,
    width: int,
    height: int,
    rms: float | None = None,
) -> Path:
    """Save camera matrix + distortion to an ``.npz`` (the shared intrinsics file)."""
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        camera_matrix=np.asarray(K, dtype=np.float64),
        dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
        width=int(width),
        height=int(height),
        rms=float("nan") if rms is None else float(rms),
    )
    return out


def load_intrinsics(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(K, dist)`` from an intrinsics ``.npz``."""
    data = np.load(_resolve(path))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    return K, dist


def load_camera_intrinsics(
    path: str | Path,
    width: int | None = None,
    height: int | None = None,
) -> CameraIntrinsics:
    """Load a full :class:`CameraIntrinsics` (K + distortion) from the npz.

    Loads at the **stored** calibration resolution. If ``width``/``height`` are
    given and differ from the stored resolution, ``K`` is **rescaled** to that
    resolution (``CameraIntrinsics.scaled_to`` — assumes a pure resize, not a
    cropped sensor profile); distortion is resolution-independent and unchanged.
    """
    data = np.load(_resolve(path))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["dist_coeffs"], dtype=np.float64).reshape(-1)
    stored_w = int(data["width"]) if "width" in data else width
    stored_h = int(data["height"]) if "height" in data else height
    if stored_w is None or stored_h is None:
        raise ValueError("Resolution not in npz; pass width/height explicitly.")
    intr = CameraIntrinsics.from_matrix(K, width=stored_w, height=stored_h, dist=dist)
    if width is not None and height is not None and (width, height) != (stored_w, stored_h):
        intr = intr.scaled_to(width, height)
    return intr


# --------------------------------------------------------------------------
# Part 2 — extrinsics (base_T_camera)
# --------------------------------------------------------------------------
def save_extrinsic(path: str | Path, base_T_camera: np.ndarray) -> Path:
    out = _resolve(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, base_T_camera.astype(np.float64))
    return out


def load_extrinsic(path: str | Path) -> np.ndarray:
    src = _resolve(path)
    if not src.exists():
        raise FileNotFoundError(f"No extrinsic at {src}. Run run_calibration.py first.")
    return np.load(src)


def save_calibration(
    path: str | Path,
    base_T_camera: np.ndarray,
    method: str,
    all_methods: dict[str, np.ndarray],
    residual: dict,
    num_samples: int,
) -> Path:
    """Write a full ``.npz`` extrinsics record next to the ``.npy`` (audit/compare)."""
    out = _resolve(path).with_suffix(".calib.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_T_camera": base_T_camera.astype(np.float64),
        "method": method,
        "num_samples": int(num_samples),
        "trans_residual_m": float(residual["trans_m"]["mean"]),
        "rot_residual_deg": float(residual["rot_deg"]["mean"]),
    }
    for name, X in all_methods.items():
        payload[f"X_{name}"] = X.astype(np.float64)
    np.savez(out, **payload)
    return out
