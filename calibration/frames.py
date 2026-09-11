"""Which frame an extrinsic ``.npy`` is expressed in — fixed camera vs arm-mounted.

Two camera mounts now coexist on this robot, and they store **different**
transforms in a file of the same shape:

* ``mount: fixed`` (the chest 335) — ``<cam>.npy`` holds ``base_T_camera``, a
  constant. The camera and the arm base ride the torso lift together, so it never
  changes. Solved by :mod:`calibration.extrinsic` (eye-to-hand).
* ``mount: arm`` (a camera bolted to the arm) — ``<cam>.npy`` holds
  ``gripper_T_camera``. ``base_T_camera`` exists only together with a live arm
  pose: ``base_T_camera = base_T_gripper @ gripper_T_camera``. Solved by
  :mod:`calibration.eye_in_hand`.

Both are a bare 4x4 float array, so **nothing in the file itself says which one
it is** — and using one as the other produces coordinates that look entirely
plausible and are wrong by the whole length of the arm. That is the silent
failure mode this project keeps paying for, so the frame is recorded in a
sidecar ``<stem>.frame`` next to the ``.npy`` and checked at load time.

Legacy files have no sidecar and are read as ``base``, which is what every
eye-to-hand calibration already on disk is — existing setups keep working with
nothing to migrate.

Depends only on ``pathlib`` and :mod:`calibration.io` (for ``CALIB_DIR``), so the
runtime can import it without pulling in ``cv2``.
"""

from __future__ import annotations

from pathlib import Path

from calibration.io import CALIB_DIR

# The two frames an extrinsic file can be expressed in.
FRAME_BASE = "base"        # base_T_camera      — camera fixed in the workspace
FRAME_GRIPPER = "gripper"  # gripper_T_camera   — camera rides the arm

# cameras.yaml `mount:` -> the frame its extrinsic file must be in.
MOUNT_FRAME = {"fixed": FRAME_BASE, "arm": FRAME_GRIPPER}


def resolve(path: str | Path) -> Path:
    """Resolve a calibration path: absolute as-is, otherwise under ``data/calibration/``."""
    path = Path(path)
    return path if path.is_absolute() else CALIB_DIR / path


_resolve = resolve   # internal alias


def frame_tag_path(extrinsic_path: str | Path) -> Path:
    """Sidecar path for an extrinsic ``.npy`` (``cam_arm.npy`` -> ``cam_arm.frame``)."""
    return _resolve(extrinsic_path).with_suffix(".frame")


def save_frame_tag(extrinsic_path: str | Path, frame: str) -> Path:
    """Record which frame an extrinsic file is in. Call this whenever one is saved."""
    if frame not in (FRAME_BASE, FRAME_GRIPPER):
        raise ValueError(f"frame must be {FRAME_BASE!r} or {FRAME_GRIPPER!r}, got {frame!r}")
    out = frame_tag_path(extrinsic_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(frame + "\n")
    return out


def load_frame_tag(extrinsic_path: str | Path) -> str:
    """Frame of an extrinsic file. Missing sidecar => ``base`` (pre-sidecar eye-to-hand)."""
    tag = frame_tag_path(extrinsic_path)
    if not tag.exists():
        return FRAME_BASE
    frame = tag.read_text().strip()
    if frame not in (FRAME_BASE, FRAME_GRIPPER):
        raise ValueError(f"{tag} contains {frame!r}; expected {FRAME_BASE!r} or {FRAME_GRIPPER!r}")
    return frame


def expected_frame(mount: str) -> str:
    """Frame required by a ``cameras.yaml`` ``mount:`` value."""
    try:
        return MOUNT_FRAME[mount]
    except KeyError:
        raise ValueError(f"unknown mount {mount!r}; expected one of {sorted(MOUNT_FRAME)}") from None


def check_frame(extrinsic_path: str | Path, mount: str) -> str:
    """Raise unless the file's frame matches the mount. Returns the frame.

    Fails CLOSED on purpose: a mismatch means the runtime is about to treat
    ``gripper_T_camera`` as ``base_T_camera`` (or vice versa), which yields
    confident, wrong 3D points rather than an error.
    """
    want = expected_frame(mount)
    got = load_frame_tag(extrinsic_path)
    if got != want:
        raise ValueError(
            f"{_resolve(extrinsic_path)} is a {got!r}-frame extrinsic but the camera is "
            f"configured mount: {mount!r}, which needs {want!r}. Re-calibrate with "
            f"{'initialization/run_calibration_arm_cam.py' if want == FRAME_GRIPPER else 'initialization/run_calibration.py'}, "
            f"or fix `mount:` in configs/cameras.yaml."
        )
    return got
