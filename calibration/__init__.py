"""Camera calibration — done once per camera, shared by the whole runtime.

Two parts:
  * Part 1 — intrinsics  (:mod:`calibration.intrinsic`): K + distortion.
  * Part 2 — extrinsics  (:mod:`calibration.extrinsic`): eye-to-hand base_T_camera,
    which consumes Part 1's intrinsics. Use this for a camera fixed to the torso.
  * Part 2b — extrinsics (:mod:`calibration.eye_in_hand`): eye-IN-hand
    gripper_T_camera, for a camera that rides the arm. Same Part 1, same board,
    different geometry; :mod:`calibration.frames` tags which of the two a saved
    ``.npy`` holds so the runtime cannot confuse them.

Both write to ``data/calibration/`` and are loaded once by
:class:`~core.camera.manager.CameraManager` and reused everywhere.
"""

from calibration.board import BoardDetector, CharucoBoard
from calibration.capture_ui import display_available, live_capture_loop

# Part 1 — intrinsics
from calibration.intrinsic import (
    IntrinsicCalibration,
    calibrate_intrinsics_from_images,
    intrinsics_from_camera,
)

# Part 2 — extrinsics (eye-to-hand)
from calibration.extrinsic import (
    CalibrationSession,
    ExtrinsicCalibrationSession,
    consistency_residual,
    reprojection_residual,
    select_best,
    solve_eye_to_hand,
    solve_eye_to_hand_all_methods,
)

# Part 2b — extrinsics (eye-IN-hand, arm-mounted camera)
from calibration.eye_in_hand import (
    EyeInHandSession,
    EyeInHandThresholds,
    base_T_camera,
    rotation_diversity,
    save_eye_in_hand,
    solve_eye_in_hand,
    solve_eye_in_hand_all_methods,
    validate_eye_in_hand,
)
from calibration.frames import (
    FRAME_BASE,
    FRAME_GRIPPER,
    check_frame,
    load_frame_tag,
    save_frame_tag,
)

# Shared persistence
from calibration.io import (
    CALIB_DIR,
    load_camera_intrinsics,
    load_extrinsic,
    load_intrinsics,
    save_calibration,
    save_extrinsic,
    save_intrinsics,
)

# Validation (run after calibration)
from calibration.validate import (
    ExtrinsicThresholds,
    IntrinsicThresholds,
    ValidationReport,
    save_report,
    validate_extrinsics,
    validate_intrinsics,
)

__all__ = [
    # shared
    "BoardDetector",
    "CharucoBoard",
    "live_capture_loop",
    "display_available",
    "CALIB_DIR",
    # Part 1 — intrinsics
    "IntrinsicCalibration",
    "intrinsics_from_camera",
    "calibrate_intrinsics_from_images",
    "save_intrinsics",
    "load_intrinsics",
    "load_camera_intrinsics",
    # Part 2 — extrinsics
    "ExtrinsicCalibrationSession",
    "CalibrationSession",
    "solve_eye_to_hand",
    "solve_eye_to_hand_all_methods",
    "select_best",
    "consistency_residual",
    "reprojection_residual",
    "save_extrinsic",
    "load_extrinsic",
    "save_calibration",
    # Part 2b — extrinsics (eye-in-hand)
    "EyeInHandSession",
    "EyeInHandThresholds",
    "solve_eye_in_hand",
    "solve_eye_in_hand_all_methods",
    "validate_eye_in_hand",
    "rotation_diversity",
    "base_T_camera",
    "save_eye_in_hand",
    "FRAME_BASE",
    "FRAME_GRIPPER",
    "check_frame",
    "load_frame_tag",
    "save_frame_tag",
    # validation
    "validate_intrinsics",
    "validate_extrinsics",
    "save_report",
    "ValidationReport",
    "IntrinsicThresholds",
    "ExtrinsicThresholds",
]
