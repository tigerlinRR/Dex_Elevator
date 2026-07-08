"""ChArUco board detection — shared by both calibration parts.

* Part 1 (intrinsics): :meth:`BoardDetector.detect_corners` gives the per-image
  (object points, image points) that ``cv2.calibrateCamera`` needs.
* Part 2 (extrinsics): :meth:`BoardDetector.detect` gives ``cam_T_target`` (the
  board pose in the camera frame) via ``solvePnP``.

Uses OpenCV's modern ``cv2.aruco.CharucoDetector`` API (auto dict lock-in,
subpixel refinement). Self-contained: depends only on ``cv2``, ``numpy`` and
:mod:`core.transforms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.transforms import make_transform

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - cv2 is a hard runtime dep for calibration
    cv2 = None


# Dictionaries tried (in order) when auto-detecting which ChArUco dictionary a
# board uses. Largest-first within each family: OpenCV's predefined dicts are
# nested (DICT_5X5_50 is a prefix of DICT_5X5_100 ... of DICT_5X5_1000), so
# trying the largest first recognizes ALL of a smaller board's markers, whereas
# trying a smaller dict first would silently drop the board's higher-id markers.
# (The scripts pass --dictionary explicitly; this is only the fallback path.)
_COMMON_DICTS = (
    "DICT_5X5_1000", "DICT_5X5_250", "DICT_5X5_100", "DICT_5X5_50",
    "DICT_4X4_1000", "DICT_4X4_250", "DICT_4X4_100", "DICT_4X4_50",
)


@dataclass
class CharucoBoard:
    """A ChArUco calibration board.

    Defaults match the calib.io board used here: a 14x9 grid, 20 mm squares,
    15 mm markers, ``DICT_5X5_100`` (the smallest 5x5 dict that fits the board's
    ~63 markers). **Measure your own** ``square_length`` / ``marker_length``
    (meters) — getting them wrong scales the whole calibration.

    ``dictionary`` is the integer ``cv2.aruco.DICT_*`` id, or ``None`` to
    auto-detect among :data:`_COMMON_DICTS` on the first good frame. The scripts
    set it explicitly (``--dictionary``, default ``DICT_5X5_100``).
    """

    squares_x: int = 14           # columns (X across)
    squares_y: int = 9            # rows (Y down)
    square_length: float = 0.020  # meters — side of a chessboard square
    marker_length: float = 0.015  # meters — side of an ArUco marker
    dictionary: Optional[int] = None

    def opencv_board(self, dictionary_id: int):
        """Build the ``(cv2.aruco.CharucoBoard, dictionary)`` for a given dict id."""
        if cv2 is None:
            raise ImportError("opencv-contrib-python (cv2.aruco) is required.")
        d = cv2.aruco.getPredefinedDictionary(dictionary_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length, self.marker_length, d,
        )
        return board, d


def _detector_params(accurate: bool):
    """Tuned ``DetectorParameters`` (accurate -> subpixel + wide threshold sweep)."""
    p = cv2.aruco.DetectorParameters()
    if accurate:
        p.adaptiveThreshWinSizeMin = 5
        p.adaptiveThreshWinSizeMax = 75
        p.adaptiveThreshWinSizeStep = 4
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        p.cornerRefinementWinSize = 20
        p.cornerRefinementMaxIterations = 100
        p.cornerRefinementMinAccuracy = 0.01
        p.detectInvertedMarker = True
        p.minMarkerPerimeterRate = 0.005
        p.maxMarkerPerimeterRate = 6.0
        p.adaptiveThreshConstant = 3
        p.minCornerDistanceRate = 0.003
        # NOTE: useAruco3Detection stays OFF. Aruco3 downsamples the image for the
        # initial marker pass, which silently drops most markers once the board is
        # small/tilted in frame (observed: 7/63 markers vs 63/63 with it off) — it
        # only happened to work during capture because the board filled the frame.
        # It's a *speed* optimization; the accuracy here comes from the subpixel
        # corner refinement below, which is independent of it.
        p.useAruco3Detection = False
    else:
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 4
        p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        p.detectInvertedMarker = False
        p.useAruco3Detection = False
    return p


@dataclass
class BoardDetector:
    """Detects a :class:`CharucoBoard`.

    Stateful only to *lock in* the auto-detected dictionary after the first good
    frame; detection itself is independent per frame.
    """

    board: CharucoBoard = field(default_factory=CharucoBoard)
    accurate: bool = True
    min_corners: int = 10         # min ChArUco corners required to trust a result
    image_is_rgb: bool = True     # CameraFrame.rgb is RGB; set False for BGR

    # internal lock-in state
    _locked_dict: Optional[int] = field(default=None, init=False)
    _candidates: dict = field(default_factory=dict, init=False)
    last_markers: int = field(default=0, init=False)
    last_corners: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if cv2 is None:
            raise ImportError("opencv-contrib-python (cv2.aruco) is required.")
        if self.board.dictionary is not None:
            self._locked_dict = self.board.dictionary

    def _detector_for(self, dictionary_id: int):
        """Cache and return a ``(CharucoDetector, opencv_board)`` for a dict id."""
        if dictionary_id not in self._candidates:
            opencv_board, _ = self.board.opencv_board(dictionary_id)
            detector = cv2.aruco.CharucoDetector(opencv_board)
            detector.setDetectorParameters(_detector_params(self.accurate))
            self._candidates[dictionary_id] = (detector, opencv_board)
        return self._candidates[dictionary_id]

    def _dict_ids_to_try(self):
        if self._locked_dict is not None:
            return [self._locked_dict]
        return [getattr(cv2.aruco, name) for name in _COMMON_DICTS]

    def detect_corners(self, image: np.ndarray):
        """Return ``(charuco_corners, charuco_ids, opencv_board)`` or ``None``.

        Corners are subpixel-refined (when ``accurate``). Used by Part 1
        (intrinsics) to accumulate per-image points for ``cv2.calibrateCamera``.
        """
        for dict_id in self._dict_ids_to_try():
            detector, opencv_board = self._detector_for(dict_id)
            charuco_corners, charuco_ids, _mc, marker_ids = detector.detectBoard(image)

            self.last_markers = 0 if marker_ids is None else int(len(marker_ids))
            self.last_corners = 0 if charuco_ids is None else int(len(charuco_ids))

            if charuco_ids is None or len(charuco_ids) < self.min_corners:
                continue

            self._locked_dict = dict_id
            return self._refine(image, charuco_corners), charuco_ids, opencv_board
        return None

    def detect(
        self,
        image: np.ndarray,
        K: np.ndarray,
        dist: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Return ``cam_T_target`` (4x4) for the detected board, or ``None``.

        Args:
            image: H×W×3 color image (RGB by default; see ``image_is_rgb``).
            K:     3x3 camera matrix.
            dist:  distortion coefficients (``None`` -> zeros).
        """
        found = self.detect_corners(image)
        if found is None:
            return None
        corners, charuco_ids, opencv_board = found

        dist = np.zeros(5, dtype=np.float64) if dist is None else np.asarray(dist, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)

        # solvePnP against the board's object points for the detected ids.
        obj_points = opencv_board.getChessboardCorners()[charuco_ids.flatten()]
        ok, rvec, tvec = cv2.solvePnP(obj_points, corners, K, dist)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        return make_transform(R, tvec.reshape(3))

    def _refine(self, image: np.ndarray, charuco_corners: np.ndarray) -> np.ndarray:
        """Subpixel-refine ChArUco corners (no-op when accuracy is off)."""
        if not self.accurate or charuco_corners is None or len(charuco_corners) == 0:
            return charuco_corners
        code = cv2.COLOR_RGB2GRAY if self.image_is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(image, code)
        corners = np.asarray(charuco_corners, dtype=np.float32)
        cv2.cornerSubPix(
            gray, corners, (5, 5), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
        )
        return corners
