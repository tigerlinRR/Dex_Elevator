"""YOLO elevator-button detector + floor-label reader (SKELETON).

Two responsibilities, deliberately split so localization is decoupled from
identification (swapping the reader never touches the geometry pipeline):

  * :class:`ButtonDetector` — a YOLO detector that finds button *boxes*
    (single class ``button``) and returns a :class:`~core.types.Detection` list.
    Its center pixel feeds :func:`core.press.press_target_from_pixel`.
  * :func:`read_floor_label` — turn a button crop into a floor string ("3",
    "B1", "G", "▲"…). **Stub for now** — this is the real open problem: no
    off-the-shelf model reads engraved/low-contrast elevator labels well, so
    expect a label+train loop (custom OCR or a multi-class YOLO) on the dataset
    already being captured at ``~/dataset`` on the robot.

Nothing here decides the panel plane or which floor was requested — see
:mod:`core.press` and :mod:`core.elevator_pipeline`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.types import Detection

try:
    from ultralytics import YOLO  # type: ignore

    _YOLO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _YOLO_AVAILABLE = False


def _to_bgr(rgb: np.ndarray) -> np.ndarray:
    """RGB -> contiguous BGR for Ultralytics (trained on cv2/BGR images)."""
    return np.ascontiguousarray(rgb[..., ::-1])


def centroid_pixel(detection: Detection) -> np.ndarray:
    """Centroid pixel ``(u, v)`` of a detection.

    Uses the segmentation mask centroid when a mask is present, otherwise the
    bbox center. Returns float pixels. Generic (numpy only) — feeds the button's
    press target in :mod:`core.press`.
    """
    mask = detection.mask
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)
    return detection.center


class ButtonDetector:
    """Locate elevator buttons in an RGB frame. Returns detections in pixel coords.

    ``weights`` should be a YOLO model trained on the button dataset. Until one
    exists, this class still constructs (given any .pt) — training the button
    model is the next data task. A ``-seg`` model gives a mask (tighter center);
    a plain detector gives only the bbox, which is fine for a point press.
    """

    def __init__(self, weights: str | Path, conf: float = 0.25, device: str = "cuda",
                 imgsz: int | None = None):
        if not _YOLO_AVAILABLE:
            raise ImportError("ultralytics is not installed: pip install ultralytics")
        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = device
        self.imgsz = imgsz

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        """Detect buttons; returns :class:`Detection` list sorted by confidence."""
        kw = {"imgsz": self.imgsz} if self.imgsz else {}
        results = self.model.predict(
            _to_bgr(rgb), conf=self.conf, device=self.device, verbose=False, **kw,
        )
        dets: list[Detection] = []
        for r in results:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0])
                dets.append(Detection(
                    label=names.get(cls_id, str(cls_id)),
                    confidence=float(box.conf[0]),
                    bbox_xyxy=box.xyxy[0].cpu().numpy().astype(np.float64),
                    class_id=cls_id,
                ))
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets


def read_floor_label(rgb: np.ndarray, detection: Detection) -> str:
    """Read the floor label from a button crop. STUB — returns "".

    TODO (the identification problem): crop ``rgb`` to ``detection.bbox_xyxy`` and
    classify the digit/symbol. Options to evaluate on the captured dataset:
      * a small OCR (PaddleOCR/EasyOCR) fine-tuned on button labels;
      * a multi-class YOLO where each floor label is its own class;
      * template matching for a fixed, known panel (simplest for one demo cell).
    Returning "" means "unknown"; the pipeline treats unlabeled buttons as
    non-targets until this is implemented.
    """
    return ""
