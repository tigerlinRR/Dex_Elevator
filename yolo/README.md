# yolo — elevator button detection

Detects **elevator buttons** in an RGB frame and returns each as a
`Detection` (bbox, optional mask, confidence). The button's center pixel feeds
the press geometry (`core/press.py`), which turns it into a 3D contact point on
the panel via the eye-to-hand extrinsic.

## Files
- `button_detector.py`
  - `ButtonDetector` — Ultralytics YOLO wrapper; `detect(rgb) -> list[Detection]`
    (single class `button`). A `-seg` model gives a mask (tighter center); a plain
    detector gives only the bbox, which is fine for a point press.
  - `read_floor_label(rgb, detection) -> str` — **STUB**. Turn a button crop into a
    floor string ("3", "B1", "G", "▲"). This is the open identification problem
    (see below).
  - `centroid_pixel(detection)` — mask-centroid (or bbox-center) pixel, generic.

## Localization vs identification (kept separate)
- **Localization** (where the button is in 3D) is pure geometry + calibration —
  robust and done. YOLO only needs to give a reliable pixel.
- **Identification** (which floor a button is) is the hard part. `read_floor_label`
  is a stub; options to evaluate on the captured dataset:
  1. a small OCR (PaddleOCR/EasyOCR) fine-tuned on button labels,
  2. a multi-class YOLO where each floor label is its own class,
  3. template matching for one fixed demo panel (simplest).
  Because the two are decoupled, swapping the label reader never touches the
  press pipeline.

## Train the button detector
1. Collect + label button images (the DEX robot already captures raw frames under
   `~/dataset` on the Jetson). Single class `button` (bbox; segmentation optional).
2. Train with Ultralytics, e.g. `yolo detect train data=buttons.yaml model=yolo11n.pt`.
3. Point `configs/pipeline.yaml:button_yolo.weights` at the resulting `.pt`.

## Contract
- **Input:** RGB image `(H, W, 3)`.
- **Output:** `list[Detection]` sorted by confidence (bbox in pixels, optional mask).
