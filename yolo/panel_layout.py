"""Turn button detections into a verified label -> pixel map.

Replaces the OpenCV Hough-circle stand-in (`yolo/button_circles.py`) with the
detector, and replaces the hard-coded label grid in `press_buttons.py` with a
registered layout from ``configs/panels.yaml``.

Two measurements shaped this design, both on our panel:

* **Positions are reliable, labels are not.** The detector finds 10/10 buttons on
  the panel crop at imgsz 640 (centre error 0.3-6.3 px, matching Hough), but only
  reads 5 of 8 markings correctly — the etched digits are barely in the image. So
  positions come from the detector and labels come from the layout.
* **A whole-panel pass classifies much worse than per-button crops.** On the crop,
  `3` comes back as `empty` (0.46); cropped alone it comes back as `3` (0.48), and
  `2`/`open`/`close` reach 0.90+. So detection runs once on the panel, and each
  detected button is then classified from its own crop.

The anchors are the safety mechanism. If a row is missed, every label below it
shifts by one and the robot presses the wrong floor with no error anywhere — a
silent failure. Anchor cells (`expect:` in the layout) are the buttons the model
reads confidently; a shifted grid breaks all of them at once, so the press is
refused instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np
import yaml

from core.config import REPO_ROOT

PANELS_FILE = REPO_ROOT / "configs" / "panels.yaml"


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    """One position on the faceplate."""

    label: str                      # what a caller asks for ("3", "open")
    expect: Optional[str] = None    # model class used to verify alignment (anchor)


@dataclass
class PanelLayout:
    id: str
    grid: list[list[Cell]]          # rows top->bottom, each row left->right
    min_anchors: int = 2
    min_conf: float = 0.40
    description: str = ""

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(r) for r in self.grid)

    @property
    def labels(self) -> list[str]:
        return [c.label for row in self.grid for c in row]

    @property
    def anchors(self) -> list[tuple[int, int, Cell]]:
        return [(i, j, c) for i, row in enumerate(self.grid)
                for j, c in enumerate(row) if c.expect]


def load_panel(panel_id: str, path=PANELS_FILE) -> PanelLayout:
    """Load one registered layout by id."""
    cfg = yaml.safe_load(open(path).read())
    for entry in cfg.get("panels", []):
        if entry.get("id") != panel_id:
            continue
        grid = [[Cell(label=str(c["label"]), expect=c.get("expect")) for c in row]
                for row in entry["grid"]]
        v = entry.get("verify", {}) or {}
        return PanelLayout(id=panel_id, grid=grid,
                           min_anchors=int(v.get("min_anchors", 2)),
                           min_conf=float(v.get("min_conf", 0.40)),
                           description=(entry.get("description") or "").strip())
    known = [e.get("id") for e in cfg.get("panels", [])]
    raise KeyError(f"panel {panel_id!r} is not registered in {path}; have {known}")


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
class Detector(Protocol):
    """Anything with ``detect(bgr) -> list[Detection]`` (see yolo/trt_detector.py)."""

    def detect(self, bgr: np.ndarray) -> Sequence: ...


@dataclass
class Found:
    """One detected button, in ORIGINAL frame pixels."""

    u: float
    v: float
    w: float
    h: float
    label: str          # class from the whole-panel pass (often wrong — see module doc)
    conf: float
    solo_label: str = ""    # class from this button's own crop (the accurate one)
    solo_conf: float = 0.0


def detect_positions(bgr: np.ndarray, roi: tuple[int, int, int, int],
                     detector: Detector, min_conf: float = 0.25) -> list[Found]:
    """Detect buttons inside ``roi``. Returns centres in FULL-frame pixels.

    The ROI matters for accuracy, not just speed: run on the whole 1280x720 frame
    the buttons are 3.9 % of the image width and one is missed at conf 0.25, while
    on the faceplate crop all ten come back with a single spurious box.
    """
    x0, y0, x1, y1 = roi
    crop = bgr[y0:y1, x0:x1]
    out: list[Found] = []
    for d in detector.detect(crop):
        if float(d.confidence) < min_conf:
            continue
        bx1, by1, bx2, by2 = [float(v) for v in d.bbox_xyxy]
        out.append(Found(u=(bx1 + bx2) / 2 + x0, v=(by1 + by2) / 2 + y0,
                         w=bx2 - bx1, h=by2 - by1,
                         label=str(d.label), conf=float(d.confidence)))
    return out


def panel_roi(bgr: np.ndarray, detector: Detector, min_conf: float = 0.10,
              margin: float = 0.35, link: float = 3.0,
              verbose: bool = False) -> Optional[tuple[int, int, int, int]]:
    """ROI covering the faceplate, derived from the detections themselves.

    Deriving it beats a hard-coded box for two reasons: the panel lands somewhere
    different every time the base docks, and anything else inside a fixed ROI drags
    the depth plane fit — the robot's own hand cost 23 mm of error that way.

    The union of ALL detections is too crude, though: a low-confidence full-frame
    pass also fires on the cabinet's keyhole and its warning label, which stretched
    the ROI to 330x499 px and made the refined pass report 12 buttons in a (3,2,3,2,2)
    grid. Buttons on a faceplate form one dense cluster, while those extras sit alone,
    so single-linkage clustering and keeping the largest cluster separates them.
    """
    dets = [d for d in detector.detect(bgr) if float(d.confidence) >= min_conf]
    if not dets:
        return None
    xy = np.array([d.bbox_xyxy for d in dets], dtype=np.float64)
    cx, cy = (xy[:, 0] + xy[:, 2]) / 2, (xy[:, 1] + xy[:, 3]) / 2
    w = float(np.median(xy[:, 2] - xy[:, 0]))

    # single-linkage: connect centres closer than `link` button widths
    n = len(dets)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    thr = link * w
    for i in range(n):
        for j in range(i + 1, n):
            if (cx[i] - cx[j]) ** 2 + (cy[i] - cy[j]) ** 2 <= thr * thr:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    keep = max(groups.values(), key=len)
    if verbose and len(keep) < n:
        print(f"    ROI: kept the {len(keep)}-button cluster, dropped {n - len(keep)} "
              f"isolated detection(s) "
              f"({[dets[i].label for i in range(n) if i not in keep]})")

    sub = xy[keep]
    pad = margin * w
    h, wid = bgr.shape[:2]
    return (max(0, int(sub[:, 0].min() - pad)), max(0, int(sub[:, 1].min() - pad)),
            min(wid, int(sub[:, 2].max() + pad)), min(h, int(sub[:, 3].max() + pad)))


def classify_solo(bgr: np.ndarray, f: Found, detector: Detector,
                  pad_scale: float = 2.1, size: int = 256) -> tuple[str, float]:
    """Re-classify one button from its own crop, which is markedly more accurate.

    Measured on our panel: `3` reads `empty` (0.46) in the whole-panel pass and `3`
    (0.48) alone; `2` goes 0.90 -> 0.93; `open`/`close` reach 0.92.

    ``pad_scale`` is NOT a free parameter — how much context the crop includes
    changes the answer a lot, and tighter is worse. Swept on a live frame with
    ~50 px buttons: 44 px crop 1/10 correct, 68 px 3/10, 104 px 4/10. 2.1 button
    widths reproduces that best case; 1.6 (80 px) dropped every anchor to a
    mismatch, which looked exactly like a scrambled grid.
    """
    import cv2

    pad = int(max(f.w, f.h) * pad_scale / 2)
    u, v = int(round(f.u)), int(round(f.v))
    sub = bgr[max(0, v - pad):v + pad, max(0, u - pad):u + pad]
    if sub.size == 0:
        return "", 0.0
    big = cv2.resize(sub, (size, size), interpolation=cv2.INTER_CUBIC)
    best = ("", 0.0)
    for d in detector.detect(big):
        if float(d.confidence) > best[1]:
            best = (str(d.label), float(d.confidence))
    return best


# --------------------------------------------------------------------------- #
# grid assembly + verification
# --------------------------------------------------------------------------- #
@dataclass
class Report:
    """Why the layout matched, or why it did not. Printed before any motion."""

    ok: bool
    reason: str = ""
    detected: int = 0
    shape: tuple[int, ...] = ()
    anchors_ok: int = 0
    anchors_total: int = 0
    anchor_detail: list[str] = field(default_factory=list)
    found: list = field(default_factory=list)   # the raw detections, for tight_roi()

    def lines(self) -> list[str]:
        out = [f"detected {self.detected} buttons, rows {self.shape}"]
        if self.anchors_total:
            out.append(f"anchors {self.anchors_ok}/{self.anchors_total} agree")
            out.extend(f"    {d}" for d in self.anchor_detail)
        if not self.ok:
            out.append(f"REFUSED: {self.reason}")
        return out


def _rows_from(found: list[Found], layout: PanelLayout) -> Optional[list[list[Found]]]:
    """Cluster detections into rows by v, then order each row left to right.

    Row tolerance comes from the detected box height rather than a constant: the
    same panel fills a different fraction of the frame at every docking distance.
    """
    if not found:
        return None
    tol = 0.6 * float(np.median([f.h for f in found]))
    rows: list[list[Found]] = []
    for f in sorted(found, key=lambda f: f.v):
        if rows and abs(f.v - np.mean([g.v for g in rows[-1]])) <= tol:
            rows[-1].append(f)
        else:
            rows.append([f])
    for r in rows:
        r.sort(key=lambda f: f.u)
    return rows


def tight_roi(found: Sequence[Found], shape: tuple[int, int],
              margin: float = 0.35) -> tuple[int, int, int, int]:
    """ROI around the ACTUAL buttons, for the depth plane fit.

    The coarse ROI from :func:`panel_roi` is deliberately generous, so it can reach
    well past the faceplate (measured: 145 px below it). The wall and the cabinet
    behind are separate, roughly parallel planes, so anything extra in the fit's ROI
    biases the plane. Once the buttons themselves are known, box them instead.
    """
    u = np.array([f.u for f in found]); v = np.array([f.v for f in found])
    w = float(np.median([f.w for f in found])); h = float(np.median([f.h for f in found]))
    pad_u, pad_v = w * (0.5 + margin), h * (0.5 + margin)
    H, W = shape
    return (max(0, int(u.min() - pad_u)), max(0, int(v.min() - pad_v)),
            min(W, int(u.max() + pad_u)), min(H, int(v.max() + pad_v)))


def assign(bgr: np.ndarray, roi: tuple[int, int, int, int], detector: Detector,
           layout: PanelLayout, min_conf: float = 0.25,
           verify: bool = True) -> tuple[dict[str, tuple[float, float]], Report]:
    """Map each registered label to a pixel, verifying the grid first.

    Returns ``({label: (u, v)}, report)``. On failure the mapping is EMPTY and
    ``report.ok`` is False — callers must refuse to press rather than fall back to
    a guess, because the failure mode being guarded against (a shifted grid) still
    produces perfectly plausible coordinates.
    """
    found = detect_positions(bgr, roi, detector, min_conf=min_conf)
    rows = _rows_from(found, layout)
    shape = tuple(len(r) for r in rows) if rows else ()
    rep = Report(ok=False, detected=len(found), shape=shape)
    rep.found = found

    if not rows:
        rep.reason = "no buttons detected in the panel ROI"
        return {}, rep
    if shape != layout.shape:
        rep.reason = (f"grid {shape} does not match the registered layout "
                      f"{layout.shape} for panel {layout.id!r}")
        return {}, rep

    mapping = {layout.grid[i][j].label: (rows[i][j].u, rows[i][j].v)
               for i in range(layout.rows) for j in range(len(layout.grid[i]))}

    anchors = layout.anchors
    rep.anchors_total = len(anchors)
    if verify and anchors:
        for i, j, cell in anchors:
            got, conf = classify_solo(bgr, rows[i][j], detector)
            rows[i][j].solo_label, rows[i][j].solo_conf = got, conf
            good = got == cell.expect and conf >= layout.min_conf
            rep.anchors_ok += good
            rep.anchor_detail.append(
                f"{cell.label:<6} expect {cell.expect:<6} got "
                f"{(got or '-'):<8} {conf:.2f}  {'ok' if good else 'MISMATCH'}")
        if rep.anchors_ok < layout.min_anchors:
            rep.reason = (f"only {rep.anchors_ok} of {len(anchors)} anchors agree "
                          f"(need {layout.min_anchors}); the grid is probably shifted, "
                          "which would press the wrong floor")
            return {}, rep

    rep.ok = True
    return mapping, rep
