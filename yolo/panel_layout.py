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
              margin: float = 0.35, link: float = 2.2,
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

    ``link`` was 3.0 and that was too loose. The cabinet's keyhole sits about 2.25
    button widths from the nearest button, so it was merged on some frames and dropped
    on others — and when merged the ROI stretched 100 px further left, the refined pass
    returned 10-11 "buttons" including the keyhole and the warning label, and the
    lattice fit failed outright. Neighbouring buttons are only ~1.6 widths apart, so
    2.2 still links the grid while leaving the keyhole out.
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
    inferred: list[str] = field(default_factory=list)  # cells filled from the lattice
    residual: float = 0.0                       # worst lattice fit error, px

    def lines(self) -> list[str]:
        out = [f"detected {self.detected} buttons"]
        if self.residual:
            out[0] += f", lattice residual {self.residual:.1f} px"
        if self.inferred:
            out.append(f"inferred {len(self.inferred)} missed cell(s) from the lattice: "
                       + ", ".join(self.inferred))
        if self.anchors_total:
            out.append(f"anchors {self.anchors_ok}/{self.anchors_total} agree")
            out.extend(f"    {d}" for d in self.anchor_detail)
        if not self.ok:
            out.append(f"REFUSED: {self.reason}")
        return out


def fit_lattice(found: Sequence[Found], rows: int, cols: int,
                iters: int = 6) -> Optional[dict]:
    """Fit the buttons' regular lattice, so MISSED buttons can be filled in.

    Elevator faceplates are laid out on a grid, and ours is regular to about a pixel:
    row pitch 43.4 mm / 68 px and column pitch 56.6 mm / 79 px, consistent across all
    five rows. Requiring every cell to be detected therefore throws away good frames
    for no reason — after the base re-docked 12 cm further out the buttons shrank from
    50 to 44 px and the strict shape check failed on 6 of 6 attempts, with grids like
    (2,1,2,2,3), while all ten buttons were plainly visible.

    Model: ``pixel(i, j) = origin + i * row_vec + j * col_vec``, fitted by least
    squares with the cell assignment re-estimated each iteration. Affine rather than a
    homography: over a faceplate this small the difference stays inside the residual
    budget, and the residual is checked, so a genuinely skewed view is rejected rather
    than silently extrapolated.

    Fixing ``rows``/``cols`` from the registered layout is what makes this safe. If a
    whole row were missed, the fit would have to spread ``rows`` rows across one row
    less of real buttons, and the residual blows up — so the same check that fills
    gaps also catches the shift that would press the wrong floor.

    Returns ``{origin, row_vec, col_vec, cells, residual, assigned}`` or ``None``.
    """
    if len(found) < 4:
        return None
    P = np.array([[f.u, f.v] for f in found], dtype=np.float64)
    u, v = P[:, 0], P[:, 1]
    # Initial guess: the detections span the panel, so the extremes bracket the grid.
    r_pitch = (v.max() - v.min()) / max(rows - 1, 1)
    c_pitch = (u.max() - u.min()) / max(cols - 1, 1)
    if r_pitch <= 1 or c_pitch <= 1:
        return None
    o = np.array([u.min(), v.min()])
    rv = np.array([0.0, r_pitch])
    cv = np.array([c_pitch, 0.0])

    idx = None
    for _ in range(iters):
        # assign each detection to its nearest cell under the current lattice
        M = np.column_stack([rv, cv])                       # 2x2: [row_vec col_vec]
        try:
            ij = np.linalg.lstsq(M, (P - o).T, rcond=None)[0].T
        except np.linalg.LinAlgError:
            return None
        ij = np.rint(ij).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, rows - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, cols - 1)
        if idx is not None and np.array_equal(ij, idx):
            break
        idx = ij
        # least squares for (origin, row_vec, col_vec) given the assignment
        A = np.column_stack([np.ones(len(P)), ij[:, 0], ij[:, 1]])
        sol, *_ = np.linalg.lstsq(A, P, rcond=None)
        o, rv, cv = sol[0], sol[1], sol[2]

    # Two detections landing on one cell means the assignment is not trustworthy.
    keys = [tuple(k) for k in idx]
    if len(set(keys)) != len(keys):
        return None
    pred = o + idx[:, 0:1] * rv + idx[:, 1:2] * cv
    residual = float(np.sqrt(((P - pred) ** 2).sum(1)).max())
    cells = {(i, j): tuple(o + i * rv + j * cv)
             for i in range(rows) for j in range(cols)}
    return {"origin": o, "row_vec": rv, "col_vec": cv, "cells": cells,
            "residual": residual, "assigned": {k: n for n, k in enumerate(keys)}}


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
           layout: PanelLayout, min_conf: float = 0.25, verify: bool = True,
           max_residual: float = 6.0,
           min_detected_frac: float = 0.7) -> tuple[dict[str, tuple[float, float]], Report]:
    """Map each registered label to a pixel, verifying the geometry first.

    Returns ``({label: (u, v)}, report)``. On failure the mapping is EMPTY and
    ``report.ok`` is False — callers must refuse to press rather than fall back to a
    guess, because the failure being guarded against (a shifted grid) still produces
    perfectly plausible coordinates.

    Verification is GEOMETRIC, by fitting the button lattice. That replaced an exact
    row-shape match, which was rejecting good frames: at a re-docked distance 12 cm
    further out the detector drops a button or two per frame, and the shape check
    failed 6 times out of 6 while all ten buttons were clearly visible. The lattice
    both fills those gaps and catches a real shift, since the row and column counts
    come from the registered layout and a missing row cannot be fitted without the
    residual blowing up.

    Anchors are kept, but only as a check against a SHIFT rather than as a confidence
    test on the classifier. They were an absolute test ("2 of 4 must read correctly"),
    and that refused a correct grid outright once the buttons shrank from 50 to 44 px —
    a false refusal that leaves the robot stuck. Now the unshifted alignment merely has
    to beat every shifted alternative.
    """
    found = detect_positions(bgr, roi, detector, min_conf=min_conf)
    rows, cols = layout.rows, max(len(r) for r in layout.grid)
    n_cells = sum(len(r) for r in layout.grid)
    rep = Report(ok=False, detected=len(found))
    rep.found = found

    if len(found) < min_detected_frac * n_cells:
        rep.reason = (f"only {len(found)} of {n_cells} buttons detected "
                      f"(need {min_detected_frac:.0%}); the panel is not clearly enough "
                      "in view to infer the rest")
        return {}, rep

    # If the fit fails outright there is usually one non-button in the set (the
    # keyhole, a label). Drop the detection furthest from the group and retry rather
    # than discarding an otherwise good frame — tightening `link` reduced this but the
    # geometry is close enough to the threshold that it will still happen sometimes.
    lat, used = fit_lattice(found, rows, cols), list(found)
    dropped_outliers: list[str] = []
    while lat is None and len(used) > n_cells:
        P = np.array([[f.u, f.v] for f in used])
        med = np.median(P, axis=0)
        far = int(np.argmax(np.linalg.norm(P - med, axis=1)))
        dropped_outliers.append(f"{used[far].label}@({used[far].u:.0f},"
                                f"{used[far].v:.0f})")
        used.pop(far)
        lat = fit_lattice(used, rows, cols)
    if dropped_outliers:
        rep.inferred.append("(dropped as non-button: "
                            + ", ".join(dropped_outliers) + ")")
    found = used
    rep.found = found
    if lat is None:
        rep.reason = ("could not fit a lattice to the detections even after dropping "
                      f"{len(dropped_outliers)} outlier(s)")
        return {}, rep
    rep.residual = lat["residual"]
    if lat["residual"] > max_residual:
        rep.reason = (f"lattice residual {lat['residual']:.1f} px exceeds "
                      f"{max_residual} px — the detections do not form the registered "
                      f"{rows}x{cols} grid, so a row may be missing or this is a "
                      "different panel")
        return {}, rep

    # Cell -> detection, and the cells that have to be inferred.
    at = lat["assigned"]
    mapping: dict[str, tuple[float, float]] = {}
    detected_cell: dict[tuple[int, int], Found] = {}
    for i, row in enumerate(layout.grid):
        for j, cell in enumerate(row):
            k = (i, j)
            if k in at:
                f = found[at[k]]
                mapping[cell.label] = (f.u, f.v)
                detected_cell[k] = f
            else:
                mapping[cell.label] = lat["cells"][k]
                rep.inferred.append(cell.label)

    # Anchors: does the unshifted alignment explain the classes better than a shift?
    # Only REAL detections may vote — scoring inferred cells would be circular.
    anchors = layout.anchors
    rep.anchors_total = len(anchors)
    if verify and anchors:
        solo: dict[tuple[int, int], tuple[str, float]] = {}

        def score(shift: int) -> int:
            hits = 0
            for i, j, c in anchors:
                k = ((i + shift) % rows, j)
                f = detected_cell.get(k)
                if f is None:
                    continue
                if k not in solo:
                    solo[k] = classify_solo(bgr, f, detector)
                got, conf = solo[k]
                hits += got == c.expect and conf >= layout.min_conf
            return hits

        base = score(0)
        rep.anchors_ok = base
        for i, j, c in anchors:
            f = detected_cell.get((i, j))
            got, conf = solo.get((i, j), ("", 0.0))
            state = ("inferred cell" if f is None else
                     ("ok" if (got == c.expect and conf >= layout.min_conf)
                      else "mismatch"))
            rep.anchor_detail.append(
                f"{c.label:<6} expect {c.expect:<6} got {(got or '-'):<8} "
                f"{conf:.2f}  {state}")
        others = [score(sh) for sh in range(1, rows)]
        if others and base <= max(others):
            rep.reason = (f"a shifted alignment explains the anchors at least as well "
                          f"({base} vs {max(others)}); refusing rather than risk "
                          "pressing the wrong floor")
            return {}, rep
        if base == 0 and not any(others):
            rep.anchor_detail.append(
                "no anchor read confidently either way — accepted on the lattice fit "
                f"alone (residual {lat['residual']:.1f} px)")

    rep.ok = True
    return mapping, rep
