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
    # An APPEARANCE anchor: this cell must be darker than `dark` times the median
    # brightness of the cells in the SAME frame. For a cell whose distinguishing feature
    # is not a marking to be read — here the unlit black disc among nine bright metal
    # discs — this verifies alignment without the classifier being involved at all.
    # Relative, never absolute: an absolute threshold inherits every change the room
    # lighting makes, which is exactly what took anchor reads from 6/8 to nothing when
    # the lab lights went off. Measured 2026-09-18 over 12 frames at the registered
    # viewing pose: `dot` 0.23 (range 0.20-0.25) against a next-darkest cell at 0.76
    # (0.70-0.77) and a bright cluster at 0.87-1.18 — a factor of three with no overlap,
    # while the ABSOLUTE brightnesses of those same cells scattered by +-20 grey levels
    # frame to frame. The scatter was entirely common-mode.
    dark: Optional[float] = None


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
        """Cells that can testify about alignment — by class OR by appearance."""
        return [(i, j, c) for i, row in enumerate(self.grid)
                for j, c in enumerate(row) if c.expect or c.dark is not None]


def load_panel(panel_id: str, path=PANELS_FILE) -> PanelLayout:
    """Load one registered layout by id."""
    cfg = yaml.safe_load(open(path).read())
    for entry in cfg.get("panels", []):
        if entry.get("id") != panel_id:
            continue
        grid = [[Cell(label=str(c["label"]), expect=c.get("expect"),
                      dark=(None if c.get("dark") is None else float(c["dark"])))
                 for c in row] for row in entry["grid"]]
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


def _lines_along(vals: np.ndarray, gap: float) -> list[float]:
    """Group 1-D coordinates into lines (rows or columns), returning their centres."""
    order = np.argsort(vals)
    lines, cur = [], [float(vals[order[0]])]
    for i in order[1:]:
        v = float(vals[i])
        if v - cur[-1] > gap:
            lines.append(sum(cur) / len(cur)); cur = [v]
        else:
            cur.append(v)
    lines.append(sum(cur) / len(cur))
    return lines


def panel_roi(bgr: np.ndarray, detector: Detector, min_conf: float = 0.10,
              margin: float = 0.35, link: float = 2.2,
              shape: tuple[int, ...] = (),
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

    ``shape`` — the registered layout's row shape — closes a failure that is
    self-sealing without it. The full-frame pass runs at 1280x720 scaled into the
    model's 640, so a 44 px button becomes ~22 px, and it can miss a whole edge ROW:
    measured on a live frame, `open`/`close` scored 0.16 and **0.06** full-frame while
    the same two buttons scored **0.91 and 0.96** on a crop of that row alone. The ROI
    is derived FROM these detections, so a missed bottom row put the ROI's bottom edge
    above it, the refined pass never looked there, and the lattice fit was handed four
    rows for a five-row layout — residual 11-12.5 px against a 6 px threshold, refused
    on 15 of 15 attempts with the panel plainly in view.

    So when fewer rows (or columns) are found than the layout registers, the ROI is
    grown by the MISSING count times the measured pitch, in both directions along that
    axis. Bounded by what is missing rather than a bigger blanket margin: a complete
    grid expands by nothing, so a frame that works today behaves identically, and the
    growth cannot run away. Sized in pitches because that is the distance a missed row
    actually sits at — on the failing frame it extended the bottom edge from 410 to
    463 px, still 81 px clear of the cabinet keyhole at 544 that `link` exists to
    exclude.
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
    x0, y0 = sub[:, 0].min() - pad, sub[:, 1].min() - pad
    x1, y1 = sub[:, 2].max() + pad, sub[:, 3].max() + pad

    if shape:
        gap = 0.6 * w                       # centres within 0.6 widths are one line
        rows = _lines_along(cy[keep], gap)
        cols = _lines_along(cx[keep], gap)
        want_rows, want_cols = len(shape), max(shape)
        for found_lines, want, lo, hi, axis in (
                (rows, want_rows, "y0", "y1", "row"),
                (cols, want_cols, "x0", "x1", "col")):
            missing = want - len(found_lines)
            if missing <= 0:
                continue
            if len(found_lines) >= 2:
                pitch = float(np.median(np.diff(found_lines)))
            else:
                # One line found: no pitch to measure. 1.3 button widths is this
                # panel's row pitch (53 px against 44 px buttons) and is only ever
                # used in a case that is already close to hopeless.
                pitch = 1.3 * w
            grow = missing * pitch
            if axis == "row":
                y0 -= grow; y1 += grow
            else:
                x0 -= grow; x1 += grow
            if verbose:
                print(f"    ROI: only {len(found_lines)} of {want} {axis}s found — "
                      f"growing +-{grow:.0f}px (pitch {pitch:.0f}) so the refined pass "
                      "can look where the full-frame pass missed")

    h, wid = bgr.shape[:2]
    return (max(0, int(x0)), max(0, int(y0)),
            min(wid, int(x1)), min(h, int(y1)))


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
    # (row, col) -> the detection occupying that cell. Exposed so a caller can classify
    # the same cell across several frames and vote, which one frame cannot do.
    detected_cell: dict = field(default_factory=dict)
    # (row, col) -> the FITTED pixel centre of every cell, detected or inferred. Exposed
    # so a caller can sample an appearance anchor across several frames and vote it.
    cells_px: dict = field(default_factory=dict)
    votes: list[str] = field(default_factory=list)   # how a voted read was decided
    # True only when anchors were actually READ and the evidence still fails to put the
    # unshifted alignment ahead - i.e. something points at a shift. A refusal with NO
    # anchor read either way is a different thing entirely (no evidence, not contrary
    # evidence) and leaves this False. Callers must branch on this flag, never on the
    # wording of `reason`: a safety decision taken by substring match is one rewording
    # away from silently inverting.
    alignment_doubt: bool = False
    # Mean grey level of the frame this report came from. Recorded because "0 buttons
    # detected" has two completely different causes that read identically in a log: the
    # detector failing, and THE ROOM LIGHT being off. Measured 2026-09-16, lights off
    # took the frame mean 162 -> 36 and every anchor to 0/N, and neither exposure nor
    # gain recovers it — so the first question after a detection failure is what the
    # brightness was, and a log that cannot answer it sends the next hour into the
    # classifier instead. Two runs on 2026-09-18 died with 0 buttons on all 15 frames
    # from the best-positioned landing of the session, and this number was missing.
    frame_mean: float = 0.0

    def lines(self) -> list[str]:
        out = [f"detected {self.detected} buttons"]
        if self.detected == 0 and self.frame_mean:
            out[0] += (f"  [frame mean {self.frame_mean:.0f}/255 — "
                       + ("LOW: suspect the room light, not the classifier"
                          if self.frame_mean < 80 else
                          "brightness is normal, so this is not the light")
                       + "]")
        if self.residual:
            out[0] += f", lattice residual {self.residual:.1f} px"
        if self.inferred:
            out.append(f"inferred {len(self.inferred)} missed cell(s) from the lattice: "
                       + ", ".join(self.inferred))
        out.extend(self.votes)
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


def cell_brightness(bgr, cells_px: dict, button_px: float) -> dict:
    """Relative brightness of each grid cell: its own mean over the frame's median.

    Sampled at the LATTICE position, so a cell counts whether or not the detector found
    it. That matters here: the black disc is exactly the thing a button detector trained
    on bright discs tends to miss (measured 4 of 12 frames one run, 9 of 12 another), and
    an anchor that is usually absent cannot testify.

    Is it circular to verify the lattice using positions the lattice predicted? No, and
    the distinction is worth being precise about. The fit is made from POSITIONS; this
    asks what the thing AT a position looks like. Shift the grid by a row and the cell
    the layout calls `dot` is predicted onto a bright metal button, and the ratio jumps
    from ~0.23 to ~1.0 — so it still separates the alignments, which is the whole job.
    What it cannot catch on its own is a fit with the wrong PITCH, where the patch could
    land between buttons; the residual gate and the extrapolation check guard that.

    Returns {(row, col): ratio}; cells whose patch falls outside the image are omitted,
    which reads downstream as "no evidence here" rather than as a pass.
    """
    import numpy as _np

    h, w = bgr.shape[:2]
    r = max(3, int(button_px * 0.35))
    vals, raw = {}, {}
    for k, (u, v) in cells_px.items():
        u, v = int(round(u)), int(round(v))
        if u - r < 0 or v - r < 0 or u + r >= w or v + r >= h:
            continue
        raw[k] = float(bgr[v - r:v + r, u - r:u + r].mean())
    if len(raw) < 4:
        return {}
    med = float(_np.median(list(raw.values())))
    if med <= 1.0:
        return {}
    for k, val in raw.items():
        vals[k] = val / med
    return vals


def assign_voted(bgrs: list[np.ndarray], detector: Detector, layout: PanelLayout,
                 min_good: int = 3, min_conf: float | None = None,
                 max_residual: float | None = None,
                 min_detected_frac: float | None = None,
                 ) -> tuple[dict[str, tuple[float, float]], Report, int]:
    """Decide the anchors by MAJORITY over a fixed set of frames, not from one frame.

    Returns ``(mapping, report, index)`` where ``index`` says which of ``bgrs`` the
    geometry came from, so the caller can pair it with that frame's depth.

    **Why this is not "retry until it passes".** The number of frames is fixed before
    any of them is looked at, every one of them votes, and the decision is taken once.
    Retrying is drawing again after seeing a result you did not like; this is averaging
    a measurement of something that is not moving. The distinction matters because the
    in-run retry loop deliberately refuses to retry past an alignment doubt, and this
    must not become a way around that.

    It exists because one frame is not enough on this panel. Measured 2026-09-16 at the
    registered viewing pose, over 8 localised frames: `open` read correctly 8/8 but `A`
    only 6/8, and every other cell is noise. Two usable anchors means that when `A`
    misreads, `open` alone leaves a 4-row shift a tie and the whole run is refused —
    which is correct, but happened on two consecutive attempts.

    Rules, all conservative:
    * geometry comes from the frame with the LOWEST lattice residual, chosen on the
      fit alone and therefore independently of how the anchors voted;
    * a cell counts as OBSERVED only if it was detected in a strict majority of the
      good frames, and a label wins only on a strict majority of ALL the frames asked
      for — so a split reads as nothing, nothing ties the alignments, and a tie refuses.
      Counting against the fixed total rather than the survivors stops a run with fewer
      usable frames from reaching a verdict a full set could not;
    * fewer than ``min_good`` frames with a usable geometry is a refusal, not a vote
      among whatever survived.
    """
    # ONE source of truth for the per-frame thresholds. These used to be repeated here
    # as defaults, and when `assign`'s `min_detected_frac` was lowered from 0.7 to 0.5 on
    # the evidence of the hold-out measurement, this copy stayed at 0.7 and SHADOWED it —
    # every frame that came through the voting path was still being asked for 7 of 10
    # detections. It presented as "the panel cannot be localised from this station":
    # `assign` called directly accepted 5 and 6 of 6 live frames while `assign_voted`
    # accepted 0 of 5 of the same scene, minutes apart. A duplicated default is a second
    # place to change, and the one that is forgotten fails silently.
    kw = {k: v for k, v in (("min_conf", min_conf),
                            ("max_residual", max_residual),
                            ("min_detected_frac", min_detected_frac)) if v is not None}

    n = len(bgrs)
    geom: list[tuple[int, tuple[int, int, int, int], Report]] = []
    for idx, bgr in enumerate(bgrs):
        roi = panel_roi(bgr, detector, shape=layout.shape, verbose=False)
        if roi is None:
            continue
        _, rep = assign(bgr, roi, detector, layout, verify=False, **kw)
        if rep.ok:
            geom.append((idx, roi, rep))

    if len(geom) < min_good:
        bad = Report(ok=False, detected=0)
        # Carry the brightness up to the refusal the caller actually sees: without it
        # a voted refusal reports "0 of 5 frames" and says nothing about whether the
        # lights were on, which is the first thing worth knowing.
        if bgrs:
            bad.frame_mean = float(np.mean(bgrs[0]))
        bad.reason = (f"only {len(geom)} of {n} frames gave a usable panel geometry "
                      f"(need {min_good}) — there is not enough evidence to vote on, "
                      f"and voting among whichever frames happened to survive is the "
                      f"retry-until-it-passes this is meant to avoid")
        return {}, bad, -1

    # Only the cells the shift test can ever reference: each anchor's own cell and the
    # cell every shift would put in its place. Classifying all ten would cost four
    # times as much for evidence nothing looks at.
    rows = layout.rows
    need = {((i + sh) % rows, j) for i, j, _ in layout.anchors for sh in range(rows)}
    seen_n = {k: 0 for k in need}
    tally: dict[tuple[int, int], dict[str, list[float]]] = {k: {} for k in need}
    for idx, _roi, rep in geom:
        for k in need:
            f = rep.detected_cell.get(k)
            if f is None:
                continue
            seen_n[k] += 1
            lbl, conf = classify_solo(bgrs[idx], f, detector)
            if conf >= layout.min_conf:
                tally[k].setdefault(lbl, []).append(conf)

    # Appearance evidence, voted the way the classes are: the MEDIAN ratio across the
    # good frames. A median because one frame with a glint in the patch should not drag
    # the verdict, and because the quantity is already very stable once normalised
    # (spread 0.01-0.03 against +-20 grey levels before normalising).
    rel_votes: dict = {}
    for idx_g, _roi_g, rep_g in geom:
        bw = (float(np.median([max(f.w, f.h) for f in rep_g.found]))
              if rep_g.found else 40.0)
        for k, v in cell_brightness(bgrs[idx_g], rep_g.cells_px, bw).items():
            rel_votes.setdefault(k, []).append(v)
    voted_rel = {k: float(np.median(v)) for k, v in rel_votes.items()
                 if len(v) > len(geom) / 2.0}

    # A label wins only on a strict majority of the FIXED frame count, not of however
    # many frames happened to survive. Tying it to the survivors would let a run where
    # two frames failed reach a verdict on two votes that a full set could not — fewer
    # observations making a decision EASIER, which is the same hazard the anchor scoring
    # was fixed for. `observed` still keys on the good frames, because a cell cannot be
    # seen in a frame that produced no geometry at all.
    half_all, half_good = n / 2.0, len(geom) / 2.0
    observed = {k for k in need if seen_n[k] > half_good}
    reads: dict[tuple[int, int], tuple[str, float]] = {}
    for k in observed:
        best = max(tally[k].items(), key=lambda kv: len(kv[1]), default=None)
        reads[k] = ((best[0], float(np.mean(best[1]))) if best and len(best[1]) > half_all
                    else ("", 0.0))

    idx, roi, _ = min(geom, key=lambda g: g[2].residual)
    mapping, rep = assign(bgrs[idx], roi, detector, layout, verify=True,
                          reads=reads, observed=observed, rel=voted_rel, **kw)
    rep.votes.append(f"anchors voted over {len(geom)} of {n} frames "
                     f"(geometry from the cleanest, residual {rep.residual:.1f} px)")
    for i, j, c in layout.anchors:
        k = (i, j)
        if c.dark is not None:
            vs = rel_votes.get(k, [])
            shown = ("%.2f over %d frames (%.2f-%.2f)"
                     % (np.median(vs), len(vs), min(vs), max(vs))) if vs else "none"
            rep.votes.append(f"    {c.label:<6} brightness vs frame median: {shown}")
            continue
        counts = ", ".join(f"{lbl or '-'} x{len(v)}" for lbl, v in
                           sorted(tally[k].items(), key=lambda kv: -len(kv[1]))) or "none"
        rep.votes.append(f"    {c.label:<6} seen {seen_n[k]}/{len(geom)}  votes: {counts}")
    return mapping, rep, idx


def assign(bgr: np.ndarray, roi: tuple[int, int, int, int], detector: Detector,
           layout: PanelLayout, min_conf: float = 0.25, verify: bool = True,
           max_residual: float = 6.0,
           min_detected_frac: float = 0.5,
           reads: dict | None = None,
           observed: set | None = None,
           rel: dict | None = None) -> tuple[dict[str, tuple[float, float]], Report]:
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

    ``reads`` and ``observed`` let a caller supply the anchor classifications from
    somewhere other than this one frame — see `assign_voted`, which votes them over a
    fixed number of frames. They are always supplied together: ``reads`` says what each
    cell was read as, ``observed`` says which cells there is evidence about at all, and
    the shift test needs both (absence of evidence must not acquit a shift). With
    neither, the classification and the evidence both come from this frame, as before.

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
    rep.frame_mean = float(np.mean(bgr))

    # 0.5, not the 0.7 this carried while the chest camera could see all ten. On the
    # arm the plunger is bolted to the SAME limb as the lens, so it hides two cells from
    # every frame at the registered viewing pose (`4` and `2`, measured 0/8) — 0.7 of ten
    # then demands 7 of the 8 that are ever visible, i.e. 87 %, and one ordinary missed
    # detection fails the frame. The threshold tightened when the camera moved, and
    # nobody changed it.
    #
    # Lowering it is only safe because the EXTRAPOLATION check below now guards the
    # thing this count was standing in for. Measured with that check in place, over
    # random held-out subsets: five detections give a worst inferred-cell error of
    # 5.56 mm against a 20 mm button (86.66 mm without it). Without the check this
    # number could not be moved.
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

    # An inferred cell must be INTERPOLATED, never extrapolated. Measured 2026-09-17 by
    # holding cells out of the fit and comparing the inferred position against where the
    # button was actually detected:
    #
    #   kept   median   p90    worst px   worst mm
    #     7     1.37    2.51     5.72       4.72
    #     6     1.71    5.57     6.27       5.18
    #     5     2.06    6.11   105.02      86.66   <- a different button entirely
    #     4     2.45   66.13   157.22     129.74
    #
    # and — the part that matters — **the residual gate does not catch it**: 334 of 360
    # five-point subsets fitted under the 6 px limit while placing a cell 87 mm away.
    # With six unknowns (origin, row vector, column vector) a five-point fit is nearly
    # exactly determined, so a low residual there is arithmetic, not evidence.
    #
    # What separates the safe cases from the catastrophic ones is not the COUNT but
    # whether the missing cells sit inside the span of the ones that were seen. A cell
    # between detected rows and columns is interpolated and lands within a few mm; one
    # beyond the last detected row is extrapolated along a lever arm and can land on a
    # different button. So the test is the extrapolation itself, which is the thing the
    # count was standing in for.
    at = lat["assigned"]
    if at:
        det_rows = {i for i, _ in at}
        det_cols = {j for _, j in at}
        r_lo, r_hi, c_lo, c_hi = (min(det_rows), max(det_rows),
                                  min(det_cols), max(det_cols))
        outside = [f"{layout.grid[i][j].label}"
                   for i, row in enumerate(layout.grid) for j, _ in enumerate(row)
                   if (i, j) not in at and not (r_lo <= i <= r_hi and c_lo <= j <= c_hi)]
        if outside:
            rep.reason = (f"{len(outside)} cell(s) would have to be EXTRAPOLATED past "
                          f"the detections ({', '.join(outside)}): rows "
                          f"{r_lo}-{r_hi} and columns {c_lo}-{c_hi} were seen, and a "
                          f"cell outside that span is not pinned by anything. Measured, "
                          f"such a cell can land 87 mm out while the lattice residual "
                          f"still passes, so the residual cannot be relied on here")
            return {}, rep
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
    rep.detected_cell = detected_cell
    rep.cells_px = lat["cells"]

    # Appearance evidence for every cell, sampled at the lattice positions. Computed
    # here when the caller did not vote it across frames.
    if rel is None:
        bw = float(np.median([max(f.w, f.h) for f in found])) if found else 40.0
        rel = cell_brightness(bgr, lat["cells"], bw)

    # Anchors: does the unshifted alignment explain the classes better than a shift?
    # Only REAL detections may vote — scoring inferred cells would be circular.
    #
    # An alignment must be dismissed by DISAGREEMENT, never by absent evidence. An
    # earlier version scored each alignment over whichever of its cells happened to be
    # detected in that frame, and a shifted alignment could therefore score 0 simply
    # because its cells were missed — which read as "the shift is ruled out" and let the
    # press proceed. Measured 2026-09-16 against a deliberately shifted layout: it was
    # correctly refused on 4 frames and ACCEPTED on the 5th, and the accepting frame was
    # the one with LESS evidence (2 inferred cells instead of 3). Fewer observations must
    # never turn a refusal into an acceptance. So every alignment is now scored over the
    # SAME anchors — those observed at both the unshifted and the shifted position — and
    # if that common set is empty there is nothing to tell the alignments apart, which is
    # a refusal rather than a pass.
    anchors = layout.anchors
    rep.anchors_total = len(anchors)
    if verify and anchors:
        solo: dict[tuple[int, int], tuple[str, float]] = {}

        def _read(k: tuple[int, int]) -> tuple[str, float]:
            if reads is not None:
                return reads.get(k, ("", 0.0))
            f = detected_cell.get(k)
            if f is None:
                return ("", 0.0)
            if k not in solo:
                solo[k] = classify_solo(bgr, f, detector)
            return solo[k]

        def seen(cell: "Cell", k: tuple[int, int]) -> bool:
            """Is there evidence about this cell, OF THE KIND this anchor needs?

            Not "did it read correctly" — a cell that was looked at and could not be
            read confidently still counts as evidence, and contributes 0 to every
            alignment, which makes them tie and therefore refuses. That is the
            conservative direction and it is deliberate.

            The kind matters. A CLASS anchor needs the detector to have found the cell,
            because only then can it be classified. An APPEARANCE anchor needs only a
            patch of image, which is why it can testify on frames where the detector
            missed the button entirely.
            """
            if cell.dark is not None:
                return k in rel
            return k in observed if observed is not None else k in detected_cell

        def hit(cell: "Cell", k: tuple[int, int]) -> bool:
            """Does what is AT k match what this anchor expects to be there?"""
            if cell.dark is not None:
                r = rel.get(k)
                return r is not None and r <= cell.dark
            got, conf = _read(k)
            return got == cell.expect and conf >= layout.min_conf

        def comparable(shift: int) -> list[tuple[int, int, "Cell"]]:
            """Anchors observed at BOTH alignments — the only fair basis to compare."""
            return [(i, j, c) for i, j, c in anchors
                    if seen(c, (i, j)) and seen(c, ((i + shift) % rows, j))]

        def score_over(shift: int, over) -> int:
            return sum(1 for i, j, c in over if hit(c, ((i + shift) % rows, j)))

        def score(shift: int) -> int:
            return score_over(shift, anchors if shift == 0 else comparable(shift))

        base = score(0)
        rep.anchors_ok = base
        for i, j, c in anchors:
            state = ("not observed" if not seen(c, (i, j)) else
                     ("ok" if hit(c, (i, j)) else "mismatch"))
            if c.dark is not None:
                r = rel.get((i, j))
                rep.anchor_detail.append(
                    f"{c.label:<6} expect dark<{c.dark:.2f} got "
                    f"{('%.2f' % r) if r is not None else '-':<8} {state}")
            else:
                got, conf = _read((i, j))
                rep.anchor_detail.append(
                    f"{c.label:<6} expect {c.expect:<6} got {(got or '-'):<8} "
                    f"{conf:.2f}  {state}")
        worst = None
        for sh in range(1, rows):
            common = comparable(sh)
            if not common:
                # Nothing was observed at both alignments, so this shift cannot be
                # ruled out by evidence. Absence is not acquittal.
                rep.reason = (f"shift {sh} cannot be ruled out: no anchor was observed "
                              f"at both the unshifted and the shifted position, so there "
                              f"is no evidence distinguishing them")
                return {}, rep
            b, o = score_over(0, common), score_over(sh, common)
            if worst is None or o - b > worst[1] - worst[2]:
                worst = (sh, o, b)
            if b <= o:
                # Both zero means nothing was read at either alignment: no evidence, not
                # contrary evidence. Still a refusal - an unverified layout is never
                # pressed - but it is about THIS attempt reads, so it may be retried
                # like any other frame-quality failure. Any non-zero score means the
                # classifier did read something and the shift survived it anyway, which
                # is the silent-wrong-floor case and must end the run.
                rep.alignment_doubt = (b > 0 or o > 0)
                how = ("explains the anchors at least as well"
                       if rep.alignment_doubt else
                       "cannot be ruled out - NO anchor was read confidently at either "
                       "alignment, so there is no evidence to separate them")
                rep.reason = (f"a shifted alignment {how} "
                              f"(unshifted {b} vs shift {sh} {o}, compared over the "
                              f"{len(common)} anchor(s) observed at both); refusing "
                              "rather than risk pressing the wrong floor")
                return {}, rep
        # (A branch here used to claim a frame could be accepted on the lattice alone
        # when no anchor read either way. It referenced an undefined name and was
        # unreachable — base == 0 means every shift scored at least as well, so the
        # loop above has already refused. Removed rather than left as a latent
        # NameError sitting inside a safety check.)

    rep.ok = True
    return mapping, rep
