"""Locate panel buttons by Hough-circle detection — a stand-in for the YOLO detector.

Why this exists: the multi-class YOLO model trained on public CC BY data does
locate buttons on our panel (8-10 detections at 0.9 confidence once the exposure
is fixed), but it labels almost all of them ``empty`` — the buttons here are
same-colour metal with the digits embossed rather than printed, so there is no
contrast for it to read. Until the model is fine-tuned on our own captures, the
geometry pipeline still needs button centres, and these buttons are perfect
circles on a flat faceplate, which Hough finds reliably.

Accuracy matters more than it looks: eyeballed pixel coordinates were off by only
~2.5 px, yet that put the plunger on the button's chamfer instead of its centre
and it would not actuate even at 2 mm of push. Circle centres fixed it.

This module knows nothing about which floor each button is — it returns the grid
positions, and the caller maps them to labels via the panel's known layout. That
is exactly the job YOLO takes over once it can read the digits.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None


def find_circles(bgr, roi, param2: int = 32,
                 min_radius: int = 18, max_radius: int = 34):
    """Circle centres (u, v) inside ``roi``=(u0, v0, u1, v1). Input is **BGR**."""
    if cv2 is None:
        raise ImportError("opencv is required")
    u0, v0, u1, v1 = roi
    gray = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (5, 5), 1.5)
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.0, minDist=40,
                             param1=100, param2=param2,
                             minRadius=min_radius, maxRadius=max_radius)
    if found is None:
        return []
    return [(float(x), float(y)) for x, y, _r in found[0]
            if u0 < x < u1 and v0 < y < v1]


def to_grid(points, labels, row_tol: int = 28):
    """Sort circle centres into a rows x 2 grid and attach ``labels``.

    ``labels`` is a list of (left, right) name pairs, top row first.

    Deliberately tolerant of over/under-detection: cluster by v into rows, then
    take the leftmost and rightmost of each row. Requiring an exact count is what
    made an earlier version fail 2 out of 3 runs — a single button's edge dropping
    out under a reflection killed the whole cycle.
    """
    if len(points) < 2 * len(labels) - 2:
        return None
    pts = sorted(points, key=lambda q: q[1])
    rows: list[list] = []
    for q in pts:
        if rows and abs(q[1] - np.mean([z[1] for z in rows[-1]])) < row_tol:
            rows[-1].append(q)
        else:
            rows.append([q])
    rows = [r for r in rows if len(r) >= 2]
    if len(rows) != len(labels):
        return None
    out = {}
    for row, (left, right) in zip(rows, labels):
        row.sort(key=lambda q: q[0])
        out[left] = (round(row[0][0]), round(row[0][1]))
        out[right] = (round(row[-1][0]), round(row[-1][1]))
    return out


def detect_buttons(bgr, roi, labels, param2_candidates=(32, 28, 25, 36)):
    """``{label: (u, v)}`` or ``None``. Sweeps ``param2`` before giving up."""
    for p2 in param2_candidates:
        grid = to_grid(find_circles(bgr, roi, param2=p2), labels)
        if grid is not None:
            return grid, p2
    return None, None
