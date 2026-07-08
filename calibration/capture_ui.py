"""Live capture UI for calibration — show the camera, capture on keypress.

A small OpenCV preview loop shared by both calibration scripts:

    [c]  capture the current frame
    [d]  delete the last capture
    [Esc] finish (or [q])

The board is detected each frame and overlaid so the operator can see coverage
before pressing ``c``. Capture / delete are delegated to callbacks so the same UI
drives both the intrinsics (store frame) and extrinsics (read robot + add sample)
workflows. Falls back to a prompt-based loop when no display is available
(headless / no ``$DISPLAY``) so it still works over SSH.
"""

from __future__ import annotations

import os
import sys
from typing import Callable

import numpy as np

from calibration.board import BoardDetector, CharucoBoard

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

# Callback signatures:
#   on_capture(frame, board_visible: bool) -> str   (status message to print)
#   on_delete() -> str
#   get_hud() -> list[str]                           (extra HUD lines)
CaptureFn = Callable[[object, bool], str]
DeleteFn = Callable[[], str]
HudFn = Callable[[], list]


def display_available() -> bool:
    """True if an OpenCV GUI window can plausibly be shown."""
    if cv2 is None:
        return False
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True  # macOS / Windows generally have a display


def _draw_hud(vis, hud_lines: list, board_visible: bool, n_corners: int) -> None:
    h = vis.shape[0]
    green, red, white = (0, 200, 0), (0, 0, 255), (255, 255, 255)
    status = f"BOARD DETECTED ({n_corners} corners)" if board_visible else "no board"
    cv2.putText(vis, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                green if board_visible else red, 2)
    y = 60
    for line in hud_lines:
        cv2.putText(vis, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, white, 2)
        y += 28
    cv2.putText(vis, "[c] capture   [d] delete last   [Esc] finish",
                (15, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def live_capture_loop(
    *,
    camera,
    board: CharucoBoard,
    on_capture: CaptureFn,
    on_delete: DeleteFn,
    get_hud: HudFn,
    window: str = "calibration capture",
    image_is_rgb: bool = True,
    force_headless: bool = False,
) -> None:
    """Run the capture UI until the operator finishes (Esc/q)."""
    if cv2 is None:
        raise ImportError("opencv-contrib-python is required for the capture UI.")
    if force_headless or not display_available():
        _headless_loop(camera, board, on_capture, on_delete, get_hud)
        return

    det = BoardDetector(board=board, accurate=False)  # fast overlay detection
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = camera.capture()
            rgb = frame.rgb
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if image_is_rgb else rgb.copy()

            found = det.detect_corners(rgb)
            visible = found is not None
            if visible:
                corners, _ids, _b = found
                for c in corners.reshape(-1, 2):
                    cv2.circle(vis, (int(round(c[0])), int(round(c[1]))), 5, (0, 200, 0), -1)
            _draw_hud(vis, get_hud(), visible, det.last_corners)

            cv2.imshow(window, vis)
            # Bail out if the operator closed the window with the title-bar button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):       # Esc / q -> finish
                break
            elif key == ord("c"):
                print("  " + on_capture(frame, visible))
            elif key == ord("d"):
                print("  " + on_delete())
    finally:
        cv2.destroyWindow(window)


def _headless_loop(camera, board, on_capture, on_delete, get_hud) -> None:
    """Prompt-based fallback when no display is available (e.g. over SSH)."""
    det = BoardDetector(board=board, accurate=False)
    print("(no display — headless capture) commands: c=capture, d=delete last, q/Esc=finish")
    while True:
        try:
            cmd = input(f"[{' | '.join(get_hud())}] c/d/q > ").strip().lower()
        except EOFError:
            break
        if cmd in {"q", "quit", "esc", ""}:
            break
        if cmd == "d":
            print("  " + on_delete())
            continue
        if cmd == "c":
            frame = camera.capture()
            visible = det.detect_corners(frame.rgb) is not None
            print("  " + on_capture(frame, visible))
