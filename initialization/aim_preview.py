"""Live chest-camera preview for AIMING the base at a panel, in a browser.

The robot is headless, so positioning it by hand meant grabbing a still, copying it
off, looking, nudging, repeating. This streams the chest camera as MJPEG with the
button detector running on every frame, and answers the only question that matters
while someone is pushing the robot around: *is the whole faceplate in view, at the
right distance, square on?*

    python3 initialization/aim_preview.py                 # http://<robot-ip>:8010/
    python3 initialization/aim_preview.py --port 8011 --expect 10

Read-only: it opens the camera and (optionally) the calibration, and never touches
the arm, the hand, the lift or the base. Nothing here presses anything.

The overlay reports, per frame:

* every detected button, and how many against the layout's expected count;
* whether any detection touches the image border — a faceplate cut off by the edge
  detects as a partial grid, which is the failure that looks like "the model got
  worse" when it is really "half the panel is outside the picture";
* the panel's distance and bearing off the camera axis, and its position in the ARM
  BASE frame, which is the frame reachability is actually decided in.

Bearing and base-frame position come from the depth of the detected buttons only, so
they mean nothing until buttons are found — the HUD says so rather than printing a
number derived from whatever happened to be in a fixed box.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from math import atan2, degrees
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from core.camera.orbbec import OrbbecCamera  # noqa: E402
from core.config import REPO_ROOT  # noqa: E402
from core.transforms import deproject_pixel, transform_point  # noqa: E402

# What "good" looks like, measured on the mock panel over every successful press:
# the faceplate sits ~0.68 m in front of the arm base and ~0.25 m to its right.
TARGET_X_M = 0.68
TARGET_Y_M = -0.25
TOL_X_M = 0.08
TOL_BEARING_DEG = 8.0

PAGE = """<!doctype html><meta charset=utf-8><title>DEX aim preview</title>
<style>
 body{margin:0;background:#111;color:#eee;font:14px system-ui,sans-serif;text-align:center}
 img{max-width:100vw;height:auto;display:block;margin:0 auto}
 p{margin:8px}
</style>
<img src="/stream.mjpg" alt="chest camera">
<p>chest camera &mdash; live. Green = ready to press, red = not yet.</p>
"""


def _put(img, text, xy, colour, scale=0.7, thick=2):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def annotate(rgb, depth, intr, base_T_cam, dets, expect):
    """Draw the detections and the aiming HUD. Returns a BGR image."""
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    cv2.line(img, (w // 2, 0), (w // 2, h), (0, 200, 0), 1)

    boxes = [d.bbox_xyxy for d in dets]
    touching = False
    for (x1, y1, x2, y2) in boxes:
        edge = x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3
        touching = touching or edge
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 0, 255) if edge else (0, 220, 255), 2)

    lines, ok = [], False
    if not boxes:
        lines.append(("NO BUTTONS IN VIEW", (0, 0, 255)))
    else:
        xs = [(b[0] + b[2]) / 2 for b in boxes]
        ys = [(b[1] + b[3]) / 2 for b in boxes]
        cx, cy = float(np.mean(xs)), float(np.mean(ys))
        bearing = degrees(atan2(cx - intr.cx, intr.fx))

        # Depth over the detected buttons only — never a fixed box, which would
        # happily measure a wall or a person standing in front of the robot.
        zs = []
        for (x1, y1, x2, y2) in boxes:
            patch = depth[int(y1):int(y2), int(x1):int(x2)]
            good = patch[patch > 0.05]
            if good.size:
                zs.append(float(np.median(good)))
        dist = float(np.median(zs)) if zs else None

        n_col = (0, 220, 0) if len(boxes) >= expect else (0, 165, 255)
        lines.append((f"buttons {len(boxes)}/{expect}", n_col))
        if touching:
            lines.append(("PANEL CUT OFF BY IMAGE EDGE", (0, 0, 255)))
        lines.append((f"bearing {bearing:+.1f} deg"
                      f"  {'OK' if abs(bearing) <= TOL_BEARING_DEG else 'TURN'}",
                      (0, 220, 0) if abs(bearing) <= TOL_BEARING_DEG else (0, 165, 255)))

        if dist is not None:
            lines.append((f"distance {dist:.2f} m", (220, 220, 220)))
            if base_T_cam is not None:
                p_cam = deproject_pixel(intr, cx, cy, dist)
                bx, by, bz = transform_point(base_T_cam, p_cam)
                dx = bx - TARGET_X_M
                lines.append((f"base x{bx:+.2f} y{by:+.2f} z{bz:+.2f} m",
                              (220, 220, 220)))
                lines.append((f"x off target {dx:+.2f} m"
                              f"  (want {TARGET_X_M:.2f})",
                              (0, 220, 0) if abs(dx) <= TOL_X_M else (0, 165, 255)))
                ok = (len(boxes) >= expect and not touching
                      and abs(bearing) <= TOL_BEARING_DEG and abs(dx) <= TOL_X_M)
        cv2.drawMarker(img, (int(cx), int(cy)), (255, 0, 255), cv2.MARKER_CROSS, 26, 2)

    banner = "READY" if ok else "NOT READY"
    cv2.rectangle(img, (0, 0), (w, 40), (0, 120, 0) if ok else (0, 0, 140), -1)
    _put(img, banner, (12, 29), (255, 255, 255), 0.9)
    for i, (text, colour) in enumerate(lines):
        _put(img, text, (12, 72 + 30 * i), colour)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="cam_chest")
    ap.add_argument("--match", default="335", help="device name suffix (335 / 335L)")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--expect", type=int, default=10,
                    help="buttons the registered layout has (mock_cabinet = 10)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--exposure", default="156",
                    help="locked value, or 'auto' to let the camera meter")
    ap.add_argument("--gain", type=int, default=16)
    args = ap.parse_args()

    exposure = args.exposure if args.exposure == "auto" else int(args.exposure)

    ext_path = REPO_ROOT / "data/calibration" / f"{args.camera}.npy"
    base_T_cam = np.load(ext_path) if ext_path.exists() else None
    if base_T_cam is None:
        print(f"[warn] no extrinsic at {ext_path} — base-frame readout disabled")

    from yolo.trt_detector import TrtButtonDetector
    det = TrtButtonDetector(conf=args.conf)
    print("[aim] detector ready")

    cam = OrbbecCamera(camera_id=args.camera, match_name=args.match,
                       exposure=exposure, gain=args.gain)
    cam.start()
    print("[aim] camera ready")

    state = {"jpeg": None}
    stop = threading.Event()

    def grab():
        while not stop.is_set():
            try:
                f = cam.capture()
                dets = det.detect(f.rgb)
                img = annotate(f.rgb, f.depth, f.intrinsics, base_T_cam,
                               dets, args.expect)
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    state["jpeg"] = buf.tobytes()
            except Exception as e:  # noqa: BLE001 — a preview must not die on one frame
                print("[aim] frame error:", e)
                time.sleep(0.3)

    threading.Thread(target=grab, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while not stop.is_set():
                        jpg = state["jpeg"]
                        if jpg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n"
                                         + jpg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"[aim] open http://<robot-ip>:{args.port}/   (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        cam.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
