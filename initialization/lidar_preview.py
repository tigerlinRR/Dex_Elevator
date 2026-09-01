"""Live top-down lidar view from the chassis, in a browser.

Why this exists: positioning the robot for a straight drive needs to know what is
behind it as well as in front, and the chest camera structurally cannot answer that —
it faces only the arm side. The chassis's own 360-degree lidar can, and it publishes
at 1.87 Hz on the same local 8090 API that the straight-line driver uses.

    python3 initialization/lidar_preview.py            # http://<robot-ip>:8011/
    python3 initialization/lidar_preview.py --span 12  # wider view, metres across

READ-ONLY. Subscribes to /scan_matched_points2 and /tracked_pose and draws them. It
never changes the control mode and never sends a twist, so it is safe to leave running
while someone pushes the robot around by hand.

What the overlay shows, and why each number is there:

* **forward / back clearance** inside the base's own width — the two numbers the
  straight-line driver gates on, so the browser shows exactly what it will decide on.
  The hull already reaches 0.40 m along the axis, so the usable travel figure has that
  subtracted; that is the number displayed.
* **corridor width and centring** — left and right distance separately. Driving into
  an elevator straight is only useful if the robot starts near the middle of the
  opening, and a single "nearest obstacle" number hides which side it is drifting to.
* the raw points, so a misleading summary can always be checked against the scan.

Points arrive in MAP coordinates; they are rotated into the body frame with the current
pose. Angle 0 and the +X axis of the drawing are along `ori`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import websocket  # noqa: E402

HOST = "192.168.25.25"
PORT = 8090

# From /robot/footprint: the base spans x -0.40..0.40, y -0.369..0.379.
HALF_WIDTH_M = 0.40
HULL_M = 0.40
BODY_L, BODY_W = 0.80, 0.76

PAGE = """<!doctype html><meta charset=utf-8><title>DEX lidar</title>
<style>
 body{margin:0;background:#0b0b0b;color:#ddd;font:14px system-ui,sans-serif;text-align:center}
 img{max-width:100vw;height:auto;display:block;margin:0 auto}
 p{margin:8px;color:#999}
</style>
<img src="/stream.mjpg" alt="lidar">
<p>chassis lidar, top-down &mdash; robot at centre, nose points right (+ori).
Read-only: this page never commands the base.</p>
"""


def _put(img, text, xy, colour, scale=0.6, thick=1):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick)


def render(pts_body, span_m, size=720):
    """Draw the body-frame points. +X (along ori) is drawn to the RIGHT."""
    img = np.full((size, size, 3), 18, np.uint8)
    ppm = size / span_m                       # pixels per metre
    cx = cy = size // 2

    # range rings every metre
    for r in range(1, int(span_m / 2) + 1):
        cv2.circle(img, (cx, cy), int(r * ppm), (44, 44, 44), 1)
        _put(img, f"{r}m", (cx + int(r * ppm) - 22, cy - 6), (90, 90, 90), 0.45)
    cv2.line(img, (0, cy), (size, cy), (52, 52, 52), 1)
    cv2.line(img, (cx, 0), (cx, size), (52, 52, 52), 1)

    # the travel corridor the driver actually gates on
    half_px = int(HALF_WIDTH_M * ppm)
    cv2.line(img, (0, cy - half_px), (size, cy - half_px), (70, 70, 110), 1)
    cv2.line(img, (0, cy + half_px), (size, cy + half_px), (70, 70, 110), 1)

    fwd, back, left, right = [], [], [], []
    for bx, by in pts_body:
        px = int(cx + bx * ppm)
        py = int(cy - by * ppm)
        in_corridor = abs(by) <= HALF_WIDTH_M
        if 0 <= px < size and 0 <= py < size:
            cv2.circle(img, (px, py), 2,
                       (90, 220, 255) if in_corridor else (110, 110, 110), -1)
        if in_corridor:
            (fwd if bx > 0 else back).append(abs(bx))
        if abs(bx) <= HALF_WIDTH_M:
            (left if by > 0 else right).append(abs(by))

    # the robot's own footprint
    bl, bw = BODY_L * ppm / 2, BODY_W * ppm / 2
    cv2.rectangle(img, (int(cx - bl), int(cy - bw)), (int(cx + bl), int(cy + bw)),
                  (0, 200, 0), 2)
    cv2.arrowedLine(img, (cx, cy), (int(cx + bl * 1.5), cy), (0, 255, 0), 2,
                    tipLength=0.3)

    def usable(vals):
        if not vals:
            return None
        return max(0.0, min(vals) - HULL_M)

    f, b = usable(fwd), usable(back)
    lft = min(left) if left else None
    rgt = min(right) if right else None

    def txt(v):
        return "clear" if v is None else f"{v:.2f} m"

    def col(v):
        if v is None:
            return (120, 255, 120)
        return (120, 255, 120) if v >= 2.0 else ((120, 220, 255) if v >= 0.8
                                                else (110, 110, 255))

    lines = [
        (f"forward (+ori)  {txt(f)}", col(f)),
        (f"back    (-ori)  {txt(b)}", col(b)),
        (f"left  {txt(lft) if lft is None else f'{lft:.2f} m'}   "
         f"right {txt(rgt) if rgt is None else f'{rgt:.2f} m'}", (200, 200, 200)),
    ]
    if lft is not None and rgt is not None:
        width = lft + rgt
        off = (lft - rgt) / 2.0
        lines.append((f"corridor {width:.2f} m wide, robot {abs(off) * 100:.0f} cm "
                      f"{'left' if off > 0 else 'right'} of centre",
                      (120, 255, 120) if abs(off) < 0.15 else (120, 220, 255)))
    lines.append((f"{len(pts_body)} points   usable travel = nearest - "
                  f"{HULL_M:.2f} m hull", (140, 140, 140)))
    for i, (t, c) in enumerate(lines):
        _put(img, t, (12, 26 + 26 * i), c)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT, help="chassis API port")
    ap.add_argument("--http-port", type=int, default=8011)
    ap.add_argument("--span", type=float, default=8.0, metavar="M",
                    help="width of the view in metres")
    ap.add_argument("--secret", default="")
    args = ap.parse_args()

    state = {"jpeg": None}
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                header = [f"Secret: {args.secret}"] if args.secret else []
                ws = websocket.create_connection(
                    f"ws://{args.host}:{args.port}/ws/v2/topics",
                    header=header, timeout=8)
                ws.settimeout(0.2)
                for t in ("/scan_matched_points2", "/tracked_pose"):
                    ws.send(json.dumps({"enable_topic": t}))
                pose = None
                while not stop.is_set():
                    try:
                        raw = ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not raw:
                        continue
                    try:
                        msg = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    topic = msg.get("topic")
                    if topic == "/tracked_pose":
                        p, o = msg.get("pos"), msg.get("ori")
                        if p and o is not None:
                            pose = (float(p[0]), float(p[1]), float(o))
                    elif topic == "/scan_matched_points2" and pose:
                        pts = msg.get("points") or []
                        x0, y0, ori = pose
                        c, s = math.cos(ori), math.sin(ori)
                        body = [((p[0] - x0) * c + (p[1] - y0) * s,
                                 -(p[0] - x0) * s + (p[1] - y0) * c) for p in pts]
                        img = render(body, args.span)
                        ok, buf = cv2.imencode(".jpg", img,
                                               [cv2.IMWRITE_JPEG_QUALITY, 82])
                        if ok:
                            state["jpeg"] = buf.tobytes()
                ws.close()
            except Exception as e:  # noqa: BLE001 — a preview must survive a drop
                print("[lidar] reader error:", e)
                time.sleep(1.0)

    threading.Thread(target=reader, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
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
                            time.sleep(0.1)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n"
                                         + jpg + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE.encode())

    srv = ThreadingHTTPServer(("0.0.0.0", args.http_port), Handler)
    print(f"[lidar] open http://<robot-ip>:{args.http_port}/   (ctrl-c to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
