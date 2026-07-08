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


def web_capture_loop(
    *,
    camera,
    board: CharucoBoard,
    on_capture: CaptureFn,
    on_delete: DeleteFn,
    get_hud: HudFn,
    host: str = "0.0.0.0",
    port: int = 8010,
    image_is_rgb: bool = True,
) -> None:
    """Browser-based capture for a **headless robot** — stream the camera (with the
    board overlay) as MJPEG to a web page and capture via on-page buttons/keys.

    Same callback contract as :func:`live_capture_loop`
    (``on_capture(frame, board_visible)`` / ``on_delete()``), so both calibration
    scripts reuse it unchanged. The Orbbec is a single pipeline, so exactly ONE
    loop here grabs frames; the HTTP handlers only read the latest annotated JPEG
    (for the stream) and the latest raw frame (for a capture). Open
    ``http://<robot-ip>:<port>/`` on the viewing machine (same LAN). Stdlib only.
    """
    import json
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if cv2 is None:
        raise ImportError("opencv-contrib-python is required for the web capture UI.")

    det = BoardDetector(board=board, accurate=False)      # fast overlay detection
    state = {"jpeg": None, "frame": None, "visible": False, "msg": "", "seq": 0}
    flock = threading.Lock()     # guards state (frame/jpeg/seq/msg)
    alock = threading.Lock()     # serializes capture/delete callbacks
    stop = threading.Event()

    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>calibration capture</title><style>"
        "body{font-family:sans-serif;background:#111;color:#eee;text-align:center;margin:0;padding:10px}"
        "img{max-width:100%;height:auto;border:1px solid #333;background:#000}"
        "button{font-size:18px;padding:10px 16px;margin:6px;border:0;border-radius:8px;cursor:pointer;color:#fff}"
        "#cap{background:#2a8a5f}#del{background:#a5602a}#fin{background:#555}"
        "#status{font-size:16px;margin-top:8px;min-height:24px}"
        "</style></head><body>"
        "<img id='v' src='/stream'><div>"
        "<button id='cap' onclick=\"act('/capture')\">Capture (c)</button>"
        "<button id='del' onclick=\"act('/delete')\">Delete last (d)</button>"
        "<button id='fin' onclick=\"act('/finish')\">Finish</button></div>"
        "<div id='status'>…</div><script>"
        "async function act(p){const r=await fetch(p,{method:'POST'});"
        "document.getElementById('status').textContent=await r.text();poll();}"
        "async function poll(){try{const r=await fetch('/status');const j=await r.json();"
        "document.getElementById('status').textContent=(j.hud||[]).join('  |  ')+(j.msg?('  —  '+j.msg):'');}"
        "catch(e){}}setInterval(poll,700);poll();"
        "document.addEventListener('keydown',e=>{if(e.key==='c')act('/capture');if(e.key==='d')act('/delete');});"
        "</script></body></html>"
    ).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):     # keep the console clean
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", page)
            elif self.path == "/status":
                with flock:
                    body = json.dumps({"hud": get_hud(), "msg": state["msg"]}).encode()
                self._send(200, "application/json", body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last = -1
                try:
                    while not stop.is_set():
                        with flock:
                            jpeg, seq = state["jpeg"], state["seq"]
                        if jpeg is not None and seq != last:
                            last = seq
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
                            self.wfile.write(str(len(jpeg)).encode())
                            self.wfile.write(b"\r\n\r\n")
                            self.wfile.write(jpeg)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.04)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path == "/capture":
                with flock:
                    frame, visible = state["frame"], state["visible"]
                if frame is None:
                    msg = "no frame yet — wait for the preview"
                else:
                    with alock:
                        msg = on_capture(frame, visible)
                with flock:
                    state["msg"] = msg
                self._send(200, "text/plain; charset=utf-8", msg.encode())
            elif self.path == "/delete":
                with alock:
                    msg = on_delete()
                with flock:
                    state["msg"] = msg
                self._send(200, "text/plain; charset=utf-8", msg.encode())
            elif self.path == "/finish":
                stop.set()
                self._send(200, "text/plain", b"finishing")
            else:
                self._send(404, "text/plain", b"not found")

    try:
        server = ThreadingHTTPServer((host, port), Handler)
    except OSError as e:
        raise SystemExit(f"could not bind the web preview to {host}:{port} ({e}); "
                         "pass a different --web-port.")
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[web capture] open  http://<robot-ip>:{port}/  in your browser "
          f"(this robot is likely http://192.168.11.41:{port}/).")
    print("             capture with the page's buttons or keys (c/d); click Finish "
          "(or Ctrl-C here) to end.")
    try:
        while not stop.is_set():
            try:
                frame = camera.capture()
            except Exception as e:      # a transient grab error must not kill the session
                print(f"  [web capture] frame grab failed, retrying: {e}")
                time.sleep(0.1)
                continue
            rgb = frame.rgb
            vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if image_is_rgb else rgb.copy()
            found = det.detect_corners(rgb)
            visible = found is not None
            if visible:
                corners, _ids, _b = found
                for c in corners.reshape(-1, 2):
                    cv2.circle(vis, (int(round(c[0])), int(round(c[1]))), 5, (0, 200, 0), -1)
            _draw_hud(vis, get_hud(), visible, det.last_corners)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                with flock:
                    state["jpeg"] = buf.tobytes()
                    state["frame"] = frame
                    state["visible"] = visible
                    state["seq"] += 1
    except KeyboardInterrupt:
        print("\n[web capture] stopping.")
    finally:
        stop.set()
        server.shutdown()
