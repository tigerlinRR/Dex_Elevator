#!/usr/bin/env python3
"""Live view of the CHASSIS's own front camera, in a browser.

The chassis has a forward-facing camera — it records to /videos/ as `*-front.h264`
and publishes live on its websocket as `/rgb_cameras/front/compressed`. Nothing in
this repo used it before: the chassis sits on 192.168.25.x, a subnet only the
Jetson's eno1 can reach, so there was no way to look at it from a laptop.

Worth being clear about what this camera is and is not. It faces the base's own
direction of travel, which on this robot is **180 degrees away from the arm and the
chest camera** — so it watches where the robot drives, never where the arm reaches.
It is a monitoring and HRI view, not a sensor the press can use.

Read-only: subscribes, decodes, serves. It commands nothing.

    python3 initialization/base_camera_preview.py            # http://<robot-ip>:8013/
    python3 initialization/base_camera_preview.py --port 8013 --topic /rgb_cameras/front/video
"""

from __future__ import annotations

import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from websocket import create_connection

HOST = "192.168.25.25"
WS_PORT = 8090

_latest: dict = {"jpeg": None, "stamp": 0.0, "count": 0, "fps": 0.0, "err": ""}
_lock = threading.Lock()


def _reader(topic: str) -> None:
    """Subscribe and keep the most recent frame. Reconnects: the chassis drops idle sockets."""
    while True:
        try:
            ws = create_connection(f"ws://{HOST}:{WS_PORT}/ws/v2/topics", timeout=10)
            ws.send(json.dumps({"enable_topic": topic}))
            t0, n0 = time.time(), _latest["count"]
            while True:
                msg = ws.recv()
                if isinstance(msg, bytes):
                    continue
                d = json.loads(msg)
                if d.get("topic") != topic:
                    continue
                data = d.get("data")
                if not data:
                    continue
                try:
                    raw = base64.b64decode(data) if isinstance(data, str) else bytes(data)
                except Exception:
                    continue
                # Accept only real JPEGs; the topic can carry other encodings.
                if not raw.startswith(b"\xff\xd8"):
                    with _lock:
                        _latest["err"] = f"frame is not JPEG (starts {raw[:4].hex()})"
                    continue
                with _lock:
                    _latest["jpeg"] = raw
                    _latest["stamp"] = time.time()
                    _latest["count"] += 1
                    _latest["err"] = ""
                    dt = time.time() - t0
                    if dt >= 2.0:
                        _latest["fps"] = (_latest["count"] - n0) / dt
                        t0, n0 = time.time(), _latest["count"]
        except Exception as e:  # pragma: no cover - network dependent
            with _lock:
                _latest["err"] = f"{type(e).__name__}: {e}"
            time.sleep(2.0)


PAGE = b"""<!doctype html><html><head><meta charset='utf-8'>
<title>chassis front camera</title><style>
body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:12px}
img{max-width:100%;height:auto;border:1px solid #444;background:#000}
.h{font-size:14px;color:#9c9;margin:6px}
</style></head><body>
<div class='h'>chassis front camera &mdash; faces the direction of travel
(180&deg; away from the arm)</div>
<img id='v' src='/stream'>
<div class='h' id='s'>&nbsp;</div>
<script>
setInterval(async()=>{try{const r=await fetch('/status');const d=await r.json();
document.getElementById('s').textContent =
  d.frames+' frames  '+d.fps.toFixed(1)+' fps  '+(d.age<9?d.age.toFixed(1)+'s ago':'STALE')+
  (d.err?('  ['+d.err+']'):'');}catch(e){}},1000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):        # keep the console clean
        pass

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(PAGE)))
            self.end_headers()
            self.wfile.write(PAGE)
        elif self.path == "/status":
            with _lock:
                body = json.dumps({
                    "frames": _latest["count"], "fps": _latest["fps"],
                    "age": time.time() - _latest["stamp"] if _latest["stamp"] else 999,
                    "err": _latest["err"],
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = -1
            try:
                while True:
                    with _lock:
                        jpg, n = _latest["jpeg"], _latest["count"]
                    if jpg is not None and n != last:
                        last = n
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                         b"Content-Length: " + str(len(jpg)).encode() +
                                         b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--topic", default="/rgb_cameras/front/compressed")
    args = ap.parse_args()
    threading.Thread(target=_reader, args=(args.topic,), daemon=True).start()
    print(f"chassis front camera -> http://<robot-ip>:{args.port}/   (topic {args.topic})",
          flush=True)
    ThreadingHTTPServer(("0.0.0.0", args.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
