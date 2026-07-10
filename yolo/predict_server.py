"""Browser验证工具:上传一张电梯面板图 → YOLO 检测按钮并识别楼层 → 页面显示标注图 + 清单。

机器人无头(没显示器),所以和标定的 ``--web`` 一样:在机器人上跑这个服务,浏览器在
局域网访问 ``http://<robot-ip>:8011/``,选图上传即可看到框和楼层标签。

只用已装的依赖(ultralytics + cv2 + numpy + 标准库),无 Flask。前端直接把图片二进制 POST
到 ``/predict?conf=..``,服务端推理后返回 JSON(内嵌 base64 标注图 + 检测清单)。

run(在机器人的 ultralytics env):
    PYTHONPATH=~/Dex_Elevator ~/miniconda3/envs/ultralytics/bin/python yolo/predict_server.py
"""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np

from core.config import REPO_ROOT

MODEL = None
DEFAULT_IMGSZ = 1280

PAGE = r"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>电梯按钮识别 · 验证</title>
<style>
  :root { color-scheme: light dark; --bg:#f6f7f9; --card:#fff; --fg:#1a1a1a; --mut:#666; --line:#e3e5e8; --accent:#2f6feb; }
  @media (prefers-color-scheme: dark){ :root{ --bg:#15171a; --card:#1e2126; --fg:#e8eaed; --mut:#9aa0a6; --line:#2c2f36; --accent:#5b8dff; } }
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,Segoe UI,Roboto,"PingFang SC",sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:24px}
  h1{font-size:19px;margin:0 0 4px} .sub{color:var(--mut);font-size:13px;margin-bottom:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
  #drop{border:2px dashed var(--line);border-radius:12px;padding:34px;text-align:center;color:var(--mut);cursor:pointer;transition:.15s}
  #drop:hover,#drop.hot{border-color:var(--accent);color:var(--fg)}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
  .ctl{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--mut)}
  input[type=range]{accent-color:var(--accent)}
  .grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
  @media(max-width:720px){.grid{grid-template-columns:1fr}}
  img#out{width:100%;border-radius:10px;border:1px solid var(--line);display:block}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}
  th{color:var(--mut);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
  .lab{font-weight:700;font-size:16px} .bar{height:6px;border-radius:3px;background:var(--accent);min-width:2px}
  .muted{color:var(--mut)} .big{font-size:22px;font-weight:700}
  .hidden{display:none}
</style></head>
<body><div class="wrap">
  <h1>电梯按钮识别 · 验证</h1>
  <div class="sub">上传一张面板图,模型框出每个按钮并标注识别到的楼层 / 信息。</div>

  <div class="card">
    <div id="drop">点击选择图片,或拖拽到这里</div>
    <input id="file" type="file" accept="image/*" class="hidden">
    <div class="row" style="margin-top:14px">
      <div class="ctl">置信度阈值 <input id="conf" type="range" min="5" max="90" value="25"><span id="confv">0.25</span></div>
      <div class="ctl muted" id="status"></div>
    </div>
  </div>

  <div id="result" class="card hidden">
    <div class="grid">
      <div><img id="out" alt="结果"></div>
      <div>
        <div class="row" style="justify-content:space-between;margin-bottom:8px">
          <div class="big" id="count">–</div><div class="muted" id="ms"></div>
        </div>
        <table><thead><tr><th>#</th><th>楼层 / 信息</th><th>置信度</th></tr></thead>
        <tbody id="rows"></tbody></table>
        <div class="muted" style="font-size:12px;margin-top:10px">按面板从上到下排序。</div>
      </div>
    </div>
  </div>

<script>
const $=s=>document.querySelector(s);
const drop=$("#drop"),file=$("#file"),conf=$("#conf"),confv=$("#confv");
conf.oninput=()=>confv.textContent=(conf.value/100).toFixed(2);
drop.onclick=()=>file.click();
["dragover","dragenter"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add("hot")}));
["dragleave","drop"].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove("hot")}));
drop.addEventListener("drop",ev=>{if(ev.dataTransfer.files[0])send(ev.dataTransfer.files[0])});
file.onchange=()=>{if(file.files[0])send(file.files[0])};

async function send(f){
  $("#status").textContent="推理中… ("+f.name+")";
  try{
    const c=(conf.value/100).toFixed(2);
    const r=await fetch("/predict?conf="+c,{method:"POST",body:f});
    if(!r.ok){throw new Error(await r.text())}
    const d=await r.json();
    $("#out").src="data:image/jpeg;base64,"+d.image;
    $("#count").textContent=d.detections.length+" 个检测";
    $("#ms").textContent=d.ms+" ms · imgsz "+d.imgsz;
    $("#rows").innerHTML=d.detections.map((x,i)=>
      `<tr><td class="muted">${i+1}</td><td class="lab">${x.label}</td>
       <td><div style="display:flex;align-items:center;gap:8px">
       <div class="bar" style="width:${Math.round(x.conf*90)}px"></div>${(x.conf*100).toFixed(0)}%</div></td></tr>`).join("");
    $("#result").classList.remove("hidden");
    $("#status").textContent="完成";
  }catch(e){ $("#status").textContent="出错:"+e.message; }
}
</script>
</div></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 安静
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/":
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(204, b"")

    def do_POST(self):
        if urlparse(self.path).path != "/predict":
            self._send(404, "not found", "text/plain")
            return
        try:
            qs = parse_qs(urlparse(self.path).query)
            conf = float(qs.get("conf", ["0.25"])[0])
            n = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(n)
            img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                self._send(400, "无法解码图片", "text/plain")
                return
            import time
            t0 = time.time()
            r = MODEL.predict(img, imgsz=DEFAULT_IMGSZ, conf=conf, verbose=False)[0]
            ms = round((time.time() - t0) * 1000)
            annotated = r.plot()  # BGR, 带框+标签+置信度
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            dets = []
            for b in r.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append({"label": MODEL.names[int(b.cls[0])],
                             "conf": float(b.conf[0]), "y": y1,
                             "box": [round(x1), round(y1), round(x2), round(y2)]})
            dets.sort(key=lambda d: d["y"])  # 面板从上到下
            self._send(200, json.dumps({"image": b64, "detections": dets,
                                        "ms": ms, "imgsz": DEFAULT_IMGSZ}))
        except Exception as e:  # noqa
            self._send(500, f"推理出错: {e}", "text/plain")


def main() -> None:
    global MODEL, DEFAULT_IMGSZ
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", default=str(REPO_ROOT / "data/weights/buttons.pt"))
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    args = ap.parse_args()

    from ultralytics import YOLO
    DEFAULT_IMGSZ = args.imgsz
    MODEL = YOLO(args.weights)
    print(f"loaded {args.weights}  ({len(MODEL.names)} classes)")
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"验证页面: http://<robot-ip>:{args.port}/   (Ctrl-C 停止)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
