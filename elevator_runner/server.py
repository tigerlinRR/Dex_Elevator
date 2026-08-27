#!/usr/bin/env python3
"""Elevator Runner — a small standalone local tool.

Pick navigation points, mark one as the ELEVATOR point, set how many times to run
(a loop), and dispatch the robot through AutoXing. When the robot finishes a task
and stops AT the elevator point (within a tolerance), it SSHes into the AGX and
runs the button press (`press_buttons.py <floors> --go --lift`).

Why the tolerance gate matters: the arm re-detects the panel and re-fits the plane
live on every press, so it already absorbs a few cm of docking error — but past its
REACH ENVELOPE (~10 cm) it cannot reach the buttons at all. So we measure the real
arrival error against the elevator point and only press when it is close enough;
otherwise we RETRY the task (see the run loop for the retry policy).

This is deliberately standalone: it does NOT import the big CloudPlatform app. It
speaks the same AutoXing v3 endpoints that robot_api.py/task_poc.py use, with the
same auth (md5 sign -> token -> X-Token).

Config: copy config.example.json to config.json and fill it in (or use env vars).
Run:   python3 server.py   then open http://127.0.0.1:8765/
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def _load_dotenv(path: Path) -> None:
    """Populate os.environ from a .env file (KEY=value, optional quotes). Does not
    overwrite variables already set in the real environment. Kept dependency-free."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def load_config() -> dict:
    _load_dotenv(HERE / ".env")
    cfg = {
        "autoxing_host": os.environ.get("AUTOXING_HOST", "https://apiglobal.autoxing.com"),
        "app_id": os.environ.get("AUTOXING_APP_ID", ""),
        "app_secret": os.environ.get("AUTOXING_APP_SECRET", ""),
        "app_code": os.environ.get("AUTOXING_APP_CODE", ""),
        "robot_serial": os.environ.get("ROBOT_SERIAL", ""),
        # SSH target for the AGX press (uses your ~/.ssh/config alias). The press
        # command is run verbatim; floors are appended.
        "agx_ssh": os.environ.get("AGX_SSH", "dex5-wired"),
        "press_cmd": os.environ.get(
            "PRESS_CMD",
            "cd ~/Dex_Elevator && PYTHONPATH=. /usr/bin/python3 "
            "initialization/press_buttons.py {floors} --go --lift",
        ),
    }
    f = HERE / "config.json"
    if f.exists():
        try:
            cfg.update({k: v for k, v in json.loads(f.read_text()).items() if v != ""})
        except Exception as e:  # noqa: BLE001
            print(f"[config] could not read config.json: {e}")
    return cfg


CFG = load_config()


# ---------------------------------------------------------------------------
# AutoXing client (mirrors robot_api.py auth + endpoints)
# ---------------------------------------------------------------------------
class AutoXing:
    def __init__(self, cfg):
        self.host = cfg["autoxing_host"].rstrip("/")
        self.app_id = cfg["app_id"]
        self.app_secret = cfg["app_secret"]
        self.app_code = cfg["app_code"]
        self._token = None
        self._token_exp = 0
        self._s = requests.Session()

    def _url(self, path):
        return f"{self.host}{path if path.startswith('/') else '/' + path}"

    def token(self):
        if self._token and time.time() < self._token_exp - 30:
            return self._token
        ts = int(time.time() * 1000)
        sign = hashlib.md5(f"{self.app_id}{ts}{self.app_secret}".encode()).hexdigest()
        r = self._s.post(
            self._url("/auth/v1.1/token"),
            headers={"Authorization": self.app_code, "Content-Type": "application/json"},
            json={"appId": self.app_id, "timestamp": ts, "sign": sign},
            timeout=10,
        )
        r.raise_for_status()
        body = r.json()
        if body.get("status") != 200:
            raise RuntimeError(f"token error: {body.get('status')} {body.get('message')}")
        data = body["data"]
        self._token = data["token"]
        self._token_exp = time.time() + float(data.get("expireTime", 3600))
        return self._token

    def api(self, method, path, body=None, timeout=15):
        headers = {"X-Token": self.token(), "Content-Type": "application/json"}
        url = self._url(path)
        if method == "GET":
            r = self._s.get(url, headers=headers, timeout=timeout)
        else:
            r = self._s.post(url, headers=headers, json=body or {}, timeout=timeout)
        r.raise_for_status()
        return r.json()

    # -- specific endpoints --------------------------------------------------
    def list_pois(self, robot_serial):
        """All POIs for the robot's map. Returns list of dicts."""
        out, page = [], 1
        while page <= 20:
            body = self.api("POST", "/map/v1.1/poi/list",
                            {"robotId": robot_serial, "pageNum": page, "pageSize": 200})
            data = body.get("data") or []
            # data may be a list or {list:[...]}
            items = data if isinstance(data, list) else data.get("list", [])
            if not items:
                break
            for p in items:
                # AutoXing returns a POI's position as `coordinate: [x, y]`, NOT as
                # top-level x/y -- unlike robot_state, which does use top-level x/y.
                # Reading p["x"] therefore defaulted every POI to (0, 0), silently.
                # The arrival gate then measured the robot's distance to the MAP
                # ORIGIN (~29 m here) against an 8 cm tolerance, so a live run would
                # drive to the elevator, refuse to press as "out of reach", back off
                # and retry until it gave up. The dry run hides this because it
                # substitutes a fixed 3.0 cm simulated arrival error.
                coord = p.get("coordinate") or []
                out.append({
                    "id": p.get("id") or (p.get("ext") or {}).get("id", ""),
                    "name": (p.get("ext") or {}).get("name") or p.get("name") or "Waypoint",
                    "x": coord[0] if len(coord) > 0 else p.get("x", 0),
                    "y": coord[1] if len(coord) > 1 else p.get("y", 0),
                    "yaw": p.get("yaw", 0),
                    "type": p.get("type", -1), "areaId": p.get("areaId", ""),
                    "floor": p.get("floor", ""),
                })
            if len(items) < 200:
                break
            page += 1
        return out

    def robot_state(self, robot_serial):
        for path in (f"/robot/v2.0/{robot_serial}/state",
                     f"/robot/v1.1/{robot_serial}/state"):
            try:
                body = self.api("GET", path)
                if body.get("status") == 200 and body.get("data"):
                    return body["data"]
            except Exception:
                continue
        return {}

    def create_task(self, robot_serial, task_pts, name, run_mode=2):
        body = {
            "name": name, "robotId": robot_serial,
            "dispatchType": 0, "routeMode": 1, "runMode": int(run_mode),
            "runNum": 1, "taskType": 4, "runType": 22, "sourceType": 6,
            "ignorePublicSite": False, "speed": 1.0, "detourRadius": 1.0,
            "taskPts": task_pts,
        }
        return self.api("POST", "/task/v3/create", body)

    def task_status(self, task_id):
        return self.api("GET", f"/task/v3/{task_id}")

    def cancel_task(self, task_id):
        try:
            return self.api("POST", f"/task/v3/{task_id}/cancel", {})
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}


AX = AutoXing(CFG)

ELEVATOR_POI_TYPES = {6, 28}  # "Elevator Waiting Point" (from task_poc TASK_SELECTOR_POI_TYPES)


def make_task_point(poi, area_id, pause_time=0):
    step_acts = []
    if pause_time and pause_time > 0:
        step_acts.append({"type": 18, "data": {"pauseTime": pause_time}})
    return {
        "areaId": poi.get("areaId") or area_id,
        "x": float(poi["x"]), "y": float(poi["y"]), "yaw": int(poi.get("yaw", 0)),
        "type": -1, "stopRadius": 1,
        "ext": {"id": poi.get("id", ""), "name": poi.get("name", "Waypoint")},
        "stepActs": step_acts,
    }


# AutoXing reports a task's outcome as `taskStatus` (4 = completed, with
# `taskStatusNameEN: "completed"`), NOT as the boolean `isFinish`/`isCancel` that this
# code used to read. Those keys simply are not in the response, so `.get()` returned
# None and a task that had ALREADY completed was polled until the 240 s timeout and
# then reported as "task did not finish". Observed live: robot parked 3.3 cm from the
# elevator point with moveState "succeeded", taskStatus 4, endTime set, 8.2 m driven —
# and the runner sat there waiting. Same failure shape as the POI coordinates: a field
# name that does not exist, read through `.get()`, so it fails silently.
#
# The booleans are kept as a fallback in case a different API version does send them.
TASK_STATUS_COMPLETED = 4


def task_finished(st):
    if st.get("isFinish"):
        return True
    if st.get("taskStatus") == TASK_STATUS_COMPLETED:
        return True
    return str(st.get("taskStatusNameEN", "")).lower() == "completed"


def task_cancelled(st):
    if st.get("isCancel"):
        return True
    name = str(st.get("taskStatusNameEN", "")).lower()
    if name in ("cancelled", "canceled", "failed", "aborted"):
        return True
    # A non-zero failureCode on a task that has ended is a failure, not a completion.
    return bool(st.get("failureCode")) and bool(st.get("endTime"))


def dist_cm(ax, ay, bx, by):
    try:
        return ((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5 * 100.0
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Run manager — one run at a time, background thread
# ---------------------------------------------------------------------------
class Run:
    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = False
        self.active = False
        self.log_lines = []
        self.state = {}

    def log(self, msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        with self.lock:
            self.log_lines.append(line)
            self.log_lines = self.log_lines[-400:]
        print("[run]", line)

    def set(self, **kw):
        with self.lock:
            self.state.update(kw)

    def snapshot(self):
        with self.lock:
            return {"active": self.active, "state": dict(self.state),
                    "log": list(self.log_lines[-120:])}


RUN = Run()


def run_loop(params):
    """The orchestration loop. See the retry policy inline."""
    RUN.active = True
    RUN.stop_flag = False
    robot = params["robot_serial"]
    route = params["route"]                 # ordered list of POI dicts (last should be elevator)
    elevator = params["elevator"]           # POI dict
    floors = params["floors"]               # e.g. "3" or "1 4 2"
    loops = int(params["loops"])
    tol_cm = float(params["tolerance_cm"])
    # Yaw gating is OFF by default because a POI's stored yaw and robot_state's yaw
    # are NOT the same convention here: the elevator POI records 154.70 while the
    # robot arrives at ~2.7 every single time (two arrivals measured 2.71 and 2.69),
    # and 2.7 is exactly the orientation the arm pressed 4/4 from all day. Comparing
    # them rejected a perfectly good dock with a 152 deg "error". Set yaw_tol_deg
    # together with yaw_ref_deg to gate against an orientation known to work.
    yaw_tol = params.get("yaw_tol_deg")
    yaw_tol = None if yaw_tol in (None, "", 0) else float(yaw_tol)
    yaw_ref = params.get("yaw_ref_deg")
    yaw_ref = None if yaw_ref in (None, "") else float(yaw_ref)
    max_retries = int(params["max_retries"])
    backoff = float(params["backoff_sec"])
    run_mode = int(params.get("run_mode", 2))
    dry = bool(params.get("dry_run", True))
    poll_timeout = float(params.get("poll_timeout_sec", 240))

    RUN.set(loops_total=loops, loop=0, phase="starting", dry_run=dry,
            successes=0, failures=0, last_error_cm=None, last_press=None)
    RUN.log(f"START robot={robot} loops={loops} floors='{floors}' tol={tol_cm}cm "
            f"yawGate={'off' if yaw_tol is None else str(yaw_tol) + 'deg'} "
            f"maxRetries={max_retries} dry_run={dry}")

    area_id = elevator.get("areaId", "")
    successes = failures = 0

    for i in range(1, loops + 1):
        if RUN.stop_flag:
            RUN.log("STOP requested — ending loop"); break
        RUN.set(loop=i, phase="dispatching")
        pressed = False

        for attempt in range(0, max_retries + 1):
            if RUN.stop_flag:
                break
            tag = f"loop {i}/{loops} attempt {attempt+1}/{max_retries+1}"
            task_pts = [make_task_point(p, area_id) for p in route]
            name = f"ElevatorRun L{i} #{attempt+1}"

            if dry:
                RUN.log(f"{tag}: [DRY RUN] would create task with {len(task_pts)} pts "
                        f"ending at '{elevator['name']}'")
                # Simulate a good arrival in dry run so the UI flow is visible.
                err = 3.0
                RUN.set(phase="arrived(dry)", last_error_cm=round(err, 1))
                RUN.log(f"{tag}: [DRY RUN] simulated arrival error {err}cm -> would press '{floors}'")
                RUN.set(last_press="DRY RUN (not sent)")
                pressed = True
                break

            # --- real dispatch ---
            try:
                resp = AX.create_task(robot, task_pts, name, run_mode)
            except Exception as e:  # noqa: BLE001
                RUN.log(f"{tag}: create_task FAILED: {e}; backoff {backoff}s"); time.sleep(backoff); continue
            task_id = ((resp.get("data") or {}).get("taskId")
                       or resp.get("taskId") or (resp.get("data") or {}).get("id"))
            if not task_id:
                RUN.log(f"{tag}: no taskId in response: {resp}; backoff {backoff}s"); time.sleep(backoff); continue
            RUN.set(phase="navigating", task_id=task_id)
            RUN.log(f"{tag}: dispatched task {task_id}")

            # --- wait for the drive to FINISH, then press ---
            # The trigger has to be "the base has finished driving", and the task status
            # is the only source for that. Substituting the camera's "panel visible and
            # still" caused a collision: during final deceleration consecutive frames
            # fall inside the 2 mm stillness threshold, so the press locked coordinates
            # and moved while the robot was still travelling, and the arm went for where
            # the panel had been. Stillness over a short window is not arrival. The cloud
            # costs 20-55 s of latency and that is the price of a trustworthy signal; a
            # real fix needs a local arrival signal from the base, not a proxy for one.
            t0 = time.time()
            finished = cancelled = False
            while time.time() - t0 < poll_timeout:
                if RUN.stop_flag:
                    AX.cancel_task(task_id); RUN.log(f"{tag}: cancelled on stop"); break
                time.sleep(3)
                try:
                    st = AX.task_status(task_id).get("data") or {}
                except Exception as e:  # noqa: BLE001
                    RUN.log(f"{tag}: status poll error: {e}"); continue
                if task_finished(st):
                    finished = True; break
                if task_cancelled(st):
                    cancelled = True; break
            if RUN.stop_flag:
                break

            if not finished:
                why = "cancelled/obstacle" if cancelled else "timeout"
                RUN.log(f"{tag}: task did not finish ({why}). Backoff {backoff}s "
                        "then retry.")
                RUN.set(phase=f"retry({why})"); time.sleep(backoff); continue

            # --- arrived: gate on the measured position error ---
            RUN.set(phase="checking-arrival")
            state = AX.robot_state(robot)
            rx, ry = state.get("x"), state.get("y")
            err = dist_cm(rx, ry, elevator["x"], elevator["y"])
            yaw_err = None
            # ONLY an explicit yaw_ref_deg may gate on yaw. A POI's stored yaw is a
            # different convention from robot_state's: the elevator point reads 154.70
            # while the robot arrives at ~2.7 every time, physically facing the panel —
            # the orientation every successful press was made from. Comparing them
            # rejected a correct dock with a 152 deg "error".
            ref = yaw_ref
            if yaw_tol is not None and state.get("yaw") is not None and ref is not None:
                dy = abs(float(state["yaw"]) - float(ref)) % 360
                yaw_err = min(dy, 360 - dy)
            RUN.set(last_error_cm=None if err is None else round(err, 1),
                    last_yaw_err=None if yaw_err is None else round(yaw_err, 1))
            RUN.log(f"{tag}: arrived. position error={err}cm yaw_error={yaw_err}deg "
                    f"(gate {tol_cm}cm / {yaw_tol}deg)")

            if err is None:
                RUN.log(f"{tag}: no live position from robot — cannot verify; retry.")
                time.sleep(backoff); continue
            pos_bad = err > tol_cm
            yaw_bad = yaw_tol is not None and yaw_err is not None and yaw_err > yaw_tol
            if pos_bad or yaw_bad:
                # Usually a SYSTEMATIC docking offset that repeating will not fix, so
                # retry a bounded number of times then surface it. Name WHICH gate
                # failed: the old message always quoted the position numbers, so a yaw
                # rejection read as "OUT OF REACH (1.41cm > 8.0cm)", which is not even
                # arithmetically true and sent the diagnosis the wrong way.
                why = []
                if pos_bad:
                    why.append(f"position {err:.1f}cm > {tol_cm}cm")
                if yaw_bad:
                    why.append(f"yaw {yaw_err:.1f}deg > {yaw_tol}deg")
                RUN.log(f"{tag}: NOT PRESSING — {'; '.join(why)}")
                RUN.set(phase="out-of-reach"); time.sleep(backoff); continue

            # --- within reach: press ---
            RUN.set(phase="pressing")
            ok, out = press_on_agx(floors)
            RUN.set(last_press=out)
            if ok:
                RUN.log(f"{tag}: PRESS OK — {out}")
                pressed = True
                break
            RUN.log(f"{tag}: PRESS FAILED — {out}; retry.")
            time.sleep(backoff); continue

        if pressed:
            successes += 1
        else:
            failures += 1
            RUN.log(f"loop {i}/{loops}: FAILED after {max_retries+1} attempts — surfaced for operator")
        RUN.set(successes=successes, failures=failures)

    RUN.set(phase="stopped" if RUN.stop_flag else "done")
    RUN.log(f"END successes={successes} failures={failures}")
    RUN.active = False


def press_on_agx(floors):
    """Run the button press. Returns (ok, output_tail).

    Two modes, chosen by AGX_SSH:
      * "local" / "" / "localhost" -> the tool is RUNNING ON THE AGX; run
        press_buttons.py directly, no SSH. This is the on-robot deployment.
      * anything else              -> treat it as an ssh target/alias and run the
        press remotely over SSH (tool running on a laptop).
    """
    cmd = CFG["press_cmd"].format(floors=floors)
    if CFG["agx_ssh"] in ("", "local", "localhost"):
        full = ["bash", "-lc", cmd]            # local: tool is on the AGX
    else:
        full = ["ssh", CFG["agx_ssh"], cmd]    # remote: over SSH from a laptop
    try:
        # 240 s of --auto watching plus a ~51 s press already reaches 291 s, so
        # 300 left no margin at all.
        r = subprocess.run(full, capture_output=True, text=True, timeout=420)
    except subprocess.TimeoutExpired:
        return False, "press timed out (420s)"
    except Exception as e:  # noqa: BLE001
        return False, f"ssh error: {e}"
    tail = (r.stdout or "").strip().splitlines()[-3:]
    out = " | ".join(tail) if tail else (r.stderr or "").strip()[-200:]
    ok = r.returncode == 0 and "pressed" in (r.stdout or "").lower()
    return ok, out or f"(exit {r.returncode})"


# ---------------------------------------------------------------------------
# HTTP routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(HERE, "index.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "robot_serial": CFG["robot_serial"],
        "agx_ssh": CFG["agx_ssh"],
        "creds_configured": bool(CFG["app_id"] and CFG["app_secret"] and CFG["app_code"]),
        "elevator_poi_types": sorted(ELEVATOR_POI_TYPES),
    })


def _require_allowed_robot(requested):
    """Single-robot allowlist. The tool may ONLY ever touch CFG['robot_serial'].
    A blank/absent request defaults to it; any other serial is refused."""
    allowed = CFG["robot_serial"]
    if not allowed:
        return None, (jsonify({"error": "no robot serial configured in .env"}), 400)
    if requested and requested != allowed:
        return None, (jsonify({"error": "this tool is locked to a single robot; "
                               "that serial is not permitted"}), 403)
    return allowed, None


@app.route("/api/pois")
def api_pois():
    robot, err = _require_allowed_robot(request.args.get("robot"))
    if err:
        return err
    try:
        pois = AX.list_pois(robot)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 502
    for p in pois:
        p["isElevator"] = p["type"] in ELEVATOR_POI_TYPES
    return jsonify({"robot": robot, "pois": pois})


@app.route("/api/run", methods=["POST"])
def api_run():
    if RUN.active:
        return jsonify({"error": "a run is already active"}), 409
    body = request.get_json(force=True)
    robot, err = _require_allowed_robot(body.get("robot_serial"))
    if err:
        return err
    route = body.get("route") or []
    elevator = body.get("elevator")
    if not route or not elevator:
        return jsonify({"error": "need a route and an elevator point"}), 400
    if not body.get("dry_run", True) and not (CFG["app_id"] and CFG["app_code"]):
        return jsonify({"error": "AutoXing credentials not configured (config.json)"}), 400
    params = {
        "robot_serial": robot, "route": route, "elevator": elevator,
        "floors": (body.get("floors") or "").strip(),
        "loops": max(1, int(body.get("loops", 1))),
        "tolerance_cm": float(body.get("tolerance_cm", 8)),
        "yaw_tol_deg": float(body.get("yaw_tol_deg", 12)),
        "max_retries": max(0, int(body.get("max_retries", 3))),
        "backoff_sec": float(body.get("backoff_sec", 20)),
        "run_mode": int(body.get("run_mode", 2)),
        "dry_run": bool(body.get("dry_run", True)),
    }
    if not params["floors"] and not params["dry_run"]:
        return jsonify({"error": "no floors to press"}), 400
    RUN.thread = threading.Thread(target=run_loop, args=(params,), daemon=True)
    RUN.thread.start()
    return jsonify({"ok": True})


@app.route("/api/status")
def api_status():
    return jsonify(RUN.snapshot())


@app.route("/api/stop", methods=["POST"])
def api_stop():
    RUN.stop_flag = True
    RUN.log("STOP flag set by user")
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    print(f"Elevator Runner on http://127.0.0.1:{port}/  (robot={CFG['robot_serial']}, "
          f"AGX={CFG['agx_ssh']}, creds={'yes' if CFG['app_id'] else 'NO'})")
    app.run(host="127.0.0.1", port=port, threaded=True)
