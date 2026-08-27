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
import shlex
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
        # PRE-WARM. The press process spends ~7.3 s on startup that does not depend on
        # where the robot is: 2.8 s loading the TensorRT engine, 2.5 s opening the
        # camera, ~1.5 s ramping the GPU off its 306 MHz idle clock, 0.4 s connecting
        # to the arm. Started at DISPATCH instead of on arrival, all of it overlaps the
        # drive. It does NOT move the arrival decision into the press process — the
        # child blocks on a token file that only this loop writes, and only after the
        # cloud has confirmed the task finished AND the position gate has passed.
        # (Letting the press infer arrival from the camera is what drove the arm into
        # the panel once already; that mistake is not being repeated here.)
        # Set PREWARM=0 to fall back to the serial cold start.
        "prewarm": os.environ.get("PREWARM", "1") not in ("0", "false", "no"),
        "prewarm_token": os.environ.get("PREWARM_TOKEN", "/tmp/dex_press_go"),
        # Where each press's FULL output is kept. Only a three-line tail reaches the
        # UI, and when a press failed mid-sequence that tail was the camera's startup
        # banner, which says nothing about why. The press prints the chosen lift
        # height, joint margin, clearance and residual for every button; that is the
        # record of what the robot did, and throwing it away is how a failure becomes
        # undiagnosable after the fact.
        "press_log": os.environ.get("PRESS_LOG", "/tmp/dex_press_last.log"),
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


# Values robot_state.moveState takes when the base's own move controller considers
# the move over. Anything else (moving / running / idle / failed) is not arrival.
MOVE_DONE = {"succeeded", "success", "finished", "completed", "idle_succeeded"}


def base_arrived(st, dispatched_ms, elevator, tol_cm):
    """Has the BASE itself reported that it finished driving, at the elevator point?

    Why this exists: the navigation TASK status is 20-55 s behind reality (measured
    against three loops' endTime: 49 s, 20 s, 55 s), and that latency lands entirely
    between the robot parking and the arm pressing. robot_state is a level lower — it
    is the base's own move controller — and it is event-driven: parked, its timestamp
    goes 60 s without moving, but the report that FOLLOWS a change arrives ~2 s old.

    Four conditions, all required, because each covers a different way of being wrong:
      * timestamp after dispatch -> not a stale report from the previous run, which is
        the failure that matters most (moveState reads "succeeded" the whole time the
        robot sits parked from last time).
      * moveState done + speed 0  -> the base says it stopped, rather than us guessing
        from the outside. Inferring arrival from the camera's "panel still" is what put
        the arm into the panel once; this is the base's own word, not a proxy.
      * within tolerance of the elevator point -> filters intermediate route points,
        where a multi-point task also reports a completed move.

    Returns (arrived, err_cm, why) — `why` names the condition that is not met yet.
    """
    ts = st.get("timestamp")
    if not ts or ts <= dispatched_ms:
        return False, None, "stale report"
    if str(st.get("moveState", "")).lower() not in MOVE_DONE:
        return False, None, f"moveState={st.get('moveState')}"
    sp = st.get("speed")
    if sp is not None and abs(float(sp)) > 0.02:
        return False, None, f"speed={sp}"
    err = dist_cm(st.get("x"), st.get("y"), elevator["x"], elevator["y"])
    if err is None:
        return False, None, "no position"
    if err > tol_cm:
        return False, err, f"{err:.1f}cm from the point"
    return True, err, "arrived"


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


def wait_for_arrival(robot, task_id, target, tol_cm, poll_timeout,
                     fast_arrival, observe, tag):
    """Wait until the robot has arrived at `target`. Returns (finished, cancelled,
    fast, would_fire).

    ONE implementation, shared by the route and by the small repositioning moves
    between button presses — an arrival gate with two copies is an arrival gate with
    two behaviours.

    Two independent signals:
      * FAST  — the base's own moveState / speed / position. Measured side by side on
        one drive it beat the cloud task-status poll by 24 s (51 s vs 75 s), and the
        position it reported differed from the one taken 24 s later by 0.6 mm, i.e.
        the base really had stopped.
      * SLOW  — the cloud task status, kept as the backstop: it is what catches a task
        that was cancelled, or that ended somewhere the fast gate never accepts.
    """
    t0 = time.time()
    dispatched_ms = int(t0 * 1000)
    finished = cancelled = False
    fast = False
    confirms = 0                 # consecutive base reports that say "arrived"
    last_why = ""
    # The robot may ALREADY be parked at the elevator point when the route is
    # dispatched (it is, every loop after the first). Its stale "succeeded"
    # then satisfies every arrival condition at once, and we would press while
    # it is about to drive away. So arrival is only accepted after departure
    # has actually been observed.
    departed = False
    would_fire = None            # observe mode: when the fast gate first agreed
    prev_xy = None               # position at the previous agreeing report
    last_task_poll = 0.0
    while time.time() - t0 < poll_timeout:
        if RUN.stop_flag:
            AX.cancel_task(task_id); RUN.log(f"{tag}: cancelled on stop"); break
        time.sleep(1)

        # FAST PATH: the base's own move controller, ~2 s behind reality.
        if fast_arrival:
            try:
                rs = AX.robot_state(robot)
            except Exception:  # noqa: BLE001
                rs = {}
            ok_now, ferr, why = base_arrived(rs, dispatched_ms, target, tol_cm)
            if not departed:
                moving = (str(rs.get("moveState", "")).lower() not in MOVE_DONE
                          or abs(float(rs.get("speed") or 0)) > 0.02)
                away = False
                d = dist_cm(rs.get("x"), rs.get("y"),
                            target["x"], target["y"])
                if d is not None and d > max(tol_cm * 3, 50):
                    away = True
                if moving or away:
                    departed = True
                    RUN.log(f"{tag}: base has left the point "
                            f"({'moving' if moving else f'{d:.0f}cm away'})")
                ok_now = False
                why = "waiting for departure"
            if why != last_why:
                RUN.set(phase=f"navigating ({why})"); last_why = why
            # Two consecutive agreeing reports, AND the position must not have
            # drifted between them. Everything else here tests "the base says
            # it stopped"; this tests whether it actually did. A base still
            # creeping to its goal is the one way this signal can be early,
            # and it is exactly the failure that put the arm into the panel
            # when the camera was the trigger. Measured on the observe run:
            # 0.6 mm of drift over the 24 s between this signal and the
            # cloud's, so a 1 cm gate is loose against real settling and tight
            # against travel.
            xy = (rs.get("x"), rs.get("y"))
            if ok_now and confirms and prev_xy is not None:
                drift = dist_cm(xy[0], xy[1], prev_xy[0], prev_xy[1])
                if drift is not None and drift > 1.0:
                    RUN.log(f"{tag}: base still drifting ({drift:.1f}cm between "
                            "reports) — not arrival yet")
                    ok_now = False
            confirms = confirms + 1 if ok_now else 0
            prev_xy = xy if ok_now else None
            if confirms >= 2:
                elapsed = time.time() - t0
                if observe:
                    if would_fire is None:
                        would_fire = elapsed
                        RUN.log(f"{tag}: [observe] base reported arrival "
                                f"({ferr:.1f}cm) at {elapsed:.0f}s — still "
                                "waiting for the cloud, for comparison")
                else:
                    fast = finished = True
                    RUN.log(f"{tag}: base reports arrival ({ferr:.1f}cm) after "
                            f"{elapsed:.0f}s — pressing without waiting for "
                            "the cloud's task status")
                    break

        # BACKSTOP: the task status. Slower, but it is what catches a task
        # that was cancelled or that ended somewhere the fast path never
        # accepts, so it stays in the loop rather than being replaced.
        if time.time() - last_task_poll < 3:
            continue
        last_task_poll = time.time()
        try:
            st = AX.task_status(task_id).get("data") or {}
        except Exception as e:  # noqa: BLE001
            RUN.log(f"{tag}: status poll error: {e}"); continue
        if task_finished(st):
            slow = time.time() - t0
            RUN.log(f"{tag}: cloud task status says finished ({slow:.0f}s)")
            if would_fire is not None:
                RUN.log(f"{tag}: [observe] the base signal was {slow - would_fire:.0f}s "
                        f"earlier ({would_fire:.0f}s vs {slow:.0f}s)")
            finished = True; break
        if task_cancelled(st):
            cancelled = True; break

    return finished, cancelled, fast, would_fire


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
    # Corrective drives are separate from max_retries: a retry re-runs the whole route,
    # a corrective drive just closes the remaining distance to the elevator point.
    corrective = max(0, int(params.get("corrective_drives", 2)))
    backoff = float(params["backoff_sec"])
    run_mode = int(params.get("run_mode", 2))
    dry = bool(params.get("dry_run", True))
    poll_timeout = float(params.get("poll_timeout_sec", 240))
    # How arrival is decided:
    #   True      -> the base's own moveState (fast: the report after a change is ~2 s
    #                old, where the cloud's task status ran 20-55 s behind).
    #   "observe" -> compute the fast signal and LOG when it would have fired, but
    #                still wait for the cloud. One loop in this mode measures the real
    #                saving against the real cloud lag, on the same drive, at no risk.
    #   False     -> cloud task status only (the original behaviour).
    fast_arrival = params.get("fast_arrival", True)
    observe = str(fast_arrival).lower() == "observe"
    fast_arrival = observe or bool(fast_arrival)

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

            # Pay the press process's position-independent startup NOW, in parallel
            # with the drive. It blocks on a token file until this loop says go.
            warm = press_prewarm(floors)

            # --- wait for the drive to FINISH, then press ---
            # The trigger has to be "the base has finished driving", and the task status
            # is the only source for that. Substituting the camera's "panel visible and
            # still" caused a collision: during final deceleration consecutive frames
            # fall inside the 2 mm stillness threshold, so the press locked coordinates
            # and moved while the robot was still travelling, and the arm went for where
            # the panel had been. Stillness over a short window is not arrival. The cloud
            # costs 20-55 s of latency and that is the price of a trustworthy signal; a
            # real fix needs a local arrival signal from the base, not a proxy for one.
            finished, cancelled, fast, _ = wait_for_arrival(
                robot, task_id, elevator, tol_cm, poll_timeout,
                fast_arrival, observe, tag)
            if RUN.stop_flag:
                press_kill(warm)
                break

            if not finished:
                why = "cancelled/obstacle" if cancelled else "timeout"
                RUN.log(f"{tag}: task did not finish ({why}). Backoff {backoff}s "
                        "then retry.")
                RUN.set(phase=f"retry({why})"); press_kill(warm)
                time.sleep(backoff); continue

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
                press_kill(warm)
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
                RUN.log(f"{tag}: not close enough — {'; '.join(why)}")
                RUN.set(phase="out-of-reach")

                # Drive the rest of the way rather than stopping dead. An avoidance
                # manoeuvre can leave the robot short of the goal with the task still
                # reporting "completed" (measured: 73 cm short after someone stepped in
                # front of it), and simply re-dispatching closed the gap first time.
                corrected = False
                for c in range(1, corrective + 1):
                    RUN.log(f"{tag}: corrective drive {c}/{corrective} to "
                            f"'{elevator['name']}' ({err:.1f}cm to close)")
                    try:
                        r2 = AX.create_task(
                            robot, [make_task_point(elevator, area_id)],
                            f"{name} correct {c}", run_mode)
                    except Exception as e:  # noqa: BLE001
                        RUN.log(f"{tag}: corrective create_task failed: {e}")
                        break
                    tid2 = ((r2.get("data") or {}).get("taskId") or r2.get("taskId")
                            or (r2.get("data") or {}).get("id"))
                    if not tid2:
                        RUN.log(f"{tag}: corrective task got no id")
                        break
                    t2 = time.time()
                    while time.time() - t2 < poll_timeout:
                        if RUN.stop_flag:
                            AX.cancel_task(tid2); break
                        time.sleep(3)
                        try:
                            s2 = AX.task_status(tid2).get("data") or {}
                        except Exception:  # noqa: BLE001
                            continue
                        if task_finished(s2) or task_cancelled(s2):
                            break
                    if RUN.stop_flag:
                        break
                    stt = AX.robot_state(robot)
                    err = dist_cm(stt.get("x"), stt.get("y"),
                                  elevator["x"], elevator["y"])
                    RUN.set(last_error_cm=None if err is None else round(err, 1))
                    RUN.log(f"{tag}: after corrective {c}: position error={err}cm")
                    if err is not None and err <= tol_cm:
                        corrected = True
                        break
                if RUN.stop_flag:
                    press_kill(warm)
                    break
                if not corrected:
                    RUN.log(f"{tag}: still not within {tol_cm}cm after {corrective} "
                            "corrective drive(s) — surfacing rather than shuffling")
                    press_kill(warm)
                    time.sleep(backoff); continue
                RUN.log(f"{tag}: now within reach ({err:.1f}cm)")

            # --- within reach: press ---
            RUN.set(phase="pressing")
            ok, out = press_release(warm) if warm is not None else press_on_agx(floors)
            RUN.set(last_press=out)
            if fast:
                # We pressed before the cloud admitted the task was over, so the task
                # may still read "running". Let it settle before the next dispatch
                # rather than stacking tasks on top of each other. This costs nothing
                # in the normal case: the press already takes as long as the cloud's
                # worst measured lag.
                t_drain = time.time()
                while time.time() - t_drain < 60:
                    try:
                        stt = AX.task_status(task_id).get("data") or {}
                    except Exception:  # noqa: BLE001
                        break
                    if task_finished(stt) or task_cancelled(stt):
                        break
                    time.sleep(2)
                RUN.log(f"{tag}: cloud caught up {time.time() - t_drain:.0f}s after the press")
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


def _press_argv(cmd):
    """Wrap a press command for local (on-AGX) or remote (SSH) execution."""
    if CFG["agx_ssh"] in ("", "local", "localhost"):
        return ["bash", "-lc", cmd]
    return ["ssh", CFG["agx_ssh"], cmd]


def press_prewarm(floors):
    """Start the press process in pre-warm mode; return it once it reports READY.

    Returns None if pre-warm is disabled or the child failed to come up, in which
    case the caller falls back to the ordinary serial press.

    The child holds the camera and the arm connection open while it waits, so EVERY
    path that abandons an attempt has to call press_kill — otherwise the next
    attempt cannot open the camera and fails with "no color frame after 40 tries".
    """
    if not CFG.get("prewarm"):
        return None
    token = CFG["prewarm_token"]
    cmd = (CFG["press_cmd"].format(floors=floors)
           + f" --wait-go {shlex.quote(token)} --wait-go-timeout 900")
    try:
        proc = subprocess.Popen(_press_argv(cmd), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except Exception as e:  # noqa: BLE001
        RUN.log(f"prewarm: could not start ({e}) — serial press instead")
        return None
    proc._pre = []
    # Block until PREWARM_READY so a startup failure (camera busy, engine missing)
    # surfaces here, during the drive, instead of as a mystery stall after arrival.
    t0 = time.time()
    while time.time() - t0 < 90:
        line = proc.stdout.readline()
        if not line:
            break
        proc._pre.append(line.rstrip())
        if "PREWARM_READY" in line:
            RUN.log(f"prewarm: ready in {time.time() - t0:.1f}s — engine, camera and "
                    "GPU warm, waiting for the go signal")
            return proc
    tail = " | ".join(proc._pre[-3:])
    RUN.log(f"prewarm: did not come up ({tail or 'no output'}) — serial press instead")
    press_kill(proc)
    return None


def press_kill(proc):
    """Stop a pre-warm child so it releases the camera and the arm connection."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        try:
            proc.kill()
        except Exception:  # noqa: BLE001
            pass


def press_release(proc):
    """Write the go token, then wait for the pre-warmed press. Same contract as
    press_on_agx: (ok, output_tail)."""
    token = CFG["prewarm_token"]
    try:
        subprocess.run(_press_argv(f"touch {shlex.quote(token)}"), timeout=20, check=True)
    except Exception as e:  # noqa: BLE001
        press_kill(proc)
        return False, f"could not write the go token: {e}"
    lines = list(getattr(proc, "_pre", []))
    try:
        for line in proc.stdout:
            lines.append(line.rstrip())
        proc.wait(timeout=420)
    except Exception as e:  # noqa: BLE001
        press_kill(proc)
        return False, f"press did not finish: {e}"
    return _press_result(lines, proc.returncode)


def _press_result(lines, returncode):
    """Keep the whole press transcript on disk; return the short tail for the UI."""
    body = "\n".join(lines)
    try:
        Path(CFG["press_log"]).write_text(body + f"\n(exit {returncode})\n")
    except Exception:  # noqa: BLE001
        pass
    tail = " | ".join([ln for ln in lines if ln.strip()][-3:])
    ok = returncode == 0 and "pressed" in body.lower()
    return ok, tail or f"(exit {returncode})"


def press_on_agx(floors):
    """Run the button press with NO pre-warm (fallback). Returns (ok, output_tail).

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
    return _press_result(((r.stdout or "") + (r.stderr or "")).splitlines(),
                         r.returncode)


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
        # True | False | "observe" — see run_loop.
        "fast_arrival": body.get("fast_arrival", True),
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
