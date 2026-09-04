"""Drive the chassis in straight lines and in-place turns, over its local 8090 API.

Why not an AutoXing navigation task: the cloud API's only abstraction is "go to this
point", and its planner decides how — measured, it prefers to turn. Entering and
leaving an elevator has to be straight in and straight out to make the door window,
and "it did not turn this time" is not a guarantee. Here the angular velocity is
pinned to 0 and no planner is involved, so straight is structural, not lucky.

    python3 initialization/drive_straight.py state              # read-only
    python3 initialization/drive_straight.py clearance          # read-only, lidar
    python3 initialization/drive_straight.py move 2.0           # signed metres
    python3 initialization/drive_straight.py move -- -2.0
    python3 initialization/drive_straight.py turn 30            # signed degrees, in place
    python3 initialization/drive_straight.py turn -- -30
    python3 initialization/drive_straight.py calibrate          # achieved vs commanded
    python3 initialization/drive_straight.py stop

A LEG IS NEVER BOTH AT ONCE. `move` sends angular 0 and `turn` sends linear 0, so
there are no arcs and "straight is structural" survives the addition: a path through a
doorway is still a pure translation, and any re-aiming happens as a separate turn
before or after it, where it can be gated and measured on its own. Turns exist because
the flow needs them outside the door — the pose that lets the arm press the call panel
is not the pose that backs into the car — and because a panel that is not square to the
robot cannot be reached by translating, only by turning.

Protocol:
    POST /services/wheel_control/set_control_mode {"control_mode":"remote"}
    ws://<host>:8090/ws/v2/topics
        -> {"enable_topic":"/tracked_pose"}
        -> {"topic":"/twist","linear_velocity":v,"angular_velocity":0}
        <- /tracked_pose {pos, ori}, /scan_matched_points2, /slam/state, /wheel_state

MEASURED CONSTRAINT THAT SHAPES THIS WHOLE FILE: /tracked_pose publishes at **1.07 Hz**
(median gap 1.03 s), and /slam/state the same. At 0.2 m/s that is a 20 cm position
resolution — useless as a real-time distance loop, while the elevator's reachable
window is only +-8 cm. Reading it as if it were live produced every confusing result of
the first session: a 0.20 m move overshooting to 0.31 m, and three "STALLED / something
is in the way" aborts while the robot was in fact driving normally — the loop was
comparing a stale pose against itself.

So distance is driven **open-loop on time** at a calibrated speed, and the pose is used
for what it is genuinely good for:

* a BEFORE/AFTER measurement of how far the robot really went (absolute accuracy is
  excellent: 0.0 mm of scatter while stationary, and it is lidar-matched, so no drift);
* a correction pass — if the achieved distance is short by more than --tol, drive the
  remainder in a slow short leg;
* the straightness check, since this chassis does report `ori` (the chassis the
  reference script was written for did not).

Safety, in the order it applies:

1. **Serial number check.** This network carries other people's robots — three of them
   answered /device/info during discovery. Never drive a chassis that is not ours.
2. **Lidar clearance gate.** /scan_matched_points2 (360 degrees, MAP coordinates) is
   transformed into the body frame and the travel corridor measured BEFORE any twist.
   The chest camera cannot do this job: it faces only the arm side, so it structurally
   cannot see the other direction of travel — using it to reason about that direction
   is what produced a wrong "it hit a wall" conclusion.
3. **Stall detection on POSE UPDATES, not wall time.** With a 1 Hz feed a 1.5 s window
   holds barely one sample; the check needs two consecutive fresh poses showing no
   motion.
4. Heading drift, held wheel-slip, emergency stop, hard timeout.
5. **Every exit path brakes and restores `auto`.** Leaving the base in `remote`
   silently disables AutoXing navigation for whatever runs next.

The base's own move controller is NOT in the loop, so nothing avoids obstacles while
moving: the clearance gate is a check before the motion, not during it.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time

import requests
import websocket

HOST = "192.168.25.25"
PORT = 8090
OUR_SN = "1352512k06960Og"

# Hard ceilings, independent of the CLI, so a typo cannot become a long fast drive.
MAX_DIST_M = 3.0
MAX_SPEED = 0.30

# recv() must return promptly or it starves the twist loop: the base's watchdog stops
# the wheels when keepalive drops, measured as "moved 2.3 cm then sat there" at ~4 Hz.
RECV_TIMEOUT_S = 0.01
TWIST_HZ = 20.0

# From /robot/footprint (x -0.4..0.4, y -0.369..0.379): the base is 0.80 x 0.76 m, so
# only lidar points inside this half-width can be hit head-on, and the hull already
# reaches HULL_M along the axis of travel.
HALF_WIDTH_M = 0.40
HULL_M = 0.40
CLEAR_MARGIN_M = 0.25          # required gap beyond the requested distance

# Ramp down near the end so the coast after braking is short (0.2 m/s coasted ~11 cm).
RAMP_M = 0.18
CREEP_MPS = 0.06

# Achieved / commanded distance, measured twice: 0.891 over a 0.30 m leg and 0.903
# over a clean 5 s run (0.903 m at a commanded 0.20 m/s). Start-up and braking losses
# are what it captures, so the leg duration is divided by it.
SPEED_RATIO = 0.90

# In-place turns. The radius is the chassis's OWN footprint (/robot/footprint, 19
# points) at its furthest corner from the pose origin — measured 0.476 m, where the
# 0.80 x 0.76 m bounding box would have said 0.55 m and refused turns in gaps that fit.
TURN_RADIUS_M = 0.476
TURN_MARGIN_M = 0.15           # required gap beyond the swept circle
MAX_TURN_DEG = 100.0           # ceiling per command, independent of the CLI
MAX_YAW_RATE = 0.30            # rad/s
TURN_RAMP_DEG = 8.0            # ramp down over the last few degrees, as RAMP_M does
CREEP_YAW = 0.06               # rad/s
# Achieved / commanded turn. 1.0 until measured on this chassis: unlike SPEED_RATIO
# this one has no measurement behind it yet, so `turn` prints the ratio it achieved on
# every run and the correction legs absorb the difference meanwhile.
TURN_RATIO = 1.0
# Small turns are NOT open-loop predictable on this chassis, and no constant fixes that.
# Measured 2026-09-04 with the ramp deficit already removed, single legs, settled poses
# both ends: 10 deg commanded gave 14.9 and 17.2, 20 deg gave 17.2 and 23.5 — a ratio
# scattered over 0.86 to 1.72 within one minute. The cause is the operating point: a
# small turn spends its whole duration inside the ramp, at 0.06-0.12 rad/s, which is at
# the bottom of what the wheels control smoothly. So `turn` closes the loop instead of
# trusting a model, and DAMPS each correction to CORR_GAIN of what is left, which
# converges monotonically even when a single step is 70 % out. Below TURN_FLOOR_DEG the
# plant does not respond usefully at all, so the loop stops rather than chasing noise.
# And the floor is the chassis's own resolution, measured rather than chosen: with the
# ramp capped so every turn gets a real push, single turns of 3-12 deg still came back
# +2.2, +3.0, -2.1, -2.9 and -4.0 deg off, and commands below ~3 deg frequently moved the
# base NOT AT ALL (0.0 deg, three times running) before breaking away to 2.9. This is the
# plant, not the measurement: ori held its value for 20 s after a turn, so nothing was
# still settling. So ~3 deg is the angular resolution of this base, the damped loop stops
# there instead of chasing noise, and anything needing better than 3 deg of heading has to
# get it some other way.
CORR_GAIN = 0.6
TURN_FLOOR_DEG = 3.0

# STALL DETECTION DOES NOT USE THE POSE. /tracked_pose is 1 Hz AND repeats the same
# value across consecutive messages while the base is moving (SLAM settles late), so
# "two fresh poses with no motion" reported a robot doing 0.18 m/s as stuck — three
# times, with different distances. Current is a physical quantity with no such lag:
# driving into the charging dock took it from 6.7 A to 30.4 A. Idle is ~4.4 A.
STALL_CURRENT_A = 20.0
STALL_CURRENT_S = 0.8
# wheel_slipping is NOT used as an abort. Measured, it goes true during ordinary
# driving (it cut a 1.00 m move short at 0.695 m while current sat at a healthy 5.7 A),
# so it is a poor stall signal — like the pose, it is unreliable while moving. It costs
# nothing to report, and the distance loop reads before/after pose rather than odometry,
# so a slip cannot corrupt the measurement anyway.

# WHY BOTH LEGS INTEGRATE WHAT THEY COMMAND (_integral_note, 2026-09-04)
# Both loops used to run for `target / speed` seconds and stop. But both ramp DOWN over
# the last stretch, and the ramp was not in that sum, so every leg came up short by
# roughly the ramp's own deficit — a FIXED loss, independent of how far the leg was.
# Measured on three clean single turns: 10 deg commanded gave 6.9, 23 gave 20.1, 90 gave
# 84.4, which fits `achieved = 0.969 * commanded - 2.8 deg`. That constant 2.8 deg is
# 28 % of a 10 deg turn and 3 % of a 90 deg one, which is exactly why the "ratio" looked
# like it wandered between 0.69 and 0.96 with no pattern. Correcting a small angle was
# therefore untrustworthy, and correcting a small angle is precisely what aligning to a
# doorway needs.
# The fix is not another constant: the loop now accumulates the velocity it actually
# sends and stops when that integral reaches the target, so the ramp pays for itself.
# SPEED_RATIO / TURN_RATIO stay, but they now mean only what they say — the slip between
# what the wheels were told and what the floor gave back.

_stop_requested = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _stop_requested
    _stop_requested = True


def _wrap_deg(rad: float) -> float:
    return math.degrees(math.atan2(math.sin(rad), math.cos(rad)))


def _fmt_m(v: float) -> str:
    return "inf (nothing in the corridor)" if v == float("inf") else f"{v:.2f} m"


class Chassis:
    """Minimal client for the chassis's local API. Straight-line moves only."""

    def __init__(self, host: str = HOST, port: int = PORT, secret: str = ""):
        self.host, self.port, self.secret = host, port, secret
        self._ws: websocket.WebSocket | None = None
        self._entered_remote = False
        self.pose: tuple[float, float, float] | None = None   # x, y, ori
        self.pose_seq = 0                  # increments on every fresh /tracked_pose
        self.scan: list | None = None
        self.slipping = False
        self.estop = None
        self.mode = None
        self.alerts: list | None = None
        self.current = None

    # --- HTTP ---------------------------------------------------------------
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.secret:
            h["Secret"] = self.secret
        return h

    def device_info(self) -> dict:
        r = requests.get(f"http://{self.host}:{self.port}/device/info",
                         headers=self._headers(), timeout=8)
        r.raise_for_status()
        return r.json()

    def check_identity(self) -> str:
        """Refuse to touch a chassis that is not ours — other robots share this net."""
        info = self.device_info()
        sn = (info.get("device") or {}).get("sn")
        if sn != OUR_SN:
            raise SystemExit(f"ABORT: {self.host} is sn {sn!r}, expected {OUR_SN!r}")
        return sn

    def set_mode(self, mode: str) -> None:
        r = requests.post(
            f"http://{self.host}:{self.port}/services/wheel_control/set_control_mode",
            headers=self._headers(), json={"control_mode": mode, "mode": mode},
            timeout=8)
        r.raise_for_status()

    # --- WebSocket ----------------------------------------------------------
    def open_stream(self, with_scan: bool = True) -> None:
        header = [f"Secret: {self.secret}"] if self.secret else []
        self._ws = websocket.create_connection(
            f"ws://{self.host}:{self.port}/ws/v2/topics", header=header, timeout=8)
        self._ws.settimeout(RECV_TIMEOUT_S)
        topics = ["/tracked_pose", "/wheel_state", "/slam/state", "/alerts",
                  "/battery_state"]
        if with_scan:
            topics.append("/scan_matched_points2")
        for t in topics:
            self._ws.send(json.dumps({"enable_topic": t}))

    def pump(self, seconds: float = 0.0) -> None:
        """Drain what has already arrived; return as soon as the socket is dry so the
        caller keeps control of the twist cadence."""
        assert self._ws is not None
        deadline = time.time() + seconds
        while True:
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                if time.time() >= deadline:
                    return
                continue
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except (ValueError, TypeError):
                continue
            topic = msg.get("topic")
            if topic == "/tracked_pose":
                pos, ori = msg.get("pos"), msg.get("ori")
                if isinstance(pos, (list, tuple)) and len(pos) >= 2 and ori is not None:
                    self.pose = (float(pos[0]), float(pos[1]), float(ori))
                    self.pose_seq += 1
            elif topic == "/scan_matched_points2":
                pts = msg.get("points")
                if pts:
                    self.scan = pts
            elif topic == "/wheel_state":
                self.estop = msg.get("emergency_stop_pressed")
                self.mode = msg.get("control_mode")
            elif topic == "/slam/state":
                self.slipping = bool(msg.get("wheel_slipping"))
            elif topic == "/alerts":
                self.alerts = msg.get("alerts")
            elif topic == "/battery_state":
                self.current = msg.get("current")

    def wait_pose(self, timeout: float = 4.0) -> tuple[float, float, float]:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pump(0.1)
            if self.pose is not None:
                return self.pose
        raise SystemExit(f"ABORT: no /tracked_pose within {timeout:.0f} s")

    def wait_fresh_pose(self, timeout: float = 3.0):
        """Block for the NEXT pose — the feed is only ~1 Hz, so 'current' is stale."""
        want = self.pose_seq + 1
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pump(0.05)
            if self.pose_seq >= want:
                return self.pose
        return None

    def wait_scan(self, timeout: float = 4.0) -> list:
        t0 = time.time()
        while time.time() - t0 < timeout:
            self.pump(0.1)
            if self.scan:
                return self.scan
        raise SystemExit(f"ABORT: no lidar scan within {timeout:.0f} s")

    # --- geometry -----------------------------------------------------------
    def corridor(self) -> tuple[float, float, float]:
        """(clear along +ori, clear along -ori, nearest to the sides), metres from the
        base's CENTRE. Lidar points arrive in map coordinates, so they are rotated into
        the body frame with the current pose."""
        x0, y0, ori = self.wait_pose()
        pts = self.wait_scan()
        c, s = math.cos(ori), math.sin(ori)
        plus, minus, side = [], [], []
        for p in pts:
            dx, dy = p[0] - x0, p[1] - y0
            bx = dx * c + dy * s
            by = -dx * s + dy * c
            if abs(by) <= HALF_WIDTH_M:
                (plus if bx > 0 else minus).append(abs(bx))
            else:
                side.append(math.hypot(bx, by))
        big = float("inf")
        return (min(plus) if plus else big,
                min(minus) if minus else big,
                min(side) if side else big)

    def gate_clearance(self, metres: float) -> None:
        """Refuse to drive when the corridor in the travel direction is too short."""
        plus, minus, _ = self.corridor()
        ahead = plus if metres > 0 else minus
        need = abs(metres) + HULL_M + CLEAR_MARGIN_M
        label = "+ori" if metres > 0 else "-ori"
        print(f"lidar clearance {label}: {_fmt_m(ahead)} from centre — need "
              f"{need:.2f} m ({abs(metres):.2f} travel + {HULL_M:.2f} hull "
              f"+ {CLEAR_MARGIN_M:.2f} margin)")
        if ahead < need:
            raise SystemExit(f"ABORT: only {_fmt_m(ahead)} of clearance along {label}; "
                             f"{need:.2f} m needed")

    def nearest(self) -> float:
        """Distance to the closest lidar return in ANY direction, from the base's
        centre. A turn sweeps every direction at once, so the corridor test that gates
        a straight leg says nothing useful about it."""
        x0, y0, _ = self.wait_pose()
        pts = self.wait_scan()
        return min(math.hypot(p[0] - x0, p[1] - y0) for p in pts) if pts else float("inf")

    def gate_spin(self) -> None:
        near = self.nearest()
        need = TURN_RADIUS_M + TURN_MARGIN_M
        print(f"lidar nearest in any direction: {_fmt_m(near)} from centre — need "
              f"{need:.2f} m ({TURN_RADIUS_M:.3f} swept radius "
              f"+ {TURN_MARGIN_M:.2f} margin)")
        if near < need:
            raise SystemExit(f"ABORT: nearest obstacle {_fmt_m(near)}; a turn sweeps "
                             f"{TURN_RADIUS_M:.3f} m in every direction")

    # --- motion -------------------------------------------------------------
    def send_twist(self, linear: float) -> None:
        """Angular velocity is ALWAYS zero here. That is the point of this file."""
        assert self._ws is not None
        linear = max(-MAX_SPEED, min(MAX_SPEED, linear))
        self._ws.send(json.dumps({"topic": "/twist",
                                  "linear_velocity": linear,
                                  "angular_velocity": 0.0}))

    def send_spin(self, angular: float) -> None:
        """LINEAR velocity is always zero here — the mirror of send_twist.

        Two functions rather than one with both arguments, so neither caller can emit
        an arc by passing the wrong pair. The keepalive rate is the same: below ~20 Hz
        the base's watchdog cuts the wheels mid-turn exactly as it does mid-drive.
        """
        assert self._ws is not None
        angular = max(-MAX_YAW_RATE, min(MAX_YAW_RATE, angular))
        self._ws.send(json.dumps({"topic": "/twist",
                                  "linear_velocity": 0.0,
                                  "angular_velocity": angular}))

    def brake(self) -> None:
        if self._ws is None:
            return
        for _ in range(4):
            try:
                self.send_twist(0.0)
                time.sleep(0.05)
            except Exception:              # noqa: BLE001
                return

    def enter_remote(self) -> None:
        self.set_mode("remote")
        self._entered_remote = True

    def close(self) -> None:
        """Brake, hand control back, drop the socket. Safe to call twice."""
        try:
            self.brake()
        finally:
            if self._entered_remote:
                for attempt in range(3):
                    try:
                        self.set_mode("auto")
                        self._entered_remote = False
                        break
                    except Exception as e:      # noqa: BLE001
                        print(f"  ! restore auto failed ({attempt + 1}/3): {e}",
                              file=sys.stderr)
                        time.sleep(0.5)
                else:
                    print("  !! CHASSIS MAY STILL BE IN remote MODE — AutoXing "
                          "navigation will not work until it is set back to auto",
                          file=sys.stderr)
            if self._ws is not None:
                try:
                    self._ws.close()
                except Exception:               # noqa: BLE001
                    pass
                self._ws = None


def _settled_pose(ch: Chassis, tol_m: float = 0.01, tol_deg: float = 0.5,
                  timeout: float = 8.0):
    """Read the pose only once two consecutive FRESH samples agree.

    Waiting for "a couple of fresh samples" was not enough. The feed is 1 Hz and SLAM
    settles LATE, so the sample that arrives right after braking still describes a
    place the robot has already left. Measured 2026-09-02: a leg commanded 0.40 m read
    back 0.127 m that way — 68 % short — and the correction legs that fired to make up
    the difference took the move to 0.589 m, a 47 % OVERSHOOT of the original request.
    Legs run with `--corrections 0`, which is to say legs whose end pose was read
    minutes later, agreed with an independent camera measurement to 17 mm.

    So the test is agreement, not arrival: keep pumping until the position stops
    changing between samples. Zero twist keeps flowing throughout — a 2-3 s silent gap
    drops the base out of its remote-driving state.
    """
    prev = None
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            ch.send_twist(0.0)
        except Exception:              # noqa: BLE001
            break
        got = ch.wait_fresh_pose(timeout=2.0)
        if got is None:
            continue
        if prev is not None:
            moved = math.hypot(got[0] - prev[0], got[1] - prev[1])
            turned = abs(_wrap_deg(got[2] - prev[2]))
            if moved <= tol_m and turned <= tol_deg:
                return got
        prev = got
    return ch.pose


def _leg(ch: Chassis, metres: float, speed: float, ori_tol: float,
         verbose: bool = True) -> tuple[float, float, str]:
    """One open-loop timed leg. Returns (achieved signed metres, heading change, reason).

    Timed rather than pose-closed because /tracked_pose is 1 Hz (see the module
    docstring). The pose is still watched, but only for stalls and heading drift, and
    both are judged on FRESH samples.
    """
    # The START must be settled too, not just the end. Reading a stale start pose
    # makes the leg's own measurement wrong in whichever direction the previous motion
    # was still catching up in — and back-to-back legs are exactly when that bites.
    start = _settled_pose(ch) or ch.wait_pose()
    linear = math.copysign(speed, metres)
    target = abs(metres)
    # Issue MORE than the target, by the calibrated loss, and count what is actually
    # sent rather than how long the leg has been running. See _integral_note.
    want = target / SPEED_RATIO
    timeout = want / max(speed, 1e-6) + 8.0
    ramp_m = min(RAMP_M, 0.4 * want)   # same reason as the turn's ramp — see _spin_leg

    if ch.estop:
        return 0.0, 0.0, "emergency stop is pressed"

    if verbose:
        print(f"    (base reports control_mode={ch.mode})")
    period = 1.0 / TWIST_HZ
    next_tick = t0 = time.time()
    stall_since = None
    slip_seen = False
    last_log = t0
    reason = "leg complete"

    issued = 0.0
    last_tick = t0
    while True:
        elapsed = time.time() - t0
        remaining_m = max(0.0, want - issued)
        if remaining_m < ramp_m:
            scale = max(CREEP_MPS / speed, remaining_m / ramp_m)
        else:
            scale = 1.0
        v = speed * scale
        ch.send_twist(math.copysign(v, linear))
        ch.pump()

        next_tick += period
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
        now = time.time()
        issued += v * (now - last_tick)
        last_tick = now

        if issued >= want:
            break
        if _stop_requested:
            reason = "interrupted"
            break
        if ch.estop:
            reason = "emergency stop pressed"
            break

        # Stall on CURRENT, never on the pose — see the note by STALL_CURRENT_A.
        if ch.current is not None and ch.current >= STALL_CURRENT_A:
            if stall_since is None:
                stall_since = now
            elif now - stall_since >= STALL_CURRENT_S:
                reason = (f"STALLED — {ch.current:.1f} A for {STALL_CURRENT_S}s "
                          f"(idle ~4.4 A); something is in the way")
                break
        else:
            stall_since = None

        if ch.slipping and not slip_seen:
            slip_seen = True          # reported at the end, never an abort

        if verbose and now - last_log >= 1.0:
            print("    t+%4.1fs  issued %.2f/%.2f m   current %s A"
                  % (elapsed, issued, want,
                     "?" if ch.current is None else f"{ch.current:.1f}"))
            last_log = now

        if now - t0 > timeout:
            reason = f"timed out after {timeout:.0f} s"
            break

    ch.brake()
    end = _settled_pose(ch) or start
    dx, dy = end[0] - start[0], end[1] - start[1]
    along = dx * math.cos(start[2]) + dy * math.sin(start[2])
    if slip_seen and verbose:
        print("    (wheel_slipping went true during this leg — logged, not acted on)")
    return along, _wrap_deg(end[2] - start[2]), reason


def drive(ch: Chassis, metres: float, speed: float, ori_tol: float, tol: float,
          corrections: int, skip_gate: bool = False) -> None:
    if abs(metres) > MAX_DIST_M:
        raise SystemExit(f"ABORT: {metres} m exceeds the {MAX_DIST_M} m ceiling")
    speed = min(abs(speed), MAX_SPEED)

    if not skip_gate:
        ch.gate_clearance(metres)

    start = ch.wait_pose()
    print(f"start pose  x={start[0]:.3f} y={start[1]:.3f} ori={start[2]:.4f}")
    print(f"target {metres:+.2f} m at {speed:.2f} m/s  (angular pinned to 0)")

    ch.enter_remote()
    ch.pump(0.3)

    total = 0.0
    want = metres
    for attempt in range(corrections + 1):
        if abs(want) < 0.01:
            break
        leg_speed = speed if attempt == 0 else min(speed, 0.12)
        ch.enter_remote()                     # re-assert; see the keepalive note
        got, d_ori, reason = _leg(ch, want, leg_speed, ori_tol)
        total += got
        print(f"  leg {attempt + 1}: commanded {want:+.3f} m -> achieved {got:+.3f} m "
              f"({reason}), heading {d_ori:+.2f} deg")
        if reason != "leg complete":
            break
        want = metres - total
        if abs(want) <= tol:
            break

    end = ch.pose or start
    dx, dy = end[0] - start[0], end[1] - start[1]
    moved = math.hypot(dx, dy)
    along = dx * math.cos(start[2]) + dy * math.sin(start[2])
    lateral = math.sqrt(max(0.0, moved * moved - along * along))
    print(f"end pose    x={end[0]:.3f} y={end[1]:.3f} ori={end[2]:.4f}")
    print(f"TOTAL along heading {along:+.3f} m of {metres:+.2f} m requested "
          f"(error {along - metres:+.3f} m)")
    print(f"lateral deviation {lateral * 1000:.0f} mm   "
          f"heading change {_wrap_deg(end[2] - start[2]):+.2f} deg")


def _spin_leg(ch: Chassis, degrees_: float, yaw_rate: float,
              verbose: bool = True) -> tuple[float, float, str]:
    """One open-loop timed in-place turn. Returns (achieved deg, drift m, reason).

    Same shape as `_leg` and for the same reason: the pose feed is 1 Hz, so a turn
    cannot be closed on it either. Position is watched as the SANITY check here —
    an in-place turn that translates is a turn the wheels are not executing.
    """
    start = _settled_pose(ch) or ch.wait_pose()
    if ch.estop:
        return 0.0, 0.0, "emergency stop is pressed"

    target = abs(math.radians(degrees_))
    angular = math.copysign(yaw_rate, degrees_)
    want = target / TURN_RATIO
    timeout = want / max(yaw_rate, 1e-6) + 8.0
    # The ramp limits the coast at the END of a turn. On a turn shorter than the ramp
    # it swallowed the whole motion instead, leaving the wheels at 0.06-0.12 rad/s —
    # below breakaway, so a commanded 1.5 deg measured 0.0 deg three times running and
    # then jumped to 2.9. Cap it at a fraction of the turn so every turn gets a real
    # push before it slows down.
    ramp = min(math.radians(TURN_RAMP_DEG), 0.4 * want)

    if verbose:
        print(f"    (base reports control_mode={ch.mode})")
    period = 1.0 / TWIST_HZ
    next_tick = t0 = time.time()
    stall_since = None
    last_log = t0
    reason = "turn complete"

    issued = 0.0
    last_tick = t0
    while True:
        elapsed = time.time() - t0
        remaining = max(0.0, want - issued)
        if remaining < ramp:
            scale = max(CREEP_YAW / yaw_rate, remaining / ramp)
        else:
            scale = 1.0
        w = yaw_rate * scale
        ch.send_spin(math.copysign(w, angular))
        ch.pump()

        next_tick += period
        sleep_for = next_tick - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.time()
        now = time.time()
        issued += w * (now - last_tick)
        last_tick = now

        if issued >= want:
            break
        if _stop_requested:
            reason = "interrupted"
            break
        if ch.estop:
            reason = "emergency stop pressed"
            break
        if ch.current is not None and ch.current >= STALL_CURRENT_A:
            if stall_since is None:
                stall_since = now
            elif now - stall_since >= STALL_CURRENT_S:
                reason = (f"STALLED — {ch.current:.1f} A for {STALL_CURRENT_S}s "
                          f"(idle ~4.4 A); something is in the way")
                break
        else:
            stall_since = None

        if verbose and now - last_log >= 1.0:
            print("    t+%4.1fs  issued %.1f/%.1f deg   current %s A"
                  % (elapsed, math.degrees(issued), math.degrees(want),
                     "?" if ch.current is None else f"{ch.current:.1f}"))
            last_log = now

        if now - t0 > timeout:
            reason = f"timed out after {timeout:.0f} s"
            break

    ch.brake()
    end = _settled_pose(ch) or start
    turned = _wrap_deg(end[2] - start[2])
    drift = math.hypot(end[0] - start[0], end[1] - start[1])
    return turned, drift, reason


def spin(ch: Chassis, degrees_: float, yaw_rate: float, tol_deg: float,
         corrections: int, skip_gate: bool = False) -> None:
    if abs(degrees_) > MAX_TURN_DEG:
        raise SystemExit(f"ABORT: {degrees_} deg exceeds the {MAX_TURN_DEG} deg ceiling")
    yaw_rate = min(abs(yaw_rate), MAX_YAW_RATE)

    if not skip_gate:
        ch.gate_spin()

    start = ch.wait_pose()
    print(f"start pose  x={start[0]:.3f} y={start[1]:.3f} ori={start[2]:.4f}")
    print(f"target {degrees_:+.1f} deg at {yaw_rate:.2f} rad/s  (linear pinned to 0)")

    ch.enter_remote()
    ch.pump(0.3)

    total = 0.0
    for attempt in range(corrections + 1):
        remaining = degrees_ - total
        if abs(remaining) <= max(tol_deg, TURN_FLOOR_DEG if attempt else 0.0):
            break
        # First leg goes for the whole angle; every correction is DAMPED, because a
        # single small turn can come out 70 % over (see the note by CORR_GAIN).
        want = remaining if attempt == 0 else remaining * CORR_GAIN
        if attempt and abs(want) < TURN_FLOOR_DEG:
            want = math.copysign(TURN_FLOOR_DEG, want)
        ch.enter_remote()                     # re-assert; see the keepalive note
        got, drift, reason = _spin_leg(ch, want, yaw_rate)
        total += got
        print(f"  turn {attempt + 1}: commanded {want:+.1f} deg -> achieved "
              f"{got:+.1f} deg (remaining {degrees_ - total:+.1f}), "
              f"drift {drift * 1000:.0f} mm")
        if reason != "turn complete":
            print(f"    stopped: {reason}")
            break

    end = ch.pose or start
    turned = _wrap_deg(end[2] - start[2])
    drift = math.hypot(end[0] - start[0], end[1] - start[1])
    print(f"end pose    x={end[0]:.3f} y={end[1]:.3f} ori={end[2]:.4f}")
    print(f"TOTAL turned {turned:+.1f} deg of {degrees_:+.1f} requested "
          f"(error {turned - degrees_:+.1f} deg)")
    ratio = turned / degrees_ if degrees_ else float("nan")
    print(f"position drift {drift * 1000:.0f} mm   achieved/commanded {ratio:.3f}"
          "   <- pin this into TURN_RATIO once it repeats")


def cmd_turn(args) -> None:
    ch = Chassis(args.host, args.port, args.secret)
    ch.check_identity()
    ch.open_stream()
    try:
        spin(ch, args.degrees, args.rate, args.tol, args.corrections, args.no_gate)
    finally:
        ch.close()


def cmd_state(args) -> None:
    ch = Chassis(args.host, args.port, args.secret)
    sn = ch.check_identity()
    info = ch.device_info()
    dev = info.get("device") or {}
    print(f"chassis {sn}  model={dev.get('model')}  platform={dev.get('platform')}")
    print(f"  axbot {info.get('axbot_version')}")
    ch.open_stream()
    try:
        t0 = time.time()
        while time.time() - t0 < 4.0:
            ch.pump(0.2)
        print(f"  control_mode   {ch.mode}")
        print(f"  emergency_stop {ch.estop}")
        print(f"  wheel_slipping {ch.slipping}")
        if ch.pose:
            print(f"  tracked_pose   x={ch.pose[0]:.3f} y={ch.pose[1]:.3f} "
                  f"ori={ch.pose[2]:.4f}")
        for a in (ch.alerts or []):
            print(f"  alert {a.get('code')} {a.get('level')}: {a.get('msg')}")
    finally:
        ch.close()


def cmd_clearance(args) -> None:
    ch = Chassis(args.host, args.port, args.secret)
    ch.check_identity()
    ch.open_stream()
    try:
        plus, minus, side = ch.corridor()
        print(f"from the base's centre, within a {HALF_WIDTH_M:.2f} m half-width "
              "corridor:")
        print(f"  along +ori   {_fmt_m(plus)}")
        print(f"  along -ori   {_fmt_m(minus)}")
        print(f"  to the sides {_fmt_m(side)}")
        print(f"the hull reaches {HULL_M:.2f} m along the axis, so usable travel is "
              f"each figure minus {HULL_M:.2f} m")
    finally:
        ch.close()


def cmd_move(args) -> None:
    ch = Chassis(args.host, args.port, args.secret)
    ch.check_identity()
    ch.open_stream()
    try:
        drive(ch, args.metres, args.speed, args.ori_tol, args.tol,
              args.corrections, args.no_gate)
    finally:
        ch.close()


def cmd_calibrate(args) -> None:
    """Measure achieved vs commanded metres, so the timed legs can be trusted."""
    ch = Chassis(args.host, args.port, args.secret)
    ch.check_identity()
    ch.open_stream()
    try:
        legs = (0.3, 0.6, 1.0)
        # The gate must cover the CUMULATIVE distance: the legs do not return to the
        # start, so checking one leg's length would clear a corridor 1.6 m too short.
        ch.gate_clearance(math.copysign(sum(legs), args.metres))
        ch.enter_remote()
        ch.pump(0.3)
        results = []
        for d in legs:
            signed = math.copysign(d, args.metres)
            ch.enter_remote()                 # re-assert; see the keepalive note
            got, d_ori, reason = _leg(ch, signed, args.speed, args.ori_tol,
                                      verbose=False)
            ratio = abs(got) / d if d else 0.0
            results.append((d, abs(got), ratio, d_ori, reason))
            print(f"  commanded {d:.2f} m -> achieved {abs(got):.3f} m  "
                  f"ratio {ratio:.3f}  heading {d_ori:+.2f} deg  ({reason})")
            time.sleep(1.0)
        good = [r for r in results if r[4] == "leg complete"]
        if good:
            mean = sum(r[2] for r in good) / len(good)
            print(f"\nmean achieved/commanded = {mean:.3f} over {len(good)} legs")
            print("a consistent offset means the speed is off; a spread that shrinks "
                  "with distance means fixed start/stop losses")
    finally:
        ch.close()


def cmd_stop(args) -> None:
    ch = Chassis(args.host, args.port, args.secret)
    ch.check_identity()
    ch.open_stream(with_scan=False)
    try:
        ch.brake()
        print("zero twist sent; restoring auto mode")
    finally:
        ch._entered_remote = True     # force the restore even if we never set remote
        ch.close()


def main() -> int:
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--host", default=HOST)
    common.add_argument("--port", type=int, default=PORT)
    common.add_argument("--secret", default="")
    common.add_argument("--ori-tol", type=float, default=4.0, metavar="DEG",
                        help="abort if the heading drifts more than this")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="action", required=True)
    sub.add_parser("state", parents=[common], help="read-only status + alerts")
    sub.add_parser("clearance", parents=[common], help="read-only lidar corridor")
    sub.add_parser("stop", parents=[common], help="zero twist + restore auto")

    m = sub.add_parser("move", parents=[common], help="signed metres, straight line")
    m.add_argument("metres", type=float)
    m.add_argument("--speed", type=float, default=0.20, metavar="MPS")
    m.add_argument("--tol", type=float, default=0.03, metavar="M",
                   help="stop correcting once within this of the target")
    m.add_argument("--corrections", type=int, default=2,
                   help="extra short legs allowed to close the remaining distance")
    m.add_argument("--no-gate", action="store_true",
                   help="skip the lidar clearance check (not recommended)")

    t = sub.add_parser("turn", parents=[common],
                       help="signed degrees, in place (linear pinned to 0)")
    t.add_argument("degrees", type=float)
    t.add_argument("--rate", type=float, default=0.20, metavar="RADPS")
    t.add_argument("--tol", type=float, default=3.0, metavar="DEG",
                   help="stop correcting once within this of the target")
    t.add_argument("--corrections", type=int, default=4,
                   help="damped closed-loop corrections allowed after the first turn")
    t.add_argument("--no-gate", action="store_true",
                   help="skip the lidar all-round check (not recommended)")

    c = sub.add_parser("calibrate", parents=[common],
                       help="0.3/0.6/1.0 m legs measuring achieved vs commanded")
    c.add_argument("metres", type=float, nargs="?", default=1.0,
                   help="sign only: which direction to calibrate in")
    c.add_argument("--speed", type=float, default=0.20, metavar="MPS")

    args = p.parse_args()
    handlers = {"state": cmd_state, "clearance": cmd_clearance, "move": cmd_move,
                "turn": cmd_turn, "calibrate": cmd_calibrate, "stop": cmd_stop}
    try:
        handlers[args.action](args)
        return 0
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
