"""One live loop: drive via a start waypoint to the elevator point and press buttons.

    python3 liverun.py                  # BBB -> elevator test, press 1 4 2 5
    python3 liverun.py AAA              # AAA -> elevator test
    python3 liverun.py BBB "1 4"        # different floors

The start waypoint is only a place to drive FROM, so its own docking accuracy does not
matter — measured, `BBB` lands 3-68 cm from its point while `elevator test` closes to
1.5-2.3 cm every time, and only the latter governs whether the press works. Passing "-"
skips the start point and drives straight to the elevator from wherever the robot is.

Pre-flight before anything is dispatched. An earlier attempt spent two route
dispatches and ten minutes discovering from a timeout that the emergency stop was
engaged — a fact that is one field away.
"""
import socket, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S

def port_open(ip, port=8080):
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect((ip, port)); return True
    except Exception:
        return False
    finally:
        s.close()

serial = S.CFG["robot_serial"]
st = S.AX.robot_state(serial)
right, left = port_open("192.168.11.33"), port_open("192.168.11.32")
print(f"pre-flight: eStop={st.get('isEmergencyStop')} battery={st.get('battery')}% "
      f"online={st.get('isOnline')} moveState={st.get('moveState')} "
      f"RIGHT={'up' if right else 'DOWN'} LEFT={'up' if left else 'DOWN'}", flush=True)
if st.get("isEmergencyStop"):
    sys.exit("ABORT: emergency stop engaged")
if not right:
    sys.exit("ABORT: right arm controller not answering")
if not left:
    sys.exit("ABORT: left arm controller not answering (it owns the lift)")
if not st.get("isOnline"):
    sys.exit("ABORT: base offline")

start = sys.argv[1] if len(sys.argv) > 1 else "BBB"
floors = sys.argv[2] if len(sys.argv) > 2 else "1 4 2 5"

pois = {p["name"]: p for p in S.AX.list_pois(serial)}
for name in ([] if start == "-" else [start]) + ["elevator test"]:
    if name not in pois:
        sys.exit(f"ABORT: no waypoint named {name!r}")
route = ([] if start == "-" else [pois[start]]) + [pois["elevator test"]]
print(f"route: {' -> '.join(p['name'] for p in route)}  floors: {floors}", flush=True)

params = {
    "robot_serial": serial,
    "route": route,
    "elevator": pois["elevator test"],
    "floors": floors,
    "loops": 1,
    "tolerance_cm": 8.0,
    "max_retries": 0,          # one attempt; a retry re-drives the whole route
    "backoff_sec": 5.0,
    "run_mode": 2,
    "dry_run": False,
    "poll_timeout_sec": 300,
    "corrective_drives": 1,
    "fast_arrival": True,
}
t = threading.Thread(target=S.run_loop, args=(params,), daemon=True)
t.start()
seen = 0
while True:
    lines = S.RUN.snapshot()["log"]
    while seen < len(lines):
        print(lines[seen], flush=True); seen += 1
    if not t.is_alive():
        break
    time.sleep(1)
print("=== final ===", S.RUN.snapshot()["state"], flush=True)
