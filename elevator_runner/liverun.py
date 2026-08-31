"""One live loop: drive to the elevator point and press 1 4 2 5.

New this run: the torso lift now picks the BEST-margin height for each button
(arm.lift.objective: margin) instead of staying put whenever the current height is
merely good enough, so the body visibly tracks the panel row by row.

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

pois = {p["name"]: p for p in S.AX.list_pois(serial)}
params = {
    "robot_serial": serial,
    "route": [pois["AAA"], pois["elevator test"]],
    "elevator": pois["elevator test"],
    "floors": "1 4 2 5",
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
