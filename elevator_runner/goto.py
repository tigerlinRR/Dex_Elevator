"""Drive to a named waypoint, streaming the base's situation as it goes.

Uses the same wait_for_arrival as the elevator runs — one implementation, so this
behaves the way a real run does rather than being a second, differently-behaved path.
"""
import sys, time, threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import server as S

TARGET = sys.argv[1] if len(sys.argv) > 1 else "BBB"
r = S.CFG["robot_serial"]
st = S.AX.robot_state(r)
if st.get("isEmergencyStop"):
    sys.exit("ABORT: emergency stop engaged")
pois = {p["name"]: p for p in S.AX.list_pois(r)}
if TARGET not in pois:
    sys.exit(f"ABORT: no waypoint named {TARGET!r}")
dest = pois[TARGET]
d0 = S.dist_cm(st.get("x"), st.get("y"), dest["x"], dest["y"])
print(f"battery {st.get('battery')}%  now ({st.get('x')}, {st.get('y')})")
print(f"driving to {TARGET!r} ({dest['x']:.2f}, {dest['y']:.2f}) — {d0/100:.2f} m\n")

# Mirror RUN.log to stdout so the terminal follows the drive.
seen = [0]
def pump():
    while True:
        lines = S.RUN.snapshot()["log"]
        while seen[0] < len(lines):
            print(lines[seen[0]], flush=True); seen[0] += 1
        time.sleep(0.5)
threading.Thread(target=pump, daemon=True).start()

resp = S.AX.create_task(r, [S.make_task_point(dest, dest.get("areaId", ""))],
                        f"Goto {TARGET}", 2)
tid = ((resp.get("data") or {}).get("taskId") or resp.get("taskId")
       or (resp.get("data") or {}).get("id"))
if not tid:
    sys.exit(f"ABORT: no taskId in {resp}")
S.RUN.log(f"dispatched {tid}")
fin, canc, fast, _ = S.wait_for_arrival(r, tid, dest, 15.0, 90.0, True, False,
                                        f"goto {TARGET}", 600.0)
time.sleep(1)
st = S.AX.robot_state(r)
err = S.dist_cm(st.get("x"), st.get("y"), dest["x"], dest["y"])
print(f"\narrived={fin} cancelled={canc}  now ({st.get('x')}, {st.get('y')}) "
      f"yaw={st.get('yaw')}  {err:.0f} cm from {TARGET!r}  battery {st.get('battery')}%")
if not fin:
    try:
        S.AX.cancel_task(tid); print("task cancelled")
    except Exception as e:
        print("cancel failed:", e)
