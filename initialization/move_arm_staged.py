"""Move the arm to a joint target in CHECKED legs, never in one sweep.

    python3 initialization/move_arm_staged.py --to="j1,..,j6" [--go] [--speed S]


Written after a single 224.9 deg movej took the arm toward a wall: the controller's
self-collision model covers the arm's own links only, so "IK returned a solution" and
"self-collision clear" together still say nothing about the room or the robot's torso.
Each leg is sampled in joint space and every sample is checked three ways before
anything moves, and the leg is only executed with --go.
"""
import argparse, sys, numpy as np
from core.robot.realman import RealmanArm
from core.config import load_pipeline
from core.transforms import quat_to_matrix

PANEL_X = 530.0          # mm, panel face in the base frame (hand-touched truth at
                         # the registered station; see configs/stations.yaml)
COLUMN_R = 120.0         # mm, CRUDE radius of the torso/lift column about the base z
                         # axis. There is no model of the torso, so this is a stand-in
                         # and it only watches the PLUNGER TIP — not the elbow, not the
                         # forearm, not the camera bracket or its cabling. Passing all
                         # three checks is NOT "it will not collide": the controller's
                         # own self-collision model covers the arm's links and end
                         # effector only. A large re-positioning move still wants eyes
                         # on it the first time it is run.

ap = argparse.ArgumentParser()
ap.add_argument("--to", required=True, help="target joints, degrees, comma separated")
ap.add_argument("--go", action="store_true")
ap.add_argument("--speed", type=float, default=0.10)
args = ap.parse_args()

target = np.array([float(v) for v in args.to.split(",")], dtype=float)
cfg = load_pipeline()
tcp = np.array(cfg["end_effector"]["tcp_offset"], dtype=float) * 1000.0
c = cfg["arm"]
arm = RealmanArm(side=c.get("side", "right"), ip=c.get("ip"), port=c.get("port", 8080))
arm.connect()
sdk = arm._require()
start = np.array(arm.get_joint_angles(), dtype=float)

# Two legs: flip the WRIST first (the body stays put, so the swept sphere is the only
# thing to clear), then move the BODY with the tool already oriented. Splitting it this
# way means each leg has one intelligible failure mode instead of a tangled one.
legs = [("wrist J4-J6", np.concatenate([start[:3], target[3:]])),
        ("body  J1-J3", target)]

def tip_of(q):
    pose = sdk.rm_algo_forward_kinematics([float(v) for v in q], 0)
    R = quat_to_matrix(*[float(v) for v in pose[3:7]])
    p = np.array([float(v) for v in pose[:3]]) * 1000.0
    return p + R @ tcp

ok = True
frm = start
for name, to in legs:
    path = np.linspace(frm, to, 60)
    hits = sum(1 for q in path
               if sdk.rm_algo_safety_robot_self_collision_detection([float(v) for v in q]) != 0)
    tips = np.stack([tip_of(q) for q in path])
    gap_panel = PANEL_X - tips[:, 0].max()                    # + = still short of the panel
    radial = np.hypot(tips[:, 0], tips[:, 1]).min()           # distance from the base z axis
    moved = np.abs(to - frm)
    print(f"{name}: largest joint change {moved.max():6.1f} deg")
    print(f"   self-collision samples hit : {hits} of {len(path)}")
    print(f"   closest the tip gets to the panel : {gap_panel:7.1f} mm"
          f"   {'OK' if gap_panel > 80 else 'TOO CLOSE'}")
    print(f"   closest the tip gets to the column: {radial:7.1f} mm"
          f"   {'OK' if radial > COLUMN_R else 'CHECK BY EYE'}")
    if hits or gap_panel <= 80 or radial <= COLUMN_R:
        ok = False
    frm = to

if not ok:
    print("\nREFUSED: a leg did not pass. Nothing moved.")
    arm.disconnect(); sys.exit(1)
if not args.go:
    print("\nall legs pass — re-run with --go to execute them one at a time")
    arm.disconnect(); sys.exit(0)

frm = start
for name, to in legs:
    print(f"\nmoving: {name} -> {np.round(to, 2).tolist()}")
    if not arm.move_joints_sync(to.tolist(), speed=args.speed):
        # A False here means arrival was NOT OBSERVED, not that the arm stopped —
        # rm_movej keeps executing in the background. Stop it before deciding anything.
        sdk.rm_set_arm_stop(); sdk.rm_set_delete_current_trajectory()
        print("!! arrival not observed — arm stopped and trajectory deleted")
        arm.disconnect(); sys.exit(1)
    print("   arrived", np.round(arm.get_joint_angles(), 2).tolist())
    frm = to
arm.disconnect()
