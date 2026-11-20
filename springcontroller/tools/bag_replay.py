#!/usr/bin/env python3
"""Turn a springcontroller rosbag into an interactive replay page (HTML).

Usage (ROS sourced, springcontroller venv python, so pinocchio is available):
    bag_replay.py BAG [--robot gen2|gen3] [--start S] [--end E] [--out replay.html]

The page shows, over time: the arm (forward kinematics of the recorded joint
states), the gripper point and its pointing axis, the trajectory spring's
anchor and target axis, the springs' pulls on the gripper, every Cartesian
spring's line to its target, and linked charts (tracking error, pulls,
moment, wrist joints) with pauses shaded. Events from the bag's /rosout
(torque enable, drop-outs, trajectory aborts, collisions) are listed and
clickable.

Where the springs were:
  * Bags recorded after 2026-11-20 carry them: ~/trajectory/state "anchor"
    and ~/spring_states. Those are used as-is.
  * Older bags don't. The tracking anchor is then estimated from the
    recorded spring torques: torques minus collision and joint-limit
    repulsion, solved through the gripper Jacobian for the force and moment,
    minus the tip spring's recorded force; anchor = gripper + recorded error
    size along that force. Damping and the force cap bend the direction a
    little (a median 1.4 cm against the exact anchor on 2026-11-20). Spring
    targets come from ~/spring_forces (force direction x recorded distance).
  The page says which source it used.

Only reads the bag: starts no ROS nodes and publishes nothing.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
from springcontroller.trajectory_spring import rotation_exp  # noqa: E402
from springcontroller.urdf_arm_configuration import URDFArmConfiguration  # noqa: E402

ROBOTS = {
    "gen2": {
        "urdf": PKG / "flat_urdf_files" / "gen2_robotiq_flat.urdf",
        "link": "j2s7s300_ee_link",
        "chain": ["j2s7s300_link_base"] + [f"j2s7s300_link_{i}" for i in range(1, 8)]
                 + ["j2s7s300_ee_link"],
        "joint_states": "/gen2/joint_states_lowlevel",
        "scene": PKG / "config" / "gen2_collision_scene.yaml",
        "locked": ["gripper_finger1_joint", "gripper_finger2_joint",
                   "gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint",
                   "gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint"],
    },
    "gen3": {
        "urdf": PKG / "flat_urdf_files" / "gen3_kinova_flat.urdf",
        "link": "end_effector_link",
        "chain": ["base_link"] + [f"{n}_link" for n in ("shoulder", "half_arm_1", "half_arm_2",
                                                        "forearm", "spherical_wrist_1",
                                                        "spherical_wrist_2", "bracelet")]
                 + ["end_effector_link"],
        "joint_states": "/kinova/joint_states_lowlevel",
        "scene": PKG / "config" / "gen3_collision_scene.yaml",
        "locked": ["robotiq_85_left_knuckle_joint", "robotiq_85_right_knuckle_joint",
                   "robotiq_85_left_inner_knuckle_joint", "robotiq_85_right_inner_knuckle_joint",
                   "robotiq_85_left_finger_tip_joint", "robotiq_85_right_finger_tip_joint"],
    },
}
EVENT_KEYS = ("trajectory_spring", "Torque control ENABLED", "left torque mode",
              "COLLISION", "torque control DISABLED", "Joint move finished")


def read_bag(bag, topics, t0, t1):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    present = [t for t in topics if t in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=present))
    out = {t: ([], []) for t in topics}
    start = None
    while r.has_next():
        topic, data, t = r.read_next()
        t *= 1e-9
        start = t if start is None else start
        rel = t - start
        if rel < t0 - 1.0 or rel > t1 + 1.0:
            continue
        try:
            msg = deserialize_message(data, get_message(types[topic]))
        except (ModuleNotFoundError, AttributeError) as exc:     # interfaces not sourced
            sys.exit(f"Can't read {topic} ({types[topic]}): {exc}. Source the workspace.")
        out[topic][0].append(rel)
        out[topic][1].append(msg)
    return out, types


def latest(series, t):
    ts, vs = series
    if not ts:
        return None
    i = bisect.bisect_right(ts, t) - 1
    return vs[max(i, 0)] if i >= 0 else None


def quat_z_axis(xyzw):
    x, y, z, w = xyzw
    return [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]


def scene_boxes(path):
    try:
        objs = (yaml.safe_load(open(path)) or {}).get("objects", {})
    except OSError:
        return []
    boxes = []
    for name, o in objs.items():
        if o.get("shape") == "box":
            boxes.append({"name": name, "center": o["pose"]["position"], "size": o["dimensions"]})
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag")
    ap.add_argument("--robot", choices=ROBOTS, default="gen2")
    ap.add_argument("--start", type=float, default=0.0, help="bag seconds (default: start)")
    ap.add_argument("--end", type=float, default=1e9, help="bag seconds (default: end)")
    ap.add_argument("--joint-states", help="joint state topic (default: per robot)")
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="cap on page samples; the step grows to fit (default 4000)")
    ap.add_argument("--out", help="output HTML (default: <bag>_replay.html next to the bag)")
    a = ap.parse_args()

    R = ROBOTS[a.robot]
    js_topic = a.joint_states or R["joint_states"]
    bag = Path(a.bag).resolve()
    out_path = Path(a.out) if a.out else bag.parent / f"{bag.name}_replay.html"
    T = {"js": js_topic, "tau": "/virtual_spring_node/joint_torques",
         "rep": "/virtual_spring_node/repulsion_torques",
         "jl": "/virtual_spring_node/joint_limit_repulsion_torques",
         "sf": "/virtual_spring_node/spring_forces",
         "st": "/virtual_spring_node/trajectory/state",
         "ss": "/virtual_spring_node/spring_states",
         "torque": f"/{a.robot}_torque_control/status", "log": "/rosout"}
    data, types = read_bag(bag, list(T.values()), a.start, a.end)
    S = {k: data[v] for k, v in T.items()}
    if not S["js"][0]:
        sys.exit(f"No {js_topic} in the bag (topics: {sorted(types)}). Use --joint-states.")

    arm = URDFArmConfiguration.from_urdf(str(R["urdf"]), locked_joint_names=R["locked"])
    names = arm.joint_names
    link = R["link"]
    chain = [n for n in R["chain"] if arm._model.existFrame(n)]

    t_first, t_last = S["js"][0][0], S["js"][0][-1]
    t0, t1 = max(a.start, t_first), min(a.end, t_last)
    dt = max(0.05, (t1 - t0) / a.max_samples)
    recorded_anchor = any("anchor" in json.loads(m.data) for m in S["st"][1][:50]) if S["st"][1] else False
    recorded_states = bool(S["ss"][1])

    keys = ["t", "state", "torque", "links", "x", "gz", "anchor", "az", "F", "M", "Ftip",
            "springs", "ep", "eR", "pt", "q"]
    D = {k: [] for k in keys}
    caps = {}
    for t in np.arange(t0, t1, dt):
        jmsg = latest(S["js"], t)
        idx = [list(jmsg.name).index(n) for n in names]
        q = np.array([jmsg.position[i] for i in idx])
        arm.update_from_angles(q, np.zeros(len(q)))
        Tl = arm.get_link_transform(link)
        x, Rg = Tl[:3, 3], Tl[:3, :3]
        st = latest(S["st"], t)
        st = json.loads(st.data) if st is not None else {}
        state = st.get("state", "IDLE")
        tracking = state in ("RUNNING", "PAUSED", "HOLDING") and st.get("mix", 0) > 0
        trk = (st.get("springs") or {}).get("tracking") or {}
        if trk:
            caps = {"max_force": trk.get("max_force"), "max_moment": trk.get("max_moment")}

        # Springs' pulls at the gripper, from the torques (works on every bag).
        F = M = None
        tau, rep, jl = (latest(S[k], t) for k in ("tau", "rep", "jl"))
        sf = latest(S["sf"], t)
        # Position springs only: a PoseSpring's "force" field is a moment,
        # a JointSpring's is zero (see SpringForce.msg).
        forces = ({s.name: (np.array(s.force), s.distance) for s in sf.springs
                   if s.kind == "force"} if sf else {})
        F_tip = sum((f for f, _ in forces.values()), np.zeros(3))
        if tau is not None and rep is not None and jl is not None:
            J = arm.get_jacobian(link, np.zeros(3))
            w, *_ = np.linalg.lstsq(J.T, np.array(tau.effort) - np.array(rep.effort)
                                    - np.array(jl.effort), rcond=None)
            F, M = w[:3] - F_tip, w[3:]
        anc = st.get("anchor") if tracking else None
        if anc is not None:                                  # recorded (new bags)
            anchor, az = anc["position"], quat_z_axis(anc["orientation_xyzw"])
            if anc.get("force") is not None:
                F, M = np.array(anc["force"]), np.array(anc["moment"])
        elif tracking and F is not None and st.get("position_error_m") is not None \
                and np.linalg.norm(F) > 0.05:                 # estimated (old bags)
            anchor = (x + st["position_error_m"] * F / np.linalg.norm(F)).tolist()
            az = None
            if st.get("rotation_error_rad") and np.linalg.norm(M) > 1e-3:
                eR = st["rotation_error_rad"] * M / np.linalg.norm(M)
                az = (rotation_exp(eR) @ Rg)[:, 2].tolist()
        else:
            anchor = az = None

        # Cartesian springs' lines to their targets.
        springs = []
        ss = latest(S["ss"], t)
        if ss is not None:
            for s in json.loads(ss.data).get("springs", []):
                if s.get("target") and s.get("attachment_world") and s.get("enabled") \
                        and s.get("stiffness", 0) > 0:
                    springs.append([s["name"], s["attachment_world"], s["target"]])
        else:
            for name, (f, dist) in forces.items():
                if np.linalg.norm(f) > 0.02:
                    springs.append([name, x.tolist(), (x + dist * f / np.linalg.norm(f)).tolist()])

        tq = latest(S["torque"], t)
        r = lambda v, n=4: None if v is None else [round(float(c), n) for c in v]  # noqa: E731
        D["t"].append(round(float(t), 3))
        D["state"].append(state)
        D["torque"].append(tq.data if tq is not None else "")
        D["links"].append([r(arm.get_link_transform(n)[:3, 3], 3) for n in chain])
        D["x"].append(r(x))
        D["gz"].append(r(Rg[:, 2], 3))
        D["anchor"].append(r(anchor))
        D["az"].append(r(az, 3))
        D["F"].append(r(F, 2))
        D["M"].append(r(M, 2))
        D["Ftip"].append(r(F_tip, 2))
        D["springs"].append([[n, r(p, 4), r(q2, 4)] for n, p, q2 in springs])
        D["ep"].append(None if st.get("position_error_m") is None
                       else round(st["position_error_m"] * 100, 2))
        D["eR"].append(st.get("rotation_error_rad"))
        D["pt"].append(st.get("t"))
        D["q"].append([round(float(v), 3) for v in q])

    events = []
    for t, m in zip(*S["log"]):
        if t0 <= t <= t1 and any(k in m.msg for k in EVENT_KEYS):
            events.append([round(t, 2), m.name, m.msg[:300]])
    prev = "IDLE"
    for t, s in zip(D["t"], D["state"]):
        if s == "RUNNING" and prev in ("IDLE", "HOLDING"):
            events.append([t, "trajectory", "Trajectory started."])
        prev = s
    events.sort(key=lambda e: e[0])

    D.update({
        "bag": bag.name, "robot": a.robot, "dt": round(dt, 3), "caps": caps, "events": events,
        "anchor_source": ("recorded" if recorded_anchor else "estimated from spring torques"),
        "springs_source": ("recorded (~/spring_states)" if recorded_states
                           else "estimated from ~/spring_forces"),
        "scene": scene_boxes(R["scene"]),
    })
    html = (HERE / "bag_replay_template.html").read_text()
    out_path.write_text(html.replace("__DATA__", json.dumps(D, separators=(",", ":"))))
    print(f"{len(D['t'])} samples ({D['dt']} s step), {len(events)} events, "
          f"anchor {D['anchor_source']} -> {out_path} ({os.path.getsize(out_path) // 1024} KB)")


if __name__ == "__main__":
    main()
