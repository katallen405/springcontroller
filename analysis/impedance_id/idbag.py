"""
idbag.py -- the bag + kinematics half of impedance_id.

Reads joint states (and, if present, the UR3e wrist wrench and the
commanded spring torques) from one bag, and turns joint angles into tip
position (and orientation, needed to rotate the UR3e's wrench into the
base frame).

ROS imports (rosbag2_py, rclpy, message types) and pinocchio live inside
the functions, so the rest imports without them (tests, plotting).

Times: everything is in seconds since the FIRST JOINT-STATE MESSAGE of the
bag, using header stamps where set and the bag receive time otherwise.
`impedance_id.py overview` plots on this same time axis, so segment
start/end times read off that plot line up.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[2] / "springcontroller"
URDF_DIR = PKG / "flat_urdf_files"

UR3E_JOINTS = [
    "ur3e_shoulder_pan_joint", "ur3e_shoulder_lift_joint", "ur3e_elbow_joint",
    "ur3e_wrist_1_joint", "ur3e_wrist_2_joint", "ur3e_wrist_3_joint",
]

ROBOTS = {
    "gen2": {
        "joint_states": "/gen2/joint_states_lowlevel",
        "joints": [f"j2s7s300_joint_{i}" for i in range(1, 8)],
        "urdf": URDF_DIR / "gen2_robotiq_flat.urdf",
        "tip": "j2s7s300_ee_link",
        "wrench": None,
    },
    "gen3": {
        "joint_states": "/kinova/joint_states_lowlevel",
        "joints": [f"joint_{i}" for i in range(1, 8)],
        "urdf": URDF_DIR / "gen3_kinova_flat.urdf",
        "tip": "end_effector_link",
        "wrench": None,
    },
    "ur3e": {
        "joint_states": "/joint_states",
        "joints": UR3E_JOINTS,
        "urdf": None,        # UR's published DH parameters instead (see ur3e_fk)
        "tip": "tool0",
        "wrench": "/force_torque_sensor_broadcaster/wrench",
    },
}

TORQUE_TOPIC = "/virtual_spring_node/joint_torques"


# ── Bag reading ───────────────────────────────────────────────────────────────

def _storage_id(bag: str) -> str:
    """mcap unless the bag folder holds .db3 files."""
    p = Path(bag)
    if p.is_dir() and any(p.glob("*.db3")):
        return "sqlite3"
    if p.suffix == ".db3":
        return "sqlite3"
    return "mcap"


def bag_topics(bag: str) -> dict:
    import rosbag2_py
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=_storage_id(bag)),
           rosbag2_py.ConverterOptions("", ""))
    return {t.name: t.type for t in r.get_all_topics_and_types()}


def read_bag(bag: str, robot: str, joint_states_topic: str | None = None,
             wrench_topic: str | None = None, torque_topic: str | None = TORQUE_TOPIC):
    """
    Returns a dict:
      t, q            joint-state times (s, from first joint state) and angles,
                      columns in ROBOTS[robot]["joints"] order
      wrench_t, wrench_f, wrench_frame   (if the wrench topic is in the bag)
      tau_t, tau      commanded spring torques (if recorded)
      topics          every topic in the bag (for error messages)
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    R = ROBOTS[robot]
    js_topic = joint_states_topic or R["joint_states"]
    w_topic = wrench_topic if wrench_topic is not None else R["wrench"]

    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id=_storage_id(bag)),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    if js_topic not in types:
        raise ValueError(f"{bag}: no {js_topic} (topics: {sorted(types)}). "
                         "Pass --joint-states.")
    wanted = [js_topic] + [tp for tp in (w_topic, torque_topic) if tp and tp in types]
    r.set_filter(rosbag2_py.StorageFilter(topics=wanted))
    cls = {tp: get_message(types[tp]) for tp in wanted}

    def stamp(msg, t_ns):
        h = getattr(msg, "header", None)
        if h is not None:
            s = h.stamp.sec + h.stamp.nanosec * 1e-9
            if s > 0:
                return s
        return t_ns * 1e-9

    joints = R["joints"]
    js_t, js_q = [], []
    w_t, w_f, w_frame = [], [], ""
    tau_t, tau = [], []
    tau_names = None
    while r.has_next():
        topic, data, t_ns = r.read_next()
        msg = deserialize_message(data, cls[topic])
        if topic == js_topic:
            names = list(msg.name)
            try:
                order = [names.index(n) for n in joints]
            except ValueError:
                # Some drivers publish only a subset per message (e.g. a
                # separate gripper joint_states). Skip messages without the arm.
                continue
            js_t.append(stamp(msg, t_ns))
            js_q.append([msg.position[i] for i in order])
        elif topic == w_topic:
            w_t.append(stamp(msg, t_ns))
            f = msg.wrench.force
            w_f.append([f.x, f.y, f.z])
            w_frame = msg.header.frame_id
        elif topic == torque_topic:
            names = list(msg.name)
            if tau_names is None:
                tau_names = [n for n in joints if n in names] or names
            try:
                tau.append([msg.effort[names.index(n)] for n in tau_names])
            except (ValueError, IndexError):
                continue
            tau_t.append(stamp(msg, t_ns))

    if not js_t:
        raise ValueError(f"{bag}: {js_topic} has no messages with joints {joints}")
    t0 = min(js_t)
    out = {"t": np.asarray(js_t) - t0, "q": np.asarray(js_q), "topics": types,
           "joint_states_topic": js_topic}
    if w_t:
        out.update(wrench_t=np.asarray(w_t) - t0, wrench_f=np.asarray(w_f),
                   wrench_frame=w_frame, wrench_topic=w_topic)
    if tau_t:
        out.update(tau_t=np.asarray(tau_t) - t0, tau=np.asarray(tau))
    return out


# ── Forward kinematics ────────────────────────────────────────────────────────

# UR3e DH parameters (Universal Robots' published values). Frame 0 is UR's
# "base" frame (ROS ur_description: `base`, which is base_link rotated 180
# deg about z); the last frame is the tool flange (ROS `tool0`).
UR3E_DH = {
    "d": [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921],
    "a": [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0],
    "alpha": [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0],
}


def ur3e_fk_one(q):
    """4x4 pose of tool0 in UR's base frame for one 6-vector of angles."""
    T = np.eye(4)
    for th, d, a, al in zip(q, UR3E_DH["d"], UR3E_DH["a"], UR3E_DH["alpha"]):
        ct, st, ca, sa = np.cos(th), np.sin(th), np.cos(al), np.sin(al)
        T = T @ np.array([[ct, -st * ca, st * sa, a * ct],
                          [st, ct * ca, -ct * sa, a * st],
                          [0, sa, ca, d],
                          [0, 0, 0, 1]])
    return T


def make_fk(robot: str, urdf: str | None = None, tip: str | None = None):
    """
    fk(q (N, n_joints)) -> (positions (N, 3), rotations (N, 3, 3)) of the
    tip in the robot's base frame. UR3e uses DH unless --urdf is given.
    """
    R = ROBOTS[robot]
    urdf = urdf or (str(R["urdf"]) if R["urdf"] else None)
    tip = tip or R["tip"]
    if urdf is None:
        def fk(q):
            q = np.atleast_2d(q)
            Ts = np.array([ur3e_fk_one(qi) for qi in q])
            return Ts[:, :3, 3], Ts[:, :3, :3]
        return fk
    return _pinocchio_fk(urdf, R["joints"], tip)


def _pinocchio_fk(urdf: str, joints, tip: str):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(os.path.expanduser(urdf))
    data = model.createData()
    if not model.existFrame(tip):
        raise ValueError(f"{urdf} has no frame {tip!r}")
    fid = model.getFrameId(tip)
    slots = []
    for name in joints:
        if not model.existJointName(name):
            raise ValueError(f"{urdf} has no joint {name!r}")
        j = model.joints[model.getJointId(name)]
        slots.append((j.idx_q, j.nq))

    def fk(q):
        q = np.atleast_2d(q)
        P = np.empty((len(q), 3))
        Rm = np.empty((len(q), 3, 3))
        cfg = pin.neutral(model)
        for i, qi in enumerate(q):
            for (k, nq), a in zip(slots, qi):
                if nq == 2:          # continuous joint: (cos, sin)
                    cfg[k], cfg[k + 1] = np.cos(a), np.sin(a)
                else:
                    cfg[k] = a
            pin.framesForwardKinematics(model, data, cfg)
            P[i] = data.oMf[fid].translation
            Rm[i] = data.oMf[fid].rotation
        return P, Rm
    return fk


def wrench_to_base(f, R_tip, frame: str):
    """
    Force vectors into the base frame. frame "base" leaves them; "tool"
    rotates each by the tip orientation at that sample (R_tip (N, 3, 3),
    already matched in time to f).
    """
    f = np.asarray(f, dtype=float)
    if frame == "base":
        return f
    if frame == "tool":
        return np.einsum("nij,nj->ni", R_tip, f)
    raise ValueError(f"wrench frame must be 'tool' or 'base', not {frame!r}")
