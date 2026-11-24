#!/usr/bin/env python3
"""Run OscillationWatchdog over real Gen2 bags: the tracking spring's force and
moment reconstructed at 100 Hz from the recorded torques (total - repulsion
- joint-limit repulsion, solved through the gripper Jacobian, minus position
springs' recorded forces). Reports trips and the largest RMS seen.

Usage (ROS + workspace sourced): bag_watchdog.py BAG [BAG ...]
Use it to check watchdog thresholds against real runs: on 2026-11 the 25
working bags gave no trips and the 2026-11-23 incident tripped at 183.5 s.
"""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from pathlib import Path
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG))
from springcontroller.trajectory_spring import OscillationWatchdog  # noqa: E402
from springcontroller.urdf_arm_configuration import URDFArmConfiguration  # noqa: E402

URDF = str(PKG / "flat_urdf_files" / "gen2_robotiq_flat.urdf")
LOCKED = ["gripper_finger1_joint", "gripper_finger2_joint",
          "gripper_finger1_inner_knuckle_joint", "gripper_finger2_inner_knuckle_joint",
          "gripper_finger1_finger_tip_joint", "gripper_finger2_finger_tip_joint"]
arm = URDFArmConfiguration.from_urdf(URDF, locked_joint_names=LOCKED)

for bag in sys.argv[1:]:
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    T = {"/gen2/joint_states_lowlevel": "js", "/virtual_spring_node/joint_torques": "tau",
         "/virtual_spring_node/repulsion_torques": "rep",
         "/virtual_spring_node/joint_limit_repulsion_torques": "jl",
         "/virtual_spring_node/spring_forces": "sf", "/gen2_torque_control/status": "st"}
    r.set_filter(rosbag2_py.StorageFilter(topics=[t for t in T if t in types]))
    last = {}
    w = OscillationWatchdog()
    start, trips, peak = None, [], (0.0, 0.0, 0.0)
    while r.has_next():
        topic, data, t = r.read_next()
        t *= 1e-9
        start = t if start is None else start
        rel = t - start
        if topic == "/virtual_spring_node/spring_forces" and "springcontroller_interfaces" not in sys.modules:
            try:
                import springcontroller_interfaces.msg  # noqa: F401
            except ImportError:
                continue
        m = deserialize_message(data, get_message(types[topic]))
        k = T[topic]
        last[k] = m
        if k != "tau" or not all(x in last for x in ("js", "rep", "jl")):
            continue
        if getattr(last.get("st"), "data", "ENABLED") != "ENABLED":
            w.reset()
            continue
        js = last["js"]
        q = np.array([js.position[list(js.name).index(n)] for n in arm.joint_names])
        arm.update_from_angles(q, np.zeros(7))
        J = arm.get_jacobian("j2s7s300_ee_link", np.zeros(3))
        tau = np.array(m.effort) - np.array(last["rep"].effort) - np.array(last["jl"].effort)
        wr, *_ = np.linalg.lstsq(J.T, tau, rcond=None)
        F, M = wr[:3], wr[3:]
        sf = last.get("sf")
        if sf is not None:
            for s in sf.springs:
                if s.kind == "force":
                    F = F - np.array(s.force)
        why = w.update(rel, F, M)
        if w.last["force_rms"] > peak[0]:
            peak = (w.last["force_rms"], w.last["freq_hz"], rel)
        if why:
            trips.append((rel, why))
            w.reset()
    name = bag.rstrip("/").split("/")[-1]
    print(f"{name}: {len(trips)} trip(s); largest force swing {peak[0]:.2f} N RMS "
          f"at {peak[1]:.1f} Hz ({peak[2]:.1f} s)")
    for tt, why in trips[:5]:
        print(f"    {tt:7.2f} s  {why}")
