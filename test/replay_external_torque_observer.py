#!/usr/bin/env python3
"""
Offline replay of ExternalTorqueObserver over recorded Gen3 bags (no live
ROS graph). Runs the observer on /kinova/joint_states_lowlevel during
torque-ENABLED windows and reports per-joint |τ̂_ext| statistics at rest vs.
while moving (in these bags, motion while torque-enabled is a person
hand-guiding the arm), plus the learned rest bias. Use it to pick
ext_torque_deadband / ext_torque_bias_max before any live tuning.

Usage:
    python3 test/replay_external_torque_observer.py <bag> [<bag> ...] [--gain 20]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from springcontroller.external_torque_observer import ExternalTorqueObserver
from springcontroller.urdf_arm_configuration import URDFArmConfiguration

URDF_PATH = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "gen3_kinova_flat.urdf"
)
GRIPPER_LOCKED_JOINTS = [
    "robotiq_85_left_knuckle_joint",
    "robotiq_85_right_knuckle_joint",
    "robotiq_85_left_inner_knuckle_joint",
    "robotiq_85_right_inner_knuckle_joint",
    "robotiq_85_left_finger_tip_joint",
    "robotiq_85_right_finger_tip_joint",
]
JS_TOPIC = "/kinova/joint_states_lowlevel"
STATUS_TOPIC = "/gen3_torque_control/status"
MAX_GAP_S = 0.05
REST_VEL = 0.02
MOVING_VEL = 0.1


def replay(bag: str, gain: float, bias_max: float) -> dict:
    arm = URDFArmConfiguration.from_urdf(URDF_PATH, locked_joint_names=GRIPPER_LOCKED_JOINTS)
    obs = ExternalTorqueObserver(
        arm.n_dof, gain=gain, bias_max=bias_max, rest_velocity=REST_VEL,
        rest_time=1.0, bias_time_constant=2.0,
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=bag), rosbag2_py.ConverterOptions("", ""))
    reader.set_filter(rosbag2_py.StorageFilter(topics=[JS_TOPIC, STATUS_TOPIC]))

    names = arm.joint_names
    order = None
    enabled = False
    last_t = None
    rest_est, rest_raw, moving_est = [], [], []
    enabled_s = 0.0
    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        t = stamp_ns / 1e9
        if topic == STATUS_TOPIC:
            now_enabled = deserialize_message(raw, String).data == "ENABLED"
            if now_enabled and not enabled:
                obs.reset()
                last_t = None
            enabled = now_enabled
            continue
        if not enabled:
            continue
        msg = deserialize_message(raw, JointState)
        if order is None:
            order = [list(msg.name).index(n) for n in names]
        if len(msg.effort) < len(msg.name):
            continue
        q = np.array(msg.position)[order]
        v = np.array(msg.velocity)[order]
        tau = np.array(msg.effort)[order]
        if last_t is None or t - last_t > MAX_GAP_S:
            obs.reset()
            dt = 0.0
        else:
            dt = t - last_t
            enabled_s += dt
        last_t = t
        arm.update_from_angles(q, v)
        M, C, g = arm.get_dynamics()
        est = obs.step(M, C, g, v, tau, dt)
        speed = np.abs(v).max()
        if speed < REST_VEL:
            rest_est.append(np.abs(est))
            rest_raw.append(obs.residual)
        elif speed > MOVING_VEL:
            moving_est.append(np.abs(est))
    return {
        "enabled_s": enabled_s,
        "rest_est": np.array(rest_est).reshape(-1, arm.n_dof),
        "rest_raw": np.array(rest_raw).reshape(-1, arm.n_dof),
        "moving_est": np.array(moving_est).reshape(-1, arm.n_dof),
        "bias": obs.bias,
    }


def fmt(a: np.ndarray) -> str:
    return " ".join(f"{x:6.2f}" for x in a)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bags", nargs="+")
    parser.add_argument("--gain", type=float, default=20.0)
    parser.add_argument("--bias-max", type=float, default=1.0)
    args = parser.parse_args()

    all_rest, all_moving = [], []
    hdr = "j1     j2     j3     j4     j5     j6     j7"
    for bag in args.bags:
        res = replay(bag, args.gain, args.bias_max)
        print(f"== {Path(bag).name}: {res['enabled_s']:.0f}s enabled, "
              f"{len(res['rest_est'])} rest / {len(res['moving_est'])} moving samples")
        if len(res["rest_raw"]):
            print(f"   raw r at rest, mean   [N·m] {fmt(res['rest_raw'].mean(0))}")
            print(f"   |est| at rest, p95    [N·m] {fmt(np.percentile(res['rest_est'], 95, 0))}")
        if len(res["moving_est"]):
            print(f"   |est| moving, p50     [N·m] {fmt(np.percentile(res['moving_est'], 50, 0))}")
        print(f"   learned bias at end   [N·m] {fmt(res['bias'])}")
        all_rest.append(res["rest_est"])
        all_moving.append(res["moving_est"])

    rest = np.vstack(all_rest)
    moving = np.vstack(all_moving)
    print(f"\n== ALL BAGS                    {hdr}")
    if len(rest):
        print(f"   |est| at rest, p95    [N·m] {fmt(np.percentile(rest, 95, 0))}")
        print(f"   |est| at rest, p99    [N·m] {fmt(np.percentile(rest, 99, 0))}")
    if len(moving):
        print(f"   |est| moving, p50     [N·m] {fmt(np.percentile(moving, 50, 0))}")
        print(f"   |est| moving, p90     [N·m] {fmt(np.percentile(moving, 90, 0))}")


if __name__ == "__main__":
    main()
