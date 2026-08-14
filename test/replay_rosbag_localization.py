#!/usr/bin/env python3
"""
replay_rosbag_localization.py

Offline validation harness for the press_to_pin contact-localization fix
(see _localize_contact in press_to_pin.py). Replays a labeled rosbag --
physical presses correlated to bag timestamps via /press_to_pin/test_marker,
see gen3_spring.launch.py's recording setup and the "labeled marker" workflow
it documents -- through PressToPin's actual callbacks directly (no live ROS
graph, no spinning, same technique test/test_press_to_pin.py uses), and
reports which link each press latched onto.

For direct before/after comparison against the measured 9/11 misattribution
baseline (2026-08-12 labeled bag, `~/gen3_bags/rosbag2_2026_08_12-14_39_49/`),
each latch also reports what the OLD "most distal joint over threshold"
heuristic would have picked from the exact same residual vector -- that
heuristic isn't in press_to_pin.py anymore, so it's reimplemented here,
standalone, purely for this comparison.

A fake clock driven by the bag's own recorded timestamps stands in for
self.get_clock() so the state machine's hold_time/rearm_time hysteresis sees
realistic elapsed arm-time regardless of how fast this script itself runs.

Usage:
    python3 test/replay_rosbag_localization.py <bag_path> [--window SECONDS]

Example:
    python3 test/replay_rosbag_localization.py \\
        ~/gen3_bags/rosbag2_2026_08_12-14_39_49/ --window 5
"""

from __future__ import annotations

import argparse
import types
import unittest.mock
from pathlib import Path

import numpy as np
import rclpy
import rclpy.client
import rclpy.task
import rclpy.time
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String

import springcontroller.press_to_pin as ptp
from springcontroller.press_to_pin import PressToPin
from springcontroller_interfaces.srv import AddSpring

GEN3_ARM_URDF = str(
    Path(__file__).resolve().parents[1]
    / "springcontroller" / "flat_urdf_files" / "gen3_kinova_flat_arm_only.urdf"
)

# Matches gen3_press_to_pin.yaml's commanded_torque_topic (the topic name
# actually differs from press_to_pin's own default -- see that config file's
# comment on why Gen3 needs the override).
COMMANDED_TORQUE_TOPIC = "/virtual_spring_node/joint_torques"

TOPIC_TYPES = {
    "/joint_states": JointState,
    COMMANDED_TORQUE_TOPIC: JointState,
    "/press_to_pin/test_marker": String,
}


class ReplayClock:
    """Stand-in for Node.get_clock(), advanced from bag timestamps rather
    than wall-clock time -- see module docstring."""

    def __init__(self) -> None:
        self.t_ns = 0

    def now(self) -> rclpy.time.Time:
        return rclpy.time.Time(nanoseconds=self.t_ns)


def read_messages(bag_path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        ),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=list(TOPIC_TYPES)))
    while reader.has_next():
        topic, data, t = reader.read_next()
        yield topic, deserialize_message(data, TOPIC_TYPES[topic]), t


def build_node() -> tuple[PressToPin, ReplayClock]:
    rclpy.init(args=[
        "--ros-args",
        "-p", f"urdf_path:={GEN3_ARM_URDF}",
        "-p", f"commanded_torque_topic:={COMMANDED_TORQUE_TOPIC}",
        "-p", "subtract_model_gravity:=true",
    ])
    with unittest.mock.patch.object(
        ptp, "fetch_robot_description", lambda node, timeout_sec=3.0: None
    ), unittest.mock.patch.object(
        rclpy.client.Client, "wait_for_service",
        lambda self, timeout_sec=None: True,
    ):
        node = PressToPin()

    clock = ReplayClock()
    node.get_clock = lambda: clock  # see ReplayClock docstring

    node.sent_requests = []

    def fake_call_async(req):
        node.sent_requests.append(req)
        future = rclpy.task.Future()
        future.set_result(
            AddSpring.Response(success=True, message="ok",
                                id=len(node.sent_requests))
        )
        return future

    node._add_spring_client.call_async = fake_call_async
    return node, clock


def install_latch_comparison(node: PressToPin, clock: ReplayClock) -> list[dict]:
    """
    Wrap _latch to log (bag time, new pick, old-heuristic pick) for every
    press that actually latches, without changing latching behavior itself.
    """
    latch_log: list[dict] = []
    orig_latch = PressToPin._latch

    def wrapped_latch(self, joint_idx, link, point, residual):
        over = np.abs(residual) > self._thresholds
        old_link = (
            self._contact_links[int(np.flatnonzero(over)[-1])]
            if over.any() else None
        )
        latch_log.append({
            "t_ns": clock.t_ns,
            "new_link": link,
            "old_link": old_link,
        })
        return orig_latch(self, joint_idx, link, point, residual)

    node._latch = types.MethodType(wrapped_latch, node)
    return latch_log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag_path")
    parser.add_argument(
        "--window", type=float, default=5.0,
        help="seconds after each test_marker to look for a latch (default 5.0)",
    )
    args = parser.parse_args()

    node, clock = build_node()
    latch_log = install_latch_comparison(node, clock)

    markers: list[tuple[int, str]] = []
    n_msgs = 0
    for topic, msg, t in read_messages(args.bag_path):
        clock.t_ns = t
        n_msgs += 1
        if topic == "/press_to_pin/test_marker":
            markers.append((t, msg.data))
        elif topic == COMMANDED_TORQUE_TOPIC:
            node._commanded_cb(msg)
        elif topic == "/joint_states":
            node._joint_state_cb(msg)

    print(f"Replayed {n_msgs} messages: {len(markers)} markers, "
          f"{len(latch_log)} latches.\n")

    window_ns = int(args.window * 1e9)
    matched = 0
    print(f"{'marker':<40} {'new_link':<14} {'old_link':<14} {'dt(s)':>6}")
    print("-" * 76)
    for t_marker, label in markers:
        candidates = [
            entry for entry in latch_log
            if t_marker <= entry["t_ns"] <= t_marker + window_ns
        ]
        if not candidates:
            print(f"{label:<40} {'(none)':<14} {'(none)':<14} {'-':>6}")
            continue
        entry = candidates[0]
        dt = (entry["t_ns"] - t_marker) / 1e9
        print(f"{label:<40} {entry['new_link']:<14} "
              f"{str(entry['old_link']):<14} {dt:6.2f}")
        matched += 1

    print(f"\n{matched}/{len(markers)} markers matched to a latch within "
          f"{args.window}s.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
