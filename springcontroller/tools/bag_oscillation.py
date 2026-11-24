#!/usr/bin/env python3
"""Oscillation frequency and joint involvement in a Gen2 bag window.

Usage: bag_oscillation.py BAG T0 T1   (bag seconds; ROS sourced)
Per joint: detrended amplitude and dominant frequency of joint position,
spring torque and the relayed command; plus spring-torque -> command latency.
Used for docs/gen2_stability_analysis.md (3.7 Hz on 2026-11-23)."""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
want = ["/gen2/joint_states_lowlevel", "/virtual_spring_node/joint_torques",
        "/gen2/joint_torque_command"]
r.set_filter(rosbag2_py.StorageFilter(topics=want))
start = None
S = {w: ([], []) for w in want}
while r.has_next():
    topic, data, t = r.read_next()
    t *= 1e-9
    start = t if start is None else start
    rel = t - start
    if not t0 <= rel <= t1:
        continue
    m = deserialize_message(data, get_message(types[topic]))
    val = list(m.position[:7]) if topic.endswith("lowlevel") else (
        list(m.effort[:7]) if hasattr(m, "effort") else list(m.data[:7]))
    S[topic][0].append(rel)
    S[topic][1].append(val)
for topic, (ts, vs) in S.items():
    ts, vs = np.array(ts), np.array(vs)
    if len(ts) < 50:
        print(topic, "too few samples", len(ts))
        continue
    dt = np.median(np.diff(ts))
    print(f"{topic}: {len(ts)} msgs, {1 / dt:.0f} Hz")
    for j in range(7):
        x = vs[:, j] - np.convolve(vs[:, j], np.ones(25) / 25, mode="same")
        x = x[25:-25]
        f = np.fft.rfftfreq(len(x), d=dt)
        P = np.abs(np.fft.rfft(x)) ** 2
        k = np.argmax(P[1:]) + 1
        print(f"   j{j + 1}: amplitude {x.std():8.4f}  peak {f[k]:5.2f} Hz")
# Latency estimate: lag between the spring node's torque and the relayed command.
a, b = S["/virtual_spring_node/joint_torques"], S["/gen2/joint_torque_command"]
if a[0] and b[0]:
    ta, tb = np.array(a[0]), np.array(b[0])
    lags = [tb[np.searchsorted(tb, t)] - t for t in ta[:-5] if np.searchsorted(tb, t) < len(tb)]
    print(f"spring torque -> relayed command: median {np.median(lags) * 1000:.1f} ms")
