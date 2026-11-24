#!/usr/bin/env python3
"""Lag between commanded joint torque and the measured (sensor) joint torque,
by cross-correlation of the detrended signals in a bag window.

Usage: bag_torque_lag.py BAG T0 T1   (bag seconds, during an oscillation; ROS sourced)
Used for docs/gen2_stability_analysis.md (52-64 ms on 2026-11-23)."""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

bag, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri=bag, storage_id=""), rosbag2_py.ConverterOptions("", ""))
types = {t.name: t.type for t in r.get_all_topics_and_types()}
cmd_t, cmd, js_t, eff, pos = [], [], [], [], []
r.set_filter(rosbag2_py.StorageFilter(topics=["/gen2/joint_torque_command",
                                              "/gen2/joint_states_lowlevel"]))
start = None
while r.has_next():
    topic, data, t = r.read_next()
    t *= 1e-9
    start = t if start is None else start
    if not t0 <= t - start <= t1:
        continue
    m = deserialize_message(data, get_message(types[topic]))
    if topic == "/gen2/joint_torque_command":
        cmd_t.append(t - start)
        cmd.append(list(m.data[:7]))
    else:
        js_t.append(t - start)
        eff.append(list(m.effort[:7]))
        pos.append(list(m.position[:7]))
cmd_t, cmd, js_t, eff, pos = map(np.array, (cmd_t, cmd, js_t, eff, pos))
grid = np.arange(max(cmd_t[0], js_t[0]), min(cmd_t[-1], js_t[-1]), 0.002)   # 2 ms grid


def detrend(x):
    return x - np.convolve(x, np.ones(51) / 51, mode="same")


for j in (0, 2, 4):
    c = detrend(np.interp(grid, cmd_t, cmd[:, j]))[60:-60]
    e = detrend(np.interp(grid, js_t, eff[:, j]))[60:-60]
    p = detrend(np.interp(grid, js_t, pos[:, j]))[60:-60]
    lags = np.arange(-50, 51)                                    # +-100 ms
    xc = [np.corrcoef(c[max(0, -k):len(c) - max(0, k)], e[max(0, k):len(e) - max(0, -k)])[0, 1]
          for k in lags]
    k = lags[int(np.argmax(np.abs(xc)))]
    print(f"joint {j + 1}: measured torque follows the command by {k * 2:+d} ms "
          f"(correlation {xc[int(np.argmax(np.abs(xc)))]:+.2f}); "
          f"command amplitude {c.std():.2f} N*m, measured {e.std():.2f} N*m, "
          f"position {p.std() * 1000:.1f} mrad")
