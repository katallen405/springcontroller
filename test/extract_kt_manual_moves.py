#!/usr/bin/env python3
"""
extract_kt_manual_moves.py

Offline extraction of manual-guidance ("KT" condition) motion episodes from a
recorded rosbag. The Gen3's own wrist-button admittance/impedance modes have
no event, getter, or notification anywhere in the Kortex API (checked against
the vendored kortex_api headers) -- so there's no way to directly capture a
button press. Instead this proxies it from /kinova/joint_states_lowlevel
(already recorded by gen3_spring.launch.py for every condition, including
KT): the arm reports exactly 0 rad/s on every joint while no button is held
("no buttons = position hold", see study_procedure.txt), so any joint
velocity clearing a low threshold marks a manual-move episode.

/gen3_torque_control/move_status (also already recorded) is used to exclude
any window where the arm was moving under its own MOVING state instead of
the participant's hands -- e.g. an operator using the study control panel's
move-to-position controls mid－recording. In practice the one-time "move
arm to study start location" step happens during per-participant
calibration, before the per-condition relaunch that starts the KT bag, so it
usually isn't present in the bag at all -- but the exclusion is unconditional
so a bag where it does appear doesn't get misattributed.

Validated against real KT pilot data (~/gen3_study_data/test4/
KT_2026_09_04-15_35_37): at rest the reported velocity is exactly 0.0 rad/s
(not just small), so the extracted episodes are insensitive to the exact
threshold across a wide range (0.02-0.15 rad/s all agree closely) -- 0.05
rad/s (~3 deg/s) is the default.

Usage:
    python3 test/extract_kt_manual_moves.py <bag_path> [--threshold RAD_S] [--csv OUT_CSV]

Example:
    python3 test/extract_kt_manual_moves.py \\
        ~/gen3_study_data/P003/KT_2026_09_10-14_20_00 --csv P003_KT_moves.csv
"""

from __future__ import annotations

import argparse
import csv as csv_module

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState
from std_msgs.msg import String

JOINT_STATES_TOPIC = "/kinova/joint_states_lowlevel"
MOVE_STATUS_TOPIC = "/gen3_torque_control/move_status"


def read_bag(bag_path: str):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[JOINT_STATES_TOPIC, MOVE_STATUS_TOPIC]))

    joint_samples = []
    move_status_events = []
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns / 1e9
        if topic == JOINT_STATES_TOPIC:
            msg = deserialize_message(data, JointState)
            joint_samples.append((t, list(msg.velocity)))
        elif topic == MOVE_STATUS_TOPIC:
            msg = deserialize_message(data, String)
            move_status_events.append((t, msg.data))
    return joint_samples, move_status_events


def moving_windows(move_status_events, last_t: float):
    """[(start, end), ...] spans where move_status == MOVING, e.g. a UI-driven
    move-to-position -- these are excluded, not counted as manual moves."""
    windows = []
    start = None
    for t, status in move_status_events:
        if status == "MOVING" and start is None:
            start = t
        elif status != "MOVING" and start is not None:
            windows.append((start, t))
            start = None
    if start is not None:
        windows.append((start, last_t))
    return windows


def extract_events(joint_samples, windows, threshold: float):
    def excluded(t):
        return any(a <= t <= b for a, b in windows)

    events = []
    in_event = False
    ev_start = ev_peak = ev_peak_joint = None
    for t, vel in joint_samples:
        if excluded(t):
            if in_event:
                events.append((ev_start, t, ev_peak, ev_peak_joint))
                in_event = False
            continue
        peak = max((abs(v), i) for i, v in enumerate(vel))
        flagged = peak[0] > threshold
        if flagged and not in_event:
            in_event, ev_start, ev_peak, ev_peak_joint = True, t, peak[0], peak[1]
        elif flagged and in_event:
            if peak[0] > ev_peak:
                ev_peak, ev_peak_joint = peak
        elif not flagged and in_event:
            events.append((ev_start, t, ev_peak, ev_peak_joint))
            in_event = False
    if in_event:
        events.append((ev_start, joint_samples[-1][0], ev_peak, ev_peak_joint))
    return events


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bag_path")
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Per-joint |velocity| (rad/s) above which the arm is considered "
             "manually moved. Default 0.05 -- see module docstring.",
    )
    parser.add_argument("--csv", help="Optional path to write events as CSV.")
    args = parser.parse_args()

    joint_samples, move_status_events = read_bag(args.bag_path)
    if not joint_samples:
        print(f"No {JOINT_STATES_TOPIC} samples found in {args.bag_path}")
        return

    t0 = joint_samples[0][0]
    windows = moving_windows(move_status_events, joint_samples[-1][0])
    if windows:
        print(f"Excluding {len(windows)} move_status=MOVING window(s):")
        for a, b in windows:
            print(f"  t={a - t0:.2f}s .. {b - t0:.2f}s (dur {b - a:.2f}s)")

    events = extract_events(joint_samples, windows, args.threshold)
    print(f"\n{len(events)} manual-move episode(s) at threshold={args.threshold} rad/s:")
    for start, end, peak, peak_joint in events:
        print(
            f"  t={start - t0:7.2f}s .. {end - t0:7.2f}s  "
            f"dur={end - start:5.2f}s  peak={peak:.3f} rad/s (joint{peak_joint})"
        )

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv_module.writer(f)
            writer.writerow(["start_t_rel_s", "end_t_rel_s", "duration_s", "peak_vel_rad_s", "peak_joint"])
            for start, end, peak, peak_joint in events:
                writer.writerow([f"{start - t0:.3f}", f"{end - t0:.3f}", f"{end - start:.3f}", f"{peak:.4f}", peak_joint])
        print(f"\nWrote {len(events)} events to {args.csv}")


if __name__ == "__main__":
    main()
