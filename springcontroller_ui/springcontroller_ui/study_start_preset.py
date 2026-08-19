"""
study_start_preset.py

Pure read/write helpers for the "study start location" joint-angle preset
used by the study control panel's move-to-start / set-current-as-start
buttons. No ROS imports, so this is testable without rclpy or a running
node -- see test/test_fk_helper.py's sibling test file for the ROS-free
unit tests.
"""
from __future__ import annotations

import datetime
import os
from typing import Optional

import yaml


def load_preset(path: str) -> Optional[dict]:
    """
    Load a saved study-start preset.

    Returns None if the file doesn't exist yet, or if it exists but is
    unreadable/malformed -- both cases mean "no valid preset available",
    which callers should treat the same way (tell the user to capture one).
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if "joint_names" not in data or "joint_angles" not in data:
        return None
    if len(data["joint_names"]) != len(data["joint_angles"]):
        return None
    return data


def save_preset(path: str, joint_names: list[str], joint_angles: list[float]) -> None:
    """
    Save the current pose as the study-start preset, overwriting any
    existing file.

    Writes to a temp file in the same directory and renames it into place
    (os.replace is atomic on POSIX), so a crash mid-write can't leave a
    truncated/corrupt preset file behind for the next load_preset() to trip
    over.
    """
    path = os.path.expanduser(path)
    data = {
        "joint_names": list(joint_names),
        "joint_angles": [float(a) for a in joint_angles],
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f)
    os.replace(tmp_path, path)
