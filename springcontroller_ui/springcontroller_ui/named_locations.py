"""
named_locations.py

Pure read/write helpers for "named locations" -- reusable target points
defined as a world-frame x/y/z offset from a robot link's *current*
position (e.g. "block" = end_effector_link + [0, 0, 0.13]), shared between
the study workspace calibration panel and the add/remove springs panel.
Deliberately stores the (link_name, offset) definition, not a frozen
world point -- resolving it to an actual point happens at use time via
get_link_pose, so a named location tracks wherever its link currently is.

No ROS imports, so this is testable without rclpy or a running node --
same rationale as study_start_preset.py's docstring.
"""
from __future__ import annotations

import datetime
import os

import yaml


def load_locations(path: str) -> dict:
    """
    Load all saved named locations, keyed by name.

    Returns an empty dict if the file doesn't exist yet, or if it exists
    but is unreadable/malformed -- both cases mean "no named locations
    saved yet", which callers should treat the same way.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except Exception:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("locations"), dict):
        return {}
    locations = {}
    for name, entry in data["locations"].items():
        if not isinstance(entry, dict):
            continue
        if "link_name" not in entry or "offset" not in entry:
            continue
        if len(entry["offset"]) != 3:
            continue
        locations[name] = entry
    return locations


def save_location(path: str, name: str, link_name: str, offset: list[float]) -> None:
    """
    Save (or overwrite) one named location, leaving every other saved
    location in the file untouched.

    Writes to a temp file in the same directory and renames it into place
    (os.replace is atomic on POSIX), so a crash mid-write can't leave a
    truncated/corrupt file behind for the next load_locations() to trip
    over.
    """
    path = os.path.expanduser(path)
    locations = load_locations(path)
    locations[name] = {
        "link_name": link_name,
        "offset": [float(v) for v in offset],
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump({"locations": locations}, f, sort_keys=False)
    os.replace(tmp_path, path)
