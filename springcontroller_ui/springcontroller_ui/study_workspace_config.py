"""
study_workspace_config.py

Pure geometry + YAML-writing helpers for the study control panel's
workspace-calibration flow: turn two participant measurements (seated eye
height, elbow-to-fingertip arm length) plus a live-approved target point
into the two spring-config YAML files used by study conditions 1 and 2,
and log the measurement session against the participant ID. No ROS
imports, so this is testable without rclpy or a running node -- see
study_start_preset.py's docstring for the same rationale.

All lengths in this module are meters (matching the rest of this repo's
spring configs, e.g. gen3_springs.yaml) except where a `_cm` suffix marks
a centimeter-denominated input straight from a tape measure/meter stick.
"""
from __future__ import annotations

import csv
import datetime
import math
import os

import yaml

CM_TO_M = 0.01

# Elbow-height band above the table (world z=0, see gen3_collision_scene.yaml),
# and the eye-level cone the workspace must stay within -- both are fixed
# ergonomic specs for this study, independent of what unit measurements are
# entered in.
HEIGHT_BAND_CM = 20.32  # 8 inches
EYE_ANGLE_DEG = 30.0

# Condition 2's orientation-spring "look-at" target sits in front of the
# participant's actual face, not above wherever the reach-center (used for
# condition 1's position spring) happens to land -- those are two different
# points on the participant's body. This is a fixed offset from the seat
# mark along the same +y axis compute_candidate_center reaches across,
# just much shorter than a full arm's reach. Confirmed wrong 2026-08-21:
# it had been reusing the approved reach-center's x/y outright.
EYE_TARGET_Y_OFFSET_CM = 20.0


def _cm(value_cm: float) -> float:
    return value_cm * CM_TO_M


def compute_candidate_center(
    seat_x: float, seat_y: float, eye_height_cm: float, arm_length_cm: float,
) -> dict:
    """
    Initial candidate for the workspace center, before the human-in-the-loop
    push/adjust step. x/y: forearm's length across the table from the seat,
    plus a fixed 10cm clearance margin (+y, the table's short axis -- the
    participant sits along its long edge and reaches inward across it, not
    further along it). z: a third of the way from the table (world z=0) to
    eye height, unless that's more than 30 deg below eye level at the
    measured arm length (reach distance), in which case it's raised to
    exactly the 30 deg cutoff instead -- one-third-eye-height is always
    below eye level, never above, so this is a one-sided clamp.
    """
    eye_height_m = _cm(eye_height_cm)
    arm_length_m = _cm(arm_length_cm)
    z = eye_height_m / 3.0
    min_z = eye_height_m - arm_length_m * math.tan(math.radians(EYE_ANGLE_DEG))
    z = max(z, min_z)
    return {
        "x": seat_x,
        "y": seat_y + arm_length_m + 0.1,
        "z": z,
    }


def compute_eye_location(seat_x: float, seat_y: float, eye_height_cm: float) -> dict:
    """
    Condition 2's orientation-spring target: directly in front of the
    participant's face, at the chair's x, the chair's y plus a fixed
    EYE_TARGET_Y_OFFSET_CM (toward the table -- same direction as
    compute_candidate_center's reach, just much shorter), and eye height
    above the table. Independent of arm_length_cm and of wherever the
    reach-center (condition 1's position-spring target) ended up after
    the human-in-the-loop push/adjust step -- the participant's face
    doesn't move just because their hand did.
    """
    return {
        "x": seat_x,
        "y": seat_y + _cm(EYE_TARGET_Y_OFFSET_CM),
        "z": _cm(eye_height_cm),
    }


def compute_condition_params(
    center: dict, eye_height_cm: float, arm_length_cm: float, ramp_margin_cm: float = 2.54,
) -> dict:
    """
    Derive the condition-1 dead-zone sphere from the *final, approved*
    center point.

    inner_radius = min(half the 8" height band, the +-30 deg eye-level cone
    converted to linear distance at the measured arm length) -- whichever
    bound is tighter. outer_radius adds a small ramp margin rather than a
    hard on/off. rest_length == inner_radius keeps VirtualSpring's force
    continuous at the dead-zone boundary instead of jumping by
    stiffness*inner_radius the instant a target leaves the dead zone (see
    virtual_spring.py's force computation).

    Returns radii/rest_length in meters plus a list of human-readable
    warning strings (empty if none) -- these are advisory, not blocking,
    since a person always reviews the candidate live before finalizing.
    """
    arm_length_m = _cm(arm_length_cm)
    eye_height_m = _cm(eye_height_cm)

    inner_radius = min(_cm(HEIGHT_BAND_CM / 2.0), arm_length_m * math.tan(math.radians(EYE_ANGLE_DEG)))
    outer_radius = inner_radius + _cm(ramp_margin_cm)
    rest_length = inner_radius

    warnings: list[str] = []

    height_above_table = center["z"]
    if not (0.0 <= height_above_table <= _cm(HEIGHT_BAND_CM)):
        warnings.append(
            f"Center height {height_above_table / CM_TO_M:.1f}cm above the table is "
            f"outside the confirmed [0, {HEIGHT_BAND_CM:.1f}cm] elbow-height band."
        )

    elevation_deg = math.degrees(math.atan2(eye_height_m - height_above_table, arm_length_m))
    if abs(elevation_deg) > EYE_ANGLE_DEG:
        warnings.append(
            f"Center is {elevation_deg:.1f} deg from eye level, outside the "
            f"confirmed +-{EYE_ANGLE_DEG:.0f} deg cone."
        )

    return {
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "rest_length": rest_length,
        "warnings": warnings,
    }


def _atomic_write_yaml(path: str, data: dict) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    os.replace(tmp_path, path)


def write_condition_yaml(
    path: str,
    spring_name: str,
    spring_params: dict,
    include_orientation: bool = False,
    orientation_name: str = "",
    orientation_params: dict | None = None,
) -> None:
    """
    Write one condition's springs.yaml, matching the schema
    virtual_spring_node._load_springs_from_params expects (see
    gen3_springs.yaml / gen3_orientation_spring_test.yaml). `spring_params`
    keys: link_name, local_point, target, stiffness, damping, rest_length,
    inner_radius, outer_radius. `orientation_params` keys (condition 2
    only): link_name, local_point, local_face_normal, target, stiffness,
    damping.
    """
    params: dict = {
        "spring_names": [spring_name],
        "springs": {spring_name: dict(spring_params)},
    }
    if include_orientation:
        params["orientation_spring_names"] = [orientation_name]
        params["orientation_springs"] = {orientation_name: dict(orientation_params)}

    data = {"/**": {"ros__parameters": params}}
    _atomic_write_yaml(path, data)


def log_measurement(
    csv_path: str,
    participant_id: str,
    eye_height_cm: float,
    arm_length_cm: float,
    center: dict,
    condition_params: dict,
    orientation_target: dict,
    condition1_path: str,
    condition2_path: str,
) -> None:
    """
    Append one row to the shared measurement log, for later comparison
    against robot/rosbag data. Creates the file (with a header row) on
    first use; every call after that only appends -- never truncates or
    rewrites prior participants' rows.
    """
    csv_path = os.path.expanduser(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.isfile(csv_path)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "eye_height_cm": eye_height_cm,
        "arm_length_cm": arm_length_cm,
        "center_x": center["x"],
        "center_y": center["y"],
        "center_z": center["z"],
        "inner_radius": condition_params["inner_radius"],
        "outer_radius": condition_params["outer_radius"],
        "rest_length": condition_params["rest_length"],
        "orientation_target_x": orientation_target["x"],
        "orientation_target_y": orientation_target["y"],
        "orientation_target_z": orientation_target["z"],
        "warnings": "; ".join(condition_params["warnings"]),
        "condition1_path": condition1_path,
        "condition2_path": condition2_path,
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def assign_condition_order(assignments_path: str, participant_id: str) -> tuple[str, bool]:
    """
    Counterbalances which condition (condition1 vs condition2) a
    participant should run first: alternates based on how many
    participants have already been assigned, so roughly half the study
    runs condition1-first and half condition2-first regardless of
    enrollment order. Idempotent -- looking up a participant_id that's
    already been assigned returns that same order again (never
    re-randomizes/flip-flops on a repeat visit to the panel).

    Kept in its own small append-only CSV rather than reusing
    measurements.csv (see log_measurement) since assignment needs to
    happen *before* the workspace-calibration flow that produces a
    measurements.csv row -- an experimenter may open the panel to check
    a participant's assigned order well before (or without ever)
    finalizing that participant's condition files that session.

    Returns (first_condition, already_assigned).
    """
    assignments_path = os.path.expanduser(assignments_path)
    os.makedirs(os.path.dirname(assignments_path), exist_ok=True)

    rows: list[dict] = []
    if os.path.isfile(assignments_path):
        with open(assignments_path, newline="") as f:
            rows = list(csv.DictReader(f))

    for row in rows:
        if row["participant_id"] == participant_id:
            return row["first_condition"], True

    first_condition = "condition1" if len(rows) % 2 == 0 else "condition2"
    new_row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "first_condition": first_condition,
    }
    is_new = not os.path.isfile(assignments_path)
    with open(assignments_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(new_row)
        f.flush()

    return first_condition, False
