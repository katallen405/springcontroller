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
import itertools
import math
import os
import random
from typing import Optional

import yaml

CM_TO_M = 0.01

# Elbow-height band above the table (world z=0, see gen3_collision_scene.yaml),
# and the eye-level cone the workspace must stay within -- both are fixed
# ergonomic specs for this study, independent of what unit measurements are
# entered in.
HEIGHT_BAND_CM = 20.32  # 8 inches
EYE_ANGLE_DEG = 30.0

# Condition 2's pose-spring "look-at" target sits in front of the
# participant's actual face, not above wherever the reach-center (used for
# condition 1's position spring) happens to land -- those are two different
# points on the participant's body. This is a fixed offset from the seat
# mark along the same +y axis compute_candidate_center reaches across,
# just much shorter than a full arm's reach.
EYE_TARGET_Y_OFFSET_CM = 20.0

# Shared dead-zone radius for both conditions' springs (tip_spring's
# outer_radius, pose spring's position_radius) -- a fixed value rather than
# a per-participant computed one, so both conditions always agree on
# exactly the same reach tolerance regardless of a given participant's
# measurements.
SHARED_SPRING_RADIUS_M = 0.07


def _cm(value_cm: float) -> float:
    return value_cm * CM_TO_M


def validate_participant_id(participant_id: str) -> Optional[str]:
    """Returns an error message if participant_id is unsafe to use as a
    single path component under a study data dir, else None. Guards
    against a typo'd '/' or '..' writing/reading outside the intended
    per-participant directory, not just an empty string."""
    participant_id = participant_id.strip()
    if not participant_id:
        return "participant_id is required."
    if os.path.basename(participant_id) != participant_id or participant_id in (".", ".."):
        return "participant_id must be a single path component (no '/', '..', etc.)."
    return None


def compute_candidate_center(
    seat_x: float, seat_y: float, eye_height_cm: float, arm_length_cm: float,
) -> dict:
    """
    Initial candidate for the workspace center, before the human-in-the-loop
    push/adjust step. x/y: forearm's length across the table from the seat,
    plus a fixed 10cm clearance margin (+y, the table's short axis -- the
    participant sits along its long edge and reaches inward across it, not
    further along it). z: the higher of (half the way from the table
    (world z=0) to eye height) and (the 30 deg-below-eye-level cutoff at
    the measured arm length/reach distance) -- i.e. whichever puts the
    candidate closer to eye level.
    """
    eye_height_m = _cm(eye_height_cm)
    arm_length_m = _cm(arm_length_cm)
    z = eye_height_m / 2.0
    min_z = eye_height_m - arm_length_m * math.tan(math.radians(EYE_ANGLE_DEG))
    z = max(z, min_z)
    return {
        "x": seat_x,
        "y": seat_y + arm_length_m + 0.1,
        "z": z,
    }


def compute_eye_location(seat_x: float, seat_y: float, eye_height_cm: float) -> dict:
    """Condition 2's pose-spring target: directly in front of
    the participant's face, at the chair's x, the chair's y plus a
    fixed EYE_TARGET_Y_OFFSET_CM (toward the table), and eye height
    above the table. Independent of arm_length_cm and of wherever the
    reach-center ended up after the human-in-the-loop push/adjust step
    -- the participant's face doesn't move just because their hand
    did.
    """
    return {
        "x": seat_x,
        "y": seat_y + _cm(EYE_TARGET_Y_OFFSET_CM),
        "z": _cm(eye_height_cm),
    }


def compute_condition_params(
    center: dict, eye_height_cm: float, arm_length_cm: float, ramp_margin_cm: float = 2.54,
) -> dict:
    """No inner dead-zone shell for either condition: inner_radius is
    unconditionally 0, so tip_spring always exerts some pull. outer_radius
    is the fixed SHARED_SPRING_RADIUS_M, not computed per-participant --
    both conditions' springs (tip_spring's outer_radius, pose spring's
    position_radius) always agree on the same reach tolerance regardless
    of a given participant's measurements. ramp_margin_cm is accepted but
    unused -- kept for call-site/wire compatibility with
    FinalizeStudyConditions.srv rather than threading a signature change
    through the UI/service layer for a parameter with nothing left to do.

    Returns radii/rest_length in meters plus a list of human-readable
    warning strings (empty if none) -- these are advisory, not blocking,
    since a person always reviews the candidate live before finalizing.

    """
    arm_length_m = _cm(arm_length_cm)
    eye_height_m = _cm(eye_height_cm)

    inner_radius = 0.0
    outer_radius = SHARED_SPRING_RADIUS_M
    rest_length = 0

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



class _NoAliasDumper(yaml.SafeDumper):
    """
    ROS 2's params-file YAML parser (rcl_yaml_param_parser) doesn't support
    YAML anchors/aliases at all ("Will not support aliasing") -- but
    PyYAML's default SafeDumper auto-emits them whenever the *same*
    Python list/dict object (by identity) appears more than once in the
    data being dumped. write_condition_yaml's condition-2 output does
    exactly that: spring_params and pose_params both carry the
    same local_point list object (see _finalize_study_conditions_cb in
    orchestration_node.py).
    """
    def ignore_aliases(self, data):
        return True


def _atomic_write_yaml(path: str, data: dict) -> None:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        yaml.dump(data, f, Dumper=_NoAliasDumper, sort_keys=False)
    os.replace(tmp_path, path)


def write_condition_yaml(
    path: str,
    spring_name: str = "",
    spring_params: dict | None = None,
    include_pose: bool = False,
    pose_name: str = "",
    pose_params: dict | None = None,
    extra_params: dict | None = None,
) -> None:
    """
    Write one condition's springs.yaml, matching the schema
    virtual_spring_node._load_springs_from_params expects (see
    gen3_springs.yaml / gen3_pose_spring_test.yaml). `spring_params`
    keys: link_name, local_point, target, stiffness, damping, rest_length,
    inner_radius, outer_radius. `pose_params` keys (pose condition
    only): link_name, local_point, local_face_normal, target, stiffness,
    damping, position_center, position_radius -- the last two
    (EXPERIMENTAL, see PoseSpring in virtual_spring.py) are required by
    the loader just like the others, no safe default.

    spring_name/spring_params are optional -- the pose condition writes no
    base spring at all, since position_center/position_radius (inside
    pose_params) already define where a tip spring would have gone; the
    pose spring's own dead zone stands in for a real accompanying spring,
    not a supplement to one.

    extra_params is merged in unconditionally, independent of whatever
    spring content is or isn't present -- used to embed participant_id/
    condition_name so gen3_spring.launch.py can route the rosbag into the
    right study folder straight from `config:=` alone (see
    _make_record_rosbag_action), without also needing a separate
    participant_id:=/condition_name:= on the launch command line. This is
    the only content the KT condition's YAML carries at all (no springs,
    no pose_springs -- KT is "no torque controller").
    """
    params: dict = {}
    if spring_name:
        params["spring_names"] = [spring_name]
        params["springs"] = {spring_name: dict(spring_params)}
    if include_pose:
        params["pose_spring_names"] = [pose_name]
        params["pose_springs"] = {pose_name: dict(pose_params)}
    if extra_params:
        params.update(extra_params)

    data = {"/**": {"ros__parameters": params}}
    _atomic_write_yaml(path, data)


def log_measurement(
    csv_path: str,
    participant_id: str,
    eye_height_cm: float,
    arm_length_cm: float,
    center: dict,
    condition_params: dict,
    pose_target: dict,
    position_path: str,
    pose_path: str,
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
        "pose_target_x": pose_target["x"],
        "pose_target_y": pose_target["y"],
        "pose_target_z": pose_target["z"],
        "warnings": "; ".join(condition_params["warnings"]),
        "position_path": position_path,
        "pose_path": pose_path,
    }

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


# All 6 distinct orderings of the 3 study conditions -- see
# assign_condition_order's docstring for why the whole order (not just
# which one runs first) is block-randomized against this set.
ALL_CONDITION_ORDERS = list(itertools.permutations(["KT", "position", "pose"]))


def assign_condition_order(
    assignments_path: str, participant_id: str, rng: Optional[random.Random] = None,
) -> tuple[str, str, str, bool]:
    """
    Block-randomizes the FULL order (not just which condition runs first)
    a participant should run the three conditions (KT/position/pose) in.
    With 3 conditions there are 3! = 6 distinct orderings
    (ALL_CONDITION_ORDERS); every consecutive block of 6 participants gets
    each of those 6 orderings exactly once, in random order within the
    block -- picked by narrowing to whichever orderings haven't been used
    yet in the current (possibly partial) block and choosing uniformly at
    random among those, so a block that's just starting can land on any of
    the 6, and the block's last slot is whatever's left over.

    Purely advisory to the experimenter -- Kat runs the conditions
    herself, this just tells her the order, it doesn't drive any robot
    behavior.

    Idempotent -- looking up a participant_id that's already been
    assigned returns that same order again (never re-randomizes/
    flip-flops on a repeat visit to the panel).

    Kept in its own small append-only CSV rather than reusing
    measurements.csv (see log_measurement) since assignment needs to
    happen *before* the workspace-calibration flow that produces a
    measurements.csv row -- an experimenter may open the panel to check
    a participant's assigned order well before (or without ever)
    finalizing that participant's condition files that session.

    rng is normally left as None (a fresh random.Random() per call) --
    only tests inject a seeded one for determinism.

    Returns (first_condition, second_condition, third_condition, already_assigned).
    """
    assignments_path = os.path.expanduser(assignments_path)
    os.makedirs(os.path.dirname(assignments_path), exist_ok=True)

    rows: list[dict] = []
    if os.path.isfile(assignments_path):
        with open(assignments_path, newline="") as f:
            rows = list(csv.DictReader(f))

    required_columns = {"participant_id", "first_condition", "second_condition", "third_condition"}
    for row in rows:
        if not required_columns.issubset(row.keys()):
            raise ValueError(
                f"Malformed row in {assignments_path}: expected "
                f"{sorted(required_columns)} columns."
            )
        if row["participant_id"] == participant_id:
            return row["first_condition"], row["second_condition"], row["third_condition"], True

    block_start = (len(rows) // 6) * 6
    used_in_block = {
        (row["first_condition"], row["second_condition"], row["third_condition"])
        for row in rows[block_start:]
    }
    remaining = [order for order in ALL_CONDITION_ORDERS if order not in used_in_block]
    rng = rng or random.Random()
    first_condition, second_condition, third_condition = rng.choice(remaining)

    new_row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "participant_id": participant_id,
        "first_condition": first_condition,
        "second_condition": second_condition,
        "third_condition": third_condition,
    }
    is_new = not os.path.isfile(assignments_path)
    with open(assignments_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(new_row)
        f.flush()

    return first_condition, second_condition, third_condition, False


def log_event(csv_path: str, event_text: str, condition: str, notes: str = "") -> str:
    """
    Append one timestamped row to the per-participant session event log
    (the "Session timer & event log" panel's Log button) -- the
    timestamp is this server's clock, not whatever the browser's clock
    reads, so it stays comparable across the operator's other
    machine-local recordings (e.g. video). Same append-only,
    header-written-once-on-first-use pattern as log_measurement -- never
    truncates or rewrites prior rows.

    notes is always its own CSV column, independent of event_text/
    condition -- not folded into either of them, and not gated behind
    picking an "Other" option in either dropdown; it's free text the
    operator can attach to any row.

    Returns the ISO timestamp actually logged.
    """
    csv_path = os.path.expanduser(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    is_new = not os.path.isfile(csv_path)

    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    row = {"timestamp": timestamp, "event": event_text, "condition": condition, "notes": notes}

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

    return timestamp
