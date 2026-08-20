"""
Offline unit tests for the workspace-calibration geometry/YAML/log helpers
(springcontroller_ui.study_workspace_config). No ROS, no rclpy -- pure
arithmetic and file I/O, same rationale as test_fk_helper.py's docstring.
"""
import csv
import math
import os

import pytest
import yaml

from springcontroller_ui.study_workspace_config import (
    INCH_TO_M,
    compute_candidate_center,
    compute_condition_params,
    log_measurement,
    write_condition_yaml,
)


def test_candidate_center_reaches_forward_by_arm_length():
    center = compute_candidate_center(seat_x=0.5, seat_y=-0.1, eye_height_in=28.0, arm_length_in=18.0)
    assert center["x"] == pytest.approx(0.5 + 18.0 * INCH_TO_M)
    assert center["y"] == pytest.approx(-0.1)
    # Midpoint of the confirmed 0-8" elbow-height band.
    assert center["z"] == pytest.approx(4.0 * INCH_TO_M)


def test_inner_radius_is_the_tighter_of_height_band_and_eye_cone():
    center = {"x": 0.0, "y": 0.0, "z": 4.0 * INCH_TO_M}

    # Long arm -> the eye-level cone is looser than the height band, so the
    # height-band half-width (4") wins.
    loose = compute_condition_params(center, eye_height_in=30.0, arm_length_in=40.0)
    assert loose["inner_radius"] == pytest.approx(4.0 * INCH_TO_M)

    # Short arm -> the eye-level cone is the tighter bound instead.
    arm_length_in = 3.0
    tight = compute_condition_params(center, eye_height_in=30.0, arm_length_in=arm_length_in)
    expected = (arm_length_in * INCH_TO_M) * math.tan(math.radians(30.0))
    assert tight["inner_radius"] == pytest.approx(expected)
    assert tight["inner_radius"] < 4.0 * INCH_TO_M


def test_rest_length_equals_inner_radius():
    center = {"x": 0.0, "y": 0.0, "z": 4.0 * INCH_TO_M}
    params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0)
    assert params["rest_length"] == params["inner_radius"]


def test_outer_radius_adds_ramp_margin():
    center = {"x": 0.0, "y": 0.0, "z": 4.0 * INCH_TO_M}
    params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0, ramp_margin_in=2.0)
    assert params["outer_radius"] == pytest.approx(params["inner_radius"] + 2.0 * INCH_TO_M)


def test_warns_when_center_height_outside_confirmed_band():
    too_high = {"x": 0.0, "y": 0.0, "z": 20.0 * INCH_TO_M}
    params = compute_condition_params(too_high, eye_height_in=28.0, arm_length_in=18.0)
    assert any("elbow-height band" in w for w in params["warnings"])


def test_no_height_warning_when_center_within_band():
    ok = {"x": 0.0, "y": 0.0, "z": 4.0 * INCH_TO_M}
    params = compute_condition_params(ok, eye_height_in=28.0, arm_length_in=18.0)
    assert not any("elbow-height band" in w for w in params["warnings"])


def test_warns_when_center_outside_eye_level_cone():
    # Eyes far above the center and a short arm -> steep look-down angle.
    center = {"x": 0.0, "y": 0.0, "z": 0.0}
    params = compute_condition_params(center, eye_height_in=40.0, arm_length_in=6.0)
    assert any("eye level" in w for w in params["warnings"])


def test_no_eye_angle_warning_when_within_cone():
    # Eyes roughly level with the center -> ~0 deg elevation, well inside +-30.
    center = {"x": 0.0, "y": 0.0, "z": 28.0 * INCH_TO_M}
    params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0)
    assert not any("eye level" in w for w in params["warnings"])


def _spring_params(center, condition_params):
    return {
        "link_name": "end_effector_link",
        "local_point": [0.0, 0.0, 0.1],
        "target": [center["x"], center["y"], center["z"]],
        "stiffness": 50.0,
        "damping": 5.0,
        "rest_length": condition_params["rest_length"],
        "inner_radius": condition_params["inner_radius"],
        "outer_radius": condition_params["outer_radius"],
    }


def test_write_condition_yaml_matches_loader_schema(tmp_path):
    center = {"x": 0.5, "y": 0.0, "z": 4.0 * INCH_TO_M}
    condition_params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0)
    path = str(tmp_path / "condition1.yaml")

    write_condition_yaml(path, "tip_spring", _spring_params(center, condition_params))

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert params["spring_names"] == ["tip_spring"]
    spring = params["springs"]["tip_spring"]
    assert spring["link_name"] == "end_effector_link"
    assert spring["target"] == [0.5, 0.0, pytest.approx(4.0 * INCH_TO_M)]
    assert spring["inner_radius"] == pytest.approx(condition_params["inner_radius"])
    assert "orientation_spring_names" not in params


def test_write_condition_yaml_with_orientation_adds_gaze_spring(tmp_path):
    center = {"x": 0.5, "y": 0.0, "z": 4.0 * INCH_TO_M}
    condition_params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0)
    path = str(tmp_path / "condition2.yaml")
    orientation_params = {
        "link_name": "end_effector_link",
        "local_point": [0.0, 0.0, 0.1],
        "local_face_normal": [0.0, 0.0, 1.0],
        "target": [0.5, 0.0, 28.0 * INCH_TO_M],
        "stiffness": 2.0,
        "damping": 0.2,
    }

    write_condition_yaml(
        path, "tip_spring", _spring_params(center, condition_params),
        include_orientation=True, orientation_name="face_participant",
        orientation_params=orientation_params,
    )

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    # Condition 2 keeps the identical dead-zone sphere from condition 1...
    assert params["springs"]["tip_spring"]["inner_radius"] == pytest.approx(condition_params["inner_radius"])
    # ...and adds the gaze spring on top.
    assert params["orientation_spring_names"] == ["face_participant"]
    assert params["orientation_springs"]["face_participant"]["target"] == [0.5, 0.0, pytest.approx(28.0 * INCH_TO_M)]


def test_log_measurement_creates_header_once_and_appends(tmp_path):
    csv_path = str(tmp_path / "measurements.csv")
    center = {"x": 0.5, "y": 0.0, "z": 4.0 * INCH_TO_M}
    condition_params = compute_condition_params(center, eye_height_in=28.0, arm_length_in=18.0)
    orientation_target = {"x": 0.5, "y": 0.0, "z": 28.0 * INCH_TO_M}

    log_measurement(
        csv_path, "P001", 28.0, 18.0, center, condition_params, orientation_target,
        "condition1_P001.yaml", "condition2_P001.yaml",
    )
    log_measurement(
        csv_path, "P002", 27.0, 17.0, center, condition_params, orientation_target,
        "condition1_P002.yaml", "condition2_P002.yaml",
    )

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["participant_id"] == "P001"
    assert rows[1]["participant_id"] == "P002"
    # Header appears exactly once, not duplicated per append.
    with open(csv_path) as f:
        header_lines = [l for l in f.readlines() if l.startswith("timestamp,")]
    assert len(header_lines) == 1


def test_log_measurement_records_warnings(tmp_path):
    csv_path = str(tmp_path / "measurements.csv")
    too_high = {"x": 0.0, "y": 0.0, "z": 20.0 * INCH_TO_M}
    condition_params = compute_condition_params(too_high, eye_height_in=28.0, arm_length_in=18.0)
    orientation_target = {"x": 0.0, "y": 0.0, "z": 28.0 * INCH_TO_M}

    log_measurement(
        csv_path, "P003", 28.0, 18.0, too_high, condition_params, orientation_target,
        "c1.yaml", "c2.yaml",
    )

    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert "elbow-height band" in row["warnings"]
