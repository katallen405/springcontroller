"""
Offline unit tests for the workspace-calibration geometry/YAML/log helpers
(springcontroller_ui.study_workspace_config). No ROS, no rclpy -- pure
arithmetic and file I/O, same rationale as test_fk_helper.py's docstring.
"""
import csv
import math
import os
import random

import pytest
import yaml

from springcontroller_ui.study_workspace_config import (
    ALL_CONDITION_ORDERS,
    CM_TO_M,
    EYE_TARGET_Y_OFFSET_CM,
    SHARED_SPRING_RADIUS_M,
    assign_condition_order,
    compute_candidate_center,
    compute_condition_params,
    compute_eye_location,
    describe_candidate_center,
    log_event,
    log_measurement,
    validate_participant_id,
    write_condition_yaml,
)


def test_candidate_center_reaches_across_table_by_arm_length():
    # Participant sits along the table's long edge and reaches inward
    # across it (+y), not further along it (x stays at the seat's x).
    center = compute_candidate_center(seat_x=0.5, seat_y=-0.1, eye_height_cm=71.0, arm_length_cm=46.0)
    assert center["x"] == pytest.approx(0.5)
    assert center["y"] == pytest.approx(-0.1 + 46.0 * CM_TO_M + 0.1)


def test_eye_location_uses_seat_xy_not_reach_center():
    # The face is above the chair, not above wherever the reach-center
    # (arm's-length across the table) ended up -- independent of arm_length.
    eye = compute_eye_location(seat_x=0.5, seat_y=-0.1, eye_height_cm=71.0)
    assert eye["x"] == pytest.approx(0.5)
    assert eye["y"] == pytest.approx(-0.1 + EYE_TARGET_Y_OFFSET_CM * CM_TO_M)
    assert eye["z"] == pytest.approx(71.0 * CM_TO_M)


def test_eye_location_y_offset_much_smaller_than_a_full_arm_reach():
    eye = compute_eye_location(seat_x=0.0, seat_y=0.0, eye_height_cm=71.0)
    center = compute_candidate_center(seat_x=0.0, seat_y=0.0, eye_height_cm=71.0, arm_length_cm=46.0)
    assert eye["y"] < center["y"]


def test_candidate_center_z_is_half_eye_height_when_within_cone():
    # Long arm -> half-eye-height isn't more than 30 deg below eye level,
    # so it's used unclamped.
    center = compute_candidate_center(seat_x=0.0, seat_y=0.0, eye_height_cm=40.0, arm_length_cm=100.0)
    assert center["z"] == pytest.approx(40.0 * CM_TO_M / 2.0)


def test_candidate_center_z_clamped_to_30deg_cone_when_half_eye_height_too_steep():
    # Typical measurements -- half-eye-height would be a >30 deg look-down
    # at this reach, so z gets raised to exactly the 30 deg cutoff instead.
    eye_height_cm, arm_length_cm = 71.0, 46.0
    center = compute_candidate_center(seat_x=0.0, seat_y=0.0, eye_height_cm=eye_height_cm, arm_length_cm=arm_length_cm)
    half_eye_height_m = (eye_height_cm * CM_TO_M) / 2.0
    expected_z = (eye_height_cm * CM_TO_M) - (arm_length_cm * CM_TO_M) * math.tan(math.radians(30.0))
    assert center["z"] == pytest.approx(expected_z)
    assert center["z"] > half_eye_height_m  # raised above the naive half-height point
    elevation_deg = math.degrees(math.atan2((eye_height_cm * CM_TO_M) - center["z"], arm_length_cm * CM_TO_M))
    assert elevation_deg == pytest.approx(30.0)


def test_describe_candidate_center_flags_cone_clamp_when_used():
    eye_height_cm, arm_length_cm = 71.0, 46.0
    center = compute_candidate_center(seat_x=0.0, seat_y=-0.1, eye_height_cm=eye_height_cm, arm_length_cm=arm_length_cm)
    text = describe_candidate_center(0.0, -0.1, arm_length_cm, center)
    assert "cone-clamped" in text
    assert f"{center['z']:.3f}" in text


def test_describe_candidate_center_no_clamp_note_when_unclamped():
    eye_height_cm, arm_length_cm = 40.0, 100.0
    center = compute_candidate_center(seat_x=0.0, seat_y=0.0, eye_height_cm=eye_height_cm, arm_length_cm=arm_length_cm)
    text = describe_candidate_center(0.0, 0.0, arm_length_cm, center)
    assert "cone-clamped" not in text


def test_inner_radius_is_always_zero():
    # No dead-zone shell -- tip_spring always exerts some pull, regardless
    # of eye height / arm length.
    center = {"x": 0.0, "y": 0.0, "z": 10.16 * CM_TO_M}
    loose = compute_condition_params(center, eye_height_cm=76.0, arm_length_cm=100.0)
    assert loose["inner_radius"] == 0.0

    tight = compute_condition_params(center, eye_height_cm=76.0, arm_length_cm=8.0)
    assert tight["inner_radius"] == 0.0


def test_rest_length_is_zero():
    # rest_length is unconditionally 0 (not inner_radius) -- the position
    # spring always has a gentle pull toward the literal center of the
    # workspace, no dead-zone shell equilibrium.
    center = {"x": 0.0, "y": 0.0, "z": 10.16 * CM_TO_M}
    params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    assert params["rest_length"] == 0


def test_outer_radius_is_fixed_regardless_of_measurements():
    # outer_radius is a fixed SHARED_SPRING_RADIUS_M, not derived from
    # eye_height/arm_length -- both conditions always agree on the same
    # reach tolerance regardless of a given participant's measurements.
    center = {"x": 0.0, "y": 0.0, "z": 10.16 * CM_TO_M}

    a = compute_condition_params(center, eye_height_cm=76.0, arm_length_cm=100.0)
    assert a["outer_radius"] == pytest.approx(SHARED_SPRING_RADIUS_M)

    b = compute_condition_params(center, eye_height_cm=76.0, arm_length_cm=8.0)
    assert b["outer_radius"] == pytest.approx(SHARED_SPRING_RADIUS_M)


def test_outer_radius_ignores_ramp_margin():
    # ramp_margin_cm is accepted for call-site compatibility but no longer
    # affects outer_radius.
    center = {"x": 0.0, "y": 0.0, "z": 10.16 * CM_TO_M}
    with_margin = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0, ramp_margin_cm=5.0)
    default = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    assert with_margin["outer_radius"] == pytest.approx(default["outer_radius"])


def test_warns_when_center_outside_eye_level_cone():
    # Eyes far above the center and a short arm -> steep look-down angle.
    center = {"x": 0.0, "y": 0.0, "z": 0.0}
    params = compute_condition_params(center, eye_height_cm=100.0, arm_length_cm=15.0)
    assert any("eye level" in w for w in params["warnings"])


def test_no_eye_angle_warning_when_within_cone():
    # Eyes roughly level with the center -> ~0 deg elevation, well inside +-30.
    center = {"x": 0.0, "y": 0.0, "z": 71.0 * CM_TO_M}
    params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    assert not any("eye level" in w for w in params["warnings"])


def test_no_eye_angle_warning_at_exactly_the_30deg_clamp():
    # A center placed via compute_candidate_center's own 30 deg clamp
    # (i.e. exactly at the cutoff by construction) must never trip this
    # warning, even though the tan/atan2 round-trip can land it a hair
    # past 30.0 from floating-point error alone.
    eye_height_cm, arm_length_cm = 71.0, 46.0
    center = compute_candidate_center(
        seat_x=0.0, seat_y=0.0, eye_height_cm=eye_height_cm, arm_length_cm=arm_length_cm,
    )
    params = compute_condition_params(center, eye_height_cm=eye_height_cm, arm_length_cm=arm_length_cm)
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
    center = {"x": 0.5, "y": 0.0, "z": 10.16 * CM_TO_M}
    condition_params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    path = str(tmp_path / "position.yaml")

    write_condition_yaml(path, "tip_spring", _spring_params(center, condition_params))

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert params["spring_names"] == ["tip_spring"]
    spring = params["springs"]["tip_spring"]
    assert spring["link_name"] == "end_effector_link"
    assert spring["target"] == [0.5, 0.0, pytest.approx(10.16 * CM_TO_M)]
    assert spring["inner_radius"] == pytest.approx(condition_params["inner_radius"])
    assert "pose_spring_names" not in params


def test_write_condition_yaml_pose_only_has_no_tip_spring(tmp_path):
    center = {"x": 0.5, "y": 0.0, "z": 10.16 * CM_TO_M}
    condition_params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    path = str(tmp_path / "pose.yaml")
    pose_params = {
        "link_name": "end_effector_link",
        "local_point": [0.0, 0.0, 0.1],
        "local_face_normal": [0.0, 0.0, 1.0],
        "target": [0.5, 0.0, 71.0 * CM_TO_M],
        "stiffness": 2.0,
        "damping": 0.2,
        "position_center": [center["x"], center["y"], center["z"]],
        "position_radius": condition_params["outer_radius"],
        "position_stiffness": 10.0,
    }

    write_condition_yaml(
        path, include_pose=True, pose_name="face_participant",
        pose_params=pose_params,
    )

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    # No base spring at all -- position_center/position_radius below stand
    # in for where a tip spring would have gone, not a supplement to one.
    assert "springs" not in params
    assert "spring_names" not in params
    assert params["pose_spring_names"] == ["face_participant"]
    assert params["pose_springs"]["face_participant"]["target"] == [0.5, 0.0, pytest.approx(71.0 * CM_TO_M)]
    # ...bounded by the same center/radius the position condition's own
    # tip spring would use, just with no actual spring pulling toward it --
    # position_stiffness is what actually pulls the arm back once it
    # drifts past position_radius (see PoseSpring in virtual_spring.py).
    assert params["pose_springs"]["face_participant"]["position_center"] == [center["x"], center["y"], pytest.approx(center["z"])]
    assert params["pose_springs"]["face_participant"]["position_stiffness"] == 10.0


def test_write_condition_yaml_joint_spring_is_damping_only(tmp_path):
    # Position condition gets a damping-only joint_7 spring -- see
    # write_condition_yaml's docstring for why (tip_spring has zero
    # authority over joint_7's own rotation axis).
    center = {"x": 0.5, "y": 0.0, "z": 10.16 * CM_TO_M}
    condition_params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    path = str(tmp_path / "position.yaml")
    joint_spring_params = {"joint_name": "joint_7", "stiffness": 0.0, "damping": 0.3}

    write_condition_yaml(
        path, "tip_spring", _spring_params(center, condition_params),
        joint_spring_name="joint7_damper", joint_spring_params=joint_spring_params,
    )

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert params["joint_spring_names"] == ["joint7_damper"]
    joint_spring = params["joint_springs"]["joint7_damper"]
    assert joint_spring["joint_name"] == "joint_7"
    assert joint_spring["stiffness"] == 0.0
    assert joint_spring["damping"] == pytest.approx(0.3)


def test_write_condition_yaml_pose_condition_has_no_joint_spring(tmp_path):
    # The pose condition's own PoseSpring already drives joint_7 to aim
    # local_face_normal at the target -- a joint spring there would fight
    # that correction, so it's never passed for this condition.
    path = str(tmp_path / "pose.yaml")
    pose_params = {
        "link_name": "end_effector_link",
        "local_point": [0.0, 0.0, 0.1],
        "local_face_normal": [0.0, 1.0, 0.0],
        "target": [0.5, 0.0, 0.71],
        "stiffness": 5.0,
        "damping": 0.2,
        "position_center": [0.5, 0.0, 0.1016],
        "position_radius": 0.07,
        "position_stiffness": 10.0,
    }

    write_condition_yaml(path, include_pose=True, pose_name="face_participant", pose_params=pose_params)

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert "joint_spring_names" not in params
    assert "joint_springs" not in params


def test_write_condition_yaml_extra_params_merged_alongside_springs(tmp_path):
    center = {"x": 0.5, "y": 0.0, "z": 10.16 * CM_TO_M}
    condition_params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    path = str(tmp_path / "position.yaml")

    write_condition_yaml(
        path, "tip_spring", _spring_params(center, condition_params),
        extra_params={"participant_id": "P001", "condition_name": "position"},
    )

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert params["springs"]["tip_spring"]["link_name"] == "end_effector_link"
    assert params["participant_id"] == "P001"
    assert params["condition_name"] == "position"


def test_write_condition_yaml_kt_has_only_extra_params(tmp_path):
    # KT ("no torque controller") has no spring content at all -- the file
    # exists purely to carry rosbag routing (see gen3_spring.launch.py's
    # _make_record_rosbag_action).
    path = str(tmp_path / "KT.yaml")

    write_condition_yaml(
        path, extra_params={"participant_id": "P001", "condition_name": "KT"},
    )

    with open(path) as f:
        data = yaml.safe_load(f)
    params = data["/**"]["ros__parameters"]
    assert "springs" not in params
    assert "spring_names" not in params
    assert "pose_springs" not in params
    assert "pose_spring_names" not in params
    assert params == {"participant_id": "P001", "condition_name": "KT"}


def test_log_measurement_creates_header_once_and_appends(tmp_path):
    csv_path = str(tmp_path / "measurements.csv")
    center = {"x": 0.5, "y": 0.0, "z": 10.16 * CM_TO_M}
    condition_params = compute_condition_params(center, eye_height_cm=71.0, arm_length_cm=46.0)
    pose_target = {"x": 0.5, "y": 0.0, "z": 71.0 * CM_TO_M}

    log_measurement(
        csv_path, "P001", 71.0, 46.0, center, condition_params, pose_target,
        "position_P001.yaml", "pose_P001.yaml",
    )
    log_measurement(
        csv_path, "P002", 68.0, 43.0, center, condition_params, pose_target,
        "position_P002.yaml", "pose_P002.yaml",
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
    # Eyes far above the center and a short arm -> steep look-down angle,
    # outside the eye-level cone (the only warning compute_condition_params
    # can still produce -- see test_warns_when_center_outside_eye_level_cone).
    steep = {"x": 0.0, "y": 0.0, "z": 0.0}
    condition_params = compute_condition_params(steep, eye_height_cm=100.0, arm_length_cm=15.0)
    pose_target = {"x": 0.0, "y": 0.0, "z": 71.0 * CM_TO_M}

    log_measurement(
        csv_path, "P003", 100.0, 15.0, steep, condition_params, pose_target,
        "c1.yaml", "c2.yaml",
    )

    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert "eye level" in row["warnings"]


def test_validate_participant_id_accepts_ordinary_id():
    assert validate_participant_id("P001") is None


def test_validate_participant_id_rejects_empty():
    assert validate_participant_id("") is not None
    assert validate_participant_id("   ") is not None


def test_validate_participant_id_rejects_path_separator():
    assert validate_participant_id("../other") is not None
    assert validate_participant_id("a/b") is not None
    assert validate_participant_id("..") is not None


def test_assign_condition_order_raises_on_malformed_row(tmp_path):
    assignments_path = str(tmp_path / "condition_order.csv")
    with open(assignments_path, "w", newline="") as f:
        f.write("not_the_right,columns\nx,y\n")

    with pytest.raises(ValueError):
        assign_condition_order(assignments_path, "P001")


def test_assign_condition_order_block_of_6_is_a_permutation_of_all_orders(tmp_path):
    # Every consecutive block of 6 participants should see each of the 6
    # distinct (first, second, third) orderings exactly once, in random
    # (seeded here for determinism) order within the block.
    assignments_path = str(tmp_path / "condition_order.csv")
    rng = random.Random(42)

    orders = []
    for i in range(6):
        first, second, third, already = assign_condition_order(
            assignments_path, f"P{i:03d}", rng=rng,
        )
        assert already is False
        orders.append((first, second, third))

    assert sorted(orders) == sorted(ALL_CONDITION_ORDERS)

    # A 7th participant starts a fresh block -- again drawn from the full
    # set of 6 orderings, not constrained by the previous block.
    seventh, _, _, already = assign_condition_order(assignments_path, "P006", rng=rng)
    assert already is False
    assert seventh in {o[0] for o in ALL_CONDITION_ORDERS}


def test_assign_condition_order_two_blocks_both_balanced(tmp_path):
    assignments_path = str(tmp_path / "condition_order.csv")
    rng = random.Random(7)

    orders = [
        assign_condition_order(assignments_path, f"P{i:03d}", rng=rng)[:3]
        for i in range(12)
    ]

    first_block, second_block = orders[:6], orders[6:]
    assert sorted(first_block) == sorted(ALL_CONDITION_ORDERS)
    assert sorted(second_block) == sorted(ALL_CONDITION_ORDERS)


def test_assign_condition_order_is_idempotent_per_participant(tmp_path):
    assignments_path = str(tmp_path / "condition_order.csv")
    rng = random.Random(1)

    first, second, third, already = assign_condition_order(assignments_path, "P001", rng=rng)
    assert already is False

    # Same participant ID looked up again -- returns the same assignment,
    # doesn't advance the block or re-randomize.
    again_first, again_second, again_third, already = assign_condition_order(
        assignments_path, "P001", rng=rng,
    )
    assert (again_first, again_second, again_third) == (first, second, third)
    assert already is True

    # A second, different participant still gets a fresh slot in the
    # current block, unaffected by the repeat lookup above.
    second_first, second_second, second_third, already = assign_condition_order(
        assignments_path, "P002", rng=rng,
    )
    assert (second_first, second_second, second_third) != (first, second, third)
    assert already is False


def test_log_event_creates_header_once_and_appends(tmp_path):
    csv_path = str(tmp_path / "event_log.csv")

    log_event(csv_path, "Trial start", "KT side 1")
    log_event(csv_path, "Trial end", "KT side 1", "participant asked to pause briefly")

    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["event"] == "Trial start"
    assert rows[0]["condition"] == "KT side 1"
    assert rows[0]["notes"] == ""
    assert rows[1]["event"] == "Trial end"
    assert rows[1]["notes"] == "participant asked to pause briefly"
    # Header appears exactly once, not duplicated per append.
    with open(csv_path) as f:
        header_lines = [l for l in f.readlines() if l.startswith("timestamp,")]
    assert len(header_lines) == 1


def test_log_event_notes_is_a_separate_column_not_tied_to_other(tmp_path):
    # notes is independent of event_text/condition -- present (or not)
    # regardless of whether either dropdown's value happens to be "Other".
    csv_path = str(tmp_path / "event_log.csv")
    log_event(csv_path, "Other", "Other", "ran out of paint, improvised")

    with open(csv_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert row["event"] == "Other"
    assert row["condition"] == "Other"
    assert row["notes"] == "ran out of paint, improvised"


def test_log_event_returns_iso_timestamp(tmp_path):
    csv_path = str(tmp_path / "event_log.csv")
    timestamp = log_event(csv_path, "Participant comment", "position side 2")
    # isoformat(timespec="seconds") -- parseable, and round-trips through
    # datetime.fromisoformat without raising.
    import datetime
    datetime.datetime.fromisoformat(timestamp)
