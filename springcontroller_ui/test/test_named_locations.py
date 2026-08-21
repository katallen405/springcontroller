"""
Offline unit tests for named_locations.py (springcontroller_ui.named_locations).
No ROS, no rclpy -- pure arithmetic and file I/O, same rationale as
test_study_workspace_config.py's docstring.
"""
import os

from springcontroller_ui.named_locations import load_locations, save_location


def test_load_locations_missing_file_returns_empty_dict(tmp_path):
    assert load_locations(str(tmp_path / "nope.yaml")) == {}


def test_save_then_load_round_trips(tmp_path):
    path = str(tmp_path / "locations.yaml")
    save_location(path, "block", "end_effector_link", [0.0, 0.0, 0.13])

    locations = load_locations(path)
    assert set(locations.keys()) == {"block"}
    assert locations["block"]["link_name"] == "end_effector_link"
    assert locations["block"]["offset"] == [0.0, 0.0, 0.13]
    assert "saved_at" in locations["block"]


def test_saving_a_second_location_keeps_the_first(tmp_path):
    path = str(tmp_path / "locations.yaml")
    save_location(path, "block", "end_effector_link", [0.0, 0.0, 0.13])
    save_location(path, "home", "base_link", [0.2, 0.0, 0.4])

    locations = load_locations(path)
    assert set(locations.keys()) == {"block", "home"}
    assert locations["home"]["link_name"] == "base_link"


def test_saving_same_name_again_overwrites_in_place(tmp_path):
    path = str(tmp_path / "locations.yaml")
    save_location(path, "block", "end_effector_link", [0.0, 0.0, 0.13])
    save_location(path, "block", "end_effector_link", [0.0, 0.0, 0.2])

    locations = load_locations(path)
    assert set(locations.keys()) == {"block"}
    assert locations["block"]["offset"] == [0.0, 0.0, 0.2]


def test_load_locations_malformed_file_returns_empty_dict(tmp_path):
    path = tmp_path / "locations.yaml"
    path.write_text("not a mapping with locations\n")
    assert load_locations(str(path)) == {}


def test_load_locations_expands_user_path(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    save_location("~/locations.yaml", "block", "end_effector_link", [0.0, 0.0, 0.13])
    assert os.path.isfile(tmp_path / "locations.yaml")
    assert "block" in load_locations("~/locations.yaml")
