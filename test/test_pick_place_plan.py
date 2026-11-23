"""tools/pick_place_test.py planning and weighing helpers (plain Python; the
module imports ROS only inside run())."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "springcontroller" / "tools" / "pick_place_test.py"
spec = importlib.util.spec_from_file_location("pick_place_test", PATH)
pp = importlib.util.module_from_spec(spec)
sys.modules["pick_place_test"] = pp            # dataclasses look the module up
spec.loader.exec_module(pp)

START = [-0.45, -0.25, 0.45]
PICK = [-0.50, -0.20, 0.05]
PLACE = [-0.30, -0.40, 0.05]


def test_module_does_not_import_ros_at_top_level():
    import ast
    tree = ast.parse(PATH.read_text())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top if isinstance(n, ast.Import) for a in n.names]
    names += [n.module for n in top if isinstance(n, ast.ImportFrom)]
    assert not [m for m in names if m and m.split(".")[0] in (
        "rclpy", "tf2_ros", "springcontroller_interfaces", "std_srvs", "geometry_msgs")], names


def test_sequence_with_weighing():
    steps = pp.plan_moves(START, PICK, PLACE)
    assert [s.kind for s in steps] == ["move", "baseline", "move", "close", "move", "weigh",
                                       "move", "move", "open", "unload", "move"]
    moves = [s for s in steps if s.kind == "move"]
    up = np.array([0, 0, 0.08])
    targets = [np.array(s.point) for s in moves]
    np.testing.assert_allclose(targets, [np.add(PICK, up), PICK, np.add(PICK, up),
                                         np.add(PLACE, up), PLACE, np.add(PLACE, up)])
    # reach is checked only where the gripper closes and opens
    assert [s.check_reach for s in moves] == [False, True, False, False, True, False]


def test_given_mass_applies_before_lift():
    kinds = [s.kind for s in pp.plan_moves(START, PICK, PLACE, weigh=False)]
    assert "weigh" not in kinds and "baseline" in kinds
    assert kinds.index("payload") == kinds.index("close") + 1


@pytest.mark.parametrize("speed,descent", [(0.05, 0.02), (0.08, 0.01)])
def test_durations_keep_peak_speed(speed, descent):
    steps = pp.plan_moves(START, PICK, PLACE, speed=speed, descent=descent)
    prev = np.array(START)
    for s in steps:
        if s.kind != "move":
            continue
        dist = np.linalg.norm(np.array(s.point) - prev)
        vertical = abs(s.point[2] - prev[2]) > 1e-9 and np.allclose(s.point[:2], prev[:2])
        limit = descent if vertical else speed
        assert s.duration >= 1.5
        assert pp.MIN_JERK_PEAK * dist / s.duration <= limit + 1e-12
        prev = np.array(s.point)


@pytest.mark.parametrize("bad", [[-0.5, -0.2, -0.02], [0, 0, float("nan")], [0, 0]])
def test_rejects_bad_spots(bad):
    with pytest.raises(ValueError):
        pp.plan_moves(START, bad, PLACE)


def test_object_mass():
    m, note = pp.object_mass(2.0, 2.0 + 0.08 * pp.GRAVITY)
    assert m == pytest.approx(0.08) and "80 g" in note
    assert pp.object_mass(2.0, 2.05)[0] == 0.0                      # 5 g: friction-level
    assert pp.object_mass(2.0, 1.0)[0] == 0.0                       # negative
    m, note = pp.object_mass(0.0, 15.0, max_kg=1.0)
    assert m == 0.0 and "limit" in note


def test_cli_rejects_half_given_spots():
    with pytest.raises(SystemExit):
        pp.main(["--pick", "0", "0", "0.1"])
    with pytest.raises(SystemExit):
        pp.main(["--object-mass", "5"])
