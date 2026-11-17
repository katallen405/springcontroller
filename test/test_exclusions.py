"""Tests for analysis/exclusions.py (shared --exclude handling)."""

import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "analysis"))
import exclusions as ex  # noqa: E402


def test_parse_normalizes_pid_and_condition():
    assert ex.parse_exclusion("p10:Pose") == ("p10", "pose", None)
    assert ex.parse_exclusion("10:POSE") == ("p10", "pose", None)
    assert ex.parse_exclusion("P10:pose:2") == ("p10", "pose", "2")


@pytest.mark.parametrize("bad", ["p10", "p10:", ":Pose", "p10:Pose:2:x", ""])
def test_parse_rejects_malformed(bad):
    with pytest.raises(argparse.ArgumentTypeError):
        ex.parse_exclusion(bad)


def test_matches_condition_and_optional_side():
    whole = ex.parse_exclusion("p10:Pose")
    one_side = ex.parse_exclusion("p10:Pose:2")
    assert ex.matches(whole, "p10", "Pose", "1")
    assert ex.matches(whole, "10", "pose", 2)       # bag-folder pid, int side
    assert not ex.matches(whole, "p1", "Pose", "1")  # p1 is not p10
    assert not ex.matches(whole, "p10", "Position", "1")
    assert ex.matches(one_side, "p10", "Pose", "2")
    assert not ex.matches(one_side, "p10", "Pose", "1")


def test_add_argument_is_repeatable():
    ap = argparse.ArgumentParser()
    ex.add_argument(ap)
    args = ap.parse_args(["--exclude", "p10:Pose", "--exclude", "p3:KT:1"])
    assert args.exclude == [("p10", "pose", None), ("p3", "kt", "1")]
    assert ap.parse_args([]).exclude == []
