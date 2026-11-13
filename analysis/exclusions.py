"""
exclusions.py

Shared --exclude PID:CONDITION[:SIDE] handling for the analysis scripts,
for periods known to be invalid for reasons outside the data, e.g.
p10:Pose (experimenter error: the arm wasn't reset to the start pose
before the condition began, leaving nothing to adjust).

No third-party imports, so scripts run under plain system python3 can
use it.
"""

from __future__ import annotations

import argparse

HELP = ("Drop an invalid period before analysis (repeatable), as "
        "PID:CONDITION or PID:CONDITION:SIDE, e.g. p10:Pose.")


def norm_pid(pid) -> str:
    """'p10', 'P10' and '10' all name the same participant."""
    pid = str(pid).strip().lower()
    return pid if pid.startswith("p") else f"p{pid}"


def parse_exclusion(spec: str) -> tuple:
    """'p10:Pose' -> ('p10', 'pose', None); 'p10:Pose:2' -> (..., '2').
    Raises ArgumentTypeError so argparse reports a bad spec cleanly."""
    parts = spec.split(":")
    if len(parts) not in (2, 3) or not all(p.strip() for p in parts):
        raise argparse.ArgumentTypeError(
            f"--exclude {spec!r}: expected PID:CONDITION or PID:CONDITION:SIDE")
    side = parts[2].strip() if len(parts) == 3 else None
    return (norm_pid(parts[0]), parts[1].strip().lower(), side)


def matches(exclusion: tuple, pid, condition, side) -> bool:
    p, c, s = exclusion
    return (norm_pid(pid) == p and str(condition).strip().lower() == c
            and (s is None or str(side).strip() == s))


def describe(exclusion: tuple) -> str:
    p, c, s = exclusion
    return f"{p}:{c}" + (f":{s}" if s else "")


def add_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exclude", action="append", default=[], type=parse_exclusion,
                        metavar="PID:CONDITION[:SIDE]", help=HELP)
