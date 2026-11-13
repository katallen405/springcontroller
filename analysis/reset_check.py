#!/usr/bin/env python3
"""
reset_check.py

Was the Gen3 reset to the study start pose before each adjustment period?
Reads joint_kinematics.py timeseries (any run: --pad 60, --whole-bags, ...)
plus the adjustment-times CSV, and lists for every participant x condition
x side:

  * the last reset move before the logged adjustment start (wall-clock
    start-end, from the commanded_move column, i.e. move_status == MOVING)
    and whether the arm actually reached the start pose at its end;
  * when the arm last LEFT the start pose before the adjustment start, and
    how far from it (max over joints, degrees) it was at the adjustment start;
  * a verdict, with a wall-clock time to check on video:
      reset OK               at the start pose at some point before the
                             logged start (no later reset needed)
      logged start before reset
                             a reset move begins AFTER the logged adjustment
                             start but before its end -- the logged start is
                             too early (e.g. p6 KT side 1)
      no reset found         never at the start pose before the logged start
                             in the data available ("lookback_s" says how much
                             there was; a short lookback is inconclusive)
      at start pose          already at the start pose at the logged start
    Side 2 normally isn't reset (it starts where side 1 ended), so its
    verdicts are informational.

"At the start pose" means every joint within --tol-deg (default ~1.15 deg,
0.02 rad) of the preset from --preset, or of any --alt-pose. Spring
conditions can move the arm off the pose between the reset and the
adjustment start; that shows up as "reset OK" with a left_start_pose_at
before the logged start, not as a missed reset.

Torque enable = start of possible spring-driven motion. If the run also
wrote events.csv (joint_kinematics.py reads /gen3_torque_control/status and
/virtual_spring_node/springs_updated), each period also gets the last
torque-ENABLED switch before its logged start, the springs loaded at that
moment, and how far the arm was from the start pose then. A spring
condition that was reset but had already left the start pose by the time
torque came on is flagged "off start pose at torque enable". KT runs with
our torque control disabled (Kinova's own hand-guiding), so this doesn't
apply to it.

Usage:
    python3 analysis/reset_check.py ~/CQHRI_joint_kinematics_allbags/timeseries \\
        --adjustment-times ~/deidentified_study_data/corrected/adjustment_times.csv \\
        --preset ~/deidentified_study_data/springcontroller_ui_study_start.yaml \\
        --alt-pose 1.11589,1.2001267805,-1.2322845838,1.3244174407,-2.7061365206,-1.9298189376,0.9780178806 \\
        --out reset_check.csv
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import os
import re
import sys
from collections import defaultdict
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_kinematics import JOINT_NAMES, _norm_participant, load_periods  # noqa: E402

FILE_RE = re.compile(r"^(?P<pid>[^_]+)_(?P<cond>[^_]+)_.*_side(?P<side>[^_.]+)\.csv$")
CANON = {"kt": "KT", "pose": "Pose", "position": "Position"}
FIELDS = ["participant_id", "condition", "side", "logged_adjustment_start",
          "verdict", "check_on_video_at", "last_reset_move", "reset_reached_start_pose",
          "left_start_pose_at", "torque_enabled_at", "left_before_torque_on_s",
          "deg_from_start_pose_at_torque_enable",
          "springs_at_torque_enable", "deg_from_start_pose_at_logged_start",
          "closest_deg_before_logged_start", "lookback_s"]


def load_preset(path: str) -> np.ndarray:
    """Joint angles from the study-start preset YAML (joint_angles + joint_names)."""
    import yaml
    with open(os.path.expanduser(path)) as f:
        d = yaml.safe_load(f)
    by_name = dict(zip(d["joint_names"], d["joint_angles"]))
    return np.array([float(by_name[n]) for n in JOINT_NAMES])


def parse_pose(spec: str) -> np.ndarray:
    vals = [float(x) for x in spec.split(",")]
    if len(vals) != len(JOINT_NAMES):
        raise argparse.ArgumentTypeError(f"--alt-pose needs {len(JOINT_NAMES)} comma-separated angles")
    return np.array(vals)


def load_events(path: str) -> dict:
    """{(pid 'pN', condition): [(epoch, kind, value), ...]} from events.csv."""
    out = defaultdict(list)
    if not path or not os.path.isfile(os.path.expanduser(path)):
        return out
    with open(os.path.expanduser(path), newline="") as f:
        for r in csv.DictReader(f):
            key = (_norm_key(r["participant_id"]), CANON.get(r["condition"].lower(), r["condition"]))
            out[key].append((float(r["epoch_s"]), r["kind"], r["value"]))
    for v in out.values():
        v.sort()
    return out


def load_timeseries(ts_dir: str) -> dict:
    """{(pid 'pN', condition): (t, q, commanded)} merged across bags and
    sides, sorted by time, duplicate timestamps dropped."""
    acc = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(os.path.expanduser(ts_dir), "*.csv"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        with open(path, newline="") as f:
            r = csv.reader(f)
            h = next(r)
            qi = [h.index(f"q_{n}") for n in JOINT_NAMES]
            ci = h.index("commanded_move") if "commanded_move" in h else None
            for row in r:
                acc[(_norm_key(m["pid"]), CANON.get(m["cond"].lower(), m["cond"]))].append(
                    (float(row[0]), int(row[ci]) if ci is not None else 0,
                     [float(row[i]) for i in qi]))
    out = {}
    for k, rows in acc.items():
        rows.sort(key=lambda x: x[0])
        t = np.array([x[0] for x in rows])
        keep = np.concatenate(([True], np.diff(t) > 0))
        out[k] = (t[keep], np.array([x[2] for x in rows])[keep], np.array([x[1] for x in rows])[keep])
    return out


def _norm_key(pid: str) -> str:
    return f"p{_norm_participant(pid)}"


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(first index, last index), ...] of each contiguous True run."""
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def deg_from(q: np.ndarray, poses: list[np.ndarray]) -> np.ndarray:
    """Per sample: max-over-joints angular distance (deg) to the NEAREST pose."""
    d = [np.degrees(np.abs((q - p + np.pi) % (2 * np.pi) - np.pi).max(axis=1)) for p in poses]
    return np.min(d, axis=0)


def check_period(t, q, cmd, start: float, end: float, poses, tol_deg: float, tz: str,
                 events=(), condition: str = "", enable_slack_s: float = 2.0) -> dict:
    iso = lambda x: datetime.datetime.fromtimestamp(x, ZoneInfo(tz)).strftime("%H:%M:%S")
    dev = deg_from(q, poses)
    at = dev <= tol_deg
    before = t < start
    i_start = min(int(np.searchsorted(t, start)), len(t) - 1)
    moves = [(t[a], t[b], b) for a, b in runs(cmd == 1)]
    row = {"lookback_s": round(start - t[0], 1) if t[0] < start else 0.0,
           "deg_from_start_pose_at_logged_start": round(float(dev[i_start]), 1),
           "closest_deg_before_logged_start": round(float(dev[before].min()), 1) if before.any() else "",
           "last_reset_move": "", "reset_reached_start_pose": "", "left_start_pose_at": ""}

    prior = [m for m in moves if m[1] < start]
    if prior:
        a, b, ib = prior[-1]
        row["last_reset_move"] = f"{iso(a)}-{iso(b)}"
        row["reset_reached_start_pose"] = "yes" if at[min(ib + 1, len(at) - 1)] or at[ib] else "no"
    at_runs_before = [(a, b) for a, b in runs(at & before)]
    left_t = t[at_runs_before[-1][1]] if at_runs_before else None
    if left_t is not None:
        row["left_start_pose_at"] = iso(left_t)

    # Last torque-ENABLED switch before the logged start (after the reset,
    # if there was one): from here on the springs can move the arm.
    after = prior[-1][1] if prior else t[0]
    enables = [e for e in events if e[1] == "torque" and e[2] == "ENABLED" and after <= e[0] < start]
    dev_at_enable = None
    if enables:
        te = enables[-1][0]
        # The "ENABLED" status is published slightly AFTER torque actually
        # comes on (in the study bags the springs have typically started
        # moving the arm 0-1 s earlier), and joint feedback pauses during the
        # switch. So judge the pose at the last sample before te - slack.
        ref_t = te - enable_slack_s
        dev_at_enable = float(dev[max(int(np.searchsorted(t, ref_t, side="right")) - 1, 0)])
        springs = [e[2] for e in events if e[1].startswith("springs") and e[0] <= te]
        if left_t is not None:
            row["left_before_torque_on_s"] = round(te - left_t, 1)
        row.update({"torque_enabled_at": iso(te),
                    "deg_from_start_pose_at_torque_enable": round(dev_at_enable, 1),
                    "springs_at_torque_enable": springs[-1] if springs else ""})

    late = [m for m in moves if start <= m[0] <= end]
    if late:
        row["verdict"] = "logged start before reset"
        row["check_on_video_at"] = f"{iso(start)} (logged) vs reset {iso(late[0][0])}"
    elif at[i_start]:
        row["verdict"] = "at start pose"
        row["check_on_video_at"] = iso(start)
    elif at_runs_before and condition != "KT" and dev_at_enable is not None and dev_at_enable > tol_deg:
        row["verdict"] = "off start pose at torque enable"
        row["check_on_video_at"] = f"{row['left_start_pose_at']} (left) / {row['torque_enabled_at']} (torque on)"
    elif at_runs_before:
        row["verdict"] = "reset OK"
        row["check_on_video_at"] = row["left_start_pose_at"]
    else:
        row["verdict"] = "no reset found"
        row["check_on_video_at"] = iso(start)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("timeseries_dir", help="joint_kinematics.py --out DIR/timeseries")
    ap.add_argument("--adjustment-times", required=True, help="adjustment periods CSV")
    ap.add_argument("--preset", required=True, help="study-start preset YAML")
    ap.add_argument("--alt-pose", action="append", default=[], type=parse_pose,
                    help="another accepted start pose, 7 comma-separated angles (repeatable)")
    ap.add_argument("--tol-deg", type=float, default=float(np.degrees(0.02)),
                    help="per-joint tolerance for 'at the start pose' (default ~1.15 deg)")
    ap.add_argument("--events", default=None,
                    help="events.csv from the same joint_kinematics.py run "
                         "(default: next to the timeseries folder, if present)")
    ap.add_argument("--enable-slack-s", type=float, default=2.0,
                    help="leaving the start pose within this many seconds before the "
                         "torque-ENABLED status message counts as the springs starting "
                         "(the status lags the actual switch); default 2")
    ap.add_argument("--tz", default="America/New_York")
    ap.add_argument("--out", default=None, help="write the table as CSV too")
    args = ap.parse_args()

    poses = [load_preset(args.preset)] + args.alt_pose
    data = load_timeseries(args.timeseries_dir)
    events_path = args.events or os.path.join(
        os.path.dirname(os.path.normpath(os.path.expanduser(args.timeseries_dir))), "events.csv")
    events = load_events(events_path)
    if not events:
        print(f"(no events.csv at {events_path}: torque-enable columns left blank)")
    periods = load_periods(args.adjustment_times, args.tz)
    rows = []
    for p in sorted(periods, key=lambda p: (int(_norm_participant(p["participant_id"]))
                                            if _norm_participant(p["participant_id"]).isdigit() else 0,
                                            p["condition"], p["side"])):
        key = (_norm_key(p["participant_id"]), CANON.get(p["condition"].lower(), p["condition"]))
        base = {"participant_id": key[0], "condition": key[1], "side": p["side"],
                "logged_adjustment_start": datetime.datetime.fromtimestamp(
                    p["start"], ZoneInfo(args.tz)).strftime("%Y-%m-%d %H:%M:%S")}
        if key not in data:
            rows.append({**base, "verdict": "no timeseries"})
            continue
        t, q, cmd = data[key]
        if not (t[0] <= p["end"] and t[-1] >= p["start"]):
            rows.append({**base, "verdict": "no timeseries"})
            continue
        rows.append({**base, **check_period(t, q, cmd, p["start"], p["end"], poses,
                                             args.tol_deg, args.tz, events.get(key, ()),
                                             key[1], args.enable_slack_s)})

    if args.out:
        with open(os.path.expanduser(args.out), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    for r in rows:
        if r["side"] != "1":
            continue
        print(f"{r['participant_id']:>4} {r['condition']:9} side 1  {r['verdict']:26} "
              f"check {r.get('check_on_video_at', ''):<34} "
              f"reset {r.get('last_reset_move', '') or '-':<17} "
              f"torque-on {r.get('torque_enabled_at', '') or '-':<8} "
              f"@start {r.get('deg_from_start_pose_at_logged_start', '')}deg "
              f"(lookback {r.get('lookback_s', '')}s)")
    counts = defaultdict(int)
    for r in rows:
        if r["side"] == "1":
            counts[r["verdict"]] += 1
    print("\nside 1:", dict(counts), "| full table (incl. side 2):", args.out or "(use --out)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
