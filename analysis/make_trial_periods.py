#!/usr/bin/env python3
"""
make_trial_periods.py

Build the --periods CSV for joint_kinematics.py that spans each side from
ADJUSTMENT START to TRIAL END.

  * start: "Adjustment Start" from adjustment_times.csv (built from the
           corrected event logs by posture/adjustment_times.py; logged
           button presses, not generally video-verified)
  * end:   the last "Trial end" event for that participant/condition/side in
           the corrected event logs (event_log_<N>_corrected.csv)

Output columns: participant_id,condition,side,start,end  (local wall-clock
ISO times, the format joint_kinematics.load_periods() reads). Sides with a
missing/invalid start or end are left out and listed on stderr, so nothing
is silently dropped.

Usage:
    python3 analysis/make_trial_periods.py --logs-dir corrected \\
        --times corrected/adjustment_times.csv --out adjustment_to_trial_end.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from datetime import datetime

CONDITION_SIDE = re.compile(r"^\s*(KT|pose|position)\s+side\s+(\d+)\s*$", re.I)
CANON = {"kt": "KT", "pose": "Pose", "position": "Position"}


def parse_condition_side(text: str):
    m = CONDITION_SIDE.match(text or "")
    return (CANON[m.group(1).lower()], m.group(2)) if m else None


def load_trial_ends(logs_dir: str) -> dict:
    """{(participant 'pN', condition, side): [trial-end ISO strings]}"""
    ends: dict = {}
    for path in glob.glob(os.path.join(logs_dir, "event_log_*_corrected.csv")):
        n = int(re.findall(r"\d+", os.path.basename(path))[0])
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if (row.get("event") or "").strip().lower() != "trial end":
                    continue
                cs = parse_condition_side(row.get("condition"))
                ts = (row.get("timestamp") or "").strip()
                if cs and ts:
                    ends.setdefault((f"p{n}",) + cs, []).append(ts)
    return ends


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--logs-dir", required=True)
    ap.add_argument("--times", required=True, help="adjustment_times.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    ends = load_trial_ends(args.logs_dir)
    kept, problems = [], []
    with open(args.times, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            pid, cond, side = row["Participant"].strip(), row["Condition"].strip(), row["Side"].strip()
            start = row["Adjustment Start"].strip()
            cand = ends.get((pid, cond, side), [])
            if not start or not cand:
                problems.append(f"{pid} {cond} side {side}: missing "
                                f"{'start' if not start else 'trial end'}")
                continue
            end = max(cand)  # ISO strings sort chronologically
            if datetime.fromisoformat(end) <= datetime.fromisoformat(start):
                problems.append(f"{pid} {cond} side {side}: trial end {end} "
                                f"is not after start {start}")
                continue
            if len(cand) > 1:
                problems.append(f"{pid} {cond} side {side}: {len(cand)} "
                                f"'Trial end' events, used the last ({end})")
            kept.append([pid, cond, side, start, end])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["participant_id", "condition", "side", "start", "end"])
        w.writerows(kept)
    print(f"wrote {len(kept)} periods to {args.out}", file=sys.stderr)
    for p in problems:
        print("  note:", p, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
