#!/usr/bin/env python3
"""
fluency_summary.py

Collapse joint_kinematics.py's summary.csv (one row per period x joint) into
fluency numbers per participant x condition, then compare conditions.

Per period (one side):
    rms_acc, rms_jerk    RMS across the 7 joints: sqrt(mean_j rms_j^2)
    peak_acc, peak_jerk  max across the 7 joints of each joint's peak |.|
Per participant x condition (the two sides):
    rms_*   pooled RMS over both sides, weighted by duration:
            sqrt(sum_i d_i * rms_i^2 / sum_i d_i)
    peak_*  max over the two sides
Across conditions (participants with all three conditions only):
    Friedman omnibus, then paired Wilcoxon signed-rank for each pair with
    Holm correction across the three pairs (same scheme as analyze.py).

Peaks depend on the Savitzky-Golay window and on how long a period is;
treat them as secondary and check RMS ordering holds at 2-3 --sg-window-ms
values before relying on jerk.

Usage:
    python3 analysis/fluency_summary.py kinematics_out/summary.csv --out fluency_out
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

CONDITIONS = ["KT", "Pose", "Position"]
METRICS = ["rms_acc", "rms_jerk", "peak_acc", "peak_jerk"]
CANON = {"kt": "KT", "pose": "Pose", "position": "Position"}


def load_periods(path: str) -> dict:
    """{(participant, condition, side): [joint rows]} from summary.csv."""
    periods = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cond = CANON.get(row["condition"].strip().lower(), row["condition"].strip())
            periods[(row["participant_id"].strip(), cond, row["side"].strip())].append(row)
    return periods


def per_period(periods: dict) -> list[dict]:
    out = []
    for (pid, cond, side), rows in sorted(periods.items()):
        f = lambda k: np.array([float(r[k]) for r in rows])
        out.append({
            "participant_id": pid, "condition": cond, "side": side,
            "n_joints": len(rows),
            "duration_s": float(rows[0]["duration_s"]),
            "rms_acc": float(np.sqrt(np.mean(f("rms_acc") ** 2))),
            "rms_jerk": float(np.sqrt(np.mean(f("rms_jerk") ** 2))),
            "peak_acc": float(np.max(f("peak_abs_acc"))),
            "peak_jerk": float(np.max(f("peak_abs_jerk"))),
        })
    return out


def per_participant_condition(period_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in period_rows:
        groups[(r["participant_id"], r["condition"])].append(r)
    out = []
    for (pid, cond), rs in sorted(groups.items()):
        d = np.array([r["duration_s"] for r in rs])
        row = {"participant_id": pid, "condition": cond,
               "n_sides": len(rs), "total_duration_s": float(d.sum())}
        for m in ("rms_acc", "rms_jerk"):
            row[m] = float(np.sqrt(np.sum(d * np.array([r[m] for r in rs]) ** 2) / d.sum()))
        for m in ("peak_acc", "peak_jerk"):
            row[m] = float(max(r[m] for r in rs))
        out.append(row)
    return out


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def compare(pc_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """(descriptives, friedman, pairwise) over participants with all conditions."""
    by = defaultdict(dict)
    for r in pc_rows:
        by[r["participant_id"]][r["condition"]] = r
    complete = sorted(p for p, d in by.items() if all(c in d for c in CONDITIONS))
    desc, fried, pair = [], [], []
    for m in METRICS:
        data = {c: np.array([by[p][c][m] for p in complete]) for c in CONDITIONS}
        for c in CONDITIONS:
            x = data[c]
            desc.append({"metric": m, "condition": c, "n": len(x),
                         "mean": float(x.mean()) if len(x) else float("nan"),
                         "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"),
                         "median": float(np.median(x)) if len(x) else float("nan")})
        if len(complete) >= 3:
            stat, p = friedmanchisquare(*[data[c] for c in CONDITIONS])
            fried.append({"metric": m, "n": len(complete),
                          "friedman_chi2": float(stat), "p": float(p)})
        rows = []
        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1:]:
                diff = data[a] - data[b]
                if len(complete) >= 1 and np.any(diff != 0):
                    w, p = wilcoxon(data[a], data[b])
                    w, p = float(w), float(p)
                else:
                    w, p = float("nan"), float("nan")
                rows.append({"metric": m, "pair": f"{a} vs {b}", "n": len(complete),
                             "median_a": float(np.median(data[a])) if len(complete) else float("nan"),
                             "median_b": float(np.median(data[b])) if len(complete) else float("nan"),
                             "wilcoxon_W": w, "p": p})
        valid = [r for r in rows if np.isfinite(r["p"])]
        for r, adj in zip(valid, holm([r["p"] for r in valid])):
            r["p_holm"] = adj
        for r in rows:
            r.setdefault("p_holm", float("nan"))
        pair.extend(rows)
    return desc, fried, pair, complete


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary_csv", help="summary.csv from joint_kinematics.py")
    ap.add_argument("--out", required=True, help="Output directory.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    period_rows = per_period(load_periods(args.summary_csv))
    pc_rows = per_participant_condition(period_rows)
    desc, fried, pair, complete = compare(pc_rows)

    write_csv(os.path.join(args.out, "fluency_by_period.csv"), period_rows)
    write_csv(os.path.join(args.out, "fluency_by_participant_condition.csv"), pc_rows)
    write_csv(os.path.join(args.out, "fluency_by_condition.csv"), desc)
    write_csv(os.path.join(args.out, "fluency_friedman.csv"), fried)
    write_csv(os.path.join(args.out, "fluency_pairwise.csv"), pair)

    print(f"{len(period_rows)} periods, {len({r['participant_id'] for r in pc_rows})} "
          f"participants; {len(complete)} have all three conditions "
          f"(used for the tests).", file=sys.stderr)
    if len(complete) < 6:
        print("  note: too few complete participants for meaningful tests yet.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
