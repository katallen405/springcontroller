#!/usr/bin/env python3
"""
fluency_sensitivity.py

Sensitivity and movement-amount checks on the fluency (acceleration/jerk)
comparison between conditions, from joint_kinematics.py's summary.csv and
the hand-coded freeze_error_candidates.csv.

Periods named with --exclude PID:CONDITION[:SIDE] (e.g. p10:Pose, an
experimenter error) are dropped from everything below first, including
all_periods and the speed/ratio tables.

Three analysis versions, each run on the same data:
    all_periods                     every period (after --exclude)
    excl_all_robot_error_periods    drop any (participant, condition, side)
                                    with a row coded robot_error
    excl_robot_only_error_periods   as above, but keep periods whose
                                    robot_error row is also coded user_error

Outputs (written to --out):
    sensitivity_pairwise.csv        paired Wilcoxon per metric/pair/version
                                    (pairwise-complete participants), Holm
                                    across the three pairs of each metric
    speed_by_participant_condition.csv
                                    total joint path, duration, mean joint
                                    speed, rms acc/jerk per participant x
                                    condition (all periods)
    movement_normalized_ratio.csv   rms_acc / speed and rms_jerk / speed,
                                    paired Wilcoxon + Holm (all periods).
                                    Assumes the metric scales 1:1 with
                                    speed, which the data do not support;
                                    prefer the regression.
    movement_normalized_regression_by_exclusion.csv
                                    log(metric) ~ condition + log(speed) +
                                    participant effects (OLS), contrasts
                                    reported as ratios with 95% CI, Holm
                                    across the three pairs.

Speed = total joint-space path length / total duration over the periods
kept, i.e. averaged over the whole adjustment-to-trial-end span including
pauses -- a rough measure of movement amount.

Usage:
    python3 analysis/fluency_sensitivity.py kinematics_out/summary.csv \\
        --coded freeze_error_candidates.csv --out fluency_sensitivity \\
        --exclude p10:Pose
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import t as t_dist
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exclusions as ex_mod  # noqa: E402
import fluency_summary as fs  # noqa: E402

CONDITIONS = fs.CONDITIONS
SIDE_RE = re.compile(r"\s*(KT|pose|position)\s+side\s+(\d+)", re.I)


def load_segments(summary_csv: str) -> list[dict]:
    """Per period x bag segment: fluency metrics plus joint path length."""
    periods = fs.load_periods(summary_csv)
    segs = fs.per_period(periods)
    for s in segs:
        rows = periods[(s["participant_id"], s["condition"], s["side"], s["bag"])]
        s["path_rad"] = float(rows[0]["joint_path_length_rad"])
    return segs


def apply_exclusions(segs: list[dict], exclusions: list) -> list[dict]:
    """Drop segments matching any --exclude. One that matches nothing is
    an error, so a typo can't silently keep an invalid period."""
    for ex in exclusions:
        hit = [s for s in segs if ex_mod.matches(ex, s["participant_id"], s["condition"], s["side"])]
        if not hit:
            raise SystemExit(f"--exclude {ex_mod.describe(ex)} matches no periods")
        print(f"excluded {ex_mod.describe(ex)}: {len(hit)} segment(s)", file=sys.stderr)
    return [s for s in segs
            if not any(ex_mod.matches(ex, s["participant_id"], s["condition"], s["side"])
                       for ex in exclusions)]


def coded_exclusions(coded_csv: str, robot_only: bool) -> set:
    """{(pid 'pN', condition, side)} for periods holding a robot_error row."""
    out = set()
    with open(coded_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    col = next(h for h in rows[0] if h.startswith("coded_as"))
    for r in rows:
        tags = {t.strip().lower() for t in (r[col] or "").split(",") if t.strip()}
        if "robot_error" not in tags or (robot_only and "user_error" in tags):
            continue
        m = SIDE_RE.match(r["condition"] or "")
        if m:
            out.add((f"p{int(r['participant'])}", fs.CANON[m.group(1).lower()], m.group(2)))
    return out


def by_participant(pc_rows: list[dict]) -> dict:
    d = defaultdict(dict)
    for r in pc_rows:
        d[r["participant_id"]][r["condition"]] = r
    return d


def speeds(segs: list[dict]) -> dict:
    """{(pid, condition): mean joint speed rad/s}, plus totals."""
    tot = defaultdict(lambda: [0.0, 0.0])
    for s in segs:
        t = tot[(s["participant_id"], s["condition"])]
        t[0] += s["path_rad"]
        t[1] += s["duration_s"]
    return {k: v for k, v in tot.items()}


def pairwise_wilcoxon(pc_rows: list[dict], label: str) -> list[list]:
    by = by_participant(pc_rows)
    res = []
    for m in fs.METRICS:
        block = []
        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1:]:
                ids = [p for p, d in by.items() if a in d and b in d]
                x = np.array([by[p][a][m] for p in ids])
                y = np.array([by[p][b][m] for p in ids])
                p = float(wilcoxon(x, y)[1]) if np.any(x != y) else float("nan")
                block.append([label, m, f"{a} vs {b}", len(ids),
                              float(np.median(x)), float(np.median(y)), p])
        valid = [r for r in block if np.isfinite(r[-1])]
        adj = dict(zip(map(id, valid), fs.holm([r[-1] for r in valid])))
        res += [r + [adj.get(id(r), float("nan"))] for r in block]
    return res


def regression(segs: list[dict], metric: str):
    """OLS of log(metric) on condition dummies, log speed, participant
    dummies. Returns (contrast rows, speed elasticity, df, n obs)."""
    pc = {(r["participant_id"], r["condition"]): r
          for r in fs.per_participant_condition(segs)}
    spd = speeds(segs)
    keys = sorted(pc, key=lambda q: (int(q[0][1:]), CONDITIONS.index(q[1])))
    ids = sorted({p for p, _ in keys}, key=lambda s: int(s[1:]))
    y = np.array([np.log(pc[k][metric]) for k in keys])
    X = np.array([[1.0, np.log(spd[k][0] / spd[k][1]),
                   float(k[1] == "Pose"), float(k[1] == "Position")]
                  + [float(k[0] == q) for q in ids[1:]] for k in keys])
    beta, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    df = len(y) - rank
    cov = (resid @ resid / df) * np.linalg.pinv(X.T @ X)

    def contrast(c, name):
        c = np.array(c + [0] * (X.shape[1] - len(c)), dtype=float)
        est, se = c @ beta, float(np.sqrt(c @ cov @ c))
        p = float(2 * t_dist.sf(abs(est / se), df))
        return [name, float(np.exp(est)), float(np.exp(est - 1.96 * se)),
                float(np.exp(est + 1.96 * se)), p]

    rows = [contrast([0, 0, -1, 0], "KT/Pose"),
            contrast([0, 0, 0, -1], "KT/Position"),
            contrast([0, 0, 1, -1], "Pose/Position")]
    for r, h in zip(rows, fs.holm([r[4] for r in rows])):
        r.append(h)
    return rows, float(beta[1]), int(df), len(keys)


def write(path: str, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("summary_csv", help="summary.csv from joint_kinematics.py")
    ap.add_argument("--coded", required=True,
                    help="Hand-coded freeze_error_candidates.csv (coded_as column).")
    ap.add_argument("--out", required=True, help="Output directory.")
    ex_mod.add_argument(ap)
    args = ap.parse_args()

    segs = apply_exclusions(load_segments(args.summary_csv), args.exclude)
    os.makedirs(args.out, exist_ok=True)
    versions = [("all_periods", set()),
                ("excl_all_robot_error_periods", coded_exclusions(args.coded, False)),
                ("excl_robot_only_error_periods", coded_exclusions(args.coded, True))]

    pair_rows, reg_rows = [], []
    for label, ex in versions:
        kept = [s for s in segs if (s["participant_id"], s["condition"], s["side"]) not in ex]
        print(f"{label}: {len(ex)} excluded sides, {len(kept)} of {len(segs)} segments kept",
              file=sys.stderr)
        pc = fs.per_participant_condition(kept)
        pair_rows += pairwise_wilcoxon(pc, label)
        for metric in ("rms_acc", "rms_jerk"):
            rows, elast, df, n = regression(kept, metric)
            for r in rows:
                reg_rows.append([label, metric, r[0], round(r[1], 4), round(r[2], 4),
                                 round(r[3], 4), f"{r[4]:.3g}", f"{r[5]:.3g}",
                                 round(elast, 3), df, len(kept), n])
    write(os.path.join(args.out, "sensitivity_pairwise.csv"),
          ["analysis", "metric", "pair", "n", "median_a", "median_b", "p", "p_holm"], pair_rows)
    write(os.path.join(args.out, "movement_normalized_regression_by_exclusion.csv"),
          ["analysis", "metric", "comparison", "ratio", "ci95_low", "ci95_high", "p",
           "p_holm", "speed_elasticity", "df", "n_periods_kept", "n_participant_conditions"],
          reg_rows)

    # All-periods speed table and per-unit-speed ratio tests.
    pc = fs.per_participant_condition(segs)
    pcd = {(r["participant_id"], r["condition"]): r for r in pc}
    spd = speeds(segs)
    ids = sorted({p for p, _ in spd}, key=lambda s: int(s[1:]))
    write(os.path.join(args.out, "speed_by_participant_condition.csv"),
          ["participant_id", "condition", "total_path_rad", "total_duration_s",
           "mean_joint_speed_rad_s", "rms_acc", "rms_jerk"],
          [[p, c, spd[(p, c)][0], spd[(p, c)][1], spd[(p, c)][0] / spd[(p, c)][1],
            pcd[(p, c)]["rms_acc"], pcd[(p, c)]["rms_jerk"]]
           for p in ids for c in CONDITIONS if (p, c) in spd])
    ratio_rows = []
    full = [p for p in ids if all((p, c) in spd for c in CONDITIONS)]
    for m in ("rms_acc", "rms_jerk"):
        v = {c: np.array([pcd[(p, c)][m] / (spd[(p, c)][0] / spd[(p, c)][1]) for p in full])
             for c in CONDITIONS}
        block = []
        for i, a in enumerate(CONDITIONS):
            for b in CONDITIONS[i + 1:]:
                block.append([m, f"{a} vs {b}", len(full),
                              float(np.median(v[a] / v[b])), float(wilcoxon(v[a], v[b])[1])])
        for r, h in zip(block, fs.holm([r[-1] for r in block])):
            ratio_rows.append(r + [h])
    write(os.path.join(args.out, "movement_normalized_ratio.csv"),
          ["metric", "pair", "n", "median_within_person_ratio", "p", "p_holm"], ratio_rows)
    print(f"wrote 4 files to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
