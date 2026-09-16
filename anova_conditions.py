#!/usr/bin/env python3
"""Repeated-measures ANOVA of posture zone exposure across conditions.

The study is within-subjects: every participant did all three conditions
(KT/pose/position), each on two sides (side1/side2). compare_conditions.py's
condition_comparison_summary.csv pools *all* sessions into one row per
condition, which has no within-group variance to test -- not usable for an
ANOVA. This script instead builds one row per (participant, condition) --
combining a participant's two sides into a single pooled frame count for
that condition, per-frame (not averaging the two sides' percentages) -- and
runs a one-way repeated-measures ANOVA (condition as the within-subject
factor, participant as the subject) on each zone percentage, then Holm-
corrected post-hoc pairwise comparisons (paired t-tests) for any metric
whose omnibus ANOVA comes back significant.

Per-frame classification reuses classify_ergonomics.py's classify_row (via
compare_conditions.py's classify_session_counts), so zones here match
classify_ergonomics.py exactly; each session's shoulder-fallback percentile
is still computed session-relative, before its frames get pooled.

Pass --latex to also write a LaTeX (booktabs) table of each metric's
omnibus result, and of its post-hoc comparisons if they ran, for dropping
into a paper.

Requires pingouin + pandas (not part of the base posture_analysis venv
setup in the README -- pip install pingouin pandas).
"""
import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from classify_ergonomics import ZONE_BOTH, ZONE_HUNCH, ZONE_LEAN, ZONE_NEUTRAL
from compare_conditions import CATEGORIES, UNCLASSIFIED, classify_session_counts, extract_condition, extract_participant
from stats_helpers import posthoc_to_latex, rm_anova_to_latex, rm_anova_with_posthoc, warn_if_unbalanced

METRIC_COLUMNS = {
    ZONE_NEUTRAL: "pct_neutral",
    ZONE_LEAN: "pct_lean",
    ZONE_HUNCH: "pct_hunch",
    ZONE_BOTH: "pct_hunch_lean",
}


def build_participant_condition_table(files, hunch_thresh_deg, lean_thresh_deg, shoulder_percentile):
    """One row per (participant, condition): that participant's two sides'
    frame counts combined, then turned into zone percentages."""
    pooled = {}  # (participant, condition) -> Counter
    skipped = []
    for f in files:
        counts = classify_session_counts(f, hunch_thresh_deg, lean_thresh_deg, shoulder_percentile)
        if counts is None:
            skipped.append(f.name)
            continue
        key = (extract_participant(f), extract_condition(f))
        pooled.setdefault(key, Counter()).update(counts)

    rows = []
    for (participant, condition), counts in pooled.items():
        total = sum(counts.values())
        row = {"participant": participant, "condition": condition, "n_frames": total}
        row.update({col: 100.0 * counts.get(zone, 0) / total for zone, col in METRIC_COLUMNS.items()})
        row["pct_unclassified"] = 100.0 * counts.get(UNCLASSIFIED, 0) / total
        rows.append(row)
    return pd.DataFrame(rows), skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing per-session posture CSVs")
    ap.add_argument(
        "--pattern", default="*_trialonly.csv",
        help="glob pattern for input files (default: *_trialonly.csv)",
    )
    ap.add_argument("--hunch-threshold-deg", type=float, default=20.0)
    ap.add_argument("--lean-threshold-deg", type=float, default=15.0)
    ap.add_argument("--shoulder-percentile", type=float, default=75.0)
    ap.add_argument(
        "--out-table", default=None,
        help="per-participant-per-condition table used as ANOVA input "
        "(default: <directory>/anova_participant_condition_table.csv)",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="significance threshold for running post-hoc pairwise comparisons "
        "after a metric's omnibus ANOVA (default: 0.05)",
    )
    ap.add_argument("--latex", action="store_true", help="also write a LaTeX table of each metric's results")
    ap.add_argument(
        "--latex-out", default=None,
        help="LaTeX output path (default: <directory>/anova_conditions.tex)",
    )
    args = ap.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern!r} in {directory}")

    out_table = Path(args.out_table) if args.out_table else directory / "anova_participant_condition_table.csv"

    df, skipped = build_participant_condition_table(files, args.hunch_threshold_deg, args.lean_threshold_deg, args.shoulder_percentile)
    if df.empty:
        raise SystemExit(f"none of the {len(files)} matched file(s) had any rows")

    df.to_csv(out_table, index=False)
    print(f"wrote {out_table} ({len(df)} participant x condition rows)")
    if skipped:
        print(f"{len(skipped)} file(s) skipped (empty): {', '.join(skipped)}")

    warn_if_unbalanced(df, "condition", "participant")

    n_participants = df["participant"].nunique()
    print(f"\n{n_participants} participants x {df['condition'].nunique()} conditions\n")

    latex_tables = []
    for metric_col in list(METRIC_COLUMNS.values()) + ["pct_unclassified"]:
        print(f"=== {metric_col} ===")
        anova_result, posthoc_result = rm_anova_with_posthoc(df, dv=metric_col, within="condition", subject="participant", alpha=args.alpha)
        print()

        if args.latex:
            metric_label = metric_col.replace("_", "-")
            latex_tables.append(rm_anova_to_latex(
                anova_result,
                caption=f"Repeated-measures ANOVA of {metric_col} by condition",
                label=f"tab:anova-{metric_label}",
            ))
            if posthoc_result is not None:
                latex_tables.append(posthoc_to_latex(
                    posthoc_result,
                    caption=f"Post-hoc pairwise comparisons of {metric_col} (Holm-corrected)",
                    label=f"tab:posthoc-{metric_label}",
                ))

    if args.latex:
        latex_out = Path(args.latex_out) if args.latex_out else directory / "anova_conditions.tex"
        latex_out.write_text("\n\n".join(latex_tables) + "\n")
        print(f"wrote {latex_out}")


if __name__ == "__main__":
    main()
