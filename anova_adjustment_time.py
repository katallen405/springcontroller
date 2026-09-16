#!/usr/bin/env python3
"""Repeated-measures ANOVA of adjustment time across conditions, from the
table written by adjustment_times.py.

Same within-subjects design as anova_conditions.py (every participant did
all three conditions, each on two sides): builds one row per (Participant,
Condition) by summing that participant's two sides' Total Adjustment Time
for the condition, then runs a one-way repeated-measures ANOVA (condition
as the within-subject factor, participant as the subject) followed by
Holm-corrected post-hoc pairwise comparisons if the omnibus result is
significant -- see stats_helpers.rm_anova_with_posthoc.

Requires pingouin + pandas (see README's ANOVA section for install).
"""
import argparse

import pandas as pd

from stats_helpers import rm_anova_with_posthoc, warn_if_unbalanced


def build_participant_condition_table(adjustment_times_csv):
    """One row per (Participant, Condition): that participant's two sides'
    Total Adjustment Time summed."""
    df = pd.read_csv(adjustment_times_csv)
    return (
        df.groupby(["Participant", "Condition"], as_index=False)["Total Adjustment Time"]
        .sum()
        .rename(columns={"Total Adjustment Time": "total_adjustment_time_sec"})
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("adjustment_times_csv", help="output of adjustment_times.py")
    ap.add_argument(
        "--alpha", type=float, default=0.05,
        help="significance threshold for running post-hoc pairwise comparisons "
        "after the omnibus ANOVA (default: 0.05)",
    )
    args = ap.parse_args()

    per_condition = build_participant_condition_table(args.adjustment_times_csv)

    warn_if_unbalanced(per_condition, "Condition", "Participant")

    n_participants = per_condition["Participant"].nunique()
    print(f"{n_participants} participants x {per_condition['Condition'].nunique()} conditions "
          f"(each participant's two sides summed per condition)\n")

    rm_anova_with_posthoc(per_condition, dv="total_adjustment_time_sec", within="Condition", subject="Participant", alpha=args.alpha)


if __name__ == "__main__":
    main()
