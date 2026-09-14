"""
analyze.py

Scores the CQHRI Qualtrics export into a tidy long-format table (one row
per participant x condition) and compares the three within-subject robot
control conditions (KT / Position / Pose) on:

  - fluency        -- single 5-point item (QID10/QID12/QID14)
  - adapts         -- Perceived Social Intelligence Scale "Adapts to Human
                       Behaviors" subscale (items 1-4 of the QID11/13/15
                       matrix), mean of 1-5 item scores
  - helpfulness    -- PSI "Helpfulness" subscale (items 5-8 of the same
                       matrix), mean of 1-5 item scores

The survey block-randomizes which condition is shown in trial position 1
vs 2 vs 3 per participant; the Condition question at the top of each block
(QID24/25/26) records which condition that block's ratings actually belong
to, so this script re-keys the data by condition rather than by trial
position before comparing.

Trial position (1st/2nd/3rd, i.e. which block QID24/25/26 the row came
from) is kept as an "order" column, since anecdotally people tend to rate
whichever condition they saw *last* more favorably. Because each
participant only ever sees each condition at one order position (a Latin
square, not a full condition x order factorial per participant),
condition and order can't both be tested with plain repeated-measures
ANOVA -- there's no within-subject replication to separate them that way.
Instead, controlling for order uses a linear mixed model (participant as
a random intercept, condition and order as fixed effects) with a
likelihood-ratio test comparing the full model to the model with the
other factor dropped.

See qualtrics_parser.py for why this reads the raw CSV directly (via the
`csv` module + Qualtrics's row-3 ImportId) instead of pandas.read_csv: the
export has several literal duplicate column headers, and the ImportId row
is the only reliable way to tell them apart.

Usage:
  python analyze.py <path/to/choice-text-export.csv> [--out scored_data.csv]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel, chi2
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from qualtrics_parser import (
    load_qualtrics_csv,
    column_index,
    score_choice_text,
    FLUENCY_SCALE,
    AGREEMENT_SCALE,
)

CONDITIONS = ["KT", "Position", "Pose"]
ORDERS = [1, 2, 3]

# (order, condition_qid, fluency_qid, adapts_item_qids, helpfulness_item_qids)
BLOCKS = [
    (1, "QID24", "QID10", [f"QID11_{i}" for i in range(1, 5)], [f"QID11_{i}" for i in range(5, 9)]),
    (2, "QID25", "QID12", [f"QID13_{i}" for i in range(1, 5)], [f"QID13_{i}" for i in range(5, 9)]),
    (3, "QID26", "QID14", [f"QID15_{i}" for i in range(1, 5)], [f"QID15_{i}" for i in range(5, 9)]),
]

MEASURES = ["fluency", "adapts", "helpfulness"]


def mean_or_nan(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else float("nan")


def build_tidy_table(path):
    import_ids, data_rows = load_qualtrics_csv(path)
    pid_col = column_index(import_ids, "QID23_TEXT")

    tidy_rows = []
    for row_num, row in enumerate(data_rows, start=1):
        participant_id = row[pid_col].strip() or f"row{row_num}"

        for order, cond_qid, fluency_qid, adapts_qids, help_qids in BLOCKS:
            condition = row[column_index(import_ids, cond_qid)].strip()
            if not condition:
                continue
            if condition not in CONDITIONS:
                print(f"WARNING: participant {participant_id} block {cond_qid} has "
                      f"unrecognized condition {condition!r} -- skipping this block")
                continue

            fluency_text = row[column_index(import_ids, fluency_qid)]
            fluency = score_choice_text(
                fluency_text, FLUENCY_SCALE,
                context=f"participant {participant_id}, {fluency_qid}",
            )

            adapts_scores = [
                score_choice_text(
                    row[column_index(import_ids, qid)], AGREEMENT_SCALE,
                    context=f"participant {participant_id}, {qid}",
                )
                for qid in adapts_qids
            ]
            help_scores = [
                score_choice_text(
                    row[column_index(import_ids, qid)], AGREEMENT_SCALE,
                    context=f"participant {participant_id}, {qid}",
                )
                for qid in help_qids
            ]

            tidy_rows.append({
                "participant_id": participant_id,
                "condition": condition,
                "order": order,
                "fluency": fluency,
                "adapts": mean_or_nan(adapts_scores),
                "helpfulness": mean_or_nan(help_scores),
            })

    return pd.DataFrame(tidy_rows)


def check_condition_labeling(tidy):
    """Each participant should have exactly one block per condition. Flag
    anyone who doesn't (duplicate or missing condition label) -- likely a
    data-entry mistake on the Condition question, not something to average
    away silently via pivot_table."""
    for pid, conds in tidy.groupby("participant_id")["condition"]:
        conds = conds.tolist()
        if sorted(conds) != sorted(CONDITIONS):
            print(f"WARNING: participant {pid} has condition labels {conds} "
                  f"(expected exactly one each of {CONDITIONS}) -- check the "
                  f"Condition question answers for this participant; their "
                  f"data may be misassigned or incomplete.")


def print_descriptives(tidy):
    print("\n=== Descriptive statistics by condition ===")
    for measure in MEASURES:
        print(f"\n-- {measure} --")
        desc = tidy.groupby("condition")[measure].agg(["count", "mean", "median", "std"])
        desc = desc.reindex(CONDITIONS)
        print(desc.round(2).to_string())


def print_order_descriptives(tidy):
    print("\n=== Descriptive statistics by trial order (1st/2nd/3rd seen, "
          "regardless of condition) ===")
    for measure in MEASURES:
        print(f"\n-- {measure} --")
        desc = tidy.groupby("order")[measure].agg(["count", "mean", "median", "std"])
        desc = desc.reindex(ORDERS)
        print(desc.round(2).to_string())


def wide_for_measure(tidy, measure):
    """Participant x condition table; only participants with all 3
    conditions present are usable for the repeated-measures tests."""
    wide = tidy.pivot_table(index="participant_id", columns="condition", values=measure)
    wide = wide.reindex(columns=CONDITIONS)
    complete = wide.dropna()
    return wide, complete


def run_friedman_and_posthoc(complete, measure):
    print(f"\n-- {measure}: Friedman test (n={len(complete)} complete participants) --")
    if len(complete) < 3:
        print("Not enough complete participants to run the Friedman test.")
        return
    stat, p = friedmanchisquare(*[complete[c] for c in CONDITIONS])
    print(f"Friedman chi-square = {stat:.3f}, p = {p:.4f}")

    pairs = [("KT", "Position"), ("KT", "Pose"), ("Position", "Pose")]
    pvals = []
    for a, b in pairs:
        try:
            _, p_pair = wilcoxon(complete[a], complete[b])
        except ValueError as e:
            p_pair = float("nan")
            print(f"  Wilcoxon {a} vs {b}: could not compute ({e})")
        pvals.append(p_pair)
    valid = [p for p in pvals if not np.isnan(p)]
    if valid:
        _, corrected, _, _ = multipletests(pvals, method="holm")
        for (a, b), raw_p, adj_p in zip(pairs, pvals, corrected):
            print(f"  Wilcoxon {a} vs {b}: p = {raw_p:.4f} (Holm-corrected: {adj_p:.4f})")


def run_rm_anova_and_posthoc(complete, measure):
    print(f"\n-- {measure}: Repeated-measures ANOVA (n={len(complete)} complete participants) --")
    if len(complete) < 3:
        print("Not enough complete participants to run the RM-ANOVA.")
        return
    long = complete.reset_index().melt(id_vars="participant_id", value_vars=CONDITIONS,
                                        var_name="condition", value_name=measure)
    try:
        result = AnovaRM(long, depvar=measure, subject="participant_id", within=["condition"]).fit()
        print(result.summary())
    except Exception as e:
        print(f"Could not run RM-ANOVA: {e}")
        return

    pairs = [("KT", "Position"), ("KT", "Pose"), ("Position", "Pose")]
    pvals = []
    for a, b in pairs:
        _, p_pair = ttest_rel(complete[a], complete[b])
        pvals.append(p_pair)
    _, corrected, _, _ = multipletests(pvals, method="holm")
    for (a, b), raw_p, adj_p in zip(pairs, pvals, corrected):
        print(f"  Paired t-test {a} vs {b}: p = {raw_p:.4f} (Holm-corrected: {adj_p:.4f})")


def run_mixed_model_controlling_for_order(tidy, measure):
    """
    Linear mixed model: measure ~ C(condition) + C(order), with a random
    intercept per participant. Condition and order can't both be tested
    with plain RM-ANOVA here because each participant sees each condition
    at exactly one order position (a Latin square) -- there's no
    within-subject replication of condition at a fixed order or vice versa.
    A mixed model instead pools information across participants to
    estimate each factor's effect while adjusting for the other, and also
    tolerates the unbalanced/missing cells in this data (e.g. participant
    4's mislabeled condition, or the one blank matrix item).

    Reports the fixed-effect estimates from the full model, plus a
    likelihood-ratio test for each factor (comparing the full model to the
    model with that factor dropped) -- this is what actually answers
    "does condition matter after controlling for order?" and vice versa.
    """
    sub = tidy.dropna(subset=[measure]).copy()
    sub["condition"] = pd.Categorical(sub["condition"], categories=CONDITIONS)
    sub["order"] = pd.Categorical(sub["order"], categories=ORDERS)

    print(f"\n-- {measure}: mixed model controlling for condition and order "
          f"(n={len(sub)} block ratings, {sub['participant_id'].nunique()} participants) --")

    try:
        full = smf.mixedlm(f"{measure} ~ C(condition) + C(order)", data=sub,
                            groups=sub["participant_id"]).fit(reml=False)
        order_only = smf.mixedlm(f"{measure} ~ C(order)", data=sub,
                                  groups=sub["participant_id"]).fit(reml=False)
        condition_only = smf.mixedlm(f"{measure} ~ C(condition)", data=sub,
                                      groups=sub["participant_id"]).fit(reml=False)
    except Exception as e:
        print(f"Could not fit the mixed model: {e}")
        return

    print(full.summary().tables[1].to_string())

    condition_lr = 2 * (full.llf - order_only.llf)
    condition_p = chi2.sf(condition_lr, df=len(CONDITIONS) - 1)
    print(f"\n  Condition effect (controlling for order): "
          f"LR chi-square = {condition_lr:.3f}, df = {len(CONDITIONS) - 1}, p = {condition_p:.4f}")

    order_lr = 2 * (full.llf - condition_only.llf)
    order_p = chi2.sf(order_lr, df=len(ORDERS) - 1)
    print(f"  Order effect (controlling for condition): "
          f"LR chi-square = {order_lr:.3f}, df = {len(ORDERS) - 1}, p = {order_p:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to a Qualtrics choice-text CSV export")
    parser.add_argument("--out", default="scored_data.csv",
                         help="Where to write the tidy scored CSV (default: scored_data.csv)")
    args = parser.parse_args()

    tidy = build_tidy_table(args.csv_path)
    if tidy.empty:
        print("No scoreable rows found in the input file.", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    tidy.to_csv(out_path, index=False)
    print(f"Wrote tidy scored data ({len(tidy)} rows) to {out_path}")

    check_condition_labeling(tidy)
    print_descriptives(tidy)
    print_order_descriptives(tidy)

    print("\n=== Repeated-measures comparisons across KT / Position / Pose "
          "(condition only, order ignored) ===")
    for measure in MEASURES:
        _, complete = wide_for_measure(tidy, measure)
        run_friedman_and_posthoc(complete, measure)
        run_rm_anova_and_posthoc(complete, measure)

    print("\n=== Controlling for order (mixed model) ===")
    for measure in MEASURES:
        run_mixed_model_controlling_for_order(tidy, measure)


if __name__ == "__main__":
    main()
