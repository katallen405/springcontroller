#!/usr/bin/env python3
"""Pool a directory of per-session posture CSVs (extract_posture.py output,
or a trial-cropped *_trialonly.csv derived from it -- same columns either
way) by condition, and plot one horizontal stacked bar per condition
showing hunch/lean/hunch+lean/neutral/unclassified as % of total frames.

Condition (e.g. KT/pose/position) is parsed out of each filename by
stripping the participant prefix (p<N>_) and any trailing
_side<N>/_trialonly/_posture segments -- see extract_condition(). All
sessions sharing a condition are pooled into one row, frame-counted
together, so a condition with more/longer sessions naturally carries more
weight in that row -- this is a per-frame, not a per-session, average.

Reuses classify_ergonomics.py's own per-frame classification
(load_posture_rows/shoulder_fallback_thresholds/classify_row) rather than
reimplementing it, so a frame's zone here always matches what
`classify_ergonomics.py <that_session>_posture.csv` would call it --
thresholds stay session-relative per classify_ergonomics.py's design (see
its docstring): each session's own shoulder-fallback percentile is computed
from that session alone, before pooling.

Does NOT write each session's *_zones.csv/*_episodes.csv/*_ergo_summary.csv
-- run classify_ergonomics.py directly on a session if you need those (e.g.
to spot-check a condition that looks like an outlier here against its
overlay video).
"""
import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from classify_ergonomics import ZONE_BOTH, ZONE_HUNCH, ZONE_LEAN, ZONE_NEUTRAL, classify_row, load_posture_rows, shoulder_fallback_thresholds

UNCLASSIFIED = "unclassified"
CATEGORIES = [ZONE_NEUTRAL, ZONE_LEAN, ZONE_HUNCH, ZONE_BOTH, UNCLASSIFIED]

# SmartPlayground's categorical palette (VideoDataAnalysis/plot_infographic.py),
# reassigned here low-to-high severity; "unclassified" isn't a posture zone
# at all (no signal, not "good posture"), so it stays a neutral gray rather
# than borrowing one of the four hues used for real zones.
CATEGORY_COLORS = {
    ZONE_NEUTRAL: "#29788e",
    ZONE_LEAN: "#fca635",
    ZONE_HUNCH: "#e06461",
    ZONE_BOTH: "#6a00a7",
    UNCLASSIFIED: "#999999",
}

# Mirrors CONDITIONS in ~/sandbox/src/springcontroller/analysis/plot_conditions.py
PREFERRED_CONDITION_ORDER = ["KT", "Position", "Pose"]

# Display labels for chart tick labels only -- mirrors DISPLAY_LABELS in
# ~/sandbox/src/springcontroller/analysis/plot_conditions.py. Keys stay
# "KT"/"Position"/"Pose" everywhere else since those match the filename
# tokens parsed by extract_condition().
DISPLAY_LABELS = {
    "KT": "Baseline",
    "Position": "Pos.-only",
    "Pose": "Pose",
}

SESSION_FILE_SUFFIXES = ("_trialonly", "_posture")


def extract_condition(csv_path):
    """'p10_position_side1_trialonly.csv' -> 'position'; 'p3_KT_posture.csv' -> 'KT'."""
    stem = csv_path.stem
    for suffix in SESSION_FILE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = re.sub(r"_side\d+$", "", stem)
    stem = re.sub(r"^p\d+_", "", stem)
    return stem


def extract_participant(csv_path):
    """'p10_position_side1_trialonly.csv' -> 'p10'."""
    match = re.match(r"^(p\d+)_", csv_path.stem)
    if not match:
        raise ValueError(f"{csv_path.name}: filename doesn't start with a p<N>_ participant prefix")
    return match.group(1)


def condition_sort_key(condition):
    for i, preferred in enumerate(PREFERRED_CONDITION_ORDER):
        if preferred.lower() == condition.lower():
            return (0, i)
    return (1, condition.lower())


def canonical_condition_label(condition):
    """Display label for a condition parsed from a filename, e.g. 'position' ->
    'Pos.-only' (see DISPLAY_LABELS). Falls back to the raw string for
    anything not in PREFERRED_CONDITION_ORDER, so an unrecognized condition
    still shows up rather than being silently dropped."""
    for preferred in PREFERRED_CONDITION_ORDER:
        if preferred.lower() == condition.lower():
            return DISPLAY_LABELS.get(preferred, preferred)
    return condition


def classify_session_counts(csv_path, hunch_thresh_deg, lean_thresh_deg, shoulder_percentile):
    """Per-frame zone counts for one session (None in place of a zone means
    unclassified). Returns None if the file had no rows."""
    rows = load_posture_rows(csv_path)
    if not rows:
        return None
    dist_thresh, line_thresh = shoulder_fallback_thresholds(rows, shoulder_percentile)
    counts = Counter()
    for r in rows:
        zone, _ = classify_row(r, hunch_thresh_deg, lean_thresh_deg, dist_thresh, line_thresh)
        counts[zone if zone is not None else UNCLASSIFIED] += 1
    return counts


def plot_condition_comparison(conditions, pct_by_condition, out_path, title=True, font_scale=1.0):
    base_fontsize = plt.rcParams["font.size"] * font_scale
    fig, ax = plt.subplots(figsize=(10, 0.9 * len(conditions) + 1.5))
    y = list(range(len(conditions)))
    left = [0.0] * len(conditions)
    for cat in CATEGORIES:
        vals = [pct_by_condition[c][cat] for c in conditions]
        ax.barh(y, vals, left=left, height=0.6, label=cat, color=CATEGORY_COLORS[cat])
        left = [l + v for l, v in zip(left, vals)]

    ax.set_yticks(y)
    ax.set_yticklabels([canonical_condition_label(c) for c in conditions], fontsize=base_fontsize)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of total frames", fontsize=base_fontsize)
    ax.tick_params(axis="x", labelsize=base_fontsize)
    if title:
        ax.set_title("Posture zone exposure by condition (all sessions pooled)", fontsize=base_fontsize)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=base_fontsize)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing per-session posture CSVs")
    ap.add_argument(
        "--pattern", default="*_trialonly.csv",
        help="glob pattern for input files (default: *_trialonly.csv; pass "
        "'*_posture.csv' for uncropped whole-session files instead)",
    )
    ap.add_argument("--hunch-threshold-deg", type=float, default=20.0)
    ap.add_argument("--lean-threshold-deg", type=float, default=15.0)
    ap.add_argument("--shoulder-percentile", type=float, default=75.0)
    ap.add_argument("--out", default=None, help="output PNG (default: <directory>/condition_comparison.png)")
    ap.add_argument("--out-csv", default=None, help="per-condition summary CSV (default: <directory>/condition_comparison_summary.csv)")
    ap.add_argument("--no-title", action="store_true", help="omit the chart title (e.g. for a paper figure with its own caption)")
    ap.add_argument("--font-scale", type=float, default=1.0, help="multiplier on all chart text size (default: 1.0)")
    args = ap.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern!r} in {directory}")

    out_path = Path(args.out) if args.out else directory / "condition_comparison.png"
    out_csv = Path(args.out_csv) if args.out_csv else directory / "condition_comparison_summary.csv"

    counts_by_condition = {}
    n_sessions_by_condition = Counter()
    skipped = []
    for f in files:
        counts = classify_session_counts(f, args.hunch_threshold_deg, args.lean_threshold_deg, args.shoulder_percentile)
        if counts is None:
            skipped.append(f.name)
            continue
        condition = extract_condition(f)
        counts_by_condition.setdefault(condition, Counter())
        counts_by_condition[condition].update(counts)
        n_sessions_by_condition[condition] += 1

    if not counts_by_condition:
        raise SystemExit(f"none of the {len(files)} matched file(s) had any rows")

    conditions = sorted(counts_by_condition, key=condition_sort_key)
    pct_by_condition = {}
    n_frames_by_condition = {}
    for c in conditions:
        counts = counts_by_condition[c]
        total = sum(counts.values())
        n_frames_by_condition[c] = total
        pct_by_condition[c] = {cat: 100.0 * counts.get(cat, 0) / total for cat in CATEGORIES}

    plot_condition_comparison(conditions, pct_by_condition, out_path, title=not args.no_title, font_scale=args.font_scale)

    fieldnames = ["condition", "n_sessions", "n_frames"] + [f"pct_{cat.replace('+', '_')}" for cat in CATEGORIES]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for c in conditions:
            row = {"condition": c, "n_sessions": n_sessions_by_condition[c], "n_frames": n_frames_by_condition[c]}
            row.update({f"pct_{cat.replace('+', '_')}": pct_by_condition[c][cat] for cat in CATEGORIES})
            w.writerow(row)

    for c in conditions:
        pct = pct_by_condition[c]
        print(
            f"{c}: {n_sessions_by_condition[c]} session(s), {n_frames_by_condition[c]} frames -- "
            f"neutral {pct[ZONE_NEUTRAL]:.0f}%, lean {pct[ZONE_LEAN]:.0f}%, hunch {pct[ZONE_HUNCH]:.0f}%, "
            f"hunch+lean {pct[ZONE_BOTH]:.0f}%, unclassified {pct[UNCLASSIFIED]:.0f}%"
        )
    if skipped:
        print(f"{len(skipped)} file(s) skipped (empty): {', '.join(skipped)}")
    print(f"wrote {out_path}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
