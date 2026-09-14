"""
plot_conditions.py

Reads the tidy scored CSV produced by analyze.py (one row per
participant x condition, columns: participant_id, condition, fluency,
adapts, helpfulness) and produces one boxplot per measure comparing the
three robot control conditions (KT / Position / Pose).

Deliberately separate from analyze.py: re-plotting doesn't require
re-parsing the raw Qualtrics export, and this script's conventions mirror
the plotting scripts in
~/Dropbox/Research/SmartPlayground/VideoDataAnalysis/ (argparse CLI,
matplotlib Agg backend, a fixed colorblind-friendly palette, one output
file per chart).

Usage:
  python plot_conditions.py --input scored_data.csv --output charts/

Options:
  --input     Tidy CSV written by analyze.py (default: scored_data.csv)
  --output    Folder where chart files are saved (created if needed)
  --dpi       Resolution of output images (default: 150)
  --format    Output format: png, pdf, svg (default: png)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CONDITIONS = ["KT", "Position", "Pose"]

# Fixed 3-color colorblind-friendly palette (Okabe-Ito subset), one color
# per condition in the order above.
CONDITION_COLORS = {
    "KT": "#0072B2",
    "Position": "#E69F00",
    "Pose": "#009E73",
}

MEASURE_TITLES = {
    "fluency": "Fluency",
    "adapts": "PSI: Adapts to Human Behaviors",
    "helpfulness": "PSI: Helpfulness",
}


def make_boxplot(df, measure, out_path, dpi=150):
    data = [df.loc[df["condition"] == c, measure].dropna().values for c in CONDITIONS]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(data, labels=CONDITIONS, patch_artist=True, showmeans=True)
    for patch, cond in zip(bp["boxes"], CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[cond])
        patch.set_alpha(0.7)

    for i, values in enumerate(data, start=1):
        jitter = (pd.Series(range(len(values))).sub(len(values) / 2) * 0.02) if len(values) else []
        ax.scatter([i + j for j in jitter], values, color="black", alpha=0.5, s=15, zorder=3)

    ax.set_title(MEASURE_TITLES.get(measure, measure))
    ax.set_ylabel("Score (1-5)")
    ax.set_xlabel("Condition")
    ax.set_ylim(0.5, 5.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="scored_data.csv",
                         help="Tidy CSV written by analyze.py (default: scored_data.csv)")
    parser.add_argument("--output", default="charts",
                         help="Folder to save chart files into (default: charts)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for measure in ["fluency", "adapts", "helpfulness"]:
        out_path = out_dir / f"{measure}_by_condition.{args.format}"
        make_boxplot(df, measure, out_path, dpi=args.dpi)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
