#!/usr/bin/env python3
"""Extract per-trial adjustment duration from corrected event logs
(fix_event_logs.py's event_log_<N>_corrected.csv) into one row per
(Participant, Condition, Side): Adjustment Start, Adjustment End, Total
Adjustment Time (seconds).

Pairs each 'Adjustment start'/'Adjustment end' row by its combined
condition field (e.g. 'KT side 1'), mirroring classify_ergonomics.py's
load_trials(): a new start overwrites any still-unresolved pending start
for that slot (latest start wins), and the first end seen after that
closes it. fix_event_logs.py's own corrections report documents ~32
duplicate Adjustment start/end rows across the 24 logs (kept, not deleted,
during that correction pass) -- this is where that ambiguity actually gets
resolved into a single number, so every duplicate/skip is printed as a
warning rather than resolved silently.

A handful of rows still carry a '[MISSING - fill in from video]' note even
though fix_event_logs.py already filled in a timestamp -- someone (a
person, not this script) typed in an estimate from the video but hasn't
cleared the note yet. Those are used as given; only a genuinely blank
timestamp, or an end that lands at/before its start (seen once so far --
event_log_14, Position side 1, off by -31s -- almost certainly a
transcription slip during that manual fill-in), gets skipped and warned
about instead of silently corrupting Total Adjustment Time.
"""
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

ADJUSTMENT_START = "Adjustment start"
ADJUSTMENT_END = "Adjustment end"

CONDITION_SIDE_RE = re.compile(r"^(.*)\s+side\s+(\d+)$", re.IGNORECASE)
PARTICIPANT_RE = re.compile(r"event_log_(\d+)_corrected\.csv$")


def participant_id(csv_path):
    match = PARTICIPANT_RE.search(csv_path.name)
    if not match:
        raise ValueError(f"{csv_path.name}: doesn't look like an event_log_<N>_corrected.csv")
    return f"p{match.group(1)}"


def split_condition_side(raw_condition):
    match = CONDITION_SIDE_RE.match(raw_condition.strip())
    if not match:
        return None
    return match.group(1).strip(), match.group(2).strip()


def extract_adjustment_times(csv_path):
    """Returns (rows, warnings) for one event log."""
    participant = participant_id(csv_path)
    rows = []
    warnings = []
    pending = {}  # (condition, side) -> start datetime

    with open(csv_path, newline="") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            event = (row.get("event") or "").strip()
            if event not in (ADJUSTMENT_START, ADJUSTMENT_END):
                continue
            raw_condition = (row.get("condition") or "").strip()
            split = split_condition_side(raw_condition)
            if split is None:
                warnings.append(f"{csv_path.name}:{line_no}: condition {raw_condition!r} doesn't match '<condition> side <N>' -- skipped")
                continue
            slot = split
            timestamp = (row.get("timestamp") or "").strip()
            if not timestamp:
                warnings.append(f"{csv_path.name}:{line_no}: {event} for {raw_condition!r} has no timestamp -- skipped")
                if event == ADJUSTMENT_START:
                    pending.pop(slot, None)
                continue
            dt = datetime.fromisoformat(timestamp)

            if event == ADJUSTMENT_START:
                if slot in pending:
                    warnings.append(
                        f"{csv_path.name}:{line_no}: duplicate Adjustment start for {raw_condition!r} -- "
                        f"using {dt.isoformat()}, discarding earlier {pending[slot].isoformat()}"
                    )
                pending[slot] = dt
                continue

            # Adjustment end
            start_dt = pending.pop(slot, None)
            if start_dt is None:
                warnings.append(f"{csv_path.name}:{line_no}: Adjustment end for {raw_condition!r} with no pending start -- skipped")
                continue
            duration = (dt - start_dt).total_seconds()
            if duration < 0:
                warnings.append(
                    f"{csv_path.name}:{line_no}: Adjustment end ({dt.isoformat()}) is before its start "
                    f"({start_dt.isoformat()}) for {raw_condition!r} -- skipped, check the source log"
                )
                continue
            if duration == 0:
                warnings.append(f"{csv_path.name}:{line_no}: Adjustment start == end for {raw_condition!r} ({dt.isoformat()}) -- kept as 0s")
            condition, side = slot
            rows.append({
                "Participant": participant,
                "Condition": condition,
                "Side": side,
                "Adjustment Start": start_dt.isoformat(),
                "Adjustment End": dt.isoformat(),
                "Total Adjustment Time": int(duration),
            })

    for (condition, side), start_dt in pending.items():
        warnings.append(f"{csv_path.name}: Adjustment start for '{condition} side {side}' at {start_dt.isoformat()} has no matching end -- skipped")

    return rows, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory", help="directory containing event_log_<N>_corrected.csv files")
    ap.add_argument("--pattern", default="event_log_*_corrected.csv", help="glob pattern for input files")
    ap.add_argument("--out", default=None, help="output CSV (default: <directory>/adjustment_times.csv)")
    args = ap.parse_args()

    directory = Path(args.directory)
    files = sorted(directory.glob(args.pattern))
    if not files:
        raise SystemExit(f"no files matching {args.pattern!r} in {directory}")

    out_path = Path(args.out) if args.out else directory / "adjustment_times.csv"

    all_rows, all_warnings = [], []
    for f in files:
        rows, warnings = extract_adjustment_times(f)
        all_rows.extend(rows)
        all_warnings.extend(warnings)

    all_rows.sort(key=lambda r: (int(r["Participant"][1:]), r["Condition"], r["Side"]))

    fieldnames = ["Participant", "Condition", "Side", "Adjustment Start", "Adjustment End", "Total Adjustment Time"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"wrote {out_path} ({len(all_rows)} rows from {len(files)} file(s))")
    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s):")
        for w in all_warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
