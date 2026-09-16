#!/usr/bin/env python3
"""Batch-run label_video.py over every rosbag2 found under a directory tree.

Walks `root` recursively for rosbag2 directories (any directory containing
metadata.yaml -- the same check label_video.py itself uses) and runs
label_video.py against each one found.

Usage:
    source /opt/ros/kilted/setup.bash
    source ~/.posture_venv/bin/activate

    python3 label_videos_batch.py ~/gen3_bags
    python3 label_videos_batch.py ~/gen3_bags --out-dir ~/labeled_videos --every 2
    python3 label_videos_batch.py ~/gen3_bags --skip-existing

Any argument this script doesn't recognize is passed straight through to
label_video.py for every bag (e.g. --topic, --fps, --font-scale). --out is
rejected in batch mode since it names a single output file, which would
collide across bags -- use --out-dir instead, which mirrors each bag's path
relative to `root` so same-named bags in different subfolders don't collide.
"""
import argparse
import subprocess
import sys
from pathlib import Path

LABEL_VIDEO = Path(__file__).parent / "label_video.py"


def find_bags(root):
    return sorted(p.parent for p in Path(root).rglob("metadata.yaml"))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("root", help="directory to search recursively for rosbag2 folders")
    ap.add_argument(
        "--out-dir", default=None,
        help="base directory for outputs, mirroring each bag's path relative to `root` "
        "(default: write each bag's videos alongside it, same as a single label_video.py run)",
    )
    ap.add_argument(
        "--skip-existing", action="store_true",
        help="skip a bag if output video(s) matching '<bag>_camera*_labeled.mp4' already "
        "exist in its output location (best-effort: only recognizes the default topic names)",
    )
    args, passthrough = ap.parse_known_args()

    if "--out" in passthrough:
        raise SystemExit(
            "--out names a single file and doesn't make sense across multiple bags in "
            "batch mode -- use --out-dir instead"
        )

    root = Path(args.root)
    bags = find_bags(root)
    if not bags:
        raise SystemExit(f"no rosbag2 directories (metadata.yaml) found under {root}")

    print(f"found {len(bags)} rosbag2 director{'y' if len(bags) == 1 else 'ies'} under {root}")

    n_ok = n_failed = n_skipped = 0
    for bag in bags:
        cmd = [sys.executable, str(LABEL_VIDEO), str(bag)]
        out_dir = None
        if args.out_dir:
            out_dir = Path(args.out_dir) / bag.relative_to(root).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd += ["--out-dir", str(out_dir)]
        cmd += passthrough

        if args.skip_existing:
            check_dir = out_dir if out_dir else bag.parent
            existing = sorted(p.name for p in check_dir.glob(f"{bag.name}_camera*_labeled.mp4"))
            if existing:
                print(f"skip {bag} (found {existing})")
                n_skipped += 1
                continue

        print(f"--- {bag} ---")
        result = subprocess.run(cmd)
        if result.returncode == 0:
            n_ok += 1
        else:
            n_failed += 1
            print(f"FAILED: {bag} (exit {result.returncode})")

    print(f"\ndone: {n_ok} ok, {n_failed} failed, {n_skipped} skipped, {len(bags)} total")
    if n_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
