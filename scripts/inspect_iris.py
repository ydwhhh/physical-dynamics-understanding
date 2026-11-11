#!/usr/bin/env python3
"""Inspect a dataset tree and optionally create a manifest template."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from iris_phase1.dataset import discover_videos, write_manifest_template


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--write-manifest", type=Path)
    parser.add_argument("--show", type=int, default=20, help="number of relative paths to print")
    args = parser.parse_args()

    root = args.dataset_root.expanduser().resolve()
    videos = discover_videos(root)
    print(f"Dataset root: {root}")
    print(f"Discovered videos: {len(videos)}")
    print(f"Extensions: {dict(sorted(Counter(p.suffix.lower() for p in videos).items()))}")
    for path in videos[: args.show]:
        print(path.relative_to(root))
    if args.write_manifest:
        count = write_manifest_template(root, args.write_manifest)
        print(f"Wrote {count} rows to {args.write_manifest.expanduser().resolve()}")
        print("Fill the label column manually with one of the four canonical labels.")


if __name__ == "__main__":
    main()
