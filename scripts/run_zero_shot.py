#!/usr/bin/env python3
"""Run closed-set zero-shot classification from an explicit CSV manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from iris_phase1.dataset import read_manifest
from iris_phase1.model import DEFAULT_MODEL_ID, QwenClassifier
from iris_phase1.pipeline import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--dataset-root", type=Path, help="base for relative video paths")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame-counts", nargs="+", type=int, choices=(1, 8), default=[1, 8])
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--limit", type=int, help="process only the first N manifest rows")
    args = parser.parse_args()

    items = read_manifest(args.manifest, args.dataset_root)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        items = items[: args.limit]
    classifier = QwenClassifier(args.model_id)
    count = run_inference(items, args.output, args.frame_counts, classifier)
    print(f"Wrote {count} result rows to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
