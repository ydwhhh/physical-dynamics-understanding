#!/usr/bin/env python3
"""Run both 1-frame and 8-frame inference for exactly one labeled video."""

from __future__ import annotations

import argparse
from pathlib import Path

from iris_phase1.dataset import ManifestItem
from iris_phase1.labels import normalize_label
from iris_phase1.model import DEFAULT_MODEL_ID, QwenClassifier
from iris_phase1.pipeline import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--video-id")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = parser.parse_args()

    path = args.video.expanduser().resolve()
    item = ManifestItem(args.video_id or path.stem, path, normalize_label(args.label))
    classifier = QwenClassifier(args.model_id)
    count = run_inference([item], args.output, (1, 8), classifier)
    print(f"Smoke test completed; wrote {count} rows to {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()
