"""Manifest inference orchestration and durable CSV output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .dataset import ManifestItem
from .labels import parse_choice
from .model import QwenClassifier
from .prompting import classification_prompt
from .video import sample_video

RESULT_FIELDS = [
    "video_id",
    "video_path",
    "true_label",
    "predicted_choice",
    "predicted_label",
    "valid_prediction",
    "frame_count",
    "timestamps_seconds",
    "prompt",
    "model_id",
    "raw_response",
    "error",
]


def _timestamp_json(values: list[float]) -> str:
    serializable = [round(value, 6) if value == value else None for value in values]
    return json.dumps(serializable, separators=(",", ":"))


def run_inference(
    items: Iterable[ManifestItem],
    output_csv: str | Path,
    frame_counts: Iterable[int],
    classifier: QwenClassifier,
) -> int:
    """Run every item/frame-count combination, recording per-row failures."""
    destination = Path(output_csv).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for item in items:
            for frame_count in frame_counts:
                prompt = classification_prompt(frame_count)
                row = {
                    "video_id": item.video_id,
                    "video_path": str(item.video_path),
                    "true_label": item.true_label,
                    "predicted_choice": "",
                    "predicted_label": "",
                    "valid_prediction": False,
                    "frame_count": frame_count,
                    "timestamps_seconds": "[]",
                    "prompt": prompt,
                    "model_id": classifier.model_id,
                    "raw_response": "",
                    "error": "",
                }
                try:
                    sampled = sample_video(item.video_path, frame_count)
                    row["timestamps_seconds"] = _timestamp_json(sampled.timestamps_seconds)
                    raw = classifier.classify(sampled.frames, prompt)
                    choice, label = parse_choice(raw)
                    row["raw_response"] = raw
                    if choice is None:
                        row["error"] = "invalid model response; expected exactly A/B/C/D"
                    else:
                        row["predicted_choice"] = choice
                        row["predicted_label"] = label
                        row["valid_prediction"] = True
                except Exception as exc:  # Per-video resilience is intentional.
                    row["error"] = f"{type(exc).__name__}: {exc}"
                writer.writerow(row)
                handle.flush()
                rows_written += 1
    return rows_written
