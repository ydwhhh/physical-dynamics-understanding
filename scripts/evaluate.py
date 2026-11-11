#!/usr/bin/env python3
"""Evaluate valid zero-shot predictions in a result CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from iris_phase1.labels import LABELS


def metrics_for(group: pd.DataFrame) -> dict[str, object]:
    has_truth = group["true_label"].isin(LABELS)
    evaluated = group.loc[has_truth]
    valid_prediction = evaluated["predicted_label"].isin(LABELS)
    invalid_count = int((~valid_prediction).sum())
    if evaluated.empty:
        return {
            "total_rows": len(group),
            "evaluated_rows": 0,
            "invalid_rows": invalid_count,
            "accuracy": None,
            "macro_f1": None,
            "per_class_accuracy": {label: None for label in LABELS},
            "confusion_matrix": [[0] * len(LABELS) for _ in LABELS],
        }
    truth = evaluated["true_label"]
    prediction = evaluated["predicted_label"]
    matrix = confusion_matrix(truth, prediction, labels=LABELS)
    per_class = {}
    for index, label in enumerate(LABELS):
        denominator = int((truth == label).sum())
        per_class[label] = float(matrix[index, index] / denominator) if denominator else None
    return {
        "total_rows": len(group),
        "evaluated_rows": len(evaluated),
        "invalid_rows": invalid_count,
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, labels=LABELS, average="macro", zero_division=0)),
        "per_class_accuracy": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    frame = pd.read_csv(args.results, keep_default_na=False)
    required = {"true_label", "predicted_label", "frame_count"}
    missing = required.difference(frame.columns)
    if missing:
        parser.error(f"results CSV is missing columns: {sorted(missing)}")

    report = {
        "class_order": list(LABELS),
        "note": (
            "Rows with a canonical true label are evaluated. Invalid predictions count as wrong "
            "for accuracy, macro-F1, and per-class accuracy; they cannot occupy a closed-set "
            "confusion-matrix column and are reported separately."
        ),
        "by_frame_count": {
            str(count): metrics_for(group)
            for count, group in frame.groupby("frame_count", sort=True)
        },
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output_json:
        destination = args.output_json.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
