"""Dataset discovery and explicit manifest handling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .labels import normalize_label

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpeg", ".mpg", ".m4v"}


@dataclass(frozen=True)
class ManifestItem:
    video_id: str
    video_path: Path
    true_label: str


def discover_videos(root: str | Path) -> list[Path]:
    """Recursively list likely videos without making label assumptions."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"dataset root is not a directory: {base}")
    return sorted(
        (path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: str(path.relative_to(base)),
    )


def write_manifest_template(root: str | Path, output_csv: str | Path) -> int:
    """Write discovered paths with blank labels for manual annotation."""
    base = Path(root).expanduser().resolve()
    videos = discover_videos(base)
    destination = Path(output_csv).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_id", "video_path", "label"])
        writer.writeheader()
        for index, path in enumerate(videos):
            writer.writerow(
                {
                    "video_id": path.stem or f"video_{index:06d}",
                    "video_path": str(path.relative_to(base)),
                    "label": "",
                }
            )
    return len(videos)


def read_manifest(manifest_csv: str | Path, dataset_root: str | Path | None = None) -> list[ManifestItem]:
    """Read and validate a CSV containing video_path and label columns."""
    manifest = Path(manifest_csv).expanduser().resolve()
    base = Path(dataset_root).expanduser().resolve() if dataset_root else manifest.parent
    items: list[ManifestItem] = []
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"video_path", "label"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain columns {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("video_path") or "").strip()
            if not raw_path:
                raise ValueError(f"row {row_number}: video_path is blank")
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = base / path
            path = path.resolve()
            label = normalize_label(row.get("label") or "")
            video_id = (row.get("video_id") or "").strip() or path.stem
            items.append(ManifestItem(video_id, path, label))
    if not items:
        raise ValueError("manifest contains no data rows")
    return items
