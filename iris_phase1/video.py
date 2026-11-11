"""Deterministic OpenCV video decoding and temporal sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class VideoDecodeError(RuntimeError):
    """Raised when a video cannot provide any decoded frames."""


@dataclass(frozen=True)
class SampledVideo:
    frames: list[Image.Image]
    timestamps_seconds: list[float]
    source_frame_indices: list[int]
    decoded_frame_count: int
    fps: float | None


def _scan_video(path: Path) -> tuple[int, float | None]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"OpenCV could not open video: {path}")
    fps_value = float(capture.get(cv2.CAP_PROP_FPS))
    fps = fps_value if np.isfinite(fps_value) and fps_value > 0 else None
    count = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        count += 1
    capture.release()
    if count == 0:
        raise VideoDecodeError(f"no frames could be decoded: {path}")
    return count, fps


def sample_video(path: str | Path, num_frames: int) -> SampledVideo:
    """Uniformly sample an ordered, deterministic number of RGB frames.

    The file is decoded sequentially twice. The first pass obtains the number of
    actually decodable frames instead of trusting container metadata. Target
    For one frame, the temporal center is used. For multiple frames, target
    indices are inclusive linspace endpoints rounded to the nearest integer.
    If a video has fewer frames than requested, indices (and frames) repeat.
    """
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise VideoDecodeError(f"video does not exist: {video_path}")

    decoded_count, fps = _scan_video(video_path)
    if num_frames == 1:
        targets = [(decoded_count - 1) // 2]
    else:
        targets = np.rint(np.linspace(0, decoded_count - 1, num_frames)).astype(int).tolist()
    needed = set(targets)
    decoded: dict[int, Image.Image] = {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoDecodeError(f"OpenCV could not reopen video: {video_path}")
    index = 0
    while index <= targets[-1]:
        ok, bgr = capture.read()
        if not ok:
            break
        if index in needed:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decoded[index] = Image.fromarray(rgb)
        index += 1
    capture.release()

    missing = sorted(needed.difference(decoded))
    if missing:
        raise VideoDecodeError(
            f"video became unreadable during sampling; missing frame indices {missing}: {video_path}"
        )

    frames = [decoded[index].copy() for index in targets]
    if fps is not None:
        timestamps = [index / fps for index in targets]
    else:
        # Frame indices remain available; timestamps cannot be inferred safely.
        timestamps = [float("nan")] * len(targets)
    return SampledVideo(frames, timestamps, targets, decoded_count, fps)
